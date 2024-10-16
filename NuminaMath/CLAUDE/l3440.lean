import Mathlib

namespace NUMINAMATH_CALUDE_ball_probabilities_l3440_344074

/-- Represents a bag of balls with a given number of black and white balls. -/
structure BagOfBalls where
  blackBalls : ℕ
  whiteBalls : ℕ

/-- Calculates the probability of drawing two black balls without replacement. -/
def probabilityTwoBlackBalls (bag : BagOfBalls) : ℚ :=
  let totalBalls := bag.blackBalls + bag.whiteBalls
  (bag.blackBalls.choose 2 : ℚ) / (totalBalls.choose 2)

/-- Calculates the probability of drawing a black ball on the second draw,
    given that a black ball was drawn on the first draw. -/
def probabilitySecondBlackGivenFirstBlack (bag : BagOfBalls) : ℚ :=
  (bag.blackBalls - 1 : ℚ) / (bag.blackBalls + bag.whiteBalls - 1)

theorem ball_probabilities (bag : BagOfBalls) 
  (h1 : bag.blackBalls = 6) (h2 : bag.whiteBalls = 4) : 
  probabilityTwoBlackBalls bag = 1/3 ∧ 
  probabilitySecondBlackGivenFirstBlack bag = 5/9 := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l3440_344074


namespace NUMINAMATH_CALUDE_university_volunteer_selection_l3440_344083

theorem university_volunteer_selection (undergrad : ℕ) (masters : ℕ) (doctoral : ℕ) 
  (selected_doctoral : ℕ) (h1 : undergrad = 4400) (h2 : masters = 400) (h3 : doctoral = 200) 
  (h4 : selected_doctoral = 10) :
  (undergrad + masters + doctoral) * selected_doctoral / doctoral = 250 := by
  sorry

end NUMINAMATH_CALUDE_university_volunteer_selection_l3440_344083


namespace NUMINAMATH_CALUDE_lara_cookies_count_l3440_344005

/-- Calculates the total number of cookies baked by Lara --/
def total_cookies (
  num_trays : ℕ
  ) (
  large_rows_per_tray : ℕ
  ) (
  medium_rows_per_tray : ℕ
  ) (
  small_rows_per_tray : ℕ
  ) (
  large_cookies_per_row : ℕ
  ) (
  medium_cookies_per_row : ℕ
  ) (
  small_cookies_per_row : ℕ
  ) (
  extra_large_cookies : ℕ
  ) : ℕ :=
  (large_rows_per_tray * large_cookies_per_row * num_trays + extra_large_cookies) +
  (medium_rows_per_tray * medium_cookies_per_row * num_trays) +
  (small_rows_per_tray * small_cookies_per_row * num_trays)

theorem lara_cookies_count :
  total_cookies 4 5 4 6 6 7 8 6 = 430 := by
  sorry

end NUMINAMATH_CALUDE_lara_cookies_count_l3440_344005


namespace NUMINAMATH_CALUDE_max_angle_on_perp_bisector_l3440_344007

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the perpendicular bisector of a line segment
def perpBisector (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem max_angle_on_perp_bisector 
  (O A : ℝ × ℝ) (r : ℝ) 
  (h_circle : Circle O r)
  (h_interior : A ∈ interior (Circle O r))
  (h_different : A ≠ O) :
  ∃ P : ℝ × ℝ, P ∈ Circle O r ∧ 
    P ∈ perpBisector O A ∧
    ∀ Q : ℝ × ℝ, Q ∈ Circle O r → angle O P A ≥ angle O Q A :=
sorry

end NUMINAMATH_CALUDE_max_angle_on_perp_bisector_l3440_344007


namespace NUMINAMATH_CALUDE_perimeter_of_figure_C_l3440_344079

/-- Given a large rectangle composed of 20 identical small rectangles,
    prove that the perimeter of figure C is 40 cm given the perimeters of figures A and B. -/
theorem perimeter_of_figure_C (x y : ℝ) : 
  (x > 0) → 
  (y > 0) → 
  (6 * x + 2 * y = 56) →  -- Perimeter of figure A
  (4 * x + 6 * y = 56) →  -- Perimeter of figure B
  (2 * x + 6 * y = 40)    -- Perimeter of figure C
  := by sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_C_l3440_344079


namespace NUMINAMATH_CALUDE_interval_length_theorem_l3440_344008

theorem interval_length_theorem (c d : ℝ) : 
  (∃ (x_min x_max : ℝ), 
    (∀ x : ℝ, c ≤ 3*x + 4 ∧ 3*x + 4 ≤ d ↔ x_min ≤ x ∧ x ≤ x_max) ∧
    x_max - x_min = 15) →
  d - c = 45 := by
sorry

end NUMINAMATH_CALUDE_interval_length_theorem_l3440_344008


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l3440_344023

theorem complex_sum_of_powers (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l3440_344023


namespace NUMINAMATH_CALUDE_softball_opponent_score_l3440_344067

theorem softball_opponent_score :
  let team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let num_games := team_scores.length
  let num_losses := (team_scores.filter (λ x => x % 2 = 1)).length
  let opponent_scores_losses := (team_scores.filter (λ x => x % 2 = 1)).map (λ x => x + 1)
  let opponent_scores_wins := (team_scores.filter (λ x => x % 2 = 0)).map (λ x => x / 2)
  num_games = 10 →
  num_losses = 5 →
  opponent_scores_losses.sum + opponent_scores_wins.sum = 45 :=
by sorry

end NUMINAMATH_CALUDE_softball_opponent_score_l3440_344067


namespace NUMINAMATH_CALUDE_arthur_susan_age_difference_l3440_344031

def susan_age : ℕ := 15
def bob_age : ℕ := 11
def tom_age : ℕ := bob_age - 3
def total_age : ℕ := 51

theorem arthur_susan_age_difference : 
  ∃ (arthur_age : ℕ), arthur_age = total_age - susan_age - bob_age - tom_age ∧ arthur_age - susan_age = 2 :=
sorry

end NUMINAMATH_CALUDE_arthur_susan_age_difference_l3440_344031


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3440_344029

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3440_344029


namespace NUMINAMATH_CALUDE_cos_equality_solution_l3440_344093

theorem cos_equality_solution (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 360) 
  (h3 : Real.cos (m * π / 180) = Real.cos (970 * π / 180)) : 
  m = 110 ∨ m = 250 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solution_l3440_344093


namespace NUMINAMATH_CALUDE_inverse_f_at_4_equals_2_l3440_344017

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem inverse_f_at_4_equals_2 :
  ∃ (f_inv : ℝ → ℝ), (∀ x > 0, f_inv (f x) = x) ∧ (f_inv 4 = 2) := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_4_equals_2_l3440_344017


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3440_344062

open Set

def U : Set Nat := {1,2,3,4,5,6}
def P : Set Nat := {1,3,5}
def Q : Set Nat := {1,2,4}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {1,2,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3440_344062


namespace NUMINAMATH_CALUDE_solve_blueberry_problem_l3440_344028

/-- The number of blueberry bushes Natalie needs to pick -/
def blueberry_problem (containers_per_bush : ℕ) (containers_per_zucchini : ℕ) (containers_to_keep : ℕ) (target_zucchinis : ℕ) : ℕ :=
  (target_zucchinis * containers_per_zucchini + containers_to_keep) / containers_per_bush

/-- Theorem stating the solution to Natalie's blueberry problem -/
theorem solve_blueberry_problem :
  blueberry_problem 10 4 20 60 = 26 := by
  sorry

end NUMINAMATH_CALUDE_solve_blueberry_problem_l3440_344028


namespace NUMINAMATH_CALUDE_alice_sequence_characterization_l3440_344024

/-- Represents the sequence of numbers generated by Alice's operations -/
def AliceSequence (a₀ : ℕ+) : ℕ → ℚ
| 0 => a₀
| (n+1) => if AliceSequence a₀ n > 8763 then 1 / (AliceSequence a₀ n)
           else if AliceSequence a₀ n ≤ 8763 ∧ AliceSequence a₀ (n-1) = 1 / (AliceSequence a₀ (n-2))
                then 2 * (AliceSequence a₀ n) + 1
           else if (AliceSequence a₀ n).den = 1 then 1 / (AliceSequence a₀ n)
           else 2 * (AliceSequence a₀ n) + 1

/-- The set of indices where the sequence value is a natural number -/
def NaturalIndices (a₀ : ℕ+) : Set ℕ :=
  {i | (AliceSequence a₀ i).den = 1}

/-- The theorem stating the characterization of initial values -/
theorem alice_sequence_characterization :
  {a₀ : ℕ+ | Set.Infinite (NaturalIndices a₀)} =
  {a₀ : ℕ+ | a₀ ≤ 17526 ∧ Even a₀} :=
sorry

end NUMINAMATH_CALUDE_alice_sequence_characterization_l3440_344024


namespace NUMINAMATH_CALUDE_current_speed_calculation_l3440_344091

/-- The speed of the current in a river -/
def current_speed : ℝ := 3

/-- The speed at which a man can row in still water (in km/hr) -/
def still_water_speed : ℝ := 15

/-- The time taken to cover 100 meters downstream (in seconds) -/
def downstream_time : ℝ := 20

/-- The distance covered downstream (in meters) -/
def downstream_distance : ℝ := 100

/-- Conversion factor from m/s to km/hr -/
def ms_to_kmhr : ℝ := 3.6

theorem current_speed_calculation :
  current_speed = 
    (downstream_distance / downstream_time * ms_to_kmhr) - still_water_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_calculation_l3440_344091


namespace NUMINAMATH_CALUDE_f_2022_is_zero_l3440_344049

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem f_2022_is_zero (f : ℝ → ℝ) 
  (h1 : is_even_function (fun x ↦ f (2*x + 1)))
  (h2 : is_odd_function (fun x ↦ f (x + 2))) :
  f 2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2022_is_zero_l3440_344049


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l3440_344046

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, 
    if S_10 : S_5 = 1 : 2, then (S_5 + S_10 + S_15) / (S_10 - S_5) = -9/2 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_geom : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) 
    (h_ratio : S 10 / S 5 = 1 / 2) :
    (S 5 + S 10 + S 15) / (S 10 - S 5) = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l3440_344046


namespace NUMINAMATH_CALUDE_solve_for_y_l3440_344043

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 6 = y + 2) (h2 : x = 5) : y = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3440_344043


namespace NUMINAMATH_CALUDE_ellipse_minimum_area_l3440_344009

/-- An ellipse containing two specific circles has a minimum area of (3√3/2)π -/
theorem ellipse_minimum_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  π * a * b ≥ (3 * Real.sqrt 3 / 2) * π := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minimum_area_l3440_344009


namespace NUMINAMATH_CALUDE_china_students_reading_l3440_344056

/-- Represents how a number is read in words -/
def NumberInWords : Type := String

/-- The correct way to read a given number -/
def correctReading (n : Float) : NumberInWords := sorry

/-- The number of primary school students enrolled in China in 2004 (in millions) -/
def chinaStudents2004 : Float := 11246.23

theorem china_students_reading :
  correctReading chinaStudents2004 = "eleven thousand two hundred forty-six point two three" := by
  sorry

end NUMINAMATH_CALUDE_china_students_reading_l3440_344056


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l3440_344080

/-- Prove that for points on an inverse proportion function with k < 0,
    the y-coordinates have a specific ordering. -/
theorem inverse_proportion_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hk : k < 0)
  (h1 : y₁ = k / (-2))
  (h2 : y₂ = k / 1)
  (h3 : y₃ = k / 2) :
  y₂ < y₃ ∧ y₃ < y₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l3440_344080


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l3440_344040

/-- The function f(x) = x^2 + bx + 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- Theorem: -3 is not in the range of f(x) = x^2 + bx + 3 if and only if b ∈ (-2√6, 2√6) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -3) ↔ b ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l3440_344040


namespace NUMINAMATH_CALUDE_candy_problem_l3440_344047

theorem candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_remaining := day1_remaining - (day1_remaining / 4) - 5
  day2_remaining = 10 →
  initial_candies = 84 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l3440_344047


namespace NUMINAMATH_CALUDE_paper_width_problem_l3440_344098

theorem paper_width_problem (sheet1_length sheet1_width sheet2_length : ℝ)
  (h1 : sheet1_length = 11)
  (h2 : sheet1_width = 13)
  (h3 : sheet2_length = 11)
  (h4 : 2 * sheet1_length * sheet1_width = 2 * sheet2_length * sheet2_width + 100) :
  sheet2_width = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_paper_width_problem_l3440_344098


namespace NUMINAMATH_CALUDE_gcd_18_30_l3440_344061

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l3440_344061


namespace NUMINAMATH_CALUDE_investment_ratio_l3440_344026

/-- Represents an investor in a shop -/
structure Investor where
  name : String
  investment : ℕ
  profit_ratio : ℕ

/-- Represents a shop with two investors -/
structure Shop where
  investor1 : Investor
  investor2 : Investor

/-- Theorem stating the relationship between investments and profit ratios -/
theorem investment_ratio (shop : Shop) 
  (h1 : shop.investor1.profit_ratio = 2)
  (h2 : shop.investor2.profit_ratio = 4)
  (h3 : shop.investor2.investment = 1000000) :
  shop.investor1.investment = 500000 := by
  sorry

#check investment_ratio

end NUMINAMATH_CALUDE_investment_ratio_l3440_344026


namespace NUMINAMATH_CALUDE_length_of_cd_l3440_344001

/-- Given a line segment CD with points R and S on it, prove that CD has length 189 -/
theorem length_of_cd (C D R S : ℝ) : 
  (∃ (x y u v : ℝ), 
    C < R ∧ R < S ∧ S < D ∧  -- R and S are on the same side of midpoint
    4 * (R - C) = 3 * (D - R) ∧  -- R divides CD in ratio 3:4
    5 * (S - C) = 4 * (D - S) ∧  -- S divides CD in ratio 4:5
    S - R = 3) →  -- RS = 3
  D - C = 189 := by
sorry

end NUMINAMATH_CALUDE_length_of_cd_l3440_344001


namespace NUMINAMATH_CALUDE_striped_area_equals_circle_area_l3440_344037

theorem striped_area_equals_circle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_diagonal := Real.sqrt (a^2 + b^2)
  let striped_area := π * (a^2 + b^2) / 4
  let circle_area := π * (rectangle_diagonal / 2)^2
  striped_area = circle_area := by
  sorry

end NUMINAMATH_CALUDE_striped_area_equals_circle_area_l3440_344037


namespace NUMINAMATH_CALUDE_smallest_common_multiple_l3440_344068

theorem smallest_common_multiple : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 6 = 0 ∧ m % 5 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 → m ≥ n) ∧
  n = 360 :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_l3440_344068


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3440_344070

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (7 * x₁^2 - (a + 13) * x₁ + a^2 - a - 2 = 0) ∧ 
    (7 * x₂^2 - (a + 13) * x₂ + a^2 - a - 2 = 0) ∧ 
    (0 < x₁) ∧ (x₁ < 1) ∧ (1 < x₂) ∧ (x₂ < 2)) →
  ((-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3440_344070


namespace NUMINAMATH_CALUDE_sin_2theta_problem_l3440_344086

theorem sin_2theta_problem (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.cos (π/2 - θ) = 3/5) :
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_problem_l3440_344086


namespace NUMINAMATH_CALUDE_closest_multiple_of_12_to_1987_is_correct_l3440_344036

def is_multiple_of_12 (n : ℤ) : Prop := n % 12 = 0

def closest_multiple_of_12_to_1987 : ℤ := 1984

theorem closest_multiple_of_12_to_1987_is_correct :
  is_multiple_of_12 closest_multiple_of_12_to_1987 ∧
  ∀ m : ℤ, is_multiple_of_12 m →
    |m - 1987| ≥ |closest_multiple_of_12_to_1987 - 1987| :=
by sorry

end NUMINAMATH_CALUDE_closest_multiple_of_12_to_1987_is_correct_l3440_344036


namespace NUMINAMATH_CALUDE_range_of_p_l3440_344069

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, 29 ≤ p x ∧ p x ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l3440_344069


namespace NUMINAMATH_CALUDE_middle_number_proof_l3440_344087

theorem middle_number_proof (x y z : ℕ) 
  (sum_xy : x + y = 20)
  (sum_xz : x + z = 26)
  (sum_yz : y + z = 30) :
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3440_344087


namespace NUMINAMATH_CALUDE_equation_has_six_roots_l3440_344092

/-- The number of roots of the equation √(14-x²)(sin x - cos 2x) = 0 in the interval [-√14, √14] -/
def num_roots : ℕ := 6

/-- The equation √(14-x²)(sin x - cos 2x) = 0 -/
def equation (x : ℝ) : Prop :=
  Real.sqrt (14 - x^2) * (Real.sin x - Real.cos (2 * x)) = 0

/-- The domain of the equation -/
def domain (x : ℝ) : Prop :=
  x ≥ -Real.sqrt 14 ∧ x ≤ Real.sqrt 14

/-- Theorem stating that the equation has exactly 6 roots in the given domain -/
theorem equation_has_six_roots :
  ∃! (s : Finset ℝ), s.card = num_roots ∧ 
  (∀ x ∈ s, domain x ∧ equation x) ∧
  (∀ x, domain x → equation x → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_equation_has_six_roots_l3440_344092


namespace NUMINAMATH_CALUDE_quadratic_radicals_same_type_l3440_344053

theorem quadratic_radicals_same_type (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ a - 3 = k * (12 - 2*a)) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_same_type_l3440_344053


namespace NUMINAMATH_CALUDE_exponent_simplification_l3440_344014

theorem exponent_simplification (a b : ℝ) (m n : ℤ) 
    (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) (hn : n ≠ 0) :
  (a^m)^(1/n) = a^(m/n) ∧
  (a^(1/n))^(n/m) = a^(1/m) ∧
  (a^n * b)^(1/n) = a * b^(1/n) ∧
  (a^n * b^m)^(1/(m*n)) = a^(1/m) * b^(1/n) ∧
  (a^n / b^m)^(1/(m*n)) = (a^(1/m)) / (b^(1/n)) :=
by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3440_344014


namespace NUMINAMATH_CALUDE_min_value_fraction_l3440_344048

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  (1/a + 1/(b+1)) ≥ (3 + 2*Real.sqrt 2) / 6 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 4 ∧ 1/a₀ + 1/(b₀+1) = (3 + 2*Real.sqrt 2) / 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3440_344048


namespace NUMINAMATH_CALUDE_odd_function_property_l3440_344076

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h1 : IsOdd (fun x ↦ f (x + 1)))
  (h2 : IsOdd (fun x ↦ f (x - 1))) :
  IsOdd (fun x ↦ f (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3440_344076


namespace NUMINAMATH_CALUDE_min_sum_squares_of_roots_l3440_344050

theorem min_sum_squares_of_roots (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + a*x₁ + a + 3 = 0 → 
  x₂^2 + a*x₂ + a + 3 = 0 → 
  x₁ ≠ x₂ →
  ∃ (m : ℝ), ∀ (b : ℝ) (y₁ y₂ : ℝ), 
    y₁^2 + b*y₁ + b + 3 = 0 → 
    y₂^2 + b*y₂ + b + 3 = 0 → 
    y₁ ≠ y₂ →
    y₁^2 + y₂^2 ≥ m ∧ 
    x₁^2 + x₂^2 = m ∧
    m = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_of_roots_l3440_344050


namespace NUMINAMATH_CALUDE_cube_prism_cuboid_rectangular_prism_subset_l3440_344073

-- Define the sets
variable (M : Set (Set ℝ)) -- Set of all right prisms
variable (N : Set (Set ℝ)) -- Set of all cuboids
variable (Q : Set (Set ℝ)) -- Set of all cubes
variable (P : Set (Set ℝ)) -- Set of all right rectangular prisms

-- State the theorem
theorem cube_prism_cuboid_rectangular_prism_subset : Q ⊆ M ∧ M ⊆ N ∧ N ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_cube_prism_cuboid_rectangular_prism_subset_l3440_344073


namespace NUMINAMATH_CALUDE_base_3_8_digit_difference_l3440_344038

/-- The number of digits in the base-b representation of a positive integer n -/
def numDigits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

/-- The theorem stating the difference in the number of digits between base-3 and base-8 representations of 2035 -/
theorem base_3_8_digit_difference :
  numDigits 2035 3 - numDigits 2035 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_3_8_digit_difference_l3440_344038


namespace NUMINAMATH_CALUDE_one_fourth_x_equals_nine_l3440_344094

theorem one_fourth_x_equals_nine (x : ℝ) (h : (1 / 3) * x = 12) : (1 / 4) * x = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_x_equals_nine_l3440_344094


namespace NUMINAMATH_CALUDE_sexagenary_cycle_2016_2017_l3440_344097

/-- Represents the Heavenly Stems in the Sexagenary cycle -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Sexagenary cycle -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- Returns the next Heavenly Stem in the cycle -/
def nextStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Yi
  | HeavenlyStem.Yi => HeavenlyStem.Bing
  | HeavenlyStem.Bing => HeavenlyStem.Ding
  | HeavenlyStem.Ding => HeavenlyStem.Wu
  | HeavenlyStem.Wu => HeavenlyStem.Ji
  | HeavenlyStem.Ji => HeavenlyStem.Geng
  | HeavenlyStem.Geng => HeavenlyStem.Xin
  | HeavenlyStem.Xin => HeavenlyStem.Ren
  | HeavenlyStem.Ren => HeavenlyStem.Gui
  | HeavenlyStem.Gui => HeavenlyStem.Jia

/-- Returns the next Earthly Branch in the cycle -/
def nextBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Chou
  | EarthlyBranch.Chou => EarthlyBranch.Yin
  | EarthlyBranch.Yin => EarthlyBranch.Mao
  | EarthlyBranch.Mao => EarthlyBranch.Chen
  | EarthlyBranch.Chen => EarthlyBranch.Si
  | EarthlyBranch.Si => EarthlyBranch.Wu
  | EarthlyBranch.Wu => EarthlyBranch.Wei
  | EarthlyBranch.Wei => EarthlyBranch.Shen
  | EarthlyBranch.Shen => EarthlyBranch.You
  | EarthlyBranch.You => EarthlyBranch.Xu
  | EarthlyBranch.Xu => EarthlyBranch.Hai
  | EarthlyBranch.Hai => EarthlyBranch.Zi

/-- Returns the next year in the Sexagenary cycle -/
def nextYear (y : SexagenaryYear) : SexagenaryYear :=
  { stem := nextStem y.stem, branch := nextBranch y.branch }

theorem sexagenary_cycle_2016_2017 :
  ∀ (y2016 : SexagenaryYear),
    y2016.stem = HeavenlyStem.Bing ∧ y2016.branch = EarthlyBranch.Shen →
    (nextYear y2016).stem = HeavenlyStem.Ding ∧ (nextYear y2016).branch = EarthlyBranch.You :=
by sorry

end NUMINAMATH_CALUDE_sexagenary_cycle_2016_2017_l3440_344097


namespace NUMINAMATH_CALUDE_johns_money_l3440_344000

theorem johns_money (total : ℕ) (nadas_money : ℕ) : 
  total = nadas_money + (nadas_money - 5) + 4 * nadas_money →
  total = 67 →
  4 * nadas_money = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_money_l3440_344000


namespace NUMINAMATH_CALUDE_system_solution_l3440_344021

theorem system_solution (x y z : ℝ) : 
  x + y + z = 3 ∧ 
  x^2 + y^2 + z^2 = 7 ∧ 
  x^3 + y^3 + z^3 = 15 ↔ 
  (x = 1 ∧ y = 1 + Real.sqrt 2 ∧ z = 1 - Real.sqrt 2) ∨
  (x = 1 ∧ y = 1 - Real.sqrt 2 ∧ z = 1 + Real.sqrt 2) ∨
  (x = 1 + Real.sqrt 2 ∧ y = 1 ∧ z = 1 - Real.sqrt 2) ∨
  (x = 1 + Real.sqrt 2 ∧ y = 1 - Real.sqrt 2 ∧ z = 1) ∨
  (x = 1 - Real.sqrt 2 ∧ y = 1 ∧ z = 1 + Real.sqrt 2) ∨
  (x = 1 - Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3440_344021


namespace NUMINAMATH_CALUDE_fraction_equality_implies_equality_l3440_344058

theorem fraction_equality_implies_equality (x y : ℝ) : x / 2 = y / 2 → x = y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_equality_l3440_344058


namespace NUMINAMATH_CALUDE_distribute_five_prizes_to_three_students_l3440_344034

/-- The number of ways to distribute n different prizes to k students,
    with each student receiving at least one prize -/
def distribute_prizes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different prizes to 3 students,
    with each student receiving at least one prize, is 150 -/
theorem distribute_five_prizes_to_three_students :
  distribute_prizes 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_prizes_to_three_students_l3440_344034


namespace NUMINAMATH_CALUDE_isabel_songs_proof_l3440_344088

/-- The number of songs Isabel bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Isabel bought 72 songs -/
theorem isabel_songs_proof :
  total_songs 6 2 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_songs_proof_l3440_344088


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3440_344065

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : ∃ (winner_votes loser_votes : ℕ), 
    winner_votes + loser_votes = total_votes ∧ 
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 192) :
  total_votes = 480 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3440_344065


namespace NUMINAMATH_CALUDE_eggs_left_proof_l3440_344072

def eggs_left (initial : ℕ) (harry_takes : ℕ) (jenny_takes : ℕ) : ℕ :=
  initial - (harry_takes + jenny_takes)

theorem eggs_left_proof :
  eggs_left 47 5 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_proof_l3440_344072


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l3440_344081

/-- If the cost price of 65 chocolates equals the selling price of 50 chocolates,
    then the gain percent is 30%. -/
theorem chocolate_gain_percent (C S : ℝ) (h : 65 * C = 50 * S) :
  (S - C) / C * 100 = 30 :=
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l3440_344081


namespace NUMINAMATH_CALUDE_supply_duration_l3440_344025

/-- Represents the number of pills in one supply -/
def supply_size : ℕ := 90

/-- Represents the number of days between taking each pill -/
def days_per_pill : ℕ := 3

/-- Represents the approximate number of days in a month -/
def days_per_month : ℕ := 30

/-- Proves that a supply of pills lasts approximately 9 months -/
theorem supply_duration : 
  (supply_size * days_per_pill) / days_per_month = 9 := by
  sorry

end NUMINAMATH_CALUDE_supply_duration_l3440_344025


namespace NUMINAMATH_CALUDE_origami_distribution_l3440_344063

theorem origami_distribution (total_papers : ℕ) (num_cousins : ℕ) (papers_per_cousin : ℕ) : 
  total_papers = 48 → 
  num_cousins = 6 → 
  total_papers = num_cousins * papers_per_cousin → 
  papers_per_cousin = 8 := by
sorry

end NUMINAMATH_CALUDE_origami_distribution_l3440_344063


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3440_344012

/-- The circle C₁ -/
def C₁ (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- M is a point on both C₁ and l -/
def M (m : ℝ) : ℝ × ℝ := sorry

/-- N is a point on both C₁ and l, distinct from M -/
def N (m : ℝ) : ℝ × ℝ := sorry

/-- OM is perpendicular to ON -/
def perpendicular (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

theorem circle_line_intersection (m : ℝ) :
  C₁ (M m).1 (M m).2 m ∧
  C₁ (N m).1 (N m).2 m ∧
  l (M m).1 (M m).2 ∧
  l (N m).1 (N m).2 ∧
  perpendicular (M m) (N m) →
  m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3440_344012


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l3440_344060

theorem four_digit_number_problem (a b c d : Nat) : 
  (a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9) →  -- Ensuring each digit is between 0 and 9
  (a ≠ 0) →  -- Ensuring 'a' is not 0 (as it's a four-digit number)
  (1000 * a + 100 * b + 10 * c + d) - (100 * a + 10 * b + c) - (10 * a + b) - a = 1787 →
  (1000 * a + 100 * b + 10 * c + d = 2009 ∨ 1000 * a + 100 * b + 10 * c + d = 2010) :=
by sorry


end NUMINAMATH_CALUDE_four_digit_number_problem_l3440_344060


namespace NUMINAMATH_CALUDE_shift_linear_function_l3440_344022

-- Define the original linear function
def original_function (x : ℝ) : ℝ := 5 * x - 8

-- Define the shift amount
def shift : ℝ := 4

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function x + shift

-- Theorem statement
theorem shift_linear_function :
  ∀ x : ℝ, shifted_function x = 5 * x - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_shift_linear_function_l3440_344022


namespace NUMINAMATH_CALUDE_function_inequality_l3440_344064

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / Real.exp x

-- State the theorem
theorem function_inequality
  (f : ℝ → ℝ)
  (f_diff : Differentiable ℝ f)
  (h : ∀ x, deriv f x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l3440_344064


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3440_344002

/-- Given a quadratic equation with roots r and s, prove the value of a in the new equation --/
theorem quadratic_roots_transformation (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 →
  s^2 - 5*s + 6 = 0 →
  r + s = 5 →
  r * s = 6 →
  ∃ b, (r^2 + 1)^2 + (-15)*(r^2 + 1) + b = 0 ∧ (s^2 + 1)^2 + (-15)*(s^2 + 1) + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3440_344002


namespace NUMINAMATH_CALUDE_crossing_stretch_distance_l3440_344030

theorem crossing_stretch_distance :
  ∀ (num_people : ℕ) (run_speed bike_speed : ℝ) (total_time : ℝ),
    num_people = 4 →
    run_speed = 10 →
    bike_speed = 50 →
    total_time = 58 / 3 →
    (5 * (116 / 3) / bike_speed = total_time) :=
by
  sorry

end NUMINAMATH_CALUDE_crossing_stretch_distance_l3440_344030


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3440_344041

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 2 / 9) 
  (h2 : material2 = 1 / 8) 
  (h3 : leftover = 4 / 18) : 
  material1 + material2 - leftover = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l3440_344041


namespace NUMINAMATH_CALUDE_dividend_calculation_and_modulo_l3440_344020

theorem dividend_calculation_and_modulo (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 37)
  (h_quotient : quotient = 214)
  (h_remainder : remainder = 12) :
  let dividend := divisor * quotient + remainder
  dividend = 7930 ∧ dividend % 31 = 25 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_and_modulo_l3440_344020


namespace NUMINAMATH_CALUDE_triangle_side_length_l3440_344010

theorem triangle_side_length (a : ℝ) (B C : Real) (h1 : a = 8) (h2 : B = 60) (h3 : C = 75) :
  let A : ℝ := 180 - B - C
  let b : ℝ := a * Real.sin (B * π / 180) / Real.sin (A * π / 180)
  b = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3440_344010


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l3440_344013

/-- The number of large seats on the Ferris wheel -/
def num_large_seats : ℕ := 7

/-- The weight limit for each large seat (in pounds) -/
def weight_limit_per_seat : ℕ := 1500

/-- The average weight of each person (in pounds) -/
def avg_weight_per_person : ℕ := 180

/-- The maximum number of people that can ride on large seats without violating the weight limit -/
def max_people_on_large_seats : ℕ := 
  (num_large_seats * (weight_limit_per_seat / avg_weight_per_person))

theorem ferris_wheel_capacity : max_people_on_large_seats = 56 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l3440_344013


namespace NUMINAMATH_CALUDE_vertical_asymptotes_sum_l3440_344082

theorem vertical_asymptotes_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 3) * (x - 1) * (x - 4)) →
  A + B + C = -1 := by
sorry

end NUMINAMATH_CALUDE_vertical_asymptotes_sum_l3440_344082


namespace NUMINAMATH_CALUDE_original_number_is_seven_l3440_344059

theorem original_number_is_seven : ∃ x : ℝ, 3 * x - 5 = 16 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_seven_l3440_344059


namespace NUMINAMATH_CALUDE_angle_equality_l3440_344099

theorem angle_equality (a b c t : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < t) :
  let S := Real.sqrt (a^2 + b^2 + c^2)
  let ω1 := Real.arctan ((4*t) / (a^2 + b^2 + c^2))
  let ω2 := Real.arccos ((a^2 + b^2 + c^2) / S)
  ω1 = ω2 := by sorry

end NUMINAMATH_CALUDE_angle_equality_l3440_344099


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3440_344045

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3440_344045


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_half_equals_one_l3440_344090

theorem sqrt_two_times_sqrt_half_equals_one :
  Real.sqrt 2 * Real.sqrt (1/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_half_equals_one_l3440_344090


namespace NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l3440_344006

/-- A positive integer has exactly three distinct divisors if and only if it is the square of a prime number. -/
theorem three_divisors_iff_prime_square (n : ℕ) :
  (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ d ∈ s, d ∣ n) ↔ ∃ p : ℕ, Nat.Prime p ∧ n = p^2 :=
sorry

end NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l3440_344006


namespace NUMINAMATH_CALUDE_will_money_left_l3440_344051

/-- The amount of money Will has left after shopping --/
def money_left (initial_amount : ℝ) (sweater_price : ℝ) (tshirt_price : ℝ) (shoes_price : ℝ) 
  (hat_price : ℝ) (socks_price : ℝ) (shoe_refund_rate : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_cost := sweater_price + tshirt_price + shoes_price + hat_price + socks_price
  let refund := shoes_price * shoe_refund_rate
  let new_total := total_cost - refund
  let remaining_items_cost := sweater_price + tshirt_price + hat_price + socks_price
  let discount := remaining_items_cost * discount_rate
  let discounted_total := new_total - discount
  let sales_tax := discounted_total * tax_rate
  let final_cost := discounted_total + sales_tax
  initial_amount - final_cost

/-- Theorem stating that Will has $41.87 left after shopping --/
theorem will_money_left : 
  money_left 74 9 11 30 5 4 0.85 0.1 0.05 = 41.87 := by
  sorry

end NUMINAMATH_CALUDE_will_money_left_l3440_344051


namespace NUMINAMATH_CALUDE_final_produce_theorem_l3440_344089

/-- Represents the quantity of produce -/
structure Produce where
  potatoes : ℕ
  cantaloupes : ℕ
  cucumbers : ℕ

/-- Calculates the final quantity of produce after various events -/
def finalProduce (initial : Produce) : Produce :=
  let potatoesAfterRabbits := initial.potatoes - initial.potatoes / 2
  let cantaloupesAfterSquirrels := initial.cantaloupes - initial.cantaloupes / 4
  let cantaloupesAfterGift := cantaloupesAfterSquirrels + initial.cantaloupes / 2
  let cucumbersAfterRabbits := initial.cucumbers - 2
  let cucumbersAfterHarvest := cucumbersAfterRabbits - (cucumbersAfterRabbits * 3) / 4
  { potatoes := potatoesAfterRabbits,
    cantaloupes := cantaloupesAfterGift,
    cucumbers := cucumbersAfterHarvest }

theorem final_produce_theorem (initial : Produce) :
  initial.potatoes = 7 ∧ initial.cantaloupes = 4 ∧ initial.cucumbers = 5 →
  finalProduce initial = { potatoes := 4, cantaloupes := 5, cucumbers := 1 } :=
by sorry

end NUMINAMATH_CALUDE_final_produce_theorem_l3440_344089


namespace NUMINAMATH_CALUDE_cube_edge_15cm_l3440_344084

/-- The edge length of a cube that displaces a specific volume of water -/
def cube_edge_length (base_length : ℝ) (base_width : ℝ) (water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with the given properties has an edge length of 15 cm -/
theorem cube_edge_15cm :
  cube_edge_length 20 15 11.25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_15cm_l3440_344084


namespace NUMINAMATH_CALUDE_shift_standard_parabola_2_right_l3440_344032

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola :=
  { f := fun x => x^2 }

/-- Theorem: Shifting the standard parabola 2 units right results in y = (x - 2)^2 -/
theorem shift_standard_parabola_2_right :
  (shift_parabola standard_parabola 2).f = fun x => (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_shift_standard_parabola_2_right_l3440_344032


namespace NUMINAMATH_CALUDE_mushroom_ratio_l3440_344011

/-- Represents the types of mushrooms -/
inductive MushroomType
  | Spotted
  | Gilled

/-- Represents a mushroom -/
structure Mushroom where
  type : MushroomType

def total_mushrooms : Nat := 30
def gilled_mushrooms : Nat := 3

theorem mushroom_ratio :
  let spotted_mushrooms := total_mushrooms - gilled_mushrooms
  (gilled_mushrooms : Rat) / spotted_mushrooms = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_ratio_l3440_344011


namespace NUMINAMATH_CALUDE_g_of_5_l3440_344057

def g (x : ℝ) : ℝ := 3*x^4 - 8*x^3 + 15*x^2 - 10*x - 75

theorem g_of_5 : g 5 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l3440_344057


namespace NUMINAMATH_CALUDE_function_properties_l3440_344003

def continuous_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x ∈ s, ∀ ε > 0, ∃ δ > 0, ∀ y ∈ s, |y - x| < δ → |f y - f x| < ε

theorem function_properties (f : ℝ → ℝ) 
    (h_cont : continuous_on f (Set.univ : Set ℝ))
    (h_even : ∀ x : ℝ, f (-x) = f x)
    (h_incr : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0)
    (h_zero : f (-1) = 0) :
  (f 3 < f (-4)) ∧ 
  (∀ x : ℝ, f x / x > 0 → (x > 1 ∨ (-1 < x ∧ x < 0))) ∧
  (∃ M : ℝ, ∀ x : ℝ, f x ≥ M) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3440_344003


namespace NUMINAMATH_CALUDE_sean_initial_apples_l3440_344039

theorem sean_initial_apples (initial : ℕ) (received : ℕ) (total : ℕ) : 
  received = 8 → total = 17 → initial + received = total → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_sean_initial_apples_l3440_344039


namespace NUMINAMATH_CALUDE_sin_theta_value_l3440_344004

theorem sin_theta_value (θ : Real) 
  (h1 : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3440_344004


namespace NUMINAMATH_CALUDE_blackboard_final_number_l3440_344075

/-- Represents the state of the blackboard after each operation --/
structure BoardState where
  numbers : List ℕ
  deriving Repr

/-- Operation A: Subtract a natural number from each number on the board --/
def operationA (state : BoardState) (subtrahend : ℕ) : BoardState :=
  { numbers := state.numbers.map (λ x => x - subtrahend) }

/-- Operation B: Remove two numbers and add their sum --/
def operationB (state : BoardState) : BoardState :=
  match state.numbers with
  | x :: y :: rest => { numbers := (x + y) :: rest }
  | _ => state  -- This case should not occur in our problem

/-- Perform alternating A and B operations --/
def performOperations (initialState : BoardState) (subtrahends : List ℕ) : BoardState :=
  subtrahends.foldl (λ state subtrahend => operationB (operationA state subtrahend)) initialState

/-- The main theorem to prove --/
theorem blackboard_final_number :
  let initialNumbers := List.range 1988
  let initialState : BoardState := { numbers := initialNumbers.map (λ x => x + 1) }
  let subtrahends := List.replicate 1987 1
  (performOperations initialState subtrahends).numbers = [1] := by
  sorry


end NUMINAMATH_CALUDE_blackboard_final_number_l3440_344075


namespace NUMINAMATH_CALUDE_john_average_increase_l3440_344055

def john_scores : List ℝ := [92, 85, 91, 95]

theorem john_average_increase :
  let first_three_avg := (john_scores.take 3).sum / 3
  let all_four_avg := john_scores.sum / 4
  all_four_avg - first_three_avg = 1.42 := by sorry

end NUMINAMATH_CALUDE_john_average_increase_l3440_344055


namespace NUMINAMATH_CALUDE_quadratic_function_value_l3440_344027

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  f a b c 1 = 7 → f a b c 3 = 19 → f a b c 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l3440_344027


namespace NUMINAMATH_CALUDE_problem_solution_l3440_344033

theorem problem_solution (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3440_344033


namespace NUMINAMATH_CALUDE_union_sets_implies_m_equals_three_l3440_344042

theorem union_sets_implies_m_equals_three (m : ℝ) :
  let A : Set ℝ := {2, m}
  let B : Set ℝ := {1, m^2}
  A ∪ B = {1, 2, 3, 9} →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_union_sets_implies_m_equals_three_l3440_344042


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l3440_344018

/-- Given a square with perimeter 60 and a right triangle with one leg 20,
    if their areas are equal, then the other leg of the triangle is 22.5 -/
theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_leg : ℝ) (other_leg : ℝ) :
  square_perimeter = 60 →
  triangle_leg = 20 →
  (square_perimeter / 4) ^ 2 = (triangle_leg * other_leg) / 2 →
  other_leg = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l3440_344018


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3440_344054

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (7 + Real.sqrt 1) / 2
  let r₂ := (7 - Real.sqrt 1) / 2
  r₁ + r₂ = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3440_344054


namespace NUMINAMATH_CALUDE_pet_store_cages_l3440_344071

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 6

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 72

/-- Theorem stating that the number of bird cages is correct -/
theorem pet_store_cages :
  num_cages * (parrots_per_cage + parakeets_per_cage) = total_birds :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3440_344071


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l3440_344015

/-- An inscribed convex octagon with alternating side lengths of 2 and 6√2 -/
structure InscribedOctagon where
  -- The octagon is inscribed in a circle (implied by the problem)
  isInscribed : Bool
  -- The octagon is convex
  isConvex : Bool
  -- The octagon has 8 sides
  numSides : Nat
  -- Four sides have length 2
  shortSideLength : ℝ
  -- Four sides have length 6√2
  longSideLength : ℝ
  -- Conditions
  inscribed_condition : isInscribed = true
  convex_condition : isConvex = true
  sides_condition : numSides = 8
  short_side_condition : shortSideLength = 2
  long_side_condition : longSideLength = 6 * Real.sqrt 2

/-- The area of the inscribed convex octagon -/
def area (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating that the area of the inscribed convex octagon is 124 -/
theorem inscribed_octagon_area (o : InscribedOctagon) : area o = 124 := by sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l3440_344015


namespace NUMINAMATH_CALUDE_last_day_of_month_l3440_344096

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

/-- Theorem: If the 24th day of a 31-day month is a Wednesday, 
    then the last day of the month (31st) is also a Wednesday -/
theorem last_day_of_month (d : DayOfWeek) (h : d = DayOfWeek.Wednesday) :
  advanceDay d 7 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_last_day_of_month_l3440_344096


namespace NUMINAMATH_CALUDE_albert_pizza_consumption_l3440_344095

/-- The number of large pizzas Albert buys -/
def large_pizzas : ℕ := 2

/-- The number of small pizzas Albert buys -/
def small_pizzas : ℕ := 2

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of pizza slices Albert eats in one day -/
def total_slices : ℕ := large_pizzas * large_pizza_slices + small_pizzas * small_pizza_slices

theorem albert_pizza_consumption :
  total_slices = 48 := by
  sorry

end NUMINAMATH_CALUDE_albert_pizza_consumption_l3440_344095


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_11_l3440_344077

theorem smallest_five_digit_mod_11 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 11 = 9 → n ≥ 10000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_11_l3440_344077


namespace NUMINAMATH_CALUDE_max_s_value_l3440_344078

/-- Given two regular polygons P₁ (r-gon) and P₂ (s-gon), where the interior angle of P₁ is 68/67 times
    the interior angle of P₂, this theorem states that the maximum possible value of s is 135. -/
theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) 
  (h_angle : (r - 2) * s * 68 = (s - 2) * r * 67) : s ≤ 135 ∧ ∃ r : ℕ, r ≥ 135 ∧ (135 - 2) * r * 67 = (r - 2) * 135 * 68 := by
  sorry

#check max_s_value

end NUMINAMATH_CALUDE_max_s_value_l3440_344078


namespace NUMINAMATH_CALUDE_sin_eq_cos_necessary_not_sufficient_l3440_344035

open Real

theorem sin_eq_cos_necessary_not_sufficient :
  (∃ α, sin α = cos α ∧ ¬(∃ k : ℤ, α = π / 4 + 2 * k * π)) ∧
  (∀ α, (∃ k : ℤ, α = π / 4 + 2 * k * π) → sin α = cos α) :=
by sorry

end NUMINAMATH_CALUDE_sin_eq_cos_necessary_not_sufficient_l3440_344035


namespace NUMINAMATH_CALUDE_mabel_handled_90_l3440_344044

/-- The number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (t : Transactions) : Prop :=
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.cal = (2 * t.anthony) / 3 ∧
  t.jade = t.cal + 18 ∧
  t.jade = 84

/-- The theorem stating that Mabel handled 90 transactions -/
theorem mabel_handled_90 :
  ∃ (t : Transactions), satisfiesConditions t ∧ t.mabel = 90 := by
  sorry


end NUMINAMATH_CALUDE_mabel_handled_90_l3440_344044


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3440_344019

theorem quadratic_roots_transformation (α β : ℝ) (p : ℝ) : 
  (3 * α^2 + 5 * α + 2 = 0) →
  (3 * β^2 + 5 * β + 2 = 0) →
  ((α^2 + 2) + (β^2 + 2) = -(p : ℝ)) →
  ((α^2 + 2) * (β^2 + 2) = q) →
  p = -49/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3440_344019


namespace NUMINAMATH_CALUDE_system_solution_inequality_solution_set_inequality_solution_transformation_l3440_344052

-- Problem 1
theorem system_solution :
  let system (x y : ℝ) := y = x + 1 ∧ x^2 + 4*y^2 = 4
  ∃ (x₁ y₁ x₂ y₂ : ℝ), system x₁ y₁ ∧ system x₂ y₂ ∧
    ((x₁ = 0 ∧ y₁ = 1) ∨ (x₁ = -8/5 ∧ y₁ = -3/5)) ∧
    ((x₂ = 0 ∧ y₂ = 1) ∨ (x₂ = -8/5 ∧ y₂ = -3/5)) ∧
    x₁ ≠ x₂ := by sorry

-- Problem 2
theorem inequality_solution_set (t : ℝ) :
  let solution_set := {x : ℝ | x^2 - 2*t*x + 1 > 0}
  (t < -1 ∨ t > 1 → ∃ (a b : ℝ), solution_set = {x | x < a ∨ x > b}) ∧
  (-1 < t ∧ t < 1 → solution_set = Set.univ) ∧
  (t = 1 → solution_set = {x | x ≠ 1}) ∧
  (t = -1 → solution_set = {x | x ≠ -1}) := by sorry

-- Problem 3
theorem inequality_solution_transformation (a b c : ℝ) :
  ({x : ℝ | a*x^2 + b*x + c > 0} = Set.Ioo 1 2) →
  {x : ℝ | c*x^2 - b*x + a < 0} = {x : ℝ | x < -1 ∨ x > -1/2} := by sorry

end NUMINAMATH_CALUDE_system_solution_inequality_solution_set_inequality_solution_transformation_l3440_344052


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3440_344066

theorem solve_exponential_equation :
  ∃ n : ℕ, (8 : ℝ)^n * (8 : ℝ)^n * (8 : ℝ)^n * (8 : ℝ)^n = (64 : ℝ)^4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3440_344066


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3440_344085

theorem ceiling_floor_difference : ⌈(16 : ℝ) / 5 * (-34 : ℝ) / 4⌉ - ⌊(16 : ℝ) / 5 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3440_344085


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3440_344016

theorem logarithm_sum_simplification :
  1 / (Real.log 2 / Real.log 7 + 1) +
  1 / (Real.log 3 / Real.log 11 + 1) +
  1 / (Real.log 5 / Real.log 13 + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3440_344016
