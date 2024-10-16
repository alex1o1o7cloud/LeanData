import Mathlib

namespace NUMINAMATH_CALUDE_alex_mean_score_l2550_255006

def scores : List ℝ := [86, 88, 90, 91, 95, 99]

def jane_score_count : ℕ := 2
def alex_score_count : ℕ := 4
def jane_mean_score : ℝ := 93

theorem alex_mean_score : 
  (scores.sum - jane_score_count * jane_mean_score) / alex_score_count = 90.75 := by
  sorry

end NUMINAMATH_CALUDE_alex_mean_score_l2550_255006


namespace NUMINAMATH_CALUDE_min_equal_fruits_cost_l2550_255057

/-- Represents a package of fruits -/
structure Package where
  apples : ℕ
  oranges : ℕ
  cost : ℕ

/-- The two available packages -/
def package1 : Package := ⟨3, 12, 5⟩
def package2 : Package := ⟨20, 5, 13⟩

/-- The minimum nonzero amount to spend for equal apples and oranges -/
def minEqualFruitsCost : ℕ := 64

/-- Theorem stating the minimum cost for equal fruits -/
theorem min_equal_fruits_cost :
  ∀ x y : ℕ,
    x * package1.apples + y * package2.apples = x * package1.oranges + y * package2.oranges →
    x > 0 ∨ y > 0 →
    x * package1.cost + y * package2.cost ≥ minEqualFruitsCost :=
sorry

end NUMINAMATH_CALUDE_min_equal_fruits_cost_l2550_255057


namespace NUMINAMATH_CALUDE_gathering_women_count_l2550_255081

/-- Represents a gathering with men and women dancing --/
structure Gathering where
  num_men : ℕ
  num_women : ℕ
  men_dance_count : ℕ
  women_dance_count : ℕ

/-- Theorem: In a gathering where each man dances with 4 women, each woman dances with 3 men, 
    and there are 15 men, the number of women is 20 --/
theorem gathering_women_count (g : Gathering) 
  (h1 : g.num_men = 15)
  (h2 : g.men_dance_count = 4)
  (h3 : g.women_dance_count = 3)
  : g.num_women = 20 := by
  sorry

end NUMINAMATH_CALUDE_gathering_women_count_l2550_255081


namespace NUMINAMATH_CALUDE_min_visible_sum_is_90_l2550_255092

/-- Represents a cube with integers on each face -/
structure SmallCube where
  faces : Fin 6 → ℕ

/-- Represents the larger 3x3x3 cube -/
structure LargeCube where
  smallCubes : Fin 27 → SmallCube

/-- Calculates the sum of visible faces on the larger cube -/
def visibleSum (c : LargeCube) : ℕ := sorry

/-- The minimum possible sum of visible faces -/
def minVisibleSum : ℕ := 90

/-- Theorem stating that the minimum possible sum is 90 -/
theorem min_visible_sum_is_90 :
  ∀ c : LargeCube, visibleSum c ≥ minVisibleSum :=
sorry

end NUMINAMATH_CALUDE_min_visible_sum_is_90_l2550_255092


namespace NUMINAMATH_CALUDE_problem_solution_l2550_255005

theorem problem_solution (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2550_255005


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2550_255059

-- Problem 1
theorem problem_one : 
  (2 / 3 : ℝ) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3 : ℝ) * Real.sqrt 27 = -(4 / 3 : ℝ) * Real.sqrt 6 := by
  sorry

-- Problem 2
theorem problem_two : 
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2550_255059


namespace NUMINAMATH_CALUDE_floor_sqrt_ten_l2550_255022

theorem floor_sqrt_ten : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_ten_l2550_255022


namespace NUMINAMATH_CALUDE_prove_initial_person_count_l2550_255037

/-- The number of persons initially in a group, given that:
    - The average weight increases by 2.5 kg when a new person replaces someone
    - The replaced person weighs 70 kg
    - The new person weighs 90 kg
-/
def initialPersonCount : ℕ := 8

theorem prove_initial_person_count :
  let averageWeightIncrease : ℚ := 2.5
  let replacedPersonWeight : ℕ := 70
  let newPersonWeight : ℕ := 90
  averageWeightIncrease * initialPersonCount = newPersonWeight - replacedPersonWeight :=
by sorry

#eval initialPersonCount

end NUMINAMATH_CALUDE_prove_initial_person_count_l2550_255037


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l2550_255054

theorem mod_fifteen_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [ZMOD 15] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l2550_255054


namespace NUMINAMATH_CALUDE_johns_umbrella_cost_l2550_255060

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * cost_per_umbrella

/-- Proof that John's total cost for umbrellas is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_umbrella_cost_l2550_255060


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_two_l2550_255040

theorem points_four_units_from_negative_two :
  ∀ x : ℝ, |x - (-2)| = 4 ↔ x = 2 ∨ x = -6 := by sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_two_l2550_255040


namespace NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l2550_255003

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_and_parallel_perpendicular_lines :
  -- P is on both l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧ 
  -- Parallel line equation
  (∀ x y : ℝ, 3*x - 4*y + 8 = 0 ↔ (y - P.2 = (3/4) * (x - P.1))) ∧
  -- Perpendicular line equation
  (∀ x y : ℝ, 4*x + 3*y - 6 = 0 ↔ (y - P.2 = -(4/3) * (x - P.1))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l2550_255003


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2550_255016

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 → x^3 - 3*x^2 - 9*x + 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2550_255016


namespace NUMINAMATH_CALUDE_hilton_marbles_l2550_255023

/-- Calculates the final number of marbles Hilton has --/
def final_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

/-- Proves that Hilton ends up with 42 marbles given the initial conditions --/
theorem hilton_marbles : final_marbles 26 6 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_hilton_marbles_l2550_255023


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2550_255024

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of an arithmetic sequence given specific conditions -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (h1 : seq.a 4 + seq.a 5 = 24)
  (h2 : seq.S 6 = 48) :
  seq.d = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2550_255024


namespace NUMINAMATH_CALUDE_square_of_binomial_identity_l2550_255094

/-- The square of a binomial formula -/
def square_of_binomial (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

/-- Expression A -/
def expr_A (a b : ℝ) : ℝ := (a + b) * (a + b)

/-- Expression B -/
def expr_B (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)

/-- Expression C -/
def expr_C (a : ℝ) : ℝ := (a - 3) * (3 - a)

/-- Expression D -/
def expr_D (a b : ℝ) : ℝ := (2*a - b) * (-2*a + 3*b)

theorem square_of_binomial_identity (a b : ℝ) :
  expr_A a b = square_of_binomial a b ∧
  ∃ x y, expr_B x y ≠ square_of_binomial x y ∧
  ∃ a, expr_C a ≠ square_of_binomial (a - 3) 3 ∧
  ∃ a b, expr_D a b ≠ square_of_binomial (2*a - b) (-2*a + 3*b) :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_identity_l2550_255094


namespace NUMINAMATH_CALUDE_equation_solution_l2550_255007

theorem equation_solution : 
  ∀ x : ℝ, (x - 1)^2 = 64 ↔ x = 9 ∨ x = -7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2550_255007


namespace NUMINAMATH_CALUDE_bacterial_growth_result_l2550_255083

/-- Represents the bacterial population growth model -/
structure BacterialGrowth where
  initial_population : ℕ
  triple_rate : ℕ  -- number of 5-minute intervals where population triples
  double_rate : ℕ  -- number of 10-minute intervals where population doubles

/-- Calculates the final population given a BacterialGrowth model -/
def final_population (model : BacterialGrowth) : ℕ :=
  model.initial_population * (3 ^ model.triple_rate) * (2 ^ model.double_rate)

/-- Theorem stating that under the given conditions, the final population is 16200 -/
theorem bacterial_growth_result :
  let model : BacterialGrowth := {
    initial_population := 50,
    triple_rate := 4,
    double_rate := 2
  }
  final_population model = 16200 := by sorry

end NUMINAMATH_CALUDE_bacterial_growth_result_l2550_255083


namespace NUMINAMATH_CALUDE_two_and_one_third_of_x_is_42_l2550_255052

theorem two_and_one_third_of_x_is_42 : ∃ x : ℚ, (7/3) * x = 42 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_and_one_third_of_x_is_42_l2550_255052


namespace NUMINAMATH_CALUDE_ellipse_foci_l2550_255025

/-- The foci of the ellipse x^2/6 + y^2/9 = 1 are at (0, √3) and (0, -√3) -/
theorem ellipse_foci (x y : ℝ) : 
  (x^2 / 6 + y^2 / 9 = 1) → 
  (∃ (f₁ f₂ : ℝ × ℝ), 
    f₁ = (0, Real.sqrt 3) ∧ 
    f₂ = (0, -Real.sqrt 3) ∧ 
    (∀ (p : ℝ × ℝ), p.1^2 / 6 + p.2^2 / 9 = 1 → 
      (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
       Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 2 * 3))) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_foci_l2550_255025


namespace NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l2550_255026

/-- The axis of symmetry of a parabola y = ax² + bx + c is x = -b / (2a) -/
theorem axis_of_symmetry_parabola (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x, f ((-b / (2 * a)) + x) = f ((-b / (2 * a)) - x)) := by sorry

/-- The axis of symmetry of the parabola y = -3x² + 6x - 1 is the line x = 1 -/
theorem axis_of_symmetry_specific_parabola :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 6 * x - 1
  (∀ x, f (1 + x) = f (1 - x)) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l2550_255026


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2550_255087

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 ≤ 1 → -1 ≤ x ∧ x ≤ 1) ↔
  (∀ x : ℝ, (x < -1 ∨ x > 1) → x^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2550_255087


namespace NUMINAMATH_CALUDE_original_number_proof_l2550_255008

theorem original_number_proof (N : ℝ) (x : ℝ) : 
  (N * 1.2 = 480) → 
  (480 * 0.85 * x^2 = 5*x^3 + 24*x - 50) → 
  N = 400 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2550_255008


namespace NUMINAMATH_CALUDE_exists_valid_partition_l2550_255097

/-- A directed graph where each vertex has outdegree 2 -/
structure Graph (V : Type*) :=
  (edges : V → Finset V)
  (outdegree_two : ∀ v : V, (edges v).card = 2)

/-- A partition of vertices into three sets -/
def Partition (V : Type*) := V → Fin 3

/-- The main theorem statement -/
theorem exists_valid_partition {V : Type*} [Fintype V] (G : Graph V) :
  ∃ (p : Partition V), ∀ (v : V),
    ∃ (w : V), w ∈ G.edges v ∧ p w ≠ p v :=
sorry

end NUMINAMATH_CALUDE_exists_valid_partition_l2550_255097


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2550_255028

theorem fraction_sum_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a + b) / c = a / c + b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2550_255028


namespace NUMINAMATH_CALUDE_length_of_A_l2550_255009

-- Define the points
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 14)
def C : ℝ × ℝ := (3, 6)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the intersection of line segments
def intersect (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (1 - t) • p + t • r = q

-- Main theorem
theorem length_of_A'B' (A' B' : ℝ × ℝ) :
  line_y_eq_x A' →
  line_y_eq_x B' →
  intersect A A' C →
  intersect B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 90 * Real.sqrt 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_length_of_A_l2550_255009


namespace NUMINAMATH_CALUDE_circle_condition_l2550_255017

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0

-- Theorem statement
theorem circle_condition (a : ℝ) :
  (∃ h k r, ∀ x y, circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2550_255017


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_is_52_l2550_255044

theorem large_rectangle_perimeter : ℝ → Prop :=
  fun p =>
    ∀ (square_perimeter small_rect_perimeter : ℝ),
      square_perimeter = 24 →
      small_rect_perimeter = 16 →
      let square_side := square_perimeter / 4
      let small_rect_length := square_side
      let small_rect_width := (small_rect_perimeter / 2) - small_rect_length
      let large_rect_height := square_side + 2 * small_rect_width
      let large_rect_width := 3 * square_side
      2 * (large_rect_height + large_rect_width) = p →
      p = 52

theorem large_rectangle_perimeter_is_52 : large_rectangle_perimeter 52 := by
  sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_is_52_l2550_255044


namespace NUMINAMATH_CALUDE_angle_sum_in_special_figure_l2550_255034

theorem angle_sum_in_special_figure (A B C x y : ℝ) : 
  A = 34 → B = 80 → C = 30 →
  (A + B + (360 - x) + 90 + (120 - y) = 720) →
  x + y = 36 := by sorry

end NUMINAMATH_CALUDE_angle_sum_in_special_figure_l2550_255034


namespace NUMINAMATH_CALUDE_pastor_prayer_ratio_l2550_255088

/-- Represents the number of prayers for a pastor on a given day --/
structure DailyPrayers where
  weekday : ℕ
  sunday : ℕ

/-- Represents the total prayers for a pastor in a week --/
def WeeklyPrayers (d : DailyPrayers) : ℕ := 6 * d.weekday + d.sunday

/-- Pastor Paul's prayer schedule --/
def paul : DailyPrayers :=
  { weekday := 20
    sunday := 40 }

/-- Pastor Bruce's prayer schedule --/
def bruce : DailyPrayers :=
  { weekday := paul.weekday / 2
    sunday := WeeklyPrayers paul - WeeklyPrayers { weekday := paul.weekday / 2, sunday := 0 } - 20 }

theorem pastor_prayer_ratio :
  bruce.sunday / paul.sunday = 2 := by sorry

end NUMINAMATH_CALUDE_pastor_prayer_ratio_l2550_255088


namespace NUMINAMATH_CALUDE_binomial_200_200_l2550_255029

theorem binomial_200_200 : Nat.choose 200 200 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_200_200_l2550_255029


namespace NUMINAMATH_CALUDE_cookies_for_thanksgiving_l2550_255096

/-- The number of cookies Helen baked three days ago -/
def cookies_day1 : ℕ := 31

/-- The number of cookies Helen baked two days ago -/
def cookies_day2 : ℕ := 270

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day3 : ℕ := 419

/-- The number of cookies Beaky ate from the first day's batch -/
def cookies_eaten_by_beaky : ℕ := 5

/-- The percentage of cookies that crumbled from the second day's batch -/
def crumble_percentage : ℚ := 15 / 100

/-- The number of cookies Helen gave away from the third day's batch -/
def cookies_given_away : ℕ := 30

/-- The number of cookies Helen received as a gift from Lucy -/
def cookies_gifted : ℕ := 45

/-- The total number of cookies available at Helen's house for Thanksgiving -/
def total_cookies : ℕ := 690

theorem cookies_for_thanksgiving :
  (cookies_day1 - cookies_eaten_by_beaky) +
  (cookies_day2 - Int.floor (crumble_percentage * cookies_day2)) +
  (cookies_day3 - cookies_given_away) +
  cookies_gifted = total_cookies := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_thanksgiving_l2550_255096


namespace NUMINAMATH_CALUDE_algebraic_identity_l2550_255043

theorem algebraic_identity (a b : ℝ) : a * b - 2 * (a * b) = -(a * b) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2550_255043


namespace NUMINAMATH_CALUDE_wand_price_l2550_255012

theorem wand_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 4 → 
  purchase_price = (1/8) * original_price → 
  original_price = 32 := by
sorry

end NUMINAMATH_CALUDE_wand_price_l2550_255012


namespace NUMINAMATH_CALUDE_number_of_observations_l2550_255014

theorem number_of_observations (initial_mean old_value new_value new_mean : ℝ) : 
  initial_mean = 36 →
  old_value = 40 →
  new_value = 25 →
  new_mean = 34.9 →
  ∃ (n : ℕ), (n : ℝ) * initial_mean - old_value + new_value = (n : ℝ) * new_mean ∧ 
              n = 14 :=
by sorry

end NUMINAMATH_CALUDE_number_of_observations_l2550_255014


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2550_255086

theorem quadratic_roots_sum_of_squares (p q : ℝ) (r s : ℝ) : 
  (2 * r^2 - p * r + q = 0) → 
  (2 * s^2 - p * s + q = 0) → 
  (r^2 + s^2 = p^2 / 4 - q) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2550_255086


namespace NUMINAMATH_CALUDE_blue_red_face_ratio_l2550_255073

theorem blue_red_face_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2) / (6 * n^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_red_face_ratio_l2550_255073


namespace NUMINAMATH_CALUDE_carries_profit_l2550_255046

/-- Calculates the profit for a cake maker after taxes and expenses -/
def cake_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℚ) 
                (supply_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_hours := hours_per_day * days_worked
  let gross_earnings := hourly_rate * total_hours
  let tax_amount := gross_earnings * tax_rate
  let after_tax_earnings := gross_earnings - tax_amount
  after_tax_earnings - supply_cost

/-- Theorem stating that Carrie's profit is $631.20 given the problem conditions -/
theorem carries_profit :
  cake_profit 4 6 35 150 (7/100) = 631.2 := by
  sorry

end NUMINAMATH_CALUDE_carries_profit_l2550_255046


namespace NUMINAMATH_CALUDE_vector_operation_l2550_255027

/-- Given plane vectors a and b, prove that -2a - b equals (-3, -1) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (-2 : ℝ) • a - b = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2550_255027


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l2550_255002

theorem piggy_bank_problem (total_money : ℕ) (total_bills : ℕ) 
  (h1 : total_money = 66) 
  (h2 : total_bills = 49) : 
  ∃ (one_dollar_bills two_dollar_bills : ℕ), 
    one_dollar_bills + two_dollar_bills = total_bills ∧ 
    one_dollar_bills + 2 * two_dollar_bills = total_money ∧
    one_dollar_bills = 32 :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l2550_255002


namespace NUMINAMATH_CALUDE_drums_filled_per_day_l2550_255051

/-- The number of drums filled per day given the total number of drums and days -/
def drums_per_day (total_drums : ℕ) (total_days : ℕ) : ℕ :=
  total_drums / total_days

/-- Theorem stating that 90 drums filled in 6 days results in 15 drums per day -/
theorem drums_filled_per_day :
  drums_per_day 90 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_drums_filled_per_day_l2550_255051


namespace NUMINAMATH_CALUDE_find_unknown_numbers_l2550_255019

/-- Given four real numbers A, B, C, and D satisfying certain conditions,
    prove that they have specific values. -/
theorem find_unknown_numbers (A B C D : ℝ) 
    (h1 : 0.05 * A = 0.20 * 650 + 0.10 * B)
    (h2 : A + B = 4000)
    (h3 : C = 2 * B)
    (h4 : A + B + C = 0.40 * D) :
    A = 3533.3333333333335 ∧ 
    B = 466.6666666666667 ∧ 
    C = 933.3333333333334 ∧ 
    D = 12333.333333333334 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_numbers_l2550_255019


namespace NUMINAMATH_CALUDE_balance_theorem_l2550_255062

/-- Represents the balance between shapes -/
structure Balance where
  triangle : ℚ
  diamond : ℚ
  circle : ℚ

/-- First balance equation: 5 triangles + 2 diamonds = 12 circles -/
def balance1 : Balance := { triangle := 5, diamond := 2, circle := 12 }

/-- Second balance equation: 1 triangle = 1 diamond + 3 circles -/
def balance2 : Balance := { triangle := 1, diamond := 1, circle := 3 }

/-- The balance we want to prove: 4 diamonds = 12/7 circles -/
def target_balance : Balance := { triangle := 0, diamond := 4, circle := 12/7 }

/-- Checks if two balances are equivalent -/
def is_equivalent (b1 b2 : Balance) : Prop :=
  b1.triangle / b2.triangle = b1.diamond / b2.diamond ∧
  b1.triangle / b2.triangle = b1.circle / b2.circle

/-- The main theorem to prove -/
theorem balance_theorem (b1 b2 : Balance) (h1 : is_equivalent b1 balance1) 
    (h2 : is_equivalent b2 balance2) : 
  is_equivalent target_balance { triangle := 0, diamond := 1, circle := 3/7 } := by
  sorry

end NUMINAMATH_CALUDE_balance_theorem_l2550_255062


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2550_255031

def is_divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

def digit_sum (a : ℕ) : ℕ :=
  3 + a + a + 1

theorem four_digit_divisible_by_9 :
  ∃ A : ℕ, A < 10 ∧ is_divisible_by_9 (3000 + 100 * A + 10 * A + 1) ∧ A = 7 :=
sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2550_255031


namespace NUMINAMATH_CALUDE_triangle_side_length_l2550_255036

theorem triangle_side_length (a b : ℝ) (C : ℝ) (S : ℝ) :
  a = 1 →
  C = π / 4 →
  S = 2 * a →
  S = 1 / 2 * a * b * Real.sin C →
  b = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2550_255036


namespace NUMINAMATH_CALUDE_correct_guessing_probability_l2550_255004

/-- Represents a 6-digit password where each digit is a number from 0 to 9 -/
def Password := Fin 10 → Fin 6

/-- The set of even digits -/
def EvenDigits : Set (Fin 10) := {0, 2, 4, 6, 8}

/-- The probability of guessing the correct last digit in no more than 2 attempts -/
def guessing_probability : ℚ :=
  let total_outcomes := Nat.choose 5 2  -- number of ways to choose 2 digits from 5 even digits
  let successful_outcomes := Nat.choose 4 1 * Nat.choose 2 2  -- number of ways to choose the correct digit in 2 attempts
  successful_outcomes / total_outcomes

/-- Theorem stating the probability of guessing the correct last digit -/
theorem correct_guessing_probability :
  guessing_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_correct_guessing_probability_l2550_255004


namespace NUMINAMATH_CALUDE_james_oranges_l2550_255035

theorem james_oranges (pieces_per_orange : ℕ) (num_people : ℕ) (calories_per_orange : ℕ) (calories_per_person : ℕ) :
  pieces_per_orange = 8 →
  num_people = 4 →
  calories_per_orange = 80 →
  calories_per_person = 100 →
  (calories_per_person * num_people) / calories_per_orange * pieces_per_orange / pieces_per_orange = 5 :=
by sorry

end NUMINAMATH_CALUDE_james_oranges_l2550_255035


namespace NUMINAMATH_CALUDE_relation_between_x_and_y_l2550_255080

theorem relation_between_x_and_y (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_relation_between_x_and_y_l2550_255080


namespace NUMINAMATH_CALUDE_jade_transactions_l2550_255075

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel * 10 / 100) →
  cal = anthony * 2 / 3 →
  jade = cal + 15 →
  jade = 81 := by
sorry

end NUMINAMATH_CALUDE_jade_transactions_l2550_255075


namespace NUMINAMATH_CALUDE_books_not_sold_l2550_255091

theorem books_not_sold (initial_stock : ℕ) (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) :
  initial_stock = 800 →
  monday_sales = 60 →
  tuesday_sales = 10 →
  wednesday_sales = 20 →
  thursday_sales = 44 →
  friday_sales = 66 →
  initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales) = 600 := by
  sorry

end NUMINAMATH_CALUDE_books_not_sold_l2550_255091


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2550_255093

theorem a_minus_b_value (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) (hab : a + b > 0) :
  a - b = 2 ∨ a - b = 8 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2550_255093


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2550_255077

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ 
  (∀ (x y z : ℝ), x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧
  (∀ (C' : ℝ), C' > C → ∃ (x y z : ℝ), x^2 + y^2 + z^3 + 1 < C' * (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2550_255077


namespace NUMINAMATH_CALUDE_largest_equal_digit_sums_l2550_255069

/-- Calculates the sum of digits of a natural number in a given base. -/
def digit_sum (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a number has equal digit sums in base 10 and base 3. -/
def equal_digit_sums (n : ℕ) : Prop :=
  digit_sum n 10 = digit_sum n 3

theorem largest_equal_digit_sums :
  ∀ m : ℕ, m < 1000 → m > 310 → ¬(equal_digit_sums m) ∧ equal_digit_sums 310 := by sorry

end NUMINAMATH_CALUDE_largest_equal_digit_sums_l2550_255069


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2550_255067

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2550_255067


namespace NUMINAMATH_CALUDE_ad_greater_than_bc_l2550_255030

theorem ad_greater_than_bc (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (sum_eq : a + d = b + c)
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
sorry

end NUMINAMATH_CALUDE_ad_greater_than_bc_l2550_255030


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2550_255048

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3*b + 5*c + 7*d = 14) :
  a^2 + b^2 + c^2 + d^2 ≥ 7/3 ∧
  (a^2 + b^2 + c^2 + d^2 = 7/3 ↔ a = 1/6 ∧ b = 1/2 ∧ c = 5/6 ∧ d = 7/6) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2550_255048


namespace NUMINAMATH_CALUDE_smallest_positive_e_l2550_255071

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- Checks if a given rational number is a root of the polynomial -/
def isRoot (p : Polynomial4) (r : ℚ) : Prop :=
  p.a * r^4 + p.b * r^3 + p.c * r^2 + p.d * r + p.e = 0

/-- The main theorem to be proved -/
theorem smallest_positive_e (p : Polynomial4) : 
  isRoot p (-3) → 
  isRoot p 4 → 
  isRoot p 10 → 
  isRoot p (-1/4) → 
  (∃ (q : Polynomial4), 
    isRoot q (-3) ∧ 
    isRoot q 4 ∧ 
    isRoot q 10 ∧ 
    isRoot q (-1/4) ∧ 
    q.e > 0 ∧ 
    q.e ≤ p.e) → 
  p.e = 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_e_l2550_255071


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2550_255032

/-- The minimum distance between curves C₁ and C₂ is 0 -/
theorem min_distance_between_curves (x y : ℝ) : 
  let C₁ := {(x, y) | x^2/8 + y^2/4 = 1}
  let C₂ := {(x, y) | x - Real.sqrt 2 * y - 4 = 0}
  ∃ (p q : ℝ × ℝ), p ∈ C₁ ∧ q ∈ C₂ ∧ 
    ∀ (p' q' : ℝ × ℝ), p' ∈ C₁ → q' ∈ C₂ → 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ≥ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2550_255032


namespace NUMINAMATH_CALUDE_cubic_identity_for_fifty_l2550_255090

theorem cubic_identity_for_fifty : 50^3 + 3*(50^2) + 3*50 + 1 = 261051 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_for_fifty_l2550_255090


namespace NUMINAMATH_CALUDE_exists_specific_polyhedron_l2550_255011

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A polyhedron -/
structure Polyhedron where
  faces : List Face

/-- Counts the number of faces with a given number of sides -/
def countFaces (p : Polyhedron) (n : ℕ) : ℕ :=
  p.faces.filter (λ f => f.sides = n) |>.length

/-- Theorem: There exists a polyhedron with exactly 6 faces,
    where 2 faces are triangles, 2 faces are quadrilaterals, and 2 faces are pentagons -/
theorem exists_specific_polyhedron :
  ∃ p : Polyhedron,
    p.faces.length = 6 ∧
    countFaces p 3 = 2 ∧
    countFaces p 4 = 2 ∧
    countFaces p 5 = 2 :=
  sorry

end NUMINAMATH_CALUDE_exists_specific_polyhedron_l2550_255011


namespace NUMINAMATH_CALUDE_picture_area_l2550_255018

theorem picture_area (x y : ℕ) (h1 : x > 1) (h2 : y > 1) (h3 : (3 * x + 3) * (y + 2) = 110) : x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l2550_255018


namespace NUMINAMATH_CALUDE_solution_pairs_l2550_255055

/-- The type of pairs of positive integers satisfying the divisibility condition -/
def SolutionPairs : Type := 
  {p : Nat × Nat // p.1 > 0 ∧ p.2 > 0 ∧ (2^(2^p.1) + 1) * (2^(2^p.2) + 1) % (p.1 * p.2) = 0}

/-- The theorem stating the solution pairs -/
theorem solution_pairs : 
  {p : SolutionPairs | p.val = (1, 1) ∨ p.val = (1, 5) ∨ p.val = (5, 1)} = 
  {p : SolutionPairs | true} := by sorry

end NUMINAMATH_CALUDE_solution_pairs_l2550_255055


namespace NUMINAMATH_CALUDE_brand_y_pen_price_l2550_255064

theorem brand_y_pen_price 
  (price_x : ℝ) 
  (total_pens : ℕ) 
  (total_cost : ℝ) 
  (num_x_pens : ℕ) 
  (h1 : price_x = 4)
  (h2 : total_pens = 12)
  (h3 : total_cost = 42)
  (h4 : num_x_pens = 6) :
  (total_cost - price_x * num_x_pens) / (total_pens - num_x_pens) = 3 := by
  sorry

end NUMINAMATH_CALUDE_brand_y_pen_price_l2550_255064


namespace NUMINAMATH_CALUDE_total_weight_theorem_l2550_255072

/-- The total weight of three balls -/
def total_weight (blue_weight brown_weight green_weight : ℝ) : ℝ :=
  blue_weight + brown_weight + green_weight

/-- Theorem: The total weight of the three balls is 9.12 + x -/
theorem total_weight_theorem (x : ℝ) :
  total_weight 6 3.12 x = 9.12 + x := by
  sorry

end NUMINAMATH_CALUDE_total_weight_theorem_l2550_255072


namespace NUMINAMATH_CALUDE_toothpick_20th_stage_l2550_255095

def toothpick_sequence (n : ℕ) : ℕ := 5 + 3 * (n - 1)

theorem toothpick_20th_stage :
  toothpick_sequence 20 = 62 := by
sorry

end NUMINAMATH_CALUDE_toothpick_20th_stage_l2550_255095


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2550_255074

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2550_255074


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2550_255053

theorem existence_of_special_integers : 
  ∃ (a b : ℕ+), 
    (¬ (7 ∣ a.val)) ∧ 
    (¬ (7 ∣ b.val)) ∧ 
    (¬ (7 ∣ (a.val + b.val))) ∧ 
    (7^7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2550_255053


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2550_255033

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 3)
  (h_d : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2550_255033


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2550_255099

theorem reciprocal_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2550_255099


namespace NUMINAMATH_CALUDE_complex_product_proof_l2550_255001

theorem complex_product_proof : Complex.I * Complex.I = -1 → (1 - Complex.I) * (1 + 2 * Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_proof_l2550_255001


namespace NUMINAMATH_CALUDE_problem_solution_l2550_255068

theorem problem_solution :
  ∀ a b c d : ℝ,
  (100 * a = 35^2 - 15^2) →
  ((a - 1)^2 = 3^(4 * b)) →
  (b^2 + c * b - 5 = 0) →
  (∃ k : ℝ, 2 * (x^2) + 3 * x + 4 * d = (x + c) * k) →
  (a = 10 ∧ b = 1 ∧ c = 4 ∧ d = -5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2550_255068


namespace NUMINAMATH_CALUDE_tangent_line_point_on_circle_l2550_255042

/-- Given a circle C defined by x^2 + y^2 = 1 and a line L defined by ax + by = 1 
    that is tangent to C, prove that the point (a, b) lies on C. -/
theorem tangent_line_point_on_circle (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (a*x + b*y = 1 → x^2 + y^2 > 1 ∨ a*x + b*y > 1)) → 
  a^2 + b^2 = 1 := by
  sorry

#check tangent_line_point_on_circle

end NUMINAMATH_CALUDE_tangent_line_point_on_circle_l2550_255042


namespace NUMINAMATH_CALUDE_calculate_expression_l2550_255015

theorem calculate_expression : 5 + 12 / 3 - 2^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2550_255015


namespace NUMINAMATH_CALUDE_diamond_three_four_l2550_255021

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l2550_255021


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2550_255056

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 3 ∧ b = 5) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5) →
  (a^2 + b^2 = c^2) →
  c = 4 ∨ c = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2550_255056


namespace NUMINAMATH_CALUDE_inequality_proof_l2550_255013

theorem inequality_proof (x a : ℝ) (h : |x - a| < 1) : 
  let f := fun (t : ℝ) => t^2 - 2*t
  |f x - f a| < 2*|a| + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2550_255013


namespace NUMINAMATH_CALUDE_greatest_square_power_of_three_under_200_l2550_255084

theorem greatest_square_power_of_three_under_200 : ∃ n : ℕ, 
  n < 200 ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  (∃ k : ℕ, n = 3^k) ∧
  (∀ x : ℕ, x < 200 → (∃ y : ℕ, x = y^2) → (∃ z : ℕ, x = 3^z) → x ≤ n) ∧
  n = 81 :=
by sorry

end NUMINAMATH_CALUDE_greatest_square_power_of_three_under_200_l2550_255084


namespace NUMINAMATH_CALUDE_potato_distribution_l2550_255089

theorem potato_distribution (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_potato_distribution_l2550_255089


namespace NUMINAMATH_CALUDE_middle_school_students_l2550_255039

theorem middle_school_students (band_percentage : ℝ) (band_students : ℕ) 
  (h1 : band_percentage = 0.20)
  (h2 : band_students = 168) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_middle_school_students_l2550_255039


namespace NUMINAMATH_CALUDE_power_product_rule_l2550_255066

theorem power_product_rule (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l2550_255066


namespace NUMINAMATH_CALUDE_upper_line_formula_l2550_255078

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The sequence of numbers in the lower line -/
def x : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => x (n + 1) + x n + 1

/-- The sequence of numbers in the upper line -/
def a (n : ℕ) : ℕ := x (n + 1) - 1

theorem upper_line_formula (n : ℕ) : a n = fib (n + 3) - 2 := by
  sorry

#check upper_line_formula

end NUMINAMATH_CALUDE_upper_line_formula_l2550_255078


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2550_255041

def type_a_cost : ℚ := 9
def type_a_quantity : ℕ := 4
def type_b_extra_cost : ℚ := 5
def type_b_quantity : ℕ := 2
def clay_pot_extra_cost : ℚ := 20
def soil_cost_reduction : ℚ := 2
def fertilizer_percentage : ℚ := 1.5
def gardening_tools_percentage : ℚ := 0.75

def total_cost : ℚ :=
  type_a_cost * type_a_quantity +
  (type_a_cost + type_b_extra_cost) * type_b_quantity +
  (type_a_cost + clay_pot_extra_cost) +
  (type_a_cost - soil_cost_reduction) +
  (type_a_cost * fertilizer_percentage) +
  ((type_a_cost + clay_pot_extra_cost) * gardening_tools_percentage)

theorem total_cost_is_correct : total_cost = 135.25 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2550_255041


namespace NUMINAMATH_CALUDE_divisibility_by_five_units_digit_zero_l2550_255045

theorem divisibility_by_five (n : ℕ) (h_odd : Odd n) : 
  (5 : ℤ) ∣ (4^n + 6^n) := by
  sorry

-- The main theorem
theorem units_digit_zero : 
  (4^1993 + 6^1993) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_units_digit_zero_l2550_255045


namespace NUMINAMATH_CALUDE_odd_minus_odd_is_even_l2550_255098

theorem odd_minus_odd_is_even (a b : ℤ) (ha : Odd a) (hb : Odd b) : Even (a - b) := by
  sorry

end NUMINAMATH_CALUDE_odd_minus_odd_is_even_l2550_255098


namespace NUMINAMATH_CALUDE_triangle_formation_l2550_255082

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 6 ∧
  ¬can_form_triangle 2 2 5 ∧
  ¬can_form_triangle 1 (Real.sqrt 3) 3 ∧
  ¬can_form_triangle 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l2550_255082


namespace NUMINAMATH_CALUDE_cube_root_two_not_rational_plus_sqrt_l2550_255049

theorem cube_root_two_not_rational_plus_sqrt (a b c : ℚ) (hc : c > 0) :
  (a + b * Real.sqrt c) ^ 3 ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_cube_root_two_not_rational_plus_sqrt_l2550_255049


namespace NUMINAMATH_CALUDE_john_zoo_animals_l2550_255065

def zoo_animals (snakes : ℕ) : ℕ :=
  let monkeys := 2 * snakes
  let lions := monkeys - 5
  let pandas := lions + 8
  let dogs := pandas / 3
  snakes + monkeys + lions + pandas + dogs

theorem john_zoo_animals :
  zoo_animals 15 = 114 := by sorry

end NUMINAMATH_CALUDE_john_zoo_animals_l2550_255065


namespace NUMINAMATH_CALUDE_integral_of_root_and_polynomial_l2550_255058

open Real

theorem integral_of_root_and_polynomial (x : ℝ) :
  let f := λ x : ℝ => x^(1/2) * (3 + 2*x^(3/4))^(1/2)
  let F := λ x : ℝ => (2/15) * (3 + 2*x^(3/4))^(5/2) - (2/3) * (3 + 2*x^(3/4))^(3/2)
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_integral_of_root_and_polynomial_l2550_255058


namespace NUMINAMATH_CALUDE_abs_equality_l2550_255020

theorem abs_equality (x : ℝ) : 
  (|x| = Real.sqrt (x^2)) ∧ 
  (|x| = if x ≥ 0 then x else -x) := by sorry

end NUMINAMATH_CALUDE_abs_equality_l2550_255020


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l2550_255047

def A : ℕ := Nat.gcd 9 (Nat.gcd 12 18)
def B : ℕ := Nat.lcm 9 (Nat.lcm 12 18)

theorem gcf_lcm_sum : A + B = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l2550_255047


namespace NUMINAMATH_CALUDE_cube_sum_equals_one_l2550_255050

theorem cube_sum_equals_one (x y z : ℝ) 
  (h1 : x * y + y * z + z * x = x * y * z) 
  (h2 : x + y + z = 1) : 
  x^3 + y^3 + z^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_one_l2550_255050


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_inequality_holds_l2550_255076

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} :=
by sorry

-- Part 2
theorem range_of_a_when_inequality_holds :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_inequality_holds_l2550_255076


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l2550_255000

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 13 ∧ n ≡ 1729 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l2550_255000


namespace NUMINAMATH_CALUDE_largest_x_value_l2550_255079

theorem largest_x_value : 
  let f (x : ℝ) := (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x
  ∀ x : ℝ, f x = 8 * x - 2 → x ≤ 5/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2550_255079


namespace NUMINAMATH_CALUDE_expression_simplification_l2550_255070

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (((x^2 + 1) / (x - 1) - x + 1) / ((x^2) / (1 - x))) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2550_255070


namespace NUMINAMATH_CALUDE_blue_balls_count_l2550_255061

theorem blue_balls_count (total : ℕ) 
  (h_green : (1 : ℚ) / 4 * total = (total / 4 : ℕ))
  (h_blue : (1 : ℚ) / 8 * total = (total / 8 : ℕ))
  (h_yellow : (1 : ℚ) / 12 * total = (total / 12 : ℕ))
  (h_white : total - (total / 4 + total / 8 + total / 12) = 26) :
  total / 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2550_255061


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2550_255010

theorem matrix_inverse_proof : 
  let M : Matrix (Fin 3) (Fin 3) ℚ := ![![7/29, 5/29, 0], ![3/29, 2/29, 0], ![0, 0, 1]]
  let A : Matrix (Fin 3) (Fin 3) ℚ := ![![2, -5, 0], ![-3, 7, 0], ![0, 0, 1]]
  M * A = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2550_255010


namespace NUMINAMATH_CALUDE_sqrt_identity_l2550_255085

theorem sqrt_identity : (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l2550_255085


namespace NUMINAMATH_CALUDE_distance_sum_is_48_l2550_255063

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 14)
  (ca_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15)

-- Define points Q and R
def Q (t : Triangle) : ℝ × ℝ := sorry
def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the right angle condition
def is_right_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define similarity of triangles
def are_similar (t1 t2 t3 : Triangle) : Prop := sorry

-- Define the distance from a point to a line
def distance_to_line (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem distance_sum_is_48 (t : Triangle) 
  (h1 : is_right_angle (Q t) (t.C) (t.B))
  (h2 : is_right_angle (R t) (t.B) (t.C))
  (P1 P2 : ℝ × ℝ)
  (h3 : are_similar 
    ⟨P1, Q t, R t, sorry, sorry, sorry⟩ 
    ⟨P2, Q t, R t, sorry, sorry, sorry⟩ 
    t) :
  distance_to_line P1 t.B t.C + distance_to_line P2 t.B t.C = 48 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_is_48_l2550_255063


namespace NUMINAMATH_CALUDE_house_number_theorem_l2550_255038

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ := (n / 100) + ((n / 10) % 10) + (n % 10)

def all_digits_same (n : ℕ) : Prop :=
  (n / 100) = ((n / 10) % 10) ∧ ((n / 10) % 10) = (n % 10)

def two_digits_same (n : ℕ) : Prop :=
  (n / 100 = (n / 10) % 10) ∨ ((n / 10) % 10 = n % 10) ∨ (n / 100 = n % 10)

def all_digits_different (n : ℕ) : Prop :=
  (n / 100) ≠ ((n / 10) % 10) ∧ ((n / 10) % 10) ≠ (n % 10) ∧ (n / 100) ≠ (n % 10)

theorem house_number_theorem :
  (∃! n : ℕ, is_three_digit n ∧ digit_sum n = 24 ∧ all_digits_same n) ∧
  (∃ l : List ℕ, l.length = 3 ∧ ∀ n ∈ l, is_three_digit n ∧ digit_sum n = 24 ∧ two_digits_same n) ∧
  (∃ l : List ℕ, l.length = 6 ∧ ∀ n ∈ l, is_three_digit n ∧ digit_sum n = 24 ∧ all_digits_different n) :=
sorry

end NUMINAMATH_CALUDE_house_number_theorem_l2550_255038
