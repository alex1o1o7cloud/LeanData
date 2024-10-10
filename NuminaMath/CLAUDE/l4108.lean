import Mathlib

namespace geometric_sequence_sum_l4108_410884

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℚ) :
  is_geometric_sequence a →
  a 2 * a 5 = -3/4 →
  a 2 + a 3 + a 4 + a 5 = 5/4 →
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
by
  sorry

end geometric_sequence_sum_l4108_410884


namespace expression_value_l4108_410871

theorem expression_value : 
  let a := 2021
  (a^3 - 3*a^2*(a+1) + 4*a*(a+1)^2 - (a+1)^3 + 2) / (a*(a+1)) = 1 + 1/a := by
  sorry

end expression_value_l4108_410871


namespace no_solution_for_k_2_and_3_solution_exists_for_k_ge_4_l4108_410852

theorem no_solution_for_k_2_and_3 :
  (¬ ∃ (m n : ℕ+), m * (m + 2) = n * (n + 1)) ∧
  (¬ ∃ (m n : ℕ+), m * (m + 3) = n * (n + 1)) :=
sorry

theorem solution_exists_for_k_ge_4 :
  ∀ (k : ℕ), k ≥ 4 → ∃ (m n : ℕ+), m * (m + k) = n * (n + 1) :=
sorry

end no_solution_for_k_2_and_3_solution_exists_for_k_ge_4_l4108_410852


namespace cards_given_to_jeff_main_theorem_l4108_410896

/-- Proves that the number of cards Nell gave to Jeff is 276 --/
theorem cards_given_to_jeff : ℕ → ℕ → ℕ → Prop :=
  fun nell_initial nell_remaining cards_given =>
    nell_initial = 528 →
    nell_remaining = 252 →
    cards_given = nell_initial - nell_remaining →
    cards_given = 276

/-- The main theorem --/
theorem main_theorem : ∃ (cards_given : ℕ), cards_given_to_jeff 528 252 cards_given :=
  sorry

end cards_given_to_jeff_main_theorem_l4108_410896


namespace triangle_problem_l4108_410845

theorem triangle_problem (A B C : Real) (BC AB AC : Real) :
  BC = 7 →
  AB = 3 →
  (Real.sin C) / (Real.sin B) = 3/5 →
  AC = 5 ∧ Real.cos A = -1/2 :=
by sorry

end triangle_problem_l4108_410845


namespace john_daily_earnings_l4108_410835

/-- Calculate daily earnings from website visits -/
def daily_earnings (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℚ) : ℚ :=
  (visits_per_month : ℚ) * earnings_per_visit / (days_per_month : ℚ)

/-- Prove that John's daily earnings are $10 -/
theorem john_daily_earnings :
  daily_earnings 30000 30 (1 / 100) = 10 := by
  sorry

end john_daily_earnings_l4108_410835


namespace arrangements_not_adjacent_l4108_410863

theorem arrangements_not_adjacent (n : ℕ) (h : n = 6) : 
  (n.factorial : ℕ) - 2 * ((n - 1).factorial : ℕ) = 480 := by
  sorry

end arrangements_not_adjacent_l4108_410863


namespace sticker_problem_l4108_410850

theorem sticker_problem (x : ℝ) : 
  (x * (1 - 0.25) * (1 - 0.20) = 45) → x = 75 := by
sorry

end sticker_problem_l4108_410850


namespace car_speed_proof_l4108_410841

theorem car_speed_proof (v : ℝ) : v > 0 →
  (3600 / v - 3600 / 225 = 2) ↔ v = 200 :=
by
  sorry

#check car_speed_proof

end car_speed_proof_l4108_410841


namespace number_line_inequalities_l4108_410861

theorem number_line_inequalities (a b c d : ℝ) 
  (ha_neg : a < 0) (hb_neg : b < 0) (hc_pos : c > 0) (hd_pos : d > 0)
  (hc_bounds : 0 < |c| ∧ |c| < 1)
  (hb_bounds : 1 < |b| ∧ |b| < 2)
  (ha_bounds : 2 < |a| ∧ |a| < 4)
  (hd_bounds : 1 < |d| ∧ |d| < 2) : 
  (|a| < 4) ∧ 
  (|b| < 2) ∧ 
  (|c| < 2) ∧ 
  (|a| > |b|) ∧ 
  (|c| < |d|) ∧ 
  (|a - b| < 4) ∧ 
  (|b - c| < 2) ∧ 
  (|c - a| > 1) := by
sorry

end number_line_inequalities_l4108_410861


namespace f_recursive_relation_l4108_410821

def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i => i * i)

theorem f_recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1)^2 + (2 * k + 2)^2 := by
  sorry

end f_recursive_relation_l4108_410821


namespace arithmetic_sequence_4_to_256_l4108_410898

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- The last term of an arithmetic sequence -/
def arithmetic_sequence_last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_4_to_256 :
  arithmetic_sequence_length 4 4 256 = 64 :=
sorry

end arithmetic_sequence_4_to_256_l4108_410898


namespace solution_set_of_inequality_l4108_410874

theorem solution_set_of_inequality (a : ℝ) (h : a > 1) :
  {x : ℝ | |x| + a > 1} = Set.univ :=
by sorry

end solution_set_of_inequality_l4108_410874


namespace necessary_but_not_sufficient_l4108_410827

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0) ∧
  (∃ x y : ℝ, x = 0 ∧ x^2 + y^2 ≠ 0) := by
  sorry

end necessary_but_not_sufficient_l4108_410827


namespace book_pages_calculation_l4108_410897

theorem book_pages_calculation (chapters : ℕ) (days : ℕ) (pages_per_day : ℕ) 
  (h1 : chapters = 41)
  (h2 : days = 30)
  (h3 : pages_per_day = 15) :
  chapters * (days * pages_per_day / chapters) = days * pages_per_day :=
by
  sorry

#check book_pages_calculation

end book_pages_calculation_l4108_410897


namespace lawrence_county_camp_attendance_l4108_410885

/-- The number of kids from Lawrence county who went to camp -/
def kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) : ℕ :=
  total_kids - kids_at_home

/-- Theorem: Given the total number of kids in Lawrence county and the number of kids who stayed home,
    prove that the number of kids who went to camp is 893,835 -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1538832 644997 = 893835 := by
  sorry

end lawrence_county_camp_attendance_l4108_410885


namespace square_of_binomial_l4108_410818

theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 20 * x + 9 = (r * x + s)^2) → 
  a = 100 / 9 := by
sorry

end square_of_binomial_l4108_410818


namespace abc_plus_def_equals_zero_l4108_410860

/-- Represents the transformation of numbers in the circle --/
def transform (v : Fin 6 → ℝ) : Fin 6 → ℝ := fun i =>
  v i + v (i - 1) + v (i + 1)

/-- The condition that after 2022 iterations, the numbers return to their initial values --/
def returns_to_initial (v : Fin 6 → ℝ) : Prop :=
  (transform^[2022]) v = v

theorem abc_plus_def_equals_zero 
  (v : Fin 6 → ℝ) 
  (h : returns_to_initial v) : 
  v 0 * v 1 * v 2 + v 3 * v 4 * v 5 = 0 := by
  sorry

end abc_plus_def_equals_zero_l4108_410860


namespace at_least_one_negative_l4108_410805

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 + 1/b = b^2 + 1/a) :
  a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l4108_410805


namespace square_minus_product_plus_square_l4108_410823

theorem square_minus_product_plus_square : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end square_minus_product_plus_square_l4108_410823


namespace mode_of_scores_l4108_410899

def scores : List ℕ := [35, 37, 39, 37, 38, 38, 37]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_scores :
  mode scores = 37 := by
  sorry

end mode_of_scores_l4108_410899


namespace age_difference_l4108_410849

-- Define the ages as natural numbers
def rona_age : ℕ := 8
def rachel_age : ℕ := 2 * rona_age
def collete_age : ℕ := rona_age / 2

-- Theorem statement
theorem age_difference : rachel_age - collete_age = 12 := by
  sorry

end age_difference_l4108_410849


namespace gdp_growth_problem_l4108_410887

/-- The GDP growth over a period of years -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ years

/-- The GDP growth problem -/
theorem gdp_growth_problem :
  let initial_gdp := 9593.3
  let growth_rate := 0.073
  let years := 4
  ∃ ε > 0, |gdp_growth initial_gdp growth_rate years - 127165| < ε :=
by sorry

end gdp_growth_problem_l4108_410887


namespace power_inequality_l4108_410843

theorem power_inequality (x : ℝ) (h : x < 27) : 27^9 > x^24 := by
  sorry

end power_inequality_l4108_410843


namespace simplify_fraction_l4108_410844

theorem simplify_fraction : 15 * (16 / 9) * (-45 / 32) = -25 / 6 := by
  sorry

end simplify_fraction_l4108_410844


namespace triangle_with_arithmetic_angles_and_side_ratio_l4108_410880

theorem triangle_with_arithmetic_angles_and_side_ratio (α β γ : Real) (a b c : Real) :
  -- Angles form an arithmetic progression
  β - α = γ - β →
  -- Sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- Smallest side is half of the largest side
  a = c / 2 →
  -- Side lengths satisfy the sine law
  a / Real.sin α = b / Real.sin β →
  b / Real.sin β = c / Real.sin γ →
  -- Angles are positive
  0 < α ∧ 0 < β ∧ 0 < γ →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Prove that the angles are 30°, 60°, and 90°
  (α = 30 ∧ β = 60 ∧ γ = 90) :=
by sorry

end triangle_with_arithmetic_angles_and_side_ratio_l4108_410880


namespace train_speed_l4108_410832

/-- The speed of a train given specific passing times -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) 
  (h_pole : t_pole = 10)
  (h_stationary : t_stationary = 30)
  (h_length : l_stationary = 600) :
  ∃ v : ℝ, v = 30 ∧ v * t_pole = v * t_stationary - l_stationary :=
by sorry

end train_speed_l4108_410832


namespace product_of_squares_and_fourth_powers_l4108_410876

theorem product_of_squares_and_fourth_powers (r s : ℝ) 
  (h_positive_r : r > 0) (h_positive_s : s > 0)
  (h_sum_squares : r^2 + s^2 = 1) 
  (h_sum_fourth_powers : r^4 + s^4 = 7/8) : r * s = 1/4 := by
  sorry

end product_of_squares_and_fourth_powers_l4108_410876


namespace f_monotone_and_F_lower_bound_l4108_410879

noncomputable section

variables (m : ℝ) (x x₀ : ℝ)

def f (x : ℝ) : ℝ := x * Real.exp x - m * x

def F (x : ℝ) : ℝ := f m x - m * Real.log x

theorem f_monotone_and_F_lower_bound (hm : m < -Real.exp (-2)) 
  (h_crit : deriv (F m) x₀ = 0) (h_pos : F m x₀ > 0) :
  (∀ x y, x < y → f m x < f m y) ∧ F m x₀ > -2 * x₀^3 + 2 * x₀ := by
  sorry

end f_monotone_and_F_lower_bound_l4108_410879


namespace one_third_and_three_eightyone_in_cantor_cantor_iteration_length_l4108_410802

/-- The Cantor set constructed by repeatedly removing the middle third of each interval --/
def CantorSet : Set ℝ :=
  sorry

/-- The nth iteration in the Cantor set construction --/
def CantorIteration (n : ℕ) : Set (Set ℝ) :=
  sorry

/-- The length of the nth iteration in the Cantor set construction --/
def CantorIterationLength (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating that 1/3 and 3/81 belong to the Cantor set --/
theorem one_third_and_three_eightyone_in_cantor :
  (1/3 : ℝ) ∈ CantorSet ∧ (3/81 : ℝ) ∈ CantorSet :=
sorry

/-- Theorem stating the length of the nth iteration in the Cantor set construction --/
theorem cantor_iteration_length (n : ℕ) :
  CantorIterationLength n = (2/3 : ℝ) ^ (n - 1) :=
sorry

end one_third_and_three_eightyone_in_cantor_cantor_iteration_length_l4108_410802


namespace unique_solution_x_squared_minus_two_factorial_y_l4108_410872

theorem unique_solution_x_squared_minus_two_factorial_y : 
  ∃! (x y : ℕ+), x^2 - 2 * Nat.factorial y.val = 2021 :=
by
  -- The proof goes here
  sorry

end unique_solution_x_squared_minus_two_factorial_y_l4108_410872


namespace smallest_digit_sum_of_successor_l4108_410889

def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_digit_sum_of_successor (n : ℕ) (h : digit_sum n = 2017) :
  ∃ (m : ℕ), digit_sum (n + 1) = m ∧ ∀ (k : ℕ), digit_sum (n + 1) ≤ k := by sorry

end smallest_digit_sum_of_successor_l4108_410889


namespace sqrt_equation_root_l4108_410811

theorem sqrt_equation_root : 
  ∃ x : ℝ, x = 35.0625 ∧ Real.sqrt (x - 2) + Real.sqrt (x + 4) = 12 := by
  sorry

end sqrt_equation_root_l4108_410811


namespace min_fraction_sum_l4108_410891

theorem min_fraction_sum (A B C D : ℕ) : 
  A ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  B ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  C ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  D ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (A : ℚ) / B + (C : ℚ) / D ≥ 1 / 8 :=
by sorry

end min_fraction_sum_l4108_410891


namespace triangle_theorem_l4108_410888

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with geometric sequence sides -/
theorem triangle_theorem (t : Triangle) 
  (geom_seq : t.b^2 = t.a * t.c)
  (cos_B : Real.cos t.B = 3/5)
  (area : 1/2 * t.a * t.c * Real.sin t.B = 2) :
  (t.a + t.b + t.c = Real.sqrt 5 + Real.sqrt 21) ∧
  ((Real.sqrt 5 - 1)/2 < (Real.sin t.A + Real.cos t.A * Real.tan t.C) / 
                         (Real.sin t.B + Real.cos t.B * Real.tan t.C) ∧
   (Real.sin t.A + Real.cos t.A * Real.tan t.C) / 
   (Real.sin t.B + Real.cos t.B * Real.tan t.C) < (Real.sqrt 5 + 1)/2) :=
by sorry

end triangle_theorem_l4108_410888


namespace no_solution_when_m_negative_four_point_five_l4108_410873

/-- The vector equation has no solutions when m = -4.5 -/
theorem no_solution_when_m_negative_four_point_five :
  let m : ℝ := -4.5
  let v1 : ℝ × ℝ := (1, 3)
  let v2 : ℝ × ℝ := (2, -3)
  let v3 : ℝ × ℝ := (-1, 4)
  let v4 : ℝ × ℝ := (3, m)
  ¬∃ (t s : ℝ), v1 + t • v2 = v3 + s • v4 :=
by sorry


end no_solution_when_m_negative_four_point_five_l4108_410873


namespace gcd_seven_factorial_six_factorial_l4108_410812

theorem gcd_seven_factorial_six_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 6) = Nat.factorial 6 := by
  sorry

end gcd_seven_factorial_six_factorial_l4108_410812


namespace problem_solution_l4108_410804

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + a

-- State the theorem
theorem problem_solution :
  -- Part 1: Find the value of a
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  -- Part 2: Find the minimum value of m
  (let a := 1 -- Use the value of a found in part 1
   ∃ (m : ℝ), (∃ (n : ℝ), f n a ≤ m - f (-n) a) ∧
              ∀ (m' : ℝ), (∃ (n : ℝ), f n a ≤ m' - f (-n) a) → m' ≥ m) ∧
  -- The actual solutions
  (let a := 1
   let m := 4
   (∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
   (∃ (n : ℝ), f n a ≤ m - f (-n) a) ∧
   ∀ (m' : ℝ), (∃ (n : ℝ), f n a ≤ m' - f (-n) a) → m' ≥ m) :=
by sorry

end problem_solution_l4108_410804


namespace horizontal_grid_lines_length_6_10_l4108_410867

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the total length of horizontal grid lines inside a right-angled triangle -/
def horizontalGridLinesLength (t : GridTriangle) : ℕ :=
  (t.base * (t.height - 1)) / 2

/-- The theorem stating the total length of horizontal grid lines for the specific triangle -/
theorem horizontal_grid_lines_length_6_10 :
  horizontalGridLinesLength { base := 10, height := 6 } = 27 := by
  sorry

#eval horizontalGridLinesLength { base := 10, height := 6 }

end horizontal_grid_lines_length_6_10_l4108_410867


namespace max_distance_for_given_tires_l4108_410803

/-- Represents the maximum distance a car can travel with one tire swap --/
def max_distance_with_swap (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  front_tire_life + (rear_tire_life - front_tire_life) / 2

/-- Theorem: Given specific tire lifespans, the maximum distance with one swap is 48,000 km --/
theorem max_distance_for_given_tires :
  max_distance_with_swap 42000 56000 = 48000 := by
  sorry

#eval max_distance_with_swap 42000 56000

end max_distance_for_given_tires_l4108_410803


namespace circle_intersection_theorem_l4108_410866

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 10

-- Define the line y = -x + m
def line (x y m : ℝ) : Prop := y = -x + m

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 1)
def point_B : ℝ × ℝ := (1, 3)

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ line x₁ y₁ m ∧
    circle_C x₂ y₂ ∧ line x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the condition for MN to pass through the origin
def passes_through_origin (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ line x₁ y₁ m ∧
    circle_C x₂ y₂ ∧ line x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ + x₂)^2 + (y₁ + y₂)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem circle_intersection_theorem :
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 ∧
  (∀ m : ℝ, intersection_points m) →
  passes_through_origin (1 + Real.sqrt 7) ∧
  passes_through_origin (1 - Real.sqrt 7) :=
sorry

end circle_intersection_theorem_l4108_410866


namespace lucy_fish_count_l4108_410815

theorem lucy_fish_count (initial_fish : ℕ) (fish_to_buy : ℕ) (total_fish : ℕ) : 
  initial_fish = 212 → fish_to_buy = 68 → total_fish = initial_fish + fish_to_buy → total_fish = 280 := by
  sorry

end lucy_fish_count_l4108_410815


namespace opposite_numbers_and_reciprocal_l4108_410808

theorem opposite_numbers_and_reciprocal (a b c : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : 1 / c = 4)  -- the reciprocal of c is 4
  : 3 * a + 3 * b - 4 * c = -1 := by
  sorry

end opposite_numbers_and_reciprocal_l4108_410808


namespace cosine_of_angle_between_vectors_l4108_410828

/-- Given vectors a and b in ℝ², prove that the cosine of the angle between them is √5 / 5 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (2, -4) → b = (-3, -4) → 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 5 / 5 := by
  sorry

end cosine_of_angle_between_vectors_l4108_410828


namespace opposite_of_abs_neg_half_l4108_410831

theorem opposite_of_abs_neg_half : 
  -(|(-0.5 : ℝ)|) = -0.5 := by sorry

end opposite_of_abs_neg_half_l4108_410831


namespace regular_polygon_sides_and_exterior_angle_l4108_410870

/-- 
Theorem: For a regular polygon with n sides, if the sum of its interior angles 
is greater than the sum of its exterior angles by 360°, then n = 6 and each 
exterior angle measures 60°.
-/
theorem regular_polygon_sides_and_exterior_angle (n : ℕ) : 
  (n ≥ 3) →  -- Ensure the polygon has at least 3 sides
  (180 * (n - 2) = 360 + 360) →  -- Sum of interior angles equals 360° + sum of exterior angles
  (n = 6 ∧ 360 / n = 60) :=
by sorry

end regular_polygon_sides_and_exterior_angle_l4108_410870


namespace given_terms_are_like_l4108_410839

/-- Two algebraic terms are like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : String) : Prop := sorry

/-- The first term in the pair. -/
def term1 : String := "-m^2n^3"

/-- The second term in the pair. -/
def term2 : String := "-3n^3m^2"

/-- Theorem stating that the given terms are like terms. -/
theorem given_terms_are_like : are_like_terms term1 term2 := by sorry

end given_terms_are_like_l4108_410839


namespace boat_current_speed_l4108_410865

/-- Proves that given a boat with a speed of 20 km/hr in still water, 
    traveling 9.2 km downstream in 24 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : downstream_distance = 9.2)
  (h3 : time_minutes = 24) :
  let time_hours : ℝ := time_minutes / 60
  let current_speed : ℝ := downstream_distance / time_hours - boat_speed
  current_speed = 3 := by sorry

end boat_current_speed_l4108_410865


namespace chess_tournament_games_l4108_410814

/-- The number of games played in a chess tournament -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 15 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 105. -/
theorem chess_tournament_games :
  numGames 15 = 105 := by
  sorry

end chess_tournament_games_l4108_410814


namespace time_rosa_sees_leo_l4108_410851

/-- Calculates the time Rosa can see Leo given their speeds and distances -/
theorem time_rosa_sees_leo (rosa_speed leo_speed initial_distance final_distance : ℚ) :
  rosa_speed = 15 →
  leo_speed = 5 →
  initial_distance = 3/4 →
  final_distance = 3/4 →
  (initial_distance + final_distance) / (rosa_speed - leo_speed) * 60 = 9 := by
  sorry

#check time_rosa_sees_leo

end time_rosa_sees_leo_l4108_410851


namespace problem_solution_l4108_410834

theorem problem_solution (a b : ℝ) 
  (h1 : a^3 - b^3 = 4)
  (h2 : a^2 + a*b + b^2 + a - b = 4) : 
  a - b = 2 := by sorry

end problem_solution_l4108_410834


namespace sum_removal_proof_l4108_410848

theorem sum_removal_proof : 
  let original_sum := (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13
  let removed_terms := 1 / 11 + 1 / 13
  original_sum - removed_terms = 3 / 2 := by
  sorry

end sum_removal_proof_l4108_410848


namespace probability_of_specific_three_card_arrangement_l4108_410816

/-- The number of possible arrangements of n distinct objects -/
def numberOfArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The probability of a specific arrangement given n distinct objects -/
def probabilityOfSpecificArrangement (n : ℕ) : ℚ :=
  1 / (numberOfArrangements n)

theorem probability_of_specific_three_card_arrangement :
  probabilityOfSpecificArrangement 3 = 1 / 6 := by
  sorry

end probability_of_specific_three_card_arrangement_l4108_410816


namespace num_cows_bought_l4108_410853

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of goats Zara bought -/
def num_goats : ℕ := 113

/-- The number of groups for transporting animals -/
def num_groups : ℕ := 3

/-- The number of animals per group -/
def animals_per_group : ℕ := 48

/-- The total number of animals Zara bought -/
def total_animals : ℕ := num_groups * animals_per_group

theorem num_cows_bought : 
  total_animals - (num_sheep + num_goats) = 24 :=
by sorry

end num_cows_bought_l4108_410853


namespace quadratic_solution_difference_squared_l4108_410822

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ, (2 * a^2 + 7 * a - 15 = 0) ∧ (2 * b^2 + 7 * b - 15 = 0) → (a - b)^2 = 169/4 := by
  sorry

end quadratic_solution_difference_squared_l4108_410822


namespace quadrilateral_centers_collinearity_l4108_410890

-- Define the points
variable (A B C D E U H V K : Euclidean_plane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Euclidean_plane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D E : Euclidean_plane) : Prop := sorry

-- Define the circumcenter
def is_circumcenter (U A B E : Euclidean_plane) : Prop := sorry

-- Define the orthocenter
def is_orthocenter (H A B E : Euclidean_plane) : Prop := sorry

-- Define collinearity
def collinear (P Q R : Euclidean_plane) : Prop := sorry

-- State the theorem
theorem quadrilateral_centers_collinearity 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_intersect_at A B C D E)
  (h3 : is_circumcenter U A B E)
  (h4 : is_orthocenter H A B E)
  (h5 : is_circumcenter V C D E)
  (h6 : is_orthocenter K C D E) :
  collinear U E K ↔ collinear V E H := by sorry

end quadrilateral_centers_collinearity_l4108_410890


namespace proportion_sum_condition_l4108_410864

theorem proportion_sum_condition 
  (a b c d a₁ b₁ c₁ d₁ : ℚ) 
  (h1 : a / b = c / d) 
  (h2 : a₁ / b₁ = c₁ / d₁) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) 
  (h5 : b₁ ≠ 0) 
  (h6 : d₁ ≠ 0) 
  (h7 : b + b₁ ≠ 0) 
  (h8 : d + d₁ ≠ 0) : 
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ :=
by sorry

end proportion_sum_condition_l4108_410864


namespace f_triple_eq_f_solutions_bound_l4108_410854

noncomputable def f (x : ℝ) : ℝ := -3 * Real.sin (Real.pi * x)

theorem f_triple_eq_f_solutions_bound :
  ∃ (S : Finset ℝ), (∀ x ∈ S, -1 ≤ x ∧ x ≤ 1) ∧ 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (f (f x)) = f x → x ∈ S) ∧
  Finset.card S ≤ 6 :=
sorry

end f_triple_eq_f_solutions_bound_l4108_410854


namespace tan_315_eq_neg_one_l4108_410894

/-- Prove that the tangent of 315 degrees is equal to -1 -/
theorem tan_315_eq_neg_one : Real.tan (315 * π / 180) = -1 := by
  sorry


end tan_315_eq_neg_one_l4108_410894


namespace minutes_to_seconds_conversion_seconds_to_minutes_conversion_l4108_410882

-- Define the conversion factor
def seconds_per_minute : ℝ := 60

-- Define the number of minutes
def minutes : ℝ := 8.5

-- Theorem to prove
theorem minutes_to_seconds_conversion :
  minutes * seconds_per_minute = 510 := by
  sorry

-- Verification theorem
theorem seconds_to_minutes_conversion :
  510 / seconds_per_minute = minutes := by
  sorry

end minutes_to_seconds_conversion_seconds_to_minutes_conversion_l4108_410882


namespace gina_sister_choice_ratio_l4108_410877

/-- The ratio of Gina's choices to her sister's choices on Netflix --/
theorem gina_sister_choice_ratio :
  ∀ (sister_shows : ℕ) (show_length : ℕ) (gina_minutes : ℕ),
  sister_shows = 24 →
  show_length = 50 →
  gina_minutes = 900 →
  (gina_minutes : ℚ) / (sister_shows * show_length : ℚ) = 3 / 4 := by
  sorry

end gina_sister_choice_ratio_l4108_410877


namespace triangle_side_square_sum_bound_l4108_410840

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_radius : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The sum of squares of triangle sides is less than or equal to 9 times the square of its circumradius -/
theorem triangle_side_square_sum_bound (t : Triangle) : t.a^2 + t.b^2 + t.c^2 ≤ 9 * t.R^2 := by
  sorry

end triangle_side_square_sum_bound_l4108_410840


namespace average_speed_three_sections_l4108_410801

/-- The average speed of a person traveling on a 1 km street divided into three equal sections,
    with speeds of 4 km/h, 10 km/h, and 6 km/h in each section respectively. -/
theorem average_speed_three_sections (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_distance = 1 →
  speed1 = 4 →
  speed2 = 10 →
  speed3 = 6 →
  let section_distance := total_distance / 3
  let time1 := section_distance / speed1
  let time2 := section_distance / speed2
  let time3 := section_distance / speed3
  let total_time := time1 + time2 + time3
  total_distance / total_time = 180 / 31 :=
by sorry

end average_speed_three_sections_l4108_410801


namespace max_value_x_cubed_over_polynomial_l4108_410857

theorem max_value_x_cubed_over_polynomial (x : ℝ) :
  x^3 / (x^6 + x^4 + x^3 - 3*x^2 + 9) ≤ 1/7 ∧
  ∃ y : ℝ, y^3 / (y^6 + y^4 + y^3 - 3*y^2 + 9) = 1/7 :=
by sorry

end max_value_x_cubed_over_polynomial_l4108_410857


namespace probability_in_specific_club_l4108_410893

/-- A club with members of different genders and seniority levels -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  senior_boys : ℕ
  junior_boys : ℕ
  senior_girls : ℕ
  junior_girls : ℕ

/-- The probability of selecting two girls, one senior and one junior, from the club -/
def probability_two_girls_diff_seniority (c : Club) : ℚ :=
  (c.senior_girls.choose 1 * c.junior_girls.choose 1 : ℚ) / c.total_members.choose 2

/-- Theorem stating the probability for the given club configuration -/
theorem probability_in_specific_club : 
  ∃ c : Club, 
    c.total_members = 12 ∧ 
    c.boys = 6 ∧ 
    c.girls = 6 ∧ 
    c.senior_boys = 3 ∧ 
    c.junior_boys = 3 ∧ 
    c.senior_girls = 3 ∧ 
    c.junior_girls = 3 ∧ 
    probability_two_girls_diff_seniority c = 9 / 66 := by
  sorry

end probability_in_specific_club_l4108_410893


namespace binary_101101_equals_45_l4108_410807

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_equals_45_l4108_410807


namespace sphere_only_identical_views_l4108_410817

-- Define the possible geometric bodies
inductive GeometricBody
  | Sphere
  | Cube
  | RegularTetrahedron

-- Define a function that checks if all views are identical
def hasIdenticalViews (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_identical_views :
  ∀ (body : GeometricBody),
    hasIdenticalViews body ↔ body = GeometricBody.Sphere :=
by sorry

end sphere_only_identical_views_l4108_410817


namespace function_inequality_l4108_410892

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 1

-- State the theorem
theorem function_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 1| < b → |f x + 4| < a) ↔ b ≤ a / 4 := by sorry

end function_inequality_l4108_410892


namespace volume_of_sphere_wedge_l4108_410836

/-- The volume of a wedge when a sphere with circumference 18π is cut into 6 congruent parts -/
theorem volume_of_sphere_wedge : 
  ∀ (r : ℝ) (V : ℝ),
  (2 * Real.pi * r = 18 * Real.pi) →  -- Circumference condition
  (V = (4/3) * Real.pi * r^3) →       -- Volume of sphere formula
  (V / 6 = 162 * Real.pi) :=          -- Volume of one wedge
by sorry

end volume_of_sphere_wedge_l4108_410836


namespace perfect_square_function_characterization_l4108_410813

/-- A function g: ℕ → ℕ satisfies the perfect square property if 
    (g(m) + n)(m + g(n)) is a perfect square for all m, n ∈ ℕ -/
def PerfectSquareProperty (g : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (m + g n) = k * k

/-- The main theorem characterizing functions with the perfect square property -/
theorem perfect_square_function_characterization :
  ∀ g : ℕ → ℕ, PerfectSquareProperty g ↔ ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by sorry

end perfect_square_function_characterization_l4108_410813


namespace nested_square_root_value_l4108_410856

theorem nested_square_root_value : ∃ y : ℝ, y > 0 ∧ y = Real.sqrt (3 - y) ∧ y = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_square_root_value_l4108_410856


namespace road_cleaning_problem_l4108_410842

/-- The distance between East City and West City in kilometers -/
def total_distance : ℝ := 60

/-- The time it takes Vehicle A to clean the entire road alone in hours -/
def time_A : ℝ := 10

/-- The time it takes Vehicle B to clean the entire road alone in hours -/
def time_B : ℝ := 15

/-- The additional distance cleaned by Vehicle A compared to Vehicle B when they meet, in kilometers -/
def extra_distance_A : ℝ := 12

theorem road_cleaning_problem :
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let combined_speed := speed_A + speed_B
  let meeting_time := total_distance / combined_speed
  speed_A * meeting_time - speed_B * meeting_time = extra_distance_A :=
by sorry

#check road_cleaning_problem

end road_cleaning_problem_l4108_410842


namespace polynomial_divisibility_l4108_410806

def polynomial (p q : ℚ) (x : ℚ) : ℚ :=
  x^6 - x^5 + x^4 - p*x^3 + q*x^2 + 6*x - 8

theorem polynomial_divisibility (p q : ℚ) :
  (∀ x, (x + 2 = 0 ∨ x - 1 = 0 ∨ x - 3 = 0) → polynomial p q x = 0) ↔
  (p = -26/3 ∧ q = -26/3) :=
sorry

end polynomial_divisibility_l4108_410806


namespace range_of_a_l4108_410847

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x / (x - 1) < 1

def q (x a : ℝ) : Prop := (x + a) * (x - 3) > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ∈ Set.Iic (-1 : ℝ) := by sorry

end range_of_a_l4108_410847


namespace fraction_problem_l4108_410825

theorem fraction_problem (f : ℚ) : f * 50 - 4 = 6 → f = 1/5 := by
  sorry

end fraction_problem_l4108_410825


namespace lizzys_money_l4108_410862

theorem lizzys_money (mother_gave : ℕ) (spent_on_candy : ℕ) (uncle_gave : ℕ) (final_amount : ℕ) :
  mother_gave = 80 →
  spent_on_candy = 50 →
  uncle_gave = 70 →
  final_amount = 140 →
  ∃ (father_gave : ℕ), father_gave = 40 ∧ mother_gave + father_gave - spent_on_candy + uncle_gave = final_amount :=
by sorry

end lizzys_money_l4108_410862


namespace two_digit_number_ratio_l4108_410895

theorem two_digit_number_ratio (a b : ℕ) : 
  a ≤ 9 ∧ b ≤ 9 ∧ a ≠ 0 → -- Ensure a and b are single digits and a is not 0
  (10 * a + b) * 6 = (10 * b + a) * 5 → -- Ratio condition
  10 * a + b = 45 := by
sorry

end two_digit_number_ratio_l4108_410895


namespace february_2020_average_rainfall_l4108_410883

/-- Calculate the average rainfall per hour in February 2020 --/
theorem february_2020_average_rainfall
  (total_rainfall : ℝ)
  (february_days : ℕ)
  (hours_per_day : ℕ)
  (h1 : total_rainfall = 290)
  (h2 : february_days = 29)
  (h3 : hours_per_day = 24) :
  total_rainfall / (february_days * hours_per_day : ℝ) = 290 / 696 :=
by sorry

end february_2020_average_rainfall_l4108_410883


namespace coupon_collection_probability_l4108_410886

theorem coupon_collection_probability (n m k : ℕ) (hn : n = 17) (hm : m = 9) (hk : k = 6) :
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end coupon_collection_probability_l4108_410886


namespace sum_of_fractions_l4108_410830

theorem sum_of_fractions : (1 : ℚ) / 9 + (1 : ℚ) / 11 = 20 / 99 := by sorry

end sum_of_fractions_l4108_410830


namespace impossible_circle_arrangement_l4108_410846

theorem impossible_circle_arrangement : ¬ ∃ (a : Fin 7 → ℕ),
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 1) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 2) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 3) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 4) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 5) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 6) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 7) :=
by
  sorry


end impossible_circle_arrangement_l4108_410846


namespace goldfish_cost_price_l4108_410810

theorem goldfish_cost_price (selling_price : ℝ) (goldfish_sold : ℕ) (tank_cost : ℝ) (profit_percentage : ℝ) :
  selling_price = 0.75 →
  goldfish_sold = 110 →
  tank_cost = 100 →
  profit_percentage = 0.55 →
  ∃ (cost_price : ℝ),
    cost_price = 0.25 ∧
    (goldfish_sold : ℝ) * (selling_price - cost_price) = profit_percentage * tank_cost :=
by sorry

end goldfish_cost_price_l4108_410810


namespace range_of_a_l4108_410859

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + a

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 0 2, f a x ≥ 0
def q (a : ℝ) : Prop := ∃ x > 0, g a x = 0

-- State the theorem
theorem range_of_a :
  {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} =
  Set.Icc (-1) (-1/2) ∪ Set.Ioi 0 :=
by sorry

end range_of_a_l4108_410859


namespace ball_probability_l4108_410881

theorem ball_probability (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 3 →
  (red_balls : ℚ) / (red_balls + white_balls : ℚ) = 3 / 7 →
  white_balls = 4 := by
sorry

end ball_probability_l4108_410881


namespace tan_x_minus_pi_sixth_l4108_410809

theorem tan_x_minus_pi_sixth (x : Real) 
  (h : Real.sin (π / 3 - x) = (1 / 2) * Real.cos (x - π / 2)) : 
  Real.tan (x - π / 6) = Real.sqrt 3 / 9 := by
  sorry

end tan_x_minus_pi_sixth_l4108_410809


namespace complex_fraction_simplification_l4108_410838

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (1 + 2*i) / i = -2 + i := by
  sorry

end complex_fraction_simplification_l4108_410838


namespace exam_attendance_l4108_410800

theorem exam_attendance (passed_percentage : ℝ) (failed_count : ℕ) : 
  passed_percentage = 35 → 
  failed_count = 351 → 
  (failed_count : ℝ) / (100 - passed_percentage) * 100 = 540 := by
sorry

end exam_attendance_l4108_410800


namespace total_bookmark_sales_l4108_410824

/-- Represents the sales of bookmarks over two days -/
structure BookmarkSales where
  /-- Number of bookmarks sold on the first day -/
  day1 : ℕ
  /-- Number of bookmarks sold on the second day -/
  day2 : ℕ

/-- Theorem stating that the total number of bookmarks sold over two days is 3m-3 -/
theorem total_bookmark_sales (m : ℕ) (sales : BookmarkSales)
    (h1 : sales.day1 = m)
    (h2 : sales.day2 = 2 * m - 3) :
    sales.day1 + sales.day2 = 3 * m - 3 := by
  sorry

end total_bookmark_sales_l4108_410824


namespace absolute_value_ratio_l4108_410858

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 18*a*b) :
  |((a+b)/(a-b))| = Real.sqrt 5 / 2 := by
sorry

end absolute_value_ratio_l4108_410858


namespace solve_equation_l4108_410868

theorem solve_equation (p q : ℝ) (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 := by
  sorry

end solve_equation_l4108_410868


namespace f_difference_at_five_l4108_410829

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 50 := by
  sorry

end f_difference_at_five_l4108_410829


namespace smallest_two_digit_prime_reverse_square_l4108_410855

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem stating that 19 is the smallest two-digit prime number
    whose reverse is a perfect square -/
theorem smallest_two_digit_prime_reverse_square :
  (∀ n : ℕ, 10 ≤ n ∧ n < 19 ∧ is_prime n → ¬(is_square (reverse_digits n))) ∧
  (19 ≤ 99 ∧ is_prime 19 ∧ is_square (reverse_digits 19)) :=
sorry

end smallest_two_digit_prime_reverse_square_l4108_410855


namespace bucket_ratio_l4108_410878

theorem bucket_ratio (small_bucket : ℚ) (large_bucket : ℚ) : 
  (∃ (n : ℚ), large_bucket = n * small_bucket + 3) →
  2 * small_bucket + 5 * large_bucket = 63 →
  large_bucket = 4 →
  large_bucket / small_bucket = 4 := by
sorry

end bucket_ratio_l4108_410878


namespace tricycle_count_l4108_410833

/-- Represents the number of tricycles in a group of children riding bicycles and tricycles -/
def num_tricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  total_wheels - 2 * total_children

/-- Theorem stating that given 7 children and 19 wheels, the number of tricycles is 5 -/
theorem tricycle_count : num_tricycles 7 19 = 5 := by
  sorry

#eval num_tricycles 7 19  -- Should output 5

end tricycle_count_l4108_410833


namespace parabolas_same_vertex_l4108_410819

/-- 
Two parabolas have the same vertex if and only if their coefficients satisfy specific relations.
-/
theorem parabolas_same_vertex (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (∃ (x y : ℝ), 
    (x = -b / (2 * a) ∧ y = a * x^2 + b * x + c) ∧
    (x = -c / (2 * b) ∧ y = b * x^2 + c * x + a)) ↔
  (b = -2 * a ∧ c = 4 * a) :=
sorry

end parabolas_same_vertex_l4108_410819


namespace angle_trigonometry_l4108_410820

theorem angle_trigonometry (a : ℝ) (θ : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (4*a, 3*a)
  (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ) →
  (Real.sin θ = -3/5 ∧ Real.cos θ = -4/5) ∧
  ((1 + 2 * Real.sin (π + θ) * Real.cos (2023 * π - θ)) / 
   (Real.sin (π/2 + θ)^2 - Real.cos (5*π/2 - θ)^2) = 7) :=
by sorry

end angle_trigonometry_l4108_410820


namespace cubic_expression_value_l4108_410875

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 7 * p - 6 = 0 →
  3 * q^2 - 7 * q - 6 = 0 →
  p ≠ q →
  (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  sorry

end cubic_expression_value_l4108_410875


namespace proposition_false_iff_a_in_range_l4108_410869

theorem proposition_false_iff_a_in_range (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
  sorry

end proposition_false_iff_a_in_range_l4108_410869


namespace circle_passes_through_intersections_and_tangent_to_line_l4108_410837

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1)^2 = 5/4

-- Theorem statement
theorem circle_passes_through_intersections_and_tangent_to_line :
  ∀ x y : ℝ,
  (C₁ x y ∧ C₂ x y → desiredCircle x y) ∧
  (∃ t : ℝ, l (1/2 + t) (1 - t/2) ∧
    ∀ s : ℝ, s ≠ t → ¬(desiredCircle (1/2 + s) (1 - s/2))) :=
by sorry

end circle_passes_through_intersections_and_tangent_to_line_l4108_410837


namespace abigail_report_time_l4108_410826

/-- Calculates the time needed to finish a report given the total words required,
    words already written, and typing speed. -/
def timeToFinishReport (totalWords : ℕ) (writtenWords : ℕ) (wordsPerHalfHour : ℕ) : ℕ :=
  let remainingWords := totalWords - writtenWords
  let wordsPerMinute := wordsPerHalfHour / 30
  remainingWords / wordsPerMinute

/-- Proves that given the conditions in the problem, 
    it will take 80 minutes to finish the report. -/
theorem abigail_report_time : 
  timeToFinishReport 1000 200 300 = 80 := by
  sorry

end abigail_report_time_l4108_410826
