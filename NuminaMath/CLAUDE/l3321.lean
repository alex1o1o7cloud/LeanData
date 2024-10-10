import Mathlib

namespace trees_in_yard_l3321_332125

/-- The number of trees planted along a yard with given specifications -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating the number of trees planted along the yard -/
theorem trees_in_yard :
  number_of_trees 273 21 = 14 := by
  sorry

end trees_in_yard_l3321_332125


namespace dog_bones_total_dog_bones_example_l3321_332114

/-- Given a dog with an initial number of bones and a number of bones dug up,
    the total number of bones is equal to the sum of the initial bones and dug up bones. -/
theorem dog_bones_total (initial_bones dug_up_bones : ℕ) :
  initial_bones + dug_up_bones = initial_bones + dug_up_bones := by
  sorry

/-- The specific case from the problem -/
theorem dog_bones_example : 
  493 + 367 = 860 := by
  sorry

end dog_bones_total_dog_bones_example_l3321_332114


namespace union_C_R_A_B_eq_expected_result_l3321_332177

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

-- Define the complement of A with respect to ℝ
def C_R_A : Set ℝ := { x | x ∉ A }

-- Define the union of C_R A and B
def union_C_R_A_B : Set ℝ := C_R_A ∪ B

-- Define the expected result
def expected_result : Set ℝ := { x | x < -1 ∨ 1 < x }

-- Theorem statement
theorem union_C_R_A_B_eq_expected_result : union_C_R_A_B = expected_result := by
  sorry

end union_C_R_A_B_eq_expected_result_l3321_332177


namespace inequality_proof_l3321_332187

theorem inequality_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  a + b < b + c := by
  sorry

end inequality_proof_l3321_332187


namespace all_trains_return_to_initial_positions_city_n_trains_return_after_2016_minutes_l3321_332130

/-- Represents a metro line with its one-way travel time -/
structure MetroLine where
  one_way_time : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  red_line : MetroLine
  blue_line : MetroLine
  green_line : MetroLine

/-- Checks if a train returns to its initial position after given minutes -/
def returns_to_initial_position (line : MetroLine) (minutes : ℕ) : Prop :=
  minutes % (2 * line.one_way_time) = 0

/-- The theorem stating that all trains return to their initial positions after 2016 minutes -/
theorem all_trains_return_to_initial_positions (metro : MetroSystem) :
  returns_to_initial_position metro.red_line 2016 ∧
  returns_to_initial_position metro.blue_line 2016 ∧
  returns_to_initial_position metro.green_line 2016 :=
by
  sorry

/-- The metro system of city N -/
def city_n_metro : MetroSystem :=
  { red_line := { one_way_time := 7 }
  , blue_line := { one_way_time := 8 }
  , green_line := { one_way_time := 9 }
  }

/-- The main theorem proving that all trains in city N's metro return to their initial positions after 2016 minutes -/
theorem city_n_trains_return_after_2016_minutes :
  returns_to_initial_position city_n_metro.red_line 2016 ∧
  returns_to_initial_position city_n_metro.blue_line 2016 ∧
  returns_to_initial_position city_n_metro.green_line 2016 :=
by
  apply all_trains_return_to_initial_positions

end all_trains_return_to_initial_positions_city_n_trains_return_after_2016_minutes_l3321_332130


namespace youtube_views_multiple_l3321_332101

/-- The multiple by which views increased on the fourth day -/
def viewMultiple (initialViews : ℕ) (totalViews : ℕ) (additionalViews : ℕ) : ℚ :=
  (totalViews - additionalViews - initialViews) / initialViews

theorem youtube_views_multiple :
  let initialViews : ℕ := 4000
  let totalViews : ℕ := 94000
  let additionalViews : ℕ := 50000
  viewMultiple initialViews totalViews additionalViews = 11 := by
sorry

end youtube_views_multiple_l3321_332101


namespace tangent_product_special_angles_l3321_332160

theorem tangent_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 60 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end tangent_product_special_angles_l3321_332160


namespace a_minus_b_value_l3321_332171

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 5) (h3 : a + b > 0) :
  a - b = 3 ∨ a - b = 13 := by
sorry

end a_minus_b_value_l3321_332171


namespace prime_from_phi_and_omega_l3321_332113

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of prime divisors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- A number is prime if it has exactly two divisors -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_from_phi_and_omega (n : ℕ) 
  (h1 : phi n ∣ (n - 1)) 
  (h2 : omega n ≤ 3) : 
  is_prime n :=
sorry

end prime_from_phi_and_omega_l3321_332113


namespace defective_units_percentage_l3321_332194

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 5

/-- The percentage of all units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.4

/-- The percentage of all units that are defective -/
def defective_percent : ℝ := 8

theorem defective_units_percentage :
  defective_shipped_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end defective_units_percentage_l3321_332194


namespace square_sum_halving_l3321_332176

theorem square_sum_halving (a b : ℕ) (h : a^2 + b^2 = 18728) :
  ∃ (n m : ℕ), n^2 + m^2 = 9364 ∧ ((n = 30 ∧ m = 92) ∨ (n = 92 ∧ m = 30)) :=
by
  sorry

end square_sum_halving_l3321_332176


namespace lines_perp_to_plane_are_parallel_perp_line_to_parallel_planes_l3321_332136

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem lines_perp_to_plane_are_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel_lines m n := by sorry

-- Theorem 2: If two planes are parallel, and a line is perpendicular to one of them,
-- then it is perpendicular to the other
theorem perp_line_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β) (h2 : parallel_planes β γ) (h3 : perpendicular m α) :
  perpendicular m γ := by sorry

end lines_perp_to_plane_are_parallel_perp_line_to_parallel_planes_l3321_332136


namespace waiter_customers_problem_l3321_332148

theorem waiter_customers_problem :
  ∃ x : ℝ, x > 0 ∧ ((x - 19.0) - 14.0 = 3) → x = 36.0 := by
  sorry

end waiter_customers_problem_l3321_332148


namespace fish_tank_count_l3321_332120

theorem fish_tank_count : 
  ∀ (n : ℕ) (first_tank : ℕ) (other_tanks : ℕ),
    n = 3 →
    first_tank = 20 →
    other_tanks = 2 * first_tank →
    first_tank + (n - 1) * other_tanks = 100 := by
  sorry

end fish_tank_count_l3321_332120


namespace perimeter_of_modified_square_l3321_332107

/-- Given a square with perimeter 64 inches, cutting out an equilateral triangle
    with side length equal to the square's side and translating it to form a new figure
    results in a figure with perimeter 80 inches. -/
theorem perimeter_of_modified_square (square_perimeter : ℝ) (new_figure_perimeter : ℝ) :
  square_perimeter = 64 →
  new_figure_perimeter = square_perimeter + 2 * (square_perimeter / 4) - (square_perimeter / 4) →
  new_figure_perimeter = 80 :=
by sorry

end perimeter_of_modified_square_l3321_332107


namespace profit_achieved_l3321_332173

/-- Calculates the number of pens needed to be sold to achieve a specific profit --/
def pens_to_sell (num_purchased : ℕ) (purchase_price : ℚ) (sell_price : ℚ) (desired_profit : ℚ) : ℕ :=
  let total_cost := num_purchased * purchase_price
  let revenue_needed := total_cost + desired_profit
  (revenue_needed / sell_price).ceil.toNat

/-- Theorem stating that selling 1500 pens achieves the desired profit --/
theorem profit_achieved (num_purchased : ℕ) (purchase_price sell_price desired_profit : ℚ) :
  num_purchased = 2000 →
  purchase_price = 15/100 →
  sell_price = 30/100 →
  desired_profit = 150 →
  pens_to_sell num_purchased purchase_price sell_price desired_profit = 1500 := by
  sorry

end profit_achieved_l3321_332173


namespace power_function_m_value_l3321_332175

/-- A function y = (m^2 + 2m - 2)x^m is a power function and increasing in the first quadrant -/
def is_power_and_increasing (m : ℝ) : Prop :=
  (m^2 + 2*m - 2 = 1) ∧ (m > 0)

/-- If y = (m^2 + 2m - 2)x^m is a power function and increasing in the first quadrant, then m = 1 -/
theorem power_function_m_value :
  ∀ m : ℝ, is_power_and_increasing m → m = 1 := by
  sorry

end power_function_m_value_l3321_332175


namespace log_10_7_in_terms_of_r_s_l3321_332111

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_10_7_in_terms_of_r_s (r s : ℝ) 
  (h1 : log 4 2 = r) 
  (h2 : log 2 7 = s) : 
  log 10 7 = s / (1 + s) := by
  sorry

end log_10_7_in_terms_of_r_s_l3321_332111


namespace average_monthly_balance_l3321_332196

def monthly_balances : List ℝ := [100, 200, 250, 50, 300, 300]
def num_months : ℕ := 6

theorem average_monthly_balance :
  (monthly_balances.sum / num_months) = 200 := by sorry

end average_monthly_balance_l3321_332196


namespace product_of_h_at_roots_of_p_l3321_332116

theorem product_of_h_at_roots_of_p (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 + 1) * (y₂^2 + 1) * (y₃^2 + 1) * (y₄^2 + 1) * (y₅^2 + 1) = Complex.I) :=
by sorry

end product_of_h_at_roots_of_p_l3321_332116


namespace complement_A_intersect_B_when_m_zero_A_subset_B_iff_m_leq_neg_three_l3321_332122

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < 5}

-- Theorem for part (1)
theorem complement_A_intersect_B_when_m_zero :
  (Set.univ \ A) ∩ B 0 = Set.Icc 2 5 := by sorry

-- Theorem for part (2)
theorem A_subset_B_iff_m_leq_neg_three (m : ℝ) :
  A ⊆ B m ↔ m ≤ -3 := by sorry

end complement_A_intersect_B_when_m_zero_A_subset_B_iff_m_leq_neg_three_l3321_332122


namespace simplify_nested_roots_l3321_332162

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/8))^(1/4))^3 * (((a^16)^(1/4))^(1/8))^3 = a^3 := by
  sorry

end simplify_nested_roots_l3321_332162


namespace ten_sparklers_to_crackers_five_ornaments_one_cracker_more_valuable_l3321_332115

-- Define the exchange rates
def ornament_to_cracker : ℚ := 2
def sparkler_to_garland : ℚ := 2/5
def ornament_to_garland : ℚ := 1/4

-- Define the conversion function
def convert (item : String) (quantity : ℚ) : ℚ :=
  match item with
  | "sparkler" => quantity * sparkler_to_garland * (1 / ornament_to_garland) * ornament_to_cracker
  | "ornament" => quantity * ornament_to_cracker
  | _ => 0

-- Theorem for part (a)
theorem ten_sparklers_to_crackers :
  convert "sparkler" 10 = 32 := by sorry

-- Theorem for part (b)
theorem five_ornaments_one_cracker_more_valuable :
  convert "ornament" 5 + 1 > convert "sparkler" 2 := by sorry

end ten_sparklers_to_crackers_five_ornaments_one_cracker_more_valuable_l3321_332115


namespace physics_marks_calculation_l3321_332137

def english_marks : ℕ := 74
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def average_marks : ℚ := 75.6
def num_subjects : ℕ := 5

theorem physics_marks_calculation :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 82 :=
by sorry

end physics_marks_calculation_l3321_332137


namespace trapezoid_area_sum_l3321_332197

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Calculates the area of a trapezoid using Heron's formula -/
def area (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is a square-free positive integer -/
def isSquareFree (n : ℕ) : Prop := sorry

/-- Theorem: The sum of areas of all possible trapezoids with sides 4, 6, 8, and 10
    can be expressed as r₁√n₁ + r₂√n₂ + r₃, where r₁, r₂, r₃ are rational,
    n₁, n₂ are distinct square-free positive integers, and r₁ + r₂ + r₃ + n₁ + n₂ = 80 -/
theorem trapezoid_area_sum :
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    let t₁ : Trapezoid := ⟨4, 6, 8, 10⟩
    let t₂ : Trapezoid := ⟨6, 10, 4, 8⟩
    isSquareFree n₁ ∧ isSquareFree n₂ ∧ n₁ ≠ n₂ ∧
    area t₁ + area t₂ = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    r₁ + r₂ + r₃ + n₁ + n₂ = 80 := by
  sorry

end trapezoid_area_sum_l3321_332197


namespace similar_triangle_longest_side_l3321_332140

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 10) 
  (hc : c = 12) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = 150) 
  (h_similar : ∃ k : ℝ, k > 0 ∧ perimeter = k * (a + b + c)) :
  ∃ longest_side : ℝ, longest_side = 60 ∧ 
    longest_side = max (k * a) (max (k * b) (k * c)) :=
by sorry


end similar_triangle_longest_side_l3321_332140


namespace wx_length_is_25_l3321_332110

/-- A quadrilateral with two right angles and specific side lengths -/
structure RightQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  right_angle_X : (X.1 - W.1) * (Y.1 - X.1) + (X.2 - W.2) * (Y.2 - X.2) = 0
  right_angle_Y : (Y.1 - X.1) * (Z.1 - Y.1) + (Y.2 - X.2) * (Z.2 - Y.2) = 0
  wz_length : Real.sqrt ((W.1 - Z.1)^2 + (W.2 - Z.2)^2) = 7
  xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 14
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 24

/-- The length of WX in the given quadrilateral is 25 -/
theorem wx_length_is_25 (q : RightQuadrilateral) :
  Real.sqrt ((q.W.1 - q.X.1)^2 + (q.W.2 - q.X.2)^2) = 25 := by
  sorry

end wx_length_is_25_l3321_332110


namespace red_balls_count_l3321_332105

theorem red_balls_count (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  white_balls = 4 →
  (red_balls : ℚ) / total_balls = 6 / 10 →
  red_balls = 6 := by
  sorry

end red_balls_count_l3321_332105


namespace complex_equation_solution_l3321_332100

theorem complex_equation_solution (x y : ℝ) (h : (x : ℂ) / (1 + Complex.I) = 1 - y * Complex.I) :
  (x : ℂ) + y * Complex.I = 2 + Complex.I := by
  sorry

end complex_equation_solution_l3321_332100


namespace number_and_percentage_problem_l3321_332123

theorem number_and_percentage_problem (N P : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 25 ∧ 
  (P/100 : ℝ) * N = 300 →
  N = 750 ∧ P = 40 := by
  sorry

end number_and_percentage_problem_l3321_332123


namespace tan_beta_minus_2alpha_l3321_332183

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -1 / 3) : 
  Real.tan (β - 2 * α) = -1 / 7 := by
  sorry

end tan_beta_minus_2alpha_l3321_332183


namespace calculate_3Z5_l3321_332108

-- Define the Z operation
def Z (a b : ℝ) : ℝ := b + 15 * a - a^3

-- Theorem statement
theorem calculate_3Z5 : Z 3 5 = 23 := by
  sorry

end calculate_3Z5_l3321_332108


namespace intersection_of_A_and_B_l3321_332145

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l3321_332145


namespace floor_times_x_equals_48_l3321_332152

theorem floor_times_x_equals_48 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 48 ∧ x = 8 := by
  sorry

end floor_times_x_equals_48_l3321_332152


namespace complex_magnitude_squared_l3321_332102

theorem complex_magnitude_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (z - Complex.abs z = 4 - 6*I) → Complex.normSq z = 42.25 := by
  sorry

end complex_magnitude_squared_l3321_332102


namespace problem_solution_l3321_332112

theorem problem_solution (x y : ℝ) : x / y = 12 / 3 → y = 27 → x = 108 := by
  sorry

end problem_solution_l3321_332112


namespace quadratic_root_sum_product_l3321_332172

theorem quadratic_root_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 + 3*x₁ - 2023 = 0 →
  x₂^2 + 3*x₂ - 2023 = 0 →
  x₁^2 * x₂ + x₁ * x₂^2 = 6069 :=
by sorry

end quadratic_root_sum_product_l3321_332172


namespace negation_of_exists_negation_of_proposition_l3321_332191

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 0, 2 * x + 3 ≤ 0) ↔ (∀ x > 0, 2 * x + 3 > 0) :=
by sorry

end negation_of_exists_negation_of_proposition_l3321_332191


namespace amoeba_count_after_ten_days_l3321_332142

/-- The number of amoebas after n days, given an initial population of 1 and a tripling growth rate each day. -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of amoebas after 10 days is equal to 3^10. -/
theorem amoeba_count_after_ten_days : amoeba_count 10 = 3^10 := by
  sorry

end amoeba_count_after_ten_days_l3321_332142


namespace cube_root_equation_solution_l3321_332198

theorem cube_root_equation_solution (x : ℝ) :
  (7 * x * (x^2)^(1/2))^(1/3) = 5 → x = 5 * (35^(1/2)) / 7 ∨ x = -5 * (35^(1/2)) / 7 := by
  sorry

end cube_root_equation_solution_l3321_332198


namespace parallel_vectors_k_value_l3321_332150

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value (k : ℝ) :
  vector_parallel (1, k) (2, 2) → k = 1 := by
  sorry

end parallel_vectors_k_value_l3321_332150


namespace negation_of_universal_proposition_l3321_332127

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) := by sorry

end negation_of_universal_proposition_l3321_332127


namespace total_money_proof_l3321_332190

/-- The amount of money Beth currently has -/
def beth_money : ℕ := 70

/-- The amount of money Jan currently has -/
def jan_money : ℕ := 80

/-- The amount of money Tom currently has -/
def tom_money : ℕ := 210

theorem total_money_proof :
  (beth_money + 35 = 105) ∧
  (jan_money - 10 = beth_money) ∧
  (tom_money = 3 * (jan_money - 10)) →
  beth_money + jan_money + tom_money = 360 := by
sorry

end total_money_proof_l3321_332190


namespace hyperbola_asymptotes_l3321_332104

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_rel : a = Real.sqrt 2 * b

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x}

/-- Theorem: The asymptotes of the given hyperbola are y = ±√2x -/
theorem hyperbola_asymptotes (h : Hyperbola) : 
  asymptote_equation h = {(x, y) | y^2 / h.a^2 - x^2 / h.b^2 = 1} := by
  sorry

end hyperbola_asymptotes_l3321_332104


namespace two_axes_implies_center_symmetry_l3321_332192

/-- A geometric figure in a 2D plane. -/
structure Figure where
  -- The implementation details of the figure are abstracted away

/-- An axis of symmetry for a figure. -/
structure AxisOfSymmetry where
  -- The implementation details of the axis are abstracted away

/-- A center of symmetry for a figure. -/
structure CenterOfSymmetry where
  -- The implementation details of the center are abstracted away

/-- Predicate to check if a figure has an axis of symmetry. -/
def hasAxisOfSymmetry (f : Figure) (a : AxisOfSymmetry) : Prop :=
  sorry

/-- Predicate to check if a figure has a center of symmetry. -/
def hasCenterOfSymmetry (f : Figure) (c : CenterOfSymmetry) : Prop :=
  sorry

/-- Theorem: If a figure has exactly two axes of symmetry, it must have a center of symmetry. -/
theorem two_axes_implies_center_symmetry (f : Figure) (a1 a2 : AxisOfSymmetry) :
  (hasAxisOfSymmetry f a1) ∧ 
  (hasAxisOfSymmetry f a2) ∧ 
  (a1 ≠ a2) ∧
  (∀ a : AxisOfSymmetry, hasAxisOfSymmetry f a → (a = a1 ∨ a = a2)) →
  ∃ c : CenterOfSymmetry, hasCenterOfSymmetry f c :=
sorry

end two_axes_implies_center_symmetry_l3321_332192


namespace rounding_and_scientific_notation_l3321_332186

-- Define rounding to significant figures
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define rounding to decimal places
def roundToDecimalPlaces (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define scientific notation
def scientificNotation (x : ℝ) : ℝ × ℤ := sorry

theorem rounding_and_scientific_notation :
  (roundToSignificantFigures 12.349 2 = 12) ∧
  (roundToDecimalPlaces 0.12349 3 = 0.123) ∧
  (scientificNotation 201200 = (2.012, 5)) ∧
  (scientificNotation 0.0002012 = (2.012, -4)) := by sorry

end rounding_and_scientific_notation_l3321_332186


namespace darry_climbed_152_steps_l3321_332126

/-- The number of steps Darry climbed today -/
def total_steps : ℕ :=
  let full_ladder_steps : ℕ := 11
  let full_ladder_climbs : ℕ := 10
  let small_ladder_steps : ℕ := 6
  let small_ladder_climbs : ℕ := 7
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

theorem darry_climbed_152_steps : total_steps = 152 := by
  sorry

end darry_climbed_152_steps_l3321_332126


namespace base_five_of_156_l3321_332168

def base_five_equiv (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_five_of_156 :
  base_five_equiv 156 = [1, 1, 1, 1] := by
  sorry

end base_five_of_156_l3321_332168


namespace ticket_cost_proof_l3321_332164

def initial_amount : ℕ := 760
def remaining_amount : ℕ := 310
def ticket_cost : ℕ := 300

theorem ticket_cost_proof :
  (initial_amount - remaining_amount = ticket_cost + ticket_cost / 2) →
  ticket_cost = 300 := by
  sorry

end ticket_cost_proof_l3321_332164


namespace candy_mixture_cost_l3321_332188

/-- The cost per pound of the first candy -/
def first_candy_cost : ℝ := 8

/-- The weight of the first candy in pounds -/
def first_candy_weight : ℝ := 30

/-- The cost per pound of the second candy -/
def second_candy_cost : ℝ := 5

/-- The weight of the second candy in pounds -/
def second_candy_weight : ℝ := 60

/-- The cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := first_candy_weight + second_candy_weight

theorem candy_mixture_cost :
  first_candy_cost * first_candy_weight + second_candy_cost * second_candy_weight =
  mixture_cost * total_weight :=
by sorry

end candy_mixture_cost_l3321_332188


namespace convergence_bound_l3321_332167

def v : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => (3/2) * v n - (3/2) * (v n)^2

def M : ℚ := 1/2

theorem convergence_bound (k : ℕ) : k = 5 ↔ (∀ j < k, |v j - M| > 1/2^20) ∧ |v k - M| ≤ 1/2^20 := by
  sorry

end convergence_bound_l3321_332167


namespace jenga_blocks_removed_l3321_332134

def blocks_removed (num_players : ℕ) (num_rounds : ℕ) : ℕ :=
  (num_players * num_rounds * (num_rounds + 1)) / 2

def blocks_removed_sixth_round (num_players : ℕ) (num_rounds : ℕ) : ℕ :=
  blocks_removed num_players num_rounds + (num_rounds + 1)

theorem jenga_blocks_removed : 
  let num_players : ℕ := 5
  let num_rounds : ℕ := 5
  blocks_removed_sixth_round num_players num_rounds = 81 := by
  sorry

end jenga_blocks_removed_l3321_332134


namespace intersection_distance_l3321_332199

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 :=
sorry

end intersection_distance_l3321_332199


namespace train_distance_in_three_hours_l3321_332189

-- Define the train's speed
def train_speed : ℚ := 1 / 2

-- Define the duration in hours
def duration : ℚ := 3

-- Define the number of minutes in an hour
def minutes_per_hour : ℚ := 60

-- Theorem statement
theorem train_distance_in_three_hours :
  train_speed * minutes_per_hour * duration = 90 := by
  sorry


end train_distance_in_three_hours_l3321_332189


namespace songs_added_l3321_332121

theorem songs_added (initial_songs deleted_songs final_songs : ℕ) : 
  initial_songs = 8 → deleted_songs = 5 → final_songs = 33 →
  final_songs - (initial_songs - deleted_songs) = 30 :=
by sorry

end songs_added_l3321_332121


namespace original_number_proof_l3321_332133

theorem original_number_proof (x : ℝ) (h : 1.40 * x = 700) : x = 500 := by
  sorry

end original_number_proof_l3321_332133


namespace age_difference_l3321_332184

theorem age_difference (matt_age john_age : ℕ) 
  (h1 : matt_age + john_age = 52)
  (h2 : ∃ k : ℕ, matt_age + k = 4 * john_age) : 
  4 * john_age - matt_age = 3 := by
sorry

end age_difference_l3321_332184


namespace problem_statement_l3321_332147

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 5)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 1 / 4 := by
sorry

end problem_statement_l3321_332147


namespace quadratic_factorization_l3321_332157

theorem quadratic_factorization (a x y : ℝ) : a * x^2 + 2*a*x*y + a * y^2 = a * (x + y)^2 := by
  sorry

end quadratic_factorization_l3321_332157


namespace circle_inequality_l3321_332170

theorem circle_inequality (r s d : ℝ) (h1 : r > s) (h2 : r > 0) (h3 : s > 0) (h4 : d > 0) :
  r - s ≤ d :=
sorry

end circle_inequality_l3321_332170


namespace unique_prime_double_squares_l3321_332129

theorem unique_prime_double_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x y : ℕ), p + 7 = 2 * x^2 ∧ p^2 + 7 = 2 * y^2) ∧ 
    p = 11 := by
  sorry

end unique_prime_double_squares_l3321_332129


namespace sum_of_mixed_numbers_l3321_332169

theorem sum_of_mixed_numbers : 
  (481 + 1/6 : ℚ) + (265 + 1/12 : ℚ) + (904 + 1/20 : ℚ) - 
  (184 + 29/30 : ℚ) - (160 + 41/42 : ℚ) - (703 + 55/56 : ℚ) = 
  603 + 3/8 := by sorry

end sum_of_mixed_numbers_l3321_332169


namespace total_movies_is_nineteen_l3321_332159

/-- The number of movies shown on each screen in a movie theater --/
def movies_per_screen : List Nat := [3, 4, 2, 3, 5, 2]

/-- The total number of movies shown in the theater --/
def total_movies : Nat := movies_per_screen.sum

/-- Theorem stating that the total number of movies shown is 19 --/
theorem total_movies_is_nineteen : total_movies = 19 := by
  sorry

end total_movies_is_nineteen_l3321_332159


namespace original_price_calculation_l3321_332181

/-- Given an item sold at a 20% loss with a selling price of 480, prove that the original price was 600. -/
theorem original_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 480 → 
  loss_percentage = 20 → 
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 - loss_percentage / 100) ∧ 
    original_price = 600 := by
  sorry

end original_price_calculation_l3321_332181


namespace cereal_eating_time_l3321_332131

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate thin_rate total_amount : ℚ) : ℚ :=
  total_amount / (fat_rate + thin_rate)

theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let total_amount : ℚ := 4   -- Total amount of cereal in pounds
  time_to_eat_together fat_rate thin_rate total_amount = 75 / 2 := by
  sorry

#eval (75 : ℚ) / 2 -- Should output 37.5

end cereal_eating_time_l3321_332131


namespace nine_sided_polygon_diagonals_l3321_332154

-- Define a convex polygon
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool

-- Define the number of right angles in the polygon
def right_angles (p : ConvexPolygon) : ℕ := 2

-- Define the function to calculate the number of diagonals
def num_diagonals (p : ConvexPolygon) : ℕ :=
  p.sides * (p.sides - 3) / 2

-- Theorem statement
theorem nine_sided_polygon_diagonals (p : ConvexPolygon) :
  p.sides = 9 → p.is_convex = true → right_angles p = 2 → num_diagonals p = 27 := by
  sorry

end nine_sided_polygon_diagonals_l3321_332154


namespace smallest_value_of_fraction_sum_l3321_332139

theorem smallest_value_of_fraction_sum (a b : ℤ) (h : a > b) :
  (((a - b : ℚ) / (a + b)) + ((a + b : ℚ) / (a - b))) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' > b' ∧ (((a' - b' : ℚ) / (a' + b')) + ((a' + b' : ℚ) / (a' - b'))) = 2 :=
by sorry

end smallest_value_of_fraction_sum_l3321_332139


namespace min_cost_water_tank_l3321_332106

/-- Represents the dimensions and cost of a rectangular water tank. -/
structure WaterTank where
  length : ℝ
  width : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank. -/
def totalCost (tank : WaterTank) : ℝ :=
  tank.bottomCost * tank.length * tank.width +
  tank.wallCost * 2 * (tank.length + tank.width) * tank.depth

/-- Theorem stating the minimum cost configuration for the water tank. -/
theorem min_cost_water_tank :
  ∃ (tank : WaterTank),
    tank.depth = 3 ∧
    tank.length * tank.width * tank.depth = 48 ∧
    tank.bottomCost = 40 ∧
    tank.wallCost = 20 ∧
    tank.length = 4 ∧
    tank.width = 4 ∧
    totalCost tank = 1600 ∧
    (∀ (other : WaterTank),
      other.depth = 3 →
      other.length * other.width * other.depth = 48 →
      other.bottomCost = 40 →
      other.wallCost = 20 →
      totalCost other ≥ totalCost tank) := by
  sorry

end min_cost_water_tank_l3321_332106


namespace minimum_raft_capacity_l3321_332180

/-- Represents an animal with a specific weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with a weight capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if a raft can carry at least two mice -/
def canCarryTwoMice (r : Raft) (mouseWeight : ℕ) : Prop :=
  r.capacity ≥ 2 * mouseWeight

/-- Checks if all animals can be transported given a raft capacity -/
def canTransportAll (r : Raft) (mice moles hamsters : List Animal) : Prop :=
  (mice ++ moles ++ hamsters).all (fun a => a.weight ≤ r.capacity)

theorem minimum_raft_capacity
  (mice : List Animal)
  (moles : List Animal)
  (hamsters : List Animal)
  (h_mice_count : mice.length = 5)
  (h_moles_count : moles.length = 3)
  (h_hamsters_count : hamsters.length = 4)
  (h_mice_weight : ∀ m ∈ mice, m.weight = 70)
  (h_moles_weight : ∀ m ∈ moles, m.weight = 90)
  (h_hamsters_weight : ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ 
    canCarryTwoMice r 70 ∧
    canTransportAll r mice moles hamsters :=
  sorry

#check minimum_raft_capacity

end minimum_raft_capacity_l3321_332180


namespace calculate_principal_amount_l3321_332156

/-- Given simple interest, time period, and interest rate, calculate the principal amount -/
theorem calculate_principal_amount (simple_interest rate : ℚ) (time : ℕ) : 
  simple_interest = 200 → 
  time = 4 → 
  rate = 3125 / 100000 → 
  simple_interest = (1600 : ℚ) * rate * (time : ℚ) := by
  sorry

#check calculate_principal_amount

end calculate_principal_amount_l3321_332156


namespace vlad_score_in_competition_l3321_332155

/-- A video game competition between two players -/
structure VideoGameCompetition where
  rounds : ℕ
  points_per_win : ℕ
  taro_score : ℕ

/-- Calculate Vlad's score in the video game competition -/
def vlad_score (game : VideoGameCompetition) : ℕ :=
  game.rounds * game.points_per_win - game.taro_score

/-- Theorem stating Vlad's score in the specific competition described in the problem -/
theorem vlad_score_in_competition :
  let game : VideoGameCompetition := {
    rounds := 30,
    points_per_win := 5,
    taro_score := 3 * (30 * 5) / 5 - 4
  }
  vlad_score game = 64 := by sorry

end vlad_score_in_competition_l3321_332155


namespace express_y_in_terms_of_x_l3321_332138

theorem express_y_in_terms_of_x (n : ℕ) (x y : ℝ) : 
  x = 3^n → y = 2 + 9^n → y = 2 + x^2 := by
sorry

end express_y_in_terms_of_x_l3321_332138


namespace binomial_30_3_l3321_332141

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l3321_332141


namespace quadratic_inequality_solution_l3321_332118

-- Define the quadratic function
def f (x : ℝ) := x^2 - 3*x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | f x < 0}

-- State the theorem
theorem quadratic_inequality_solution :
  solution_set = Set.Ioo 1 2 := by sorry

end quadratic_inequality_solution_l3321_332118


namespace taxi_truck_speed_ratio_l3321_332135

/-- Given a truck that travels 2.1 km in 1 minute and a taxi that travels 10.5 km in 4 minutes,
    prove that the taxi is 1.25 times faster than the truck. -/
theorem taxi_truck_speed_ratio :
  let truck_speed := 2.1 -- km per minute
  let taxi_speed := 10.5 / 4 -- km per minute
  taxi_speed / truck_speed = 1.25 := by sorry

end taxi_truck_speed_ratio_l3321_332135


namespace job_completion_time_l3321_332185

theorem job_completion_time (y : ℝ) 
  (h1 : (1 : ℝ) / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) : y = 3/2 := by
  sorry

end job_completion_time_l3321_332185


namespace license_plate_count_l3321_332146

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet (including Y) -/
def num_consonants : ℕ := 21

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_consonants * num_vowels * num_vowels * num_digits

theorem license_plate_count :
  total_license_plates = 110250 := by
  sorry

end license_plate_count_l3321_332146


namespace geometric_sequence_product_l3321_332161

/-- A geometric sequence with first term 1 and common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => q * geometric_sequence q n

theorem geometric_sequence_product (q : ℝ) (h : q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1) :
  ∃ m : ℕ, geometric_sequence q (m - 1) = (geometric_sequence q 0) *
    (geometric_sequence q 1) * (geometric_sequence q 2) *
    (geometric_sequence q 3) * (geometric_sequence q 4) ∧ m = 11 := by
  sorry

end geometric_sequence_product_l3321_332161


namespace fraction_of_cats_l3321_332182

theorem fraction_of_cats (total_animals : ℕ) (total_dog_legs : ℕ) : 
  total_animals = 300 →
  total_dog_legs = 400 →
  (2 : ℚ) / 3 = (total_animals - (total_dog_legs / 4)) / total_animals :=
by sorry

end fraction_of_cats_l3321_332182


namespace triangle_theorem_l3321_332149

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 2 * t.b * Real.sin t.C + t.a * Real.sin t.A = t.b * Real.sin t.B + t.c * Real.sin t.C

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h1 : condition t) (h2 : t.a = Real.sqrt 2) :
  t.A = π / 4 ∧ 
  ∀ (AD : ℝ), AD ≤ 1 + Real.sqrt 2 / 2 :=
sorry

end triangle_theorem_l3321_332149


namespace incorrect_complex_analogy_l3321_332144

def complex_square_property (z : ℂ) : Prop :=
  Complex.abs z ^ 2 = z ^ 2

theorem incorrect_complex_analogy :
  ∃ z : ℂ, ¬(complex_square_property z) :=
sorry

end incorrect_complex_analogy_l3321_332144


namespace negation_of_some_primes_even_l3321_332128

theorem negation_of_some_primes_even :
  (¬ ∃ p, Nat.Prime p ∧ Even p) ↔ (∀ p, Nat.Prime p → Odd p) :=
by sorry

end negation_of_some_primes_even_l3321_332128


namespace problem_statement_l3321_332178

theorem problem_statement :
  (∃ x : ℝ, x^2 + 1 ≤ 2*x) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt ((x^2 + y^2)/2) ≥ (2*x*y)/(x + y)) ∧
  ¬(∀ x : ℝ, x ≠ 0 → x + 1/x ≥ 2) :=
by sorry

end problem_statement_l3321_332178


namespace coordinates_wrt_origin_specific_point_coordinates_l3321_332153

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point2D := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin are the same as its coordinates -/
theorem coordinates_wrt_origin (P : Point2D) : 
  P.x = P.x - origin.x ∧ P.y = P.y - origin.y :=
by sorry

/-- For the specific point P(-2, -4), its coordinates with respect to the origin are (-2, -4) -/
theorem specific_point_coordinates : 
  let P : Point2D := ⟨-2, -4⟩
  P.x - origin.x = -2 ∧ P.y - origin.y = -4 :=
by sorry

end coordinates_wrt_origin_specific_point_coordinates_l3321_332153


namespace circle_area_radius_increase_l3321_332103

theorem circle_area_radius_increase : 
  ∀ (r : ℝ) (r' : ℝ), r > 0 → r' > 0 → 
  (π * r' ^ 2 = 4 * π * r ^ 2) → 
  (r' = 2 * r) := by
sorry

end circle_area_radius_increase_l3321_332103


namespace union_P_S_when_m_2_S_subset_P_iff_m_in_zero_one_l3321_332193

-- Define the sets P and S
def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Part 1: P ∪ S when m = 2
theorem union_P_S_when_m_2 : 
  P ∪ S 2 = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Part 2: S ⊆ P iff m ∈ [0, 1]
theorem S_subset_P_iff_m_in_zero_one (m : ℝ) : 
  S m ⊆ P ↔ 0 ≤ m ∧ m ≤ 1 := by sorry

end union_P_S_when_m_2_S_subset_P_iff_m_in_zero_one_l3321_332193


namespace proposition_equivalence_l3321_332195

theorem proposition_equivalence (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 1 :=
by sorry

end proposition_equivalence_l3321_332195


namespace product_of_sum_and_difference_l3321_332117

theorem product_of_sum_and_difference (x y : ℝ) (h1 : x > y) (h2 : x + y = 20) (h3 : x - y = 4) :
  (3 * x) * y = 288 := by
  sorry

end product_of_sum_and_difference_l3321_332117


namespace line_symmetry_l3321_332179

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Symmetry condition for two lines with respect to y = x -/
def symmetric_about_y_eq_x (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = 1 ∧ l1.intercept + l2.intercept = 0

theorem line_symmetry (a b : ℝ) :
  let l1 : Line := ⟨a, 2⟩
  let l2 : Line := ⟨3, -b⟩
  symmetric_about_y_eq_x l1 l2 → a = 1/3 ∧ b = 6 := by
  sorry

#check line_symmetry

end line_symmetry_l3321_332179


namespace linear_function_composition_l3321_332163

/-- A linear function f: ℝ → ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b ∧ a ≠ 0

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x : ℝ, f (f x) = x - 2) → ∀ x : ℝ, f x = x - 1 := by
  sorry

end linear_function_composition_l3321_332163


namespace profit_and_marginal_profit_max_not_equal_l3321_332132

/-- Revenue function --/
def R (x : ℕ+) : ℚ := 3000 * x - 20 * x^2

/-- Cost function --/
def C (x : ℕ+) : ℚ := 500 * x + 4000

/-- Profit function --/
def P (x : ℕ+) : ℚ := R x - C x

/-- Marginal profit function --/
def MP (x : ℕ+) : ℚ := P (x + 1) - P x

/-- The maximum allowed production --/
def max_production : ℕ+ := 100

theorem profit_and_marginal_profit_max_not_equal :
  (∃ x : ℕ+, x ≤ max_production ∧ ∀ y : ℕ+, y ≤ max_production → P y ≤ P x) ≠
  (∃ x : ℕ+, x ≤ max_production ∧ ∀ y : ℕ+, y ≤ max_production → MP y ≤ MP x) :=
by sorry

end profit_and_marginal_profit_max_not_equal_l3321_332132


namespace sum_a_d_equals_two_l3321_332165

theorem sum_a_d_equals_two (a b c d : ℝ) 
  (h1 : a + b = 4)
  (h2 : b + c = 7)
  (h3 : c + d = 5) :
  a + d = 2 := by
sorry

end sum_a_d_equals_two_l3321_332165


namespace max_area_rectangle_perimeter_40_l3321_332158

/-- The maximum area of a rectangle with perimeter 40 is 100 -/
theorem max_area_rectangle_perimeter_40 :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 40 →
  x * y ≤ 100 := by
sorry

end max_area_rectangle_perimeter_40_l3321_332158


namespace arithmetic_sequence_problem_l3321_332119

theorem arithmetic_sequence_problem (x : ℚ) : 
  let a₁ : ℚ := 1/3
  let a₂ : ℚ := x - 2
  let a₃ : ℚ := 4*x
  (a₂ - a₁ = a₃ - a₂) → x = -13/6 := by
sorry

end arithmetic_sequence_problem_l3321_332119


namespace pure_imaginary_complex_number_l3321_332151

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, a - (10 : ℂ) / (3 - Complex.I) = b * Complex.I) → a = 3 := by
  sorry

end pure_imaginary_complex_number_l3321_332151


namespace select_blocks_count_l3321_332143

/-- The number of ways to select 4 blocks from a 6x6 grid such that no two blocks are in the same row or column -/
def select_blocks : ℕ :=
  Nat.choose 6 4 * Nat.choose 6 4 * Nat.factorial 4

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    such that no two blocks are in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end select_blocks_count_l3321_332143


namespace abs_sine_period_l3321_332174

-- Define the sine function and its period
noncomputable def sine_period : ℝ := 2 * Real.pi

-- Define the property that sine has this period
axiom sine_periodic (x : ℝ) : Real.sin (x + sine_period) = Real.sin x

-- Define the property that taking absolute value halves the period
axiom abs_halves_period {f : ℝ → ℝ} {p : ℝ} (h : ∀ x, f (x + p) = f x) :
  ∀ x, |f (x + p/2)| = |f x|

-- State the theorem
theorem abs_sine_period : 
  ∃ p : ℝ, p > 0 ∧ p = Real.pi ∧ ∀ x, |Real.sin (x + p)| = |Real.sin x| ∧
  ∀ q, q > 0 → (∀ x, |Real.sin (x + q)| = |Real.sin x|) → p ≤ q :=
sorry

end abs_sine_period_l3321_332174


namespace find_set_M_l3321_332109

def U : Set Nat := {0, 1, 2, 3}

theorem find_set_M (M : Set Nat) (h : Set.compl M = {2}) : M = {0, 1, 3} := by
  sorry

end find_set_M_l3321_332109


namespace number_of_friends_l3321_332166

/-- Given that Sam, Dan, Tom, and Keith each have 14 Pokemon cards, 
    prove that the number of friends is 4. -/
theorem number_of_friends : ℕ := by
  sorry

end number_of_friends_l3321_332166


namespace correct_selection_ways_l3321_332124

/-- Represents the selection of athletes for a commendation meeting. -/
structure AthletesSelection where
  totalMales : Nat
  totalFemales : Nat
  maleCaptain : Nat
  femaleCaptain : Nat
  selectionSize : Nat

/-- Calculates the number of ways to select athletes under different conditions. -/
def selectionWays (s : AthletesSelection) : Nat × Nat × Nat × Nat :=
  let totalAthletes := s.totalMales + s.totalFemales
  let totalCaptains := s.maleCaptain + s.femaleCaptain
  let nonCaptains := totalAthletes - totalCaptains
  (
    Nat.choose s.totalMales 3 * Nat.choose s.totalFemales 2,
    Nat.choose totalCaptains 1 * Nat.choose nonCaptains 4 + Nat.choose totalCaptains 2 * Nat.choose nonCaptains 3,
    Nat.choose totalAthletes s.selectionSize - Nat.choose s.totalMales s.selectionSize,
    Nat.choose totalAthletes s.selectionSize - Nat.choose nonCaptains s.selectionSize - Nat.choose (s.totalMales - 1) (s.selectionSize - 1)
  )

/-- Theorem stating the correct number of ways to select athletes under different conditions. -/
theorem correct_selection_ways (s : AthletesSelection) 
  (h1 : s.totalMales = 6)
  (h2 : s.totalFemales = 4)
  (h3 : s.maleCaptain = 1)
  (h4 : s.femaleCaptain = 1)
  (h5 : s.selectionSize = 5) :
  selectionWays s = (120, 196, 246, 191) := by
  sorry

end correct_selection_ways_l3321_332124
