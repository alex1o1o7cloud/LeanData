import Mathlib

namespace rectangle_side_difference_l415_41580

theorem rectangle_side_difference (p d : ℝ) (hp : p > 0) (hd : d > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ 2 * (a + b) = p ∧ a^2 + b^2 = d^2 ∧ a - b = (Real.sqrt (8 * d^2 - p^2)) / 2 :=
sorry

end rectangle_side_difference_l415_41580


namespace sum_largest_smallest_prime_factors_546_l415_41556

def largest_prime_factor (n : ℕ) : ℕ := sorry

def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem sum_largest_smallest_prime_factors_546 :
  largest_prime_factor 546 + smallest_prime_factor 546 = 15 := by sorry

end sum_largest_smallest_prime_factors_546_l415_41556


namespace f_odd_and_decreasing_l415_41579

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x ≤ y → f y ≤ f x) := by
  sorry

end f_odd_and_decreasing_l415_41579


namespace race_distance_l415_41517

/-- The race problem -/
theorem race_distance (a_time b_time : ℕ) (beat_distance : ℕ) (total_distance : ℕ) : 
  a_time = 20 →
  b_time = 25 →
  beat_distance = 26 →
  (total_distance : ℚ) / a_time * b_time = total_distance + beat_distance →
  total_distance = 104 := by
  sorry

#check race_distance

end race_distance_l415_41517


namespace resort_tips_multiple_l415_41564

theorem resort_tips_multiple (total_months : ℕ) (august_ratio : ℝ) : 
  total_months = 7 → 
  august_ratio = 0.25 → 
  (7 * august_ratio) / (1 - august_ratio) = 1.75 := by
sorry

end resort_tips_multiple_l415_41564


namespace gas_tank_cost_l415_41505

theorem gas_tank_cost (initial_fullness : ℚ) (after_adding_fullness : ℚ) 
  (added_amount : ℚ) (gas_price : ℚ) : 
  initial_fullness = 1/8 →
  after_adding_fullness = 3/4 →
  added_amount = 30 →
  gas_price = 138/100 →
  (1 - after_adding_fullness) * 
    (added_amount / (after_adding_fullness - initial_fullness)) * 
    gas_price = 1656/100 := by
  sorry

#eval (1 : ℚ) - 3/4  -- Expected: 1/4
#eval 30 / (3/4 - 1/8)  -- Expected: 48
#eval 1/4 * 48  -- Expected: 12
#eval 12 * 138/100  -- Expected: 16.56

end gas_tank_cost_l415_41505


namespace particle_probability_l415_41582

def probability (x y : ℕ) : ℚ :=
  sorry

theorem particle_probability :
  let start_x : ℕ := 5
  let start_y : ℕ := 5
  probability start_x start_y = 1 / 243 :=
by
  sorry

axiom probability_recursive (x y : ℕ) :
  x > 0 → y > 0 →
  probability x y = (1/3) * probability (x-1) y + 
                    (1/3) * probability x (y-1) + 
                    (1/3) * probability (x-1) (y-1)

axiom probability_boundary_zero (x y : ℕ) :
  (x = 0 ∧ y > 0) ∨ (x > 0 ∧ y = 0) →
  probability x y = 0

axiom probability_origin :
  probability 0 0 = 1

end particle_probability_l415_41582


namespace least_n_multiple_of_1000_l415_41558

theorem least_n_multiple_of_1000 : ∃ (n : ℕ), n > 0 ∧ n = 797 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(1000 ∣ (2^m + 5^m - m))) ∧ 
  (1000 ∣ (2^n + 5^n - n)) := by
  sorry

end least_n_multiple_of_1000_l415_41558


namespace permutation_square_diff_l415_41567

theorem permutation_square_diff (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, n = 2 * k + 1) :
  (∃ a : Fin (n / 2 + 1) → Fin (n / 2 + 1),
    Function.Bijective a ∧
    ∀ i : Fin (n / 2), ∃ d : ℕ, ∀ j : Fin (n / 2),
      (a (j + 1))^2 - (a j)^2 ≡ d [ZMOD n]) →
  n = 3 ∨ n = 5 := by
sorry

end permutation_square_diff_l415_41567


namespace profit_and_max_profit_l415_41540

/-- The daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 300

/-- The daily profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 10) * sales_quantity x

theorem profit_and_max_profit :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit x₁ = 750 ∧ profit x₂ = 750 ∧ 
    ((∀ x : ℝ, profit x = 750 → x = x₁ ∨ x = x₂) ∧ 
    (x₁ = 15 ∨ x₁ = 25) ∧ (x₂ = 15 ∨ x₂ = 25))) ∧
  (∃ max_profit : ℝ, max_profit = 1000 ∧ ∀ x : ℝ, profit x ≤ max_profit) :=
by sorry

end profit_and_max_profit_l415_41540


namespace square_root_of_16_l415_41542

theorem square_root_of_16 : Real.sqrt 16 = 4 ∧ Real.sqrt 16 = -4 := by
  sorry

end square_root_of_16_l415_41542


namespace calculate_X_l415_41570

theorem calculate_X : ∀ M N X : ℚ,
  M = 3009 / 3 →
  N = M / 4 →
  X = M + 2 * N →
  X = 1504.5 := by
sorry

end calculate_X_l415_41570


namespace unique_solution_l415_41583

theorem unique_solution : ∃! x : ℝ, 
  -1 < x ∧ x ≤ 2 ∧ 
  Real.sqrt (2 - x) + Real.sqrt (2 + 2*x) = Real.sqrt ((x^4 + 1)/(x^2 + 1)) + (x + 3)/(x + 1) :=
by sorry

end unique_solution_l415_41583


namespace days_to_reach_goal_chris_breath_holding_days_l415_41514

/-- Given Chris's breath-holding capacity and improvement rate, calculate the number of days to reach his goal. -/
theorem days_to_reach_goal (start_capacity : ℕ) (daily_improvement : ℕ) (goal : ℕ) : ℕ :=
  let days := (goal - start_capacity) / daily_improvement
  days

/-- Prove that Chris needs 6 more days to reach his goal. -/
theorem chris_breath_holding_days : days_to_reach_goal 30 10 90 = 6 := by
  sorry

end days_to_reach_goal_chris_breath_holding_days_l415_41514


namespace allocation_theorem_l415_41562

/-- The number of ways to allocate employees to departments -/
def allocation_count (total_employees : ℕ) (num_departments : ℕ) : ℕ :=
  sorry

/-- Two employees are considered as one unit -/
def combined_employees : ℕ := 4

/-- Number of ways to distribute combined employees into departments -/
def distribution_ways : ℕ := sorry

/-- Number of ways to assign groups to departments -/
def assignment_ways : ℕ := sorry

theorem allocation_theorem :
  allocation_count 5 3 = distribution_ways * assignment_ways ∧
  distribution_ways = 6 ∧
  assignment_ways = 6 ∧
  allocation_count 5 3 = 36 :=
sorry

end allocation_theorem_l415_41562


namespace hyperbola_focus_asymptote_distance_l415_41535

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, eccentricity e, and length of real axis 2a,
    prove that the distance from the focus to the asymptote line is √3 when e = 2 and 2a = 2. -/
theorem hyperbola_focus_asymptote_distance
  (a b c : ℝ)
  (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : c / a = 2)
  (h_real_axis : 2 * a = 2) :
  (b * c) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 :=
sorry

end hyperbola_focus_asymptote_distance_l415_41535


namespace f_properties_l415_41598

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - Real.cos (2 * x) + 1) / (2 * Real.sin x)

theorem f_properties :
  (∃ (S : Set ℝ), S = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi} ∧ (∀ x : ℝ, x ∈ S ↔ f x ≠ 0)) ∧
  (Set.range f = Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Ioo (-1) 1 ∪ Set.Icc 1 (Real.sqrt 2)) ∧
  (∀ α : ℝ, 0 < α ∧ α < Real.pi / 2 → Real.tan (α / 2) = 1 / 2 → f α = 7 / 5) :=
by sorry

end f_properties_l415_41598


namespace age_multiple_l415_41591

def rons_current_age : ℕ := 43
def maurices_current_age : ℕ := 7
def years_passed : ℕ := 5

theorem age_multiple : 
  (rons_current_age + years_passed) / (maurices_current_age + years_passed) = 4 := by
  sorry

end age_multiple_l415_41591


namespace invalid_external_diagonals_l415_41510

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given lengths satisfy the triangle inequality for external diagonals -/
def satisfies_triangle_inequality (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ 
  d.b^2 + d.c^2 > d.a^2 ∧ 
  d.a^2 + d.c^2 > d.b^2

/-- Theorem stating that {5, 6, 8} cannot be the lengths of external diagonals of a right regular prism -/
theorem invalid_external_diagonals : 
  ¬(satisfies_triangle_inequality ⟨5, 6, 8⟩) :=
by
  sorry

end invalid_external_diagonals_l415_41510


namespace intersection_product_range_l415_41503

open Real

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := ∃ φ, x = 2 * cos φ ∧ y = sin φ
def l (θ ρ : ℝ) (α : ℝ) : Prop := θ = α ∧ ρ > 0

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α ≤ π/4

-- Define the polar equations
def C₁_polar (ρ θ : ℝ) : Prop := ρ = 4 * cos θ
def C₂_polar (ρ θ : ℝ) : Prop := ρ^2 = 4 / (1 + 3 * sin θ^2)

-- Define the intersection points
def M (ρ_M : ℝ) (α : ℝ) : Prop := C₁_polar ρ_M α ∧ l α ρ_M α
def N (ρ_N : ℝ) (α : ℝ) : Prop := C₂_polar ρ_N α ∧ l α ρ_N α

-- State the theorem
theorem intersection_product_range :
  ∀ α ρ_M ρ_N, α_range α → M ρ_M α → N ρ_N α → ρ_M ≠ 0 → ρ_N ≠ 0 →
  (8 * sqrt 5 / 5) ≤ ρ_M * ρ_N ∧ ρ_M * ρ_N ≤ 8 :=
sorry

end intersection_product_range_l415_41503


namespace power_of_1307_squared_cubed_l415_41539

theorem power_of_1307_squared_cubed : (1307 * 1307)^3 = 4984209203082045649 := by
  sorry

end power_of_1307_squared_cubed_l415_41539


namespace hexagon_circumscribable_l415_41547

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon defined by six points -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Checks if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def equal_length (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a circle can be circumscribed around a set of points -/
def can_circumscribe (points : List Point) : Prop := sorry

/-- Theorem: A circle can be circumscribed around a hexagon with the given properties -/
theorem hexagon_circumscribable (h : Hexagon) :
  parallel h.A h.B h.D h.E →
  parallel h.B h.C h.E h.F →
  parallel h.C h.D h.F h.A →
  equal_length h.A h.D h.B h.E →
  equal_length h.A h.D h.C h.F →
  can_circumscribe [h.A, h.B, h.C, h.D, h.E, h.F] := by
  sorry

end hexagon_circumscribable_l415_41547


namespace one_acrobat_l415_41592

/-- Represents the count of animals at the zoo -/
structure ZooCount where
  acrobats : ℕ
  elephants : ℕ
  monkeys : ℕ

/-- Checks if the given ZooCount satisfies the conditions of the problem -/
def isValidCount (count : ZooCount) : Prop :=
  2 * count.acrobats + 4 * count.elephants + 2 * count.monkeys = 134 ∧
  count.acrobats + count.elephants + count.monkeys = 45

/-- Theorem stating that there is exactly one acrobat in the valid zoo count -/
theorem one_acrobat :
  ∃! (count : ZooCount), isValidCount count ∧ count.acrobats = 1 := by
  sorry

#check one_acrobat

end one_acrobat_l415_41592


namespace function_through_point_l415_41581

theorem function_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x : ℝ ↦ a^x) (-1) = 2 → (fun x : ℝ ↦ a^x) = (fun x : ℝ ↦ (1/2)^x) := by
  sorry

end function_through_point_l415_41581


namespace least_n_satisfying_inequality_l415_41590

theorem least_n_satisfying_inequality : ∀ n : ℕ, n > 0 → 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 2) < (1 : ℚ) / 15) ↔ n ≥ 5 :=
by sorry

end least_n_satisfying_inequality_l415_41590


namespace imaginary_part_of_complex_number_l415_41550

theorem imaginary_part_of_complex_number :
  let z : ℂ := 3 - 2 * I
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_number_l415_41550


namespace original_average_proof_l415_41524

theorem original_average_proof (n : ℕ) (original_average : ℚ) : 
  n = 12 → 
  (2 * original_average * n) / n = 100 →
  original_average = 50 := by
sorry

end original_average_proof_l415_41524


namespace apple_distribution_l415_41594

theorem apple_distribution (total_apples : ℕ) (ratio_1_2 ratio_1_3 ratio_2_3 : ℚ) :
  total_apples = 169 →
  ratio_1_2 = 1 / 2 →
  ratio_1_3 = 1 / 3 →
  ratio_2_3 = 1 / 2 →
  ∃ (boy1 boy2 boy3 : ℕ),
    boy1 + boy2 + boy3 = total_apples ∧
    boy1 = 78 ∧
    boy2 = 52 ∧
    boy3 = 39 ∧
    (boy1 : ℚ) / (boy2 : ℚ) = ratio_1_2 ∧
    (boy1 : ℚ) / (boy3 : ℚ) = ratio_1_3 ∧
    (boy2 : ℚ) / (boy3 : ℚ) = ratio_2_3 :=
by
  sorry

#check apple_distribution

end apple_distribution_l415_41594


namespace locus_of_nine_point_center_on_BC_l415_41515

/-- Triangle ABC with fixed vertices B and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ := (-1, 0)
  C : ℝ × ℝ := (1, 0)

/-- The nine-point center of a triangle -/
def ninePointCenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is on a line segment -/
def isOnSegment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

/-- The locus of point A -/
def locusOfA (x y : ℝ) : Prop := x^2 - y^2 = 1

theorem locus_of_nine_point_center_on_BC (t : Triangle) :
  isOnSegment (ninePointCenter t) t.B t.C ↔ locusOfA t.A.1 t.A.2 := by sorry

end locus_of_nine_point_center_on_BC_l415_41515


namespace student_count_third_row_l415_41573

/-- The number of students in the first row -/
def students_first_row : ℕ := 12

/-- The number of students in the second row -/
def students_second_row : ℕ := 12

/-- The change in average age (in weeks) for the first row after rearrangement -/
def change_first_row : ℤ := 1

/-- The change in average age (in weeks) for the second row after rearrangement -/
def change_second_row : ℤ := 2

/-- The change in average age (in weeks) for the third row after rearrangement -/
def change_third_row : ℤ := -4

/-- The number of students in the third row -/
def students_third_row : ℕ := 9

theorem student_count_third_row : 
  students_first_row * change_first_row + 
  students_second_row * change_second_row + 
  students_third_row * change_third_row = 0 :=
by sorry

end student_count_third_row_l415_41573


namespace max_students_distribution_l415_41508

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 2500) (h2 : pencils = 1575) : 
  Nat.gcd pens pencils = 25 := by
  sorry

end max_students_distribution_l415_41508


namespace sequence_properties_l415_41544

def arithmetic_seq (a b n : ℕ) : ℕ := a + (n - 1) * b

def geometric_seq (b a n : ℕ) : ℕ := b * a^(n - 1)

def c_seq (a b n : ℕ) : ℚ := (arithmetic_seq a b n - 8) / (geometric_seq b a n)

theorem sequence_properties (a b : ℕ) :
  (a > 0) →
  (b > 0) →
  (arithmetic_seq a b 1 < geometric_seq b a 1) →
  (geometric_seq b a 1 < arithmetic_seq a b 2) →
  (arithmetic_seq a b 2 < geometric_seq b a 2) →
  (geometric_seq b a 2 < arithmetic_seq a b 3) →
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ arithmetic_seq a b m + 1 = geometric_seq b a n) →
  (a = 2 ∧ b = 3 ∧ ∃ k : ℕ, ∀ n : ℕ, n > 0 → c_seq a b n ≤ c_seq a b k ∧ c_seq a b k = 1/8) :=
by sorry

end sequence_properties_l415_41544


namespace shortest_path_length_l415_41585

/-- The shortest path length from (0,0) to (12,16) avoiding a circle -/
theorem shortest_path_length (start end_ circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  let path_length := 10 * Real.sqrt 3 + 5 * Real.pi / 3
  by
    sorry

#check shortest_path_length (0, 0) (12, 16) (6, 8) 5

end shortest_path_length_l415_41585


namespace min_distance_between_line_and_curve_l415_41500

/-- The minimum distance between a point on y = 2x + 1 and a point on y = x + ln x -/
theorem min_distance_between_line_and_curve : ∃ (d : ℝ), d = (2 * Real.sqrt 5) / 5 ∧
  ∀ (P Q : ℝ × ℝ),
    (P.2 = 2 * P.1 + 1) →
    (Q.2 = Q.1 + Real.log Q.1) →
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end min_distance_between_line_and_curve_l415_41500


namespace gym_attendance_l415_41530

theorem gym_attendance (initial : ℕ) 
  (h1 : initial + 5 - 2 = 19) : initial = 16 := by
  sorry

end gym_attendance_l415_41530


namespace paper_shredder_capacity_l415_41568

theorem paper_shredder_capacity (total_contracts : ℕ) (shred_operations : ℕ) : 
  total_contracts = 2132 → shred_operations = 44 → 
  (total_contracts / shred_operations : ℕ) = 48 := by
  sorry

end paper_shredder_capacity_l415_41568


namespace min_value_problem_l415_41549

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  ∀ x y, x > 0 ∧ y > 1 ∧ x + y = 2 → (4 / a + 1 / (b - 1) ≤ 4 / x + 1 / (y - 1)) ∧
  (∃ x y, x > 0 ∧ y > 1 ∧ x + y = 2 ∧ 4 / x + 1 / (y - 1) = 9) :=
sorry

end min_value_problem_l415_41549


namespace root_value_theorem_l415_41506

theorem root_value_theorem (m : ℝ) (h : m^2 - 2*m - 3 = 0) : 2026 - m^2 + 2*m = 2023 := by
  sorry

end root_value_theorem_l415_41506


namespace painting_wall_percentage_l415_41516

/-- Calculates the percentage of a wall taken up by a painting -/
theorem painting_wall_percentage 
  (painting_width : ℝ) 
  (painting_height : ℝ) 
  (wall_width : ℝ) 
  (wall_height : ℝ) 
  (h1 : painting_width = 2) 
  (h2 : painting_height = 4) 
  (h3 : wall_width = 5) 
  (h4 : wall_height = 10) : 
  (painting_width * painting_height) / (wall_width * wall_height) * 100 = 16 := by
  sorry

#check painting_wall_percentage

end painting_wall_percentage_l415_41516


namespace sin_sum_to_product_l415_41551

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end sin_sum_to_product_l415_41551


namespace quadratic_function_theorem_l415_41546

-- Define the quadratic function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function F
def F (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := F x - k * x

theorem quadratic_function_theorem (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x : ℝ, F x = f a b x) ∧ 
  (∀ k : ℝ, (∀ x ∈ Set.Icc (-2) 2, Monotone (g k)) ↔ (k ≤ -2 ∨ k ≥ 6)) :=
by sorry

end quadratic_function_theorem_l415_41546


namespace simplify_tan_product_l415_41595

theorem simplify_tan_product (tan30 tan15 : ℝ) : 
  tan30 + tan15 = 1 - tan30 * tan15 → (1 + tan30) * (1 + tan15) = 2 := by
  sorry

end simplify_tan_product_l415_41595


namespace binomial_distribution_unique_parameters_l415_41533

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  expectation : ℝ
  variance : ℝ
  h1 : 0 < p ∧ p < 1
  h2 : expectation = n * p
  h3 : variance = n * p * (1 - p)

/-- Theorem stating that a binomial distribution with given expectation and variance has specific n and p values -/
theorem binomial_distribution_unique_parameters
  (ξ : BinomialDistribution)
  (h_expectation : ξ.expectation = 2.4)
  (h_variance : ξ.variance = 1.44) :
  ξ.n = 6 ∧ ξ.p = 0.4 :=
sorry

end binomial_distribution_unique_parameters_l415_41533


namespace max_common_roots_and_coefficients_l415_41541

/-- A polynomial of degree 2020 with non-zero coefficients -/
def Polynomial2020 : Type := { p : Polynomial ℝ // p.degree = some 2020 ∧ ∀ i, p.coeff i ≠ 0 }

/-- The number of common real roots (counting multiplicity) of two polynomials -/
noncomputable def commonRoots (P Q : Polynomial2020) : ℕ := sorry

/-- The number of common coefficients of two polynomials -/
def commonCoefficients (P Q : Polynomial2020) : ℕ := sorry

/-- The main theorem: the maximum possible value of r + s is 3029 -/
theorem max_common_roots_and_coefficients (P Q : Polynomial2020) (h : P ≠ Q) :
  commonRoots P Q + commonCoefficients P Q ≤ 3029 := by sorry

end max_common_roots_and_coefficients_l415_41541


namespace min_value_on_circle_l415_41563

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

/-- A point on the circle -/
structure PointOnCircle where
  a : ℝ
  b : ℝ
  on_circle : circle_equation a b

/-- The theorem stating the minimum value of a^2 + b^2 for points on the circle -/
theorem min_value_on_circle :
  ∀ P : PointOnCircle, ∃ m : ℝ, 
    (∀ Q : PointOnCircle, m ≤ Q.a^2 + Q.b^2) ∧
    m = 30 - 10 * Real.sqrt 5 :=
sorry

end min_value_on_circle_l415_41563


namespace minimum_containers_l415_41566

theorem minimum_containers (medium_capacity small_capacity : ℚ) 
  (h1 : medium_capacity = 450)
  (h2 : small_capacity = 28) : 
  ⌈medium_capacity / small_capacity⌉ = 17 := by
  sorry

end minimum_containers_l415_41566


namespace longest_working_secretary_time_l415_41511

/-- Proves that given three secretaries whose working times are in the ratio of 2:3:5 
    and who worked a combined total of 110 hours, the secretary who worked the longest 
    spent 55 hours on the project. -/
theorem longest_working_secretary_time (a b c : ℕ) : 
  a + b + c = 110 →
  2 * a = 3 * b →
  2 * a = 5 * c →
  c = 55 := by
  sorry

end longest_working_secretary_time_l415_41511


namespace salary_and_new_savings_l415_41552

/-- Represents expenses as percentages of salary -/
structure Expenses where
  food : ℚ
  rent : ℚ
  entertainment : ℚ
  conveyance : ℚ
  utilities : ℚ
  miscellaneous : ℚ

/-- Calculates the total expenses as a percentage -/
def totalExpenses (e : Expenses) : ℚ :=
  e.food + e.rent + e.entertainment + e.conveyance + e.utilities + e.miscellaneous

/-- Calculates the savings percentage -/
def savingsPercentage (e : Expenses) : ℚ :=
  1 - totalExpenses e

/-- Theorem: Given the initial expenses and savings, prove the monthly salary and new savings percentage -/
theorem salary_and_new_savings 
  (initial_expenses : Expenses)
  (initial_savings : ℚ)
  (salary : ℚ)
  (new_entertainment : ℚ)
  (new_conveyance : ℚ)
  (h1 : initial_expenses.food = 0.30)
  (h2 : initial_expenses.rent = 0.25)
  (h3 : initial_expenses.entertainment = 0.15)
  (h4 : initial_expenses.conveyance = 0.10)
  (h5 : initial_expenses.utilities = 0.05)
  (h6 : initial_expenses.miscellaneous = 0.05)
  (h7 : initial_savings = 1500)
  (h8 : savingsPercentage initial_expenses * salary = initial_savings)
  (h9 : new_entertainment = initial_expenses.entertainment + 0.05)
  (h10 : new_conveyance = initial_expenses.conveyance - 0.03)
  : salary = 15000 ∧ 
    savingsPercentage { initial_expenses with 
      entertainment := new_entertainment,
      conveyance := new_conveyance 
    } = 0.08 := by
  sorry

end salary_and_new_savings_l415_41552


namespace nearest_city_distance_l415_41572

theorem nearest_city_distance (d : ℝ) : 
  (¬ (d ≥ 13)) ∧ (¬ (d ≤ 10)) ∧ (¬ (d ≤ 8)) → d ∈ Set.Ioo 10 13 :=
by sorry

end nearest_city_distance_l415_41572


namespace point_transformation_final_coordinates_l415_41512

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to find the symmetric point about the origin -/
def symmetricAboutOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Function to move a point to the left -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  { x := p.x - units, y := p.y }

/-- Theorem stating that the given transformations result in the expected point -/
theorem point_transformation (initialPoint : Point) :
  initialPoint.x = -2 ∧ initialPoint.y = 3 →
  (moveLeft (symmetricAboutOrigin initialPoint) 2).x = 0 ∧
  (moveLeft (symmetricAboutOrigin initialPoint) 2).y = -3 := by
  sorry

/-- Main theorem proving the final coordinates -/
theorem final_coordinates : ∃ (p : Point),
  p.x = -2 ∧ p.y = 3 ∧
  (moveLeft (symmetricAboutOrigin p) 2).x = 0 ∧
  (moveLeft (symmetricAboutOrigin p) 2).y = -3 := by
  sorry

end point_transformation_final_coordinates_l415_41512


namespace problem_solving_percentage_l415_41577

theorem problem_solving_percentage (total : ℕ) (multiple_choice : ℕ) : 
  total = 50 → multiple_choice = 10 → 
  (((total - multiple_choice) : ℚ) / total) * 100 = 80 := by
sorry

end problem_solving_percentage_l415_41577


namespace discount_order_difference_l415_41528

theorem discount_order_difference (initial_price : ℝ) (flat_discount : ℝ) (percentage_discount : ℝ) : 
  initial_price = 30 ∧ 
  flat_discount = 5 ∧ 
  percentage_discount = 0.25 →
  (initial_price - flat_discount) * (1 - percentage_discount) - 
  (initial_price * (1 - percentage_discount) - flat_discount) = 1.25 := by
sorry

end discount_order_difference_l415_41528


namespace petya_candies_when_masha_gets_101_l415_41569

def candy_game (n : ℕ) : ℕ × ℕ := 
  let masha_sum := n^2
  let petya_sum := n * (n + 1)
  (masha_sum, petya_sum)

theorem petya_candies_when_masha_gets_101 : 
  ∃ n : ℕ, (candy_game n).1 ≥ 101 ∧ (candy_game (n-1)).1 < 101 → (candy_game (n-1)).2 = 110 :=
by sorry

end petya_candies_when_masha_gets_101_l415_41569


namespace existence_of_x0_l415_41523

theorem existence_of_x0 (a : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1) 1 ∧ x₀^3 < -x₀^2 + x₀ - a) ↔
  (a > 5/27) :=
by sorry

end existence_of_x0_l415_41523


namespace cover_room_with_tiles_l415_41526

/-- The length of the room -/
def room_length : ℝ := 8

/-- The width of the room -/
def room_width : ℝ := 12

/-- The length of a tile -/
def tile_length : ℝ := 1.5

/-- The width of a tile -/
def tile_width : ℝ := 2

/-- The number of tiles needed to cover the room -/
def tiles_needed : ℕ := 32

theorem cover_room_with_tiles : 
  (room_length * room_width) / (tile_length * tile_width) = tiles_needed := by
  sorry

end cover_room_with_tiles_l415_41526


namespace distance_sum_on_corresponding_segments_l415_41587

/-- Given two line segments AB and A'B' with lengths 6 and 16 respectively,
    and a linear correspondence between points on these segments,
    prove that the sum of distances from A to P and A' to P' is 18/5 * a,
    where a is the distance from A to P. -/
theorem distance_sum_on_corresponding_segments
  (AB : Real) (A'B' : Real)
  (a : Real)
  (h1 : AB = 6)
  (h2 : A'B' = 16)
  (h3 : 0 ≤ a ∧ a ≤ AB)
  (correspondence : Real → Real)
  (h4 : correspondence 1 = 3)
  (h5 : ∀ x, 0 ≤ x ∧ x ≤ AB → 0 ≤ correspondence x ∧ correspondence x ≤ A'B')
  (h6 : ∀ x y, (0 ≤ x ∧ x ≤ AB ∧ 0 ≤ y ∧ y ≤ AB) →
              (correspondence x - correspondence y) / (x - y) = (correspondence 1 - 0) / (1 - 0)) :
  a + correspondence a = 18/5 * a := by
  sorry

end distance_sum_on_corresponding_segments_l415_41587


namespace first_lift_weight_l415_41548

/-- Given two lifts with a total weight of 600 pounds, where twice the weight of the first lift
    is 300 pounds more than the weight of the second lift, prove that the weight of the first lift
    is 300 pounds. -/
theorem first_lift_weight (first_lift second_lift : ℕ) 
  (total_weight : first_lift + second_lift = 600)
  (lift_relation : 2 * first_lift = second_lift + 300) : 
  first_lift = 300 := by
  sorry

end first_lift_weight_l415_41548


namespace last_duck_bread_pieces_l415_41507

theorem last_duck_bread_pieces (total : ℕ) (left : ℕ) (first_duck : ℕ) (second_duck : ℕ) :
  total = 100 →
  left = 30 →
  first_duck = total / 2 →
  second_duck = 13 →
  total - left - first_duck - second_duck = 7 :=
by sorry

end last_duck_bread_pieces_l415_41507


namespace max_value_cube_root_sum_l415_41501

theorem max_value_cube_root_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a^2 * b^2 * c^2)^(1/3) + ((1 - a^2) * (1 - b^2) * (1 - c^2))^(1/3) ≤ 1 ∧
  ∃ x y z, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 ∧
    (x^2 * y^2 * z^2)^(1/3) + ((1 - x^2) * (1 - y^2) * (1 - z^2))^(1/3) = 1 :=
by sorry

end max_value_cube_root_sum_l415_41501


namespace min_toothpicks_removal_l415_41531

/-- Represents a triangular grid figure made of toothpicks -/
structure TriangularGrid where
  total_toothpicks : ℕ
  total_triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (grid : TriangularGrid) : ℕ :=
  grid.horizontal_toothpicks

/-- Theorem: For a specific triangular grid, the minimum number of toothpicks 
    to remove to eliminate all triangles is 15 -/
theorem min_toothpicks_removal (grid : TriangularGrid) 
    (h1 : grid.total_toothpicks = 40)
    (h2 : grid.total_triangles > 35)
    (h3 : grid.horizontal_toothpicks = 15) : 
  min_toothpicks_to_remove grid = 15 := by
  sorry

end min_toothpicks_removal_l415_41531


namespace log_7_18_l415_41520

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : Real.log 2 / Real.log 10 = a)
variable (h2 : Real.log 3 / Real.log 10 = b)

-- State the theorem to be proved
theorem log_7_18 : Real.log 18 / Real.log 7 = (a + 2*b) / (1 - a) := by
  sorry

end log_7_18_l415_41520


namespace xyz_product_magnitude_l415_41534

theorem xyz_product_magnitude (x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  x ≠ y → y ≠ z → x ≠ z →
  x + 1/y = y + 1/z → y + 1/z = z + 1/x + 1 →
  |x*y*z| = 1 := by
sorry

end xyz_product_magnitude_l415_41534


namespace cube_roots_of_primes_not_in_arithmetic_progression_l415_41532

theorem cube_roots_of_primes_not_in_arithmetic_progression 
  (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ¬∃ (a d : ℝ), {(p : ℝ)^(1/3), (q : ℝ)^(1/3), (r : ℝ)^(1/3)} ⊆ {a + n * d | n : ℤ} :=
by sorry

end cube_roots_of_primes_not_in_arithmetic_progression_l415_41532


namespace equation_solution_l415_41588

theorem equation_solution : ∃! x : ℚ, (x - 35) / 3 = (3 * x + 10) / 8 ∧ x = -310 := by sorry

end equation_solution_l415_41588


namespace eight_digit_non_decreasing_remainder_l415_41521

/-- The number of ways to arrange n indistinguishable objects into k distinguishable boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of 8-digit positive integers with non-decreasing digits -/
def M : ℕ := stars_and_bars 8 10

theorem eight_digit_non_decreasing_remainder :
  M % 1000 = 310 := by sorry

end eight_digit_non_decreasing_remainder_l415_41521


namespace quadratic_roots_difference_l415_41557

-- Define the quadratic equation
def quadratic (x p : ℝ) : ℝ := 2 * x^2 + p * x + 4

-- Define the condition that the roots differ by 2
def roots_differ_by_two (p : ℤ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ quadratic x p = 0 ∧ quadratic y p = 0 ∧ |x - y| = 2

-- The theorem to prove
theorem quadratic_roots_difference (p : ℤ) :
  roots_differ_by_two p → p = 7 ∨ p = -7 := by
  sorry


end quadratic_roots_difference_l415_41557


namespace chess_match_outcomes_count_l415_41543

/-- The number of different possible outcomes for a chess match draw -/
def chessMatchOutcomes : ℕ :=
  2^8 * Nat.factorial 8

/-- Theorem stating the number of different possible outcomes for a chess match draw -/
theorem chess_match_outcomes_count :
  chessMatchOutcomes = 2^8 * Nat.factorial 8 := by
  sorry

#eval chessMatchOutcomes

end chess_match_outcomes_count_l415_41543


namespace square_equals_eight_times_reciprocal_l415_41576

theorem square_equals_eight_times_reciprocal (x : ℝ) : 
  x > 0 → x^2 = 8 * (1/x) → x = 2 := by sorry

end square_equals_eight_times_reciprocal_l415_41576


namespace sum_of_parts_l415_41596

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 54) (h2 : y = 34) : 10 * x + 22 * y = 948 := by
  sorry

end sum_of_parts_l415_41596


namespace flashlight_distance_ratio_l415_41559

/-- Proves that the ratio of Freddie's flashlight distance to Veronica's is 3:1 --/
theorem flashlight_distance_ratio :
  ∀ (V F : ℕ),
  V = 1000 →
  F > V →
  ∃ (D : ℕ), D = 5 * F - 2000 →
  D = V + 12000 →
  F / V = 3 :=
by sorry

end flashlight_distance_ratio_l415_41559


namespace min_value_problem_l415_41522

theorem min_value_problem (x y : ℝ) (h1 : x * y + 1 = 4 * x + y) (h2 : x > 1) :
  ∃ (min : ℝ), min = 27 ∧ ∀ z, z = (x + 1) * (y + 2) → z ≥ min :=
by sorry

end min_value_problem_l415_41522


namespace skyscraper_anniversary_l415_41504

theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 - years_since_built →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary = 95 := by
  sorry

end skyscraper_anniversary_l415_41504


namespace system_solution_l415_41529

theorem system_solution (x y z t : ℤ) : 
  (x * y + z * t = 1 ∧ 
   x * z + y * t = 1 ∧ 
   x * t + y * z = 1) ↔ 
  ((x, y, z, t) = (0, 1, 1, 1) ∨
   (x, y, z, t) = (1, 0, 1, 1) ∨
   (x, y, z, t) = (1, 1, 0, 1) ∨
   (x, y, z, t) = (1, 1, 1, 0) ∨
   (x, y, z, t) = (0, -1, -1, -1) ∨
   (x, y, z, t) = (-1, 0, -1, -1) ∨
   (x, y, z, t) = (-1, -1, 0, -1) ∨
   (x, y, z, t) = (-1, -1, -1, 0)) :=
by sorry

end system_solution_l415_41529


namespace binomial_congruence_characterization_l415_41571

theorem binomial_congruence_characterization (n : ℕ) (hn : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n →
    (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  ∃ p : ℕ, p > 0 ∧ n = 2^p - 2 :=
by sorry

end binomial_congruence_characterization_l415_41571


namespace roses_cut_l415_41554

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) :
  final_roses - initial_roses = 11 := by
  sorry

end roses_cut_l415_41554


namespace certain_number_power_l415_41519

theorem certain_number_power (m : ℤ) (a : ℝ) : 
  (-2 : ℝ)^(2*m) = a^(21-m) → m = 7 → a = -2 := by sorry

end certain_number_power_l415_41519


namespace quadratic_inequality_range_l415_41574

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x - 2 < 0) ↔ a ∈ Set.Ioc (-2) 0 :=
by sorry

end quadratic_inequality_range_l415_41574


namespace new_student_weight_l415_41589

theorem new_student_weight (n : ℕ) (w_avg : ℝ) (w_new_avg : ℝ) (w_new : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new_avg = 27.1 →
  n * w_avg + w_new = (n + 1) * w_new_avg →
  w_new = 1 := by
sorry

end new_student_weight_l415_41589


namespace modulus_of_z_l415_41575

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 3 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l415_41575


namespace prime_sum_square_fourth_power_l415_41586

theorem prime_sum_square_fourth_power : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p + q^2 = r^4 → 
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end prime_sum_square_fourth_power_l415_41586


namespace point_coordinates_l415_41553

/-- Given a point A(-m, √m) in the Cartesian coordinate system,
    prove that its coordinates are (-16, 4) if its distance to the x-axis is 4. -/
theorem point_coordinates (m : ℝ) :
  (∃ A : ℝ × ℝ, A = (-m, Real.sqrt m) ∧ |A.2| = 4) →
  (∃ A : ℝ × ℝ, A = (-16, 4)) :=
by sorry

end point_coordinates_l415_41553


namespace golden_ratio_approximation_l415_41593

theorem golden_ratio_approximation :
  (∃ (S : Set ℚ), Set.Infinite S ∧
    ∀ r ∈ S, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ r = q / p ∧
      |r - (Real.sqrt 5 - 1) / 2| < 1 / p^2) ∧
  (∀ p q : ℤ, p > 0 → Int.gcd p q = 1 →
    |(q : ℝ) / p - (Real.sqrt 5 - 1) / 2| > 1 / (Real.sqrt 5 + 1) / p^2) := by
  sorry

end golden_ratio_approximation_l415_41593


namespace cathys_final_balance_l415_41537

def cathys_money (initial_balance dad_contribution : ℕ) : ℕ :=
  initial_balance + dad_contribution + 2 * dad_contribution

theorem cathys_final_balance :
  cathys_money 12 25 = 87 :=
by sorry

end cathys_final_balance_l415_41537


namespace zachary_did_19_pushups_l415_41509

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups David did -/
def david_pushups : ℕ := 58

/-- The difference between David's and Zachary's push-ups -/
def difference : ℕ := 39

/-- Theorem stating that Zachary did 19 push-ups given the conditions -/
theorem zachary_did_19_pushups : zachary_pushups = 19 := by sorry

end zachary_did_19_pushups_l415_41509


namespace root_difference_cubic_equation_l415_41599

theorem root_difference_cubic_equation :
  ∃ (α β γ : ℝ),
    (81 * α^3 - 162 * α^2 + 90 * α - 10 = 0) ∧
    (81 * β^3 - 162 * β^2 + 90 * β - 10 = 0) ∧
    (81 * γ^3 - 162 * γ^2 + 90 * γ - 10 = 0) ∧
    (β = 2 * α ∨ γ = 2 * α ∨ γ = 2 * β) ∧
    (max α (max β γ) - min α (min β γ) = 1) :=
sorry

end root_difference_cubic_equation_l415_41599


namespace composite_face_dots_l415_41536

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the four faces of interest in the composite figure -/
inductive CompositeFace
  | A
  | B
  | C
  | D

/-- A function that returns the number of dots on a die face -/
def dots_on_face (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- A function that maps a composite face to its corresponding die face -/
def composite_to_die_face (face : CompositeFace) : DieFace :=
  match face with
  | CompositeFace.A => DieFace.three
  | CompositeFace.B => DieFace.five
  | CompositeFace.C => DieFace.six
  | CompositeFace.D => DieFace.five

/-- Theorem stating the number of dots on each composite face -/
theorem composite_face_dots (face : CompositeFace) :
  dots_on_face (composite_to_die_face face) =
    match face with
    | CompositeFace.A => 3
    | CompositeFace.B => 5
    | CompositeFace.C => 6
    | CompositeFace.D => 5 := by
  sorry

end composite_face_dots_l415_41536


namespace circle_polygons_l415_41545

theorem circle_polygons (n : ℕ) (h : n = 15) :
  let quadrilaterals := Nat.choose n 4
  let triangles := Nat.choose n 3
  quadrilaterals + triangles = 1820 := by
  sorry

end circle_polygons_l415_41545


namespace complex_roots_of_quadratic_l415_41502

theorem complex_roots_of_quadratic : 
  let z₁ : ℂ := -1 + Real.sqrt 2 - Complex.I * Real.sqrt 2
  let z₂ : ℂ := -1 - Real.sqrt 2 + Complex.I * Real.sqrt 2
  (z₁^2 + 2*z₁ = 3 - 4*Complex.I) ∧ (z₂^2 + 2*z₂ = 3 - 4*Complex.I) := by
  sorry


end complex_roots_of_quadratic_l415_41502


namespace train_length_proof_l415_41518

/-- The length of a train that passes a pole in 15 seconds and a 100-meter platform in 40 seconds -/
def train_length : ℝ := 60

theorem train_length_proof (t : ℝ) (h1 : t > 0) :
  (t / 15 = (t + 100) / 40) → t = train_length :=
by
  sorry

#check train_length_proof

end train_length_proof_l415_41518


namespace polynomial_root_problem_l415_41555

theorem polynomial_root_problem (d : ℚ) :
  (∃ (x : ℝ), x^3 + 4*x + d = 0 ∧ x = 2 + Real.sqrt 5) →
  (∃ (n : ℤ), n^3 + 4*n + d = 0) →
  (∃ (n : ℤ), n^3 + 4*n + d = 0 ∧ n = -4) :=
by sorry

end polynomial_root_problem_l415_41555


namespace area_of_triangle_from_centers_area_is_sqrt_three_l415_41597

/-- The area of an equilateral triangle formed by connecting the centers of three equilateral
    triangles of side length 2, arranged around a vertex of a square. -/
theorem area_of_triangle_from_centers : ℝ :=
  let side_length : ℝ := 2
  let triangle_centers_distance : ℝ := side_length
  let area_formula (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2
  area_formula triangle_centers_distance

/-- The area of the triangle formed by connecting the centers is √3. -/
theorem area_is_sqrt_three : area_of_triangle_from_centers = Real.sqrt 3 := by
  sorry

end area_of_triangle_from_centers_area_is_sqrt_three_l415_41597


namespace bobby_shoes_count_l415_41578

/-- Given information about the number of shoes owned by Bonny, Becky, and Bobby, 
    prove that Bobby has 27 pairs of shoes. -/
theorem bobby_shoes_count : 
  ∀ (becky_shoes : ℕ), 
  (13 = 2 * becky_shoes - 5) →  -- Bonny's shoes are 5 less than twice Becky's
  (27 = 3 * becky_shoes) -- Bobby has 3 times as many shoes as Becky
  := by sorry

end bobby_shoes_count_l415_41578


namespace remaining_sales_to_goal_l415_41525

def goal : ℕ := 100

def grandmother_sales : ℕ := 5
def uncle_initial_sales : ℕ := 12
def neighbor_initial_sales : ℕ := 8
def mother_friend_sales : ℕ := 25
def cousin_initial_sales : ℕ := 3
def uncle_additional_sales : ℕ := 10
def neighbor_returns : ℕ := 4
def cousin_additional_sales : ℕ := 5

def total_sales : ℕ := 
  grandmother_sales + 
  (uncle_initial_sales + uncle_additional_sales) + 
  (neighbor_initial_sales - neighbor_returns) + 
  mother_friend_sales + 
  (cousin_initial_sales + cousin_additional_sales)

theorem remaining_sales_to_goal : goal - total_sales = 36 := by
  sorry

end remaining_sales_to_goal_l415_41525


namespace equal_fractions_from_given_numbers_l415_41538

theorem equal_fractions_from_given_numbers : 
  let numbers : Finset ℕ := {2, 4, 5, 6, 12, 15}
  ∃ (a b c d e f : ℕ), 
    a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a : ℚ) / b = (c : ℚ) / d ∧ (c : ℚ) / d = (e : ℚ) / f :=
by
  sorry

end equal_fractions_from_given_numbers_l415_41538


namespace price_increase_calculation_l415_41560

/-- Represents the ticket pricing model for an airline -/
structure TicketPricing where
  basePrice : ℝ
  daysBeforeDeparture : ℕ
  dailyIncreaseRate : ℝ

/-- Calculates the price increase for buying a ticket one day later -/
def priceIncrease (pricing : TicketPricing) : ℝ :=
  pricing.basePrice * pricing.dailyIncreaseRate

/-- Theorem: The price increase for buying a ticket one day later is $52.50 -/
theorem price_increase_calculation (pricing : TicketPricing)
  (h1 : pricing.basePrice = 1050)
  (h2 : pricing.daysBeforeDeparture = 14)
  (h3 : pricing.dailyIncreaseRate = 0.05) :
  priceIncrease pricing = 52.50 := by
  sorry

#eval priceIncrease { basePrice := 1050, daysBeforeDeparture := 14, dailyIncreaseRate := 0.05 }

end price_increase_calculation_l415_41560


namespace opposite_quadratics_solution_l415_41584

theorem opposite_quadratics_solution (x : ℚ) : 
  (2 * x^2 + 1 = -(4 * x^2 - 2 * x - 5)) → (x = 1 ∨ x = -2/3) := by
  sorry

end opposite_quadratics_solution_l415_41584


namespace function_inequality_l415_41561

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

-- State the theorem
theorem function_inequality :
  (∀ x, -8 < x → x < 8 → f x ≠ 0) →  -- f is defined on (-8, 8)
  is_even f →
  is_monotonic_on f 0 8 →
  f (-3) < f 2 →
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
sorry

end function_inequality_l415_41561


namespace essay_word_ratio_l415_41565

def johnny_words : ℕ := 150
def timothy_words (madeline_words : ℕ) : ℕ := madeline_words + 30
def total_pages : ℕ := 3
def words_per_page : ℕ := 260

theorem essay_word_ratio (madeline_words : ℕ) :
  (johnny_words + madeline_words + timothy_words madeline_words = total_pages * words_per_page) →
  (madeline_words : ℚ) / johnny_words = 2 := by
  sorry

end essay_word_ratio_l415_41565


namespace power_of_64_l415_41527

theorem power_of_64 : (64 : ℝ) ^ (5/3) = 1024 := by
  sorry

end power_of_64_l415_41527


namespace trajectory_of_point_m_l415_41513

/-- The trajectory of point M on a line segment AB with given conditions -/
theorem trajectory_of_point_m (a : ℝ) (x y : ℝ) :
  (∃ (m b : ℝ),
    -- AB has length 2a
    m^2 + b^2 = (2*a)^2 ∧
    -- M divides AB in ratio 1:2
    x = (2/3) * m ∧
    y = (1/3) * b) →
  x^2 / ((4/3 * a)^2) + y^2 / ((2/3 * a)^2) = 1 :=
by sorry

end trajectory_of_point_m_l415_41513
