import Mathlib

namespace reducible_factorial_fraction_l2621_262132

theorem reducible_factorial_fraction (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ k ∣ n.factorial ∧ k ∣ (n + 1)) ↔
  (n % 2 = 1 ∧ n > 1) ∨ (n % 2 = 0 ∧ ¬(Nat.Prime (n + 1))) :=
by sorry

end reducible_factorial_fraction_l2621_262132


namespace actual_journey_equation_hypothetical_journey_equation_distance_AB_l2621_262173

/-- The distance between dock A and dock B in kilometers -/
def distance : ℝ := 270

/-- The initial speed of the steamboat in km/hr -/
noncomputable def initial_speed : ℝ := distance / 22.5

/-- Time equation for the actual journey -/
theorem actual_journey_equation :
  distance / initial_speed + 3.5 = 3 + (distance - 2 * initial_speed) / (0.8 * initial_speed) :=
sorry

/-- Time equation for the hypothetical journey with later stop -/
theorem hypothetical_journey_equation :
  distance / initial_speed + 1.5 = 3 + 180 / initial_speed + (distance - 2 * initial_speed - 180) / (0.8 * initial_speed) :=
sorry

/-- The distance AB is 270 km -/
theorem distance_AB : distance = 270 :=
sorry

end actual_journey_equation_hypothetical_journey_equation_distance_AB_l2621_262173


namespace two_pizzas_not_enough_l2621_262187

/-- Represents a pizza with its toppings -/
structure Pizza where
  hasTomatoes : Bool
  hasMushrooms : Bool
  hasSausage : Bool

/-- Represents a child's pizza preference -/
structure Preference where
  wantsTomatoes : Option Bool
  wantsMushrooms : Option Bool
  wantsSausage : Option Bool

/-- Checks if a pizza satisfies a child's preference -/
def satisfiesPreference (pizza : Pizza) (pref : Preference) : Bool :=
  (pref.wantsTomatoes.isNone || pref.wantsTomatoes == some pizza.hasTomatoes) &&
  (pref.wantsMushrooms.isNone || pref.wantsMushrooms == some pizza.hasMushrooms) &&
  (pref.wantsSausage.isNone || pref.wantsSausage == some pizza.hasSausage)

def masha : Preference := { wantsTomatoes := some true, wantsMushrooms := none, wantsSausage := some false }
def vanya : Preference := { wantsTomatoes := none, wantsMushrooms := some true, wantsSausage := none }
def dasha : Preference := { wantsTomatoes := some false, wantsMushrooms := none, wantsSausage := none }
def nikita : Preference := { wantsTomatoes := some true, wantsMushrooms := some false, wantsSausage := none }
def igor : Preference := { wantsTomatoes := none, wantsMushrooms := some false, wantsSausage := some true }

theorem two_pizzas_not_enough : 
  ∀ (pizza1 pizza2 : Pizza), 
  ¬(satisfiesPreference pizza1 masha ∨ satisfiesPreference pizza2 masha) ∨
  ¬(satisfiesPreference pizza1 vanya ∨ satisfiesPreference pizza2 vanya) ∨
  ¬(satisfiesPreference pizza1 dasha ∨ satisfiesPreference pizza2 dasha) ∨
  ¬(satisfiesPreference pizza1 nikita ∨ satisfiesPreference pizza2 nikita) ∨
  ¬(satisfiesPreference pizza1 igor ∨ satisfiesPreference pizza2 igor) :=
sorry

end two_pizzas_not_enough_l2621_262187


namespace dividend_divisor_quotient_remainder_l2621_262113

theorem dividend_divisor_quotient_remainder (n : ℕ) : 
  n / 9 = 6 ∧ n % 9 = 4 → n = 58 := by
  sorry

end dividend_divisor_quotient_remainder_l2621_262113


namespace product_of_ab_is_one_l2621_262116

theorem product_of_ab_is_one (a b : ℝ) 
  (h1 : a + 1/b = 4) 
  (h2 : 1/a + b = 16/15) : 
  (a * b) * (a * b) - 34/15 * (a * b) + 1 = 0 := by
  sorry

end product_of_ab_is_one_l2621_262116


namespace root_in_interval_l2621_262110

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x - 5

-- State the theorem
theorem root_in_interval :
  Continuous f →
  f 2 < 0 →
  f 2.5 > 0 →
  ∃ x ∈ Set.Ioo 2 2.5, f x = 0 := by sorry

end root_in_interval_l2621_262110


namespace collinear_complex_points_l2621_262107

theorem collinear_complex_points (z : ℂ) : 
  (∃ (t : ℝ), z = 1 + t * (Complex.I - 1)) → Complex.abs z = 5 → 
  (z = 4 - 3 * Complex.I ∨ z = -3 + 4 * Complex.I) :=
by sorry

end collinear_complex_points_l2621_262107


namespace twelve_numbers_divisible_by_three_l2621_262178

theorem twelve_numbers_divisible_by_three (n : ℕ) : 
  (n ≥ 10) ∧ 
  (∃ (seq : List ℕ), seq.length = 12 ∧ 
    (∀ x ∈ seq, x ≥ 10 ∧ x ≤ n ∧ x % 3 = 0) ∧
    (∀ y, y ≥ 10 ∧ y ≤ n ∧ y % 3 = 0 → y ∈ seq)) →
  n = 45 :=
by sorry

end twelve_numbers_divisible_by_three_l2621_262178


namespace total_candies_l2621_262185

/-- The number of candies each person has -/
structure Candies where
  adam : ℕ
  james : ℕ
  rubert : ℕ
  lisa : ℕ
  chris : ℕ
  max : ℕ
  emily : ℕ

/-- The conditions of the candy distribution -/
def candy_conditions (c : Candies) : Prop :=
  c.adam = 6 ∧
  c.james = 3 * c.adam ∧
  c.rubert = 4 * c.james ∧
  c.lisa = 2 * c.rubert - 5 ∧
  c.chris = c.lisa / 2 + 7 ∧
  c.max = c.rubert + c.chris + 2 ∧
  c.emily = 3 * c.chris - (c.max - c.lisa)

/-- The theorem stating the total number of candies -/
theorem total_candies (c : Candies) (h : candy_conditions c) : 
  c.adam + c.james + c.rubert + c.lisa + c.chris + c.max + c.emily = 678 := by
  sorry

end total_candies_l2621_262185


namespace min_transportation_cost_min_cost_at_ten_l2621_262199

/-- Represents the total transportation cost function --/
def transportation_cost (x : ℝ) : ℝ := 4 * x + 1980

/-- Theorem stating the minimum transportation cost --/
theorem min_transportation_cost :
  ∀ x : ℝ, 10 ≤ x ∧ x ≤ 50 → transportation_cost x ≥ 2020 :=
by
  sorry

/-- Theorem stating that the minimum cost occurs at x = 10 --/
theorem min_cost_at_ten :
  transportation_cost 10 = 2020 :=
by
  sorry

end min_transportation_cost_min_cost_at_ten_l2621_262199


namespace congruence_solutions_count_l2621_262119

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => x < 150 ∧ (x + 17) % 46 = 75 % 46) (Finset.range 150)).card = 3 := by
  sorry

end congruence_solutions_count_l2621_262119


namespace power_sum_equality_l2621_262133

theorem power_sum_equality (a b : ℕ+) (h1 : 2^(a:ℕ) = 8^(b:ℕ)) (h2 : a + 2*b = 5) :
  2^(a:ℕ) + 8^(b:ℕ) = 16 := by
  sorry

end power_sum_equality_l2621_262133


namespace solve_equation_l2621_262198

theorem solve_equation : 48 / (7 - 3/4) = 192/25 := by
  sorry

end solve_equation_l2621_262198


namespace protein_percentage_in_mixture_l2621_262103

/-- Calculates the protein percentage in a mixture of soybean meal and cornmeal. -/
theorem protein_percentage_in_mixture 
  (soybean_protein_percent : ℝ)
  (cornmeal_protein_percent : ℝ)
  (total_mixture_weight : ℝ)
  (soybean_weight : ℝ)
  (cornmeal_weight : ℝ)
  (h1 : soybean_protein_percent = 0.14)
  (h2 : cornmeal_protein_percent = 0.07)
  (h3 : total_mixture_weight = 280)
  (h4 : soybean_weight = 240)
  (h5 : cornmeal_weight = 40)
  (h6 : total_mixture_weight = soybean_weight + cornmeal_weight) :
  (soybean_weight * soybean_protein_percent + cornmeal_weight * cornmeal_protein_percent) / total_mixture_weight = 0.13 := by
  sorry


end protein_percentage_in_mixture_l2621_262103


namespace horner_method_v3_l2621_262117

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 - 3x^3 + 2x^2 + 5x + 1 -/
def f : List ℝ := [4, 0, -3, 2, 5, 1]

theorem horner_method_v3 :
  let v3 := (horner (f.take 4) 3)
  v3 = 101 := by sorry

end horner_method_v3_l2621_262117


namespace jenny_cat_expenses_l2621_262165

theorem jenny_cat_expenses (adoption_fee : ℝ) (vet_visits : ℝ) (monthly_food_cost : ℝ) (toy_cost : ℝ) :
  adoption_fee = 50 →
  vet_visits = 500 →
  monthly_food_cost = 25 →
  toy_cost = 200 →
  (adoption_fee + vet_visits + 12 * monthly_food_cost) / 2 + toy_cost = 625 := by
  sorry

end jenny_cat_expenses_l2621_262165


namespace inequality_solution_one_inequality_solution_two_l2621_262163

-- Part 1
theorem inequality_solution_one : 
  {x : ℝ | 1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4)} := by sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | (x - a) / (x - a^2) < 0}

theorem inequality_solution_two (a : ℝ) : 
  (a = 0 ∨ a = 1 → solution_set a = ∅) ∧
  (0 < a ∧ a < 1 → solution_set a = {x : ℝ | a^2 < x ∧ x < a}) ∧
  ((a < 0 ∨ a > 1) → solution_set a = {x : ℝ | a < x ∧ x < a^2}) := by sorry

end inequality_solution_one_inequality_solution_two_l2621_262163


namespace mean_score_remaining_students_l2621_262141

theorem mean_score_remaining_students 
  (n : ℕ) 
  (h1 : n > 20) 
  (h2 : (15 : ℝ) * 10 = (15 : ℝ) * mean_first_15)
  (h3 : (5 : ℝ) * 16 = (5 : ℝ) * mean_next_5)
  (h4 : ((15 : ℝ) * mean_first_15 + (5 : ℝ) * mean_next_5 + (n - 20 : ℝ) * mean_remaining) / n = 11) :
  mean_remaining = (11 * n - 230) / (n - 20) := by
  sorry

end mean_score_remaining_students_l2621_262141


namespace solution_value_l2621_262137

theorem solution_value (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ 2 * x - m = -3) → m = 5 :=
by sorry

end solution_value_l2621_262137


namespace ellipse_line_intersection_range_l2621_262115

-- Define the ellipse and line
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def line (x y : ℝ) : Prop := x - y = 1

-- Define the theorem
theorem ellipse_line_intersection_range (a b : ℝ) (P Q : ℝ × ℝ) :
  a > 0 → b > 0 → a > b →
  ellipse a b P.1 P.2 →
  ellipse a b Q.1 Q.2 →
  line P.1 P.2 →
  line Q.1 Q.2 →
  (P.1 * Q.1 + P.2 * Q.2 = 0) →
  Real.sqrt 2 / 2 * a ≤ b →
  b ≤ Real.sqrt 6 / 3 * a →
  Real.sqrt 5 / 2 ≤ a ∧ a ≤ Real.sqrt 6 / 2 :=
by sorry

#check ellipse_line_intersection_range

end ellipse_line_intersection_range_l2621_262115


namespace sine_of_angle_plus_three_half_pi_l2621_262111

theorem sine_of_angle_plus_three_half_pi (α : Real) :
  (∃ (x y : Real), x = -5 ∧ y = -12 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
  sorry

end sine_of_angle_plus_three_half_pi_l2621_262111


namespace evans_class_enrollment_l2621_262158

theorem evans_class_enrollment (q1 q2 both not_taken : ℕ) 
  (h1 : q1 = 19)
  (h2 : q2 = 24)
  (h3 : both = 19)
  (h4 : not_taken = 5) :
  q1 + q2 - both + not_taken = 29 := by
sorry

end evans_class_enrollment_l2621_262158


namespace simplest_quadratic_radical_l2621_262186

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (x : ℝ) : Prop :=
  x > 0 ∧ ¬∃ (a b : ℕ), (b > 1) ∧ (¬isPerfectSquare b) ∧ (x = (a : ℝ) * Real.sqrt b)

-- Theorem statement
theorem simplest_quadratic_radical :
  isSimplestQuadraticRadical (Real.sqrt 7) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt (2/3)) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt 0.3) :=
sorry

end simplest_quadratic_radical_l2621_262186


namespace school_sample_theorem_l2621_262105

theorem school_sample_theorem (total_students sample_size : ℕ) 
  (h_total : total_students = 1200)
  (h_sample : sample_size = 200)
  (h_stratified : ∃ (boys girls : ℕ), boys + girls = sample_size ∧ boys = girls + 10) :
  ∃ (school_boys : ℕ), 
    school_boys * sample_size = 105 * total_students ∧
    school_boys = 630 := by
sorry

end school_sample_theorem_l2621_262105


namespace parabola_vertex_below_x_axis_l2621_262160

/-- A parabola with equation y = x^2 + 2x + a has its vertex below the x-axis -/
def vertex_below_x_axis (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x^2 + 2*x + a ∧ y < 0 ∧ ∀ (x' : ℝ), x'^2 + 2*x' + a ≥ y

theorem parabola_vertex_below_x_axis (a : ℝ) :
  vertex_below_x_axis a → a < 1 := by
  sorry

end parabola_vertex_below_x_axis_l2621_262160


namespace expression_simplification_l2621_262150

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = 2 - Real.sqrt 2) : 
  (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a)) = -1/2 := by
  sorry

end expression_simplification_l2621_262150


namespace complement_of_union_theorem_l2621_262175

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_theorem :
  (U \ (A ∪ B)) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end complement_of_union_theorem_l2621_262175


namespace range_of_expressions_l2621_262182

theorem range_of_expressions (a b : ℝ) 
  (ha : -6 < a ∧ a < 8) 
  (hb : 2 < b ∧ b < 3) : 
  (-10 < 2*a + b ∧ 2*a + b < 19) ∧ 
  (-9 < a - b ∧ a - b < 6) ∧ 
  (-2 < a / b ∧ a / b < 4) := by
sorry

end range_of_expressions_l2621_262182


namespace reciprocal_of_negative_one_third_l2621_262147

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  (x * y = 1) → (y = 1 / x) :=
by sorry

end reciprocal_of_negative_one_third_l2621_262147


namespace obtuse_triangle_equilateral_triangle_l2621_262168

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem 1: If cos A * cos B * cos C < 0, then the triangle is obtuse
theorem obtuse_triangle (t : Triangle) :
  Real.cos t.A * Real.cos t.B * Real.cos t.C < 0 →
  (t.A > π/2 ∨ t.B > π/2 ∨ t.C > π/2) :=
sorry

-- Theorem 2: If cos(A-C) * cos(B-C) * cos(C-A) = 1, then the triangle is equilateral
theorem equilateral_triangle (t : Triangle) :
  Real.cos (t.A - t.C) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1 →
  t.A = π/3 ∧ t.B = π/3 ∧ t.C = π/3 :=
sorry

end obtuse_triangle_equilateral_triangle_l2621_262168


namespace total_miles_on_wednesdays_l2621_262174

/-- The total miles flown on Wednesdays over a 4-week period, given that a pilot flies
    the same number of miles each week and x miles each Wednesday. -/
theorem total_miles_on_wednesdays
  (x : ℕ)  -- Miles flown on Wednesday
  (h1 : ∀ week : Fin 4, ∃ miles : ℕ, miles = x)  -- Same miles flown each Wednesday for 4 weeks
  : ∃ total : ℕ, total = 4 * x :=
by sorry

end total_miles_on_wednesdays_l2621_262174


namespace rectangle_area_perimeter_l2621_262194

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: A rectangle with length 5 and width 3 has area 15 and perimeter 16 -/
theorem rectangle_area_perimeter :
  let r : Rectangle := ⟨5, 3⟩
  area r = 15 ∧ perimeter r = 16 := by
  sorry

end rectangle_area_perimeter_l2621_262194


namespace parabola_line_tangency_l2621_262127

theorem parabola_line_tangency (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 ∧ x + y = Real.sqrt m ∧ 
   (∀ x' y' : ℝ, y' = x'^2 → x' + y' = Real.sqrt m → (x', y') = (x, y))) → 
  m = 1/16 := by
sorry

end parabola_line_tangency_l2621_262127


namespace typhoon_tree_difference_l2621_262155

theorem typhoon_tree_difference (initial_trees : ℕ) (survival_rate : ℚ) : 
  initial_trees = 25 → 
  survival_rate = 2/5 → 
  (initial_trees - (survival_rate * initial_trees).floor) - (survival_rate * initial_trees).floor = 5 := by
  sorry

end typhoon_tree_difference_l2621_262155


namespace usb_drive_available_space_l2621_262193

theorem usb_drive_available_space (total_capacity : ℝ) (used_percentage : ℝ) 
  (h1 : total_capacity = 16)
  (h2 : used_percentage = 50)
  : total_capacity * (1 - used_percentage / 100) = 8 := by
  sorry

end usb_drive_available_space_l2621_262193


namespace digit_sum_puzzle_l2621_262164

def DigitSet : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

theorem digit_sum_puzzle (a b c x z : Nat) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ a ≠ z ∧
                b ≠ c ∧ b ≠ x ∧ b ≠ z ∧
                c ≠ x ∧ c ≠ z ∧
                x ≠ z)
  (h_in_set : a ∈ DigitSet ∧ b ∈ DigitSet ∧ c ∈ DigitSet ∧ x ∈ DigitSet ∧ z ∈ DigitSet)
  (h_vertical_sum : a + b + c = 17)
  (h_horizontal_sum : x + b + z = 14) :
  a + b + c + x + z = 26 := by
sorry

end digit_sum_puzzle_l2621_262164


namespace fraction_equivalence_l2621_262114

theorem fraction_equivalence : 
  ∃ (n : ℤ), (4 + n) / (7 + n) = 6 / 7 :=
by
  use 14
  sorry

end fraction_equivalence_l2621_262114


namespace time_difference_steve_jennifer_l2621_262171

/-- Represents the time in minutes for various running distances --/
structure RunningTimes where
  danny_to_steve : ℝ
  jennifer_to_danny : ℝ

/-- Theorem stating the difference in time between Steve and Jennifer reaching their respective halfway points --/
theorem time_difference_steve_jennifer (times : RunningTimes) 
  (h1 : times.danny_to_steve = 35)
  (h2 : times.jennifer_to_danny = 10)
  (h3 : times.jennifer_to_danny * 2 = times.danny_to_steve) : 
  (2 * times.danny_to_steve) / 2 - times.jennifer_to_danny / 2 = 30 := by
  sorry


end time_difference_steve_jennifer_l2621_262171


namespace tangent_intersection_x_coordinate_l2621_262135

/-- Given two circles with radii 3 and 5, centered at (0, 0) and (12, 0) respectively,
    the x-coordinate of the point where a common tangent line intersects the x-axis is 9/2. -/
theorem tangent_intersection_x_coordinate :
  let circle1_radius : ℝ := 3
  let circle1_center : ℝ × ℝ := (0, 0)
  let circle2_radius : ℝ := 5
  let circle2_center : ℝ × ℝ := (12, 0)
  ∃ x : ℝ, x > 0 ∧ 
    (x / (12 - x) = circle1_radius / circle2_radius) ∧
    x = 9 / 2 := by
  sorry

end tangent_intersection_x_coordinate_l2621_262135


namespace existence_of_polynomial_l2621_262102

/-- The polynomial a(x, y) -/
def a (x y : ℝ) : ℝ := x^2 * y + x * y^2

/-- The polynomial b(x, y) -/
def b (x y : ℝ) : ℝ := x^2 + x * y + y^2

/-- The statement to be proved -/
theorem existence_of_polynomial (n : ℕ) : 
  ∃ (p : ℝ → ℝ → ℝ), ∀ (x y : ℝ), 
    p (a x y) (b x y) = (x + y)^n + (-1)^n * (x^n + y^n) := by
  sorry

end existence_of_polynomial_l2621_262102


namespace max_value_of_sin_cos_function_l2621_262184

theorem max_value_of_sin_cos_function :
  ∃ (M : ℝ), M = 17 ∧ ∀ x, 8 * Real.sin x + 15 * Real.cos x ≤ M := by
  sorry

end max_value_of_sin_cos_function_l2621_262184


namespace contour_area_ratio_l2621_262104

theorem contour_area_ratio (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : 0 < A₁) :
  A₂ / A₁ = (r₂ / r₁)^2 :=
sorry

end contour_area_ratio_l2621_262104


namespace range_of_m_l2621_262189

theorem range_of_m (p q : Prop) (h1 : p ↔ ∀ x : ℝ, x^2 - 2*x + 1 - m ≥ 0)
  (h2 : q ↔ ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ a^2 = 1 ∧ b^2 = 1 / (m + 2))
  (h3 : (p ∨ q) ∧ ¬(p ∧ q)) :
  m ≤ -2 ∨ m > 0 := by sorry

end range_of_m_l2621_262189


namespace quadratic_inequality_solution_sets_l2621_262157

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 2) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ -3 < x ∧ x < 1/2 :=
by sorry

end quadratic_inequality_solution_sets_l2621_262157


namespace quadratic_inequality_empty_solution_l2621_262136

theorem quadratic_inequality_empty_solution (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 1 > 0) ↔ -2 < b ∧ b < 2 :=
by sorry

end quadratic_inequality_empty_solution_l2621_262136


namespace B_is_largest_l2621_262196

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem B_is_largest : B > A ∧ B > C := by
  sorry

end B_is_largest_l2621_262196


namespace reciprocal_of_i_l2621_262146

theorem reciprocal_of_i : Complex.I⁻¹ = -Complex.I := by sorry

end reciprocal_of_i_l2621_262146


namespace fraction_sum_equality_l2621_262154

theorem fraction_sum_equality : (1 : ℚ) / 5 * 3 / 7 + 1 / 2 = 41 / 70 := by sorry

end fraction_sum_equality_l2621_262154


namespace ababab_divisible_by_101_l2621_262130

/-- Represents a 6-digit number of the form ababab -/
def ababab_number (a b : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

/-- Theorem stating that 101 is a factor of any ababab number -/
theorem ababab_divisible_by_101 (a b : Nat) (h : 0 < a ∧ a ≤ 9 ∧ b ≤ 9) :
  101 ∣ ababab_number a b :=
sorry

end ababab_divisible_by_101_l2621_262130


namespace problem_solution_l2621_262118

def p (a : ℝ) : Prop := (1 + a)^2 + (1 - a)^2 < 4

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 ≥ 0

theorem problem_solution (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) ↔ a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2 :=
sorry

end problem_solution_l2621_262118


namespace smallest_solution_floor_equation_l2621_262106

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 12 → y ≥ x) ∧
  ⌊x^2⌋ - ⌊x⌋^2 = 12 :=
by sorry

end smallest_solution_floor_equation_l2621_262106


namespace rental_cost_calculation_l2621_262162

def base_daily_rate : ℚ := 30
def per_mile_rate : ℚ := 0.25
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 5
def rental_days : ℕ := 6
def miles_driven : ℕ := 500

def calculate_total_cost : ℚ :=
  let daily_cost := if rental_days > discount_threshold
                    then base_daily_rate * (1 - discount_rate) * rental_days
                    else base_daily_rate * rental_days
  let mileage_cost := per_mile_rate * miles_driven
  daily_cost + mileage_cost

theorem rental_cost_calculation :
  calculate_total_cost = 287 :=
by sorry

end rental_cost_calculation_l2621_262162


namespace divisibility_by_two_iff_last_digit_even_l2621_262112

theorem divisibility_by_two_iff_last_digit_even (a : ℕ) : 
  ∃ b c : ℕ, a = 10 * b + c ∧ c < 10 → (∃ k : ℕ, a = 2 * k ↔ ∃ m : ℕ, c = 2 * m) :=
sorry

end divisibility_by_two_iff_last_digit_even_l2621_262112


namespace policeman_catches_gangster_l2621_262101

-- Define the square
def Square := {p : ℝ × ℝ | -3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the perimeter of the square
def Perimeter := {p : ℝ × ℝ | (p.1 = -3 ∨ p.1 = 3) ∧ -3 ≤ p.2 ∧ p.2 ≤ 3} ∪
                 {p : ℝ × ℝ | (p.2 = -3 ∨ p.2 = 3) ∧ -3 ≤ p.1 ∧ p.1 ≤ 3}

-- Define the center of the square
def Center : ℝ × ℝ := (0, 0)

-- Define a vertex of the square
def Vertex : ℝ × ℝ := (3, 3)

-- Define the theorem
theorem policeman_catches_gangster (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  ∃ (t : ℝ) (p g : ℝ × ℝ), t ≥ 0 ∧ p ∈ Square ∧ g ∈ Perimeter ∧ p = g ↔ u/v > 1/3 :=
sorry

end policeman_catches_gangster_l2621_262101


namespace negative_fraction_comparison_l2621_262183

theorem negative_fraction_comparison : -6/5 > -5/4 := by
  sorry

end negative_fraction_comparison_l2621_262183


namespace road_repair_hours_l2621_262188

/-- Given that 39 persons can repair a road in 12 days working h hours a day,
    and 30 persons working 6 hours a day can complete the same work in 26 days,
    prove that h = 10. -/
theorem road_repair_hours (h : ℝ) : 
  39 * h * 12 = 30 * 6 * 26 → h = 10 := by
sorry

end road_repair_hours_l2621_262188


namespace reflection_across_x_axis_l2621_262139

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis (p : Point) :
  p.x = -3 ∧ p.y = 2 → (reflect_x p).x = -3 ∧ (reflect_x p).y = -2 := by
  sorry

end reflection_across_x_axis_l2621_262139


namespace negation_equivalence_l2621_262131

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_equivalence_l2621_262131


namespace smallest_marble_count_l2621_262153

def is_valid_marble_count (n : ℕ) : Prop :=
  n > 2 ∧ n % 6 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ n % 11 = 2

theorem smallest_marble_count :
  ∃ (n : ℕ), is_valid_marble_count n ∧ ∀ (m : ℕ), is_valid_marble_count m → n ≤ m :=
by
  use 3698
  sorry

end smallest_marble_count_l2621_262153


namespace yvonne_probability_l2621_262179

theorem yvonne_probability (xavier_prob zelda_prob joint_prob : ℝ) :
  xavier_prob = 1/4 →
  zelda_prob = 5/8 →
  joint_prob = 0.0625 →
  ∃ yvonne_prob : ℝ,
    yvonne_prob = 1/16 ∧
    xavier_prob * yvonne_prob * (1 - zelda_prob) = joint_prob :=
by sorry

end yvonne_probability_l2621_262179


namespace intersection_range_l2621_262126

open Set Real

theorem intersection_range (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| + |x + 1| ≤ 3}
  let B : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}
  (A ∩ B).Nonempty → m > -3/2 ∧ m < 3/2 := by
  sorry

end intersection_range_l2621_262126


namespace simple_interest_calculation_l2621_262181

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 15)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 60 :=
by sorry

end simple_interest_calculation_l2621_262181


namespace travel_group_average_age_l2621_262190

theorem travel_group_average_age 
  (num_men : ℕ) 
  (num_women : ℕ) 
  (avg_age_men : ℚ) 
  (avg_age_women : ℚ) 
  (h1 : num_men = 6) 
  (h2 : num_women = 9) 
  (h3 : avg_age_men = 57) 
  (h4 : avg_age_women = 52) :
  (num_men * avg_age_men + num_women * avg_age_women) / (num_men + num_women) = 54 := by
sorry

end travel_group_average_age_l2621_262190


namespace unitsDigitOfSumOfSquares2023_l2621_262129

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitsDigitOfSumOfSquares (n : ℕ) : ℕ :=
  (n * 1 + n * 9 + (n / 2 + n % 2) * 5) % 10

/-- The theorem stating that the units digit of the sum of the squares 
    of the first 2023 odd, positive integers is 5 -/
theorem unitsDigitOfSumOfSquares2023 : 
  unitsDigitOfSumOfSquares 2023 = 5 := by
  sorry

#eval unitsDigitOfSumOfSquares 2023

end unitsDigitOfSumOfSquares2023_l2621_262129


namespace sum_consecutive_odd_integers_mod_18_l2621_262176

def consecutive_odd_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

theorem sum_consecutive_odd_integers_mod_18 (start : ℕ) (h : start = 11065) :
  (consecutive_odd_integers start 9).sum % 18 =
  ([1, 3, 5, 7, 9, 11, 13, 15, 17].map (λ x => x % 18)).sum % 18 := by
  sorry

end sum_consecutive_odd_integers_mod_18_l2621_262176


namespace probability_of_one_each_item_l2621_262152

def drawer_items : ℕ := 8

def total_items : ℕ := 4 * drawer_items

def items_removed : ℕ := 4

def total_combinations : ℕ := Nat.choose total_items items_removed

def favorable_outcomes : ℕ := drawer_items^items_removed

theorem probability_of_one_each_item : 
  (favorable_outcomes : ℚ) / total_combinations = 128 / 1125 := by sorry

end probability_of_one_each_item_l2621_262152


namespace jared_earnings_proof_l2621_262177

/-- The monthly salary of a diploma holder in dollars -/
def diploma_salary : ℕ := 4000

/-- The ratio of a degree holder's salary to a diploma holder's salary -/
def degree_to_diploma_ratio : ℕ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Jared's annual earnings after graduating with a degree -/
def jared_annual_earnings : ℕ := degree_to_diploma_ratio * diploma_salary * months_in_year

theorem jared_earnings_proof :
  jared_annual_earnings = 144000 :=
sorry

end jared_earnings_proof_l2621_262177


namespace total_geckos_sold_l2621_262145

def geckos_sold_last_year : ℕ := 86

theorem total_geckos_sold (geckos_sold_before : ℕ) 
  (h : geckos_sold_before = 2 * geckos_sold_last_year) : 
  geckos_sold_last_year + geckos_sold_before = 258 := by
  sorry

end total_geckos_sold_l2621_262145


namespace horse_speed_around_square_field_l2621_262169

theorem horse_speed_around_square_field 
  (field_area : ℝ) 
  (time_to_run_around : ℝ) 
  (horse_speed : ℝ) : 
  field_area = 400 ∧ 
  time_to_run_around = 4 → 
  horse_speed = 20 :=
by
  sorry

end horse_speed_around_square_field_l2621_262169


namespace min_value_of_f_l2621_262134

open Real

noncomputable def f (x : ℝ) := 2 * x - log x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (exp 1) ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 0 (exp 1) → f y ≥ f x) ∧
  f x = 1 + log 2 := by
  sorry

end min_value_of_f_l2621_262134


namespace initial_payment_calculation_l2621_262191

theorem initial_payment_calculation (car_cost installment_amount : ℕ) (num_installments : ℕ) 
  (h1 : car_cost = 18000)
  (h2 : installment_amount = 2500)
  (h3 : num_installments = 6) :
  car_cost - (num_installments * installment_amount) = 3000 := by
  sorry

end initial_payment_calculation_l2621_262191


namespace problem_solution_l2621_262122

theorem problem_solution (x y : ℝ) 
  (h1 : x * y + x + y = 17) 
  (h2 : x^2 * y + x * y^2 = 66) : 
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 := by
  sorry

end problem_solution_l2621_262122


namespace parabola_chord_constant_sum_l2621_262144

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def parabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Point C on the y-axis -/
def C : Point :=
  ⟨0, 2⟩

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The theorem to be proved -/
theorem parabola_chord_constant_sum :
  ∀ A B : Point,
  parabola A → parabola B →
  (C.y - A.y) / (C.x - A.x) = (B.y - A.y) / (B.x - A.x) →
  (1 / distanceSquared A C + 1 / distanceSquared B C) = 3/2 :=
by sorry

end parabola_chord_constant_sum_l2621_262144


namespace probability_two_non_defective_pens_l2621_262156

/-- Represents the probability of selecting non-defective pens from a box -/
def probability_non_defective (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) : ℚ :=
  let non_defective := total_pens - defective_pens
  (non_defective.choose selected_pens : ℚ) / (total_pens.choose selected_pens)

/-- Theorem stating the probability of selecting 2 non-defective pens from a box of 16 pens with 3 defective pens -/
theorem probability_two_non_defective_pens :
  probability_non_defective 16 3 2 = 13/20 := by
  sorry

#eval probability_non_defective 16 3 2

end probability_two_non_defective_pens_l2621_262156


namespace power_equality_l2621_262140

theorem power_equality (q : ℕ) : 81^10 = 3^q → q = 40 := by
  sorry

end power_equality_l2621_262140


namespace brother_scores_double_l2621_262108

/-- Represents the hockey goal scoring scenario of Louie and his brother -/
structure HockeyScenario where
  louie_last_match : ℕ
  louie_previous : ℕ
  brother_seasons : ℕ
  games_per_season : ℕ
  total_goals : ℕ

/-- The ratio of Louie's brother's goals per game to Louie's goals in the last match -/
def brother_to_louie_ratio (h : HockeyScenario) : ℚ :=
  let brother_total_games := h.brother_seasons * h.games_per_season
  let brother_total_goals := h.total_goals - (h.louie_last_match + h.louie_previous)
  (brother_total_goals / brother_total_games : ℚ) / h.louie_last_match

/-- The main theorem stating the ratio is 2:1 -/
theorem brother_scores_double (h : HockeyScenario) 
    (h_louie_last : h.louie_last_match = 4)
    (h_louie_prev : h.louie_previous = 40)
    (h_seasons : h.brother_seasons = 3)
    (h_games : h.games_per_season = 50)
    (h_total : h.total_goals = 1244) : 
  brother_to_louie_ratio h = 2 := by
  sorry

end brother_scores_double_l2621_262108


namespace hyperbola_eccentricity_l2621_262100

/-- The eccentricity of a hyperbola with the given properties is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let asymptote₁ : ℝ → ℝ := λ x => (b / a) * x
  let asymptote₂ : ℝ → ℝ := λ x => -(b / a) * x
  ∃ (G H : ℝ × ℝ) (c : ℝ),
    (∃ x, G.1 = x ∧ G.2 = asymptote₁ x) ∧ 
    (∃ x, H.1 = x ∧ H.2 = asymptote₂ x) ∧
    (G.2 - F₁.2) * (G.1 - F₂.1) = -(G.1 - F₁.1) * (G.2 - F₂.2) ∧
    H = ((G.1 + F₁.1) / 2, (G.2 + F₁.2) / 2) →
    c = 2 * a :=
by
  sorry

end hyperbola_eccentricity_l2621_262100


namespace correct_stratified_sample_l2621_262180

/-- Represents the number of people in each stratum -/
structure Strata :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)
  (remaining : ℕ)

/-- Represents the sample sizes for each stratum -/
structure Sample :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)
  (remaining : ℕ)

/-- Calculates the total population size -/
def totalPopulation (s : Strata) : ℕ :=
  s.senior + s.intermediate + s.junior + s.remaining

/-- Checks if the sample sizes are proportional to the strata sizes -/
def isProportionalSample (strata : Strata) (sample : Sample) (totalSampleSize : ℕ) : Prop :=
  let total := totalPopulation strata
  sample.senior * total = strata.senior * totalSampleSize ∧
  sample.intermediate * total = strata.intermediate * totalSampleSize ∧
  sample.junior * total = strata.junior * totalSampleSize ∧
  sample.remaining * total = strata.remaining * totalSampleSize

/-- Theorem: The given sample sizes are proportional for the given strata -/
theorem correct_stratified_sample :
  let strata : Strata := ⟨160, 320, 200, 120⟩
  let sample : Sample := ⟨8, 16, 10, 6⟩
  let totalSampleSize : ℕ := 40
  totalPopulation strata = 800 →
  isProportionalSample strata sample totalSampleSize :=
sorry

end correct_stratified_sample_l2621_262180


namespace real_roots_imply_b_equals_one_l2621_262167

theorem real_roots_imply_b_equals_one (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*I*x + b = 1) → b = 1 := by
  sorry

end real_roots_imply_b_equals_one_l2621_262167


namespace shaniqua_earnings_l2621_262124

/-- Calculates the total earnings for Shaniqua's hair services -/
def total_earnings (haircut_price : ℕ) (style_price : ℕ) (num_haircuts : ℕ) (num_styles : ℕ) : ℕ :=
  haircut_price * num_haircuts + style_price * num_styles

/-- Proves that Shaniqua's total earnings for 8 haircuts and 5 styles are $221 -/
theorem shaniqua_earnings : total_earnings 12 25 8 5 = 221 := by
  sorry

end shaniqua_earnings_l2621_262124


namespace exists_valid_coloring_l2621_262121

/-- A coloring function that satisfies the given conditions -/
def valid_coloring (n : ℕ) (S : Finset ℕ) (f : Finset ℕ → Fin 8) : Prop :=
  (S.card = 3 * n) ∧
  ∀ A B C : Finset ℕ,
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S →
    A.card = n ∧ B.card = n ∧ C.card = n →
    A ≠ B ∧ A ≠ C ∧ B ≠ C →
    (A ∩ B).card ≤ 1 ∧ (A ∩ C).card ≤ 1 ∧ (B ∩ C).card ≤ 1 →
    f A ≠ f B ∨ f A ≠ f C ∨ f B ≠ f C

/-- There exists a valid coloring for any set S with 3n elements -/
theorem exists_valid_coloring (n : ℕ) :
  ∀ S : Finset ℕ, S.card = 3 * n → ∃ f : Finset ℕ → Fin 8, valid_coloring n S f := by
  sorry

end exists_valid_coloring_l2621_262121


namespace intersection_empty_iff_k_greater_than_six_l2621_262192

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 7}
def N (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2*k - 1}

theorem intersection_empty_iff_k_greater_than_six (k : ℝ) : 
  M ∩ N k = ∅ ↔ k > 6 := by sorry

end intersection_empty_iff_k_greater_than_six_l2621_262192


namespace second_class_size_l2621_262195

theorem second_class_size (students1 : ℕ) (avg1 : ℕ) (avg2 : ℕ) (avg_total : ℕ) :
  students1 = 12 →
  avg1 = 40 →
  avg2 = 60 →
  avg_total = 54 →
  ∃ students2 : ℕ, 
    students2 = 28 ∧
    (students1 * avg1 + students2 * avg2) = (students1 + students2) * avg_total :=
by sorry


end second_class_size_l2621_262195


namespace red_black_red_probability_l2621_262138

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of red cards in a standard deck -/
def RedCards : ℕ := 26

/-- Number of black cards in a standard deck -/
def BlackCards : ℕ := 26

/-- Probability of drawing a red card, then a black card, then a red card from a standard deck -/
theorem red_black_red_probability :
  (RedCards : ℚ) * BlackCards * (RedCards - 1) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 13 / 102 := by
  sorry

end red_black_red_probability_l2621_262138


namespace ellipse_chord_slope_l2621_262161

/-- Given an ellipse with equation 4x^2 + 9y^2 = 144 containing a point P(3, 2),
    the slope of the line containing the chord with P as its midpoint is -2/3. -/
theorem ellipse_chord_slope :
  let ellipse := {(x, y) : ℝ × ℝ | 4 * x^2 + 9 * y^2 = 144}
  let P : ℝ × ℝ := (3, 2)
  P ∈ ellipse →
  ∃ (A B : ℝ × ℝ),
    A ∈ ellipse ∧ B ∈ ellipse ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (B.2 - A.2) / (B.1 - A.1) = -2/3 :=
by sorry

end ellipse_chord_slope_l2621_262161


namespace det_eq_ten_l2621_262123

/-- The matrix for which we need to calculate the determinant -/
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 3, 2*x]

/-- The theorem stating the condition for the determinant to be 10 -/
theorem det_eq_ten (x : ℝ) : 
  Matrix.det (A x) = 10 ↔ x = Real.sqrt (8/3) ∨ x = -Real.sqrt (8/3) := by
  sorry

end det_eq_ten_l2621_262123


namespace equation_solution_l2621_262197

theorem equation_solution : ∃ f : ℝ, 
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - 6) / f = (2 * 0.3 + 4) / 3) ∧ 
  (abs (f - 18) < 0.01) := by
  sorry

end equation_solution_l2621_262197


namespace percentage_sum_equality_l2621_262109

theorem percentage_sum_equality : 
  (25 / 100 * 2018) + (2018 / 100 * 25) = 1009 := by
  sorry

end percentage_sum_equality_l2621_262109


namespace set_intersection_theorem_l2621_262125

def A : Set ℝ := {x | x ≤ 2}
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 2 < x ∧ x ≤ 3} := by sorry

end set_intersection_theorem_l2621_262125


namespace exactly_one_common_course_l2621_262120

/-- The number of ways two people can choose 2 courses each from 4 courses with exactly one course in common -/
def common_course_choices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose n k - Nat.choose n k - Nat.choose n k

theorem exactly_one_common_course :
  common_course_choices 4 2 = 24 := by
  sorry

end exactly_one_common_course_l2621_262120


namespace percentage_reduction_l2621_262170

theorem percentage_reduction (initial : ℝ) (increase_percent : ℝ) (final : ℝ) : 
  initial = 1500 →
  increase_percent = 20 →
  final = 1080 →
  let increased := initial * (1 + increase_percent / 100)
  (increased - final) / increased * 100 = 40 := by
sorry

end percentage_reduction_l2621_262170


namespace max_salary_is_400000_l2621_262151

/-- Represents a baseball team -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  maxTotalSalary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def maxSinglePlayerSalary (team : BaseballTeam) : ℕ :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem: The maximum salary for a single player in the given conditions is $400,000 -/
theorem max_salary_is_400000 (team : BaseballTeam)
  (h1 : team.players = 21)
  (h2 : team.minSalary = 15000)
  (h3 : team.maxTotalSalary = 700000) :
  maxSinglePlayerSalary team = 400000 := by
  sorry

#eval maxSinglePlayerSalary { players := 21, minSalary := 15000, maxTotalSalary := 700000 }

end max_salary_is_400000_l2621_262151


namespace f_leq_one_iff_a_range_l2621_262148

-- Define the function f
def f (a x : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- State the theorem
theorem f_leq_one_iff_a_range (a : ℝ) :
  (∀ x, f a x ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) :=
sorry

end f_leq_one_iff_a_range_l2621_262148


namespace sector_area_l2621_262143

/-- Given a sector with central angle 2 radians and arc length 4, its area is 4. -/
theorem sector_area (θ : Real) (l : Real) (A : Real) : 
  θ = 2 → l = 4 → A = (1/2) * (l/θ)^2 * θ → A = 4 := by
  sorry

end sector_area_l2621_262143


namespace haley_tree_count_l2621_262149

/-- The number of trees Haley has after growing some, losing some to a typhoon, and growing more. -/
def final_tree_count (initial : ℕ) (lost : ℕ) (new : ℕ) : ℕ :=
  initial - lost + new

/-- Theorem stating that with 9 initial trees, 4 lost, and 5 new, the final count is 10. -/
theorem haley_tree_count : final_tree_count 9 4 5 = 10 := by
  sorry

end haley_tree_count_l2621_262149


namespace point_in_unit_circle_l2621_262128

theorem point_in_unit_circle (z : ℂ) (h : Complex.abs z ≤ 1) :
  (z.re)^2 + (z.im)^2 ≤ 1 := by
sorry

end point_in_unit_circle_l2621_262128


namespace prob_A_and_B_selected_is_three_tenths_l2621_262159

def total_students : ℕ := 5
def students_to_select : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (total_students - students_to_select + 1 : ℚ) / (total_students.choose students_to_select)

theorem prob_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end prob_A_and_B_selected_is_three_tenths_l2621_262159


namespace initial_bags_calculation_l2621_262172

/-- Given the total number of cookies, total number of candies, and current number of bags,
    calculate the initial number of bags. -/
def initialBags (totalCookies : ℕ) (totalCandies : ℕ) (currentBags : ℕ) : ℕ :=
  sorry

theorem initial_bags_calculation (totalCookies totalCandies currentBags : ℕ) 
    (h1 : totalCookies = 28)
    (h2 : totalCandies = 86)
    (h3 : currentBags = 2)
    (h4 : totalCookies % currentBags = 0)  -- Ensures equal distribution of cookies
    (h5 : totalCandies % (initialBags totalCookies totalCandies currentBags) = 0)  -- Ensures equal distribution of candies
    (h6 : totalCookies / currentBags = totalCandies / (initialBags totalCookies totalCandies currentBags))  -- Cookies per bag equals candies per bag
    : initialBags totalCookies totalCandies currentBags = 6 :=
  sorry

end initial_bags_calculation_l2621_262172


namespace angle_C_value_angle_C_range_l2621_262142

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides

-- Define the function f
def f (x : Real) : Real := a^2 * x^2 - (a^2 - b^2) * x - 4 * c^2

-- Theorem 1
theorem angle_C_value (h1 : f 1 = 0) (h2 : B - C = π/3) : C = π/6 := by
  sorry

-- Theorem 2
theorem angle_C_range (h : f 2 = 0) : 0 < C ∧ C ≤ π/3 := by
  sorry

end

end angle_C_value_angle_C_range_l2621_262142


namespace payment_problem_l2621_262166

/-- The payment problem -/
theorem payment_problem (a_days b_days total_days : ℕ) (total_payment : ℚ) : 
  a_days = 6 →
  b_days = 8 →
  total_days = 3 →
  total_payment = 3680 →
  let a_work_per_day : ℚ := 1 / a_days
  let b_work_per_day : ℚ := 1 / b_days
  let ab_work_in_total_days : ℚ := (a_work_per_day + b_work_per_day) * total_days
  let c_work : ℚ := 1 - ab_work_in_total_days
  let c_payment : ℚ := c_work * total_payment
  c_payment = 460 :=
sorry

end payment_problem_l2621_262166
