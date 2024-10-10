import Mathlib

namespace equiv_class_characterization_l3862_386283

/-- Given a positive integer m and an integer a, this theorem states that 
    an integer b is in the equivalence class of a modulo m if and only if 
    there exists an integer t such that b = m * t + a. -/
theorem equiv_class_characterization (m : ℕ+) (a b : ℤ) : 
  b ≡ a [ZMOD m] ↔ ∃ t : ℤ, b = m * t + a := by sorry

end equiv_class_characterization_l3862_386283


namespace outfits_count_l3862_386235

/-- The number of different outfits that can be created -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem stating the number of outfits given specific quantities of clothing items -/
theorem outfits_count :
  number_of_outfits 8 4 3 2 = 360 :=
by sorry

end outfits_count_l3862_386235


namespace perpendicular_implies_parallel_l3862_386208

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the given conditions
variable (l m n : Line)
variable (α β γ : Plane)

variable (different_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- State the theorem
theorem perpendicular_implies_parallel 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) : 
  parallel α β :=
sorry

end perpendicular_implies_parallel_l3862_386208


namespace questionnaires_from_unit_d_l3862_386271

/-- Represents the number of questionnaires drawn from each unit -/
structure SampledQuestionnaires :=
  (a b c d : ℕ)

/-- Represents the total number of questionnaires collected from each unit -/
structure TotalQuestionnaires :=
  (a b c d : ℕ)

/-- The properties of the survey and sampling -/
class SurveyProperties (sampled : SampledQuestionnaires) (total : TotalQuestionnaires) :=
  (total_sum : total.a + total.b + total.c + total.d = 1000)
  (sample_sum : sampled.a + sampled.b + sampled.c + sampled.d = 150)
  (total_arithmetic : ∃ (r : ℕ), total.b = total.a + r ∧ total.c = total.b + r ∧ total.d = total.c + r)
  (sample_arithmetic : ∃ (d : ℤ), sampled.b = sampled.a + d ∧ sampled.c = sampled.b + d ∧ sampled.d = sampled.c + d)
  (unit_b_sample : sampled.b = 30)
  (stratified_sampling : ∀ (x y : Fin 4), 
    (sampled.a * total.b = sampled.b * total.a) ∧
    (sampled.b * total.c = sampled.c * total.b) ∧
    (sampled.c * total.d = sampled.d * total.c))

/-- The main theorem to prove -/
theorem questionnaires_from_unit_d 
  (sampled : SampledQuestionnaires) 
  (total : TotalQuestionnaires) 
  [SurveyProperties sampled total] : 
  sampled.d = 60 := by
  sorry

end questionnaires_from_unit_d_l3862_386271


namespace fib_matrix_power_eq_fib_relation_l3862_386202

/-- Fibonacci matrix -/
def fib_matrix : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 0]

/-- n-th power of Fibonacci matrix -/
def fib_matrix_power (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ := fib_matrix ^ n

/-- n-th Fibonacci number -/
def F (n : ℕ) : ℕ := (fib_matrix_power n) 0 1

/-- Theorem stating the relation between Fibonacci numbers and matrix power -/
theorem fib_matrix_power_eq (n : ℕ) :
  fib_matrix_power n = !![F (n + 1), F n; F n, F (n - 1)] := by sorry

/-- Main theorem to prove -/
theorem fib_relation :
  F 1001 * F 1003 - F 1002 * F 1002 = 1 := by sorry

end fib_matrix_power_eq_fib_relation_l3862_386202


namespace car_selling_price_l3862_386206

/-- Calculates the selling price of a car given its purchase price, repair costs, and profit percentage. -/
def selling_price (purchase_price repair_costs profit_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let profit := (profit_percent / 100) * total_cost
  total_cost + profit

/-- Theorem stating that for the given conditions, the selling price is 64900. -/
theorem car_selling_price :
  selling_price 42000 8000 29.8 = 64900 := by
  sorry

end car_selling_price_l3862_386206


namespace problem_classification_l3862_386290

-- Define a type for problems
inductive Problem
  | EquilateralTrianglePerimeter
  | ArithmeticMean
  | SmallerOfTwo
  | PiecewiseFunction

-- Define a function to determine if a problem requires a conditional statement
def requiresConditionalStatement (p : Problem) : Prop :=
  match p with
  | Problem.EquilateralTrianglePerimeter => False
  | Problem.ArithmeticMean => False
  | Problem.SmallerOfTwo => True
  | Problem.PiecewiseFunction => True

-- Theorem statement
theorem problem_classification :
  (¬ requiresConditionalStatement Problem.EquilateralTrianglePerimeter) ∧
  (¬ requiresConditionalStatement Problem.ArithmeticMean) ∧
  (requiresConditionalStatement Problem.SmallerOfTwo) ∧
  (requiresConditionalStatement Problem.PiecewiseFunction) :=
by sorry

end problem_classification_l3862_386290


namespace evaluate_expression_l3862_386216

theorem evaluate_expression : (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 := by
  sorry

end evaluate_expression_l3862_386216


namespace elevator_weight_problem_l3862_386243

/-- Given an elevator scenario, prove the initial average weight --/
theorem elevator_weight_problem (initial_people : ℕ) (new_person_weight : ℕ) (new_average : ℕ) :
  initial_people = 6 →
  new_person_weight = 145 →
  new_average = 151 →
  (initial_people * (initial_average : ℕ) + new_person_weight) / (initial_people + 1) = new_average →
  initial_average = 152 :=
by
  sorry

#check elevator_weight_problem

end elevator_weight_problem_l3862_386243


namespace min_delivery_time_75_minutes_l3862_386256

/-- Represents the train's cargo and delivery constraints -/
structure TrainDelivery where
  coal_cars : ℕ
  iron_cars : ℕ
  wood_cars : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  travel_time : ℕ

/-- Calculates the minimum number of stops required to deliver all cars -/
def min_stops (td : TrainDelivery) : ℕ :=
  max (td.coal_cars / td.max_coal_deposit)
      (max (td.iron_cars / td.max_iron_deposit)
           (td.wood_cars / td.max_wood_deposit))

/-- Calculates the total delivery time based on the number of stops -/
def total_delivery_time (td : TrainDelivery) : ℕ :=
  (min_stops td - 1) * td.travel_time

/-- The main theorem stating the minimum time required for delivery -/
theorem min_delivery_time_75_minutes (td : TrainDelivery) 
  (h1 : td.coal_cars = 6)
  (h2 : td.iron_cars = 12)
  (h3 : td.wood_cars = 2)
  (h4 : td.max_coal_deposit = 2)
  (h5 : td.max_iron_deposit = 3)
  (h6 : td.max_wood_deposit = 1)
  (h7 : td.travel_time = 25) :
  total_delivery_time td = 75 := by
  sorry

#eval total_delivery_time {
  coal_cars := 6,
  iron_cars := 12,
  wood_cars := 2,
  max_coal_deposit := 2,
  max_iron_deposit := 3,
  max_wood_deposit := 1,
  travel_time := 25
}

end min_delivery_time_75_minutes_l3862_386256


namespace tuesday_wednesday_thursday_avg_l3862_386254

def tuesday_temp : ℝ := 38
def friday_temp : ℝ := 44
def wed_thur_fri_avg : ℝ := 34

theorem tuesday_wednesday_thursday_avg :
  let wed_thur_sum := 3 * wed_thur_fri_avg - friday_temp
  (tuesday_temp + wed_thur_sum) / 3 = 32 :=
by sorry

end tuesday_wednesday_thursday_avg_l3862_386254


namespace dairy_farm_husk_consumption_l3862_386276

/-- If 46 cows eat 46 bags of husk in 46 days, then one cow will eat one bag of husk in 46 days. -/
theorem dairy_farm_husk_consumption 
  (cows : ℕ) (bags : ℕ) (days : ℕ) (one_cow_days : ℕ) :
  cows = 46 → bags = 46 → days = 46 → 
  (cows * bags = cows * days) →
  one_cow_days = 46 :=
by sorry

end dairy_farm_husk_consumption_l3862_386276


namespace union_equals_B_l3862_386210

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x - a ≤ 0}

-- State the theorem
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a ≥ 2 := by
  sorry

end union_equals_B_l3862_386210


namespace machine_work_time_l3862_386280

theorem machine_work_time (x : ℝ) 
  (h1 : (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x)) = 1 / x) 
  (h2 : x > 0) : x = 2/3 := by
  sorry

end machine_work_time_l3862_386280


namespace h_range_l3862_386266

-- Define the function h
def h (x : ℝ) : ℝ := 3 * (x - 5)

-- State the theorem
theorem h_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -9 ∧ h x = y} = {y : ℝ | y < -42 ∨ y > -42} :=
by sorry

end h_range_l3862_386266


namespace robert_claire_photo_difference_l3862_386225

/-- 
Given that:
- Lisa and Robert have taken the same number of photos
- Lisa has taken 3 times as many photos as Claire
- Claire has taken 6 photos

Prove that Robert has taken 12 more photos than Claire.
-/
theorem robert_claire_photo_difference : 
  ∀ (lisa robert claire : ℕ),
  robert = lisa →
  lisa = 3 * claire →
  claire = 6 →
  robert - claire = 12 :=
by sorry

end robert_claire_photo_difference_l3862_386225


namespace triangle_tan_A_l3862_386250

theorem triangle_tan_A (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / b = (b + Real.sqrt 3 * c) / a →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  Real.tan A = Real.sqrt 3 / 3 :=
by sorry

end triangle_tan_A_l3862_386250


namespace gcd_840_1764_l3862_386232

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l3862_386232


namespace linear_function_property_l3862_386224

/-- Given a linear function f(x) = ax + b, if f(1) = 2 and f'(1) = 2, then f(2) = 4 -/
theorem linear_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x + b)
    (h2 : f 1 = 2)
    (h3 : (deriv f) 1 = 2) : 
  f 2 = 4 := by
  sorry

end linear_function_property_l3862_386224


namespace solution_set_inequality_l3862_386247

theorem solution_set_inequality (x : ℝ) : 
  (1 / x > 1 / (x - 1)) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_inequality_l3862_386247


namespace pyramid_width_height_difference_l3862_386261

/-- The Great Pyramid of Giza's dimensions --/
structure PyramidDimensions where
  height : ℝ
  width : ℝ
  height_is_520 : height = 520
  width_greater_than_height : width > height
  sum_of_dimensions : height + width = 1274

/-- The difference between the width and height of the pyramid is 234 feet --/
theorem pyramid_width_height_difference (p : PyramidDimensions) : 
  p.width - p.height = 234 := by
sorry

end pyramid_width_height_difference_l3862_386261


namespace number_equation_solution_l3862_386222

theorem number_equation_solution : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108/19 := by
  sorry

end number_equation_solution_l3862_386222


namespace log_2_base_10_upper_bound_l3862_386251

theorem log_2_base_10_upper_bound (h1 : 10^3 = 1000) (h2 : 10^4 = 10000)
  (h3 : 2^9 = 512) (h4 : 2^11 = 2048) : Real.log 2 / Real.log 10 < 4/11 := by
  sorry

end log_2_base_10_upper_bound_l3862_386251


namespace cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero_l3862_386248

theorem cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero (α k : ℝ) : 
  (∃ x y : ℝ, x = Real.tan α ∧ y = (Real.tan α)⁻¹ ∧ 
   x^2 - k*x + k^2 - 3 = 0 ∧ y^2 - k*y + k^2 - 3 = 0) →
  3*Real.pi < α ∧ α < (7/2)*Real.pi →
  Real.cos (3*Real.pi + α) - Real.sin (Real.pi + α) = 0 := by
sorry

end cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero_l3862_386248


namespace probability_triangle_or_square_l3862_386240

theorem probability_triangle_or_square (total_figures : ℕ) 
  (triangle_count : ℕ) (square_count : ℕ) :
  total_figures = 10 →
  triangle_count = 3 →
  square_count = 4 →
  (triangle_count + square_count : ℚ) / total_figures = 7 / 10 := by
sorry

end probability_triangle_or_square_l3862_386240


namespace equation_solution_l3862_386282

/-- The solutions to the equation (8y^2 + 135y + 5) / (3y + 35) = 4y + 2 -/
theorem equation_solution : 
  let y₁ : ℂ := (-11 + Complex.I * Real.sqrt 919) / 8
  let y₂ : ℂ := (-11 - Complex.I * Real.sqrt 919) / 8
  ∀ y : ℂ, (8 * y^2 + 135 * y + 5) / (3 * y + 35) = 4 * y + 2 ↔ y = y₁ ∨ y = y₂ :=
by sorry

#check equation_solution

end equation_solution_l3862_386282


namespace set_equality_implies_a_values_l3862_386298

theorem set_equality_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}
  let B : Set ℝ := {y | ∃ x ∈ A, y = x + 1}
  let C : Set ℝ := {y | ∃ x ∈ A, y = x^2}
  A.Nonempty → B = C → a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := by
  sorry

end set_equality_implies_a_values_l3862_386298


namespace quadratic_trinomial_characterization_l3862_386294

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: Replacing any coefficient with 1 results in a trinomial with exactly one root -/
def has_one_root_when_replaced (qt : QuadraticTrinomial) : Prop :=
  (qt.b^2 - 4*qt.c = 0) ∧ 
  (1 - 4*qt.a*qt.c = 0) ∧ 
  (qt.b^2 - 4*qt.a = 0)

/-- Theorem: Characterization of quadratic trinomials satisfying the condition -/
theorem quadratic_trinomial_characterization (qt : QuadraticTrinomial) :
  has_one_root_when_replaced qt →
  (qt.a = 1/2 ∧ qt.c = 1/2 ∧ (qt.b = Real.sqrt 2 ∨ qt.b = -Real.sqrt 2)) :=
by sorry

end quadratic_trinomial_characterization_l3862_386294


namespace product_equals_eight_l3862_386265

theorem product_equals_eight (x : ℝ) (hx : x ≠ 0) : 
  ∃ y : ℝ, x * y = 8 ∧ y = 8 / x := by sorry

end product_equals_eight_l3862_386265


namespace football_count_white_patch_count_l3862_386297

/- Define the number of students -/
def num_students : ℕ := 36

/- Define the number of footballs -/
def num_footballs : ℕ := 27

/- Define the number of black patches -/
def num_black_patches : ℕ := 12

/- Define the number of white patches -/
def num_white_patches : ℕ := 20

/- Theorem for the number of footballs -/
theorem football_count : 
  (num_students - 9 = num_footballs) ∧ 
  (num_students / 2 + 9 = num_footballs) := by
  sorry

/- Theorem for the number of white patches -/
theorem white_patch_count :
  2 * num_black_patches * 5 = 6 * num_white_patches := by
  sorry

end football_count_white_patch_count_l3862_386297


namespace coffee_table_price_is_330_l3862_386203

/-- The price of the coffee table in a living room set purchase --/
def coffee_table_price (sofa_price armchair_price total_invoice : ℕ) (num_armchairs : ℕ) : ℕ :=
  total_invoice - (sofa_price + num_armchairs * armchair_price)

/-- Theorem stating the price of the coffee table in the given scenario --/
theorem coffee_table_price_is_330 :
  coffee_table_price 1250 425 2430 2 = 330 := by
  sorry

end coffee_table_price_is_330_l3862_386203


namespace symmetric_line_equation_l3862_386214

/-- The parabola that touches the x-axis at one point -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x + 2

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- The x-intercept of the parabola -/
def x_intercept : ℝ := -1

/-- The y-intercept of the parabola -/
def y_intercept : ℝ := 2

/-- The slope of the line symmetric to the line joining x-intercept and y-intercept -/
def symmetric_line_slope : ℝ := -2

/-- The y-intercept of the line symmetric to the line joining x-intercept and y-intercept -/
def symmetric_line_y_intercept : ℝ := -2

/-- Theorem stating that the symmetric line has the equation y = -2x - 2 -/
theorem symmetric_line_equation :
  ∀ x y : ℝ, y = symmetric_line_slope * x + symmetric_line_y_intercept ↔ 
  y = -2 * x - 2 :=
sorry

end symmetric_line_equation_l3862_386214


namespace rectangle_tiling_l3862_386215

theorem rectangle_tiling (a b m n : ℕ) (h : a > 0 ∧ b > 0 ∧ m > 0 ∧ n > 0) :
  (∃ (v h : ℕ → ℕ → ℕ), ∀ (i j : ℕ), i < m ∧ j < n →
    (v i j = b ∧ h i j = 0) ∨ (v i j = 0 ∧ h i j = a)) →
  b ∣ m ∨ a ∣ n := by
  sorry


end rectangle_tiling_l3862_386215


namespace three_digit_factorial_sum_l3862_386279

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def digit_factorial_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map factorial).sum

def contains_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d ∈ n.digits 10

theorem three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    contains_digits n [7, 2, 1] ∧
    n = digit_factorial_sum n :=
  sorry

end three_digit_factorial_sum_l3862_386279


namespace man_mass_from_boat_displacement_l3862_386227

/-- Calculates the mass of a man based on the displacement of a boat -/
theorem man_mass_from_boat_displacement (boat_length boat_breadth additional_depth water_density : ℝ) 
  (h1 : boat_length = 4)
  (h2 : boat_breadth = 2)
  (h3 : additional_depth = 0.01)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * additional_depth * water_density = 80 := by
  sorry

end man_mass_from_boat_displacement_l3862_386227


namespace second_team_soup_amount_l3862_386211

/-- Given the total required amount of soup and the amounts made by the first and third teams,
    calculate the amount the second team should prepare. -/
theorem second_team_soup_amount (total_required : ℕ) (first_team : ℕ) (third_team : ℕ) :
  total_required = 280 →
  first_team = 90 →
  third_team = 70 →
  total_required - (first_team + third_team) = 120 := by
sorry

end second_team_soup_amount_l3862_386211


namespace ab_power_2022_l3862_386291

theorem ab_power_2022 (a b : ℝ) (h : (a - 1/2)^2 + |b + 2| = 0) : (a * b)^2022 = 1 := by
  sorry

end ab_power_2022_l3862_386291


namespace seventh_term_equals_33_l3862_386293

/-- An arithmetic sequence with 15 terms, first term 3, and last term 72 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  3 + (n - 1) * ((72 - 3) / 14)

/-- The 7th term of the arithmetic sequence -/
def seventh_term : ℚ := arithmetic_sequence 7

theorem seventh_term_equals_33 : ⌊seventh_term⌋ = 33 := by
  sorry

end seventh_term_equals_33_l3862_386293


namespace alex_makes_100_dresses_l3862_386287

/-- Given the initial amount of silk, silk given to friends, and silk required per dress,
    calculate the number of dresses Alex can make. -/
def dresses_alex_can_make (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (silk_per_dress : ℕ) : ℕ :=
  (initial_silk - friends * silk_per_friend) / silk_per_dress

/-- Prove that Alex can make 100 dresses given the conditions. -/
theorem alex_makes_100_dresses :
  dresses_alex_can_make 600 5 20 5 = 100 := by
  sorry

end alex_makes_100_dresses_l3862_386287


namespace functional_equation_identity_l3862_386233

theorem functional_equation_identity (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) → 
  (∀ x : ℝ, f x = x) := by
sorry

end functional_equation_identity_l3862_386233


namespace fourth_selected_is_48_l3862_386218

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total : ℕ
  sample_size : ℕ
  first_three : Fin 3 → ℕ

/-- Calculates the interval for systematic sampling -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total / s.sample_size

/-- Theorem: In the given systematic sampling scenario, the fourth selected number is 48 -/
theorem fourth_selected_is_48 (s : SystematicSampling) 
  (h_total : s.total = 60)
  (h_sample_size : s.sample_size = 4)
  (h_first_three : s.first_three = ![3, 18, 33]) :
  s.first_three 2 + sampling_interval s = 48 := by
  sorry

end fourth_selected_is_48_l3862_386218


namespace brendas_blisters_l3862_386200

theorem brendas_blisters (blisters_per_arm : ℕ) : 
  (2 * blisters_per_arm + 80 = 200) → blisters_per_arm = 60 := by
  sorry

end brendas_blisters_l3862_386200


namespace trig_identity_proof_l3862_386272

theorem trig_identity_proof (α : Real) (h : Real.tan α = 3) : 
  (Real.cos (α + π/4))^2 - (Real.cos (α - π/4))^2 = -3/5 := by
  sorry

end trig_identity_proof_l3862_386272


namespace box_length_proof_l3862_386221

/-- Proves that a rectangular box with given dimensions has a length of 55.5 meters -/
theorem box_length_proof (width : ℝ) (road_width : ℝ) (lawn_area : ℝ) :
  width = 40 →
  road_width = 3 →
  lawn_area = 2109 →
  ∃ (length : ℝ),
    length * width - 2 * (length / 3) * road_width = lawn_area ∧
    length = 55.5 := by
  sorry

end box_length_proof_l3862_386221


namespace rod_length_proof_l3862_386264

/-- Given that a 12-meter long rod weighs 14 kg, prove that a rod weighing 7 kg is 6 meters long -/
theorem rod_length_proof (weight_per_meter : ℝ) (h1 : weight_per_meter = 14 / 12) : 
  7 / weight_per_meter = 6 := by
  sorry

end rod_length_proof_l3862_386264


namespace max_intersections_circle_line_parabola_l3862_386245

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in a plane -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum number of intersection points between a circle and a line -/
def max_intersections_circle_line : ℕ := 2

/-- The maximum number of intersection points between a parabola and a line -/
def max_intersections_parabola_line : ℕ := 2

/-- The maximum number of intersection points between a circle and a parabola -/
def max_intersections_circle_parabola : ℕ := 4

/-- Theorem: The maximum number of intersection points among a circle, a line, and a parabola on a plane is 8 -/
theorem max_intersections_circle_line_parabola :
  ∀ (c : Circle) (l : Line) (p : Parabola),
  max_intersections_circle_line +
  max_intersections_parabola_line +
  max_intersections_circle_parabola = 8 :=
by sorry

end max_intersections_circle_line_parabola_l3862_386245


namespace smallest_rectangle_area_l3862_386246

theorem smallest_rectangle_area (r : ℝ) (h : r = 5) :
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ w * l = 200 ∧
  ∀ (w' l' : ℝ), w' > 0 → l' > 0 → w' * l' ≥ 200 →
  (∀ (x y : ℝ), x^2 + y^2 ≤ r^2 → 0 ≤ x ∧ x ≤ w' ∧ 0 ≤ y ∧ y ≤ l') :=
by sorry

end smallest_rectangle_area_l3862_386246


namespace hiker_distance_l3862_386285

/-- Given a hiker's movements, calculate the final distance from the starting point -/
theorem hiker_distance (north east south west : ℝ) :
  north = 15 ∧ east = 8 ∧ south = 3 ∧ west = 4 →
  Real.sqrt ((north - south)^2 + (east - west)^2) = 4 * Real.sqrt 10 := by
  sorry

end hiker_distance_l3862_386285


namespace conductor_loop_properties_l3862_386229

/-- Parameters for the conductor and loop setup -/
structure ConductorLoopSetup where
  k : Real  -- Current rate of change (A/s)
  r : Real  -- Side length of square loop (m)
  R : Real  -- Resistance of loop (Ω)
  l : Real  -- Distance from straight conductor to loop (m)

/-- Calculate the induced voltage in the loop -/
noncomputable def inducedVoltage (setup : ConductorLoopSetup) : Real :=
  (setup.k * setup.r * Real.log (1 + setup.r / setup.l)) / (2 * Real.pi)

/-- Calculate the time when magnetic induction at the center is zero -/
noncomputable def zeroInductionTime (setup : ConductorLoopSetup) : Real :=
  (4 * Real.sqrt 2 * (setup.l + setup.r / 2) * (inducedVoltage setup / setup.R)) / (setup.k * setup.r)

/-- Theorem stating the properties of the conductor-loop system -/
theorem conductor_loop_properties (setup : ConductorLoopSetup) 
  (h_k : setup.k = 1000)
  (h_r : setup.r = 0.2)
  (h_R : setup.R = 0.01)
  (h_l : setup.l = 0.05) :
  abs (inducedVoltage setup - 6.44e-5) < 1e-7 ∧ 
  abs (zeroInductionTime setup - 2.73e-4) < 1e-6 := by
  sorry


end conductor_loop_properties_l3862_386229


namespace inequality_proof_l3862_386201

theorem inequality_proof (u v x y a b c d : ℝ) 
  (hu : u > 0) (hv : v > 0) (hx : x > 0) (hy : y > 0)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (u / x + v / y ≥ 4 * (u * y + v * x) / ((x + y) ^ 2)) ∧
  (a / (b + 2 * c + d) + b / (c + 2 * d + a) + c / (d + 2 * a + b) + d / (a + 2 * b + c) ≥ 1) := by
  sorry

end inequality_proof_l3862_386201


namespace yanna_afternoon_butter_cookies_l3862_386242

/-- Represents the number of butter cookies Yanna baked in the afternoon -/
def afternoon_butter_cookies : ℕ := sorry

/-- Represents the number of butter cookies Yanna baked in the morning -/
def morning_butter_cookies : ℕ := 20

/-- Represents the number of biscuits Yanna baked in the morning -/
def morning_biscuits : ℕ := 40

/-- Represents the number of biscuits Yanna baked in the afternoon -/
def afternoon_biscuits : ℕ := 20

theorem yanna_afternoon_butter_cookies :
  afternoon_butter_cookies = 20 ∧
  morning_butter_cookies + afternoon_butter_cookies + 30 =
  morning_biscuits + afternoon_biscuits :=
sorry

end yanna_afternoon_butter_cookies_l3862_386242


namespace prob_box1_given_defective_l3862_386231

-- Define the number of components and defective components in each box
def total_box1 : ℕ := 10
def defective_box1 : ℕ := 3
def total_box2 : ℕ := 20
def defective_box2 : ℕ := 2

-- Define the probability of selecting each box
def prob_select_box1 : ℚ := 1/2
def prob_select_box2 : ℚ := 1/2

-- Define the probability of selecting a defective component from each box
def prob_defective_given_box1 : ℚ := defective_box1 / total_box1
def prob_defective_given_box2 : ℚ := defective_box2 / total_box2

-- Define the overall probability of selecting a defective component
def prob_defective : ℚ := 
  prob_select_box1 * prob_defective_given_box1 + 
  prob_select_box2 * prob_defective_given_box2

-- State the theorem
theorem prob_box1_given_defective : 
  (prob_select_box1 * prob_defective_given_box1) / prob_defective = 3/4 := by
  sorry

end prob_box1_given_defective_l3862_386231


namespace jerrys_average_score_l3862_386262

theorem jerrys_average_score (A : ℝ) : 
  (∀ (new_average : ℝ), new_average = A + 2 → 
    3 * A + 102 = 4 * new_average) → 
  A = 94 := by
sorry

end jerrys_average_score_l3862_386262


namespace chocolates_distribution_l3862_386219

theorem chocolates_distribution (total_chocolates : ℕ) (total_children : ℕ) (boys : ℕ) (girls : ℕ) 
  (chocolates_per_girl : ℕ) (h1 : total_chocolates = 3000) (h2 : total_children = 120) 
  (h3 : boys = 60) (h4 : girls = 60) (h5 : chocolates_per_girl = 3) 
  (h6 : total_children = boys + girls) : 
  (total_chocolates - girls * chocolates_per_girl) / boys = 47 := by
  sorry

end chocolates_distribution_l3862_386219


namespace largest_three_digit_and_smallest_four_digit_l3862_386204

theorem largest_three_digit_and_smallest_four_digit : 
  (∃ n : ℕ, n = 999 ∧ ∀ m : ℕ, m < 1000 → m ≤ n) ∧
  (∃ k : ℕ, k = 1000 ∧ ∀ l : ℕ, l ≥ 1000 → l ≥ k) ∧
  (1000 - 999 = 1) :=
by sorry

end largest_three_digit_and_smallest_four_digit_l3862_386204


namespace solve_linear_equation_l3862_386260

theorem solve_linear_equation (x : ℝ) :
  3 * x - 5 * x + 7 * x = 140 → x = 28 := by
  sorry

end solve_linear_equation_l3862_386260


namespace analogical_conclusions_correctness_l3862_386252

theorem analogical_conclusions_correctness :
  (∃! i : Fin 3, 
    (i = 0 → ∀ (a b : ℝ) (n : ℕ), (a + b)^n = a^n + b^n) ∨
    (i = 1 → ∀ (α β : ℝ), Real.sin (α + β) = Real.sin α * Real.sin β) ∨
    (i = 2 → ∀ (a b : ℝ), (a + b)^2 = a^2 + 2*a*b + b^2)) :=
by sorry

end analogical_conclusions_correctness_l3862_386252


namespace ball_distribution_problem_l3862_386239

/-- The number of ways to distribute n distinct objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where the first box is not empty -/
def distributeWithFirstBoxNonEmpty (n k : ℕ) : ℕ :=
  distribute n k - distribute n (k - 1)

/-- The problem statement -/
theorem ball_distribution_problem :
  distributeWithFirstBoxNonEmpty 3 4 = 37 := by sorry

end ball_distribution_problem_l3862_386239


namespace square_configuration_counts_l3862_386275

/-- A configuration of points and line segments in a square -/
structure SquareConfiguration where
  /-- The number of interior points in the square -/
  interior_points : Nat
  /-- The total number of line segments -/
  line_segments : Nat
  /-- The total number of triangles formed -/
  triangles : Nat
  /-- No three points (including square vertices) are collinear -/
  no_collinear_triple : Prop
  /-- No two segments (except at endpoints) share common points -/
  no_intersecting_segments : Prop

/-- Theorem about the number of line segments and triangles in a specific square configuration -/
theorem square_configuration_counts (config : SquareConfiguration) :
  config.interior_points = 1000 →
  config.no_collinear_triple →
  config.no_intersecting_segments →
  config.line_segments = 3001 ∧ config.triangles = 2002 := by
  sorry


end square_configuration_counts_l3862_386275


namespace adam_picked_apples_for_30_days_l3862_386212

/-- The number of days Adam picked apples -/
def days_picked : ℕ := 30

/-- The number of apples Adam picked each day -/
def apples_per_day : ℕ := 4

/-- The number of remaining apples Adam collected after a month -/
def remaining_apples : ℕ := 230

/-- The total number of apples Adam collected -/
def total_apples : ℕ := 350

/-- Theorem stating that the number of days Adam picked apples is 30 -/
theorem adam_picked_apples_for_30_days :
  days_picked * apples_per_day + remaining_apples = total_apples :=
by sorry

end adam_picked_apples_for_30_days_l3862_386212


namespace y_equation_implies_expression_equals_two_l3862_386255

theorem y_equation_implies_expression_equals_two (y : ℝ) (h : y + 2/y = 2) :
  y^6 + 3*y^4 - 4*y^2 + 2 = 2 := by
  sorry

end y_equation_implies_expression_equals_two_l3862_386255


namespace elements_not_in_either_set_l3862_386209

/-- Given sets A and B that are subsets of a finite universal set U, 
    this theorem calculates the number of elements in U that are not in either A or B. -/
theorem elements_not_in_either_set 
  (U A B : Finset ℕ) 
  (h_subset_A : A ⊆ U) 
  (h_subset_B : B ⊆ U) 
  (h_card_U : U.card = 193)
  (h_card_A : A.card = 116)
  (h_card_B : B.card = 41)
  (h_card_inter : (A ∩ B).card = 23) :
  (U \ (A ∪ B)).card = 59 := by
  sorry

#check elements_not_in_either_set

end elements_not_in_either_set_l3862_386209


namespace blue_notes_scattered_l3862_386258

def red_rows : ℕ := 5
def red_notes_per_row : ℕ := 6
def blue_notes_under_each_red : ℕ := 2
def total_notes : ℕ := 100

theorem blue_notes_scattered (red_rows : ℕ) (red_notes_per_row : ℕ) (blue_notes_under_each_red : ℕ) (total_notes : ℕ) :
  red_rows = 5 →
  red_notes_per_row = 6 →
  blue_notes_under_each_red = 2 →
  total_notes = 100 →
  total_notes - (red_rows * red_notes_per_row + red_rows * red_notes_per_row * blue_notes_under_each_red) = 10 :=
by sorry

end blue_notes_scattered_l3862_386258


namespace triangle_area_triangle_area_proof_l3862_386263

/-- The area of a triangle with vertices at (0, 0), (2, 2), and (4, 0) is 4 -/
theorem triangle_area : ℝ → Prop :=
  fun a =>
    let X : ℝ × ℝ := (0, 0)
    let Y : ℝ × ℝ := (2, 2)
    let Z : ℝ × ℝ := (4, 0)
    let base : ℝ := 4
    let height : ℝ := 2
    a = (1 / 2) * base * height ∧ a = 4

/-- The proof of the theorem -/
theorem triangle_area_proof : triangle_area 4 := by
  sorry

end triangle_area_triangle_area_proof_l3862_386263


namespace seashells_given_l3862_386286

theorem seashells_given (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 62) 
  (h2 : remaining_seashells = 13) : 
  initial_seashells - remaining_seashells = 49 := by
  sorry

end seashells_given_l3862_386286


namespace janice_spent_1940_l3862_386217

/-- Calculates the total amount spent by Janice given the prices and quantities of items purchased --/
def total_spent (juice_price : ℚ) (sandwich_price : ℚ) (pastry_price : ℚ) (salad_price : ℚ) : ℚ :=
  let discounted_salad_price := salad_price * (1 - 0.2)
  sandwich_price + juice_price + 2 * pastry_price + discounted_salad_price

/-- Theorem stating that Janice spent $19.40 given the conditions in the problem --/
theorem janice_spent_1940 :
  let juice_price : ℚ := 10 / 5
  let sandwich_price : ℚ := 6 / 2
  let pastry_price : ℚ := 4
  let salad_price : ℚ := 8
  total_spent juice_price sandwich_price pastry_price salad_price = 1940 / 100 := by
  sorry

#eval total_spent (10/5) (6/2) 4 8

end janice_spent_1940_l3862_386217


namespace patio_rearrangement_l3862_386257

theorem patio_rearrangement (total_tiles : ℕ) (initial_rows : ℕ) (column_reduction : ℕ) :
  total_tiles = 96 →
  initial_rows = 8 →
  column_reduction = 2 →
  let initial_columns := total_tiles / initial_rows
  let new_columns := initial_columns - column_reduction
  let new_rows := total_tiles / new_columns
  new_rows - initial_rows = 4 :=
by sorry

end patio_rearrangement_l3862_386257


namespace inequality_solution_l3862_386226

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 2) ≥ 1 / (x - 2) + 3 / 4) ↔ (x > -2 ∧ x ≠ 2) :=
sorry

end inequality_solution_l3862_386226


namespace trajectory_theorem_l3862_386284

def trajectory_problem (R h : ℝ) (θ : ℝ) : Prop :=
  let r₁ := R * Real.cos θ
  let r₂ := (R + h) * Real.cos θ
  let s := 2 * Real.pi * r₂ - 2 * Real.pi * r₁
  s = h ∧ θ = Real.arccos (1 / (2 * Real.pi))

theorem trajectory_theorem :
  ∀ (R h : ℝ), R > 0 → h > 0 → ∃ θ : ℝ, trajectory_problem R h θ :=
sorry

end trajectory_theorem_l3862_386284


namespace smallest_divisible_term_l3862_386236

def geometric_sequence (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) : ℚ :=
  a₁ * (a₂ / a₁) ^ (n - 1)

def is_divisible_by_ten_million (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k * 10000000

theorem smallest_divisible_term : 
  (∀ n < 8, ¬ is_divisible_by_ten_million (geometric_sequence (5/6) 25 n)) ∧ 
  is_divisible_by_ten_million (geometric_sequence (5/6) 25 8) := by
  sorry

end smallest_divisible_term_l3862_386236


namespace area_ratio_AMK_ABC_l3862_386220

/-- Triangle ABC with points M on AB and K on AC -/
structure TriangleWithPoints where
  /-- The area of triangle ABC -/
  area_ABC : ℝ
  /-- The ratio of AM to MB -/
  ratio_AM_MB : ℝ × ℝ
  /-- The ratio of AK to KC -/
  ratio_AK_KC : ℝ × ℝ

/-- The theorem stating the area ratio of triangle AMK to triangle ABC -/
theorem area_ratio_AMK_ABC (t : TriangleWithPoints) (h1 : t.area_ABC = 50) 
  (h2 : t.ratio_AM_MB = (1, 5)) (h3 : t.ratio_AK_KC = (3, 2)) : 
  (∃ (area_AMK : ℝ), area_AMK / t.area_ABC = 1 / 10) :=
sorry

end area_ratio_AMK_ABC_l3862_386220


namespace platform_length_l3862_386292

/-- Given a train with the following properties:
  * Length: 300 meters
  * Starting from rest
  * Constant acceleration
  * Crosses a signal pole in 24 seconds
  * Crosses a platform in 39 seconds
  Prove that the length of the platform is approximately 492.19 meters. -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : pole_time = 24)
  (h3 : platform_time = 39) :
  ∃ (platform_length : ℝ), 
    (abs (platform_length - 492.19) < 0.01) ∧ 
    (∃ (a : ℝ), 
      (train_length = (1/2) * a * pole_time^2) ∧
      (train_length + platform_length = (1/2) * a * platform_time^2)) :=
by sorry

end platform_length_l3862_386292


namespace divisibility_property_l3862_386299

theorem divisibility_property (a b : ℕ+) 
  (h : ∀ k : ℕ+, k < b → (b + k) ∣ (a + k)) :
  ∀ k : ℕ+, k < b → (b - k) ∣ (a - k) := by
  sorry

end divisibility_property_l3862_386299


namespace altitude_sum_less_than_perimeter_l3862_386259

/-- For any triangle, the sum of its altitudes is less than its perimeter -/
theorem altitude_sum_less_than_perimeter (a b c h_a h_b h_c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)
  (altitude_a : h_a ≤ b ∧ h_a ≤ c)
  (altitude_b : h_b ≤ a ∧ h_b ≤ c)
  (altitude_c : h_c ≤ a ∧ h_c ≤ b)
  (non_degenerate : h_a < b ∨ h_a < c ∨ h_b < a ∨ h_b < c ∨ h_c < a ∨ h_c < b) :
  h_a + h_b + h_c < a + b + c := by
  sorry

end altitude_sum_less_than_perimeter_l3862_386259


namespace inequality_proof_l3862_386230

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end inequality_proof_l3862_386230


namespace expression_simplification_l3862_386253

theorem expression_simplification (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  (((x / (x + b)) + (b / (x - b))) / ((b / (x + b)) - (x / (x - b)))) = -1 := by
  sorry

end expression_simplification_l3862_386253


namespace field_trip_attendance_l3862_386237

theorem field_trip_attendance (vans buses : ℕ) (people_per_van people_per_bus : ℕ) 
  (h1 : vans = 9)
  (h2 : buses = 10)
  (h3 : people_per_van = 8)
  (h4 : people_per_bus = 27) :
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end field_trip_attendance_l3862_386237


namespace exponential_and_logarithm_inequalities_l3862_386205

-- Define the exponential function
noncomputable def exp (base : ℝ) (exponent : ℝ) : ℝ := Real.exp (exponent * Real.log base)

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem exponential_and_logarithm_inequalities :
  (exp 0.8 (-0.1) < exp 0.8 (-0.2)) ∧ (log 7 6 > log 8 6) := by
  sorry

end exponential_and_logarithm_inequalities_l3862_386205


namespace marble_count_l3862_386223

theorem marble_count (total : ℕ) (yellow : ℕ) (blue_ratio : ℕ) (red_ratio : ℕ) 
  (h1 : total = 19)
  (h2 : yellow = 5)
  (h3 : blue_ratio = 3)
  (h4 : red_ratio = 4) :
  let remaining := total - yellow
  let share := remaining / (blue_ratio + red_ratio)
  let red := red_ratio * share
  red - yellow = 3 := by sorry

end marble_count_l3862_386223


namespace units_digit_of_product_l3862_386270

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_product (n : ℕ) :
  (3 * sum_factorials n) % 10 = 9 :=
sorry

end units_digit_of_product_l3862_386270


namespace line_parallel_plane_condition_l3862_386278

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the "lies in" relation for a line in a plane
variable (liesIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_condition 
  (a : Line) (α β : Plane) :
  parallelPlane α β → liesIn a β → parallelLinePlane a α :=
by sorry

end line_parallel_plane_condition_l3862_386278


namespace bottle_cap_cost_l3862_386228

/-- Given that 5 bottle caps cost $25, prove that each bottle cap costs $5. -/
theorem bottle_cap_cost : 
  ∀ (cost_per_cap : ℚ), 
  (5 : ℚ) * cost_per_cap = 25 → cost_per_cap = 5 := by
  sorry

end bottle_cap_cost_l3862_386228


namespace square_side_length_l3862_386277

/-- Given a rectangle composed of rectangles R1 and R2, and squares S1, S2, and S3,
    this theorem proves that the side length of S2 is 875 units. -/
theorem square_side_length (total_width total_height : ℕ) (s2 s3 : ℕ) :
  total_width = 4020 →
  total_height = 2160 →
  s3 = s2 + 110 →
  ∃ (r : ℕ), 
    2 * r + s2 = total_height ∧
    2 * r + 3 * s2 + 110 = total_width →
  s2 = 875 :=
by sorry

end square_side_length_l3862_386277


namespace sum_of_roots_cubic_equation_l3862_386234

theorem sum_of_roots_cubic_equation : 
  let p (x : ℝ) := 5 * x^3 - 10 * x^2 + x - 24
  ∃ (r₁ r₂ r₃ : ℝ), p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0 ∧ r₁ + r₂ + r₃ = 2 :=
by sorry

end sum_of_roots_cubic_equation_l3862_386234


namespace smallest_of_five_consecutive_even_numbers_l3862_386288

theorem smallest_of_five_consecutive_even_numbers (a b c d e : ℕ) : 
  (∀ n : ℕ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6 ∧ e = 2*n + 8) → 
  a + b + c + d + e = 320 → 
  a = 60 :=
by sorry

end smallest_of_five_consecutive_even_numbers_l3862_386288


namespace wall_height_proof_l3862_386207

/-- Proves that the height of a wall is 600 cm given specific conditions --/
theorem wall_height_proof (brick_length brick_width brick_height : ℝ)
                          (wall_length wall_width : ℝ)
                          (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 850 →
  wall_width = 22.5 →
  num_bricks = 6800 →
  ∃ (wall_height : ℝ),
    wall_height = 600 ∧
    num_bricks * (brick_length * brick_width * brick_height) =
    wall_length * wall_width * wall_height :=
by
  sorry

#check wall_height_proof

end wall_height_proof_l3862_386207


namespace average_equation_solution_l3862_386289

theorem average_equation_solution (x : ℝ) : 
  ((2*x + 4) + (5*x + 3) + (3*x + 8)) / 3 = 3*x - 5 → x = -30 := by
  sorry

end average_equation_solution_l3862_386289


namespace factorization_problems_l3862_386269

theorem factorization_problems :
  (∀ a : ℝ, a^3 - 4*a = a*(a+2)*(a-2)) ∧
  (∀ m x y : ℝ, 3*m*x^2 - 6*m*x*y + 3*m*y^2 = 3*m*(x-y)^2) := by
  sorry

end factorization_problems_l3862_386269


namespace hugo_win_given_six_l3862_386244

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 6

/-- The probability of winning the game -/
def prob_win : ℚ := 1 / num_players

/-- The probability of rolling a 6 -/
def prob_roll_six : ℚ := 1 / die_sides

/-- The probability that no other player rolls a 6 -/
def prob_no_other_six : ℚ := (1 - 1 / die_sides) ^ (num_players - 1)

theorem hugo_win_given_six (
  hugo_first_roll : ℕ
) : 
  hugo_first_roll = die_sides →
  (prob_roll_six * prob_no_other_six) / prob_win = 3125 / 7776 := by
  sorry

end hugo_win_given_six_l3862_386244


namespace terminal_side_angles_l3862_386238

def angle_set (k : ℤ) : ℝ := k * 360 - 1560

theorem terminal_side_angles :
  (∃ k : ℤ, angle_set k = 240) ∧
  (∃ k : ℤ, angle_set k = -120) ∧
  (∀ α : ℝ, (∃ k : ℤ, angle_set k = α) → α ≥ 240 ∨ α ≤ -120) :=
sorry

end terminal_side_angles_l3862_386238


namespace sqrt_of_nine_l3862_386213

theorem sqrt_of_nine : {x : ℝ | x ^ 2 = 9} = {3, -3} := by sorry

end sqrt_of_nine_l3862_386213


namespace f_of_g_10_l3862_386273

-- Define the functions g and f
def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 8

-- State the theorem
theorem f_of_g_10 : f (g 10) = 262 := by
  sorry

end f_of_g_10_l3862_386273


namespace max_magnitude_c_l3862_386295

open Real

/-- Given vectors a and b, and a vector c satisfying the dot product condition,
    prove that the maximum magnitude of c is √2. -/
theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  a = (1, 0) → 
  b = (0, 1) → 
  (c.1 + a.1, c.2 + a.2) • (c.1 + b.1, c.2 + b.2) = 0 → 
  (∀ c' : ℝ × ℝ, (c'.1 + a.1, c'.2 + a.2) • (c'.1 + b.1, c'.2 + b.2) = 0 → 
    Real.sqrt (c.1^2 + c.2^2) ≥ Real.sqrt (c'.1^2 + c'.2^2)) → 
  Real.sqrt (c.1^2 + c.2^2) = sqrt 2 :=
by sorry


end max_magnitude_c_l3862_386295


namespace set_intersection_equality_l3862_386249

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≥ 2}
def N : Set ℝ := {x | x^2 - 4*x ≥ 0}

-- State the theorem
theorem set_intersection_equality : 
  M ∩ N = {x | x ≤ -1 ∨ x ≥ 4} := by sorry

end set_intersection_equality_l3862_386249


namespace percentage_of_200_to_50_percentage_proof_l3862_386281

theorem percentage_of_200_to_50 : ℝ → Prop :=
  fun x => (200 / 50) * 100 = x ∧ x = 400

-- The proof would go here
theorem percentage_proof : percentage_of_200_to_50 400 := by
  sorry

end percentage_of_200_to_50_percentage_proof_l3862_386281


namespace gnome_ratio_is_half_l3862_386241

/-- Represents the ratio of gnomes with big noses to total gnomes -/
def gnome_ratio (total_gnomes red_hat_gnomes blue_hat_gnomes big_nose_blue_hat small_nose_red_hat : ℕ) : ℚ := 
  let big_nose_red_hat := red_hat_gnomes - small_nose_red_hat
  let total_big_nose := big_nose_blue_hat + big_nose_red_hat
  (total_big_nose : ℚ) / total_gnomes

theorem gnome_ratio_is_half :
  let total_gnomes : ℕ := 28
  let red_hat_gnomes : ℕ := (3 * total_gnomes) / 4
  let blue_hat_gnomes : ℕ := total_gnomes - red_hat_gnomes
  let big_nose_blue_hat : ℕ := 6
  let small_nose_red_hat : ℕ := 13
  gnome_ratio total_gnomes red_hat_gnomes blue_hat_gnomes big_nose_blue_hat small_nose_red_hat = 1/2 := by
  sorry

end gnome_ratio_is_half_l3862_386241


namespace g_composite_three_roots_l3862_386267

/-- The function g(x) defined as x^2 - 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + d

/-- The composite function g(g(x)) -/
def g_composite (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_exactly_three_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ 
    (f x = 0 ∧ f y = 0 ∧ f z = 0) ∧
    (∀ w : ℝ, f w = 0 → w = x ∨ w = y ∨ w = z)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots iff d = 3 -/
theorem g_composite_three_roots (d : ℝ) :
  has_exactly_three_distinct_real_roots (g_composite d) ↔ d = 3 :=
sorry

end g_composite_three_roots_l3862_386267


namespace find_x_l3862_386268

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (a^2)^(2*b) = a^b * x^b → x = a^3 := by
  sorry

end find_x_l3862_386268


namespace train_crossing_time_l3862_386274

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (signal_pole_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 300) 
  (h2 : signal_pole_time = 18) 
  (h3 : platform_length = 600.0000000000001) : 
  (train_length + platform_length) / (train_length / signal_pole_time) = 54.00000000000001 := by
  sorry

end train_crossing_time_l3862_386274


namespace parallel_line_implies_a_value_l3862_386296

/-- Two points are on a line parallel to the y-axis if their x-coordinates are equal -/
def parallel_to_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂

/-- The theorem stating that if M(a-3, a+4) and N(5, 9) form a line segment
    parallel to the y-axis, then a = 8 -/
theorem parallel_line_implies_a_value :
  ∀ a : ℝ,
  parallel_to_y_axis (a - 3) (a + 4) 5 9 →
  a = 8 :=
by
  sorry


end parallel_line_implies_a_value_l3862_386296
