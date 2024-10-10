import Mathlib

namespace gildas_marbles_l690_69027

theorem gildas_marbles (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) :
  let remaining_after_pedro := initial_marbles * (1 - 0.25)
  let remaining_after_ebony := remaining_after_pedro * (1 - 0.15)
  let remaining_after_jimmy := remaining_after_ebony * (1 - 0.30)
  (remaining_after_jimmy / initial_marbles) * 100 = 44.625 := by
sorry

end gildas_marbles_l690_69027


namespace opposite_five_fourteen_implies_eighteen_l690_69094

/-- A structure representing a circle with n equally spaced natural numbers -/
structure NumberCircle where
  n : ℕ
  numbers : Fin n → ℕ
  ordered : ∀ i : Fin n, numbers i = i.val + 1

/-- Definition of opposite numbers on the circle -/
def are_opposite (circle : NumberCircle) (a b : ℕ) : Prop :=
  ∃ i j : Fin circle.n,
    circle.numbers i = a ∧
    circle.numbers j = b ∧
    (j.val + circle.n / 2) % circle.n = i.val

/-- The main theorem -/
theorem opposite_five_fourteen_implies_eighteen (circle : NumberCircle) :
  are_opposite circle 5 14 → circle.n = 18 :=
by sorry

end opposite_five_fourteen_implies_eighteen_l690_69094


namespace ratio_of_sum_and_difference_l690_69004

theorem ratio_of_sum_and_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end ratio_of_sum_and_difference_l690_69004


namespace distance_to_SFL_is_81_l690_69023

/-- The distance to Super Fun-tastic Land -/
def distance_to_SFL (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that the distance to Super Fun-tastic Land is 81 miles -/
theorem distance_to_SFL_is_81 :
  distance_to_SFL 27 3 = 81 := by
  sorry

end distance_to_SFL_is_81_l690_69023


namespace evaluate_expression_at_negative_one_l690_69031

-- Define the expression as a function of x
def f (x : ℚ) : ℚ := (4 + x * (4 + x) - 4^2) / (x - 4 + x^3)

-- State the theorem
theorem evaluate_expression_at_negative_one :
  f (-1) = 5 / 2 := by
  sorry

end evaluate_expression_at_negative_one_l690_69031


namespace immediate_sale_more_profitable_l690_69059

/-- Proves that selling flowers immediately is more profitable than selling after dehydration --/
theorem immediate_sale_more_profitable (initial_weight : ℝ) (initial_price : ℝ) (price_increase : ℝ) 
  (weight_loss_fraction : ℝ) (hw : initial_weight = 49) (hp : initial_price = 1.25) 
  (hpi : price_increase = 2) (hwl : weight_loss_fraction = 5/7) :
  initial_weight * initial_price > 
  (initial_weight * (1 - weight_loss_fraction)) * (initial_price + price_increase) :=
by sorry

end immediate_sale_more_profitable_l690_69059


namespace hotdog_price_l690_69075

/-- The cost of a hamburger -/
def hamburger_cost : ℝ := sorry

/-- The cost of a hot dog -/
def hotdog_cost : ℝ := sorry

/-- First day's purchase equation -/
axiom first_day : 3 * hamburger_cost + 4 * hotdog_cost = 10

/-- Second day's purchase equation -/
axiom second_day : 2 * hamburger_cost + 3 * hotdog_cost = 7

/-- Theorem stating that a hot dog costs 1 dollar -/
theorem hotdog_price : hotdog_cost = 1 := by sorry

end hotdog_price_l690_69075


namespace pauls_fishing_theorem_l690_69037

/-- Calculates the number of fish caught given a fishing rate and total time -/
def fish_caught (rate : ℚ) (period : ℚ) (total_time : ℚ) : ℚ :=
  (total_time / period) * rate

theorem pauls_fishing_theorem :
  let rate : ℚ := 5 / 2  -- 5 fish per 2 hours
  let total_time : ℚ := 12
  fish_caught rate 2 total_time = 30 := by
sorry

end pauls_fishing_theorem_l690_69037


namespace jiyoung_pocket_money_l690_69045

theorem jiyoung_pocket_money (total : ℕ) (difference : ℕ) (jiyoung : ℕ) :
  total = 12000 →
  difference = 1000 →
  total = jiyoung + (jiyoung - difference) →
  jiyoung = 6500 := by
  sorry

end jiyoung_pocket_money_l690_69045


namespace line_slope_equation_l690_69052

/-- Given a line passing through points (-1, -4) and (3, k), where the slope
    of the line is equal to k, prove that k = 4/3 -/
theorem line_slope_equation (k : ℝ) : 
  (let x₁ : ℝ := -1
   let y₁ : ℝ := -4
   let x₂ : ℝ := 3
   let y₂ : ℝ := k
   let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
   slope = k) → k = 4/3 := by
sorry

end line_slope_equation_l690_69052


namespace certain_number_equation_l690_69061

theorem certain_number_equation (x : ℝ) : 5 * 1.6 - (2 * 1.4) / x = 4 ↔ x = 0.7 := by
  sorry

end certain_number_equation_l690_69061


namespace roots_of_product_equation_l690_69095

theorem roots_of_product_equation (p r : ℝ) (f g : ℝ → ℝ) 
  (hp : p > 0) (hr : r > 0)
  (hf : ∀ x, f x = 0 ↔ x = p)
  (hg : ∀ x, g x = 0 ↔ x = r)
  (hlin_f : ∃ a b, ∀ x, f x = a * x + b)
  (hlin_g : ∃ c d, ∀ x, g x = c * x + d) :
  ∀ x, f x * g x = f 0 * g 0 ↔ x = 0 ∨ x = p + r :=
sorry

end roots_of_product_equation_l690_69095


namespace fraction_problem_l690_69046

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * F * (2/5) * N = 15)
  (h2 : (40/100) * N = 180) : 
  F = 2/3 := by sorry

end fraction_problem_l690_69046


namespace quadratic_rewrite_ratio_l690_69078

theorem quadratic_rewrite_ratio : 
  ∃ (c p q : ℚ), 
    (∀ j, 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ 
    q / p = -77 := by
  sorry

end quadratic_rewrite_ratio_l690_69078


namespace exam_probabilities_l690_69030

def prob_above_90 : ℝ := 0.18
def prob_80_to_89 : ℝ := 0.51
def prob_70_to_79 : ℝ := 0.15
def prob_60_to_69 : ℝ := 0.09

theorem exam_probabilities :
  (prob_above_90 + prob_80_to_89 = 0.69) ∧
  (prob_above_90 + prob_80_to_89 + prob_70_to_79 + prob_60_to_69 = 0.93) := by
sorry

end exam_probabilities_l690_69030


namespace octagon_game_areas_l690_69064

/-- A regular octagon inscribed in a circle of radius 2 -/
structure RegularOctagon :=
  (radius : ℝ)
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : ∀ i : Fin 8, (vertices i).1^2 + (vertices i).2^2 = radius^2)

/-- The set of vertices selected by a player -/
def PlayerSelection := Finset (Fin 8)

/-- Predicate for optimal play -/
def OptimalPlay (octagon : RegularOctagon) (alice_selection : PlayerSelection) (bob_selection : PlayerSelection) : Prop :=
  sorry

/-- The area of the convex polygon formed by a player's selection -/
def PolygonArea (octagon : RegularOctagon) (selection : PlayerSelection) : ℝ :=
  sorry

/-- The main theorem -/
theorem octagon_game_areas (octagon : RegularOctagon) (alice_selection : PlayerSelection) (bob_selection : PlayerSelection) :
  octagon.radius = 2 →
  OptimalPlay octagon alice_selection bob_selection →
  alice_selection.card = 4 →
  bob_selection.card = 4 →
  (PolygonArea octagon alice_selection = 2 * Real.sqrt 2 ∨
   PolygonArea octagon alice_selection = 4 + 2 * Real.sqrt 2) :=
by sorry

end octagon_game_areas_l690_69064


namespace log_equation_implies_sum_of_cubes_l690_69090

theorem log_equation_implies_sum_of_cubes (x y : ℝ) 
  (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 
       3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^3 + y^3 = 307 := by
sorry

end log_equation_implies_sum_of_cubes_l690_69090


namespace a_value_l690_69010

def set_A : Set ℝ := {1, -2}

def set_B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem a_value (b : ℝ) : set_A = set_B 1 b → 1 ∈ set_A ∧ -2 ∈ set_A := by sorry

end a_value_l690_69010


namespace real_part_of_z_l690_69057

theorem real_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.re = (Real.sqrt 2 - 1) / 2 := by
  sorry

end real_part_of_z_l690_69057


namespace chessboard_pythagorean_triple_exists_l690_69074

/-- Represents a point on an infinite chessboard --/
structure ChessboardPoint where
  x : Int
  y : Int

/-- Distance function between two ChessboardPoints --/
def distance (p q : ChessboardPoint) : Nat :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).natAbs

/-- Predicate to check if three points are non-collinear --/
def nonCollinear (p q r : ChessboardPoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

/-- Theorem stating the existence of points satisfying the given conditions --/
theorem chessboard_pythagorean_triple_exists : 
  ∃ (A B C : ChessboardPoint), 
    nonCollinear A B C ∧ 
    (distance A C)^2 + (distance B C)^2 = (distance A B)^2 := by
  sorry


end chessboard_pythagorean_triple_exists_l690_69074


namespace peter_investment_duration_l690_69036

/-- Calculates the final amount after simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem peter_investment_duration :
  let principal : ℝ := 710
  let peterFinalAmount : ℝ := 815
  let davidFinalAmount : ℝ := 850
  let davidTime : ℝ := 4
  ∃ (rate : ℝ), 
    (simpleInterest principal rate davidTime = davidFinalAmount) ∧
    (simpleInterest principal rate 3 = peterFinalAmount) := by
  sorry

end peter_investment_duration_l690_69036


namespace line_intersection_y_axis_l690_69076

/-- The line passing through points (2, 9) and (4, 15) intersects the y-axis at (0, 3) -/
theorem line_intersection_y_axis :
  let p₁ : ℝ × ℝ := (2, 9)
  let p₂ : ℝ × ℝ := (4, 15)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, 3) := by
  sorry

end line_intersection_y_axis_l690_69076


namespace tangent_ellipse_d_value_l690_69022

/-- An ellipse in the first quadrant tangent to the x-axis and y-axis with foci at (5,9) and (d,9) -/
structure TangentEllipse where
  d : ℝ
  focus1 : ℝ × ℝ := (5, 9)
  focus2 : ℝ × ℝ := (d, 9)
  first_quadrant : d > 5
  tangent_to_axes : True  -- We assume this property without formally defining it

/-- The value of d for the given ellipse is 29.9 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 29.9 := by
  sorry

#check tangent_ellipse_d_value

end tangent_ellipse_d_value_l690_69022


namespace sufficient_not_necessary_condition_l690_69088

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a < 0 ∧ b < 0 → a + b < 0) ∧
  ∃ (x y : ℝ), x + y < 0 ∧ ¬(x < 0 ∧ y < 0) :=
by sorry

end sufficient_not_necessary_condition_l690_69088


namespace perfect_square_equation_l690_69041

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end perfect_square_equation_l690_69041


namespace parallelogram_sides_l690_69058

theorem parallelogram_sides (a b : ℝ) (h1 : a = 3 * b) (h2 : 2 * a + 2 * b = 24) :
  (a = 9 ∧ b = 3) := by
  sorry

end parallelogram_sides_l690_69058


namespace matrix_transformation_l690_69015

/-- Given a 2nd-order matrix M satisfying the condition, prove M and the transformed curve equation -/
theorem matrix_transformation (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M * !![1, 2; 3, 4] = !![7, 10; 4, 6]) : 
  (M = !![1, 2; 1, 1]) ∧ 
  (∀ x' y' : ℝ, (∃ x y : ℝ, 3*x^2 + 8*x*y + 6*y^2 = 1 ∧ 
                            x' = x + 2*y ∧ 
                            y' = x + y) ↔ 
                x'^2 + 2*y'^2 = 1) := by
  sorry


end matrix_transformation_l690_69015


namespace major_axis_length_is_three_l690_69085

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The major axis length is 3 when the cylinder radius is 1 and the major axis is 50% longer than the minor axis -/
theorem major_axis_length_is_three :
  major_axis_length 1 0.5 = 3 := by
  sorry

end major_axis_length_is_three_l690_69085


namespace train_speed_calculation_l690_69012

/-- Prove that given two trains of equal length, where the faster train travels at a given speed
    and passes the slower train in a given time, the speed of the slower train can be calculated. -/
theorem train_speed_calculation (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 65 →
  faster_speed = 49 →
  passing_time = 36 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    2 * train_length = (faster_speed - slower_speed) * (5 / 18) * passing_time :=
by sorry

end train_speed_calculation_l690_69012


namespace triangle_segment_calculation_l690_69009

/-- Given a triangle ABC with point D on AB and point E on AD, prove that FC has the specified value. -/
theorem triangle_segment_calculation (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 10) 
  (h3 : (1 : ℝ)/5 * AD = AD - DC - CB) (h4 : (3 : ℝ)/4 * AD = ED) : 
  let CA := CB + AD - DC - CB
  let FC := ED * CA / AD
  FC = 11.025 := by sorry

end triangle_segment_calculation_l690_69009


namespace old_clock_slow_12_minutes_l690_69007

/-- Represents the time interval between hand overlaps on the old clock -/
def old_clock_overlap_interval : ℚ := 66

/-- Represents the standard time interval between hand overlaps -/
def standard_overlap_interval : ℚ := 720 / 11

/-- Represents the number of minutes in a standard day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Theorem stating that the old clock is 12 minutes slow over a 24-hour period -/
theorem old_clock_slow_12_minutes :
  (standard_day_minutes : ℚ) / standard_overlap_interval * old_clock_overlap_interval
  - standard_day_minutes = 12 := by sorry

end old_clock_slow_12_minutes_l690_69007


namespace power_subtraction_equals_6444_l690_69047

theorem power_subtraction_equals_6444 : 3^(1+3+4) - (3^1 * 3 + 3^3 + 3^4) = 6444 := by
  sorry

end power_subtraction_equals_6444_l690_69047


namespace fish_cost_per_kg_proof_l690_69072

-- Define the constants
def total_cost_case1 : ℕ := 530
def fish_kg_case1 : ℕ := 4
def pork_kg_case1 : ℕ := 2
def pork_kg_case2 : ℕ := 3
def total_cost_case2 : ℕ := 875
def fish_cost_per_kg : ℕ := 80

-- Define the theorem
theorem fish_cost_per_kg_proof :
  let pork_cost_case1 := total_cost_case1 - fish_cost_per_kg * fish_kg_case1
  let pork_cost_per_kg := pork_cost_case1 / pork_kg_case1
  let pork_cost_case2 := pork_cost_per_kg * pork_kg_case2
  let fish_cost_case2 := total_cost_case2 - pork_cost_case2
  fish_cost_case2 / (fish_cost_case2 / fish_cost_per_kg) = fish_cost_per_kg :=
by
  sorry

#check fish_cost_per_kg_proof

end fish_cost_per_kg_proof_l690_69072


namespace intersection_complement_equality_l690_69055

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end intersection_complement_equality_l690_69055


namespace three_dozen_cost_l690_69065

/-- The cost of apples in dollars -/
def apple_cost (dozens : ℚ) : ℚ := 15.60 * dozens / 2

/-- Theorem: The cost of three dozen apples is $23.40 -/
theorem three_dozen_cost : apple_cost 3 = 23.40 := by
  sorry

end three_dozen_cost_l690_69065


namespace right_rectangular_prism_volume_l690_69005

theorem right_rectangular_prism_volume
  (x y z : ℝ)
  (h_side : x * y = 15)
  (h_front : y * z = 10)
  (h_bottom : x * z = 6) :
  x * y * z = 30 := by
sorry

end right_rectangular_prism_volume_l690_69005


namespace hyperbola_center_l690_69006

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f₁ f₂ c : ℝ × ℝ) : 
  f₁ = (3, 2) → f₂ = (11, 6) → c = (7, 4) → 
  c = ((f₁.1 + f₂.1) / 2, (f₁.2 + f₂.2) / 2) := by
sorry

end hyperbola_center_l690_69006


namespace problem_solution_l690_69043

theorem problem_solution :
  ∀ (x y : ℕ), 
    y > 3 → 
    x^2 + y^4 = 2*((x-6)^2 + (y+1)^2) → 
    x^2 + y^4 = 1994 := by
  sorry

end problem_solution_l690_69043


namespace sin_2alpha_plus_pi_12_l690_69028

theorem sin_2alpha_plus_pi_12 (α : ℝ) 
  (h1 : -π/6 < α ∧ α < π/6) 
  (h2 : Real.cos (α + π/6) = 4/5) : 
  Real.sin (2*α + π/12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end sin_2alpha_plus_pi_12_l690_69028


namespace annual_percentage_increase_l690_69035

theorem annual_percentage_increase (initial_population final_population : ℕ) 
  (h1 : initial_population = 10000)
  (h2 : final_population = 12000) :
  (((final_population - initial_population) : ℚ) / initial_population) * 100 = 20 := by
  sorry

end annual_percentage_increase_l690_69035


namespace quadratic_polynomial_value_l690_69000

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := fun x ↦ a * x^2 + b * x + c

/-- The condition that [q(x)]^3 - x is divisible by (x - 2)(x + 2)(x - 5) -/
def DivisibilityCondition (q : ℚ → ℚ) : Prop :=
  ∀ x, x = 2 ∨ x = -2 ∨ x = 5 → q x ^ 3 = x

theorem quadratic_polynomial_value (a b c : ℚ) :
  (∃ q : ℚ → ℚ, q = QuadraticPolynomial a b c ∧ DivisibilityCondition q) →
  QuadraticPolynomial a b c 10 = -58/7 := by
  sorry

end quadratic_polynomial_value_l690_69000


namespace harold_remaining_amount_l690_69063

def calculate_remaining_amount (primary_income : ℚ) (freelance_income : ℚ) 
  (rent : ℚ) (car_payment : ℚ) (car_insurance : ℚ) (internet : ℚ) 
  (groceries : ℚ) (miscellaneous : ℚ) : ℚ :=
  let total_income := primary_income + freelance_income
  let electricity := 0.25 * car_payment
  let water_sewage := 0.15 * rent
  let total_expenses := rent + car_payment + car_insurance + electricity + water_sewage + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let savings := (2 / 3) * amount_before_savings
  amount_before_savings - savings

theorem harold_remaining_amount :
  calculate_remaining_amount 2500 500 700 300 125 75 200 150 = 423.34 := by
  sorry

end harold_remaining_amount_l690_69063


namespace ceiling_sqrt_200_l690_69039

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end ceiling_sqrt_200_l690_69039


namespace prob_at_least_three_cured_value_l690_69042

-- Define the probability of success for the drug
def drug_success_rate : ℝ := 0.9

-- Define the number of patients
def num_patients : ℕ := 4

-- Define the minimum number of successes we're interested in
def min_successes : ℕ := 3

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_three_cured : ℝ :=
  1 - (Nat.choose num_patients 0 * drug_success_rate^0 * (1 - drug_success_rate)^4 +
       Nat.choose num_patients 1 * drug_success_rate^1 * (1 - drug_success_rate)^3 +
       Nat.choose num_patients 2 * drug_success_rate^2 * (1 - drug_success_rate)^2)

-- Theorem statement
theorem prob_at_least_three_cured_value :
  prob_at_least_three_cured = 0.9477 := by
  sorry

end prob_at_least_three_cured_value_l690_69042


namespace right_triangle_integer_area_l690_69096

theorem right_triangle_integer_area 
  (a b c : ℕ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ A : ℕ, 2 * A = a * b :=
sorry

end right_triangle_integer_area_l690_69096


namespace simplify_fraction_l690_69038

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (4 * x) / (x^2 - 4) - 2 / (x - 2) - 1 = -x / (x + 2) := by
  sorry

end simplify_fraction_l690_69038


namespace basis_linear_independence_l690_69050

-- Define a 2D vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the property of being a basis for a plane
def IsBasisForPlane (e₁ e₂ : V) : Prop :=
  ∀ v : V, ∃ (m n : ℝ), v = m • e₁ + n • e₂

-- Define the property of vectors being not collinear
def NotCollinear (e₁ e₂ : V) : Prop :=
  ∀ (k : ℝ), k • e₁ ≠ e₂

-- The main theorem
theorem basis_linear_independence
  (e₁ e₂ : V)
  (h_basis : IsBasisForPlane e₁ e₂)
  (h_not_collinear : NotCollinear e₁ e₂) :
  ∀ (m n : ℝ), m • e₁ + n • e₂ = 0 → m = 0 ∧ n = 0 :=
by sorry

end basis_linear_independence_l690_69050


namespace total_tylenol_grams_l690_69029

-- Define the parameters
def tablet_count : ℕ := 2
def tablet_mg : ℕ := 500
def hours_between_doses : ℕ := 4
def total_hours : ℕ := 12
def mg_per_gram : ℕ := 1000

-- Theorem statement
theorem total_tylenol_grams : 
  (total_hours / hours_between_doses) * tablet_count * tablet_mg / mg_per_gram = 3 := by
  sorry

end total_tylenol_grams_l690_69029


namespace congruence_condition_l690_69040

/-- A triangle specified by two sides and an angle --/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (angle : ℝ)

/-- Predicate to check if a triangle specification guarantees congruence --/
def guarantees_congruence (t : Triangle) : Prop :=
  t.side1 > 0 ∧ t.side2 > 0 ∧ t.angle > 0 ∧ t.angle < 180 ∧
  (t.angle > 90 ∨ (t.angle = 90 ∧ t.side1 ≠ t.side2))

/-- The triangles from the problem options --/
def triangle_A : Triangle := { side1 := 2, side2 := 0, angle := 60 }
def triangle_B : Triangle := { side1 := 2, side2 := 3, angle := 0 }
def triangle_C : Triangle := { side1 := 3, side2 := 5, angle := 150 }
def triangle_D : Triangle := { side1 := 3, side2 := 2, angle := 30 }

theorem congruence_condition :
  guarantees_congruence triangle_C ∧
  ¬guarantees_congruence triangle_A ∧
  ¬guarantees_congruence triangle_B ∧
  ¬guarantees_congruence triangle_D :=
sorry

end congruence_condition_l690_69040


namespace book_club_boys_count_l690_69091

theorem book_club_boys_count (total_members attendees : ℕ) 
  (h_total : total_members = 30)
  (h_attendees : attendees = 18)
  (h_all_boys_attended : ∃ boys girls : ℕ, 
    boys + girls = total_members ∧
    boys + (girls / 3) = attendees) : 
  ∃ boys : ℕ, boys = 12 ∧ ∃ girls : ℕ, boys + girls = total_members :=
sorry

end book_club_boys_count_l690_69091


namespace reading_time_proof_l690_69073

def total_chapters : Nat := 31
def reading_time_per_chapter : Nat := 20

def chapters_read (n : Nat) : Nat :=
  n - (n / 3)

def total_reading_time_minutes (n : Nat) (t : Nat) : Nat :=
  (chapters_read n) * t

def total_reading_time_hours (n : Nat) (t : Nat) : Nat :=
  (total_reading_time_minutes n t) / 60

theorem reading_time_proof :
  total_reading_time_hours total_chapters reading_time_per_chapter = 7 := by
  sorry

end reading_time_proof_l690_69073


namespace f_at_three_l690_69021

/-- Horner's method representation of the polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := (((2 * x + 3) * x + 0) * x + 5) * x - 4

/-- Theorem stating that f(3) = 254 -/
theorem f_at_three : f 3 = 254 := by sorry

end f_at_three_l690_69021


namespace susan_pencil_purchase_l690_69024

/-- The number of pencils Susan bought -/
def num_pencils : ℕ := 16

/-- The number of pens Susan bought -/
def num_pens : ℕ := 36 - num_pencils

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := 25

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 80

/-- The total amount Susan spent in cents -/
def total_spent : ℕ := 2000

theorem susan_pencil_purchase :
  num_pencils + num_pens = 36 ∧
  pencil_cost * num_pencils + pen_cost * num_pens = total_spent :=
by sorry

#check susan_pencil_purchase

end susan_pencil_purchase_l690_69024


namespace equation_system_solution_l690_69011

theorem equation_system_solution (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end equation_system_solution_l690_69011


namespace dress_designs_count_l690_69069

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of fabric materials available for each color -/
def num_materials : ℕ := 2

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_materials * num_patterns

theorem dress_designs_count : total_designs = 40 := by
  sorry

end dress_designs_count_l690_69069


namespace dress_making_hours_l690_69054

/-- Calculates the total hours required to make dresses given the available fabric, fabric per dress, and time per dress. -/
def total_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (time_per_dress : ℕ) : ℕ :=
  (total_fabric / fabric_per_dress) * time_per_dress

/-- Proves that given 56 square meters of fabric, where each dress requires 4 square meters of fabric
    and 3 hours to make, the total number of hours required to make all possible dresses is 42 hours. -/
theorem dress_making_hours : total_hours 56 4 3 = 42 := by
  sorry

end dress_making_hours_l690_69054


namespace complex_symmetric_division_l690_69053

/-- Two complex numbers are symmetric about the origin if their sum is zero -/
def symmetric_about_origin (z₁ z₂ : ℂ) : Prop := z₁ + z₂ = 0

theorem complex_symmetric_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_about_origin z₁ z₂) (h_z₁ : z₁ = 2 - I) : 
  z₁ / z₂ = -1 := by
  sorry

end complex_symmetric_division_l690_69053


namespace square_sum_given_difference_and_product_l690_69066

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = -8) : 
  x^2 + y^2 = 33 := by sorry

end square_sum_given_difference_and_product_l690_69066


namespace megan_earnings_after_discount_l690_69002

/-- Calculates Megan's earnings from selling necklaces at a garage sale with a discount --/
theorem megan_earnings_after_discount :
  let bead_necklaces : ℕ := 7
  let bead_price : ℕ := 5
  let gem_necklaces : ℕ := 3
  let gem_price : ℕ := 15
  let discount_rate : ℚ := 1/5  -- 20% as a rational number
  
  let total_before_discount := bead_necklaces * bead_price + gem_necklaces * gem_price
  let discount_amount := (total_before_discount : ℚ) * discount_rate
  let earnings_after_discount := (total_before_discount : ℚ) - discount_amount
  
  earnings_after_discount = 64 := by sorry

end megan_earnings_after_discount_l690_69002


namespace cube_sum_magnitude_l690_69062

theorem cube_sum_magnitude (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 15) :
  Complex.abs (w^3 + z^3) = 41 := by
  sorry

end cube_sum_magnitude_l690_69062


namespace quadratic_inequality_l690_69017

theorem quadratic_inequality (x : ℝ) : x^2 + 7*x + 6 < 0 ↔ -6 < x ∧ x < -1 := by
  sorry

end quadratic_inequality_l690_69017


namespace parabola_range_l690_69003

-- Define the function f(x) = x^2 - 4x + 5
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem parabola_range :
  ∀ y : ℝ, (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = y) ↔ 1 ≤ y ∧ y < 5 := by sorry

end parabola_range_l690_69003


namespace largest_difference_l690_69048

def S : Set Int := {-20, -5, 1, 5, 7, 19}

theorem largest_difference (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x - y = 39 ∧ ∀ (c d : Int), c ∈ S → d ∈ S → c - d ≤ 39 := by
  sorry

end largest_difference_l690_69048


namespace line_parallel_to_plane_iff_no_intersection_l690_69082

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Check if a line is parallel to a plane -/
def isParallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Check if a line intersects with another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Get a line in a plane -/
def lineInPlane (p : Plane3D) : Line3D :=
  sorry

theorem line_parallel_to_plane_iff_no_intersection (l : Line3D) (p : Plane3D) :
  isParallel l p ↔ ∀ (l' : Line3D), lineInPlane p = l' → ¬ intersects l l' :=
sorry

end line_parallel_to_plane_iff_no_intersection_l690_69082


namespace factor_implies_absolute_value_l690_69033

/-- Given a polynomial 3x^4 - mx^2 + nx + p with factors (x-3), (x+1), and (x-2), 
    prove that |3m - 2n| = 25 -/
theorem factor_implies_absolute_value (m n p : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x + 1) * (x - 2) ∣ (3 * x^4 - m * x^2 + n * x + p)) → 
  |3 * m - 2 * n| = 25 := by
  sorry

end factor_implies_absolute_value_l690_69033


namespace arithmetic_geometric_sequence_l690_69067

/-- Given an arithmetic sequence with common difference 2, if a₁, a₃, a₄ form a geometric sequence, then a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
sorry

end arithmetic_geometric_sequence_l690_69067


namespace upstream_downstream_time_ratio_l690_69018

/-- Proves that the ratio of upstream to downstream rowing time is 2:1 given specific speeds -/
theorem upstream_downstream_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 4.5)
  (h2 : current_speed = 1.5) : 
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

#check upstream_downstream_time_ratio

end upstream_downstream_time_ratio_l690_69018


namespace sum_of_coefficients_l690_69016

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end sum_of_coefficients_l690_69016


namespace alpha_plus_beta_equals_111_l690_69019

theorem alpha_plus_beta_equals_111 :
  ∀ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1343) / (x^2 + 63*x - 3360)) →
  α + β = 111 :=
by
  sorry

end alpha_plus_beta_equals_111_l690_69019


namespace two_solutions_sine_equation_l690_69060

theorem two_solutions_sine_equation (x : ℝ) (a : ℝ) : 
  (x ∈ Set.Icc 0 Real.pi) →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ ∈ Set.Icc 0 Real.pi ∧ 
    x₂ ∈ Set.Icc 0 Real.pi ∧
    2 * Real.sin (x₁ + Real.pi / 3) = a ∧ 
    2 * Real.sin (x₂ + Real.pi / 3) = a) ↔
  (a > Real.sqrt 3 ∧ a < 2) :=
by sorry

end two_solutions_sine_equation_l690_69060


namespace boots_sold_l690_69077

theorem boots_sold (sneakers sandals total : ℕ) 
  (h1 : sneakers = 2)
  (h2 : sandals = 4)
  (h3 : total = 17) :
  total - (sneakers + sandals) = 11 := by
  sorry

end boots_sold_l690_69077


namespace f_extrema_l690_69079

def f (x : ℝ) := 3 * x^4 - 6 * x^2 + 4

theorem f_extrema :
  (∀ x ∈ Set.Icc (-1) 3, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 1) ∧
  (∀ x ∈ Set.Icc (-1) 3, f x ≤ 193) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 193) := by
  sorry

end f_extrema_l690_69079


namespace tan_product_seventh_pi_l690_69013

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end tan_product_seventh_pi_l690_69013


namespace distance_P_to_AB_l690_69008

-- Define the rectangle ABCD
def rectangle_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 8 ∧
  B.1 = 6 ∧ B.2 = 8 ∧
  C.1 = 6 ∧ C.2 = 0 ∧
  D.1 = 0 ∧ D.2 = 0

-- Define point M as the midpoint of CD
def point_M (C D M : ℝ × ℝ) : Prop :=
  M.1 = (C.1 + D.1) / 2 ∧ M.2 = (C.2 + D.2) / 2

-- Define the circle with center M and radius 3
def circle_M (M P : ℝ × ℝ) : Prop :=
  (P.1 - M.1)^2 + (P.2 - M.2)^2 = 3^2

-- Define the circle with center B and radius 5
def circle_B (B P : ℝ × ℝ) : Prop :=
  (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5^2

-- Theorem statement
theorem distance_P_to_AB (A B C D M P : ℝ × ℝ) :
  rectangle_ABCD A B C D →
  point_M C D M →
  circle_M M P →
  circle_B B P →
  P.1 = 18/5 := by sorry

end distance_P_to_AB_l690_69008


namespace principal_amount_proof_l690_69071

/-- Proves that given the specified conditions, the principal amount is 4000 (rs.) --/
theorem principal_amount_proof (rate : ℚ) (amount : ℚ) (time : ℚ) : 
  rate = 8 / 100 → amount = 640 → time = 2 → 
  (amount * 100) / (rate * time) = 4000 := by
  sorry

end principal_amount_proof_l690_69071


namespace sum_of_number_and_reverse_divisible_by_11_l690_69083

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) : 
  A < 10 → B < 10 → A ≠ B → 
  11 ∣ ((10 * A + B) + (10 * B + A)) := by
sorry

end sum_of_number_and_reverse_divisible_by_11_l690_69083


namespace license_plate_count_l690_69081

/-- The number of vowels in the alphabet, considering Y as a vowel -/
def num_vowels : ℕ := 6

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 20

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_vowels * num_consonants * num_vowels * num_digits

theorem license_plate_count : total_license_plates = 403200 := by
  sorry

end license_plate_count_l690_69081


namespace total_marigolds_sold_l690_69087

/-- The number of marigolds sold during a three-day sale -/
def marigolds_sold (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of marigolds sold during the sale -/
theorem total_marigolds_sold :
  let day1 := 14
  let day2 := 25
  let day3 := 2 * day2
  marigolds_sold day1 day2 day3 = 89 := by
  sorry

end total_marigolds_sold_l690_69087


namespace diagonals_15_sided_polygon_l690_69098

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 15 sides is 90 -/
theorem diagonals_15_sided_polygon : num_diagonals 15 = 90 := by
  sorry

end diagonals_15_sided_polygon_l690_69098


namespace sum_reciprocals_l690_69093

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
  sorry

end sum_reciprocals_l690_69093


namespace smallest_sum_of_primes_with_all_digits_l690_69044

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the digits of a number -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if a list contains all digits from 0 to 9 exactly once -/
def hasAllDigitsOnce (l : List ℕ) : Prop := sorry

/-- The theorem stating the smallest sum of primes using all digits once -/
theorem smallest_sum_of_primes_with_all_digits : 
  ∃ (s : List ℕ), 
    (∀ n ∈ s, isPrime n) ∧ 
    hasAllDigitsOnce (s.bind digits) ∧
    (s.sum = 208) ∧
    (∀ (t : List ℕ), 
      (∀ m ∈ t, isPrime m) → 
      hasAllDigitsOnce (t.bind digits) → 
      t.sum ≥ 208) := by
  sorry

end smallest_sum_of_primes_with_all_digits_l690_69044


namespace complex_equation_solution_l690_69086

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃! (a : ℝ), i * (1 + a * i) = 2 + i :=
by
  -- The proof goes here
  sorry

end complex_equation_solution_l690_69086


namespace sum_of_A_and_D_is_six_l690_69032

-- Define single-digit numbers
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- Define three-digit number ABX
def ThreeDigitABX (A B X : ℕ) : ℕ := 100 * A + 10 * B + X

-- Define three-digit number CDY
def ThreeDigitCDY (C D Y : ℕ) : ℕ := 100 * C + 10 * D + Y

-- Define four-digit number XYXY
def FourDigitXYXY (X Y : ℕ) : ℕ := 1000 * X + 100 * Y + 10 * X + Y

-- Theorem statement
theorem sum_of_A_and_D_is_six 
  (A B C D X Y : ℕ) 
  (hA : SingleDigit A) (hB : SingleDigit B) (hC : SingleDigit C) 
  (hD : SingleDigit D) (hX : SingleDigit X) (hY : SingleDigit Y)
  (h_sum : ThreeDigitABX A B X + ThreeDigitCDY C D Y = FourDigitXYXY X Y) :
  A + D = 6 := by
  sorry


end sum_of_A_and_D_is_six_l690_69032


namespace problem_solution_l690_69084

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 3|

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 5) ↔ a ≥ 4) :=
sorry

end problem_solution_l690_69084


namespace sum_remainder_thirteen_l690_69025

theorem sum_remainder_thirteen : ∃ k : ℕ, (5000 + 5001 + 5002 + 5003 + 5004 + 5005 + 5006) = 13 * k + 3 := by
  sorry

end sum_remainder_thirteen_l690_69025


namespace equation_solution_l690_69099

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 8 ∧ x = 112 := by
  sorry

end equation_solution_l690_69099


namespace litter_patrol_pickup_l690_69051

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := 8

/-- The total number of pieces of litter picked up by the Litter Patrol -/
def total_litter : ℕ := glass_bottles + aluminum_cans

theorem litter_patrol_pickup :
  total_litter = 18 := by sorry

end litter_patrol_pickup_l690_69051


namespace overlapping_triangles_sum_l690_69080

/-- Represents a triangle with angles a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180

/-- Configuration of two pairs of overlapping triangles -/
structure OverlappingTriangles where
  t1 : Triangle
  t2 : Triangle

/-- The sum of all distinct angles in a configuration of two pairs of overlapping triangles is 360° -/
theorem overlapping_triangles_sum (ot : OverlappingTriangles) : 
  ot.t1.a + ot.t1.b + ot.t1.c + ot.t2.a + ot.t2.b + ot.t2.c = 360 := by
  sorry


end overlapping_triangles_sum_l690_69080


namespace gym_time_is_two_hours_l690_69068

/-- Represents the daily schedule of a working mom --/
structure DailySchedule where
  wakeTime : Nat
  sleepTime : Nat
  workHours : Nat
  cookingTime : Real
  bathTime : Real
  homeworkTime : Real
  lunchPackingTime : Real
  cleaningTime : Real
  leisureTime : Real

/-- Calculates the total awake hours in a day --/
def awakeHours (schedule : DailySchedule) : Nat :=
  schedule.sleepTime - schedule.wakeTime

/-- Calculates the total time spent on activities excluding work and gym --/
def otherActivitiesTime (schedule : DailySchedule) : Real :=
  schedule.cookingTime + schedule.bathTime + schedule.homeworkTime +
  schedule.lunchPackingTime + schedule.cleaningTime + schedule.leisureTime

/-- Theorem: The working mom spends 2 hours at the gym --/
theorem gym_time_is_two_hours (schedule : DailySchedule) 
    (h1 : schedule.wakeTime = 7)
    (h2 : schedule.sleepTime = 23)
    (h3 : schedule.workHours = 8)
    (h4 : schedule.workHours = awakeHours schedule / 2)
    (h5 : schedule.cookingTime = 1.5)
    (h6 : schedule.bathTime = 0.5)
    (h7 : schedule.homeworkTime = 1)
    (h8 : schedule.lunchPackingTime = 0.5)
    (h9 : schedule.cleaningTime = 0.5)
    (h10 : schedule.leisureTime = 2) :
    awakeHours schedule - schedule.workHours - otherActivitiesTime schedule = 2 := by
  sorry


end gym_time_is_two_hours_l690_69068


namespace sum_of_four_primes_divisible_by_60_l690_69049

theorem sum_of_four_primes_divisible_by_60 
  (p q r s : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hs : Nat.Prime s) 
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) : 
  60 ∣ (p + q + r + s) := by
sorry

end sum_of_four_primes_divisible_by_60_l690_69049


namespace lawn_mowing_problem_lawn_mowing_solution_l690_69020

theorem lawn_mowing_problem (original_people : ℕ) (original_time : ℝ) 
  (new_time : ℝ) (efficiency : ℝ) (new_people : ℕ) : Prop :=
  original_people = 8 →
  original_time = 3 →
  new_time = 2 →
  efficiency = 0.9 →
  (original_people : ℝ) * original_time = (new_people : ℝ) * new_time * efficiency →
  new_people = 14

-- The proof of the theorem
theorem lawn_mowing_solution : lawn_mowing_problem 8 3 2 0.9 14 := by
  sorry

end lawn_mowing_problem_lawn_mowing_solution_l690_69020


namespace hotel_rooms_count_l690_69034

/-- Calculates the total number of rooms in a hotel with three wings. -/
def total_rooms_in_hotel (
  wing1_floors : ℕ) (wing1_halls_per_floor : ℕ) (wing1_rooms_per_hall : ℕ)
  (wing2_floors : ℕ) (wing2_halls_per_floor : ℕ) (wing2_rooms_per_hall : ℕ)
  (wing3_floors : ℕ) (wing3_halls_per_floor : ℕ) (wing3_rooms_per_hall : ℕ) : ℕ :=
  wing1_floors * wing1_halls_per_floor * wing1_rooms_per_hall +
  wing2_floors * wing2_halls_per_floor * wing2_rooms_per_hall +
  wing3_floors * wing3_halls_per_floor * wing3_rooms_per_hall

/-- Theorem stating that the total number of rooms in the hotel is 6648. -/
theorem hotel_rooms_count : 
  total_rooms_in_hotel 9 6 32 7 9 40 12 4 50 = 6648 := by
  sorry

end hotel_rooms_count_l690_69034


namespace rook_removal_theorem_l690_69014

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks a square (i, j) -/
def attacks (x y i j : Fin 8) : Bool :=
  x = i ∨ y = j

/-- A configuration of rooks on a chessboard -/
def RookConfiguration := Fin 20 → Fin 8 × Fin 8

/-- Checks if a configuration of rooks attacks the entire board -/
def attacks_all_squares (config : RookConfiguration) : Prop :=
  ∀ i j, ∃ k, attacks (config k).1 (config k).2 i j

/-- Represents a subset of 8 rooks from the original 20 -/
def Subset := Fin 8 → Fin 20

theorem rook_removal_theorem (initial_config : RookConfiguration) 
  (h : attacks_all_squares initial_config) :
  ∃ (subset : Subset), attacks_all_squares (λ i => initial_config (subset i)) :=
sorry

end rook_removal_theorem_l690_69014


namespace count_valid_pairs_l690_69092

def is_valid_pair (x y : ℕ) : Prop :=
  Nat.Prime x ∧ Nat.Prime y ∧ x ≠ y ∧ (621 * x * y) % (x + y) = 0

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, is_valid_pair p.1 p.2) ∧ 
    (∀ x y, is_valid_pair x y → (x, y) ∈ pairs) ∧
    pairs.card = 6 := by
  sorry

end count_valid_pairs_l690_69092


namespace find_x_value_l690_69097

def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

def median (l : List ℝ) : ℝ :=
  l[l.length / 2]!

theorem find_x_value (l : List ℝ) (h_length : l.length = 5) 
  (h_ascending : is_ascending l) (h_median : median l = 22) 
  (h_elements : l = [14, 19, x, 23, 27]) : x = 22 :=
sorry

end find_x_value_l690_69097


namespace caitlin_uniform_number_l690_69089

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ Nat.Prime n

theorem caitlin_uniform_number
  (a b c : ℕ)
  (ha : is_two_digit_prime a)
  (hb : is_two_digit_prime b)
  (hc : is_two_digit_prime c)
  (hab : a ≠ b)
  (hac : a ≠ c)
  (hbc : b ≠ c)
  (sum_ac : a + c = 24)
  (sum_ab : a + b = 30)
  (sum_bc : b + c = 28) :
  c = 11 := by
  sorry

end caitlin_uniform_number_l690_69089


namespace simplify_fraction_multiplication_l690_69056

theorem simplify_fraction_multiplication :
  (180 : ℚ) / 1620 * 20 = 20 / 9 := by
  sorry

end simplify_fraction_multiplication_l690_69056


namespace light_wash_count_l690_69070

/-- Represents the number of gallons of water used per load for different wash types -/
structure WaterUsage where
  heavy : ℕ
  regular : ℕ
  light : ℕ

/-- Represents the number of loads for each wash type -/
structure Loads where
  heavy : ℕ
  regular : ℕ
  light : ℕ
  bleached : ℕ

def totalWaterUsage (usage : WaterUsage) (loads : Loads) : ℕ :=
  usage.heavy * loads.heavy +
  usage.regular * loads.regular +
  usage.light * (loads.light + loads.bleached)

theorem light_wash_count (usage : WaterUsage) (loads : Loads) :
  usage.heavy = 20 →
  usage.regular = 10 →
  usage.light = 2 →
  loads.heavy = 2 →
  loads.regular = 3 →
  loads.bleached = 2 →
  totalWaterUsage usage loads = 76 →
  loads.light = 1 :=
sorry

end light_wash_count_l690_69070


namespace locus_of_P_l690_69026

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the property for point P
def P_property (P : ℝ × ℝ) : Prop :=
  ∃ (B : ℝ × ℝ),
    B ∈ Circle ∧
    (P.1 - A.1) * B.1 = P.2 * B.2 ∧  -- AP || OB
    (P.1 - A.1) * (B.1 - A.1) + P.2 * B.2 = 1  -- AP · AB = 1

-- The theorem to prove
theorem locus_of_P :
  ∀ (P : ℝ × ℝ), P_property P ↔ P.2^2 = 2 * P.1 - 1 :=
sorry

end locus_of_P_l690_69026


namespace M_remainder_1000_l690_69001

/-- M is the greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ :=
  sorry

/-- The remainder when M is divided by 1000 is 621 -/
theorem M_remainder_1000 : M % 1000 = 621 :=
  sorry

end M_remainder_1000_l690_69001
