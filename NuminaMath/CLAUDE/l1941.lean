import Mathlib

namespace unique_solution_condition_l1941_194198

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ k = 35 / 4 := by sorry

end unique_solution_condition_l1941_194198


namespace linear_function_constraint_l1941_194165

/-- A linear function y = kx + b -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Predicate to check if a point (x, y) is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Predicate to check if a point (x, y) is the origin -/
def is_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

/-- Theorem stating that if a linear function doesn't pass through the third quadrant or origin, 
    then k < 0 and b > 0 -/
theorem linear_function_constraint (k b : ℝ) :
  (∀ x : ℝ, ¬(in_third_quadrant x (linear_function k b x))) ∧
  (∀ x : ℝ, ¬(is_origin x (linear_function k b x))) →
  k < 0 ∧ b > 0 :=
sorry

end linear_function_constraint_l1941_194165


namespace interest_rate_is_zero_l1941_194140

/-- The interest rate for a TV purchase with installment payments -/
def interest_rate_tv_purchase (tv_price : ℕ) (num_installments : ℕ) (installment_amount : ℕ) (last_installment : ℕ) : ℚ :=
  if tv_price = 60000 ∧ 
     num_installments = 20 ∧ 
     installment_amount = 1000 ∧ 
     last_installment = 59000 ∧
     tv_price - installment_amount = last_installment
  then 0
  else 1 -- arbitrary non-zero value for other cases

/-- Theorem stating that the interest rate is 0% for the given TV purchase conditions -/
theorem interest_rate_is_zero :
  interest_rate_tv_purchase 60000 20 1000 59000 = 0 := by
  sorry

end interest_rate_is_zero_l1941_194140


namespace no_definitive_inference_l1941_194141

-- Define the sets
variable (Mem Ens Vee : Set α)

-- Define the conditions
variable (h1 : ∃ x, x ∈ Mem ∧ x ∉ Ens)
variable (h2 : Ens ∩ Vee = ∅)

-- Define the potential inferences
def inference_A := ∃ x, x ∈ Mem ∧ x ∉ Vee
def inference_B := ∃ x, x ∈ Vee ∧ x ∉ Mem
def inference_C := Mem ∩ Vee = ∅
def inference_D := ∃ x, x ∈ Mem ∧ x ∈ Vee

-- The theorem to prove
theorem no_definitive_inference :
  ¬(inference_A Mem Vee) ∧
  ¬(inference_B Mem Vee) ∧
  ¬(inference_C Mem Vee) ∧
  ¬(inference_D Mem Vee) :=
sorry

end no_definitive_inference_l1941_194141


namespace negative_one_to_zero_power_equals_one_l1941_194159

theorem negative_one_to_zero_power_equals_one : (-1 : ℤ) ^ (0 : ℕ) = 1 := by
  sorry

end negative_one_to_zero_power_equals_one_l1941_194159


namespace min_value_quadratic_sum_l1941_194175

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 1) :
  2 * x^2 + 3 * y^2 + z^2 ≥ 6/11 :=
by sorry

end min_value_quadratic_sum_l1941_194175


namespace triangle_angle_R_l1941_194148

theorem triangle_angle_R (P Q R : Real) (h1 : 2 * Real.sin P + 5 * Real.cos Q = 4) 
  (h2 : 5 * Real.sin Q + 2 * Real.cos P = 3) 
  (h3 : P + Q + R = Real.pi) : R = Real.pi := by
  sorry

end triangle_angle_R_l1941_194148


namespace smallest_base_for_90_l1941_194169

theorem smallest_base_for_90 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(∃ (d₁ d₂ d₃ : ℕ), d₁ < x ∧ d₂ < x ∧ d₃ < x ∧ 
    90 = d₁ * x^2 + d₂ * x + d₃)) ∧
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < b ∧ d₂ < b ∧ d₃ < b ∧ 
    90 = d₁ * b^2 + d₂ * b + d₃) :=
by sorry

end smallest_base_for_90_l1941_194169


namespace quadratic_polynomial_value_at_14_l1941_194122

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x, (q x)^3 - x = p x * (x - 2) * (x + 2) * (x - 9)

theorem quadratic_polynomial_value_at_14 
  (q : QuadraticPolynomial ℝ) 
  (h : DivisibilityCondition q) : 
  q 14 = -82 := by
  sorry

end quadratic_polynomial_value_at_14_l1941_194122


namespace chord_line_equation_l1941_194161

/-- The equation of a line containing a chord of an ellipse --/
theorem chord_line_equation (x y : ℝ) :
  let ellipse := fun (x y : ℝ) ↦ x^2 / 16 + y^2 / 9 = 1
  let midpoint := (2, (3 : ℝ) / 2)
  let chord_line := fun (x y : ℝ) ↦ 3 * x + 4 * y - 12 = 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) = midpoint ∧
    (∀ x y, chord_line x y ↔ ∃ t, x = (1 - t) * x₁ + t * x₂ ∧ y = (1 - t) * y₁ + t * y₂) :=
by
  sorry

end chord_line_equation_l1941_194161


namespace cereal_box_total_price_l1941_194115

/-- Calculates the total price paid for discounted cereal boxes -/
theorem cereal_box_total_price 
  (initial_price : ℕ) 
  (price_reduction : ℕ) 
  (num_boxes : ℕ) 
  (h1 : initial_price = 104)
  (h2 : price_reduction = 24)
  (h3 : num_boxes = 20) : 
  (initial_price - price_reduction) * num_boxes = 1600 := by
  sorry

#check cereal_box_total_price

end cereal_box_total_price_l1941_194115


namespace first_number_equation_l1941_194119

theorem first_number_equation (x : ℝ) : (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end first_number_equation_l1941_194119


namespace complex_modulus_sqrt_two_l1941_194157

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end complex_modulus_sqrt_two_l1941_194157


namespace water_amount_for_scaled_solution_l1941_194186

theorem water_amount_for_scaled_solution 
  (chemical_a : Real) 
  (water : Real) 
  (total : Real) 
  (new_total : Real) 
  (h1 : chemical_a + water = total)
  (h2 : chemical_a = 0.07)
  (h3 : water = 0.03)
  (h4 : total = 0.1)
  (h5 : new_total = 0.6) : 
  (water / total) * new_total = 0.18 := by
  sorry

end water_amount_for_scaled_solution_l1941_194186


namespace special_circle_properties_l1941_194190

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- The circle passes through these two points
  pointA : ℝ × ℝ := (1, 4)
  pointB : ℝ × ℝ := (3, 2)
  -- The center lies on this line
  centerLine : ℝ → ℝ := fun x => 3 - x

/-- The equation of the circle -/
def circleEquation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 4

/-- A point on the circle -/
def pointOnCircle (c : SpecialCircle) (p : ℝ × ℝ) : Prop :=
  circleEquation c p.1 p.2

theorem special_circle_properties (c : SpecialCircle) :
  -- The circle equation is correct
  (∀ x y, circleEquation c x y ↔ pointOnCircle c (x, y)) ∧
  -- The maximum value of x+y for points on the circle
  (∃ max : ℝ, max = 3 + 2 * Real.sqrt 2 ∧
    ∀ p, pointOnCircle c p → p.1 + p.2 ≤ max) := by
  sorry

end special_circle_properties_l1941_194190


namespace difference_of_squares_1027_l1941_194174

theorem difference_of_squares_1027 : (1027 : ℤ) * 1027 - 1026 * 1028 = 1 := by
  sorry

end difference_of_squares_1027_l1941_194174


namespace james_cycling_distance_l1941_194194

theorem james_cycling_distance (speed : ℝ) (morning_time : ℝ) (afternoon_time : ℝ) 
  (h1 : speed = 8)
  (h2 : morning_time = 2.5)
  (h3 : afternoon_time = 1.5) :
  speed * morning_time + speed * afternoon_time = 32 :=
by sorry

end james_cycling_distance_l1941_194194


namespace meena_cookies_left_l1941_194152

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of cookies Meena bakes -/
def meena_bakes : ℕ := 5

/-- The number of dozens of cookies sold to the biology teacher -/
def sold_to_teacher : ℕ := 2

/-- The number of cookies Brock buys -/
def brock_buys : ℕ := 7

/-- Katy buys twice as many cookies as Brock -/
def katy_buys : ℕ := 2 * brock_buys

/-- The total number of cookies Meena initially bakes -/
def total_baked : ℕ := meena_bakes * dozen

/-- The number of cookies sold to the biology teacher -/
def teacher_cookies : ℕ := sold_to_teacher * dozen

/-- The total number of cookies sold -/
def total_sold : ℕ := teacher_cookies + brock_buys + katy_buys

/-- The number of cookies Meena has left -/
def cookies_left : ℕ := total_baked - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l1941_194152


namespace cyclic_quadrilateral_side_length_l1941_194129

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the circle
def Circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

-- Define the theorem
theorem cyclic_quadrilateral_side_length 
  (ABCD : Quadrilateral) 
  (inscribed : ABCD.A ∈ Circle ∧ ABCD.B ∈ Circle ∧ ABCD.C ∈ Circle ∧ ABCD.D ∈ Circle) 
  (perp_diagonals : (ABCD.A.1 - ABCD.C.1) * (ABCD.B.1 - ABCD.D.1) + 
                    (ABCD.A.2 - ABCD.C.2) * (ABCD.B.2 - ABCD.D.2) = 0)
  (AB_length : (ABCD.A.1 - ABCD.B.1)^2 + (ABCD.A.2 - ABCD.B.2)^2 = 9) :
  (ABCD.C.1 - ABCD.D.1)^2 + (ABCD.C.2 - ABCD.D.2)^2 = 7 :=
sorry

end cyclic_quadrilateral_side_length_l1941_194129


namespace green_ball_probability_l1941_194151

/-- Represents a container with balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from two containers -/
def prob_green (a b : Container) : ℚ :=
  let total_a := a.red + a.green
  let total_b := b.red + b.green
  let prob_a := (a.green : ℚ) / (2 * total_a)
  let prob_b := (b.green : ℚ) / (2 * total_b)
  prob_a + prob_b

theorem green_ball_probability :
  let a := Container.mk 5 5
  let b := Container.mk 7 3
  prob_green a b = 2/5 := by
  sorry

end green_ball_probability_l1941_194151


namespace eliot_votes_l1941_194164

/-- Given the vote distribution in a school election, prove that Eliot got 160 votes. -/
theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ) : 
  randy_votes = 16 → 
  shaun_votes = 5 * randy_votes → 
  eliot_votes = 2 * shaun_votes → 
  eliot_votes = 160 := by
sorry


end eliot_votes_l1941_194164


namespace product_of_sum_and_sum_of_cubes_l1941_194114

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) : 
  a + b = 5 → a^3 + b^3 = 125 → a * b = 0 := by sorry

end product_of_sum_and_sum_of_cubes_l1941_194114


namespace cupcakes_eaten_later_is_22_l1941_194142

/-- Represents the cupcake business scenario --/
structure CupcakeBusiness where
  cost_per_cupcake : ℚ
  burnt_cupcakes : ℕ
  perfect_cupcakes : ℕ
  eaten_immediately : ℕ
  made_later : ℕ
  selling_price : ℚ
  net_profit : ℚ

/-- Calculates the number of cupcakes eaten later --/
def cupcakes_eaten_later (business : CupcakeBusiness) : ℚ :=
  let total_made := business.perfect_cupcakes + business.made_later
  let total_cost := (business.burnt_cupcakes + total_made) * business.cost_per_cupcake
  let available_for_sale := total_made - business.eaten_immediately
  ((available_for_sale * business.selling_price - total_cost - business.net_profit) / business.selling_price)

/-- Theorem stating the number of cupcakes eaten later --/
theorem cupcakes_eaten_later_is_22 (business : CupcakeBusiness)
  (h1 : business.cost_per_cupcake = 3/4)
  (h2 : business.burnt_cupcakes = 24)
  (h3 : business.perfect_cupcakes = 24)
  (h4 : business.eaten_immediately = 5)
  (h5 : business.made_later = 24)
  (h6 : business.selling_price = 2)
  (h7 : business.net_profit = 24) :
  cupcakes_eaten_later business = 22 := by
  sorry

end cupcakes_eaten_later_is_22_l1941_194142


namespace mouse_cost_l1941_194182

theorem mouse_cost (mouse_cost keyboard_cost total_cost : ℝ) : 
  keyboard_cost = 3 * mouse_cost →
  total_cost = mouse_cost + keyboard_cost →
  total_cost = 64 →
  mouse_cost = 16 := by
sorry

end mouse_cost_l1941_194182


namespace gcd_765432_654321_l1941_194168

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l1941_194168


namespace plant_arrangement_count_l1941_194139

/-- Represents the number of ways to arrange plants under lamps -/
def plant_arrangements : ℕ := 49

/-- The number of basil plants -/
def num_basil : ℕ := 3

/-- The number of aloe plants -/
def num_aloe : ℕ := 1

/-- The number of lamp colors -/
def num_lamp_colors : ℕ := 3

/-- The number of lamps per color -/
def lamps_per_color : ℕ := 2

/-- The total number of lamps -/
def total_lamps : ℕ := num_lamp_colors * lamps_per_color

/-- The total number of plants -/
def total_plants : ℕ := num_basil + num_aloe

theorem plant_arrangement_count :
  (num_basil = 3) →
  (num_aloe = 1) →
  (num_lamp_colors = 3) →
  (lamps_per_color = 2) →
  (total_plants ≤ total_lamps) →
  (plant_arrangements = 49) := by
  sorry

end plant_arrangement_count_l1941_194139


namespace greatest_integer_with_gcf_four_exists_unique_greatest_solution_is_148_l1941_194189

theorem greatest_integer_with_gcf_four (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 4 → n ≤ 148 := by
  sorry

theorem exists_unique_greatest : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 24 = 4 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 24 = 4 → m ≤ n := by
  sorry

theorem solution_is_148 : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 24 = 4 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 24 = 4 → m ≤ n ∧ n = 148 := by
  sorry

end greatest_integer_with_gcf_four_exists_unique_greatest_solution_is_148_l1941_194189


namespace binary_channel_properties_l1941_194196

/-- A binary channel with error probabilities α and β -/
structure BinaryChannel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def prob_single_101 (bc : BinaryChannel) : ℝ := (1 - bc.α) * (1 - bc.β)^2

/-- Probability of receiving 1,0,1 when sending 1 in triple transmission -/
def prob_triple_101 (bc : BinaryChannel) : ℝ := bc.β * (1 - bc.β)^2

/-- Probability of decoding 1 when sending 1 in triple transmission -/
def prob_triple_decode_1 (bc : BinaryChannel) : ℝ := 3 * bc.β * (1 - bc.β)^2 + (1 - bc.β)^3

/-- Probability of decoding 0 when sending 0 in single transmission -/
def prob_single_decode_0 (bc : BinaryChannel) : ℝ := 1 - bc.α

/-- Probability of decoding 0 when sending 0 in triple transmission -/
def prob_triple_decode_0 (bc : BinaryChannel) : ℝ := 3 * bc.α * (1 - bc.α)^2 + (1 - bc.α)^3

theorem binary_channel_properties (bc : BinaryChannel) :
  prob_single_101 bc = (1 - bc.α) * (1 - bc.β)^2 ∧
  prob_triple_101 bc = bc.β * (1 - bc.β)^2 ∧
  prob_triple_decode_1 bc = 3 * bc.β * (1 - bc.β)^2 + (1 - bc.β)^3 ∧
  (bc.α < 0.5 → prob_triple_decode_0 bc > prob_single_decode_0 bc) :=
by sorry

end binary_channel_properties_l1941_194196


namespace textbook_selection_ways_l1941_194103

/-- The number of ways to select textbooks from two categories -/
def select_textbooks (required : ℕ) (selective : ℕ) (total : ℕ) : ℕ :=
  (required.choose 1 * selective.choose 2) + (required.choose 2 * selective.choose 1)

/-- Theorem stating that selecting 3 textbooks from 2 required and 3 selective, 
    with at least one from each category, can be done in 9 ways -/
theorem textbook_selection_ways :
  select_textbooks 2 3 3 = 9 := by
  sorry

#eval select_textbooks 2 3 3

end textbook_selection_ways_l1941_194103


namespace smallest_possible_n_l1941_194184

theorem smallest_possible_n (a b c n : ℕ) : 
  a < b → b < c → c < n → 
  a + b + c + n = 100 → 
  (∀ m : ℕ, m < n → ¬∃ x y z : ℕ, x < y ∧ y < z ∧ z < m ∧ x + y + z + m = 100) →
  n = 27 := by
  sorry

end smallest_possible_n_l1941_194184


namespace circus_crowns_l1941_194145

theorem circus_crowns (total_feathers : ℕ) (feathers_per_crown : ℕ) (h1 : total_feathers = 6538) (h2 : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := by
sorry

end circus_crowns_l1941_194145


namespace smallest_d_divisible_by_11_l1941_194127

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def number_with_d (d : ℕ) : ℕ :=
  457000 + d * 100 + 1

theorem smallest_d_divisible_by_11 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_11 (number_with_d d) → d ≥ 5) ∧
    (is_divisible_by_11 (number_with_d 5)) :=
by sorry

end smallest_d_divisible_by_11_l1941_194127


namespace circle_center_correct_l1941_194180

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -3)

/-- Theorem: The center of the circle defined by circle_equation is circle_center -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 13 :=
by sorry

end circle_center_correct_l1941_194180


namespace estimated_probability_one_hit_l1941_194153

-- Define the type for a throw result
inductive ThrowResult
| Hit
| Miss

-- Define a type for a set of two throws
def TwoThrows := (ThrowResult × ThrowResult)

-- Define the simulation data
def simulation_data : List TwoThrows := sorry

-- Define the function to count sets with exactly one hit
def count_one_hit (data : List TwoThrows) : Nat := sorry

-- Theorem statement
theorem estimated_probability_one_hit 
  (h1 : simulation_data.length = 20)
  (h2 : count_one_hit simulation_data = 10) :
  (count_one_hit simulation_data : ℚ) / simulation_data.length = 1/2 := by
  sorry

end estimated_probability_one_hit_l1941_194153


namespace conditional_equivalence_l1941_194105

theorem conditional_equivalence (P Q : Prop) :
  (P → ¬Q) ↔ (Q → ¬P) := by sorry

end conditional_equivalence_l1941_194105


namespace purely_imaginary_complex_number_l1941_194106

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 - Complex.I) * (-2 + a * Complex.I)
  (z.re = 0) → a = 2 := by
  sorry

end purely_imaginary_complex_number_l1941_194106


namespace diophantine_equation_solutions_l1941_194171

theorem diophantine_equation_solutions : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 806 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 807) (Finset.range 807))).card = 67 := by
  sorry

end diophantine_equation_solutions_l1941_194171


namespace power_comparison_a_l1941_194109

theorem power_comparison_a : 3^200 > 2^300 := by sorry

end power_comparison_a_l1941_194109


namespace zero_subset_M_l1941_194192

def M : Set ℝ := {x | x > -2}

theorem zero_subset_M : {0} ⊆ M := by
  sorry

end zero_subset_M_l1941_194192


namespace inequality_minimum_a_l1941_194173

theorem inequality_minimum_a : 
  (∀ a : ℝ, (∀ x : ℝ, x > a → (2*x^2 - 2*a*x + 2) / (x - a) ≥ 5) → a ≥ 1/2) ∧
  (∃ a : ℝ, a = 1/2 ∧ ∀ x : ℝ, x > a → (2*x^2 - 2*a*x + 2) / (x - a) ≥ 5) :=
by sorry

end inequality_minimum_a_l1941_194173


namespace triangle_problem_l1941_194166

/-- Given a triangle ABC with area 3√15, b - c = 2, and cos A = -1/4, prove the following: -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (1/2 * b * c * Real.sqrt (1 - (-1/4)^2) = 3 * Real.sqrt 15) →
  (b - c = 2) →
  (Real.cos A = -1/4) →
  (a^2 = b^2 + c^2 - 2*b*c*(-1/4)) →
  (a / Real.sqrt (1 - (-1/4)^2) = c / Real.sin C) →
  (a = 8 ∧ 
   Real.sin C = Real.sqrt 15 / 8 ∧ 
   Real.cos (2*A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) :=
by sorry

end triangle_problem_l1941_194166


namespace largest_constant_K_l1941_194154

theorem largest_constant_K : ∃ (K : ℝ), K > 0 ∧
  (∀ (k : ℝ) (a b c : ℝ), 0 ≤ k ∧ k ≤ K ∧ 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
    a^2 + b^2 + c^2 + k*a*b*c = k + 3 →
    a + b + c ≤ 3) ∧
  (∀ (K' : ℝ), K' > K →
    ∃ (k : ℝ) (a b c : ℝ), 0 ≤ k ∧ k ≤ K' ∧ 
      a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
      a^2 + b^2 + c^2 + k*a*b*c = k + 3 ∧
      a + b + c > 3) ∧
  K = 1 := by
sorry

end largest_constant_K_l1941_194154


namespace book_sale_loss_percentage_l1941_194101

theorem book_sale_loss_percentage (selling_price_loss : ℝ) (selling_price_gain : ℝ) 
  (gain_percentage : ℝ) (loss_percentage : ℝ) :
  selling_price_loss = 450 →
  selling_price_gain = 550 →
  gain_percentage = 10 →
  (selling_price_gain = (100 + gain_percentage) / 100 * (100 / (100 + gain_percentage) * selling_price_gain)) →
  (loss_percentage = (((100 / (100 + gain_percentage) * selling_price_gain) - selling_price_loss) / 
    (100 / (100 + gain_percentage) * selling_price_gain)) * 100) →
  loss_percentage = 10 := by
  sorry

end book_sale_loss_percentage_l1941_194101


namespace sqrt_seven_less_than_three_l1941_194143

theorem sqrt_seven_less_than_three : Real.sqrt 7 < 3 := by
  sorry

end sqrt_seven_less_than_three_l1941_194143


namespace second_year_interest_rate_l1941_194191

/-- Calculates the interest rate for the second year given the initial amount,
    first-year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_amount : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_amount = 9000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 9828) :
  ∃ (second_year_rate : ℝ),
    second_year_rate = 0.05 ∧
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry


end second_year_interest_rate_l1941_194191


namespace ellipse_max_value_l1941_194147

theorem ellipse_max_value (x y : ℝ) :
  (x^2 / 6 + y^2 / 4 = 1) →
  (∃ (max : ℝ), ∀ (a b : ℝ), a^2 / 6 + b^2 / 4 = 1 → x + 2*y ≤ max ∧ max = Real.sqrt 22) :=
by sorry

end ellipse_max_value_l1941_194147


namespace real_roots_range_not_p_and_q_implies_range_l1941_194108

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0
def q (m : ℝ) : Prop := m ∈ Set.Icc (-1 : ℝ) 5

-- Theorem 1
theorem real_roots_range (m : ℝ) : p m → m ∈ Set.Iic 1 := by sorry

-- Theorem 2
theorem not_p_and_q_implies_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m ∈ Set.Iio (-1) ∪ Set.Ioc 1 5 := by sorry

end real_roots_range_not_p_and_q_implies_range_l1941_194108


namespace scientific_notation_of_34_million_l1941_194132

theorem scientific_notation_of_34_million :
  ∃ (a : ℝ) (n : ℤ), 34000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.4 ∧ n = 7 := by
  sorry

end scientific_notation_of_34_million_l1941_194132


namespace range_of_a_when_one_in_B_range_of_a_when_B_subset_A_l1941_194128

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 1) < 0}

-- Part 1: Range of a when 1 ∈ B
theorem range_of_a_when_one_in_B :
  ∀ a : ℝ, 1 ∈ B a ↔ 0 < a ∧ a < 1 := by sorry

-- Part 2: Range of a when B is a proper subset of A
theorem range_of_a_when_B_subset_A :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ (∃ y : ℝ, y ∈ A ∧ y ∉ B a) ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

end range_of_a_when_one_in_B_range_of_a_when_B_subset_A_l1941_194128


namespace initial_water_percentage_l1941_194155

/-- Proves that the initial water percentage in a mixture is 60% given the specified conditions --/
theorem initial_water_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 300 →
  added_water = 100 →
  final_water_percentage = 70 →
  (initial_volume * x / 100 + added_water) / (initial_volume + added_water) * 100 = final_water_percentage →
  x = 60 := by
  sorry

#check initial_water_percentage

end initial_water_percentage_l1941_194155


namespace inequality_properties_l1941_194112

theorem inequality_properties (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧
  (1 / (a - b) < 1 / a) ∧
  (|a| > -b) ∧
  (Real.sqrt (-a) > Real.sqrt (-b)) := by
  sorry

end inequality_properties_l1941_194112


namespace order_mnpq_l1941_194199

theorem order_mnpq (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n :=
by sorry

end order_mnpq_l1941_194199


namespace complex_sum_equality_l1941_194187

/-- Given complex numbers a and b, prove that 2a + 3b = 1 + i -/
theorem complex_sum_equality (a b : ℂ) (ha : a = 2 - I) (hb : b = -1 + I) :
  2 * a + 3 * b = 1 + I := by
  sorry

end complex_sum_equality_l1941_194187


namespace opposite_sides_range_l1941_194111

theorem opposite_sides_range (m : ℝ) : 
  let A : ℝ × ℝ := (m, 2)
  let B : ℝ × ℝ := (2, m)
  let line (x y : ℝ) := x + 2*y - 4
  (line A.1 A.2) * (line B.1 B.2) < 0 → 0 < m ∧ m < 1 := by
sorry

end opposite_sides_range_l1941_194111


namespace h_j_h_3_equals_277_l1941_194125

def h (x : ℝ) : ℝ := 5 * x + 2

def j (x : ℝ) : ℝ := 3 * x + 4

theorem h_j_h_3_equals_277 : h (j (h 3)) = 277 := by
  sorry

end h_j_h_3_equals_277_l1941_194125


namespace first_cut_ratio_l1941_194181

/-- Proves the ratio of the first cut rope to the initial rope length is 1/2 -/
theorem first_cut_ratio (initial_length : ℝ) (final_piece_length : ℝ) : 
  initial_length = 100 → 
  final_piece_length = 5 → 
  (initial_length / 2) / initial_length = 1 / 2 := by
sorry

end first_cut_ratio_l1941_194181


namespace midpoint_sum_l1941_194123

/-- Given points A (a, 6), B (-2, b), and P (2, 3) where P bisects AB, prove a + b = 6 -/
theorem midpoint_sum (a b : ℝ) : 
  (2 : ℝ) = (a + (-2)) / 2 → 
  (3 : ℝ) = (6 + b) / 2 → 
  a + b = 6 := by
  sorry

end midpoint_sum_l1941_194123


namespace total_raisins_added_l1941_194158

theorem total_raisins_added (yellow_raisins : ℝ) (black_raisins : ℝ)
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4) :
  yellow_raisins + black_raisins = 0.7 := by
  sorry

end total_raisins_added_l1941_194158


namespace z_in_fourth_quadrant_l1941_194104

theorem z_in_fourth_quadrant (z : ℂ) (h : z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3)) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end z_in_fourth_quadrant_l1941_194104


namespace grain_output_scientific_notation_l1941_194126

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem grain_output_scientific_notation :
  toScientificNotation 736000000 = ScientificNotation.mk 7.36 8 (by norm_num) := by
  sorry

end grain_output_scientific_notation_l1941_194126


namespace die_roll_probability_l1941_194138

theorem die_roll_probability : 
  let p_two : ℚ := 1 / 6  -- probability of rolling a 2
  let p_not_two : ℚ := 5 / 6  -- probability of not rolling a 2
  let num_rolls : ℕ := 5  -- number of rolls
  let num_twos : ℕ := 4  -- number of 2s we want
  
  -- probability of rolling exactly four 2s in first four rolls and not a 2 in last roll
  p_two ^ num_twos * p_not_two = 5 / 7776 :=
by sorry

end die_roll_probability_l1941_194138


namespace remainder_after_adding_2024_l1941_194183

theorem remainder_after_adding_2024 (n : ℤ) (h : n % 8 = 3) : (n + 2024) % 8 = 3 := by
  sorry

end remainder_after_adding_2024_l1941_194183


namespace stock_price_increase_l1941_194137

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 25)
  (h2 : closing_price = 28) :
  (closing_price - opening_price) / opening_price * 100 = 12 := by
  sorry

end stock_price_increase_l1941_194137


namespace inequality_proof_l1941_194170

theorem inequality_proof (x : ℝ) : x > 0 ∧ |4*x - 5| < 8 → 0 < x ∧ x < 13/4 := by
  sorry

end inequality_proof_l1941_194170


namespace range_of_a_l1941_194195

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 := by
  sorry

end range_of_a_l1941_194195


namespace profit_calculation_l1941_194160

/-- Calculates the actual percent profit given the markup percentage and discount percentage -/
def actualPercentProfit (markup : ℝ) (discount : ℝ) : ℝ :=
  let labeledPrice := 1 + markup
  let sellingPrice := labeledPrice * (1 - discount)
  (sellingPrice - 1) * 100

/-- Theorem stating that a 40% markup with a 5% discount results in a 33% profit -/
theorem profit_calculation (markup discount : ℝ) 
  (h1 : markup = 0.4) 
  (h2 : discount = 0.05) : 
  actualPercentProfit markup discount = 33 := by
  sorry

end profit_calculation_l1941_194160


namespace power_of_product_l1941_194136

theorem power_of_product (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by
  sorry

end power_of_product_l1941_194136


namespace binomial_probability_theorem_l1941_194110

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Probability mass function for a binomial random variable -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k : ℝ) * X.p ^ k * (1 - X.p) ^ (X.n - k)

theorem binomial_probability_theorem (X : BinomialRV) 
  (h_exp : expected_value X = 2)
  (h_var : variance X = 4/3) :
  pmf X 2 = 80/243 := by
  sorry

end binomial_probability_theorem_l1941_194110


namespace caramel_candy_probability_l1941_194121

/-- The probability of selecting a caramel-flavored candy from a set of candies -/
theorem caramel_candy_probability 
  (total_candies : ℕ) 
  (caramel_candies : ℕ) 
  (lemon_candies : ℕ) 
  (h1 : total_candies = caramel_candies + lemon_candies)
  (h2 : caramel_candies = 3)
  (h3 : lemon_candies = 4) :
  (caramel_candies : ℚ) / total_candies = 3 / 7 :=
sorry

end caramel_candy_probability_l1941_194121


namespace unique_number_with_three_prime_divisors_including_13_l1941_194172

theorem unique_number_with_three_prime_divisors_including_13 :
  ∀ x n : ℕ,
  x = 9^n - 1 →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 := by sorry

end unique_number_with_three_prime_divisors_including_13_l1941_194172


namespace triangle_side_length_l1941_194116

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (h1 : t.c = 1)           -- AC = 1
  (h2 : t.b = 3)           -- BC = 3
  (h3 : t.A + t.B = π / 3) -- A + B = 60° (in radians)
  : t.a = 2 * Real.sqrt 13 := by
  sorry


end triangle_side_length_l1941_194116


namespace similar_radical_expressions_l1941_194188

-- Define the concept of similar radical expressions
def are_similar_radical_expressions (x y : ℝ) : Prop :=
  ∃ (k : ℝ) (n : ℕ), k > 0 ∧ x = k * (y^(1/n))

theorem similar_radical_expressions :
  ∀ (a : ℝ), a > 0 →
  (are_similar_radical_expressions (a^(1/3) * (3^(1/3))) 3) ∧
  ¬(are_similar_radical_expressions a (3*a/2)) ∧
  ¬(are_similar_radical_expressions (2*a) (a^(1/2))) ∧
  ¬(are_similar_radical_expressions (2*a) ((3*a^2)^(1/2))) :=
by sorry

end similar_radical_expressions_l1941_194188


namespace pencil_box_sequence_l1941_194107

theorem pencil_box_sequence (a : ℕ → ℕ) (h1 : a 1 = 78) (h2 : a 2 = 87) (h3 : a 3 = 96) (h5 : a 5 = 114)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 9) : a 4 = 105 := by
  sorry

end pencil_box_sequence_l1941_194107


namespace derek_dogs_now_l1941_194193

-- Define the number of dogs Derek had at age 7
def dogs_at_7 : ℕ := 120

-- Define the number of cars Derek had at age 7
def cars_at_7 : ℕ := dogs_at_7 / 4

-- Define the number of cars Derek bought
def cars_bought : ℕ := 350

-- Define the total number of cars Derek has now
def cars_now : ℕ := cars_at_7 + cars_bought

-- Define the number of dogs Derek has now
def dogs_now : ℕ := cars_now / 3

-- Theorem to prove
theorem derek_dogs_now : dogs_now = 126 := by
  sorry

end derek_dogs_now_l1941_194193


namespace sufficient_to_necessary_contrapositive_l1941_194156

theorem sufficient_to_necessary_contrapositive (a b : Prop) :
  (a → b) → (¬b → ¬a) := by
  sorry

end sufficient_to_necessary_contrapositive_l1941_194156


namespace andrews_friends_pizza_slices_l1941_194149

/-- The total number of pizza slices brought by Andrew's friends -/
def total_pizza_slices (num_friends : ℕ) (slices_per_friend : ℕ) : ℕ :=
  num_friends * slices_per_friend

/-- Theorem stating that the total number of pizza slices is 16 -/
theorem andrews_friends_pizza_slices :
  total_pizza_slices 4 4 = 16 := by
  sorry

end andrews_friends_pizza_slices_l1941_194149


namespace train_distance_problem_l1941_194133

/-- The distance between two cities given train travel conditions -/
theorem train_distance_problem : ∃ (dist : ℝ) (speed_A speed_B : ℝ),
  -- Two trains meet after 3.3 hours
  dist = 3.3 * (speed_A + speed_B) ∧
  -- Train A departing 24 minutes earlier condition
  0.4 * speed_A + 3 * (speed_A + speed_B) + 14 = 3.3 * (speed_A + speed_B) ∧
  -- Train B departing 36 minutes earlier condition
  0.6 * speed_B + 3 * (speed_A + speed_B) + 9 = 3.3 * (speed_A + speed_B) ∧
  -- The distance between the two cities is 660 km
  dist = 660 := by
  sorry

end train_distance_problem_l1941_194133


namespace line_properties_l1941_194179

/-- The slope of the line sqrt(3)x - y - 1 = 0 is sqrt(3) and its inclination angle is 60° --/
theorem line_properties :
  let line := fun (x y : ℝ) => Real.sqrt 3 * x - y - 1 = 0
  ∃ (m θ : ℝ),
    (∀ x y, line x y → y = m * x - 1) ∧ 
    m = Real.sqrt 3 ∧
    θ = 60 * π / 180 ∧
    Real.tan θ = m :=
by sorry

end line_properties_l1941_194179


namespace fixed_point_exponential_function_l1941_194167

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by sorry

end fixed_point_exponential_function_l1941_194167


namespace a_card_is_one_three_l1941_194185

structure Card where
  n1 : Nat
  n2 : Nat
  deriving Repr

structure Person where
  name : String
  card : Card
  deriving Repr

def validCards : List Card := [⟨1, 2⟩, ⟨1, 3⟩, ⟨2, 3⟩]

def commonNumber (c1 c2 : Card) : Nat :=
  if c1.n1 = c2.n1 ∨ c1.n1 = c2.n2 then c1.n1
  else if c1.n2 = c2.n1 ∨ c1.n2 = c2.n2 then c1.n2
  else 0

theorem a_card_is_one_three 
  (a b c : Person)
  (h1 : a.card ∈ validCards ∧ b.card ∈ validCards ∧ c.card ∈ validCards)
  (h2 : a.card ≠ b.card ∧ b.card ≠ c.card ∧ a.card ≠ c.card)
  (h3 : commonNumber a.card b.card ≠ 2)
  (h4 : commonNumber b.card c.card ≠ 1)
  (h5 : c.card.n1 + c.card.n2 ≠ 5) :
  a.card = ⟨1, 3⟩ := by
sorry

end a_card_is_one_three_l1941_194185


namespace smallest_n_for_radio_profit_l1941_194197

theorem smallest_n_for_radio_profit (n d : ℕ) (h1 : d > 0) : 
  (∃ (m : ℕ), m ≥ n ∧ 
    d - (3 * d) / (2 * m) + 10 * m - 30 = d + 100 ∧
    (∀ k : ℕ, k < m → d - (3 * d) / (2 * k) + 10 * k - 30 ≠ d + 100)) →
  n = 13 := by
sorry

end smallest_n_for_radio_profit_l1941_194197


namespace negation_of_universal_proposition_l1941_194177

theorem negation_of_universal_proposition :
  ¬(∀ n : ℤ, n % 5 = 0 → Odd n) ↔ ∃ n : ℤ, n % 5 = 0 ∧ ¬(Odd n) :=
by sorry

end negation_of_universal_proposition_l1941_194177


namespace marks_ratio_polly_willy_l1941_194113

/-- Given the ratios of marks between students, prove the ratio between Polly and Willy -/
theorem marks_ratio_polly_willy (p s w : ℝ) 
  (h1 : p / s = 4 / 5) 
  (h2 : s / w = 5 / 2) : 
  p / w = 2 / 1 := by
  sorry

end marks_ratio_polly_willy_l1941_194113


namespace triangle_problem_l1941_194124

theorem triangle_problem (A B C a b c : ℝ) : 
  a ≠ b →
  c = Real.sqrt 7 →
  b * Real.sin B - a * Real.sin A = Real.sqrt 3 * a * Real.cos A - Real.sqrt 3 * b * Real.cos B →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →
  C = π / 3 ∧ ((a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2)) :=
by sorry

end triangle_problem_l1941_194124


namespace caleb_caught_two_trouts_l1941_194146

/-- The number of trouts Caleb caught -/
def caleb_trouts : ℕ := 2

/-- The number of trouts Caleb's dad caught -/
def dad_trouts : ℕ := 3 * caleb_trouts

theorem caleb_caught_two_trouts :
  (dad_trouts = 3 * caleb_trouts) ∧
  (dad_trouts = caleb_trouts + 4) →
  caleb_trouts = 2 := by
  sorry

end caleb_caught_two_trouts_l1941_194146


namespace fraction_addition_l1941_194100

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 5 = (17 : ℚ) / 20 := by
  sorry

end fraction_addition_l1941_194100


namespace angles_in_range_l1941_194118

-- Define the set S
def S : Set ℝ := {x | ∃ k : ℤ, x = k * 360 + 370 + 23 / 60}

-- Define the range of angles
def inRange (x : ℝ) : Prop := -720 ≤ x ∧ x < 360

-- State the theorem
theorem angles_in_range :
  ∃! (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    inRange a ∧ inRange b ∧ inRange c ∧
    a = -709 - 37 / 60 ∧
    b = -349 - 37 / 60 ∧
    c = 10 + 23 / 60 :=
  sorry

end angles_in_range_l1941_194118


namespace tan_double_angle_special_case_l1941_194178

theorem tan_double_angle_special_case (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_special_case_l1941_194178


namespace minimum_cost_green_plants_l1941_194117

/-- Represents the number of pots of green lily -/
def green_lily_pots : ℕ → Prop :=
  λ x => x ≥ 31 ∧ x ≤ 46

/-- Represents the number of pots of spider plant -/
def spider_plant_pots : ℕ → Prop :=
  λ y => y ≥ 0 ∧ y ≤ 15

/-- The total cost of purchasing the plants -/
def total_cost (x y : ℕ) : ℕ :=
  9 * x + 6 * y

theorem minimum_cost_green_plants :
  ∀ x y : ℕ,
    green_lily_pots x →
    spider_plant_pots y →
    x + y = 46 →
    x ≥ 2 * y →
    total_cost x y ≥ 369 :=
by
  sorry

#check minimum_cost_green_plants

end minimum_cost_green_plants_l1941_194117


namespace turnover_growth_equation_l1941_194176

/-- Represents the turnover growth of a supermarket from January to March -/
structure SupermarketGrowth where
  january_turnover : ℝ
  march_turnover : ℝ
  monthly_growth_rate : ℝ

/-- The equation correctly represents the relationship between turnovers and growth rate -/
theorem turnover_growth_equation (sg : SupermarketGrowth) 
  (h1 : sg.january_turnover = 36)
  (h2 : sg.march_turnover = 48) :
  sg.january_turnover * (1 + sg.monthly_growth_rate)^2 = sg.march_turnover :=
sorry

end turnover_growth_equation_l1941_194176


namespace hadley_total_distance_l1941_194134

-- Define the distances
def distance_to_grocery : ℕ := 2
def distance_to_pet_store : ℕ := 2 - 1
def distance_to_home : ℕ := 4 - 1

-- State the theorem
theorem hadley_total_distance :
  distance_to_grocery + distance_to_pet_store + distance_to_home = 6 := by
  sorry

end hadley_total_distance_l1941_194134


namespace monotone_increasing_iff_t_geq_five_l1941_194131

def a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
def b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

def f (x t : ℝ) : ℝ := (a x).1 * (b x t).1 + (a x).2 * (b x t).2

theorem monotone_increasing_iff_t_geq_five :
  ∀ t : ℝ, (∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x t < f y t) ↔ t ≥ 5 :=
by sorry

end monotone_increasing_iff_t_geq_five_l1941_194131


namespace cube_side_ratio_l1941_194150

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 25 → a / b = 5 := by
  sorry

end cube_side_ratio_l1941_194150


namespace completing_square_l1941_194120

theorem completing_square (x : ℝ) : x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
  sorry

end completing_square_l1941_194120


namespace holiday_duty_arrangements_l1941_194144

def staff_count : ℕ := 6
def days_count : ℕ := 3
def staff_per_day : ℕ := 2

def arrangement_count (n m k : ℕ) (restricted_days : ℕ) : ℕ :=
  (Nat.choose n k * Nat.choose (n - k) k) -
  (restricted_days * Nat.choose (n - 1) k * Nat.choose (n - k - 1) k) +
  (Nat.choose (n - 2) k * Nat.choose (n - k - 2) k)

theorem holiday_duty_arrangements :
  arrangement_count staff_count days_count staff_per_day 2 = 42 := by
  sorry

end holiday_duty_arrangements_l1941_194144


namespace pencil_sharpening_hours_l1941_194102

/-- The number of times Jenine can sharpen a pencil before it runs out -/
def sharpen_times : ℕ := 5

/-- The number of pencils Jenine already has -/
def initial_pencils : ℕ := 10

/-- The total number of hours Jenine needs to write -/
def total_writing_hours : ℕ := 105

/-- The cost of a new pencil in dollars -/
def pencil_cost : ℕ := 2

/-- The amount Jenine needs to spend on more pencils in dollars -/
def additional_pencil_cost : ℕ := 8

/-- The number of hours of use Jenine gets from sharpening a pencil once -/
def hours_per_sharpen : ℚ := 1.5

theorem pencil_sharpening_hours :
  let total_pencils := initial_pencils + additional_pencil_cost / pencil_cost
  total_pencils * sharpen_times * hours_per_sharpen = total_writing_hours :=
by sorry

end pencil_sharpening_hours_l1941_194102


namespace system_negative_solution_l1941_194162

theorem system_negative_solution (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧
    a * x + b * y = c ∧
    b * x + c * y = a ∧
    c * x + a * y = b) ↔
  a + b + c = 0 := by
  sorry

end system_negative_solution_l1941_194162


namespace intersection_point_y_value_l1941_194130

def f (x : ℝ) := 2 * x^2 - 3 * x + 10

theorem intersection_point_y_value :
  ∀ c : ℝ, f 7 = c → c = 87 := by
  sorry

end intersection_point_y_value_l1941_194130


namespace kaleb_ferris_wheel_cost_l1941_194135

/-- The amount of money Kaleb spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Kaleb spent 27 dollars on the ferris wheel ride -/
theorem kaleb_ferris_wheel_cost :
  ferris_wheel_cost 6 3 9 = 27 := by
  sorry

end kaleb_ferris_wheel_cost_l1941_194135


namespace min_value_expression_l1941_194163

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (2 * a / b) + (3 * b / c) + (4 * c / a) ≥ 9 ∧
  ((2 * a / b) + (3 * b / c) + (4 * c / a) = 9 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end min_value_expression_l1941_194163
