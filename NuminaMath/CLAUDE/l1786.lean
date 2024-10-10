import Mathlib

namespace inequality_proof_l1786_178658

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e) (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := by
  sorry

end inequality_proof_l1786_178658


namespace george_dimes_count_l1786_178684

/-- Prove the number of dimes in George's collection --/
theorem george_dimes_count :
  let total_coins : ℕ := 28
  let total_value : ℚ := 260/100
  let nickel_count : ℕ := 4
  let nickel_value : ℚ := 5/100
  let dime_value : ℚ := 10/100
  ∃ dime_count : ℕ,
    dime_count = 24 ∧
    dime_count + nickel_count = total_coins ∧
    dime_count * dime_value + nickel_count * nickel_value = total_value :=
by sorry

end george_dimes_count_l1786_178684


namespace shirt_cost_l1786_178663

theorem shirt_cost (J S K : ℚ) 
  (eq1 : 3 * J + 2 * S + K = 110)
  (eq2 : 2 * J + 3 * S + 2 * K = 176)
  (eq3 : 4 * J + S + 3 * K = 254) :
  S = 5.6 := by
  sorry

end shirt_cost_l1786_178663


namespace waste_paper_collection_l1786_178622

/-- Proves that given the conditions of the waste paper collection problem,
    Vitya collected 15 kg and Vova collected 12 kg. -/
theorem waste_paper_collection :
  ∀ (v w : ℕ),
  v + w = 27 →
  5 * v + 3 * w = 111 →
  v = 15 ∧ w = 12 := by
sorry

end waste_paper_collection_l1786_178622


namespace smallest_divisible_by_three_l1786_178686

theorem smallest_divisible_by_three :
  ∃ (B : ℕ), B < 10 ∧ 
    (∀ (k : ℕ), k < B → ¬(800000 + 100000 * k + 4635) % 3 = 0) ∧
    (800000 + 100000 * B + 4635) % 3 = 0 :=
by sorry

end smallest_divisible_by_three_l1786_178686


namespace min_shift_for_monotonic_decrease_l1786_178695

open Real

theorem min_shift_for_monotonic_decrease (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = sin (2*x + 2*m + π/6)) →
  (∀ x ∈ [-π/12, 5*π/12], ∀ y ∈ [-π/12, 5*π/12], x < y → f x > f y) →
  m > 0 →
  m ≥ π/4 :=
by sorry

end min_shift_for_monotonic_decrease_l1786_178695


namespace fiftieth_central_ring_number_l1786_178604

/-- Returns the number of digits in a positive integer -/
def numDigits (n : ℕ+) : ℕ :=
  (Nat.log 10 n.val) + 1

/-- Defines a Central Ring Number -/
def isCentralRingNumber (x : ℕ+) : Prop :=
  numDigits (3 * x) > numDigits x

/-- Returns the nth Central Ring Number -/
def nthCentralRingNumber (n : ℕ+) : ℕ+ :=
  sorry

theorem fiftieth_central_ring_number :
  nthCentralRingNumber 50 = 81 :=
sorry

end fiftieth_central_ring_number_l1786_178604


namespace geometric_progression_b_equals_four_l1786_178602

-- Define a geometric progression
def is_geometric_progression (seq : Fin 5 → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ i : Fin 4, seq (i + 1) = seq i * q

-- State the theorem
theorem geometric_progression_b_equals_four
  (seq : Fin 5 → ℝ)
  (h_gp : is_geometric_progression seq)
  (h_first : seq 0 = 1)
  (h_last : seq 4 = 16) :
  seq 2 = 4 := by
sorry

end geometric_progression_b_equals_four_l1786_178602


namespace shorter_tank_radius_l1786_178659

/-- Given two cylindrical tanks with equal volume, where one tank's height is four times the other
    and the taller tank has a radius of 10 units, the radius of the shorter tank is 20 units. -/
theorem shorter_tank_radius (h : ℝ) (h_pos : h > 0) : 
  π * (10 ^ 2) * (4 * h) = π * (20 ^ 2) * h := by sorry

end shorter_tank_radius_l1786_178659


namespace y_gets_0_45_per_x_rupee_l1786_178615

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ  -- amount x gets
  y : ℝ  -- amount y gets
  z : ℝ  -- amount z gets
  a : ℝ  -- amount y gets for each rupee x gets

/-- Conditions of the money distribution problem -/
def valid_distribution (d : MoneyDistribution) : Prop :=
  d.z = 0.5 * d.x ∧  -- z gets 50 paisa for each rupee x gets
  d.y = 27 ∧  -- y's share is 27 rupees
  d.x + d.y + d.z = 117 ∧  -- total amount is 117 rupees
  d.y = d.a * d.x  -- relationship between y's share and x's share

/-- Theorem stating that under the given conditions, y gets 0.45 rupees for each rupee x gets -/
theorem y_gets_0_45_per_x_rupee (d : MoneyDistribution) :
  valid_distribution d → d.a = 0.45 := by
  sorry


end y_gets_0_45_per_x_rupee_l1786_178615


namespace linear_equation_exponent_l1786_178643

theorem linear_equation_exponent (n : ℕ) : 
  (∀ x, ∃ a b, x^(2*n - 5) - 2 = a*x + b) → n = 3 := by
  sorry

end linear_equation_exponent_l1786_178643


namespace greatest_integer_satisfying_inequality_l1786_178680

def satisfies_inequality (x : ℤ) : Prop :=
  |7 * x - 3| - 2 * x < 5 - 3 * x

theorem greatest_integer_satisfying_inequality :
  satisfies_inequality 0 ∧
  ∀ y : ℤ, y > 0 → ¬(satisfies_inequality y) :=
sorry

end greatest_integer_satisfying_inequality_l1786_178680


namespace quadratic_equation_properties_l1786_178645

/-- A quadratic equation x^2 - 2kx + k^2 + k + 1 = 0 with two real roots -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*k*x + k^2 + k + 1 = 0

/-- The equation has two real roots -/
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ ∧ quadratic_equation k x₂

theorem quadratic_equation_properties :
  (∀ k : ℝ, has_two_real_roots k → k ≤ -1) ∧
  (∀ k : ℝ, has_two_real_roots k → (∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁^2 + x₂^2 = 10) → k = -2) ∧
  (∀ k : ℝ, has_two_real_roots k → (∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ |x₁| + |x₂| = 2) → k = -1) :=
by sorry

end quadratic_equation_properties_l1786_178645


namespace opposite_of_three_l1786_178689

theorem opposite_of_three : -(3 : ℝ) = -3 := by sorry

end opposite_of_three_l1786_178689


namespace bakers_purchase_problem_l1786_178682

/-- A baker's purchase problem -/
theorem bakers_purchase_problem 
  (total_cost : ℕ)
  (flour_cost : ℕ)
  (egg_cost : ℕ)
  (egg_quantity : ℕ)
  (milk_cost : ℕ)
  (milk_quantity : ℕ)
  (soda_cost : ℕ)
  (soda_quantity : ℕ)
  (h1 : total_cost = 80)
  (h2 : flour_cost = 3)
  (h3 : egg_cost = 10)
  (h4 : egg_quantity = 3)
  (h5 : milk_cost = 5)
  (h6 : milk_quantity = 7)
  (h7 : soda_cost = 3)
  (h8 : soda_quantity = 2) :
  ∃ (flour_quantity : ℕ), 
    flour_quantity * flour_cost + 
    egg_quantity * egg_cost + 
    milk_quantity * milk_cost + 
    soda_quantity * soda_cost = total_cost ∧ 
    flour_quantity = 3 :=
by sorry

end bakers_purchase_problem_l1786_178682


namespace x_value_l1786_178639

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 12 ∧ x = 28 := by
  sorry

end x_value_l1786_178639


namespace percent_equality_l1786_178624

theorem percent_equality (x : ℝ) (h : (0.3 * (0.2 * x)) = 24) :
  (0.2 * (0.3 * x)) = 24 := by
  sorry

end percent_equality_l1786_178624


namespace cos_is_even_l1786_178674

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem cos_is_even : IsEven Real.cos := by
  sorry

end cos_is_even_l1786_178674


namespace guaranteed_scores_theorem_l1786_178603

/-- Represents a player in the card game -/
inductive Player : Type
| First : Player
| Second : Player

/-- The card game with given conditions -/
structure CardGame where
  first_player_cards : Finset Nat
  second_player_cards : Finset Nat
  total_turns : Nat

/-- Define the game with the given conditions -/
def game : CardGame :=
  { first_player_cards := Finset.range 1000 |>.image (fun n => 2 * n + 2),
    second_player_cards := Finset.range 1001 |>.image (fun n => 2 * n + 1),
    total_turns := 1000 }

/-- The score a player can guarantee for themselves -/
def guaranteed_score (player : Player) (g : CardGame) : Nat :=
  match player with
  | Player.First => g.total_turns - 1
  | Player.Second => 1

/-- Theorem stating the guaranteed scores for both players -/
theorem guaranteed_scores_theorem (g : CardGame) :
  (guaranteed_score Player.First g = 999) ∧
  (guaranteed_score Player.Second g = 1) :=
sorry

end guaranteed_scores_theorem_l1786_178603


namespace children_got_on_bus_l1786_178642

/-- Proves that the number of children who got on the bus at the bus stop is 14 -/
theorem children_got_on_bus (initial_children : ℕ) (final_children : ℕ) 
  (h1 : initial_children = 64) (h2 : final_children = 78) : 
  final_children - initial_children = 14 := by
  sorry

end children_got_on_bus_l1786_178642


namespace taehun_shortest_hair_l1786_178606

-- Define the hair lengths
def junseop_hair_cm : ℝ := 9
def junseop_hair_mm : ℝ := 8
def taehun_hair : ℝ := 8.9
def hayul_hair : ℝ := 9.3

-- Define the conversion factor from mm to cm
def mm_to_cm : ℝ := 0.1

-- Theorem statement
theorem taehun_shortest_hair :
  let junseop_total := junseop_hair_cm + junseop_hair_mm * mm_to_cm
  taehun_hair < junseop_total ∧ taehun_hair < hayul_hair := by sorry

end taehun_shortest_hair_l1786_178606


namespace penny_difference_l1786_178652

theorem penny_difference (kate_pennies john_pennies : ℕ) 
  (h1 : kate_pennies = 223) 
  (h2 : john_pennies = 388) : 
  john_pennies - kate_pennies = 165 := by
sorry

end penny_difference_l1786_178652


namespace binomial_coefficient_congruence_l1786_178664

theorem binomial_coefficient_congruence (p n : ℕ) (hp : Prime p) :
  (Nat.choose (n * p) n) ≡ n [MOD p^2] := by sorry

end binomial_coefficient_congruence_l1786_178664


namespace rectangle_dimensions_l1786_178618

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- Theorem: A rectangle with perimeter 150 cm and length 15 cm greater than width
    has width 30 cm and length 45 cm -/
theorem rectangle_dimensions :
  ∃ (r : Rectangle),
    perimeter r = 150 ∧
    r.length = r.width + 15 ∧
    r.width = 30 ∧
    r.length = 45 := by
  sorry

end rectangle_dimensions_l1786_178618


namespace inequality_max_m_l1786_178616

theorem inequality_max_m : 
  ∀ (m : ℝ), 
  (∀ (a b : ℝ), a > 0 → b > 0 → (2/a + 1/b ≥ m/(2*a+b))) ↔ m ≤ 9 :=
by sorry

end inequality_max_m_l1786_178616


namespace find_divisor_l1786_178613

theorem find_divisor (D : ℕ) (x : ℕ) : 
  (∃ (x : ℕ), x ≤ 11 ∧ (2000 - x) % D = 0) → 
  (2000 - x = 1989) →
  D = 11 := by
sorry

end find_divisor_l1786_178613


namespace certain_number_multiplication_l1786_178625

theorem certain_number_multiplication : ∃ x : ℤ, (x - 7 = 9) ∧ (x * 3 = 48) := by
  sorry

end certain_number_multiplication_l1786_178625


namespace negation_equivalence_l1786_178679

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∀ x : ℝ, (x - a)^2 + 2 > 0) ↔ (∃ x : ℝ, (x - a)^2 + 2 ≤ 0) :=
by sorry

end negation_equivalence_l1786_178679


namespace det_special_matrix_is_zero_l1786_178670

open Matrix

theorem det_special_matrix_is_zero (x y z : ℝ) : 
  det ![![1, x + z, y - z],
       ![1, x + y + z, y - z],
       ![1, x + z, x + y]] = 0 := by
  sorry

end det_special_matrix_is_zero_l1786_178670


namespace imaginary_sum_zero_l1786_178694

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^15324 + i^15325 + i^15326 + i^15327 = 0 := by
  sorry

end imaginary_sum_zero_l1786_178694


namespace quiz_show_probability_l1786_178638

-- Define the number of questions and choices
def num_questions : ℕ := 4
def num_choices : ℕ := 4

-- Define the minimum number of correct answers needed to win
def min_correct : ℕ := 3

-- Define the probability of guessing a single question correctly
def prob_correct : ℚ := 1 / num_choices

-- Define the probability of guessing a single question incorrectly
def prob_incorrect : ℚ := 1 - prob_correct

-- Define the function to calculate the probability of winning
def prob_win : ℚ :=
  (num_questions.choose min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct)) +
  (prob_correct ^ num_questions)

-- State the theorem
theorem quiz_show_probability :
  prob_win = 13 / 256 := by sorry

end quiz_show_probability_l1786_178638


namespace other_endpoint_of_line_segment_l1786_178683

/-- Given a line segment with midpoint (-1, 3) and one endpoint (2, -4),
    prove that the other endpoint is (-4, 10). -/
theorem other_endpoint_of_line_segment
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (-1, 3))
  (h_endpoint1 : endpoint1 = (2, -4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-4, 10) ∧
    midpoint = (
      (endpoint1.1 + endpoint2.1) / 2,
      (endpoint1.2 + endpoint2.2) / 2
    ) :=
by sorry

end other_endpoint_of_line_segment_l1786_178683


namespace spinner_probability_l1786_178608

theorem spinner_probability : 
  ∀ (p_A p_B p_C p_D p_E : ℚ),
  p_A = 2/7 →
  p_B = 3/14 →
  p_C = p_E →
  p_D = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end spinner_probability_l1786_178608


namespace line_parallel_to_plane_l1786_178676

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m n)
  (h2 : perpendicularLP n α)
  (h3 : ¬ intersects m α) :
  parallel m α :=
sorry

end line_parallel_to_plane_l1786_178676


namespace mobile_phone_price_decrease_l1786_178688

theorem mobile_phone_price_decrease (current_price : ℝ) (yearly_decrease_rate : ℝ) (years : ℕ) : 
  current_price = 1000 ∧ yearly_decrease_rate = 0.2 ∧ years = 2 →
  current_price = (1562.5 : ℝ) * (1 - yearly_decrease_rate) ^ years := by
sorry

end mobile_phone_price_decrease_l1786_178688


namespace surface_area_increase_l1786_178648

/-- Given a cube with edge length a that is cut into 27 identical smaller cubes,
    the increase in surface area is 12a². -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  27 * 6 * (a / 3)^2 - 6 * a^2 = 12 * a^2 := by
  sorry

end surface_area_increase_l1786_178648


namespace rice_mixture_cost_l1786_178690

/-- 
Given two varieties of rice mixed in a specific ratio to create a mixture with a known cost,
this theorem proves the cost of the first variety of rice.
-/
theorem rice_mixture_cost 
  (cost_second : ℝ) 
  (cost_mixture : ℝ) 
  (mix_ratio : ℝ) 
  (h1 : cost_second = 8.75)
  (h2 : cost_mixture = 7.50)
  (h3 : mix_ratio = 0.625)
  : ∃ (cost_first : ℝ), cost_first = 8.28125 := by
  sorry

end rice_mixture_cost_l1786_178690


namespace fraction_evaluation_l1786_178612

theorem fraction_evaluation : (16 + 8) / (4 - 2) = 12 := by
  sorry

end fraction_evaluation_l1786_178612


namespace pie_crust_flour_calculation_l1786_178696

theorem pie_crust_flour_calculation (initial_crusts : ℕ) (new_crusts : ℕ) (initial_flour : ℚ) :
  initial_crusts = 36 →
  new_crusts = 24 →
  initial_flour = 1/8 →
  (initial_crusts : ℚ) * initial_flour = (new_crusts : ℚ) * ((3:ℚ)/16) :=
by sorry

end pie_crust_flour_calculation_l1786_178696


namespace cube_sum_of_roots_l1786_178677

theorem cube_sum_of_roots (u v w : ℝ) : 
  (u - Real.rpow 17 (1/3 : ℝ)) * (u - Real.rpow 67 (1/3 : ℝ)) * (u - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  (v - Real.rpow 17 (1/3 : ℝ)) * (v - Real.rpow 67 (1/3 : ℝ)) * (v - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  (w - Real.rpow 17 (1/3 : ℝ)) * (w - Real.rpow 67 (1/3 : ℝ)) * (w - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 211.75 := by
sorry

end cube_sum_of_roots_l1786_178677


namespace count_divisible_by_2_3_or_5_count_divisible_by_2_3_or_5_is_74_l1786_178655

theorem count_divisible_by_2_3_or_5 : ℕ :=
  let n := 100
  let A₂ := n / 2
  let A₃ := n / 3
  let A₅ := n / 5
  let A₂₃ := n / 6
  let A₂₅ := n / 10
  let A₃₅ := n / 15
  let A₂₃₅ := n / 30
  A₂ + A₃ + A₅ - A₂₃ - A₂₅ - A₃₅ + A₂₃₅

theorem count_divisible_by_2_3_or_5_is_74 : 
  count_divisible_by_2_3_or_5 = 74 := by sorry

end count_divisible_by_2_3_or_5_count_divisible_by_2_3_or_5_is_74_l1786_178655


namespace basketball_weight_is_20_l1786_178685

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 20

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 40

/-- Theorem stating that one basketball weighs 20 pounds given the conditions -/
theorem basketball_weight_is_20 :
  (8 * basketball_weight = 4 * bicycle_weight) →
  (3 * bicycle_weight = 120) →
  basketball_weight = 20 := by
sorry

end basketball_weight_is_20_l1786_178685


namespace slope_of_line_l1786_178629

theorem slope_of_line (x y : ℝ) :
  3 * x + 4 * y + 12 = 0 → (y - 0) / (x - 0) = -3 / 4 :=
by sorry

end slope_of_line_l1786_178629


namespace base_6_to_10_54123_l1786_178672

def base_6_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

theorem base_6_to_10_54123 :
  base_6_to_10 [3, 2, 1, 4, 5] = 7395 := by
  sorry

end base_6_to_10_54123_l1786_178672


namespace valentines_day_theorem_l1786_178631

theorem valentines_day_theorem (male_students female_students : ℕ) : 
  (male_students * female_students = male_students + female_students + 42) → 
  (male_students * female_students = 88) :=
by
  sorry

end valentines_day_theorem_l1786_178631


namespace second_planner_cheaper_l1786_178651

/-- Represents the cost function for an event planner -/
structure EventPlanner where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for an event planner given the number of people -/
def totalCost (planner : EventPlanner) (people : ℕ) : ℕ :=
  planner.basicFee + planner.perPersonFee * people

/-- The first event planner's pricing structure -/
def planner1 : EventPlanner := ⟨120, 18⟩

/-- The second event planner's pricing structure -/
def planner2 : EventPlanner := ⟨250, 15⟩

/-- Theorem stating the conditions for when the second planner becomes less expensive -/
theorem second_planner_cheaper (n : ℕ) :
  (n < 44 → totalCost planner1 n ≤ totalCost planner2 n) ∧
  (n ≥ 44 → totalCost planner2 n < totalCost planner1 n) :=
sorry

end second_planner_cheaper_l1786_178651


namespace fruit_sales_calculation_l1786_178681

/-- Calculate the total money collected from selling fruits with price increases -/
theorem fruit_sales_calculation (lemon_price grape_price orange_price apple_price : ℚ)
  (lemon_count grape_count orange_count apple_count : ℕ)
  (lemon_increase grape_increase orange_increase apple_increase : ℚ) :
  let new_lemon_price := lemon_price * (1 + lemon_increase)
  let new_grape_price := grape_price * (1 + grape_increase)
  let new_orange_price := orange_price * (1 + orange_increase)
  let new_apple_price := apple_price * (1 + apple_increase)
  lemon_count * new_lemon_price + grape_count * new_grape_price +
  orange_count * new_orange_price + apple_count * new_apple_price = 2995 :=
by
  sorry

#check fruit_sales_calculation 8 7 5 4 80 140 60 100 (1/2) (1/4) (1/10) (1/5)

end fruit_sales_calculation_l1786_178681


namespace hyperbola_eccentricity_hyperbola_eccentricity_is_sqrt_5_l1786_178678

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let P : ℝ × ℝ := sorry
  let F₁ : ℝ × ℝ := sorry
  let F₂ : ℝ × ℝ := sorry
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ x^2 + y^2 = a^2 + b^2
  let distance := fun (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  have h1 : hyperbola P.1 P.2 := sorry
  have h2 : circle P.1 P.2 := sorry
  have h3 : P.1 ≥ 0 ∧ P.2 ≥ 0 := sorry  -- P is in the first quadrant
  have h4 : distance P F₁ = 2 * distance P F₂ := sorry
  Real.sqrt 5

theorem hyperbola_eccentricity_is_sqrt_5 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = Real.sqrt 5 := by sorry

end hyperbola_eccentricity_hyperbola_eccentricity_is_sqrt_5_l1786_178678


namespace can_lids_per_box_l1786_178623

theorem can_lids_per_box 
  (num_boxes : ℕ) 
  (initial_lids : ℕ) 
  (final_total_lids : ℕ) 
  (h1 : num_boxes = 3) 
  (h2 : initial_lids = 14) 
  (h3 : final_total_lids = 53) :
  (final_total_lids - initial_lids) / num_boxes = 13 := by
  sorry

#check can_lids_per_box

end can_lids_per_box_l1786_178623


namespace chord_length_concentric_circles_l1786_178662

theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (R^2 - r^2 = 20) →
  ∃ c : ℝ, c > 0 ∧ c^2 / 4 + r^2 = R^2 ∧ c = 4 * Real.sqrt 5 :=
by sorry

end chord_length_concentric_circles_l1786_178662


namespace trig_identity_l1786_178661

theorem trig_identity (x y : ℝ) :
  Real.cos x ^ 2 + Real.cos (x + y) ^ 2 - 2 * Real.cos x * Real.cos y * Real.cos (x + y) = Real.sin y ^ 2 := by
  sorry

end trig_identity_l1786_178661


namespace ashwin_rental_hours_verify_solution_l1786_178637

/-- Calculates the total rental hours given the rental conditions and total amount paid --/
def rental_hours (first_hour_cost : ℕ) (additional_hour_cost : ℕ) (total_paid : ℕ) : ℕ :=
  let additional_hours := (total_paid - first_hour_cost) / additional_hour_cost
  1 + additional_hours

/-- Proves that Ashwin rented the tool for 11 hours given the specified conditions --/
theorem ashwin_rental_hours :
  rental_hours 25 10 125 = 11 := by
  sorry

/-- Verifies the solution satisfies the original problem conditions --/
theorem verify_solution :
  25 + 10 * (rental_hours 25 10 125 - 1) = 125 := by
  sorry

end ashwin_rental_hours_verify_solution_l1786_178637


namespace product_of_fractions_l1786_178671

theorem product_of_fractions :
  (3 : ℚ) / 5 * (4 : ℚ) / 7 * (5 : ℚ) / 9 = (4 : ℚ) / 21 := by
  sorry

end product_of_fractions_l1786_178671


namespace negative_a_sixth_divided_by_a_third_l1786_178666

theorem negative_a_sixth_divided_by_a_third (a : ℝ) : (-a)^6 / a^3 = a^3 := by
  sorry

end negative_a_sixth_divided_by_a_third_l1786_178666


namespace eel_species_count_l1786_178646

/-- Given the number of species identified in an aquarium, prove the number of eel species. -/
theorem eel_species_count (total : ℕ) (sharks : ℕ) (whales : ℕ) (h1 : total = 55) (h2 : sharks = 35) (h3 : whales = 5) :
  total - sharks - whales = 15 := by
  sorry

end eel_species_count_l1786_178646


namespace P_value_at_seven_l1786_178665

-- Define the polynomial P(x)
def P (a b c d e f : ℝ) (x : ℂ) : ℂ :=
  (3 * x^4 - 39 * x^3 + a * x^2 + b * x + c) *
  (4 * x^4 - 64 * x^3 + d * x^2 + e * x + f)

-- State the theorem
theorem P_value_at_seven 
  (a b c d e f : ℝ) 
  (h : Set.range (fun (x : ℂ) => P a b c d e f x) = {1, 2, 3, 4, 6}) : 
  P a b c d e f 7 = 69120 := by
sorry

end P_value_at_seven_l1786_178665


namespace maryann_client_call_time_l1786_178673

theorem maryann_client_call_time (total_time accounting_time client_time : ℕ) : 
  total_time = 560 →
  accounting_time = 7 * client_time →
  total_time = accounting_time + client_time →
  client_time = 70 := by
sorry

end maryann_client_call_time_l1786_178673


namespace even_function_inequality_l1786_178617

/-- An even function from ℝ to ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf_even : EvenFunction f)
  (hf_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x, 2 * f x + x * f' x < 2) :
  ∀ x, x^2 * f x - f 1 < x^2 - 1 ↔ |x| > 1 := by
sorry

end even_function_inequality_l1786_178617


namespace triangle_perimeter_l1786_178657

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 4 → Even c → 
  a + b > c ∧ b + c > a ∧ c + a > b → 
  a + b + c = 10 :=
sorry

end triangle_perimeter_l1786_178657


namespace bart_firewood_needs_l1786_178634

/-- The number of pieces of firewood obtained from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of logs burned per day -/
def logs_per_day : ℕ := 5

/-- The number of days from November 1 through February 28 -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_needed : ℕ := 8

theorem bart_firewood_needs :
  trees_needed = (total_days * logs_per_day + pieces_per_tree - 1) / pieces_per_tree :=
by sorry

end bart_firewood_needs_l1786_178634


namespace inequality_solution_set_l1786_178656

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, ax - b > 0 ↔ x > 1) →
  (∀ x, (a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by sorry

end inequality_solution_set_l1786_178656


namespace intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l1786_178644

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Theorem for part (1)
theorem intersection_A_complement_B (m : ℝ) (h : m = 3) :
  A ∩ (Set.univ \ B m) = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for part (2)
theorem intersection_A_B_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

-- Theorem for part (3)
theorem intersection_A_B_equals_A (m : ℝ) :
  A ∩ B m = A ↔ m ≥ 4 := by sorry

end intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l1786_178644


namespace time_to_see_again_is_48_l1786_178620

-- Define the parameters of the problem
def path_distance : ℝ := 300
def building_diameter : ℝ := 150
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def initial_distance : ℝ := 300

-- Define the function to calculate the time until they can see each other again
def time_to_see_again (pd : ℝ) (bd : ℝ) (ks : ℝ) (js : ℝ) (id : ℝ) : ℝ :=
  -- The actual calculation would go here, but we'll use sorry to skip the proof
  sorry

-- State the theorem
theorem time_to_see_again_is_48 :
  time_to_see_again path_distance building_diameter kenny_speed jenny_speed initial_distance = 48 := by
  sorry

end time_to_see_again_is_48_l1786_178620


namespace log_inequality_iff_x_range_l1786_178669

-- Define the domain constraints
def domain (x : ℝ) : Prop := x > -2 ∧ x ≠ -1

-- Define the logarithmic inequality
def log_inequality (x : ℝ) : Prop :=
  Real.log (8 + x^3) / Real.log (2 + x) ≤ Real.log ((2 + x)^3) / Real.log (2 + x)

-- State the theorem
theorem log_inequality_iff_x_range (x : ℝ) :
  domain x → (log_inequality x ↔ (-2 < x ∧ x < -1) ∨ x ≥ 0) :=
by sorry

end log_inequality_iff_x_range_l1786_178669


namespace vector_colinearity_l1786_178600

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (2, 4)

def colinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 * w.2 = t * v.2 * w.1

theorem vector_colinearity (k : ℝ) :
  colinear (a.1 + k * b.1, a.2 + k * b.2) c →
  k = -1/3 :=
by sorry

end vector_colinearity_l1786_178600


namespace partial_fraction_decomposition_l1786_178619

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 12) (h2 : x ≠ -4) :
  (7 * x - 5) / (x^2 - 8*x - 48) = (79/16) / (x - 12) + (33/16) / (x + 4) := by
  sorry

end partial_fraction_decomposition_l1786_178619


namespace tan_alpha_plus_beta_l1786_178693

theorem tan_alpha_plus_beta (α β : Real) 
  (h1 : Real.tan α = 1) 
  (h2 : 3 * Real.sin β = Real.sin (2 * α + β)) : 
  Real.tan (α + β) = 2 := by sorry

end tan_alpha_plus_beta_l1786_178693


namespace smallest_multiple_l1786_178653

theorem smallest_multiple (n : ℕ) (h : n = 5) : 
  (∃ m : ℕ, m * n - 15 > 2 * n ∧ ∀ k : ℕ, k < m → k * n - 15 ≤ 2 * n) → 
  (∃ m : ℕ, m * n - 15 > 2 * n ∧ ∀ k : ℕ, k < m → k * n - 15 ≤ 2 * n ∧ m = 6) :=
by sorry

end smallest_multiple_l1786_178653


namespace quadratic_equation_solution_l1786_178641

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 7
  ∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 11 ∧ x2 = 2 - Real.sqrt 11 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_equation_solution_l1786_178641


namespace unique_lcm_solution_l1786_178607

theorem unique_lcm_solution : ∃! (n : ℕ), n > 0 ∧ Nat.lcm n (n - 30) = n + 1320 :=
by
  -- The proof goes here
  sorry

end unique_lcm_solution_l1786_178607


namespace quadratic_solution_l1786_178675

theorem quadratic_solution (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8) - 15 = 0) → b = 49/8 := by
  sorry

end quadratic_solution_l1786_178675


namespace partner_A_investment_l1786_178697

/-- Calculates the investment of partner A in a business partnership --/
theorem partner_A_investment
  (b_investment : ℕ)
  (c_investment : ℕ)
  (total_profit : ℕ)
  (a_profit_share : ℕ)
  (h1 : b_investment = 4200)
  (h2 : c_investment = 10500)
  (h3 : total_profit = 12200)
  (h4 : a_profit_share = 3660) :
  ∃ a_investment : ℕ,
    a_investment = 6725 ∧
    a_investment * total_profit = a_profit_share * (a_investment + b_investment + c_investment) :=
by
  sorry


end partner_A_investment_l1786_178697


namespace circumscribed_trapezoid_inequality_l1786_178636

/-- A trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- Radius of the inscribed circle -/
  R : ℝ
  /-- Length of one base of the trapezoid -/
  a : ℝ
  /-- Length of the other base of the trapezoid -/
  b : ℝ
  /-- The trapezoid is circumscribed around the circle -/
  circumscribed : True

/-- For a trapezoid circumscribed around a circle with radius R and bases a and b, ab ≥ 4R^2 -/
theorem circumscribed_trapezoid_inequality (t : CircumscribedTrapezoid) : t.a * t.b ≥ 4 * t.R^2 := by
  sorry

end circumscribed_trapezoid_inequality_l1786_178636


namespace boys_to_girls_ratio_l1786_178667

/-- Given a total of 68 students in eighth grade with 28 girls, 
    the ratio of boys to girls is 10:7. -/
theorem boys_to_girls_ratio : 
  let total_students : ℕ := 68
  let girls : ℕ := 28
  let boys : ℕ := total_students - girls
  ∃ (a b : ℕ), a = 10 ∧ b = 7 ∧ boys * b = girls * a :=
by sorry

end boys_to_girls_ratio_l1786_178667


namespace cross_product_example_l1786_178698

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example : 
  cross_product (3, -4, 7) (2, 5, -3) = (-23, 23, 23) := by
  sorry

end cross_product_example_l1786_178698


namespace jane_cans_count_l1786_178687

theorem jane_cans_count (total_seeds : ℝ) (seeds_per_can : ℕ) (h1 : total_seeds = 54.0) (h2 : seeds_per_can = 6) :
  (total_seeds / seeds_per_can : ℝ) = 9 := by
  sorry

end jane_cans_count_l1786_178687


namespace f_composition_sqrt2_l1786_178611

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x + 1 else |x|

theorem f_composition_sqrt2 :
  f (f (-Real.sqrt 2)) = 3 * Real.sqrt 2 + 1 := by
  sorry

end f_composition_sqrt2_l1786_178611


namespace salary_problem_l1786_178649

theorem salary_problem (total : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ) 
  (h1 : total = 6000)
  (h2 : a_spend_rate = 0.95)
  (h3 : b_spend_rate = 0.85)
  (h4 : (1 - a_spend_rate) * a = (1 - b_spend_rate) * (total - a)) :
  a = 4500 :=
by
  sorry

end salary_problem_l1786_178649


namespace salary_comparison_l1786_178609

/-- Given salaries of A, B, and C with specified relationships, prove the percentage differences -/
theorem salary_comparison (a b c : ℝ) 
  (h1 : a = b * 0.8)  -- A's salary is 20% less than B's
  (h2 : c = a * 1.3)  -- C's salary is 30% more than A's
  : (b - a) / a = 0.25 ∧ (c - b) / b = 0.04 := by
  sorry

end salary_comparison_l1786_178609


namespace gcf_of_3150_and_9800_l1786_178660

theorem gcf_of_3150_and_9800 : Nat.gcd 3150 9800 = 350 := by
  sorry

end gcf_of_3150_and_9800_l1786_178660


namespace intersection_A_B_union_complement_A_B_l1786_178632

-- Define the universe set U
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x < 5} :=
sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 < x ∧ x ≤ 7)} :=
sorry

end intersection_A_B_union_complement_A_B_l1786_178632


namespace smallest_number_with_remainders_l1786_178627

theorem smallest_number_with_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 5 = 3 ∧
  b % 4 = 2 ∧
  b % 6 = 2 ∧
  (∀ c : ℕ, c > 0 ∧ c % 5 = 3 ∧ c % 4 = 2 ∧ c % 6 = 2 → b ≤ c) ∧
  b = 38 :=
by sorry

end smallest_number_with_remainders_l1786_178627


namespace julio_mocktail_days_l1786_178692

/-- The number of days Julio made mocktails given the specified conditions -/
def mocktail_days (lime_juice_per_mocktail : ℚ) (juice_per_lime : ℚ) (limes_per_dollar : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent * limes_per_dollar * juice_per_lime) / lime_juice_per_mocktail

/-- Theorem stating that Julio made mocktails for 30 days under the given conditions -/
theorem julio_mocktail_days :
  mocktail_days 1 2 3 5 = 30 := by
  sorry

end julio_mocktail_days_l1786_178692


namespace house_savings_l1786_178610

theorem house_savings (total_savings : ℕ) (years : ℕ) (people : ℕ) : 
  total_savings = 108000 → 
  years = 3 → 
  people = 2 → 
  (total_savings / (years * 12)) / people = 1500 := by
sorry

end house_savings_l1786_178610


namespace equal_perimeters_shapes_l1786_178621

theorem equal_perimeters_shapes (x y : ℝ) : 
  (4 * (x + 2) = 6 * x) ∧ (6 * x = 2 * Real.pi * y) → x = 4 ∧ y = 12 / Real.pi := by
  sorry

end equal_perimeters_shapes_l1786_178621


namespace sequence_general_term_l1786_178654

/-- A sequence satisfying the given recurrence relation -/
def Sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n + 2 * a (n + 1) = 7 * 3^(n - 1)) ∧ a 1 = 1

/-- Theorem stating that the sequence has the general term a_n = 3^(n-1) -/
theorem sequence_general_term (a : ℕ → ℝ) (h : Sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) := by
  sorry

end sequence_general_term_l1786_178654


namespace total_cats_is_twenty_l1786_178635

-- Define the number of cats for each person
def jamie_persian : Nat := 4
def jamie_maine_coon : Nat := 2
def gordon_persian : Nat := jamie_persian / 2
def gordon_maine_coon : Nat := jamie_maine_coon + 1
def hawkeye_persian : Nat := 0
def hawkeye_maine_coon : Nat := gordon_maine_coon - 1
def natasha_persian : Nat := 3
def natasha_maine_coon : Nat := 4

-- Define the total number of cats
def total_cats : Nat :=
  jamie_persian + jamie_maine_coon +
  gordon_persian + gordon_maine_coon +
  hawkeye_persian + hawkeye_maine_coon +
  natasha_persian + natasha_maine_coon

-- Theorem to prove
theorem total_cats_is_twenty : total_cats = 20 := by
  sorry

end total_cats_is_twenty_l1786_178635


namespace train_speed_l1786_178630

/-- The speed of a train given its length and time to pass an observer -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  passing_time = 12 →
  (train_length / 1000) / (passing_time / 3600) = 30 := by
  sorry

end train_speed_l1786_178630


namespace complex_modulus_l1786_178668

theorem complex_modulus (z : ℂ) :
  (((2 : ℂ) + 4 * I) / z = 1 + I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l1786_178668


namespace day_after_tomorrow_l1786_178633

/-- Represents days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns the previous day -/
def prevDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Saturday
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday

theorem day_after_tomorrow (today : Day) :
  (nextDay (nextDay today) = Day.Saturday) → (today = Day.Thursday) →
  (prevDay (nextDay (nextDay today)) = Day.Friday) :=
by
  sorry


end day_after_tomorrow_l1786_178633


namespace least_four_digit_multiple_l1786_178614

theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → 3 ∣ m → 7 ∣ m → 11 ∣ m → n ≤ m) ∧
  n = 1155 :=
by sorry

end least_four_digit_multiple_l1786_178614


namespace ellipse_intersection_midpoint_l1786_178626

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  y = x + 2

-- Theorem statement
theorem ellipse_intersection_midpoint :
  -- Given conditions
  let f1 : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let f2 : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let major_axis_length : ℝ := 6

  -- Prove that
  -- 1. The standard equation of ellipse C is x²/9 + y² = 1
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 9 + y^2 = 1) ∧
  -- 2. The midpoint of intersection points has coordinates (-9/5, 1/5)
  (∃ x1 y1 x2 y2 : ℝ,
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line x1 y1 ∧ line x2 y2 ∧
    x1 ≠ x2 ∧
    (x1 + x2) / 2 = -9/5 ∧
    (y1 + y2) / 2 = 1/5) :=
by sorry

end ellipse_intersection_midpoint_l1786_178626


namespace veranda_area_is_196_l1786_178691

/-- Represents the dimensions and characteristics of a room with a trapezoidal veranda. -/
structure RoomWithVeranda where
  room_length : ℝ
  room_width : ℝ
  veranda_short_side : ℝ
  veranda_long_side : ℝ

/-- Calculates the area of the trapezoidal veranda surrounding the room. -/
def verandaArea (r : RoomWithVeranda) : ℝ :=
  (r.room_length + 2 * r.veranda_long_side) * (r.room_width + 2 * r.veranda_short_side) - r.room_length * r.room_width

/-- Theorem stating that the area of the trapezoidal veranda is 196 m² for the given dimensions. -/
theorem veranda_area_is_196 (r : RoomWithVeranda)
    (h1 : r.room_length = 17)
    (h2 : r.room_width = 12)
    (h3 : r.veranda_short_side = 2)
    (h4 : r.veranda_long_side = 4) :
    verandaArea r = 196 := by
  sorry

#eval verandaArea { room_length := 17, room_width := 12, veranda_short_side := 2, veranda_long_side := 4 }

end veranda_area_is_196_l1786_178691


namespace parallel_vectors_sum_magnitude_l1786_178640

/-- Given vectors p and q in ℝ², where p is parallel to q, prove that |p + q| = √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) :
  p = (2, -3) →
  q.1 = x ∧ q.2 = 6 →
  (∃ (k : ℝ), q = k • p) →
  ‖p + q‖ = Real.sqrt 13 := by
  sorry

end parallel_vectors_sum_magnitude_l1786_178640


namespace six_digit_divisibility_l1786_178605

/-- Given a two-digit number, constructs a six-digit number by repeating it three times -/
def repeat_twice (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + n

/-- Theorem: For any two-digit number, the six-digit number formed by repeating it three times is divisible by 10101 -/
theorem six_digit_divisibility (n : ℕ) (h : n ≥ 10 ∧ n ≤ 99) : 
  (repeat_twice n) % 10101 = 0 := by
  sorry


end six_digit_divisibility_l1786_178605


namespace not_divisible_by_seven_l1786_178628

theorem not_divisible_by_seven (n : ℤ) : ¬(7 ∣ (n^2 + 1)) := by
  sorry

end not_divisible_by_seven_l1786_178628


namespace last_segment_speed_l1786_178650

/-- Proves that the average speed for the last segment is 67 mph given the conditions of the problem -/
theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (first_segment_speed : ℝ) (second_segment_speed : ℝ) : ℝ :=
  by
  have h1 : total_distance = 96 := by sorry
  have h2 : total_time = 90 / 60 := by sorry
  have h3 : first_segment_speed = 60 := by sorry
  have h4 : second_segment_speed = 65 := by sorry
  
  let overall_average_speed := total_distance / total_time
  have h5 : overall_average_speed = 64 := by sorry
  
  let last_segment_speed := 3 * overall_average_speed - first_segment_speed - second_segment_speed
  
  exact last_segment_speed

end last_segment_speed_l1786_178650


namespace square_side_length_l1786_178699

theorem square_side_length (d : ℝ) (s : ℝ) : d = 2 * Real.sqrt 2 → s * Real.sqrt 2 = d → s = 2 := by
  sorry

end square_side_length_l1786_178699


namespace shanghai_masters_matches_l1786_178601

/-- Represents the Shanghai Masters tennis tournament structure -/
structure ShangHaiMasters where
  totalPlayers : Nat
  groupCount : Nat
  playersPerGroup : Nat
  advancingPerGroup : Nat

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total number of matches in the Shanghai Masters tournament -/
def totalMatches (tournament : ShangHaiMasters) : Nat :=
  let groupMatches := tournament.groupCount * roundRobinMatches tournament.playersPerGroup
  let knockoutMatches := tournament.groupCount * tournament.advancingPerGroup / 2
  let finalMatches := 2
  groupMatches + knockoutMatches + finalMatches

/-- Theorem stating that the total number of matches in the Shanghai Masters is 16 -/
theorem shanghai_masters_matches :
  ∃ (tournament : ShangHaiMasters),
    tournament.totalPlayers = 8 ∧
    tournament.groupCount = 2 ∧
    tournament.playersPerGroup = 4 ∧
    tournament.advancingPerGroup = 2 ∧
    totalMatches tournament = 16 := by
  sorry


end shanghai_masters_matches_l1786_178601


namespace water_volume_in_cone_l1786_178647

/-- 
Theorem: For a cone filled with water up to 2/3 of its height, 
the volume of water is 8/27 of the total volume of the cone.
-/
theorem water_volume_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  (1/3 * π * (2/3 * r)^2 * (2/3 * h)) / (1/3 * π * r^2 * h) = 8/27 := by
  sorry


end water_volume_in_cone_l1786_178647
