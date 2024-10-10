import Mathlib

namespace min_correct_answers_for_target_score_l2014_201434

/-- Represents a quiz with specified scoring rules -/
structure Quiz where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_deduction : ℕ

/-- Calculates the score for a given number of correct answers -/
def score (q : Quiz) (correct_answers : ℕ) : ℤ :=
  (q.correct_points * correct_answers : ℤ) - 
  (q.incorrect_deduction * (q.total_questions - correct_answers) : ℤ)

/-- Theorem stating the minimum number of correct answers needed to achieve a target score -/
theorem min_correct_answers_for_target_score 
  (q : Quiz) 
  (target_score : ℤ) 
  (correct_answers : ℕ) :
  q.total_questions = 20 →
  q.correct_points = 5 →
  q.incorrect_deduction = 1 →
  target_score = 88 →
  score q correct_answers ≥ target_score →
  (5 : ℤ) * correct_answers - (20 - correct_answers) ≥ 88 := by
  sorry

#check min_correct_answers_for_target_score

end min_correct_answers_for_target_score_l2014_201434


namespace quadratic_minimum_value_l2014_201485

/-- Represents a quadratic function of the form y = a(x-m)(x-m-k) -/
def quadratic_function (a m k x : ℝ) : ℝ := a * (x - m) * (x - m - k)

/-- The minimum value of the quadratic function when k = 2 -/
def min_value (a m : ℝ) : ℝ := -a

theorem quadratic_minimum_value (a m : ℝ) (h : a > 0) :
  ∃ x, quadratic_function a m 2 x = min_value a m ∧
  ∀ y, quadratic_function a m 2 y ≥ min_value a m :=
by sorry

end quadratic_minimum_value_l2014_201485


namespace lcm_of_385_and_180_l2014_201416

theorem lcm_of_385_and_180 :
  let a := 385
  let b := 180
  let hcf := 30
  Nat.lcm a b = 2310 :=
by
  sorry

end lcm_of_385_and_180_l2014_201416


namespace problem_solution_l2014_201441

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x) / x + b

def has_tangent_line (f : ℝ → ℝ) (x₀ y₀ m : ℝ) : Prop :=
  ∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧ f' x₀ = m ∧ f x₀ = y₀

noncomputable def g (c x : ℝ) : ℝ := Real.log x / Real.log c - x

def has_zero_point (g : ℝ → ℝ) : Prop := ∃ x > 0, g x = 0

theorem problem_solution :
  ∀ (a b : ℝ),
    (has_tangent_line (f a b) 1 0 1) →
    (a = 1 ∧ b = 0) ∧
    (∀ x > 0, f 1 0 x ≤ 1 / Real.exp 1) ∧
    (∀ c > 0, c ≠ 1 → has_zero_point (g c) → c ≤ Real.exp (1 / Real.exp 1)) :=
sorry

end problem_solution_l2014_201441


namespace turtle_arrival_time_l2014_201418

/-- Represents the speeds and distances of the animals -/
structure AnimalData where
  turtle_speed : ℝ
  lion1_speed : ℝ
  lion2_speed : ℝ
  turtle_distance : ℝ
  lion1_distance : ℝ

/-- Represents the time intervals between events -/
structure TimeIntervals where
  between_encounters : ℝ
  after_second_encounter : ℝ

/-- The main theorem stating the time for the turtle to reach the watering hole -/
theorem turtle_arrival_time 
  (data : AnimalData) 
  (time : TimeIntervals) 
  (h1 : data.lion1_distance = 6 * data.lion1_speed)
  (h2 : data.lion2_speed = 1.5 * data.lion1_speed)
  (h3 : data.turtle_distance = 32 * data.turtle_speed)
  (h4 : time.between_encounters = 2.4)
  (h5 : (data.lion1_distance - data.turtle_distance) / (data.lion1_speed - data.turtle_speed) + 
        time.between_encounters = 
        data.turtle_distance / (data.turtle_speed + data.lion2_speed))
  : time.after_second_encounter = 28.8 := by
  sorry

end turtle_arrival_time_l2014_201418


namespace inequality_solution_set_l2014_201456

theorem inequality_solution_set (t : ℝ) (h : 0 < t ∧ t < 1) :
  {x : ℝ | (t - x) * (x - 1/t) > 0} = {x : ℝ | t < x ∧ x < 1/t} := by
  sorry

end inequality_solution_set_l2014_201456


namespace number_equation_l2014_201407

theorem number_equation (x : ℝ) : (1/4 : ℝ) * x + 15 = 27 ↔ x = 48 := by
  sorry

end number_equation_l2014_201407


namespace alice_rearrangement_time_l2014_201437

/-- The time in hours required to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  (Nat.factorial name_length : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Theorem: Given a name with 5 unique letters and the ability to write 12 rearrangements per minute,
    it takes 1/6 hours to write all possible rearrangements -/
theorem alice_rearrangement_time :
  time_to_write_rearrangements 5 12 = 1/6 := by
  sorry


end alice_rearrangement_time_l2014_201437


namespace consecutive_product_prime_power_and_perfect_power_l2014_201405

theorem consecutive_product_prime_power_and_perfect_power (m : ℕ) : m ≥ 1 → (
  (∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m * (m + 1) = p ^ k) ↔ m = 1
) ∧ (
  ¬∃ (a k : ℕ), a ≥ 1 ∧ k ≥ 2 ∧ m * (m + 1) = a ^ k
) := by sorry

end consecutive_product_prime_power_and_perfect_power_l2014_201405


namespace train_meeting_point_l2014_201435

theorem train_meeting_point 
  (route_length : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (h1 : route_length = 75)
  (h2 : speed_A = 25)
  (h3 : speed_B = 37.5)
  : (route_length * speed_A) / (speed_A + speed_B) = 30 := by
  sorry

end train_meeting_point_l2014_201435


namespace probability_one_girl_in_pair_l2014_201410

theorem probability_one_girl_in_pair (n_boys n_girls : ℕ) (h_boys : n_boys = 4) (h_girls : n_girls = 2) :
  let total := n_boys + n_girls
  let total_pairs := total.choose 2
  let favorable_outcomes := n_boys * n_girls
  (favorable_outcomes : ℚ) / total_pairs = 8 / 15 := by
  sorry

end probability_one_girl_in_pair_l2014_201410


namespace system_a_l2014_201411

theorem system_a (x y : ℝ) : 
  y^4 + x*y^2 - 2*x^2 = 0 ∧ x + y = 6 →
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = -3) :=
sorry

end system_a_l2014_201411


namespace prime_congruent_three_mod_four_divides_x_l2014_201417

theorem prime_congruent_three_mod_four_divides_x (p : ℕ) (x₀ y₀ : ℕ) :
  Prime p →
  p % 4 = 3 →
  x₀ > 0 →
  y₀ > 0 →
  (p + 2) * x₀^2 - (p + 1) * y₀^2 + p * x₀ + (p + 2) * y₀ = 1 →
  p ∣ x₀ := by
  sorry

end prime_congruent_three_mod_four_divides_x_l2014_201417


namespace second_project_length_l2014_201443

/-- Represents a digging project -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth dug in a project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject := ⟨100, 25, 30⟩

/-- The second digging project with unknown length -/
def project2 (l : ℝ) : DiggingProject := ⟨75, l, 50⟩

/-- The theorem stating that the length of the second project is 20 meters -/
theorem second_project_length :
  ∃ l : ℝ, volume project1 = volume (project2 l) ∧ l = 20 := by
  sorry


end second_project_length_l2014_201443


namespace negative_inequality_l2014_201469

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end negative_inequality_l2014_201469


namespace red_tile_cost_courtyard_red_tile_cost_l2014_201433

/-- Calculates the cost of each red tile in a courtyard tiling project. -/
theorem red_tile_cost (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (tiles_per_sqft : ℝ) (green_tile_percentage : ℝ) (green_tile_cost : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_area := courtyard_length * courtyard_width
  let total_tiles := total_area * tiles_per_sqft
  let green_tiles := green_tile_percentage * total_tiles
  let red_tiles := total_tiles - green_tiles
  let green_cost := green_tiles * green_tile_cost
  let red_cost := total_cost - green_cost
  red_cost / red_tiles

/-- The cost of each red tile in the given courtyard tiling project is $1.50. -/
theorem courtyard_red_tile_cost : 
  red_tile_cost 25 10 4 0.4 3 2100 = 1.5 := by
  sorry

end red_tile_cost_courtyard_red_tile_cost_l2014_201433


namespace eight_number_sequence_proof_l2014_201401

theorem eight_number_sequence_proof :
  ∀ (a : Fin 8 → ℕ),
  (a 0 = 20) →
  (a 7 = 16) →
  (∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 100) →
  (∀ i : Fin 8, a i = [20, 16, 64, 20, 16, 64, 20, 16].get i) :=
by
  sorry

end eight_number_sequence_proof_l2014_201401


namespace joes_fast_food_cost_purchase_cost_l2014_201490

/-- The cost of purchasing sandwiches and sodas at Joe's Fast Food -/
theorem joes_fast_food_cost : ℕ → ℕ → ℕ
  | sandwich_count, soda_count => 
    4 * sandwich_count + 3 * soda_count

/-- Proof that purchasing 7 sandwiches and 9 sodas costs $55 -/
theorem purchase_cost : joes_fast_food_cost 7 9 = 55 := by
  sorry

end joes_fast_food_cost_purchase_cost_l2014_201490


namespace rectangle_area_l2014_201464

/-- The area of a rectangle with width 2 feet and length 5 feet is 10 square feet. -/
theorem rectangle_area : 
  let width : ℝ := 2
  let length : ℝ := 5
  width * length = 10 := by sorry

end rectangle_area_l2014_201464


namespace equation_solution_l2014_201494

theorem equation_solution (x : ℝ) :
  x ≠ 5 ∧ x ≠ 6 →
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ x = 1 ∨ x = 2 ∨ x = 3 := by
sorry

end equation_solution_l2014_201494


namespace division_problem_l2014_201454

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12) 
  (h2 : (x : ℝ) % (y : ℝ) = 5.76) : 
  y = 48 := by
sorry

end division_problem_l2014_201454


namespace multiples_of_four_between_100_and_300_l2014_201499

theorem multiples_of_four_between_100_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 100 ∧ n < 300) (Finset.range 300)).card = 49 := by
  sorry

end multiples_of_four_between_100_and_300_l2014_201499


namespace circle_equation_proof_l2014_201468

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The equation of a circle. -/
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 
    (c.passesThrough.1 - c.center.1)^2 + (c.passesThrough.2 - c.center.2)^2

/-- The specific circle from the problem. -/
def C : Circle :=
  { center := (2, -3)
  , passesThrough := (0, 0) }

theorem circle_equation_proof :
  ∀ x y : ℝ, circleEquation C x y ↔ (x - 2)^2 + (y + 3)^2 = 13 := by
  sorry

end circle_equation_proof_l2014_201468


namespace division_problem_l2014_201475

theorem division_problem (x : ℝ) : 100 / x = 400 → x = 0.25 := by
  sorry

end division_problem_l2014_201475


namespace inverse_sum_lower_bound_l2014_201403

theorem inverse_sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hab_sum : a + b = 1) :
  1 / a + 1 / b > 4 := by
sorry

end inverse_sum_lower_bound_l2014_201403


namespace john_remaining_money_l2014_201438

def calculate_remaining_money (base_income : ℝ) (bonus_rate : ℝ) (transport_rate : ℝ)
  (rent : ℝ) (utilities : ℝ) (food : ℝ) (misc_rate : ℝ) (emergency_rate : ℝ)
  (retirement_rate : ℝ) (medical_expense : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_income := base_income * (1 + bonus_rate)
  let after_tax_income := total_income - (base_income * tax_rate)
  let fixed_expenses := rent + utilities + food
  let variable_expenses := (total_income * transport_rate) + (total_income * misc_rate)
  let savings_and_investments := (total_income * emergency_rate) + (total_income * retirement_rate)
  let total_expenses := fixed_expenses + variable_expenses + medical_expense + savings_and_investments
  after_tax_income - total_expenses

theorem john_remaining_money :
  calculate_remaining_money 2000 0.15 0.05 500 100 300 0.10 0.07 0.05 250 0.15 = 229 := by
  sorry

end john_remaining_money_l2014_201438


namespace max_value_expression_l2014_201450

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({0, 1, 2, 3} : Set ℕ) →
  b ∈ ({0, 1, 2, 3} : Set ℕ) →
  c ∈ ({0, 1, 2, 3} : Set ℕ) →
  d ∈ ({0, 1, 2, 3} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  d ≠ 0 →
  c * a^b - d ≤ 2 :=
by sorry

end max_value_expression_l2014_201450


namespace unused_sector_angle_l2014_201402

/-- Given a circular piece of paper with radius 20 cm, from which a sector is removed
    to form a cone with radius 15 cm and volume 900π cubic cm,
    prove that the measure of the angle of the unused sector is 90°. -/
theorem unused_sector_angle (r_paper : ℝ) (r_cone : ℝ) (v_cone : ℝ) :
  r_paper = 20 →
  r_cone = 15 →
  v_cone = 900 * Real.pi →
  ∃ (h : ℝ) (s : ℝ),
    v_cone = (1/3) * Real.pi * r_cone^2 * h ∧
    s^2 = r_cone^2 + h^2 ∧
    s ≤ r_paper ∧
    (2 * Real.pi * r_cone) / (2 * Real.pi * r_paper) * 360 = 270 :=
by sorry

end unused_sector_angle_l2014_201402


namespace time_between_ticks_at_6_l2014_201486

/-- The number of ticks at 6 o'clock -/
def ticks_at_6 : ℕ := 6

/-- The number of ticks at 8 o'clock -/
def ticks_at_8 : ℕ := 8

/-- The time between the first and last ticks at 8 o'clock in seconds -/
def time_at_8 : ℕ := 42

/-- The theorem stating the time between the first and last ticks at 6 o'clock -/
theorem time_between_ticks_at_6 : ℕ := by
  -- Assume the time between each tick is constant for any hour
  -- Calculate the time between ticks at 6 o'clock
  sorry

end time_between_ticks_at_6_l2014_201486


namespace jumping_contest_l2014_201432

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_jump = 15)
  (h3 : mouse_jump + 44 = frog_jump) :
  grasshopper_jump - frog_jump = 4 := by
  sorry


end jumping_contest_l2014_201432


namespace car_distance_proof_l2014_201476

theorem car_distance_proof (D : ℝ) : 
  (D / 60 = D / 90 + 1/2) → D = 90 := by
  sorry

end car_distance_proof_l2014_201476


namespace ricardo_coin_value_difference_l2014_201478

/-- The total number of coins Ricardo has -/
def total_coins : ℕ := 3030

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculate the total value in cents given the number of pennies -/
def total_value (num_pennies : ℕ) : ℕ :=
  num_pennies * penny_value + (total_coins - num_pennies) * nickel_value

/-- The minimum number of pennies Ricardo can have -/
def min_pennies : ℕ := 1

/-- The maximum number of pennies Ricardo can have -/
def max_pennies : ℕ := total_coins - 1

theorem ricardo_coin_value_difference :
  (total_value min_pennies) - (total_value max_pennies) = 12112 := by
  sorry

end ricardo_coin_value_difference_l2014_201478


namespace unique_function_satisfying_conditions_l2014_201445

-- Define the property of having at most finitely many zeros
def HasFinitelyManyZeros (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x, f x = 0 → x ∈ S

-- Define the functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x^4 + y) = x^3 * f x + f (f y)

-- Theorem statement
theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, HasFinitelyManyZeros f ∧ SatisfiesFunctionalEquation f ∧ (∀ x, f x = x) :=
sorry

end unique_function_satisfying_conditions_l2014_201445


namespace daughter_and_child_weight_l2014_201497

/-- The combined weight of a daughter and her daughter (child) given specific family weight conditions -/
theorem daughter_and_child_weight (total_weight mother_weight daughter_weight child_weight : ℝ) :
  total_weight = mother_weight + daughter_weight + child_weight →
  child_weight = (1 / 5) * mother_weight →
  daughter_weight = 46 →
  total_weight = 130 →
  daughter_weight + child_weight = 60 := by
  sorry

end daughter_and_child_weight_l2014_201497


namespace television_price_proof_l2014_201492

theorem television_price_proof (discount_rate : ℝ) (final_price : ℝ) (num_tvs : ℕ) :
  discount_rate = 0.25 →
  final_price = 975 →
  num_tvs = 2 →
  ∃ (original_price : ℝ),
    original_price = 650 ∧
    final_price = (1 - discount_rate) * (num_tvs * original_price) :=
by sorry

end television_price_proof_l2014_201492


namespace min_value_geometric_sequence_l2014_201474

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 6a₂ + 7a₃ is -9/7 -/
theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) : 
  a₁ = 1 → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) → 
  (∀ x y : ℝ, x = a₂ ∧ y = a₃ → 6*x + 7*y ≥ -9/7) ∧ 
  (∃ x y : ℝ, x = a₂ ∧ y = a₃ ∧ 6*x + 7*y = -9/7) :=
by sorry

end min_value_geometric_sequence_l2014_201474


namespace nine_trailing_zeros_l2014_201436

def binary_trailing_zeros (n : ℕ) : ℕ :=
  (n.digits 2).reverse.takeWhile (· = 0) |>.length

theorem nine_trailing_zeros (n : ℕ) : binary_trailing_zeros (n * 1024 + 4 * 64 + 2) = 9 := by
  sorry

end nine_trailing_zeros_l2014_201436


namespace right_triangle_condition_l2014_201442

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) + a*cos(C) = b + c, then the triangle is right-angled. -/
theorem right_triangle_condition (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B + a * Real.cos C = b + c →
  a^2 = b^2 + c^2 := by
sorry

end right_triangle_condition_l2014_201442


namespace y_intercept_of_line_l2014_201419

/-- The y-intercept of the line 4x + 7y = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → x = 0 → y = 4 := by
  sorry

end y_intercept_of_line_l2014_201419


namespace ellipse_and_fixed_point_l2014_201482

/-- Ellipse C₁ -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Parabola C₂ -/
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- Tangent line to parabola -/
def tangent_line (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

/-- Circle with diameter AB passing through T -/
def circle_passes_through (A B T : ℝ × ℝ) : Prop :=
  (T.1 - A.1) * (T.1 - B.1) + (T.2 - A.2) * (T.2 - B.2) = 0

theorem ellipse_and_fixed_point 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a^2 - b^2 = a^2 / 2) -- Eccentricity condition
  (h4 : ∃ (x y : ℝ), tangent_line 1 x y ∧ parabola x y) :
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (A B : ℝ × ℝ), 
    ellipse a b A.1 A.2 → 
    ellipse a b B.1 B.2 → 
    (∃ (k : ℝ), A.2 = k * A.1 - 1/3 ∧ B.2 = k * B.1 - 1/3) →
    circle_passes_through A B (0, 1)) :=
sorry

end ellipse_and_fixed_point_l2014_201482


namespace bead_arrangement_theorem_l2014_201498

/-- Represents a bead with a color --/
structure Bead where
  color : Nat

/-- Represents a necklace of beads --/
def Necklace := List Bead

/-- Checks if a segment of beads contains at least k different colors --/
def hasAtLeastKColors (segment : List Bead) (k : Nat) : Prop :=
  (segment.map (·.color)).toFinset.card ≥ k

/-- The property we want to prove --/
theorem bead_arrangement_theorem (total_beads : Nat) (num_colors : Nat) (beads_per_color : Nat)
    (h1 : total_beads = 1000)
    (h2 : num_colors = 50)
    (h3 : beads_per_color = 20)
    (h4 : total_beads = num_colors * beads_per_color) :
    ∃ (n : Nat),
      (∀ (necklace : Necklace),
        necklace.length = total_beads →
        (∀ (i : Nat),
          i + n ≤ necklace.length →
          hasAtLeastKColors (necklace.take n) 25)) ∧
      (∀ (m : Nat),
        m < n →
        ∃ (necklace : Necklace),
          necklace.length = total_beads ∧
          ∃ (i : Nat),
            i + m ≤ necklace.length ∧
            ¬hasAtLeastKColors (necklace.take m) 25) :=
  sorry

#check bead_arrangement_theorem

end bead_arrangement_theorem_l2014_201498


namespace problem_1_l2014_201412

theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) :
  deriv f 1 = 1 := by sorry

end problem_1_l2014_201412


namespace fabric_length_l2014_201422

/-- Given a rectangular piece of fabric with width 3 cm and area 24 cm², prove its length is 8 cm. -/
theorem fabric_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → area = 24 → area = length * width → length = 8 := by
sorry

end fabric_length_l2014_201422


namespace probability_multiple_of_three_l2014_201425

/-- A fair cubic die with 6 faces numbered 1 to 6 -/
def FairDie : Finset ℕ := Finset.range 6

/-- The set of outcomes that are multiples of 3 -/
def MultiplesOfThree : Finset ℕ := Finset.filter (fun n => n % 3 = 0) FairDie

/-- The probability of an event in a finite sample space -/
def probability (event : Finset ℕ) (sampleSpace : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_multiple_of_three : 
  probability MultiplesOfThree FairDie = 1 / 3 := by sorry

end probability_multiple_of_three_l2014_201425


namespace basketball_wins_l2014_201453

/-- The total number of wins for a basketball team over four competitions -/
def total_wins (first_wins : ℕ) : ℕ :=
  let second_wins := (first_wins * 5) / 8
  let third_wins := first_wins + second_wins
  let fourth_wins := ((first_wins + second_wins + third_wins) * 3) / 5
  first_wins + second_wins + third_wins + fourth_wins

/-- Theorem stating that given 40 wins in the first competition, the total wins over four competitions is 208 -/
theorem basketball_wins : total_wins 40 = 208 := by
  sorry

end basketball_wins_l2014_201453


namespace race_speed_ratio_l2014_201495

theorem race_speed_ratio (L : ℝ) (h_L : L > 0) : 
  let head_start := 0.35 * L
  let winning_distance := 0.25 * L
  let a_distance := L + head_start
  let b_distance := L + winning_distance
  ∃ R : ℝ, R * (L / b_distance) = a_distance / b_distance ∧ R = 1.08 :=
by sorry

end race_speed_ratio_l2014_201495


namespace parabola_point_M_x_coordinate_l2014_201400

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define line l passing through F and intersecting the parabola at A and B
def line_l (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  ∃ k : ℝ, A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1)

-- Define point M as the midpoint of A and B
def point_M (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define point P on the parabola
def point_P (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define the distance between P and F is 2
def PF_distance (P : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 4

theorem parabola_point_M_x_coordinate 
  (A B M P : ℝ × ℝ) 
  (h1 : line_l A B) 
  (h2 : point_M A B M) 
  (h3 : point_P P) 
  (h4 : PF_distance P) :
  M.1 = 3 := by sorry

end parabola_point_M_x_coordinate_l2014_201400


namespace domain_of_composite_function_l2014_201477

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-2) 3

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc (-1/2) 2 := by sorry

end domain_of_composite_function_l2014_201477


namespace unique_friend_groups_l2014_201488

theorem unique_friend_groups (n : ℕ) (h : n = 10) : 
  Finset.card (Finset.powerset (Finset.range n)) = 2^n := by
  sorry

end unique_friend_groups_l2014_201488


namespace cube_edge_length_l2014_201444

-- Define the cube
structure Cube where
  edge_length : ℝ
  sum_of_edges : ℝ

-- State the theorem
theorem cube_edge_length (c : Cube) (h : c.sum_of_edges = 108) : c.edge_length = 9 := by
  sorry

end cube_edge_length_l2014_201444


namespace clock_angle_at_9am_l2014_201415

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees each hour represents -/
def degrees_per_hour : ℕ := 30

/-- The position of the minute hand at 9:00 a.m. in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 9:00 a.m. in degrees -/
def hour_hand_position : ℕ := 270

/-- The smaller angle between the minute hand and hour hand at 9:00 a.m. -/
def smaller_angle : ℕ := 90

/-- Theorem stating that the smaller angle between the minute hand and hour hand at 9:00 a.m. is 90 degrees -/
theorem clock_angle_at_9am :
  smaller_angle = min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position)) :=
by sorry

end clock_angle_at_9am_l2014_201415


namespace sum_of_coefficients_equals_387420501_l2014_201408

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (3*x + 4*y)^9 + (2*x - 5*y)^9

-- Define the sum of coefficients function
def sum_of_coefficients (p : ℤ → ℤ → ℤ) (x y : ℤ) : ℤ := p x y

-- Theorem statement
theorem sum_of_coefficients_equals_387420501 :
  sum_of_coefficients polynomial 2 (-1) = 387420501 := by
  sorry

end sum_of_coefficients_equals_387420501_l2014_201408


namespace problem_solution_l2014_201455

theorem problem_solution (x y a : ℝ) 
  (h1 : |x + 1| + (y + 2)^2 = 0)
  (h2 : a * x - 3 * a * y = 1) : 
  a = 0.2 := by
sorry

end problem_solution_l2014_201455


namespace wall_width_proof_l2014_201479

/-- Proves that the width of a wall is 2 meters given specific brick and wall dimensions --/
theorem wall_width_proof (brick_length : Real) (brick_width : Real) (brick_height : Real)
  (wall_length : Real) (wall_height : Real) (num_bricks : Nat) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 27 →
  wall_height = 0.75 →
  num_bricks = 27000 →
  ∃ (wall_width : Real), wall_width = 2 ∧
    brick_length * brick_width * brick_height * num_bricks =
    wall_length * wall_width * wall_height := by
  sorry

end wall_width_proof_l2014_201479


namespace summer_break_length_l2014_201487

/-- Represents the summer break reading scenario --/
structure SummerReading where
  deshaun_books : ℕ
  avg_pages_per_book : ℕ
  second_person_percentage : ℚ
  second_person_daily_pages : ℕ

/-- Calculates the number of days in the summer break --/
def summer_break_days (sr : SummerReading) : ℚ :=
  (sr.deshaun_books * sr.avg_pages_per_book * sr.second_person_percentage) / sr.second_person_daily_pages

/-- Theorem stating that the summer break is 80 days long --/
theorem summer_break_length (sr : SummerReading) 
  (h1 : sr.deshaun_books = 60)
  (h2 : sr.avg_pages_per_book = 320)
  (h3 : sr.second_person_percentage = 3/4)
  (h4 : sr.second_person_daily_pages = 180) :
  summer_break_days sr = 80 := by
  sorry

#eval summer_break_days { 
  deshaun_books := 60, 
  avg_pages_per_book := 320, 
  second_person_percentage := 3/4, 
  second_person_daily_pages := 180 
}

end summer_break_length_l2014_201487


namespace sum_of_k_values_l2014_201429

theorem sum_of_k_values : ∃ (S : Finset ℤ), 
  (∀ k ∈ S, ∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 9 = 0 ∧ 3 * y^2 - k * y + 9 = 0) ∧
  (∀ k : ℤ, (∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 9 = 0 ∧ 3 * y^2 - k * y + 9 = 0) → k ∈ S) ∧
  (S.sum id = 0) :=
sorry

end sum_of_k_values_l2014_201429


namespace probability_both_above_400_l2014_201481

def total_students : ℕ := 600
def male_students : ℕ := 220
def female_students : ℕ := 380
def selected_students : ℕ := 10
def selected_females : ℕ := 6
def females_above_400 : ℕ := 3
def discussion_group_size : ℕ := 2

theorem probability_both_above_400 :
  (female_students = total_students - male_students) →
  (selected_females ≤ selected_students) →
  (females_above_400 ≤ selected_females) →
  (discussion_group_size ≤ selected_females) →
  (Nat.choose females_above_400 discussion_group_size) / (Nat.choose selected_females discussion_group_size) = 1 / 5 := by
  sorry

end probability_both_above_400_l2014_201481


namespace race_finish_count_l2014_201463

/-- Calculates the number of men who finished a race given specific conditions --/
def men_finished_race (total_men : ℕ) : ℕ :=
  let tripped := total_men / 4
  let tripped_finished := tripped / 3
  let remaining_after_trip := total_men - tripped
  let dehydrated := remaining_after_trip * 2 / 3
  let dehydrated_finished := dehydrated * 4 / 5
  let remaining_after_dehydration := remaining_after_trip - dehydrated
  let lost := remaining_after_dehydration * 12 / 100
  let lost_finished := lost / 2
  let remaining_after_lost := remaining_after_dehydration - lost
  let faced_obstacle := remaining_after_lost * 3 / 8
  let obstacle_finished := faced_obstacle * 2 / 5
  tripped_finished + dehydrated_finished + lost_finished + obstacle_finished

/-- Theorem stating that given 80 men in the race, 41 men finished --/
theorem race_finish_count : men_finished_race 80 = 41 := by
  sorry

#eval men_finished_race 80

end race_finish_count_l2014_201463


namespace equal_color_squares_count_l2014_201406

/-- Represents a 5x5 grid with some cells painted black -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Counts the number of squares in the grid with equal black and white cells -/
def countEqualColorSquares (g : Grid) : ℕ :=
  let count2x2 := (5 - 2 + 1)^2 - 2  -- Total 2x2 squares minus those containing the center
  let count4x4 := 2  -- Lower two 4x4 squares meet the criterion
  count2x2 + count4x4

/-- Theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_color_squares_count (g : Grid) : countEqualColorSquares g = 16 := by
  sorry

end equal_color_squares_count_l2014_201406


namespace function_composition_equality_l2014_201428

theorem function_composition_equality (a b : ℝ) :
  (∀ x, (3 * ((a * x + b) : ℝ) - 4 = 4 * x + 5)) →
  a + b = 13 / 3 := by
  sorry

end function_composition_equality_l2014_201428


namespace polynomial_identity_l2014_201448

theorem polynomial_identity (x : ℝ) : 
  (x + 1)^4 + 4*(x + 1)^3 + 6*(x + 1)^2 + 4*(x + 1) + 1 = (x + 2)^4 := by
  sorry

end polynomial_identity_l2014_201448


namespace base8_243_equals_base10_163_l2014_201440

def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base8_243_equals_base10_163 :
  base8_to_base10 [3, 4, 2] = 163 := by
  sorry

end base8_243_equals_base10_163_l2014_201440


namespace nail_container_problem_l2014_201431

theorem nail_container_problem (N : ℝ) : 
  (N > 0) →
  (0.7 * N - 0.7 * (0.7 * N) = 84) →
  N = 400 := by
sorry

end nail_container_problem_l2014_201431


namespace simplify_fraction_1_simplify_fraction_2_l2014_201471

-- Problem 1
theorem simplify_fraction_1 (x y : ℝ) (h : y ≠ 0) :
  (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) := by sorry

-- Problem 2
theorem simplify_fraction_2 (m n : ℝ) (h1 : m ≠ n) (h2 : m ≠ -n) :
  m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 := by sorry

end simplify_fraction_1_simplify_fraction_2_l2014_201471


namespace initial_girls_count_l2014_201421

theorem initial_girls_count (initial_total : ℕ) (initial_girls : ℕ) : 
  initial_girls = 12 ∧ initial_total = 24 :=
  by
  have h1 : initial_girls = initial_total / 2 := by sorry
  have h2 : (initial_girls - 2) * 100 = 40 * (initial_total + 1) := by sorry
  have h3 : initial_girls * 100 = 45 * (initial_total - 1) := by sorry
  sorry

#check initial_girls_count

end initial_girls_count_l2014_201421


namespace f_positive_iff_m_range_f_root_in_zero_one_iff_m_range_l2014_201470

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem f_positive_iff_m_range (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ -2*Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2*Real.sqrt 6 + 5 := by
  sorry

theorem f_root_in_zero_one_iff_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f m x = 0) ↔ m ∈ Set.Ioo (-2) 0 := by
  sorry

end f_positive_iff_m_range_f_root_in_zero_one_iff_m_range_l2014_201470


namespace no_four_integers_product_plus_2002_square_l2014_201458

theorem no_four_integers_product_plus_2002_square : 
  ¬ ∃ (n₁ n₂ n₃ n₄ : ℕ+), 
    (∀ (i j : Fin 4), i ≠ j → ∃ (m : ℕ), (n₁ :: n₂ :: n₃ :: n₄ :: []).get i * (n₁ :: n₂ :: n₃ :: n₄ :: []).get j + 2002 = m^2) :=
by sorry

end no_four_integers_product_plus_2002_square_l2014_201458


namespace sum_of_x_and_y_for_given_equation_l2014_201460

theorem sum_of_x_and_y_for_given_equation (x y : ℝ) : 
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0 → x + y = -9/2 := by
  sorry

end sum_of_x_and_y_for_given_equation_l2014_201460


namespace part1_part2_l2014_201491

-- Part 1
theorem part1 (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x = Real.exp x + Real.sin x + b) →
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) →
  b ≥ -1 := by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (b m : ℝ) :
  (∀ x : ℝ, f x = Real.exp x + b) →
  (f 0 = 1 ∧ (deriv f) 0 = 1) →
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = (m - 2*x₁) / x₁ ∧ f x₂ = (m - 2*x₂) / x₂) →
  -1 / Real.exp 1 < m ∧ m < 0 := by sorry

end part1_part2_l2014_201491


namespace cabinet_area_l2014_201404

theorem cabinet_area : 
  ∀ (width length area : ℝ),
  width = 1.2 →
  length = 1.8 →
  area = width * length →
  area = 2.16 := by
sorry

end cabinet_area_l2014_201404


namespace fraction_equals_decimal_l2014_201409

theorem fraction_equals_decimal : (1 : ℚ) / 4 = 0.25 := by sorry

end fraction_equals_decimal_l2014_201409


namespace cars_time_passage_l2014_201447

/-- Given that a car comes down the road every 20 minutes, 
    prove that the time passed for 30 cars is 10 hours. -/
theorem cars_time_passage (interval : ℕ) (num_cars : ℕ) (hours_per_day : ℕ) :
  interval = 20 →
  num_cars = 30 →
  hours_per_day = 24 →
  (interval * num_cars) / 60 = 10 := by
  sorry

end cars_time_passage_l2014_201447


namespace tan_negative_fifty_five_sixths_pi_l2014_201480

theorem tan_negative_fifty_five_sixths_pi : 
  Real.tan (-55 / 6 * Real.pi) = -Real.sqrt 3 / 3 := by
sorry

end tan_negative_fifty_five_sixths_pi_l2014_201480


namespace smallest_equal_distribution_l2014_201465

def apple_per_box : ℕ := 18
def grapes_per_container : ℕ := 9
def orange_per_container : ℕ := 12
def cherries_per_bag : ℕ := 6

theorem smallest_equal_distribution (n : ℕ) :
  (n % apple_per_box = 0) ∧
  (n % grapes_per_container = 0) ∧
  (n % orange_per_container = 0) ∧
  (n % cherries_per_bag = 0) ∧
  (∀ m : ℕ, m < n →
    ¬((m % apple_per_box = 0) ∧
      (m % grapes_per_container = 0) ∧
      (m % orange_per_container = 0) ∧
      (m % cherries_per_bag = 0))) →
  n = 36 := by
sorry

end smallest_equal_distribution_l2014_201465


namespace comparison_of_rational_numbers_l2014_201414

theorem comparison_of_rational_numbers :
  (- (- (1 / 5 : ℚ)) > - (1 / 5 : ℚ)) ∧
  (- (- (17 / 5 : ℚ)) > - (17 / 5 : ℚ)) ∧
  (- (4 : ℚ) < (4 : ℚ)) ∧
  ((- (11 / 10 : ℚ)) < 0) :=
by sorry

end comparison_of_rational_numbers_l2014_201414


namespace max_trees_cut_2001_l2014_201420

/-- Represents a square grid of trees -/
structure TreeGrid where
  size : Nat
  is_square : size * size = size * size

/-- Represents the maximum number of trees that can be cut down -/
def max_trees_cut (grid : TreeGrid) : Nat :=
  (grid.size / 2) * (grid.size / 2) + 1

/-- The theorem to be proved -/
theorem max_trees_cut_2001 :
  ∀ (grid : TreeGrid),
    grid.size = 2001 →
    max_trees_cut grid = 1001001 := by
  sorry

end max_trees_cut_2001_l2014_201420


namespace triangle_inequality_l2014_201451

/-- For any triangle ABC with semiperimeter p and inradius r, 
    the sum of the reciprocals of the square roots of twice the sines of its angles 
    is less than or equal to the square root of the ratio of its semiperimeter to its inradius. -/
theorem triangle_inequality (A B C : Real) (p r : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 0 < p ∧ 0 < r → 
  1 / Real.sqrt (2 * Real.sin A) + 1 / Real.sqrt (2 * Real.sin B) + 1 / Real.sqrt (2 * Real.sin C) ≤ Real.sqrt (p / r) := by
  sorry


end triangle_inequality_l2014_201451


namespace registration_methods_count_l2014_201466

/-- Represents the number of courses -/
def num_courses : ℕ := 3

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- 
Calculates the number of ways to distribute n distinct objects into k distinct boxes,
where each box must contain at least one object.
-/
def distribution_count (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the number of registration methods is 150 -/
theorem registration_methods_count : distribution_count num_students num_courses = 150 :=
  sorry

end registration_methods_count_l2014_201466


namespace last_mile_speed_l2014_201452

/-- Represents the problem of calculating the required speed for the last mile of a journey --/
theorem last_mile_speed (total_distance : ℝ) (normal_speed : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (last_part_distance : ℝ) : 
  total_distance = 3 →
  normal_speed = 10 →
  first_part_distance = 2 →
  first_part_speed = 5 →
  last_part_distance = 1 →
  (total_distance / normal_speed = first_part_distance / first_part_speed + last_part_distance / 10) := by
  sorry

end last_mile_speed_l2014_201452


namespace triangle_zero_sum_implies_zero_function_l2014_201427

/-- A function f: ℝ² → ℝ with the property that the sum of its values
    at the vertices of any equilateral triangle with side length 1 is zero. -/
def TriangleZeroSum (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ A B C : ℝ × ℝ, 
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    f A + f B + f C = 0

/-- Theorem stating that any function with the TriangleZeroSum property
    is identically zero everywhere. -/
theorem triangle_zero_sum_implies_zero_function 
  (f : ℝ × ℝ → ℝ) (h : TriangleZeroSum f) : 
  ∀ x : ℝ × ℝ, f x = 0 := by
  sorry

end triangle_zero_sum_implies_zero_function_l2014_201427


namespace consecutive_composites_under_40_l2014_201430

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem consecutive_composites_under_40 :
  ∃ (a : ℕ),
    (∀ i : Fin 6, isTwoDigit (a + i) ∧ a + i < 40) ∧
    (∀ i : Fin 6, ¬ isPrime (a + i)) ∧
    (∀ n : ℕ, n > a + 5 →
      ¬(∀ i : Fin 6, isTwoDigit (n - i) ∧ n - i < 40 ∧ ¬ isPrime (n - i))) :=
by sorry

end consecutive_composites_under_40_l2014_201430


namespace restaurant_bill_proof_l2014_201413

/-- Calculates the total bill given the number of people and the amount each person paid -/
def totalBill (numPeople : ℕ) (amountPerPerson : ℕ) : ℕ :=
  numPeople * amountPerPerson

/-- Proves that if three people divide a bill evenly and each pays $45, then the total bill is $135 -/
theorem restaurant_bill_proof :
  totalBill 3 45 = 135 := by
  sorry

end restaurant_bill_proof_l2014_201413


namespace function_characterization_l2014_201449

def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

theorem function_characterization (f : ℕ → ℕ) : 
  (∀ a b : ℕ, a > 0 → b > 0 → 
    (iterate f a b + iterate f b a) ∣ (2 * (f (a * b) + b^2 - 1))) → 
  ((∀ x : ℕ, f x = x + 1) ∨ (f 1 ∣ 4)) :=
sorry

end function_characterization_l2014_201449


namespace doubling_base_theorem_l2014_201424

theorem doubling_base_theorem (a b x : ℝ) (h1 : b ≠ 0) :
  (2 * a) ^ b = a ^ b * x ^ b → x = 2 := by
  sorry

end doubling_base_theorem_l2014_201424


namespace solve_equation_l2014_201484

theorem solve_equation : ∃ x : ℝ, x + 2*x = 400 - (3*x + 4*x) ∧ x = 40 := by
  sorry

end solve_equation_l2014_201484


namespace adult_ticket_cost_l2014_201489

theorem adult_ticket_cost (num_adults num_children : ℕ) (total_bill child_ticket_cost : ℚ) :
  num_adults = 10 →
  num_children = 11 →
  total_bill = 124 →
  child_ticket_cost = 4 →
  (total_bill - num_children * child_ticket_cost) / num_adults = 8 :=
by sorry

end adult_ticket_cost_l2014_201489


namespace sum_357_eq_42_l2014_201467

/-- A geometric sequence with first term 3 and the sum of the first, third, and fifth terms equal to 21 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  first_term : a 1 = 3
  sum_135 : a 1 + a 3 + a 5 = 21

/-- The sum of the third, fifth, and seventh terms of the geometric sequence is 42 -/
theorem sum_357_eq_42 (seq : GeometricSequence) : seq.a 3 + seq.a 5 + seq.a 7 = 42 := by
  sorry

end sum_357_eq_42_l2014_201467


namespace equation_proof_l2014_201423

theorem equation_proof : 529 + 2 * 23 * 11 + 121 = 1156 := by
  sorry

end equation_proof_l2014_201423


namespace smallest_constant_D_l2014_201461

theorem smallest_constant_D :
  ∃ (D : ℝ), D = Real.sqrt (8 / 17) ∧
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D'*(2*x + 3*y) + 4) → D' ≥ D) :=
by sorry

end smallest_constant_D_l2014_201461


namespace rope_average_length_l2014_201462

/-- Given 6 ropes where one third have an average length of 70 cm and the rest have an average length of 85 cm, prove that the overall average length is 80 cm. -/
theorem rope_average_length : 
  let total_ropes : ℕ := 6
  let third_ropes : ℕ := total_ropes / 3
  let remaining_ropes : ℕ := total_ropes - third_ropes
  let third_avg_length : ℝ := 70
  let remaining_avg_length : ℝ := 85
  let total_length : ℝ := (third_ropes : ℝ) * third_avg_length + (remaining_ropes : ℝ) * remaining_avg_length
  let overall_avg_length : ℝ := total_length / (total_ropes : ℝ)
  overall_avg_length = 80 := by
sorry

end rope_average_length_l2014_201462


namespace r_value_when_n_is_3_l2014_201446

theorem r_value_when_n_is_3 : 
  ∀ (n s r : ℕ), 
    n = 3 → 
    s = 2^n + 2 → 
    r = 4^s + 3*s → 
    r = 1048606 := by
  sorry

end r_value_when_n_is_3_l2014_201446


namespace double_layer_cake_cost_double_layer_cake_cost_is_seven_l2014_201459

theorem double_layer_cake_cost (single_layer_cost : ℝ) 
                               (single_layer_quantity : ℕ) 
                               (double_layer_quantity : ℕ) 
                               (total_paid : ℝ) 
                               (change_received : ℝ) : ℝ :=
  let total_spent := total_paid - change_received
  let single_layer_total := single_layer_cost * single_layer_quantity
  let double_layer_total := total_spent - single_layer_total
  double_layer_total / double_layer_quantity

theorem double_layer_cake_cost_is_seven :
  double_layer_cake_cost 4 7 5 100 37 = 7 := by
  sorry

end double_layer_cake_cost_double_layer_cake_cost_is_seven_l2014_201459


namespace field_trip_van_occupancy_l2014_201472

theorem field_trip_van_occupancy (num_vans num_buses people_per_bus total_people : ℕ) 
  (h1 : num_vans = 9)
  (h2 : num_buses = 10)
  (h3 : people_per_bus = 27)
  (h4 : total_people = 342) :
  (total_people - num_buses * people_per_bus) / num_vans = 8 := by
  sorry

end field_trip_van_occupancy_l2014_201472


namespace father_son_work_time_work_completed_in_three_days_l2014_201493

/-- Given a task that takes 6 days for either a man or his son to complete alone,
    prove that they can complete it together in 3 days. -/
theorem father_son_work_time : ℝ → Prop :=
  fun total_work =>
    let man_rate := total_work / 6
    let son_rate := total_work / 6
    let combined_rate := man_rate + son_rate
    (total_work / combined_rate) = 3

/-- The main theorem stating that the work will be completed in 3 days -/
theorem work_completed_in_three_days (total_work : ℝ) (h : total_work > 0) :
  father_son_work_time total_work := by
  sorry

#check work_completed_in_three_days

end father_son_work_time_work_completed_in_three_days_l2014_201493


namespace square_difference_fourth_power_l2014_201426

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end square_difference_fourth_power_l2014_201426


namespace absolute_value_inequality_solution_set_l2014_201473

/-- The solution set of the inequality |x| + |x - 1| < 2 -/
theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x| + |x - 1| < 2} = Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) := by
  sorry

end absolute_value_inequality_solution_set_l2014_201473


namespace bucket_problem_l2014_201439

/-- Given two buckets A and B with unknown amounts of water, if transferring 6 liters from A to B
    results in A containing one-third of B's new amount, and transferring 6 liters from B to A
    results in B containing one-half of A's new amount, then A initially contains 13.2 liters of water. -/
theorem bucket_problem (A B : ℝ) 
    (h1 : A - 6 = (1/3) * (B + 6))
    (h2 : B - 6 = (1/2) * (A + 6)) :
    A = 13.2 := by
  sorry

end bucket_problem_l2014_201439


namespace sqrt_of_nine_l2014_201496

theorem sqrt_of_nine (x : ℝ) : x = Real.sqrt 9 → x = 3 := by
  sorry

end sqrt_of_nine_l2014_201496


namespace arcMTN_constant_l2014_201483

/-- Represents an equilateral triangle ABC with a circle rolling along side AB -/
structure RollingCircleTriangle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of the circle, equal to the triangle's altitude -/
  radius : ℝ
  /-- The circle's radius is equal to the triangle's altitude -/
  radius_eq_altitude : radius = side * Real.sqrt 3 / 2

/-- The measure of arc MTN in degrees -/
def arcMTN (rct : RollingCircleTriangle) : ℝ :=
  60

/-- Theorem stating that arc MTN always measures 60° -/
theorem arcMTN_constant (rct : RollingCircleTriangle) :
  arcMTN rct = 60 := by
  sorry

end arcMTN_constant_l2014_201483


namespace pavilion_pillar_height_l2014_201457

-- Define the regular octagon
structure RegularOctagon where
  side_length : ℝ
  center : ℝ × ℝ

-- Define a pillar
structure Pillar where
  base : ℝ × ℝ
  height : ℝ

-- Define the pavilion
structure Pavilion where
  octagon : RegularOctagon
  pillars : Fin 8 → Pillar

-- Define the theorem
theorem pavilion_pillar_height 
  (pav : Pavilion) 
  (h_a : (pav.pillars 0).height = 15)
  (h_b : (pav.pillars 1).height = 11)
  (h_c : (pav.pillars 2).height = 13) :
  (pav.pillars 5).height = 32 :=
sorry

end pavilion_pillar_height_l2014_201457
