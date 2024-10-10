import Mathlib

namespace inequality_proof_l3797_379773

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a * b + b * c + c * a = 1 / 3) : 
  1 / (a^2 - b*c + 1) + 1 / (b^2 - c*a + 1) + 1 / (c^2 - a*b + 1) ≤ 3 :=
by sorry

end inequality_proof_l3797_379773


namespace common_root_equations_unique_integer_solution_l3797_379793

theorem common_root_equations (x p : ℤ) : 
  (3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ (p = 3 ∧ x = 1) :=
by sorry

theorem unique_integer_solution : 
  ∃! p : ℤ, ∃ x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 :=
by sorry

end common_root_equations_unique_integer_solution_l3797_379793


namespace intersection_equality_condition_l3797_379778

theorem intersection_equality_condition (M N P : Set α) :
  (M = N → M ∩ P = N ∩ P) ∧
  ∃ M N P : Set ℕ, (M ∩ P = N ∩ P) ∧ (M ≠ N) := by
  sorry

end intersection_equality_condition_l3797_379778


namespace absolute_value_reciprocal_2023_l3797_379705

theorem absolute_value_reciprocal_2023 :
  {x : ℝ | |x| = (1 : ℝ) / 2023} = {-(1 : ℝ) / 2023, (1 : ℝ) / 2023} := by
  sorry

end absolute_value_reciprocal_2023_l3797_379705


namespace carl_personal_share_l3797_379788

/-- Carl's car accident costs and insurance coverage -/
structure AccidentCost where
  propertyDamage : ℝ
  medicalBills : ℝ
  insuranceCoverage : ℝ

/-- Calculate Carl's personal share of the accident costs -/
def calculatePersonalShare (cost : AccidentCost) : ℝ :=
  (cost.propertyDamage + cost.medicalBills) * (1 - cost.insuranceCoverage)

/-- Theorem stating that Carl's personal share is $22,000 -/
theorem carl_personal_share :
  let cost : AccidentCost := {
    propertyDamage := 40000,
    medicalBills := 70000,
    insuranceCoverage := 0.8
  }
  calculatePersonalShare cost = 22000 := by
  sorry


end carl_personal_share_l3797_379788


namespace triangle_probability_l3797_379721

def stick_lengths : List ℕ := [3, 4, 6, 8, 10, 12, 15, 18]

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle_combinations : List (ℕ × ℕ × ℕ) :=
  [(4, 6, 8), (6, 8, 10), (8, 10, 12), (10, 12, 15)]

def total_combinations : ℕ := Nat.choose 8 3

theorem triangle_probability :
  (List.length valid_triangle_combinations : ℚ) / total_combinations = 1 / 14 := by
  sorry

end triangle_probability_l3797_379721


namespace map_to_actual_distance_l3797_379711

/-- Given a map scale and a road length on the map, calculate the actual road length in kilometers. -/
theorem map_to_actual_distance (scale : ℚ) (map_length : ℚ) (actual_length : ℚ) : 
  scale = 1 / 50000 →
  map_length = 15 →
  actual_length = 7.5 →
  scale * actual_length = map_length := by
  sorry

#check map_to_actual_distance

end map_to_actual_distance_l3797_379711


namespace cars_served_4pm_to_6pm_l3797_379733

def peak_service_rate : ℕ := 12
def off_peak_service_rate : ℕ := 8
def blocks_per_hour : ℕ := 4

def cars_served_peak_hour : ℕ := peak_service_rate * blocks_per_hour
def cars_served_off_peak_hour : ℕ := off_peak_service_rate * blocks_per_hour

def total_cars_served : ℕ := cars_served_peak_hour + cars_served_off_peak_hour

theorem cars_served_4pm_to_6pm : total_cars_served = 80 := by
  sorry

end cars_served_4pm_to_6pm_l3797_379733


namespace velocity_at_4_seconds_l3797_379798

-- Define the motion equation
def motion_equation (t : ℝ) : ℝ := t^2 - t + 2

-- Define the instantaneous velocity function
def instantaneous_velocity (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem velocity_at_4_seconds :
  instantaneous_velocity 4 = 7 := by
  sorry

end velocity_at_4_seconds_l3797_379798


namespace quadratic_properties_l3797_379730

/-- The quadratic function f(x) = ax² + 4x + 2 passing through (3, -4) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 2

/-- The value of a for which f(x) passes through (3, -4) -/
def a : ℝ := -2

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := 1

theorem quadratic_properties :
  (f a 3 = -4) ∧
  (∀ x : ℝ, f a x = f a (2 * axis_of_symmetry - x)) ∧
  (∀ x : ℝ, x ≥ axis_of_symmetry → ∀ y : ℝ, y > x → f a y < f a x) :=
by sorry

end quadratic_properties_l3797_379730


namespace sum_of_first_five_primes_mod_sixth_prime_l3797_379714

def first_five_primes : List Nat := [2, 3, 5, 7, 11]
def sixth_prime : Nat := 13

theorem sum_of_first_five_primes_mod_sixth_prime :
  (first_five_primes.sum % sixth_prime) = 2 := by
  sorry

end sum_of_first_five_primes_mod_sixth_prime_l3797_379714


namespace shortest_major_axis_ellipse_l3797_379700

/-- The line l: y = x + 9 -/
def line_l (x : ℝ) : ℝ := x + 9

/-- The first focus of the ellipse -/
def F₁ : ℝ × ℝ := (-3, 0)

/-- The second focus of the ellipse -/
def F₂ : ℝ × ℝ := (3, 0)

/-- Definition of the ellipse equation -/
def is_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The theorem stating the equation of the ellipse with shortest major axis -/
theorem shortest_major_axis_ellipse :
  ∃ (P : ℝ × ℝ),
    (P.2 = line_l P.1) ∧
    is_ellipse_equation 45 36 P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ),
      (Q.2 = line_l Q.1) →
      (Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2 + (Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2 ≥
      (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 :=
by sorry

end shortest_major_axis_ellipse_l3797_379700


namespace outfit_combinations_l3797_379791

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) 
  (h_shirts : shirts = 8)
  (h_pants : pants = 5)
  (h_ties : ties = 4)
  (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end outfit_combinations_l3797_379791


namespace special_ellipse_equation_l3797_379747

/-- An ellipse with center at the origin, one focus at (0,2), intersected by the line y = 3x + 7 
    such that the midpoint of the intersection chord has a y-coordinate of 1 -/
structure SpecialEllipse where
  /-- The equation of the ellipse in the form (x²/a²) + (y²/b²) = 1 -/
  equation : ℝ → ℝ → Prop
  /-- One focus of the ellipse is at (0,2) -/
  focus_at_0_2 : ∃ (x y : ℝ), equation x y ∧ x = 0 ∧ y = 2
  /-- The line y = 3x + 7 intersects the ellipse -/
  intersects_line : ∃ (x y : ℝ), equation x y ∧ y = 3*x + 7
  /-- The midpoint of the intersection chord has a y-coordinate of 1 -/
  midpoint_y_is_1 : 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      equation x₁ y₁ ∧ y₁ = 3*x₁ + 7 ∧
      equation x₂ y₂ ∧ y₂ = 3*x₂ + 7 ∧
      (y₁ + y₂) / 2 = 1

/-- The equation of the special ellipse is x²/8 + y²/12 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : 
  e.equation = fun x y => x^2/8 + y^2/12 = 1 := by
  sorry

end special_ellipse_equation_l3797_379747


namespace bryden_receives_correct_amount_l3797_379709

/-- The amount Bryden receives for selling state quarters -/
def bryden_receive (num_quarters : ℕ) (face_value : ℚ) (collector_offer_percent : ℕ) : ℚ :=
  num_quarters * face_value * (collector_offer_percent : ℚ) / 100

/-- Theorem stating that Bryden receives $31.25 for selling five state quarters -/
theorem bryden_receives_correct_amount :
  bryden_receive 5 (1/4) 2500 = 125/4 :=
sorry

end bryden_receives_correct_amount_l3797_379709


namespace intersection_of_sets_l3797_379718

theorem intersection_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-2, 2}
  A ∩ B = {2} := by
sorry

end intersection_of_sets_l3797_379718


namespace binomial_expansion_theorem_l3797_379703

-- Define the binomial expansion function
def binomial_expansion (a : ℝ) (n : ℕ) : ℝ → ℝ := sorry

-- Define the sum of coefficients function
def sum_of_coefficients (a : ℝ) (n : ℕ) : ℝ := sorry

-- Define the sum of binomial coefficients function
def sum_of_binomial_coefficients (n : ℕ) : ℕ := sorry

-- Define the coefficient of x^2 function
def coefficient_of_x_squared (a : ℝ) (n : ℕ) : ℝ := sorry

theorem binomial_expansion_theorem (a : ℝ) (n : ℕ) :
  sum_of_coefficients a n = -1 ∧
  sum_of_binomial_coefficients n = 32 →
  coefficient_of_x_squared a n = 120 := by sorry

end binomial_expansion_theorem_l3797_379703


namespace right_angled_triangle_l3797_379722

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem right_angled_triangle (abc : Triangle) 
  (h : Real.sin abc.A = Real.sin abc.C * Real.cos abc.B) : 
  abc.C = π / 2 := by
  sorry

end right_angled_triangle_l3797_379722


namespace sues_necklace_beads_l3797_379755

/-- The number of beads in Sue's necklace -/
def total_beads (purple : ℕ) (blue : ℕ) (green : ℕ) : ℕ :=
  purple + blue + green

/-- Theorem stating the total number of beads in Sue's necklace -/
theorem sues_necklace_beads : 
  ∀ (purple blue green : ℕ),
    purple = 7 →
    blue = 2 * purple →
    green = blue + 11 →
    total_beads purple blue green = 46 := by
  sorry

end sues_necklace_beads_l3797_379755


namespace part_one_part_two_l3797_379779

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (2 * x - 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a < 0}

-- Part (I)
theorem part_one :
  (A ∩ B 4 = {x | 1/2 < x ∧ x < 2}) ∧
  (A ∪ B 4 = {x | -2 < x ∧ x ≤ 3}) := by sorry

-- Part (II)
theorem part_two :
  (∀ a, B a ∩ (Set.univ \ A) = B a) →
  {a | a ≤ 1/4} = Set.Iic (1/4) := by sorry

end part_one_part_two_l3797_379779


namespace white_longer_than_blue_l3797_379770

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.666666666666667

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := 3.3333333333333335

/-- The difference in length between the white and blue lines -/
def length_difference : ℝ := white_line_length - blue_line_length

theorem white_longer_than_blue :
  length_difference = 4.333333333333333 := by sorry

end white_longer_than_blue_l3797_379770


namespace number_in_scientific_notation_l3797_379738

/-- Scientific notation representation of a positive real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem number_in_scientific_notation :
  toScientificNotation 36600 = ScientificNotation.mk 3.66 4 (by sorry) :=
sorry

end number_in_scientific_notation_l3797_379738


namespace arithmetic_sequence_first_term_l3797_379739

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term
  (d : ℚ)
  (h1 : d = 5)
  (h2 : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → arithmetic_sum a d (4 * n) / arithmetic_sum a d n = c) :
  a = 5 / 2 :=
sorry

end arithmetic_sequence_first_term_l3797_379739


namespace number_of_envelopes_l3797_379761

-- Define the weight of a single envelope in grams
def envelope_weight : ℝ := 8.5

-- Define the total weight in kilograms
def total_weight_kg : ℝ := 6.8

-- Define the conversion factor from kg to g
def kg_to_g : ℝ := 1000

-- Theorem to prove
theorem number_of_envelopes : 
  (total_weight_kg * kg_to_g) / envelope_weight = 800 := by
  sorry

end number_of_envelopes_l3797_379761


namespace cookie_jar_solution_l3797_379736

def cookie_jar_problem (initial_amount : ℝ) : Prop :=
  let doris_spent : ℝ := 6
  let martha_spent : ℝ := doris_spent / 2
  let remaining_after_doris_martha : ℝ := initial_amount - doris_spent - martha_spent
  let john_spent_percentage : ℝ := 0.2
  let john_spent : ℝ := john_spent_percentage * remaining_after_doris_martha
  let final_amount : ℝ := remaining_after_doris_martha - john_spent
  final_amount = 15

theorem cookie_jar_solution :
  ∃ (initial_amount : ℝ), cookie_jar_problem initial_amount ∧ initial_amount = 27.75 := by
  sorry

end cookie_jar_solution_l3797_379736


namespace henry_earnings_l3797_379764

/-- Henry's lawn mowing earnings calculation -/
theorem henry_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  rate = 5 → total_lawns = 12 → forgotten_lawns = 7 →
  (total_lawns - forgotten_lawns) * rate = 25 :=
by sorry

end henry_earnings_l3797_379764


namespace dave_baseball_cards_pages_l3797_379776

/-- The number of pages needed to organize baseball cards in a binder -/
def pages_needed (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards + cards_per_page - 1) / cards_per_page

/-- Proof that Dave needs 2 pages to organize his baseball cards -/
theorem dave_baseball_cards_pages :
  pages_needed 8 3 13 = 2 := by
  sorry

end dave_baseball_cards_pages_l3797_379776


namespace locus_of_tangent_circles_l3797_379782

-- Define the circles C₃ and C₄
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₄ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define the property of being externally tangent to C₃
def externally_tangent_C₃ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define the property of being internally tangent to C₄
def internally_tangent_C₄ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := a^2 + 5*b^2 - 32*a - 51 = 0

-- State the theorem
theorem locus_of_tangent_circles :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_tangent_C₃ a b r ∧ internally_tangent_C₄ a b r) ↔
  locus_equation a b :=
sorry

end locus_of_tangent_circles_l3797_379782


namespace games_purchased_l3797_379792

theorem games_purchased (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 104 → spent_amount = 41 → game_cost = 9 →
  (initial_amount - spent_amount) / game_cost = 7 := by
  sorry

end games_purchased_l3797_379792


namespace real_roots_condition_one_root_triple_other_l3797_379727

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x + y = a ∧ 1/x + 1/y = 1/b

-- Theorem for real roots condition
theorem real_roots_condition (a b : ℝ) :
  (∃ x y, system x y a b) ↔ (a > 0 ∧ b ≤ a/4) ∨ (a < 0 ∧ b ≥ a/4) :=
sorry

-- Theorem for one root being three times the other
theorem one_root_triple_other (a b : ℝ) :
  (∃ x y, system x y a b ∧ x = 3*y) ↔ b = 3*a/16 :=
sorry

end real_roots_condition_one_root_triple_other_l3797_379727


namespace sosnovka_petrovka_distance_l3797_379745

/-- The distance between two points on a road --/
def distance (a b : ℕ) : ℕ := max a b - min a b

theorem sosnovka_petrovka_distance :
  ∀ (A B P S : ℕ),
  distance A P = 70 →
  distance A B = 20 →
  distance B S = 130 →
  distance S P = 180 :=
by
  sorry

end sosnovka_petrovka_distance_l3797_379745


namespace height_tiles_count_l3797_379754

def shower_tiles (num_walls : ℕ) (width_tiles : ℕ) (total_tiles : ℕ) : ℕ :=
  total_tiles / (num_walls * width_tiles)

theorem height_tiles_count : shower_tiles 3 8 480 = 20 := by
  sorry

end height_tiles_count_l3797_379754


namespace prime_pair_divisibility_l3797_379765

theorem prime_pair_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q →
  (p^p + q^q + 1 ≡ 0 [MOD pq] ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
  sorry

end prime_pair_divisibility_l3797_379765


namespace jakes_drink_volume_l3797_379799

/-- A drink recipe with parts of different ingredients -/
structure DrinkRecipe where
  coke_parts : ℕ
  sprite_parts : ℕ
  mountain_dew_parts : ℕ

/-- Calculate the total volume of a drink given its recipe and the volume of one ingredient -/
def total_volume (recipe : DrinkRecipe) (coke_volume : ℚ) : ℚ :=
  let total_parts := recipe.coke_parts + recipe.sprite_parts + recipe.mountain_dew_parts
  let volume_per_part := coke_volume / recipe.coke_parts
  total_parts * volume_per_part

/-- Theorem stating that for the given recipe and Coke volume, the total volume is 22 ounces -/
theorem jakes_drink_volume : 
  let recipe := DrinkRecipe.mk 4 2 5
  total_volume recipe 8 = 22 := by
  sorry

end jakes_drink_volume_l3797_379799


namespace max_k_logarithm_inequality_l3797_379740

theorem max_k_logarithm_inequality (x₀ x₁ x₂ x₃ : ℝ) 
  (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) :
  let log_base (b a : ℝ) := Real.log a / Real.log b
  9 * log_base (x₀ / x₃) 1993 ≤ 
    log_base (x₀ / x₁) 1993 + log_base (x₁ / x₂) 1993 + log_base (x₂ / x₃) 1993 ∧
  ∀ k > 9, ∃ x₀' x₁' x₂' x₃' : ℝ, x₀' > x₁' ∧ x₁' > x₂' ∧ x₂' > x₃' ∧ x₃' > 0 ∧
    k * log_base (x₀' / x₃') 1993 > 
      log_base (x₀' / x₁') 1993 + log_base (x₁' / x₂') 1993 + log_base (x₂' / x₃') 1993 :=
by sorry

end max_k_logarithm_inequality_l3797_379740


namespace snow_leopard_arrangement_l3797_379719

theorem snow_leopard_arrangement (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 3) :
  (k.factorial) * ((n - k).factorial) = 4320 := by
  sorry

end snow_leopard_arrangement_l3797_379719


namespace f_of_2_eq_neg_2_l3797_379750

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_of_2_eq_neg_2 : f 2 = -2 := by
  sorry

end f_of_2_eq_neg_2_l3797_379750


namespace john_beats_per_minute_l3797_379771

/-- Calculates the number of beats per minute John can play given his playing schedule and total beats played. -/
def beats_per_minute (hours_per_day : ℕ) (days : ℕ) (total_beats : ℕ) : ℕ :=
  total_beats / (hours_per_day * days * 60)

/-- Theorem stating that John can play 200 beats per minute given the problem conditions. -/
theorem john_beats_per_minute :
  beats_per_minute 2 3 72000 = 200 := by
  sorry

end john_beats_per_minute_l3797_379771


namespace jenna_work_hours_l3797_379790

def concert_ticket_cost : ℝ := 181
def drink_ticket_cost : ℝ := 7
def num_drink_tickets : ℕ := 5
def hourly_wage : ℝ := 18
def salary_percentage : ℝ := 0.1
def weeks_per_month : ℕ := 4

theorem jenna_work_hours :
  ∀ (weekly_hours : ℝ),
  (concert_ticket_cost + num_drink_tickets * drink_ticket_cost = 
   salary_percentage * (weekly_hours * hourly_wage * weeks_per_month)) →
  weekly_hours = 30 := by
  sorry

end jenna_work_hours_l3797_379790


namespace xy_and_sum_of_squares_l3797_379734

theorem xy_and_sum_of_squares (x y : ℝ) 
  (sum_eq : x + y = 3) 
  (prod_eq : (x + 2) * (y + 2) = 12) : 
  (xy = 2) ∧ (x^2 + 3*x*y + y^2 = 11) := by
  sorry


end xy_and_sum_of_squares_l3797_379734


namespace jerry_has_36_stickers_l3797_379717

-- Define the number of stickers for each person
def fred_stickers : ℕ := 18
def george_stickers : ℕ := fred_stickers - 6
def jerry_stickers : ℕ := 3 * george_stickers
def carla_stickers : ℕ := jerry_stickers + (jerry_stickers / 4)

-- Theorem to prove
theorem jerry_has_36_stickers : jerry_stickers = 36 := by
  sorry

end jerry_has_36_stickers_l3797_379717


namespace g_negative_in_range_l3797_379751

def f (a x : ℝ) : ℝ := x^3 + 3*a*x - 1

def g (a x : ℝ) : ℝ := (3*x^2 + 3*a) - a*x - 5

theorem g_negative_in_range :
  ∀ x : ℝ, -2/3 < x → x < 1 →
    ∀ a : ℝ, -1 ≤ a → a ≤ 1 →
      g a x < 0 :=
by sorry

end g_negative_in_range_l3797_379751


namespace trig_expression_evaluation_l3797_379797

open Real

theorem trig_expression_evaluation (x : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = fun x => sin x + cos x) 
  (hf' : deriv f = fun x => 3 * f x) : 
  (sin x)^2 - 3 / ((cos x)^2 + 1) = -14/9 := by
sorry

end trig_expression_evaluation_l3797_379797


namespace triangle_height_l3797_379732

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 10 → area = 25 → area = (base * height) / 2 → height = 5 := by
  sorry

end triangle_height_l3797_379732


namespace complex_fraction_simplification_l3797_379724

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = (1 : ℂ) / 3 + (14 : ℂ) / 15 * i :=
by sorry

end complex_fraction_simplification_l3797_379724


namespace debt_payment_additional_amount_l3797_379710

theorem debt_payment_additional_amount 
  (total_installments : ℕ)
  (first_payment_count : ℕ)
  (remaining_payment_count : ℕ)
  (first_payment_amount : ℚ)
  (average_payment : ℚ)
  (h1 : total_installments = 52)
  (h2 : first_payment_count = 12)
  (h3 : remaining_payment_count = total_installments - first_payment_count)
  (h4 : first_payment_amount = 410)
  (h5 : average_payment = 460) :
  let additional_amount := (total_installments * average_payment - 
    first_payment_count * first_payment_amount) / remaining_payment_count - 
    first_payment_amount
  additional_amount = 65 := by sorry

end debt_payment_additional_amount_l3797_379710


namespace quadratic_vertex_form_equivalence_l3797_379768

/-- Represents a quadratic function in the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadratic function in vertex form a(x - m)² + n -/
structure VertexForm where
  a : ℝ
  m : ℝ
  n : ℝ

/-- The vertex of a quadratic function -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem stating the equivalence of the standard and vertex forms of a specific quadratic function,
    and identifying its vertex -/
theorem quadratic_vertex_form_equivalence :
  let f : QuadraticFunction := { a := 2, b := -12, c := -12 }
  let v : VertexForm := { a := 2, m := 3, n := -30 }
  let vertex : Vertex := { x := 3, y := -30 }
  (∀ x, 2 * x^2 - 12 * x - 12 = 2 * (x - 3)^2 - 30) ∧
  (vertex.x = -f.b / (2 * f.a) ∧ vertex.y = f.c - f.b^2 / (4 * f.a)) := by
  sorry

end quadratic_vertex_form_equivalence_l3797_379768


namespace ahmed_hassan_tree_difference_l3797_379781

/-- Represents the number of trees in an orchard -/
structure Orchard :=
  (apple : ℕ)
  (orange : ℕ)

/-- Calculate the total number of trees in an orchard -/
def totalTrees (o : Orchard) : ℕ := o.apple + o.orange

/-- The difference in the number of trees between two orchards -/
def treeDifference (o1 o2 : Orchard) : ℕ := (totalTrees o1) - (totalTrees o2)

theorem ahmed_hassan_tree_difference :
  let ahmed : Orchard := { apple := 4, orange := 8 }
  let hassan : Orchard := { apple := 1, orange := 2 }
  treeDifference ahmed hassan = 9 := by
  sorry

end ahmed_hassan_tree_difference_l3797_379781


namespace employee_pay_problem_l3797_379763

theorem employee_pay_problem (total_pay : ℝ) (a_percent : ℝ) (b_pay : ℝ) :
  total_pay = 550 →
  a_percent = 1.2 →
  total_pay = b_pay + a_percent * b_pay →
  b_pay = 250 := by
  sorry

end employee_pay_problem_l3797_379763


namespace intersection_points_sum_l3797_379746

theorem intersection_points_sum (m : ℕ) (h : m = 17) : 
  ∃ (x : ℕ), 
    (∀ y : ℕ, (y ≡ 6*x + 3 [MOD m] ↔ y ≡ 13*x + 8 [MOD m])) ∧ 
    x = 7 := by
  sorry

end intersection_points_sum_l3797_379746


namespace quadratic_negative_root_l3797_379720

theorem quadratic_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by sorry

end quadratic_negative_root_l3797_379720


namespace tim_zoo_cost_l3797_379737

/-- The total cost of animals Tim bought for his zoo -/
def total_cost (goat_price : ℝ) (goat_count : ℕ) (llama_price_ratio : ℝ) : ℝ :=
  let llama_count := 2 * goat_count
  let llama_price := goat_price * llama_price_ratio
  goat_price * goat_count + llama_price * llama_count

/-- Theorem stating the total cost of animals for Tim's zoo -/
theorem tim_zoo_cost : total_cost 400 3 1.5 = 4800 := by
  sorry

end tim_zoo_cost_l3797_379737


namespace final_lives_calculation_l3797_379774

def calculate_final_lives (initial_lives lives_lost gain_factor : ℕ) : ℕ :=
  initial_lives - lives_lost + gain_factor * lives_lost

theorem final_lives_calculation (initial_lives lives_lost gain_factor : ℕ) :
  calculate_final_lives initial_lives lives_lost gain_factor =
  initial_lives - lives_lost + gain_factor * lives_lost :=
by
  sorry

-- Example usage
example : calculate_final_lives 75 28 3 = 131 :=
by
  sorry

end final_lives_calculation_l3797_379774


namespace cross_product_perpendicular_l3797_379787

def v1 : ℝ × ℝ × ℝ := (4, 3, -5)
def v2 : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  a₁ * b₁ + a₂ * b₂ + a₃ * b₃

theorem cross_product_perpendicular :
  let result := cross_product v1 v2
  result = (7, -26, -10) ∧
  dot_product v1 result = 0 ∧
  dot_product v2 result = 0 := by
  sorry

end cross_product_perpendicular_l3797_379787


namespace complex_roots_magnitude_l3797_379748

theorem complex_roots_magnitude (p : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, x^2 + p*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  Complex.abs x₁ = 2 := by
  sorry

end complex_roots_magnitude_l3797_379748


namespace probability_differ_by_2_l3797_379742

/-- A standard 6-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling a die twice -/
def TwoRolls : Type := Die × Die

/-- The condition for two rolls to differ by 2 -/
def DifferBy2 (roll : TwoRolls) : Prop :=
  (roll.1.val + 1 = roll.2.val) ∨ (roll.1.val = roll.2.val + 1)

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 8

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 36

theorem probability_differ_by_2 :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 2 / 9 := by
  sorry

end probability_differ_by_2_l3797_379742


namespace square_area_not_covered_by_circle_l3797_379712

theorem square_area_not_covered_by_circle (d : ℝ) (h : d = 8) :
  let r := d / 2
  let square_area := d^2
  let circle_area := π * r^2
  square_area - circle_area = 64 - 16 * π := by
  sorry

end square_area_not_covered_by_circle_l3797_379712


namespace age_difference_l3797_379715

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 28 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 30 := by
  sorry

end age_difference_l3797_379715


namespace binary_sum_theorem_l3797_379713

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n |>.reverse

def a : List Bool := [true, true, false]
def b : List Bool := [true, false, true]
def c : List Bool := [true, false, true, true]
def d : List Bool := [true, false, false, true, true]

theorem binary_sum_theorem :
  decimal_to_binary (binary_to_decimal a + binary_to_decimal b + 
                     binary_to_decimal c + binary_to_decimal d) =
  [true, false, true, false, false, true] := by
  sorry

end binary_sum_theorem_l3797_379713


namespace eight_b_value_l3797_379766

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := by
  sorry

end eight_b_value_l3797_379766


namespace hash_three_times_100_l3797_379701

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N + N

-- Theorem statement
theorem hash_three_times_100 : hash (hash (hash 100)) = 337.5 := by
  sorry

end hash_three_times_100_l3797_379701


namespace g_of_5_eq_50_l3797_379708

/-- The polynomial g(x) = 3x^4 - 20x^3 + 40x^2 - 50x - 75 -/
def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 40*x^2 - 50*x - 75

/-- Theorem: g(5) = 50 -/
theorem g_of_5_eq_50 : g 5 = 50 := by
  sorry

end g_of_5_eq_50_l3797_379708


namespace cupcake_ratio_l3797_379752

theorem cupcake_ratio (total : ℕ) (gluten_free : ℕ) (vegan : ℕ) (non_vegan_gluten : ℕ) :
  total = 80 →
  gluten_free = total / 2 →
  vegan = 24 →
  non_vegan_gluten = 28 →
  (gluten_free - non_vegan_gluten) / vegan = 1 / 2 := by
sorry

end cupcake_ratio_l3797_379752


namespace f_equals_g_l3797_379735

-- Define the functions
def f (x : ℝ) : ℝ := (76 * x^6)^7
def g (x : ℝ) : ℝ := |x|

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end f_equals_g_l3797_379735


namespace point_transformation_l3797_379723

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (a b : ℝ) :
  let Q := (a, b)
  let rotated := rotate_180 a b 2 3
  let final := reflect_y_eq_neg_x rotated.1 rotated.2
  final = (5, -1) → b - a = 6 := by
  sorry

end point_transformation_l3797_379723


namespace base_conversion_1729_l3797_379716

def base_10_to_base_6 (n : ℕ) : List ℕ :=
  sorry

theorem base_conversion_1729 :
  base_10_to_base_6 1729 = [1, 2, 0, 0, 1] :=
sorry

end base_conversion_1729_l3797_379716


namespace similar_triangles_height_l3797_379731

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  let scale_factor := Real.sqrt area_ratio
  let h_large := h_small * scale_factor
  h_small = 5 →
  h_large = 15 := by
  sorry

end similar_triangles_height_l3797_379731


namespace problem_solution_l3797_379704

theorem problem_solution (x : ℤ) (h : x = 40) : x * 6 - 138 = 102 := by
  sorry

end problem_solution_l3797_379704


namespace quadratic_function_minimum_l3797_379789

theorem quadratic_function_minimum (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → a * (x - 1)^2 - a ≥ -4) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 4 ∧ a * (x - 1)^2 - a = -4) →
  a = 4 ∨ a = -1/2 := by
sorry

end quadratic_function_minimum_l3797_379789


namespace discount_reduction_l3797_379757

theorem discount_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.2
  let remaining_after_first := 1 - first_discount
  let remaining_after_second := 1 - second_discount
  let final_price := original_price * remaining_after_first * remaining_after_second
  (original_price - final_price) / original_price = 0.44 :=
by
  sorry

end discount_reduction_l3797_379757


namespace total_buyers_is_140_l3797_379728

/-- The number of buyers who visited a store over three consecutive days -/
def total_buyers (day_before_yesterday yesterday today : ℕ) : ℕ :=
  day_before_yesterday + yesterday + today

/-- Theorem stating the total number of buyers over three days -/
theorem total_buyers_is_140 :
  ∃ (yesterday today : ℕ),
    yesterday = 50 / 2 ∧
    today = yesterday + 40 ∧
    total_buyers 50 yesterday today = 140 :=
by sorry

end total_buyers_is_140_l3797_379728


namespace unpainted_area_triangular_board_l3797_379780

/-- The area of the unpainted region on a triangular board that intersects with a rectangular board -/
theorem unpainted_area_triangular_board (base height width intersection_angle : ℝ) 
  (h_base : base = 8)
  (h_height : height = 10)
  (h_width : width = 5)
  (h_angle : intersection_angle = 45) :
  base * height / 2 - width * height = 50 := by
  sorry

end unpainted_area_triangular_board_l3797_379780


namespace square_area_ratio_after_doubling_l3797_379795

theorem square_area_ratio_after_doubling (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
sorry

end square_area_ratio_after_doubling_l3797_379795


namespace ham_slices_per_sandwich_l3797_379786

theorem ham_slices_per_sandwich :
  ∀ (initial_slices : ℕ) (additional_slices : ℕ) (total_sandwiches : ℕ),
    initial_slices = 31 →
    additional_slices = 119 →
    total_sandwiches = 50 →
    (initial_slices + additional_slices) / total_sandwiches = 3 :=
by
  sorry

end ham_slices_per_sandwich_l3797_379786


namespace seniority_ranking_l3797_379762

-- Define the colleagues
inductive Colleague
| Julia
| Kevin
| Lana

-- Define the seniority relation
def more_senior (a b : Colleague) : Prop := sorry

-- Define the most senior and least senior
def most_senior (c : Colleague) : Prop :=
  ∀ other, c ≠ other → more_senior c other

def least_senior (c : Colleague) : Prop :=
  ∀ other, c ≠ other → more_senior other c

-- Define the statements
def statement_I : Prop := most_senior Colleague.Kevin
def statement_II : Prop := least_senior Colleague.Lana
def statement_III : Prop := ¬(least_senior Colleague.Julia)

-- Main theorem
theorem seniority_ranking :
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) →
  (more_senior Colleague.Kevin Colleague.Lana ∧
   more_senior Colleague.Lana Colleague.Julia) :=
by sorry

end seniority_ranking_l3797_379762


namespace dice_product_div_eight_prob_l3797_379785

/-- Represents a standard 6-sided die --/
def Die : Type := Fin 6

/-- The probability space of rolling 8 dice --/
def DiceRoll : Type := Fin 8 → Die

/-- A function that determines if a number is divisible by 8 --/
def divisible_by_eight (n : ℕ) : Prop := n % 8 = 0

/-- The product of the numbers shown on the dice --/
def dice_product (roll : DiceRoll) : ℕ :=
  (List.range 8).foldl (λ acc i => acc * (roll i).val.succ) 1

/-- The event that the product of the dice roll is divisible by 8 --/
def event_divisible_by_eight (roll : DiceRoll) : Prop :=
  divisible_by_eight (dice_product roll)

/-- The probability measure on the dice roll space --/
axiom prob : (DiceRoll → Prop) → ℚ

/-- The probability of the event is well-defined --/
axiom prob_well_defined : ∀ (E : DiceRoll → Prop), 0 ≤ prob E ∧ prob E ≤ 1

theorem dice_product_div_eight_prob :
  prob event_divisible_by_eight = 199 / 256 := by
  sorry

end dice_product_div_eight_prob_l3797_379785


namespace stratified_sample_theorem_l3797_379772

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  total_population : ℕ
  boys_population : ℕ
  girls_population : ℕ
  sample_size : ℕ

/-- Calculates the number of boys in the sample -/
def boys_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.boys_population) / s.total_population

/-- Calculates the number of girls in the sample -/
def girls_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.girls_population) / s.total_population

/-- Theorem stating the correct number of boys and girls in the sample -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_population = 700)
  (h2 : s.boys_population = 385)
  (h3 : s.girls_population = 315)
  (h4 : s.sample_size = 60) :
  boys_in_sample s = 33 ∧ girls_in_sample s = 27 := by
  sorry

#eval boys_in_sample { total_population := 700, boys_population := 385, girls_population := 315, sample_size := 60 }
#eval girls_in_sample { total_population := 700, boys_population := 385, girls_population := 315, sample_size := 60 }

end stratified_sample_theorem_l3797_379772


namespace optimal_price_and_quantity_l3797_379743

/-- Represents the pricing and sales model of a shopping mall --/
structure ShoppingMall where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_elasticity : ℝ
  target_profit : ℝ

/-- Calculates the monthly sales volume based on the new price --/
def sales_volume (mall : ShoppingMall) (new_price : ℝ) : ℝ :=
  mall.initial_sales - mall.price_elasticity * (new_price - mall.initial_price)

/-- Calculates the monthly profit based on the new price --/
def monthly_profit (mall : ShoppingMall) (new_price : ℝ) : ℝ :=
  (new_price - mall.cost_price) * sales_volume mall new_price

/-- Theorem stating that the new price and purchase quantity achieve the target profit --/
theorem optimal_price_and_quantity (mall : ShoppingMall) 
  (h_cost : mall.cost_price = 20)
  (h_initial_price : mall.initial_price = 30)
  (h_initial_sales : mall.initial_sales = 800)
  (h_elasticity : mall.price_elasticity = 20)
  (h_target : mall.target_profit = 12000) :
  ∃ (new_price : ℝ), 
    new_price = 40 ∧ 
    sales_volume mall new_price = 600 ∧ 
    monthly_profit mall new_price = mall.target_profit := by
  sorry

end optimal_price_and_quantity_l3797_379743


namespace admission_cutoff_score_l3797_379784

theorem admission_cutoff_score (
  admitted_fraction : Real)
  (admitted_avg_diff : Real)
  (not_admitted_avg_diff : Real)
  (overall_avg : Real)
  (h1 : admitted_fraction = 2 / 5)
  (h2 : admitted_avg_diff = 15)
  (h3 : not_admitted_avg_diff = -20)
  (h4 : overall_avg = 90) :
  let cutoff_score := 
    (overall_avg - admitted_fraction * admitted_avg_diff - (1 - admitted_fraction) * not_admitted_avg_diff) /
    (admitted_fraction + (1 - admitted_fraction))
  cutoff_score = 96 := by
  sorry

end admission_cutoff_score_l3797_379784


namespace methane_combustion_l3797_379760

/-- Represents the balanced chemical equation for methane combustion -/
structure MethaneReaction where
  ch4 : ℚ
  o2 : ℚ
  co2 : ℚ
  h2o : ℚ
  balanced : ch4 = 1 ∧ o2 = 2 ∧ co2 = 1 ∧ h2o = 2

/-- Theorem stating the number of moles of CH₄ required and CO₂ formed when 2 moles of O₂ react -/
theorem methane_combustion (reaction : MethaneReaction) (o2_moles : ℚ) 
  (h_o2 : o2_moles = 2) : 
  let ch4_required := o2_moles / reaction.o2 * reaction.ch4
  let co2_formed := ch4_required / reaction.ch4 * reaction.co2
  ch4_required = 1 ∧ co2_formed = 1 := by
  sorry


end methane_combustion_l3797_379760


namespace candy_count_third_set_l3797_379744

/-- Represents a set of candies with hard candies, chocolates, and gummy candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies in a set -/
def CandySet.total (s : CandySet) : ℕ := s.hard + s.chocolate + s.gummy

theorem candy_count_third_set (set1 set2 set3 : CandySet) : 
  /- Total number of each type is equal across all sets -/
  (set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate) ∧
  (set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy) ∧
  /- First set conditions -/
  (set1.chocolate = set1.gummy) ∧
  (set1.hard = set1.chocolate + 7) ∧
  /- Second set conditions -/
  (set2.hard = set2.chocolate) ∧
  (set2.gummy = set2.hard - 15) ∧
  /- Third set condition -/
  (set3.hard = 0) →
  /- Conclusion: total number of candies in the third set is 29 -/
  set3.total = 29 := by
  sorry

end candy_count_third_set_l3797_379744


namespace expression_evaluation_l3797_379783

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = a + 8)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23 / 15 := by
  sorry

end expression_evaluation_l3797_379783


namespace remainder_problem_l3797_379758

theorem remainder_problem (d : ℕ) (h1 : d = 170) (h2 : d ∣ 690) (h3 : d ∣ 875) 
  (h4 : 875 % d = 25) (h5 : ∀ k : ℕ, k > d → ¬(k ∣ 690 ∧ k ∣ 875)) : 
  690 % d = 10 := by
  sorry

end remainder_problem_l3797_379758


namespace pet_store_siamese_cats_l3797_379777

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 5

/-- The total number of cats sold during the sale -/
def cats_sold : ℕ := 10

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is 13 -/
theorem pet_store_siamese_cats : 
  initial_siamese_cats = 13 ∧ 
  initial_siamese_cats + initial_house_cats = cats_sold + cats_remaining :=
by sorry

end pet_store_siamese_cats_l3797_379777


namespace clerical_staff_reduction_l3797_379706

theorem clerical_staff_reduction (total_employees : ℕ) 
  (initial_clerical_ratio : ℚ) (clerical_reduction_ratio : ℚ) : 
  total_employees = 3600 →
  initial_clerical_ratio = 1/3 →
  clerical_reduction_ratio = 1/2 →
  let initial_clerical := (initial_clerical_ratio * total_employees : ℚ).num
  let remaining_clerical := (1 - clerical_reduction_ratio) * initial_clerical
  let new_total := total_employees - (clerical_reduction_ratio * initial_clerical : ℚ).num
  (remaining_clerical / new_total : ℚ) = 1/5 := by
sorry

end clerical_staff_reduction_l3797_379706


namespace super_vcd_cost_price_l3797_379741

theorem super_vcd_cost_price (x : ℝ) : 
  x * (1 + 0.4) * 0.9 - 50 = x + 340 → x = 1500 := by sorry

end super_vcd_cost_price_l3797_379741


namespace noah_age_after_10_years_l3797_379775

def joe_age : ℕ := 6
def noah_age : ℕ := 2 * joe_age
def years_passed : ℕ := 10

theorem noah_age_after_10_years :
  noah_age + years_passed = 22 := by
  sorry

end noah_age_after_10_years_l3797_379775


namespace corridor_lights_l3797_379753

/-- The number of ways to choose k non-adjacent items from n consecutive items -/
def nonAdjacentChoices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

/-- Theorem: There are 20 ways to choose 3 non-adjacent positions from 8 consecutive positions -/
theorem corridor_lights : nonAdjacentChoices 8 3 = 20 := by
  sorry

#eval nonAdjacentChoices 8 3

end corridor_lights_l3797_379753


namespace first_marvelous_monday_l3797_379767

/-- Represents a date in October --/
structure OctoberDate :=
  (day : ℕ)
  (is_monday : Bool)

/-- The number of days in October --/
def october_days : ℕ := 31

/-- The first day of school --/
def school_start : OctoberDate :=
  { day := 2, is_monday := true }

/-- A function to find the next Monday given a current date --/
def next_monday (d : OctoberDate) : OctoberDate :=
  { day := d.day + 7, is_monday := true }

/-- The definition of a Marvelous Monday --/
def is_marvelous_monday (d : OctoberDate) : Prop :=
  d.is_monday ∧ d.day ≤ october_days ∧ 
  (∀ m : OctoberDate, m.is_monday ∧ m.day > d.day → m.day > october_days)

/-- The theorem to prove --/
theorem first_marvelous_monday : 
  ∃ d : OctoberDate, d.day = 30 ∧ is_marvelous_monday d :=
sorry

end first_marvelous_monday_l3797_379767


namespace donald_oranges_l3797_379726

/-- Given that Donald has 4 oranges initially and finds 5 more,
    prove that he has 9 oranges in total. -/
theorem donald_oranges (initial : Nat) (found : Nat) (total : Nat) 
    (h1 : initial = 4) 
    (h2 : found = 5) 
    (h3 : total = initial + found) : 
  total = 9 := by
  sorry

end donald_oranges_l3797_379726


namespace find_b_value_l3797_379759

theorem find_b_value (a b : ℝ) (h1 : 2 * a + 3 = 5) (h2 : b - a = 2) : b = 3 := by
  sorry

end find_b_value_l3797_379759


namespace dog_feed_mix_problem_l3797_379702

/-- The cost per pound of the cheaper kind of feed -/
def cheaper_feed_cost : ℝ := 0.18

theorem dog_feed_mix_problem :
  -- Total weight of the mix
  let total_weight : ℝ := 35
  -- Cost per pound of the final mix
  let final_mix_cost : ℝ := 0.36
  -- Cost per pound of the more expensive feed
  let expensive_feed_cost : ℝ := 0.53
  -- Weight of the cheaper feed used
  let cheaper_feed_weight : ℝ := 17
  -- Weight of the more expensive feed used
  let expensive_feed_weight : ℝ := total_weight - cheaper_feed_weight
  -- Total value of the final mix
  let total_value : ℝ := total_weight * final_mix_cost
  -- Value of the more expensive feed
  let expensive_feed_value : ℝ := expensive_feed_weight * expensive_feed_cost
  -- Equation for the total value
  cheaper_feed_weight * cheaper_feed_cost + expensive_feed_value = total_value →
  cheaper_feed_cost = 0.18 := by
sorry

end dog_feed_mix_problem_l3797_379702


namespace complex_power_magnitude_l3797_379794

theorem complex_power_magnitude : 
  Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 3))^4 = 256 := by
  sorry

end complex_power_magnitude_l3797_379794


namespace triangle_problem_l3797_379796

theorem triangle_problem (A B C : Real) (a b c : Real) : 
  -- Given conditions
  (a = 2) →
  (b = Real.sqrt 6) →
  (B = 60 * π / 180) →
  -- Triangle properties
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusions
  (A = 45 * π / 180 ∧ 
   C = 75 * π / 180 ∧ 
   c = 1 + Real.sqrt 3) :=
by sorry

end triangle_problem_l3797_379796


namespace overall_discount_percentage_l3797_379756

/-- Calculate the overall discount percentage for three items given their cost prices, markups, and sale prices. -/
theorem overall_discount_percentage
  (cost_A cost_B cost_C : ℝ)
  (markup_A markup_B markup_C : ℝ)
  (sale_A sale_B sale_C : ℝ)
  (h_cost_A : cost_A = 540)
  (h_cost_B : cost_B = 620)
  (h_cost_C : cost_C = 475)
  (h_markup_A : markup_A = 0.15)
  (h_markup_B : markup_B = 0.20)
  (h_markup_C : markup_C = 0.25)
  (h_sale_A : sale_A = 462)
  (h_sale_B : sale_B = 558)
  (h_sale_C : sale_C = 405) :
  let marked_A := cost_A * (1 + markup_A)
  let marked_B := cost_B * (1 + markup_B)
  let marked_C := cost_C * (1 + markup_C)
  let total_marked := marked_A + marked_B + marked_C
  let total_sale := sale_A + sale_B + sale_C
  let discount_percentage := (total_marked - total_sale) / total_marked * 100
  ∃ ε > 0, |discount_percentage - 27.26| < ε :=
by sorry


end overall_discount_percentage_l3797_379756


namespace jack_weight_jack_weight_proof_l3797_379749

/-- Jack and Anna's see-saw problem -/
theorem jack_weight (anna_weight : ℕ) (num_rocks : ℕ) (rock_weight : ℕ) : ℕ :=
  let total_rock_weight := num_rocks * rock_weight
  let jack_weight := anna_weight - total_rock_weight
  jack_weight

/-- Proof of Jack's weight -/
theorem jack_weight_proof :
  jack_weight 40 5 4 = 20 := by
  sorry

end jack_weight_jack_weight_proof_l3797_379749


namespace parallel_line_not_through_point_l3797_379729

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

theorem parallel_line_not_through_point
    (L : Line)
    (P : Point)
    (h_not_on : ¬ P.onLine L) :
    ∃ (k : ℝ),
      k ≠ 0 ∧
      (∀ (x y : ℝ),
        L.A * x + L.B * y + L.C + (L.A * P.x + L.B * P.y + L.C) = 0 ↔
        L.A * x + L.B * y + L.C + k = 0) ∧
      (L.A * P.x + L.B * P.y + L.C + k ≠ 0) :=
  sorry

end parallel_line_not_through_point_l3797_379729


namespace inequality_equivalence_l3797_379769

theorem inequality_equivalence (x : ℝ) : 2 * x - 3 < x + 1 ↔ x < 4 := by
  sorry

end inequality_equivalence_l3797_379769


namespace P_root_nature_l3797_379707

/-- The polynomial P(x) = x^6 - 4x^5 - 9x^3 + 2x + 9 -/
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

theorem P_root_nature :
  (∀ x < 0, P x > 0) ∧ (∃ x > 0, P x = 0) := by sorry


end P_root_nature_l3797_379707


namespace last_date_with_sum_property_l3797_379725

/-- Represents a date in DD.MM.YYYY format -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  deriving Repr

/-- Checks if a given date is valid in the year 2008 -/
def isValidDate (d : Date) : Prop :=
  d.year = 2008 ∧
  d.month ≥ 1 ∧ d.month ≤ 12 ∧
  d.day ≥ 1 ∧ d.day ≤ 31 ∧
  (d.month ∈ [4, 6, 9, 11] → d.day ≤ 30) ∧
  (d.month = 2 → d.day ≤ 29)

/-- Extracts individual digits from a number -/
def digits (n : Nat) : List Nat :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

/-- Calculates the sum of the first four digits of a date -/
def sumFirstFour (d : Date) : Nat :=
  List.sum (List.take 4 (digits d.day ++ digits d.month))

/-- Calculates the sum of the last four digits of a date -/
def sumLastFour (d : Date) : Nat :=
  List.sum (List.take 4 (digits d.year).reverse)

/-- Checks if the sum of the first four digits equals the sum of the last four digits -/
def hasSumProperty (d : Date) : Prop :=
  sumFirstFour d = sumLastFour d

/-- States that December 25, 2008 is the last date in 2008 with the sum property -/
theorem last_date_with_sum_property :
  ∀ d : Date, isValidDate d → hasSumProperty d →
  d.year = 2008 → d.month ≤ 12 → d.day ≤ 25 :=
sorry

end last_date_with_sum_property_l3797_379725
