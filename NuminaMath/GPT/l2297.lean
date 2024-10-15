import Mathlib

namespace NUMINAMATH_GPT_number_of_possible_ceil_values_l2297_229753

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end NUMINAMATH_GPT_number_of_possible_ceil_values_l2297_229753


namespace NUMINAMATH_GPT_fraction_sum_is_half_l2297_229787

theorem fraction_sum_is_half :
  (1/5 : ℚ) + (3/10 : ℚ) = 1/2 :=
by linarith

end NUMINAMATH_GPT_fraction_sum_is_half_l2297_229787


namespace NUMINAMATH_GPT_correct_propositions_l2297_229773

-- Definitions according to the given conditions
def generatrix_cylinder (p1 p2 : Point) (c : Cylinder) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def generatrix_cone (v : Point) (p : Point) (c : Cone) : Prop :=
  -- Check if the line from the vertex to a base point is a generatrix
  sorry

def generatrix_frustum (p1 p2 : Point) (f : Frustum) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def parallel_generatrices_cylinder (gen1 gen2 : Line) (c : Cylinder) : Prop :=
  -- Check if two generatrices of the cylinder are parallel
  sorry

-- The theorem stating propositions ② and ④ are correct
theorem correct_propositions :
  generatrix_cone vertex point cone ∧
  parallel_generatrices_cylinder gen1 gen2 cylinder :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l2297_229773


namespace NUMINAMATH_GPT_inequality_solution_l2297_229739

noncomputable def solve_inequality (m : ℝ) (m_lt_neg2 : m < -2) : Set ℝ :=
  if h : m = -3 then {x | 1 < x}
  else if h' : -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
  else {x | 1 < x ∧ x < m / (m + 3)}

theorem inequality_solution (m : ℝ) (m_lt_neg2 : m < -2) :
  (solve_inequality m m_lt_neg2) = 
    if m = -3 then {x | 1 < x}
    else if -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
    else {x | 1 < x ∧ x < m / (m + 3)} :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2297_229739


namespace NUMINAMATH_GPT_polynomial_value_at_one_l2297_229703

theorem polynomial_value_at_one
  (a b c : ℝ)
  (h1 : -a - b - c + 1 = 6)
  : a + b + c + 1 = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_value_at_one_l2297_229703


namespace NUMINAMATH_GPT_num_integer_solutions_eq_3_l2297_229727

theorem num_integer_solutions_eq_3 :
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((2 * x^2) + (x * y) + (y^2) - x + 2 * y + 1 = 0 ↔ (x, y) ∈ S)) ∧ 
  S.card = 3 :=
sorry

end NUMINAMATH_GPT_num_integer_solutions_eq_3_l2297_229727


namespace NUMINAMATH_GPT_suitable_value_for_x_evaluates_to_neg1_l2297_229734

noncomputable def given_expression (x : ℝ) : ℝ :=
  (x^3 + 2 * x^2) / (x^2 - 4 * x + 4) / (4 * x + 8) - 1 / (x - 2)

theorem suitable_value_for_x_evaluates_to_neg1 : 
  given_expression (-6) = -1 :=
by
  sorry

end NUMINAMATH_GPT_suitable_value_for_x_evaluates_to_neg1_l2297_229734


namespace NUMINAMATH_GPT_temperature_on_Friday_l2297_229768

variable {M T W Th F : ℝ}

theorem temperature_on_Friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (hM : M = 41) :
  F = 33 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_temperature_on_Friday_l2297_229768


namespace NUMINAMATH_GPT_ab_value_l2297_229730

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_GPT_ab_value_l2297_229730


namespace NUMINAMATH_GPT_maximum_value_of_a_l2297_229759

theorem maximum_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) : a ≤ 2924 := 
sorry

end NUMINAMATH_GPT_maximum_value_of_a_l2297_229759


namespace NUMINAMATH_GPT_scientific_notation_example_l2297_229737

theorem scientific_notation_example :
  110000 = 1.1 * 10^5 :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_example_l2297_229737


namespace NUMINAMATH_GPT_rhombus_area_l2297_229735

theorem rhombus_area
  (side_length : ℝ)
  (h₀ : side_length = 2 * Real.sqrt 3)
  (tri_a_base : ℝ)
  (tri_b_base : ℝ)
  (h₁ : tri_a_base = side_length)
  (h₂ : tri_b_base = side_length) :
  ∃ rhombus_area : ℝ,
    rhombus_area = 8 * Real.sqrt 3 - 12 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l2297_229735


namespace NUMINAMATH_GPT_maximum_ab_expression_l2297_229740

open Function Real

theorem maximum_ab_expression {a b : ℝ} (h : 0 < a ∧ 0 < b ∧ 5 * a + 6 * b < 110) :
  ab * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 :=
sorry

end NUMINAMATH_GPT_maximum_ab_expression_l2297_229740


namespace NUMINAMATH_GPT_repaired_shoes_last_correct_l2297_229785

noncomputable def repaired_shoes_last := 
  let repair_cost: ℝ := 10.50
  let new_shoes_cost: ℝ := 30.00
  let new_shoes_years: ℝ := 2.0
  let percentage_increase: ℝ := 42.857142857142854 / 100
  (T : ℝ) -> 15.00 = (repair_cost / T) * (1 + percentage_increase) → T = 1

theorem repaired_shoes_last_correct : repaired_shoes_last :=
by
  sorry

end NUMINAMATH_GPT_repaired_shoes_last_correct_l2297_229785


namespace NUMINAMATH_GPT_swimmers_pass_each_other_l2297_229746

/-- Two swimmers in a 100-foot pool, one swimming at 4 feet per second, the other at 3 feet per second,
    continuously for 12 minutes, pass each other exactly 32 times. -/
theorem swimmers_pass_each_other 
  (pool_length : ℕ) 
  (time : ℕ) 
  (rate1 : ℕ)
  (rate2 : ℕ)
  (meet_times : ℕ)
  (hp : pool_length = 100) 
  (ht : time = 720) -- 12 minutes = 720 seconds
  (hr1 : rate1 = 4) 
  (hr2 : rate2 = 3)
  : meet_times = 32 := 
sorry

end NUMINAMATH_GPT_swimmers_pass_each_other_l2297_229746


namespace NUMINAMATH_GPT_min_value_of_fraction_l2297_229704

theorem min_value_of_fraction (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : (m * (-3) + n * (-1) + 2 = 0)) 
    (h4 : (m * (-2) + n * 0 + 2 = 0)) : 
    (1 / m + 3 / n) = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_fraction_l2297_229704


namespace NUMINAMATH_GPT_exists_five_consecutive_divisible_by_2014_l2297_229711

theorem exists_five_consecutive_divisible_by_2014 :
  ∃ (a b c d e : ℕ), 53 = a ∧ 54 = b ∧ 55 = c ∧ 56 = d ∧ 57 = e ∧ 100 > a ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ 2014 ∣ (a * b * c * d * e) :=
by 
  sorry

end NUMINAMATH_GPT_exists_five_consecutive_divisible_by_2014_l2297_229711


namespace NUMINAMATH_GPT_polynomial_lt_factorial_l2297_229706

theorem polynomial_lt_factorial (A B C : ℝ) : ∃N : ℕ, ∀n : ℕ, n > N → An^2 + Bn + C < n! := 
by
  sorry

end NUMINAMATH_GPT_polynomial_lt_factorial_l2297_229706


namespace NUMINAMATH_GPT_automobile_travel_distance_5_minutes_l2297_229743

variable (a r : ℝ)

theorem automobile_travel_distance_5_minutes (h0 : r ≠ 0) :
  let distance_in_feet := (2 * a) / 5
  let time_in_seconds := 300
  (distance_in_feet / r) * time_in_seconds / 3 = 40 * a / r :=
by
  sorry

end NUMINAMATH_GPT_automobile_travel_distance_5_minutes_l2297_229743


namespace NUMINAMATH_GPT_sum_cubic_polynomial_l2297_229712

noncomputable def q : ℤ → ℤ := sorry  -- We use a placeholder definition for q

theorem sum_cubic_polynomial :
  q 3 = 2 ∧ q 8 = 22 ∧ q 12 = 10 ∧ q 17 = 32 →
  (q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18) = 272 :=
sorry

end NUMINAMATH_GPT_sum_cubic_polynomial_l2297_229712


namespace NUMINAMATH_GPT_find_a_find_m_l2297_229771

-- Definition of the odd function condition
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- The first proof problem
theorem find_a (a : ℝ) (h_odd : odd_function (fun x => Real.log (Real.exp x + a + 1))) : a = -1 :=
sorry

-- Definitions of the two functions involved in the second proof problem
noncomputable def f1 (x : ℝ) : ℝ :=
if x = 0 then 0 else Real.log x / x

noncomputable def f2 (x m : ℝ) : ℝ :=
x^2 - 2 * Real.exp 1 * x + m

-- The second proof problem
theorem find_m (m : ℝ) (h_root : ∃! x, f1 x = f2 x m) : m = Real.exp 2 + 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_find_a_find_m_l2297_229771


namespace NUMINAMATH_GPT_minimum_value_problem1_minimum_value_problem2_l2297_229779

theorem minimum_value_problem1 (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y >= 6 := 
sorry

theorem minimum_value_problem2 (x : ℝ) (h : x > 1) : 
  ∃ y, y = (x^2 + 8) / (x - 1) ∧ y >= 8 := 
sorry

end NUMINAMATH_GPT_minimum_value_problem1_minimum_value_problem2_l2297_229779


namespace NUMINAMATH_GPT_heloise_total_pets_l2297_229716

-- Define initial data
def ratio_dogs_to_cats := (10, 17)
def dogs_given_away := 10
def dogs_remaining := 60

-- Definition of initial number of dogs based on conditions
def initial_dogs := dogs_remaining + dogs_given_away

-- Definition based on ratio of dogs to cats
def dogs_per_set := ratio_dogs_to_cats.1
def cats_per_set := ratio_dogs_to_cats.2

-- Compute the number of sets of dogs
def sets_of_dogs := initial_dogs / dogs_per_set

-- Compute the number of cats
def initial_cats := sets_of_dogs * cats_per_set

-- Definition of the total number of pets
def total_pets := dogs_remaining + initial_cats

-- Lean statement for the proof
theorem heloise_total_pets :
  initial_dogs = 70 ∧
  sets_of_dogs = 7 ∧
  initial_cats = 119 ∧
  total_pets = 179 :=
by
  -- The statements to be proved are listed as conjunctions (∧)
  sorry

end NUMINAMATH_GPT_heloise_total_pets_l2297_229716


namespace NUMINAMATH_GPT_spicy_hot_noodles_plates_l2297_229766

theorem spicy_hot_noodles_plates (total_plates lobster_rolls seafood_noodles spicy_hot_noodles : ℕ) :
  total_plates = 55 →
  lobster_rolls = 25 →
  seafood_noodles = 16 →
  spicy_hot_noodles = total_plates - (lobster_rolls + seafood_noodles) →
  spicy_hot_noodles = 14 := by
  intros h_total h_lobster h_seafood h_eq
  rw [h_total, h_lobster, h_seafood] at h_eq
  exact h_eq

end NUMINAMATH_GPT_spicy_hot_noodles_plates_l2297_229766


namespace NUMINAMATH_GPT_greatest_possible_sum_of_visible_numbers_l2297_229741

theorem greatest_possible_sum_of_visible_numbers :
  ∀ (numbers : ℕ → ℕ) (Cubes : Fin 4 → ℤ), 
  (numbers 0 = 1) → (numbers 1 = 3) → (numbers 2 = 9) → (numbers 3 = 27) → (numbers 4 = 81) → (numbers 5 = 243) →
  (Cubes 0 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) → 
  (Cubes 1 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) →
  (Cubes 2 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 3 = 16 * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 0 + Cubes 1 + Cubes 2 + Cubes 3 = 1452) :=
by 
  sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_visible_numbers_l2297_229741


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2297_229722

theorem solution_set_of_inequality (x : ℝ) : 
  (1 / x ≤ 1 ↔ (0 < x ∧ x < 1) ∨ (1 ≤ x)) :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2297_229722


namespace NUMINAMATH_GPT_calculate_expression_l2297_229797

theorem calculate_expression :
  500 * 1986 * 0.3972 * 100 = 20 * 1986^2 :=
by sorry

end NUMINAMATH_GPT_calculate_expression_l2297_229797


namespace NUMINAMATH_GPT_sum_of_legs_of_larger_triangle_l2297_229760

theorem sum_of_legs_of_larger_triangle 
  (area_small area_large : ℝ)
  (hypotenuse_small : ℝ)
  (A : area_small = 10)
  (B : area_large = 250)
  (C : hypotenuse_small = 13) : 
  ∃ a b : ℝ, (a + b = 35) := 
sorry

end NUMINAMATH_GPT_sum_of_legs_of_larger_triangle_l2297_229760


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2297_229700

noncomputable def a : ℤ := 2

theorem point_in_fourth_quadrant (x y : ℤ) (h1 : x = a - 1) (h2 : y = a - 3) (h3 : x > 0) (h4 : y < 0) : a = 2 := by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2297_229700


namespace NUMINAMATH_GPT_norma_found_cards_l2297_229742

/-- Assume Norma originally had 88.0 cards. -/
def original_cards : ℝ := 88.0

/-- Assume Norma now has a total of 158 cards. -/
def total_cards : ℝ := 158

/-- Prove that Norma found 70 cards. -/
theorem norma_found_cards : total_cards - original_cards = 70 := 
by
  sorry

end NUMINAMATH_GPT_norma_found_cards_l2297_229742


namespace NUMINAMATH_GPT_expected_value_is_one_third_l2297_229733

noncomputable def expected_value_of_winnings : ℚ :=
  let p1 := (1/6 : ℚ)
  let p2 := (1/6 : ℚ)
  let p3 := (1/6 : ℚ)
  let p4 := (1/6 : ℚ)
  let p5 := (1/6 : ℚ)
  let p6 := (1/6 : ℚ)
  let winnings1 := (5 : ℚ)
  let winnings2 := (5 : ℚ)
  let winnings3 := (0 : ℚ)
  let winnings4 := (0 : ℚ)
  let winnings5 := (-4 : ℚ)
  let winnings6 := (-4 : ℚ)
  (p1 * winnings1 + p2 * winnings2 + p3 * winnings3 + p4 * winnings4 + p5 * winnings5 + p6 * winnings6)

theorem expected_value_is_one_third : expected_value_of_winnings = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_one_third_l2297_229733


namespace NUMINAMATH_GPT_find_number_l2297_229748

def condition (x : ℤ) : Prop := 3 * (x + 8) = 36

theorem find_number (x : ℤ) (h : condition x) : x = 4 := by
  sorry

end NUMINAMATH_GPT_find_number_l2297_229748


namespace NUMINAMATH_GPT_total_miles_driven_l2297_229769

-- Given constants and conditions
def city_mpg : ℝ := 30
def highway_mpg : ℝ := 37
def total_gallons : ℝ := 11
def highway_extra_miles : ℕ := 5

-- Variable for the number of city miles
variable (x : ℝ)

-- Conditions encapsulated in a theorem statement
theorem total_miles_driven:
  (x / city_mpg) + ((x + highway_extra_miles) / highway_mpg) = total_gallons →
  x + (x + highway_extra_miles) = 365 :=
by
  sorry

end NUMINAMATH_GPT_total_miles_driven_l2297_229769


namespace NUMINAMATH_GPT_people_per_bus_l2297_229791

def num_vans : ℝ := 6.0
def num_buses : ℝ := 8.0
def people_per_van : ℝ := 6.0
def extra_people : ℝ := 108.0

theorem people_per_bus :
  let people_vans := num_vans * people_per_van
  let people_buses := people_vans + extra_people
  let people_per_bus := people_buses / num_buses
  people_per_bus = 18.0 :=
by 
  sorry

end NUMINAMATH_GPT_people_per_bus_l2297_229791


namespace NUMINAMATH_GPT_icing_cubes_count_31_l2297_229786

def cake_cubed (n : ℕ) := n^3

noncomputable def slabs_with_icing (n : ℕ): ℕ := 
    let num_faces := 3
    let edge_per_face := n - 1
    let edges_with_icing := num_faces * edge_per_face * (n - 2)
    edges_with_icing + (n - 2) * 4 * (n - 2)

theorem icing_cubes_count_31 : ∀ (n : ℕ), n = 5 → slabs_with_icing n = 31 :=
by
  intros n hn
  revert hn
  sorry

end NUMINAMATH_GPT_icing_cubes_count_31_l2297_229786


namespace NUMINAMATH_GPT_ring_binder_price_l2297_229713

theorem ring_binder_price (x : ℝ) (h1 : 50 + 5 = 55) (h2 : ∀ x, 55 + 3 * (x - 2) = 109) :
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_ring_binder_price_l2297_229713


namespace NUMINAMATH_GPT_remainder_mod_7_l2297_229701

theorem remainder_mod_7 (n m p : ℕ) 
  (h₁ : n % 4 = 3)
  (h₂ : m % 7 = 5)
  (h₃ : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_mod_7_l2297_229701


namespace NUMINAMATH_GPT_ellipse_focal_length_l2297_229717

theorem ellipse_focal_length (k : ℝ) :
  (∀ x y : ℝ, x^2 / k + y^2 / 2 = 1) →
  (∃ c : ℝ, 2 * c = 2 ∧ (k = 1 ∨ k = 3)) :=
by
  -- Given condition: equation of ellipse and focal length  
  intro h  
  sorry

end NUMINAMATH_GPT_ellipse_focal_length_l2297_229717


namespace NUMINAMATH_GPT_problem_prime_square_plus_two_l2297_229795

theorem problem_prime_square_plus_two (P : ℕ) (hP_prime : Prime P) (hP2_plus_2_prime : Prime (P^2 + 2)) : P^4 + 1921 = 2002 :=
by
  sorry

end NUMINAMATH_GPT_problem_prime_square_plus_two_l2297_229795


namespace NUMINAMATH_GPT_inequality_solution_l2297_229763

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l2297_229763


namespace NUMINAMATH_GPT_proof_inequality_l2297_229799

theorem proof_inequality (p q r : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hpr_lt_qr : p * r < q * r) : 
  p < q :=
by 
  sorry

end NUMINAMATH_GPT_proof_inequality_l2297_229799


namespace NUMINAMATH_GPT_boys_camp_problem_l2297_229788

noncomputable def total_boys_in_camp : ℝ :=
  let schoolA_fraction := 0.20
  let science_fraction := 0.30
  let non_science_boys := 63
  let non_science_fraction := 1 - science_fraction
  let schoolA_boys := (non_science_boys / non_science_fraction)
  schoolA_boys / schoolA_fraction

theorem boys_camp_problem : total_boys_in_camp = 450 := 
by
  sorry

end NUMINAMATH_GPT_boys_camp_problem_l2297_229788


namespace NUMINAMATH_GPT_train_arrival_time_l2297_229774

-- Define the time type
structure Time where
  hour : Nat
  minute : Nat

namespace Time

-- Define the addition of minutes to a time.
def add_minutes (t : Time) (m : Nat) : Time :=
  let new_minutes := t.minute + m
  if new_minutes < 60 then 
    { hour := t.hour, minute := new_minutes }
  else 
    { hour := t.hour + new_minutes / 60, minute := new_minutes % 60 }

-- Define the departure time
def departure_time : Time := { hour := 9, minute := 45 }

-- Define the travel time in minutes
def travel_time : Nat := 15

-- Define the expected arrival time
def expected_arrival_time : Time := { hour := 10, minute := 0 }

-- The theorem we need to prove
theorem train_arrival_time:
  add_minutes departure_time travel_time = expected_arrival_time := by
  sorry

end NUMINAMATH_GPT_train_arrival_time_l2297_229774


namespace NUMINAMATH_GPT_max_marks_obtainable_l2297_229798

theorem max_marks_obtainable 
  (math_pass_percentage : ℝ := 36 / 100)
  (phys_pass_percentage : ℝ := 40 / 100)
  (chem_pass_percentage : ℝ := 45 / 100)
  (math_marks : ℕ := 130)
  (math_fail_margin : ℕ := 14)
  (phys_marks : ℕ := 120)
  (phys_fail_margin : ℕ := 20)
  (chem_marks : ℕ := 160)
  (chem_fail_margin : ℕ := 10) : 
  ∃ max_total_marks : ℤ, max_total_marks = 1127 := 
by 
  sorry  -- Proof not required

end NUMINAMATH_GPT_max_marks_obtainable_l2297_229798


namespace NUMINAMATH_GPT_cricket_team_right_handed_players_l2297_229772

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (non_throwers : ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_throwers : ℕ := throwers)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers)
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers = non_throwers / 3) :
  total_right_handed = 59 :=
by
  rw [h1, h2] at *
  -- The remaining parts of the proof here are omitted for brevity.
  sorry

end NUMINAMATH_GPT_cricket_team_right_handed_players_l2297_229772


namespace NUMINAMATH_GPT_evaluate_star_l2297_229776

-- Define the operation c star d
def star (c d : ℤ) : ℤ := c^2 - 2 * c * d + d^2

-- State the theorem to prove the given problem
theorem evaluate_star : (star 3 5) = 4 := by
  sorry

end NUMINAMATH_GPT_evaluate_star_l2297_229776


namespace NUMINAMATH_GPT_area_of_given_triangle_l2297_229783

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def vertex_A : ℝ × ℝ := (-1, 4)
def vertex_B : ℝ × ℝ := (7, 0)
def vertex_C : ℝ × ℝ := (11, 5)

theorem area_of_given_triangle : area_of_triangle vertex_A vertex_B vertex_C = 28 :=
by
  show 1 / 2 * |(-1) * (0 - 5) + 7 * (5 - 4) + 11 * (4 - 0)| = 28
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_l2297_229783


namespace NUMINAMATH_GPT_kite_area_correct_l2297_229723

-- Define the coordinates of the vertices
def vertex1 : (ℤ × ℤ) := (3, 0)
def vertex2 : (ℤ × ℤ) := (0, 5)
def vertex3 : (ℤ × ℤ) := (3, 7)
def vertex4 : (ℤ × ℤ) := (6, 5)

-- Define the area of a kite using the Shoelace formula for a quadrilateral
-- with given vertices
def kite_area (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))) / 2

theorem kite_area_correct : kite_area vertex1 vertex2 vertex3 vertex4 = 21 := 
  sorry

end NUMINAMATH_GPT_kite_area_correct_l2297_229723


namespace NUMINAMATH_GPT_smallest_n_l2297_229749

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ 2000 * n % 21 = 0 ∧ ∀ m : ℕ, m > 0 ∧ 2000 * m % 21 = 0 → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_n_l2297_229749


namespace NUMINAMATH_GPT_mouse_jump_distance_l2297_229702

theorem mouse_jump_distance
  (g : ℕ) 
  (f : ℕ) 
  (m : ℕ)
  (h1 : g = 25)
  (h2 : f = g + 32)
  (h3 : m = f - 26) : 
  m = 31 :=
by
  sorry

end NUMINAMATH_GPT_mouse_jump_distance_l2297_229702


namespace NUMINAMATH_GPT_distinct_painted_cubes_l2297_229792

-- Define the context of the problem
def num_faces : ℕ := 6

def total_paintings : ℕ := num_faces.factorial

def num_rotations : ℕ := 24

-- Statement of the theorem
theorem distinct_painted_cubes (h1 : total_paintings = 720) (h2 : num_rotations = 24) : 
  total_paintings / num_rotations = 30 := by
  sorry

end NUMINAMATH_GPT_distinct_painted_cubes_l2297_229792


namespace NUMINAMATH_GPT_range_of_m_l2297_229775

noncomputable def quadraticExpr (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + 4 * m * x + m + 3

theorem range_of_m :
  (∀ x : ℝ, quadraticExpr m x ≥ 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2297_229775


namespace NUMINAMATH_GPT_projection_of_vector_l2297_229736

open Real EuclideanSpace

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b

theorem projection_of_vector : 
  vector_projection (6, -3) (3, 0) = (6, 0) := 
by 
  sorry

end NUMINAMATH_GPT_projection_of_vector_l2297_229736


namespace NUMINAMATH_GPT_range_of_m_l2297_229724

theorem range_of_m (m : ℝ) :
  ¬(1^2 + 2*1 - m > 0) ∧ (2^2 + 2*2 - m > 0) ↔ (3 ≤ m ∧ m < 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2297_229724


namespace NUMINAMATH_GPT_depth_of_well_l2297_229718

theorem depth_of_well
  (d : ℝ)
  (h1 : ∃ t1 t2 : ℝ, 18 * t1^2 = d ∧ t2 = d / 1150 ∧ t1 + t2 = 8) :
  d = 33.18 :=
sorry

end NUMINAMATH_GPT_depth_of_well_l2297_229718


namespace NUMINAMATH_GPT_grid_segments_divisible_by_4_l2297_229728

-- Definition: square grid where each cell has a side length of 1
structure SquareGrid (n : ℕ) :=
  (segments : ℕ)

-- Condition: Function to calculate the total length of segments in the grid
def total_length {n : ℕ} (Q : SquareGrid n) : ℕ := Q.segments

-- Lean 4 statement: Prove that for any grid, the total length is divisible by 4
theorem grid_segments_divisible_by_4 {n : ℕ} (Q : SquareGrid n) :
  total_length Q % 4 = 0 :=
sorry

end NUMINAMATH_GPT_grid_segments_divisible_by_4_l2297_229728


namespace NUMINAMATH_GPT_second_chapter_pages_l2297_229710

theorem second_chapter_pages (x : ℕ) (h1 : 48 = x + 37) : x = 11 := 
sorry

end NUMINAMATH_GPT_second_chapter_pages_l2297_229710


namespace NUMINAMATH_GPT_larger_pie_crust_flour_l2297_229752

theorem larger_pie_crust_flour
  (p1 p2 : ℕ)
  (f1 f2 c : ℚ)
  (h1 : p1 = 36)
  (h2 : p2 = 24)
  (h3 : f1 = 1 / 8)
  (h4 : p1 * f1 = c)
  (h5 : p2 * f2 = c)
  : f2 = 3 / 16 :=
sorry

end NUMINAMATH_GPT_larger_pie_crust_flour_l2297_229752


namespace NUMINAMATH_GPT_min_value_of_expression_l2297_229747

theorem min_value_of_expression (x y : ℝ) : 
  ∃ m : ℝ, m = (xy - 1)^2 + (x + y)^2 ∧ (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ m) := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2297_229747


namespace NUMINAMATH_GPT_find_divisor_l2297_229708

theorem find_divisor :
  ∃ d : ℕ, (d = 859560) ∧ ∃ n : ℕ, (n + 859622) % d = 0 ∧ n = 859560 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l2297_229708


namespace NUMINAMATH_GPT_rational_equation_solutions_l2297_229764

open Real

theorem rational_equation_solutions :
  (∃ x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ ((x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0)) →
  ∃ S : Finset ℝ, S.card = 2 ∧ ∀ x ∈ S, (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_rational_equation_solutions_l2297_229764


namespace NUMINAMATH_GPT_find_unknown_rate_l2297_229757

variable {x : ℝ}

theorem find_unknown_rate (h : (3 * 100 + 1 * 150 + 2 * x) / 6 = 150) : x = 225 :=
by 
  sorry

end NUMINAMATH_GPT_find_unknown_rate_l2297_229757


namespace NUMINAMATH_GPT_prove_road_length_l2297_229790

-- Define variables for days taken by team A, B, and C
variables {a b c : ℕ}

-- Define the daily completion rates for teams A, B, and C
def rateA : ℕ := 300
def rateB : ℕ := 240
def rateC : ℕ := 180

-- Define the maximum length of the road
def max_length : ℕ := 3500

-- Define the total section of the road that team A completes in a days
def total_A (a : ℕ) : ℕ := a * rateA

-- Define the total section of the road that team B completes in b days and 18 hours
def total_B (a b : ℕ) : ℕ := 240 * (a + b) + 180

-- Define the total section of the road that team C completes in c days and 8 hours
def total_C (a b c : ℕ) : ℕ := 180 * (a + b + c) + 60

-- Define the constraint on the sum of days taken: a + b + c
def total_days (a b c : ℕ) : ℕ := a + b + c

-- The proof goal: Prove that (a * 300 == 3300) given the conditions
theorem prove_road_length :
  (total_A a = 3300) ∧ (total_B a b ≤ max_length) ∧ (total_C a b c ≤ max_length) ∧ (total_days a b c ≤ 19) :=
sorry

end NUMINAMATH_GPT_prove_road_length_l2297_229790


namespace NUMINAMATH_GPT_merchant_markup_percentage_l2297_229796

theorem merchant_markup_percentage
  (CP : ℕ) (discount_percent : ℚ) (profit_percent : ℚ)
  (mp : ℚ := CP + x)
  (sp : ℚ := (1 - discount_percent) * mp)
  (final_sp : ℚ := CP * (1 + profit_percent)) :
  discount_percent = 15 / 100 ∧ profit_percent = 19 / 100 ∧ CP = 100 → 
  sp = 85 + 0.85 * x → 
  final_sp = 119 →
  x = 40 :=
by 
  sorry

end NUMINAMATH_GPT_merchant_markup_percentage_l2297_229796


namespace NUMINAMATH_GPT_bulbs_arrangement_l2297_229793

theorem bulbs_arrangement :
  let blue_bulbs := 5
  let red_bulbs := 8
  let white_bulbs := 11
  let total_non_white_bulbs := blue_bulbs + red_bulbs
  let total_gaps := total_non_white_bulbs + 1
  (Nat.choose 13 5) * (Nat.choose total_gaps white_bulbs) = 468468 :=
by
  sorry

end NUMINAMATH_GPT_bulbs_arrangement_l2297_229793


namespace NUMINAMATH_GPT_arnold_danny_age_l2297_229750

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 13) : x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_arnold_danny_age_l2297_229750


namespace NUMINAMATH_GPT_students_neither_math_physics_l2297_229744

theorem students_neither_math_physics (total_students math_students physics_students both_students : ℕ) 
  (h1 : total_students = 120)
  (h2 : math_students = 80)
  (h3 : physics_students = 50)
  (h4 : both_students = 15) : 
  total_students - (math_students - both_students + physics_students - both_students + both_students) = 5 :=
by
  -- Each of the hypotheses are used exactly as given in the conditions.
  -- We omit the proof as requested.
  sorry

end NUMINAMATH_GPT_students_neither_math_physics_l2297_229744


namespace NUMINAMATH_GPT_find_g_neg6_l2297_229770

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_g_neg6_l2297_229770


namespace NUMINAMATH_GPT_handshakes_7_boys_l2297_229794

theorem handshakes_7_boys : Nat.choose 7 2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_7_boys_l2297_229794


namespace NUMINAMATH_GPT_find_q_l2297_229719

theorem find_q (a b m p q : ℚ) 
  (h1 : ∀ x, x^2 - m * x + 3 = (x - a) * (x - b)) 
  (h2 : a * b = 3) 
  (h3 : (x^2 - p * x + q) = (x - (a + 1/b)) * (x - (b + 1/a))) : 
  q = 16 / 3 := 
by sorry

end NUMINAMATH_GPT_find_q_l2297_229719


namespace NUMINAMATH_GPT_rectangle_semicircle_area_split_l2297_229754

open Real

/-- The main problem statement -/
theorem rectangle_semicircle_area_split 
  (A B D C N U T : ℝ)
  (AU_AN_UAlengths : AU = 84 ∧ AN = 126 ∧ UB = 168)
  (area_ratio : ∃ (ℓ : ℝ), ∃ (N U T : ℝ), 1 / 2 = area_differ / (area_left + area_right))
  (DA_calculation : DA = 63 * sqrt 6) :
  63 + 6 = 69
:=
sorry

end NUMINAMATH_GPT_rectangle_semicircle_area_split_l2297_229754


namespace NUMINAMATH_GPT_gcd_g50_g52_l2297_229789

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 2 * x^2 + x + 2023

-- Define the integers n1 and n2 corresponding to g(50) and g(52)
def n1 : ℤ := g 50
def n2 : ℤ := g 52

-- Statement of the proof goal
theorem gcd_g50_g52 : Int.gcd n1 n2 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_g50_g52_l2297_229789


namespace NUMINAMATH_GPT_productivity_increase_is_233_33_percent_l2297_229729

noncomputable def productivity_increase :
  Real :=
  let B := 1 -- represents the base number of bears made per week
  let H := 1 -- represents the base number of hours worked per week
  let P := B / H -- base productivity in bears per hour

  let B1 := 1.80 * B -- bears per week with first assistant
  let H1 := 0.90 * H -- hours per week with first assistant
  let P1 := B1 / H1 -- productivity with first assistant

  let B2 := 1.60 * B -- bears per week with second assistant
  let H2 := 0.80 * H -- hours per week with second assistant
  let P2 := B2 / H2 -- productivity with second assistant

  let B_both := B1 + B2 - B -- total bears with both assistants
  let H_both := H1 * H2 / H -- total hours with both assistants
  let P_both := B_both / H_both -- productivity with both assistants

  (P_both / P - 1) * 100

theorem productivity_increase_is_233_33_percent :
  productivity_increase = 233.33 :=
by
  sorry

end NUMINAMATH_GPT_productivity_increase_is_233_33_percent_l2297_229729


namespace NUMINAMATH_GPT_total_shirts_sold_l2297_229780

theorem total_shirts_sold (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 : ℕ) (h1 : p1 = 20) (h2 : p2 = 22) (h3 : p3 = 25)
(h4 : p4 + p5 + p6 + p7 + p8 + p9 + p10 = 133) (h5 : ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10) / 10) > 20)
: p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 = 200 ∧ 10 = 10 := sorry

end NUMINAMATH_GPT_total_shirts_sold_l2297_229780


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l2297_229782

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotonically_increasing_interval : 
  ∃ (a b : ℝ), a = -Real.pi / 3 ∧ b = Real.pi / 6 ∧ ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l2297_229782


namespace NUMINAMATH_GPT_problem_g3_1_l2297_229758

theorem problem_g3_1 (a : ℝ) : 
  (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = a ↔ a = 2005 := 
sorry

end NUMINAMATH_GPT_problem_g3_1_l2297_229758


namespace NUMINAMATH_GPT_hcf_36_84_l2297_229745

def highestCommonFactor (a b : ℕ) : ℕ := Nat.gcd a b

theorem hcf_36_84 : highestCommonFactor 36 84 = 12 := by
  sorry

end NUMINAMATH_GPT_hcf_36_84_l2297_229745


namespace NUMINAMATH_GPT_sequence_properties_l2297_229767

theorem sequence_properties
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 = 5)
  (h2 : ∀ n, n % 2 = 1 → a (n + 1) - a n = 1)
  (h3 : ∀ n, n % 2 = 0 → a (n + 1) - a n = 3) :
  (a 1 = 2) ∧ (a 2 = 3) ∧
  (∀ n, a (2 * n - 1) = 2 * (2 * n - 1)) ∧
  (∀ n, a (2 * n) = 2 * 2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l2297_229767


namespace NUMINAMATH_GPT_tub_volume_ratio_l2297_229761

theorem tub_volume_ratio (C D : ℝ) 
  (h₁ : 0 < C) 
  (h₂ : 0 < D)
  (h₃ : (3/4) * C = (2/3) * D) : 
  C / D = 8 / 9 := 
sorry

end NUMINAMATH_GPT_tub_volume_ratio_l2297_229761


namespace NUMINAMATH_GPT_major_axis_length_l2297_229726

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship given in the problem
def major_axis_ratio : ℝ := 1.6

-- Define the calculation for minor axis
def minor_axis : ℝ := 2 * cylinder_radius

-- Define the calculation for major axis
def major_axis : ℝ := major_axis_ratio * minor_axis

-- The theorem statement
theorem major_axis_length:
  major_axis = 6.4 :=
by 
  sorry -- Proof to be provided later

end NUMINAMATH_GPT_major_axis_length_l2297_229726


namespace NUMINAMATH_GPT_sum_of_three_largest_l2297_229714

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_largest_l2297_229714


namespace NUMINAMATH_GPT_greatest_possible_price_per_notebook_l2297_229731

theorem greatest_possible_price_per_notebook (budget entrance_fee : ℝ) (notebooks : ℕ) (tax_rate : ℝ) (price_per_notebook : ℝ) :
  budget = 160 ∧ entrance_fee = 5 ∧ notebooks = 18 ∧ tax_rate = 0.05 ∧ price_per_notebook * notebooks * (1 + tax_rate) ≤ (budget - entrance_fee) →
  price_per_notebook = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_price_per_notebook_l2297_229731


namespace NUMINAMATH_GPT_train_crosses_pole_in_l2297_229738

noncomputable def train_crossing_time (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (5.0 / 18.0)
  length / speed_m_s

theorem train_crosses_pole_in : train_crossing_time 175 180 = 3.5 :=
by
  -- Proof would be here, but for now, it is omitted.
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_l2297_229738


namespace NUMINAMATH_GPT_f_of_x_l2297_229715

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end NUMINAMATH_GPT_f_of_x_l2297_229715


namespace NUMINAMATH_GPT_nomogram_relation_l2297_229705

noncomputable def root_of_eq (x p q : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem nomogram_relation (x p q : ℝ) (hx : root_of_eq x p q) : 
  q = -x * p - x^2 :=
by 
  sorry

end NUMINAMATH_GPT_nomogram_relation_l2297_229705


namespace NUMINAMATH_GPT_hikers_rate_l2297_229778

-- Define the conditions from the problem
variables (R : ℝ) (time_up time_down : ℝ) (distance_down : ℝ)

-- Conditions given in the problem
axiom condition1 : time_up = 2
axiom condition2 : time_down = 2
axiom condition3 : distance_down = 9
axiom condition4 : (distance_down / time_down) = 1.5 * R

-- The proof goal
theorem hikers_rate (h1 : time_up = 2) 
                    (h2 : time_down = 2) 
                    (h3 : distance_down = 9) 
                    (h4 : distance_down / time_down = 1.5 * R) : R = 3 := 
by 
  sorry

end NUMINAMATH_GPT_hikers_rate_l2297_229778


namespace NUMINAMATH_GPT_exists_k_in_octahedron_l2297_229755

theorem exists_k_in_octahedron
  (x0 y0 z0 : ℚ)
  (h : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ 
                 x0 + y0 - z0 ≠ n ∧ 
                 x0 - y0 + z0 ≠ n ∧ 
                 x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, ∃ (xk yk zk : ℚ), 
    k ≠ 0 ∧ 
    xk = k * x0 ∧ 
    yk = k * y0 ∧ 
    zk = k * z0 ∧
    ∀ n : ℤ, 
      (xk + yk + zk < ↑n → xk + yk + zk > ↑(n - 1)) ∧ 
      (xk + yk - zk < ↑n → xk + yk - zk > ↑(n - 1)) ∧ 
      (xk - yk + zk < ↑n → xk - yk + zk > ↑(n - 1)) ∧ 
      (xk - yk - zk < ↑n → xk - yk - zk > ↑(n - 1)) :=
sorry

end NUMINAMATH_GPT_exists_k_in_octahedron_l2297_229755


namespace NUMINAMATH_GPT_DogHeight_is_24_l2297_229781

-- Define the given conditions as Lean definitions (variables and equations)
variable (CarterHeight DogHeight BettyHeight : ℝ)

-- Assume the conditions given in the problem
axiom h1 : CarterHeight = 2 * DogHeight
axiom h2 : BettyHeight + 12 = CarterHeight
axiom h3 : BettyHeight = 36

-- State the proposition (the height of Carter's dog)
theorem DogHeight_is_24 : DogHeight = 24 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_DogHeight_is_24_l2297_229781


namespace NUMINAMATH_GPT_factorize_expression_l2297_229725

variable {a b : ℕ}

theorem factorize_expression (h : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1)) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l2297_229725


namespace NUMINAMATH_GPT_initial_bales_l2297_229751

theorem initial_bales (bales_initially bales_added bales_now : ℕ)
  (h₀ : bales_added = 26)
  (h₁ : bales_now = 54)
  (h₂ : bales_now = bales_initially + bales_added) :
  bales_initially = 28 :=
by
  sorry

end NUMINAMATH_GPT_initial_bales_l2297_229751


namespace NUMINAMATH_GPT_hcf_of_two_numbers_of_given_conditions_l2297_229707

theorem hcf_of_two_numbers_of_given_conditions :
  ∃ B H, (588 = H * 84) ∧ H = Nat.gcd 588 B ∧ H = 7 :=
by
  use 84, 7
  have h₁ : 588 = 7 * 84 := by sorry
  have h₂ : 7 = Nat.gcd 588 84 := by sorry
  exact ⟨h₁, h₂, rfl⟩

end NUMINAMATH_GPT_hcf_of_two_numbers_of_given_conditions_l2297_229707


namespace NUMINAMATH_GPT_systematic_sampling_removal_count_l2297_229721

-- Define the conditions
def total_population : Nat := 1252
def sample_size : Nat := 50

-- Define the remainder after division
def remainder := total_population % sample_size

-- Proof statement
theorem systematic_sampling_removal_count :
  remainder = 2 := by
    sorry

end NUMINAMATH_GPT_systematic_sampling_removal_count_l2297_229721


namespace NUMINAMATH_GPT_liked_both_desserts_l2297_229765

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end NUMINAMATH_GPT_liked_both_desserts_l2297_229765


namespace NUMINAMATH_GPT_compare_fractions_l2297_229777

variable {a b c d : ℝ}

theorem compare_fractions (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  (b / (a - c)) < (a / (b - d)) := 
by
  sorry

end NUMINAMATH_GPT_compare_fractions_l2297_229777


namespace NUMINAMATH_GPT_binary_to_decimal_l2297_229732

theorem binary_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5) :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l2297_229732


namespace NUMINAMATH_GPT_find_r_cubed_l2297_229762

theorem find_r_cubed (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_find_r_cubed_l2297_229762


namespace NUMINAMATH_GPT_part_1_solution_part_2_solution_l2297_229784

def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem part_1_solution (x : ℝ) : f x < 3 ↔ -4 / 3 < x ∧ x < 0 :=
by
  sorry

theorem part_2_solution (a : ℝ) : (∀ x, ¬ (f x < a)) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_part_1_solution_part_2_solution_l2297_229784


namespace NUMINAMATH_GPT_no_two_or_more_consecutive_sum_30_l2297_229709

theorem no_two_or_more_consecutive_sum_30 :
  ∀ (a n : ℕ), n ≥ 2 → (n * (2 * a + n - 1) = 60) → false :=
by
  intro a n hn h
  sorry

end NUMINAMATH_GPT_no_two_or_more_consecutive_sum_30_l2297_229709


namespace NUMINAMATH_GPT_cos_identity_example_l2297_229720

theorem cos_identity_example (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 3 / 5) : Real.cos (Real.pi / 3 - α) = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_cos_identity_example_l2297_229720


namespace NUMINAMATH_GPT_total_frogs_in_pond_l2297_229756

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_frogs_in_pond_l2297_229756
