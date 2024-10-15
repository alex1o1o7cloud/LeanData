import Mathlib

namespace NUMINAMATH_GPT_rancher_cows_l1394_139438

theorem rancher_cows (H C : ℕ) (h1 : C = 5 * H) (h2 : C + H = 168) : C = 140 := by
  sorry

end NUMINAMATH_GPT_rancher_cows_l1394_139438


namespace NUMINAMATH_GPT_snowfall_on_friday_l1394_139445

def snowstorm (snow_wednesday snow_thursday total_snow : ℝ) : ℝ :=
  total_snow - (snow_wednesday + snow_thursday)

theorem snowfall_on_friday :
  snowstorm 0.33 0.33 0.89 = 0.23 := 
by
  -- (Conditions)
  -- snow_wednesday = 0.33
  -- snow_thursday = 0.33
  -- total_snow = 0.89
  -- (Conclusion) snowstorm 0.33 0.33 0.89 = 0.23
  sorry

end NUMINAMATH_GPT_snowfall_on_friday_l1394_139445


namespace NUMINAMATH_GPT_trip_attendees_trip_cost_savings_l1394_139437

theorem trip_attendees (total_people : ℕ) (total_cost : ℕ) (adult_ticket : ℕ) 
(student_discount : ℕ) (group_discount : ℕ) (adults : ℕ) (students : ℕ) :
total_people = 130 → total_cost = 9600 → adult_ticket = 120 →
student_discount = 50 → group_discount = 40 → 
total_people = adults + students → 
total_cost = adults * adult_ticket + students * (adult_ticket * student_discount / 100) →
adults = 30 ∧ students = 100 :=
by sorry

theorem trip_cost_savings (total_people : ℕ) (individual_total_cost : ℕ) 
(group_total_cost : ℕ) (student_tickets : ℕ) (group_tickets : ℕ) 
(adult_ticket : ℕ) (student_discount : ℕ) (group_discount : ℕ) :
(total_people = 130) → (individual_total_cost = 7200 + 1800) → 
(group_total_cost = total_people * (adult_ticket * group_discount / 100)) →
(adult_ticket = 120) → (student_discount = 50) → (group_discount = 40) → 
(total_people = student_tickets + group_tickets) → (student_tickets = 30) → 
(group_tickets = 100) → (7200 + 1800 < 9360) → 
student_tickets = 30 ∧ group_tickets = 100 :=
by sorry

end NUMINAMATH_GPT_trip_attendees_trip_cost_savings_l1394_139437


namespace NUMINAMATH_GPT_percentage_blue_and_red_l1394_139472

theorem percentage_blue_and_red (F : ℕ) (h_even: F % 2 = 0)
  (h1: ∃ C, 50 / 100 * C = F / 2)
  (h2: ∃ C, 60 / 100 * C = F / 2)
  (h3: ∃ C, 40 / 100 * C = F / 2) :
  ∃ C, (50 / 100 * C + 60 / 100 * C - 100 / 100 * C) = 10 / 100 * C :=
sorry

end NUMINAMATH_GPT_percentage_blue_and_red_l1394_139472


namespace NUMINAMATH_GPT_packs_of_chocolate_l1394_139423

theorem packs_of_chocolate (t c k x : ℕ) (ht : t = 42) (hc : c = 4) (hk : k = 22) (hx : x = t - (c + k)) : x = 16 :=
by
  rw [ht, hc, hk] at hx
  simp at hx
  exact hx

end NUMINAMATH_GPT_packs_of_chocolate_l1394_139423


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1394_139475

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1394_139475


namespace NUMINAMATH_GPT_find_p_l1394_139489

theorem find_p (x y : ℝ) (h : y = 1.15 * x * (1 - p / 100)) : p = 15 :=
sorry

end NUMINAMATH_GPT_find_p_l1394_139489


namespace NUMINAMATH_GPT_find_constant_t_l1394_139426

theorem find_constant_t : ∃ t : ℝ, 
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (2 * x^2 + t * x + 8) = 6 * x^4 + (-26) * x^3 + 58 * x^2 + (-76) * x + 40) ↔ t = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_constant_t_l1394_139426


namespace NUMINAMATH_GPT_total_cups_l1394_139422

variable (eggs : ℕ) (flour : ℕ)
variable (h : eggs = 60) (h1 : flour = eggs / 2)

theorem total_cups (eggs : ℕ) (flour : ℕ) (h : eggs = 60) (h1 : flour = eggs / 2) : 
  eggs + flour = 90 := 
by
  sorry

end NUMINAMATH_GPT_total_cups_l1394_139422


namespace NUMINAMATH_GPT_quadratic_equation_m_l1394_139456

theorem quadratic_equation_m (m : ℝ) (h1 : |m| + 1 = 2) (h2 : m + 1 ≠ 0) : m = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_m_l1394_139456


namespace NUMINAMATH_GPT_bottles_left_l1394_139486

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end NUMINAMATH_GPT_bottles_left_l1394_139486


namespace NUMINAMATH_GPT_problem_statement_l1394_139474

open Complex

theorem problem_statement :
  (3 - I) / (2 + I) = 1 - I :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1394_139474


namespace NUMINAMATH_GPT_polynomial_coefficients_l1394_139453

theorem polynomial_coefficients
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
  (h : (x-3)^8 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3 + 
                a_4 * (x-2)^4 + a_5 * (x-2)^5 + a_6 * (x-2)^6 + 
                a_7 * (x-2)^7 + a_8 * (x-2)^8) :
  (a_0 = 1) ∧ 
  (a_1 / 2 + a_2 / 2^2 + a_3 / 2^3 + a_4 / 2^4 + a_5 / 2^5 + 
   a_6 / 2^6 + a_7 / 2^7 + a_8 / 2^8 = -255 / 256) ∧ 
  (a_0 + a_2 + a_4 + a_6 + a_8 = 128) :=
by sorry

end NUMINAMATH_GPT_polynomial_coefficients_l1394_139453


namespace NUMINAMATH_GPT_line_passes_through_point_l1394_139419

-- We declare the variables for the real numbers a, b, and c
variables (a b c : ℝ)

-- We state the condition that a + b - c = 0
def condition1 : Prop := a + b - c = 0

-- We state the condition that not all of a, b, c are zero
def condition2 : Prop := ¬ (a = 0 ∧ b = 0 ∧ c = 0)

-- We state the theorem: the line ax + by + c = 0 passes through the point (-1, -1)
theorem line_passes_through_point (h1 : condition1 a b c) (h2 : condition2 a b c) :
  a * (-1) + b * (-1) + c = 0 := sorry

end NUMINAMATH_GPT_line_passes_through_point_l1394_139419


namespace NUMINAMATH_GPT_find_m_l1394_139420

theorem find_m (m : ℝ) (a a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (1 + m)^6 = a + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 = 63)
  (h3 : a = 1) : m = 1 ∨ m = -3 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1394_139420


namespace NUMINAMATH_GPT_range_of_m_l1394_139463

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m+1)*x^2 + (m+1)*x + (m+2) ≥ 0) ↔ m ≥ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1394_139463


namespace NUMINAMATH_GPT_dodecahedron_interior_diagonals_count_l1394_139497

-- Define a dodecahedron structure
structure Dodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (faces_per_vertex : ℕ)

-- Define the property of a dodecahedron
def dodecahedron_property : Dodecahedron :=
{
  vertices := 20,
  edges_per_vertex := 3,
  faces_per_vertex := 3
}

-- The theorem statement
theorem dodecahedron_interior_diagonals_count (d : Dodecahedron)
  (h1 : d.vertices = 20)
  (h2 : d.edges_per_vertex = 3)
  (h3 : d.faces_per_vertex = 3) : 
  (d.vertices * (d.vertices - d.edges_per_vertex)) / 2 = 160 :=
by
  sorry

end NUMINAMATH_GPT_dodecahedron_interior_diagonals_count_l1394_139497


namespace NUMINAMATH_GPT_overlapping_rectangles_perimeter_l1394_139493

namespace RectangleOverlappingPerimeter

def length := 7
def width := 3

/-- Prove that the perimeter of the shape formed by overlapping two rectangles,
    each measuring 7 cm by 3 cm, is 28 cm. -/
theorem overlapping_rectangles_perimeter : 
  let total_perimeter := 2 * (length + (2 * width))
  total_perimeter = 28 :=
by
  sorry

end RectangleOverlappingPerimeter

end NUMINAMATH_GPT_overlapping_rectangles_perimeter_l1394_139493


namespace NUMINAMATH_GPT_polar_to_cartesian_l1394_139402

theorem polar_to_cartesian (p θ : ℝ) (x y : ℝ) (hp : p = 8 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 + y^2 = 8 * x := 
sorry

end NUMINAMATH_GPT_polar_to_cartesian_l1394_139402


namespace NUMINAMATH_GPT_total_cost_two_rackets_l1394_139462

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end NUMINAMATH_GPT_total_cost_two_rackets_l1394_139462


namespace NUMINAMATH_GPT_puddle_base_area_l1394_139427

theorem puddle_base_area (rate depth hours : ℝ) (A : ℝ) 
  (h1 : rate = 10) 
  (h2 : depth = 30) 
  (h3 : hours = 3) 
  (h4 : depth * A = rate * hours) : 
  A = 1 := 
by 
  sorry

end NUMINAMATH_GPT_puddle_base_area_l1394_139427


namespace NUMINAMATH_GPT_chandra_monsters_l1394_139410

def monsters_day_1 : Nat := 2
def monsters_day_2 : Nat := monsters_day_1 * 3
def monsters_day_3 : Nat := monsters_day_2 * 4
def monsters_day_4 : Nat := monsters_day_3 * 5
def monsters_day_5 : Nat := monsters_day_4 * 6

def total_monsters : Nat := monsters_day_1 + monsters_day_2 + monsters_day_3 + monsters_day_4 + monsters_day_5

theorem chandra_monsters : total_monsters = 872 :=
by
  unfold total_monsters
  unfold monsters_day_1
  unfold monsters_day_2
  unfold monsters_day_3
  unfold monsters_day_4
  unfold monsters_day_5
  sorry

end NUMINAMATH_GPT_chandra_monsters_l1394_139410


namespace NUMINAMATH_GPT_john_paid_correct_amount_l1394_139498

theorem john_paid_correct_amount : 
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  john_share = 8400 :=
by
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  show john_share = 8400
  sorry

end NUMINAMATH_GPT_john_paid_correct_amount_l1394_139498


namespace NUMINAMATH_GPT_linear_system_solution_l1394_139446

theorem linear_system_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 7) (h2 : 6 * x - 5 * y = 4) :
  x = 43 / 27 ∧ y = 10 / 9 :=
sorry

end NUMINAMATH_GPT_linear_system_solution_l1394_139446


namespace NUMINAMATH_GPT_first_statement_second_statement_difference_between_statements_l1394_139439

variable (A B C : Prop)

-- First statement: (A ∨ B) → C
theorem first_statement : (A ∨ B) → C :=
sorry

-- Second statement: (A ∧ B) → C
theorem second_statement : (A ∧ B) → C :=
sorry

-- Proof that shows the difference between the two statements
theorem difference_between_statements :
  ((A ∨ B) → C) ↔ ¬((A ∧ B) → C) :=
sorry

end NUMINAMATH_GPT_first_statement_second_statement_difference_between_statements_l1394_139439


namespace NUMINAMATH_GPT_wise_men_correct_guesses_l1394_139461

noncomputable def max_correct_guesses (n k : ℕ) : ℕ :=
  if n > k + 1 then n - k - 1 else 0

theorem wise_men_correct_guesses (n k : ℕ) :
  ∃ (m : ℕ), m = max_correct_guesses n k ∧ m ≤ n - k - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_wise_men_correct_guesses_l1394_139461


namespace NUMINAMATH_GPT_euler_disproof_l1394_139417

theorem euler_disproof :
  ∃ (n : ℕ), 0 < n ∧ (133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144) :=
by
  sorry

end NUMINAMATH_GPT_euler_disproof_l1394_139417


namespace NUMINAMATH_GPT_time_to_cover_length_l1394_139479

-- Definitions from conditions
def escalator_speed : Real := 15 -- ft/sec
def escalator_length : Real := 180 -- feet
def person_speed : Real := 3 -- ft/sec

-- Combined speed definition
def combined_speed : Real := escalator_speed + person_speed

-- Lean theorem statement proving the time taken
theorem time_to_cover_length : escalator_length / combined_speed = 10 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l1394_139479


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1394_139411

-- Define sets M and N
def M : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℤ := {0, 1, 2}

-- The theorem to be proven: M ∩ N = {0, 1, 2}
theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1394_139411


namespace NUMINAMATH_GPT_kids_played_on_tuesday_l1394_139492

-- Definitions of the conditions
def kids_played_on_wednesday (julia : Type) : Nat := 4
def kids_played_on_monday (julia : Type) : Nat := 6
def difference_monday_wednesday (julia : Type) : Nat := 2

-- Define the statement to prove
theorem kids_played_on_tuesday (julia : Type) :
  (kids_played_on_monday julia - difference_monday_wednesday julia) = kids_played_on_wednesday julia :=
by
  sorry

end NUMINAMATH_GPT_kids_played_on_tuesday_l1394_139492


namespace NUMINAMATH_GPT_freezer_temperature_is_minus_12_l1394_139442

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end NUMINAMATH_GPT_freezer_temperature_is_minus_12_l1394_139442


namespace NUMINAMATH_GPT_integer_condition_l1394_139407

theorem integer_condition (p : ℕ) (h : p > 0) : 
  (∃ n : ℤ, (3 * (p: ℤ) + 25) = n * (2 * (p: ℤ) - 5)) ↔ (3 ≤ p ∧ p ≤ 35) :=
sorry

end NUMINAMATH_GPT_integer_condition_l1394_139407


namespace NUMINAMATH_GPT_sqrt_six_lt_a_lt_cubic_two_l1394_139496

theorem sqrt_six_lt_a_lt_cubic_two (a : ℝ) (h : a^5 - a^3 + a = 2) : (Real.sqrt 3)^6 < a ∧ a < 2^(1/3) :=
sorry

end NUMINAMATH_GPT_sqrt_six_lt_a_lt_cubic_two_l1394_139496


namespace NUMINAMATH_GPT_store_owner_loss_percentage_l1394_139405

theorem store_owner_loss_percentage :
  ∀ (initial_value : ℝ) (profit_margin : ℝ) (loss1 : ℝ) (loss2 : ℝ) (loss3 : ℝ) (tax_rate : ℝ),
    initial_value = 100 → profit_margin = 0.10 → loss1 = 0.20 → loss2 = 0.30 → loss3 = 0.25 → tax_rate = 0.12 →
      ((initial_value - initial_value * (1 - loss1) * (1 - loss2) * (1 - loss3)) / initial_value * 100) = 58 :=
by
  intros initial_value profit_margin loss1 loss2 loss3 tax_rate h_initial_value h_profit_margin h_loss1 h_loss2 h_loss3 h_tax_rate
  -- Variable assignments as per given conditions
  have h1 : initial_value = 100 := h_initial_value
  have h2 : profit_margin = 0.10 := h_profit_margin
  have h3 : loss1 = 0.20 := h_loss1
  have h4 : loss2 = 0.30 := h_loss2
  have h5 : loss3 = 0.25 := h_loss3
  have h6 : tax_rate = 0.12 := h_tax_rate
  
  sorry

end NUMINAMATH_GPT_store_owner_loss_percentage_l1394_139405


namespace NUMINAMATH_GPT_snow_at_Brecknock_l1394_139477

theorem snow_at_Brecknock (hilt_snow brecknock_snow : ℕ) (h1 : hilt_snow = 29) (h2 : hilt_snow = brecknock_snow + 12) : brecknock_snow = 17 :=
by
  sorry

end NUMINAMATH_GPT_snow_at_Brecknock_l1394_139477


namespace NUMINAMATH_GPT_nate_age_when_ember_is_14_l1394_139400

theorem nate_age_when_ember_is_14
  (nate_age : ℕ)
  (ember_age : ℕ)
  (h_half_age : ember_age = nate_age / 2)
  (h_nate_current_age : nate_age = 14) :
  nate_age + (14 - ember_age) = 21 :=
by
  sorry

end NUMINAMATH_GPT_nate_age_when_ember_is_14_l1394_139400


namespace NUMINAMATH_GPT_triangle_area_l1394_139432

-- Defining the rectangle dimensions
def length : ℝ := 35
def width : ℝ := 48

-- Defining the area of the right triangle formed by the diagonal of the rectangle
theorem triangle_area : (1 / 2) * length * width = 840 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1394_139432


namespace NUMINAMATH_GPT_john_reams_needed_l1394_139454

theorem john_reams_needed 
  (pages_flash_fiction_weekly : ℕ := 20) 
  (pages_short_story_weekly : ℕ := 50) 
  (pages_novel_annual : ℕ := 1500) 
  (weeks_in_year : ℕ := 52) 
  (sheets_per_ream : ℕ := 500) 
  (sheets_flash_fiction_weekly : ℕ := 10)
  (sheets_short_story_weekly : ℕ := 25) :
  let sheets_flash_fiction_annual := sheets_flash_fiction_weekly * weeks_in_year
  let sheets_short_story_annual := sheets_short_story_weekly * weeks_in_year
  let total_sheets_annual := sheets_flash_fiction_annual + sheets_short_story_annual + pages_novel_annual
  let reams_needed := (total_sheets_annual + sheets_per_ream - 1) / sheets_per_ream
  reams_needed = 7 := 
by sorry

end NUMINAMATH_GPT_john_reams_needed_l1394_139454


namespace NUMINAMATH_GPT_supplemental_tank_time_l1394_139481

-- Define the given conditions as assumptions
def primary_tank_time : Nat := 2
def total_time_needed : Nat := 8
def supplemental_tanks : Nat := 6
def additional_time_needed : Nat := total_time_needed - primary_tank_time

-- Define the theorem to prove
theorem supplemental_tank_time :
  additional_time_needed / supplemental_tanks = 1 :=
by
  -- Here we would provide the proof, but it is omitted with "sorry"
  sorry

end NUMINAMATH_GPT_supplemental_tank_time_l1394_139481


namespace NUMINAMATH_GPT_binary_predecessor_l1394_139425

theorem binary_predecessor (N : ℕ) (hN : N = 0b11000) : 0b10111 + 1 = N := 
by
  sorry

end NUMINAMATH_GPT_binary_predecessor_l1394_139425


namespace NUMINAMATH_GPT_sum_first_12_terms_l1394_139478

variable (S : ℕ → ℝ)

def sum_of_first_n_terms (n : ℕ) : ℝ :=
  S n

theorem sum_first_12_terms (h₁ : sum_of_first_n_terms 4 = 30) (h₂ : sum_of_first_n_terms 8 = 100) :
  sum_of_first_n_terms 12 = 210 := 
sorry

end NUMINAMATH_GPT_sum_first_12_terms_l1394_139478


namespace NUMINAMATH_GPT_haley_seeds_in_big_garden_l1394_139416

def seeds_in_big_garden (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem haley_seeds_in_big_garden :
  let total_seeds := 56
  let small_gardens := 7
  let seeds_per_small_garden := 3
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 :=
by
  sorry

end NUMINAMATH_GPT_haley_seeds_in_big_garden_l1394_139416


namespace NUMINAMATH_GPT_find_b_plus_m_l1394_139415

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x + 1) / Real.log a + b 

variable (a b m : ℝ)
-- Conditions
axiom h1 : a > 0
axiom h2 : a ≠ 1
axiom h3 : f a b m = 3

theorem find_b_plus_m : b + m = 3 :=
sorry

end NUMINAMATH_GPT_find_b_plus_m_l1394_139415


namespace NUMINAMATH_GPT_smallest_b_for_factoring_l1394_139428

theorem smallest_b_for_factoring :
  ∃ b : ℕ, b > 0 ∧
    (∀ r s : ℤ, r * s = 2016 → r + s ≠ b) ∧
    (∀ r s : ℤ, r * s = 2016 → r + s = b → b = 92) :=
sorry

end NUMINAMATH_GPT_smallest_b_for_factoring_l1394_139428


namespace NUMINAMATH_GPT_fourth_hexagon_dots_l1394_139440

def dots_in_hexagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 1 + (12 * (n * (n + 1) / 2))

theorem fourth_hexagon_dots : dots_in_hexagon 4 = 85 :=
by
  unfold dots_in_hexagon
  norm_num
  sorry

end NUMINAMATH_GPT_fourth_hexagon_dots_l1394_139440


namespace NUMINAMATH_GPT_one_elephant_lake_empty_in_365_days_l1394_139485

variables (C K V : ℝ)
variables (t : ℝ)

noncomputable def lake_empty_one_day (C K V : ℝ) := 183 * C = V + K
noncomputable def lake_empty_five_days (C K V : ℝ) := 185 * C = V + 5 * K

noncomputable def elephant_time (C K V t : ℝ) : Prop :=
  (t * C = V + t * K) → (t = 365)

theorem one_elephant_lake_empty_in_365_days (C K V t : ℝ) :
  (lake_empty_one_day C K V) →
  (lake_empty_five_days C K V) →
  (elephant_time C K V t) := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_one_elephant_lake_empty_in_365_days_l1394_139485


namespace NUMINAMATH_GPT_fisher_eligibility_l1394_139447

theorem fisher_eligibility (A1 A2 S : ℕ) (hA1 : A1 = 84) (hS : S = 82) :
  (S ≥ 80) → (A1 + A2 ≥ 170) → (A2 = 86) :=
by
  sorry

end NUMINAMATH_GPT_fisher_eligibility_l1394_139447


namespace NUMINAMATH_GPT_number_of_possible_lists_l1394_139412

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end NUMINAMATH_GPT_number_of_possible_lists_l1394_139412


namespace NUMINAMATH_GPT_inheritance_amount_l1394_139476

theorem inheritance_amount (x : ℝ) 
  (h1 : x * 0.25 + (x - x * 0.25) * 0.12 = 13600) : x = 40000 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_inheritance_amount_l1394_139476


namespace NUMINAMATH_GPT_log_base_change_l1394_139435

theorem log_base_change (log_16_32 log_16_inv2: ℝ) : 
  (log_16_32 * log_16_inv2 = -5 / 16) :=
by
  sorry

end NUMINAMATH_GPT_log_base_change_l1394_139435


namespace NUMINAMATH_GPT_divisibility_of_a81_l1394_139443

theorem divisibility_of_a81 
  (p : ℕ) (hp : Nat.Prime p) (hp_gt2 : 2 < p)
  (a : ℕ → ℕ) (h_rec : ∀ n, n * a (n + 1) = (n + 1) * a n - (p / 2)^4) 
  (h_a1 : a 1 = 5) :
  16 ∣ a 81 := 
sorry

end NUMINAMATH_GPT_divisibility_of_a81_l1394_139443


namespace NUMINAMATH_GPT_hyperbola_condition_l1394_139433

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) ↔ ∀ x y : ℝ, (x^2 / (k - 1)) + (y^2 / (k + 2)) = 1 → 
  (k - 1 < 0 ∧ k + 2 > 0 ∨ k - 1 > 0 ∧ k + 2 < 0) := 
sorry

end NUMINAMATH_GPT_hyperbola_condition_l1394_139433


namespace NUMINAMATH_GPT_two_point_questions_l1394_139458

theorem two_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_two_point_questions_l1394_139458


namespace NUMINAMATH_GPT_complement_union_is_correct_l1394_139451

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_is_correct : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_is_correct_l1394_139451


namespace NUMINAMATH_GPT_partA_partB_partC_l1394_139464
noncomputable section

def n : ℕ := 100
def p : ℝ := 0.8
def q : ℝ := 1 - p

def binomial_prob (k1 k2 : ℕ) : ℝ := sorry

theorem partA : binomial_prob 70 85 = 0.8882 := sorry
theorem partB : binomial_prob 70 100 = 0.9938 := sorry
theorem partC : binomial_prob 0 69 = 0.0062 := sorry

end NUMINAMATH_GPT_partA_partB_partC_l1394_139464


namespace NUMINAMATH_GPT_car_division_ways_l1394_139488

/-- 
Prove that the number of ways to divide 6 people 
into two different cars, with each car holding 
a maximum of 4 people, is equal to 50. 
-/
theorem car_division_ways : 
  (∃ s1 s2 : Finset ℕ, s1.card = 2 ∧ s2.card = 4) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 3 ∧ s2.card = 3) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 4 ∧ s2.card = 2) →
  (15 + 20 + 15 = 50) := 
by 
  sorry

end NUMINAMATH_GPT_car_division_ways_l1394_139488


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1394_139473

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (c : ℝ) (h3 : a^2 + b^2 = c^2) 
  (h4 : ∃ M : ℝ × ℝ, (M.fst^2 / a^2 - M.snd^2 / b^2 = 1) ∧ (M.snd^2 = 8 * M.fst)
    ∧ (|M.fst - 2| + |M.snd| = 5)) : 
  (c / a = 2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1394_139473


namespace NUMINAMATH_GPT_bob_friends_l1394_139491

-- Define the total price and the amount paid by each person
def total_price : ℕ := 40
def amount_per_person : ℕ := 8

-- Define the total number of people who paid
def total_people : ℕ := total_price / amount_per_person

-- Define Bob's presence and require proving the number of his friends
theorem bob_friends (total_price amount_per_person total_people : ℕ) (h1 : total_price = 40)
  (h2 : amount_per_person = 8) (h3 : total_people = total_price / amount_per_person) : 
  total_people - 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_bob_friends_l1394_139491


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l1394_139449

theorem convert_to_scientific_notation (H : 1 = 10^9) : 
  3600 * (10 : ℝ)^9 = 3.6 * (10 : ℝ)^12 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l1394_139449


namespace NUMINAMATH_GPT_symmetric_graph_inverse_l1394_139482

def f (x : ℝ) : ℝ := sorry -- We assume f is defined accordingly somewhere, as the inverse of ln.

theorem symmetric_graph_inverse (h : ∀ x, f (f x) = x) : f 2 = Real.exp 2 := by
  sorry

end NUMINAMATH_GPT_symmetric_graph_inverse_l1394_139482


namespace NUMINAMATH_GPT_gumballs_per_package_correct_l1394_139470

-- Define the conditions
def total_gumballs_eaten : ℕ := 20
def number_of_boxes_finished : ℕ := 4

-- Define the target number of gumballs in each package
def gumballs_in_each_package := 5

theorem gumballs_per_package_correct :
  total_gumballs_eaten / number_of_boxes_finished = gumballs_in_each_package :=
by
  sorry

end NUMINAMATH_GPT_gumballs_per_package_correct_l1394_139470


namespace NUMINAMATH_GPT_lower_right_is_one_l1394_139403

def initial_grid : Matrix (Fin 5) (Fin 5) (Option (Fin 5)) :=
![![some 0, none, some 1, none, none],
  ![some 1, some 3, none, none, none],
  ![none, none, none, some 4, none],
  ![none, some 4, none, none, none],
  ![none, none, none, none, none]]

theorem lower_right_is_one 
  (complete_grid : Matrix (Fin 5) (Fin 5) (Fin 5)) 
  (unique_row_col : ∀ i j k, 
      complete_grid i j = complete_grid i k ↔ j = k ∧ 
      complete_grid i j = complete_grid k j ↔ i = k)
  (matches_partial : ∀ i j, ∃ x, 
      initial_grid i j = some x → complete_grid i j = x) :
  complete_grid 4 4 = 0 := 
sorry

end NUMINAMATH_GPT_lower_right_is_one_l1394_139403


namespace NUMINAMATH_GPT_custom_operation_correct_l1394_139413

def custom_operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem custom_operation_correct : custom_operation 6 3 = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_custom_operation_correct_l1394_139413


namespace NUMINAMATH_GPT_lauren_total_earnings_l1394_139471

-- Define earnings conditions
def mondayCommercialEarnings (views : ℕ) : ℝ := views * 0.40
def mondaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 0.80

def tuesdayCommercialEarnings (views : ℕ) : ℝ := views * 0.50
def tuesdaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 1.00

def weekendMerchandiseEarnings (sales : ℝ) : ℝ := 0.10 * sales

-- Specific conditions for each day
def mondayTotalEarnings : ℝ := mondayCommercialEarnings 80 + mondaySubscriptionEarnings 20
def tuesdayTotalEarnings : ℝ := tuesdayCommercialEarnings 100 + tuesdaySubscriptionEarnings 27
def weekendTotalEarnings : ℝ := weekendMerchandiseEarnings 150

-- Total earnings for the period
def totalEarnings : ℝ := mondayTotalEarnings + tuesdayTotalEarnings + weekendTotalEarnings

-- Examining the final value
theorem lauren_total_earnings : totalEarnings = 140.00 := by
  sorry

end NUMINAMATH_GPT_lauren_total_earnings_l1394_139471


namespace NUMINAMATH_GPT_f_minus_ten_l1394_139490

noncomputable def f : ℝ → ℝ := sorry

theorem f_minus_ten :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  (f 1 = 2) →
  f (-10) = 90 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_f_minus_ten_l1394_139490


namespace NUMINAMATH_GPT_percentage_relations_with_respect_to_z_l1394_139429

variable (x y z w : ℝ)
variable (h1 : x = 1.30 * y)
variable (h2 : y = 0.50 * z)
variable (h3 : w = 2 * x)

theorem percentage_relations_with_respect_to_z : 
  x = 0.65 * z ∧ y = 0.50 * z ∧ w = 1.30 * z := by
  sorry

end NUMINAMATH_GPT_percentage_relations_with_respect_to_z_l1394_139429


namespace NUMINAMATH_GPT_Ryanne_is_7_years_older_than_Hezekiah_l1394_139414

theorem Ryanne_is_7_years_older_than_Hezekiah
  (H : ℕ) (R : ℕ)
  (h1 : H = 4)
  (h2 : R + H = 15) :
  R - H = 7 := by
  sorry

end NUMINAMATH_GPT_Ryanne_is_7_years_older_than_Hezekiah_l1394_139414


namespace NUMINAMATH_GPT_quality_of_algorithm_reflects_number_of_operations_l1394_139418

-- Definitions
def speed_of_operation_is_important (c : Type) : Prop :=
  ∀ (c1 : c), true

-- Theorem stating that the number of operations within a unit of time is an important sign of the quality of an algorithm
theorem quality_of_algorithm_reflects_number_of_operations {c : Type} 
    (h_speed_important : speed_of_operation_is_important c) : 
  ∀ (a : Type) (q : a), true := 
sorry

end NUMINAMATH_GPT_quality_of_algorithm_reflects_number_of_operations_l1394_139418


namespace NUMINAMATH_GPT_second_discarded_number_l1394_139465

theorem second_discarded_number (S : ℝ) (X : ℝ) :
  (S = 50 * 44) →
  ((S - 45 - X) / 48 = 43.75) →
  X = 55 :=
by
  intros h1 h2
  -- The proof steps would go here, but we leave it unproved
  sorry

end NUMINAMATH_GPT_second_discarded_number_l1394_139465


namespace NUMINAMATH_GPT_sum_mod_17_l1394_139494

theorem sum_mod_17 :
  (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) % 17 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_17_l1394_139494


namespace NUMINAMATH_GPT_fourth_number_ninth_row_eq_206_l1394_139495

-- Define the first number in a given row
def first_number_in_row (i : Nat) : Nat :=
  2 + 4 * 6 * (i - 1)

-- Define the number in the j-th position in the i-th row
def number_in_row (i j : Nat) : Nat :=
  first_number_in_row i + 4 * (j - 1)

-- Define the 9th row and fourth number in it
def fourth_number_ninth_row : Nat :=
  number_in_row 9 4

-- The theorem to prove the fourth number in the 9th row is 206
theorem fourth_number_ninth_row_eq_206 : fourth_number_ninth_row = 206 := by
  sorry

end NUMINAMATH_GPT_fourth_number_ninth_row_eq_206_l1394_139495


namespace NUMINAMATH_GPT_problem_statement_l1394_139460

noncomputable def f (x : ℝ) : ℝ := 2 / (2019^x + 1) + Real.sin x

noncomputable def f' (x : ℝ) := (deriv f) x

theorem problem_statement :
  f 2018 + f (-2018) + f' 2019 - f' (-2019) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1394_139460


namespace NUMINAMATH_GPT_smallest_integer_solution_l1394_139401

theorem smallest_integer_solution : ∀ x : ℤ, (x < 2 * x - 7) → (8 = x) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_l1394_139401


namespace NUMINAMATH_GPT_decreasing_power_function_has_specific_m_l1394_139421

theorem decreasing_power_function_has_specific_m (m : ℝ) (x : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → 
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_power_function_has_specific_m_l1394_139421


namespace NUMINAMATH_GPT_equilibrium_temperature_l1394_139468

theorem equilibrium_temperature 
  (c_B : ℝ) (c_m : ℝ)
  (m_B : ℝ) (m_m : ℝ)
  (T₁ : ℝ) (T_eq₁ : ℝ) (T_metal : ℝ) 
  (T_eq₂ : ℝ)
  (h₁ : T₁ = 80)
  (h₂ : T_eq₁ = 60)
  (h₃ : T_metal = 20)
  (h₄ : T₂ = 50)
  (h_ratio : c_B * m_B = 2 * c_m * m_m) :
  T_eq₂ = 50 :=
by
  sorry

end NUMINAMATH_GPT_equilibrium_temperature_l1394_139468


namespace NUMINAMATH_GPT_fraction_problem_l1394_139431

theorem fraction_problem (b : ℕ) (h₀ : 0 < b) (h₁ : (b : ℝ) / (b + 35) = 0.869) : b = 232 := 
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l1394_139431


namespace NUMINAMATH_GPT_poly_has_int_solution_iff_l1394_139436

theorem poly_has_int_solution_iff (a : ℤ) : 
  (a > 0 ∧ (∃ x : ℤ, a * x^2 + 2 * (2 * a - 1) * x + 4 * a - 7 = 0)) ↔ (a = 1 ∨ a = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_poly_has_int_solution_iff_l1394_139436


namespace NUMINAMATH_GPT_base7_number_divisibility_l1394_139408

theorem base7_number_divisibility (x : ℕ) (h : 0 ≤ x ∧ x ≤ 6) :
  (5 * 343 + 2 * 49 + x * 7 + 4) % 29 = 0 ↔ x = 6 := 
by
  sorry

end NUMINAMATH_GPT_base7_number_divisibility_l1394_139408


namespace NUMINAMATH_GPT_cost_of_new_shoes_l1394_139434

theorem cost_of_new_shoes 
    (R : ℝ) 
    (L_r : ℝ) 
    (L_n : ℝ) 
    (increase_percent : ℝ) 
    (H_R : R = 13.50) 
    (H_L_r : L_r = 1) 
    (H_L_n : L_n = 2) 
    (H_inc_percent : increase_percent = 0.1852) : 
    2 * (R * (1 + increase_percent) / L_n) = 32.0004 := 
by
    sorry

end NUMINAMATH_GPT_cost_of_new_shoes_l1394_139434


namespace NUMINAMATH_GPT_total_onions_grown_l1394_139459

theorem total_onions_grown :
  let onions_per_day_nancy := 3
  let days_worked_nancy := 4
  let onions_per_day_dan := 4
  let days_worked_dan := 6
  let onions_per_day_mike := 5
  let days_worked_mike := 5
  let onions_per_day_sasha := 6
  let days_worked_sasha := 4
  let onions_per_day_becky := 2
  let days_worked_becky := 6

  let total_onions_nancy := onions_per_day_nancy * days_worked_nancy
  let total_onions_dan := onions_per_day_dan * days_worked_dan
  let total_onions_mike := onions_per_day_mike * days_worked_mike
  let total_onions_sasha := onions_per_day_sasha * days_worked_sasha
  let total_onions_becky := onions_per_day_becky * days_worked_becky

  let total_onions := total_onions_nancy + total_onions_dan + total_onions_mike + total_onions_sasha + total_onions_becky

  total_onions = 97 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_onions_grown_l1394_139459


namespace NUMINAMATH_GPT_abs_expr_evaluation_l1394_139467

theorem abs_expr_evaluation : abs (abs (-abs (-1 + 2) - 2) + 3) = 6 := by
  sorry

end NUMINAMATH_GPT_abs_expr_evaluation_l1394_139467


namespace NUMINAMATH_GPT_cube_inequality_l1394_139452

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_inequality_l1394_139452


namespace NUMINAMATH_GPT_speed_of_train_is_correct_l1394_139484

-- Given conditions
def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_to_cross_bridge : ℝ := 20

-- Derived definition
def total_distance : ℝ := length_of_train + length_of_bridge

-- Goal to be proved
theorem speed_of_train_is_correct : total_distance / time_to_cross_bridge = 18.5 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_train_is_correct_l1394_139484


namespace NUMINAMATH_GPT_simplify_expression_l1394_139457

theorem simplify_expression : 
  (81 ^ (1 / Real.logb 5 9) + 3 ^ (3 / Real.logb (Real.sqrt 6) 3)) / 409 * 
  ((Real.sqrt 7) ^ (2 / Real.logb 25 7) - 125 ^ (Real.logb 25 6)) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1394_139457


namespace NUMINAMATH_GPT_sum_of_powers_divisible_by_6_l1394_139480

theorem sum_of_powers_divisible_by_6 (a1 a2 a3 a4 : ℤ)
  (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) (k : ℕ) (hk : k % 2 = 1) :
  6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
sorry

end NUMINAMATH_GPT_sum_of_powers_divisible_by_6_l1394_139480


namespace NUMINAMATH_GPT_angus_tokens_l1394_139487

theorem angus_tokens (x : ℕ) (h1 : x = 60 - (25 / 100) * 60) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_angus_tokens_l1394_139487


namespace NUMINAMATH_GPT_sports_lottery_systematic_sampling_l1394_139450

-- Definition of the sports lottery condition
def is_first_prize_ticket (n : ℕ) : Prop := n % 1000 = 345

-- Statement of the proof problem
theorem sports_lottery_systematic_sampling :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → is_first_prize_ticket n) →
  ∃ interval, (∀ segment_start : ℕ,  segment_start < 1000 → is_first_prize_ticket (segment_start + interval * 999))
  := by sorry

end NUMINAMATH_GPT_sports_lottery_systematic_sampling_l1394_139450


namespace NUMINAMATH_GPT_remainder_is_83_l1394_139466

-- Define the condition: the values for the division
def value1 : ℤ := 2021
def value2 : ℤ := 102

-- State the theorem: remainder when 2021 is divided by 102 is 83
theorem remainder_is_83 : value1 % value2 = 83 := by
  sorry

end NUMINAMATH_GPT_remainder_is_83_l1394_139466


namespace NUMINAMATH_GPT_perfect_square_expression_l1394_139499

theorem perfect_square_expression (x y z : ℤ) :
    9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3 * x * y * z) =
      ((x + y + z)^2 - 6 * (x * y + y * z + z * x))^2 := 
by 
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l1394_139499


namespace NUMINAMATH_GPT_part_1_part_2_l1394_139430

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - m * x
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + 1 - m
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - x - a * x^3
noncomputable def r (x : ℝ) : ℝ := (Real.log x - 1) / x^2
noncomputable def r' (x : ℝ) : ℝ := (3 - 2 * Real.log x) / x^3

theorem part_1 (x : ℝ) (m : ℝ) (h1 : f x m = -1) (h2 : f' x m = 0) :
  m = 1 ∧ (∀ y, y > 0 → y < x → f' y 1 < 0) ∧ (∀ y, y > x → f' y 1 > 0) :=
sorry

theorem part_2 (a : ℝ) :
  (a > 1 / (2 * Real.exp 3) → ∀ x, h x a ≠ 0) ∧
  (a ≤ 0 ∨ a = 1 / (2 * Real.exp 3) → ∃ x, h x a = 0 ∧ ∀ y, h y a = 0 → y = x) ∧
  (0 < a ∧ a < 1 / (2 * Real.exp 3) → ∃ x1 x2, x1 ≠ x2 ∧ h x1 a = 0 ∧ h x2 a = 0) :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l1394_139430


namespace NUMINAMATH_GPT_beads_problem_l1394_139441

theorem beads_problem :
  ∃ b : ℕ, (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ (b = 179) :=
by
  sorry

end NUMINAMATH_GPT_beads_problem_l1394_139441


namespace NUMINAMATH_GPT_find_a_plus_b_l1394_139424

theorem find_a_plus_b (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + a * x + b) 
  (h2 : { x : ℝ | 0 ≤ f x ∧ f x ≤ 6 - x } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } ∪ {6}) 
  : a + b = 9 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1394_139424


namespace NUMINAMATH_GPT_max_value_x_l1394_139406

theorem max_value_x : ∃ x, x ^ 2 = 38 ∧ x = Real.sqrt 38 := by
  sorry

end NUMINAMATH_GPT_max_value_x_l1394_139406


namespace NUMINAMATH_GPT_sum_consecutive_equals_prime_l1394_139448

theorem sum_consecutive_equals_prime (m k p : ℕ) (h_prime : Nat.Prime p) :
  (∃ S, S = (m * (2 * k + m - 1)) / 2 ∧ S = p) →
  m = 1 ∨ m = 2 :=
sorry

end NUMINAMATH_GPT_sum_consecutive_equals_prime_l1394_139448


namespace NUMINAMATH_GPT_max_value_of_PQ_l1394_139483

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 12)
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 12)

theorem max_value_of_PQ (t : ℝ) : abs (f t - g t) ≤ 2 :=
by sorry

end NUMINAMATH_GPT_max_value_of_PQ_l1394_139483


namespace NUMINAMATH_GPT_part_a_l1394_139455

theorem part_a (n : ℕ) (h_condition : n < 135) : ∃ r, r = 239 % n ∧ r ≤ 119 := 
sorry

end NUMINAMATH_GPT_part_a_l1394_139455


namespace NUMINAMATH_GPT_club_president_vice_president_combinations_144_l1394_139469

variables (boys_total girls_total : Nat)
variables (senior_boys junior_boys senior_girls junior_girls : Nat)
variables (choose_president_vice_president : Nat)

-- Define the conditions
def club_conditions : Prop :=
  boys_total = 12 ∧
  girls_total = 12 ∧
  senior_boys = 6 ∧
  junior_boys = 6 ∧
  senior_girls = 6 ∧
  junior_girls = 6

-- Define the proof problem
def president_vice_president_combinations (boys_total girls_total senior_boys junior_boys senior_girls junior_girls : Nat) : Nat :=
  2 * senior_boys * junior_boys + 2 * senior_girls * junior_girls

-- The main theorem to prove
theorem club_president_vice_president_combinations_144 :
  club_conditions boys_total girls_total senior_boys junior_boys senior_girls junior_girls →
  president_vice_president_combinations boys_total girls_total senior_boys junior_boys senior_girls junior_girls = 144 :=
sorry

end NUMINAMATH_GPT_club_president_vice_president_combinations_144_l1394_139469


namespace NUMINAMATH_GPT_jovana_added_pounds_l1394_139409

noncomputable def initial_amount : ℕ := 5
noncomputable def final_amount : ℕ := 28

theorem jovana_added_pounds : final_amount - initial_amount = 23 := by
  sorry

end NUMINAMATH_GPT_jovana_added_pounds_l1394_139409


namespace NUMINAMATH_GPT_sum_of_two_rel_prime_numbers_l1394_139444

theorem sum_of_two_rel_prime_numbers (k : ℕ) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 ∧ k = a + b) ↔ (k = 5 ∨ k ≥ 7) := sorry

end NUMINAMATH_GPT_sum_of_two_rel_prime_numbers_l1394_139444


namespace NUMINAMATH_GPT_product_eq_5832_l1394_139404

-- Define the integers A, B, C, D that satisfy the given conditions.
variables (A B C D : ℕ)

-- Define the conditions in the problem.
def conditions : Prop :=
  (A + B + C + D = 48) ∧
  (A + 3 = B - 3) ∧
  (A + 3 = C * 3) ∧
  (A + 3 = D / 3)

-- State the final theorem we want to prove.
theorem product_eq_5832 : conditions A B C D → A * B * C * D = 5832 :=
by 
  sorry

end NUMINAMATH_GPT_product_eq_5832_l1394_139404
