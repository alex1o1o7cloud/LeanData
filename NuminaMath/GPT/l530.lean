import Mathlib

namespace NUMINAMATH_GPT_maximize_take_home_pay_l530_53002

def tax_collected (x : ℝ) : ℝ :=
  10 * x^2

def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - tax_collected x

theorem maximize_take_home_pay : ∃ x : ℝ, (x * 1000 = 50000) ∧ (∀ y : ℝ, take_home_pay x ≥ take_home_pay y) := 
sorry

end NUMINAMATH_GPT_maximize_take_home_pay_l530_53002


namespace NUMINAMATH_GPT_clark_family_ticket_cost_l530_53070

theorem clark_family_ticket_cost
  (regular_price children's_price seniors_price : ℝ)
  (number_youngest_gen number_second_youngest_gen number_second_oldest_gen number_oldest_gen : ℕ)
  (h_senior_discount : seniors_price = 0.7 * regular_price)
  (h_senior_ticket_cost : seniors_price = 7)
  (h_child_discount : children's_price = 0.6 * regular_price)
  (h_number_youngest_gen : number_youngest_gen = 3)
  (h_number_second_youngest_gen : number_second_youngest_gen = 1)
  (h_number_second_oldest_gen : number_second_oldest_gen = 2)
  (h_number_oldest_gen : number_oldest_gen = 1)
  : 3 * children's_price + 1 * regular_price + 2 * seniors_price + 1 * regular_price = 52 := by
  sorry

end NUMINAMATH_GPT_clark_family_ticket_cost_l530_53070


namespace NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l530_53084

theorem no_integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), a^3 + 3 * a^2 + 2 * a ≠ 125 * b^3 + 75 * b^2 + 15 * b + 2 :=
by
  intro a b
  sorry

end NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l530_53084


namespace NUMINAMATH_GPT_integer_values_in_interval_l530_53071

theorem integer_values_in_interval : (∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, abs x < 4 * π ↔ -12 ≤ x ∧ x ≤ 12) :=
by
  sorry

end NUMINAMATH_GPT_integer_values_in_interval_l530_53071


namespace NUMINAMATH_GPT_slower_ball_speed_l530_53010

open Real

variables (v u C : ℝ)

theorem slower_ball_speed :
  (20 * (v - u) = C) → (4 * (v + u) = C) → ((v + u) * 3 = 75) → u = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_slower_ball_speed_l530_53010


namespace NUMINAMATH_GPT_find_simple_interest_sum_l530_53035

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

noncomputable def simple_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * r * n / 100

theorem find_simple_interest_sum (P CIsum : ℝ)
  (simple_rate : ℝ) (simple_years : ℕ)
  (compound_rate : ℝ) (compound_years : ℕ)
  (compound_principal : ℝ)
  (hP : simple_interest P simple_rate simple_years = CIsum)
  (hCI : CIsum = (compound_interest compound_principal compound_rate compound_years - compound_principal) / 2) :
  P = 1272 :=
by
  sorry

end NUMINAMATH_GPT_find_simple_interest_sum_l530_53035


namespace NUMINAMATH_GPT_total_chickens_l530_53082

-- Definitions from conditions
def ducks : ℕ := 40
def rabbits : ℕ := 30
def hens : ℕ := ducks + 20
def roosters : ℕ := rabbits - 10

-- Theorem statement: total number of chickens
theorem total_chickens : hens + roosters = 80 := 
sorry

end NUMINAMATH_GPT_total_chickens_l530_53082


namespace NUMINAMATH_GPT_fraction_of_repeating_decimal_l530_53021

theorem fraction_of_repeating_decimal:
  let a := (4 / 10 : ℝ)
  let r := (1 / 10 : ℝ)
  (∑' n:ℕ, a * r^n) = (4 / 9 : ℝ) := by
  sorry

end NUMINAMATH_GPT_fraction_of_repeating_decimal_l530_53021


namespace NUMINAMATH_GPT_seating_arrangement_l530_53031

theorem seating_arrangement (x y : ℕ) (h1 : x * 8 + y * 7 = 55) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l530_53031


namespace NUMINAMATH_GPT_rate_of_interest_is_5_percent_l530_53085

-- Defining the conditions as constants
def simple_interest : ℝ := 4016.25
def principal : ℝ := 16065
def time_period : ℝ := 5

-- Proving that the rate of interest is 5%
theorem rate_of_interest_is_5_percent (R : ℝ) : 
  simple_interest = (principal * R * time_period) / 100 → 
  R = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rate_of_interest_is_5_percent_l530_53085


namespace NUMINAMATH_GPT_cookies_per_pack_l530_53044

theorem cookies_per_pack
  (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ)
  (h1 : trays = 8) (h2 : cookies_per_tray = 36) (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 :=
by
  sorry

end NUMINAMATH_GPT_cookies_per_pack_l530_53044


namespace NUMINAMATH_GPT_round_robin_tournament_l530_53086

theorem round_robin_tournament (n k : ℕ) (h : (n-2) * (n-3) = 2 * 3^k): n = 5 :=
sorry

end NUMINAMATH_GPT_round_robin_tournament_l530_53086


namespace NUMINAMATH_GPT_jovana_added_shells_l530_53068

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h1 : initial_amount = 5) 
  (h2 : final_amount = 17) 
  (h3 : added_amount = final_amount - initial_amount) : 
  added_amount = 12 := 
by 
  -- Since the proof is not required, we add sorry here to skip the proof.
  sorry 

end NUMINAMATH_GPT_jovana_added_shells_l530_53068


namespace NUMINAMATH_GPT_toy_store_bears_shelves_l530_53067

theorem toy_store_bears_shelves (initial_stock shipment bears_per_shelf total_bears number_of_shelves : ℕ)
  (h1 : initial_stock = 17)
  (h2 : shipment = 10)
  (h3 : bears_per_shelf = 9)
  (h4 : total_bears = initial_stock + shipment)
  (h5 : number_of_shelves = total_bears / bears_per_shelf) :
  number_of_shelves = 3 :=
by
  sorry

end NUMINAMATH_GPT_toy_store_bears_shelves_l530_53067


namespace NUMINAMATH_GPT_find_x0_and_m_l530_53017

theorem find_x0_and_m (x : ℝ) (m : ℝ) (x0 : ℝ) :
  (abs (x + 3) - 2 * x - 1 < 0 ↔ x > 2) ∧ 
  (∃ x, abs (x - m) + abs (x + 1 / m) - 2 = 0) → 
  (x0 = 2 ∧ m = 1) := 
by
  sorry

end NUMINAMATH_GPT_find_x0_and_m_l530_53017


namespace NUMINAMATH_GPT_cost_of_paints_is_5_l530_53000

-- Define folders due to 6 classes
def folder_cost_per_item := 6
def num_classes := 6
def total_folder_cost : ℕ := folder_cost_per_item * num_classes

-- Define pencils due to the 6 classes and need per class
def pencil_cost_per_item := 2
def pencil_per_class := 3
def total_pencils : ℕ := pencil_per_class * num_classes
def total_pencil_cost : ℕ := pencil_cost_per_item * total_pencils

-- Define erasers needed based on pencils and their cost
def eraser_cost_per_item := 1
def pencils_per_eraser := 6
def total_erasers : ℕ := total_pencils / pencils_per_eraser
def total_eraser_cost : ℕ := eraser_cost_per_item * total_erasers

-- Total cost spent on folders, pencils, and erasers
def total_spent : ℕ := 80
def total_cost_supplies : ℕ := total_folder_cost + total_pencil_cost + total_eraser_cost

-- Cost of paints is the remaining amount when total cost is subtracted from total spent
def cost_of_paints : ℕ := total_spent - total_cost_supplies

-- The goal is to prove the cost of paints
theorem cost_of_paints_is_5 : cost_of_paints = 5 := by
  sorry

end NUMINAMATH_GPT_cost_of_paints_is_5_l530_53000


namespace NUMINAMATH_GPT_rosa_bonheur_birth_day_l530_53028

/--
Given that Rosa Bonheur's 210th birthday was celebrated on a Wednesday,
prove that she was born on a Sunday.
-/
theorem rosa_bonheur_birth_day :
  let anniversary_year := 2022
  let birth_year := 1812
  let total_years := anniversary_year - birth_year
  let leap_years := (total_years / 4) - (total_years / 100) + (total_years / 400)
  let regular_years := total_years - leap_years
  let day_shifts := regular_years + 2 * leap_years
  (3 - day_shifts % 7) % 7 = 0 := 
sorry

end NUMINAMATH_GPT_rosa_bonheur_birth_day_l530_53028


namespace NUMINAMATH_GPT_repeating_decimal_base_l530_53034

theorem repeating_decimal_base (k : ℕ) (h_pos : 0 < k) (h_repr : (9 : ℚ) / 61 = (3 * k + 4) / (k^2 - 1)) : k = 21 :=
  sorry

end NUMINAMATH_GPT_repeating_decimal_base_l530_53034


namespace NUMINAMATH_GPT_gcd_80_180_450_l530_53093

theorem gcd_80_180_450 : Int.gcd (Int.gcd 80 180) 450 = 10 := by
  sorry

end NUMINAMATH_GPT_gcd_80_180_450_l530_53093


namespace NUMINAMATH_GPT_carlos_fraction_l530_53030

theorem carlos_fraction (f : ℝ) :
  (1 - f) ^ 4 * 64 = 4 → f = 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_carlos_fraction_l530_53030


namespace NUMINAMATH_GPT_remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l530_53087

section Doughnuts

variable (initial_glazed : Nat := 10)
variable (initial_chocolate : Nat := 8)
variable (initial_raspberry : Nat := 6)

variable (personA_glazed : Nat := 2)
variable (personA_chocolate : Nat := 1)
variable (personB_glazed : Nat := 1)
variable (personC_chocolate : Nat := 3)
variable (personD_glazed : Nat := 1)
variable (personD_raspberry : Nat := 1)
variable (personE_raspberry : Nat := 1)
variable (personF_raspberry : Nat := 2)

def remaining_glazed : Nat :=
  initial_glazed - (personA_glazed + personB_glazed + personD_glazed)

def remaining_chocolate : Nat :=
  initial_chocolate - (personA_chocolate + personC_chocolate)

def remaining_raspberry : Nat :=
  initial_raspberry - (personD_raspberry + personE_raspberry + personF_raspberry)

theorem remaining_glazed_correct :
  remaining_glazed initial_glazed personA_glazed personB_glazed personD_glazed = 6 :=
by
  sorry

theorem remaining_chocolate_correct :
  remaining_chocolate initial_chocolate personA_chocolate personC_chocolate = 4 :=
by
  sorry

theorem remaining_raspberry_correct :
  remaining_raspberry initial_raspberry personD_raspberry personE_raspberry personF_raspberry = 2 :=
by
  sorry

end Doughnuts

end NUMINAMATH_GPT_remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l530_53087


namespace NUMINAMATH_GPT_determine_N_l530_53032

theorem determine_N (N : ℕ) : 995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end NUMINAMATH_GPT_determine_N_l530_53032


namespace NUMINAMATH_GPT_number_of_matches_in_first_set_l530_53078

theorem number_of_matches_in_first_set
  (x : ℕ)
  (h1 : (30 : ℚ) * x + 15 * 10 = 25 * (x + 10)) :
  x = 20 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_number_of_matches_in_first_set_l530_53078


namespace NUMINAMATH_GPT_supermarket_sold_54_pints_l530_53008

theorem supermarket_sold_54_pints (x s : ℝ) 
  (h1 : x * s = 216)
  (h2 : x * (s + 2) = 324) : 
  x = 54 := 
by 
  sorry

end NUMINAMATH_GPT_supermarket_sold_54_pints_l530_53008


namespace NUMINAMATH_GPT_statues_added_in_third_year_l530_53049

/-
Definition of the turtle statues problem:

1. Initially, there are 4 statues in the first year.
2. In the second year, the number of statues quadruples.
3. In the third year, x statues are added, and then 3 statues are broken.
4. In the fourth year, 2 * 3 new statues are added.
5. In total, at the end of the fourth year, there are 31 statues.
-/

def year1_statues : ℕ := 4
def year2_statues : ℕ := 4 * year1_statues
def before_hailstorm_year3_statues (x : ℕ) : ℕ := year2_statues + x
def after_hailstorm_year3_statues (x : ℕ) : ℕ := before_hailstorm_year3_statues x - 3
def total_year4_statues (x : ℕ) : ℕ := after_hailstorm_year3_statues x + 2 * 3

theorem statues_added_in_third_year (x : ℕ) (h : total_year4_statues x = 31) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_statues_added_in_third_year_l530_53049


namespace NUMINAMATH_GPT_letters_containing_only_dot_l530_53020

theorem letters_containing_only_dot (DS S_only : ℕ) (total : ℕ) (h1 : DS = 20) (h2 : S_only = 36) (h3 : total = 60) :
  total - (DS + S_only) = 4 :=
by
  sorry

end NUMINAMATH_GPT_letters_containing_only_dot_l530_53020


namespace NUMINAMATH_GPT_hourly_wage_12_5_l530_53024

theorem hourly_wage_12_5 
  (H : ℝ)
  (work_hours : ℝ := 40)
  (widgets_per_week : ℝ := 1000)
  (widget_earnings_per_widget : ℝ := 0.16)
  (total_earnings : ℝ := 660) :
  (40 * H + 1000 * 0.16 = 660) → (H = 12.5) :=
by
  sorry

end NUMINAMATH_GPT_hourly_wage_12_5_l530_53024


namespace NUMINAMATH_GPT_spinner_final_direction_l530_53048

-- Define the directions as an enumeration
inductive Direction
| north
| east
| south
| west

-- Convert between revolution fractions to direction
def direction_after_revolutions (initial : Direction) (revolutions : ℚ) : Direction :=
  let quarters := (revolutions * 4) % 4
  match initial with
  | Direction.south => if quarters == 0 then Direction.south
                       else if quarters == 1 then Direction.west
                       else if quarters == 2 then Direction.north
                       else Direction.east
  | Direction.east  => if quarters == 0 then Direction.east
                       else if quarters == 1 then Direction.south
                       else if quarters == 2 then Direction.west
                       else Direction.north
  | Direction.north => if quarters == 0 then Direction.north
                       else if quarters == 1 then Direction.east
                       else if quarters == 2 then Direction.south
                       else Direction.west
  | Direction.west  => if quarters == 0 then Direction.west
                       else if quarters == 1 then Direction.north
                       else if quarters == 2 then Direction.east
                       else Direction.south

-- Final proof statement
theorem spinner_final_direction : direction_after_revolutions Direction.south (4 + 3/4 - (6 + 1/2)) = Direction.east := 
by 
  sorry

end NUMINAMATH_GPT_spinner_final_direction_l530_53048


namespace NUMINAMATH_GPT_rosemary_leaves_count_l530_53052

-- Define the number of pots for each plant type
def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6

-- Define the number of leaves each plant type has
def basil_leaves : ℕ := 4
def thyme_leaves : ℕ := 30
def total_leaves : ℕ := 354

-- Prove that the number of leaves on each rosemary plant is 18
theorem rosemary_leaves_count (R : ℕ) (h : basil_pots * basil_leaves + rosemary_pots * R + thyme_pots * thyme_leaves = total_leaves) : R = 18 :=
by {
  -- Following steps are within the theorem's proof
  sorry
}

end NUMINAMATH_GPT_rosemary_leaves_count_l530_53052


namespace NUMINAMATH_GPT_parabola_focus_eq_l530_53077

/-- Given the equation of a parabola y = -4x^2 - 8x + 1, prove that its focus is at (-1, 79/16). -/
theorem parabola_focus_eq :
  ∀ x y : ℝ, y = -4 * x ^ 2 - 8 * x + 1 → 
  ∃ h k p : ℝ, y = -4 * (x + 1)^2 + 5 ∧ 
  h = -1 ∧ k = 5 ∧ p = -1 / 16 ∧ (h, k + p) = (-1, 79/16) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_eq_l530_53077


namespace NUMINAMATH_GPT_minimum_value_ineq_l530_53094

theorem minimum_value_ineq (x : ℝ) (hx : 0 < x) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_ineq_l530_53094


namespace NUMINAMATH_GPT_gasoline_price_decrease_l530_53083

theorem gasoline_price_decrease (a : ℝ) (h : 0 ≤ a) :
  8.1 * (1 - a / 100) ^ 2 = 7.8 :=
sorry

end NUMINAMATH_GPT_gasoline_price_decrease_l530_53083


namespace NUMINAMATH_GPT_caleb_counted_right_angles_l530_53023

-- Definitions for conditions
def rectangular_park_angles : ℕ := 4
def square_field_angles : ℕ := 4
def total_angles (x y : ℕ) : ℕ := x + y

-- Theorem stating the problem
theorem caleb_counted_right_angles (h : total_angles rectangular_park_angles square_field_angles = 8) : 
   "type of anges Caleb counted" = "right angles" :=
sorry

end NUMINAMATH_GPT_caleb_counted_right_angles_l530_53023


namespace NUMINAMATH_GPT_center_and_radius_of_circle_l530_53012

def circle_equation := ∀ (x y : ℝ), x^2 + y^2 - 2*x - 3 = 0

theorem center_and_radius_of_circle :
  (∃ h k r : ℝ, (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x - 3 = 0) ∧ h = 1 ∧ k = 0 ∧ r = 2) :=
sorry

end NUMINAMATH_GPT_center_and_radius_of_circle_l530_53012


namespace NUMINAMATH_GPT_part1_unique_zero_part2_inequality_l530_53047

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x + 1 / x

theorem part1_unique_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

theorem part2_inequality (n : ℕ) (h : n > 0) : 
  Real.log ((n + 1) / n) < 1 / Real.sqrt (n^2 + n) := by
  sorry

end NUMINAMATH_GPT_part1_unique_zero_part2_inequality_l530_53047


namespace NUMINAMATH_GPT_product_of_x_and_y_l530_53064

variables (EF FG GH HE : ℕ) (x y : ℕ)

theorem product_of_x_and_y (h1: EF = 42) (h2: FG = 4 * y^3) (h3: GH = 2 * x + 10) (h4: HE = 32) (h5: EF = GH) (h6: FG = HE) :
  x * y = 32 :=
by
  sorry

end NUMINAMATH_GPT_product_of_x_and_y_l530_53064


namespace NUMINAMATH_GPT_quadratic_prime_roots_l530_53058

theorem quadratic_prime_roots (k : ℕ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p + q = 101 → p * q = k → False :=
by
  sorry

end NUMINAMATH_GPT_quadratic_prime_roots_l530_53058


namespace NUMINAMATH_GPT_boys_in_class_l530_53037

theorem boys_in_class (total_students : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ)
    (h_ratio : ratio_girls = 3) (h_ratio_boys : ratio_boys = 4)
    (h_total_students : total_students = 35) :
    ∃ boys, boys = 20 :=
by
  let k := total_students / (ratio_girls + ratio_boys)
  have hk : k = 5 := by sorry
  let boys := ratio_boys * k
  have h_boys : boys = 20 := by sorry
  exact ⟨boys, h_boys⟩

end NUMINAMATH_GPT_boys_in_class_l530_53037


namespace NUMINAMATH_GPT_emily_garden_larger_l530_53056

-- Define the dimensions and conditions given in the problem
def john_length : ℕ := 30
def john_width : ℕ := 60
def emily_length : ℕ := 35
def emily_width : ℕ := 55

-- Define the effective area for John’s garden given the double space requirement
def john_usable_area : ℕ := (john_length * john_width) / 2

-- Define the total area for Emily’s garden
def emily_usable_area : ℕ := emily_length * emily_width

-- State the theorem to be proved
theorem emily_garden_larger : emily_usable_area - john_usable_area = 1025 :=
by
  sorry

end NUMINAMATH_GPT_emily_garden_larger_l530_53056


namespace NUMINAMATH_GPT_sum_of_cubes_divisible_by_9n_l530_53055

theorem sum_of_cubes_divisible_by_9n (n : ℕ) (h : n % 3 ≠ 0) : 
  ((n - 1)^3 + n^3 + (n + 1)^3) % (9 * n) = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_divisible_by_9n_l530_53055


namespace NUMINAMATH_GPT_at_least_one_equals_a_l530_53033

theorem at_least_one_equals_a (x y z a : ℝ) (hx_ne_0 : x ≠ 0) (hy_ne_0 : y ≠ 0) (hz_ne_0 : z ≠ 0) (ha_ne_0 : a ≠ 0)
  (h1 : x + y + z = a) (h2 : 1/x + 1/y + 1/z = 1/a) : x = a ∨ y = a ∨ z = a :=
  sorry

end NUMINAMATH_GPT_at_least_one_equals_a_l530_53033


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_m_l530_53007

noncomputable def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

/- Part I -/
theorem part_one_solution_set (x : ℝ) : 
  (f x (-1) <= 2) ↔ (0 <= x ∧ x <= 4 / 3) := 
sorry

/- Part II -/
theorem part_two_range_of_m (m : ℝ) : 
  (∀ x ∈ (Set.Icc 1 2), f x m <= |2 * x + 1|) ↔ (-3 <= m ∧ m <= 0) := 
sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_m_l530_53007


namespace NUMINAMATH_GPT_function_increasing_l530_53013

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem function_increasing (a b c : ℝ) (h : a^2 - 3 * b < 0) : 
  ∀ x y : ℝ, x < y → f x a b c < f y a b c := sorry

end NUMINAMATH_GPT_function_increasing_l530_53013


namespace NUMINAMATH_GPT_cos_alpha_sub_beta_cos_alpha_l530_53019

section

variables (α β : ℝ)
variables (cos_α : ℝ) (sin_α : ℝ) (cos_β : ℝ) (sin_β : ℝ)

-- The given conditions as premises
variable (h1: cos_α = Real.cos α)
variable (h2: sin_α = Real.sin α)
variable (h3: cos_β = Real.cos β)
variable (h4: sin_β = Real.sin β)
variable (h5: 0 < α ∧ α < π / 2)
variable (h6: -π / 2 < β ∧ β < 0)
variable (h7: (cos_α - cos_β)^2 + (sin_α - sin_β)^2 = 4 / 5)

-- Part I: Prove that cos(α - β) = 3/5
theorem cos_alpha_sub_beta : Real.cos (α - β) = 3 / 5 :=
by
  sorry

-- Additional condition for Part II
variable (h8: cos_β = 12 / 13)

-- Part II: Prove that cos α = 56 / 65
theorem cos_alpha : Real.cos α = 56 / 65 :=
by
  sorry

end

end NUMINAMATH_GPT_cos_alpha_sub_beta_cos_alpha_l530_53019


namespace NUMINAMATH_GPT_Shekar_marks_in_English_l530_53038

theorem Shekar_marks_in_English 
  (math_marks : ℕ) (science_marks : ℕ) (socialstudies_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (num_subjects : ℕ) 
  (mathscore : math_marks = 76)
  (sciencescore : science_marks = 65)
  (socialstudiesscore : socialstudies_marks = 82)
  (biologyscore : biology_marks = 85)
  (averagescore : average_marks = 74)
  (numsubjects : num_subjects = 5) :
  ∃ (english_marks : ℕ), english_marks = 62 :=
by
  sorry

end NUMINAMATH_GPT_Shekar_marks_in_English_l530_53038


namespace NUMINAMATH_GPT_M_squared_is_odd_l530_53026

theorem M_squared_is_odd (a b : ℤ) (h1 : a = b + 1) (c : ℤ) (h2 : c = a * b) (M : ℤ) (h3 : M^2 = a^2 + b^2 + c^2) : M^2 % 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_M_squared_is_odd_l530_53026


namespace NUMINAMATH_GPT_eccentricity_range_of_isosceles_right_triangle_l530_53088

theorem eccentricity_range_of_isosceles_right_triangle
  (a : ℝ) (e : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + y^2 = 1)
  (h_a_gt_1 : a > 1)
  (B C : ℝ × ℝ)
  (isosceles_right_triangle : ∀ (A B C : ℝ × ℝ), ∃ k : ℝ, k > 0 ∧ 
    B = (-(2*k*a^2)/(1 + a^2*k^2), 0) ∧ 
    C = ((2*k*a^2)/(a^2 + k^2), 0) ∧ 
    (B.1^2 + B.2^2 = C.1^2 + C.2^2 + 1))
  (unique_solution : ∀ (k : ℝ), ∃! k', k' = 1)
  : 0 < e ∧ e ≤ (Real.sqrt 6) / 3 :=
sorry

end NUMINAMATH_GPT_eccentricity_range_of_isosceles_right_triangle_l530_53088


namespace NUMINAMATH_GPT_number_of_5_digit_numbers_l530_53075

/-- There are 324 five-digit numbers starting with 2 that have exactly three identical digits which are not 2. -/
theorem number_of_5_digit_numbers : ∃ n : ℕ, n = 324 ∧ ∀ (d₁ d₂ : ℕ), 
  (d₁ ≠ 2) ∧ (d₁ ≠ d₂) ∧ (0 ≤ d₁ ∧ d₁ < 10) ∧ (0 ≤ d₂ ∧ d₂ < 10) → 
  n = 4 * 9 * 9 := by
  sorry

end NUMINAMATH_GPT_number_of_5_digit_numbers_l530_53075


namespace NUMINAMATH_GPT_f_le_2x_f_not_le_1_9x_l530_53027

-- Define the function f and conditions
def f : ℝ → ℝ := sorry

axiom non_neg_f : ∀ x, 0 ≤ x → 0 ≤ f x
axiom f_at_1 : f 1 = 1
axiom f_additivity : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- Proof for part (1): f(x) ≤ 2x for all x in [0, 1]
theorem f_le_2x : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x := 
by
  sorry

-- Part (2): The inequality f(x) ≤ 1.9x does not hold for all x
theorem f_not_le_1_9x : ¬ (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 1.9 * x) := 
by
  sorry

end NUMINAMATH_GPT_f_le_2x_f_not_le_1_9x_l530_53027


namespace NUMINAMATH_GPT_find_d_and_a11_l530_53091

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_d_and_a11 (a : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence a d →
  a 5 = 6 →
  a 8 = 15 →
  d = 3 ∧ a 11 = 24 :=
by
  intros h_seq h_a5 h_a8
  sorry

end NUMINAMATH_GPT_find_d_and_a11_l530_53091


namespace NUMINAMATH_GPT_john_behind_steve_l530_53063

theorem john_behind_steve
  (vJ : ℝ) (vS : ℝ) (ahead : ℝ) (t : ℝ) (d : ℝ)
  (hJ : vJ = 4.2) (hS : vS = 3.8) (hA : ahead = 2) (hT : t = 42.5)
  (h1 : vJ * t = d + ahead)
  (h2 : vS * t + ahead = vJ * t - ahead) :
  d = 15 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_john_behind_steve_l530_53063


namespace NUMINAMATH_GPT_inequality_condition_l530_53014

theorem inequality_condition (a x : ℝ) : 
  x^3 + 13 * a^2 * x > 5 * a * x^2 + 9 * a^3 ↔ x > a := 
by
  sorry

end NUMINAMATH_GPT_inequality_condition_l530_53014


namespace NUMINAMATH_GPT_eq_y_as_x_l530_53050

theorem eq_y_as_x (y x : ℝ) : 
  (y = 2*x - 3*y) ∨ (x = 2 - 3*y) ∨ (-y = 2*x - 1) ∨ (y = x) → (y = x) :=
by
  sorry

end NUMINAMATH_GPT_eq_y_as_x_l530_53050


namespace NUMINAMATH_GPT_max_wx_xy_yz_zt_l530_53036

theorem max_wx_xy_yz_zt {w x y z t : ℕ} (h_sum : w + x + y + z + t = 120)
  (hnn_w : 0 ≤ w) (hnn_x : 0 ≤ x) (hnn_y : 0 ≤ y) (hnn_z : 0 ≤ z) (hnn_t : 0 ≤ t) :
  wx + xy + yz + zt ≤ 3600 := 
sorry

end NUMINAMATH_GPT_max_wx_xy_yz_zt_l530_53036


namespace NUMINAMATH_GPT_laborer_monthly_income_l530_53080

theorem laborer_monthly_income
  (I : ℕ)
  (D : ℕ)
  (h1 : 6 * I + D = 510)
  (h2 : 4 * I - D = 270) : I = 78 := by
  sorry

end NUMINAMATH_GPT_laborer_monthly_income_l530_53080


namespace NUMINAMATH_GPT_solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l530_53041

variable (a x : ℝ)

theorem solve_inequality_case_a_lt_neg1 (h : a < -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

theorem solve_inequality_case_a_eq_neg1 (h : a = -1) :
  ((x - 1) * (x + a) > 0) ↔ (x ≠ 1) := sorry

theorem solve_inequality_case_a_gt_neg1 (h : a > -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

end NUMINAMATH_GPT_solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l530_53041


namespace NUMINAMATH_GPT_container_volume_ratio_l530_53045

theorem container_volume_ratio
  (C D : ℕ)
  (h1 : (3 / 5 : ℚ) * C = (1 / 2 : ℚ) * D)
  (h2 : (1 / 3 : ℚ) * ((1 / 2 : ℚ) * D) + (3 / 5 : ℚ) * C = C) :
  (C : ℚ) / D = 5 / 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_container_volume_ratio_l530_53045


namespace NUMINAMATH_GPT_hypotenuse_length_50_l530_53011

theorem hypotenuse_length_50 (a b : ℕ) (h₁ : a = 14) (h₂ : b = 48) :
  ∃ c : ℕ, c = 50 ∧ c = Nat.sqrt (a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_50_l530_53011


namespace NUMINAMATH_GPT_DeansCalculatorGame_l530_53089

theorem DeansCalculatorGame (r : ℕ) (c1 c2 c3 : ℤ) (h1 : r = 45) (h2 : c1 = 1) (h3 : c2 = 0) (h4 : c3 = -2) : 
  let final1 := (c1 ^ 3)
  let final2 := (c2 ^ 2)
  let final3 := (-c3)^45
  final1 + final2 + final3 = 3 := 
by
  sorry

end NUMINAMATH_GPT_DeansCalculatorGame_l530_53089


namespace NUMINAMATH_GPT_bus_total_capacity_l530_53015

-- Definitions based on conditions in a)
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seats_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 12

-- Proof statement
theorem bus_total_capacity : (left_side_seats + right_side_seats) * seats_per_seat + back_seat_capacity = 93 := by
  sorry

end NUMINAMATH_GPT_bus_total_capacity_l530_53015


namespace NUMINAMATH_GPT_range_of_m_l530_53097

noncomputable def isEllipse (m : ℝ) : Prop := (m^2 > 2 * m + 8) ∧ (2 * m + 8 > 0)
noncomputable def intersectsXAxisAtTwoPoints (m : ℝ) : Prop := (2 * m - 3)^2 - 1 > 0

theorem range_of_m (m : ℝ) :
  ((m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∨ (2 * m - 3)^2 - 1 > 0) ∧
  ¬ (m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∧ (2 * m - 3)^2 - 1 > 0)) →
  (m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l530_53097


namespace NUMINAMATH_GPT_remainder_of_98_mul_102_mod_9_l530_53003

theorem remainder_of_98_mul_102_mod_9 : (98 * 102) % 9 = 6 := 
by 
  -- Introducing the variables and arithmetic
  let x := 98 * 102 
  have h1 : x = 9996 := 
    by norm_num
  have h2 : x % 9 = 6 := 
    by norm_num
  -- Result
  exact h2

end NUMINAMATH_GPT_remainder_of_98_mul_102_mod_9_l530_53003


namespace NUMINAMATH_GPT_color_copies_comparison_l530_53060

theorem color_copies_comparison (n : ℕ) (pX pY : ℝ) (charge_diff : ℝ) 
  (h₀ : pX = 1.20) (h₁ : pY = 1.70) (h₂ : charge_diff = 35) 
  (h₃ : pY * n = pX * n + charge_diff) : n = 70 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_color_copies_comparison_l530_53060


namespace NUMINAMATH_GPT_sum_super_cool_rectangle_areas_eq_84_l530_53053

theorem sum_super_cool_rectangle_areas_eq_84 :
  ∀ (a b : ℕ), 
  (a * b = 3 * (a + b)) → 
  ∃ (S : ℕ), 
  S = 84 :=
by
  sorry

end NUMINAMATH_GPT_sum_super_cool_rectangle_areas_eq_84_l530_53053


namespace NUMINAMATH_GPT_trigonometric_identity_l530_53059

theorem trigonometric_identity
  (θ : ℝ)
  (h : (2 + (1 / (Real.sin θ) ^ 2)) / (1 + Real.sin θ) = 1) :
  (1 + Real.sin θ) * (2 + Real.cos θ) = 4 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l530_53059


namespace NUMINAMATH_GPT_evaluate_expr_at_neg3_l530_53042

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end NUMINAMATH_GPT_evaluate_expr_at_neg3_l530_53042


namespace NUMINAMATH_GPT_correct_flag_positions_l530_53009

-- Definitions for the gears and their relations
structure Gear where
  flag_position : ℝ -- position of the flag in degrees

-- Condition: Two identical gears
def identical_gears (A B : Gear) : Prop := true

-- Conditions: Initial positions and gear interaction
def initial_position_A (A : Gear) : Prop := A.flag_position = 0
def initial_position_B (B : Gear) : Prop := B.flag_position = 180
def gear_interaction (A B : Gear) (theta : ℝ) : Prop :=
  A.flag_position = -theta ∧ B.flag_position = theta

-- Definition for the final positions given a rotation angle θ
def final_position (A B : Gear) (theta : ℝ) : Prop :=
  identical_gears A B ∧ initial_position_A A ∧ initial_position_B B ∧ gear_interaction A B theta

-- Theorem stating the positions after some rotation θ
theorem correct_flag_positions (A B : Gear) (theta : ℝ) : final_position A B theta → 
  A.flag_position = -theta ∧ B.flag_position = theta :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_correct_flag_positions_l530_53009


namespace NUMINAMATH_GPT_football_total_points_l530_53079

theorem football_total_points :
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  Zach_points + Ben_points + Sarah_points + Emily_points = 109.0 :=
by
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  have h : Zach_points + Ben_points + Sarah_points + Emily_points = 42.0 + 21.0 + 18.5 + 27.5 := by rfl
  have total_points := 42.0 + 21.0 + 18.5 + 27.5
  have result := 109.0
  sorry

end NUMINAMATH_GPT_football_total_points_l530_53079


namespace NUMINAMATH_GPT_largest_value_n_under_100000_l530_53081

theorem largest_value_n_under_100000 :
  ∃ n : ℕ,
    0 ≤ n ∧
    n < 100000 ∧
    (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 ∧
    n = 99999 :=
sorry

end NUMINAMATH_GPT_largest_value_n_under_100000_l530_53081


namespace NUMINAMATH_GPT_min_value_of_expr_l530_53004

theorem min_value_of_expr (a : ℝ) (h : a > 3) : ∃ m, (∀ b > 3, b + 4 / (b - 3) ≥ m) ∧ m = 7 :=
sorry

end NUMINAMATH_GPT_min_value_of_expr_l530_53004


namespace NUMINAMATH_GPT_expression_simplification_l530_53040

def base_expr := (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) *
                (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64)

theorem expression_simplification :
  base_expr = 3^128 - 4^128 := by
  sorry

end NUMINAMATH_GPT_expression_simplification_l530_53040


namespace NUMINAMATH_GPT_probability_exactly_one_each_is_correct_l530_53096

def probability_one_each (total forks spoons knives teaspoons : ℕ) : ℚ :=
  (forks * spoons * knives * teaspoons : ℚ) / ((total.choose 4) : ℚ)

theorem probability_exactly_one_each_is_correct :
  probability_one_each 34 8 9 10 7 = 40 / 367 :=
by sorry

end NUMINAMATH_GPT_probability_exactly_one_each_is_correct_l530_53096


namespace NUMINAMATH_GPT_union_A_B_l530_53054

noncomputable def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
noncomputable def B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_A_B : A ∪ B = {x : ℝ | -1 < x} := by
  sorry

end NUMINAMATH_GPT_union_A_B_l530_53054


namespace NUMINAMATH_GPT_cos_of_sin_given_l530_53001

theorem cos_of_sin_given (θ : ℝ) (h : Real.sin (88 * Real.pi / 180 + θ) = 2 / 3) :
  Real.cos (178 * Real.pi / 180 + θ) = - (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_cos_of_sin_given_l530_53001


namespace NUMINAMATH_GPT_find_n_l530_53022

theorem find_n (n : ℕ) : (256 : ℝ)^(1/4) = (4 : ℝ)^n → n = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l530_53022


namespace NUMINAMATH_GPT_pizza_eaten_after_six_trips_l530_53046

theorem pizza_eaten_after_six_trips :
  (1 / 3) + (1 / 3) / 2 + (1 / 3) / 2 / 2 + (1 / 3) / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 / 2 = 21 / 32 :=
by
  sorry

end NUMINAMATH_GPT_pizza_eaten_after_six_trips_l530_53046


namespace NUMINAMATH_GPT_victor_percentage_80_l530_53005

def percentage_of_marks (marks_obtained : ℕ) (maximum_marks : ℕ) : ℕ :=
  (marks_obtained * 100) / maximum_marks

theorem victor_percentage_80 :
  percentage_of_marks 240 300 = 80 := by
  sorry

end NUMINAMATH_GPT_victor_percentage_80_l530_53005


namespace NUMINAMATH_GPT_sum_divisors_of_24_is_60_and_not_prime_l530_53062

def divisors (n : Nat) : List Nat :=
  List.filter (λ d => n % d = 0) (List.range (n + 1))

def sum_divisors (n : Nat) : Nat :=
  (divisors n).sum

def is_prime (n : Nat) : Bool :=
  n > 1 ∧ (List.filter (λ d => d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))).length = 0

theorem sum_divisors_of_24_is_60_and_not_prime :
  sum_divisors 24 = 60 ∧ ¬ is_prime 60 := 
by
  sorry

end NUMINAMATH_GPT_sum_divisors_of_24_is_60_and_not_prime_l530_53062


namespace NUMINAMATH_GPT_angles_arith_prog_triangle_l530_53069

noncomputable def a : ℕ := 8
noncomputable def b : ℕ := 37
noncomputable def c : ℕ := 0

theorem angles_arith_prog_triangle (y : ℝ) (h1 : y = 8 ∨ y * y = 37) :
  a + b + c = 45 := by
  -- skipping the detailed proof steps
  sorry

end NUMINAMATH_GPT_angles_arith_prog_triangle_l530_53069


namespace NUMINAMATH_GPT_range_of_m_correct_l530_53025

noncomputable def range_of_m (x : ℝ) (m : ℝ) : Prop :=
  (x + m) / (x - 2) - (2 * m) / (x - 2) = 3 ∧ x > 0 ∧ x ≠ 2

theorem range_of_m_correct (m : ℝ) : 
  (∃ x : ℝ, range_of_m x m) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_correct_l530_53025


namespace NUMINAMATH_GPT_exists_integer_solution_l530_53018

theorem exists_integer_solution (x : ℤ) (h : x - 1 < 0) : ∃ y : ℤ, y < 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_integer_solution_l530_53018


namespace NUMINAMATH_GPT_combined_weight_l530_53016

-- We define the variables and the conditions
variables (x y : ℝ)

-- First condition 
def condition1 : Prop := y = (16 - 4) + (30 - 6) + (x - 3)

-- Second condition
def condition2 : Prop := y = 12 + 24 + (x - 3)

-- The statement to prove
theorem combined_weight (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : y = x + 33 :=
by
  -- Skipping the proof part
  sorry

end NUMINAMATH_GPT_combined_weight_l530_53016


namespace NUMINAMATH_GPT_temperature_at_midnight_l530_53066

-- Define the variables for initial conditions and changes
def T_morning : ℤ := 7 -- Morning temperature in degrees Celsius
def ΔT_noon : ℤ := 2   -- Temperature increase at noon in degrees Celsius
def ΔT_midnight : ℤ := -10  -- Temperature drop at midnight in degrees Celsius

-- Calculate the temperatures at noon and midnight
def T_noon := T_morning + ΔT_noon
def T_midnight := T_noon + ΔT_midnight

-- State the theorem to prove the temperature at midnight
theorem temperature_at_midnight : T_midnight = -1 := by
  sorry

end NUMINAMATH_GPT_temperature_at_midnight_l530_53066


namespace NUMINAMATH_GPT_find_prime_and_integer_l530_53072

theorem find_prime_and_integer (p x : ℕ) (hp : Nat.Prime p) 
  (hx1 : 1 ≤ x) (hx2 : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (p, x) = (2, 1) ∨ (p, x) = (2, 2) ∨ (p, x) = (3, 1) ∨ (p, x) = (3, 3) ∨ ((p ≥ 5) ∧ (x = 1)) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_and_integer_l530_53072


namespace NUMINAMATH_GPT_find_length_of_DE_l530_53076

-- Define the setup: five points A, B, C, D, E on a circle
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define the given distances 
def AB : ℝ := 7
def BC : ℝ := 7
def AD : ℝ := 10

-- Define the total distance AC
def AC : ℝ := AB + BC

-- Define the length DE to be solved
def DE : ℝ := 0.2

-- State the theorem to be proved given the conditions
theorem find_length_of_DE : 
  DE = 0.2 :=
sorry

end NUMINAMATH_GPT_find_length_of_DE_l530_53076


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l530_53057

noncomputable def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ((2016 * (a 1 + a 2016)) / 2 = 1008)

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) (h : arithmetic_sequence_condition a) :
  ∃ x : ℝ, x = 4 ∧ (∀ y, y = (1 / a 1001 + 1 / a 1016) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l530_53057


namespace NUMINAMATH_GPT_rhombus_area_l530_53043

-- Define the rhombus with given conditions
def rhombus (a d1 d2 : ℝ) : Prop :=
  a = 9 ∧ abs (d1 - d2) = 10 

-- The theorem stating the area of the rhombus
theorem rhombus_area (a d1 d2 : ℝ) (h : rhombus a d1 d2) : 
  (d1 * d2) / 2 = 72 :=
by
  sorry

#check rhombus_area

end NUMINAMATH_GPT_rhombus_area_l530_53043


namespace NUMINAMATH_GPT_find_a_b_l530_53092

theorem find_a_b (a b : ℕ) (h1 : (a^3 - a^2 + 1) * (b^3 - b^2 + 2) = 2020) : 10 * a + b = 53 :=
by {
  -- Proof to be completed
  sorry
}

end NUMINAMATH_GPT_find_a_b_l530_53092


namespace NUMINAMATH_GPT_find_point_B_l530_53090

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (-3, -1)
def line_y_eq_2x (x : ℝ) : ℝ × ℝ := (x, 2 * x)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1 

theorem find_point_B (B : ℝ × ℝ) (hB : B = line_y_eq_2x B.1) (h_parallel : is_parallel (B.1 + 3, B.2 + 1) vector_a) :
  B = (2, 4) := 
  sorry

end NUMINAMATH_GPT_find_point_B_l530_53090


namespace NUMINAMATH_GPT_smallest_of_three_consecutive_odd_numbers_l530_53006

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) 
(h_sum : x + (x+2) + (x+4) = 69) : x = 21 :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_three_consecutive_odd_numbers_l530_53006


namespace NUMINAMATH_GPT_bench_cost_150_l530_53029

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end NUMINAMATH_GPT_bench_cost_150_l530_53029


namespace NUMINAMATH_GPT_sum_of_edges_of_rectangular_solid_l530_53061

theorem sum_of_edges_of_rectangular_solid 
(volume : ℝ) (surface_area : ℝ) (a b c : ℝ)
(h1 : volume = a * b * c)
(h2 : surface_area = 2 * (a * b + b * c + c * a))
(h3 : ∃ s : ℝ, s ≠ 0 ∧ a = b / s ∧ c = b * s)
(h4 : volume = 512)
(h5 : surface_area = 384) :
a + b + c = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_edges_of_rectangular_solid_l530_53061


namespace NUMINAMATH_GPT_simplify_expression_correct_l530_53065

noncomputable def simplify_expression (α : ℝ) : ℝ :=
    (2 * (Real.cos (2 * α))^2 - 1) / 
    (2 * Real.tan ((Real.pi / 4) - 2 * α) * (Real.sin ((3 * Real.pi / 4) - 2 * α))^2) -
    Real.tan (2 * α) + Real.cos (2 * α) - Real.sin (2 * α)

theorem simplify_expression_correct (α : ℝ) : 
    simplify_expression α = 
    (2 * Real.sqrt 2 * Real.sin ((Real.pi / 4) - 2 * α) * (Real.cos α)^2) /
    Real.cos (2 * α) := by
    sorry

end NUMINAMATH_GPT_simplify_expression_correct_l530_53065


namespace NUMINAMATH_GPT_count_valid_n_l530_53095

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, 300 < n^2 ∧ n^2 < 1200 ∧ n % 3 = 0) ∧
                     S.card = 6 := sorry

end NUMINAMATH_GPT_count_valid_n_l530_53095


namespace NUMINAMATH_GPT_product_of_three_consecutive_integers_l530_53099

theorem product_of_three_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 740)
    (x1 : ℕ := x - 1) (x2 : ℕ := x) (x3 : ℕ := x + 1) :
    x1 * x2 * x3 = 17550 :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_consecutive_integers_l530_53099


namespace NUMINAMATH_GPT_number_of_days_A_left_l530_53039

noncomputable def work_problem (W : ℝ) : Prop :=
  let A_rate := W / 45
  let B_rate := W / 40
  let days_B_alone := 23
  ∃ x : ℝ, x * (A_rate + B_rate) + days_B_alone * B_rate = W ∧ x = 9

theorem number_of_days_A_left (W : ℝ) : work_problem W :=
  sorry

end NUMINAMATH_GPT_number_of_days_A_left_l530_53039


namespace NUMINAMATH_GPT_right_triangle_area_l530_53051

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 21) (h2 : c = 15) (h3 : a^2 + b^2 = c^2):
  (1/2) * a * b = 54 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l530_53051


namespace NUMINAMATH_GPT_combined_weight_l530_53073

variable (a b c d : ℕ)

theorem combined_weight :
  a + b = 260 →
  b + c = 245 →
  c + d = 270 →
  a + d = 285 :=
by
  intros hab hbc hcd
  sorry

end NUMINAMATH_GPT_combined_weight_l530_53073


namespace NUMINAMATH_GPT_original_amount_of_rice_l530_53074

theorem original_amount_of_rice
  (x : ℕ) -- the total amount of rice in kilograms
  (h1 : x = 10 * 500) -- statement that needs to be proven
  (h2 : 210 = x * (21 / 50)) -- remaining rice condition after given fractions are consumed
  (consume_day_one : x - (3 / 10) * x  = (7 / 10) * x) -- after the first day's consumption
  (consume_day_two : ((7 / 10) * x) - ((2 / 5) * ((7 / 10) * x)) = 210) -- after the second day's consumption
  : x = 500 :=
by
  sorry

end NUMINAMATH_GPT_original_amount_of_rice_l530_53074


namespace NUMINAMATH_GPT_area_XMY_l530_53098

-- Definitions
structure Triangle :=
(area : ℝ)

def ratio (a b : ℝ) : Prop := ∃ k : ℝ, (a = k * b)

-- Given conditions
variables {XYZ XMY YZ MY : ℝ}
variables (h1 : ratio XYZ 35)
variables (h2 : ratio (XM / MY) (5 / 2))

-- Theorem to prove
theorem area_XMY (hYZ_ratio : YZ = XM + MY) (hshared_height : true) : XMY = 10 :=
by
  sorry

end NUMINAMATH_GPT_area_XMY_l530_53098
