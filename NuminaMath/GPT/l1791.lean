import Mathlib

namespace NUMINAMATH_GPT_alex_avg_speed_l1791_179123

theorem alex_avg_speed (v : ℝ) : 
  (4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) → v = 20 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_alex_avg_speed_l1791_179123


namespace NUMINAMATH_GPT_percent_decaf_coffee_l1791_179171

variable (initial_stock new_stock decaf_initial_percent decaf_new_percent : ℝ)
variable (initial_stock_pos new_stock_pos : initial_stock > 0 ∧ new_stock > 0)

theorem percent_decaf_coffee :
    initial_stock = 400 → 
    decaf_initial_percent = 20 → 
    new_stock = 100 → 
    decaf_new_percent = 60 → 
    (100 * ((decaf_initial_percent / 100 * initial_stock + decaf_new_percent / 100 * new_stock) / (initial_stock + new_stock))) = 28 := 
by
  sorry

end NUMINAMATH_GPT_percent_decaf_coffee_l1791_179171


namespace NUMINAMATH_GPT_product_of_consecutive_numbers_l1791_179192

theorem product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h₁: n * (n + 1) * (n + 2) = 210) (h₂: n + (n + 1) = 11) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_numbers_l1791_179192


namespace NUMINAMATH_GPT_electronics_weight_l1791_179198

variable (B C E : ℝ)
variable (h1 : B / (B * (4 / 7) - 8) = 2 * (B / (B * (4 / 7))))
variable (h2 : C = B * (4 / 7))
variable (h3 : E = B * (3 / 7))

theorem electronics_weight : E = 12 := by
  sorry

end NUMINAMATH_GPT_electronics_weight_l1791_179198


namespace NUMINAMATH_GPT_area_closed_figure_sqrt_x_x_cube_l1791_179126

noncomputable def integral_diff_sqrt_x_cube (a b : ℝ) :=
∫ x in a..b, (Real.sqrt x - x^3)

theorem area_closed_figure_sqrt_x_x_cube :
  integral_diff_sqrt_x_cube 0 1 = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_area_closed_figure_sqrt_x_x_cube_l1791_179126


namespace NUMINAMATH_GPT_cuboid_to_cube_surface_area_l1791_179197

variable (h w l : ℝ)
variable (volume_decreases : 64 = w^3 - w^2 * h)

theorem cuboid_to_cube_surface_area 
  (h w l : ℝ) 
  (cube_condition : w = l ∧ h = w + 4)
  (volume_condition : w^2 * h - w^3 = 64) : 
  (6 * w^2 = 96) :=
by
  sorry

end NUMINAMATH_GPT_cuboid_to_cube_surface_area_l1791_179197


namespace NUMINAMATH_GPT_correct_factorization_l1791_179102

-- Definitions of the options given in the problem
def optionA (a : ℝ) := a^3 - a = a * (a^2 - 1)
def optionB (a b : ℝ) := a^2 - 4 * b^2 = (a + 4 * b) * (a - 4 * b)
def optionC (a : ℝ) := a^2 - 2 * a - 8 = a * (a - 2) - 8
def optionD (a : ℝ) := a^2 - a + 1/4 = (a - 1/2)^2

-- Stating the proof problem
theorem correct_factorization : ∀ (a : ℝ), optionD a :=
by
  sorry

end NUMINAMATH_GPT_correct_factorization_l1791_179102


namespace NUMINAMATH_GPT_lamps_purchased_min_type_B_lamps_l1791_179160

variables (x y m : ℕ)

def total_lamps := x + y = 50
def total_cost := 40 * x + 65 * y = 2500
def profit_type_A := 60 - 40 = 20
def profit_type_B := 100 - 65 = 35
def profit_requirement := 20 * (50 - m) + 35 * m ≥ 1400

theorem lamps_purchased (h₁ : total_lamps x y) (h₂ : total_cost x y) : 
  x = 30 ∧ y = 20 :=
  sorry

theorem min_type_B_lamps (h₃ : profit_type_A) (h₄ : profit_type_B) (h₅ : profit_requirement m) : 
  m ≥ 27 :=
  sorry

end NUMINAMATH_GPT_lamps_purchased_min_type_B_lamps_l1791_179160


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1791_179157

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1 → x^2 - 2*x + 1 > 0) ∧ (¬(x^2 - 2*x + 1 > 0 → x > 1)) := by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1791_179157


namespace NUMINAMATH_GPT_incorrect_conclusion_l1791_179159

theorem incorrect_conclusion
  (a b : ℝ) 
  (h₁ : 1/a < 1/b) 
  (h₂ : 1/b < 0) 
  (h₃ : a < 0) 
  (h₄ : b < 0) 
  (h₅ : a > b) : ¬ (|a| + |b| > |a + b|) := 
sorry

end NUMINAMATH_GPT_incorrect_conclusion_l1791_179159


namespace NUMINAMATH_GPT_fly_distance_to_ceiling_l1791_179148

theorem fly_distance_to_ceiling :
  ∀ (x y z : ℝ), 
  (x = 3) → 
  (y = 4) → 
  (z * z + 25 = 49) →
  z = 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_fly_distance_to_ceiling_l1791_179148


namespace NUMINAMATH_GPT_latest_time_to_reach_80_degrees_l1791_179128

theorem latest_time_to_reach_80_degrees :
  ∀ (t : ℝ), (-t^2 + 14 * t + 40 = 80) → t ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_latest_time_to_reach_80_degrees_l1791_179128


namespace NUMINAMATH_GPT_cucumber_new_weight_l1791_179107

-- Definitions for the problem conditions
def initial_weight : ℝ := 100
def initial_water_percentage : ℝ := 0.99
def final_water_percentage : ℝ := 0.96
noncomputable def new_weight : ℝ := initial_weight * (1 - initial_water_percentage) / (1 - final_water_percentage)

-- The theorem stating the problem to be solved
theorem cucumber_new_weight : new_weight = 25 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_cucumber_new_weight_l1791_179107


namespace NUMINAMATH_GPT_license_plate_difference_l1791_179185

theorem license_plate_difference : 
    let alpha_plates := 26^4 * 10^4
    let beta_plates := 26^3 * 10^4
    alpha_plates - beta_plates = 10^4 * 26^3 * 25 := 
by sorry

end NUMINAMATH_GPT_license_plate_difference_l1791_179185


namespace NUMINAMATH_GPT_carol_first_toss_six_probability_l1791_179138

theorem carol_first_toss_six_probability :
  let p := 1 / 6
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  (prob_carol_first_six / (1 - prob_cycle)) = 125 / 671 :=
by
  let p := (1 / 6:ℚ)
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  have sum_geo_series : prob_carol_first_six / (1 - prob_cycle) = 125 / 671 := sorry
  exact sum_geo_series

end NUMINAMATH_GPT_carol_first_toss_six_probability_l1791_179138


namespace NUMINAMATH_GPT_total_letters_sent_l1791_179177

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_letters_sent_l1791_179177


namespace NUMINAMATH_GPT_find_f6_l1791_179140

-- Define the function f and the necessary properties
variable (f : ℕ+ → ℕ+)
variable (h1 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1)
variable (h2 : f 1 ≠ 1)

-- State the theorem to prove that f(6) = 5
theorem find_f6 : f 6 = 5 :=
sorry

end NUMINAMATH_GPT_find_f6_l1791_179140


namespace NUMINAMATH_GPT_jake_more_balloons_than_allan_l1791_179131

-- Define the initial and additional balloons for Allan
def initial_allan_balloons : Nat := 2
def additional_allan_balloons : Nat := 3

-- Total balloons Allan has in the park
def total_allan_balloons : Nat := initial_allan_balloons + additional_allan_balloons

-- Number of balloons Jake has
def jake_balloons : Nat := 6

-- The proof statement
theorem jake_more_balloons_than_allan : jake_balloons - total_allan_balloons = 1 := by
  sorry

end NUMINAMATH_GPT_jake_more_balloons_than_allan_l1791_179131


namespace NUMINAMATH_GPT_max_distance_l1791_179169

theorem max_distance (front_lifespan : ℕ) (rear_lifespan : ℕ)
  (h_front : front_lifespan = 21000)
  (h_rear : rear_lifespan = 28000) :
  ∃ (max_dist : ℕ), max_dist = 24000 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_l1791_179169


namespace NUMINAMATH_GPT_age_of_new_person_l1791_179136

theorem age_of_new_person 
    (n : ℕ) 
    (T : ℕ := n * 14) 
    (n_eq : n = 9) 
    (new_average : (T + A) / (n + 1) = 16) 
    (A : ℕ) : A = 34 :=
by
  sorry

end NUMINAMATH_GPT_age_of_new_person_l1791_179136


namespace NUMINAMATH_GPT_rectangle_same_color_exists_l1791_179168

def color := ℕ -- We use ℕ as a stand-in for three colors {0, 1, 2}

def same_color_rectangle_exists (coloring : (Fin 4) → (Fin 82) → color) : Prop :=
  ∃ (i j : Fin 4) (k l : Fin 82), i ≠ j ∧ k ≠ l ∧
    coloring i k = coloring i l ∧
    coloring j k = coloring j l ∧
    coloring i k = coloring j k

theorem rectangle_same_color_exists :
  ∀ (coloring : (Fin 4) → (Fin 82) → color),
  same_color_rectangle_exists coloring :=
by
  sorry

end NUMINAMATH_GPT_rectangle_same_color_exists_l1791_179168


namespace NUMINAMATH_GPT_count_semiprimes_expressed_as_x_cubed_minus_1_l1791_179106

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_semiprime (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p * q = n

theorem count_semiprimes_expressed_as_x_cubed_minus_1 :
  (∃ S : Finset ℕ, 
    S.card = 4 ∧ 
    ∀ n ∈ S, n < 2018 ∧ 
    ∃ x : ℕ, x > 0 ∧ x^3 - 1 = n ∧ is_semiprime n) :=
sorry

end NUMINAMATH_GPT_count_semiprimes_expressed_as_x_cubed_minus_1_l1791_179106


namespace NUMINAMATH_GPT_distributive_addition_over_multiplication_not_hold_l1791_179125

def complex_add (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 + z2.1, z1.2 + z2.2)

def complex_mul (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 * z2.1 - z1.2 * z2.2, z1.1 * z2.2 + z1.2 * z2.1)

theorem distributive_addition_over_multiplication_not_hold (x y x1 y1 x2 y2 : ℝ) :
  complex_add (x, y) (complex_mul (x1, y1) (x2, y2)) ≠
    complex_mul (complex_add (x, y) (x1, y1)) (complex_add (x, y) (x2, y2)) :=
sorry

end NUMINAMATH_GPT_distributive_addition_over_multiplication_not_hold_l1791_179125


namespace NUMINAMATH_GPT_boys_count_l1791_179163

/-
Conditions:
1. The total number of members in the chess team is 26.
2. 18 members were present at the last session.
3. One-third of the girls attended the session.
4. All of the boys attended the session.
-/
def TotalMembers : Nat := 26
def LastSessionAttendance : Nat := 18
def GirlsAttendance (G : Nat) : Nat := G / 3
def BoysAttendance (B : Nat) : Nat := B

/-
Main theorem statement:
Prove that the number of boys in the chess team is 14.
-/
theorem boys_count (B G : Nat) (h1 : B + G = TotalMembers) (h2 : GirlsAttendance G + BoysAttendance B = LastSessionAttendance) : B = 14 :=
by
  sorry

end NUMINAMATH_GPT_boys_count_l1791_179163


namespace NUMINAMATH_GPT_pizza_pasta_cost_difference_l1791_179194

variable (x y z : ℝ)
variable (A1 : 2 * x + 3 * y + 4 * z = 53)
variable (A2 : 5 * x + 6 * y + 7 * z = 107)

theorem pizza_pasta_cost_difference :
  x - z = 1 :=
by
  sorry

end NUMINAMATH_GPT_pizza_pasta_cost_difference_l1791_179194


namespace NUMINAMATH_GPT_distance_karen_covers_l1791_179109

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end NUMINAMATH_GPT_distance_karen_covers_l1791_179109


namespace NUMINAMATH_GPT_gcd_of_differences_is_10_l1791_179143

theorem gcd_of_differences_is_10 (a b c : ℕ) (h1 : b > a) (h2 : c > b) (h3 : c > a)
  (h4 : b - a = 20) (h5 : c - b = 50) (h6 : c - a = 70) : Int.gcd (b - a) (Int.gcd (c - b) (c - a)) = 10 := 
sorry

end NUMINAMATH_GPT_gcd_of_differences_is_10_l1791_179143


namespace NUMINAMATH_GPT_largest_possible_square_area_l1791_179189

def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 6
def largest_square_side : ℕ := rectangle_width
def largest_square_area : ℕ := largest_square_side * largest_square_side

theorem largest_possible_square_area :
  largest_square_area = 36 := by
    sorry

end NUMINAMATH_GPT_largest_possible_square_area_l1791_179189


namespace NUMINAMATH_GPT_compare_combined_sums_l1791_179130

def numeral1 := 7524258
def numeral2 := 523625072

def place_value_2_numeral1 := 200000 + 20
def place_value_5_numeral1 := 50000 + 500
def combined_sum_numeral1 := place_value_2_numeral1 + place_value_5_numeral1

def place_value_2_numeral2 := 200000000 + 20
def place_value_5_numeral2 := 500000 + 50
def combined_sum_numeral2 := place_value_2_numeral2 + place_value_5_numeral2

def difference := combined_sum_numeral2 - combined_sum_numeral1

theorem compare_combined_sums :
  difference = 200249550 := by
  sorry

end NUMINAMATH_GPT_compare_combined_sums_l1791_179130


namespace NUMINAMATH_GPT_find_b_l1791_179199

-- Define the conditions of the equations
def condition_1 (x y a : ℝ) : Prop := x * Real.cos a + y * Real.sin a + 3 ≤ 0
def condition_2 (x y b : ℝ) : Prop := x^2 + y^2 + 8 * x - 4 * y - b^2 + 6 * b + 11 = 0

-- Define the proof problem
theorem find_b (b : ℝ) :
  (∀ a x y, condition_1 x y a → condition_2 x y b) →
  b ∈ Set.Iic (-2 * Real.sqrt 5) ∪ Set.Ici (6 + 2 * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1791_179199


namespace NUMINAMATH_GPT_radius_wire_is_4_cm_l1791_179115

noncomputable def radius_of_wire_cross_section (r_sphere : ℝ) (length_wire : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * r_sphere^3
  let volume_wire := volume_sphere / length_wire
  Real.sqrt (volume_wire / Real.pi)

theorem radius_wire_is_4_cm :
  radius_of_wire_cross_section 12 144 = 4 :=
by
  unfold radius_of_wire_cross_section
  sorry

end NUMINAMATH_GPT_radius_wire_is_4_cm_l1791_179115


namespace NUMINAMATH_GPT_buses_more_than_vans_l1791_179195

-- Definitions based on conditions
def vans : Float := 6.0
def buses : Float := 8.0
def people_per_van : Float := 6.0
def people_per_bus : Float := 18.0

-- Calculate total people in vans and buses
def total_people_vans : Float := vans * people_per_van
def total_people_buses : Float := buses * people_per_bus

-- Prove the difference
theorem buses_more_than_vans : total_people_buses - total_people_vans = 108.0 :=
by
  sorry

end NUMINAMATH_GPT_buses_more_than_vans_l1791_179195


namespace NUMINAMATH_GPT_sum_a_b_range_l1791_179146

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem sum_a_b_range : let a := 0
                       let b := 3
                       a + b = 3 := by
  sorry

end NUMINAMATH_GPT_sum_a_b_range_l1791_179146


namespace NUMINAMATH_GPT_total_distance_walked_l1791_179147

noncomputable def desk_to_fountain_distance : ℕ := 30
noncomputable def number_of_trips : ℕ := 4

theorem total_distance_walked :
  2 * desk_to_fountain_distance * number_of_trips = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_walked_l1791_179147


namespace NUMINAMATH_GPT_max_value_4287_5_l1791_179145

noncomputable def maximum_value_of_expression (x y : ℝ) := x * y * (105 - 2 * x - 5 * y)

theorem max_value_4287_5 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 105) :
  maximum_value_of_expression x y ≤ 4287.5 :=
sorry

end NUMINAMATH_GPT_max_value_4287_5_l1791_179145


namespace NUMINAMATH_GPT_tank_capacity_l1791_179118

theorem tank_capacity (x : ℝ) (h₁ : (3/4) * x = (1/3) * x + 18) : x = 43.2 := sorry

end NUMINAMATH_GPT_tank_capacity_l1791_179118


namespace NUMINAMATH_GPT_at_least_one_fuse_blows_l1791_179154

theorem at_least_one_fuse_blows (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.74) (independent : ∀ (A B : Prop), A ∧ B → ¬(A ∨ B)) :
  1 - (1 - pA) * (1 - pB) = 0.961 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_fuse_blows_l1791_179154


namespace NUMINAMATH_GPT_sum_of_converted_2016_is_correct_l1791_179144

theorem sum_of_converted_2016_is_correct :
  (20.16 + 20.16 + 20.16 + 201.6 + 201.6 + 201.6 = 463.68 ∨
   2.016 + 2.016 + 2.016 + 20.16 + 20.16 + 20.16 = 46.368) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_converted_2016_is_correct_l1791_179144


namespace NUMINAMATH_GPT_thomas_spends_40000_in_a_decade_l1791_179129

/-- 
Thomas spends 4k dollars every year on his car insurance.
One decade is 10 years.
-/
def spending_per_year : ℕ := 4000

def years_in_a_decade : ℕ := 10

/-- 
We need to prove that the total amount Thomas spends in a decade on car insurance equals $40,000.
-/
theorem thomas_spends_40000_in_a_decade : spending_per_year * years_in_a_decade = 40000 := by
  sorry

end NUMINAMATH_GPT_thomas_spends_40000_in_a_decade_l1791_179129


namespace NUMINAMATH_GPT_principal_trebled_after_5_years_l1791_179142

theorem principal_trebled_after_5_years (P R: ℝ) (n: ℝ) :
  (P * R * 10 / 100 = 700) →
  ((P * R * n + 3 * P * R * (10 - n)) / 100 = 1400) →
  n = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_principal_trebled_after_5_years_l1791_179142


namespace NUMINAMATH_GPT_range_of_m_l1791_179112

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x > 0 → (y = 1 - 3 * m / x) → y > 0) ↔ (m > 1 / 3) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1791_179112


namespace NUMINAMATH_GPT_union_S_T_l1791_179183

def S : Set ℝ := { x | 3 < x ∧ x ≤ 6 }
def T : Set ℝ := { x | x^2 - 4*x - 5 ≤ 0 }

theorem union_S_T : S ∪ T = { x | -1 ≤ x ∧ x ≤ 6 } := 
by 
  sorry

end NUMINAMATH_GPT_union_S_T_l1791_179183


namespace NUMINAMATH_GPT_correct_option_l1791_179121

-- Definitions based on conditions
def sentence_structure : String := "He’s never interested in what ______ is doing."

def option_A : String := "no one else"
def option_B : String := "anyone else"
def option_C : String := "someone else"
def option_D : String := "nobody else"

-- The proof statement
theorem correct_option : option_B = "anyone else" := by
  sorry

end NUMINAMATH_GPT_correct_option_l1791_179121


namespace NUMINAMATH_GPT_factor_of_increase_l1791_179172

-- Define the conditions
def interest_rate : ℝ := 0.25
def time_period : ℕ := 4

-- Define the principal amount as a variable
variable (P : ℝ)

-- Define the simple interest formula
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * R * (T : ℝ)

-- Define the total amount function
def total_amount (P : ℝ) (SI : ℝ) : ℝ := P + SI

-- The theorem that we need to prove: The factor by which the sum of money increases is 2
theorem factor_of_increase :
  total_amount P (simple_interest P interest_rate time_period) = 2 * P := by
  sorry

end NUMINAMATH_GPT_factor_of_increase_l1791_179172


namespace NUMINAMATH_GPT_conclusion_l1791_179117

-- Assuming U is the universal set and Predicates represent Mems, Ens, and Veens
variable (U : Type)
variable (Mem : U → Prop)
variable (En : U → Prop)
variable (Veen : U → Prop)

-- Hypotheses
variable (h1 : ∀ x, Mem x → En x)          -- Hypothesis I: All Mems are Ens
variable (h2 : ∀ x, En x → ¬Veen x)        -- Hypothesis II: No Ens are Veens

-- To be proven
theorem conclusion (x : U) : (Mem x → ¬Veen x) ∧ (Mem x → ¬Veen x) := sorry

end NUMINAMATH_GPT_conclusion_l1791_179117


namespace NUMINAMATH_GPT_problem_l1791_179108

def f (x a : ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem problem (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 0) : f 1 a = 4 :=
sorry

end NUMINAMATH_GPT_problem_l1791_179108


namespace NUMINAMATH_GPT_avg_wx_l1791_179127

theorem avg_wx (w x y : ℝ) (h1 : 3 / w + 3 / x = 3 / y) (h2 : w * x = y) : (w + x) / 2 = 1 / 2 :=
by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_avg_wx_l1791_179127


namespace NUMINAMATH_GPT_temperature_difference_l1791_179153

theorem temperature_difference :
  let T_midnight := -4
  let T_10am := 5
  T_10am - T_midnight = 9 :=
by
  let T_midnight := -4
  let T_10am := 5
  show T_10am - T_midnight = 9
  sorry

end NUMINAMATH_GPT_temperature_difference_l1791_179153


namespace NUMINAMATH_GPT_price_of_candied_grape_l1791_179103

theorem price_of_candied_grape (x : ℝ) (h : 15 * 2 + 12 * x = 48) : x = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_price_of_candied_grape_l1791_179103


namespace NUMINAMATH_GPT_number_subsets_property_p_l1791_179190

def has_property_p (a b : ℕ) : Prop := 17 ∣ (a + b)

noncomputable def num_subsets_with_property_p : ℕ :=
  -- sorry, put computation result here using the steps above but skipping actual computation for brevity
  3928

theorem number_subsets_property_p :
  num_subsets_with_property_p = 3928 := sorry

end NUMINAMATH_GPT_number_subsets_property_p_l1791_179190


namespace NUMINAMATH_GPT_exists_schoolchild_who_participated_in_all_competitions_l1791_179113

theorem exists_schoolchild_who_participated_in_all_competitions
    (competitions : Fin 50 → Finset ℕ)
    (h_card : ∀ i, (competitions i).card = 30)
    (h_unique : ∀ i j, i ≠ j → competitions i ≠ competitions j)
    (h_intersect : ∀ S : Finset (Fin 50), S.card = 30 → 
      ∃ x, ∀ i ∈ S, x ∈ competitions i) :
    ∃ x, ∀ i, x ∈ competitions i :=
by
  sorry

end NUMINAMATH_GPT_exists_schoolchild_who_participated_in_all_competitions_l1791_179113


namespace NUMINAMATH_GPT_shares_difference_l1791_179175

noncomputable def Faruk_share (V : ℕ) : ℕ := (3 * (V / 5))
noncomputable def Ranjith_share (V : ℕ) : ℕ := (7 * (V / 5))

theorem shares_difference {V : ℕ} (hV : V = 1500) : 
  Ranjith_share V - Faruk_share V = 1200 :=
by
  rw [Faruk_share, Ranjith_share]
  subst hV
  -- It's just a declaration of the problem and sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_shares_difference_l1791_179175


namespace NUMINAMATH_GPT_cube_volume_given_surface_area_l1791_179166

/-- Surface area of a cube given the side length. -/
def surface_area (side_length : ℝ) := 6 * side_length^2

/-- Volume of a cube given the side length. -/
def volume (side_length : ℝ) := side_length^3

theorem cube_volume_given_surface_area :
  ∃ side_length : ℝ, surface_area side_length = 24 ∧ volume side_length = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_given_surface_area_l1791_179166


namespace NUMINAMATH_GPT_number_of_children_l1791_179156

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end NUMINAMATH_GPT_number_of_children_l1791_179156


namespace NUMINAMATH_GPT_solve_for_x_l1791_179181

theorem solve_for_x :
  exists x : ℝ, 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02) ^ 2 ∧ x = 0.04 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1791_179181


namespace NUMINAMATH_GPT_mod_exp_result_l1791_179151

theorem mod_exp_result :
  (2 ^ 46655) % 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_mod_exp_result_l1791_179151


namespace NUMINAMATH_GPT_sequence_strictly_increasing_from_14_l1791_179162

def a (n : ℕ) : ℤ := n^4 - 20 * n^2 - 10 * n + 1

theorem sequence_strictly_increasing_from_14 :
  ∀ n : ℕ, n ≥ 14 → a (n + 1) > a n :=
by
  sorry

end NUMINAMATH_GPT_sequence_strictly_increasing_from_14_l1791_179162


namespace NUMINAMATH_GPT_function_range_cosine_identity_l1791_179178

theorem function_range_cosine_identity
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h₀ : 0 < ω)
  (h₁ : ∀ x, f x = (1/2) * Real.cos (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x))
  (h₂ : ∀ x, f (x + π / ω) = f x) :
  Set.Icc (f (-π / 3)) (f (π / 6)) = Set.Icc (-1 / 2) 1 :=
by
  sorry

end NUMINAMATH_GPT_function_range_cosine_identity_l1791_179178


namespace NUMINAMATH_GPT_mango_rate_l1791_179149

theorem mango_rate (x : ℕ) : 
  (sells_rate : ℕ) = 3 → 
  (profit_percent : ℕ) = 50 → 
  (buying_price : ℚ) = 2 := by
  sorry

end NUMINAMATH_GPT_mango_rate_l1791_179149


namespace NUMINAMATH_GPT_number_of_equilateral_triangles_l1791_179111

noncomputable def parabola_equilateral_triangles (y x : ℝ) : Prop :=
  y^2 = 4 * x

theorem number_of_equilateral_triangles : ∃ n : ℕ, n = 2 ∧
  ∀ (a b c d e : ℝ), 
    (parabola_equilateral_triangles (a - 1) b) ∧ 
    (parabola_equilateral_triangles (c - 1) d) ∧ 
    ((a = e ∧ b = 0) ∨ (c = e ∧ d = 0)) → n = 2 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_equilateral_triangles_l1791_179111


namespace NUMINAMATH_GPT_smallest_add_to_multiple_of_4_l1791_179176

theorem smallest_add_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ (587 + n) % 4 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (587 + m) % 4 = 0 → n ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_add_to_multiple_of_4_l1791_179176


namespace NUMINAMATH_GPT_estimate_num_2016_digit_squares_l1791_179182

noncomputable def num_estimate_2016_digit_squares : ℕ := 2016

theorem estimate_num_2016_digit_squares :
  let t1 := (10 ^ (2016 / 2) - 10 ^ (2015 / 2) - 1)
  let t2 := (2017 ^ 10)
  let result := t1 / t2
  t1 > 10 ^ 1000 → 
  result > 10 ^ 900 →
  result == num_estimate_2016_digit_squares :=
by
  intros
  sorry

end NUMINAMATH_GPT_estimate_num_2016_digit_squares_l1791_179182


namespace NUMINAMATH_GPT_john_february_phone_bill_l1791_179119

-- Define given conditions
def base_cost : ℕ := 30
def included_hours : ℕ := 50
def overage_cost_per_minute : ℕ := 15 -- costs per minute in cents
def hours_talked_in_February : ℕ := 52

-- Define conversion from dollars to cents
def cents_per_dollar : ℕ := 100

-- Define total cost calculation
def total_cost (base_cost : ℕ) (included_hours : ℕ) (overage_cost_per_minute : ℕ) (hours_talked : ℕ) : ℕ :=
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_cost := extra_minutes * overage_cost_per_minute
  base_cost * cents_per_dollar + extra_cost

-- State the theorem
theorem john_february_phone_bill : total_cost base_cost included_hours overage_cost_per_minute hours_talked_in_February = 4800 := by
  sorry

end NUMINAMATH_GPT_john_february_phone_bill_l1791_179119


namespace NUMINAMATH_GPT_find_angle_l1791_179120

-- Define the conditions
variables (x : ℝ)

-- Conditions given in the problem
def angle_complement_condition (x : ℝ) := (10 : ℝ) + 3 * x
def complementary_condition (x : ℝ) := x + angle_complement_condition x = 90

-- Prove that the angle x equals to 20 degrees
theorem find_angle : (complementary_condition x) → x = 20 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_angle_l1791_179120


namespace NUMINAMATH_GPT_south_movement_notation_l1791_179161

/-- If moving north 8m is denoted as +8m, then moving south 5m is denoted as -5m. -/
theorem south_movement_notation (north south : ℤ) (h1 : north = 8) (h2 : south = -north) : south = -5 :=
by
  sorry

end NUMINAMATH_GPT_south_movement_notation_l1791_179161


namespace NUMINAMATH_GPT_turtles_remaining_on_log_l1791_179135
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end NUMINAMATH_GPT_turtles_remaining_on_log_l1791_179135


namespace NUMINAMATH_GPT_order_abc_l1791_179105

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := Real.exp (0.9 * Real.log 1.1)

theorem order_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_order_abc_l1791_179105


namespace NUMINAMATH_GPT_ara_height_l1791_179101

theorem ara_height (shea_height_now : ℝ) (shea_growth_percent : ℝ) (ara_growth_fraction : ℝ)
    (height_now : shea_height_now = 75) (growth_percent : shea_growth_percent = 0.25) 
    (growth_fraction : ara_growth_fraction = (2/3)) : 
    ∃ ara_height_now : ℝ, ara_height_now = 70 := by
  sorry

end NUMINAMATH_GPT_ara_height_l1791_179101


namespace NUMINAMATH_GPT_quincy_sold_more_than_jake_l1791_179133

variables (T : ℕ) (Jake Quincy : ℕ)

def thors_sales (T : ℕ) := T
def jakes_sales (T : ℕ) := T + 10
def quincys_sales (T : ℕ) := 10 * T

theorem quincy_sold_more_than_jake (h1 : jakes_sales T = Jake) 
  (h2 : quincys_sales T = Quincy) (h3 : Quincy = 200) : 
  Quincy - Jake = 170 :=
by
  sorry

end NUMINAMATH_GPT_quincy_sold_more_than_jake_l1791_179133


namespace NUMINAMATH_GPT_volume_of_pyramid_l1791_179152

theorem volume_of_pyramid (A B C : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (28, 0)) (hC : C = (12, 20))
  (D : ℝ × ℝ) (hD : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (E : ℝ × ℝ) (hE : E = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (F : ℝ × ℝ) (hF : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (∃ h : ℝ, h = 10 ∧ ∃ V : ℝ, V = (1 / 3) * 70 * h ∧ V = 700 / 3) :=
by sorry

end NUMINAMATH_GPT_volume_of_pyramid_l1791_179152


namespace NUMINAMATH_GPT_fraction_of_b_eq_two_thirds_l1791_179170

theorem fraction_of_b_eq_two_thirds (A B : ℝ) (x : ℝ) (h1 : A + B = 1210) (h2 : B = 484)
  (h3 : (2/3) * A = x * B) : x = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_b_eq_two_thirds_l1791_179170


namespace NUMINAMATH_GPT_trig_identity_problem_l1791_179179

theorem trig_identity_problem {α : ℝ} (h : Real.tan α = 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_problem_l1791_179179


namespace NUMINAMATH_GPT_sin_func_even_min_period_2pi_l1791_179180

noncomputable def f (x : ℝ) : ℝ := Real.sin (13 * Real.pi / 2 - x)

theorem sin_func_even_min_period_2pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ 2 * Real.pi) ∧ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) :=
by
  sorry

end NUMINAMATH_GPT_sin_func_even_min_period_2pi_l1791_179180


namespace NUMINAMATH_GPT_total_strength_college_l1791_179184

-- Defining the conditions
def C : ℕ := 500
def B : ℕ := 600
def Both : ℕ := 220

-- Declaring the theorem
theorem total_strength_college : (C + B - Both) = 880 :=
by
  -- The proof is not required, put sorry
  sorry

end NUMINAMATH_GPT_total_strength_college_l1791_179184


namespace NUMINAMATH_GPT_probability_math_majors_consecutive_l1791_179158

noncomputable def total_ways := Nat.choose 11 4 -- Number of ways to choose 5 persons out of 12 (fixing one)
noncomputable def favorable_ways := 12         -- Number of ways to arrange 5 math majors consecutively around a round table

theorem probability_math_majors_consecutive :
  (favorable_ways : ℚ) / total_ways = 2 / 55 :=
by
  sorry

end NUMINAMATH_GPT_probability_math_majors_consecutive_l1791_179158


namespace NUMINAMATH_GPT_discriminant_of_quadratic_5x2_minus_2x_minus_7_l1791_179193

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem discriminant_of_quadratic_5x2_minus_2x_minus_7 :
  quadratic_discriminant 5 (-2) (-7) = 144 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_5x2_minus_2x_minus_7_l1791_179193


namespace NUMINAMATH_GPT_Edmund_earns_64_dollars_l1791_179188

-- Conditions
def chores_per_week : Nat := 12
def pay_per_extra_chore : Nat := 2
def chores_per_day : Nat := 4
def weeks : Nat := 2
def days_per_week : Nat := 7

-- Goal
theorem Edmund_earns_64_dollars :
  let total_chores_without_extra := chores_per_week * weeks
  let total_chores_with_extra := chores_per_day * (days_per_week * weeks)
  let extra_chores := total_chores_with_extra - total_chores_without_extra
  let earnings := pay_per_extra_chore * extra_chores
  earnings = 64 :=
by
  sorry

end NUMINAMATH_GPT_Edmund_earns_64_dollars_l1791_179188


namespace NUMINAMATH_GPT_least_possible_value_expression_l1791_179132

theorem least_possible_value_expression :
  ∃ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end NUMINAMATH_GPT_least_possible_value_expression_l1791_179132


namespace NUMINAMATH_GPT_area_of_quadrilateral_l1791_179141

theorem area_of_quadrilateral (θ : ℝ) (sin_θ : Real.sin θ = 4/5) (b1 b2 : ℝ) (h: ℝ) (base1 : b1 = 14) (base2 : b2 = 20) (height : h = 8) : 
  (1 / 2) * (b1 + b2) * h = 136 := by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l1791_179141


namespace NUMINAMATH_GPT_num_real_a_with_int_roots_l1791_179196

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end NUMINAMATH_GPT_num_real_a_with_int_roots_l1791_179196


namespace NUMINAMATH_GPT_function_passes_through_point_l1791_179137

theorem function_passes_through_point (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, -1) ∧ ∀ x : ℝ, (y = a^(x-1) - 2) → y = -1 := by
  sorry

end NUMINAMATH_GPT_function_passes_through_point_l1791_179137


namespace NUMINAMATH_GPT_oranges_count_l1791_179155

noncomputable def initial_oranges (O : ℕ) : Prop :=
  let apples := 14
  let blueberries := 6
  let remaining_fruits := 26
  13 + (O - 1) + 5 = remaining_fruits

theorem oranges_count (O : ℕ) (h : initial_oranges O) : O = 9 :=
by
  have eq : 13 + (O - 1) + 5 = 26 := h
  -- Simplify the equation to find O
  sorry

end NUMINAMATH_GPT_oranges_count_l1791_179155


namespace NUMINAMATH_GPT_find_numbers_satisfying_conditions_l1791_179122

theorem find_numbers_satisfying_conditions (x y z : ℝ)
(h1 : x + y + z = 11 / 18)
(h2 : 1 / x + 1 / y + 1 / z = 18)
(h3 : 2 / y = 1 / x + 1 / z) :
x = 1 / 9 ∧ y = 1 / 6 ∧ z = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_numbers_satisfying_conditions_l1791_179122


namespace NUMINAMATH_GPT_dice_probability_theorem_l1791_179139

def at_least_three_same_value_probability (num_dice : ℕ) (num_sides : ℕ) : ℚ :=
  if num_dice = 5 ∧ num_sides = 10 then
    -- Calculating the probability
    (81 / 10000) + (9 / 20000) + (1 / 10000)
  else
    0

theorem dice_probability_theorem :
  at_least_three_same_value_probability 5 10 = 173 / 20000 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_theorem_l1791_179139


namespace NUMINAMATH_GPT_greatest_product_obtainable_l1791_179104

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_obtainable_l1791_179104


namespace NUMINAMATH_GPT_min_distinct_integers_for_ap_and_gp_l1791_179134

theorem min_distinct_integers_for_ap_and_gp (n : ℕ) :
  (∀ (b q a d : ℤ), b ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 →
    (∃ (i : ℕ), i < 5 → b * (q ^ i) = a + i * d) ∧ 
    (∃ (j : ℕ), j < 5 → b * (q ^ j) ≠ a + j * d) ↔ n ≥ 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_distinct_integers_for_ap_and_gp_l1791_179134


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l1791_179124

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l1791_179124


namespace NUMINAMATH_GPT_second_machine_copies_per_minute_l1791_179187

-- Definitions based on conditions
def copies_per_minute_first := 35
def total_copies_half_hour := 3300
def time_minutes := 30

-- Theorem statement
theorem second_machine_copies_per_minute : 
  ∃ (x : ℕ), (copies_per_minute_first * time_minutes + x * time_minutes = total_copies_half_hour) ∧ (x = 75) := by
  sorry

end NUMINAMATH_GPT_second_machine_copies_per_minute_l1791_179187


namespace NUMINAMATH_GPT_trajectory_line_or_hyperbola_l1791_179173

theorem trajectory_line_or_hyperbola
  (a b : ℝ)
  (ab_pos : a * b > 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b) :
  (∃ s t : ℝ, f (s-t) * f (s+t) = (f s)^2) →
  (∃ s t : ℝ, ((t = 0) ∨ (a * t^2 - 2 * a * s^2 + 2 * b = 0))) → true := sorry

end NUMINAMATH_GPT_trajectory_line_or_hyperbola_l1791_179173


namespace NUMINAMATH_GPT_ratio_goats_sold_to_total_l1791_179164

-- Define the conditions
variables (G S : ℕ) (total_revenue goat_sold : ℕ)
-- The ratio of goats to sheep is 5:7
axiom ratio_goats_to_sheep : G = (5/7) * S
-- The total number of sheep and goats is 360
axiom total_animals : G + S = 360
-- Mr. Mathews makes $7200 from selling some goats and 2/3 of the sheep
axiom selling_conditions : 40 * goat_sold + 30 * (2/3) * S = 7200

-- Prove the ratio of the number of goats sold to the total number of goats
theorem ratio_goats_sold_to_total : goat_sold / G = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_goats_sold_to_total_l1791_179164


namespace NUMINAMATH_GPT_coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l1791_179114

-- Definition of the conditions
def condition_1 (Px: ℝ) (Py: ℝ) : Prop := Px = 0

def condition_2 (Px: ℝ) (Py: ℝ) : Prop := Py = Px + 3

def condition_3 (Px: ℝ) (Py: ℝ) : Prop := 
  abs Py = 2 ∧ Px > 0 ∧ Py < 0

-- Proof problem for condition 1
theorem coordinate_P_condition_1 : ∃ (Px Py: ℝ), condition_1 Px Py ∧ Px = 0 ∧ Py = -7 := 
  sorry

-- Proof problem for condition 2
theorem coordinate_P_condition_2 : ∃ (Px Py: ℝ), condition_2 Px Py ∧ Px = 10 ∧ Py = 13 :=
  sorry

-- Proof problem for condition 3
theorem coordinate_P_condition_3 : ∃ (Px Py: ℝ), condition_3 Px Py ∧ Px = 5/2 ∧ Py = -2 :=
  sorry

end NUMINAMATH_GPT_coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l1791_179114


namespace NUMINAMATH_GPT_ratio_of_perimeter_to_length_XY_l1791_179110

noncomputable def XY : ℝ := 17
noncomputable def XZ : ℝ := 8
noncomputable def YZ : ℝ := 15
noncomputable def ZD : ℝ := 240 / 17

-- Defining the perimeter P
noncomputable def P : ℝ := 17 + 2 * (240 / 17)

-- Finally, the statement with the ratio in the desired form
theorem ratio_of_perimeter_to_length_XY : 
  (P / XY) = (654 / 289) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeter_to_length_XY_l1791_179110


namespace NUMINAMATH_GPT_theater_loss_l1791_179165

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end NUMINAMATH_GPT_theater_loss_l1791_179165


namespace NUMINAMATH_GPT_find_number_l1791_179191

theorem find_number (x : ℤ) (h : x - 254 + 329 = 695) : x = 620 :=
sorry

end NUMINAMATH_GPT_find_number_l1791_179191


namespace NUMINAMATH_GPT_solve_equation_l1791_179116

theorem solve_equation :
  ∀ x : ℝ, 
    (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 4)) ↔ 
      (x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2) := 
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_l1791_179116


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1791_179167

variable (m x : ℝ)

-- Defining the condition
def inequality (m x : ℝ) := x^2 - (2 * m - 1) * x + m^2 - m > 0

-- Problem statement
theorem solution_set_of_inequality (h : inequality m x) : x < m-1 ∨ x > m :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1791_179167


namespace NUMINAMATH_GPT_vitamin_C_in_apple_juice_l1791_179174

theorem vitamin_C_in_apple_juice (A O : ℝ) 
  (h₁ : A + O = 185) 
  (h₂ : 2 * A + 3 * O = 452) :
  A = 103 :=
sorry

end NUMINAMATH_GPT_vitamin_C_in_apple_juice_l1791_179174


namespace NUMINAMATH_GPT_box_third_dimension_l1791_179186

theorem box_third_dimension (num_cubes : ℕ) (cube_volume box_vol : ℝ) (dim1 dim2 h : ℝ) (h_num_cubes : num_cubes = 24) (h_cube_volume : cube_volume = 27) (h_dim1 : dim1 = 9) (h_dim2 : dim2 = 12) (h_box_vol : box_vol = num_cubes * cube_volume) :
  box_vol = dim1 * dim2 * h → h = 6 := 
by
  sorry

end NUMINAMATH_GPT_box_third_dimension_l1791_179186


namespace NUMINAMATH_GPT_remainder_of_N_mod_D_l1791_179150

/-- The given number N and the divisor 252 defined in terms of its prime factors. -/
def N : ℕ := 9876543210123456789
def D : ℕ := 252

/-- The remainders of N modulo 4, 9, and 7 as given in the solution -/
def N_mod_4 : ℕ := 1
def N_mod_9 : ℕ := 0
def N_mod_7 : ℕ := 6

theorem remainder_of_N_mod_D :
  N % D = 27 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_N_mod_D_l1791_179150


namespace NUMINAMATH_GPT_flight_duration_l1791_179100

theorem flight_duration (h m : ℕ) (Hh : h = 2) (Hm : m = 32) : h + m = 34 := by
  sorry

end NUMINAMATH_GPT_flight_duration_l1791_179100
