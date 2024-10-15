import Mathlib

namespace NUMINAMATH_GPT_fraction_addition_l1763_176362

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end NUMINAMATH_GPT_fraction_addition_l1763_176362


namespace NUMINAMATH_GPT_total_volume_of_quiche_l1763_176305

def raw_spinach_volume : ℝ := 40
def cooked_volume_percentage : ℝ := 0.20
def cream_cheese_volume : ℝ := 6
def eggs_volume : ℝ := 4

theorem total_volume_of_quiche :
  raw_spinach_volume * cooked_volume_percentage + cream_cheese_volume + eggs_volume = 18 := by
  sorry

end NUMINAMATH_GPT_total_volume_of_quiche_l1763_176305


namespace NUMINAMATH_GPT_neighbors_have_even_total_bells_not_always_divisible_by_3_l1763_176392

def num_bushes : ℕ := 19

def is_neighbor (circ : ℕ → ℕ) (i j : ℕ) : Prop := 
  if i = num_bushes - 1 then j = 0
  else j = i + 1

-- Part (a)
theorem neighbors_have_even_total_bells (bells : Fin num_bushes → ℕ) :
  ∃ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 2 = 0 := sorry

-- Part (b)
theorem not_always_divisible_by_3 (bells : Fin num_bushes → ℕ) :
  ¬ (∀ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 3 = 0) := sorry

end NUMINAMATH_GPT_neighbors_have_even_total_bells_not_always_divisible_by_3_l1763_176392


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1763_176347

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1763_176347


namespace NUMINAMATH_GPT_mole_fractions_C4H8O2_l1763_176379

/-- 
Given:
- The molecular formula of C4H8O2,
- 4 moles of carbon (C) atoms,
- 8 moles of hydrogen (H) atoms,
- 2 moles of oxygen (O) atoms.

Prove that:
The mole fractions of each element in C4H8O2 are:
- Carbon (C): 2/7
- Hydrogen (H): 4/7
- Oxygen (O): 1/7
--/
theorem mole_fractions_C4H8O2 :
  let m_C := 4
  let m_H := 8
  let m_O := 2
  let total_moles := m_C + m_H + m_O
  let mole_fraction_C := m_C / total_moles
  let mole_fraction_H := m_H / total_moles
  let mole_fraction_O := m_O / total_moles
  mole_fraction_C = 2 / 7 ∧ mole_fraction_H = 4 / 7 ∧ mole_fraction_O = 1 / 7 := by
  sorry

end NUMINAMATH_GPT_mole_fractions_C4H8O2_l1763_176379


namespace NUMINAMATH_GPT_proposition_does_not_hold_at_2_l1763_176358

variable (P : ℕ+ → Prop)
open Nat

theorem proposition_does_not_hold_at_2
  (h₁ : ¬ P 3)
  (h₂ : ∀ k : ℕ+, P k → P (k + 1)) :
  ¬ P 2 :=
by
  sorry

end NUMINAMATH_GPT_proposition_does_not_hold_at_2_l1763_176358


namespace NUMINAMATH_GPT_find_original_number_l1763_176312

theorem find_original_number (x : ℤ) (h : (x + 19) % 25 = 0) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1763_176312


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l1763_176340

theorem largest_divisor_of_expression (n : ℤ) : ∃ k : ℤ, k = 6 ∧ (n^3 - n + 15) % k = 0 := 
by
  use 6
  sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l1763_176340


namespace NUMINAMATH_GPT_sum_of_ratios_is_3_or_neg3_l1763_176349

theorem sum_of_ratios_is_3_or_neg3 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a / b + b / c + c / a : ℚ).den = 1 ) 
  (h5 : (b / a + c / b + a / c : ℚ).den = 1) :
  (a / b + b / c + c / a = 3 ∨ a / b + b / c + c / a = -3) ∧ 
  (b / a + c / b + a / c = 3 ∨ b / a + c / b + a / c = -3) := 
sorry

end NUMINAMATH_GPT_sum_of_ratios_is_3_or_neg3_l1763_176349


namespace NUMINAMATH_GPT_reciprocal_neg_one_over_2023_l1763_176316

theorem reciprocal_neg_one_over_2023 : 1 / (- (1 / 2023 : ℝ)) = -2023 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_one_over_2023_l1763_176316


namespace NUMINAMATH_GPT_sally_balloon_count_l1763_176346

theorem sally_balloon_count 
  (joan_balloons : Nat)
  (jessica_balloons : Nat)
  (total_balloons : Nat)
  (sally_balloons : Nat)
  (h_joan : joan_balloons = 9)
  (h_jessica : jessica_balloons = 2)
  (h_total : total_balloons = 16)
  (h_eq : total_balloons = joan_balloons + jessica_balloons + sally_balloons) : 
  sally_balloons = 5 :=
by
  sorry

end NUMINAMATH_GPT_sally_balloon_count_l1763_176346


namespace NUMINAMATH_GPT_polynomial_abs_value_at_neg_one_l1763_176398

theorem polynomial_abs_value_at_neg_one:
  ∃ g : Polynomial ℝ, 
  (∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g.eval x| = 15) → 
  |g.eval (-1)| = 75 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_abs_value_at_neg_one_l1763_176398


namespace NUMINAMATH_GPT_fraction_irreducible_l1763_176384

theorem fraction_irreducible (a b c d : ℤ) (h : a * d - b * c = 1) : ∀ m : ℤ, m > 1 → ¬ (m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d)) :=
by sorry

end NUMINAMATH_GPT_fraction_irreducible_l1763_176384


namespace NUMINAMATH_GPT_solve_for_a_l1763_176350

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1763_176350


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_max_l1763_176310

theorem sum_arithmetic_sequence_max (d : ℝ) (a : ℕ → ℝ) 
  (h1 : d < 0) (h2 : (a 1)^2 = (a 13)^2) :
  ∃ n, n = 6 ∨ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_max_l1763_176310


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1763_176359

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ),
  -- Condition: axes are parallel to the coordinate axes (implicitly given by tangency points).
  a = 3 ∧
  b = 2 ∧
  c = Real.sqrt (a^2 - b^2) ∧
  2 * c = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1763_176359


namespace NUMINAMATH_GPT_xyz_divisible_by_55_l1763_176393

-- Definitions and conditions from part (a)
variables (x y z a b c : ℤ)
variable (h1 : x^2 + y^2 = a^2)
variable (h2 : y^2 + z^2 = b^2)
variable (h3 : z^2 + x^2 = c^2)

-- The final statement to prove that xyz is divisible by 55
theorem xyz_divisible_by_55 : 55 ∣ x * y * z := 
by sorry

end NUMINAMATH_GPT_xyz_divisible_by_55_l1763_176393


namespace NUMINAMATH_GPT_area_of_square_BDEF_l1763_176381

noncomputable def right_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
∃ (AB BC AC : ℝ), AB = 15 ∧ BC = 20 ∧ AC = Real.sqrt (AB^2 + BC^2)

noncomputable def is_square (B D E F : Type*) [MetricSpace B] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
∃ (BD DE EF FB : ℝ), BD = DE ∧ DE = EF ∧ EF = FB

noncomputable def height_of_triangle (E H M : Type*) [MetricSpace E] [MetricSpace H] [MetricSpace M] : Prop :=
∃ (EH : ℝ), EH = 2

theorem area_of_square_BDEF (A B C D E F H M N : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F]
  [MetricSpace H] [MetricSpace M] [MetricSpace N]
  (H1 : right_triangle A B C)
  (H2 : is_square B D E F)
  (H3 : height_of_triangle E H M) :
  ∃ (area : ℝ), area = 100 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_BDEF_l1763_176381


namespace NUMINAMATH_GPT_sqrt_expression_identity_l1763_176355

theorem sqrt_expression_identity :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_identity_l1763_176355


namespace NUMINAMATH_GPT_CarlaDailyItems_l1763_176377

theorem CarlaDailyItems (leaves bugs days : ℕ) 
  (h_leaves : leaves = 30) 
  (h_bugs : bugs = 20) 
  (h_days : days = 10) : 
  (leaves + bugs) / days = 5 := 
by 
  sorry

end NUMINAMATH_GPT_CarlaDailyItems_l1763_176377


namespace NUMINAMATH_GPT_seashells_total_l1763_176399

def seashells :=
  let sam_seashells := 18
  let mary_seashells := 47
  sam_seashells + mary_seashells

theorem seashells_total : seashells = 65 := by
  sorry

end NUMINAMATH_GPT_seashells_total_l1763_176399


namespace NUMINAMATH_GPT_books_finished_correct_l1763_176313

def miles_traveled : ℕ := 6760
def miles_per_book : ℕ := 450
def books_finished (miles_traveled miles_per_book : ℕ) : ℕ :=
  miles_traveled / miles_per_book

theorem books_finished_correct :
  books_finished miles_traveled miles_per_book = 15 :=
by
  -- The steps of the proof would go here
  sorry

end NUMINAMATH_GPT_books_finished_correct_l1763_176313


namespace NUMINAMATH_GPT_jason_initial_speed_correct_l1763_176322

noncomputable def jason_initial_speed (d : ℝ) (t1 t2 : ℝ) (v2 : ℝ) : ℝ :=
  let t_total := t1 + t2
  let d2 := v2 * t2
  let d1 := d - d2
  let v1 := d1 / t1
  v1

theorem jason_initial_speed_correct :
  jason_initial_speed 120 0.5 1 90 = 60 := 
by 
  sorry

end NUMINAMATH_GPT_jason_initial_speed_correct_l1763_176322


namespace NUMINAMATH_GPT_A_can_finish_remaining_work_in_4_days_l1763_176300

theorem A_can_finish_remaining_work_in_4_days
  (A_days : ℕ) (B_days : ℕ) (B_worked_days : ℕ) : 
  A_days = 12 → B_days = 15 → B_worked_days = 10 → 
  (4 * (1 / A_days) = 1 / 3 - B_worked_days * (1 / B_days)) :=
by
  intros hA hB hBwork
  sorry

end NUMINAMATH_GPT_A_can_finish_remaining_work_in_4_days_l1763_176300


namespace NUMINAMATH_GPT_element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l1763_176311

-- defining element, nuclide, and valence based on protons, neutrons, and electrons
def Element (protons : ℕ) := protons
def Nuclide (protons neutrons : ℕ) := (protons, neutrons)
def ChemicalProperties (outermostElectrons : ℕ) := outermostElectrons
def HighestPositiveValence (mainGroupNum : ℕ) := mainGroupNum

-- The proof problems as Lean theorems
theorem element_type_determined_by_protons (protons : ℕ) :
  Element protons = protons := sorry

theorem nuclide_type_determined_by_protons_neutrons (protons neutrons : ℕ) :
  Nuclide protons neutrons = (protons, neutrons) := sorry

theorem chemical_properties_determined_by_outermost_electrons (outermostElectrons : ℕ) :
  ChemicalProperties outermostElectrons = outermostElectrons := sorry
  
theorem highest_positive_valence_determined_by_main_group_num (mainGroupNum : ℕ) :
  HighestPositiveValence mainGroupNum = mainGroupNum := sorry

end NUMINAMATH_GPT_element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l1763_176311


namespace NUMINAMATH_GPT_tray_contains_40_brownies_l1763_176330

-- Definitions based on conditions
def tray_length : ℝ := 24
def tray_width : ℝ := 15
def brownie_length : ℝ := 3
def brownie_width : ℝ := 3

-- The mathematical statement to prove
theorem tray_contains_40_brownies :
  (tray_length * tray_width) / (brownie_length * brownie_width) = 40 :=
by
  sorry

end NUMINAMATH_GPT_tray_contains_40_brownies_l1763_176330


namespace NUMINAMATH_GPT_find_X_l1763_176352

theorem find_X : ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ X = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_find_X_l1763_176352


namespace NUMINAMATH_GPT_find_integer_l1763_176306

theorem find_integer (n : ℤ) (h1 : n + 10 > 11) (h2 : -4 * n > -12) : 
  n = 2 :=
sorry

end NUMINAMATH_GPT_find_integer_l1763_176306


namespace NUMINAMATH_GPT_current_population_correct_l1763_176353

def initial_population : ℕ := 4079
def percentage_died : ℕ := 5
def percentage_left : ℕ := 15

def calculate_current_population (initial_population : ℕ) (percentage_died : ℕ) (percentage_left : ℕ) : ℕ :=
  let died := (initial_population * percentage_died) / 100
  let remaining_after_bombardment := initial_population - died
  let left := (remaining_after_bombardment * percentage_left) / 100
  remaining_after_bombardment - left

theorem current_population_correct : calculate_current_population initial_population percentage_died percentage_left = 3295 :=
  by
  unfold calculate_current_population
  sorry

end NUMINAMATH_GPT_current_population_correct_l1763_176353


namespace NUMINAMATH_GPT_part_1_part_2_part_3_l1763_176319

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end NUMINAMATH_GPT_part_1_part_2_part_3_l1763_176319


namespace NUMINAMATH_GPT_max_area_ABC_l1763_176332

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end NUMINAMATH_GPT_max_area_ABC_l1763_176332


namespace NUMINAMATH_GPT_sixth_term_sequence_l1763_176387

theorem sixth_term_sequence (a b c d : ℚ)
  (h1 : a = 1/4 * (3 + b))
  (h2 : b = 1/4 * (a + c))
  (h3 : c = 1/4 * (b + 48))
  (h4 : 48 = 1/4 * (c + d)) :
  d = 2001 / 14 :=
sorry

end NUMINAMATH_GPT_sixth_term_sequence_l1763_176387


namespace NUMINAMATH_GPT_expression1_expression2_expression3_expression4_l1763_176394

theorem expression1 : 12 - (-10) + 7 = 29 := 
by
  sorry

theorem expression2 : 1 + (-2) * abs (-2 - 3) - 5 = -14 :=
by
  sorry

theorem expression3 : (-8 * (-1 / 6 + 3 / 4 - 1 / 12)) / (1 / 6) = -24 :=
by
  sorry

theorem expression4 : -1 ^ 2 - (2 - (-2) ^ 3) / (-2 / 5) * (5 / 2) = 123 / 2 := 
by
  sorry

end NUMINAMATH_GPT_expression1_expression2_expression3_expression4_l1763_176394


namespace NUMINAMATH_GPT_sum_first_19_terms_l1763_176380

variable {α : Type} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + n * d

def sum_of_arithmetic_sequence (a d : α) (n : ℕ) : α := (n : α) / 2 * (2 * a + (n - 1) * d)

theorem sum_first_19_terms (a d : α) 
  (h1 : ∀ n, arithmetic_sequence a d (2 + n) + arithmetic_sequence a d (16 + n) = 10)
  (S19 : α) :
  sum_of_arithmetic_sequence a d 19 = 95 := by
  sorry

end NUMINAMATH_GPT_sum_first_19_terms_l1763_176380


namespace NUMINAMATH_GPT_ram_krish_task_completion_l1763_176383

theorem ram_krish_task_completion
  (ram_days : ℝ)
  (krish_efficiency_factor : ℝ)
  (task_time : ℝ) 
  (H1 : krish_efficiency_factor = 2)
  (H2 : ram_days = 27) 
  (H3 : task_time = 9) :
  (1 / task_time) = (1 / ram_days + 1 / (ram_days / krish_efficiency_factor)) := 
sorry

end NUMINAMATH_GPT_ram_krish_task_completion_l1763_176383


namespace NUMINAMATH_GPT_number_of_slices_l1763_176336

theorem number_of_slices 
  (pepperoni ham sausage total_meat pieces_per_slice : ℕ)
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : total_meat = pepperoni + ham + sausage)
  (h5 : pieces_per_slice = 22) :
  total_meat / pieces_per_slice = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_slices_l1763_176336


namespace NUMINAMATH_GPT_sum_digits_of_three_digit_numbers_l1763_176331

theorem sum_digits_of_three_digit_numbers (a c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hc : 1 ≤ c ∧ c < 10) 
  (h1 : (300 + 10 * a + 7) + 414 = 700 + 10 * c + 1)
  (h2 : ∃ k : ℤ, 700 + 10 * c + 1 = 11 * k) :
  a + c = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_of_three_digit_numbers_l1763_176331


namespace NUMINAMATH_GPT_sum_of_digits_joeys_age_l1763_176367

-- Given conditions
variables (C : ℕ) (J : ℕ := C + 2) (Z : ℕ := 1)

-- Define the condition that the sum of Joey's and Chloe's ages will be an integral multiple of Zoe's age.
def sum_is_multiple_of_zoe (n : ℕ) : Prop :=
  ∃ k : ℕ, (J + C) = k * Z

-- Define the problem of finding the sum of digits the first time Joey's age alone is a multiple of Zoe's age.
def sum_of_digits_first_multiple (J Z : ℕ) : ℕ :=
  (J / 10) + (J % 10)

-- The theorem we need to prove
theorem sum_of_digits_joeys_age : (sum_of_digits_first_multiple J Z = 1) :=
sorry

end NUMINAMATH_GPT_sum_of_digits_joeys_age_l1763_176367


namespace NUMINAMATH_GPT_fraction_reducible_l1763_176307

theorem fraction_reducible (l : ℤ) : ∃ d : ℤ, d ≠ 1 ∧ d > 0 ∧ d = gcd (5 * l + 6) (8 * l + 7) := by 
  use 13
  sorry

end NUMINAMATH_GPT_fraction_reducible_l1763_176307


namespace NUMINAMATH_GPT_bruce_total_payment_l1763_176342

def grapes_quantity : ℕ := 8
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_grapes : ℕ := grapes_quantity * grapes_rate
def cost_mangoes : ℕ := mangoes_quantity * mangoes_rate
def total_cost : ℕ := cost_grapes + cost_mangoes

theorem bruce_total_payment : total_cost = 1055 := by
  sorry

end NUMINAMATH_GPT_bruce_total_payment_l1763_176342


namespace NUMINAMATH_GPT_exists_triangle_cut_into_2005_congruent_l1763_176321

theorem exists_triangle_cut_into_2005_congruent :
  ∃ (Δ : Type) (a b c : Δ → ℝ )
  (h₁ : a^2 + b^2 = c^2) (h₂ : a * b / 2 = 2005 / 2),
  true :=
sorry

end NUMINAMATH_GPT_exists_triangle_cut_into_2005_congruent_l1763_176321


namespace NUMINAMATH_GPT_eq_inf_solutions_l1763_176337

theorem eq_inf_solutions (a b : ℝ) : 
    (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -(4 / 3) * a := by
  sorry

end NUMINAMATH_GPT_eq_inf_solutions_l1763_176337


namespace NUMINAMATH_GPT_prime_quadratic_roots_l1763_176318

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_integer_roots (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, (a * x * x + b * x + c = 0) ∧ (a * y * y + b * y + c = 0)

theorem prime_quadratic_roots (p : ℕ) (h_prime : is_prime p)
  (h_roots : has_integer_roots 1 (p : ℤ) (-444 * (p : ℤ))) :
  31 < p ∧ p ≤ 41 :=
sorry

end NUMINAMATH_GPT_prime_quadratic_roots_l1763_176318


namespace NUMINAMATH_GPT_evaluate_expression_l1763_176341

theorem evaluate_expression :
  2 ^ (0 ^ (1 ^ 9)) + ((2 ^ 0) ^ 1) ^ 9 = 2 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l1763_176341


namespace NUMINAMATH_GPT_LCM_14_21_l1763_176374

theorem LCM_14_21 : Nat.lcm 14 21 = 42 := 
by
  sorry

end NUMINAMATH_GPT_LCM_14_21_l1763_176374


namespace NUMINAMATH_GPT_scientific_notation_of_1500_l1763_176376

theorem scientific_notation_of_1500 :
  (1500 : ℝ) = 1.5 * 10^3 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_1500_l1763_176376


namespace NUMINAMATH_GPT_intersection_of_lines_l1763_176320

theorem intersection_of_lines : ∃ (x y : ℝ), (9 * x - 4 * y = 30) ∧ (7 * x + y = 11) ∧ (x = 2) ∧ (y = -3) := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1763_176320


namespace NUMINAMATH_GPT_problem_l1763_176390

theorem problem
  (a b : ℚ)
  (h1 : 3 * a + 5 * b = 47)
  (h2 : 7 * a + 2 * b = 52)
  : a + b = 35 / 3 :=
sorry

end NUMINAMATH_GPT_problem_l1763_176390


namespace NUMINAMATH_GPT_same_solution_eq_l1763_176317

theorem same_solution_eq (a b : ℤ) (x y : ℤ) 
  (h₁ : 4 * x + 3 * y = 11)
  (h₂ : a * x + b * y = -2)
  (h₃ : 3 * x - 5 * y = 1)
  (h₄ : b * x - a * y = 6) :
  (a + b) ^ 2023 = 0 := by
  sorry

end NUMINAMATH_GPT_same_solution_eq_l1763_176317


namespace NUMINAMATH_GPT_tiffany_mile_fraction_l1763_176369

/-- Tiffany's daily running fraction (x) for Wednesday, Thursday, and Friday must be 1/3
    such that both Billy and Tiffany run the same total miles over a week. --/
theorem tiffany_mile_fraction :
  ∃ x : ℚ, (3 * 1 + 1) = 1 + (3 * 2 + 3 * x) → x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_mile_fraction_l1763_176369


namespace NUMINAMATH_GPT_intersection_M_N_l1763_176397

def M : Set ℕ := { y | y < 6 }
def N : Set ℕ := {2, 3, 6}

theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1763_176397


namespace NUMINAMATH_GPT_sequence_solution_l1763_176388

theorem sequence_solution (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * a n - 2^n + 1) : a n = n * 2^(n-1) :=
sorry

end NUMINAMATH_GPT_sequence_solution_l1763_176388


namespace NUMINAMATH_GPT_cost_price_computer_table_l1763_176382

noncomputable def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem cost_price_computer_table (SP : ℝ) (CP : ℝ) (h : SP = 7967) (h2 : SP = 1.24 * CP) : 
  approx_eq CP 6424 0.01 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l1763_176382


namespace NUMINAMATH_GPT_convert_110110001_to_base4_l1763_176348

def binary_to_base4_conversion (b : ℕ) : ℕ :=
  -- assuming b is the binary representation of the number to be converted
  1 * 4^4 + 3 * 4^3 + 2 * 4^2 + 0 * 4^1 + 1 * 4^0

theorem convert_110110001_to_base4 : binary_to_base4_conversion 110110001 = 13201 :=
  sorry

end NUMINAMATH_GPT_convert_110110001_to_base4_l1763_176348


namespace NUMINAMATH_GPT_largest_real_solution_sum_l1763_176328

theorem largest_real_solution_sum (d e f : ℕ) (x : ℝ) (h : d = 13 ∧ e = 61 ∧ f = 0) : 
  (∃ d e f : ℕ, d + e + f = 74) ↔ 
  (n : ℝ) * n = (x - d)^2 ∧ 
  (∀ x : ℝ, 
    (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 13 * x - 6 → 
    n = x) :=
sorry

end NUMINAMATH_GPT_largest_real_solution_sum_l1763_176328


namespace NUMINAMATH_GPT_sad_girls_count_l1763_176361

-- Statement of the problem in Lean 4
theorem sad_girls_count :
  ∀ (total_children happy_children sad_children neither_happy_nor_sad children boys girls happy_boys boys_neither_happy_nor_sad : ℕ),
    total_children = 60 →
    happy_children = 30 →
    sad_children = 10 →
    neither_happy_nor_sad = 20 →
    children = total_children →
    boys = 19 →
    girls = total_children - boys →
    happy_boys = 6 →
    boys_neither_happy_nor_sad = 7 →
    girls = 41 →
    sad_children = 10 →
    (sad_children = 6 + (total_children - boys - girls - neither_happy_nor_sad - happy_children)) → 
    ∃ sad_girls, sad_girls = 4 := by
  sorry

end NUMINAMATH_GPT_sad_girls_count_l1763_176361


namespace NUMINAMATH_GPT_marcia_project_hours_l1763_176345

theorem marcia_project_hours (minutes_spent : ℕ) (minutes_per_hour : ℕ) 
  (h1 : minutes_spent = 300) 
  (h2 : minutes_per_hour = 60) : 
  (minutes_spent / minutes_per_hour) = 5 :=
by
  sorry

end NUMINAMATH_GPT_marcia_project_hours_l1763_176345


namespace NUMINAMATH_GPT_sin_arithmetic_sequence_l1763_176385

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end NUMINAMATH_GPT_sin_arithmetic_sequence_l1763_176385


namespace NUMINAMATH_GPT_time_per_window_l1763_176365

-- Definitions of the given conditions
def total_windows : ℕ := 10
def installed_windows : ℕ := 6
def remaining_windows := total_windows - installed_windows
def total_hours : ℕ := 20
def hours_per_window := total_hours / remaining_windows

-- The theorem we need to prove
theorem time_per_window : hours_per_window = 5 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_time_per_window_l1763_176365


namespace NUMINAMATH_GPT_Jean_average_speed_correct_l1763_176308

noncomputable def Jean_avg_speed_until_meet
    (total_distance : ℕ)
    (chantal_flat_distance : ℕ)
    (chantal_flat_speed : ℕ)
    (chantal_steep_distance : ℕ)
    (chantal_steep_ascend_speed : ℕ)
    (chantal_steep_descend_distance : ℕ)
    (chantal_steep_descend_speed : ℕ)
    (jean_meet_position_ratio : ℚ) : ℚ :=
  let chantal_flat_time := (chantal_flat_distance : ℚ) / chantal_flat_speed
  let chantal_steep_ascend_time := (chantal_steep_distance : ℚ) / chantal_steep_ascend_speed
  let chantal_steep_descend_time := (chantal_steep_descend_distance : ℚ) / chantal_steep_descend_speed
  let total_time_until_meet := chantal_flat_time + chantal_steep_ascend_time + chantal_steep_descend_time
  let jean_distance_until_meet := (jean_meet_position_ratio * chantal_steep_distance : ℚ) + chantal_flat_distance
  jean_distance_until_meet / total_time_until_meet

theorem Jean_average_speed_correct :
  Jean_avg_speed_until_meet 6 3 5 3 3 1 4 (1 / 3) = 80 / 37 :=
by
  sorry

end NUMINAMATH_GPT_Jean_average_speed_correct_l1763_176308


namespace NUMINAMATH_GPT_polynomial_coefficients_l1763_176356

theorem polynomial_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (2 * x - 1) * ((x + 1) ^ 7) = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + 
  (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 + (a 7) * x^7 + (a 8) * x^8) →
  (a 0 = -1) ∧
  (a 0 + a 2 + a 4 + a 6 + a 8 = 64) ∧
  (a 1 + 2 * (a 2) + 3 * (a 3) + 4 * (a 4) + 5 * (a 5) + 6 * (a 6) + 7 * (a 7) + 8 * (a 8) = 704) := by
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_l1763_176356


namespace NUMINAMATH_GPT_simplify_expression_l1763_176386

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1763_176386


namespace NUMINAMATH_GPT_find_integer_pairs_l1763_176334

theorem find_integer_pairs :
  {p : ℤ × ℤ | p.1 * (p.1 + 1) * (p.1 + 7) * (p.1 + 8) = p.2^2} =
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (0, 0), (-8, 0), (-4, -12), (-4, 12), (-1, 0), (-7, 0)} :=
sorry

end NUMINAMATH_GPT_find_integer_pairs_l1763_176334


namespace NUMINAMATH_GPT_trajectory_of_moving_circle_l1763_176325

def circle1 (x y : ℝ) := (x + 4) ^ 2 + y ^ 2 = 2
def circle2 (x y : ℝ) := (x - 4) ^ 2 + y ^ 2 = 2

theorem trajectory_of_moving_circle (x y : ℝ) : 
  (x = 0) ∨ (x ^ 2 / 2 - y ^ 2 / 14 = 1) := 
  sorry

end NUMINAMATH_GPT_trajectory_of_moving_circle_l1763_176325


namespace NUMINAMATH_GPT_bat_pattern_area_l1763_176315

-- Define the areas of the individual components
def area_large_square : ℕ := 8
def num_large_squares : ℕ := 2

def area_medium_square : ℕ := 4
def num_medium_squares : ℕ := 2

def area_triangle : ℕ := 1
def num_triangles : ℕ := 3

-- Define the total area calculation
def total_area : ℕ :=
  (num_large_squares * area_large_square) +
  (num_medium_squares * area_medium_square) +
  (num_triangles * area_triangle)

-- The theorem statement
theorem bat_pattern_area : total_area = 27 := by
  sorry

end NUMINAMATH_GPT_bat_pattern_area_l1763_176315


namespace NUMINAMATH_GPT_num_arithmetic_sequences_l1763_176363

theorem num_arithmetic_sequences (a d : ℕ) (n : ℕ) (h1 : n >= 3) (h2 : n * (2 * a + (n - 1) * d) = 2 * 97^2) :
  ∃ seqs : ℕ, seqs = 4 :=
by sorry

end NUMINAMATH_GPT_num_arithmetic_sequences_l1763_176363


namespace NUMINAMATH_GPT_mutually_exclusive_necessary_for_complementary_l1763_176375

variables {Ω : Type} -- Define the sample space type
variables (A1 A2 : Ω → Prop) -- Define the events as predicates over the sample space

-- Define mutually exclusive events
def mutually_exclusive (A1 A2 : Ω → Prop) : Prop :=
∀ ω, A1 ω → ¬ A2 ω

-- Define complementary events
def complementary (A1 A2 : Ω → Prop) : Prop :=
∀ ω, (A1 ω ↔ ¬ A2 ω)

-- The proof problem: Statement 1 is a necessary but not sufficient condition for Statement 2
theorem mutually_exclusive_necessary_for_complementary (A1 A2 : Ω → Prop) :
  (mutually_exclusive A1 A2) → (complementary A1 A2) → (mutually_exclusive A1 A2) ∧ ¬ (complementary A1 A2 → mutually_exclusive A1 A2) :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_necessary_for_complementary_l1763_176375


namespace NUMINAMATH_GPT_smallest_m_for_divisibility_l1763_176338

theorem smallest_m_for_divisibility : 
  ∃ (m : ℕ), 2^1990 ∣ 1989^m - 1 ∧ m = 2^1988 := 
sorry

end NUMINAMATH_GPT_smallest_m_for_divisibility_l1763_176338


namespace NUMINAMATH_GPT_cubic_root_conditions_l1763_176324

-- Define the cubic polynomial
def cubic (a b : ℝ) (x : ℝ) : ℝ := x^3 + a * x + b

-- Define a predicate for the cubic equation having exactly one real root
def has_one_real_root (a b : ℝ) : Prop :=
  ∀ y : ℝ, cubic a b y = 0 → ∃! x : ℝ, cubic a b x = 0

-- Theorem statement
theorem cubic_root_conditions (a b : ℝ) :
  (a = -3 ∧ b = -3) ∨ (a = -3 ∧ b > 2) ∨ (a = 0 ∧ b = 2) → has_one_real_root a b :=
sorry

end NUMINAMATH_GPT_cubic_root_conditions_l1763_176324


namespace NUMINAMATH_GPT_product_of_decimals_l1763_176360

def x : ℝ := 0.8
def y : ℝ := 0.12

theorem product_of_decimals : x * y = 0.096 :=
by
  sorry

end NUMINAMATH_GPT_product_of_decimals_l1763_176360


namespace NUMINAMATH_GPT_coffee_machine_price_l1763_176354

noncomputable def original_machine_price : ℝ :=
  let coffees_prior_cost_per_day := 2 * 4
  let new_coffees_cost_per_day := 3
  let daily_savings := coffees_prior_cost_per_day - new_coffees_cost_per_day
  let total_savings := 36 * daily_savings
  let discounted_price := total_savings
  let discount := 20
  discounted_price + discount

theorem coffee_machine_price
  (coffees_prior_cost_per_day : ℝ := 2 * 4)
  (new_coffees_cost_per_day : ℝ := 3)
  (daily_savings : ℝ := coffees_prior_cost_per_day - new_coffees_cost_per_day)
  (total_savings : ℝ := 36 * daily_savings)
  (discounted_price : ℝ := total_savings)
  (discount : ℝ := 20) :
  original_machine_price = 200 :=
by
  sorry

end NUMINAMATH_GPT_coffee_machine_price_l1763_176354


namespace NUMINAMATH_GPT_no_solution_for_x_l1763_176333

theorem no_solution_for_x (a : ℝ) (h : a ≤ 8) : ¬ ∃ x : ℝ, |x - 5| + |x + 3| < a :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_x_l1763_176333


namespace NUMINAMATH_GPT_find_x_l1763_176364

variable (x : ℕ)

def f (x : ℕ) : ℕ := 2 * x + 5
def g (y : ℕ) : ℕ := 3 * y

theorem find_x (h : g (f x) = 123) : x = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1763_176364


namespace NUMINAMATH_GPT_distinct_fib_sum_2017_l1763_176344

-- Define the Fibonacci sequence as given.
def fib : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => (fib (n+1)) + (fib n)

-- Define the predicate for representing a number as a sum of distinct Fibonacci numbers.
def can_be_written_as_sum_of_distinct_fibs (n : ℕ) : Prop :=
  ∃ s : Finset ℕ, (s.sum fib = n) ∧ (∀ (i j : ℕ), i ≠ j → i ∉ s → j ∉ s)

theorem distinct_fib_sum_2017 : ∃! s : Finset ℕ, s.sum fib = 2017 ∧ (∀ (i j : ℕ), i ≠ j → i ≠ j → i ∉ s → j ∉ s) :=
sorry

end NUMINAMATH_GPT_distinct_fib_sum_2017_l1763_176344


namespace NUMINAMATH_GPT_smallest_expression_l1763_176335

theorem smallest_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (y / x = 1 / 2) ∧ (y / x < x + y) ∧ (y / x < x * y) ∧ (y / x < x - y) ∧ (y / x < x / y) :=
by
  -- The proof is to be filled by the user
  sorry

end NUMINAMATH_GPT_smallest_expression_l1763_176335


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1763_176323

theorem sufficient_but_not_necessary_condition (a b : ℝ) : (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(∀ a b, a^2 + b ≥ 0 → b ≥ 0) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1763_176323


namespace NUMINAMATH_GPT_jeffery_fish_count_l1763_176329

variable (J R Y : ℕ)

theorem jeffery_fish_count :
  (R = 3 * J) → (Y = 2 * R) → (J + R + Y = 100) → (Y = 60) :=
by
  intros hR hY hTotal
  have h1 : R = 3 * J := hR
  have h2 : Y = 2 * R := hY
  rw [h1, h2] at hTotal
  sorry

end NUMINAMATH_GPT_jeffery_fish_count_l1763_176329


namespace NUMINAMATH_GPT_total_earnings_first_two_weeks_l1763_176303

-- Conditions
variable (x : ℝ)  -- Xenia's hourly wage
variable (earnings_first_week : ℝ := 12 * x)  -- Earnings in the first week
variable (earnings_second_week : ℝ := 20 * x)  -- Earnings in the second week

-- Xenia earned $36 more in the second week than in the first
axiom h1 : earnings_second_week = earnings_first_week + 36

-- Proof statement
theorem total_earnings_first_two_weeks : earnings_first_week + earnings_second_week = 144 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_earnings_first_two_weeks_l1763_176303


namespace NUMINAMATH_GPT_soda_preference_respondents_l1763_176304

noncomputable def fraction_of_soda (angle_soda : ℝ) (total_angle : ℝ) : ℝ :=
  angle_soda / total_angle

noncomputable def number_of_soda_preference (total_people : ℕ) (fraction : ℝ) : ℝ :=
  total_people * fraction

theorem soda_preference_respondents (total_people : ℕ) (angle_soda : ℝ) (total_angle : ℝ) : 
  total_people = 520 → angle_soda = 298 → total_angle = 360 → 
  number_of_soda_preference total_people (fraction_of_soda angle_soda total_angle) = 429 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold fraction_of_soda number_of_soda_preference
  -- further calculation steps
  sorry

end NUMINAMATH_GPT_soda_preference_respondents_l1763_176304


namespace NUMINAMATH_GPT_range_of_a_l1763_176301

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = Real.exp x + a * x) ∧ (∃ x, 0 < x ∧ (DifferentiableAt ℝ f x) ∧ (deriv f x = 0)) → a < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1763_176301


namespace NUMINAMATH_GPT_find_x_l1763_176314

theorem find_x (x : ℝ) (h : (x * 74) / 30 = 1938.8) : x = 786 := by
  sorry

end NUMINAMATH_GPT_find_x_l1763_176314


namespace NUMINAMATH_GPT_parallel_vectors_l1763_176389

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

theorem parallel_vectors (m : ℝ) (h : (1 : ℝ) / (-1 : ℝ) = (2 : ℝ) / m) : m = -2 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_l1763_176389


namespace NUMINAMATH_GPT_janelle_total_marbles_l1763_176371

def initial_green_marbles := 26
def bags_of_blue_marbles := 12
def marbles_per_bag := 15
def gift_red_marbles := 7
def gift_green_marbles := 9
def gift_blue_marbles := 12
def gift_red_marbles_given := 3
def returned_blue_marbles := 8

theorem janelle_total_marbles :
  let total_green := initial_green_marbles - gift_green_marbles
  let total_blue := (bags_of_blue_marbles * marbles_per_bag) - gift_blue_marbles + returned_blue_marbles
  let total_red := gift_red_marbles - gift_red_marbles_given
  total_green + total_blue + total_red = 197 :=
by
  sorry

end NUMINAMATH_GPT_janelle_total_marbles_l1763_176371


namespace NUMINAMATH_GPT_mass_percentage_O_in_C6H8O6_l1763_176396

theorem mass_percentage_O_in_C6H8O6 :
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  mass_percentage_O = 72.67 :=
by
  -- Definitions
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  -- Proof
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_C6H8O6_l1763_176396


namespace NUMINAMATH_GPT_complement_U_A_l1763_176339

-- Definitions of U and A based on problem conditions
def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 2}

-- Definition of the complement in Lean
def complement (A B : Set ℤ) : Set ℤ := {x | x ∈ A ∧ x ∉ B}

-- The main statement to be proved
theorem complement_U_A :
  complement U A = {1, 3} :=
sorry

end NUMINAMATH_GPT_complement_U_A_l1763_176339


namespace NUMINAMATH_GPT_find_r_squared_l1763_176343

noncomputable def parabola_intersect_circle_radius_squared : Prop :=
  ∀ (x y : ℝ), y = (x - 1)^2 ∧ x - 3 = (y + 2)^2 → (x - 3/2)^2 + (y + 3/2)^2 = 1/2

theorem find_r_squared : parabola_intersect_circle_radius_squared :=
sorry

end NUMINAMATH_GPT_find_r_squared_l1763_176343


namespace NUMINAMATH_GPT_inequality_interval_l1763_176370

theorem inequality_interval : ∀ x : ℝ, (x^2 - 3 * x - 4 < 0) ↔ (-1 < x ∧ x < 4) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_inequality_interval_l1763_176370


namespace NUMINAMATH_GPT_extra_days_per_grade_below_b_l1763_176366

theorem extra_days_per_grade_below_b :
  ∀ (total_days lying_days grades_below_B : ℕ), 
  total_days = 26 → lying_days = 14 → grades_below_B = 4 → 
  (total_days - lying_days) / grades_below_B = 3 :=
by
  -- conditions and steps of the proof will be here
  sorry

end NUMINAMATH_GPT_extra_days_per_grade_below_b_l1763_176366


namespace NUMINAMATH_GPT_upper_side_length_trapezoid_l1763_176326

theorem upper_side_length_trapezoid
  (L U : ℝ) 
  (h : ℝ := 8) 
  (A : ℝ := 72) 
  (cond1 : U = L - 6)
  (cond2 : 1/2 * (L + U) * h = A) :
  U = 6 := 
by 
  sorry

end NUMINAMATH_GPT_upper_side_length_trapezoid_l1763_176326


namespace NUMINAMATH_GPT_division_addition_rational_eq_l1763_176395

theorem division_addition_rational_eq :
  (3 / 7 / 4) + (1 / 2) = 17 / 28 :=
by
  sorry

end NUMINAMATH_GPT_division_addition_rational_eq_l1763_176395


namespace NUMINAMATH_GPT_cos_double_angle_l1763_176368

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 5) : Real.cos (2 * α) = 23 / 25 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1763_176368


namespace NUMINAMATH_GPT_hiking_trip_time_l1763_176302

noncomputable def R_up : ℝ := 7
noncomputable def R_down : ℝ := 1.5 * R_up
noncomputable def Distance_down : ℝ := 21
noncomputable def T_down : ℝ := Distance_down / R_down
noncomputable def T_up : ℝ := T_down

theorem hiking_trip_time :
  T_up = 2 := by
      sorry

end NUMINAMATH_GPT_hiking_trip_time_l1763_176302


namespace NUMINAMATH_GPT_total_distance_of_journey_l1763_176373

variables (x v : ℝ)
variable (d : ℝ := 600)  -- d is the total distance given by the solution to be 600 miles

-- Define the conditions stated in the problem
def condition_1 := (x = 10 * v)  -- x = 10 * v (from first part of the solution)
def condition_2 := (3 * v * d - 90 * v = -28.5 * 3 * v)  -- 2nd condition translated from second part

theorem total_distance_of_journey : 
  ∀ (x v : ℝ), condition_1 x v ∧ condition_2 x v -> x = d :=
sorry

end NUMINAMATH_GPT_total_distance_of_journey_l1763_176373


namespace NUMINAMATH_GPT_identity_element_is_neg4_l1763_176351

def op (a b : ℝ) := a + b + 4

def is_identity (e : ℝ) := ∀ a : ℝ, op e a = a

theorem identity_element_is_neg4 : ∃ e : ℝ, is_identity e ∧ e = -4 :=
by
  use -4
  sorry

end NUMINAMATH_GPT_identity_element_is_neg4_l1763_176351


namespace NUMINAMATH_GPT_minimum_boxes_l1763_176391

theorem minimum_boxes (x y z : ℕ) (h1 : 50 * x = 40 * y) (h2 : 50 * x = 25 * z) :
  x + y + z = 17 :=
by
  -- Prove that given these equations, the minimum total number of boxes (x + y + z) is 17
  sorry

end NUMINAMATH_GPT_minimum_boxes_l1763_176391


namespace NUMINAMATH_GPT_distribution_schemes_l1763_176357

theorem distribution_schemes 
    (total_professors : ℕ)
    (high_schools : Finset ℕ) 
    (A : ℕ) 
    (B : ℕ) 
    (C : ℕ)
    (D : ℕ)
    (cond1 : total_professors = 6) 
    (cond2 : A = 1)
    (cond3 : B ≥ 1)
    (cond4 : C ≥ 1)
    (D' := (total_professors - A - B - C)) 
    (cond5 : D' ≥ 1) : 
    ∃ N : ℕ, N = 900 := by
  sorry

end NUMINAMATH_GPT_distribution_schemes_l1763_176357


namespace NUMINAMATH_GPT_projectile_height_35_l1763_176327

noncomputable def projectile_height (t : ℝ) : ℝ := -4.9 * t^2 + 30 * t

theorem projectile_height_35 (t : ℝ) :
  projectile_height t = 35 ↔ t = 10/7 :=
by {
  sorry
}

end NUMINAMATH_GPT_projectile_height_35_l1763_176327


namespace NUMINAMATH_GPT_find_a_l1763_176372

-- Defining the curve y and its derivative y'
def y (x : ℝ) (a : ℝ) : ℝ := x^4 + a * x^2 + 1
def y' (x : ℝ) (a : ℝ) : ℝ := 4 * x^3 + 2 * a * x

theorem find_a (a : ℝ) : 
  y' (-1) a = 8 -> a = -6 := 
by
  -- proof here
  sorry

end NUMINAMATH_GPT_find_a_l1763_176372


namespace NUMINAMATH_GPT_tangent_line_equation_l1763_176309

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / x

theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := f x₀
  let m := deriv f x₀
  y₀ = 0 →
  m = 3 →
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = 3 * x - 3) :=
by
  intros x₀ y₀ m h₀ hm x y
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1763_176309


namespace NUMINAMATH_GPT_proof_problem_l1763_176378

variable (a b c x y z : ℝ)

theorem proof_problem
  (h1 : x + y - z = a - b)
  (h2 : x - y + z = b - c)
  (h3 : - x + y + z = c - a) : 
  x + y + z = 0 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1763_176378
