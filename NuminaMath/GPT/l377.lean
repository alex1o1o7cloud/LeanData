import Complex
import Mathlib
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Mod
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Probability.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Angle
import Mathlib.NumberTheory.ArithmeticSequence
import Mathlib.Probability.Conditional
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Real
import probability_theory.probability

namespace enthalpy_change_correct_l377_377014

def CC_bond_energy : ℝ := 347
def CO_bond_energy : ℝ := 358
def OH_bond_energy_CH2OH : ℝ := 463
def CO_double_bond_energy_COOH : ℝ := 745
def OH_bond_energy_COOH : ℝ := 467
def OO_double_bond_energy : ℝ := 498
def OH_bond_energy_H2O : ℝ := 467

def total_bond_energy_reactants : ℝ :=
  CC_bond_energy + CO_bond_energy + OH_bond_energy_CH2OH + 1.5 * OO_double_bond_energy

def total_bond_energy_products : ℝ :=
  CO_double_bond_energy_COOH + OH_bond_energy_COOH + OH_bond_energy_H2O

def deltaH : ℝ := total_bond_energy_reactants - total_bond_energy_products

theorem enthalpy_change_correct :
  deltaH = 236 := by
  sorry

end enthalpy_change_correct_l377_377014


namespace algebra_expression_value_l377_377490

theorem algebra_expression_value (m : ℝ) (hm : m^2 - m - 1 = 0) : m^2 - m + 2008 = 2009 :=
by
  sorry

end algebra_expression_value_l377_377490


namespace remainder_of_product_l377_377734

theorem remainder_of_product (a b : ℤ) (n : ℕ) (h1 : a ≡ -496 [MOD n]) 
  (h2 : b ≡ 1 [MOD n]) : (a * b) % n = 504 :=
by
  sorry

end remainder_of_product_l377_377734


namespace floor_sequence_u_l377_377688

noncomputable def sequence_u : ℕ → ℚ
| 0     := 2
| 1     := 5/2
| (n+2) := sequence_u (n+1) * (sequence_u n ^ 2 - 2) - sequence_u 1

def floor (x : ℚ) : ℤ := int.floor x

def exponent_x (n : ℕ) : ℚ :=
  (2^n - (-1 : ℚ)^n) / 3

theorem floor_sequence_u (n : ℕ) (hn : 0 < n) :
  floor (sequence_u n) = 2 ^ (exponent_x n) := sorry

end floor_sequence_u_l377_377688


namespace find_a_range_l377_377887

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 + 2 * a * x - Real.log x

theorem find_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (1/3) 2, f' x a ≥ 0) → a ≥ 4 / 3 :=
by
  intros h
  -- define the derivative of f
  let f' := fun x => x + 2 * a - 1 / x
  have d_f_ge := h -- this holds by assumption
  sorry

end find_a_range_l377_377887


namespace trigonometric_identity_l377_377226

theorem trigonometric_identity (t : ℝ) : 
  1 + sin (t / 2) * sin t - cos (t / 2) * (sin t) ^ 2 = 2 * (cos (π / 4 - t / 2)) ^ 2 :=
sorry

end trigonometric_identity_l377_377226


namespace max_sum_of_arithmetic_sequence_l377_377471

theorem max_sum_of_arithmetic_sequence 
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hS18 : S 18 > 0)
  (hS19 : S 19 < 0)
  (hSn_def : ∀ n, S n = n / 2 * (a 1 + a n))
  : S 9 = max (S n) :=
by {
  sorry
}

end max_sum_of_arithmetic_sequence_l377_377471


namespace parallel_lines_l377_377500

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end parallel_lines_l377_377500


namespace least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l377_377439

noncomputable def sum_of_cubes (n : ℕ) : ℕ :=
  (n * (n + 1)/2)^2

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem least_m_for_sum_of_cubes_is_perfect_cube 
  (h : ∃ m : ℕ, ∀ (a : ℕ), (sum_of_cubes (2*m+1) = a^3) → a = 6):
  m = 1 := sorry

theorem least_k_for_sum_of_squares_is_perfect_square 
  (h : ∃ k : ℕ, ∀ (b : ℕ), (sum_of_squares (2*k+1) = b^2) → b = 77):
  k = 5 := sorry

end least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l377_377439


namespace money_needed_l377_377634

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end money_needed_l377_377634


namespace systematic_sampling_probability_l377_377586

theorem systematic_sampling_probability :
  ∀ (population_size sample_size : ℕ),
  population_size = 1003 →
  sample_size = 50 →
  (SystematicSampling.probability population_size sample_size) = 50 / 1003 :=
by
  intros population_size sample_size hpop hsamp
  rw [hpop, hsamp]
  sorry

end systematic_sampling_probability_l377_377586


namespace alpha_beta_total_ways_l377_377406

def total_flavors_of_oreos : ℕ := 6
def total_flavors_of_milk : ℕ := 4
def oreos_exclude_chocolate : ℕ := 5
def total_products_alpha_and_beta_leave_with : ℕ := 3

def alpha_choices (oreos : ℕ) (milks : ℕ) : ℕ :=
  oreos + milks
  
def beta_choices (oreos : ℕ) : ℕ :=
  total_flavors_of_oreos

noncomputable def alpha_purchases_3_items : ℕ :=
  nat.choose (alpha_choices oreos_exclude_chocolate total_flavors_of_milk) total_products_alpha_and_beta_leave_with

noncomputable def alpha_purchases_2_items_beta_1_item : ℕ :=
  let alpha_comb := nat.choose (alpha_choices oreos_exclude_chocolate total_flavors_of_milk) 2 in
  let beta = total_flavors_of_oreos in
  alpha_comb * beta

noncomputable def alpha_purchases_1_item_beta_2_items : ℕ :=
  let alpha = alpha_choices oreos_exclude_chocolate total_flavors_of_milk in
  let beta = (nat.choose total_flavors_of_oreos 2) + total_flavors_of_oreos in
  alpha * beta

noncomputable def alpha_purchases_0_items_beta_3_items : ℕ :=
  let distinct_oreos := nat.choose total_flavors_of_oreos 3 in
  let doubles_and_single := total_flavors_of_oreos * (total_flavors_of_oreos - 1) in
  let triples := total_flavors_of_oreos in
  distinct_oreos + doubles_and_single + triples

noncomputable def total_number_of_ways : ℕ :=
  alpha_purchases_3_items + alpha_purchases_2_items_beta_1_item + alpha_purchases_1_item_beta_2_items + alpha_purchases_0_items_beta_3_items

theorem alpha_beta_total_ways : total_number_of_ways = 545 :=
by
  sorry

end alpha_beta_total_ways_l377_377406


namespace coin_flip_probability_difference_l377_377311

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377311


namespace roots_of_polynomial_l377_377033

noncomputable def polynomial := (x : ℝ) → (x * (x + 3) ^ 2 * (5 - x) = 0)
namespace RootProof

def is_root (p : ℝ → ℝ) (r : ℝ) := p r = 0

theorem roots_of_polynomial : 
  ∀ (x : ℝ), polynomial x = 0 ↔ x = 0 ∨ x = -3 ∨ x = 5 := 
by
  intro x
  split
  · intro hx
    by_cases h₁ : x = 0
    · left
      exact h₁
    by_cases h₂ : x + 3 = 0
    · right; left
      linarith
    by_cases h₃ : 5 - x = 0
    · right; right
      linarith
    · exfalso
      sorry
  · intro h
    cases h
    · exact sorry
    cases h
    · exact sorry
    · exact sorry

end roots_of_polynomial_l377_377033


namespace time_to_fill_tank_l377_377393

theorem time_to_fill_tank (T : ℝ) :
  (1 / 2 * T) + ((1 / 2 * T) / 4) = 10 → T = 16 :=
by { sorry }

end time_to_fill_tank_l377_377393


namespace correct_statement_l377_377793

-- Definitions based on problem's conditions
def meiosis_and_fertilization_info : Prop :=
  ∀ (person : Type),
  (∀ (cells : Type) (genetic_material : cells → Type),
    -- A: Half of the genetic material in my cells comes from my dad, and the other half from my mom.
    ¬ (∀ cell, genetic_material cell = "half from dad ∧ half from mom") ∧
    -- B: My brother and I have the same parents, so the genetic material in our cells is also the same.
    ∀ (siblings : person) (genetic_material_siblings : siblings → Type),
      ¬ ∀ (sibling1 sibling2: siblings),
      genetic_material_siblings sibling1 = genetic_material_siblings sibling2 ∧
    -- C: Each pair of homologous chromosomes in my cells is provided jointly by my parents.
    ∀ (chromosomes : cells → Type),
      chromosomes = "provided jointly by parents" ∧
    -- D: Each pair of homologous chromosomes in a cell is the same size.
    ∀ (chromosomes_size : cells → Type),
      ¬ ∀ (chromosome : chromosomes_size), chromosome = "same size"
  )

-- Theorem stating the correct sentence
theorem correct_statement : meiosis_and_fertilization_info → C :=
by
  sorry

end correct_statement_l377_377793


namespace area_of_quad_ABC_l377_377591

noncomputable def area_quad (AB BC CD AD : ℝ) (O : Point) (angle_AOB : ℝ) : ℝ :=
  let ab := AB
  let bc := BC
  let cd := CD
  let da := AD
  let a := OA O
  let b := OB O
  let c := OC O
  let d := OD O
  let aob := angle_AOB
  let area_OAB := (1/2) * a * b * Real.sin aob
  let area_OBC := (1/2) * b * c * Real.sin (aob + π/3)
  let area_OCD := (1/2) * c * d * Real.sin aob
  let area_ODA := (1/2) * d * a * Real.sin (aob + π/3)
  let S := area_OAB + area_OBC + area_OCD + area_ODA
  S

theorem area_of_quad_ABC (O : Point) (angle_AOB : ℝ) :
  (26 : ℝ) = 26 → 26 → 26 → 30 * sqrt 3 →
  let ab := 26
  let bc := 26
  let cd := 26
  let da := 30 * sqrt 3
  let aob := π/3
  let area := area_quad ab bc cd da O aob
  area = 506 * sqrt 3 :=
by
  intros
  sorry

end area_of_quad_ABC_l377_377591


namespace factorization_of_x12_sub_729_l377_377017

theorem factorization_of_x12_sub_729 (x : ℝ) :
  x^12 - 729 = (x^3 + 3) * (x - real.cbrt 3) * (x^2 + x * real.cbrt 3 + (real.cbrt 3)^2) * (x^12 + 9 * x^6 + 81) := 
sorry

end factorization_of_x12_sub_729_l377_377017


namespace maximum_food_per_guest_l377_377680

theorem maximum_food_per_guest (total_food : ℕ) (min_guests : ℕ) (total_food_eq : total_food = 337) (min_guests_eq : min_guests = 169) :
  ∃ max_food_per_guest, max_food_per_guest = total_food / min_guests ∧ max_food_per_guest = 2 := 
by
  sorry

end maximum_food_per_guest_l377_377680


namespace count_diff_of_squares_l377_377553

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377553


namespace people_on_bus_initially_l377_377809

theorem people_on_bus_initially:
  let p4_after := 39 in
  let p4_off := 112 in
  let p3_on := 53 in
  let p3_off := 96 in
  let p2_on := 88 in
  let p2_off := 59 in
  let p1_on := 62 in
  let p1_off := 75 in
  let p4_before := p4_after + p4_off in
  let p3_after := p4_before - p3_on in
  let p3_before := p3_after + p3_off in
  let p2_after := p3_before - p2_on in
  let p2_before := p2_after + p2_off in
  let p1_after := p2_before - p1_on in
  let p1_before := p1_after + p1_off in
  p1_before = 178 :=
by
  sorry

end people_on_bus_initially_l377_377809


namespace connectivity_of_graph_l377_377970

variables {V : Type*} [fintype V] (G : simple_graph V) [decidable_rel G.adj]

def min_deg (G : simple_graph V) : ℕ :=
  finset.min' (finset.image (λ v, G.degree v) finset.univ)
    (by { simp only [finset.nonempty_image_iff], use classical.some (exists_mem finset.univ) })

theorem connectivity_of_graph (h : min_deg G ≥ (fintype.card V - 1) / 2) : G.connected :=
sorry

end connectivity_of_graph_l377_377970


namespace mass_of_cubic_meter_l377_377252

/-- Condition 1: The volume in cubic centimeters of 1 gram of this substance is 2 cm³ -/
def volume_per_gram : ℝ := 2

/-- Condition 2: 1 cubic meter (m³) is equal to 1,000,000 cubic centimeters (cm³) -/
def cubic_meter_to_cubic_cm : ℝ := 1_000_000

/-- Condition 3: 1 kilogram (kg) is equal to 1,000 grams (g) -/
def kilogram_to_gram : ℝ := 1_000

/-- The mass of 1 cubic meter of the substance under these conditions is 500 kilograms -/
theorem mass_of_cubic_meter : (cubic_meter_to_cubic_cm / volume_per_gram) / kilogram_to_gram = 500 :=
by
  sorry

end mass_of_cubic_meter_l377_377252


namespace walter_age_1999_l377_377010

variable (w g : ℕ) -- represents Walter's age (w) and his grandmother's age (g) in 1994
variable (birth_sum : ℕ) (w_age_1994 : ℕ) (g_age_1994 : ℕ)

axiom h1 : g = 2 * w
axiom h2 : (1994 - w) + (1994 - g) = 3838

theorem walter_age_1999 (w g : ℕ) (h1 : g = 2 * w) (h2 : (1994 - w) + (1994 - g) = 3838) : w + 5 = 55 :=
by
  sorry

end walter_age_1999_l377_377010


namespace ella_spent_on_video_games_last_year_l377_377430

theorem ella_spent_on_video_games_last_year 
  (new_salary : ℝ) 
  (raise : ℝ) 
  (percentage_spent_on_video_games : ℝ) 
  (h_new_salary : new_salary = 275) 
  (h_raise : raise = 0.10) 
  (h_percentage_spent : percentage_spent_on_video_games = 0.40) :
  (new_salary / (1 + raise) * percentage_spent_on_video_games = 100) :=
by
  sorry

end ella_spent_on_video_games_last_year_l377_377430


namespace Route_Y_saves_0_8_minutes_l377_377215

-- Define the conditions related to Route X
def distance_X : ℝ := 8 -- miles
def speed_X : ℝ := 40 -- mph

-- Define the conditions related to Route Y
def distance_Y_total : ℝ := 7 -- miles
def distance_Y1 : ℝ := 6 -- miles (non-construction)
def speed_Y1 : ℝ := 50 -- mph (non-construction)
def distance_Y2 : ℝ := 1 -- mile (construction zone)
def speed_Y2 : ℝ := 15 -- mph (construction zone)

-- Define travel time calculations
def time_X : ℝ := distance_X / speed_X * 60 -- in minutes
def time_Y1 : ℝ := distance_Y1 / speed_Y1 * 60 -- in minutes
def time_Y2 : ℝ := distance_Y2 / speed_Y2 * 60 -- in minutes
def time_Y : ℝ := time_Y1 + time_Y2 -- in minutes

-- Define the savings calculation
def savings : ℝ := time_X - time_Y -- in minutes

-- Define the theorem that needs to be proved
theorem Route_Y_saves_0_8_minutes : savings = 0.8 :=
by
  -- Proof would go here
  sorry

end Route_Y_saves_0_8_minutes_l377_377215


namespace color_overlap_l377_377748

variable (G1 G2 : Fin 1982 × Fin 1983 → Bool)
variable (P1 : ∀ i : Fin 1982, (Σ j, G1 (i, j)) % 2 = 0)
variable (P2 : ∀ j : Fin 1983, (Σ i, G1 (i, j)) % 2 = 0)
variable (P3 : ∀ i : Fin 1982, (Σ j, G2 (i, j)) % 2 = 0)
variable (P4 : ∀ j : Fin 1983, (Σ i, G2 (i, j)) % 2 = 0)

theorem color_overlap (h : ∃ i j, G1 (i, j) ≠ G2 (i, j)) : 
  ∃ i1 j1 i2 j2 i3 j3, 
    (G1 (i1, j1) ≠ G2 (i1, j1)) ∧ 
    (G1 (i2, j2) ≠ G2 (i2, j2)) ∧ 
    (G1 (i3, j3) ≠ G2 (i3, j3)) ∧ 
    (i1 ≠ i ∨ j1 ≠ j) ∧ 
    (i2 ≠ i ∨ j2 ≠ j) ∧ 
    (i3 ≠ i ∨ j3 ≠ j) := 
sorry

end color_overlap_l377_377748


namespace find_maximum_marks_l377_377786

theorem find_maximum_marks (M : ℝ) 
  (h1 : 0.60 * M = 270)
  (h2 : ∀ x : ℝ, 220 + 50 = x → x = 270) : 
  M = 450 :=
by
  sorry

end find_maximum_marks_l377_377786


namespace min_sum_squares_l377_377623

variable {a b c t : ℝ}

def min_value_of_sum_squares (a b c : ℝ) (t : ℝ) : ℝ :=
  a^2 + b^2 + c^2

theorem min_sum_squares (h : a + b + c = t) : min_value_of_sum_squares a b c t ≥ t^2 / 3 :=
by
  sorry

end min_sum_squares_l377_377623


namespace monomial_sum_like_terms_l377_377918

theorem monomial_sum_like_terms (a b : ℝ) (m n : ℤ)
  (h : ∃ c : ℝ, a^(m-1) * b^2 + (1/2) * a * b^(n+5) = c * a^e * b^f) :
  n ^ m = 9 :=
by
  sorry

end monomial_sum_like_terms_l377_377918


namespace positive_difference_prob_3_and_4_heads_l377_377324

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377324


namespace tucker_circle_l377_377224

variables {K : Type*} [EuclideanGeometry K]
variables (A B C A' B' C' A1 A2 B1 B2 C1 C2 : K)

def is_homothetic_with_center (P Q : K) (center : K) : Prop :=
  ∃ k : ℝ, ∃ P' Q' : K, (k • (P - Q)) = (P' - Q') ∧ (P' - center) = k • (P - center)

def lie_on_sides (A1 A2 B1 B2 C1 C2 A B C : K) : Prop :=
  Collinear A1 B C ∧ Collinear A2 C A ∧ Collinear B1 C A ∧ Collinear B2 A B ∧ Collinear C1 A B ∧ Collinear C2 B C

def are_antiparallel (u v w x y z : K) : Prop :=
  ∃ θ : ℝ, Antiparallel u v w x ∧ Antiparallel y z x w ∧ Antiparallel y z x v

theorem tucker_circle
  (h_ABC : Triangle A B C)
  (h_A'B'C' : Triangle A' B' C')
  (h_A'B'C'_homothetic : is_homothetic_with_center A' B' K ∧ is_homothetic_with_center B' C' K ∧ is_homothetic_with_center C' A' K)
  (h_collinear : lie_on_sides A1 A2 B1 B2 C1 C2 A B C)
  (h_antiparallel : are_antiparallel A1 B2 B1 C2 C1 A2 AB BC CA) :
  ∃ (O : Point), Circle_Center O A1 B2 ∧ Circle_Center O B1 C2 ∧ Circle_Center O C1 A2 :=
sorry

end tucker_circle_l377_377224


namespace paula_route_count_l377_377382

def cities_and_roads : Prop :=
  -- There are 15 cities and 20 roads
  ∃ (cities : Finset City) (roads : Finset (City × City)), 
    cities.card = 15 ∧ roads.card = 20

def path_specs (start finish : City) (roads : Finset (City × City)) : Prop :=
  -- Paula starts at city A, ends at city M, and travels exactly 15 roads without repeating
  let path := @list City in
  ∃ (p : path), 
    p.head = start ∧ p.last = finish ∧ (list.cons_mem_eq_of_istr path length_hom p roads).card = 15

def pass_through (city : City) (path : list City) : Prop :=
  -- Paula must pass through city C at least once
  ∃ (i : ℕ), path.nth i = some city

theorem paula_route_count :
  ∀ (cities : Finset City) (roads : Finset (City × City)) (A M C : City),
  cities.card = 15 →
  roads.card = 20 →
  path_specs A M roads →
  pass_through C (list City) →
  ∃ (count : ℕ), count = 4 := 
by
  -- Proof goes here
  sorry

end paula_route_count_l377_377382


namespace Matilda_fathers_chocolate_bars_l377_377214

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l377_377214


namespace matilda_father_chocolates_l377_377209

theorem matilda_father_chocolates 
  (total_chocolates : ℕ) 
  (total_people : ℕ) 
  (give_up_fraction : ℚ) 
  (mother_chocolates : ℕ) 
  (father_eats : ℕ) 
  (father_left : ℕ) :
  total_chocolates = 20 →
  total_people = 5 →
  give_up_fraction = 1 / 2 →
  mother_chocolates = 3 →
  father_eats = 2 →
  father_left = 5 →
  let chocolates_per_person := total_chocolates / total_people,
      father_chocolates := (chocolates_per_person * total_people * give_up_fraction).nat_abs - mother_chocolates - father_eats
  in father_chocolates = father_left := by
  intros h1 h2 h3 h4 h5 h6
  have h_chocolates_per_person : total_chocolates / total_people = 4 := by sorry
  have h_chocolates_given_up : (chocolates_per_person * total_people * give_up_fraction).nat_abs = 10 := by sorry
  have h_father_chocolates : 10 - mother_chocolates - father_eats = 5 := by sorry
  exact h_father_chocolates

end matilda_father_chocolates_l377_377209


namespace least_number_to_add_l377_377732

theorem least_number_to_add (n : ℕ) (h : n = 28523) : 
  ∃ x, x + n = 29560 ∧ 3 ∣ (x + n) ∧ 5 ∣ (x + n) ∧ 7 ∣ (x + n) ∧ 8 ∣ (x + n) :=
by 
  sorry

end least_number_to_add_l377_377732


namespace prove_tan_2x_prove_sin_x_plus_pi_over_4_l377_377092

open Real

noncomputable def tan_2x (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) : Prop :=
  tan (2 * x) = 4 / 3

noncomputable def sin_x_plus_pi_over_4 (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) : Prop :=
  sin (x + π/4) = sqrt 10 / 10

theorem prove_tan_2x (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) :
  tan_2x x hx h_tan :=
sorry

theorem prove_sin_x_plus_pi_over_4 (x : ℝ) (hx : x ∈ set.Ioo (π / 2) π) (h_tan : abs (tan x) = 2) :
  sin_x_plus_pi_over_4 x hx h_tan :=
sorry

end prove_tan_2x_prove_sin_x_plus_pi_over_4_l377_377092


namespace equal_angles_of_intersecting_circles_l377_377749

-- Defining the problem with conditions and target goals in Lean 4
theorem equal_angles_of_intersecting_circles
  (circle1 circle2 : Circle) (P Q : Point)
  (circle3 : Circle) (A B C D: Point)
  (h1 : circle1.Intersects circle2 P Q)
  (h2 : circle3.center = P)
  (h3 : circle3.Intersects circle1 A B)
  (h4 : circle3.Intersects circle2 C D) :
  angle A Q D = angle B Q C :=
by
  sorry

end equal_angles_of_intersecting_circles_l377_377749


namespace charlotte_needs_to_bring_l377_377016

noncomputable def total_amount_to_bring : ℝ :=
let boots_price := 90.0 in
let jacket_price := 120.0 in
let scarf_price := 30.0 in
let boots_discount := 0.20 in
let jacket_discount := 0.15 in
let scarf_discount := 0.10 in
let sales_tax_rate := 0.07 in

-- Calculate discounted prices
let discounted_boots_price := boots_price * (1 - boots_discount) in
let discounted_jacket_price := jacket_price * (1 - jacket_discount) in
let discounted_scarf_price := scarf_price * (1 - scarf_discount) in

-- Calculate total before tax
let total_before_tax := discounted_boots_price + discounted_jacket_price + discounted_scarf_price in

-- Calculate sales tax
let sales_tax := total_before_tax * sales_tax_rate in

-- Calculate total amount to bring
let total_final := total_before_tax + sales_tax in

-- Return the final amount
total_final

theorem charlotte_needs_to_bring : total_amount_to_bring = 215.07 :=
by 
  unfold total_amount_to_bring
  norm_num
  -- Only 'sorry' here to skip the actual detailed proof steps if needed.
  -- sorry


end charlotte_needs_to_bring_l377_377016


namespace combined_mixture_nuts_l377_377237

def sue_percentage_nuts : ℝ := 0.30
def sue_percentage_dried_fruit : ℝ := 0.70

def jane_percentage_nuts : ℝ := 0.60
def combined_percentage_dried_fruit : ℝ := 0.35

theorem combined_mixture_nuts :
  let sue_contribution := 100.0
  let jane_contribution := 100.0
  let sue_nuts := sue_contribution * sue_percentage_nuts
  let jane_nuts := jane_contribution * jane_percentage_nuts
  let combined_nuts := sue_nuts + jane_nuts
  let total_weight := sue_contribution + jane_contribution
  (combined_nuts / total_weight) * 100 = 45 :=
by
  sorry

end combined_mixture_nuts_l377_377237


namespace function_monotonically_increasing_intervals_l377_377110

-- Given the function f(x)
def f (x : ℝ) := (Real.cos x) ^ 2 + Real.sqrt 3 * (Real.sin x) * (Real.cos x) - 1 / 2

-- Statement: The intervals where the function f(x) is monotonically increasing.
theorem function_monotonically_increasing_intervals : 
  ∀ (x : ℝ) (k : ℤ), -Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi → 
  MonotoneOn f {x | -Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi} := 
sorry

end function_monotonically_increasing_intervals_l377_377110


namespace positive_difference_prob_3_and_4_heads_l377_377326

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377326


namespace subset_splits_count_14_l377_377574

def is_subset_split {α : Type*} (M N Ω : set α) : Prop :=
  M ∪ N = Ω

def unique_subset_splits_count (Ω : set ℕ) : ℕ :=
  if Ω = {1, 2, 3} then 14 else 0

theorem subset_splits_count_14 (Ω : set ℕ) (h : Ω = {1, 2, 3}) :
  unique_subset_splits_count Ω = 14 :=
by {
  rw [unique_subset_splits_count, if_pos h],
  refl,
}

end subset_splits_count_14_l377_377574


namespace min_value_of_f_l377_377050

open Real

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem min_value_of_f : ∃ x : ℝ, x > 2 ∧ ∀ y : ℝ, y > 2 → f y ≥ f 3 := by
  sorry

end min_value_of_f_l377_377050


namespace book_arrangement_l377_377563

theorem book_arrangement : 
  let math_books : Fin 4 → ℕ := fun i => i
  let english_books : Fin 5 → ℕ := fun i => i + 4
  ∃ (ways : ℕ), 
    (ways = 3! * 4!) ∧
    (∀ i j, i < j → 0 ≤ math_books i ∧ math_books i < math_books j ∧ math_books j < 4) ∧
    (∀ k l, k < l → 4 ≤ english_books k ∧ english_books k < english_books l ∧ english_books l < 9) ∧
    (math_books 0 = 0) ∧
    (english_books 4 = 8).
Proof.
  sorry

end book_arrangement_l377_377563


namespace velocity_of_current_l377_377355

theorem velocity_of_current
  (v c : ℝ) 
  (h1 : 32 = (v + c) * 6) 
  (h2 : 14 = (v - c) * 6) :
  c = 1.5 :=
by
  sorry

end velocity_of_current_l377_377355


namespace find_value_of_a_plus_b_plus_c_l377_377463

theorem find_value_of_a_plus_b_plus_c (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_ineq : a^2 + b^2 + c^2 + 43 ≤ a * b + 9 * b + 8 * c) : 
  a + b + c = 13 := 
begin
  sorry,
end

end find_value_of_a_plus_b_plus_c_l377_377463


namespace problem_statement_l377_377868

open Complex

theorem problem_statement (a : ℝ) : (∃ x : ℝ, (1 + a * Complex.i) / (2 + Complex.i) = x) → a = 1 / 2 :=
by
  sorry

end problem_statement_l377_377868


namespace factor_x12_minus_729_l377_377020

theorem factor_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) := 
by
  sorry

end factor_x12_minus_729_l377_377020


namespace mean_problem_l377_377358

theorem mean_problem (x : ℝ) (h : (12 + x + 42 + 78 + 104) / 5 = 62) :
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end mean_problem_l377_377358


namespace faye_homework_problems_l377_377436

----- Definitions based on the conditions given -----

def total_math_problems : ℕ := 46
def total_science_problems : ℕ := 9
def problems_finished_at_school : ℕ := 40

----- Theorem statement -----

theorem faye_homework_problems : total_math_problems + total_science_problems - problems_finished_at_school = 15 := by
  -- Sorry is used here to skip the proof
  sorry

end faye_homework_problems_l377_377436


namespace largest_constant_exists_l377_377416

noncomputable def is_interesting_sequence (z : ℕ → ℂ) : Prop :=
  |∀ n, 4 * (z (n + 1))^2 + 2 * z n * z (n + 1) + (z n)^2 = 0

theorem largest_constant_exists (C : ℝ) : 
  (C = real.sqrt 3 / 3) →
  ∀ (z : ℕ → ℂ), (| z 1 | = 1) ∧ is_interesting_sequence z →
  ∀ m : ℕ, m > 0 → (| ∑ i in range m, z (i + 1) | ≥ C) :=
begin 
  sorry
end

end largest_constant_exists_l377_377416


namespace positive_difference_probability_l377_377295

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377295


namespace side_lengths_and_angles_l377_377689

-- Define the ratio and the condition of sum of the squares
def triangle_sides (a b c : ℝ) : Prop :=
  a = 3 * 4 ∧ b = 4 * 4 ∧ c = 5 * 4

def sum_of_squares (a b c : ℝ) : Prop :=
  (a^2 + b^2 + c^2 = 800)

theorem side_lengths_and_angles (a b c : ℝ) (alpha beta gamma : ℝ) :
  triangle_sides a b c ∧ sum_of_squares a b c →
  (a = 12 ∧ b = 16 ∧ c = 20) ∧ 
  (gamma = 90) ∧ 
  (alpha = Real.arcsin(3/5) * 180 / Real.pi) ∧ 
  (beta = 90 - alpha) :=
by
  sorry

end side_lengths_and_angles_l377_377689


namespace count_of_squares_difference_l377_377518

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377518


namespace find_k_value_l377_377145

-- Define the inverse proportion function
def inverse_proportion_function (k x : ℝ) : ℝ := (k - 1) / x

-- Define the condition where the function passes through (-1, -2)
def passes_through (k : ℝ) : Prop := inverse_proportion_function k (-1) = -2

-- State the theorem asserting the value of k
theorem find_k_value (k : ℝ) (h : passes_through k) : k = 3 :=
sorry

end find_k_value_l377_377145


namespace sum_of_possible_y_values_l377_377117

noncomputable def mean (y : ℝ) : ℝ := (30 + y) / 7
def mode : ℝ := 3

def median (y : ℝ) : ℝ :=
if y ≤ 3 then 3
else if y < 5 then y
else 5

def is_geometric_progression (a b c : ℝ) : Prop :=
  b / a = c / b

theorem sum_of_possible_y_values : 
  let list := [5, 3, 7, 3, 9, 3, y] in 
  ∑ y in {y | is_geometric_progression 3 3 (mean y) ∨ 
                  is_geometric_progression 3 5 (mean y)}, y = 20 := 
sorry

end sum_of_possible_y_values_l377_377117


namespace recorded_score_l377_377266

theorem recorded_score (base_score student_score recorded : Int) (h_base : base_score = 60) (h_student_score : student_score = 54) : recorded = student_score - base_score → recorded = -6 :=
by
  intros
  rw [h_base, h_student_score]
  calculate_expr (54 - 60)
  exact eq.rfl

end recorded_score_l377_377266


namespace arithmetic_sequence_a8_l377_377568

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (n : ℕ)

-- Arithmetic sequence definition
def arithmetic_seq (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

-- Specific conditions for our problem
def specific_conditions : Prop :=
  arithmetic_seq a (-1) (-3)

-- Theorem statement
theorem arithmetic_sequence_a8 :
  specific_conditions a d →
  a 8 = -22 :=
by
  sorry

end arithmetic_sequence_a8_l377_377568


namespace angle_equivalence_l377_377640

-- Definitions of geometric objects
variables {A B C D E : Point}
variables {triangle ABC : Triangle}

-- Extensions to handle specific geometric configurations
def is_diameter_endpoint (C : Point) (D : Point) (circle : Circumcircle ABC) :=
  D = endpoint_of_diameter(C, circle)

def is_altitude_intersection (C : Point) (E : Point) (A B : Point) (circle : Circumcircle ABC) :=
  meets_circle(altitude(C, A, B), circle) = E

-- The main theorem statement
theorem angle_equivalence (triangle ABC : Triangle)
  (circumcircle : Circumcircle ABC)
  (C : Point)
  (D : Point)
  (E : Point)
  (hD : is_diameter_endpoint C D circumcircle)
  (hE : is_altitude_intersection C E A B circumcircle) :
  angle AD = angle EB :=
sorry

end angle_equivalence_l377_377640


namespace base10_to_base7_l377_377730

theorem base10_to_base7 (n : ℕ) : n = 803 → nat.toDigits 7 n = [2, 2, 2, 5] :=
by
  intro h
  rw [h]
  sorry -- proof goes here

end base10_to_base7_l377_377730


namespace number_of_boxes_in_each_case_l377_377204

theorem number_of_boxes_in_each_case (a b : ℕ) :
    a + b = 2 → 9 = a * 8 + b :=
by
    intro h
    sorry

end number_of_boxes_in_each_case_l377_377204


namespace significant_figures_and_precision_l377_377684

noncomputable def num : ℝ := 0.320

def significant_figures (n : ℝ) : ℕ :=
  -- This is a hypothetical function to calculate significant figures
  if n = 0 then 1 else
  n.to_string.count_digits - n.to_string.find_first_nonzero_digit

def precision (n : ℝ) : String :=
  -- This is a hypothetical function to calculate precision
  if n.to_string.endswith "0" then "thousandth place" else "accurate to last non-zero digit"

theorem significant_figures_and_precision : significant_figures num = 3 ∧ precision num = "thousandth place" :=
  sorry

end significant_figures_and_precision_l377_377684


namespace absolute_value_example_l377_377137

theorem absolute_value_example (x : ℝ) (h : x = 4) : |x - 5| = 1 :=
by
  sorry

end absolute_value_example_l377_377137


namespace parallel_lines_slope_l377_377086

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 = 0) →
  a = -3 :=
by
  sorry

end parallel_lines_slope_l377_377086


namespace halfway_between_one_eighth_and_one_third_l377_377683

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 + 1 / 3) / 2 = 11 / 48 :=
by
  -- Skipping the proof here
  sorry

end halfway_between_one_eighth_and_one_third_l377_377683


namespace equal_distribution_of_cookies_l377_377061

theorem equal_distribution_of_cookies :
  (let friends := 4 in
  let packages := 3 in
  let cookies_per_package := 25 in
  let children := friends + 1 in
  let total_cookies := packages * cookies_per_package in
  (total_cookies / children) = 15) :=
sorry

end equal_distribution_of_cookies_l377_377061


namespace hexagon_height_correct_l377_377670

-- Define the dimensions of the original rectangle
def original_rectangle_width := 16
def original_rectangle_height := 9
def original_rectangle_area := original_rectangle_width * original_rectangle_height

-- Define the dimensions of the new rectangle formed by the hexagons
def new_rectangle_width := 12
def new_rectangle_height := 12
def new_rectangle_area := new_rectangle_width * new_rectangle_height

-- Define the parameter x, which is the height of the hexagons
def hexagon_height := 6

-- Theorem stating the equivalence of the areas and the specific height x
theorem hexagon_height_correct :
  original_rectangle_area = new_rectangle_area ∧
  hexagon_height * 2 = new_rectangle_height :=
by
  sorry

end hexagon_height_correct_l377_377670


namespace prove_f_iterative_value_l377_377993

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem prove_f_iterative_value (p q : ℝ) (h1 : ∀ x, 2 ≤ x → x ≤ 4 → -1/2 ≤ f x p q ∧ f x p q ≤ 1/2) :
  let initial_value := (5 - Real.sqrt 11) / 2 in
  let res := (iterate (f p q) 2017 initial_value) in
  res = (5 + Real.sqrt 11) / 2 :=
by
  sorry

end prove_f_iterative_value_l377_377993


namespace ravi_jump_height_l377_377647

theorem ravi_jump_height (j1 j2 j3 : ℕ) (average : ℕ) (ravi_jump_height : ℕ) (h : j1 = 23 ∧ j2 = 27 ∧ j3 = 28) 
  (ha : average = (j1 + j2 + j3) / 3) (hr : ravi_jump_height = 3 * average / 2) : ravi_jump_height = 39 :=
by
  sorry

end ravi_jump_height_l377_377647


namespace option_B_option_D_l377_377134

variables {Ω : Type*} {P : MeasureTheory.ProbabilityMeasure Ω}
variables (A B : Set Ω)

theorem option_B (hA : P(A) > 0) (hB : P(B) > 0) : 
  (MeasureTheory.condprob (Set.compl A) B * P.toOuterMeasure B = 
   MeasureTheory.condprob B (Set.compl A) * P.toOuterMeasure (Set.compl A)) :=
sorry

theorem option_D (hA : P(A) > 0) (hB : P(B) > 0) :
  (MeasureTheory.condprob A B = P.toOuterMeasure A) → 
  (MeasureTheory.condprob B A = P.toOuterMeasure B) :=
sorry

end option_B_option_D_l377_377134


namespace tickets_spent_on_beanie_l377_377808

variable (initial_tickets won_tickets tickets_left tickets_spent: ℕ)

theorem tickets_spent_on_beanie
  (h1 : initial_tickets = 49)
  (h2 : won_tickets = 6)
  (h3 : tickets_left = 30)
  (h4 : tickets_spent = initial_tickets + won_tickets - tickets_left) :
  tickets_spent = 25 :=
by
  sorry

end tickets_spent_on_beanie_l377_377808


namespace emily_weight_l377_377506

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end emily_weight_l377_377506


namespace age_of_person_l377_377638

/-- Given that Noah's age is twice someone's age and Noah will be 22 years old after 10 years, 
    this theorem states that the age of the person whose age is half of Noah's age is 6 years old. -/
theorem age_of_person (N : ℕ) (P : ℕ) (h1 : P = N / 2) (h2 : N + 10 = 22) : P = 6 := by
  sorry

end age_of_person_l377_377638


namespace chord_ratio_l377_377715

theorem chord_ratio
  (E F G H Q : Type)
  (EQ FQ GQ HQ : ℝ)
  (h1 : EQ = 5)
  (h2 : GQ = 12)
  (H1 : EQ * FQ = GQ * HQ) :
  FQ / HQ = 12 / 5 :=
by
  have h3 := H1
  rw [←h1, ←h2] at h3
  field_simp at h3
  exact h3

end chord_ratio_l377_377715


namespace price_change_theorem_l377_377409

-- Define initial prices
def candy_box_price_before : ℝ := 10
def soda_can_price_before : ℝ := 9
def popcorn_bag_price_before : ℝ := 5
def gum_pack_price_before : ℝ := 2

-- Define price changes
def candy_box_price_increase := candy_box_price_before * 0.25
def soda_can_price_decrease := soda_can_price_before * 0.15
def popcorn_bag_price_factor := 2
def gum_pack_price_change := 0

-- Compute prices after the policy changes
def candy_box_price_after := candy_box_price_before + candy_box_price_increase
def soda_can_price_after := soda_can_price_before - soda_can_price_decrease
def popcorn_bag_price_after := popcorn_bag_price_before * popcorn_bag_price_factor
def gum_pack_price_after := gum_pack_price_before

-- Compute total costs
def total_cost_before := candy_box_price_before + soda_can_price_before + popcorn_bag_price_before + gum_pack_price_before
def total_cost_after := candy_box_price_after + soda_can_price_after + popcorn_bag_price_after + gum_pack_price_after

-- The statement to be proven
theorem price_change_theorem :
  total_cost_before = 26 ∧ total_cost_after = 32.15 :=
by
  -- This part requires proof, add 'sorry' for now
  sorry

end price_change_theorem_l377_377409


namespace count_of_squares_difference_l377_377519

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377519


namespace polynomials_exist_l377_377952

theorem polynomials_exist (p : ℕ) (hp : Nat.Prime p) :
  ∃ (P Q : Polynomial ℤ),
  ¬(Polynomial.degree P = 0) ∧ ¬(Polynomial.degree Q = 0) ∧
  (∀ n, (Polynomial.coeff (P * Q) n).natAbs % p =
    if n = 0 then 1
    else if n = 4 then 1
    else if n = 2 then p - 2
    else 0) :=
sorry

end polynomials_exist_l377_377952


namespace cubic_sum_l377_377916

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 :=
by
  sorry

end cubic_sum_l377_377916


namespace john_mary_next_to_each_other_on_long_side_l377_377956

/-- 
Given five people: John, Mary, Alice, Bob, and Clara, seated randomly around a rectangular table where two people sit on each of the longer sides and one person on each of the shorter sides, prove that the probability that John and Mary are seated next to each other on one of the longer sides is 1/4.
-/
theorem john_mary_next_to_each_other_on_long_side : 
  let people := ["John", "Mary", "Alice", "Bob", "Clara"]
  let longer_sides := 2
  let shorter_sides := 1
  (∀ (arrangements : list (list string)), 
    (arrangements.length = 24 ∧
    ∃ (arrangement : list string), 
      arrangement ∈ arrangements ∧
      ["John", "Mary"] ⊆ arrangement ∧
      arrangement.length = longer_sides) →
    (probability : ℚ) = 1) → 
  probability = 1/4 :=
by
  sorry

end john_mary_next_to_each_other_on_long_side_l377_377956


namespace faster_speed_14_l377_377769

theorem faster_speed_14 
    (d₁ : ℕ) -- actual distance traveled
    (d₂ : ℕ) -- additional distance at faster speed
    (s₁ : ℕ) -- initial speed
    (s₂ : ℕ) -- faster speed
    (h₁ : d₁ = 50)
    (h₂ : s₁ = 10)
    (h₃ : d₂ = 20) : 
    s₂ = 14 :=
by
  have t := d₁ / s₁     -- Calculate time taken to travel distance d₁ at speed s₁
  have d := d₁ + d₂     -- Total distance covered when walking at the faster speed s₂
  have s := d / t       -- Calculate the faster speed s₂ as total distance divided by time
  have eq1 : t = 5 := by simp [h₁, h₂]
  have eq2 : d = 70 := by simp [h₁, h₃]
  exact by simp [eq1, eq2]

end faster_speed_14_l377_377769


namespace hyperbola_has_equation_and_area_l377_377893

noncomputable def hyperbola_eq (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∃ λ : ℝ, λ ≠ 0 ∧ λ ≠ 1 ∧
  (∀ x y : ℝ, (x, y) = (2, 3) → (y^2 / 6 - x^2 / 2 = λ) ∧ (x^2 - y^2 / 3 = 1))

theorem hyperbola_has_equation_and_area :
  hyperbola_eq 2 (3 / 2) → ∃ (F1 F2 : ℝ × ℝ), ∃ A B : ℝ × ℝ,
  A ∈ {p : ℝ × ℝ | p.2 = 2 - p.1 ∧ (3 * p.1^2 - p.2^2 = 3)} ∧
  B ∈ {p : ℝ × ℝ | p.2 = 2 - p.1 ∧ (3 * p.1^2 - p.2^2 = 3)} ∧
  let d := (|-2 + 0 - 2| : ℝ) / (√2 : ℝ) in
  let ab := (sqrt(1 + 1) * sqrt(4 - 4 * (-7/2)) : ℝ) in
  let area := (1 / 2 * ab * d) in
  area = 6 * sqrt 2 :=
by sorry

end hyperbola_has_equation_and_area_l377_377893


namespace tetrahedron_volume_correct_l377_377723

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2))

theorem tetrahedron_volume_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 = c^2) :
  tetrahedron_volume a b c = (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2)) :=
by
  sorry

end tetrahedron_volume_correct_l377_377723


namespace lee_can_make_36_cookies_l377_377179

-- Conditions
def initial_cups_of_flour : ℕ := 2
def initial_cookies_made : ℕ := 18
def initial_total_flour : ℕ := 5
def spilled_flour : ℕ := 1

-- Define the remaining cups of flour after spilling
def remaining_flour := initial_total_flour - spilled_flour

-- Define the proportion to solve for the number of cookies made with remaining_flour
def cookies_with_remaining_flour (c : ℕ) : Prop :=
  (initial_cookies_made / initial_cups_of_flour) = (c / remaining_flour)

-- The statement to prove
theorem lee_can_make_36_cookies : cookies_with_remaining_flour 36 :=
  sorry

end lee_can_make_36_cookies_l377_377179


namespace max_distance_product_eq_five_l377_377188

-- Conditions
variable (m : ℝ)

def lineA (x y m : ℝ) := x + m * y = 0
def lineB (x y m : ℝ) := m * x - y - m + 3 = 0
def pointA := (0,0)
def pointB := (1,3)
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Question equivalence to the original problem
theorem max_distance_product_eq_five (x y : ℝ) :
  lineA x y m →
  lineB x y m →
  let P := (x, y)
  let PA := distance P pointA
  let PB := distance P pointB
  let AB := distance pointA pointB
  AB = Real.sqrt 10 →
  PA * PB ≤ 5 :=
sorry

end max_distance_product_eq_five_l377_377188


namespace trigonometric_identity_l377_377904

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := 
by 
  sorry

end trigonometric_identity_l377_377904


namespace product_of_two_numbers_l377_377719

theorem product_of_two_numbers :
  ∀ x y: ℝ, 
  ((x - y)^2) / ((x + y)^3) = 4 / 27 → 
  x + y = 5 * (x - y) + 3 → 
  x * y = 15.75 :=
by 
  intro x y
  sorry

end product_of_two_numbers_l377_377719


namespace range_of_a_l377_377487

def f (x : ℝ) : ℝ :=
if x < 1 then (1/2) * x - (1/2)
else Real.log x

theorem range_of_a (a : ℝ) : (f (f a) = Real.log (f a)) ↔ e ≤ a :=
by sorry

end range_of_a_l377_377487


namespace Gumble_words_total_l377_377251

noncomputable def num_letters := 25
noncomputable def exclude_B := 24

noncomputable def total_5_letters_or_less (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 5 then num_letters^n - exclude_B^n else 0

noncomputable def total_Gumble_words : ℕ :=
  (total_5_letters_or_less 1) + (total_5_letters_or_less 2) + (total_5_letters_or_less 3) +
  (total_5_letters_or_less 4) + (total_5_letters_or_less 5)

theorem Gumble_words_total :
  total_Gumble_words = 1863701 := by
  sorry

end Gumble_words_total_l377_377251


namespace iterated_f_result_l377_377995

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p*x + q

theorem iterated_f_result (p q : ℝ) 
  (h1 : ∀ x ∈ set.Icc (2 : ℝ) (4 : ℝ), |f x p q| ≤ 1/2) :
  iter 2017 (λ x, f x p q) ((5 - real.sqrt 11) / 2) = real.of_rat 4.16 :=
sorry

end iterated_f_result_l377_377995


namespace evaluate_star_l377_377027

def star (A B : ℝ) : ℝ := (A * B + A) / 5

theorem evaluate_star : star (star 3 6) 4 = 4.2 := by
  sorry

end evaluate_star_l377_377027


namespace min_value_of_quotient_l377_377202

theorem min_value_of_quotient :
  ∀ (a b c : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 → 
  (∃ m, ∀ (x y z : ℕ), 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 →
    (m ≤ (100 * x + 10 * y + z) / (x + y + z))) ∧ 
  m = 505 / 10 := 
begin
  sorry
end

end min_value_of_quotient_l377_377202


namespace rolls_per_day_per_bathroom_l377_377218

theorem rolls_per_day_per_bathroom (packs_bought : ℕ) (rolls_per_pack : ℕ) (weeks : ℕ) (bathrooms : ℕ) (days_per_week : ℕ) : 
  packs_bought = 14 → rolls_per_pack = 12 → weeks = 4 → bathrooms = 6 → days_per_week = 7 → 
  (packs_bought * rolls_per_pack / (weeks * days_per_week * bathrooms) = 1) := 
by
  intros h1 h2 h3 h4 h5
  have h_total_rolls := congrArg (λ x, x * 12) h1
  rw [h1] at h_total_rolls
  rw [mul_comm 12 14] at h_total_rolls
  rw [h3] at h_total_rolls
  have h_weekly_rolls := h2 * 14 / 4
  rw [h_weeks] at h_weekly_rolls
  have h_daily_rolls := h_weekly_rolls / 7
  rw [h_days] at h_daily_rolls
  have h_per_bathroom := h_daily_rolls / 6 
  rw [h_bathrooms] at h_per_bathroom
  exact h_per_bathroom

end rolls_per_day_per_bathroom_l377_377218


namespace max_independent_set_l377_377925

-- Define the given conditions
variables (P : Type) [fintype P]
variables (acquainted : P → P → Prop)
variables (h_size : fintype.card P = 30)
variables (h_acquainted_max : ∀ p : P, (finset.filter (λ q, acquainted p q) finset.univ).card ≤ 5)
variables (h_group_of_five : ∀ (s : finset P), s.card = 5 → ∃ (p q : P), p ∈ s ∧ q ∈ s ∧ ¬acquainted p q)

-- Define the independent set
def independent_set (s : finset P) : Prop :=
  ∀ p q ∈ s, p ≠ q → ¬acquainted p q

-- State the main theorem
theorem max_independent_set : ∃ s : finset P, s.card = 6 ∧ independent_set s :=
sorry

end max_independent_set_l377_377925


namespace area_triangle_algebraic_expression_l377_377894

-- Define the parabola and the conditions
def parabola (b : ℝ) := λ x : ℝ, x^2 + b * x - 2

-- Given the parabola passes through (1, 3)
def condition1 (b : ℝ) : Prop := (parabola b 1 = 3)

-- Given k is the root of k^2 + 4k - 2 = 0
def condition2 (k : ℝ) : Prop := k^2 + 4 * k = 2

-- Prove the area of the triangle OAB
theorem area_triangle (b : ℝ) (h_b : condition1 b) :
  let A := (0 : ℝ, -2 : ℝ),
      B := (-2 : ℝ, -6 : ℝ),
      O := (0 : ℝ, 0 : ℝ) in
  (1 / 2 * (abs (0 - -2) * abs (1 - 0)) = 2) :=
by
  -- proof goes here
  sorry

-- Prove the value of the algebraic expression
theorem algebraic_expression (k : ℝ) (h_k : condition2 k) :
  (4 * k^4 + 3 * k^2 + 12 * k - 6) / (k^8 + 2 * k^6 + k^5 - 2 * k^3 + 8 * k^2 + 16) = (1 / 107) :=
by
  -- proof goes here
  sorry

end area_triangle_algebraic_expression_l377_377894


namespace smallest_squares_key_smallest_diagonal_key_l377_377404

-- Definition for n code grid
def is_n_code (n : ℕ) (grid : ℕ → ℕ → ℤ) : Prop :=
  ∀ i j k : ℕ, i < n → j < n → k < n → (grid i j - grid i (j + 1) = grid i (j + 1) - grid i (j + 2)) ∧ (grid i j - grid (i + 1) j = grid (i + 1) j - grid (i + 2) j)

-- Part a
theorem smallest_squares_key (n : ℕ) (h : n ≥ 4) : ∃ s : ℕ, s = 2 * n ∧
  ∀ (grid : ℕ → ℕ → ℤ), is_n_code n grid →
  (∀ (chosen_squares : list (ℕ × ℕ)), chosen_squares.length = s →
    (∀ (a b : ℕ), ∃ (i j : ℕ), (i, j) ∈ chosen_squares) →
    ∃ (unique_grid : ℕ → ℕ → ℤ), unique_grid = grid) :=
sorry

-- Part b
theorem smallest_diagonal_key (n : ℕ) (h : n ≥ 4) : ∃ t : ℕ, t = 3 ∧
  ∀ (grid : ℕ → ℕ → ℤ), is_n_code n grid →
  (∀ (chosen_diagonals : list (ℕ × ℕ)), chosen_diagonals.length = t → 
    (∀ (i j : ℕ), (grid (i, i) ∈ chosen_diagonals ∨ grid (i, n-i-1) ∈ chosen_diagonals)) →
    ∃ (unique_grid : ℕ → ℕ → ℤ), unique_grid = grid) :=
sorry

end smallest_squares_key_smallest_diagonal_key_l377_377404


namespace exists_nonzero_lambdas_divisible_by_three_l377_377627

-- Define the problem statement
theorem exists_nonzero_lambdas_divisible_by_three (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : b₁ > 0) (h₅ : b₂ > 0) (h₆ : b₃ > 0):
  ∃ (λ₁ λ₂ λ₃ : ℕ), λ₁ ≠ 0 ∧ λ₂ ≠ 0 ∧ λ₃ ≠ 0 ∧ λ₁ ≤ 2 ∧ λ₂ ≤ 2 ∧ λ₃ ≤ 2 ∧
  (λ₁ * a₁ + λ₂ * a₂ + λ₃ * a₃) % 3 = 0 ∧ (λ₁ * b₁ + λ₂ * b₂ + λ₃ * b₃) % 3 = 0 :=
by
  sorry

end exists_nonzero_lambdas_divisible_by_three_l377_377627


namespace coin_flip_probability_difference_l377_377338

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377338


namespace find_ticket_cost_l377_377713

-- Define the initial amount Tony had
def initial_amount : ℕ := 20

-- Define the amount Tony paid for a hot dog
def hot_dog_cost : ℕ := 3

-- Define the amount Tony had after buying the ticket and the hot dog
def remaining_amount : ℕ := 9

-- Define the function to find the baseball ticket cost
def ticket_cost (t : ℕ) : Prop := initial_amount - t - hot_dog_cost = remaining_amount

-- The statement to prove
theorem find_ticket_cost : ∃ t : ℕ, ticket_cost t ∧ t = 8 := 
by 
  existsi 8
  unfold ticket_cost
  simp
  exact sorry

end find_ticket_cost_l377_377713


namespace least_period_of_g_l377_377817

theorem least_period_of_g (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x + 2) + g(x - 2) = g(x)) : ∃ q > 0, (q = 12 ∧ ∀ p > 0, (∀ x : ℝ, g(x + p) = g(x)) → p >= q) :=
by { sorry }

end least_period_of_g_l377_377817


namespace find_m_l377_377074

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

def symmetric_line (x y m : ℝ) : Prop := x + m * y + 4 = 0

theorem find_m (m : ℝ) (h1 : circle_equation (-1) 3) (h2 : symmetric_line (-1) 3 m) : m = -1 := by
  sorry

end find_m_l377_377074


namespace part_I_part_II_l377_377112

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 2 * sin x * cos x + 3 * cos x ^ 2

theorem part_I {k : ℤ} : ∀ x, (k * π - 3 * π / 8 <= x ∧ x <= k * π + π / 8) → 
  monotone_on (λ y, f y) (Icc (k * π - 3 * π / 8) (k * π + π / 8)) := 
sorry

theorem part_II : 
  let interval := Icc (0:ℝ) (π / 2)
  { f_max := (λ x, f x = sqrt 2 + 2 ↔ x = π / 8),
    f_min := (λ x, f x = 1 ↔ x = π / 2) } :=
sorry

end part_I_part_II_l377_377112


namespace range_of_b_l377_377891

theorem range_of_b (f g : ℝ → ℝ) (a b : ℝ)
  (hf : ∀ x, f x = Real.exp x - 1)
  (hg : ∀ x, g x = -x^2 + 4*x - 3)
  (h : f a = g b) :
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by
  sorry

end range_of_b_l377_377891


namespace triangle_classification_l377_377154

theorem triangle_classification (A B C a b c : ℝ) (h1 : b ≠ 1)
  (h2 : ∀ x, (log (sqrt 6) x = log b (4 * x - 4)) → (x = C / A ∨ x = sin B / sin A)) : 
  (B = 90) ∧ (A = C ∨ A ≠ C) :=
by
  sorry

end triangle_classification_l377_377154


namespace total_fish_l377_377651

statement : Prop :=
  let initial_fish : ℕ := 26
  let bought_fish : ℕ := 6
  initial_fish + bought_fish = 32

theorem total_fish : statement :=
begin
  sorry
end

end total_fish_l377_377651


namespace rubles_left_l377_377383

-- Defining the problem statement based on the conditions
theorem rubles_left (r : ℕ → ℕ) (h_diff : ∀ i j : ℕ, i ≠ j → r i ≠ r j)
  (h_pos : ∀ i : ℕ, 1 ≤ r i ∧ r i ≤ 9)
  (h_sum : ∑ i in (Finset.range 8), r i = 40) :
  ∃ s : Finset ℕ, s = {1, 2, 3, 4, 6, 7, 8, 9} ∧ ∀ i : ℕ, (i ∈ s → r i ∈ s) :=
sorry

end rubles_left_l377_377383


namespace unique_solution_to_function_equation_l377_377842

theorem unique_solution_to_function_equation (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (2 * n) = 2 * f n)
  (h2 : ∀ n : ℕ, f (2 * n + 1) = 2 * f n + 1) :
  ∀ n : ℕ, f n = n :=
by
  sorry

end unique_solution_to_function_equation_l377_377842


namespace george_painting_combinations_l377_377852

namespace Combinations

/-- George's painting problem -/
theorem george_painting_combinations :
  let colors := 10
  let colors_to_pick := 3
  let textures := 2
  ((colors) * (colors - 1) * (colors - 2) / (colors_to_pick * (colors_to_pick - 1) * 1)) * (textures ^ colors_to_pick) = 960 :=
by
  sorry

end Combinations

end george_painting_combinations_l377_377852


namespace number_of_ordered_pairs_l377_377032

theorem number_of_ordered_pairs :
  ∃ n : ℕ, n = 89 ∧ (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x < y ∧ 2 * x * y = 8 ^ 30 * (x + y)) := sorry

end number_of_ordered_pairs_l377_377032


namespace point_inside_circle_l377_377881

theorem point_inside_circle 
  (a b c : ℝ) (e : ℝ) (x1 x2 : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a > b) 
  (h4 : e = 1 / 2) 
  (h5 : c = e * a)
  (h6 : b = sqrt(3) / 2 * a) 
  (h7 : a ≠ 0)
  (h8 : ax1^2 + bx1 - c = 0)
  (h9 : ax2^2 + bx2 - c = 0) :
  (x1, x2).1^2 + (x1, x2).2^2 < 2 :=
begin
  sorry
end

end point_inside_circle_l377_377881


namespace ceiling_evaluation_l377_377431

def problem_statement : ℝ :=
  4 * (7 - 3 / 4)

theorem ceiling_evaluation : ⌈problem_statement⌉ = 25 :=
by
  sorry

end ceiling_evaluation_l377_377431


namespace hyperbola_equation_l377_377479

noncomputable def focus_of_parabola_eq (y2_eq_20x :  ∀ x y, y^2 = 20 * x) : Real := 
    5

noncomputable def hyperbola_asymptote_eq (y_eq_2x : ∀ x y, y = 2*x) : Real := 
    ∀ a b, b/a = 2

theorem hyperbola_equation (focus_parabola : ∀ x y, y^2 = 20 * x)
    (asymptote_hyperbola : ∀ x y, y = 2 * x)
    (focus_hyperbola_eq_focus_parabola: ∀ x, x = 5)
    : ∃ a b : ℝ,
      (a^2 + (2*a)^2 = 25) ∧
      (a^2 = 5) ∧
      (b^2 = 20) ∧
      (∀ x y : ℝ, (x^2 / 5) - (y^2 / 20) = 1) :=
begin
  sorry
end

end hyperbola_equation_l377_377479


namespace integer_diff_of_squares_1_to_2000_l377_377542

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377542


namespace train_passing_time_l377_377787

theorem train_passing_time 
  (length_of_train : ℕ) 
  (length_of_platform : ℕ) 
  (time_to_pass_pole : ℕ) 
  (speed_of_train : ℕ) 
  (combined_length : ℕ) 
  (time_to_pass_platform : ℕ) 
  (h1 : length_of_train = 240) 
  (h2 : length_of_platform = 650)
  (h3 : time_to_pass_pole = 24)
  (h4 : speed_of_train = length_of_train / time_to_pass_pole)
  (h5 : combined_length = length_of_train + length_of_platform)
  (h6 : time_to_pass_platform = combined_length / speed_of_train) : 
  time_to_pass_platform = 89 :=
sorry

end train_passing_time_l377_377787


namespace find_ab_product_l377_377419

-- Lean 4 statement to define the problem conditions
def permutation_condition (n : ℕ) (σ : List ℕ) : Prop :=
  ∀ k ∈ σ, (List.filter (λ x => x < k) (σ.dropWhile (λ x => x ≠ k)).tail).length % 2 = 0

-- Lean 4 statement to express the main theorem 
theorem find_ab_product : ∃ a b : ℕ, ∏ σ in (List.permutations (List.range 2022)), 
  permutation_condition 2022 σ ∧ 
  |S| = (a!)^b ∧ 
  a = 1011 ∧ 
  b = 2 ∧
  a * b = 2022 := 
sorry

end find_ab_product_l377_377419


namespace boy_running_time_l377_377509

theorem boy_running_time (s : ℝ) (v : ℝ) (h1 : s = 35) (h2 : v = 9) : 
  (4 * s) / (v * 1000 / 3600) = 56 := by
  sorry

end boy_running_time_l377_377509


namespace f_ff_equality_l377_377484

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) else real.log x / real.log 2

theorem f_ff_equality : f(f(1 / 4)) = 4 := by
  sorry

end f_ff_equality_l377_377484


namespace binomial_coeff_x2_l377_377031

noncomputable def binomial_coeff (n k : ℕ) : ℕ := nat.choose n k

theorem binomial_coeff_x2 (x : ℝ) : 
  (∑ k in Finset.range (7 + 1), binomial_coeff 7 k * (1:ℝ) ^ (7 - k) * x ^ k) = 
  1 * x ^ 2 * 21 + ∑ k in (Finset.range (7 + 1)).filter (λ k, k ≠ 2), binomial_coeff 7 k * (1:ℝ) ^ (7 - k) * x ^ k := 
by {
    sorry
}

end binomial_coeff_x2_l377_377031


namespace coin_flip_probability_difference_l377_377315

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377315


namespace combinatorial_identity_l377_377194

theorem combinatorial_identity (n m : ℕ) : 
  (nat.choose n m) + (2 * nat.choose (n-1) m) + ∑ k in finset.range (n + 1 - m), (nat.choose (n - k) m) * (k + 1) = nat.choose (n + 2) (m + 2) :=
by
  sorry

end combinatorial_identity_l377_377194


namespace diff_of_squares_1500_l377_377537

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377537


namespace find_5_digit_number_l377_377837

theorem find_5_digit_number {A B C D E : ℕ} 
  (hA_even : A % 2 = 0) 
  (hB_even : B % 2 = 0) 
  (hA_half_B : A = B / 2) 
  (hC_sum : C = A + B) 
  (hDE_prime : Prime (10 * D + E)) 
  (hD_3B : D = 3 * B) : 
  10000 * A + 1000 * B + 100 * C + 10 * D + E = 48247 := 
sorry

end find_5_digit_number_l377_377837


namespace problem_a_part_problem_b_part_l377_377622

-- Definitions of positive integers a and b with a > b
variables {a b : ℕ}
-- Assuming a and b are positive integers and a > b
hypothesis h1 : a > b
-- Assuming integer k exists such that sqrt(sqrt(a) + sqrt(b)) + sqrt(sqrt(a) - sqrt(b)) is an integer
variable (k : ℤ)
hypothesis h2 : k = Int.natAbs (Int.sqrt (int.of_nat (nat.sqrt a + nat.sqrt b)) + Int.sqrt (int.of_nat (nat.sqrt a - nat.sqrt b)))

-- Main proof statement (statement only, no proof, no solution steps)
theorem problem_a_part : ∀ {a b : ℕ}, a > b → k ∈ ℤ → nat.sqrt a ∈ ℕ :=
by { intros, sorry }

theorem problem_b_part : ∀ {a b : ℕ}, a > b → k ∈ ℤ → ¬(∀ {b : ℕ}, b > 0 → (nat.sqrt b ∈ ℕ)) :=
by { intros, sorry }

end problem_a_part_problem_b_part_l377_377622


namespace Matilda_fathers_chocolate_bars_l377_377213

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l377_377213


namespace max_flights_28_cities_l377_377926

def city : Type := {n // n ∈ Fin 28}
def AirportType := {t // t = "small" ∨ t = "medium" ∨ t = "big"}

structure City := 
  (id : city)
  (airport : AirportType)

def unique_airports (cities : List City) : Prop :=
  cities.all (λ ci, ci.airport.exists (∈ ["small", "medium", "big"]))

structure Flight := 
  (from : City)
  (to : City)

def valid_route (route : List Flight) : Prop :=
  route.head? = route.reverse.head? ∧ 
  route.nodup ∧ 
  ∃ (a b c : City), a ∈ route ∧ b ∈ route ∧ c ∈ route ∧ 
  a.airport = "small" ∧ b.airport = "medium" ∧ c.airport = "big"

def max_flights (n_cities : Nat) (flights : List Flight) : Nat :=
  let unique_cities := flights.foldl (λ acc f => acc ∪ {f.from, f.to}) ∅
  unique_cities.size

theorem max_flights_28_cities :
  ∃ flights, valid_route flights ∧ max_flights 28 flights = 286 := sorry

end max_flights_28_cities_l377_377926


namespace correct_reference_l377_377132

variable (house : String) 
variable (beautiful_garden_in_front : Bool)
variable (I_like_this_house : Bool)
variable (enough_money_to_buy : Bool)

-- Statement: Given the conditions, prove that the correct word to fill in the blank is "it".
theorem correct_reference : I_like_this_house ∧ beautiful_garden_in_front ∧ ¬ enough_money_to_buy → "it" = "correct choice" :=
by
  sorry

end correct_reference_l377_377132


namespace angle_AHD_right_l377_377962

-- Define a trapezoid ABCD with AD || BC
variables {A B C D M C1 K H : Point}

-- Define a function that states point M is the midpoint of AD
def is_midpoint (M : Point) (A D : Point) : Prop :=
  dist A M = dist M D

-- Define a function that states C1 is the symmetric point to C with respect to BD
def is_symmetric (C1 C B D : Point) : Prop :=
  dist B C1 = dist B C ∧ dist D C1 = dist D C

-- Define a function that states segment BM meets diagonal AC at K
def meets_at_K (B M A C K : Point) : Prop :=
  collinear B M K ∧ collinear A C K

-- Define a function that states ray C1K meets line BD at H
def meets_at_H (C1 K B D H : Point) : Prop :=
  collinear C1 K H ∧ collinear B D H

-- State the goal to prove that angle AHD is a right angle
theorem angle_AHD_right (h_trapezoid : parallel AD BC)
  (h_midpoint : is_midpoint M A D)
  (h_symmetric : is_symmetric C1 C B D)
  (h_BM_meets_AC_at_K : meets_at_K B M A C K)
  (h_C1K_meets_BD_at_H : meets_at_H C1 K B D H) :
  angle A H D = 90 :=
sorry

end angle_AHD_right_l377_377962


namespace polynomial_divisibility_l377_377613

noncomputable def polynomial_with_positive_int_coeffs : Type :=
{ f : ℕ → ℕ // ∀ m n : ℕ, f m < f n ↔ m < n }

theorem polynomial_divisibility
  (f : polynomial_with_positive_int_coeffs)
  (n : ℕ) (hn : n > 0) :
  f.1 n ∣ f.1 (f.1 n + 1) ↔ n = 1 :=
sorry

end polynomial_divisibility_l377_377613


namespace max_imaginary_part_l377_377423

noncomputable def theta (z : ℂ) : ℝ :=
  Real.arcsin (z.im / abs z)

theorem max_imaginary_part :
  ∀ (z : ℂ), z^6 - z^4 + z^2 - z + 1 = 0 →
  Complex.abs (Complex.sin (theta z)) ≤ Complex.sin (Real.of_rat (900 / 7)) := 
by simp; sorry

end max_imaginary_part_l377_377423


namespace emily_weight_l377_377507

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end emily_weight_l377_377507


namespace not_two_by_two_green_prob_l377_377037

theorem not_two_by_two_green_prob (m n : ℕ) (h_rel_prime : Nat.coprime m n) 
  (h_prob : (m : ℚ) / (n : ℚ) = 1 - (9 * 2^12 - 12 * 2^10 + 6 * 2^8 - 9) / 2^16) : 
  m + n = 987 := 
sorry

end not_two_by_two_green_prob_l377_377037


namespace range_of_m_l377_377885

noncomputable def f (x : ℝ) : ℝ := 2 * sin (3 * x + π / 6) + 1

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) (π / 3), f x + m = 0 →  ∃! x1, ∃! x2, x1 ≠ x2) ↔ m ∈ Icc (-3 : ℝ) (-2) :=
by sorry

end range_of_m_l377_377885


namespace prop_2_prop_3_prop_4_l377_377486

variable (f : ℝ → ℝ) (a : ℝ)

def f_def (x : ℝ) : f x = Real.exp x + a * Real.log x := by sorry

theorem prop_2 : a = -1 → ∀ x > 0, Real.exp x - Real.log x > 0 := by sorry

theorem prop_3 : 0 < a → ∀ x_1 x_2, (0 < x_1 ∧ 0 < x_2 ∧ x_1 < x_2) →
  (f_def f a x_1 < f_def f a x_2) := by sorry

theorem prop_4 : a < 0 → ∃ x, ∀ y > 0, Real.exp y + a * Real.log y ≥ (Real.exp x + a * Real.log x) := by sorry

end prop_2_prop_3_prop_4_l377_377486


namespace simplify_fraction_subtraction_l377_377663

theorem simplify_fraction_subtraction : (7 / 3) - (5 / 6) = 3 / 2 := by
  sorry

end simplify_fraction_subtraction_l377_377663


namespace num_points_P_l377_377166

theorem num_points_P (ABCD : Type)
  [square ABCD] :
  ∃ P : Point, (isosceles_triangle P A B) 
    ∧ (isosceles_triangle P B C) 
    ∧ (isosceles_triangle P C D) 
    ∧ (isosceles_triangle P D A) := 
sorry

end num_points_P_l377_377166


namespace no_such_number_l377_377387

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def productOfDigitsIsPerfectSquare (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ isPerfectSquare (d1 * d2)

theorem no_such_number :
  ¬ ∃ (N : ℕ),
    (N > 9) ∧ (N < 100) ∧ -- N is a two-digit number
    (N % 2 = 0) ∧        -- N is even
    (N % 13 = 0) ∧       -- N is a multiple of 13
    productOfDigitsIsPerfectSquare N := -- The product of digits of N is a perfect square
by
  sorry

end no_such_number_l377_377387


namespace new_mean_rent_l377_377673

theorem new_mean_rent (avg_rent : ℕ) (num_friends : ℕ) (rent_increase_pct : ℕ) (initial_rent : ℕ) :
  avg_rent = 800 →
  num_friends = 4 →
  rent_increase_pct = 25 →
  initial_rent = 800 →
  (avg_rent * num_friends + initial_rent * rent_increase_pct / 100) / num_friends = 850 :=
by
  intros h_avg h_num h_pct h_init
  sorry

end new_mean_rent_l377_377673


namespace average_salary_of_officers_l377_377588

-- Define the given conditions
def avg_salary_total := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 480

-- Define the expected result
def avg_salary_officers := 440

-- Define the problem and statement to be proved in Lean
theorem average_salary_of_officers :
  (num_officers + num_non_officers) * avg_salary_total - num_non_officers * avg_salary_non_officers = num_officers * avg_salary_officers := 
by
  sorry

end average_salary_of_officers_l377_377588


namespace probability_complement_given_A_l377_377466

theorem probability_complement_given_A :
  (∀ (A B : Type) [MeasureTheory.ProbabilityMeasure A] [MeasureTheory.ProbabilityMeasure B],
  let PA := MeasureTheory.Measure.probability
  let PB := MeasureTheory.Measure.probability
  let PAB := PA * PB in
  PA = 1/3 → PB = 1/4 → PA (B | A) = 3/4 →
  PA (¬ B | A) = 7/16) :=
by
  intros A B _ _ PA PB PAB hPA hPB hPA_B
  sorry

end probability_complement_given_A_l377_377466


namespace suff_not_nec_vectors_collinear_l377_377738

theorem suff_not_nec_vectors_collinear (a b : Vector) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a + b = 0) : a ∥ b :=
sorry

end suff_not_nec_vectors_collinear_l377_377738


namespace river_width_l377_377779

theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_min : ℝ) (width : ℝ) :
  depth = 2 ∧ flow_rate = 116.67 ∧ volume_per_min = 10500 → width ≈ 45 :=
by
  sorry -- Proof here

end river_width_l377_377779


namespace positive_difference_prob_3_and_4_heads_l377_377291

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377291


namespace count_diff_of_squares_l377_377559

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377559


namespace number_of_terms_in_arithmetic_sequence_is_20_l377_377083

theorem number_of_terms_in_arithmetic_sequence_is_20
  (a : ℕ → ℤ)
  (common_difference : ℤ)
  (h1 : common_difference = 2)
  (even_num_terms : ℕ)
  (h2 : ∃ k, even_num_terms = 2 * k)
  (sum_odd_terms sum_even_terms : ℤ)
  (h3 : sum_odd_terms = 15)
  (h4 : sum_even_terms = 35)
  (h5 : ∀ n, a n = a 0 + n * common_difference) :
  even_num_terms = 20 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_20_l377_377083


namespace allens_mothers_age_l377_377795

-- Define the conditions
variables (A M S : ℕ) -- Declare variables for ages of Allen, his mother, and his sister

-- Define Allen is 30 years younger than his mother
axiom h1 : A = M - 30

-- Define Allen's sister is 5 years older than him
axiom h2 : S = A + 5

-- Define in 7 years, the sum of their ages will be 110
axiom h3 : (A + 7) + (M + 7) + (S + 7) = 110

-- Define the age difference between Allen's mother and sister is 25 years
axiom h4 : M - S = 25

-- State the theorem: what is the present age of Allen's mother
theorem allens_mothers_age : M = 48 :=
by sorry

end allens_mothers_age_l377_377795


namespace integer_diff_of_squares_1_to_2000_l377_377545

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377545


namespace probability_of_successful_pairs_l377_377919

def is_multiple_of (x n : ℕ) : Prop := ∃ k, n * k = x

def multiples_of (n : ℕ) (s : set ℕ) : set ℕ := { x ∈ s | is_multiple_of x n }

def set_a := {3, 5, 15, 21, 28, 35, 45, 63}

def pairs (s : set ℕ) : set (ℕ × ℕ) := { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def successful_pairs (s : set ℕ) : set (ℕ × ℕ) :=
  { (x, y) ∈ pairs s | is_multiple_of (x * y) 105 }

def total_pairs (s : set ℕ) : ℕ := (s.to_finset.card * (s.to_finset.card - 1)) / 2

theorem probability_of_successful_pairs :
  let s := set_a in
  let successful := successful_pairs s in
  let total := total_pairs s in
  (successful.to_finset.card / total.to_finset.card : ℚ) = (5 / 28 : ℚ) :=
by
  let s := set_a
  let successful := successful_pairs s
  let total := total_pairs s
  have h_successful_pairs : (successful.to_finset.card) = 5, sorry
  have h_total_pairs : (total.to_finset.card) = 28, sorry
  rw [h_successful_pairs, h_total_pairs]
  norm_num
  sorry

end probability_of_successful_pairs_l377_377919


namespace fixed_point_exists_l377_377481

variables {a b k m c e : ℝ}
variables {F : ℝ × ℝ} (hF : F = (-sqrt 3, 0))
variables {C : ℝ → ℝ → Prop}
variables {l : ℝ → ℝ}

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def line (x y : ℝ) : Prop := y = k * x + m

theorem fixed_point_exists (ha : a = 2)
  (hb : b = 1)
  (he : e = sqrt 3 / 2)
  (hl_focus : (∀ x y : ℝ, ellipse x y ↔ (x^2 / 4 + y^2 = 1)))
  (hl_line : (∀ x y : ℝ, line x y ↔ y = k * x + m))
  (hne : k ≠ 0)
  (hMN : ∃ x1 y1 x2 y2 : ℝ, ellipse x1 y1 ∧ ellipse x2 y2 ∧ line x1 y1 ∧ line x2 y2 ∧ (2 - x2) * (2 - x1) + y1 * y2 = 0)
  (h_intersect : ∀ x1 y1 x2 y2 : ℝ, hMN(x1, y1, x2, y2) ∧ (2 - x2) * (2 - x1) + y1 * y2 = 0) :
  (∃ p : ℝ × ℝ, line p.1 p.2 ∧ p = (6/5, 0)) :=
sorry

end fixed_point_exists_l377_377481


namespace emily_weight_l377_377508

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end emily_weight_l377_377508


namespace intersection_sets_A_B_l377_377866

-- Define Sets A and B based on the given conditions
def A : Set ℕ := {x | |x| < 3}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- The theorem that proves the intersection of sets A and B
theorem intersection_sets_A_B : (A ∩ B) = {0, 1} :=
by
  sorry

end intersection_sets_A_B_l377_377866


namespace number_of_cut_red_orchids_l377_377707

variable (initial_red_orchids added_red_orchids final_red_orchids : ℕ)

-- Conditions
def initial_red_orchids_in_vase (initial_red_orchids : ℕ) : Prop :=
  initial_red_orchids = 9

def final_red_orchids_in_vase (final_red_orchids : ℕ) : Prop :=
  final_red_orchids = 15

-- Proof statement
theorem number_of_cut_red_orchids (initial_red_orchids added_red_orchids final_red_orchids : ℕ)
  (h1 : initial_red_orchids_in_vase initial_red_orchids) 
  (h2 : final_red_orchids_in_vase final_red_orchids) :
  final_red_orchids = initial_red_orchids + added_red_orchids → added_red_orchids = 6 := by
  simp [initial_red_orchids_in_vase, final_red_orchids_in_vase] at *
  sorry

end number_of_cut_red_orchids_l377_377707


namespace faster_speed_l377_377770

theorem faster_speed (v : ℝ) (h1 : ∀ (t : ℝ), t = 50 / 10) (h2 : ∀ (d : ℝ), d = 50 + 20) (h3 : ∀ (t : ℝ), t = 70 / v) : v = 14 :=
by
  sorry

end faster_speed_l377_377770


namespace product_of_integers_abs_gt_3_le_6_l377_377256

theorem product_of_integers_abs_gt_3_le_6 : 
  ∏ n in ({n : ℤ | abs n > 3 ∧ abs n ≤ 6}).toFinset, n = -14400 := 
by
  sorry

end product_of_integers_abs_gt_3_le_6_l377_377256


namespace harmonic_sum_identity_l377_377187

def harmonic (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (k m : ℕ) (hk : 0 < k) :
  ∑ n in Finset.range m, 1 / ((n + 1 + k : ℕ) * (harmonic (n + 1)) * (harmonic (n + 1 + k))) =
  ∑ n in Finset.range m, (1 / (harmonic (n + 1)) - 1 / (harmonic (n + 1 + k))) :=
by
  sorry

end harmonic_sum_identity_l377_377187


namespace integral_of_sqrt_expression_l377_377102

theorem integral_of_sqrt_expression (a : ℝ) (h : ∃ (a : ℝ), (a+(a-2)*complex.I).re = a ∧ (a+(a-2)*complex.I).im = 0) :
  ∫ x in 0..a, real.sqrt (4 - x^2) = real.pi :=
sorry

end integral_of_sqrt_expression_l377_377102


namespace range_of_m_l377_377976

variable {f : ℝ → ℝ}
variable {m : ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x ≥ f y
def p (f : ℝ → ℝ) : Prop := monotonically_decreasing f (set.Icc 0 2)
def q (f : ℝ → ℝ) (m : ℝ) : Prop := f (1 - m) ≥ f m

theorem range_of_m
  (hf_odd : odd_function f)
  (hf_dec : p f)
  (h_not_p_or_q_false : ¬¬p f ∧ ¬q f m) :
  -1 ≤ m ∧ m < 1 / 2 :=
begin
  sorry
end

end range_of_m_l377_377976


namespace cosine_of_acute_angle_l377_377504

theorem cosine_of_acute_angle (α : ℝ) (h : Vector.parallel ((1 / 3), (Real.tan α)) ((Real.cos α), 1)) (hα : 0 < α ∧ α < Real.pi / 2) : Real.cos α = 2 * Real.sqrt 2 / 3 := 
sorry

end cosine_of_acute_angle_l377_377504


namespace arithmetic_geometric_sequences_l377_377480

theorem arithmetic_geometric_sequences :
  (a₁ a₂ b₁ b₂ b₃ : ℝ) (h_arith : ∀ n, 1 + n*(9 - 1)/3 = n) (h_geom : ∀ n, 1 * (3:ℝ)^n = n):
  b₂ / (a₁ + a₂) = 3 / 10 :=
by
  have h_arithmetic := h_arith,
  have h_geometric := h_geom,
  sorry

end arithmetic_geometric_sequences_l377_377480


namespace triangle_area_proof_l377_377425

-- Define the triangle sides and median
variables (AB BC BD AC : ℝ)

-- Assume given values
def AB_value : AB = 1 := by sorry 
def BC_value : BC = Real.sqrt 15 := by sorry
def BD_value : BD = 2 := by sorry

-- Assume AC calculated from problem
def AC_value : AC = 4 := by sorry

-- Final proof statement
theorem triangle_area_proof 
  (hAB : AB = 1)
  (hBC : BC = Real.sqrt 15)
  (hBD : BD = 2)
  (hAC : AC = 4) :
  (1 / 2) * AB * BC = (Real.sqrt 15) / 2 := 
sorry

end triangle_area_proof_l377_377425


namespace matrix_pow_4_equals_l377_377021

theorem matrix_pow_4_equals :
  let A := Matrix.of (λ i j, ![(1, -2), (2, 1)] i j)
  A^4 = Matrix.of (λ i j, ![(-7, 24), (-24, 7)] i j) :=
by
  sorry

end matrix_pow_4_equals_l377_377021


namespace evaluate_g_at_neg3_l377_377905

def g (x : ℤ) := 10 * x^3 - 7 * x^2 - 5 * x + 6

theorem evaluate_g_at_neg3 : g (-3) = -312 :=
by
  simp [g]
  sorry

end evaluate_g_at_neg3_l377_377905


namespace years_to_rise_to_chief_l377_377955

-- Definitions based on the conditions
def ageWhenRetired : ℕ := 46
def ageWhenJoined : ℕ := 18
def additionalYearsAsMasterChief : ℕ := 10
def multiplierForChiefToMasterChief : ℚ := 1.25

-- Total years spent in the military
def totalYearsInMilitary : ℕ := ageWhenRetired - ageWhenJoined

-- Given conditions and correct answer
theorem years_to_rise_to_chief (x : ℚ) (h : totalYearsInMilitary = x + multiplierForChiefToMasterChief * x + additionalYearsAsMasterChief) :
  x = 8 := by
  sorry

end years_to_rise_to_chief_l377_377955


namespace number_of_mutually_exclusive_pairs_l377_377851

-- Definitions of events as predicates
namespace BallDrawing

def at_least_one_white (balls : Finset String) : Prop := 
  "white" ∈ balls

def both_white (balls : Finset String) : Prop := 
  balls = {"white", "white"}

def at_least_one_red (balls : Finset String) : Prop := 
  "red" ∈ balls

def both_red (balls : Finset String) : Prop := 
  balls = {"red", "red"}

def exactly_one_white (balls : Finset String) : Prop := 
  balls.count "white" = 1

def exactly_two_white (balls : Finset String) : Prop := 
  balls.count "white" = 2

-- Number of mutually exclusive events
def num_mutually_exclusive_pairs : Nat := 
  (ite (¬∀ balls, (at_least_one_white balls → ¬both_white balls)) 0 1) +
  (ite (¬∀ balls, (at_least_one_white balls → ¬at_least_one_red balls)) 0 1) +
  (ite (∀ balls, (exactly_one_white balls → ¬exactly_two_white balls)) 1 0) +
  (ite (∀ balls, (at_least_one_white balls → ¬both_red balls)) 1 0)

-- Theorem statement
theorem number_of_mutually_exclusive_pairs : num_mutually_exclusive_pairs = 2 :=
by
  sorry

end BallDrawing

end number_of_mutually_exclusive_pairs_l377_377851


namespace probability_sum_exceeds_one_l377_377927

theorem probability_sum_exceeds_one :
  ∀ (p_X p_Y p_Z : ℚ), 
  p_X = 1/2 ∧ p_Y = 1/4 ∧ p_Z = 1/3 ∧
  p_X + p_Y + p_Z = 13/12 →
  false :=
by
  intro p_X p_Y p_Z h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5] at h6
  norm_num at h6
  contradiction

end probability_sum_exceeds_one_l377_377927


namespace distance_after_15_minutes_l377_377676

-- Define the conditions
def initial_distance : ℝ := 2.5
def hyoseong_speed : ℝ := 0.08 / 1 -- km per minute
def mimi_speed_per_hour : ℝ := 2.4 -- km per hour
def mimi_speed : ℝ := mimi_speed_per_hour / 60 -- convert to km per minute
def time : ℝ := 15 -- time in minutes

-- Calculate the distance covered
def combined_speed : ℝ := hyoseong_speed + mimi_speed
def distance_covered : ℝ := combined_speed * time

-- Calculate the remaining distance
noncomputable def remaining_distance := initial_distance - distance_covered

-- Problem statement: Prove that the remaining distance after 15 minutes is 0.7 km
theorem distance_after_15_minutes : remaining_distance = 0.7 := by
  sorry

end distance_after_15_minutes_l377_377676


namespace total_fuel_needed_l377_377267

/-- Given that Car B can travel 30 miles per gallon and needs to cover a distance of 750 miles,
    and Car C has a fuel consumption rate of 20 miles per gallon and will travel 900 miles,
    prove that the total combined fuel required for Cars B and C is 70 gallons. -/
theorem total_fuel_needed (miles_per_gallon_B : ℕ) (miles_per_gallon_C : ℕ)
  (distance_B : ℕ) (distance_C : ℕ)
  (hB : miles_per_gallon_B = 30) (hC : miles_per_gallon_C = 20)
  (dB : distance_B = 750) (dC : distance_C = 900) :
  (distance_B / miles_per_gallon_B) + (distance_C / miles_per_gallon_C) = 70 := by {
    sorry 
}

end total_fuel_needed_l377_377267


namespace valerie_skips_per_minute_l377_377228

-- Definitions based on the conditions
def roberto_skips_per_hour : ℕ := 4200
def total_skips_in_15_minutes : ℕ := 2250
def time_in_minutes : ℕ := 15

-- Theorem stating the relationship derived from the conditions
theorem valerie_skips_per_minute : (valerie_skips : ℕ) 
  (h1 : valerie_skips * time_in_minutes = total_skips_in_15_minutes - (roberto_skips_per_hour / 4)) :
  valerie_skips = 80 :=
by
  sorry

end valerie_skips_per_minute_l377_377228


namespace cone_volume_max_height_l377_377078

noncomputable def cone_max_height : ℝ := 4 / 3

theorem cone_volume_max_height (h : ℝ) (r : ℝ) (V : ℝ) :
  (h - 1)^2 + r^2 = 1 ∧ V = (π / 3) * r^2 * h ∧ 
  (∀ h < cone_max_height, (π / 3) * r^2 * h < V) ∧ (∀ h > cone_max_height, V < (π / 3) * r^2 * h)
  → h = cone_max_height :=
begin
  intros,
  sorry
end

end cone_volume_max_height_l377_377078


namespace probability_not_orange_l377_377850

noncomputable def probability_A_not_orange : ℚ :=
  let event_A := 1 / 3
  in 1 - event_A

theorem probability_not_orange :
  probability_A_not_orange = 2 / 3 :=
by
  dsimp [probability_A_not_orange]
  norm_num
  sorry

end probability_not_orange_l377_377850


namespace min_blue_points_l377_377278

theorem min_blue_points (n : ℕ) (hn : n ≥ 2) :
  let points := { p : (ℝ × ℝ) // ∀ i j : ℕ, i < j → p i ≠ p j } in
  ∃ (blue_points : finset (ℝ × ℝ)),
    (∀ i j : fin n, i ≠ j → let mp := ((p i).1 + (p j).1) / 2, ((p i).2 + (p j).2) / 2 in mp ∈ blue_points) ∧
    blue_points.card = 2 * n - 3 :=
sorry

end min_blue_points_l377_377278


namespace triangle_incircle_excircle_midpoint_equality_l377_377180
-- Import the Mathlib library for all necessary theorems and definitions

-- Statement of the theorem
theorem triangle_incircle_excircle_midpoint_equality
  (ABC : Triangle) -- ABC is a scalene triangle
  (M : Point)      -- M is the midpoint of BC
  (P : Point)      -- P is the common point of AM and the incircle of ABC closest to A
  (Q : Point)      -- Q is the common point of the ray AM and the excircle farthest from A
  (X : Point)      -- X is the point where the tangent to the incircle at P meets BC
  (Y : Point)      -- Y is the point where the tangent to the excircle at Q meets BC
  (h1 : is_midpoint M B C) -- M is the midpoint of BC
  (h2 : is_common_point P (incircle_of ABC) (ray_through A M) closest_to_A) -- P is the common point of AM and incircle closest to A
  (h3 : is_common_point Q (excircle_of ABC) (ray_through A M) farthest_from_A) -- Q is the common point of the ray AM and the excircle farthest from A
  (h4 : tangent_at P (incircle_of ABC) X) -- The tangent to the incircle at P meets BC at X
  (h5 : tangent_at Q (excircle_of ABC) Y) -- The tangent to the excircle at Q meets BC at Y
  :
  dist M X = dist M Y := 
sorry

end triangle_incircle_excircle_midpoint_equality_l377_377180


namespace positive_difference_probability_l377_377300

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377300


namespace valid_three_digit_numbers_l377_377561

theorem valid_three_digit_numbers (Total three_digit total: ℕ = 900) (restricted total: ℕ = 81) : 
  Total three_digit total - restricted = 819 :=
by
  sorry -- Proof omitted

end valid_three_digit_numbers_l377_377561


namespace num_true_statements_l377_377106

theorem num_true_statements (m n p : ℝ) (A B C O : Point) 
  (h: m • (O - A) + n • (O - B) + p • (O - C) = 0) : 
  number_of_true_statements [
    is_inside_triangle O A B C,
    is_centroid O A B C,
    is_incenter O A B C,
    is_circumcenter O A B C
  ] = 3 := by
  sorry

end num_true_statements_l377_377106


namespace find_x_l377_377090

-- We are given points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 2)

-- Vector a is (2x + 3, x^2 - 4)
def vec_a (x : ℝ) : ℝ × ℝ := (2 * x + 3, x^2 - 4)

-- Vector AB is calculated as
def vec_AB : ℝ × ℝ := (3 - 1, 2 - 2)

-- Define the condition that vec_a and vec_AB form 0° angle
def forms_zero_angle (u v : ℝ × ℝ) : Prop := (u.1 * v.2 - u.2 * v.1) = 0 ∧ (u.1 = v.1 ∧ v.2 = 0)

-- The proof statement
theorem find_x (x : ℝ) (h₁ : forms_zero_angle (vec_a x) vec_AB) : x = 2 :=
by
  sorry

end find_x_l377_377090


namespace magnitude_product_l377_377863

-- Define the complex numbers z1 and z2
variables (z1 z2 : ℂ)

-- Assume the given conditions
def magnitude_z1 := 3
def z2_complex := (2 : ℂ) + (1 : ℂ) * complex.i

-- Define the given magnitudes and properties
axiom magnitude_z1_def : |z1| = magnitude_z1
axiom z2_def : z2 = z2_complex

-- Prove the magnitude of the product
theorem magnitude_product : |z1 * z2| = 3 * real.sqrt 5 :=
by
  sorry

end magnitude_product_l377_377863


namespace coin_flip_probability_difference_l377_377314

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377314


namespace coin_flip_probability_difference_l377_377335

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377335


namespace sin_add_tan_special_angles_l377_377411

theorem sin_add_tan_special_angles : 
  sin (Real.pi / 6) + tan (Real.pi / 3) = 1 / 2 + Real.sqrt 3 :=
sorry

end sin_add_tan_special_angles_l377_377411


namespace product_of_real_roots_l377_377444

noncomputable def poly := (λ (x : ℝ), x^4 - 6 * x - 3)

theorem product_of_real_roots : ∃ r₁ r₂ : ℝ, poly r₁ = 0 ∧ poly r₂ = 0 ∧ r₁ * r₂ = -1 := 
sorry

end product_of_real_roots_l377_377444


namespace complex_equation_l377_377986

theorem complex_equation (z : ℂ) (h : 10 * (complex.abs z) ^ 2 = 
    3 * (complex.abs (z + 1)) ^ 2 + (complex.abs (z ^ 2 - 2)) ^ 2 + 42) :
  z + 10 / z = 1 := 
sorry

end complex_equation_l377_377986


namespace max_perfect_squares_l377_377657

theorem max_perfect_squares (a b : ℕ) (h_d : a ≠ b) :
  2 ≤ card (filter is_square [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)]) :=
begin
  sorry
end

end max_perfect_squares_l377_377657


namespace trapezoid_EFGH_area_l377_377285

-- Define points E, F, G, H
def E := (0, 0 : ℝ × ℝ)
def F := (0, 3 : ℝ × ℝ)
def G := (5, 3 : ℝ × ℝ)
def H := (3, 0 : ℝ × ℝ)

-- Verify that these points form a trapezoid
def is_trapezoid (E F G H : ℝ × ℝ) : Prop :=
  let (x1, y1) := E
  let (x2, y2) := F
  let (x3, y3) := G
  let (x4, y4) := H
  (y1 = y2) ∧ (y3 = y4) -- F to G and E to H are opposite sides of the trapezoid

-- Calculate the area of the trapezoid
def trapezoid_area (E F G H : ℝ × ℝ) : ℝ :=
  let base1 := real.abs (E.2 - F.2) -- y-coordinate difference between E and F
  let base2 := real.abs (G.1 - H.1) -- x-coordinate difference between G and H
  let height := real.abs (E.1 - H.1) -- x-coordinate difference between E and H
  0.5 * (base1 + base2) * height

-- Main theorem to show the area of the trapezoid is 7.5 square units
theorem trapezoid_EFGH_area : trapezoid_area E F G H = 7.5 :=
by 
  -- Hypothesis that the points form a trapezoid
  have h1 : is_trapezoid E F G H := by
    dp.linarith [/(0,3),(5,3),(3,0)]
  sorry

end trapezoid_EFGH_area_l377_377285


namespace positive_difference_of_probabilities_l377_377310

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377310


namespace ordered_triple_solution_l377_377974

theorem ordered_triple_solution (a b c : ℝ) (h1 : a > 5) (h2 : b > 5) (h3 : c > 5)
  (h4 : (a + 3) * (a + 3) / (b + c - 5) + (b + 5) * (b + 5) / (c + a - 7) + (c + 7) * (c + 7) / (a + b - 9) = 49) :
  (a, b, c) = (13, 9, 6) :=
sorry

end ordered_triple_solution_l377_377974


namespace count_of_numbers_in_pascals_triangle_l377_377825

theorem count_of_numbers_in_pascals_triangle (n : ℕ) (h : n = 30) : ∑ k in Finset.range (n + 1), (k + 1) = 465 :=
by
  sorry

end count_of_numbers_in_pascals_triangle_l377_377825


namespace max_value_of_a_l377_377200

-- Define the function f(x) for x > 0 as specified
def f (x : ℝ) : ℝ := -x^2 + 2 * x

-- Define the properties needed for the proof
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonic_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- Combine all conditions into a theorem statement
theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) →
  is_monotonic_increasing f (set.Icc (-1) (a - 2)) →
  a ≤ 3 :=
sorry  -- the proof is omitted

end max_value_of_a_l377_377200


namespace find_k_l377_377275

theorem find_k
  (k : ℝ)
  (P : ℝ × ℝ)
  (S : ℝ × ℝ)
  (QR : ℝ)
  (h1 : P = (5, 12))
  (h2 : S = (0, k))
  (h3 : QR = 4)
  (h4 : ∃ r₁ r₂ : ℝ, r₁ > 0 ∧ r₂ > 0 ∧ S ∈ circle (0, 0) r₂ ∧ P ∈ circle (0, 0) r₁)
  (h5 : r₁ - QR = r₂) :
  k = 9 :=
by
  sorry

end find_k_l377_377275


namespace at_least_one_gt_one_of_sum_gt_two_l377_377984

theorem at_least_one_gt_one_of_sum_gt_two (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 := 
by sorry

end at_least_one_gt_one_of_sum_gt_two_l377_377984


namespace sum_sin_sixths_l377_377814

open Real

theorem sum_sin_sixths :
  (∑ k in Finset.range 181, sin (k * π / 180) ^ 6) = 225 / 4 :=
by
  sorry

end sum_sin_sixths_l377_377814


namespace first_number_is_nine_l377_377642

theorem first_number_is_nine (x : ℤ) (h : 11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) : x = 9 :=
by {
  sorry
}

end first_number_is_nine_l377_377642


namespace notebook_cost_proof_l377_377767

-- Let n be the cost of the notebook and p be the cost of the pen.
variable (n p : ℝ)

-- Conditions:
def total_cost : Prop := n + p = 2.50
def notebook_more_pen : Prop := n = 2 + p

-- Theorem: Prove that the cost of the notebook is $2.25
theorem notebook_cost_proof (h1 : total_cost n p) (h2 : notebook_more_pen n p) : n = 2.25 := 
by 
  sorry

end notebook_cost_proof_l377_377767


namespace fraction_problem_l377_377665

theorem fraction_problem 
  (x : ℚ)
  (h : x = 45 / (8 - (3 / 7))) : 
  x = 315 / 53 := 
sorry

end fraction_problem_l377_377665


namespace total_square_footage_after_expansion_l377_377703

-- Definitions from the conditions
def size_smaller_house_initial : ℕ := 5200
def size_larger_house : ℕ := 7300
def expansion_smaller_house : ℕ := 3500

-- The new size of the smaller house after expansion
def size_smaller_house_after_expansion : ℕ :=
  size_smaller_house_initial + expansion_smaller_house

-- The new total square footage
def new_total_square_footage : ℕ :=
  size_smaller_house_after_expansion + size_larger_house

-- Goal statement: Prove the total new square footage is 16000 sq. ft.
theorem total_square_footage_after_expansion : new_total_square_footage = 16000 := by
  sorry

end total_square_footage_after_expansion_l377_377703


namespace maximum_sphere_radius_squared_l377_377717

def cone_base_radius : ℝ := 4
def cone_height : ℝ := 10
def axes_intersection_distance_from_base : ℝ := 4

theorem maximum_sphere_radius_squared :
  let m : ℕ := 144
  let n : ℕ := 29
  m + n = 173 :=
by
  sorry

end maximum_sphere_radius_squared_l377_377717


namespace function_domain_l377_377246

noncomputable def domain_function (x : ℝ) : Prop :=
  x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0

theorem function_domain :
  { x : ℝ | domain_function x } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end function_domain_l377_377246


namespace find_c_l377_377190

structure ProblemData where
  (r : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h : r (s 3) = 20)

def r (x : ℝ) : ℝ := 5 * x - 10
def s (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

theorem find_c (c : ℝ) (h : (r (s 3 c)) = 20) : c = 6 :=
sorry

end find_c_l377_377190


namespace cone_volume_max_height_l377_377077

noncomputable def cone_max_height : ℝ := 4 / 3

theorem cone_volume_max_height (h : ℝ) (r : ℝ) (V : ℝ) :
  (h - 1)^2 + r^2 = 1 ∧ V = (π / 3) * r^2 * h ∧ 
  (∀ h < cone_max_height, (π / 3) * r^2 * h < V) ∧ (∀ h > cone_max_height, V < (π / 3) * r^2 * h)
  → h = cone_max_height :=
begin
  intros,
  sorry
end

end cone_volume_max_height_l377_377077


namespace find_inverse_eight_l377_377107

def f (x : ℝ) : ℝ := 1 - 3 * (x - 1) + 3 * (x - 1) ^ 2 - (x - 1) ^ 3

theorem find_inverse_eight : f (0) = 8 → 0 = (x : ℝ) where (f x = 8) :=
sorry

end find_inverse_eight_l377_377107


namespace sum_due_is_l377_377674

-- Definitions and conditions from the problem
def BD : ℤ := 288
def TD : ℤ := 240
def face_value (FV : ℤ) : Prop := BD = TD + (TD * TD) / FV

-- Proof statement
theorem sum_due_is (FV : ℤ) (h : face_value FV) : FV = 1200 :=
sorry

end sum_due_is_l377_377674


namespace arithmetic_sequence_problem_l377_377873

variable (d a1 : ℝ)
variable (h1 : a1 ≠ d)
variable (h2 : d ≠ 0)

theorem arithmetic_sequence_problem (S20 M : ℝ)
  (h3 : S20 = 10 * M)
  (x y : ℝ)
  (h4 : M = x * (a1 + 9 * d) + y * d) :
  x = 2 ∧ y = 1 := 
by 
  sorry

end arithmetic_sequence_problem_l377_377873


namespace monotonic_intervals_and_extremum_values_l377_377108

noncomputable def f : ℝ → ℝ := λ x, Real.exp(x) * (x^2 + x + 1)

theorem monotonic_intervals_and_extremum_values :
  (∀ x, x < -2 → f' x > 0) ∧ 
  (∀ x, -2 < x ∧ x < -1 → f' x < 0) ∧ 
  (∀ x, x > -1 → f' x > 0) ∧ 
  (f (-2) = 3 / Real.exp(2)) ∧ 
  (f (-1) = 1 / Real.exp(1)) := 
by
  sorry

end monotonic_intervals_and_extremum_values_l377_377108


namespace problem_l377_377931

open Real

def sum_of_distances_eq_4 (M F1 F2 : Point) : Prop :=
  dist M F1 + dist M F2 = 4

def is_ellipse (C : Set Point) : Prop :=
  ∀ M, M ∈ C ↔ sum_of_distances_eq_4 M (⟨-sqrt 3, 0⟩) (⟨sqrt 3, 0⟩)

def on_the_line (P A B : Point) : Prop :=
  ∃ k : Real, A.y = k * (A.x - 3) ∧ B.y = k * (B.x - 3)

def const_value_QA_dot_QB (Q A B : Point) (C : Set Point) : Prop :=
  ∃ c : Real, ∀ A B ∈ C, Q ≠ A ∧ Q ≠ B → A ≠ B → 
    ((A - Q) • (B - Q)) = c

theorem problem (C : Set Point) :
  is_ellipse C ∧ ∀ P A B (l : Line), 
  P = ⟨3, 0⟩ ∧ on_the_line P A B → 
  ∃ Q, Q = (⟨19 / 8, 0⟩) ∧ const_value_QA_dot_QB Q A B C :=
sorry

end problem_l377_377931


namespace diff_of_squares_1500_l377_377535

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377535


namespace average_speed_calculation_l377_377140

-- Definitions based on the conditions provided
def D : ℝ := 1  -- Assume total distance D is some arbitrary positive value, normalized to 1 for simplicity.
def t1 : ℝ := D / 240  -- Time for first third of the distance
def t2 : ℝ := D / 72   -- Time for second third of the distance
def t3 : ℝ := D / 90   -- Time for last third of the distance
def total_time : ℝ := t1 + t2 + t3

-- Conversion using common denominator (720 in this case)
def t1_720 : ℝ := 3 * D / 720
def t2_720 : ℝ := 10 * D / 720
def t3_720 : ℝ := 8 * D / 720

-- Average speed calculated
def avg_speed : ℝ := D / total_time

-- The main theorem statement proving the average speed is as calculated
theorem average_speed_calculation : avg_speed = 34.2857 :=
by
  -- The actual proof logic is omitted, hence, using sorry
  sorry

end average_speed_calculation_l377_377140


namespace largest_prime_divisor_of_sum_cubed_s_divisors_2014_2014_l377_377968

def s (n : ℕ) : ℕ := n.divisors.count.to_nat

def sum_cubed_s_divisors (n : ℕ) : ℕ :=
  (n.divisors.to_list.map (λ k, s(k) ^ 3)).sum

theorem largest_prime_divisor_of_sum_cubed_s_divisors_2014_2014 :
  nat.greatest_prime_divisor (sum_cubed_s_divisors (2014 ^ 2014)) = 31 :=
sorry

end largest_prime_divisor_of_sum_cubed_s_divisors_2014_2014_l377_377968


namespace edge_length_of_cube_l377_377125

theorem edge_length_of_cube (V : ℝ) (e : ℝ) (h1 : V = 2744) (h2 : V = e^3) : e = 14 := 
by 
  sorry

end edge_length_of_cube_l377_377125


namespace shark_ratio_l377_377026

theorem shark_ratio (N D : ℕ) (h1 : N = 22) (h2 : D + N = 110) (h3 : ∃ x : ℕ, D = x * N) : 
  (D / N) = 4 :=
by
  -- conditions use only definitions given in the problem.
  sorry

end shark_ratio_l377_377026


namespace isosceles_trapezoid_area_eq569275_l377_377003

theorem isosceles_trapezoid_area_eq569275
  (leg_length : ℝ) 
  (diagonal_length : ℝ) 
  (long_base : ℝ)
  (h_leg : leg_length = 25)
  (h_diagonal : diagonal_length = 34)
  (h_long_base : long_base = 40) : 
  (∃ area : ℝ, 
    area = (1/2) * (long_base + (long_base - 2 * real.sqrt (leg_length^2 - (1/2 * (diagonal_length^2 - leg_length^2 + long_base^2)))) * real.sqrt (diagonal_length^2 - (1/2 * (diagonal_length^2 - leg_length^2 + long_base^2))^2)) * 
    real.sqrt (leg_length^2 - (1/2 * (diagonal_length^2 - leg_length^2 + long_base^2))^2)
    ∧ area = 569.275) := sorry

end isosceles_trapezoid_area_eq569275_l377_377003


namespace probability_last_ball_white_l377_377806

variables (n m : ℕ)

def is_odd (n : ℕ) : Prop := n % 2 = 1
def urn_final_ball_is_white (n : ℕ) : ℕ :=
  if is_odd n then 1 else 0

theorem probability_last_ball_white (n m : ℕ) :
  urn_final_ball_is_white n = 
  if is_odd n then 1 else 0 :=
sorry

end probability_last_ball_white_l377_377806


namespace angle_between_vectors_l377_377118

-- Conditions
variables {a b : ℝ} {θ : ℝ}
-- The magnitude of vectors a and b
def mag_a : ℝ := 1
def mag_b : ℝ := Real.sqrt 2

-- Dot product of vectors a and b
def dot_prod : ℝ := 1

-- The target Lean statement
theorem angle_between_vectors :
  mag_a * mag_b * Real.cos θ = dot_prod → θ = Real.arccos (1 / Real.sqrt 2) → θ = (Real.pi / 4) :=
by
  intros hθ hcos
  sorry

end angle_between_vectors_l377_377118


namespace half_angle_quadrant_l377_377569

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 / 2 * Real.pi)
  : (k % 2 = 0 → k * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi + 3 / 4 * Real.pi) ∨
    (k % 2 = 1 → (k + 1) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k + 1) * Real.pi + 3 / 4 * Real.pi) :=
by
  sorry

end half_angle_quadrant_l377_377569


namespace arithmetic_seq_S10_l377_377594

open BigOperators

variables (a : ℕ → ℚ) (d : ℚ)

-- Definitions based on the conditions
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) := ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 5 = 1
axiom h2 : a 1 + a 7 + a 10 = a 4 + a 6

-- We aim to prove the sum of the first 10 terms
def S (n : ℕ) :=
  ∑ i in Finset.range n, a (i + 1)

theorem arithmetic_seq_S10 : arithmetic_seq a d → S a 10 = 25 / 3 :=
by
  sorry

end arithmetic_seq_S10_l377_377594


namespace count_diff_of_squares_l377_377558

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377558


namespace trig_identity_simplification_l377_377234

theorem trig_identity_simplification :
  (sin (40 * (Real.pi / 180)) + sin (80 * (Real.pi / 180))) / 
  (cos (40 * (Real.pi / 180)) + cos (80 * (Real.pi / 180))) = Real.sqrt 3 :=
by sorry

end trig_identity_simplification_l377_377234


namespace find_b100_l377_377816

def sequence (n : ℕ) : ℚ :=
  if h₁ : n = 1 then 2
  else if h₂ : n = 2 then 1
  else if h₃ : n ≥ 3 then (2 - sequence (n - 1)) / (3 * sequence (n - 2))
  else 0

theorem find_b100 : sequence 100 = 101 / 300 :=
by {
  sorry
}

end find_b100_l377_377816


namespace number_of_integers_as_difference_of_squares_l377_377526

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377526


namespace mode_of_data_set_l377_377782

-- Definition: A mode is a value that appears most frequently in a set.
def mode (l : List ℕ) : ℕ := l.foldr (λ n acc, if l.count n > l.count acc then n else acc) (l.head!)

-- Set of data provided as a condition:
def data_set : List ℕ := [2, 3, 3, 2, 2]

-- The target statement to prove:
theorem mode_of_data_set : mode data_set = 2 := by
  sorry

end mode_of_data_set_l377_377782


namespace matilda_father_chocolates_left_l377_377206

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l377_377206


namespace sqrt_fraction_expression_l377_377433

theorem sqrt_fraction_expression : 
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + (Real.sqrt (9 / 4) + Real.sqrt (4 / 9))^2) = (199 / 36) := 
by
  sorry

end sqrt_fraction_expression_l377_377433


namespace positive_difference_probability_l377_377299

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377299


namespace beads_arrangement_count_l377_377162

theorem beads_arrangement_count : 
  ∃ n : ℕ, n = 41 ∧ ∀ (beads : ℕ → Prop),
  (∀ i, beads i ∈ {0, 1}) ∧
  (∀ i, ¬(beads i = 1 ∧ beads ((i + 1) % 13) = 1)) ∧
  (∀ i j, beads i = beads j → (beads (i + 1) % 13) = (beads (j + 1) % 13)) →
  (count_unique_arrangements beads 13 = n) :=
sorry

end beads_arrangement_count_l377_377162


namespace circumscribed_sphere_contains_l377_377191

-- Define the conditions for the problem
variables {I A' B' C' D' O L H : Point}
variables {R r : Real}
variables {ABCD IBCD ICDA IDBA IABC : Tetrahedron}

-- Given statements
def is_center_inscribed_sphere (I : Point) (T : Tetrahedron) : Prop := sorry
def is_center_circumscribed_sphere (A' : Point) (T : Tetrahedron) : Prop := sorry
def center_circumscribed_sphere (T : Tetrahedron) : Point := sorry
def radius_circumscribed_sphere (T : Tetrahedron) : Real := sorry
def radius_inscribed_sphere (T : Tetrahedron) : Real := sorry
def circumcenter_triangle (L : Point) (Δ : Triangle) : Prop := sorry
def projection_point_on_plane (H : Point) (P : Plane) : Point := sorry

-- Main theorem statement
theorem circumscribed_sphere_contains 
  (h_inscrib_center : is_center_inscribed_sphere I ABCD)
  (h_circum_center_A' : is_center_circumscribed_sphere A' IBCD)
  (h_circum_center_B' : is_center_circumscribed_sphere B' ICDA)
  (h_circum_center_C' : is_center_circumscribed_sphere C' IDBA)
  (h_circum_center_D' : is_center_circumscribed_sphere D' IABC)
  (h_center_circum : O = center_circumscribed_sphere ABCD)
  (h_radius_circum : R = radius_circumscribed_sphere ABCD)
  (h_radius_inscribed : r = radius_inscribed_sphere ABCD)
  (h_circumcenter_L: circumcenter_triangle L (Δ ABC))
  (h_projection_H: H = projection_point_on_plane I (Plane_of_triangle ABC)) :
  ∃ R' : Real, 
    (R' = radius_circumscribed_sphere (Tetrahedron_construct A' B' C' D')) ∧ 
    (R ≤ R') :=
sorry

end circumscribed_sphere_contains_l377_377191


namespace mike_needs_more_money_l377_377635

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end mike_needs_more_money_l377_377635


namespace combine_dolls_l377_377172

theorem combine_dolls (j_dolls : ℕ) (g_dolls : ℕ) (hj : j_dolls = 1209) (hg : g_dolls = 2186) :
  j_dolls + g_dolls = 3395 :=
by
  rw [hj, hg]
  norm_num

end combine_dolls_l377_377172


namespace money_needed_l377_377633

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end money_needed_l377_377633


namespace spherical_coordinates_l377_377820

noncomputable def rho (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

noncomputable def phi (z : ℝ) (ρ : ℝ) : ℝ := Real.acos (z / ρ)

noncomputable def theta (x y ρ φ : ℝ) : ℝ := Real.atan2 y x

theorem spherical_coordinates (x y z : ℝ) (ρ θ φ : ℝ) :
    -2 = x → 2 * Real.sqrt 3 = y → 4 = z →
    ρ > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi → 0 ≤ φ ∧ φ ≤ Real.pi →
    ρ = rho x y z ∧ φ = phi z ρ ∧ θ = θ -2 2 * Real.sqrt 3 ρ φ →
    ρ = 4 * Real.sqrt 2 ∧ θ = 2 * Real.pi / 3 ∧ φ = Real.pi / 4 := by
  intros
  sorry

end spherical_coordinates_l377_377820


namespace points_are_symmetric_wrt_z_axis_l377_377169

-- Definition of point A and point B
def PointA : ℝ × ℝ × ℝ := (2, 2, 4)
def PointB : ℝ × ℝ × ℝ := (-2, -2, 4)

-- Condition for symmetry with respect to the z-axis
def symmetric_z_axis (p1 p2 : ℝ × ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2 ∧ p1.3 = p2.3

-- Theorem statement
theorem points_are_symmetric_wrt_z_axis : symmetric_z_axis PointA PointB := 
  sorry

end points_are_symmetric_wrt_z_axis_l377_377169


namespace minimum_phi_symmetric_y_axis_l377_377143

def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)
def g (x ϕ : ℝ) : ℝ := f (x + ϕ)

theorem minimum_phi_symmetric_y_axis :
  (∀ (x : ℝ), g(x, ϕ) = g(-x, ϕ)) ↔ ϕ = π / 8 :=
by
  sorry

end minimum_phi_symmetric_y_axis_l377_377143


namespace circumcircle_A1B1C1_contains_P_and_Q_l377_377183

-- Definitions of Brocard points should come from the conditions provided, 
-- no explicit construction steps or detailed properties from the solution.

noncomputable def BrocardPoint1 (A B C : Point) : Point := sorry
noncomputable def BrocardPoint2 (A B C : Point) : Point := sorry

noncomputable def intersection (l1 l2 : Line) : Point := sorry

variable {A B C P Q A1 B1 C1 : Point}

-- Assume the given conditions:
axiom h1 : P = BrocardPoint1 A B C
axiom h2 : Q = BrocardPoint2 A B C
axiom h3 : A1 = intersection (line_through C P) (line_through B Q)
axiom h4 : B1 = intersection (line_through A P) (line_through C Q)
axiom h5 : C1 = intersection (line_through B P) (line_through A Q)

theorem circumcircle_A1B1C1_contains_P_and_Q :
  ∃ (circumcircle : Circle), P ∈ circumcircle ∧ Q ∈ circumcircle ∧
  A1 ∈ circumcircle ∧ B1 ∈ circumcircle ∧ C1 ∈ circumcircle := sorry

end circumcircle_A1B1C1_contains_P_and_Q_l377_377183


namespace triangle_area_is_correct_l377_377625

def vector_a : ℝ × ℝ × ℝ := (4, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (-3, 3, 2)

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((u.2 * v.3 - u.3 * v.2), (u.3 * v.1 - u.1 * v.3), (u.1 * v.2 - u.2 * v.1))

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def triangle_area (u v : ℝ × ℝ × ℝ) : ℝ :=
(magnitude (cross_product u v) / 2)

theorem triangle_area_is_correct :
  triangle_area vector_a vector_b = Real.sqrt 446 / 2 :=
by
  sorry

end triangle_area_is_correct_l377_377625


namespace find_circle_center_l377_377376

theorem find_circle_center :
  ∃ (a b : ℝ), a = 1 / 2 ∧ b = 7 / 6 ∧
  (0 - a)^2 + (1 - b)^2 = (1 - a)^2 + (1 - b)^2 ∧
  (1 - a) * 3 = b - 1 :=
by {
  sorry
}

end find_circle_center_l377_377376


namespace unique_four_digit_number_l377_377729

theorem unique_four_digit_number (a b c d : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) (hd : d ≤ 9)
  (h1 : a + b = c + d)
  (h2 : b + d = 2 * (a + c))
  (h3 : a + d = c)
  (h4 : b + c - a = 3 * d) :
  a = 1 ∧ b = 8 ∧ c = 5 ∧ d = 4 :=
by
  sorry

end unique_four_digit_number_l377_377729


namespace distance_between_Tolya_and_Anton_when_Anton_finishes_l377_377364

-- Define the speeds and constants
variables (v_A v_S v_T : ℝ) (t_A : ℝ)

-- Define the conditions as premises
axiom (cond1 : ∀ t_A, v_A * t_A = 100)
axiom (cond2 : ∀ t_A, v_S * t_A = 90)
axiom (cond3 : v_S = 0.9 * v_A)
axiom (cond4 : v_T = 0.81 * v_A)

-- State the problem as a theorem
theorem distance_between_Tolya_and_Anton_when_Anton_finishes :
  let distance_Tolya_Anton := 100 - (v_T * t_A) in
  distance_Tolya_Anton = 19 :=
by
  sorry

end distance_between_Tolya_and_Anton_when_Anton_finishes_l377_377364


namespace positive_difference_prob_3_and_4_heads_l377_377288

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377288


namespace certain_event_is_C_l377_377401

def event_A : Prop := ∀ (tv_on : Bool), tv_on = true → ∃ (show_commercial : Bool), show_commercial = true
def event_B : Prop := ∃ (height : ℝ), ∀ (thumbtack_drop : Bool), thumbtack_drop = true → ∃ (lands_point_up : Bool), lands_point_up = true
def event_C : Prop := ∀ (jar : Set Color), (∀ (ball : Color), ball ∈ jar → ball = Color.white) → ∃ (drawn_ball : Color), drawn_ball = Color.white
def event_D : Prop := ∃ (date : ℕ), date = 1 → ∃ (month : ℕ), month = 10 → ∃ (city : String), city = "Xiamen" → ∃ (weather : String), weather = "sunny"

theorem certain_event_is_C : event_C :=
by sorry

end certain_event_is_C_l377_377401


namespace boolean_logic_problem_l377_377367

theorem boolean_logic_problem (p q : Prop) (h₁ : ¬(p ∧ q)) (h₂ : ¬(¬p)) : ¬q :=
by {
  sorry
}

end boolean_logic_problem_l377_377367


namespace tilde_p_plus_tilde_q_l377_377253

noncomputable def question_and_conditions (p q : ℕ) (b : ℚ) (x sum : ℚ) :=
  p > 0 ∧ q > 0 ∧ Nat.coprime p q ∧ b = (p : ℚ) / (q : ℚ) ∧ sum = 4032 ∧
  (∀ x : ℚ, ∃ w f : ℚ, w = nat.floor x ∧ f = x - w ∧ 0 ≤ f ∧ f < 1 ∧
   w * f = b * x^2 + 1 / w -> x) ∧ 
  (sum = ∑ x, x)

theorem tilde_p_plus_tilde_q (p q : ℕ) (b : ℚ) (sum : ℚ) :
  question_and_conditions p q b sum → p + q = 15 := 
sorry

end tilde_p_plus_tilde_q_l377_377253


namespace find_A_l377_377028

def heartsuit (A B : ℤ) : ℤ := 4 * A + A * B + 3 * B + 6

theorem find_A (A : ℤ) : heartsuit A 3 = 75 ↔ A = 60 / 7 := sorry

end find_A_l377_377028


namespace john_replace_bedroom_doors_l377_377173

variable (B O : ℕ)
variable (cost_outside cost_bedroom total_cost : ℕ)

def john_has_to_replace_bedroom_doors : Prop :=
  let outside_doors_replaced := 2
  let cost_of_outside_door := 20
  let cost_of_bedroom_door := 10
  let total_replacement_cost := 70
  O = outside_doors_replaced ∧
  cost_outside = cost_of_outside_door ∧
  cost_bedroom = cost_of_bedroom_door ∧
  total_cost = total_replacement_cost ∧
  20 * O + 10 * B = total_cost →
  B = 3

theorem john_replace_bedroom_doors : john_has_to_replace_bedroom_doors B O cost_outside cost_bedroom total_cost :=
sorry

end john_replace_bedroom_doors_l377_377173


namespace diff_of_squares_count_l377_377516

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377516


namespace find_f_find_k_l377_377460

-- Definition of the function f and the conditions
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

-- Functional equation given in the problem
axiom h1 : ∀ x y : ℝ, f(x + y) - f(y) = (x + 2 * y - 2) * x

-- Additional condition f(1) = 0
axiom h2 : f 1 = 0

-- Definition of g(x)
def g (x : ℝ) : ℝ := (f x - 2 * x) / x

-- Inequality condition for g(2^x)
axiom h3 : ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), g (2^x) - k * (2^x) ≤ 0

-- Theorem stating the explicit expression for f(x)
theorem find_f : f = λ x : ℝ, (x - 1)^2 :=
sorry

-- Theorem stating the range of the constant k
theorem find_k (k : ℝ) : k ∈ set.Ici (1 : ℝ) :=
sorry


end find_f_find_k_l377_377460


namespace quadratic_function_two_distinct_roots_l377_377848

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks the conditions for the quadratic to have two distinct real roots
theorem quadratic_function_two_distinct_roots (a : ℝ) : 
  (0 < a ∧ a < 2) → (discriminant a (-4) 2 > 0) :=
by
  sorry

end quadratic_function_two_distinct_roots_l377_377848


namespace Luiza_start_distance_l377_377005

noncomputable def Luiza_distance_before_start (A B : ℕ) (distance_ana_left : ℕ) : ℕ :=
  let total_distance := B - A
  let v_ana := total_distance - distance_ana_left
  let ratio := total_distance.to_rat / v_ana.to_rat
  let x := ratio * total_distance - total_distance
  x.to_nat

theorem Luiza_start_distance (A B : ℕ) (distance_ana_left : ℕ) :
  A = 0 ∧ B = 3000 ∧ distance_ana_left = 120 →
  Luiza_distance_before_start A B distance_ana_left = 125 :=
by
  intros h
  cases h with hA h
  cases h with hB h
  have hAna : distance_ana_left = 120 := h
  rw [hA, hB, hAna]
  unfold Luiza_distance_before_start
  norm_num
  sorry

end Luiza_start_distance_l377_377005


namespace probability_even_product_odd_sum_l377_377269

-- Define the set and conditions
def numSet : set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Function to calculate the number of combinations of k elements from a set of n elements
def combinations (n : ℕ) (k : ℕ) : ℕ := (nat.choose n k)

-- Define the probability calculation function
def probability {A B : Type} {Ω : finset A} {f : A → B} (P : finset B) : ℚ :=
  (P.card : ℚ) / (Ω.card : ℚ)

-- Define the main theorem to prove the probability
theorem probability_even_product_odd_sum : 
  let total_selections := combinations 7 3 in
  let acceptable_cases := 20 in -- we assume this from problem statement for probability
  let expected_probability := acceptable_cases in
  probability (finset.attach (sorry : finset {s : finset ℕ // s.card = 3 ∧ s ⊆ numSet ∧
    (∃ a b c : ℕ, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ 
                  ((a*b*c) % 2 = 0 ∧ (a+b+c) % 2 = 1)})) sorry) = 4 / 7 := sorry

end probability_even_product_odd_sum_l377_377269


namespace quadratic_inequality_l377_377491

theorem quadratic_inequality (t x₁ x₂ : ℝ) (α β : ℝ)
  (ht : (2 * x₁^2 - t * x₁ - 2 = 0) ∧ (2 * x₂^2 - t * x₂ - 2 = 0))
  (hx : α ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ β)
  (hαβ : α < β)
  (roots : α + β = t / 2 ∧ α * β = -1) :
  4*x₁*x₂ - t*(x₁ + x₂) - 4 < 0 := 
sorry

end quadratic_inequality_l377_377491


namespace curve_eccentricity_d_l377_377001

theorem curve_eccentricity_d : (∀ x y : ℝ, x^2 / 9 + y^2 = 1 → 2 * Real.sqrt 2 / 3) :=
sorry

end curve_eccentricity_d_l377_377001


namespace tangent_line_at_one_l377_377144

variables {R : Type*} [LinearOrderedField R]
variables (f : R → R)

theorem tangent_line_at_one (h₁ : ∃ M, M = (1, f 1))
    (h₂ : ∀ x, f'(x) = 1 ∧ f(1)=3) : 
    f(1) + f'(1) = 4 := 
by
  sorry

end tangent_line_at_one_l377_377144


namespace find_f_neg_3_l377_377073

def f (x a : ℝ) : ℝ := x * (Real.sin x + 1) + a * x^2

theorem find_f_neg_3 (a : ℝ) (ha : f 3 a = 5) : f (-3) a = -1 := by
  sorry

end find_f_neg_3_l377_377073


namespace coin_flip_probability_difference_l377_377316

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377316


namespace relationship_among_abc_l377_377071

-- Define a, b, c
def a : ℕ := 22 ^ 55
def b : ℕ := 33 ^ 44
def c : ℕ := 55 ^ 33

-- State the theorem regarding the relationship among a, b, and c
theorem relationship_among_abc : a > b ∧ b > c := 
by
  -- Placeholder for the proof, not required for this task
  sorry

end relationship_among_abc_l377_377071


namespace triangle_shape_and_side_length_l377_377942

variables {A B C : ℝ} -- angles in triangle ABC
variables {a b c : ℝ} -- sides opposite to angles A, B, C respectively

theorem triangle_shape_and_side_length (h₁ : log a - log b = log (cos B) - log (cos A))
    (h₂ : ∀ x, ax + 3 = x ↔ (1/3)*x - b = x)
    (ha : a = 3) (hb : b = 1) : 
    ((A = B) ∨ (A = π / 2 ∨ B = π / 2)) ∧ c = √10 :=
by
  sorry

end triangle_shape_and_side_length_l377_377942


namespace functional_equation_l377_377989

noncomputable def f (x : ℚ) (hx : 0 < x) : ℚ :=
  let primes := (factorization x).toFinset \u00A0in
  primes.fold (1 : ℚ) (λ p acc,
    let ai := (factorization x).coeff p in
    let q_i := if isEven p then p.prev else p.succ in
      acc * q_i ^ ai)

theorem functional_equation (x y : ℚ) (hx : 0 < x) (hy : 0 < y) :
    f (x * f y hy) (mul_pos hx (f_pos y hy)) = (f x hx) / y :=
sorry

end functional_equation_l377_377989


namespace number_of_students_playing_soccer_l377_377598

-- Given conditions
def total_students := 420
def total_boys := 312
def boys_percentage := 0.78
def girls_not_playing_soccer := 53

-- Define the number of girls
def total_girls := total_students - total_boys

-- Define the number of girls playing soccer
def girls_playing_soccer := total_girls - girls_not_playing_soccer

-- Define the total number of students playing soccer
def total_students_playing_soccer := girls_playing_soccer / (1 - boys_percentage)

-- Statement: Proving the number of students playing soccer
theorem number_of_students_playing_soccer : total_students_playing_soccer = 250 :=
by
  -- skipping the proof here by using 'sorry'
  sorry

end number_of_students_playing_soccer_l377_377598


namespace problem_l377_377965

theorem problem (n : ℕ) (x : ℕ → (ℤ)) (h1 : ∀ i, x i = 1 ∨ x i = -1) (h2 : (Σ i : ℕ, i < n) → x (i + 1) = x 0) (h3 : (Σ i : ℕ, i < n) → (Σ j : ℕ, j< n) →  x (i + 1)* (j) :- x(0)  = 0 ):
(n % 4 = 0) :=
begin
    sorry
end

end problem_l377_377965


namespace integer_part_a_2019_l377_377998

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else (sequence (n-1) + 1 / sequence (n-1))

theorem integer_part_a_2019 : ⌊sequence 2019⌋ = 63 := 
  sorry

end integer_part_a_2019_l377_377998


namespace number_of_integers_as_difference_of_squares_l377_377527

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377527


namespace kyunghwan_spent_the_most_l377_377698

-- Define initial pocket money for everyone
def initial_money : ℕ := 20000

-- Define remaining money
def remaining_S : ℕ := initial_money / 4
def remaining_K : ℕ := initial_money / 8
def remaining_D : ℕ := initial_money / 5

-- Calculate spent money
def spent_S : ℕ := initial_money - remaining_S
def spent_K : ℕ := initial_money - remaining_K
def spent_D : ℕ := initial_money - remaining_D

theorem kyunghwan_spent_the_most 
  (h1 : remaining_S = initial_money / 4)
  (h2 : remaining_K = initial_money / 8)
  (h3 : remaining_D = initial_money / 5) :
  spent_K > spent_S ∧ spent_K > spent_D :=
by
  -- Proof skipped
  sorry

end kyunghwan_spent_the_most_l377_377698


namespace green_paint_quarts_l377_377453

theorem green_paint_quarts (blue green white : ℕ) (h_ratio : 3 = blue ∧ 2 = green ∧ 4 = white) 
  (h_white_paint : white = 12) : green = 6 := 
by
  sorry

end green_paint_quarts_l377_377453


namespace fibonacci_sequence_last_two_digits_l377_377929

def is_fibonacci_sequence (seq : List Nat) : Prop :=
  ∀ (n : Nat), 2 ≤ n → n < seq.length → seq[n] = seq[n - 1] + seq[n - 2]

theorem fibonacci_sequence_last_two_digits :
  ∃ (a b : Nat), a = 2 ∧ b = 1 ∧ is_fibonacci_sequence [1, 1, 2, 3, 5, 8, 13, a, b] :=
by
  sorry

end fibonacci_sequence_last_two_digits_l377_377929


namespace ravi_jump_height_l377_377648

theorem ravi_jump_height (j1 j2 j3 : ℕ) (average : ℕ) (ravi_jump_height : ℕ) (h : j1 = 23 ∧ j2 = 27 ∧ j3 = 28) 
  (ha : average = (j1 + j2 + j3) / 3) (hr : ravi_jump_height = 3 * average / 2) : ravi_jump_height = 39 :=
by
  sorry

end ravi_jump_height_l377_377648


namespace brocard_angle_property_l377_377420

-- Define the triangle ABC with angles α, β, and γ.
variables {α β γ ω : ℝ}

-- Define the condition that ω is the Brocard angle such that angle KAB = angle KBC = angle KCA.
def brocard_angle (α β γ ω : ℝ) : Prop :=
  ∃ (K : EuclideanGeometry.Point),
    (EuclideanGeometry.angle K (EuclideanGeometry.Point.mk 0 0) (EuclideanGeometry.Point.mk 1 0) = ω) ∧
    (EuclideanGeometry.angle K (EuclideanGeometry.Point.mk 1 0) (EuclideanGeometry.Point.mk 0 1) = ω) ∧
    (EuclideanGeometry.angle K (EuclideanGeometry.Point.mk 0 1) (EuclideanGeometry.Point.mk 0 0) = ω)

-- Prove the desired property of the Brocard angle.
theorem brocard_angle_property (α β γ : ℝ) (hαβγ : α + β + γ = π) :
  brocard_angle α β γ ω → real.cot ω = real.cot α + real.cot β + real.cot γ :=
by
  sorry

end brocard_angle_property_l377_377420


namespace must_be_same_type_l377_377949

-- Definitions based on conditions.
def Person : Type := Type
def A : Person := sorry
def B : Person := sorry
def C : Person := sorry

-- A and B are not brothers given C is a distinct person.
axiom not_brothers (A : Person) (B : Person) (C : Person) : ¬(A = B) ∧ A ≠ C ∧ B ≠ C

-- A and B being of the same type.
def same_type (A : Person) (B : Person) : Prop := sorry

-- The main theorem stating that A and B must be of the same type.
theorem must_be_same_type (A B : Person) (C : Person) (h : ¬(A = B) ∧ A ≠ C ∧ B ≠ C) : same_type A B :=
sorry

end must_be_same_type_l377_377949


namespace maximize_monthly_profit_l377_377675

-- Definitions from the conditions
def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def initial_units_sold_per_month : ℕ := 210
def price_increase_effect_on_units_sold : ℕ := 10
def max_selling_price_per_unit : ℕ := 65

-- Function representing monthly profit
def profit (x : ℕ) := 
  (initial_units_sold_per_month - price_increase_effect_on_units_sold * x) *
  (initial_selling_price_per_unit + x - cost_price_per_unit)

-- The range for x based on the problem statement
def valid_x (x : ℕ) := 0 < x ∧ x ≤ 15

-- The monthly profit function
def monthly_profit (x : ℕ) : ℕ :=
  if valid_x x then profit x else 0

-- The statement that encodes finding the maximum profit
def max_profit := 2400

-- Conditions for the selling price to achieve maximum profit.
def optimal_prices := {price | 
  (price = initial_selling_price_per_unit + 5 ∨ 
   price = initial_selling_price_per_unit + 6) ∧ 
  monthly_profit (price - initial_selling_price_per_unit) = max_profit}

-- Lean Proof Statement
theorem maximize_monthly_profit :
  ∃ price, price ∈ optimal_prices :=
sorry

end maximize_monthly_profit_l377_377675


namespace polygons_with_A1_more_than_without_l377_377865

variable (n : ℕ)
variable (A : Fin n → Type)

def num_polygons_containing_A1 : ℕ := 
  ∑ k in Finset.range(2, n), Nat.choose (n - 1) k

def num_polygons_not_containing_A1 : ℕ :=
  ∑ k in Finset.range(3, n), Nat.choose n k - num_polygons_containing_A1

theorem polygons_with_A1_more_than_without (h : n = 16) :
  num_polygons_containing_A1 n A > num_polygons_not_containing_A1 n A := 
by
  sorry

end polygons_with_A1_more_than_without_l377_377865


namespace find_value_of_expression_l377_377846

variable {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : x + y = x * y + 1)

theorem find_value_of_expression (h : x + y = x * y + 1) : 
  (1 / x) + (1 / y) = 1 + (1 / (x * y)) :=
  sorry

end find_value_of_expression_l377_377846


namespace total_snowfall_l377_377924

theorem total_snowfall (morning_snowfall : ℝ) (afternoon_snowfall : ℝ) (h_morning : morning_snowfall = 0.125) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 :=
by 
  sorry

end total_snowfall_l377_377924


namespace max_perfect_squares_l377_377661

theorem max_perfect_squares (a b : ℕ) (h : a ≠ b) : 
  let prod1 := a * (a + 2),
      prod2 := a * b,
      prod3 := a * (b + 2),
      prod4 := (a + 2) * b,
      prod5 := (a + 2) * (b + 2),
      prod6 := b * (b + 2) in
  (prod1.is_square ℕ + prod2.is_square ℕ + prod3.is_square ℕ + prod4.is_square ℕ + prod5.is_square ℕ + prod6.is_square ℕ) ≤ 2 := 
  sorry

end max_perfect_squares_l377_377661


namespace coin_flip_probability_difference_l377_377339

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377339


namespace zero_order_l377_377489
noncomputable def f (x : ℝ) : ℝ := 2 * x + x
noncomputable def g (x : ℝ) : ℝ := real.log (2) x + x
noncomputable def h (x : ℝ) : ℝ := x^3 + x

def zero_of_f : ℝ := classical.some (exists_zero_of_continuous f)
def zero_of_g : ℝ := classical.some (exists_zero_of_continuous g)
def zero_of_h : ℝ := 0

theorem zero_order (a b c : ℝ) 
  (Hf : f a = 0) (Hg : g b = 0) (Hh : h c = 0)
  (H1 : a ∈ set.Ioo (-1 : ℝ) 0)
  (H2 : b ∈ set.Ioo (0 : ℝ) 1)
  (H3 : c = 0) :
  a < c ∧ c < b :=
by sorry

end zero_order_l377_377489


namespace find_value_m_sq_plus_2m_plus_n_l377_377987

noncomputable def m_n_roots (x : ℝ) : Prop := x^2 + x - 1001 = 0

theorem find_value_m_sq_plus_2m_plus_n
  (m n : ℝ)
  (hm : m_n_roots m)
  (hn : m_n_roots n)
  (h_sum : m + n = -1)
  (h_prod : m * n = -1001) :
  m^2 + 2 * m + n = 1000 :=
sorry

end find_value_m_sq_plus_2m_plus_n_l377_377987


namespace population_control_l377_377610

   noncomputable def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
   initial_population * (1 + growth_rate / 100) ^ years

   theorem population_control {initial_population : ℝ} {threshold_population : ℝ} {growth_rate : ℝ} {years : ℕ} :
     initial_population = 1.3 ∧ threshold_population = 1.4 ∧ growth_rate = 0.74 ∧ years = 10 →
     population_growth initial_population growth_rate years < threshold_population :=
   by
     intros
     sorry
   
end population_control_l377_377610


namespace parametric_to_cartesian_l377_377895

variable (R t : ℝ)

theorem parametric_to_cartesian (x y : ℝ) (h1 : x = R * Real.cos t) (h2 : y = R * Real.sin t) : 
  x^2 + y^2 = R^2 := 
by
  sorry

end parametric_to_cartesian_l377_377895


namespace probability_of_at_least_one_accurate_forecast_l377_377230

theorem probability_of_at_least_one_accurate_forecast (PA PB : ℝ) (hA : PA = 0.8) (hB : PB = 0.75) :
  1 - ((1 - PA) * (1 - PB)) = 0.95 :=
by
  rw [hA, hB]
  sorry

end probability_of_at_least_one_accurate_forecast_l377_377230


namespace find_conjugate_l377_377855

def z (z : ℂ) : Prop :=
  z / (1 + 2 * complex.I) = 2 - complex.I

theorem find_conjugate (z : ℂ) (h : z / (1 + 2 * complex.I) = 2 - complex.I) : complex.conj z = 4 - 3 * complex.I :=
sorry

end find_conjugate_l377_377855


namespace range_of_g_g_increasing_on_1_infty_l377_377853

noncomputable def g (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

-- Statement for the range of the function g(x)
theorem range_of_g : set.range g = {y : ℝ | y ≠ 2} :=
sorry

-- Statement for g(x) being an increasing function on [1, +∞)
theorem g_increasing_on_1_infty : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ 1 ≤ x2 ∧ x1 < x2 → g x1 < g x2 :=
sorry

end range_of_g_g_increasing_on_1_infty_l377_377853


namespace number_of_valid_triples_l377_377560

open Nat

theorem number_of_valid_triples :
  let valid_triples := {t : ℕ × ℕ × ℕ | 
    (lcm t.1 t.2 = 180) ∧ 
    (lcm t.1 t.3 = 504) ∧ 
    (lcm t.2 t.3 = 1260) }
  in valid_triples.card = 6 := sorry

end number_of_valid_triples_l377_377560


namespace number_of_odd_divisors_lt_100_l377_377129

theorem number_of_odd_divisors_lt_100 : 
  (∃! (n : ℕ), n < 100 ∧ ∃! (k : ℕ), n = k * k) = 9 :=
sorry

end number_of_odd_divisors_lt_100_l377_377129


namespace find_x_l377_377418

variables (z y x : Int)

def condition1 : Prop := z + 1 = 0
def condition2 : Prop := y - 1 = 1
def condition3 : Prop := x + 2 = -1

theorem find_x (h1 : condition1 z) (h2 : condition2 y) (h3 : condition3 x) : x = -3 :=
by
  sorry

end find_x_l377_377418


namespace standard_01_sequence_count_l377_377824

theorem standard_01_sequence_count (m : ℕ) (h : m = 4) :
  ∃ n, n = 14 ∧ count_standard_01_sequences 4 = n :=
by
  let count_standard_01_sequences := sorry
  have h_seq_count : count_standard_01_sequences 4 = 14 := by sorry
  use 14
  tauto

end standard_01_sequence_count_l377_377824


namespace prism_width_l377_377909

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end prism_width_l377_377909


namespace not_symmetric_about_point_l377_377485

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (4 - x)

theorem not_symmetric_about_point : ¬ (∀ h : ℝ, f (1 + h) = f (1 - h)) :=
by
  sorry

end not_symmetric_about_point_l377_377485


namespace solution_set_of_inequality_l377_377691

theorem solution_set_of_inequality (x : ℝ) : (x^2 - |x| > 0) ↔ (x < -1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l377_377691


namespace number_of_solutions_l377_377441

-- Define the function g
def g (n : ℤ) : ℤ := (⌈ (108 * n : ℚ) / 109 ⌉ - ⌊ (110 * n : ℚ) / 111 ⌋)

-- The statement that we need to prove
theorem number_of_solutions : 
  {n : ℤ | g n = 1}.finite.to_finset.card = 12129 :=
by
  sorry

end number_of_solutions_l377_377441


namespace travel_time_difference_l377_377643

variable (x : ℝ)

theorem travel_time_difference 
  (distance : ℝ) 
  (speed_diff : ℝ)
  (time_diff_minutes : ℝ)
  (personB_speed : ℝ) 
  (personA_speed := personB_speed - speed_diff) 
  (time_diff_hours := time_diff_minutes / 60) :
  distance = 30 ∧ speed_diff = 3 ∧ time_diff_minutes = 40 ∧ personB_speed = x → 
    (30 / (x - 3)) - (30 / x) = 40 / 60 := 
by 
  sorry

end travel_time_difference_l377_377643


namespace sum_of_good_indices_nonneg_l377_377624

noncomputable def is_good_index (a : ℕ → ℝ) (n m : ℕ) (k : ℕ) : Prop :=
∃ (ℓ : ℕ), 1 ≤ ℓ ∧ ℓ ≤ m ∧ (k + ℓ - 1).mod n < n ∧
(a k + ∑ i in range(ℓ - 1), a((k + i).mod n)) ≥ 0

noncomputable def T (a : ℕ → ℝ) (n m : ℕ) : set ℕ := 
{k | 1 ≤ k ∧ k ≤ n ∧ is_good_index a n m k}

theorem sum_of_good_indices_nonneg (a : ℕ → ℝ) (n m : ℕ) (h_mn : m < n) : 
  ∑ k in (finset.filter (λ k, k ∈ T a n m) (finset.range n)), a k ≥ 0 := 
sorry

end sum_of_good_indices_nonneg_l377_377624


namespace sixth_group_number_l377_377780

theorem sixth_group_number (m : ℕ) (n : ℕ) (h1 : n = 9 * m) : (m = 14) → (14 + 5 * 16 = 94) := 
by 
  intros h2,
  rw h2,
  norm_num

end sixth_group_number_l377_377780


namespace count_diff_of_squares_l377_377555

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377555


namespace maximum_area_of_triangle_l377_377150

theorem maximum_area_of_triangle (BC AC : ℝ) (hBC : BC = 4) (hAB : AB = √2 * AC) :
  ∃ x, (x = AC ∧ maximum_area_of_triangle ABC = 8 * √2) :=
sorry

end maximum_area_of_triangle_l377_377150


namespace sum_of_exterior_angles_hexagon_l377_377693

theorem sum_of_exterior_angles_hexagon : 
  (∑ (i : Fin 6), exterior_angle_of_polygon (hexagon i)) = 360 := 
sorry

end sum_of_exterior_angles_hexagon_l377_377693


namespace circles_touching_opposite_points_l377_377268

theorem circles_touching_opposite_points
  (C1 C2 C3 : Circle)
  (O1 O2 O3 : Point)
  (A B C : Point)
  (h1 : C1.center = O1) (h2 : C2.center = O2) (h3 : C3.center = O3)
  (h4 : C1.touches C2 A) (h5 : C2.touches C3 B) (h6 : C3.touches C1 C)
  (h7 : ∀ (p : Point), C1.on p → C2.on p → C3.on p → p ∈ {A, B, C}) :
  ∃ X Y : Point, 
    (line_through A B).intersect_circle C3 = X ∧ 
    (line_through A C).intersect_circle C3 = Y ∧
    diametrically_opposite C3 X Y :=
  sorry

end circles_touching_opposite_points_l377_377268


namespace faster_speed_l377_377771

theorem faster_speed (v : ℝ) (h1 : ∀ (t : ℝ), t = 50 / 10) (h2 : ∀ (d : ℝ), d = 50 + 20) (h3 : ∀ (t : ℝ), t = 70 / v) : v = 14 :=
by
  sorry

end faster_speed_l377_377771


namespace arrangement_count_l377_377369

theorem arrangement_count (students : Fin 6) (teacher : Bool) :
  (teacher = true) ∧
  ∀ (A B : Fin 6), 
    A ≠ 0 ∧ B ≠ 5 →
    A ≠ B →
    (Sorry) = 960 := sorry

end arrangement_count_l377_377369


namespace pow_mod_eq_one_remainder_when_9_pow_150_div_50_l377_377344

theorem pow_mod_eq_one (n : ℕ) (h : n % 10 = 0) : (9^n) % 50 = 1 :=
by sorry

noncomputable def remainder_9_pow_150_mod_50 : ℕ :=
9^150 % 50

theorem remainder_when_9_pow_150_div_50 : remainder_9_pow_150_mod_50 = 1 :=
by {
  show 9^150 % 50 = 1,
  from pow_mod_eq_one 150 (by norm_num)
}

end pow_mod_eq_one_remainder_when_9_pow_150_div_50_l377_377344


namespace registration_methods_count_l377_377060

theorem registration_methods_count :
  let students := 5
  let group_A_min := 2
  let group_B_min := 1
  ∃ (total_method_count : ℕ), total_method_count = 25 := 
by {
  let total_method_count := 
    Nat.choose students group_A_min + 
    Nat.choose students (students - group_A_min - group_B_min + 1) + 
    Nat.choose students (students - 1)
  have h : total_method_count = 25 := sorry
  exact ⟨total_method_count, h⟩
}

end registration_methods_count_l377_377060


namespace points_earned_l377_377365

def each_enemy_points : ℕ := 3
def total_enemies : ℕ := 6
def defeated_enemies : ℕ := total_enemies - 2

theorem points_earned : defeated_enemies * each_enemy_points = 12 :=
by
  -- proof goes here
  sorry

end points_earned_l377_377365


namespace arctan_sum_in_right_triangle_l377_377603

theorem arctan_sum_in_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  (Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4) :=
sorry

end arctan_sum_in_right_triangle_l377_377603


namespace minimum_containers_l377_377764

theorem minimum_containers (capacity_jug : ℕ) (capacity_container : ℕ) (containers_needed : ℕ) :
  capacity_jug = 800 → capacity_container = 48 → containers_needed = (800 + 48 - 1) / 48 → containers_needed = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end minimum_containers_l377_377764


namespace total_revenue_from_sale_l377_377800

def total_weight_of_potatoes : ℕ := 6500
def weight_of_damaged_potatoes : ℕ := 150
def weight_per_bag : ℕ := 50
def price_per_bag : ℕ := 72

theorem total_revenue_from_sale :
  (total_weight_of_potatoes - weight_of_damaged_potatoes) / weight_per_bag * price_per_bag = 9144 := 
begin
  sorry
end

end total_revenue_from_sale_l377_377800


namespace no_such_number_l377_377388

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def productOfDigitsIsPerfectSquare (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ isPerfectSquare (d1 * d2)

theorem no_such_number :
  ¬ ∃ (N : ℕ),
    (N > 9) ∧ (N < 100) ∧ -- N is a two-digit number
    (N % 2 = 0) ∧        -- N is even
    (N % 13 = 0) ∧       -- N is a multiple of 13
    productOfDigitsIsPerfectSquare N := -- The product of digits of N is a perfect square
by
  sorry

end no_such_number_l377_377388


namespace find_smallest_n_l377_377445

theorem find_smallest_n 
  (n : ℕ)
  (sin_x : ℕ → ℝ) 
  (h1 : ∑ i in (Finset.range n).map Finset.nat.succ, sin_x i = 0)
  (h2 : ∑ i in (Finset.range n).map Finset.nat.succ, i * sin_x i = 100) 
  : n = 20 := 
by
  sorry

end find_smallest_n_l377_377445


namespace AD_parallel_BC_l377_377859

-- Defining points and datatypes
variables {Point : Type} [Torus Point]

-- Assuming circles are defined with a center and points on the periphery.
variables {circle : Point → Point → Prop} (O1 O2 A B C D : Point)

-- Conditions on lines passing through the centers and touching the circles
variables {center : Point → Point} {tangent : Point → Point → Point → Prop}

-- The condition that each line passes through the center of a circle and touches another circle, and points are related as given
def conditions : Prop :=
  (center O1 = midpoint A B) ∧
  (center O2 = midpoint C D) ∧
  (tangent O1 A B C D) ∧
  (tangent O2 C D A B)

-- Final proof problem statement
theorem AD_parallel_BC : conditions O1 O2 A B C D → parallel (AD A D) (BC B C) :=
sorry

end AD_parallel_BC_l377_377859


namespace sum_MK_MN_eq_WJ_l377_377592

variable {Point Line : Type}

-- Add geometric definitions and predicates
variable (W X Y Z M N K J L : Point)
variable (WX WY WZ WJ MN MK ML : Line)

def perpendicular (l1 l2 : Line) : Prop := sorry -- Placeholder for perpendicularity predicate
def rectangle (a b c d : Point) : Prop := sorry -- Placeholder for rectangle predicate
def on_segment (p : Point) (l : Line) : Prop := sorry -- Placeholder for point on line segment predicate
def length (l : Line) : ℝ := sorry -- Placeholder for length function

-- Given Conditions
def rect_wxyz := rectangle W X Y Z
def point_M_on_WY := on_segment M WY 
def mn_perp_wz := perpendicular MN WZ
def mk_perp_wx := perpendicular MK WX
def wj_perp_wz := perpendicular WJ WZ
def ml_perp_wj := perpendicular ML WJ

-- The goal statement
theorem sum_MK_MN_eq_WJ 
  (h1 : rect_wxyz) 
  (h2 : point_M_on_WY)
  (h3 : mn_perp_wz)
  (h4 : mk_perp_wx)
  (h5 : wj_perp_wz)
  (h6 : ml_perp_wj) : 
  length MK + length MN = length WJ := 
sorry

end sum_MK_MN_eq_WJ_l377_377592


namespace problem_inequality_l377_377111

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x ^ 2 - (2 * a + 2) * x + (2 * a + 1) * Real.log x

theorem problem_inequality (a x₁ x₂ λ : ℝ) (ha : 1/2 ≤ a ∧ a ≤ 2) (hx₁ : 1 ≤ x₁ ∧ x₁ ≤ 2) (hx₂ : 1 ≤ x₂ ∧ x₂ ≤ 2) (hxneq : x₁ ≠ x₂) (hλ : 6 ≤ λ) :
  |f x₁ a - f x₂ a| < λ * |1 / x₁ - 1 / x₂| :=
sorry

end problem_inequality_l377_377111


namespace probability_heads_odd_after_30_flips_l377_377805

def unfair_coin_probability_heads : ℝ := 3 / 5

def P : ℕ → ℝ 
| 0       := 0
| (n + 1) := (3 / 5) - (1 / 5) * P n

theorem probability_heads_odd_after_30_flips :
  P 30 = 1 / 2 * (1 - 1 / 5^30) :=
by
  sorry

end probability_heads_odd_after_30_flips_l377_377805


namespace positive_difference_prob_3_and_4_heads_l377_377325

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377325


namespace area_of_triangle_KBC_l377_377807

theorem area_of_triangle_KBC : 
  (AB BJ : ℝ) (A B J K C : Point) (H1 : eq_sq_side_lengths ABJ 18) 
  (H2 : eq_sq_side_lengths FEHG 32) (H3 : equilateral JBK) (H4 : EF = BC) : 
  area_triangle K B C = 12 :=
sorry

end area_of_triangle_KBC_l377_377807


namespace trapezoid_area_l377_377804

noncomputable def scale_factor := 250 -- conversion factor from inches to miles

def diagonal_map_inches := 10 -- length of diagonal on map in inches

def diagonal_miles := scale_factor * diagonal_map_inches -- conversion to miles

def half_diagonal := diagonal_miles / 2 -- half of the diagonal in miles

def area_of_one_triangle := (1 / 2) * half_diagonal * half_diagonal -- area of one of the right-angled triangles

def total_area := 4 * area_of_one_triangle -- total area of the trapezoid

theorem trapezoid_area :
  total_area = 3125000 := 
by 
  sorry

end trapezoid_area_l377_377804


namespace infinite_geometric_series_first_term_l377_377803

theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (a : ℝ) 
  (h1 : r = -3/7) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) : 
  a = 180 / 7 := by
  -- omitted proof
  sorry

end infinite_geometric_series_first_term_l377_377803


namespace persons_in_first_group_l377_377235

-- Define the given conditions
def first_group_work_done (P : ℕ) : ℕ := P * 12 * 10
def second_group_work_done : ℕ := 30 * 26 * 6

-- Define the proof problem statement
theorem persons_in_first_group (P : ℕ) (h : first_group_work_done P = second_group_work_done) : P = 39 :=
by
  unfold first_group_work_done second_group_work_done at h
  sorry

end persons_in_first_group_l377_377235


namespace maximum_sum_of_triplets_l377_377217

-- Define a list representing a 9-digit number consisting of digits 1 to 9 in some order
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ ∀ n, n ∈ digits → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]
  
def sum_of_triplets (digits : List ℕ) : ℕ :=
  100 * digits[0]! + 10 * digits[1]! + digits[2]! +
  100 * digits[1]! + 10 * digits[2]! + digits[3]! +
  100 * digits[2]! + 10 * digits[3]! + digits[4]! +
  100 * digits[3]! + 10 * digits[4]! + digits[5]! +
  100 * digits[4]! + 10 * digits[5]! + digits[6]! +
  100 * digits[5]! + 10 * digits[6]! + digits[7]! +
  100 * digits[6]! + 10 * digits[7]! + digits[8]!

theorem maximum_sum_of_triplets :
  ∃ digits : List ℕ, valid_digits digits ∧ sum_of_triplets digits = 4648 :=
  sorry

end maximum_sum_of_triplets_l377_377217


namespace total_clothes_l377_377408

-- Defining the conditions
def shirts := 12
def pants := 5 * shirts
def shorts := (1 / 4) * pants

-- Theorem to prove the total number of pieces of clothes
theorem total_clothes : shirts + pants + shorts = 87 := by
  -- using sorry to skip the proof
  sorry

end total_clothes_l377_377408


namespace P_is_not_optimal_Q_is_optimal_at_most_three_sets_max_elements_in_optimal_set_l377_377971

/-- Definition of an optimal set --/
def is_optimal_set (S : Finset (Finset ℕ)) : Prop :=
  (∀ A ∈ S, A.card = 3) ∧
  (∀ {A B : Finset ℕ}, A ∈ S → B ∈ S → A ≠ B → (A ∩ B).card = 1) ∧
  (Finset.card (Finset.filter (λ (x : ℕ), ∀ A ∈ S, x ∈ A) (⋃₀ S)) = 0)

def P : Finset (Finset ℕ) :=
  { {1, 2, 3}, {2, 4, 5} }

def Q : Finset (Finset ℕ) :=
  { {1, 2, 3}, {1, 4, 5}, {2, 5, 7} }

/-- Test if P is not an optimal set --/
theorem P_is_not_optimal : ¬ is_optimal_set P := sorry

/-- Test if Q is an optimal set --/
theorem Q_is_optimal : is_optimal_set Q := sorry

/-- If S is an optimal set, then for all x in A1, x belongs to at most three sets in S --/
theorem at_most_three_sets (S : Finset (Finset ℕ)) (hS : is_optimal_set S) (A1 ∈ S) (x : ℕ) :
  x ∈ A1 → ∃ (T : Finset (Finset ℕ)), T ⊆ S ∧ T.card ≤ 3 ∧ (∀ B ∈ T, x ∈ B) := sorry

/-- The maximum number of elements in an optimal set is 7 --/
theorem max_elements_in_optimal_set (S : Finset (Finset ℕ)) (hS : is_optimal_set S) :
  S.card ≤ 7 := sorry

end P_is_not_optimal_Q_is_optimal_at_most_three_sets_max_elements_in_optimal_set_l377_377971


namespace num_valid_numbers_l377_377564

lemma count_numbers (x : ℕ) :
  (4 ≤ x ∧ x ≤ 12 ∧ x % 3 ≠ 0) ↔ x ∈ {4, 5, 7, 8, 10, 11} :=
by sorry

theorem num_valid_numbers :
  {x : ℕ | 4 ≤ x ∧ x ≤ 12 ∧ x % 3 ≠ 0}.card = 6 :=
by sorry

end num_valid_numbers_l377_377564


namespace integer_diff_of_squares_1_to_2000_l377_377543

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377543


namespace sequence_x_value_l377_377601

theorem sequence_x_value :
  let a : ℕ → ℕ := 
    fun n => 
      match n with
      | 0 => 1
      | 1 => 3
      | 2 => 6
      | 3 => 10
      | k+4 => a k + (k+4) + 1
  in a 4 = 15 :=
by
  sorry

end sequence_x_value_l377_377601


namespace count_of_squares_difference_l377_377522

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377522


namespace find_range_of_a_l377_377870

noncomputable def log_decreasing_on_interval (a : ℝ) : Prop :=
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → log a (2 - a * x) > log a (2 - a * y)) ∧
  a > 0 ∧ a ≠ 1 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 - a * x > 0)

theorem find_range_of_a : set_of (log_decreasing_on_interval) = {a : ℝ | 1 < a ∧ a < 2} :=
sorry

end find_range_of_a_l377_377870


namespace squared_difference_l377_377566

theorem squared_difference (x y : ℝ) (h₁ : (x + y)^2 = 49) (h₂ : x * y = 8) : (x - y)^2 = 17 := 
by
  -- Proof omitted
  sorry

end squared_difference_l377_377566


namespace modulus_of_given_z_l377_377878

def z : ℂ := (2 * complex.i) / (1 - complex.i)

def modulus (z : ℂ) : ℝ := complex.abs z

theorem modulus_of_given_z : modulus z = real.sqrt 2 := by
  sorry

end modulus_of_given_z_l377_377878


namespace coin_probability_difference_l377_377330

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377330


namespace square_sum_of_corner_cells_l377_377593

theorem square_sum_of_corner_cells (a : ℕ → ℕ) (h1 : ∀ i j, i ≠ j → a i ≠ a j) :
  ∀ (i j n : ℕ), (i < 100) → (j < 100) → (i + n < 100) → (j + n < 100) →
  ∃ c : ℕ, (a (i + j - 1) + a (i + j + n - 1) + a (i + n + j - 1) + a (i + n + j + n - 1)) = c^2 :=
begin
  sorry
end

end square_sum_of_corner_cells_l377_377593


namespace fibonacci_eighth_term_l377_377261

theorem fibonacci_eighth_term
  (F : ℕ → ℕ)
  (h1 : F 9 = 34)
  (h2 : F 10 = 55)
  (fib : ∀ n, F (n + 2) = F (n + 1) + F n) :
  F 8 = 21 :=
by
  sorry

end fibonacci_eighth_term_l377_377261


namespace solid_is_tetrahedron_l377_377147

-- Definitions for the problem conditions
def solid_has_three_triangle_projections (S : Type) : Prop :=
  ∀ (v1 v2 v3 : S → Triangle), (v1 S) ∧ (v2 S) ∧ (v3 S)

-- Mathematically equivalent proof problem statement in Lean 4
theorem solid_is_tetrahedron (S : Type) (h : solid_has_three_triangle_projections S) : is_tetrahedron S :=
by
sory

end solid_is_tetrahedron_l377_377147


namespace print_time_l377_377772

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end print_time_l377_377772


namespace connie_tickets_l377_377415

variable (T : ℕ)

theorem connie_tickets (h : T = T / 2 + 10 + 15) : T = 50 :=
by 
sorry

end connie_tickets_l377_377415


namespace coin_probability_difference_l377_377328

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377328


namespace min_value_of_z_l377_377283

theorem min_value_of_z : ∀ (x : ℝ), ∃ z : ℝ, z = 5 * x^2 - 20 * x + 45 ∧ z ≥ 25 :=
by sorry

end min_value_of_z_l377_377283


namespace closest_point_exists_l377_377443

def closest_point_on_line_to_point (x : ℝ) (y : ℝ) : Prop :=
  ∃(p : ℝ × ℝ), p = (3, 1) ∧ ∀(q : ℝ × ℝ), q.2 = (q.1 + 3) / 3 → dist p (3, 2) ≤ dist q (3, 2)

theorem closest_point_exists :
  closest_point_on_line_to_point 3 2 :=
sorry

end closest_point_exists_l377_377443


namespace max_PA_PB_value_l377_377222

theorem max_PA_PB_value : 
  ∀ (P A B : ℝ × ℝ),
    (P.1^2 / 9 + P.2^2 / 25 = 1) →
    (A.1^2 + (A.2 - 4)^2 = 16) →
    (B.1^2 + (B.2 + 4)^2 = 4) →
    PA_distance (P A B) (sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) ≤ 16 :=
by
  sorry

end max_PA_PB_value_l377_377222


namespace x_finishes_remaining_work_in_14_days_l377_377746

-- Define the work rates of X and Y
def work_rate_X : ℚ := 1 / 21
def work_rate_Y : ℚ := 1 / 15

-- Define the amount of work Y completed in 5 days
def work_done_by_Y_in_5_days : ℚ := 5 * work_rate_Y

-- Define the remaining work after Y left
def remaining_work : ℚ := 1 - work_done_by_Y_in_5_days

-- Define the number of days needed for X to finish the remaining work
def x_days_remaining : ℚ := remaining_work / work_rate_X

-- Statement to prove
theorem x_finishes_remaining_work_in_14_days : x_days_remaining = 14 := by
  sorry

end x_finishes_remaining_work_in_14_days_l377_377746


namespace parallel_planes_from_skew_lines_l377_377621

-- Definitions for lines and planes
variable {Line : Type} {Plane : Type}

-- Relations and conditions
variable (m n : Line)
variable (α β : Plane)

-- Skewness and parallelism conditions
def skew (l₁ l₂ : Line) : Prop := ¬ ∃ p : Point, lies_on p l₁ ∧ lies_on p l₂

def parallel (l : Line) (p : Plane) : Prop := ∀ (x : Point), l ∈ x → False
def parallel (p₁ p₂ : Plane) : Prop := ∀ (x : Line), (x ∈ x p₁ ∧ x ∈ x p₂) → x = p₁

-- Main theorem statement
theorem parallel_planes_from_skew_lines
  (h1 : skew m n)
  (h2 : parallel m α)
  (h3 : parallel n α)
  (h4 : parallel m β)
  (h5 : parallel n β) : parallel α β := by
    sorry

end parallel_planes_from_skew_lines_l377_377621


namespace find_angle_HCO_l377_377721

-- Define the setting and conditions
variables (A B C O H : Point)
variables (circle : Circle)
variables (tangentA_O_B tangentA_O_C : Tangent)
variables (triABC : Triangle)
variables (angleBAC : Real)
variables (angleHCO : Real)

-- Define the conditions using Lean definitions
axiom TangentDefinition : tangentA_O_B.circle = circle ∧ tangentA_O_C.circle = circle
axiom TangentTouching : tangentA_O_B.point = B ∧ tangentA_O_C.point = C
axiom AngleBAC_given : ∠BAC = 40
axiom OrthocenterH : Orthocenter triABC = H

-- Define the problem that needs proof
theorem find_angle_HCO
    (tangent_definition : TangentDefinition)
    (tangent_touching : TangentTouching)
    (angle_BAC : AngleBAC_given)
    (orthocenter_h : OrthocenterH) :
    ∠HCO = 50 := 
sorry

end find_angle_HCO_l377_377721


namespace positive_difference_probability_l377_377302

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377302


namespace ratio_of_distances_l377_377615

noncomputable def regular_tetrahedron (A B C D F : Point) : Prop :=
  equilateral A B C D ∧
  edge AD contains F ∧ 
  symmetric_faces A B C ∧ symmetric_faces B C D ∧ symmetric_faces C D A

theorem ratio_of_distances (A B C D F : Point) (t T : ℝ) (h : regular_tetrahedron A B C D F)
  (d₁ : t = sum_of_distances_to_faces F A B C D)
  (d₂ : T = sum_of_distances_to_edges F A B C D) :
  t / T = sqrt 6 / 2 := by
  sorry

end ratio_of_distances_l377_377615


namespace no_two_digit_number_with_properties_l377_377389

theorem no_two_digit_number_with_properties :
  ¬ (∃ (N : ℕ), (10 ≤ N ∧ N < 100) ∧ 2 ∣ N ∧ 13 ∣ N ∧ (∃ (a b : ℕ), 
    (N = 10 * a + b) ∧ (a * b) ^ (1/2) ∈ ℕ)) :=
by {
  sorry
}

end no_two_digit_number_with_properties_l377_377389


namespace oil_consumption_relation_l377_377374

noncomputable def initial_oil : ℝ := 62

noncomputable def remaining_oil (x : ℝ) : ℝ :=
  if x = 100 then 50
  else if x = 200 then 38
  else if x = 300 then 26
  else if x = 400 then 14
  else 62 - 0.12 * x

theorem oil_consumption_relation (x : ℝ) :
  remaining_oil x = 62 - 0.12 * x := by
  sorry

end oil_consumption_relation_l377_377374


namespace diff_of_squares_count_l377_377512

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377512


namespace H_five_times_two_eq_six_l377_377883

def H : ℝ → ℝ :=
  fun x => if x = 2 then -4 else if x = -4 then 6 else if x = 6 then 6 else 0 -- Placeholder for other values

theorem H_five_times_two_eq_six :
  H (H (H (H (H 2)))) = 6 :=
by
  -- Define the points on the graph
  have H_2 : H 2 = -4 := by rfl
  have H_neg4 : H (-4) = 6 := by rfl
  have H_6 : H 6 = 6 := by rfl

  -- Start from H(2)
  transitivity H (H (H (H (-4)))) -- H(2) = -4
  rw [H_2]
  transitivity H (H (H 6)) -- H(-4) = 6
  rw [H_neg4]
  transitivity H (H 6) -- H(6) = 6
  rw [H_6]
  transitivity H 6 -- H(6) = 6 again
  rw [H_6]
  exact H_6 -- Final value is 6

end H_five_times_two_eq_six_l377_377883


namespace max_perfect_squares_l377_377658

theorem max_perfect_squares (a b : ℕ) (h_d : a ≠ b) :
  2 ≤ card (filter is_square [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)]) :=
begin
  sorry
end

end max_perfect_squares_l377_377658


namespace monotonic_increasing_interval_l377_377682

noncomputable def function := λ x: ℝ, log (1/2) (-x^2 + x + 6)

theorem monotonic_increasing_interval :
  (∀ x y: ℝ, -2 < x ∧ x < 3 ∧ -2 < y ∧ y < 3 ∧ x < y → function x < function y) ↔ (∀ x, 1/2 < x ∧ x < 3) :=
sorry

end monotonic_increasing_interval_l377_377682


namespace geometric_progression_fourth_term_l377_377095

-- define the three initial terms of the geometric progression
def a₁ := 5^(1/5)
def a₂ := 5^(1/10)
def a₃ := 5^(1/20)

-- define the fourth term as what we need to prove
def a₄ := 5^(1/40)

theorem geometric_progression_fourth_term :
  ∀ a₁ a₂ a₃, (a₁ = 5^(1/5)) ∧ (a₂ = 5^(1/10)) ∧ (a₃ = 5^(1/20)) → a₄ = 5^(1/10) :=
by 
  intros a₁ a₂ a₃ h
  rw [←h.left, ←h.right.left, ←h.right.right]
  sorry

end geometric_progression_fourth_term_l377_377095


namespace not_prime_polynomial_l377_377662

theorem not_prime_polynomial (n : ℕ) (hn : n > 0) : 
  ¬ is_prime (n^2 - 2^2014 * 2014 * n + 4^2013 * (2014^2 - 1)) :=
sorry

end not_prime_polynomial_l377_377662


namespace percentage_red_candies_remaining_l377_377372

variable (R O G B Y T : ℝ)

-- Conditions
def kaz_eats_all_green_and_35_percent_orange : ℝ := 0.35 * O
def selina_eats_60_percent_red_and_1_3_yellow : ℝ := 0.60 * R + 1/3 * Y
def carlos_eats_half_blue_and_1_4_of_remaining_orange : ℝ := 0.5 * B + 0.25 * (0.65 * O)
def remaining_red : ℝ := 0.40 * R
def remaining_green : ℝ := 0
def remaining_orange : ℝ := 0.4875 * O
def remaining_blue : ℝ := 0.10 * B
def remaining_yellow : ℝ := 2/3 * Y
def total_remaining_candies : ℝ := 0.10 * T

-- Question: What percent of the red candies remains?
theorem percentage_red_candies_remaining
  (hR : remaining_red = 0.40 * R) 
  (hO : remaining_orange = 0.4875 * O)
  (hG : remaining_green = 0) 
  (hB : remaining_blue = 0.10 * B) 
  (hY : remaining_yellow = 2/3 * Y) 
  (hT : remaining_red + remaining_orange + remaining_blue + remaining_yellow = 0.10 * T) :
  (remaining_red / R) * 100 = 40 := 
by
  sorry

end percentage_red_candies_remaining_l377_377372


namespace triangle_scale_no_triangle_l377_377819

theorem triangle_scale_no_triangle (PQ PR QR : ℕ) (hPQ : PQ = 15) (hPR : PR = 20) (hQR : QR = 25) :
  ¬ (let PQ' := 3 * PQ,
         PR' := 2 * PR,
         QR' := QR in
     PQ' + PR' > QR' ∧ PQ' + QR' > PR' ∧ PR' + QR' > PQ') :=
by
  rw [hPQ, hPR, hQR]
  let PQ' := 3 * 15
  let PR' := 2 * 20
  let QR' := 25
  have h1 : PQ' = 45 := rfl
  have h2 : PR' = 40 := rfl
  have h3 : QR' = 25 := rfl
  rw [h1, h2, h3]
  norm_num
  intro h
  cases h with _ h
  cases h with _ h
  apply not_lt_of_ge
  norm_num
  assumption

end triangle_scale_no_triangle_l377_377819


namespace at_least_2_same_color_inevitable_l377_377161

-- Definitions based on the conditions
def num_balls := 6
def num_red_balls := 3
def num_yellow_balls := 3

-- This is the event we need to prove as inevitable
def at_least_2_same_color_event := 
  ∀(draws : set ℕ), draws.card = 3 → 
  ∃ (color : ℕ), draws.filter color = 2 ∨ draws.filter color = 3

-- The main theorem statement
theorem at_least_2_same_color_inevitable :
  at_least_2_same_color_event :=
  sorry

end at_least_2_same_color_inevitable_l377_377161


namespace sum_of_exterior_angles_hexagon_l377_377694

theorem sum_of_exterior_angles_hexagon : 
  (∑ (i : Fin 6), exterior_angle_of_polygon (hexagon i)) = 360 := 
sorry

end sum_of_exterior_angles_hexagon_l377_377694


namespace trajectory_C_of_point_N_slope_k_range_l377_377088

section
variables (M P Q N T A B E : Point) (k x0 : Real)

-- Given points M, P, Q, N (Condition statements in game)
-- P is on y-axis, Q is on positive x-axis
-- MP dot PN = 0
-- PN = 1/2 * NQ
-- Let's define these first as conditions in Lean

def point_on_y_axis (P : Point) : Prop := ∃ y : Real, P = (0, y)
def point_on_positive_x_axis (Q : Point) : Prop := ∃ x : Real, x > 0 ∧ Q = (x, 0)
def dot_product (v1 v2 : Vector) : Real := sorry -- Implementation of dot product
def midpoint (p1 p2 : Point) : Point := sorry -- Implementation of midpoint
def triangle (p1 p2 p3 : Point) : Prop := sorry -- Implementation to verify three points form a triangle
def right_triangle_90_at (p1 p2 p3 : Point) : Prop := sorry -- Implementation to verify right triangle at p2

-- Define the vectors: 
-- Vector MP Definition here:
def vector_MP (M P : Point) : Vector := (P.1 - M.1, P.2 - M.2)
-- and so on for PN etc.

-- Define the proof problem, wherein the derived statements incorporate these conditions and arrive at solutions as given
theorem trajectory_C_of_point_N (M P Q N : Point) (h₁ : dot_product (vector_MP M P) (vector_MP P N) = 0)
  (h₂ : vector_MP P N = (1 / 2 : Real) • vector_MP N Q ) :
  ∃ x y : Real, N = (x, y) ∧ y^2 = 4 * x :=
sorry

theorem slope_k_range (T A B E : Point) (k : Real) (hAEB : right_triangle_90_at A E B) (h_line_l : ∃ m : Real, T = (-1/2, 0) ∧ line_l = {p : Point | p.2 = m * (p.1 + 1 / 2)}):
  k ∈ set.Ico (-1) 0 ∪ set.Ioc 0 1 :=
sorry
end

end trajectory_C_of_point_N_slope_k_range_l377_377088


namespace sum_six_smallest_multiples_of_eleven_l377_377346

theorem sum_six_smallest_multiples_of_eleven : 
  (11 + 22 + 33 + 44 + 55 + 66) = 231 :=
by
  sorry

end sum_six_smallest_multiples_of_eleven_l377_377346


namespace count_of_squares_difference_l377_377520

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377520


namespace num_5_letter_words_with_at_least_one_vowel_l377_377124

-- Definitions to capture the conditions
def set_of_letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
def is_consonant (c : Char) : Prop := ¬ is_vowel c

def are_all_consonants (s : List Char) : Prop :=
  ∀ c ∈ s, is_consonant c

def is_valid_5_letter_word (s : List Char) : Prop :=
  s.length = 5 ∧ s.all (fun c => c ∈ set_of_letters)

-- Theorem statement
theorem num_5_letter_words_with_at_least_one_vowel :
  Finset.card ((Finset.filter (fun s : List Char => is_valid_5_letter_word s ∧ ∃ c ∈ s, is_vowel c) (Finset.powerset_of_card_eq (set_of_letters.val.bind (λ _, ['A', 'B', 'C', 'D', 'E', 'F'])) 5))) = 6752 :=
by sorry

end num_5_letter_words_with_at_least_one_vowel_l377_377124


namespace fraction_of_money_left_l377_377013

theorem fraction_of_money_left 
  (m c : ℝ) 
  (h1 : (1/4 : ℝ) * m = (1/2) * c) : 
  (m - c) / m = (1/2 : ℝ) :=
by
  -- the proof will be written here
  sorry

end fraction_of_money_left_l377_377013


namespace max_sum_unit_hexagons_l377_377629

theorem max_sum_unit_hexagons (k : ℕ) (hk : k ≥ 3) : 
  ∃ S, S = 6 + (3 * k - 9) * k * (k + 1) / 2 + (3 * (k^2 - 2)) * (k * (k + 1) * (2 * k + 1) / 6) / 6 ∧
       S = 3 * (k * k - 14 * k + 33 * k - 28) / 2 :=
by
  sorry

end max_sum_unit_hexagons_l377_377629


namespace no_two_digit_number_with_properties_l377_377390

theorem no_two_digit_number_with_properties :
  ¬ (∃ (N : ℕ), (10 ≤ N ∧ N < 100) ∧ 2 ∣ N ∧ 13 ∣ N ∧ (∃ (a b : ℕ), 
    (N = 10 * a + b) ∧ (a * b) ^ (1/2) ∈ ℕ)) :=
by {
  sorry
}

end no_two_digit_number_with_properties_l377_377390


namespace max_perfect_squares_among_pairwise_products_l377_377653

theorem max_perfect_squares_among_pairwise_products 
  (a b : ℕ) 
  (h_distinct : a ≠ b) 
  (h_distinct2 : a + 2 ≠ b) 
  (h_distinct3 : a ≠ b + 2) 
  (h_distinct4 : a + 2 ≠ b + 2) : 
  ∃ p1 p2 : ℕ, 
  (p1 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} ∧
   is_square p1) ∧
  (p2 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} ∧
   is_square p2) ∧
  p1 ≠ p2 ∧ 
  ∀ p3, p3 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} 
  → is_square p3 → p3 = p1 ∨ p3 = p2 := 
begin
  sorry
end

end max_perfect_squares_among_pairwise_products_l377_377653


namespace sasha_quarters_max_l377_377229

/-- Sasha has \$4.80 in U.S. coins. She has four times as many dimes as she has nickels 
and the same number of quarters as nickels. Prove that the greatest number 
of quarters she could have is 6. -/
theorem sasha_quarters_max (q n d : ℝ) (h1 : 0.25 * q + 0.05 * n + 0.1 * d = 4.80)
  (h2 : n = q) (h3 : d = 4 * n) : q = 6 := 
sorry

end sasha_quarters_max_l377_377229


namespace total_cost_with_discounts_l377_377171

theorem total_cost_with_discounts :
  let red_roses := 2 * 12
  let white_roses := 1 * 12
  let yellow_roses := 2 * 12
  let cost_red := red_roses * 6
  let cost_white := white_roses * 7
  let cost_yellow := yellow_roses * 5
  let total_cost_before_discount := cost_red + cost_white + cost_yellow
  let first_discount := 0.15 * total_cost_before_discount
  let cost_after_first_discount := total_cost_before_discount - first_discount
  let additional_discount := 0.10 * cost_after_first_discount
  let total_cost := cost_after_first_discount - additional_discount
  total_cost = 266.22 := by
  sorry

end total_cost_with_discounts_l377_377171


namespace coin_flip_probability_difference_l377_377312

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377312


namespace expand_expression_l377_377834

theorem expand_expression :
  (6 * (Polynomial.C (Complex.ofReal 1) * (Polynomial.X - 3)) * (Polynomial.X^2 + 4 * Polynomial.X + 16)).coeffs
   = (6 * Polynomial.X^3 + 6 * Polynomial.X^2 + 24 * Polynomial.X - 288).coeffs := by
  sorry

end expand_expression_l377_377834


namespace distance_from_P_to_CD_l377_377160

theorem distance_from_P_to_CD (AB CD : ℝ) (P Q R : Point) (h : ℝ) 
  (h₁ : AB > CD) (h₂ : height_from_A_and_B_to_CD = 2)
  (h₃ : line_through_P_parallel_to_bases AB CD P Q R) (h₄ : divides_trapezoid_into_equal_areas P Q R):
  distance_from_P_to_CD = 1 :=
sorry

end distance_from_P_to_CD_l377_377160


namespace probability_A_and_B_same_county_l377_377398

/-
We have four experts and three counties. We need to assign the experts to the counties such 
that each county has at least one expert. We need to prove that the probability of experts 
A and B being dispatched to the same county is 1/6.
-/

def num_experts : Nat := 4
def num_counties : Nat := 3

def total_possible_events : Nat := 36
def favorable_events : Nat := 6

theorem probability_A_and_B_same_county :
  (favorable_events : ℚ) / total_possible_events = 1 / 6 := by sorry

end probability_A_and_B_same_county_l377_377398


namespace positive_difference_prob_3_and_4_heads_l377_377292

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377292


namespace locus_equation_l377_377067

-- Definitions of the conditions
def point_F : ℝ × ℝ := (0, -3)  -- Point F(0, -3)
def line_y : ℝ → Prop := λ y, y = -5  -- Line y = -5

-- Definition of the required proof
theorem locus_equation :
  ∀ (x y : ℝ), (√(x^2 + (y + 3)^2) = |y + 5|) ↔ (y = 1/4 * x^2 - 4) :=
by
  sorry

end locus_equation_l377_377067


namespace pelican_speed_l377_377802

theorem pelican_speed
  (eagle_speed falcon_speed hummingbird_speed total_distance time : ℕ)
  (eagle_distance falcon_distance hummingbird_distance : ℕ)
  (H1 : eagle_speed = 15)
  (H2 : falcon_speed = 46)
  (H3 : hummingbird_speed = 30)
  (H4 : time = 2)
  (H5 : total_distance = 248)
  (H6 : eagle_distance = eagle_speed * time)
  (H7 : falcon_distance = falcon_speed * time)
  (H8 : hummingbird_distance = hummingbird_speed * time)
  (total_other_birds_distance : ℕ)
  (H9 : total_other_birds_distance = eagle_distance + falcon_distance + hummingbird_distance)
  (pelican_distance : ℕ)
  (H10 : pelican_distance = total_distance - total_other_birds_distance)
  (pelican_speed : ℕ)
  (H11 : pelican_speed = pelican_distance / time) :
  pelican_speed = 33 := 
  sorry

end pelican_speed_l377_377802


namespace coin_flip_probability_difference_l377_377336

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377336


namespace equation_of_line_l377_377048

noncomputable def P : ℝ × ℝ := (-2, 3)

theorem equation_of_line 
  (a b c : ℝ) 
  (h₁ : b ≠ 0 ∨ c ≠ 0)
  (h₂ : (a * fst P + b * snd P + c = 0))
  (h₃ : (a ≠ 0 → b ≠ 0 → a = b → c = a)) :
  (a * x + b * y + c = 0) → (a * x + b * y + c = 0 = (x + y - 1 = 0) ∨ (3 * x + 2 * y = 0)) :=
by
  sorry

end equation_of_line_l377_377048


namespace coin_flip_probability_difference_l377_377313

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377313


namespace triangulation_three_colorable_l377_377789

structure Triangulation (P : Type) :=
  (triangles : Set (Set P))
  (is_triangle : ∀ t ∈ triangles, ∃ a b c : P, t = {a, b, c})
  (adjacent : ∀ ta tb ∈ triangles, ta ≠ tb → (ta ∩ tb).card ≤ 2)

def three_colorable (P : Type) (T : Triangulation P) : Prop :=
  ∃ C : T.triangles → Fin 3,
  ∀ ta tb ∈ T.triangles, ta ≠ tb → (ta ∩ tb).card = 2 → C ta ≠ C tb

theorem triangulation_three_colorable (P : Type) (T : Triangulation P) :
  three_colorable P T := by
  sorry

end triangulation_three_colorable_l377_377789


namespace largest_k_inequality_l377_377468

theorem largest_k_inequality {a b c : ℝ} (h1 : a ≤ b) (h2 : b ≤ c) (h3 : ab + bc + ca = 0) (h4 : abc = 1) :
  |a + b| ≥ 4 * |c| :=
sorry

end largest_k_inequality_l377_377468


namespace integral_sin3_cos_l377_377041

open Real

theorem integral_sin3_cos :
  ∫ z in (π / 4)..(π / 2), sin z ^ 3 * cos z = 3 / 16 := by
  sorry

end integral_sin3_cos_l377_377041


namespace matilda_father_chocolates_l377_377210

theorem matilda_father_chocolates 
  (total_chocolates : ℕ) 
  (total_people : ℕ) 
  (give_up_fraction : ℚ) 
  (mother_chocolates : ℕ) 
  (father_eats : ℕ) 
  (father_left : ℕ) :
  total_chocolates = 20 →
  total_people = 5 →
  give_up_fraction = 1 / 2 →
  mother_chocolates = 3 →
  father_eats = 2 →
  father_left = 5 →
  let chocolates_per_person := total_chocolates / total_people,
      father_chocolates := (chocolates_per_person * total_people * give_up_fraction).nat_abs - mother_chocolates - father_eats
  in father_chocolates = father_left := by
  intros h1 h2 h3 h4 h5 h6
  have h_chocolates_per_person : total_chocolates / total_people = 4 := by sorry
  have h_chocolates_given_up : (chocolates_per_person * total_people * give_up_fraction).nat_abs = 10 := by sorry
  have h_father_chocolates : 10 - mother_chocolates - father_eats = 5 := by sorry
  exact h_father_chocolates

end matilda_father_chocolates_l377_377210


namespace truck_speed_kmph_l377_377790

theorem truck_speed_kmph (distance_m : ℝ) (time_s : ℝ) (km_m : ℝ) (sec_hr : ℝ) (speed_kmph : ℝ) :
    distance_m = 600 →
    time_s = 40 →
    km_m = 1000 →
    sec_hr = 3600 →
    speed_kmph = (distance_m / time_s) * (sec_hr / km_m) →
    speed_kmph = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end truck_speed_kmph_l377_377790


namespace PH_bisects_EF_l377_377602

open Classical

variables {A B C D E F M P H I : Point} -- Defining the Points

-- Conditions
variables (incircle_tangent_BC : tangent_circle D B C I)
variables (incircle_tangent_CA : tangent_circle E C A I)
variables (incircle_tangent_AB : tangent_circle F A B I)
variables (foot_M : foot_perpendicular D E F M)
variables (P_on_DM : on_line_segment D M P)
variables (DP_eq_MP : distance_eq D P M P)
variables (H_orthocenter_BIC : orthocenter B I C H)

-- Theorem to prove PH bisects EF
theorem PH_bisects_EF 
  (incircle_tangent_BC : tangent_circle D B C I)
  (incircle_tangent_CA : tangent_circle E C A I)
  (incircle_tangent_AB : tangent_circle F A B I)
  (foot_M : foot_perpendicular D E F M)
  (P_on_DM : on_line_segment D M P)
  (DP_eq_MP : distance_eq D P M P)
  (H_orthocenter_BIC : orthocenter B I C H) :
  bisects P H E F :=
sorry

end PH_bisects_EF_l377_377602


namespace Ravi_jumps_39_inches_l377_377650

-- Define the heights of the next three highest jumpers
def h₁ : ℝ := 23
def h₂ : ℝ := 27
def h₃ : ℝ := 28

-- Define the average height of the three jumpers
def average_height : ℝ := (h₁ + h₂ + h₃) / 3

-- Define Ravi's jump height
def Ravi_jump_height : ℝ := 1.5 * average_height

-- The theorem to prove
theorem Ravi_jumps_39_inches : Ravi_jump_height = 39 := by
  sorry
 
end Ravi_jumps_39_inches_l377_377650


namespace find_dividend_l377_377280

theorem find_dividend
  (divisor : ℕ) (quotient : ℕ) (remainder : ℕ)
  (h_divisor : divisor = 17)
  (h_quotient : quotient = 8)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 141 :=
by
  rw [h_divisor, h_quotient, h_remainder]
  norm_num
  sorry

end find_dividend_l377_377280


namespace total_students_is_45_l377_377584

-- Define the initial conditions with the definitions provided
def drunk_drivers : Nat := 6
def speeders : Nat := 7 * drunk_drivers - 3
def total_students : Nat := drunk_drivers + speeders

-- The theorem to prove that the total number of students is 45
theorem total_students_is_45 : total_students = 45 :=
by
  sorry

end total_students_is_45_l377_377584


namespace operation_4_3_is_5_l377_377743

def custom_operation (m n : ℕ) : ℕ := n ^ 2 - m

theorem operation_4_3_is_5 : custom_operation 4 3 = 5 :=
by
  -- Proof goes here
  sorry

end operation_4_3_is_5_l377_377743


namespace area_triangle_AOB_const_circle_equation_l377_377459

open Real

noncomputable def circle (t : ℝ) (h : t ≠ 0) : set (ℝ × ℝ) :=
  {p | (p.1 - t) ^ 2 + (p.2 - 1) ^ 2 = 5}

theorem area_triangle_AOB_const (t : ℝ) (h : t ≠ 0) :
  let A := (2 * t, 0)
  let B := (0, -1)
  let O := (0, 0)
  abs (A.1 * B.2 - A.2 * B.1) / 2 = 4 := by
  sorry

theorem circle_equation (M N : ℝ × ℝ) (t : ℝ) (h : t ≠ 0)
  (H : M ∈ circle t h ∧ N ∈ circle t h ∧ (2 * M.1 + M.2 - 4 = 0) ∧ (2 * N.1 + N.2 - 4 = 0) ∧
  ((0,0) - M).norm = ((0,0) - N).norm) :
  let C := circle 2 h
  C = {(x, y) | (x - 2) ^ 2 + (y - 1) ^ 2 = 5} := by
  sorry

end area_triangle_AOB_const_circle_equation_l377_377459


namespace max_volume_height_l377_377075

-- Define the conditions
def on_same_sphere (r h : ℝ) : Prop := (1:ℝ)^2 = (h - 1)^2 + r^2

-- Define the volume function for the cone
def cone_volume (h : ℝ) : ℝ := (π / 3) * h * (2 * h - h^2)

-- The main theorem
theorem max_volume_height (h : ℝ) (r : ℝ) (h_nonneg : 0 ≤ h) (h_le_2 : h ≤ 2) (r_nonneg : 0 ≤ r) 
  (sphere_condition : on_same_sphere r h) :
  h = 4 / 3 :=
by
  sorry

end max_volume_height_l377_377075


namespace different_genre_pairs_count_l377_377565

theorem different_genre_pairs_count 
  (mystery_books : Finset ℕ)
  (fantasy_books : Finset ℕ)
  (biographies : Finset ℕ)
  (h1 : mystery_books.card = 4)
  (h2 : fantasy_books.card = 4)
  (h3 : biographies.card = 4) :
  (mystery_books.product (fantasy_books ∪ biographies)).card +
  (fantasy_books.product (mystery_books ∪ biographies)).card +
  (biographies.product (mystery_books ∪ fantasy_books)).card = 48 := 
sorry

end different_genre_pairs_count_l377_377565


namespace minimal_fraction_difference_l377_377981

theorem minimal_fraction_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧ (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < (2 : ℚ) / 3 ∧
  (∀ r s : ℕ, (3 : ℚ) / 5 < (r : ℚ) / s ∧ (r : ℚ) / s < (2 : ℚ) / 3 → 0 < s → q ≤ s) ∧
  q - p = 3 :=
begin
  sorry
end

end minimal_fraction_difference_l377_377981


namespace definite_integral_evaluation_l377_377432

theorem definite_integral_evaluation :
  ∫ x in -2..2, (sqrt (4 - x^2) + |x|) = 2 * Real.pi + 4 :=
by
  sorry

end definite_integral_evaluation_l377_377432


namespace count_non_empty_subsets_P_l377_377182

def pos_nat : Set ℕ := {n | n > 0}

def P : Set (ℕ × ℕ) := {p | p.1 ∈ pos_nat ∧ p.2 ∈ pos_nat ∧ p.1 + p.2 < 4}

theorem count_non_empty_subsets_P : 
  ∃ n : ℕ, n = 7 ∧ n = 2^((P.to_finset.card : ℕ)) - 1 :=
by 
  sorry

end count_non_empty_subsets_P_l377_377182


namespace range_of_a_l377_377879

noncomputable def f (x : ℝ) : ℝ := -exp x - x
noncomputable def g (a x : ℝ) : ℝ := 3 * a * x + 2 * cos x
noncomputable def f'_deriv (x : ℝ) : ℝ := -exp x - 1
noncomputable def g'_deriv (a x : ℝ) : ℝ := 3 * a - 2 * sin x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f'_deriv x * g'_deriv a x = -1) → (- 1 / 3 ≤ a ∧ a ≤ 2 / 3) :=
sorry

end range_of_a_l377_377879


namespace count_of_squares_difference_l377_377523

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377523


namespace print_time_l377_377775

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end print_time_l377_377775


namespace O_center_of_symmetry_l377_377220

variable (P : Type) [Polygon P]
variable (O : Point)

-- Basic assumptions for convex polygon P
-- We'll assume some definitions for a convex polygon and point containment/inclusivity in Lean context
-- Convex Polygon P and point O inside P
axiom O_in_P : O ∈ P

-- Each line through O divides P into two parts of equal area
axiom equal_area_division : ∀ (L : Line), passes_through O L → divides_into_equal_area P L

-- Theorem: O is the center of symmetry of the polygon P
theorem O_center_of_symmetry : is_center_of_symmetry O P :=
sorry

end O_center_of_symmetry_l377_377220


namespace complex_division_square_real_l377_377871

theorem complex_division_square_real
  (z1 z2 : ℂ) (hz1 : z1 ≠ 0) (hz2 : z2 ≠ 0) 
  (h : abs (z1 + z2) = abs (z1 - z2)) : 
  ∃ r : ℝ, (z1 / z2)^2 = r := 
sorry

end complex_division_square_real_l377_377871


namespace option_a_correct_option_d_correct_l377_377608

noncomputable def b : ℕ → ℕ
| 1 := 1
| 2 := 3
| (n+1) + 1 := b (n+1) + b (n)

theorem option_a_correct : (∑ i in finset.range (100/2 + 1), b (2 * i)) = b 101 - 1 :=
sorry

theorem option_d_correct : (∑ i in finset.range (100), b (i + 1) ^ 2) = b 100 * b 101 - 2 :=
sorry

end option_a_correct_option_d_correct_l377_377608


namespace mike_needs_more_money_l377_377636

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end mike_needs_more_money_l377_377636


namespace twice_volume_exists_l377_377856

def cylinder_volume (r h : ℕ) := π * r^2 * h

theorem twice_volume_exists :
  let r := 6
  let h := 12
  let V := cylinder_volume r h
  let r' := 6
  let h' := 24
  let V' := cylinder_volume r' h'
  2 * V = V' := by
  sorry

end twice_volume_exists_l377_377856


namespace num_counting_numbers_l377_377900

theorem num_counting_numbers (n : ℕ) (h1 : n ∣ (53 - 3)) (h2 : n > 3) (h3 : 4 ∣ n) : (finset.filter (λ x, x > 3 ∧ 4 ∣ x) (finset.divisors 50)).card = 0 := 
sorry

end num_counting_numbers_l377_377900


namespace matilda_father_chocolates_left_l377_377207

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l377_377207


namespace sector_area_is_correct_l377_377242

-- Define the conditions
def radius : ℝ := 3
def central_angle_degrees : ℝ := 120
def central_angle_radians : ℝ := (central_angle_degrees * (Real.pi / 180))

-- Define the question in terms of verifying the area of the sector
def area_of_sector (r : ℝ) (theta : ℝ) : ℝ := 1 / 2 * theta * r

-- The theorem that needs to be proven
theorem sector_area_is_correct :
  area_of_sector radius central_angle_radians = 3 * Real.pi := 
sorry

end sector_area_is_correct_l377_377242


namespace sum_first_four_terms_l377_377941

noncomputable def a : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := (a (n + 1))^2 - 1

theorem sum_first_four_terms : a 1 + a 2 + a 3 + a 4 = 0 :=
by
  sorry

end sum_first_four_terms_l377_377941


namespace standard_eq_of_ellipse_max_area_triangle_l377_377084

-- Definitions based on conditions
variable {e : ℝ} (he1 : e = (4 * 3 / (8 * sqrt 3))) -- eccentricity based on given e = 4 / (8 sqrt 3 / 3)
variable (a b c : ℝ) (ha1 : a = 2) (hb1 : b = sqrt(4 - c^2)) (hc1 : c = e * a)
variable {P Q : ℝ × ℝ} {A B M : ℝ × ℝ} (hA : A = (-2, 0)) (hB : B = (2, 0))
variable (k₁ k₂ : ℝ) (hk : k₁ = 7 * k₂) 

-- The given form of the ellipse and distance conditions
def ellipse_eq (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def distance_AB (A B : ℝ × ℝ) := dist A B = 4

-- Part 1: Standard equation of the ellipse
theorem standard_eq_of_ellipse (hC : ellipse_eq 2 1 1 0) : ellipse_eq 2 1 x y :=
sorry

-- Part 2: Maximum area of the triangle APM
theorem max_area_triangle (hM : M = (-3 / 2, 0)) :
  ∃ t₁ y₁ : ℝ, y₁ = 2 * 1 / (1 + 4) ∧ t₁ = 2 → (1 / 2) * (y₁) = 1 / 4 :=
sorry


end standard_eq_of_ellipse_max_area_triangle_l377_377084


namespace complex_modulus_problem_l377_377912

noncomputable def complex_conjugate (z : ℂ) : ℂ := conj z

theorem complex_modulus_problem (z : ℂ) (h : (1 - z) / (1 + z) = complex.i) :
  | complex_conjugate z - 2 | = Real.sqrt 5 :=
sorry

end complex_modulus_problem_l377_377912


namespace probability_S_9_given_S_odd_l377_377708

variable (S : ℤ)
variable (sum_is_odd : S % 2 = 1)
variable (prob_S_is_9 : ℚ)

-- We define the possible outcomes for two dice rolls
def possible_outcomes : Set (ℤ × ℤ) :=
  { (d1, d2) | d1 ∈ {1, 2, 3, 4, 5, 6} ∧ d2 ∈ {1, 2, 3, 4, 5, 6} }

-- We define the specific outcomes where the sum S is odd
def odd_sums (d1 d2 : ℤ) : Prop :=
  (d1 + d2) % 2 = 1

-- We define the specific outcomes where the sum S is 9
def sum_is_9 (d1 d2 : ℤ) : Prop :=
  d1 + d2 = 9

-- The events where the sum S is 9, given that S is odd
def events_S_odd : Set (ℤ × ℤ) := 
  { outcome ∈ possible_outcomes | odd_sums outcome.1 outcome.2 }

def events_S_9 : Set (ℤ × ℤ) := 
  { outcome ∈ possible_outcomes | sum_is_9 outcome.1 outcome.2 }

-- The probability of an event happening given the fair dice assumption
noncomputable def prob (event : Set (ℤ × ℤ)) : ℚ := 
  (event.to_finset.card : ℚ) / (possible_outcomes.to_finset.card : ℚ)

-- The probability S = 9 given S is odd
noncomputable def prob_S_9_given_odd : ℚ := 
  prob events_S_9 / prob events_S_odd

-- Theorem statement
theorem probability_S_9_given_S_odd : 
  sum_is_odd →
  prob_S_9_given_odd = 2 / 9 :=
sorry

end probability_S_9_given_S_odd_l377_377708


namespace max_students_for_distribution_l377_377745

theorem max_students_for_distribution : 
  ∃ (n : Nat), (∀ k, k ∣ 1048 ∧ k ∣ 828 → k ≤ n) ∧ 
               (n ∣ 1048 ∧ n ∣ 828) ∧ 
               n = 4 :=
by
  sorry

end max_students_for_distribution_l377_377745


namespace range_magnitude_l377_377070

def a : ℝ × ℝ := (3, -4)
def b (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def range_of_magnitude (α : ℝ) : ℝ :=
  magnitude (a.1 + 2 * (b α).1, a.2 + 2 * (b α).2)

theorem range_magnitude (α : ℝ) : 3 ≤ range_of_magnitude α ∧ range_of_magnitude α ≤ 7 :=
  sorry

end range_magnitude_l377_377070


namespace triangle_inequalities_l377_377921

variables {α : Type*} [linear_ordered_field α] {a b c s R : α} {A B C : α}

/-- Given a triangle ABC with sides a, b, and c opposite angles A, B, and C respectively,
    and the semi-perimeter s and circumradius R, we have the following inequalities. -/
theorem triangle_inequalities
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_A : A = arccos ((b^2 + c^2 - a^2) / (2 * b * c)))
  (h_B : B = arccos ((a^2 + c^2 - b^2) / (2 * a * c)))
  (h_C : C = arccos ((a^2 + b^2 - c^2) / (2 * a * b)))
  (h_s : s = (a + b + c) / 2)
  (h_R : R = (a * b * c) / (4 * sqrt (s * (s - a) * (s - b) * (s - c)))) :
  (a * cos A + b * cos B + c * cos C ≤ s) ∧
  (sin (2 * A) + sin (2 * B) + sin (2 * C) ≤ s / R) := by sorry

end triangle_inequalities_l377_377921


namespace distinct_numerators_count_l377_377991

/-- Helper to check if a number is a digit from 0 to 9 --/
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

/-- Helper to check if a number is a valid three-digit number --/
def is_valid_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

/-- Helper to check if two numbers are coprime --/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set S of rational numbers between 0 and 1 that can be expressed as repeating decimals --/
def S : set ℚ := { r : ℚ | ∃ (abc : ℕ), 0 < r ∧ r < 1 ∧ abc > 0 
  ∧ is_valid_three_digit abc 
  ∧ r = abc / 999 
  ∧ coprime abc 999 }

noncomputable def count_distinct_numerators : ℕ := 
  set.to_finset (set.image (λ r : ℚ, (r.num : ℕ)) S).card

theorem distinct_numerators_count : count_distinct_numerators = 660 := 
sorry

end distinct_numerators_count_l377_377991


namespace triangle_perimeter_l377_377442

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem triangle_perimeter :
  let A := (1, 2) : ℝ × ℝ
      B := (1, 7) : ℝ × ℝ
      C := (4, 5) : ℝ × ℝ in
  distance A B + distance B C + distance C A = 5 + real.sqrt 13 + 3 * real.sqrt 2 :=
by
  sorry

end triangle_perimeter_l377_377442


namespace sum_of_consecutive_2017_numbers_is_power_of_2017th_l377_377812

theorem sum_of_consecutive_2017_numbers_is_power_of_2017th :
  ∃ n : ℕ, (sum (finset.range 2017).map (λ i, n + i)) = 2017^2017 :=
sorry

end sum_of_consecutive_2017_numbers_is_power_of_2017th_l377_377812


namespace lines_parallel_l377_377718

-- Definitions based on conditions
variable (line1 line2 : ℝ → ℝ → Prop) -- Assuming lines as relations for simplicity
variable (plane : ℝ → ℝ → ℝ → Prop) -- Assuming plane as a relation for simplicity

-- Condition: Both lines are perpendicular to the same plane
def perpendicular_to_plane (line : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ (x y z : ℝ), plane x y z → line x y

axiom line1_perpendicular : perpendicular_to_plane line1 plane
axiom line2_perpendicular : perpendicular_to_plane line2 plane

-- Theorem: Both lines are parallel
theorem lines_parallel : ∀ (line1 line2 : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop),
  (perpendicular_to_plane line1 plane) →
  (perpendicular_to_plane line2 plane) →
  (∀ x y : ℝ, line1 x y → line2 x y) := sorry

end lines_parallel_l377_377718


namespace cyclic_quadrilateral_tangential_quadrilateral_l377_377641

open EuclideanGeometry

variables (A B C A1 B1 C1 M : Point)

-- Given conditions: Points C1 on AB, A1 on BC, B1 on CA, and lines AA1, BB1, CC1 intersect at M
variables (h1 : collinear A B C1)
variables (h2 : collinear B C A1)
variables (h3 : collinear C A B1)
variables (h4 : intersecting_at AA1 BB1 M)
variables (h5 : intersecting_at BB1 CC1 M)
variables (h6 : intersecting_at CC1 AA1 M)

-- Propositions to prove:
-- If MA1CB1 and MB1AC1 are cyclic, then MA1BC1 is cyclic
theorem cyclic_quadrilateral 
  (h_cyclic1 : is_cyclic_quadrilateral M A1 C B1)
  (h_cyclic2 : is_cyclic_quadrilateral M B1 A C1) :
  is_cyclic_quadrilateral M A1 B C1 :=
sorry

-- If MA1CB1 and MB1AC1 are tangential, then MA1BC1 is tangential
theorem tangential_quadrilateral
  (h_tangential1 : is_tangential_quadrilateral M A1 C B1)
  (h_tangential2 : is_tangential_quadrilateral M B1 A C1) :
  is_tangential_quadrilateral M A1 B C1 :=
sorry

end cyclic_quadrilateral_tangential_quadrilateral_l377_377641


namespace ratio_side_length_to_brush_width_l377_377386

theorem ratio_side_length_to_brush_width (s w : ℝ) (h1 : w = s / 4) (h2 : s^2 / 3 = w^2 + ((s - w)^2) / 2) :
    s / w = 4 := by
  sorry

end ratio_side_length_to_brush_width_l377_377386


namespace brenda_skittles_l377_377012

theorem brenda_skittles (initial additional : ℕ) (h1 : initial = 7) (h2 : additional = 8) :
  initial + additional = 15 :=
by {
  -- Proof would go here
  sorry
}

end brenda_skittles_l377_377012


namespace proof_triangle_circumcircle_diameter_eq_proof_triangle_area_eq_l377_377170

noncomputable def triangle_circumcircle_diameter (a b c : ℝ) (A B C : ℝ) : ℝ :=
2 * R
  where
    R := (b / (Real.sin B))

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * a * c * (Real.sin B)

theorem proof_triangle_circumcircle_diameter_eq :
  ∀ (a b c A B C : ℝ),
    (a = 2 * Real.sqrt 1.5 / 1.5) →
    (b = 2) →
    (a - b) * (Real.sin A + Real.sin B) = c * (Real.sin A - Real.sin C) →
    triangle_circumcircle_diameter a b c A B C = 4 * Real.sqrt(3) / 3 :=
begin
  intros,
  sorry
end

theorem proof_triangle_area_eq :
  ∀ (a b c A B C : ℝ),
    (a = 2 * Real.sqrt 1.5 / 1.5) →
    (b = 2) →
    (a - b) * (Real.sin A + Real.sin B) = c * (Real.sin A - Real.sin C) →
    triangle_area a b c A B C = Real.sqrt 3 / 3 + 1 :=
begin
  intros,
  sorry
end

end proof_triangle_circumcircle_diameter_eq_proof_triangle_area_eq_l377_377170


namespace coin_probability_difference_l377_377334

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377334


namespace parallel_lines_l377_377502

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end parallel_lines_l377_377502


namespace solution_set_of_inequality_l377_377259

theorem solution_set_of_inequality :
  {x : ℝ | (|x| + x) * (sin x - 2) < 0} = {x | 0 < x} :=
by
  sorry

end solution_set_of_inequality_l377_377259


namespace tan_2alpha_val_l377_377488

noncomputable def f (x : Real) : Real := sqrt 3 * sin (2 * x) + 2 * cos x ^ 2

theorem tan_2alpha_val (α : Real) (h_α : α ∈ Icc (π / 12) (5 * π / 12)) (h_f_α : f α = 13 / 5) : 
  ∃ (k : ℤ), 
  (∀ x, x ∈ Icc (-π / 6 + k * π) (π / 3 + k * π) → 
    has_deriv_at f x 0 → 0 < deriv f x) ∧ 
  tan (2 * α) = (48 + 25 * sqrt 3) / 11 := 
sorry

end tan_2alpha_val_l377_377488


namespace part1_part2_l377_377109

open Real

def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |3 * x + a|

theorem part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by
  sorry

theorem part2 (h : ∃ x_0 : ℝ, f x_0 (a := a) + 2 * |x_0 - 2| < 3) : -9 < a ∧ a < -3 := by
  sorry

end part1_part2_l377_377109


namespace no_extreme_values_poly_extreme_values_fractional_l377_377035

-- 1. Prove that the function y = 8x^3 - 12x^2 + 6x + 1 does not have any extreme values.
theorem no_extreme_values_poly :
  ¬∃ x : ℝ, has_strict_local_min_on (λ x, 8 * x^3 - 12 * x^2 + 6 * x + 1) (set.univ : set ℝ) x 
  ∧ ¬∃ x : ℝ, has_strict_local_max_on (λ x, 8 * x^3 - 12 * x^2 + 6 * x + 1) (set.univ : set ℝ) x := sorry

-- 2. Prove that the function y = 1 - (x-2)^(2/3) has a maximum value at x = 2 and y = 1, and does not have a minimum value.
theorem extreme_values_fractional :
  (∃ x : ℝ, x = 2 ∧ (λ x, 1 - (x - 2)^(2/3)) x = 1 ∧ 
  (has_strict_local_max_on (λ x, 1 - (x - 2)^(2/3)) (set.univ : set ℝ) x) ∧ 
  ¬∃ x : ℝ, has_strict_local_min_on (λ x, 1 - (x - 2)^(2/3)) (set.univ : set ℝ) x) := sorry

end no_extreme_values_poly_extreme_values_fractional_l377_377035


namespace correct_propositions_count_l377_377403

theorem correct_propositions_count :
  let p1 : Prop := ∀ (P Q R: Plane), (P ∥ R ∧ Q ∥ R) → (P ∥ Q)
  let p2 : Prop := ∀ (P Q: Plane), P ∥ Q → Q ∥ P  -- This captures the transitivity condition
  let p3 : Prop := ∀ (l m n: Line), (l ⊥ n ∧ m ⊥ n) → (l ∥ m)
  let p4 : Prop := ∀ (l m: Line) (Q: Plane), (l ⊥ Q ∧ m ⊥ Q) → (l ∥ m)
  (¬p1 ∧ p2 ∧ ¬p3 ∧ p4) → (number_of_true [p1, p2, p3, p4] = 2) := 
by
  -- Definitions of Plane, Line and parallel/perpendicular relations are implicitly imported
  -- number_of_true should be defined as a function that counts the number of true Booleans in a list
  sorry

end correct_propositions_count_l377_377403


namespace geometric_sequence_fraction_l377_377461

open Classical

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_fraction {a : ℕ → ℝ} {q : ℝ}
  (h₀ : ∀ n, 0 < a n)
  (h₁ : geometric_seq a q)
  (h₂ : 2 * (1 / 2 * a 2) = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_fraction_l377_377461


namespace sum_of_digits_l377_377454

noncomputable def digits_divisibility (C F : ℕ) : Prop :=
  (C >= 0 ∧ C <= 9 ∧ F >= 0 ∧ F <= 9) ∧
  (C + 8 + 5 + 4 + F + 7 + 2) % 9 = 0 ∧
  (100 * 4 + 10 * F + 72) % 8 = 0

theorem sum_of_digits (C F : ℕ) (h : digits_divisibility C F) : C + F = 10 :=
sorry

end sum_of_digits_l377_377454


namespace binomial_expansion_constant_term_l377_377876

noncomputable def constant_term_of_binomial_expansion (n : ℕ) (h : n > 0) : ℚ :=
  let sum_binom := (1 + n + (n * (n - 1)) / 2) in
  if sum_binom = 56 then (1 / 2 ^ 8) * ((Nat.choose 10 8) : ℚ) else 0

theorem binomial_expansion_constant_term :
  ∀ (n : ℕ) (h : n > 0), ((1 + n + (n * (n - 1)) / 2) = 56) → constant_term_of_binomial_expansion n h = 45 / 256 :=
by sorry

end binomial_expansion_constant_term_l377_377876


namespace katie_games_diff_l377_377175

-- Definitions based on conditions in a)
def Katie_new_games : Nat := 57
def Katie_old_games : Nat := 39
def First_friend_new_games : Nat := 34
def First_friend_old_games : Nat := 28
def Second_friend_new_games : Nat := 25
def Second_friend_old_games : Nat := 32
def Third_friend_new_games : Nat := 12
def Third_friend_old_games : Nat := 21

-- Calculate total games for Katie and each friend
def Katie_total_games : Nat := Katie_new_games + Katie_old_games
def First_friend_total_games : Nat := First_friend_new_games + First_friend_old_games
def Second_friend_total_games : Nat := Second_friend_new_games + Second_friend_old_games
def Third_friend_total_games : Nat := Third_friend_new_games + Third_friend_old_games

-- Total games for all friends
def Total_friends_games : Nat := First_friend_total_games + Second_friend_total_games + Third_friend_total_games

-- The proof statement
theorem katie_games_diff :
  (Katie_total_games - Total_friends_games).abs = 56 :=
by
  sorry

end katie_games_diff_l377_377175


namespace midpoint_chord_equality_l377_377085

open Real

variable (a : ℝ)
variable (λ : ℝ)
variable (k₁ k₂ : ℝ)

noncomputable def conic_section := {x y : ℝ // A * x^2 + B * x * y + y^2 + D * x - a^2 + λ * (y - k₁ * x) * (y - k₂ * x) = 0}

noncomputable def O := (0, 0)

theorem midpoint_chord_equality (P Q A B C D : (ℝ × ℝ))
    (hP : P = (0, a))
    (hQ : Q = (0, -a))
    (hR : P ∈ conic_section ∧ Q ∈ conic_section)
    (hAB : ∀ k₁ k₂, k₁, k₂ ≠ 0 → line_passes_through_origin k₁ k₂)
    (hCD : ∀ k₁ k₂, k₁, k₂ ≠ 0 → line_passes_through_origin k₁ k₂) 
    (intersect_R : line_intersects_conic F 0 y₁ O)
    (intersect_S : line_intersects_conic F 0 y₂ O)
    (y₁ : ℝ) (y₂ : ℝ)
    :
    ( |O - R| = |O - S| ) := sorry

    -- Definitions and auxiliary theorems
    def line_passes_through_origin (k₁ k₂ : ℝ) : Prop :=
    ∀ (x y : ℝ), y = k₁ * x ∨ y = k₂ * x

    def line_intersects_conic (F : ℝ × ℝ → ℝ) (x y : ℝ) (O : ℝ × ℝ) : Prop :=
    ∀ (x y : ℝ), F (x, y) = 0 ∧ y ≠ 0

end midpoint_chord_equality_l377_377085


namespace sum_of_digits_is_28_l377_377168

theorem sum_of_digits_is_28 (D O G C A T : ℕ) 
  (h_diff : ∀ x y : ℕ, (x ≠ y) → ¬(x ∈ {D, O, G, C, A, T} ∧ y ∈ {D, O, G, C, A, T})) 
  (h_range : ∀ x : ℕ, x ∈ {D, O, G, C, A, T} → 0 ≤ x ∧ x ≤ 9) 
  (h_eqn : D * 100 + O * 10 + G + C * 100 + A * 10 + T = 1000) : 
  D + O + G + C + A + T = 28 := 
by
  sorry

end sum_of_digits_is_28_l377_377168


namespace point_D_cartesian_coordinates_l377_377596

variable (α : ℝ) (t : ℝ) (φ : ℝ)

def line_parametric_eq_x (t α : ℝ) : ℝ := t * Real.cos α
def line_parametric_eq_y (t α : ℝ) : ℝ := -2 + t * Real.sin α

def semicircle_parametric_eq_x (φ : ℝ) : ℝ := Real.cos φ
def semicircle_parametric_eq_y (φ : ℝ) : ℝ := 1 + Real.sin φ

def general_line_eq (x α : ℝ) : ℝ := x * Real.tan α - 2

def semicircle_eq (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1 ∧ y > 1

theorem point_D_cartesian_coordinates :
  ∃ α, α ∈ (0 : ℝ, Real.pi / 2) ∧
      general_line_eq 0 (Real.pi / 4) = 0 ∧
      line_parametric_eq_x t α = 0 ∧
      semicircle_parametric_eq_x (2 * α) = 0 ∧
      semicircle_parametric_eq_y (2 * α) = 2 ∧
      1 / 2 * (2 / Real.sin α) * (3 * Real.cos α + Real.sin α) = 4 :=
sorry

end point_D_cartesian_coordinates_l377_377596


namespace vectors_perpendicular_l377_377472

noncomputable def angle_between_vectors (u v : ℝ^n) : ℝ :=
real.arccos ((u ⬝ v) / (∥u∥ * ∥v∥))

theorem vectors_perpendicular {u v : ℝ^n} (hu : u ≠ 0) (hv : v ≠ 0)
  (h : ∥u + 2 • v∥ = ∥u - 2 • v∥) :
  angle_between_vectors u v = real.pi / 2 :=
by
  -- Proof needs to be filled in (sorry means the proof is omitted)
  sorry

end vectors_perpendicular_l377_377472


namespace smallest_five_digit_divisible_by_12_l377_377345

-- Define what it means to be a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000

-- Define the divisibility by 12
def divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

-- Define the digit counting function
def count_digits (n : ℕ) : (ℕ → ℕ → ℕ) :=
  let digits : List ℕ := (n.toString.data.map (λ char, char.toNat - '0'.toNat)) in
  λ evens odds, (evens, odds) + List.foldl (λ (acc : ℕ × ℕ) digit, if digit % 2 = 0 then (acc.1 + 1, acc.2) else (acc.1, acc.2 + 1)) (0, 0) digits

-- Define the condition for having three odd digits and two even digits
def has_three_odd_two_even (n : ℕ) : Prop :=
  count_digits n 0 0 = (2, 3)

theorem smallest_five_digit_divisible_by_12 : ∃ n : ℕ, is_five_digit n ∧ divisible_by_12 n ∧ has_three_odd_two_even n ∧ n = 10542 :=
by
  existsi 10542
  split
  -- 10542 is a five-digit number
  · unfold is_five_digit
    norm_num
  split
  -- 10542 is divisible by 12
  · unfold divisible_by_12
    norm_num
  split
  -- 10542 has three odd digits and two even digits
  · unfold has_three_odd_two_even 
    unfold count_digits
    simp
    norm_num
  -- n = 10542
  rfl

end smallest_five_digit_divisible_by_12_l377_377345


namespace positive_difference_prob_3_and_4_heads_l377_377290

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377290


namespace not_all_prime_l377_377192

theorem not_all_prime (a b c : ℕ) (h1 : a > b > c ∧ c ≥ 3) 
  (h2 : a ∣ (b * c + b + c)) 
  (h3 : b ∣ (c * a + c + a)) 
  (h4 : c ∣ (a * b + a + b)) 
  : ¬ (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) := 
begin
  sorry,
end

end not_all_prime_l377_377192


namespace sum_of_a_and_b_is_two_l377_377250

variable (a b : ℝ)
variable (h_a_nonzero : a ≠ 0)
variable (h_fn_passes_through_point : (a * 1^2 + b * 1 - 1) = 1)

theorem sum_of_a_and_b_is_two : a + b = 2 := 
by
  sorry

end sum_of_a_and_b_is_two_l377_377250


namespace shaded_area_correct_l377_377765

noncomputable def side_length : ℝ := 24
noncomputable def radius : ℝ := side_length / 4
noncomputable def area_of_square : ℝ := side_length ^ 2
noncomputable def area_of_one_circle : ℝ := Real.pi * radius ^ 2
noncomputable def total_area_of_circles : ℝ := 5 * area_of_one_circle
noncomputable def shaded_area : ℝ := area_of_square - total_area_of_circles

theorem shaded_area_correct :
  shaded_area = 576 - 180 * Real.pi := by
  sorry

end shaded_area_correct_l377_377765


namespace infinite_solutions_a_l377_377066

theorem infinite_solutions_a (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 :=
by
  sorry

end infinite_solutions_a_l377_377066


namespace probability_black_ball_l377_377582

open ProbabilityTheory

/-- In a bag filled with red, white, and black balls of the same size, prove that the
probability of drawing a black ball is 0.3,
given that the probability of drawing a red ball is 0.42
and the probability of drawing a white ball is 0.28. -/
theorem probability_black_ball :
  let P : String → ℝ := λ s =>
    if s = "R" then 0.42 else
    if s = "W" then 0.28 else
    if s = "B" then 1 - (0.42 + 0.28) else 0
  in P "B" = 0.3 :=
by
  sorry

end probability_black_ball_l377_377582


namespace pyramid_volume_l377_377042

theorem pyramid_volume (a : ℝ) (h : a = 2)
  (b : ℝ) (hb : b = 18) :
  ∃ V, V = 2 * Real.sqrt 2 :=
by
  sorry

end pyramid_volume_l377_377042


namespace distribute_balls_l377_377562

theorem distribute_balls : 
  let B := 3 -- Number of boxes
  let N := 8 -- Total number of balls
  let min_balls_in_box1 := 2 -- Minimum number of balls in Box 1
  ∃ (count : ℕ), count = 7 ∧
  (∃ (a b c : ℕ), a + b + c = N ∧ a ≥ min_balls_in_box1) :=
begin
  sorry
end

end distribute_balls_l377_377562


namespace molecular_weight_calculation_l377_377284

theorem molecular_weight_calculation :
  let atomic_weight_K := 39.10
  let atomic_weight_Br := 79.90
  let atomic_weight_O := 16.00
  let num_K := 1
  let num_Br := 1
  let num_O := 3
  let molecular_weight := (num_K * atomic_weight_K) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)
  molecular_weight = 167.00 :=
by
  sorry

end molecular_weight_calculation_l377_377284


namespace count_diff_of_squares_l377_377557

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377557


namespace union_is_necessary_for_complement_union_is_not_sufficient_for_complement_l377_377122

variable {U : Type} (A B : Set U)

theorem union_is_necessary_for_complement (h₁ : A ∪ B = Set.univ) :
  (B = Aᶜ) → h₁ :=
by
  sorry

theorem union_is_not_sufficient_for_complement (h₁ : A ∪ B = Set.univ) :
  ¬(B = Aᶜ) :=
by
  sorry

end union_is_necessary_for_complement_union_is_not_sufficient_for_complement_l377_377122


namespace cauchy_schwarz_2d_cauchy_schwarz_2d_eq_condition_max_value_function_l377_377700

-- Prove Cauchy-Schwarz inequality in 2D case
theorem cauchy_schwarz_2d (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 :=
by sorry

-- Equality condition for Cauchy-Schwarz inequality in 2D case
theorem cauchy_schwarz_2d_eq_condition (a b c d : ℝ) : 
  ((a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2) ↔ (a * d = b * c) :=
by sorry

-- Show the maximum value of the function y = 3 √(x-1) + √(20-4x)
theorem max_value_function : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → (3 * real.sqrt(x-1) + real.sqrt(20 - 4 * x)) ≤ 2 * real.sqrt 13) :=
by sorry

end cauchy_schwarz_2d_cauchy_schwarz_2d_eq_condition_max_value_function_l377_377700


namespace range_of_a_l377_377091

variable {a : ℝ} (x₁ x₂ : ℝ)
variable (x : ℝ → ℝ)

noncomputable def f (x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x^2

noncomputable def local_min (x₁ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Icc (x₁ - ε) (x₁ + ε), f x₁ ≤ f x

noncomputable def local_max (x₂ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Icc (x₂ - ε) (x₂ + ε), f x₂ ≥ f x

theorem range_of_a (h1 : 0 < a ∧ a ≠ 1)
  (h2 : local_min x₁) (h3 : local_max x₂) (h4: x₁ < x₂) : 
  1 / exp 1 < a ∧ a < 1 := 
sorry

end range_of_a_l377_377091


namespace max_sum_of_arithmetic_seq_l377_377862

theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h₁ : a 1 = 11) (h₂ : a 5 = -1) 
  (h₃ : ∀ n, a n = 14 - 3 * (n - 1)) 
  : ∀ n, (S n = (n * (a 1 + a n) / 2)) → max (S n) = 26 :=
sorry

end max_sum_of_arithmetic_seq_l377_377862


namespace square_area_of_equal_perimeters_l377_377704

theorem square_area_of_equal_perimeters (side_a side_b : ℝ) (h : side_a = 6) (k : side_b = 8) :
  let hypotenuse := real.sqrt (side_a ^ 2 + side_b ^ 2)
  let perimeter_triangle := side_a + side_b + hypotenuse
  let side_square := perimeter_triangle / 4
  let area_square := side_square ^ 2
  perimeter_triangle = 24 → area_square = 36 :=
by
  intros
  calc
    hypotenuse = real.sqrt (side_a ^ 2 + side_b ^ 2) : by rfl
    side_square = perimeter_triangle / 4 : by rfl
    area_square = side_square ^ 2 : by rfl
  sorry

end square_area_of_equal_perimeters_l377_377704


namespace count_four_digit_even_numbers_l377_377126

theorem count_four_digit_even_numbers : 
  let digits := {0, 1, 2, 3, 4, 5}
  let valid_numbers := { n ∈ finset.range 10000 |
    let d1 := n / 1000,
        d2 := (n / 100) % 10,
        d3 := (n / 10) % 10,
        d4 := n % 10 in
    d1 ∈ digits ∧
    d2 ∈ digits ∧
    d3 ∈ digits ∧
    d4 ∈ digits ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧
    d3 ≠ d4 ∧
    d4 % 2 = 0 ∧
    d1 ≠ 0 }
  in valid_numbers.card = 156 := 
by sorry

end count_four_digit_even_numbers_l377_377126


namespace maximum_gold_coins_l377_377740

theorem maximum_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n ≤ 146 :=
by
  sorry

end maximum_gold_coins_l377_377740


namespace feed_mixture_hay_calculation_l377_377236

theorem feed_mixture_hay_calculation
  (hay_Stepan_percent oats_Pavel_percent corn_mixture_percent : ℝ)
  (hay_Stepan_mass_Stepan hay_Pavel_mass_Pavel total_mixture_mass : ℝ):
  hay_Stepan_percent = 0.4 ∧
  oats_Pavel_percent = 0.26 ∧
  (∃ (x : ℝ), 
  x > 0 ∧ 
  hay_Pavel_percent =  0.74 - x ∧ 
  0.15 * x + 0.25 * x = 0.3 * total_mixture_mass ∧
  hay_Stepan_mass_Stepan = 0.40 * 150 ∧
  hay_Pavel_mass_Pavel = (0.74 - x) * 250 ∧ 
  total_mixture_mass = 150 + 250) → 
  hay_Stepan_mass_Stepan + hay_Pavel_mass_Pavel = 170 := 
by
  intro h
  obtain ⟨h1, h2, ⟨x, hx1, hx2, hx3, hx4, hx5, hx6⟩⟩ := h
  /- proof -/
  sorry

end feed_mixture_hay_calculation_l377_377236


namespace extreme_points_range_a_inequality_extreme_points_l377_377889

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + (a / 2) * x^2 - x

-- Theorem 1: If f(x) has two extreme points in (0, +∞), then a ∈ (-1/e, 0)
theorem extreme_points_range_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ f'.apply_deriv x₁ = 0 ∧ f'.apply_deriv x₂ = 0) →
  a ∈ (-(1 / Real.exp 1), 0) :=
sorry

-- Theorem 2: Given the conditions, prove a + 2/(x₁ + x₂) < 0
theorem inequality_extreme_points (a x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < Real.exp 1) (h₃ : Real.exp 1 < x₂) 
  (h₄ : f'.apply_deriv x₁ = 0) (h₅ : f'.apply_deriv x₂ = 0) :
  a + 2 / (x₁ + x₂) < 0 :=
sorry

end extreme_points_range_a_inequality_extreme_points_l377_377889


namespace diff_of_squares_count_l377_377517

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377517


namespace number_of_valid_pairs_l377_377053

-- Definitions based on conditions
def isValidPair (x y : ℕ) : Prop := 
  (1 ≤ x ∧ x ≤ 1000) ∧ (1 ≤ y ∧ y ≤ 1000) ∧ (x^2 + y^2) % 5 = 0

def countValidPairs : ℕ := 
  (Finset.range 1000).filter (λ x, (x + 1) % 5 = 0 ∨ (x + 1) % 5 = 1 ∨ (x + 1) % 5 = 4).card *
  (Finset.range 1000).filter (λ y, (y + 1) % 5 = 0 ∨ (y + 1) % 5 = 1 ∨ (y + 1) % 5 = 4).card +
  2 * (
    (Finset.range 1000).filter (λ x, (x + 1) % 5 = 1).card *
    (Finset.range 1000).filter (λ y, (y + 1) % 5 = 4).card *
    2
  )

theorem number_of_valid_pairs : countValidPairs = 200000 := by
  sorry

end number_of_valid_pairs_l377_377053


namespace dinner_cakes_today_6_l377_377777

-- Definitions based on conditions
def lunch_cakes_today : ℕ := 5
def dinner_cakes_today (x : ℕ) : ℕ := x
def yesterday_cakes : ℕ := 3
def total_cakes_served : ℕ := 14

-- Lean statement to prove the mathematical equivalence
theorem dinner_cakes_today_6 (x : ℕ) (h : lunch_cakes_today + dinner_cakes_today x + yesterday_cakes = total_cakes_served) : x = 6 :=
by {
  sorry -- Proof to be completed.
}

end dinner_cakes_today_6_l377_377777


namespace integral_cos_div_l377_377193

theorem integral_cos_div (a b : ℝ) (h : a > b) (h0 : b > 0) : 
  ∫ θ in 0..2*Real.pi, 1 / (a + b * Real.cos θ) = 2 * Real.pi / Real.sqrt (a^2 - b^2) :=
sorry

end integral_cos_div_l377_377193


namespace integer_diff_of_squares_1_to_2000_l377_377539

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377539


namespace value_of_9_2_minus_9_2_star_l377_377847

noncomputable def greatest_even_le_y (y : ℝ) : ℝ :=
  if (y ≥ 2) then 2 * Int.floor (y / 2) else 0

theorem value_of_9_2_minus_9_2_star : 
  let y : ℝ := 9.2 in
  y - greatest_even_le_y y = 1.2 :=
by
  sorry

end value_of_9_2_minus_9_2_star_l377_377847


namespace third_purchase_cost_l377_377785

def listed_price : ℝ := 25000
def first_discount : ℝ := 0.15
def second_discount : ℝ := 0.20
def third_discount : ℝ := 0.10

theorem third_purchase_cost :
  let first_cost := listed_price * (1 - first_discount) in
  let second_cost := first_cost * (1 - second_discount) in
  let third_cost := second_cost * (1 - third_discount) in
  third_cost = 15300 := by
  sorry

end third_purchase_cost_l377_377785


namespace value_of_T_l377_377960

theorem value_of_T :
  let T := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - 2 * Real.sqrt 3)) + (2 / (2 * Real.sqrt 3 - Real.sqrt 12))
            - (1 / (Real.sqrt 12 - 3)) + (1 / (3 - Real.sqrt 8))
  in T = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by
  let T := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - 2 * Real.sqrt 3))
            + (2 / (2 * Real.sqrt 3 - Real.sqrt 12))
            - (1 / (Real.sqrt 12 - 3)) + (1 / (3 - Real.sqrt 8))
  have T_value : T = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 2 := sorry
  exact T_value

end value_of_T_l377_377960


namespace initial_birds_proof_l377_377264

-- Define the conditions given in the problem
variables (initial_birds : ℕ) (landed_birds : ℕ) (total_birds : ℕ)

-- Express the conditions in terms of Lean 4 definitions
def conditions := landed_birds = 8 ∧ total_birds = 20

-- Formulate the statement to prove the initial number of birds
theorem initial_birds_proof (h : conditions initial_birds) : initial_birds = 12 :=
sorry

end initial_birds_proof_l377_377264


namespace yellow_jelly_bean_ratio_l377_377068

theorem yellow_jelly_bean_ratio:
  let A := 24
  let B := 32
  let C := 35
  let D := 40
  let yellowA := 0.40 * A
  let yellowB := 0.30 * B
  let yellowC := 0.25 * C
  let yellowD := 0.15 * D
  let totalYellow := yellowA + yellowB + yellowC + yellowD
  let totalBeans := A + B + C + D
  (totalYellow / totalBeans) ≈ 0.26 := by
  sorry

end yellow_jelly_bean_ratio_l377_377068


namespace interior_angle_of_dodecagon_angle_MDE_angle_DME_angle_CBM_points_B_M_F_collinear_l377_377685

variable (A B C D E F G H I J K L M : Type) 
variable [RegularPolygon A B C D E F G H I J K L (12 : ℕ)] -- Assume regular 12-sided polygon

-- Define intersection condition for point M
variable (hMk : IntersectionPoint M A E D K) -- Point M is intersection of AE and DK

-- Statement for part (a)
theorem interior_angle_of_dodecagon : measure_internal_angle A B C D E F G H I J K L = 150 := by
  sorry

-- Statements for part (b)
theorem angle_MDE : angle M D E = 90 := by
  sorry

theorem angle_DME : angle D M E = 45 := by
  sorry

-- Statement for part (c)
theorem angle_CBM : angle C B M = 45 := by
  sorry

-- Statement for part (d)
theorem points_B_M_F_collinear : collinear B M F := by
  sorry

end interior_angle_of_dodecagon_angle_MDE_angle_DME_angle_CBM_points_B_M_F_collinear_l377_377685


namespace product_three_consecutive_not_power_l377_377363

theorem product_three_consecutive_not_power (n k m : ℕ) (hn : n > 0) (hm : m ≥ 2) : 
  (n-1) * n * (n+1) ≠ k^m :=
by sorry

end product_three_consecutive_not_power_l377_377363


namespace right_regular_quadrilateral_prism_condition_l377_377349

-- Definitions for geometric shapes
structure Square (s : Type) :=
(sides : s)
(is_square : True)

structure Rectangle (r : Type) :=
(sides : r)
(is_rectangle : True)

structure Prism (b : Type) :=
(base : b)
(sides : list b)
(is_prism : True)

def is_right_regular_quadrilateral_prism {b : Type} [Square b] (p : Prism b) : Prop :=
∀ side : b, rectangle side ∧ (∀ s1 s2 : b, congruent s1 s2)

-- Main theorem
theorem right_regular_quadrilateral_prism_condition (b : Type) [Square b] [Rectangle b] 
(p : Prism b) (h : ∀ side : b, rectangle side ∧ (∀ s1 s2 : b, congruent s1 s2)) : 
is_right_regular_quadrilateral_prism p :=
by sorry


end right_regular_quadrilateral_prism_condition_l377_377349


namespace sin_cos_identity_l377_377146

variable {α λ : ℝ}

theorem sin_cos_identity (h1 : ∃ λ : ℝ, λ ≠ 0 ∧ (3 * λ, 4 * λ) = (-3 * λ, 4 * λ)) :
  (sin α + cos α) / (sin α - cos α) = 1 / 7 :=
by
  sorry

end sin_cos_identity_l377_377146


namespace count_numbers_with_odd_divisors_under_100_l377_377128

-- Definition of the required conditions: less than 100 and having an odd number of divisors
def is_less_than_100 (n : ℕ) : Prop := n < 100
def has_odd_number_of_divisors (n : ℕ) : Prop := (finset.filter (λ (d : ℕ), n % d = 0) (finset.range (n + 1))).card % 2 = 1

-- The theorem to prove the required statement
theorem count_numbers_with_odd_divisors_under_100 : 
  finset.card (finset.filter (λ n, is_less_than_100 n ∧ has_odd_number_of_divisors n) (finset.range 100)) = 9 :=
by
  -- Placeholder for the actual proof
  sorry

end count_numbers_with_odd_divisors_under_100_l377_377128


namespace triangle_AB_hypo_length_l377_377223

open Real

-- Define A and B on the curve y = x^2
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-2, 4)

-- Define function for distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the proof that the length of hypotenuse AB is 4
theorem triangle_AB_hypo_length : dist A B = 4 :=
  sorry

end triangle_AB_hypo_length_l377_377223


namespace sum_W_Y_eq_seven_l377_377036

theorem sum_W_Y_eq_seven :
  ∃ (W X Y Z : ℕ), W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  W ∈ {1, 2, 3, 4} ∧ X ∈ {1, 2, 3, 4} ∧ Y ∈ {1, 2, 3, 4} ∧ Z ∈ {1, 2, 3, 4} ∧
  (W / X : ℚ) - (Y / Z : ℚ) = 1 ∧ W + Y = 7 :=
by
  sorry

end sum_W_Y_eq_seven_l377_377036


namespace potato_sales_l377_377798

theorem potato_sales :
  let total_weight := 6500
  let damaged_weight := 150
  let bag_weight := 50
  let price_per_bag := 72
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_weight
  let total_revenue := num_bags * price_per_bag
  total_revenue = 9144 :=
by
  sorry

end potato_sales_l377_377798


namespace correct_inequality_l377_377096

-- Define the function types and the condition function
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f (x : ℝ) : ℝ := g (x - 2)

-- Given conditions as hypotheses
variables (a: ℝ) (h1 : g (x) = g (-x)) -- g is even function
variables (h2: 1 < a) (h3: a < 3)
variables (h4 : (x ≠ 2) → (x - 2) * (derivative f x) > 0)

-- The main statement we need to prove
theorem correct_inequality :
  f 3 < f (real.log a / real.log 3) ∧ f (real.log a / real.log 3) < f (4^a) := sorry

end correct_inequality_l377_377096


namespace donna_additional_flyers_l377_377632

theorem donna_additional_flyers (m d a : ℕ) (h1 : m = 33) (h2 : d = 2 * m + a) (h3 : d = 71) : a = 5 :=
by
  have m_val : m = 33 := h1
  rw [m_val] at h2
  linarith [h3, h2]

end donna_additional_flyers_l377_377632


namespace part1_part2_final_answer_l377_377457

-- Define the conditions
variables (a x y : ℝ)
def x_def : ℝ := 1 - 2 * a
def y_def : ℝ := 3 * a - 4

-- Define the problems
theorem part1 (h : x = 1 - 2 * a) (hx : real.sqrt x = 3) : a = -4 :=
by sorry

theorem part2 (h1 : x = 1 - 2 * a) (h2 : y = 3 * a - 4) : x = y ∨ x^2 = y^2 :=
by sorry

theorem final_answer (h1 : x = 1 - 2 * a) (h2 : y = 3 * a - 4) (h : x = y ∨ x^2 = y^2) : x^2 = 1 ∨ x^2 = 25 :=
by sorry

end part1_part2_final_answer_l377_377457


namespace sector_area_is_8pi_over_3_l377_377241

noncomputable def sector_area {r θ1 θ2 : ℝ} 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (hr : r = 4) : ℝ := 
    1 / 2 * (θ2 - θ1) * r ^ 2

theorem sector_area_is_8pi_over_3 (θ1 θ2 : ℝ) 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (r : ℝ) (hr : r = 4) : 
  sector_area hθ1 hθ2 hr = 8 * π / 3 :=
by
  sorry

end sector_area_is_8pi_over_3_l377_377241


namespace TriangleABC_CyclicQuadrilateral_l377_377605

theorem TriangleABC_CyclicQuadrilateral :
  (∀ (A B C : Point) (D E F G : Point),
    is_midpoint D B C →
    is_midpoint E A C →
    is_midpoint F A B →
    is_centroid G A B C →
    ∀ α : Angle, α ≤ 60 →
    is_cyclic_quadrilateral A E G F →
    ∃! (Δ : Triangle), Δ = Triangle.mk A B C) := by
  sorry

end TriangleABC_CyclicQuadrilateral_l377_377605


namespace arithmetic_sequence_sum_l377_377184

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (n a_1 d : α) : α :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem arithmetic_sequence_sum (a_1 d : α) :
  S 5 a_1 d = 5 → S 9 a_1 d = 27 → S 7 a_1 d = 14 :=
by
  sorry

end arithmetic_sequence_sum_l377_377184


namespace bouquet_combinations_l377_377377

theorem bouquet_combinations : 
  (∃ (sols : finset (ℕ × ℕ × ℕ)), sols.card = 21 ∧ 
  ∀ (r c t : ℕ), (r, c, t) ∈ sols ↔ 3 * r + 2 * c + 4 * t = 60) :=
sorry

end bouquet_combinations_l377_377377


namespace max_area_of_sector_l377_377094

theorem max_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 12) : 
  (1 / 2) * l * r ≤ 9 :=
by sorry

end max_area_of_sector_l377_377094


namespace cone_height_equals_fifteen_l377_377391

-- Definition of the cone's volume
def cone_volume (r h : ℝ) := (1 / 3) * real.pi * r^2 * h

-- Definition of the cylinder's volume
def cylinder_volume (r h : ℝ) := real.pi * r^2 * h

-- The theorem to prove
theorem cone_height_equals_fifteen :
  ∀ (r₁ r₂ h_cylinder h_cone : ℝ),
    r₁ = 3 → r₂ = 3 → h_cylinder = 5 →
    cone_volume r₁ h_cone = cylinder_volume r₂ h_cylinder →
    h_cone = 15 :=
by
  intros r₁ r₂ h_cylinder h_cone hr₁ hr₂ hh_cylinder heq
  sorry

end cone_height_equals_fifteen_l377_377391


namespace florida_north_dakota_license_plate_difference_l377_377631

theorem florida_north_dakota_license_plate_difference :
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  florida_license_plates = north_dakota_license_plates :=
by
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  show florida_license_plates = north_dakota_license_plates
  sorry

end florida_north_dakota_license_plate_difference_l377_377631


namespace similar_triangles_and_parallel_l377_377149

variable {A B C D E F G I M N O : Type*}
variable {triangle_ABC : Triangle A B C}
variable {incenter_I : Incenter I triangle_ABC}
variable {incircle_ω : Incircle I triangle_ABC}
variable {circumcircle_O : Circumcircle O triangle_ABC}
variable (AB_lt_AC : ∃ (p : ℝ), p = AB / AC ∧ p < 1) -- Given condition AB < AC
variable (D E : Point)
variable (D_on_BC : OnCircle D incircle_ω)
variable (E_on_CA : OnCircle E incircle_ω)
variable (N : Point)
variable (N_on_circumcircle_O : Intersection (Line AI) circumcircle_O N)
variable (G : Point)
variable (G_on_ND_circumcircle_O : Intersection (Line ND) circumcircle_O G)
variable (M : Point)
variable (M_on_NO_circumcircle_O : Intersection (Line NO) circumcircle_O M)
variable (F : Point)
variable (F_on_GE_circumcircle_O : Intersection (Line GE) circumcircle_O F)

theorem similar_triangles_and_parallel (h₁ : SimilarTriangle N I G N D I) (h₂ : ParallelLine MF AC) :
  SimilarTriangle N I G N D I ∧ ParallelLine MF AC :=
by {
  sorry
}

end similar_triangles_and_parallel_l377_377149


namespace fly_distance_from_ceiling_l377_377722

theorem fly_distance_from_ceiling (x y z : ℝ) (P : ℝ × ℝ × ℝ) 
  (hP : P = (0, 0, 0))
  (h1 : x = 3)
  (h2 : y = 4)
  (h3 : sqrt (x^2 + y^2 + z^2) = 5) : 
  z = 0 :=
by 
  subst_vars
  sorry

end fly_distance_from_ceiling_l377_377722


namespace sin_rule_l377_377151

theorem sin_rule (A B : ℝ) (a b : ℝ) 
  (hB : B = π / 4)
  (hb : b = 5)
  (hA : sin A = 1 / 3) :
  a = (5 * real.sqrt 2) / 3 :=
sorry

end sin_rule_l377_377151


namespace O₁_eq_O₂_l377_377273

variable (M A B C D O₁ O₂ : Point)
variable (θ₁ θ₂ : ℝ)

-- The condition that two triangles MAB and MCD are similar but have opposite orientations
axiom similar_opposite_orientation (h₁ : similar_triangle M A B M C D) : orientation_opposite M A B M C D

-- Define the centers of rotation
axiom O₁_center_rotation (h₂ : rotation_center O₁ (angle2θ (A B) (B M)) A C)
axiom O₂_center_rotation (h₃ : rotation_center O₂ (angle2θ (A B) (A M)) B D)

-- Statement to prove
theorem O₁_eq_O₂ (h₁ : similar_triangle M A B M C D) (h₂ : rotation_center O₁ (angle2θ (A B) (B M)) A C) (h₃ : rotation_center O₂ (angle2θ (A B) (A M)) B D) : O₁ = O₂ :=
sorry

end O₁_eq_O₂_l377_377273


namespace find_theta_l377_377277

theorem find_theta
  (ACB : ℝ)
  (FEG : ℝ)
  (H_ACB : ACB = 10)
  (H_FEG : FEG = 26) : 
  θ = 11 := by
  -- Given angles
  let ACB_complement := 90 - ACB
  let FEG_complement := 90 - FEG

  -- Derive other angles using the properties of a rectangle and supplementary angles
  let DCE := 180 - ACB_complement - 14
  let DEC := 180 - FEG_complement - 33

  -- Angle sum property of a triangle
  let θ := 180 - DCE - DEC

  -- Simplify to find the result
  have : θ = 11 := sorry
  exact this

end find_theta_l377_377277


namespace curve_can_be_expressed_l377_377763

theorem curve_can_be_expressed (
  a b c : ℝ := sorry,
  x y : ℝ := sorry,
  h1 : ∀ t : ℝ, x = 3 * cos t + 2 * sin t,
  h2 : ∀ t : ℝ, y = 5 * sin t,
  h : ∃ a b c : ℝ, (a = 1 / 9) ∧ (b = -4 / 45) ∧ (c = 1 / 25)
) : 
  ∀ t : ℝ, a * x^2 + b * x * y + c * y^2 = 1 :=
sorry

end curve_can_be_expressed_l377_377763


namespace diff_of_squares_count_l377_377511

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377511


namespace value_of_b_minus_a_l377_377123

theorem value_of_b_minus_a (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) : b - a = 2 ∨ b - a = 4 :=
sorry

end value_of_b_minus_a_l377_377123


namespace cost_of_each_item_l377_377216

theorem cost_of_each_item (initial_order items : ℕ) (price per_item_reduction additional_orders : ℕ) (reduced_order total_order reduced_price profit_per_item : ℕ) 
  (h1 : initial_order = 60)
  (h2 : price = 100)
  (h3 : per_item_reduction = 1)
  (h4 : additional_orders = 3)
  (h5 : reduced_price = price - price * 4 / 100)
  (h6 : total_order = initial_order + additional_orders * (price * 4 / 100))
  (h7 : reduced_order = total_order)
  (h8 : profit_per_item = price - per_item_reduction )
  (h9 : profit_per_item = 24)
  (h10 : items * profit_per_item = reduced_order * (profit_per_item - per_item_reduction)) :
  (price - profit_per_item = 76) :=
by sorry

end cost_of_each_item_l377_377216


namespace perfect_square_condition_l377_377232

-- Definitions from conditions
def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

-- Theorem statement
theorem perfect_square_condition (n : ℤ) (h1 : 0 < n) (h2 : is_integer (2 + 2 * Real.sqrt (1 + 12 * (n: ℝ)^2))) : 
  is_perfect_square n :=
by
  sorry

end perfect_square_condition_l377_377232


namespace first_peak_point_coordinates_l377_377933

variables (x_P y_P x_Q y_Q : ℝ)

def transformation (x_P y_P : ℝ) := 
  let x_Q := y_P + x_P
  let y_Q := y_P - x_P
  (x_Q, y_Q)

def distance_from_origin (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2)

def distance_ratio (x_P y_P x_Q y_Q : ℝ) : ℝ :=
  (distance_from_origin x_Q y_Q) / (distance_from_origin x_P y_P)

def angle_between_vectors (x_P y_P x_Q y_Q : ℝ) : ℝ :=
  real.arccos (
    (x_P * x_Q + y_P * y_Q) / 
    (distance_from_origin x_P y_P * distance_from_origin x_Q y_Q)
  )

noncomputable def first_peak_point (m θ : ℝ) : ℝ × ℝ :=
  let x := θ
  let y := m * real.sin(θ + θ)
  (x, y)

theorem first_peak_point_coordinates :
  let x_Q := y_P + x_P
  let y_Q := y_P - x_P
  x_Q = y_P + x_P →
  y_Q = y_P - x_P →
  distance_ratio x_P y_P x_Q y_Q = real.sqrt 2 →
  angle_between_vectors x_P y_P x_Q y_Q = real.pi / 4 →
  first_peak_point (real.sqrt 2) (real.pi / 4) = (real.pi / 4, real.sqrt 2) :=
by
  intros
  sorry

end first_peak_point_coordinates_l377_377933


namespace imaginary_part_of_conjugate_l377_377854

theorem imaginary_part_of_conjugate (z : ℂ) (hz : z = (1 + 2 * complex.i) / complex.i) :
  complex.im (complex.conj z) = 1 :=
sorry

end imaginary_part_of_conjugate_l377_377854


namespace parabola_equation_l377_377081

-- Definitions from the problem's conditions.
variables {a b p c : ℝ} (h_gt_0_a : a > 0) (h_gt_0_b : b > 0)
variable (e : ℝ)
variable (d : ℝ)

-- Conditions
def hyperbola := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1
def eccentricity (e : ℝ := 3) := e = (Real.sqrt (a^2 + b^2)) / a
def parabola_focus_distance (d : ℝ := 2/3) := 
  ∀ (focus : ℝ), focus = (p / 2) ∧ d = (a * p / (2 * c))

-- The theorem to be proved 
theorem parabola_equation (h : hyperbola) (he : eccentricity e) (hd : parabola_focus_distance d) :
  ∀ x y : ℝ, x^2 = 2 * p * y → x^2 = 8 * y :=
sorry

end parabola_equation_l377_377081


namespace length_of_AP_in_unit_rectangle_l377_377587

theorem length_of_AP_in_unit_rectangle :
  (let r : ℝ := 1,
       A := (-1 : ℝ, 1 : ℝ),
       M := (0 : ℝ, -1: ℝ),
       P := (4 / 5 : ℝ, -3 / 5 : ℝ) in
   real.sqrt (((A.1 - P.1) ^ 2 + (A.2 + P.2) ^ 2)) = real.sqrt 145 / 5) := sorry

end length_of_AP_in_unit_rectangle_l377_377587


namespace polynomial_factorization_l377_377577

theorem polynomial_factorization (m n : ℤ) (h₁ : (x^2 + m * x + 6 : ℤ) = (x - 2) * (x + n)) : m = -5 := by
  sorry

end polynomial_factorization_l377_377577


namespace ducks_in_marsh_l377_377706

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) : total_birds - geese = 37 := by
  sorry

end ducks_in_marsh_l377_377706


namespace area_ratio_DFE_ABEF_l377_377087

noncomputable def parallelogram := 
(0, 0) ∧ (2, 3) ∧ (5, 3) ∧ (3, 0)

theorem area_ratio_DFE_ABEF : 
  let A := (0 : ℝ, 0 : ℝ),
      B := (2 : ℝ, 3 : ℝ),
      C := (5 : ℝ, 3 : ℝ),
      D := (3 : ℝ, 0 : ℝ),
      E := ((0 + 5) / 2, (0 + 3) / 2),
      F := (1, 0) in
  (2 * |3 * (0 - 2.5) + 1 * (2.5 - 0) + 1.5 * (0 - 0)| / 2) /
  ((|2 * (0 - 2.5) + 1.5 * (2.5 - 3)| / 2 + |1 * (0 - 2.5)| / 2) +  |2 * (3 - 0) + 1.5 * (0 - -0)| / 2) = 1.5 := 
by sorry

end area_ratio_DFE_ABEF_l377_377087


namespace angle_aod_is_144_l377_377939

theorem angle_aod_is_144 
  (OA_perp_OC : ∀ (A O C : Type) [inner_product_space ℝ A] [inner_product_space ℝ O] [inner_product_space ℝ C], inner (O - A) (O - C) = 0)
  (OB_perp_OD : ∀ (B O D : Type) [inner_product_space ℝ B] [inner_product_space ℝ O] [inner_product_space ℝ D], inner (O - B) (O - D) = 0)
  (angle_aod_eq_4_times_boc : ∀ (A O D B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ O] [inner_product_space ℝ D] [inner_product_space ℝ B] [inner_product_space ℝ C], 
    ∃ (measure_angle : ℝ), measure_angle ≡ 4 * angle BOC): 
  measure_angle = 144 :=
by
  sorry

end angle_aod_is_144_l377_377939


namespace positive_difference_of_probabilities_l377_377306

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377306


namespace angle_equality_l377_377628

theorem angle_equality
  (A B C M P N Q E F: Point)
  (h1: collinear A C M)
  (h2: collinear B M M)
  (h3: collinear C P N)
  (h4: collinear A B N)
  (h5: collinear M N Q)
  (h6: collinear A P Q)
  (h7: collinear Q E P)
  (h8: collinear Q F A)
  (h9: collinear E D P)
  (h10: collinear F D A): 
  ∠ E D A = ∠ F D A := sorry

end angle_equality_l377_377628


namespace faster_speed_14_l377_377768

theorem faster_speed_14 
    (d₁ : ℕ) -- actual distance traveled
    (d₂ : ℕ) -- additional distance at faster speed
    (s₁ : ℕ) -- initial speed
    (s₂ : ℕ) -- faster speed
    (h₁ : d₁ = 50)
    (h₂ : s₁ = 10)
    (h₃ : d₂ = 20) : 
    s₂ = 14 :=
by
  have t := d₁ / s₁     -- Calculate time taken to travel distance d₁ at speed s₁
  have d := d₁ + d₂     -- Total distance covered when walking at the faster speed s₂
  have s := d / t       -- Calculate the faster speed s₂ as total distance divided by time
  have eq1 : t = 5 := by simp [h₁, h₂]
  have eq2 : d = 70 := by simp [h₁, h₃]
  exact by simp [eq1, eq2]

end faster_speed_14_l377_377768


namespace coin_flip_probability_l377_377669

noncomputable def probability_successful_outcomes : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 3
  successful_outcomes / total_outcomes

theorem coin_flip_probability :
  probability_successful_outcomes = 3 / 32 :=
by
  sorry

end coin_flip_probability_l377_377669


namespace hyperbola_eccentricity_l377_377023

variables {a b c : ℝ} (F1 A B F2 : ℝ)
variables (m : ℝ)
variables (hyp_eq1 : (a > 0) ∧ (b > 0))
variables (hyp_eq2 : F1 = 2 * a)
variables (tri_cond : (A = rightangled) ∧ (isosceles F1 A B))

theorem hyperbola_eccentricity (h1 : 4*a = sqrt 2 * m) :
  let e := c / a in e^2 = 5 - 2 * sqrt 2 := 
by
  sorry

end hyperbola_eccentricity_l377_377023


namespace find_k_l377_377473

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the condition for vectors to be parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Translate the problem condition
def problem_condition (k : ℝ) : Prop :=
  let lhs := (k * a.1 + b.1, k * a.2 + b.2)
  let rhs := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  is_parallel lhs rhs

-- The goal is to find k such that the condition holds
theorem find_k : problem_condition (-1/3) :=
by
  sorry

end find_k_l377_377473


namespace percentage_students_passed_is_35_l377_377930

/-
The problem is to prove the percentage of students who passed the examination, given that 520 out of 800 students failed, is 35%.
-/

def total_students : ℕ := 800
def failed_students : ℕ := 520
def passed_students : ℕ := total_students - failed_students

def percentage_passed : ℕ := (passed_students * 100) / total_students

theorem percentage_students_passed_is_35 : percentage_passed = 35 :=
by
  -- Here the proof will go.
  sorry

end percentage_students_passed_is_35_l377_377930


namespace determine_d_l377_377034

-- Given conditions
def equation (d x : ℝ) : Prop := 3 * (5 + d * x) = 15 * x + 15

-- Proof statement
theorem determine_d (d : ℝ) : (∀ x : ℝ, equation d x) ↔ d = 5 :=
by
  sorry

end determine_d_l377_377034


namespace positive_difference_of_probabilities_l377_377303

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377303


namespace max_value_of_f_l377_377097

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem max_value_of_f (a b : ℝ) (h1 : ∀ x ∈ set.Ioo (-1 : ℝ) 1, f a b x < 0)
  (h2 : f a b 1 = 2) : ∃ x, ∀ y, f a b x ≥ f a b y ∧ f a b x = 6 :=
begin
  sorry
end

end max_value_of_f_l377_377097


namespace digit_for_multiple_of_9_l377_377451

theorem digit_for_multiple_of_9 (d : ℕ) : (23450 + d) % 9 = 0 ↔ d = 4 := by
  sorry

end digit_for_multiple_of_9_l377_377451


namespace integral_correct_l377_377412

noncomputable def integral_problem : Real :=
  ∫ x in -2..2, (sin x + 2)

theorem integral_correct : integral_problem = 8 :=
  sorry

end integral_correct_l377_377412


namespace third_ball_white_probability_l377_377979

theorem third_ball_white_probability (n : ℕ) (h : 2 < n) :
  let prob := (n - 1) / (2 * n : ℚ) in
  (prob : ℚ) = (∑ k in Finset.range n, (n - k - 1) / (n * n : ℚ)) :=
by
  sorry

end third_ball_white_probability_l377_377979


namespace distance_covered_downstream_l377_377758

-- Conditions
def boat_speed_still_water : ℝ := 16
def stream_rate : ℝ := 5
def time_downstream : ℝ := 6

-- Effective speed downstream
def effective_speed_downstream := boat_speed_still_water + stream_rate

-- Distance covered downstream
def distance_downstream := effective_speed_downstream * time_downstream

-- Theorem to prove
theorem distance_covered_downstream :
  (distance_downstream = 126) :=
by
  sorry

end distance_covered_downstream_l377_377758


namespace potato_sales_l377_377799

theorem potato_sales :
  let total_weight := 6500
  let damaged_weight := 150
  let bag_weight := 50
  let price_per_bag := 72
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_weight
  let total_revenue := num_bags * price_per_bag
  total_revenue = 9144 :=
by
  sorry

end potato_sales_l377_377799


namespace intersection_eq_l377_377896

noncomputable def set_A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
noncomputable def set_B : Set ℝ := { x | x < 0 ∨ x > 3 }

theorem intersection_eq : set_A ∩ set_B = { x | 3 < x ∧ x ≤ 5 } := by
  sorry

end intersection_eq_l377_377896


namespace units_digit_square_l377_377350

theorem units_digit_square (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) (h2 : (n % 10 = 2) ∨ (n % 10 = 7)) :
  ∀ (d : ℕ), (d = 2 ∨ d = 6 ∨ d = 3) → (n^2 % 10 ≠ d) :=
by
  sorry

end units_digit_square_l377_377350


namespace positive_difference_prob_3_and_4_heads_l377_377322

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377322


namespace train_passing_time_l377_377590

def mph_to_fps (speed_mph: ℝ) : ℝ :=
  speed_mph * 5280 / 3600

theorem train_passing_time :
  ∀ (train_length : ℝ) (train_speed_mph : ℝ),
  train_length = 2500 → train_speed_mph = 120 →
  train_length / (mph_to_fps train_speed_mph) ≈ 14.2045 :=
by
  intros train_length train_speed_mph h₁ h₂
  rw [h₁, h₂]
  have fps : mph_to_fps 120 = 176 := by
    unfold mph_to_fps
    norm_num
  rw [fps]
  norm_num
  sorry

end train_passing_time_l377_377590


namespace determine_b_l377_377255

variable (q a b c : ℝ)

-- Conditions
def parabola_equation (x : ℝ) : ℝ := a * x^2 + b * x + c
def vertex_of_parabola : Prop := ∀ x, parabola_equation q = 2 * q
def y_intercept : Prop := parabola_equation 0 = -3 * q
def q_nonzero : Prop := q ≠ 0

-- Main theorem statement
theorem determine_b (h_vertex : vertex_of_parabola q a b c) (h_y_intercept : y_intercept q a b c) (h_q_nonzero : q_nonzero q) : b = 10 :=
sorry

end determine_b_l377_377255


namespace diff_of_squares_1500_l377_377532

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377532


namespace parabolaIntersectionSum_is_4_l377_377058

open Real

noncomputable def parabolaIntersectionSum : ℝ :=
  let x_eqn := (y : ℝ) → (x - 2 = y) in
  let y_eqn := (x : ℝ) → (y + 1 = x) in
  let xs := {x : ℝ | ∃ y : ℝ, y = (x - 2)^2 ∧ x + 6 = (y + 1)^2} in
  let ys := {y : ℝ | ∃ x : ℝ, x + 6 = (y + 1)^2 ∧ y = (x - 2)^2} in
  ∑ x in xs, x + ∑ y in ys, y

theorem parabolaIntersectionSum_is_4 :
  parabolaIntersectionSum = 4 := by
  sorry

end parabolaIntersectionSum_is_4_l377_377058


namespace volume_ratio_l377_377243

/-- Conditions for the problem -/
variables {x y z : ℝ} {n : ℝ} (h_n : 1 < n)
  (AD BC : ℝ) (h_AD_BC : AD / BC = n)

noncomputable def V_prism (BC y z : ℝ) (n : ℝ) : ℝ :=
  (n + 1) * BC * y * z / 2

noncomputable def V_pyramid (BC y z : ℝ) (n : ℝ) : ℝ :=
  1 / 6 * ((5 * n + 3) / 2) * BC * ((5 * n + 3) / (n + 1)) * z * y * 
  (3 / 2 + (n / (n + 1))) 

theorem volume_ratio (h_AD_BC : AD / BC = n) :
  (V_pyramid BC y z n) / (V_prism BC y z n) = (5 * n + 3)^3 / (12 * (n + 1)^3) :=
sorry

end volume_ratio_l377_377243


namespace find_a1_find_a_n_l377_377875

noncomputable def sequence_sum (n : ℕ) : ℤ := n^2 - 3

def a1 : ℤ := sequence_sum 1
def a_n (n : ℕ) (hn : n ≥ 2) : ℤ := sequence_sum n - sequence_sum (n - 1)

theorem find_a1 : a1 = -2 :=
by {
  unfold a1,
  unfold sequence_sum,
  simp,
  norm_num,
  sorry
}

theorem find_a_n (n : ℕ) (hn : n ≥ 2) : a_n n hn = 2 * n - 1 :=
by {
  unfold a_n,
  unfold sequence_sum,
  cases hn with hn₂ hn₁,
  simp,
  norm_num,
  sorry
}

end find_a1_find_a_n_l377_377875


namespace perpendicular_vectors_magnitude_equality_l377_377201

open Real

noncomputable def alpha : Type := ℝ
def a (α : ℝ) : ℝ × ℝ := (cos α, sin α)
def b : ℝ × ℝ := (-1 / 2, sqrt 3 / 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Problem 1
theorem perpendicular_vectors (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * pi) :
  dot_product (a α + b) (a α - b) = 0 :=
sorry

-- Problem 2
theorem magnitude_equality (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * pi) :
  (√3 * (a α) + b).prod.snd = ((a α) - √3 * b).prod.snd →
  (α = pi / 6 ∨ α = 7 * pi / 6) :=
sorry

end perpendicular_vectors_magnitude_equality_l377_377201


namespace g_inv_eq_neg_one_l377_377977

variable {a b x : ℝ}
-- Define the function g
def g (x : ℝ) : ℝ := 1 / (2 * a * x + 3 * b)

-- Non-zero constraints on constants a and b
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)

-- Lean statement to prove the equivalence
theorem g_inv_eq_neg_one (ha : a ≠ 0) (hb : b ≠ 0) : 
  g x = -1 ↔ x = (-1 - 3 * b) / (2 * a) := 
by
  sorry

end g_inv_eq_neg_one_l377_377977


namespace max_angle_subtended_at_foci_l377_377225

theorem max_angle_subtended_at_foci (a b : ℝ) (hp : a > b ∧ b > 0) (x y : ℝ) 
  (h : b^2 * x^2 + a^2 * y^2 = a^2 * b^2) :
  ∃ P, ∀ F1 F2, ∠ F1 P F2 ≤ real.arccos (2 * b^2 / a^2 - 1) :=
  sorry

end max_angle_subtended_at_foci_l377_377225


namespace ratio_area_ABC_to_BPC_l377_377616

variables {A B C P : Type} 
variables [AffineSpace ℝ V] [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points A, B, C, and P
variables (a b c p : V)

-- Given condition
axiom condition : 2 • (a - p) + (b - p) + 3 • (c - p) = 0

-- Define areas
noncomputable def area_ABC : ℝ := sorry
noncomputable def area_BPC : ℝ := sorry

-- Main theorem to prove
theorem ratio_area_ABC_to_BPC (h : condition) : 
  area_ABC a b c / area_BPC b p c = 3 / 2 := 
sorry

end ratio_area_ABC_to_BPC_l377_377616


namespace age_difference_is_100_l377_377254

-- Definition of the ages
variables {X Y Z : ℕ}

-- Conditions from the problem statement
axiom age_condition1 : X + Y > Y + Z
axiom age_condition2 : Z = X - 100

-- Proof to show the difference is 100 years
theorem age_difference_is_100 : (X + Y) - (Y + Z) = 100 :=
by sorry

end age_difference_is_100_l377_377254


namespace sum_of_digits_of_m_l377_377996

theorem sum_of_digits_of_m {m : ℕ} (h1: 0 < m) (h2 : (m+2)! + (m+3)! = m! * 1080) : 
  nat.digits 10 m.sum = 7 :=
sorry

end sum_of_digits_of_m_l377_377996


namespace arithmetic_mean_of_1_and_4_l377_377671

theorem arithmetic_mean_of_1_and_4 : 
  (1 + 4) / 2 = 5 / 2 := by
  sorry

end arithmetic_mean_of_1_and_4_l377_377671


namespace max_common_ratio_proof_l377_377578

noncomputable def max_common_ratio_q (a1 : ℝ) (q : ℝ) : ℝ :=
  if a1 ≠ 0 ∧ a1 * (a1 * q + a1 * q^2) = 6 * a1 - 9
  then ((-1 + real.sqrt 5) / 2)
  else 0

theorem max_common_ratio_proof :
  ∀ (a1 q: ℝ), a1 ≠ 0 ∧ a1 * (a1 * q + a1 * q ^ 2) = 6 * a1 - 9  → 
  q ≤ ((-1 + real.sqrt 5) / 2) :=
by
  intros a1 q h,
  sorry

end max_common_ratio_proof_l377_377578


namespace sum_of_chords_lt_k_pi_l377_377964

theorem sum_of_chords_lt_k_pi (k : ℕ) (h_k_pos : k > 0) (chords : set (metric.segment ℝ (1 : ℝ)))
  (h_chords_fin : set.finite chords)
  (h_diameter_intersects : ∀ d : set (metric.segment ℝ (1 : ℝ)),
    (∃ p₁ p₂, metric.segment p₁ p₂ = d ∧ p₁ + p₂ = 0) → (∃ N, N ≤ k ∧ ∀ p ∈ metric.ball 0 1, d ∈ chords → p ∈ d)) :
  (sum (λ c, metric.length c) chords.to_finset) < k * Real.pi :=
sorry

end sum_of_chords_lt_k_pi_l377_377964


namespace incorrect_reasoning_exponential_l377_377823

theorem incorrect_reasoning_exponential :
  (∀ (a : ℝ), a > 0 ∧ a ≠ 1 → ∀ (x y : ℝ), x < y → a^x < a^y) → -- Exponential function increasing condition
  (∀ (x : ℝ), 2^x = real.exp (x * real.log 2)) → -- y=2^x is an exponential function
  (∃ (a : ℝ), (a > 1) → ∀ x, real.log a x is_increasing) ∨ -- Increasing log function condition when a > 1
  (0 < a ∧ a < 1) → ∀ x, real.log a x is_decreasing) ∨ -- Decreasing log function condition when 0 < a < 1
  ∃ (x : ℝ), real.exp (x * real.log 2) is_increasing := -- Incorrect conclusion inferred
sorry

end incorrect_reasoning_exponential_l377_377823


namespace solution_set_abs_inequality_l377_377260

theorem solution_set_abs_inequality (x : ℝ) :
  {x : ℝ | |x| ≥ 2 * (x - 1)} = set.Iic 2 :=
sorry

end solution_set_abs_inequality_l377_377260


namespace find_annual_interest_rate_l377_377844

-- Define the given conditions
def principal : ℝ := 10000
def time : ℝ := 1  -- since 12 months is 1 year for annual rate
def simple_interest : ℝ := 800

-- Define the annual interest rate to be proved
def annual_interest_rate : ℝ := 0.08

-- The theorem stating the problem
theorem find_annual_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) : 
  P = principal → 
  T = time → 
  SI = simple_interest → 
  SI = P * annual_interest_rate * T := 
by
  intros hP hT hSI
  rw [hP, hT, hSI]
  unfold annual_interest_rate
  -- here's where we skip the proof
  sorry

end find_annual_interest_rate_l377_377844


namespace total_invested_expression_l377_377579

variables (x y T : ℝ)

axiom annual_income_exceed_65 : 0.10 * x - 0.08 * y = 65
axiom total_invested_is_T : x + y = T

theorem total_invested_expression :
  T = 1.8 * y + 650 :=
sorry

end total_invested_expression_l377_377579


namespace solve_inequality_l377_377667

theorem solve_inequality :
  ∀ (x : ℝ), (4 * log 16 (cos (2 * x)) + 2 * log 4 (sin x) + log 2 (cos x) + 3 < 0) ↔
  (0 < x ∧ x < π / 24) ∨ (5 * π / 24 < x ∧ x < π / 4) :=
by
  sorry

end solve_inequality_l377_377667


namespace find_numbers_l377_377446

theorem find_numbers (x y : ℝ) (h₁ : x + y = x * y) (h₂ : x * y = x / y) :
  (x = 1 / 2) ∧ (y = -1) := by
  sorry

end find_numbers_l377_377446


namespace diff_of_squares_count_l377_377515

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377515


namespace coin_probability_difference_l377_377329

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377329


namespace range_of_a_l377_377906

theorem range_of_a (x a : ℝ) (h1 : -2 < x) (h2 : x ≤ 1) (h3 : |x - 2| < a) : a ≤ 0 :=
sorry

end range_of_a_l377_377906


namespace max_determinant_value_l377_377049

open Real

def matrix_determinant (θ : ℝ) : ℝ :=
  det ![
    ![1 + sin θ, 1 + cos θ, 1],
    ![1, 1, 1 + sin θ],
    ![1 + cos θ, 1 + sin θ, 1]
  ]

theorem max_determinant_value : ∀ θ : ℝ, matrix_determinant θ ≤ -1 :=
by
  sorry

end max_determinant_value_l377_377049


namespace number_of_proper_subsets_M_inter_N_l377_377120

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = 1 + Real.sin (Real.pi * x / 2)}

theorem number_of_proper_subsets_M_inter_N : 
  (Finset.univ.filter (λ s: Set ℝ, s ⊆ (Set.inter M N) ∧ s ≠ (Set.inter M N))).card = 3 :=
sorry

end number_of_proper_subsets_M_inter_N_l377_377120


namespace arithmetic_sequence_sum_l377_377462

/-- Given an arithmetic sequence {a_n} with the sum of its first n terms denoted as S_n. It is known that a_2 = 0 and S_5 = 2a_4 - 1. Prove that
the sum of the first n terms for the sequence {b_n} defined as b_n = 2^(a_n) is T_n = 8 - 2^(2-n). -/
theorem arithmetic_sequence_sum (a S T : ℕ → ℝ)
  (h₁ : a 2 = 0)
  (h₂ : S 5 = 2 * a 4 - 1)
  (h₃ : ∀ n, a n = 2 - n)
  (h₄ : ∀ n, T n = ∑ i in finset.range n, 2^(a i)) :
  ∀ n, T n = 8 - 2^(2 - n) :=
by
  sorry

end arithmetic_sequence_sum_l377_377462


namespace area_of_shaded_region_l377_377158

theorem area_of_shaded_region :
  let width := 10
  let height := 5
  let base_triangle := 3
  let height_triangle := 2
  let top_base_trapezoid := 3
  let bottom_base_trapezoid := 6
  let height_trapezoid := 3
  let area_rectangle := width * height
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle
  let area_trapezoid := (1 / 2 : ℝ) * (top_base_trapezoid + bottom_base_trapezoid) * height_trapezoid
  let area_shaded := area_rectangle - area_triangle - area_trapezoid
  area_shaded = 33.5 :=
by
  sorry

end area_of_shaded_region_l377_377158


namespace count_of_squares_difference_l377_377521

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377521


namespace tetrahedron_non_coplanar_points_count_l377_377262

theorem tetrahedron_non_coplanar_points_count :
  let points := (vertices_of_tetrahedron ∪ midpoints_of_edges_tetrahedron : set ℝ),
      num_points := 10,
      num_selected := 4,
      num_coplanar_on_face := 4,
      num_coplanar_on_edges := 6,
      num_coplanar_parallelogram := 3,
      total_coplanar := num_coplanar_on_face + num_coplanar_on_edges + num_coplanar_parallelogram
  in
  points.card = num_points →
  combinatorics.choose num_points num_selected - total_coplanar = 197 :=
sorry

end tetrahedron_non_coplanar_points_count_l377_377262


namespace max_perfect_squares_among_pairwise_products_l377_377655

theorem max_perfect_squares_among_pairwise_products 
  (a b : ℕ) 
  (h_distinct : a ≠ b) 
  (h_distinct2 : a + 2 ≠ b) 
  (h_distinct3 : a ≠ b + 2) 
  (h_distinct4 : a + 2 ≠ b + 2) : 
  ∃ p1 p2 : ℕ, 
  (p1 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} ∧
   is_square p1) ∧
  (p2 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} ∧
   is_square p2) ∧
  p1 ≠ p2 ∧ 
  ∀ p3, p3 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} 
  → is_square p3 → p3 = p1 ∨ p3 = p2 := 
begin
  sorry
end

end max_perfect_squares_among_pairwise_products_l377_377655


namespace factorize_expression_l377_377435

variable {R : Type} [CommRing R] (m a : R)

theorem factorize_expression : m * a^2 - m = m * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expression_l377_377435


namespace number_of_integers_as_difference_of_squares_l377_377529

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377529


namespace age_difference_l377_377699

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end age_difference_l377_377699


namespace sum_of_exterior_angles_of_hexagon_l377_377696

theorem sum_of_exterior_angles_of_hexagon (hex : polygon 6) : sum_of_exterior_angles hex = 360 :=
sorry

end sum_of_exterior_angles_of_hexagon_l377_377696


namespace minimum_common_ratio_l377_377099

theorem minimum_common_ratio (a : ℕ) (n : ℕ) (q : ℝ) (h_pos : ∀ i, i < n → 0 < a * q^i) (h_geom : ∀ i j, i < j → a * q^i < a * q^j) (h_q : 1 < q ∧ q < 2) : q = 6 / 5 :=
by
  sorry

end minimum_common_ratio_l377_377099


namespace no_positive_integer_sequence_exists_irrational_sequence_exists_l377_377368

-- Define the sequence properties and the inequality condition
def sequence_exists (S : ℕ → ℤ) (P : ℕ → Prop) (R : ℤ → ℤ → ℤ → Prop) : Prop :=
  ∀ (n : ℕ), P (S n) → R (S (n+1)) (S n) (S (n+2))

-- Problem 1: Positive integers sequence does not exist
theorem no_positive_integer_sequence_exists :
  ¬ ∃ (S : ℕ → ℤ) (P : ℕ → Prop) (R : ℤ → ℤ → ℤ → Prop),
    (∀ n, P (S n) → R (S (n+1)) (S n) (S (n+2))) ∧
    (∀ x, P x → x > 0) ∧
    R = (λ a b c, a * a ≥ 2 * b * c) := sorry

-- Problem 2: Irrational numbers sequence exists
theorem irrational_sequence_exists :
  ∃ (S : ℕ → ℝ) (P : ℕ → Prop) (R : ℝ → ℝ → ℝ → Prop),
    (∀ n, P (S n) → R (S (n-1)) (S n) (S (n-2))) ∧
    (∀ x, P x → x ∉ ℚ) ∧
    R = (λ a b c, a * a ≥ 2 * b * c) := sorry

end no_positive_integer_sequence_exists_irrational_sequence_exists_l377_377368


namespace max_exp_e_X_correct_l377_377617

noncomputable def max_exp_e_X 
  (X : Type) [ProbabilitySpace X] (b σ : ℝ) (hb : b > 0) (hσ : σ > 0)
  (hx_abs : ∀ x, |x| ≤ b) (hx_mean : E[X] = 0) (hx_var : Var[X] = σ^2) : ℝ :=
  (exp b * σ^2 + exp (-(σ^2 / b)) * b^2) / (σ^2 + b^2)

theorem max_exp_e_X_correct 
  (X : Type) [ProbabilitySpace X] (b σ : ℝ) (hb : b > 0) (hσ : σ > 0)
  (hx_abs : ∀ x, |x| ≤ b) (hx_mean : E[X] = 0) (hx_var : Var[X] = σ^2) : 
  E[exp X] ≤ max_exp_e_X X b σ hb hσ hx_abs hx_mean hx_var := 
sorry

end max_exp_e_X_correct_l377_377617


namespace frog_hops_coprime_l377_377366

theorem frog_hops_coprime (p q d : ℕ) (h_coprime : Nat.coprime p q)
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_d_lt_pq : d < p + q)
  (frog_returns : ∃ (n m : ℤ), n * p = m * q): 
  ∃ (a b : ℤ), (frog_visits a) ∧ (frog_visits b) ∧ (abs (a - b) = d) :=
by
  sorry

end frog_hops_coprime_l377_377366


namespace sum_is_correct_l377_377410

noncomputable def calculate_sum : ℚ :=
  (4 / 3) + (13 / 9) + (40 / 27) + (121 / 81) - (8 / 3)

theorem sum_is_correct : calculate_sum = 171 / 81 := 
by {
  sorry
}

end sum_is_correct_l377_377410


namespace graph_of_equation_l377_377022

theorem graph_of_equation (x y : ℝ) : (x - y) ^ 2 = x ^ 2 + y ^ 2 ↔ (x = 0 ∨ y = 0) :=
by
  intro h
  refine ⟨fun h => _, fun h => _⟩
  sorry

end graph_of_equation_l377_377022


namespace min_moves_to_visit_all_non_forbidden_squares_l377_377864

def min_diagonal_moves (n : ℕ) : ℕ :=
  2 * (n / 2) - 1

theorem min_moves_to_visit_all_non_forbidden_squares (n : ℕ) :
  min_diagonal_moves n = 2 * (n / 2) - 1 := by
  sorry

end min_moves_to_visit_all_non_forbidden_squares_l377_377864


namespace total_oranges_l377_377265

variable a : ℝ
variable x : ℝ
variable y : ℝ

-- Conditions:
def cond1 : a = 22.5 := sorry
def cond2 : x = 2 * a + 3 := sorry
def cond3 : y = x - 11.5 := sorry

-- Question: Prove the total number of oranges
theorem total_oranges : a + x + y = 107 := by
  apply cond1
  apply cond2
  apply cond3
  sorry

end total_oranges_l377_377265


namespace triangle_equality_of_altitude_and_angle_l377_377580

noncomputable theory

-- Define the triangle and its properties
variables {A B C D : Type}
variables (a b c d : A)
variables [field A]
variables [decidable_eq A]
variables [linear_ordered_field A]

-- Given conditions
def angle_B (a b c : A) := (75 : ℝ) -- ∠B is 75 degrees
def altitude_condition (b c : A) :=
  ∃ D : A, ∃ h_d1 : A, ∃ h_d2 : A, h_d1 = b → h_d2 = c → D = h_d1/2 -- The altitude from B to AC is half of BC

-- Proof goal
theorem triangle_equality_of_altitude_and_angle
  (ABC : ∀ (A B C : A), angle_B A B C = 75)
  (alt_cond : altitude_condition b c) :
  a = b := -- Prove AC = BC
sorry

end triangle_equality_of_altitude_and_angle_l377_377580


namespace recurrence_relation_exponential_generating_function_bell_numbers_less_than_factorial_bell_numbers_limit_l377_377239

noncomputable def bellNumbers : ℕ → ℕ 
| 0     := 1
| (n+1) := ∑ k in Finset.range (n+1), Stirling.secondKind (n+1) k

theorem recurrence_relation (N : ℕ) (h : N ≥ 1) :
  bellNumbers N = ∑ k in Finset.range N, Nat.choose (N-1) k * bellNumbers (N-k-1) :=
sorry

noncomputable def E_B (x : ℝ) : ℝ := ∑ n in Finset.range 100, (bellNumbers n * x^n) / nat.factorial n

theorem exponential_generating_function (x : ℝ) :
  E_B(x) = Real.exp (Real.exp x - 1) :=
sorry

theorem bell_numbers_less_than_factorial (N : ℕ) (h : N ≥ 1) :
  bellNumbers N < nat.factorial N :=
sorry

theorem bell_numbers_limit :
  filter.tendsto (λ N, (bellNumbers N / nat.factorial N)^(1 / (N:ℝ))) filter.at_top (𝓝 0) :=
sorry

example : bellNumbers 1 = 1 := rfl
example : bellNumbers 2 = 2 := rfl
example : bellNumbers 3 = 5 := rfl
example : bellNumbers 4 = 15 := rfl
example : bellNumbers 5 = 52 := rfl

end recurrence_relation_exponential_generating_function_bell_numbers_less_than_factorial_bell_numbers_limit_l377_377239


namespace linear_regression_decrease_l377_377858

theorem linear_regression_decrease (x : ℝ) (y : ℝ) :
  (h : ∃ c₀ c₁, (c₀ = 2) ∧ (c₁ = -1.5) ∧ y = c₀ - c₁ * x) →
  ( ∃ Δx, Δx = 1 → ∃ Δy, Δy = -1.5) :=
by 
  sorry

end linear_regression_decrease_l377_377858


namespace pints_in_1_5_liters_l377_377139

theorem pints_in_1_5_liters:
  let conversion_factor := 2.1
  let liters := 1.5
  let expected_pints := 3.2 in
  (liters * conversion_factor).round == expected_pints :=
by
  let conversion_factor := 2.1
  let liters := 1.5
  let expected_pints := 3.2
  have h : (1.5 * 2.1).round = 3.2 := sorry
  exact h

end pints_in_1_5_liters_l377_377139


namespace area_of_triangle_FNV_l377_377165

theorem area_of_triangle_FNV (EFGH_rectangle : ∀ (E F G H : Point), Rectangle E F G H) 
  (EK WF : ℝ) (EF : ℝ) (area_trap_KWFG : ℝ) (KV_eq_VW : ∀ (K V W : Point), Midpoint V K W) 
  (h1 : EK = 5) (h2 : WF = 5) (h3 : EF = 15) (h4 : area_trap_KWFG = 150) : 
  area_triangle F N V = 125 := 
by
  sorry

end area_of_triangle_FNV_l377_377165


namespace number_of_girls_calculation_l377_377705

theorem number_of_girls_calculation : 
  ∀ (number_of_boys number_of_girls total_children : ℕ), 
  number_of_boys = 27 → total_children = 62 → number_of_girls = total_children - number_of_boys → number_of_girls = 35 :=
by
  intros number_of_boys number_of_girls total_children 
  intros h_boys h_total h_calc
  rw [h_boys, h_total] at h_calc
  simp at h_calc
  exact h_calc

end number_of_girls_calculation_l377_377705


namespace area_of_square_l377_377784

theorem area_of_square (side_length : ℝ) (h : side_length = 15) : 
  (side_length * side_length) = 225 :=
by
  rw h
  norm_num
  sorry

end area_of_square_l377_377784


namespace cubic_roots_relations_l377_377646

theorem cubic_roots_relations 
    (a b c d : ℚ) 
    (x1 x2 x3 : ℚ) 
    (h : a ≠ 0)
    (hroots : a * x1^3 + b * x1^2 + c * x1 + d = 0 
      ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 
      ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
    :
    (x1 + x2 + x3 = -b / a) 
    ∧ (x1 * x2 + x1 * x3 + x2 * x3 = c / a) 
    ∧ (x1 * x2 * x3 = -d / a) := 
sorry

end cubic_roots_relations_l377_377646


namespace shaniqua_haircuts_l377_377231

theorem shaniqua_haircuts
  (H : ℕ) -- number of haircuts
  (haircut_income : ℕ) (style_income : ℕ)
  (total_styles : ℕ) (total_income : ℕ)
  (haircut_income_eq : haircut_income = 12)
  (style_income_eq : style_income = 25)
  (total_styles_eq : total_styles = 5)
  (total_income_eq : total_income = 221)
  (income_from_styles : ℕ := total_styles * style_income)
  (income_from_haircuts : ℕ := total_income - income_from_styles) :
  H = income_from_haircuts / haircut_income :=
sorry

end shaniqua_haircuts_l377_377231


namespace new_person_weight_is_75_l377_377744

noncomputable def new_person_weight (previous_person_weight: ℝ) (average_increase: ℝ) (total_people: ℕ): ℝ :=
  previous_person_weight + total_people * average_increase

theorem new_person_weight_is_75 :
  new_person_weight 55 2.5 8 = 75 := 
by
  sorry

end new_person_weight_is_75_l377_377744


namespace total_aprons_l377_377899

-- Define the given conditions
def initial_aprons : ℕ := 13
def aprons_today : ℕ := 3 * initial_aprons
def sewn_so_far : ℕ := initial_aprons + aprons_today
def aprons_needed_tomorrow : ℕ := 49
def remaining_aprons_needed : ℕ := 2 * aprons_needed_tomorrow
def total_aprons_needed : ℕ := sewn_so_far + remaining_aprons_needed

-- The theorem to prove the total number of aprons needed
theorem total_aprons (initial_aprons = 13)
                     (aprons_today = 3 * initial_aprons)
                     (sewn_so_far = initial_aprons + aprons_today)
                     (remaining_aprons_needed = 2 * aprons_needed_tomorrow)
                     (total_aprons_needed = sewn_so_far + remaining_aprons_needed)
                     (total_aprons_needed = 150) : total_aprons_needed = 150 :=
by
  sorry

end total_aprons_l377_377899


namespace numberOfSolutions_l377_377843

noncomputable def numberOfRealPositiveSolutions(x : ℝ) : Prop := 
  (x^6 + 1) * (x^4 + x^2 + 1) = 6 * x^5

theorem numberOfSolutions : ∃! x : ℝ, numberOfRealPositiveSolutions x := 
by
  sorry

end numberOfSolutions_l377_377843


namespace distance_points_3_12_and_10_0_l377_377840

theorem distance_points_3_12_and_10_0 : 
  Real.sqrt ((10 - 3)^2 + (0 - 12)^2) = Real.sqrt 193 := 
by
  sorry

end distance_points_3_12_and_10_0_l377_377840


namespace strawberries_left_in_each_bucket_l377_377413

-- Definitions of initial conditions
variables (total_strawberries buckets strawberries_removed_per_bucket : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  total_strawberries = 300 ∧ buckets = 5 ∧ strawberries_removed_per_bucket = 20

-- The theorem representing the proof problem
theorem strawberries_left_in_each_bucket
  (h : initial_conditions total_strawberries buckets strawberries_removed_per_bucket) :
  (total_strawberries / buckets - strawberries_removed_per_bucket) = 40 :=
by {
  obtain ⟨h1, h2, h3⟩ := h,
  sorry
}

end strawberries_left_in_each_bucket_l377_377413


namespace binomial_coefficient_8_0_l377_377813

theorem binomial_coefficient_8_0 : nat.choose 8 0 = 1 := 
by {
 sorry
}

end binomial_coefficient_8_0_l377_377813


namespace Larry_wins_probability_l377_377177

noncomputable def probability_Larry_wins (p_Larry p_Julius : ℚ) : ℚ :=
  let a := p_Larry in
  let r := (1 - p_Larry) * (1 - p_Julius) in
  a / (1 - r)

theorem Larry_wins_probability :
  probability_Larry_wins (1/3) (1/4) = 2/3 := by
  sorry

end Larry_wins_probability_l377_377177


namespace calories_per_serving_l377_377710

theorem calories_per_serving (x : ℕ) (total_calories bread_calories servings : ℕ)
    (h1: total_calories = 500) (h2: bread_calories = 100) (h3: servings = 2)
    (h4: total_calories = bread_calories + (servings * x)) :
    x = 200 :=
by
  sorry

end calories_per_serving_l377_377710


namespace avg_age_of_community_l377_377156

def ratio_of_populations (w m : ℕ) : Prop := w * 2 = m * 3
def avg_age (total_age population : ℚ) : ℚ := total_age / population

theorem avg_age_of_community 
    (k : ℕ)
    (total_women : ℕ := 3 * k) 
    (total_men : ℕ := 2 * k)
    (total_children : ℚ := (2 * k : ℚ) / 3)
    (avg_women_age : ℚ := 40)
    (avg_men_age : ℚ := 36)
    (avg_children_age : ℚ := 10)
    (total_women_age : ℚ := 40 * (3 * k))
    (total_men_age : ℚ := 36 * (2 * k))
    (total_children_age : ℚ := 10 * (total_children)) : 
    avg_age (total_women_age + total_men_age + total_children_age) (total_women + total_men + total_children) = 35 := 
    sorry

end avg_age_of_community_l377_377156


namespace total_dollars_l377_377203

def mark_dollars : ℚ := 4 / 5
def carolyn_dollars : ℚ := 2 / 5
def jack_dollars : ℚ := 1 / 2

theorem total_dollars :
  mark_dollars + carolyn_dollars + jack_dollars = 1.7 := 
sorry

end total_dollars_l377_377203


namespace toby_loaded_speed_l377_377711

noncomputable def toby_speed_when_loaded : ℕ := 
let 
  v := 10, -- Let v be the speed we're trying to find
  speed_unloaded := 20, -- given speed of unloaded sled
  time_unloaded_first_part := 120 / speed_unloaded,
  time_unloaded_second_part := 140 / speed_unloaded,
  journey_time := 39,
  distance_loaded_first_part := 180,
  distance_loaded_second_part := 80
in 
  v

theorem toby_loaded_speed : toby_speed_when_loaded = 10 := 
by 
  let v := 10
  let speed_unloaded := 20
  let time_unloaded_first_part := (120 : ℕ) / speed_unloaded
  let time_unloaded_second_part := (140 : ℕ) / speed_unloaded
  let journey_time := 39
  let distance_loaded_first_part := 180
  let distance_loaded_second_part := 80
  let time_loaded := (distance_loaded_first_part + distance_loaded_second_part) / v
  have h_loading_time_appropriate : (distance_loaded_first_part + distance_loaded_second_part) / v + time_unloaded_first_part + time_unloaded_second_part = journey_time := by 
    sorry 
  exact h_loading_time_appropriate

end toby_loaded_speed_l377_377711


namespace intersection_of_A_and_B_l377_377495

-- Given conditions as definitions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 2 ^ x > 1}
def B : Set ℝ := {x | x^2 - 3 * x - 4 > 0}

-- The proof statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | x > 4} :=
sorry

end intersection_of_A_and_B_l377_377495


namespace positive_difference_of_probabilities_l377_377309

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377309


namespace positive_difference_probability_l377_377297

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377297


namespace coin_probability_difference_l377_377331

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377331


namespace polynomial_exists_with_integer_coeffs_l377_377449

theorem polynomial_exists_with_integer_coeffs (n : ℕ) (hn : n > 0) :
  ∃ P : ℤ[X], degree P = n ∧ (∀ i j : ℕ, i ≠ j → i ≤ n → j ≤ n → P.eval i ≠ P.eval j) ∧ (∀ i : ℕ, i ≤ n → ∃ k : ℤ, P.eval i = 2^k) :=
sorry

end polynomial_exists_with_integer_coeffs_l377_377449


namespace find_a_l377_377089

theorem find_a (x a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  A = (7, 1) ∧ B = (1, 4) ∧ C = (x, a * x) ∧ 
  (x - 7, a * x - 1) = (2 * (1 - x), 2 * (4 - a * x)) → 
  a = 1 :=
sorry

end find_a_l377_377089


namespace bags_on_monday_l377_377271

/-- Define the problem conditions -/
def t : Nat := 8  -- total number of bags
def f : Nat := 4  -- number of bags found the next day

-- Define the statement to be proven
theorem bags_on_monday : t - f = 4 := by
  -- Sorry to skip the proof
  sorry

end bags_on_monday_l377_377271


namespace binomial_constant_term_l377_377244

theorem binomial_constant_term :
  let x := (λ x : ℝ, x^6 + 1 / (x * real.sqrt x)) in
  binomial_expansion x 5 = 5 :=
by
  sorry

end binomial_constant_term_l377_377244


namespace sampling_interval_l377_377724

def population : ℕ := 1003
def sample_size : ℕ := 50

theorem sampling_interval (N : ℕ) (n : ℕ) (hN : N = 1003) (hn : n = 50) :
  (N - N % n) / n = 20 :=
by
  rw [hN, hn]
  sorry

end sampling_interval_l377_377724


namespace total_elementary_students_l377_377176

theorem total_elementary_students : ∀ (number_of_schools : ℕ) (students_per_school : ℕ), 
  number_of_schools = 25 → students_per_school = 247 → (number_of_schools * students_per_school) = 6175 :=
by
  intros number_of_schools students_per_school,
  intros h1 h2,
  rw [h1, h2],
  norm_num,
  exact eq.refl 6175

end total_elementary_students_l377_377176


namespace pencil_price_units_l377_377141

def pencil_price_in_units (pencil_price : ℕ) : ℚ := pencil_price / 10000

theorem pencil_price_units 
  (price_of_pencil : ℕ) 
  (h1 : price_of_pencil = 5000 - 20) : 
  pencil_price_in_units price_of_pencil = 0.5 := 
by
  sorry

end pencil_price_units_l377_377141


namespace smallest_positive_omega_l377_377714

theorem smallest_positive_omega (f g : ℝ → ℝ) (ω : ℝ) 
  (hf : ∀ x, f x = Real.cos (ω * x)) 
  (hg : ∀ x, g x = Real.sin (ω * x - π / 4)) 
  (heq : ∀ x, f (x - π / 2) = g x) :
  ω = 3 / 2 :=
sorry

end smallest_positive_omega_l377_377714


namespace find_f1_l377_377884

def f : ℝ → ℝ
| x := if x > 2 then 2^x else f (x + 1)

theorem find_f1 : f 1 = 8 := sorry

end find_f1_l377_377884


namespace derivative_at_neg_one_l377_377576

theorem derivative_at_neg_one (a b c : ℝ) (h : (4*a*(1:ℝ)^3 + 2*b*(1:ℝ)) = 2) :
  (4*a*(-1:ℝ)^3 + 2*b*(-1:ℝ)) = -2 :=
by
  sorry

end derivative_at_neg_one_l377_377576


namespace no_valid_replacement_l377_377951

theorem no_valid_replacement (f : Fin 10 → ℤ) (h : ∀ i : Fin 10, f i = i + 1 ∨ f i = -(i + 1)) : ∑ i, f i ≠ 0 :=
by
  -- Define a function summing the sequence 1 through 10 with replacements as + or -
  -- Then ensure no matter the replacement, the sum cannot be zero
  sorry

end no_valid_replacement_l377_377951


namespace max_perfect_squares_among_pairwise_products_l377_377654

theorem max_perfect_squares_among_pairwise_products 
  (a b : ℕ) 
  (h_distinct : a ≠ b) 
  (h_distinct2 : a + 2 ≠ b) 
  (h_distinct3 : a ≠ b + 2) 
  (h_distinct4 : a + 2 ≠ b + 2) : 
  ∃ p1 p2 : ℕ, 
  (p1 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} ∧
   is_square p1) ∧
  (p2 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} ∧
   is_square p2) ∧
  p1 ≠ p2 ∧ 
  ∀ p3, p3 ∈ {a * b, a * (b + 2), (a + 2) * b, a * (a + 2), b * (b + 2), (a + 2) * (b + 2)} 
  → is_square p3 → p3 = p1 ∨ p3 = p2 := 
begin
  sorry
end

end max_perfect_squares_among_pairwise_products_l377_377654


namespace projection_statements_correctness_l377_377797

theorem projection_statements_correctness :
  ∃ n : ℕ, n = 1 ∧ (
    (∀ (P C : Type) [is_parallel_projection P] [is_central_projection C],
      let parallel_projection := ∀ (l1 l2 : line), (proj P l1 l2 -> parallel l1 l2)
      let central_projection := ∀ (l1 l2 : line), (proj C l1 l2 -> intersect_at_one_point l1 l2)
      let space_figure := ∀ (sf : space_figure), (central_proj C sf -> line_remains_line_or_point sf)
      let parallel_lines := ∀ (l1 l2 : line), (parallel l1 l2 -> central_proj C (l1, l2) -> intersect l1 l2)
      in
      parallel_projection -> central_projection -> space_figure -> parallel_lines)
      ↔ (1 = 1 ∨ 2 = 3 ∨ 1 = 1))
  := sorry

end projection_statements_correctness_l377_377797


namespace range_of_f_l377_377257

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 * Real.tan x + (Real.cos x) ^ 4 * Real.cot x

theorem range_of_f :
  set.range f = set.Ioo (-∞) (-1/2) ∪ set.Ioo (1/2) ∞ := 
sorry

end range_of_f_l377_377257


namespace x_value_l377_377736

theorem x_value (x : ℤ) (h : x = (2009^2 - 2009) / 2009) : x = 2008 := by
  sorry

end x_value_l377_377736


namespace variable_range_neq_2_l377_377686

theorem variable_range_neq_2 (x : ℝ) : (1 / (x - 2)) ≠ 0 → x ≠ 2 :=
by
  intro h₁ : (1 / (x - 2)) ≠ 0
  intro h₂ : x = 2
  have : 1 / (2 - 2) = 1 / 0 by rw h₂
  contradiction
    sorry

end variable_range_neq_2_l377_377686


namespace dice_impossible_divisible_by_10_l377_377348

theorem dice_impossible_divisible_by_10 :
  ¬ ∃ n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ), n % 10 = 0 :=
by
  sorry

end dice_impossible_divisible_by_10_l377_377348


namespace intersection_M_N_l377_377199

def M := {0, 1, 2}
def N := {x : ℕ | x ≥ 1}

theorem intersection_M_N :
  M ∩ N = {1, 2} := 
sorry

end intersection_M_N_l377_377199


namespace max_writers_and_editors_l377_377754

theorem max_writers_and_editors (T W : ℕ) (E : ℕ) (x : ℕ) (hT : T = 100) (hW : W = 35) (hE : E > 38) (h_comb : W + E + x = T)
    (h_neither : T = W + E + x) : x = 26 := by
  sorry

end max_writers_and_editors_l377_377754


namespace bakery_buns_distribution_l377_377935

theorem bakery_buns_distribution :
  ∃ (n₁ n₂ n₃ : ℕ), -- the number of buns of each type (first-type, second-type, third-type)
    -- total children count and corresponding money per child
    let n := 6 in -- total number of children
    let per_child_cents := (7 / 6 : ℚ) in
    -- cost conditions for each type of buns
    n₁ + 2 * n₂ + 3 * n₃ = n * per_child_cents ∧
    -- total cents condition
    n₁ + 2 * n₂ + 3 * n₃ = 7 ∧
    -- each child gets the same amount
    ∀ (buns_per_child : ℕ), 
      (buns_per_child = 2 * (n₃ / n) + 1 * (n₂ / n)) →
      ∃ c₃, c₃ = buns_per_child :=
begin
  -- Definitions as conditions
  let buns_per_child := 2 + 1,
  -- The proof part will be added here
  sorry
end

end bakery_buns_distribution_l377_377935


namespace triangle_side_c_l377_377497

theorem triangle_side_c
  (a b c : ℝ)
  (A B C : ℝ)
  (h_bc : b = 3)
  (h_sinC : Real.sin C = 56 / 65)
  (h_sinB : Real.sin B = 12 / 13)
  (h_Angles : A + B + C = π)
  (h_valid_triangle : ∀ {x y z : ℝ}, x + y > z ∧ x + z > y ∧ y + z > x):
  c = 14 / 5 :=
sorry

end triangle_side_c_l377_377497


namespace complex_mult_example_l377_377434

-- Define a structure for complex number
structure Complex where
  re : Int
  im : Int

def i : Complex := { re := 0, im := 1 }

-- Define multiplication for complex numbers
instance : Mul Complex where
  mul z w := {
    re := z.re * w.re - z.im * w.im,
    im := z.re * w.im + z.im * w.re,
  }

-- Define a simplification theorem using the given condition i^2 = -1
theorem complex_mult_example : 
  (({re := 3, im := -4} : Complex) * ({re := -7, im := 2} : Complex) = {re := -13, im := 34}) :=
by
  sorry

end complex_mult_example_l377_377434


namespace positive_difference_prob_3_and_4_heads_l377_377321

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377321


namespace series_sum_equals_one_sixth_l377_377818

/-- T represents the sum of the infinite series 1 - (1/3) + (1/6) - (1/12) + (1/24) - (1/48) + ... -/
def T : ℝ := ∑' n, (-1)^n / (2^(n+1) * (n+1))

theorem series_sum_equals_one_sixth : T = 1 / 6 := 
by sorry

end series_sum_equals_one_sixth_l377_377818


namespace determinant_transformation_l377_377988

variables (u v w : ℝ^3)
variable (D : ℝ)
hypothesis hD : D = u.dot_product (v.cross_product w)

theorem determinant_transformation : 
  let u' := 2 • u + v,
      v' := v + 2 • w,
      w' := 2 • w + u
  in matrix.det (matrix.vec![u', v', w']) = 6 * D :=
by 
  sorry

end determinant_transformation_l377_377988


namespace faye_age_l377_377428

theorem faye_age (D E C F : ℤ)
  (h1 : D = E - 4)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (hD : D = 18) :
  F = 21 :=
by
  sorry

end faye_age_l377_377428


namespace truncated_cone_surface_area_and_volume_l377_377426

-- Define the truncated cone and its properties
variables (l r : ℝ) 

-- Lateral surface area of the truncated cone
def lateral_surface_area_of_truncated_cone (l r : ℝ) : ℝ := π * l^2

-- Volume of the truncated cone
def volume_of_truncated_cone (l r : ℝ) : ℝ := (2/3) * π * r * (l^2 - r^2)

theorem truncated_cone_surface_area_and_volume (l r : ℝ) (h_l_pos : l > 0) (h_r_pos : r > 0) :
  lateral_surface_area_of_truncated_cone l r = π * l^2 ∧ 
  volume_of_truncated_cone l r = (2/3) * π * r * (l^2 - r^2) := 
by
  sorry

end truncated_cone_surface_area_and_volume_l377_377426


namespace hyperbola_eccentricity_range_l377_377135

theorem hyperbola_eccentricity_range (a : ℝ) (h : 1 < a) : 
  ∃ e : ℝ, (e = (real.sqrt (1 + 1 / a^2)) ∧ 1 < e ∧ e < real.sqrt 2) :=
sorry

end hyperbola_eccentricity_range_l377_377135


namespace max_perfect_squares_l377_377660

theorem max_perfect_squares (a b : ℕ) (h : a ≠ b) : 
  let prod1 := a * (a + 2),
      prod2 := a * b,
      prod3 := a * (b + 2),
      prod4 := (a + 2) * b,
      prod5 := (a + 2) * (b + 2),
      prod6 := b * (b + 2) in
  (prod1.is_square ℕ + prod2.is_square ℕ + prod3.is_square ℕ + prod4.is_square ℕ + prod5.is_square ℕ + prod6.is_square ℕ) ≤ 2 := 
  sorry

end max_perfect_squares_l377_377660


namespace monthly_rent_l377_377397

theorem monthly_rent (investment : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (annual_insurance : ℝ) (maintenance_rate : ℝ) :
  investment = 12000 →
  annual_return_rate = 0.06 →
  annual_taxes = 360 →
  annual_insurance = 240 →
  maintenance_rate = 0.1 →
  (let annual_return := annual_return_rate * investment in
   let total_annual_expenses := annual_return + annual_taxes + annual_insurance in
   let monthly_earning_requirement := total_annual_expenses / 12 in
   let rent := monthly_earning_requirement / (1 - maintenance_rate) in
   rent = 122.22) :=
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end monthly_rent_l377_377397


namespace inequality_proof_l377_377142

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
sorry

end inequality_proof_l377_377142


namespace problem_A_problem_D_l377_377607

def fibonacci_like_seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | k + 2 => fibonacci_like_seq k + fibonacci_like_seq (k + 1)

theorem problem_A : (∑ i in finset.filter (λ i, i % 2 = 1) (finset.range 101), fibonacci_like_seq i) = fibonacci_like_seq 100 - 1 :=
by sorry

theorem problem_D : (∑ i in finset.range 100, (fibonacci_like_seq i) ^ 2) = fibonacci_like_seq 99 * fibonacci_like_seq 100 - 2 :=
by sorry

end problem_A_problem_D_l377_377607


namespace change_in_surface_area_l377_377776

-- Define the original dimensions of the rectangular solid
def length : ℝ := 4
def width : ℝ := 3
def height : ℝ := 5

-- Calculate the original surface area
def SA_original : ℝ := 2 * (length * width + length * height + width * height)

-- Define the radius of the removed sphere
def radius : ℝ := 1

-- Calculate the surface area of the removed sphere
def SA_sphere : ℝ := 4 * Real.pi * radius^2

-- Calculate the area of the exposed spherical cap
def SA_exposed : ℝ := 2 * Real.pi * radius * radius

-- Define the net change in surface area
def net_change : ℝ := SA_sphere - SA_exposed

-- Calculate the new surface area after the removal of the spherical section
def SA_new : ℝ := SA_original - net_change

-- Theorem statement
theorem change_in_surface_area : SA_new = 94 - 2 * Real.pi := by
  -- Original surface area calculation
  have h1 : SA_original = 94 := by
    calc
      SA_original = 2 * (length * width + length * height + width * height) : rfl
      ... = 2 * (4 * 3 + 4 * 5 + 3 * 5) : rfl
      ... = 2 * (12 + 20 + 15) : rfl
      ... = 2 * 47 : rfl
      ... = 94 : rfl
    
  -- Surface area of the removed sphere
  have h2 : SA_sphere = 4 * Real.pi := by
    calc
      SA_sphere = 4 * Real.pi * radius^2 : rfl
      ... = 4 * Real.pi * 1^2 : rfl
      ... = 4 * Real.pi : rfl

  -- Area of the exposed spherical cap
  have h3 : SA_exposed = 2 * Real.pi := by
    calc
      SA_exposed = 2 * Real.pi * radius * radius : rfl
      ... = 2 * Real.pi * 1 * 1 : rfl
      ... = 2 * Real.pi : rfl

  -- Net change in surface area
  have h4 : net_change = 2 * Real.pi := by
    calc
      net_change = SA_sphere - SA_exposed : rfl
      ... = 4 * Real.pi - 2 * Real.pi : by rw [h2, h3]
      ... = 2 * Real.pi : rfl

  -- Calculate the new surface area
  calc
    SA_new = SA_original - net_change : rfl
    ... = 94 - 2 * Real.pi : by rw [h1, h4]

  -- Complete the proof
  sorry

end change_in_surface_area_l377_377776


namespace count_true_propositions_l377_377901

theorem count_true_propositions (a b : ℝ) (c : ℝ) :
  let original_prop := (a > b → a * c^2 > b * c^2)
  let inverse_prop := (a * c^2 > b * c^2 → a > b)
  let converse_prop := (a * c^2 > b * c^2 → c^2 ≠ 0 ∧ c^2 > 0)
  let contrapositive_prop := (a * c^2 ≤ b * c^2 → a ≤ b)
  (¬original_prop ∧ ¬contrapositive_prop ∧ inverse_prop ∧ converse_prop) →
  (2 = (if original_prop then 1 else 0) +
       (if inverse_prop then 1 else 0) +
       (if converse_prop then 1 else 0) +
       (if contrapositive_prop then 1 else 0)) := 
by {
  let original_prop := (a > b → a * c^2 > b * c^2),
  let inverse_prop := (a * c^2 > b * c^2 → a > b),
  let converse_prop := (a * c^2 > b * c^2 → c^2 ≠ 0 ∧ c^2 > 0),
  let contrapositive_prop := (a * c^2 ≤ b * c^2 → a ≤ b),
  sorry -- Proof goes here
}

end count_true_propositions_l377_377901


namespace vertices_of_square_l377_377747

-- Variable declarations
variables {A B C : Type} [AddGroup A] [AddGroup B] [AddGroup C]

-- Definitions of side lengths and the area of the triangle
def AB (A B : A) : ℝ := sorry
def BC (B C : B) : ℝ := sorry
def area (A B C : C) : ℝ := sorry

-- Main theorem statement
theorem vertices_of_square (A B C : Type) [AddGroup A] [AddGroup B] [AddGroup C] 
  (a c : ℝ) (T : ℝ) :
  (AB A B + BC B C)^2 < 8 * T + 1 → 
  (A, B, C are vertices of a square) :=
by
  sorry

end vertices_of_square_l377_377747


namespace diff_of_squares_1500_l377_377533

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377533


namespace james_weekly_earnings_l377_377954

def rate_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

def daily_earnings : ℕ := rate_per_hour * hours_per_day
def weekly_earnings : ℕ := daily_earnings * days_per_week

theorem james_weekly_earnings : weekly_earnings = 640 := sorry

end james_weekly_earnings_l377_377954


namespace intersection_of_M_and_N_l377_377121

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_of_M_and_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ 2} :=
by sorry

end intersection_of_M_and_N_l377_377121


namespace Carla_servings_l377_377015

-- Define the volumes involved
def volume_watermelon : ℕ := 500
def volume_cream : ℕ := 100
def volume_per_serving : ℕ := 150

-- The total volume is the sum of the watermelon and cream volumes
def total_volume : ℕ := volume_watermelon + volume_cream

-- The number of servings is the total volume divided by the volume per serving
def n_servings : ℕ := total_volume / volume_per_serving

-- The theorem to prove that Carla can make 4 servings of smoothies
theorem Carla_servings : n_servings = 4 := by
  sorry

end Carla_servings_l377_377015


namespace probability_not_B_given_A_l377_377464

noncomputable def PA : ℚ := 1/3
noncomputable def PB : ℚ := 1/4
noncomputable def P_A_given_B : ℚ := 3/4

def PnotB_given_A : ℚ := 1 - (P_A_given_B * PB / PA)

theorem probability_not_B_given_A (PA PB P_A_given_B : ℚ) 
  (hPA : PA = 1/3) (hPB : PB = 1/4) (hP_A_given_B : P_A_given_B = 3/4) : 
  PnotB_given_A = 7/16 :=
by
  rw [hPA, hPB, hP_A_given_B]
  simp [PnotB_given_A, PA, PB, P_A_given_B]
  sorry

end probability_not_B_given_A_l377_377464


namespace coin_flip_probability_difference_l377_377341

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377341


namespace domain_of_function_l377_377841

theorem domain_of_function :
  {x : ℝ | x^3 + 5*x^2 + 6*x ≠ 0} =
  {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < -2} ∪ {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry

end domain_of_function_l377_377841


namespace unique_intersection_of_curve_and_line_l377_377827

noncomputable def f (a : ℝ) (x : ℝ) := (1/8)*x^2 - a*x

theorem unique_intersection_of_curve_and_line (a : ℝ) (n : ℝ) (x : ℝ)
  (h1 : a = -1/8) (h2 : 0 < x ∧ x < 2) : 
  ∃! t ∈ Ioo 0 2, f a t = f a x := 
sorry

end unique_intersection_of_curve_and_line_l377_377827


namespace number_of_medium_boxes_l377_377822

def large_box_tape := 4
def medium_box_tape := 2
def small_box_tape := 1
def label_tape := 1

def num_large_boxes := 2
def num_small_boxes := 5
def total_tape := 44

theorem number_of_medium_boxes :
  let tape_used_large_boxes := num_large_boxes * (large_box_tape + label_tape)
  let tape_used_small_boxes := num_small_boxes * (small_box_tape + label_tape)
  let tape_used_medium_boxes := total_tape - (tape_used_large_boxes + tape_used_small_boxes)
  let medium_box_total_tape := medium_box_tape + label_tape
  let num_medium_boxes := tape_used_medium_boxes / medium_box_total_tape
  num_medium_boxes = 8 :=
by
  sorry

end number_of_medium_boxes_l377_377822


namespace positive_difference_prob_3_and_4_heads_l377_377319

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377319


namespace dogs_not_liking_any_food_l377_377585

-- Declare variables
variable (n w s ws c cs : ℕ)

-- Define problem conditions
def total_dogs := n
def dogs_like_watermelon := w
def dogs_like_salmon := s
def dogs_like_watermelon_and_salmon := ws
def dogs_like_chicken := c
def dogs_like_chicken_and_salmon_but_not_watermelon := cs

-- Define the statement proving the number of dogs that do not like any of the three foods
theorem dogs_not_liking_any_food : 
  n = 75 → 
  w = 15 → 
  s = 54 → 
  ws = 12 → 
  c = 20 → 
  cs = 7 → 
  (75 - ((w - ws) + (s - ws - cs) + (c - cs) + ws + cs) = 5) :=
by
  intros _ _ _ _ _ _
  sorry

end dogs_not_liking_any_food_l377_377585


namespace count_numbers_with_odd_divisors_under_100_l377_377127

-- Definition of the required conditions: less than 100 and having an odd number of divisors
def is_less_than_100 (n : ℕ) : Prop := n < 100
def has_odd_number_of_divisors (n : ℕ) : Prop := (finset.filter (λ (d : ℕ), n % d = 0) (finset.range (n + 1))).card % 2 = 1

-- The theorem to prove the required statement
theorem count_numbers_with_odd_divisors_under_100 : 
  finset.card (finset.filter (λ n, is_less_than_100 n ∧ has_odd_number_of_divisors n) (finset.range 100)) = 9 :=
by
  -- Placeholder for the actual proof
  sorry

end count_numbers_with_odd_divisors_under_100_l377_377127


namespace fourth_student_in_sample_l377_377760

theorem fourth_student_in_sample :
  ∃ n : ℕ, n = 1 ∧ 
  (let total_students := 54 in
   let sample_size := 4 in
   let included_students := [2, 28, 41] in
   let sampling_interval := 13 in
   let computed_sample := [2, 15, 28, 41] in
   computed_sample.contains (2 + sampling_interval * 0) ∧ 
   computed_sample.contains (2 + sampling_interval * 1) ∧ 
   computed_sample.contains(2 + sampling_interval * 2) ∧ 
   computed_sample.contains (2 + sampling_interval * 3)
  ) := 
sorry

end fourth_student_in_sample_l377_377760


namespace ratio_AE_EQ_l377_377221

-- Define the points in a square ABCD
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 0 }
def C : Point := { x := 1, y := 1 }
def D : Point := { x := 0, y := 1 }

-- Define the point P which divides AB in the ratio 1:4
def P : Point := { x := 1 / 5, y := 0 }

-- Define the point Q which divides BC in the ratio 5:1
def Q : Point := { x := 1, y := 1 / 6 }

-- Define the intersection point E of lines DP and AQ
def E : Point := { x := 6 / 31, y := 1 / 31 }

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the distances AE and EQ
def AE : ℝ := distance A E
def EQ : ℝ := distance E Q

-- The theorem to prove the ratio AE : EQ = 6 : 29
theorem ratio_AE_EQ : AE / EQ = 6 / 29 := by
  sorry

end ratio_AE_EQ_l377_377221


namespace correct_statement_l377_377352

def ℕ := {x : ℕ | x ≥ 1} -- Natural numbers exclude 0 and negative numbers
def ℕ* := {x : ℕ | x ≥ 1} -- Positive integers
def ℚ := {x : ℝ | ∃ a b : ℤ, b ≠ 0 ∧ x = a / b} -- Rational numbers
def ℝ := {x : ℝ | true} -- Real numbers include all rationals and irrationals

theorem correct_statement :
  0 ∉ ℕ* :=
by sorry

end correct_statement_l377_377352


namespace positive_difference_of_probabilities_l377_377304

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377304


namespace sum_of_valid_n_l377_377869

theorem sum_of_valid_n (n : ℤ) (h1 : 0 < 5 * n) (h2 : 5 * n < 35) : 
  (∑ i in (Finset.filter (λ k : ℤ, 0 < k ∧ k < 7) (Finset.range 7)), i) = 21 := 
by
  sorry

end sum_of_valid_n_l377_377869


namespace quadrilateral_perimeter_proof_l377_377286

noncomputable def perimeter_quadrilateral (AB BC CD AD : ℝ) : ℝ :=
  AB + BC + CD + AD

theorem quadrilateral_perimeter_proof
  (AB BC CD AD : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 10)
  (h3 : CD = 6)
  (h4 : AB = AD)
  (h5 : AD = Real.sqrt 181)
  : perimeter_quadrilateral AB BC CD AD = 31 + Real.sqrt 181 := by
  unfold perimeter_quadrilateral
  rw [h1, h2, h3, h5]
  sorry

end quadrilateral_perimeter_proof_l377_377286


namespace count_of_squares_difference_l377_377524

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l377_377524


namespace problem1_problem2_problem3_l377_377888

-- Definition of the function f
def f (a x : ℝ) := a * exp (2 * x) + exp (x) + x

-- Definition of the derivative of f
def f' (a x : ℝ) := 2 * a * exp (2 * x) + exp (x) + 1

-- Problem 1: Prove that if f(x) has an extremum at x = 0, then a = -1
theorem problem1 (a : ℝ) (h : f' a 0 = 0) : a = -1 :=
sorry

-- Definition of the function g
def g (a x : ℝ) := f a x - (a + 3) * exp (x)

-- Definition of the derivative of g
def g' (a x : ℝ) := (a * exp (x) - 1) * (2 * exp (x) - 1)

-- Problem 2: Discuss the monotonicity of g(x)
theorem problem2 (a x : ℝ) : true :=
sorry

-- Problem 3: Prove that e^(x₁) + e^(x₂) > 1/2 given the condition
theorem problem3 (x₁ x₂ : ℝ) (h : 2 * exp(2 * x₁) + exp(x₁) + x₁ + 2 * exp(2 * x₂) + exp(x₂) + x₂ + 3 * exp(x₁) * exp(x₂) = 0) : exp(x₁) + exp(x₂) > 1 / 2 :=
sorry

end problem1_problem2_problem3_l377_377888


namespace mean_home_runs_l377_377249

-- Declaring the given conditions as variables
variables (n1 n2 n3 n4 n5 : ℕ)
variables (h1 h2 h3 h4 h5 : ℕ)

-- Assigning the given values from the problem conditions
def players_1 := 7
def runs_1 := 5

def players_2 := 5
def runs_2 := 6

def players_3 := 4
def runs_3 := 8

def players_4 := 2
def runs_4 := 9

def players_5 := 1
def runs_5 := 11

-- Using these variables to state our theorem
theorem mean_home_runs : 
  ( (players_1 * runs_1) + (players_2 * runs_2) + (players_3 * runs_3) + (players_4 * runs_4) + (players_5 * runs_5) )
  / 
  (players_1 + players_2 + players_3 + players_4 + players_5) = 126 / 19 :=
sorry

end mean_home_runs_l377_377249


namespace length_of_AX_l377_377946

theorem length_of_AX 
  (A B C X : Type) 
  (AB AC BC AX BX : ℕ) 
  (hx : AX + BX = AB)
  (h_angle_bisector : AC * BX = BC * AX)
  (h_AB : AB = 40)
  (h_BC : BC = 35)
  (h_AC : AC = 21) : 
  AX = 15 :=
by
  sorry

end length_of_AX_l377_377946


namespace positive_difference_prob_3_and_4_heads_l377_377323

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377323


namespace coin_flip_probability_difference_l377_377340

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377340


namespace phi_value_of_even_function_translated_left_by_pi_div_3_l377_377483

theorem phi_value_of_even_function_translated_left_by_pi_div_3 :
  ∀ (φ : ℝ), abs φ < (Real.pi / 2) →
    (∀ x : ℝ, sin (2 * (x + Real.pi / 3) + φ) = sin (2 * (-x + Real.pi / 3) + φ)) →
    φ = -(Real.pi / 6) := 
by
  sorry

end phi_value_of_even_function_translated_left_by_pi_div_3_l377_377483


namespace integer_diff_of_squares_1_to_2000_l377_377544

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377544


namespace diff_of_squares_1500_l377_377534

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377534


namespace number_of_pairs_divisible_by_5_l377_377052

theorem number_of_pairs_divisible_by_5 :
  let n := 1000
  let count := 200000
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n → (x^2 + y^2) % 5 = 0) →
  (∃ count : ℕ, count == 200000) :=
begin
  sorry
end

end number_of_pairs_divisible_by_5_l377_377052


namespace triangle_area_l377_377186

theorem triangle_area {PQ X Y Z : Point} (circle : Circle) (diameter : Real) 
  (hx : diameter = 2) (hy : circle.radius = 1)
  (on_arc_XY : X ∈ circle.arc PQ ∧ Y ∈ circle.arc PQ)
  (px : dist P X = 1 / 2) 
  (qy : dist Q Y = 3 / 4)
  (on_arc_Z : Z ∈ circle.arc_other PQ) 
  (angle_right : angle P Z Q = π / 2) : 
  area_triangle X Y Z = π^2 / 32 :=
sorry

end triangle_area_l377_377186


namespace problem_statement_l377_377619

open Complex

noncomputable def is_unit_circle (z : ℂ) : Prop := abs z = 1

theorem problem_statement (a b c : ℂ) (h1 : is_unit_circle a) 
  (h2 : is_unit_circle b) (h3 : is_unit_circle c)
  (h4 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = -1) :
  |a + b + c| = 1 ∨ |a + b + c| = 2 :=
sorry

end problem_statement_l377_377619


namespace factorize_expr_l377_377835

theorem factorize_expr (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := 
  sorry

end factorize_expr_l377_377835


namespace max_annual_profit_l377_377794

noncomputable def R (x : ℝ) : ℝ :=
  if x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

noncomputable def W (x : ℝ) : ℝ :=
  if x < 40 then -10 * x^2 + 600 * x - 260
  else -x + 9190 - 10000 / x

theorem max_annual_profit : ∃ x : ℝ, W 100 = 8990 :=
by {
  use 100,
  sorry
}

end max_annual_profit_l377_377794


namespace eval_expression_eq_one_l377_377039

theorem eval_expression_eq_one (x : ℝ) (hx1 : x^3 + 1 = (x+1)*(x^2 - x + 1)) (hx2 : x^3 - 1 = (x-1)*(x^2 + x + 1)) :
  ( ((x+1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * ((x-1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 ) = 1 :=
by
  sorry

end eval_expression_eq_one_l377_377039


namespace beta_value_l377_377474

theorem beta_value (α β : ℝ) 
  (h1 : sin α = (sqrt 5) / 5) 
  (h2 : sin (α - β) = - (sqrt 10) / 10) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) : 
  β = π / 4 :=
sorry

end beta_value_l377_377474


namespace positive_difference_prob_3_and_4_heads_l377_377287

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377287


namespace find_f_3_l377_377072

noncomputable def f (x : ℝ) : ℝ := 2^x * Real.log 2 x

theorem find_f_3 : f 3 = 0 := 
by
  sorry

end find_f_3_l377_377072


namespace polynomial_g_l377_377678

open Real

noncomputable def alpha : ℝ := (1 + sqrt 5) / 2

def f : ℕ → ℕ
| 0       := 0
| (n + 1) := n + 1 - f (f n)

theorem polynomial_g (g : ℕ → ℝ) (H : ∀ n, f n = floor (g n)) :
  g = λ n, (n + 1) / alpha := sorry

end polynomial_g_l377_377678


namespace solve_for_x_and_y_l377_377356

theorem solve_for_x_and_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 ∧ y = 5 :=
by
  sorry

end solve_for_x_and_y_l377_377356


namespace trigonometric_identity_l377_377475

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 8 :=
by 
  sorry

end trigonometric_identity_l377_377475


namespace projection_circle_diameter_eq_l377_377024

variables {A B C : Type} -- Vertices of triangle ABC
variables (triangle_ABC : Triangle A B C) -- A triangle constituted by vertices A, B, and C

-- External bisectors can be defined if required further in the reasoning
variables (bisector_ABC : ExternalBisector (angle triangle_ABC A B C))
variables (bisector_BAC : ExternalBisector (angle triangle_ABC B A C))
variables (bisector_ACB : ExternalBisector (angle triangle_ABC A C B))

-- Projections on external bisectors
variables (proj_A : Projection A bisector_ABC)
variables (proj_B : Projection B bisector_BAC)
variables (proj_C : Projection C bisector_ACB)

-- Diameter of the circumcircle formed by projections
noncomputable def circumcircle_diameter : ℝ :=
  diameter (circumcircle (foot_of_proj proj_A proj_B proj_C))

variables (r : ℝ) (p : ℝ) -- inradius and semiperimeter of triangle ABC

-- Goal: Prove r^2 + p^2 = d^2
theorem projection_circle_diameter_eq {d : ℝ} :
  circumcircle_diameter = d → r^2 + p^2 = d^2 := by
  sorry

end projection_circle_diameter_eq_l377_377024


namespace trigonometric_identity1_trigonometric_identity2_l377_377664

-- Statement 1
theorem trigonometric_identity1 (α : ℝ) : 
  (sin (α + π))^2 * cos (π + α) * cos (-α - 2 * π) /
  (tan (π + α) * (sin (π / 2 + α))^3 * sin (-α - 2 * π)) = 1 :=
by sorry

-- Statement 2
theorem trigonometric_identity2 : 
  (sqrt (1 + 2 * sin (20 * Real.pi / 180) * cos (160 * Real.pi / 180))) / 
  (sin (160 * Real.pi / 180) - sqrt (1 - (sin (20 * Real.pi / 180))^2)) = -1 :=
by sorry

end trigonometric_identity1_trigonometric_identity2_l377_377664


namespace count_of_valid_triplets_l377_377131

open Finset

noncomputable def orig_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def mean_of_remaining_set (s : Finset ℕ) : Prop :=
  (∑ x in s, (x : ℝ)) / (s.card : ℝ) = 5

def valid_triplets : Finset (Finset ℕ) :=
  orig_set.powerset.filter (λ s, s.card = 3 ∧ ∑ x in s, x = 15)

theorem count_of_valid_triplets : valid_triplets.card = 5 := sorry

end count_of_valid_triplets_l377_377131


namespace bee_fraction_remaining_l377_377762

theorem bee_fraction_remaining (N : ℕ) (L : ℕ) (D : ℕ) (hN : N = 80000) (hL : L = 1200) (hD : D = 50) :
  (N - (L * D)) / N = 1 / 4 :=
by
  sorry

end bee_fraction_remaining_l377_377762


namespace factorization_of_x12_sub_729_l377_377018

theorem factorization_of_x12_sub_729 (x : ℝ) :
  x^12 - 729 = (x^3 + 3) * (x - real.cbrt 3) * (x^2 + x * real.cbrt 3 + (real.cbrt 3)^2) * (x^12 + 9 * x^6 + 81) := 
sorry

end factorization_of_x12_sub_729_l377_377018


namespace airline_flights_l377_377583

theorem airline_flights (n : ℕ) (hn : n ≥ 5)
  (connected : ∀ u v : Fin n, u ≠ v → (∃ airline : Bool, (disjoint_cycle_length airline < 6 ∧ connected_by_airline u v airline))) 
  (no_short_cycle : ∀ airline : Bool, ¬(∃ cycle : List (Fin n), cycle.length < 6 ∧ is_cycle airline cycle)) :
  ∃ flights : list (Fin n × Fin n), flights.length < (n^2) / 3 :=
by sorry

end airline_flights_l377_377583


namespace proposition_correctness_l377_377402

-- Define conditions
def n : ℕ := 800
def n' : ℕ := 40
def systematic_sampling_interval (total sample_size : ℕ) : ℕ := total / sample_size

theorem proposition_correctness :
  systematic_sampling_interval n n' ≠ 40 ∧
  (∀ (x y : ℝ) (ŷ b a : ℝ), ŷ = b * x + a → (ŷ = b * x + a → true)) ∧
  (∀ (σ : ℝ), σ > 0 → ∀ (ξ : ℝ → ℝ) Π ξ, normal (2, σ^2) 
    → P (-∞ < ξ < 1) = 0.1 → P (2 < ξ < 3) = 0.4) ∧
  (∀ (E : set ℝ), ∀ (P_E : \Omega → Prop), measure_theory.measure_theory.measureP.P.reflect(E) = 0 → ¬ (P_E = ∅)) → 
  number of true propositions = 2 :=
by
  sorry

end proposition_correctness_l377_377402


namespace proof_correct_statement_B_l377_377353

def event_occur_every_time (times: ℕ) := ∀ n ∈ (finset.range times), event n = true

def certain_event_every_trial (times: ℕ) := ∀ n ∈ (finset.range times), event n = true

def event_not_occur_any (times: ℕ) := ∀ n ∈ (finset.range times), event n = false

def random_event := true -- Placeholder for actual definition

-- Definitions used for statements
def statement_A (times: ℕ) := event_occur_every_time times
def statement_B (times: ℕ) := certain_event_every_trial times
def statement_C (times: ℕ) := event_not_occur_any times
def statement_D := ∀ n, n > 6 → random_event

theorem proof_correct_statement_B (times: ℕ) : statement_B times = true := 
by {
  -- Proof is provided here.
  sorry
}

end proof_correct_statement_B_l377_377353


namespace inscribe_rectangle_in_triangle_l377_377947

variables {A B C P Q R S : Type} [EuclideanGeometry : EuclideanGeometry R2]

/-- 
To inscribe a rectangle PQRS in a triangle ABC such that 
vertices R and Q lie on sides AB and BC, respectively, 
vertices P and S lie on side AC, and the diagonal of the 
rectangle has a given length.
-/
theorem inscribe_rectangle_in_triangle
  {A B C P Q R S : Point}
  (hR_AB : R ∈ segment A B)
  (hQ_BC : Q ∈ segment B C)
  (hP_AC : P ∈ segment A C)
  (hS_AC : S ∈ segment A C)
  (diagonal_length : ℝ)
  (hDiagonal : dist P Q = diagonal_length ∨ dist P S = diagonal_length) :
  ∃ (P Q R S : Point), 
    R ∈ segment A B ∧ Q ∈ segment B C ∧
    P ∈ segment A C ∧ S ∈ segment A C ∧
    (dist P Q = diagonal_length ∨ dist P S = diagonal_length) :=
by
  sorry

end inscribe_rectangle_in_triangle_l377_377947


namespace positive_difference_prob_3_and_4_heads_l377_377293

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377293


namespace numerator_simplified_is_less_l377_377652

theorem numerator_simplified_is_less
  (a b n : ℕ)
  (h1 : a ≠ 1)
  (h2 : (a : ℚ) / b > 0)
  (h3 : 1 / (n : ℚ) > (a : ℚ) / b)
  (h4 : (a : ℚ) / b > 1 / (n + 1)) :
  let num := a * (n + 1) - b in num < a :=
by 
  let num := a * (n + 1) - b
  exact sorry

end numerator_simplified_is_less_l377_377652


namespace positive_difference_prob_3_and_4_heads_l377_377294

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377294


namespace polynomial_sum_zero_l377_377567

theorem polynomial_sum_zero :
  (A B C D : ℤ) (x : ℤ) (h : (x - 3) * (4 * x^2 + 2 * x - 6) = A * x^3 + B * x^2 + C * x + D) :
  A + B + C + D = 0 :=
by 
  sorry

end polynomial_sum_zero_l377_377567


namespace coin_flip_sequences_l377_377761

theorem coin_flip_sequences :
  let total_sequences := 2^10
  let sequences_starting_with_two_heads := 2^8
  total_sequences - sequences_starting_with_two_heads = 768 :=
by
  sorry

end coin_flip_sequences_l377_377761


namespace probability_is_7_over_20_l377_377384

/-- Original sets of m and n -/
def m_set := {11, 13, 15, 17, 19}
def n_set := {1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
              2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018}

/-- Reduced units digits sets -/
def reduced_m_set := {1, 3, 5, 7, 9}

/-- Conditions for n to get units digit 1 in various cases -/
def units_digit_1 (m n : ℕ) : ℕ :=
  match m, n with
  | 1, _ => 1
  | 3, _ => if (n % 4) = 0 then 1 else 0
  | 5, _ => 0
  | 7, _ => if (n % 4) = 0 then 1 else 0
  | 9, _ => if (n % 2) = 0 then 1 else 0
  | _, _ => 0

/-- Calculation of probability -/
noncomputable def probability_units_digit_1 : ℚ :=
  (1/5 : ℚ) * (20/20 : ℚ) + /-- 1-Case --/
  (1/5 : ℚ) * (5/20 : ℚ) + /-- 3-Case --/
  (1/5 : ℚ) * (0/20 : ℚ) + /-- 5-Case --/
  (1/5 : ℚ) * (5/20 : ℚ) + /-- 7-Case --/
  (1/5 : ℚ) * (10/20 : ℚ) /-- 9-Case --/

/-- Theorem stating the final probability is 7/20 -/
theorem probability_is_7_over_20 : probability_units_digit_1 = (7/20 : ℚ) := 
by
  sorry

end probability_is_7_over_20_l377_377384


namespace value_of_a_l377_377897

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

-- The main theorem statement
theorem value_of_a (a : ℝ) (H : B a ⊆ A) : a = 0 ∨ a = 1 ∨ a = -1 :=
by 
  sorry

end value_of_a_l377_377897


namespace positive_difference_prob_3_and_4_heads_l377_377320

theorem positive_difference_prob_3_and_4_heads :
  let p_3 := (choose 4 3) * (1 / 2) ^ 3 * (1 / 2) in
  let p_4 := (1 / 2) ^ 4 in
  abs (p_3 - p_4) = 7 / 16 :=
by
  -- Definitions for binomial coefficient and probabilities
  let p_3 := (Nat.choose 4 3) * (1 / 2)^3 * (1 / 2)
  let p_4 := (1 / 2)^4
  -- The difference between probabilities
  let diff := p_3 - p_4
  -- The desired equality to prove
  show abs diff = 7 / 16
  sorry

end positive_difference_prob_3_and_4_heads_l377_377320


namespace find_nine_prime_factors_with_difference_l377_377967

theorem find_nine_prime_factors_with_difference :
  ∃ p1 p2 p3 p4 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ 
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
  (let n := p1 * p2 * p3 * p4 in 
    n < 2001 ∧ 
    ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16 : ℕ), 
    d1 = 1 ∧ 
    d16 = n ∧ 
    d9 - d8 = 22 ∧ 
    n = 1995) :=
sorry

end find_nine_prime_factors_with_difference_l377_377967


namespace pierre_ate_correct_l377_377759

-- Define the conditions
def cake_weight : ℝ := 546
def cake_parts : ℝ := 12
def nathalie_fraction : ℝ := 1 / cake_parts
def pierre_multiplier : ℝ := 2.5

-- Define the quantities based on the conditions
def nathalie_ate : ℝ := cake_weight * nathalie_fraction
def pierre_ate : ℝ := nathalie_ate * pierre_multiplier

-- State the theorem to be proved
theorem pierre_ate_correct :
  pierre_ate = 113.75 :=
by
  -- The steps of the solution are omitted here, hence the use of sorry
  sorry

end pierre_ate_correct_l377_377759


namespace roots_of_composition_l377_377417

noncomputable def f (x : ℝ) : ℝ := 1 + 2 / x

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  nat.iterate f n x

theorem roots_of_composition :
  (f_iter 10 (-1) = -1) ∧ (f_iter 10 2 = 2) :=
by
  sorry

end roots_of_composition_l377_377417


namespace bronze_balls_balance_l377_377006

theorem bronze_balls_balance
  (a : Fin 10 → ℝ) -- masses of the 10 iron weights
  (m : Fin 10 → ℝ) -- masses of the bronze balls
  (h : ∀ i : Fin 10, m i = |a (Fin.mod (i + 1) 10) - a i|) -- mass of each bronze ball
  : ∃ (s : Fin 10 → ℤ), (∑ i, s i * m i = 0) ∧ (∑ i, s i = 0) :=
  sorry

end bronze_balls_balance_l377_377006


namespace count_ways_to_replace_digits_divisible_by_45_l377_377599

theorem count_ways_to_replace_digits_divisible_by_45 : 
  let a := 2
  let b := 0
  let c := 1
  let d := 6
  let e := 0
  -- Each of the digits x1, x2, x3, x4, x5 can be between 0 and 9
  ∃ (x₁ x₂ x₃ x₄ : ℕ), x₁ ∈ finset.range 10 ∧ x₂ ∈ finset.range 10 ∧ x₃ ∈ finset.range 10 ∧ x₄ ∈ finset.range 10 
  -- The fifth digit x5 will be chosen so that the sum is divisible by 9. 
  let xi_sum := x₁ + x₂ + x₃ + x₄ 
  let x₅ := (9 - (a + b + c + d + e + xi_sum) % 9) % 9 
  in (x₅ ∈ finset.range 10) 
     ∧ ((a * 10 ^ 9 + x₁ * 10 ^ 8 + b * 10 ^ 7 + x₂ * 10 ^ 6 + c * 10 ^ 5 + x₃ * 10 ^ 4 + d * 10 ^ 3 
        + x₄ * 10 ^ 2 + e * 10 + x₅) % 45 = 0) 
     ∧ (2 * 9 ^ 4 = 13122) := sorry

end count_ways_to_replace_digits_divisible_by_45_l377_377599


namespace max_det_A_l377_377972

noncomputable def v : ℝ^3 := ![3, -2, 4]
noncomputable def w : ℝ^3 := ![-1, 5, 2]
noncomputable def u (direction : ℝ^3) : ℝ^3 := direction / ∥direction∥ -- assume u is in the direction of given vector and it's unit length

theorem max_det_A : 
  let v := v
  let w := w
  let u := u (v × w) -- u in the direction of cross product of v and w 
  let A := matrix.of_vec [u, v, w]
  det A = ∥v × w∥ := by
  sorry

#eval max_det_A

end max_det_A_l377_377972


namespace circle_tangent_radius_l377_377716

noncomputable def R : ℝ := 4
noncomputable def r : ℝ := 3
noncomputable def O1O2 : ℝ := R + r
noncomputable def r_inscribed : ℝ := (R * r) / O1O2

theorem circle_tangent_radius :
  r_inscribed = (24 : ℝ) / 7 :=
by
  -- The proof would go here
  sorry

end circle_tangent_radius_l377_377716


namespace pick_theorem_l377_377701

def lattice_polygon (vertices : List (ℤ × ℤ)) : Prop :=
  ∀ v ∈ vertices, ∃ i j : ℤ, v = (i, j)

variables {n m : ℕ}
variables {A : ℤ}
variables {vertices : List (ℤ × ℤ)}

def lattice_point_count_inside (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count inside points
  sorry

def lattice_point_count_boundary (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count boundary points
  sorry

theorem pick_theorem (h : lattice_polygon vertices) :
  lattice_point_count_inside vertices = n → 
  lattice_point_count_boundary vertices = m → 
  A = n + m / 2 - 1 :=
sorry

end pick_theorem_l377_377701


namespace log_eq_implies_root_l377_377030

theorem log_eq_implies_root (x : ℝ) (h : 4 * log 3 x = log 3 (4 * x)) : x = real.cbrt 4 := sorry

end log_eq_implies_root_l377_377030


namespace solution_set_of_equation_l377_377985

theorem solution_set_of_equation (x : ℝ) :
  (|1 - x| + |2 * x - 1| = |3 * x - 2|) ↔ (x ∈ Iic (1 / 2) ∨ x ∈ Ici 1) :=
sorry

end solution_set_of_equation_l377_377985


namespace factorial_inequality_l377_377227

theorem factorial_inequality (n : ℕ) (h : n > 1) : n! < ( (n + 1) / 2 )^n := by
  sorry

end factorial_inequality_l377_377227


namespace max_volume_height_l377_377076

-- Define the conditions
def on_same_sphere (r h : ℝ) : Prop := (1:ℝ)^2 = (h - 1)^2 + r^2

-- Define the volume function for the cone
def cone_volume (h : ℝ) : ℝ := (π / 3) * h * (2 * h - h^2)

-- The main theorem
theorem max_volume_height (h : ℝ) (r : ℝ) (h_nonneg : 0 ≤ h) (h_le_2 : h ≤ 2) (r_nonneg : 0 ≤ r) 
  (sphere_condition : on_same_sphere r h) :
  h = 4 / 3 :=
by
  sorry

end max_volume_height_l377_377076


namespace impossibility_of_cubical_wireframe_construction_l377_377830

-- Conditions as definitions in Lean 4
def is_P_shaped_piece (piece : Type) : Prop :=
  ∃ a b c : ℕ, a + b + c = 3 -- Definition of П-shaped part with three unit segments.

def points_27 (p : Type) : Prop :=
  p.card = 27 -- The wireframe consists of 27 points.

def neighbors_connected_by_one_segment (p : Type) : Prop :=
  ∀ a b : p, neighbor a b → ∃ s : Segment, connects s a b -- Any two neighboring points are connected by one segment.

-- Proof goal
theorem impossibility_of_cubical_wireframe_construction (p : Type) (pieces : Type)
  [is_P_shaped_piece pieces] [points_27 p] [neighbors_connected_by_one_segment p] : false :=
sorry

end impossibility_of_cubical_wireframe_construction_l377_377830


namespace area_of_triangle_roots_of_cubic_eq_l377_377677

def cubic_eq (x : ℝ) : Prop := x^3 - 4 * x^2 + 5 * x - 19/10 = 0

noncomputable def semi_perimeter (r s t : ℝ) : ℝ := (r + s + t) / 2

noncomputable def area_triangle (r s t : ℝ) : ℝ :=
  let p := semi_perimeter r s t in
  √(p * (p - r) * (p - s) * (p - t))

theorem area_of_triangle_roots_of_cubic_eq (r s t : ℝ) (hr : cubic_eq r) (hs : cubic_eq s) (ht : cubic_eq t) :
  area_triangle r s t = √5 / 5 := by
  sorry

end area_of_triangle_roots_of_cubic_eq_l377_377677


namespace maximum_volume_of_figure_formed_by_5_vertices_of_unit_cube_l377_377219

-- Definition of the unit cube with vertices A, B, C, D, E, F, G, H
def unit_cube_vertices : set (ℝ × ℝ × ℝ) := 
  { (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), 
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1) }

-- Function to calculate volume of a tetrahedron given its vertices
noncomputable def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 6) * abs (det3 (matrix3 a b c d))

-- The theorem statement
theorem maximum_volume_of_figure_formed_by_5_vertices_of_unit_cube : 
  ∀ (vertices : set (ℝ × ℝ × ℝ)), vertices ⊆ unit_cube_vertices → vertices.card = 5 → volume_of(vertices) = 1 / 2 :=
by
  intros vertices h_subset h_card
  sorry -- Proof to be completed

end maximum_volume_of_figure_formed_by_5_vertices_of_unit_cube_l377_377219


namespace ellipse_properties_slope_range_l377_377103

def standard_equation_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
 (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : Prop :=
 c / a = 1 / 2

def passes_through (x y : ℝ) (px py : ℝ) : Prop :=
 x = px ∧ y = py

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
 if x2 - x1 ≠ 0 then (y2 - y1) / (x2 - x1) else 0

theorem ellipse_properties 
  (a b c : ℝ)
  (h1 : 0 < b ∧ b < a)
  (h2 : eccentricity a c)
  (h3 : passes_through (-1) (3 / 2) (-1) (3 / 2))
  (h4 : a^2 = b^2 + c^2) :
  ∃ (a : ℝ) (b : ℝ), standard_equation_ellipse a b x y ∧ (a = 2 ∧ b = sqrt 3 ∧ c = 1) :=
sorry

theorem slope_range 
  (a b c m : ℝ)
  (x y : ℝ)
  (h1 : 0 < b ∧ b < a)
  (h2 : eccentricity a c)
  (h3 : passes_through (-1) (3 / 2) (-1) (3 / 2))
  (h4 : a^2 = b^2 + c^2)
  (h5 : standard_equation_ellipse a b x y)
  (h6 : ∀ P Q : ℝ × ℝ, ∃ R : ℝ × ℝ, slope (fst P) (snd P) (fst Q) (snd Q) = m → (fst P + fst Q) / 2 = fst R ∧ (snd P + snd Q) / 2 = snd R)
  (h7 : 4 * abs m + 4 / abs m ≥ 8) :
  0 ≤ slope 0 0 1 1 ∧ slope 0 0 1 1 ≤ 1 / 8 :=
sorry

end ellipse_properties_slope_range_l377_377103


namespace range_of_a_l377_377119

theorem range_of_a (a : ℝ) : (1 ∉ {x : ℝ | (x - a) / (x + a) < 0}) → ( -1 ≤ a ∧ a ≤ 1 ) := 
by
  intro h
  sorry

end range_of_a_l377_377119


namespace length_of_bridge_l377_377681

/--  Train length (in meters) -/
def train_length : ℝ := 90

/--  Train speed (in km/hr) -/
def train_speed_kmph : ℝ := 45

/--  Time taken by train to cross the bridge (in seconds) -/
def crossing_time : ℝ := 30

/-- Function to convert speed from km/hr to m/s -/
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

theorem length_of_bridge : 
  let train_speed_mps := kmph_to_mps train_speed_kmph
  let distance_covered := train_speed_mps * crossing_time
  distance_covered - train_length = 285 :=
by
  -- Proof skipped
  sorry

end length_of_bridge_l377_377681


namespace fraction_inequality_fraction_inequality_equality_case_l377_377469

variables {α β a b : ℝ}

theorem fraction_inequality 
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b) ≤ (β / α + α / β) :=
sorry

-- Additional equality statement
theorem fraction_inequality_equality_case
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b = β / α + α / β) ↔ (a = α ∧ b = β ∨ a = β ∧ b = α) :=
sorry

end fraction_inequality_fraction_inequality_equality_case_l377_377469


namespace find_d_plus_q_l377_377973

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + d * (n * (n - 1) / 2)

noncomputable def sum_geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁
  else b₁ * (q ^ n - 1) / (q - 1)

noncomputable def sum_combined_sequence (a₁ d b₁ q : ℝ) (n : ℕ) : ℝ :=
  sum_arithmetic_sequence a₁ d n + sum_geometric_sequence b₁ q n

theorem find_d_plus_q (a₁ d b₁ q : ℝ) (h_seq: ∀ n : ℕ, 0 < n → sum_combined_sequence a₁ d b₁ q n = n^2 - n + 2^n - 1) :
  d + q = 4 :=
  sorry

end find_d_plus_q_l377_377973


namespace area_of_triangle_BXC_l377_377272

/-- Definition of a trapezoid with certain bases and area, and finding the area of a specific triangle formed by diagonals intersecting. -/
theorem area_of_triangle_BXC 
  (AB CD : ℝ) 
  (h : ℝ) 
  (area_trapezoid : ℝ)
  (H_area_trapezoid : area_trapezoid = 300) 
  (H_AB : AB = 20) 
  (H_CD : CD = 30) 
  (H_h : h = 12) :
  ∃ area_BXC : ℝ, area_BXC = 72 := 
by 
  -- given conditions
  have Hheight : (300 : ℝ) = 0.5 * h * (AB + CD), from calc
    300 = 0.5 * h * (20 + 30) : 
      by rw [H_AB, H_CD, H_h],
  sorry -- proof needed

end area_of_triangle_BXC_l377_377272


namespace parabola_directrix_distance_l377_377913

theorem parabola_directrix_distance (m : ℝ) (h : |1 / (4 * m)| = 2) : m = 1/8 ∨ m = -1/8 :=
by { sorry }

end parabola_directrix_distance_l377_377913


namespace real_roots_exist_for_nonzero_K_l377_377065

theorem real_roots_exist_for_nonzero_K (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by
  sorry

end real_roots_exist_for_nonzero_K_l377_377065


namespace good_set_representation_l377_377728

-- Definition of a good set
def is_good_set (s : set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ s → b ∈ s → a ≤ b → b % a = 0

-- Main theorem statement
theorem good_set_representation (n : ℕ) :
  ∃ (x : fin n → ℕ), is_good_set (set.range x) ∧ (∑ i, (i + 1) * x i) = (n + 1)! - 1 ∧ 
  card {x | is_good_set (set.range x) ∧ (∑ i, (i + 1) * x i) = (n + 1)! - 1} ≥ n! :=
by {
  sorry
}

end good_set_representation_l377_377728


namespace positive_difference_prob_3_and_4_heads_l377_377289

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_prob_3_and_4_heads (p : ℚ) (hp : p = 1/2) (n : ℕ) (hn : n = 4) :
  let p1 := binomial_prob n 3 p in
  let p2 := binomial_prob n 4 p in
  p1 - p2 = 3/16 :=
by
  sorry

end positive_difference_prob_3_and_4_heads_l377_377289


namespace max_profit_at_l377_377668

variables (k x : ℝ) (hk : k > 0)

-- Define the quantities based on problem conditions
def profit (k x : ℝ) : ℝ :=
  0.072 * k * x ^ 2 - k * x ^ 3

-- State the theorem
theorem max_profit_at (k : ℝ) (hk : k > 0) : 
  ∃ x, profit k x = 0.072 * k * x ^ 2 - k * x ^ 3 ∧ x = 0.048 :=
sorry

end max_profit_at_l377_377668


namespace beef_weight_after_processing_l377_377783

theorem beef_weight_after_processing :
  ∀ (W_before W_after : ℝ), 
  W_before = 714.2857142857143 → 
  W_after = 0.70 * W_before → 
  W_after ≈ 500 := 
by 
  intros W_before W_after h_before h_after
  sorry

end beef_weight_after_processing_l377_377783


namespace Ravi_jumps_39_inches_l377_377649

-- Define the heights of the next three highest jumpers
def h₁ : ℝ := 23
def h₂ : ℝ := 27
def h₃ : ℝ := 28

-- Define the average height of the three jumpers
def average_height : ℝ := (h₁ + h₂ + h₃) / 3

-- Define Ravi's jump height
def Ravi_jump_height : ℝ := 1.5 * average_height

-- The theorem to prove
theorem Ravi_jumps_39_inches : Ravi_jump_height = 39 := by
  sorry
 
end Ravi_jumps_39_inches_l377_377649


namespace tangent_line_at_one_extreme_points_of_F_l377_377975

-- Define the function f
def f (x : ℝ) : ℝ := Real.exp x * Real.log x

-- Define the function g as the tangent line at x = 1
def g (x : ℝ) : ℝ := Real.exp 1 * (x - 1)

-- Define the function F as the difference between f and g
def F (x : ℝ) : ℝ := f x - g x

-- Statement 1: Prove that the tangent line to y = f(x) at x = 1 is g(x) = e(x - 1)
theorem tangent_line_at_one (x : ℝ) : f 1 = g x := sorry

-- Statement 2: Prove that the function y = f(x) - g(x) has exactly 2 extreme points
theorem extreme_points_of_F : ∃! x : ℝ, ∃! y : ℝ, F' x = 0 ∧ F'' y = 0 := sorry

end tangent_line_at_one_extreme_points_of_F_l377_377975


namespace probability_complement_given_A_l377_377467

theorem probability_complement_given_A :
  (∀ (A B : Type) [MeasureTheory.ProbabilityMeasure A] [MeasureTheory.ProbabilityMeasure B],
  let PA := MeasureTheory.Measure.probability
  let PB := MeasureTheory.Measure.probability
  let PAB := PA * PB in
  PA = 1/3 → PB = 1/4 → PA (B | A) = 3/4 →
  PA (¬ B | A) = 7/16) :=
by
  intros A B _ _ PA PB PAB hPA hPB hPA_B
  sorry

end probability_complement_given_A_l377_377467


namespace sum_possible_B_values_l377_377385

theorem sum_possible_B_values : 
  let sum_digits := 8 + 5 + 2 + 7 + 2 + 9 + 5 in
  ∃ (B : ℕ), B < 10 ∧ (sum_digits + B) % 9 = 0 → B = 1 ∨ B = 4 ∨ B = 7 ∧ 1 + 4 + 7 = 12 :=
by
  sorry

end sum_possible_B_values_l377_377385


namespace part1_part2_l377_377944

-- Defining the basic setup for the triangle and conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def condition1 (A B : ℝ) (a b : ℝ) : Prop :=
  b * sin A = sqrt 3 * a * cos B

def condition2 (A C : ℝ) : Prop :=
  sin C = 2 * sin A

-- Part 1: Prove measure of angle B
theorem part1 {A B : ℝ} {a b : ℝ} (h1 : condition1 A B a b) : B = π / 3 :=
sorry

-- Part 2: Prove the values of a and c
theorem part2 {A B C : ℝ} {a b c : ℝ} (h1 : condition1 A B a b) (h2 : condition2 A C) (hb : b = 3) : 
  a = sqrt 3 ∧ c = 2 * sqrt 3 :=
sorry

end part1_part2_l377_377944


namespace inequality_solution_exists_l377_377914

theorem inequality_solution_exists (x m : ℝ) (h1: 1 < x) (h2: x ≤ 2) (h3: x > m) : m < 2 :=
sorry

end inequality_solution_exists_l377_377914


namespace max_sum_of_digits_l377_377395

theorem max_sum_of_digits (a b c : ℕ) (x : ℕ) (N : ℕ) :
  N = 100 * a + 10 * b + c →
  100 <= N →
  N < 1000 →
  a ≠ 0 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1730 + x →
  a + b + c = 20 :=
by
  intros hN hN_ge_100 hN_lt_1000 ha_ne_0 hsum
  sorry

end max_sum_of_digits_l377_377395


namespace emily_weight_l377_377505

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end emily_weight_l377_377505


namespace arithmetic_sequences_count_l377_377990

noncomputable def countArithmeticSequences (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4

theorem arithmetic_sequences_count :
  ∀ n : ℕ, countArithmeticSequences n = if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4 :=
by sorry

end arithmetic_sequences_count_l377_377990


namespace find_x_l377_377572

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) : hash x 7 = 63 → x = 3 :=
by
  sorry

end find_x_l377_377572


namespace number_of_integers_as_difference_of_squares_l377_377531

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377531


namespace product_of_variables_l377_377741

variables (a b c d : ℚ)

theorem product_of_variables :
  4 * a + 5 * b + 7 * c + 9 * d = 82 →
  d + c = 2 * b →
  2 * b + 2 * c = 3 * a →
  c - 2 = d →
  a * b * c * d = 276264960 / 14747943 := by
  sorry

end product_of_variables_l377_377741


namespace grade_assignment_ways_l377_377394

theorem grade_assignment_ways :
  let students := 12
  let grades := 4
  grades ^ students = 16777216 :=
by
  let students := 12
  let grades := 4
  have h : grades ^ students = 16777216 := rfl
  exact h

end grade_assignment_ways_l377_377394


namespace problem_solution_l377_377482

noncomputable def circle_eq (x y : ℝ) := (x - 2)^2 + y^2 = 1
def point_P := (3, 4 : ℝ)

-- Define the dot product
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  let PAx := (A.1 - P.1) in
  let PAy := (A.2 - P.2) in
  let PBx := (B.1 - P.1) in
  let PBy := (B.2 - P.2) in
  PAx * PBx + PAy * PBy

theorem problem_solution :
  ∀ (A B : ℝ × ℝ), circle_eq A.1 A.2 → circle_eq B.1 B.2 → dot_product point_P A B = 16 :=
by {
  intros A B hA hB,
  sorry
}

end problem_solution_l377_377482


namespace length_of_BD_l377_377937

noncomputable def BD_length (AC AD DE : ℝ) : ℝ :=
  let DC := AC - AD
  let DE_plus_DC := DE + DC
  real.sqrt (AD * DE_plus_DC)

theorem length_of_BD :
  ∃ (BD : ℝ), with_conditions (angle_ABC angle_ADB AC AD DE : ℝ),
  BD = 3 * real.sqrt 10
  :=
by 
{
  let AC := 20,
  let AD := 5,
  let DE := 3,
  let BD := BD_length AC AD DE,
  use BD,
  sorry
}

end length_of_BD_l377_377937


namespace constant_distance_from_origin_l377_377104

open Real

-- Definitions and Conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def eccentricity (a c : ℝ) : ℝ := 
  c / a

def triangle_area (s1 s2 : ℝ) : ℝ := 
  0.5 * s1 * s2

def line (y k x : ℝ) : Prop := 
  y = k * x

-- Problem Statement
theorem constant_distance_from_origin (a b c : ℝ) (ecc : eccentricity a c) 
  (h₀ : is_ellipse a b) (P F₁ F₂ M N : ℝ × ℝ) :
  ecc = 1 / 2 →
  (let area := triangle_area (abs (P.1 - F₁.1)) (abs (P.2 - F₂.2)) in area = 3) →
  (let line_l := λ y x, line y (2 * sqrt 3) x; N.2 = 2 * sqrt 3) →
  (O : ℝ × ℝ := (0, 0)) →
  (M ∈ E) →
  (N ∈ line_l) →
  (OM ⊥ ON) →
  distance (O) (line_through_points M N) = sqrt 3 :=
sorry


end constant_distance_from_origin_l377_377104


namespace cone_volume_approx_max_prism_volume_approx_l377_377392

-- Part 1: Volume of the cone
theorem cone_volume_approx :
  ∀ (l r : ℝ), 
  (1/4 * 2 * real.pi * l = 2 * real.pi * r) → 
  (l + r + real.sqrt 2 * r = real.sqrt 2) →
  let h := real.sqrt 15 * r in 
  abs ((1/3 * real.pi * r^2 * h) - 0.0435) < 0.0001 :=
begin
  intros l r h1 h2,
  sorry
end

-- Part 2: Maximum volume of the rectangular prism
theorem max_prism_volume_approx :
  ∀ (a b h : ℝ), 
  (a + h = 1) →
  (2 * b + 2 * h = 1) →
  (0 < h ∧ h < 0.5) → 
  let V := a * b * h in 
  abs (V - 0.0481) < 0.0001 :=
begin
  intros a b h h1 h2 h3,
  sorry
end

end cone_volume_approx_max_prism_volume_approx_l377_377392


namespace gcd_459_357_l377_377725

-- Define the numbers involved
def num1 := 459
def num2 := 357

-- State the proof problem
theorem gcd_459_357 : Int.gcd num1 num2 = 51 := by
  sorry

end gcd_459_357_l377_377725


namespace diff_of_squares_count_l377_377513

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377513


namespace boxes_to_fill_l377_377849

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) (h₁ : total_boxes = 25.75) (h₂ : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := 
by
  sorry

end boxes_to_fill_l377_377849


namespace relationship_l377_377573

variables {A : Type} {b : set A} {β : set (set A)}

-- Conditions
def A_on_b (A : A) (b : set A) : Prop := A ∈ b
def b_in_beta (b : set A) (β : set (set A)) : Prop := b ∈ β

-- Theorem statement
theorem relationship (A : A) (b : set A) (β : set (set A)) :
  A_on_b A b → b_in_beta b β → (A ∈ b ∧ b ∈ β) :=
by sorry

end relationship_l377_377573


namespace sum_of_intervals_length_l377_377233

noncomputable def f (x : ℝ) : ℝ := ∑ n in finset.range(70), n / (x - n)

theorem sum_of_intervals_length : 
  ∑ i in finset.range(70), (classical.some (exists_unique (λ x, f x = (5/4))).1 - i) = 1988 := 
sorry

end sum_of_intervals_length_l377_377233


namespace positive_difference_of_probabilities_l377_377307

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377307


namespace coin_probability_difference_l377_377333

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377333


namespace part_a_part_b_l377_377197

def balanced (V : Finset (ℝ × ℝ)) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ V → B ∈ V → A ≠ B → ∃ C : ℝ × ℝ, C ∈ V ∧ (dist C A = dist C B)

def center_free (V : Finset (ℝ × ℝ)) : Prop :=
  ¬ ∃ (A B C P : ℝ × ℝ), A ∈ V → B ∈ V → C ∈ V → P ∈ V →
                         A ≠ B ∧ B ≠ C ∧ A ≠ C →
                         (dist P A = dist P B ∧ dist P B = dist P C)

theorem part_a (n : ℕ) (hn : 3 ≤ n) :
  ∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V :=
by sorry

theorem part_b : ∀ n : ℕ, 3 ≤ n →
  (∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V ∧ center_free V ↔ n % 2 = 1) :=
by sorry

end part_a_part_b_l377_377197


namespace largest_of_five_consecutive_odd_integers_with_product_93555_l377_377447

theorem largest_of_five_consecutive_odd_integers_with_product_93555 : 
  ∃ n, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) ∧ (n + 8 = 19) :=
sorry

end largest_of_five_consecutive_odd_integers_with_product_93555_l377_377447


namespace bulb_power_usage_daily_l377_377375

variables (W : ℝ)
variables (num_bulbs : ℝ) (cost_per_watt : ℝ) (total_cost : ℝ) (days_in_june : ℝ)

-- Conditions from part a)
def conditions :=
  num_bulbs = 40 ∧ 
  cost_per_watt = 0.20 ∧ 
  total_cost = 14400 ∧ 
  days_in_june = 30

-- Question translated to Lean statement
theorem bulb_power_usage_daily (h : conditions) : W = 60 :=
begin
  sorry
end

end bulb_power_usage_daily_l377_377375


namespace positive_difference_probability_l377_377298

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377298


namespace matilda_father_chocolates_left_l377_377208

-- definitions for each condition
def initial_chocolates : ℕ := 20
def persons : ℕ := 5
def chocolates_per_person := initial_chocolates / persons
def half_chocolates_per_person := chocolates_per_person / 2
def total_given_to_father := half_chocolates_per_person * persons
def chocolates_given_to_mother := 3
def chocolates_eaten_by_father := 2

-- statement to prove
theorem matilda_father_chocolates_left :
  total_given_to_father - chocolates_given_to_mother - chocolates_eaten_by_father = 5 :=
by
  sorry

end matilda_father_chocolates_left_l377_377208


namespace problem1_problem2_l377_377040

-- Problem 1
theorem problem1 :
  sin (-1 * 200 * (π / 180)) * cos (1 * 290 * (π / 180)) + cos (-1 * 020 * (π / 180)) * sin (-1 * 050 * (π / 180)) = 1 :=
by
  sorry

-- Problem 2
def f (α : ℝ) : ℝ := 
  (2 * sin (π + α) * cos (3 * π - α) + cos (4 * π - α)) / 
  (1 + sin (α) ^ 2 + cos ((3 * π / 2) + α) - sin (π / 2 + α) ^ 2)

theorem problem2 (h : 1 + 2 * sin (-23 * π / 6) ^ 2 ≠ 0) :
  f (-23 * π / 6) = sqrt 3 :=
by
  sorry

end problem1_problem2_l377_377040


namespace positive_difference_probability_l377_377301

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377301


namespace probability_last_flip_heads_l377_377611

/-- Justine has a biased coin with specific properties. Given that the coin will come up the same as 
the last flip 2/3 of the time and the other side 1/3 of the time, and given that the coin initially 
lands heads, what is the probability that the last flip out of 2010 flips is heads? Prove that -/
theorem probability_last_flip_heads :
  (∑ i in Finset.range 1006, (Nat.choose 2010 (2 * i) : ℝ) * ((1 / 3) ^ (2 * i)) * ((2 / 3) ^ (2010 - 2 * i))
   = ∑ i in Finset.range 1006, (Nat.choose 2010 (2 * i + 1) : ℝ) * ((1 / 3) ^ (2 * i + 1)) * ((2 / 3) ^ (2010 - (2 * i + 1)))) →
  (∑ i in Finset.range 1006, (Nat.choose 2010 (2 * i) : ℝ) * ((1 / 3) ^ (2 * i)) * ((2 / 3) ^ (2010 - 2 * i)) =
   (3^2010 + 1) / (2 * 3^2010)) := 
begin
  sorry
end

end probability_last_flip_heads_l377_377611


namespace permutation_count_l377_377614

theorem permutation_count : 
  (∃ (b : Fin 14 → ℕ), 
    (∀ i, i ∈ Finset.univ → b i ∈ (Finset.range 14)) ∧
    Finset.univ.image b = Finset.univ.image Fin.val ∧
    b 0 > b 1 ∧ b 1 > b 2 ∧ b 2 > b 3 ∧
    b 3 > b 4 ∧ b 4 > b 5 ∧ b 5 > b 6 ∧
    b 6 < b 7 ∧ b 7 < b 8 ∧ b 8 < b 9 ∧
    b 9 < b 10 ∧ b 10 < b 11 ∧ b 11 < b 12 ∧ 
    b 12 < b 13) ↔ 1716 := 
sorry

end permutation_count_l377_377614


namespace inequality_holds_l377_377196

noncomputable theory

def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem inequality_holds (q : ℕ) (hq : ¬ ∃ k : ℕ, q = k ^ 3) :
  ∃ c : ℝ, c = (13 * ↑q^(4 / 3))⁻¹ ∧ ∀ n : ℕ, 0 < n → 
    fractional_part(n * ↑q^(1 / 3)) + fractional_part(n * ↑q^(2 / 3)) ≥ c * n^(-1 / 2) :=
sorry

end inequality_holds_l377_377196


namespace probability_of_even_adjacent_is_0_25_l377_377726

open Finset

def even_digit_five_digit_numbers : Finset (Finset ℕ) :=
  (finset.range 5).powerset.filter (λ s, s.card = 5 ∧
    (s.filter (λ n, n % 2 = 0)).nonempty ∧
    (s.contains 1 → s.contains 2))

noncomputable def total_five_digit_numbers :=
  (finset.range 5).powerset.filter (λ s, s.card = 5)

noncomputable def probability_even_adjacent :=
  (even_digit_five_digit_numbers.card : ℝ) / (total_five_digit_numbers.card : ℝ)

theorem probability_of_even_adjacent_is_0_25 :
  probability_even_adjacent = 0.25 :=
begin
  sorry
end

end probability_of_even_adjacent_is_0_25_l377_377726


namespace find_ratio_l377_377815

/- Let A, B, C, D be the vertices of a 1x1 square grid: -/
def A := (0, 1)
def B := (0, 0)
def C := (1, 0)
def D := (1, 1)

/- Let E be the midpoint of segment CD -/
def E := (1, 0.5)

/- Let F be the point on segment BC satisfying BF = 2CF -/
def F := (2 / 3, 0)

/- Let P be the intersection of lines AF and BE -/
-- Function to find the intersection of two lines defined by points
noncomputable def line_intersection (p1 p2 p3 p4 : ℝ × ℝ) : (ℝ × ℝ) :=
let a1 := p2.2 - p1.2 in
let b1 := p1.1 - p2.1 in
let c1 := a1 * p1.1 + b1 * p1.2 in
let a2 := p4.2 - p3.2 in
let b2 := p3.1 - p4.1 in
let c2 := a2 * p3.1 + b2 * p3.2 in
let det := a1 * b2 - a2 * b1 in
((b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det)

noncomputable def P := line_intersection A F B E

-- Function to compute the Euclidean distance between points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/- Define lengths AP and PF -/
noncomputable def AP := distance A P
noncomputable def PF := distance P F

/- Now we state the theorem to prove the ratio -/
theorem find_ratio : AP / PF = 2 * Real.sqrt 2 :=
sorry

end find_ratio_l377_377815


namespace eccentricity_of_ellipse_is_sqrt3_over_2_l377_377105

-- Definitions based on conditions
def ellipse_eq (x y : ℝ) : Prop := 16 * x^2 + 4 * y^2 = 1

-- Theorem that we want to prove
theorem eccentricity_of_ellipse_is_sqrt3_over_2 :
  ∀ a b : ℝ, (a = 1/2) → (b = 1/4) → 
  let c := Real.sqrt (a^2 - b^2) in 
  let e := c / a in 
  e = Real.sqrt 3 / 2 :=
by
  intros a b ha hb
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  rw [ha, hb]
  have h1 : c = Real.sqrt (1/4 - 1/16), from sorry
  have h2 : c = Real.sqrt 3 / 4, from sorry
  have h3 : e = (Real.sqrt 3 / 4) / (1 / 2), from sorry
  have h4 : e = Real.sqrt 3 / 2, from sorry
  exact h4

end eccentricity_of_ellipse_is_sqrt3_over_2_l377_377105


namespace find_positive_integers_l377_377046

open Nat

def num_divisors (n : ℕ) : ℕ :=
  (range n).count (λ d => n % d = 0)

theorem find_positive_integers :
  ∃ a b c d : ℕ,
    (a ≤ 70000 ∧ b ≤ 70000 ∧ c ≤ 70000 ∧ d ≤ 70000) ∧
    (num_divisors a > 100) ∧
    (num_divisors b > 100) ∧
    (num_divisors c > 100) ∧
    (num_divisors d > 100) :=
by
  sorry

end find_positive_integers_l377_377046


namespace S_is_square_l377_377966

-- Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def S (n : ℕ) : ℕ :=
  Nat.Cardinal.mk {τ : Fin n → Fin n // ∀ k : Fin n, is_prime (k.1^4 + (τ k).1^4)}

-- Proof goal
theorem S_is_square (n : ℕ) (h : n > 0) : ∃ k : ℕ, k * k = S n := by sorry

end S_is_square_l377_377966


namespace matilda_father_chocolates_l377_377211

theorem matilda_father_chocolates 
  (total_chocolates : ℕ) 
  (total_people : ℕ) 
  (give_up_fraction : ℚ) 
  (mother_chocolates : ℕ) 
  (father_eats : ℕ) 
  (father_left : ℕ) :
  total_chocolates = 20 →
  total_people = 5 →
  give_up_fraction = 1 / 2 →
  mother_chocolates = 3 →
  father_eats = 2 →
  father_left = 5 →
  let chocolates_per_person := total_chocolates / total_people,
      father_chocolates := (chocolates_per_person * total_people * give_up_fraction).nat_abs - mother_chocolates - father_eats
  in father_chocolates = father_left := by
  intros h1 h2 h3 h4 h5 h6
  have h_chocolates_per_person : total_chocolates / total_people = 4 := by sorry
  have h_chocolates_given_up : (chocolates_per_person * total_people * give_up_fraction).nat_abs = 10 := by sorry
  have h_father_chocolates : 10 - mother_chocolates - father_eats = 5 := by sorry
  exact h_father_chocolates

end matilda_father_chocolates_l377_377211


namespace solution_k_values_l377_377045

theorem solution_k_values (k : ℕ) : 
  (∃ m n : ℕ, 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) 
  → k = 1 ∨ 4 ≤ k := 
by
  sorry

end solution_k_values_l377_377045


namespace cost_price_A_l377_377781

theorem cost_price_A (CP_A : ℝ) (h : 300 = 1.20 * 1.30 * 1.25 * 0.85 * CP_A) : CP_A ≈ 181.06 :=
by
  sorry

end cost_price_A_l377_377781


namespace average_speed_including_stoppages_l377_377833

theorem average_speed_including_stoppages
  (speed_excluding_stoppages : ℕ)
  (stoppage_time_per_hour : ℕ)
  (total_time_per_hour : ℕ)
  (speed_excluding_stoppages_eq : speed_excluding_stoppages = 60)
  (stoppage_time_per_hour_eq : stoppage_time_per_hour = 15)
  (total_time_per_hour_eq : total_time_per_hour = 60) :
  (speed_excluding_stoppages * (total_time_per_hour - stoppage_time_per_hour)) / total_time_per_hour = 45 :=
by
  rw [speed_excluding_stoppages_eq, stoppage_time_per_hour_eq, total_time_per_hour_eq]
  norm_num
  sorry

end average_speed_including_stoppages_l377_377833


namespace max_degree_p_for_horizontal_asymptote_l377_377063

noncomputable def rational_function_max_degree (p q : Polynomial ℝ) : Prop :=
  (degree p ≤ degree q) ∧ (degree q = 6)

theorem max_degree_p_for_horizontal_asymptote (p : Polynomial ℝ) :
  rational_function_max_degree p (Polynomial.C 3 * (Polynomial.X)^6 - Polynomial.C 2 * (Polynomial.X)^3 + Polynomial.X - Polynomial.C 4) → degree p ≤ 6 :=
by
  sorry

end max_degree_p_for_horizontal_asymptote_l377_377063


namespace exists_phi_l377_377115

noncomputable def f (x φ : ℝ) : ℝ :=
  sin (2 * x + φ) + cos x ^ 2

def M (φ : ℝ) : ℝ :=
  real.sqrt ((5 : ℝ) / 4 + sin φ) + (1 : ℝ) / 2

def m (φ : ℝ) : ℝ :=
  -real.sqrt ((5 : ℝ) / 4 + sin φ) + (1 : ℝ) / 2

theorem exists_phi : ∃ φ : ℝ, |M φ / m φ| = real.pi :=
by
  sorry

end exists_phi_l377_377115


namespace cyclic_quadrilateral_3AE_10_l377_377163

theorem cyclic_quadrilateral_3AE_10
  (ABCD : Quadrilateral)
  (cyclic : cyclic_quadrilateral ABCD)
  (equal_sides : AB = AC)
  (line_tangent : tangent_to_circle FG C)
  (parallel_lines : BD ∥ FG)
  (AB_length : AB = 6)
  (BC_length : BC = 4) :
  3 * AE = 10 :=
sorry

end cyclic_quadrilateral_3AE_10_l377_377163


namespace eq_D_is_linear_l377_377351

-- Define the given equations as Lean expressions
def eq_A (x y : ℝ) := x - y + 1 = 0
def eq_B (x : ℝ) := x^2 - 4x + 4 = 0
def eq_C (x : ℝ) := 1/x = 2
def eq_D (x : ℝ) := π * x - 2 = 0

-- Define the property of being a linear equation in one variable
def is_linear_equation_in_one_variable (eq : ℝ -> Prop) : Prop :=
  ∃ a b : ℝ, ∃ x : ℝ, eq x ∧ a ≠ 0 ∧ eq = λ x, a * x + b = 0

-- State that eq_D is a linear equation in one variable
theorem eq_D_is_linear : is_linear_equation_in_one_variable eq_D :=
sorry

end eq_D_is_linear_l377_377351


namespace inequality_solution_set_l377_377116

theorem inequality_solution_set (a c x : ℝ) 
  (h1 : -1/3 < x ∧ x < 1/2 → 0 < a * x^2 + 2 * x + c) :
  -2 < x ∧ x < 3 ↔ -c * x^2 + 2 * x - a > 0 :=
by sorry

end inequality_solution_set_l377_377116


namespace solve_linear_equation_l377_377692

theorem solve_linear_equation (x : ℝ) (h : 2 * x - 1 = 1) : x = 1 :=
sorry

end solve_linear_equation_l377_377692


namespace quadratic_has_distinct_real_roots_l377_377690

-- Definitions of the coefficients
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := -2

-- Definition of the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem stating the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots :
  discriminant a b c > 0 :=
by
  -- Coefficients specific to the problem
  unfold a b c
  -- Calculate the discriminant
  unfold discriminant
  -- Substitute the values and compute
  sorry -- Skipping the actual proof as per instructions

end quadratic_has_distinct_real_roots_l377_377690


namespace frog_paths_l377_377757

theorem frog_paths (n : ℕ) : (∃ e_2n e_2n_minus_1 : ℕ,
  e_2n_minus_1 = 0 ∧
  e_2n = (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))) :=
by {
  sorry
}

end frog_paths_l377_377757


namespace integer_diff_of_squares_1_to_2000_l377_377541

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377541


namespace total_time_correct_l377_377361

def speed_boat : ℕ := 16
def speed_stream : ℕ := 2
def distance : ℕ := 7560

def downstream_speed := speed_boat + speed_stream
def upstream_speed := speed_boat - speed_stream

def time_downstream := distance / downstream_speed
def time_upstream := distance / upstream_speed

def total_time := time_downstream + time_upstream

theorem total_time_correct : total_time = 960 := by
  unfold downstream_speed upstream_speed time_downstream time_upstream total_time
  rw [Nat.add_comm, Nat.add_sub_assoc, Nat.sub_add_comm, Nat.add_sub_cancel]
  exact rfl

end total_time_correct_l377_377361


namespace clear_chessboard_l377_377831

-- Define the operations on the chessboard
def removeOneChipFromColumn (board : ℕ → ℕ → ℕ) (col : ℕ) : ℕ → ℕ → ℕ :=
  λ i j => if j = col then board i j - 1 else board i j

def doubleChipsInRow (board : ℕ → ℕ → ℕ) (row : ℕ) : ℕ → ℕ → ℕ :=
  λ i j => if i = row then 2 * board i j else board i j

-- Proposition to clear the board using above operations
theorem clear_chessboard (initial_board : ℕ → ℕ → ℕ) :
  ∃ num_operations : ℕ, ∃ operations : (ℕ → ℕ → ℕ) → ℕ → (ℕ → ℕ → ℕ),
  ∀ (board : ℕ → ℕ → ℕ), board = initial_board →
  (operations board num_operations) = (λ i j, 0) :=
sorry

end clear_chessboard_l377_377831


namespace model_4_best_fitting_effect_l377_377159

noncomputable def model_1_r_squared : ℝ := 0.25
noncomputable def model_2_r_squared : ℝ := 0.50
noncomputable def model_3_r_squared : ℝ := 0.80
noncomputable def model_4_r_squared : ℝ := 0.98

theorem model_4_best_fitting_effect :
  model_4_r_squared = 0.98 →
  model_3_r_squared = 0.80 →
  model_2_r_squared = 0.50 →
  model_1_r_squared = 0.25 →
  model_4_r_squared > model_3_r_squared ∧
  model_4_r_squared > model_2_r_squared ∧
  model_4_r_squared > model_1_r_squared :=
by
  intros h4 h3 h2 h1
  split; sorry

end model_4_best_fitting_effect_l377_377159


namespace twice_original_price_l377_377915

theorem twice_original_price (P : ℝ) (h : 377 = 1.30 * P) : 2 * P = 580 :=
by {
  -- proof steps will go here
  sorry
}

end twice_original_price_l377_377915


namespace shaded_cubes_total_l377_377378

/-- A large cube is made up of 64 smaller cubes, arranged in a 4x4x4 structure.
/ For each face of the large cube, exactly half of the cubes are shaded.
/ The shading on each face follows a checkered pattern, and each face's pattern is mirrored on its opposite face.
/ We want to prove that the total number of smaller cubes that have at least one face shaded is 16. -/
theorem shaded_cubes_total :
  let n := 4
  let N := n * n * n
  let face_cube_count := n * n / 2
  let total_face_cubes := 6 * face_cube_count
  let corners_count := 8
  let edges_without_corners := (total_face_cubes - corners_count) / 2
  corners_count + edges_without_corners = 16 :=
by
  let n := 4
  let N := n * n * n
  let face_cube_count := n * n / 2
  let total_face_cubes := 6 * face_cube_count
  let corners_count := 8
  let edges_without_corners := (total_face_cubes - corners_count) / 2
  exact eq.refl 16

end shaded_cubes_total_l377_377378


namespace donation_amount_by_end_of_5th_day_l377_377167

-- Defining conditions based on the problem statement
def initial_donors : ℕ := 10
def initial_avg_donation : ℕ := 10
def days : ℕ := 5

-- Defining the number of donors function
def donors (n : ℕ) : ℕ :=
  initial_donors * 2 ^ (n - 1)

-- Defining the average donation function
def avg_donation (n : ℕ) : ℕ :=
  initial_avg_donation + (n - 1) * 5

-- Summing the total donations over the given days
def total_donations : ℕ :=
  ∑ n in Finset.range days, donors (n + 1) * avg_donation (n + 1)

-- The theorem to prove
theorem donation_amount_by_end_of_5th_day : total_donations = 8000 := by
  sorry

end donation_amount_by_end_of_5th_day_l377_377167


namespace f_zero_is_one_f_is_even_l377_377079

-- Define the function and its properties
variable (f : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f(x + y) + f(x - y) = 2 * f(x) * f(y))
variable (h₀ : f 0 ≠ 0)

-- Prove that f(0) = 1
theorem f_zero_is_one : f 0 = 1 := 
  sorry

-- Prove that f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x :=
  sorry

end f_zero_is_one_f_is_even_l377_377079


namespace slope_of_line_where_points_lie_l377_377450

theorem slope_of_line_where_points_lie (t : ℝ) : 
  let p := (λ t : ℝ, let a := 2 * (41 * t + 2) / 17,
                     let b := (8 * t + 5 - 2 * (82 * t + 4) / 17) / -3,
                     (a, b)) in 
  ∀ t1 t2, 
    let (x1, y1) := p t1;
    let (x2, y2) := p t2;
    (y2 - y1) / (x2 - x1) = -306 / 697 := 
by
  sorry

end slope_of_line_where_points_lie_l377_377450


namespace parallel_lines_l377_377499

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end parallel_lines_l377_377499


namespace diff_of_squares_1500_l377_377538

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377538


namespace fx_solution_l377_377620

variable {f : ℝ → ℝ}

def fx_conditions : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x * f y - 2 * y * f x = f (x / y)

theorem fx_solution (h : fx_conditions) : f 50 = 0 :=
by
  sorry

end fx_solution_l377_377620


namespace diff_of_squares_1500_l377_377536

theorem diff_of_squares_1500 : 
  (∃ count : ℕ, count = 1500 ∧ ∀ n ∈ set.Icc 1 2000, (∃ a b : ℕ, n = a^2 - b^2) ↔ (n % 2 = 1 ∨ n % 4 = 0)) :=
by
  sorry

end diff_of_squares_1500_l377_377536


namespace x_intercept_of_line_l377_377360

theorem x_intercept_of_line : ∀ (x1 y1 x2 y2 : ℝ), (x1 = 10 ∧ y1 = 3 ∧ x2 = -4 ∧ y2 = -4) → ∃ x : ℝ, (x, 0) is on the line through (x1, y1) and (x2, y2) ∧ x = 4 :=
by 
  intros x1 y1 x2 y2 h,
  sorry

end x_intercept_of_line_l377_377360


namespace two_digit_numbers_odd_or_even_l377_377903

theorem two_digit_numbers_odd_or_even : 
  let odd_digits := {1, 3, 5, 7, 9}
  let even_digits := {0, 2, 4, 6, 8}
  (∑ (tens in odd_digits) (units in odd_digits), 1) + 
  (∑ (tens in {2, 4, 6, 8}) (units in even_digits), 1) = 45 :=
by
  sorry

end two_digit_numbers_odd_or_even_l377_377903


namespace number_of_valid_pairs_l377_377054

-- Definitions based on conditions
def isValidPair (x y : ℕ) : Prop := 
  (1 ≤ x ∧ x ≤ 1000) ∧ (1 ≤ y ∧ y ≤ 1000) ∧ (x^2 + y^2) % 5 = 0

def countValidPairs : ℕ := 
  (Finset.range 1000).filter (λ x, (x + 1) % 5 = 0 ∨ (x + 1) % 5 = 1 ∨ (x + 1) % 5 = 4).card *
  (Finset.range 1000).filter (λ y, (y + 1) % 5 = 0 ∨ (y + 1) % 5 = 1 ∨ (y + 1) % 5 = 4).card +
  2 * (
    (Finset.range 1000).filter (λ x, (x + 1) % 5 = 1).card *
    (Finset.range 1000).filter (λ y, (y + 1) % 5 = 4).card *
    2
  )

theorem number_of_valid_pairs : countValidPairs = 200000 := by
  sorry

end number_of_valid_pairs_l377_377054


namespace problem_A_problem_D_l377_377606

def fibonacci_like_seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | k + 2 => fibonacci_like_seq k + fibonacci_like_seq (k + 1)

theorem problem_A : (∑ i in finset.filter (λ i, i % 2 = 1) (finset.range 101), fibonacci_like_seq i) = fibonacci_like_seq 100 - 1 :=
by sorry

theorem problem_D : (∑ i in finset.range 100, (fibonacci_like_seq i) ^ 2) = fibonacci_like_seq 99 * fibonacci_like_seq 100 - 2 :=
by sorry

end problem_A_problem_D_l377_377606


namespace probability_equal_split_l377_377429

/-- 
Given 6 people (3 teachers and 3 students) divided into 3 groups,
the probability that each group has exactly 1 teacher and 1 student is 2/5.
-/
theorem probability_equal_split
  (teachers students : Finset ℕ)
  (h_teacher_count : teachers.card = 3)
  (h_student_count : students.card = 3) :
  let total_ways := (6.choose 2) * (4.choose 2) * (2.choose 2),
  let favorable_ways := (3.choose 1) * (3.choose 1) * (2.choose 1) * (2.choose 1) * (1.choose 1) * (1.choose 1),
  (favorable_ways / total_ways : ℚ) = 2 / 5 :=
by sorry

end probability_equal_split_l377_377429


namespace num_positive_integer_multiples_2002_l377_377902

theorem num_positive_integer_multiples_2002 : 
  (finset.card (finset.filter 
    (λ (p : ℕ × ℕ), 
      let i := p.fst in 
      let j := p.snd in 
      0 ≤ i ∧ i < j ∧ j ≤ 199 ∧ 2002 ∣ (10^j - 10^i)) 
    ((finset.range (200 + 1)).product (finset.range (200 + 1)))) = 6468) :=
by sorry

end num_positive_integer_multiples_2002_l377_377902


namespace graph_always_passes_fixed_point_l377_377114

theorem graph_always_passes_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ (∀ x : ℝ, y = a^(x+2)-2 → y = -1 ∧ x = -2) :=
by
  use (-2, -1)
  sorry

end graph_always_passes_fixed_point_l377_377114


namespace positive_difference_of_probabilities_l377_377308

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377308


namespace passengers_on_ship_l377_377357

theorem passengers_on_ship (P : ℕ)
  (h1 : P / 12 + P / 4 + P / 9 + P / 6 + 42 = P) :
  P = 108 := 
by sorry

end passengers_on_ship_l377_377357


namespace calc_expression1_l377_377811

theorem calc_expression1 :
    ((9/4:ℚ)^(1/2) - (-2.5)^0 - (8/27:ℚ)^(2/3) + (3/2:ℚ)^(-2)) = 1/2 := 
  sorry

end calc_expression1_l377_377811


namespace postage_cost_is_correct_l377_377778

-- Definitions from the problem as conditions
def base_rate_cents : ℕ := 50
def additional_rate_cents_per_ounce : ℕ := 30
def letter_weight_ounces : ℕ := 625 / 100

-- Mathematical and conversion constants
def base_ounces : ℕ := 1
def cent_to_dollar : ℕ := 100

-- Derived condition
def additional_weight_ounces : ℕ := letter_weight_ounces - base_ounces
def num_additional_charges : ℕ := Nat.ceil additional_weight_ounces.toReal

-- Proposed answer in cents
def total_cost_cents : ℕ := base_rate_cents + num_additional_charges * additional_rate_cents_per_ounce

-- Proposed final answer in dollars
def total_cost_dollars : ℝ := total_cost_cents.toReal / cent_to_dollar

theorem postage_cost_is_correct : 
  total_cost_dollars = 2.30 :=
by
  -- Proof steps go here
  sorry

end postage_cost_is_correct_l377_377778


namespace mass_percentage_O_in_Al2_CO3_3_correct_l377_377440

noncomputable def mass_percentage_O_in_Al2_CO3_3 : ℚ := 
  let mass_O := 9 * 16.00
  let molar_mass_Al2_CO3_3 := (2 * 26.98) + (3 * 12.01) + (9 * 16.00)
  (mass_O / molar_mass_Al2_CO3_3) * 100

theorem mass_percentage_O_in_Al2_CO3_3_correct :
  mass_percentage_O_in_Al2_CO3_3 = 61.54 :=
by
  unfold mass_percentage_O_in_Al2_CO3_3
  sorry

end mass_percentage_O_in_Al2_CO3_3_correct_l377_377440


namespace calculate_shaded_area_l377_377928

def side_length_square : ℝ := 36

def grid_of_circles : ℕ := 3

def side_length_smaller_square : ℝ := side_length_square / grid_of_circles

def radius_of_circle : ℝ := side_length_smaller_square / 2

def area_of_circle : ℝ := π * radius_of_circle ^ 2

def total_area_of_circles : ℝ := 9 * area_of_circle

def area_of_large_square : ℝ := side_length_square ^ 2

def shaded_area : ℝ := area_of_large_square - total_area_of_circles

theorem calculate_shaded_area : shaded_area = 1296 - 324 * π := sorry

end calculate_shaded_area_l377_377928


namespace original_cost_price_l377_377381

theorem original_cost_price (selling_price_friend : ℝ) (friend_gain_tax_rate : ℝ) (man_loss_rate : ℝ) (maintenance_rate : ℝ) (man_tax_rate : ℝ)
  (expected_original_cost : ℝ) :
  (selling_price_friend = 54000) →
  (friend_gain_tax_rate = 1.20 * 1.10) →
  (man_loss_rate = 0.88) →
  (maintenance_rate = 0.05) →
  (man_tax_rate = 1.10) →
  let S := selling_price_friend / (friend_gain_tax_rate) in
  let man_selling_price_before_tax := (man_loss_rate + maintenance_rate) in
  let required_original_cost_price := (S * man_tax_rate) / man_selling_price_before_tax in
  (required_original_cost_price = expected_original_cost) :=
begin
  intros,
  have S := selling_price_friend / friend_gain_tax_rate,
  have man_selling_price_before_tax := (man_loss_rate + maintenance_rate),
  have required_original_cost_price := (S * man_tax_rate) / man_selling_price_before_tax,
  rw S at required_original_cost_price,
  rw man_selling_price_before_tax at required_original_cost_price,
  exact required_original_cost_price
end

end original_cost_price_l377_377381


namespace at_least_three_with_same_mistakes_l377_377940

theorem at_least_three_with_same_mistakes :
  ∃ (k : ℕ), k ∈ (0:ℕ)..11 ∧ ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ∀ (students : ℕ → ℕ), (∀ n, n < 30 → students n ≤ 12) ∧ (∃ n, students n = 12) →
  (∃ n, ∀ i, i < 29 → students (f n i) = k) :=
by
  sorry

end at_least_three_with_same_mistakes_l377_377940


namespace graph_shift_right_eq_l377_377709

theorem graph_shift_right_eq (a : ℝ) (h1 : 0 < a) (h2 : a < π / 2) :
  ∀ x : ℝ, 2 * sin (2 * x + π / 4) = 2 * cos (2 * (x - a)) → a = π / 8 :=
by
  intro x
  sorry

end graph_shift_right_eq_l377_377709


namespace remaining_water_l377_377957

theorem remaining_water (gallons_start: ℚ) (gallons_used: ℚ) (remaining_gallons: ℚ):
  gallons_start = 3 → gallons_used = 11/4 → remaining_gallons = gallons_start - gallons_used → remaining_gallons = 1/4 := 
by
  intros h_start h_used h_sub
  rw [h_start, h_used] at h_sub
  norm_num at h_sub
  exact h_sub

end remaining_water_l377_377957


namespace number_of_integers_as_difference_of_squares_l377_377528

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377528


namespace cookies_and_sugar_needed_l377_377178

-- Definitions derived from the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def initial_sugar : ℝ := 1.5
def flour_needed : ℕ := 5

-- The proof statement
theorem cookies_and_sugar_needed :
  (initial_cookies / initial_flour) * flour_needed = 40 ∧ (initial_sugar / initial_flour) * flour_needed = 2.5 :=
by
  sorry

end cookies_and_sugar_needed_l377_377178


namespace factor_x12_minus_729_l377_377019

theorem factor_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) := 
by
  sorry

end factor_x12_minus_729_l377_377019


namespace diff_squares_count_l377_377551

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377551


namespace candy_lasts_for_days_l377_377828

-- Definitions based on conditions
def candy_from_neighbors : ℕ := 75
def candy_from_sister : ℕ := 130
def candy_traded : ℕ := 25
def candy_lost : ℕ := 15
def candy_eaten_per_day : ℕ := 7

-- Total candy calculation
def total_candy : ℕ := candy_from_neighbors + candy_from_sister - candy_traded - candy_lost
def days_candy_lasts : ℕ := total_candy / candy_eaten_per_day

-- Proof statement
theorem candy_lasts_for_days : days_candy_lasts = 23 := by
  -- sorry is used to skip the actual proof
  sorry

end candy_lasts_for_days_l377_377828


namespace curve_and_line_properties_l377_377880

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 3 * Real.sin θ)

def line (t : ℝ) : ℝ × ℝ :=
  (2 + t, 2 - 2 * t)

def distance (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ :=
  | 4 * P.1 + 3 * P.2 - 6 | / Real.sqrt 5

def max_distance (θ : ℝ) : ℝ := 
  Real.sqrt 5 * | 6 - 5 * Real.sin (θ + Real.arctan (4 / 3)) | / 5

def min_distance : ℝ := 
  Real.sqrt 5 / 5

theorem curve_and_line_properties :
  (∀ θ, curve θ = (2 * Real.cos θ, 3 * Real.sin θ)) ∧
  ( ∀ t, 2 * (2 + t) + (2 - 2 * t) - 6 = 0) ∧
  (∀ P : ℝ × ℝ, (P = curve x) → max_distance θ = (11 * Real.sqrt 5 / 5) ∧ min_distance = (Real.sqrt 5 / 5)) := 
by sorry

end curve_and_line_properties_l377_377880


namespace tan_product_inequality_l377_377626

-- Define the conditions and theorem
theorem tan_product_inequality
  {n : ℕ}
  {a : fin (n+1) → ℝ}
  (h0 : ∀ i, 0 < a i ∧ a i < π / 2)
  (h1 : ∑ i, Real.tan (a i - π / 4) ≥ n - 1)
  : ∏ i, Real.tan (a i) ≥ n ^ (n - 1) := 
sorry

end tan_product_inequality_l377_377626


namespace triangle_angle_C_triangle_perimeter_l377_377945

theorem triangle_angle_C {C : ℝ} (h1 : sin (2 * C) = sqrt 3 * cos C) (h2 : π / 2 < C ∧ C < π) :
  C = 2 * π / 3 :=
sorry

theorem triangle_perimeter {a b c : ℝ} (hC : ∠C = 2 * π / 3) (b_eq : b = 6) (area_eq : 6 * sqrt 3) :
  a + b + c = 10 + 2 * sqrt 19 :=
sorry

end triangle_angle_C_triangle_perimeter_l377_377945


namespace vector_projection_constant_l377_377832

structure Vector (α : Type) := (x y : α)

noncomputable def proj (v w : Vector ℝ) : Vector ℝ :=
(let num := v.x * w.x + v.y * w.y in
 let den := w.x * w.x + w.y * w.y in
  { x := (num / den) * w.x, y := (num / den) * w.y })

def line_vector (a : ℝ) : Vector ℝ :=
{ x := a, y := (3 / 4) * a + 3 }

def w' : Vector ℝ := { x := -(3 / 4), y := 1 }

theorem vector_projection_constant :
  ∀ (a : ℝ), proj (line_vector a) w' = { x := (-36 / 25), y := (48 / 25) } :=
by sorry

end vector_projection_constant_l377_377832


namespace ratio_of_roots_l377_377055

theorem ratio_of_roots 
  (a b c : ℝ) 
  (h : a * b * c ≠ 0)
  (x1 x2 : ℝ) 
  (root1 : x1 = 2022 * x2) 
  (root2 : a * x1 ^ 2 + b * x1 + c = 0) 
  (root3 : a * x2 ^ 2 + b * x2 + c = 0) : 
  2023 * a * c / b ^ 2 = 2022 / 2023 :=
by
  sorry

end ratio_of_roots_l377_377055


namespace greg_rolls_more_ones_than_fives_l377_377571

def probability_more_ones_than_fives (n : ℕ) : ℚ :=
  if n = 6 then 695 / 1944 else 0

theorem greg_rolls_more_ones_than_fives :
  probability_more_ones_than_fives 6 = 695 / 1944 :=
by sorry

end greg_rolls_more_ones_than_fives_l377_377571


namespace desired_sum_of_reciprocals_l377_377195

noncomputable def polynomial_roots : Prop :=
  let p q r s t : ℝ
  in roots (X^5 + 10*X^4 + 20*X^3 + 15*X^2 + 6*X + 3) = [p, q, r, s, t]

theorem desired_sum_of_reciprocals :
  polynomial_roots →
  pq + pr + ps + pt + qr + qs + qt + rs + rt + st = 20 →
  pqrst = 3 →
  (1 / pq) + (1 / pr) + (1 / ps) + (1 / pt) +
  (1 / qr) + (1 / qs) + (1 / qt) +
  (1 / rs) + (1 / rt) + (1 / st) = 20 / 3 :=
sorry

end desired_sum_of_reciprocals_l377_377195


namespace find_m_n_l377_377047

theorem find_m_n (m n : ℕ) (h : (m^(1/3 : ℝ) + n^(1/3 : ℝ) - 1)^2 = 49 + 20 * 6^(1/3 : ℝ)) : 
  m = 48 ∧ n = 288 := 
sorry

end find_m_n_l377_377047


namespace f_is_even_implies_f2_eq_3_l377_377679

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * (x - a)

theorem f_is_even_implies_f2_eq_3 (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : f 2 1 = 3 :=
by {
  -- sorry to skip the proof
  sorry,
}

end f_is_even_implies_f2_eq_3_l377_377679


namespace solve_frac_eqn_l377_377839

theorem solve_frac_eqn (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) +
   1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8) ↔ 
  (x = 13 ∨ x = -3) :=
by
  sorry

end solve_frac_eqn_l377_377839


namespace path_count_to_B_in_blocked_grid_l377_377637

theorem path_count_to_B_in_blocked_grid :
  let total_paths := nat.choose (14) (4)
  let paths_thru_first_block := (nat.choose (5) (2)) * (nat.choose (9) (2))
  let paths_thru_second_block := (nat.choose (9) (3)) * (nat.choose (5) (1))
  let valid_paths := total_paths - paths_thru_first_block - paths_thru_second_block
  total_paths = 1001 ∧
  paths_thru_first_block = 360 ∧
  paths_thru_second_block = 420 ∧
  valid_paths = 221
: valid_paths = 221 :=
by
  -- The body of the proof is skipped by using sorry.
  sorry

end path_count_to_B_in_blocked_grid_l377_377637


namespace binom_product_l377_377414

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_product :
  binom 10 3 * binom 8 3 = 6720 := by
  sorry

end binom_product_l377_377414


namespace sum_of_consecutive_pairs_eq_pow_two_l377_377258

theorem sum_of_consecutive_pairs_eq_pow_two (n m : ℕ) :
  ∃ n m : ℕ, (n * (n + 1) + m * (m + 1) = 2 ^ 2021) :=
sorry

end sum_of_consecutive_pairs_eq_pow_two_l377_377258


namespace math_equivalent_proof_l377_377898

noncomputable def intersection_point (m : ℝ) (l1 l2 : (ℝ × ℝ × ℝ)) : (ℝ × ℝ) :=
if h : m = 1 then
  let soln := linear_system_solver (
    [l1.1, l1.2, l1.3],
    [l2.1, l2.2, l2.3]
  ) in soln else (0,0)

theorem math_equivalent_proof (m x y : ℝ) (h1 : x + y - 3 * m = 0)
(h2 : 2 * x - y + 2 * m - 1 = 0) (h3 : 3 * m = 3)
: x = 2 / 3 ∧ y = 7 / 3 ∧
(∃ c : ℝ,  x + 2 * y + c = 0 ∧ c = -16 / 3) ∧
(∃ (a b d : ℝ), a = 3 ∧ b = 6 ∧ d = -16 ∧ a * x + b * y + d = 0) :=
by
  sorry

end math_equivalent_proof_l377_377898


namespace question1_question2_l377_377113

def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

theorem question1 (x : ℝ) : ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) → m ≤ 8 :=
by sorry

theorem question2 (x : ℝ) : (∀ x : ℝ, |x - 3| - 2 * x ≤ 2 * 8 - 12) ↔ (x ≥ -1/3) :=
by sorry

end question1_question2_l377_377113


namespace multiples_of_4_between_80_and_300_l377_377510

theorem multiples_of_4_between_80_and_300 : 
  set.count {n : ℕ | 80 < n ∧ n < 300 ∧ 4 ∣ n} = 54 :=
by
  sorry

end multiples_of_4_between_80_and_300_l377_377510


namespace youngest_child_cakes_l377_377999

theorem youngest_child_cakes : 
  let total_cakes := 60
  let oldest_cakes := (1 / 4 : ℚ) * total_cakes
  let second_oldest_cakes := (3 / 10 : ℚ) * total_cakes
  let middle_cakes := (1 / 6 : ℚ) * total_cakes
  let second_youngest_cakes := (1 / 5 : ℚ) * total_cakes
  let distributed_cakes := oldest_cakes + second_oldest_cakes + middle_cakes + second_youngest_cakes
  let youngest_cakes := total_cakes - distributed_cakes
  youngest_cakes = 5 := 
by
  exact sorry

end youngest_child_cakes_l377_377999


namespace coin_probability_difference_l377_377332

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377332


namespace conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l377_377756

theorem conversion_7_dms_to_cms :
  7 * 100 = 700 :=
by
  sorry

theorem conversion_5_hectares_to_sms :
  5 * 10000 = 50000 :=
by
  sorry

theorem conversion_600_hectares_to_sqkms :
  600 / 100 = 6 :=
by
  sorry

theorem conversion_200_sqsmeters_to_smeters :
  200 / 100 = 2 :=
by
  sorry

end conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l377_377756


namespace count_irrationals_l377_377920

open Real

-- Lean definitions for the numbers given
def num1 := -sqrt 2
def num2 := Real.mkP 31 99  -- 0.3131 as a repeating decimal can be approximated
def num3 := pi / 3
def num4 := 1 / 7
def num5 := 40.108 / 50 -- same for 0.80108 as a terminating decimal, expressed as a fraction

-- Lean definitions to check if a number is irrational
def is_irrational (x : Real) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

-- The statement we need to prove
theorem count_irrationals :
  (is_irrational num1 ∧ ¬is_irrational num2 ∧ 
   is_irrational num3 ∧ ¬is_irrational num4 ∧ ¬is_irrational num5) → 
  2 = List.length (List.filter is_irrational [num1, num2, num3, num4, num5]) :=
by
  sorry -- Here is where the proof would be

end count_irrationals_l377_377920


namespace part1_part2_l377_377943

-- Defining the basic setup for the triangle and conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def condition1 (A B : ℝ) (a b : ℝ) : Prop :=
  b * sin A = sqrt 3 * a * cos B

def condition2 (A C : ℝ) : Prop :=
  sin C = 2 * sin A

-- Part 1: Prove measure of angle B
theorem part1 {A B : ℝ} {a b : ℝ} (h1 : condition1 A B a b) : B = π / 3 :=
sorry

-- Part 2: Prove the values of a and c
theorem part2 {A B C : ℝ} {a b c : ℝ} (h1 : condition1 A B a b) (h2 : condition2 A C) (hb : b = 3) : 
  a = sqrt 3 ∧ c = 2 * sqrt 3 :=
sorry

end part1_part2_l377_377943


namespace Jack_and_Jill_meet_distance_from_top_l377_377953

theorem Jack_and_Jill_meet_distance_from_top :
  let 
    total_distance := 12 -- km
    hill_distance := 6 -- km
    jack_start_ahead := (1:ℝ)/4 -- hours
    jack_speed_uphill := 12 -- km/hr
    jack_speed_downhill := 18 -- km/hr
    jill_speed_uphill := 14 -- km/hr
    jill_speed_downhill := 20 -- km/hr
    
    jack_time_uphill := hill_distance / jack_speed_uphill -- hours
    jill_time_uphill := hill_distance / jill_speed_uphill -- hours
    
    t_meet := (23:ℝ) / 64 -- hours after Jack starts
    
    meet_position := 14 * (t_meet - jack_start_ahead) -- km from start
    
    distance_from_top := 6 - meet_position -- km from top
  in 
    distance_from_top = (87:ℝ) / 32 := 
sorry

end Jack_and_Jill_meet_distance_from_top_l377_377953


namespace integer_diff_of_squares_1_to_2000_l377_377540

theorem integer_diff_of_squares_1_to_2000 :
  let count_diff_squares := (λ n, ∃ a b : ℕ, (a^2 - b^2 = n)) in
  (1 to 2000).filter count_diff_squares |>.length = 1500 :=
by
  sorry

end integer_diff_of_squares_1_to_2000_l377_377540


namespace number_of_pairs_divisible_by_5_l377_377051

theorem number_of_pairs_divisible_by_5 :
  let n := 1000
  let count := 200000
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n → (x^2 + y^2) % 5 = 0) →
  (∃ count : ℕ, count == 200000) :=
begin
  sorry
end

end number_of_pairs_divisible_by_5_l377_377051


namespace candy_bar_cost_correct_l377_377422

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def candy_bar_cost : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost_correct : candy_bar_cost = 1 := by
  unfold candy_bar_cost
  sorry

end candy_bar_cost_correct_l377_377422


namespace trig_values_through_point_trig_expression_value_l377_377100

noncomputable def angle_through_point (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2 in
  true  -- This definition captures that we have a point.

theorem trig_values_through_point (P : ℝ × ℝ) (hP : angle_through_point P):
  let θ := real.arctan2 P.2 P.1 in
  P = (-4, 3) →
  real.sin θ = 3 / 5 ∧
  real.cos θ = -4 / 5 ∧
  real.tan θ = -3 / 4 := 
by
  sorry

theorem trig_expression_value (P : ℝ × ℝ) (hP : angle_through_point P), 
  let θ := real.arctan2 P.2 P.1 in
  P = (-4, 3) →
  (real.cos (θ - real.pi / 2) / real.sin (real.pi / 2 + θ)) * real.sin (θ + real.pi) * real.cos (2 * real.pi - θ) = -9 / 25 :=
by
  sorry

end trig_values_through_point_trig_expression_value_l377_377100


namespace general_term_formula_sum_of_first_n_terms_l377_377493

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom a_1 : a 1 = 1 / 2
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = (1 / 2) * a n + (2 * n + 1) / 2^(n + 1)

-- First part: general term formula
theorem general_term_formula (n : ℕ) (h : n > 0) : a n = n^2 / 2^n :=
sorry

-- Conditions for sum of the first n terms
axiom S_definition : ∀ n : ℕ, S n = ∑ k in Finset.range (n + 1), a (k + 1)

-- Second part: sum of the first n terms
theorem sum_of_first_n_terms (n : ℕ) : S n = 6 - (n^2 + 4 * n + 6) / 2^n :=
sorry

end general_term_formula_sum_of_first_n_terms_l377_377493


namespace total_number_of_toys_is_105_l377_377205

-- Definitions
variables {a k : ℕ}

-- Conditions
def condition_1 (a k : ℕ) : Prop := k ≥ 2
def katya_toys (a : ℕ) : ℕ := a
def lena_toys (a k : ℕ) : ℕ := k * a
def masha_toys (a k : ℕ) : ℕ := k^2 * a

def after_katya_gave_toys (a : ℕ) : ℕ := a - 2
def after_lena_received_toys (a k : ℕ) : ℕ := k * a + 5
def after_masha_gave_toys (a k : ℕ) : ℕ := k^2 * a - 3

def arithmetic_progression (x1 x2 x3 : ℕ) : Prop :=
  2 * x2 = x1 + x3

-- Problem statement to prove
theorem total_number_of_toys_is_105 (a k : ℕ) (h1 : condition_1 a k)
  (h2 : arithmetic_progression (after_katya_gave_toys a) (after_lena_received_toys a k) (after_masha_gave_toys a k)) :
  katya_toys a + lena_toys a k + masha_toys a k = 105 :=
sorry

end total_number_of_toys_is_105_l377_377205


namespace area_of_triangle_PQR_l377_377279

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := -4, y := 2 }
def Q : Point := { x := 8, y := 2 }
def R : Point := { x := 6, y := -4 }

noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangle_area P Q R = 36 := by
  sorry

end area_of_triangle_PQR_l377_377279


namespace max_values_sum_l377_377456

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem max_values_sum : 
  (∑ k in Finset.range 1008, f ((2 * k + 1) * Real.pi)) = 
  (Real.exp Real.pi * (1 - Real.exp (2014 * Real.pi)) / 
  (1 - Real.exp (2 * Real.pi))) := 
sorry

end max_values_sum_l377_377456


namespace minimal_sticks_sieve_l377_377448

def is_sieve {n : ℕ} (A : fin n × fin n → Prop) : Prop :=
  n ≥ 2 ∧ (∀ r, ∃! c, ¬A (r, c)) ∧ (∀ c, ∃! r, ¬A (r, c))

def minimal_sticks {n : ℕ} (A : fin n × fin n → Prop) : ℕ :=
  -- Placeholder for the actual definition of minimal_sticks
  sorry

theorem minimal_sticks_sieve
  (n : ℕ)
  (A : fin n × fin n → Prop)
  (h : is_sieve A) :
  minimal_sticks A = 2 * n - 2 :=
sorry

end minimal_sticks_sieve_l377_377448


namespace valid_pairs_satisfy_condition_l377_377838

theorem valid_pairs_satisfy_condition :
  ∀ (m n : ℕ),
    0 < m → 0 < n →
    ((∃ k: ℕ, 2 ≤ k ∧ m = k ∧ n = 1) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) ↔
    (1 ≤ m^n - n^m ∧ m^n - n^m ≤ m * n) :=
by
  intros m n m_pos n_pos
  split
  {
    intro h
    cases h
    {
      cases h with k hk
      cases hk
      cases hk_right
      constructor
      {
        rw hk_left_right at *
        exact nat.le_sub_right_of_add_le (nat.add_one_le_iff.mpr hk_left_left)
      },
      {
        rw hk_left_right at *
        calc
          m * 1 = m                     : by rw nat.mul_one
          ...   = k                     : by rw hk_left_right
          ...   ≤ 1 + k - k             : (by simp)
          ...   ≤ k^1 - 1^k             : (by exact hk_left_left)
      }
    },
    {
      cases h,
      {
        cases h with m_eq2 n_eq5
        rw m_eq2 at *
        rw n_eq5 at *
        constructor
        {
          exact nat.le_trans (nat.le_of_sub add_zero) nat.le_rfl
        },
        {
          rw nat.ne_zero (2 - 5)
          simp
        }
      },
      {
        cases h with h32 h32
        cases h32
        rw h32_left at *
        rw h32_right at *
        constructor
        {
          rw nat.le_of_sub (3 - 8)
          simp
        },
        {
          rw nat.le_zero (3 * 2 - 9)
          simp
        }
      }
    }
  },
  sorry

end valid_pairs_satisfy_condition_l377_377838


namespace total_revenue_from_sale_l377_377801

def total_weight_of_potatoes : ℕ := 6500
def weight_of_damaged_potatoes : ℕ := 150
def weight_per_bag : ℕ := 50
def price_per_bag : ℕ := 72

theorem total_revenue_from_sale :
  (total_weight_of_potatoes - weight_of_damaged_potatoes) / weight_per_bag * price_per_bag = 9144 := 
begin
  sorry
end

end total_revenue_from_sale_l377_377801


namespace vehicles_traveled_last_year_correct_l377_377958

-- Define the given conditions
def ratio_accidents : ℝ := 75 / 100000000 -- Ratio of accidents per vehicles
def number_of_accidents : ℝ := 4500      -- Number of accidents last year

-- Define the variable to find: total number of vehicles traveled
def vehicles_traveled_last_year : ℝ := number_of_accidents * 100000000 / 75

-- Statement to prove
theorem vehicles_traveled_last_year_correct : 
  vehicles_traveled_last_year = 6000000000 := 
by
  -- The proof goes here
  sorry

end vehicles_traveled_last_year_correct_l377_377958


namespace coin_probability_difference_l377_377327

theorem coin_probability_difference :
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  p3 - p4 = (7/16 : ℚ) :=
by
  let p3 := (4.choose 3) * (1/2)^3 * (1/2)^1
  let p4 := (1/2)^4
  have h1 : p3 = (1/2 : ℚ), by norm_num [finset.range_succ]
  have h2 : p4 = (1/16 : ℚ), by norm_num
  rw [h1, h2]
  norm_num

end coin_probability_difference_l377_377327


namespace matrix_mul_pow_l377_377959

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![0, 0, 1], 
    ![1, 0, 0], 
    ![0, 1, 0]]

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, 1, 1], 
    ![0, 1, -1], 
    ![1, 0, 0]]

noncomputable def C : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, 0, 0], 
    ![1, 1, 1], 
    ![0, 1, -1]]

theorem matrix_mul_pow (A₉₉B_eq : (A^99) * B = C) : ∃ A B C, (A^99) * B = C := 
by
  sorry

end matrix_mul_pow_l377_377959


namespace numberOfAntiPalindromes_l377_377029

-- Define what it means for a number to be an anti-palindrome in base 3
def isAntiPalindrome (n : ℕ) : Prop :=
  ∀ (a b : ℕ), a + b = 2 → a ≠ b

-- Define the constraint of no two consecutive digits being the same
def noConsecutiveDigits (digits : List ℕ) : Prop :=
  ∀ (i : ℕ), i < digits.length - 1 → digits.nthLe i sorry ≠ digits.nthLe (i + 1) sorry

-- We want to find the number of anti-palindromes less than 3^12 fulfilling both conditions
def countAntiPalindromes (m : ℕ) (base : ℕ) : ℕ :=
  sorry -- Placeholder definition for the count, to be implemented

-- The main theorem to prove
theorem numberOfAntiPalindromes : countAntiPalindromes (3^12) 3 = 126 :=
  sorry -- Proof to be filled

end numberOfAntiPalindromes_l377_377029


namespace solution_1_solution_2_l377_377753

noncomputable def problem_1 (m : ℝ) : Prop :=
  2 * 3 = m * (m + 1)

theorem solution_1 (m : ℝ) :
  (2x + (m + 1)y + 4 = 0) ∧ (mx + 3y - 2 = 0) →
  problem_1 m →
  (m = -3 ∨ m = 2) :=
sorry

noncomputable def problem_2 (a : ℝ) : Prop :=
  (a + 2) * (a - 1) + (1 - a) * (2a + 3) = 0

theorem solution_2 (a : ℝ) :
  ((a+2)x + (1-a)y - 1 = 0) ∧ ((a-1)x + (2a+3)y + 2 = 0) →
  problem_2 a →
  (a = 1 ∨ a = -1) :=
sorry

end solution_1_solution_2_l377_377753


namespace equivalent_expression_l377_377731

theorem equivalent_expression (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (a^(-2) * b^(-3)) / (a^(-4) + b^(-6)) = (a^2 * b^3) / (b^6 + a^4) :=
sorry

end equivalent_expression_l377_377731


namespace six_digit_numbers_divisible_by_13_l377_377189

theorem six_digit_numbers_divisible_by_13 (q r n : ℕ) :
  (1000 ≤ q ∧ q ≤ 9999) →
  (0 ≤ r ∧ r ≤ 99) →
  (n = 100 * q + r) →
  (q + r) % 13 = 0 →
  72000 = (count_six_digit_numbers_mod_13 q r n) :=
sorry

end six_digit_numbers_divisible_by_13_l377_377189


namespace central_angle_radian_measure_l377_377877

-- Definitions of the given conditions
def circumference_of_sector (r α : ℝ) : ℝ := α * r + 2 * r
def area_of_sector (r α : ℝ) : ℝ := (1 / 2) * α * r^2

-- The proof problem translated to a Lean statement
theorem central_angle_radian_measure (α r : ℝ) 
  (h1 : circumference_of_sector r α = 8) 
  (h2 : area_of_sector r α = 4) : α = 2 :=
by
  sorry

end central_angle_radian_measure_l377_377877


namespace find_b_l377_377238

def f (x : ℝ) : ℝ := (3 * x) / 7 + 4
def g (x : ℝ) : ℝ := 5 - 2 * x

theorem find_b (b : ℝ) (h : f (g b) = 10) : b = -4.5 := by
  sorry

end find_b_l377_377238


namespace angle_CXD_tangents_triangle_l377_377604

theorem angle_CXD_tangents_triangle (CQD : Triangle) (center : Point) (h_tangents : ∀ P, IsTangent PQ center) (angle_CQD : CQD.angle = 50) : 
  CQD.angle_at_center = 65 :=
by
  sorry

end angle_CXD_tangents_triangle_l377_377604


namespace students_taking_neither_l377_377379

theorem students_taking_neither (total_students music_students art_students dance_students music_art music_dance art_dance music_art_dance : ℕ) :
  total_students = 2500 →
  music_students = 200 →
  art_students = 150 →
  dance_students = 100 →
  music_art = 75 →
  art_dance = 50 →
  music_dance = 40 →
  music_art_dance = 25 →
  total_students - ((music_students + art_students + dance_students) - (music_art + art_dance + music_dance) + music_art_dance) = 2190 :=
by
  intros
  sorry

end students_taking_neither_l377_377379


namespace percent_tenured_professors_l377_377008

variables (totalProfs : ℕ) (W T M : ℕ)
variables (percWomen percWomenTenuredOrBoth percMenTenured : ℝ)

-- The conditions
axiom h1 : percWomen = 0.70
axiom h2 : percWomenTenuredOrBoth = 0.90
axiom h3 : percMenTenured = 0.50

-- Definitions based on the conditions
def numWomen := percWomen * totalProfs
def numMen := (1 - percWomen) * totalProfs
def numMenTenured := percMenTenured * numMen
def numWomenTenured := percWomenTenuredOrBoth * totalProfs - numMenTenured
def totalTenured := numWomenTenured + numMenTenured

-- The theorem to prove
theorem percent_tenured_professors :
  totalTenured / totalProfs = 0.90 :=
  sorry

end percent_tenured_professors_l377_377008


namespace values_of_z_l377_377845

theorem values_of_z (x z : ℝ) 
  (h1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (h2 : 3 * x + z + 4 = 0) : 
  z^2 + 20 * z - 14 = 0 := 
sorry

end values_of_z_l377_377845


namespace segment_equality_l377_377644

variables (A B C D P K L M Q R S : Point)
variables (AC BD KL AD BC : Line)
variables [LineThrough BD P] [LineThrough AC P]

-- Given conditions
axiom H1 : midpoint K AC 
axiom H2 : midpoint L BD
axiom H3 : midpoint M K L
axiom H4 : symmetric Q P M
axiom H5 : parallel_line_through Q KL
axiom H6 : intersects R Q AD
axiom H7 : intersects S Q BC

-- The theorem to be proved
theorem segment_equality :
  segment (Q ↔ R) ∩ (AD ∪ AC) = segment (Q ↔ S) ∩ (BC ∪ BD) :=
sorry

end segment_equality_l377_377644


namespace find_mean_l377_377672

noncomputable def mean_of_normal_distribution (σ : ℝ) (value : ℝ) (std_devs : ℝ) : ℝ :=
value + std_devs * σ

theorem find_mean
  (σ : ℝ := 1.5)
  (value : ℝ := 12)
  (std_devs : ℝ := 2)
  (h : value = mean_of_normal_distribution σ (value - std_devs * σ) std_devs) :
  mean_of_normal_distribution σ value std_devs = 15 :=
sorry

end find_mean_l377_377672


namespace two_digit_numbers_count_l377_377733

theorem two_digit_numbers_count : 
  ∃ n : ℕ, (∀ (d₁ d₂ : ℕ), d₁ ∈ {1, 2, 3, 4, 5, 6} ∧ d₂ ∈ {1, 2, 3, 4, 5, 6} ∧ d₁ ≠ d₂ → n = 30) :=
sorry

end two_digit_numbers_count_l377_377733


namespace unique_number_not_in_range_l377_377248

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p * x + q) / (r * x + s)

theorem unique_number_not_in_range (p q r s : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : r ≠ 0) (h₃ : s ≠ 0) 
  (h₄ : g p q r s 23 = 23) (h₅ : g p q r s 101 = 101) (h₆ : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  p / r = 62 :=
sorry

end unique_number_not_in_range_l377_377248


namespace concyclic_BPRQ_l377_377478

noncomputable theory

-- Definitions given in the problem
variables {Γ : Type*} [circle Γ]
variables {A B C D E P Q R : Type*} [point A] [point B] [point C] [point D] [point E] [point P] [point Q] [point R]
variables {l : Type*} [line l]
variables (circumcircle : ∀{X Y Z W : Type*}, [cyclic_quadrilateral X Y Z W] → Prop)
variables (tangent : ∀{X Y : Type*}, [tangent_to_line X l] → Prop)
variables (intersect : ∀{X : Type*}, [intersect_with_circle X Γ] → Prop)
variables (line_through : ∀{X Y : Type*}, [on_line X l] → Prop)

-- Conditions described in the problem
axiom hG : circumcircle A B C D
axiom hE : intersect E Γ
axiom hP : line_through E P ∧ line_through P A ∧ line_through P B
axiom hQ : line_through E Q ∧ line_through Q B ∧ line_through Q C

-- The conjunctive condition that there is a circle passing through D, is tangent to l at E, and intersects Γ at a second point R.
axiom hR : tangent (circle_through_points D E) l ∧ intersect R (circle_through_points D E) ∧ intersect R Γ

-- The goal: show that B, P, R, and Q are concyclic
theorem concyclic_BPRQ : circumcircle B P R Q :=
sorry

end concyclic_BPRQ_l377_377478


namespace passengers_on_ship_l377_377570

theorem passengers_on_ship :
  (∀ P : ℕ, 
    (P / 12) + (P / 8) + (P / 3) + (P / 6) + 35 = P) → P = 120 :=
by 
  sorry

end passengers_on_ship_l377_377570


namespace transport_cost_l377_377240

-- Define the conditions given in the problem
def mass_kg := 0.15
def cost_per_kg := 25000

-- Lean statement for the mathematically equivalent proof problem
theorem transport_cost :
  mass_kg * cost_per_kg = 3750 := 
by
  sorry

end transport_cost_l377_377240


namespace width_of_rectangular_prism_l377_377907

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end width_of_rectangular_prism_l377_377907


namespace common_difference_of_arithmetic_sequence_l377_377934

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : a 1 + a 9 = 10)
  (h2 : a 2 = -1)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l377_377934


namespace train_speed_l377_377788

noncomputable def speed_of_train_kmph (L V : ℝ) : ℝ :=
  3.6 * V

theorem train_speed
  (L V : ℝ)
  (h1 : L = 18 * V)
  (h2 : L + 340 = 35 * V) :
  speed_of_train_kmph L V = 72 :=
by
  sorry

end train_speed_l377_377788


namespace time_to_cross_platform_l377_377370

def length_train : ℝ := 180
def speed_kmph : ℝ := 72
def length_platform : ℝ := 220.03199999999998

def total_distance : ℝ := length_train + length_platform
def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem time_to_cross_platform : total_distance / speed_mps ≈ 20 :=
by
  sorry

end time_to_cross_platform_l377_377370


namespace game_fairness_l377_377270

theorem game_fairness (k : ℕ) (hk : k > 0 ∧ k ≤ 1986) :
    ∃ (S : finset ℕ), S = finset.range 1986 ∧ 
    (∀ A : finset ℕ, A.card = k → (∀ a ∈ A, a ∈ S) →
      (∃ B C : finset ℕ, 
        B.card = k ∧ ∀ b ∈ B, b ∈ S ∧ (b ∉ A ∧ 
        ∀ c ∈ C, c ∈ S ∧ (c ∉ A ∧ 
        (A.sum % 3 = 0 ∧ B.sum % 3 = 1 ∧ C.sum % 3 = 2 ∧
        A.sum % 3 = B.sum % 3 ∧ B.sum % 3 = C.sum % 3)))))) ↔ k % 3 ≠ 0 := 
by
  sorry

end game_fairness_l377_377270


namespace modified_short_bingo_first_column_possibilities_l377_377923

theorem modified_short_bingo_first_column_possibilities :
  (∃ (first: ℕ) (second: ℕ) (third: ℕ) (fourth: ℕ) (fifth: ℕ),
    first ∈ Finset.range 16 ∧
    second ∈ Finset.range 16 ∧
    third ∈ Finset.range 16 ∧
    fourth ∈ Finset.range 16 ∧
    fifth ∈ Finset.range 16 ∧
    first ≠ second ∧
    first ≠ third ∧
    first ≠ fourth ∧
    first ≠ fifth ∧
    second ≠ third ∧
    second ≠ fourth ∧
    second ≠ fifth ∧
    third ≠ fourth ∧
    third ≠ fifth ∧
    fourth ≠ fifth
  ) = 360360 :=
by
  sorry

end modified_short_bingo_first_column_possibilities_l377_377923


namespace speed_in_still_water_is_28_l377_377354

-- Define the upstream speed and downstream speed as given in the conditions.
def upstream_speed : ℝ := 26
def downstream_speed : ℝ := 30

-- Define the speed in still water as the average of upstream and downstream speeds.
def still_water_speed (u d : ℝ) : ℝ := (u + d) / 2

-- Theorem statement to prove that the speed in still water is 28 kmph.
theorem speed_in_still_water_is_28 : still_water_speed upstream_speed downstream_speed = 28 :=
by
  sorry

end speed_in_still_water_is_28_l377_377354


namespace find_x_l377_377751

theorem find_x (x : ℝ) (hx : (sqrt x + sqrt 567) / sqrt 175 = 2.6) : x = 112 := 
sorry

end find_x_l377_377751


namespace sqrt_2abc_sum_l377_377618

theorem sqrt_2abc_sum (a b c : ℝ)
  (h1 : b + c = 20)
  (h2 : c + a = 22)
  (h3 : a + b = 24)
  :
  sqrt (2 * a * b * c * (a + b + c)) = 1287 := 
by 
  sorry

end sqrt_2abc_sum_l377_377618


namespace ratio_of_larger_to_smaller_l377_377697

theorem ratio_of_larger_to_smaller
  (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h3 : x + y = 6 * (x - y)) :
  x / y = 7 / 5 :=
by sorry

end ratio_of_larger_to_smaller_l377_377697


namespace find_digit_A_l377_377589

open Nat

theorem find_digit_A :
  let n := 52
  let k := 13
  let number_of_hands := choose n k
  number_of_hands = 635013587600 → 0 = 0 := by
  suffices h: 635013587600 = 635013587600 by
    simp [h]
  sorry

end find_digit_A_l377_377589


namespace min_total_cost_l377_377004

theorem min_total_cost (p_event : ℝ) (loss : ℝ) (cost_A : ℝ) (cost_B : ℝ) (p_no_event_A : ℝ) (p_no_event_B : ℝ) :
  p_event = 0.3 →
  loss = 4 →
  cost_A = 0.45 →
  cost_B = 0.3 →
  p_no_event_A = 0.9 →
  p_no_event_B = 0.85 →
  let total_cost_no_measures := loss * p_event,
      total_cost_A := cost_A + loss * (1 - p_no_event_A),
      total_cost_B := cost_B + loss * (1 - p_no_event_B),
      total_cost_A_B := cost_A + cost_B + loss * (1 - p_no_event_A) * (1 - p_no_event_B)
  in min (min total_cost_no_measures total_cost_A) (min total_cost_B total_cost_A_B) = 0.81 :=
sorry

end min_total_cost_l377_377004


namespace number_of_endings_l377_377612

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := (n / 100 + (n / 10) % 10 + n % 10) % 3 = 0

theorem number_of_endings (n : ℕ) (hn1 : n < 1000) (hn2 : ∃ k, k < 25 ∧ n = k * 4 ∧ sum_of_digits_divisible_by_3 n) :
    ∃ unique_pairs, unique_pairs.count = 45 := sorry

end number_of_endings_l377_377612


namespace width_of_rectangular_prism_l377_377908

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end width_of_rectangular_prism_l377_377908


namespace find_k_value_l377_377860

noncomputable def quadratic_roots_complex_and_diff (k : ℝ) : Prop :=
  let a : ℝ := 2
  let b : ℝ := k
  let c : ℝ := 26
  let Δ : ℝ := b^2 - 4 * a * c
  -- Roots are complex if the discriminant is negative
  Δ < 0 ∧ let x1_x2_diff := (a^2 * b^2 - 4 * a * b * c)^2 - 4 * 13 in
  -- The difference in magnitude of the roots is 6
  |√(b^2 / (2 * a)^2 - 52)| = 6

theorem find_k_value : ∃ k : ℝ, quadratic_roots_complex_and_diff k ∧ k = 4 * √22 :=
  sorry

end find_k_value_l377_377860


namespace amount_after_two_years_is_correct_l377_377938

-- Define the initial conditions
def present_value : ℝ := 70400
def first_year_increase_rate : ℝ := 1 / 8
def second_year_increase_rate : ℝ := 1 / 6

-- Define the theoretical final amount calculation
def amount_after_first_year (initial : ℝ) : ℝ :=
  initial + (first_year_increase_rate * initial)

def amount_after_second_year (after_first_year : ℝ) : ℝ :=
  after_first_year + (second_year_increase_rate * after_first_year)

-- State the theorem
theorem amount_after_two_years_is_correct :
  amount_after_second_year (amount_after_first_year present_value) = 92400 :=
by
  sorry

end amount_after_two_years_is_correct_l377_377938


namespace iterated_f_result_l377_377994

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p*x + q

theorem iterated_f_result (p q : ℝ) 
  (h1 : ∀ x ∈ set.Icc (2 : ℝ) (4 : ℝ), |f x p q| ≤ 1/2) :
  iter 2017 (λ x, f x p q) ((5 - real.sqrt 11) / 2) = real.of_rat 4.16 :=
sorry

end iterated_f_result_l377_377994


namespace sequence_n_value_l377_377861

theorem sequence_n_value (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) (h3 : a n = 2008) : n = 670 :=
by
 sorry

end sequence_n_value_l377_377861


namespace sum_imaginary_powers_l377_377057

theorem sum_imaginary_powers :
  (∑ k in Finset.range 2003, (k + 1) * (Complex.I ^ (k + 1))) = -1000 + 997 * Complex.I := 
by 
  sorry

end sum_imaginary_powers_l377_377057


namespace chair_arrangements_l377_377157

theorem chair_arrangements :
  let total_chairs := 10
  let unique_positions := total_chairs + 1
  let stool_positions := choose total_chairs 3
  unique_positions * stool_positions = 1320 :=
by
  sorry

end chair_arrangements_l377_377157


namespace expression_meaningful_if_not_three_l377_377575

-- Definition of meaningful expression
def meaningful_expr (x : ℝ) : Prop := (x ≠ 3)

theorem expression_meaningful_if_not_three (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ meaningful_expr x := by
  sorry

end expression_meaningful_if_not_three_l377_377575


namespace project_completion_l377_377373

theorem project_completion (x : ℕ) :
  (21 - x) * (1 / 12 : ℚ) + x * (1 / 30 : ℚ) = 1 → x = 15 :=
by
  sorry

end project_completion_l377_377373


namespace smallest_q_difference_l377_377983

theorem smallest_q_difference (p q : ℕ) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_fraction1 : 3 * q < 5 * p)
  (h_fraction2 : 5 * p < 6 * q)
  (h_smallest : ∀ r s : ℕ, 0 < s → 3 * s < 5 * r → 5 * r < 6 * s → q ≤ s) :
  q - p = 3 :=
by
  sorry

end smallest_q_difference_l377_377983


namespace volume_difference_l377_377059

theorem volume_difference (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
                          (h : (a * b * c).round = 2017) :
  let V_max := ((a + 0.5).round) * ((b + 0.5).round) * ((c + 0.5).round),
      V_min := ((a - 0.5).round) * ((b - 0.5).round) * ((c - 0.5).round)
  in V_max - V_min = 7174 :=
by
  -- Proof to be filled in
  sorry

end volume_difference_l377_377059


namespace sequence_2017th_term_l377_377829

def sequence : ℕ → ℕ
| 0     := 2
| 1     := 3
| (n+2) := (sequence n * sequence (n+1)) % 10

theorem sequence_2017th_term : sequence 2016 = 2 :=
by
  sorry

end sequence_2017th_term_l377_377829


namespace trajectory_equation_fixed_point_A_l377_377932

-- Problem 1
theorem trajectory_equation (x y : ℝ) (P M : ℝ × ℝ)
    (P_def : P = (x, y)) (M_def : M = (x, -4))
    (circle_through_origin : ∃ (C : set (ℝ × ℝ)), C = {Q | dist Q P + dist Q M = dist P M} ∧ (0, 0) ∈ C) :
    x^2 = 4 * y :=
sorry

-- Problem 2
theorem fixed_point_A'B (k x₁ y₁ x₂ y₂ : ℝ) (l_eq : y₁ = k * x₁ - 4 ∧ y₂ = k * x₂ - 4)
    (trajectory_A : x₁^2 = 4 * y₁) (trajectory_B : x₂^2 = 4 * y₂)
    (A'_def : ∃ A' : ℝ × ℝ, A' = (-x₁, y₁)) :
    ∃ (fixed : ℝ × ℝ), fixed = (0, 4) ∧ line_through A' (x₂, y₂) :=
sorry

end trajectory_equation_fixed_point_A_l377_377932


namespace angle_B_l377_377922

theorem angle_B (a b c : ℝ) (h : a^2 + c^2 - b^2 = sqrt 3 * a * c) : 
  ∃ B : ℝ, B = π / 6 ∧ cos B = (a^2 + c^2 - b^2) / (2 * a * c) :=
by
  use π / 6
  split
  · rfl
  · rw [h, cos_pi_div_six]
  sorry -- Skipping the proof with sorry

end angle_B_l377_377922


namespace square_area_l377_377639

theorem square_area (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 7) (h4 : y2 = 4) :
  ∃ A : ℝ, A = 20 :=
by
  let s := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 / 2)
  let A := s * s
  have h5 : A = 20 := by sorry
  use A
  exact h5

end square_area_l377_377639


namespace count_diff_of_squares_l377_377554

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377554


namespace geometric_sequence_sixth_term_correct_l377_377948

noncomputable def geometric_sequence_sixth_term (a₁ a₉ : ℝ) (r : ℝ) (h₁ : a₁ = 4) (h₂ : a₉ = 78732) (h₃ : a₉ = a₁ * r^8) : ℝ :=
  a₁ * r^5
  
theorem geometric_sequence_sixth_term_correct : geometric_sequence_sixth_term 4 78732 3 (by rfl) (by rfl) (by norm_num) = 972 :=
  sorry

end geometric_sequence_sixth_term_correct_l377_377948


namespace parallel_lines_l377_377501

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end parallel_lines_l377_377501


namespace pencils_left_l377_377702

theorem pencils_left (initial_pencils : ℕ := 79) (pencils_taken : ℕ := 4) : initial_pencils - pencils_taken = 75 :=
by
  sorry

end pencils_left_l377_377702


namespace least_integer_satisfies_inequality_l377_377282

theorem least_integer_satisfies_inequality : ∃ x : ℤ, (abs (3 * (x:ℤ)^2 - 2 * (x:ℤ) + 5) ≤ 29) ∧ ∀ y : ℤ, (abs (3 * (y:ℤ)^2 - 2 * (y:ℤ) + 5) ≤ 29) → x ≤ y :=
begin
  sorry
end

end least_integer_satisfies_inequality_l377_377282


namespace inequality1_inequality2_l377_377890

def f (x : ℝ) (a : ℝ) := (1 / 2) * |x - a|

theorem inequality1 (a : ℝ) (x : ℝ) (h : a = 3) : 
  (|x - 1 / 2| + f x a ≥ 2) ↔ (x ≤ 0 ∨ x ≥ 2) :=
by
  sorry

theorem inequality2 (a : ℝ) : 
  (∀ x, (1 / 2) ≤ x ∧ x ≤ 1 → (|x - 1 / 2| + f x a ≤ x)) ↔ (0 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end inequality1_inequality2_l377_377890


namespace highest_x_value_satisfies_equation_l377_377810

theorem highest_x_value_satisfies_equation:
  ∃ x, x ≤ 4 ∧ (∀ x1, x1 ≤ 4 → x1 = 4 ↔ (15 * x1^2 - 40 * x1 + 18) / (4 * x1 - 3) + 7 * x1 = 9 * x1 - 2) :=
by
  sorry

end highest_x_value_satisfies_equation_l377_377810


namespace diff_of_squares_count_l377_377514

theorem diff_of_squares_count : 
  { n : ℤ | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b : ℤ, a^2 - b^2 = n) }.count = 1500 := 
by
  sorry

end diff_of_squares_count_l377_377514


namespace no_factors_l377_377427

-- Define the polynomial P(x) = x^4 + 3x^2 + 8
def P (x : ℝ) : ℝ := x^4 + 3*x^2 + 8

-- Define the possible factor options
def A (x : ℝ) : ℝ := x^2 + 4
def B (x : ℝ) : ℝ := x + 2
def C (x : ℝ) : ℝ := x^2 - 4
def D (x : ℝ) : ℝ := x^2 - x - 2

-- Statement to prove that none of the options A, B, C, D are factors of P
theorem no_factors : ¬(∃ p : ℝ → ℝ, P = λ x, A x * p x) ∧
                     ¬(∃ p : ℝ → ℝ, P = λ x, B x * p x) ∧
                     ¬(∃ p : ℝ → ℝ, P = λ x, C x * p x) ∧
                     ¬(∃ p : ℝ → ℝ, P = λ x, D x * p x) :=
sorry

end no_factors_l377_377427


namespace question_1_question_2_l377_377886

/-- 
For the function f(x) = (1/2) * x^2 - m * ln(x), prove that f is increasing on 
the interval (1/2, +∞) if and only if m ≤ 1/4.
-/
theorem question_1 (m : ℝ) : 
  (∀ x, x > 1 / 2 → (1 / 2) * x^2 - m * Real.log x) → m ≤ 1 / 4 :=
sorry

/--
When m = 2, prove that the maximum value of f(x) = (1/2) * x^2 - 2 * ln(x) on the 
interval [1, e] is (e^2 - 4) / 2 and the minimum value is 1 - ln 2.
-/
theorem question_2 : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → 
    (1 / 2) * x^2 - 2 * Real.log x ≤ (Real.exp 1^2 - 4) / 2 ∧
    (1 / 2) * x^2 - 2 * Real.log x ≥ 1 - Real.log 2) :=
sorry

end question_1_question_2_l377_377886


namespace variance_scaling_half_l377_377069

variable (D : ℝ → ℝ)
variable (X : ℝ)
variable (a : ℝ)

-- Given condition
axiom D_X : D X = 2

-- Property of the variance operator
axiom D_scaling : ∀ (a : ℝ) (X : ℝ), D (a * X) = a^2 * D X

-- Proof statement
theorem variance_scaling_half : D (1/2 * X) = 1/2 :=
by
  rw [D_scaling, D_X]
  norm_num

end variance_scaling_half_l377_377069


namespace prism_width_l377_377910

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end prism_width_l377_377910


namespace at_least_three_points_in_circle_l377_377437

noncomputable def point_in_circle (p : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
(dist p c) ≤ r

theorem at_least_three_points_in_circle (points : Fin 51 → (ℝ × ℝ)) (side_length : ℝ) (circle_radius : ℝ)
  (h_side_length : side_length = 1) (h_circle_radius : circle_radius = 1 / 7) : 
  ∃ (c : ℝ × ℝ), ∃ (p1 p2 p3 : Fin 51), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    point_in_circle (points p1) c circle_radius ∧ 
    point_in_circle (points p2) c circle_radius ∧ 
    point_in_circle (points p3) c circle_radius :=
sorry

end at_least_three_points_in_circle_l377_377437


namespace coin_flip_probability_difference_l377_377318

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377318


namespace extreme_point_range_a_l377_377247

def f (x : ℝ) (a : ℝ) : ℝ :=
  2^x * Real.log2Exp - 2 * Real.log x - a * x + 3

theorem extreme_point_range_a :
  ∃ a : ℝ, 0 < a ∧ a < 3 ∧ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ is_local_max (f x a) :=
sorry

end extreme_point_range_a_l377_377247


namespace same_terminal_side_l377_377737

theorem same_terminal_side : 
  let θ1 := 23 * Real.pi / 3
  let θ2 := 5 * Real.pi / 3
  (∃ k : ℤ, θ1 - 2 * k * Real.pi = θ2) :=
sorry

end same_terminal_side_l377_377737


namespace number_of_odd_divisors_lt_100_l377_377130

theorem number_of_odd_divisors_lt_100 : 
  (∃! (n : ℕ), n < 100 ∧ ∃! (k : ℕ), n = k * k) = 9 :=
sorry

end number_of_odd_divisors_lt_100_l377_377130


namespace find_f_prime_e_l377_377874

-- Define the function f(x) and its derivative
variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ} (h_deriv : ∀ x, deriv f x = f' x)

-- Define the condition given in the problem
def f_condition (x : ℝ) : Prop := f x = 2 * x * f' e + real.log x

-- State the theorem that needs to be proved
theorem find_f_prime_e (h_f_cond : ∀ x, f_condition x) : f' e = -1 / e :=
by
  sorry

end find_f_prime_e_l377_377874


namespace coin_flip_probability_difference_l377_377317

theorem coin_flip_probability_difference :
  let p := 1 / 2,
  p_3 := (Nat.choose 4 3) * (p ^ 3) * (p ^ 1),
  p_4 := p ^ 4
  in abs (p_3 - p_4) = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l377_377317


namespace belindas_age_l377_377712

theorem belindas_age (T B : ℕ) (h1 : T + B = 56) (h2 : B = 2 * T + 8) (h3 : T = 16) : B = 40 :=
by
  sorry

end belindas_age_l377_377712


namespace tan_sum_relation_l377_377152

theorem tan_sum_relation (a b c : ℝ) (A B C : ℝ) (h_cosB : cos B = 4 / 5) (h_geom_seq : b^2 = a * c) 
  (h_angles : A + B + C = π) 
  (h_sides : a = sin A ∧ b = sin B ∧ c = sin C):
  (1 / tan A) + (1 / tan C) = 5 / 3 := 
by sorry

end tan_sum_relation_l377_377152


namespace largest_n_less_50000_multiple_of_seven_l377_377281

theorem largest_n_less_50000_multiple_of_seven :
  ∃ n : ℕ, n < 50000 ∧ (3 * (n - 3) ^ 2 - 4 * n + 28) % 7 = 0 ∧
  ∀ m : ℕ, m < 50000 ∧ (3 * (m - 3) ^ 2 - 4 * m + 28) % 7 = 0 → m ≤ n :=
begin
  use 49999,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  sorry
end

end largest_n_less_50000_multiple_of_seven_l377_377281


namespace no_real_roots_l377_377424

noncomputable def polynomial (p : ℝ) (x : ℝ) : ℝ :=
  x^4 + 4 * p * x^3 + 6 * x^2 + 4 * p * x + 1

theorem no_real_roots (p : ℝ) :
  (p > -Real.sqrt 5 / 2) ∧ (p < Real.sqrt 5 / 2) ↔ ¬(∃ x : ℝ, polynomial p x = 0) := by
  sorry

end no_real_roots_l377_377424


namespace probability_diff_colors_l377_377274

theorem probability_diff_colors (n : ℕ) (colors : Fin n): 
  (n = 4) → 
  (¬ ∃ (i : Fin n) (j : Fin n), i ≠ j) → 
  (@Set.univ (Fin n) (4 * 4 = 16) ∨ 4 * 3 = 12 → ℚ) ↔ 
  (Probability diff colors: finset (partial_sums → set.length.succ → list.fin (partial_split Fin.take) 4 → ℚ)) :
  3 / 4 := 
  by 
  -- Sorry statement to skip the proof
  sorry

end probability_diff_colors_l377_377274


namespace diff_squares_count_l377_377549

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377549


namespace pentagonal_pyramid_no_zero_sum_l377_377950

-- Define the pentagonal pyramid edges
def pentagonalPyramidEdges : Type := {e // e ∈ fin 10}

-- Base vectors sum to zero
axiom base_vectors_sum_zero : ∀ (base_vectors : fin 5 → ℝ^3), ∑ i in finset.univ, base_vectors i = 0

-- Side vectors have a height component
axiom side_vectors_height : ∀ (side_vectors : fin 5 → ℝ), ∃ h : ℝ, (∀ i, side_vectors i = h) ∧ h ≠ 0

-- Prove that the total vector sum cannot be zero
theorem pentagonal_pyramid_no_zero_sum : 
  ∀ (direction : pentagonalPyramidEdges → ℝ^3), 
  ∑ e in finset.univ, direction e ≠ 0 :=
by
  intros direction
  sorry

end pentagonal_pyramid_no_zero_sum_l377_377950


namespace time_difference_l377_377739

theorem time_difference (speed_Xanthia speed_Molly book_pages : ℕ) (minutes_in_hour : ℕ) :
  speed_Xanthia = 120 ∧ speed_Molly = 40 ∧ book_pages = 360 ∧ minutes_in_hour = 60 →
  (book_pages / speed_Molly - book_pages / speed_Xanthia) * minutes_in_hour = 360 := by
  sorry

end time_difference_l377_377739


namespace number_of_integer_solutions_l377_377826

theorem number_of_integer_solutions :
  {x : ℤ // x^6 - 75 * x^4 + 1000 * x^2 - 6000 < 0}.card = 12 :=
sorry

end number_of_integer_solutions_l377_377826


namespace axis_of_symmetry_l377_377136

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) :
  ∀ y : ℝ, (∃ x₁ x₂ : ℝ, y = f x₁ ∧ y = f x₂ ∧ (x₁ + x₂) / 2 = 2) :=
by
  sorry

end axis_of_symmetry_l377_377136


namespace proof_l377_377452

inductive BallColor
| Red
| Black

def is_exactly_one_black (draw : List BallColor) : Prop :=
  draw.count BallColor.Black = 1

def is_exactly_two_black (draw : List BallColor) : Prop :=
  draw.count BallColor.Black = 2

noncomputable def are_mutually_exclusive_not_complementary : Prop :=
  let bag := [BallColor.Red, BallColor.Red, BallColor.Black, BallColor.Black]
  let draws : List (List BallColor) := bag.combinations 2
  let event1 := draws.filter is_exactly_one_black
  let event2 := draws.filter is_exactly_two_black
  event1 ∩ event2 = [] ∧ event1 ∪ event2 ≠ draws

theorem proof : are_mutually_exclusive_not_complementary := sorry

end proof_l377_377452


namespace triangle_conditions_l377_377153

-- Define the conditions
def angle_B := (60 : ℝ)
def side_b := (4 : ℝ)

-- Theorem statement
theorem triangle_conditions 
  {a c : ℝ} 
  (h_c1 : c = real.sqrt 3 → ∀ C : ℝ, ∃! C)
  (h_c2 : (∃ a1 a2 : ℝ, a1 ≠ a2 ∧ ∃ h : ℝ, height_from_side a c B AC = 3 * real.sqrt 3)
  (h_c3 : a + c ≠ 9) :
  (¬ h_c1 ∧ h_c2 ∧ h_c3) :=
by sorry

end triangle_conditions_l377_377153


namespace contrapositive_of_x_squared_eq_one_l377_377245

theorem contrapositive_of_x_squared_eq_one (x : ℝ) 
  (h : x^2 = 1 → x = 1 ∨ x = -1) : (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1 :=
by
  sorry

end contrapositive_of_x_squared_eq_one_l377_377245


namespace number_of_integers_as_difference_of_squares_l377_377525

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377525


namespace max_perfect_squares_l377_377656

theorem max_perfect_squares (a b : ℕ) (h_d : a ≠ b) :
  2 ≤ card (filter is_square [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)]) :=
begin
  sorry
end

end max_perfect_squares_l377_377656


namespace g_translation_correct_g_monotonic_decreasing_l377_377882

def f (x : ℝ) : ℝ := 1 / x
def g (x : ℝ) : ℝ := (x / (x - 1))

theorem g_translation_correct :
  ∀ x, g x = (1 / (x - 1)) + 1 := 
sorry

theorem g_monotonic_decreasing (a b : ℝ) (h1 : a < b) :
  ((a < 1) ∨ (1 < a ∧ 1 < b)) → g a > g b := 
sorry

end g_translation_correct_g_monotonic_decreasing_l377_377882


namespace number_of_integers_as_difference_of_squares_l377_377530

theorem number_of_integers_as_difference_of_squares : 
  { n | 1 ≤ n ∧ n ≤ 2000 ∧ (∃ a b, n = a^2 - b^2 ∧ 0 ≤ a ∧ 0 ≤ b) }.card = 1500 := 
sorry

end number_of_integers_as_difference_of_squares_l377_377530


namespace find_probability_of_B_l377_377455

variable (P : Event → ℚ)

variables (A B : Event)
variable (H1 : P A = 3/5)
variable (H2 : P (A ∧ B) / P A = 1/2)
variable (H3 : P (¬(A ∧ B ¬)) / P (¬A) = 2/3)

theorem find_probability_of_B (H1 : P A = 3/5) 
    (H2 : P (A ∧ B) / P A = 1/2) 
    (H3 : P (¬(A ∧ B ¬)) / P (¬A) = 2/3) :
    P B = 13/30 := 
sorry

end find_probability_of_B_l377_377455


namespace sqrt_D_irrational_l377_377969

theorem sqrt_D_irrational (x : ℤ) : 
  let a := 2 * x,
      b := 2 * x + 2,
      c := a + b,
      D := a^2 + b^2 + c^2 in
  ¬ ∃ (r : ℚ), r^2 = D := 
by
  sorry

end sqrt_D_irrational_l377_377969


namespace vincent_die_problem_l377_377727

theorem vincent_die_problem :
  ∃ (r s t : ℚ)
    (h₀ : r > 0)
    (h₁ : s > 0)
    (h₂ : t > 0)
    (h₃ : ¬ ∃ (a b : ℕ) (h : b > 1), t = a^b)
    (m n : ℕ) (h₄ : Nat.coprime m n)
    (E : ℚ)
    (h₅ : E = r - s * Real.log t / Real.log (6 : ℚ))
    (h₆ : r + s + t = m / n),
  100 * m + n = 13112 := 
sorry

end vincent_die_problem_l377_377727


namespace print_time_l377_377773

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end print_time_l377_377773


namespace positive_difference_of_probabilities_l377_377305

theorem positive_difference_of_probabilities :
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1 in
  let p2 := (1 / 2) ^ 4 in
  abs (p1 - p2) = 7 / 16 :=
by
  let p1 := (Nat.choose 4 3) * (1 / 2) ^ 3 * (1 / 2) ^ 1
  let p2 := (1 / 2) ^ 4
  have p1_calc: p1 = 1 / 2
  { rw [Nat.choose, _root_.choose, binomial_eq, algebraic_ident]
    sorry  -- The detailed proof of p1 = 1 / 2 will go here
  }
  have p2_calc: p2 = 1 / 16
  { rw [(1/2)^4]
    sorry  -- The detailed proof of p2 = 1 / 16 will go here
  }
  have diff_calc: abs (p1 - p2) = 7 / 16
  { rw [p1_calc, p2_calc]
    sorry  -- The detailed proof of abs (p1 - p2) = 7 / 16 will go here
  }
  exact diff_calc

end positive_difference_of_probabilities_l377_377305


namespace find_ratio_l377_377892

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def arithmetic_sequence_relation (a₁ a₂ a₃ : ℝ) :=
  (a₂ - a₁ = a₃ - a₂)

theorem find_ratio (a : ℕ → ℝ) (hq : ∀ n, 0 < a n)
  (h_geom : geometric_sequence a q)
  (h_arith : arithmetic_sequence_relation a 1 (1 / 2 * a 3) (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * sqrt 2 :=
sorry

end find_ratio_l377_377892


namespace positive_difference_probability_l377_377296

theorem positive_difference_probability (fair_coin : Prop) (four_flips : ℕ) (fair : fair_coin) (flips : four_flips = 4) :
  let p1 := (Nat.choose 4 3)*((1/2)^3)*((1/2)^1) in
  let p2 := (1/2)^4 in
  abs (p1 - p2) = 7/16 :=
by
  -- Definitions
  let p1 := (Nat.choose 4 3) * ((1/2)^3) * ((1/2)^1)
  let p2 := (1/2)^4
  have h : abs (p1 - p2) = 7 / 16
  sorry

end positive_difference_probability_l377_377296


namespace probability_not_B_given_A_l377_377465

noncomputable def PA : ℚ := 1/3
noncomputable def PB : ℚ := 1/4
noncomputable def P_A_given_B : ℚ := 3/4

def PnotB_given_A : ℚ := 1 - (P_A_given_B * PB / PA)

theorem probability_not_B_given_A (PA PB P_A_given_B : ℚ) 
  (hPA : PA = 1/3) (hPB : PB = 1/4) (hP_A_given_B : P_A_given_B = 3/4) : 
  PnotB_given_A = 7/16 :=
by
  rw [hPA, hPB, hP_A_given_B]
  simp [PnotB_given_A, PA, PB, P_A_given_B]
  sorry

end probability_not_B_given_A_l377_377465


namespace max_b_value_l377_377263

theorem max_b_value (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 10 :=
sorry

end max_b_value_l377_377263


namespace ellipse_eccentricity_and_foci_l377_377872

noncomputable def eccentricity_and_foci (a b : ℝ) :=
  let c := real.sqrt (a^2 - b^2) in
  (c / a, [(-c, 0), (c, 0)])

theorem ellipse_eccentricity_and_foci :
  ∀ (A B M : point) (a b c : ℝ) (h_ellipse : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 = 12)
    (h_A : A ≠ M) (h_B : B ≠ M) (h_slope_product : slope (line_through M A) * slope (line_through M B) = 1 / 4),
    eccentricity_and_foci 2 (real.sqrt 3) = (1 / 2, [(-1, 0), (1, 0)]) ∧ (∀ p : point, is_on_line p (line_through A B) → p = (-4, 0)) :=
begin
  sorry
end

end ellipse_eccentricity_and_foci_l377_377872


namespace michael_last_10_shots_successful_count_l377_377792

theorem michael_last_10_shots_successful_count :
  ∀ (InitialShots : ℕ) (InitialPercentage : ℚ) (AdditionalShots : ℕ) (FinalPercentage : ℚ),
    InitialShots = 50 →
    InitialPercentage = 0.6 →
    AdditionalShots = 10 →
    FinalPercentage = 0.62 →
    let initialSuccess = InitialPercentage * InitialShots in
    let finalTotalShots = InitialShots + AdditionalShots in
    let finalSuccess = FinalPercentage * finalTotalShots in
    (⌊finalSuccess⌋₊ - initialSuccess) = 7 := by
{
  intros InitialShots InitialPercentage AdditionalShots FinalPercentage
         InitialShots_eq InitialPercentage_eq AdditionalShots_eq FinalPercentage_eq,
  sorry
}

end michael_last_10_shots_successful_count_l377_377792


namespace f_lg_lg_3_l377_377093

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.cbrt x + 4

theorem f_lg_lg_3 (a b : ℝ) (h : f a b (Real.logb 10 3) = 5) :
  f a b (Real.logb 10 (Real.logb 10 3)) = 3 :=
sorry

end f_lg_lg_3_l377_377093


namespace probability_of_sum_being_6_l377_377343

noncomputable def prob_sum_6 : ℚ :=
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_being_6 :
  prob_sum_6 = 5 / 36 :=
by
  sorry

end probability_of_sum_being_6_l377_377343


namespace convex_hexagon_coloring_l377_377038

open Function

theorem convex_hexagon_coloring :
  ∃ (colors : Finset ℕ) (A B C D E F : ℕ),
  colors.card = 7 ∧
  A ∈ colors ∧
  B ∈ colors ∧
  C ∈ colors ∧
  D ∈ colors ∧
  E ∈ colors ∧
  F ∈ colors ∧
  A ≠ D ∧
  B ≠ E ∧
  C ≠ F ∧
  Finset.sizeof (Finset.image id (Finset.singleton A ∪ Finset.singleton B ∪ Finset.singleton C ∪ Finset.singleton D ∪ Finset.singleton E ∪ Finset.singleton F)) = 74088 :=
by sorry

end convex_hexagon_coloring_l377_377038


namespace triangle_area_128_l377_377961

theorem triangle_area_128 (ABC : Triangle) (R : ℝ) (r : ℝ)
    (hR : R = 17) (hr : r = 4) 
    (Γ : Circle) (Ω : Excircle) 
    (hΩ_reflection_tangent : reflection_Ω_over_BC_tangent_to_Γ ABC Ω) :
    (area ABC) = 128 :=
sorry

end triangle_area_128_l377_377961


namespace jon_training_sessions_l377_377174

theorem jon_training_sessions :
  let initial_speed := 80
  let final_speed := initial_speed * 120 / 100
  let speed_increase_per_week := 1
  let weeks_per_training_session := 4
  let total_training_sessions := (final_speed - initial_speed) / (speed_increase_per_week * weeks_per_training_session)
  total_training_sessions = 4 :=
by
  -- Definitions
  let initial_speed : ℝ := 80
  let final_speed : ℝ := initial_speed * 120 / 100
  let speed_increase_per_week : ℝ := 1
  let weeks_per_training_session : ℝ := 4
  let total_training_sessions : ℝ := (final_speed - initial_speed) / (speed_increase_per_week * weeks_per_training_session)
  have h: final_speed = 96 := by norm_num
  have h2: total_training_sessions = 4 := by
    calc total_training_sessions = (final_speed - initial_speed) / (speed_increase_per_week * weeks_per_training_session) : by rfl
    ... = (96 - 80) / (1 * 4) : by congr; norm_num
    ... = 16 / 4 : by norm_num
    ... = 4 : by norm_num
  exact h2

end jon_training_sessions_l377_377174


namespace walk_to_bus_stop_usual_time_l377_377362

variable (S : ℝ) -- assuming S is the usual speed, a positive real number
variable (T : ℝ) -- assuming T is the usual time, which we need to determine
variable (new_speed : ℝ := (4 / 5) * S) -- the new speed is 4/5 of usual speed
noncomputable def time_to_bus_at_usual_speed : ℝ := T -- time to bus stop at usual speed

theorem walk_to_bus_stop_usual_time :
  (time_to_bus_at_usual_speed S = 30) ↔ (S * (T + 6) = (4 / 5) * S * T) :=
by
  sorry

end walk_to_bus_stop_usual_time_l377_377362


namespace coin_flip_probability_difference_l377_377342

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377342


namespace intersection_A_B_l377_377470

variable {α : Type*} [linear_ordered_field α]

def A : set α := {x | x > 0}

def B : set α := {y | ∃ x, y = 2^x + 1}

theorem intersection_A_B : A ∩ B = {y | y > 1} :=
by {
  sorry
}

end intersection_A_B_l377_377470


namespace decimal_to_base_five_l377_377421

theorem decimal_to_base_five : 
  (2 * 5^3 + 1 * 5^1 + 0 * 5^2 + 0 * 5^0 = 255) := 
by
  sorry

end decimal_to_base_five_l377_377421


namespace linear_eq_implies_m_eq_1_l377_377867

theorem linear_eq_implies_m_eq_1 (x y m : ℝ) (h : 3 * (x ^ |m|) + (m + 1) * y = 6) (hm_abs : |m| = 1) (hm_ne_zero : m + 1 ≠ 0) : m = 1 :=
  sorry

end linear_eq_implies_m_eq_1_l377_377867


namespace diff_squares_count_l377_377548

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377548


namespace a_gt_b_l377_377062

theorem a_gt_b (n : ℕ) (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hn_ge_two : n ≥ 2)
  (ha_eq : a^n = a + 1) (hb_eq : b^(2*n) = b + 3 * a) : a > b :=
by
  sorry

end a_gt_b_l377_377062


namespace divides_expression_l377_377138

noncomputable def largest_divisor (y : ℤ) (h : Even y) : ℤ :=
  let expr := (8 * y + 4) * (8 * y + 8) * (4 * y + 6) * (4 * y + 2)
  96

theorem divides_expression (y : ℤ) (h : Even y) : 96 ∣ (8 * y + 4) * (8 * y + 8) * (4 * y + 6) * (4 * y + 2) :=
by { exact dvd_trans (by refl) sorry }

end divides_expression_l377_377138


namespace determine_f_a_plus_f_b_l377_377098

noncomputable def f (x : Real) (b : Real) : Real := 2016 * x^3 - 5 * x + b + 2

def is_odd_function (f : Real → Real) : Prop := ∀ x: Real, f (-x) = -f x

theorem determine_f_a_plus_f_b :
  ∀ (a b : Real), is_odd_function (f ∘ b) ∧ (∀ x : Real, (a - 4 ≤ x ∧ x ≤ 2 * a - 2) → (a = 2 ∧ b = -2)) →
  (f a b + f b b = 0) :=
by
  sorry

end determine_f_a_plus_f_b_l377_377098


namespace shaded_region_perimeter_is_12_l377_377164

-- Define the circumference of each circle
def circumference : ℝ := 24

-- Define the angle formed at the centers of touching circles
def touching_angle : ℝ := 90

-- Define the perimeter calculation as per the given conditions
def perimeter_of_shaded_region (c : ℝ) (angle : ℝ) : ℝ :=
  2 * (c / 4)

-- State the theorem to prove the perimeter of the shaded region is 12
theorem shaded_region_perimeter_is_12 :
  perimeter_of_shaded_region circumference touching_angle = 12 :=
by 
  sorry

end shaded_region_perimeter_is_12_l377_377164


namespace people_happy_correct_l377_377399

-- Define the size and happiness percentage of an institution.
variables (size : ℕ) (happiness_percentage : ℚ)

-- Assume the size is between 100 and 200.
axiom size_range : 100 ≤ size ∧ size ≤ 200

-- Assume the happiness percentage is between 0.6 and 0.95.
axiom happiness_percentage_range : 0.6 ≤ happiness_percentage ∧ happiness_percentage ≤ 0.95

-- Define the number of people made happy at an institution.
def people_made_happy (size : ℕ) (happiness_percentage : ℚ) : ℚ := 
  size * happiness_percentage

-- Theorem stating that the number of people made happy is as expected.
theorem people_happy_correct : 
  ∀ (size : ℕ) (happiness_percentage : ℚ), 
  100 ≤ size → size ≤ 200 → 
  0.6 ≤ happiness_percentage → happiness_percentage ≤ 0.95 → 
  people_made_happy size happiness_percentage = size * happiness_percentage := 
by 
  intros size happiness_percentage hsize1 hsize2 hperc1 hperc2
  unfold people_made_happy
  sorry

end people_happy_correct_l377_377399


namespace extremum_implies_a_max_min_values_in_interval_l377_377477

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x

-- Problem statements
theorem extremum_implies_a (a : ℝ) (h : ∃ x, x = 1 ∧ (f' x = 0)) : a = 3 := by
  sorry

theorem max_min_values_in_interval (a : ℝ) (h : a = 3) : 
  ∃ (M m : ℝ), 
    (∀ x ∈ Icc (0:ℝ) (2:ℝ), f x 3 ≤ M) ∧ 
    (∀ x ∈ Icc (0:ℝ) (2:ℝ), f x 3 ≥ m) ∧
    M = 2 ∧ m = -2 := by
  sorry

end extremum_implies_a_max_min_values_in_interval_l377_377477


namespace smallest_nat_number_l377_377766

theorem smallest_nat_number (x : ℕ) 
  (h1 : ∃ z : ℕ, x + 3 = 5 * z) 
  (h2 : ∃ n : ℕ, x - 3 = 6 * n) : x = 27 := 
sorry

end smallest_nat_number_l377_377766


namespace trig_inequality_l377_377185
open Real

theorem trig_inequality (α β γ x y z : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : x + y + z = 0) :
  y * z * (sin α)^2 + z * x * (sin β)^2 + x * y * (sin γ)^2 ≤ 0 := 
sorry

end trig_inequality_l377_377185


namespace triangle_isosceles_y_value_l377_377597

theorem triangle_isosceles_y_value
  (PR QS X : Type) {P Q R S : PR}
  (right_angle_PQX : ∠ PQX = 90)
  (angle_QPX : ∠ QPX = 62)
  (isosceles_RXS : RX = SX) :
  ∃ y : ℝ, y = 76 :=
by
  sorry

end triangle_isosceles_y_value_l377_377597


namespace length_of_second_race_l377_377155

theorem length_of_second_race :
  ∀ (V_A V_B V_C T T' L : ℝ),
  (V_A * T = 200) →
  (V_B * T = 180) →
  (V_C * T = 162) →
  (V_B * T' = L) →
  (V_C * T' = L - 60) →
  (L = 600) :=
by
  intros V_A V_B V_C T T' L h1 h2 h3 h4 h5
  sorry

end length_of_second_race_l377_377155


namespace count_diff_of_squares_l377_377556

def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

theorem count_diff_of_squares :
  (Finset.filter is_diff_of_squares (Finset.Icc 1 2000)).card = 1500 := 
sorry

end count_diff_of_squares_l377_377556


namespace minimal_fraction_difference_l377_377980

theorem minimal_fraction_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧ (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < (2 : ℚ) / 3 ∧
  (∀ r s : ℕ, (3 : ℚ) / 5 < (r : ℚ) / s ∧ (r : ℚ) / s < (2 : ℚ) / 3 → 0 < s → q ≤ s) ∧
  q - p = 3 :=
begin
  sorry
end

end minimal_fraction_difference_l377_377980


namespace rate_of_increase_in_10th_year_l377_377007

noncomputable def p (t : ℝ) : ℝ := 1.05^t

theorem rate_of_increase_in_10th_year :
  deriv p 10 = 1.05^10 * real.log 1.05 :=
by
  sorry

end rate_of_increase_in_10th_year_l377_377007


namespace coin_flip_probability_difference_l377_377337

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l377_377337


namespace divisibility_of_f_by_cubic_factor_l377_377645

noncomputable def f (x : ℂ) (m n : ℕ) : ℂ := x^(3 * m + 2) + (-x^2 - 1)^(3 * n + 1) + 1

theorem divisibility_of_f_by_cubic_factor (m n : ℕ) : ∀ x : ℂ, x^2 + x + 1 = 0 → f x m n = 0 :=
by
  sorry

end divisibility_of_f_by_cubic_factor_l377_377645


namespace sin_A_value_c_value_l377_377581

-- Define constants and triangle properties
constant A B C : Type
constant angle_C : Real := 2 * Real.pi / 3
constant a : Real := 6
constant triangle_area : Real := 3 * Real.sqrt 3

-- Problem (I)
theorem sin_A_value (c : Real) (h1 : c = 14) : sin_A = (3 * Real.sqrt 3) / 14 :=
by
  sorry

-- Problem (II)
theorem c_value (area : Real) (h1 : area = triangle_area) : c = 2 * Real.sqrt 13 :=
by
  sorry

end sin_A_value_c_value_l377_377581


namespace factorize_m_sq_minus_one_l377_377836

theorem factorize_m_sq_minus_one (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := 
by
  sorry

end factorize_m_sq_minus_one_l377_377836


namespace math_problem_proof_l377_377148

noncomputable def x := (∑ n in (Finset.Icc 10 30), n)
def y := 11

theorem math_problem_proof : x + y = 431 → x = 420 :=
by
  intro h
  sorry

end math_problem_proof_l377_377148


namespace distance_between_circle_centers_l377_377498

theorem distance_between_circle_centers
  (R r d : ℝ)
  (h1 : R = 7)
  (h2 : r = 4)
  (h3 : d = 5 + 1)
  (h_total_diameter : 5 + 8 + 1 = 14)
  (h_radius_R : R = 14 / 2)
  (h_radius_r : r = 8 / 2) : d = 6 := 
by sorry

end distance_between_circle_centers_l377_377498


namespace total_players_l377_377359

theorem total_players 
  (cricket_players : ℕ) (hockey_players : ℕ)
  (football_players : ℕ) (softball_players : ℕ)
  (h_cricket : cricket_players = 12)
  (h_hockey : hockey_players = 17)
  (h_football : football_players = 11)
  (h_softball : softball_players = 10)
  : cricket_players + hockey_players + football_players + softball_players = 50 :=
by sorry

end total_players_l377_377359


namespace area_of_triangle_AOB_l377_377600

-- Definitions of points A and B in polar coordinates
def A : ℝ × ℝ := (3, Real.pi / 3)
def B : ℝ × ℝ := (-4, 7 * Real.pi / 6)

-- Statement to prove the area of triangle AOB is 3
theorem area_of_triangle_AOB : 
  let O : ℝ × ℝ := (0, 0) in
  let area (A B : ℝ × ℝ) : ℝ := 
    0.5 * (A.1 * B.1 * Real.sin (A.2 - B.2)) in
  area A B = 3 :=
by 
  sorry

end area_of_triangle_AOB_l377_377600


namespace x_1000_is_2002_l377_377044

open Nat

def isComposite (n : ℕ) : Prop := ¬ prime n ∧ n > 1

noncomputable def x : ℕ → ℕ
| 1 => 4
| 2 => 6
| (n + 1) => if n = 0 then 6 else
              have T := 2 * x n - x (n - 1)
              let m := T + 1
              Nat.find (λ k => isComposite (m + k))

theorem x_1000_is_2002 : x 1000 = 2002 :=
  sorry

end x_1000_is_2002_l377_377044


namespace arithmetic_sequence_sum_10_l377_377595

variable (a : ℕ → ℕ)

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ a1 d, (∀ n, a n = a1 + n * d)

def sum_up_to (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range n, a i

theorem arithmetic_sequence_sum_10
  (a : ℕ → ℕ)
  (h1 : a 0 + a 1 = 5)
  (h2 : a 3 + a 4 = 23)
  (arith_seq : arithmetic_sequence a) :
  sum_up_to 10 a = 145 := sorry

end arithmetic_sequence_sum_10_l377_377595


namespace log_expr_value_range_of_m_l377_377752

-- Definition for first problem
def log_expr := log 3 (427 / 3) + log 10 25 + log 10 4 + log 7 (7 ^ 2) + (log 2 3) * (log 3 4)

-- Proof that the logarithmic expression equals 23/4
theorem log_expr_value : log_expr = 23 / 4 := 
by sorry

-- Definitions for second problem
def A (x : Real) := 1/32 ≤ 2 ^ (-x) ∧ 2 ^ (-x) ≤ 4
def B (x m : Real) := m - 1 < x ∧ x < 2 * m + 1
def A_def := { x : Real | A x } = Set.Icc (-2 : Real) 5 

-- Proof that the range of values for m is (-∞, -2] ∪ [-1, 2]
theorem range_of_m (m : Real) : 
    (∀ x, A x → B x m → A x) ↔ (m ∈ Set.Iic (-2) ∨ m ∈ Set.Icc (-1, 2)) := 
by sorry

end log_expr_value_range_of_m_l377_377752


namespace smallest_number_div_by_225_with_digits_0_1_l377_377735

theorem smallest_number_div_by_225_with_digits_0_1 :
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ n ∧ (∀ m : ℕ, (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ m → n ≤ m) ∧ n = 11111111100 :=
sorry

end smallest_number_div_by_225_with_digits_0_1_l377_377735


namespace major_axis_length_l377_377002

def length_of_major_axis 
  (tangent_x : ℝ) (f1 : ℝ × ℝ) (f2 : ℝ × ℝ) : ℝ :=
  sorry

theorem major_axis_length 
  (hx_tangent : (4, 0) = (4, 0)) 
  (foci : (4, 2 + 2 * Real.sqrt 2) = (4, 2 + 2 * Real.sqrt 2) ∧ 
         (4, 2 - 2 * Real.sqrt 2) = (4, 2 - 2 * Real.sqrt 2)) :
  length_of_major_axis 4 
  (4, 2 + 2 * Real.sqrt 2) (4, 2 - 2 * Real.sqrt 2) = 4 :=
sorry

end major_axis_length_l377_377002


namespace quadratic_has_real_solution_l377_377492

theorem quadratic_has_real_solution (a b c : ℝ) : 
  ∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0 ∨ 
           x^2 + (b - c) * x + (c - a) = 0 ∨ 
           x^2 + (c - a) * x + (a - b) = 0 :=
  sorry

end quadratic_has_real_solution_l377_377492


namespace bread_to_butter_ratio_l377_377276

noncomputable def cheese_price := 5
noncomputable def butter_price := 0.8 * cheese_price
noncomputable def tea_price := 10
noncomputable def total_paid := 21
noncomputable def bread_price := total_paid - (butter_price + cheese_price + tea_price)

theorem bread_to_butter_ratio : bread_price / butter_price = 1 / 2 := 
by 
  have h_total : total_paid = bread_price + butter_price + cheese_price + tea_price := sorry
  have h_cheese : cheese_price = 5 := sorry
  have h_butter : butter_price = 4 := sorry
  have h_bread : bread_price = 2 := sorry
  have h_ratio : bread_price / butter_price = 1 / 2 := sorry
  exact h_ratio

end bread_to_butter_ratio_l377_377276


namespace option_a_correct_option_d_correct_l377_377609

noncomputable def b : ℕ → ℕ
| 1 := 1
| 2 := 3
| (n+1) + 1 := b (n+1) + b (n)

theorem option_a_correct : (∑ i in finset.range (100/2 + 1), b (2 * i)) = b 101 - 1 :=
sorry

theorem option_d_correct : (∑ i in finset.range (100), b (i + 1) ^ 2) = b 100 * b 101 - 2 :=
sorry

end option_a_correct_option_d_correct_l377_377609


namespace solution_set_bx2_ax_1_l377_377133

noncomputable def quad_ineq_solution {a b : ℝ} :=
  ∀ x : ℝ, (x^2 + a*x + b < 0) ↔ (2 < x ∧ x < 3)

theorem solution_set_bx2_ax_1 {a b : ℝ} (h : quad_ineq_solution):
  ∀ x : ℝ, (b*x^2 + a*x + 1 > 0) ↔ (x < 1/3 ∨ x > 1/2) :=
by
  sorry

end solution_set_bx2_ax_1_l377_377133


namespace cookout_bun_packs_l377_377400

theorem cookout_bun_packs
  (total_friends : ℕ) (non_meat_eaters : ℕ) (non_bread_eaters : ℕ) 
  (gluten_free_friends : ℕ) (nut_allergy_friends : ℕ) 
  (burgers_per_friend : ℕ) (bun_pack_regular : ℕ) 
  (bun_pack_gluten_free : ℕ) (bun_pack_nut_free : ℕ) : 
  total_friends = 35 →
  non_meat_eaters = 7 →
  non_bread_eaters = 4 →
  gluten_free_friends = 3 →
  nut_allergy_friends = 1 →
  burgers_per_friend = 3 →
  bun_pack_regular = 15 →
  bun_pack_gluten_free = 6 →
  bun_pack_nut_free = 5 →
  let meat_eating_friends := total_friends - non_meat_eaters in
  let regular_bread_eaters := meat_eating_friends - non_bread_eaters in
  let regular_buns_needed := regular_bread_eaters * burgers_per_friend in
  let packs_of_regular_buns := (regular_buns_needed + bun_pack_regular - 1) / bun_pack_regular in
  let gluten_free_buns_needed := gluten_free_friends * burgers_per_friend in
  let packs_of_gluten_free_buns := (gluten_free_buns_needed + bun_pack_gluten_free - 1) / bun_pack_gluten_free in
  let nut_free_buns_needed := nut_allergy_friends * burgers_per_friend in
  let packs_of_nut_free_buns := (nut_free_buns_needed + bun_pack_nut_free - 1) / bun_pack_nut_free in
  packs_of_regular_buns = 5 ∧ packs_of_gluten_free_buns = 2 ∧ packs_of_nut_free_buns = 1 :=
by 
  intros; 
  sorry

end cookout_bun_packs_l377_377400


namespace cube_edge_adjacency_l377_377011

def is_beautiful (f: Finset ℕ) := 
  ∃ a b c d, f = {a, b, c, d} ∧ a = b + c + d

def cube_is_beautiful (faces: Finset (Finset ℕ)) :=
  ∃ t1 t2 t3, t1 ∈ faces ∧ t2 ∈ faces ∧ t3 ∈ faces ∧
  is_beautiful t1 ∧ is_beautiful t2 ∧ is_beautiful t3

def valid_adjacency (v: ℕ) (n1 n2 n3: ℕ) := 
  v = 6 ∧ ((n1 = 2 ∧ n2 = 3 ∧ n3 = 5) ∨
           (n1 = 2 ∧ n2 = 3 ∧ n3 = 7) ∨
           (n1 = 3 ∧ n2 = 5 ∧ n3 = 7))

theorem cube_edge_adjacency : 
  ∀ faces: Finset (Finset ℕ), 
  ∃ v n1 n2 n3, 
  (v = 6 ∧ (valid_adjacency v n1 n2 n3)) ∧
  cube_is_beautiful faces := 
by
  -- Entails the proof, which is not required here
  sorry

end cube_edge_adjacency_l377_377011


namespace find_k_l377_377064

noncomputable def sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  3 * 2^n + k

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a n * a (n + 2) = (a (n + 1))^2

theorem find_k
  (a : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a n = sequence_sum (n + 1) k - sequence_sum n k)
  (h2 : geometric_sequence a) :
  k = -3 :=
  by sorry

end find_k_l377_377064


namespace percentage_free_lunch_is_40_l377_377009

-- Definitions for conditions
def total_cost : Real := 210
def total_students : Nat := 50
def cost_per_paying_student : Real := 7

-- Calculate the number of paying students
def number_of_paying_students : Nat := total_cost / cost_per_paying_student

-- Calculate the percentage of students receiving free lunch
def percentage_free_lunch : Real := (total_students - number_of_paying_students) / total_students * 100

-- Proof statement
theorem percentage_free_lunch_is_40 :
  percentage_free_lunch = 40 := sorry

end percentage_free_lunch_is_40_l377_377009


namespace modular_arithmetic_proof_l377_377978

open Nat

theorem modular_arithmetic_proof (m : ℕ) (h0 : 0 ≤ m ∧ m < 37) (h1 : 4 * m ≡ 1 [MOD 37]) :
  (3^m)^4 ≡ 27 + 3 [MOD 37] :=
by
  -- Although some parts like modular inverse calculation or finding specific m are skipped,
  -- the conclusion directly should reflect (3^m)^4 ≡ 27 + 3 [MOD 37]
  -- Considering (3^m)^4 - 3 ≡ 24 [MOD 37] translates to the above statement
  sorry

end modular_arithmetic_proof_l377_377978


namespace set_intersection_l377_377496

noncomputable def U := set.univ
noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B := {x : ℝ | ∃ (y : ℝ), y = real.log(1 - x)}
noncomputable def comp_B := {x : ℝ | ¬(∃ (y : ℝ), y = real.log(1 - x))}
noncomputable def intersection := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem set_intersection : A ∩ comp_B = intersection :=
by sorry

end set_intersection_l377_377496


namespace geometric_progression_sum_of_squares_l377_377857

theorem geometric_progression_sum_of_squares :
  ∃ (a r : ℕ), 
    0 < a ∧ a < 100 ∧ 
    0 < r ∧ r < 100 ∧ 
    (let terms := [a, a * r, a * r^2, a * r^3, a * r^4] in 
     list.sum terms = 201 ∧ 
     list.sum (terms.filter (λ t, ∃ n : ℕ, n^2 = t)) = 52) :=
sorry

end geometric_progression_sum_of_squares_l377_377857


namespace sum_of_exterior_angles_of_hexagon_l377_377695

theorem sum_of_exterior_angles_of_hexagon (hex : polygon 6) : sum_of_exterior_angles hex = 360 :=
sorry

end sum_of_exterior_angles_of_hexagon_l377_377695


namespace ratio_of_pens_to_notebooks_is_5_to_4_l377_377687

theorem ratio_of_pens_to_notebooks_is_5_to_4 (P N : ℕ) (hP : P = 50) (hN : N = 40) :
  (P / Nat.gcd P N) = 5 ∧ (N / Nat.gcd P N) = 4 :=
by
  -- Proof goes here
  sorry

end ratio_of_pens_to_notebooks_is_5_to_4_l377_377687


namespace solve_for_x_l377_377666

theorem solve_for_x : 
  let x := (sqrt ((8^2 : ℝ) + 15^2)) / (sqrt (25 + 36)) in
  x = (17 : ℝ) / (sqrt 61) := 
by
  let x := (sqrt ((8^2 : ℝ) + 15^2)) / (sqrt (25 + 36))
  have : sqrt ((8^2 : ℝ) + 15^2) = 17 := by norm_num
  have : sqrt (25 + 36) = sqrt 61 := by norm_num
  rw [this, this]
  norm_num
  sorry

end solve_for_x_l377_377666


namespace abc_quad_is_cyclic_l377_377371

open EuclideanGeometry

-- Definitions used in conditions
variables {A B C D K L M N : Point}
variables [convex_quadrilateral ABCD]
variables [point_on_seg K AB]
variables [point_on_seg L BC]
variables [point_on_seg M CD]
variables [point_on_seg N DA]
variables [trajectory_closed KLMN]

-- The final theorem statement to prove quadrilateral inscriptibility
theorem abc_quad_is_cyclic 
  (hL : reflect L BC)
  (hM : reflect M CD)
  (hN : reflect N DA)
  (hK : from_to_traj K L M N)
  : cyclic_quad ABCD :=
begin
  sorry
end

end abc_quad_is_cyclic_l377_377371


namespace telescoping_sum_ge_l377_377963

theorem telescoping_sum_ge(
  (a : ℕ → ℝ) (n : ℕ)
  (positive : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0)
  (product_one : ∏ i in finset.range(n + 1), a i = 1)
) : 
∑ i in finset.range(n + 1), a i / (∏ j in finset.range(i + 1), 1 + a j) ≥ (2 ^ n - 1) / 2 ^ n :=
sorry

end telescoping_sum_ge_l377_377963


namespace find_speed_of_second_train_l377_377720

def speed_of_second_train (d : ℝ) (s1 : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  (d - s1 * t1) / t2

theorem find_speed_of_second_train :
  let d := 200
  let s1 := 20
  let t1 := 5
  let t2 := 4
  speed_of_second_train d s1 t1 t2 = 25 := by
  sorry

end find_speed_of_second_train_l377_377720


namespace DansAgeCalculation_l377_377821

theorem DansAgeCalculation (D x : ℕ) (h1 : D = 8) (h2 : D + 20 = 7 * (D - x)) : x = 4 :=
by
  sorry

end DansAgeCalculation_l377_377821


namespace diff_squares_count_l377_377552

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377552


namespace infinite_AP_composite_numbers_infinite_AP_perfect_squares_l377_377742

theorem infinite_AP_composite_numbers {a d : ℕ} (h : a > 1) :
  ∃ᶠ n in filter.at_top, ¬ nat.prime (a + n * d) :=
sorry

theorem infinite_AP_perfect_squares {a d : ℕ} :
  (∀ n, ¬ ∃ k, (a + n * d) = k^2) ∨ (∃ᶠ n in filter.at_top, ∃ k, (a + n * d) = k^2) :=
sorry

end infinite_AP_composite_numbers_infinite_AP_perfect_squares_l377_377742


namespace least_coeff_of_x_l377_377917

theorem least_coeff_of_x (P : ℕ → ℕ) (h : ∀ i, P i ∈ {0, 1, 2, 3, 4, 5}) (hP_6 : P 6 = 2013) :
    (∃ a1, (∀ j, a1 = P 1 → a1 ≤ P j) ∧ a1 = 5) := sorry

end least_coeff_of_x_l377_377917


namespace regular_bike_wheels_eq_two_l377_377407

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end regular_bike_wheels_eq_two_l377_377407


namespace max_gcd_sequence_l377_377025

open Int

theorem max_gcd_sequence :
  ∀ n : ℕ, gcd (99 + n^2) (99 + (n + 1)^2) = 1 :=
by
  sorry

end max_gcd_sequence_l377_377025


namespace polynomial_remainder_l377_377056

theorem polynomial_remainder :
  ∀ (x : ℝ), (x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 5) % (x^2 - 3 * x + 2) = (24 * x - 25) :=
by
  sorry

end polynomial_remainder_l377_377056


namespace semicircle_radius_l_l377_377405

open Real

theorem semicircle_radius_l (R l : ℝ) (hR : 0 < R) :
  (∃ C D : ℝ, C ≠ D ∧ C^2 = R^2 ∧ D^2 = R^2 - C^2 ∧
  (C + sqrt ((D^2 - C^2) / 2R) - R = l)) → l ≤ 4 * R :=
by
  sorry

end semicircle_radius_l_l377_377405


namespace total_distance_walked_l377_377796

theorem total_distance_walked :
  ∀ (speed1 speed2 : ℕ) (time1 time2 timeBreak totalTime : ℝ),
  speed1 = 2 →
  speed2 = 3 →
  time1 = 1.5 →
  timeBreak = 0.5 →
  totalTime = 3.5 →
  speed1 * time1 + speed2 * (totalTime - timeBreak - time1) = 7.5 :=
by
  intros speed1 speed2 time1 time2 timeBreak totalTime
  intros hs1 hs2 ht1 htBreak htTotal
  rw [hs1, hs2, ht1, htBreak, htTotal]
  simp
  norm_num
  sorry

end total_distance_walked_l377_377796


namespace sum_fractions_equal_l377_377347

theorem sum_fractions_equal :
  (\frac{(10 + 20 + 30 + 40)}{10} + \frac{10}{(10 + 20 + 30 + 40)}) = 10.1 := 
by
  let A := 10 + 20 + 30 + 40
  have hA : A = 100 := by norm_num
  have hB : (A / 10) = 10 := by
    rw [hA]
    norm_num
  have hC : (10 / A) = 0.1 := by
    rw [hA]
    norm_num
  rw [←hB, ←hC]
  norm_num

end sum_fractions_equal_l377_377347


namespace union_A_B_complement_intersect_B_intersection_sub_C_l377_377181

-- Define set A
def A : Set ℝ := {x | -5 < x ∧ x < 1}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 8}

-- Define set C with variable parameter a
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Problem (1): Prove A ∪ B = { x | -5 < x < 8 }
theorem union_A_B : A ∪ B = {x | -5 < x ∧ x < 8} := 
by sorry

-- Problem (1): Prove (complement R A) ∩ B = { x | 1 ≤ x < 8 }
theorem complement_intersect_B : (Aᶜ) ∩ B = {x | 1 ≤ x ∧ x < 8} :=
by sorry

-- Problem (2): If A ∩ B ⊆ C, prove a ≥ 1
theorem intersection_sub_C (a : ℝ) (h : A ∩ B ⊆ C a) : 1 ≤ a :=
by sorry

end union_A_B_complement_intersect_B_intersection_sub_C_l377_377181


namespace smallest_q_difference_l377_377982

theorem smallest_q_difference (p q : ℕ) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_fraction1 : 3 * q < 5 * p)
  (h_fraction2 : 5 * p < 6 * q)
  (h_smallest : ∀ r s : ℕ, 0 < s → 3 * s < 5 * r → 5 * r < 6 * s → q ≤ s) :
  q - p = 3 :=
by
  sorry

end smallest_q_difference_l377_377982


namespace complex_expr_evaluation_l377_377101

def z : ℂ := 1 + complex.i

theorem complex_expr_evaluation : (z^2 - 2 * z) / (1 - z) = -2 * complex.i := by
  sorry

end complex_expr_evaluation_l377_377101


namespace max_perfect_squares_l377_377659

theorem max_perfect_squares (a b : ℕ) (h : a ≠ b) : 
  let prod1 := a * (a + 2),
      prod2 := a * b,
      prod3 := a * (b + 2),
      prod4 := (a + 2) * b,
      prod5 := (a + 2) * (b + 2),
      prod6 := b * (b + 2) in
  (prod1.is_square ℕ + prod2.is_square ℕ + prod3.is_square ℕ + prod4.is_square ℕ + prod5.is_square ℕ + prod6.is_square ℕ) ≤ 2 := 
  sorry

end max_perfect_squares_l377_377659


namespace angle_A_is_60_degrees_triangle_area_l377_377476

-- Define the basic setup for the triangle and its angles
variables (a b c : ℝ) -- internal angles of the triangle ABC
variables (B C : ℝ) -- sides opposite to angles b and c respectively

-- Given conditions
axiom equation_1 : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a
axiom perimeter_condition : a + b + c = 8
axiom circumradius_condition : ∃ R : ℝ, R = Real.sqrt 3

-- Question 1: Prove the measure of angle A is 60 degrees
theorem angle_A_is_60_degrees (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a) : 
  a = 60 :=
sorry

-- Question 2: Prove the area of triangle ABC
theorem triangle_area (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a)
(h_perimeter : a + b + c = 8) (h_circumradius : ∃ R : ℝ, R = Real.sqrt 3) :
  ∃ S : ℝ, S = 4 * Real.sqrt 3 / 3 :=
sorry

end angle_A_is_60_degrees_triangle_area_l377_377476


namespace values_of_a_and_b_f_monotonically_increasing_l377_377080

-- Initial definitions and conditions
def f (a b x : ℝ) : ℝ := (a * x + b) / (x^2 + 2)

-- Given conditions
variables (a b : ℝ)

-- Condition f(0) = 0
def condition1 : Prop := f a b 0 = 0

-- Condition f(1/3) = 3/19
def condition2 : Prop := f a b (1/3) = 3/19

-- Verify the values of a and b
theorem values_of_a_and_b : condition1 a b → condition2 a b → a = 1 ∧ b = 0 := sorry

-- Redefine the function with determined a and b
def f_simplified (x : ℝ) := f 1 0 x

-- Prove monotonicity on the interval (-√2, √2)
theorem f_monotonically_increasing :
  ∀ x₁ x₂ : ℝ, -real.sqrt 2 < x₁ → x₁ < x₂ → x₂ < real.sqrt 2 → f_simplified x₁ < f_simplified x₂ := sorry

end values_of_a_and_b_f_monotonically_increasing_l377_377080


namespace diff_squares_count_l377_377550

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377550


namespace correct_conclusion_l377_377791

theorem correct_conclusion (x : ℝ) (hx : x > 1/2) : -2 * x + 1 < 0 :=
by
  -- sorry placeholder
  sorry

end correct_conclusion_l377_377791


namespace union_of_P_and_Q_l377_377494

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem union_of_P_and_Q : (P ∪ Q) = {x | -1 < x ∧ x < 3} := by
  -- skipping the proof
  sorry

end union_of_P_and_Q_l377_377494


namespace equal_sides_pentagon_angle_l377_377936

theorem equal_sides_pentagon_angle (P Q R S T : Point) 
  (h1 : convex_pentagon P Q R S T)
  (h2 : dist P Q = dist Q R)
  (h3 : dist Q R = dist R S)
  (h4 : dist R S = dist S T)
  (h5 : dist S T = dist T P)
  (h6 : ∠ P R T = 1 / 2 * ∠ Q R S) : 
  ∠ P R T = 30 := 
sorry

end equal_sides_pentagon_angle_l377_377936


namespace Matilda_fathers_chocolate_bars_l377_377212

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l377_377212


namespace max_surface_area_of_rectangular_solid_on_sphere_l377_377911

noncomputable def max_surface_area_rectangular_solid (a b c : ℝ) :=
  2 * a * b + 2 * a * c + 2 * b * c

theorem max_surface_area_of_rectangular_solid_on_sphere :
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 36 → max_surface_area_rectangular_solid a b c ≤ 72) :=
by
  intros a b c h
  sorry

end max_surface_area_of_rectangular_solid_on_sphere_l377_377911


namespace total_students_proof_l377_377000

variables (M P n : ℕ)

def total_students (M P n total : ℕ) : Prop :=
  (0.2 * M = n) ∧ (0.25 * P = n) ∧ (M = 5 * n) ∧ (P = 4 * n) ∧ (M + P - n + 2 = total) ∧ (20 < total) ∧ (total < 30)

theorem total_students_proof (M P n : ℕ) (hn : 0.2 * M = n) (hp : 0.25 * P = n)  
  (hm : M = 5 * n) (hp' : P = 4 * n) (h : 20 < 8 * n + 2) (h' : 8 * n + 2 < 30) :
  exists total, total_students (M) (P) (n) (total) := 
begin
  use (8 * n + 2),
  split,
  { exact hn, },
  split,
  { exact hp, },
  split,
  { exact hm, },
  split,
  { exact hp', },
  split,
  { rw [hm, hp', ←sub_add_eq_add_sub, add_assoc],
    ring, },
  split,
  { exact h, },
  { exact h', },
end

end total_students_proof_l377_377000


namespace solve_cos_theta_l377_377380

def cos_theta_proof (v1 v2 : ℝ × ℝ) (θ : ℝ) : Prop :=
  let dot_product := (v1.1 * v2.1 + v1.2 * v2.2)
  let norm_v1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let norm_v2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  let cos_theta := dot_product / (norm_v1 * norm_v2)
  cos_theta = 43 / Real.sqrt 2173

theorem solve_cos_theta :
  cos_theta_proof (4, 5) (2, 7) (43 / Real.sqrt 2173) :=
by
  sorry

end solve_cos_theta_l377_377380


namespace prove_f_iterative_value_l377_377992

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem prove_f_iterative_value (p q : ℝ) (h1 : ∀ x, 2 ≤ x → x ≤ 4 → -1/2 ≤ f x p q ∧ f x p q ≤ 1/2) :
  let initial_value := (5 - Real.sqrt 11) / 2 in
  let res := (iterate (f p q) 2017 initial_value) in
  res = (5 + Real.sqrt 11) / 2 :=
by
  sorry

end prove_f_iterative_value_l377_377992


namespace diff_squares_count_l377_377547

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377547


namespace diff_squares_count_l377_377546

/-- The number of integers between 1 and 2000 (inclusive) that can be expressed 
as the difference of the squares of two nonnegative integers is 1500. -/
theorem diff_squares_count : (1 ≤ n ∧ n ≤ 2000 → ∃ a b : ℤ, n = a^2 - b^2) = 1500 := 
by
  sorry

end diff_squares_count_l377_377546


namespace find_ad_bc_l377_377198

variables {R : Type*} [Field R]
variables (a b c d : R)
def A : Matrix (Fin 2) (Fin 2) R :=
  !![a, b; 
     c, d]

def alpha1 : Vector R (Fin 2) := !![1, -1]
def alpha2 : Vector R (Fin 2) := !![3, 2]

def lambda1 : R := -1
def lambda2 : R := 4

theorem find_ad_bc : 
  (A.mulVec alpha1 = lambda1 • alpha1) → 
  (A.mulVec alpha2 = lambda2 • alpha2) → 
  a * d - b * c = -4 := 
by 
  intros h1 h2 
  sorry

end find_ad_bc_l377_377198


namespace vector_magnitude_5sqrt5_l377_377503

theorem vector_magnitude_5sqrt5 
  (y : ℝ)
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ := (-2, y))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  ∥(a.1 + 3 * b.1, a.2 + 3 * b.2)∥ = 5 * Real.sqrt 5 := 
by
  sorry

end vector_magnitude_5sqrt5_l377_377503


namespace probability_0_to_2_l377_377082

noncomputable def X : Type := sorry

axiom normal_distribution : ProbTheory.Distribution StdReal.StdNormal where
  mean zero_var : ∀ {X : Type}, ∀ {ε : Type} [MetricSpace X] [TopologicalSpace.SecondCountableTopology X] [LinearOrder X] [OrderClosedTopology X] (μ : ProbabilityMeasure X) (M : ℝ), 0 = 0
  ident_normal : ∀ {σ : ℝ}, 0 = 0 -- This serves as a placeholder

axiom P_X_gt_neg2_eq_0_9 : ProbTheory.Prob (X > -2) = 0.9

theorem probability_0_to_2 : ProbTheory.Prob (0 ≤ X ≤ 2) = 0.4 :=
by
  sorry

end probability_0_to_2_l377_377082


namespace redemption_start_day_l377_377630

-- Definitions for the conditions in the problem
structure CouponRedemption :=
  (coupons : Nat)
  (interval_days : Nat)
  (max_day : Nat := 7)
  (closed_day : Fin max_day := 6)  -- Saturday is represented as 6

-- Defining the main theorem
theorem redemption_start_day (n : Nat) (cr : CouponRedemption) : 
  (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) →
  ∃ start_day : Fin cr.max_day, start_day.val = 0 := 
by
  sorry

end redemption_start_day_l377_377630


namespace series_identity_l377_377750

theorem series_identity (n : ℕ) : 
  1 - ∑ m in Finset.range n, (-1)^(m + 1) * (1 / (m + 1)) * Nat.choose n m = 1 / (n + 1) := 
sorry

end series_identity_l377_377750


namespace value_of_x_l377_377458

theorem value_of_x (x y : ℝ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end value_of_x_l377_377458


namespace find_n_l377_377438

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2023 [MOD 8] ∧ n = 1 :=
by
  use 1
  split
  linarith
  split
  linarith
  split
  norm_num
  sorry

end find_n_l377_377438


namespace factorize_expr_l377_377043

theorem factorize_expr (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := 
by
  sorry

end factorize_expr_l377_377043


namespace train_length_l377_377396

open Real

/--
A train of a certain length can cross an electric pole in 30 sec with a speed of 43.2 km/h.
Prove that the length of the train is 360 meters.
-/
theorem train_length (t : ℝ) (v_kmh : ℝ) (length : ℝ) 
  (h_time : t = 30) 
  (h_speed_kmh : v_kmh = 43.2) 
  (h_length : length = v_kmh * (t * (1000 / 3600))) : 
  length = 360 := 
by
  -- skip the actual proof steps
  sorry

end train_length_l377_377396


namespace correct_addition_l377_377755

-- Define the initial conditions and goal
theorem correct_addition (x : ℕ) : (x + 26 = 61) → (x + 62 = 97) :=
by
  intro h
  -- Proof steps would be provided here
  sorry

end correct_addition_l377_377755


namespace print_time_l377_377774

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end print_time_l377_377774


namespace num_elements_in_B_inter_C_l377_377997

open Set

-- Defining sets A, B, and C
def A : Set ℕ := { x | 1 ≤ x ∧ x ≤ 99 }
def B : Set ℕ := { x | ∃ y ∈ A, x = 2 * y }
def C : Set ℕ := { x | 2 * x ∈ A }

-- Theorem to prove
theorem num_elements_in_B_inter_C : 
  (Finset.card ((A.filter (λ x, 2 * x ∈ A)).image (λ x, 2 * x)) : ℕ) ∩ 
  (Finset.card (A.filter (λ x, 2 * x ∈ A)) : ℕ) = 24 := 
sorry

end num_elements_in_B_inter_C_l377_377997
