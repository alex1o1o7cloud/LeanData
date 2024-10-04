import Mathlib
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.TimesContDiff
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Mod
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinations
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Sequence
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SolveByElim

namespace expected_flips_is_610_l609_609629

namespace AliceCoinFlip

-- Define conditions
def fair_coin : ℕ → bool → Prop
| 0, _ := true
| (n+1), b := fair_coin n (if (classical.some (randomBoolChoice ())) = tt then b = ff else b = tt)

def reads_fiction (b : bool) : Prop := b = tt
def reads_nonfiction (b : bool) : Prop := b = ff

def flips_again (b1 b2 : bool) : Prop := b1 = tt ∧ b2 = ff

def E : ℝ := 5 / 3

-- Define expected number of flips in a leap year
def expected_flips_leap_year : ℝ := E * 366

-- Define the theorem
theorem expected_flips_is_610 : expected_flips_leap_year = 610 := by
  sorry

end AliceCoinFlip

end expected_flips_is_610_l609_609629


namespace proof_solution_l609_609013

noncomputable def total_arrangements : ℕ := 8!

noncomputable def wilma_paul_together : ℕ := 7! * 2!

noncomputable def xavier_avoids_wilma_paul : ℕ := 6! * 3! - 7! * 2!

noncomputable def restricted_arrangements : ℕ := wilma_paul_together + xavier_avoids_wilma_paul

noncomputable def acceptable_arrangements : ℕ := total_arrangements - restricted_arrangements

theorem proof_solution : acceptable_arrangements = 36000 := by
  unfold total_arrangements wilma_paul_together xavier_avoids_wilma_paul restricted_arrangements acceptable_arrangements
  sorry

end proof_solution_l609_609013


namespace sin_value_l609_609364

theorem sin_value (α : ℝ) (h : Real.sin (π / 6 - α) = √2 / 3) : 
  Real.sin (2 * α + π / 6) = 5 / 9 :=
  sorry

end sin_value_l609_609364


namespace count_total_wheels_l609_609133

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end count_total_wheels_l609_609133


namespace delta_cheaper_than_gamma_l609_609668

theorem delta_cheaper_than_gamma (n : ℕ) :
  (40 + 7 * n < 11 * n) → n ≥ 11 :=
by
  intro h
  have : 40 < 4 * n, from (show 40 + 7 * n < 11 * n, from h).sub_self 7n
  sorry

end delta_cheaper_than_gamma_l609_609668


namespace tan_alpha_add_tan_beta_l609_609331

namespace Trigonometry

variables {α β : ℝ}

theorem tan_alpha_add_tan_beta (h1 : sin α + sin β = (4 / 5) * sqrt 2) 
                               (h2 : cos α + cos β = (4 / 5) * sqrt 3) :
  tan α + tan β = (sin (α + β)) / (cos α * cos β) :=
by sorry

end Trigonometry

end tan_alpha_add_tan_beta_l609_609331


namespace grandpa_uncle_ratio_l609_609065

def initial_collection := 150
def dad_gift := 10
def mum_gift := dad_gift + 5
def auntie_gift := 6
def uncle_gift := auntie_gift - 1
def final_collection := 196
def total_cars_needed := final_collection - initial_collection
def other_gifts := dad_gift + mum_gift + auntie_gift + uncle_gift
def grandpa_gift := total_cars_needed - other_gifts

theorem grandpa_uncle_ratio : grandpa_gift = 2 * uncle_gift := by
  sorry

end grandpa_uncle_ratio_l609_609065


namespace alex_total_cost_l609_609593

noncomputable def calculate_cost (base_cost per_text_cost per_minute_cost num_texts num_minutes_over) : ℕ :=
  base_cost + (num_texts * per_text_cost / 100) + (num_minutes_over * per_minute_cost / 100)

theorem alex_total_cost : 
  calculate_cost 25 8 15 150 480 = 109 :=
by
  sorry

end alex_total_cost_l609_609593


namespace volume_inequality_find_min_k_l609_609660

noncomputable def cone_volume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

noncomputable def cylinder_volume (R h : ℝ) : ℝ :=
    let r := (R * h) / Real.sqrt (R^2 + h^2)
    Real.pi * r^2 * h

noncomputable def k_value (R h : ℝ) : ℝ := (R^2 + h^2) / (3 * h^2)

theorem volume_inequality (R h : ℝ) (h_pos : R > 0 ∧ h > 0) : 
    cone_volume R h ≠ cylinder_volume R h := by sorry

theorem find_min_k (R h : ℝ) (h_pos : R > 0 ∧ h > 0) (k : ℝ) :
    cone_volume R h = k * cylinder_volume R h → k = (R^2 + h^2) / (3 * h^2) := by sorry

end volume_inequality_find_min_k_l609_609660


namespace sequence_count_654_l609_609044

def T : set (ℤ × ℤ × ℤ) := 
  {t | ∃ b1 b2 b3, t = (b1, b2, b3) ∧ 1 ≤ b1 ∧ b1 ≤ 15 ∧ 1 ≤ b2 ∧ b2 ≤ 15 ∧ 1 ≤ b3 ∧ b3 ≤ 15}

def generates_sequence (b : ℕ → ℤ) :=
  ∀ n ≥ 4, b n = (b (n-1)) * (abs (b (n-2) - b (n-3)))

def valid_sequence (b : ℕ → ℤ) :=
  ∃ n ≤ 10, b n = 0

theorem sequence_count_654 : 
  (∃ seq : ℕ → ℤ, (∃ b1 b2 b3 ∈ {b | 1 ≤ b ∧ b ≤ 15}, seq 1 = b1 ∧ seq 2 = b2 ∧ seq 3 = b3) ∧ generates_sequence seq ∧ valid_sequence seq) = 654 := 
sorry

end sequence_count_654_l609_609044


namespace correct_propositions_l609_609664

def proposition1 : Prop := ∀ (s : finset ℕ) (n ∈ s), 
  (_ : ∀ x ∈ s, x ≠ n → P(x) = Q(x)), 
  (random_sample s).member n

def proposition2 : Prop :=
  let f := λ x : ℝ, x^5 + 2 * x^3 - x^2 + 3 * x + 1,
  Qin_Algorithm(f, 1) = 2

def proposition3 : Prop := 
  (λ (m : ℝ), -3 < m ∧ m < 5) → 
  (∃ k₁ k₂ : ℝ, k₁ = 5 - m ∧ k₂ = m + 3 ∧ k₁ > 0 ∧ k₂ > 0 ∧ k₁ ≠ k₂) 

def proposition4 : Prop := 
  ∃ a : ℝ, ∀ x : ℝ, x^2 + 2*x + a < 0

def true_propositions : Prop := 
  proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4

theorem correct_propositions : true_propositions := 
begin
  sorry -- Proof is omitted
end

end correct_propositions_l609_609664


namespace scientific_notation_of_280000_l609_609017

theorem scientific_notation_of_280000 : 280000 = 2.8 * 10^5 := 
begin
  sorry
end

end scientific_notation_of_280000_l609_609017


namespace ideal_number_for_extended_sequence_l609_609058

variable {S : ℕ → ℕ} -- Define a sequence S such that S(n) gives the partial sum of the sequence's first n terms.

def T (n : ℕ) : ℕ := (∑ i in Finset.range n, S (i + 1)) / n

theorem ideal_number_for_extended_sequence (h : T 500 = 2004) : T 501 = 2012 :=
by 
  sorry -- Proof will be provided here

end ideal_number_for_extended_sequence_l609_609058


namespace jeongyeon_height_correct_l609_609852

def Jeongyeon_height (Joohyun_height : ℝ) : ℝ :=
  1.06 * Joohyun_height

theorem jeongyeon_height_correct (Joohyun_height : ℝ) (h : Joohyun_height = 134.5) : 
  Jeongyeon_height Joohyun_height = 142.57 :=
by
  rw [Jeongyeon_height, h]
  norm_num
  sorry

end jeongyeon_height_correct_l609_609852


namespace combined_population_l609_609958

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end combined_population_l609_609958


namespace c_value_for_local_maximum_l609_609791

def f (x c : ℝ) := x * (x - c) ^ 2

theorem c_value_for_local_maximum (c : ℝ) :
  (∃ f' f'' (f' (2 : ℝ) = 0 ∧ f'' (2 : ℝ) < 0), f' f'' := 
    (d/dx (f x c) : ℝ)).2 (2 : ℝ) < 0 → c = 6 :=
sorry

end c_value_for_local_maximum_l609_609791


namespace greatest_product_of_two_even_integers_whose_sum_is_300_l609_609565

theorem greatest_product_of_two_even_integers_whose_sum_is_300 :
  ∃ (x y : ℕ), (2 ∣ x) ∧ (2 ∣ y) ∧ (x + y = 300) ∧ (x * y = 22500) :=
by
  sorry

end greatest_product_of_two_even_integers_whose_sum_is_300_l609_609565


namespace value_of_f1_plus_g3_l609_609048

def f (x : ℝ) := 3 * x - 4
def g (x : ℝ) := x + 2

theorem value_of_f1_plus_g3 : f (1 + g 3) = 14 := by
  sorry

end value_of_f1_plus_g3_l609_609048


namespace scientific_notation_of_86_million_l609_609104

theorem scientific_notation_of_86_million :
  86000000 = 8.6 * 10^7 :=
sorry

end scientific_notation_of_86_million_l609_609104


namespace number_of_queens_on_chessboard_l609_609166

theorem number_of_queens_on_chessboard : 
  (finset.filter (λ n, ∃ k, k * k = n) (finset.range 65)).card = 8 :=
by 
  sorry

end number_of_queens_on_chessboard_l609_609166


namespace min_value_of_f_at_a_eq_3_8_one_zero_when_a_in_neg_1_to_0_range_of_a_for_two_zeros_l609_609794

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - Real.log x

-- Define the theorems based on the problem statements

theorem min_value_of_f_at_a_eq_3_8 : 
  f (3/8) 2 = - 1 / 2 - Real.log 2 :=
sorry

theorem one_zero_when_a_in_neg_1_to_0 (a : ℝ) (h : -1 ≤ a ∧ a ≤ 0) : 
  ∃! x : ℝ, 0 < x ∧ f a x = 0 :=
sorry

theorem range_of_a_for_two_zeros (h : ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  0 < a ∧ a < 1 :=
sorry

end min_value_of_f_at_a_eq_3_8_one_zero_when_a_in_neg_1_to_0_range_of_a_for_two_zeros_l609_609794


namespace sin_expression_l609_609306

variable α : ℝ
variable h1 : (Real.tan α + 1 / Real.tan α = 5 / 2)
variable h2 : (α > Real.pi / 4 ∧ α < Real.pi / 2)

theorem sin_expression : 
  Real.sin (2 * α - Real.pi / 4) = (7 * Real.sqrt 2) / 10 :=
by
  sorry

end sin_expression_l609_609306


namespace deposit_increases_l609_609897

theorem deposit_increases (X r s : ℝ) (hX : 0 < X) (hr : 0 ≤ r) (hs : s < 20) : 
  r > 100 * s / (100 - s) :=
by sorry

end deposit_increases_l609_609897


namespace part1_part2_l609_609350

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem part1 (x : ℝ) (h : f x 2 ≥ 2) : x ≤ 1/2 ∨ x ≥ 2.5 := by
  sorry

theorem part2 (a : ℝ) (h_even : ∀ x : ℝ, f (-x) a = f x a) : a = -1 := by
  sorry

end part1_part2_l609_609350


namespace shaded_wheel_percentage_l609_609609

noncomputable def path_length : ℝ := 14
noncomputable def num_stripes : ℕ := 7
noncomputable def stripe_length : ℝ := 1
noncomputable def wheel_radius : ℝ := 2
noncomputable def wheel_circumference : ℝ := 2 * Real.pi * wheel_radius
noncomputable def wheel_quarters : ℕ := 4
noncomputable def wheel_travel_distance : ℝ := wheel_circumference
-- Assume: The wheel makes exactly 1 complete revolution.

def shaded_on_shaded_contact_percentage : ℝ :=
  ((1 + (Real.pi - 3) + 1 + (3 * Real.pi - 9)) / wheel_circumference) * 100

theorem shaded_wheel_percentage :
  shaded_on_shaded_contact_percentage = 20 := by
  sorry

end shaded_wheel_percentage_l609_609609


namespace probability_one_common_number_approx_l609_609389

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l609_609389


namespace perpendicular_lines_necessary_condition_l609_609739

variables (α : Type _) [plane α] (a b l : Type _) [line a] [line b] [line l]

noncomputable def line_perpendicular_to_plane (l : Type _) [line l] (α : Type _) [plane α] : Prop :=
forall (x : Type _) [line x], x ⊆ α -> l ⊥ x

noncomputable def line_perpendicular_to_line (l a : Type _) [line l] [line a] : Prop :=
l ⊥ a

theorem perpendicular_lines_necessary_condition :
  ∀ (α : Type _) [plane α] (a b l : Type _) [line a] [line b] [line l],
    a ⊆ α → b ⊆ α →
    (line_perpendicular_to_line l a) →
    (line_perpendicular_to_line l b) →
    (line_perpendicular_to_plane l α) :=
by
  intros,
  sorry

end perpendicular_lines_necessary_condition_l609_609739


namespace product_of_consecutive_integers_l609_609884

theorem product_of_consecutive_integers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_less : a < b) :
  ∃ (x y : ℕ), x ≠ y ∧ x * y % (a * b) = 0 :=
by
  sorry

end product_of_consecutive_integers_l609_609884


namespace external_tangent_sphere_radius_l609_609994

theorem external_tangent_sphere_radius :
  ∃ R : ℝ, let O1 O2 O3 O4 O : ℝ × ℝ × ℝ :=
  (⟨0, 0, 0⟩ : ℝ × ℝ × ℝ, ⟨2, 0, 0⟩, ⟨1, sqrt 3, 0⟩, ⟨1, 1 / sqrt 3, sqrt (23 / 3)⟩, O),
  let R_candidates := [sqrt (1 / 3 + x ^ 2) - 1, sqrt 69 / 3 - x - 2] in
    ∀ x : ℝ, R ∈ R_candidates → R = (sqrt 69 - 7) / 5 :=
begin
  sorry
end

end external_tangent_sphere_radius_l609_609994


namespace final_salt_content_l609_609156

def initial_solution_weight : ℝ := 60
def initial_salt_content : ℝ := 0.20
def added_salt_weight : ℝ := 3

theorem final_salt_content :
  let initial_salt := initial_solution_weight * initial_salt_content,
      total_salt := initial_salt + added_salt_weight,
      new_solution_weight := initial_solution_weight + added_salt_weight,
      final_percentage := (total_salt / new_solution_weight) * 100 in
  abs (final_percentage - 23.81) < 0.01 :=
by { sorry }

end final_salt_content_l609_609156


namespace gift_package_combinations_l609_609603

theorem gift_package_combinations (wrapping_papers ribbons cards tags : ℕ) 
  (h_wrap: wrapping_papers = 10) 
  (h_ribbons: ribbons = 5) 
  (h_cards: cards = 5) 
  (h_tags: tags = 2) : 
  wrapping_papers * ribbons * cards * tags = 500 :=
by
  rw [h_wrap, h_ribbons, h_cards, h_tags]
  norm_num
  sorry

end gift_package_combinations_l609_609603


namespace ellipse_equation_l609_609749

noncomputable def ellipseProblem : Prop :=
  ∃ (a b : ℝ), ∃ (F1 F2 P : ℝ × ℝ), 
  (a > 0 ∧ b > 0 ∧ a > b ∧ a = 2 * b ∧ 
   (P ∈ { p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1 }) ∧ 
   dist P F1 * dist P F2 = 2 ∧ 
   (∃ t1 t2 : ℝ, P = (t1, sqrt (b ^ 2 * (1 - t1 ^ 2 / a ^ 2))) ∧ F1 = (-c, 0) ∧ F2 = (c, 0) ∧ 
   t1 * c = 0 ∧ t2 * c = 0)) ∧ 
  (a^2 = 4 ∧ b^2 = 1)

theorem ellipse_equation : ellipseProblem := sorry

end ellipse_equation_l609_609749


namespace inequality_relationship_l609_609785

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l609_609785


namespace value_x_plus_2y_plus_3z_l609_609322

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end value_x_plus_2y_plus_3z_l609_609322


namespace shortest_time_optimization_l609_609507

def total_workers : ℕ := 10
def total_products : ℕ := 50
def painting_time_per_product : ℕ := 10
def drying_time_per_product : ℕ := 5
def assembling_time_per_product : ℕ := 20

theorem shortest_time_optimization :
  ∃ (painters assemblers : ℕ), 
    painters + assemblers ≤ total_workers ∧ 
    painters = 3 ∧ 
    assemblers = 6 :=
begin
  use [3, 6],
  split,
  { exact nat.le_of_eq rfl },
  split; refl
end

end shortest_time_optimization_l609_609507


namespace cooking_time_l609_609183

theorem cooking_time :
  ∀ (total_potatoes cooked_potatoes time_per_potato : ℕ),
    total_potatoes = 15 →
    cooked_potatoes = 8 →
    time_per_potato = 9 →
    (total_potatoes - cooked_potatoes) * time_per_potato = 63 := 
by 
  intros total_potatoes cooked_potatoes time_per_potato ht hc hp
  rw [ht, hc, hp]
  sorry

end cooking_time_l609_609183


namespace profit_percentage_l609_609178

def cost_price : ℝ := 60
def selling_price : ℝ := 78

theorem profit_percentage : ((selling_price - cost_price) / cost_price) * 100 = 30 := 
by
  sorry

end profit_percentage_l609_609178


namespace value_of_a_l609_609345

/-- Given the binomial expression (√x + a/∛x)^n,
the sum of the binomial coefficients is 32,
and the constant term is 80.
Prove that a = 2. -/
theorem value_of_a (a x : ℝ) (n : ℕ)
  (h1 : (∀ k : ℕ, (n.choose k) * (√x)^ (n - k) * (a/(∛x))^k) = 32)
  (h2 : (n.choose 3 * a^3) = 80):
  a = 2 :=
by
  sorry

end value_of_a_l609_609345


namespace range_of_even_function_l609_609443

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem range_of_even_function :
  ∀ (a b : ℝ), (∀ x, f a b x = f a b (-x)) ∧ 1 + a = -2 ∧ b = 0 → 
  ∀ y, y ∈ set.range (f a b) ↔ -10 ≤ y ∧ y ≤ 2 :=
by
  sorry

end range_of_even_function_l609_609443


namespace billy_dishes_to_equal_time_l609_609649

-- These are conditions given in the problem
def minutes_sweeping_per_room : ℕ := 3
def minutes_dishes_per_dish : ℕ := 2
def minutes_laundry_per_load : ℕ := 9
def anna_rooms : ℕ := 10
def billy_loads : ℕ := 2

-- Calculations derived from the conditions
def anna_sweeping_time : ℕ := anna_rooms * minutes_sweeping_per_room
def billy_laundry_time : ℕ := billy_loads * minutes_laundry_per_load

-- Here's the theorem that proves the problem statement which is based on the final result from solution steps.
theorem billy_dishes_to_equal_time : ∃ (dishes : ℕ), (billy_laundry_time + dishes * minutes_dishes_per_dish = anna_sweeping_time) ∧ dishes = 6 :=
by
  use 6
  rw [anna_sweeping_time, billy_laundry_time, minutes_sweeping_per_room, minutes_laundry_per_load, minutes_dishes_per_dish]
  simp
  split
  · simp
  · refl

end billy_dishes_to_equal_time_l609_609649


namespace kylie_stamps_l609_609038

theorem kylie_stamps (K N : ℕ) (h1 : N = K + 44) (h2 : K + N = 112) : K = 34 :=
by
  sorry

end kylie_stamps_l609_609038


namespace initial_people_in_elevator_l609_609990

theorem initial_people_in_elevator (W n : ℕ) (avg_initial_weight avg_new_weight new_person_weight : ℚ)
  (h1 : avg_initial_weight = 152)
  (h2 : avg_new_weight = 151)
  (h3 : new_person_weight = 145)
  (h4 : W = n * avg_initial_weight)
  (h5 : W + new_person_weight = (n + 1) * avg_new_weight) :
  n = 6 :=
by
  sorry

end initial_people_in_elevator_l609_609990


namespace range_of_g_l609_609294

open Real

def g (t : ℝ) : ℝ := (t^2 + 3/4 * t) / (t^2 + 1)

theorem range_of_g : Set.Icc (-1/8 : ℝ) (9/8 : ℝ) = {y | ∃ t : ℝ, g t = y} :=
by
  sorry

end range_of_g_l609_609294


namespace evaluate_expression_l609_609275

theorem evaluate_expression : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 :=
by sorry

end evaluate_expression_l609_609275


namespace second_year_students_sampled_l609_609559

def total_students (f s t : ℕ) : ℕ := f + s + t

def proportion_second_year (s total_stu : ℕ) : ℚ := s / total_stu

def sampled_second_year_students (p : ℚ) (n : ℕ) : ℚ := p * n

theorem second_year_students_sampled
  (f s t : ℕ) (n : ℕ)
  (h1 : f = 600)
  (h2 : s = 780)
  (h3 : t = 720)
  (h4 : n = 35) :
  sampled_second_year_students (proportion_second_year s (total_students f s t)) n = 13 := 
sorry

end second_year_students_sampled_l609_609559


namespace factorize_xy2_minus_x_l609_609683

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609683


namespace parallelogram_angles_l609_609833

noncomputable def calculate_parallelogram_angles (a b : ℝ) (α : ℝ) (h : a > b) (acute_α: 0 < α ∧ α < π / 2): ℝ × ℝ :=
let sin_θ := (a^2 - b^2) / (2 * a * b) * tan(α),
    θ := Real.arcsin sin_θ in
    (θ, π - θ)

theorem parallelogram_angles (a b α : ℝ) (h : a > b) (acute_α: 0 < α ∧ α < π / 2) :
  let (θ, θ') := calculate_parallelogram_angles a b α h acute_α in
  θ = Real.arcsin ((a^2 - b^2) / (2 * a * b) * tan(α)) ∧
  θ' = π - Real.arcsin ((a^2 - b^2) / (2 * a * b) * tan(α)) :=
by
  let sin_θ := (a^2 - b^2) / (2 * a * b) * tan(α)
  have θ := Real.arcsin sin_θ
  exact (θ, π - θ)
  sorry

end parallelogram_angles_l609_609833


namespace angle_ADC_is_140_l609_609381

-- Let A, B, C be points in the Euclidean plane, and D be a point on AC
variables {A B C D : Point}

-- Assume D is a point on AC, BD = DC, and ∠BCD = 70 degrees
variables (h1 : Collinear A C D) 
variables (h2 : Distance B D = Distance D C)
variables (h3 : angle B C D = 70)

-- We need to prove that ∠ADB = 140 degrees
theorem angle_ADC_is_140 : angle A D B = 140 :=
  sorry

end angle_ADC_is_140_l609_609381


namespace compare_values_l609_609775

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l609_609775


namespace minimum_glue_drops_to_prevent_37_gram_subset_l609_609251

def stones : List ℕ := List.range' 1 36  -- List of stones with masses from 1 to 36 grams

def glue_drop_combination_invalid (stones : List ℕ) : Prop :=
  ¬ (∃ (subset : List ℕ), subset.sum = 37 ∧ (∀ s ∈ subset, s ∈ stones))

def min_glue_drops (stones : List ℕ) : ℕ := 
  9 -- as per the solution

theorem minimum_glue_drops_to_prevent_37_gram_subset :
  ∀ (s : List ℕ), s = stones → glue_drop_combination_invalid s → min_glue_drops s = 9 :=
by intros; sorry

end minimum_glue_drops_to_prevent_37_gram_subset_l609_609251


namespace real_roots_greater_than_2_l609_609974

theorem real_roots_greater_than_2 (k : ℝ) :
  (∀ x ∈ real_roots (λ x, x^2 + (k-2)*x + 5 - k), x > 2) ↔ -5 < k ∧ k < -4 :=
by sorry

end real_roots_greater_than_2_l609_609974


namespace fib_arith_seq_l609_609089

-- Definitions of Fibonacci sequence and arithmetic properties
def fibonacci : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+3) := fibonacci (n+2) + fibonacci (n+1)

-- Condition for forming an arithmetic sequence
def arithmetic (a b c : ℕ) : Prop :=
  b - a = c - b

-- Main theorem statement in Lean 4
theorem fib_arith_seq (a b c : ℕ) (h_cond : a + b + c = 2500) (h_form : (a, b, c) = (a, a+2, a+4)) (h_arith : arithmetic (fibonacci a) (fibonacci b) (fibonacci c)) : a = 831 :=
by
  -- The actual proof will be filled in here
  sorry

end fib_arith_seq_l609_609089


namespace cake_remaining_portion_l609_609628

theorem cake_remaining_portion (initial_cake : ℝ) (alex_share_percentage : ℝ) (jordan_share_fraction : ℝ) :
  initial_cake = 1 ∧ alex_share_percentage = 0.4 ∧ jordan_share_fraction = 0.5 →
  (initial_cake - alex_share_percentage * initial_cake) * (1 - jordan_share_fraction) = 0.3 :=
by
  sorry

end cake_remaining_portion_l609_609628


namespace range_of_m_l609_609301

def local_odd_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ set.Icc a b, f (-x) = -f x

def p (m : ℝ) : Prop :=
  local_odd_function (λ x, m + 2^x) (-1) 2

def q (m : ℝ) : Prop :=
  let g := λ x : ℝ, x^2 + (5 * m + 1) * x + 1 in
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 = 0 ∧ g x2 = 0

theorem range_of_m (m : ℝ) (h1 : ¬ (p m ∧ q m)) (h2 : p m ∨ q m) :
  m < -5 / 4 ∨ (-1 < m ∧ m < -3 / 5) ∨ m > 1 / 5 := 
sorry

end range_of_m_l609_609301


namespace sum_of_p_and_q_l609_609948

theorem sum_of_p_and_q (
  (a b c r x : ℝ) 
  (p : ℝ → ℝ := λ x, a * x + b)
  (q : ℝ → ℝ := λ x, c * (x + 2) * (x - r))
  (h1 : 3 * a + b = 5)
  (h2 : b = 4)
  (h3 : c = 2 / (15 - 5 * r))
) :
  p x + q x = (1 / 3) * x + 4 + (2 / (15 - 5 * r)) * (x + 2) * (x - r) :=
sorry

end sum_of_p_and_q_l609_609948


namespace new_ratio_of_subtracted_numbers_l609_609964

theorem new_ratio_of_subtracted_numbers : ∀ (a b : ℕ),
  a = 6 * (x : ℕ) →
  b = 5 * x →
  a - b = 5 →
  (a - 5) / gcd (a - 5) (b - 5) = 5 ∧ (b - 5) / gcd (a - 5) (b - 5) = 4 :=
by
  assume a b x h1 h2 h3
  sorry

end new_ratio_of_subtracted_numbers_l609_609964


namespace basketball_game_score_difference_l609_609419

theorem basketball_game_score_difference :
  let blueFreeThrows := 18
  let blueTwoPointers := 25
  let blueThreePointers := 6
  let redFreeThrows := 15
  let redTwoPointers := 22
  let redThreePointers := 5
  let blueScore := blueFreeThrows * 1 + blueTwoPointers * 2 + blueThreePointers * 3
  let redScore := redFreeThrows * 1 + redTwoPointers * 2 + redThreePointers * 3
  blueScore - redScore = 12 := by
  sorry

end basketball_game_score_difference_l609_609419


namespace vanya_correct_answers_l609_609471

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609471


namespace part1_part2_l609_609756

noncomputable def f (omega x : ℝ) : ℝ := 
  (Real.sqrt 3 * Real.sin (omega * x) * Real.cos (omega * x)) + 
  (Real.sin (omega * x))^2 - 1 / 2

theorem part1 (omega x : ℝ) (h : ∀ x, f omega x = f omega (x - π)) : 
  (omega = 1) → (f omega x = Real.sin (2 * x - π / 6)) :=
begin
  sorry
end

theorem part2 (omega : ℝ) (h_omega : omega = 1) : 
  ∃ (x_max x_min ∈ Set.Icc 0 (π / 2)), 
    f omega x_max = 1 ∧ f omega x_min = -1 / 2 :=
begin
  use [π / 3, 0],
  split,
  { split; norm_num, linarith, },
  { intros _ h_x,
    norm_num[sqrt _2_le, *],
    split; norm_num, },
  all_goals { sorry, },
end

end part1_part2_l609_609756


namespace lottery_probability_exactly_one_common_l609_609402

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609402


namespace math_problem_l609_609505

-- Definition of n* (the reciprocal of n)
def n_star (n : ℕ) : ℚ := 1 / n

-- Statement i: 4* + 8* = 12*
def statement_i : Prop := n_star 4 + n_star 8 = n_star 12

-- Statement ii: 8* - 3* = 5*
def statement_ii : Prop := n_star 8 - n_star 3 = n_star 5

-- Statement iii: 3* * 9* = 27*
def statement_iii : Prop := n_star 3 * n_star 9 = n_star 27

-- Statement iv: 15* / 3* = 5*
def statement_iv : Prop := n_star 15 / n_star 3 = n_star 5

-- The condition that counts true statements among i), ii), iii), and iv)
def number_of_true_statements : ℕ :=
  [statement_i, statement_ii, statement_iii, statement_iv].count (λ s => s)

-- The actual mathematical problem
theorem math_problem : number_of_true_statements = 2 := by
  sorry

end math_problem_l609_609505


namespace inequality_abc_l609_609779

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l609_609779


namespace northernmost_town_l609_609973

theorem northernmost_town (Cans Ernie Dundee Arva Blythe : Type) 
  (h1 : ∀ x, Cans x → ¬ Ernie x)
  (h2 : ∀ x, (Dundee x → ¬ Cans x) ∧ (Dundee x → ¬ Ernie x))
  (h3 : ∀ x, (Arva x → ¬ Blythe x) ∧ (Arva x → (Dundee x → false)) ∧ (Arva x → (Cans x → false))) :
  ∀ x, Blythe x → (Cans x ∨ Ernie x ∨ Dundee x ∨ Arva x → false) :=
by
  sorry

end northernmost_town_l609_609973


namespace winners_make_zeros_l609_609844

theorem winners_make_zeros :
  (∃ c : ℕ, c = 999 ∧ (∀ c1 : ℕ, c1 > c → 
    let new_m := 2007777 - c1 * 2007 in
    (new_m ≥ 0 → ∃ c2 : ℕ, (new_m - c2 * 2007) ≥ 0))):
  sorry

end winners_make_zeros_l609_609844


namespace sum_of_logs_of_tangents_l609_609276

theorem sum_of_logs_of_tangents :
  (∑ k in finset.range 17, real.logb 2 (real.tan (real.pi / 18 * (k + 1)))) = 0 :=
by
  -- Proof goes here
  sorry

end sum_of_logs_of_tangents_l609_609276


namespace find_g_of_neg_one_l609_609892

def g (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 - 1 else 7 - 3 * x

theorem find_g_of_neg_one : g (-1) = 0 := 
by
  -- Insert the proof here
  sorry

end find_g_of_neg_one_l609_609892


namespace alex_has_more_pens_than_jane_l609_609627

-- Definitions based on the conditions
def starting_pens_alex : ℕ := 4
def pens_jane_after_month : ℕ := 16

-- Alex's pen count after each week
def pens_alex_after_week (w : ℕ) : ℕ :=
  starting_pens_alex * 2 ^ w

-- Proof statement
theorem alex_has_more_pens_than_jane :
  pens_alex_after_week 4 - pens_jane_after_month = 16 := by
  sorry

end alex_has_more_pens_than_jane_l609_609627


namespace find_a_value_l609_609891

noncomputable def geometric_arithmetic_sequence (a b c : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧
  (b^2 = a * c) ∧
  let log_ca2 := (Real.log c (a^2))
      log_bsqrtc := (Real.log b (Real.sqrt c))
      log_acb := (Real.log a (c * b))
      d := 1 in
      log_bsqrtc - log_ca2 = d ∧ log_acb - log_bsqrtc = d

theorem find_a_value (a b c : ℝ) (h : geometric_arithmetic_sequence a b c) : 
  a = 10^(((-3 + Real.sqrt 13) / 4)) :=
sorry

end find_a_value_l609_609891


namespace min_distance_circle_to_line_l609_609311

-- Definitions
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

def circle_C (θ : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ θ : ℝ, ∀ x y : ℝ, (x, y) = circle_C θ →
    line_l x y →
    dist (x, y) = 2 * Real.sqrt 2 - 2 :=
sorry

end min_distance_circle_to_line_l609_609311


namespace sampling_method_selection_l609_609149

-- Define the sampling methods as data type
inductive SamplingMethod
| SimpleRandomSampling : SamplingMethod
| SystematicSampling : SamplingMethod
| StratifiedSampling : SamplingMethod
| SamplingWithReplacement : SamplingMethod

-- Define our conditions
def basketballs : Nat := 10
def is_random_selection : Bool := true
def no_obvious_stratification : Bool := true

-- The theorem to prove the correct sampling method
theorem sampling_method_selection 
  (b : Nat) 
  (random_selection : Bool) 
  (no_stratification : Bool) : 
  SamplingMethod :=
  if b = 10 ∧ random_selection ∧ no_stratification then SamplingMethod.SimpleRandomSampling 
  else sorry

-- Prove the correct sampling method given our conditions
example : sampling_method_selection basketballs is_random_selection no_obvious_stratification = SamplingMethod.SimpleRandomSampling := 
by
-- skipping the proof here with sorry
sorry

end sampling_method_selection_l609_609149


namespace sum_and_product_of_reciprocals_l609_609970

theorem sum_and_product_of_reciprocals (x y : ℝ) (h_sum : x + y = 12) (h_prod : x * y = 32) :
  (1/x + 1/y = 3/8) ∧ (1/x * 1/y = 1/32) :=
by
  sorry

end sum_and_product_of_reciprocals_l609_609970


namespace tangents_intersect_at_one_point_tangents_lie_in_one_plane_l609_609440

-- Define regular polyhedron and associated properties
variable {Polyhedron : Type} [RegularPolyhedron Polyhedron]

-- For any face π of polyhedron T, the lines drawn through the midpoints of its edges, tangent to the midsphere, and perpendicular to the edges intersect at one point.
theorem tangents_intersect_at_one_point (T : Polyhedron) :
  ∀ (π : Face T), ∃ (P : Point), ∀ e ∈ Edges π,
  let m := TangentMidPointMidspherePerpendicular e in P ∈ LineThroughMidPoint m :=
sorry

-- For any vertex v of polyhedron T, the lines drawn through the midpoints of its originating edges, tangent to the midsphere, and perpendicular to the edges lie in one plane.
theorem tangents_lie_in_one_plane (T : Polyhedron) :
  ∀ (v : Vertex T), ∃ (P : Plane), ∀ e ∈ EdgesFrom v,
  let m := TangentMidPointMidspherePerpendicular e in m ∈ P :=
sorry

end tangents_intersect_at_one_point_tangents_lie_in_one_plane_l609_609440


namespace vanya_correct_answers_l609_609484

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l609_609484


namespace quadratic_inequality_solution_l609_609932

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 :=
by {
  sorry
}

end quadratic_inequality_solution_l609_609932


namespace area_of_isosceles_triangle_l609_609237

theorem area_of_isosceles_triangle (h1 : ∃ b s: ℝ, 2 * s + 2 * b = 40 ∧ b^2 + 100 = s^2) : 
  let b := 7.5 in
  let s := 12.5 in
  1/2 * 2 * b * 10 = 75 :=
by
  admit

end area_of_isosceles_triangle_l609_609237


namespace excenter_property_l609_609330

variable (A B C M T I_b : Type)

-- Using exists to indicate existence of these points with given properties
axiom excenter_opposite_B : ∃ (I_b : Type), true
axiom midpoint_arc_BC_not_containing_A : ∃ (M : Type), true
axiom intersection_of_MI_b_with_circumcircle : ∃ (T : Type), true

theorem excenter_property 
  (I_b_opposite_B : excenter_opposite_B I_b)
  (M_midpoint_arc_BC : midpoint_arc_BC_not_containing_A M)
  (T_intersection : intersection_of_MI_b_with_circumcircle T):
  (dist T I_b) ^ 2 = (dist T B) * (dist T C) :=
sorry

end excenter_property_l609_609330


namespace complex_number_quadrant_l609_609346

theorem complex_number_quadrant :
  let z := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z < 0 ∧ Complex.im z > 0 :=
by
  sorry

end complex_number_quadrant_l609_609346


namespace sam_subtracts_quantity_l609_609558

theorem sam_subtracts_quantity : ∃ (subtracted_quantity : ℤ), 49^2 = 50^2 - subtracted_quantity + 1 ∧ subtracted_quantity = 100 := 
by 
  use 100
  split
  {
    sorry
  }

end sam_subtracts_quantity_l609_609558


namespace existence_of_term_3001_number_of_x_values_l609_609663

noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0       := x
| 1       := 3000
| (n + 2) := (sequence (n + 1) + 1) / sequence n

theorem existence_of_term_3001 :
  {x : ℝ | ∃ n : ℕ, sequence x n = 3001}.finite :=
sorry

theorem number_of_x_values :
  {x : ℝ | ∃ n : ℕ, sequence x n = 3001}.to_finset.card = 4 :=
sorry

end existence_of_term_3001_number_of_x_values_l609_609663


namespace percent_relation_l609_609380

variable (x y z : ℝ)

theorem percent_relation (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : x = 0.78 * z :=
by sorry

end percent_relation_l609_609380


namespace problem_part1_problem_part2_l609_609747

-- Define points A and B, moving point M with angle condition and distance condition.
noncomputable def A := (⟨-2, 0⟩ : ℝ × ℝ)
noncomputable def B := (⟨2, 0⟩ : ℝ × ℝ)

-- Conditions for M and vector magnitudes.
variables (M : ℝ × ℝ) (θ : ℝ)
def angle_condition : Prop := ∠ A M B = 2*θ
def distance_product_condition : Prop := (dist A M) * (dist B M) = 4 / (cos θ)^2

-- Prove total length and trajectory curve
theorem problem_part1 (h1 : angle_condition M θ) (h2 : distance_product_condition M θ) :
  (dist A M) + (dist B M) = 4*real.sqrt 2 ∧ ∃ (a b : ℝ), a = 2*real.sqrt 2 ∧ b = 2 ∧ 
  (M.1^2)/(a*a) + (M.2^2)/(b*b) = 1 := sorry

-- Prove existence of circle with radius
variables (l : ℝ → ℝ) (k m : ℝ)
def line := λ x, k*x + m
theorem problem_part2 (h_non_parallel : ∀ P Q, P ≠ Q → angle_condition P θ → angle_condition Q θ → vector.angle P Q = π/2)
    (h1 : angle_condition M θ) (h2 : distance_product_condition M θ) :
  ∃ r : ℝ, l = (λ x, r^2-x^2-y^2) ∧ r^2 = 8/3 := sorry

end problem_part1_problem_part2_l609_609747


namespace product_of_equal_numbers_l609_609512

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l609_609512


namespace Vanya_correct_answers_l609_609476

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l609_609476


namespace sum_mi_equals_400_l609_609023

open Triangle

def is_triangle (A B C : Point) : Prop := collinear A B C ∧ collinear B A C ∧ collinear C A B

variables (A B C : Point)
variables (P : ℕ → Point) (n : ℕ)

axiom triangle_ABC : is_triangle A B C
axiom equal_sides_AB_AC : distance A B = 2 ∧ distance A C = 2
axiom points_on_BC : ∀ i : ℕ, i < 100 → collinear B P(i) C
axiom mi_definition : ∀ i : ℕ, i < 100 → let AP := distance A (P i), BP := distance B (P i), PC := distance (P i) C in m i = AP^2 + BP * PC

theorem sum_mi_equals_400 : (∀ i : ℕ, i < 100 → m i = 4) → (∑ i in range 100, m i) = 400 :=
by
  assumption -- The assumption that m(i) = 4
  apply finset.sum_congr
  · intro i hi
    rw mi_definition i
    exact hi
  · refl
  sorry -- Skipping the full detailed steps of the proof

/- Note:
This Lean statement assumes the definitions of distance, collinear, and properties of the triangle from a geometry library in Lean.
It can be adjusted to use definitions directly from the "Mathlib" or other geometry libraries as per availability.
-/

end sum_mi_equals_400_l609_609023


namespace total_transport_cost_l609_609508

def cost_per_kg : ℝ := 25000
def mass_sensor_g : ℝ := 350
def mass_communication_g : ℝ := 150

theorem total_transport_cost : 
  (cost_per_kg * (mass_sensor_g / 1000) + cost_per_kg * (mass_communication_g / 1000)) = 12500 :=
by
  sorry

end total_transport_cost_l609_609508


namespace tip_percentage_l609_609189

theorem tip_percentage
  (total_amount_paid : ℝ)
  (price_of_food : ℝ)
  (sales_tax_rate : ℝ)
  (total_amount : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_amount_paid = 184.80)
  (h2 : price_of_food = 140)
  (h3 : sales_tax_rate = 0.10)
  (h4 : total_amount = price_of_food + (price_of_food * sales_tax_rate))
  (h5 : tip_percentage = ((total_amount_paid - total_amount) / total_amount) * 100) :
  tip_percentage = 20 := sorry

end tip_percentage_l609_609189


namespace repeated_three_digit_divisible_l609_609821

theorem repeated_three_digit_divisible (μ : ℕ) (h : 100 ≤ μ ∧ μ < 1000) :
  ∃ k : ℕ, (1000 * μ + μ) = k * 7 * 11 * 13 := by
sorry

end repeated_three_digit_divisible_l609_609821


namespace find_num_cows_l609_609412

variable (num_cows num_pigs : ℕ)

theorem find_num_cows (h1 : 4 * num_cows + 24 + 4 * num_pigs = 20 + 2 * (num_cows + 6 + num_pigs)) 
                      (h2 : 6 = 6) 
                      (h3 : ∀x, 2 * x = x + x) 
                      (h4 : ∀x, 4 * x = 2 * 2 * x) 
                      (h5 : ∀x, 4 * x = 4 * x) : 
                      num_cows = 6 := 
by {
  sorry
}

end find_num_cows_l609_609412


namespace correct_polynomial_degree_l609_609229

def polynomial_A : ℤ[X] := 3 * X^2 + X - 1
def polynomial_B : ℤ[X] := 3 * X * Y + X^2 * Y - 1
def polynomial_C : ℤ[X] := X^3 * Y^3 + X * Y - 1
def polynomial_D : ℤ[X] := 5 * X^3

theorem correct_polynomial_degree : 
  ∃ p, (p = polynomial_B ∧ polynomial.degree p = 3) :=
by
  use polynomial_B
  sorry

end correct_polynomial_degree_l609_609229


namespace range_of_y_over_x_l609_609075

theorem range_of_y_over_x 
  (x y : ℝ) 
  (h1 : 1 ≤ x ∧ x ≤ 2) 
  (h2 : 3 * x - 2 * y - 5 = 0) : 
  ∃ z, z ∈ Set.interval (-1 : ℝ) (1 / 4 : ℝ) ∧ z = y / x :=
by
  sorry

end range_of_y_over_x_l609_609075


namespace pyramid_volume_l609_609091

-- Define the given conditions and the problem
theorem pyramid_volume (r : ℝ) (acute_angle_base : ℝ) (lateral_faces_inclined : ℝ) :
  acute_angle_base = 30 → lateral_faces_inclined = 60 →
  let volume : ℝ := (8 / 3) * r^3 * Real.sqrt 3 in
  ∃ (V : ℝ), V = volume := 
by 
  intros h1 h2 
  use (8 / 3) * r^3 * Real.sqrt 3 
  sorry

end pyramid_volume_l609_609091


namespace billy_dishes_to_equal_time_l609_609648

-- These are conditions given in the problem
def minutes_sweeping_per_room : ℕ := 3
def minutes_dishes_per_dish : ℕ := 2
def minutes_laundry_per_load : ℕ := 9
def anna_rooms : ℕ := 10
def billy_loads : ℕ := 2

-- Calculations derived from the conditions
def anna_sweeping_time : ℕ := anna_rooms * minutes_sweeping_per_room
def billy_laundry_time : ℕ := billy_loads * minutes_laundry_per_load

-- Here's the theorem that proves the problem statement which is based on the final result from solution steps.
theorem billy_dishes_to_equal_time : ∃ (dishes : ℕ), (billy_laundry_time + dishes * minutes_dishes_per_dish = anna_sweeping_time) ∧ dishes = 6 :=
by
  use 6
  rw [anna_sweeping_time, billy_laundry_time, minutes_sweeping_per_room, minutes_laundry_per_load, minutes_dishes_per_dish]
  simp
  split
  · simp
  · refl

end billy_dishes_to_equal_time_l609_609648


namespace systematic_sample_seat_number_l609_609253

theorem systematic_sample_seat_number (total_students sample_size interval : ℕ) (seat1 seat2 seat3 : ℕ) 
  (H_total_students : total_students = 56)
  (H_sample_size : sample_size = 4)
  (H_interval : interval = total_students / sample_size)
  (H_seat1 : seat1 = 3)
  (H_seat2 : seat2 = 31)
  (H_seat3 : seat3 = 45) :
  ∃ seat4 : ℕ, seat4 = 17 :=
by 
  sorry

end systematic_sample_seat_number_l609_609253


namespace checkerboard_ratio_l609_609659

theorem checkerboard_ratio:
  let s := (finset.range 7).sum (λ n, n * n),
  let r := (finset.range 7).card.choose 2 * (finset.range 7).card.choose 2,
  let ratio := (nat.gcd s r, s / (nat.gcd s r), r / (nat.gcd s r))
  in ratio.1 + ratio.2 = 8 := 
by
  let squares_in_6x6 := 6^2 + 5^2 + 4^2 + 3^2 + 2^2 + 1^2,
  have hs : s = 91 := by sorry,
  let rectangles_in_6x6 := 21 * 21, 
  have hr : r = 441 := by sorry,
  have ratio_corr := 91 / nat.gcd 91 441 == 1 ∧ r / nat.gcd 91 441 == 7, 
  have ratio_sum := (1 + 7) = 8, 
  exact ratio_sum,
sorry

end checkerboard_ratio_l609_609659


namespace find_100c_rounded_l609_609876

-- Define the sequence a₀, a₁, a₂, ...
def sequence_a : ℕ → ℝ
| 0       := 5 / 13
| (n + 1) := 2 * (sequence_a n)^2 - 1

-- Define the condition for c
def satisfies_inequality (c : ℝ) : Prop :=
  ∀ n : ℕ, 0 < n →
  |∏ i in finset.range n, sequence_a i| ≤ c / 2^n

-- Define the proof problem
theorem find_100c_rounded :
  (∃ c : ℝ, satisfies_inequality c) →
  ∃ (c : ℝ), 100 * c ≈ 108 :=
begin
  intro h,
  sorry
end

end find_100c_rounded_l609_609876


namespace num_factors_of_M_l609_609864

theorem num_factors_of_M :
  let M := 58^3 + 3 * 58^2 + 3 * 58 + 1 in
  nat.num_factors M = 4 :=
by
  sorry

end num_factors_of_M_l609_609864


namespace main_theorem_l609_609054

noncomputable def problem_statement (n : ℕ) (x : Fin n → ℝ) (s : ℝ) : Prop :=
  n ≥ 3 ∧ (∀ i, 0 ≤ x i ∧ x i ≤ 1) ∧ (s = ∑ i in Finset.range n, x i) ∧ (s ≥ 3) →
  ∃ (i j : Fin n), 1 ≤ i.val + 1 ∧ i.val < j.val ∧ j.val ≤ n ∧ 2^(j.val - i.val) * x i * x j > 2^(s-3)

theorem main_theorem (n : ℕ) (x : Fin n → ℝ) (s : ℝ) :
  problem_statement n x s :=
by
  sorry

end main_theorem_l609_609054


namespace factorization_l609_609692

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609692


namespace inequality_l609_609764

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l609_609764


namespace distance_from_circle_center_to_line_l609_609847

def polar_circle (p : ℝ × ℝ) : Prop := ∃ θ, p.1 = 4 * Real.sin θ
def polar_line (p : ℝ × ℝ) : Prop := ∃ ρ, p.2 = ρ ∧ θ = π / 6

theorem distance_from_circle_center_to_line 
  (C : ℝ × ℝ := (0, 2)) 
  (line : ℝ × ℝ → Prop := polar_line) : 
  (distance_to_line C line = sqrt 3) := 
sorry

end distance_from_circle_center_to_line_l609_609847


namespace other_root_l609_609907

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l609_609907


namespace solve_parabola_vertex_l609_609936

noncomputable def smallest_value_of_a
  (vertex : Prod ℚ ℚ)
  (some_real : ℚ)
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)
  (cond1 : vertex = (1/3, -25/27))
  (cond2 : 3 * a + 2 * b + 4 * c ∈ ℤ)
  (cond3 : a > 0)
  : ℚ := 
  300 / 19

theorem solve_parabola_vertex
  (vertex : Prod ℚ ℚ := (1/3, -25/27))
  (a b c : ℚ)
  (cond1 : vertex = (1/3, -25/27))
  (cond2 : b = -2 * a / 3)
  (cond3 : c = a / 9 - 25 / 27)
  (cond4 : 3 * a + 2 * b + 4 * c ∈ ℤ)
  (cond5 : a > 0)
  : smallest_value_of_a vertex (300 / 19) a b c cond1 cond4 cond5 = 300 / 19 := 
sorry

end solve_parabola_vertex_l609_609936


namespace max_sum_x_y_min_diff_x_y_l609_609977

def circle_points (x y : ℤ) : Prop := (x - 1)^2 + (y + 2)^2 = 36

theorem max_sum_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x + y ≥ x' + y') :=
  by sorry

theorem min_diff_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x - y ≤ x' - y') :=
  by sorry

end max_sum_x_y_min_diff_x_y_l609_609977


namespace sequence_zero_count_l609_609432

def satisfies_rule (a : ℕ → ℕ) (a1 a2 a3 : ℕ) : Prop :=
  a 1 = a1 ∧ a 2 = a2 ∧ a 3 = a3 ∧
  ∀ n ≥ 4, a n = a (n - 1) * (|a (n - 2) - a (n - 3)|)

def zero_for_some_n (a : ℕ → ℕ) : Prop :=
  ∃ n, a n = 0

theorem sequence_zero_count :
  (∃ seqs : Finset (ℕ × ℕ × ℕ), 
    ∀ (x y z : ℕ), (x, y, z) ∈ seqs ↔ (1 ≤ x ∧ x ≤ 10 ∧ 1 ≤ y ∧ y ≤ 10 ∧ 1 ≤ z ∧ z ≤ 10) 
    ∧ (zero_for_some_n (λ n, if n = 1 then x else if n = 2 then y else if n = 3 then z else _))) →
  seqs.card = 494 :=
sorry

end sequence_zero_count_l609_609432


namespace factorization_l609_609697

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609697


namespace ratio_AD_DC_l609_609848

variables {A B C D E : Type} [triangle : is_triangle A B C]
          (BD bisects angle_ABC : bool)
          (BE bisects angle_DBA : bool)
          [angle_bisector_ABC : angle_bisector BD (angle ABC)]
          [angle_bisector_DBA : angle_bisector BE (angle DBA)]

theorem ratio_AD_DC (h1 : BD bisects angle_ABC) (h2: BE bisects angle_DBA) :
  AD / DC = AB / BC :=
sorry

end ratio_AD_DC_l609_609848


namespace a_n_formula_root_proof_l609_609316

-- Definitions for the sequence {a_n} and the sum of the first n terms S_n.
def a (n : ℕ) : ℚ := 
  if n = 1 then 1/2 
  else if n = 2 then 1/6 
  else 1/n/(n+1)

def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, a (i + 1))

-- Main theorem to prove
theorem a_n_formula : ∀ n : ℕ, 0 < n → a n = 1 / (n * (n + 1)) :=
by
  intros n hn
  sorry

-- Proof of the roots being S_n - 1
theorem root_proof : ∀ n : ℕ, 0 < n → ∃ x : ℚ, x^2 - a n * x - a n = 0 ∧ x = S n - 1 :=
by
  intros n hn
  use S n - 1
  sorry

end a_n_formula_root_proof_l609_609316


namespace ratio_unchanged_l609_609100

-- Define the initial ratio
def initial_ratio (a b : ℕ) : ℚ := a / b

-- Define the new ratio after transformation
def new_ratio (a b : ℕ) : ℚ := (3 * a) / (b / (1/3))

-- The theorem stating that the ratio remains unchanged
theorem ratio_unchanged (a b : ℕ) (hb : b ≠ 0) :
  initial_ratio a b = new_ratio a b :=
by
  sorry

end ratio_unchanged_l609_609100


namespace trigonometric_identity_l609_609657

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) + Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 * Real.tan (10 * Real.pi / 180) :=
by
  sorry

end trigonometric_identity_l609_609657


namespace instantaneous_velocity_at_t_3_l609_609636

variable (t : ℝ)
def s (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_t_3 : 
  ∃ v, v = -1 + 2 * 3 ∧ v = 5 :=
by
  sorry

end instantaneous_velocity_at_t_3_l609_609636


namespace sally_out_of_pocket_cost_l609_609078

/-- Definitions of the given conditions -/
def given_money : Int := 320
def cost_per_book : Int := 15
def number_of_students : Int := 35

/-- Theorem to prove the amount Sally needs to pay out of pocket -/
theorem sally_out_of_pocket_cost : 
  let total_cost := number_of_students * cost_per_book
  let amount_given := given_money
  let out_of_pocket_cost := total_cost - amount_given
  out_of_pocket_cost = 205 := by
  sorry

end sally_out_of_pocket_cost_l609_609078


namespace seating_arrangements_l609_609014

-- Definitions for conditions in the problem
def total_unrestricted_seating (n : ℕ) : ℕ := Nat.fact n
def restricted_seating_wp_as_unit (n : ℕ) : ℕ := Nat.fact (n - 1) * 2
def restricted_seating_with_eve (n : ℕ) : ℕ :=
  2 * (Nat.fact (n - 1) - restricted_seating_wp_as_unit n)

-- The final proof statement
theorem seating_arrangements : total_unrestricted_seating 8 -
  restricted_seating_with_eve 8 = 20160 := by
  sorry

end seating_arrangements_l609_609014


namespace value_x_plus_2y_plus_3z_l609_609321

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end value_x_plus_2y_plus_3z_l609_609321


namespace original_amount_l609_609591

theorem original_amount (X : ℝ) (h : 0.05 * X = 25) : X = 500 :=
sorry

end original_amount_l609_609591


namespace semicircle_circumference_l609_609109

theorem semicircle_circumference :
  let length := 18
  let breadth := 14
  let perimeter_rect := 2 * (length + breadth)
  let side_square := perimeter_rect / 4
  let diameter := side_square
  let semicircle_circumference := (Real.pi * diameter) / 2 + diameter
  side_square * 4 = perimeter_rect → 
  (semicircle_circumference ≈ 41.12) :=
by
  intro length breadth perimeter_rect side_square diameter semicircle_circumference
  -- provided conditions and the need to equivalently prove the circumference ≈ 41.12
  exact sorry

end semicircle_circumference_l609_609109


namespace tiles_per_row_l609_609942

theorem tiles_per_row (area_square_meters : ℕ) (tile_side_cm : ℕ) (side_length_meters: ℕ) 
(h1 : area_square_meters = 144)
(h2 : tile_side_cm = 30)
(h3 : side_length_meters = nat.sqrt 144)
: side_length_meters * 100 / tile_side_cm = 40 :=
by 
  sorry

end tiles_per_row_l609_609942


namespace length_HY_l609_609173

-- Variables and assumptions based on conditions and problem.
variable (C D E F G H Y : Type) -- Points of the hexagon and point Y
variable [Distance C D E F G H Y] [RegularHexagon C D E F G H] -- Hexagon properties
variable (CD : Real)
variable (side_length : CD = 3)
variable (CY : Real) (extension : CY = 4 * CD)

-- The theorem we want to prove.
theorem length_HY (H : Point) (Y : Point) (Q : Point)
  (CD : ℝ) (side_length : CD = 3)
  (CY : ℝ) (extension : CY = 4 * CD)
  (QH : ℝ) (CQ : ℝ) (QY : ℝ) (QY_eq : QY = CQ + CY)
  (HQY : ℝ) (HY²_eq : HQY = QH² + QY²) :
  HQY = 3 * sqrt 20.75 := 
sorry

end length_HY_l609_609173


namespace train_length_l609_609218

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℚ) : 
  speed_kmh = 120 → 
  time_s = 25 → 
  length_m = 833.25 → 
  (speed_kmh * 1000 / 3600) * time_s = length_m :=
by
  intros
  sorry

end train_length_l609_609218


namespace knights_liars_pairs_l609_609066

/-- On an island, there are 100 knights and 100 liars. Each of them has at least one friend. 
    One day, exactly 100 people said: "All my friends are knights," and exactly 100 people said:
    "All my friends are liars." Prove that the smallest possible number of pairs of friends
    where one is a knight and the other is a liar is 50. -/
theorem knights_liars_pairs :
  ∃ (n : ℕ), n = 50 ∧ ∀ (K : Fin 100 → Prop) (L : Fin 100 → Prop) (friends : Fin 200 → Fin 200 → Prop),
    (∀ x, ∃ y, friends x y) ∧
    (∃ S, (∀ x, S x → K x ∨ L x) ∧ (∀ x, S x → ¬K x → ¬L x) ∧
      (∃ T, (∀ x, T x → K x ∨ L x) ∧ (∀ x, T x → ¬L x → ¬K x))) ∧
    (∑ x in (finset.univ : finset (fin 200)), 
      ite (∃ y, friends x y ∧ ((K x ∧ L y) ∨ (L x ∧ K y))) 1 0) = n :=
begin
  sorry
end

end knights_liars_pairs_l609_609066


namespace factorize_expression_l609_609691

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609691


namespace equal_numbers_product_l609_609511

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l609_609511


namespace series_properties_l609_609802

noncomputable def geometric_series_sum (a r : ℝ) := a / (1 - r)

def series_validity :=
  let a := 3 in
  let r := 1 / 2 in
  let S := geometric_series_sum a r in
  (S = 6) ∧
  (∀ ε > 0, ∀ n, |a * r ^ n - 0| ≥ ε) ∧
  (S ≠ ∞) ∧
  (S < 7) ∧
  (∀ ε > 0, |S - 6| < ε)

theorem series_properties : 
  series_validity := by
  sorry

end series_properties_l609_609802


namespace concrete_order_l609_609187

-- Definitions
def width : ℝ := 4 / 3
def length : ℝ := 100 / 3
def thickness_base : ℝ := 1 / 9
def thickness_top : ℝ := 1 / 18

-- Calculations
def volume_base : ℝ := width * length * thickness_base
def volume_top : ℝ := width * length * thickness_top
def total_volume : ℝ := volume_base + volume_top

-- Theorem to Prove
theorem concrete_order :
  let V_total := let V := total_volume in V
  (V_total.ceil) = 8 :=
by sorry

end concrete_order_l609_609187


namespace inequality_l609_609765

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l609_609765


namespace find_T10_l609_609540

-- Definitions
variable (a : ℕ → ℝ) -- Sequence {a_n}
variable (T : ℕ → ℝ) -- Product of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) = r * a n

def product_of_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 1
| (n + 1) := a (n + 1) * product_of_first_n_terms n

def T (n : ℕ) : ℝ := product_of_first_n_terms a n

axiom geometric_sequence (r : ℝ) (h : r ≠ 0): is_geometric_sequence a
axiom T2_eq_T8 : T 2 = T 8

-- Goal
theorem find_T10 (r : ℝ) (h : r ≠ 0) :
  is_geometric_sequence a →
  T 2 = T 8 →
  T 10 = 1 :=
by
  intros h1 h2
  sorry

end find_T10_l609_609540


namespace tan_sin_cos_log_expression_simplification_l609_609171

-- Proof Problem 1 Statement in Lean 4
theorem tan_sin_cos (α : ℝ) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by
  sorry

-- Proof Problem 2 Statement in Lean 4
theorem log_expression_simplification : 
  Real.logb 3 (Real.sqrt 27) + Real.logb 10 25 + Real.logb 10 4 + 
  (7 : ℝ) ^ Real.logb 7 2 + (-9.8) ^ 0 = 13 / 2 :=
by
  sorry

end tan_sin_cos_log_expression_simplification_l609_609171


namespace find_range_of_a_l609_609879

-- Statement of the problem

variable (f : ℝ → ℝ) -- Function f is defined on ℝ
variable (a : ℝ)     -- a is the real number we need to find the range for

-- Conditions from the problem
def condition1 : Prop := ∀ x : ℝ, f (x + 1) = f (-(x + 1))      -- f(x+1) is even
def condition2 : Prop := ∀ x > 1, (x - 1) * (f' x) + f x > 0  -- Given inequality when x > 1
def condition3 : Prop := ∀ x : ℝ, e^(2 * x) * f (e^(2 * x) + 1) ≥ (2 * x - a) * f (2 * x + 1 - a)

-- Main theorem stating the range of a
theorem find_range_of_a (h1 : condition1 f) (h2 : condition2 f) (h3 : condition3 f) : 
  ∀ a : ℝ, a ≥ -1 :=
sorry

end find_range_of_a_l609_609879


namespace constant_term_expansion_l609_609094

theorem constant_term_expansion : 
  let x : ℝ
  let binomial_expansion (n : ℕ) (a b : ℝ) := (∑ r in Finset.range (n+1), (Nat.choose n r) * (a ^ (n-r)) * (b ^ r))
  in binomial_expansion 5 1 (-1/√x) = 10 :=
by
  let x : ℝ := 1 -- x can be substituted with any specific value as it gets cancelled out in the process
  let binomial_expansion := (∑ r in Finset.range (6), (Nat.choose 5 r) * (1 ^ (5-r)) * ((-1/√x) ^ r))
  show x * binomial_expansion = 10 from sorry

end constant_term_expansion_l609_609094


namespace jack_jill_meeting_distance_l609_609025

-- Definitions for Jack's and Jill's initial conditions
def jack_speed_uphill := 12 -- km/hr
def jack_speed_downhill := 15 -- km/hr
def jill_speed_uphill := 14 -- km/hr
def jill_speed_downhill := 18 -- km/hr

def head_start := 0.2 -- hours
def total_distance := 12 -- km
def turn_point_distance := 7 -- km
def return_distance := 5 -- km

-- Statement of the problem to prove the distance from the turning point where they meet
theorem jack_jill_meeting_distance :
  let jack_time_to_turn := (turn_point_distance : ℚ) / jack_speed_uphill
  let jill_time_to_turn := (turn_point_distance : ℚ) / jill_speed_uphill
  let x_meet := (8.95 : ℚ) / 29
  7 - (14 * ((x_meet - 0.2) / 1)) = (772 / 145 : ℚ) := 
sorry

end jack_jill_meeting_distance_l609_609025


namespace ratio_of_fifteenth_terms_l609_609866

variable {S_n T_n : ℕ → ℚ}
variable {a d b e : ℚ}

-- Conditions
def sum_formula_first_series (n : ℕ) := (n : ℚ) / 2 * (2 * a + (n - 1) * d)
def sum_formula_second_series (n : ℕ) := (n : ℚ) / 2 * (2 * b + (n - 1) * e)
def ratio_condition (n : ℕ) := (S_n n / T_n n) = (5 * n + 3) / (3 * n + 17)
def terms_condition : (2*a + (n-1)*d) / (2*b + (n-1)*e) = (5*n + 3) / (3*n + 17)

-- Given the above conditions, prove the following:
theorem ratio_of_fifteenth_terms :
  ∀ (a d b e : ℚ),
    (∀ n, S_n n = sum_formula_first_series n) →
    (∀ n, T_n n = sum_formula_second_series n) →
    (∀ n, ratio_condition n) →
    (2 * a / 2 * b = 2 / 5) →
    (2 * a + d) / (2 * b + e) = 13 / 23 →
    let a_15 := a + 14 * d,
        b_15 := b + 14 * e in
    (a_15 / b_15) = 44 / 95 :=
by
  intros
  let n := 1
  specialize H1 n
  rw [sum_formula_first_series, sum_formula_second_series] at H1
  -- Further necessary steps here
  sorry

end ratio_of_fifteenth_terms_l609_609866


namespace angle_between_a_and_c_is_60_degrees_l609_609889

variables (a b c : EuclideanSpace (Fin 3) ℝ)

-- Conditions: unit vectors and the given equation
def unit_vector (v : EuclideanSpace (Fin 3) ℝ) : Prop :=
  ‖v‖ = 1

axiom unit_a : unit_vector a
axiom unit_b : unit_vector b
axiom unit_c : unit_vector c

axiom given_equation : a × (b × c) = (b + (2 : ℝ) • c) / 2

-- Linearly independent set
def linearly_independent_set {n : Type*} [Fintype n] [DecidableEq n] 
  (s : Finset (EuclideanSpace n ℝ)) : Prop :=
  LinearIndependent ℝ (fun v => v)

axiom lin_indep : linearly_independent_set {a, b, c, b + c}

-- Problem statement: Proving the angle between a and c is 60 degrees
theorem angle_between_a_and_c_is_60_degrees :
  real.arccos (inner a c) = real.pi / 3 :=
sorry

end angle_between_a_and_c_is_60_degrees_l609_609889


namespace canoes_vs_kayaks_l609_609583

theorem canoes_vs_kayaks (C K : ℕ) (h1 : 9 * C + 12 * K = 432) (h2 : C = 4 * K / 3) : C - K = 6 :=
sorry

end canoes_vs_kayaks_l609_609583


namespace valid_y_series_sum_unique_l609_609296

theorem valid_y_series_sum_unique (y : ℝ) (h_converges : |y| < 1) :
  y = 2 - 2 * y + 2 * y^2 - 2 * y^3 + 2 * y^4 - 2 * y^5 + ∑ i in range 6, (-2) * y ^ (i : ℤ) → y = 1 :=
by
  sorry  -- To be proved

end valid_y_series_sum_unique_l609_609296


namespace suraj_new_average_l609_609086

noncomputable def suraj_average (A : ℝ) : ℝ := A + 8

theorem suraj_new_average (A : ℝ) (h_conditions : 14 * A + 140 = 15 * (A + 8)) :
  suraj_average A = 28 :=
by
  sorry

end suraj_new_average_l609_609086


namespace students_suggesting_bacon_l609_609640

theorem students_suggesting_bacon (S : ℕ) (M : ℕ) (h1: S = 310) (h2: M = 185) : S - M = 125 := 
by
  -- proof here
  sorry

end students_suggesting_bacon_l609_609640


namespace proposition_p_and_q_false_true_l609_609071

theorem proposition_p_and_q_false_true {a b x : ℝ} :
  (∀ a b : ℝ, ¬ (|a| + |b| > 1 → |a + b| > 1)) ∧
  (∀ x : ℝ, (∃ y : ℝ, y = sqrt (abs (x - 1) - 2)) ↔ (x ≤ -1 ∨ x ≥ 3)) :=
by
  sorry

end proposition_p_and_q_false_true_l609_609071


namespace interest_rate_is_correct_l609_609249

noncomputable def principal : ℝ := 54
noncomputable def amount : ℝ := 56.7
noncomputable def interest : ℝ := amount - principal

theorem interest_rate_is_correct : interest = principal * 0.05 * 1 :=
by
  have h_interest : interest = 2.7 := by
    simp [interest, amount, principal]
  rw [h_interest]
  norm_num

end interest_rate_is_correct_l609_609249


namespace board_game_cost_l609_609428

theorem board_game_cost
  (v h : ℝ)
  (h1 : 3 * v = h + 490)
  (h2 : 5 * v = 2 * h + 540) :
  h = 830 := by
  sorry

end board_game_cost_l609_609428


namespace find_x_plus_y_l609_609727

theorem find_x_plus_y (x y : ℝ) (h1 : 4 ^ x = 16 ^ (y + 2)) (h2 : 16 ^ y = 4 ^ (x - 4)) : x + y = 4 :=
by
  have h1' : (2^2)^x = (2^4)^(y+2) := by rw [← pow_mul, ← pow_mul]; rw h1
  have h2' : (2^4)^y = (2^2)^(x-4) := by rw [← pow_mul, ← pow_mul]; rw h2
  sorry

end find_x_plus_y_l609_609727


namespace sum_floor_ceil_eq_seven_l609_609543

theorem sum_floor_ceil_eq_seven (x : ℝ) 
  (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 := 
sorry

end sum_floor_ceil_eq_seven_l609_609543


namespace cyclic_sum_inequality_l609_609890

theorem cyclic_sum_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ( (b + c - a)^2 / (a^2 + (b + c)^2) +
    (c + a - b)^2 / (b^2 + (c + a)^2) +
    (a + b - c)^2 / (c^2 + (a + b)^2) ) ≥ 3 / 5 :=
  sorry

end cyclic_sum_inequality_l609_609890


namespace letter_250_in_pattern_l609_609997

def repeatingPattern (n : Nat) : Char :=
  let pattern := "ABCDE"
  pattern.get! (n % pattern.length - 1)

theorem letter_250_in_pattern : repeatingPattern 250 = 'E' := by
  sorry

end letter_250_in_pattern_l609_609997


namespace cost_price_of_second_item_l609_609620

theorem cost_price_of_second_item 
  (C₁ : ℝ) (C₂ : ℝ) (sell_price : ℝ := 432) 
  (loss_20_percent_first : C₁ * 0.80 = 216)
  (total_profit_20_percent : sell_price = (C₁ + C₂) * 1.20) : 
  C₂ = 90 :=
begin
  -- placeholder for actual proof
  sorry
end

end cost_price_of_second_item_l609_609620


namespace probability_one_common_number_approx_l609_609391

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l609_609391


namespace min_value_l609_609758

theorem min_value (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/29 :=
sorry

end min_value_l609_609758


namespace factorize_expression_l609_609689

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609689


namespace quadratic_inequality_solution_range_l609_609378

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, ¬ (x^2 + a * x + 4 < 0)) → a ∈ set.Icc (-4 : ℝ) (4 : ℝ) :=
by
  intros h
  sorry

end quadratic_inequality_solution_range_l609_609378


namespace largest_root_ratio_l609_609888

-- Define the polynomials f(x) and g(x)
def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4

-- Define the property that x1 is the largest root of f(x) and x2 is the largest root of g(x)
def is_largest_root (p : ℝ → ℝ) (r : ℝ) : Prop := 
  p r = 0 ∧ ∀ x : ℝ, p x = 0 → x ≤ r

-- The main theorem
theorem largest_root_ratio (x1 x2 : ℝ) 
  (hx1 : is_largest_root f x1) 
  (hx2 : is_largest_root g x2) : x2 = 2 * x1 :=
sorry

end largest_root_ratio_l609_609888


namespace min_val_PM_l609_609621

-- Definitions based on problem conditions
def circle (x y : ℝ) := (x + 1)^2 + (y - 2)^2 = 1
def tangent_at_point (P M : ℝ × ℝ) := ∃ c : ℝ, circle M.1 M.2 ∧ dist P M = c
def on_line (P : ℝ × ℝ) := P.1 - 2 * P.2 + 2 = 0
def dist (P O : ℝ × ℝ) := real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)

-- The theorem statement
theorem min_val_PM (P M O : ℝ × ℝ) (hCircle : circle M.1 M.2) (hTangent : tangent_at_point P M)
  (hEqual : dist P M = dist P O) : dist P M = 2 * real.sqrt 5 / 5 :=
by 
  sorry

end min_val_PM_l609_609621


namespace rectangle_area_is_27_l609_609213

noncomputable def area_of_rectangle : ℕ :=
  let square_area := 36
  let square_side_length := Int.sqrt square_area
  let square_perimeter := 4 * square_side_length
  let rect_perimeter := square_perimeter
  let width := rect_perimeter / 8 -- Since 2(3w + w) = 24 =>
  let length := 3 * width
  width * length

theorem rectangle_area_is_27 :
  (area_of_rectangle = 27) :=
by
  sorry

end rectangle_area_is_27_l609_609213


namespace count_valid_primes_l609_609809

open Nat

def is_valid_prime (p : ℕ) : Prop :=
  p > 50 ∧ p < 100 ∧
  Nat.Prime p ∧
  (Nat.Prime (p % 12)) ∧
  (p % 10 ≠ 1)

theorem count_valid_primes : (Finset.filter is_valid_prime (Finset.filter Nat.Prime (Finset.range 101))).card = 6 := by
  sorry

end count_valid_primes_l609_609809


namespace other_root_l609_609911

theorem other_root (m : ℝ) :
  ∃ r, (r = -7 / 3) ∧ (3 * 1 ^ 2 + m * 1 - 7 = 0) := 
begin
  use -7 / 3,
  split,
  { refl },
  { linarith }
end

end other_root_l609_609911


namespace spherical_to_rectangular_example_l609_609261

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (π / 3) (π / 4) = (sqrt 2, sqrt 6, 2 * sqrt 2) :=
by
  sorry

end spherical_to_rectangular_example_l609_609261


namespace arithmetic_progression_trig_identity_l609_609961

theorem arithmetic_progression_trig_identity (α β γ : ℝ) (h : β = (α + γ) / 2) :
    (sin α - sin γ) / (cos γ - cos α) = Real.cot β := 
  sorry

end arithmetic_progression_trig_identity_l609_609961


namespace quadratic_intersection_length_l609_609314

theorem quadratic_intersection_length :
  (∃ a b c : ℝ, (λ x, a * x^2 + b * x + c) (-1) = -1 ∧
                  (λ x, a * x^2 + b * x + c) 0 = -2 ∧
                  (λ x, a * x^2 + b * x + c) 1 = 1 ∧
    ∃ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c = 0) ∧
                   (a * x2^2 + b * x2 + c = 0) ∧
                   abs (x1 - x2) = (Real.sqrt 17) / 2) :=
sorry

end quadratic_intersection_length_l609_609314


namespace problem_statement_l609_609170

-- Define the sum of the first n terms of an arithmetic sequence S_n
def S (n : ℕ) : ℕ := n * n

-- Define the sequence b_n
def b (n : ℕ) : ℤ := n - (-1) ^ n * S n

-- Define T_n as the sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℤ := (List.range n).sum (λ x => b (x + 1))

-- Statement of the problem
theorem problem_statement :
  (∀ n, b n = n - (-1 : ℤ) ^ n * S n) →
  T 10 = 0 →
  (let A := {n : ℕ | n ≤ 100 ∧ T n ≤ 100 ∧ n ≠ 0} in A.sum id = 2575) :=
sorry

end problem_statement_l609_609170


namespace simpl_eval_l609_609501

variable (a b : ℚ)

theorem simpl_eval (h_a : a = 1/2) (h_b : b = -1/3) :
    5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (- a * b ^ 2 + 3 * a ^ 2 * b) = -11 / 36 := by
  sorry

end simpl_eval_l609_609501


namespace symmetric_graph_b_value_l609_609826

theorem symmetric_graph_b_value (a b : ℝ) 
  (h1 : x ∈ Icc a b → x ∈ Icc a b) 
  (h2 : ∀ x, x^2 + (a - 2) * x + 3 = (1 - (x - 1)) * (1 - (x - 1)) - 1) : 
  b = 2 :=
begin
  sorry, -- Proof is not required
end

end symmetric_graph_b_value_l609_609826


namespace division_result_l609_609244

theorem division_result : 180 / 6 / 3 / 2 = 5 := by
  sorry

end division_result_l609_609244


namespace smallest_n_1000_solutions_l609_609049

noncomputable def frac (x : ℝ) : ℝ := x - (Int.floor x)

noncomputable def f (x : ℝ) : ℝ := Real.sin (π * (frac x))

theorem smallest_n_1000_solutions :
  ∃ n : ℕ, (∀ x : ℝ, 0 < x → x < n + 1 → n * f (x * f x) + 1 = x) ∧ (n ≥ 500) := 
begin
  use 500,
  split,
  { sorry }, -- Placeholder for the proof that there are at least 1000 solutions when n = 500
  { linarith }
end

end smallest_n_1000_solutions_l609_609049


namespace domain_of_h_l609_609670

theorem domain_of_h (x : ℝ) : |x - 5| + |x + 3| ≠ 0 := by
  sorry

end domain_of_h_l609_609670


namespace lucas_average_speed_l609_609446

-- Define the initial odometer reading
def initial_reading : ℕ := 27372

-- Define the next palindrome reading and calculate distance traveled
def next_palindrome : ℕ := 27472
def distance_traveled : ℕ := next_palindrome - initial_reading

-- Define the time elapsed (in hours)
def time_elapsed : ℕ := 3

-- Calculate the average speed
def average_speed : ℕ := distance_traveled / time_elapsed

-- Prove that the average speed is 33 mph
theorem lucas_average_speed : average_speed = 33 := by
  unfold average_speed distance_traveled
  have : next_palindrome - initial_reading = 100 := by sorry
  have : 100 / 3 = 33 := by sorry
  rw [‹next_palindrome - initial_reading = 100›, ‹100 / 3 = 33›]
  sorry

end lucas_average_speed_l609_609446


namespace find_t_l609_609858

-- Given a quadratic equation
def quadratic_eq (x : ℝ) := 4 * x ^ 2 - 16 * x - 200

-- Completing the square to find t
theorem find_t : ∃ q t : ℝ, (x : ℝ) → (quadratic_eq x = 0) → (x + q) ^ 2 = t ∧ t = 54 :=
by
  sorry

end find_t_l609_609858


namespace problem_statement_l609_609329

noncomputable def a (n : ℕ) := n^2

theorem problem_statement (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n > 0) :
  x + a n / x ^ n ≥ n + 1 :=
sorry

end problem_statement_l609_609329


namespace last_score_is_68_l609_609448

theorem last_score_is_68
(scores : List ℕ)
(h : scores = [64, 68, 74, 77, 85, 90])
(have_integer_avg : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6 → ((scores.take n).sum / n : ℚ).denominator = 1) :
  scores.nth_le 5 (by decide) = 68 :=
by {
  cases scores,
  sorry
}

end last_score_is_68_l609_609448


namespace cos_shift_left_pi_fifth_l609_609130

theorem cos_shift_left_pi_fifth (x : ℝ) : 
  cos (x + π / 5) = cos (x - (-π / 5)) := 
sorry

end cos_shift_left_pi_fifth_l609_609130


namespace lottery_probability_exactly_one_common_l609_609400

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609400


namespace total_growing_space_l609_609633

theorem total_growing_space : 
  ∀ (beds1 beds2 : ℕ) (l1 w1 l2 w2 : ℕ), 
  beds1 = 2 → l1 = 3 → w1 = 3 → 
  beds2 = 2 → l2 = 4 → w2 = 3 → 
  (beds1 * (l1 * w1) + beds2 * (l2 * w2) = 42) :=
by
  intros beds1 beds2 l1 w1 l2 w2 h_beds1 h_l1 h_w1 h_beds2 h_l2 h_w2
  rw [h_beds1, h_l1, h_w1, h_beds2, h_l2, h_w2]
  norm_num
  sorry

end total_growing_space_l609_609633


namespace eval_f_at_neg1_eval_f_at_pos1_l609_609347

def f : ℝ → ℝ :=
  λ x, if x >= 0 then 2 else x + 1

theorem eval_f_at_neg1 : f (-1) = 0 :=
by {
  simp [f],
  split_ifs,
  sorry
}

theorem eval_f_at_pos1 : f 1 = 2 :=
by {
  simp [f],
  split_ifs,
  sorry
}

end eval_f_at_neg1_eval_f_at_pos1_l609_609347


namespace one_minus_repeating_eight_l609_609284

-- Given the condition
def b : ℚ := 8 / 9

-- The proof problem statement
theorem one_minus_repeating_eight : 1 - b = 1 / 9 := 
by
  sorry  -- proof to be provided

end one_minus_repeating_eight_l609_609284


namespace log_inequality_l609_609308

theorem log_inequality {a x y : ℝ} (ha : 0 < a) (h1 : a < 1) (h2 : x^2 + y = 0) :
  log a (a^x + a^y) ≤ log a 2 + (1 / 8) :=
sorry

end log_inequality_l609_609308


namespace ratio_area_triangle_BEF_to_square_ABCD_l609_609638

-- Defining the square and the points in it.
variables (s : ℝ) (A B C D E F : ℝ × ℝ)
variables (AE_ratio : ℝ) (DF_ratio : ℝ)

-- Given conditions.
def square_ABCD := A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ (A.1 = D.1) ∧ (A.2 - D.2 = s) ∧ (D.1 = C.1) ∧ (D.2 = C.2 - s) ∧ (C.1 = B.1) ∧ (C.2 = B.2 - s) ∧ (B.1 = A.1) ∧ (B.2 - A.2 = s)

def on_side_AD := E.1 = A.1 ∧ 0 ≤ E.2 ∧ E.2 ≤ s ∧ 2*E.2 = s
def on_side_DC := F.2 = C.2 ∧ 0 ≤ F.1 ∧ F.1 ≤ s ∧ 3*F.1 = s

-- The proof goal.
theorem ratio_area_triangle_BEF_to_square_ABCD (h1 : square_ABCD s A B C D) (h2 : on_side_AD A D E s) (h3 : on_side_DC C D F s) :
  (area_triangle A B F + area_triangle E D C + area_triangle E F C + area_triangle A E D + area_triangle B C D - area_triangle B E F) / (s * s) = 2 / 3 :=
sorry

end ratio_area_triangle_BEF_to_square_ABCD_l609_609638


namespace arithmetic_seq_root_problem_l609_609418

theorem arithmetic_seq_root_problem :
  ∃ (a_n : ℕ → ℚ),
  (∃ a_2 a_10 : ℚ, (2 * a_2^2 - a_2 - 7 = 0) ∧ (2 * a_10^2 - a_10 - 7 = 0) ∧
  (∀ n m, a_n = λ n : ℕ, a_2 + (n - m) * (a_10 - a_2) / (10 - 2)) ∧ 
  (a_2 + a_10 = 1 / 2)) → 
  a_n 6 = 1 / 4 :=
begin
  sorry
end

end arithmetic_seq_root_problem_l609_609418


namespace students_speaking_both_l609_609195

noncomputable def num_students := 200
noncomputable def pct_non_french := 0.75
noncomputable def pct_french := 1 - pct_non_french
noncomputable def num_french := pct_french * num_students
noncomputable def num_french_only := 40

theorem students_speaking_both :
  let num_non_french := pct_non_french * num_students
  let num_french_and_english := num_french - num_french_only
  num_non_french = 0.75 * 200 ∧
  num_french = 0.25 * 200 ∧
  num_french_and_english = 10 :=
by
  sorry

end students_speaking_both_l609_609195


namespace combined_population_port_perry_lazy_harbor_l609_609956

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end combined_population_port_perry_lazy_harbor_l609_609956


namespace probability_perfect_square_sum_l609_609125

def is_perfect_square_sum (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

def count_perfect_square_sums : ℕ :=
  let possible_outcomes := 216
  let favorable_outcomes := 32
  favorable_outcomes

theorem probability_perfect_square_sum :
  (count_perfect_square_sums : ℚ) / 216 = 4 / 27 :=
by
  sorry

end probability_perfect_square_sum_l609_609125


namespace problem1_problem2_l609_609928

-- Definition for the problem conditions.
def pi_minus_five_lt_zero : Prop := (π - 5 < 0)
def two_minus_pi : Prop := (2 - π)

-- Statement of the proof goal (questions rephrased as assertions under given conditions).
theorem problem1 (h1 : pi_minus_five_lt_zero) : (sqrt ((π-5)^2) - real.cbrt ((2-π)^3)) = 3 :=
by
  sorry

theorem problem2 : (0.06 * 4^(-1/3) + (-5/2)^0 - real.sqrt (9/4) + 0.1^(-2)) = 102 :=
by
  sorry

end problem1_problem2_l609_609928


namespace sequence_prime_count_eq_one_l609_609266

theorem sequence_prime_count_eq_one : 
  ∃! n, n ∈ ([37, 3737, 373737, ...]) ∧ nat.prime n := 
begin
  sorry
end

end sequence_prime_count_eq_one_l609_609266


namespace infinite_geometric_series_sum_l609_609642

theorem infinite_geometric_series_sum
  (a : ℚ) (r : ℚ) (h_a : a = 1) (h_r : r = 2 / 3) (h_r_abs_lt_one : |r| < 1) :
  ∑' (n : ℕ), a * r^n = 3 :=
by
  -- Import necessary lemmas and properties for infinite series
  sorry -- Proof is omitted.

end infinite_geometric_series_sum_l609_609642


namespace det_dilation_matrix_l609_609046

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![12, 0], ![0, 12]]

theorem det_dilation_matrix : Matrix.det E = 144 := by
  sorry

end det_dilation_matrix_l609_609046


namespace paul_initial_stock_l609_609456

def paul_pencils_proof : Prop :=
  ∀ (pencils_per_day : ℕ) (days : ℕ) (sold : ℕ) (end_stock : ℕ),
    pencils_per_day = 100 →
    days = 5 →
    sold = 350 →
    end_stock = 230 →
    (pencils_per_day * days + end_stock - sold = 380)

theorem paul_initial_stock : paul_pencils_proof :=
by
  intros pencils_per_day days sold end_stock
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    100 * 5 + 230 - 350 = 500 + 230 - 350 := rfl
    ... = 730 - 350 := rfl
    ... = 380 := rfl

end paul_initial_stock_l609_609456


namespace triangle_dimensions_after_cutting_square_diagonally_l609_609619

theorem triangle_dimensions_after_cutting_square_diagonally :
  ∀ (s : ℝ), s = 10 → ∃ (a b c : ℝ), a = 10 ∧ b = 10 ∧ c = 10 * Real.sqrt 2 :=
by
  intro s
  intro h_s
  use 10
  use 10
  use 10 * Real.sqrt 2
  split
  · refl
  split
  · refl
  · rw h_s
    refl
  sorry

end triangle_dimensions_after_cutting_square_diagonally_l609_609619


namespace factor_polynomial_l609_609677

theorem factor_polynomial (x : ℝ) : 54*x^3 - 135*x^5 = 27*x^3*(2 - 5*x^2) := 
by
  sorry

end factor_polynomial_l609_609677


namespace total_growing_space_l609_609634

theorem total_growing_space : 
  ∀ (beds1 beds2 : ℕ) (l1 w1 l2 w2 : ℕ), 
  beds1 = 2 → l1 = 3 → w1 = 3 → 
  beds2 = 2 → l2 = 4 → w2 = 3 → 
  (beds1 * (l1 * w1) + beds2 * (l2 * w2) = 42) :=
by
  intros beds1 beds2 l1 w1 l2 w2 h_beds1 h_l1 h_w1 h_beds2 h_l2 h_w2
  rw [h_beds1, h_l1, h_w1, h_beds2, h_l2, h_w2]
  norm_num
  sorry

end total_growing_space_l609_609634


namespace exp_rectangular_form_l609_609258

theorem exp_rectangular_form : exp(13 * π * I / 2) = I :=
by 
  let θ := 13 * π / 2
  have h1 : exp(θ * I) = cos(θ) + I * sin(θ), from Complex.exp_eq_cos_add_sin θ
  have h2 : cos(θ) = 0, from calc
    cos(θ) = cos(13 * π / 2) : by sorry -- periodic property simplification here
         ... = cos(π / 2) : by sorry -- simplified to fundamental period
         ... = 0 : by sorry
  have h3 : sin(θ) = 1, from calc
    sin(θ) = sin(13 * π / 2) : by sorry -- periodic property simplification here
         ... = sin(π / 2) : by sorry -- simplified to fundamental period
         ... = 1 : by sorry
  show exp(13 * π * I / 2) = 0 + I * 1, by rw [h1, h2, h3]
  show exp(13 * π * I / 2) = I, by simp

end

end exp_rectangular_form_l609_609258


namespace tank_full_capacity_l609_609572

theorem tank_full_capacity (full_capacity : ℝ) (h : 40% * full_capacity = (60% * full_capacity) - 54) : full_capacity = 270 := 
by sorry

end tank_full_capacity_l609_609572


namespace routeB_quicker_than_routeA_l609_609901

def routeA_time : ℕ := 
  let normal_zone_time := ((7 : ℚ) / 40) * 60
  let construction_zone_time := ((1 : ℚ) / 10) * 60
  normal_zone_time + construction_zone_time

def routeB_time : ℕ := 
  let normal_zone_time := ((6 : ℚ) / 50) * 60
  let traffic_zone_time := ((1 : ℚ) / 25) * 60
  normal_zone_time + traffic_zone_time

theorem routeB_quicker_than_routeA :
  routeB_time < routeA_time ∧ ((routeA_time - routeB_time) = 6.9 * 60) :=
by
  -- Skip the proof with sorry
  sorry

end routeB_quicker_than_routeA_l609_609901


namespace sum_divisible_by_four_l609_609080

theorem sum_divisible_by_four (n : ℤ) : 
  (2 * n - 1)^(2 * n + 1) + (2 * n + 1)^(2 * n - 1) % 4 = 0 :=
by
  sorry

end sum_divisible_by_four_l609_609080


namespace bob_password_probability_l609_609240

/-
Define the various sets and probabilities:
- odd_nums: the set of odd single-digit numbers
- uppercase_letters: the set of uppercase letters
- positive_single_digits: the set of non-zero single-digit numbers
- The probabilities associated with each set
-/

open Probability -- Assuming there's a module for handling probabilities

def odd_digit_probability : ℚ := 5 / 10 -- Probability of an odd single-digit number
def uppercase_letter_probability : ℚ := 26 / 52 -- Probability of an uppercase letter
def positive_digit_probability : ℚ := 9 / 10 -- Probability of a positive single-digit number

-- The final statement that we need to prove
theorem bob_password_probability :
  odd_digit_probability * uppercase_letter_probability * positive_digit_probability = 9 / 40 :=
by
  sorry

end bob_password_probability_l609_609240


namespace newer_pump_drainage_time_l609_609915

-- Define the conditions
variables {x : ℝ}

-- The given conditions as Lean definitions
def older_pump_rate : ℝ := 1 / 9
def combined_time : ℝ := 3.6
def combined_rate : ℝ := 1 / combined_time
def newer_pump_rate : ℝ := 1 / x

-- The equation from the problem's condition
def combined_rate_defn : Prop := older_pump_rate + newer_pump_rate = combined_rate

-- The proof problem in Lean
theorem newer_pump_drainage_time (h : combined_rate_defn) : x = 6 :=
by
  -- you can add the proof steps here
  sorry

end newer_pump_drainage_time_l609_609915


namespace gcd_2720_1530_l609_609103

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end gcd_2720_1530_l609_609103


namespace inscribed_sphere_radius_base_height_l609_609212

noncomputable def radius_of_inscribed_sphere (r base_radius height : ℝ) := 
  r = (30 / (Real.sqrt 5 + 1)) * (Real.sqrt 5 - 1) 

theorem inscribed_sphere_radius_base_height (r : ℝ) (b d : ℝ) (base_radius height : ℝ) 
  (h_base: base_radius = 15) (h_height: height = 30) 
  (h_radius: radius_of_inscribed_sphere r base_radius height) 
  (h_expr: r = b * (Real.sqrt d) - b) : 
  b + d = 12.5 :=
sorry

end inscribed_sphere_radius_base_height_l609_609212


namespace domain_log_composition_l609_609334

theorem domain_log_composition :
  (∀ x, x ≤ 1 → 0 < 2^x ∧ 2^x ≤ 2) →
  (∀ x, f.log₃(2 - x) ∈ Set.Iio 1 → -7 ≤ x ∧ x < 1) :=
by
  intro h
  sorry

end domain_log_composition_l609_609334


namespace lines_concurrent_l609_609055

theorem lines_concurrent (n : ℕ) (h₁ : n ≥ 3) (lines : Fin n → AffineSubspace ℝ ℝ) 
(h₂ : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, k ≠ i ∧ k ≠ j ∧ 
(lines i ∩ lines j).nonempty ∧ (lines k : Set (AffineSubspace ℝ ℝ)) ⊆ lines i ∩ lines j) : 
∃ P : AffineSubspace ℝ ℝ, ∀ i : Fin n, P ∈ lines i :=
sorry

end lines_concurrent_l609_609055


namespace julia_played_kids_l609_609035

theorem julia_played_kids (kids_Monday : ℕ) (kids_Tuesday : ℕ) :
  kids_Monday = 15 → kids_Tuesday = 18 → kids_Monday + kids_Tuesday = 33 :=
by
  -- assigning the conditions from problem statement
  intros h1 h2
  -- simplifying and substituting the given values 
  rw [h1, h2]
  -- proving the necessary result
  exact nat.add_comm 15 18 ▸ rfl

end julia_played_kids_l609_609035


namespace probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l609_609992

-- Probability for different numbers facing up when die is thrown twice
theorem probability_different_numbers :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := n_faces * (n_faces - 1)
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry -- Proof to be filled

-- Probability for sum of numbers being 6 when die is thrown twice
theorem probability_sum_six :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := 5
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 36 :=
by
  sorry -- Proof to be filled

-- Probability for exactly three outcomes being odd when die is thrown five times
theorem probability_three_odds_in_five_throws :
  let n_faces := 6
  let n_throws := 5
  let p_odd := 3 / n_faces
  let p_even := 1 - p_odd
  let binomial_coeff := Nat.choose n_throws 3
  let p_three_odds := (binomial_coeff : ℚ) * (p_odd ^ 3) * (p_even ^ 2)
  p_three_odds = 5 / 16 :=
by
  sorry -- Proof to be filled

end probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l609_609992


namespace percentage_of_girls_l609_609969

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 900) (h2 : B = 90) :
  (G / (B + G) : ℚ) * 100 = 90 :=
  by
  sorry

end percentage_of_girls_l609_609969


namespace decrease_in_demand_l609_609959

theorem decrease_in_demand:
  (P D : ℝ) (hP : P > 0) (hD : D > 0) :
  let new_price := 1.40 * P
      new_income := 1.10 * P * D
      new_demand := 1.10 * D / 1.40
      decrease := 1 - (new_demand / D) in
  decrease = 3 / 14 :=
by
  sorry

end decrease_in_demand_l609_609959


namespace unique_even_increasing_function_l609_609671

-- Define the functions
def f_A (x : ℝ) : ℝ := cos (2 * x)
def f_B (x : ℝ) : ℝ := (exp x - exp (-x)) / 2
def f_C (x : ℝ) : ℝ := x^3 + 1
def f_D (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Definitions for even functions and increasing functions
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

-- The interval (1, 2)
def interval_1_2 : Set ℝ := {x | 1 < x ∧ x < 2}

-- The main theorem
theorem unique_even_increasing_function :
  ∀ f : ℝ → ℝ,
    (f = f_A ∨ f = f_B ∨ f = f_C ∨ f = f_D) →
    is_even f →
    is_increasing_on f interval_1_2 →
    f = f_D :=
by
  intros f h_cases h_even h_increasing
  iterate 4 {
    cases h_cases with hf h_cases
    { sorry }
  }

end unique_even_increasing_function_l609_609671


namespace find_solutions_l609_609717

noncomputable def solution_exists (x y z p : ℝ) : Prop :=
  (x^2 - 1 = p * (y + z)) ∧
  (y^2 - 1 = p * (z + x)) ∧
  (z^2 - 1 = p * (x + y))

theorem find_solutions (x y z p : ℝ) :
  solution_exists x y z p ↔
  (x = (p + Real.sqrt (p^2 + 1)) ∧ y = (p + Real.sqrt (p^2 + 1)) ∧ z = (p + Real.sqrt (p^2 + 1)) ∨
   x = (p - Real.sqrt (p^2 + 1)) ∧ y = (p - Real.sqrt (p^2 + 1)) ∧ z = (p - Real.sqrt (p^2 + 1))) ∨
  (x = (Real.sqrt (1 - p^2)) ∧ y = (Real.sqrt (1 - p^2)) ∧ z = (-p - Real.sqrt (1 - p^2)) ∨
   x = (-Real.sqrt (1 - p^2)) ∧ y = (-Real.sqrt (1 - p^2)) ∧ z = (-p + Real.sqrt (1 - p^2))) :=
by
  -- Proof starts here
  sorry

end find_solutions_l609_609717


namespace cross_product_correct_l609_609290

  def a : ℝ × ℝ × ℝ := (5, 2, -6)
  def b : ℝ × ℝ × ℝ := (1, 1, 3)
  def cross_product (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
    (x.2 * y.3 - x.3 * y.2, x.3 * y.1 - x.1 * y.3, x.1 * y.2 - x.2 * y.1)

  theorem cross_product_correct : cross_product a b = (12, -21, 3) :=
  sorry
  
end cross_product_correct_l609_609290


namespace whitewashing_cost_l609_609099

/-- Dimensions of the room:
  length: 25 feet,
  width: 15 feet,
  height: 12 feet. -/
variables (length width height : ℕ)
variables (door_height door_width : ℕ)
variables (window_height window_width : ℕ)
variables (number_of_windows : ℕ)
variables (cost_per_sqft : ℕ)
variables (net_area_to_be_whitewashed : ℕ)
variables (total_cost : ℕ)

-- Room dimensions
def room_dimensions : Prop :=
  length = 25 ∧
  width = 15 ∧
  height = 12

-- Door dimensions
def door_dimensions : Prop :=
  door_height = 6 ∧
  door_width = 3

-- Window dimensions and count
def window_dimensions : Prop :=
  window_height = 4 ∧
  window_width = 3 ∧
  number_of_windows = 3

-- Cost per square foot
def cost_info : Prop :=
  cost_per_sqft = 10

-- Total net area to be whitewashed
def net_area_calc : Prop :=
  net_area_to_be_whitewashed = (2 * length * height + 2 * width * height) - (door_height * door_width + number_of_windows * window_height * window_width)

-- Total cost calculation
def total_cost_calc : Prop :=
  total_cost = net_area_to_be_whitewashed * cost_per_sqft

theorem whitewashing_cost : room_dimensions length width height ∧ door_dimensions door_height door_width ∧ window_dimensions window_height window_width number_of_windows ∧ cost_info cost_per_sqft ∧ net_area_calc length width height door_height door_width window_height window_width number_of_windows net_area_to_be_whitewashed ∧ total_cost_calc net_area_to_be_whitewashed cost_per_sqft total_cost → total_cost = 9060 :=
by
  sorry

end whitewashing_cost_l609_609099


namespace square_area_percentage_error_l609_609198

variable (L : ℝ)

def measured_length (L : ℝ) : ℝ := L + 0.03 * L
def measured_width (L : ℝ) : ℝ := L - 0.02 * L
def actual_area (L : ℝ) : ℝ := L * L
def measured_area (L : ℝ) : ℝ := (measured_length L) * (measured_width L)
def percentage_error (L : ℝ) : ℝ := ((measured_area L - actual_area L) / actual_area L) * 100

theorem square_area_percentage_error : percentage_error L = 0.94 := by
  sorry

end square_area_percentage_error_l609_609198


namespace Vanya_correct_answers_l609_609475

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l609_609475


namespace simplify_complex_expression_l609_609081

theorem simplify_complex_expression : 
  (complex.normSq (2 + 3*complex.I) / complex.normSq (2 - 3*complex.I))^8 * 3 = 3 :=
by
  sorry

end simplify_complex_expression_l609_609081


namespace doubled_cylinder_volume_original_cylinder_holds_3_gallons_l609_609190

def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem doubled_cylinder_volume (V : ℝ) (r h : ℝ) 
  (hV: volume_of_cylinder r h = V) : volume_of_cylinder (2*r) (2*h) = 8 * V := by
  sorry

theorem original_cylinder_holds_3_gallons : ∀ (r h : ℝ), volume_of_cylinder r h = 3 → volume_of_cylinder (2*r) (2*h) = 24 :=
  by
  intros r h h1
  exact doubled_cylinder_volume 3 r h h1

end doubled_cylinder_volume_original_cylinder_holds_3_gallons_l609_609190


namespace vanya_correct_answers_l609_609488

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609488


namespace distinct_balls_one_empty_identical_balls_one_empty_distinct_balls_empty_boxes_identical_balls_empty_boxes_l609_609724

-- Problem 1: Distinct balls, one box empty
theorem distinct_balls_one_empty :
  (number_of_ways_to_place_distinct_balls_one_box_empty 4 4 = 144) :=
sorry

-- Problem 2: Identical balls, one box empty
theorem identical_balls_one_empty :
  (number_of_ways_to_place_identical_balls_one_box_empty 4 4 = 12) :=
sorry

-- Problem 3: Distinct balls, empty boxes allowed
theorem distinct_balls_empty_boxes :
  (number_of_ways_to_place_distinct_balls_with_empty_boxes_allowed 4 4 = 256) :=
sorry

-- Problem 4: Identical balls, empty boxes allowed
theorem identical_balls_empty_boxes :
  (number_of_ways_to_place_identical_balls_with_empty_boxes_allowed 4 4 = 35) :=
sorry

end distinct_balls_one_empty_identical_balls_one_empty_distinct_balls_empty_boxes_identical_balls_empty_boxes_l609_609724


namespace Buckingham_palace_visitors_difference_l609_609223

theorem Buckingham_palace_visitors_difference :
  ∀ (visitors_today visitors_yesterday : ℕ),
  visitors_today = 317 →
  visitors_yesterday = 295 →
  (visitors_today - visitors_yesterday) = 22 :=
by
  intros visitors_today visitors_yesterday H_today H_yesterday
  rw [H_today, H_yesterday]
  exact Nat.sub_eq_iff_eq_add.mpr (rfl)
sorry

end Buckingham_palace_visitors_difference_l609_609223


namespace circumradius_of_isosceles_triangle_l609_609238

-- Define the sides of the isosceles triangle
def a := 13
def b := 13
def c := 10

-- Define the semi-perimeter of the triangle
def s := (a + b + c) / 2

-- Calculate the area of the triangle using Heron's formula
def K := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Calculate the circumradius of the triangle
def R := (a * b * c) / (4 * K)

-- The target statement we want to prove
theorem circumradius_of_isosceles_triangle : R = 169 / 24 := 
  by
  -- Use this placeholder to skip the proof
  sorry

end circumradius_of_isosceles_triangle_l609_609238


namespace Vanya_correct_answers_l609_609480

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l609_609480


namespace min_a4_b2_l609_609885

open Real

noncomputable def min_value (t : ℝ) : ℝ :=
  (t ^ 4) / 16 + (t ^ 2) / 4

theorem min_a4_b2 (a b t : ℝ) (h : a + b = t) :
  ∃ a b, a + b = t ∧ a ^ 4 + b ^ 2 = min_value t :=
begin
  use [t / 2, t / 2],
  split,
  { simp, },
  { simp [min_value],
    ring, sorry, }
end

end min_a4_b2_l609_609885


namespace victoria_should_return_22_l609_609995

theorem victoria_should_return_22 :
  let initial_money := 50
  let pizza_cost_per_box := 12
  let pizzas_bought := 2
  let juice_cost_per_pack := 2
  let juices_bought := 2
  let total_spent := (pizza_cost_per_box * pizzas_bought) + (juice_cost_per_pack * juices_bought)
  let money_returned := initial_money - total_spent
  money_returned = 22 :=
by
  sorry

end victoria_should_return_22_l609_609995


namespace find_f_rational_l609_609192

noncomputable theory

def f : ℝ → ℝ :=
  λ x, if x ∈ (-1:ℝ, 0] then x^3 else sorry -- placeholder for the actual function definition

theorem find_f_rational :
  (∀ x : ℝ, f (x + 1) = 2 * f x) →
  (∀ x : ℝ, x ∈ (-1, 0] → f x = x ^ 3) →
  f (21 / 2) = -256 :=
by sorry

end find_f_rational_l609_609192


namespace new_sign_cost_l609_609219

theorem new_sign_cost 
  (p_s : ℕ) (p_c : ℕ) (n : ℕ) (h_ps : p_s = 30) (h_pc : p_c = 26) (h_n : n = 10) : 
  (p_s - p_c) * n / 2 = 20 := 
by 
  sorry

end new_sign_cost_l609_609219


namespace purchase_price_correct_l609_609111

def purchase_price (P : ℝ) : Prop :=
  (0.15 * P + 12 = 40)

theorem purchase_price_correct : ∃ P, purchase_price P ∧ P = 186.67 :=
by
  use 186.67
  have h1 : 0.15 * 186.67 + 12 = 40 := by norm_num
  exact ⟨h1, rfl⟩

end purchase_price_correct_l609_609111


namespace find_f_1991_l609_609039

namespace FunctionProof

-- Defining the given conditions as statements in Lean
def func_f (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

def poly_g (f g : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, g n = g (f n)

-- Statement of the problem
theorem find_f_1991 
  (f g : ℤ → ℤ)
  (Hf : func_f f)
  (Hg : poly_g f g) :
  f 1991 = -1992 := 
sorry

end FunctionProof

end find_f_1991_l609_609039


namespace derivative_problem1_derivative_problem2_l609_609712

-- Proof statement for the first function
theorem derivative_problem1 (x : ℝ):
  has_deriv_at (λ x, (x + 1) * (x + 2) * (x + 3)) (3 * x^2 + 12 * x + 11) x :=
sorry

-- Proof statement for the second function
theorem derivative_problem2 (x : ℝ):
  has_deriv_at (λ x, 2 * x * tan x) ((2 * sin x * cos x + 2 * x) / cos x^2) x :=
sorry

end derivative_problem1_derivative_problem2_l609_609712


namespace four_digit_numbers_with_3_or_4_l609_609360

theorem four_digit_numbers_with_3_or_4 : 
  let total_count := 9000 in
  let digit_choices_first := 7 in
  let digit_choices_other := 8 in
  let count_without_3_or_4 := digit_choices_first * digit_choices_other^3 in
  let count_with_3_or_4 := total_count - count_without_3_or_4 in
  count_with_3_or_4 = 5416 :=
by 
  let total_count := 9000
  let digit_choices_first := 7
  let digit_choices_other := 8
  let count_without_3_or_4 := digit_choices_first * digit_choices_other^3
  let count_with_3_or_4 := total_count - count_without_3_or_4
  show count_with_3_or_4 = 5416
  from sorry

end four_digit_numbers_with_3_or_4_l609_609360


namespace correct_result_l609_609575

theorem correct_result (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end correct_result_l609_609575


namespace inequality_abc_l609_609781

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l609_609781


namespace inequality_abc_l609_609783

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l609_609783


namespace curved_surface_area_of_sphere_l609_609097

theorem curved_surface_area_of_sphere (r : ℝ) (h : r = 4) : 4 * π * r^2 = 64 * π :=
by
  rw [h, sq]
  norm_num
  sorry

end curved_surface_area_of_sphere_l609_609097


namespace tan_pi_add_theta_l609_609813

theorem tan_pi_add_theta (θ : ℝ) (h : Real.tan (Real.pi + θ) = 2) : 
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 :=
by
  sorry

end tan_pi_add_theta_l609_609813


namespace geometric_sequence_sum_l609_609867

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (∑ k in Finset.range (n + 1), a k)

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

theorem geometric_sequence_sum 
  (h_geo : is_geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_S6 : S 6 = 4)
  (h_S18 : S 18 = 28) :
  S 12 = 12 :=
begin
  sorry
end

end geometric_sequence_sum_l609_609867


namespace repeatable_transformation_l609_609624

theorem repeatable_transformation (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
  (2 * c > a + b) ∧ (2 * a > b + c) ∧ (2 * b > c + a) := 
sorry

end repeatable_transformation_l609_609624


namespace factorization_l609_609693

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609693


namespace math_problem_l609_609203

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, a = b * b

theorem math_problem (a m : ℕ) (h1: m = 2992) (h2: a = m^2 + m^2 * (m+1)^2 + (m+1)^2) : is_perfect_square a :=
  sorry

end math_problem_l609_609203


namespace isosceles_right_triangle_AC_length_l609_609746

theorem isosceles_right_triangle_AC_length (A B C K L M N : Point)
  (hTriangle : isIsoscelesRightTriangle A B C ∧ angle_at A = 90)
  (hSquare : isSquare KLMN ∧ K ∈ Segment AB ∧ L ∈ Segment BC ∧ N ∈ Segment AC ∧ M ∈ Triangle ABC)
  (hAK : distance A K = 7)
  (hAN : distance A N = 3) :
  distance A C = 17 :=
sorry

end isosceles_right_triangle_AC_length_l609_609746


namespace find_x_squared_inv_x_squared_l609_609752

theorem find_x_squared_inv_x_squared (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 :=
sorry

end find_x_squared_inv_x_squared_l609_609752


namespace possible_values_of_m_l609_609365

theorem possible_values_of_m (a b : ℤ) (h1 : a * b = -14) :
  ∃ m : ℤ, m = a + b ∧ (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
by
  sorry

end possible_values_of_m_l609_609365


namespace ants_approximate_count_l609_609613

theorem ants_approximate_count 
  (width_ft : ℕ) (length_ft : ℕ) 
  (ants_per_sq_inch : ℕ) (inches_per_foot : ℕ) :
  width_ft = 200 →
  length_ft = 500 →
  ants_per_sq_inch = 5 →
  inches_per_foot = 12 →
  width_ft * inches_per_foot * length_ft * inches_per_foot * ants_per_sq_inch = 72000000 :=
by
  intros w_ft l_ft ap_si ipf hw hl ha hi
  subst hw hl ha hi
  calc
    (200 * 12) * (500 * 12) * 5 = 2400 * 6000 * 5 : by congr
    ... = 14400000 * 5 : by norm_num
    ... = 72000000 : by norm_num
 
#eval ants_approximate_count (200 : ℕ) (500 : ℕ) (5 : ℕ) (12 : ℕ)  rfl rfl rfl rfl

end ants_approximate_count_l609_609613


namespace part1_part2_l609_609798

open Real

-- Definitions based on the conditions
def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n ≥ 2, a n = a (n - 1) + 1 / a (n - 1))

noncomputable def a (n : ℕ) : ℝ :=
if h : n = 1 then 1 else
  have hn : n ≥ 2 := (n - 1) + 1,
  a (n - 1) + 1 / a (n - 1)

-- Statement 1: Prove \( \forall n \geq 2, a_n \geq \sqrt{2n}\).
theorem part1 (a : ℕ → ℝ) (h : sequence a) : ∀ n ≥ 2, a n ≥ sqrt (2 * n) :=
sorry

-- Statement 2: Prove there is no real number \( C \) such that \( a_n < \sqrt{2n + C} \) holds for all \( n \).
theorem part2 (a : ℕ → ℝ) (h : sequence a) : ¬∃ C : ℝ, ∀ n, a n < sqrt (2 * n + C) :=
sorry

end part1_part2_l609_609798


namespace percentage_problem_l609_609718

theorem percentage_problem :
  ∃ P : ℝ, ((P / 100) * 1442 - (0.36 * 1412)) + 63 = 3 ∧ P ≈ 31.08 :=
by
  sorry

end percentage_problem_l609_609718


namespace yumi_counted_value_l609_609578

theorem yumi_counted_value (original_number : Int) (reductions : Int) (decrement : Int) 
  (times : Int) (expected_value : Int) : 
  original_number = 320 → reductions = 4 → decrement = 10 → times = 4 → expected_value = 280 →
  (original_number - reductions * decrement = expected_value) :=
by
  intros
  rw [←sub_mul]
  exact rfl

-- includes the given conditions of figure it 4, the initial value, the reduction steps etc.
-- the conditions from a) should be expressed within theorem's parameter and assumptions.
-- the goal is to match the resulting value of 280 starting from 320 and then performing 4 reductions of 10
-- (320 - 4 * 10) = 280, which corresponds to the final calculated value.

end yumi_counted_value_l609_609578


namespace part_I_part_II_l609_609348

noncomputable def f (x a : ℝ) : ℝ := (2 * x - 4) * Real.exp(x) + a * (x + 2)^2

theorem part_I (a : ℝ) (h : a ≥ 1 / 2) : 
  ∀ {x : ℝ}, x > 0 → ∃ (f' : ℝ → ℝ), f' = λ x, (2 * x - 2) * Real.exp x + 2 * a * (x + 2) → ∀ x > 0, f' x ≥ 0 := 
sorry

theorem part_II (a : ℝ) (h : 0 < a ∧ a < 1 / 2) :
  ∃ t ∈ (0, 1), ∀ x, f x a > f t a ∧ f t = (2 * t - 4) * Real.exp t - (t - 1) * (t + 2) * Real.exp t ∧ 
  -2 * Real.exp 1 < (2 * t - 4) * Real.exp t - (t - 1) * (t + 2) * Real.exp t < -2 := 
sorry

end part_I_part_II_l609_609348


namespace valid_duty_schedules_l609_609602

noncomputable def validSchedules : ℕ := 
  let A_schedule := Nat.choose 7 4  -- \binom{7}{4} for A
  let B_schedule := Nat.choose 4 4  -- \binom{4}{4} for B
  let C_schedule := Nat.choose 6 3  -- \binom{6}{3} for C
  let D_schedule := Nat.choose 5 5  -- \binom{5}{5} for D
  A_schedule * B_schedule * C_schedule * D_schedule

theorem valid_duty_schedules : validSchedules = 700 := by
  -- proof steps will go here
  sorry

end valid_duty_schedules_l609_609602


namespace johns_elevation_after_travel_l609_609031

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end johns_elevation_after_travel_l609_609031


namespace minimum_radius_circumscribed_sphere_l609_609315

-- Given conditions: Right triangular prism and side area
variables (x y : ℝ)
axiom angle_BAC_right : angle BAC = 90
axiom area_BCCB : 4 * x * y = 16

-- The radius of the circumscribed sphere
def radius_of_circumscribed_sphere (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2)

-- Minimum radius proof problem statement
theorem minimum_radius_circumscribed_sphere :
  ∃ (r : ℝ), r = 2 * sqrt 2 ∧ ∀ (x y : ℝ), 4 * x * y = 16 → r ≤ sqrt (x^2 + y^2) := by
  sorry

end minimum_radius_circumscribed_sphere_l609_609315


namespace price_per_strawberry_basket_is_9_l609_609903

-- Define the conditions
def strawberry_plants := 5
def tomato_plants := 7
def strawberries_per_plant := 14
def tomatoes_per_plant := 16
def items_per_basket := 7
def price_per_tomato_basket := 6
def total_revenue := 186

-- Define the total number of strawberries and tomatoes harvested
def total_strawberries := strawberry_plants * strawberries_per_plant
def total_tomatoes := tomato_plants * tomatoes_per_plant

-- Define the number of baskets of strawberries and tomatoes
def strawberry_baskets := total_strawberries / items_per_basket
def tomato_baskets := total_tomatoes / items_per_basket

-- Define the revenue from tomato baskets
def revenue_tomatoes := tomato_baskets * price_per_tomato_basket

-- Define the revenue from strawberry baskets
def revenue_strawberries := total_revenue - revenue_tomatoes

-- Calculate the price per basket of strawberries (which should be $9)
def price_per_strawberry_basket := revenue_strawberries / strawberry_baskets

theorem price_per_strawberry_basket_is_9 : 
  price_per_strawberry_basket = 9 := by
    sorry

end price_per_strawberry_basket_is_9_l609_609903


namespace Vanya_correct_answers_l609_609497

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l609_609497


namespace work_rate_B_l609_609579

theorem work_rate_B (A B : Type) (work : A → ℝ) (work_together : work A = 1/6) (work_A : work B = 1/11) : work B = 1/13.2 :=
sorry

end work_rate_B_l609_609579


namespace num_men_in_second_group_l609_609366

def total_work_hours_week (men: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  men * hours_per_day * days_per_week

def earnings_per_man_hour (total_earnings: ℕ) (total_work_hours: ℕ) : ℚ :=
  total_earnings / total_work_hours

def required_man_hours (total_earnings: ℕ) (earnings_per_hour: ℚ) : ℚ :=
  total_earnings / earnings_per_hour

def number_of_men (total_man_hours: ℚ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℚ :=
  total_man_hours / (hours_per_day * days_per_week)

theorem num_men_in_second_group :
  let hours_per_day_1 := 10
  let hours_per_day_2 := 6
  let days_per_week := 7
  let men_1 := 4
  let earnings_1 := 1000
  let earnings_2 := 1350
  let work_hours_1 := total_work_hours_week men_1 hours_per_day_1 days_per_week
  let rate_1 := earnings_per_man_hour earnings_1 work_hours_1
  let work_hours_2 := required_man_hours earnings_2 rate_1
  number_of_men work_hours_2 hours_per_day_2 days_per_week = 9 := by
  sorry

end num_men_in_second_group_l609_609366


namespace proof_problem_l609_609946

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotone_increasing : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ x ≤ y → f(x) ≤ f(y)
axiom f_even_property : ∀ (x : ℝ), f(x + 2) = f(2 - x)

theorem proof_problem : (f(7 / 2) < f(1) ∧ f(1) < f(5 / 2)) :=
by
  sorry

end proof_problem_l609_609946


namespace fem_current_age_l609_609064

theorem fem_current_age (F : ℕ) 
  (h1 : ∃ M : ℕ, M = 4 * F) 
  (h2 : (F + 2) + (4 * F + 2) = 59) : 
  F = 11 :=
sorry

end fem_current_age_l609_609064


namespace hyperbola_eccentricity_l609_609865

-- Definition of the parabola C1: y^2 = 2px with p > 0.
def parabola (p : ℝ) (p_pos : 0 < p) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Definition of the hyperbola C2: x^2 / a^2 - y^2 / b^2 = 1 with a > 0 and b > 0.
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

-- Definition of having a common focus F at (p / 2, 0).
def common_focus (p a b c : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) : Prop := 
  c = p / 2 ∧ c^2 = a^2 + b^2

-- Definition for points A and B on parabola C1 and point M on hyperbola C2.
def points_A_B_M (c a b : ℝ) (x1 y1 x2 y2 yM : ℝ) : Prop := 
  x1 = c ∧ y1 = 2 * c ∧ x2 = c ∧ y2 = -2 * c ∧ yM = b^2 / a

-- Condition for OM, OA, and OB relation and mn = 1/8.
def OM_OA_OB_relation (m n : ℝ) : Prop := 
  m * n = 1 / 8

-- Theorem statement: Given the conditions, the eccentricity of hyperbola C2 is √6 + √2 / 2.
theorem hyperbola_eccentricity (p a b c m n : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) :
  parabola p p_pos c (2 * c) → 
  hyperbola a b a_pos b_pos c (b^2 / a) → 
  common_focus p a b c p_pos a_pos b_pos →
  points_A_B_M c a b c (2 * c) c (-2 * c) (b^2 / a) →
  OM_OA_OB_relation m n → 
  m * n = 1 / 8 →
  ∃ e : ℝ, e = (Real.sqrt 6 + Real.sqrt 2) / 2 :=
sorry

end hyperbola_eccentricity_l609_609865


namespace intersection_of_complements_l609_609896

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem intersection_of_complements (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 5}) :
  (U \ M) ∩ (U \ N) = {4} :=
by
  rw [hU, hM, hN]
  dsimp
  sorry

end intersection_of_complements_l609_609896


namespace coins_count_l609_609989

noncomputable def total_number_of_coins (coin20_count coin25_count : ℕ) : bool :=
  let total_value_paise := 7100
  let value_of_20_paise_coins := 20 * coin20_count
  let value_of_25_paise_coins := 25 * coin25_count
  (coin20_count = 250) ∧ (value_of_20_paise_coins + value_of_25_paise_coins = total_value_paise)

theorem coins_count (coin20_count coin25_count total_coins : ℕ) :
  total_number_of_coins coin20_count coin25_count → total_coins = 334 :=
by
  sorry

end coins_count_l609_609989


namespace find_pyramid_volume_l609_609092

noncomputable def volume_of_pyramid (α β R : ℝ) : ℝ :=
  (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β

theorem find_pyramid_volume (α β R : ℝ) 
  (base_isosceles : ∀ {a b c : ℝ}, a = b) -- Represents the isosceles triangle condition
  (dihedral_angles_equal : ∀ {angle : ℝ}, angle = β) -- Dihedral angle at the base
  (circumcircle_radius : {radius : ℝ // radius = R}) -- Radius of the circumcircle
  (height_through_point : true) -- Condition: height passes through a point inside the triangle
  :
  volume_of_pyramid α β R = (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β :=
by {
  sorry
}

end find_pyramid_volume_l609_609092


namespace ratio_of_volumes_l609_609563

def cone_radius_X := 10
def cone_height_X := 15
def cone_radius_Y := 15
def cone_height_Y := 10

noncomputable def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h

noncomputable def volume_X := volume_cone cone_radius_X cone_height_X
noncomputable def volume_Y := volume_cone cone_radius_Y cone_height_Y

theorem ratio_of_volumes : volume_X / volume_Y = 2 / 3 := sorry

end ratio_of_volumes_l609_609563


namespace percentage_error_in_area_is_correct_l609_609201

-- Given conditions
variables (L : ℝ) -- actual length of the side of the square
def length_with_error := 1.03 * L
def width_with_error := 0.98 * L
def actual_area := L^2
def calculated_area := length_with_error * width_with_error

-- We must prove that the percentage error in the calculated area of the square is 2.94%
theorem percentage_error_in_area_is_correct :
  (calculated_area - actual_area) / actual_area * 100 = 2.94 := by
  sorry

end percentage_error_in_area_is_correct_l609_609201


namespace negation_proposition_l609_609537

-- Definitions related to the problem
variables {A B C : Type} -- representing the vertices of the triangle
variables {a b c : ℝ} -- representing the lengths of the sides opposite to the angles at vertices A, B, and C respectively
variables {α β γ : ℝ} -- representing the measures of the angles at vertices A, B, and C respectively

-- Conditions for the triangle
axiom angle_inequality : α > β
axiom sides_opposite_angles : ∀ {A B C : Type}, a = side_length_opposite_angle α ∧ b = side_length_opposite_angle β
-- The specific claim to be proven
theorem negation_proposition : (∃ (α β : ℝ) (a b : ℝ), α > β ∧ a ≤ b) :=
sorry

end negation_proposition_l609_609537


namespace integral_of_x_squared_l609_609674

/-- Prove that the integral of \(x^2\) from \(-1\) to \(1\) equals \(\frac{2}{3}\). -/
theorem integral_of_x_squared :
  ∫ x in -1..1, x^2 = 2 / 3 :=
by
  -- Proof omitted
  sorry

end integral_of_x_squared_l609_609674


namespace part1_div1_part1_div2_part2_div_part3_div_l609_609550

theorem part1_div1 {a : ℚ} (h : a ≠ 0) (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : a = 1/2) :
  (a ^ 5) / (a ^ 2) = 1 / 8 := sorry

theorem part1_div2 {a : ℚ} (h : a ≠ 0) (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : a = 4) :
  (a ^ 3) / (a ^ 5) = 1 / 16 := sorry

theorem part2_div {x : ℚ} (h : 3 ^ (x - 1) / 3 ^ (3 * x - 4) = 1 / 27) :
  x = 3 := sorry

theorem part3_div {x : ℚ} (h : (x - 1) ^ (2 * x + 2) / (x - 1) ^ (x + 6) = 1) :
  x = 4 ∨ x = 0 ∨ x = 2 := sorry

end part1_div1_part1_div2_part2_div_part3_div_l609_609550


namespace product_of_midpoint_coords_l609_609567

theorem product_of_midpoint_coords :
  let A : (ℝ × ℝ) := (3, 6)
  let B : (ℝ × ℝ) := (-4, 10)
  let midpoint : (ℝ × ℝ) := ((fst A + fst B) / 2, (snd A + snd B) / 2)
  (fst midpoint) * (snd midpoint) = -4 :=
by
  -- Definitions of A, B, and midpoint
  let A := (3 : ℝ, 6 : ℝ)
  let B := (-4 : ℝ, 10 : ℝ)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  
  -- Check product of coordinates
  suffices (fst midpoint) * (snd midpoint) = -4 by
    sorry

  -- Check values of coordinates
  have x_mid : (fst midpoint) = -1 / 2 := sorry
  have y_mid : (snd midpoint) = 8 := sorry
  
  -- Combine to check final product
  calc (fst midpoint) * (snd midpoint)
      = (-1 / 2) * 8 : by rw [x_mid, y_mid]
  ... = -4            : by norm_num
  
  sorry

end product_of_midpoint_coords_l609_609567


namespace remainder_hx10_div_hx_l609_609880

noncomputable def h (x : ℕ) := x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_hx10_div_hx (x : ℕ) : (h x ^ 10) % (h x) = 7 := by
  sorry

end remainder_hx10_div_hx_l609_609880


namespace max_possible_sum_l609_609720

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, 2 ≤ m → m < n → n % m ≠ 0

def length (n : ℕ) : ℕ := 
  if n == 1 then 0 
  else (Multiset.card (Multiset.filter (λ p, p.prime) (Multiset.repeat p 1)))

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then
    Multiset.min' (Multiset.filter is_prime (Multiset.map id (Multiset.range (n + 1)))) sorry
  else 0

theorem max_possible_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) (h3 : x + 3 * y < 1000)
  (h4 : ∃ p, p ∣ x ∧ ∃ k, k % 2 = 0 ∧ x = p ^ k)
  (h5 : ∃ q, q ∣ y ∧ ∃ l, l % 2 = 0 ∧ y = q ^ l) (p := smallest_prime_factor x)
  (q := smallest_prime_factor y) (h6 : p + q % 3 = 0) : 
  length x + length y = 13 := 
sorry

end max_possible_sum_l609_609720


namespace decreasing_condition_l609_609732

noncomputable def is_decreasing (y : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 < x2 → y x1 > y x2

noncomputable def abs (x : ℝ) := if x ≥ 0 then x else -x

theorem decreasing_condition (a : ℝ) :
  (abs (a - 1) + abs a ≤ 1) → ¬is_decreasing (λ x, a ^ x) :=
sorry

end decreasing_condition_l609_609732


namespace conclusion_l609_609829

def balls : Finset ℕ := (Finset.range 6).map (Function.Embedding.coeFn (Finset.range 1)) ∪ (Finset.range 7 10).map (Function.Embedding.coeFn (Finset.range 1))

def n_black : ℕ := 6
def n_white : ℕ := 4
def n_total : ℕ := n_black + n_white
def num_selected : ℕ := 4

/-
The main conditions are:
1. There are 6 black balls and 4 white balls, making 10 balls total.
2. 4 balls are randomly selected.
-/

theorem conclusion (X : ℕ) (Y : ℕ) (P : ℕ → ℚ) : 
  (¬ (X = hypergeometric_dist)) ∧ 
  (Y = hypergeometric_dist) ∧ 
  (P 2 ≠ 1 / 14) ∧ 
  (prob_score_high_score 2 1 2 = 1 / 14) :=
sorry

end conclusion_l609_609829


namespace skye_race_l609_609830

noncomputable def first_part_length := 3

theorem skye_race 
  (total_track_length : ℕ := 6)
  (speed_first_part : ℕ := 150)
  (distance_second_part : ℕ := 2)
  (speed_second_part : ℕ := 200)
  (distance_third_part : ℕ := 1)
  (speed_third_part : ℕ := 300)
  (avg_speed : ℕ := 180) :
  first_part_length = 3 :=
  sorry

end skye_race_l609_609830


namespace probability_same_color_l609_609625

theorem probability_same_color :
  let Abe := [(blue, 2), (green, 1)],
      Bob := [(blue, 1), (green, 2), (yellow, 1)],
      Cara := [(blue, 3), (green, 2), (red, 1)] in
  let P (jelly_count : List (Color × ℕ)) (color : Color) :=
        (jelly_count.filter (λ p, p.1 = color)).sum (λ p, p.2) / (jelly_count.sum (λ p, p.2) : ℚ) in
  (P Abe blue * P Bob blue * P Cara blue + P Abe green * P Bob green * P Cara green) = 5 / 36 :=
sorry

end probability_same_color_l609_609625


namespace total_growing_space_correct_l609_609632

-- Define the dimensions of the garden beds
def length_bed1 : ℕ := 3
def width_bed1 : ℕ := 3
def num_bed1 : ℕ := 2

def length_bed2 : ℕ := 4
def width_bed2 : ℕ := 3
def num_bed2 : ℕ := 2

-- Define the areas of the individual beds and total growing space
def area_bed1 : ℕ := length_bed1 * width_bed1
def total_area_bed1 : ℕ := area_bed1 * num_bed1

def area_bed2 : ℕ := length_bed2 * width_bed2
def total_area_bed2 : ℕ := area_bed2 * num_bed2

def total_growing_space : ℕ := total_area_bed1 + total_area_bed2

-- The theorem proving the total growing space
theorem total_growing_space_correct : total_growing_space = 42 := by
  sorry

end total_growing_space_correct_l609_609632


namespace smallest_y_with_factors_18_and_28_l609_609534

theorem smallest_y_with_factors_18_and_28 (y : ℕ) (h1 : dvd 18 y) (h2 : dvd 28 y) (h3 : y.num_factors = 24) : y = 504 :=
sorry

end smallest_y_with_factors_18_and_28_l609_609534


namespace units_digit_p_plus_two_l609_609754

theorem units_digit_p_plus_two (p q : ℕ) (h_even : p % 2 = 0) (h_pos : p > 0) 
  (h_pos_digit : p % 10 ∈ {2, 4, 6, 8}) (h_eq_0 : (p ^ 3) % 10 = (p ^ 2) % 10) 
  (h_div_by_q : q.prime ∧ (∑ d in Nat.digits 10 p, d) % q = 0) : 
  (p + 2) % 10 = 8 := 
sorry

end units_digit_p_plus_two_l609_609754


namespace bank_card_payment_technology_order_l609_609981

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end bank_card_payment_technology_order_l609_609981


namespace num_distinct_real_roots_ffx_eq_0_l609_609801

noncomputable def f : ℝ → ℝ := λ x, x^2 - 3*x + 2

theorem num_distinct_real_roots_ffx_eq_0 :
  ∃ S : set ℝ, (∀ x ∈ S, f (f x) = 0) ∧ S.finite ∧ S.card = 4 :=
sorry

end num_distinct_real_roots_ffx_eq_0_l609_609801


namespace Vanya_correct_answers_l609_609495

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l609_609495


namespace complement_intersection_l609_609804

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 3}

-- Define the complement of A with respect to U
def complement_U (s : Set ℕ) : Set ℕ := {x ∈ U | x ∉ s}

-- Define the statement to be proven
theorem complement_intersection :
  (complement_U A ∩ B) = {2} :=
by
  sorry

end complement_intersection_l609_609804


namespace sin_a6_plus_b6_eq_neg_half_l609_609868

open Real

theorem sin_a6_plus_b6_eq_neg_half {a : ℕ → ℝ} {b : ℕ → ℝ}
  (hS11 : S 11 = (11 / 3) * π)
  (hS11_def : S 11 = 11 * a 6)
  (hb4b8_eq : ∀ x, 4 * x ^ 2 + 100 * x + π ^ 2 = 0 → x = b 4 ∨ x = b 8)
  (hb4b8_sum : b 4 + b 8 = -25)
  (hb4b8_prod : b 4 * b 8 = (π ^ 2) / 4)
  (hb4_neg : b 4 < 0)
  (hb8_neg : b 8 < 0) :
  sin (a 6 + b 6) = -1 / 2 :=
by
  sorry

end sin_a6_plus_b6_eq_neg_half_l609_609868


namespace divide_water_equally_l609_609120

-- Define the type for the state of the barrels
structure BarrelState :=
  (barrel1 : ℕ) -- number of buckets in barrel 1
  (barrel2 : ℕ) -- number of buckets in barrel 2
  (barrel3 : ℕ) -- number of buckets in barrel 3
  (barrel4 : ℕ) -- number of buckets in barrel 4

-- Define the initial state
def initial_state : BarrelState := {
  barrel1 := 24,
  barrel2 := 0,
  barrel3 := 0,
  barrel4 := 0
}

-- Define the target state
def target_state : BarrelState := {
  barrel1 := 8,
  barrel2 := 8,
  barrel3 := 8,
  barrel4 := 0
}

-- The theorem that there exists a sequence of transfers to partition water equally
theorem divide_water_equally : ∃ (transfers : list (BarrelState → BarrelState)), 
  (transfers.foldl (λ s f, f s) initial_state = target_state) := 
sorry

end divide_water_equally_l609_609120


namespace parallel_BC_MN_l609_609009

/-- In an acute-triangle ABC, the angle bisector of ∠A intersects side BC at D.
    The circle centered at B with radius BD intersects side AB at M.
    The circle centered at C with radius CD intersects side AC at N.
    Prove that BC is parallel to MN. -/
theorem parallel_BC_MN {A B C D M N : Point}
  (h_acute : is_acute_triangle A B C)
  (h_angle_bisector_A : angle_bisector A B C D)
  (h_circle_B : circle_eq_radius B BD AB M)
  (h_circle_C : circle_eq_radius C CD AC N) :
  parallel BC MN :=
sorry

end parallel_BC_MN_l609_609009


namespace find_c_l609_609539

theorem find_c (c q : ℤ) (h : ∃ (a b : ℤ), (3*x^3 + c*x + 9 = (x^2 + q*x + 1) * (a*x + b))) : c = -24 :=
sorry

end find_c_l609_609539


namespace alice_reeboks_sold_l609_609630

theorem alice_reeboks_sold
  (quota : ℝ)
  (price_adidas : ℝ)
  (price_nike : ℝ)
  (price_reeboks : ℝ)
  (num_nike : ℕ)
  (num_adidas : ℕ)
  (excess : ℝ)
  (total_sales_goal : ℝ)
  (total_sales : ℝ)
  (sales_nikes_adidas : ℝ)
  (sales_reeboks : ℝ)
  (num_reeboks : ℕ) :
  quota = 1000 →
  price_adidas = 45 →
  price_nike = 60 →
  price_reeboks = 35 →
  num_nike = 8 →
  num_adidas = 6 →
  excess = 65 →
  total_sales_goal = quota + excess →
  total_sales = 1065 →
  sales_nikes_adidas = price_nike * num_nike + price_adidas * num_adidas →
  sales_reeboks = total_sales - sales_nikes_adidas →
  num_reeboks = sales_reeboks / price_reeboks →
  num_reeboks = 9 :=
by
  intros
  sorry

end alice_reeboks_sold_l609_609630


namespace frog_eyes_in_pond_l609_609227

-- Definitions based on conditions
def num_frogs : ℕ := 6
def eyes_per_frog : ℕ := 2

-- The property to be proved
theorem frog_eyes_in_pond : num_frogs * eyes_per_frog = 12 :=
by
  sorry

end frog_eyes_in_pond_l609_609227


namespace rectangular_prism_sum_of_dimensions_l609_609975

theorem rectangular_prism_sum_of_dimensions (a b c : ℕ) (h_volume : a * b * c = 21) 
(h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
a + b + c = 11 :=
sorry

end rectangular_prism_sum_of_dimensions_l609_609975


namespace even_func_smallest_period_pi_l609_609737

def func (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem even_func_smallest_period_pi : 
  (∀ x : ℝ, func x = func (-x)) ∧ (∃ T > 0, ∀ x : ℝ, func (x + T) = func x ∧ T = Real.pi) := by
  sorry

end even_func_smallest_period_pi_l609_609737


namespace lottery_probability_exactly_one_common_l609_609398

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609398


namespace sum_reciprocals_pq_p_plus_q_eq_43_l609_609863

noncomputable def B : Set ℕ :=
  {n | ∀ p ∣ n, p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7}

theorem sum_reciprocals_pq :
  let s := ∑' (n : ℕ) in B, (1 : ℚ) / n
  s = 35 / 8 ∧ Nat.gcd 35 8 = 1 :=
by
  sorry

theorem p_plus_q_eq_43 :
  let s := ∑' (n : ℕ) in B, (1 : ℚ) / n
  s = 35 / 8 → 35 + 8 = 43 :=
by
  sorry

end sum_reciprocals_pq_p_plus_q_eq_43_l609_609863


namespace N_is_even_l609_609851

def sum_of_digits : ℕ → ℕ := sorry

theorem N_is_even 
  (N : ℕ)
  (h1 : sum_of_digits N = 100)
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N :=
sorry

end N_is_even_l609_609851


namespace percentage_differences_equal_l609_609141

noncomputable def calculation1 : ℝ := 0.60 * 50
noncomputable def calculation2 : ℝ := 0.30 * 30
noncomputable def calculation3 : ℝ := 0.45 * 90
noncomputable def calculation4 : ℝ := 0.20 * 40

noncomputable def diff1 : ℝ := abs (calculation1 - calculation2)
noncomputable def diff2 : ℝ := abs (calculation2 - calculation3)
noncomputable def diff3 : ℝ := abs (calculation3 - calculation4)
noncomputable def largest_diff1 : ℝ := max diff1 (max diff2 diff3)

noncomputable def calculation5 : ℝ := 0.40 * 120
noncomputable def calculation6 : ℝ := 0.25 * 80
noncomputable def calculation7 : ℝ := 0.35 * 150
noncomputable def calculation8 : ℝ := 0.55 * 60

noncomputable def diff4 : ℝ := abs (calculation5 - calculation6)
noncomputable def diff5 : ℝ := abs (calculation6 - calculation7)
noncomputable def diff6 : ℝ := abs (calculation7 - calculation8)
noncomputable def largest_diff2 : ℝ := max diff4 (max diff5 diff6)

theorem percentage_differences_equal :
  largest_diff1 = largest_diff2 :=
sorry

end percentage_differences_equal_l609_609141


namespace distance_from_A_l609_609618

theorem distance_from_A (s : ℝ) (h_area : s ^ 2 = 16) 
  (h_fold : ∀ A : ℝ, 
    let x := (4 * real.sqrt 6) / 3 in 
    (16 - (1 / 2) * x ^ 2 = (1 / 2) * x ^ 2)) :
  ∃ d : ℝ, d = (8 * real.sqrt 3) / 3 :=
by
  sorry

end distance_from_A_l609_609618


namespace sufficient_condition_l609_609320

variables (a b : ℝ^3)
variables (ha : a ≠ 0)
variables (hb : b ≠ 0)
variables (h : ∥a - b∥ = ∥a∥ + ∥b∥)

theorem sufficient_condition (a b : ℝ^3) (ha : a ≠ 0) (hb : b ≠ 0)
    (h : ∥a - b∥ = ∥a∥ + ∥b∥) : a + 2 * b = 0 :=
sorry

end sufficient_condition_l609_609320


namespace probability_of_one_common_l609_609405

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l609_609405


namespace contrapositive_example_l609_609067

theorem contrapositive_example (a b : ℝ) (h : a^2 + b^2 < 4) : a + b ≠ 3 :=
sorry

end contrapositive_example_l609_609067


namespace cost_effectiveness_order_l609_609599

variables {cS cM cL qS qM qL : ℝ}
variables (h1 : cM = 2 * cS)
variables (h2 : qM = 0.7 * qL)
variables (h3 : qL = 3 * qS)
variables (h4 : cL = 1.2 * cM)

theorem cost_effectiveness_order :
  (cL / qL <= cM / qM) ∧ (cM / qM <= cS / qS) :=
by
  sorry

end cost_effectiveness_order_l609_609599


namespace cos_inv_one_third_div_pi_irrational_l609_609922

theorem cos_inv_one_third_div_pi_irrational : ¬ ∃ (r : ℚ), (real.arccos (1 / 3)) = r * real.pi :=
sorry

end cos_inv_one_third_div_pi_irrational_l609_609922


namespace starting_player_ensures_non_trivial_solution_l609_609549

theorem starting_player_ensures_non_trivial_solution :
  ∀ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℚ), 
    ∃ (x y z : ℚ), 
    ((a1 * x + b1 * y + c1 * z = 0) ∧ 
     (a2 * x + b2 * y + c2 * z = 0) ∧ 
     (a3 * x + b3 * y + c3 * z = 0)) 
    ∧ ((a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2) = 0) ∧ 
         (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by
  intros a1 b1 c1 a2 b2 c2 a3 b3 c3
  sorry

end starting_player_ensures_non_trivial_solution_l609_609549


namespace existence_of_two_marked_cells_along_diagonal_l609_609904

def marked_cells_condition (n : ℕ) (is_marked : ℕ × ℕ → Prop) : Prop :=
  ¬ is_marked (1, 1) ∧ ¬ is_marked (n, n) ∧
    ∀ path : list (ℕ × ℕ), path.head = (1, 1) ∧ path.last = (n, n) →
    (∀ (i : ℕ), 0 < i → i < path.length → is_marked (path.nth i))

theorem existence_of_two_marked_cells_along_diagonal (n : ℕ) (h : 3 < n) :
  (∃ k : ℕ, n = 3 * k + 1) →
  ∃ (is_marked : ℕ × ℕ → Prop), marked_cells_condition n is_marked →
    ∃ (i j : ℕ), (1 ≤ i ∧ i + 2 ≤ n ∧ ∃ (d : ℕ), d ∈ [0..n-1] ∧ is_marked (i + d, j + d) ∧ is_marked (i + d + 1, j + d + 1)) :=

sorry

end existence_of_two_marked_cells_along_diagonal_l609_609904


namespace paper_pieces_possible_l609_609263

theorem paper_pieces_possible (k : ℕ) : ∃ k, 1003 = 3 * k + 1 :=
by
have h : 1003 = 3 * 334 + 1,
  exact by rw [mul_add, mul_one, add_assoc, nat.succ_eq_add_one],
existsi 334,
exact h
sorry -- proof is to be completed here

end paper_pieces_possible_l609_609263


namespace min_value_f_on_interval_l609_609292

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

theorem min_value_f_on_interval : ∃ y ∈ set.Icc (0 : ℝ) 1, f y = -1 ∧ ∀ x ∈ set.Icc (0 : ℝ) 1, f y ≤ f x :=
by {
  sorry -- proof to be filled in 
}

end min_value_f_on_interval_l609_609292


namespace rectangular_prism_sum_l609_609655

theorem rectangular_prism_sum : 
  let edges := 12
  let vertices := 8
  let faces := 6
  edges + vertices + faces = 26 := by
sorry

end rectangular_prism_sum_l609_609655


namespace max_good_set_size_l609_609586

def A := { v : Fin 8 → ℕ // ∀ i, 1 ≤ v i ∧ v i ≤ i + 1 }

def is_good_set (X : Set A) : Prop :=
  ∀ (x y : A), x ≠ y → (∃ i1 i2 i3 : Fin 8, i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧ x.1 i1 ≠ y.1 i1 ∧ x.1 i2 ≠ y.1 i2 ∧ x.1 i3 ≠ y.1 i3)

-- Main theorem
theorem max_good_set_size : ∃ X : Set A, is_good_set X ∧ Set.card X = 5040 := 
by sorry

end max_good_set_size_l609_609586


namespace right_triangle_integral_sides_parity_l609_609924

theorem right_triangle_integral_sides_parity 
  (a b c : ℕ) 
  (h : a^2 + b^2 = c^2) 
  (ha : a % 2 = 1 ∨ a % 2 = 0) 
  (hb : b % 2 = 1 ∨ b % 2 = 0) 
  (hc : c % 2 = 1 ∨ c % 2 = 0) : 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) := 
sorry

end right_triangle_integral_sides_parity_l609_609924


namespace rationalize_denominator_eq_l609_609074

-- Define the original expression
def original_expr : ℚ(√3) := (1 + real.sqrt 3) / (1 - real.sqrt 3)

-- Define the target form A + B * sqrt(C)
def target_expr : ℚ(√3) := -2 - real.sqrt 3

-- The main theorem statement
theorem rationalize_denominator_eq :
  original_expr = target_expr :=
sorry

end rationalize_denominator_eq_l609_609074


namespace equal_numbers_product_l609_609510

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l609_609510


namespace vanya_correct_answers_l609_609483

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l609_609483


namespace find_value_six_x_plus_two_y_l609_609807

-- Let the vectors be defined as follows
variables {x y : ℝ}
def a : ℝ × ℝ × ℝ := (2 * x, 1, 3)
def b : ℝ × ℝ × ℝ := (1, -2 * y, 9)

-- Define the condition that the vectors are parallel
def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
    ∃ k : ℝ, a = (k * b.1, k * b.2, k * b.3)

-- The theorem we want to prove
theorem find_value_six_x_plus_two_y (h : are_parallel a b) : 6 * x + 2 * y = -2 :=
by
  sorry

end find_value_six_x_plus_two_y_l609_609807


namespace inequality_abc_l609_609780

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l609_609780


namespace emma_time_l609_609673

theorem emma_time (E : ℝ) (h1 : 2 * E + E = 60) : E = 20 :=
sorry

end emma_time_l609_609673


namespace euler_thirteen_pi_half_l609_609260

noncomputable def convertEuler (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem euler_thirteen_pi_half :
  convertEuler (13 * real.pi / 2) = complex.I :=
by
  -- Given Euler's formula: e^(ix) = cos(x) + i*sin(x)
  have euler_formula : ∀ x : ℝ, complex.exp (complex.I * x) = complex.of_real (real.cos x) + complex.I * real.sin x,
    from complex.exp_eq_cos_add_sin,
  -- Simplify using Euler's formula and the periodicity of cos and sin
  rw [← euler_formula (13 * real.pi / 2)],
  -- Simplify further
  rw [real.cos_add_pi_div_two, real.sin_add_pi_div_two],
  sorry

end euler_thirteen_pi_half_l609_609260


namespace correct_order_of_actions_l609_609986

-- Definitions based on the conditions
def actions : ℕ → String
| 1 => "tap"
| 2 => "pay online"
| 3 => "swipe"
| 4 => "insert into terminal"
| _ => "undefined"

def paymentTechnology : ℕ → String
| 1 => "PayPass"
| 2 => "CVC"
| 3 => "magnetic stripe"
| 4 => "chip"
| _ => "undefined"

-- Proof problem statement
theorem correct_order_of_actions :
  (actions 4 = "insert into terminal") ∧
  (actions 3 = "swipe") ∧
  (actions 1 = "tap") ∧
  (actions 2 = "pay online") →
  [4, 3, 1, 2] corresponds to ["chip", "magnetic stripe", "PayPass", "CVC"]
:=
by
  sorry

end correct_order_of_actions_l609_609986


namespace remaining_crayons_proof_l609_609547

def initial_crayons : ℕ := 360
def kiley_fraction : ℚ := 7 / 12
def joe_fraction : ℚ := 11 / 18

def kiley_crayons (initial: ℕ) (fraction: ℚ) : ℕ := (initial * fraction).toNat
def remaining_after_kiley (initial: ℕ) (kiley: ℕ) : ℕ := initial - kiley
def joe_crayons (remaining: ℕ) (fraction: ℚ) : ℕ := (remaining * fraction).toNat
def remaining_after_joe (remaining: ℕ) (joe: ℕ) : ℕ := remaining - joe

theorem remaining_crayons_proof : remaining_after_joe (remaining_after_kiley initial_crayons (kiley_crayons initial_crayons kiley_fraction)) 
                                    (joe_crayons (remaining_after_kiley initial_crayons (kiley_crayons initial_crayons kiley_fraction)) joe_fraction) = 59 := by
sorry

end remaining_crayons_proof_l609_609547


namespace polynomial_sum_squares_l609_609726

theorem polynomial_sum_squares (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ)
  (h₁ : (1 - 2) ^ 7 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)
  (h₂ : (1 + -2) ^ 7 = a0 - a1 + a2 - a3 + a4 - a5 + a6 - a7) :
  (a0 + a2 + a4 + a6) ^ 2 - (a1 + a3 + a5 + a7) ^ 2 = -2187 := 
  sorry

end polynomial_sum_squares_l609_609726


namespace one_minus_repeating_eight_l609_609283

-- Given the condition
def b : ℚ := 8 / 9

-- The proof problem statement
theorem one_minus_repeating_eight : 1 - b = 1 / 9 := 
by
  sorry  -- proof to be provided

end one_minus_repeating_eight_l609_609283


namespace problem_a_problem_b_l609_609860

def isGermanSet {n : ℕ} (grid : Fin n × Fin n → ℕ) (s : Fin n → Fin n × Fin n) : Prop :=
  Function.Injective s

def germanProduct {n : ℕ} (grid : Fin n × Fin n → ℕ) (s : Fin n → Fin n × Fin n) : ℕ :=
  Finset.univ.prod (λ i => grid (s i))

def existsGermanLabeling (n : ℕ) (d : ℕ) : Prop :=
  ∃ grid : Fin n × Fin n → ℕ,
    ∀ s1 s2 : Fin n → Fin n × Fin n,
      isGermanSet grid s1 →
      isGermanSet grid s2 →
      grid (s1 0) ≠ grid (s2 0) →
      (germanProduct grid s1 - germanProduct grid s2) % d = 0

theorem problem_a : ¬ existsGermanLabeling 8 65 :=
  sorry

theorem problem_b : existsGermanLabeling 10 101 :=
  sorry

end problem_a_problem_b_l609_609860


namespace finite_primes_not_in_seq_l609_609047

/-- Define the sequence a_n where 
  a_1 ≥ 1 and a_n = ⌊√(n * a_(n-1))⌋ for n ≥ 2 -/
def seq (a : ℕ) : ℕ → ℕ
| 1        := a
| (n + 1)  := (Nat.floor (Real.sqrt (n + 1) * (seq n)))

-- Proof statement: There are only finitely many prime numbers that do not appear in the sequence.
theorem finite_primes_not_in_seq (a : ℤ) (h : a ≥ 1) :
  {p ∈ (finset.primes) | ∀ n, seq a (n + 1) ≠ p}.finite := sorry

end finite_primes_not_in_seq_l609_609047


namespace other_root_of_quadratic_eq_l609_609914

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l609_609914


namespace correct_action_order_l609_609983

inductive Action
| tap : Action
| pay_online : Action
| swipe : Action
| insert_into_terminal : Action
deriving DecidableEq, Repr, Inhabited

inductive Technology
| chip : Technology
| magnetic_stripe : Technology
| paypass : Technology
| cvc : Technology
deriving DecidableEq, Repr, Inhabited

def action_for_technology : Technology → Action
| Technology.chip := Action.insert_into_terminal
| Technology.magnetic_stripe := Action.swipe
| Technology.paypass := Action.tap
| Technology.cvc := Action.pay_online

theorem correct_action_order :
  [action_for_technology Technology.chip, action_for_technology Technology.magnetic_stripe,
   action_for_technology Technology.paypass, action_for_technology Technology.cvc] = 
  [Action.insert_into_terminal, Action.swipe, Action.tap, Action.pay_online] := 
sorry

end correct_action_order_l609_609983


namespace ratio_of_area_of_triangle_DEF_to_square_ABCD_l609_609840

theorem ratio_of_area_of_triangle_DEF_to_square_ABCD (ABCD : Type) [square ABCD]
  (E F : Point) (H₁ : E ∈ segment AB) (H₂ : F ∈ segment AB)
  (H₃ : ∠ ADE + ∠ EDF + ∠ FDC = 90°) (H₄ : ∠ ADE = ∠ EDF) (H₅ : ∠ EDF = ∠ FDC) :
  (area (triangle DEF)) / (area ABCD) = 1 / 8 := 
sorry

end ratio_of_area_of_triangle_DEF_to_square_ABCD_l609_609840


namespace spatial_body_translation_is_prism_l609_609117

theorem spatial_body_translation_is_prism (polygon : Type) (translation : polygon → polygon) 
    (h : ∀ (p : polygon), translation p = p) : 
    (spatial_geometric_body_formed_by translation) = "prism" := 
by
  sorry

end spatial_body_translation_is_prism_l609_609117


namespace intersection_complement_l609_609445

def R := Set ℝ
def A : Set ℝ := {-3, -2, -1, 0, 1, 2, 3}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement :
  A ∩ (R \ B) = {-3, -2, -1, 0, 1} := by
  sorry

end intersection_complement_l609_609445


namespace number_of_children_admitted_l609_609090

variable (children adults : ℕ)

def admission_fee_children : ℝ := 1.5
def admission_fee_adults  : ℝ := 4

def total_people : ℕ := 315
def total_fees   : ℝ := 810

theorem number_of_children_admitted :
  ∃ (C A : ℕ), C + A = total_people ∧ admission_fee_children * C + admission_fee_adults * A = total_fees ∧ C = 180 :=
by
  sorry

end number_of_children_admitted_l609_609090


namespace magnitude_of_sum_l609_609806

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := sorry  -- need more structure to rigorously define this, but can be directly implied

def vec_angle (v1 v2 : ℝ × ℝ) : ℝ := sorry  -- This should ideally compute the angle between two vectors
def vec_mag (v : ℝ × ℝ) := real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_sum :
  vec_angle a b = real.pi / 4 → vec_mag b = 1 → vec_mag (a.1 + b.1, a.2 + b.2) = real.sqrt 5 :=
sorry

end magnitude_of_sum_l609_609806


namespace compute_expression_l609_609658

theorem compute_expression : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end compute_expression_l609_609658


namespace num_ways_399_as_sum_of_consecutive_ints_l609_609012

theorem num_ways_399_as_sum_of_consecutive_ints :
  (∃ n k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 ≥ n ∧ n * (2 * k + n - 1) = 798) ∧
  (finset.univ.filter (λ n, ∃ k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 ≥ n ∧ n * (2 * k + n - 1) = 798)).card = 6 :=
by sorry

end num_ways_399_as_sum_of_consecutive_ints_l609_609012


namespace seating_arrangement_l609_609217

def numWaysCableCars (adults children cars capacity : ℕ) : ℕ := 
  sorry 

theorem seating_arrangement :
  numWaysCableCars 4 2 3 3 = 348 :=
by {
  sorry
}

end seating_arrangement_l609_609217


namespace count_sequences_l609_609084

noncomputable def S (m k : ℕ) : ℕ :=
if m = 1 then
  0
else
  (k - 1) * (-1)^m + (k - 1)^m

theorem count_sequences (m k : ℕ) :
  let S_m := S m k in
  (S_m = (k - 1) * (-1)^m + (k - 1)^m) ↔
  (1 ≤ k ∧ m = 1 → S_m = 0) ∧
  (1≤m ∧ 1 ≤ k → S_m = (k - 1) * (-1)^m + (k - 1)^m) := by
  sorry

end count_sequences_l609_609084


namespace vanya_correct_answers_l609_609465

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l609_609465


namespace scaling_transformation_l609_609135

theorem scaling_transformation (a b : ℝ) :
  (∀ x y : ℝ, (y = 1 - x → y' = b * (1 - x))
    → (y' = b - b * x)) 
  ∧
  (∀ x' y' : ℝ, (y = (2 / 3) * x' + 2)
    → (y' = (2 / 3) * (a * x) + 2))
  → a = 3 ∧ b = 2 := by
  sorry

end scaling_transformation_l609_609135


namespace coefficient_of_x_cubed_l609_609998

noncomputable def P (x : ℚ) : ℚ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 8 * x + 5
noncomputable def Q (x : ℚ) : ℚ := 2 * x^2 - 3 * x + 7

theorem coefficient_of_x_cubed : (λ x, (P x) * (Q x)).coeff 3 = -26 :=
by
  sorry

end coefficient_of_x_cubed_l609_609998


namespace vanya_correct_answers_l609_609486

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l609_609486


namespace unique_filling_of_polyhedron_l609_609309

theorem unique_filling_of_polyhedron :
  ∀ (faces : Finset ℝ) (f : ℕ → ℝ), (fin 2022) → ℝ,
  f 0 = 26 → f 1 = 4 → f 2 = 2022 →
  (∀ face : fin 2022, face ≠ 0 ∧ face ≠ 1 ∧ face ≠ 2 →
    f face = (∑ edge in edges face, f (shared_face edge)) / (count shared_face edges face)) →
  ∃! f' : (fin 2022) → ℝ, f' = f :=
begin
  sorry
end

end unique_filling_of_polyhedron_l609_609309


namespace magnitude_of_sum_l609_609333

open Real

-- Define the vectors and their properties
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (cos (π / 4), sin (π / 4))

-- Define the magnitudes
def mag_a : ℝ := sqrt ((1:ℝ)^2 + (0:ℝ)^2)
def mag_b : ℝ := sqrt 2

theorem magnitude_of_sum : abs (2 * a.1 + b.1, 2 * a.2 + b.2) = sqrt 10 :=
by 
  -- The conditions and calculations from the problem are used to define the theorem to be proved
  sorry

end magnitude_of_sum_l609_609333


namespace graphQ_is_linear_with_slope_zero_l609_609151

def is_linear_with_slope_zero (graph : Type) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, graph x = c

constant GraphP GraphQ GraphR GraphS GraphT : Type

theorem graphQ_is_linear_with_slope_zero :
  is_linear_with_slope_zero GraphQ := sorry

end graphQ_is_linear_with_slope_zero_l609_609151


namespace compare_values_l609_609777

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l609_609777


namespace num_distinct_element_products_l609_609435

noncomputable theory

def divisors (n : ℕ) : finset ℕ := (finset.Icc 1 n).filter (λ d, n % d = 0)

def T : finset ℕ := divisors 72000

-- Count the number of products of two distinct elements of T resulting in a valid number
theorem num_distinct_element_products : (T.prod (λ x, T.filter (λ y, y ≠ x).card) = 450) :=
sorry

end num_distinct_element_products_l609_609435


namespace kingfishers_percentage_l609_609000

theorem kingfishers_percentage (total_birds hawks paddyfield_warblers others non_hawks kingfishers : ℕ)
  (h1 : hawks = 30 / 100 * total_birds)
  (h2 : non_hawks = total_birds - hawks)
  (h3 : paddyfield_warblers = 40 / 100 * non_hawks)
  (h4 : others = 35 / 100 * total_birds)
  (h5 : kingfishers = x / 100 * paddyfield_warblers)
  (h6 : total_birds = hawks + paddyfield_warblers + kingfishers + others) :
  x = 25 :=
begin
  sorry
end

end kingfishers_percentage_l609_609000


namespace product_of_equal_numbers_l609_609516

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l609_609516


namespace angle_QRT_15_l609_609662

noncomputable theory

variables {P Q R S T U : Point}
variables {a b c d e f : Real} -- Assuming some variables for coordinates if points need to be referred in the proof.

-- Definition of equilateral triangle
def equilateral_triangle (P Q R : Point) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = dist P Q

-- Definition of square 
def square (R S T U : Point) : Prop :=
  dist R S = dist S T ∧ dist S T = dist T U ∧ dist T U = dist U R ∧ dist U R = dist R S ∧
  angle R S T = 90 ∧ angle S T U = 90 ∧ angle T U R = 90 ∧ angle U R S = 90

-- Theorem statement
theorem angle_QRT_15 {P Q R S T U : Point}
  (h₁ : equilateral_triangle P Q R)
  (h₂ : square R S T U)
  (h₃ : coplanar {P, Q, R, S, T, U}) : 
  angle Q R T = 15 := 
sorry

end angle_QRT_15_l609_609662


namespace find_lambda_for_collinear_find_lambda_for_right_angle_triangle_l609_609436

def mutually_perpendicular_unit_vectors (e1 e2 : ℝ) : Prop :=
  ∥e1∥ = 1 ∧ ∥e2∥ = 1 ∧ e1 ⬝ e2 = 0

def vectors (e1 e2 : ℝ) :=
  3 * e1 + 2 * e2

def another_vector (e1 e2 : ℝ) (λ : ℝ) :=
  e1 - λ * e2

def vector_cd (e1 e2 : ℝ) :=
  -2 * e1 + e2

theorem find_lambda_for_collinear (e1 e2 : ℝ) (λ : ℝ) :
  mutually_perpendicular_unit_vectors e1 e2 →
  vectors e1 e2 = 3 * e1 + 2 * e2 →
  another_vector e1 e2 λ = e1 - λ * e2 →
  vector_cd e1 e2 = -2 * e1 + e2 →
  ∃ μ : ℝ, λ = -3 :=
sorry

theorem find_lambda_for_right_angle_triangle (e1 e2 : ℝ) (λ : ℝ) :
  mutually_perpendicular_unit_vectors e1 e2 →
  vectors e1 e2 = 3 * e1 + 2 * e2 →
  another_vector e1 e2 λ = e1 - λ * e2 →
  vector_cd e1 e2 = -2 * e1 + e2 →
  λ = -3 ∨ λ = -1 ∨ λ = (7 / 2) :=
sorry

end find_lambda_for_collinear_find_lambda_for_right_angle_triangle_l609_609436


namespace shirley_cases_l609_609500

-- Given conditions
def T : ℕ := 54  -- boxes of Trefoils sold
def S : ℕ := 36  -- boxes of Samoas sold
def M : ℕ := 48  -- boxes of Thin Mints sold
def t_per_case : ℕ := 4  -- boxes of Trefoils per case
def s_per_case : ℕ := 3  -- boxes of Samoas per case
def m_per_case : ℕ := 5  -- boxes of Thin Mints per case

-- Amount of boxes delivered per case should meet the required demand
theorem shirley_cases : ∃ (n_cases : ℕ), 
  n_cases * t_per_case ≥ T ∧ 
  n_cases * s_per_case ≥ S ∧ 
  n_cases * m_per_case ≥ M :=
by
  use 14
  sorry

end shirley_cases_l609_609500


namespace percent_students_with_C_is_25_l609_609411

-- Define the grading scale ranges
def A_range (score : ℕ) : Prop := 95 ≤ score ∧ score ≤ 100
def B_range (score : ℕ) : Prop := 87 ≤ score ∧ score ≤ 94
def C_range (score : ℕ) : Prop := 78 ≤ score ∧ score ≤ 86
def D_range (score : ℕ) : Prop := 65 ≤ score ∧ score ≤ 77
def F_range (score : ℕ) : Prop := 0 ≤ score ∧ score ≤ 64

-- List of scores of the students
def scores := [49, 58, 65, 77, 84, 70, 88, 94, 55, 82, 60, 86, 68, 74, 99, 81, 73, 79, 53, 91]

-- Function to count the number of students who fall within a range
def count_students_in_range (scores : List ℕ) (range : ℕ → Prop) : ℕ :=
scores.countp range

-- Number of students
def total_students : ℕ := scores.length

-- Number of students who received a grade C
def students_with_C := count_students_in_range scores C_range

-- Percentage calculation
def percentage (num : ℕ) (den : ℕ) : ℚ := (num.to_q / den.to_q) * 100

-- Main theorem: Prove that the percentage of students who got a C is 25%
theorem percent_students_with_C_is_25 : percentage students_with_C total_students = 25 := 
by
  sorry

end percent_students_with_C_is_25_l609_609411


namespace floor_x_plus_x_eq_13_div_3_l609_609706

-- Statement representing the mathematical problem
theorem floor_x_plus_x_eq_13_div_3 (x : ℚ) (h : ⌊x⌋ + x = 13/3) : x = 7/3 := 
sorry

end floor_x_plus_x_eq_13_div_3_l609_609706


namespace initial_blocks_of_ann_l609_609222
-- Import the necessary library

-- Define the conditions and the question to be proved
theorem initial_blocks_of_ann (final_blocks : ℕ) (found_blocks : ℕ) (initial_blocks : ℕ) 
  (h1 : final_blocks = 53) (h2 : found_blocks = 44) : initial_blocks = 9 :=
by
  -- Use the conditions h1 and h2 to state the required proof
  have ha : initial_blocks = final_blocks - found_blocks,
  sorry

end initial_blocks_of_ann_l609_609222


namespace price_of_one_table_l609_609580

variable (C T : ℝ)

def cond1 := 2 * C + T = 0.6 * (C + 2 * T)
def cond2 := C + T = 60
def solution := T = 52.5

theorem price_of_one_table (h1 : cond1 C T) (h2 : cond2 C T) : solution T :=
by
  sorry

end price_of_one_table_l609_609580


namespace other_root_l609_609908

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l609_609908


namespace correct_action_order_l609_609985

inductive Action
| tap : Action
| pay_online : Action
| swipe : Action
| insert_into_terminal : Action
deriving DecidableEq, Repr, Inhabited

inductive Technology
| chip : Technology
| magnetic_stripe : Technology
| paypass : Technology
| cvc : Technology
deriving DecidableEq, Repr, Inhabited

def action_for_technology : Technology → Action
| Technology.chip := Action.insert_into_terminal
| Technology.magnetic_stripe := Action.swipe
| Technology.paypass := Action.tap
| Technology.cvc := Action.pay_online

theorem correct_action_order :
  [action_for_technology Technology.chip, action_for_technology Technology.magnetic_stripe,
   action_for_technology Technology.paypass, action_for_technology Technology.cvc] = 
  [Action.insert_into_terminal, Action.swipe, Action.tap, Action.pay_online] := 
sorry

end correct_action_order_l609_609985


namespace bound_on_sum_and_reciprocal_sum_l609_609053

theorem bound_on_sum_and_reciprocal_sum (a b : ℝ) (n : ℕ) (x : ℕ → ℝ) 
  (h₁ : 0 < a) (h₂ : a < b) (h₃ : ∀ i, 1 ≤ i ∧ i ≤ n → a ≤ x i ∧ x i ≤ b) :
  (∑ i in Finset.range n, x i) * (∑ i in Finset.range n, (x i)⁻¹) ≤ ((a + b)^2 / (4 * a * b)) * n^2 := by
  sorry

end bound_on_sum_and_reciprocal_sum_l609_609053


namespace find_three_tuple_solutions_l609_609709

open Real

theorem find_three_tuple_solutions :
  (x y z : ℝ) → (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z)
  → (3 * x^2 + 2 * y^2 + z^2 = 240)
  → (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) :=
by
  intro x y z
  intro h1 h2
  sorry

end find_three_tuple_solutions_l609_609709


namespace total_spent_l609_609300

variables {B D : ℝ}

def ben_david_spend_total (B D : ℝ) : Prop :=
  D = 0.60 * B ∧ B = D + 16 ∧ B + D = 64

theorem total_spent (B D : ℝ) (h : ben_david_spend_total B D) : B + D = 64 :=
  by {
    exact h.2.2,
  }

end total_spent_l609_609300


namespace is_increasing_sequence_l609_609750

universe u
  
variable {α : Type u} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) (n : ℕ) :=
  ∃ (a_1 d : α), ∀ m, a (m + 1) = a_1 + m * d

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) :=
  ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem is_increasing_sequence (a : ℕ → α) (S : ℕ → α)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (h_condition : ∀ n, n ≥ 2 → S n < n * a n) :
  ∀ m, a (m + 1) > a m :=
by
  intros
  sorry

end is_increasing_sequence_l609_609750


namespace quadrilateral_area_l609_609304

section
variables {α : Type*} [linear_ordered_field α] [real.sin α]
noncomputable def AreaQuadrilateral (β : α) : α := 1 / 2 * (real.sin β) ^ 2

theorem quadrilateral_area (A B C D : point) (R1 R2 : ray)
  (h₁ : square ABCD 1)
  (h₂ : intersects_vertex R1 R2 A)
  (h₃ : angle_between R1 R2 = β) :
  quadrilateral_from_perpendiculars_area B D R1 R2 = AreaQuadrilateral β := 
sorry
end

end quadrilateral_area_l609_609304


namespace ways_to_sit_l609_609838

theorem ways_to_sit (total_chairs people : ℕ) :
  total_chairs = 6 → people = 3 → (∏ i in finset.range people, total_chairs - i) = 120 :=
by
  intros h_tc h_p
  rw [h_tc, h_p]
  norm_num
  sorry

end ways_to_sit_l609_609838


namespace Vanya_correct_answers_l609_609479

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l609_609479


namespace discount_percentage_correct_l609_609096

noncomputable def cost_per_copy : ℝ := 0.02

def individual_copies : ℕ := 80

def total_copies : ℕ := 160

def individual_savings : ℝ := 0.40

def total_cost_without_discount : ℝ :=
  (individual_copies : ℝ) * cost_per_copy * 2

def total_savings : ℝ :=
  individual_savings * 2

def total_cost_with_discount : ℝ :=
  total_cost_without_discount - total_savings

def cost_per_copy_with_discount : ℝ :=
  total_cost_with_discount / (total_copies : ℝ)

def discount_per_copy : ℝ :=
  cost_per_copy - cost_per_copy_with_discount

def percentage_discount : ℝ :=
  (discount_per_copy / cost_per_copy) * 100

theorem discount_percentage_correct :
  percentage_discount = 25 := by
  sorry

end discount_percentage_correct_l609_609096


namespace lottery_probability_exactly_one_common_l609_609395

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609395


namespace animal_jump_distances_l609_609531

-- Definitions from the conditions
def grasshopper_jump := 19
def frog_jump := grasshopper_jump + 39
def mouse_jump := frog_jump - 9
def average_jump := (grasshopper_jump + frog_jump) / 2
def squirrel_jump := average_jump + 17
def rabbit_jump := 2 * (squirrel_jump - mouse_jump)

-- Theorem to prove the distances
theorem animal_jump_distances :
  grasshopper_jump = 19 ∧
  frog_jump = 58 ∧
  mouse_jump = 49 ∧
  squirrel_jump = 55.5 ∧
  rabbit_jump = 13
:= by
  unfold grasshopper_jump frog_jump mouse_jump average_jump squirrel_jump rabbit_jump
  dsimp
  split; norm_num
  done

sorry

end animal_jump_distances_l609_609531


namespace some_mythical_are_magical_l609_609937

variable (Dragon Mythical Magical : Type)
variable (AllDragonsAreMythical : ∀ (d : Dragon), Mythical d)
variable (SomeMagicalAreDragons : ∃ (m : Magical), Dragon m)

theorem some_mythical_are_magical : ∃ (m : Mythical), Magical m := 
sorry

end some_mythical_are_magical_l609_609937


namespace vanya_correct_answers_l609_609474

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609474


namespace remainder_of_sum_first_150_numbers_l609_609145

theorem remainder_of_sum_first_150_numbers (n : ℕ) (h_n : n = 150) : 
  (n * (n + 1) / 2) % 11325 = 0 :=
by 
  have h₁ : n * (n + 1) / 2 = 11325, 
  sorry
  
  rw h₁,
  exact nat.mod_self 11325

end remainder_of_sum_first_150_numbers_l609_609145


namespace inequality_preservation_l609_609816

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l609_609816


namespace perimeter_triangle_ABF2_min_eccentricity_given_AF1_dot_AF2_eq_5c2_max_eccentricity_given_AF1_dot_AF2_eq_6c2_l609_609318

-- Definitions of ellipse parameters and conditions
variable {a b : ℝ}
variable (h_ellipse_def : a > b > 0)
noncomputable def e := (sqrt (a^2 - b^2)) / a

-- Statement: Perimeter of triangle ABF2
theorem perimeter_triangle_ABF2 : 
  forall F1 F2 A B P, 
  (sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2)) + 
  (sqrt ((B.1 - F2.1)^2 + (B.2 - F2.2)^2)) + 
  (sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)) = 4 * a :=
sorry  -- Proof goes here

-- Statement: Minimum value of eccentricity
theorem min_eccentricity_given_AF1_dot_AF2_eq_5c2 : 
  ∀ AF1 AF2 (c := sqrt (a^2 - b^2)),
  (AF1 • AF2 = 5 * c^2) → 
  (e >= sqrt(7) / 7) :=
sorry  -- Proof goes here

-- Statement: Maximum value of eccentricity
theorem max_eccentricity_given_AF1_dot_AF2_eq_6c2 : 
  ∀ AF1 AF2 (c := sqrt (a^2 - b^2)),
  (AF1 • AF2 = 6 * c^2) → 
  (e <= sqrt(7) / 7) :=
sorry  -- Proof goes here

end perimeter_triangle_ABF2_min_eccentricity_given_AF1_dot_AF2_eq_5c2_max_eccentricity_given_AF1_dot_AF2_eq_6c2_l609_609318


namespace inequality_relationship_l609_609787

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l609_609787


namespace condition1_condition2_l609_609437

noncomputable theory
open Real

structure Triangle where
  A B C : ℝ
  a b c : ℝ
  acute : A < π/2 ∧ B < π/2 ∧ C < π/2
  angle_sum : A + B + C = π 
  side_opposite_angles : a = 2 * sin A * sqrt (1 - cos^2 (π - (A + B))) ∧ b = 2 * sin B * sqrt (1 - cos^2 (π - (A + B))) ∧ c = 2 * sin C * sqrt (1 - cos^2 (π - (A + B)))

def area (t : Triangle) : ℝ := 1/2 * t.b * t.c * sin t.A

theorem condition1 (t : Triangle) (h₁ : t.A = π/3) (h₂ : t.a = 2 * sqrt 3) :
  2 * sqrt 3 < area t ∧ area t ≤ 3 * sqrt 3 := 
sorry

theorem condition2 (t : Triangle) (h₁ : t.A = π/3) (h₂ : t.b = 2) :
  sqrt 3 / 2 < area t ∧ area t < 2 * sqrt 3 := 
sorry

end condition1_condition2_l609_609437


namespace perpendicular_line_plane_condition_l609_609738

-- Definition for a line perpendicular to all lines in a plane
def isPerpendicularToAllLinesInPlane (l : Line) (a : Plane) : Prop :=
  ∀ m : Line, m ∈ LinesInPlane a → isPerpendicular l m

-- Definition for a line perpendicular to a plane
def isPerpendicularToPlane (l : Line) (a : Plane) : Prop :=
  ∀ m : Line, m ∈ LinesInPlane a → isPerpendicular l m

theorem perpendicular_line_plane_condition (l : Line) (a : Plane) :
  (isPerpendicularToAllLinesInPlane l a → isPerpendicularToPlane l a) ∧ 
  ¬(isPerpendicularToPlane l a → isPerpendicularToAllLinesInPlane l a) :=
begin
  sorry
end

end perpendicular_line_plane_condition_l609_609738


namespace translation_makes_odd_l609_609267

def f (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

def translated_f (x : ℝ) : ℝ := f (x + π / 3)

theorem translation_makes_odd :
  ∀ x : ℝ, translated_f x = -translated_f (-x) :=
by
  sorry

#print translation_makes_odd

end translation_makes_odd_l609_609267


namespace triangle_area_difference_l609_609136

theorem triangle_area_difference 
  (b h : ℝ)
  (hb : 0 < b)
  (hh : 0 < h)
  (A_base : ℝ) (A_height : ℝ)
  (hA_base: A_base = 1.20 * b)
  (hA_height: A_height = 0.80 * h)
  (A_area: ℝ) (B_area: ℝ)
  (hA_area: A_area = 0.5 * A_base * A_height)
  (hB_area: B_area = 0.5 * b * h) :
  (B_area - A_area) / B_area = 0.04 := 
by sorry

end triangle_area_difference_l609_609136


namespace probability_of_pairing_with_friends_l609_609831

theorem probability_of_pairing_with_friends (n : ℕ) (f : ℕ) (h1 : n = 32) (h2 : f = 2):
  (f / (n - 1) : ℚ) = 2 / 31 :=
by
  rw [h1, h2]
  norm_num

end probability_of_pairing_with_friends_l609_609831


namespace cube_cut_volume_l609_609600

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := ⟨0, 0, 0⟩
def X : Point3D := ⟨2, 1, 0⟩
def Y : Point3D := ⟨0, 2, 1⟩

noncomputable def volume_smaller_solid (P X Y : Point3D) : ℝ :=
  if h : (P = ⟨0, 0, 0⟩) ∧ (X = ⟨2, 1, 0⟩) ∧ (Y = ⟨0, 2, 1⟩) then
    7 / 12 
  else 
    0

theorem cube_cut_volume :
  volume_smaller_solid P X Y = 7 / 12 :=
by
  sorry

end cube_cut_volume_l609_609600


namespace no_intersecting_disk_l609_609587

open Set

-- Definitions for the problem

def polygon (P : Type) : Type := finset P
def convex_polygon (P : Type) (S : finset P) : Prop := convex ℝ (S : set P)

variables {P : Type} [metric_space P] [normed_group P] [normed_space ℝ P]

noncomputable def area (S : finset P) : ℝ := sorry -- Needs proper area definition
noncomputable def perimeter (S : finset P) : ℝ := sorry -- Needs proper perimeter definition

noncomputable def square : finset (ℝ × ℝ) := { 
  p | ∃ x y, 0 ≤ x ∧ x ≤ 38 ∧ 0 ≤ y ∧ y ≤ 38 
}

noncomputable def disk_center_in_square (c : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ x y, 1 ≤ x ∧ x ≤ 37 ∧ 1 ≤ y ∧ y ≤ 37 ∧ metric.ball (c : ℝ × ℝ) 1 ⊆ (square : set (ℝ × ℝ))

theorem no_intersecting_disk (polygons : finset (polygon (ℝ × ℝ)))
  (h_convex : ∀ S ∈ polygons, convex_polygon (ℝ × ℝ) S)
  (h_area : ∀ S ∈ polygons, area S < π)
  (h_perimeter : ∀ S ∈ polygons, perimeter S < 2 * π)
  (h_card : polygons.card = 100) :
  ∃ c : ℝ × ℝ, disk_center_in_square c 1 ∧ ∀ P ∈ polygons, metric.ball c 1 ∩ P.to_set = ∅ :=
begin
  sorry
end

end no_intersecting_disk_l609_609587


namespace find_lines_l609_609291

theorem find_lines (l : ℝ → ℝ × ℝ) (P : ℝ × ℝ) (x y : ℝ) :
  (P = (5, -2)) →
  x - y - 5 = 0 →
  (∃ θ : ℝ, θ = real.arctan 1 ∧ abs (real.arctan (y / (x - 5)) - θ) = π / 4) →
  (∃ eq, eq = l ∧ (eq = (λ y, (5, y)) ∨ eq = (λ x, (x, -2)))) :=
by
  sorry

end find_lines_l609_609291


namespace greatest_multiple_less_than_110_l609_609144

theorem greatest_multiple_less_than_110 (a b : ℕ) (h1 : a = 9) (h2 : b = 15) : 
  let lcm_value := Nat.lcm a b in 
  lcm_value = 45 ∧ ∃ k : ℕ, k * lcm_value < 110 ∧ (k + 1) * lcm_value ≥ 110 ∧ k * lcm_value = 90 :=
by 
  sorry

end greatest_multiple_less_than_110_l609_609144


namespace word_count_in_language_l609_609003

theorem word_count_in_language :
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  num_words = 900 :=
by
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  have : num_words = 900 := sorry
  exact this

end word_count_in_language_l609_609003


namespace no_closed_loop_l609_609546

-- Define the concept of a section of the children's toy train track
structure Section (radius : ℝ) :=
  (angle : ℝ)
  (shape : angle = π / 2) -- Each section is a quarter circle with angle π/2

-- Define the condition that sections are connected end to end smoothly
structure ConnectedSections (N : ℕ) (radius : ℝ) :=
  (sections : Fin N → Section radius)
  (connected_end_to_end : ∀ i, i < N - 1 → (sections i).shape = (sections (i + 1)).shape)

-- Define the proof problem
theorem no_closed_loop (N : ℕ) (radius : ℝ) (hN : ConnectedSections N radius)
  : ¬ ∃ (path : Fin N → Section radius), 
      (∀ i, i < N - 1 → (path i).shape = (path (i + 1)).shape) ∧ 
      (path 0).shape = (path (N - 1)).shape ∧ 
      (∀ i, i < N → (path i).shape = π / 2) :=
sorry

end no_closed_loop_l609_609546


namespace exp_rectangular_form_l609_609257

theorem exp_rectangular_form : exp(13 * π * I / 2) = I :=
by 
  let θ := 13 * π / 2
  have h1 : exp(θ * I) = cos(θ) + I * sin(θ), from Complex.exp_eq_cos_add_sin θ
  have h2 : cos(θ) = 0, from calc
    cos(θ) = cos(13 * π / 2) : by sorry -- periodic property simplification here
         ... = cos(π / 2) : by sorry -- simplified to fundamental period
         ... = 0 : by sorry
  have h3 : sin(θ) = 1, from calc
    sin(θ) = sin(13 * π / 2) : by sorry -- periodic property simplification here
         ... = sin(π / 2) : by sorry -- simplified to fundamental period
         ... = 1 : by sorry
  show exp(13 * π * I / 2) = 0 + I * 1, by rw [h1, h2, h3]
  show exp(13 * π * I / 2) = I, by simp

end

end exp_rectangular_form_l609_609257


namespace inequality_preservation_l609_609817

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l609_609817


namespace right_triangle_DE_length_l609_609019

variables (D E F : Type) [inhabited D] [inhabited E] [inhabited F] 
variables (angle : E → F → D → ℝ)

noncomputable def cos (θ : ℝ) : ℝ := sorry  -- cosine function

theorem right_triangle_DE_length
  (h_right : angle E F D = π / 2)
  (h_cos : cos (angle D E F) = 12 * real.sqrt 244 / 244)
  (h_DF : DF = real.sqrt 244) :
  DE = 12 :=
sorry

end right_triangle_DE_length_l609_609019


namespace problem421_l609_609433

theorem problem421 : ∃ p q : ℕ, Nat.coprime p q ∧ p / q = 15 / 406 ∧ p + q = 421 := by
  use 15
  use 406
  split
  · exact Nat.coprime_of_div_gcd 15 406 rfl
  split
  · norm_num
  · norm_num


end problem421_l609_609433


namespace number_of_ordered_pairs_l609_609929

theorem number_of_ordered_pairs (n : ℕ) (f m : ℕ) :
  n = 6 →
  0 ≤ f ∧ 0 ≤ m →
  ∃ p : Set (ℕ × ℕ), p = {pairs | pairs = (f1, m1) ∧
    (f1, m1) ∈ { (0, 6), (6, 0), (2, 6), (4, 6), (6, 6) }} ∧
  p.card = 5 :=
by
  sorry

end number_of_ordered_pairs_l609_609929


namespace sum_of_roots_eq_zero_l609_609140

theorem sum_of_roots_eq_zero (x : ℝ) (h : |x| = y) (hy : y = 2) : let roots := [2, -2] in list.sum roots = 0 :=
by
  sorry

end sum_of_roots_eq_zero_l609_609140


namespace equivalent_statements_correct_l609_609154

theorem equivalent_statements_correct:
  (∀ (f : ℝ → ℝ), Differentiable ℝ f → (∀ x_0 : ℝ, f' x_0 = 0 → CriticalPoint f x_0)) ∧
  (∀ (f : ℝ → ℝ) (a b : ℝ), MonotonicOn f (Set.Ioo a b) → ¬ ∃ x : ℝ, LocalExtrema f (Set.Ioo a b)) ∧
  (∃ (f : ℝ → ℝ), CubicFunction f ∧ ¬ ∃ x : ℝ, LocalExtrema f) :=
by
  sorry

end equivalent_statements_correct_l609_609154


namespace bank_card_payment_technology_order_l609_609982

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end bank_card_payment_technology_order_l609_609982


namespace magnitude_z_l609_609882

open Complex

theorem magnitude_z (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + r * (1 / z) = 2) :
    |z| = 3 :=
sorry

end magnitude_z_l609_609882


namespace probability_of_one_common_l609_609403

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l609_609403


namespace find_c_value_l609_609431

noncomputable def c_value : ℝ := 4 / 9

theorem find_c_value
  (O : ℝ × ℝ)
  (c : ℝ)
  (curve_eq : ℝ → ℝ)
  (line_eq : ℝ → ℝ)
  (P Q : ℝ × ℝ)
  (int_points : ∃ x : ℝ, curve_eq x = c ∧ x ≥ 0)
  (OPR_area : ∫ x in 0..min P.1 Q.1, c dx)
  (PQ_area : ∫ x in min P.1 Q.1..max P.1 Q.1, curve_eq x - c dx) :
  (OPR_area) = PQ_area →  c = c_value := by
  -- remaining proof
  sorry

end find_c_value_l609_609431


namespace line_equation_minimized_l609_609949

def point (x y : ℝ) := (x, y)
def line_through (P Q : ℝ × ℝ) (k : ℝ) := ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a * P.1 + b * P.2 + c = 0 ∧ a * Q.1 + b * Q.2 + c = 0

def minimize_distance_through_origin {P Q : ℝ × ℝ} (k : ℝ) : Prop :=
  P = (1, 4) ∧
  (Q.1 = 1 - 4 / k) ∧ (Q.2 = 0) ∧
  (Q.2 = 4 - k) ∧ (Q.1 = 0) ∧
  (∀ k : ℝ, 5 - (k + 4 / k) ≥ 9) ∧
  k = -2

theorem line_equation_minimized :
  ∃ k : ℝ, minimize_distance_through_origin k ∧
  ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b + c = 0 ∧ a + b * 4 + c = 0 :=
by
  sorry

end line_equation_minimized_l609_609949


namespace ratio_of_adults_to_children_is_24_over_25_l609_609940

theorem ratio_of_adults_to_children_is_24_over_25
  (a c : ℕ) (h₁ : a ≥ 1) (h₂ : c ≥ 1) 
  (h₃ : 30 * a + 18 * c = 2340) 
  (h₄ : c % 5 = 0) :
  a = 48 ∧ c = 50 ∧ (a / c : ℚ) = 24 / 25 :=
sorry

end ratio_of_adults_to_children_is_24_over_25_l609_609940


namespace ratio_EC_BE_eq_3_l609_609427

open_locale classical
noncomputable theory

-- Given points and segments
variables {A B C F G E : Type}

-- Conditions
variables (h₁ : ∃ (AF FC : ℝ), FC = 2 * AF ∧ F ∈ segment ℝ A C)
          (h₂ : mid_point G B F)
          (h₃ : ∃ (AG : ℝ), line AG ∧ inter AG BC = E)

-- Theorem statement
theorem ratio_EC_BE_eq_3 : 
  ∀ (A B C F G E : Type), (∃ (AF FC : ℝ), FC = 2 * AF ∧ F ∈ segment ℝ A C) →
                           (mid_point G B F) →
                           (∃ (AG : ℝ), line AG ∧ inter AG BC = E) →
  (distance E C) / (distance B E) = 3 :=
begin
  intros,
  sorry
end

end ratio_EC_BE_eq_3_l609_609427


namespace find_a_l609_609369

theorem find_a : 
  (∃ a : ℝ, (binom 5 2) * a^3 * (-1)^2 = 80) → a = 2 :=
by
  intro h,
  cases h with a ha,
  have h1 : (binom 5 2) = 10 := by sorry,  -- This should be replaced with an appropriate library theorem about binomial coefficients.
  have h2 : (-1)^2 = 1 := by norm_num,
  rw [h1, h2] at ha,
  rw ← mul_assoc at ha,
  have h3 : 10 * a^3 = 80 := ha,
  norm_num at h3,
  rw ← eq_div_iff at h3,
  norm_num at h3,
  have h4 : a^3 = 8 := h3,
  have h5 : a = real.cbrt 8 := by sorry,
  have h6 : real.cbrt 8 = 2 := by norm_num,
  rw h6 at h5,
  exact h5


end find_a_l609_609369


namespace log_product_value_l609_609248

noncomputable def log_prod_expression := 
  ∑ (a : ℕ) in finset.range (20183), ∑ (b : ℕ) in finset.range (2019), 
    real.logb 2 (1 + complex.exp ((2 * a * b * real.pi * complex.I) / 2019))

theorem log_product_value : log_prod_expression = 6725 := sorry

end log_product_value_l609_609248


namespace tangent_line_to_curve_l609_609523

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  y = (x^2 - 2 * x) * Real.log (2 * x)

theorem tangent_line_to_curve :
  tangent_line_equation (x y : ℝ) (1, -Real.log 2) → x + y + Real.log 2 - 1 = 0 :=
sorry

end tangent_line_to_curve_l609_609523


namespace expected_value_eight_sided_die_l609_609234

theorem expected_value_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8] in 
  (∑ i in outcomes, i) / (outcomes.length : ℝ) = 4.5 :=
by
  sorry

end expected_value_eight_sided_die_l609_609234


namespace expand_expression_l609_609278

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l609_609278


namespace As_share_in_profit_l609_609162

def x (total_gain : ℝ) (invA_rate : ℝ) (invB_rate : ℝ) (invC_rate : ℝ) (invB_time_frac : ℝ) (invC_time_frac : ℝ) : ℝ :=
    total_gain / (invA_rate + invB_rate * invB_time_frac + invC_rate * invC_time_frac)

def shareA (investA : ℝ) (invA_rate : ℝ) : ℝ := investA * invA_rate

theorem As_share_in_profit (total_gain : ℝ) (approx_shareA : ℝ) (invA_rate invB_rate invC_rate : ℝ) (invB_time_frac invC_time_frac : ℝ) :
    total_gain = 60000 ∧ invA_rate = 0.15 ∧ invB_rate = 0.20 ∧ invC_rate = 0.18 ∧ invB_time_frac = 0.5 ∧ invC_time_frac = 1/3 ∧ approx_shareA ≈ 14469.60 :=
    let investA := x total_gain invA_rate invB_rate invC_rate invB_time_frac invC_time_frac in
    shareA investA invA_rate ≈ approx_shareA :=
    by
        sorry

#eval As_share_in_profit 60000 14469.60 0.15 0.20 0.18 0.5 (1/3)

end As_share_in_profit_l609_609162


namespace triangle_angles_l609_609835

theorem triangle_angles (second_angle first_angle third_angle : ℝ) 
  (h1 : first_angle = 2 * second_angle)
  (h2 : third_angle = second_angle + 30)
  (h3 : second_angle + first_angle + third_angle = 180) :
  second_angle = 37.5 ∧ first_angle = 75 ∧ third_angle = 67.5 :=
sorry

end triangle_angles_l609_609835


namespace area_of_intersections_is_correct_l609_609004
open Real

-- Definition of the circle radius
def radius : ℝ := 5

-- Definition of the area of right-angled triangles formed by the perpendicular diameters
def triangle_area : ℝ := 2 * (1/2 * radius * radius)

-- Definition of the area of the sectors formed by the perpendicular diameters
def sector_area : ℝ := 2 * (π * (radius^2) / 4)

-- The total shaded area
def total_area : ℝ := triangle_area + sector_area

-- The theorem to prove the given area calculation
theorem area_of_intersections_is_correct : total_area = 25 + 12.5 * π := by
  sorry

end area_of_intersections_is_correct_l609_609004


namespace find_y_l609_609875

noncomputable def diamond (a b : ℕ) : ℝ := (real.sqrt (3 * a + 2 * b))^3

theorem find_y : ∃ y : ℤ, diamond 6 y = 64 ∧ y = -1 :=
by
  use -1
  split
  case left { sorry }
  case right { rfl }

end find_y_l609_609875


namespace smallest_x_for_M_cube_l609_609116

theorem smallest_x_for_M_cube (x M : ℤ) (h1 : 1890 * x = M^3) : x = 4900 :=
sorry

end smallest_x_for_M_cube_l609_609116


namespace sum_b_1000_l609_609438

noncomputable def b : ℕ → ℕ
| 0 := 2
| 1 := 2
| 2 := 2
| (n+3) := if (9 * b n ^ 2 / 4 - b (n+1) * b (n+2) > 0) then 4 else if (9 * b n ^ 2 / 4 - b (n+1) * b (n+2) = 0) then 2 else 0

theorem sum_b_1000 : (Finset.range 1000).sum (λ n => b n) = 3998 := sorry

end sum_b_1000_l609_609438


namespace other_root_l609_609906

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l609_609906


namespace hyperbola_eccentricity_l609_609841

-- Define the parameters for the hyperbola
def a_squared : ℝ := 4
def b_squared : ℝ := 3
def c : ℝ := Real.sqrt (a_squared + b_squared)
def e : ℝ := c / (Real.sqrt a_squared)

-- Statement to prove the eccentricity e is sqrt(7)/2
theorem hyperbola_eccentricity :
  e = Real.sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l609_609841


namespace area_lt_perimeter_probability_l609_609202

theorem area_lt_perimeter_probability :
  (∃ s : ℕ, s ≥ 2 ∧ s ≤ 12 ∧ s * (s - 4) < 0) ∧ (1/36 + 1/18 = 1/12) :=
begin
  -- defines the probability space of rolling a pair of 6-sided dice
  let diceProb := pmf.of_finset {(i, j) | i ∈ finset.range 1 6 ∧ j ∈ finset.range 1 6} sorry,
  -- defines the event that the side length of the square (sum of dice) is such that the area < perimeter
  let event := {s | s ∈ finset.range 2 4},
  -- calculates the probability of the event
  have h : ∑ s in event, diceProb (λ (p : ℕ × ℕ), p.1 + p.2 = s) = 1 / 12, {
    sorry
  },
  -- proves the final equality of probabilities
  exact h,
end

end area_lt_perimeter_probability_l609_609202


namespace B_can_do_work_alone_in_10_days_l609_609367

theorem B_can_do_work_alone_in_10_days (W : ℝ) (A_rate : ℝ) (combined_rate : ℝ) : 
  (A_rate = W / 10) ∧ (combined_rate = W / 5) → (W / (combined_rate - A_rate) = 10) :=
by
  intros h
  cases h with hA hAB
  sorry

end B_can_do_work_alone_in_10_days_l609_609367


namespace min_distance_curve_l609_609536

theorem min_distance_curve
  (t : ℝ)
  (A : ℝ × ℝ)
  (y : ℝ → ℝ)
  (min_dist : ℝ)
  (m : ℝ) :
  A = (t, 0) ∧ 
  y = λ x, Real.exp x ∧ 
  min_dist = 2 * Real.sqrt 3 ∧ 
  (A,fnd m),
    (m,A) := Real.sqrt ((m - t)^2 + (Real.exp m)^2) = 2 * Real.sqrt 3 ∧ 
  t = 3 + Real.log 3 / 2 := 
sorry

end min_distance_curve_l609_609536


namespace count_total_wheels_l609_609134

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end count_total_wheels_l609_609134


namespace vanya_correct_answers_l609_609491

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609491


namespace minimum_distance_origin_to_line_l609_609714

theorem minimum_distance_origin_to_line (x y : ℝ) (h : 3 * x + 4 * y = 24) : 
  ∃ z, z = sqrt (x^2 + y^2) ∧ z = 24 / 5 :=
by
  sorry

end minimum_distance_origin_to_line_l609_609714


namespace correct_action_order_l609_609984

inductive Action
| tap : Action
| pay_online : Action
| swipe : Action
| insert_into_terminal : Action
deriving DecidableEq, Repr, Inhabited

inductive Technology
| chip : Technology
| magnetic_stripe : Technology
| paypass : Technology
| cvc : Technology
deriving DecidableEq, Repr, Inhabited

def action_for_technology : Technology → Action
| Technology.chip := Action.insert_into_terminal
| Technology.magnetic_stripe := Action.swipe
| Technology.paypass := Action.tap
| Technology.cvc := Action.pay_online

theorem correct_action_order :
  [action_for_technology Technology.chip, action_for_technology Technology.magnetic_stripe,
   action_for_technology Technology.paypass, action_for_technology Technology.cvc] = 
  [Action.insert_into_terminal, Action.swipe, Action.tap, Action.pay_online] := 
sorry

end correct_action_order_l609_609984


namespace part1_part2_part3_l609_609325

noncomputable theory

variables {a : ℕ → ℕ} {S : ℕ → ℕ}
variables (λ μ : ℕ)

-- Conditions
def condition1 (h₁ : λ > 2 ∧ Prime λ) (h₂ : ∀ n, 2 * S n = λ * a n - μ) : Prop :=
  ∃ r, ∀ n ≥ 2, a n = r * a (n - 1)

def condition2 (h₁ : λ = 3) (h₂ : ∃ x y, x ≠ y ∧ 2010 = μ * (3^(x - 1) + 3^(y - 1))): Prop :=
  μ = 67 ∨ μ = 201

def condition3 (n : ℕ)(h₁ : ∀ x ∈ {a i + a j | i j : ℕ, i ≠ j}, 5 * μ * 3^(n-1) < x ∧ x < 5 * μ * 3^(n)): Prop :=
  ∃ b : ℕ, b = n + 1

-- Lean Statements
theorem part1 (h₁ : λ > 2 ∧ Prime λ) (h₂ : ∀ n, 2 * S n = λ * a n - μ) :
  condition1 h₁ h₂ :=
sorry

theorem part2 (h₁ : λ = 3) (h₂ : ∃ x y, x ≠ y ∧ 2010 = μ * (3^(x - 1) + 3^(y - 1))) :
  condition2 h₁ h₂ :=
sorry

theorem part3 (n : ℕ) (h₁ : ∀ x ∈ {a i + a j | i j : ℕ, i ≠ j}, 5 * μ * 3^(n-1) < x ∧ x < 5 * μ * 3^(n)) :
  condition3 n h₁ :=
sorry

end part1_part2_part3_l609_609325


namespace athlete_arrangement_l609_609175

def consecutive_tracks_count : ℕ := 6

noncomputable def permutations (n : ℕ) : ℕ :=
  nat.factorial n

theorem athlete_arrangement :
  let three_athlete_permutations := permutations 3;
  let five_athlete_permutations := permutations 5;
  consecutive_tracks_count * three_athlete_permutations * five_athlete_permutations = 4320 :=
by
  sorry

end athlete_arrangement_l609_609175


namespace two_digit_numbers_with_sum_121_l609_609669

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n <= 99

theorem two_digit_numbers_with_sum_121 :
  (∃ (a b : ℕ), is_digit a ∧ is_digit b ∧ is_two_digit_number (10 * a + b) ∧ (10 * a + b + 10 * b + a = 121)) =
  8 :=
by
  sorry

end two_digit_numbers_with_sum_121_l609_609669


namespace first_six_divisors_l609_609085

theorem first_six_divisors (a b : ℤ) (h : 5 * b = 14 - 3 * a) : 
  ∃ n, n = 5 ∧ ∀ k ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), (3 * b + 18) % k = 0 ↔ k ∈ ({1, 2, 3, 5, 6} : Finset ℕ) :=
by
  sorry

end first_six_divisors_l609_609085


namespace least_sum_of_exponents_2035_l609_609819

theorem least_sum_of_exponents_2035 :
  ∃ (S : Finset ℕ), (2035 = S.sum (λ n, 2^n)) ∧ (S.sum id = 50) :=
sorry

end least_sum_of_exponents_2035_l609_609819


namespace one_div_lt_one_div_of_gt_l609_609520

theorem one_div_lt_one_div_of_gt {a b : ℝ} (hab : a > b) (hb0 : b > 0) : (1 / a) < (1 / b) :=
sorry

end one_div_lt_one_div_of_gt_l609_609520


namespace ap_of_squares_l609_609460

theorem ap_of_squares 
  (a b c : ℝ)
  (h : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d) :
  (a^2 + a * b + b^2, a^2 + a * c + c^2, b^2 + b * c + c^2 are in arithmetic progression) :=
begin
  sorry
end

end ap_of_squares_l609_609460


namespace factorial_identity_l609_609147

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem factorial_identity : (factorial 12 - factorial 11) / factorial 10 = 121 := by
  sorry

end factorial_identity_l609_609147


namespace combined_weight_of_Alexa_and_Katerina_l609_609225

variable (total_weight: ℝ) (alexas_weight: ℝ) (michaels_weight: ℝ)

theorem combined_weight_of_Alexa_and_Katerina
  (h1: total_weight = 154)
  (h2: alexas_weight = 46)
  (h3: michaels_weight = 62) :
  total_weight - michaels_weight = 92 :=
by 
  sorry

end combined_weight_of_Alexa_and_Katerina_l609_609225


namespace probability_of_rain_on_both_days_l609_609006

variable {Ω : Type} {P : Measure Ω} 
variable {M T N : {x : Ω // True}}

-- Defining the probabilities based on the given conditions
variable (pM pT pN pMcapT : ℝ)
variable (hpM : P M = 0.62)
variable (hpT : P T = 0.54)
variable (hpN : P N = 0.28)

-- Statement for the problem
theorem probability_of_rain_on_both_days :
  pMcapT = (pM + pT - (1 - pN)) :=
sorry

end probability_of_rain_on_both_days_l609_609006


namespace sequence_count_415800_l609_609661

open Equiv.Perm

/-- Define the transformations A, B, C, D as permutations -/
def A : equiv.perm (fin 4) := @equiv.perm.transposition _ _ 0 1
def B : equiv.perm (fin 4) := (A)⁻¹
def C : equiv.perm (fin 4) := @equiv.perm.transposition _ _ 0 2
def D : equiv.perm (fin 4) := @equiv.perm.transposition _ _ 1 3

/-- Define the dihedral group D4 -/
inductive D4 : Type
| id | A | A2 | A3 | A4 | C | D | AC | BD

/-- Define the identity element -/
def id : D4 := D4.id

/-- Define a proof that there are 415800 sequences of 12 transformations
    using {A, B, C, D} that restore square WXYZ to its original positions -/
theorem sequence_count_415800 :
  ∃ (seq : list (D4)), seq.length = 12 ∧
  (list.foldl (•) id seq = id) ∧ (list.permutations seq).length = 415800 :=
sorry

end sequence_count_415800_l609_609661


namespace correct_option_is_D_l609_609153

-- Conditions defined as options (A, B, C, D)
def optionA : Prop := (-2)^4 = -8
def optionB : Prop := (-2)^4 = 4 ^ (-2)
def optionC : Prop := (-2)^4 = -2 * 2 * 2 * 2
def optionD : Prop := (-2)^4 = (-2)

-- The goal is to prove that the correct option is D
theorem correct_option_is_D : 
  (-2)^4 = 16 ∧ ¬ optionA ∧ ¬ optionB ∧ ¬ optionC ∧ optionD :=
by sorry

end correct_option_is_D_l609_609153


namespace find_a0_to_a5_sum_find_absolute_a0_to_a5_sum_find_a1_a3_a5_sum_find_squared_difference_l609_609323

noncomputable def polynomial := (2 : ℝ) * X - 1

theorem find_a0_to_a5_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial ^ 5 = a0 + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5 →
  a0 + a1 + a2 + a3 + a4 + a5 = 1 :=
begin
  intro h,
  sorry
end

theorem find_absolute_a0_to_a5_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial ^ 5 = a0 + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5 →
  |a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243 :=
begin
  intro h,
  sorry
end

theorem find_a1_a3_a5_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial ^ 5 = a0 + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5 →
  a1 + a3 + a5 = 122 :=
begin
  intro h,
  sorry
end

theorem find_squared_difference (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial ^ 5 = a0 + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5 →
  (a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243 :=
begin
  intro h,
  sorry
end

end find_a0_to_a5_sum_find_absolute_a0_to_a5_sum_find_a1_a3_a5_sum_find_squared_difference_l609_609323


namespace remaining_standby_time_l609_609645

variable (fully_charged_standby : ℝ) (fully_charged_gaming : ℝ)
variable (standby_time : ℝ) (gaming_time : ℝ)

theorem remaining_standby_time
  (h1 : fully_charged_standby = 10)
  (h2 : fully_charged_gaming = 2)
  (h3 : standby_time = 4)
  (h4 : gaming_time = 1.5) :
  (10 - ((standby_time * (1 / fully_charged_standby)) + (gaming_time * (1 / fully_charged_gaming)))) * 10 = 1 :=
by
  sorry

end remaining_standby_time_l609_609645


namespace prism_volume_l609_609554

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end prism_volume_l609_609554


namespace inscribed_sphere_radius_of_tetrahedron_l609_609743

variables (V S1 S2 S3 S4 R : ℝ)

theorem inscribed_sphere_radius_of_tetrahedron
  (hV_pos : 0 < V)
  (hS_pos : 0 < S1) (hS2_pos : 0 < S2) (hS3_pos : 0 < S3) (hS4_pos : 0 < S4) :
  R = 3 * V / (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_radius_of_tetrahedron_l609_609743


namespace porter_daily_rate_l609_609070

theorem porter_daily_rate:
  ∀ (D : ℝ),
  (let weekly_earning := 5 * D in
  let weekly_overtime := 1.5 * D in
  let monthly_earning := 4 * (weekly_earning + weekly_overtime) in
  monthly_earning = 208) → D = 8 :=
begin
  sorry
end

end porter_daily_rate_l609_609070


namespace solve_x_l609_609113

theorem solve_x (x : ℝ) : (81 ^ (x - 1) / 9 ^ (x + 1) = 729 ^ (x + 2)) → x = -9 / 2 :=
by
  sorry

end solve_x_l609_609113


namespace chores_equality_l609_609646

theorem chores_equality :
  let mins_sweeping_per_room := 3
      mins_dishes_per_dish := 2
      mins_laundry_per_load := 9
      rooms_anna := 10
      loads_billy := 2
      time_anna := rooms_anna * mins_sweeping_per_room
      time_billy := loads_billy * mins_laundry_per_load
      dishes_billy := (time_anna - time_billy) / mins_dishes_per_dish
  in dishes_billy = 6 :=
by
  sorry

end chores_equality_l609_609646


namespace lottery_probability_exactly_one_common_l609_609399

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609399


namespace right_triangle_segment_ratio_l609_609740

open EuclideanGeometry

noncomputable def hypotenuse_ratio : ℝ :=
  let x : ℝ := arbitrary ℝ in
  let BC : ℝ := x in
  let AB : ℝ := 3 * x in
  let AC : ℝ := x * sqrt 10 in
  let point := arbitrary in
  let D := point in
  -- Define the points such that B is the right angle
  let A : Point := arbitrary in
  let B : Point := arbitrary in
  let C : Point := arbitrary in
  let ⟨A, B, C, D⟩ := is_right_triangle ABC in
  -- The altitude intersects the hypotenuse at point D
  let altitude_BD := perpendicular_from B AC D in
  -- Similar triangles relationship
  let similar1 := similarity_of_triangles ADB BDC ABC in
  -- Prove the ratio
  let ratio := (CD / AD = 9) in
  ratio

theorem right_triangle_segment_ratio (x : ℝ) (h₁ : x > 0) :
  let BC : ℝ := x in
  let AB : ℝ := 3 * x in
  let AC : ℝ := x * sqrt 10 in
  let CD : ℝ := arbitrary ℝ in
  AD : ℝ := arbitrary ℝ in
  x > 0 → 
  ∃ (BD CD AD : ℝ), CD / AD = 9 :=
begin
  sorry
end

end right_triangle_segment_ratio_l609_609740


namespace polynomial_remainders_eq_m_zero_l609_609110

theorem polynomial_remainders_eq_m_zero (m : ℝ) :
  let p1 := λ y : ℝ, 29 * 42 * y^2 + m * y + 2,
      p2 := λ y : ℝ, y^2 + m * y + 2,
      R1 := p1 1,
      R2 := p2 (-1)
  in R1 = R2 → m = 0 := 
by {
  intro h,
  let R1 := p1 1,
  let R2 := p2 (-1),
  have h1 : R1 = 3 + m,
  {
    simp [p1],
    linarith,
  },
  have h2 : R2 = 3 - m,
  {
    simp [p2],
    linarith,
  },
  rw [h1, h2] at h,
  linarith,
}

end polynomial_remainders_eq_m_zero_l609_609110


namespace max_binomial_coefficient_expansion_l609_609968

theorem max_binomial_coefficient_expansion (m x : ℕ) (h1 : (5 / x^(1/2) - x)^m = 256) : 
  nat.choose 4 2 = 6 :=
by 
  have m_val : m = 4,
  { 
    sorry -- Deduce m from the given condition
  },
  have binom_eq : nat.choose 4 2 = 6,
  { 
    exact nat.choose_succ_succ 2 2,
  },
  exact binom_eq

end max_binomial_coefficient_expansion_l609_609968


namespace mass_fraction_K2SO4_l609_609270

theorem mass_fraction_K2SO4 :
  (2.61 * 100 / 160) = 1.63 :=
by
  -- Proof details are not required as per instructions
  sorry

end mass_fraction_K2SO4_l609_609270


namespace number_b_is_three_times_number_a_l609_609108

theorem number_b_is_three_times_number_a (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 :=
by
  -- This is where the proof would go
  sorry

end number_b_is_three_times_number_a_l609_609108


namespace inequality_l609_609766

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l609_609766


namespace factorize_xy_squared_minus_x_l609_609705

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609705


namespace product_of_equal_numbers_l609_609517

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l609_609517


namespace std_eq_circle_C_l609_609328

theorem std_eq_circle_C (m p : ℝ) (h₀ : m > 0) (h₁ : p > 0) (h₂ : (4, m) ∈ setOf (λ y, y^2 = 2 * p * 4)) (h₃ : ∃ A : ℝ × ℝ, A = (4, m) ∧ ∃ F : ℝ × ℝ, (|dist A F| = 5) ∧ (|dist A ⟨4, y⟩| = 5) ∧ (abs (distance (0, 0) ⟨0, y⟩) = 3)) :
  ∃ (C : ℝ × ℝ → ℝ), C = λ ⟨x, y⟩, (x - 4) ^ 2 + (y - 4) ^ 2 = 25 :=
by
  sorry

end std_eq_circle_C_l609_609328


namespace vanya_correct_answers_l609_609489

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609489


namespace tracy_total_books_collected_l609_609560

variable (weekly_books_first_week : ℕ)
variable (multiplier : ℕ)
variable (weeks_next_period : ℕ)

-- Conditions
def first_week_books := 9
def second_period_books_per_week := first_week_books * 10
def books_next_five_weeks := second_period_books_per_week * 5

-- Theorem
theorem tracy_total_books_collected : 
  (first_week_books + books_next_five_weeks) = 459 := 
by 
  sorry

end tracy_total_books_collected_l609_609560


namespace seq_problem_l609_609057

theorem seq_problem (a : ℕ → ℚ) (d : ℚ) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d )
 (h1 : a 1 = 2)
 (h_geom : (a 1 - 1) * (a 5 + 5) = (a 3)^2) :
  a 2017 = 1010 := 
sorry

end seq_problem_l609_609057


namespace line_parallelism_theorem_l609_609805

-- Definitions of the relevant geometric conditions
variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions as hypotheses
axiom line_parallel_plane (m : Line) (α : Plane) : Prop
axiom line_in_plane (n : Line) (α : Plane) : Prop
axiom plane_intersection_line (α β : Plane) : Line
axiom line_parallel (m n : Line) : Prop

-- The problem statement in Lean 4
theorem line_parallelism_theorem 
  (h1 : line_parallel_plane m α) 
  (h2 : line_in_plane n β) 
  (h3 : plane_intersection_line α β = n) 
  (h4 : line_parallel_plane m β) : line_parallel m n :=
sorry

end line_parallelism_theorem_l609_609805


namespace bottles_per_case_l609_609186

theorem bottles_per_case (total_bottles_per_day : ℕ) (cases_required : ℕ) (bottles_per_case : ℕ)
  (h1 : total_bottles_per_day = 65000)
  (h2 : cases_required = 5000) :
  bottles_per_case = total_bottles_per_day / cases_required :=
by
  sorry

end bottles_per_case_l609_609186


namespace find_intersection_point_l609_609430
noncomputable theory

-- Definitions for functions f and g
def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 11*x - 6) / (2*x - 4)
def g (x : ℝ) (d : ℝ) : ℝ := (2*x^3 - 6*x^2 + 5*x + d) / (x - 2)

-- Main theorem stating the conditions
theorem find_intersection_point (d : ℝ) :
  (∀ x, 2*x = 4 → ∃ l, is_limit f x l) ∧  -- Same vertical asymptote.
  (∀ x, f(x) = g(x) → Exists (λ p, p ≠ (-1 : ℝ) ∧ p = 3)) →  -- Intersection points
  g (-1) (-6) = -1 ∧ -- d determined by the intersection at x = -1 
  f 3 = 0 ∧
  g 3 (-6) = 0 := 
sorry -- Proof is omitted.

end find_intersection_point_l609_609430


namespace probability_exactly_one_common_number_l609_609385

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l609_609385


namespace greatest_four_digit_multiple_of_17_l609_609999

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 17 = 0 ∧ ∀ m, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 17 = 0) → m ≤ n :=
  ⟨9996, by {
        split,
        { linarith },
        { split,
            { linarith },
            { split,
                { exact ModEq.rfl },
                { intros m hm hle,
                  have h : m ≤ 9999 := hm.2.2,
                  have : m = 17 * (m / 17) := (Nat.div_mul_cancel hm.2.1).symm,
                  have : 17 * (m / 17) ≤ 17 * 588 := Nat.mul_le_mul_left 17 (Nat.div_le_of_le_mul (by linarith)),
                  linarith,
                },
            },
        },
    },
  ⟩ sorry

end greatest_four_digit_multiple_of_17_l609_609999


namespace negation_of_implication_l609_609336

theorem negation_of_implication {r p q : Prop} :
  ¬ (r → (p ∨ q)) ↔ (¬ r → (¬ p ∧ ¬ q)) :=
by sorry

end negation_of_implication_l609_609336


namespace intersection_of_sets_l609_609356

noncomputable def setA : Set ℝ := {x | 1 / (x - 1) ≤ 1}
def setB : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_sets : setA ∩ setB = {-1, 0, 2} := 
by
  sorry

end intersection_of_sets_l609_609356


namespace increase_probability_l609_609654

variable (S : Set ℕ) 
          (numbers : ℕ)
          (P : ℕ → ℕ → Prop)
          (remove : ℕ) 

noncomputable def meets_condition (x y : ℕ) : Prop := x + y = 15

theorem increase_probability
  (removal_in_S : remove ∈ S) 
  (initial_set : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (pairs_sum_15 : ∃ a b ∈ S, a ≠ b ∧ meets_condition a b)
  (new_n : remove = 10) :
  let S_new := S \ {remove} in
  let initial_prob := 4 / (Finset.card (Finset.powersetLen 2 (Finset.filter (λ x, x ∈ S) (Finset.range 11)))) in 
  let new_prob := 4 / (Finset.card (Finset.powersetLen 2 (Finset.filter (λ x, x ∈ S_new) (Finset.range 11)))) in
  new_prob > initial_prob  :=
  sorry

end increase_probability_l609_609654


namespace camel_cost_l609_609174

def cost (n : ℕ) (price : ℕ) : Prop := n * price

theorem camel_cost (C H O E : ℕ) 
  (h1 : cost 10 C = cost 24 H) 
  (h2 : cost 16 H = cost 4 O) 
  (h3 : cost 6 O = cost 4 E) 
  (h4 : cost 10 E = 110000) : 
  C = 4400 := by
  sorry

end camel_cost_l609_609174


namespace marks_in_english_l609_609265

theorem marks_in_english :
  let m := 35             -- Marks in Mathematics
  let p := 52             -- Marks in Physics
  let c := 47             -- Marks in Chemistry
  let b := 55             -- Marks in Biology
  let n := 5              -- Number of subjects
  let avg := 46.8         -- Average marks
  let total_marks := avg * n
  total_marks - (m + p + c + b) = 45 := sorry

end marks_in_english_l609_609265


namespace factorize_xy2_minus_x_l609_609681

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609681


namespace batsman_sixes_l609_609177

theorem batsman_sixes (total_runs : ℕ) (boundaries : ℕ) (running_percentage : ℝ) (score_per_boundary : ℕ) (score_per_six : ℕ)
  (h1 : total_runs = 150)
  (h2 : boundaries = 5)
  (h3 : running_percentage = 66.67)
  (h4 : score_per_boundary = 4)
  (h5 : score_per_six = 6) :
  ∃ (sixes : ℕ), sixes = 5 :=
by
  -- Calculations omitted
  existsi 5
  sorry

end batsman_sixes_l609_609177


namespace prism_volume_l609_609557

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end prism_volume_l609_609557


namespace juniors_in_program_l609_609834

variable (J S : ℕ)
variable (j s : ℕ)
variable (h1 : J + S = 40)
variable (h2 : 0.3 * J = 0.3 * S)
variable (h3 : j = 0.3 * J)
variable (h4 : s = 0.2 * S)

theorem juniors_in_program : J = 20 :=
by
  sorry

end juniors_in_program_l609_609834


namespace common_difference_l609_609542

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1) * d)

-- State the theorem to be proven
theorem common_difference (a : ℝ) (d : ℝ) :
  (sum_arithmetic_sequence a d 12 / 12 - sum_arithmetic_sequence a d 10 / 10 = -2) → d = -2 :=
by
  sorry

end common_difference_l609_609542


namespace place_point_C_on_perpendicular_l609_609413

-- Define the points A and B in the plane
variable (A B : Point)
-- Define the condition for the ratio of distances
variable (CA CB : ℝ) (h : CA / CB = 2.6)
-- Define the perpendicular condition and distance relation
variable (AB : ℝ)

-- We need to prove that point C should be on the perpendicular from B to AB at a specific distance
theorem place_point_C_on_perpendicular 
  (C : Point) 
  (h1 : perpendicular_between B AB C) 
  (h2 : distance_between C B = (5 / 12) * AB)
  : True := 
sorry

end place_point_C_on_perpendicular_l609_609413


namespace marcel_potatoes_eq_l609_609062

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end marcel_potatoes_eq_l609_609062


namespace find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l609_609605

open Real

-- Given conditions
def line_passes_through (x1 y1 x2 y2 : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l x1 y1 ∧ l x2 y2

def circle_tangent_to_x_axis (center_x center_y : ℝ) (r : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  C center_x center_y ∧ center_y = r

-- We want to prove:
-- 1. The equation of line l is x - 2y = 0
theorem find_line_equation_through_two_points:
  ∃ l : ℝ → ℝ → Prop, line_passes_through 2 1 6 3 l ∧ (∀ x y, l x y ↔ x - 2 * y = 0) :=
  sorry

-- 2. The equation of circle C is (x - 2)^2 + (y - 1)^2 = 1
theorem find_circle_equation_tangent_to_x_axis:
  ∃ C : ℝ → ℝ → Prop, circle_tangent_to_x_axis 2 1 1 C ∧ (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :=
  sorry

end find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l609_609605


namespace extreme_values_f_range_of_a_inequality_ln_n_l609_609351

noncomputable def f (x : ℝ) := -x^3 + x^2
noncomputable def g (x : ℝ) (a : ℝ) := a * log x

theorem extreme_values_f :
  (∀ x, f 0 = 0 ∧ f (2/3) = 4/27) ∧ 
  (∀ x, (0 < x ∧ x < 2/3 → deriv f x > 0) ∧ (x < 0 ∨ x > 2/3 → deriv f x < 0)) := 
sorry

theorem range_of_a (a : ℝ) :
  a ≠ 0 → (∀ x ∈ Icc 1 ∞, f x + g x a ≥ -x^3 + (a + 2) * x) → a ≤ -1 :=
sorry

theorem inequality_ln_n (n : ℕ) :
  n > 0 → (∑ k in finset.range (2016), (1 / log (n + k.succ)) > 2015 / (n * (n + 2015))) :=
sorry

end extreme_values_f_range_of_a_inequality_ln_n_l609_609351


namespace intersection_point_l609_609338

theorem intersection_point :
  (∃ x y : ℝ, (2 * x + 1 = y ∧ -x + 4 = y) ∧ x = 1) ↔ (1, 3) :=
by
  sorry

end intersection_point_l609_609338


namespace lottery_probability_exactly_one_common_l609_609394

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609394


namespace vanya_correct_answers_l609_609468

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l609_609468


namespace minimum_drops_needed_l609_609939

-- Define the problem setup
def floors := 163
def phones := 2

-- Define the problem conditions
-- A phone breaks if dropped from a certain floor or higher
def phone_breaks_at_floor (floor : ℕ) (phone_broken : ℕ) : Prop :=
  floor ≥ phone_broken

-- Minimum number of drops required to determine the floor with given conditions
def minimum_drops (floors phones : ℕ) : ℕ :=
  if floors = 163 ∧ phones = 2 then 18 else sorry

-- The main theorem to prove: we require a specific minimum number of drops
theorem minimum_drops_needed : minimum_drops floors phones = 18 :=
by
  proper sorry

end minimum_drops_needed_l609_609939


namespace range_of_g_l609_609101

theorem range_of_g (x : ℝ) : set.range (λ x, ⌈x⌉ - x) = set.Ico 0 1 :=
by sorry

end range_of_g_l609_609101


namespace vanya_correct_answers_l609_609492

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609492


namespace part_I_part_II_l609_609423

-- Define the conditions
def seq_pos (a : ℕ → ℝ) := ∀ n, a n > 0

def sum_seq (S : ℕ → ℝ) := ∀ n, 
  S n ^ 2 - (n^2 + 2 * n - 1) * S n - (n^2 + 2 * n) = 0

-- Given S_n = n^2 + 2n
def Sn (n : ℕ) : ℝ := n^2 + 2 * n

-- Part I: Prove general term a_n of sequence
def general_term (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n ≥ 1, a n = S n - S (n - 1)

-- Part II: Define b_n and sum formula
def bn (a : ℕ → ℝ) (n : ℕ) : ℝ := (a n - 5) / 2^n

def sum_bn (b : ℕ → ℝ) (T : ℕ → ℝ) := 
  ∀ n, T n = ∑ i in range n, b (2 * (i + 1))

-- Final proof statements
theorem part_I (a : ℕ → ℝ) (S : ℕ → ℝ) (h_seq_pos : seq_pos a) (h_sum_seq : sum_seq S):
  general_term a S → 
  ∀ n ≥ 1, a n = 2 * n + 1 :=
by
  intro general_term_def
  have Sn_def : ∀ n ≥ 1, S n = Sn n := sorry
  apply sorry

theorem part_II (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (h_seq_pos : seq_pos a) (h_sum_seq : sum_seq Sn):
  (∀ n, b n = bn a n) →
  sum_bn b T → 
  ∀ n, T n = (1 / 9) * (4 - (3 * n + 1) / 4^(n - 1)) :=
by
  intro bn_def sum_bn_def
  apply sorry

end part_I_part_II_l609_609423


namespace Ramya_reads_total_124_pages_l609_609450

theorem Ramya_reads_total_124_pages :
  let total_pages : ℕ := 300
  let pages_read_monday := (1/5 : ℚ) * total_pages
  let pages_remaining := total_pages - pages_read_monday
  let pages_read_tuesday := (4/15 : ℚ) * pages_remaining
  pages_read_monday + pages_read_tuesday = 124 := 
by
  sorry

end Ramya_reads_total_124_pages_l609_609450


namespace rods_no_self_intersection_probability_l609_609236

-- Definitions for angles and conditions
def valid_alpha (alpha : ℝ) : Prop := 0 ≤ alpha ∧ alpha ≤ Real.pi
def valid_beta (beta : ℝ) : Prop := 0 ≤ beta ∧ beta ≤ 2 * Real.pi

def intersects (alpha beta : ℝ) : Prop := 
  (0 ≤ beta ∧ beta < Real.pi / 2 - alpha / 2) ∧ 
  (0 ≤ alpha ∧ alpha < Real.pi / 2 - beta / 2)

def no_self_intersection_probability : ℝ := 11 / 12

-- Main theorem statement
theorem rods_no_self_intersection_probability : 
  (∀ alpha beta : ℝ, valid_alpha alpha → valid_beta beta → ¬intersects alpha beta) ↔ 
  no_self_intersection_probability = 11 / 12 :=
sorry

end rods_no_self_intersection_probability_l609_609236


namespace gcd_153_119_l609_609532

theorem gcd_153_119 : Nat.gcd 153 119 = 17 :=
by
  sorry

end gcd_153_119_l609_609532


namespace probability_one_common_number_approx_l609_609388

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l609_609388


namespace valid_b2_count_l609_609208

open Nat

-- Sequence definition
def sequence (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then b 1
  else if n = 1 then b 2
  else abs (b (n + 1) - b n)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def not_divisible_by (n d : ℕ) : Prop := ¬(d ∣ n)

noncomputable def num_valid_b2 : ℕ :=
  let odds := {b2 | b2 < 1001 ∧ is_odd b2}.to_finset.card
  let div7 := {b2 | b2 < 1001 ∧ b2 % 7 = 0 ∧ is_odd b2}.to_finset.card
  let div11 := {b2 | b2 < 1001 ∧ b2 % 11 = 0 ∧ is_odd b2}.to_finset.card
  let div13 := {b2 | b2 < 1001 ∧ b2 % 13 = 0 ∧ is_odd b2}.to_finset.card
  let div77 := {b2 | b2 < 1001 ∧ b2 % (7 * 11) = 0 ∧ is_odd b2}.to_finset.card
  let div91 := {b2 | b2 < 1001 ∧ b2 % (7 * 13) = 0 ∧ is_odd b2}.to_finset.card
  let div143 := {b2 | b2 < 1001 ∧ b2 % (11 * 13) = 0 ∧ is_odd b2}.to_finset.card
  let div1001 := {b2 | b2 < 1001 ∧ b2 % (7 * 11 * 13) = 0 ∧ is_odd b2}.to_finset.card
  odds - (div7 + div11 + div13 - div77 - div91 - div143 + div1001)

theorem valid_b2_count (b : ℕ → ℕ) (h1 : b 1 = 1001) (h2 : b 2023 = 1) :
  num_valid_b2 = 219 :=
by sorry

end valid_b2_count_l609_609208


namespace factorize_xy2_minus_x_l609_609680

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609680


namespace product_of_equal_numbers_l609_609514

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l609_609514


namespace probability_of_one_common_l609_609407

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l609_609407


namespace solve_for_y_l609_609082

theorem solve_for_y (y : ℝ) 
  (h : log 3 ((4 * y + 12) / (6 * y - 4)) + log 3 ((6 * y - 4) / (y - 3)) = 3) : 
  y = 93 / 23 :=
by 
  sorry

end solve_for_y_l609_609082


namespace max_temp_range_l609_609518

theorem max_temp_range (temps : Fin 5 → ℝ) (h_avg : (∑ i, temps i) = 250) (h_min : ∃ i, temps i = 40) :
  ∃ M m, (M - m = 50 ∧ ∀ i, temps i = 40) :=
by
  sorry

end max_temp_range_l609_609518


namespace sum_ineq_l609_609861

theorem sum_ineq (n : ℕ) (x : ℕ → ℝ) 
    (hx : ∑ i in Finset.range (n+1), (x i)^2 = (1 : ℝ)) :
  (∑ k in Finset.range (n+1), (1 - k / ∑ i in Finset.range (n+1), i * (x i)^2)^2 * (x k)^2 / k) 
  ≤ (n-1 : ℝ)^2 / (n+1 : ℝ)^2 * ∑ k in Finset.range (n+1), (x k)^2 / k := 
sorry

end sum_ineq_l609_609861


namespace factorize_xy_squared_minus_x_l609_609700

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609700


namespace minimum_days_person_A_work_l609_609182

theorem minimum_days_person_A_work (d_a d_b d_c : ℕ) (d_a_days: d_a = 24) (d_b_days: d_b = 36) (d_c_days: d_c = 60)
  (total_days : ℕ) (h_total_days : total_days ≤ 18) (h_total_work : ∀ d_a_worked d_b_worked d_c_worked : ℕ,
    (d_a_worked ≤ total_days) ∧ (d_b_worked ≤ total_days) ∧ (d_c_worked ≤ total_days) → 
    (d_a_worked / d_a + d_b_worked / d_b + d_c_worked / d_c = 1)) : 
  ∃ d_a_worked : ℕ, d_a_worked ≤ total_days ∧ d_a_worked = 6 := 
begin
  sorry -- Fill in the proof steps only if needed
end

end minimum_days_person_A_work_l609_609182


namespace mrs_taylor_total_cost_correct_l609_609902

def smart_television_price : ℝ := 800
def soundbar_price : ℝ := 350
def bluetooth_speaker_price : ℝ := 100

def num_smart_televisions : ℝ := 2
def num_soundbars : ℝ := 4
def num_bluetooth_speakers : ℝ := 6

def discount_smart_televisions : ℝ := 0.20
def discount_soundbars : ℝ := 0.15

def total_smart_television_cost : ℝ := num_smart_televisions * smart_television_price
def total_soundbar_cost : ℝ := num_soundbars * soundbar_price
def total_bluetooth_speaker_cost : ℝ := num_bluetooth_speakers * bluetooth_speaker_price

def discounted_smart_television_cost : ℝ := total_smart_television_cost * (1 - discount_smart_televisions)
def discounted_soundbar_cost : ℝ := total_soundbar_cost * (1 - discount_soundbars)

def effective_bluetooth_speakers : ℝ := num_bluetooth_speakers / 2
def discounted_bluetooth_speaker_cost : ℝ := effective_bluetooth_speakers * bluetooth_speaker_price

def total_cost : ℝ := discounted_smart_television_cost + discounted_soundbar_cost + discounted_bluetooth_speaker_cost

theorem mrs_taylor_total_cost_correct : total_cost = 2770 := by
  sorry

end mrs_taylor_total_cost_correct_l609_609902


namespace crate_stacking_probability_l609_609506

theorem crate_stacking_probability :
  let crates := 10
  let dimensions := {2, 3, 5}
  let total_height := 38
  let total_stacking_configurations := 3^crates
  let valid_stacking_configurations := 2940
  (valid_stacking_configurations / total_stacking_configurations).num = 980 := by
    sorry

end crate_stacking_probability_l609_609506


namespace problem_solution_l609_609769

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l609_609769


namespace total_point_value_of_test_l609_609577

theorem total_point_value_of_test (total_questions : ℕ) (five_point_questions : ℕ) 
  (ten_point_questions : ℕ) (points_5 : ℕ) (points_10 : ℕ) 
  (h1 : total_questions = 30) (h2 : five_point_questions = 20) 
  (h3 : ten_point_questions = total_questions - five_point_questions) 
  (h4 : points_5 = 5) (h5 : points_10 = 10) : 
  five_point_questions * points_5 + ten_point_questions * points_10 = 200 :=
by
  sorry

end total_point_value_of_test_l609_609577


namespace area_FEC_D_is_960_l609_609425

-- Definitions of points and given lengths
variables (A B C D E F : Point)
variables (AB AD BF : ℝ)
variables (h1: AD // BC) (BD_perp_DC: BD ⊥ DC) (AF_perp_BD: AF ⊥ BD) (AE_ext_AF: AF_ext AE)
variables (AB_val : AB = 41) (AD_val : AD = 50) (BF_val : BF = 9)

-- Key geometric relations and their properties
noncomputable def area_quadrilateral_FEC_D : ℝ :=
  let FEC_D := quadrilateral F E C D,
  trapezoid_area FEC_D EF DC FD

theorem area_FEC_D_is_960 :
  area_quadrilateral_FEC_D = 960 :=
sorry

end area_FEC_D_is_960_l609_609425


namespace inequality_relationship_l609_609786

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l609_609786


namespace minimum_perimeter_area_l609_609332

-- Define the focus point F of the parabola and point A
def F : ℝ × ℝ := (1, 0)  -- Focus for the parabola y² = 4x is (1, 0)
def A : ℝ × ℝ := (5, 4)

-- Parabola definition as a set of points (x, y) such that y² = 4x
def is_on_parabola (B : ℝ × ℝ) : Prop := B.2 * B.2 = 4 * B.1

-- The area of triangle ABF
def triangle_area (A B F : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 - B.1) * (A.2 - F.2) - (A.1 - F.1) * (A.2 - B.2))

-- Statement: The area of ∆ABF is 2 when the perimeter of ∆ABF is minimum
theorem minimum_perimeter_area (B : ℝ × ℝ) (hB : is_on_parabola B) 
  (hA_B_perimeter_min : ∀ (C : ℝ × ℝ), is_on_parabola C → 
                        (dist A C + dist C F ≥ dist A B + dist B F)) : 
  triangle_area A B F = 2 := 
sorry

end minimum_perimeter_area_l609_609332


namespace ShapeB_is_symmetric_l609_609230

-- Definitions of properties for each shape
def is_axially_symmetric : Type -> Prop := sorry
def is_centrally_symmetric : Type -> Prop := sorry

noncomputable def ShapeA : Type := Parallelogram
noncomputable def ShapeB : Type := Circle
noncomputable def ShapeC : Type := EquilateralTriangle
noncomputable def ShapeD : Type := RegularPentagon

-- Conditions for each shape
axiom ShapeA_central_symmetry : is_centrally_symmetric ShapeA
axiom ShapeA_not_axial_symmetry : ¬ is_axially_symmetric ShapeA

axiom ShapeB_central_symmetry : is_centrally_symmetric ShapeB
axiom ShapeB_axial_symmetry : is_axially_symmetric ShapeB

axiom ShapeC_axial_symmetry : is_axially_symmetric ShapeC
axiom ShapeC_not_central_symmetry : ¬ is_centrally_symmetric ShapeC

axiom ShapeD_axial_symmetry : is_axially_symmetric ShapeD
axiom ShapeD_not_central_symmetry : ¬ is_centrally_symmetric ShapeD

-- The theorem to be proved
theorem ShapeB_is_symmetric : is_axially_symmetric ShapeB ∧ is_centrally_symmetric ShapeB := by
  sorry

end ShapeB_is_symmetric_l609_609230


namespace problem1_problem2_l609_609741

-- Defining the sequences and conditions given in the problem

def seq_a_q1_d2 (n : ℕ) : ℕ → ℕ
| 0     := 4
| (n+1) := 1 * seq_a_q1_d2 n + 2

def a_2017 : ℕ := seq_a_q1_d2 2017

-- Statement for question 1
theorem problem1 : a_2017 = 4036 := sorry

-- Sequence definitions as per the second question's conditions
def seq_a_q3_d_2 (n : ℕ) : ℕ → ℕ
| 0     := 4
| (n+1) := 3 * seq_a_q3_d_2 n - 2

def b_n (n : ℕ) := 1 / (seq_a_q3_d_2 n - 1 : ℝ)

def S_n (n : ℕ) : ℝ := (finset.range n).sum (λ i, b_n i)

-- Statement for question 2
theorem problem2 (n : ℕ) : S_n n < 1 / 2 := sorry

end problem1_problem2_l609_609741


namespace determinant_log_eq_zero_iff_x_eq_4_l609_609324

theorem determinant_log_eq_zero_iff_x_eq_4 (x : ℝ) (h : det (matrix.of ![![real.logb 2 x, -1], ![-4, 2]]) = 0) : x = 4 := 
by sorry

end determinant_log_eq_zero_iff_x_eq_4_l609_609324


namespace ratio_of_areas_of_triangles_l609_609426

-- Define the given conditions
variables {X Y Z T : Type}
variable (distance_XY : ℝ)
variable (distance_XZ : ℝ)
variable (distance_YZ : ℝ)
variable (is_angle_bisector : Prop)

-- Define the correct answer as a goal
theorem ratio_of_areas_of_triangles (h1 : distance_XY = 15)
    (h2 : distance_XZ = 25)
    (h3 : distance_YZ = 34)
    (h4 : is_angle_bisector) : 
    -- Ratio of the areas of triangle XYT to triangle XZT
    ∃ (ratio : ℝ), ratio = 3 / 5 :=
by
  -- This is where the proof would go, omitted with "sorry"
  sorry

end ratio_of_areas_of_triangles_l609_609426


namespace angle_a_value_sin_2B_A_triangle_perimeter_l609_609326

-- Question 1
theorem angle_a_value (a b c : ℝ) (A B C : ℝ) 
  (h : 2 * b = c + 2 * a * Real.cos C) 
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : 0 < A ∧ A < real.pi) 
  (h₄ : 0 < B ∧ B < real.pi) 
  (h₅ : 0 < C ∧ C < real.pi) : 
  A = real.pi / 3 :=
sorry

-- Question 2
theorem sin_2B_A (a b c : ℝ) (A B C : ℝ)
  (h : 2 * b = c + 2 * a * Real.cos C)
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0)
  (h₃ : A = real.pi / 3) 
  (h₄ : Real.cos B = Real.sqrt 3 / 3) 
  (h₅ : 0 < A ∧ A < real.pi)
  (h₆ : 0 < B ∧ B < real.pi)
  (h₇ : 0 < C ∧ C < real.pi) : 
  Real.sin (2 * B - A) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
sorry

-- Question 3
theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ)
  (area : ℝ)
  (h : 2 * b = c + 2 * a * Real.cos C)
  (h₀ : a = 3)
  (h₁ : area = 10 * Real.sqrt 3 / 3)
  (h₂ : A = real.pi / 3)
  (h₃ : 0 < A ∧ A < real.pi)
  (h₄ : 0 < B ∧ B < real.pi)
  (h₅ : 0 < C ∧ C < real.pi) :
  a + b + c = 10 :=
sorry

end angle_a_value_sin_2B_A_triangle_perimeter_l609_609326


namespace monotonically_decreasing_interval_l609_609107

noncomputable def function_y (x : ℝ) := (1 / x) - (Real.log x)

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x > 0 → ∃ a b : ℝ, a = 0 ∧ b = ∞ ∧ ∀ z ∈ Set.Ioo a b, derivative (function_y) z < 0 :=
by
  sorry

end monotonically_decreasing_interval_l609_609107


namespace tangent_line_at_1_0_l609_609945

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_line_at_1_0 :
  ∃ m b, ∀ x, f 1 = 0 ∧ (∀ x, deriv f x = 2 / x) ∧ m = 2 ∧ f 1 + m * (x - 1) = 2 * (x - 1) -> 
  f 1 + m * (x - 1) = 2x - 2 :=
by
  sorry

end tangent_line_at_1_0_l609_609945


namespace apples_for_fruit_juice_l609_609996

noncomputable def total_harvest := 405
noncomputable def apples_given_to_restaurant := 60
noncomputable def total_revenue := 408
noncomputable def price_per_bag := 8
noncomputable def weight_per_bag := 5

theorem apples_for_fruit_juice :
  let bags_sold := total_revenue / price_per_bag in
  let apples_sold_in_bags := bags_sold * weight_per_bag in
  total_harvest - (apples_given_to_restaurant + apples_sold_in_bags) = 90 :=
by
  sorry

end apples_for_fruit_juice_l609_609996


namespace cube_root_of_neg_27_l609_609943

theorem cube_root_of_neg_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 :=
by
  use -3
  split
  { norm_num }
  { refl }

end cube_root_of_neg_27_l609_609943


namespace factorization_l609_609694

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609694


namespace orchids_in_vase_now_l609_609991

-- Let's define the basic conditions first
def initial_roses : ℕ := 7
def initial_orchids : ℕ := 12
def current_roses : ℕ := 11
def orchids_more_than_roses : ℕ := 9

-- Now, we state the theorem
theorem orchids_in_vase_now : ℕ :=
  have O : ℕ := current_roses + orchids_more_than_roses,
  O

-- We add a proof that should satisfy the theorem statement. Let's have a proof with sorry for now
example : orchids_in_vase_now = 20 := by
  -- Sorry, proof is skipped
  sorry

end orchids_in_vase_now_l609_609991


namespace flat_track_stopping_distance_incline_track_stopping_distance_l609_609822

def friction_flat (speed : ℝ) (friction_coefficient : ℝ) (g : ℝ) : ℝ :=
  let v := speed * (1000 / 3600) -- converting km/h to m/s
  (1 / 2 * v^2) / (friction_coefficient * g)

def stopping_distance_flat := friction_flat 60 0.004 9.81

theorem flat_track_stopping_distance :
  stopping_distance_flat = 3538 := 
sorry

def friction_incline (speed : ℝ) (friction_coefficient : ℝ) (slope_coefficient : ℝ) (g : ℝ) : ℝ :=
  let v := speed * (1000 / 3600) -- converting km/h to m/s
  (1 / 2 * v^2) / ((friction_coefficient + slope_coefficient) * g)

def stopping_distance_incline := friction_incline 60 0.004 0.015 9.81

theorem incline_track_stopping_distance :
  stopping_distance_incline = 745 := 
sorry

end flat_track_stopping_distance_incline_track_stopping_distance_l609_609822


namespace probability_divisible_by_3_of_two_digit_prime_digit_l609_609581

open Nat

/--
  Let n be a two-digit integer formed by prime digits less than 10.
  We are required to prove that the probability w that n is divisible by 3 is 1/3.
-/
theorem probability_divisible_by_3_of_two_digit_prime_digit :
  let two_digit_prime_numbers := {23, 25, 27, 32, 35, 37, 52, 53, 57, 72, 73, 75}
  let divisible_by_3_numbers := {27, 57, 72, 75}
  let total_two_digit_prime := 12
  let count_divisible_by_3 := 4
  let w := count_divisible_by_3 / total_two_digit_prime
  w = 1 / 3 :=
by
  sorry

end probability_divisible_by_3_of_two_digit_prime_digit_l609_609581


namespace intersection_M_N_l609_609444

-- Definitions of sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The statement to prove
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := 
by 
  sorry

end intersection_M_N_l609_609444


namespace max_value_a4_a6_l609_609545

theorem max_value_a4_a6 (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6) :
  ∃ m, ∀ (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6), a 4 * a 6 ≤ m :=
sorry

end max_value_a4_a6_l609_609545


namespace sum_of_coordinates_D_l609_609919

theorem sum_of_coordinates_D (x y : ℝ) 
  (M_midpoint : (4, 10) = ((8 + x) / 2, (6 + y) / 2)) : 
  x + y = 14 := 
by 
  sorry

end sum_of_coordinates_D_l609_609919


namespace least_positive_difference_3_l609_609499

def geometric_sequence_up_to_max (a r max : ℕ) : List ℕ :=
  List.takeWhile (λ x => x ≤ max) (List.iterate (· * r) a)

def arithmetic_sequence_up_to_max (a d max : ℕ) : List ℕ :=
  List.takeWhile (λ x => x ≤ max) (List.iterate (· + d) a)

def min_positive_difference (lst1 lst2 : List ℕ) : ℕ :=
  List.minimum (List.filter (λ x => x > 0) [abs (x - y) | x ← lst1, y ← lst2])

theorem least_positive_difference_3 :
  min_positive_difference
    (geometric_sequence_up_to_max 3 3 300)
    (arithmetic_sequence_up_to_max 15 15 300) = 3 := by
  sorry

end least_positive_difference_3_l609_609499


namespace expenditure_representation_l609_609652

theorem expenditure_representation
    (income_representation : ℤ)
    (income_is_positive : income_representation = 60)
    (use_of_opposite_signs : ∀ (n : ℤ), n >= 0 ↔ n = 60)
    (representation_criteria : ∀ (x : ℤ), x <= 0 → x = -40) :
  ∀ (expenditure : ℤ), expenditure = 40 → -expenditure = -40 :=
by
  intro expenditure
  intro expenditure_criteria
  rw [neg_eq_neg_one_mul]
  change (-1) * expenditure = (-40)
  apply representation_criteria
  exact le_of_eq rfl
  sorry

end expenditure_representation_l609_609652


namespace factorize_xy2_minus_x_l609_609682

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609682


namespace curlers_total_l609_609273

theorem curlers_total (P B G : ℕ) (h1 : 4 * P = P + B + G) (h2 : B = 2 * P) (h3 : G = 4) : 
  4 * P = 16 := 
by sorry

end curlers_total_l609_609273


namespace factorize_xy_squared_minus_x_l609_609704

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609704


namespace length_of_bridge_l609_609138

theorem length_of_bridge (L : ℝ) 
  (length_A : ℝ := 130) (speed_A_kmhr : ℝ := 45)
  (length_B : ℝ := 150) (speed_B_kmhr : ℝ := 60)
  (time_sec : ℝ := 30) :
  (let speed_A := speed_A_kmhr * 1000 / 3600 in
   let speed_B := speed_B_kmhr * 1000 / 3600 in
   let distance_A := speed_A * time_sec in
   let distance_B := speed_B * time_sec in
   let total_distance := distance_A + length_A + distance_B + length_B in
   L = total_distance) :=
by
  sorry

end length_of_bridge_l609_609138


namespace probability_arithmetic_progression_l609_609124

-- Condition: Three fair ten-sided dice are rolled.
-- Define a probability space and the events of interest.
def num_faces := 10
def total_outcomes := num_faces ^ 3
def arithmetic_progressions : Finset (Finset (Fin num_faces)) :=
  { {(1,2,3), (2,3,4), (3,4,5), (4,5,6), (5,6,7), (6,7,8), (7,8,9), (8,9,10)} }

def count_permutations (s : Finset (Fin num_faces)) : ℕ :=
  s.card * Nat.factorial 3

-- Lean statement: Proving the probability
theorem probability_arithmetic_progression :
  let favorable_outcomes := arithmetic_progressions.sum count_permutations in
  favorable_outcomes / total_outcomes = 6 / 125 := sorry

end probability_arithmetic_progression_l609_609124


namespace sum_of_A_and_B_l609_609571

theorem sum_of_A_and_B (A B : ℕ) (h1 : (1 / 6 : ℚ) * (1 / 3) = 1 / (A * 3))
                       (h2 : (1 / 6 : ℚ) * (1 / 3) = 1 / B) : A + B = 24 :=
by
  sorry

end sum_of_A_and_B_l609_609571


namespace percentage_big_bottles_sold_l609_609211

/-- 
Given:
1. The company had 6000 small bottles in storage.
2. The company had 15000 big bottles in storage.
3. 11% of the small bottles have been sold.
4. The total number of bottles remaining in storage is 18540.

Prove the percentage of big bottles sold.
-/

theorem percentage_big_bottles_sold (total_small : ℕ) (total_big : ℕ)
  (small_sold_percent : ℝ) (remaining_bottles : ℕ) :
  total_small = 6000 → total_big = 15000 → small_sold_percent = 0.11 → 
  remaining_bottles = 18540 → 
  let small_sold := (small_sold_percent : ℝ) * (total_small : ℝ) in
  let big_sold := (12 / 100 : ℝ) * (total_big : ℝ) in
  total_small - small_sold + total_big - big_sold = remaining_bottles :=
by {
  intros,
  sorry
}

end percentage_big_bottles_sold_l609_609211


namespace connor_study_time_proof_l609_609037

-- Define Kwame's study time in hours
def kwame_study_hours : ℝ := 2.5

-- Define Lexia's study time in minutes
def lexia_study_minutes : ℝ := 97

-- Define the additional time Kwame and Connor studied together compared to Lexia
def additional_time_minutes : ℝ := 143

-- Convert Kwame's study time to minutes
def kwame_study_minutes : ℝ := kwame_study_hours * 60

-- Define the combined study time of Kwame and Connor in minutes
def combined_study_minutes : ℝ := lexia_study_minutes + additional_time_minutes

-- Define Connor's study time in minutes
def connor_study_minutes : ℝ := combined_study_minutes - kwame_study_minutes

-- Convert Connor's study time back to hours
def connor_study_hours : ℝ := connor_study_minutes / 60

-- The theorem to prove
theorem connor_study_time_proof : connor_study_hours = 1.5 := 
by
  -- Placeholder for the proof
  sorry

end connor_study_time_proof_l609_609037


namespace omega_sum_equals_one_l609_609872

variables (ω : ℂ) (h₀ : ω^5 = 1) (h₁ : ω ≠ 1)

theorem omega_sum_equals_one :
  (ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45) = 1 :=
begin
  sorry
end

end omega_sum_equals_one_l609_609872


namespace erased_number_is_six_l609_609905

theorem erased_number_is_six (n x : ℕ) (h1 : (n * (n + 1)) / 2 - x = 45 * (n - 1) / 4):
  x = 6 :=
by
  sorry

end erased_number_is_six_l609_609905


namespace number_of_red_balls_l609_609011

theorem number_of_red_balls
    (black_balls : ℕ)
    (frequency : ℝ)
    (total_balls : ℕ)
    (red_balls : ℕ) 
    (h_black : black_balls = 5)
    (h_frequency : frequency = 0.25)
    (h_total : total_balls = black_balls / frequency) :
    red_balls = total_balls - black_balls → red_balls = 15 :=
by
  intros h_red
  sorry

end number_of_red_balls_l609_609011


namespace fraction_of_girls_in_debate_l609_609639

theorem fraction_of_girls_in_debate (g b : ℕ) (h : g = b) :
  ((2 / 3) * g) / ((2 / 3) * g + (3 / 5) * b) = 30 / 57 :=
by
  sorry

end fraction_of_girls_in_debate_l609_609639


namespace trig_simplify_l609_609753

variable (α : ℝ) (h1 : sin α > 0) (h2 : cos α < 0)

theorem trig_simplify : 
  (√(1 + 2 * sin (5 * real.pi - α) * cos (α - real.pi)) / 
    (sin (α - 3 / 2 * real.pi) - √(1 - sin (3 / 2 * real.pi + α)^2))) = -1 :=
  sorry

end trig_simplify_l609_609753


namespace pens_left_after_sale_l609_609455

variable (initial_pens sold_pens final_pens : ℕ)

-- Conditions
axiom initial_pens_condition : initial_pens = 106
axiom sold_pens_condition : sold_pens = 92
axiom final_pens_condition : final_pens = initial_pens - sold_pens

-- Statement to prove
theorem pens_left_after_sale : final_pens = 14 :=
by
  -- Conditions recap
  have hi : initial_pens = 106 := initial_pens_condition
  have hs : sold_pens = 92 := sold_pens_condition
  have hf : final_pens = initial_pens - sold_pens := final_pens_condition
  -- The expected proof step, skipped with "sorry".
  sorry

end pens_left_after_sale_l609_609455


namespace angle_bisector_length_l609_609073

variable (a b : ℝ) (α l : ℝ)

theorem angle_bisector_length (ha : 0 < a) (hb : 0 < b) (hα : 0 < α) (hl : l = (2 * a * b * Real.cos (α / 2)) / (a + b)) :
  l = (2 * a * b * Real.cos (α / 2)) / (a + b) := by
  -- problem assumptions
  have h1 : a > 0 := ha
  have h2 : b > 0 := hb
  have h3 : α > 0 := hα
  -- conclusion
  exact hl

end angle_bisector_length_l609_609073


namespace distinct_sequences_equal_sixty_l609_609359

-- Define the available letters and conditions
def available_letters : List Char := ['R', 'E', 'Q', 'U', 'E', 'N', 'C']
def vowels : List Char := ['E', 'U']
def start_letter : Char := 'F'
def end_letter : Char := 'Y'

-- Function to count distinct valid sequences
noncomputable def count_sequences (letters : List Char) (vowels : List Char) (start end : Char) : Nat :=
  let second_pos_choices := vowels.length    -- Choices for the second position
  let third_pos_choices := (letters.length - 1)  -- Choices for third position after fixing second
  let fourth_pos_choices := (letters.length - 2) -- Choices for fourth position after fixing third
  second_pos_choices * third_pos_choices * fourth_pos_choices

-- Proof goal
theorem distinct_sequences_equal_sixty :
  count_sequences available_letters vowels start_letter end_letter = 60 :=
by
  sorry

end distinct_sequences_equal_sixty_l609_609359


namespace sue_total_expenditure_l609_609676

def cost_of_apples : ℕ := 4 * 2
def cost_of_juice : ℕ := 2 * 6
def cost_of_bread : ℕ := 3 * 3
def cost_of_cheese (discounted_price : ℕ) : ℕ := 2 * discounted_price
def cost_of_cereal : ℕ := 1 * 8

def discounted_price_of_cheese (original_price : ℕ) (discount : ℕ) : ℕ :=
  original_price - (original_price * discount / 100)

def total_without_coupon (apples : ℕ) (juice : ℕ) (bread : ℕ) (cheese : ℕ) (cereal : ℕ) : ℕ :=
  apples + juice + bread + cheese + cereal

def discount_amount (total : ℕ) (discount : ℕ) : ℕ :=
  total * discount / 100

def final_total (total : ℕ) (discount : ℕ) : ℕ :=
  total - discount

theorem sue_total_expenditure :
  let original_cheese_price := 4
  let cheese_discount := 25
  let coupon_threshold := 40
  let coupon_discount := 10
  let discounted_cheese_price := discounted_price_of_cheese original_cheese_price cheese_discount
  let total := total_without_coupon cost_of_apples cost_of_juice cost_of_bread (cost_of_cheese discounted_cheese_price) cost_of_cereal
  let final_amount := if total >= coupon_threshold then final_total total (discount_amount total coupon_discount) else total
  final_amount = 38.7 :=
by
  sorry

end sue_total_expenditure_l609_609676


namespace t_values_range_l609_609327

variable {f : ℝ → ℝ}

-- Definitions and conditions
def odd_function_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := 
∀ x ∈ set.Icc a b, f (-x) = -f x

def function_bounds (f : ℝ → ℝ) (t a : ℝ) : Prop := 
∀ x ∈ set.Icc (-1 : ℝ) 1, f x ≤ t^2 - 2 * a * t + 1

def increasing_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y ∈ set.Icc a b, x < y → f x < f y

def condition_3 (f : ℝ → ℝ) (m n : ℝ) : Prop :=
m + n ≠ 0 → (m ∈ set.Icc (-1 : ℝ) 1) → (n ∈ set.Icc (-1 : ℝ) 1) → (f m + f n) / (m + n) > 0

theorem t_values_range (f : ℝ → ℝ) 
  (h1 : odd_function_on_interval f (-1) 1)
  (h2 : f 1 = 1)
  (h3 : ∀ m n ∈ set.Icc (-1 : ℝ) 1, m + n ≠ 0 → (f m + f n) / (m + n) > 0)
  (h4 : ∀ t, ∀ a ∈ set.Icc (-1 : ℝ) 1, function_bounds f t a) :
  ∀ t : ℝ, (t ≤ -2) ∨ (t = 0) ∨ (t ≥ 2) :=
sorry

end t_values_range_l609_609327


namespace convex_polyhedron_volume_surface_area_l609_609188

theorem convex_polyhedron_volume_surface_area :
  ∃ (P : Polyhedron), P.convex ∧ (volume_below_water P / total_volume P = 0.9) ∧ (surface_area_above_water P / total_surface_area P > 0.5) :=
sorry

end convex_polyhedron_volume_surface_area_l609_609188


namespace buyers_muffin_mix_l609_609594

variable (P C M CM: ℕ)

theorem buyers_muffin_mix
    (h_total: P = 100)
    (h_cake: C = 50)
    (h_both: CM = 17)
    (h_neither: P - (C + M - CM) = 27)
    : M = 73 :=
by sorry

end buyers_muffin_mix_l609_609594


namespace compare_values_l609_609773

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l609_609773


namespace parallelogram_diagonal_l609_609839

-- Definitions of vectors and properties of a parallelogram
variable (V : Type) [AddCommGroup V] [Module ℝ V] -- assuming vectors belong to a real vector space
variables (A B C D : V)

-- Condition: ABCD is a parallelogram
def is_parallelogram := (B - A = D - C) ∧ (C - B = D - A)

-- Theorem: In a parallelogram ABCD, the vector sum of AB and AD equals AC
theorem parallelogram_diagonal (h : is_parallelogram A B C D) : (B - A) + (D - A) = C - A :=
by
  sorry

end parallelogram_diagonal_l609_609839


namespace part_a_part_b_l609_609641

noncomputable def curl (A : Π (i : ℕ), ℝ^3 → ℝ) : ℝ^3 → ℝ^3 :=
  λ x, ⟨(∂ (A 2 x) / ∂ x.2 - ∂ (A 1 x) / ∂ x.3,
         ∂ (A 2 x) / ∂ x.1 - ∂ (A 0 x) / ∂ x.3,
         ∂ (A 1 x) / ∂ x.1 - ∂ (A 0 x) / ∂ x.2)⟩

theorem part_a (x y z : ℝ) :
  curl (λ i, match i with
                 | 0 := λ_, x
                 | 1 := λ_, -z^2
                 | _ := λ_, y^2) = (λ x, ⟨2 * x.2 + 2 * x.3, 0, 0⟩) :=
by
  ext1
  simp [curl, Function.uncurry]
  repeat { sorry }

theorem part_b (x y z : ℝ) :
  curl (λ i, match i with
                 | 0 := λ_, y * z
                 | 1 := λ_, x * z
                 | _ := λ_, x * y) = 0 :=
by
  ext1
  simp [curl, Function.uncurry]
  repeat { sorry }

end part_a_part_b_l609_609641


namespace soccer_team_wins_l609_609210

theorem soccer_team_wins 
  (total_matches : ℕ)
  (total_points : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ)
  (points_per_loss : ℕ)
  (losses : ℕ)
  (H1 : total_matches = 10)
  (H2 : total_points = 17)
  (H3 : points_per_win = 3)
  (H4 : points_per_draw = 1)
  (H5 : points_per_loss = 0)
  (H6 : losses = 3) : 
  ∃ (wins : ℕ), wins = 5 := 
by
  sorry

end soccer_team_wins_l609_609210


namespace minimize_S_n_l609_609344

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d a1 : ℤ)

-- Conditions
def arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a1 + n * d
def sum_of_first_n_terms : Prop := ∀ n : ℕ, S n = n * (a1 + a1 + (n - 1) * d) / 2
def sequence_conditions : Prop := a1 + 2 * (a1 + 2 * d) = -14 ∧ (9 * (a1 + 4 * d)) / 2 = -27

-- Goal
theorem minimize_S_n (h1 : arithmetic_sequence a S) (h2 : sum_of_first_n_terms S a1 d) (h3 : sequence_conditions a1 d) :
  ∃ n, S n = n * (a1 + a1 + (n - 1) * d) / 2 ∧ n = 6 := 
sorry

end minimize_S_n_l609_609344


namespace jeff_average_skips_is_14_l609_609079

-- Definitions of the given conditions in the problem
def sam_skips_per_round : ℕ := 16
def rounds : ℕ := 4

-- Number of skips by Jeff in each round based on the conditions
def jeff_first_round_skips : ℕ := sam_skips_per_round - 1
def jeff_second_round_skips : ℕ := sam_skips_per_round - 3
def jeff_third_round_skips : ℕ := sam_skips_per_round + 4
def jeff_fourth_round_skips : ℕ := sam_skips_per_round / 2

-- Total skips by Jeff in all rounds
def jeff_total_skips : ℕ := jeff_first_round_skips + 
                           jeff_second_round_skips + 
                           jeff_third_round_skips + 
                           jeff_fourth_round_skips

-- Average skips per round by Jeff
def jeff_average_skips : ℕ := jeff_total_skips / rounds

-- Theorem statement
theorem jeff_average_skips_is_14 : jeff_average_skips = 14 := 
by 
    sorry

end jeff_average_skips_is_14_l609_609079


namespace prism_volume_l609_609556

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end prism_volume_l609_609556


namespace constant_term_of_expansion_l609_609343

-- Define the expression and the sum of coefficients condition
def expr (x : ℝ) : ℝ := x^3 + 2 / x^2

-- Given condition
def sum_of_coeffs (n : ℕ) := (expr 1) ^ n

-- Required to prove
theorem constant_term_of_expansion :
  sum_of_coeffs 5 = 243 →
  let n := 5 in
  let term := finset.sum (finset.range (n + 1)) (λ r, nat.choose n r * 2^r * (1 : ℝ)^(15 - 5*r)) in
  term = 80 :=
by
  intro h
  sorry

end constant_term_of_expansion_l609_609343


namespace inequality_abc_l609_609784

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l609_609784


namespace smallest_area_of_triangle_l609_609040

section
variables (t : ℝ)
def A := (-1, 1, 2 : ℝ × ℝ × ℝ)
def B := (1, 2, 3 : ℝ × ℝ × ℝ)
def C := (t, 2, 2 : ℝ × ℝ × ℝ)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.1 - v.1, u.2 - v.2, u.3 - v.3)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

def area_of_triangle (A B C : ℝ × ℝ × ℝ) : ℝ :=
let AB := vector_sub B A,
    AC := vector_sub C A,
    cp := cross_product AB AC
in (1 / 2) * magnitude cp

theorem smallest_area_of_triangle :
  ∃ (t : ℝ), area_of_triangle A B C = 1.5 :=
sorry
end

end smallest_area_of_triangle_l609_609040


namespace difference_sum_even_odd_1000_l609_609142

open Nat

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_sum_even_odd_1000 :
  sum_first_n_even 1000 - sum_first_n_odd 1000 = 1000 :=
by
  sorry

end difference_sum_even_odd_1000_l609_609142


namespace find_line_eq_l609_609312

-- Definitions for the conditions
def passes_through_M (l : ℝ × ℝ) : Prop :=
  l = (1, 2)

def segment_intercepted_length (l : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ,
    ∀ p : ℝ × ℝ, l p → ((4 * p.1 + 3 * p.2 + 1 = 0 ∨ 4 * p.1 + 3 * p.2 + 6 = 0) ∧ (A = p ∨ B = p)) ∧
    dist A B = Real.sqrt 2

-- Predicates for the lines to be proven
def line_eq1 (p : ℝ × ℝ) : Prop :=
  p.1 + 7 * p.2 = 15

def line_eq2 (p : ℝ × ℝ) : Prop :=
  7 * p.1 - p.2 = 5

-- The proof problem statement
theorem find_line_eq (l : ℝ × ℝ → Prop) :
  passes_through_M (1, 2) →
  segment_intercepted_length l →
  (∀ p, l p → line_eq1 p) ∨ (∀ p, l p → line_eq2 p) :=
by
  sorry

end find_line_eq_l609_609312


namespace cube_root_simplify_l609_609570

theorem cube_root_simplify : 
  ∃ (c d : ℕ), (c * real.cbrt d = real.cbrt 1600) ∧ (d < 1600) ∧ (c + d = 29) :=
sorry

end cube_root_simplify_l609_609570


namespace total_shaded_area_l609_609616

theorem total_shaded_area 
  (side': ℝ) (d: ℝ) (s: ℝ)
  (h1: 12 / d = 4)
  (h2: d / s = 4) : 
  d = 3 →
  s = 3 / 4 →
  (π * (d / 2) ^ 2 + 8 * s ^ 2) = 9 * π / 4 + 9 / 2 :=
by
  intro h3 h4
  have h5 : d = 3 := h3
  have h6 : s = 3 / 4 := h4
  rw [h5, h6]
  sorry

end total_shaded_area_l609_609616


namespace tangent_circumcircle_l609_609422

theorem tangent_circumcircle (
  (A B C : Type) [metric_space A] [inner_product_space ℝ A] 
  (tri : triangle A B C) 
  (D : point A) (hD : D ∈ line.segment AC)
  (hABC_right : ∠ACB = 90°)
  (E : point A) (hE : ∃ (DE_perpendicular_AB : line.perpendicular_to DE AB), E ∈ circle.circumference_tri ABC DE_perpendicular_AB)
) : AE.tangent_to (circle.circumcircle ΔCDE) := sorry

end tangent_circumcircle_l609_609422


namespace diagonal_less_than_half_perimeter_l609_609167

theorem diagonal_less_than_half_perimeter (a b c d x : ℝ) 
  (h1 : x < a + b) (h2 : x < c + d) : x < (a + b + c + d) / 2 := 
by
  sorry

end diagonal_less_than_half_perimeter_l609_609167


namespace find_range_for_two_real_solutions_l609_609352

noncomputable def f (k x : ℝ) := k * x
noncomputable def g (x : ℝ) := (Real.log x) / x

noncomputable def h (x : ℝ) := (Real.log x) / (x^2)

theorem find_range_for_two_real_solutions :
  (∃ k : ℝ, ∀ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → (f k x = g x ↔ k ∈ Set.Icc (1 / Real.exp 2) (1 / (2 * Real.exp 1)))) :=
sorry

end find_range_for_two_real_solutions_l609_609352


namespace intervals_monotonically_increasing_cosine_angle_POQ_l609_609730

/-
  Given vectors and function:
  a = (sin (π * x), 1)
  b = ( √3, cos (π * x) )
  f(x) = a · b
-/

def a (x : ℝ) : ℝ × ℝ := (Real.sin (Real.pi * x), 1)
def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.cos (Real.pi * x))
def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- 1. Prove that the intervals where f(x) is monotonically increasing are [0, 1/3] and [4/3, 2]

theorem intervals_monotonically_increasing :
  (∀ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), (Set.Icc (0 : ℝ) (1/3 : ℝ)).indicator (λ x, Dite.coe (λ (hx : x ∈ Set.Icc 0 2), True, Dite.false) x) x 0 ∧
   (Set.Icc (4/3 : ℝ) (2 : ℝ)).indicator (λ x, Dite.coe (λ (hx : x ∈ Set.Icc 0 2), True, Dite.false) x) x 0) :=
by
  sorry

-- 2. Prove that the cosine value of ∠POQ is -16 * (√481) / 481

def P : ℝ × ℝ := (1/3, 2)
def Q : ℝ × ℝ := (4/3, -2)
def O : ℝ × ℝ := (0, 0)

def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem cosine_angle_POQ:
  let OP := distance O P
  let OQ := distance O Q
  let PQ := distance P Q
  (OP ^ 2 + OQ ^ 2 - PQ ^ 2) / (2 * OP * OQ) = - (16 * Real.sqrt 481) / 481 :=
by
  sorry

end intervals_monotonically_increasing_cosine_angle_POQ_l609_609730


namespace proof_C_D_values_l609_609527

-- Given the conditions
def denominator_factorization (x : ℝ) : Prop :=
  3 * x ^ 2 - x - 14 = (3 * x + 7) * (x - 2)

def fraction_equality (x : ℝ) (C D : ℝ) : Prop :=
  (3 * x ^ 2 + 7 * x - 20) / (3 * x ^ 2 - x - 14) =
  C / (x - 2) + D / (3 * x + 7)

-- The values to be proven
def values_C_D : Prop :=
  ∃ C D : ℝ, C = -14 / 13 ∧ D = 81 / 13 ∧ ∀ x : ℝ, (denominator_factorization x → fraction_equality x C D)

theorem proof_C_D_values : values_C_D :=
sorry

end proof_C_D_values_l609_609527


namespace axis_of_symmetry_translated_sine_function_l609_609376

theorem axis_of_symmetry_translated_sine_function :
  ∀ k : ℤ, ∀ x : ℝ,
    (∃ a : ℝ, a = -(π / 12)) → y = 2 * sin (2 * (x + a) + π / 6) →
    (2 * x + π / 3 = k * π + π / 2) →

    x = k * (π / 2) + π / 12 :=
by
  sorry

end axis_of_symmetry_translated_sine_function_l609_609376


namespace inequality_relationship_l609_609788

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l609_609788


namespace smallest_n_simplest_form_fractions_l609_609302

noncomputable def smallest_n : ℕ :=
  let k := n + 1 in
    if ∀ m ∈ ({5, 6,..24}: finset ℕ), Nat.coprime k m then n else 0

theorem smallest_n_simplest_form_fractions :
  smallest_n = 28 :=
  sorry

end smallest_n_simplest_form_fractions_l609_609302


namespace quadratic_value_at_5_l609_609530

theorem quadratic_value_at_5 {a b c m : ℝ} 
  (h1 : ∀ x, y = a * x ^ 2 + b * x + c)
  (h2 : (2:ℝ) \in [4,4])
  (h3 : y (0) = -8)
  (h4: y(5) = m):
  m =  31 :=
sorry

end quadratic_value_at_5_l609_609530


namespace extra_birds_l609_609976

def num_sparrows : ℕ := 10
def num_robins : ℕ := 5
def num_bluebirds : ℕ := 3
def nests_for_sparrows : ℕ := 4
def nests_for_robins : ℕ := 2
def nests_for_bluebirds : ℕ := 2

theorem extra_birds (num_sparrows : ℕ)
                    (num_robins : ℕ)
                    (num_bluebirds : ℕ)
                    (nests_for_sparrows : ℕ)
                    (nests_for_robins : ℕ)
                    (nests_for_bluebirds : ℕ) :
    num_sparrows = 10 ∧ 
    num_robins = 5 ∧ 
    num_bluebirds = 3 ∧ 
    nests_for_sparrows = 4 ∧ 
    nests_for_robins = 2 ∧ 
    nests_for_bluebirds = 2 ->
    num_sparrows - nests_for_sparrows = 6 ∧ 
    num_robins - nests_for_robins = 3 ∧ 
    num_bluebirds - nests_for_bluebirds = 1 :=
by sorry

end extra_birds_l609_609976


namespace range_of_a_l609_609947

noncomputable def decreasing_log_function (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 1 - real.sqrt 3 → 2 * x < a ∧ (x ^ 2 - a * x - a) > 0

theorem range_of_a (a : ℝ) : 
  decreasing_log_function a ↔ 2 * (1 - real.sqrt 3) ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l609_609947


namespace joe_paint_usage_l609_609429

theorem joe_paint_usage :
  let total_paint := 360
  let paint_first_week := total_paint * (1 / 4)
  let remaining_paint_after_first_week := total_paint - paint_first_week
  let paint_second_week := remaining_paint_after_first_week * (1 / 7)
  paint_first_week + paint_second_week = 128.57 :=
by
  sorry

end joe_paint_usage_l609_609429


namespace circle_radius_squared_l609_609598

theorem circle_radius_squared
  {r : ℝ} {AB CD BP : ℝ} 
  (h1 : AB = 12) 
  (h2 : CD = 8) 
  (h3 : BP = 9) 
  (h4 : ∃ P : ℝ, ∀ A B C D : ℝ, ∠APD = 90)
  : r^2 = 97.361 :=
sorry

end circle_radius_squared_l609_609598


namespace inheritance_total_1464_l609_609611

theorem inheritance_total_1464
  (S1 S2 S3 S4 D N : ℕ)
  (h1 : sqrt S1 = S2 / 2)
  (h2 : sqrt S1 = S3 - 2)
  (h3 : sqrt S1 = S4 + 2)
  (h4 : sqrt S1 = 2 * D)
  (h5 : sqrt S1 = N ^ 2)
  (h6 : S1 + S2 + S3 + S4 + D + N = 1464) :
  true :=
sorry

end inheritance_total_1464_l609_609611


namespace problem_l609_609368

noncomputable def cubeRoot (x : ℝ) : ℝ :=
  x ^ (1 / 3)

theorem problem (t : ℝ) (h : t = 1 / (1 - cubeRoot 2)) :
  t = (1 + cubeRoot 2) * (1 + cubeRoot 4) :=
by
  sorry

end problem_l609_609368


namespace line_intersects_parabola_at_one_point_l609_609950

theorem line_intersects_parabola_at_one_point (k : ℝ) :
    (∃ y : ℝ, x = -3 * y^2 - 4 * y + 7) ↔ (x = k) := by
  sorry

end line_intersects_parabola_at_one_point_l609_609950


namespace sample_group_b_correct_l609_609128

noncomputable def stratified_sample_group_b (total_cities: ℕ) (group_b_cities: ℕ) (sample_size: ℕ) : ℕ :=
  (sample_size * group_b_cities) / total_cities

theorem sample_group_b_correct : stratified_sample_group_b 36 12 12 = 4 := by
  sorry

end sample_group_b_correct_l609_609128


namespace complex_point_addition_l609_609371

theorem complex_point_addition (z : ℂ) (hz : z = 2 + 5 * complex.i) : 
    (1 + z).re = 3 ∧ (1 + z).im = 5 := by
    sorry

end complex_point_addition_l609_609371


namespace final_number_of_infected_computers_is_zero_l609_609254

-- Define the given conditions as Lean definitions and the final proof problem statement

def computers_ring : list ℕ := list.range 1 101

def initial_viruses : list ℕ := list.range 1 101

noncomputable def infects (current_computer : ℕ) : list ℕ :=
  if current_computer ∈ computers_ring then
    list.take_while (λ c, c ≠ current_computer) (list.drop_while (λ c, c ≠ current_computer) (computers_ring ++ computers_ring))
  else list.nil

noncomputable def virus_dies (current_computer : ℕ) (infected_computers : list ℕ) : bool :=
  current_computer ∈ infected_computers

noncomputable def number_of_infected_computers (computers : list ℕ) (viruses : list ℕ) : ℕ :=
  -- Here, implement the logic that returns the number of infected computers
  -- This logic will have to consider the movement of viruses, infection state, and when they die
  sorry

theorem final_number_of_infected_computers_is_zero :
  number_of_infected_computers computers_ring initial_viruses = 0 :=
begin
  -- Proof involves showing that after all virus movements and restorations,
  -- the count of infected computers remains zero
  sorry
end

end final_number_of_infected_computers_is_zero_l609_609254


namespace probability_pants_different_color_shirt_l609_609859

open Set

def colors_pants : Set String := {"black", "gold", "silver"}
def colors_shirts : Set String := {"black", "white", "gold"}

def equally_likely (s : Set α) : Prop := ∀ x ∈ s, 1 / (cardinal.mk s).to_nat = 1 / (cardinal.mk s).to_nat

def probability_different_color (pants : Set String) (shirts : Set String) : ℚ :=
  let total_configurations := (cardinal.mk pants * cardinal.mk shirts).to_nat
  let non_matching_configurations :=
    (pants.filter (λ p, (shirts.filter (λ s, p ≠ s)).nonempty)).to_finset.card
  (non_matching_configurations : ℚ) / total_configurations

theorem probability_pants_different_color_shirt :
  equally_likely colors_pants →
  equally_likely colors_shirts →
  probability_different_color colors_pants colors_shirts = 7 / 9 :=
by
  sorry

end probability_pants_different_color_shirt_l609_609859


namespace collinear_F_G_T_l609_609021

-- Assume all points and their relationships from the problem.
variables {A B C D E F G T : Point}

-- Geometric Relationships
axiom triangle_ABC (hABC : Triangle A B C)
axiom acute_triangle (hacute : acute hABC)
axiom AB_gt_AC (hAB_GT_AC : distance A B > distance A C)

axiom altitude_CD (hCD : Altitude C D A B)
axiom altitude_BE (hBE : Altitude B E A C)

axiom DE_intersects_BC_at_T (hDET : LineThrough D E ∩ LineThrough B C = {T})

axiom DF_perpendicular_to_BC (hDF : Perpendicular D F B C ∧ LineThrough D F ∩ LineThrough B E = {F})
axiom EG_perpendicular_to_BC (hEG : Perpendicular E G B C ∧ LineThrough E G ∩ LineThrough C D = {G})

-- Theorem
theorem collinear_F_G_T : Collinear F G T :=
sorry

end collinear_F_G_T_l609_609021


namespace mode_of_scores_is_95_l609_609966

def scores : List ℕ := [62, 65, 65, 71, 73, 79, 80, 84, 86, 86, 88, 92, 94, 95, 95, 95, 95, 95, 101, 101, 101, 103, 103, 110, 110, 110]

theorem mode_of_scores_is_95 : mode scores = 95 :=
by sorry

end mode_of_scores_is_95_l609_609966


namespace log_equation_solution_l609_609363

theorem log_equation_solution :
  ∀ y : ℝ, (log 3 (y^2) + log (1/3) y = 6) → y = 729 :=
by
  sorry

end log_equation_solution_l609_609363


namespace geometric_progression_first_term_l609_609971

theorem geometric_progression_first_term (a r : ℝ) 
  (h_sum_infinity : a / (1 - r) = 15)
  (h_sum_two_terms : a + a * r = 10) :
  (a = 15 * (sqrt 3 - 1) / sqrt 3) ∨ (a = 15 * (sqrt 3 + 1) / sqrt 3) :=
by
  sorry

end geometric_progression_first_term_l609_609971


namespace divide_to_one_in_seven_steps_l609_609504

noncomputable def divide_and_floor (n : ℕ) : ℕ :=
  n / 2

theorem divide_to_one_in_seven_steps :
  let start := 128 in
  let step1 := divide_and_floor start in
  let step2 := divide_and_floor step1 in
  let step3 := divide_and_floor step2 in
  let step4 := divide_and_floor step3 in
  let step5 := divide_and_floor step4 in
  let step6 := divide_and_floor step5 in
  let step7 := divide_and_floor step6 in
  step7 = 1 :=
by
  sorry

end divide_to_one_in_seven_steps_l609_609504


namespace wise_men_opinions_stable_l609_609588

-- Define the setup
def wise_men_opinions (n : ℕ) := Vector bool n

-- Define the rule of changing opinion
def change_opinion (opinions : wise_men_opinions 101) : wise_men_opinions 101 :=
  opinions.map_with_index (λ i opinion, 
    let prev_opinion := opinions.get (i - 1) in
    let next_opinion := opinions.get ((i + 1) % 101) in
    if prev_opinion ≠ opinion ∧ opinion ≠ next_opinion then !opinion else opinion)

-- State that eventually the opinions converge to a stable state
theorem wise_men_opinions_stable : 
  ∃ n : ℕ, ∀ m ≥ n, change_opinion^m (initial_opinions : wise_men_opinions 101) = change_opinion^(m + 1) (initial_opinions) :=
sorry

end wise_men_opinions_stable_l609_609588


namespace counterexample_to_not_prime_implies_prime_l609_609666

theorem counterexample_to_not_prime_implies_prime (n : ℕ) (hn : ¬Prime n) (hn2 : ¬Prime (n + 2)) : n = 8 :=
by {
  have h8_not_prime : ¬Prime 8 := by sorry,
  have h10_not_prime : ¬Prime 10 := by sorry,
  -- To complete the proof, we would need detailed verification of above statements,
  -- but since the task does not require the proof, we'll use "sorry".
  sorry
}

end counterexample_to_not_prime_implies_prime_l609_609666


namespace total_surface_area_of_cuts_l609_609601

/-- Given a cube with an edge length of 2 Chinese feet, 
    which is cut 4 times horizontally and 5 times vertically, 
    the total surface area of all the small blocks is 96 square Chinese feet. --/
theorem total_surface_area_of_cuts 
  (edge_length : ℝ := 2) 
  (horizontal_cuts : ℕ := 4) 
  (vertical_cuts : ℕ := 5) : 
  let original_surface_area := 6 * (edge_length ^ 2),
      new_faces_from_horizontal := 8 * (edge_length ^ 2),
      new_faces_from_vertical := 10 * (edge_length ^ 2),
      added_surface_area := new_faces_from_horizontal + new_faces_from_vertical in
  original_surface_area + added_surface_area = 96 := 
by 
  sorry

end total_surface_area_of_cuts_l609_609601


namespace total_pieces_of_gum_l609_609077

def packages := 43
def pieces_per_package := 23
def extra_pieces := 8

theorem total_pieces_of_gum :
  (packages * pieces_per_package) + extra_pieces = 997 := sorry

end total_pieces_of_gum_l609_609077


namespace n_equals_4_l609_609735

noncomputable def binomial_sum (n : ℕ) : ℕ :=
 ∑ k in finset.range n, (nat.choose n (k + 1)) * 2^k

theorem n_equals_4 (n : ℕ) (h1 : 0 < n) (h2 : binomial_sum n = 40) : n = 4 := 
by
  sorry

end n_equals_4_l609_609735


namespace perimeter_ABFCDE_l609_609503

-- Conditions
def square_perimeter (ABCD : ℝ) : ℝ := 4 * ABCD
def equilateral_triangle_perimeter (BFC : ℝ) : ℝ := 3 * BFC

-- Define the side length of the square and equilateral triangle
noncomputable def side_length_square := 10
noncomputable def side_length_triangle := 10

-- Assertion
theorem perimeter_ABFCDE :
  let a := side_length_square in
  let b := side_length_triangle in
  (2 * a + 2 * b + a + b = 60) :=
by
  sorry

end perimeter_ABFCDE_l609_609503


namespace min_lines_to_cross_all_points_in_grid_l609_609002

theorem min_lines_to_cross_all_points_in_grid : 
  ∃ (n : ℕ), n = 18 ∧ ∀ (grid : ℕ) (centers_marked : Bool), 
  grid = 10 → centers_marked = true → 
  min_lines_to_cross_points grid centers_marked = n := 
sorry

end min_lines_to_cross_all_points_in_grid_l609_609002


namespace prove_expression_l609_609931

def problem_statement : Prop :=
  let c := 0.008 in
  (0.76 * 0.76 * 0.76 - c) / (0.76 * 0.76 + 0.76 * 0.2 + 0.04) = 0.5601

theorem prove_expression : problem_statement :=
by
  sorry

end prove_expression_l609_609931


namespace sin_add_double_alpha_l609_609731

open Real

theorem sin_add_double_alpha (alpha : ℝ) (h : sin (π / 6 - alpha) = 3 / 5) :
  sin (π / 6 + 2 * alpha) = 7 / 25 :=
by
  sorry

end sin_add_double_alpha_l609_609731


namespace calculate_base_length_l609_609941

variable (A b h : ℝ)

def is_parallelogram_base_length (A : ℝ) (b : ℝ) (h : ℝ) : Prop :=
  (A = b * h) ∧ (h = 2 * b)

theorem calculate_base_length (H : is_parallelogram_base_length A b h) : b = 15 := by
  -- H gives us the hypothesis that (A = b * h) and (h = 2 * b)
  have H1 : A = b * h := H.1
  have H2 : h = 2 * b := H.2
  -- Use substitution and algebra to solve for b
  sorry

end calculate_base_length_l609_609941


namespace minimum_phone_calls_l609_609886

theorem minimum_phone_calls (n : ℤ) (h : n > 0) : ∃ k, k = 2 * (n - 1) :=
begin
  use 2 * (n - 1),
  reflexivity,
end

end minimum_phone_calls_l609_609886


namespace investment_doubling_time_l609_609823

theorem investment_doubling_time :
  ∀ (r : ℝ) (initial_investment future_investment : ℝ),
  r = 8 →
  initial_investment = 5000 →
  future_investment = 20000 →
  (future_investment = initial_investment * 2 ^ (70 / r * 2)) →
  70 / r * 2 = 17.5 :=
by
  intros r initial_investment future_investment h_r h_initial h_future h_double
  sorry

end investment_doubling_time_l609_609823


namespace bryan_shelves_l609_609242

theorem bryan_shelves (samples_per_shelf total_samples : ℕ) (h_samples_per_shelf : samples_per_shelf = 65) (h_total_samples : total_samples = 455) : total_samples / samples_per_shelf = 7 :=
by
  rw [h_samples_per_shelf, h_total_samples]
  exact Nat.div_eq_of_lt (by norm_num) 455 65
  sorry

end bryan_shelves_l609_609242


namespace num_proper_subsets_of_set_mult_l609_609667

-- Definition of the operation *
def set_mult (A B : Set ℤ) : Set ℤ :=
  {x | ∃ a ∈ A, ∃ b ∈ B, x = a * b}

-- Given sets A and B
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {1, 2}

-- The main theorem to be proved
theorem num_proper_subsets_of_set_mult : 
  let AB := set_mult A B in AB.Card = 5 → (2 ^ 5 - 1 = 31) := 
by
  sorry

end num_proper_subsets_of_set_mult_l609_609667


namespace berries_from_fifth_bush_l609_609447

noncomputable def berries_from_bush (bush_number : ℕ) : ℕ :=
  match bush_number with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 4 => 12
  | _ => berries_from_bush (bush_number - 1) + (bush_number - 1 + bush_number - 3)

theorem berries_from_fifth_bush : berries_from_bush 5 = 19 :=
by
  sorry

end berries_from_fifth_bush_l609_609447


namespace polynomial_division_quotient_l609_609293

theorem polynomial_division_quotient :
  ∀ (x : ℝ), (x^5 - 21*x^3 + 8*x^2 - 17*x + 12) / (x - 3) = (x^4 + 3*x^3 - 12*x^2 - 28*x - 101) :=
by
  sorry

end polynomial_division_quotient_l609_609293


namespace ice_cream_flavors_l609_609361

open Nat

theorem ice_cream_flavors:
  let basic_flavors := 4 in
  let scoops := 5 in
  (basic_flavors + scoops - 1).choose (basic_flavors - 1) = 56 := 
by
  sorry

end ice_cream_flavors_l609_609361


namespace airplane_cost_correct_l609_609221

-- Define the conditions
def initial_amount : ℝ := 5.00
def change_received : ℝ := 0.72

-- Define the cost calculation
def airplane_cost (initial : ℝ) (change : ℝ) : ℝ := initial - change

-- Prove that the airplane cost is $4.28 given the conditions
theorem airplane_cost_correct : airplane_cost initial_amount change_received = 4.28 :=
by
  -- The actual proof goes here
  sorry

end airplane_cost_correct_l609_609221


namespace complex_sum_zero_l609_609870

noncomputable def complexSum {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^(15) + ω^(18) + ω^(21) + ω^(24) + ω^(27) + ω^(30) +
  ω^(33) + ω^(36) + ω^(39) + ω^(42) + ω^(45)

theorem complex_sum_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : complexSum h1 h2 = 0 :=
by
  sorry

end complex_sum_zero_l609_609870


namespace common_point_circumcircles_l609_609862

-- Definitions for geometric entities and conditions.
variables {A B C D E F S T : Type} 
  [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group D] 
  [add_comm_group E] [add_comm_group F] [add_comm_group S] [add_comm_group T]
  [module ℝ A] [module ℝ B] [module ℝ C] [module ℝ D] [module ℝ E] [module ℝ F] [module ℝ S] [module ℝ T]

-- Suppose % details the position related conditions in terms of lengths ratios
variable (h_ratio : ∀ {E F : Type} [add_comm_group E] [module ℝ E] [add_comm_group F] [module ℝ F], 
  ∃ (AE ED BF FC : ℝ), AE / ED = BF / FC)

-- Intersection points
variable (h_intersections : ∀ {E F S T : Type} [add_comm_group E] [module ℝ E] [add_comm_group F] [module ℝ F]
  [add_comm_group S] [module ℝ S] [add_comm_group T] [module ℝ T],
  ∃ (FE BA CD : set (ℝ × ℝ)), ∃ (S T : ℝ × ℝ), FE.E ∩ BA.A = S ∧ FE.F ∩ CD.D = T)

-- The theorem statement for proving concurrency of the circumcircles.
theorem common_point_circumcircles 
(A B C D E F S T M : Type) 
  [add_comm_group A] [module ℝ A] [add_comm_group B] [module ℝ B]
  [add_comm_group C] [module ℝ C] [add_comm_group D] [module ℝ D]
  [add_comm_group E] [module ℝ E] [add_comm_group F] [module ℝ F]
  [add_comm_group S] [module ℝ S] [add_comm_group T] [module ℝ T]
  [add_comm_group M] [module ℝ M]
  (h1 : h_ratio E F)
  (h2 : h_intersections E F S T)
  : (exists M : Type, (circumcircle S A E).contains M ∧ (circumcircle S B F).contains M ∧ (circumcircle T C F).contains M ∧ (circumcircle T D E).contains M) :=
sorry

end common_point_circumcircles_l609_609862


namespace cos_two_times_angle_BPC_l609_609459

theorem cos_two_times_angle_BPC 
  (A B C D P : Point)
  (h_eq: dist A B = dist B C ∧ dist B C = dist C D)
  (APB_angle: ∠APB = α)
  (CPD_angle: ∠CPD = β)
  (cos_APB: cos α = 3/5)
  (cos_CPD: cos β = 12/13) :
  cos (2 * ∠BPC) = -7/25 := 
  sorry

end cos_two_times_angle_BPC_l609_609459


namespace jerome_gave_to_meg_l609_609028

theorem jerome_gave_to_meg (init_money half_money given_away meg bianca : ℝ) 
    (h1 : half_money = 43) 
    (h2 : init_money = 2 * half_money) 
    (h3 : 54 = init_money - given_away)
    (h4 : given_away = meg + bianca)
    (h5 : bianca = 3 * meg) : 
    meg = 8 :=
by
  sorry

end jerome_gave_to_meg_l609_609028


namespace factorize_xy_squared_minus_x_l609_609701

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609701


namespace proper_subsets_of_A_l609_609895

open Set

namespace ProofProblem

def universal_set : Set ℕ := {0, 1, 2, 3}
def complement_of_A : Set ℕ := {2}
def A : Set ℕ := universal_set \ complement_of_A

theorem proper_subsets_of_A :
  (A = {0, 1, 3}) →
  (complement_of_A = universal_set \ A) →
  ∃ n, n = 2^Fintype.card A - 1 ∧ n = 7 :=
by {
  intros hA hComplementA,
  existsi (2 ^ Fintype.card A - 1),
  split,
  {
    rw [←hA, Fintype.card, Fintype.card, Finset.card],
    simp,
    norm_num,
  },
  { 
    norm_num,
  }
}

end ProofProblem

end proper_subsets_of_A_l609_609895


namespace Vanya_correct_answers_l609_609498

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l609_609498


namespace vanya_correct_answers_l609_609490

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609490


namespace lottery_probability_exactly_one_common_l609_609393

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609393


namespace pascal_fifth_number_l609_609022

theorem pascal_fifth_number (k n : ℕ) (h_k : k = 4) (h_n : n = 14) : binomial n k = 1001 :=
by
  simp [binomial, h_k, h_n]
  sorry

end pascal_fifth_number_l609_609022


namespace teachers_with_neither_condition_l609_609215

theorem teachers_with_neither_condition (total_teachers : ℕ) (total_high_bp : ℕ) (total_heart_trouble : ℕ) (both_conditions : ℕ) :
  total_teachers = 150 →
  total_high_bp = 90 →
  total_heart_trouble = 60 →
  both_conditions = 30 →
  let only_high_bp := total_high_bp - both_conditions in
  let only_heart_trouble := total_heart_trouble - both_conditions in
  let either_or_both := only_high_bp + only_heart_trouble + both_conditions in
  let neither := total_teachers - either_or_both in
  (neither * 100) / total_teachers = 20 :=
begin
  intros,
  sorry
end

end teachers_with_neither_condition_l609_609215


namespace quadratic_solve_l609_609083

theorem quadratic_solve (x : ℝ) : (x + 4)^2 = 5 * (x + 4) → x = -4 ∨ x = 1 :=
by sorry

end quadratic_solve_l609_609083


namespace combined_population_port_perry_lazy_harbor_l609_609955

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end combined_population_port_perry_lazy_harbor_l609_609955


namespace coefficient_third_term_expansion_l609_609289

theorem coefficient_third_term_expansion :
  ∀ x : ℝ, coefficient (expansion (1 - x) (expansion (1 + 2 * x) 5) 2) = 30 :=
by 
  sorry

end coefficient_third_term_expansion_l609_609289


namespace other_root_l609_609910

theorem other_root (m : ℝ) :
  ∃ r, (r = -7 / 3) ∧ (3 * 1 ^ 2 + m * 1 - 7 = 0) := 
begin
  use -7 / 3,
  split,
  { refl },
  { linarith }
end

end other_root_l609_609910


namespace f_strictly_decreasing_intervals_f_max_min_on_interval_l609_609796

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 6 * x^2 - 9 * x + 3

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := -3 * x^2 - 12 * x - 9

-- Statement for part (I)
theorem f_strictly_decreasing_intervals :
  (∀ x : ℝ, x < -3 → f_deriv x < 0) ∧ (∀ x : ℝ, x > -1 → f_deriv x < 0) := by
  sorry

-- Statement for part (II)
theorem f_max_min_on_interval :
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≥ -47) :=
  sorry

end f_strictly_decreasing_intervals_f_max_min_on_interval_l609_609796


namespace length_DH_correct_l609_609010

/- Definitions for the given problem -/
structure Triangle :=
  (A B C : Point)
  (AB BC CA : ℝ)
  (equilateral : AB = BC ∧ BC = CA)

structure PointOnSegment (P Q R : Point) :=
  (onSegment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = P + t • (Q - P))

def ProblemConditions (A B C D E F G H : Point) : Prop :=
  let TriangleABC := Triangle A B C 2 2 2 (eq.trans rfl (eq.refl 2)) (eq.trans rfl (eq.refl 2)) in
  let AD := 0.5 in
  let DE := 1 in
  let DF := 0.5 in
  let FG := 1 in
  let FC := 0.5 in
  TriangleABC.R ∧
  PointOnSegment A C E ∧ PointOnSegment A C G ∧ 
  PointOnSegment A B D ∧ PointOnSegment A B F ∧ 
  (DE = FG ∧ DE = 1) ∧
  ParallelLines (line_through D E) (line_through B C) ∧
  ParallelLines (line_through F G) (line_through B C) ∧
  ParallelLines (line_through D H) (line_through A C)

def length_DH (A B C D E F G H : Point) (h : ProblemConditions A B C D E F G H) : ℝ :=
  1

theorem length_DH_correct (A B C D E F G H : Point) (h : ProblemConditions A B C D E F G H) :
  length_DH A B C D E F G H h = 1 :=
sorry

end length_DH_correct_l609_609010


namespace find_valid_polynomial_l609_609298

def largest_prime_factor (m : ℤ) : ℤ :=
  if m = 0 then 0 else
  if m = 1 ∨ m = -1 then 1 else
  (List.reverse (m.factorization.support.toList)).head

def sequence_is_bounded_above (f : ℤ → ℤ) : Prop :=
  ∃ M, ∀ n, largest_prime_factor (f (n * n)) - 2 * n ≤ M

def polynomial_has_integer_coefficients (f : ℤ → ℤ) : Prop :=
  ∀ n, f n ∈ ℤ

def polynomial_nonzero (f : ℤ → ℤ) : Prop :=
  ∀ n, f (n * n) ≠ 0

noncomputable def is_valid_polynomial (a : List ℤ) (c : ℤ) (f : ℤ → ℤ) : Prop :=
  polynomial_has_integer_coefficients f ∧ polynomial_nonzero f ∧
  (∀ x, f x = c * (List.prod (a.map (λ ai, 4 * x - ai^2))) ∧ (∀ ai ∈ a, ¬ ∃ n, 2 * n = ai))

theorem find_valid_polynomial :
  ∃ f : ℤ → ℤ,
    polynomial_has_integer_coefficients f ∧
    polynomial_nonzero f ∧
    (∃ a : List ℤ, ∃ c : ℤ, c ≠ 0 ∧ (∀ ai ∈ a, ¬ ∃ n : ℤ, 2 * n = ai) ∧ is_valid_polynomial a c f) ∧
    sequence_is_bounded_above f :=
sorry

end find_valid_polynomial_l609_609298


namespace total_number_of_coins_l609_609179

theorem total_number_of_coins (x : ℕ) :
  5 * x + 10 * x + 25 * x = 120 → 3 * x = 9 :=
by
  intro h
  sorry

end total_number_of_coins_l609_609179


namespace sum_cubes_div_product_eq_three_l609_609439

-- Given that x, y, z are non-zero real numbers and x + y + z = 3,
-- we need to prove that the possible value of (x^3 + y^3 + z^3) / xyz is 3.

theorem sum_cubes_div_product_eq_three 
  (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hxyz_sum : x + y + z = 3) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end sum_cubes_div_product_eq_three_l609_609439


namespace sum_11_terms_sequence_l609_609742

def a_n (n : ℕ) : ℤ := -2 * n + 1

def S_n (n : ℕ) : ℤ := -n^2

def sum_first_11_terms : ℤ :=
  (Finset.range 11).sum (λ i, - (i + 1))

theorem sum_11_terms_sequence :
  sum_first_11_terms = -66 := by
  sorry

end sum_11_terms_sequence_l609_609742


namespace largest_prime_factor_of_sum_powers_l609_609269

def sum_of_powers (n : ℕ) (p : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), k ^ p

theorem largest_prime_factor_of_sum_powers :
  (nat.factors (sum_of_powers 11 5)).max = some 263 :=
by
  sorry

end largest_prime_factor_of_sum_powers_l609_609269


namespace bank_card_payment_technology_order_l609_609980

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end bank_card_payment_technology_order_l609_609980


namespace factorize_xy2_minus_x_l609_609679

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609679


namespace cos_identity_l609_609849

-- Define the triangle with given angle relationships.
variables {A B C : ℝ}
hypothesis (h1 : A = 3 * B)
hypothesis (h2 : A = 9 * C)
hypothesis (h3 : A + B + C = Real.pi)

-- State the main theorem to prove.
theorem cos_identity : 
  (Real.cos A * Real.cos B + Real.cos B * Real.cos C + Real.cos C * Real.cos A) = -1 / 4 := 
by 
  sorry

end cos_identity_l609_609849


namespace batsman_average_l609_609157

variable (A : ℕ) -- The batsman's average before the 17th innings
variable (R : ℕ) -- The runs scored in the 17th innings
variable (I : ℕ) -- The increase in average after the 17th innings
variable (N : ℕ) -- The number of innings

theorem batsman_average (A R I N : ℕ) 
    (hR : R = 85) 
    (hI : I = 3) 
    (hN : N = 16) 
    (increase_condition : A + N * I = 85 + 17 * (A + I)) :
    (A + 3) = 37 := 
by
    -- definitions
    have h1 : R = 85 := by exact hR
    have h2 : I = 3 := by exact hI
    have h3 : N = 16 := by exact hN
    have h4 : increase_condition := by exact increase_condition
    sorry

end batsman_average_l609_609157


namespace units_digit_sum_eq_seven_l609_609584

theorem units_digit_sum_eq_seven : 
  let units_digit (n : ℕ) := n % 10
  in units_digit (734^99) + units_digit (347^83) = 7 :=
by 
  let units_digit := (fun n : ℕ => n % 10)
  have h1 : units_digit (734 ^ 99) = units_digit (4 ^ 99),
    from congr_arg units_digit (Nat.pow_mod_right 734 99 10),
  have h2 : units_digit (347 ^ 83) = units_digit (7 ^ 83),
    from congr_arg units_digit (Nat.pow_mod_right 347 83 10),
  have hu4 : ∀ k, units_digit (4^k) = if k % 2 = 0 then 6 else 4,
  { intro k,
    have : [4, 6] = List.map units_digit [4, 16],
    { norm_num, simp [units_digit] },
    cases Nat.mod_two_eq_zero_or_one k,
    { rw [this],
      simp,
      exact h },
    { rw [this],
      simp,
      exact h }},
  have hu7 : ∀ k, units_digit (7^k) = [7, 9, 3, 1].nthLe (k % 4) sorry,
  { intro k,
    have : [7, 9, 3, 1] = List.map units_digit [7, 49, 343, 2401],
    { norm_num, simp [units_digit] },
    rw [this] },
  have u4_mod := hu4 99,
  have u7_mod := hu7 83,
  specialize hu4 99,
  specialize hu7 83,
  simp [units_digit, h1, h2, if_pos (Nat.mod_two_ne_one 99)] at u4_mod,
  simp [units_digit, h1, h2, Nat.mod_eq_of_lt (by norm_num : 83 % 4 < 4)] at u7_mod,
  rw u4_mod,
  rw u7_mod,
  norm_num,
  apply nat_mod_eq_zero_or_one
  sorry

end units_digit_sum_eq_seven_l609_609584


namespace one_minus_repeating_eight_l609_609285

-- Define the repeating decimal
def repeating_eight : Real := 0.8888888888 -- repeating of 8

-- Define the repeating decimal as a fraction
def repeating_eight_as_fraction : Real := 8 / 9

-- The proof statement
theorem one_minus_repeating_eight : 1 - repeating_eight = 1 / 9 := by
  -- Since proof is not required, we use sorry
  sorry

end one_minus_repeating_eight_l609_609285


namespace power_of_x_is_one_l609_609548

-- The problem setup, defining the existence of distinct primes and conditions on exponents
theorem power_of_x_is_one (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (a b c : ℕ) (h_divisors : (a + 1) * (b + 1) * (c + 1) = 12) :
  a = 1 :=
sorry

end power_of_x_is_one_l609_609548


namespace minimum_framing_feet_l609_609233

theorem minimum_framing_feet (original_length : ℕ) (original_width : ℕ) (scale_factor : ℕ) (border_width : ℕ) (foot_conversion : ℕ) :
  original_length = 8 →
  original_width = 10 →
  scale_factor = 4 →
  border_width = 3 →
  foot_conversion = 12 →
  let new_length := original_length * scale_factor,
      new_width := original_width * scale_factor,
      bordered_length := new_length + 2 * border_width,
      bordered_width := new_width + 2 * border_width,
      perimeter := 2 * (bordered_length + bordered_width),
      feet_needed := perimeter / foot_conversion in
  feet_needed = 14 :=
by
  intros h1 h2 h3 h4 h5
  let new_length := original_length * scale_factor
  let new_width := original_width * scale_factor
  let bordered_length := new_length + 2 * border_width
  let bordered_width := new_width + 2 * border_width
  let perimeter := 2 * (bordered_length + bordered_width)
  let feet_needed := perimeter / foot_conversion
  have h6 : new_length = 32, sorry
  have h7 : new_width = 40, sorry
  have h8 : bordered_length = 38, sorry
  have h9 : bordered_width = 46, sorry
  have h10 : perimeter = 168, sorry
  have h11 : feet_needed = 14, sorry
  exact h11

end minimum_framing_feet_l609_609233


namespace ones_digit_of_first_in_sequence_l609_609297

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
  
def in_arithmetic_sequence (a d : ℕ) (n : ℕ) : Prop :=
  ∃ k, a = k * d + n

theorem ones_digit_of_first_in_sequence {p q r s t : ℕ}
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (ht : is_prime t)
  (hseq : in_arithmetic_sequence p 10 q ∧ 
          in_arithmetic_sequence q 10 r ∧
          in_arithmetic_sequence r 10 s ∧
          in_arithmetic_sequence s 10 t)
  (hincr : p < q ∧ q < r ∧ r < s ∧ s < t)
  (hstart : p > 5) :
  p % 10 = 1 := sorry

end ones_digit_of_first_in_sequence_l609_609297


namespace people_sitting_on_benches_l609_609608

-- Conditions
def num_benches : ℕ := 50
def capacity_per_bench : ℕ := 4
def available_spaces : ℕ := 120

-- Define the statement to prove
theorem people_sitting_on_benches :
  let total_capacity := num_benches * capacity_per_bench in
  let people_sitting := total_capacity - available_spaces in
  people_sitting = 80 := 
by
  sorry

end people_sitting_on_benches_l609_609608


namespace power_expression_evaluation_l609_609589

theorem power_expression_evaluation :
  (1 / 2) ^ 2016 * (-2) ^ 2017 * (-1) ^ 2017 = 2 := 
by
  sorry

end power_expression_evaluation_l609_609589


namespace wanda_walks_days_per_week_l609_609139

theorem wanda_walks_days_per_week 
  (daily_distance : ℝ) (weekly_distance : ℝ) (weeks : ℕ) (total_distance : ℝ) 
  (h_daily_walk: daily_distance = 2) 
  (h_total_walk: total_distance = 40) 
  (h_weeks: weeks = 4) : 
  ∃ d : ℕ, (d * daily_distance * weeks = total_distance) ∧ (d = 5) := 
by 
  sorry

end wanda_walks_days_per_week_l609_609139


namespace angle_complement_30_l609_609744

def complement_angle (x : ℝ) : ℝ := 90 - x

theorem angle_complement_30 (x : ℝ) (h : x = complement_angle x - 30) : x = 30 :=
by
  sorry

end angle_complement_30_l609_609744


namespace relationship_among_m_n_p_l609_609734

noncomputable def m : ℝ := Real.log 5 / Real.log 0.5
noncomputable def n : ℝ := 5.1 ^ (-3)
noncomputable def p : ℝ := 5.1 ^ 0.3

theorem relationship_among_m_n_p :
  m < n ∧ n < p :=
by
  let m := Real.log 5 / Real.log 0.5
  let n := 5.1 ^ (-3)
  let p := 5.1 ^ 0.3
  sorry

end relationship_among_m_n_p_l609_609734


namespace quadratic_has_real_roots_l609_609759

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 - 4 * x - 2 * k + 8 = 0) ->
  k ≥ 2 :=
by
  sorry

end quadratic_has_real_roots_l609_609759


namespace fish_in_pond_l609_609408

-- Conditions
variable (N : ℕ)
variable (h₁ : 80 * 80 = 2 * N)

-- Theorem to prove 
theorem fish_in_pond (h₁ : 80 * 80 = 2 * N) : N = 3200 := 
by 
  sorry

end fish_in_pond_l609_609408


namespace function_minimum_value_in_interval_l609_609102

theorem function_minimum_value_in_interval (a : ℝ) :
  (∃ x ∈ set.Iio (1 : ℝ), ∀ y ∈ set.Iio (1 : ℝ), f y ≥ f x) →
  a < 1 :=
by
  let f := λ x : ℝ, x^2 - 2*a*x + a
  intro h
  sorry

end function_minimum_value_in_interval_l609_609102


namespace seq_geometric_and_k_range_l609_609059

-- Definitions and Conditions
def seq (b : ℕ → ℝ) : Prop :=
  b 1 = 7 / 2 ∧ ∀ n : ℕ, b (n + 1) = (1 / 2) * b n + 1 / 4

def T (b : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum b

theorem seq_geometric_and_k_range (b : ℕ → ℝ) (k : ℝ) :
  seq b →
  (∃ a r, ∀ n, b (n + 1) - 1 / 2 = r * (b n - 1 / 2) ∧ a = b 1 - 1 / 2 ∧ r = 1 / 2) →
  (∀ n : ℕ, b n = 3 * (1 / 2)^(n-1) + 1 / 2) →
  (∀ n : ℕ, ∀ k : ℝ, (2 * T b n + 3 * 2^(2 * n - 1) - 10) / k ≤ n^2 + 4 * n + 5 → k ≥ 3 / 10) :=
by
  intros
  sorry

end seq_geometric_and_k_range_l609_609059


namespace eq_sum_of_factorial_fractions_l609_609287

theorem eq_sum_of_factorial_fractions (b2 b3 b5 b6 b7 b8 : ℤ)
  (h2 : 0 ≤ b2 ∧ b2 < 2)
  (h3 : 0 ≤ b3 ∧ b3 < 3)
  (h5 : 0 ≤ b5 ∧ b5 < 5)
  (h6 : 0 ≤ b6 ∧ b6 < 6)
  (h7 : 0 ≤ b7 ∧ b7 < 7)
  (h8 : 0 ≤ b8 ∧ b8 < 8)
  (h_eq : (3 / 8 : ℚ) = (b2 / (2 * 1) + b3 / (3 * 2 * 1) + b5 / (5 * 4 * 3 * 2 * 1) +
                          b6 / (6 * 5 * 4 * 3 * 2 * 1) + b7 / (7 * 6 * 5 * 4 * 3 * 2 * 1) +
                          b8 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) : ℚ)) :
  b2 + b3 + b5 + b6 + b7 + b8 = 12 :=
by
  sorry

end eq_sum_of_factorial_fractions_l609_609287


namespace expenditure_representation_l609_609653

theorem expenditure_representation
    (income_representation : ℤ)
    (income_is_positive : income_representation = 60)
    (use_of_opposite_signs : ∀ (n : ℤ), n >= 0 ↔ n = 60)
    (representation_criteria : ∀ (x : ℤ), x <= 0 → x = -40) :
  ∀ (expenditure : ℤ), expenditure = 40 → -expenditure = -40 :=
by
  intro expenditure
  intro expenditure_criteria
  rw [neg_eq_neg_one_mul]
  change (-1) * expenditure = (-40)
  apply representation_criteria
  exact le_of_eq rfl
  sorry

end expenditure_representation_l609_609653


namespace initial_velocity_of_athlete_l609_609533

noncomputable def height_function (t : ℝ) : ℝ :=
  -4.9 * t^2 + 6.5 * t + 10

theorem initial_velocity_of_athlete :
  let h := height_function in
  ∀ t : ℝ, ((deriv h) 0 = 6.5) :=
by
  -- Statement only, proof is not required
  sorry

end initial_velocity_of_athlete_l609_609533


namespace time_for_worker_C_l609_609126

theorem time_for_worker_C (time_A time_B time_total : ℝ) (time_A_pos : 0 < time_A) (time_B_pos : 0 < time_B) (time_total_pos : 0 < time_total) 
  (hA : time_A = 12) (hB : time_B = 15) (hTotal : time_total = 6) : 
  (1 / (1 / time_total - 1 / time_A - 1 / time_B) = 60) :=
by 
  sorry

end time_for_worker_C_l609_609126


namespace max_value_and_period_of_f_l609_609952

noncomputable def f (x : ℝ) : ℝ := (Real.cos (4 * x)) * (Real.cos (2 * x)) * (Real.cos x) * (Real.sin x)

theorem max_value_and_period_of_f :
  (∀ x, f x ≤ 1 / 8) ∧ (∃ x, f x = 1 / 8) ∧
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π / 4) :=
by
  -- Maximum value
  split
  . intro x
    simp [f]
    sorry
  . split
    . use 0
      simp [f]
      sorry
    . use π / 4
      split
      . linarith
      . intro x
        simp [f]
        sorry

end max_value_and_period_of_f_l609_609952


namespace inequality_l609_609763

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l609_609763


namespace vanya_correct_answers_l609_609472

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609472


namespace range_of_PA2_PB2_l609_609728

-- Definitions for points and distances
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def dist_squared (P Q : Point2D) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Circle definition
def on_circle (P : Point2D) (radius : ℝ) : Prop :=
  P.x^2 + P.y^2 = radius^2

-- Points A and B
def A : Point2D := ⟨-3, 0⟩
def B : Point2D := ⟨4, 2⟩

-- Calculating the targeted range
theorem range_of_PA2_PB2 (P : Point2D) (h : on_circle P 2) :
  37 - 4 * real.sqrt 5 ≤ dist_squared P A + dist_squared P B ∧
  dist_squared P A + dist_squared P B ≤ 37 + 4 * real.sqrt 5 :=
sorry

end range_of_PA2_PB2_l609_609728


namespace basketball_problem_l609_609228

theorem basketball_problem :
  ∃ x y : ℕ, (3 + x + y = 14) ∧ (3 * 3 + 2 * x + y = 28) ∧ (x = 8) ∧ (y = 3) :=
by
  sorry

end basketball_problem_l609_609228


namespace probability_exactly_one_common_number_l609_609384

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l609_609384


namespace find_length_AC_l609_609024

variable {A B C : Type*} [metric_space A] [add_group A] [add_comm_group B] [module ℝ B]

-- Define the triangle and the given conditions
variables (a b c : B)
variable (𝛼 𝛽 𝛾 : real.angle)

-- The given conditions
def triangle_ABC (a b c : B) (𝛼 𝛽 𝛾 : real.angle) : Prop :=
  metric_space.angle a b c = 𝛾 ∧
  𝛾 = 3 * 𝛼 ∧
  dist b c = 5 ∧
  dist a b = 11

-- Prove the length of AC
theorem find_length_AC (a b c : B) (𝛼 𝛽 𝛾 : real.angle)
  (h : triangle_ABC a b c 𝛼 𝛽 𝛾) : dist a c = 24 * real.sqrt 5 / 5 :=
begin
  sorry
end

end find_length_AC_l609_609024


namespace f_2023_equals_1_l609_609374

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the even function definition

-- Assume the given conditions as explicit assumptions
axiom even_function : ∀ x : ℝ, f(-x) = f(x)
axiom functional_equation : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f(x+2) = f(2-x)
axiom interval_definition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f(x) = x^2

-- The main statement to prove
theorem f_2023_equals_1 : f(2023) = 1 :=
by
  -- Proof to follow
  sorry

end f_2023_equals_1_l609_609374


namespace fish_remaining_correct_l609_609264

variable (initial_guppies : ℕ) (initial_angelfish : ℕ)
          (initial_tiger_sharks : ℕ) (initial_oscar_fish : ℕ)
          (initial_discus_fish : ℕ)

-- Assumptions/Conditions
def initial_conditions :=
  initial_guppies = 225 ∧ initial_angelfish = 175 ∧ initial_tiger_sharks = 200 ∧
  initial_oscar_fish = 140 ∧ initial_discus_fish = 120

def transactions := 
  let guppies_sold := 0.60 * initial_guppies in
  let angelfish_sold := 0.43 * initial_angelfish in
  let tiger_sharks_sold := 0.25 * initial_tiger_sharks in
  let oscar_fish_sold := 0.50 * initial_oscar_fish in
  let discus_kept := (2/3) * initial_discus_fish in
  let discus_increase := 0.35 * discus_kept in
  let remaining_guppies := initial_guppies - guppies_sold in
  let remaining_angelfish := (initial_angelfish - angelfish_sold) + 18 in
  let remaining_tiger_sharks := (initial_tiger_sharks - tiger_sharks_sold) - 12 in
  let remaining_oscar_fish := initial_oscar_fish - oscar_fish_sold in
  let remaining_discus_fish := discus_kept + discus_increase in
  remaining_guppies + remaining_angelfish + remaining_tiger_sharks + remaining_oscar_fish + remaining_discus_fish

theorem fish_remaining_correct :
  initial_conditions →
  transactions initial_guppies initial_angelfish initial_tiger_sharks initial_oscar_fish initial_discus_fish = 524 :=
by
  intros h
  sorry

end fish_remaining_correct_l609_609264


namespace traffic_light_probability_l609_609622

theorem traffic_light_probability : 
  let green := 45
  let yellow := 5
  let red := 40
  let total_cycle := green + yellow + red
  let changes := 3 -- there are three changes from green to yellow, yellow to red, and red to green
  let observation_duration := 4
  in (changes * observation_duration : ℝ) / total_cycle = 2 / 15 := 
by
  sorry

end traffic_light_probability_l609_609622


namespace no_real_roots_iff_no_positive_discriminant_l609_609373

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem no_real_roots_iff_no_positive_discriminant (m : ℝ) 
  (h : discriminant m (-2*(m+2)) (m+5) < 0) : 
  (discriminant (m-5) (-2*(m+2)) m < 0 ∨ discriminant (m-5) (-2*(m+2)) m > 0 ∨ m - 5 = 0) :=
by 
  sorry

end no_real_roots_iff_no_positive_discriminant_l609_609373


namespace johns_elevation_after_descent_l609_609033

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end johns_elevation_after_descent_l609_609033


namespace combinations_three_out_of_seven_l609_609927

theorem combinations_three_out_of_seven : nat.choose 7 3 = 35 :=
by
  sorry

end combinations_three_out_of_seven_l609_609927


namespace problem_lean_l609_609934

theorem problem_lean :
  (86 * 95 * 107) % 20 = 10 :=
sorry

end problem_lean_l609_609934


namespace chain_of_concepts_l609_609850

-- Definition of initial and final concepts
def initialConcept : Prop := ∃ n : ℕ, n = 3

-- Intermediate steps in the chain of concepts
def isNaturalNumber (n : ℕ) : Prop := n ∈ set.univ -- ℕ is the set of all natural numbers
def isInteger (n : ℤ) : Prop := n ∈ set.univ   -- ℤ is the set of all integers
def isRationalNumber (r : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ r = a / b
def isRealNumber (r : ℝ) : Prop := r ∈ set.univ -- ℝ is the set of all real numbers
def isNumber (r : ℝ) : Prop := r ∈ set.univ -- A real number is a number

-- The proposition we need to prove
theorem chain_of_concepts :
  initialConcept →
  ( ∀ n : ℕ, isNaturalNumber n → ( ∃ z : ℤ, isInteger z ∧ ↑n = z) →
  ( ∀ z : ℤ, isInteger z → ( ∃ q : ℚ, isRationalNumber q ∧ ↑z = q) →
  ( ∀ q : ℚ, isRationalNumber q → ( ∃ r : ℝ, isRealNumber r ∧ ↑q = r) →
  ( ∀ r : ℝ, isRealNumber r → isNumber r ) )) ) ) :=
by
  sorry

end chain_of_concepts_l609_609850


namespace circle_intersection_l609_609963

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_intersection (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ (∃ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0)) ↔ 4 < m ∧ m < 144 :=
by
  have h1 : distance (0, 0) (3, -4) = 5 := by sorry
  have h2 : ∀ m, |7 - Real.sqrt m| < 5 ↔ 4 < m ∧ m < 144 := by sorry
  exact sorry

end circle_intersection_l609_609963


namespace factory_profit_starts_at_year_three_maximum_annual_average_net_profit_at_six_maximum_total_net_profit_at_ten_option_one_more_cost_effective_l609_609635

noncomputable def total_expenditure (n : ℕ) : ℕ :=
2 * n^2 + 10 * n

noncomputable def total_net_profit (n : ℕ) : ℤ :=
500000 * n - total_expenditure n - 720000

def annual_average_net_profit (n : ℕ) : ℚ :=
(total_net_profit n : ℚ) / n

theorem factory_profit_starts_at_year_three :
  ∀ {n : ℕ}, n ≥ 3 → total_net_profit n > 0 := 
by
  sorry

theorem maximum_annual_average_net_profit_at_six :
  ∀ {n : ℕ}, n = 6 → annual_average_net_profit n = 16 := 
by
  sorry

theorem maximum_total_net_profit_at_ten :
  ∀ {n : ℕ}, n = 10 → total_net_profit n = 128 := 
by
  sorry

theorem option_one_more_cost_effective :
  ∀ {n : ℕ}, (annual_average_net_profit 6 * 6 + 480000) > (total_net_profit 10 + 100000) :=
by
  sorry

end factory_profit_starts_at_year_three_maximum_annual_average_net_profit_at_six_maximum_total_net_profit_at_ten_option_one_more_cost_effective_l609_609635


namespace vanya_correct_answers_l609_609469

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609469


namespace one_of_five_wins_l609_609007

-- Define the probabilities of each car
def prob_A := 1 / 4
def prob_B := 1 / 8
def prob_C := 1 / 12
def prob_D := 1 / 20
def prob_E := 1 / 30

-- Define the expected result
def total_prob := 65 / 120

-- Prove that the sum of individual probabilities equals the total probability
theorem one_of_five_wins : prob_A + prob_B + prob_C + prob_D + prob_E = total_prob := 
by
  -- Convert each probability to the same denominator (120)
  have hA : prob_A = 30 / 120 := by sorry
  have hB : prob_B = 15 / 120 := by sorry
  have hC : prob_C = 10 / 120 := by sorry
  have hD : prob_D = 6 / 120 := by sorry
  have hE : prob_E = 4 / 120 := by sorry
  -- Add the probabilities
  calc prob_A + prob_B + prob_C + prob_D + prob_E
       = 30 / 120 + 15 / 120 + 10 / 120 + 6 / 120 + 4 / 120 : by rw [hA, hB, hC, hD, hE]
   ... =  65 / 120 : by norm_num

end one_of_five_wins_l609_609007


namespace find_scalar_s_l609_609295

open Real

def u (s : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 4 * s, -2 - 6 * s, 1 + 2 * s)

def b : ℝ × ℝ × ℝ :=
  (1, 5, -3)

def direction : ℝ × ℝ × ℝ :=
  (4, -6, 2)

-- Define the dot product of two 3-dimensional vectors
def dot (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- The proof problem to solve
theorem find_scalar_s : ∃ s : ℝ, (dot (u s - b) direction) = 0 :=
  sorry

end find_scalar_s_l609_609295


namespace ratio_jerky_l609_609027

/-
  Given conditions:
  1. Janette camps for 5 days.
  2. She has an initial 40 pieces of beef jerky.
  3. She eats 4 pieces of beef jerky per day.
  4. She will have 10 pieces of beef jerky left after giving some to her brother.

  Prove that the ratio of the pieces of beef jerky she gives to her brother 
  to the remaining pieces is 1:1.
-/

theorem ratio_jerky (days : ℕ) (initial_jerky : ℕ) (jerky_per_day : ℕ) (jerky_left_after_trip : ℕ)
  (h1 : days = 5) (h2 : initial_jerky = 40) (h3 : jerky_per_day = 4) (h4 : jerky_left_after_trip = 10) :
  (initial_jerky - days * jerky_per_day - jerky_left_after_trip) = jerky_left_after_trip :=
by
  sorry

end ratio_jerky_l609_609027


namespace simplify_expression_l609_609247

variable (a b : ℝ)

theorem simplify_expression : (a + b) * (3 * a - b) - b * (a - b) = 3 * a ^ 2 + a * b :=
by
  sorry

end simplify_expression_l609_609247


namespace sum_of_repeating_decimals_l609_609281

-- Definitions of the repeating decimals as fractions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 99
def z : ℚ := 3 / 999

-- Theorem stating the sum of these fractions is equal to the expected result
theorem sum_of_repeating_decimals : x + y + z = 164 / 1221 := 
  sorry

end sum_of_repeating_decimals_l609_609281


namespace game_cost_l609_609574

theorem game_cost
    (initial_amount : ℕ)
    (cost_per_toy : ℕ)
    (num_toys : ℕ)
    (remaining_amount := initial_amount - cost_per_toy * num_toys)
    (cost_of_game := initial_amount - remaining_amount)
    (h1 : initial_amount = 57)
    (h2 : cost_per_toy = 6)
    (h3 : num_toys = 5) :
  cost_of_game = 27 :=
by
  sorry

end game_cost_l609_609574


namespace total_people_surveyed_l609_609416

theorem total_people_surveyed (x y : ℝ) (h1 : 0.536 * x = 30) (h2 : 0.794 * y = x) : y = 71 :=
by
  sorry

end total_people_surveyed_l609_609416


namespace smallest_angle_between_l609_609869

noncomputable def angle_between (a b c : ℝ) := real.arccos (-1 / 4)

theorem smallest_angle_between {a b c : ℝ^3} 
  (ha : ‖a‖ = 2) 
  (hb : ‖b‖ = 3)
  (hc : ‖c‖ = 4)
  (habc : a × (b × c) + 2 • b = 0) : 
  angle_between a b c = real.arccos (-1 / 4) :=
sorry

end smallest_angle_between_l609_609869


namespace distance_between_trees_l609_609226

theorem distance_between_trees
    (yard_length : ℕ)
    (num_trees : ℕ)
    (yard_length_eq : yard_length = 434)
    (num_trees_eq : num_trees = 32) :
    (yard_length / (num_trees - 1) = 14) :=
by
  rw [yard_length_eq, num_trees_eq]
  norm_num
  sorry

end distance_between_trees_l609_609226


namespace correct_calculation_l609_609152

theorem correct_calculation :
  (√2 + √3 ≠ √5) ∧ 
  (3 * √2 - √2 ≠ 3) ∧ 
  (√6 / 2 ≠ √3) ∧ 
  (√((-4) * (-2)) = √8 ∧ √8 = 2 * √2) :=
by
  -- will provide details here when proving each calculation
  sorry

end correct_calculation_l609_609152


namespace intersection_A_Z_l609_609803

def setA : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem intersection_A_Z : setA ∩ (Set.of (0 : ℤ) ∪ Set.of (1 : ℤ) ∪ Set.of (2 : ℤ)) = {0, 1, 2} :=
by
  sorry

end intersection_A_Z_l609_609803


namespace factorization_l609_609696

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609696


namespace parallel_line_at_distance_l609_609606

theorem parallel_line_at_distance 
  (A B C₁ C₂ : ℝ) 
  (hA : A = 4) 
  (hB : B = 3) 
  (hC₁ : C₁ = -5) 
  (distance : ℝ) 
  (h_distance : distance = 3) :
  (C₂ = 10 ∨ C₂ = -20) →
  ∃ C₂, 4 * x + 3 * y + C₂ = 0 ∧ 
  ((abs (C₂ + 5)) / sqrt (4^2 + 3^2) = 3) := 
by {
  sorry,
}

end parallel_line_at_distance_l609_609606


namespace linear_eq_conditions_l609_609812

theorem linear_eq_conditions (m : ℤ) (h : abs m = 1) (h₂ : m + 1 ≠ 0) : m = 1 :=
by
  sorry

end linear_eq_conditions_l609_609812


namespace probability_one_common_number_approx_l609_609392

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l609_609392


namespace total_down_payment_l609_609592

theorem total_down_payment: 
  let people := 3
  let single_payment := 1166.67
  let payment_rounded := 1167
  in payment_rounded * people = 3501 := 
by
  sorry

end total_down_payment_l609_609592


namespace radium_decay_solution_l609_609098

noncomputable def radium_decay (R₀ : ℝ) : ℝ → ℝ := 
  λ t, R₀ * Real.exp (-0.00043 * t)

theorem radium_decay_solution (R R₀ : ℝ) (t : ℝ) (hR : R = R₀ * Real.exp (-0.00043 * t)) :
  (∀ t, (deriv (λ t, radium_decay R₀ t)) t = -0.00043 * radium_decay R₀ t) ∧
  radium_decay R₀ 1600 = R₀ / 2 :=
  by
    sorry

end radium_decay_solution_l609_609098


namespace find_digit_l609_609143

theorem find_digit (a : ℕ) (n1 n2 n3 : ℕ) (h1 : n1 = a * 1000) (h2 : n2 = a * 1000 + 998) (h3 : n3 = a * 1000 + 999) (h4 : n1 + n2 + n3 = 22997) :
  a = 7 :=
by
  sorry

end find_digit_l609_609143


namespace Matilda_age_is_35_l609_609854

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end Matilda_age_is_35_l609_609854


namespace compare_values_l609_609776

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l609_609776


namespace f_is_periodic_l609_609736

-- Given conditions: x is a real number
variable (x : ℝ)

-- Define the floor function as the greatest integer less than or equal to x
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
def f (x : ℝ) : ℝ := x - floor x

-- Lean 4 statement to show that f(x) is periodic
theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x := by sorry

end f_is_periodic_l609_609736


namespace min_sum_of_chord_lengths_l609_609353

noncomputable def parabola := {p : ℝ × ℝ | p.snd ^ 2 = 2 * p.fst}

def focus := (1, 0)  -- Since focus of the parabola y^2 = 2x is at (1, 0)

def line_through_focus (k : ℝ) := {p : ℝ × ℝ | p.snd = k * (p.fst - 1)}

def is_point_on_parabola (p : ℝ × ℝ) := p ∈ parabola

def intersection_points (k : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | is_point_on_parabola p ∧ p ∈ line_through_focus k}

theorem min_sum_of_chord_lengths (k1 k2 : ℝ)
  (h : k1^2 + k2^2 = 2)
  (A B : ℝ × ℝ)
  (D E : ℝ × ℝ)
  (hAB : A ∈ intersection_points k1 ∧ B ∈ intersection_points k1)
  (hDE : D ∈ intersection_points k2 ∧ E ∈ intersection_points k2) :
  |(A.fst - B.fst, A.snd - B.snd).norm + (D.fst - E.fst, D.snd - E.snd).norm| ≥ 8 :=
sorry

end min_sum_of_chord_lengths_l609_609353


namespace jackson_spends_on_school_supplies_l609_609026

theorem jackson_spends_on_school_supplies :
  let num_students := 50
  let pens_per_student := 7
  let notebooks_per_student := 5
  let binders_per_student := 3
  let highlighters_per_student := 4
  let folders_per_student := 2
  let cost_pen := 0.70
  let cost_notebook := 1.60
  let cost_binder := 5.10
  let cost_highlighter := 0.90
  let cost_folder := 1.15
  let teacher_discount := 135
  let bulk_discount := 25
  let sales_tax_rate := 0.05
  let total_cost := 
    (num_students * pens_per_student * cost_pen) + 
    (num_students * notebooks_per_student * cost_notebook) + 
    (num_students * binders_per_student * cost_binder) + 
    (num_students * highlighters_per_student * cost_highlighter) + 
    (num_students * folders_per_student * cost_folder)
  let discounted_cost := total_cost - teacher_discount - bulk_discount
  let sales_tax := discounted_cost * sales_tax_rate
  let final_cost := discounted_cost + sales_tax
  final_cost = 1622.25 := by
  sorry

end jackson_spends_on_school_supplies_l609_609026


namespace expression_f_range_a_l609_609442

noncomputable def f (x : ℝ) : ℝ :=
if h : -1 ≤ x ∧ x ≤ 1 then x^3
else if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
else (x-4)^3

theorem expression_f (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) :
  f x =
    if h : 1 ≤ x ∧ x < 3 then -(x-2)^3
    else (x-4)^3 :=
by sorry

theorem range_a (a : ℝ) : 
  (∃ x, f x > a) ↔ a < 1 :=
by sorry

end expression_f_range_a_l609_609442


namespace rational_solutions_k_l609_609723

theorem rational_solutions_k (k : ℕ) (hpos : k > 0) : (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 :=
by
  sorry

end rational_solutions_k_l609_609723


namespace solution_80_percent_needs_12_ounces_l609_609576

theorem solution_80_percent_needs_12_ounces:
  ∀ (x y: ℝ), (x + y = 40) → (0.30 * x + 0.80 * y = 0.45 * 40) → (y = 12) :=
by
  intros x y h1 h2
  sorry

end solution_80_percent_needs_12_ounces_l609_609576


namespace find_x_for_parallel_vectors_l609_609808

theorem find_x_for_parallel_vectors :
  ∃ x : ℝ,
    let a : ℝ × ℝ := (1, 2),
    let b : ℝ × ℝ := (x, 1),
    let u := (a.1 + 2 * b.1, a.2 + 2 * b.2),
    let v := (2 * a.1 - b.1, 2 * a.2 - b.2)
  in (u.1 * v.2 - u.2 * v.1 = 0) ∧ (x = 1 / 2) :=
by
  sorry

end find_x_for_parallel_vectors_l609_609808


namespace inequality_l609_609762

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l609_609762


namespace sin_theta_correct_l609_609056

noncomputable def sin_theta : ℝ :=
  let d := (3, 4, 8) in
  let n := (-4, -8, 5) in
  let dot_product := d.1 * n.1 + d.2 * n.2 + d.3 * n.3 in
  let magnitude_d := Real.sqrt (d.1^2 + d.2^2 + d.3^2) in
  let magnitude_n := Real.sqrt (n.1^2 + n.2^2 + n.3^2) in
  Real.abs dot_product / (magnitude_d * magnitude_n)

theorem sin_theta_correct :
  sin_theta = 4 / Real.sqrt 9355 :=
begin
  -- proof goes here
  sorry
end

end sin_theta_correct_l609_609056


namespace product_of_equal_numbers_l609_609515

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l609_609515


namespace other_root_l609_609909

theorem other_root (m : ℝ) :
  ∃ r, (r = -7 / 3) ∧ (3 * 1 ^ 2 + m * 1 - 7 = 0) := 
begin
  use -7 / 3,
  split,
  { refl },
  { linarith }
end

end other_root_l609_609909


namespace vanya_correct_answers_l609_609463

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l609_609463


namespace clinic_discount_percentage_l609_609993

theorem clinic_discount_percentage : 
  ∀ (normal_cost discount_savings : ℝ),
  normal_cost = 200 →
  discount_savings = 80 →
  (∃ discount_percent : ℝ, 
    let clinic_cost := normal_cost - discount_savings,
        per_visit_cost := clinic_cost / 2,
        discount := normal_cost - per_visit_cost in
    discount_percent = (discount / normal_cost) * 100 ∧ discount_percent = 70) :=
by
  -- We assert the existence of a discount_percentage that satisfies the conditions given
  intros normal_cost discount_savings hnormal_cost hdiscount_savings
  use (70 : ℝ)
  have hclinic_cost : clinic_cost = normal_cost - discount_savings := rfl
  have hper_visit_cost : per_visit_cost = clinic_cost / 2 := rfl
  have hdiscount : discount = normal_cost - per_visit_cost := rfl
  rw [hclinic_cost, hnormal_cost, hdiscount_savings]
  split
  have h : ((200 - 80) / 2) = 60 := sorry
  rw [h]
  have h' : 200 - 60 = 140 := sorry
  rw [h']
  have h'' : (140 / 200) * 100 = 70 := sorry
  exact h''

end clinic_discount_percentage_l609_609993


namespace max_grandchildren_l609_609899

theorem max_grandchildren (children_count : ℕ) (common_gc : ℕ) (special_gc_count : ℕ) : 
  children_count = 8 ∧ common_gc = 8 ∧ special_gc_count = 5 →
  (6 * common_gc + 2 * special_gc_count) = 58 := by
  sorry

end max_grandchildren_l609_609899


namespace factorization_l609_609698

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609698


namespace Matilda_age_is_35_l609_609853

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end Matilda_age_is_35_l609_609853


namespace compare_values_l609_609778

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l609_609778


namespace freddy_talk_time_dad_l609_609303

-- Conditions
def localRate : ℝ := 0.05
def internationalRate : ℝ := 0.25
def talkTimeBrother : ℕ := 31
def totalCost : ℝ := 10.0

-- Goal: Prove the duration of Freddy's local call to his dad is 45 minutes
theorem freddy_talk_time_dad : 
  ∃ (talkTimeDad : ℕ), 
    talkTimeDad = 45 ∧
    totalCost = (talkTimeBrother : ℝ) * internationalRate + (talkTimeDad : ℝ) * localRate := 
by
  sorry

end freddy_talk_time_dad_l609_609303


namespace factorize_expression_l609_609687

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609687


namespace share_of_a_l609_609169

variables (A B C : ℝ)

def conditions :=
  A = (2 / 3) * (B + C) ∧
  B = (2 / 3) * (A + C) ∧
  A + B + C = 700

theorem share_of_a (h : conditions A B C) : A = 280 :=
by { sorry }

end share_of_a_l609_609169


namespace Matilda_correct_age_l609_609856

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end Matilda_correct_age_l609_609856


namespace div_powers_half_div_powers_four_div_powers_three_x_div_powers_x_minus_one_l609_609553
noncomputable theory

-- Statement 1
theorem div_powers_half : 
  ((1 / 2)^5 / (1 / 2)^2 = (1 / 8)) := by
  sorry

-- Statement 2
theorem div_powers_four : 
  (4^3 / 4^5 = (1 / 16)) := by
  sorry

-- Statement 3
theorem div_powers_three_x (x : ℝ) : 
  (3^(x - 1) / 3^(3 * x - 4) = (1 / 27)) → x = 3 := by
  sorry

-- Statement 4
theorem div_powers_x_minus_one (x : ℝ) :
  ((x - 1)^(2 * x + 2) / (x - 1)^(x + 6) = 1) → 
  x = 4 ∨ x = 0 ∨ x = 2 := by
  sorry

end div_powers_half_div_powers_four_div_powers_three_x_div_powers_x_minus_one_l609_609553


namespace length_of_PQ_l609_609837

variables (x y : ℝ)
open_locale big_operators

def isosceles_triangle_area (x : ℝ) : ℝ := (1 / 2) * x^2
def right_angled_triangle_area (x y : ℝ) : ℝ := (1 / 2) * x * y

theorem length_of_PQ (h1 : isosceles_triangle_area x = 120)
                     (h2 : right_angled_triangle_area x y = 80) : 
  x = 4 * Real.sqrt 15 :=
by 
  have h3 : x^2 = 240 := by 
    rw [isosceles_triangle_area, eq_comm] at h1
    apply (eq_div_iff_mul_eq' (by norm_num : (2 : ℝ) ≠ 0)).mp
    simpa using h1
  have h4 : y = (2 * x) / 3 := by
    rw [right_angled_triangle_area, eq_comm] at h2
    have hx : x ≠ 0 := by
      rw [sq_eq_zero_iff, ne.def, not_false_iff]
      exact ne_of_gt (sqrt_pos.mpr h3)
    apply (eq_div_iff_mul_eq' (by norm_num : (2 : ℝ) ≠ 0)).mp
    field_simp [h2, h3, hx]
  have h5 : x = Real.sqrt 240 := sq_eq_sqrt_iff h3.symm
  exact eq_of_mul_eq_mul_left (by norm_num : (4 : ℝ) ≠ 0) (by norm_num [Real.sqrt_mul', h5])

end length_of_PQ_l609_609837


namespace integer_grid_triangle_centroid_l609_609118

theorem integer_grid_triangle_centroid
  (A B C O : ℤ × ℤ)
  (hA : A ∈ ℤ)
  (hB : B ∈ ℤ)
  (hC : C ∈ ℤ)
  (h_no_points_on_sides : ∀ (P: ℤ × ℤ), P ∈ segment A B ∨ P ∈ segment B C ∨ P ∈ segment C A → P = A ∨ P = B ∨ P = C)
  (h_only_one_interior : ∀ (P: ℤ × ℤ), P ∈ interior_triangle A B C → P = O) :
  is_centroid O A B C :=
sorry

end integer_grid_triangle_centroid_l609_609118


namespace solve_for_x_cube_root_l609_609930

theorem solve_for_x_cube_root (x : ℝ) (h : (∛(5 - 2 / x) = -3)) : x = 1 / 16 :=
by
  sorry

end solve_for_x_cube_root_l609_609930


namespace gross_profit_percentage_l609_609965

theorem gross_profit_percentage (sales_price gross_profit : ℝ) (h_sales_price : sales_price = 91) (h_gross_profit : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 :=
by
  sorry

end gross_profit_percentage_l609_609965


namespace largest_possible_value_of_e_is_24_l609_609874

open Real

noncomputable def largest_e : ℝ := 14 - 8 * sqrt 2

theorem largest_possible_value_of_e_is_24 :
  ∃ (u v w : ℕ), (w > 0 ∧ ∀ p : ℕ, p.prime → p^2 ∣ w → false) ∧ u - v * sqrt (w:ℝ) = largest_e ∧ u + v + w = 24 :=
by 
  use 14, 8, 2
  simp
  sorry

end largest_possible_value_of_e_is_24_l609_609874


namespace sum_of_three_digit_numbers_using_cards_l609_609561

-- Definitions based on the conditions
def cards : List ℕ := [2, 4, 6, 8]

-- Problem statement
theorem sum_of_three_digit_numbers_using_cards : 
  (∑ x in cards, 100 * x * 16 + ∑ x in cards, 10 * x * 16 + ∑ x in cards, x * 16) = 35520 :=
by 
  sorry

end sum_of_three_digit_numbers_using_cards_l609_609561


namespace parabola_properties_l609_609972

variable {b c m n : ℝ}

theorem parabola_properties (h1 : 0 = 1^2 + b * 1 + c)
  (h2 : m = 2^2 + b * 2 + c)
  (h3 : -4 = 3^2 + b * 3 + c)
  (h4 : n = 4^2 + b * 4 + c)
  (h5 : 0 = 5^2 + b * 5 + c) :
  (∀ x, y = x^2 + bx + c → y > 0 → x < 1 ∨ x > 5) ∧
  y = (x - 3)^2 - 4 ∧
  axis_of_symmetry y = 3 ∧
  parabola_opens_upwards y := sorry

end parabola_properties_l609_609972


namespace smallest_h_r_l609_609719

-- Stating the problem conditions and required proof in Lean 4
theorem smallest_h_r (r : ℕ) (hr : r ≥ 1) : ∃ h : ℕ, h = 2 * r ∧ 
  ∀ (partition : finset (finset (fin h))), 
    (partition.card = r) → 
    (∃ (a x y : ℕ), 0 ≤ a ∧ 1 ≤ x ∧ x ≤ y ∧ (∃ C ∈ partition, {a + x, a + y, a + x + y} ⊆ C)) :=
sorry

end smallest_h_r_l609_609719


namespace sum_distances_ge_fourth_vertex_l609_609461

variable {Point : Type}
variable [MetricSpace Point]

def is_isosceles_trapezoid (A B C D : Point) : Prop :=
  ∃ (AB_CD_parallel AD_eq_BC : Prop),
    (AB_CD_parallel → convex_hull [A, B] ∩ convex_hull [C, D] = ∅) ∧
    (AD_eq_BC → dist A D = dist B C)

theorem sum_distances_ge_fourth_vertex
    {A B C D M : Point} 
    (h : is_isosceles_trapezoid A B C D) :
  dist A M + dist B M + dist C M ≥ dist D M := sorry

end sum_distances_ge_fourth_vertex_l609_609461


namespace range_of_m_l609_609729

-- Define sets A and B
def A := {x : ℝ | x ≤ 1}
def B (m : ℝ) := {x : ℝ | x ≤ m}

-- Statement: Prove the range of m such that B ⊆ A
theorem range_of_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ (m ≤ 1) :=
by sorry

end range_of_m_l609_609729


namespace symmetric_points_and_triangle_properties_l609_609457

theorem symmetric_points_and_triangle_properties :
  let A := (5, 1)
  let B := (5, -1)
  let C := (-5, -1)
  let midpoint_AB := (5 / 2, 0)
  let midpoint_BC := (0, -1)
  -- Condition 1: Point A (5, 1) is symmetric to point B (5, -1) about the x-axis
  -- Condition 2: Point A (5, 1) is symmetric to point C (-5, -1) about the origin
  (∀ (x y : ℝ), x∈ A ∧ y ∈ B -> (x, y) = (5, 1)) ∧
  (∀ (x y : ℝ), x∈ A ∧ y ∈ C -> (x, y) = (-5, -1)) →
  -- Prove (question == answer given conditions)
  (line_eq : (2 : ℝ) * x - (5 : ℝ) * y - (5 : ℝ) = 0 ∧ (area_eq : (10 : ℝ))) :=
by
  sorry

end symmetric_points_and_triangle_properties_l609_609457


namespace probability_one_common_number_approx_l609_609390

noncomputable def probability_exactly_one_common : ℝ :=
  let total_combinations := Nat.choose 45 6
  let successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  successful_outcomes / total_combinations

theorem probability_one_common_number_approx :
  (probability_exactly_one_common ≈ 0.424) :=
by
  -- Definitions from conditions
  have total_combinations := Nat.choose 45 6
  have successful_outcomes := Nat.choose 6 1 * Nat.choose 39 5
  
  -- Statement of probability
  have prob := (successful_outcomes : ℝ) / total_combinations
  
  -- Approximation
  show prob ≈ 0.424 from sorry

end probability_one_common_number_approx_l609_609390


namespace tan_theta_plus_pi_l609_609379

theorem tan_theta_plus_pi (x y : ℝ) (h : y = -4) (hx : x = 3) (hne : x ≠ 0) :
  Real.tan (Real.atan2 y x + Real.pi) = 4 / 3 :=
by
  -- proof goes here
  sorry

end tan_theta_plus_pi_l609_609379


namespace smallest_z_l609_609842

-- Given conditions
def distinct_consecutive_even_positive_perfect_cubes (w x y z : ℕ) : Prop :=
  w^3 + x^3 + y^3 = z^3 ∧
  ∃ a b c d : ℕ, 
    a < b ∧ b < c ∧ c < d ∧
    2 * a = w ∧ 2 * b = x ∧ 2 * c = y ∧ 2 * d = z

-- The smallest value of z proving the equation holds
theorem smallest_z (w x y z : ℕ) (h : distinct_consecutive_even_positive_perfect_cubes w x y z) : z = 12 :=
  sorry

end smallest_z_l609_609842


namespace find_cos_C_l609_609001

noncomputable def cos_C_eq (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : Prop :=
  Real.cos C = 7 / 25

theorem find_cos_C (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) :
  cos_C_eq A B C a b c h1 h2 :=
sorry

end find_cos_C_l609_609001


namespace geometric_sequence_l609_609255

open Nat

def a : ℕ → ℕ
| 1 := 1
| (n + 1) := ((2 * ∑ k in range n, (k + 1) * a (k + 1)) + (n + 1) * a n) / (n + 2)

theorem geometric_sequence (n : ℕ) (hn : n ≥ 3) : 3 * (n - 1) * a (n - 1) = n * a n :=
by
  sorry

end geometric_sequence_l609_609255


namespace tatuya_ivanna_ratio_l609_609088

theorem tatuya_ivanna_ratio:
  let D := 90 in
  let I := (3 / 5) * D in
  let T := (252 - 144) in
  (T + I + D) / 3 = 84 → T / I = 2 := by
  sorry

end tatuya_ivanna_ratio_l609_609088


namespace part1_part2_l609_609172

variable {x : ℝ}

/-- Prove that the range of the function f(x) = (sqrt(1+x) + sqrt(1-x) + 2) * (sqrt(1-x^2) + 1) for 0 ≤ x ≤ 1 is (0, 8]. -/
theorem part1 (hx : 0 ≤ x ∧ x ≤ 1) :
  0 < ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ∧ 
  ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ≤ 8 :=
sorry

/-- Prove that for 0 ≤ x ≤ 1, there exists a positive number β such that sqrt(1+x) + sqrt(1-x) ≤ 2 - x^2 / β, with the minimal β = 4. -/
theorem part2 (hx : 0 ≤ x ∧ x ≤ 1) :
  ∃ β : ℝ, β > 0 ∧ β = 4 ∧ (Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) :=
sorry

end part1_part2_l609_609172


namespace inequality_relationship_l609_609789

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l609_609789


namespace range_of_f_l609_609112

noncomputable def f (x : ℝ) : ℝ := cos (π / 3 - x) + sin (π / 2 + x)

theorem range_of_f : (set.range f) = set.Icc (-real.sqrt 3) (real.sqrt 3) :=
sorry

end range_of_f_l609_609112


namespace find_ordered_pair_l609_609710

theorem find_ordered_pair : ∃ (x y : ℚ), 
  3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ 
  x = 57 / 31 ∧ y = 195 / 62 :=
by {
  sorry
}

end find_ordered_pair_l609_609710


namespace Vanya_correct_answers_l609_609496

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l609_609496


namespace factorize_xy2_minus_x_l609_609684

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609684


namespace possible_n_values_for_convex_n_gon_l609_609372

theorem possible_n_values_for_convex_n_gon (n : ℕ) (h : n ≥ 3) :
  (∃ (x : ℝ), let angles := [ℝ] in
    (∀ k, k ∈ Finset.range 1 (n+1) → (angles k = x * k) ∧ angles.Sum = (n-2) * 180)
    ∧ ∀ k, k ∈ Finset.range 1 (n+1) → x * k < 180)
  ↔ n = 3 ∨ n = 4 :=
sorry

end possible_n_values_for_convex_n_gon_l609_609372


namespace probability_exactly_one_common_number_l609_609387

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l609_609387


namespace sum_identity_l609_609656

noncomputable def sum_problem : ℝ :=
  ∑' (k : ℕ) in (Set.Ici 1), (12^k : ℝ) / ((4^k : ℝ) - (3^k : ℝ)) / ((4^(k + 1) : ℝ) - (3^(k + 1) : ℝ))

theorem sum_identity : sum_problem = 5 :=
  sorry

end sum_identity_l609_609656


namespace batsman_average_after_12th_inning_l609_609176

theorem batsman_average_after_12th_inning (average_initial : ℕ) (score_12th : ℕ) (average_increase : ℕ) (total_innings : ℕ) 
    (h_avg_init : average_initial = 29) (h_score_12th : score_12th = 65) (h_avg_inc : average_increase = 3) 
    (h_total_innings : total_innings = 12) : 
    (average_initial + average_increase = 32) := 
by
  sorry

end batsman_average_after_12th_inning_l609_609176


namespace decaffeinated_percentage_l609_609194

theorem decaffeinated_percentage (x : ℝ) :
  (30 / 100 * 400) + (x * 100) = 36 / 100 * (400 + 100) → x = 60 / 100 :=
by
  intros h
  have h1 : 30 / 100 * 400 = 120 by norm_num
  have h2 : 36 / 100 * (400 + 100) = 180 by norm_num
  rw [h1, h2] at h
  linarith

end decaffeinated_percentage_l609_609194


namespace Vanya_correct_answers_l609_609478

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l609_609478


namespace range_of_theta_l609_609239

-- Definitions
def regular_hexagon := { A B C D E F : Type }

-- Conditions
def mid_point (A B : Type) := sorry
def point_on_side (X Y : Type) := sorry

-- The range condition for θ
theorem range_of_theta (A B C D E F P Q : Type) (hex : regular_hexagon)
  (hP : mid_point A B)
  (hQ : point_on_side B C)
  (hθ : ∃ θ : ℝ, θ = real.angle B P Q) :
  arcsin (3 * sqrt 3 / sqrt 127 : ℝ) < θ ∧ θ < arcsin (3 * sqrt 3 / sqrt 91 : ℝ) :=
sorry

end range_of_theta_l609_609239


namespace cell_proliferation_correct_l609_609231

def cell_proliferation_statements : Prop :=
  ∀ A B C D : Prop,
  (A = "At the late stage of mitosis in diploid animal cells, each pole of the cell does not contain homologous chromosomes.")
  ∧ (B = "The genetic material in the cytoplasm of diploid organism cells is distributed randomly and unequally during cell division.")
  ∧ (C = "In the cells at the late stage of the second meiotic division of diploid organisms, the number of chromosomes is half that of somatic cells.")
  ∧ (D = "The separation of alleles occurs during the first meiotic division, and the independent assortment of non-allelic genes occurs during the second meiotic division.")
  → B

theorem cell_proliferation_correct : cell_proliferation_statements :=
  by
  intros A B C D h
  have := h.2.1
  sorry

end cell_proliferation_correct_l609_609231


namespace proof_arithmetic_sequence_sum_l609_609757

def arithmetic_sequence_sum_proof (a : ℕ → ℝ) (d : ℝ) :=
  -- The sequence is arithmetic
  (∀ n, a (n + 1) = a n + d) ∧ 
  -- Given condition
  (2 * a 8 - a 12 = 4) →
  -- Sum of first seven terms is 28
  (∑ i in Finset.range 7, a i) = 28

theorem proof_arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : 2 * a 8 - a 12 = 4) : 
  (∑ i in Finset.range 7, a i) = 28 :=
sorry

end proof_arithmetic_sequence_sum_l609_609757


namespace probability_exactly_one_common_number_l609_609386

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l609_609386


namespace draw_reds_first_probability_l609_609216

noncomputable def probability_all_red_before_both_greens : ℚ :=
  let total_arrangements := Nat.choose 6 2 in
  let favorable_arrangements := Nat.choose 5 1 in
  favorable_arrangements / total_arrangements

theorem draw_reds_first_probability :
  probability_all_red_before_both_greens = 1/3 :=
sorry

end draw_reds_first_probability_l609_609216


namespace quadratic_equation_has_root_l609_609072

theorem quadratic_equation_has_root (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + 2 * b * x + c = 0) ∨
             (b * x^2 + 2 * c * x + a = 0) ∨
             (c * x^2 + 2 * a * x + b = 0) :=
sorry

end quadratic_equation_has_root_l609_609072


namespace factorize_expression_l609_609688

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609688


namespace complement_M_in_U_l609_609061

open Set

theorem complement_M_in_U : 
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  U \ M = {3, 7} := 
by
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  sorry

end complement_M_in_U_l609_609061


namespace p_6_eq_163_l609_609881

noncomputable def p (x : ℕ) : ℕ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + x + 1

theorem p_6_eq_163 : p 6 = 163 :=
by
  sorry

end p_6_eq_163_l609_609881


namespace rope_cut_probability_l609_609207

theorem rope_cut_probability (L : ℝ) (cut_position : ℝ) (P : ℝ) :
  L = 4 → (∀ cut_position, 0 ≤ cut_position ∧ cut_position ≤ L →
  (cut_position ≥ 1.5 ∧ (L - cut_position) ≥ 1.5)) → P = 1 / 4 :=
by
  intros hL hcut
  sorry

end rope_cut_probability_l609_609207


namespace math_problem_solution_l609_609339

theorem math_problem_solution :
  (∃ a b c : ℤ,
    (sqrt (2 * a + 5) = 3 ∨ sqrt (2 * a + 5) = -3) ∧ 
    (∛(2 * b - a) = 2) ∧
    (c = int.floor (sqrt 5))) ∧ 
    (∃ r : ℝ, r = sqrt (a + 2 * b - c) ∧ r = sqrt 10) :=
by 
  use [2, 5, 2]
  have ha : sqrt (2 * 2 + 5) = 3, by norm_num
  have hb : ∛(2 * 5 - 2) = 2, by norm_num
  have hc : (2 : ℤ) = int.floor (sqrt 5), by norm_num
  split
  all_goals {sorry}

end math_problem_solution_l609_609339


namespace prism_volume_from_sphere_l609_609317

open Real 

noncomputable def circumscribed_sphere_volume (R : ℝ) : ℝ := 
  (4 / 3) * π * R^3

noncomputable def triangular_base_area (a : ℝ) : ℝ :=
  (sqrt 3 / 4) * a^2

noncomputable def prism_volume (a R : ℝ) : ℝ :=
  (sqrt 3 / 4) * a^2 * (3 * R / 2)

theorem prism_volume_from_sphere (a R V : ℝ) (h : ℝ) :
  V = circumscribed_sphere_volume R →
  h = 3 * R / 2 →
  triangular_base_area a * h = prism_volume a R :=
by
  intro h₁ h₂
  rw [circumscribed_sphere_volume, h₂, <-eq_comm]
  sorry

end prism_volume_from_sphere_l609_609317


namespace sum_rows_equals_ten_l609_609087

-- Define the domino as a structure for clarity
structure Domino :=
  (a : Nat)
  (b : Nat)

-- List of unique dominoes that are given
def dominos : List Domino :=
  [
    ⟨0, 1⟩, ⟨0, 2⟩, ⟨0, 3⟩, ⟨0, 4⟩, ⟨0, 5⟩, ⟨0, 6⟩,
    ⟨1, 2⟩, ⟨1, 3⟩, ⟨1, 4⟩, ⟨1, 5⟩, ⟨1, 6⟩,
    ⟨2, 3⟩, ⟨2, 4⟩, ⟨2, 5⟩, ⟨2, 6⟩
  ]

-- Convert a domino to fraction with denominator 60
def to_fraction (d : Domino) : ℚ :=
  if d.b = 0 then 0 else d.a / d.b

-- Define the rows consisting of dominos
def row1 : List Domino := [⟨0, 6⟩, ⟨1, 5⟩, ⟨2, 4⟩, ⟨0, 5⟩, ⟨1, 4⟩]
def row2 : List Domino := [⟨0, 4⟩, ⟨1, 3⟩, ⟨2, 2⟩, ⟨0, 3⟩, ⟨1, 2⟩]
def row3 : List Domino := [⟨0, 2⟩, ⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩, ⟨4, 1⟩]

-- Function to sum the fractions in a row
def sum_fractions (row : List Domino) : ℚ :=
  row.map to_fraction |> List.sum

-- Lean 4 Statement: Proving the sums equal 10
theorem sum_rows_equals_ten :
  sum_fractions row1 + sum_fractions row2 + sum_fractions row3 = 30 :=
by
  -- Each row sum should be equal to 10
  have h1 : sum_fractions row1 = 10 := sorry
  have h2 : sum_fractions row2 = 10 := sorry
  have h3 : sum_fractions row3 = 10 := sorry
  -- Hence their total would be 30
  calc
    sum_fractions row1 + sum_fractions row2 + sum_fractions row3 
      = 10 + sum_fractions row2 + sum_fractions row3 : by rw h1
      = 10 + 10 + sum_fractions row3 : by rw h2
      = 10 + 10 + 10 : by rw h3
      = 30 : by rfl

end sum_rows_equals_ten_l609_609087


namespace other_root_of_quadratic_eq_l609_609912

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l609_609912


namespace min_ab_given_parallel_l609_609751

-- Define the conditions
def parallel_vectors (a b : ℝ) : Prop :=
  4 * b - a * (b - 1) = 0 ∧ b > 1

-- Prove the main statement
theorem min_ab_given_parallel (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h_parallel : parallel_vectors a b) :
  a + b = 9 :=
sorry  -- Proof is omitted

end min_ab_given_parallel_l609_609751


namespace correct_props_l609_609793

noncomputable def f (a b x : ℝ) : ℝ :=
  a * Real.sin (2 * x) + b * Real.cos (2 * x)

def prop1 (a b : ℝ) (h : a = 1 ∧ b = -Real.sqrt 3) : Prop :=
  f a b x = 2 * Real.sin (2 * (x - Real.pi / 6))

def prop2 (a b : ℝ) (h : a = 1 ∧ b = -1) : Prop :=
  ¬(Real.sin (2 * (Real.pi / 4) - Real.pi / 4) = 0 ∧ f a b (Real.pi / 4) = 0)

def prop3 (a b : ℝ) (h : ∃ x, Real.sin (2 * x) * a + Real.cos (2 * x) * b = Real.sqrt (a^2 + b^2) ∧ x = Real.pi / 8) : Prop :=
  a = b

def prop4 (a b : ℝ) (h : ∃ m, ∀ x1 x2 x3, a * Real.sin (2 * x1) + b * Real.cos (2 * x1) = m ∧
  a * Real.sin (2 * x2) + b * Real.cos (2 * x2) = m ∧
  a * Real.sin (2 * x3) + b * Real.cos (2 * x3) = m ∧
  x2 - x1 = Real.pi ∧ x3 - x2 = Real.pi) : Prop :=
  false

theorem correct_props (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (prop1 a b ⟨rfl, rfl⟩ ∧ prop3 a b ⟨x, ⟨rfl, rfl⟩⟩) ∧ 
  ¬(prop2 a b ⟨rfl, rfl⟩ ∨ prop4 a b ⟨m, ⟨x1, x2, x3, hx⟩⟩) :=
begin
  sorry
end

end correct_props_l609_609793


namespace compare_values_l609_609774

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l609_609774


namespace a_n_formula_b_n_formula_sum_first_n_b_formula_l609_609060

def sequence_a_n (n : ℕ) : ℝ :=
  if n % 2 = 0 then n / 2 - 1 else (n + 1) / 2

def sequence_b_n (n : ℕ) : ℝ :=
  n^2 - n

def sum_first_n_b (n : ℕ) : ℝ :=
  (n * (n^2 - 1)) / 3

theorem a_n_formula (n : ℕ) : 
  sequence_a_n n = 
  if n % 2 = 0 then n / 2 - 1 else (n + 1) / 2 := 
sorry

theorem b_n_formula (n : ℕ) : 
  sequence_b_n n = n^2 - n := 
sorry

theorem sum_first_n_b_formula (n : ℕ) : 
  sum_first_n_b n = ∑ k in Finset.range n, sequence_b_n (k + 1) := 
sorry

end a_n_formula_b_n_formula_sum_first_n_b_formula_l609_609060


namespace find_sin_2x_minus_y_l609_609760

noncomputable theory -- The problem uses trigonometric functions which are noncomputable

open Real

def equation1 (x : ℝ) : Prop := x + sin x * cos x = 1

def equation2 (y : ℝ) : Prop := 2 * cos y - 2 * y + π + 4 = 0

theorem find_sin_2x_minus_y (x y : ℝ) (h1 : equation1 x) (h2 : equation2 y) : sin (2 * x - y) = -1 :=
sorry -- The proof is omitted as per instructions

end find_sin_2x_minus_y_l609_609760


namespace train_crossing_time_l609_609161

def train_length : ℕ := 300
def bridge_length : ℕ := 115
def train_speed_kmh : ℝ := 35

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance : ℕ := train_length + bridge_length

noncomputable def crossing_time : ℝ := total_distance / train_speed_ms

theorem train_crossing_time : 
  crossing_time ≈ 42.7 :=
by sorry

end train_crossing_time_l609_609161


namespace chimney_bricks_l609_609241

noncomputable def brenda_rate (x : ℝ) : ℝ := x / 9
noncomputable def brandon_rate (x : ℝ) : ℝ := x / 10
noncomputable def charlie_rate (x : ℝ) : ℝ := x / 12
noncomputable def combined_rate (x : ℝ) : ℝ := (brenda_rate x + brandon_rate x + charlie_rate x) - 15
noncomputable def time : ℝ := 3.5

theorem chimney_bricks : 
  (λ x : ℝ, (combined_rate x) * time = x) 330 :=
by
  simp [brenda_rate, brandon_rate, charlie_rate, combined_rate, time]
  sorry

end chimney_bricks_l609_609241


namespace total_participants_l609_609502

-- Define the number of indoor and outdoor participants
variables (x y : ℕ)

-- First condition: number of outdoor participants is 480 more than indoor participants
def condition1 : Prop := y = x + 480

-- Second condition: moving 50 participants results in outdoor participants being 5 times the indoor participants
def condition2 : Prop := y + 50 = 5 * (x - 50)

-- Theorem statement: the total number of participants is 870
theorem total_participants (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 870 :=
sorry

end total_participants_l609_609502


namespace total_water_filled_jars_l609_609029

theorem total_water_filled_jars (x : ℕ) (h : 4 * x + 2 * x + x = 14 * 4) : 3 * x = 24 :=
by
  sorry

end total_water_filled_jars_l609_609029


namespace vanya_correct_answers_l609_609467

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l609_609467


namespace obtuse_angle_perpendicular_division_l609_609610

theorem obtuse_angle_perpendicular_division :
  ∀ θ, (∃ φ, θ = 6 * φ + φ ∧ φ = 15) → θ = 105 := 
by
  intros θ h
  rcases h with ⟨φ, h1, h2⟩
  rw h2 at h1
  norm_num at h1
  sorry

end obtuse_angle_perpendicular_division_l609_609610


namespace find_c_plus_d_l609_609052

theorem find_c_plus_d (c d : ℝ)
  (h_eq : ∀ (x : ℝ), y = c + d / x)
  (h1: h_eq 3 = 2)
  (h2: h_eq (-3) = 6) :
  c + d = -2 :=
sorry

end find_c_plus_d_l609_609052


namespace cos_phi_expression_l609_609441

theorem cos_phi_expression (a b c : ℝ) (φ R : ℝ)
  (habc : a > 0 ∧ b > 0 ∧ c > 0)
  (angles : 2 * φ + 3 * φ + 4 * φ = π)
  (law_of_sines : a / Real.sin (2 * φ) = 2 * R ∧ b / Real.sin (3 * φ) = 2 * R ∧ c / Real.sin (4 * φ) = 2 * R) :
  Real.cos φ = (a + c) / (2 * b) := 
by 
  sorry

end cos_phi_expression_l609_609441


namespace eval_expression_l609_609675

-- Define the given expression
def given_expression : ℤ := -( (16 / 2) * 12 - 75 + 4 * (2 * 5) + 25 )

-- State the desired result in a theorem
theorem eval_expression : given_expression = -86 := by
  -- Skipping the proof as per instructions
  sorry

end eval_expression_l609_609675


namespace range_of_m_l609_609795

noncomputable def f (x m : ℝ) : ℝ := (1 / 4) * x^4 - (2 / 3) * x^3 + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x m + (1 / 3) ≥ 0) ↔ m ≥ 1 := 
sorry

end range_of_m_l609_609795


namespace magnitude_evaluation_l609_609274

def complex_magnitude : ℂ := 7 - 24 * complex.I

theorem magnitude_evaluation : complex.abs complex_magnitude = 25 := 
by
  sorry

end magnitude_evaluation_l609_609274


namespace factorize_expression_l609_609690

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609690


namespace tangent_line_at_one_l609_609815

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem tangent_line_at_one : 
  let x := 1
  let y := f x
  let slope := deriv f x
  ∃ m b : ℝ, (slope = m) ∧ (y = m * x + b) ∧ (m * x - 1 * y + b = 0) :=
begin
  sorry
end

end tangent_line_at_one_l609_609815


namespace lottery_probability_exactly_one_common_l609_609397

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609397


namespace area_triangle_DBC_l609_609846

-- Define points A, B, and C
def A : ℝ × ℝ := (2, 10)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (14, 2)

-- Define midpoint function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define points D, and E as midpoints
def D : ℝ × ℝ := midpoint A B
def E : ℝ × ℝ := midpoint B C

-- Function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define base and height from the calculated distances
def base : ℝ := distance B C
def height : ℝ := abs (D.2 - B.2)  -- D and B share the same x-coordinate

-- Theorem statement: area of triangle DBC
theorem area_triangle_DBC : (1 / 2) * base * height = 24 := by
  sorry

end area_triangle_DBC_l609_609846


namespace minimum_possible_value_l609_609051

-- Define the set of distinct elements
def distinct_elems : Set ℤ := {-8, -6, -4, -1, 1, 3, 7, 12}

-- Define the existence of distinct elements
def elem_distinct (p q r s t u v w : ℤ) : Prop :=
  p ∈ distinct_elems ∧ q ∈ distinct_elems ∧ r ∈ distinct_elems ∧ s ∈ distinct_elems ∧ 
  t ∈ distinct_elems ∧ u ∈ distinct_elems ∧ v ∈ distinct_elems ∧ w ∈ distinct_elems ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ 
  s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ 
  t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ 
  u ≠ v ∧ u ≠ w ∧ 
  v ≠ w

-- The main proof problem
theorem minimum_possible_value :
  ∀ (p q r s t u v w : ℤ), elem_distinct p q r s t u v w ->
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 := 
sorry

end minimum_possible_value_l609_609051


namespace initial_distance_between_trains_l609_609564

-- Given lengths of the trains
def length_train1 : ℝ := 120
def length_train2 : ℝ := 210

-- Given speeds of the trains in kmph
def speed_train1_kmph : ℝ := 69
def speed_train2_kmph : ℝ := 82

-- Given time until the trains meet in hours
def time_to_meet_hrs : ℝ := 1.9071321976361095

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

-- Speeds of the trains in m/s
def speed_train1_mps : ℝ := kmph_to_mps speed_train1_kmph
def speed_train2_mps : ℝ := kmph_to_mps speed_train2_kmph

-- Relative speed of the two trains moving towards each other
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

-- Time until the trains meet in seconds
def time_to_meet_secs : ℝ := time_to_meet_hrs * 3600

-- Distance covered while moving towards each other
def distance_covered : ℝ := relative_speed_mps * time_to_meet_secs

-- Initial distance between the trains
def initial_distance : ℝ := distance_covered - (length_train1 + length_train2)

-- The proof statement
theorem initial_distance_between_trains : initial_distance = 287670 :=
by sorry

end initial_distance_between_trains_l609_609564


namespace bittu_days_proof_l609_609637

noncomputable def bittu_work_days (anand_days : ℕ) (chandu_days : ℕ) (bittu_days : ℝ) : Prop :=
  (anand_days = 7) ∧ (chandu_days = 6) ∧ 
  (2 * ((1.0 / anand_days.toReal) + (1.0 / bittu_days) + (1.0 / chandu_days.toReal)) + (1.0 / anand_days.toReal) = 1)

theorem bittu_days_proof : (∃ bittu_days : ℝ, bittu_work_days 7 6 bittu_days ∧ bittu_days = 8.4) :=
by {
  /- Proof to be filled in -/
  sorry
}

end bittu_days_proof_l609_609637


namespace second_quadrant_distance_l609_609824

theorem second_quadrant_distance 
    (m : ℝ) 
    (P : ℝ × ℝ)
    (hP1 : P = (m - 3, m + 2))
    (hP2 : (m + 2) > 0)
    (hP3 : (m - 3) < 0)
    (hDist : |(m + 2)| = 4) : P = (-1, 4) := 
by
  have h1 : m + 2 = 4 := sorry
  have h2 : m = 2 := sorry
  have h3 : P = (2 - 3, 2 + 2) := sorry
  have h4 : P = (-1, 4) := sorry
  exact h4

end second_quadrant_distance_l609_609824


namespace jill_other_items_tax_l609_609451

variable (total_amount : ℝ)
variable (clothing_percentage : ℝ := 0.50)
variable (food_percentage : ℝ := 0.20)
variable (other_items_percentage : ℝ := 0.30)
variable (clothing_tax_rate : ℝ := 0.05)
variable (food_tax_rate : ℝ := 0.00)
variable (total_tax_rate : ℝ := 0.055)

-- Tax rate on other items
def other_items_tax_rate (total_amount : ℝ) : ℝ :=
  let clothing_expenditure := total_amount * clothing_percentage
  let food_expenditure := total_amount * food_percentage
  let other_items_expenditure := total_amount * other_items_percentage
  let clothing_tax := clothing_expenditure * clothing_tax_rate
  let food_tax := food_expenditure * food_tax_rate
  let total_tax := total_amount * total_tax_rate
  let other_items_tax := total_tax - clothing_tax - food_tax
  (other_items_tax / other_items_expenditure) * 100

theorem jill_other_items_tax :
  other_items_tax_rate total_amount = 10 :=
by
  sorry

end jill_other_items_tax_l609_609451


namespace LimingFatherAge_l609_609220

theorem LimingFatherAge
  (age month day : ℕ)
  (age_condition : 18 ≤ age ∧ age ≤ 70)
  (product_condition : age * month * day = 2975)
  (valid_month : 1 ≤ month ∧ month ≤ 12)
  (valid_day : 1 ≤ day ∧ day ≤ 31)
  : age = 35 := sorry

end LimingFatherAge_l609_609220


namespace integer_roots_polynomial_l609_609707

theorem integer_roots_polynomial (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x + 9 = 0) ↔ 
  (a = -109 ∨ a = -21 ∨ a = -13 ∨ a = 3 ∨ a = 11 ∨ a = 53) :=
by
  sorry

end integer_roots_polynomial_l609_609707


namespace lottery_probability_exactly_one_common_l609_609396

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem lottery_probability_exactly_one_common :
  let total_ways := choose 45 6
  let successful_ways := choose 6 1 * choose 39 5
  let probability := successful_ways.toReal / total_ways.toReal
  probability = 6 * (choose 39 5).toReal / (choose 45 6).toReal :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609396


namespace chores_equality_l609_609647

theorem chores_equality :
  let mins_sweeping_per_room := 3
      mins_dishes_per_dish := 2
      mins_laundry_per_load := 9
      rooms_anna := 10
      loads_billy := 2
      time_anna := rooms_anna * mins_sweeping_per_room
      time_billy := loads_billy * mins_laundry_per_load
      dishes_billy := (time_anna - time_billy) / mins_dishes_per_dish
  in dishes_billy = 6 :=
by
  sorry

end chores_equality_l609_609647


namespace inscribable_quadrilateral_l609_609944

theorem inscribable_quadrilateral (A B C D O : Type) 
  [Point A] [Point B] [Point C] [Point D] [Point O]
  (diagonals_bisect_angles : ∀ {α β γ δ : ℝ}, 
    ∠(A O B) = ∠(D O C) ∧ ∠(C O B) = ∠(A O D)) :
  ∃ (r : ℝ) (O_center : Point O), is_incircle A B C D O_center r :=
sorry

end inscribable_quadrilateral_l609_609944


namespace boxes_needed_l609_609168

def initial_games : ℕ := 76
def games_sold : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : (initial_games - games_sold) / games_per_box = 6 := by
  sorry

end boxes_needed_l609_609168


namespace sum_of_values_l609_609893

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 5 * x - 3 else x^2 - 4 * x + 3

theorem sum_of_values (s : Finset ℝ) : 
  (∀ x ∈ s, f x = 2) → s.sum id = 4 :=
by 
  sorry

end ProofProblem

end sum_of_values_l609_609893


namespace track_circumference_is_720_l609_609590

variables (x : ℝ) -- half the circumference
variables (distance_travelled_by_B_first_meet : ℝ) (distance_shy_of_A_full_lap : ℝ)
variables (circumference : ℝ)

-- Conditions
axiom distance_travelled_by_B_first_meet_eq : distance_travelled_by_B_first_meet = 150
axiom distance_shy_of_A_full_lap_eq : distance_shy_of_A_full_lap = 90
axiom circumference_eq : circumference = 2 * x

-- The problem to be resolved
theorem track_circumference_is_720 :
  let A_distance_first_meet := x - 150,
      A_distance_second_meet := 2 * x - 90,
      B_distance_second_meet := x + 90 in
  A_distance_first_meet = (circumference / 2) - 150 →
  150 / A_distance_first_meet = B_distance_second_meet / A_distance_second_meet →
  (x = 360) →
  circumference = 720 :=
sorry

end track_circumference_is_720_l609_609590


namespace polygon_intersections_l609_609462

def regular_polygon (n : ℕ) (r : ℝ) : set (ℂ) := sorry

def total_intersection_points (polygons : list (set ℂ)) : ℕ := sorry

theorem polygon_intersections :
  let P4 := regular_polygon 4 1 in
  let P5 := regular_polygon 5 1 in
  let P7 := regular_polygon 7 1 in
  let P9 := regular_polygon 9 1 in
  (total_intersection_points [P4, P5, P7, P9] = 58) :=
by sorry

end polygon_intersections_l609_609462


namespace increasing_function_range_a_l609_609792

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1 then -x^2 - a * x - 7 else a / x

theorem increasing_function_range_a :
  (∀ x y, x ≤ y → f a x ≤ f a y) → -4 ≤ a ∧ a ≤ -2 :=
by
  intro h
  sorry

end increasing_function_range_a_l609_609792


namespace g_x0_symmetry_range_m_l609_609797

-- Definitions
def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 12) ^ 2
def g (x : ℝ) : ℝ := 1 + (1 / 2) * Real.sin (2 * x)
def h (x : ℝ) : ℝ := f x + g x
def I : Set ℝ := Set.Icc (-Real.pi / 12) (5 * Real.pi / 12)
def symmetric_axis (x0 : ℝ) : Prop := ∃ k : ℤ, x0 = k * Real.pi / 2 - Real.pi / 12

-- Proof problem for g(x0)
theorem g_x0_symmetry (x0 : ℝ) (h_symmetry : symmetric_axis x0) : 
  g x0 = 3 / 4 ∨ g x0 = 5 / 4 :=
by
  sorry

-- Proof problem for range of m
theorem range_m :
  {m : ℝ | ∀ x ∈ I, |h x - m| ≤ 1} = Set.Icc (1 : ℝ) (9 / 4) :=
by
  sorry

end g_x0_symmetry_range_m_l609_609797


namespace limit_of_derivative_l609_609814

variable {f : ℝ → ℝ}
variable {x0 : ℝ}

/-
Given that the derivative of f at x0 is -3, we want to prove:
lim_{h \to 0} (f (x0 + h) - f (x0 - 3 * h)) / h = -12.
-/
theorem limit_of_derivative (h : ℝ) :
  deriv f x0 = -3 → (tendsto (fun h => (f (x0 + h) - f (x0 - 3 * h)) / h) (nhds 0) (nhds (-12))) :=
begin
  sorry
end

end limit_of_derivative_l609_609814


namespace general_equation_of_curve_C_rectangular_equation_of_line_l_max_distance_from_curve_to_line_l609_609800

-- Condition 1: Parametric equation of curve C
def curve_parametric (θ : ℝ) : ℝ × ℝ := (√3 * Real.cos θ, Real.sin θ)

-- Condition 2: Polar coordinate equation of line l
def line_polar (θ : ℝ) (ρ: ℝ) : Prop := ρ * Real.sin (θ + (π / 4)) = 2 * √2

-- Theorem statement for the first condition, proving the general equation of curve C
theorem general_equation_of_curve_C (x y θ : ℝ) : (x, y) = curve_parametric θ → (x^2 / 3 + y^2 = 1) :=
sorry

-- Theorem statement for the second condition, proving the rectangular coordinate equation of line l
theorem rectangular_equation_of_line_l (x y : ℝ) : line_polar (θ) (Real.sqrt (x^2 + y^2)) →
  (x + y - 4 = 0) :=
sorry

-- Theorem statement for the third condition, proving the maximum distance from P to line l
theorem max_distance_from_curve_to_line (θ : ℝ) :
  let P := curve_parametric θ in
  let d := (abs (P.1 + P.2 - 4)) / Real.sqrt 2 in
  d ≤ 3 * √2 :=
sorry

end general_equation_of_curve_C_rectangular_equation_of_line_l_max_distance_from_curve_to_line_l609_609800


namespace number_of_slate_rocks_l609_609121

theorem number_of_slate_rocks (S : ℕ) :
  (S / (S + 15)) * ((S - 1) / (S + 14)) = 0.15 → S = 10 := 
by
  intro h
  sorry

end number_of_slate_rocks_l609_609121


namespace ellipse_properties_l609_609319

noncomputable def ellipse (a b : ℝ) := {p : ℝ × ℝ // (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}
variables (a b c : ℝ) (e : ℝ) (k t : ℝ) (P Q : ellipse a b) (M N : ℝ × ℝ)

axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : e = (Real.sqrt 2) / 2
axiom h4 : ∃ (rhomb_area : ℝ), rhomb_area = 2 * Real.sqrt 2
axiom h5 : ∀ P Q : ellipse a b, ∃ l : ℝ × ℝ → ℝ × ℝ, l = (fun (p : ℝ × ℝ) => (k * p.1 + t, p.2))
axiom h6 : ∀ P Q : ellipse a b, ∃ M N : ℝ × ℝ, M.1 = P.1 ∧ N.1 = Q.1 ∧ M.2 = 0 ∧ N.2 = 0
axiom h7 : abs (M.1 * N.1) = 2

theorem ellipse_properties :
  ∃ a b : ℝ, a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2)/2 + y^2 = 1) →
  ∃ t : ℝ, t = 0 ∧ ∀ k : ℝ, ∀ P Q : ellipse a b, 
  (∀ x y : ℝ, P.1 = x ∧ P.2 = y → l(p) = (fun p => (k * p.1 + t, p.2)) → l(p) passes through (0, 0)) :=
sorry

end ellipse_properties_l609_609319


namespace problem_statement_l609_609843

variables {A B C P D E F Q : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
          (AP : A → P) (BP : B → P) (CP : C → P)
          (AD : A → D) (BD : B → D) (CD : C → D) 
          (AE : A → E) (BE : B → E) (CE : C → E)
          (AF : A → F) (BF : B → F) (CF : C → F)
          (EQ : E → Q) (FQ : F → Q)

theorem problem_statement 
  (point_inside_triangle : ∀ (P : P), P ∈ interior (triangle A B C))
  (extensions_intersect_opposite_sides : ∀ (P D E F : Type), 
    (AP : A → P) → (BP : B → P) → (CP : C → P) →
    (AD : A → D) → (AE : A → E) → (AF : A → F) → 
    (BD : B → D) → (BE : B → E) → (BF : B → F) →
    (CD : C → D) → (CE : C → E) → (CF : C → F) →
    line_intercept AP AD = D → 
    line_intercept BP BE = E → 
    line_intercept CP CF = F)
  (intersect_ef_ad : ∀ (E F Q : Type), 
    (EF : E → F) → (AD : A → D) → 
    line_intercept EF AD = Q) :
   PQ ≤ (3 - 2 * sqrt 2) * AD :=
sorry

end problem_statement_l609_609843


namespace find_numbers_l609_609595

theorem find_numbers (a b c : ℕ) (h₁ : 10 ≤ b ∧ b < 100) (h₂ : 10 ≤ c ∧ c < 100)
    (h₃ : 10^4 * a + 100 * b + c = (a + b + c)^3) : (a = 9 ∧ b = 11 ∧ c = 25) :=
by
  sorry

end find_numbers_l609_609595


namespace expand_expression_l609_609280

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l609_609280


namespace vanya_correct_answers_l609_609481

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l609_609481


namespace expenditure_representation_l609_609651

theorem expenditure_representation (income expenditure : ℤ)
  (h_income : income = 60)
  (h_expenditure : expenditure = 40) :
  -expenditure = -40 :=
by {
  sorry
}

end expenditure_representation_l609_609651


namespace num_ways_to_arrange_books_l609_609015

-- Definitions based on the problem's conditions
def num_english_books : ℕ := 2
def num_science_books : ℕ := 4
def num_arrangements : ℕ := 48

-- Statement of the theorem
theorem num_ways_to_arrange_books : 
  ∃ e_books s_books : ℕ, e_books = num_english_books ∧ s_books = num_science_books ∧
  (e_books * 1 * (nat.factorial s_books) = num_arrangements) :=
begin
  let e_books := num_english_books,
  let s_books := num_science_books,
  have h1 : e_books = num_english_books := rfl,
  have h2 : s_books = num_science_books := rfl,
  have h3 : e_books * 1 * (nat.factorial s_books) = num_arrangements,
  { simp [e_books, s_books, num_arrangements, nat.factorial], },
  use [e_books, s_books],
  tauto,
end

end num_ways_to_arrange_books_l609_609015


namespace transformation_is_rightward_shift_l609_609529

-- Definitions for the problem
def original_function (x : ℝ) : ℝ := Real.cos x
def transformed_function (x : ℝ) : ℝ := Real.cos (x - π / 3)

-- Theorem to prove that the transformation is a rightward shift by π / 3
theorem transformation_is_rightward_shift : ∀ x : ℝ, transformed_function x = original_function (x - π / 3) := by
    sorry

end transformation_is_rightward_shift_l609_609529


namespace behavior_at_infinities_l609_609268

def f (x : Real) : Real := -3 * x^4 + 4 * x^2 + 5

theorem behavior_at_infinities : 
  (tendsto (fun x => f x) atTop atBot) ∧ (tendsto (fun x => f x) atBot atBot) :=
by
  sorry

end behavior_at_infinities_l609_609268


namespace unique_tangent_circle_of_radius_2_l609_609123

noncomputable def is_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  dist c₁ c₂ = r₁ + r₂

theorem unique_tangent_circle_of_radius_2
    (C1_center C2_center C3_center : ℝ × ℝ)
    (h_C1_C2 : is_tangent C1_center C2_center 1 1)
    (h_C2_C3 : is_tangent C2_center C3_center 1 1)
    (h_C3_C1 : is_tangent C3_center C1_center 1 1):
    ∃! center : ℝ × ℝ, is_tangent center C1_center 2 1 ∧
                        is_tangent center C2_center 2 1 ∧
                        is_tangent center C3_center 2 1 := sorry

end unique_tangent_circle_of_radius_2_l609_609123


namespace problem_solution_l609_609771

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l609_609771


namespace median_interval_l609_609665

-- Defining the histogram data
def histogram : List (ℕ × ℕ) := [
  (90, 22),
  (80, 18),
  (70, 20),
  (60, 15),
  (50, 25)
]

-- Proving the median is in the interval 70-79
theorem median_interval (histogram : List (ℕ × ℕ)) (n : ℕ) :
  n = 100 →
  let median_position := n / 2
  let cumulative_counts := histogram.scanl (λ acc pair, acc + pair.snd) 0
  cumulative_counts.findIndex (λ count, count ≥ median_position) = 2 →
  (70 ≤ median_position) ∧ (median_position < 80) :=
by
  intros h1 h2
  sorry

end median_interval_l609_609665


namespace peggy_dolls_after_all_events_l609_609917

def initial_dolls : Nat := 6
def grandmother_gift : Nat := 28
def birthday_gift : Nat := grandmother_gift / 2
def lost_dolls (total : Nat) : Nat := (10 * total + 9) / 100  -- using integer division for rounding 10% up
def easter_gift : Nat := (birthday_gift + 2) / 3  -- using integer division for rounding one-third up
def friend_exchange_gain : Int := -1  -- gaining 1 doll but losing 2
def christmas_gift (easter_dolls : Nat) : Nat := (20 * easter_dolls) / 100 + easter_dolls  -- 20% more dolls
def ruined_dolls : Nat := 3

theorem peggy_dolls_after_all_events : initial_dolls + grandmother_gift + birthday_gift - lost_dolls (initial_dolls + grandmother_gift + birthday_gift) + easter_gift + friend_exchange_gain.toNat + christmas_gift easter_gift - ruined_dolls = 50 :=
by
  sorry

end peggy_dolls_after_all_events_l609_609917


namespace quesadilla_cost_l609_609900

theorem quesadilla_cost :
  let pasta_per_kg := 1.5
  let pasta_kg := 2
  let beef_per_kg := 8
  let beef_kg := 1/4
  let sauce_per_jar := 2
  let sauce_jars := 2
  let total_money := 15
  let total_cost := pasta_per_kg * pasta_kg + beef_per_kg * beef_kg + sauce_per_jar * sauce_jars
  in total_money - total_cost = 6 :=
by
  sorry

end quesadilla_cost_l609_609900


namespace bullying_instances_l609_609036

noncomputable def typical_fingers_toes : ℕ := 10 + 10

noncomputable def suspension_days_per_instance : ℕ := 3

noncomputable def kris_total_suspension_days : ℕ := 3 * typical_fingers_toes

theorem bullying_instances : ∀ (days_per_instance total_days : ℕ), 
  total_days = 3 * (10 + 10) → days_per_instance = 3 → total_days / days_per_instance = 20 :=
by
  intros days_per_instance total_days h1 h2
  rw [h1, h2]
  exact sorry

end bullying_instances_l609_609036


namespace students_like_neither_l609_609417

theorem students_like_neither (N_Total N_Chinese N_Math N_Both N_Neither : ℕ)
  (h_total: N_Total = 62)
  (h_chinese: N_Chinese = 37)
  (h_math: N_Math = 49)
  (h_both: N_Both = 30)
  (h_neither: N_Neither = N_Total - (N_Chinese - N_Both) - (N_Math - N_Both) - N_Both) : 
  N_Neither = 6 :=
by 
  rw [h_total, h_chinese, h_math, h_both] at h_neither
  exact h_neither.trans (by norm_num)


end students_like_neither_l609_609417


namespace correct_order_of_actions_l609_609987

-- Definitions based on the conditions
def actions : ℕ → String
| 1 => "tap"
| 2 => "pay online"
| 3 => "swipe"
| 4 => "insert into terminal"
| _ => "undefined"

def paymentTechnology : ℕ → String
| 1 => "PayPass"
| 2 => "CVC"
| 3 => "magnetic stripe"
| 4 => "chip"
| _ => "undefined"

-- Proof problem statement
theorem correct_order_of_actions :
  (actions 4 = "insert into terminal") ∧
  (actions 3 = "swipe") ∧
  (actions 1 = "tap") ∧
  (actions 2 = "pay online") →
  [4, 3, 1, 2] corresponds to ["chip", "magnetic stripe", "PayPass", "CVC"]
:=
by
  sorry

end correct_order_of_actions_l609_609987


namespace sum_of_samples_is_six_l609_609197

-- Defining the conditions
def grains_varieties : ℕ := 40
def vegetable_oil_varieties : ℕ := 10
def animal_products_varieties : ℕ := 30
def fruits_and_vegetables_varieties : ℕ := 20
def sample_size : ℕ := 20
def total_varieties : ℕ := grains_varieties + vegetable_oil_varieties + animal_products_varieties + fruits_and_vegetables_varieties

def proportion_sample := (sample_size : ℚ) / total_varieties

-- Definitions for the problem
def vegetable_oil_sampled := (vegetable_oil_varieties : ℚ) * proportion_sample
def fruits_and_vegetables_sampled := (fruits_and_vegetables_varieties : ℚ) * proportion_sample

-- Lean 4 statement for the proof problem
theorem sum_of_samples_is_six :
  vegetable_oil_sampled + fruits_and_vegetables_sampled = 6 := by
  sorry

end sum_of_samples_is_six_l609_609197


namespace converse_l609_609521

section
variables (a b : ℝ) -- Assumption that a and b are real numbers

-- The initial proposition
definition init_prop (a b : ℝ) : Prop := a > b → 2^a > 2^b - 1

-- The converse proposition we want to prove
theorem converse (a b : ℝ) : (2^a > 2^b - 1) → (a > b) :=
sorry  -- Proof is skipped
end

end converse_l609_609521


namespace calculate_f7_plus_f9_l609_609528

-- Definitions based on the conditions
noncomputable def f : ℝ → ℝ := sorry
axiom period_f : ∀ x : ℝ, f(x + 4) = f(x)
axiom odd_fx_minus_1 : ∀ x : ℝ, f(-x+1) = -f(x-1)
axiom f_one : f(1) = 1

-- Theorem to prove
theorem calculate_f7_plus_f9 : f(7) + f(9) = 1 :=
by sorry

end calculate_f7_plus_f9_l609_609528


namespace proof_x_plus_y_sum_l609_609305

noncomputable def x_and_y_sum (x y : ℝ) : Prop := 31.25 / x = 100 / 9.6 ∧ 13.75 / x = y / 9.6

theorem proof_x_plus_y_sum (x y : ℝ) (h : x_and_y_sum x y) : x + y = 47 :=
sorry

end proof_x_plus_y_sum_l609_609305


namespace image_of_center_of_circle_T_after_transformations_l609_609252

def circle_center : ℝ × ℝ := (-2, 6)

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.fst, -p.snd)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.fst + d, p.snd)

theorem image_of_center_of_circle_T_after_transformations :
  let reflected_center := reflect_across_x_axis circle_center in
  let translated_center := translate_right reflected_center 5 in
  translated_center = (3, -6) :=
by
  sorry

end image_of_center_of_circle_T_after_transformations_l609_609252


namespace percentage_girls_own_cat_l609_609005

-- Definitions extracted from the problem conditions
def total_students := 30
def fraction_boys := 1 / 3
def girls_own_dogs_fraction := 0.40
def girls_no_pet := 8

-- Declaring the proof statement
theorem percentage_girls_own_cat :
  let total_girls := total_students - nat.ceil(fraction_boys * total_students) in
  let girls_own_dogs := nat.floor(girls_own_dogs_fraction * total_girls) in
  let girls_own_cat := total_girls - (girls_own_dogs + girls_no_pet) in
  let percentage_girls_own_cat := (girls_own_cat * 100) / total_girls in
  percentage_girls_own_cat = 20 :=
by
  -- Placeholder for the proof
  sorry

end percentage_girls_own_cat_l609_609005


namespace number_to_add_l609_609165

theorem number_to_add (a m : ℕ) (h₁ : a = 7844213) (h₂ : m = 549) :
  ∃ n, (a + n) % m = 0 ∧ n = m - (a % m) :=
by
  sorry

end number_to_add_l609_609165


namespace cone_surface_area_ratio_l609_609519

noncomputable def sector_angle := 135
noncomputable def sector_area (B : ℝ) := B
noncomputable def cone (A : ℝ) (B : ℝ) := A

theorem cone_surface_area_ratio (A B : ℝ) (h_sector_angle: sector_angle = 135) (h_sector_area: sector_area B = B) (h_cone_formed: cone A B = A) :
  A / B = 11 / 8 :=
by
  sorry

end cone_surface_area_ratio_l609_609519


namespace marcel_potatoes_eq_l609_609063

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end marcel_potatoes_eq_l609_609063


namespace projection_abs_value_is_3_l609_609799

theorem projection_abs_value_is_3
  (A B : ℝ×ℝ)
  (h₁ : line_through (1, 0) = ⟨A, B⟩)
  (h₂ : circle_radius_eq_center_distance (A, B) (0, 0) 2)
  (h₃ : line_circle_intersection (A, B) (1, -√3, 2))
  : | (fst B - fst A) | = 3 :=
sorry

end projection_abs_value_is_3_l609_609799


namespace min_distance_origin_to_line_l609_609421

theorem min_distance_origin_to_line : 
  ∃ P : ℝ × ℝ, (P.1, P.2) ∈ { p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 4 = 0 } ∧ 
  ∀ Q : ℝ × ℝ, (Q.1, Q.2) ∈ { q : ℝ × ℝ | 3 * q.1 + 4 * q.2 - 4 = 0 } → 
  (0 - Q.1)^2 + (0 - Q.2)^2 ≥ (0 - P.1)^2 + (0 - P.2)^2 :=
begin
  sorry
end

end min_distance_origin_to_line_l609_609421


namespace problem_solution_l609_609772

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l609_609772


namespace probability_of_x_gt_2y_l609_609068

open Set

def rectangular_region : Set (ℝ × ℝ) := 
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2010 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2011}

def probability_x_gt_2y : ℝ :=
  let triangular_region_area := (1/2) * 2010 * 1005
  let rectangular_region_area := 2010 * 2011
  triangular_region_area / rectangular_region_area

theorem probability_of_x_gt_2y :
  probability_x_gt_2y = 1005 / 2011 := by
  sorry

end probability_of_x_gt_2y_l609_609068


namespace constant_term_expansion_l609_609341

theorem constant_term_expansion :
  (∑ k in Finset.range (5 + 1), Nat.choose 5 k * (2 ^ k)) = 243 →
  ∃ k, Nat.choose 5 k * (2 ^ k) = 80 :=
by
  sorry

end constant_term_expansion_l609_609341


namespace purchase_price_of_article_l609_609960

theorem purchase_price_of_article (P M : ℝ) (h1 : M = 55) (h2 : M = 0.30 * P + 12) : P = 143.33 :=
  sorry

end purchase_price_of_article_l609_609960


namespace problem_statement_l609_609979

-- Definitions of the propositions and their converses
def prop1 (a b : ℝ) : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0
def conv_prop1 (a b : ℝ) : Prop := (a = 0 ∧ b = 0) → a^2 + b^2 = 0

def prop2 : Prop := ∀ (T1 T2 : Triangle), Congruent T1 T2 → Area T1 = Area T2
def neg_prop2 : Prop := ¬ (∀ (T1 T2 : Triangle), Congruent T1 T2 → Area T1 = Area T2)

def prop3 (q : ℝ) (x : ℝ) : Prop := q ≤ 1 → ∃ (x : ℝ), x^2 + 2*x + q = 0
def conv_prop3 (q : ℝ) : Prop := (∃ (x : ℝ), x^2 + 2*x + q = 0) → q ≤ 1

def prop4 (R : Rectangle) : Prop := DiagonalsEqual R
def conv_prop4 (Q : Quadrilateral) : Prop := DiagonalsEqual Q → IsRectangle Q

-- The main statement asserting the truth of specific propositions
theorem problem_statement (a b q x : ℝ) (T1 T2 : Triangle) (R : Rectangle) (Q : Quadrilateral) : 
  (prop1 a b) ∧ (conv_prop1 a b) ∧ (¬ neg_prop2) ∧ (conv_prop3 q) ∧ (¬ conv_prop4 Q) → 
  (conv_prop1 a b) ∧ (conv_prop3 q) :=
by
  intros h
  -- Placeholder, actual proof logic goes here
  sorry

end problem_statement_l609_609979


namespace period_of_f_f_on_interval_2_to_4_l609_609878

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_add_f_plus_three (x : ℝ) : f(x) + f(x + 3) = 0

axiom f_on_interval (x : ℝ) (h : -1 < x ∧ x ≤ 1) : f(x) = 2 * x - 3

-- Proof Problems

-- 1. Prove \( f(x) \) has a period of 6
theorem period_of_f : ∃ T > 0, ∀ x : ℝ, f(x + T) = f(x) := 
by {
  use 6,
  split,
  { linarith, },
  { intro x,
    sorry 
  } 
}

-- 2. Prove that when \( 2 < x ≤ 4 \), \( f(x) = -2x + 9 \)
theorem f_on_interval_2_to_4 (x : ℝ) (hx : 2 < x ∧ x ≤ 4) : f(x) = -2 * x + 9 :=
by {
  sorry
}

end period_of_f_f_on_interval_2_to_4_l609_609878


namespace Hugo_rolls_7_given_he_wins_l609_609836

-- Definitions based on conditions
def number_of_players : ℕ := 6
def sides_of_die : ℕ := 8

-- Event Definitions
def H1_event (roll : ℕ) : Prop := roll = 7
def W_event (rolls : Vector ℕ number_of_players) : Prop :=
  let highest_roll := rolls.foldl max 0
  highest_roll > (rolls.dropn 1).foldl max 0

-- Main proof statement
theorem Hugo_rolls_7_given_he_wins (players_rolls : Vector ℕ number_of_players) (Hugo_rolls: players_rolls.head = 7) :
  probability (W_event players_rolls) (players_rolls.head = 7) = 8856 / 32768 :=
by
  -- skipping proof
  sorry

end Hugo_rolls_7_given_he_wins_l609_609836


namespace min_markdown_percentage_l609_609596

theorem min_markdown_percentage (x : ℤ) (h : (1.0 * (1 + 0.10) * (1 + 0.10) * (1 + 0.05)) * (1 - x / 100) = 1.0) : x = 22 :=
sorry

end min_markdown_percentage_l609_609596


namespace equal_numbers_product_l609_609509

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l609_609509


namespace lcm_proof_l609_609204

-- Define the given conditions
def ratio (A B : ℕ) := A = 3 * (B / 4)

-- Given conditions in the problem
def first_number := 45
def second_number := 60
def lcm_result := 180

-- The proof problem
theorem lcm_proof : ratio first_number second_number → ∃ (B : ℕ), lcm first_number B = lcm_result :=
by
  intros h
  use second_number
  rw [Nat.lcm_eq]
  simp
  apply Nat.eq_of_mul_eq_mul_right
  sorry

end lcm_proof_l609_609204


namespace correct_order_of_actions_l609_609988

-- Definitions based on the conditions
def actions : ℕ → String
| 1 => "tap"
| 2 => "pay online"
| 3 => "swipe"
| 4 => "insert into terminal"
| _ => "undefined"

def paymentTechnology : ℕ → String
| 1 => "PayPass"
| 2 => "CVC"
| 3 => "magnetic stripe"
| 4 => "chip"
| _ => "undefined"

-- Proof problem statement
theorem correct_order_of_actions :
  (actions 4 = "insert into terminal") ∧
  (actions 3 = "swipe") ∧
  (actions 1 = "tap") ∧
  (actions 2 = "pay online") →
  [4, 3, 1, 2] corresponds to ["chip", "magnetic stripe", "PayPass", "CVC"]
:=
by
  sorry

end correct_order_of_actions_l609_609988


namespace determine_a_l609_609526

theorem determine_a
  (a b : ℝ)
  (P1 P2 : ℝ × ℝ)
  (direction_vector : ℝ × ℝ)
  (h1 : P1 = (-3, 4))
  (h2 : P2 = (4, -1))
  (h3 : direction_vector = (4 - (-3), -1 - 4))
  (h4 : b = a / 2)
  (h5 : direction_vector = (7, -5)) :
  a = -10 :=
sorry

end determine_a_l609_609526


namespace kitten_problem_solution_l609_609918

def kitten_arrangement : Prop :=
  ∃ (A B C D E F : ℝ × ℝ),
    A = (0, 0) ∧ B = (1, 0) ∧ C = (0.5, 0.5 * Real.sqrt 3) ∧
    D = (0.5, 0) ∧ E = (0.75, 0.25 * Real.sqrt 3) ∧ F = (0.25, 0.25 * Real.sqrt 3) ∧
    -- Rows of 3 kittens
    ({A, D, B} ∨ {B, E, C} ∨ {C, F, A}).card = 3 ∧
    -- Rows of 2 kittens
    ({A, D} ∨ {D, B} ∨ {B, E} ∨ {E, C} ∨ {C, F} ∨ {F, A}).card = 2

theorem kitten_problem_solution : kitten_arrangement :=
  sorry

end kitten_problem_solution_l609_609918


namespace total_cost_l609_609415

-- Definitions based on conditions
variables (x y : ℕ) -- assuming prices are natural numbers for simplicity

-- Problem statement
theorem total_cost (x y : ℕ) : 
  let discounted_price := x - y,
      total_cost := 5 * discounted_price in
  total_cost = 5 * (x - y) := 
by
  -- The proof itself is omitted as per instructions
  sorry

end total_cost_l609_609415


namespace factorize_xy_squared_minus_x_l609_609703

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609703


namespace mutually_exclusive_not_complementary_l609_609271

-- Define the people
inductive Person
| A 
| B 
| C

open Person

-- Define the colors
inductive Color
| Red
| Yellow
| Blue

open Color

-- Event A: Person A gets the Red card
def event_a (assignment: Person → Color) : Prop := assignment A = Red

-- Event B: Person B gets the Red card
def event_b (assignment: Person → Color) : Prop := assignment B = Red

-- Definition of mutually exclusive events
def mutually_exclusive (P Q: Prop): Prop := P → ¬Q

-- Definition of complementary events
def complementary (P Q: Prop): Prop := P ↔ ¬Q

theorem mutually_exclusive_not_complementary :
  ∀ (assignment: Person → Color),
  mutually_exclusive (event_a assignment) (event_b assignment) ∧ ¬complementary (event_a assignment) (event_b assignment) :=
by
  sorry

end mutually_exclusive_not_complementary_l609_609271


namespace constant_term_expansion_l609_609340

theorem constant_term_expansion :
  (∑ k in Finset.range (5 + 1), Nat.choose 5 k * (2 ^ k)) = 243 →
  ∃ k, Nat.choose 5 k * (2 ^ k) = 80 :=
by
  sorry

end constant_term_expansion_l609_609340


namespace instructors_meeting_l609_609076

theorem instructors_meeting (R P E M : ℕ) (hR : R = 5) (hP : P = 8) (hE : E = 10) (hM : M = 9) :
  Nat.lcm (Nat.lcm R P) (Nat.lcm E M) = 360 :=
by
  rw [hR, hP, hE, hM]
  sorry

end instructors_meeting_l609_609076


namespace div_powers_half_div_powers_four_div_powers_three_x_div_powers_x_minus_one_l609_609552
noncomputable theory

-- Statement 1
theorem div_powers_half : 
  ((1 / 2)^5 / (1 / 2)^2 = (1 / 8)) := by
  sorry

-- Statement 2
theorem div_powers_four : 
  (4^3 / 4^5 = (1 / 16)) := by
  sorry

-- Statement 3
theorem div_powers_three_x (x : ℝ) : 
  (3^(x - 1) / 3^(3 * x - 4) = (1 / 27)) → x = 3 := by
  sorry

-- Statement 4
theorem div_powers_x_minus_one (x : ℝ) :
  ((x - 1)^(2 * x + 2) / (x - 1)^(x + 6) = 1) → 
  x = 4 ∨ x = 0 ∨ x = 2 := by
  sorry

end div_powers_half_div_powers_four_div_powers_three_x_div_powers_x_minus_one_l609_609552


namespace sum_of_two_digit_divisors_l609_609877

theorem sum_of_two_digit_divisors (d : ℕ) (h₁ : d > 0) (h₂ : 143 % d = 11) : 
  ∑ (x : ℕ) in (Finset.filter (λ x, 10 ≤ x ∧ x < 100) (Finset.divisors 132)), x = 67 := 
by 
  sorry

end sum_of_two_digit_divisors_l609_609877


namespace student_ages_inconsistent_l609_609409

theorem student_ages_inconsistent :
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  total_age_all_students < total_age_group1 + total_age_group2 + total_age_group3 :=
by {
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  have h₁ : total_age_all_students = 24 * 18 := rfl
  have h₂ : total_age_group1 = 6 * 16 := rfl
  have h₃ : total_age_group2 = 10 * 20 := rfl
  have h₄ : total_age_group3 = 7 * 22 := rfl
  have h₅ : 432 = 24 * 18 := by norm_num
  have h₆ : 96 = 6 * 16 := by norm_num
  have h₇ : 200 = 10 * 20 := by norm_num
  have h₈ : 154 = 7 * 22 := by norm_num
  have h₉ : 432 < 96 + 200 + 154 := by norm_num
  exact h₉
}

end student_ages_inconsistent_l609_609409


namespace follow_pierre_advice_better_than_random_l609_609585

-- Define the probabilities associated with Pierre's accuracy and behavior
def probability_pierre_correct : ℚ := 3 / 4
def probability_pierre_incorrect : ℚ := 1 / 4

-- Calculate the probability of getting both dates correct if Jean guesses randomly
def probability_random_correct : ℚ := 1 / 4

-- Calculate the composite probabilities
def probability_both_correct_if_pierre_correct : ℚ := 
  probability_pierre_correct * probability_pierre_correct
def probability_both_correct_if_pierre_incorrect : ℚ := 
  probability_pierre_incorrect * probability_pierre_incorrect

-- Calculate the overall probability when following Pierre's advice
def probability_follow_pierre_correct : ℚ := 
  (probability_pierre_correct * probability_both_correct_if_pierre_correct) +
  (probability_pierre_incorrect * probability_both_correct_if_pierre_incorrect)

-- The theorem stating that following Pierre's advice gives a higher probability of correctness
theorem follow_pierre_advice_better_than_random :
  probability_follow_pierre_correct > probability_random_correct :=
by
  -- We provided the theorem statement which compares the two probabilities
  exact dec_trivial

end follow_pierre_advice_better_than_random_l609_609585


namespace probability_of_B_winning_is_correct_l609_609573

noncomputable def prob_A_wins : ℝ := 0.2
noncomputable def prob_draw : ℝ := 0.5
noncomputable def prob_B_wins : ℝ := 1 - (prob_A_wins + prob_draw)

theorem probability_of_B_winning_is_correct : prob_B_wins = 0.3 := by
  sorry

end probability_of_B_winning_is_correct_l609_609573


namespace factorize_xy_squared_minus_x_l609_609702

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609702


namespace factorize_expression_l609_609685

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609685


namespace proof_problem_l609_609307

structure Plane := (name : String)
structure Line := (name : String)

def parallel_planes (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def parallel_lines (m n : Line) : Prop := sorry

theorem proof_problem (m : Line) (α β : Plane) :
  parallel_planes α β → in_plane m α → parallel_lines m (Line.mk β.name) :=
sorry

end proof_problem_l609_609307


namespace roots_of_quadratic_are_integers_l609_609925

theorem roots_of_quadratic_are_integers
  (b c : ℤ)
  (Δ : ℤ)
  (h_discriminant: Δ = b^2 - 4 * c)
  (h_perfect_square: ∃ k : ℤ, k^2 = Δ)
  : (∃ x1 x2 : ℤ, x1 * x2 = c ∧ x1 + x2 = -b) :=
by
  sorry

end roots_of_quadratic_are_integers_l609_609925


namespace area_of_trapezoid_is_integer_l609_609420

/-- Given trapezoid ABCD with conditions:
    1. AB ⊥ BC
    2. BC ⊥ CD
    3. BC is tangent to the circle with center O and diameter AD
    Prove that the area of trapezoid ABCD is an integer when AB = 12 and CD = 3. -/
theorem area_of_trapezoid_is_integer (AB CD BC : ℝ) (h1 : AB = 12) (h2 : CD = 3) (h3 : BC = sqrt (AB * CD))
  (h4 : (AB+CD) * BC / 2 = (AB + CD) * BC / 2 ∈ {n : ℝ // is_integer n}) : 
  is_integer ((AB + CD) * BC / 2) :=
sorry

end area_of_trapezoid_is_integer_l609_609420


namespace book_pages_count_l609_609129

theorem book_pages_count (digits_used : ℕ) (h_digits_used : digits_used = 3289)
    (h_pages1_9 : ∀ n, 1 ≤ n ∧ n ≤ 9 → n.toString.length = 1)
    (h_pages10_99 : ∀ n, 10 ≤ n ∧ n ≤ 99 → n.toString.length = 2)
    (h_pages100_999 : ∀ n, 100 ≤ n ∧ n ≤ 999 → n.toString.length = 3)
    (h_pages1000_plus : ∀ n, 1000 ≤ n → n.toString.length = 4) : 
    ∃ pages, pages = 1099 ∧ 
        let digits1_9 := (9) * 1 in
        let digits10_99 := (99 - 10 + 1) * 2 in
        let digits100_999 := (999 - 100 + 1) * 3 in
        let digits1_999 := digits1_9 + digits10_99 + digits100_999 in
        let remaining_digits := digits_used - digits1_999 in
        let additional_pages := remaining_digits / 4 in
        pages = 999 + additional_pages :=
by
    use 1099
    sorry

end book_pages_count_l609_609129


namespace total_pupils_count_l609_609414

theorem total_pupils_count (girls boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) : girls + boys = 926 :=
by 
  sorry

end total_pupils_count_l609_609414


namespace seating_5_out_of_6_around_circle_l609_609164

def number_of_ways_to_seat_5_out_of_6_in_circle : Nat :=
  Nat.factorial 4

theorem seating_5_out_of_6_around_circle : number_of_ways_to_seat_5_out_of_6_in_circle = 24 :=
by {
  -- proof would be here
  sorry
}

end seating_5_out_of_6_around_circle_l609_609164


namespace number_of_lines_through_P_is_2_l609_609357

open Set

variables {α : Type*} [Inhabited α] [NormedSpace ℝ α]

def line_through_point (a b : α) (P : α) (angle_a : ℝ) (angle_b : ℝ) : Prop :=
  let S := sphere P 1 in
  let C_a := {x ∈ S | angle x a = angle_a} in
  let C_b := {x ∈ S | angle x b = angle_b} in
  ∃ (n : ℕ), n = 2 ∧ ∀ x, x ∈ C_a ∩ C_b

noncomputable def number_of_lines_through_P (a b P : α) (angle_between_ab angle_with_a angle_with_b : ℝ) : ℕ :=
  if h : angle_between_ab = 50 ∧ angle_with_a = 30 ∧ angle_with_b = 30 then 2 else 0

theorem number_of_lines_through_P_is_2 {a b P : α} (h1 : ¬coplanar {a, b})
    (h2 : ∃ θ : ℝ, θ = 50) 
    (h3 : angle a P = 30)
    (h4 : angle b P = 30) :
  (number_of_lines_through_P a b P 50 30 30) = 2 :=
by
  sorry

end number_of_lines_through_P_is_2_l609_609357


namespace average_of_first_100_terms_l609_609256

-- Define the sequence a_n = (-1)^n * n^2
def a_n (n : ℕ) : ℤ := (-1) ^ n * (n ^ 2)

-- Statement of the problem
theorem average_of_first_100_terms :
  let seq := λ (n : ℕ), a_n n in
  let avg := (∑ i in range 1 101, seq i) / 100 in
  avg = -49.5 := sorry

end average_of_first_100_terms_l609_609256


namespace part1_div1_part1_div2_part2_div_part3_div_l609_609551

theorem part1_div1 {a : ℚ} (h : a ≠ 0) (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : a = 1/2) :
  (a ^ 5) / (a ^ 2) = 1 / 8 := sorry

theorem part1_div2 {a : ℚ} (h : a ≠ 0) (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : a = 4) :
  (a ^ 3) / (a ^ 5) = 1 / 16 := sorry

theorem part2_div {x : ℚ} (h : 3 ^ (x - 1) / 3 ^ (3 * x - 4) = 1 / 27) :
  x = 3 := sorry

theorem part3_div {x : ℚ} (h : (x - 1) ^ (2 * x + 2) / (x - 1) ^ (x + 6) = 1) :
  x = 4 ∨ x = 0 ∨ x = 2 := sorry

end part1_div1_part1_div2_part2_div_part3_div_l609_609551


namespace gain_percent_correct_l609_609159

variable (CP SP Gain : ℝ)
variable (H₁ : CP = 900)
variable (H₂ : SP = 1125)
variable (H₃ : Gain = SP - CP)

theorem gain_percent_correct : (Gain / CP) * 100 = 25 :=
by
  sorry

end gain_percent_correct_l609_609159


namespace directional_derivative_at_point_l609_609713

def function_z (x y : ℝ) : ℝ := x^2 + y^2

def point_M := (3 : ℝ, 1 : ℝ)
def point_M1 := (0 : ℝ, 5 : ℝ)

def direction_vector (M M1 : ℝ × ℝ) : ℝ × ℝ :=
  (M1.1 - M.1, M1.2 - M.2)

noncomputable def directional_derivative 
  (f : ℝ → ℝ → ℝ) (M M1 : ℝ × ℝ) : ℝ :=
  let l := direction_vector M M1 in
  let magnitude := Real.sqrt ((l.1)^2 + (l.2)^2) in
  let unit_vector := (l.1 / magnitude, l.2 / magnitude) in
  let partial_x := (f M.1 M.2).partialDifferential M.1 in
  let partial_y := (f M.1 M.2).partialDifferential M.2 in
  partial_x * unit_vector.1 + partial_y * unit_vector.2

theorem directional_derivative_at_point 
  : directional_derivative function_z point_M point_M1 = -2 := 
sorry

end directional_derivative_at_point_l609_609713


namespace boat_speed_in_still_water_l609_609016

/-- In one hour, a boat goes 9 km along the stream and 5 km against the stream.
Prove that the speed of the boat in still water is 7 km/hr. -/
theorem boat_speed_in_still_water (B S : ℝ) 
  (h1 : B + S = 9) 
  (h2 : B - S = 5) : 
  B = 7 :=
by
  sorry

end boat_speed_in_still_water_l609_609016


namespace shirts_per_minute_l609_609235

/--
An industrial machine made 8 shirts today and worked for 4 minutes today. 
Prove that the machine can make 2 shirts per minute.
-/
theorem shirts_per_minute (shirts_today : ℕ) (minutes_today : ℕ)
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  (shirts_today / minutes_today) = 2 :=
by sorry

end shirts_per_minute_l609_609235


namespace math_problem_smaller_root_l609_609716

noncomputable def findSmallerRoot (x : ℚ) : Prop :=
  (x - 1/3)^2 + (x - 1/3)*(x + 1/6) = 0 ∧ x = 1/12

theorem math_problem_smaller_root : ∃ x : ℚ, findSmallerRoot x :=
by
  use 1/12
  unfold findSmallerRoot
  split
  -- prove the first part that the equation equals zero
  sorry
  -- prove the second part that x equals 1/12
  ring

end math_problem_smaller_root_l609_609716


namespace crop_planting_ways_l609_609191

-- Definitions of the sections S1, S2, S3, S4 representing the grid
inductive Section
| S1 | S2 | S3 | S4

-- Definitions of the crops
inductive Crop
| Orange | Apple | Pear | Cherry

-- Predicate to determine adjacency
def adjacent : Section → Section → Prop
| Section.S1 Section.S2 := true
| Section.S1 Section.S3 := true
| Section.S2 Section.S1 := true
| Section.S2 Section.S4 := true
| Section.S3 Section.S1 := true
| Section.S3 Section.S4 := true
| Section.S4 Section.S2 := true
| Section.S4 Section.S3 := true
| _ _ := false

-- Predicate to determine diagonal relationships
def diagonal : Section → Section → Prop
| Section.S1 Section.S4 := true
| Section.S4 Section.S1 := true
| Section.S2 Section.S3 := true
| Section.S3 Section.S2 := true
| _ _ := false

-- Main theorem statement
theorem crop_planting_ways :
  ∃ (ways : ℕ), ways = 12 ∧
  ∀ (f : Section → Crop),
    (adjacent Section.S1 Section.S2 → (f Section.S1 = Crop.Orange ∨ f Section.S1 = Crop.Pear) → (f Section.S2 ≠ Crop.Orange ∧ f Section.S2 ≠ Crop.Pear)) ∧
    (adjacent Section.S1 Section.S3 → (f Section.S1 = Crop.Orange ∨ f Section.S1 = Crop.Pear) → (f Section.S3 ≠ Crop.Orange ∧ f Section.S3 ≠ Crop.Pear)) ∧
    (adjacent Section.S2 Section.S1 → (f Section.S2 = Crop.Orange ∨ f Section.S2 = Crop.Pear) → (f Section.S1 ≠ Crop.Orange ∧ f Section.S1 ≠ Crop.Pear)) ∧
    (adjacent Section.S2 Section.S4 → (f Section.S2 = Crop.Orange ∨ f Section.S2 = Crop.Pear) → (f Section.S4 ≠ Crop.Orange ∧ f Section.S4 ≠ Crop.Pear)) ∧
    (adjacent Section.S3 Section.S1 → (f Section.S3 = Crop.Orange ∨ f Section.S3 = Crop.Pear) → (f Section.S1 ≠ Crop.Orange ∧ f Section.S1 ≠ Crop.Pear)) ∧
    (adjacent Section.S3 Section.S4 → (f Section.S3 = Crop.Orange ∨ f Section.S3 = Crop.Pear) → (f Section.S4 ≠ Crop.Orange ∧ f Section.S4 ≠ Crop.Pear)) ∧
    (adjacent Section.S4 Section.S2 → (f Section.S4 = Crop.Orange ∨ f Section.S4 = Crop.Pear) → (f Section.S2 ≠ Crop.Orange ∧ f Section.S2 ≠ Crop.Pear)) ∧
    (adjacent Section.S4 Section.S3 → (f Section.S4 = Crop.Orange ∨ f Section.S4 = Crop.Pear) → (f Section.S3 ≠ Crop.Orange ∧ f Section.S3 ≠ Crop.Pear)) ∧
    (diagonal Section.S1 Section.S4 → (f Section.S1 = Crop.Apple ∨ f Section.S1 = Crop.Cherry) → (f Section.S4 ≠ Crop.Apple ∧ f Section.S4 ≠ Crop.Cherry)) ∧
    (diagonal Section.S2 Section.S3 → (f Section.S2 = Crop.Apple ∨ f Section.S2 = Crop.Cherry) → (f Section.S3 ≠ Crop.Apple ∧ f Section.S3 ≠ Crop.Cherry)) := sorry

end crop_planting_ways_l609_609191


namespace factorization_l609_609695

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l609_609695


namespace tommy_saw_100_wheels_l609_609131

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end tommy_saw_100_wheels_l609_609131


namespace arithmetic_calculation_l609_609243

theorem arithmetic_calculation : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 :=
by
  sorry

end arithmetic_calculation_l609_609243


namespace round_robin_tournament_matches_l609_609614

theorem round_robin_tournament_matches (n : Nat) (hn : n = 8) :
  (n * (n - 1)) / 2 = 28 :=
by
  rw [hn]
  simp
  norm_num
  sorry

end round_robin_tournament_matches_l609_609614


namespace geom_seq_value_a5a6a7_l609_609845

theorem geom_seq_value_a5a6a7 (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_eq_roots : ∃ x1 x2 : ℝ, (3 * x1^2 - 11 * x1 + 9 = 0) ∧ (3 * x2^2 - 11 * x2 + 9 = 0) ∧ a 3 = x1 ∧ a 9 = x2):
   a 5 * a 6 * a 7 = 3 * real.sqrt 3 ∨ a 5 * a 6 * a 7 = -3 * real.sqrt 3 :=
by
  sorry

end geom_seq_value_a5a6a7_l609_609845


namespace two_dice_even_product_prob_l609_609150

theorem two_dice_even_product_prob :
  let outcomes := (fin 6) × (fin 6) in
  let total_outcomes := fintype.card outcomes in
  let even_product_outcomes := {outcome : outcomes | (outcome.fst + 1) * (outcome.snd + 1) % 2 = 0} in
  (fintype.card even_product_outcomes) / total_outcomes = 3 / 4 :=
begin
  -- Proof here
  sorry
end

end two_dice_even_product_prob_l609_609150


namespace num_integer_side_length_triangles_l609_609358

open Real

theorem num_integer_side_length_triangles : 
  ∃ (n : ℕ), 
  (n = 8) ∧ 
  ∀ (a b c : ℕ), 
    (a + b > c + 5) ∧ 
    (b + c > a + 5) ∧ 
    (c + a > b + 5) →
    let s := (a + b + c) / 2 in
    let A := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
    A = 2 * (a + b + c) :=
sorry

end num_integer_side_length_triangles_l609_609358


namespace corrected_multiplication_result_l609_609158

theorem corrected_multiplication_result :
  ∃ n : ℕ, 987 * n = 559989 ∧ 987 * n ≠ 559981 ∧ 559981 % 100 = 98 :=
by
  sorry

end corrected_multiplication_result_l609_609158


namespace tangents_isosceles_circle_ratio_l609_609069

theorem tangents_isosceles_circle_ratio (A C O B D : Point)
  (h_circle : Circle O A C)
  (h_isosceles : ∠ BAC = 45 ∧ ∠ BCA = 45)
  (h_tangent_BA : Tangent B A O)
  (h_tangent_BC : Tangent B C O)
  (h_D_on_BO : D ∈ LineSegment O B)
  : BD / BO = 1 - Real.sin (22.5 * Real.pi / 180) :=
by
  sorry

end tangents_isosceles_circle_ratio_l609_609069


namespace problem_statement_l609_609232

noncomputable def A := "Taller students in a class"
noncomputable def B := "Long-lived people"
noncomputable def C := "Approximate values of √2"
noncomputable def D := { x : ℝ | x = 1 ∨ x = -1 }

theorem problem_statement : D = { x : ℝ | x.recip = x } :=
by
  sorry

end problem_statement_l609_609232


namespace solution_pairs_l609_609708

theorem solution_pairs (x y : ℝ) (h1 : 3 * y - sqrt (y / x) - 6 * sqrt (x * y) + 2 = 0)
  (h2 : x^2 + 81 * x^2 * y^4 = 2 * y^2) (hx_pos : x > 0) (hy_pos : y > 0) :
  (x = (31 ^ (1 / 4)) / 12 ∧ y = (31 ^ (1 / 4)) / 3) ∨ (x = 1 / 3 ∧ y = 1 / 3) :=
sorry

end solution_pairs_l609_609708


namespace max_value_carry_l609_609250

structure Rock :=
(weight : ℕ)
(value : ℕ)

def stones : List Rock :=
  [ {weight := 6, value := 18},
    {weight := 3, value := 9},
    {weight := 2, value := 5} ]

def maxWeight : ℕ := 24

def max_value (maxWeight : ℕ) (stones : List Rock) : ℕ :=
  -- Skip computation, as proof is not required
  72 -- The maximum value as given by the problem statement

theorem max_value_carry (h : list.sum (stones.map Rock.weight) ≤ maxWeight) :
  max_value maxWeight stones = 72 :=
by sorry

end max_value_carry_l609_609250


namespace ainsley_wins_100a_plus_b_eq_109_l609_609626

theorem ainsley_wins_100a_plus_b_eq_109 : 
  let a b : ℕ := 1, 9,
  100 * a + b = 109 :=
by
  sorry

end ainsley_wins_100a_plus_b_eq_109_l609_609626


namespace marble_weight_l609_609916

theorem marble_weight (W : ℝ) (h : 2 * W + 0.08333333333333333 = 0.75) : 
  W = 0.33333333333333335 := 
by 
  -- Skipping the proof as specified
  sorry

end marble_weight_l609_609916


namespace vanya_correct_answers_l609_609482

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l609_609482


namespace least_number_subtracted_divisible_by_5_l609_609569

def subtract_least_number (n : ℕ) (m : ℕ) : ℕ :=
  n % m

theorem least_number_subtracted_divisible_by_5 : subtract_least_number 9671 5 = 1 :=
by
  sorry

end least_number_subtracted_divisible_by_5_l609_609569


namespace amount_spent_on_milk_l609_609224

theorem amount_spent_on_milk (rent groceries education petrol misc savings milk total_salary: ℕ)
  (h_rent: rent = 5000)
  (h_groceries: groceries = 4500)
  (h_education: education = 2500)
  (h_petrol: petrol = 2000)
  (h_misc: misc = 6100)
  (h_savings_pct: 10% of total_salary = savings)
  (h_savings: savings = 2400)
  (h_total_salary: total_salary = (2400 / 0.10).nat_abs) :
  milk = total_salary - (rent + groceries + education + petrol + misc + savings) →
  milk = 1500 := 
by
  sorry

end amount_spent_on_milk_l609_609224


namespace angle_B_equals_triangle_area_l609_609828

-- Definitions based on conditions
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def cosine_ratio (B C b a c : ℝ) : Prop :=
  cos B / cos C = b / (2 * a + c)

-- The given math problem translated to Lean statements
theorem angle_B_equals (a b c A B C : ℝ)
  (h_sides : triangle_sides a b c A B C)
  (h_ratio : cosine_ratio B C b a c) :
  B = 2 * π / 3 :=
sorry

theorem triangle_area (a b c A B C : ℝ)
  (h_sides : triangle_sides a b c A B C)
  (h_ratio : cosine_ratio B C b a c)
  (hb : b = sqrt 13)
  (hac : a + c = 4)
  (hB : B = 2 * π / 3) :
  (1/2) * a * c * sin B = (3 * sqrt 3) / 4 :=
sorry

end angle_B_equals_triangle_area_l609_609828


namespace sum_consecutive_odd_l609_609568

theorem sum_consecutive_odd (n : ℕ) (h : n = 25) : 
  (finset.Ico 1 (n+1)).filter (λ x, x % 2 = 1).sum id = 169 :=
by
  -- h will be necessary in the proof steps
  -- Calculations steps are omitted as proof is not required
  sorry

end sum_consecutive_odd_l609_609568


namespace chord_length_of_tangent_lines_intersection_l609_609535

open Real

theorem chord_length_of_tangent_lines_intersection 
  {O A B M : Point} (hTangentA : is_tangent A O) (hTangentB : is_tangent B O) 
  (hIntersection : intersect A M B M) (hDivide : divide M O A B 2 18) :
  distance A B = 12 :=
by sorry

end chord_length_of_tangent_lines_intersection_l609_609535


namespace cistern_fill_time_l609_609185

theorem cistern_fill_time (hA : ℝ) (hB : ℝ) (hC : ℝ) : hA = 12 → hB = 18 → hC = 15 → 
  1 / ((1 / hA) + (1 / hB) - (1 / hC)) = 180 / 13 :=
by
  intros hA_eq hB_eq hC_eq
  rw [hA_eq, hB_eq, hC_eq]
  sorry

end cistern_fill_time_l609_609185


namespace count_ordered_triples_l609_609050

noncomputable def num_ordered_triples (p n : ℕ) (hn : n > 0) (hp : Prime p) 
  (hn_repr : ∀ i : ℕ, i ≤ t → (n.bit0 >> (i * p)) % p ≤ p - 1) : ℕ :=
  ∏ i in finset.range (t + 1), (binomial (n.bit0 >> (i * p) % p + 2) 2)

theorem count_ordered_triples {p n : ℕ} (hp : Prime p) (hn : n > 0) 
  (coeffs_cond : ∀ i : ℕ, i ≤ t → (n.bit0 >> (i * p)) % p ≤ p - 1) :
  ∃ S_n : set (ℕ × ℕ × ℕ), (∀ (a b c : ℕ), (a, b, c) ∈ S_n ↔ a + b + c = n ∧ p ∤ (n.factorial / ((a.factorial * b.factorial) * c.factorial)).natAbs) ∧
    S_n.count = num_ordered_triples p n coeffs_cond :=
  sorry

end count_ordered_triples_l609_609050


namespace fraction_simplification_l609_609643

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (2 * x - 5) / (x ^ 2 - 1) + 3 / (1 - x) = - (x + 8) / (x ^ 2 - 1) :=
  sorry

end fraction_simplification_l609_609643


namespace vanya_correct_answers_l609_609470

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609470


namespace johns_average_speed_l609_609030

def total_distance := 2 + 2
def total_time_minutes := 20 + 5 + 10
def total_time_hours := (total_time_minutes / 60 : ℝ)
def average_speed := total_distance / total_time_hours

theorem johns_average_speed :
  average_speed = (48 / 7 : ℝ) := by
  sorry

end johns_average_speed_l609_609030


namespace expand_expression_l609_609277

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l609_609277


namespace square_diagonals_properties_l609_609617

theorem square_diagonals_properties :
  ∀ (A B C D : Type) [square A B C D] [rect A B C D] [rhombus A B C D],
  bisect diags A B C D ∧ equal diags A B C D ∧ perp diags A B C D :=
by
  -- Definitions and conditions can be introduced here, but the proof is skipped
  sorry

end square_diagonals_properties_l609_609617


namespace largest_divisor_for_odd_n_l609_609566

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem largest_divisor_for_odd_n (n : ℤ) (h : is_odd n ∧ n > 0) : 
  15 ∣ (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) := 
by 
  sorry

end largest_divisor_for_odd_n_l609_609566


namespace mathematician_coffee_consumption_correct_coffee_consumption_on_tuesday_l609_609607

-- Define all conditions
def k := 32

-- Define the relationship modeled previously
theorem mathematician_coffee_consumption (h w g : ℝ) (k : ℝ) : 
  (ghw = k) :=
begin
  sorry
end

-- Given Monday's data
def monday_data := (8 : ℝ, 4 : ℝ, 1 : ℝ)

-- Given Tuesday's data to find g
def tuesday_data (h w : ℝ) : ℝ := k / (h * w)

-- Given data to the relationships and proving it
theorem correct_coffee_consumption_on_tuesday :
  (tuesday_data 5 7 = 32 / 35) :=
begin
  sorry
end

end mathematician_coffee_consumption_correct_coffee_consumption_on_tuesday_l609_609607


namespace decagon_interior_angle_l609_609953

theorem decagon_interior_angle (n : ℕ) (h₁ : n = 10) :
  (180 * (n - 2) / n : ℝ) = 144 :=
by
  have h₂ : (n = 10) := h₁,
  rw [h₂],
  norm_num -- This simplifies the calculation

end decagon_interior_angle_l609_609953


namespace expenditure_representation_l609_609650

theorem expenditure_representation (income expenditure : ℤ)
  (h_income : income = 60)
  (h_expenditure : expenditure = 40) :
  -expenditure = -40 :=
by {
  sorry
}

end expenditure_representation_l609_609650


namespace total_growing_space_correct_l609_609631

-- Define the dimensions of the garden beds
def length_bed1 : ℕ := 3
def width_bed1 : ℕ := 3
def num_bed1 : ℕ := 2

def length_bed2 : ℕ := 4
def width_bed2 : ℕ := 3
def num_bed2 : ℕ := 2

-- Define the areas of the individual beds and total growing space
def area_bed1 : ℕ := length_bed1 * width_bed1
def total_area_bed1 : ℕ := area_bed1 * num_bed1

def area_bed2 : ℕ := length_bed2 * width_bed2
def total_area_bed2 : ℕ := area_bed2 * num_bed2

def total_growing_space : ℕ := total_area_bed1 + total_area_bed2

-- The theorem proving the total growing space
theorem total_growing_space_correct : total_growing_space = 42 := by
  sorry

end total_growing_space_correct_l609_609631


namespace range_of_a_l609_609748

noncomputable def A : set ℝ := {x | x < -4 ∨ x > 2}
noncomputable def B (a : ℝ) : set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ∪ B a = set.univ) : a > 2 :=
by
  sorry

end range_of_a_l609_609748


namespace raduzhny_population_l609_609424

theorem raduzhny_population 
  (pop_znoynoe : Nat)
  (pop_diff_avg : Nat)
  (total_villages : Nat)
  (max_diff : Nat) 
  (raduzhny_pop : Nat)
  (h1 : pop_znoynoe = 1000)
  (h2 : pop_diff_avg = 90)
  (h3 : total_villages = 10)
  (h4 : max_diff = 100)
  (h5 : ∀ n (n ∈ {1, 2, ..., total_villages-1}), n ≠ pop_znoynoe → abs (n - pop_znoynoe) ≤ max_diff)
  (h6 : raduzhny_pop = 900) : 
  raduzhny_pop = 900 := by 
  sorry

end raduzhny_population_l609_609424


namespace find_length_KM_l609_609623

noncomputable def triangle_ABC := sorry

noncomputable def angle_bisector_intersects_BC_at_K (ABC : triangle_ABC) := sorry

def line_through_B_parallel_AC_intersects_AK_at_M (ABC : triangle_ABC) := sorry

theorem find_length_KM (ABC : triangle_ABC) (A B C K M : Point) 
  (hAB : distance A B = 4) 
  (hAC : distance A C = 2) 
  (hBC : distance B C = 3) 
  (hAK : ray A K = angle_bisector_intersects_BC_at_K ABC) 
  (hBM : line_through_B_parallel_AC_intersects_AK_at_M ABC) :
  distance K M = 2 * Real.sqrt 6 := 
  sorry

end find_length_KM_l609_609623


namespace extreme_values_of_f_l609_609894

-- Definitions based on conditions
def f (x : ℝ) : ℝ := log (4 * x) / log 2 * log (2 * x) / log 2
def t (x : ℝ) : ℝ := log x / log 2

-- Conditions
def domain := {x : ℝ | 1 / 4 ≤ x ∧ x ≤ 4}
def t_range := {t : ℝ | -2 ≤ t ∧ t ≤ 2}

-- Lean 4 statement, no proof required
theorem extreme_values_of_f :
  (∀ x ∈ domain, t x ∈ t_range) →
  (∃ x₁ x₂ ∈ domain, f x₁ = 12 ∧ f x₂ = -1 / 4) :=
sorry

end extreme_values_of_f_l609_609894


namespace square_area_percentage_error_l609_609199

variable (L : ℝ)

def measured_length (L : ℝ) : ℝ := L + 0.03 * L
def measured_width (L : ℝ) : ℝ := L - 0.02 * L
def actual_area (L : ℝ) : ℝ := L * L
def measured_area (L : ℝ) : ℝ := (measured_length L) * (measured_width L)
def percentage_error (L : ℝ) : ℝ := ((measured_area L - actual_area L) / actual_area L) * 100

theorem square_area_percentage_error : percentage_error L = 0.94 := by
  sorry

end square_area_percentage_error_l609_609199


namespace picnic_total_cost_is_correct_l609_609119

-- Define the conditions given in the problem
def number_of_people : Nat := 4
def cost_per_sandwich : Nat := 5
def cost_per_fruit_salad : Nat := 3
def sodas_per_person : Nat := 2
def cost_per_soda : Nat := 2
def number_of_snack_bags : Nat := 3
def cost_per_snack_bag : Nat := 4

-- Calculate the total cost based on the given conditions
def total_cost_sandwiches : Nat := number_of_people * cost_per_sandwich
def total_cost_fruit_salads : Nat := number_of_people * cost_per_fruit_salad
def total_cost_sodas : Nat := number_of_people * sodas_per_person * cost_per_soda
def total_cost_snack_bags : Nat := number_of_snack_bags * cost_per_snack_bag

def total_spent : Nat := total_cost_sandwiches + total_cost_fruit_salads + total_cost_sodas + total_cost_snack_bags

-- The statement we want to prove
theorem picnic_total_cost_is_correct : total_spent = 60 :=
by
  -- Proof would be written here
  sorry

end picnic_total_cost_is_correct_l609_609119


namespace real_roots_condition_l609_609722

theorem real_roots_condition (k m : ℝ) (h : m ≠ 0) : (∃ x : ℝ, x^2 + k * x + m = 0) ↔ (m ≤ k^2 / 4) :=
by
  sorry

end real_roots_condition_l609_609722


namespace Vanya_correct_answers_l609_609477

theorem Vanya_correct_answers (x : ℕ) (total_questions : ℕ) (correct_candies : ℕ) (incorrect_candies : ℕ)
  (h1 : total_questions = 50)
  (h2 : correct_candies = 7)
  (h3 : incorrect_candies = 3)
  (h4 : 7 * x - 3 * (total_questions - x) = 0) :
  x = 15 :=
by
  rw [h1, h2, h3] at h4
  sorry

end Vanya_correct_answers_l609_609477


namespace problem_l609_609043

structure RectangularPrism (α : Type*) :=
(length width height : α)

def B : RectangularPrism ℝ :=
{ length := 1, width := 3, height := 4 }

def S (r : ℝ) (B : RectangularPrism ℝ) :=
  { p : ℝ × ℝ × ℝ | ∃ q ∈ B, (dist p q ≤ r) }

noncomputable def volume_expression (r : ℝ) :=
  let a := (4 * real.pi) / 3 in
  let b := 8 * real.pi in
  let c := 38 in
  let d := 12 in
  a * r ^ 3 + b * r ^ 2 + c * r + d

theorem problem (a b c d : ℝ) (ha : a = (4 * real.pi) / 3) (hb : b = 8 * real.pi)
  (hc : c = 38) (hd : d = 12) : (b * c) / (a * d) = 19 :=
by
  -- Placeholder for the proof.
  sorry

end problem_l609_609043


namespace other_root_of_quadratic_eq_l609_609913

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l609_609913


namespace preceding_binary_is_101001_l609_609362

def M_binary : ℕ := 0b101010  -- M in binary, where 0b prefix denotes binary

theorem preceding_binary_is_101001 :
  Nat.pred M_binary = 0b101001 := 
by
  -- Convert M from binary to decimal
  have M_decimal : M_binary = 42 := by norm_num
  -- Subtract 1 from the decimal representation
  have pred_decimal : Nat.pred 42 = 41 := by norm_num
  -- Convert the result back to binary and verify
  have pred_binary : 41 = 0b101001 := by norm_num
  -- Combine results to conclude the proof
  rw [M_decimal, pred_decimal, ← pred_binary]
  exact rfl

end preceding_binary_is_101001_l609_609362


namespace simplify_G_l609_609883

def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

def replace_x (x : ℝ) : ℝ := (2 * x - x^2) / (1 + 2 * x^2)

def G (x : ℝ) : ℝ := F (replace_x x)

theorem simplify_G (x : ℝ) : G x = 2 * F x := by
  sorry

end simplify_G_l609_609883


namespace range_of_a_l609_609355

-- Define the power function f with α = 3
def f (x : ℝ) : ℝ := x ^ 3

-- The condition given that f passes through the point (2,8)
axiom power_function_cond (α : ℝ) : 2 ^ α = 8

-- The final goal
theorem range_of_a (a : ℝ) (α : ℝ) (h : 2 ^ α = 8) : f(a + 1) < f(3 - 2 * a) ↔ a < (2 / 3) :=
by
  -- Proof will be placed here
  sorry

end range_of_a_l609_609355


namespace johns_elevation_after_descent_l609_609034

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end johns_elevation_after_descent_l609_609034


namespace area_triangle_AMC_l609_609018

open Real

-- Definitions: Define the points A, B, C, D such that they form a rectangle
-- Define midpoint M of \overline{AD}

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def A : Point := {x := 0, y := 0}
noncomputable def B : Point := {x := 6, y := 0}
noncomputable def D : Point := {x := 0, y := 8}
noncomputable def C : Point := {x := 6, y := 8}
noncomputable def M : Point := {x := 0, y := 4} -- midpoint of AD

-- Function to compute the area of triangle AMC
noncomputable def triangle_area (A M C : Point) : ℝ :=
  (1 / 2 : ℝ) * abs ((A.x - C.x) * (M.y - A.y) - (A.x - M.x) * (C.y - A.y))

-- The theorem to prove
theorem area_triangle_AMC : triangle_area A M C = 12 :=
by
  sorry

end area_triangle_AMC_l609_609018


namespace relationship_between_a_b_c_l609_609733

def a : ℝ := 0.5 ^ 2
def b : ℝ := 2 ^ 0.5
def c : ℝ := real.log 2 / real.log (0.5)

theorem relationship_between_a_b_c : c < a ∧ a < b :=
by {
  -- Calculate specific values
  let a_val := 0.25,
  let b_val := real.sqrt 2,
  let c_val := -1,

  -- State value properties
  have a_calc : a = a_val := by sorry,
  have b_calc : b = b_val := by sorry,
  have c_calc : c = c_val := by sorry,

  -- Establish final comparison
  show c < a ∧ a < b, from
  ⟨ by { rw c_calc, rw a_calc, norm_num, },
    by { rw a_calc, rw b_calc, norm_num, apply real.sqrt_pos.mpr, norm_num, } ⟩
}

end relationship_between_a_b_c_l609_609733


namespace sum_of_even_powers_equals_zero_l609_609045

theorem sum_of_even_powers_equals_zero :
  (∑ α in ({-2, -1, 1, 2} : Finset ℤ), if even α then α else 0) = 0 := 
by
  sorry  -- Proof is not required as per instructions.

end sum_of_even_powers_equals_zero_l609_609045


namespace inequality_abc_l609_609782

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l609_609782


namespace center_of_symmetry_correct_l609_609811

-- Define the given function
def f (x : ℝ) : ℝ := Real.cos (2*x + Real.pi/3)

-- Define the center of symmetry
def center_of_symmetry := (Real.pi/12, 0 : ℝ)

-- The goal is to prove that this point is the center of symmetry for the given function
theorem center_of_symmetry_correct : ∃ p : ℝ × ℝ, p = center_of_symmetry ∧ ∀ x : ℝ, f (p.1 + x) = f (p.1 - x) :=
by
  use center_of_symmetry
  sorry

end center_of_symmetry_correct_l609_609811


namespace S9_is_neg_54_l609_609434

variable {a_n : ℕ → ℤ}

-- Conditions
def a1 : ℤ := 2
def a5 : ℤ := 3 * (a_n 3)

theorem S9_is_neg_54 :
  a_n 1 = 2 →
  a_n 5 = 3 * a_n 3 →
  let d : ℤ := a_n 2 - a_n 1 in
  9 * a1 + (9 * 8 / 2) * d = -54 :=
by
  sorry

end S9_is_neg_54_l609_609434


namespace problem_solution_l609_609768

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l609_609768


namespace factorize_xy2_minus_x_l609_609678

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l609_609678


namespace inequality_solution_l609_609933

theorem inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l609_609933


namespace maximize_revenue_l609_609615

-- Defining the revenue function
def revenue (p : ℝ) : ℝ := 200 * p - 4 * p^2

-- Defining the maximum price constraint
def price_constraint (p : ℝ) : Prop := p ≤ 40

-- Statement to be proven
theorem maximize_revenue : ∃ (p : ℝ), price_constraint p ∧ revenue p = 2500 ∧ (∀ q : ℝ, price_constraint q → revenue q ≤ revenue p) :=
sorry

end maximize_revenue_l609_609615


namespace reflection_ray_properties_l609_609205

/-- Define points P and Q, and their intersection on the x-axis. -/
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨6, 4⟩
def Q : Point := ⟨2, 0⟩

/-- Definitions of the equations of the incident and reflected rays -/
def incident_ray_equation (line : Point → ℝ) (P Q : Point) : Prop :=
  line P = 0 ∧ line Q = 0

def reflected_ray_equation (line : Point → ℝ) (P' Q : Point) : Prop :=
  line P' = 0 ∧ line Q = 0

/-- Proof problem using the definitions given -/
theorem reflection_ray_properties :
  (∃ (incident_line reflected_line : Point → ℝ),
    incident_ray_equation incident_line P Q ∧
    incident_line = (λ p, p.x - p.y - 2) ∧
    reflected_ray_equation reflected_line {x := -P.x + 2 * Q.x, y := P.y} Q ∧
    reflected_line = (λ p, p.x + p.y - 2)) := sorry

end reflection_ray_properties_l609_609205


namespace factorize_xy_squared_minus_x_l609_609699

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l609_609699


namespace combined_population_l609_609957

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end combined_population_l609_609957


namespace probability_of_one_common_l609_609404

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l609_609404


namespace range_of_m_in_fourth_quadrant_l609_609370

theorem range_of_m_in_fourth_quadrant (m : ℝ) : 
  let c := (complex.mk m 1) / (complex.mk 1 1)
  (0 < c.re ∧ c.im < 0) ↔ m ∈ set.Ioo (-1 : ℝ) 1 :=
by 
  sorry

end range_of_m_in_fourth_quadrant_l609_609370


namespace problem_solution_l609_609767

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l609_609767


namespace Vanya_correct_answers_l609_609493

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l609_609493


namespace distinct_divisors_sum_factorial_l609_609923

theorem distinct_divisors_sum_factorial (n : ℕ) (h : n ≥ 3) :
  ∃ (d : Fin n → ℕ), (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n!) ∧ (n! = (Finset.univ.sum d)) :=
sorry

end distinct_divisors_sum_factorial_l609_609923


namespace fewer_white_chairs_than_green_blue_l609_609410

-- Definitions of the conditions
def blue_chairs : ℕ := 10
def green_chairs : ℕ := 3 * blue_chairs
def total_chairs : ℕ := 67
def green_blue_chairs : ℕ := green_chairs + blue_chairs
def white_chairs : ℕ := total_chairs - green_blue_chairs

-- Statement of the theorem
theorem fewer_white_chairs_than_green_blue : green_blue_chairs - white_chairs = 13 :=
by
  -- This is where the proof would go, but we're omitting it as per instruction
  sorry

end fewer_white_chairs_than_green_blue_l609_609410


namespace complex_expression_nonzero_l609_609377

theorem complex_expression_nonzero (a : ℝ) : (1 - Complex.i) + (1 + Complex.i) * a ≠ 0 → a ≠ -1 ∧ a ≠ 1 := 
by 
  sorry

end complex_expression_nonzero_l609_609377


namespace problem_statement_l609_609745

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  Prop := (a > b) ∧ (a = 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1))

noncomputable def eccentricity (a c : ℝ) : Prop := (c / a = sqrt 3 / 2)

noncomputable def perimeter_of_F2MN (a : ℝ) : Prop := (4 * a = 8)

noncomputable def max_chord_len (m : ℝ) : Prop :=
  |m| ≥ 1 ∧ (∀ k : ℝ, let k_sq : ℝ := k ^ 2 in
  let term1 := 64 * k_sq ^ 2 * m ^ 2 in
  let term2 := 16 * (1 + 4 * k_sq) * (4 * k_sq * m ^ 2 - 4) in
  let max_val := 4 * sqrt 3 * |m| / (m ^ 2 + 3) in
  m = sqrt 3 → max_val = 2)

theorem problem_statement (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (ecc_cond : eccentricity a c)
                         (peri_cond : perimeter_of_F2MN a) :
  ellipse_equation a b ha hb ∧ max_chord_len m :=
begin
  sorry
end

end problem_statement_l609_609745


namespace Matilda_correct_age_l609_609855

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end Matilda_correct_age_l609_609855


namespace centers_of_squares_form_square_l609_609452

noncomputable theory

variables (A B C D P Q R : Type) [quadrilateral : parallelogram A B C D] 
  (center_P : is_center_square P A D) (center_Q : is_center_square Q A B) 
  (center_R : is_center_square R B C) (angle_alpha : acute_angle A B C D)

theorem centers_of_squares_form_square : vertices_form_square P Q R :=
sorry

end centers_of_squares_form_square_l609_609452


namespace polar_to_rectangular_conversion_l609_609262

open Real

theorem polar_to_rectangular_conversion :
  let polar_point := (7 : ℝ, π / 3 : ℝ)
  let rectangular_point := (3.5 : ℝ, 7 * (sqrt 3) / 2 : ℝ)
  ∀ (r θ : ℝ), 
  (r, θ) = polar_point →
  (r * cos θ, r * sin θ) = rectangular_point :=
by
  intro r θ
  intro h
  cases h
  simp [cos, sin]
  sorry

end polar_to_rectangular_conversion_l609_609262


namespace cylinder_volume_ratio_theorem_l609_609755

noncomputable def cylinder_volume_ratio (r1 r2 h1 h2 : ℝ) : Prop :=
  r1 = 2 * r2 ∧ 2 * Real.pi * r1 * h1 = 2 * Real.pi * r2 * h2 →
  (π * r1^2 * h1) / (π * r2^2 * h2) = 2

theorem cylinder_volume_ratio_theorem : ∀ (r1 r2 h1 h2 : ℝ),
  r1 = 2 * r2 ∧ 2 * Real.pi * r1 * h1 = 2 * Real.pi * r2 * h2 →
  (π * r1^2 * h1) / (π * r2^2 * h2) = 2 :=
begin
  intros r1 r2 h1 h2 h,
  sorry
end

end cylinder_volume_ratio_theorem_l609_609755


namespace vanya_correct_answers_l609_609485

theorem vanya_correct_answers (candies_received_per_correct : ℕ) 
  (candies_lost_per_incorrect : ℕ) (total_questions : ℕ) (initial_candies_difference : ℤ) :
  candies_received_per_correct = 7 → 
  candies_lost_per_incorrect = 3 → 
  total_questions = 50 → 
  initial_candies_difference = 0 → 
  ∃ (x : ℕ), x = 15 ∧ candies_received_per_correct * x = candies_lost_per_incorrect * (total_questions - x) := 
by 
  intros cr cl tq ic hd cr_eq cl_eq tq_eq ic_eq hd_eq
  use 15
  sorry

end vanya_correct_answers_l609_609485


namespace find_original_price_l609_609180

-- Define the given data
def final_price : ℝ := 12000
def discount_factors : List ℝ := [0.75, 0.85, 0.90, 0.95, 0.98]

-- Define the product of discounts
def total_discount : ℝ := List.foldr (· * ·) 1 discount_factors

-- Define the original price calculation
noncomputable def original_price : ℝ := final_price / total_discount

-- The theorem statement that original_price equals 22458.07 approximately
theorem find_original_price :
  original_price ≈ 22458.07 :=
by
  -- Lean proof can be done here
  sorry

end find_original_price_l609_609180


namespace lottery_probability_exactly_one_common_l609_609401

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l609_609401


namespace inequality_l609_609761

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l609_609761


namespace area_square_WXYZ_eq_48_l609_609020

-- Define the square and its properties
variable (W X Y Z P Q : Point)
variable (WXYZ : Square W X Y Z)
variable (trisected : TrisectedAngle W P Q)
variable (P_on_XY : OnSegment P X Y)
variable (Q_on_WZ : OnSegment Q W Z)
variable (YP_eq_4 : Distance Y P = 4)
variable (QZ_eq_3 : Distance Q Z = 3)

-- The theorem we aim to prove
theorem area_square_WXYZ_eq_48 :
  area WXYZ = 48 := 
by 
  sorry

end area_square_WXYZ_eq_48_l609_609020


namespace sum_x_coords_intersections_l609_609127

-- Define the point (9,9)
def point : (ℝ × ℝ) := (9, 9)

-- Define that the lines passing through (9,9) divide the plane into 9-degree angles
def lines (θ : ℕ) : Prop := θ % 9 = 0

-- Define the line y = 10 - x
def line_eq (x y : ℝ) : Prop := y = 10 - x

-- Prove the sum of the x-coordinates of the intersection points where lines pass through (9,9)
theorem sum_x_coords_intersections : 
  ∀ (L : ℕ → (ℝ × ℝ) → Prop), 
    (∀ θ, lines θ → L θ point) →
    ∑ θ in (Finset.filter lines (Finset.range 180)), (λ θ, (L θ point).fst) = 95 :=
by
  -- Placeholder for the proof
  sorry

end sum_x_coords_intersections_l609_609127


namespace hyperbola_eccentricity_correct_l609_609721

def hyperbola_eccentricity : ℝ :=
let a² := 3 in 
let b² := (16:ℝ)⁻¹ * p^2 in 
let c := sqrt(a² + b²) in
c / sqrt(a²)

theorem hyperbola_eccentricity_correct (p : ℝ) (hp : 0 < p) :
  (let a² := 3 in 
   let b² := (16:ℝ)⁻¹ * p^2 in 
   let c := sqrt(a² + b²) in
   let e := c / (sqrt a²) in 
   e = 2 * sqrt(3) / 3) :=
begin
  sorry
end

end hyperbola_eccentricity_correct_l609_609721


namespace net_change_in_price_l609_609827

def initial_price := P : ℝ

def price_after_first_decrease (P : ℝ) : ℝ := 0.75 * P
def price_after_first_increase (P : ℝ) : ℝ := 0.75 * P * 1.20
def price_after_second_decrease (P : ℝ) : ℝ := (0.75 * P * 1.20) * 0.85
def final_price (P : ℝ) : ℝ := ((0.75 * P * 1.20) * 0.85) * 1.30

theorem net_change_in_price (P : ℝ) : final_price P = 0.9945 * P := 
by
  sorry

#check net_change_in_price

end net_change_in_price_l609_609827


namespace intersection_nonempty_iff_l609_609042

/-- Define sets A and B as described in the problem. -/
def A (x : ℝ) : Prop := -2 < x ∧ x ≤ 1
def B (x : ℝ) (k : ℝ) : Prop := x ≥ k

/-- The main theorem to prove the range of k where the intersection of A and B is non-empty. -/
theorem intersection_nonempty_iff (k : ℝ) : (∃ x, A x ∧ B x k) ↔ k ≤ 1 :=
by
  sorry

end intersection_nonempty_iff_l609_609042


namespace number_of_workers_is_25_l609_609181

noncomputable def original_workers (W : ℕ) :=
  W * 35 = (W + 10) * 25

theorem number_of_workers_is_25 : ∃ W, original_workers W ∧ W = 25 :=
by
  use 25
  unfold original_workers
  sorry

end number_of_workers_is_25_l609_609181


namespace problem_solution_l609_609770

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l609_609770


namespace complex_sum_zero_l609_609871

noncomputable def complexSum {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^(15) + ω^(18) + ω^(21) + ω^(24) + ω^(27) + ω^(30) +
  ω^(33) + ω^(36) + ω^(39) + ω^(42) + ω^(45)

theorem complex_sum_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : complexSum h1 h2 = 0 :=
by
  sorry

end complex_sum_zero_l609_609871


namespace sum_distances_to_other_sides_l609_609458

-- Define the point M on the side of a regular hexagon with side length 12
variable (M : Point)
variable (hexagon : RegularHexagon)
variable (hx_side_length : hexagon.side_length = 12)
variable (on_hexagon_side : M ∈ hexagon.sides)

-- Prove the sum of distances from M to lines containing other sides is 36 * sqrt 3
theorem sum_distances_to_other_sides (M : Point)
  (hexagon : RegularHexagon)
  (hx_side_length : hexagon.side_length = 12)
  (on_hexagon_side : M ∈ hexagon.sides) :
  ∑ side in hexagon.other_sides M, distance M side = 36 * Real.sqrt 3 := 
sorry

end sum_distances_to_other_sides_l609_609458


namespace omega_sum_equals_one_l609_609873

variables (ω : ℂ) (h₀ : ω^5 = 1) (h₁ : ω ≠ 1)

theorem omega_sum_equals_one :
  (ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45) = 1 :=
begin
  sorry
end

end omega_sum_equals_one_l609_609873


namespace exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l609_609887

theorem exists_y_lt_p_div2_py_plus1_not_product_of_greater_y (p : ℕ) [hp : Fact (Nat.Prime p)] (h3 : 3 < p) :
  ∃ y : ℕ, y < p / 2 ∧ ∀ a b : ℕ, py + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by
  sorry

end exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l609_609887


namespace range_of_smallest_side_l609_609337

theorem range_of_smallest_side 
  (c : ℝ) -- the perimeter of the triangle
  (a : ℝ) (b : ℝ) (A : ℝ)  -- three sides of the triangle
  (ha : 0 < a) 
  (hb : b = 2 * a) 
  (hc : a + b + A = c)
  (htriangle : a + b > A ∧ a + A > b ∧ b + A > a) 
  : 
  ∃ (l u : ℝ), l = c / 6 ∧ u = c / 4 ∧ l < a ∧ a < u 
:= sorry

end range_of_smallest_side_l609_609337


namespace tangent_intersection_y_coord_l609_609041

theorem tangent_intersection_y_coord (a b : ℝ) (ha : A = (a, 4 * a ^ 2)) (hb : B = (b, 4 * b ^ 2))
  (h_parab : ∀ x : ℝ, y x = 4 * x ^ 2)
  (h_perp : (8 * a) * (8 * b) = -1) : P.y = -1 / 8 :=
begin
  -- Definitions of points on the parabola
  let A.x := a
  let A.y := 4 * a ^ 2

  let B.x := b
  let B.y := 4 * b ^ 2

  -- Coordinates for the intersection point P
  let P.x := (a + b) / 2
  let P.y := 4 * a * b

  -- Given the perpendicular slopes condition: (8a)(8b) = -1
  have h_tangent_perpendicular : (8 * a) * (8 * b) = -1,
    from h_perp,

  -- Using the slope condition to find ab
  have h_ab : a * b = -1 / 32,
    from eq_div_iff_mul_eq.2 (by rw [mul_comm, ←mul_assoc, h_tangent_perpendicular]; norm_num),

  -- Therefore P.y = 4ab = 4 * (-1 / 32) = -1 / 8
  have h_P_y : P.y = 4 * a * b,
    from h_ab,

  -- Thus the y-coordinate of the intersection is -1 / 8
  show P.y = -1 / 8, by norm_num
end

end tangent_intersection_y_coord_l609_609041


namespace vanya_correct_answers_l609_609487

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609487


namespace length_AC_proof_l609_609954

open Classical

-- Definitions
def is_median (A B C M : Point) : Prop := midpoint B C M ∧ line_through A M
def parallel (l1 l2 : Line) : Prop := ∀ (P : Point), P ∈ l1 → P ∈ l2
def similar_triangles (A B C D E F : Point) : Prop := 
  (AB / DE = BC / EF) ∧ (BC / EF = CA / FA) ∧ (CA / FA = AB / DE)

variables (A B C M P Q R : Point)
variables (length_PQ length_QR length_AC : ℝ)

-- Conditions as definitions
axiom h1 : is_median A B C M  -- AM is the median of triangle ABC
axiom h2 : parallel (line PR) (line AC)  -- PR is parallel to AC
axiom h3 : length_PQ = 5
axiom h4 : length_QR = 3

-- To Prove
theorem length_AC_proof : length_AC = 13 := sorry

end length_AC_proof_l609_609954


namespace quadratic_function_properties_l609_609155

noncomputable def quadratic_function : (ℝ → ℝ) :=
  λ x, x^2 + 1

theorem quadratic_function_properties :
  (∀ x : ℝ, quadratic_function x = x^2 + 1) ∧
  quadratic_function 0 = 1 ∧
  (∀ x : ℝ, ∃ d > 0, quadratic_function (x + d) > quadratic_function x) :=
begin
  sorry
end

end quadratic_function_properties_l609_609155


namespace sum_x_coords_of_A_l609_609137

open Real

noncomputable def area_triangle (A B C : (ℝ × ℝ)) : ℝ := 
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

theorem sum_x_coords_of_A :
  ∀ A : (ℝ × ℝ),
  (area_triangle B C A = 2010) →
  (area_triangle A D F = 8020) →
  B = (0, 0) →
  C = (226, 0) →
  D = (680, 380) →
  F = (700, 400) →
  (sum_of_possible_x_coords_of_A A = -635.6)
sorry

end sum_x_coords_of_A_l609_609137


namespace prism_volume_l609_609555

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end prism_volume_l609_609555


namespace marker_lines_align_after_rotations_l609_609122

-- Define the radii of the driving and driven rollers
def radius_driver : ℕ := 105
def radius_driven : ℕ := 90

-- Define the Least Common Multiple (LCM) function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The number of complete rotations needed by the driver roller
def rotations_driver (r_driver r_driven : ℕ) : ℕ := lcm r_driver r_driven / r_driver

-- The statement we want to prove
theorem marker_lines_align_after_rotations :
  rotations_driver radius_driver radius_driven = 6 :=
by
  sorry

end marker_lines_align_after_rotations_l609_609122


namespace mat_length_correct_l609_609962

noncomputable def table_radius : ℝ := 5
noncomputable def mat_width : ℝ := 1.5
noncomputable def mat_corners_touch_radius: ℝ := 1.91
noncomputable def hypotenuse : ℝ := 5
noncomputable def half_width: ℝ := 0.75

theorem mat_length_correct :
  ∃ (y : ℝ), (hypotenuse ^ 2 = half_width ^ 2 + (y + mat_corners_touch_radius) ^ 2) ∧ y = 3.03 := 
by 
  exists 3.03
  split
  sorry 

end mat_length_correct_l609_609962


namespace constant_term_of_expansion_l609_609342

-- Define the expression and the sum of coefficients condition
def expr (x : ℝ) : ℝ := x^3 + 2 / x^2

-- Given condition
def sum_of_coeffs (n : ℕ) := (expr 1) ^ n

-- Required to prove
theorem constant_term_of_expansion :
  sum_of_coeffs 5 = 243 →
  let n := 5 in
  let term := finset.sum (finset.range (n + 1)) (λ r, nat.choose n r * 2^r * (1 : ℝ)^(15 - 5*r)) in
  term = 80 :=
by
  intro h
  sorry

end constant_term_of_expansion_l609_609342


namespace number_of_zeros_in_Q_l609_609544

def R (k : ℕ) : ℕ := (10^k - 1) / 9

theorem number_of_zeros_in_Q : 
  let Q := R 30 / R 6
  in (nat.to_digits 10 Q).count 0 = 30 :=
by
  -- Definitions for R30 and R6 based on the given R(k) formula
  let R30 := R 30
  let R6 := R 6

  -- Calculate quotient Q
  let Q := R30 / R6

  -- Verify the number of zeros in decimal representation of Q
  have H : (nat.to_digits 10 Q).count 0 = 30 := sorry
  exact H

end number_of_zeros_in_Q_l609_609544


namespace measure_of_angle_A_maximum_value_of_f_l609_609382

-- Defining the context of the triangle and conditions
variables {a b c : ℝ}
variables {A B C : ℝ}  -- Angles in radians

-- The conditions of the problem
axiom condition1 : a^2 + c^2 - b^2 = ac
axiom condition2 : sqrt(2) * b = sqrt(3) * c

-- Finding the measure of angle A
theorem measure_of_angle_A : A = 5 * π / 12 :=
by
  -- Proof omitted
  sorry

-- Finding the maximum value of the function
noncomputable def f (x : ℝ) := 1 + cos (2 * x + π / 3) - cos (2 * x)

theorem maximum_value_of_f : ∀ x, f x ≤ 2 :=
by
  -- Proof omitted
  sorry

end measure_of_angle_A_maximum_value_of_f_l609_609382


namespace locus_of_points_A_and_B_is_circular_arcs_l609_609604

theorem locus_of_points_A_and_B_is_circular_arcs
  (M : Point)
  (c : ℝ)
  (e : Line)
  (F : Point) 
  (H: ∀ (A B: Point), (onLine e A) ∧ (onLine e B) ∧ (midpoint A B = F) ∧ (AM < BM) → (AM * MB = c^2)) 
  (H2: ∀ (P: Point), (onCircleWithCenter M P) (midpoint A B = F)):
  ∀ (A B: Point), (onLine e A) ∧ (onLine e B) ∧ (AM < BM) → (locus A B = circular_arcs_with_constraints) :=
by
  sorry

end locus_of_points_A_and_B_is_circular_arcs_l609_609604


namespace mary_fruit_problem_l609_609163

theorem mary_fruit_problem :
  ∃ (A O O' : ℕ), 
  -- Initial conditions
  A + O = 20 ∧ 
  40 * A + 60 * O = 1120 ∧
  -- New conditions for the average price of 52 cents
  O = 16 ∧
  A = 4 ∧
  -- Number of oranges put back
  let T := A + (O - (16 - 6)) in
  (40 * A + 60 * (O - 10)) / T = 52 ∧
  A + (O - 10) = T :=
begin
  existsi (4 : ℕ),
  existsi (16 : ℕ),
  existsi (6 : ℕ),
  split,
  { -- A + O = 20
    exact rfl,
  },
  split,
  { -- 40 * A + 60 * O = 1120
    norm_num,
  },
  split,
  { -- O = 16
    exact rfl,
  },
  split,
  { -- A = 4
    exact rfl,
  },
  { -- (40 * A + 60 * (O - 10)) / (A + (O - 10)) = 52
    norm_num,
  },
end

end mary_fruit_problem_l609_609163


namespace division_base4_l609_609282

-- Definitions of the numbers in base 4 expressed in base 10 for Lean.
def div_base4 := (1 * 4^4 + 2 * 4^3 + 3 * 4^2 + 4 * 4^1 + 5 * 4^0)  -- 12345_4 in decimal
def divr_base4 := (2 * 4^1 + 3 * 4^0)  -- 23_4 in decimal
def quotient_base4 := (5 * 4^2 + 3 * 4^1 + 5 * 4^0) -- 535_4 in decimal

theorem division_base4 :
  nat.div div_base4 divr_base4 = quotient_base4 :=
by
  -- The proof would be placed here.
  sorry

end division_base4_l609_609282


namespace min_period_and_max_value_l609_609349

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

theorem min_period_and_max_value :
  (∀ x, f (x + π) = f x) ∧ (∀ x, f x ≤ 4) ∧ (∃ x, f x = 4) :=
by
  sorry

end min_period_and_max_value_l609_609349


namespace find_DC_l609_609597

def cyclic_quadrilateral (AB CD AD BC : ℝ) (K : ℝ) (BK_perimeter BK_area : ℝ) : Prop :=
  (AB > BC) ∧ (BC > K) ∧ 
  BK = 4 + Real.sqrt 2 ∧ 
  BK_perimeter = 14 ∧ 
  BK_area = 7

theorem find_DC (AB CD AD BC K BK_perimeter BK_area : ℝ) (h : cyclic_quadrilateral AB CD AD BC K BK_perimeter BK_area) : 
  CD = 6 :=
  sorry

end find_DC_l609_609597


namespace vanya_correct_answers_l609_609466

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l609_609466


namespace goods_train_speed_l609_609193

theorem goods_train_speed (train_length platform_length : ℝ) (time_sec : ℝ) : 
  train_length = 270.0416 ∧ platform_length = 250 ∧ time_sec = 26 → 
  (train_length + platform_length) / time_sec * 3.6 = 72.00576 :=
by
  sorry

end goods_train_speed_l609_609193


namespace f_is_odd_l609_609335

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 2*x - 3
  else if x = 0 then 0
  else -x^2 + 2*x + 3

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end f_is_odd_l609_609335


namespace mb_range_l609_609951

noncomputable def m : ℚ := 3 / 4
noncomputable def b : ℚ := -5 / 3

theorem mb_range : m * b < -1 := by
  -- Assumed values
  have m_val : m = 3 / 4 := rfl
  have b_val : b = -5 / 3 := rfl
  -- Calculate the product
  calc m * b = (3 / 4) * (-5 / 3) := by rw [m_val, b_val]
  ...       = -15 / 12 := by norm_num
  ...       = -5 / 4 := by norm_num
  -- Inequality check
  ... < -1 := by norm_num

end mb_range_l609_609951


namespace length_of_paper_l609_609612

-- Given conditions
variables (width : ℝ) (margin : ℝ) (area : ℝ)
variables (L : ℝ)

-- Define specific values for the conditions
def width_val : ℝ := 8.5
def margin_val : ℝ := 1.5
def area_val : ℝ := 38.5

-- The width of the picture area (excluding the margins)
def picture_width := width_val - 2 * margin_val

-- The length of the picture area (excluding the margins)
def Lp := L - 2 * margin_val

-- The equation based on the area of the picture
def area_equation := Lp * picture_width = area_val

-- The proof statement
theorem length_of_paper (h1 : width = width_val) (h2 : margin = margin_val) (h3 : area = area_val) (h4 : picture_width = 5.5) (h5 : area_equation) :
  L = 10 :=
by
  sorry

end length_of_paper_l609_609612


namespace vanya_correct_answers_l609_609464

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l609_609464


namespace original_cube_volume_l609_609453

theorem original_cube_volume 
  (a : ℕ) 
  (h : 3 * a * (a - a / 2) * a - a^3 = 2 * a^2) : 
  a = 4 → a^3 = 64 := 
by
  sorry

end original_cube_volume_l609_609453


namespace distance_from_center_to_chord_l609_609184

theorem distance_from_center_to_chord (a b : ℝ) : 
  ∃ d : ℝ, d = (1/4) * |a - b| := 
sorry

end distance_from_center_to_chord_l609_609184


namespace shooting_competition_sequences_l609_609008

/-- In a shooting competition, eight clay targets are set up in three hanging columns with the
configuration: three targets in the first column (A), two targets in the second column (B), 
and three targets in the third column (C). We need to count the number of different sequences
the shooter can follow to break all targets, following these rules:
1) The shooter selects any one of the columns to shoot a target from.
2) The shooter must then hit the lowest remaining target in the selected column.
-/
theorem shooting_competition_sequences : 
  let A := 3 in let B := 2 in let C := 3 in (A + B + C = 8) →
  (∑ n : ℕ in {n | (n = 8)}, 
    Multinomial (A + B + C) ! ! (A !, B !, C !)) = 560 :=
by {
  have hA : 3,
  have hB : 2,
  have hC : 3,
  have h_total : A + B + C = 8 := by norm_num,
  sorry -- proof to be completed
}

end shooting_competition_sequences_l609_609008


namespace polar_C1_correct_intersection_points_C1_C2_l609_609354

def C1_param (t : Real) : Real × Real :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

def polar_C2 (θ : Real) : Real :=
  2 * Real.sin θ

def polar_equation_C1 (ρ θ : Real) : Prop :=
  ρ^2 - 8 * ρ * Real.cos θ - 10 * ρ * Real.sin θ + 16 = 0

theorem polar_C1_correct (ρ θ : Real) :
  (∃ t, C1_param t = (ρ * Real.cos θ, ρ * Real.sin θ)) →
  polar_equation_C1 ρ θ :=
sorry

theorem intersection_points_C1_C2 (ρ₁ θ₁ ρ₂ θ₂ : Real) :
  (polar_C2 θ₁ = ρ₁ ∧ polar_C2 θ₂ = ρ₂) →
  polar_equation_C1 ρ₁ θ₁ ∧ polar_equation_C1 ρ₂ θ₂ →
  ((ρ₁, θ₁) = (Real.sqrt 2, Real.pi / 4) ∨ (ρ₁, θ₁) = (2, Real.pi / 2)) ∧
  ((ρ₂, θ₂) = (Real.sqrt 2, Real.pi / 4) ∨ (ρ₂, θ₂) = (2, Real.pi / 2)) :=
sorry

end polar_C1_correct_intersection_points_C1_C2_l609_609354


namespace minimum_segments_through_centers_l609_609214

theorem minimum_segments_through_centers (n : ℕ) (h: n ≥ 1) : 
  ∀ grid : fin n → fin n, (∃ broken_line : list (fin n × fin n), 
    (∀ p : fin n × fin n, p ∈ broken_line) ∧ length broken_line = 2 * n - 2) :=
sorry

end minimum_segments_through_centers_l609_609214


namespace polynomial_integer_values_l609_609926

theorem polynomial_integer_values (n : ℕ) (x : ℤ) : (∃ k : ℤ, k = (prod (range n).map (λ i, x - i) / (n!))) :=
by
  sorry

end polynomial_integer_values_l609_609926


namespace inequality_holds_l609_609825

variable (f : ℝ → ℝ)

theorem inequality_holds (h1 : ∀ x : ℝ, differentiable ℝ f) 
                         (h2 : ∀ x : ℝ, f x > deriv f x) 
                         (a b : ℝ) 
                         (h3 : a > b) : 
    e^a * f b > e^b * f a :=
sorry

end inequality_holds_l609_609825


namespace prob1_prob2_prob3_l609_609449

-- Define the sequences for rows ①, ②, and ③
def seq1 (n : ℕ) : ℤ := (-2) ^ n
def seq2 (m : ℕ) : ℤ := (-2) ^ (m - 1)
def seq3 (m : ℕ) : ℤ := (-2) ^ (m - 1) - 1

-- Prove the $n^{th}$ number in row ①
theorem prob1 (n : ℕ) : seq1 n = (-2) ^ n :=
by sorry

-- Prove the relationship between $m^{th}$ numbers in row ② and row ③
theorem prob2 (m : ℕ) : seq3 m = seq2 m - 1 :=
by sorry

-- Prove the value of $x + y + z$ where $x$, $y$, and $z$ are the $2019^{th}$ numbers in rows ①, ②, and ③, respectively
theorem prob3 : seq1 2019 + seq2 2019 + seq3 2019 = -1 :=
by sorry

end prob1_prob2_prob3_l609_609449


namespace line_slope_sqrt2_l609_609115

theorem line_slope_sqrt2 (x y : ℝ) (h : x + sqrt 2 * y - 1 = 0) : 
  ∃ k, k = -sqrt 2 / 2 ∧ y = k * x + 1 / sqrt 2 :=
by
  sorry

end line_slope_sqrt2_l609_609115


namespace solve_slope_angle_l609_609114

def slope_angle_of_line (α : ℝ) : Prop :=
  let k := -√3 in 
  let slope_prop := k = -√3 in
  let alpha_prop := α ∈ set.Ico 0 180 ∧ Real.tan (α * Real.pi / 180) = k in
  alpha_prop

theorem solve_slope_angle : slope_angle_of_line 120 :=
by
  sorry

end solve_slope_angle_l609_609114


namespace inequality_preservation_l609_609818

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l609_609818


namespace count_divisible_3_in_first_105_l609_609209

-- Define the sequence
def sequence : List Nat := List.join (List.map (fun n => List.replicate n n) (List.range 15))

-- Define a helper function to count the numbers divisible by 3 in the first 105 numbers of the sequence
def count_divisible_by_3 (lst : List Nat) : Nat :=
  List.length (List.filter (fun x => x % 3 = 0) (List.take 105 lst))

-- Main theorem to prove
theorem count_divisible_3_in_first_105 :
  count_divisible_by_3 sequence = 30 :=
by
  sorry

end count_divisible_3_in_first_105_l609_609209


namespace sum_of_cube_angles_l609_609538

theorem sum_of_cube_angles (W X Y Z : Point) (cube : Cube)
  (angle_WXY angle_XYZ angle_YZW angle_ZWX : ℝ)
  (h₁ : angle_WXY = 90)
  (h₂ : angle_XYZ = 90)
  (h₃ : angle_YZW = 90)
  (h₄ : angle_ZWX = 60) :
  angle_WXY + angle_XYZ + angle_YZW + angle_ZWX = 330 := by
  sorry

end sum_of_cube_angles_l609_609538


namespace necessary_but_not_sufficient_condition_counterexample_not_sufficient_l609_609299

def greatest_integer (x : ℝ) : ℤ := ⌊x⌋

theorem necessary_but_not_sufficient_condition (x y : ℝ) :
  (|x - y| < 1) → (greatest_integer x = greatest_integer y) :=
by
  sorry

theorem counterexample_not_sufficient (x y : ℝ) :
  (|x - y| < 1) ∧ ¬ (greatest_integer x = greatest_integer y) :=
by
  -- provide counterexample of x = 1.2 and y = 2.1:
  use 1.2, 2.1
  sorry

end necessary_but_not_sufficient_condition_counterexample_not_sufficient_l609_609299


namespace noLShapeTiling_l609_609725

def LShapedPiece (cells : Finset (Fin 8 × Fin 8)) : Prop := cells.card = 4 ∧ ∃ c1 c2 c3 c4, 
  cells = {c1, c2, c3, c4} ∧ (
    (c2.1 = c1.1 ∧ c3.1 = c1.1 + 1 ∧ c4.1 = c1.1 + 1 ∧ c3.2 = c2.2 ∧ c4.2 = c2.2 + 1) ∨
    -- Add other three variations for L-shaped piece
    sorry
  )

def centralSquareRemoved (cells : Finset (Fin 8 × Fin 8)) : Prop :=
  ∀ x y, cells (⟨x, H⟩, ⟨y, H⟩) ↔ ¬ (6 ≤ x ∧ x < 10 ∧ 6 ≤ y ∧ y < 10)

theorem noLShapeTiling : 
  ¬ ∃ (tiling : Finset (Finset (Fin 8 × Fin 8))),
  (∀ piece ∈ tiling, LShapedPiece piece) ∧ 
  tiling.card = 15 ∧ ⟦⟨⟨x, y⟩ | ∀ x y, x < 16 ∧ y < 16⟩\((6 ≤ x ∧ x < 10) ∧ (6 ≤ y ∧ y < 10))⟩ = ⋃₀ tiling.to_set 
:=
sorry

end noLShapeTiling_l609_609725


namespace part_a_part_b_l609_609245

-- Define the matrices as given in the conditions

-- Part (a) - Define the problem
def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 5], ![-3, -4]]

theorem part_a : det matrix_a = 7 := by
  sorry

-- Part (b) - Define the problem
variables (a b : ℤ)

def matrix_b : Matrix (Fin 2) (Fin 2) (ℤ) := ![![a^2, a * b], ![a * b, b^2]]

theorem part_b : det (matrix_b a b) = 0 := by
  sorry

end part_a_part_b_l609_609245


namespace Louie_monthly_payment_l609_609898

noncomputable def monthly_payment (P : ℕ) (r : ℚ) (n t : ℕ) : ℚ :=
  (P : ℚ) * (1 + r / n)^(n * t) / t

theorem Louie_monthly_payment : 
  monthly_payment 2000 0.10 1 3 = 887 := 
by
  sorry

end Louie_monthly_payment_l609_609898


namespace log_bounds_sum_l609_609288

theorem log_bounds_sum : 
  ∃ c d : ℤ, c < log 50 / log 10 ∧ log 50 / log 10 < d ∧ c + d = 3 :=
by
  sorry

end log_bounds_sum_l609_609288


namespace Vanya_correct_answers_l609_609494

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l609_609494


namespace midpoint_of_intersections_l609_609711

theorem midpoint_of_intersections :
  let A := (2.6, 0, 2.8)
  let B := (11, 14, 0)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
  M = (6.8, 7, 1.4) :=
by
  let A := (2.6, 0, 2.8)
  let B := (11, 14, 0)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
  show M = (6.8, 7, 1.4)
  sorry

end midpoint_of_intersections_l609_609711


namespace triangle_ABC_is_isosceles_l609_609921

variables {A B C D E : Type*}
variables [geometry A B C D E]

-- Assuming cyclic quadrilateral and segment conditions
axiom BAC_cyclic : cyclic A D E C
axiom BD_plus_DE_eq_BC : BD + DE = BC
axiom BE_plus_ED_eq_AB : BE + ED = AB

-- Proving triangle ABC is isosceles
theorem triangle_ABC_is_isosceles : triangle_is_isosceles A B C :=
begin
  sorry -- Proof
end

end triangle_ABC_is_isosceles_l609_609921


namespace find_b_l609_609935

def h (x : ℝ) : ℝ := x / 3 + 2
def k (x : ℝ) : ℝ := 5 - 2 * x

theorem find_b (b : ℝ) (hkb : h (k b) = 4) : b = -1/2 :=
by
  sorry

end find_b_l609_609935


namespace percentage_error_in_area_is_correct_l609_609200

-- Given conditions
variables (L : ℝ) -- actual length of the side of the square
def length_with_error := 1.03 * L
def width_with_error := 0.98 * L
def actual_area := L^2
def calculated_area := length_with_error * width_with_error

-- We must prove that the percentage error in the calculated area of the square is 2.94%
theorem percentage_error_in_area_is_correct :
  (calculated_area - actual_area) / actual_area * 100 = 2.94 := by
  sorry

end percentage_error_in_area_is_correct_l609_609200


namespace slowest_pipe_time_l609_609454

noncomputable def fill_tank_rate (R : ℝ) : Prop :=
  let rate1 := 6 * R
  let rate3 := 2 * R
  let combined_rate := 9 * R
  combined_rate = 1 / 30

theorem slowest_pipe_time (R : ℝ) (h : fill_tank_rate R) : 1 / R = 270 :=
by
  have h1 := h
  sorry

end slowest_pipe_time_l609_609454


namespace triangle_area_5_6_7_l609_609525

def QinJiushaoArea (a b c : ℝ) : ℝ :=
  sqrt ((1 / 4) * (a^2 * b^2 - ((a^2 + b^2 - c^2) / 2)^2))

theorem triangle_area_5_6_7 :
  QinJiushaoArea 5 6 7 = 6 * sqrt 6 := 
  sorry

end triangle_area_5_6_7_l609_609525


namespace ratio_of_areas_of_pentagons_l609_609524

theorem ratio_of_areas_of_pentagons
  (s : ℝ) -- side length of the small pentagons
  (perimeter_small : ℝ := 5 * s) -- perimeter of each small pentagon
  (total_fencing : ℝ := 5 * perimeter_small) -- total perimeter of five small pentagons
  (S : ℝ := total_fencing / 5) -- side length of the large pentagon
  (area_pentagon : real → ℝ := λ s, (1 / 4) * real.sqrt (5 * (5 + 2 * real.sqrt 5)) * s^2)
  (area_total_small := 5 * area_pentagon s)
  (area_large := area_pentagon S) :
  (area_total_small / area_large) = (1 / 5) :=
by sorry

end ratio_of_areas_of_pentagons_l609_609524


namespace inequality_relationship_l609_609790

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l609_609790


namespace total_students_l609_609160

theorem total_students (rank_right rank_left : ℕ) 
  (h_right : rank_right = 6) 
  (h_left : rank_left = 5) 
  : rank_right + rank_left - 1 = 10 :=
by
  -- Use the given conditions to form the proof.
  rw [h_right, h_left]
  sorry

end total_students_l609_609160


namespace coeff_x3yz4_in_expansion_l609_609093

theorem coeff_x3yz4_in_expansion (x y z : ℕ) :
  (x + y + z = 8) → (x = 3 ∧ y = 1 ∧ z = 4) → (nat.choose 8 3 * nat.choose 5 4 * nat.choose 1 1 = 280) :=
by
  intro h1 h2
  cases h2 with hx h3
  cases h3 with hy hz
  rw [hx, hy, hz] at h1
  sorry

end coeff_x3yz4_in_expansion_l609_609093


namespace weight_of_second_new_player_l609_609978

theorem weight_of_second_new_player
  (number_of_original_players : ℕ)
  (average_weight_of_original_players : ℝ)
  (weight_of_first_new_player : ℝ)
  (new_average_weight : ℝ)
  (total_number_of_players : ℕ)
  (total_weight_of_9_players : ℝ)
  (combined_weight_of_original_and_first_new : ℝ)
  (weight_of_second_new_player : ℝ)
  (h1 : number_of_original_players = 7)
  (h2 : average_weight_of_original_players = 103)
  (h3 : weight_of_first_new_player = 110)
  (h4 : new_average_weight = 99)
  (h5 : total_number_of_players = 9)
  (h6 : total_weight_of_9_players = total_number_of_players * new_average_weight)
  (h7 : combined_weight_of_original_and_first_new = number_of_original_players * average_weight_of_original_players + weight_of_first_new_player)
  (h8 : total_weight_of_9_players - combined_weight_of_original_and_first_new = weight_of_second_new_player) :
  weight_of_second_new_player = 60 :=
by
  sorry

end weight_of_second_new_player_l609_609978


namespace probability_of_one_common_l609_609406

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions
def total_numbers := 45
def chosen_numbers := 6

-- Define the probability calculation as a Lean function
def probability_exactly_one_common : ℚ :=
  let total_combinations := binom total_numbers chosen_numbers
  let successful_combinations := 6 * binom (total_numbers - chosen_numbers) (chosen_numbers - 1)
  successful_combinations / total_combinations

-- The theorem we need to prove
theorem probability_of_one_common :
  probability_exactly_one_common = (6 * binom 39 5 : ℚ) / binom 45 6 :=
sorry

end probability_of_one_common_l609_609406


namespace order_wxyz_l609_609820

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_wxyz : x < y ∧ y < z ∧ z < w := by
  sorry

end order_wxyz_l609_609820


namespace probability_exactly_one_common_number_l609_609383

-- Define the combinatorial function
def C (n k : ℕ) : ℕ := Nat.combination n k

-- State the given conditions
def total_combinations : ℕ := C 45 6
def successful_combinations : ℕ := 6 * (C 39 5)

-- Define the probability function
def probability : ℚ := successful_combinations / total_combinations

-- State the theorem to be proved
theorem probability_exactly_one_common_number :
  probability = 0.424 := 
sorry

end probability_exactly_one_common_number_l609_609383


namespace vanya_correct_answers_l609_609473

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l609_609473


namespace qualifying_rate_l609_609105

theorem qualifying_rate (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = 1 - a - b + a * b :=
by sorry

end qualifying_rate_l609_609105


namespace one_minus_repeating_eight_l609_609286

-- Define the repeating decimal
def repeating_eight : Real := 0.8888888888 -- repeating of 8

-- Define the repeating decimal as a fraction
def repeating_eight_as_fraction : Real := 8 / 9

-- The proof statement
theorem one_minus_repeating_eight : 1 - repeating_eight = 1 / 9 := by
  -- Since proof is not required, we use sorry
  sorry

end one_minus_repeating_eight_l609_609286


namespace win_prob_scientific_notation_l609_609272

-- Definition of the probability of winning the first prize
def win_prob : ℝ := 1 / 200000

-- Statement that this probability equals \(5 \times 10^{-6}\)
theorem win_prob_scientific_notation : win_prob = 5 * 10^(-6) :=
by
  sorry

end win_prob_scientific_notation_l609_609272


namespace problem_statement_l609_609967

noncomputable def length_of_CD (radius : ℝ) (total_volume : ℝ) (hemisphere_volume : ℝ) : ℝ :=
  let h := (total_volume - 2 * hemisphere_volume) / (π * radius^2)
  h

theorem problem_statement :
  ∀ (radius : ℝ) (volume : ℝ),
    radius = 4 →
    volume = 288 * π →
    let hemisphere_volume := (2/3) * π * radius^3 in
    length_of_CD radius volume hemisphere_volume = 92 / 3 :=
by
  intros radius volume rad_eq vol_eq hemisphere_volume_def
  rw [rad_eq, vol_eq, hemisphere_volume_def]
  have h := length_of_CD 4 (288 * π) (256 / 3 * π)
  rw [show 256 / 3 from (256 / 3), show 288 * π from (288 * π)] at h
  norm_num at h
  exact h

end problem_statement_l609_609967


namespace powerfully_even_count_below_3000_l609_609644

def powerfully_even (n : ℕ) : Prop :=
  ∃ (a : ℕ) (b : ℕ), b > 1 ∧ b % 2 = 0 ∧ a^b = n

theorem powerfully_even_count_below_3000 : 
  (Finset.filter powerfully_even (Finset.range 3000)).card = 14 :=
by
  sorry

end powerfully_even_count_below_3000_l609_609644


namespace election_valid_votes_l609_609582

theorem election_valid_votes (V : ℕ) (h1 : 0.70 * V - 0.30 * V = 180) : V = 450 :=
by
  sorry

end election_valid_votes_l609_609582


namespace instantaneous_rate_of_change_at_x1_l609_609206

open Real

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - x^2 + 8

theorem instantaneous_rate_of_change_at_x1 : deriv f 1 = -1 := by
  sorry

end instantaneous_rate_of_change_at_x1_l609_609206


namespace abs_diff_of_points_on_curve_l609_609920

theorem abs_diff_of_points_on_curve :
  ∀ a b : ℝ, (a ≠ b) → (∃ x : ℝ, x = real.sqrt 2 ∧ (b ^ 2 + x ^ 4 = 4 * x ^ 2 * a - 5) ∧ (a ^ 2 + x ^ 4 = 4 * x ^ 2 * b - 5)) →
  |a - b| = 2 * real.sqrt 7 :=
by
  intro a b h_distinct h_points
  sorry

end abs_diff_of_points_on_curve_l609_609920


namespace expand_expression_l609_609279

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l609_609279


namespace geometric_sequence_product_bound_l609_609310

theorem geometric_sequence_product_bound {a1 a2 a3 m q : ℝ} (h_sum : a1 + a2 + a3 = 3 * m) (h_m_pos : 0 < m) (h_q_pos : 0 < q) (h_geom : a1 = a2 / q ∧ a3 = a2 * q) : 
  0 < a1 * a2 * a3 ∧ a1 * a2 * a3 ≤ m^3 := 
sorry

end geometric_sequence_product_bound_l609_609310


namespace smallest_integer_in_set_138_l609_609106

def is_odd_sequence (s : List Int) : Prop := 
  ∀ (n : ℕ), n < s.length → s[n] % 2 = 1

def median (l : List ℕ) : ℕ := 
  if l.length % 2 = 1 then l[(l.length / 2).toNat] 
  else (l[(l.length / 2).toNat] + l[(l.length / 2).toNat - 1]) / 2

def has_median (s : List Int) (m : ℤ) : Prop :=
  median s = m

def max_elem (s : List Int) (max : ℤ) : Prop :=
  s.maximum = some max

theorem smallest_integer_in_set_138 :
  ∃ (s : List Int), 
  is_odd_sequence s ∧ 
  has_median s 152.5 ∧ 
  max_elem s 161 ∧ 
  s.minimum = some 138 := 
  sorry

end smallest_integer_in_set_138_l609_609106


namespace calculate_expression_l609_609246

theorem calculate_expression :
  ((1 / 3 : ℝ) ^ (-2 : ℝ)) + Real.tan (Real.pi / 4) - Real.sqrt ((-10 : ℝ) ^ 2) = 0 := by
  sorry

end calculate_expression_l609_609246


namespace inradius_inequality_l609_609832

variables {A B C D S : Type}
variables (AB BC CD DA : ℝ)
variables (r1 r2 r3 r4 : ℝ)
variables (midpoint_S : (A + B) * 0.5 = S ∧ (C + D) * 0.5 = S)

-- Definition: Convex quadrilateral and the midpoints of the diagonals
def convex_quadrilateral_midpoint (A B C D S : Type) :=
  convex_quadrilateral A B C D ∧ 
  diagonal_midpoints A B C D S

-- Main theorem statement
theorem inradius_inequality
  (convex_quadrilateral_midpoint A B C D S)
  (r1 r2 r3 r4 : ℝ)
  : |r1 - r2 + r3 - r4| ≤ 1 / 8 * |AB - BC + CD - DA| :=
by
  sorry

end inradius_inequality_l609_609832


namespace simplify_expression_l609_609938

theorem simplify_expression (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x := 
by
  sorry

end simplify_expression_l609_609938


namespace grid_rectangles_count_3x3_l609_609810

theorem grid_rectangles_count_3x3 : 
  let n := 3
  let m := 3
  ∃ count : ℕ, count = 10 ∧
    (forall top_left top_right bottom_left bottom_right : ℕ, 
      top_left < top_right ∧ bottom_left < bottom_right ∧
      top_left.div3 < bottom_left.div3 ∧
      top_right.div3 < bottom_right.div3 ∧
      top_left % 3 = bottom_left % 3 ∧ 
      top_right % 3 = bottom_right % 3 → 
      True) :=
begin
  sorry
end

end grid_rectangles_count_3x3_l609_609810


namespace euler_thirteen_pi_half_l609_609259

noncomputable def convertEuler (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem euler_thirteen_pi_half :
  convertEuler (13 * real.pi / 2) = complex.I :=
by
  -- Given Euler's formula: e^(ix) = cos(x) + i*sin(x)
  have euler_formula : ∀ x : ℝ, complex.exp (complex.I * x) = complex.of_real (real.cos x) + complex.I * real.sin x,
    from complex.exp_eq_cos_add_sin,
  -- Simplify using Euler's formula and the periodicity of cos and sin
  rw [← euler_formula (13 * real.pi / 2)],
  -- Simplify further
  rw [real.cos_add_pi_div_two, real.sin_add_pi_div_two],
  sorry

end euler_thirteen_pi_half_l609_609259


namespace donna_hourly_wage_l609_609672

-- Definitions derived from conditions
def total_dog_walking_hours : ℕ := 14
def card_shop_earnings : ℝ := 125
def babysitting_earnings : ℝ := 40
def total_earnings : ℝ := 305

-- Proven statement that Donna makes $10 per hour walking dogs
theorem donna_hourly_wage :
  ∃ (D : ℝ), D = 10 ∧ (total_dog_walking_hours * D + card_shop_earnings + babysitting_earnings = total_earnings) :=
begin
  use 10,
  split,
  { refl },
  {
    calc
      total_dog_walking_hours * 10 + card_shop_earnings + babysitting_earnings
          = 14 * 10 + 125 + 40 : by refl
      ... = 140 + 125 + 40 : by norm_num
      ... = 305 : by norm_num,
  },
end

end donna_hourly_wage_l609_609672


namespace rainfall_march_correct_l609_609541

def rainfall_march : ℝ :=
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  total_expected - total_april_to_july

theorem rainfall_march_correct (march_rainfall : ℝ) :
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let average := 4
  let total_expected := 5 * average
  let total_april_to_july := april + may + june + july
  march_rainfall = total_expected - total_april_to_july :=
by
  sorry

end rainfall_march_correct_l609_609541


namespace tommy_saw_100_wheels_l609_609132

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end tommy_saw_100_wheels_l609_609132


namespace johns_elevation_after_travel_l609_609032

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end johns_elevation_after_travel_l609_609032


namespace smallest_integer_l609_609146

theorem smallest_integer :
  ∃ (M : ℕ), M > 0 ∧
             M % 3 = 2 ∧
             M % 4 = 3 ∧
             M % 5 = 4 ∧
             M % 6 = 5 ∧
             M % 7 = 6 ∧
             M % 11 = 10 ∧
             M = 4619 :=
by
  sorry

end smallest_integer_l609_609146


namespace painted_unit_cube_probability_l609_609196

theorem painted_unit_cube_probability :
  let total_cubes := 125
  let three_painted_faces := 8
  let two_painted_faces := 22
  let one_painted_face := 78
  let no_painted_faces := total_cubes - three_painted_faces - two_painted_faces - one_painted_face
  let total_ways_to_choose_two := Nat.choose total_cubes 2
  let ways_to_choose_target_cubes := two_painted_faces * no_painted_faces
  let probability := ways_to_choose_target_cubes / total_ways_to_choose_two
in
  probability = 187 / 3875 :=
by
  sorry

end painted_unit_cube_probability_l609_609196


namespace range_of_m_real_values_l609_609715

-- Defining the quadratic inequality condition
def quadratic_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, mx^2 - mx - 1 < 0

-- The statement to prove that the range of real values for m is (-4, 0]
theorem range_of_m_real_values :
  { m : ℝ // quadratic_inequality m } = { m : ℝ // -4 < m ∧ m ≤ 0 } :=
by {
  -- the proof will be inserted here
  sorry
}

end range_of_m_real_values_l609_609715


namespace remainder_of_sum_is_12_l609_609148

theorem remainder_of_sum_is_12 (D k1 k2 : ℤ) (h1 : 242 = k1 * D + 4) (h2 : 698 = k2 * D + 8) : (242 + 698) % D = 12 :=
by
  sorry

end remainder_of_sum_is_12_l609_609148


namespace common_chord_length_l609_609562

/-- Given two circles each with a radius of 12 cm and the distance between their centers is 18 cm,
    the length of their common chord is 6√7 cm. -/
theorem common_chord_length :
  ∀ (r : ℝ) (d : ℝ), r = 12 → d = 18 → ∃ (l : ℝ), l = 6 * real.sqrt 7 := by
sorry

end common_chord_length_l609_609562


namespace tangent_parallel_points_l609_609522

noncomputable theory

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ P₀ : ℝ × ℝ, 
  let a := P₀.1 in
  let y := P₀.2 in
  (∃ (a : ℝ), f'(a) = 4 ∧ f(a) = y ) →
    (P₀ = (1, 0) ∨ P₀ = (-1, -4)) := by
    sorry

#check tangent_parallel_points

end tangent_parallel_points_l609_609522


namespace total_paint_used_correct_l609_609857

-- Definitions based on conditions
def initial_paint := 360
def paint_used_first_week (total_paint : ℕ) := (1/3 : ℚ) * total_paint
def remaining_paint_after_first_week (total_paint : ℕ) := total_paint - paint_used_first_week total_paint
def paint_used_second_week (remaining_paint : ℕ) := (1/5 : ℚ) * remaining_paint

-- Statement to prove
theorem total_paint_used_correct : 
  let used_first_week := paint_used_first_week initial_paint,
      remaining_after_first_week := initial_paint - used_first_week,
      used_second_week := paint_used_second_week remaining_after_first_week
  in used_first_week + used_second_week = 168 :=
by
  sorry

end total_paint_used_correct_l609_609857


namespace g_of_3_l609_609375

theorem g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x + 2) = 2 * x + 3) : g 3 = 5 :=
by
  have x := 1
  have hx : 3 = x + 2 := by norm_num
  rw [hx] at h
  exact h 1
  sorry -- Proof is omitted

end g_of_3_l609_609375


namespace product_of_equal_numbers_l609_609513

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l609_609513


namespace parabola_focus_coordinates_l609_609095

theorem parabola_focus_coordinates :
  (∃ f : ℝ × ℝ, f = (0, 2) ∧ ∀ x y : ℝ, y = (1/8) * x^2 ↔ f = (0, 2)) :=
sorry

end parabola_focus_coordinates_l609_609095


namespace min_elements_in_A_l609_609313

-- Conditions
variables (n : ℕ) (A : Set ℕ)

-- n is a positive integer such that n ≥ 2
axiom n_pos : n ≥ 2

-- A contains positive integers with the smallest element being 1
axiom A_props : ∀ x ∈ A, x > 0 ∧ 1 ∈ A

-- The largest element of A is a such that 7 * 3^n < a < 3^(n + 2)
axiom A_bounds : ∃ a, a ∈ A ∧ 7 * 3^n < a ∧ a < 3^(n + 2)

-- For any element x ∈ A (x ≠ 1), there exist y, z, w ∈ A (which can be the same) such that x = y + z + w
axiom A_condition : ∀ x ∈ A, x ≠ 1 → ∃ y z w ∈ A, x = y + z + w

-- Goal: Determine the minimal number of elements in set A
theorem min_elements_in_A : (∃ A : Set ℕ, (∀ x ∈ A, x > 0) ∧ 1 ∈ A ∧ (∃ a, a ∈ A ∧ 7 * 3^n < a ∧ a < 3^(n + 2))
  ∧ (∀ x ∈ A, x ≠ 1 → ∃ y z w ∈ A, x = y + z + w) ∧ (Finset.card (A.to_finset) = n + 4)) := sorry

end min_elements_in_A_l609_609313


namespace factorize_expression_l609_609686

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l609_609686
