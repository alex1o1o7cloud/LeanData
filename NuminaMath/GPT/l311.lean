import Mathlib

namespace largest_possible_m_value_l311_311094

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_possible_m_value :
  ∃ (m x y : ℕ), is_three_digit m ∧ is_prime x ∧ is_prime y ∧ x ≠ y ∧
  x < 10 ∧ y < 10 ∧ is_prime (10 * x - y) ∧ m = x * y * (10 * x - y) ∧ m = 705 := sorry

end largest_possible_m_value_l311_311094


namespace min_moves_to_reset_counters_l311_311571

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l311_311571


namespace max_k_guarded_l311_311879

-- Define the size of the board
def board_size : ℕ := 8

-- Define the directions a guard can look
inductive Direction
| up | down | left | right

-- Define a guard's position on the board as a pair of Fin 8
def Position := Fin board_size × Fin board_size

-- Guard record that contains its position and direction
structure Guard where
  pos : Position
  dir : Direction

-- Function to determine if guard A is guarding guard B
def is_guarding (a b : Guard) : Bool :=
  match a.dir with
  | Direction.up    => a.pos.1 < b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.down  => a.pos.1 > b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.left  => a.pos.1 = b.pos.1 ∧ a.pos.2 > b.pos.2
  | Direction.right => a.pos.1 = b.pos.1 ∧ a.pos.2 < b.pos.2

-- The main theorem states that the maximum k is 5
theorem max_k_guarded : ∃ k : ℕ, (∀ g : Guard, ∃ S : Finset Guard, (S.card ≥ k) ∧ (∀ s ∈ S, is_guarding s g)) ∧ k = 5 :=
by
  sorry

end max_k_guarded_l311_311879


namespace initial_apples_proof_l311_311850

-- Define the variables and conditions
def initial_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ): ℕ := 
  handed_out + pies * apples_per_pie

-- Define the proof statement
theorem initial_apples_proof : initial_apples 30 7 8 = 86 := by 
  sorry

end initial_apples_proof_l311_311850


namespace part_a_l311_311115

-- Part (a)
theorem part_a (x : ℕ)  : (x^2 - x + 2) % 7 = 0 → x % 7 = 4 := by 
  sorry

end part_a_l311_311115


namespace six_times_product_plus_one_equals_seven_pow_sixteen_l311_311899

theorem six_times_product_plus_one_equals_seven_pow_sixteen :
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := 
  sorry

end six_times_product_plus_one_equals_seven_pow_sixteen_l311_311899


namespace chuck_play_area_l311_311779

-- Defining the conditions and required constants
def shed_length : ℝ := 3
def shed_width : ℝ := 4
def leash_length : ℝ := 4
def tree_distance : ℝ := 1.5
def accessible_area : ℝ := 9.42 * Real.pi

-- Main statement
theorem chuck_play_area : 
  chuck_play_area <= 9.42 * Real.pi := 
begin
  sorry
end

end chuck_play_area_l311_311779


namespace investment_value_l311_311142

noncomputable def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r)^n

theorem investment_value :
  ∀ (P : ℕ) (r : ℚ) (n : ℕ),
  P = 8000 →
  r = 0.05 →
  n = 3 →
  compound_interest P r n = 9250 := by
    intros P r n hP hr hn
    unfold compound_interest
    -- calculation steps would be here
    sorry

end investment_value_l311_311142


namespace sum_of_squares_inequality_l311_311544

theorem sum_of_squares_inequality (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end sum_of_squares_inequality_l311_311544


namespace overlap_area_of_sectors_l311_311737

/--
Given two sectors of a circle with radius 10, with centers at points P and R respectively, 
one having a central angle of 45 degrees and the other having a central angle of 90 degrees, 
prove that the area of the shaded region where they overlap is 12.5π.
-/
theorem overlap_area_of_sectors 
  (r : ℝ) (θ₁ θ₂ : ℝ) (A₁ A₂ : ℝ)
  (h₀ : r = 10)
  (h₁ : θ₁ = 45)
  (h₂ : θ₂ = 90)
  (hA₁ : A₁ = (θ₁ / 360) * π * r ^ 2)
  (hA₂ : A₂ = (θ₂ / 360) * π * r ^ 2)
  : A₁ = 12.5 * π := 
sorry

end overlap_area_of_sectors_l311_311737


namespace driver_travel_distance_per_week_l311_311125

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end driver_travel_distance_per_week_l311_311125


namespace minimize_quadratic_l311_311741

theorem minimize_quadratic (y : ℝ) : 
  ∃ m, m = 3 * y ^ 2 - 18 * y + 11 ∧ 
       (∀ z : ℝ, 3 * z ^ 2 - 18 * z + 11 ≥ m) ∧ 
       m = -16 := 
sorry

end minimize_quadratic_l311_311741


namespace right_triangle_sin_sum_l311_311050

/--
In a right triangle ABC with ∠A = 90°, prove that sin A + sin^2 B + sin^2 C = 2.
-/
theorem right_triangle_sin_sum (A B C : ℝ) (hA : A = 90) (hABC : A + B + C = 180) :
  Real.sin (A * π / 180) + Real.sin (B * π / 180) ^ 2 + Real.sin (C * π / 180) ^ 2 = 2 :=
sorry

end right_triangle_sin_sum_l311_311050


namespace distance_halfway_along_orbit_l311_311857

-- Define the conditions
variables (perihelion aphelion : ℝ) (perihelion_dist : perihelion = 3) (aphelion_dist : aphelion = 15)

-- State the theorem
theorem distance_halfway_along_orbit : 
  ∃ d, d = (perihelion + aphelion) / 2 ∧ d = 9 :=
by
  sorry

end distance_halfway_along_orbit_l311_311857


namespace different_suits_choice_count_l311_311193

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l311_311193


namespace smallest_value_abs_sum_l311_311267

theorem smallest_value_abs_sum : 
  ∃ x : ℝ, (λ x, |x + 3| + |x + 5| + |x + 6| = 5) ∧ 
           (∀ y : ℝ, |y + 3| + |y + 5| + |y + 6| ≥ 5) :=
by
  sorry

end smallest_value_abs_sum_l311_311267


namespace roses_distribution_l311_311416

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) : 
  initial_roses = 40 → 
  stolen_roses = 4 → 
  people = 9 → 
  (initial_roses - stolen_roses) / people = 4 :=
by
  intros h_initial_roses h_stolen_roses h_people
  rw [h_initial_roses, h_stolen_roses, h_people]
  norm_num
  sorry

end roses_distribution_l311_311416


namespace non_congruent_triangles_with_perimeter_18_l311_311951

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l311_311951


namespace number_of_players_in_tournament_l311_311546

theorem number_of_players_in_tournament (G : ℕ) (h1 : G = 42) (h2 : ∀ n : ℕ, G = n * (n - 1)) : ∃ n : ℕ, G = 42 ∧ n = 7 :=
by
  -- Let's suppose n is the number of players, then we need to prove
  -- ∃ n : ℕ, 42 = n * (n - 1) ∧ n = 7
  sorry

end number_of_players_in_tournament_l311_311546


namespace hex_to_decimal_B4E_l311_311018

def hex_B := 11
def hex_4 := 4
def hex_E := 14
def base := 16
def hex_value := hex_B * base^2 + hex_4 * base^1 + hex_E * base^0

theorem hex_to_decimal_B4E : hex_value = 2894 :=
by
  -- here we would write the proof steps, this is skipped with "sorry"
  sorry

end hex_to_decimal_B4E_l311_311018


namespace last_passenger_probability_last_passenger_probability_l311_311087

theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned (n : ℕ) : ℝ :=
begin
  sorry
end

def prob_last_passenger_sit_in_assigned n : ℝ :=
begin
  -- Conditions in the problem
  -- Define the probability calculation logic based on the seating rules.
  sorry
end

-- The theorem that we need to prove
theorem last_passenger_probability (n : ℕ) (h1 : n > 0) : 
  prob_last_passenger_sit_in_assigned n = 1/2 :=
by sorry

end last_passenger_probability_last_passenger_probability_l311_311087


namespace find_b_l311_311305

theorem find_b {a b : ℝ} (h₁ : 2 * 2 + b = 1 - 2 * a) (h₂ : -2 * 2 + b = -15 + 2 * a) : 
  b = -7 := sorry

end find_b_l311_311305


namespace likelihood_of_white_crows_at_birch_unchanged_l311_311732

theorem likelihood_of_white_crows_at_birch_unchanged 
  (a b c d : ℕ) 
  (h1 : a + b = 50) 
  (h2 : c + d = 50) 
  (h3 : b ≥ a) 
  (h4 : d ≥ c - 1) : 
  (bd + ac + a + b : ℝ) / 2550 > (bc + ad : ℝ) / 2550 := by 
  sorry

end likelihood_of_white_crows_at_birch_unchanged_l311_311732


namespace base_conversion_l311_311742

theorem base_conversion (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 7 * A = 5 * B) : 8 * A + B = 47 :=
by
  sorry

end base_conversion_l311_311742


namespace matches_in_each_matchbook_l311_311058

-- Conditions given in the problem
def one_stamp_worth_matches (s : ℕ) : Prop := s = 12
def tonya_initial_stamps (t : ℕ) : Prop := t = 13
def tonya_final_stamps (t : ℕ) : Prop := t = 3
def jimmy_initial_matchbooks (j : ℕ) : Prop := j = 5

-- Goal: prove M = 24
theorem matches_in_each_matchbook (M : ℕ) (s t_initial t_final j : ℕ) 
  (h1 : one_stamp_worth_matches s) 
  (h2 : tonya_initial_stamps t_initial) 
  (h3 : tonya_final_stamps t_final) 
  (h4 : jimmy_initial_matchbooks j) : M = 24 := by
  sorry

end matches_in_each_matchbook_l311_311058


namespace lemon_heads_per_package_l311_311225

theorem lemon_heads_per_package (total_lemon_heads boxes : ℕ)
  (H : total_lemon_heads = 54)
  (B : boxes = 9)
  (no_leftover : total_lemon_heads % boxes = 0) :
  total_lemon_heads / boxes = 6 :=
sorry

end lemon_heads_per_package_l311_311225


namespace cubic_yards_to_cubic_feet_l311_311363

def conversion_factor := 3 -- 1 yard = 3 feet
def cubic_conversion := conversion_factor ^ 3 -- 1 cubic yard = (3 feet) ^ 3

theorem cubic_yards_to_cubic_feet :
  5 * cubic_conversion = 135 :=
by
  unfold conversion_factor cubic_conversion
  norm_num
  sorry

end cubic_yards_to_cubic_feet_l311_311363


namespace min_abs_sum_l311_311268

theorem min_abs_sum : ∃ x : ℝ, ∀ x : ℝ, 
  let f := λ x : ℝ, abs (x + 3) + abs (x + 5) + abs (x + 6) in
  f x = 5 :=
sorry

end min_abs_sum_l311_311268


namespace geometric_sequence_a4_value_l311_311048

variable {α : Type} [LinearOrderedField α]

noncomputable def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, n < m → ∃ r : α, 0 < r ∧ a m = a n * r^(m - n)

theorem geometric_sequence_a4_value (a : ℕ → α)
  (pos : ∀ n, 0 < a n)
  (geo_seq : is_geometric_sequence a)
  (h : a 1 * a 7 = 36) :
  a 4 = 6 :=
by 
  sorry

end geometric_sequence_a4_value_l311_311048


namespace range_of_a_l311_311326

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- The theorem we need to prove
theorem range_of_a {a : ℝ} (h : A a ⊆ B) : 1 ≤ a ∧ a < 5 := 
sorry

end range_of_a_l311_311326


namespace non_congruent_triangles_with_perimeter_18_l311_311948

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l311_311948


namespace horizontal_distance_parabola_l311_311306

theorem horizontal_distance_parabola :
  ∀ x_p x_q : ℝ, 
  (x_p^2 + 3*x_p - 4 = 8) → 
  (x_q^2 + 3*x_q - 4 = 0) → 
  x_p ≠ x_q → 
  abs (x_p - x_q) = 2 :=
sorry

end horizontal_distance_parabola_l311_311306


namespace count_non_congruent_triangles_with_perimeter_18_l311_311962

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l311_311962


namespace sum_possible_values_of_p_l311_311700

theorem sum_possible_values_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (α β : ℕ), (10 * α * β = q) ∧ (10 * (α + β) = -p)) :
  p = -3100 :=
by
  sorry

end sum_possible_values_of_p_l311_311700


namespace polygon_sides_eq_eight_l311_311044

theorem polygon_sides_eq_eight (n : ℕ) (h : (n - 2) * 180 = 3 * 360) : n = 8 := by 
  sorry

end polygon_sides_eq_eight_l311_311044


namespace prime_division_or_divisibility_l311_311232

open Nat

theorem prime_division_or_divisibility (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hodd : Odd p) (hd : p ∣ q^r + 1) :
    (2 * r ∣ p - 1) ∨ (p ∣ q^2 - 1) := 
sorry

end prime_division_or_divisibility_l311_311232


namespace average_sleep_time_l311_311316

def sleep_times : List ℕ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end average_sleep_time_l311_311316


namespace minimum_moves_to_reset_counters_l311_311565

-- Definitions
def counter_in_initial_range (c : ℕ) := 1 ≤ c ∧ c ≤ 2017
def valid_move (decrements : ℕ) (counters : list ℕ) : list ℕ :=
  counters.map (λ c, if c ≥ decrements then c - decrements else c)
def all_counters_zero (counters : list ℕ) : Prop :=
  counters.all (λ c, c = 0)

-- Problem statement
theorem minimum_moves_to_reset_counters :
  ∀ (counters : list ℕ)
  (h : counters.length = 28)
  (h' : ∀ c ∈ counters, counter_in_initial_range c),
  ∃ (moves : ℕ), moves = 11 ∧
    ∀ (f : ℕ → list ℕ → list ℕ)
    (hm : ∀ ds cs, ds > 0 → cs.length = 28 → 
           (∀ c ∈ cs, counter_in_initial_range c) →
           ds ≤ 2017 → f ds cs = valid_move ds cs),
    all_counters_zero (nat.iterate (f (λ m cs, valid_move m cs)) 11 counters) :=
sorry

end minimum_moves_to_reset_counters_l311_311565


namespace term_number_l311_311362

theorem term_number (n : ℕ) : 
  (n ≥ 1) ∧ (5 * Real.sqrt 3 = Real.sqrt (3 + 4 * (n - 1))) → n = 19 :=
by
  intro h
  let h1 := h.1
  let h2 := h.2
  have h3 : (5 * Real.sqrt 3)^2 = (Real.sqrt (3 + 4 * (n - 1)))^2 := by sorry
  sorry

end term_number_l311_311362


namespace isosceles_triangle_count_l311_311216

-- Define the variables
variables {A B C D E F : Type}

-- Define the angles and congruence conditions
axiom h1 : ∠BAC + ∠ABC + ∠ACB = 180 -- Sum of angles in triangle ABC
axiom h2 : AB = AC                   -- AB is congruent to AC
axiom h3 : ∠ABC = 60                 -- measure of angle ABC is 60 degrees
axiom h4 : BD bisects ∠ABC           -- Segment BD bisects angle ABC
axiom h5 : BD ⟨intersection⟩ AC = D  -- Point D on side AC
axiom h6 : E ⟨intersection⟩ BC = E   -- Point E on side BC
axiom h7 : DE ∥ AB                   -- Segment DE is parallel to AB
axiom h8 : F ⟨intersection⟩ AC = F   -- Point F on side AC
axiom h9 : EF ∥ BD                   -- Segment EF is parallel to BD

-- Define the goal for the proof
theorem isosceles_triangle_count : 
  ∃ A B C D E F : Type, 
  ∠BAC + ∠ABC + ∠ACB = 180 ∧ 
  AB = AC ∧ 
  ∠ABC = 60 ∧ 
  BD bisects ∠ABC ∧ 
  BD ⟨intersection⟩ AC = D ∧ 
  E ⟨intersection⟩ BC = E ∧ 
  DE ∥ AB ∧ 
  F ⟨intersection⟩ AC = F ∧ 
  EF ∥ BD ∧ 
  -- Number of isosceles triangles in the figure
  (isosceles_triangles A B C D E F = 6) :=
sorry

end isosceles_triangle_count_l311_311216


namespace circle_diameter_l311_311266

theorem circle_diameter (A : ℝ) (hA : A = 25 * π) (r : ℝ) (h : A = π * r^2) : 2 * r = 10 := by
  sorry

end circle_diameter_l311_311266


namespace isosceles_triangle_l311_311540

variable (a b c : ℝ)
variable (α β γ : ℝ)
variable (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β))
variable (triangle_angles : γ = π - (α + β))

theorem isosceles_triangle : α = β :=
by
  sorry

end isosceles_triangle_l311_311540


namespace added_classes_l311_311549

def original_classes := 15
def students_per_class := 20
def new_total_students := 400

theorem added_classes : 
  new_total_students = original_classes * students_per_class + 5 * students_per_class :=
by
  sorry

end added_classes_l311_311549


namespace total_people_count_l311_311258

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l311_311258


namespace solve_equation_l311_311043

theorem solve_equation (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 / x + 4 / y = 1) : 
  x = 3 * y / (y - 4) :=
sorry

end solve_equation_l311_311043


namespace ab_c_work_days_l311_311875

noncomputable def W_ab : ℝ := 1 / 15
noncomputable def W_c : ℝ := 1 / 30
noncomputable def W_abc : ℝ := W_ab + W_c

theorem ab_c_work_days :
  (1 / W_abc) = 10 :=
by
  sorry

end ab_c_work_days_l311_311875


namespace susan_more_cats_than_bob_after_transfer_l311_311082

-- Definitions and conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def cats_transferred : ℕ := 4

-- Question statement translated to Lean
theorem susan_more_cats_than_bob_after_transfer :
  (susan_initial_cats - cats_transferred) - bob_initial_cats = 14 :=
by
  sorry

end susan_more_cats_than_bob_after_transfer_l311_311082


namespace trigonometric_identity_l311_311636

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / (Real.cos (3 * π / 2 - θ) - Real.sin (π - θ)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l311_311636


namespace other_root_of_equation_l311_311038

theorem other_root_of_equation (m : ℤ) (h₁ : (2 : ℤ) ∈ {x : ℤ | x ^ 2 - 3 * x - m = 0}) : 
  ∃ x, x ≠ 2 ∧ (x ^ 2 - 3 * x - m = 0) ∧ x = 1 :=
by {
  sorry
}

end other_root_of_equation_l311_311038


namespace number_of_square_integers_l311_311919

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l311_311919


namespace prob_sum_is_10_prob_term_index_distance_even_l311_311713

open Finset

def set_A : Finset ℕ := {1, 2, 3, 4, 5}

def is_ordered_triple (a b c : ℕ) : Prop := a < b ∧ b < c

def sum_is_10 (a b c : ℕ) : Prop := a + b + c = 10

def term_index_distance (a b c : ℕ) : ℕ := |a - 1| + |b - 2| + |c - 3|

def even (n : ℕ) : Prop := n % 2 = 0

theorem prob_sum_is_10 : (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2 ∧ sum_is_10 t.1 t.2.1 t.2.2), 1) / (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2), 1)  = 1 / 5 := 
by sorry

theorem prob_term_index_distance_even : (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2 ∧ even (term_index_distance t.1 t.2.1 t.2.2)), 1) / (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2), 1) = 3 / 5 := 
by sorry

end prob_sum_is_10_prob_term_index_distance_even_l311_311713


namespace math_problem_l311_311828

-- Arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 8 ∧ a 3 + a 5 = 4 * a 2

-- General term of the arithmetic sequence {a_n}
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 4 * n

-- Geometric sequence {b_n}
def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 4 = a 1 ∧ b 6 = a 4

-- The sum S_n of the first n terms of the sequence {b_n - a_n}
def sum_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (2 ^ (n - 1) - 1 / 2 - 2 * n ^ 2 - 2 * n)

-- Full proof statement
theorem math_problem (a : ℕ → ℕ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  general_term a →
  ∀ a_n : ℕ → ℝ, a_n 1 = 4 ∧ a_n 4 = 16 →
  geometric_sequence b a_n →
  sum_sequence b a_n S :=
by
  intros h_arith_seq h_gen_term h_a_n h_geom_seq
  sorry

end math_problem_l311_311828


namespace percentage_decrease_l311_311848

theorem percentage_decrease (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.20 * A) : 
  ∃ y : ℝ, A = C - (y/100) * C ∧ y = 50 / 3 :=
by {
  sorry
}

end percentage_decrease_l311_311848


namespace henri_drove_more_miles_l311_311786

-- Defining the conditions
def Gervais_average_miles_per_day := 315
def Gervais_days_driven := 3
def Henri_total_miles := 1250

-- Total miles driven by Gervais
def Gervais_total_miles := Gervais_average_miles_per_day * Gervais_days_driven

-- The proof problem statement
theorem henri_drove_more_miles : Henri_total_miles - Gervais_total_miles = 305 := 
by 
  sorry

end henri_drove_more_miles_l311_311786


namespace evaluate_polynomial_at_3_l311_311574

def f (x : ℕ) : ℕ := 3 * x^7 + 2 * x^5 + 4 * x^3 + x

theorem evaluate_polynomial_at_3 : f 3 = 7158 := by
  sorry

end evaluate_polynomial_at_3_l311_311574


namespace overall_average_score_l311_311423

noncomputable def average_score (scores : List ℝ) : ℝ :=
  scores.sum / (scores.length)

theorem overall_average_score :
  let male_scores_avg := 82
  let female_scores_avg := 92
  let num_male_students := 8
  let num_female_students := 32
  let total_students := num_male_students + num_female_students
  let combined_scores_total := num_male_students * male_scores_avg + num_female_students * female_scores_avg
  average_score ([combined_scores_total]) / total_students = 90 :=
by 
  sorry

end overall_average_score_l311_311423


namespace triangle_side_lengths_approx_l311_311739

noncomputable def approx_side_lengths (AB : ℝ) (BAC ABC : ℝ) : ℝ × ℝ :=
  let α := BAC * Real.pi / 180
  let β := ABC * Real.pi / 180
  let c := AB
  let β1 := (90 - (BAC)) * Real.pi / 180
  let m := 2 * c * α * (β1 + 3) / (9 - α * β1)
  let c1 := 2 * c * β1 * (α + 3) / (9 - α * β1)
  let β2 := β1 - β
  let γ1 := α + β
  let a1 := β2 / γ1 * (γ1 + 3) / (β2 + 3) * m
  let a := (9 - β2 * γ1) / (2 * γ1 * (β2 + 3)) * m
  let b := c1 - a1
  (a, b)

theorem triangle_side_lengths_approx (AB : ℝ) (BAC ABC : ℝ) (hAB : AB = 441) (hBAC : BAC = 16.2) (hABC : ABC = 40.6) :
  approx_side_lengths AB BAC ABC = (147, 344) := by
  sorry

end triangle_side_lengths_approx_l311_311739


namespace repeating_decimal_subtraction_simplified_l311_311102

theorem repeating_decimal_subtraction_simplified :
  let x := (567 / 999 : ℚ)
  let y := (234 / 999 : ℚ)
  let z := (891 / 999 : ℚ)
  x - y - z = -186 / 333 :=
by
  sorry

end repeating_decimal_subtraction_simplified_l311_311102


namespace running_time_of_BeastOfWar_is_100_l311_311563

noncomputable def Millennium := 120  -- minutes
noncomputable def AlphaEpsilon := Millennium - 30  -- minutes
noncomputable def BeastOfWar := AlphaEpsilon + 10  -- minutes
noncomputable def DeltaSquadron := 2 * BeastOfWar  -- minutes

theorem running_time_of_BeastOfWar_is_100 :
  BeastOfWar = 100 :=
by
  -- Proof goes here
  sorry

end running_time_of_BeastOfWar_is_100_l311_311563


namespace find_function_l311_311330

def satisfies_condition (f : ℕ+ → ℕ+) :=
  ∀ a b : ℕ+, f a + b ∣ a^2 + f a * f b

theorem find_function :
  ∀ f : ℕ+ → ℕ+, satisfies_condition f → (∀ a : ℕ+, f a = a) :=
by
  intros f h
  sorry

end find_function_l311_311330


namespace range_of_m_l311_311406

def A (x : ℝ) := x^2 - 3 * x - 10 ≤ 0
def B (x m : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_of_m (m : ℝ) (h : ∀ x, B x m → A x) : m ≤ 3 := by
  sorry

end range_of_m_l311_311406


namespace even_function_behavior_l311_311229

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x2 - x1) * (f x2 - f x1) > 0

theorem even_function_behavior (f : ℝ → ℝ) (h_even : is_even_function f) (h_condition : condition f) 
  (n : ℕ) (h_n : n > 0) : 
  f (n+1) < f (-n) ∧ f (-n) < f (n-1) :=
sorry

end even_function_behavior_l311_311229


namespace fraction_meaningful_condition_l311_311718

theorem fraction_meaningful_condition (x : ℝ) : 3 - x ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_condition_l311_311718


namespace mangoes_rate_l311_311602

theorem mangoes_rate (grapes_weight mangoes_weight total_amount grapes_rate mango_rate : ℕ)
  (h1 : grapes_weight = 7)
  (h2 : grapes_rate = 68)
  (h3 : total_amount = 908)
  (h4 : mangoes_weight = 9)
  (h5 : total_amount - grapes_weight * grapes_rate = mangoes_weight * mango_rate) :
  mango_rate = 48 :=
by
  sorry

end mangoes_rate_l311_311602


namespace multiplications_in_three_hours_l311_311593

theorem multiplications_in_three_hours :
  let rate := 15000  -- multiplications per second
  let seconds_in_three_hours := 3 * 3600  -- seconds in three hours
  let total_multiplications := rate * seconds_in_three_hours
  total_multiplications = 162000000 :=
by
  let rate := 15000
  let seconds_in_three_hours := 3 * 3600
  let total_multiplications := rate * seconds_in_three_hours
  have h : total_multiplications = 162000000 := sorry
  exact h

end multiplications_in_three_hours_l311_311593


namespace restore_original_price_l311_311763

-- Defining the original price of the jacket
def original_price (P : ℝ) := P

-- Defining the price after each step of reduction
def price_after_first_reduction (P : ℝ) := P * (1 - 0.25)
def price_after_second_reduction (P : ℝ) := price_after_first_reduction P * (1 - 0.20)
def price_after_third_reduction (P : ℝ) := price_after_second_reduction P * (1 - 0.10)

-- Express the condition to restore the original price
theorem restore_original_price (P : ℝ) (x : ℝ) : 
  original_price P = price_after_third_reduction P * (1 + x) → 
  x = 0.85185185 := 
by
  sorry

end restore_original_price_l311_311763


namespace union_A_B_complement_intersection_A_B_l311_311805

-- Define universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | -5 ≤ x ∧ x ≤ -1 }

-- Define set B
def B : Set ℝ := { x | x ≥ -4 }

-- Prove A ∪ B = [-5, +∞)
theorem union_A_B : A ∪ B = { x : ℝ | -5 ≤ x } :=
by {
  sorry
}

-- Prove complement of A ∩ B with respect to U = (-∞, -4) ∪ (-1, +∞)
theorem complement_intersection_A_B : U \ (A ∩ B) = { x : ℝ | x < -4 } ∪ { x : ℝ | x > -1 } :=
by {
  sorry
}

end union_A_B_complement_intersection_A_B_l311_311805


namespace select_best_athlete_l311_311862

theorem select_best_athlete :
  let avg_A := 185
  let var_A := 3.6
  let avg_B := 180
  let var_B := 3.6
  let avg_C := 185
  let var_C := 7.4
  let avg_D := 180
  let var_D := 8.1
  avg_A = 185 ∧ var_A = 3.6 ∧
  avg_B = 180 ∧ var_B = 3.6 ∧
  avg_C = 185 ∧ var_C = 7.4 ∧
  avg_D = 180 ∧ var_D = 8.1 →
  (∃ x, (x = avg_A ∧ avg_A = 185 ∧ var_A = 3.6) ∧
        (∀ (y : ℕ), (y = avg_A) 
        → avg_A = 185 
        ∧ var_A <= var_C ∧ 
        var_A <= var_D 
        ∧ var_A <= var_B)) :=
by {
  sorry
}

end select_best_athlete_l311_311862


namespace base_length_l311_311208

-- Definition: Isosceles triangle
structure IsoscelesTriangle :=
  (perimeter : ℝ)
  (side : ℝ)

-- Conditions: Perimeter and one side of the isosceles triangle
def given_triangle : IsoscelesTriangle := {
  perimeter := 26,
  side := 11
}

-- The problem to solve: length of the base given the perimeter and one side
theorem base_length : 
  (given_triangle.perimeter = 26 ∧ given_triangle.side = 11) →
  (∃ b : ℝ, b = 11 ∨ b = 7.5) :=
by 
  sorry

end base_length_l311_311208


namespace min_value_3x_4y_l311_311165

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 25 :=
sorry

end min_value_3x_4y_l311_311165


namespace binary_101_is_5_l311_311478

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l311_311478


namespace how_many_years_later_will_tom_be_twice_tim_l311_311733

-- Conditions
def toms_age := 15
def total_age := 21
def tims_age := total_age - toms_age

-- Define the problem statement
theorem how_many_years_later_will_tom_be_twice_tim (x : ℕ) 
  (h1 : toms_age + tims_age = total_age) 
  (h2 : toms_age = 15) 
  (h3 : ∀ y : ℕ, toms_age + y = 2 * (tims_age + y) ↔ y = x) : 
  x = 3 
:= sorry

end how_many_years_later_will_tom_be_twice_tim_l311_311733


namespace non_congruent_triangles_with_perimeter_18_l311_311947

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l311_311947


namespace focus_on_negative_y_axis_l311_311854

-- Definition of the condition: equation of the parabola
def parabola (x y : ℝ) := x^2 + y = 0

-- Statement of the problem
theorem focus_on_negative_y_axis (x y : ℝ) (h : parabola x y) : 
  -- The focus of the parabola lies on the negative half of the y-axis
  ∃ y, y < 0 :=
sorry

end focus_on_negative_y_axis_l311_311854


namespace emily_beads_l311_311155

theorem emily_beads (n : ℕ) (b : ℕ) (total_beads : ℕ) (h1 : n = 26) (h2 : b = 2) (h3 : total_beads = n * b) : total_beads = 52 :=
by
  sorry

end emily_beads_l311_311155


namespace find_p_plus_q_l311_311685

/--
In \(\triangle{XYZ}\), \(XY = 12\), \(\angle{X} = 45^\circ\), and \(\angle{Y} = 60^\circ\).
Let \(G, E,\) and \(L\) be points on the line \(YZ\) such that \(XG \perp YZ\), 
\(\angle{XYE} = \angle{EYX}\), and \(YL = LY\). Point \(O\) is the midpoint of 
the segment \(GL\), and point \(Q\) is on ray \(XE\) such that \(QO \perp YZ\).
Prove that \(XQ^2 = \dfrac{81}{2}\) and thus \(p + q = 83\), where \(p\) and \(q\) 
are relatively prime positive integers.
-/
theorem find_p_plus_q :
  ∃ (p q : ℕ), gcd p q = 1 ∧ XQ^2 = 81 / 2 ∧ p + q = 83 :=
sorry

end find_p_plus_q_l311_311685


namespace xiaoyangs_scores_l311_311874

theorem xiaoyangs_scores (average : ℕ) (diff : ℕ) (h_average : average = 96) (h_diff : diff = 8) :
  ∃ chinese_score math_score : ℕ, chinese_score = 92 ∧ math_score = 100 :=
by
  sorry

end xiaoyangs_scores_l311_311874


namespace geometric_sequence_solution_l311_311249

open Real

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q ^ m

theorem geometric_sequence_solution :
  ∃ (a : ℕ → ℝ) (q : ℝ), geometric_sequence a q ∧
    (∀ n, 1 ≤ n ∧ n ≤ 5 → 10^8 ≤ a n ∧ a n < 10^9) ∧
    (∀ n, 6 ≤ n ∧ n ≤ 10 → 10^9 ≤ a n ∧ a n < 10^10) ∧
    (∀ n, 11 ≤ n ∧ n ≤ 14 → 10^10 ≤ a n ∧ a n < 10^11) ∧
    (∀ n, 15 ≤ n ∧ n ≤ 16 → 10^11 ≤ a n ∧ a n < 10^12) ∧
    (∀ i, a i = 7 * 3^(16-i) * 5^(i-1)) := sorry

end geometric_sequence_solution_l311_311249


namespace different_suits_choice_count_l311_311194

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l311_311194


namespace ratio_garbage_zane_dewei_l311_311328

-- Define the weights of garbage picked up by Daliah, Dewei, and Zane.
def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah_garbage - 2
def zane_garbage : ℝ := 62

-- The theorem that we need to prove
theorem ratio_garbage_zane_dewei : zane_garbage / dewei_garbage = 4 :=
by
  sorry

end ratio_garbage_zane_dewei_l311_311328


namespace percent_primes_divisible_by_3_l311_311274

-- Definition of primes less than 20
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Definition of divisibility by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Definition of the main theorem
theorem percent_primes_divisible_by_3 : 
  (card {p ∈ primes_less_than_20 | is_divisible_by_3 p} : ℚ) / card primes_less_than_20 = 0.125 :=
by
  sorry

end percent_primes_divisible_by_3_l311_311274


namespace minimum_value_x_plus_4_div_x_l311_311788

theorem minimum_value_x_plus_4_div_x (x : ℝ) (hx : x > 0) : x + 4 / x ≥ 4 :=
sorry

end minimum_value_x_plus_4_div_x_l311_311788


namespace y_in_terms_of_x_l311_311501

theorem y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = -2 * x + 5 :=
sorry

end y_in_terms_of_x_l311_311501


namespace horizontal_force_magnitude_l311_311754

-- We state our assumptions and goal
theorem horizontal_force_magnitude (W : ℝ) : 
  (∀ μ : ℝ, μ = (Real.sin (Real.pi / 6)) / (Real.cos (Real.pi / 6)) ∧ 
    (∀ P : ℝ, 
      (P * (Real.sin (Real.pi / 3))) = 
      ((μ * (W * (Real.cos (Real.pi / 6)) + P * (Real.cos (Real.pi / 3)))) + W * (Real.sin (Real.pi / 6))) →
      P = W * Real.sqrt 3)) :=
sorry

end horizontal_force_magnitude_l311_311754


namespace great_dane_weight_l311_311757

theorem great_dane_weight : 
  ∀ (C P G : ℕ), 
    C + P + G = 439 ∧ P = 3 * C ∧ G = 3 * P + 10 → G = 307 := by
    sorry

end great_dane_weight_l311_311757


namespace min_value_x_plus_y_l311_311528

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_plus_y_l311_311528


namespace initial_hotdogs_l311_311126

-- Definitions
variable (x : ℕ)

-- Conditions
def condition : Prop := x - 2 = 97 

-- Statement to prove
theorem initial_hotdogs (h : condition x) : x = 99 :=
  by
    sorry

end initial_hotdogs_l311_311126


namespace probability_divisor_of_8_on_8_sided_die_l311_311881

def divisor_probability : ℚ :=
  let sample_space := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_8 := {1, 2, 4, 8}
  let favorable_outcomes := divisors_of_8 ∩ sample_space
  favorable_outcomes.card / sample_space.card

theorem probability_divisor_of_8_on_8_sided_die :
  divisor_probability = 1 / 2 :=
sorry

end probability_divisor_of_8_on_8_sided_die_l311_311881


namespace probability_red_next_ball_l311_311014

-- Definitions of initial conditions
def initial_red_balls : ℕ := 50
def initial_blue_balls : ℕ := 50
def initial_yellow_balls : ℕ := 30
def total_pulled_balls : ℕ := 65

-- Condition that Calvin pulled out 5 more red balls than blue balls
def red_balls_pulled (blue_balls_pulled : ℕ) : ℕ := blue_balls_pulled + 5

-- Compute the remaining balls
def remaining_balls (blue_balls_pulled : ℕ) : Prop :=
  let remaining_red_balls := initial_red_balls - red_balls_pulled blue_balls_pulled
  let remaining_blue_balls := initial_blue_balls - blue_balls_pulled
  let remaining_yellow_balls := initial_yellow_balls - (total_pulled_balls - red_balls_pulled blue_balls_pulled - blue_balls_pulled)
  (remaining_red_balls + remaining_blue_balls + remaining_yellow_balls) = 15

-- Main theorem to be proven
theorem probability_red_next_ball (blue_balls_pulled : ℕ) (h : remaining_balls blue_balls_pulled) :
  (initial_red_balls - red_balls_pulled blue_balls_pulled) / 15 = 9 / 26 :=
sorry

end probability_red_next_ball_l311_311014


namespace compute_100a_b_l311_311405

theorem compute_100a_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) * (x + 10) = 0 ↔ x = -a ∨ x = -b ∨ x = -10)
  (h2 : a ≠ -4 ∧ b ≠ -4 ∧ 10 ≠ -4)
  (h3 : ∀ x : ℝ, (x + 2 * a) * (x + 5) * (x + 8) = 0 ↔ x = -5)
  (hb : b = 8)
  (ha : 2 * a = 5) :
  100 * a + b = 258 := 
sorry

end compute_100a_b_l311_311405


namespace boat_speed_still_water_l311_311287

theorem boat_speed_still_water (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 16) (h2 : upstream_speed = 9) : 
  (downstream_speed + upstream_speed) / 2 = 12.5 := 
by
  -- conditions explicitly stated above
  sorry

end boat_speed_still_water_l311_311287


namespace neg_p_sufficient_for_neg_q_l311_311793

def p (a : ℝ) := a ≤ 2
def q (a : ℝ) := a * (a - 2) ≤ 0

theorem neg_p_sufficient_for_neg_q (a : ℝ) : ¬ p a → ¬ q a :=
sorry

end neg_p_sufficient_for_neg_q_l311_311793


namespace inequality_solution_l311_311913

theorem inequality_solution :
  {x : Real | (2 * x - 5) * (x - 3) / x ≥ 0} = {x : Real | (x ∈ Set.Ioc 0 (5 / 2)) ∨ (x ∈ Set.Ici 3)} := 
sorry

end inequality_solution_l311_311913


namespace peter_has_142_nickels_l311_311413

-- Define the conditions
def nickels (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2

-- The theorem to prove the number of nickels
theorem peter_has_142_nickels : ∃ (n : ℕ), nickels n ∧ n = 142 :=
by {
  sorry
}

end peter_has_142_nickels_l311_311413


namespace last_passenger_sits_in_assigned_seat_l311_311086

-- Define the problem with the given conditions
def probability_last_passenger_assigned_seat (n : ℕ) : ℝ :=
  if n > 0 then 1 / 2 else 0

-- Given conditions in Lean definitions
variables {n : ℕ} (absent_minded_scientist_seat : ℕ) (seats : Fin n → ℕ) (passengers : Fin n → ℕ)
  (is_random_choice : Prop) (is_seat_free : Fin n → Prop) (take_first_available_seat : Prop)

-- Prove that the last passenger will sit in their assigned seat with probability 1/2
theorem last_passenger_sits_in_assigned_seat :
  n > 0 → probability_last_passenger_assigned_seat n = 1 / 2 :=
by
  intro hn
  sorry

end last_passenger_sits_in_assigned_seat_l311_311086


namespace determine_multiplier_l311_311459

theorem determine_multiplier (x : ℝ) : 125 * x - 138 = 112 → x = 2 :=
by
  sorry

end determine_multiplier_l311_311459


namespace total_number_of_people_l311_311265

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l311_311265


namespace odd_factors_of_360_l311_311653

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l311_311653


namespace tetrahedron_distance_sum_l311_311641

theorem tetrahedron_distance_sum (S₁ S₂ S₃ S₄ H₁ H₂ H₃ H₄ V k : ℝ) 
  (h1 : S₁ = k) (h2 : S₂ = 2 * k) (h3 : S₃ = 3 * k) (h4 : S₄ = 4 * k)
  (V_eq : (1 / 3) * S₁ * H₁ + (1 / 3) * S₂ * H₂ + (1 / 3) * S₃ * H₃ + (1 / 3) * S₄ * H₄ = V) :
  1 * H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = (3 * V) / k :=
by
  sorry

end tetrahedron_distance_sum_l311_311641


namespace length_of_segment_BD_is_sqrt_3_l311_311052

open Real

-- Define the triangle ABC and the point D according to the problem conditions
def triangle_ABC (A B C : ℝ × ℝ) :=
  B.1 = 0 ∧ B.2 = 0 ∧
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = 3 ∧
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = 7 ∧
  C.2 = 0 ∧ (A.1 - C.1) ^ 2 + A.2 ^ 2 = 10

def point_D (A B C D : ℝ × ℝ) :=
  ∃ BD DC : ℝ, BD + DC = sqrt 7 ∧
  BD / DC = sqrt 3 / sqrt 7 ∧
  D.1 = BD / sqrt 7 ∧ D.2 = 0

-- The theorem to prove
theorem length_of_segment_BD_is_sqrt_3 (A B C D : ℝ × ℝ)
  (h₁ : triangle_ABC A B C)
  (h₂ : point_D A B C D) :
  (sqrt ((D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2)) = sqrt 3 :=
sorry

end length_of_segment_BD_is_sqrt_3_l311_311052


namespace dot_product_is_one_l311_311684

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (-1, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

theorem dot_product_is_one : dot_product vec_a vec_b = 1 :=
by sorry

end dot_product_is_one_l311_311684


namespace lattice_points_on_hyperbola_l311_311814

-- Define the problem
def countLatticePoints (n : ℤ) : ℕ :=
  let factoredCount := (2 + 1) * (2 + 1) * (4 + 1) -- Number of divisors of 2^2 * 3^2 * 5^4
  2 * factoredCount -- Each pair has two solutions considering positive and negative values

-- The theorem to be proven
theorem lattice_points_on_hyperbola : countLatticePoints 1800 = 90 := sorry

end lattice_points_on_hyperbola_l311_311814


namespace profit_percent_eq_20_l311_311601

-- Define cost price 'C' and original selling price 'S'
variable (C S : ℝ)

-- Hypothesis: selling at 2/3 of the original price results in a 20% loss 
def condition (C S : ℝ) : Prop :=
  (2 / 3) * S = 0.8 * C

-- Main theorem: profit percent when selling at the original price is 20%
theorem profit_percent_eq_20 (C S : ℝ) (h : condition C S) : (S - C) / C * 100 = 20 :=
by
  -- Proof steps would go here but we use sorry to indicate the proof is omitted
  sorry

end profit_percent_eq_20_l311_311601


namespace calc_expression_l311_311339

theorem calc_expression : 5 + 2 * (8 - 3) = 15 :=
by
  -- Proof steps would go here
  sorry

end calc_expression_l311_311339


namespace trigonometric_expression_value_l311_311270

-- Define the line equation and the conditions about the slope angle
def line_eq (x y : ℝ) : Prop := 6 * x - 2 * y - 5 = 0

-- The slope angle alpha
variable (α : ℝ)

-- Given conditions
axiom slope_tan : Real.tan α = 3

-- The expression we need to prove equals -2
theorem trigonometric_expression_value :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 :=
by
  sorry

end trigonometric_expression_value_l311_311270


namespace find_k_for_linear_dependence_l311_311780

structure vector2 :=
  (x : ℝ)
  (y : ℝ)

def linear_dependent (v1 v2 : vector2) :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
  c1 * v1.x + c2 * v2.x = 0 ∧
  c1 * v1.y + c2 * v2.y = 0

theorem find_k_for_linear_dependence :
  ∀ (k : ℝ), linear_dependent (vector2.mk 2 3) (vector2.mk 4 k) ↔ k = 6 :=
by sorry

end find_k_for_linear_dependence_l311_311780


namespace probability_divisor_of_8_is_half_l311_311884

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l311_311884


namespace count_triangles_with_positive_area_l311_311674

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l311_311674


namespace parabola_distance_relation_l311_311632

theorem parabola_distance_relation {n : ℝ} {x₁ x₂ y₁ y₂ : ℝ}
  (h₁ : y₁ = x₁^2 - 4 * x₁ + n)
  (h₂ : y₂ = x₂^2 - 4 * x₂ + n)
  (h : y₁ > y₂) :
  |x₁ - 2| > |x₂ - 2| := 
sorry

end parabola_distance_relation_l311_311632


namespace ted_age_l311_311011

variable (t s : ℕ)

theorem ted_age (h1 : t = 3 * s - 10) (h2 : t + s = 65) : t = 46 := by
  sorry

end ted_age_l311_311011


namespace min_point_transformed_graph_l311_311560

noncomputable def original_eq (x : ℝ) : ℝ := 2 * |x| - 4

noncomputable def translated_eq (x : ℝ) : ℝ := 2 * |x - 3| - 8

theorem min_point_transformed_graph : translated_eq 3 = -8 :=
by
  -- Solution steps would go here
  sorry

end min_point_transformed_graph_l311_311560


namespace gcd_5280_12155_l311_311297

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 5 :=
by
  sorry

end gcd_5280_12155_l311_311297


namespace hendecagon_diagonal_probability_l311_311867

theorem hendecagon_diagonal_probability :
  let n := 11
  let total_diagonals := (n * (n - 3) / 2)
  let total_pairs := (total_diagonals * (total_diagonals - 1) / 2)
  let intersecting_pairs := (n.choose 4)
  (intersecting_pairs / total_pairs) = (165 / 473) := by {
    
    sorry
  }

end hendecagon_diagonal_probability_l311_311867


namespace relationship_abc_l311_311494

noncomputable def a : ℝ := 4 / 5
noncomputable def b : ℝ := Real.sin (2 / 3)
noncomputable def c : ℝ := Real.cos (1 / 3)

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end relationship_abc_l311_311494


namespace magnesium_is_limiting_l311_311171

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end magnesium_is_limiting_l311_311171


namespace inequality_proof_l311_311078

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≥ b) (h5 : b ≥ c) :
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ∧
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) :=
by
  sorry

end inequality_proof_l311_311078


namespace ben_gave_18_fish_l311_311536

variable (initial_fish : ℕ) (total_fish : ℕ) (given_fish : ℕ)

theorem ben_gave_18_fish
    (h1 : initial_fish = 31)
    (h2 : total_fish = 49)
    (h3 : total_fish = initial_fish + given_fish) :
    given_fish = 18 :=
by
  sorry

end ben_gave_18_fish_l311_311536


namespace percentage_of_primes_divisible_by_3_l311_311273

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def count (p : ℕ → Prop) (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if p x then acc + 1 else acc) 0

def percentage (num denom : ℕ) : ℝ := 
  (num.toFloat / denom.toFloat) * 100.0

theorem percentage_of_primes_divisible_by_3 : percentage (count is_divisible_by_three primes_less_than_twenty) (primes_less_than_twenty.length) = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_l311_311273


namespace select_p_elements_with_integer_mean_l311_311062

theorem select_p_elements_with_integer_mean {p : ℕ} (hp : Nat.Prime p) (p_odd : p % 2 = 1) :
  ∃ (M : Finset ℕ), (M.card = (p^2 + 1) / 2) ∧ ∃ (S : Finset ℕ), (S.card = p) ∧ ((S.sum id) % p = 0) :=
by
  -- sorry to skip the proof
  sorry

end select_p_elements_with_integer_mean_l311_311062


namespace new_train_distance_l311_311892

-- Define the given conditions
def distance_old : ℝ := 300
def percentage_increase : ℝ := 0.3

-- Define the target distance to prove
def distance_new : ℝ := distance_old + (percentage_increase * distance_old)

-- State the theorem
theorem new_train_distance : distance_new = 390 := by
  sorry

end new_train_distance_l311_311892


namespace travel_distance_proof_l311_311313

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end travel_distance_proof_l311_311313


namespace driver_weekly_distance_l311_311122

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end driver_weekly_distance_l311_311122


namespace line_intersects_circle_midpoint_trajectory_l311_311113

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

def line_eq (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Statement of the problem
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

theorem midpoint_trajectory :
  ∀ (x y : ℝ), 
  (∃ (xa ya xb yb : ℝ), circle_eq xa ya ∧ line_eq m xa ya ∧ 
   circle_eq xb yb ∧ line_eq m xb yb ∧ (x, y) = ((xa + xb) / 2, (ya + yb) / 2)) ↔
   ( x - 1 / 2)^2 + (y - 1)^2 = 1 / 4 :=
sorry

end line_intersects_circle_midpoint_trajectory_l311_311113


namespace fraction_operation_l311_311900

theorem fraction_operation : (3 / 5 - 1 / 10 + 2 / 15 = 19 / 30) :=
by
  sorry

end fraction_operation_l311_311900


namespace binary_mul_correct_l311_311336

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l311_311336


namespace no_such_function_exists_l311_311022

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 :=
by
  sorry

end no_such_function_exists_l311_311022


namespace total_sleep_correct_l311_311398

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l311_311398


namespace Ram_has_amount_l311_311430

theorem Ram_has_amount (R G K : ℕ)
    (h1 : R = 7 * G / 17)
    (h2 : G = 7 * K / 17)
    (h3 : K = 3757) : R = 637 := by
  sorry

end Ram_has_amount_l311_311430


namespace find_jamals_grade_l311_311517

noncomputable def jamals_grade (n_students : ℕ) (absent_students : ℕ) (test_avg_28_students : ℕ) (new_total_avg_30_students : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let total_28_students := 28 * test_avg_28_students
  let total_30_students := 30 * new_total_avg_30_students
  let combined_score := total_30_students - total_28_students
  combined_score - taqeesha_score

theorem find_jamals_grade :
  jamals_grade 30 2 85 86 92 = 108 :=
by
  sorry

end find_jamals_grade_l311_311517


namespace choose_4_cards_of_different_suits_l311_311189

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l311_311189


namespace find_sides_from_diagonals_l311_311978

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l311_311978


namespace evaluate_fg_neg3_l311_311620

theorem evaluate_fg_neg3 :
  let f (x : ℝ) := 3 - Real.sqrt x
  let g (x : ℝ) := -x + 3 * x^2
  f (g (-3)) = 3 - Real.sqrt 30 :=
by
  sorry

end evaluate_fg_neg3_l311_311620


namespace spiders_loose_l311_311129

noncomputable def initial_birds : ℕ := 12
noncomputable def initial_puppies : ℕ := 9
noncomputable def initial_cats : ℕ := 5
noncomputable def initial_spiders : ℕ := 15
noncomputable def birds_sold : ℕ := initial_birds / 2
noncomputable def puppies_adopted : ℕ := 3
noncomputable def remaining_puppies : ℕ := initial_puppies - puppies_adopted
noncomputable def remaining_cats : ℕ := initial_cats
noncomputable def total_remaining_animals_except_spiders : ℕ := birds_sold + remaining_puppies + remaining_cats
noncomputable def total_animals_left : ℕ := 25
noncomputable def remaining_spiders : ℕ := total_animals_left - total_remaining_animals_except_spiders
noncomputable def spiders_went_loose : ℕ := initial_spiders - remaining_spiders

theorem spiders_loose : spiders_went_loose = 7 := by
  sorry

end spiders_loose_l311_311129


namespace probability_of_at_least_one_three_l311_311435

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l311_311435


namespace value_of_a_plus_d_l311_311512

variable (a b c d : ℝ)

theorem value_of_a_plus_d
  (h1 : a + b = 4)
  (h2 : b + c = 5)
  (h3 : c + d = 3) :
  a + d = 1 :=
by
sorry

end value_of_a_plus_d_l311_311512


namespace period_is_seven_l311_311916

-- Define the conditions
def apples_per_sandwich (a : ℕ) := a = 4
def sandwiches_per_day (s : ℕ) := s = 10
def total_apples (t : ℕ) := t = 280

-- Define the question to prove the period
theorem period_is_seven (a s t d : ℕ) 
  (h1 : apples_per_sandwich a)
  (h2 : sandwiches_per_day s)
  (h3 : total_apples t)
  (h4 : d = t / (a * s)) 
  : d = 7 := 
sorry

end period_is_seven_l311_311916


namespace geometric_sequence_sum_l311_311682

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (a_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 1) (h_a3a7_a5 : a 3 * a 7 - a 5 = 56)
  (S_eq : ∀ n, S n = (a 1 * (1 - (2 : ℝ) ^ n)) / (1 - 2)) :
  S 5 = 31 / 2 := by
  sorry

end geometric_sequence_sum_l311_311682


namespace wyatt_bought_4_cartons_of_juice_l311_311105

/-- 
Wyatt's mother gave him $74 to go to the store.
Wyatt bought 5 loaves of bread, each costing $5.
Each carton of orange juice cost $2.
Wyatt has $41 left.
We need to prove that Wyatt bought 4 cartons of orange juice.
-/
theorem wyatt_bought_4_cartons_of_juice (initial_money spent_money loaves_price juice_price loaves_qty money_left juice_qty : ℕ)
  (h1 : initial_money = 74)
  (h2 : money_left = 41)
  (h3 : loaves_price = 5)
  (h4 : juice_price = 2)
  (h5 : loaves_qty = 5)
  (h6 : spent_money = initial_money - money_left)
  (h7 : spent_money = loaves_qty * loaves_price + juice_qty * juice_price) :
  juice_qty = 4 :=
by
  -- the proof would go here
  sorry

end wyatt_bought_4_cartons_of_juice_l311_311105


namespace find_line_through_M_and_parallel_l311_311914
-- Lean code to represent the proof problem

def M : Prop := ∃ (x y : ℝ), 3 * x + 4 * y - 5 = 0 ∧ 2 * x - 3 * y + 8 = 0 

def line_parallel : Prop := ∃ (m b : ℝ), 2 * m + b = 0

theorem find_line_through_M_and_parallel :
  M → line_parallel → ∃ (a b c : ℝ), (a = 2) ∧ (b = 1) ∧ (c = 0) :=
by
  intros hM hLineParallel
  sorry

end find_line_through_M_and_parallel_l311_311914


namespace number_of_books_bought_l311_311407

def initial_books : ℕ := 35
def books_given_away : ℕ := 12
def final_books : ℕ := 56

theorem number_of_books_bought : initial_books - books_given_away + (final_books - (initial_books - books_given_away)) = final_books :=
by
  sorry

end number_of_books_bought_l311_311407


namespace find_x_given_k_l311_311343

-- Define the equation under consideration
def equation (x : ℝ) : Prop := (x - 3) / (x - 4) = (x - 5) / (x - 8)

theorem find_x_given_k {k : ℝ} (h : k = 7) : ∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → equation x → x = 2 :=
by
  intro x hx h_eq
  sorry

end find_x_given_k_l311_311343


namespace find_sides_from_diagonals_l311_311979

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l311_311979


namespace initial_parking_hours_proof_l311_311557

noncomputable def initial_parking_hours (total_cost : ℝ) (excess_hourly_rate : ℝ) (average_cost : ℝ) (total_hours : ℕ) : ℝ :=
  let h := (total_hours * average_cost - total_cost) / excess_hourly_rate
  h

theorem initial_parking_hours_proof : initial_parking_hours 21.25 1.75 2.361111111111111 9 = 2 :=
by
  sorry

end initial_parking_hours_proof_l311_311557


namespace no_partition_with_sum_k_plus_2013_l311_311220

open Nat

theorem no_partition_with_sum_k_plus_2013 (A : ℕ → Finset ℕ) (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j)) 
  (h_sum : ∀ k, (A k).sum id = k + 2013) : False :=
by
  sorry

end no_partition_with_sum_k_plus_2013_l311_311220


namespace Ryanne_is_7_years_older_than_Hezekiah_l311_311079

theorem Ryanne_is_7_years_older_than_Hezekiah
  (H : ℕ) (R : ℕ)
  (h1 : H = 4)
  (h2 : R + H = 15) :
  R - H = 7 := by
  sorry

end Ryanne_is_7_years_older_than_Hezekiah_l311_311079


namespace functions_satisfying_equation_l311_311781

theorem functions_satisfying_equation 
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, g x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, h x = a * x + b) :=
sorry

end functions_satisfying_equation_l311_311781


namespace max_distinct_integer_solutions_le_2_l311_311346

def f (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem max_distinct_integer_solutions_le_2 
  (a b c : ℝ) (h₀ : a > 100) :
  ∀ (x : ℤ), |f a b c (x : ℝ)| ≤ 50 → 
  ∃ (x₁ x₂ : ℤ), x = x₁ ∨ x = x₂ :=
by
  sorry

end max_distinct_integer_solutions_le_2_l311_311346


namespace stratified_sample_selection_l311_311761

def TotalStudents : ℕ := 900
def FirstYearStudents : ℕ := 300
def SecondYearStudents : ℕ := 200
def ThirdYearStudents : ℕ := 400
def SampleSize : ℕ := 45
def SamplingRatio : ℚ := 1 / 20

theorem stratified_sample_selection :
  (FirstYearStudents * SamplingRatio = 15) ∧
  (SecondYearStudents * SamplingRatio = 10) ∧
  (ThirdYearStudents * SamplingRatio = 20) :=
by
  sorry

end stratified_sample_selection_l311_311761


namespace remainder_add_l311_311704

theorem remainder_add (a b : ℤ) (n m : ℤ) 
  (ha : a = 60 * n + 41) 
  (hb : b = 45 * m + 14) : 
  (a + b) % 15 = 10 := by 
  sorry

end remainder_add_l311_311704


namespace probability_at_least_one_three_l311_311439

theorem probability_at_least_one_three :
  let E := { (d1, d2) : Fin 8 × Fin 8 | d1 = 2 ∨ d2 = 2 } in
  (↑E.card / ↑((Fin 8 × Fin 8).card) : ℚ) = 15 / 64 :=
by
  /- Let E be the set of outcomes where at least one die shows a 3. -/
  sorry

end probability_at_least_one_three_l311_311439


namespace fraction_under_11_is_one_third_l311_311073

def fraction_under_11 (T : ℕ) (fraction_above_11_under_13 : ℚ) (students_above_13 : ℕ) : ℚ :=
  let fraction_under_11 := 1 - (fraction_above_11_under_13 + students_above_13 / T)
  fraction_under_11

theorem fraction_under_11_is_one_third :
  fraction_under_11 45 (2/5) 12 = 1/3 :=
by
  sorry

end fraction_under_11_is_one_third_l311_311073


namespace combustion_CH₄_forming_water_l311_311157

/-
Combustion reaction for Methane: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Given:
  3 moles of Methane
  6 moles of Oxygen
  Balanced equation: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Goal: Prove that 6 moles of Water (H₂O) are formed.
-/

-- Define the necessary definitions for the context
def moles_CH₄ : ℝ := 3
def moles_O₂ : ℝ := 6
def ratio_water_methane : ℝ := 2

theorem combustion_CH₄_forming_water :
  moles_CH₄ * ratio_water_methane = 6 :=
by
  sorry

end combustion_CH₄_forming_water_l311_311157


namespace find_physics_marks_l311_311451

theorem find_physics_marks (P C M : ℕ) (h1 : P + C + M = 210) (h2 : P + M = 180) (h3 : P + C = 140) : P = 110 :=
sorry

end find_physics_marks_l311_311451


namespace equation_contains_2020_l311_311173

def first_term (n : Nat) : Nat :=
  2 * n^2

theorem equation_contains_2020 :
  ∃ n, first_term n = 2020 :=
by
  use 31
  sorry

end equation_contains_2020_l311_311173


namespace find_sides_from_diagonals_l311_311980

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l311_311980


namespace no_integer_roots_l311_311839

theorem no_integer_roots (a b c : ℤ) (h1 : a ≠ 0) (h2 : a % 2 = 1) (h3 : b % 2 = 1) (h4 : c % 2 = 1) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 :=
by
  sorry

end no_integer_roots_l311_311839


namespace possible_values_of_m_l311_311790

theorem possible_values_of_m (k m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 4 → y = k * x + m) →
  (∃ (d : ℝ) (a : ℝ), d = sqrt (4 - (a / 2)^2) ∧ d = sqrt 3 ∧ a = 2) →
  (m = sqrt 3 ∨ m = -sqrt 3) :=
by sorry

end possible_values_of_m_l311_311790


namespace carrie_profit_l311_311607

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end carrie_profit_l311_311607


namespace cos_pi_zero_l311_311708

theorem cos_pi_zero : ∃ f : ℝ → ℝ, (∀ x, f x = (Real.cos x) ^ 2 + Real.cos x) ∧ f Real.pi = 0 := by
  sorry

end cos_pi_zero_l311_311708


namespace odd_factors_of_360_l311_311649

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l311_311649


namespace arithmetic_seq_solution_l311_311248

variables (a : ℕ → ℤ) (d : ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) - a n = d

def seq_cond (a : ℕ → ℤ) (d : ℤ) : Prop :=
is_arithmetic_sequence a d ∧ (a 2 + a 6 = a 8)

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem arithmetic_seq_solution :
  ∀ (a : ℕ → ℤ) (d : ℤ), seq_cond a d → (a 2 - a 1 ≠ 0) → 
    (sum_first_n a 5 / a 5) = 3 :=
by
  intros a d h_cond h_d_ne_zero
  sorry

end arithmetic_seq_solution_l311_311248


namespace trigonometric_expression_value_l311_311036

variable (θ : ℝ)

-- Conditions
axiom tan_theta_eq_two : Real.tan θ = 2

-- Theorem to prove
theorem trigonometric_expression_value : 
  Real.sin θ * Real.sin θ + 
  Real.sin θ * Real.cos θ - 
  2 * Real.cos θ * Real.cos θ = 4 / 5 := 
by
  sorry

end trigonometric_expression_value_l311_311036


namespace minimum_value_of_f_l311_311801

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem minimum_value_of_f :
  ∃ x : ℝ, f x = -(4 / 3) :=
by
  use 2
  have hf : f 2 = -(4 / 3) := by
    sorry
  exact hf

end minimum_value_of_f_l311_311801


namespace binary_multiplication_correct_l311_311337

theorem binary_multiplication_correct :
  nat.of_digits 2 [1, 0, 1, 0, 0, 1, 1] = 
  (nat.of_digits 2 [1, 1, 0, 1] * nat.of_digits 2 [1, 1, 1]) :=
by
  sorry

end binary_multiplication_correct_l311_311337


namespace increasing_function_iff_l311_311174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a ^ x else (3 - a) * x + (1 / 2) * a

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 ≤ a ∧ a < 3 :=
by
  sorry

end increasing_function_iff_l311_311174


namespace probability_divisor_of_8_is_half_l311_311887

theorem probability_divisor_of_8_is_half :
  let outcomes := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let divisors_of_8 := [ 1, 2, 4, 8 ]
  let favorable_outcomes := list.filter (λ x, x ∣ 8) outcomes
  let favorable_probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  favorable_probability = (1 / 2 : ℚ) := by
  sorry

end probability_divisor_of_8_is_half_l311_311887


namespace sqrt_sum_eq_fraction_l311_311680

-- Definitions as per conditions
def w : ℕ := 4
def x : ℕ := 9
def z : ℕ := 25

-- Main theorem statement
theorem sqrt_sum_eq_fraction : (Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15) := by
  sorry

end sqrt_sum_eq_fraction_l311_311680


namespace seventh_graders_more_than_sixth_graders_l311_311845

-- Definitions based on conditions
variables (S6 S7 : ℕ)
variable (h : 7 * S6 = 6 * S7)

-- Proposition based on the conclusion
theorem seventh_graders_more_than_sixth_graders (h : 7 * S6 = 6 * S7) : S7 > S6 :=
by {
  -- Skipping the proof with sorry
  sorry
}

end seventh_graders_more_than_sixth_graders_l311_311845


namespace liquid_X_percentage_correct_l311_311605

noncomputable def percent_liquid_X_in_solution_A := 0.8 / 100
noncomputable def percent_liquid_X_in_solution_B := 1.8 / 100

noncomputable def weight_solution_A := 400.0
noncomputable def weight_solution_B := 700.0

noncomputable def weight_liquid_X_in_A := percent_liquid_X_in_solution_A * weight_solution_A
noncomputable def weight_liquid_X_in_B := percent_liquid_X_in_solution_B * weight_solution_B

noncomputable def total_weight_solution := weight_solution_A + weight_solution_B
noncomputable def total_weight_liquid_X := weight_liquid_X_in_A + weight_liquid_X_in_B

noncomputable def percent_liquid_X_in_mixed_solution := (total_weight_liquid_X / total_weight_solution) * 100

theorem liquid_X_percentage_correct :
  percent_liquid_X_in_mixed_solution = 1.44 :=
by
  sorry

end liquid_X_percentage_correct_l311_311605


namespace equation_relationship_linear_l311_311151

theorem equation_relationship_linear 
  (x y : ℕ)
  (h1 : (x, y) = (0, 200) ∨ (x, y) = (1, 160) ∨ (x, y) = (2, 120) ∨ (x, y) = (3, 80) ∨ (x, y) = (4, 40)) :
  y = 200 - 40 * x :=
  sorry

end equation_relationship_linear_l311_311151


namespace number_of_odd_factors_of_360_l311_311662

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l311_311662


namespace polygon_diagonals_l311_311987

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l311_311987


namespace product_of_sums_of_conjugates_l311_311470

theorem product_of_sums_of_conjugates :
  let a := 8 - Real.sqrt 500
  let b := 8 + Real.sqrt 500
  let c := 12 - Real.sqrt 72
  let d := 12 + Real.sqrt 72
  (a + b) * (c + d) = 384 :=
by
  sorry

end product_of_sums_of_conjugates_l311_311470


namespace multiple_of_other_number_l311_311539

theorem multiple_of_other_number (S L k : ℤ) (h₁ : S = 18) (h₂ : L = k * S - 3) (h₃ : S + L = 51) : k = 2 :=
by
  sorry

end multiple_of_other_number_l311_311539


namespace inequality_solution_l311_311640

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 → x < 1) ↔ a < -1 := by
  sorry

end inequality_solution_l311_311640


namespace john_order_cost_l311_311457

-- Definitions from the problem conditions
def discount_rate : ℝ := 0.10
def item_price : ℝ := 200
def num_items : ℕ := 7
def discount_threshold : ℝ := 1000

-- Final proof statement
theorem john_order_cost : 
  (num_items * item_price) - 
  (if (num_items * item_price) > discount_threshold then 
    discount_rate * ((num_items * item_price) - discount_threshold) 
  else 0) = 1360 := 
sorry

end john_order_cost_l311_311457


namespace quadratic_solution_l311_311243

theorem quadratic_solution (x : ℝ) : 
  x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by 
  sorry

end quadratic_solution_l311_311243


namespace op_7_3_eq_70_l311_311035

noncomputable def op (x y : ℝ) : ℝ := sorry

axiom ax1 : ∀ x : ℝ, op x 0 = x
axiom ax2 : ∀ x y : ℝ, op x y = op y x
axiom ax3 : ∀ x y : ℝ, op (x + 1) y = (op x y) + y + 2

theorem op_7_3_eq_70 : op 7 3 = 70 := by
  sorry

end op_7_3_eq_70_l311_311035


namespace train_pass_bridge_time_l311_311461

noncomputable def trainLength : ℝ := 360
noncomputable def trainSpeedKMH : ℝ := 45
noncomputable def bridgeLength : ℝ := 160
noncomputable def totalDistance : ℝ := trainLength + bridgeLength
noncomputable def trainSpeedMS : ℝ := trainSpeedKMH * (1000 / 3600)
noncomputable def timeToPassBridge : ℝ := totalDistance / trainSpeedMS

theorem train_pass_bridge_time : timeToPassBridge = 41.6 := sorry

end train_pass_bridge_time_l311_311461


namespace binary_101_is_5_l311_311480

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l311_311480


namespace pool_capacity_l311_311765

theorem pool_capacity (C : ℝ) (h1 : 300 = 0.30 * C) : C = 1000 :=
by
  sorry

end pool_capacity_l311_311765


namespace hyperbola_eccentricity_l311_311166

variable (a b c e : ℝ)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (hyperbola_eq : c = Real.sqrt (a^2 + b^2))
variable (y_B : ℝ)
variable (slope_eq : 3 = (y_B - 0) / (c - a))
variable (y_B_on_hyperbola : y_B = b^2 / a)

theorem hyperbola_eccentricity (h : a > 0) (h' : b > 0) (c_def : c = Real.sqrt (a^2 + b^2))
    (slope_cond : 3 = (y_B - 0) / (c - a)) (y_B_cond : y_B = b^2 / a) :
    e = 2 :=
sorry

end hyperbola_eccentricity_l311_311166


namespace lattice_points_on_hyperbola_l311_311816

theorem lattice_points_on_hyperbola :
  ∃ (n : ℕ), n = 90 ∧
  (∀ (x y : ℤ), x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | true} ) :=
begin
  -- Convert mathematical conditions to Lean definitions
  let a := 1800^2,
  have even_factors : (∀ (x y : ℤ), (x - y) * (x + y) = a → even (x - y) ∧ even (x+y)),
  {
    sorry,
  },
  -- Assert the number of lattice points is 90
  use [90],
  split; simp,
  sorry,
end

end lattice_points_on_hyperbola_l311_311816


namespace percentage_of_employees_driving_l311_311759

theorem percentage_of_employees_driving
  (total_employees : ℕ)
  (drivers : ℕ)
  (public_transport : ℕ)
  (H1 : total_employees = 200)
  (H2 : drivers = public_transport + 40)
  (H3 : public_transport = (total_employees - drivers) / 2) :
  (drivers:ℝ) / (total_employees:ℝ) * 100 = 46.5 :=
by {
  sorry
}

end percentage_of_employees_driving_l311_311759


namespace boys_collected_200_insects_l311_311717

theorem boys_collected_200_insects
  (girls_insects : ℕ)
  (groups : ℕ)
  (insects_per_group : ℕ)
  (total_insects : ℕ)
  (boys_insects : ℕ)
  (H1 : girls_insects = 300)
  (H2 : groups = 4)
  (H3 : insects_per_group = 125)
  (H4 : total_insects = groups * insects_per_group)
  (H5 : boys_insects = total_insects - girls_insects) :
  boys_insects = 200 :=
  by sorry

end boys_collected_200_insects_l311_311717


namespace abs_diff_of_sum_and_product_l311_311861

theorem abs_diff_of_sum_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_sum_and_product_l311_311861


namespace cadence_old_company_salary_l311_311147

variable (S : ℝ)

def oldCompanyMonths : ℝ := 36
def newCompanyMonths : ℝ := 41
def newSalaryMultiplier : ℝ := 1.20
def totalEarnings : ℝ := 426000

theorem cadence_old_company_salary :
  (oldCompanyMonths * S) + (newCompanyMonths * newSalaryMultiplier * S) = totalEarnings → 
  S = 5000 :=
by
  sorry

end cadence_old_company_salary_l311_311147


namespace store_paid_price_l311_311728

-- Definition of the conditions
def selling_price : ℕ := 34
def difference_price : ℕ := 8

-- Statement that needs to be proven.
theorem store_paid_price : (selling_price - difference_price) = 26 :=
by
  sorry

end store_paid_price_l311_311728


namespace number_of_cows_l311_311823

theorem number_of_cows (x y : ℕ) 
  (h1 : 4 * x + 2 * y = 14 + 2 * (x + y)) : 
  x = 7 :=
by
  sorry

end number_of_cows_l311_311823


namespace tan_cos_identity_l311_311469

theorem tan_cos_identity :
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  (tanθ^2 - cosθ^2) / (tanθ^2 * cosθ^2) = -5 / 3 :=
by
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  have h_tan : tanθ^2 = (Real.sin θ)^2 / (Real.cos θ)^2 := by sorry; -- Given condition 1
  have h_cos : cosθ^2 = 3 / 4 := by sorry; -- Given condition 2
  -- Prove the statement
  sorry

end tan_cos_identity_l311_311469


namespace infinite_triangle_area_sum_l311_311401

noncomputable def rectangle_area_sum : ℝ :=
  let AB := 2
  let BC := 1
  let Q₁ := 0.5
  let base_area := (1/2) * Q₁ * (1/4)
  base_area * (1/(1 - 1/4))

theorem infinite_triangle_area_sum :
  rectangle_area_sum = 1/12 :=
by
  sorry

end infinite_triangle_area_sum_l311_311401


namespace price_of_each_lemon_square_l311_311055

-- Given
def brownies_sold : Nat := 4
def price_per_brownie : Nat := 3
def lemon_squares_sold : Nat := 5
def goal_amount : Nat := 50
def cookies_sold : Nat := 7
def price_per_cookie : Nat := 4

-- Prove
theorem price_of_each_lemon_square :
  (brownies_sold * price_per_brownie + lemon_squares_sold * L + cookies_sold * price_per_cookie = goal_amount) →
  L = 2 :=
by
  sorry

end price_of_each_lemon_square_l311_311055


namespace cardinals_count_l311_311320

theorem cardinals_count (C R B S : ℕ) 
  (hR : R = 4 * C)
  (hB : B = 2 * C)
  (hS : S = 3 * C + 1)
  (h_total : C + R + B + S = 31) :
  C = 3 :=
by
  sorry

end cardinals_count_l311_311320


namespace coprime_powers_l311_311843

theorem coprime_powers (n : ℕ) : Nat.gcd (n^5 + 4 * n^3 + 3 * n) (n^4 + 3 * n^2 + 1) = 1 :=
sorry

end coprime_powers_l311_311843


namespace unique_triangles_count_l311_311944

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l311_311944


namespace bobby_pizzas_l311_311072

theorem bobby_pizzas (B : ℕ) (h_slices : (1 / 4 : ℝ) * B = 3) (h_slices_per_pizza : 6 > 0) :
  B / 6 = 2 := by
  sorry

end bobby_pizzas_l311_311072


namespace popcorn_probability_l311_311117

theorem popcorn_probability {w y b : ℝ} (hw : w = 3/5) (hy : y = 1/5) (hb : b = 1/5)
  {pw py pb : ℝ} (hpw : pw = 1/3) (hpy : py = 3/4) (hpb : pb = 1/2) :
  (y * py) / (w * pw + y * py + b * pb) = 1/3 := 
sorry

end popcorn_probability_l311_311117


namespace train_length_is_1400_l311_311771

theorem train_length_is_1400
  (L : ℝ) 
  (h1 : ∃ speed, speed = L / 100) 
  (h2 : ∃ speed, speed = (L + 700) / 150) :
  L = 1400 :=
by sorry

end train_length_is_1400_l311_311771


namespace number_of_sides_of_polygon_24_deg_exterior_angle_l311_311426

theorem number_of_sides_of_polygon_24_deg_exterior_angle :
  (∀ (n : ℕ), (∀ (k : ℕ), k = 360 / 24 → n = k)) :=
by
  sorry

end number_of_sides_of_polygon_24_deg_exterior_angle_l311_311426


namespace lucas_change_l311_311534

-- Define the costs of items and the initial amount.
def initial_amount : ℝ := 20.00
def cost_avocados : ℝ := 1.50 + 2.25 + 3.00
def cost_water : ℝ := 2 * 1.75
def cost_apples : ℝ := 4 * 0.75

-- Define the total cost.
def total_cost : ℝ := cost_avocados + cost_water + cost_apples

-- Define the expected change.
def expected_change : ℝ := initial_amount - total_cost

-- The proposition (statement) we want to prove.
theorem lucas_change : expected_change = 6.75 :=
by
  sorry -- Proof to be completed.

end lucas_change_l311_311534


namespace max_gcd_of_polynomials_l311_311775

def max_gcd (a b : ℤ) : ℤ :=
  let g := Nat.gcd a.natAbs b.natAbs
  Int.ofNat g

theorem max_gcd_of_polynomials :
  ∃ n : ℕ, (n > 0) → max_gcd (14 * ↑n + 5) (9 * ↑n + 2) = 4 :=
by
  sorry

end max_gcd_of_polynomials_l311_311775


namespace product_p_yi_eq_neg26_l311_311065

-- Definitions of the polynomials h and p.
def h (y : ℂ) : ℂ := y^3 - 3 * y + 1
def p (y : ℂ) : ℂ := y^3 + 2

-- Given that y1, y2, y3 are roots of h(y)
variables (y1 y2 y3 : ℂ) (H1 : h y1 = 0) (H2 : h y2 = 0) (H3 : h y3 = 0)

-- State the theorem to show p(y1) * p(y2) * p(y3) = -26
theorem product_p_yi_eq_neg26 : p y1 * p y2 * p y3 = -26 :=
sorry

end product_p_yi_eq_neg26_l311_311065


namespace binary_101_to_decimal_l311_311474

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l311_311474


namespace binary_mul_correct_l311_311335

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l311_311335


namespace acceptable_colorings_correct_l311_311628

def acceptableColorings (n : ℕ) : ℕ :=
  (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2

theorem acceptable_colorings_correct (n : ℕ) :
  acceptableColorings n = (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2 :=
by
  sorry

end acceptable_colorings_correct_l311_311628


namespace grapefruits_orchards_proof_l311_311592

/-- 
Given the following conditions:
1. There are 40 orchards in total.
2. 15 orchards are dedicated to lemons.
3. The number of orchards for oranges is two-thirds of the number of orchards for lemons.
4. Limes and grapefruits have an equal number of orchards.
5. Mandarins have half as many orchards as limes or grapefruits.
Prove that the number of citrus orchards growing grapefruits is 6.
-/
def num_grapefruit_orchards (TotalOrchards Lemons Oranges L G M : ℕ) : Prop :=
  TotalOrchards = 40 ∧
  Lemons = 15 ∧
  Oranges = 2 * Lemons / 3 ∧
  L = G ∧
  M = G / 2 ∧
  L + G + M = TotalOrchards - (Lemons + Oranges) ∧
  G = 6

theorem grapefruits_orchards_proof : ∃ (TotalOrchards Lemons Oranges L G M : ℕ), num_grapefruit_orchards TotalOrchards Lemons Oranges L G M :=
by
  sorry

end grapefruits_orchards_proof_l311_311592


namespace percent_of_x_is_y_in_terms_of_z_l311_311042

theorem percent_of_x_is_y_in_terms_of_z (x y z : ℝ) (h1 : 0.7 * (x - y) = 0.3 * (x + y))
    (h2 : 0.6 * (x + z) = 0.4 * (y - z)) : y / x = 0.4 :=
  sorry

end percent_of_x_is_y_in_terms_of_z_l311_311042


namespace problem_remainder_6_pow_83_add_8_pow_83_mod_49_l311_311028

-- Definitions based on the conditions.
def euler_totient_49 : ℕ := 42

theorem problem_remainder_6_pow_83_add_8_pow_83_mod_49 
  (h1 : 6 ^ euler_totient_49 ≡ 1 [MOD 49])
  (h2 : 8 ^ euler_totient_49 ≡ 1 [MOD 49]) :
  (6 ^ 83 + 8 ^ 83) % 49 = 35 :=
by
  sorry

end problem_remainder_6_pow_83_add_8_pow_83_mod_49_l311_311028


namespace max_sheep_pen_area_l311_311599

theorem max_sheep_pen_area :
  ∃ x y : ℝ, 15 * 2 = 30 ∧ (x + 2 * y = 30) ∧
  (x > 0 ∧ y > 0) ∧
  (x * y = 112) := by
  sorry

end max_sheep_pen_area_l311_311599


namespace minimum_cost_to_store_food_l311_311873

-- Define the problem setting
def total_volume : ℕ := 15
def capacity_A : ℕ := 2
def capacity_B : ℕ := 3
def price_A : ℕ := 13
def price_B : ℕ := 15
def cashback_threshold : ℕ := 3
def cashback : ℕ := 10

-- The mathematical theorem statement for the proof problem
theorem minimum_cost_to_store_food : 
  ∃ (x y : ℕ), 
    capacity_A * x + capacity_B * y = total_volume ∧ 
    (y = 5 ∧ price_B * y = 75) ∨ 
    (x = 3 ∧ y = 3 ∧ price_A * x + price_B * y - cashback = 74) :=
sorry

end minimum_cost_to_store_food_l311_311873


namespace overall_loss_percentage_l311_311556

theorem overall_loss_percentage
  (cost_price : ℝ)
  (discount : ℝ)
  (sales_tax : ℝ)
  (depreciation : ℝ)
  (final_selling_price : ℝ) :
  cost_price = 1900 →
  discount = 0.15 →
  sales_tax = 0.12 →
  depreciation = 0.05 →
  final_selling_price = 1330 →
  ((cost_price - (discount * cost_price)) * (1 + sales_tax) * (1 - depreciation) - final_selling_price) / cost_price * 100 = 20.44 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overall_loss_percentage_l311_311556


namespace abs_neg_seventeen_l311_311901

theorem abs_neg_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end abs_neg_seventeen_l311_311901


namespace grid_labelings_count_l311_311860

theorem grid_labelings_count :
  ∃ (labeling_count : ℕ), 
    labeling_count = 2448 ∧ 
    (∀ (grid : Matrix (Fin 3) (Fin 3) ℕ),
      grid 0 0 = 1 ∧ 
      grid 2 2 = 2009 ∧ 
      (∀ (i j : Fin 3), j < 2 → grid i j ∣ grid i (j + 1)) ∧ 
      (∀ (i j : Fin 3), i < 2 → grid i j ∣ grid (i + 1) j)) :=
sorry

end grid_labelings_count_l311_311860


namespace probability_Q_within_2_of_origin_eq_pi_div_9_l311_311000

noncomputable def probability_within_circle (π : ℝ) : ℝ :=
  let area_of_square := (2 * 3)^2
  let area_of_circle := π * 2^2
  area_of_circle / area_of_square

theorem probability_Q_within_2_of_origin_eq_pi_div_9 :
  probability_within_circle Real.pi = Real.pi / 9 :=
by
  sorry

end probability_Q_within_2_of_origin_eq_pi_div_9_l311_311000


namespace budget_allocation_genetically_modified_microorganisms_l311_311590

theorem budget_allocation_genetically_modified_microorganisms :
  let microphotonics := 14
  let home_electronics := 19
  let food_additives := 10
  let industrial_lubricants := 8
  let total_percentage := 100
  let basic_astrophysics_percentage := 25
  let known_percentage := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let genetically_modified_microorganisms := total_percentage - known_percentage
  genetically_modified_microorganisms = 24 := 
by
  sorry

end budget_allocation_genetically_modified_microorganisms_l311_311590


namespace find_original_number_l311_311625

theorem find_original_number (x : ℝ) (h : 0.5 * x = 30) : x = 60 :=
sorry

end find_original_number_l311_311625


namespace time_to_cross_bridge_l311_311455

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (speed_conversion_factor : ℝ) (time_conversion_factor : ℝ) (expected_time : ℝ) :
  speed_km_hr = 5 →
  length_m = 1250 →
  speed_conversion_factor = 1000 →
  time_conversion_factor = 60 →
  expected_time = length_m / (speed_km_hr * (speed_conversion_factor / time_conversion_factor)) →
  expected_time = 15 :=
by
  intros
  sorry

end time_to_cross_bridge_l311_311455


namespace number_of_non_congruent_triangles_perimeter_18_l311_311957

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l311_311957


namespace max_slope_no_lattice_points_l311_311615

theorem max_slope_no_lattice_points :
  (∃ b : ℚ, (∀ m : ℚ, 1 / 3 < m ∧ m < b → ∀ x : ℤ, 0 < x ∧ x ≤ 200 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ b = 68 / 203) := 
sorry

end max_slope_no_lattice_points_l311_311615


namespace xiaozhang_participates_in_martial_arts_l311_311550

theorem xiaozhang_participates_in_martial_arts
  (row : Prop) (shoot : Prop) (martial : Prop)
  (Zhang Wang Li: Prop → Prop)
  (H1 : ¬  Zhang row ∧ ¬ Wang row)
  (H2 : ∃ (n m : ℕ), Zhang (shoot ∨ martial) = (n > 0) ∧ Wang (shoot ∨ martial) = (m > 0) ∧ m = n + 1)
  (H3 : ¬ Li shoot ∧ (Li martial ∨ Li row)) :
  Zhang martial :=
by
  sorry

end xiaozhang_participates_in_martial_arts_l311_311550


namespace max_elves_without_caps_proof_max_elves_with_caps_proof_l311_311688

-- Defining the conditions and the problem statement
open Nat

-- We model the problem with the following:
axiom truth_teller : Type
axiom liar_with_caps : Type
axiom dwarf_with_caps : Type
axiom dwarf_without_caps : Type

noncomputable def max_elves_without_caps : ℕ :=
  59

noncomputable def max_elves_with_caps : ℕ :=
  30

-- Part (a): Given the conditions, we show that the maximum number of elves without caps is 59
theorem max_elves_without_caps_proof : max_elves_without_caps = 59 :=
by
  sorry

-- Part (b): Given the conditions, we show that the maximum number of elves with caps is 30
theorem max_elves_with_caps_proof : max_elves_with_caps = 30 :=
by
  sorry

end max_elves_without_caps_proof_max_elves_with_caps_proof_l311_311688


namespace minimize_distance_midpoint_Q5_Q6_l311_311349

theorem minimize_distance_midpoint_Q5_Q6 
  (Q : ℝ → ℝ)
  (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 : ℝ)
  (h1 : Q2 = Q1 + 1)
  (h2 : Q3 = Q2 + 1)
  (h3 : Q4 = Q3 + 1)
  (h4 : Q5 = Q4 + 1)
  (h5 : Q6 = Q5 + 2)
  (h6 : Q7 = Q6 + 2)
  (h7 : Q8 = Q7 + 2)
  (h8 : Q9 = Q8 + 2)
  (h9 : Q10 = Q9 + 2) :
  Q ((Q5 + Q6) / 2) = (Q ((Q1 + Q2) / 2) + Q ((Q3 + Q4) / 2) + Q ((Q7 + Q8) / 2) + Q ((Q9 + Q10) / 2)) :=
sorry

end minimize_distance_midpoint_Q5_Q6_l311_311349


namespace polygon_diagonals_l311_311983

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l311_311983


namespace net_change_in_salary_l311_311727

variable (S : ℝ)

theorem net_change_in_salary : 
  let increased_salary := S + (0.1 * S)
  let final_salary := increased_salary - (0.1 * increased_salary)
  final_salary - S = -0.01 * S :=
by
  sorry

end net_change_in_salary_l311_311727


namespace proposition_3_proposition_4_l311_311527

variable {Line Plane : Type} -- Introduce the types for lines and planes
variable (m n : Line) (α β : Plane) -- Introduce specific lines and planes

-- Define parallel and perpendicular relations
variables {parallel : Line → Plane → Prop} {perpendicular : Line → Plane → Prop}
variables {parallel_line : Line → Line → Prop} {perpendicular_line : Line → Line → Prop}
variables {parallel_plane : Plane → Plane → Prop} {perpendicular_plane : Plane → Plane → Prop}

-- Define subset: a line n is in a plane α
variable {subset : Line → Plane → Prop}

-- Hypotheses for propositions 3 and 4
axiom prop3_hyp1 : perpendicular m α
axiom prop3_hyp2 : parallel_line m n
axiom prop3_hyp3 : parallel_plane α β

axiom prop4_hyp1 : perpendicular_line m n
axiom prop4_hyp2 : perpendicular m α
axiom prop4_hyp3 : perpendicular n β

theorem proposition_3 (h1 : perpendicular m α) (h2 : parallel_line m n) (h3 : parallel_plane α β) : perpendicular n β := sorry

theorem proposition_4 (h1 : perpendicular_line m n) (h2 : perpendicular m α) (h3 : perpendicular n β) : perpendicular_plane α β := sorry

end proposition_3_proposition_4_l311_311527


namespace q_simplification_l311_311699

noncomputable def q (x a b c D : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem q_simplification (a b c D x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q x a b c D = a + b + c + 2 * x + 3 * D / (a + b + c) :=
by
  sorry

end q_simplification_l311_311699


namespace evaluate_expression_l311_311484

theorem evaluate_expression : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := 
by 
  sorry

end evaluate_expression_l311_311484


namespace binary_multiplication_correct_l311_311338

theorem binary_multiplication_correct :
  nat.of_digits 2 [1, 0, 1, 0, 0, 1, 1] = 
  (nat.of_digits 2 [1, 1, 0, 1] * nat.of_digits 2 [1, 1, 1]) :=
by
  sorry

end binary_multiplication_correct_l311_311338


namespace beef_cubes_per_slab_l311_311136

-- Define the conditions as variables
variables (kabob_sticks : ℕ) (cubes_per_stick : ℕ) (cost_per_slab : ℕ) (total_cost : ℕ) (total_kabob_sticks : ℕ)

-- Assume the conditions from step a)
theorem beef_cubes_per_slab 
  (h1 : cubes_per_stick = 4) 
  (h2 : cost_per_slab = 25) 
  (h3 : total_cost = 50) 
  (h4 : total_kabob_sticks = 40)
  : total_cost / cost_per_slab * (total_kabob_sticks * cubes_per_stick) / (total_cost / cost_per_slab) = 80 := 
by {
  -- the proof goes here
  sorry
}

end beef_cubes_per_slab_l311_311136


namespace non_congruent_triangles_with_perimeter_18_l311_311950

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l311_311950


namespace ellipse_parameters_l311_311001

theorem ellipse_parameters 
  (x y : ℝ)
  (h : 2 * x^2 + y^2 + 42 = 8 * x + 36 * y) :
  ∃ (h k : ℝ) (a b : ℝ), 
    (h = 2) ∧ (k = 18) ∧ (a = Real.sqrt 290) ∧ (b = Real.sqrt 145) ∧ 
    ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
sorry

end ellipse_parameters_l311_311001


namespace green_tiles_in_50th_row_l311_311382

-- Conditions
def tiles_in_row (n : ℕ) : ℕ := 2 * n - 1

def green_tiles_in_row (n : ℕ) : ℕ := (tiles_in_row n - 1) / 2

-- Prove the number of green tiles in the 50th row
theorem green_tiles_in_50th_row : green_tiles_in_row 50 = 49 :=
by
  -- Placeholder proof
  sorry

end green_tiles_in_50th_row_l311_311382


namespace last_passenger_probability_l311_311084

noncomputable def probability_last_passenger_seat (n : ℕ) : ℚ :=
if h : n > 0 then 1 / 2 else 0

theorem last_passenger_probability (n : ℕ) (h : n > 0) :
  probability_last_passenger_seat n = 1 / 2 :=
begin
  sorry
end

end last_passenger_probability_l311_311084


namespace problem_statement_l311_311606

noncomputable def expr : ℝ :=
  (1 - Real.sqrt 5)^0 + abs (-Real.sqrt 2) - 2 * Real.cos (Real.pi / 4) + (1 / 4 : ℝ)⁻¹

theorem problem_statement : expr = 5 := by
  sorry

end problem_statement_l311_311606


namespace numberOfWaysToChoose4Cards_l311_311182

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l311_311182


namespace different_suits_card_combinations_l311_311200

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l311_311200


namespace system_of_equations_solution_exists_l311_311545

theorem system_of_equations_solution_exists :
  ∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    (x = 3 ∧ y = 2021 ∧ z = 4 ∨ 
    x = -1 ∧ y = 2019 ∧ z = -2) := 
sorry

end system_of_equations_solution_exists_l311_311545


namespace rohan_food_percentage_l311_311418

noncomputable def rohan_salary : ℝ := 7500
noncomputable def rohan_savings : ℝ := 1500
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def entertainment_percentage : ℝ := 0.10
noncomputable def conveyance_percentage : ℝ := 0.10
noncomputable def total_spent : ℝ := rohan_salary - rohan_savings
noncomputable def known_percentage : ℝ := house_rent_percentage + entertainment_percentage + conveyance_percentage

theorem rohan_food_percentage (F : ℝ) :
  total_spent = rohan_salary * (1 - known_percentage - F) →
  F = 0.20 :=
sorry

end rohan_food_percentage_l311_311418


namespace inequality_always_true_l311_311446

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_always_true_l311_311446


namespace testing_methods_first_problem_testing_methods_second_problem_l311_311359

open_locale big_operators

-- Definition of the problem and its constraints:
def products := range 8
def defective_products := 3

-- Question 1 condition:
def first_defective_on_second_test := true
def last_defective_on_sixth_test := true

-- Question 2 condition:
def at_most_five_tests := true

-- Lean statement:

-- The first proof problem:
theorem testing_methods_first_problem (products : Finset ℕ) (defective_products : ℕ)
  (first_defective_on_second_test : Bool) (last_defective_on_sixth_test : Bool) :
  first_defective_on_second_test = true →
  last_defective_on_sixth_test = true →
  products.card = 8 →
  defective_products = 3 →
  -- The number of distinct testing methods is 1080
  (∑ p in products.powerset, p.card) = 1080 :=
sorry

-- The second proof problem:
theorem testing_methods_second_problem (products : Finset ℕ) (defective_products : ℕ)
  (at_most_five_tests : Bool) :
  at_most_five_tests = true →
  products.card = 8 →
  defective_products = 3 →
  -- The number of distinct testing methods is 936
  (∑ p in products.powerset, p.card) = 936 :=
sorry

end testing_methods_first_problem_testing_methods_second_problem_l311_311359


namespace train_distance_difference_l311_311100

theorem train_distance_difference 
  (speed1 speed2 : ℕ) (distance : ℕ) (meet_time : ℕ)
  (h_speed1 : speed1 = 16)
  (h_speed2 : speed2 = 21)
  (h_distance : distance = 444)
  (h_meet_time : meet_time = distance / (speed1 + speed2)) :
  (speed2 * meet_time) - (speed1 * meet_time) = 60 :=
by
  sorry

end train_distance_difference_l311_311100


namespace odd_factors_252_l311_311365

theorem odd_factors_252 : 
  {n : ℕ | n ∈ finset.filter (λ d, d % 2 = 1) (finset.divisors 252)}.card = 6 := 
sorry

end odd_factors_252_l311_311365


namespace center_of_circle_l311_311908

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

-- Define the condition for the center of the circle
def is_center_of_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = 4

-- The main theorem to be proved
theorem center_of_circle : is_center_of_circle 1 (-1) :=
by
  sorry

end center_of_circle_l311_311908


namespace vinegar_final_percentage_l311_311121

def vinegar_percentage (volume1 volume2 : ℕ) (percent1 percent2 : ℚ) : ℚ :=
  let vinegar1 := volume1 * percent1 / 100
  let vinegar2 := volume2 * percent2 / 100
  (vinegar1 + vinegar2) / (volume1 + volume2) * 100

theorem vinegar_final_percentage:
  vinegar_percentage 128 128 8 13 = 10.5 :=
  sorry

end vinegar_final_percentage_l311_311121


namespace initial_distance_from_lens_l311_311768

def focal_length := 150 -- focal length F in cm
def screen_shift := 40  -- screen moved by 40 cm

theorem initial_distance_from_lens (d : ℝ) (f : ℝ) (s : ℝ) 
  (h_focal_length : f = focal_length) 
  (h_screen_shift : s = screen_shift) 
  (h_parallel_beam : d = f / 2 ∨ d = 3 * f / 2) : 
  d = 130 ∨ d = 170 := 
by 
  sorry

end initial_distance_from_lens_l311_311768


namespace smallest_angle_of_triangle_l311_311421

theorem smallest_angle_of_triangle (x : ℝ) (h : 3 * x + 4 * x + 5 * x = 180) : 3 * x = 45 :=
by
  sorry

end smallest_angle_of_triangle_l311_311421


namespace tetrahedron_distance_sum_eq_l311_311034

-- Defining the necessary conditions
variables {V K : ℝ}
variables {S_1 S_2 S_3 S_4 H_1 H_2 H_3 H_4 : ℝ}

axiom ratio_eq (i : ℕ) (Si : ℝ) (K : ℝ) : (Si / i = K)
axiom volume_eq : S_1 * H_1 + S_2 * H_2 + S_3 * H_3 + S_4 * H_4 = 3 * V

-- Main theorem stating that the desired result holds under the given conditions
theorem tetrahedron_distance_sum_eq :
  H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4 = 3 * V / K :=
by
have h1 : S_1 = K * 1 := by sorry
have h2 : S_2 = K * 2 := by sorry
have h3 : S_3 = K * 3 := by sorry
have h4 : S_4 = K * 4 := by sorry
have sum_eq : K * (H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4) = 3 * V := by sorry
exact sorry

end tetrahedron_distance_sum_eq_l311_311034


namespace max_value_of_function_l311_311586

theorem max_value_of_function : ∀ x : ℝ, (0 < x ∧ x < 1) → x * (1 - x) ≤ 1 / 4 :=
sorry

end max_value_of_function_l311_311586


namespace delivery_parcels_problem_l311_311755

theorem delivery_parcels_problem (x : ℝ) (h1 : 2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28) : 
  2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28 :=
by
  exact h1

end delivery_parcels_problem_l311_311755


namespace find_a_min_value_of_f_l311_311644

theorem find_a (a : ℕ) (h1 : 3 / 2 < 2 + a) (h2 : 1 / 2 ≥ 2 - a) : a = 1 := by
  sorry

theorem min_value_of_f (a x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) : 
    (a = 1) → ∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, |x + a| + |x - 2| ≥ m := by
  sorry

end find_a_min_value_of_f_l311_311644


namespace units_digit_product_l311_311209

theorem units_digit_product (a b : ℕ) (h1 : (a % 10 ≠ 0) ∧ (b % 10 ≠ 0)) : (a * b % 10 = 0) ∨ (a * b % 10 ≠ 0) :=
by
  sorry

end units_digit_product_l311_311209


namespace num_integers_sq_condition_l311_311921

theorem num_integers_sq_condition : 
  {n : ℤ | n < 30 ∧ (∃ k : ℤ, k ^ 2 = n / (30 - n))}.to_finset.card = 3 := 
by
  sorry

end num_integers_sq_condition_l311_311921


namespace binary_101_to_decimal_l311_311473

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l311_311473


namespace ground_beef_total_cost_l311_311647

-- Define the conditions
def price_per_kg : ℝ := 5.00
def quantity_in_kg : ℝ := 12

-- The total cost calculation
def total_cost (price_per_kg quantity_in_kg : ℝ) : ℝ := price_per_kg * quantity_in_kg

-- Theorem statement
theorem ground_beef_total_cost :
  total_cost price_per_kg quantity_in_kg = 60.00 :=
sorry

end ground_beef_total_cost_l311_311647


namespace count_non_congruent_triangles_with_perimeter_18_l311_311961

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l311_311961


namespace triangle_equality_condition_l311_311709

-- Define the triangle and angles
variables {A B C : Point} -- Points in the triangle
variables {alpha beta gamma : ℝ} -- Angles at the vertices of the triangle
variables {BC XY : ℝ} -- Lengths of the sides

-- Conditions
def angle_tan_mul_condition := if (Real.tan beta * Real.tan gamma = 3 ∨ Real.tan beta * Real.tan gamma = -1) then True else False

-- The theorem to be proven
theorem triangle_equality_condition:
  angle_tan_mul_condition →
  BC = XY :=
sorry

end triangle_equality_condition_l311_311709


namespace lattice_points_on_hyperbola_l311_311812

theorem lattice_points_on_hyperbola :
  {p : (ℤ × ℤ) // p.1^2 - p.2^2 = 1800^2}.card = 150 :=
sorry

end lattice_points_on_hyperbola_l311_311812


namespace m_squared_divisible_by_64_l311_311514

theorem m_squared_divisible_by_64 (m : ℕ) (h : 8 ∣ m) : 64 ∣ m * m :=
sorry

end m_squared_divisible_by_64_l311_311514


namespace gravel_cost_correct_l311_311106

-- Definitions from the conditions
def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 60
def road_width : ℕ := 15
def gravel_cost_per_sq_m : ℕ := 3

-- Calculate areas of the roads
def area_road_length : ℕ := lawn_length * road_width
def area_road_breadth : ℕ := (lawn_breadth - road_width) * road_width

-- Total area to be graveled
def total_area : ℕ := area_road_length + area_road_breadth

-- Total cost
def total_cost : ℕ := total_area * gravel_cost_per_sq_m

-- Prove the total cost is 5625 Rs
theorem gravel_cost_correct : total_cost = 5625 := by
  sorry

end gravel_cost_correct_l311_311106


namespace polynomial_at_x_is_minus_80_l311_311738

def polynomial (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def x_value : ℤ := 2

theorem polynomial_at_x_is_minus_80 : polynomial x_value = -80 := 
by
  sorry

end polynomial_at_x_is_minus_80_l311_311738


namespace sin_minus_cos_eq_minus_1_l311_311205

theorem sin_minus_cos_eq_minus_1 (x : ℝ) 
  (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) :
  Real.sin x - Real.cos x = -1 := by
  sorry

end sin_minus_cos_eq_minus_1_l311_311205


namespace square_perimeter_ratio_l311_311245

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 :=
by
  sorry

end square_perimeter_ratio_l311_311245


namespace number_of_positive_area_triangles_l311_311669

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l311_311669


namespace age_ratio_l311_311551

noncomputable def rahul_present_age (future_age : ℕ) (years_passed : ℕ) : ℕ := future_age - years_passed

theorem age_ratio (future_rahul_age : ℕ) (years_passed : ℕ) (deepak_age : ℕ) :
  future_rahul_age = 26 →
  years_passed = 6 →
  deepak_age = 15 →
  rahul_present_age future_rahul_age years_passed / deepak_age = 4 / 3 :=
by
  intros
  have h1 : rahul_present_age 26 6 = 20 := rfl
  sorry

end age_ratio_l311_311551


namespace janice_purchase_l311_311057

theorem janice_purchase (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 30 * a + 200 * b + 300 * c = 3000) : a = 20 :=
sorry

end janice_purchase_l311_311057


namespace primes_divisible_by_3_percentage_is_12_5_l311_311279

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l311_311279


namespace even_m_n_l311_311631

variable {m n : ℕ}

theorem even_m_n
  (h_m : ∃ k : ℕ, m = 2 * k + 1)
  (h_n : ∃ k : ℕ, n = 2 * k + 1) :
  Even ((m - n) ^ 2) ∧ Even ((m - n - 4) ^ 2) ∧ Even (2 * m * n + 4) :=
by
  sorry

end even_m_n_l311_311631


namespace distinct_roots_of_transformed_polynomial_l311_311694

theorem distinct_roots_of_transformed_polynomial
  (a b c : ℝ)
  (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
                    (a * x^5 + b * x^4 + c = 0) ∧ 
                    (a * y^5 + b * y^4 + c = 0) ∧ 
                    (a * z^5 + b * z^4 + c = 0)) :
  ∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
               (c * u^5 + b * u + a = 0) ∧ 
               (c * v^5 + b * v + a = 0) ∧ 
               (c * w^5 + b * w + a = 0) :=
  sorry

end distinct_roots_of_transformed_polynomial_l311_311694


namespace sum_of_consecutive_naturals_l311_311333

theorem sum_of_consecutive_naturals (n : ℕ) : 
  ∃ S : ℕ, S = n * (n + 1) / 2 :=
by
  sorry

end sum_of_consecutive_naturals_l311_311333


namespace compound_interest_l311_311716

theorem compound_interest 
  (P : ℝ) (r : ℝ) (t : ℕ) : P = 500 → r = 0.02 → t = 3 → (P * (1 + r)^t) - P = 30.60 :=
by
  intros P_invest rate years
  simp [P_invest, rate, years]
  sorry

end compound_interest_l311_311716


namespace min_moves_to_reset_counters_l311_311572

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l311_311572


namespace arithmetic_sequence_sum_l311_311431

theorem arithmetic_sequence_sum (x y z d : ℤ)
  (h₀ : d = 10 - 3)
  (h₁ : 10 = 3 + d)
  (h₂ : 17 = 10 + d)
  (h₃ : x = 17 + d)
  (h₄ : y = x + d)
  (h₅ : 31 = y + d)
  (h₆ : z = 31 + d) :
  x + y + z = 93 := by
sorry

end arithmetic_sequence_sum_l311_311431


namespace doubled_team_completes_half_in_three_days_l311_311139

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end doubled_team_completes_half_in_three_days_l311_311139


namespace find_sides_from_diagonals_l311_311977

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l311_311977


namespace circle_line_chord_length_l311_311791

theorem circle_line_chord_length :
  ∀ (k m : ℝ), (∀ x y : ℝ, x^2 + y^2 = 4 → y = k * x + m → ∃ (a : ℝ), a = 2) →
    |m| = Real.sqrt 3 :=
by 
  intros k m h
  sorry

end circle_line_chord_length_l311_311791


namespace unique_solution_condition_l311_311351

theorem unique_solution_condition (a b c : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 :=
sorry

end unique_solution_condition_l311_311351


namespace totalSleepIsThirtyHours_l311_311396

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l311_311396


namespace find_w_squared_l311_311507

theorem find_w_squared (w : ℝ) (h : (2 * w + 19) ^ 2 = (4 * w + 9) * (3 * w + 13)) :
  w ^ 2 = ((6 + Real.sqrt 524) / 4) ^ 2 :=
sorry

end find_w_squared_l311_311507


namespace Mary_books_check_out_l311_311071

theorem Mary_books_check_out
  (initial_books : ℕ)
  (returned_unhelpful_books : ℕ)
  (returned_later_books : ℕ)
  (checked_out_later_books : ℕ)
  (total_books_now : ℕ)
  (h1 : initial_books = 5)
  (h2 : returned_unhelpful_books = 3)
  (h3 : returned_later_books = 2)
  (h4 : checked_out_later_books = 7)
  (h5 : total_books_now = 12) :
  ∃ (x : ℕ), (initial_books - returned_unhelpful_books + x - returned_later_books + checked_out_later_books = total_books_now) ∧ x = 5 :=
by {
  sorry
}

end Mary_books_check_out_l311_311071


namespace last_passenger_sits_in_assigned_seat_l311_311083

theorem last_passenger_sits_in_assigned_seat (n : ℕ) (h : n > 0) :
  let prob := 1 / 2 in
  (∃ (s : set (fin n)), (∀ i ∈ s, i.val < n) ∧ 
   (∀ (ps : fin n), ∃ (t : fin n), t ∈ s ∧ ps ≠ t)) →
  (∃ (prob : ℚ), prob = 1 / 2) :=
by
  sorry

end last_passenger_sits_in_assigned_seat_l311_311083


namespace find_x_l311_311271

theorem find_x (x : ℝ) : x * 2.25 - (5 * 0.85) / 2.5 = 5.5 → x = 3.2 :=
by
  sorry

end find_x_l311_311271


namespace choose_4_cards_of_different_suits_l311_311191

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l311_311191


namespace hyperbola_equation_l311_311357

theorem hyperbola_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∀ {x y : ℝ}, x^2 / 12 + y^2 / 4 = 1 → True) →
  (∀ {x y : ℝ}, x^2 / a^2 - y^2 / b^2 = 1 → True) →
  (∀ {x y : ℝ}, y = Real.sqrt 3 * x → True) →
  (∃ k : ℝ, 4 < k ∧ k < 12 ∧ 2 = 12 - k ∧ 6 = k - 4) →
  a = 2 ∧ b = 6 := by
  intros h_ellipse h_hyperbola h_asymptote h_k
  sorry

end hyperbola_equation_l311_311357


namespace problem1_problem2_l311_311902

theorem problem1 : (1 * (-5) - (-6) + (-7)) = -6 :=
by
  sorry

theorem problem2 : (-1)^2021 + (-18) * abs (-2 / 9) - 4 / (-2) = -3 :=
by
  sorry

end problem1_problem2_l311_311902


namespace find_x_l311_311969

-- Definitions of the conditions
def eq1 (x y z : ℕ) : Prop := x + y + z = 25
def eq2 (y z : ℕ) : Prop := y + z = 14

-- Statement of the mathematically equivalent proof problem
theorem find_x (x y z : ℕ) (h1 : eq1 x y z) (h2 : eq2 y z) : x = 11 :=
by {
  -- This is where the proof would go, but we can omit it for now:
  sorry
}

end find_x_l311_311969


namespace num_integers_sq_condition_l311_311922

theorem num_integers_sq_condition : 
  {n : ℤ | n < 30 ∧ (∃ k : ℤ, k ^ 2 = n / (30 - n))}.to_finset.card = 3 := 
by
  sorry

end num_integers_sq_condition_l311_311922


namespace polygon_sides_from_diagonals_l311_311989

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l311_311989


namespace necessary_and_sufficient_condition_l311_311246

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x - 4 * a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
sorry

end necessary_and_sufficient_condition_l311_311246


namespace count_non_congruent_triangles_with_perimeter_18_l311_311960

-- Definitions:
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c

def perimeter_is_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

-- Theorem statement
theorem count_non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // is_triangle t.1 t.2 t.3 ∧ is_non_congruent_triangle t.1 t.2 t.3 ∧ perimeter_is_18 t.1 t.2 t.3 }.to_finset.card = 7 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_18_l311_311960


namespace polygon_sides_from_diagonals_l311_311993

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l311_311993


namespace cost_50_jasmines_discounted_l311_311464

variable (cost_per_8_jasmines : ℝ) (num_jasmines : ℕ) (discount : ℝ)
variable (proportional : Prop) (c_50_jasmines : ℝ)

-- Given the cost of a bouquet with 8 jasmines
def cost_of_8_jasmines : ℝ := 24

-- Given the price is directly proportional to the number of jasmines
def price_proportional := ∀ (n : ℕ), num_jasmines = 8 → proportional

-- Given the bouquet with 50 jasmines
def num_jasmines_50 : ℕ := 50

-- Applying a 10% discount
def ten_percent_discount : ℝ := 0.9

-- Prove the cost of the bouquet with 50 jasmines after a 10% discount
theorem cost_50_jasmines_discounted :
  proportional ∧ (c_50_jasmines = (cost_of_8_jasmines / 8) * num_jasmines_50) →
  (c_50_jasmines * ten_percent_discount) = 135 :=
by
  sorry

end cost_50_jasmines_discounted_l311_311464


namespace possible_values_of_inverse_sum_l311_311526

open Set

theorem possible_values_of_inverse_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
  ∃ s : Set ℝ, s = { x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b) } ∧ 
  s = Ici 2 :=
sorry

end possible_values_of_inverse_sum_l311_311526


namespace rectangle_area_l311_311764

variables (y w : ℝ)

-- Definitions from conditions
def is_width_of_rectangle : Prop := w = y / Real.sqrt 10
def is_length_of_rectangle : Prop := 3 * w = y / Real.sqrt 10

-- Theorem to be proved
theorem rectangle_area (h1 : is_width_of_rectangle y w) (h2 : is_length_of_rectangle y w) : 
  3 * (w^2) = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l311_311764


namespace total_flowers_l311_311542

theorem total_flowers (initial_rosas_flowers andre_gifted_flowers : ℝ) 
  (h1 : initial_rosas_flowers = 67.0) 
  (h2 : andre_gifted_flowers = 90.0) : 
  initial_rosas_flowers + andre_gifted_flowers = 157.0 :=
  by
  sorry

end total_flowers_l311_311542


namespace non_congruent_triangles_with_perimeter_18_l311_311939

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l311_311939


namespace same_solution_implies_value_of_m_l311_311381

theorem same_solution_implies_value_of_m (x m : ℤ) (h₁ : -5 * x - 6 = 3 * x + 10) (h₂ : -2 * m - 3 * x = 10) : m = -2 :=
by
  sorry

end same_solution_implies_value_of_m_l311_311381


namespace cloth_sales_worth_l311_311582

theorem cloth_sales_worth 
  (commission : ℝ) 
  (commission_rate : ℝ) 
  (commission_received : ℝ) 
  (commission_rate_of_sales : commission_rate = 2.5)
  (commission_received_rs : commission_received = 21) 
  : (commission_received / (commission_rate / 100)) = 840 :=
by
  sorry

end cloth_sales_worth_l311_311582


namespace laura_total_miles_per_week_l311_311523

def round_trip_school : ℕ := 20
def round_trip_supermarket : ℕ := 40
def round_trip_gym : ℕ := 10
def round_trip_friends_house : ℕ := 24

def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2
def gym_trips_per_week : ℕ := 3
def friends_house_trips_per_week : ℕ := 1

def total_miles_driven_per_week :=
  round_trip_school * school_trips_per_week +
  round_trip_supermarket * supermarket_trips_per_week +
  round_trip_gym * gym_trips_per_week +
  round_trip_friends_house * friends_house_trips_per_week

theorem laura_total_miles_per_week : total_miles_driven_per_week = 234 :=
by
  sorry

end laura_total_miles_per_week_l311_311523


namespace negation_of_p_l311_311504
open Classical

variable (n : ℕ)

def p : Prop := ∀ n : ℕ, n^2 < 2^n

theorem negation_of_p : ¬ p ↔ ∃ n₀ : ℕ, n₀^2 ≥ 2^n₀ := 
by
  sorry

end negation_of_p_l311_311504


namespace tens_digit_of_3_pow_2010_l311_311740

theorem tens_digit_of_3_pow_2010 : (3^2010 / 10) % 10 = 4 := by
  sorry

end tens_digit_of_3_pow_2010_l311_311740


namespace expand_and_simplify_l311_311023

theorem expand_and_simplify (x : ℝ) :
  2 * (x + 3) * (x^2 + 2 * x + 7) = 2 * x^3 + 10 * x^2 + 26 * x + 42 :=
by
  sorry

end expand_and_simplify_l311_311023


namespace log_problem_l311_311926

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_problem :
  let x := (log_base 8 2) ^ (log_base 2 8)
  log_base 3 x = -3 :=
by
  sorry

end log_problem_l311_311926


namespace range_of_m_l311_311683

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → -3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end range_of_m_l311_311683


namespace subtract_from_sum_base8_l311_311029

def add_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) + (b % 8)) % 8
  + (((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) % 8) * 8
  + (((((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) / 8) + ((a / 64) % 8 + (b / 64) % 8)) % 8) * 64

def subtract_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) - (b % 8) + 8) % 8
  + (((a / 8) % 8 - (b / 8) % 8 - if (a % 8) < (b % 8) then 1 else 0 + 8) % 8) * 8
  + (((a / 64) - (b / 64) - if (a / 8) % 8 < (b / 8) % 8 then 1 else 0) % 8) * 64

theorem subtract_from_sum_base8 :
  subtract_in_base_8 (add_in_base_8 652 147) 53 = 50 := by
  sorry

end subtract_from_sum_base8_l311_311029


namespace abs_of_neg_one_third_l311_311088

theorem abs_of_neg_one_third : abs (- (1 / 3)) = (1 / 3) := by
  sorry

end abs_of_neg_one_third_l311_311088


namespace ways_to_choose_4_cards_of_different_suits_l311_311187

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l311_311187


namespace sides_of_regular_polygon_with_20_diagonals_l311_311996

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l311_311996


namespace num_odd_factors_of_252_l311_311366

theorem num_odd_factors_of_252 : 
  ∃ n : ℕ, n = 252 ∧ 
  ∃ k : ℕ, (k = ∏ d in (divisors_filter (λ x, x % 2 = 1) n), 1) 
  ∧ k = 6 := 
sorry

end num_odd_factors_of_252_l311_311366


namespace number_of_ways_at_least_one_different_l311_311007

open Finset Nat

theorem number_of_ways_at_least_one_different :
  (∑ x in (finset.range 4).powersetLen 2, ∑ y in (finset.range (4 - x.card)).powersetLen 2, 1) +
  (∑ x in (finset.range 4).powersetLen 1, ∑ y in (finset.range (3 - x.card)).powersetLen 1, ∑ z in (finset.range (2 - y.card)).powersetLen 1, 1) = 30 :=
by
  sorry

end number_of_ways_at_least_one_different_l311_311007


namespace total_sleep_correct_l311_311397

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l311_311397


namespace sides_of_regular_polygon_with_20_diagonals_l311_311998

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l311_311998


namespace expand_product_l311_311622

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  sorry

end expand_product_l311_311622


namespace route_y_saves_time_l311_311705

theorem route_y_saves_time (distance_X speed_X : ℕ)
                           (distance_Y_WOCZ distance_Y_CZ speed_Y speed_Y_CZ : ℕ)
                           (time_saved_in_minutes : ℚ) :
  distance_X = 8 → 
  speed_X = 40 → 
  distance_Y_WOCZ = 6 → 
  distance_Y_CZ = 1 → 
  speed_Y = 50 → 
  speed_Y_CZ = 25 → 
  time_saved_in_minutes = 2.4 →
  (distance_X / speed_X : ℚ) * 60 - 
  ((distance_Y_WOCZ / speed_Y + distance_Y_CZ / speed_Y_CZ) * 60) = time_saved_in_minutes :=
by
  intros
  sorry

end route_y_saves_time_l311_311705


namespace proportional_segments_l311_311235

-- Define the tetrahedron and points
structure Tetrahedron :=
(A B C D O A1 B1 C1 : ℝ)

-- Define the conditions of the problem
variables {tetra : Tetrahedron}

-- Define the segments and their relationships
axiom segments_parallel (DA : ℝ) (DB : ℝ) (DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1

-- The theorem to prove, which follows directly from the given axiom 
theorem proportional_segments (DA DB DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1 :=
segments_parallel DA DB DC OA1 OB1 OC1

end proportional_segments_l311_311235


namespace eggs_in_second_tree_l311_311696

theorem eggs_in_second_tree
  (nests_in_first_tree : ℕ)
  (eggs_per_nest : ℕ)
  (eggs_in_front_yard : ℕ)
  (total_eggs : ℕ)
  (eggs_in_second_tree : ℕ)
  (h1 : nests_in_first_tree = 2)
  (h2 : eggs_per_nest = 5)
  (h3 : eggs_in_front_yard = 4)
  (h4 : total_eggs = 17)
  (h5 : nests_in_first_tree * eggs_per_nest + eggs_in_front_yard + eggs_in_second_tree = total_eggs) :
  eggs_in_second_tree = 3 :=
sorry

end eggs_in_second_tree_l311_311696


namespace has_root_in_interval_l311_311561

def f (x : ℝ) := x^3 - 3*x - 3

theorem has_root_in_interval : ∃ c ∈ (Set.Ioo (2:ℝ) 3), f c = 0 :=
by 
    sorry

end has_root_in_interval_l311_311561


namespace odd_factors_360_l311_311654

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l311_311654


namespace human_height_weight_correlated_l311_311579

-- Define the relationships as types
def taxiFareDistanceRelated : Prop := ∀ x y : ℕ, x = y → True
def houseSizePriceRelated : Prop := ∀ x y : ℕ, x = y → True
def humanHeightWeightCorrelated : Prop := ∃ k : ℕ, ∀ x y : ℕ, x / k = y
def ironBlockMassRelated : Prop := ∀ x y : ℕ, x = y → True

-- Main theorem statement
theorem human_height_weight_correlated : humanHeightWeightCorrelated :=
  sorry

end human_height_weight_correlated_l311_311579


namespace binary_101_eq_5_l311_311475

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l311_311475


namespace prove_equivalence_l311_311627

variable (x : ℝ)

def operation1 (x : ℝ) : ℝ := 8 - x

def operation2 (x : ℝ) : ℝ := x - 8

theorem prove_equivalence : operation2 (operation1 14) = -14 := by
  sorry

end prove_equivalence_l311_311627


namespace find_C_l311_311729

theorem find_C (A B C : ℕ) (h1 : (19 + A + B) % 3 = 0) (h2 : (15 + A + B + C) % 3 = 0) : C = 1 := by
  sorry

end find_C_l311_311729


namespace circus_juggling_l311_311091

theorem circus_juggling (jugglers : ℕ) (balls_per_juggler : ℕ) (total_balls : ℕ)
  (h1 : jugglers = 5000)
  (h2 : balls_per_juggler = 12)
  (h3 : total_balls = jugglers * balls_per_juggler) :
  total_balls = 60000 :=
by
  rw [h1, h2] at h3
  exact h3

end circus_juggling_l311_311091


namespace parabola_expression_l311_311798

open Real

-- Given the conditions of the parabola obtaining points A and B
def parabola (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x - 5

-- Defining the points A and B where parabola intersects the x-axis
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (5, 0)

-- The proof statement we need to show
theorem parabola_expression (a b : ℝ) (hxA : parabola a b A.fst = A.snd) (hxB : parabola a b B.fst = B.snd) : 
  ∀ x : ℝ, parabola a b x = x^2 - 4 * x - 5 :=
sorry

end parabola_expression_l311_311798


namespace ways_to_choose_4_cards_of_different_suits_l311_311186

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l311_311186


namespace find_side_b_in_triangle_l311_311210

theorem find_side_b_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180) 
  (h3 : a + c = 8) 
  (h4 : a * c = 15) 
  (h5 : (b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos (B * Real.pi / 180))) : 
  b = Real.sqrt 19 := 
  by sorry

end find_side_b_in_triangle_l311_311210


namespace cost_per_metre_of_carpet_l311_311424

theorem cost_per_metre_of_carpet :
  (length_of_room = 18) →
  (breadth_of_room = 7.5) →
  (carpet_width = 0.75) →
  (total_cost = 810) →
  (cost_per_metre = 4.5) :=
by
  intros length_of_room breadth_of_room carpet_width total_cost
  sorry

end cost_per_metre_of_carpet_l311_311424


namespace intersection_with_complement_N_l311_311039

open Set Real

def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def N : Set ℝ := {x | 0 < x ∧ x < 2}
def complement_N : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

theorem intersection_with_complement_N : M ∩ complement_N = Ico 2 3 :=
by {
  sorry
}

end intersection_with_complement_N_l311_311039


namespace f_at_five_l311_311376

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 3 * n + 17

theorem f_at_five : f 5 = 207 := 
by 
sorry

end f_at_five_l311_311376


namespace actual_average_height_calculation_l311_311876

noncomputable def actual_average_height (incorrect_avg_height : ℚ) (number_of_boys : ℕ) (incorrect_recorded_height : ℚ) (actual_height : ℚ) : ℚ :=
  let incorrect_total_height := incorrect_avg_height * number_of_boys
  let overestimated_height := incorrect_recorded_height - actual_height
  let correct_total_height := incorrect_total_height - overestimated_height
  correct_total_height / number_of_boys

theorem actual_average_height_calculation :
  actual_average_height 182 35 166 106 = 180.29 :=
by
  -- The detailed proof is omitted here.
  sorry

end actual_average_height_calculation_l311_311876


namespace inequality_proof_l311_311168

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : a + b + c > 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_proof_l311_311168


namespace non_congruent_triangles_count_l311_311965

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l311_311965


namespace binary_101_eq_5_l311_311476

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l311_311476


namespace sample_size_is_50_l311_311687

theorem sample_size_is_50 (n : ℕ) :
  (n > 0) → 
  (10 / n = 2 / (2 + 3 + 5)) → 
  n = 50 := 
by
  sorry

end sample_size_is_50_l311_311687


namespace time_taken_by_abc_l311_311580

-- Define the work rates for a, b, and c
def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 1 / 41.25

-- Define the combined work rate for a, b, and c
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

-- Define the reciprocal of the combined work rate, which is the time taken
def time_taken : ℚ := 1 / combined_work_rate

-- Prove that the time taken by a, b, and c together is 11 days
theorem time_taken_by_abc : time_taken = 11 := by
  -- Substitute the values to compute the result
  sorry

end time_taken_by_abc_l311_311580


namespace number_of_white_balls_l311_311453

theorem number_of_white_balls (r w : ℕ) (h_r : r = 8) (h_prob : (r : ℚ) / (r + w) = 2 / 5) : w = 12 :=
by sorry

end number_of_white_balls_l311_311453


namespace find_sides_from_diagonals_l311_311976

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l311_311976


namespace point_on_circle_l311_311158

theorem point_on_circle (t : ℝ) : 
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  x^2 + y^2 = 1 :=
by
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  sorry

end point_on_circle_l311_311158


namespace unique_real_solution_l311_311020

theorem unique_real_solution (x y z : ℝ) :
  (x^3 - 3 * x = 4 - y) ∧ 
  (2 * y^3 - 6 * y = 6 - z) ∧ 
  (3 * z^3 - 9 * z = 8 - x) ↔ 
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end unique_real_solution_l311_311020


namespace exists_good_placement_l311_311471

-- Define a function that checks if a placement is "good" with respect to a symmetry axis
def is_good (f : Fin 1983 → ℕ) : Prop :=
  ∀ (i : Fin 1983), f i < f (i + 991) ∨ f (i + 991) < f i

-- Prove the existence of a "good" placement for the regular 1983-gon
theorem exists_good_placement : ∃ f : Fin 1983 → ℕ, is_good f :=
sorry

end exists_good_placement_l311_311471


namespace min_val_l311_311511

theorem min_val (x y : ℝ) (h : x + 2 * y = 1) : 2^x + 4^y = 2 * Real.sqrt 2 :=
sorry

end min_val_l311_311511


namespace find_k_value_l311_311162

noncomputable def quadratic_root (k : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + k * x - 3 = 0 ∧ x = 1

theorem find_k_value (k : ℝ) (h : quadratic_root k) : k = 2 :=
by
  cases h with x hx,
  cases hx with hx1 hx2,
  rw hx2 at hx1,
  norm_num at hx1,
  exact hx1
  sorry

end find_k_value_l311_311162


namespace proposition_holds_for_all_positive_odd_numbers_l311_311130

theorem proposition_holds_for_all_positive_odd_numbers
  (P : ℕ → Prop)
  (h1 : P 1)
  (h2 : ∀ k, k ≥ 1 → P k → P (k + 2)) :
  ∀ n, n % 2 = 1 → n ≥ 1 → P n :=
by
  sorry

end proposition_holds_for_all_positive_odd_numbers_l311_311130


namespace arrange_abc_l311_311354

theorem arrange_abc : 
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4 / 5)
  let c := Real.sin 1
  a < c ∧ c < b := 
by
  sorry

end arrange_abc_l311_311354


namespace inequality_proof_l311_311415

theorem inequality_proof (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) : 
  2 * Real.sin α + Real.tan α > 3 * α := 
by
  sorry

end inequality_proof_l311_311415


namespace history_books_count_l311_311095

-- Definitions based on conditions
def total_books : Nat := 100
def geography_books : Nat := 25
def math_books : Nat := 43

-- Problem statement: proving the number of history books
theorem history_books_count : total_books - geography_books - math_books = 32 := by
  sorry

end history_books_count_l311_311095


namespace sum_of_squares_of_roots_eq_213_l311_311031

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end sum_of_squares_of_roots_eq_213_l311_311031


namespace choir_row_lengths_l311_311758

theorem choir_row_lengths (x : ℕ) : 
  ((x ∈ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ∧ (90 % x = 0)) → (x = 5 ∨ x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15) :=
by
  intro h
  cases h
  sorry

end choir_row_lengths_l311_311758


namespace kim_average_round_correct_answers_l311_311049

theorem kim_average_round_correct_answers (x : ℕ) :
  (6 * 2) + (x * 3) + (4 * 5) = 38 → x = 2 :=
by
  intros h
  sorry

end kim_average_round_correct_answers_l311_311049


namespace value_of_f_10_l311_311966

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem value_of_f_10 : f 10 = 107 := by
  sorry

end value_of_f_10_l311_311966


namespace total_value_of_pile_l311_311493

def value_of_pile (total_coins dimes : ℕ) (value_dime value_nickel : ℝ) : ℝ :=
  let nickels := total_coins - dimes
  let value_dimes := dimes * value_dime
  let value_nickels := nickels * value_nickel
  value_dimes + value_nickels

theorem total_value_of_pile :
  value_of_pile 50 14 0.10 0.05 = 3.20 := by
  sorry

end total_value_of_pile_l311_311493


namespace find_a2_b2_geom_sequences_unique_c_l311_311806

-- Define the sequences as per the problem statement
def seqs (a b : ℕ → ℝ) :=
  a 1 = 0 ∧ b 1 = 2013 ∧
  ∀ n : ℕ, (1 ≤ n → (2 * a (n+1) = a n + b n)) ∧ (1 ≤ n → (4 * b (n+1) = a n + 3 * b n))

-- (1) Find values of a_2 and b_2
theorem find_a2_b2 {a b : ℕ → ℝ} (h : seqs a b) :
  a 2 = 1006.5 ∧ b 2 = 1509.75 :=
sorry

-- (2) Prove that {a_n - b_n} and {a_n + 2b_n} are geometric sequences
theorem geom_sequences {a b : ℕ → ℝ} (h : seqs a b) :
  ∃ r s : ℝ, (∃ c : ℝ, ∀ n : ℕ, a n - b n = c * r^n) ∧
             (∃ d : ℝ, ∀ n : ℕ, a n + 2 * b n = d * s^n) :=
sorry

-- (3) Prove there is a unique positive integer c such that a_n < c < b_n always holds
theorem unique_c {a b : ℕ → ℝ} (h : seqs a b) :
  ∃! c : ℝ, (0 < c) ∧ (∀ n : ℕ, 1 ≤ n → a n < c ∧ c < b n) :=
sorry

end find_a2_b2_geom_sequences_unique_c_l311_311806


namespace expression_as_polynomial_l311_311746

theorem expression_as_polynomial (x : ℝ) :
  (3 * x^3 + 2 * x^2 + 5 * x + 9) * (x - 2) -
  (x - 2) * (2 * x^3 + 5 * x^2 - 74) +
  (4 * x - 17) * (x - 2) * (x + 4) = 
  x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 30 :=
sorry

end expression_as_polynomial_l311_311746


namespace number_of_valid_triangles_l311_311667

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l311_311667


namespace ticket_1000_wins_probability_l311_311756

-- Define the total number of tickets
def n_tickets := 1000

-- Define the number of odd tickets
def n_odd_tickets := 500

-- Define the number of relevant tickets (ticket 1000 + odd tickets)
def n_relevant_tickets := 501

-- Define the probability that ticket number 1000 wins a prize
def win_probability : ℚ := 1 / n_relevant_tickets

-- State the theorem
theorem ticket_1000_wins_probability : win_probability = 1 / 501 :=
by
  -- The proof would go here
  sorry

end ticket_1000_wins_probability_l311_311756


namespace fraction_product_simplification_l311_311317

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_simplification_l311_311317


namespace numberOfWaysToChoose4Cards_l311_311179

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l311_311179


namespace application_methods_count_l311_311135

theorem application_methods_count (total_universities: ℕ) (universities_with_coinciding_exams: ℕ) (chosen_universities: ℕ) 
  (remaining_universities: ℕ) (remaining_combinations: ℕ) : 
  total_universities = 6 → universities_with_coinciding_exams = 2 → chosen_universities = 3 → 
  remaining_universities = 4 → remaining_combinations = 16 := 
by
  intros
  sorry

end application_methods_count_l311_311135


namespace regular_polygon_sides_l311_311972

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l311_311972


namespace total_profit_is_92000_l311_311108

def investment_a : ℚ := 24000
def investment_b : ℚ := 32000
def investment_c : ℚ := 36000
def profit_c : ℚ := 36000

theorem total_profit_is_92000 (total_profit : ℚ) : 
  ((investment_c / (investment_a + investment_b + investment_c)) * total_profit = profit_c) → 
  total_profit = 92000 :=
by
  sorry

end total_profit_is_92000_l311_311108


namespace find_9a_value_l311_311725

theorem find_9a_value (a : ℚ) 
  (h : (4 - a) / (5 - a) = (4 / 5) ^ 2) : 9 * a = 20 :=
by
  sorry

end find_9a_value_l311_311725


namespace polygon_sides_from_diagonals_l311_311988

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l311_311988


namespace part_I_part_II_l311_311361

noncomputable def f (x m : ℝ) : ℝ := |3 * x + m|
noncomputable def g (x m : ℝ) : ℝ := f x m - 2 * |x - 1|

theorem part_I (m : ℝ) : (∀ x : ℝ, (f x m - m ≤ 9) ↔ (-1 ≤ x ∧ x ≤ 3)) → m = -3 :=
by
  sorry

theorem part_II (m : ℝ) (h : m > 0) : (∃ A B C : ℝ × ℝ, 
  let A := (-m-2, 0)
  let B := ((2-m)/5, 0)
  let C := (-m/3, -2*m/3-2)
  let Area : ℝ := 1/2 * |(B.1 - A.1) * (C.2 - 0) - (B.2 - A.2) * (C.1 - A.1)|
  Area > 60 ) → m > 12 :=
by
  sorry

end part_I_part_II_l311_311361


namespace total_people_count_l311_311259

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l311_311259


namespace quiz_answer_key_count_l311_311111

theorem quiz_answer_key_count :
  let tf_combinations := 6 -- Combinations of true-false questions
  let mc_combinations := 4 ^ 3 -- Combinations of multiple-choice questions
  tf_combinations * mc_combinations = 384 := by
  -- The values and conditions are directly taken from the problem statement.
  let tf_combinations := 6
  let mc_combinations := 4 ^ 3
  sorry

end quiz_answer_key_count_l311_311111


namespace frog_return_prob_A_after_2022_l311_311897

def initial_prob_A : ℚ := 1
def transition_prob_A_to_adj : ℚ := 1/3
def transition_prob_adj_to_A : ℚ := 1/3
def transition_prob_adj_to_adj : ℚ := 2/3

noncomputable def prob_A_return (n : ℕ) : ℚ :=
if (n % 2 = 0) then
  (2/9) * (1/2^(n/2)) + (1/9)
else
  0

theorem frog_return_prob_A_after_2022 : prob_A_return 2022 = (2/9) * (1/2^1010) + (1/9) :=
by
  sorry

end frog_return_prob_A_after_2022_l311_311897


namespace math_problem_l311_311160

open Real

theorem math_problem (x : ℝ) (p q : ℕ)
  (h1 : (1 + sin x) * (1 + cos x) = 9 / 4)
  (h2 : (1 - sin x) * (1 - cos x) = p - sqrt q)
  (hp_pos : p > 0) (hq_pos : q > 0) : p + q = 1 := sorry

end math_problem_l311_311160


namespace tyler_total_puppies_l311_311441

/-- 
  Tyler has 15 dogs, and each dog has 5 puppies.
  We want to prove that the total number of puppies is 75.
-/
def tyler_dogs : Nat := 15
def puppies_per_dog : Nat := 5
def total_puppies_tyler_has : Nat := tyler_dogs * puppies_per_dog

theorem tyler_total_puppies : total_puppies_tyler_has = 75 := by
  sorry

end tyler_total_puppies_l311_311441


namespace physical_education_class_min_size_l311_311824

theorem physical_education_class_min_size :
  ∃ (x : Nat), 3 * x + 2 * (x + 1) > 50 ∧ 5 * x + 2 = 52 := by
  sorry

end physical_education_class_min_size_l311_311824


namespace value_of_a_l311_311936

theorem value_of_a {a : ℝ} (A : Set ℝ) (B : Set ℝ) (hA : A = {-1, 0, 2}) (hB : B = {2^a}) (hSub : B ⊆ A) : a = 1 := 
sorry

end value_of_a_l311_311936


namespace reciprocal_square_inequality_l311_311495

variable (x y : ℝ)
variable (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≤ y)

theorem reciprocal_square_inequality :
  (1 / y^2) ≤ (1 / x^2) :=
sorry

end reciprocal_square_inequality_l311_311495


namespace number_of_student_clubs_l311_311252

theorem number_of_student_clubs : 
  let n := 2019 in
  let advisory_board_members := 12 in
  let club_members := 27 in
  let total_clubs_with_27_members := Nat.choose (club_members + advisory_board_members - 1) (advisory_board_members - 1) in
  total_clubs_with_27_members = Nat.choose 2003 11 :=
by 
  sorry

end number_of_student_clubs_l311_311252


namespace schools_in_competition_l311_311849

theorem schools_in_competition (x : ℕ) (h : (1/2) * x * (x - 1) = 28) : x = 8 := by
  sorry

end schools_in_competition_l311_311849


namespace probability_divisible_by_three_dice_roll_l311_311872

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def fair_dice_distribution (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 6

noncomputable def probability_divisible_by_3 : ℝ :=
  #[(3, 3), (3, 6), (6, 3), (6, 6)] / (6 * 6 : ℝ)

theorem probability_divisible_by_three_dice_roll : 
  probability_divisible_by_3 = (1 / 9 : ℝ) :=
  sorry

end probability_divisible_by_three_dice_roll_l311_311872


namespace angle_mtb_l311_311127

open Real EuclideanSpace

noncomputable def calculate_mtb_angle (A B C M N O K T : ℝ × ℝ) : Prop :=
  let ⟨ax, ay⟩ := A
  let ⟨bx, by⟩ := B
  let ⟨cx, cy⟩ := C
  let ⟨mx, my⟩ := M
  let ⟨nx, ny⟩ := N
  let ⟨ox, oy⟩ := O
  let ⟨kx, ky⟩ := K
  let ⟨tx, ty⟩ := T in
  (cy = 0) ∧
  (by = 0) ∧
  (by = bx) ∧ -- coordinate transformation for simplicity
  (bx * 2 = ax) ∧
  (ny = 0) ∧
  (nx / bx = 2/3) ∧
  (my = ax) ∧
  (mx = nx) ∧
  (ox = (nx + mx) / 2) ∧
  (oy = (cy + my) / 2) ∧
  (K = (ox + nx, oy + ny)) ∧
  (by / (ox - kx) = (oy - ty) / (ox - tx)) ∧
  -- Need to show angle MTB = 90 degrees
  (angle (M - T) (B - T) = π / 2)

theorem angle_mtb {A B C M N O K T : ℝ × ℝ} :
  calculate_mtb_angle A B C M N O K T → angle (M - T) (B - T) = π / 2 :=
by
  intro h
  sorry

end angle_mtb_l311_311127


namespace total_time_to_complete_project_l311_311009

-- Define the initial conditions
def initial_people : ℕ := 6
def initial_days : ℕ := 35
def fraction_completed : ℚ := 1 / 3

-- Define the additional conditions after more people joined
def additional_people : ℕ := initial_people
def total_people : ℕ := initial_people + additional_people
def remaining_fraction : ℚ := 1 - fraction_completed

-- Total time taken to complete the project
theorem total_time_to_complete_project (initial_people initial_days additional_people : ℕ) (fraction_completed remaining_fraction : ℚ)
  (h1 : initial_people * initial_days * fraction_completed = 1/3) 
  (h2 : additional_people = initial_people) 
  (h3 : total_people = initial_people + additional_people)
  (h4 : remaining_fraction = 1 - fraction_completed) : 
  (initial_days + (remaining_fraction / (total_people * (fraction_completed / (initial_people * initial_days)))) = 70) :=
sorry

end total_time_to_complete_project_l311_311009


namespace parabola_equation_1_parabola_equation_2_l311_311298

noncomputable def parabola_vertex_focus (vertex focus : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, (focus.1 = p / 2 ∧ focus.2 = 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 24 * x)

noncomputable def standard_parabola_through_point (point : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, ( ( point.1^2 = 2 * p * point.2 ∧ point.2 ≠ 0 ∧ point.1 ≠ 0) ∧ (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = y / 2) ) ∨
           ( ( point.2^2 = 2 * p * point.1 ∧ point.1 ≠ 0 ∧ point.2 ≠ 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) )

theorem parabola_equation_1 : parabola_vertex_focus (0, 0) (6, 0) := 
  sorry

theorem parabola_equation_2 : standard_parabola_through_point (1, 2) := 
  sorry

end parabola_equation_1_parabola_equation_2_l311_311298


namespace find_range_l311_311827

noncomputable theory

variable (A B C a b c : ℝ)

theorem find_range
  (hacutriangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (htriangle : A + B + C = π)
  (ha2 : a = 2)
  (htanA : tan A = (cos A + cos C) / (sin A + sin C)) :
  (4 * real.sqrt 3) / 3 < (b + c) / (sin B + sin C) ∧ (b + c) / (sin B + sin C) < 4 :=
sorry

end find_range_l311_311827


namespace min_moves_to_zero_l311_311567

-- Define the problem setting and conditions

def initial_counters : ℕ := 28
def max_value : ℕ := 2017

-- Definition for the minimum number of moves required to reduce all counters to zero

theorem min_moves_to_zero : 
  ∀ (counters : list ℕ), (∀ c ∈ counters, 1 ≤ c ∧ c ≤ max_value) → counters.length = initial_counters →
  ∃ (m : ℕ), m = 11 ∧ 
    (∀ (f : ℕ → ℕ → ℕ), f 0 0 = 0 → (∃ i, 0 < i ∧ i ≤ m ∧ ∀ n ∈ counters, f i n = 0)) :=
by
  sorry

end min_moves_to_zero_l311_311567


namespace polygon_sides_from_diagonals_l311_311991

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l311_311991


namespace cristine_lemons_left_l311_311150

theorem cristine_lemons_left (initial_lemons : ℕ) (given_fraction : ℚ) (exchanged_lemons : ℕ) (h1 : initial_lemons = 12) (h2 : given_fraction = 1/4) (h3 : exchanged_lemons = 2) : 
  initial_lemons - initial_lemons * given_fraction - exchanged_lemons = 7 :=
by 
  sorry

end cristine_lemons_left_l311_311150


namespace total_ticket_cost_l311_311098

theorem total_ticket_cost 
  (young_discount : ℝ := 0.55) 
  (old_discount : ℝ := 0.30) 
  (full_price : ℝ := 10)
  (num_young : ℕ := 2) 
  (num_middle : ℕ := 2) 
  (num_old : ℕ := 2) 
  (grandma_ticket_cost : ℝ := 7) :
  2 * (full_price * young_discount) + 2 * full_price + 2 * grandma_ticket_cost = 43 :=
by 
  sorry

end total_ticket_cost_l311_311098


namespace value_of_a_plus_b_l311_311509

theorem value_of_a_plus_b (a b : ℝ) (h1 : a + 2 * b = 8) (h2 : 3 * a + 4 * b = 18) : a + b = 5 := 
by 
  sorry

end value_of_a_plus_b_l311_311509


namespace observed_wheels_l311_311212

theorem observed_wheels (num_cars wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) : num_cars * wheels_per_car = 48 := by
  sorry

end observed_wheels_l311_311212


namespace andre_max_points_visited_l311_311010
noncomputable def largest_points_to_visit_in_alphabetical_order : ℕ :=
  10

theorem andre_max_points_visited : largest_points_to_visit_in_alphabetical_order = 10 := 
by
  sorry

end andre_max_points_visited_l311_311010


namespace find_x_values_l311_311033

theorem find_x_values (x : ℝ) : 
  ((x + 1)^2 = 36 ∨ (x + 10)^3 = -27) ↔ (x = 5 ∨ x = -7 ∨ x = -13) :=
by
  sorry

end find_x_values_l311_311033


namespace percentage_saved_l311_311690

noncomputable def calculateSavedPercentage : ℚ :=
  let first_tier_free_tickets := 1
  let second_tier_free_tickets_per_ticket := 2
  let number_of_tickets_purchased := 10
  let total_free_tickets :=
    first_tier_free_tickets +
    (number_of_tickets_purchased - 5) * second_tier_free_tickets_per_ticket
  let total_tickets_received := number_of_tickets_purchased + total_free_tickets
  let free_tickets := total_tickets_received - number_of_tickets_purchased
  (free_tickets / total_tickets_received) * 100

theorem percentage_saved : calculateSavedPercentage = 52.38 :=
by
  sorry

end percentage_saved_l311_311690


namespace Maxim_is_correct_l311_311292

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end Maxim_is_correct_l311_311292


namespace hyperbola_eccentricity_sqrt3_l311_311496

theorem hyperbola_eccentricity_sqrt3
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : b / a = Real.sqrt 2) :
  (let e := Real.sqrt (1 + (b^2 / a^2)) in e = Real.sqrt 3) :=
by
  sorry

end hyperbola_eccentricity_sqrt3_l311_311496


namespace choose_4_cards_of_different_suits_l311_311192

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l311_311192


namespace area_evaluation_l311_311149

noncomputable def radius : ℝ := 6
noncomputable def central_angle : ℝ := 90
noncomputable def p := 18
noncomputable def q := 3
noncomputable def r : ℝ := -27 / 2

theorem area_evaluation :
  p + q + r = 7.5 :=
by
  sorry

end area_evaluation_l311_311149


namespace binary_101_to_decimal_l311_311472

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l311_311472


namespace max_value_of_b_minus_a_l311_311037

theorem max_value_of_b_minus_a (a b : ℝ) (h₀ : a < 0)
  (h₁ : ∀ x : ℝ, a < x ∧ x < b → (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) :
  b - a ≤ 2017 :=
sorry

end max_value_of_b_minus_a_l311_311037


namespace odd_factors_360_l311_311655

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l311_311655


namespace photos_per_album_correct_l311_311895

-- Define the conditions
def total_photos : ℕ := 4500
def first_batch_photos : ℕ := 1500
def first_batch_albums : ℕ := 30
def second_batch_albums : ℕ := 60
def remaining_photos : ℕ := total_photos - first_batch_photos

-- Define the number of photos per album for the first batch (should be 50)
def photos_per_album_first_batch : ℕ := first_batch_photos / first_batch_albums

-- Define the number of photos per album for the second batch (should be 50)
def photos_per_album_second_batch : ℕ := remaining_photos / second_batch_albums

-- Statement to prove
theorem photos_per_album_correct :
  photos_per_album_first_batch = 50 ∧ photos_per_album_second_batch = 50 :=
by
  simp [photos_per_album_first_batch, photos_per_album_second_batch, remaining_photos]
  sorry

end photos_per_album_correct_l311_311895


namespace cara_cats_correct_l311_311841

def martha_cats_rats : ℕ := 3
def martha_cats_birds : ℕ := 7
def martha_cats_animals : ℕ := martha_cats_rats + martha_cats_birds

def cara_cats_animals : ℕ := 5 * martha_cats_animals - 3

theorem cara_cats_correct : cara_cats_animals = 47 :=
by
  -- Proof omitted
  -- Here's where the actual calculation steps would go, but we'll just use sorry for now.
  sorry

end cara_cats_correct_l311_311841


namespace min_moves_to_zero_l311_311568

-- Define the problem setting and conditions

def initial_counters : ℕ := 28
def max_value : ℕ := 2017

-- Definition for the minimum number of moves required to reduce all counters to zero

theorem min_moves_to_zero : 
  ∀ (counters : list ℕ), (∀ c ∈ counters, 1 ≤ c ∧ c ≤ max_value) → counters.length = initial_counters →
  ∃ (m : ℕ), m = 11 ∧ 
    (∀ (f : ℕ → ℕ → ℕ), f 0 0 = 0 → (∃ i, 0 < i ∧ i ≤ m ∧ ∀ n ∈ counters, f i n = 0)) :=
by
  sorry

end min_moves_to_zero_l311_311568


namespace simple_interest_sum_l311_311004

theorem simple_interest_sum (SI R T : ℝ) (hSI : SI = 4016.25) (hR : R = 0.01) (hT : T = 3) :
  SI / (R * T) = 133875 := by
  sorry

end simple_interest_sum_l311_311004


namespace find_c_l311_311822

variable (y c : ℝ)

theorem find_c (h : y > 0) (h_expr : (7 * y / 20 + c * y / 10) = 0.6499999999999999 * y) : c = 3 := by
  sorry

end find_c_l311_311822


namespace similar_triangles_iff_sides_proportional_l311_311077

theorem similar_triangles_iff_sides_proportional
  (a b c a1 b1 c1 : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a1 ∧ 0 < b1 ∧ 0 < c1) :
  (Real.sqrt (a * a1) + Real.sqrt (b * b1) + Real.sqrt (c * c1) =
   Real.sqrt ((a + b + c) * (a1 + b1 + c1))) ↔
  (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
by
  sorry

end similar_triangles_iff_sides_proportional_l311_311077


namespace sum_of_roots_eq_p_l311_311374

variable (p q : ℝ)
variable (hq : q = p^2 - 1)

theorem sum_of_roots_eq_p (h : q = p^2 - 1) : 
  let r1 := p
  let r2 := q
  r1 + r2 = p := 
sorry

end sum_of_roots_eq_p_l311_311374


namespace equilateral_triangle_side_length_l311_311871

theorem equilateral_triangle_side_length (a : ℝ) (h : 3 * a = 18) : a = 6 :=
by
  sorry

end equilateral_triangle_side_length_l311_311871


namespace sin_75_l311_311612

theorem sin_75 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_l311_311612


namespace cupcakes_difference_l311_311604

theorem cupcakes_difference (h : ℕ) (betty_rate : ℕ) (dora_rate : ℕ) (betty_break : ℕ) 
  (cupcakes_difference : ℕ) 
  (H₁ : betty_rate = 10) 
  (H₂ : dora_rate = 8) 
  (H₃ : betty_break = 2) 
  (H₄ : cupcakes_difference = 10) : 
  8 * h - 10 * (h - 2) = 10 → h = 5 :=
by
  intro H
  sorry

end cupcakes_difference_l311_311604


namespace ali_babas_cave_min_moves_l311_311569

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l311_311569


namespace find_n_l311_311928

theorem find_n (n : ℤ) (h1 : n > 4)
  (h2 : ∀ (x y : ℂ), (2 * n * (complex.sqrt y) - n) = (-1)^(n - 3) * 2 * n * (n - 1) * (n - 2) * y) :
  n = 6 :=
by sorry

end find_n_l311_311928


namespace employees_bonus_l311_311564

theorem employees_bonus (x y z : ℝ) 
  (h1 : x + y + z = 2970) 
  (h2 : y = (1 / 3) * x + 180) 
  (h3 : z = (1 / 3) * y + 130) :
  x = 1800 ∧ y = 780 ∧ z = 390 :=
by
  sorry

end employees_bonus_l311_311564


namespace total_annual_donation_l311_311074

-- Defining the conditions provided in the problem
def monthly_donation : ℕ := 1707
def months_in_year : ℕ := 12

-- Stating the theorem that answers the question
theorem total_annual_donation : monthly_donation * months_in_year = 20484 := 
by
  -- The proof is omitted for brevity
  sorry

end total_annual_donation_l311_311074


namespace arithmetic_sequence_formula_sum_Tn_formula_l311_311358

variable {a : ℕ → ℤ} -- The sequence a_n
variable {S : ℕ → ℤ} -- The sum S_n
variable {a₃ : ℤ} (h₁ : a₃ = 20)
variable {S₃ S₄ : ℤ} (h₂ : 2 * S₃ = S₄ + 8)

/- The general formula for the arithmetic sequence a_n -/
theorem arithmetic_sequence_formula (d : ℤ) (a₁ : ℤ)
  (h₃ : (a₃ = a₁ + 2 * d))
  (h₄ : (S₃ = 3 * a₁ + 3 * d))
  (h₅ : (S₄ = 4 * a₁ + 6 * d)) :
  ∀ n : ℕ, a n = 8 * n - 4 :=
by
  sorry

variable {b : ℕ → ℚ} -- Define b_n
variable {T : ℕ → ℚ} -- Define T_n
variable {S_general : ℕ → ℚ} (h₆ : ∀ n, S n = 4 * n ^ 2)
variable {b_general : ℚ → ℚ} (h₇ : ∀ n, b n = 1 / (S n - 1))
variable {T_general : ℕ → ℚ} -- Define T_n

/- The formula for T_n given b_n -/
theorem sum_Tn_formula :
  ∀ n : ℕ, T n = n / (2 * n + 1) :=
by
  sorry

end arithmetic_sequence_formula_sum_Tn_formula_l311_311358


namespace combinations_of_balls_and_hats_l311_311889

def validCombinations (b h : ℕ) : Prop :=
  6 * b + 4 * h = 100 ∧ h ≥ 2

theorem combinations_of_balls_and_hats : 
  (∃ (n : ℕ), n = 8 ∧ (∀ b h : ℕ, validCombinations b h → validCombinations b h)) :=
by
  sorry

end combinations_of_balls_and_hats_l311_311889


namespace probability_at_least_one_die_shows_three_l311_311438

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end probability_at_least_one_die_shows_three_l311_311438


namespace non_congruent_triangles_with_perimeter_18_l311_311946

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l311_311946


namespace puzzles_pieces_count_l311_311075

theorem puzzles_pieces_count :
  let pieces_per_hour := 100
  let hours_per_day := 7
  let days := 7
  let total_pieces_can_put_together := pieces_per_hour * hours_per_day * days
  let pieces_per_puzzle1 := 300
  let number_of_puzzles1 := 8
  let total_pieces_puzzles1 := pieces_per_puzzle1 * number_of_puzzles1
  let remaining_pieces := total_pieces_can_put_together - total_pieces_puzzles1
  let number_of_puzzles2 := 5
  remaining_pieces / number_of_puzzles2 = 500
:= by
  sorry

end puzzles_pieces_count_l311_311075


namespace smallest_value_between_0_and_1_l311_311681

theorem smallest_value_between_0_and_1 (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3 * y ∧ y^3 < y^(1/3 : ℝ) ∧ y^3 < 1 ∧ y^3 < 1 / y :=
by
  sorry

end smallest_value_between_0_and_1_l311_311681


namespace sequence_explicit_form_l311_311692

noncomputable def a_sequence : ℕ → ℝ
| 0       := 2 * Real.sqrt 3
| (n + 1) := (4 * a_sequence n) / (4 - (a_sequence n)^2)

theorem sequence_explicit_form (n : ℕ) :
  a_sequence n = 2 * Real.tan (Real.pi / (3 * 2^n)) :=
sorry

end sequence_explicit_form_l311_311692


namespace compare_fractions_l311_311322

theorem compare_fractions : -(2 / 3 : ℚ) < -(3 / 5 : ℚ) :=
by sorry

end compare_fractions_l311_311322


namespace sin_expression_value_l311_311678

theorem sin_expression_value (α : ℝ) (h : Real.cos (α + π / 5) = 4 / 5) :
  Real.sin (2 * α + 9 * π / 10) = 7 / 25 :=
sorry

end sin_expression_value_l311_311678


namespace wall_area_l311_311595

-- Define the conditions
variables (R J D : ℕ) (L W : ℝ)
variable (area_regular_tiles : ℝ)
variables (ratio_regular : ℕ) (ratio_jumbo : ℕ) (ratio_diamond : ℕ)
variables (length_ratio_jumbo : ℝ) (width_ratio_jumbo : ℝ)
variables (length_ratio_diamond : ℝ) (width_ratio_diamond : ℝ)
variable (total_area : ℝ)

-- Assign values to the conditions
axiom ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1
axiom size_regular : area_regular_tiles = 80
axiom jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3
axiom diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5

-- Define the statement
theorem wall_area (ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1)
    (size_regular : area_regular_tiles = 80)
    (jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3)
    (diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5):
    total_area = 140 := 
sorry

end wall_area_l311_311595


namespace proof_problem_l311_311404

open Real

noncomputable def problem_condition1 (A B : ℝ) : Prop :=
  (sin A - sin B) * (sin A + sin B) = sin (π/3 - B) * sin (π/3 + B)

noncomputable def problem_condition2 (b c : ℝ) (a : ℝ) (dot_product : ℝ) : Prop :=
  b * c * cos (π / 3) = dot_product ∧ a = 2 * sqrt 7

noncomputable def problem_condition3 (a b c : ℝ) : Prop := 
  a^2 = (b + c)^2 - 3 * b * c

noncomputable def problem_condition4 (b c : ℝ) : Prop := 
  b < c

theorem proof_problem (A B : ℝ) (a b c dot_product : ℝ)
  (h1 : problem_condition1 A B)
  (h2 : problem_condition2 b c a dot_product)
  (h3 : problem_condition3 a b c)
  (h4 : problem_condition4 b c) :
  (A = π / 3) ∧ (b = 4 ∧ c = 6) :=
by {
  sorry
}

end proof_problem_l311_311404


namespace circle_center_l311_311591

theorem circle_center (a b : ℝ)
  (passes_through_point : (a - 0)^2 + (b - 9)^2 = r^2)
  (is_tangent : (a - 3)^2 + (b - 9)^2 = r^2 ∧ b = 6 * (a - 3) + 9 ∧ (b - 9) / (a - 3) = -1/6) :
  a = 3/2 ∧ b = 37/4 := 
by 
  sorry

end circle_center_l311_311591


namespace number_of_odd_factors_of_360_l311_311659

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l311_311659


namespace parabola_expression_l311_311795

theorem parabola_expression :
  ∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x - 5 = 0 → (x = -1 ∨ x = 5)) ∧ (a * (-1)^2 + b * (-1) - 5 = 0) ∧ (a * 5^2 + b * 5 - 5 = 0) ∧ (a * 1 - 4 = 1) :=
sorry

end parabola_expression_l311_311795


namespace probability_divisor_of_8_is_half_l311_311883

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l311_311883


namespace max_value_of_expression_l311_311529

variable (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = 3)

theorem max_value_of_expression :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end max_value_of_expression_l311_311529


namespace different_suits_choice_count_l311_311196

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l311_311196


namespace num_int_solutions_l311_311909

theorem num_int_solutions (x : ℤ) : 
  (x^4 - 39 * x^2 + 140 < 0) ↔ (x = 3 ∨ x = -3 ∨ x = 4 ∨ x = -4 ∨ x = 5 ∨ x = -5) := 
sorry

end num_int_solutions_l311_311909


namespace number_of_cyclic_sets_l311_311826

-- Definition of conditions: number of teams and wins/losses
def num_teams : ℕ := 21
def wins (team : ℕ) : ℕ := 12
def losses (team : ℕ) : ℕ := 8
def played_everyone_once (team1 team2 : ℕ) : Prop := (team1 ≠ team2)

-- Proposition to prove:
theorem number_of_cyclic_sets (h_teams: ∀ t, wins t = 12 ∧ losses t = 8)
  (h_played_once: ∀ t1 t2, played_everyone_once t1 t2) : 
  ∃ n, n = 144 :=
sorry

end number_of_cyclic_sets_l311_311826


namespace academic_academy_pass_criteria_l311_311603

theorem academic_academy_pass_criteria :
  ∀ (total_problems : ℕ) (passing_percentage : ℕ)
  (max_missed : ℕ),
  total_problems = 35 →
  passing_percentage = 80 →
  max_missed = total_problems - (passing_percentage * total_problems) / 100 →
  max_missed = 7 :=
by 
  intros total_problems passing_percentage max_missed
  intros h_total_problems h_passing_percentage h_calculation
  rw [h_total_problems, h_passing_percentage] at h_calculation
  sorry

end academic_academy_pass_criteria_l311_311603


namespace negation_of_p_l311_311702

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.exp x > Real.log x

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.exp x ≤ Real.log x

-- The statement we want to prove
theorem negation_of_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_l311_311702


namespace team_total_games_123_l311_311767

theorem team_total_games_123 {G : ℕ} 
  (h1 : (55 / 100) * 35 + (90 / 100) * (G - 35) = (80 / 100) * G) : 
  G = 123 :=
sorry

end team_total_games_123_l311_311767


namespace third_prize_probability_winning_prize_probability_l311_311051

noncomputable def probability_winning_third_prize : ℚ :=
  1 / 4

noncomputable def probability_winning_prize : ℚ :=
  9 / 16

theorem third_prize_probability (draws : Finset (ℕ × ℕ))
    (balls : Fin 4)
    (draws_considered : Finset.univ = {(0, 3), (1, 2), (2, 1), (3, 0)} ) :
    probability_winning_third_prize = (4 / 16 : ℚ) := 
  by sorry

theorem winning_prize_probability (draws : Finset (ℕ × ℕ))
    (balls : Fin 4)
    (draws_considered : Finset.univ = {(0, 3), (1, 2), (2, 1), (3, 0), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2)}) :
    probability_winning_prize = (9 / 16 : ℚ) := 
  by sorry

end third_prize_probability_winning_prize_probability_l311_311051


namespace cylinder_heights_relationship_l311_311736

variables {r1 r2 h1 h2 : ℝ}

theorem cylinder_heights_relationship
    (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
    (radius_relation : r2 = 1.2 * r1) :
    h1 = 1.44 * h2 :=
by sorry

end cylinder_heights_relationship_l311_311736


namespace non_congruent_triangles_with_perimeter_18_l311_311945

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l311_311945


namespace odd_factors_of_360_l311_311652

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l311_311652


namespace repeating_prime_exists_l311_311929

open Nat

theorem repeating_prime_exists (p : Fin 2021 → ℕ) 
  (prime_seq : ∀ i : Fin 2021, Nat.Prime (p i))
  (diff_condition : ∀ i : Fin 2019, (p (i + 1) - p i = 6 ∨ p (i + 1) - p i = 12) ∧ (p (i + 2) - p (i + 1) = 6 ∨ p (i + 2) - p (i + 1) = 12)) : 
  ∃ i j : Fin 2021, i ≠ j ∧ p i = p j := by
  sorry

end repeating_prime_exists_l311_311929


namespace proof_problem_l311_311213

-- Definitions for the arithmetic and geometric sequences
def a_n (n : ℕ) : ℚ := 2 * n - 4
def b_n (n : ℕ) : ℚ := 2^(n - 2)

-- Conditions based on initial problem statements
axiom a_2 : a_n 2 = 0
axiom b_2 : b_n 2 = 1
axiom a_3_eq_b_3 : a_n 3 = b_n 3
axiom a_4_eq_b_4 : a_n 4 = b_n 4

-- Sum of first n terms of the sequence {n * b_n}
def S_n (n : ℕ) : ℚ := (n-1) * 2^(n-1) + 1/2

-- The main theorem to prove
theorem proof_problem (n : ℕ) : ∃ a_n b_n S_n, 
    (a_n = 2 * n - 4) ∧
    (b_n = 2^(n - 2)) ∧
    (S_n = (n-1) * 2^(n-1) + 1/2) :=
by {
    sorry
}

end proof_problem_l311_311213


namespace bad_carrots_l311_311452

theorem bad_carrots (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) (total_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : good_carrots = 38)
  (h4 : total_carrots = carol_carrots + mom_carrots)
  (h5 : bad_carrots = total_carrots - good_carrots) :
  bad_carrots = 7 := by
  sorry

end bad_carrots_l311_311452


namespace tan_C_in_triangle_l311_311389

theorem tan_C_in_triangle (A B C : ℝ) (hA : Real.tan A = 1 / 2) (hB : Real.cos B = 3 * Real.sqrt 10 / 10) :
  Real.tan C = -1 :=
sorry

end tan_C_in_triangle_l311_311389


namespace line_slope_through_origin_intersects_parabola_l311_311128

theorem line_slope_through_origin_intersects_parabola (k : ℝ) :
  (∃ x1 x2 : ℝ, 5 * (kx1) = 2 * x1 ^ 2 - 9 * x1 + 10 ∧ 5 * (kx2) = 2 * x2 ^ 2 - 9 * x2 + 10 ∧ x1 + x2 = 77) → k = 29 :=
by
  intro h
  sorry

end line_slope_through_origin_intersects_parabola_l311_311128


namespace number_of_positive_area_triangles_l311_311673

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l311_311673


namespace total_number_of_people_l311_311261

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l311_311261


namespace arccot_identity_problem_statement_l311_311749

noncomputable def arccot (x : ℝ) : ℝ := 
  if x = 0 then π/2 
  else if x > 0 then arctan (1/x) 
  else π + arctan (1/x)

theorem arccot_identity (x : ℝ) : arccot x + arctan x = π / 2 :=
begin
  by_cases h : x > 0,
  { rw [arccot, if_pos h, arctan_add_arctan_one_div h], norm_num },
  { rw [arccot, if_neg (ne_of_gt (lt_of_not_ge h))], norm_num }
end

theorem problem_statement : 
  2 * arccot (-1/2) + arccot (-2) = 2 * π := 
by 
  sorry

end arccot_identity_problem_statement_l311_311749


namespace container_capacity_l311_311390

theorem container_capacity (C : ℝ) (h₁ : C > 15) (h₂ : 0 < (81 : ℝ)) (h₃ : (337 : ℝ) > 0) :
  ((C - 15) / C) ^ 4 = 81 / 337 :=
sorry

end container_capacity_l311_311390


namespace cot_half_angle_product_geq_3sqrt3_l311_311831

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_half_angle_product_geq_3sqrt3 {A B C : ℝ} (h : A + B + C = π) :
    cot (A / 2) * cot (B / 2) * cot (C / 2) ≥ 3 * Real.sqrt 3 := 
  sorry

end cot_half_angle_product_geq_3sqrt3_l311_311831


namespace odd_factors_count_l311_311367

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l311_311367


namespace apples_left_over_l311_311532

-- Defining the number of apples collected by Liam, Mia, and Noah
def liam_apples := 53
def mia_apples := 68
def noah_apples := 22

-- The total number of apples collected
def total_apples := liam_apples + mia_apples + noah_apples

-- Proving that the remainder when the total number of apples is divided by 10 is 3
theorem apples_left_over : total_apples % 10 = 3 := by
  -- Placeholder for proof
  sorry

end apples_left_over_l311_311532


namespace cafeteria_students_count_l311_311864

def total_students : ℕ := 90

def initial_in_cafeteria : ℕ := total_students * 2 / 3

def initial_outside : ℕ := total_students / 3

def ran_inside : ℕ := initial_outside / 3

def ran_outside : ℕ := 3

def net_change_in_cafeteria : ℕ := ran_inside - ran_outside

def final_in_cafeteria : ℕ := initial_in_cafeteria + net_change_in_cafeteria

theorem cafeteria_students_count : final_in_cafeteria = 67 := 
by
  sorry

end cafeteria_students_count_l311_311864


namespace water_balloon_packs_l311_311234

theorem water_balloon_packs (P : ℕ) : 
  (6 * P + 12 = 30) → P = 3 := by
  sorry

end water_balloon_packs_l311_311234


namespace find_f6_l311_311722

variable {R : Type*} [AddGroup R] [Semiring R]

def functional_equation (f : R → R) :=
∀ x y : R, f (x + y) = f x + f y

theorem find_f6 (f : ℝ → ℝ) (h1 : functional_equation f) (h2 : f 4 = 10) : f 6 = 10 :=
sorry

end find_f6_l311_311722


namespace ladybugs_with_spots_l311_311419

theorem ladybugs_with_spots (total_ladybugs without_spots with_spots : ℕ) 
  (h1 : total_ladybugs = 67082) 
  (h2 : without_spots = 54912) 
  (h3 : with_spots = total_ladybugs - without_spots) : 
  with_spots = 12170 := 
by 
  -- hole for the proof 
  sorry

end ladybugs_with_spots_l311_311419


namespace function_zero_solution_l311_311342

def floor (x : ℝ) : ℤ := sorry -- Define floor function properly.

theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = (-1) ^ (floor y) * f x + (-1) ^ (floor x) * f y) →
  (∀ x : ℝ, f x = 0) := 
by
  -- Proof goes here
  sorry

end function_zero_solution_l311_311342


namespace total_houses_in_lincoln_county_l311_311731

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (houses_built : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : houses_built = 97741) : 
  original_houses + houses_built = 118558 := 
by 
  -- Proof steps or tactics would go here
  sorry

end total_houses_in_lincoln_county_l311_311731


namespace corrected_mean_l311_311288

theorem corrected_mean (n : ℕ) (mean old_obs new_obs : ℝ) 
    (obs_count : n = 50) (old_mean : mean = 36) (incorrect_obs : old_obs = 23) (correct_obs : new_obs = 46) :
    (mean * n - old_obs + new_obs) / n = 36.46 := by
  sorry

end corrected_mean_l311_311288


namespace non_adjacent_divisibility_l311_311543

theorem non_adjacent_divisibility (a : Fin 7 → ℕ) (h : ∀ i, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) :
  ∃ i j : Fin 7, i ≠ j ∧ (¬(i + 1)%7 = j) ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end non_adjacent_divisibility_l311_311543


namespace distance_between_cities_l311_311719

variable (a b : Nat)

theorem distance_between_cities :
  (a = (10 * a + b) - (10 * b + a)) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → 10 * a + b = 98 := by
  sorry

end distance_between_cities_l311_311719


namespace plane_angle_divides_cube_l311_311598

noncomputable def angle_between_planes (m n : ℕ) (h : m ≤ n) : ℝ :=
  Real.arctan (2 * m / (m + n))

theorem plane_angle_divides_cube (m n : ℕ) (h : m ≤ n) :
  ∃ α, α = angle_between_planes m n h :=
sorry

end plane_angle_divides_cube_l311_311598


namespace inequality_holds_l311_311750

variable (b : ℝ)

theorem inequality_holds (b : ℝ) : (3 * b - 1) * (4 * b + 1) > (2 * b + 1) * (5 * b - 3) :=
by
  sorry

end inequality_holds_l311_311750


namespace grain_to_rice_system_l311_311518

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end grain_to_rice_system_l311_311518


namespace trig_expression_value_l311_311617

theorem trig_expression_value : 
  (2 * (Real.sin (25 * Real.pi / 180))^2 - 1) / 
  (Real.sin (20 * Real.pi / 180) * Real.cos (20 * Real.pi / 180)) = -2 := 
by
  -- Proof goes here
  sorry

end trig_expression_value_l311_311617


namespace percentage_primes_divisible_by_3_l311_311282

theorem percentage_primes_divisible_by_3 : 
  (let primes_lt_20 := {2, 3, 5, 7, 11, 13, 17, 19};
       primes_div_by_3 := primes_lt_20.filter (λ x, x % 3 = 0) in
   100 * primes_div_by_3.card / primes_lt_20.card = 12.5) := sorry

end percentage_primes_divisible_by_3_l311_311282


namespace total_accepted_cartons_l311_311060

-- Definitions for the number of cartons delivered and damaged for each customer
def cartons_delivered_first_two : Nat := 300
def cartons_delivered_last_three : Nat := 200

def cartons_damaged_first : Nat := 70
def cartons_damaged_second : Nat := 50
def cartons_damaged_third : Nat := 40
def cartons_damaged_fourth : Nat := 30
def cartons_damaged_fifth : Nat := 20

-- Statement to prove
theorem total_accepted_cartons :
  let accepted_first := cartons_delivered_first_two - cartons_damaged_first
  let accepted_second := cartons_delivered_first_two - cartons_damaged_second
  let accepted_third := cartons_delivered_last_three - cartons_damaged_third
  let accepted_fourth := cartons_delivered_last_three - cartons_damaged_fourth
  let accepted_fifth := cartons_delivered_last_three - cartons_damaged_fifth
  accepted_first + accepted_second + accepted_third + accepted_fourth + accepted_fifth = 990 :=
by
  sorry

end total_accepted_cartons_l311_311060


namespace taxi_ride_cost_l311_311309

-- Define the base fare
def base_fare : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the distance traveled
def distance : ℝ := 8.00

-- Define the total cost function
def total_cost (base : ℝ) (per_mile : ℝ) (miles : ℝ) : ℝ :=
  base + (per_mile * miles)

-- The statement to prove: the total cost of an 8-mile taxi ride
theorem taxi_ride_cost : total_cost base_fare cost_per_mile distance = 4.40 :=
by
sorry

end taxi_ride_cost_l311_311309


namespace ji_hoon_original_answer_l311_311223

-- Define the conditions: Ji-hoon's mistake
def ji_hoon_mistake (x : ℝ) := x - 7 = 0.45

-- The theorem statement
theorem ji_hoon_original_answer (x : ℝ) (h : ji_hoon_mistake x) : x * 7 = 52.15 :=
by
  sorry

end ji_hoon_original_answer_l311_311223


namespace cannot_fit_rectangle_l311_311290

theorem cannot_fit_rectangle 
  (w1 h1 : ℕ) (w2 h2 : ℕ) 
  (h1_pos : 0 < h1) (w1_pos : 0 < w1)
  (h2_pos : 0 < h2) (w2_pos : 0 < w2) :
  w1 = 5 → h1 = 6 → w2 = 3 → h2 = 8 →
  ¬(w2 ≤ w1 ∧ h2 ≤ h1) :=
by
  intros H1 W1 H2 W2
  sorry

end cannot_fit_rectangle_l311_311290


namespace different_suits_card_combinations_l311_311198

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l311_311198


namespace perfect_square_of_polynomial_l311_311967

theorem perfect_square_of_polynomial (k : ℝ) (h : ∃ (p : ℝ), ∀ x : ℝ, x^2 + 6*x + k^2 = (x + p)^2) : k = 3 ∨ k = -3 := 
sorry

end perfect_square_of_polynomial_l311_311967


namespace max_value_f_l311_311385

noncomputable def op_add (a b : ℝ) : ℝ :=
if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
(op_add 1 x) + (op_add 2 x)

theorem max_value_f :
  ∃ x ∈ Set.Icc (-2 : ℝ) 3, ∀ y ∈ Set.Icc (-2 : ℝ) 3, f y ≤ f x := 
sorry

end max_value_f_l311_311385


namespace inequality_has_real_solution_l311_311296

variable {f : ℝ → ℝ}

theorem inequality_has_real_solution (h : ∃ x : ℝ, f x > 0) : 
    (∃ x : ℝ, f x > 0) :=
by
  sorry

end inequality_has_real_solution_l311_311296


namespace problem_divisibility_l311_311350

theorem problem_divisibility 
  (a b c : ℕ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : b ∣ a^3)
  (h2 : c ∣ b^3)
  (h3 : a ∣ c^3) : 
  (a + b + c) ^ 13 ∣ a * b * c := 
sorry

end problem_divisibility_l311_311350


namespace value_a8_l311_311840

def sequence_sum (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem value_a8 : a 8 = 15 :=
by
  sorry

end value_a8_l311_311840


namespace interest_rate_second_share_l311_311769

variable (T : ℝ) (r1 : ℝ) (I2 : ℝ) (T_i : ℝ)

theorem interest_rate_second_share 
  (h1 : T = 100000)
  (h2 : r1 = 0.09)
  (h3 : I2 = 24999.999999999996)
  (h4 : T_i = 0.095 * T) : 
  (2750 / I2) * 100 = 11 :=
by {
  sorry
}

end interest_rate_second_share_l311_311769


namespace find_t_l311_311535

-- Define the utility function
def utility (r j : ℕ) : ℕ := r * j

-- Define the Wednesday and Thursday utilities
def utility_wednesday (t : ℕ) : ℕ := utility (t + 1) (7 - t)
def utility_thursday (t : ℕ) : ℕ := utility (3 - t) (t + 4)

theorem find_t : (utility_wednesday t = utility_thursday t) → t = 5 / 8 :=
by
  sorry

end find_t_l311_311535


namespace fraction_of_succeeding_number_l311_311412

theorem fraction_of_succeeding_number (N : ℝ) (hN : N = 24.000000000000004) :
  ∃ f : ℝ, (1 / 4) * N > f * (N + 1) + 1 ∧ f = 0.2 :=
by
  sorry

end fraction_of_succeeding_number_l311_311412


namespace percentage_primes_divisible_by_3_l311_311281

theorem percentage_primes_divisible_by_3 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100 
  percentage = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let primes_div_by_3 := {p ∈ primes | p % 3 = 0}
  let percentage := (primes_div_by_3.card.toReal / primes.card.toReal) * 100
  exact sorry

end percentage_primes_divisible_by_3_l311_311281


namespace smallest_positive_z_l311_311420

open Real

theorem smallest_positive_z (x y z : ℝ) (m k n : ℤ) 
  (h1 : cos x = 0) 
  (h2 : sin y = 1) 
  (h3 : cos (x + z) = -1 / 2) :
  z = 5 * π / 6 :=
by
  sorry

end smallest_positive_z_l311_311420


namespace digit_172_in_decimal_expansion_of_5_over_13_l311_311868

theorem digit_172_in_decimal_expansion_of_5_over_13 : 
  (decimal_digit (rat.of_int 5 / 13) 172) = 6 :=
sorry

end digit_172_in_decimal_expansion_of_5_over_13_l311_311868


namespace sum_of_reciprocal_transformed_roots_l311_311614

theorem sum_of_reciprocal_transformed_roots :
  ∀ (a b c : ℝ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    -1 < a ∧ a < 1 ∧
    -1 < b ∧ b < 1 ∧
    -1 < c ∧ c < 1 ∧
    (45 * a ^ 3 - 70 * a ^ 2 + 28 * a - 2 = 0) ∧
    (45 * b ^ 3 - 70 * b ^ 2 + 28 * b - 2 = 0) ∧
    (45 * c ^ 3 - 70 * c ^ 2 + 28 * c - 2 = 0)
  → (1 - a)⁻¹ + (1 - b)⁻¹ + (1 - c)⁻¹ = 13 / 9 := 
by 
  sorry

end sum_of_reciprocal_transformed_roots_l311_311614


namespace kayak_manufacture_total_l311_311777

theorem kayak_manufacture_total :
  let feb : ℕ := 5
  let mar : ℕ := 3 * feb
  let apr : ℕ := 3 * mar
  let may : ℕ := 3 * apr
  feb + mar + apr + may = 200 := by
  sorry

end kayak_manufacture_total_l311_311777


namespace correct_solutions_l311_311578

noncomputable def sample_size : ℕ := 5
noncomputable def population_size : ℕ := 50
noncomputable def prob_individual_selected : ℝ := 0.1

def data_set : List ℝ := [10, 11, 11, 12, 13, 14, 16, 18, 20, 22]

noncomputable def transformed_variance : ℝ := 8
noncomputable def original_variance : ℝ := 2

noncomputable def strata_mean_1 : ℝ := 0  -- placeholder, as it equals strata_mean_2
noncomputable def strata_mean_2 : ℝ := strata_mean_1
noncomputable def strata_var_1 : ℝ := 0  -- needs an appropriate value
noncomputable def strata_var_2 : ℝ := 0  -- needs an appropriate value
noncomputable def population_variance : ℝ := 1 / 2 * (strata_var_1 + strata_var_2)

theorem correct_solutions : 
  (prob_individual_selected = (sample_size:ℝ) / (population_size:ℝ)) ∧
  (list.nth_le data_set 5 sorry = 14 → list.nth_le data_set 6 sorry = 16 → 
   (list.nth_le data_set 5 sorry + list.nth_le data_set 6 sorry) / 2 ≠ 15) ∧
  (transformed_variance = 4 * original_variance) ∧
  population_variance ≠ 1 / 2 * (strata_var_1 + strata_var_2) →
  "{A, C}" = "{correct_solutions}"
:= sorry

end correct_solutions_l311_311578


namespace probability_prime_l311_311547

open finset

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

def possible_outcomes : finset ℕ := finset.range 7

def prime_outcomes : finset ℕ := {2, 3, 5}

theorem probability_prime :
  (prime_outcomes.card.to_real / possible_outcomes.card.to_real) = 1 / 2 :=
by
  sorry

end probability_prime_l311_311547


namespace probability_of_event_l311_311097

-- Definitions for the problem setup

-- Box C and its range
def boxC := {i : ℕ | 1 ≤ i ∧ i ≤ 30}

-- Box D and its range
def boxD := {i : ℕ | 21 ≤ i ∧ i ≤ 50}

-- Condition for a tile from box C being less than 20
def tile_from_C_less_than_20 (i : ℕ) : Prop := i ∈ boxC ∧ i < 20

-- Condition for a tile from box D being odd or greater than 45
def tile_from_D_odd_or_greater_than_45 (i : ℕ) : Prop := i ∈ boxD ∧ (i % 2 = 1 ∨ i > 45)

-- Main statement
theorem probability_of_event :
  (19 / 30 : ℚ) * (17 / 30 : ℚ) = (323 / 900 : ℚ) :=
by sorry

end probability_of_event_l311_311097


namespace empty_set_l311_311315

def setA := {x : ℝ | x^2 - 4 = 0}
def setB := {x : ℝ | x > 9 ∨ x < 3}
def setC := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}
def setD := {x : ℝ | x > 9 ∧ x < 3}

theorem empty_set : setD = ∅ := 
  sorry

end empty_set_l311_311315


namespace totalSleepIsThirtyHours_l311_311395

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l311_311395


namespace domain_f_correct_domain_g_correct_l311_311488

noncomputable def domain_f : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ x ≠ 1}

noncomputable def expected_domain_f : Set ℝ :=
  {x | (-1 ≤ x ∧ x < 1) ∨ x > 1}

theorem domain_f_correct :
  domain_f = expected_domain_f :=
by
  sorry

noncomputable def domain_g : Set ℝ :=
  {x | 3 - 4 * x > 0}

noncomputable def expected_domain_g : Set ℝ :=
  {x | x < 3 / 4}

theorem domain_g_correct :
  domain_g = expected_domain_g :=
by
  sorry

end domain_f_correct_domain_g_correct_l311_311488


namespace binary_101_is_5_l311_311479

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l311_311479


namespace quadratic_root_l311_311163

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end quadratic_root_l311_311163


namespace booklet_cost_l311_311715

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) ∧ (12 * b > 17) → b = 1.42 := by
  sorry

end booklet_cost_l311_311715


namespace find_integer_part_of_m_l311_311428

theorem find_integer_part_of_m {m : ℝ} (h_lecture_duration : m > 0) 
    (h_swap_positions : ∃ k : ℤ, 120 + m = 60 + k * 12 * 60 / 13 ∧ (120 + m) % 60 = 60 * (120 + m) / 720) : 
    ⌊m⌋ = 46 :=
by
  sorry

end find_integer_part_of_m_l311_311428


namespace min_value_of_inverse_sum_l311_311169

variable (n : ℕ) (p q : ℝ)

-- We need to define that a variable X follows a binomial distribution B(n, p)
-- and has expectations and variance as given.
noncomputable def X := Binomial n p

axiom h1 : (X.bExpectation = 4)
axiom h2 : (X.bVariance = q)

theorem min_value_of_inverse_sum : (frac_one_div_p + frac_one_div_q ≥ 9 / 4) :=
by {
  sorry,
}

end min_value_of_inverse_sum_l311_311169


namespace maximum_guaranteed_money_l311_311587

theorem maximum_guaranteed_money (board_width board_height tromino_width tromino_height guaranteed_rubles : ℕ) 
  (h_board_width : board_width = 21) 
  (h_board_height : board_height = 20)
  (h_tromino_width : tromino_width = 3) 
  (h_tromino_height : tromino_height = 1)
  (h_guaranteed_rubles : guaranteed_rubles = 14) :
  true := by
  sorry

end maximum_guaranteed_money_l311_311587


namespace incorrect_tripling_radius_l311_311706

-- Let r be the radius of a circle, and A be its area.
-- The claim is that tripling the radius quadruples the area.
-- We need to prove this claim is incorrect.

theorem incorrect_tripling_radius (r : ℝ) (A : ℝ) (π : ℝ) (hA : A = π * r^2) : 
    (π * (3 * r)^2) ≠ 4 * A :=
by
  sorry

end incorrect_tripling_radius_l311_311706


namespace arcsin_cos_solution_l311_311081

theorem arcsin_cos_solution (x : ℝ) (h : -π/2 ≤ x/3 ∧ x/3 ≤ π/2) :
  x = 3*π/10 ∨ x = 3*π/8 := 
sorry

end arcsin_cos_solution_l311_311081


namespace count_lattice_points_on_hyperbola_l311_311811

theorem count_lattice_points_on_hyperbola : 
  (∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S ↔ (p.1 ^ 2 - p.2 ^ 2 = 1800 ^ 2)) ∧ S.card = 150) :=
sorry

end count_lattice_points_on_hyperbola_l311_311811


namespace complex_unit_circle_sum_l311_311898

theorem complex_unit_circle_sum :
  let z1 := (1 + Complex.I * Real.sqrt 3) / 2
  let z2 := (1 - Complex.I * Real.sqrt 3) / 2
  (z1 ^ 8 + z2 ^ 8 = -1) :=
by
  sorry

end complex_unit_circle_sum_l311_311898


namespace probability_prime_rolled_l311_311548

open Finset

def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def prime_outcomes : Finset ℕ := outcomes.filter is_prime

theorem probability_prime_rolled : (prime_outcomes.card : ℚ) / outcomes.card = 1 / 2 :=
by
  -- Proof would go here
  sorry

end probability_prime_rolled_l311_311548


namespace total_number_of_people_l311_311263

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l311_311263


namespace simplify_expression_l311_311066

variables {R : Type*} [LinearOrderedField R]
variables {a b c x : R}
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem simplify_expression (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ a b c x : R, 
   (h_distinct) →
   ((a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)) →
   ( 
     ( (x + a)^2 / ( (a - b) * (a - c) ) + 
       (x + b)^2 / ( (b - a) * (b - c) ) + 
       (x + c)^2 / ( (c - a) * (c - b) )
     ) = -1
   )
  ) := sorry

end simplify_expression_l311_311066


namespace total_people_count_l311_311257

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l311_311257


namespace valid_six_digit_numbers_l311_311562

def is_divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

def digit_sum (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def is_divisible_by_9 (n : Nat) : Prop :=
  digit_sum n % 9 = 0

def is_valid_six_digit_number (n : Nat) : Prop :=
  ∃ (a b : Nat), n = b * 100000 + 20140 + a ∧ is_divisible_by_4 (10 * 2014 + a) ∧ is_divisible_by_9 (b * 100000 + 20140 + a)

theorem valid_six_digit_numbers :
  { n | is_valid_six_digit_number n } = {220140, 720144, 320148} :=
by
  sorry

end valid_six_digit_numbers_l311_311562


namespace total_number_of_people_l311_311264

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l311_311264


namespace xy_value_l311_311818

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 :=
by
  sorry

end xy_value_l311_311818


namespace integer_combination_zero_l311_311236

theorem integer_combination_zero (a b c : ℤ) (h : a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integer_combination_zero_l311_311236


namespace integer_roots_l311_311487

-- Define the polynomial
def polynomial (x : ℤ) : ℤ := x^3 - 4 * x^2 - 7 * x + 10

-- Define the proof problem statement
theorem integer_roots :
  {x : ℤ | polynomial x = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_l311_311487


namespace impossibility_of_4_level_ideal_interval_tan_l311_311852

def has_ideal_interval (f : ℝ → ℝ) (D : Set ℝ) (k : ℝ) :=
  ∃ (a b : ℝ), a ≤ b ∧ Set.Icc a b ⊆ D ∧ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x) ∧
  (Set.image f (Set.Icc a b) = Set.Icc (k * a) (k * b))

def option_D_incorrect : Prop :=
  ¬ has_ideal_interval (fun x => Real.tan x) (Set.Ioc (-(Real.pi / 2)) (Real.pi / 2)) 4

theorem impossibility_of_4_level_ideal_interval_tan :
  option_D_incorrect :=
sorry

end impossibility_of_4_level_ideal_interval_tan_l311_311852


namespace number_of_positive_area_triangles_l311_311670

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l311_311670


namespace race_distance_l311_311825

theorem race_distance
  (A B : Type)
  (D : ℕ) -- D is the total distance of the race
  (Va Vb : ℕ) -- A's speed and B's speed
  (H1 : D / 28 = Va) -- A's speed calculated from D and time
  (H2 : (D - 56) / 28 = Vb) -- B's speed calculated from distance and time
  (H3 : 56 / 7 = Vb) -- B's speed can also be calculated directly
  (H4 : Va = D / 28)
  (H5 : Vb = (D - 56) / 28) :
  D = 280 := sorry

end race_distance_l311_311825


namespace russia_is_one_third_bigger_l311_311903

theorem russia_is_one_third_bigger (U : ℝ) (Canada Russia : ℝ) 
  (h1 : Canada = 1.5 * U) (h2 : Russia = 2 * U) : 
  (Russia - Canada) / Canada = 1 / 3 :=
by
  sorry

end russia_is_one_third_bigger_l311_311903


namespace part1_solution_set_part2_solution_l311_311531

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem part1_solution_set :
  {x : ℝ | f x > 2} = {x | x > 1} ∪ {x | x < -5} :=
by
  sorry

theorem part2_solution (t : ℝ) :
  (∀ x, f x ≥ t^2 - (11 / 2) * t) ↔ (1 / 2 ≤ t ∧ t ≤ 5) :=
by
  sorry

end part1_solution_set_part2_solution_l311_311531


namespace area_of_border_l311_311002

theorem area_of_border (height_painting width_painting border_width : ℕ)
    (area_painting framed_height framed_width : ℕ)
    (H1 : height_painting = 12)
    (H2 : width_painting = 15)
    (H3 : border_width = 3)
    (H4 : area_painting = height_painting * width_painting)
    (H5 : framed_height = height_painting + 2 * border_width)
    (H6 : framed_width = width_painting + 2 * border_width)
    (area_framed : ℕ)
    (H7 : area_framed = framed_height * framed_width) :
    area_framed - area_painting = 198 := 
sorry

end area_of_border_l311_311002


namespace int_solutions_of_quadratic_l311_311380

theorem int_solutions_of_quadratic (k : ℝ) :
  (∃ x1 x2 : ℝ, 
    (k^2 - 2*k) * x1^2 - (6*k - 4) * x1 + 8 = 0 ∧ 
    (k^2 - 2*k) * x2^2 - (6*k - 4) * x2 + 8 = 0 ∧
    x1 ∈ ℤ ∧ x2 ∈ ℤ) → 
  (k = 0 ∨ k = 2 ∨ k = 1 ∨ k = -2 ∨ k = 2/3) :=
sorry

end int_solutions_of_quadratic_l311_311380


namespace number_of_positive_area_triangles_l311_311672

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l311_311672


namespace largest_inscribed_triangle_area_l311_311468

theorem largest_inscribed_triangle_area (r : ℝ) (h_r : r = 12) : ∃ A : ℝ, A = 144 :=
by
  sorry

end largest_inscribed_triangle_area_l311_311468


namespace percentage_reduction_in_price_l311_311594

-- Definitions based on conditions
def original_price (P : ℝ) (X : ℝ) := P * X
def reduced_price (R : ℝ) (X : ℝ) := R * (X + 5)

-- Theorem statement based on the problem to prove
theorem percentage_reduction_in_price
  (R : ℝ) (H1 : R = 55)
  (H2 : original_price P X = 1100)
  (H3 : reduced_price R X = 1100) :
  ((P - R) / P) * 100 = 25 :=
by
  sorry

end percentage_reduction_in_price_l311_311594


namespace percent_primes_divisible_by_3_less_than_20_l311_311278

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def count_primes_divisible_by_3 (primes: List ℕ) : ℕ :=
  primes.count (λ p => p % 3 = 0)

def percentage (part whole: ℕ) : ℚ :=
  (part * 100) / whole

theorem percent_primes_divisible_by_3_less_than_20 :
  percentage (count_primes_divisible_by_3 primes_less_than_20) primes_less_than_20.length = 12.5 := 
by
  sorry

end percent_primes_divisible_by_3_less_than_20_l311_311278


namespace steve_travel_time_l311_311720

noncomputable def total_travel_time (distance: ℕ) (speed_to_work: ℕ) (speed_back: ℕ) : ℕ :=
  (distance / speed_to_work) + (distance / speed_back)

theorem steve_travel_time : 
  ∀ (distance speed_back speed_to_work : ℕ), 
  (speed_to_work = speed_back / 2) → 
  speed_back = 15 → 
  distance = 30 → 
  total_travel_time distance speed_to_work speed_back = 6 := 
by
  intros
  rw [total_travel_time]
  sorry

end steve_travel_time_l311_311720


namespace fraction_inequality_l311_311639

theorem fraction_inequality 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : 1 / a > 1 / b)
  (h2 : x > y) : 
  x / (x + a) > y / (y + b) := 
  sorry

end fraction_inequality_l311_311639


namespace find_m_plus_n_l311_311244

-- Define the number of ways Blair and Corey can draw the remaining cards
def num_ways_blair_and_corey_draw : ℕ := Nat.choose 50 2

-- Define the function q(a) as given in the problem
noncomputable def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / num_ways_blair_and_corey_draw

-- Define the problem statement to find the minimum value of a for which q(a) >= 1/2
noncomputable def minimum_a : ℤ :=
  if q 7 >= 1/2 then 7 else 36 -- According to the solution, these are the points of interest

-- The final statement to be proved
theorem find_m_plus_n : minimum_a = 7 ∨ minimum_a = 36 :=
  sorry

end find_m_plus_n_l311_311244


namespace primes_divisible_by_3_percentage_l311_311276

def primesLessThanTwenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def countDivisibleBy (n : ℕ) (lst : List ℕ) : Nat :=
  lst.count fun x => x % n == 0

theorem primes_divisible_by_3_percentage : 
  countDivisibleBy 3 primesLessThanTwenty * 100 / primesLessThanTwenty.length = 12.5 :=
by
  sorry

end primes_divisible_by_3_percentage_l311_311276


namespace round_robin_tournament_participant_can_mention_all_l311_311910

theorem round_robin_tournament_participant_can_mention_all :
  ∀ (n : ℕ) (participants : Fin n → Fin n → Prop),
  (∀ i j : Fin n, i ≠ j → (participants i j ∨ participants j i)) →
  (∃ A : Fin n, ∀ (B : Fin n), B ≠ A → (participants A B ∨ ∃ C : Fin n, participants A C ∧ participants C B)) := by
  sorry

end round_robin_tournament_participant_can_mention_all_l311_311910


namespace root_monotonicity_l311_311499

noncomputable def f (x : ℝ) := 3^x + 2 / (1 - x)

theorem root_monotonicity
  (x0 : ℝ) (H_root : f x0 = 0)
  (x1 x2 : ℝ) (H1 : x1 > 1) (H2 : x1 < x0) (H3 : x2 > x0) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end root_monotonicity_l311_311499


namespace complex_number_pure_imaginary_l311_311377

theorem complex_number_pure_imaginary (a : ℝ) 
  (h1 : ∃ a : ℝ, (a^2 - 2*a - 3 = 0) ∧ (a + 1 ≠ 0)) 
  : a = 3 := sorry

end complex_number_pure_imaginary_l311_311377


namespace circle_positions_n_l311_311092

theorem circle_positions_n (n : ℕ) (h1 : n ≥ 23) (h2 : (23 - 7) * 2 + 2 = n) : n = 32 :=
sorry

end circle_positions_n_l311_311092


namespace Tim_Linda_Mow_Lawn_l311_311054

theorem Tim_Linda_Mow_Lawn :
  let tim_time := 1.5
  let linda_time := 2
  let tim_rate := 1 / tim_time
  let linda_rate := 1 / linda_time
  let combined_rate := tim_rate + linda_rate
  let combined_time_hours := 1 / combined_rate
  let combined_time_minutes := combined_time_hours * 60
  combined_time_minutes = 51.43 := 
by
    sorry

end Tim_Linda_Mow_Lawn_l311_311054


namespace pie_filling_cans_l311_311132

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end pie_filling_cans_l311_311132


namespace well_depth_l311_311003

variable (d : ℝ)

-- Conditions
def total_time (t₁ t₂ : ℝ) : Prop := t₁ + t₂ = 8.5
def stone_fall (t₁ : ℝ) : Prop := d = 16 * t₁^2 
def sound_travel (t₂ : ℝ) : Prop := t₂ = d / 1100

theorem well_depth : 
  ∃ t₁ t₂ : ℝ, total_time t₁ t₂ ∧ stone_fall d t₁ ∧ sound_travel d t₂ → d = 918.09 := 
by
  sorry

end well_depth_l311_311003


namespace prob_neither_defective_l311_311047

-- Definitions for the conditions
def totalPens : ℕ := 8
def defectivePens : ℕ := 2
def nonDefectivePens : ℕ := totalPens - defectivePens
def selectedPens : ℕ := 2

-- Theorem statement for the probability that neither of the two selected pens is defective
theorem prob_neither_defective : 
  (nonDefectivePens / totalPens) * ((nonDefectivePens - 1) / (totalPens - 1)) = 15 / 28 := 
  sorry

end prob_neither_defective_l311_311047


namespace jane_chickens_l311_311391

-- Conditions
def eggs_per_chicken_per_week : ℕ := 6
def egg_price_per_dozen : ℕ := 2
def total_income_in_2_weeks : ℕ := 20

-- Mathematical problem
theorem jane_chickens : (total_income_in_2_weeks / egg_price_per_dozen) * 12 / (eggs_per_chicken_per_week * 2) = 10 :=
by
  sorry

end jane_chickens_l311_311391


namespace temp_product_l311_311145

theorem temp_product (N : ℤ) (M D : ℤ)
  (h1 : M = D + N)
  (h2 : M - 8 = D + N - 8)
  (h3 : D + 5 = D + 5)
  (h4 : abs ((D + N - 8) - (D + 5)) = 3) :
  (N = 16 ∨ N = 10) →
  16 * 10 = 160 := 
by sorry

end temp_product_l311_311145


namespace sum_transformed_roots_l311_311559

theorem sum_transformed_roots :
  ∀ (a b c : ℝ),
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  (45 * a^3 - 75 * a^2 + 33 * a - 2 = 0) →
  (45 * b^3 - 75 * b^2 + 33 * b - 2 = 0) →
  (45 * c^3 - 75 * c^2 + 33 * c - 2 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by
  intros a b c h_bounds h_poly_a h_poly_b h_poly_c h_distinct
  sorry

end sum_transformed_roots_l311_311559


namespace sides_of_regular_polygon_with_20_diagonals_l311_311997

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l311_311997


namespace polygon_sides_from_diagonals_l311_311992

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l311_311992


namespace find_number_of_hens_l311_311581

theorem find_number_of_hens
  (H C : ℕ)
  (h1 : H + C = 48)
  (h2 : 2 * H + 4 * C = 140) :
  H = 26 :=
by
  sorry

end find_number_of_hens_l311_311581


namespace cost_of_perfume_l311_311321

-- Definitions and Constants
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def neighbors_yards_mowed : ℕ := 4
def charge_per_yard : ℕ := 5
def dogs_walked : ℕ := 6
def charge_per_dog : ℕ := 2
def additional_amount_needed : ℕ := 6

-- Theorem Statement
theorem cost_of_perfume :
  let christian_earnings := neighbors_yards_mowed * charge_per_yard
  let sue_earnings := dogs_walked * charge_per_dog
  let christian_savings := christian_initial_savings + christian_earnings
  let sue_savings := sue_initial_savings + sue_earnings
  let total_savings := christian_savings + sue_savings
  total_savings + additional_amount_needed = 50 := 
by
  sorry

end cost_of_perfume_l311_311321


namespace odd_factors_of_360_l311_311656

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l311_311656


namespace find_a_plus_b_l311_311345

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 3 * a + b

theorem find_a_plus_b (a b : ℝ) (h1 : ∀ x : ℝ, f a b x = f a b (-x)) (h2 : 2 * a = 3 - a) : a + b = 1 :=
by
  unfold f at h1
  sorry

end find_a_plus_b_l311_311345


namespace f_is_monotonic_l311_311927

variable (f : ℝ → ℝ)

theorem f_is_monotonic (h : ∀ a b x : ℝ, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  (∀ x y : ℝ, x ≤ y → f x <= f y) ∨ (∀ x y : ℝ, x ≤ y → f x >= f y) :=
sorry

end f_is_monotonic_l311_311927


namespace find_s2_side_length_l311_311710

-- Define the variables involved
variables (r s : ℕ)

-- Conditions based on problem statement
def height_eq : Prop := 2 * r + s = 2160
def width_eq : Prop := 2 * r + 3 * s + 110 = 4020

-- The theorem stating that s = 875 given the conditions
theorem find_s2_side_length (h1 : height_eq r s) (h2 : width_eq r s) : s = 875 :=
by {
  sorry
}

end find_s2_side_length_l311_311710


namespace false_converse_implication_l311_311141

theorem false_converse_implication : ∃ x : ℝ, (0 < x) ∧ (x - 3 ≤ 0) := by
  sorry

end false_converse_implication_l311_311141


namespace regular_polygon_sides_l311_311973

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l311_311973


namespace sum_f_positive_l311_311642

variable (a b c : ℝ)

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (h1 : a + b > 0) (h2 : a + c > 0) (h3 : b + c > 0) :
  f a + f b + f c > 0 :=
sorry

end sum_f_positive_l311_311642


namespace george_speed_l311_311159

theorem george_speed : 
  ∀ (d_tot d_1st : ℝ) (v_tot v_1st : ℝ) (v_2nd : ℝ),
    d_tot = 1 ∧ d_1st = 1 / 2 ∧ v_tot = 3 ∧ v_1st = 2 ∧ ((d_tot / v_tot) = (d_1st / v_1st + d_1st / v_2nd)) →
    v_2nd = 6 :=
by
  -- Proof here
  sorry

end george_speed_l311_311159


namespace scarlett_oil_amount_l311_311255

theorem scarlett_oil_amount (initial_oil add_oil : ℝ) (h1 : initial_oil = 0.17) (h2 : add_oil = 0.67) :
  initial_oil + add_oil = 0.84 :=
by
  rw [h1, h2]
  -- Proof step goes here
  sorry

end scarlett_oil_amount_l311_311255


namespace selling_price_per_unit_profit_per_unit_after_discount_l311_311120

-- Define the initial cost per unit
variable (a : ℝ)

-- Problem statement for part 1: Selling price per unit is 1.22a yuan
theorem selling_price_per_unit (a : ℝ) : 1.22 * a = a + 0.22 * a :=
by
  sorry

-- Problem statement for part 2: Profit per unit after 15% discount is still 0.037a yuan
theorem profit_per_unit_after_discount (a : ℝ) : 
  (1.22 * a * 0.85) - a = 0.037 * a :=
by
  sorry

end selling_price_per_unit_profit_per_unit_after_discount_l311_311120


namespace opposite_vertices_equal_l311_311830

-- Define the angles of a regular convex hexagon
variables {α β γ δ ε ζ : ℝ}

-- Regular hexagon condition: The sum of the alternating angles
axiom angle_sum_condition :
  α + γ + ε = β + δ + ε

-- Define the final theorem to prove that the opposite vertices have equal angles
theorem opposite_vertices_equal (h : α + γ + ε = β + δ + ε) :
  α = δ ∧ β = ε ∧ γ = ζ :=
sorry

end opposite_vertices_equal_l311_311830


namespace find_a_l311_311637

open Real

noncomputable def valid_solutions (a b : ℝ) : Prop :=
  a + 2 / b = 17 ∧ b + 2 / a = 1 / 3

theorem find_a (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : valid_solutions a b) :
  a = 6 ∨ a = 17 :=
by sorry

end find_a_l311_311637


namespace part1_part2_l311_311589

-- Definitions and conditions
def a : ℕ := 60
def b : ℕ := 40
def c : ℕ := 80
def d : ℕ := 20
def n : ℕ := a + b + c + d

-- Given critical value for 99% certainty
def critical_value_99 : ℝ := 6.635

-- Calculate K^2 using the given formula
noncomputable def K_squared : ℝ := (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Calculation of probability of selecting 2 qualified products from 5 before renovation
def total_sampled : ℕ := 5
def qualified_before_renovation : ℕ := 3
def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def prob_selecting_2_qualified : ℚ := (total_combinations qualified_before_renovation 2 : ℚ) / 
                                      (total_combinations total_sampled 2 : ℚ)

-- Proof statements
theorem part1 : K_squared > critical_value_99 := by
  sorry

theorem part2 : prob_selecting_2_qualified = 3 / 10 := by
  sorry

end part1_part2_l311_311589


namespace henri_drove_more_miles_l311_311787

-- Defining the conditions
def Gervais_average_miles_per_day := 315
def Gervais_days_driven := 3
def Henri_total_miles := 1250

-- Total miles driven by Gervais
def Gervais_total_miles := Gervais_average_miles_per_day * Gervais_days_driven

-- The proof problem statement
theorem henri_drove_more_miles : Henri_total_miles - Gervais_total_miles = 305 := 
by 
  sorry

end henri_drove_more_miles_l311_311787


namespace abc_over_ab_bc_ca_l311_311726

variable {a b c : ℝ}

theorem abc_over_ab_bc_ca (h1 : ab / (a + b) = 2)
                          (h2 : bc / (b + c) = 5)
                          (h3 : ca / (c + a) = 7) :
        abc / (ab + bc + ca) = 35 / 44 :=
by
  -- The proof would go here.
  sorry

end abc_over_ab_bc_ca_l311_311726


namespace driver_weekly_distance_l311_311123

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end driver_weekly_distance_l311_311123


namespace lattice_points_on_hyperbola_l311_311809

theorem lattice_points_on_hyperbola : 
  let hyperbola_eq := λ x y : ℤ, x^2 - y^2 = 1800^2 in
  (∃ (x y : ℤ), hyperbola_eq x y) ∧ 
  ∃ (n : ℕ), n = 54 :=
by
  sorry

end lattice_points_on_hyperbola_l311_311809


namespace parabola_expression_l311_311797

open Real

-- Given the conditions of the parabola obtaining points A and B
def parabola (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x - 5

-- Defining the points A and B where parabola intersects the x-axis
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (5, 0)

-- The proof statement we need to show
theorem parabola_expression (a b : ℝ) (hxA : parabola a b A.fst = A.snd) (hxB : parabola a b B.fst = B.snd) : 
  ∀ x : ℝ, parabola a b x = x^2 - 4 * x - 5 :=
sorry

end parabola_expression_l311_311797


namespace probability_divisor_of_8_is_half_l311_311888

theorem probability_divisor_of_8_is_half :
  let outcomes := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let divisors_of_8 := [ 1, 2, 4, 8 ]
  let favorable_outcomes := list.filter (λ x, x ∣ 8) outcomes
  let favorable_probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  favorable_probability = (1 / 2 : ℚ) := by
  sorry

end probability_divisor_of_8_is_half_l311_311888


namespace fraction_power_l311_311467

theorem fraction_power (a b : ℕ) (ha : a = 5) (hb : b = 6) : (a / b : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end fraction_power_l311_311467


namespace pages_difference_l311_311752

def second_chapter_pages : ℕ := 18
def third_chapter_pages : ℕ := 3

theorem pages_difference : second_chapter_pages - third_chapter_pages = 15 := by 
  sorry

end pages_difference_l311_311752


namespace polar_to_rectangular_coordinates_l311_311905

noncomputable def rectangular_coordinates_from_polar (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_coordinates :
  rectangular_coordinates_from_polar 12 (5 * Real.pi / 4) = (-6 * Real.sqrt 2, -6 * Real.sqrt 2) :=
  sorry

end polar_to_rectangular_coordinates_l311_311905


namespace second_rooster_weight_l311_311619

theorem second_rooster_weight (cost_per_kg : ℝ) (weight_1 : ℝ) (total_earnings : ℝ) (weight_2 : ℝ) :
  cost_per_kg = 0.5 →
  weight_1 = 30 →
  total_earnings = 35 →
  total_earnings = weight_1 * cost_per_kg + weight_2 * cost_per_kg →
  weight_2 = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end second_rooster_weight_l311_311619


namespace lowest_score_jack_l311_311553

noncomputable def lowest_possible_score (mean : ℝ) (std_dev : ℝ) := 
  max ((1.28 * std_dev) + mean) (mean + 2 * std_dev)

theorem lowest_score_jack (mean : ℝ := 60) (std_dev : ℝ := 10) :
  lowest_possible_score mean std_dev = 73 := 
by
  -- We need to show that the minimum score Jack could get is 73 based on problem conditions
  sorry

end lowest_score_jack_l311_311553


namespace fraction_equality_l311_311319

theorem fraction_equality :
  (2 - (1 / 2) * (1 - (1 / 4))) / (2 - (1 - (1 / 3))) = 39 / 32 := 
  sorry

end fraction_equality_l311_311319


namespace asymptote_of_hyperbola_l311_311422

theorem asymptote_of_hyperbola (h : (∀ x y : ℝ, y^2 / 3 - x^2 / 2 = 1)) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (sqrt6 / 2) * x ∨ y = - (sqrt6 / 2) * x) :=
sorry

end asymptote_of_hyperbola_l311_311422


namespace differences_l311_311027

def seq (n : ℕ) : ℕ := n^2 + 1

def first_diff (n : ℕ) : ℕ := (seq (n + 1)) - (seq n)

def second_diff (n : ℕ) : ℕ := (first_diff (n + 1)) - (first_diff n)

def third_diff (n : ℕ) : ℕ := (second_diff (n + 1)) - (second_diff n)

theorem differences (n : ℕ) : first_diff n = 2 * n + 1 ∧ 
                             second_diff n = 2 ∧ 
                             third_diff n = 0 := by 
  sorry

end differences_l311_311027


namespace sum_of_youngest_and_oldest_cousins_l311_311006

theorem sum_of_youngest_and_oldest_cousins :
  ∃ (ages : Fin 5 → ℝ), (∃ (a1 a5 : ℝ), ages 0 = a1 ∧ ages 4 = a5 ∧ a1 + a5 = 29) ∧
                        (∃ (median : ℝ), median = ages 2 ∧ median = 7) ∧
                        (∃ (mean : ℝ), mean = (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 ∧ mean = 10) :=
by sorry

end sum_of_youngest_and_oldest_cousins_l311_311006


namespace percentage_of_primes_divisible_by_3_l311_311275

-- Define prime numbers less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is divisible by 3
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Count the number of prime numbers less than 20 that are divisible by 3
def count_divisibles_by_3 : ℕ :=
  primes_less_than_20.countp is_divisible_by_3

-- Total prime numbers less than 20
def total_primes : ℕ := primes_less_than_20.length

-- Calculate the percentage of prime numbers less than 20 that are divisible by 3
def percentage_divisibles_by_3 : ℚ := 
  (count_divisibles_by_3.to_rat / total_primes.to_rat) * 100

-- The theorem we need to prove
theorem percentage_of_primes_divisible_by_3 : percentage_divisibles_by_3 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_3_l311_311275


namespace count_triangles_with_positive_area_l311_311676

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l311_311676


namespace count_lattice_points_on_hyperbola_l311_311810

theorem count_lattice_points_on_hyperbola : 
  (∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S ↔ (p.1 ^ 2 - p.2 ^ 2 = 1800 ^ 2)) ∧ S.card = 150) :=
sorry

end count_lattice_points_on_hyperbola_l311_311810


namespace fourth_root_expression_l311_311144

-- Define a positive real number y
variable (y : ℝ) (hy : 0 < y)

-- State the problem in Lean
theorem fourth_root_expression : 
  Real.sqrt (Real.sqrt (y^2 * Real.sqrt y)) = y^(5/8) := sorry

end fourth_root_expression_l311_311144


namespace time_per_page_l311_311237

theorem time_per_page 
    (planning_time : ℝ := 3) 
    (fraction : ℝ := 3/4) 
    (pages_read : ℕ := 9) 
    (minutes_per_hour : ℕ := 60) : 
    (fraction * planning_time * minutes_per_hour) / pages_read = 15 := 
by
  sorry

end time_per_page_l311_311237


namespace geometric_sequence_common_ratio_l311_311356

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
  (h : ∀ n, a n * a (n + 1) = 16^n) :
  ∃ r : ℝ, r = 4 ∧ ∀ n, a (n + 1) = a n * r :=
sorry

end geometric_sequence_common_ratio_l311_311356


namespace rahul_work_days_l311_311541

variable (R : ℕ)

theorem rahul_work_days
  (rajesh_days : ℕ := 2)
  (total_money : ℕ := 355)
  (rahul_share : ℕ := 142)
  (rajesh_share : ℕ := total_money - rahul_share)
  (payment_ratio : ℕ := rahul_share / rajesh_share)
  (work_rate_ratio : ℕ := rajesh_days / R) :
  payment_ratio = work_rate_ratio → R = 3 :=
by
  sorry

end rahul_work_days_l311_311541


namespace smallest_number_to_add_for_divisibility_l311_311870

theorem smallest_number_to_add_for_divisibility :
  ∃ x : ℕ, 1275890 + x ≡ 0 [MOD 2375] ∧ x = 1360 :=
by sorry

end smallest_number_to_add_for_divisibility_l311_311870


namespace compare_neg_sqrt_l311_311611

theorem compare_neg_sqrt :
  -5 > -Real.sqrt 26 := 
sorry

end compare_neg_sqrt_l311_311611


namespace percentage_increase_B_over_C_l311_311856

noncomputable def A_m : ℕ := 537600 / 12
noncomputable def C_m : ℕ := 16000
noncomputable def ratio : ℚ := 5 / 2

noncomputable def B_m (A_m : ℕ) : ℚ := (2 * A_m) / 5

theorem percentage_increase_B_over_C :
  B_m A_m = 17920 →
  C_m = 16000 →
  (B_m A_m - C_m) / C_m * 100 = 12 :=
by
  sorry

end percentage_increase_B_over_C_l311_311856


namespace possible_values_of_reciprocal_sum_l311_311525

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  ∃ x, x ∈ set.Ici (2:ℝ) ∧ x = (1 / a + 1 / b) :=
by sorry

end possible_values_of_reciprocal_sum_l311_311525


namespace parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l311_311498

-- Definitions for the problem conditions
def parabola_symmetry_axis := "coordinate axis"
def parabola_vertex := (0, 0)
def directrix_equation := "x = -1"
def intersects_at_two_points (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) := (l P.1 = P.2) ∧ (l Q.1 = Q.2) ∧ (P ≠ Q)

-- Main theorem statements
theorem parabola_standard_equation : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") → 
  ∃ p, 0 < p ∧ ∀ y x, y^2 = 4 * p * x := 
  sorry

theorem oa_dot_ob_value (l : ℝ → ℝ) (focus : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  l focus.1 = focus.2 → 
  (P.1 * Q.1 + P.2 * Q.2 = -3) := 
  sorry

theorem line_passes_fixed_point (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = -4) → 
  ∃ fp, fp = (2,0) := 
  sorry

end parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l311_311498


namespace bill_profit_difference_l311_311776

theorem bill_profit_difference (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P)
  (h2 : SP = 659.9999999999994)
  (h3 : NP = 0.90 * P)
  (h4 : NSP = 1.30 * NP) :
  NSP - SP = 42 := 
sorry

end bill_profit_difference_l311_311776


namespace m_plus_n_sum_l311_311618

theorem m_plus_n_sum :
  let m := 271
  let n := 273
  m + n = 544 :=
by {
  -- sorry included to skip the proof steps
  sorry
}

end m_plus_n_sum_l311_311618


namespace sides_of_regular_polygon_with_20_diagonals_l311_311995

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l311_311995


namespace compute_expression_l311_311353

theorem compute_expression (x y : ℝ) (hx : 1/x + 1/y = 4) (hy : x*y + x + y = 5) : 
  x^2 * y + x * y^2 + x^2 + y^2 = 18 := 
by 
  -- Proof goes here 
  sorry

end compute_expression_l311_311353


namespace binary_addition_to_decimal_l311_311575

theorem binary_addition_to_decimal : (0b111111111 + 0b1000001 = 576) :=
by {
  sorry
}

end binary_addition_to_decimal_l311_311575


namespace range_of_m_l311_311626

theorem range_of_m (m : ℝ) : 
  ((∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ (m < -1)) :=
sorry

end range_of_m_l311_311626


namespace primes_less_than_20_divisible_by_3_percentage_l311_311280

theorem primes_less_than_20_divisible_by_3_percentage :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let divisible_by_3 := primes.filter (λ p, p % 3 = 0)
  (divisible_by_3.length / primes.length : ℝ) * 100 = 12.5 := by
sorry

end primes_less_than_20_divisible_by_3_percentage_l311_311280


namespace sin_minus_cos_eq_neg_one_l311_311204

theorem sin_minus_cos_eq_neg_one (x : ℝ) 
    (h1 : sin x ^ 3 - cos x ^ 3 = -1)
    (h2 : sin x ^ 2 + cos x ^ 2 = 1) : 
    sin x - cos x = -1 :=
sorry

end sin_minus_cos_eq_neg_one_l311_311204


namespace score_entered_twice_l311_311409

theorem score_entered_twice (scores : List ℕ) (h : scores = [68, 74, 77, 82, 85, 90]) :
  ∃ (s : ℕ), s = 82 ∧ ∀ (entered : List ℕ), entered.length = 7 ∧ (∀ i, (List.take (i + 1) entered).sum % (i + 1) = 0) →
  (List.count (List.insertNth i 82 scores)) = 2 ∧ (∀ x, x ∈ scores.remove 82 → x ≠ s) :=
by
  sorry

end score_entered_twice_l311_311409


namespace calculate_f_f_neg3_l311_311802

def f (x : ℚ) : ℚ := (1 / x) + (1 / (x + 1))

theorem calculate_f_f_neg3 : f (f (-3)) = 24 / 5 := by
  sorry

end calculate_f_f_neg3_l311_311802


namespace sequence_periodic_a_n_plus_2_eq_a_n_l311_311295

-- Definition of the sequence and conditions
noncomputable def seq (a : ℕ → ℤ) :=
  ∀ n : ℕ, ∃ α k : ℕ, a n = Int.ofNat (2^α) * k ∧ Int.gcd (Int.ofNat k) 2 = 1 ∧ a (n+1) = Int.ofNat (2^α) - k

-- Definition of periodic sequence
def periodic (a : ℕ → ℤ) (d : ℕ) :=
  ∀ n : ℕ, a (n + d) = a n

-- Proving the desired property
theorem sequence_periodic_a_n_plus_2_eq_a_n (a : ℕ → ℤ) (d : ℕ) (h_seq : seq a) (h_periodic : periodic a d) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end sequence_periodic_a_n_plus_2_eq_a_n_l311_311295


namespace dan_money_left_l311_311019

def money_left (initial_amount spent_on_candy spent_on_gum : ℝ) : ℝ :=
  initial_amount - (spent_on_candy + spent_on_gum)

theorem dan_money_left :
  money_left 3.75 1.25 0.80 = 1.70 :=
by
  sorry

end dan_money_left_l311_311019


namespace students_in_cafeteria_after_moves_l311_311863

theorem students_in_cafeteria_after_moves :
  ∀ (total_students cafeterial_fraction outside_fraction one_third_outside moved_to_outside total_outside moved_to_inside),
  total_students = 90 →
  cafeterial_fraction = 2 / 3 →
  outside_fraction = 1 / 3 →
  moved_to_outside = 3 →
  (total_outside = total_students - (total_students * cafeterial_fraction)) →
  moved_to_inside = total_outside * outside_fraction →
  let initial_in_cafeteria := total_students * cafeterial_fraction in
  let final_in_cafeteria := initial_in_cafeteria + moved_to_inside - moved_to_outside in
  final_in_cafeteria = 67 :=
begin
  intros total_students cafeterial_fraction outside_fraction one_third_outside moved_to_outside total_outside moved_to_inside,
  intros h1 h2 h3 h4 h5 h6,
  let initial_in_cafeteria := total_students * cafeterial_fraction,
  let final_in_cafeteria := initial_in_cafeteria + moved_to_inside - moved_to_outside,
  sorry
end

end students_in_cafeteria_after_moves_l311_311863


namespace jessie_interest_l311_311392

noncomputable def compoundInterest 
  (P : ℝ) -- Principal
  (r : ℝ) -- annual interest rate
  (n : ℕ) -- number of times interest applied per time period
  (t : ℝ) -- time periods elapsed
  : ℝ :=
  P * (1 + r / n)^(n * t)

theorem jessie_interest :
  let P := 1200
  let annual_rate := 0.08
  let periods_per_year := 2
  let years := 5
  let A := compoundInterest P annual_rate periods_per_year years
  let interest := A - P
  interest = 576.29 :=
by
  sorry

end jessie_interest_l311_311392


namespace find_natural_numbers_l311_311026

theorem find_natural_numbers (n : ℕ) :
  (∀ k : ℕ, k^2 + ⌊ (n : ℝ) / (k^2 : ℝ) ⌋ ≥ 1991) ∧
  (∃ k_0 : ℕ, k_0^2 + ⌊ (n : ℝ) / (k_0^2 : ℝ) ⌋ < 1992) ↔
  990208 ≤ n ∧ n ≤ 991231 :=
by sorry

end find_natural_numbers_l311_311026


namespace func_equiv_l311_311930

noncomputable def f (x : ℝ) : ℝ := if x = 0 then 0 else x + 1 / x

theorem func_equiv {a b : ℝ} (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) :
  (∀ x, f (2 * x) = a * f x + b * x) ∧ (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y)) :=
sorry

end func_equiv_l311_311930


namespace infinite_arithmetic_progression_intersects_segments_l311_311463

open Classical

-- Define segments and their properties
structure Segment :=
(start end : ℝ)
(length : ℝ := end - start)
(non_overlapping : ∀ (s1 s2 : Segment), s1 ≠ s2 → (s1.end ≤ s2.start ∨ s2.end ≤ s1.start))

-- Define the arithmetic progression
def arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem infinite_arithmetic_progression_intersects_segments:
  ∀ (a d : ℝ) (S : set Segment), 
    (∀ s ∈ S, s.length = 1) →
    (∀ s1 s2 ∈ S, s1 ≠ s2 → (s1.end ≤ s2.start ∨ s2.end ≤ s1.start)) →
    ∃ (s : Segment) (n : ℕ), s ∈ S ∧ (arithmetic_progression a d n ∈ set.Icc s.start s.end) :=
by
  sorry

end infinite_arithmetic_progression_intersects_segments_l311_311463


namespace seq_geom_prog_l311_311698

theorem seq_geom_prog (a : ℕ → ℝ) (b : ℝ) (h_pos_b : 0 < b)
  (h_pos_a : ∀ n, 0 < a n)
  (h_recurrence : ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)) :
  (∃ r, ∀ n, a (n + 1) = r * a n) ↔ a 0 = a 1 :=
sorry

end seq_geom_prog_l311_311698


namespace number_of_positive_area_triangles_l311_311668

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l311_311668


namespace non_congruent_triangles_count_l311_311964

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l311_311964


namespace pie_filling_cans_l311_311131

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end pie_filling_cans_l311_311131


namespace maxim_is_correct_l311_311294

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end maxim_is_correct_l311_311294


namespace height_of_sky_island_l311_311388

theorem height_of_sky_island (day_climb : ℕ) (night_slide : ℕ) (days : ℕ) (final_day_climb : ℕ) :
  day_climb = 25 →
  night_slide = 3 →
  days = 64 →
  final_day_climb = 25 →
  (days - 1) * (day_climb - night_slide) + final_day_climb = 1411 :=
by
  -- Add the formal proof here
  sorry

end height_of_sky_island_l311_311388


namespace longer_piece_length_l311_311773

theorem longer_piece_length (x : ℝ) (h1 : x + (x + 2) = 30) : x + 2 = 16 :=
by sorry

end longer_piece_length_l311_311773


namespace range_of_x_l311_311679

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) : x > 1/3 :=
by
  sorry

end range_of_x_l311_311679


namespace length_of_field_l311_311093

theorem length_of_field (width : ℕ) (distance_covered : ℕ) (n : ℕ) (L : ℕ) 
  (h1 : width = 15) 
  (h2 : distance_covered = 540) 
  (h3 : n = 3) 
  (h4 : 2 * (L + width) = perimeter)
  (h5 : n * perimeter = distance_covered) : 
  L = 75 :=
by 
  sorry

end length_of_field_l311_311093


namespace find_k_value_l311_311924

theorem find_k_value (k : ℚ) :
  (∀ x y : ℚ, (x = 1/3 ∧ y = -8 → -3/4 - 3 * k * x = 7 * y)) → k = 55.25 :=
by
  sorry

end find_k_value_l311_311924


namespace nearest_integer_to_x_minus_y_l311_311819

theorem nearest_integer_to_x_minus_y
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : abs x + y = 3) (h2 : abs x * y + x^3 = 0) :
  Int.nearest (x - y) = -3 := 
sorry

end nearest_integer_to_x_minus_y_l311_311819


namespace discount_is_20_percent_l311_311400

noncomputable def discount_percentage 
  (puppy_cost : ℝ := 20.0)
  (dog_food_cost : ℝ := 20.0)
  (treat_cost : ℝ := 2.5)
  (num_treats : ℕ := 2)
  (toy_cost : ℝ := 15.0)
  (crate_cost : ℝ := 20.0)
  (bed_cost : ℝ := 20.0)
  (collar_leash_cost : ℝ := 15.0)
  (total_spent : ℝ := 96.0) : ℝ := 
  let total_cost_before_discount := dog_food_cost + (num_treats * treat_cost) + toy_cost + crate_cost + bed_cost + collar_leash_cost
  let spend_at_store := total_spent - puppy_cost
  let discount_amount := total_cost_before_discount - spend_at_store
  (discount_amount / total_cost_before_discount) * 100

theorem discount_is_20_percent : discount_percentage = 20 := sorry

end discount_is_20_percent_l311_311400


namespace temperature_representation_l311_311369

def represents_zero_degrees_celsius (t₁ : ℝ) : Prop := t₁ = 10

theorem temperature_representation (t₁ t₂ : ℝ) (h₀ : represents_zero_degrees_celsius t₁) 
    (h₁ : t₂ > t₁):
    t₂ = 17 :=
by
  -- Proof is omitted here
  sorry

end temperature_representation_l311_311369


namespace butterflies_in_the_garden_l311_311483

variable (total_butterflies : Nat) (fly_away : Nat)

def butterflies_left (total_butterflies : Nat) (fly_away : Nat) : Nat :=
  total_butterflies - fly_away

theorem butterflies_in_the_garden :
  (total_butterflies = 9) → (fly_away = 1 / 3 * total_butterflies) → butterflies_left total_butterflies fly_away = 6 :=
by
  intro h1 h2
  sorry

end butterflies_in_the_garden_l311_311483


namespace rotate_parabola_180_l311_311711

theorem rotate_parabola_180 (x: ℝ) : 
  let original_parabola := λ x, 2 * (x - 3)^2 - 2,
      rotated_parabola := λ x, -2 * (x - 3)^2 - 2 in
  original_parabola x = rotated_parabola x :=
sorry

end rotate_parabola_180_l311_311711


namespace ratio_of_a_to_b_l311_311513

theorem ratio_of_a_to_b 
  (b c d a : ℚ)
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := 
by sorry

end ratio_of_a_to_b_l311_311513


namespace compute_expression_l311_311233

theorem compute_expression (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 6) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end compute_expression_l311_311233


namespace rowing_time_75_minutes_l311_311329

-- Definition of time duration Ethan rowed.
def EthanRowingTime : ℕ := 25  -- minutes

-- Definition of the time duration Frank rowed.
def FrankRowingTime : ℕ := 2 * EthanRowingTime  -- twice as long as Ethan.

-- Definition of the total rowing time.
def TotalRowingTime : ℕ := EthanRowingTime + FrankRowingTime

-- Theorem statement proving the total rowing time is 75 minutes.
theorem rowing_time_75_minutes : TotalRowingTime = 75 := by
  -- The proof is omitted.
  sorry

end rowing_time_75_minutes_l311_311329


namespace different_suits_card_combinations_l311_311201

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l311_311201


namespace Henry_trays_per_trip_l311_311938

theorem Henry_trays_per_trip (trays1 trays2 trips : ℕ) (h1 : trays1 = 29) (h2 : trays2 = 52) (h3 : trips = 9) :
  (trays1 + trays2) / trips = 9 :=
by
  sorry

end Henry_trays_per_trip_l311_311938


namespace sandy_initial_fish_l311_311238

theorem sandy_initial_fish (bought_fish : ℕ) (total_fish : ℕ) (h1 : bought_fish = 6) (h2 : total_fish = 32) :
  total_fish - bought_fish = 26 :=
by
  sorry

end sandy_initial_fish_l311_311238


namespace final_lives_equals_20_l311_311449

def initial_lives : ℕ := 30
def lives_lost : ℕ := 12
def bonus_lives : ℕ := 5
def penalty_lives : ℕ := 3

theorem final_lives_equals_20 : (initial_lives - lives_lost + bonus_lives - penalty_lives) = 20 :=
by 
  sorry

end final_lives_equals_20_l311_311449


namespace probability_exists_x0_l311_311697

-- Define the conditions as given
noncomputable def h (n : ℕ) (θ : fin n → ℝ) (x : ℝ) : ℝ :=
  (1 / n : ℝ) * ((finset.univ.filter (λ k, θ k < x)).card : ℝ)

-- Define the main Theorem
theorem probability_exists_x0 (n : ℕ) (θ : fin n → ℝ) 
  (h_uniform : ∀ i, 0 <= θ i ∧ θ i <= 1) 
  (h_independent : ∀ i j, i ≠ j → θ i ≠ θ j)
  : prob (∃ x0 ∈ (0, 1), h n θ x0 = x0) = 1 - (1 / n : ℝ) := 
sorry

end probability_exists_x0_l311_311697


namespace probability_X1_lt_X2_lt_X3_is_1_6_l311_311061

noncomputable def probability_X1_lt_X2_lt_X3 (n : ℕ) (h : n ≥ 3) : ℚ :=
if h : n ≥ 3 then
  1/6
else
  0

theorem probability_X1_lt_X2_lt_X3_is_1_6 (n : ℕ) (h : n ≥ 3) :
  probability_X1_lt_X2_lt_X3 n h = 1/6 :=
sorry

end probability_X1_lt_X2_lt_X3_is_1_6_l311_311061


namespace problem_statement_l311_311633

def sequence_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n+1) / 2^(n+1) - a n / 2^n = 1)

theorem problem_statement : 
  ∃ a : ℕ → ℝ, a 1 = 2 ∧ a 2 = 8 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2^(n+1)) → sequence_arithmetic a :=
by
  sorry

end problem_statement_l311_311633


namespace car_distance_in_45_minutes_l311_311506

theorem car_distance_in_45_minutes
  (train_speed : ℝ)
  (car_speed_ratio : ℝ)
  (time_minutes : ℝ)
  (h_train_speed : train_speed = 90)
  (h_car_speed_ratio : car_speed_ratio = 5 / 6)
  (h_time_minutes : time_minutes = 45) :
  ∃ d : ℝ, d = 56.25 ∧ d = (car_speed_ratio * train_speed) * (time_minutes / 60) :=
by
  sorry

end car_distance_in_45_minutes_l311_311506


namespace lattice_points_on_hyperbola_l311_311817

theorem lattice_points_on_hyperbola :
  ∃ (n : ℕ), n = 90 ∧
  (∀ (x y : ℤ), x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | true} ) :=
begin
  -- Convert mathematical conditions to Lean definitions
  let a := 1800^2,
  have even_factors : (∀ (x y : ℤ), (x - y) * (x + y) = a → even (x - y) ∧ even (x+y)),
  {
    sorry,
  },
  -- Assert the number of lattice points is 90
  use [90],
  split; simp,
  sorry,
end

end lattice_points_on_hyperbola_l311_311817


namespace train_speed_approx_l311_311588

noncomputable def man_speed_kmh : ℝ := 3
noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600
noncomputable def train_length : ℝ := 900
noncomputable def time_to_cross : ℝ := 53.99568034557235
noncomputable def train_speed_ms := (train_length / time_to_cross) + man_speed_ms
noncomputable def train_speed_kmh := (train_speed_ms * 3600) / 1000

theorem train_speed_approx :
  abs (train_speed_kmh - 63.009972) < 1e-5 := sorry

end train_speed_approx_l311_311588


namespace remainder_when_divided_by_11_l311_311878

theorem remainder_when_divided_by_11 {k x : ℕ} (h : x = 66 * k + 14) : x % 11 = 3 :=
by
  sorry

end remainder_when_divided_by_11_l311_311878


namespace probability_multiple_choice_and_essay_correct_l311_311890

noncomputable def probability_multiple_choice_and_essay (C : ℕ → ℕ → ℕ) : ℚ :=
    (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3)

theorem probability_multiple_choice_and_essay_correct (C : ℕ → ℕ → ℕ) :
    probability_multiple_choice_and_essay C = (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3) :=
by
  sorry

end probability_multiple_choice_and_essay_correct_l311_311890


namespace original_rice_amount_l311_311520

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end original_rice_amount_l311_311520


namespace fraction_equality_l311_311324

theorem fraction_equality : (18 / (5 * 107 + 3) = 18 / 538) := 
by
  -- Proof skipped
  sorry

end fraction_equality_l311_311324


namespace polygon_diagonals_l311_311985

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l311_311985


namespace no_12_term_geometric_seq_in_1_to_100_l311_311221

theorem no_12_term_geometric_seq_in_1_to_100 :
  ¬ ∃ (s : Fin 12 → Set ℕ),
    (∀ i, ∃ (a q : ℕ), (s i = {a * q^n | n : ℕ}) ∧ (∀ x ∈ s i, 1 ≤ x ∧ x ≤ 100)) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ i, n ∈ s i) := 
sorry

end no_12_term_geometric_seq_in_1_to_100_l311_311221


namespace numberOfWaysToChoose4Cards_l311_311180

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l311_311180


namespace alpha_eq_two_thirds_l311_311226

theorem alpha_eq_two_thirds (α : ℚ) (h1 : 0 < α) (h2 : α < 1) (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : α = 2 / 3 :=
sorry

end alpha_eq_two_thirds_l311_311226


namespace frank_can_buy_seven_candies_l311_311450

def tickets_won_whackamole := 33
def tickets_won_skeeball := 9
def cost_per_candy := 6

theorem frank_can_buy_seven_candies : (tickets_won_whackamole + tickets_won_skeeball) / cost_per_candy = 7 :=
by
  sorry

end frank_can_buy_seven_candies_l311_311450


namespace four_digit_integer_5533_l311_311851

theorem four_digit_integer_5533
  (a b c d : ℕ)
  (h1 : a + b + c + d = 16)
  (h2 : b + c = 8)
  (h3 : a - d = 2)
  (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  1000 * a + 100 * b + 10 * c + d = 5533 :=
by {
  sorry
}

end four_digit_integer_5533_l311_311851


namespace compute_4_star_3_l311_311371

def custom_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem compute_4_star_3 : custom_op 4 3 = 13 :=
by
  sorry

end compute_4_star_3_l311_311371


namespace infinite_unlucky_numbers_l311_311597

def is_unlucky (n : ℕ) : Prop :=
  ¬(∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (n = x^2 - 1 ∨ n = y^2 - 1))

theorem infinite_unlucky_numbers : ∀ᶠ n in at_top, is_unlucky n := sorry

end infinite_unlucky_numbers_l311_311597


namespace factorization_correct_l311_311486

theorem factorization_correct (x : ℝ) : 
  (x^2 + 5 * x + 2) * (x^2 + 5 * x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5 * x - 1) :=
by
  sorry

end factorization_correct_l311_311486


namespace find_k_range_l311_311503

open Nat

def a_n (n : ℕ) : ℕ := 2^ (5 - n)

def b_n (n : ℕ) (k : ℤ) : ℤ := n + k

def c_n (n : ℕ) (k : ℤ) : ℤ :=
if (a_n n : ℤ) ≤ (b_n n k) then b_n n k else a_n n

theorem find_k_range : 
  (∀ n ∈ { m : ℕ | m > 0 }, c_n 5 = a_n 5 ∧ c_n 5 ≤ c_n n) → 
  (∃ k : ℤ, -5 ≤ k ∧ k ≤ -3) :=
by
  sorry

end find_k_range_l311_311503


namespace camel_steps_divisibility_l311_311836

variables (A B : Type) (p q : ℕ)

-- Description of the conditions
-- let A, B be vertices
-- p and q be the steps to travel from A to B in different paths

theorem camel_steps_divisibility (h1: ∃ r : ℕ, p + r ≡ 0 [MOD 3])
                                  (h2: ∃ r : ℕ, q + r ≡ 0 [MOD 3]) : (p - q) % 3 = 0 := by
  sorry

end camel_steps_divisibility_l311_311836


namespace maximum_point_of_f_l311_311934

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x - 2) * Real.exp x

theorem maximum_point_of_f : ∃ x : ℝ, x = -2 ∧
  ∀ y : ℝ, f y ≤ f x :=
sorry

end maximum_point_of_f_l311_311934


namespace product_remainder_l311_311283

theorem product_remainder
    (a b c : ℕ)
    (h₁ : a % 36 = 16)
    (h₂ : b % 36 = 8)
    (h₃ : c % 36 = 24) :
    (a * b * c) % 36 = 12 := 
    by
    sorry

end product_remainder_l311_311283


namespace ellipse_standard_equation_l311_311427

theorem ellipse_standard_equation (c a : ℝ) (h1 : 2 * c = 8) (h2 : 2 * a = 10) : 
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ ( ( ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) ∨ ( ∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ) )) :=
by
  sorry

end ellipse_standard_equation_l311_311427


namespace pots_on_each_shelf_l311_311465

variable (x : ℕ)
variable (h1 : 4 * 3 * x = 60)

theorem pots_on_each_shelf : x = 5 := by
  -- proof will go here
  sorry

end pots_on_each_shelf_l311_311465


namespace number_of_odd_factors_of_252_l311_311364

def numOddFactors (n : ℕ) : ℕ :=
  if ∀ d : ℕ, n % d = 0 → ¬(d % 2 = 0) then d
  else 0

theorem number_of_odd_factors_of_252 : numOddFactors 252 = 6 := by
  -- Definition of n
  let n := 252
  -- Factor n into 2^2 * 63
  have h1 : n = 2^2 * 63 := rfl
  -- Find the number of odd factors of 63 since factors of 252 that are odd are the same as factors of 63
  have h2 : 63 = 3^2 * 7 := rfl
  -- Check the number of factors of 63
  sorry

end number_of_odd_factors_of_252_l311_311364


namespace tan_sum_eq_tan_prod_l311_311515

noncomputable def tan (x : Real) : Real :=
  Real.sin x / Real.cos x

theorem tan_sum_eq_tan_prod (α β γ : Real) (h : tan α + tan β + tan γ = tan α * tan β * tan γ) :
  ∃ k : Int, α + β + γ = k * Real.pi :=
by
  sorry

end tan_sum_eq_tan_prod_l311_311515


namespace inverse_function_value_l311_311250

-- Defining the function g as a list of pairs
def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 3
  | 2 => 6
  | 3 => 1
  | 4 => 5
  | 5 => 4
  | 6 => 2
  | _ => 0 -- default case which should not be used

-- Defining the inverse function g_inv using the values determined from g
def g_inv (y : ℕ) : ℕ :=
  match y with
  | 3 => 1
  | 6 => 2
  | 1 => 3
  | 5 => 4
  | 4 => 5
  | 2 => 6
  | _ => 0 -- default case which should not be used

theorem inverse_function_value :
  g_inv (g_inv (g_inv 6)) = 2 :=
by
  sorry

end inverse_function_value_l311_311250


namespace intersection_correct_l311_311352

def A : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x : ℝ | 2 < x ∧ x < 4 }

theorem intersection_correct : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_correct_l311_311352


namespace intersection_point_parabola_l311_311821

theorem intersection_point_parabola :
  ∃ k : ℝ, (∀ x : ℝ, (3 * (x - 4)^2 + k = 0 ↔ x = 2 ∨ x = 6)) :=
by
  sorry

end intersection_point_parabola_l311_311821


namespace minimum_moves_to_reset_counters_l311_311566

-- Definitions
def counter_in_initial_range (c : ℕ) := 1 ≤ c ∧ c ≤ 2017
def valid_move (decrements : ℕ) (counters : list ℕ) : list ℕ :=
  counters.map (λ c, if c ≥ decrements then c - decrements else c)
def all_counters_zero (counters : list ℕ) : Prop :=
  counters.all (λ c, c = 0)

-- Problem statement
theorem minimum_moves_to_reset_counters :
  ∀ (counters : list ℕ)
  (h : counters.length = 28)
  (h' : ∀ c ∈ counters, counter_in_initial_range c),
  ∃ (moves : ℕ), moves = 11 ∧
    ∀ (f : ℕ → list ℕ → list ℕ)
    (hm : ∀ ds cs, ds > 0 → cs.length = 28 → 
           (∀ c ∈ cs, counter_in_initial_range c) →
           ds ≤ 2017 → f ds cs = valid_move ds cs),
    all_counters_zero (nat.iterate (f (λ m cs, valid_move m cs)) 11 counters) :=
sorry

end minimum_moves_to_reset_counters_l311_311566


namespace sides_of_regular_polygon_with_20_diagonals_l311_311994

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l311_311994


namespace number_of_non_congruent_triangles_perimeter_18_l311_311958

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l311_311958


namespace complex_div_eq_i_l311_311554

open Complex

theorem complex_div_eq_i : (1 + I) / (1 - I) = I := by
  sorry

end complex_div_eq_i_l311_311554


namespace vanessa_recycled_correct_l311_311442

-- Define conditions as separate hypotheses
variable (weight_per_point : ℕ := 9)
variable (points_earned : ℕ := 4)
variable (friends_recycled : ℕ := 16)

-- Define the total weight recycled as points earned times the weight per point
def total_weight_recycled (points_earned weight_per_point : ℕ) : ℕ := points_earned * weight_per_point

-- Define the weight recycled by Vanessa
def vanessa_recycled (total_recycled friends_recycled : ℕ) : ℕ := total_recycled - friends_recycled

-- Main theorem statement
theorem vanessa_recycled_correct (weight_per_point points_earned friends_recycled : ℕ) 
    (hw : weight_per_point = 9) (hp : points_earned = 4) (hf : friends_recycled = 16) : 
    vanessa_recycled (total_weight_recycled points_earned weight_per_point) friends_recycled = 20 := 
by 
  sorry

end vanessa_recycled_correct_l311_311442


namespace exists_six_subjects_l311_311080

-- Define the number of students and subjects
def students := Fin 7
def subjects := Fin 12

-- Assume each student has a unique 12-tuple of marks represented by a function from students to subjects
variables (marks : students → subjects → ℕ) 

-- Condition: No two students have identical marks in all 12 subjects
axiom unique_marks : ∀ x y : students, x ≠ y → ∃ s : subjects, marks x s ≠ marks y s

-- Prove that we can choose 6 subjects such that any two of the students have different marks in at least one of these subjects
theorem exists_six_subjects : ∃ (S : Finset subjects) (h : S.card = 6), 
  ∀ x y : students, x ≠ y → ∃ s ∈ S, marks x s ≠ marks y s :=
sorry

end exists_six_subjects_l311_311080


namespace min_value_of_2x_plus_y_l311_311800

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 :=
sorry

end min_value_of_2x_plus_y_l311_311800


namespace different_suits_choice_count_l311_311197

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l311_311197


namespace integer_product_is_192_l311_311877

theorem integer_product_is_192 (A B C : ℤ)
  (h1 : A + B + C = 33)
  (h2 : C = 3 * B)
  (h3 : A = C - 23) :
  A * B * C = 192 :=
sorry

end integer_product_is_192_l311_311877


namespace max_a_satisfies_no_lattice_points_l311_311334

-- Define the conditions
def no_lattice_points (m : ℚ) (x_upper : ℕ) :=
  ∀ x : ℕ, 0 < x ∧ x ≤ x_upper → ¬∃ y : ℤ, y = m * x + 3

-- Final statement we need to prove
theorem max_a_satisfies_no_lattice_points :
  ∃ a : ℚ, a = 51 / 151 ∧ ∀ m : ℚ, 1 / 3 < m → m < a → no_lattice_points m 150 :=
sorry

end max_a_satisfies_no_lattice_points_l311_311334


namespace marks_in_mathematics_l311_311906

-- Define the marks obtained in each subject and the average
def marks_in_english : ℕ := 86
def marks_in_physics : ℕ := 82
def marks_in_chemistry : ℕ := 87
def marks_in_biology : ℕ := 85
def average_marks : ℕ := 85
def number_of_subjects : ℕ := 5

-- The theorem to prove the marks in Mathematics
theorem marks_in_mathematics : ℕ :=
  let sum_of_marks := average_marks * number_of_subjects
  let sum_of_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology
  sum_of_marks - sum_of_known_marks

-- The expected result that we need to prove
example : marks_in_mathematics = 85 := by
  -- skip the proof
  sorry

end marks_in_mathematics_l311_311906


namespace number_of_positive_area_triangles_l311_311671

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l311_311671


namespace volume_of_cube_l311_311842

theorem volume_of_cube (a : ℕ) (h : ((a - 2) * a * (a + 2)) = a^3 - 16) : a^3 = 64 :=
sorry

end volume_of_cube_l311_311842


namespace non_congruent_triangles_with_perimeter_18_l311_311940

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l311_311940


namespace fraction_product_simplification_l311_311318

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_simplification_l311_311318


namespace number_of_routes_jack_to_jill_l311_311056

def num_routes_avoiding (start goal avoid : ℕ × ℕ) : ℕ := sorry

theorem number_of_routes_jack_to_jill : 
  num_routes_avoiding (0,0) (3,2) (1,1) = 4 :=
sorry

end number_of_routes_jack_to_jill_l311_311056


namespace school_students_unique_l311_311689

theorem school_students_unique 
  (n : ℕ)
  (h1 : 70 < n) 
  (h2 : n < 130) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2)
  (h5 : n % 6 = 2) : 
  (n = 92 ∨ n = 122) :=
  sorry

end school_students_unique_l311_311689


namespace find_missing_number_l311_311723

theorem find_missing_number
  (mean : ℝ)
  (n : ℕ)
  (nums : List ℝ)
  (total_sum : ℝ)
  (sum_known_numbers : ℝ)
  (missing_number : ℝ) :
  mean = 20 → 
  n = 8 →
  nums = [1, 22, 23, 24, 25, missing_number, 27, 2] →
  total_sum = mean * n →
  sum_known_numbers = 1 + 22 + 23 + 24 + 25 + 27 + 2 →
  missing_number = total_sum - sum_known_numbers :=
by
  intros
  sorry

end find_missing_number_l311_311723


namespace non_congruent_triangles_perimeter_18_l311_311956

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l311_311956


namespace odd_factors_360_l311_311651

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l311_311651


namespace sqrt_x_minus_1_meaningful_l311_311099

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_x_minus_1_meaningful_l311_311099


namespace incorrect_statement_C_l311_311745

theorem incorrect_statement_C (a b : ℤ) (h : |a| = |b|) : (a ≠ b ∧ a = -b) :=
by
  sorry

end incorrect_statement_C_l311_311745


namespace henri_drove_farther_l311_311785

theorem henri_drove_farther (gervais_avg_miles_per_day : ℕ) (gervais_days : ℕ) (henri_total_miles : ℕ)
  (h1 : gervais_avg_miles_per_day = 315) (h2 : gervais_days = 3) (h3 : henri_total_miles = 1250) :
  (henri_total_miles - (gervais_avg_miles_per_day * gervais_days) = 305) :=
by
  -- Here we would provide the proof, but we are omitting it as requested
  sorry

end henri_drove_farther_l311_311785


namespace evaluate_expression_m_4_evaluate_expression_m_negative_4_l311_311370

variables (a b c d m : ℝ)

theorem evaluate_expression_m_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_4 : m = 4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = 35 :=
by sorry

theorem evaluate_expression_m_negative_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_negative_4 : m = -4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = -13 :=
by sorry

end evaluate_expression_m_4_evaluate_expression_m_negative_4_l311_311370


namespace vector_on_plane_l311_311835

-- Define the vectors w and the condition for proj_w v
def w : ℝ × ℝ × ℝ := (3, -3, 3)
def v (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)
def projection_condition (x y z : ℝ) : Prop :=
  ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * (-3) = -6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6

-- Define the plane equation
def plane_eq (x y z : ℝ) : Prop := x - y + z - 18 = 0

-- Prove that the set of vectors v lies on the plane
theorem vector_on_plane (x y z : ℝ) (h : projection_condition x y z) : plane_eq x y z :=
  sorry

end vector_on_plane_l311_311835


namespace count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l311_311664

theorem count_positive_even_multiples_of_3_less_than_5000_perfect_squares :
  ∃ n : ℕ, (n = 11) ∧ ∀ k : ℕ, (k < 5000) → (k % 2 = 0) → (k % 3 = 0) → (∃ m : ℕ, k = m * m) → k ≤ 36 * 11 * 11 :=
by {
  sorry
}

end count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l311_311664


namespace line_point_relation_l311_311386

-- Define the points and equations
def polar_point_A := (real.sqrt 2, real.pi / 4)

def polar_eq_of_line (ρ θ : ℝ) (a : ℝ) := ρ * real.cos (θ - real.pi / 4) = a

-- Points A lies on line l
axiom A_on_l : polar_eq_of_line (real.sqrt 2) (real.pi / 4) a

-- Cartesian equation conversions and the circle definition
def cartesian_eq_of_line (x y a : ℝ) := x + y - a = 0

def parametric_circle (α : ℝ) : (ℝ × ℝ) :=
  (1 + real.cos α, real.sin α)

def cartesian_eq_of_circle (x y : ℝ) := (x - 1)^2 + y^2 = 1

-- The conditions and statements to prove
theorem line_point_relation (a : ℝ) (x y : ℝ) : 
  (polar_point_A = (real.sqrt 2, real.pi / 4)) → 
  (polar_eq_of_line (real.sqrt 2) (real.pi / 4) a) →
  (∀ (α : ℝ), parametric_circle α = (1 + real.cos α, real.sin α)) →
  a = real.sqrt 2 ∧ cartesian_eq_of_line x y 2 = 0 ∧ 
  ∃ α : ℝ, (x, y) = parametric_circle α → 
  cartesian_eq_of_circle x y 
by
  sorry

end line_point_relation_l311_311386


namespace find_sides_from_diagonals_l311_311981

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l311_311981


namespace point_B_coordinates_l311_311538

theorem point_B_coordinates (A B : ℝ) (hA : A = -2) (hDist : |A - B| = 3) : B = -5 ∨ B = 1 :=
by
  sorry

end point_B_coordinates_l311_311538


namespace odd_factors_of_360_l311_311657

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l311_311657


namespace binary_101_eq_5_l311_311477

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l311_311477


namespace sqrt_inequality_l311_311932

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) :=
sorry

end sqrt_inequality_l311_311932


namespace lattice_points_on_hyperbola_l311_311813

theorem lattice_points_on_hyperbola :
  {p : (ℤ × ℤ) // p.1^2 - p.2^2 = 1800^2}.card = 150 :=
sorry

end lattice_points_on_hyperbola_l311_311813


namespace percentage_of_primes_divisible_by_3_l311_311272

-- Define the set of prime numbers less than 20
def primeNumbersLessThanTwenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define a function to check divisibility by 3
def divisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

-- Define the subset of primes less than 20 that are divisible by 3
def primesDivisibleBy3 : Set ℕ :=
  {n ∈ primeNumbersLessThanTwenty | divisibleBy3 n}

theorem percentage_of_primes_divisible_by_3 :
  (primesDivisibleBy3.to_finset.card : ℚ) / (primeNumbersLessThanTwenty.to_finset.card : ℚ) = 0.125 :=
by
  -- Proof goes here
  sorry

end percentage_of_primes_divisible_by_3_l311_311272


namespace coordinates_of_C_prime_l311_311866

noncomputable def transform_C (C : ℝ × ℝ) : ℝ × ℝ :=
  let C' := (-C.1, C.2)
  let C'' := (C'.1, -C'.2)
  let C''' := (C''.1 + 3, C''.2 - 4)
  C'''

theorem coordinates_of_C_prime :
  let C := (3 : ℝ, 3 : ℝ)
  transform_C C = (0, -7) :=
by
  sorry

end coordinates_of_C_prime_l311_311866


namespace frood_least_throw_points_more_than_eat_l311_311829

theorem frood_least_throw_points_more_than_eat (n : ℕ) : n^2 > 12 * n ↔ n ≥ 13 :=
sorry

end frood_least_throw_points_more_than_eat_l311_311829


namespace brad_read_more_books_l311_311284

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end brad_read_more_books_l311_311284


namespace percentage_of_primes_divisible_by_3_is_12_5_l311_311277

-- Define the set of all prime numbers less than 20
def primes_less_than_twenty : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the primes less than 20 that are divisible by 3
def primes_divisible_by_3 : set ℕ := {3}

-- Define the total number of primes less than 20
def total_primes : ℕ := 8

-- Calculate the percentage of primes less than 20 that are divisible by 3
def percentage_primes_divisible_by_3 := (card primes_divisible_by_3 * 100) / total_primes

-- Prove that the percentage of primes less than 20 that are divisible by 3 is 12.5%
theorem percentage_of_primes_divisible_by_3_is_12_5 :
    percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_3_is_12_5_l311_311277


namespace ways_to_choose_4_cards_of_different_suits_l311_311185

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l311_311185


namespace polygon_diagonals_l311_311986

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l311_311986


namespace combined_original_price_l311_311070

theorem combined_original_price (S P : ℝ) 
  (hS : 0.25 * S = 6) 
  (hP : 0.60 * P = 12) :
  S + P = 44 :=
by
  sorry

end combined_original_price_l311_311070


namespace sum_is_1716_l311_311837

-- Given conditions:
variables (a b c d : ℤ)
variable (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h_roots1 : ∀ t, t * t - 12 * a * t - 13 * b = 0 ↔ t = c ∨ t = d)
variable (h_roots2 : ∀ t, t * t - 12 * c * t - 13 * d = 0 ↔ t = a ∨ t = b)

-- Prove the desired sum of the constants:
theorem sum_is_1716 : a + b + c + d = 1716 :=
by
  sorry

end sum_is_1716_l311_311837


namespace total_surface_area_of_three_face_painted_cubes_l311_311302

def cube_side_length : ℕ := 9
def small_cube_side_length : ℕ := 1
def num_small_cubes_with_three_faces_painted : ℕ := 8
def surface_area_of_each_painted_face : ℕ := 6

theorem total_surface_area_of_three_face_painted_cubes :
  num_small_cubes_with_three_faces_painted * surface_area_of_each_painted_face = 48 := by
  sorry

end total_surface_area_of_three_face_painted_cubes_l311_311302


namespace find_counterfeit_coin_l311_311596

def is_counterfeit (coins : Fin 9 → ℝ) (i : Fin 9) : Prop :=
  ∀ j : Fin 9, j ≠ i → coins j = coins 0 ∧ coins i < coins 0

def algorithm_exists (coins : Fin 9 → ℝ) : Prop :=
  ∃ f : (Fin 9 → ℝ) → Fin 9, is_counterfeit coins (f coins)

theorem find_counterfeit_coin (coins : Fin 9 → ℝ) (h : ∃ i : Fin 9, is_counterfeit coins i) : algorithm_exists coins :=
by sorry

end find_counterfeit_coin_l311_311596


namespace expression_for_f_pos_f_monotone_on_pos_l311_311530

section

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_neg : ∀ x, -1 ≤ x ∧ x < 0 → f x = 2 * x + 1 / x^2)

-- Part 1: Prove the expression for f(x) when x ∈ (0,1]
theorem expression_for_f_pos (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  f x = 2 * x - 1 / x^2 :=
sorry

-- Part 2: Prove the monotonicity of f(x) on (0,1]
theorem f_monotone_on_pos : 
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y ≤ 1 → f x < f y :=
sorry

end

end expression_for_f_pos_f_monotone_on_pos_l311_311530


namespace pumpkin_pie_filling_l311_311134

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end pumpkin_pie_filling_l311_311134


namespace simplify_fraction_sum_l311_311067

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem simplify_fraction_sum (x : ℝ) (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ( (x + a) ^ 2 / ((a - b) * (a - c))
  + (x + b) ^ 2 / ((b - a) * (b - c))
  + (x + c) ^ 2 / ((c - a) * (c - b)) )
  = a * x + b * x + c * x - a - b - c :=
sorry

end simplify_fraction_sum_l311_311067


namespace number_of_valid_triangles_l311_311666

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l311_311666


namespace odd_factors_360_l311_311650

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l311_311650


namespace angle_measure_l311_311008

theorem angle_measure (α : ℝ) (h1 : α - (90 - α) = 20) : α = 55 := by
  -- Proof to be provided here
  sorry

end angle_measure_l311_311008


namespace domain_of_function_l311_311721

section
variable (x : ℝ)

def condition_1 := x + 4 ≥ 0
def condition_2 := x + 2 ≠ 0
def domain := { x : ℝ | x ≥ -4 ∧ x ≠ -2 }

theorem domain_of_function : (condition_1 x ∧ condition_2 x) ↔ (x ∈ domain) :=
by
  sorry
end

end domain_of_function_l311_311721


namespace hyperbola_eccentricity_l311_311497

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hyperbola_eq : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (asymptote : ∀ x, asymptote := (y = √2 * x)) :
  e = √3 := by
  sorry

end hyperbola_eccentricity_l311_311497


namespace initial_lives_emily_l311_311482

theorem initial_lives_emily (L : ℕ) (h1 : L - 25 + 24 = 41) : L = 42 :=
by
  sorry

end initial_lives_emily_l311_311482


namespace y_intercept_of_line_l311_311251

theorem y_intercept_of_line : ∀ (x y : ℝ), (5 * x - 2 * y - 10 = 0) → (x = 0) → (y = -5) :=
by
  intros x y h1 h2
  sorry

end y_intercept_of_line_l311_311251


namespace perp_DM_PN_l311_311069

-- Definitions of the triangle and its elements
variables {A B C M N P D : Point}
variables (triangle_incircle_touch : ∀ (A B C : Point) (triangle : Triangle ABC),
  touches_incircle_at triangle B C M ∧ 
  touches_incircle_at triangle C A N ∧ 
  touches_incircle_at triangle A B P)
variables (point_D : lies_on_segment D N P)
variables {BD CD DP DN : ℝ}
variables (ratio_condition : DP / DN = BD / CD)

-- The theorem statement
theorem perp_DM_PN 
  (h1 : triangle_incircle_touch A B C) 
  (h2 : point_D)
  (h3 : ratio_condition) : 
  is_perpendicular D M P N := 
sorry

end perp_DM_PN_l311_311069


namespace socks_selection_l311_311239

theorem socks_selection :
  (Nat.choose 7 3) - (Nat.choose 6 3) = 15 :=
by sorry

end socks_selection_l311_311239


namespace max_passengers_l311_311730

theorem max_passengers (total_stops : ℕ) (bus_capacity : ℕ)
  (h_total_stops : total_stops = 12) 
  (h_bus_capacity : bus_capacity = 20) 
  (h_no_same_stop : ∀ (a b : ℕ), a ≠ b → (a < total_stops) → (b < total_stops) → 
    ∃ x y : ℕ, x ≠ y ∧ x < total_stops ∧ y < total_stops ∧ 
    ((x = a ∧ y ≠ a) ∨ (x ≠ b ∧ y = b))) :
  ∃ max_passengers : ℕ, max_passengers = 50 :=
  sorry

end max_passengers_l311_311730


namespace carrie_profit_l311_311608

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end carrie_profit_l311_311608


namespace diameter_of_circle_A_l311_311904

theorem diameter_of_circle_A (r_B r_C : ℝ) (h1 : r_B = 12) (h2 : r_C = 3)
  (area_relation : ∀ (r_A : ℝ), π * (r_B^2 - r_A^2) = 4 * (π * r_C^2)) :
  ∃ r_A : ℝ, 2 * r_A = 12 * Real.sqrt 3 := by
  -- We will club the given conditions and logical sequence here
  sorry

end diameter_of_circle_A_l311_311904


namespace probability_divisor_of_8_on_8_sided_die_l311_311882

def divisor_probability : ℚ :=
  let sample_space := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_8 := {1, 2, 4, 8}
  let favorable_outcomes := divisors_of_8 ∩ sample_space
  favorable_outcomes.card / sample_space.card

theorem probability_divisor_of_8_on_8_sided_die :
  divisor_probability = 1 / 2 :=
sorry

end probability_divisor_of_8_on_8_sided_die_l311_311882


namespace non_congruent_triangles_perimeter_18_l311_311954

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l311_311954


namespace parabola_rotation_180_equivalent_l311_311712

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 - 2

-- Define the expected rotated parabola equation
def rotated_parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- Prove that the rotated parabola is correctly transformed
theorem parabola_rotation_180_equivalent :
  ∀ x, rotated_parabola x = -2 * (x - 3)^2 - 2 := 
by
  intro x
  unfold rotated_parabola
  sorry

end parabola_rotation_180_equivalent_l311_311712


namespace farmer_rent_l311_311286

-- Definitions based on given conditions
def rent_per_acre_per_month : ℕ := 60
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Problem statement: 
-- Prove that the monthly rent to rent the rectangular plot is $600.
theorem farmer_rent : 
  (length_of_plot * width_of_plot) / square_feet_per_acre * rent_per_acre_per_month = 600 :=
by
  sorry

end farmer_rent_l311_311286


namespace time_to_cross_bridge_l311_311583

-- Defining the given conditions
def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 140

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

-- Calculating the speed in m/s
def speed_of_train_ms : ℚ := kmh_to_ms speed_of_train_kmh

-- Calculating total distance to be covered
def total_distance : ℕ := length_of_train + length_of_bridge

-- Expected time to cross the bridge
def expected_time : ℚ := total_distance / speed_of_train_ms

-- The proof statement
theorem time_to_cross_bridge :
  expected_time = 12.5 := by
  sorry

end time_to_cross_bridge_l311_311583


namespace numberOfWaysToChoose4Cards_l311_311178

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l311_311178


namespace sequence_is_arithmetic_sum_of_sequence_l311_311792

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n + 2 * 3 ^ (n + 1)

def arithmetic_seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, (a (n + 1) / 3 ^ (n + 1)) - (a n / 3 ^ n) = c

def sum_S (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = (n - 1) * 3 ^ (n + 1) + 3

theorem sequence_is_arithmetic (a : ℕ → ℕ)
  (h : sequence_a a) : 
  arithmetic_seq a 2 :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_a a) :
  sum_S a S :=
sorry

end sequence_is_arithmetic_sum_of_sequence_l311_311792


namespace find_ratio_l311_311375

def given_conditions (a b c x y z : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 25 ∧ x^2 + y^2 + z^2 = 36 ∧ a * x + b * y + c * z = 30

theorem find_ratio (a b c x y z : ℝ)
  (h : given_conditions a b c x y z) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
sorry

end find_ratio_l311_311375


namespace odd_factors_of_360_l311_311648

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l311_311648


namespace koala_fiber_intake_l311_311833

theorem koala_fiber_intake 
  (absorption_rate : ℝ) 
  (absorbed_fiber : ℝ) 
  (eaten_fiber : ℝ) 
  (h1 : absorption_rate = 0.40) 
  (h2 : absorbed_fiber = 16)
  (h3 : absorbed_fiber = absorption_rate * eaten_fiber) :
  eaten_fiber = 40 := 
  sorry

end koala_fiber_intake_l311_311833


namespace probability_at_least_one_die_shows_three_l311_311437

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end probability_at_least_one_die_shows_three_l311_311437


namespace tangent_line_x_squared_l311_311915

theorem tangent_line_x_squared (P : ℝ × ℝ) (hP : P = (1, -1)) :
  ∃ (a : ℝ), a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 ∧
    ((∀ x : ℝ, (2 * (1 + Real.sqrt 2) * x - (3 + 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 + Real.sqrt 2) * P.1 - (3 + 2 * Real.sqrt 2))) ∨
    (∀ x : ℝ, (2 * (1 - Real.sqrt 2) * x - (3 - 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 - Real.sqrt 2) * P.1 - (3 - 2 * Real.sqrt 2)))) := by
  sorry

end tangent_line_x_squared_l311_311915


namespace find_scalars_l311_311228

noncomputable def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-2, 0]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

theorem find_scalars (r s : ℤ) (h_r : r = 3) (h_s : s = -8) :
    N * N = r • N + s • I :=
by
  rw [h_r, h_s]
  sorry

end find_scalars_l311_311228


namespace number_of_valid_triangles_l311_311665

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l311_311665


namespace find_ab_l311_311177

-- Define the conditions and the goal
theorem find_ab (a b : ℝ) (h1 : a^2 + b^2 = 26) (h2 : a + b = 7) : ab = 23 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end find_ab_l311_311177


namespace sum_of_digits_of_n_l311_311230

theorem sum_of_digits_of_n
  (n : ℕ) (h : 0 < n)
  (eqn : (n+1)! + (n+3)! = n! * 964) :
  n = 7 ∧ (Nat.digits 10 7).sum = 7 := 
sorry

end sum_of_digits_of_n_l311_311230


namespace problem_gcd_polynomials_l311_311167

theorem problem_gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * k ∧ k % 2 = 0) :
  gcd (4 * b ^ 2 + 55 * b + 120) (3 * b + 12) = 12 :=
by
  sorry

end problem_gcd_polynomials_l311_311167


namespace grain_to_rice_system_l311_311519

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end grain_to_rice_system_l311_311519


namespace diagonals_in_polygon_l311_311613

-- Define the number of sides of the polygon
def n : ℕ := 30

-- Define the formula for the total number of diagonals in an n-sided polygon
def total_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Define the number of excluded diagonals for being parallel to one given side
def excluded_diagonals : ℕ := 1

-- Define the final count of valid diagonals after exclusion
def valid_diagonals : ℕ := total_diagonals n - excluded_diagonals

-- State the theorem to prove
theorem diagonals_in_polygon : valid_diagonals = 404 := by
  sorry


end diagonals_in_polygon_l311_311613


namespace number_of_non_congruent_triangles_perimeter_18_l311_311959

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l311_311959


namespace percentage_decrease_l311_311207

noncomputable def original_fraction (N D : ℝ) : Prop := N / D = 0.75
noncomputable def new_fraction (N D x : ℝ) : Prop := (1.15 * N) / (D * (1 - x / 100)) = 15 / 16

theorem percentage_decrease (N D x : ℝ) (h1 : original_fraction N D) (h2 : new_fraction N D x) : 
  x = 22.67 := 
sorry

end percentage_decrease_l311_311207


namespace present_age_of_son_l311_311303

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 29) (h2 : M + 2 = 2 * (S + 2)) : S = 27 :=
sorry

end present_age_of_son_l311_311303


namespace quadratic_inequality_solution_set_l311_311804

variable (a b : ℝ)

theorem quadratic_inequality_solution_set :
  (∀ x : ℝ, (a + b) * x + 2 * a - 3 * b < 0 ↔ x > -(3 / 4)) →
  (∀ x : ℝ, (a - 2 * b) * x ^ 2 + 2 * (a - b - 1) * x + (a - 2) > 0 ↔ -3 + 2 / b < x ∧ x < -1) :=
by
  sorry

end quadratic_inequality_solution_set_l311_311804


namespace rearrange_to_rectangle_l311_311327

-- Definition of a geometric figure and operations
structure Figure where
  parts : List (List (ℤ × ℤ)) -- List of parts represented by lists of coordinates

def is_cut_into_three_parts (fig : Figure) : Prop :=
  fig.parts.length = 3

def can_be_rearranged_to_form_rectangle (fig : Figure) : Prop := sorry

-- Initial given figure
variable (initial_figure : Figure)

-- Conditions
axiom figure_can_be_cut : is_cut_into_three_parts initial_figure
axiom cuts_not_along_grid_lines : True -- Replace with appropriate geometric operation when image is known
axiom parts_can_be_flipped : True -- Replace with operation allowing part flipping

-- Theorem to prove
theorem rearrange_to_rectangle : 
  is_cut_into_three_parts initial_figure →
  can_be_rearranged_to_form_rectangle initial_figure := 
sorry

end rearrange_to_rectangle_l311_311327


namespace geometric_progression_x_l311_311782

theorem geometric_progression_x :
  ∃ x : ℝ, (70 + x) ^ 2 = (30 + x) * (150 + x) ∧ x = 10 :=
by sorry

end geometric_progression_x_l311_311782


namespace monotonic_solution_l311_311332

-- Definition of a monotonic function
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- The main theorem
theorem monotonic_solution (f : ℝ → ℝ) 
  (mon : monotonic f) 
  (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = -x) :=
sorry

end monotonic_solution_l311_311332


namespace evaluate_expression_l311_311510

theorem evaluate_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) :
  5 * x + 2 * y * 3 = 38 :=
by
  sorry

end evaluate_expression_l311_311510


namespace binomial_coeff_x5y3_in_expansion_eq_56_l311_311624

theorem binomial_coeff_x5y3_in_expansion_eq_56:
  let n := 8
  let k := 3
  let binom_coeff := Nat.choose n k
  binom_coeff = 56 := 
by sorry

end binomial_coeff_x5y3_in_expansion_eq_56_l311_311624


namespace minimum_value_fraction_l311_311161

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_fraction (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h_geometric : geometric_sequence a q)
  (h_positive : ∀ k : ℕ, 0 < a k)
  (h_condition1 : a 7 = a 6 + 2 * a 5)
  (h_condition2 : ∃ r, r ^ 2 = a m * a n ∧ r = 2 * a 1) :
  (1 / m + 9 / n) ≥ 4 :=
  sorry

end minimum_value_fraction_l311_311161


namespace Maxim_is_correct_l311_311291

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end Maxim_is_correct_l311_311291


namespace problem1_solution_problem2_solution_l311_311148

noncomputable def problem1 : ℝ :=
  (Real.sqrt (1 / 3) + Real.sqrt 6) / Real.sqrt 3

noncomputable def problem2 : ℝ :=
  (Real.sqrt 3)^2 - Real.sqrt 4 + Real.sqrt ((-2)^2)

theorem problem1_solution :
  problem1 = 1 + 3 * Real.sqrt 2 :=
by
  sorry

theorem problem2_solution :
  problem2 = 3 :=
by
  sorry

end problem1_solution_problem2_solution_l311_311148


namespace range_of_m_l311_311379

theorem range_of_m (m x : ℝ) (h1 : (3 * x) / (x - 1) = m / (x - 1) + 2) (h2 : x ≥ 0) (h3 : x ≠ 1) : 
  m ≥ 2 ∧ m ≠ 3 := 
sorry

end range_of_m_l311_311379


namespace regular_polygon_sides_l311_311971

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l311_311971


namespace cat_head_start_15_minutes_l311_311734

theorem cat_head_start_15_minutes :
  ∀ (t : ℕ), (25 : ℝ) = (20 : ℝ) * (1 + (t : ℝ) / 60) → t = 15 := by
  sorry

end cat_head_start_15_minutes_l311_311734


namespace joan_football_games_l311_311059

theorem joan_football_games (games_this_year games_last_year total_games: ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_last_year = 9)
  (h3 : total_games = games_this_year + games_last_year) :
  total_games = 13 := 
by
  sorry

end joan_football_games_l311_311059


namespace milk_for_6_cookies_l311_311865

namespace CookieMilk

-- Definitions for given conditions
def n_cookies := 24
def q_milk := 4
def pints_per_quart := 2
def cups_per_pint := 2

-- Definition for quantities needed in the problem
def cookies_goal := 6

-- Calculation based on conditions
def milk_per_cup := q_milk * pints_per_quart * cups_per_pint
def milk_per_cookie := milk_per_cup / n_cookies

-- Amount of milk needed for the goal number of cookies
def milk_needed := milk_per_cookie * cookies_goal

-- Lean proof statement
theorem milk_for_6_cookies : milk_needed = 4 :=
by
  sorry

end CookieMilk

end milk_for_6_cookies_l311_311865


namespace domain_of_inverse_l311_311502

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem domain_of_inverse (x : ℝ) : f x > 0 :=
by
  sorry

end domain_of_inverse_l311_311502


namespace megatek_manufacturing_percentage_l311_311112

theorem megatek_manufacturing_percentage 
  (total_degrees : ℝ := 360)
  (manufacturing_degrees : ℝ := 18)
  (is_proportional : (manufacturing_degrees / total_degrees) * 100 = 5) :
  (manufacturing_degrees / total_degrees) * 100 = 5 := 
  by
  exact is_proportional

end megatek_manufacturing_percentage_l311_311112


namespace correct_statement_l311_311447

def angle_terminal_side (a b : ℝ) : Prop :=
∃ k : ℤ, a = b + k * 360

def obtuse_angle (θ : ℝ) : Prop :=
90 < θ ∧ θ < 180

def third_quadrant_angle (θ : ℝ) : Prop :=
180 < θ ∧ θ < 270

def first_quadrant_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

def acute_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

theorem correct_statement :
  ¬∀ a b, angle_terminal_side a b → a = b ∧
  ¬∀ θ, obtuse_angle θ → θ < θ - 360 ∧
  ¬∀ θ, first_quadrant_angle θ → acute_angle θ ∧
  ∀ θ, acute_angle θ → first_quadrant_angle θ :=
by
  sorry

end correct_statement_l311_311447


namespace fido_leash_yard_reach_area_product_l311_311912

noncomputable def fido_leash_yard_fraction : ℝ :=
  let a := 2 + Real.sqrt 2
  let b := 8
  a * b

theorem fido_leash_yard_reach_area_product :
  ∃ (a b : ℝ), 
  (fido_leash_yard_fraction = (a * b)) ∧ 
  (1 > a) ∧ -- Regular Octagon computation constraints
  (b = 8) ∧ 
  a = 2 + Real.sqrt 2 :=
sorry

end fido_leash_yard_reach_area_product_l311_311912


namespace totalSleepIsThirtyHours_l311_311394

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l311_311394


namespace probability_of_at_least_one_three_l311_311436

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l311_311436


namespace total_sleep_correct_l311_311399

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l311_311399


namespace num_shirts_sold_l311_311301

theorem num_shirts_sold (p_jeans : ℕ) (c_shirt : ℕ) (total_earnings : ℕ) (h1 : p_jeans = 10) (h2 : c_shirt = 10) (h3 : total_earnings = 400) : ℕ :=
  let c_jeans := 2 * c_shirt
  let n_shirts := 20
  have h4 : p_jeans * c_jeans + n_shirts * c_shirt = total_earnings := by sorry
  n_shirts

end num_shirts_sold_l311_311301


namespace least_number_subtracted_l311_311576

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_pos : 0 < x) (h_init : n = 427398) (h_div : ∃ k : ℕ, (n - x) = 14 * k) : x = 6 :=
sorry

end least_number_subtracted_l311_311576


namespace calc_expression_is_24_l311_311466

def calc_expression : ℕ := (30 / (8 + 2 - 5)) * 4

theorem calc_expression_is_24 : calc_expression = 24 :=
by
  sorry

end calc_expression_is_24_l311_311466


namespace total_and_per_suitcase_profit_l311_311308

theorem total_and_per_suitcase_profit
  (num_suitcases : ℕ)
  (purchase_price_per_suitcase : ℕ)
  (total_sales_revenue : ℕ)
  (total_profit : ℕ)
  (profit_per_suitcase : ℕ)
  (h_num_suitcases : num_suitcases = 60)
  (h_purchase_price : purchase_price_per_suitcase = 100)
  (h_total_sales : total_sales_revenue = 8100)
  (h_total_profit : total_profit = total_sales_revenue - num_suitcases * purchase_price_per_suitcase)
  (h_profit_per_suitcase : profit_per_suitcase = total_profit / num_suitcases) :
  total_profit = 2100 ∧ profit_per_suitcase = 35 := by
  sorry

end total_and_per_suitcase_profit_l311_311308


namespace area_of_rhombus_with_diagonals_6_and_8_l311_311170

theorem area_of_rhombus_with_diagonals_6_and_8 : 
  ∀ (d1 d2 : ℕ), d1 = 6 → d2 = 8 → (1 / 2 : ℝ) * d1 * d2 = 24 :=
by
  intros d1 d2 h1 h2
  sorry

end area_of_rhombus_with_diagonals_6_and_8_l311_311170


namespace upper_limit_of_arun_weight_l311_311686

variable (w : ℝ)

noncomputable def arun_opinion (w : ℝ) := 62 < w ∧ w < 72
noncomputable def brother_opinion (w : ℝ) := 60 < w ∧ w < 70
noncomputable def average_weight := 64

theorem upper_limit_of_arun_weight 
  (h1 : ∀ w, arun_opinion w → brother_opinion w → 64 = (62 + w) / 2 ) 
  : ∀ w, arun_opinion w ∧ brother_opinion w → w ≤ 66 :=
sorry

end upper_limit_of_arun_weight_l311_311686


namespace unique_triangles_count_l311_311943

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l311_311943


namespace savings_correct_l311_311429

noncomputable def school_price_math : Float := 45
noncomputable def school_price_science : Float := 60
noncomputable def school_price_literature : Float := 35

noncomputable def discount_math : Float := 0.20
noncomputable def discount_science : Float := 0.25
noncomputable def discount_literature : Float := 0.15

noncomputable def tax_school : Float := 0.07
noncomputable def tax_alt : Float := 0.06
noncomputable def shipping_alt : Float := 10

noncomputable def alt_price_math : Float := (school_price_math * (1 - discount_math)) * (1 + tax_alt)
noncomputable def alt_price_science : Float := (school_price_science * (1 - discount_science)) * (1 + tax_alt)
noncomputable def alt_price_literature : Float := (school_price_literature * (1 - discount_literature)) * (1 + tax_alt)

noncomputable def total_alt_cost : Float := alt_price_math + alt_price_science + alt_price_literature + shipping_alt

noncomputable def school_price_math_tax : Float := school_price_math * (1 + tax_school)
noncomputable def school_price_science_tax : Float := school_price_science * (1 + tax_school)
noncomputable def school_price_literature_tax : Float := school_price_literature * (1 + tax_school)

noncomputable def total_school_cost : Float := school_price_math_tax + school_price_science_tax + school_price_literature_tax

noncomputable def savings : Float := total_school_cost - total_alt_cost

theorem savings_correct : savings = 22.40 := by
  sorry

end savings_correct_l311_311429


namespace fewer_spoons_l311_311847

/--
Stephanie initially planned to buy 15 pieces of each type of silverware.
There are 4 types of silverware.
This totals to 60 pieces initially planned to be bought.
She only bought 44 pieces in total.
Show that she decided to purchase 4 fewer spoons.
-/
theorem fewer_spoons
  (initial_total : ℕ := 60)
  (final_total : ℕ := 44)
  (types : ℕ := 4)
  (pieces_per_type : ℕ := 15) :
  (initial_total - final_total) / types = 4 := 
by
  -- since initial_total = 60, final_total = 44, and types = 4
  -- we need to prove (60 - 44) / 4 = 4
  sorry

end fewer_spoons_l311_311847


namespace sufficient_not_necessary_l311_311445

theorem sufficient_not_necessary (x : ℝ) : abs x < 2 → (x^2 - x - 6 < 0) ∧ (¬(x^2 - x - 6 < 0) → abs x ≥ 2) :=
by
  sorry

end sufficient_not_necessary_l311_311445


namespace race_distance_l311_311383

theorem race_distance (a b c : ℝ) (d : ℝ) 
  (h1 : d / a = (d - 15) / b)
  (h2 : d / b = (d - 30) / c)
  (h3 : d / a = (d - 40) / c) : 
  d = 90 :=
by sorry

end race_distance_l311_311383


namespace nitin_rank_last_l311_311585

theorem nitin_rank_last (total_students : ℕ) (rank_start : ℕ) (rank_last : ℕ) 
  (h1 : total_students = 58) 
  (h2 : rank_start = 24) 
  (h3 : rank_last = total_students - rank_start + 1) : 
  rank_last = 35 := 
by 
  -- proof can be filled in here
  sorry

end nitin_rank_last_l311_311585


namespace agnes_twice_jane_in_years_l311_311600

def agnes_age := 25
def jane_age := 6

theorem agnes_twice_jane_in_years (x : ℕ) : 
  25 + x = 2 * (6 + x) → x = 13 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end agnes_twice_jane_in_years_l311_311600


namespace parabola_unique_intersection_x_axis_l311_311935

theorem parabola_unique_intersection_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ ∀ y, y^2 - 6*y + m = 0 → y = x) → m = 9 :=
by
  sorry

end parabola_unique_intersection_x_axis_l311_311935


namespace find_q_of_polynomial_l311_311858

noncomputable def Q (x : ℝ) (p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q_of_polynomial (p q d : ℝ) (mean_zeros twice_product sum_coeffs : ℝ)
  (h1 : mean_zeros = -p / 3)
  (h2 : twice_product = -2 * d)
  (h3 : sum_coeffs = 1 + p + q + d)
  (h4 : d = 4)
  (h5 : mean_zeros = twice_product)
  (h6 : sum_coeffs = twice_product) :
  q = -37 :=
sorry

end find_q_of_polynomial_l311_311858


namespace total_number_of_people_l311_311262

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l311_311262


namespace minimum_value_of_f_l311_311269

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end minimum_value_of_f_l311_311269


namespace volume_triangular_pyramid_correctness_l311_311340

noncomputable def volume_of_regular_triangular_pyramid 
  (a α l : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α

theorem volume_triangular_pyramid_correctness (a α l : ℝ) : volume_of_regular_triangular_pyramid a α l =
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α := 
sorry

end volume_triangular_pyramid_correctness_l311_311340


namespace min_deliveries_to_cover_cost_l311_311703

theorem min_deliveries_to_cover_cost (cost_per_van earnings_per_delivery gasoline_cost_per_delivery : ℕ) (h1 : cost_per_van = 4500) (h2 : earnings_per_delivery = 15 ) (h3 : gasoline_cost_per_delivery = 5) : 
  ∃ d : ℕ, 10 * d ≥ cost_per_van ∧ ∀ x : ℕ, x < d → 10 * x < cost_per_van :=
by
  use 450
  sorry

end min_deliveries_to_cover_cost_l311_311703


namespace largest_positive_integer_l311_311109

def binary_op (n : ℕ) : ℤ := n - (n * 5)

theorem largest_positive_integer (n : ℕ) (h : binary_op n < 21) : n ≤ 1 := 
sorry

end largest_positive_integer_l311_311109


namespace travel_remaining_distance_l311_311312

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end travel_remaining_distance_l311_311312


namespace travel_remaining_distance_l311_311311

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end travel_remaining_distance_l311_311311


namespace ricky_roses_l311_311417

theorem ricky_roses (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) (remaining_roses : ℕ)
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  (h4 : remaining_roses = initial_roses - stolen_roses) :
  remaining_roses / people = 4 :=
by sorry

end ricky_roses_l311_311417


namespace sheet_width_l311_311462

theorem sheet_width (L : ℕ) (w : ℕ) (A_typist : ℚ) 
  (L_length : L = 30)
  (A_typist_percentage : A_typist = 0.64) 
  (width_used : ∀ w, w > 0 → (w - 4) * (24 : ℕ) = A_typist * w * 30) : 
  w = 20 :=
by
  intros
  sorry

end sheet_width_l311_311462


namespace domain_of_function_l311_311090

/-- Prove the domain of the function f(x) = log10(2 * cos x - 1) + sqrt(49 - x^2) -/
theorem domain_of_function :
  { x : ℝ | -7 ≤ x ∧ x < - (5 * Real.pi) / 3 ∨ - Real.pi / 3 < x ∧ x < Real.pi / 3 ∨ (5 * Real.pi) / 3 < x ∧ x ≤ 7 }
  = { x : ℝ | 2 * Real.cos x - 1 > 0 ∧ 49 - x^2 ≥ 0 } :=
by {
  sorry
}

end domain_of_function_l311_311090


namespace probability_of_rolling_divisor_of_8_l311_311885

open_locale classical

-- Predicate: a number n is a divisor of 8
def is_divisor_of_8 (n : ℕ) : Prop := n ∣ 8

-- The total number of outcomes when rolling an 8-sided die
def total_outcomes : ℕ := 8

-- The probability of rolling a divisor of 8 on a fair 8-sided die
theorem probability_of_rolling_divisor_of_8 (is_fair_die : true) :
  (| {n | is_divisor_of_8 n} ∩ {1, 2, 3, 4, 5, 6, 7, 8} | : ℕ) / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_rolling_divisor_of_8_l311_311885


namespace sin_double_angle_l311_311508

theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 1 / 3) (h2 : (π / 2) < α ∧ α < π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := sorry

end sin_double_angle_l311_311508


namespace non_congruent_triangles_with_perimeter_18_l311_311941

theorem non_congruent_triangles_with_perimeter_18 :
  {s : Finset (Finset ℕ) // 
    ∀ (a b c : ℕ), (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ s = {a, b, c}) -> 
    a + b + c = 18 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b 
    } = 7 := 
by {
  sorry
}

end non_congruent_triangles_with_perimeter_18_l311_311941


namespace second_character_more_lines_l311_311222

theorem second_character_more_lines
  (C1 : ℕ) (S : ℕ) (T : ℕ) (X : ℕ)
  (h1 : C1 = 20)
  (h2 : C1 = S + 8)
  (h3 : T = 2)
  (h4 : S = 3 * T + X) :
  X = 6 :=
by
  -- proof can be filled in here
  sorry

end second_character_more_lines_l311_311222


namespace igor_arrangement_l311_311045

theorem igor_arrangement : 
  let n := 7 in 
  let k := 3 in 
  ∃ (ways : ℕ), ways = (n.factorial * (nat.choose (n-1) (k-1))) ∧ ways = 75600 := 
by
  let n := 7
  let k := 3
  use (n.factorial * (nat.choose (n-1) (k-1)))
  split
  · rfl
  · sorry

end igor_arrangement_l311_311045


namespace second_quadrant_coordinates_l311_311931

theorem second_quadrant_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : y^2 = 1) :
    (x, y) = (-2, 1) :=
  sorry

end second_quadrant_coordinates_l311_311931


namespace parabola_expression_l311_311796

theorem parabola_expression :
  ∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x - 5 = 0 → (x = -1 ∨ x = 5)) ∧ (a * (-1)^2 + b * (-1) - 5 = 0) ∧ (a * 5^2 + b * 5 - 5 = 0) ∧ (a * 1 - 4 = 1) :=
sorry

end parabola_expression_l311_311796


namespace aldehyde_formula_l311_311855

-- Define the problem starting with necessary variables
variables (n : ℕ)

-- Given conditions
def general_formula_aldehyde (n : ℕ) : String :=
  "CₙH_{2n}O"

def mass_percent_hydrogen (n : ℕ) : ℚ :=
  (2 * n) / (14 * n + 16)

-- Given the percentage of hydrogen in the aldehyde
def given_hydrogen_percent : ℚ := 0.12

-- The main theorem
theorem aldehyde_formula :
  (exists n : ℕ, mass_percent_hydrogen n = given_hydrogen_percent ∧ n = 6) ->
  general_formula_aldehyde 6 = "C₆H_{12}O" :=
by
  sorry

end aldehyde_formula_l311_311855


namespace necessary_but_not_sufficient_condition_l311_311114

def condition_neq_1_or_neq_2 (a b : ℤ) : Prop :=
  a ≠ 1 ∨ b ≠ 2

def statement_sum_neq_3 (a b : ℤ) : Prop :=
  a + b ≠ 3

theorem necessary_but_not_sufficient_condition :
  ∀ (a b : ℤ), condition_neq_1_or_neq_2 a b → ¬ (statement_sum_neq_3 a b) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l311_311114


namespace pow_gt_of_gt_l311_311747

variable {a x1 x2 : ℝ}

theorem pow_gt_of_gt (ha : a > 1) (hx : x1 > x2) : a^x1 > a^x2 :=
by sorry

end pow_gt_of_gt_l311_311747


namespace lemon_pie_degrees_l311_311046

noncomputable def num_students := 45
noncomputable def chocolate_pie_students := 15
noncomputable def apple_pie_students := 9
noncomputable def blueberry_pie_students := 9
noncomputable def other_pie_students := num_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
noncomputable def each_remaining_pie_students := other_pie_students / 3
noncomputable def fraction_lemon_pie := each_remaining_pie_students / num_students
noncomputable def degrees_lemon_pie := fraction_lemon_pie * 360

theorem lemon_pie_degrees : degrees_lemon_pie = 32 :=
sorry

end lemon_pie_degrees_l311_311046


namespace numberOfWaysToChoose4Cards_l311_311181

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end numberOfWaysToChoose4Cards_l311_311181


namespace henri_drove_farther_l311_311784

theorem henri_drove_farther (gervais_avg_miles_per_day : ℕ) (gervais_days : ℕ) (henri_total_miles : ℕ)
  (h1 : gervais_avg_miles_per_day = 315) (h2 : gervais_days = 3) (h3 : henri_total_miles = 1250) :
  (henri_total_miles - (gervais_avg_miles_per_day * gervais_days) = 305) :=
by
  -- Here we would provide the proof, but we are omitting it as requested
  sorry

end henri_drove_farther_l311_311784


namespace find_number_l311_311304

theorem find_number (k r n : ℤ) (hk : k = 38) (hr : r = 7) (h : n = 23 * k + r) : n = 881 := 
  by
  sorry

end find_number_l311_311304


namespace poly_has_two_distinct_negative_real_roots_l311_311907

-- Definition of the polynomial equation
def poly_eq (p x : ℝ) : Prop :=
  x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1 = 0

-- Theorem statement that needs to be proved
theorem poly_has_two_distinct_negative_real_roots (p : ℝ) :
  p > 1 → ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly_eq p x1 ∧ poly_eq p x2 :=
by
  sorry

end poly_has_two_distinct_negative_real_roots_l311_311907


namespace steve_total_time_on_roads_l311_311425

variables (d : ℝ) (v_back : ℝ) (v_to_work : ℝ)

-- Constants from the problem statement
def distance := 10 -- The distance from Steve's house to work is 10 km
def speed_back := 5 -- Steve's speed on the way back from work is 5 km/h

-- Given conditions
def speed_to_work := speed_back / 2 -- On the way back, Steve drives twice as fast as he did on the way to work

-- Define the time to get to work and back
def time_to_work := distance / speed_to_work
def time_back_home := distance / speed_back

-- Total time on roads
def total_time := time_to_work + time_back_home

-- The theorem to prove
theorem steve_total_time_on_roads : total_time = 6 := by
  -- Proof here
  sorry

end steve_total_time_on_roads_l311_311425


namespace aaron_weekly_earnings_l311_311140

def minutes_worked_monday : ℕ := 90
def minutes_worked_tuesday : ℕ := 40
def minutes_worked_wednesday : ℕ := 135
def minutes_worked_thursday : ℕ := 45
def minutes_worked_friday : ℕ := 60
def minutes_worked_saturday1 : ℕ := 90
def minutes_worked_saturday2 : ℕ := 75
def hourly_rate : ℕ := 4

def total_minutes_worked : ℕ :=
  minutes_worked_monday + 
  minutes_worked_tuesday + 
  minutes_worked_wednesday +
  minutes_worked_thursday + 
  minutes_worked_friday +
  minutes_worked_saturday1 + 
  minutes_worked_saturday2

def total_hours_worked : ℕ := total_minutes_worked / 60

def total_earnings : ℕ := total_hours_worked * hourly_rate

theorem aaron_weekly_earnings : total_earnings = 36 := by 
  sorry -- The proof is omitted.

end aaron_weekly_earnings_l311_311140


namespace polygon_diagonals_l311_311984

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l311_311984


namespace calculate_expression_l311_311012

theorem calculate_expression : ((-1 + 2) * 3 + 2^2 / (-4)) = 2 :=
by
  sorry

end calculate_expression_l311_311012


namespace choose_4_cards_of_different_suits_l311_311190

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l311_311190


namespace speed_of_first_car_l311_311735

theorem speed_of_first_car (v : ℝ) 
  (h1 : ∀ v, v > 0 → (first_speed = 1.25 * v))
  (h2 : 720 = (v + 1.25 * v) * 4) : 
  first_speed = 100 := 
by
  sorry

end speed_of_first_car_l311_311735


namespace sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l311_311176

noncomputable def P (x : ℝ) : Prop := (x - 1)^2 > 16
noncomputable def Q (x a : ℝ) : Prop := x^2 + (a - 8) * x - 8 * a ≤ 0

theorem sufficient_not_necessary (a : ℝ) (x : ℝ) :
  a = 3 →
  (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem necessary_and_sufficient (a : ℝ) :
  (-5 ≤ a ∧ a ≤ 3) ↔ ∀ x, (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem P_inter_Q (a : ℝ) (x : ℝ) :
  (a > 3 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a) ∨ (5 < x ∧ x ≤ 8)) ∧
  (-5 ≤ a ∧ a ≤ 3 → (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8)) ∧
  (-8 ≤ a ∧ a < -5 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) ∧
  (a < -8 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) :=
sorry

end sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l311_311176


namespace no_valid_six_digit_palindrome_years_l311_311456

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_six_digit_palindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ is_palindrome n

noncomputable def is_four_digit_prime_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ is_palindrome n ∧ is_prime n

noncomputable def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_palindrome n ∧ is_prime n

theorem no_valid_six_digit_palindrome_years :
  ∀ N : ℕ, is_six_digit_palindrome N →
  ¬ ∃ (p q : ℕ), is_four_digit_prime_palindrome p ∧ is_two_digit_prime_palindrome q ∧ N = p * q := 
sorry

end no_valid_six_digit_palindrome_years_l311_311456


namespace regular_polygon_sides_l311_311974

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l311_311974


namespace no_integer_solutions_l311_311331

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), (x ≠ 1 ∧ (x^7 - 1) / (x - 1) = (y^5 - 1)) :=
sorry

end no_integer_solutions_l311_311331


namespace divisible_iff_condition_l311_311701

theorem divisible_iff_condition (a b : ℤ) : 
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) :=
  sorry

end divisible_iff_condition_l311_311701


namespace ratio_of_first_to_second_l311_311433

theorem ratio_of_first_to_second (x y : ℕ) 
  (h1 : x + y + (1 / 3 : ℚ) * x = 110)
  (h2 : y = 30) :
  x / y = 2 :=
by
  sorry

end ratio_of_first_to_second_l311_311433


namespace find_abc_sol_l311_311490

theorem find_abc_sol (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (1 / ↑a + 1 / ↑b + 1 / ↑c = 1) →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end find_abc_sol_l311_311490


namespace green_pairs_count_l311_311146

theorem green_pairs_count 
  (blue_students : ℕ)
  (green_students : ℕ)
  (total_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ) 
  (mixed_pairs_students : ℕ) 
  (green_green_pairs : ℕ) 
  (count_blue : blue_students = 65)
  (count_green : green_students = 67)
  (count_total_students : total_students = 132)
  (count_total_pairs : total_pairs = 66)
  (count_blue_blue_pairs : blue_blue_pairs = 29)
  (count_mixed_blue_students : mixed_pairs_students = 7)
  (count_green_green_pairs : green_green_pairs = 30) :
  green_green_pairs = 30 :=
sorry

end green_pairs_count_l311_311146


namespace firetruck_reachable_area_l311_311214

theorem firetruck_reachable_area :
  let m := 700
  let n := 31
  let area := m / n -- The area in square miles
  let time := 1 / 10 -- The available time in hours
  let speed_highway := 50 -- Speed on the highway in miles/hour
  let speed_prairie := 14 -- Speed across the prairie in miles/hour
  -- The intersection point of highways is the origin (0, 0)
  -- The firetruck can move within the reachable area
  -- There exist regions formed by the intersection points of movement directions
  m + n = 731 :=
by
  sorry

end firetruck_reachable_area_l311_311214


namespace isosceles_triangle_count_l311_311215

namespace TriangleProblem

-- Define the basic geometric objects and their properties
structure Point := (x : ℝ) (y : ℝ)
def Triangle := (A B C : Point)

axiom is_congruent (A B : Point) : Prop
axiom is_parallel (L1 L2 : Point → Point) : Prop
axiom angle (A B C : Point) : ℝ

-- Given conditions
variables {A B C D E F : Point}
def ΔABC : Triangle := ⟨A, B, C⟩
def ΔABD : Triangle := ⟨A, B, D⟩
def ΔBDE : Triangle := ⟨B, D, E⟩
def ΔDEF : Triangle := ⟨D, E, F⟩
def ΔEFB : Triangle := ⟨E, F, B⟩
def ΔFEC : Triangle := ⟨F, E, C⟩
def ΔDEC : Triangle := ⟨D, E, C⟩

axiom H1 : is_congruent A B A C
axiom H2 : angle A B C = 60
axiom H3 : ∃ D, angle A B D = angle D B C
axiom H4 : ∃ E ∈ line B C, is_parallel (λ x, D) (λ x, A B)
axiom H5 : ∃ F ∈ line A C, is_parallel (λ x, E F) (λ x, B D)

-- Proof goal
theorem isosceles_triangle_count 
  (h1 : ΔABC.is_isosceles) 
  (h2 : ΔABD.is_isosceles) 
  (h3 : ΔBDE.is_isosceles) 
  (h4 : ΔDEF.is_isosceles) 
  (h5 : ΔEFB.is_isosceles) 
  (h6 : ΔFEC.is_isosceles) 
  (h7 : ΔDEC.is_isosceles) : 
  7 = 7 := 
sorry

end TriangleProblem

end isosceles_triangle_count_l311_311215


namespace min_cuts_for_100_quadrilaterals_l311_311766

theorem min_cuts_for_100_quadrilaterals : ∃ n : ℕ, (∃ q : ℕ, q = 100 ∧ n + 1 = q + 99) ∧ n = 1699 :=
sorry

end min_cuts_for_100_quadrilaterals_l311_311766


namespace find_n_interval_l311_311783

theorem find_n_interval :
  ∃ n : ℕ, n < 1000 ∧
  (∃ ghijkl : ℕ, (ghijkl < 999999) ∧ (ghijkl * n = 999999 * ghijkl)) ∧
  (∃ mnop : ℕ, (mnop < 9999) ∧ (mnop * (n + 5) = 9999 * mnop)) ∧
  151 ≤ n ∧ n ≤ 300 :=
sorry

end find_n_interval_l311_311783


namespace find_smallest_even_number_l311_311005

theorem find_smallest_even_number (x : ℕ) (h1 : 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14)) = 424) : 
  x = 46 := 
by
  sorry

end find_smallest_even_number_l311_311005


namespace necessary_sufficient_condition_l311_311491

noncomputable def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + (4 - 2 * a)

theorem necessary_sufficient_condition (a : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) : 
  (∀ (x : ℝ), f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end necessary_sufficient_condition_l311_311491


namespace ways_to_choose_4_cards_of_different_suits_l311_311183

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l311_311183


namespace jason_earnings_l311_311522

theorem jason_earnings :
  let fred_initial := 49
  let jason_initial := 3
  let emily_initial := 25
  let fred_increase := 1.5 
  let jason_increase := 0.625 
  let emily_increase := 0.40 
  let fred_new := fred_initial * fred_increase
  let jason_new := jason_initial * (1 + jason_increase)
  let emily_new := emily_initial * (1 + emily_increase)
  fred_new = fred_initial * fred_increase ->
  jason_new = jason_initial * (1 + jason_increase) ->
  emily_new = emily_initial * (1 + emily_increase) ->
  jason_new - jason_initial == 1.875 :=
by
  intros
  sorry

end jason_earnings_l311_311522


namespace option_c_correct_l311_311103

theorem option_c_correct : (3 * Real.sqrt 2) ^ 2 = 18 :=
by 
  -- Proof to be provided here
  sorry

end option_c_correct_l311_311103


namespace count_square_of_integer_fraction_l311_311917

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l311_311917


namespace factorization_a_minus_b_l311_311156

theorem factorization_a_minus_b (a b : ℤ) (h1 : 3 * b + a = -7) (h2 : a * b = -6) : a - b = 7 :=
sorry

end factorization_a_minus_b_l311_311156


namespace number_of_square_integers_l311_311920

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l311_311920


namespace find_n_l311_311172

def binomial_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + b) ^ n

def expanded_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + 3 * b) ^ n

theorem find_n (n : ℕ) :
  (expanded_coefficient_sum n 1 1) / (binomial_coefficient_sum n 1 1) = 64 → n = 6 :=
by 
  sorry

end find_n_l311_311172


namespace find_retail_price_l311_311107

-- Define the conditions
def wholesale_price : ℝ := 90
def discount_rate : ℝ := 0.10
def profit_rate : ℝ := 0.20

-- Calculate the necessary values from conditions
def profit : ℝ := profit_rate * wholesale_price
def selling_price : ℝ := wholesale_price + profit
def discount_factor : ℝ := 1 - discount_rate

-- Rewrite the main theorem statement
theorem find_retail_price : ∃ w : ℝ, discount_factor * w = selling_price → w = 120 :=
by sorry

end find_retail_price_l311_311107


namespace consecutive_odd_sum_count_l311_311807

theorem consecutive_odd_sum_count (N : ℕ) :
  N = 20 ↔ (
    ∃ (ns : Finset ℕ), ∃ (js : Finset ℕ),
      (∀ n ∈ ns, n < 500) ∧
      (∀ j ∈ js, j ≥ 2) ∧
      ∀ n ∈ ns, ∃ j ∈ js, ∃ k, k = 3 ∧ N = j * (2 * k + j)
  ) :=
by
  sorry

end consecutive_odd_sum_count_l311_311807


namespace tangent_from_point_to_circle_l311_311616

theorem tangent_from_point_to_circle :
  ∀ (x y : ℝ),
  (x - 6)^2 + (y - 3)^2 = 4 →
  (x = 10 → y = 0 →
    4 * x - 3 * y = 19) :=
by
  sorry

end tangent_from_point_to_circle_l311_311616


namespace clownfish_display_tank_l311_311772

theorem clownfish_display_tank
  (C B : ℕ)
  (h1 : C = B)
  (h2 : C + B = 100)
  (h3 : ∀ dC dB : ℕ, dC = dB → C - dC = 24)
  (h4 : ∀ b : ℕ, b = (1 / 3) * 24): 
  C - (1 / 3 * 24) = 16 := sorry

end clownfish_display_tank_l311_311772


namespace incorrect_statement_D_l311_311933

theorem incorrect_statement_D (a b r : ℝ) (hr : r > 0) :
  ¬ ∀ b < r, ∃ x, (x - a)^2 + (0 - b)^2 = r^2 :=
by 
  sorry

end incorrect_statement_D_l311_311933


namespace polynomial_evaluation_l311_311911

theorem polynomial_evaluation (x : ℝ) (h₁ : 0 < x) (h₂ : x^2 - 2 * x - 15 = 0) :
  x^3 - 2 * x^2 - 8 * x + 16 = 51 :=
sorry

end polynomial_evaluation_l311_311911


namespace quadratic_transformation_l311_311894

theorem quadratic_transformation :
  ∀ (x : ℝ), (x^2 + 6*x - 2 = 0) → ((x + 3)^2 = 11) :=
by
  intros x h
  sorry

end quadratic_transformation_l311_311894


namespace functional_equation_solution_l311_311231

theorem functional_equation_solution (a b : ℝ) (f : ℝ → ℝ) :
  (0 < a ∧ 0 < b) →
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y = y^a * f (x / 2) + x^b * f (y / 2)) →
  (∃ c : ℝ, ∀ x : ℝ, 0 < x → (f x = c * x^a ∨ f x = 0)) :=
by
  intros
  sorry

end functional_equation_solution_l311_311231


namespace seventy_second_number_in_S_is_573_l311_311584

open Nat

def S : Set Nat := { k | k % 8 = 5 }

theorem seventy_second_number_in_S_is_573 : ∃ k ∈ (Finset.range 650), k = 8 * 71 + 5 :=
by
  sorry -- Proof goes here

end seventy_second_number_in_S_is_573_l311_311584


namespace prob_even_product_eq_19_over_20_l311_311434

-- Define the set of integers from 1 to 6
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Lean function to calculate the probability
noncomputable def even_product_probability : ℕ :=
  let total_ways := (S.card.choose 3) in
  let odd_elements := {1, 3, 5} : Finset ℕ in
  let odd_ways := (odd_elements.card.choose 3) in
  let odd_probability := odd_ways / total_ways in
  1 - odd_probability

-- The theorem to prove
theorem prob_even_product_eq_19_over_20 : even_product_probability = 19/20 :=
  sorry

end prob_even_product_eq_19_over_20_l311_311434


namespace solution_inequality_set_l311_311859

-- Define the inequality condition
def inequality (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

-- Define the interval solution set
def solution_set := Set.Icc (-2 : ℝ) 5

-- The statement that we want to prove
theorem solution_inequality_set : {x : ℝ | inequality x} = solution_set :=
  sorry

end solution_inequality_set_l311_311859


namespace student_score_in_first_subject_l311_311138

theorem student_score_in_first_subject 
  (x : ℝ)  -- Percentage in the first subject
  (w : ℝ)  -- Constant weight (as all subjects have same weight)
  (S2_score : ℝ)  -- Score in the second subject
  (S3_score : ℝ)  -- Score in the third subject
  (target_avg : ℝ) -- Target average score
  (hS2 : S2_score = 70)  -- Second subject score is 70%
  (hS3 : S3_score = 80)  -- Third subject score is 80%
  (havg : (x + S2_score + S3_score) / 3 = target_avg) :  -- The desired average is equal to the target average
  target_avg = 70 → x = 60 :=   -- Target average score is 70%
by
  sorry

end student_score_in_first_subject_l311_311138


namespace joan_gave_sam_seashells_l311_311393

-- Definitions of initial conditions
def initial_seashells : ℕ := 70
def remaining_seashells : ℕ := 27

-- Theorem statement
theorem joan_gave_sam_seashells : initial_seashells - remaining_seashells = 43 :=
by
  sorry

end joan_gave_sam_seashells_l311_311393


namespace no_solution_exists_l311_311372

theorem no_solution_exists : ¬ ∃ n : ℕ, 0 < n ∧ (2^n % 60 = 29 ∨ 2^n % 60 = 31) := 
by
  sorry

end no_solution_exists_l311_311372


namespace min_value_ineq_l311_311799

theorem min_value_ineq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_point_on_chord : ∃ x y : ℝ, x = 4 * a ∧ y = 2 * b ∧ (x + y = 2) ∧ (x^2 + y^2 = 4) ∧ ((x - 2)^2 + (y - 2)^2 = 4)) :
  1 / a + 2 / b ≥ 8 :=
by
  sorry

end min_value_ineq_l311_311799


namespace exists_integer_lt_sqrt_10_l311_311414

theorem exists_integer_lt_sqrt_10 : ∃ k : ℤ, k < Real.sqrt 10 := by
  have h_sqrt_bounds : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
    -- Proof involving basic properties and calculations
    sorry
  exact ⟨3, h_sqrt_bounds.left⟩

end exists_integer_lt_sqrt_10_l311_311414


namespace sin_593_l311_311925

theorem sin_593 (h : Real.sin (37 * Real.pi / 180) = 3/5) : 
  Real.sin (593 * Real.pi / 180) = -3/5 :=
by
sorry

end sin_593_l311_311925


namespace javier_first_throw_distance_l311_311832

noncomputable def javelin_first_throw_initial_distance (x : Real) : Real :=
  let throw1_adjusted := 2 * x * 0.95 - 2
  let throw2_adjusted := x * 0.92 - 4
  let throw3_adjusted := 4 * x - 1
  if (throw1_adjusted + throw2_adjusted + throw3_adjusted = 1050) then
    2 * x
  else
    0

theorem javier_first_throw_distance : ∃ x : Real, javelin_first_throw_initial_distance x = 310 :=
by
  sorry

end javier_first_throw_distance_l311_311832


namespace cannot_partition_nat_l311_311218

theorem cannot_partition_nat (A : ℕ → Set ℕ) (h1 : ∀ i j, i ≠ j → Disjoint (A i) (A j))
    (h2 : ∀ k, Finite (A k) ∧ sum {n | n ∈ A k}.toFinset id = k + 2013) :
    False :=
sorry

end cannot_partition_nat_l311_311218


namespace compare_fractions_l311_311323

theorem compare_fractions (a b c d : ℤ) (h1 : a = -(2)) (h2 : b = 3) (h3 : c = -(3)) (h4 : d = 5) :
  (a : ℚ) / b < c / d := 
by {
  simp [h1, h2, h3, h4],
  norm_num,
  simp [rat.lt_iff],
  exact lt_trans (neg_lt_zero.mpr (nat.cast_pos.mpr (show 3 * 5 > 0, by norm_num))) (show -(2 * 5 : ℤ) < -(3 * 3), by norm_num [lt_one_mul]),
}

end compare_fractions_l311_311323


namespace pentagonal_tiles_count_l311_311119

theorem pentagonal_tiles_count (t s p : ℕ) 
  (h1 : t + s + p = 30) 
  (h2 : 3 * t + 4 * s + 5 * p = 120) : 
  p = 10 := by
  sorry

end pentagonal_tiles_count_l311_311119


namespace width_of_rectangle_l311_311307

-- Define the given values
def length : ℝ := 2
def area : ℝ := 8

-- State the theorem
theorem width_of_rectangle : ∃ width : ℝ, area = length * width ∧ width = 4 :=
by
  -- The proof is omitted
  sorry

end width_of_rectangle_l311_311307


namespace maxim_is_correct_l311_311293

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end maxim_is_correct_l311_311293


namespace spider_has_eight_legs_l311_311505

-- Define the number of legs a human has
def human_legs : ℕ := 2

-- Define the number of legs for a spider, based on the given condition
def spider_legs : ℕ := 2 * (2 * human_legs)

-- The theorem to be proven, that the spider has 8 legs
theorem spider_has_eight_legs : spider_legs = 8 :=
by
  sorry

end spider_has_eight_legs_l311_311505


namespace non_congruent_triangles_with_perimeter_18_l311_311949

theorem non_congruent_triangles_with_perimeter_18 :
  {t : (ℕ × ℕ × ℕ) // t.1 + t.2 + t.3 = 18 ∧
                             t.1 + t.2 > t.3 ∧
                             t.1 + t.3 > t.2 ∧
                             t.2 + t.3 > t.1}.card = 9 :=
sorry

end non_congruent_triangles_with_perimeter_18_l311_311949


namespace S_3n_plus_1_l311_311635

noncomputable def S : ℕ → ℝ := sorry  -- S_n is the sum of the first n terms of the sequence {a_n}
noncomputable def a : ℕ → ℝ := sorry  -- Sequence {a_n}

-- Given conditions
axiom S3 : S 3 = 1
axiom S4 : S 4 = 11
axiom a_recurrence (n : ℕ) : a (n + 3) = 2 * a n

-- Define S_{3n+1} in terms of n
theorem S_3n_plus_1 (n : ℕ) : S (3 * n + 1) = 3 * 2^(n+1) - 1 :=
sorry

end S_3n_plus_1_l311_311635


namespace different_suits_choice_count_l311_311195

-- Definitions based on the conditions
def standard_deck : List (Card × Suit) := 
  List.product Card.all Suit.all

def four_cards (deck : List (Card × Suit)) : Prop :=
  deck.length = 4 ∧ ∀ (i j : Fin 4), i ≠ j → (deck.nthLe i (by simp) : Card × Suit).2 ≠ (deck.nthLe j (by simp) : Card × Suit).2

-- Statement of the proof problem
theorem different_suits_choice_count :
  ∃ l : List (Card × Suit), four_cards l ∧ standard_deck.choose 4 = 28561 :=
by
  sorry

end different_suits_choice_count_l311_311195


namespace log_product_evaluation_l311_311485

noncomputable def evaluate_log_product : ℝ :=
  Real.log 9 / Real.log 2 * Real.log 16 / Real.log 3 * Real.log 27 / Real.log 7

theorem log_product_evaluation : evaluate_log_product = 24 := 
  sorry

end log_product_evaluation_l311_311485


namespace value_of_t_l311_311500

theorem value_of_t (t : ℝ) (x y : ℝ) (h : 3 * x^(t-1) + y - 5 = 0) :
  t = 2 :=
sorry

end value_of_t_l311_311500


namespace sufficient_condition_l311_311677

theorem sufficient_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a = 0 → a < 1) ↔ 
  (∀ c : ℝ, x^2 - 2 * x + c = 0 ↔ 4 - 4 * c ≥ 0 ∧ c < 1 → ¬ (∀ d : ℝ, d ≤ 1 → d < 1)) := 
by 
sorry

end sufficient_condition_l311_311677


namespace diana_apollo_probability_l311_311481

theorem diana_apollo_probability :
  let outcomes := (6 * 6)
  let successful := (5 + 4 + 3 + 2 + 1)
  (successful / outcomes) = 5 / 12 := sorry

end diana_apollo_probability_l311_311481


namespace rhombus_area_three_times_diagonals_l311_311096

theorem rhombus_area_three_times_diagonals :
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  (new_d1 * new_d2) / 2 = 108 :=
by
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  have h : (new_d1 * new_d2) / 2 = 108 := sorry
  exact h

end rhombus_area_three_times_diagonals_l311_311096


namespace polygon_sides_from_diagonals_l311_311990

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l311_311990


namespace travel_distance_proof_l311_311314

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end travel_distance_proof_l311_311314


namespace number_of_hikers_in_the_morning_l311_311751

theorem number_of_hikers_in_the_morning (H : ℕ) :
  41 + 26 + H = 71 → H = 4 :=
by
  intros h_eq
  sorry

end number_of_hikers_in_the_morning_l311_311751


namespace cost_price_of_table_l311_311289

theorem cost_price_of_table 
  (SP : ℝ) 
  (CP : ℝ) 
  (h1 : SP = 1.24 * CP) 
  (h2 : SP = 8215) :
  CP = 6625 :=
by
  sorry

end cost_price_of_table_l311_311289


namespace log_product_l311_311743

theorem log_product :
  (Real.log 100 / Real.log 10) * (Real.log (1 / 10) / Real.log 10) = -2 := by
  sorry

end log_product_l311_311743


namespace original_oil_weight_is_75_l311_311118

def initial_oil_weight (original : ℝ) : Prop :=
  let first_remaining := original / 2
  let second_remaining := first_remaining * (4 / 5)
  second_remaining = 30

theorem original_oil_weight_is_75 : ∃ (original : ℝ), initial_oil_weight original ∧ original = 75 :=
by
  use 75
  unfold initial_oil_weight
  sorry

end original_oil_weight_is_75_l311_311118


namespace time_to_cross_platform_is_correct_l311_311460

noncomputable def speed_of_train := 36 -- speed in km/h
noncomputable def time_to_cross_pole := 12 -- time in seconds
noncomputable def time_to_cross_platform := 49.996960243180546 -- time in seconds

theorem time_to_cross_platform_is_correct : time_to_cross_platform = 49.996960243180546 := by
  sorry

end time_to_cross_platform_is_correct_l311_311460


namespace regular_polygon_sides_l311_311975

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l311_311975


namespace pumpkin_pie_filling_l311_311133

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end pumpkin_pie_filling_l311_311133


namespace sunset_time_l311_311537

def length_of_daylight_in_minutes := 11 * 60 + 12
def sunrise_time_in_minutes := 6 * 60 + 45
def sunset_time_in_minutes := sunrise_time_in_minutes + length_of_daylight_in_minutes
def sunset_time_hour := sunset_time_in_minutes / 60
def sunset_time_minute := sunset_time_in_minutes % 60
def sunset_time_12hr_format := if sunset_time_hour >= 12 
    then (sunset_time_hour - 12, sunset_time_minute)
    else (sunset_time_hour, sunset_time_minute)

theorem sunset_time : sunset_time_12hr_format = (5, 57) :=
by
  sorry

end sunset_time_l311_311537


namespace find_tangent_c_l311_311032

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → (c = 1) :=
by
  intros h
  sorry

end find_tangent_c_l311_311032


namespace sum_of_digits_of_4_plus_2_pow_21_l311_311101

theorem sum_of_digits_of_4_plus_2_pow_21 :
  let x := (4 + 2)
  (x^(21) % 100).div 10 + (x^(21) % 100).mod 10 = 6 :=
by
  let x := (4 + 2)
  sorry

end sum_of_digits_of_4_plus_2_pow_21_l311_311101


namespace quiz_passing_condition_l311_311076

theorem quiz_passing_condition (P Q : Prop) :
  (Q → P) → 
    (¬P → ¬Q) ∧ 
    (¬Q → ¬P) ∧ 
    (P → Q) :=
by sorry

end quiz_passing_condition_l311_311076


namespace polynomial_root_reciprocal_square_sum_l311_311016

theorem polynomial_root_reciprocal_square_sum :
  ∀ (a b c : ℝ), (a + b + c = 6) → (a * b + b * c + c * a = 11) → (a * b * c = 6) →
  (1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 = 49 / 36) :=
by
  intros a b c h_sum h_prod_sum h_prod
  sorry

end polynomial_root_reciprocal_square_sum_l311_311016


namespace greatest_multiple_of_5_and_7_less_than_800_l311_311869

theorem greatest_multiple_of_5_and_7_less_than_800 : 
    ∀ n : ℕ, (n < 800 ∧ 35 ∣ n) → n ≤ 770 := 
by
  -- Proof steps go here
  sorry

end greatest_multiple_of_5_and_7_less_than_800_l311_311869


namespace ali_babas_cave_min_moves_l311_311570

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l311_311570


namespace not_universally_better_l311_311533

-- Definitions based on the implicitly given conditions
def can_show_quantity (chart : Type) : Prop := sorry
def can_reflect_changes (chart : Type) : Prop := sorry

-- Definitions of bar charts and line charts
inductive BarChart
| mk : BarChart

inductive LineChart
| mk : LineChart

-- Assumptions based on characteristics of the charts
axiom bar_chart_shows_quantity : can_show_quantity BarChart 
axiom line_chart_shows_quantity : can_show_quantity LineChart 
axiom line_chart_reflects_changes : can_reflect_changes LineChart 

-- Proof problem statement
theorem not_universally_better : ¬(∀ (c1 c2 : Type), can_show_quantity c1 → can_reflect_changes c1 → ¬can_show_quantity c2 → ¬can_reflect_changes c2) :=
  sorry

end not_universally_better_l311_311533


namespace inequality_problem_l311_311846

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a))) ≥ (27 / (a + b + c)^2) :=
by
  sorry

end inequality_problem_l311_311846


namespace incorrect_statement_C_l311_311448

theorem incorrect_statement_C : 
  (∀ x : ℝ, |x| = x → x = 0 ∨ x = 1) ↔ False :=
by
  -- Proof goes here
  sorry

end incorrect_statement_C_l311_311448


namespace repaved_today_l311_311880

theorem repaved_today (total before : ℕ) (h_total : total = 4938) (h_before : before = 4133) : total - before = 805 := by
  sorry

end repaved_today_l311_311880


namespace die_face_never_touches_board_l311_311707

theorem die_face_never_touches_board : 
  ∃ (cube : Type) (roll : cube → cube) (occupied : Fin 8 × Fin 8 → cube → Prop),
    (∀ p : Fin 8 × Fin 8, ∃ c : cube, occupied p c) ∧ 
    (∃ f : cube, ¬ (∃ p : Fin 8 × Fin 8, occupied p f)) :=
by sorry

end die_face_never_touches_board_l311_311707


namespace number_of_odd_factors_of_360_l311_311658

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l311_311658


namespace cows_eat_grass_l311_311217

theorem cows_eat_grass (ha_per_cow_per_week : ℝ) (ha_grow_per_week : ℝ) :
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (2, 3, 2, 2) →
    (2 : ℝ) = 3 * 2 * ha_per_cow_per_week - 2 * ha_grow_per_week) → 
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (4, 2, 4, 2) →
    (2 : ℝ) = 2 * 4 * ha_per_cow_per_week - 4 * ha_grow_per_week) → 
  ∃ (cows : ℕ), (6 : ℝ) = cows * 6 * ha_per_cow_per_week - 6 * ha_grow_per_week ∧ cows = 3 :=
sorry

end cows_eat_grass_l311_311217


namespace sum_of_first_11_terms_l311_311387

theorem sum_of_first_11_terms (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + a 8 = 16) : (11 / 2) * (a 1 + a 11) = 88 :=
by
  sorry

end sum_of_first_11_terms_l311_311387


namespace original_rice_amount_l311_311521

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end original_rice_amount_l311_311521


namespace not_associative_star_l311_311344

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

theorem not_associative_star : ¬ (∀ x y z : ℝ, star (star x y) z = star x (star y z)) :=
by
  sorry

end not_associative_star_l311_311344


namespace B4E_base16_to_base10_l311_311017

theorem B4E_base16_to_base10 : 
  let B := 11
  let four := 4
  let E := 14
  (B * 16^2 + four * 16^1 + E * 16^0) = 2894 := 
by 
  let B := 11
  let four := 4
  let E := 14
  calc
    B * 16^2 + four * 16^1 + E * 16^0 = 11 * 256 + 4 * 16 + 14 : by rfl
    ... = 2816 + 64 + 14 : by rfl
    ... = 2894 : by rfl

end B4E_base16_to_base10_l311_311017


namespace cubic_solution_l311_311853

theorem cubic_solution (a b c : ℝ) (h_eq : ∀ x, x^3 - 4*x^2 + 7*x + 6 = 34 -> x = a ∨ x = b ∨ x = c)
(h_ge : a ≥ b ∧ b ≥ c) : 2 * a + b = 8 := 
sorry

end cubic_solution_l311_311853


namespace last_passenger_probability_l311_311085

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l311_311085


namespace profit_percent_300_l311_311110

theorem profit_percent_300 (SP : ℝ) (h : SP ≠ 0) (CP : ℝ) (h1 : CP = 0.25 * SP) : 
  (SP - CP) / CP * 100 = 300 := 
  sorry

end profit_percent_300_l311_311110


namespace trigonometric_identity_l311_311203

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + 2 * sin α * cos α) = 10 / 3 :=
sorry

end trigonometric_identity_l311_311203


namespace ways_to_choose_4_cards_of_different_suits_l311_311184

theorem ways_to_choose_4_cards_of_different_suits :
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  ∃ n : ℕ, n = (choose num_suits num_suits) * cards_per_suit ^ num_suits ∧ n = 28561 :=
by
  let deck_size := 52
  let num_suits := 4
  let cards_per_suit := 13
  have ways_to_choose_suits : (choose num_suits num_suits) = 1 := by simp
  have ways_to_choose_cards : cards_per_suit ^ num_suits = 28561 := by norm_num
  let n := 1 * 28561
  use n
  constructor
  · exact by simp [ways_to_choose_suits, ways_to_choose_cards]
  · exact by rfl

end ways_to_choose_4_cards_of_different_suits_l311_311184


namespace avg_income_pr_l311_311552

theorem avg_income_pr (P Q R : ℝ) 
  (h_avgPQ : (P + Q) / 2 = 5050) 
  (h_avgQR : (Q + R) / 2 = 6250)
  (h_P : P = 4000) 
  : (P + R) / 2 = 5200 := 
by 
  sorry

end avg_income_pr_l311_311552


namespace motorcycle_time_l311_311762

theorem motorcycle_time (v_m v_b d t_m : ℝ) 
  (h1 : 12 * v_m + 9 * v_b = d)
  (h2 : 21 * v_b + 8 * v_m = d)
  (h3 : v_m = 3 * v_b) :
  t_m = 15 :=
by
  sorry

end motorcycle_time_l311_311762


namespace max_value_ratio_l311_311638

theorem max_value_ratio (a b c: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_eq: a * (a + b + c) = b * c) :
  (a / (b + c) ≤ (Real.sqrt 2 - 1) / 2) :=
sorry -- proof omitted

end max_value_ratio_l311_311638


namespace problem_l311_311227

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem problem (A B : ℕ) (hA : A = gcf 9 15 27) (hB : B = lcm 9 15 27) : A + B = 138 :=
by
  sorry

end problem_l311_311227


namespace geometric_sequence_a7_l311_311691

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a1 : a 1 = 2) (h_a3 : a 3 = 4) : a 7 = 16 := 
sorry

end geometric_sequence_a7_l311_311691


namespace music_track_duration_l311_311891

theorem music_track_duration (minutes : ℝ) (seconds_per_minute : ℝ) (duration_in_minutes : minutes = 12.5) (seconds_per_minute_is_60 : seconds_per_minute = 60) : minutes * seconds_per_minute = 750 := by
  sorry

end music_track_duration_l311_311891


namespace non_congruent_triangles_perimeter_18_l311_311955

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l311_311955


namespace sum_of_solutions_of_fx_eq_0_l311_311068

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 7 * x + 10 else 3 * x - 15

theorem sum_of_solutions_of_fx_eq_0 :
  let x1 := -10 / 7
  let x2 := 5
  f x1 = 0 ∧ f x2 = 0 ∧ x1 ≤ 1 ∧ x2 > 1 → x1 + x2 = 25 / 7 :=
by
  sorry

end sum_of_solutions_of_fx_eq_0_l311_311068


namespace Kiarra_age_l311_311224

variable (Kiarra Bea Job Figaro Harry : ℕ)

theorem Kiarra_age 
  (h1 : Kiarra = 2 * Bea)
  (h2 : Job = 3 * Bea)
  (h3 : Figaro = Job + 7)
  (h4 : Harry = Figaro / 2)
  (h5 : Harry = 26) : 
  Kiarra = 30 := sorry

end Kiarra_age_l311_311224


namespace sequence_general_term_l311_311247

noncomputable def b_n (n : ℕ) : ℚ := 2 * n - 1
noncomputable def c_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem sequence_general_term (n : ℕ) : 
  b_n n + c_n n = (4 * n^2 + n - 1) / (2 * n + 1) :=
by sorry

end sequence_general_term_l311_311247


namespace linear_function_incorrect_conclusion_C_l311_311923

theorem linear_function_incorrect_conclusion_C :
  ∀ (x y : ℝ), (y = -2 * x + 4) → ¬(∃ x, y = 0 ∧ (x = 0 ∧ y = 4)) := by
  sorry

end linear_function_incorrect_conclusion_C_l311_311923


namespace non_congruent_triangles_with_perimeter_18_l311_311953

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l311_311953


namespace find_P_Q_l311_311025

noncomputable def P := 11 / 3
noncomputable def Q := -2 / 3

theorem find_P_Q :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
    (3 * x + 12) / (x ^ 2 - 5 * x - 14) = P / (x - 7) + Q / (x + 2) :=
by
  intros x hx1 hx2
  dsimp [P, Q]  -- Unfold the definitions of P and Q
  -- The actual proof would go here, but we are skipping it
  sorry

end find_P_Q_l311_311025


namespace count_odd_factors_of_360_l311_311660

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l311_311660


namespace find_a10_l311_311347

theorem find_a10 (a : ℕ → ℝ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a n - a (n+1) = a n * a (n+1)) : 
  a 10 = 1 / 10 :=
sorry

end find_a10_l311_311347


namespace chess_team_boys_l311_311300

variable {B G : ℕ}

theorem chess_team_boys
    (h1 : B + G = 30)
    (h2 : 1/3 * G + B = 18) :
    B = 12 :=
by
  sorry

end chess_team_boys_l311_311300


namespace sandrine_washed_160_dishes_l311_311724

-- Define the number of pears picked by Charles
def charlesPears : ℕ := 50

-- Define the number of bananas cooked by Charles as 3 times the number of pears he picked
def charlesBananas : ℕ := 3 * charlesPears

-- Define the number of dishes washed by Sandrine as 10 more than the number of bananas Charles cooked
def sandrineDishes : ℕ := charlesBananas + 10

-- Prove that Sandrine washed 160 dishes
theorem sandrine_washed_160_dishes : sandrineDishes = 160 := by
  -- The proof is omitted
  sorry

end sandrine_washed_160_dishes_l311_311724


namespace planks_needed_l311_311629

theorem planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (h1 : total_nails = 4) (h2 : nails_per_plank = 2) : total_nails / nails_per_plank = 2 :=
by
  -- Prove that given the conditions, the required result is obtained
  sorry

end planks_needed_l311_311629


namespace correct_props_l311_311403

-- Definitions
variables {α : Type*} [affine_space α ℝ] (a b : ℝ) (plane : set α)

-- Propose the conditions using Lean types and variables
def parallel (a b : ℝ) : Prop := ∀ (x y : α), x ∈ line a → y ∈ line b → x -ᵥ y ∈ plane
def perpendicular (a : ℝ) (plane : set α) : Prop := ∀ (x : α), x ∈ line a → x ∈ plane
def in_plane (a : α) (plane : set α) : Prop := a ∈ plane

-- Proof problem
theorem correct_props (a b : ℝ) (plane : set α) :
  (parallel a b → perpendicular a plane → perpendicular b plane) ∧
  (perpendicular a plane → perpendicular b plane → parallel a b) :=
by sorry

end correct_props_l311_311403


namespace problem_conditions_imply_options_l311_311630

theorem problem_conditions_imply_options (a b : ℝ) 
  (h1 : a + 1 > b) 
  (h2 : b > 2 / a) 
  (h3 : 2 / a > 0) : 
  (a = 2 ∧ a + 1 > 2 / a ∧ b > 2 / 2) ∨
  (a = 1 → a + 1 ≤ 2 / a) ∨
  (b = 1 → ∃ a, a > 1 ∧ a + 1 > 1 ∧ 1 > 2 / a) ∨
  (a * b = 1 → ab ≤ 2) := 
sorry

end problem_conditions_imply_options_l311_311630


namespace simplify_expression_l311_311063

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem simplify_expression (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = a + b) :
  (a / b) + (b / a) - (1 / (a * b)) = 1 :=
by sorry

end simplify_expression_l311_311063


namespace total_tissues_used_l311_311778

-- Definitions based on the conditions
def initial_tissues := 97
def remaining_tissues := 47
def alice_tissues := 12
def bob_tissues := 2 * alice_tissues
def eve_tissues := alice_tissues - 3
def carol_tissues := initial_tissues - remaining_tissues
def friends_tissues := alice_tissues + bob_tissues + eve_tissues

-- The theorem to prove
theorem total_tissues_used : carol_tissues + friends_tissues = 95 := sorry

end total_tissues_used_l311_311778


namespace different_suits_card_combinations_l311_311199

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l311_311199


namespace non_congruent_triangles_count_l311_311963

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_18 (a b c : ℕ) : Prop :=
  a + b + c = 18

def is_non_congruent (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a = d ∧ b = e ∧ c = f) ∨
  (a = d ∧ b = f ∧ c = e) ∨
  (a = e ∧ b = d ∧ c = f) ∨
  (a = e ∧ b = f ∧ c = d) ∨
  (a = f ∧ b = d ∧ c = e) ∨
  (a = f ∧ b = e ∧ c = d)

def count_non_congruent_triangles : Prop :=
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 11 ∧ 
  (∀ (p ∈ S), ∃ a b c, p = (a, b, c) ∧ is_valid_triangle a b c ∧ perimeter_18 a b c) ∧
  (∀ (a b c : ℕ), is_valid_triangle a b c → perimeter_18 a b c → 
    ∃ (p ∈ S), p = (a, b, c) ∨
    ∃ (q ∈ S), is_non_congruent a b c (q.1, q.2.1, q.2.2))

theorem non_congruent_triangles_count : count_non_congruent_triangles :=
by
  sorry

end non_congruent_triangles_count_l311_311963


namespace probability_at_least_one_three_l311_311440

theorem probability_at_least_one_three :
  let E := { (d1, d2) : Fin 8 × Fin 8 | d1 = 2 ∨ d2 = 2 } in
  (↑E.card / ↑((Fin 8 × Fin 8).card) : ℚ) = 15 / 64 :=
by
  /- Let E be the set of outcomes where at least one die shows a 3. -/
  sorry

end probability_at_least_one_three_l311_311440


namespace find_a_for_perpendicular_lines_l311_311937

theorem find_a_for_perpendicular_lines (a : ℝ) 
    (h_perpendicular : 2 * a + (-1) * (3 - a) = 0) :
    a = 1 :=
by
  sorry

end find_a_for_perpendicular_lines_l311_311937


namespace flood_damage_conversion_l311_311760

-- Define the conversion rate and the damage in Indian Rupees as given
def rupees_to_pounds (rupees : ℕ) : ℕ := rupees / 75
def damage_in_rupees : ℕ := 45000000

-- Define the expected damage in British Pounds
def expected_damage_in_pounds : ℕ := 600000

-- The theorem to prove that the damage in British Pounds is as expected, given the conditions.
theorem flood_damage_conversion :
  rupees_to_pounds damage_in_rupees = expected_damage_in_pounds :=
by
  -- The proof goes here, but we'll use sorry to skip it as instructed.
  sorry

end flood_damage_conversion_l311_311760


namespace range_of_a_l311_311378

theorem range_of_a (a : ℝ) :
  (1 < a ∧ a < 8 ∧ a ≠ 4) ↔
  (a > 1 ∧ a < 8) ∧ (a > -4 ∧ a ≠ 4) :=
by sorry

end range_of_a_l311_311378


namespace sides_of_regular_polygon_with_20_diagonals_l311_311999

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l311_311999


namespace unique_triangles_count_l311_311942

noncomputable def number_of_non_congruent_triangles_with_perimeter_18 : ℕ :=
  sorry

theorem unique_triangles_count :
  number_of_non_congruent_triangles_with_perimeter_18 = 11 := 
  by {
    sorry
}

end unique_triangles_count_l311_311942


namespace slope_of_PQ_l311_311360

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

theorem slope_of_PQ :
  ∃ P Q : ℝ × ℝ,
    P = (0, 0) ∧ Q = (1, 8 / 3) ∧
    (∃ m : ℝ,
      m = 2 * Real.cos 0 ∧
      m = Real.sqrt 1 + 1 / Real.sqrt 1) ∧
    (Q.snd - P.snd) / (Q.fst - P.fst) = 8 / 3 :=
by
  sorry

end slope_of_PQ_l311_311360


namespace count_odd_factors_of_360_l311_311661

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l311_311661


namespace polygon_diagonals_l311_311982

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l311_311982


namespace area_of_rectangle_abcd_l311_311325

-- Definition of the problem's conditions and question
def small_square_side_length : ℝ := 1
def large_square_side_length : ℝ := 1.5
def area_rectangle_abc : ℝ := 4.5

-- Lean 4 statement: Prove the area of rectangle ABCD is 4.5 square inches
theorem area_of_rectangle_abcd :
  (3 * small_square_side_length) * large_square_side_length = area_rectangle_abc :=
by
  sorry

end area_of_rectangle_abcd_l311_311325


namespace tan_half_angle_inequality_l311_311844

theorem tan_half_angle_inequality (a b c : ℝ) (α β : ℝ)
  (h : a + b < 3 * c)
  (h_tan_identity : Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c)) :
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by
  sorry

end tan_half_angle_inequality_l311_311844


namespace triangle_area_of_tangent_line_l311_311206

theorem triangle_area_of_tangent_line (a : ℝ) 
  (h : a > 0) 
  (ha : (1/2) * 3 * a * (3 / (2 * a ^ (1/2))) = 18)
  : a = 64 := 
sorry

end triangle_area_of_tangent_line_l311_311206


namespace sum_x_coordinates_eq_3_l311_311089

def f : ℝ → ℝ := sorry -- definition of the function f as given by the five line segments

theorem sum_x_coordinates_eq_3 :
  (∃ x1 x2 x3 : ℝ, (f x1 = x1 + 1 ∧ f x2 = x2 + 1 ∧ f x3 = x3 + 1) ∧ (x1 + x2 + x3 = 3)) :=
sorry

end sum_x_coordinates_eq_3_l311_311089


namespace max_difference_is_62_l311_311443

open Real

noncomputable def max_difference_of_integers : ℝ :=
  let a (k : ℝ) := 2 * k + 1 + sqrt (8 * k)
  let b (k : ℝ) := 2 * k + 1 - sqrt (8 * k)
  let diff (k : ℝ) := a k - b k
  let max_k := 120 -- Maximum integer value k such that 2k + 1 + sqrt(8k) < 1000
  diff max_k

theorem max_difference_is_62 :
  max_difference_of_integers = 62 :=
sorry

end max_difference_is_62_l311_311443


namespace intersecting_lines_l311_311256

theorem intersecting_lines (m b : ℝ)
  (h1 : ∀ x, (9 : ℝ) = 2 * m * x + 3 → x = 3)
  (h2 : ∀ x, (9 : ℝ) = 4 * x + b → x = 3) :
  b + 2 * m = -1 :=
sorry

end intersecting_lines_l311_311256


namespace driver_travel_distance_per_week_l311_311124

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end driver_travel_distance_per_week_l311_311124


namespace gemstone_necklaces_count_l311_311024

-- Conditions
def num_bead_necklaces : ℕ := 3
def price_per_necklace : ℕ := 7
def total_earnings : ℕ := 70

-- Proof Problem
theorem gemstone_necklaces_count : (total_earnings - num_bead_necklaces * price_per_necklace) / price_per_necklace = 7 := by
  sorry

end gemstone_necklaces_count_l311_311024


namespace non_congruent_triangles_with_perimeter_18_l311_311952

theorem non_congruent_triangles_with_perimeter_18 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b c : ℕ), a + b + c = 18 → 
    a < b + c ∧ b < a + c ∧ c < a + b → 
    (a, b, c).perm = (b, c, a) ∨ (a, b, c).perm = (c, a, b) ∨ (a, b, c).perm = (a, c, b) ∨ 
    (a, b, c).perm = (b, a, c) ∨ (a, b, c).perm = (c, b, a) ∨ (a, b, c).perm = (a, b, c)) :=
sorry

end non_congruent_triangles_with_perimeter_18_l311_311952


namespace numbers_divisible_l311_311040

theorem numbers_divisible (n : ℕ) (d1 d2 : ℕ) (lcm_d1_d2 : ℕ) (limit : ℕ) (h_lcm: lcm d1 d2 = lcm_d1_d2) (h_limit : limit = 2011)
(h_d1 : d1 = 117) (h_d2 : d2 = 2) : 
  ∃ k : ℕ, k = 8 ∧ ∀ m : ℕ, m < limit → (m % lcm_d1_d2 = 0 ↔ ∃ i : ℕ, i < k ∧ m = lcm_d1_d2 * (i + 1)) :=
by
  sorry

end numbers_divisible_l311_311040


namespace max_value_g_l311_311152

def g (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_value_g : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2 ∧ ∀ y : ℝ, (0 ≤ y ∧ y ≤ 2) → g y ≤ g x) ∧ g x = 3 :=
by
  sorry

end max_value_g_l311_311152


namespace solve_equation_l311_311242

theorem solve_equation : ∀ x : ℝ, ((1 - x) / (x - 4)) + (1 / (4 - x)) = 1 → x = 2 :=
by
  intros x h
  sorry

end solve_equation_l311_311242


namespace carrie_profit_l311_311609

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end carrie_profit_l311_311609


namespace canoe_kayak_rental_l311_311444

theorem canoe_kayak_rental:
  ∀ (C K : ℕ), 
    12 * C + 18 * K = 504 → 
    C = (3 * K) / 2 → 
    C - K = 7 :=
  by
    intro C K
    intros h1 h2
    sorry

end canoe_kayak_rental_l311_311444


namespace cylinder_lateral_surface_area_l311_311558
noncomputable def lateralSurfaceArea (S : ℝ) : ℝ :=
  let l := Real.sqrt S
  let d := l
  let r := d / 2
  let h := l
  2 * Real.pi * r * h

theorem cylinder_lateral_surface_area (S : ℝ) (hS : S ≥ 0) : 
  lateralSurfaceArea S = Real.pi * S := by
  sorry

end cylinder_lateral_surface_area_l311_311558


namespace total_number_of_people_l311_311260

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l311_311260


namespace find_m_l311_311402

-- Definitions for the conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_of_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = a 1 * (1 - (a n / a 1)) / (1 - (a 2 / a 1))

def arithmetic_sequence (S3 S9 S6 : ℝ) :=
  2 * S9 = S3 + S6

def condition_3 (a : ℕ → ℝ) (m : ℕ) :=
  a 2 + a 5 = 2 * a m

-- Lean 4 statement that requires proof
theorem find_m 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ) 
  (geom_seq : geometric_sequence a a1 q)
  (sum_geom_seq : sum_of_geometric_sequence S a)
  (arith_seq : arithmetic_sequence (S 3) (S 9) (S 6))
  (cond3 : condition_3 a 8) : 
  8 = 8 := 
sorry

end find_m_l311_311402


namespace isosceles_vertex_angle_l311_311384

-- Let T be a type representing triangles, with a function base_angle returning the degree of a base angle,
-- and vertex_angle representing the degree of the vertex angle.
axiom Triangle : Type
axiom is_isosceles (t : Triangle) : Prop
axiom base_angle_deg (t : Triangle) : ℝ
axiom vertex_angle_deg (t : Triangle) : ℝ

theorem isosceles_vertex_angle (t : Triangle) (h_isosceles : is_isosceles t)
  (h_base_angle : base_angle_deg t = 50) : vertex_angle_deg t = 80 := by
  sorry

end isosceles_vertex_angle_l311_311384


namespace quadratic_no_real_roots_l311_311744

theorem quadratic_no_real_roots : ∀ (a b c : ℝ), a ≠ 0 → Δ = (b*b - 4*a*c) → x^2 + 3 = 0 → Δ < 0 := by
  sorry

end quadratic_no_real_roots_l311_311744


namespace sum_of_squares_of_roots_eq_213_l311_311030

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end sum_of_squares_of_roots_eq_213_l311_311030


namespace simplify_and_evaluate_expression_l311_311714

theorem simplify_and_evaluate_expression (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) : 
  1 - (a^2 + 2 * a * b + b^2) / (a^2 - a * b) / ((a + b) / (a - b)) = -1 := 
sorry

end simplify_and_evaluate_expression_l311_311714


namespace candies_per_pack_l311_311143

-- Conditions in Lean:
def total_candies : ℕ := 60
def packs_initially (packs_after : ℕ) : ℕ := packs_after + 1
def packs_after : ℕ := 2
def pack_count : ℕ := packs_initially packs_after

-- The statement of the proof problem:
theorem candies_per_pack : 
  total_candies / pack_count = 20 :=
by
  sorry

end candies_per_pack_l311_311143


namespace count_triangles_with_positive_area_l311_311675

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l311_311675


namespace euler_quadrilateral_theorem_l311_311695

theorem euler_quadrilateral_theorem (A1 A2 A3 A4 P Q : ℝ) 
  (midpoint_P : P = (A1 + A3) / 2)
  (midpoint_Q : Q = (A2 + A4) / 2) 
  (length_A1A2 length_A2A3 length_A3A4 length_A4A1 length_A1A3 length_A2A4 length_PQ : ℝ)
  (h1 : length_A1A2 = A1A2) (h2 : length_A2A3 = A2A3)
  (h3 : length_A3A4 = A3A4) (h4 : length_A4A1 = A4A1)
  (h5 : length_A1A3 = A1A3) (h6 : length_A2A4 = A2A4)
  (h7 : length_PQ = PQ) :
  length_A1A2^2 + length_A2A3^2 + length_A3A4^2 + length_A4A1^2 = 
  length_A1A3^2 + length_A2A4^2 + 4 * length_PQ^2 := sorry

end euler_quadrilateral_theorem_l311_311695


namespace odd_and_even_derivative_behavior_l311_311355

theorem odd_and_even_derivative_behavior 
  (f g : ℝ → ℝ)
  (Hf_odd : ∀ x, f (-x) = -f x)
  (Hg_even : ∀ x, g (-x) = g x)
  (H_f_deriv_neg : ∀ x, x < 0 → deriv f x > 0)
  (H_g_deriv_neg : ∀ x, x < 0 → deriv g x < 0) :
  (∀ x, 0 < x → deriv f x > 0) ∧ (∀ x, 0 < x → deriv g x > 0) := 
sorry

end odd_and_even_derivative_behavior_l311_311355


namespace find_a12_l311_311634

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- The Lean statement for the problem
theorem find_a12 (h_seq : arithmetic_sequence a d)
  (h_cond1 : a 7 + a 9 = 16) (h_cond2 : a 4 = 1) : 
  a 12 = 15 :=
sorry

end find_a12_l311_311634


namespace perfect_square_polynomial_l311_311577

-- Define the polynomial and the conditions
def polynomial (a b : ℚ) := fun x : ℚ => x^4 + x^3 + 2 * x^2 + a * x + b

-- The expanded form of a quadratic trinomial squared
def quadratic_square (p q : ℚ) := fun x : ℚ =>
  x^4 + 2 * p * x^3 + (p^2 + 2 * q) * x^2 + 2 * p * q * x + q^2

-- Main theorem statement
theorem perfect_square_polynomial :
  ∃ (a b : ℚ), 
  (∀ x : ℚ, polynomial a b x = (quadratic_square (1/2 : ℚ) (7/8 : ℚ) x)) ↔ 
  a = 7/8 ∧ b = 49/64 :=
by
  sorry

end perfect_square_polynomial_l311_311577


namespace evaporation_period_l311_311299

theorem evaporation_period
  (initial_amount : ℚ)
  (evaporation_rate : ℚ)
  (percentage_evaporated : ℚ)
  (actual_days : ℚ)
  (h_initial : initial_amount = 10)
  (h_evap_rate : evaporation_rate = 0.007)
  (h_percentage : percentage_evaporated = 3.5000000000000004)
  (h_days : actual_days = (percentage_evaporated / 100) * initial_amount / evaporation_rate):
  actual_days = 50 := by
  sorry

end evaporation_period_l311_311299


namespace parabola_intersection_probability_correct_l311_311573

noncomputable def parabola_intersection_probability : ℚ := sorry

theorem parabola_intersection_probability_correct :
  parabola_intersection_probability = 209 / 216 := sorry

end parabola_intersection_probability_correct_l311_311573


namespace solution_set_M_abs_ineq_l311_311643

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- The first statement to prove the solution set M for the inequality
theorem solution_set_M : ∀ x, f x < 3 ↔ x ∈ M :=
by sorry

-- The second statement to prove the inequality when a, b ∈ M
theorem abs_ineq (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + ab| :=
by sorry

end solution_set_M_abs_ineq_l311_311643


namespace seven_pow_l311_311748

theorem seven_pow (k : ℕ) (h : 7 ^ k = 2) : 7 ^ (4 * k + 2) = 784 :=
by 
  sorry

end seven_pow_l311_311748


namespace ordered_pairs_count_l311_311368

theorem ordered_pairs_count : 
  ∃ pairs : Set (ℝ × ℤ), (∀ p ∈ pairs, 0 < p.1 ∧ 3 ≤ p.2 ∧ p.2 ≤ 300 ∧ (Real.log p.1 / Real.log p.2) ^ 101 = Real.log (p.1 ^ 101) / Real.log p.2) ∧ pairs.card = 894 :=
sorry

end ordered_pairs_count_l311_311368


namespace choose_4_cards_of_different_suits_l311_311188

theorem choose_4_cards_of_different_suits :
  (∃ (n : ℕ), choose 4 4 = n) ∧
  (∃ (m : ℕ), (13^4 = m)) ∧
  (1 * (13^4) = 28561)

end choose_4_cards_of_different_suits_l311_311188


namespace rolling_cube_dot_path_l311_311454

theorem rolling_cube_dot_path (a b c : ℝ) (h_edge : a = 1) (h_dot_top : True):
  c = (1 + Real.sqrt 5) / 2 := by
  sorry

end rolling_cube_dot_path_l311_311454


namespace analogous_to_tetrahedron_is_triangle_l311_311104

-- Define the objects as types
inductive Object
| Quadrilateral
| Pyramid
| Triangle
| Prism
| Tetrahedron

-- Define the analogous relationship
def analogous (a b : Object) : Prop :=
  (a = Object.Tetrahedron ∧ b = Object.Triangle)
  ∨ (b = Object.Tetrahedron ∧ a = Object.Triangle)

-- The main statement to prove
theorem analogous_to_tetrahedron_is_triangle :
  ∃ (x : Object), analogous Object.Tetrahedron x ∧ x = Object.Triangle :=
by
  sorry

end analogous_to_tetrahedron_is_triangle_l311_311104


namespace slope_of_line_l311_311021

theorem slope_of_line (x y : ℝ) : 
  3 * y + 9 = -6 * x - 15 → 
  ∃ m b, y = m * x + b ∧ m = -2 := 
by {
  sorry
}

end slope_of_line_l311_311021


namespace find_m_l311_311516

theorem find_m (m : ℤ) (h1 : m + 1 ≠ 0) (h2 : m^2 + 3 * m + 1 = -1) : m = -2 := 
by 
  sorry

end find_m_l311_311516


namespace lattice_points_on_hyperbola_l311_311815

-- Define the problem
def countLatticePoints (n : ℤ) : ℕ :=
  let factoredCount := (2 + 1) * (2 + 1) * (4 + 1) -- Number of divisors of 2^2 * 3^2 * 5^4
  2 * factoredCount -- Each pair has two solutions considering positive and negative values

-- The theorem to be proven
theorem lattice_points_on_hyperbola : countLatticePoints 1800 = 90 := sorry

end lattice_points_on_hyperbola_l311_311815


namespace different_suits_card_combinations_l311_311202

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l311_311202


namespace find_pairs_l311_311623

theorem find_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m^2 + n^2) ∣ (3 * m * n + 3 * m) ↔ (m, n) = (1, 1) ∨ (m, n) = (4, 2) ∨ (m, n) = (4, 10) :=
sorry

end find_pairs_l311_311623


namespace second_pipe_filling_time_l311_311411

theorem second_pipe_filling_time :
  ∃ T : ℝ, (1/20 + 1/T) * 2/3 * 16 = 1 ∧ T = 160/7 :=
by
  use 160 / 7
  sorry

end second_pipe_filling_time_l311_311411


namespace find_k_for_quadratic_root_l311_311164

theorem find_k_for_quadratic_root (k : ℝ) (h : (1 : ℝ).pow 2 + k * 1 - 3 = 0) : k = 2 :=
by
  sorry

end find_k_for_quadratic_root_l311_311164


namespace area_union_of_reflected_triangles_l311_311310

def point : Type := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def reflect_y_eq_1 (P : point) : point := (P.1, 2 * 1 - P.2)

def area_of_union (A B C : point) (f : point → point) : ℝ :=
  let A' := f A
  let B' := f B
  let C' := f C
  triangle_area A B C + triangle_area A' B' C'

theorem area_union_of_reflected_triangles :
  area_of_union (3, 4) (5, -2) (6, 2) reflect_y_eq_1 = 11 :=
  sorry

end area_union_of_reflected_triangles_l311_311310


namespace total_problems_is_correct_l311_311693

/-- Definition of the number of pages of math homework. -/
def math_pages : ℕ := 2

/-- Definition of the number of pages of reading homework. -/
def reading_pages : ℕ := 4

/-- Definition that each page of homework contains 5 problems. -/
def problems_per_page : ℕ := 5

/-- The proof statement: given the number of pages of math and reading homework,
    and the number of problems per page, prove that the total number of problems is 30. -/
theorem total_problems_is_correct : (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end total_problems_is_correct_l311_311693


namespace maximum_value_of_expression_l311_311064

theorem maximum_value_of_expression
  (a b c : ℝ)
  (h1 : 0 ≤ a)
  (h2 : 0 ≤ b)
  (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + 2 * c^2 = 1) :
  ab * Real.sqrt 3 + 3 * bc ≤ Real.sqrt 7 :=
sorry

end maximum_value_of_expression_l311_311064


namespace probability_of_letter_in_mathematics_l311_311968

theorem probability_of_letter_in_mathematics :
  let distinct_letters_in_mathematics := 8
  let total_letters_in_alphabet := 26
  distinct_letters_in_mathematics.to_rat / total_letters_in_alphabet.to_rat = 4 / 13 :=
by
  sorry

end probability_of_letter_in_mathematics_l311_311968


namespace evaluate_expression_l311_311621

theorem evaluate_expression : 27^(- (2 / 3 : ℝ)) + Real.log 4 / Real.log 8 = 7 / 9 :=
by
  sorry

end evaluate_expression_l311_311621


namespace factor_expression_l311_311492

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l311_311492


namespace probability_of_drawing_white_ball_l311_311211

-- Define initial conditions
def initial_balls : ℕ := 6
def total_balls_after_white : ℕ := initial_balls + 1
def number_of_white_balls : ℕ := 1
def number_of_total_balls : ℕ := total_balls_after_white

-- Define the probability of drawing a white ball
def probability_of_white : ℚ := number_of_white_balls / number_of_total_balls

-- Statement to be proved
theorem probability_of_drawing_white_ball :
  probability_of_white = 1 / 7 :=
by
  sorry

end probability_of_drawing_white_ball_l311_311211


namespace lattice_points_on_hyperbola_l311_311808

theorem lattice_points_on_hyperbola : 
  let hyperbola_eq := λ x y : ℤ, x^2 - y^2 = 1800^2 in
  (∃ (x y : ℤ), hyperbola_eq x y) ∧ 
  ∃ (n : ℕ), n = 54 :=
by
  sorry

end lattice_points_on_hyperbola_l311_311808


namespace count_square_of_integer_fraction_l311_311918

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l311_311918


namespace value_of_difference_power_l311_311834

theorem value_of_difference_power (a b : ℝ) (h₁ : a^3 - 6 * a^2 + 15 * a = 9) 
                                  (h₂ : b^3 - 3 * b^2 + 6 * b = -1) 
                                  : (a - b)^2014 = 1 := 
by sorry

end value_of_difference_power_l311_311834


namespace expected_value_is_minus_0_point_38_l311_311408

-- Define the values for each roll
def roll_values (n : ℕ) : ℝ :=
  if n = 2 then 2 else
  if n = 3 then 3 else
  if n = 4 then -4 else
  if n = 8 then -8 else
  if n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 then 0 else
  0

-- Define the probability mass function for a fair 8-sided die
noncomputable def die_pmf : PMF (Fin 8) :=
  PMF.uniform_of_fin (Fin 8)

-- Define the expected value in terms of die rolls
noncomputable def expected_value : ℝ :=
  ∑ n in Finset.univ, (die_pmf n) * roll_values n.val

theorem expected_value_is_minus_0_point_38 :
  expected_value = -0.38 :=
by
  -- Begin your proof here (skipping for the purpose)
  sorry

end expected_value_is_minus_0_point_38_l311_311408


namespace diameter_of_circle_A_l311_311015

theorem diameter_of_circle_A
  (diameter_B : ℝ)
  (r : ℝ)
  (h1 : diameter_B = 16)
  (h2 : r^2 = (r / 8)^2 * 4):
  2 * (r / 2) = 8 :=
by
  sorry

end diameter_of_circle_A_l311_311015


namespace not_square_n5_plus_7_l311_311240

theorem not_square_n5_plus_7 (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, k^2 = n^5 + 7 := 
by
  sorry

end not_square_n5_plus_7_l311_311240


namespace john_total_cost_after_discount_l311_311458

/-- A store gives a 10% discount for the amount of the sell that was over $1000.
John buys 7 items that each cost $200. What does his order cost after the discount? -/
theorem john_total_cost_after_discount : 
  let discount_rate := 0.1
  let threshold := 1000
  let item_cost := 200
  let item_count := 7
  let total_cost := item_cost * item_count
  let discount := discount_rate * max 0 (total_cost - threshold)
  let final_cost := total_cost - discount
  in final_cost = 1360 :=
by 
  sorry

end john_total_cost_after_discount_l311_311458


namespace total_votes_l311_311053

theorem total_votes (V : ℕ) (h1 : ∃ c : ℕ, c = 84) (h2 : ∃ m : ℕ, m = 476) (h3 : ∃ d : ℕ, d = ((84 * V - 16 * V) / 100)) : 
  V = 700 := 
by 
  sorry 

end total_votes_l311_311053


namespace arithmetic_seq_min_S19_l311_311348

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_seq_min_S19
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_S8 : S a 8 ≤ 6)
  (h_S11 : S a 11 ≥ 27) :
  S a 19 ≥ 133 :=
sorry

end arithmetic_seq_min_S19_l311_311348


namespace no_partition_possible_l311_311219

noncomputable def partition_possible (A : ℕ → Set ℕ) :=
  (∀ k: ℕ, ∃ finA : Finset ℕ, (A k = finA.to_set) ∧ (finA.sum id = k + 2013)) ∧
  (∀ i j: ℕ, i ≠ j → (A i ∩ A j) = ∅) ∧
  (⋃ i, A i) = Set.univ

theorem no_partition_possible :
  ¬ ∃ A : ℕ → Set ℕ, partition_possible A := 
sorry

end no_partition_possible_l311_311219


namespace shape_described_by_constant_phi_is_cone_l311_311341

-- Definition of spherical coordinates
-- (ρ, θ, φ) where ρ is the radial distance,
-- θ is the azimuthal angle, and φ is the polar angle.
structure SphericalCoordinates :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- The condition that φ is equal to a constant d
def satisfies_condition (p : SphericalCoordinates) (d : ℝ) : Prop :=
  p.φ = d

-- The main theorem to prove
theorem shape_described_by_constant_phi_is_cone (d : ℝ) :
  ∃ (S : Set SphericalCoordinates), (∀ p ∈ S, satisfies_condition p d) ∧
  (∀ p, satisfies_condition p d → ∃ ρ θ, p = ⟨ρ, θ, d⟩) ∧
  (∀ ρ θ, ρ > 0 → θ ∈ [0, 2 * Real.pi] → SphericalCoordinates.mk ρ θ d ∈ S) :=
sorry

end shape_described_by_constant_phi_is_cone_l311_311341


namespace product_of_midpoint_l311_311489

-- Define the coordinates of the endpoints
def x1 := 5
def y1 := -4
def x2 := 1
def y2 := 14

-- Define the formulas for the midpoint coordinates
def xm := (x1 + x2) / 2
def ym := (y1 + y2) / 2

-- Define the product of the midpoint coordinates
def product := xm * ym

-- Now state the theorem
theorem product_of_midpoint :
  product = 15 := 
by
  -- Optional: detailed steps can go here if necessary
  sorry

end product_of_midpoint_l311_311489


namespace maximum_value_of_n_l311_311524

noncomputable def max_n (a b c : ℝ) (n : ℕ) :=
  a > b ∧ b > c ∧ (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2

theorem maximum_value_of_n (a b c : ℝ) (n : ℕ) : 
  a > b → b > c → (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2 :=
  by sorry

end maximum_value_of_n_l311_311524


namespace problem_statement_l311_311373

variables {Line Plane : Type}
variables {m n : Line} {alpha beta : Plane}

-- Define parallel and perpendicular relations
def parallel (l1 l2 : Line) : Prop := sorry
def perp (l : Line) (p : Plane) : Prop := sorry

-- Define that m and n are different lines
axiom diff_lines (m n : Line) : m ≠ n 

-- Define that alpha and beta are different planes
axiom diff_planes (alpha beta : Plane) : alpha ≠ beta

-- Statement to prove: If m ∥ n and m ⟂ α, then n ⟂ α
theorem problem_statement (h1 : parallel m n) (h2 : perp m alpha) : perp n alpha := 
sorry

end problem_statement_l311_311373


namespace expected_worth_coin_flip_l311_311774

noncomputable def expected_worth : ℝ := 
  (1 / 3) * 6 + (2 / 3) * (-2) - 1

theorem expected_worth_coin_flip : expected_worth = -0.33 := 
by 
  unfold expected_worth
  norm_num
  sorry

end expected_worth_coin_flip_l311_311774


namespace total_candies_correct_l311_311013

-- Define the number of candies each has
def caleb_jellybeans := 3 * 12
def caleb_chocolate_bars := 5
def caleb_gummy_bears := 8
def caleb_total := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears

def sophie_jellybeans := (caleb_jellybeans / 2)
def sophie_chocolate_bars := 3
def sophie_gummy_bears := 12
def sophie_total := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears

def max_jellybeans := (2 * 12) + sophie_jellybeans
def max_chocolate_bars := 6
def max_gummy_bears := 10
def max_total := max_jellybeans + max_chocolate_bars + max_gummy_bears

-- Define the total number of candies
def total_candies := caleb_total + sophie_total + max_total

-- Theorem statement
theorem total_candies_correct : total_candies = 140 := by
  sorry

end total_candies_correct_l311_311013


namespace shapes_identification_l311_311153

theorem shapes_identification :
  (∃ x y: ℝ, (x - 1/2)^2 + y^2 = 1/4) ∧ (∃ t: ℝ, x = -t ∧ y = 2 + t → x + y + 1 = 0) :=
by
  sorry

end shapes_identification_l311_311153


namespace muffins_equation_l311_311896

def remaining_muffins : ℕ := 48
def total_muffins : ℕ := 83
def initially_baked_muffins : ℕ := 35

theorem muffins_equation : initially_baked_muffins + remaining_muffins = total_muffins :=
  by
    -- Skipping the proof here
    sorry

end muffins_equation_l311_311896


namespace range_of_a_l311_311838

-- Define propositions p and q
def p := { x : ℝ | (4 * x - 3) ^ 2 ≤ 1 }
def q (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- Define sets A and B
def A := { x : ℝ | 1 / 2 ≤ x ∧ x ≤ 1 }
def B (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- negation of p (p' is a necessary but not sufficient condition for q')
def p_neg := { x : ℝ | ¬ ((4 * x - 3) ^ 2 ≤ 1) }
def q_neg (a : ℝ) := { x : ℝ | ¬ (a ≤ x ∧ x ≤ a + 1) }

-- range of real number a
theorem range_of_a (a : ℝ) : (A ⊆ B a ∧ A ≠ B a) → 0 ≤ a ∧ a ≤ 1 / 2 := by
  sorry

end range_of_a_l311_311838


namespace fill_pool_time_l311_311410

theorem fill_pool_time (R : ℝ) (T : ℝ) (hSlowerPipe : R = 1 / 9) (hFasterPipe : 1.25 * R = 1.25 / 9)
                     (hCombinedRate : 2.25 * R = 2.25 / 9) : T = 4 := by
  sorry

end fill_pool_time_l311_311410


namespace brad_read_more_books_l311_311285

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end brad_read_more_books_l311_311285


namespace probability_of_rolling_divisor_of_8_l311_311886

open_locale classical

-- Predicate: a number n is a divisor of 8
def is_divisor_of_8 (n : ℕ) : Prop := n ∣ 8

-- The total number of outcomes when rolling an 8-sided die
def total_outcomes : ℕ := 8

-- The probability of rolling a divisor of 8 on a fair 8-sided die
theorem probability_of_rolling_divisor_of_8 (is_fair_die : true) :
  (| {n | is_divisor_of_8 n} ∩ {1, 2, 3, 4, 5, 6, 7, 8} | : ℕ) / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_rolling_divisor_of_8_l311_311886


namespace final_velocity_l311_311770

variable (u a t : ℝ)

-- Defining the conditions
def initial_velocity := u = 0
def acceleration := a = 1.2
def time := t = 15

-- Statement of the theorem
theorem final_velocity : initial_velocity u ∧ acceleration a ∧ time t → (u + a * t = 18) := by
  sorry

end final_velocity_l311_311770


namespace probability_odd_even_draw_correct_l311_311753

noncomputable def probability_odd_even_draw : ℚ := sorry

theorem probability_odd_even_draw_correct :
  probability_odd_even_draw = 17 / 45 := 
sorry

end probability_odd_even_draw_correct_l311_311753


namespace distinct_after_removal_l311_311789

variable (n : ℕ)
variable (subsets : Fin n → Finset (Fin n))

theorem distinct_after_removal :
  ∃ k : Fin n, ∀ i j : Fin n, i ≠ j → (subsets i \ {k}) ≠ (subsets j \ {k}) := by
  sorry

end distinct_after_removal_l311_311789


namespace sum_of_intercepts_l311_311432

theorem sum_of_intercepts (x y : ℝ) (h : x / 3 - y / 4 = 1) : (x / 3 = 1 ∧ y / (-4) = 1) → 3 + (-4) = -1 :=
by
  sorry

end sum_of_intercepts_l311_311432


namespace carrie_profit_l311_311610

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end carrie_profit_l311_311610


namespace price_decrease_necessary_l311_311137

noncomputable def final_price_decrease (P : ℝ) (x : ℝ) : Prop :=
  let increased_price := 1.2 * P
  let final_price := increased_price * (1 - x / 100)
  final_price = 0.88 * P

theorem price_decrease_necessary (x : ℝ) : 
  final_price_decrease 100 x -> x = 26.67 :=
by 
  intros h
  unfold final_price_decrease at h
  sorry

end price_decrease_necessary_l311_311137


namespace monotonically_decreasing_iff_a_lt_1_l311_311175

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - 2 * x

theorem monotonically_decreasing_iff_a_lt_1 {a : ℝ} (h : ∀ x > 0, (deriv (f a) x) < 0) : a < 1 :=
sorry

end monotonically_decreasing_iff_a_lt_1_l311_311175


namespace number_of_odd_factors_of_360_l311_311663

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l311_311663


namespace handshake_problem_l311_311116

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem handshake_problem : combinations 40 2 = 780 := 
by
  sorry

end handshake_problem_l311_311116


namespace total_bananas_in_collection_l311_311254

theorem total_bananas_in_collection (groups_of_bananas : ℕ) (bananas_per_group : ℕ) 
    (h1 : groups_of_bananas = 7) (h2 : bananas_per_group = 29) :
    groups_of_bananas * bananas_per_group = 203 := by
  sorry

end total_bananas_in_collection_l311_311254


namespace inverse_matrix_eigenvalues_l311_311803

theorem inverse_matrix_eigenvalues 
  (c d : ℝ) 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (eigenvalue1 eigenvalue2 : ℝ) 
  (eigenvector1 eigenvector2 : Fin 2 → ℝ) :
  A = ![![1, 2], ![c, d]] →
  eigenvalue1 = 2 →
  eigenvalue2 = 3 →
  eigenvector1 = ![2, 1] →
  eigenvector2 = ![1, 1] →
  (A.vecMul eigenvector1 = (eigenvalue1 • eigenvector1)) →
  (A.vecMul eigenvector2 = (eigenvalue2 • eigenvector2)) →
  A⁻¹ = ![![2 / 3, -1 / 3], ![1 / 6, 1 / 6]] :=
sorry

end inverse_matrix_eigenvalues_l311_311803


namespace rooms_already_painted_l311_311893

-- Define the conditions as variables and hypotheses
variables (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
variables (h1 : total_rooms = 10)
variables (h2 : hours_per_room = 8)
variables (h3 : remaining_hours = 16)

-- Define the theorem stating the number of rooms already painted
theorem rooms_already_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 10) (h2 : hours_per_room = 8) (h3 : remaining_hours = 16) :
  (total_rooms - (remaining_hours / hours_per_room) = 8) :=
sorry

end rooms_already_painted_l311_311893


namespace cost_of_one_dozen_pens_l311_311555

theorem cost_of_one_dozen_pens (pen pencil : ℝ) (h_ratios : pen = 5 * pencil) (h_total : 3 * pen + 5 * pencil = 240) :
  12 * pen = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l311_311555


namespace simplify_expression_l311_311241

-- We need to prove that the simplified expression is equal to the expected form
theorem simplify_expression (y : ℝ) : (3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2)) = (0 * y^2 + 0 * y - 1) :=
by
  -- The detailed proof steps will go here
  sorry

end simplify_expression_l311_311241


namespace find_fg3_l311_311041

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := (x + 2)^2 - 4 * x

theorem find_fg3 : f (g 3) = 49 :=
by
  sorry

end find_fg3_l311_311041


namespace math_problem_l311_311820

theorem math_problem (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1)^2 = 0) : (x + 2 * y)^3 = 125 / 8 := 
sorry

end math_problem_l311_311820


namespace total_crayons_l311_311154

theorem total_crayons (crayons_per_child : ℕ) (number_of_children : ℕ) (h1 : crayons_per_child = 3) (h2 : number_of_children = 6) : 
  crayons_per_child * number_of_children = 18 := by
  sorry

end total_crayons_l311_311154


namespace regular_polygon_sides_l311_311970

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l311_311970


namespace complement_of_A_is_correct_l311_311646

open Set

variable (U : Set ℝ) (A : Set ℝ)

def complement_of_A (U : Set ℝ) (A : Set ℝ) :=
  {x : ℝ | x ∉ A}

theorem complement_of_A_is_correct :
  (U = univ) →
  (A = {x : ℝ | x^2 - 2 * x > 0}) →
  (complement_of_A U A = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) :=
by
  intros hU hA
  simp [hU, hA, complement_of_A]
  sorry

end complement_of_A_is_correct_l311_311646


namespace num_valid_arrangements_without_A_at_start_and_B_at_end_l311_311253

-- Define a predicate for person A being at the beginning
def A_at_beginning (arrangement : List ℕ) : Prop :=
  arrangement.head! = 1

-- Define a predicate for person B being at the end
def B_at_end (arrangement : List ℕ) : Prop :=
  arrangement.getLast! = 2

-- Main theorem stating the number of valid arrangements
theorem num_valid_arrangements_without_A_at_start_and_B_at_end : ∃ (count : ℕ), count = 78 :=
by
  have total_arrangements := Nat.factorial 5
  have A_at_start_arrangements := Nat.factorial 4
  have B_at_end_arrangements := Nat.factorial 4
  have both_A_and_B_arrangements := Nat.factorial 3
  let valid_arrangements := total_arrangements - 2 * A_at_start_arrangements + both_A_and_B_arrangements
  use valid_arrangements
  sorry

end num_valid_arrangements_without_A_at_start_and_B_at_end_l311_311253


namespace parabola_axis_symmetry_value_p_l311_311645

theorem parabola_axis_symmetry_value_p (p : ℝ) (h_parabola : ∀ y x, y^2 = 2 * p * x) (h_axis_symmetry : ∀ (a: ℝ), a = -1 → a = -p / 2) : p = 2 :=
by 
  sorry

end parabola_axis_symmetry_value_p_l311_311645


namespace a4_value_l311_311794

-- Definitions and helper theorems can go here
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- These are our conditions
axiom h1 : S 2 = a 1 + a 2
axiom h2 : a 2 = 3
axiom h3 : ∀ n, S (n + 1) = 2 * S n + 1

theorem a4_value : a 4 = 12 :=
sorry  -- proof to be filled in later

end a4_value_l311_311794
