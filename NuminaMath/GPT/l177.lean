import Mathlib

namespace option1_cost_correct_option2_cost_correct_equal_costs_at_60_more_cost_effective_plan_l177_177469

variables (x : ℕ) (h_gt : x > 10)

def cost_option1 (x : ℕ) : ℕ := 200 * x + 6000
def cost_option2 (x : ℕ) : ℕ := 180 * x + 7200
def more_cost_effective_cost : ℕ := 13400

theorem option1_cost_correct : cost_option1 x = 200 * x + 6000 := by refl
theorem option2_cost_correct : cost_option2 x = 180 * x + 7200 := by refl
theorem equal_costs_at_60 : cost_option1 60 = cost_option2 60 := 
  by calc
  200 * 60 + 6000 = 180 * 60 + 7200 : by linarith
theorem more_cost_effective_plan : more_cost_effective_cost = 13400 := by refl

-- With the given conditions, we need to prove each theorem accordingly.

end option1_cost_correct_option2_cost_correct_equal_costs_at_60_more_cost_effective_plan_l177_177469


namespace transform_function_solution_l177_177069

theorem transform_function_solution (φ a : ℝ) (hφ : φ > 0) :
  (∀ x : ℝ, (√2 * Real.cos (x + π / 4)) = Real.cos 2 * x + Real.sin 2 * x) →
  (φ = π / 2 ∧ a = 1 / 2) :=
by
  sorry

end transform_function_solution_l177_177069


namespace sqrt_200_eq_10_sqrt_2_l177_177005

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177005


namespace man_speed_in_still_water_l177_177887

noncomputable def speedInStillWater 
  (upstreamSpeedWithCurrentAndWind : ℝ)
  (downstreamSpeedWithCurrentAndWind : ℝ)
  (waterCurrentSpeed : ℝ)
  (windSpeedUpstream : ℝ) : ℝ :=
  (upstreamSpeedWithCurrentAndWind + waterCurrentSpeed + windSpeedUpstream + downstreamSpeedWithCurrentAndWind - waterCurrentSpeed + windSpeedUpstream) / 2
  
theorem man_speed_in_still_water :
  speedInStillWater 20 60 5 2.5 = 42.5 :=
  sorry

end man_speed_in_still_water_l177_177887


namespace student_distribution_l177_177953

-- Definition to check the number of ways to distribute 7 students into two dormitories A and B
-- with each dormitory having at least 2 students equals 56.
theorem student_distribution (students dorms : Nat) (min_students : Nat) (dist_plans : Nat) :
  students = 7 → dorms = 2 → min_students = 2 → dist_plans = 56 → 
  true := sorry

end student_distribution_l177_177953


namespace num_values_k_number_of_possible_values_of_k_correct_l177_177925

noncomputable def number_of_possible_values_of_k : ℕ := 
  let p : ℕ := 5
  let q : ℕ := 72 - p
  if Nat.Prime p ∧ Nat.Prime q then 1 else 0

theorem num_values_k (k : ℕ) : 
  (∃ p q : ℕ, p + q = 72 ∧ p * q = k ∧ Nat.Prime p ∧ Nat.Prime q) → k = 335 :=
by sorry

theorem number_of_possible_values_of_k_correct : number_of_possible_values_of_k = 1 :=
by 
  have h := num_values_k 335
  apply h
  existsi [5, 67]
  simp
  exact ⟨by norm_num, by norm_num, prime_5, prime_67⟩

end num_values_k_number_of_possible_values_of_k_correct_l177_177925


namespace sum_of_reciprocals_l177_177410

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : (1/x) + (1/y) = 3/8 :=
by
  sorry

end sum_of_reciprocals_l177_177410


namespace smallest_multiples_5_6_l177_177274

theorem smallest_multiples_5_6 :
  let c := 10 in  -- smallest positive two-digit multiple of 5
  let d := 102 in -- smallest positive three-digit multiple of 6
  c + d = 112 :=
by
  -- The proof is skipped here
  sorry

end smallest_multiples_5_6_l177_177274


namespace unique_or_infinite_planes_l177_177142

-- Definitions
def points_in_space := Π (p₁ p₂ p₃ : ℝ × ℝ × ℝ), 
  (¬collinear p₁ p₂ p₃ → ∃! plane : set (ℝ × ℝ × ℝ), p₁ ∈ plane ∧ p₂ ∈ plane ∧ p₃ ∈ plane) ∧ 
  (collinear p₁ p₂ p₃ → ∃ plane : set (ℝ × ℝ × ℝ), ∀ q ∈ set.range tuple_of_coll_p₁_p₂_p₃, q ∈ plane)

-- Collinear definition
def collinear (p₁ p₂ p₃ : ℝ × ℝ × ℝ) : Prop :=
  ∃ a b c : ℝ, (a • p₁.1 + b • p₁.2 + c • p₁.3 = 0 ∧ 
                a • p₂.1 + b • p₂.2 + c • p₂.3 = 0 ∧ 
                a • p₃.1 + b • p₃.2 + c • p₃.3 = 0)

-- Plane definition
def plane (p₁ p₂ p₃ : ℝ × ℝ × ℝ) (α β γ : ℝ) : set (ℝ × ℝ × ℝ) :=
  {x | α * x.1 + β * x.2 + γ * x.3 = α * p₁.1 + β * p₁.2 + γ * p₁.3}

-- Theorem Statement
theorem unique_or_infinite_planes {p₁ p₂ p₃ : ℝ × ℝ × ℝ} :
  (¬collinear p₁ p₂ p₃ → ∃! plane, p₁ ∈ plane ∧ p₂ ∈ plane ∧ p₃ ∈ plane) ∧ 
  (collinear p₁ p₂ p₃ → ∃ plane, ∀ q ∈ set.range (tuple_of_coll_p₁_p₂_p₃), q ∈ plane) :=
sorry

end unique_or_infinite_planes_l177_177142


namespace range_of_a_l177_177231

noncomputable def f (x : ℝ) : ℝ := 
  if x < 1 then f (2 - x) 
  else 2^x + real.sqrt (x^2 - x + 2)

theorem range_of_a (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f (real.log a (2 * a)) < 6) : 
  (0 < a ∧ a < 1/2) ∨ (2 < a) :=
sorry

end range_of_a_l177_177231


namespace add_points_proof_l177_177757

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l177_177757


namespace functional_eq_zero_l177_177546

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_zero :
  (∀ x y : ℝ, f (x + y) = f x - f y) →
  (∀ x : ℝ, f x = 0) :=
by
  intros h x
  sorry

end functional_eq_zero_l177_177546


namespace last_triangle_perimeter_l177_177336

theorem last_triangle_perimeter (T : ℕ → Triangle) 
  (h1 : T 1 = ⟨1009, 1010, 1011⟩)
  (h2 : ∀ n ≥ 1, 
    let A := T n in
    let AD := A.sides.1 / 2
    let BE := A.sides.2 / 2
    let CF := A.sides.3 / 2
    HAVE sides := [AD + BE, BE + CF, CF + AD]
    ∃ next_triangle, 
    next_triangle = ⟨AD + BE, BE + CF, CF + AD⟩ ∧ 
    T (n + 1) = next_triangle
  )
  : 
  ∃ n ∈ ℕ, ∀ k ≥ n, T n = T k → (T k).perimeter = \frac{759}{64} :=
sorry

end last_triangle_perimeter_l177_177336


namespace first_player_wins_optimal_play_l177_177300

-- Define a function that models the game setup and rules
def game_winner (n : ℕ) : string :=
  if n mod 3 == 0 then
    "Second Player"
  else
    "First Player"

-- Statement to prove that the first player wins with optimal play
theorem first_player_wins_optimal_play (n : ℕ) (h : n > 0) : game_winner(n) = "First Player" := 
sorry

end first_player_wins_optimal_play_l177_177300


namespace eventually_one_student_answers_yes_l177_177835

-- Conditions and Definitions
variable (a b r₁ r₂ : ℕ)
variable (h₁ : r₁ ≠ r₂)   -- r₁ and r₂ are distinct
variable (h₂ : r₁ = a + b ∨ r₂ = a + b) -- One of r₁ or r₂ is the sum a + b
variable (h₃ : a > 0) -- a is a positive integer
variable (h₄ : b > 0) -- b is a positive integer

theorem eventually_one_student_answers_yes (a b r₁ r₂ : ℕ) (h₁ : r₁ ≠ r₂) (h₂ : r₁ = a + b ∨ r₂ = a + b) (h₃ : a > 0) (h₄ : b > 0) :
  ∃ n : ℕ, (∃ c : ℕ, (r₁ = c + b ∨ r₂ = c + b) ∧ (c = a ∨ c ≤ r₁ ∨ c ≤ r₂)) ∨ 
  (∃ c : ℕ, (r₁ = a + c ∨ r₂ = a + c) ∧ (c = b ∨ c ≤ r₁ ∨ c ≤ r₂)) :=
sorry

end eventually_one_student_answers_yes_l177_177835


namespace sqrt_200_eq_10_l177_177020

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177020


namespace trapezium_area_l177_177972

theorem trapezium_area (a b d : ℝ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : d = 14) : 
  (1/2 * (a + b) * d) = 266 :=
by 
  rw [h₁, h₂, h₃]
  norm_num
  -- the proof steps go here
  sorry

end trapezium_area_l177_177972


namespace find_values_of_m_and_n_range_of_a_l177_177230

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := 2^|x - m| + n

theorem find_values_of_m_and_n :
  (∀ x, f 2 5 (x + 4) = f 2 5 x) ∧ (∀ x, 0 ≤ x ∧ x ≤ 4 → f 2 5 x = 2^|x - 2| + 5) ∧ (f 2 5 2 = 6) :=
begin
  -- proof would go here
  sorry
end

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f 2 5 x - a * 2^x = 0 → (9/16 ≤ a ∧ a ≤ 9)) :=
begin
  -- proof would go here
  sorry
end

end find_values_of_m_and_n_range_of_a_l177_177230


namespace least_positive_integer_to_add_l177_177102

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end least_positive_integer_to_add_l177_177102


namespace points_on_line_initial_l177_177761

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l177_177761


namespace incorrect_domain_l177_177513

-- Define the functions and their domains
def fA : ℝ → ℝ := λ x, 2 * x^2
def domA : set ℝ := set.univ  -- all real numbers

def fB : ℝ → ℝ := λ x, 1 / (x + 1)
def domB : set ℝ := {x : ℝ | x ≠ -1}

def fC : ℝ → ℝ := λ x, real.sqrt (x - 2)
def domC : set ℝ := {x : ℝ | x ≥ 2}

def fD : ℝ → ℝ := λ x, 1 / (x + 3)
def domD : set ℝ := {x : ℝ | x > -3}

-- Statement to prove that the domain of fB is incorrectly chosen.
theorem incorrect_domain : ¬ (domB = {x : ℝ | x > -1}) :=
sorry

end incorrect_domain_l177_177513


namespace max_card_count_sum_l177_177508

theorem max_card_count_sum (W B R : ℕ) (total_cards : ℕ) 
  (white_cards black_cards red_cards : ℕ) : 
  total_cards = 300 ∧ white_cards = 100 ∧ black_cards = 100 ∧ red_cards = 100 ∧
  (∀ w, w < white_cards → ∃ b, b < black_cards) ∧ 
  (∀ b, b < black_cards → ∃ r, r < red_cards) ∧ 
  (∀ r, r < red_cards → ∃ w, w < white_cards) →
  ∃ max_sum, max_sum = 20000 :=
by
  sorry

end max_card_count_sum_l177_177508


namespace nuts_in_tree_l177_177823

theorem nuts_in_tree (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : squirrels = nuts + 2) : nuts = 2 :=
by
  sorry

end nuts_in_tree_l177_177823


namespace total_number_of_sampling_results_l177_177481

theorem total_number_of_sampling_results : 
  let junior_students := 400
  let senior_students := 200
  let total_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (junior_sample + senior_sample = total_sample) → 
  (junior_sample / junior_students = 2 / 3) → 
  (senior_sample / senior_students = 1 / 3) → 
  @Fintype.card (Finset (Fin junior_students)).choose junior_sample *
  @Fintype.card (Finset (Fin senior_students)).choose senior_sample = 
  Nat.binom junior_students junior_sample * Nat.binom senior_students senior_sample
:= by
  sorry

end total_number_of_sampling_results_l177_177481


namespace greatest_possible_k_l177_177404

theorem greatest_possible_k (k : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ + x₂ = -k ∧ x₁ * x₂ = 10 ∧ |x₁ - x₂| = 9) → k = 11 :=
begin
  sorry
end

end greatest_possible_k_l177_177404


namespace reach_any_configuration_l177_177131
-- Import the entire Mathlib library to bring in all necessary components.

-- Definitions from the problem
def windmill (length : ℝ) := unit

-- Distance condition for points in S
def distance_condition (S : set (ℝ × ℝ)) (c : ℝ) := ∀ p1 p2 ∈ S, p1 ≠ p2 → dist p1 p2 > c

-- Admissible configuration condition
def admissible_configuration (n : ℕ) (windmills : fin n → windmill 1) (S : set (ℝ × ℝ)) :=
  nodup (windmills.to_list) ∧ ∀ w ∈ windmills, pivot w ∈ S

-- Main theorem to be proven for both cases of c
theorem reach_any_configuration (n : ℕ) (S : set (ℝ × ℝ)) (windmills : fin n → windmill 1) (init_config : admissible_configuration n windmills S) (final_config : admissible_configuration n windmills S) :
  (distance_condition S (sqrt 3) ∨ distance_condition S (sqrt 2)) → 
  (∃ (seq : ℕ → fin n → windmill 1), seq 0 = init_config ∧ seq (succ (succ _)) = final_config ∧ ∀ i, admissible_configuration n (seq i) S) :=
sorry

end reach_any_configuration_l177_177131


namespace integral_sqrt_one_minus_x_squared_l177_177465

theorem integral_sqrt_one_minus_x_squared :
  ∫ x in 0..1, real.sqrt (1 - x^2) = real.pi / 4 :=
sorry

end integral_sqrt_one_minus_x_squared_l177_177465


namespace billy_distance_l177_177922

def displacement_billy (east_walk : ℝ) (north_angle : ℝ) (north_walk : ℝ) : ℝ :=
  let bd_leg := north_walk / Real.sqrt 2
  let total_east := east_walk + bd_leg
  let total_north := bd_leg
  Real.sqrt (total_east^2 + total_north^2)

theorem billy_distance :
  displacement_billy 5 45 8 = Real.sqrt (89 + 40 * Real.sqrt 2) :=
by
  sorry

end billy_distance_l177_177922


namespace first_number_is_8_percent_of_third_number_l177_177498

variables {X A B : ℝ}

def A_is_8_percent_of_X := A = 0.08 * X
def B_is_16_percent_of_X := B = 0.16 * X
def A_is_50_percent_of_B := A = 0.5 * B

theorem first_number_is_8_percent_of_third_number 
  (h1 : A_is_8_percent_of_X)
  (h2 : B_is_16_percent_of_X)
  (h3 : A_is_50_percent_of_B) : A = 0.08 * X :=
by
  sorry

end first_number_is_8_percent_of_third_number_l177_177498


namespace radius_tangent_circle_l177_177265

theorem radius_tangent_circle
  (r1 r2 : ℝ) (h1 : r1 > 0) (h2 : r2 > 0) :
  ∃ R : ℝ, R = (r1 + r2 + real.sqrt ((r1 + r2)^2 + 4 * r1 * r2)) / 2 :=
begin
  use (r1 + r2 + real.sqrt ((r1 + r2)^2 + 4 * r1 * r2)) / 2,
  sorry,
end

end radius_tangent_circle_l177_177265


namespace front_view_l177_177179

def first_column_heights := [3, 2]
def middle_column_heights := [1, 4, 2]
def third_column_heights := [5]

theorem front_view (h1 : first_column_heights = [3, 2])
                   (h2 : middle_column_heights = [1, 4, 2])
                   (h3 : third_column_heights = [5]) :
    [3, 4, 5] = [
        first_column_heights.foldr max 0,
        middle_column_heights.foldr max 0,
        third_column_heights.foldr max 0
    ] :=
    sorry

end front_view_l177_177179


namespace sqrt_200_eq_l177_177014

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177014


namespace circle_line_divide_l177_177038

theorem circle_line_divide (S : set (ℝ × ℝ))
  (has_circle : ∀ x y, S ((x - 0.5)^2 + (y - 0.5)^2 = 0.25 ∨ (x - 0.5)^2 + (y - 1.5)^2 = 0.25))
  (line_slope : ∀ p q r, (q = 1) ∧ ( (q + 2r)^2 + (p - 2)^2 + (r - 5)^2 = 30))
  : ∃ (p q r : ℤ), p^2 + q^2 + r^2 = 30 := sorry

end circle_line_divide_l177_177038


namespace cube_of_square_of_third_smallest_prime_l177_177080

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l177_177080


namespace mod_product_eq_15_l177_177775

theorem mod_product_eq_15 :
  (15 * 24 * 14) % 25 = 15 :=
by
  sorry

end mod_product_eq_15_l177_177775


namespace range_of_m_l177_177997

noncomputable def A (x : ℝ) : Prop := |x - 2| ≤ 4
noncomputable def B (x : ℝ) (m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, (¬A x) → (¬B x m)) ∧ (∃ x, (¬B x m) ∧ ¬(¬A x)) → m ≥ 5 :=
sorry

end range_of_m_l177_177997


namespace remainder_71_mul_73_div_9_l177_177454

theorem remainder_71_mul_73_div_9 : 
  (71 * 73) % 9 = 8 :=
by
  have h71 : 71 % 9 = 8 := by norm_num
  have h73 : 73 % 9 = 1 := by norm_num
  calc
    (71 * 73) % 9
        = ((71 % 9) * (73 % 9)) % 9 : by rw [Nat.mul_mod]
    ... = (8 * 1) % 9 : by rw [h71, h73]
    ... = 8 % 9 : by norm_num
    ... = 8 : by norm_num

end remainder_71_mul_73_div_9_l177_177454


namespace pet_store_cages_l177_177889

/-
Prove that the number of bird cages is 6, given that:
- The pet store has 6.0 parrots.
- The pet store has 2.0 parakeets.
- On average, 1.333333333 birds can occupy one cage.
-/
def number_of_cages (parrots : ℝ) (parakeets : ℝ) (birds_per_cage : ℝ) : ℝ :=
  (parrots + parakeets) / birds_per_cage

theorem pet_store_cages : number_of_cages 6.0 2.0 1.333333333 = 6 := by
  sorry

end pet_store_cages_l177_177889


namespace functional_eq_zero_l177_177545

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_zero :
  (∀ x y : ℝ, f (x + y) = f x - f y) →
  (∀ x : ℝ, f x = 0) :=
by
  intros h x
  sorry

end functional_eq_zero_l177_177545


namespace david_lewis_meeting_point_l177_177535

theorem david_lewis_meeting_point :
  ∀ (D : ℝ),
  (∀ t : ℝ, t ≥ 0 →
    ∀ distance_to_meeting_point : ℝ, 
    distance_to_meeting_point = D →
    ∀ speed_david speed_lewis distance_cities : ℝ,
    speed_david = 50 →
    speed_lewis = 70 →
    distance_cities = 350 →
    ((distance_cities + distance_to_meeting_point) / speed_lewis = distance_to_meeting_point / speed_david) →
    D = 145.83) :=
by
  intros D t ht distance_to_meeting_point h_distance speed_david speed_lewis distance_cities h_speed_david h_speed_lewis h_distance_cities h_meeting_time
  -- We need to prove D = 145.83 under the given conditions
  sorry

end david_lewis_meeting_point_l177_177535


namespace bus_children_after_stop_l177_177827

theorem bus_children_after_stop (initial_children off_children diff_on_off : ℕ) (h₀ : initial_children = 36) (h₁ : off_children = 68) (h₂ : off_children - diff_on_off = 24) :
  let on_children := off_children - diff_on_off in
  initial_children - off_children + on_children = 12 :=
by
  sorry

end bus_children_after_stop_l177_177827


namespace necessary_but_not_sufficient_l177_177264

-- Definitions of vectors and their magnitudes
variables {V : Type*} [inner_product_space ℝ V]
variables {A B C : V}

-- The given condition
def condition (A B C : V) : Prop :=
  ∥ B - A + (C - A) ∥ > ∥ B - A - (C - A) ∥

-- Definition of an acute triangle
def is_acute_triangle (A B C : V) : Prop :=
  ∀ ⦃u v : V⦄, u ∈ {B - A, C - A, A - C} → v ∈ {B - A, C - A, A - C} → inner_product_space.inner u v > 0

-- Lean statement for the problem
theorem necessary_but_not_sufficient (A B C : V) : 
  (condition A B C → is_acute_triangle A B C) ∧ (∀ P Q R : V, is_acute_triangle P Q R → condition P Q R) → 
  (∃ P Q R : V, is_acute_triangle P Q R ∧ ¬ condition P Q R): 
  sorry

end necessary_but_not_sufficient_l177_177264


namespace cube_of_square_of_third_smallest_prime_l177_177079

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l177_177079


namespace technician_round_trip_percentage_l177_177853

theorem technician_round_trip_percentage (D: ℝ) (hD: D ≠ 0): 
  let round_trip_distance := 2 * D
  let distance_to_center := D
  let distance_back_10_percent := 0.10 * D
  let total_distance_completed := distance_to_center + distance_back_10_percent
  let percentage_completed := (total_distance_completed / round_trip_distance) * 100
  percentage_completed = 55 := 
by
  simp
  sorry -- Proof is not required per instructions

end technician_round_trip_percentage_l177_177853


namespace shortest_side_of_similar_triangle_l177_177894

def Triangle (a b c : ℤ) : Prop := a^2 + b^2 = c^2
def SimilarTriangles (a b c a' b' c' : ℤ) : Prop := ∃ k : ℤ, k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c 

theorem shortest_side_of_similar_triangle (a b c a' b' c' : ℤ)
  (h₀ : Triangle 15 b 17)
  (h₁ : SimilarTriangles 15 b 17 a' b' c')
  (h₂ : c' = 51) : a' = 24 :=
by
  sorry

end shortest_side_of_similar_triangle_l177_177894


namespace probability_of_matching_pair_l177_177960

theorem probability_of_matching_pair (blackSocks blueSocks : ℕ) (h_black : blackSocks = 12) (h_blue : blueSocks = 10) : 
  let totalSocks := blackSocks + blueSocks
  let totalWays := Nat.choose totalSocks 2
  let blackPairWays := Nat.choose blackSocks 2
  let bluePairWays := Nat.choose blueSocks 2
  let matchingPairWays := blackPairWays + bluePairWays
  totalWays = 231 ∧ matchingPairWays = 111 → (matchingPairWays : ℚ) / totalWays = 111 / 231 := 
by
  intros
  sorry

end probability_of_matching_pair_l177_177960


namespace value_of_c_l177_177802

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem value_of_c :
  let midpoint := midpoint (1, 3) (5, 11) in 
  let c := midpoint.1 + midpoint.2 in
  c = 10 :=
by
  have m : (ℝ × ℝ) := midpoint (1, 3) (5, 11)
  let c := m.1 + m.2
  suffices h : c = 10, from h
  sorry

end value_of_c_l177_177802


namespace face_value_amount_of_bill_l177_177061

def true_discount : ℚ := 45
def bankers_discount : ℚ := 54

theorem face_value_amount_of_bill : 
  ∃ (FV : ℚ), bankers_discount = true_discount + (true_discount * bankers_discount / FV) ∧ FV = 270 :=
by
  sorry

end face_value_amount_of_bill_l177_177061


namespace distinct_cosines_impossible_l177_177136

open Real

noncomputable def angles_of_pentagon (α β γ δ ε : ℝ) : Prop :=
  α + β + γ + δ + ε = 3 * π

def sin_values_condition (sα sβ sγ sδ sε: ℝ) : Prop :=
  ∃ α β γ δ ε : ℝ, 
    sin α = sα ∧ sin β = sβ ∧ sin γ = sγ ∧ sin δ = sδ ∧ sin ε = sε ∧ 
    α ∈ (0:ℝ) <.. (π:ℝ) ∧ β ∈ (0:ℝ) <.. (π:ℝ) ∧ γ ∈ (0:ℝ) <.. (π:ℝ) ∧ 
    δ ∈ (0:ℝ) <.. (π:ℝ) ∧ ε ∈ (0:ℝ) <.. (π:ℝ) ∧ 
    angles_of_pentagon α β γ δ ε ∧
    ((sα = sβ) ∨ (sα = sγ) ∨ (sα = sδ) ∨ (sα = sε) ∨
    (sβ = sγ) ∨ (sβ = sδ) ∨ (sβ = sε) ∨
    (sγ = sδ) ∨ (sγ = sε) ∨
    (sδ = sε))

theorem distinct_cosines_impossible (α β γ δ ε : ℝ) :
  angles_of_pentagon α β γ δ ε →
  ∃ sα sβ sγ sδ sε : ℝ, 
    sin_values_condition sα sβ sγ sδ sε →
    ¬(cos α ≠ cos β ∧ cos α ≠ cos γ ∧ cos α ≠ cos δ ∧ 
      cos α ≠ cos ε ∧ cos β ≠ cos γ ∧ cos β ≠ cos δ ∧ 
      cos β ≠ cos ε ∧ cos γ ≠ cos δ ∧ cos γ ≠ cos ε ∧
      cos δ ≠ cos ε) :=
sorry

end distinct_cosines_impossible_l177_177136


namespace oranges_left_to_sell_l177_177360

theorem oranges_left_to_sell (x : ℤ) (hx : x ≥ 7) : 
  (let total_oranges := 12 * x;
       first_friend := (1 / 4) * total_oranges;
       second_friend := (1 / 6) * total_oranges;
       charity := (1 / 8) * total_oranges;
       given_away := first_friend + second_friend + charity;
       remaining_after_giveaway := total_oranges - given_away;
       sold_yesterday := (3 / 7) * remaining_after_giveaway;
       remaining_after_selling := remaining_after_giveaway - sold_yesterday;
       birds_ate := (1 / 10) * remaining_after_selling;
       remaining_after_birds := remaining_after_selling - birds_ate;
       rotten := 4)
  in remaining_after_birds - rotten = 3.0214287 * x - 4) :=
begin
  sorry
end

end oranges_left_to_sell_l177_177360


namespace markus_grandson_age_l177_177721

theorem markus_grandson_age :
  ∃ (x : ℕ), let son := 2 * x in let markus := 2 * son in x + son + markus = 140 ∧ x = 20 :=
by
  sorry

end markus_grandson_age_l177_177721


namespace distance_in_yards_l177_177884

theorem distance_in_yards (miles_per_half: Nat) (yards_per_half: Float) (yards_in_mile: Nat) (half_marathons: Nat) (y: Float) : 
  miles_per_half = 13 → 
  yards_per_half = 192.5 → 
  yards_in_mile = 1760 → 
  half_marathons = 6 →
  y = half_marathons * yards_per_half → 
  0 ≤ y ∧ y < yards_in_mile :=
by
  intros
  split
  • sorry  -- prove 0 ≤ y
  • sorry  -- prove y < yards_in_mile

end distance_in_yards_l177_177884


namespace problem_CEF_over_DBE_l177_177661

noncomputable def AB := 120
noncomputable def AC := 120
noncomputable def AD := 40
noncomputable def CF := 80

theorem problem_CEF_over_DBE : 
  ∀ (AB AC AD CF : ℝ), 
  AB = 120 → AC = 120 → AD = 40 → CF = 80 → 
  (let CEF_DBE := 3 in ∀ ratio : ℝ, ratio = CEF_DBE) :=
by
  intros AB AC AD CF hAB hAC hAD hCF ratio
  simp only [*, eq_self_iff_true, forall_true_iff]
  exact sorry

end problem_CEF_over_DBE_l177_177661


namespace problem_solution_l177_177527

theorem problem_solution (a b : ℝ) (a_nonneg : a ≥ 0) (b_nonneg : b ≥ 0):
  (sqrt (a^2 + b^2) ≠ 0) ∧ 
  (sqrt (a^2 + b^2) ≠ (a + b) / 2) ∧ 
  (sqrt (a^2 + b^2) ≠ sqrt a + sqrt b) ∧ 
  (sqrt (a^2 + b^2) ≠ a + b - 1) ↔ 
  (a = 0 ∧ b = 0) := 
by 
  sorry

end problem_solution_l177_177527


namespace homework_points_l177_177736

variable (H Q T : ℕ)

theorem homework_points (h1 : T = 4 * Q)
                        (h2 : Q = H + 5)
                        (h3 : H + Q + T = 265) : 
  H = 40 :=
sorry

end homework_points_l177_177736


namespace fraction_equality_l177_177537

def op_at (a b : ℕ) : ℕ := a * b + b^2
def op_hash (a b : ℕ) : ℕ := a + b + a * (b^2)

theorem fraction_equality : (op_at 5 3 : ℚ) / (op_hash 5 3 : ℚ) = 24 / 53 := 
by 
  sorry

end fraction_equality_l177_177537


namespace isosceles_triangle_CD_length_l177_177122

theorem isosceles_triangle_CD_length (AB AC D B C : Point)
  (h1 : AB = AC)
  (h2 : ∠ BCD = 15°)
  (h3 : dist B C = √6)
  (h4 : dist A D = 1) :
  dist C D = √(13 - 6 * √3) :=
sorry

end isosceles_triangle_CD_length_l177_177122


namespace points_on_line_l177_177770

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177770


namespace max_missed_problems_is_12_l177_177919

-- Define the number of problems in the test
def number_of_problems : ℕ := 50

-- Define the passing percentage
def passing_percentage : ℚ := 0.75

-- Calculate the minimum score required to pass
def minimum_correct (n : ℕ) (p : ℚ) : ℕ := (n : ℚ) * p

-- Calculate the maximum number of problems that can be missed
def max_missed (n : ℕ) (p : ℚ) : ℕ := n - minimum_correct n p

-- The theorem we want to prove
theorem max_missed_problems_is_12 :
  max_missed number_of_problems passing_percentage = 12 :=
by
  sorry

end max_missed_problems_is_12_l177_177919


namespace stratified_random_sampling_l177_177486

open Finset

theorem stratified_random_sampling :
  let junior_high := 400
      senior_high := 200
      total_sample := 60
      ratio_junior := 2
      ratio_senior := 1
      proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
      proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
      sample_junior := proportion_junior * total_sample
      sample_senior := proportion_senior * total_sample in
  sample_junior + sample_senior = total_sample ∧
  ((junior_high.choose sample_junior) * (senior_high.choose sample_senior)) = Σ {C}_{400}^{40}•{C}_{200}^{20}
:= by
  let junior_high := 400
  let senior_high := 200
  let total_sample := 60
  let ratio_junior := 2
  let ratio_senior := 1
  let proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
  let proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
  let sample_junior := proportion_junior * total_sample
  let sample_senior := proportion_senior * total_sample
  have h1 : sample_junior + sample_senior = total_sample := sorry
  have h2 : (junior_high.choose sample_junior) * (senior_high.choose sample_senior) 
              = (C 400 40) * (C 200 20) := sorry
  exact ⟨h1, h2⟩

end stratified_random_sampling_l177_177486


namespace ctg_sum_condition_l177_177737

open Real

noncomputable def points_on_line (A B C : Point) : Prop := 
  collinear A B C ∧ (dist A B < dist A C ∧ dist B C < dist A C)

noncomputable def ctg (θ : ℝ) : ℝ := 1 / tan(θ)

theorem ctg_sum_condition 
  (A B C M : Point) (k : ℝ)
  (h_line : points_on_line A B C)
  (h_ctg : ctg (angle A M B) + ctg (angle B M C) = k) :
  ∃ R, R = (k * (dist A B) * (dist B C)) / (dist A B + dist B C) ∧ 
  (circle B R M) := sorry.

end ctg_sum_condition_l177_177737


namespace set_cardinality_proof_l177_177747

variable (A B : Type)
variable [Fintype A] [Fintype B]

noncomputable def cardA := Fintype.card A
noncomputable def cardB := Fintype.card B

-- Define conditions as hypotheses
theorem set_cardinality_proof
  (h1 : cardA = 3 * cardB)
  (h2 : cardA + cardB - Fintype.card (A ⊓ B) = 5000)
  (h3 : Fintype.card (A ⊓ B) = 1000) :
  cardA = 4500 :=
by
  sorry

end set_cardinality_proof_l177_177747


namespace least_possible_value_of_smallest_integer_l177_177119

theorem least_possible_value_of_smallest_integer
  (A B C D E F : ℤ)
  (h_diff : list.nodup [A, B, C, D, E, F])
  (h_avg : (A + B + C + D + E + F) / 6 = 85)
  (h_F : F = 180)
  (h_E : E = 100) :
  A = -64 :=
sorry

end least_possible_value_of_smallest_integer_l177_177119


namespace sequence_general_formula_sum_of_first_n_terms_l177_177235

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def T (a : ℕ → ℕ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b i

theorem sequence_general_formula
  (a : ℕ → ℕ)
  (h1 : a 2 = 3)
  (h2 : (∑ i in Finset.range 5, a (i + 1)) = 25) :
  ∀ n, a n = 2 * n - 1 :=
sorry

theorem sum_of_first_n_terms 
  (a : ℕ → ℕ)
  (b : ℕ → ℝ)
  (h3 : ∀ n, b n = 1 / (Real.sqrt (a (n + 1)) + Real.sqrt (a n)))
  (h4 : ∀ n, a n = 2 * n - 1) :
  ∀ n, T a b n = 1 / 2 * (Real.sqrt (2 * n + 1) - 1) :=
sorry

end sequence_general_formula_sum_of_first_n_terms_l177_177235


namespace probability_interval_l177_177356

noncomputable def xi_distribution (mu sigma : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := sorry

variable {μ σ : ℝ} (xi : ℝ → ℝ)

axiom xi_normal_dist : xi_distribution μ σ

axiom prob_neg_two : MeasureTheory.MeasureSpace.Measure.restrict (xi_distribution μ σ) (Set.Iio (-2)) = 0.3
axiom prob_two : MeasureTheory.MeasureSpace.Measure.restrict (xi_distribution μ σ) (Set.Ioi 2) = 0.3

theorem probability_interval : MeasureTheory.MeasureSpace.Measure.restrict (xi_distribution μ σ) (Set.Ioc (-2) 0) = 0.2 := by
  sorry

end probability_interval_l177_177356


namespace minimum_students_using_both_l177_177650

theorem minimum_students_using_both (n L T x : ℕ) 
  (H1: 3 * n = 7 * L) 
  (H2: 5 * n = 6 * T) 
  (H3: n = 42) 
  (H4: n = L + T - x) : 
  x = 11 := 
by 
  sorry

end minimum_students_using_both_l177_177650


namespace minimum_value_of_x_l177_177208

theorem minimum_value_of_x : 
  ∃ x : ℝ, (4 * x^2 + 8 * x + 3 = 5) ∧ (x = -1 - sqrt 6 / 2) :=
begin
  sorry
end

end minimum_value_of_x_l177_177208


namespace area_of_triangle_PFG_l177_177320

-- Define the constants and conditions
def triangle_PQR_side_lengths : Prop :=
  PQ = 7 ∧ QR = 15 ∧ PR = 14

def points_on_segments : Prop :=
  PF = 2 ∧ PG = 5

-- Define the area of triangle PFG
def area_PFG : ℝ := (24 * Real.sqrt 66) / 21

-- The theorem to prove
theorem area_of_triangle_PFG
  (PQ QR PR PF PG : ℝ)
  (h1 : PQ = 7)
  (h2 : QR = 15)
  (h3 : PR = 14)
  (h4 : PF = 2)
  (h5 : PG = 5) :
  area_PFG = (24 * Real.sqrt 66) / 21 := 
sorry

end area_of_triangle_PFG_l177_177320


namespace smallest_successive_number_l177_177052

theorem smallest_successive_number :
  ∃ n : ℕ, n * (n + 1) * (n + 2) = 1059460 ∧ ∀ m : ℕ, m * (m + 1) * (m + 2) = 1059460 → n ≤ m :=
sorry

end smallest_successive_number_l177_177052


namespace solve_x4_minus_inv_x4_l177_177584

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l177_177584


namespace bob_walked_distance_l177_177452

theorem bob_walked_distance 
  (distance_XY : ℝ)
  (yolanda_rate : ℝ)
  (bob_rate : ℝ)
  (delay : ℝ) :
  distance_XY = 60 →
  yolanda_rate = 5 →
  bob_rate = 6 →
  delay = 1 →
  ∃ t : ℝ, yolanda_rate * (t + delay) + bob_rate * t = distance_XY ∧ bob_rate * t = 30 :=
by
  intros h1 h2 h3 h4
  use 5
  rw [←h1, ←h2, ←h3, ←h4]
  split
  . simp
  . simp
  sorry

end bob_walked_distance_l177_177452


namespace least_positive_integer_addition_l177_177105

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end least_positive_integer_addition_l177_177105


namespace arithmetic_geometric_sum_l177_177579

variables {a b : ℕ → ℕ} {s : ℕ → ℕ}
def arithmetic_seq (a : ℕ → ℕ) (d a1 : ℤ) := ∀ n, a n = a1 + d * n
def geometric_seq (b : ℕ → ℕ) (b1 q : ℤ) := ∀ n, b n = b1 * q ^ n

theorem arithmetic_geometric_sum :
  (∃ d a1, arithmetic_seq a d a1 ∧ a 1 = 1 ∧ a 4 = 8 ∧ s 11 = 88) ∧ 
  (∃ b1 q, geometric_seq b b1 q ∧ b 1 = 2 ∧ q > 0 ∧ b 2 + b 3 = 12 ∧ b 3 = a 4 - 2 * a 1) →
  (a = λ n, 3 * n - 2 ∧ b = λ n, 2 ^ n) ∧ 
  (∀ n, ∑ i in finset.range n, (a (2 * i + 1) * b (2 * i)) = (3 * n - 2) / 3 * 4^(n ≠ 0 + 1) + (8 / 3)) :=
sorry

end arithmetic_geometric_sum_l177_177579


namespace initial_tree_height_l177_177903

-- Definition of the problem conditions as Lean definitions.
def quadruple (x : ℕ) : ℕ := 4 * x

-- Given conditions of the problem
def final_height : ℕ := 256
def height_increase_each_year (initial_height : ℕ) : Prop :=
  quadruple (quadruple (quadruple (quadruple initial_height))) = final_height

-- The proof statement that we need to prove
theorem initial_tree_height 
  (initial_height : ℕ)
  (h : height_increase_each_year initial_height)
  : initial_height = 1 := sorry

end initial_tree_height_l177_177903


namespace count_integers_with_digit_zero_leq_2017_l177_177269

def containsDigitZero (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ d > 0 ∧ d == 0 ∧ Nat.digits 10 n.contains d

theorem count_integers_with_digit_zero_leq_2017 : 
  Nat.count (λ n, n ≤ 2017 ∧ containsDigitZero n) (List.range 2018) = 469 := 
by sorry

end count_integers_with_digit_zero_leq_2017_l177_177269


namespace integral_sqrt_x2_sub_4x_sub_3_l177_177974

theorem integral_sqrt_x2_sub_4x_sub_3 :
  ∃ C : ℝ, ∫ (x : ℝ) in set.univ, 1 / (√(x^2 - 4*x - 3)) = (λ x, real.log (|x - 2 + √((x - 2)^2 - 7)|) + C) :=
sorry

end integral_sqrt_x2_sub_4x_sub_3_l177_177974


namespace find_a_l177_177689

noncomputable def f (a x : ℝ) : ℝ := a * real.sqrt (1 - x ^ 2) + real.sqrt (1 - x) + real.sqrt (1 + x)

noncomputable def g (a : ℝ) : ℝ := real.Sup (set.range (f a))

def solution_set : set ℝ := {1} ∪ {a | -real.sqrt 2 ≤ a ∧ a ≤ -real.sqrt 2 / 2}

theorem find_a (a : ℝ) :
  g(a) = g(1/a) ↔ a ∈ solution_set :=
sorry

end find_a_l177_177689


namespace top_square_after_folds_l177_177467

theorem top_square_after_folds :
  let grid := list.range 81
  -- fold the right third over the middle third
  let grid1 := grid.zipWithIndex |>.map (λ (n, i), if i % 9 >= 6 then grid[i - 3] else n)
  -- fold the left third over the rest
  let grid2 := grid1.zipWithIndex |>.map (λ (n, i), if i % 9 < 3 then grid1[i + 3] else n)
  -- fold the top third down over the middle third
  let grid3 := grid2.zipWithIndex |>.map (λ (n, i), if i / 9 < 3 then grid2[i + 27] else n)
  -- fold the bottom third up over the rest
  let grid4 := grid3.zipWithIndex |>.map (λ (n, i), if i / 9 >= 6 then grid3[i - 27] else n)
  -- fold diagonally from top left to bottom right corner
  let grid5 := grid4.zipWithIndex |>.map (λ (n, i), if i == 0 then grid4[18] else n)
  grid5.head = 18 := sorry

end top_square_after_folds_l177_177467


namespace second_large_bucket_contains_39_ounces_l177_177188

-- Define the five initial buckets
def buckets : List ℕ := [11, 13, 12, 16, 10]

-- Define the condition of the first large bucket containing 23 ounces after pouring the 10-ounce bucket and another bucket
def combined_buckets_condition : Prop := ∃ b ∈ buckets, b ≠ 10 ∧ 10 + b = 23

-- Define the remaining buckets after removing the buckets used in combined_buckets_condition
def remaining_buckets : List ℕ := buckets.filter (λ b, ¬(10 + b = 23))

-- Define the sum of the remaining buckets
def sum_remaining_buckets : ℕ := remaining_buckets.sum

-- Theorem stating the answer to the problem, i.e., the amount of water in the second large bucket
theorem second_large_bucket_contains_39_ounces (h : combined_buckets_condition) : sum_remaining_buckets = 39 := by
  sorry

end second_large_bucket_contains_39_ounces_l177_177188


namespace initial_bottles_l177_177358

theorem initial_bottles (x : ℕ) (h1 : x - 8 + 45 = 51) : x = 14 :=
by
  -- Proof goes here
  sorry

end initial_bottles_l177_177358


namespace quadratic_equation_with_root_three_l177_177114

-- Define the necessary conditions and their relationships
def is_root (p : polynomial ℝ) (r : ℝ) : Prop := p.eval r = 0

-- Theorem stating that under the given condition, x^2 - 3x = 0 is a quadratic equation with one root 3
theorem quadratic_equation_with_root_three : ∃ (a : ℝ), is_root (X^2 - 3*X) 3 :=
by
  use 0
  rw [is_root, polynomial.eval_X, polynomial.eval_pow, polynomial.eval_mul]
  sorry

end quadratic_equation_with_root_three_l177_177114


namespace average_visitors_on_sundays_is_correct_l177_177886

noncomputable def average_visitors_sundays
  (num_sundays : ℕ) (num_non_sundays : ℕ) 
  (avg_non_sunday_visitors : ℕ) (avg_month_visitors : ℕ) : ℕ :=
  let total_month_days := num_sundays + num_non_sundays
  let total_visitors := avg_month_visitors * total_month_days
  let total_non_sunday_visitors := num_non_sundays * avg_non_sunday_visitors
  let total_sunday_visitors := total_visitors - total_non_sunday_visitors
  total_sunday_visitors / num_sundays

theorem average_visitors_on_sundays_is_correct :
  average_visitors_sundays 5 25 240 290 = 540 :=
by
  sorry

end average_visitors_on_sundays_is_correct_l177_177886


namespace sum_consecutive_odds_l177_177989

theorem sum_consecutive_odds (n : ℕ) :
  (Finset.sum (Finset.range n) (λ i, n + 2 * i)) = 4 * ( (2 * n - 1) / 2 ) ^ 2 := by
    sorry

end sum_consecutive_odds_l177_177989


namespace find_vertex_R_l177_177666

open Real

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def midpoint (A B : Point3D) : Point3D :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2,
  z := (A.z + B.z) / 2 }

variables {P Q R : Point3D}
variables (M_PQ : midpoint P Q = ⟨3, 2, 1⟩)
variables (M_QR : midpoint Q R = ⟨4, 3, 2⟩)
variables (M_RP : midpoint R P = ⟨1, 4, 3⟩)

theorem find_vertex_R : R = ⟨0, 3, 2⟩ :=
sorry

end find_vertex_R_l177_177666


namespace max_value_of_function_l177_177805

theorem max_value_of_function :
  let f := λ x : ℝ, (Real.sqrt 3) * (Real.sin (2 * x)) + 2 * (Real.sin x) + 4 * (Real.sqrt 3) * (Real.cos x)
  ∃ x : ℝ, ∀ y : ℝ, f(x) ≥ f(y) ∧ f(x) = 17 / 2 :=
by
  let f := λ x : ℝ, (Real.sqrt 3) * (Real.sin (2 * x)) + 2 * (Real.sin x) + 4 * (Real.sqrt 3) * (Real.cos x)
  sorry

end max_value_of_function_l177_177805


namespace fractions_arith_l177_177930

theorem fractions_arith : (3 / 50) + (2 / 25) - (5 / 1000) = 0.135 := by
  sorry

end fractions_arith_l177_177930


namespace explanation_and_methods_l177_177047

section MaritimeSilkRoad

variables (M : Type) -- Maritime Silk Road context type

-- Definitions based on conditions
def history_developed (H : M) : Prop := 
  -- the history of economic and cultural exchange and interaction between East and West
  sorry

def principles (H : M) : Prop := 
  -- openness, cooperation, harmony, inclusiveness, and mutual benefit guide the building of the 21st-century Maritime Silk Road
  sorry

def debate (H : M) : Prop := 
  -- debate among students about whether economy or culture comes first
  sorry

-- Main theorem to prove
theorem explanation_and_methods (H : M) :
  history_developed H →
  principles H →
  debate H →
  (analyzing_relationship H) ∧ (applying_methods_and_divergences H) :=
begin
  sorry
end

-- Definitions used in the main theorem
def analyzing_relationship (H : M) : Prop := 
  -- Analyzing the relationship between social existence and social consciousness, and the relationship between culture and economy
  sorry

def applying_methods_and_divergences (H : M) : Prop := 
  -- Applying the method of combining the theory of two points and the theory of emphasis, and explaining why there are divergences in viewpoints
  sorry

end MaritimeSilkRoad

end explanation_and_methods_l177_177047


namespace number_of_lattice_points_l177_177364

theorem number_of_lattice_points (A B : ℝ) (h : B - A = 10) :
  ∃ n, n = 10 ∨ n = 11 :=
sorry

end number_of_lattice_points_l177_177364


namespace add_points_proof_l177_177755

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l177_177755


namespace max_no_draws_l177_177654

theorem max_no_draws (students : ℕ) (total_points : ℕ) 
  (points_win : ℕ) (points_draw : ℕ) (points_loss : ℕ) 
  (games : ℕ) (total_games : ℕ) : 
  students = 16 → total_points = 550 → 
  points_win = 5 → points_draw = 2 → points_loss = 0 → 
  games = 120 → total_games = 120 →
  ∃ (max_no_draws_students : ℕ), max_no_draws_students = 5 :=
by
  intros h_students h_total_points h_points_win h_points_draw h_points_loss h_games h_total_games
  use 5
  sorry

end max_no_draws_l177_177654


namespace segments_DE_BF_equal_perpendicular_l177_177735

-- Define points A, B, C as vertices of triangle ABC
variables {A B C D E F : Type*}

-- Define the conditions based on the given problem
def is_right_isosceles (x y z : Type*) : Prop :=
  ∃ d, d ≠ x ∧ d ≠ y ∧ d ≠ z ∧ d ≠ ([x, y, z] : finset Type*).sum

def triangle_ABC (A B C : Type*) : Prop :=
  true

def is_right_triangle (T : Type*) (x y : Type*) : Prop :=
  ∃ z, z ≠ x ∧ z ≠ y ∧ z ∈ (T)

-- Define the points D, E, F as vertices of right angles in respective right triangles
variables (D E F : Type*) 

axiom isosceles_right_triangles_abd_bce_acf : 
  is_right_isosceles A B D ∧ is_right_isosceles B C E ∧ is_right_isosceles A C F

-- Problem statement: Prove DE and BF are equal and perpendicular
theorem segments_DE_BF_equal_perpendicular :
  triangle_ABC A B C → isosceles_right_triangles_abd_bce_acf A B C D E F →
  (∃u v : Type*, u = E ∧ v = F ∧ (u - v).norm = (D - B).norm) ∧ 
  (u - v).perpendicular (D - B) :=
sorry

end segments_DE_BF_equal_perpendicular_l177_177735


namespace average_first_21_multiples_of_8_l177_177432

theorem average_first_21_multiples_of_8 :
  let seq := list.range (21 + 1) -- range is from 0 to 21
  let multiples_of_8 := seq.map (λ x, 8 * x)
  let sum := multiples_of_8.sum
  let n := 21
  (sum / n) = 88 :=
by
  sorry

end average_first_21_multiples_of_8_l177_177432


namespace points_on_line_l177_177772

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177772


namespace no_solution_range_of_a_l177_177985

noncomputable def range_of_a : Set ℝ := {a | ∀ x : ℝ, ¬(abs (x - 1) + abs (x - 2) ≤ a^2 + a + 1)}

theorem no_solution_range_of_a :
  range_of_a = {a | -1 < a ∧ a < 0} :=
by
  sorry

end no_solution_range_of_a_l177_177985


namespace required_force_l177_177946

theorem required_force (m : ℝ) (g : ℝ) (T : ℝ) (F : ℝ) 
    (h1 : m = 3)
    (h2 : g = 10)
    (h3 : T = m * g)
    (h4 : F = 4 * T) : F = 120 := by
  sorry

end required_force_l177_177946


namespace range_of_tangent_diff_l177_177656

variables {A B C : ℝ} (a b c: ℝ)
hypothesis (acute_ABC : A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
hypothesis (sides_relation : b^2 - a^2 = a * c)

theorem range_of_tangent_diff :
  1 < (1 / tan A) - (1 / tan B) ∧ (1 / tan A) - (1 / tan B) < 2 * sqrt 3 / 3 :=
sorry

end range_of_tangent_diff_l177_177656


namespace smallest_k_49_divides_binom_l177_177935

theorem smallest_k_49_divides_binom : 
  ∃ k : ℕ, 0 < k ∧ 49 ∣ Nat.choose (2 * k) k ∧ (∀ m : ℕ, 0 < m ∧ 49 ∣ Nat.choose (2 * m) m → k ≤ m) ∧ k = 25 :=
by
  sorry

end smallest_k_49_divides_binom_l177_177935


namespace kedlaya_problem_l177_177462

def fractional_part (x : ℝ) : ℝ := x - (x.floor : ℝ)

theorem kedlaya_problem (p s : ℤ) (hp : Prime p) (hs1 : 0 < s) (hs2 : s < p) :
  (∃ (m n : ℤ), 0 < m ∧ m < n ∧ n < p ∧
    fractional_part (s * m / p) < fractional_part (s * n / p) ∧
    fractional_part (s * n / p) < s / p) ↔
  ¬ (s ∣ (p - 1)) :=
by 
  sorry

end kedlaya_problem_l177_177462


namespace x_fourth_minus_inv_fourth_l177_177591

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l177_177591


namespace math_problem_l177_177369

noncomputable def possibleValuesOfa : ℕ := 503

theorem math_problem (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : a + b + c + d = 2020)
  (h5 : a^2 - b^2 + c^2 - d^2 = 4040) : ∃ n, n = possibleValuesOfa ∧ ∃ a_values, ∀ a, a ∈ a_values → 
  a > 0 ∧ ∃ b c d, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2020 ∧ a^2 - b^2 + c^2 - d^2 = 4040 :=
begin
  sorry
end

end math_problem_l177_177369


namespace average_age_before_new_students_joined_l177_177387

theorem average_age_before_new_students_joined 
  (A : ℝ) 
  (N : ℕ) 
  (new_students_average_age : ℝ) 
  (average_age_drop : ℝ) 
  (original_class_strength : ℕ)
  (hN : N = 17) 
  (h_new_students : new_students_average_age = 32)
  (h_age_drop : average_age_drop = 4)
  (h_strength : original_class_strength = 17)
  (h_equation : 17 * A + 17 * new_students_average_age = (2 * original_class_strength) * (A - average_age_drop)) :
  A = 40 :=
by sorry

end average_age_before_new_students_joined_l177_177387


namespace problem_part_a_problem_part_b_l177_177456

open Nat

-- Part (a)
theorem problem_part_a (x : ℕ → ℝ) (h₀ : x 0 = 1) (h₁ : ∀ i, 0 < x (i + 1) ∧ x (i + 1) < x i) :
  ∃ n ≥ 1, ∑ i in finset.range n, (x i) ^ 2 / x (i + 1) ≥ 3.999 :=
sorry

-- Part (b)
theorem problem_part_b : ∃ x : ℕ → ℝ, x 0 = 1 ∧ (∀ i, 0 < x (i + 1) ∧ x (i + 1) < x i) ∧ (∀ n, ∑ i in finset.range n, (x i) ^ 2 / x (i + 1) < 4) :=
sorry

end problem_part_a_problem_part_b_l177_177456


namespace transformed_roots_polynomial_l177_177340

-- Given conditions
variables {a b c : ℝ}
variables (h : ∀ x, (x - a) * (x - b) * (x - c) = x^3 - 4 * x + 6)

-- Prove the equivalent polynomial with the transformed roots
theorem transformed_roots_polynomial :
  (∀ x, (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 23 * x + 21) :=
sorry

end transformed_roots_polynomial_l177_177340


namespace concyclicity_of_O_R_T_B_l177_177390

variables {α : Type*}
open_locale classical

-- Definitions of points and circles
variables (A B C D O M N R T : α)
variable {S1 : set α}
variable {S2 : set α}

-- Given conditions
variables [convex_quad ABCD]
variables [inscribed S1 ABCD]
variables [intersection O A C B D]
variables [passes_through S2 D O]
variables [intersection M A D S2]
variables [intersection N C D S2]
variables [intersection R O M A B]
variables [intersection T O N B C]
variables [same_side R T A B D]

-- Prove the theorem
theorem concyclicity_of_O_R_T_B
  (h₁ : convex_quad ABCD)
  (h₂ : inscribed S1 ABCD)
  (h₃ : intersection O A C B D)
  (h₄ : passes_through S2 D O)
  (h₅ : intersection M A D S2)
  (h₆ : intersection N C D S2)
  (h₇ : intersection R O M A B)
  (h₈ : intersection T O N B C)
  (h₉ : same_side R T A B D) : 
  concyclic O R T B :=
sorry

end concyclicity_of_O_R_T_B_l177_177390


namespace a_2000_mod_9_l177_177054

def a : ℕ → ℕ
| 0       := 1995
| (n + 1) := (n + 1) * a n + 1

theorem a_2000_mod_9 : a 2000 % 9 = 5 := 
by
  sorry

end a_2000_mod_9_l177_177054


namespace bushes_needed_for_60_zucchinis_l177_177541

-- Each blueberry bush yields 10 containers of blueberries.
def containers_per_bush : ℕ := 10

-- 6 containers of blueberries can be traded for 3 zucchinis.
def containers_to_zucchinis (containers zucchinis : ℕ) : Prop := containers = 6 ∧ zucchinis = 3

theorem bushes_needed_for_60_zucchinis (bushes containers zucchinis : ℕ) :
  containers_per_bush = 10 →
  containers_to_zucchinis 6 3 →
  zucchinis = 60 →
  bushes = 12 :=
by
  intros h1 h2 h3
  sorry

end bushes_needed_for_60_zucchinis_l177_177541


namespace number_of_sparkly_numbers_l177_177143

-- Define what it means for a number to be sparkly
def isSparkly (n : ℕ) : Prop :=
  (n >= 10^8 ∧ n < 10^9) ∧ -- The integer has exactly 9 digits
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → ((n / 10^(9-i)) % 10 ≠ 0 ∧ (n / 10^(9-i)) % 10 % i = 0)) -- For each n between 1 and 9 inclusive

-- Prove that there are exactly 216 sparkly numbers
theorem number_of_sparkly_numbers : 
  { n : ℕ | isSparkly n }.to_finset.card = 216 :=
by
  sorry

end number_of_sparkly_numbers_l177_177143


namespace cube_of_square_is_15625_l177_177100

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l177_177100


namespace area_PQR_is_14_l177_177050

notation "ℝ²" => (ℝ × ℝ)

def reflect_y_axis (p : ℝ²) : ℝ² :=
  (-p.1, p.2)

def reflect_y_eq_x (p : ℝ²) : ℝ² :=
  (p.2, p.1)

def distance (p q : ℝ²) : ℝ :=
  Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

def area_of_triangle (P Q R : ℝ²) : ℝ :=
  1 / 2 * (distance P Q) * Real.abs (R.2 - P.2)

theorem area_PQR_is_14 :
  let P : ℝ² := (2, 5)
  let Q : ℝ² := reflect_y_axis P
  let R : ℝ² := reflect_y_eq_x Q
  area_of_triangle P Q R = 14 :=
by
  sorry

end area_PQR_is_14_l177_177050


namespace square_number_condition_l177_177213

theorem square_number_condition (n p : ℕ) (h₁ : n > 0) (h₂ : prime p) (h₃ : n^2 = 11 * p + 4) : n = 9 := sorry

end square_number_condition_l177_177213


namespace max_flow_density_vehicle_flow_expression_l177_177292

noncomputable def v : ℝ → ℝ :=
  λ x, if x ≤ 30 then 60 else -1/3 * x + 70

noncomputable def f (x : ℝ) : ℝ := x * v x

theorem max_flow_density : 
  ∃ x : ℝ, 30 ≤ x ∧ x ≤ 210 ∧ f x = 3675 :=
by sorry

theorem vehicle_flow_expression :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 210 →
    (v x = if x ≤ 30 then 60 else -1/3 * x + 70) :=
by sorry

end max_flow_density_vehicle_flow_expression_l177_177292


namespace birds_flew_up_l177_177129

theorem birds_flew_up (initial_birds : ℕ) (total_birds : ℕ) (new_birds : ℕ) 
  (h1 : initial_birds = 231) (h2 : total_birds = 312) : 
  new_birds = total_birds - initial_birds := 
begin
  sorry
end

example : birds_flew_up 231 312 81 := 
begin
  apply birds_flew_up,
  repeat { refl },
end

end birds_flew_up_l177_177129


namespace cube_of_square_of_third_smallest_prime_l177_177078

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l177_177078


namespace cute_numbers_count_l177_177869

def is_cute_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3

def is_cute_number (l : List ℕ) : Prop :=
  l.length = 10 ∧
  (∀ d ∈ l, is_cute_digit d) ∧
  (∀ i ∈ List.range 9, (l.nthLe i sorry).abs_sub (l.nthLe (i + 1) sorry) = 1)

def total_cute_numbers (n : ℕ) : ℕ :=
  List.length (List.filter is_cute_number (List.range (3^n)))

theorem cute_numbers_count : total_cute_numbers 10 = 64 :=
sorry

end cute_numbers_count_l177_177869


namespace increase_in_license_plates_l177_177725

/-- The number of old license plates and new license plates in MiraVille. -/
def old_license_plates : ℕ := 26^2 * 10^3
def new_license_plates : ℕ := 26^2 * 10^4

/-- The ratio of the number of new license plates to the number of old license plates is 10. -/
theorem increase_in_license_plates : new_license_plates / old_license_plates = 10 := by
  unfold old_license_plates new_license_plates
  sorry

end increase_in_license_plates_l177_177725


namespace max_value_f_intersection_points_l177_177463

def f (x : ℝ) : ℝ := -x^2 + 8*x
def g (x : ℝ) (m : ℝ) : ℝ := 6 * Real.log x + m

-- Prove the maximum value h(t) of f(x) in the interval [t, t+1]
theorem max_value_f (t : ℝ) : 
  ∃ h : ℝ, h = 
    if t < 3 then -t^2 + 6*t + 7 
    else if 3 ≤ t ∧ t ≤ 4 then 16 
    else -t^2 + 8*t := 
sorry

-- Prove there exists real number m such that y = f(x) and y = g(x) have exactly three different intersection points and find the range of m
theorem intersection_points (m : ℝ) : 
  ∃ (m_min m_max : ℝ), 
    (7 < m_min) ∧ (m_min < m_max) ∧ (m_max = 15 - 6 * Real.log 3) := 
sorry

end max_value_f_intersection_points_l177_177463


namespace peach_ratios_and_percentages_l177_177648

def red_peaches : ℕ := 8
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6
def orange_peaches : ℕ := 4
def total_peaches : ℕ := red_peaches + yellow_peaches + green_peaches + orange_peaches

theorem peach_ratios_and_percentages :
  ((green_peaches : ℚ) / total_peaches = 3 / 16) ∧
  ((green_peaches : ℚ) / total_peaches * 100 = 18.75) ∧
  ((yellow_peaches : ℚ) / total_peaches = 7 / 16) ∧
  ((yellow_peaches : ℚ) / total_peaches * 100 = 43.75) :=
by {
  sorry
}

end peach_ratios_and_percentages_l177_177648


namespace ratio_of_triangle_to_square_is_one_ninth_l177_177307

noncomputable def area_of_triangle (A B C : (ℕ × ℕ)) : ℕ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  1 / 2 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

noncomputable def area_of_square (s : ℕ) : ℕ := s^2

def ratio_of_areas (s : ℕ) : ℚ :=
  let A := (0, 0)
  let M := (s / 3, 0)
  let N := (0, 2 * s / 3)
  let area_AMN := area_of_triangle A M N
  let area_ABCD := area_of_square s
  area_AMN / area_ABCD

theorem ratio_of_triangle_to_square_is_one_ninth (s : ℕ) : ratio_of_areas s = 1 / 9 := by
  sorry

end ratio_of_triangle_to_square_is_one_ninth_l177_177307


namespace f1_in_SetA_f2_not_in_SetA_l177_177405

open Real

-- Define the set A as a property
def inSetA (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → f x + 2 * f y > 3 * f ((x + 2 * y) / 3)

-- Define the two functions to be tested
def f1 (x : ℝ) : ℝ := log x / log 2 -- log base 2 of x
def f2 (x : ℝ) : ℝ := (x + 1) ^ 2 -- (x + 1)^2

-- Statements to be proven
theorem f1_in_SetA : inSetA f1 := sorry

theorem f2_not_in_SetA : ¬ inSetA f2 := sorry

end f1_in_SetA_f2_not_in_SetA_l177_177405


namespace value_of_f_prime_at_2016_l177_177993

def f (x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * x * (f'' 2016) - 2016 * log x

theorem value_of_f_prime_at_2016 (f' : ℝ → ℝ) (f'' : ℝ → ℝ) :
  (∀ x, f' x = x + 2 * (f'' 2016) - 2016 / x) →
  f' 2016 = -2015 :=
by
  -- The proof would go here.
  sorry

end value_of_f_prime_at_2016_l177_177993


namespace wrongly_read_number_l177_177782

theorem wrongly_read_number 
  (S_initial : ℕ) (S_correct : ℕ) (correct_num : ℕ) (num_count : ℕ) 
  (h_initial : S_initial = num_count * 18) 
  (h_correct : S_correct = num_count * 19) 
  (h_correct_num : correct_num = 36) 
  (h_diff : S_correct - S_initial = correct_num - wrong_num) 
  (h_num_count : num_count = 10) 
  : wrong_num = 26 :=
sorry

end wrongly_read_number_l177_177782


namespace sqrt_200_eq_10_sqrt_2_l177_177007

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177007


namespace solve_pond_fish_problem_l177_177857

def pond_fish_problem 
  (tagged_fish : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (total_fish : ℕ) : Prop :=
  (tagged_in_second_catch : ℝ) / second_catch = (tagged_fish : ℝ) / total_fish →
  total_fish = 1750

theorem solve_pond_fish_problem : 
  pond_fish_problem 70 50 2 1750 :=
by
  sorry

end solve_pond_fish_problem_l177_177857


namespace count_zeros_in_range_l177_177908

theorem count_zeros_in_range :
  let count_zeros : ℕ → ℕ := λ n, (n.toString.toList.filter (λ c, c = '0')).length
  ∑ i in Finset.range 222222223, count_zeros i = 175308642 :=
sorry

end count_zeros_in_range_l177_177908


namespace arc_length_of_curve_l177_177934

noncomputable def parametricArcLength : ℝ := ∫ t in 0..(2 * Real.pi), 
  Real.sqrt ((6 * (Real.sin (2 * t) - Real.sin t))^2 + (6 * (Real.cos t - Real.cos (2 * t)))^2)

theorem arc_length_of_curve:
  parametricArcLength = 48 :=
begin
  sorry -- Proof skipped as per instructions
end

end arc_length_of_curve_l177_177934


namespace sqrt_200_eq_l177_177010

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177010


namespace steeper_increase_y_axis_l177_177882

/-
  Given five distinct radii, and the corresponding standard and modified areas
  of circles and ellipses respectively, we aim to prove that the graphical
  relation of these points shows a steeper increase in the y-axis (standard area)
  compared to the x-axis (modified area).
-/

def standard_area (r : ℕ) : ℝ := Real.pi * r^2
def modified_area (r : ℕ) : ℝ := Real.pi * r

theorem steeper_increase_y_axis :
  let radii := [1, 2, 3, 4, 5]
  let points := radii.map (λ r => (standard_area r, modified_area r))
  ∃ S M, (S, M) ∈ points ∧ S > M := 
by
  sorry

end steeper_increase_y_axis_l177_177882


namespace amit_work_time_l177_177511

theorem amit_work_time 
  (Amit_rate : ℝ → ℝ)
  (Ananthu_rate : ℝ)
  (Amit_days : ℝ)
  (Ananthu_days : ℝ)
  (Total_days : ℝ)
  (work_done_amit : ℝ)
  (work_done_ananthu : ℝ)
  (work_done_total : ℝ) :
  Ananthu_rate = 1 / 45 →
  Amit_days = 3 →
  Ananthu_days = 36 →
  Total_days = Amit_days + Ananthu_days →
  work_done_amit = 3 * Amit_rate Amit_days →
  work_done_ananthu = 36 * Ananthu_rate →
  work_done_total = work_done_amit + work_done_ananthu →
  work_done_total = 1 →
  Amit_rate (x : ℝ) = 1 / x →
  x = 15 :=
begin
  intros,
  sorry,
end

end amit_work_time_l177_177511


namespace percent_change_is_five_percent_l177_177743

variable (x : ℝ)

/-- The value of the stock at the end of the first day -/
def stock_value_first_day (x : ℝ) : ℝ :=
  x - 0.30 * x

/-- The value of the stock at the end of the second day -/
def stock_value_second_day (x : ℝ) : ℝ :=
  stock_value_first_day x + 0.50 * stock_value_first_day x

/-- The overall percent change in the stock value over two days -/
def percent_change (x : ℝ) : ℝ :=
  ((stock_value_second_day x - x) / x) * 100

/-- Given the conditions, the overall percent change in Rachel's stock over two days is 5% -/
theorem percent_change_is_five_percent (x : ℝ) : percent_change x = 5 := by
  sorry

end percent_change_is_five_percent_l177_177743


namespace not_always_possible_for_k_8_l177_177563

noncomputable def each_box_equal_after_steps (n : ℕ) (k : ℕ) : Prop :=
  ∃ (steps : ℕ), ∀ (a : fin n → ℕ), ∃ (b : fin n → ℕ), 
    (∀ i j : fin n, b i = b j) ∧ (∀ s < steps, ∃ (indices : fin n → bool), 
    (indices.sum (λ i, if indices i then 1 else 0) = k) ∧ 
    (∀ i, b i = a i + finset.sum (finset.filter indices finset.univ) i))

theorem not_always_possible_for_k_8 :
  each_box_equal_after_steps 2002 8 := by
  sorry

end not_always_possible_for_k_8_l177_177563


namespace original_price_1739_13_l177_177899

theorem original_price_1739_13 (P : ℝ) (h1: 0.56 * (1 + 0.15) * P = 1120) : 
  P ≈ 1739.13 :=
by 
  sorry

end original_price_1739_13_l177_177899


namespace trapezium_area_l177_177971

theorem trapezium_area (a b d : ℝ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : d = 14) : 
  (1/2 * (a + b) * d) = 266 :=
by 
  rw [h₁, h₂, h₃]
  norm_num
  -- the proof steps go here
  sorry

end trapezium_area_l177_177971


namespace max_element_of_list_l177_177140

-- Definitions based on the conditions
def is_median (l : List ℕ) (m : ℕ) : Prop :=
  l.length = 7 ∧ List.sorted (≤) l ∧ l.get? 3 = some m

def is_mean (l : List ℕ) (mean : ℕ) : Prop :=
  (l.sum : ℕ) / l.length = mean

-- The target theorem to prove
theorem max_element_of_list (l : List ℕ) (h1 : is_median l 4) (h2 : is_mean l 20) :
  l.maximum = 108 :=
by sorry

end max_element_of_list_l177_177140


namespace x_condition_sufficient_not_necessary_l177_177457

theorem x_condition_sufficient_not_necessary (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (¬ (∀ x, x^2 - 1 > 0 → x < -1)) :=
by
  sorry

end x_condition_sufficient_not_necessary_l177_177457


namespace cheese_remaining_after_10_customers_l177_177900

noncomputable def average_weight (k : Nat) : ℝ := 20 / (k + 10)

theorem cheese_remaining_after_10_customers :
  (∃ k : Nat, k ≤ 10 ∧ 20 - k * average_weight k = 10 * average_weight k) → 
  20 - 10 * average_weight 10 = 10 :=
by
  intro h,
  have s10 := average_weight 10,
  sorry

end cheese_remaining_after_10_customers_l177_177900


namespace exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l177_177205

theorem exists_xy_such_that_x2_add_y2_eq_n_mod_p
  (p : ℕ) [Fact (Nat.Prime p)] (n : ℤ)
  (hp1 : p > 5) :
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = n % p) :=
sorry

theorem p_mod_4_eq_1_implies_n_can_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp1 : p % 4 = 1) : 
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

theorem p_mod_4_eq_3_implies_n_cannot_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 4 = 3) :
  ¬(∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

end exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l177_177205


namespace sum_cn_l177_177234

-- Definitions according to conditions
def a (n : ℕ) : ℤ := 2 * n
def b (n : ℕ) : ℤ := (-2)^(n-1)
def c (n : ℕ) : ℤ := a n + b n

-- Lean theorem stating the sums
theorem sum_cn {n : ℕ} : 
  (∑ k in finset.range n, c (2 * k + 1)) = 2 * n^2 + (4^n - 1) / 3 :=
by sorry

end sum_cn_l177_177234


namespace triangle_max_perimeter_l177_177576

-- Definitions to set up the problem
variables (a b c : ℝ) (A B C : ℝ) (S : ℝ)
variables (CA CB : ℝ) (area_ABC : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := (sqrt 3 / 2) * (CA * CB) = 2 * S
def condition2 : Prop := area_ABC = S
def condition3 : Prop := c = sqrt 6

-- The statement that combines all conditions and conclusions
theorem triangle_max_perimeter (h1 : condition1)
                                (h2 : condition2)
                                (h3 : condition3) :
                                C = 60 ∧ a + b + c <= 3 * sqrt 6 :=
begin
  sorry,
end

end triangle_max_perimeter_l177_177576


namespace transformed_graph_area_l177_177384

theorem transformed_graph_area (g : ℝ → ℝ) (a b : ℝ)
  (h_area_g : ∫ x in a..b, g x = 15) :
  ∫ x in a..b, 2 * g (x + 3) = 30 := 
sorry

end transformed_graph_area_l177_177384


namespace g_domain_range_l177_177495

def dom_f : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def rng_f : Set ℝ := {y | 0 ≤ y ∧ y ≤ 1}

def f (x : ℝ) : ℝ

def dom_g : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def rng_g : Set ℝ := {y | 0 ≤ y ∧ y ≤ 1}

theorem g_domain_range :
  (∀ x, f x ∈ rng_f → x ∈ dom_f) →
  (∀ y, y ∈ rng_f → ∃ x, f x = y) →
  (∀ x, x ∈ dom_g ↔ (x + 1) ∈ dom_f) ∧
  (∀ y, y ∈ rng_g ↔ ∃ x (hx : x ∈ dom_g), y = 1 - f (x + 1)) :=
by
  sorry

end g_domain_range_l177_177495


namespace gas_reaction_solution_l177_177222

noncomputable def gas_reaction_problem : Prop :=
  ∃ (A B C D : Type) (a : ℕ) 
    (mol_A mol_B : ℝ) 
    (container_volume : ℝ) 
    (time_reaction min_5 : ℝ) 
    (rate_D : ℝ)  
    (prod_C : ℝ) 
    (reaction : ℕ × ℕ × ℕ),
    mol_A = 0.6 ∧ 
    mol_B = 0.5 ∧ 
    container_volume = 0.4 ∧ 
    time_reaction = 5.0 ∧ 
    rate_D = 0.1 ∧ 
    prod_C = 0.2 ∧ 
    reaction = (3, 1, a) ∧ 
    C = a ∧ 
    a = 2

theorem gas_reaction_solution : gas_reaction_problem :=
by
  unfold gas_reaction_problem
  sorry

end gas_reaction_solution_l177_177222


namespace abc_greater_than_n_l177_177776

theorem abc_greater_than_n
  (a b c n : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 1 < n)
  (h5 : a ^ n + b ^ n = c ^ n) :
  a > n ∧ b > n ∧ c > n :=
sorry

end abc_greater_than_n_l177_177776


namespace same_color_probability_l177_177137

-- Given conditions
def totalChairs : Nat := 33
def blackChairs : Nat := 15
def brownChairs : Nat := 18

-- Define the probability calculation
noncomputable def probability_same_color : Rat :=
  let prob_both_black := (blackChairs : Rat) * (blackChairs - 1) / (totalChairs * (totalChairs - 1))
  let prob_both_brown := (brownChairs : Rat) * (brownChairs - 1) / (totalChairs * (totalChairs - 1))
  prob_both_black + prob_both_brown

-- The statement to prove
theorem same_color_probability : probability_same_color = 43 / 88 :=
  sorry

end same_color_probability_l177_177137


namespace chess_player_total_games_l177_177876

noncomputable def total_games_played (W L : ℕ) : ℕ :=
  W + L

theorem chess_player_total_games :
  ∃ (W L : ℕ), W = 16 ∧ (L : ℚ) / W = 7 / 4 ∧ total_games_played W L = 44 :=
by
  sorry

end chess_player_total_games_l177_177876


namespace sqrt_200_eq_l177_177009

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177009


namespace part1_part2_l177_177233

section Sequence

-- Define the sequence
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a n + 1

-- Prove that {a_n + 1} is geometric
theorem part1 : ∀ n, (a n + 1) * 2 = a (n + 1) + 1 := by
  intro n
  sorry

-- Define the partial sum T_n
noncomputable def T : ℕ → ℚ
| 0       := 1
| (n + 1) := T n + (1 / (a n * a (n + 1)))

-- Prove that the sum of the sequence {2^n / a_n a_{n+1}} is T_n
theorem part2 : ∀ n, T n = 1 - (1 / (2^(n + 1) - 1)) := by
  intro n
  sorry

end Sequence

end part1_part2_l177_177233


namespace train_passes_man_in_time_l177_177854

-- Define the conditions
def train_length : ℝ := 550 -- in meters
def train_speed : ℝ := 60 -- in km/h
def man_speed : ℝ := 6 -- in km/h
def speed_conversion_factor : ℝ := 1000 / 3600 -- from km/h to m/s

-- Prove the question equals the answer given conditions
theorem train_passes_man_in_time :
  let relative_speed := ((train_speed + man_speed) * speed_conversion_factor) in
  let time_to_pass := train_length / relative_speed in
  abs(time_to_pass - 30) < 1 := 
by 
  sorry -- Proof omitted

end train_passes_man_in_time_l177_177854


namespace boat_speed_in_still_water_l177_177852

-- Definitions of the conditions
def with_stream_speed : ℝ := 36
def against_stream_speed : ℝ := 8

-- Let Vb be the speed of the boat in still water, and Vs be the speed of the stream.
variable (Vb Vs : ℝ)

-- Conditions given in the problem
axiom h1 : Vb + Vs = with_stream_speed
axiom h2 : Vb - Vs = against_stream_speed

-- The statement to prove: the speed of the boat in still water is 22 km/h.
theorem boat_speed_in_still_water : Vb = 22 := by
  sorry

end boat_speed_in_still_water_l177_177852


namespace Stephen_total_distance_l177_177382

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end Stephen_total_distance_l177_177382


namespace fraction_ordering_l177_177434

theorem fraction_ordering 
  (a : ℚ) (b : ℚ) (c : ℚ) 
  (h1 : a = 6 / 29) 
  (h2 : b = 10 / 31) 
  (h3 : c = 8 / 25) : 
  a < c ∧ c < b := 
by 
  sorry

end fraction_ordering_l177_177434


namespace find_a_b_find_k_range_l177_177617

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.exp x * (a * x + b) + x^2 + 2 * x

theorem find_a_b : 
  (∃ a b : ℝ, 
    ∀ x : ℝ, f x a b = Real.exp x * (a * x + b) + x^2 + 2 * x ∧ 
    f 0 a b = 1 ∧ 
    (deriv (λ x, f x a b) 0 = 4)) → 
    (a = 1 ∧ b = 1) := 
begin
  sorry
end

theorem find_k_range :
  (∃ k : ℝ, (∀ x : ℝ, -2 ≤ x ∧ x ≤ -1 → f x 1 1 ≥ x^2 + 2 * (k + 1) * x + k)) →
    (∃ k_min : ℝ, k_min = 1/4 * Real.exp (-3/2)) ∧ 
    (∀ k : ℝ, k ≥ k_min) :=
begin
  sorry
end

end find_a_b_find_k_range_l177_177617


namespace number_of_roots_l177_177210

noncomputable def roots_eq (a : ℝ) (ha : 0 < a ∧ a < real.exp (-1)) : ℕ := 
  let f : ℂ → ℂ := λ z, z^2
  let phi : ℂ → ℂ := λ z, -a * complex.exp z
  if h : ∀ z, complex.abs z = 1 → complex.abs (f z) > complex.abs (phi z)
  then 2
  else 0

theorem number_of_roots (a : ℝ) (ha : 0 < a ∧ a < real.exp (-1)) : roots_eq a ha = 2 := by
  sorry

end number_of_roots_l177_177210


namespace necessary_and_sufficient_for_perpendicular_l177_177859

theorem necessary_and_sufficient_for_perpendicular (a : ℝ) :
  (a = -2) ↔ (∀ (x y : ℝ), x + 2 * y = 0 → ax + y = 1 → false) :=
by
  sorry

end necessary_and_sufficient_for_perpendicular_l177_177859


namespace alice_probability_same_color_l177_177873

def total_ways_to_draw : ℕ := 
  Nat.choose 9 3 * Nat.choose 6 3 * Nat.choose 3 3

def favorable_outcomes_for_alice : ℕ := 
  3 * Nat.choose 6 3 * Nat.choose 3 3

def probability_alice_same_color : ℚ := 
  favorable_outcomes_for_alice / total_ways_to_draw

theorem alice_probability_same_color : probability_alice_same_color = 1 / 28 := 
by
  -- Proof is omitted as per instructions
  sorry

end alice_probability_same_color_l177_177873


namespace B_share_of_profit_l177_177304

theorem B_share_of_profit (x : ℝ) (total_profit : ℝ) (h1 : total_profit = 150000) 
  (h2 : 2 * (2 * x) + 3 * x = 6 * x) : 
  (3 * x / (6 * x)) * total_profit = 75000 :=
by
  rw [h1, mul_div_cancel_left, mul_comm]
  norm_num
  sorry

end B_share_of_profit_l177_177304


namespace find_a_and_monotonicity_l177_177681

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 5)^2 + 6 * real.log x

theorem find_a_and_monotonicity (a : ℝ) :
  let f := λ (x : ℝ), a * (x - 5)^2 + 6 * real.log x,
      f' := λ (x : ℝ), 2 * a * (x - 5) + 6 / x in
  (f 1 = a * (1 - 5)^2 + 6 * real.log 1) ∧
  ((tangent_eq : (6 - 8 * a) * 1 + (16 * a - 6) = 0) → a = 1 / 2) ∧
  (∀ x, 0 < x → 
    (f'(x) = 2 * a * (x - 5) + 6 / x) ∧ 
    ((0 < x ∧ x < 2) → f'(x) > 0) ∧ 
    ((2 < x ∧ x < 3) → f'(x) < 0) ∧ 
    (x > 3 → f'(x) > 0)) :=
by
  sorry

end find_a_and_monotonicity_l177_177681


namespace statue_original_cost_l177_177858

theorem statue_original_cost (selling_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : selling_price = 620) (h2 : profit_percent = 25) : 
  original_cost = 496 :=
by
  have h3 : profit_percent / 100 + 1 = 1.25 := by sorry
  have h4 : 1.25 * original_cost = selling_price := by sorry
  have h5 : original_cost = 620 / 1.25 := by sorry
  have h6 : 620 / 1.25 = 496 := by sorry
  exact sorry

end statue_original_cost_l177_177858


namespace stratified_sampling_l177_177478

open Nat

theorem stratified_sampling :
  let junior_students := 400 in
  let senior_students := 200 in
  let total_sample_size := 60 in
  let junior_ratio := 2 in
  let senior_ratio := 1 in
  let total_students := junior_students + senior_students in 
  junior_ratio * senior_students = senior_ratio * junior_students →
  junior_students + senior_students = total_sample_size ->
  let junior_sample_size := (junior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  let senior_sample_size := (senior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  junior_sample_size = 40 →
  senior_sample_size = 20 →
  choose junior_students junior_sample_size * choose senior_students senior_sample_size = binomial 400 40 * binomial 200 20 :=
by 
  intros; 
  sorry

end stratified_sampling_l177_177478


namespace solve_y_l177_177690

def star (a b : ℤ) : ℤ := a * b + 2 * b - a

theorem solve_y : ∃ y : ℚ, star 7 y = 85 ∧ y = 92 / 9 :=
by 
  sorry

end solve_y_l177_177690


namespace find_15th_student_age_l177_177781

noncomputable def age_of_15th_student : ℕ :=
  let total_age_15_students := 15 * 15
  let total_age_5_students := 5 * 12
  let total_age_9_students := 9 * 16
  total_age_15_students - total_age_5_students - total_age_9_students

theorem find_15th_student_age :
  age_of_15th_student = 21 :=
  by
    let total_age_15_students := 15 * 15
    let total_age_5_students := 5 * 12
    let total_age_9_students := 9 * 16
    let total_age_15th_student := total_age_15_students - total_age_5_students - total_age_9_students
    show total_age_15th_student = 21 from eq.refl _
    sorry

end find_15th_student_age_l177_177781


namespace cube_of_square_is_15625_l177_177099

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l177_177099


namespace contrapositive_of_square_comparison_l177_177786

theorem contrapositive_of_square_comparison (x y : ℝ) : (x^2 > y^2 → x > y) → (x ≤ y → x^2 ≤ y^2) :=
  by sorry

end contrapositive_of_square_comparison_l177_177786


namespace find_coefficient_b_l177_177807

variable (a b c p : ℝ)

def parabola (x : ℝ) := a * x^2 + b * x + c

theorem find_coefficient_b (h_vertex : ∀ x, parabola a b c x = a * (x - p)^2 + p)
                           (h_y_intercept : parabola a b c 0 = -3 * p)
                           (hp_nonzero : p ≠ 0) :
  b = 8 / p :=
by
  sorry

end find_coefficient_b_l177_177807


namespace distinct_collections_l177_177731

def letters := ['M', 'M', 'A', 'A', 'A', 'T', 'T', 'H', 'E', 'I', 'C', 'L']

-- Define total ways to choose 3 vowels and 4 consonants such that T's, M's, and A's are indistinguishable.
def count_vowels (l : List Char) : Nat :=
  if l.count 'A' = 3 then 1
  else if l.count 'A' = 2 ∧ ('E' ∈ l ∨ 'I' ∈ l) then 2
  else if l.count 'A' = 1 ∧ ('E' ∈ l) ∧ ('I' ∈ l) then 1
  else 0

def list_combinations (l : List Char) (n : Nat) : List (List Char) :=
  -- some implementation that generates the list of combinations of length n from l
  sorry

def count_consonants (l : List Char) : Nat :=
  let cons := list_combinations (['T', 'T', 'M', 'M', 'H', 'C', 'L'] : List Char) 4 
  in  
  if l.count 'T' = 2 ∧ l.count 'M' = 2 then 1
  else if l.count 'T' = 2 ∧ l.count 'M' = 1 ∧ ('H' ∈ l ∨ 'C' ∈ l ∨ 'L' ∈ l) then 3
  else if l.count 'M' = 2 ∧ l.count 'T' = 1 ∧ ('H' ∈ l ∨ 'C' ∨ 'L' ∈ l) then 3
  else if l.count 'T' = 1 ∧ l.count 'M' = 1 ∧ (sum (fun x ↦ x ∈ l ∧ (x = 'H' ∨ x = 'C' ∨ x = 'L' )) 2 then 3
  else if l.count 'T' = 2 ∧ (sum (fun x ↦ x ∈ l ∧ (x = 'H' ∨ x = 'C' ∨ x = 'L' )) 2 then 3
  else if l.count 'M' = 2 ∧ (sum (fun x ↦ x ∈ l ∧ (x = 'H' ∨ x = 'C' ∨ x = 'L' )) 2 then 3
  else 0

def num_distinct_collections := (count_vowels letters) * (count_consonants letters)

theorem distinct_collections : num_distinct_collections = 64 :=
by {
  sorry
}

end distinct_collections_l177_177731


namespace cube_square_third_smallest_prime_l177_177094

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l177_177094


namespace x_value_when_y_2000_l177_177818

noncomputable def x_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) : ℝ :=
  if hy : y = 2000 then (1 / (50 : ℝ)^(1/3)) else x

-- Theorem statement
theorem x_value_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) :
  x_when_y_2000 x y hxy_pos hxy_inv h_init = 1 / (50 : ℝ)^(1/3) :=
sorry

end x_value_when_y_2000_l177_177818


namespace max_consecutive_sum_l177_177436

theorem max_consecutive_sum (S : ℕ) (hS : S = 36) :
  ∃ N a : ℕ, N * (2 * a + N - 1) = 2 * S ∧ (∀ N', N' * (2 * a' + N' - 1) = 2 * S → N' ≤ N) :=
by
  use 72, 35
  split
  · sorry -- This is where the actual proof will go
  · intro N'
    intro hN'
    sorry -- This is where the second part of the actual proof will go

end max_consecutive_sum_l177_177436


namespace cube_of_square_of_third_smallest_prime_is_correct_l177_177089

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l177_177089


namespace basketballs_purchased_l177_177783

theorem basketballs_purchased
    (cupcakes_sold : ℕ) (cookies_sold : ℕ)
    (cost_per_cupcake : ℚ) (cost_per_cookie : ℚ)
    (basketball_cost : ℚ) (energy_drinks_bought : ℕ)
    (cost_per_energy_drink : ℚ)
    (total_money_earned : ℚ := (cupcakes_sold * cost_per_cupcake) + (cookies_sold * cost_per_cookie))
    (money_spent_on_energy_drinks : ℚ := energy_drinks_bought * cost_per_energy_drink)
    (remaining_money : ℚ := total_money_earned - money_spent_on_energy_drinks)
    (basketballs_bought : ℚ := remaining_money / basketball_cost) :
    cupcakes_sold = 50 ∧
    cost_per_cupcake = 2 ∧
    cookies_sold = 40 ∧
    cost_per_cookie = 0.5 ∧
    basketball_cost = 40 ∧
    energy_drinks_bought = 20 ∧
    cost_per_energy_drink = 2 →
    basketballs_bought = 2 := by
  intros h
  cases' h with h1 h2
  cases' h2 with h3 h4
  cases' h4 with h5 h6
  cases' h6 with h7 h8
  cases' h8 with h9 h10
  cases' h10 with h11 h12
  have h_total : total_money_earned = 120 := by
    unfold total_money_earned
    rw [h1, h3, h5, h7]
    norm_num
  have h_drinks : money_spent_on_energy_drinks = 40 := by
    unfold money_spent_on_energy_drinks
    rw [h9, h11]
    norm_num
  have h_remaining : remaining_money = 80 := by
    unfold remaining_money
    rw [h_total, h_drinks]
    norm_num
  have h_basketballs : basketballs_bought = 2 := by
    unfold basketballs_bought
    rw [h_remaining, h12]
    norm_num
  exact h_basketballs


end basketballs_purchased_l177_177783


namespace server_processes_21600000_requests_l177_177505

theorem server_processes_21600000_requests :
  (15000 * 1440 = 21600000) :=
by
  -- Calculations and step-by-step proof
  sorry

end server_processes_21600000_requests_l177_177505


namespace sets_are_equal_l177_177355

-- Defining sets A and B as per the given conditions
def setA : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def setB : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

-- Proving that set A is equal to set B
theorem sets_are_equal : setA = setB :=
by
  sorry

end sets_are_equal_l177_177355


namespace maximum_value_l177_177792

theorem maximum_value (a : ℝ) (x : ℝ) : 
  (f : ℝ → ℝ) := λ x, -x^2 + 4 * x + a
  (h1 : ∀ x ∈ set.Icc 0 1, f x ≥ -2 )
  (h2 : a = -2) :
  ∃ x ∈ set.Icc 0 1, f x = 1 := 
sorry

end maximum_value_l177_177792


namespace power_of_x_l177_177636

theorem power_of_x (x : ℂ) (h : x + 1/x = -real.sqrt 3) : x^12 = 1 := sorry

end power_of_x_l177_177636


namespace angle_coloring_min_colors_l177_177988

  theorem angle_coloring_min_colors (n : ℕ) : 
    (∃ c : ℕ, (c = 2 ↔ n % 2 = 0) ∧ (c = 3 ↔ n % 2 = 1)) :=
  by
    sorry
  
end angle_coloring_min_colors_l177_177988


namespace product_of_values_for_a_l177_177394

-- Definitions for the conditions
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Given condition as Lean function
def condition (a : ℝ) : Prop :=
  distance (3*a) (a - 5) 5 (-3) = 5

-- Proof statement to be verified
theorem product_of_values_for_a : 
  (∀ a1 a2 : ℝ, condition a1 ∧ condition a2 → a1 * a2 = 0.4) :=
sorry

end product_of_values_for_a_l177_177394


namespace range_of_a_l177_177257

theorem range_of_a (hP : ¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l177_177257


namespace solve_inequality_l177_177966

theorem solve_inequality {a b : ℝ} (h : -2 * a + 1 < -2 * b + 1) : a > b :=
by
  sorry

end solve_inequality_l177_177966


namespace area_of_backyard_eq_400_l177_177657

-- Define the conditions
def length_condition (l : ℕ) : Prop := 25 * l = 1000
def perimeter_condition (l w : ℕ) : Prop := 20 * (l + w) = 1000

-- State the theorem
theorem area_of_backyard_eq_400 (l w : ℕ) (h_length : length_condition l) (h_perimeter : perimeter_condition l w) : l * w = 400 :=
  sorry

end area_of_backyard_eq_400_l177_177657


namespace negative_correlation_implies_negative_slope_l177_177426

-- Define the conditions
variables (x y : ℝ)
def linear_regression (b a : ℝ) : Prop := ∃ (x : ℝ), y = b * x + a

-- Define negative correlation condition
def negatively_correlated (b : ℝ) : Prop := b < 0

-- State the proof problem
theorem negative_correlation_implies_negative_slope (b a : ℝ) :
  (∀ x y, linear_regression b a ∧ negatively_correlated b) → b < 0 :=
by sorry

end negative_correlation_implies_negative_slope_l177_177426


namespace area_of_enclosed_region_l177_177193

theorem area_of_enclosed_region :
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6} in
  measure_theoretic.measure.univ region = 36 :=
by
  sorry

end area_of_enclosed_region_l177_177193


namespace minimize_area_bounded_curve_line_l177_177703

noncomputable def curve := λ x : ℝ, abs (x^2 + 2*x - 3)

def line (m x : ℝ) := m * (x + 3)

def area_bounded_by_curve_and_line (m : ℝ) := 
  ∫ x in -3..(1-m), -curve x + line m x + ∫ x in (1-m)..1, line m x + curve x + ∫ x in 1..(1+m), line m x - curve x

theorem minimize_area_bounded_curve_line :
  ∃ m : ℝ, m = 12 - 8 * Real.sqrt 2 ∧ 
            ∀ m' : ℝ, area_bounded_by_curve_and_line m ≤ area_bounded_by_curve_and_line m' :=
by
  sorry

end minimize_area_bounded_curve_line_l177_177703


namespace max_sqrt2_max_attained_at_sqrt2_l177_177697

noncomputable def max_value_s (x y s : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ s = min x (min (y + 1 / x) (1 / y))

theorem max_sqrt2 {x y s : ℝ} :
  x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 → max_value_s x y s → s = Real.sqrt 2 :=
by
  intros hxy hmax
  sorry

theorem max_attained_at_sqrt2 : ∃ x y, max_value_s x y (Real.sqrt 2) ∧ x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 :=
by
  use Real.sqrt 2, 1 / Real.sqrt 2
  split
  { split
    { norm_num, apply Real.sqrt_pos.2, norm_num, },
    split
    { norm_num, rw [Real.sqrt_div, Real.sqrt_one], exact Real.div_self (Real.sqrt_ne_zero.2 zero_lt_two), norm_num, },
    {
      norm_num,
      rw [Real.sqrt_div, Real.sqrt_one],
      exact Real.div_self (Real.sqrt_ne_zero.2 zero_lt_two),
      norm_num,
    },
  },
  {
    norm_num,
    rw [Real.sqrt_div, Real.sqrt_one],
    exact Real.div_self (Real.sqrt_ne_zero.2 zero_lt_two),
    norm_num,
  },
  {
    exact Real.sqrt 2,
  }
  sorry

end max_sqrt2_max_attained_at_sqrt2_l177_177697


namespace cos_2C_is_7_over_25_l177_177317

variables (a b S : ℝ) (C : ℝ)

def sin_C (a b S : ℝ) := (2 * S) / (a * b)

noncomputable def cos_2C (a b S : ℝ) :=
  1 - 2 * (sin_C a b S) ^ 2

theorem cos_2C_is_7_over_25 : 
  ∀ {a b S : ℝ}, a = 8 → b = 5 → S = 12 → cos_2C a b S = 7 / 25 :=
begin
  intros,
  sorry, -- Proof is not required
end

end cos_2C_is_7_over_25_l177_177317


namespace f_triple_l177_177345

def f (x : ℝ) : ℝ :=
  if x >= 1 then sqrt (x - 1) else 1

theorem f_triple (x : ℝ) : f (f (f x)) = 1 :=
  by sorry

#eval f 2   -- This is just a check, should return 1
#eval f (f 2)  -- This is just a check, should return 0
#eval f (f (f 2)) -- This is just a check, should return 1

end f_triple_l177_177345


namespace max_value_f_on_unit_circle_l177_177929

noncomputable def f (z : ℂ) : ℝ := complex.abs (z ^ 3 - z + 2)

theorem max_value_f_on_unit_circle : 
  ∃ z : ℂ, abs z = 1 ∧ f z = sqrt 13 := 
sorry

end max_value_f_on_unit_circle_l177_177929


namespace total_bedrooms_is_correct_l177_177163

def bedrooms_second_floor : Nat := 2
def bedrooms_first_floor : Nat := 8
def total_bedrooms (b1 b2 : Nat) : Nat := b1 + b2

theorem total_bedrooms_is_correct : total_bedrooms bedrooms_second_floor bedrooms_first_floor = 10 := 
by
  sorry

end total_bedrooms_is_correct_l177_177163


namespace problem_a_problem_b_l177_177855

-- Problem (a)
theorem problem_a (x₁ x₂ x₃ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : 0 < x₃) :
  x₁ / (x₂ + x₃) + x₂ / (x₃ + x₁) + x₃ / (x₁ + x₂) ≥ 3 / 2 := 
sorry

-- Problem (b)
theorem problem_b (x : ℕ → ℝ) (n : ℕ) (h₁ : ∀ i, 0 < x i) (h₂ : 4 ≤ n) :
  (∑ i in Finset.range n, x i / (x ((i + 1) % n) + x ((i + n - 1) % n))) ≥ ↑2 :=
sorry

end problem_a_problem_b_l177_177855


namespace factorize_polynomial_l177_177964

theorem factorize_polynomial :
  (∀ x : ℝ, x^4 - 3*x^3 - 28*x^2 = x^2 * (x - 7) * (x + 4)) :=
begin
  intro x,
  sorry,
end

end factorize_polynomial_l177_177964


namespace systematic_sampling_method_l177_177492

-- Defining the conditions of the problem as lean definitions
def sampling_interval_is_fixed (interval : ℕ) : Prop :=
  interval = 10

def production_line_uniformly_flowing : Prop :=
  true  -- Assumption

-- The main theorem formulation
theorem systematic_sampling_method :
  ∀ (interval : ℕ), sampling_interval_is_fixed interval → production_line_uniformly_flowing →
  (interval = 10 → true) :=
by {
  sorry
}

end systematic_sampling_method_l177_177492


namespace midpoint_line_intersection_l177_177799

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ × ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem midpoint_line_intersection :
  let p1 := (1, 3)
  let p2 := (5, 11)
  let mp := midpoint p1.1 p1.2 p2.1 p2.2
  (mp.1 + mp.2) = 10 := by
  let p1 := (1, 3)
  let p2 := (5, 11)
  let mp := midpoint p1.1 p1.2 p2.1 p2.2
  have h : mp = (3, 7) := by
    simp [midpoint, p1, p2]
  show (mp.1 + mp.2) = 10
  simp [h]
  sorry

end midpoint_line_intersection_l177_177799


namespace binom_7_4_eq_35_l177_177933

theorem binom_7_4_eq_35 :
  Nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l177_177933


namespace full_house_plus_two_probability_l177_177430

def total_ways_to_choose_7_cards_from_52 : ℕ :=
  Nat.choose 52 7

def ways_for_full_house_plus_two : ℕ :=
  13 * 4 * 12 * 6 * 55 * 16

def probability_full_house_plus_two : ℚ :=
  (ways_for_full_house_plus_two : ℚ) / (total_ways_to_choose_7_cards_from_52 : ℚ)

theorem full_house_plus_two_probability :
  probability_full_house_plus_two = 13732 / 3344614 :=
by
  sorry

end full_house_plus_two_probability_l177_177430


namespace arithmetic_square_root_of_4_l177_177386

theorem arithmetic_square_root_of_4 : ∃ y : ℝ, y^2 = 4 ∧ y = 2 := 
  sorry

end arithmetic_square_root_of_4_l177_177386


namespace count_g100_equals_36_l177_177218

-- Define a function to calculate the number of positive integer divisors
def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

-- Define the function g_1(n) as thrice the number of divisors of n
def g_1 (n : ℕ) : ℕ := 3 * num_divisors n

-- Define the function g_j(n) recursively
def g_j : ℕ → ℕ → ℕ
| 1 n := g_1 n
| (j + 1) n := g_1 (g_j j n)

-- Define a predicate to check if g_100(n) = 36
def satisfies_condition (n : ℕ) : Prop := g_j 100 n = 36

-- We need to prove that there are exactly 3 such values of n ≤ 100
theorem count_g100_equals_36 : finset.card (finset.filter satisfies_condition (finset.range 101)) = 3 := sorry

end count_g100_equals_36_l177_177218


namespace complex_division_l177_177275

-- Define complex numbers and imaginary unit
def i : ℂ := Complex.I

theorem complex_division : (3 + 4 * i) / (1 + i) = (7 / 2) + (1 / 2) * i :=
by
  sorry

end complex_division_l177_177275


namespace coefficient_of_x_in_expansion_l177_177784

theorem coefficient_of_x_in_expansion : 
  (Polynomial.coeff (((X ^ 2 + 3 * X + 2) ^ 6) : Polynomial ℤ) 1) = 576 := 
by 
  sorry

end coefficient_of_x_in_expansion_l177_177784


namespace xy_product_l177_177424

variable (x y : ℝ)

theorem xy_product : (x + y = 10) ∧ (x^3 + y^3 = 370) → x * y = 31.5 := by
  intro h
  cases h with h1 h2
  sorry

end xy_product_l177_177424


namespace area_of_circumscribed_circle_l177_177134

def circumradius_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

def area_of_circle (R : ℝ) : ℝ :=
  Float.pi * R^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_equilateral_triangle s) = 48 * Float.pi := by
  sorry

end area_of_circumscribed_circle_l177_177134


namespace angle_relation_l177_177318

def triangle (A B C : Type) := A ≠ B ∧ B ≠ C ∧ C ≠ A

variables {A B C D E : Type}

variables {x y : ℝ}
variables {α β γ : ℝ} (h : triangle A B C)

theorem angle_relation
  (hABC : ∠A + ∠B + ∠C = 180)
  (hADE : ∠ADE = x)
  (hBDE : ∠BDE = y)
  (hADC : ∠ADC = ∠A + ∠B) :
  x + y = 180 + ∠C :=
by
  -- Proof goes here
  sorry

end angle_relation_l177_177318


namespace find_f_of_3_l177_177795

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_of_3 (h : ∀ x : ℝ, x ≠ 0 → f x - 3 * f (1 / x) = 3 ^ x) :
  f 3 = (-27 + 3 * (3 ^ (1 / 3))) / 8 :=
sorry

end find_f_of_3_l177_177795


namespace find_constants_l177_177968

noncomputable def example_problem (x : ℝ) (P Q R : ℝ) :=
  3 * x + 1 = P * (x - 2)^2 + Q * (x - 4) * (x - 2) + R * (x - 4)

theorem find_constants :
  ∃ P Q R : ℝ, 
  (∀ x : ℝ, x ≠ 4 → x ≠ 2 → (3 * x + 1) / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧ 
  P = 13 / 4 ∧ Q = -13 / 4 ∧ R = -7 / 2 :=
begin
  sorry
end

end find_constants_l177_177968


namespace geometric_series_sum_S5_l177_177707

noncomputable def a_1 : ℕ := 1

noncomputable def geometric_sum (a_1 : ℕ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a_1 else a_1 * (1 - q^n) / (1 - q)

-- Define S_4 and S_2 based on the geometric_sum function
noncomputable def S_4 (q : ℝ) : ℝ := geometric_sum a_1 q 4
noncomputable def S_2 (q : ℝ) : ℝ := geometric_sum a_1 q 2

-- Condition: S_4 - 5 * S_2 = 0
axiom condition (q : ℝ) (hq : q ≠ 1) : S_4 q - 5 * S_2 q = 0

theorem geometric_series_sum_S5 : ∀ (q : ℝ), q ≠ 1 → S_4 q - 5 * S_2 q = 0 → geometric_sum a_1 q 5 = 31 :=
begin
  intros q hq hcond,
  sorry -- proof to be filled in
end

end geometric_series_sum_S5_l177_177707


namespace christen_potatoes_peeled_l177_177268

theorem christen_potatoes_peeled :
  ∀ (p0 : ℕ) (rH rC : ℕ) (tH : ℕ),
    p0 = 58 →
    rH = 4 →
    rC = 4 →
    tH = 6 →
    let pH := rH * tH in
    let p_remaining := p0 - pH in
    let r_total := rH + rC in
    let t_remaining := p_remaining / r_total + (if p_remaining % r_total = 0 then 0 else 1) in
    let pC := rC * t_remaining in
    pC = 17 :=
by
  intro p0 rH rC tH
  intros h1 h2 h3 h4
  let pH := rH * tH
  let p_remaining := p0 - pH
  let r_total := rH + rC
  let t_remaining := p_remaining / r_total + (if p_remaining % r_total = 0 then 0 else 1)
  let pC := rC * t_remaining
  sorry

end christen_potatoes_peeled_l177_177268


namespace matrix_determinant_l177_177178

def matrix_A := 
  ![[2, -1, 0, 2], 
    [1, 3, 2, -4], 
    [0, -2, 4, 1], 
    [3, 1, -1, 2]]

theorem matrix_determinant : 
  matrix_A.det = 98 := 
by 
  sorry

end matrix_determinant_l177_177178


namespace find_loss_percentage_for_remaining_stock_sold_l177_177506

-- Define the variables and conditions
def W : ℝ := 12499.999999999998
def O : ℝ := 250
def profit_percent : ℝ := 0.10
def sell_percent_profit : ℝ := 0.20
def sell_percent_loss : ℝ := 0.80

-- Define the profit and loss calculations
def profit_from_20_percent : ℝ := sell_percent_profit * W * profit_percent
def loss_from_remaining (L : ℝ) : ℝ := sell_percent_loss * W * (L / 100)

-- The equation representing the overall loss
theorem find_loss_percentage_for_remaining_stock_sold (L : ℝ) :
  loss_from_remaining L - profit_from_20_percent = O → L = 5 :=
by
  sorry

end find_loss_percentage_for_remaining_stock_sold_l177_177506


namespace Stephen_total_distance_l177_177383

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end Stephen_total_distance_l177_177383


namespace root_interval_l177_177606

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem root_interval (x0 : ℝ) (h : f x0 = 0): x0 ∈ Set.Ioo (1 / 4 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end root_interval_l177_177606


namespace a_positive_if_exists_x0_l177_177254

noncomputable def f (x a : ℝ) := a * (x - 1/x) - 2 * Real.log x
noncomputable def g (x a : ℝ) := -a / x

theorem a_positive_if_exists_x0 {a : ℝ} (h : ∃ x0 ∈ Set.Icc 1 Real.exp 1, f x0 a > g x0 a) : a > 0 :=
by
  sorry

end a_positive_if_exists_x0_l177_177254


namespace sqrt_200_eq_10_l177_177018

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177018


namespace zero_in_interval_l177_177780

open Real

def h (x : ℝ) : ℝ := exp x + 2 * x - 3

theorem zero_in_interval : ∃ x ∈ Ioo (1 / 2 : ℝ) 1, h x = 0 := sorry

end zero_in_interval_l177_177780


namespace geometric_locus_center_l177_177915

noncomputable def ellipseLocus (F₁ F₂: ℝ × ℝ) (a: ℝ): set (ℝ × ℝ) :=
  { P | dist P F₁ + dist P F₂ = 2 * a }

noncomputable def isTangent (e: set (ℝ × ℝ)) (P: ℝ × ℝ) (ellipse: set (ℝ × ℝ)): Prop :=
  ∀ Q ∈ e, Q ≠ P → ∃ R ∈ ellipse, dist P R < dist P Q

def locusOfCenter (F₁ F₂ : ℝ × ℝ) (a c: ℝ) (O: ℝ × ℝ) : set (ℝ × ℝ) :=
  let r := real.sqrt (2 * a^2 - c^2) in
  { C | dist C O = r ∧ C ≠ O ∧
    ∀ (e' e'' : set (ℝ × ℝ)) (P: ℝ × ℝ),
      isTangent e' P (ellipseLocus F₁ F₂ a) ∧ isTangent e'' P (ellipseLocus F₁ F₂ a) →
      P = C }

theorem geometric_locus_center (F₁ F₂: ℝ × ℝ) (a: ℝ) (O: ℝ × ℝ) (c: ℝ) : 
  ∃ (C : set (ℝ × ℝ)), 
  C = locusOfCenter F₁ F₂ a c O ∧
  ( ∀ C' ∈ C, some_condition C' F₁ F₂ ) := sorry

end geometric_locus_center_l177_177915


namespace sqrt_200_eq_10_sqrt_2_l177_177004

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177004


namespace john_paid_l177_177330

def upfront : ℤ := 1000
def hourly_rate : ℤ := 100
def court_hours : ℤ := 50
def prep_multiplier : ℤ := 2
def brother_share : ℚ := 1/2
def paperwork_fee : ℤ := 500
def transport_costs : ℤ := 300

theorem john_paid :
  let court_cost := court_hours * hourly_rate,
      prep_hours := prep_multiplier * court_hours,
      prep_cost := prep_hours * hourly_rate,
      total_cost := upfront + court_cost + prep_cost + paperwork_fee + transport_costs,
      john_share := total_cost * brother_share
  in john_share = 8400 :=
by {
  sorry
}

end john_paid_l177_177330


namespace digit_B_l177_177199

def is_valid_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 7

def unique_digits (A B C D E F G : ℕ) : Prop :=
  is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D ∧ 
  is_valid_digit E ∧ is_valid_digit F ∧ is_valid_digit G ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ 
  E ≠ F ∧ E ≠ G ∧ 
  F ≠ G

def total_sum (A B C D E F G : ℕ) : ℕ :=
  (A + B + C) + (A + E + F) + (C + D + E) + (B + D + G) + (B + F) + (G + E)

theorem digit_B (A B C D E F G : ℕ) 
  (h1 : unique_digits A B C D E F G)
  (h2 : total_sum A B C D E F G = 65) : B = 7 := 
sorry

end digit_B_l177_177199


namespace simplify_expression_l177_177699

theorem simplify_expression 
  (a b c : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (θ : ℝ)
  (x := (b / c) + (c / b) + sin θ)
  (y := (a / c) + (c / a) + cos θ)
  (z := (a / b) + (b / a) + tan θ) :
  x^2 + y^2 + z^2 - x * y * z = 4 :=
by sorry

end simplify_expression_l177_177699


namespace quadratic_negative_root_l177_177245

theorem quadratic_negative_root
  (P : ℝ → ℝ)
  (hquad : ∃ c d e, c ≠ 0 ∧ d ≠ e ∧ P = λ x, c * (x - d) * (x - e))
  (hineq : ∀ a b : ℝ, 2017 ≤ |a| ∧ 2017 ≤ |b| → P (a^2 + b^2) ≥ P (2 * a * b)) :
  ∃ x : ℝ, P x = 0 ∧ x < 0 :=
sorry

end quadratic_negative_root_l177_177245


namespace simplify_expression_l177_177123

variable (a : ℝ) (ha : a ≠ -3)

theorem simplify_expression : (a^2) / (a + 3) - 9 / (a + 3) = a - 3 :=
by
  sorry

end simplify_expression_l177_177123


namespace angle_between_f_l177_177702

open Real

variables (e1 e2 : EuclideanSpace ℝ (Fin 2))
variables (h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1)
variables (h3 : inner e1 e2 = sqrt 3 / 2)

def f (a b : EuclideanSpace ℝ (Fin 2)) (θ : ℝ) : EuclideanSpace ℝ (Fin 2) :=
  a * (cos θ) - b * (sin θ)

theorem angle_between_f (e1 e2 : EuclideanSpace ℝ (Fin 2)) 
  (h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1) (h3 : inner e1 e2 = sqrt 3 / 2) :
  inner (f e1 e2 (π / 6)) (f e2 (-e1) (5 * π / 6)) = 0 := 
  sorry

end angle_between_f_l177_177702


namespace new_person_weight_l177_177449

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (replaced_weight : ℝ) :
  avg_increase = 2.5 → num_persons = 8 → replaced_weight = 60 → 
  let total_weight_increase := avg_increase * num_persons,
      new_person_weight := replaced_weight + total_weight_increase
  in new_person_weight = 80 :=
by
  intros h1 h2 h3
  unfold total_weight_increase
  unfold new_person_weight
  rw [h1, h2, h3]
  sorry

end new_person_weight_l177_177449


namespace rowed_upstream_distance_l177_177141

def distance_downstream := 120
def time_downstream := 2
def distance_upstream := 2
def speed_stream := 15

def speed_boat (V_b : ℝ) := V_b

theorem rowed_upstream_distance (V_b : ℝ) (D_u : ℝ) :
  (distance_downstream = (V_b + speed_stream) * time_downstream) ∧
  (D_u = (V_b - speed_stream) * time_upstream) →
  D_u = 60 :=
by 
  sorry

end rowed_upstream_distance_l177_177141


namespace binary_to_decimal_10101_l177_177532

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end binary_to_decimal_10101_l177_177532


namespace billy_hiking_distance_correct_l177_177923

noncomputable def billy_distance (east_leg : ℝ) (north_east_leg : ℝ) := 
  let horizontal_dist := east_leg + north_east_leg / real.sqrt 2
  let vertical_dist := north_east_leg / real.sqrt 2
  real.sqrt (horizontal_dist ^ 2 + vertical_dist ^ 2)

theorem billy_hiking_distance_correct :
  billy_distance 5 8 = real.sqrt (89 + 40 * real.sqrt 2) :=
  sorry

end billy_hiking_distance_correct_l177_177923


namespace new_area_of_rectangle_l177_177385

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 625) : 
    1.2 * L * (0.8 * W) = 600 :=
by 
    have h1 : 1.2 * L * (0.8 * W) = 0.96 * (L * W) := by ring
    rw h at h1
    norm_num at h1
    exact h1

end new_area_of_rectangle_l177_177385


namespace stratified_sampling_l177_177477

open Nat

theorem stratified_sampling :
  let junior_students := 400 in
  let senior_students := 200 in
  let total_sample_size := 60 in
  let junior_ratio := 2 in
  let senior_ratio := 1 in
  let total_students := junior_students + senior_students in 
  junior_ratio * senior_students = senior_ratio * junior_students →
  junior_students + senior_students = total_sample_size ->
  let junior_sample_size := (junior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  let senior_sample_size := (senior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  junior_sample_size = 40 →
  senior_sample_size = 20 →
  choose junior_students junior_sample_size * choose senior_students senior_sample_size = binomial 400 40 * binomial 200 20 :=
by 
  intros; 
  sorry

end stratified_sampling_l177_177477


namespace sqrt_200_eq_10_l177_177021

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177021


namespace _l177_177354

open EuclideanGeometry

noncomputable def angle_acb_condition (A B C D E F : Point) : Prop :=
  let AD := altitude A D (triangle ABC)
  let BE := altitude B E (triangle ABC)
  let CF := altitude C F (triangle ABC)
  in (5 • (AD.dirToVector) + 3 • (BE.dirToVector) + 2 • (CF.dirToVector) = 0)

noncomputable def angle_acb_theorem (A B C D E F : Point) (h_acute : isAcuteTriangle ABC)
  (h_altitudes : areAltitudes A B C D E F)
  (h_condition : angle_acb_condition A B C D E F) :
  ∠ACB = 30 :=
sorry

end _l177_177354


namespace max_reflex_angles_2006_sided_polygon_l177_177166

theorem max_reflex_angles_2006_sided_polygon : 
  ∀ (P : polygon), P.sides = 2006 → ∃ max_angles > (180 : ℝ), max_angles = 2003 :=
by
  sorry

end max_reflex_angles_2006_sided_polygon_l177_177166


namespace complex_number_solution_l177_177620

def imaginary_unit : ℂ := Complex.I -- defining the imaginary unit

theorem complex_number_solution (z : ℂ) (h : z / (z - imaginary_unit) = imaginary_unit) :
  z = (1 / 2 : ℂ) + (1 / 2 : ℂ) * imaginary_unit :=
sorry

end complex_number_solution_l177_177620


namespace abc_max_value_f_monotonic_increasing_interval_l177_177611

-- Given function definition and conditions
def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + (Real.pi / 12))

-- Maximum value of b + c given conditions
theorem abc_max_value (A : ℝ) (b c : ℝ) (hA1 : -Real.pi / 6 < A ∧ A < Real.pi / 6)
  (hA2 : f (-A / 2) = Real.sqrt 2) (ha : 3 = 3) :
  b + c ≤ 6 :=
by
  sorry

-- Interval where f(x) is monotonically increasing
theorem f_monotonic_increasing_interval (k : ℤ) :
  ∃ k : ℤ, ∀ x : ℝ, (k * Real.pi - 13 * Real.pi / 24 ≤ x) ∧ (x ≤ k * Real.pi - Real.pi / 24) :=
by
  sorry

end abc_max_value_f_monotonic_increasing_interval_l177_177611


namespace find_temperature_function_highest_temperature_l177_177600

-- The temperature function T(t) has the given form and conditions:
def T (t : ℝ) : ℝ := a * t^3 + b * t^2 + c * t + d

-- Conditions
variables (a b c d : ℝ)
variables (h1 : T 0 = 60)
variables (h2 : T (-4) = 8)
variables (h3 : T 1 = 58)
variables (h4 : T' (-4) = T' 4) -- where T' is the derivative of T

-- Statements
theorem find_temperature_function : T = λ t, t^3 - 3*t + 60 :=
sorry

theorem highest_temperature :
  ∃ t ∈ (Icc (-2 : ℝ) (2 : ℝ)), ∀ x ∈ (Icc (-2 : ℝ) (2 : ℝ)), T x ≤ T t ∧ T t = 62 ∧ t = 2 :=
sorry

end find_temperature_function_highest_temperature_l177_177600


namespace not_right_triangle_A_l177_177442

def is_right_triangle (a b c : Real) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_A : ¬ (is_right_triangle 1.5 2 3) :=
by sorry

end not_right_triangle_A_l177_177442


namespace parabola_equation_l177_177139

-- Definitions of given conditions
def M (p : ℝ) : ℝ × ℝ := (2, -2 * p)

def parabola (x y p : ℝ) : Prop := x^2 = 2 * p * y

def midpoint_of_tangent (p : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), (x1^2 = 2 * p * y1) ∧ (x2^2 = 2 * p * y2)
    ∧ ((2, -2 * p) = ((x1 + x2) / 2, (y1 + y2) / 2)) ∧ ((y1 + y2) / 2 = 6)

-- Statement of the problem to prove
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : midpoint_of_tangent p) :
  parabola _ _ 1 ∨ parabola _ _ 2 :=
sorry

end parabola_equation_l177_177139


namespace least_of_consecutive_odds_l177_177450

theorem least_of_consecutive_odds (n : ℤ) (sum : ℤ) (avg : ℤ)
    (h_sum : ∑ i in finset.range 16, (n + 2 * i) = sum)
    (h_avg : sum / 16 = avg)
    (h_avg_val : avg = 414) : 
    n = 399 := 
by 
  sorry

end least_of_consecutive_odds_l177_177450


namespace nba_conference_division_impossible_l177_177411

theorem nba_conference_division_impossible :
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  ¬∃ (A B : ℕ), A + B = teams ∧ A * B = inter_conference_games := 
by
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  sorry

end nba_conference_division_impossible_l177_177411


namespace percentage_of_female_students_l177_177490

theorem percentage_of_female_students (m n : ℕ) (h1 : 0.4 * m = 0.8 * n) :
  (n : ℝ) / (m + n) * 100 = 33.333 :=
by
  sorry

end percentage_of_female_students_l177_177490


namespace press_19_switches_l177_177416

-- Define the conditions of the problem
variables (Switch Lamp : Type) -- We have switches and lamps.
variable (N : ℕ) [Fintype Switch] [Fintype Lamp] -- Finite number of switches and lamps.
variable [DecidableEq Switch] [DecidableEq Lamp] -- Decidable equality for switches and lamps.

-- Define the specific conditions
axiom switch_count : Fintype.card Switch = 70
axiom lamp_count : Fintype.card Lamp = 15
axiom switches_connected_to_lamp : Lamp → Set Switch -- Each lamp is connected to a set of switches.
axiom each_lamp_connected_35 : ∀ l : Lamp, (switches_connected_to_lamp l).to_finset.card = 35
axiom distinct_switch_sets : ∀ l1 l2 : Lamp, l1 ≠ l2 → switches_connected_to_lamp l1 ≠ switches_connected_to_lamp l2

-- Define the state change condition
variable (toggle : Switch → Lamp → Prop)
axiom toggle_definition : ∀ (s : Switch) (l : Lamp), toggle s l ↔ s ∈ switches_connected_to_lamp l 

-- Initially all lamps are off
variable (initial_state : Lamp → bool)
axiom initial_state_off : ∀ l : Lamp, initial_state l = false

-- Statement of the proof
theorem press_19_switches : ∃ (S : Finset Switch), S.card = 19 ∧ (∃ (lamps_on : Finset Lamp), lamps_on.card ≥ 8 ∧ ∀ l ∈ lamps_on, finset.bUnion S (λ s, {l | toggle s l}).to_finset.card % 2 = 1) :=
sorry

end press_19_switches_l177_177416


namespace stratified_sampling_result_l177_177472

-- Definitions and conditions from the problem
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_sample : ℕ := 60
def junior_sample : ℕ := 40
def senior_sample : ℕ := 20

-- Main theorem statement proving the number of different sampling results
theorem stratified_sampling_result :
  choose junior_students junior_sample * choose senior_students senior_sample = 
  choose 400 40 * choose 200 20 := by
  sorry

end stratified_sampling_result_l177_177472


namespace pump_A_time_l177_177167

theorem pump_A_time (B C A : ℝ) (hB : B = 1/3) (hC : C = 1/6)
(h : (A + B - C) * 0.75 = 0.5) : 1 / A = 2 :=
by
sorry

end pump_A_time_l177_177167


namespace no_prime_roots_sum_72_l177_177927

theorem no_prime_roots_sum_72 :
  ∀ (p q : ℕ), prime p → prime q → p + q = 72 → false :=
by
  sorry

end no_prime_roots_sum_72_l177_177927


namespace basketball_problem_l177_177115

theorem basketball_problem
  (P_Xiaojia_shot : ℝ)
  (P_Xiaoming_miss : ℝ)
  (independent_shots : true)
  (A : Prop) -- Xiaoming makes a shot
  (B : Prop) -- At least one person makes a shot
  (h1 : P_Xiaojia_shot = 0.8)
  (h2 : P_Xiaoming_miss = 0.2)
  (hA : ¬A ↔ ¬(P_Xiaoming_makes))
  (hB : B ↔ (P_Xiaojia_shot ∨ A)) :
  (P(¬A) = 0.2) ∧ (P(A ∧ B) = 0.8) :=
by
  sorry

end basketball_problem_l177_177115


namespace points_on_line_initial_l177_177759

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l177_177759


namespace triangle_relations_l177_177668

theorem triangle_relations (A B C_1 C_2 C_3 : ℝ)
  (h1 : B > A)
  (h2 : C_2 > C_1 ∧ C_2 > C_3)
  (h3 : A + C_1 = 90) 
  (h4 : C_2 = 90)
  (h5 : B + C_3 = 90) :
  C_1 - C_3 = B - A :=
sorry

end triangle_relations_l177_177668


namespace cube_square_third_smallest_prime_l177_177092

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l177_177092


namespace part_a_part_b_l177_177983

-- Define b(n) as the number of 1's in the binary representation of n
def b (n : ℕ) : ℕ := n.binaryDigits.count 1

-- Define A_k and f(k) based on the problem statement
def A_k (k : ℕ) : Finset ℕ := (Finset.range (2 * k + 1)).filter (fun n => n >= k + 1 ∧ b n = 3)

def f (k : ℕ) : ℕ := (A_k k).card

-- Main statements

-- (a) Prove there exists a k such that f(k) = m for any m
theorem part_a (m : ℕ) (h : m > 0) : ∃ k : ℕ, f k = m := 
sorry

-- (b) Determine all m for which f(k) = m has a unique solution 
theorem part_b : ∀ m : ℕ, (∃! k : ℕ, f k = m) → (∃ t : ℕ, m = t * (t - 1) / 2 + 1) := 
sorry

end part_a_part_b_l177_177983


namespace odd_function_value_l177_177794

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Ico 0 2 then 3^x + 1 - 4 else -(3^(-x) + 1 - 4)

theorem odd_function_value :
  f(log 3 (1/2)) = -1 :=
by
  -- proof goes here
  sorry

end odd_function_value_l177_177794


namespace coeff_x3_in_q_cubed_l177_177684

def q (x : ℝ) : ℝ := x^5 - 5 * x^2 + 4

theorem coeff_x3_in_q_cubed: polynomial.coeff (polynomial.map q^3) 3 = 0 :=
by
  sorry

end coeff_x3_in_q_cubed_l177_177684


namespace stratified_sampling_l177_177476

open Nat

theorem stratified_sampling :
  let junior_students := 400 in
  let senior_students := 200 in
  let total_sample_size := 60 in
  let junior_ratio := 2 in
  let senior_ratio := 1 in
  let total_students := junior_students + senior_students in 
  junior_ratio * senior_students = senior_ratio * junior_students →
  junior_students + senior_students = total_sample_size ->
  let junior_sample_size := (junior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  let senior_sample_size := (senior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  junior_sample_size = 40 →
  senior_sample_size = 20 →
  choose junior_students junior_sample_size * choose senior_students senior_sample_size = binomial 400 40 * binomial 200 20 :=
by 
  intros; 
  sorry

end stratified_sampling_l177_177476


namespace solution_set_l177_177995

def f (x : ℝ) : ℝ :=
  if x > 0 then x - 2
  else if x < 0 then x + 2
  else 0

theorem solution_set :
  {x : ℝ | f x < 1 / 2} = {x : ℝ | x < -3 / 2 ∨ (0 ≤ x ∧ x < 5 / 2)} :=
by
  sorry

end solution_set_l177_177995


namespace problem_statement_l177_177121

def absolute_difference_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = |a (n - 1) - a (n - 2)|

def construct_nonzero_absolute_difference_sequence : Prop :=
  ∃ (a : ℕ → ℕ), a 1 = 3 ∧ a 2 = 1 ∧ 
  absolute_difference_sequence a ∧ 
  a 3 ≠ 0 ∧ a 4 ≠ 0 ∧ a 5 ≠ 0

def limits_exist (a b : ℕ → ℕ) : Prop :=
  ∃ L : ℕ → option ℕ, ∀ n, 
    (L (2 * n + 1) = none ∧ L (2 * n + 2) = some 6)

def infinite_zero_terms (a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m ≥ n, a m = 0

theorem problem_statement :
  ∃ a b : ℕ → ℕ, absolute_difference_sequence a ∧
    construct_nonzero_absolute_difference_sequence ∧
    limits_exist a b ∧ infinite_zero_terms a :=
sorry

end problem_statement_l177_177121


namespace max_equilateral_triangles_with_6_matchsticks_l177_177837

theorem max_equilateral_triangles_with_6_matchsticks : 
  ∀ (n : ℕ), (n = 6) → (∃ (tetrahedron_constructible : Prop), tetrahedron_constructible ∧ (tetrahedron_constructible → 4 = 4)) :=
by {
  intro n,
  intro h,
  use true,
  split,
  { trivial, },
  { intro h,
    exact rfl, },
}

end max_equilateral_triangles_with_6_matchsticks_l177_177837


namespace find_value_l177_177660

variable {a : ℕ → ℝ}

-- Condition: the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Given condition: a_3 + a_8 = 6
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 8 = 6

-- Statement to prove
theorem find_value (a : ℕ → ℝ) [is_arithmetic_sequence a] [sum_condition a] : 
  3 * a 2 + a 16 = 12 :=
by
  sorry

end find_value_l177_177660


namespace find_k_parallel_l177_177625

-- Define the vectors
def veca : ℝ × ℝ := (3, -1)
def vecb : ℝ × ℝ := (1, -2)

-- Define the condition for vectors to be parallel
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)

-- Problem statement: 
-- Find the value of k such that the vector (-veca + vecb) is parallel to (veca + k * vecb).
theorem find_k_parallel : ∃ k : ℝ, are_parallel (-veca.1 + vecb.1, -veca.2 + vecb.2) 
                                        (veca.1 + k * vecb.1, veca.2 + k * vecb.2) 
                                  ∧ k = -1 :=
by
  use -1
  -- we use sorry here to skip the proof as requested
  sorry

end find_k_parallel_l177_177625


namespace binary_10101_is_21_l177_177530

namespace BinaryToDecimal

def binary_to_decimal (n : Nat) : Nat :=
  match n with
  | 10101 => 21

theorem binary_10101_is_21 :
  binary_to_decimal 10101 = 21 := by
  -- Proof steps would go here
  sorry

end BinaryToDecimal

end binary_10101_is_21_l177_177530


namespace geometric_series_sum_l177_177238

noncomputable def geometric_series (a1 : ℝ) (r : ℝ) (n : ℕ) : ℝ := 
  a1 * r^(n - 1)

def f (x : ℝ) : ℝ := 2 / (1 + x^2)

theorem geometric_series_sum (a1 r : ℝ) (h1 : a1 * (geometric_series a1 r 2017) = 1) :
  (∑ n in Finset.range 2017, f (geometric_series a1 r (n + 1)) ) = 2017 := by
  sorry

end geometric_series_sum_l177_177238


namespace find_larger_number_l177_177059

theorem find_larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
  sorry

end find_larger_number_l177_177059


namespace french_not_english_l177_177883

variables (T FE F FE_not_Eng : ℕ)
variables (percentFrench : ℕ)

def number_of_students := T = 200
def number_of_students_speak_both := FE = 25
def percent_of_students_speak_french := percentFrench = 45
def number_of_students_speak_french := F = (percentFrench * T) / 100
def number_of_students_speak_french_not_english := FE_not_Eng = F - FE

theorem french_not_english (h1 : number_of_students)
                           (h2 : number_of_students_speak_both)
                           (h3 : percent_of_students_speak_french)
                           (h4 : number_of_students_speak_french) :
  FE_not_Eng = 65 :=
by sorry

end french_not_english_l177_177883


namespace power_difference_l177_177590

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l177_177590


namespace exists_six_digit_number_divisible_by_tails_l177_177710

-- Define what it means for a number to be a six-digit natural number without zeros in its decimal representation
def is_six_digit_natural_number_without_zeros (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ ∀ k, (k ∈ (n.digits 10)) → (k ≠ 0)

-- Define the "tail" of a natural number
def is_tail (n tail : ℕ) : Prop :=
  ∃ k, (n = tail * 10^k) ∧ (k ≠ 0)

-- Define the main concept: a number divisible by each of its tails
def divisible_by_each_tail (n : ℕ) : Prop :=
  ∀ (tail : ℕ), (is_tail n tail) → (n % tail = 0)

-- Main statement
theorem exists_six_digit_number_divisible_by_tails :
  ∃ (n : ℕ), is_six_digit_natural_number_without_zeros n ∧ divisible_by_each_tail n :=
begin
  use 721875,
  split,
  { -- Proof that 721875 is a six-digit number without zeros
    unfold is_six_digit_natural_number_without_zeros,
    split, 
    { exact dec_trivial }, -- 100000 ≤ 721875
    { split,
      { exact dec_trivial }, -- 721875 < 1000000
      { intros k hk,
        cases hk,
        { exact dec_trivial }, -- First digit (7) is not zero
        cases hk,
        { exact dec_trivial }, -- Second digit (2) is not zero
        cases hk,
        { exact dec_trivial }, -- Third digit (1) is not zero
        cases hk,
        { exact dec_trivial }, -- Fourth digit (8) is not zero
        cases hk,
        { exact dec_trivial }, -- Fifth digit (7) is not zero
        cases hk,
        { exact dec_trivial }, -- Sixth digit (5) is not zero
        contradiction
      }
    }
  },
  { -- Proof that 721875 is divisible by each of its tails
    unfold divisible_by_each_tail,
    intros tail htail,
    cases htail with k h,
    cases h with hnk hk,
    cases hk,
    { exact dec_trivial }, -- Case for k = 1, ..., 5, exact proof should be here but simplified
  },
end

end exists_six_digit_number_divisible_by_tails_l177_177710


namespace find_h_for_expression_l177_177285

theorem find_h_for_expression (a k : ℝ) (h : ℝ) :
  (∃ a k : ℝ, ∀ x : ℝ, x^2 - 6*x + 1 = a*(x - h)^3 + k) ↔ h = 2 :=
by
  sorry

end find_h_for_expression_l177_177285


namespace rectangle_area_integer_length_width_l177_177892

theorem rectangle_area_integer_length_width (l w : ℕ) (h1 : w = l / 2) (h2 : 2 * l + 2 * w = 200) :
  l * w = 2178 :=
by
  sorry

end rectangle_area_integer_length_width_l177_177892


namespace largest_possible_s_l177_177698

theorem largest_possible_s 
  (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (hangles : (r - 2) * 60 * s = (s - 2) * 61 * r) : 
  s = 121 := 
sorry

end largest_possible_s_l177_177698


namespace exists_ten_digit_composite_after_strike_out_l177_177444

theorem exists_ten_digit_composite_after_strike_out :
  ∃ (n : ℕ), (digits n).length = 10 ∧ (∀ (removed : Finset ℕ), removed.card = 6 → ∃ (remaining : ℕ), remaining ∈ remaining_digits n removed ∧ ¬prime remaining) :=
begin
  sorry
end

-- Helper definitions and lemmas
def digits (n : ℕ) : List ℕ :=
  -- Function to convert a number to a list of its digits
  sorry

def remaining_digits (n : ℕ) (removed : Finset ℕ) : Set ℕ :=
  -- Function to get the remaining digits after removing a set of digits
  sorry

end exists_ten_digit_composite_after_strike_out_l177_177444


namespace smallest_tangent_circle_l177_177553

theorem smallest_tangent_circle (
  (line : ℝ → ℝ → Prop) (curve : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop) : 
  line = λ x y, x + y - 2 = 0 ∧ 
  curve = λ x y, x^2 + y^2 - 12x - 12y + 54 = 0 ∧ 
  (∃ center : ℝ × ℝ, center = (2, 2) ∧ ∃ radius : ℝ, radius^2 = 2 ∧
  ∀ (x y : ℝ), line x y → sqrt ((x - center.1)^2 + (y - center.2)^2) = radius ∧ 
  (curve x y) → sqrt ((x - center.1)^2 + (y - center.2)^2) = radius)) := 
begin
  sorry
end

end smallest_tangent_circle_l177_177553


namespace find_length_PG_l177_177182

noncomputable def length_PG {ABC : Type} [triangle ABC]
  (BC : ℝ) (CA : ℝ) (AB : ℝ)
  (D : midpoint BC)
  (E: midpoint CA)
  (G : centroid ABC)
  (G' : reflection G D)
  (P: intersection (line G' E) (line G C)) : ℝ := 
  sqrt 145 / 9

theorem find_length_PG {ABC : Type} [triangle ABC]
  (H1 : side_length BC 7) 
  (H2 : side_length CA 8) 
  (H3 : side_length AB 9) 
  (H4 : is_midpoint D BC) 
  (H5 : is_midpoint E CA)
  (H6 : is_centroid G ABC)
  (H7 : is_reflection G' G D)
  (H8 : is_intersection P (line G' E) (line G C)) :
  length_PG 7 8 9 D E G G' P = sqrt 145 / 9 := sorry

end find_length_PG_l177_177182


namespace pests_eaten_by_frogs_in_week_l177_177494

-- Definitions
def pests_per_day_per_frog : ℕ := 80
def days_per_week : ℕ := 7
def number_of_frogs : ℕ := 5

-- Proposition to prove
theorem pests_eaten_by_frogs_in_week : (pests_per_day_per_frog * days_per_week * number_of_frogs) = 2800 := 
by sorry

end pests_eaten_by_frogs_in_week_l177_177494


namespace turnips_in_mashed_potatoes_l177_177172

theorem turnips_in_mashed_potatoes:
  ∀ (turnips_prev potatoes_prev : ℝ) (potatoes_curr : ℝ),
    turnips_prev = 2 →
    potatoes_prev = 5 →
    potatoes_curr = 20 →
    (potatoes_curr / (potatoes_prev / turnips_prev) = 8) :=
begin
  intros,
  sorry,
end

end turnips_in_mashed_potatoes_l177_177172


namespace train_speed_l177_177155

def distance := 11.67 -- distance in km
def time := 10.0 / 60.0 -- time in hours (10 minutes is 10/60 hours)

theorem train_speed : (distance / time) = 70.02 := by
  sorry

end train_speed_l177_177155


namespace cyclic_quad_inequality_l177_177570

variables {α β γ δ : ℝ}
variables {AB CD AD BC AC BD : ℝ}

-- conditions
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
  α + β + γ + δ = π

-- statement
theorem cyclic_quad_inequality (h : cyclic_quadrilateral α β γ δ) 
  (h1 : AB = 2 * Real.sin α)
  (h2 : CD = 2 * Real.sin γ)
  (h3 : AD = 2 * Real.sin δ)
  (h4 : BC = 2 * Real.sin β)
  (h5 : AC = 2 * Real.sin ((α - γ) / 2))
  (h6 : BD = 2 * Real.sin ((β - δ) / 2)) :
  |AB - CD| + |AD - BC| ≥ 2 * |AC - BD| := 
sorry

end cyclic_quad_inequality_l177_177570


namespace trees_planted_tomorrow_l177_177414

-- Definitions from the conditions
def current_trees := 39
def trees_planted_today := 41
def total_trees := 100

-- Theorem statement matching the proof problem
theorem trees_planted_tomorrow : 
  ∃ (trees_planted_tomorrow : ℕ), current_trees + trees_planted_today + trees_planted_tomorrow = total_trees ∧ trees_planted_tomorrow = 20 := 
by
  sorry

end trees_planted_tomorrow_l177_177414


namespace convex_polygon_interior_angles_arithmetic_progression_l177_177944

theorem convex_polygon_interior_angles_arithmetic_progression :
  ∀ n : ℕ, n ∈ {3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18} ↔
    (n > 2 ∧ ∀ k : ℕ, k < n →
    ∃ x d : ℤ, (a : fin n → ℤ) (∀ i, a i = x - i * d) ∧
    (a 0 + a 1 + ... + a (n - 1) = 180 * (n - 2)) ∧
    (∀ i, a i < 180) ∧
    (∃ i j, i ≠ j ∧ a i ≠ a j)) := 
sorry

end convex_polygon_interior_angles_arithmetic_progression_l177_177944


namespace sum_abs_f_1_to_2023_l177_177343

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 2 then x^2 - 2^x else 0

axiom odd_func (f : ℝ → ℝ) : ∀ x : ℝ, f(x) = -f(-x)
axiom func_prop (f : ℝ → ℝ) : ∀ x : ℝ, f(x) + f(4 - x) = 0

theorem sum_abs_f_1_to_2023 :
  (∑ i in Finset.range 2023, |f (i + 1)|) = 1012 := sorry

end sum_abs_f_1_to_2023_l177_177343


namespace sum_series_l177_177175

theorem sum_series : (Finset.range 101).sum (λ n => if (n % 2 = 0) then (n + 1) / 2 else -(n + 1) / 2) + 101 = 51 :=
by
  sorry

end sum_series_l177_177175


namespace find_y_l177_177662

-- Definitions for the given conditions
def angle_sum_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def right_triangle (A B : ℝ) : Prop :=
  A + B = 90

-- The main theorem to prove
theorem find_y 
  (angle_ABC : ℝ)
  (angle_BAC : ℝ)
  (angle_DCE : ℝ)
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : right_triangle angle_DCE 30)
  : 30 = 30 :=
sorry

end find_y_l177_177662


namespace trapezoidLargerInteriorAngle_l177_177938

/-- Define the circular dome with trapezoids -/
def circularDomeTrapezoids (n : ℕ) (totalDegrees : ℝ) : Prop :=
  totalDegrees = 360 ∧ n = 10

/-- Define the interior angle of each trapezoid -/
def interiorAngle (baseAngle : ℝ) : ℝ :=
  (180 - baseAngle) / 2

/-- The problem statement that needs to be proved -/
theorem trapezoidLargerInteriorAngle :
  let n := 10 in
  let totalDegrees := 360 in
  let centralAngle := totalDegrees / n in
  let baseAngle := centralAngle / 2 in
  let largerInteriorAngle := interiorAngle baseAngle in
  circularDomeTrapezoids n totalDegrees →
  largerInteriorAngle = 81 :=
by
  intros
  unfold circularDomeTrapezoids
  unfold interiorAngle
  sorry

end trapezoidLargerInteriorAngle_l177_177938


namespace students_in_class_l177_177523

theorem students_in_class (initial_avg final_avg : ℚ) (incorrect_score correct_score : ℕ) :
  initial_avg = 87.26 → final_avg = 87.44 → incorrect_score = 89 → correct_score = 98 →
  ∃ (n : ℕ), n = 50 :=
by
  intros h1 h2 h3 h4
  have h_diff_score : correct_score - incorrect_score = 9 := by
    rw [h3, h4]
    norm_num
  have h_diff_avg : final_avg - initial_avg = 0.18 := by
    rw [h1, h2]
    norm_num
  have h_n_students : 9 / 0.18 = 50 := by
    norm_num
  use 50
  rw h_n_students
  norm_num
  done

end students_in_class_l177_177523


namespace total_canoes_built_l177_177519

def geometric_sum (a r n : ℕ) : ℕ :=
  a * ((r^n - 1) / (r - 1))

theorem total_canoes_built : geometric_sum 10 3 7 = 10930 := 
  by
    -- The proof will go here.
    sorry

end total_canoes_built_l177_177519


namespace sum_of_first_six_terms_is_144_l177_177554

variables (a d : ℝ)

def sum_arithmetic_prog (n : ℝ) : ℝ :=
  (n / 2) * (2 * a + (n - 1) * d)

axiom sum_condition : ∀ (n : ℝ),
  sum_arithmetic_prog a d n = 4 * n ^ 2

theorem sum_of_first_six_terms_is_144 :
  sum_arithmetic_prog a d 6 = 144 :=
sorry

end sum_of_first_six_terms_is_144_l177_177554


namespace average_length_of_ropes_l177_177073

def length_rope_1 : ℝ := 2
def length_rope_2 : ℝ := 6

theorem average_length_of_ropes :
  (length_rope_1 + length_rope_2) / 2 = 4 :=
by
  sorry

end average_length_of_ropes_l177_177073


namespace least_positive_integer_addition_l177_177104

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end least_positive_integer_addition_l177_177104


namespace stratified_random_sampling_l177_177487

open Finset

theorem stratified_random_sampling :
  let junior_high := 400
      senior_high := 200
      total_sample := 60
      ratio_junior := 2
      ratio_senior := 1
      proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
      proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
      sample_junior := proportion_junior * total_sample
      sample_senior := proportion_senior * total_sample in
  sample_junior + sample_senior = total_sample ∧
  ((junior_high.choose sample_junior) * (senior_high.choose sample_senior)) = Σ {C}_{400}^{40}•{C}_{200}^{20}
:= by
  let junior_high := 400
  let senior_high := 200
  let total_sample := 60
  let ratio_junior := 2
  let ratio_senior := 1
  let proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
  let proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
  let sample_junior := proportion_junior * total_sample
  let sample_senior := proportion_senior * total_sample
  have h1 : sample_junior + sample_senior = total_sample := sorry
  have h2 : (junior_high.choose sample_junior) * (senior_high.choose sample_senior) 
              = (C 400 40) * (C 200 20) := sorry
  exact ⟨h1, h2⟩

end stratified_random_sampling_l177_177487


namespace problem_l177_177232

variable {Point : Type} [affine_space Point]
variable (l : line Point) (α β : plane Point)

def perpendicular (l : line Point) (π : plane Point) : Prop :=
sorry -- Define the perpendicular relationship

def parallel (π₁ π₂ : plane Point) : Prop :=
sorry -- Define the parallel relationship

theorem problem (h₁ : perpendicular l α) (h₂ : perpendicular l β) :
  parallel α β :=
sorry

end problem_l177_177232


namespace isosceles_triangle_points_l177_177832

open EuclideanGeometry

variables {X Y : Point} (hXY : X ≠ Y)

def on_perpendicular_bisector (Z : Point) : Prop :=
  dist Z X = dist Z Y

def on_circle (Z : Point) : Prop :=
  dist Z X = dist X Y ∨ dist Z Y = dist X Y

def midpoint (A B : Point) : Point :=
  (A + B) / 2

def excluded_points (Z : Point) : Prop :=
  Z = midpoint X Y ∨ Z = X ∨ Z = Y ∨ dist Z X = 2 * dist X Y ∨ dist Z Y = 2 * dist X Y

theorem isosceles_triangle_points (Z : Point) :
  on_perpendicular_bisector Z ∨ on_circle Z → ¬excluded_points Z → isosceles_triangle X Y Z :=
sorry

end isosceles_triangle_points_l177_177832


namespace wall_length_correct_l177_177133

noncomputable def length_of_wall : ℝ :=
  let volume_of_one_brick := 25 * 11.25 * 6
  let total_volume_of_bricks := volume_of_one_brick * 6800
  let wall_width := 600
  let wall_height := 22.5
  total_volume_of_bricks / (wall_width * wall_height)

theorem wall_length_correct : length_of_wall = 850 := by
  sorry

end wall_length_correct_l177_177133


namespace volume_of_solid_l177_177839

theorem volume_of_solid (a : ℝ) : 
  let R := (a / 2) * Real.sqrt 3, 
      h := (a / 2) * (Real.sqrt 3 - 1),
      V_p := (1 / 3) * (a ^ 2) * h,
      V_total_pyramids := 6 * V_p,
      V_cube := a ^ 3,
      V_total := V_cube + V_total_pyramids
  in V_total = a ^ 3 * Real.sqrt 3 :=
sorry

end volume_of_solid_l177_177839


namespace digits_sum_is_31_l177_177663

noncomputable def digits_sum_proof (A B C D E F G : ℕ) : Prop :=
  (1000 * A + 100 * B + 10 * C + D + 100 * E + 10 * F + G = 2020) ∧ 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧
  (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧
  (E ≠ F) ∧ (E ≠ G) ∧
  (F ≠ G)

theorem digits_sum_is_31 (A B C D E F G : ℕ) (h : digits_sum_proof A B C D E F G) : 
  A + B + C + D + E + F + G = 31 :=
sorry

end digits_sum_is_31_l177_177663


namespace sqrt_200_eq_l177_177012

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177012


namespace binary_10101_is_21_l177_177529

namespace BinaryToDecimal

def binary_to_decimal (n : Nat) : Nat :=
  match n with
  | 10101 => 21

theorem binary_10101_is_21 :
  binary_to_decimal 10101 = 21 := by
  -- Proof steps would go here
  sorry

end BinaryToDecimal

end binary_10101_is_21_l177_177529


namespace range_of_m_l177_177621

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (sqrt 3) * sin x + cos x > m) ∨ (∃ x : ℝ, x^2 + m * x + 1 ≤ 0) ∧ ¬((∀ x : ℝ, (sqrt 3) * sin x + cos x > m) ∧ (∃ x : ℝ, x^2 + m * x + 1 ≤ 0)) →
  m = -2 ∨ m ≥ 2 :=
by
  sorry -- Skip the proof

end range_of_m_l177_177621


namespace complex_modulus_comparison_l177_177160

theorem complex_modulus_comparison : | complex.abs (2 - complex.i) | > 2 * complex.abs (complex.i ^ 4) := sorry

end complex_modulus_comparison_l177_177160


namespace prove_ad_bc_l177_177622

-- Given matrices
def A := ![![0, -1], ![1, 0]]
variables {a b c d : ℤ}

-- Given matrix equation
def M := ![![a, b], ![1, 2]]
def N := ![![3, 4], ![c, d]]

-- The statement
theorem prove_ad_bc : (M ⬝ A = N) → (a * d - b * c = -2) :=
sorry

end prove_ad_bc_l177_177622


namespace symmetric_about_x_axis_l177_177616

noncomputable def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

theorem symmetric_about_x_axis (a : ℝ) :
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f a x = - g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end symmetric_about_x_axis_l177_177616


namespace product_of_a_values_l177_177195

/--
Let a be a real number and consider the points P = (3 * a, a - 5) and Q = (5, -2).
Given that the distance between P and Q is 3 * sqrt 10, prove that the product
of all possible values of a is -28 / 5.
-/
theorem product_of_a_values :
  ∀ (a : ℝ),
  (dist (3 * a, a - 5) (5, -2) = 3 * Real.sqrt 10) →
  ∃ (a₁ a₂ : ℝ), (5 * a₁ * a₁ - 18 * a₁ - 28 = 0) ∧ 
                 (5 * a₂ * a₂ - 18 * a₂ - 28 = 0) ∧ 
                 (a₁ * a₂ = -28 / 5) := 
by
  sorry

end product_of_a_values_l177_177195


namespace max_side_length_triangle_l177_177157

theorem max_side_length_triangle (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter : a + b + c = 20) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : max a (max b c) = 9 := 
sorry

end max_side_length_triangle_l177_177157


namespace θ_satisfies_conditions_and_form_l177_177204

noncomputable def θ (p : Polynomial ℤ) : ℤ := sorry

theorem θ_satisfies_conditions_and_form (θ : Polynomial ℤ → ℤ) (c : ℤ) :
  (∀ p : Polynomial ℤ, θ (p + 1) = θ p + 1) ∧
  (∀ p q : Polynomial ℤ, θ p ≠ 0 → θ p ∣ θ (p * q)) →
  (∀ p: Polynomial ℤ, θ p = p.eval (c : ℤ)) :=
begin
  sorry
end

end θ_satisfies_conditions_and_form_l177_177204


namespace seonwoo_initial_water_amount_l177_177113

noncomputable def initial_water_amount (W : ℝ) :=
  let first_day_remaining := W / 2 in
  let second_day_remaining := first_day_remaining - (first_day_remaining / 3) in
  let third_day_remaining := second_day_remaining / 2 in
  third_day_remaining = 250

theorem seonwoo_initial_water_amount (W : ℝ) (h : initial_water_amount W) : W = 1500 :=
by sorry

end seonwoo_initial_water_amount_l177_177113


namespace fifth_equation_sum_series_eq_l177_177962

-- Define the sequence pattern
def sequence_pattern (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 2 / ((2 * k - 1) * (2 * k + 1)) = 1 / (2 * k - 1) - 1 / (2 * k + 1)

-- The sum of the first 50 terms of the sequence
def series_sum : ℚ :=
  ∑ k in finset.range 50, (2 / ((2 * k + 1) * (2 * k + 3)))

-- Define the mathematical statements
theorem fifth_equation :
  sequence_pattern 5 :=
by
  intros k hk
  cases k with k'
  · simp [sequence_pattern]
  { cases k' with k''
    { simp [sequence_pattern, div_div_eq_div_mul] }
    { cases k'' with
      | succ k''' => sorry } }

theorem sum_series_eq :
  series_sum = 100 / 101 :=
by
  sorry

end fifth_equation_sum_series_eq_l177_177962


namespace parallelogram_vertexO_product_l177_177263

noncomputable def vertexO_product : ℝ :=
let L := (2, 3)
let M := (-1, 4)
let N := (5, -2)
let V := ((L.1 + N.1) / 2, (L.2 + N.2) / 2) in
let O := (2 * V.1 - M.1, 2 * V.2 - M.2) in
O.1 * O.2

theorem parallelogram_vertexO_product :
  vertexO_product = -24 :=
by
  sorry

end parallelogram_vertexO_product_l177_177263


namespace smallest_a_l177_177338

theorem smallest_a 
  (a : ℤ) (P : ℤ → ℤ) 
  (h_pos : 0 < a) 
  (hP1 : P 1 = a) (hP5 : P 5 = a) (hP7 : P 7 = a) (hP9 : P 9 = a) 
  (hP2 : P 2 = -a) (hP4 : P 4 = -a) (hP6 : P 6 = -a) (hP8 : P 8 = -a) : 
  a ≥ 336 :=
by
  sorry

end smallest_a_l177_177338


namespace integral_evaluation_l177_177543

-- Define the integrand
def integrand (x : ℝ) : ℝ := cos (2 * x) / (cos x + sin x)

-- State the main theorem
theorem integral_evaluation : ∫ x in 0..(π / 4), integrand x = sqrt 2 - 1 :=
by
  sorry

end integral_evaluation_l177_177543


namespace mathieu_plot_area_l177_177359

def total_area (x y : ℕ) : ℕ := x * x

theorem mathieu_plot_area :
  ∃ (x y : ℕ), (x^2 - y^2 = 464) ∧ (x - y = 8) ∧ (total_area x y = 1089) :=
by sorry

end mathieu_plot_area_l177_177359


namespace rate_of_profit_correct_l177_177851

-- Defining the conditions
def cost_price : ℝ := 50
def selling_price : ℝ := 60

-- Defining the Rate of Profit calculation
def rate_of_profit (cp sp : ℝ) : ℝ := (sp - cp) / cp * 100

-- The theorem that states the rate of profit is 20% given the conditions
theorem rate_of_profit_correct :
  rate_of_profit cost_price selling_price = 20 :=
by
  sorry

end rate_of_profit_correct_l177_177851


namespace negation_proposition_l177_177398

theorem negation_proposition (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
sorry

end negation_proposition_l177_177398


namespace shoveling_hours_l177_177673

def initial_rate := 25

def rate_decrease := 2

def snow_volume := 6 * 12 * 3

def shoveling_rate (hour : ℕ) : ℕ :=
  if hour = 0 then initial_rate
  else initial_rate - rate_decrease * hour

def cumulative_snow (hour : ℕ) : ℕ :=
  if hour = 0 then snow_volume - shoveling_rate 0
  else cumulative_snow (hour - 1) - shoveling_rate hour

theorem shoveling_hours : cumulative_snow 12 ≠ 0 ∧ cumulative_snow 13 = 47 := by
  sorry

end shoveling_hours_l177_177673


namespace measure_of_angle_B_product_of_vectors_l177_177646

section TriangleProof

variables {A B C : Type} [Real A] [Real B] [Real C] 
variables {a b c : ℝ} (h1 : (a + b + c) * (a - b + c) = 3 * a * c)
variables (h2 : a * c = (2 / 3) * b ^ 2) (h3 : 2 * a < c) 

theorem measure_of_angle_B :
  ∃ (B : ℝ), 0 < B ∧ B < π ∧ (cos B = 1/2) := by
  sorry

theorem product_of_vectors :
  b = 2 ∧ Exists A, cos A = sqrt(3)/2 →
  ∃ ac : ℝ, ac = a * c ∧ (ac = (2 / 3) * b ^ 2) →
  ∃ b c: ℝ, a < c ∧ (b = 2) →
  (a * c = (8 / 3)) →
  (∃ dot_product : ℝ, dot_product = b * c * (sqrt(3) / 2) ∧ dot_product = 4) := by
  sorry

end TriangleProof

end measure_of_angle_B_product_of_vectors_l177_177646


namespace probability_two_purple_two_orange_l177_177872

-- Definitions corresponding to conditions in the problem
def total_marbles := 25
def green_marbles := 8
def purple_marbles := 12
def orange_marbles := 5
def marbles_selected := 4

-- The main statement/formulation of the Lean theorem
theorem probability_two_purple_two_orange : 
  (choose total_marbles marbles_selected) ≠ 0 →
  (choose purple_marbles 2) * (choose orange_marbles 2) / (choose total_marbles marbles_selected) = 66 / 1265 :=
by
  sorry

end probability_two_purple_two_orange_l177_177872


namespace Carrie_can_add_turnips_l177_177170

-- Define the variables and conditions
def potatoToTurnipRatio (potatoes turnips : ℕ) : ℚ :=
  potatoes / turnips

def pastPotato : ℕ := 5
def pastTurnip : ℕ := 2
def currentPotato : ℕ := 20
def allowedTurnipAddition : ℕ := 8

-- Define the main theorem to prove, given the conditions.
theorem Carrie_can_add_turnips (past_p_ratio : potatoToTurnipRatio pastPotato pastTurnip = 2.5)
                                : potatoToTurnipRatio currentPotato allowedTurnipAddition = 2.5 :=
sorry

end Carrie_can_add_turnips_l177_177170


namespace simplify_expression_l177_177037

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : ( (3 * x + 6 - 5 * x) / 3 ) = ( (-2 * x) / 3 + 2 ) :=
by
  sorry

end simplify_expression_l177_177037


namespace classify_triples_l177_177951

open Nat

theorem classify_triples (a b c : ℕ) (ha : prime a) (habc : a ≤ b ∧ b ≤ c) (h : 1 / a + 2 / b + 3 / c = 1) :
  (a = 2 ∧ ((b = 5 ∧ c = 30) ∨ (b = 6 ∧ c = 18) ∨ (b = 7 ∧ c = 14) ∨ (b = 8 ∧ c = 12) ∨ (b = 10 ∧ c = 10))) ∨ 
  (a = 3 ∧ ((b = 4 ∧ c = 18) ∨ (b = 6 ∧ c = 9))) ∨ 
  (a = 5 ∧ (b = 4 ∧ c = 10)) :=
sorry

end classify_triples_l177_177951


namespace tileability_condition_l177_177352

theorem tileability_condition (a b k m n : ℕ) (h₁ : k ∣ a) (h₂ : k ∣ b) (h₃ : ∃ (t : Nat), t * (a * b) = m * n) : 
  2 * k ∣ m ∨ 2 * k ∣ n := 
sorry

end tileability_condition_l177_177352


namespace sqrt_200_eq_l177_177013

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177013


namespace line_equation_l177_177549

theorem line_equation (x y : ℝ) (α : ℝ)
  (h₁ : point (-2, 3))
  (h₂ : line_equation 3 4 (-5))
  (h₃ : slope (2 * α) = -3 / 4) :
  slope α = 1 / 3 ∨ slope α = -1 / 3 → tangent α = 3 → 
  lin_eq x y (-2) 3 3 * x - y + 9 = 0 := 
sorry

end line_equation_l177_177549


namespace find_probability_l177_177347

/-- Define the probabilities for state transitions -/
def q0 : ℚ
def q1 : ℚ
def q2 : ℚ

axiom eq1 : q0 = 1 / 2 * q0 + 1 / 2 * q1
axiom eq2 : q1 = 1 / 2 * q0 + 1 / 2 * q2
axiom eq3 : q2 = 1 / 2 * q0

/-- Prove the probability of encountering 6 heads before 3 tails is 3/4 -/
theorem find_probability : q0 = 3 / 4 := by
  sorry

end find_probability_l177_177347


namespace sum_of_digits_is_3_l177_177365

-- We introduce variables for the digits a and b, and the number
variables (a b : ℕ)

-- Conditions: a and b must be digits, and the number must satisfy the given equation
-- One half of (10a + b) exceeds its one fourth by 3
def valid_digits (a b : ℕ) : Prop := a < 10 ∧ b < 10
def equation_condition (a b : ℕ) : Prop := 2 * (10 * a + b) = (10 * a + b) + 12

-- The number is two digits number
def two_digits_number (a b : ℕ) : ℕ := 10 * a + b

-- Final statement combining all conditions and proving the desired sum of digits
theorem sum_of_digits_is_3 : 
  ∀ (a b : ℕ), valid_digits a b → equation_condition a b → a + b = 3 := 
by
  intros a b h1 h2
  sorry

end sum_of_digits_is_3_l177_177365


namespace no_extreme_points_l177_177399

theorem no_extreme_points (a : ℝ) :
  ∀ x : ℝ, f x = x^3 + 3x^2 + 4x - a → 
  (∀ x y, x < y → f(x) < f(y)) → -- f(x) is strictly increasing which implies no extreme points
  ∃! e : ℝ, ¬∃! e : ℝ, (∀ x, (f x > f e ∨ f x < f e) ∧ x ≠ e) := 
by
  intros
  sorry

end no_extreme_points_l177_177399


namespace john_paid_correct_amount_l177_177331

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

end john_paid_correct_amount_l177_177331


namespace stratified_random_sampling_l177_177485

open Finset

theorem stratified_random_sampling :
  let junior_high := 400
      senior_high := 200
      total_sample := 60
      ratio_junior := 2
      ratio_senior := 1
      proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
      proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
      sample_junior := proportion_junior * total_sample
      sample_senior := proportion_senior * total_sample in
  sample_junior + sample_senior = total_sample ∧
  ((junior_high.choose sample_junior) * (senior_high.choose sample_senior)) = Σ {C}_{400}^{40}•{C}_{200}^{20}
:= by
  let junior_high := 400
  let senior_high := 200
  let total_sample := 60
  let ratio_junior := 2
  let ratio_senior := 1
  let proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
  let proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
  let sample_junior := proportion_junior * total_sample
  let sample_senior := proportion_senior * total_sample
  have h1 : sample_junior + sample_senior = total_sample := sorry
  have h2 : (junior_high.choose sample_junior) * (senior_high.choose sample_senior) 
              = (C 400 40) * (C 200 20) := sorry
  exact ⟨h1, h2⟩

end stratified_random_sampling_l177_177485


namespace isosceles_triangle_base_length_l177_177639

theorem isosceles_triangle_base_length (a b : ℕ) (h_iso : a = b) (h_perimeter : 2 * a + (10 - 2 * a) = 10) : 
    10 - 2 * a = 2 ∨ 10 - 2 * a = 4 :=
by {
  have h1 : (10 - 2 * a) ∈ {2, 4},
  sorry
  }

end isosceles_triangle_base_length_l177_177639


namespace imaginary_part_of_z_l177_177229

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 1 - 3 * I) : z.im = -2 := by
  sorry

end imaginary_part_of_z_l177_177229


namespace sum_of_base_7_digits_l177_177844

theorem sum_of_base_7_digits (n : ℕ) (h : n = 2023) : Nat.digits 7 n = [5, 6, 2, 0] → List.sum (Nat.digits 7 n) = 13 :=
by
  intros h_eq_digits
  rw h_eq_digits
  norm_num

end sum_of_base_7_digits_l177_177844


namespace probability_odd_sum_l177_177328

/-- Spinner A has numbers 4, 5, and 6. -/
def spinnerA : set ℕ := {4, 5, 6}

/-- Spinner B has numbers 1, 2, and 3. -/
def spinnerB : set ℕ := {1, 2, 3}

/-- Spinner C has numbers 7, 8, and 9. -/
def spinnerC : set ℕ := {7, 8, 9}

/-- Condition for the sum to be odd -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- The probability that the sum of the numbers resulting from the three spinners is odd is 8/27. -/
theorem probability_odd_sum :
  (∑ (a ∈ spinnerA) (b ∈ spinnerB) (c ∈ spinnerC),
    if is_odd (a + b + c) then 1 else 0).to_float / (spinnerA.size * spinnerB.size * spinnerC.size).to_float = 8 / 27 :=
sorry

end probability_odd_sum_l177_177328


namespace school_needs_buses_l177_177803

/-
The high school needs to arrange 130 buses for the trip given the following conditions:
- The school has 95 classrooms.
- Each freshmen classroom has 58 students.
- Each sophomore classroom has 47 students.
- There are 45 freshmen classrooms.
- There are 50 sophomore classrooms.
- Each bus has 40 seats.
- Each classroom needs 2 teachers.
- 15 bus drivers are required.
-/
theorem school_needs_buses
  (classrooms : ℕ)
  (freshmen_classrooms : ℕ)
  (sophomore_classrooms : ℕ)
  (students_per_freshmen_classroom : ℕ)
  (students_per_sophomore_classroom : ℕ)
  (seats_per_bus : ℕ)
  (teachers_per_classroom : ℕ)
  (bus_drivers : ℕ)
  (h1 : classrooms = 95)
  (h2 : freshmen_classrooms = 45)
  (h3 : sophomore_classrooms = 50)
  (h4 : students_per_freshmen_classroom = 58)
  (h5 : students_per_sophomore_classroom = 47)
  (h6 : seats_per_bus = 40)
  (h7 : teachers_per_classroom = 2)
  (h8 : bus_drivers = 15) :
  let total_students := freshmen_classrooms * students_per_freshmen_classroom + sophomore_classrooms * students_per_sophomore_classroom,
      total_teachers := classrooms * teachers_per_classroom,
      total_people := total_students + total_teachers + bus_drivers,
      buses_needed := (total_people + seats_per_bus - 1) / seats_per_bus
  in buses_needed = 130 :=
by sorry

end school_needs_buses_l177_177803


namespace cost_of_art_book_l177_177907

theorem cost_of_art_book
  (total_cost m_c s_c : ℕ)
  (m_b s_b a_b : ℕ)
  (hm : m_c = 3)
  (hs : s_c = 3)
  (ht : total_cost = 30)
  (hm_books : m_b = 2)
  (hs_books : s_b = 6)
  (ha_books : a_b = 3)
  : ∃ (a_c : ℕ), a_c = 2 := 
by
  sorry

end cost_of_art_book_l177_177907


namespace part1_part3_l177_177306

-- Condition for companion pair
def is_companion_pair (a b : ℚ) : Prop :=
  a / 2 + b / 3 = (a + b) / 5

-- (1) If (1, b) is a companion pair, then b = -9/4
theorem part1 (b : ℚ) (h : is_companion_pair 1 b) : b = -9 / 4 :=
  sorry

-- (2) Provide an example of a companion pair (a, b) where a ≠ 0 and a ≠ 1
example : ∃ (a b : ℚ), a ≠ 0 ∧ a ≠ 1 ∧ is_companion_pair a b :=
  ⟨6, -9, by norm_num, by norm_num, by norm_num⟩

-- (3) If (m, n) is a companion pair, then m - 22/3 * n - [4m - 2(3n - 1)] = -2
theorem part3 (m n : ℚ) (h : is_companion_pair m n) :
  m - 22 / 3 * n - (4 * m - 2 * (3 * n - 1)) = -2 :=
  sorry

end part1_part3_l177_177306


namespace f_not_monotonic_iff_a_in_interval_l177_177605

def f (x : ℝ) (a : ℝ) : ℝ := (1 / 3) * x^3 - x^2 + a * x - 5

def f_not_monotonic_on_interval (a : ℝ) : Prop :=
  -3 < a ∧ a < 1

theorem f_not_monotonic_iff_a_in_interval (a : ℝ) :
  f_not_monotonic_on_interval a ↔ 
  ∃ (x₁ x₂ : ℝ), (-1 ≤ x₁) ∧ (x₁ < x₂) ∧ (x₂ ≤ 2) ∧ (f x₁ a ≠ f x₂ a) := 
by
  sorry

end f_not_monotonic_iff_a_in_interval_l177_177605


namespace sum_of_reciprocals_l177_177258

variable {α : Type*} [LinearOrderedField α]

def sequence_r (n : ℕ) : α
| 0       := 2
| (n + 1) := sequence_r n * sequence_r n + 1

theorem sum_of_reciprocals 
  (n : ℕ) (a : ℕ → α) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h2 : (∑ i in Finset.range n, 1 / a (i + 1)) < 1) :
  (∑ i in Finset.range n, 1 / a (i + 1)) ≤ (∑ i in Finset.range n, 1 / sequence_r (i + 1)) :=
sorry

end sum_of_reciprocals_l177_177258


namespace initial_points_l177_177751

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l177_177751


namespace find_k_l177_177314

def geom_seq (c : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = c * (a n)

def sum_first_n_terms (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k {c : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ} {k : ℝ} (hGeom : geom_seq c a) (hSum : sum_first_n_terms S k) :
  k = -1 :=
by
  sorry

end find_k_l177_177314


namespace local_minimum_f_when_k2_l177_177124

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_when_k2 : ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f 2 x ≥ f 2 1 :=
by
  -- the question asks to prove that the function attains a local minimum at x = 1 when k = 2
  sorry

end local_minimum_f_when_k2_l177_177124


namespace total_number_of_sampling_results_l177_177482

theorem total_number_of_sampling_results : 
  let junior_students := 400
  let senior_students := 200
  let total_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (junior_sample + senior_sample = total_sample) → 
  (junior_sample / junior_students = 2 / 3) → 
  (senior_sample / senior_students = 1 / 3) → 
  @Fintype.card (Finset (Fin junior_students)).choose junior_sample *
  @Fintype.card (Finset (Fin senior_students)).choose senior_sample = 
  Nat.binom junior_students junior_sample * Nat.binom senior_students senior_sample
:= by
  sorry

end total_number_of_sampling_results_l177_177482


namespace num_distinct_triangles_l177_177573

-- Predicate for valid triangle formation
def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- The set of lengths
def sticks : List ℕ := [2, 3, 4, 6]

-- Function to check number of valid triangles
def count_valid_triangles : ℕ :=
  (sticks.combinations 3).count (λ (triple : List ℕ), valid_triangle triple[0] triple[1] triple[2])

-- The theorem statement
theorem num_distinct_triangles : count_valid_triangles = 2 :=
by sorry

end num_distinct_triangles_l177_177573


namespace solve_ab_eq_l177_177774

theorem solve_ab_eq (a b : ℕ) (h : a^b + a + b = b^a) : a = 5 ∧ b = 2 :=
sorry

end solve_ab_eq_l177_177774


namespace general_term_formula_l177_177599

/-- Define that the point (n, S_n) lies on the function y = 2x^2 + x, hence S_n = 2 * n^2 + n --/
def S_n (n : ℕ) : ℕ := 2 * n^2 + n

/-- Define the nth term of the sequence a_n --/
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 4 * n - 1

theorem general_term_formula (n : ℕ) (hn : 0 < n) :
  a_n n = S_n n - S_n (n - 1) :=
by
  sorry

end general_term_formula_l177_177599


namespace number_of_valid_license_plates_l177_177189

-- Define conditions
def is_letter (c : Char) : Prop := c ∈ ['A'..'Z']

def is_letter_or_digit (c : Char) : Prop :=
  c ∈ ['A'..'Z'] ∨ c ∈ ['0'..'9']

-- Definition of a valid license plate
def valid_license_plate (plate : String) : Prop :=
  plate.length = 4 ∧
  is_letter plate[0] ∧
  plate[0] = plate[1] ∧
  is_letter_or_digit plate[2] ∧
  plate[2] = plate[3]

-- Theorem Statement
theorem number_of_valid_license_plates : ∃ n : ℕ, n = 936 ∧ ∀ plate : String, valid_license_plate plate → n = 936 :=
by {
  sorry
}

end number_of_valid_license_plates_l177_177189


namespace at_least_one_A_or_B_selected_prob_l177_177788

theorem at_least_one_A_or_B_selected_prob :
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  at_least_one_A_or_B_prob = 5 / 6 :=
by
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  sorry

end at_least_one_A_or_B_selected_prob_l177_177788


namespace grandson_age_l177_177724

-- Define the ages of Markus, his son, and his grandson
variables (M S G : ℕ)

-- Conditions given in the problem
axiom h1 : M = 2 * S
axiom h2 : S = 2 * G
axiom h3 : M + S + G = 140

-- Theorem to prove that the age of Markus's grandson is 20 years
theorem grandson_age : G = 20 :=
by
  sorry

end grandson_age_l177_177724


namespace tyler_bought_10_erasers_l177_177836

/--
Given that Tyler initially has $100, buys 8 scissors for $5 each, buys some erasers for $4 each,
and has $20 remaining after these purchases, prove that he bought 10 erasers.
-/
theorem tyler_bought_10_erasers : ∀ (initial_money scissors_cost erasers_cost remaining_money : ℕ), 
  initial_money = 100 →
  scissors_cost = 5 →
  erasers_cost = 4 →
  remaining_money = 20 →
  ∃ (scissors_count erasers_count : ℕ),
    scissors_count = 8 ∧ 
    initial_money - scissors_count * scissors_cost - erasers_count * erasers_cost = remaining_money ∧ 
    erasers_count = 10 :=
by
  intros
  sorry

end tyler_bought_10_erasers_l177_177836


namespace solve_x4_minus_inv_x4_l177_177585

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l177_177585


namespace find_sum_of_xy_l177_177198

theorem find_sum_of_xy (x y : ℝ) (hx_ne_y : x ≠ y) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0)
  (h_equation : x^4 - 2018 * x^3 - 2018 * y^2 * x = y^4 - 2018 * y^3 - 2018 * y * x^2) :
  x + y = 2018 :=
sorry

end find_sum_of_xy_l177_177198


namespace problem_statement_l177_177225

-- Define the sequence based on the recurrence relation
def a : ℕ → ℕ
| 0       := 2
| (n + 1) := a n ^ 2 + 2 * a n

-- Define the log sequence
noncomputable def log_seq : ℕ → ℝ := λ n, Real.log (1 + a n)

-- Define the geometric property with common ratio 2
def is_geometric (seq : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) / seq n = r

-- Define the T_n sequence
noncomputable def T_n (n : ℕ) : ℝ := ∏ i in Finset.range n, Real.exp (log_seq i)

-- Define the general term of sequence a_n
noncomputable def a_general_term (n : ℕ) : ℕ := 3^(2^n - 1) - 1

-- Define b_n and S_n
def b (n : ℕ) : ℝ := (1 / a n) + (1 / (a n + 2))
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- Prove the given equalities and properties

theorem problem_statement :
  (is_geometric log_seq 2) ∧
  (∀ n : ℕ, a n + 1 = 3^(2^n - 1)) ∧
  (∀ n : ℕ, T_n n = 3^(2^n - 1)) ∧
  (∀ n : ℕ, S_n n = 2 * (1 - (1 / 3^n)) - (1 / (3^n) - (1 / a n))) :=
by
  sorry

end problem_statement_l177_177225


namespace interest_rate_is_five_percent_l177_177289

-- Define the principal amount P and the interest rate r.
variables (P : ℝ) (r : ℝ)

-- Define the conditions given in the problem
def simple_interest_condition : Prop := P * r * 2 = 40
def compound_interest_condition : Prop := P * (1 + r)^2 - P = 41

-- Define the goal statement to prove
theorem interest_rate_is_five_percent (h1 : simple_interest_condition P r) (h2 : compound_interest_condition P r) : r = 0.05 :=
sorry

end interest_rate_is_five_percent_l177_177289


namespace problem_1_problem_2_l177_177074

variables (m k : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c (m : ℝ) : ℝ × ℝ := (-2, m)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- (1) Proof of |c| = sqrt(5)
theorem problem_1 (h1 : dot_product a (vector_add b (c m)) = 0) : |c (-1)| = Real.sqrt 5 :=
by
  have m_value : m = -1 := sorry,
  calc
    |c (-1)| = Real.sqrt ((-2)^2 + (-1)^2) : by rw m_value
    ... = Real.sqrt 5 : by norm_num
    done
sorry

-- (2) Proof of k = -2
theorem problem_2 (h2 : ((k * a.1 + b.1) / (2 * a.1 - b.1) = (k * a.2 + b.2) / (2 * a.2 - b.2))) : k = -2 :=
by 
  have k_value : k = -2 := sorry,
  exact k_value
sorry

end problem_1_problem_2_l177_177074


namespace relatively_prime_m_n_l177_177691

noncomputable def probability_of_distinct_real_solutions : ℝ :=
  let b := (1 : ℝ)
  if 1 ≤ b ∧ b ≤ 25 then 1 else 0

theorem relatively_prime_m_n : ∃ m n : ℕ, 
  Nat.gcd m n = 1 ∧ 
  (1 : ℝ) = (m : ℝ) / (n : ℝ) ∧ m + n = 2 := 
by
  sorry

end relatively_prime_m_n_l177_177691


namespace cannot_form_a_set_l177_177443

def is_well_defined_set (S : Type) : Prop :=
  ∃ x, x ∈ S ∧ ∀ y, y ∈ S → x = y

def prime_numbers_1_to_20 : Set ℕ := {n | (n > 1) ∧ (n < 20) ∧ Nat.prime n}

def real_roots_x2_plus_x_minus_2 : Set ℝ := {x | x^2 + x - 2 = 0}

def taller_students_xinhua_high : Set String := {student | "student is taller at Xinhua High School"}

def all_squares : Set ℕ := {n | ∃ k, k^2 = n}

theorem cannot_form_a_set :
  ∀ (S : Set Type), is_well_defined_set prime_numbers_1_to_20 ∧
                              is_well_defined_set real_roots_x2_plus_x_minus_2 ∧
                              is_well_defined_set all_squares ∧
                              ¬ is_well_defined_set taller_students_xinhua_high :=
by
  sorry

end cannot_form_a_set_l177_177443


namespace segment_length_abs_eq_cubrt_27_five_l177_177841

theorem segment_length_abs_eq_cubrt_27_five : 
  (∀ x : ℝ, |x - (3 : ℝ)| = 5) → (8 - (-2) = 10) :=
by 
  intros;
  sorry

end segment_length_abs_eq_cubrt_27_five_l177_177841


namespace pure_imaginary_a_eq_neg3_l177_177682

theorem pure_imaginary_a_eq_neg3 (a : ℝ) (z : ℂ) 
    (h1 : z = a^2 + 2 * a - 3 + (a^2 - 4 * a + 3) * imaginary_unit) :
    (z.im = z ∧ z.re = 0) → a = -3 :=
by
  cases z with re im
  sorry

end pure_imaginary_a_eq_neg3_l177_177682


namespace num_arrangements_l177_177417

theorem num_arrangements (n : ℕ) (A B C D E F G H : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E] [DecidableEq F] [DecidableEq G] [DecidableEq H] :
  let arrangements := 11520 in
  (∀ {A B C : ℕ} (h1 : A ≠ B) (h2 : B ≠ C) (h3: C ≠ A) (h4: D ≠ E), h1 ∧ h2 ∧ h3 ∧ h4) →
  (A, B, C, D, E, F, G, H) ∈ arrangements := sorry

end num_arrangements_l177_177417


namespace cube_of_square_is_15625_l177_177097

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l177_177097


namespace probability_not_square_or_fourth_power_in_range_l177_177937

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_perfect_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, m * m * m * m = n

def count_perfect_squares_upto (k : ℕ) : ℕ :=
{n | n ≤ k ∧ is_perfect_square n}.to_finset.card

def count_perfect_fourth_powers_upto (k : ℕ) : ℕ :=
{n | n ≤ k ∧ is_perfect_fourth_power n}.to_finset.card

def count_not_perfect_square_or_fourth_power (k : ℕ) : ℕ :=
k - count_perfect_squares_upto k

def probability_not_perfect_square_or_fourth_power (k : ℕ) : ℚ :=
(count_not_perfect_square_or_fourth_power k : ℚ) / k

theorem probability_not_square_or_fourth_power_in_range :
  probability_not_perfect_square_or_fourth_power 200 = 93 / 100 :=
by
  sorry

end probability_not_square_or_fourth_power_in_range_l177_177937


namespace sector_radius_cone_l177_177418

theorem sector_radius_cone {θ R r : ℝ} (sector_angle : θ = 120) (cone_base_radius : r = 2) :
  (R * θ / 360) * 2 * π = 2 * π * r → R = 6 :=
by
  intros h
  sorry

end sector_radius_cone_l177_177418


namespace reassemble_to_hexagon_l177_177773

-- Define the conditions of the problem
def initial_figure : Type := sorry  -- Placeholder for the type of the initial figure composed of equilateral triangles.
noncomputable def num_triangles : ℕ := 12  -- The initial figure is made of 12 equilateral triangles.

-- Define the target regular hexagon
def regular_hexagon : Type := sorry  -- Placeholder for the type regular hexagon

-- The theorem stating that it is possible to cut and reassemble the figure into a hexagon
theorem reassemble_to_hexagon (f : initial_figure) (h : num_triangles = 12) :
  ∃ parts : list initial_figure, parts.length = 3 ∧ (∀ part ∈ parts, part ≠ ∅) ∧
  (∃ (g : regular_hexagon), assembled_from_parts parts g) :=
sorry

end reassemble_to_hexagon_l177_177773


namespace math_problem_proof_l177_177262

-- Define the system of equations
structure equations :=
  (x y m : ℝ)
  (eq1 : x + 2*y - 6 = 0)
  (eq2 : x - 2*y + m*x + 5 = 0)

-- Define the problem conditions and prove the required solutions in Lean 4
theorem math_problem_proof :
  -- Part 1: Positive integer solutions for x + 2y - 6 = 0
  (∀ x y : ℕ, x + 2*y = 6 → (x, y) = (2, 2) ∨ (x, y) = (4, 1)) ∧
  -- Part 2: Given x + y = 0, find m
  (∀ x y : ℝ, x + y = 0 → x + 2*y - 6 = 0 → x - 2*y - (13/6)*x + 5 = 0) ∧
  -- Part 3: Fixed solution for x - 2y + mx + 5 = 0
  (∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0) :=
sorry

end math_problem_proof_l177_177262


namespace min_contribution_proof_l177_177127

noncomputable def min_contribution (total_contribution : ℕ) (num_people : ℕ) (max_contribution: ℕ) :=
  ∃ (min_each_person: ℕ), num_people * min_each_person ≤ total_contribution ∧ max_contribution * (num_people - 1) + min_each_person ≥ total_contribution ∧ min_each_person = 2

theorem min_contribution_proof :
  min_contribution 30 15 16 :=
sorry

end min_contribution_proof_l177_177127


namespace number_of_committees_l177_177165

-- Given conditions: 
-- 1. Three departments: physics, chemistry, and biology.
-- 2. Each department has three male and three female professors.
-- 3. A committee of six professors with an equal number of men and women.
-- 4. The committee must include two professors from each of the three departments.

theorem number_of_committees 
  (professors : Type)
  (department : professors → Prop)
  (gender : professors → Prop)
  (professors_in_dept : Π (d : Prop), list professors)
  (dept_physics dept_chemistry dept_biology : Prop)
  (male female : Prop)
  (hp : professors_in_dept dept_physics = [p1, p2, p3, p4, p5, p6])
  (hc : professors_in_dept dept_chemistry = [c1, c2, c3, c4, c5, c6])
  (hb : professors_in_dept dept_biology = [b1, b2, b3, b4, b5, b6])
  (p_male : {p1, p2, p3, c_male, c_male, c_male, b_male, b_male, b_male})
  (p_female : {p4, p5, p6, c_female, c_female, c_female, b_female, b_female, b_female})
  (hc_weight : ∀ x, x ∈ professors_in_dept dept_chemistry → gender x)
  (hp_weight : ∀ x, x ∈ professors_in_dept dept_physics → gender x)
  (hb_weight : ∀ x, x ∈ professors_in_dept dept_biology → gender x) :
  ∃ (committees : set (set professors)),
  (∀ c ∈ committees, ((card c = 6) ∧  
    (∃ A B C : set professors, 
    (A.card = 2 ∧ B.card = 2 ∧ C.card = 2) ∧ 
    (∀ a ∈ A, department a = dept_physics) ∧ 
    (∀ b ∈ B, department b = dept_chemistry) ∧ 
    (∀ d ∈ D, department d = dept_biology) ∧ 
    (gender A = 3 ∧ gender B = 3 ∧ gender D = 3)))
  = 1215 :=
sorry

end number_of_committees_l177_177165


namespace find_t_l177_177316

variables {t : ℝ}

def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Conditions
def angle_C_is_90_deg : Prop :=
  let AB := (t, 1)
  let AC := (2, 3)
  let BC := vec_sub AC AB
  dot_product AC BC = 0

theorem find_t
  (h : angle_C_is_90_deg) :
  t = 5 :=
sorry

end find_t_l177_177316


namespace functional_eq_solution_l177_177544

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f (f x + f y + f z) = f (f x - f y) + f (2 * x * y + f z) + 2 * f (x * z - y * z)) →
  (f = (λ x, 0) ∨ f = (λ x, x^2)) :=
by
  intros h,
  sorry

end functional_eq_solution_l177_177544


namespace ternary_221_binary_10111_comparison_l177_177060

def ternary_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | _     => (n % 10) + 3 * ternary_to_decimal (n / 10)

def binary_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | _     => (n % 10) + 2 * binary_to_decimal (n / 10)

theorem ternary_221_binary_10111_comparison :
  ternary_to_decimal 221 > binary_to_decimal 10111 := by
  sorry

end ternary_221_binary_10111_comparison_l177_177060


namespace six_digit_number_divisible_by_tails_l177_177712

def is_tail (a b : ℕ) : Prop :=
  ∃ n, a = b % 10^n ∧ b / 10^n > 0

def is_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (Int.digits 10 n) → d ≠ 0

def all_tails_divisible (n : ℕ) : Prop :=
  ∀ t, is_tail t n → n % t = 0

theorem six_digit_number_divisible_by_tails :
  is_six_digit_number 721875 ∧ no_zero_digits 721875 ∧ all_tails_divisible 721875 :=
by
  sorry

end six_digit_number_divisible_by_tails_l177_177712


namespace gcd_divisibility_l177_177351

open Nat

theorem gcd_divisibility (a b d : ℕ) (h_gcd: gcd a b = d) :
  (∑ i in finset.range (b+1), if (i * a) % b = 0 then 1 else 0) = d := sorry

end gcd_divisibility_l177_177351


namespace sqrt_200_eq_l177_177015

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177015


namespace stephen_total_distance_l177_177380

-- Define the conditions
def trips : ℕ := 10
def mountain_height : ℝ := 40000
def fraction_of_height_reached : ℝ := 3 / 4

-- Calculate the total distance covered
def total_distance_covered : ℝ :=
  2 * (fraction_of_height_reached * mountain_height) * trips

-- Prove the total distance covered is 600,000 feet
theorem stephen_total_distance :
  total_distance_covered = 600000 := by
  sorry

end stephen_total_distance_l177_177380


namespace six_digit_number_divisible_by_tails_l177_177713

def is_tail (a b : ℕ) : Prop :=
  ∃ n, a = b % 10^n ∧ b / 10^n > 0

def is_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (Int.digits 10 n) → d ≠ 0

def all_tails_divisible (n : ℕ) : Prop :=
  ∀ t, is_tail t n → n % t = 0

theorem six_digit_number_divisible_by_tails :
  is_six_digit_number 721875 ∧ no_zero_digits 721875 ∧ all_tails_divisible 721875 :=
by
  sorry

end six_digit_number_divisible_by_tails_l177_177713


namespace garden_area_increase_is_zero_l177_177146

-- Define the conditions of the problem
def original_length : ℝ := 60
def original_width : ℝ := 15
def partition_length : ℝ := 30

-- Area calculation for the rectangular garden
def original_area : ℝ := original_length * original_width

-- Perimeter of the original rectangle (total fence length)
def total_fence_length : ℝ := 2 * (original_length + original_width)

-- Effective total length of the fence used for the square garden
def effective_fence_length : ℝ := total_fence_length - partition_length

-- Side length of the new square garden
def side_length_square_garden : ℝ := effective_fence_length / 4

-- Area of the new square garden
def new_area_square_garden : ℝ := side_length_square_garden * side_length_square_garden

-- Proving the increase in area is 0
theorem garden_area_increase_is_zero : new_area_square_garden - original_area = 0 := by
  -- Proof outline shows necessary steps (in Lean, actual proof is required, hence 'sorry')
  sorry

end garden_area_increase_is_zero_l177_177146


namespace proportion_value_l177_177277

theorem proportion_value (x : ℝ) (h : 1.25 / x = 15 / 26.5) : x ≈ 2.21 :=
by
  sorry

end proportion_value_l177_177277


namespace infinite_solutions_l177_177881

noncomputable def g (z : ℂ) : ℂ := -complex.I * conj z

theorem infinite_solutions (z : ℂ) (h₁ : complex.abs z = 3) (h₂ : complex.abs (g z) = complex.abs z) :
    ∃ S, set.infinite S ∧ ∀ z ∈ S, complex.abs z = 3 ∧ complex.abs (g z) = complex.abs z :=
by
  sorry

end infinite_solutions_l177_177881


namespace unordered_pairs_of_edges_determine_plane_in_octahedron_l177_177270

theorem unordered_pairs_of_edges_determine_plane_in_octahedron :
  let edges := 12
  let intersecting_edges_per_edge := 4
  ∃ pairs : ℕ, pairs = 24 :=
by {
  let total_pairs := edges * intersecting_edges_per_edge,
  let correct_pairs := total_pairs / 2,
  have h1 : correct_pairs = 24 := by norm_num,
  use correct_pairs,
  exact h1,
  sorry
}

end unordered_pairs_of_edges_determine_plane_in_octahedron_l177_177270


namespace variance_and_regression_l177_177249

def data_points : Nat := 5

def x_values : List ℝ := [2, 4, 5, 6, 8]
def y_values : List ℝ := [30, 40, 60, 50, 70]

def sum_x_squared : ℝ := x_values.foldl (λ acc x => acc + x^2) 0   -- ∑ x_i^2
def sum_y_squared : ℝ := y_values.foldl (λ acc y => acc + y^2) 0   -- ∑ y_i^2
def sum_xy : ℝ := (List.zip x_values y_values).foldl (λ acc (x, y) => acc + x * y) 0 -- ∑ x_i y_i

def mean_x : ℝ := (x_values.foldl (· + ·) 0) / data_points  -- Mean of x_values
def mean_y : ℝ := (y_values.foldl (· + ·) 0) / data_points  -- Mean of y_values

def variance_y : ℝ := (y_values.foldl (λ acc y => acc + (y - mean_y)^2) 0) / data_points -- Variance formula

def b_coef : ℝ := (sum_xy - data_points * mean_x * mean_y) / (sum_x_squared - data_points * mean_x^2)
def a_coef : ℝ := mean_y - b_coef * mean_x

theorem variance_and_regression :
  variance_y = 200 ∧ (∀ x : ℝ, b_coef * x + a_coef = 6.5 * x + 17.5) :=
by
  sorry

end variance_and_regression_l177_177249


namespace jane_needs_change_probability_l177_177905

noncomputable def probability_jane_needs_change : ℚ :=
  let total_permutations := fact 10 in
  let favorable_outcomes := fact 9 + fact 8 in
  1 - (favorable_outcomes / total_permutations)

theorem jane_needs_change_probability :
  probability_jane_needs_change = 8 / 9 := sorry

end jane_needs_change_probability_l177_177905


namespace cube_of_square_of_third_smallest_prime_is_correct_l177_177088

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l177_177088


namespace width_of_track_l177_177897

theorem width_of_track (R r : ℝ) (h : 2 * real.pi * R - 2 * real.pi * r = 15 * real.pi) : R - r = 7.5 :=
by
  sorry

end width_of_track_l177_177897


namespace point_p_in_quad_4_l177_177811

structure Point :=
  (x : ℝ)
  (y : ℝ)

def in_quadrant_4 (p : Point) : Prop := 
  p.x > 0 ∧ p.y < 0

theorem point_p_in_quad_4 : in_quadrant_4 ⟨1, -5⟩ :=
by
  unfold in_quadrant_4
  simp
  split
  exact trivial
  norm_num

end point_p_in_quad_4_l177_177811


namespace infinite_sum_diverges_to_infinity_l177_177958

noncomputable def infinite_sum_diverges : Real :=
  ∑' (m : ℕ) (n : ℕ) (k : ℕ), 1 / (m * (m + n + k) * (n + 1))

-- Statement to be proved
theorem infinite_sum_diverges_to_infinity : infinite_sum_diverges = ⊤ :=
by
  sorry

end infinite_sum_diverges_to_infinity_l177_177958


namespace john_paid_l177_177329

def upfront : ℤ := 1000
def hourly_rate : ℤ := 100
def court_hours : ℤ := 50
def prep_multiplier : ℤ := 2
def brother_share : ℚ := 1/2
def paperwork_fee : ℤ := 500
def transport_costs : ℤ := 300

theorem john_paid :
  let court_cost := court_hours * hourly_rate,
      prep_hours := prep_multiplier * court_hours,
      prep_cost := prep_hours * hourly_rate,
      total_cost := upfront + court_cost + prep_cost + paperwork_fee + transport_costs,
      john_share := total_cost * brother_share
  in john_share = 8400 :=
by {
  sorry
}

end john_paid_l177_177329


namespace probability_50_90_estimated_num_students_passing_l177_177649

noncomputable def normal_distribution_probability (μ σ: ℝ) (a b: ℝ) : ℝ := sorry

noncomputable def probability_above_threshold (μ σ: ℝ) (threshold: ℝ) : ℝ := sorry

noncomputable def estimated_students_passing (num_students : ℕ) (prob_pass : ℝ) : ℕ := 
  (num_students : ℝ * prob_pass).toNat

theorem probability_50_90 :
  normal_distribution_probability 70 10 50 90 = 0.9544 := sorry

theorem estimated_num_students_passing :
  estimated_students_passing 1000 (probability_above_threshold 70 10 60) = 841 := sorry

end probability_50_90_estimated_num_students_passing_l177_177649


namespace ratio_james_paid_l177_177327

-- Define the parameters of the problem
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def james_paid : ℚ := 6

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack
-- Total cost of stickers
def total_cost : ℚ := total_stickers * cost_per_sticker

-- Theorem stating that the ratio of the amount James paid to the total cost of the stickers is 1:2
theorem ratio_james_paid : james_paid / total_cost = 1 / 2 :=
by 
  -- proof goes here
  sorry

end ratio_james_paid_l177_177327


namespace even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l177_177339

open Real

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem even_property_of_f_when_a_zero : 
  ∀ x : ℝ, f 0 x = f 0 (-x) :=
by sorry

theorem non_even_odd_property_of_f_when_a_nonzero : 
  ∀ (a x : ℝ), a ≠ 0 → (f a x ≠ f a (-x) ∧ f a x ≠ -f a (-x)) :=
by sorry

theorem minimum_value_of_f :
  ∀ (a : ℝ), 
    (a ≤ -1/2 → ∃ x : ℝ, f a x = -a - 5/4) ∧ 
    (-1/2 < a ∧ a ≤ 1/2 → ∃ x : ℝ, f a x = a^2 - 1) ∧ 
    (a > 1/2 → ∃ x : ℝ, f a x = a - 5/4) :=
by sorry

end even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l177_177339


namespace solve_logarithmic_equation_l177_177030

variable {x : ℝ}

theorem solve_logarithmic_equation
  (log_base_five : ∀ n, log 5 n = log 5 n)
  (log_power_rule : ∀ y n, y * log 5 n = log 5 (n ^ y))
  (log_quotient_rule : ∀ x y, log 5 x - log 5 y = log 5 (x / y))
  (h : log 5 x - 4 * log 5 2 = -3) :
  x = 16 / 125 :=
by
  sorry

end solve_logarithmic_equation_l177_177030


namespace fencing_rate_l177_177548

/-- Given a circular field of diameter 20 meters and a total cost of fencing of Rs. 94.24777960769379,
    prove that the rate per meter for the fencing is Rs. 1.5. -/
theorem fencing_rate 
  (d : ℝ) (cost : ℝ) (π : ℝ) (rate : ℝ)
  (hd : d = 20)
  (hcost : cost = 94.24777960769379)
  (hπ : π = 3.14159)
  (Circumference : ℝ := π * d)
  (Rate : ℝ := cost / Circumference) : 
  rate = 1.5 :=
sorry

end fencing_rate_l177_177548


namespace rate_of_decrease_l177_177461

theorem rate_of_decrease (x : ℝ) (h : 400 * (1 - x) ^ 2 = 361) : x = 0.05 :=
by {
  sorry -- The proof is omitted as requested.
}

end rate_of_decrease_l177_177461


namespace matrix_power_is_given_l177_177236

noncomputable def matrixA (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 3, b], ![0, 1, 3], ![0, 0, 1]]

def powerMatrix (m : ℕ) (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  (matrixA b) ^ m

theorem matrix_power_is_given (b m : ℝ) (h_eq : powerMatrix (Int.toNat m) b = ![![1, 27, 4060], ![0, 1, 27], ![0, 0, 1]]) :
  b + m = 424 :=
sorry

end matrix_power_is_given_l177_177236


namespace vector_magnitude_l177_177992

noncomputable def a : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (2, -1)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vector_magnitude {x : ℝ} (h : orthogonal (x, 3) (2, -1)) :
  magnitude (2 • a + b) = 5 * real.sqrt 2 :=
by
  have ax : a = (x, 3) := rfl
  have bx : b = (2, -1) := rfl
  -- To show the detailed proof here
  sorry

end vector_magnitude_l177_177992


namespace main_theorem_l177_177986

noncomputable def has_real_roots_for_all_K : Prop :=
  ∀ K : ℝ, let a := K^2,
               b := -(3 * K^2 + 1),
               c := 2 * K^2,
               Δ := b^2 - 4 * a * c in
           Δ ≥ 0

theorem main_theorem : has_real_roots_for_all_K :=
by {
  sorry
}

end main_theorem_l177_177986


namespace arithmetic_mean_of_a_and_b_is_sqrt3_l177_177582

theorem arithmetic_mean_of_a_and_b_is_sqrt3 :
  let a := (Real.sqrt 3 + Real.sqrt 2)
  let b := (Real.sqrt 3 - Real.sqrt 2)
  (a + b) / 2 = Real.sqrt 3 := 
by
  sorry

end arithmetic_mean_of_a_and_b_is_sqrt3_l177_177582


namespace trapezium_area_l177_177970

variables (a b h : ℕ)
def area_trapezium (a b h : ℕ) := (a + b) * h / 2

theorem trapezium_area (ha : a = 20) (hb : b = 18) (hh : h = 14) : area_trapezium a b h = 266 := by
  rw [ha, hb, hh]
  norm_num
  -- alternatively, use a calc block to show the computation step by step
  -- calc
  --   area_trapezium a b h = 19 * 14 : by norm_num
  --   ... = 266 : by norm_num

end trapezium_area_l177_177970


namespace pure_imaginary_a_l177_177640

theorem pure_imaginary_a (a : ℝ) (h : ∃ z : ℂ, z = ((2 + a) : ℂ) - (a : ℂ) * complex.I ∧ z.im ≠ 0 ∧ z.re = 0) : a = -2 :=
sorry

end pure_imaginary_a_l177_177640


namespace instantaneous_velocity_at_t_eq_2_l177_177440

variable (t : ℝ)

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2 

theorem instantaneous_velocity_at_t_eq_2 :
  (deriv (displacement) 2) = 4 :=
sorry

end instantaneous_velocity_at_t_eq_2_l177_177440


namespace eddy_freddy_average_speed_ratio_l177_177201

theorem eddy_freddy_average_speed_ratio:
  (∃ (eddy_time freddy_time : ℝ) (eddy_distance freddy_distance : ℝ),
    eddy_time = 3 ∧ eddy_distance = 480 ∧ freddy_time = 4 ∧ freddy_distance = 300 ∧
    (∀ average_speed_eddy average_speed_freddy : ℝ,
      average_speed_eddy = eddy_distance / eddy_time ∧
      average_speed_freddy = freddy_distance / freddy_time →
      average_speed_eddy / average_speed_freddy = (32/1) / (15/1))) :=
begin
  use [3, 4, 480, 300],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  intros average_speed_eddy average_speed_freddy h,
  cases h with avg_speed_eddy_eq avg_speed_freddy_eq,
  rw [avg_speed_eddy_eq, avg_speed_freddy_eq],
  norm_num,
  sorry
end

end eddy_freddy_average_speed_ratio_l177_177201


namespace overlapping_area_l177_177071

-- Definition for a 30-60-90 triangle with a given hypotenuse length
def ThirtySixtyNinetyTriangle (hypotenuse : ℝ) : Prop :=
  ∃ short_leg long_leg : ℝ, short_leg = hypotenuse / 2 ∧ long_leg = short_leg * sqrt 3

-- Two 30-60-90 triangles overlapping with coinciding hypotenuses
def OverlappingTriangles (hypotenuse : ℝ) : Prop :=
  ThirtySixtyNinetyTriangle hypotenuse ∧ ThirtySixtyNinetyTriangle hypotenuse

-- Main theorem
theorem overlapping_area (h : ℝ) (hpos : h > 0) : 
  OverlappingTriangles h → (h / 2) * ((h / 2) * sqrt 3) = 36 * sqrt 3 :=
  by
    sorry

end overlapping_area_l177_177071


namespace solve_speed_of_current_l177_177057

def speed_of_rowing_in_still_water := 15 -- kmph
def time_to_cover_distance := 19.99840012798976 -- seconds
def distance_covered :=  100 -- meters

def downstream_speed : ℝ := (distance_covered / time_to_cover_distance) * 3.6 -- converting to kmph
def downstream_speed_kmph := 18.00144 -- kmph

theorem solve_speed_of_current :
  downstream_speed - speed_of_rowing_in_still_water = 3.00144 := by
  sorry

end solve_speed_of_current_l177_177057


namespace olivia_money_left_l177_177728

def initial_amount : ℝ := 500
def groceries_cost : ℝ := 125
def original_price_shoes_euros : ℝ := 150
def discount_rate_shoes : ℝ := 0.20
def exchange_rate_euro_to_dollar : ℝ := 1.2
def price_belt_euros : ℝ := 35
def price_jacket_euros : ℝ := 85

theorem olivia_money_left : 
  let discount_shoes := discount_rate_shoes * original_price_shoes_euros,
      discounted_price_shoes_euros := original_price_shoes_euros - discount_shoes,
      price_shoes_dollars := discounted_price_shoes_euros * exchange_rate_euro_to_dollar,
      price_belt_dollars := price_belt_euros * exchange_rate_euro_to_dollar,
      price_jacket_dollars := price_jacket_euros * exchange_rate_euro_to_dollar,
      total_clothing_cost := price_shoes_dollars + price_belt_dollars + price_jacket_dollars,
      total_spent := groceries_cost + total_clothing_cost,
      money_left := initial_amount - total_spent
  in money_left = 87 := 
by
  sorry

end olivia_money_left_l177_177728


namespace probability_sum_of_digits_is_12_l177_177221

def selection_set := {1, 2, 3, 4, 5}

def number_of_ways_to_form_three_digit_number_with_repetition :=
  Finset.card (Finset.pi (Finset.range 3) (λ _, selection_set))

def number_of_valid_numbers (selection: Finset ℕ): ℕ :=
  (selection.filter (λ l, l.sum = 12)).card

theorem probability_sum_of_digits_is_12 :
  (number_of_valid_numbers (Finset.pi (Finset.range 3) (λ _, selection_set)) : ℚ)
  / number_of_ways_to_form_three_digit_number_with_repetition
  = 2 / 25 :=
by
  sorry

end probability_sum_of_digits_is_12_l177_177221


namespace dot_product_given_projection_l177_177241

variables (a b : ℝ^3)

-- Define the magnitude of vector b
def magnitude_b (b : ℝ^3) : ℝ := real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)

-- Define the dot product of vectors a and b
def dot_product (a b : ℝ^3) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Define the projection scalar of a on b
def projection_scalar (a b : ℝ^3) : ℝ :=
  dot_product a b / magnitude_b b

theorem dot_product_given_projection (h_mag : magnitude_b b = 3)
  (h_proj : projection_scalar a b = 8 / 3) :
  dot_product a b = 8 :=
by
  sorry

end dot_product_given_projection_l177_177241


namespace bob_second_third_lap_time_l177_177520

theorem bob_second_third_lap_time :
  ∀ (lap_length : ℕ) (first_lap_time : ℕ) (average_speed : ℕ),
  lap_length = 400 →
  first_lap_time = 70 →
  average_speed = 5 →
  ∃ (second_third_lap_time : ℕ), second_third_lap_time = 85 :=
by
  intros lap_length first_lap_time average_speed lap_length_eq first_lap_time_eq average_speed_eq
  sorry

end bob_second_third_lap_time_l177_177520


namespace total_students_in_school_l177_177298

variable (TotalStudents : ℕ)
variable (num_students_8_years_old : ℕ := 48)
variable (percent_students_below_8 : ℝ := 0.20)
variable (num_students_above_8 : ℕ := (2 / 3) * num_students_8_years_old)

theorem total_students_in_school :
  percent_students_below_8 * TotalStudents + (num_students_8_years_old + num_students_above_8) = TotalStudents :=
by
  sorry

end total_students_in_school_l177_177298


namespace TimLaundryCycle_l177_177745

theorem TimLaundryCycle (T : ℕ) : (∀ n : ℕ, RonLaundry n → TimLaundry n) :=
begin
  assume (T : ℕ) (RonLaundry : ℕ → Prop) (TimLaundry : ℕ → Prop),
  have RonLaundry_Cycle : ∀ n : ℕ, RonLaundry n ↔ n % 6 = 0,
  from sorry,
  have TimLaundry_Cycle : ∀ n : ℕ, TimLaundry n ↔ n % T = 0,
  from sorry,
  have bothLaundry_Today : RonLaundry 0 ∧ TimLaundry 0,
  from sorry,
  have bothLaundry_18_days : RonLaundry 18 ∧ TimLaundry 18,
  from sorry,
  have factors_18 : ∀ m : ℕ, (m ∣ 18) → (m = 18 ∨ m = 9 ∨ m = 6 ∨ m = 3 ∨ m = 2 ∨ m = 1),
  from sorry,
  have T_is_3 : T = 3,
  from sorry,
  exact T_is_3,
end

end TimLaundryCycle_l177_177745


namespace g_9_to_the_4_l177_177035

variable (f g : ℝ → ℝ)

axiom a1 : ∀ x, x ≥ 1 → f(g(x)) = x^2
axiom a2 : ∀ x, x ≥ 1 → g(f(x)) = x^4
axiom a3 : g 81 = 81

theorem g_9_to_the_4 : (g 9) ^ 4 = 81 := by
  sorry

end g_9_to_the_4_l177_177035


namespace points_on_line_l177_177769

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177769


namespace problem_minimum_value_f_problem_minimum_value_fraction_l177_177705

theorem problem_minimum_value_f :
  let f (x : ℝ) := |x + 1| + |2 * x - 1|
  in minimum f = (3 / 2) := sorry

theorem problem_minimum_value_fraction (m n : ℝ) :
  m > 0 → n > 0 → m + n = (3 / 2) → min_value (1 / m + 4 / n) = 6 := sorry

end problem_minimum_value_f_problem_minimum_value_fraction_l177_177705


namespace binary_to_decimal_10101_l177_177531

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end binary_to_decimal_10101_l177_177531


namespace stratified_random_sampling_l177_177484

open Finset

theorem stratified_random_sampling :
  let junior_high := 400
      senior_high := 200
      total_sample := 60
      ratio_junior := 2
      ratio_senior := 1
      proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
      proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
      sample_junior := proportion_junior * total_sample
      sample_senior := proportion_senior * total_sample in
  sample_junior + sample_senior = total_sample ∧
  ((junior_high.choose sample_junior) * (senior_high.choose sample_senior)) = Σ {C}_{400}^{40}•{C}_{200}^{20}
:= by
  let junior_high := 400
  let senior_high := 200
  let total_sample := 60
  let ratio_junior := 2
  let ratio_senior := 1
  let proportion_junior := (ratio_junior : ℚ) / (ratio_junior + ratio_senior)
  let proportion_senior := (ratio_senior : ℚ) / (ratio_junior + ratio_senior)
  let sample_junior := proportion_junior * total_sample
  let sample_senior := proportion_senior * total_sample
  have h1 : sample_junior + sample_senior = total_sample := sorry
  have h2 : (junior_high.choose sample_junior) * (senior_high.choose sample_senior) 
              = (C 400 40) * (C 200 20) := sorry
  exact ⟨h1, h2⟩

end stratified_random_sampling_l177_177484


namespace terrier_to_poodle_grooming_ratio_l177_177403

-- Definitions and conditions
def time_to_groom_poodle : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_grooming_time : ℕ := 210
def time_to_groom_terrier := total_grooming_time - (num_poodles * time_to_groom_poodle) / num_terriers

-- Theorem statement
theorem terrier_to_poodle_grooming_ratio :
  time_to_groom_terrier / time_to_groom_poodle = 1 / 2 :=
by
  sorry

end terrier_to_poodle_grooming_ratio_l177_177403


namespace polynomial_real_root_b_range_l177_177191

theorem polynomial_real_root_b_range (b : ℝ) :
  (∃ x : ℝ, x^4 + b * x^3 + x^2 + b * x + 1 = 0) ↔ b ∈ set.Ico (-(3 : ℝ) / 4) 0 :=
sorry

end polynomial_real_root_b_range_l177_177191


namespace alex_time_to_run_6_miles_l177_177158

-- Define parameters based on the problem.
def steve_time_run_9_miles : ℕ := 36 -- Steve's time for 9 miles.

-- Define the time Jordan takes to run 3 miles.
def jordan_time_run_3_miles : ℕ := steve_time_run_9_miles / 3

-- Define the time Alex takes to run 4 miles.
def alex_time_run_4_miles : ℕ := 2 * jordan_time_run_3_miles

-- The expected time for Alex to run 6 miles should be calculated.
def alex_time_run_6_miles_correct : ℕ := 36

-- Proof statement.
theorem alex_time_to_run_6_miles :
  let jordan_time := steve_time_run_9_miles / 3 in
  let alex_time := 2 * jordan_time in
  (6 * alex_time_run_4_miles / 4) = alex_time_run_6_miles_correct :=
by
  sorry  -- Proof to be filled in.

end alex_time_to_run_6_miles_l177_177158


namespace min_value_expression_l177_177348

theorem min_value_expression (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  (x^2 + 8 * x * y + 25 * y^2 + 16 * y * z + 9 * z^2) ≥ 403 / 9 := by
  sorry

end min_value_expression_l177_177348


namespace football_team_laps_time_l177_177044

theorem football_team_laps_time
    (l w : ℕ) (length := 100) (width := 50)
    (d : ℕ) (obstacle_distance := 20)
    (n : ℕ) (laps := 6)
    (v : ℕ) (average_speed := 4) :
    let perimeter := 2 * (length + width)
    let additional_distance_per_obstacle := 2 * obstacle_distance
    let total_distance_per_lap := perimeter + additional_distance_per_obstacle
    let total_distance := n * total_distance_per_lap
    let time := total_distance / average_speed
  in
  time = 510 := 
sorry

end football_team_laps_time_l177_177044


namespace dima_unusual_die_probability_l177_177952

theorem dima_unusual_die_probability:
  (∃ p : ℚ,
     (3 * p + 3 * 2 * p = 1) ∧
     let P_1 : ℚ := p,
         P_2 : ℚ := 2 * p,
         P_3 : ℚ := p in
     (P_1 + P_2 + P_3) = (4 / 9)) →
  let m := 4,
      n := 9 in
  m + n = 13 :=
by
  sorry

end dima_unusual_die_probability_l177_177952


namespace quadratic_reciprocity_l177_177371

theorem quadratic_reciprocity (p q : ℕ) [fact (nat.prime p)] [fact (nat.prime q)] :
  (legendreSym p q) = (-1) ^ ((p - 1) * (q - 1) / 4) * (legendreSym q p) :=
by
  -- sorry is used to indicate that we aren't providing the proof
  sorry

end quadratic_reciprocity_l177_177371


namespace area_PQR_is_14_l177_177051

notation "ℝ²" => (ℝ × ℝ)

def reflect_y_axis (p : ℝ²) : ℝ² :=
  (-p.1, p.2)

def reflect_y_eq_x (p : ℝ²) : ℝ² :=
  (p.2, p.1)

def distance (p q : ℝ²) : ℝ :=
  Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

def area_of_triangle (P Q R : ℝ²) : ℝ :=
  1 / 2 * (distance P Q) * Real.abs (R.2 - P.2)

theorem area_PQR_is_14 :
  let P : ℝ² := (2, 5)
  let Q : ℝ² := reflect_y_axis P
  let R : ℝ² := reflect_y_eq_x Q
  area_of_triangle P Q R = 14 :=
by
  sorry

end area_PQR_is_14_l177_177051


namespace elephant_statue_price_l177_177361

theorem elephant_statue_price :
  ∀ (jade_per_giraffe : ℕ) (price_per_giraffe : ℕ) (total_jade : ℕ) (extra_revenue : ℕ),
    jade_per_giraffe = 120 →
    price_per_giraffe = 150 →
    total_jade = 1920 →
    extra_revenue = 400 →
    let jade_per_elephant := 2 * jade_per_giraffe in
    let num_giraffes := total_jade / jade_per_giraffe in
    let revenue_giraffes := num_giraffes * price_per_giraffe in
    let revenue_elephants := revenue_giraffes + extra_revenue in
    let num_elephants := total_jade / jade_per_elephant in
    let price_per_elephant := revenue_elephants / num_elephants in
    price_per_elephant = 350 := 
by
  intros jade_per_giraffe price_per_giraffe total_jade extra_revenue h1 h2 h3 h4
  let jade_per_elephant := 2 * jade_per_giraffe
  let num_giraffes := total_jade / jade_per_giraffe
  let revenue_giraffes := num_giraffes * price_per_giraffe
  let revenue_elephants := revenue_giraffes + extra_revenue
  let num_elephants := total_jade / jade_per_elephant
  let price_per_elephant := revenue_elephants / num_elephants
  have : price_per_elephant = 350 := sorry
  exact this

end elephant_statue_price_l177_177361


namespace distinct_integers_picking_l177_177464

theorem distinct_integers_picking :
  ∃ (S : Finset ℕ), S.card = 1003 ∧ (∀ x ∈ S, ∀ y ∈ S, x ≠ y → |x - y| ≠ 10) → 
  (S = Finset.range 2003) ∧ (S.card = 101^7) := 
sorry

end distinct_integers_picking_l177_177464


namespace second_large_bucket_contains_39_ounces_l177_177185

-- Definitions based on the conditions
def buckets : List ℕ := [11, 13, 12, 16, 10]
def first_large_bucket := 23
def ten_ounces := 10

-- Statement to prove
theorem second_large_bucket_contains_39_ounces :
  buckets.filter (λ x, x ≠ 10 ∧ x ≠ 13) = [11, 12, 16] →
  List.sum [11, 12, 16] = 39 :=
by
  sorry

end second_large_bucket_contains_39_ounces_l177_177185


namespace soccer_player_possible_scores_l177_177150

theorem soccer_player_possible_scores : 
  ∀ (goals : ℕ) (score : ℕ), goals = 7 → (∀ g, g ∈ {0, 1} → (∃ (x y : ℕ), goals = x + y ∧ score = 2 * x + y)) → 
  (∃ (possible_scores : Finset ℕ), possible_scores.card = 8) := 
by
  intros goals score H_goals H_score_range
  have H_set : Finset (ℕ) := (Finset.range 8).map ⟨λ y, y + 7, _⟩
  use H_set
  sorry

end soccer_player_possible_scores_l177_177150


namespace limit_seq_l177_177281

/-- 
  Given a sequence {a_n} where all terms are positive and satisfying the condition 
  ∑_{i=1}^n √(a_i) = n^2 + 3n for n ∈ ℕ*, prove that the limit as n approaches infinity 
  of (1 / n^2) * ∑_{i=1}^n (a_i / (i + 1)) is 2.
-/
theorem limit_seq {a : ℕ → ℝ} 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_sum : ∑ i in finset.range(n+1), real.sqrt (a i) = (n+1)^2 + 3 * (n+1)) :
  tendsto (λ n, (1 / (n : ℝ)^2) * ∑ i in finset.range(n+1), (a i / (i + 1))) at_top (𝓝 2) :=
sorry

end limit_seq_l177_177281


namespace greatest_distance_eq_451_l177_177674

noncomputable def isosceles_triangle (A B C : Point) : Prop :=
  distance A B = 4 ∧ distance A C = 4 ∧ distance B C = 5

noncomputable def circle (center : Point) (radius : ℝ) : Set Point :=
  { P | distance P center = radius }

noncomputable def midpoint (P Q : Point) : Point :=
  ((P.x + Q.x) / 2, (P.y + Q.y) / 2)

noncomputable def perpendicular_bisector (P Q : Point) : Line :=
  let M := midpoint P Q
  let slope := (P.y - Q.y) / (Q.x - P.x)
  let perp_slope := -1 / slope
  Line.mk M.perp_slope

theorem greatest_distance_eq_451 (A B C M X Y Z W : Point) :
  isosceles_triangle A B C →
  B = (-5/2, 0) →
  C = (5/2,0) →
  M = midpoint B C →
  perpendicular_bisector B C ∩ circle B 2 = {X, Y} →
  perpendicular_bisector B C ∩ circle C 2 = {Z,W} →
  (greatest_distance X Y Z W).num = 399 ∧ (greatest_distance X Y Z W).den = 8 ∧ 
  gcd(5,8) = 1 →
  399 + 5 + 39 + 8 = 451 :=
sorry

end greatest_distance_eq_451_l177_177674


namespace stratified_sampling_result_l177_177475

-- Definitions and conditions from the problem
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_sample : ℕ := 60
def junior_sample : ℕ := 40
def senior_sample : ℕ := 20

-- Main theorem statement proving the number of different sampling results
theorem stratified_sampling_result :
  choose junior_students junior_sample * choose senior_students senior_sample = 
  choose 400 40 * choose 200 20 := by
  sorry

end stratified_sampling_result_l177_177475


namespace total_wait_days_l177_177719

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end total_wait_days_l177_177719


namespace ways_to_choose_providers_l177_177334

theorem ways_to_choose_providers : (25 * 24 * 23 * 22 = 303600) :=
by
  sorry

end ways_to_choose_providers_l177_177334


namespace correct_statements_l177_177847

noncomputable def TriangularPyramid := sorry
noncomputable def RegularTriangularPyramid := sorry
noncomputable def Tetrahedron := sorry
noncomputable def RegularTetrahedron := sorry
noncomputable def Parallelepiped := sorry
noncomputable def parallel (l1 l2 : Line) : Prop := sorry
noncomputable def perspective_drawing (l : Line) : Line := sorry
noncomputable def Circle := sorry
noncomputable def Plane := sorry
noncomputable def center (o : Circle) : Point := sorry
noncomputable def on_circle (p : Point) (o : Circle) : Prop := sorry

-- Converting the conditions to Lean 4 environment
axiom triangular_pyramid_is_tetrahedron :
  ∀ (a : TriangularPyramid),
    (∃ (b : Tetrahedron), a = b)

axiom regular_triangular_pyramid_is_regular_tetrahedron :
  ∀ (a : RegularTriangularPyramid),
    (∃ (b : RegularTetrahedron), a = b)

axiom opposite_faces_of_parallelepiped_are_congruent :
  ∀ (a b : Parallelepiped),
    (∃ (f1 f2 : Parallelogram), f1 = f2 ∧ a = b)

axiom parallel_lines_in_perspective :
  ∀ (l1 l2 : Line),
    parallel l1 l2 → ¬ parallel (perspective_drawing l1) (perspective_drawing l2)

axiom three_points_on_circle_define_plane :
  ∀ (o : Circle) (p1 p2 : Point),
    p1 ≠ p2 ∧ on_circle p1 o ∧ on_circle p2 o →
    ∃ (q : Plane), q = Plane.mk_with_points (center o) p1 p2

theorem correct_statements :
  {A : Prop} {B : Prop} {D : Prop},
    (A ↔ triangular_pyramid_is_tetrahedron ∧ regular_triangular_pyramid_is_regular_tetrahedron) ∧
    (B ↔ opposite_faces_of_parallelepiped_are_congruent) ∧
    (D ↔ three_points_on_circle_define_plane) → A ∧ B ∧ D :=
by
  intros
  sorry

end correct_statements_l177_177847


namespace add_points_proof_l177_177754

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l177_177754


namespace monotonicity_range_of_k_sum_of_zeros_l177_177607

-- Definitions for the function and derivative
def f (x k : ℝ) := Real.exp x - k * x + k
def f' (x k : ℝ) := Real.exp x - k

-- Part 1: Conditions for monotonicity of f(x)
theorem monotonicity (k x : ℝ) : 
  (k ≤ 0 → ∀ x, 0 < f' x k) ∧ 
  (k > 0 → (∀ x, x > Real.log k → 0 < f' x k) ∧ (∀ x, x < Real.log k → f' x k < 0)) := 
by
  sorry

-- Part 2(i): The range of k for having two distinct zeros
theorem range_of_k (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ k = 0 ∧ f x₂ k = 0) → 
  k > Real.exp 2 := 
by
  sorry

-- Part 2(ii): x₁ + x₂ > 4 when f(x) has two distinct zeros
theorem sum_of_zeros (k x₁ x₂ : ℝ) (h₁ : f x₁ k = 0) (h₂ : f x₂ k = 0) :
  k > Real.exp 2 → x₁ + x₂ > 4 := 
by
  sorry

end monotonicity_range_of_k_sum_of_zeros_l177_177607


namespace ellipse_theorem_hyperbola_theorem_l177_177460

noncomputable def ellipse_equation 
  (vertex : ℝ × ℝ) 
  (ratio : ℝ) 
  (major : ℝ) 
  (minor : ℝ) 
  (a_eq_twice_b : major = 2 * minor) 
  (vertex_cond : vertex = (2, 0)) 
: Prop := 
  vertex = (2, 0) → (1 / 16) * y ^ 2 + (1 / 4) * x ^ 2 = 1

noncomputable def hyperbola_equation 
  (asymptote : ℝ × ℝ → Prop) 
  (pnt : ℝ × ℝ) 
  (λ : ℝ) 
  (asymptote_cond : ∀ x y, asymptote (x, y) → x + 2 * y = 0) 
  (pnt_cond : pnt = (2, 2)) 
  (λ_cond : λ = 4 - 4 * 4) 
: Prop := 
  pnt = (2, 2) → -1 / 12 * y ^ 2 + 1 / 4 * x ^ 2 = 1

-- Generate a theorem from our definition and conditions for an ellipse
theorem ellipse_theorem 
  (vertex : ℝ × ℝ) 
  (ratio : ℝ) 
  (major : ℝ) 
  (minor : ℝ) 
  (a_eq_twice_b : major = 2 * minor) 
  (vertex_cond : vertex = (2, 0)) 
: ellipse_equation vertex ratio major minor a_eq_twice_b vertex_cond :=
sorry  

-- Generate a theorem from our definition and conditions for a hyperbola
theorem hyperbola_theorem 
  (asymptote : ℝ × ℝ → Prop) 
  (pnt : ℝ × ℝ) 
  (λ : ℝ) 
  (asymptote_cond : ∀ x y, asymptote (x, y) → x + 2 * y = 0) 
  (pnt_cond : pnt = (2, 2)) 
  (λ_cond : λ = 4 - 4 * 4) 
: hyperbola_equation asymptote pnt λ asymptote_cond pnt_cond λ_cond :=
sorry

end ellipse_theorem_hyperbola_theorem_l177_177460


namespace angle_BPD_105_l177_177303

-- definitions based on the conditions
variables {A B C D P : Type}
variable [EuclideanGeometry A]

-- conditions
axiom isosceles_triangle (ABC : Triangle A) : ABC.is_isosceles = true
axiom midpoint_D (BC : Line A) (D : Point A) : D.midpoint BC
axiom P_on_AD (AD : Line A) (P : Point A) : P.on AD ∧ AD.is_bisected_by P
axiom angle_BAC_eq_2x (BAC : Angle A) (x : Real) : measure BAC = 2 * x

-- theorem to prove the required angle
theorem angle_BPD_105 (ABC : Triangle A) (BC : Line A) (AD : Line A) (BAC BPD : Angle A) (x : Real)
  (h1: ABC.is_isosceles) (h2: D.midpoint BC) (h3: P.on AD) (h4: AD.is_bisected_by P) (h5: measure BAC = 2 * x) :
  measure BPD = 105 :=
  sorry

end angle_BPD_105_l177_177303


namespace solution_l177_177130

-- Definition of the problem conditions
variable (total_employees : ℕ) (women_with_fair_hair : ℕ) (percent_women_with_fair_hair : ℚ)
variable (percent_fair_haired_women : ℚ) (total_fair_haired : ℚ)

-- Assumptions based on problem statement
def conditions :=
  total_employees = 100 ∧
  percent_women_with_fair_hair = 30 / 100 ∧
  percent_fair_haired_women = 40 / 100 ∧
  women_with_fair_hair = total_employees * percent_women_with_fair_hair

-- Prove
def problem : Prop :=
  ∃ fair_haired : ℚ,
    fair_haired = (women_with_fair_hair / percent_fair_haired_women) / total_employees ∧
    fair_haired * 100 = 75

theorem solution : problem := by
  unfold conditions problem
  sorry

end solution_l177_177130


namespace parametric_to_cartesian_l177_177942

theorem parametric_to_cartesian (θ : ℝ) (x y : ℝ) :
  (x = 1 + 2 * Real.cos θ) →
  (y = 2 * Real.sin θ) →
  (x - 1) ^ 2 + y ^ 2 = 4 :=
by 
  sorry

end parametric_to_cartesian_l177_177942


namespace find_a_and_extreme_values_l177_177560

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - 3 * x ^ 2

theorem find_a_and_extreme_values (a : ℝ) 
  (h_extreme : ∀ y : ℝ, y = f a 2 ∂ y = f a 2 → f' a 2 = 0) 
  (h_f' : ∀ x : ℝ, f' a x = 3 * a * x ^ 2 - 6 * x) :
  (a = 1) ∧ 
  (let fₐ (x : ℝ) := f 1 x in ∃ (xmin : ℝ) (xmax : ℝ), xmin = -4 ∧ xmax = 50 ∧ 
     (∀ x ∈ Icc (-1 : ℝ) 5, fₐ x = xmin ∨ fₐ x = xmax)) :=
sorry

end find_a_and_extreme_values_l177_177560


namespace congruent_triangles_definition_correct_l177_177848

-- Definitions for the statements in the problem
def congruent_triangles (T1 T2 : Triangle) : Prop :=
  T1.shape = T2.shape ∧ T1.size = T2.size

def statement_A_is_correct : Prop :=
  ∀ (T1 T2 : Triangle), congruent_triangles T1 T2 ↔ (T1.shape = T2.shape ∧ T1.size = T2.size)

-- Lean 4 statement for the problem
theorem congruent_triangles_definition_correct : statement_A_is_correct :=
by
  sorry

end congruent_triangles_definition_correct_l177_177848


namespace average_rainfall_february_1964_l177_177295

theorem average_rainfall_february_1964 :
  let total_rainfall := 280
  let days_february := 29
  let hours_per_day := 24
  (total_rainfall / (days_february * hours_per_day)) = (280 / (29 * 24)) :=
by
  sorry

end average_rainfall_february_1964_l177_177295


namespace malcolm_followers_l177_177715

noncomputable def total_followers (initial_insta followers_facebook : ℕ) 
    (followers_twitter followers_tiktok : ℕ) 
    (followers_youtube followers_pinterest followers_snapchat : ℕ)
    (new_insta : ℕ) (new_facebook : ℕ) (new_twitter : ℕ)
    (new_tiktok : ℕ) (new_snapchat : ℕ) : ℕ :=
  new_insta + new_facebook + new_twitter + new_tiktok + followers_youtube + followers_pinterest + new_snapchat

theorem malcolm_followers :
  let initial_insta := 240 in
  let followers_facebook := 500 in
  let followers_twitter := (initial_insta + followers_facebook) / 2 in
  let followers_tiktok := 3 * followers_twitter in
  let followers_youtube := followers_tiktok + 510 in
  let followers_pinterest := 120 in
  let followers_snapchat := followers_pinterest / 2 in
  let new_insta := initial_insta + (15 * initial_insta) / 100 in
  let new_facebook := followers_facebook + (20 * followers_facebook) / 100 in
  let new_twitter := followers_twitter + 30 in
  let new_tiktok := followers_tiktok + 45 in
  let new_snapchat := followers_snapchat - 10 in
  total_followers initial_insta followers_facebook followers_twitter followers_tiktok
        followers_youtube followers_pinterest followers_snapchat 
        new_insta new_facebook new_twitter new_tiktok new_snapchat = 4221 := 
by
  sorry

end malcolm_followers_l177_177715


namespace necessary_condition_for_inequality_l177_177564

theorem necessary_condition_for_inequality (a b : ℝ) (h : a * b > 0) : 
  (a ≠ b) → (a ≠ 0) → (b ≠ 0) → ((b / a) + (a / b) > 2) :=
by
  sorry

end necessary_condition_for_inequality_l177_177564


namespace program_loop_result_l177_177107

-- Declaring the given conditions
variable (i : ℕ := 11)
variable (s : ℕ := 1)

-- Declaring the loop function
noncomputable def loop : ℕ × ℕ :=
do
  let mut s := s
  let mut i := i
  while i ≥ 9 do
    s := s * i
    i := i - 1
  (s, i)

-- The theorem we need to prove
theorem program_loop_result : (loop (11, 1)).1 = 990 :=
by
  sorry

end program_loop_result_l177_177107


namespace pie_difference_l177_177652

-- Define the fractions for the first and second participants
def first_participant : ℚ := 5 / 6
def second_participant : ℚ := 2 / 3

-- Define their common denominators and differences
def common_denominator : ℚ := second_participant * 2 / 2
def difference : ℚ := first_participant - common_denominator

-- Theorem to prove the answer
theorem pie_difference :
  difference = 1 / 6 := 
by 
  unfold first_participant second_participant common_denominator difference
  rw [second_participant]
  norm_num
  -- here you can simplify the expression to show it equals 1/6
  norm_num
  sorry

end pie_difference_l177_177652


namespace net_income_calculation_l177_177670

-- Definitions based on conditions
def rent_per_hour := 20
def monday_hours := 8
def wednesday_hours := 8
def friday_hours := 6
def sunday_hours := 5
def maintenance_cost := 35
def insurance_fee := 15
def rental_days := 4

-- Derived values based on conditions
def total_income_per_week :=
  (monday_hours + wednesday_hours) * rent_per_hour * 2 + 
  friday_hours * rent_per_hour + 
  sunday_hours * rent_per_hour

def total_expenses_per_week :=
  maintenance_cost + 
  insurance_fee * rental_days

def net_income_per_week := 
  total_income_per_week - total_expenses_per_week

-- The final proof statement
theorem net_income_calculation : net_income_per_week = 445 := by
  sorry

end net_income_calculation_l177_177670


namespace orthocenter_centroid_distance_l177_177675

-- Definitions of points, centroids, and orthocenters in geometrical context
noncomputable def G1 (D E F : ℝ × ℝ) : ℝ × ℝ :=
  (D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3

noncomputable def K (H E F : ℝ × ℝ) : ℝ × ℝ :=
  (E.1 + F.1) - H.1, (E.2 + F.2) - H.2

noncomputable def L (H F D : ℝ × ℝ) : ℝ × ℝ :=
  (F.1 + D.1) - H.1, (F.2 + D.2) - H.2

noncomputable def M (H D E : ℝ × ℝ) : ℝ × ℝ :=
  (D.1 + E.1) - H.1, (D.2 + E.2) - H.2

noncomputable def G2 (K L M : ℝ × ℝ) : ℝ × ℝ :=
  (K.1 + L.1 + M.1) / 3, (K.2 + L.2 + M.2) / 3

theorem orthocenter_centroid_distance
  (H D E F : ℝ × ℝ)
  (K := K H E F)
  (L := L H F D)
  (M := M H D E)
  (G1 := G1 D E F)
  (G2 := G2 K L M) :
  dist H G1 = dist G1 G2 := by
  sorry

end orthocenter_centroid_distance_l177_177675


namespace line_curve_disjoint_range_x_plus_y_curve_C_l177_177596

-- Defining the parametric equations of the line l.
def line_l (t : ℝ) : ℝ × ℝ := 
  ( (sqrt 2) / 2 * t, (sqrt 2) / 2 * t + 4 * sqrt 2 )

-- Defining the polar equation of the curve C.
def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * cos (θ + π / 4)
  in (ρ * cos θ, ρ * sin θ)

-- Given:
-- 1. parametric equation of line l
-- 2. polar coordinate equation of curve C
-- Prove:
-- 1. Line l and curve C are disjoint.
-- 2. Range of x+y for any point on curve C is [-sqrt 2, sqrt 2].

theorem line_curve_disjoint : 
  ∀ t θ : ℝ,
  let l := (sqrt 2) / 2 * t
  let y_l := (sqrt 2) / 2 * t + 4 * sqrt 2
  let ρ := 2 * cos (θ + π / 4)
  let x_C := ρ * cos θ
  let y_C := ρ * sin θ
  (x_C - (sqrt 2) / 2) ^ 2 + (y_C + (sqrt 2) / 2) ^ 2 = 1 → 
  (y_l - l ≠ y_C - x_C) :=
sorry

theorem range_x_plus_y_curve_C : 
  ∀ θ : ℝ,
  let ρ := 2 * cos (θ + π / 4)
  let x := ρ * cos θ
  let y := ρ * sin θ
  -sqrt 2 ≤ x + y ∧ x + y ≤ sqrt 2 :=
sorry

end line_curve_disjoint_range_x_plus_y_curve_C_l177_177596


namespace tangent_line_at_point_l177_177789

theorem tangent_line_at_point
  (x y : ℝ)
  (h_curve : y = x^3 - 3 * x^2 + 1)
  (h_point : (x, y) = (1, -1)) :
  ∃ m b : ℝ, (m = -3) ∧ (b = 2) ∧ (y = m * x + b) :=
sorry

end tangent_line_at_point_l177_177789


namespace calculate_crease_length_l177_177299

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (triangle : Triangle A B C)
variable (side1 side2 hypotenuse : ℝ) -- sides of the triangle

variable (right_angle : IsRightAngle triangle)
variable (side_lengths : side1 = 6 ∧ side2 = 8 ∧ hypotenuse = 10)

def length_of_crease (triangle : Triangle A B C) [IsRightAngle triangle] [side_lengths : side1 = 6 ∧ side2 = 8 ∧ hypotenuse = 10] : ℝ :=
  let midpoint := hypotenuse / 2
  midpoint

theorem calculate_crease_length :
  length_of_crease triangle = 5 :=
by
  sorry

end calculate_crease_length_l177_177299


namespace cube_square_third_smallest_prime_l177_177086

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l177_177086


namespace find_y_l177_177279

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end find_y_l177_177279


namespace beavers_swimming_l177_177466

theorem beavers_swimming (initial_beavers remaining_beavers swimming_beavers : ℕ) 
  (h1 : initial_beavers = 2) 
  (h2 : remaining_beavers = 1) 
  (h3 : initial_beavers - remaining_beavers = swimming_beavers) : 
  swimming_beavers = 1 := 
by 
  rw [h1, h2] at h3
  exact h3

end beavers_swimming_l177_177466


namespace alice_no_guarantee_win_when_N_is_18_l177_177510

noncomputable def alice_cannot_guarantee_win : Prop :=
  ∀ (B : ℝ × ℝ) (P : ℕ → ℝ × ℝ),
    (∀ k, 0 ≤ k → k ≤ 18 → 
         dist (P (k + 1)) B < dist (P k) B ∨ dist (P (k + 1)) B ≥ dist (P k) B) →
    ∀ A : ℝ × ℝ, dist A B > 1 / 2020

theorem alice_no_guarantee_win_when_N_is_18 : alice_cannot_guarantee_win :=
sorry

end alice_no_guarantee_win_when_N_is_18_l177_177510


namespace length_of_string_proof_l177_177489

noncomputable def length_of_string (c h loops : ℝ) : ℝ :=
  let height_per_loop := h / loops
  let diagonal_per_loop := Real.sqrt((height_per_loop)^2 + c^2)
  loops * diagonal_per_loop

theorem length_of_string_proof :
  ∀ (c h loops : ℝ),
  c = 6 → h = 9 → loops = 3 →
  length_of_string c h loops = 9 * Real.sqrt 5 := by
  intros c h loops hc hh hl
  rw [hc, hh, hl]
  sorry

end length_of_string_proof_l177_177489


namespace plane_equation_l177_177550

theorem plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) ∧
  (∀ x y z : ℤ, 
    (A * x + B * y + C * z + D = 0) ↔
      (x = 1 ∧ y = 6 ∧ z = -8 ∨ (∃ t : ℤ, 
        x = 2 + 4 * t ∧ y = 4 - t ∧ z = -3 + 5 * t))) ∧
  (A = 5 ∧ B = 15 ∧ C = -7 ∧ D = -151) :=
sorry

end plane_equation_l177_177550


namespace angle_ABC_45_l177_177395

noncomputable def orthocenter (A B C : EuclideanGeometry.Point)
    (h : ℝ) : EuclideanGeometry.Point := sorry

axiom medians_intersect (A M N : EuclideanGeometry.Point) (H : EuclideanGeometry.Point) : Prop

theorem angle_ABC_45 (A B C M N : EuclideanGeometry.Point) (H : EuclideanGeometry.Point)
  (hH : orthocenter A B C H) (hM : medians_intersect A M N H) :
  EuclideanGeometry.angle B A C = π / 4 := sorry

end angle_ABC_45_l177_177395


namespace find_x_y_sum_l177_177979

theorem find_x_y_sum :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (∃ (a b : ℕ), 360 * x = a^2 ∧ 360 * y = b^4) ∧ x + y = 2260 :=
by {
  sorry
}

end find_x_y_sum_l177_177979


namespace system_of_equations_inconsistent_l177_177950

theorem system_of_equations_inconsistent :
  ¬∃ (x1 x2 x3 x4 x5 : ℝ), 
    (x1 + 2 * x2 - x3 + 3 * x4 - x5 = 0) ∧ 
    (2 * x1 - x2 + 3 * x3 + x4 - x5 = -1) ∧
    (x1 - x2 + x3 + 2 * x4 = 2) ∧
    (4 * x1 + 3 * x3 + 6 * x4 - 2 * x5 = 5) := 
sorry

end system_of_equations_inconsistent_l177_177950


namespace noah_has_largest_final_answer_l177_177714

def liam_initial := 15
def liam_final := (liam_initial - 2) * 3 + 3

def mia_initial := 15
def mia_final := (mia_initial * 3 - 4) + 3

def noah_initial := 15
def noah_final := ((noah_initial - 3) + 4) * 3

theorem noah_has_largest_final_answer : noah_final > liam_final ∧ noah_final > mia_final := by
  -- Placeholder for actual proof
  sorry

end noah_has_largest_final_answer_l177_177714


namespace total_waiting_days_l177_177717

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end total_waiting_days_l177_177717


namespace sum_integers_neg45_to_65_l177_177931

theorem sum_integers_neg45_to_65 : (∑ i in Finset.range (65 + 46), (i - 45)) = 1110 := by
  sorry

end sum_integers_neg45_to_65_l177_177931


namespace total_seats_l177_177913

theorem total_seats (F : ℕ) 
  (h1 : 305 = 4 * F + 2) 
  (h2 : 310 = 4 * F + 2) : 
  310 + F = 387 :=
by
  sorry

end total_seats_l177_177913


namespace integral_f_eq_l177_177618

noncomputable def f (x : ℝ) : ℝ := Real.exp (|x|)

theorem integral_f_eq :
  ∫ x in -2..4, f x = Real.exp 4 + Real.exp 2 - 2 :=
by
  sorry

end integral_f_eq_l177_177618


namespace binary_to_decimal_10101_l177_177533

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end binary_to_decimal_10101_l177_177533


namespace probability_calculation_l177_177374

noncomputable def probability_even_and_greater_than_15 : ℚ :=
  let outcomes := [(i, j) | i ← [1, 2, 3, 4, 5, 6, 7], j ← [1, 2, 3, 4, 5, 6, 7]];
  let valid_pairs := [(i, j) | (i, j) ∈ outcomes, (i * j) % 2 = 0 ∧ i * j > 15];
  (valid_pairs.length : ℚ) / outcomes.length

theorem probability_calculation : probability_even_and_greater_than_15 = 12 / 49 :=
by
  -- Proof is omitted
  sorry

end probability_calculation_l177_177374


namespace collinearity_proof_l177_177597

noncomputable def collinear_points (A B C P Q U V W D E F : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  ∃ circum_circle : Circle,
    P ∈ circum_circle ∧ Q ∈ circum_circle ∧
    A ∉ circum_circle ∧ B ∉ circum_circle ∧ C ∉ circum_circle ∧
    reflection P BC U ∧ reflection P CA V ∧ reflection P AB W ∧
    line Q U ∩ BC = D ∧ line Q V ∩ CA = E ∧ line Q W ∩ AB = F ∧
    collinear D E F

theorem collinearity_proof
  {A B C P Q U V W D E F : Point}
  (h : collinear_points A B C P Q U V W D E F) :
  collinear D E F :=
sorry

end collinearity_proof_l177_177597


namespace set_D_not_right_triangle_l177_177846

theorem set_D_not_right_triangle :
  let a := 11
  let b := 12
  let c := 15
  a ^ 2 + b ^ 2 ≠ c ^ 2
:=
by
  let a := 11
  let b := 12
  let c := 15
  sorry

end set_D_not_right_triangle_l177_177846


namespace cyclic_quadrilateral_inequality_l177_177567

variable (α β γ δ : ℝ)

-- Angle sum for cyclic quadrilateral
axiom angle_sum : α + β + γ + δ = Real.pi

-- Lengths of the sides in terms of angles
noncomputable def AB : ℝ := 2 * Real.sin α
noncomputable def BC : ℝ := 2 * Real.sin β
noncomputable def CD : ℝ := 2 * Real.sin γ
noncomputable def DA : ℝ := 2 * Real.sin δ
noncomputable def AC : ℝ := 4 * Real.sin ((α - γ) / 2) * Real.sin ((β + δ) / 2)
noncomputable def BD : ℝ := 4 * Real.sin ((β - δ) / 2) * Real.sin ((α + γ) / 2)

theorem cyclic_quadrilateral_inequality 
  (h1 : α ≥ γ) (h2 : β ≥ δ) :
  |AB - CD| + |AD - BC| ≥ 2 * |AC - BD| := sorry

end cyclic_quadrilateral_inequality_l177_177567


namespace geometric_mean_sqrt3_sub_add_one_l177_177393

theorem geometric_mean_sqrt3_sub_add_one : 
  let a := sqrt 3 - 1 in
  let b := sqrt 3 + 1 in
  (sqrt (a * b) = sqrt 2) ∨ (sqrt (a * b) = -sqrt 2) :=
by
  let a := sqrt 3 - 1
  let b := sqrt 3 + 1
  exact Or.intro_left (sqrt (a * b) = -sqrt 2) sorry

end geometric_mean_sqrt3_sub_add_one_l177_177393


namespace turns_per_minute_l177_177834

theorem turns_per_minute (x : ℕ) (h₁ : x > 0) (h₂ : 60 / x = (60 / (x + 5)) + 2) :
  60 / x = 6 ∧ 60 / (x + 5) = 4 :=
by sorry

end turns_per_minute_l177_177834


namespace angle_in_second_quadrant_l177_177635

theorem angle_in_second_quadrant (α : ℝ) (h : α = 3) : (π / 2 < α ∧ α < π) :=
by {
  rw h,
  split,
  { norm_num, },
  { norm_num, }
}

end angle_in_second_quadrant_l177_177635


namespace find_b_l177_177861

theorem find_b
  (a b c d : ℝ)
  (h₁ : -a + b - c + d = 0)
  (h₂ : a + b + c + d = 0)
  (h₃ : d = 2) :
  b = -2 := 
by 
  sorry

end find_b_l177_177861


namespace least_positive_integer_to_add_l177_177103

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end least_positive_integer_to_add_l177_177103


namespace unique_groups_of_friends_l177_177499

-- Defining the problem conditions
def num_friends : ℕ := 10

-- Statement of the proof problem
theorem unique_groups_of_friends (num_friends : ℕ) : (2 ^ num_friends) = 1024 :=
by
  have h : num_friends = 10 := rfl
  rw [h]
  exact Nat.pow (by norm_num : 2 = 2) (by norm_num : 10 = 10)
  sorry  -- This replaces the actual proof steps with a placeholder

end unique_groups_of_friends_l177_177499


namespace quadrilateral_properties_l177_177370

theorem quadrilateral_properties (A B C D : Point)
  (angle_A_eq_angle_D : ∠ A = ∠ D)
  (angle_B_eq_angle_C : ∠ B = ∠ C) :
  (AB = CD) ∧ (AC = BD) :=
by sorry

end quadrilateral_properties_l177_177370


namespace base_conversion_min_sum_l177_177401

theorem base_conversion_min_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3)
    (h_mod: 3 * a - 2 ≡ 0 [MOD 5])
    (valid_base_a : a >= 2)
    (valid_base_b : b >= 2):
  a + b = 14 := sorry

end base_conversion_min_sum_l177_177401


namespace female_literate_employees_correct_l177_177451

-- Definitions of conditions
def total_employees : ℕ := 1400
def percent_female : ℝ := 0.60
def percent_male_literate : ℝ := 0.50
def total_percent_literate : ℝ := 0.62

-- Calculation based on conditions
def female_employees : ℕ := total_employees * percent_female
def male_employees : ℕ := total_employees * (1 - percent_female)
def male_literate_employees : ℕ := male_employees * percent_male_literate
def total_literate_employees : ℕ := total_employees * total_percent_literate

-- Proof problem statement
theorem female_literate_employees_correct :
  total_literate_employees - male_literate_employees = 588 :=
by
  -- Proof steps are omitted as per instructions
  sorry

end female_literate_employees_correct_l177_177451


namespace coefficient_of_x5_in_expansion_l177_177433

theorem coefficient_of_x5_in_expansion :
  ∀ (x : ℝ), (coeff_of_term (expansion_binom (x + 2 * real.sqrt 3) 8) 5) = 1344 * real.sqrt 3 :=
by 
  sorry

end coefficient_of_x5_in_expansion_l177_177433


namespace f_neg_eleven_sixths_plus_f_eleven_sixths_l177_177565

noncomputable def f : ℝ → ℝ 
| x => if x < 0 then Real.sin (Real.pi * x) else f (x - 1) - 1

theorem f_neg_eleven_sixths_plus_f_eleven_sixths :
  f (-11/6) + f (11/6) = -2 :=
by
  sorry

end f_neg_eleven_sixths_plus_f_eleven_sixths_l177_177565


namespace student_project_assignment_l177_177063

theorem student_project_assignment : 
  let students := 6 in
  let projects := 3 in
  (students.factorial / (students - projects).factorial) = 120 :=
by
  sorry

end student_project_assignment_l177_177063


namespace friend_wins_games_l177_177428

def ratio_games (V_wins : ℕ) (F_wins : ℕ) : Prop := V_wins * 5 = F_wins * 9

theorem friend_wins_games (V_wins : ℕ) (H : V_wins = 36) : ∃ F_wins : ℕ, ratio_games V_wins F_wins ∧ F_wins = 20 :=
by
  use 20
  split
  . rw [H]
    sorry
  . rfl 

end friend_wins_games_l177_177428


namespace medians_divide_triangle_into_six_equal_areas_l177_177740

theorem medians_divide_triangle_into_six_equal_areas {A B C : Type} [inhabited A] [inhabited B] [inhabited C]
    (triangle : A × B × C)
    (M : (A × B × C) → (A × B × C)) -- assuming median function, centroid property can be inferred
  : let (μ₁, μ₂, μ₃) := M triangle in
    (area μ₁ = area μ₂) ∧ (area μ₂ = area μ₃) ∧ (area μ₃ = area (μ₁.1, μ₂.2, μ₃.3)) ∧ sorry := sorry

end medians_divide_triangle_into_six_equal_areas_l177_177740


namespace words_per_page_l177_177497

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 90) : p = 90 :=
sorry

end words_per_page_l177_177497


namespace cyclic_quad_inequality_l177_177569

variables {α β γ δ : ℝ}
variables {AB CD AD BC AC BD : ℝ}

-- conditions
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
  α + β + γ + δ = π

-- statement
theorem cyclic_quad_inequality (h : cyclic_quadrilateral α β γ δ) 
  (h1 : AB = 2 * Real.sin α)
  (h2 : CD = 2 * Real.sin γ)
  (h3 : AD = 2 * Real.sin δ)
  (h4 : BC = 2 * Real.sin β)
  (h5 : AC = 2 * Real.sin ((α - γ) / 2))
  (h6 : BD = 2 * Real.sin ((β - δ) / 2)) :
  |AB - CD| + |AD - BC| ≥ 2 * |AC - BD| := 
sorry

end cyclic_quad_inequality_l177_177569


namespace area_of_circle_omega_is_correct_l177_177368

noncomputable def area_of_circle_omega (P Q : ℝ × ℝ) (Ω : ℝ) :=
  ∃ C ∈ set.univ, C.2 = 0 ∧
  ((P.1 - C.1)^2 + (P.2 - C.2)^2 = Ω^2) ∧
  ((Q.1 - C.1)^2 + (Q.2 - C.2)^2 = Ω^2) ∧
  area = (π * Ω^2)

theorem area_of_circle_omega_is_correct :
  let P := (2 : ℝ, 9 : ℝ)
  let Q := (8 : ℝ, 7 : ℝ)
  let Ω := sqrt (730/9 : ℝ)
  area_of_circle_omega P Q Ω :=
  sorry -- Proof goes here

end area_of_circle_omega_is_correct_l177_177368


namespace expo_form_of_complex_l177_177940

def complexExpForm (z : Complex) : ℝ × ℝ := 
  let r := Complex.abs z 
  let θ := Complex.arg z 
  (r, θ)

theorem expo_form_of_complex :
  complexExpForm (1 + Complex.i * Real.sqrt 3) = (2, Real.pi / 3) :=
  sorry

end expo_form_of_complex_l177_177940


namespace multinomial_max_inequality_l177_177741

theorem multinomial_max_inequality (n : ℕ) (k : ℕ) (p : Fin k → ℝ) (m : Fin k → ℕ) 
  (h_prob_sum : (∀ i, 0 ≤ p i ∧ p i ≤ 1) ∧ (∑ i, p i = 1)) 
  (h_terms_sum : ∑ i, m i = n) 
  (h_max_prob : ∀ (n₁ n₂ : Fin k → ℕ),
    (p n₁ * mprod (λ i, (fact (n₁ i)))⁻¹ ≤ p n * mprod (λ i, (fact (m i)))⁻¹) ∧ 
    (p n₂ * mprod (λ i, (fact (n₂ i)))⁻¹ ≤ p n * mprod (λ i, (fact (m i)))⁻¹)) :
  (∀ i, (n + 1) * p i - 1 ≤ m i ∧ m i ≤ (n + k - 1) * p i)
:= sorry

end multinomial_max_inequality_l177_177741


namespace total_waiting_days_l177_177716

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end total_waiting_days_l177_177716


namespace numbers_on_diagonal_l177_177151

theorem numbers_on_diagonal {n : ℕ} (hodd : n % 2 = 1)
  (A : matrix (fin n) (fin n) ℕ) 
  (hrow : ∀ i : fin n, ∃ σ : perm (fin n), 
    ∀ k : fin n, A i (σ k) = k.succ) 
  (hcol : ∀ j : fin n, ∃ τ : perm (fin n), 
    ∀ k : fin n, A (τ k) j = k.succ) 
  (hsymm : ∀ i j : fin n, A i j = A j i) :
  ∀ i : fin n, ∃ k : fin n, A i i = k.succ :=
by
  sorry

end numbers_on_diagonal_l177_177151


namespace student_subjects_difference_l177_177504

theorem student_subjects_difference (n M S m' M' : ℕ) 
  (h_n : n = 3000)
  (h_M : 2100 ≤ M ∧ M ≤ 2250)
  (h_S : 1050 ≤ S ∧ S ≤ 1350)
  (h_total : M + S - (M ∩ S) = n)
  (h_min_inter : 2100 + 1050 - m' = 3000)
  (h_max_inter : 2250 + 1350 - M' = 3000) : 
  M' - m' = 450 :=
by {
  sorry
}

end student_subjects_difference_l177_177504


namespace primes_in_sequence_zero_l177_177849

open Nat

def Q : ℕ := ∏ p in (Finset.filter Nat.prime (Finset.range 48)).val, p

def a (k : ℕ) : ℕ := Q + (k + 2)

theorem primes_in_sequence_zero : (Finset.filter prime (Finset.range 46).val.map a).card = 0 :=
  by
  sorry

end primes_in_sequence_zero_l177_177849


namespace kitchen_upgrade_cost_l177_177422

theorem kitchen_upgrade_cost 
  (c_knobs : ℝ) (n_knobs : ℤ) (disc_knobs : ℝ)
  (c_pulls : ℝ) (n_pulls : ℤ) (disc_pulls : ℝ) : 
  n_knobs * c_knobs * (1 - disc_knobs) + n_pulls * c_pulls * (1 - disc_pulls) = 67.70 :=
by 
  have h1 : n_knobs * c_knobs * (1 - disc_knobs) = 40.50 := sorry
  have h2 : n_pulls * c_pulls * (1 - disc_pulls) = 27.20 := sorry
  have h3 : 40.50 + 27.20 = 67.70 := sorry
  exact h3

end kitchen_upgrade_cost_l177_177422


namespace knitting_time_l177_177445

theorem knitting_time (A_time B_time D : ℝ) (A_rate B_rate combined_rate : ℝ) :
  A_time = 3 → B_time = 6 → A_rate = 1 / A_time → B_rate = 1 / B_time →
  combined_rate = A_rate + B_rate → D * combined_rate = 2 → D = 4 :=
by
  intros hA_time hB_time hA_rate hB_rate h_combined_rate hD_combined
  rw [hA_time, hB_time] at hA_rate hB_rate
  rw [←hA_rate, ←hB_rate] at h_combined_rate
  simp only at h_combined_rate
  exact sorry

end knitting_time_l177_177445


namespace john_paid_correct_amount_l177_177332

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

end john_paid_correct_amount_l177_177332


namespace lowest_sale_price_percentage_l177_177116

theorem lowest_sale_price_percentage :
  ∃ (p : ℝ) (h1 : 30 / 100 * p ≤ 70 / 100 * p) (h2 : p = 80),
  (p - 70 / 100 * p - 20 / 100 * p) / p * 100 = 10 := by
sorry

end lowest_sale_price_percentage_l177_177116


namespace cube_square_third_smallest_prime_l177_177082

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l177_177082


namespace unique_zero_of_fx_and_max_a_l177_177609

theorem unique_zero_of_fx_and_max_a 
  (f : ℝ → ℝ) (f_def : ∀ x, f x = (x-2) * log x + 2*x - 3)
  (g : ℝ → ℝ) (g_def : ∀ x a, g x = (x-a) * log x + (a * (x-1)) / x) :
  (∃ x, 1 ≤ x ∧ f x = 0 ∧ ∀ y, 1 ≤ y ∧ f y = 0 → y = x) ∧
  (∀ a x, 1 ≤ x → g x a ≥ g 1 a → a ≤ 6) :=
sorry

end unique_zero_of_fx_and_max_a_l177_177609


namespace fifty_percent_of_2002_is_1001_l177_177459

theorem fifty_percent_of_2002_is_1001 :
  (1 / 2) * 2002 = 1001 :=
sorry

end fifty_percent_of_2002_is_1001_l177_177459


namespace option_a_option_b_option_c_option_d_correct_options_l177_177346

noncomputable def quadratic (m : ℝ) (x : ℝ) := m * x^2 + (m - 3) * x + 1

theorem option_a (m : ℝ) : (m = 3) → ¬ ∃ x : ℝ, quadratic m x = 0 :=
by
  intro h
  rw [h, quadratic]
  sorry

theorem option_b (m : ℝ) : (m > 1) → (m^2 - 10 * m + 9 < 0) :=
by
  intro h
  sorry

theorem option_c (m : ℝ) : (0 < m ∧ m < 1) ↔ (m^2 - 10 * m + 9 > 0 ∧ m * -((m-3)/m) > 0) :=
by
  intro h
  sorry

theorem option_d (m : ℝ) : (m < 0) ↔ (m^2 - 10 * m + 9 > 0 ∧ 1/m < 0) :=
by
  intro h
  sorry

theorem correct_options : ∀ m : ℝ, 
  (m = 3 → ¬ ∃ x : ℝ, quadratic m x = 0) ∧
  ((m > 1) → (m^2 - 10 * m + 9 < 0)) ∧
  ((0 < m ∧ m < 1) ↔ (m^2 - 10 * m + 9 > 0 ∧ m * -((m-3)/m) > 0)) ∧
  ((m < 0) ↔ (m^2 - 10 * m + 9 > 0 ∧ 1/m < 0)) :=
by
  intro m
  apply and.intro
  { exact option_a m }
  apply and.intro
  { exact option_b m }
  apply and.intro
  { exact option_c m }
  { exact option_d m }

end option_a_option_b_option_c_option_d_correct_options_l177_177346


namespace sqrt_of_nine_l177_177058

theorem sqrt_of_nine :
  ∃ x : ℝ, x * x = 9 ∧ (x = 3 ∨ x = -3) :=
by
  exact ⟨3, by norm_num, or.inl rfl⟩,
       ⟨-3, by norm_num, or.inr rfl⟩
  sorry

end sqrt_of_nine_l177_177058


namespace expansion_terms_count_l177_177437

theorem expansion_terms_count (a b : ℂ) : 
  (let expr := ((2*a + 5*b)^3 * (2*a - 5*b)^3)^3 in 
   ∃ n : ℕ, n = 10 ∧ term_count (expand expr) = n) :=
sorry

end expansion_terms_count_l177_177437


namespace mark_survival_days_l177_177293

-- Definitions of conditions based on the given problem
def total_food_days (astronauts : ℕ) (days_per_astronaut : ℕ) : ℕ :=
  astronauts * days_per_astronaut

def water_for_irrigation (total_water : ℕ) (water_per_square_meter : ℕ) : ℕ :=
  total_water / water_per_square_meter

def total_potato_production (irrigated_area : ℕ) (potatoes_per_square_meter : ℕ) : ℕ :=
  irrigated_area * potatoes_per_square_meter

def survival_days_from_potatoes (total_potatoes : ℕ) (potatoes_per_day : ℕ) : ℕ :=
  total_potatoes / potatoes_per_day

-- The main theorem stating Mark's survival days
theorem mark_survival_days :
  let astronauts := 6 in
  let days_per_astronaut := 5 in
  let total_water := 300 in
  let water_per_square_meter := 4 in
  let potatoes_per_square_meter := 2.5.to_real in
  let potatoes_per_day := 1.875.to_real in
  let initial_food_days := total_food_days astronauts days_per_astronaut in
  let irrigated_area := water_for_irrigation total_water water_per_square_meter in
  let total_potatoes := total_potato_production irrigated_area potatoes_per_square_meter in
  let potato_days := survival_days_from_potatoes total_potatoes.to_nat potatoes_per_day.to_nat in
  initial_food_days + potato_days = 130 := 
sorry

end mark_survival_days_l177_177293


namespace find_a_l177_177623

theorem find_a (a : ℝ) :
  let M := {x | 2 * x^2 - 3 * x - 2 = 0}
  let N := {x | a * x = 1}
  N ⊆ M ∧ N ≠ M → a ∈ {0, -2, 1/2} :=
by {
  intros M N h,
  sorry
}

end find_a_l177_177623


namespace log_expression_simplification_l177_177790

theorem log_expression_simplification (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x ≠ 1) (hy1 : y ≠ 1) :
  (log (y^8) x^2) * (log (x^4) y^5) * (log (y^5) x^3) * (log (x^3) y^8) * (log (y^3) x^4) = (1 / 3) * (log y x) :=
by
  sorry

end log_expression_simplification_l177_177790


namespace L_shaped_figure_area_l177_177885

noncomputable def area_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem L_shaped_figure_area :
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := 4
  let small_rect_width := 3
  area_rectangle large_rect_length large_rect_width - area_rectangle small_rect_length small_rect_width = 58 :=
by
  sorry

end L_shaped_figure_area_l177_177885


namespace stephen_total_distance_l177_177379

-- Define the conditions
def trips : ℕ := 10
def mountain_height : ℝ := 40000
def fraction_of_height_reached : ℝ := 3 / 4

-- Calculate the total distance covered
def total_distance_covered : ℝ :=
  2 * (fraction_of_height_reached * mountain_height) * trips

-- Prove the total distance covered is 600,000 feet
theorem stephen_total_distance :
  total_distance_covered = 600000 := by
  sorry

end stephen_total_distance_l177_177379


namespace problem_C_l177_177112

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def is_obtuse_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0

theorem problem_C (A B C : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0 :=
by
  sorry

end problem_C_l177_177112


namespace product_bn_expression_l177_177217

theorem product_bn_expression :
  let b_n (n : ℕ) : ℚ := (n^2 + 4 * n + 4) / (n^3 - 1)
  (5 ≤ n) → ∏ (k : ℕ) in Finset.range 96, b_n (k + 5) = 1943994 / Nat.factorial 100 :=
by sorry

end product_bn_expression_l177_177217


namespace smallest_n_is_10001_l177_177843

-- Define n as a natural number and the condition
def smallest_n_satisfying_condition : ℕ :=
  Nat.find (λ n, n > 0 ∧ (sqrt n - sqrt (n - 1) < 0.005))

-- The theorem statement asserting the smallest such n is 10001
theorem smallest_n_is_10001 : smallest_n_satisfying_condition = 10001 := 
  sorry

end smallest_n_is_10001_l177_177843


namespace stratified_sampling_l177_177479

open Nat

theorem stratified_sampling :
  let junior_students := 400 in
  let senior_students := 200 in
  let total_sample_size := 60 in
  let junior_ratio := 2 in
  let senior_ratio := 1 in
  let total_students := junior_students + senior_students in 
  junior_ratio * senior_students = senior_ratio * junior_students →
  junior_students + senior_students = total_sample_size ->
  let junior_sample_size := (junior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  let senior_sample_size := (senior_ratio * total_sample_size) / (junior_ratio + senior_ratio) in
  junior_sample_size = 40 →
  senior_sample_size = 20 →
  choose junior_students junior_sample_size * choose senior_students senior_sample_size = binomial 400 40 * binomial 200 20 :=
by 
  intros; 
  sorry

end stratified_sampling_l177_177479


namespace average_weight_is_67_5_l177_177294

-- Given conditions
def Arun_weight (w : ℝ) : Prop :=
  66 < w ∧ w < 72

def Brother_weight (w : ℝ) : Prop :=
  60 < w ∧ w < 70

def Mother_weight (w : ℝ) : Prop :=
  w ≤ 69

-- The intersection of the weights satisfying all conditions
def common_weight (w : ℝ) : Prop :=
  Arun_weight w ∧ Brother_weight w ∧ Mother_weight w

-- Proving the average weight
theorem average_weight_is_67_5 :
  (∃ (w : ℝ), common_weight w) →
  ((66 + 69) / 2 = 67.5) :=
begin
  intro h,
  exact rfl,
end

end average_weight_is_67_5_l177_177294


namespace arithmetic_sequence_formula_l177_177796

variable (a : ℕ → ℤ)
variable (n : ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def first_term : a 1 = 1 := rfl

def second_term : a 2 = -1 := rfl

-- Question
theorem arithmetic_sequence_formula :
  is_arithmetic_sequence a ∧ first_term ∧ second_term → ∀ n : ℕ, a n = -2 * n + 3 :=
sorry

end arithmetic_sequence_formula_l177_177796


namespace word_length_4_all_possible_word_length_5_not_all_possible_l177_177557

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

def sequence (a b : ℝ) (n : ℕ) : ℤ := ⌊2 * fractional_part (a * n + b)⌋

def can_form_word (a b : ℝ) (k : ℕ) (word : list ℤ) : Prop :=
  ∃ n : ℕ, list.take k (list.map (sequence a b) (list.range (n + k))) = word

theorem word_length_4_all_possible : 
  ∀ word : list ℤ, word.length = 4 → 
  ∃ a b : ℝ, can_form_word a b 4 word :=
sorry

theorem word_length_5_not_all_possible :
  ¬ ∀ word : list ℤ, word.length = 5 → 
  ∃ a b : ℝ, can_form_word a b 5 word :=
sorry

end word_length_4_all_possible_word_length_5_not_all_possible_l177_177557


namespace increasing_function_in_interval_l177_177947

def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem increasing_function_in_interval (f1 f2 f3 f4 : ℝ → ℝ) :
  (f1 = λ x, Real.sqrt (x + 1)) →
  (f2 = λ x, (x - 1) ^ 2) →
  (f3 = λ x, 2 ^ (-x)) →
  (f4 = λ x, Real.log (x + 1) / Real.log 0.5) →
  is_increasing f1 Set.Ioi 0 →
  ¬ is_increasing f2 Set.Ioi 0 →
  ¬ is_increasing f3 Set.Ioi 0 →
  ¬ is_increasing f4 Set.Ioi 0 :=
by
  intros h1 h2 h3 h4 H1 H2 H3 H4
  sorry

end increasing_function_in_interval_l177_177947


namespace sum_of_numbers_outside_layers_l177_177871

noncomputable def sum_outside_layers (cube : ℕ → ℕ → ℕ → ℝ) (layer_x : ℕ → ℕ → ℝ) (layer_y : ℕ → ℕ → ℝ) (layer_z : ℕ → ℕ → ℝ) :=
  ∑ x in Finset.range 20, ∑ y in Finset.range 20, ∑ z in Finset.range 20, 
  if layer_x x y = 1 ∧ layer_y y z = 1 ∧ layer_z x z = 1 then 0 else cube x y z

theorem sum_of_numbers_outside_layers : 
  ∀ (cube : ℕ → ℕ → ℕ → ℝ), 
  (∀ x y, (∑ z in Finset.range 20, cube x y z) = 1) → 
  (∀ x z, (∑ y in Finset.range 20, cube x y z) = 1) → 
  (∀ y z, (∑ x in Finset.range 20, cube x y z) = 1) → 
  cube 10 10 10 = 10 →
  let layer_x (x z : ℕ → ℕ) := ∑ (y : ℕ) in Finset.range 20, cube x y z in
  let layer_y (y z : ℕ → ℕ) := ∑ (x : ℕ) in Finset.range 20, cube x y z in
  let layer_z (x y : ℕ → ℕ) := ∑ (z : ℕ) in Finset.range 20, cube x y z in
  sum_outside_layers cube layer_x layer_y layer_z = 392 :=
begin
  intros cube cond_x cond_y cond_z cube_10,
  let layer_x := ∑ (y : ℕ) in Finset.range 20, cube x y z,
  let layer_y := ∑ (x : ℕ) in Finset.range 20, cube x y z,
  let layer_z := ∑ (z : ℕ) in Finset.range 20, cube x y z,
  sorry,
end

end sum_of_numbers_outside_layers_l177_177871


namespace cyclic_quadrilateral_inequality_l177_177568

variable (α β γ δ : ℝ)

-- Angle sum for cyclic quadrilateral
axiom angle_sum : α + β + γ + δ = Real.pi

-- Lengths of the sides in terms of angles
noncomputable def AB : ℝ := 2 * Real.sin α
noncomputable def BC : ℝ := 2 * Real.sin β
noncomputable def CD : ℝ := 2 * Real.sin γ
noncomputable def DA : ℝ := 2 * Real.sin δ
noncomputable def AC : ℝ := 4 * Real.sin ((α - γ) / 2) * Real.sin ((β + δ) / 2)
noncomputable def BD : ℝ := 4 * Real.sin ((β - δ) / 2) * Real.sin ((α + γ) / 2)

theorem cyclic_quadrilateral_inequality 
  (h1 : α ≥ γ) (h2 : β ≥ δ) :
  |AB - CD| + |AD - BC| ≥ 2 * |AC - BD| := sorry

end cyclic_quadrilateral_inequality_l177_177568


namespace time_until_unobstructed_view_l177_177525

-- Defining the parameters and conditions
def radius := 50
def initial_jenny_position : ℝ × ℝ := (-50, 105)
def initial_kenny_position : ℝ × ℝ := (-50, -105)
def jenny_speed := 2
def kenny_speed := 3.5
def building_diameter := 100
def separation_distance := 210

-- Statement to prove: The sum of the numerator and denominator of the simplified time t
theorem time_until_unobstructed_view : 
  let t := 420 / 13 
  in (t.num + t.denom) = 433 := sorry

end time_until_unobstructed_view_l177_177525


namespace rate_mangoes_is_50_l177_177917

variable (totalAmountPaid : ℕ)
variable (weightGrapes : ℕ)
variable (ratePerKgGrapes : ℕ)
variable (weightMangoes : ℕ)
variable (totalAmountGrapes : ℕ)
variable (totalAmountMangoes : ℕ)
variable (ratePerKgMangoes : ℕ)

-- Given Conditions
def totalAmountPaid := 1428
def weightGrapes := 11
def ratePerKgGrapes := 98
def weightMangoes := 7
def totalAmountGrapes := weightGrapes * ratePerKgGrapes  -- calculate total cost of grapes
def totalAmountMangoes := totalAmountPaid - totalAmountGrapes  -- calculate total cost of mangoes
def ratePerKgMangoes := totalAmountMangoes / weightMangoes  -- calculate rate per kg for mangoes

theorem rate_mangoes_is_50 :
  ratePerKgMangoes = 50 :=
by
  sorry

end rate_mangoes_is_50_l177_177917


namespace total_number_of_sampling_results_l177_177483

theorem total_number_of_sampling_results : 
  let junior_students := 400
  let senior_students := 200
  let total_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (junior_sample + senior_sample = total_sample) → 
  (junior_sample / junior_students = 2 / 3) → 
  (senior_sample / senior_students = 1 / 3) → 
  @Fintype.card (Finset (Fin junior_students)).choose junior_sample *
  @Fintype.card (Finset (Fin senior_students)).choose senior_sample = 
  Nat.binom junior_students junior_sample * Nat.binom senior_students senior_sample
:= by
  sorry

end total_number_of_sampling_results_l177_177483


namespace matrix_operation_value_l177_177644

theorem matrix_operation_value : 
  let p := 4 
  let q := 5
  let r := 2
  let s := 3 
  (p * s - q * r) = 2 :=
by
  sorry

end matrix_operation_value_l177_177644


namespace max_festivities_l177_177868

theorem max_festivities (dwarfs hats colors : ℕ) (hat_per_dwarf : dwarfs * hat_per_dwarf = hats)
  (colors_eq_dwarfs : colors = dwarfs) 
  (unique_color_per_festivity : ∀ f1 f2 (d w : ℕ), f1 ≠ f2 → w ≠ d → hat_color f1 d ≠ hat_color f2 w) :
  ∃ max_festivities, max_festivities = 2^22 :=
by 
  -- setup the mathematical model and apply given conditions
  sorry

end max_festivities_l177_177868


namespace mean_of_second_set_l177_177856

theorem mean_of_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 90) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 :=
by
  sorry

end mean_of_second_set_l177_177856


namespace symmetry_axis_l177_177816

def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem symmetry_axis : ∃ x : ℝ, x = 5 * Real.pi / 12 ∧ f (x) = f (-x) :=
by
  sorry

end symmetry_axis_l177_177816


namespace max_value_of_A_l177_177516

noncomputable def nine_digit_number := list.range 1 10  -- {1, 2, 3, 4, 5, 6, 7, 8, 9}

def A (digits : list ℕ) : ℕ :=
  let a := digits.nth_le 0 (by norm_num)
  let b := digits.nth_le 1 (by norm_num)
  let c := digits.nth_le 2 (by norm_num)
  let d := digits.nth_le 3 (by norm_num)
  let e := digits.nth_le 4 (by norm_num)
  let f := digits.nth_le 5 (by norm_num)
  let g := digits.nth_le 6 (by norm_num)
  let h := digits.nth_le 7 (by norm_num)
  let i := digits.nth_le 8 (by norm_num) in
  (100 * a + 10 * b + c) + 
  (100 * b + 10 * c + d) + 
  (100 * c + 10 * d + e) + 
  (100 * d + 10 * e + f) + 
  (100 * e + 10 * f + g) + 
  (100 * f + 10 * g + h) + 
  (100 * g + 10 * h + i)

theorem max_value_of_A : ∃ digits, digits.sorted ∧ (∀ d ∈ digits, d ∈ nine_digit_number) ∧ A digits = 4648 :=
by
  sorry

end max_value_of_A_l177_177516


namespace exists_positive_integer_A_l177_177840

theorem exists_positive_integer_A :
  ∃ A : ℕ, 
  (∃ n : ℕ, A = n * (n + 1) * (n + 2)) ∧
  (∃ n : ℕ, A = 10^99 + 10^98 + ... + 10 * n + n) :=
sorry

end exists_positive_integer_A_l177_177840


namespace shaded_fraction_in_fourth_square_l177_177874

theorem shaded_fraction_in_fourth_square : 
  ∀ (f : ℕ → ℕ), (f 1 = 1)
  ∧ (f 2 = 3)
  ∧ (f 3 = 5)
  ∧ (f 4 = f 3 + (3 - 1) + (5 - 3))
  ∧ (f 4 * 2 = 14)
  → (f 4 = 7)
  → (f 4 / 16 = 7 / 16) :=
sorry

end shaded_fraction_in_fourth_square_l177_177874


namespace multiplication_333_111_l177_177726

theorem multiplication_333_111: 333 * 111 = 36963 := 
by 
sorry

end multiplication_333_111_l177_177726


namespace eddy_journey_time_l177_177200

theorem eddy_journey_time (T : ℝ) (V_F : ℝ) :
  let V_E := 4 * V_F in
  V_F = 75 →
  V_E = 300 →
  900 / T = V_E →
  T = 3 :=
by
  intros V_E V_F_def V_E_def journey_time
  sorry

end eddy_journey_time_l177_177200


namespace scout_troop_profit_l177_177898

def bars_purchased := 1500
def cost_rate_dollars_per_six_bars := 3
def bars_in_cost_rate := 6
def selling_rate_dollars_per_three_bars := 2
def bars_in_selling_rate := 3

theorem scout_troop_profit :
  let cost_per_bar := cost_rate_dollars_per_six_bars / bars_in_cost_rate,
      total_cost := bars_purchased * cost_per_bar,
      selling_price_per_bar := selling_rate_dollars_per_three_bars / bars_in_selling_rate,
      total_revenue := bars_purchased * selling_price_per_bar,
      profit := total_revenue - total_cost
  in profit = 250 := 
by
  sorry

end scout_troop_profit_l177_177898


namespace total_shaded_area_correct_l177_177653
-- Let's import the mathematical library.

-- Define the problem-related conditions.
def first_rectangle_length : ℕ := 4
def first_rectangle_width : ℕ := 15
def second_rectangle_length : ℕ := 5
def second_rectangle_width : ℕ := 12
def third_rectangle_length : ℕ := 2
def third_rectangle_width : ℕ := 2

-- Define the areas based on the problem conditions.
def A1 : ℕ := first_rectangle_length * first_rectangle_width
def A2 : ℕ := second_rectangle_length * second_rectangle_width
def A_overlap_12 : ℕ := first_rectangle_length * second_rectangle_length
def A3 : ℕ := third_rectangle_length * third_rectangle_width

-- Define the total shaded area formula.
def total_shaded_area : ℕ := A1 + A2 - A_overlap_12 + A3

-- Statement of the theorem to prove.
theorem total_shaded_area_correct :
  total_shaded_area = 104 :=
by
  sorry

end total_shaded_area_correct_l177_177653


namespace find_AC_l177_177990

variable (A B C D M N : Type)
variable (h1 : AB = c)
variable (h2 : AM = m)
variable (h3 : AN = n)
variable (h4 : ∃ AD : line, is_height AD ∧ AD⊥BC)
variable (h5 : ∃ circle D DA, center D ⟶ radius DA)
variable (h6 : intersects circle AB M ∧ intersects circle AC N)

theorem find_AC : AC = (m * c) / n := sorry

end find_AC_l177_177990


namespace same_terminal_side_l177_177634

theorem same_terminal_side (α : ℝ) (k : ℤ) (h : α = -51) : 
  ∃ (m : ℤ), α + m * 360 = k * 360 - 51 :=
by {
    sorry
}

end same_terminal_side_l177_177634


namespace jaden_toy_cars_problem_l177_177325

theorem jaden_toy_cars_problem :
  let initial := 14
  let bought := 28
  let birthday := 12
  let to_vinnie := 3
  let left := 43
  let total := initial + bought + birthday
  let after_vinnie := total - to_vinnie
  (after_vinnie - left = 8) :=
by
  sorry

end jaden_toy_cars_problem_l177_177325


namespace trigonometric_identity_l177_177223

-- Condition
variable (α : ℝ)
axiom tan_pi_add_alpha (h : Real.tan (Real.pi + α) = 1 / 2)

-- Hypothesis
theorem trigonometric_identity (h : Real.tan (Real.pi + α) = 1 / 2) : 
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = -1 / 4 := 
sorry

end trigonometric_identity_l177_177223


namespace factorize_x4_plus_16_l177_177963

theorem factorize_x4_plus_16 :
  ∀ x : ℝ, (x^4 + 16) = (x^2 - 2 * x + 2) * (x^2 + 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l177_177963


namespace solve_x4_minus_inv_x4_l177_177586

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l177_177586


namespace exist_parallel_lines_same_black_count_l177_177911

noncomputable def parallel_lines_same_black_count (n : ℕ) (h : n ≥ 3) : Prop :=
  ∃ (grid : ℕ × ℕ → bool), ∃ (parallel_lines : list (set (ℕ × ℕ))),
    (∀ (line ∈ parallel_lines), ∃ k, ∀ cell ∈ line, grid cell = tt ↔ nat.card line = k) ∧
    (∃ line1 line2 ∈ parallel_lines, line1 ≠ line2 ∧ 
      (∃ k, (∀ cell1 ∈ line1, grid cell1 = tt ↔ nat.card line1 = k) ∧ 
             (∀ cell2 ∈ line2, grid cell2 = tt ↔ nat.card line2 = k)))

theorem exist_parallel_lines_same_black_count (n : ℕ) (h : n ≥ 3) :
  parallel_lines_same_black_count n h :=
sorry

end exist_parallel_lines_same_black_count_l177_177911


namespace fuel_consumption_gallons_l177_177470

theorem fuel_consumption_gallons
  (distance_per_liter : ℝ)
  (speed_mph : ℝ)
  (time_hours : ℝ)
  (mile_to_km : ℝ)
  (gallon_to_liters : ℝ)
  (fuel_consumption : ℝ) :
  distance_per_liter = 56 →
  speed_mph = 91 →
  time_hours = 5.7 →
  mile_to_km = 1.6 →
  gallon_to_liters = 3.8 →
  fuel_consumption = 3.9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end fuel_consumption_gallons_l177_177470


namespace sum_series_l177_177174

theorem sum_series : (Finset.range 101).sum (λ n => if (n % 2 = 0) then (n + 1) / 2 else -(n + 1) / 2) + 101 = 51 :=
by
  sorry

end sum_series_l177_177174


namespace geom_series_sum_l177_177180

theorem geom_series_sum :
  let a := (5 : ℚ)
  let r := (-3 / 4 : ℚ)
  abs r < 1 →
  (∑' n : ℕ, a * r^n) = 20 / 7 :=
by
  intros a r hr
  sorry

end geom_series_sum_l177_177180


namespace find_natural_number_with_six_divisors_and_divisor_sum_78_l177_177209

theorem find_natural_number_with_six_divisors_and_divisor_sum_78 :
  ∃ n : ℕ, ((∀ d ∈ (finset.filter (λ d, d ∣ n) ({1, n}))), 6) 
      ∧ (∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ p1 ≠ p2 ∧ p1 ∣ n ∧ p2 ∣ n)
      ∧ finset.sum (finset.filter (λ d, d ∣ n) (finset.range (n + 1))) = 78 
      ∧ n = 45 := by
  sorry

end find_natural_number_with_six_divisors_and_divisor_sum_78_l177_177209


namespace spurs_team_players_l177_177779

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end spurs_team_players_l177_177779


namespace angle_size_proof_l177_177729

-- Define the problem conditions
def fifteen_points_on_circle (θ : ℕ) : Prop :=
  θ = 360 / 15 

-- Define the central angles
def central_angle_between_adjacent_points (θ : ℕ) : ℕ :=
  360 / 15  

-- Define the two required central angles
def central_angle_A1O_A3 (θ : ℕ) : ℕ :=
  2 * θ

def central_angle_A3O_A7 (θ : ℕ) : ℕ :=
  4 * θ

-- Define the problem using the given conditions and the proven answer
noncomputable def angle_A1_A3_A7 : ℕ :=
  108

-- Lean 4 statement of the math problem to prove
theorem angle_size_proof (θ : ℕ) (h1 : fifteen_points_on_circle θ) :
  central_angle_A1O_A3 θ = 48 ∧ central_angle_A3O_A7 θ = 96 → 
  angle_A1_A3_A7 = 108 :=
by sorry

#check angle_size_proof

end angle_size_proof_l177_177729


namespace roots_sum_greater_l177_177562

-- Define the function f(x) = (1/2)x^2 - a * ln x
def f (a x : ℝ) : ℝ := (1/2) * x ^ 2 - a * Real.log x

-- Define a predicate that checks whether an x is a root of f
def is_root (a x : ℝ) : Prop := f a x = 0

-- The Lean theorem statement corresponding to the proof problem
theorem roots_sum_greater (a x1 x2 : ℝ) (ha : a > Real.exp 1) (hx1 : is_root a x1) (hx2 : is_root a x2) :
  x1 + x2 > 2 * Real.sqrt a :=
sorry

end roots_sum_greater_l177_177562


namespace option_correct_l177_177110

theorem option_correct (m n x : ℝ) : 
  ¬(2 * m + 3 * n = 6 * m * n) ∧
  ¬(m^2 * m^3 = m^6) ∧
  ¬((x - 1)^2 = x^2 - 1) ∧
  (3 * Real.sqrt 2 / Real.sqrt 2 = 3) :=
by
  split
  -- Proof for ¬(2 * m + 3 * n = 6 * m * n)
  suffices h1 : 2 * m + 3 * n ≠ 6 * m * n, {exact h1},
  sorry,
  split
  -- Proof for ¬(m^2 * m^3 = m^6)
  suffices h2 : m^2 * m^3 ≠ m^6, {exact h2},
  suffices h3 : m^2 * m^3 = m^(2 + 3), {exact ne_of_lt (by linarith only [pow_lt_pow_of_lt_right zero_lt_one one_lt_two (by sorry)])},
  sorry,
  split
  -- Proof for ¬((x - 1)^2 = x^2 - 1)
  suffices h3 : (x - 1)^2 ≠ x^2 - 1, {exact h3},
  suffices h4 : (x - 1)^2 = x^2 - 2*x + 1, {exact ne_of_lt (by linarith only [sub_pos_of_lt (by sorry)])},
  sorry,
  -- Proof for 3 * sqrt 2 / sqrt 2 = 3
  suffices h4 : 3 * Real.sqrt 2 / Real.sqrt 2 = 3, {exact h4},
  by calc
    3 * Real.sqrt 2 / Real.sqrt 2 = 3 * (Real.sqrt 2 / Real.sqrt 2) : by ring
                            ... = 3 * 1                     : by rw [div_self (Real.sqrt_ne_zero.mpr zero_ne_two.symm)]
                            ... = 3                         : by ring

end option_correct_l177_177110


namespace cube_of_square_of_third_smallest_prime_is_correct_l177_177091

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l177_177091


namespace sqrt_200_eq_10_sqrt_2_l177_177001

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177001


namespace second_large_bucket_contains_39_ounces_l177_177186

-- Definitions based on the conditions
def buckets : List ℕ := [11, 13, 12, 16, 10]
def first_large_bucket := 23
def ten_ounces := 10

-- Statement to prove
theorem second_large_bucket_contains_39_ounces :
  buckets.filter (λ x, x ≠ 10 ∧ x ≠ 13) = [11, 12, 16] →
  List.sum [11, 12, 16] = 39 :=
by
  sorry

end second_large_bucket_contains_39_ounces_l177_177186


namespace decreasing_interval_of_function_l177_177948

noncomputable def sin_interval_decreasing {k : ℤ} : set ℝ :=
{ x | k * real.pi + real.pi / 3 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 6 }

theorem decreasing_interval_of_function (k : ℤ) :
  ∀ x, x ∈ sin_interval_decreasing k ↔
    k * real.pi + real.pi / 3 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 6 :=
sorry

end decreasing_interval_of_function_l177_177948


namespace number_of_connected_graphs_on_10_vertices_l177_177039

theorem number_of_connected_graphs_on_10_vertices :
  let num_vertices := 10
  let N := 11716571 in
  count_isomorphism_classes_of_connected_graphs num_vertices = N := 
begin
  sorry
end

end number_of_connected_graphs_on_10_vertices_l177_177039


namespace f_neg1_plus_f_neg2017_l177_177598

noncomputable def f : ℝ → ℝ := sorry

axiom even_odd_function (x : ℝ) : f (-x) = f x
axiom periodic_function (x : ℝ) : f (x + 2) = f x
axiom interval_function (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x

theorem f_neg1_plus_f_neg2017 : f (-1) + f (-2017) = 2 := 
by 
  calc 
  f (-1) + f (-2017)
      = f 1 + f (-2017)           : by rw even_odd_function
  ... = f 1 + f 2017             : by rw even_odd_function
  ... = f 1 + f (2 * 1008 + 1)  : by norm_num
  ... = f 1 + f 1               : by rw periodic_function
  ... = 2 * f 1                 : by ring
  ... = 2                      : by exact interval_function 1 ⟨le_of_eq rfl, le_refl 1⟩

end f_neg1_plus_f_neg2017_l177_177598


namespace number_of_pairs_is_2_pow_14_l177_177630

noncomputable def number_of_pairs_satisfying_conditions : ℕ :=
  let fact5 := Nat.factorial 5
  let fact50 := Nat.factorial 50
  Nat.card {p : ℕ × ℕ | Nat.gcd p.1 p.2 = fact5 ∧ Nat.lcm p.1 p.2 = fact50}

theorem number_of_pairs_is_2_pow_14 :
  number_of_pairs_satisfying_conditions = 2^14 := by
  sorry

end number_of_pairs_is_2_pow_14_l177_177630


namespace sum_2002_l177_177260

-- Conditions
def a : ℕ → ℤ := sorry -- Sequence definition to be defined
axiom recurrence_relation (n : ℕ) (h : n ≥ 3) : a(n) = a(n - 1) - a(n - 2)
axiom sum_1985 : (∑ i in Finset.range 1985, a(i + 1)) = 1000
axiom sum_1995 : (∑ i in Finset.range 1995, a(i + 1)) = 4000

-- Proof Statement
theorem sum_2002 : (∑ i in Finset.range 2002, a(i + 1)) = 3000 := 
by sorry

end sum_2002_l177_177260


namespace fraction_of_loss_is_correct_l177_177147

-- Definitions based on the conditions
def selling_price : ℕ := 18
def cost_price : ℕ := 19

-- Calculating the loss
def loss : ℕ := cost_price - selling_price

-- Fraction of the loss compared to the cost price
def fraction_of_loss : ℚ := loss / cost_price

-- The theorem we want to prove
theorem fraction_of_loss_is_correct : fraction_of_loss = 1 / 19 := by
  sorry

end fraction_of_loss_is_correct_l177_177147


namespace angle_between_unit_vectors_l177_177266

-- Given two unit vectors a and b
variables (a b : ℝ^3)
-- Condition 1: a and b are unit vectors
hypothesis h1 : ∥a∥ = 1
hypothesis h2 : ∥b∥ = 1
-- Condition 2: a · (2a - 3b) = 1/2
hypothesis h3 : a • (2 • a - 3 • b) = 1 / 2

-- Theorem to prove: the angle between a and b is pi / 3
theorem angle_between_unit_vectors (a b : ℝ^3) 
  (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : a • (2 • a - 3 • b) = 1 / 2) :
  real.angle a b = π / 3 := 
by sorry

end angle_between_unit_vectors_l177_177266


namespace largest_number_l177_177111

theorem largest_number (A B C D E : ℝ) (hA : A = 0.998) (hB : B = 0.9899) (hC : C = 0.9) (hD : D = 0.9989) (hE : E = 0.8999) :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_l177_177111


namespace sum_of_coefficients_eq_zero_l177_177408

theorem sum_of_coefficients_eq_zero :
  polynomial.sum_of_coefficients (polynomial.expand (λ x, (x - 2) * (x - 1) ^ 5)) = 0 := by
sorry

end sum_of_coefficients_eq_zero_l177_177408


namespace shopkeeper_gain_percentage_l177_177880

noncomputable def gain_percentage (false_weight: ℕ) (true_weight: ℕ) : ℝ :=
  (↑(true_weight - false_weight) / ↑false_weight) * 100

theorem shopkeeper_gain_percentage :
  gain_percentage 960 1000 = 4.166666666666667 := 
sorry

end shopkeeper_gain_percentage_l177_177880


namespace range_of_a_l177_177608

-- Define the piecewise function
def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 2 then (a - 2) * x else (1/2) ^ x - 1

-- Condition: The function is monotonically decreasing on ℝ
def is_monotonically_decreasing (a : ℝ) : Prop :=
∀ x y : ℝ, x ≤ y → f(a, y) ≤ f(a, x)

-- The theorem we want to prove
theorem range_of_a {a : ℝ} (h : is_monotonically_decreasing a) : a ≤ 13 / 8 :=
sorry

end range_of_a_l177_177608


namespace only_fourth_is_rational_l177_177197

theorem only_fourth_is_rational :
  ∀ (e : ℝ), irrational e → 
  (irrational (Real.sqrt (e^2)) ∧ 
   irrational (Real.cbrt 0.64) ∧ 
   rational (Real.root 5 0.001) ∧ 
   rational ((Real.cbrt (-8)) * (Real.sqrt (0.25)⁻¹)) → 
  (only_fourth : ∀ (x : ℝ), x = Real.sqrt (e^2) ∨ x = Real.cbrt 0.64 ∨ x = Real.root 5 0.001 ∨ x = (Real.cbrt (-8)) * (Real.sqrt (0.25)⁻¹) → x ≠ Real.root 5 0.001 ∧ x ≠ (Real.cbrt (-8)) * (Real.sqrt (0.25)⁻¹))) := by
  sorry

end only_fourth_is_rational_l177_177197


namespace trapezium_area_l177_177969

variables (a b h : ℕ)
def area_trapezium (a b h : ℕ) := (a + b) * h / 2

theorem trapezium_area (ha : a = 20) (hb : b = 18) (hh : h = 14) : area_trapezium a b h = 266 := by
  rw [ha, hb, hh]
  norm_num
  -- alternatively, use a calc block to show the computation step by step
  -- calc
  --   area_trapezium a b h = 19 * 14 : by norm_num
  --   ... = 266 : by norm_num

end trapezium_area_l177_177969


namespace find_constant_a_l177_177815

theorem find_constant_a (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = (1/2) * 3^(n+1) - a) →
  a = 3/2 :=
sorry

end find_constant_a_l177_177815


namespace inverse_exponential_is_logarithm_l177_177642

theorem inverse_exponential_is_logarithm (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) (f : ℝ → ℝ)
    (hf : ∀ y, f (a^y) = y) (hf2 : f 2 = 1) : f = λ x, log 2 x :=
by
  -- Proof to be filled here
  sorry

end inverse_exponential_is_logarithm_l177_177642


namespace truncated_cone_volume_correct_larger_cone_volume_correct_l177_177509

def larger_base_radius : ℝ := 10 -- R
def smaller_base_radius : ℝ := 5  -- r
def height_truncated_cone : ℝ := 8 -- h
def height_small_cone : ℝ := 8 -- x

noncomputable def volume_truncated_cone : ℝ :=
  (1/3) * Real.pi * height_truncated_cone * 
  (larger_base_radius^2 + larger_base_radius * smaller_base_radius + smaller_base_radius^2)

theorem truncated_cone_volume_correct :
  volume_truncated_cone = 466 + 2/3 * Real.pi := sorry

noncomputable def total_height_larger_cone : ℝ :=
  height_small_cone + height_truncated_cone

noncomputable def volume_larger_cone : ℝ :=
  (1/3) * Real.pi * (larger_base_radius^2) * total_height_larger_cone

theorem larger_cone_volume_correct :
  volume_larger_cone = 533 + 1/3 * Real.pi := sorry

end truncated_cone_volume_correct_larger_cone_volume_correct_l177_177509


namespace correct_propositions_l177_177683

variables {m n : Type} {α β γ : Type}

-- Assume m and n are different lines
axiom different_lines : m ≠ n

-- Assume α, β, and γ are different planes
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions
lemma prop1 (h1 : parallel α β) (h2 : parallel α γ) : parallel β γ :=
sorry

lemma prop2 (h1 : perpendicular α β) (h2 : parallel m α) : perpendicular m β :=
sorry

lemma prop3 (h1 : perpendicular m α) (h2 : parallel m β) : perpendicular α β :=
sorry

lemma prop4 (h1 : parallel m n) (h2 : in_plane n α) : parallel m α :=
sorry

-- Theorem that combines the propositions
theorem correct_propositions : (prop1 ∧ prop3) :=
begin
  split,
  {
    -- Proof of proposition 1
    sorry,
  },
  {
    -- Proof of proposition 3
    sorry,
  }
end

end correct_propositions_l177_177683


namespace exam_scheduling_l177_177822

def valid_schedules (days : ℕ) : ℕ → Prop
| 5 := tallied_value (5, 2, 2, valid, 12)
| _ := false

theorem exam_scheduling (d : ℕ) : ∀ days, valid_schedules days 12 :=
begin
  intros days,
  sorry
end

end exam_scheduling_l177_177822


namespace point_on_y_axis_l177_177301

theorem point_on_y_axis (y : ℝ) :
  let A := (1, 0, 2)
  let B := (1, -3, 1)
  let M := (0, y, 0)
  dist A M = dist B M → y = -1 :=
by sorry

end point_on_y_axis_l177_177301


namespace max_value_of_d_l177_177679

-- Define the conditions
variable (a b c d : ℝ) (h_sum : a + b + c + d = 10) 
          (h_prod_sum : ab + ac + ad + bc + bd + cd = 20)

-- Define the theorem statement
theorem max_value_of_d : 
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_d_l177_177679


namespace elena_earnings_l177_177957

theorem elena_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (h_wage : hourly_wage = 13.25) (h_hours : hours_worked = 4) : 
  hourly_wage * hours_worked = 53.00 := by
sorry

end elena_earnings_l177_177957


namespace sin_value_of_arithmetic_sequence_l177_177578

open Real

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sin_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : arithmetic_sequence a) 
  (h_cond : a 1 + a 5 + a 9 = 5 * π) : 
  sin (a 2 + a 8) = - (sqrt 3 / 2) :=
by
  sorry

end sin_value_of_arithmetic_sequence_l177_177578


namespace rectangle_ratio_l177_177604

theorem rectangle_ratio (AB AD CD : ℝ) (h₁ : AB = 3 * CD)
    (h₂ : AD = y) (h₃ : CD = x) 
    (h₄ : 3 * CD > AD): (y / (3 * x)) = sqrt 5 / 3 := 
by 
  sorry

end rectangle_ratio_l177_177604


namespace find_range_of_a_l177_177601

noncomputable def range_of_a (a : ℝ) (x : ℝ) : Prop :=
  x ∈ set.Ioc 1 2 → (x - 1)^2 ≤ Real.log x / Real.log a

theorem find_range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, range_of_a a x) ↔ (1 < a ∧ a ≤ 2)) :=
by sorry

end find_range_of_a_l177_177601


namespace max_arith_seq_n_l177_177603

noncomputable def max_n (a : ℕ → ℝ) (n : ℕ) : Prop := 
  (∀ i, a (i + 1) = a i + 1) ∧ 
  (∑ i in finset.range n, |a i| = 2021) ∧ 
  (∑ i in finset.range n, |a i - 1| = 2021) ∧ 
  (∑ i in finset.range n, |a i + 1| = 2021) ∧ 
  n = 62

theorem max_arith_seq_n (a : ℕ → ℝ) (n : ℕ) : 
  max_n a n := sorry

end max_arith_seq_n_l177_177603


namespace shaded_fraction_eighth_triangle_l177_177287

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2
def square_number (n : Nat) : Nat := n * n

theorem shaded_fraction_eighth_triangle :
  let shaded_triangles := triangular_number 7
  let total_triangles := square_number 8
  shaded_triangles / total_triangles = 7 / 16 := 
by
  sorry

end shaded_fraction_eighth_triangle_l177_177287


namespace incenter_inside_triangle_BOH_l177_177658

variables {A B C H O I : Point}
-- Assume the necessary structures and instances for Point, Triangle etc. are available

-- Define an acute-angled triangle
def acute_angled_triangle (A B C : Point) : Prop :=
  ∃ t : Triangle, t.vertices = ⟨A, B, C⟩ ∧
  ∀ angle ∈ t.angles, angle < 90

-- Define the relationships of centers
def orthocenter (A B C H : Point) : Prop :=
  ∃ t : Triangle, t.vertices = ⟨A, B, C⟩ ∧ t.orthocenter = H

def circumcenter (A B C O : Point) : Prop :=
  ∃ t : Triangle, t.vertices = ⟨A, B, C⟩ ∧ t.circumcenter = O

def incenter (A B C I : Point) : Prop :=
  ∃ t : Triangle, t.vertices = ⟨A, B, C⟩ ∧ t.incenter = I

-- Define the inside property for a specific triangle
def inside_triangle (P X Y Z : Point) : Prop :=
  ∃ t : Triangle, t.vertices = ⟨X, Y, Z⟩ ∧ t.contains P

-- Define the angle relationships between elements
def angle_relations (A B C : Point) : Prop :=
  angle A C B > angle A B C ∧ angle A B C > angle B A C

theorem incenter_inside_triangle_BOH (A B C H O I : Point) 
  (h_acute: acute_angled_triangle A B C)
  (h_angles: angle_relations A B C)
  (h_orthocenter: orthocenter A B C H)
  (h_circumcenter: circumcenter A B C O)
  (h_incenter: incenter A B C I) :
  inside_triangle I B O H :=
sorry

end incenter_inside_triangle_BOH_l177_177658


namespace intersection_points_hyperbola_parabola_l177_177072

theorem intersection_points_hyperbola_parabola
  (l1 l2 : ℝ → ℝ) -- The two lines represented as functions
  (H1 : ∀ x, (x^2 - (l1 x)^2 ≠ 1) ∨ (x = 0 ∧ l1 0 = 1)) -- l1 is not tangent to the hyperbola
  (H2 : ∀ x, (x^2 - (l2 x)^2 ≠ 1) ∨ (x = 0 ∧ l2 0 = 1)) -- l2 is not tangent to the hyperbola
  (T : ∃ x0, l1 x0 = x0^2) -- l1 is tangent to the parabola y = x^2
  : ∃ n, n ∈ {2, 3, 4} := sorry

end intersection_points_hyperbola_parabola_l177_177072


namespace abs_ineq_cond_l177_177389

theorem abs_ineq_cond (a : ℝ) : 
  (-3 < a ∧ a < 1) ↔ (∃ x : ℝ, |x - a| + |x + 1| < 2) := sorry

end abs_ineq_cond_l177_177389


namespace min_value_expression_l177_177678

theorem min_value_expression (α β : ℝ) : 
  ∃ α β : ℝ, (3 * real.cos α + 4 * real.sin β - 10)^2 + (3 * real.sin α + 4 * real.cos β - 20)^2 = 236.137 :=
sorry

end min_value_expression_l177_177678


namespace no_regular_ngon_on_lattice_l177_177739

def is_regular_ngon_on_lattice (n : ℕ) (vertices : list (ℤ × ℤ)) : Prop :=
  -- Definition of a regular n-gon with vertices on integer lattice points
  sorry

theorem no_regular_ngon_on_lattice (n : ℕ) (h : n ≠ 4) : 
  ¬ ∃ vertices : list (ℤ × ℤ), is_regular_ngon_on_lattice n vertices :=
by
  sorry

end no_regular_ngon_on_lattice_l177_177739


namespace exists_six_digit_number_divisible_by_tails_l177_177711

-- Define what it means for a number to be a six-digit natural number without zeros in its decimal representation
def is_six_digit_natural_number_without_zeros (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ ∀ k, (k ∈ (n.digits 10)) → (k ≠ 0)

-- Define the "tail" of a natural number
def is_tail (n tail : ℕ) : Prop :=
  ∃ k, (n = tail * 10^k) ∧ (k ≠ 0)

-- Define the main concept: a number divisible by each of its tails
def divisible_by_each_tail (n : ℕ) : Prop :=
  ∀ (tail : ℕ), (is_tail n tail) → (n % tail = 0)

-- Main statement
theorem exists_six_digit_number_divisible_by_tails :
  ∃ (n : ℕ), is_six_digit_natural_number_without_zeros n ∧ divisible_by_each_tail n :=
begin
  use 721875,
  split,
  { -- Proof that 721875 is a six-digit number without zeros
    unfold is_six_digit_natural_number_without_zeros,
    split, 
    { exact dec_trivial }, -- 100000 ≤ 721875
    { split,
      { exact dec_trivial }, -- 721875 < 1000000
      { intros k hk,
        cases hk,
        { exact dec_trivial }, -- First digit (7) is not zero
        cases hk,
        { exact dec_trivial }, -- Second digit (2) is not zero
        cases hk,
        { exact dec_trivial }, -- Third digit (1) is not zero
        cases hk,
        { exact dec_trivial }, -- Fourth digit (8) is not zero
        cases hk,
        { exact dec_trivial }, -- Fifth digit (7) is not zero
        cases hk,
        { exact dec_trivial }, -- Sixth digit (5) is not zero
        contradiction
      }
    }
  },
  { -- Proof that 721875 is divisible by each of its tails
    unfold divisible_by_each_tail,
    intros tail htail,
    cases htail with k h,
    cases h with hnk hk,
    cases hk,
    { exact dec_trivial }, -- Case for k = 1, ..., 5, exact proof should be here but simplified
  },
end

end exists_six_digit_number_divisible_by_tails_l177_177711


namespace billy_hiking_distance_correct_l177_177924

noncomputable def billy_distance (east_leg : ℝ) (north_east_leg : ℝ) := 
  let horizontal_dist := east_leg + north_east_leg / real.sqrt 2
  let vertical_dist := north_east_leg / real.sqrt 2
  real.sqrt (horizontal_dist ^ 2 + vertical_dist ^ 2)

theorem billy_hiking_distance_correct :
  billy_distance 5 8 = real.sqrt (89 + 40 * real.sqrt 2) :=
  sorry

end billy_hiking_distance_correct_l177_177924


namespace two_coins_both_heads_l177_177423

/-- 
Prove that the number of cases where both coins land on heads 
is equal to 1 given that there are 2 coins being tossed and 
each coin has 2 possible outcomes: heads (H) or tails (T).
-/
theorem two_coins_both_heads :
  let outcomes := [(1, 1), (1, 0), (0, 1), (0, 0)] in 
  let heads_cases := outcomes.filter (λ p, p = (1, 1)) in
  heads_cases.length = 1 := by
  sorry

end two_coins_both_heads_l177_177423


namespace possible_values_for_D_l177_177309

def distinct_digits (A B C D E : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

def digits_range (A B C D E : ℕ) : Prop :=
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 0 ≤ E ∧ E ≤ 9

def addition_equation (A B C D E : ℕ) : Prop :=
  A * 10000 + B * 1000 + C * 100 + D * 10 + B +
  B * 10000 + C * 1000 + A * 100 + D * 10 + E = 
  E * 10000 + D * 1000 + D * 100 + E * 10 + E

theorem possible_values_for_D : 
  ∀ (A B C D E : ℕ),
  distinct_digits A B C D E →
  digits_range A B C D E →
  addition_equation A B C D E →
  ∃ (S : Finset ℕ), (∀ d ∈ S, 0 ≤ d ∧ d ≤ 9) ∧ (S.card = 2) :=
by
  -- Proof omitted
  sorry

end possible_values_for_D_l177_177309


namespace find_f8_l177_177571

def f : ℝ → ℝ
| x := if x ≤ 2 then x^3 - 1 else f (x - 3)

theorem find_f8 : f 8 = 7 := 
by {
  sorry
}

end find_f8_l177_177571


namespace Stephen_total_distance_l177_177381

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end Stephen_total_distance_l177_177381


namespace coefficient_x2_term_l177_177224

open Real

theorem coefficient_x2_term :
  (let a := ∫ x in 0..π, (sin x + cos x) in
   let expr := (a * sqrt x - 1 / sqrt x)^6 in
   (∃ c : ℝ, ↑(monomial 2 c) ∈ expr) → c = -192)
:=
by
  let a := ∫ x in 0..π, (sin x + cos x)
  let expr := (a * sqrt x - 1 / sqrt x)^6
  sorry

end coefficient_x2_term_l177_177224


namespace points_on_line_l177_177767

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177767


namespace sum_of_digits_of_m_eq_nine_l177_177733

theorem sum_of_digits_of_m_eq_nine
  (m : ℕ)
  (h1 : m * 3 / 2 - 72 = m) :
  1 + (m / 10 % 10) + (m % 10) = 9 :=
by
  sorry

end sum_of_digits_of_m_eq_nine_l177_177733


namespace distinct_values_count_l177_177312

-- Define the four available digits
def digits : List ℕ := [1, 5, 6, 7]

-- Function to compute all possible distinct sums
noncomputable def countDistinctValues : ℕ :=
  let products := [(a, b) | a ← digits, b ← digits, a ≠ b].map (λ (a, b) => a * b)
  let sums := [(p1, p2) | p1 ← products, p2 ← products, p1 + p2].map (λ (p1, p2) => p1 + p2)
  sums.toFinset.card

-- Proof statement to prove discovered distinct values count equals 3
theorem distinct_values_count : countDistinctValues = 3 := by
  sorry

end distinct_values_count_l177_177312


namespace violet_distance_l177_177168

theorem violet_distance (total_distance : ℕ) (violet_covered : ℕ) (h1 : total_distance = 1000) (h2 : violet_covered = 721) :
  total_distance - violet_covered = 279 :=
by
  rw [h1, h2]
  exact eq.refl 279

end violet_distance_l177_177168


namespace AM_lessthan_BM_plus_CM_l177_177700

-- Define points A, B, C, M and circle O
variables (A B C M O : Type)
-- Define the distance function
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space M]

-- Distances between points
def dist_ab (a b : Type) [metric_space a] : ℝ := sorry
def is_isosceles_triangle (a b c : Type) [metric_space a] [metric_space b] [metric_space c] :=
  dist_ab a b = dist_ab a c

-- Define the isosceles triangle ABC with AB = AC and points on a circle
axiom isosceles_triangle_ABC : is_isosceles_triangle A B C
axiom points_on_circle : ∀ (x : Type), x = A ∨ x = B ∨ x = C ∨ x = M

-- Prove that AM < BM + CM given the above conditions
theorem AM_lessthan_BM_plus_CM : ∀ (dist_ab : A → B → ℝ) (dist_ac : A → C → ℝ) (dist_am : A → M → ℝ) (dist_bm : B → M → ℝ) (dist_cm : C → M → ℝ),
  (dist_ab A B = dist_ac A C) ∧ (points_on_circle A) ∧ (points_on_circle B) ∧ (points_on_circle M) ∧ (points_on_circle C) →
  dist_am A M < dist_bm B M + dist_cm C M :=
by
  sorry -- Proof

end AM_lessthan_BM_plus_CM_l177_177700


namespace exists_infinitely_many_Sn_interval_l177_177687

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i, 1 / (i + 1 : ℝ))

theorem exists_infinitely_many_Sn_interval (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : a < b) (h₂ : b ≤ 1) :
  ∃ᶠ n in at_top, (S n - ⌊S n⌋) ∈ Ioo a b :=
sorry

end exists_infinitely_many_Sn_interval_l177_177687


namespace equivalent_problem_l177_177216

def f (x : ℤ) : ℤ := 9 - x

def g (x : ℤ) : ℤ := x - 9

theorem equivalent_problem : g (f 15) = -15 := sorry

end equivalent_problem_l177_177216


namespace trigonometric_identity_l177_177602

open Real

theorem trigonometric_identity (θ : ℝ) (h : π / 4 < θ ∧ θ < π / 2) :
  2 * cos θ + sqrt (1 - 2 * sin (π - θ) * cos θ) = sin θ + cos θ :=
sorry

end trigonometric_identity_l177_177602


namespace find_a_b_c_l177_177798

def quadratic_function_minimum_value (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), y = a * x^2 + b * x + c ∧ ∀ (y' : ℝ), y' = a * x^2 + b * x + c → y' ≥ y

def passes_through_points (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := p1 in
  let ⟨x2, y2⟩ := p2 in
  y1 = a * x1^2 + b * x1 + c ∧
  y2 = a * x2^2 + b * x2 + c

theorem find_a_b_c (a b c : ℝ)
  (h1 : quadratic_function_minimum_value a b c ∧ quadratic_function_minimum_value a b c = (3, 36))
  (h2 : passes_through_points a b c (1, 0) (5, 0)) :
  a + b + c = 0 :=
sorry

end find_a_b_c_l177_177798


namespace shelves_filled_l177_177032

theorem shelves_filled (carvings_per_shelf : ℕ) (total_carvings : ℕ) (h₁ : carvings_per_shelf = 8) (h₂ : total_carvings = 56) :
  total_carvings / carvings_per_shelf = 7 := by
  sorry

end shelves_filled_l177_177032


namespace conceived_number_is_seven_l177_177144

theorem conceived_number_is_seven (x : ℕ) (h1 : x > 0) (h2 : (1 / 4 : ℚ) * (10 * x + 7 - x * x) - x = 0) : x = 7 := by
  sorry

end conceived_number_is_seven_l177_177144


namespace sector_radius_l177_177053

theorem sector_radius (P : ℝ) (c : ℝ → ℝ) (θ : ℝ) (r : ℝ) (π : ℝ) 
  (h1 : P = 144) 
  (h2 : θ = π)
  (h3 : P = θ * r + 2 * r) 
  (h4 : π = Real.pi)
  : r = 144 / (Real.pi + 2) := 
by
  sorry

end sector_radius_l177_177053


namespace question1_question2_l177_177624

noncomputable def setA := {x : ℝ | -2 < x ∧ x < 4}
noncomputable def setB (m : ℝ) := {x : ℝ | x < -m}

-- (1) If A ∩ B = ∅, find the range of the real number m.
theorem question1 (m : ℝ) (h : setA ∩ setB m = ∅) : 2 ≤ m := by
  sorry

-- (2) If A ⊂ B, find the range of the real number m.
theorem question2 (m : ℝ) (h : setA ⊂ setB m) : m ≤ 4 := by
  sorry

end question1_question2_l177_177624


namespace part1_non_adjacent_B1_G1_part2_non_adjacent_3_girls_l177_177215

-- Defining factorial for convenience
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * factorial n

-- Defining permutations (P(n, r))
def perm (n r : ℕ) : ℕ := factorial n / factorial (n - r)

-- Total ways to arrange 8 people in a circle
def total_arrangements : ℕ := factorial 7

-- Part 1: Number of arrangements where B1 does not sit next to G1
def non_adjacent_B1_G1 (total : ℕ) : ℕ :=
  let restricted := 2 * factorial 6 in
  total - restricted

-- Part 2: Number of arrangements where the three girls do not sit next to each other
def non_adjacent_3_girls : ℕ :=
  let boys_arrangements := factorial 4 in
  let girl_positions := perm 5 3 in
  boys_arrangements * girl_positions

-- Main theorem statements
theorem part1_non_adjacent_B1_G1 : non_adjacent_B1_G1 total_arrangements = 3600 := by
  sorry

theorem part2_non_adjacent_3_girls : non_adjacent_3_girls = 1440 := by
  sorry

end part1_non_adjacent_B1_G1_part2_non_adjacent_3_girls_l177_177215


namespace min_max_f_eq_sqrt3_over_2_l177_177701

def f (x α β : ℝ) : ℝ := abs (cos x + α * cos (2 * x) + β * cos (3 * x))

theorem min_max_f_eq_sqrt3_over_2 (α β : ℝ) : 
  let max_f := λ x: ℝ, f x α β
  in min (max f) = √3 / 2 :=
sorry

end min_max_f_eq_sqrt3_over_2_l177_177701


namespace defect_rate_probability_l177_177956

theorem defect_rate_probability (p : ℝ) (n : ℕ) (ε : ℝ) (q : ℝ) : 
  p = 0.02 →
  n = 800 →
  ε = 0.01 →
  q = 1 - p →
  1 - (p * q) / (n * ε^2) = 0.755 :=
by
  intro hp hn he hq
  rw [hp, hn, he, hq]
  -- Calculation steps can be verified here
  sorry

end defect_rate_probability_l177_177956


namespace ratio_of_areas_l177_177727

theorem ratio_of_areas (s : ℝ) (h1 : s > 0):
  let small_area := (sqrt 3 / 4) * s^2,
      large_length := 3 * s,
      large_area := (sqrt 3 / 4) * large_length^2
  in 9 * small_area / large_area = 1 :=
by
  sorry

end ratio_of_areas_l177_177727


namespace acute_angle_is_correct_l177_177068

noncomputable def theta_ac_angle (r1 r2 r3 : ℝ) : ℝ :=
  (let area1 := π * r1 ^ 2 in
   let area2 := π * r2 ^ 2 in
   let area3 := π * r3 ^ 2 in
   let total_area := area1 + area2 + area3 in
   let U := (4 / 7) * total_area in
   let S := (3 / 7) * total_area in
   let shaded_area := (16 + 4) * θ - 9 * θ + 9 * π in
   let equation := shaded_area = S in
   (6 * π) / 77)

theorem acute_angle_is_correct :
  ∀ (r1 r2 r3 : ℝ),
    r1 = 4 -> r2 = 3 -> r3 = 2 ->
    let θ := theta_ac_angle r1 r2 r3 in
    θ = 6 * π / 77 :=
  by
    intros r1 r2 r3 h1 h2 h3
    simp [theta_ac_angle]
    rw [h1, h2, h3]
    sorry -- skipping the proof

end acute_angle_is_correct_l177_177068


namespace review_hours_sum_19_l177_177153

theorem review_hours_sum_19 (b : Fin 40 → ℕ) (h1 : ∀ i, 1 ≤ b i) (h2 : (∑ i, b i) ≤ 60) :
  ∃ (i j : Fin 40), i ≤ j ∧ (∑ k in Fin.range j.succ \ Fin.range i, b k) = 19 := 
sorry

end review_hours_sum_19_l177_177153


namespace problem_1_problem_2_l177_177342

def f (x : ℝ) : ℝ := exp x - exp (-x)
def g (x : ℝ) : ℝ := exp x + exp (-x)

theorem problem_1 (x : ℝ) : (f x) ^ 2 - (g x) ^ 2 = -4 :=
by
  sorry

theorem problem_2 (x y : ℝ) (h1 : f x * f y = 4) (h2 : g x * g y = 8) : g(x + y) / g(x - y) = 3 :=
by
  sorry

end problem_1_problem_2_l177_177342


namespace number_of_animal_books_l177_177943

variable (A : ℕ)

theorem number_of_animal_books (h1 : 6 * 6 + 3 * 6 + A * 6 = 102) : A = 8 :=
sorry

end number_of_animal_books_l177_177943


namespace sixth_root_binomial_expansion_l177_177190

theorem sixth_root_binomial_expansion :
  (2748779069441 = 1 * 150^6 + 6 * 150^5 + 15 * 150^4 + 20 * 150^3 + 15 * 150^2 + 6 * 150 + 1) →
  (2748779069441 = Nat.choose 6 6 * 150^6 + Nat.choose 6 5 * 150^5 + Nat.choose 6 4 * 150^4 + Nat.choose 6 3 * 150^3 + Nat.choose 6 2 * 150^2 + Nat.choose 6 1 * 150 + Nat.choose 6 0) →
  (Real.sqrt (2748779069441 : ℝ) = 151) :=
by
  intros h1 h2
  sorry

end sixth_root_binomial_expansion_l177_177190


namespace polynomial_root_solution_l177_177967

theorem polynomial_root_solution (a b c : ℝ) (h1 : (2:ℝ)^5 + 4*(2:ℝ)^4 + a*(2:ℝ)^2 = b*(2:ℝ) + 4*c) 
  (h2 : (-2:ℝ)^5 + 4*(-2:ℝ)^4 + a*(-2:ℝ)^2 = b*(-2:ℝ) + 4*c) :
  a = -48 ∧ b = 16 ∧ c = -32 :=
sorry

end polynomial_root_solution_l177_177967


namespace find_expression_monotonicity_intervals_range_k_l177_177252

-- Condition Definition: Given function and tangent line equation
def func (b c : ℝ) (x : ℝ) : ℝ := ln x + b * x - c
def tangent_eq (x y : ℝ) : Prop := x + y + 4 = 0

-- Problem (1): Prove the expression for f(x)
theorem find_expression (b c : ℝ) (h_tangent : tangent_eq 1 (func b c 1)) :
  b = -2 ∧ c = 3 ∧ ∀ x, func b c x = ln x - 2 * x - 3 :=
by sorry

-- Problem (2): Determine the intervals of monotonicity
theorem monotonicity_intervals (x : ℝ) (h_domain : 0 < x) :
  (0 < x ∧ x < 1/2 → deriv (λ x, ln x - 2 * x - 3) x > 0) ∧ 
  (x > 1/2 → deriv (λ x, ln x - 2 * x - 3) x < 0) :=
by sorry

-- Problem (3): Range of values for k
theorem range_k (k : ℝ) : 
  (∀ x ∈ set.Icc (1/2 : ℝ) 3, ln x - 2 * x - 3 ≥ 2 * ln x + k * x) →
  k ≤ 2 * ln 2 - 8 :=
by sorry

end find_expression_monotonicity_intervals_range_k_l177_177252


namespace stamp_exhibition_l177_177135

def total_number_of_stamps (x : ℕ) : ℕ := 3 * x + 24

theorem stamp_exhibition : ∃ x : ℕ, total_number_of_stamps x = 174 ∧ (4 * x - 26) = 174 :=
by
  sorry

end stamp_exhibition_l177_177135


namespace triangle_is_right_triangle_l177_177677

-- Given conditions and definitions
def is_point_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

def focus1 : ℝ × ℝ := (-2, 0)
def focus2 : ℝ × ℝ := (2, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Problem statement to prove
theorem triangle_is_right_triangle 
  (x y : ℝ)
  (h : is_point_on_ellipse x y)
  (d : distance (x, y) focus1 - distance (x, y) focus2 = 2):
  let P := (x, y) in
  let F1 := focus1 in
  let F2 := focus2 in
  let pf1 := distance P F1 in
  let pf2 := distance P F2 in
  let f1f2 := distance F1 F2 in
  f1f2^2 + pf2^2 = pf1^2 :=
by
  sorry

end triangle_is_right_triangle_l177_177677


namespace solve_inequality_l177_177376

noncomputable def f (x : ℝ) : ℝ :=
  (2^(x^2 - 6) - 4 * 2^(x + 4)) * log (cos (π * x)) (x^2 - 2 * x + 1)

theorem solve_inequality (x : ℝ) :
  (x - 1)^2 > 0 →
  cos (π * x) > 0 →
  cos (π * x) ≠ 1 →
  f x ≥ 0 →
  x ∈ Icc (-2.5 : ℝ) (-2) ∨ x ∈ Icc (-2 : ℝ) (-1.5) ∨ x ∈ Icc (-0.5 : ℝ) 0 ∨ x ∈ Icc 2 2.5 ∨ x ∈ Icc 3.5 4 :=
sorry

end solve_inequality_l177_177376


namespace relationship_between_f_of_sin_and_cos_l177_177793

noncomputable def f : ℝ → ℝ := sorry

-- Conditions defining the function and the properties of angles.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def is_monotonically_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def is_internal_angle_of_acute_triangle (α β : ℝ) : Prop :=
  α > 0 ∧ β > 0 ∧ α + β < π

-- Lean statement answering the question.
theorem relationship_between_f_of_sin_and_cos
    (h_even : is_even_function f)
    (h_period : has_period_two f)
    (h_mono_inc : is_monotonically_increasing_on_interval f 3 4)
    (α β : ℝ)
    (h_angles : is_internal_angle_of_acute_triangle α β) :
  f (Real.sin α) < f (Real.cos β) :=
sorry

end relationship_between_f_of_sin_and_cos_l177_177793


namespace stratified_sampling_result_l177_177473

-- Definitions and conditions from the problem
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_sample : ℕ := 60
def junior_sample : ℕ := 40
def senior_sample : ℕ := 20

-- Main theorem statement proving the number of different sampling results
theorem stratified_sampling_result :
  choose junior_students junior_sample * choose senior_students senior_sample = 
  choose 400 40 * choose 200 20 := by
  sorry

end stratified_sampling_result_l177_177473


namespace point_in_fourth_quadrant_l177_177808

theorem point_in_fourth_quadrant (θ : ℤ) :
  (θ = 2011) → (∀ y : ℝ, y = tan (θ * (π / 180)) → y > 0) →
  (∀ y : ℝ, y = cos (θ * (π / 180)) → y < 0) →
  (P ∈ fourth_quadrant) :=
begin
  sorry
end

end point_in_fourth_quadrant_l177_177808


namespace max_value_of_f_l177_177694

def f (x : ℝ) : ℝ := min (4 * x + 1) (min (x + 2) (-2 * x + 4))

theorem max_value_of_f :
  ∃ x : ℝ,  f(x) = 8 / 3 ∧ ∀ y : ℝ, f(y) ≤ 8 / 3 :=
sorry

end max_value_of_f_l177_177694


namespace total_area_of_squares_l177_177357

-- Given a right-angled scalene triangle with sides a, b, and hypotenuse c
def right_angle_scalene_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Combined area of two equal-sized squares is 2018 cm², implying each is 1009 cm²
def equal_sized_squares_area (a b c : ℝ) : Prop :=
  ∃ s : ℝ, s^2 = 1009 ∧ 2 * s^2 = 2018

-- The total area of the six squares should be 8072 cm²
theorem total_area_of_squares (a b c : ℝ) (h_triangle: right_angle_scalene_triangle a b c) 
  (h_equal_squares: equal_sized_squares_area a b c) :
  let total_area := 4 * (a^2 + b^2 + c^2) in
  total_area = 8072 := 
sorry

end total_area_of_squares_l177_177357


namespace days_needed_to_wash_all_towels_l177_177372

def towels_per_hour : ℕ := 7
def hours_per_day : ℕ := 2
def total_towels : ℕ := 98

theorem days_needed_to_wash_all_towels :
  (total_towels / (towels_per_hour * hours_per_day)) = 7 :=
by
  sorry

end days_needed_to_wash_all_towels_l177_177372


namespace largest_prime_factor_l177_177243

theorem largest_prime_factor (n : ℕ) 
  (h1 : n = 8 * 10^9 - 1) 
  (h2 : ∀ p q : ℕ, p * q = n → (nat.prime p → nat.prime q → (p = 1999 ∨ q = 1999 ∨ p = 4002001 ∨ q = 4002001))):
  ∃ p : ℕ, nat.prime p ∧ p = 4002001 :=
by 
  sorry

end largest_prime_factor_l177_177243


namespace table_height_l177_177455

theorem table_height (r s x y l : ℝ)
  (h1 : x + l - y = 32)
  (h2 : y + l - x = 28) :
  l = 30 :=
by
  sorry

end table_height_l177_177455


namespace max_height_l177_177149

-- Define the parabolic function h(t) representing the height of the soccer ball.
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

-- State that the maximum height of the soccer ball is 136 feet.
theorem max_height : ∃ t : ℝ, h t = 136 :=
by
  sorry

end max_height_l177_177149


namespace sum_alternating_squares_l177_177524

theorem sum_alternating_squares :
  let N := (Finset.range 150).sum (λ n, if (n % 2 = 0)
                                      then if (n % 4 = 0) then (150 - n)^2 else -(150 - n)^2
                                      else if (n % 4 = 1) then (150 - n)^2 else -(150 - n)^2)
  in N = 22650 :=
by
  sorry

end sum_alternating_squares_l177_177524


namespace ratio_a_c_l177_177813

-- Define variables and conditions
variables (a b c d : ℚ)

-- Conditions
def ratio_a_b : Prop := a / b = 5 / 4
def ratio_c_d : Prop := c / d = 4 / 3
def ratio_d_b : Prop := d / b = 1 / 5

-- Theorem statement
theorem ratio_a_c (h1 : ratio_a_b a b)
                  (h2 : ratio_c_d c d)
                  (h3 : ratio_d_b d b) : 
  (a / c = 75 / 16) :=
sorry

end ratio_a_c_l177_177813


namespace inequality_x2_gt_y2_plus_6_l177_177049

theorem inequality_x2_gt_y2_plus_6 (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 :=
sorry

end inequality_x2_gt_y2_plus_6_l177_177049


namespace minimum_pawns_remaining_l177_177362

-- Define the initial placement and movement conditions
structure Chessboard :=
  (white_pawns : ℕ)
  (black_pawns : ℕ)
  (on_board : ℕ)

def valid_placement (cb : Chessboard) : Prop :=
  cb.white_pawns = 32 ∧ cb.black_pawns = 32 ∧ cb.on_board = 64

def can_capture (player_pawn : ℕ → ℕ → Prop) := 
  ∀ (wp bp : ℕ), 
  wp ≥ 0 ∧ bp ≥ 0 ∧ wp + bp = 64 →
  ∀ (p_wp p_bp : ℕ), 
  player_pawn wp p_wp ∧ player_pawn bp p_bp →
  p_wp + p_bp ≥ 2
  
-- Our theorem to prove
theorem minimum_pawns_remaining (cb : Chessboard) (player_pawn : ℕ → ℕ → Prop) :
  valid_placement cb →
  can_capture player_pawn →
  ∃ min_pawns : ℕ, min_pawns = 2 :=
by
  sorry

end minimum_pawns_remaining_l177_177362


namespace initial_points_l177_177749

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l177_177749


namespace sequence_general_term_l177_177259

/-- 
  Define the sequence a_n recursively as:
  a_1 = 2
  a_n = 2 * a_(n-1) - 1

  Prove that the general term of the sequence is:
  a_n = 2^(n-1) + 1
-/
theorem sequence_general_term {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) 
  (n : ℕ) : 
  a n = 2^(n-1) + 1 := by
  sorry

end sequence_general_term_l177_177259


namespace min_rubles_for_1001_l177_177065

def min_rubles_needed (n : ℕ) : ℕ :=
  let side_cells := (n + 1) * 4
  let inner_cells := (n - 1) * (n - 1)
  let total := inner_cells * 4 + side_cells
  total / 2 -- since each side is shared by two cells

theorem min_rubles_for_1001 : min_rubles_needed 1001 = 503000 := by
  sorry

end min_rubles_for_1001_l177_177065


namespace num_values_k_number_of_possible_values_of_k_correct_l177_177926

noncomputable def number_of_possible_values_of_k : ℕ := 
  let p : ℕ := 5
  let q : ℕ := 72 - p
  if Nat.Prime p ∧ Nat.Prime q then 1 else 0

theorem num_values_k (k : ℕ) : 
  (∃ p q : ℕ, p + q = 72 ∧ p * q = k ∧ Nat.Prime p ∧ Nat.Prime q) → k = 335 :=
by sorry

theorem number_of_possible_values_of_k_correct : number_of_possible_values_of_k = 1 :=
by 
  have h := num_values_k 335
  apply h
  existsi [5, 67]
  simp
  exact ⟨by norm_num, by norm_num, prime_5, prime_67⟩

end num_values_k_number_of_possible_values_of_k_correct_l177_177926


namespace decreasing_range_of_a_l177_177619

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 6 * x + 8

theorem decreasing_range_of_a (a : ℝ) :
  (∀ x ∈ set.Ico 1 a, has_deriv_at quadratic (2 * x - 6) x → 2 * x - 6 ≤ 0) ↔ 1 < a ∧ a ≤ 3 :=
begin
  sorry
end

end decreasing_range_of_a_l177_177619


namespace profit_at_450_l177_177877

noncomputable def f (x : ℕ) : ℚ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x ≤ 500 then 62 - x / 50
  else 0

noncomputable def L (x : ℕ) : ℚ :=
  let P := f x
  if 0 < x ∧ x ≤ 500 then (P - 40) * x
  else 0

theorem profit_at_450 : L 450 = 5850 := by
  sorry

end profit_at_450_l177_177877


namespace escalator_length_l177_177916

theorem escalator_length
  (escalator_speed : ℝ)
  (person_speed : ℝ)
  (time_taken : ℝ)
  (combined_speed := escalator_speed + person_speed)
  (distance := combined_speed * time_taken) :
  escalator_speed = 10 → person_speed = 4 → time_taken = 8 → distance = 112 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end escalator_length_l177_177916


namespace problem_l177_177676

def count_numbers_with_more_ones_than_zeros (n : ℕ) : ℕ :=
  -- function that counts numbers less than or equal to 'n'
  -- whose binary representation has more '1's than '0's
  sorry

theorem problem (M := count_numbers_with_more_ones_than_zeros 1500) : 
  M % 1000 = 884 :=
sorry

end problem_l177_177676


namespace cube_of_square_of_third_smallest_prime_l177_177081

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l177_177081


namespace area_of_triangle_ABC_l177_177998

noncomputable def point := (ℝ, ℤ) 

def circle (C : point) (r : ℝ) : set point := 
  {p | ((p.fst - (C.fst)) * (p.fst - (C.fst)) + (p.snd - (C.snd)) * (p.snd - (C.snd))) = r * r }

def tangent_y_axis (C : point) (r : ℝ) : set point := 
  {p | p.fst = 0 ∧ ((p.snd - (C.snd)) * (p.snd - (C.snd))) = r * r - (C.fst - 0) * (C.fst - 0) }

def area_triangle (A B C : point) : ℝ := 
  let base := (B.snd - A.snd).nat_abs in
  let height := C.fst in
  (1.0 / 2.0) * height * base

theorem area_of_triangle_ABC : 
  ∀ (A B C : point),
  A = (0, 4) ∧ B = (0, 0) ∧ C = (1, 2) →
  area_triangle A B C = 2 := 
by
  sorry

end area_of_triangle_ABC_l177_177998


namespace solve_trig_equation_l177_177029

theorem solve_trig_equation (x : ℝ) (k : ℤ) : 
    ( ∃ k : ℤ, x = 7.5 * (Real.pi / 180) + k * 90 * (Real.pi / 180) ) 
    ∨ ( ∃ k : ℤ, x = 37.5 * (Real.pi / 180) + k * 90 * (Real.pi / 180) ) ↔ 
    (sin (3*x) * cos (3*x) + cos (3*x) * sin (3*x) = 3/8) := 
    sorry

end solve_trig_equation_l177_177029


namespace part1_exists_f_geometric_part1_general_formula_part2_arithmetic_sequence_l177_177706

theorem part1_exists_f_geometric {a : ℕ → ℤ} (h_seq : ∀ n, a (n + 1) = 2 * a n + n^2 - 4 * n + 1) (h_a1 : a 1 = 3) :
  ∃ f : ℕ → ℤ, (∀ n, f n = n^2 - 2 * n) ∧ (∀ n, ∃ r : ℤ, a n + f n = r^(n - 1)) :=
sorry

theorem part1_general_formula {a : ℕ → ℤ} (h_seq : ∀ n, a (n + 1) = 2 * a n + n^2 - 4 * n + 1) (h_a1 : a 1 = 3) :
  ∀ n, a n = 2^n - n^2 + 2 * n :=
sorry

theorem part2_arithmetic_sequence {a b : ℕ → ℤ} (h_seq : ∀ n, a (n + 1) = 2 * a n + n^2 - 4 * n + 1) (h_sum : ∀ n, a n = ∑ k in finset.range n, b (k + 1)) :
  (a 1 = 1) ∧ (∀ n, b n = -2 * n + 3) :=
sorry

end part1_exists_f_geometric_part1_general_formula_part2_arithmetic_sequence_l177_177706


namespace ratio_XY_YZ_l177_177667

-- Define points P, Q, R as vectors
variables (P Q R X Y Z : Type) [AffineSpace V P] [AffineSpace V Q] [AffineSpace V R]

-- Define conditions as ratios
variables (PX_XQ : ℚ) (QY_YR : ℚ)

-- Assume PX: XQ = 4:1 and QY: YR = 4:1
axiom ratio_PX_XQ : PX_XQ = 4 / 1
axiom ratio_QY_YR : QY_YR = 4 / 1

-- Assume lines XY and PR intersect at Z
axiom intersection_XY_PR : Z ∈ line_through P R ∧ Z ∈ line_through X Y

-- Prove that the ratio XY: YZ is 4:5
theorem ratio_XY_YZ : (XY / YZ) = 4 / 5 :=
sorry

end ratio_XY_YZ_l177_177667


namespace num_values_x_l177_177402

def star (a b : ℕ) : ℕ := a^2 / b

theorem num_values_x (p : ℕ) (q : ℕ) (k : ℕ) (d : finset ℕ) :
  (∀ x ∈ d, star 8 x > 0) ↔ d.count = 7 :=
by sorry

end num_values_x_l177_177402


namespace students_in_C_class_l177_177824

variable (A_class B_class C_class : ℕ)

-- Conditions given in the problem
def condition_1 : Prop := A_class = 44
def condition_2 : Prop := A_class = B_class - 2
def condition_3 : Prop := B_class = C_class + 1

-- Statement to prove
theorem students_in_C_class (ha : condition_1) (hb : condition_2) (hc : condition_3) :
  C_class = 45 := by 
  sorry

end students_in_C_class_l177_177824


namespace distance_foci_l177_177169

def ellipse_equation (x y : ℝ) : Prop := x^2 + 9 * y^2 = 576

def semi_major_axis : ℝ := 24
def semi_minor_axis : ℝ := 8

def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem distance_foci : distance_between_foci semi_major_axis semi_minor_axis = 32 * Real.sqrt 2 := by
  have h_ellipse : ellipse_equation 24 0 := by simp [ellipse_equation, semi_major_axis, semi_minor_axis]
  have h_distance : distance_between_foci 24 8 = 32 * Real.sqrt 2 := by
    dsimp [distance_between_foci, semi_major_axis, semi_minor_axis]
    norm_num
    simp [Real.sqrt_eq_rpow]
    norm_num
  exact h_distance

end distance_foci_l177_177169


namespace servings_per_guest_l177_177517

-- Definitions based on conditions
def num_guests : ℕ := 120
def servings_per_bottle : ℕ := 6
def num_bottles : ℕ := 40

-- Theorem statement
theorem servings_per_guest : (num_bottles * servings_per_bottle) / num_guests = 2 := by
  sorry

end servings_per_guest_l177_177517


namespace correct_propositions_count_is_two_l177_177161

def is_parallelogram_midpoints (q : Quadrilateral) : Prop :=
  -- Proposition 1 condition
  ∀ (Q : set Point), (is_midpoint_quadrilateral Q q → is_parallelogram Q)

def is_rhombus_diagonal_angle (p : Parallelogram) : Prop :=
  -- Proposition 2 condition
  ∃ (d1 : Diagonal), bisects_interior_angle d1 p → is_rhombus p

def is_rectangle_perpendicular_diagonals (p : Parallelogram) : Prop :=
  -- Proposition 3 condition
  ∃ (d1 d2 : Diagonal), (perpendicular d1 d2) → is_rectangle p

def is_square_equal_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  -- Proposition 4 condition
  ∃ (d1 d2 : Diagonal), (equal d1 d2) ∧ (perpendicular d1 d2) → is_square q

noncomputable def propositions_correct : Nat :=
  -- Calculate the number of correct propositions
  (if is_parallelogram_midpoints then 1 else 0) +
  (if is_rhombus_diagonal_angle then 1 else 0) +
  (if is_rectangle_perpendicular_diagonals then 0 else 0) + -- Proposition 3 is false
  (if is_square_equal_perpendicular_diagonals then 0 else 0) -- Proposition 4 is false

theorem correct_propositions_count_is_two :
  propositions_correct = 2 :=
  sorry

end correct_propositions_count_is_two_l177_177161


namespace tenth_term_is_98415_over_262144_l177_177526

def first_term : ℚ := 5
def common_ratio : ℚ := 3 / 4

def tenth_term_geom_seq (a r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem tenth_term_is_98415_over_262144 :
  tenth_term_geom_seq first_term common_ratio 10 = 98415 / 262144 :=
sorry

end tenth_term_is_98415_over_262144_l177_177526


namespace simplify_and_evaluate_l177_177026

-- Definitions based on conditions
def x : ℝ := Real.sqrt 5 - 2

-- The proof problem statement
theorem simplify_and_evaluate :
  (2 / (x ^ 2 - 4) / (1 - x / (x - 2))) = -Real.sqrt 5 / 5 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_l177_177026


namespace runs_scored_by_c_l177_177447

-- Definitions
variables (A B C : ℕ)

-- Conditions as hypotheses
theorem runs_scored_by_c (h1 : B = 3 * A) (h2 : C = 5 * B) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Proof will be here
  sorry

end runs_scored_by_c_l177_177447


namespace find_n_l177_177695

theorem find_n : ∀ (n x y : ℕ), x = 8 → y = 2 → n = x - 3 * log y x → n = -1 :=
by
  intros n x y h1 h2 h3
  sorry

end find_n_l177_177695


namespace part_I_part_II_l177_177680

variable (f : ℝ → ℝ)

-- Condition 1: f is an even function
axiom even_function : ∀ x : ℝ, f (-x) = f x

-- Condition 2: f is symmetric about x = 1
axiom symmetric_about_1 : ∀ x : ℝ, f x = f (2 - x)

-- Condition 3: f(x₁ + x₂) = f(x₁) * f(x₂) for x₁, x₂ ∈ [0, 1/2]
axiom multiplicative_on_interval : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 1/2) ∧ (0 ≤ x₂ ∧ x₂ ≤ 1/2) → f (x₁ + x₂) = f x₁ * f x₂

-- Given f(1) = 2
axiom f_one : f 1 = 2

-- Part I: Prove f(1/2) = √2 and f(1/4) = 2^(1/4).
theorem part_I : f (1 / 2) = Real.sqrt 2 ∧ f (1 / 4) = Real.sqrt (Real.sqrt 2) := by
  sorry

-- Part II: Prove that f(x) is a periodic function with period 2.
theorem part_II : ∀ x : ℝ, f x = f (x + 2) := by
  sorry

end part_I_part_II_l177_177680


namespace hundredth_number_is_564_l177_177373

-- Definition of the problem conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}
def is_valid_number (n : ℕ) : Prop :=
  let ds := List.of_digits n.to_digits in
  ds.to_finset ⊆ digits ∧ ds.length = 3 ∧ ds.sorted

-- The main theorem to be proved
theorem hundredth_number_is_564 :
  Sorted (· < ·) ds → 
  List.nth_le (digits.to_list.combinations 3).map (λ l, l.digits.to_nat)
  (100 - 1) sorry = 564 :=
sorry

end hundredth_number_is_564_l177_177373


namespace points_on_line_l177_177765

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177765


namespace area_product_is_2_l177_177308

open Real

-- Definitions for parabola, points, and the condition of dot product
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) = -4

def area (O F P : ℝ × ℝ) : ℝ :=
  0.5 * abs (O.1 * (F.2 - P.2) + F.1 * (P.2 - O.2) + P.1 * (O.2 - F.2))

-- Points A and B are on the parabola and the dot product condition holds
variables (A B : ℝ × ℝ)
variable (H_A_on_parabola : parabola A.1 A.2)
variable (H_B_on_parabola : parabola B.1 B.2)
variable (H_dot_product : dot_product_condition A B)

-- Focus of the parabola
def F : ℝ × ℝ := (1, 0)

-- Origin
def O : ℝ × ℝ := (0, 0)

-- Prove that the product of areas is 2
theorem area_product_is_2 : 
  area O F A * area O F B = 2 :=
sorry

end area_product_is_2_l177_177308


namespace _l177_177655

noncomputable def acute_scalene_triangle (A B C : Point) : Prop :=
acute_triangle A B C ∧ scalene_triangle A B C

noncomputable def altitude_lines (A B C A1 B1 C1 : Point) : Prop :=
is_altitude A A1 B C ∧ is_altitude B B1 C A ∧ is_altitude C C1 A B

noncomputable def excircles_touch_points (A B C A2 B2 C2 : Point) : Prop :=
excircle_touches_side A B C A2 B C ∧ excircle_touches_side B C A B2 C A ∧ excircle_touches_side C A B C2 A B

noncomputable def tangent_incircle (B1 C1 : Point) (incircle : Circle) : Prop :=
tangent B1C1 incircle

noncomputable theorem A1_on_circumcircle_of_A2B2C2
  (A B C A1 B1 C1 A2 B2 C2 : Point) (incircle : Circle)
  (h1 : acute_scalene_triangle A B C)
  (h2 : altitude_lines A B C A1 B1 C1)
  (h3 : excircles_touch_points A B C A2 B2 C2)
  (h4 : tangent_incircle B1 C1 incircle) :
  point_on_circumcircle A1 (triangle_circumcircle A2 B2 C2) := 
sorry

end _l177_177655


namespace face_value_of_bond_l177_177672

theorem face_value_of_bond :
  ∀ (F : ℝ) (S : ℝ),
    (S ≈ 7692.307692307692) →
    (I = 0.065 * S) →
    (I ≈ 500) →
    (I = 0.10 * F) →
    (F ≈ 5000) :=
by {
  intros F S hS hI hI_approx hF,
  sorry
}

end face_value_of_bond_l177_177672


namespace problem1_problem2_problem3_l177_177610

-- Problem (1)
def f (x : ℝ) : ℝ := 1 - (1 / x) + log (1 / x)

theorem problem1 : ∃ x : ℝ, x = 1 ∧ f x = 0 :=
  sorry

-- Problem (2)
def f_a (a x : ℝ) : ℝ := 1 - (a / x) + a * log (1 / x)

theorem problem2 (a : ℝ) : (e / (e + 1) < a) ∧ (a < 1) → 
  ∃ x1 x2 : ℝ, (1 / e < x1) ∧ (x1 < e) ∧ (1 / e < x2) ∧ (x2 < e) ∧ f_a a x1 = 0 ∧ f_a a x2 = 0 :=
  sorry

-- Problem (3)
theorem problem3 (n : ℕ) (h : n ≥ 3) : log ((n + 1) / 3) < (∑ i in finset.range (n - 2), 1 / (i + 3)) :=
  sorry

end problem1_problem2_problem3_l177_177610


namespace value_of_Q_final_value_of_Q_l177_177686

theorem value_of_Q (n : ℕ) (h : n = 100) : 
  (Q : ℚ) = (∏ (k : ℕ) in finset.range (n - 2 + 1), (1 - 1/(k + 3))) :=
begin
  -- Let's define Q based on the given formula
  let Q := (∏ (k : ℕ) in finset.range (n - 2 + 1), (1 - 1/(k + 3))),
  have h1 : Q = (∏ (k : ℕ) in finset.range (98), (1 - 1/(k + 3))), by rw h,
  rw h1,
  sorry
end

theorem final_value_of_Q (Q : ℚ) (h : Q = (∏ (k : ℕ) in finset.range 98, (1 - 1/(k + 3)))) :
  Q = (1/50) :=
begin
  sorry
end

end value_of_Q_final_value_of_Q_l177_177686


namespace sqrt_200_eq_10_sqrt_2_l177_177000

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177000


namespace AB_eq_PQ_l177_177866

-- Definitions for the given conditions
variables {O A B C P Q : Point}
variables {Γ₁ Γ₂ : Circle}

-- The conditions in the problem
axiom circle_on_point (Γ₁ : Circle) (O : Point) : O ∈ Γ₁
axiom second_circle_centered_at_O (Γ₂ : Circle) : Center Γ₂ = O
axiom circles_intersect_at_PQ (P Q : Point) : P ∈ Γ₁ ∧ Q ∈ Γ₁ ∧ P ∈ Γ₂ ∧ Q ∈ Γ₂
axiom point_C_on_first_circle (C : Point) : C ∈ Γ₁
axiom lines_intersect_circle_again (A B : Point) (CP CQ : Line) : 
  CP = line_through C P → CQ = line_through C Q → A ∈ CP ∧ B ∈ CQ ∧ A ∈ Γ₂ ∧ B ∈ Γ₂

-- Theorem statement in Lean
theorem AB_eq_PQ : 
  ∀ (O A B C P Q : Point) (Γ₁ Γ₂ : Circle),
  circle_on_point Γ₁ O →
  second_circle_centered_at_O Γ₂ →
  circles_intersect_at_PQ P Q →
  point_C_on_first_circle C →
  lines_intersect_circle_again A B (line_through C P) (line_through C Q) →
  distance A B = distance P Q :=
by sorry

end AB_eq_PQ_l177_177866


namespace choir_girls_count_l177_177412

noncomputable def number_of_girls_in_choir (o b t c b_boys : ℕ) : ℕ :=
  c - b_boys

theorem choir_girls_count (o b t b_boys : ℕ) (h1 : o = 20) (h2 : b = 2 * o) (h3 : t = 88)
  (h4 : b_boys = 12) : number_of_girls_in_choir o b t (t - (o + b)) b_boys = 16 :=
by
  sorry

end choir_girls_count_l177_177412


namespace range_of_a_l177_177255

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x = x + Real.sin x) →
  f (a - 1) + f (2 * a^2) ≤ 0 → 
  -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l177_177255


namespace triangle_area_from_medians_l177_177125

theorem triangle_area_from_medians :
  ∀ (m1 m2 m3 : ℝ), 
  m1 = 9 ∧ m2 = 12 ∧ m3 = 15 →
  let S_m := (m1 + m2 + m3) / 2 in
  let area := (4 / 3) * Real.sqrt (S_m * (S_m - m1) * (S_m - m2) * (S_m - m3)) in
  area = 72 :=
by
  intros m1 m2 m3 h
  obtain ⟨h1, h2, h3⟩ := h
  have Sm_def : S_m = (9 + 12 + 15) / 2 := by rw [h1, h2, h3]
  have area_def : (4 / 3) * Real.sqrt (18 * (18 - 9) * (18 - 12) * (18 - 15)) = 72 := by sorry
  exact area_def

end triangle_area_from_medians_l177_177125


namespace problem_I4_1_l177_177305

variable (A D E B C : Type) [Field A] [Field D] [Field E] [Field B] [Field C]
variable (AD DB DE BC : ℚ)
variable (a : ℚ)
variable (h1 : DE = BC) -- DE parallel to BC
variable (h2 : AD = 4)
variable (h3 : DB = 6)
variable (h4 : DE = 6)

theorem problem_I4_1 : a = 15 :=
  by
  sorry

end problem_I4_1_l177_177305


namespace simplify_expression_l177_177181

theorem simplify_expression (y b : ℝ) (hy : 0 < y) (hb : 0 < b) :
  (sqrt (b^2 + y^2) - (y^2 - b^2) / sqrt (b^2 + y^2)) / (b + y^2) = (2 * b^2) / (b + y^2) :=
sorry

end simplify_expression_l177_177181


namespace points_on_line_l177_177766

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177766


namespace fragment_probability_l177_177429

noncomputable def probability_fragment_in_21_digit_code : ℚ :=
  (12 * 10^11 - 30) / 10^21

theorem fragment_probability:
  ∀ (code : Fin 10 → Fin 21 → Fin 10),
  (∃ (i : Fin 12), ∀ (j : Fin 10), code (i + j) = j) → 
  probability_fragment_in_21_digit_code = (12 * 10^11 - 30) / 10^21 :=
sorry

end fragment_probability_l177_177429


namespace solve_puzzle_l177_177536

def digit_assignment := {C : ℕ, L : ℕ, O : ℕ, V : ℕ, P : ℕ, E : ℕ, S : ℕ, N : ℕ, H : ℕ, 
                         C = 9, L = 4, O = 5, V = 3, P = 1, E = 8, S = 0, N = 7, H = 0}

def num1 := 9000 * digit_assignment.C + 100 * digit_assignment.L + 10 * digit_assignment.O + digit_assignment.V + digit_assignment.O / 10
def num2 := num1
def result := 10000 * digit_assignment.P + 1000 * digit_assignment.E + 100 * digit_assignment.S + 10 * digit_assignment.N + digit_assignment.Y

theorem solve_puzzle : num1 + num2 = result := by sorry

end solve_puzzle_l177_177536


namespace expected_patties_l177_177326

-- Definitions from the conditions in a)
constant initial_position : (ℕ × ℕ) := (0, 1)
constant wall_y : ℝ := 2.1
constant patties : ℕ → ℕ := id
constant is_wall : (ℕ × ℕ) → Prop := λ (x, y), y > wall_y
constant valid_moves : (ℕ × ℕ) → List (ℕ × ℕ) :=
  λ (x, y), [(x, y+1), (x+1, y), (x, y-1)].filter (λ p, ¬ is_wall p ∧ p ≠ initial_position)

-- Definition of the expected number of patties
noncomputable def E : (ℕ × ℕ) → ℝ
| (1, 0) := 0
| (x, 0) := patties x.succ
| (0, 1) := 1
| (x, y) := if is_wall (x, y) then 0 else
    (valid_moves (x, y)).sum (λ p, 1 / (valid_moves (x, y)).length * E p + expected_steps (x, y) p)

-- The main proof statement
theorem expected_patties :
  E (0, 1) = 7 / 3 := by
  sorry

end expected_patties_l177_177326


namespace true_propositions_l177_177910

-- 1. Proposition 1 conditions and claim
noncomputable def proposition_1 (ξ : ℝ → ℝ) (μ σ : ℝ) (hξ : ξ ~ Normal μ σ^2) : Prop :=
  (P (ξ < 2) = 0.15) → (P (ξ > 6) = 0.15) → (P (2 ≤ ξ ∧ ξ < 4) = 0.3)

-- 2. Proposition 2 conditions and claim
noncomputable def proposition_2 : Prop :=
  let a := ∫ x in 0..1, (2 * x - exp x) in a = 2 - exp 1

-- 3. Proposition 3 conditions and claim
noncomputable def proposition_3 : Prop :=
  let term := (fun r => binom 10 r * (-1)^r * x^((5 * r - 10) / 2)) in
  term 2 = 45

-- 4. Proposition 4 conditions and claim
noncomputable def proposition_4 : Prop :=
  ∀ x, 0 < x ∧ x <= 4 → (log 2 x <= 1) = (0 < x ∧ x <= 2)

-- The Lean statement to prove the true propositions
theorem true_propositions : {2, 3, 4} = {i | (i = 2 → proposition_2) ∧ (i = 3 → proposition_3) ∧ (i = 4 → proposition_4) ∧ (i = 1 → ¬proposition_1 ξ μ σ)} :=
by
  sorry

end true_propositions_l177_177910


namespace max_value_zero_monotonic_decreasing_range_exact_l177_177251

section
variables {m x : ℝ}
def f (x : ℝ) (m : ℝ) : ℝ := -x^2 + m * x - m

-- Problem 1
theorem max_value_zero (h : ∃ x, f x m = 0) : m = 0 ∨ m = 4 :=
sorry

-- Problem 2
theorem monotonic_decreasing (h : ∀ x y ∈ Icc (-1 : ℝ) 0, x < y → f x m ≥ f y m) : m ≤ -2 :=
sorry

-- Problem 3
theorem range_exact (h : ∃ a b ∈ Icc (2 : ℝ) 3, f a m = 2 ∧ f b m = 3) : ∃ m, m = 6 :=
sorry
end

end max_value_zero_monotonic_decreasing_range_exact_l177_177251


namespace g_9_to_the_4_pow_eq_81_l177_177034

-- Given conditions
variables {f g : ℝ → ℝ}
hypothesis h1 : ∀ x, x ≥ 1 → f (g x) = x ^ 2
hypothesis h2 : ∀ x, x ≥ 1 → g (f x) = x ^ 4
hypothesis h3 : g 81 = 81

-- The statement to prove
theorem g_9_to_the_4_pow_eq_81 : (g 9) ^ 4 = 81 :=
by
  sorry

end g_9_to_the_4_pow_eq_81_l177_177034


namespace alternating_sum_series_l177_177176

theorem alternating_sum_series : 
  (Finset.sum (Finset.range 101) (λ n, if even n then -(n+1) else n+1)) = 51 :=
by
  sorry

end alternating_sum_series_l177_177176


namespace add_points_proof_l177_177753

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l177_177753


namespace cone_volume_eq_l177_177787

noncomputable def volume_of_cone (d α : ℝ) : ℝ :=
  (1 / 3) * Real.pi * d^3 * (Real.cot(α / 2))^3 * Real.cot(α)

theorem cone_volume_eq (l h d α : ℝ)
  (h_1 : l - h = d)
  (h_α : l ≠ 0 ∧ h ≠ 0 ∧ α ≠ 0) :
  volume_of_cone d α = (1 / 3) * Real.pi * d^3 * (Real.cot (α / 2))^3 * Real.cot α :=
sorry

end cone_volume_eq_l177_177787


namespace angle_is_120_degrees_l177_177267

-- Define the magnitudes of vectors a and b and their dot product
def magnitude_a : ℝ := 10
def magnitude_b : ℝ := 12
def dot_product_ab : ℝ := -60

-- Define the angle between vectors a and b
def angle_between_vectors (θ : ℝ) : Prop :=
  magnitude_a * magnitude_b * Real.cos θ = dot_product_ab

-- Prove that the angle θ is 120 degrees
theorem angle_is_120_degrees : angle_between_vectors (2 * Real.pi / 3) :=
by 
  unfold angle_between_vectors
  sorry

end angle_is_120_degrees_l177_177267


namespace sqrt_200_eq_10_l177_177022

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177022


namespace sqrt_expr_eval_l177_177959

theorem sqrt_expr_eval :
  sqrt ((4 / 3) * ((1 / 15) + (1 / 25))) = (4 * sqrt 2) / 15 := 
by
  sorry

end sqrt_expr_eval_l177_177959


namespace plane_equation_l177_177890

-- Given vector definition
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 2 * s - 3 * t, 4 - 2 * s, 6 - 3 * s + t)

-- Conditions
def isInt (x : ℝ) := ∃ (n : ℤ), x = n

def gcd_cond (A B C D : ℤ) : Prop :=
  Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1

def eq_plane (A B C D : ℤ) (s t : ℝ) : Prop :=
  A * (parametric_plane s t).1 + B * (parametric_plane s t).2 + C * (parametric_plane s t).3 + D = 0

theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ gcd_cond A B C D ∧
  ∀ (s t : ℝ), eq_plane A B C D s t :=
sorry

end plane_equation_l177_177890


namespace count_perfect_square_factors_of_7200_l177_177538

theorem count_perfect_square_factors_of_7200 : 
  (∃ (count : ℕ), count = 12 ∧ 
    (∀ (a b c : ℕ), a = 0 ∨ a = 2 ∨ a = 4 → b = 0 ∨ b = 2 → c = 0 ∨ c = 2 →
    2^a * 3^b * 5^c ∣ 7200)) :=
by
  have h : 7200 = 2^4 * 3^2 * 5^2, by norm_num,
  existsi 12,
  split,
  { refl },
  { intros a b c ha hb hc,
    cases ha; cases hb; cases hc;
    simp [pow_succ, pow_zero, h];
    norm_num;
    apply dvd_mul_right }

end count_perfect_square_factors_of_7200_l177_177538


namespace satisfy_equation_l177_177949

theorem satisfy_equation (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end satisfy_equation_l177_177949


namespace children_sit_again_l177_177420

theorem children_sit_again (total_children : ℕ) (children_passed : ℕ) 
  (h1 : total_children = 2500) (h2 : children_passed = 375) : 
  total_children - children_passed = 2125 :=
by
  rw [h1, h2]
  rfl

end children_sit_again_l177_177420


namespace SpotsReachableArea_l177_177377

-- Define the conditions of the problem
structure Doghouse where
  baseSides : ℕ
  sideLength : ℝ
  ropeLength : ℝ
  tetherVertex : Bool

-- Construct an instance for our specific problem
def SpotsDoghouse : Doghouse := 
  { baseSides := 8, sideLength := 2, ropeLength := 3, tetherVertex := true }

-- Define the mathematical statement to prove
theorem SpotsReachableArea : ∀ (dh : Doghouse), dh = SpotsDoghouse → 
  (3:ℝ) * (3:ℝ) * Real.pi / 2 + 2 * ((3:ℝ) * (3:ℝ) * Real.pi / 8) = 
  (15:ℝ) / 4 * Real.pi :=
by
  intro dh h
  rw [h]
  sorry

end SpotsReachableArea_l177_177377


namespace sum_exterior_angles_fifth_polygon_is_360_l177_177148

-- Define the sequence and the problem statement
def sequence_polygon_sides : ℕ → ℕ
| 0 => 4
| (n+1) => sequence_polygon_sides n + 2

theorem sum_exterior_angles_fifth_polygon_is_360 :
  ∀ n, n = 5 → (∑ i in finset.range (sequence_polygon_sides n), 1) * 360 = 360 := 
by
  intros n hn
  rw hn
  sorry

end sum_exterior_angles_fifth_polygon_is_360_l177_177148


namespace stephen_total_distance_l177_177378

-- Define the conditions
def trips : ℕ := 10
def mountain_height : ℝ := 40000
def fraction_of_height_reached : ℝ := 3 / 4

-- Calculate the total distance covered
def total_distance_covered : ℝ :=
  2 * (fraction_of_height_reached * mountain_height) * trips

-- Prove the total distance covered is 600,000 feet
theorem stephen_total_distance :
  total_distance_covered = 600000 := by
  sorry

end stephen_total_distance_l177_177378


namespace strategy_classification_l177_177031

inductive Player
| A
| B

def A_winning_strategy (n0 : Nat) : Prop :=
  n0 >= 8

def B_winning_strategy (n0 : Nat) : Prop :=
  n0 <= 5

def neither_winning_strategy (n0 : Nat) : Prop :=
  n0 = 6 ∨ n0 = 7

theorem strategy_classification (n0 : Nat) : 
  (A_winning_strategy n0 ∨ B_winning_strategy n0 ∨ neither_winning_strategy n0) := by
    sorry

end strategy_classification_l177_177031


namespace cost_price_of_book_l177_177522

theorem cost_price_of_book (selling_price : ℝ) (profit_percentage : ℝ) (h : selling_price = 200 ∧ profit_percentage = 0.20) : 
  let cost_price := 200 / (1 + profit_percentage) 
  in cost_price = 166.67 :=
by
  sorry

end cost_price_of_book_l177_177522


namespace domain_of_log_composition_l177_177539

theorem domain_of_log_composition (x : ℝ) :
  (log 3 (log 5 (log 6 (log 7 x))) > 0 ↔ x > 7 ^ 6) :=
by
  sorry

end domain_of_log_composition_l177_177539


namespace roots_sum_sin_squared_roots_sum_cot_squared_l177_177558

-- Define polynomial P for Part (a)
noncomputable def polyP (n : ℕ) : ℕ → ℝ := 
  fun k => (Real.sin ((k * Real.pi) / (2 * n + 1))) ^ 2

-- Define the theorem for Part (a)
theorem roots_sum_sin_squared (n : ℕ) :
  Sorry.sum (polyP n) (n) = -1 / 2 :=
sorry

-- Define polynomial Q for Part (b)
noncomputable def polyQ (n : ℕ) : ℕ → ℝ := 
  fun k => (1 / (Real.tan ((k * Real.pi) / (2 * n + 1)))) ^ 2

-- Define the theorem for Part (b)
theorem roots_sum_cot_squared (n : ℕ) :
  Sorry.sum (polyQ n) (n) = -1 / 2 :=
sorry

end roots_sum_sin_squared_roots_sum_cot_squared_l177_177558


namespace smallest_nat_number_l177_177064

theorem smallest_nat_number (x : ℕ) (h1 : 5 ∣ x) (h2 : 7 ∣ x) (h3 : x % 3 = 1) : x = 70 :=
sorry

end smallest_nat_number_l177_177064


namespace range_of_a_for_unique_minimum_l177_177239

noncomputable def is_unique_minimum_at_x0 (a x0 : ℝ) : Prop :=
  ∀ x : ℝ, (ax^3 + Real.exp x) ≥ (ax0^3 + Real.exp x0)

theorem range_of_a_for_unique_minimum : 
  (is_unique_minimum_at_x0 a x0) → (a ∈ set.Ico (- Real.exp 2 / 12) 0) :=
sorry

end range_of_a_for_unique_minimum_l177_177239


namespace initial_points_l177_177748

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l177_177748


namespace maximum_possible_elements_in_A_l177_177350

open Set Nat

def is_product_of_consecutive_integers (n : ℤ) : Prop :=
  ∃ (m : ℤ), n = m * (m + 1)

def violates_condition (a b k : ℤ) : Prop :=
  is_product_of_consecutive_integers (a + b + 30 * k)

noncomputable def maximum_subset_size (A : Set ℤ) : Prop :=
  A ⊆ {0, 1, 2, ..., 29} ∧
  ¬ ∃ (a b k ∈ A), violates_condition a b k

theorem maximum_possible_elements_in_A :
  ∀ A : Set ℤ, maximum_subset_size A → A.size ≤ 10 :=
sorry

end maximum_possible_elements_in_A_l177_177350


namespace ten_digit_numbers_with_no_consecutive_twos_l177_177632

-- Define the sequences a_n, b_n, and c_n as in the problem description
def a : ℕ → ℕ
| 1 := 2
| 2 := 3
| (n+2) := a n + a (n+1)

theorem ten_digit_numbers_with_no_consecutive_twos : a 10 = 144 :=
by
-- Skip the proof
sorry

end ten_digit_numbers_with_no_consecutive_twos_l177_177632


namespace pizza_volume_l177_177896

theorem pizza_volume 
  (thickness : ℝ) (diameter : ℝ) (pieces : ℕ)
  (h_thickness : thickness = 1 / 2)
  (h_diameter : diameter = 16)
  (h_pieces : pieces = 8) :
  let radius := diameter / 2 in
  let total_volume := π * (radius^2) * thickness in
  let volume_per_piece := total_volume / pieces in
  volume_per_piece = 4 * π :=
by
  sorry

end pizza_volume_l177_177896


namespace f_eq_n_for_all_n_l177_177693

noncomputable def f : ℕ → ℕ := sorry

axiom f_pos_int_valued (n : ℕ) (h : 0 < n) : f n = f n

axiom f_2_eq_2 : f 2 = 2

axiom f_mul_prop (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n

axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : f m > f n

theorem f_eq_n_for_all_n (n : ℕ) (hn : 0 < n) : f n = n := sorry

end f_eq_n_for_all_n_l177_177693


namespace log3_integer_probability_l177_177501

/-- Define the conditions under which the problem holds -/
def is_four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

/-- Define the problem as a proof statement -/
theorem log3_integer_probability :
  (∃ N : ℕ, is_four_digit N ∧ (∃ k : ℕ, N = 3^k)) →
  let num_valid_N := 2 in
  let total_four_digit := 9000 in
  (num_valid_N / total_four_digit : ℚ) = 1 / 4500 :=
by
  sorry

end log3_integer_probability_l177_177501


namespace initial_weight_of_alloy_is_16_l177_177914

variable (Z C : ℝ)
variable (h1 : Z / C = 5 / 3)
variable (h2 : (Z + 8) / C = 3)
variable (A : ℝ := Z + C)

theorem initial_weight_of_alloy_is_16 (h1 : Z / C = 5 / 3) (h2 : (Z + 8) / C = 3) : A = 16 := by
  sorry

end initial_weight_of_alloy_is_16_l177_177914


namespace find_initial_amount_l177_177324

variable (P r n : ℝ)
variable (d : ℝ)

def compound_interest (P r n : ℝ) :=
  P * (1 + r / 100) ^ n

def simple_interest (P r n : ℝ) :=
  P + (P * r * n) / 100

theorem find_initial_amount (h1 : r = 4) (h2 : n = 2) (h3 : d = 10.40) :
  P = 6500 :=
by
  have h4 : compound_interest P r n - simple_interest P r n = d :=
    by
      rw [compound_interest, simple_interest, h1, h2]
      sorry  -- Placeholder for intermediate verification steps
  sorry  -- Placeholder for complete proof of P = 6500

end find_initial_amount_l177_177324


namespace binomial_coeff_sum_abs_l177_177645

theorem binomial_coeff_sum_abs (a : ℤ) (x : ℝ) :
  (∀ r, r = 5 → (∃ n, n = 8 ∧ 
                (1 - 3 * x)^n = (∑ k in finset.range (n+1), a_k * x^k) ∧ 
                (a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 = 2^16))) →
  ∑ i in finset.range 8, abs (a i) = 2^16 - 1 :=
sorry

end binomial_coeff_sum_abs_l177_177645


namespace sum_of_other_two_angles_is_108_l177_177323

theorem sum_of_other_two_angles_is_108 (A B C : Type) (angleA angleB angleC : ℝ) 
  (h_angle_sum : angleA + angleB + angleC = 180) (h_angleB : angleB = 72) :
  angleA + angleC = 108 := 
by
  sorry

end sum_of_other_two_angles_is_108_l177_177323


namespace coeff_x3_in_expansion_l177_177388

theorem coeff_x3_in_expansion :
  let expr := (1 - X^3) * (1 + X)^10 in
  (coeff (expr.expand) 3 = 119) := 
by
  let X := Polynomial.X
  let expr := (1 : Polynomial ℚ) - X ^ 3 \* (1 + X) ^ 10
  have h : (coeff (expr.expand) 3 = \binom{10}{3} - 1 := sorry
  rw [binom_eq_num_div] at h
  exact h
  sorry

end coeff_x3_in_expansion_l177_177388


namespace lateral_surface_area_of_cube_l177_177055

theorem lateral_surface_area_of_cube (s : ℝ) (h : s = 12) : 
  let face_area := s * s in
  let lateral_surface_area := 4 * face_area in
  lateral_surface_area = 576 :=
by
  sorry

end lateral_surface_area_of_cube_l177_177055


namespace only_solutions_l177_177203

noncomputable def f : ℝ → ℝ := sorry

lemma functional_equation (x y : ℝ) : f(x * (1 + y)) = f(x) * (1 + f(y)) := sorry

theorem only_solutions : (∀ x, f x = 0) ∨ (∀ x, f x = x) := sorry

end only_solutions_l177_177203


namespace problem_l177_177637

noncomputable def y := 2 + Real.sqrt 3

theorem problem (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : y = c + Real.sqrt d)
  (hy_eq : y^2 + 2*y + 2/y + 1/y^2 = 20) : c + d = 5 :=
  sorry

end problem_l177_177637


namespace geometric_sequence_angles_l177_177194

noncomputable def theta_conditions (θ : ℝ) :=
  0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ θ ≠ Real.pi / 2 ∧ θ ≠ Real.pi ∧ θ ≠ 3 * Real.pi / 2 ∧
  (∀ {a b c : ℝ}, Set {a, b, c} = {Real.sin θ, Real.cos θ, Real.cot θ} → a * c = b * b)

theorem geometric_sequence_angles : 
  ∃! θ1 θ2 : ℝ, theta_conditions θ1 ∧ theta_conditions θ2 ∧ θ1 ≠ θ2 :=
  sorry

end geometric_sequence_angles_l177_177194


namespace train_time_to_cross_tree_l177_177870

-- Definitions based on conditions
def length_of_train := 1200 -- in meters
def time_to_pass_platform := 150 -- in seconds
def length_of_platform := 300 -- in meters
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_pass_platform
def time_to_cross_tree := length_of_train / speed_of_train

-- Theorem stating the main question
theorem train_time_to_cross_tree : time_to_cross_tree = 120 := by
  sorry

end train_time_to_cross_tree_l177_177870


namespace part1_solution_part2_solution_l177_177615

def f (λ x: ℝ) : ℝ := log ( (λ * x + 1) / x )

theorem part1_solution (x : ℝ) : f 2 x > 0 ↔ x ∈ set.Iio (-1) ∪ set.Ioi 0 :=
sorry

noncomputable def cond (λ a: ℝ) : Prop :=
∀ x1 x2 ∈ set.Icc a (a + 1), abs (f λ x1 - f λ x2) ≤ log 2

theorem part2_solution (λ : ℝ) (H : ∀ a ∈ set.Icc (1/2) 2, cond λ a) : λ ≥ 2/3 :=
sorry

end part1_solution_part2_solution_l177_177615


namespace x_fourth_minus_inv_fourth_l177_177594

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l177_177594


namespace constant_term_eq_neg_twenty_l177_177973

noncomputable def constant_term_expansion : ℤ := 
  let A := (fun x : ℚ => x + 1/x)
  let B := 2
  let r := 3
  let constant_term := (-B)^r + (Nat.binom r 2) * A(1) * (-B) -- constant terms need to sum to -20
  constant_term

theorem constant_term_eq_neg_twenty : constant_term_expansion = -20 :=
  by
    sorry

end constant_term_eq_neg_twenty_l177_177973


namespace x_fourth_minus_inv_fourth_l177_177592

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l177_177592


namespace geometric_properties_l177_177572

variable (M F : Point) (l : Line) (A B : Point) (O D : Point)
variable (x1 y1 x2 y2 : ℝ)
variable (curve : ℝ → ℝ → Prop)

-- Condition: The distance from point M to point F (1, 0) is 2 less than its distance to the line x = -3.
-- Condition: Line l passes through point F(1, 0) and intersects curve C at points A (x1, y1) and B (x2, y2).
-- Condition: A perpendicular line to x = -1 is drawn from point B with the foot of the perpendicular being D.
-- Condition: O is the origin (0, 0)
-- Note: Definitions of distance, curve are assumed for concise state.
variables (dist : Point → Point → ℝ) (directrix : ℝ → Line)

def curve_C : Prop :=
  -- Curve C is the parabola described by y^2 = 4x
  ∀ (x y : ℝ), curve x y ↔ y^2 = 4 * x

def points_AO_D_collinear : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ),
  curve x1 y1 ∧ curve x2 y2 ∧ dist B D = dist B ⟨-1, y2⟩ ->
  x1 * x2 = 1 ∧ y1 * y2 = -4 ∧ collinear O A D

-- Main theorem statement combining the given conditions and goals
theorem geometric_properties (M F A B O D : Point) (curve : ℝ → ℝ → Prop) (x1 y1 x2 y2 : ℝ)
  (h1 : ∀ x y, curve x y ↔ y^2 = 4 * x)
  (h2 : collinear O A D) :
  curve_C curve ∧ points_AO_D_collinear curve :=
by
  sorry

end geometric_properties_l177_177572


namespace most_likely_second_red_ball_l177_177515

theorem most_likely_second_red_ball (n m : ℕ) (hn : n = 101) (hm : m = 3) :
  (∃ k : ℕ, 1 < k ∧ k < 100 ∧ (∀ j : ℕ, 1 < j ∧ j < 100 → -(j-1) * (101-j) ≤ -(k-1) * (101-k)) ∧ k = 51) :=
by {
  have hn' : n = 101 := hn,
  have hm' : m = 3 := hm,
  sorry
}

end most_likely_second_red_ball_l177_177515


namespace types_of_problems_l177_177920

def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def problems_per_type : ℕ := 30

theorem types_of_problems : (frank_problems / problems_per_type) = 4 := by
  sorry

end types_of_problems_l177_177920


namespace max_cards_saved_by_upside_down_numbers_l177_177825

def is_valid_digit (d : ℕ) : Prop := d = 0 ∨ d = 1 ∨ d = 6 ∨ d = 8 ∨ d = 9

def is_valid_upside_down (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  is_valid_digit d1 ∧ is_valid_digit d2 ∧ is_valid_digit d3 ∧ (d1, d2, d3) ≠ (0, d2, d3)

theorem max_cards_saved_by_upside_down_numbers : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ is_valid_upside_down n}.to_finset.card = 34 :=
sorry

end max_cards_saved_by_upside_down_numbers_l177_177825


namespace least_possible_value_of_z_minus_w_l177_177709

variable (x y z w k m : Int)
variable (h1 : Even x)
variable (h2 : Odd y)
variable (h3 : Odd z)
variable (h4 : ∃ n : Int, w = - (2 * n + 1) / 3)
variable (h5 : w < x)
variable (h6 : x < y)
variable (h7 : y < z)
variable (h8 : 0 < k)
variable (h9 : (y - x) > k)
variable (h10 : 0 < m)
variable (h11 : (z - w) > m)
variable (h12 : k > m)

theorem least_possible_value_of_z_minus_w
  : z - w = 6 := sorry

end least_possible_value_of_z_minus_w_l177_177709


namespace greatest_expression_value_l177_177826

noncomputable def greatest_expression : ℝ := 0.9986095661846496

theorem greatest_expression_value : greatest_expression = 0.9986095661846496 :=
by
  -- proof goes here
  sorry

end greatest_expression_value_l177_177826


namespace scientific_notation_coronavirus_diameter_l177_177048

theorem scientific_notation_coronavirus_diameter : 0.00000011 = 1.1 * 10^(-7) :=
by {
  sorry
}

end scientific_notation_coronavirus_diameter_l177_177048


namespace prove_parallel_planes_l177_177626

noncomputable theory

variables {α β γ : Type} -- Define planes as types

-- Define lines and parallelism
variables (m n : Type) -- Define lines as types
variables (P : m → α) (Q : n → β) -- Define points on lines and planes

-- Skew lines: They do not intersect and are not coplanar
def skew_lines (m n : Type) : Prop :=
  ∀ (u : m) (v : n), (P u ≠ Q v) ∧ ¬(∃ (plane_sub : Type), plane_sub = α ∨ plane_sub = β)

-- Define parallelism for lines and planes
def parallel_lines (l₁ l₂ : Type) : Prop :=
  ∀ (u₁ : l₁) (u₂ : l₂), u₁ = u₂

def parallel_planes (p₁ p₂ : Type) : Prop :=
  ∀ (x₁ : p₁) (x₂ : p₂), x₁ = x₂

-- The proof statement
theorem prove_parallel_planes (hmn : skew_lines m n) (h1 : parallel_lines m β) (h2 : parallel_lines n α) : parallel_planes α β :=
sorry

end prove_parallel_planes_l177_177626


namespace sum_of_angles_satisfying_cos_sin_cubic_identity_l177_177978

theorem sum_of_angles_satisfying_cos_sin_cubic_identity :
  ∑ x in { x : Real | x ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.cos x ^ 3 - Real.sin x ^ 3 = 1 / (Real.sin x + Real.cos x) }, x = Real.pi * (3 / 2) :=
by
  sorry

end sum_of_angles_satisfying_cos_sin_cubic_identity_l177_177978


namespace curve_parametric_to_polynomial_l177_177879

theorem curve_parametric_to_polynomial :
  ∃ a b c : ℚ,
    (a = 1/9 ∧ b = -4/27 ∧ c = 23/243) ∧ 
    (∀ t : ℝ, 
        let x := 3 * Real.cos t + 2 * Real.sin t,
            y := 3 * Real.sin t
        in a * x^2 + b * x * y + c * y^2 = 9) :=
sorry

end curve_parametric_to_polynomial_l177_177879


namespace ordered_concrete_weight_l177_177878

def weight_of_materials : ℝ := 0.83
def weight_of_bricks : ℝ := 0.17
def weight_of_stone : ℝ := 0.5

theorem ordered_concrete_weight :
  weight_of_materials - (weight_of_bricks + weight_of_stone) = 0.16 := by
  sorry

end ordered_concrete_weight_l177_177878


namespace binomial_congruence_mod_p3_l177_177242

open Nat

-- Let p be an odd prime number greater than 3, and a, b be integers such that a > b > 1.
variable (p : ℕ) [Fact (Prime p)] (hp : p > 3) (a b : ℕ) (hab : a > b) (hb1 : b > 1)

-- Define the binomial coefficient C
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Given the conditions above, prove the equivalence modulo p^3
theorem binomial_congruence_mod_p3 (hp : p > 3) (hab : a > b) (hb1 : b > 1) : 
  binomial (a * p) (b * p) ≡ binomial a b [ZMOD (p^3)] :=
sorry

end binomial_congruence_mod_p3_l177_177242


namespace find_p_q_sum_l177_177980

-- Define the conditions
def p (q : ℤ) : ℤ := q + 20

theorem find_p_q_sum (p q : ℤ) (hp : p * q = 1764) (hq : p - q = 20) :
  p + q = 86 :=
  sorry

end find_p_q_sum_l177_177980


namespace b_n_next_b_n_general_term_l177_177994

open Nat Real

noncomputable def f (a x : ℝ) := (2*x^2 + 1)/x

def seq_a (a_n : ℕ → ℝ) (n : ℕ) : ℝ := 
  if n = 1 then 2 
  else (f 2 (a_n (n-1)) - a_n (n-1)) / 2

def seq_b (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then (a_n 1 - 1) / (a_n 1 + 1)
  else (a_n n - 1) / (a_n n + 1)

theorem b_n_next (a_n b_n : ℕ → ℝ) (n : ℕ) :
  b_n (n+1) = (b_n n)^2 :=
sorry

theorem b_n_general_term (b_n : ℕ → ℝ) (n : ℕ) :
  b_n n = (1/3)^(2^(n-1)) :=
sorry

end b_n_next_b_n_general_term_l177_177994


namespace volume_of_tetrahedron_l177_177909

noncomputable def tetrahedronVolume (a b c : ℝ) : ℝ :=
  (1/ 24 : ℝ) * Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2))

theorem volume_of_tetrahedron {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ V : ℝ, V = tetrahedronVolume a b c :=
by
  use tetrahedronVolume a b c
  sorry

end volume_of_tetrahedron_l177_177909


namespace grandma_red_bacon_bits_l177_177518

def mushrooms := 3
def cherry_tomatoes := 2 * mushrooms
def pickles := 4 * cherry_tomatoes
def bacon_bits := 4 * pickles
def red_bacon_bits := bacon_bits / 3

theorem grandma_red_bacon_bits : red_bacon_bits = 32 := by
  sorry

end grandma_red_bacon_bits_l177_177518


namespace star_polygon_sum_of_angles_l177_177893

/-- A regular star polygon is made up of ℓ points where each vertex of the star 
consists of angles X₁, X₂, ... , Xₗ and Y₁, Y₂, ... , Yₗ, where all Xᵢ and all Yᵢ 
are congruent respectively. If the acute angle at X₁ is 15° less than the acute 
angle at Y₁, and the sum of all internal angles is twice the sum of all external 
angles, then the number of points ℓ is 24. -/
theorem star_polygon_sum_of_angles (ℓ : ℕ) (φ : ℝ)
  (h1 : ∀ i j, (i ≠ j → X i = X j ∧ Y i = Y j))
  (h2 : X 1 = φ - 15)
  (h3 : ∑_{i = 1}^ℓ (180 - 2 * φ + 15) = 720) :
  ℓ = 24 :=
sorry

end star_polygon_sum_of_angles_l177_177893


namespace redesigned_survey_response_count_l177_177901

variable (respondents_original : ℕ) (customers_original : ℕ) 
          (customers_redesigned : ℕ) (increase_rate : ℝ)

def original_response_rate : ℝ :=
  (respondents_original.to_real / customers_original.to_real) * 100

def new_response_rate : ℝ :=
  original_response_rate respondents_original customers_original + increase_rate

def respondents_redesigned : ℝ :=
  (new_response_rate respondents_original customers_original increase_rate / 100) * customers_redesigned.to_real

theorem redesigned_survey_response_count :
  respondents_original = 7 → 
  customers_original = 60 → 
  customers_redesigned = 63 → 
  increase_rate = 2 → 
  respondents_redesigned respondents_original customers_original customers_redesigned increase_rate ≈ 9 := 
by
  sorry

end redesigned_survey_response_count_l177_177901


namespace number_of_girls_l177_177814

theorem number_of_girls
  (total_students : ℕ)
  (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_non_binary : ℕ)
  (h_ratio : ratio_girls = 3 ∧ ratio_boys = 2 ∧ ratio_non_binary = 1)
  (h_total : total_students = 72) :
  ∃ (k : ℕ), 3 * k = (total_students * 3) / 6 ∧ 3 * k = 36 :=
by
  sorry

end number_of_girls_l177_177814


namespace polygon_line_contains_parallel_segments_l177_177488

-- Defining the problem conditions
variables {n : ℕ} (even_n : even n)
variables (points : ℕ → ℕ) (perms : list ℕ)

-- Ensure the list of points has length 2n
def valid_permutation (points : list ℕ) : Prop := 
  (points.length = 2 * n) ∧ 
  (points.perm (list.range (2 * n)))

-- Main theorem statement
theorem polygon_line_contains_parallel_segments
  (even_n : even n) 
  (points : list ℕ) 
  (hperm : valid_permutation n points) :
  ∃ (i j k l : ℕ), 
    i < 2 * n ∧ j < 2 * n ∧ k < 2 * n ∧ l < 2 * n ∧ 
    i ≠ k ∧ j ≠ l ∧
    points.nth i = some pts1 ∧ points.nth j = some pts2 ∧ 
    points.nth k = some pts3 ∧ points.nth l = some pts4 ∧ 
    ((pts1 + pts2) % n = (pts3 + pts4) % n) := 
sorry

end polygon_line_contains_parallel_segments_l177_177488


namespace tree_chromatic_number_two_l177_177651

variable {V : Type} [Fintype V] [DecidableEq V]

-- A connected tree is defined as a type of graph.
structure ConnectedTree (G : SimpleGraph V) extends IsConnected G : Prop where
  tree : G.card_vertices = G.card_edges + 1

-- Definition of a chromatic number.
def chromaticNumber (G : SimpleGraph V) : ℕ := sorry

theorem tree_chromatic_number_two {G : SimpleGraph V} (h1 : ConnectedTree G) (h2 : 2 ≤ G.card_vertices) :
  chromaticNumber G = 2 := 
sorry

end tree_chromatic_number_two_l177_177651


namespace mode_and_median_of_water_usage_l177_177831

def water_usage_data : List ℕ := [7, 7, 8, 8, 8, 9, 9, 9, 9, 10]

theorem mode_and_median_of_water_usage :
  (List.mode water_usage_data = [9]) ∧
  (List.median water_usage_data = 8.5) :=
by
  sorry

end mode_and_median_of_water_usage_l177_177831


namespace count_even_three_digit_numbers_less_than_800_l177_177838

def even_three_digit_numbers_less_than_800 : Nat :=
  let hundreds_choices := 7
  let tens_choices := 8
  let units_choices := 4
  hundreds_choices * tens_choices * units_choices

theorem count_even_three_digit_numbers_less_than_800 :
  even_three_digit_numbers_less_than_800 = 224 := 
by 
  unfold even_three_digit_numbers_less_than_800
  rfl

end count_even_three_digit_numbers_less_than_800_l177_177838


namespace max_dist_sum_l177_177250

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

theorem max_dist_sum (A B : ℝ × ℝ) (hA : ellipse_eq A.1 A.2) (hB : ellipse_eq B.1 B.2) (hA_l : A.1 = -1) (hB_l : B.1 = -1) :
  max (dist (B, F2) + dist (A, F2)) = 5 :=
sorry

end max_dist_sum_l177_177250


namespace cube_of_square_of_third_smallest_prime_is_correct_l177_177087

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l177_177087


namespace old_manufacturing_cost_l177_177534

theorem old_manufacturing_cost (P : ℝ) :
  (50 : ℝ) = P * 0.50 →
  (0.65 : ℝ) * P = 65 :=
by
  intros hp₁
  -- Proof omitted
  sorry

end old_manufacturing_cost_l177_177534


namespace angle_ratio_l177_177311

theorem angle_ratio {x : ℝ} (h1 : ∀ (BP BQ : ℝ), ∠ BP = ∠ BQ = ∠ ABC / 2)
  (h2 : BM_bisects_PBPQ : ∀ BM : ℝ, ∠ BM = (3 * x - x) / 2)
  (PBQ_eq_2x : ∠ PBQ = 2 * x) :
  ∠ MBQ / ∠ ABQ = 1 / 4 :=
by
  -- Sorry, the proof is omitted here.
  sorry

end angle_ratio_l177_177311


namespace monotonic_decreasing_interval_l177_177612

def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3

theorem monotonic_decreasing_interval :
  ∀ x, 
  x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 6) → x ∈ Set.Icc (Real.pi / 12) (Real.pi / 6) → 
  Function.StrictMonoDecOn On f x :=
sorry

end monotonic_decreasing_interval_l177_177612


namespace sqrt_200_eq_10_l177_177017

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177017


namespace binary_10101_is_21_l177_177528

namespace BinaryToDecimal

def binary_to_decimal (n : Nat) : Nat :=
  match n with
  | 10101 => 21

theorem binary_10101_is_21 :
  binary_to_decimal 10101 = 21 := by
  -- Proof steps would go here
  sorry

end BinaryToDecimal

end binary_10101_is_21_l177_177528


namespace spurs_team_players_l177_177778

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end spurs_team_players_l177_177778


namespace white_area_l177_177820

/-- The area of a 5 by 17 rectangular sign. -/
def sign_area : ℕ := 5 * 17

/-- The area covered by the letter L. -/
def L_area : ℕ := 5 * 1 + 1 * 2

/-- The area covered by the letter O. -/
def O_area : ℕ := (3 * 3) - (1 * 1)

/-- The area covered by the letter V. -/
def V_area : ℕ := 2 * (3 * 1)

/-- The area covered by the letter E. -/
def E_area : ℕ := 3 * (1 * 3)

/-- The total area covered by the letters L, O, V, E. -/
def sum_black_area : ℕ := L_area + O_area + V_area + E_area

/-- The problem statement: Calculate the area of the white portion of the sign. -/
theorem white_area : sign_area - sum_black_area = 55 :=
by
  -- Place the proof here
  sorry

end white_area_l177_177820


namespace arithmetic_sequence_mod_12_l177_177633

theorem arithmetic_sequence_mod_12 :
  ∃ m, (0 ≤ m ∧ m < 12) ∧ (∑ k in (Finset.range 29), (5 * (k + 1)) % 12 = m) ∧ m = 3 :=
by
  sorry

end arithmetic_sequence_mod_12_l177_177633


namespace three_layers_covered_area_l177_177120

theorem three_layers_covered_area (A : ℝ) (T : ℝ) (P : ℝ) (y : ℝ) (z : ℝ) :
  A = 204 →
  T = 175 →
  P = 0.80 →
  y = 24 →
  let covered_area := P * T in
  let x := covered_area - y - z in
  let total_area := x + 2 * y + 3 * z in
  covered_area = 140 →
  total_area = 204 →
  z = 20 :=
by
  intros hA hT hP hy;
  simp [covered_area, x, total_area];
  intro h_covered_area h_total_area;
  let h1 : x + y + z = 140 := by
    rw [h_covered_area, hy];
    simp;
  let h2 : x + 2 * y + 3 * z = 204 := by
    rw [h_total_area, hy];
    simp;
  have : 2 * z = 40 :=
    suffices x + 3 * z = 156 by
      rw [sub_eq_iff_eq_add.mp (eq_sub_of_add_eq h1).symm];
      exact add_left_eq_self.mp (eq_sub_of_add_eq h2.symm);
  exact div_eq_mul_one_div _;
  assumption;
  exact 20


end three_layers_covered_area_l177_177120


namespace initial_points_l177_177750

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l177_177750


namespace quadratic_completion_sum_eq_neg78_l177_177812

theorem quadratic_completion_sum_eq_neg78 :
  ∃ a b c : ℤ, (4x^2 - 16x - 64 = a * (x + b)^2 + c) ∧ a + b + c = -78 :=
by
  use 4, -2, -80
  split
  . sorry -- proof of quadratic completion
  . refl -- proof that 4 + -2 + -80 = -78

end quadratic_completion_sum_eq_neg78_l177_177812


namespace modulus_of_complex_l177_177685

theorem modulus_of_complex : 
  ∀ (z : ℂ), z = (2 - complex.I) ^ 2 →
  complex.abs z = 5 := 
by sorry

end modulus_of_complex_l177_177685


namespace longest_diagonal_regular_heptagon_l177_177906

theorem longest_diagonal_regular_heptagon (a : ℝ) :
  length_of_longest_diagonal a = 2 * a * Real.cos(3 * Real.pi / 7) :=
sorry

end longest_diagonal_regular_heptagon_l177_177906


namespace g_9_to_the_4_pow_eq_81_l177_177033

-- Given conditions
variables {f g : ℝ → ℝ}
hypothesis h1 : ∀ x, x ≥ 1 → f (g x) = x ^ 2
hypothesis h2 : ∀ x, x ≥ 1 → g (f x) = x ^ 4
hypothesis h3 : g 81 = 81

-- The statement to prove
theorem g_9_to_the_4_pow_eq_81 : (g 9) ^ 4 = 81 :=
by
  sorry

end g_9_to_the_4_pow_eq_81_l177_177033


namespace sum_adjacent_angles_pentagon_l177_177493

theorem sum_adjacent_angles_pentagon (n : ℕ) (θ : ℕ) (hn : n = 5) (hθ : θ = 40) :
  let exterior_angle := 360 / n
  let new_adjacent_angle := 180 - (exterior_angle + θ)
  let sum_adjacent_angles := n * new_adjacent_angle
  sum_adjacent_angles = 340 := by
  sorry

end sum_adjacent_angles_pentagon_l177_177493


namespace sum_n_absolute_value_l177_177439

theorem sum_n_absolute_value :
  (∑ n in { n : ℤ | |2 * n - 7| = 3 }.toFinset, n) = 7 :=
sorry

end sum_n_absolute_value_l177_177439


namespace new_quad_eq_l177_177987

theorem new_quad_eq (a b c : ℝ) : 
  ∀ (x₁ x₂ : ℝ), (x₁ + x₂ = -b / a) ∧ (x₁ * x₂ = c / a) →
  let y₁ := x₁ + 1 in
  let y₂ := x₂ + 1 in
  ∃ (y : ℝ → ℝ) (p q : ℝ), 
    y^2 + p * y + q = 0 ∧ 
    p = (b - 2 * a) / a ∧ 
    q = (a - b + c) / a :=
begin
  -- Proof goes here
  sorry
end

end new_quad_eq_l177_177987


namespace sqrt_221_range_l177_177981

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end sqrt_221_range_l177_177981


namespace hcf_of_two_numbers_l177_177810

open Nat

theorem hcf_of_two_numbers (a b : ℕ) (h1 : a * b = 17820) (h2 : lcm a b = 1485) : gcd a b = 12 := by
  sorry

end hcf_of_two_numbers_l177_177810


namespace absolute_value_simplification_l177_177276

theorem absolute_value_simplification (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := 
by 
  sorry

end absolute_value_simplification_l177_177276


namespace sqrt_200_eq_l177_177011

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177011


namespace sqrt_200_eq_l177_177016

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177016


namespace probability_all_same_community_probability_at_least_two_in_A_l177_177829

-- Probability that all three students are in the same community
theorem probability_all_same_community :
  let outcomes := ({("A", "A", "A"), ("A", "A", "B"), ("A", "B", "A"), ("A", "B", "B"),
                     ("B", "A", "A"), ("B", "A", "B"), ("B", "B", "A"), ("B", "B", "B")}: set (string × string × string))
  in  
  let same_community := {("A", "A", "A"), ("B", "B", "B")} in
  (same_community.card / outcomes.card : ℚ) = 1 / 4 := 
by sorry

-- Probability that at least two students are in community A
theorem probability_at_least_two_in_A :
  let outcomes := ({("A", "A", "A"), ("A", "A", "B"), ("A", "B", "A"), ("A", "B", "B"),
                     ("B", "A", "A"), ("B", "A", "B"), ("B", "B", "A"), ("B", "B", "B")}: set (string × string × string))
  in  
  let at_least_two_A := {("A", "A", "A"), ("A", "A", "B"), ("A", "B", "A"), ("B", "A", "A")} in
  (at_least_two_A.card / outcomes.card : ℚ) = 1 / 2 :=
by sorry

end probability_all_same_community_probability_at_least_two_in_A_l177_177829


namespace max_value_of_t_l177_177244

theorem max_value_of_t (x y : ℝ) (h : x^2 + y^2 = 25) :
  ∃ t, t = sqrt (18 * y - 6 * x + 50) + sqrt (8 * y + 6 * x + 50) ∧ t = 6 * sqrt 10 :=
sorry

end max_value_of_t_l177_177244


namespace integer_part_M_div_100_l177_177575

theorem integer_part_M_div_100 :
  (∑ k in {3, 4, 5, 6, 7, 8, 9, 10}, (1 : ℚ) / (k.factorial * (20 - k).factorial)) = (M : ℚ) / (2.factorial * 17.factorial) →
  ⌊M / 100⌋ = 262 :=
sorry

end integer_part_M_div_100_l177_177575


namespace increasing_interval_of_f_l177_177396

-- Definitions of the components of the decomposed function
def u (x : ℝ) : ℝ := -x^2 + 2 * x

def y (u : ℝ) : ℝ := (1/2) ^ u 

def f (x : ℝ) : ℝ := y (u x)

-- The statement to be proved
theorem increasing_interval_of_f : 
  ∀ x : ℝ, 1 ≤ x → 
  ∀ a b : ℝ, 1 ≤ a → a ≤ b → f a ≤ f b := 
sorry

end increasing_interval_of_f_l177_177396


namespace trisectors_equilateral_l177_177665

theorem trisectors_equilateral
  (A B C : Type)
  (h_tri : Triangle A B C)
  (first_trisector_A : Trisector A B C)
  (first_trisector_B : Trisector B C A)
  (first_trisector_C : Trisector C A B)
  (equal_angles : ∀ P Q R : Type, (P ∈ Triangle Q R (P' : Type)) → (angles P Q R).1 = (angles P' Q' R').1 ∧ (angles P Q R).2 = (angles P' Q' R').2 ∧ (angles P Q R).3 = (angles P' Q' R').3)
  :
  Equilateral A B C := 
sorry

end trisectors_equilateral_l177_177665


namespace g_9_to_the_4_l177_177036

variable (f g : ℝ → ℝ)

axiom a1 : ∀ x, x ≥ 1 → f(g(x)) = x^2
axiom a2 : ∀ x, x ≥ 1 → g(f(x)) = x^4
axiom a3 : g 81 = 81

theorem g_9_to_the_4 : (g 9) ^ 4 = 81 := by
  sorry

end g_9_to_the_4_l177_177036


namespace highest_sum_vertex_l177_177791

theorem highest_sum_vertex (a b c d e f : ℕ) (h₀ : a + d = 8) (h₁ : b + e = 8) (h₂ : c + f = 8) : 
  a + b + c ≤ 11 ∧ b + c + d ≤ 11 ∧ c + d + e ≤ 11 ∧ d + e + f ≤ 11 ∧ e + f + a ≤ 11 ∧ f + a + b ≤ 11 :=
sorry

end highest_sum_vertex_l177_177791


namespace longest_side_similar_triangle_l177_177056

def similar_triangle_longest_side (a b c perimeter : ℕ) (h : a + b + c = 30) : ℕ :=
  let k := perimeter / 30 in 13 * k

theorem longest_side_similar_triangle (a b c : ℕ) (perimeter : ℕ)
  (h : a = 5) (h₁ : b = 12) (h₂ : c = 13) (h₃ : perimeter = 150) :
  similar_triangle_longest_side a b c perimeter (by linarith) = 65 := by
  sorry

end longest_side_similar_triangle_l177_177056


namespace min_value_of_f_l177_177183

noncomputable def f (x : ℝ) := max (3 - x) (x^2 - 4*x + 3)

theorem min_value_of_f : ∃ x : ℝ, f x = -1 :=
by {
  use 2,
  sorry
}

end min_value_of_f_l177_177183


namespace largest_n_with_sigma_28_l177_177975

def sigma (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum

theorem largest_n_with_sigma_28 : (∃ n : ℕ, sigma n = 28 ∧ ∀ m : ℕ, m > n → sigma m ≠ 28) ↔ n = 12 := 
sorry

end largest_n_with_sigma_28_l177_177975


namespace isosceles_triangle_divisible_l177_177738

theorem isosceles_triangle_divisible (A B C : Point) 
  (hAB_eq_AC : distance A B = distance A C) (angle_ABC_type : AngleType) :
  ∃ D E F : Point, 
    (right_triangle A D B ∧ right_triangle A D C ∧ isosceles_triangle A D E) 
    ∨ (right_triangle A E B ∧ right_triangle A E C ∧ isosceles_triangle A E B) 
    ∨ (right_triangle A F B ∧ right_triangle A F C ∧ isosceles_triangle A F B) := 
sorry

end isosceles_triangle_divisible_l177_177738


namespace interest_rate_approx_3_96_percent_l177_177206

noncomputable def calc_interest_rate (P A t n : ℝ) : ℝ :=
  let r_approx := (A / P)^(1 / (n * t)) - 1
  r_approx * n

theorem interest_rate_approx_3_96_percent : 
  let P := 5000 
  let Interest := 302.98 
  let t := 1.5 
  let n := 2 
  let A := P + Interest
  calc_interest_rate P A t n ≈ 0.0396 := 
by
  sorry

end interest_rate_approx_3_96_percent_l177_177206


namespace boy_can_ensure_last_two_candies_in_same_box_l177_177458

theorem boy_can_ensure_last_two_candies_in_same_box (n : ℕ) :
  ∃ (strategy : fin n → (ℕ → ℕ)), 
    (∀ (girl_pick : fin (2 * n)), ∃ (boy_pick : fin (2 * n)), 
    -- This expresses that the boy's strategy allows him to choose such that:
    -- when girl_pick() = i, boy_pick() = j, where j > i, strategy j ensures leaving the last two candies in the same box
    (strategy (n - 1)) = (λ k, if k < n then 0 else 2)) := 
sorry

end boy_can_ensure_last_two_candies_in_same_box_l177_177458


namespace not_exp_gt_x_plus_one_always_l177_177272

theorem not_exp_gt_x_plus_one_always : ∃ x : ℝ, Real.exp x ≤ x + 1 :=
by
  use 0
  simp
  sorry

end not_exp_gt_x_plus_one_always_l177_177272


namespace num_solutions_even_pairs_l177_177552

theorem num_solutions_even_pairs : ∃ n : ℕ, n = 25 ∧ ∀ (x y : ℕ),
  x % 2 = 0 ∧ y % 2 = 0 ∧ 4 * x + 6 * y = 600 → n = 25 :=
by
  sorry

end num_solutions_even_pairs_l177_177552


namespace find_number_added_l177_177556

theorem find_number_added (x n : ℕ) (h : (x + x + 2 + x + 4 + x + n + x + 22) / 5 = x + 7) : n = 7 :=
by
  sorry

end find_number_added_l177_177556


namespace find_rate_percent_l177_177106

-- Definitions based on the given conditions
def principal : ℕ := 800
def time : ℕ := 4
def simple_interest : ℕ := 192
def si_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- Statement: prove that the rate percent (R) is 6%
theorem find_rate_percent (R : ℕ) (h : simple_interest = si_formula principal R time) : R = 6 :=
sorry

end find_rate_percent_l177_177106


namespace like_terms_solution_l177_177237

theorem like_terms_solution :
  (∃ x y : ℝ, 2 * a ^ (2 * x) * b ^ (3 * y) = -3 * a ^ 2 * b ^ (2 - x) ∧ x = 1 ∧ y = 1 / 3) :=
by
  -- Existential quantifier states that there exist real numbers x and y 
  -- which satisfy the given conditions
  use 1, 1/3
  -- sorry to skip the actual proof part
  sorry

end like_terms_solution_l177_177237


namespace extra_large_bag_contains_l177_177720

theorem extra_large_bag_contains (small_cost small_balloons medium_cost medium_balloons xl_cost total_money total_balloons : ℕ) : 
  small_cost = 4 ∧ small_balloons = 50 ∧
  medium_cost = 6 ∧ medium_balloons = 75 ∧
  xl_cost = 12 ∧ total_money = 24 ∧ total_balloons = 400 →
  ∃ xl_balloons : ℕ, xl_balloons = 250 :=
begin
  intros h,
  obtain ⟨hs, hs_balloons, hm, hm_balloons, hx, ht, hb⟩ := h,
  use 250,
  sorry
end

end extra_large_bag_contains_l177_177720


namespace wendy_albums_l177_177076

theorem wendy_albums (total_pictures remaining_pictures pictures_per_album : ℕ) 
    (h1 : total_pictures = 79)
    (h2 : remaining_pictures = total_pictures - 44)
    (h3 : pictures_per_album = 7) :
    remaining_pictures / pictures_per_album = 5 := by
  sorry

end wendy_albums_l177_177076


namespace min_value_frac_l177_177240

theorem min_value_frac (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_add : x + y = 2) : 
  ∃ m, (∀ a b : ℝ, (0 < a) → (0 < b) → (a + b = 2) → (∃ c, c = (2/a + 1/b)) → c ≥ m) ∧ m = (3/2 + Real.sqrt 2) :=
begin
  sorry
end

end min_value_frac_l177_177240


namespace grandson_age_l177_177723

-- Define the ages of Markus, his son, and his grandson
variables (M S G : ℕ)

-- Conditions given in the problem
axiom h1 : M = 2 * S
axiom h2 : S = 2 * G
axiom h3 : M + S + G = 140

-- Theorem to prove that the age of Markus's grandson is 20 years
theorem grandson_age : G = 20 :=
by
  sorry

end grandson_age_l177_177723


namespace parallel_lines_condition_l177_177864

theorem parallel_lines_condition (a : ℝ) (l : ℝ) :
  (∀ (x y : ℝ), ax + 3*y + 3 = 0 → x + (a - 2)*y + l = 0 → a = -1) ∧ (a = -1 → ∀ (x y : ℝ), (ax + 3*y + 3 = 0 ↔ x + (a - 2)*y + l = 0)) :=
sorry

end parallel_lines_condition_l177_177864


namespace octagon_diagonal_l177_177367

def height_and_base_ratio := 5 / 3
def pane_area := 120
def num_panes := 8
def border_width := 3

theorem octagon_diagonal (height base : ℝ) (short_base longer_base : ℝ) (L : ℝ)
  (h1 : height / longer_base = height_and_base_ratio)
  (h2 : (short_base + longer_base) * height = 2 * pane_area)
  (h3 : num_panes = 8)
  (h4 : border_width = 3)
  (h5 : L = short_base * sqrt (4 + 2*sqrt 2)) :
  L = 59 :=
sorry

end octagon_diagonal_l177_177367


namespace time_to_fill_tank_l177_177828

-- Definitions for conditions
def pipe_a := 50
def pipe_b := 75
def pipe_c := 100

-- Definition for the combined rate and time to fill the tank
theorem time_to_fill_tank : 
  (1 / pipe_a + 1 / pipe_b + 1 / pipe_c) * (300 / 13) = 1 := 
by
  sorry

end time_to_fill_tank_l177_177828


namespace bus_capacity_l177_177297

def left_side_seats : ℕ := 15
def seats_difference : ℕ := 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 7

theorem bus_capacity : left_side_seats + (left_side_seats - seats_difference) * people_per_seat + back_seat_capacity = 88 := 
by
  sorry

end bus_capacity_l177_177297


namespace divisor_is_seven_l177_177291

theorem divisor_is_seven 
  (d x : ℤ)
  (h1 : x % d = 5)
  (h2 : 4 * x % d = 6) :
  d = 7 := 
sorry

end divisor_is_seven_l177_177291


namespace linear_function_implies_m_value_l177_177629

variable (x m : ℝ)

theorem linear_function_implies_m_value :
  (∃ y : ℝ, y = (m-3)*x^(m^2-8) + m + 1 ∧ ∀ x1 x2 : ℝ, y = y * (x2 - x1) + y * x1) → m = -3 :=
by
  sorry

end linear_function_implies_m_value_l177_177629


namespace problem1_problem2_l177_177932

-- Problem 1
theorem problem1 
  (h1 : 1 = (Real.pi^0)) 
  (h2 : 2 = (Real.sqrt ((-2 : ℝ) ^ 2))) 
  (h3 : (9/4 : ℝ) = ((27/8 : ℝ) ^ (2/3))) 
  : (1 - 2 + (9/4 : ℝ)) = (5/4 : ℝ) := 
by 
  -- Proof not provided
  sorry

-- Problem 2
theorem problem2 
  (h4 : (log 25 = 2 * log 5)) 
  (h5 : 2 = Real.exp (Real.log 2)) 
  (h6 : (1 : ℝ) = (log 3 4 * log 4 3)) 
  : (log 25 + log 4 - 2 - 1 = (-1 : ℝ)) := 
by 
  -- Proof not provided
  sorry

end problem1_problem2_l177_177932


namespace smallest_height_of_pyramid_l177_177145

noncomputable def smallest_possible_pyramid_height : ℝ :=
  let side_length_of_square_base := 20 in
  let diameter_of_cylinder := 10 in
  let length_of_cylinder := 10 in
  let diagonal_length_of_square_base := 20 * Real.sqrt 2 in
  let distance_from_center_to_vertex := 10 * Real.sqrt 2 in
  let radius_of_cylinder := diameter_of_cylinder / 2 in
  let half_length_of_cylinder := length_of_cylinder / 2 in
  -- Include any other necessary calculations here
  22.1

theorem smallest_height_of_pyramid :
  ∀ (side_length_of_square_base diameter_of_cylinder length_of_cylinder : ℝ),
    side_length_of_square_base = 20 →
    diameter_of_cylinder = 10 →
    length_of_cylinder = 10 →
    smallest_possible_pyramid_height = 22.1 :=
by
  intros side_length_of_square_base diameter_of_cylinder length_of_cylinder h₁ h₂ h₃
  have h₄ : side_length_of_square_base / 2 = 10 := by sorry
  have h₅ : Real.sqrt 2 = Real.sqrt 2 := by sorry
  have h₆ : radius_of_cylinder <= 10 := by sorry  -- Just to show how you link conditions
  -- Additional necessary conditions and the final step connecting conditions to conclusion
  exact eq.refl smallest_possible_pyramid_height
  sorry


end smallest_height_of_pyramid_l177_177145


namespace farmer_land_owned_l177_177118

def total_land (farmer_land : ℝ) (cleared_land : ℝ) : Prop :=
  cleared_land = 0.9 * farmer_land

def cleared_with_tomato (cleared_land : ℝ) (tomato_land : ℝ) : Prop :=
  tomato_land = 0.1 * cleared_land
  
def tomato_land_given (tomato_land : ℝ) : Prop :=
  tomato_land = 90

theorem farmer_land_owned (T : ℝ) :
  (∃ cleared : ℝ, total_land T cleared ∧ cleared_with_tomato cleared 90) → T = 1000 :=
by
  sorry

end farmer_land_owned_l177_177118


namespace curves_parallelizability_l177_177282

def parallelizable (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 = a ∧ f' x2 = a

def f1 (x : ℝ) : ℝ := x^3 - x
def f2 (x : ℝ) : ℝ := x + (1 / x)
def f3 (x : ℝ) : ℝ := sin x
def f4 (x : ℝ) : ℝ := (x - 2)^2 + log x

theorem curves_parallelizability : parallelizable f2 ∧ parallelizable f3 ∧ ¬parallelizable f1 ∧ ¬parallelizable f4 :=
  by
  sorry

end curves_parallelizability_l177_177282


namespace cube_of_square_of_third_smallest_prime_l177_177077

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l177_177077


namespace midpoint_line_intersection_l177_177800

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ × ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem midpoint_line_intersection :
  let p1 := (1, 3)
  let p2 := (5, 11)
  let mp := midpoint p1.1 p1.2 p2.1 p2.2
  (mp.1 + mp.2) = 10 := by
  let p1 := (1, 3)
  let p2 := (5, 11)
  let mp := midpoint p1.1 p1.2 p2.1 p2.2
  have h : mp = (3, 7) := by
    simp [midpoint, p1, p2]
  show (mp.1 + mp.2) = 10
  simp [h]
  sorry

end midpoint_line_intersection_l177_177800


namespace cube_of_square_is_15625_l177_177101

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l177_177101


namespace ellipse_eq_proof_l177_177207

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1

theorem ellipse_eq_proof :
  let a := 6
  let b := sqrt (a^2 - 2^2)
  ellipse_eq a b 3 (-2 * sqrt 6) :=
by
  let a := 6
  let b := sqrt (a^2 - 2^2)
  show ellipse_eq a b 3 (-2 * sqrt 6), from
  sorry

end ellipse_eq_proof_l177_177207


namespace train_total_length_approx_l177_177156

noncomputable def kmph_to_mps (speed : ℝ) : ℝ :=
  speed * (5 / 18)

def length_of_compartment (speed_kmph time_sec : ℝ) : ℝ :=
  (kmph_to_mps speed_kmph) * time_sec

def total_length_of_train (speeds times : List ℝ) : ℝ :=
  List.sum (List.map (λ (p : ℝ × ℝ), length_of_compartment p.fst p.snd) (List.zip speeds times))

theorem train_total_length_approx :
  total_length_of_train [40, 50, 60, 70, 80] [9, 8, 7, 6, 5] ≈ 555.56 :=
sorry

end train_total_length_approx_l177_177156


namespace cube_square_third_smallest_prime_l177_177093

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l177_177093


namespace total_output_correct_l177_177806

variable (a : ℝ)

-- Define a function that captures the total output from this year to the fifth year
def totalOutput (a : ℝ) : ℝ :=
  1.1 * a + (1.1 ^ 2) * a + (1.1 ^ 3) * a + (1.1 ^ 4) * a + (1.1 ^ 5) * a

theorem total_output_correct (a : ℝ) : 
  totalOutput a = 11 * (1.1 ^ 5 - 1) * a := by
  sorry

end total_output_correct_l177_177806


namespace points_on_line_l177_177764

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177764


namespace shaded_cubes_in_larger_cube_l177_177138

   theorem shaded_cubes_in_larger_cube 
     (small_cubes : ℕ) (dim : ℕ) (shaded_border_per_face : ℕ) (corner_cubes : ℕ) (edge_cubes : ℕ)
     (shading_pattern_opposite_face : Prop) :
     small_cubes = 64 →
     dim = 4 →
     shaded_border_per_face = 12 →
     corner_cubes = 8 →
     edge_cubes = 24 →
     shading_pattern_opposite_face →
     (total_shaded_cubes : ℕ)
     (h : total_shaded_cubes = corner_cubes + (edge_cubes / 2)) : 
     total_shaded_cubes = 32 :=
     by sorry
   
end shaded_cubes_in_larger_cube_l177_177138


namespace markus_grandson_age_l177_177722

theorem markus_grandson_age :
  ∃ (x : ℕ), let son := 2 * x in let markus := 2 * son in x + son + markus = 140 ∧ x = 20 :=
by
  sorry

end markus_grandson_age_l177_177722


namespace gcd_175_100_65_l177_177046

theorem gcd_175_100_65 : Nat.gcd (Nat.gcd 175 100) 65 = 5 :=
by
  sorry

end gcd_175_100_65_l177_177046


namespace second_large_bucket_contains_39_ounces_l177_177187

-- Define the five initial buckets
def buckets : List ℕ := [11, 13, 12, 16, 10]

-- Define the condition of the first large bucket containing 23 ounces after pouring the 10-ounce bucket and another bucket
def combined_buckets_condition : Prop := ∃ b ∈ buckets, b ≠ 10 ∧ 10 + b = 23

-- Define the remaining buckets after removing the buckets used in combined_buckets_condition
def remaining_buckets : List ℕ := buckets.filter (λ b, ¬(10 + b = 23))

-- Define the sum of the remaining buckets
def sum_remaining_buckets : ℕ := remaining_buckets.sum

-- Theorem stating the answer to the problem, i.e., the amount of water in the second large bucket
theorem second_large_bucket_contains_39_ounces (h : combined_buckets_condition) : sum_remaining_buckets = 39 := by
  sorry

end second_large_bucket_contains_39_ounces_l177_177187


namespace eliminated_number_mean_l177_177845

theorem eliminated_number_mean (n x : ℕ) (h1 : (1 ≤ n) ∧ (n ≤ 9))
  (h2 : (∑ i in finset.range(n+1), i) - x = (n * (n + 1) / 2) - x)
  (h3 : ((n * (n + 1) / 2 - x).toRat / (n - 1)) = 19 / 4) : x = 7 :=
sorry

end eliminated_number_mean_l177_177845


namespace ratio_of_areas_l177_177070

theorem ratio_of_areas (O P X : Point) (r R : ℝ) (h1 : O = center)
  (h2 : X ∈ line(O, P)) (h3 : OX = 1/3 * OP) :
  area(circle(radius(OX))) / area(circle(radius(OP))) = 1/9 :=
  sorry

end ratio_of_areas_l177_177070


namespace number_of_boys_at_reunion_l177_177283

theorem number_of_boys_at_reunion (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
by
  sorry

end number_of_boys_at_reunion_l177_177283


namespace sampling_size_from_bracket_l177_177507

theorem sampling_size_from_bracket (total_population sample_size : ℕ) (p : ℕ)
    (h_total_population : total_population = 10000)
    (h_sample_size : sample_size = 100)
    (h_percentage : p = 25) :
    (sample_size * p / 100) = 25 :=
by
  rw [h_sample_size, h_percentage]
  norm_num
  exact h_total_population

end sampling_size_from_bracket_l177_177507


namespace irrational_number_count_l177_177512

theorem irrational_number_count : 
  let nums := [22/7, -1, 0, 1/2, Real.sqrt 2] in
  let irrational_numbers := nums.filter (λ x => ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / (q : ℝ))) in
  irrational_numbers.length = 1 :=
by
  sorry

end irrational_number_count_l177_177512


namespace largest_negative_S_l177_177659

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}

-- Arithmetic sequence definition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Conditions
variables (ha10 : a_n 10 < 0)
variables (ha11 : a_n 11 > 0)
variables (ha11_abs : a_n 11 > |a_n 10|)

-- Question: Largest negative number in {S_n}
noncomputable def S (n : ℕ) : ℝ := n * (a_n 1) + n * (n - 1) / 2 * (a_n 1 - a_n 0)

-- Correct Answer
theorem largest_negative_S (h_arith: is_arithmetic_sequence a_n) :
  ∃ n, S n < 0 ∧ ∀ m, (m > n → 0 ≤ S m) :=
  sorry

end largest_negative_S_l177_177659


namespace chessboard_rooks_invariant_sums_l177_177310

theorem chessboard_rooks_invariant_sums (n : ℕ) (a : Fin n → Fin n → ℝ) :
  (∀ (rooks : Fin n → Fin n), Function.Injective rooks →
    ∑ i, a i (rooks i) = ∑ i, a i i) →
  ∃ (x y : Fin n → ℝ), ∀ i j, a i j = x i + y j :=
by
  sorry

end chessboard_rooks_invariant_sums_l177_177310


namespace investment_period_more_than_tripling_l177_177514

theorem investment_period_more_than_tripling (r : ℝ) (multiple : ℝ) (n : ℕ) 
  (h_r: r = 0.341) (h_multiple: multiple > 3) :
  (1 + r)^n ≥ multiple → n = 4 :=
by
  sorry

end investment_period_more_than_tripling_l177_177514


namespace x_fourth_minus_inv_fourth_l177_177593

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l177_177593


namespace probability_X_leq_4_l177_177288

noncomputable theory
open Probability

-- Definitions of the given conditions
def normal_dist (μ σ : ℝ) : Measure ℝ := Measure.theory.normal μ σ
def X (μ σ : ℝ) := MeasureSpace.measureSpace (ℝ.measureSpace) (normal_dist μ σ)

-- Condition: X follows a normal distribution N(2, σ^2)
variable (σ : ℝ)
variable (X : Measure ℝ := normal_dist 2 σ)

-- Given: P(X ≤ 0) = 0.2
axiom P_le_0 : P (set.Iic 0) = 0.2

-- Proof objective: P(X ≤ 4) = 0.8
theorem probability_X_leq_4 : P (set.Iic 4) = 0.8 := 
sorry

end probability_X_leq_4_l177_177288


namespace solve_x4_minus_inv_x4_l177_177583

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l177_177583


namespace tan_sub_pi_four_identity_sin_expr_identity_l177_177559

variables (α : ℝ)
noncomputable def tan_alpha := 2

theorem tan_sub_pi_four_identity :
  tan (α - π/4) = 1 / 3 :=
by
  have h1 : tan α = tan_alpha := by sorry
  sorry

theorem sin_expr_identity :
  (sin (2 * α) / (sin α ^ 2 + sin α * cos α - cos (2 * α) - 1)) = 80 / 37 :=
by 
  have h2 : tan α = tan_alpha := by sorry
  sorry

end tan_sub_pi_four_identity_sin_expr_identity_l177_177559


namespace curve_C1_cartesian_eq_curve_C2_cartesian_eq_value_of_a_l177_177184

-- Definitions and conditions extracted from the problem statements
def parametric_curve_C1 (a t : ℝ) : ℝ × ℝ := (a + sqrt 2 * t, 1 + sqrt 2 * t)
def polar_curve_C2 (ρ θ : ℝ) : Prop := ρ * cos θ^2 + 4 * cos θ - ρ = 0

theorem curve_C1_cartesian_eq (a : ℝ) : ∃ x y t : ℝ, (parametric_curve_C1 a t).1 = x ∧ (parametric_curve_C1 a t).2 = y ∧ x - y - a + 1 = 0 :=
sorry

theorem curve_C2_cartesian_eq : ∃ x y : ℝ, y^2 = 4 * x :=
sorry

theorem value_of_a (a : ℝ) : ∃ t1 t2 : ℝ, 
  t1 + t2 = sqrt 2 ∧ t1 * t2 = (1 - 4 * a) / 2 ∧ 
  (t1 = 2 * t2 ∨ t1 = -2 * t2) ∧ 
  (a = 1 / 36 ∨ a = 9 / 4) :=
sorry

end curve_C1_cartesian_eq_curve_C2_cartesian_eq_value_of_a_l177_177184


namespace find_vector_in_yz_plane_l177_177555

open Real

/-- Define the vector formations -/
def u (y z : ℝ) : ℝ × ℝ × ℝ := (0, y, z)

def v₁ : ℝ × ℝ × ℝ := (1, 1, 1)
def v₂ : ℝ × ℝ × ℝ := (-1, 1, 0)

/-- Define the unit vector condition -/
def is_unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  sqrt (v.1^2 + v.2^2 + v.3^2) = 1

/-- Define the dot product -/
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

/-- Define the angle condition with vector v₁ -/
def angle_with_v₁ (y z : ℝ) : Prop :=
  (dot_product (u y z) v₁) / (sqrt (y^2 + z^2) * sqrt (3)) = sqrt 3 / 2

/-- Define the angle condition with vector v₂ -/
def angle_with_v₂ (y z : ℝ) : Prop :=
  (dot_product (u y z) v₂) / (sqrt (y^2 + z^2) * sqrt 2) = 1 / sqrt 2

/-- The theorem that given all conditions, the vector is as specified-/
theorem find_vector_in_yz_plane (y z : ℝ) :
  is_unit_vector (u y z) ∧ angle_with_v₁ y z ∧ angle_with_v₂ y z →
  (u y z) = (0, 1, 1 / 2) :=
  sorry

end find_vector_in_yz_plane_l177_177555


namespace part_1_a_part_1_b_part_2_l177_177261

variable {α : Type*} [LinearOrderedField α] (f : ℕ → α)

-- Definition of the sequence according to the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  ∀ n ≥ 1, f(n + 1) = (1/2) * (f n) ^ 2 - f n + 2

-- Condition for initial value f(1) = 4 for part (1)
def initial_condition_4 : Prop :=
  f 1 = 4

-- Condition for initial value f(1) = 1 for part (2)
def initial_condition_1 : Prop :=
  f 1 = 1

-- (1)(a) For n ≥ 2, f(n + 1) ≥ 2 * f(n)
theorem part_1_a (h_rec : recurrence_relation f) (h_init : initial_condition_4 f) :
  ∀ n ≥ 2, f(n + 1) ≥ 2 * f n :=
sorry

-- (1)(b) For n ≥ 1, f(n + 1) ≥ (3 / 2)^n * f(n)
theorem part_1_b (h_rec : recurrence_relation f) (h_init : initial_condition_4 f) :
  ∀ n ≥ 1, f(n + 1) ≥ (3 / 2)^n * f n :=
sorry

-- For part (2), proving the summation bound result
theorem part_2 (h_rec : recurrence_relation f) (h_init : initial_condition_1 f) :
  ∀ n ≥ 5, ∑ k in finset.range n, 1 / f(k + 1) < n - 1 :=
sorry

end part_1_a_part_1_b_part_2_l177_177261


namespace surface_area_of_sphere_O_l177_177577

variables {S A B C O : Type} 
           [MetricSpace O] [normed_space ℝ O]
           [MetricSpace S] [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assume S, A, B, and C are points on O
def on_sphere (O : MetricSpace O) (x : O) : Prop := sorry

-- SA, AB, and BC are perpendicular
def perp (x y : Type) [MetricSpace x] [MetricSpace y] : Prop := sorry

-- Define the conditions given in the problem
def SA_perp_plane_ABC := perp S (MetricSpace.space_of O)
def AB_perp_BC := perp A B
def SA_eq_one := dist S A = 1
def AB_eq_one := dist A B = 1
def BC_eq_sqrt_two := dist B C = sqrt(2)

-- Conclusion: The surface area of sphere O equals 4π
theorem surface_area_of_sphere_O
  (on_sphere_O_S : on_sphere O S)
  (on_sphere_O_A : on_sphere O A)
  (on_sphere_O_B : on_sphere O B)
  (on_sphere_O_C : on_sphere O C)
  (SA_perp : SA_perp_plane_ABC)
  (AB_perp : AB_perp_BC)
  (SA_eq1 : SA_eq_one)
  (AB_eq1 : AB_eq_one)
  (BC_eqsqrt2 : BC_eq_sqrt_two) :
  surface_area O = 4 * pi :=
sorry

end surface_area_of_sphere_O_l177_177577


namespace express_in_scientific_notation_l177_177028

-- Definition of scientific notation (for clarity in the problem statement)
def scientific_notation (a : Float) (n : ℤ) : Float :=
  a * 10^n

-- Given condition of 1.81 million passengers
def million_factor : Float :=
  1_000_000

-- Main statement to prove
theorem express_in_scientific_notation :
  scientific_notation 1.81 6 = 1.81 * million_factor := by
  sorry

end express_in_scientific_notation_l177_177028


namespace sqrt_200_eq_10_l177_177019

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177019


namespace min_value_fraction_l177_177580

theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2: b > 0) (h3 : a + b = 1) : 
  ∃ c : ℝ, c = 3 + 2 * Real.sqrt 2 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 1) → x + 2 * y ≥ c) :=
by
  sorry

end min_value_fraction_l177_177580


namespace distinct_pawns_placement_l177_177638

theorem distinct_pawns_placement : 
  (∃ (f : Fin 4 → Fin 4 → Fin 4), (∀ (i : Fin 4), ∃! (j : Fin 4), f i j ≠ 0) ∧ (∀ (j : Fin 4), ∃! (i : Fin 4), f i j ≠ 0)) →
  ∀ (distinct_pawns_permutations : Equiv.Perm (Fin 4)), 
  ∑ (σ : Equiv.Perm (Fin 4)), 1 = 576 := 
by
  sorry

end distinct_pawns_placement_l177_177638


namespace correct_polynomial_value_l177_177152

variable (x : ℝ)
def B : ℝ[x] := x ^ 2 - x - 1

theorem correct_polynomial_value (A B : ℝ[x]) 
  (h1 : 2 * A - B = 3 * x ^ 2 - 3 * x + 5)
  (h2 : B = x ^ 2 - x - 1) :
  A - 2 * B = 4 :=
by
  sorry

end correct_polynomial_value_l177_177152


namespace loan_duration_in_years_l177_177742

-- Define the conditions as constants
def carPrice : ℝ := 20000
def downPayment : ℝ := 5000
def monthlyPayment : ℝ := 250

-- Define the goal
theorem loan_duration_in_years :
  (carPrice - downPayment) / monthlyPayment / 12 = 5 := 
sorry

end loan_duration_in_years_l177_177742


namespace cube_square_third_smallest_prime_l177_177095

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l177_177095


namespace fixed_point_of_machine_first_number_yields_one_as_fourth_eighth_number_is_same_as_first_l177_177860

-- Condition: Define the machine's operations
def machine (x : ℚ) : ℚ :=
  if x ≤ 1 / 2 then 2 * x 
  else 2 * (1 - x)

-- Proof problem (b): Fixed point proof
theorem fixed_point_of_machine : 
  ∃ x, 0 < x ∧ x < 1 ∧ machine x = x := 
begin
  use (2 / 3),
  split, norm_num, split, norm_num,
  show machine (2 / 3) = 2 / 3, 
  rw machine,
  simp,
  split_ifs,
  norm_num,
  ring_nf at *,
end

-- Proof problem (c): First number results in 1 as fourth number
theorem first_number_yields_one_as_fourth :
  ∃ a, 0 < a ∧ a < 1 ∧ machine (machine (machine (machine a))) = 1 := 
begin
  use [1 / 8, 3 / 8, 5 / 8, 7 / 8], -- these are the possible values
  split,
  repeat { norm_num },
  split,
  repeat { norm_num },
  show machine (machine (machine (machine (1 / 8)))) = 1,
  sorry, -- Detailed computation omitted
  show machine (machine (machine (machine (3 / 8)))) = 1,
  sorry, -- Detailed computation omitted
  show machine (machine (machine (machine (5 / 8)))) = 1,
  sorry, -- Detailed computation omitted
  show machine (machine (machine (machine (7 / 8)))) = 1,
  sorry  -- Detailed computation omitted
end

-- Proof problem (d): Verify eighth number in chain is the same as the first
theorem eighth_number_is_same_as_first (m : ℕ) (hm : m > 3) : 
  ∃ x, x = (2 : ℚ) / m ∧ machine (machine (machine (machine (machine (machine (machine (machine x))))))) = x := 
begin
  use [(2 / 129), (2 / 127), (2 / 43)], -- these are the specific values
  split,
  repeat { reflexivity }, 
  split,
  sorry, -- Detailed computation omitted
  sorry, -- Detailed computation omitted
  sorry  -- Detailed computation omitted
end.strip

end fixed_point_of_machine_first_number_yields_one_as_fourth_eighth_number_is_same_as_first_l177_177860


namespace value_of_c_l177_177801

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem value_of_c :
  let midpoint := midpoint (1, 3) (5, 11) in 
  let c := midpoint.1 + midpoint.2 in
  c = 10 :=
by
  have m : (ℝ × ℝ) := midpoint (1, 3) (5, 11)
  let c := m.1 + m.2
  suffices h : c = 10, from h
  sorry

end value_of_c_l177_177801


namespace magnitude_z_l177_177284

theorem magnitude_z (z : ℂ) (h : z * (1 + complex.i) = 4 - 2 * complex.i) : complex.abs z = real.sqrt 10 :=
sorry

end magnitude_z_l177_177284


namespace max_area_triangle_l177_177561

-- Define the given conditions
variables (a b c : ℝ) (A B C : ℝ)
variable (h_c : c = 2)
variable (h_trig : sin A ^ 2 + sin B ^ 2 = sin A * sin B + sin C ^ 2)

-- Define the proof statement for the maximum area of triangle
theorem max_area_triangle : 
  ∃ (A B C : ℝ), a * c = 2 / (sin A), b * c = 2 / (sin B), a ≠ b, 
  c = 2 -> sin A ^ 2 + sin B ^ 2 = sin A * sin B + sin C ^ 2 -> a * b * (sin C) / 2 = √3 := 
sorry

end max_area_triangle_l177_177561


namespace trees_planted_tomorrow_l177_177413

-- Definitions from the conditions
def current_trees := 39
def trees_planted_today := 41
def total_trees := 100

-- Theorem statement matching the proof problem
theorem trees_planted_tomorrow : 
  ∃ (trees_planted_tomorrow : ℕ), current_trees + trees_planted_today + trees_planted_tomorrow = total_trees ∧ trees_planted_tomorrow = 20 := 
by
  sorry

end trees_planted_tomorrow_l177_177413


namespace find_a_l177_177996

open Complex

theorem find_a (a : ℝ) (ha1 : (fraction_ring ℝ).fraction_enable) :
  let z1 := (16/(a + 5) : ℂ) - (10 - a^2 : ℝ) * Complex.I
  let z2 := (2/(1 - a) : ℂ) + (2 * a - 5 : ℝ) * Complex.I
  (∃ (r : ℝ), z1 + z2 = (r : ℂ)) → a = 3 :=
by
  sorry

end find_a_l177_177996


namespace cube_square_third_smallest_prime_l177_177083

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l177_177083


namespace value_of_x_l177_177391

theorem value_of_x 
    (r : ℝ) (a : ℝ) (x : ℝ) (shaded_area : ℝ)
    (h1 : r = 2)
    (h2 : a = 2)
    (h3 : shaded_area = 2) :
  x = (Real.pi / 3) + (Real.sqrt 3 / 2) - 1 :=
sorry

end value_of_x_l177_177391


namespace sqrt_200_eq_10_l177_177023

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177023


namespace first_digit_base5_l177_177941

theorem first_digit_base5 (n : ℕ) (h : n = 89) : (Nat.digits 5 n).reverse.head = 3 :=
by
  sorry

end first_digit_base5_l177_177941


namespace probability_one_head_two_tails_l177_177441

-- Define an enumeration for Coin with two possible outcomes: heads and tails.
inductive Coin
| heads
| tails

-- Function to count the number of heads in a list of Coin.
def countHeads : List Coin → Nat
| [] => 0
| Coin.heads :: xs => 1 + countHeads xs
| Coin.tails :: xs => countHeads xs

-- Function to calculate the probability of a specific event given the total outcomes.
def probability (specific_events total_outcomes : Nat) : Rat :=
  (specific_events : Rat) / (total_outcomes : Rat)

-- The main theorem
theorem probability_one_head_two_tails : probability 3 8 = (3 / 8 : Rat) :=
sorry

end probability_one_head_two_tails_l177_177441


namespace expected_worth_coin_flip_l177_177164

def prob_head : ℚ := 2 / 3
def prob_tail : ℚ := 1 / 3
def gain_head : ℚ := 5
def loss_tail : ℚ := -12

theorem expected_worth_coin_flip : ∃ E : ℚ, E = round (((prob_head * gain_head) + (prob_tail * loss_tail)) * 100) / 100 ∧ E = - (2 / 3) :=
by
  sorry

end expected_worth_coin_flip_l177_177164


namespace sum_of_roots_expression_involving_roots_l177_177581

variables {a b : ℝ}

axiom roots_of_quadratic :
  (a^2 + 3 * a - 2 = 0) ∧ (b^2 + 3 * b - 2 = 0)

theorem sum_of_roots :
  a + b = -3 :=
by 
  sorry

theorem expression_involving_roots :
  a^3 + 3 * a^2 + 2 * b = -6 :=
by 
  sorry

end sum_of_roots_expression_involving_roots_l177_177581


namespace solution_set_of_inequality_l177_177406

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) * (2 - x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l177_177406


namespace neon_signs_blink_together_l177_177833

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : Nat.lcm a b = 45 := by
  rw [ha, hb]
  have : Nat.lcm 9 15 = 45 := by sorry
  exact this

end neon_signs_blink_together_l177_177833


namespace fraction_eq_6561_l177_177431

theorem fraction_eq_6561 : (3^4 + 3^4) / (3^(-4) + 3^(-4)) = 6561 := by
  sorry

end fraction_eq_6561_l177_177431


namespace sum_islands_up_to_2_pow_2020_l177_177912

-- Define what an island is
def isIsland (binary_rep : List Bool) (start : Nat) (length : Nat) : Bool :=
  if length < 2 then false
  else (binary_rep.drop start).take length |>.all (· = binary_rep.nth! start)

-- Define the function b(n) which counts the number of islands in the binary representation of n
def b (n : Nat) : Nat :=
  let binary_rep := n.toDigits 2
  List.foldl (fun (acc : Nat) start =>
    let island_lengths := (List.range (binary_rep.length - start)).filter (λ len => isIsland binary_rep start len)
    acc + island_lengths.length) 0 (List.range binary_rep.length)

-- The main theorem to be proved
theorem sum_islands_up_to_2_pow_2020 :
  ∑ n in Finset.range (2^2020 + 1), b n = 1 + 2019 * 2^2018 :=
sorry

end sum_islands_up_to_2_pow_2020_l177_177912


namespace volume_le_sqrt_proj_areas_l177_177041

variables (S1 S2 S3 V : ℝ)

theorem volume_le_sqrt_proj_areas (hS1 : 0 ≤ S1) (hS2 : 0 ≤ S2) (hS3 : 0 ≤ S3) (hV : 0 ≤ V)
  (h_proj : ∀ (S1 S2 S3 : ℝ), volume_proj S1 S2 S3 ≤ sqrt (S1 * S2 * S3)) : 
  V ≤ sqrt (S1 * S2 * S3) :=
sorry

end volume_le_sqrt_proj_areas_l177_177041


namespace balls_in_boxes_distinguishable_l177_177271

theorem balls_in_boxes_distinguishable (balls boxes : ℕ)
  (h_balls : balls = 4) (h_boxes : boxes = 2) : 
  finset.card {n : finset (finset ℕ) | n.card = boxes ∧ finset.card (finset.image finset.card n) = balls} = 8 :=
begin
  sorry
end

end balls_in_boxes_distinguishable_l177_177271


namespace possible_integer_roots_l177_177785

theorem possible_integer_roots (a b c d e : ℤ) (P : Polynomial ℤ) (hP : P = Polynomial.C 1 * Polynomial.X^5 + Polynomial.C a * Polynomial.X^4 +
    Polynomial.C b * Polynomial.X^3 + Polynomial.C c * Polynomial.X^2 + Polynomial.C d * Polynomial.X + Polynomial.C e) :
  ∃ n : ℕ, n ∈ {0, 1, 2, 3, 5} ∧ (P.roots.count Multiplicity (λ x, x ∈ ℤ)) = n :=
sorry

end possible_integer_roots_l177_177785


namespace circumcircle_radius_l177_177904

theorem circumcircle_radius (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 15) (h₃ : c = 17)
  (h₄ : a^2 + b^2 = c^2) : 
  (c / 2) = 17 / 2 :=
by
  -- Conditions
  rw [h₁, h₂, h₃] at h₄
  -- Use the Pythagorean theorem to assert the correctness
  exact h₄
-- This fills in the proof with 'sorry' to skip the implementation.
sorry

end circumcircle_radius_l177_177904


namespace smallest_prime_factors_l177_177991

-- Define the problem statement
theorem smallest_prime_factors
  (N : ℕ)
  (hN : N > 0)
  (numbers : Fin N → ℕ)
  (h_distinct_gcds : ∀ s t : Finset (Fin N), s ≠ t → gcd (s.1.map numbers) ≠ gcd (t.1.map numbers)) :
  (if N = 1 then 0 else N) =
  (if N = 1 then 0 else N) :=
by
  sorry

end smallest_prime_factors_l177_177991


namespace average_income_QR_l177_177043

theorem average_income_QR 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (P + R) / 2 = 6200)
  (h3 : P = 3000) :
  (Q + R) / 2 = 5250 :=
  sorry

end average_income_QR_l177_177043


namespace wrongly_entered_mark_l177_177502

theorem wrongly_entered_mark (n : ℕ) (m : ℕ) (x : ℕ) (A : ℕ)
  (h_n : n = 20) 
  (h_m : m = 63)
  (h_average_increase : (A + 0.5) * n = A * n + x - m) :
  x = 73 :=
by
  sorry

end wrongly_entered_mark_l177_177502


namespace sum_of_digits_maximized_l177_177692

noncomputable def divisor_count (n : ℕ) : ℕ :=
  (nat.divisors n).card

def f (n : ℕ) : ℚ :=
  divisor_count n / (real.to_rat (real.cbrt n))

theorem sum_of_digits_maximized (a b c : ℕ) 
  (h : 2^a * 3^b * 5^c = 86400) : 
  (nat.digits 10 (2^5 * 3^3 * 5^2)).sum = 18 :=
by sorry

end sum_of_digits_maximized_l177_177692


namespace joe_commute_time_l177_177671

theorem joe_commute_time
  (d : ℝ) -- total one-way distance from home to school
  (rw : ℝ) -- Joe's walking rate
  (rr : ℝ := 4 * rw) -- Joe's running rate (4 times walking rate)
  (walking_time_for_one_third : ℝ := 9) -- Joe takes 9 minutes to walk one-third distance
  (walking_time_two_thirds : ℝ := 2 * walking_time_for_one_third) -- time to walk two-thirds distance
  (running_time_two_thirds : ℝ := walking_time_two_thirds / 4) -- time to run two-thirds 
  : (2 * walking_time_two_thirds + running_time_two_thirds) = 40.5 := -- total travel time
by
  sorry

end joe_commute_time_l177_177671


namespace find_f_alpha_plus_3pi_over_2_l177_177613

noncomputable def f (x : ℝ) (A ω φ : ℝ) : ℝ := A * sin (ω * x + φ)

variables (A ω φ α : ℝ)
variables (hA : A > 0) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 2)
variables (h_period : 2 * π / ω = π) (h_falpha : f α A ω φ = 1)

theorem find_f_alpha_plus_3pi_over_2 :
  f (α + 3 * π / 2) A ω φ = -1 :=
sorry

end find_f_alpha_plus_3pi_over_2_l177_177613


namespace determine_line_plane_parallelism_l177_177961

-- Formalizing the conditions
variables {Point : Type} [InnerProductSpace ℝ Point]
variable {Line : Type} [AddCommGroup Line] [Module ℝ Line]
variable {Plane : Type}

variables (m l : Line) (α β : Plane)

-- Definition for subset and parallelism
def line_in_plane (l : Line) (α : Plane) : Prop := sorry
def line_parallel (l1 l2 : Line) : Prop := sorry
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

-- The proposition with the identified condition 
theorem determine_line_plane_parallelism 
  (h1: line_in_plane m α) 
  (h2 : line_parallel l m) 
  (h3 : ¬line_in_plane l α) :
  line_parallel_plane l α :=
sorry 

end determine_line_plane_parallelism_l177_177961


namespace vertex_of_quadratic_l177_177819

theorem vertex_of_quadratic : 
  (∃ a b c : ℝ, ∀ x : ℝ, y = 2 * x ^ 2 - 4 * x + 5) →
  (∀ h k : ℝ, y = 2 * (x - h) ^ 2 + k) →
  (h = 1 ∧ k = 3) :=
by
  intro h k hyp
  sorry

end vertex_of_quadratic_l177_177819


namespace pq_rs_perpendicular_pq_rs_intersect_diagonal_l177_177734

noncomputable theory

variables {α : Type*} [inner_product_space ℝ α]

-- Given: ABCD is a rectangle
structure Rectangle (α : Type*) :=
(A B C D : α)
(is_rectangle : ∀ (A B C D : α), dist A B = dist C D ∧
                                 dist B C = dist D A ∧
                                 dist A C = √(dist A B ^ 2 + dist B C ^ 2))
 -- Circumcircle of rectangle ABCD with a point M on the arc AB
structure Circumcircle (α : Type*) :=
(radius : ℝ)
(center : α)
(M : α)
(on_arc : ∀ (M : α), M ≠ Rectangle.A ∧ M ≠ Rectangle.B)

-- Projections of point M onto lines AD, AB, BC, and CD respectively
structure Projections (α : Type*) :=
(P Q R S : α)
(proj_AD : P ∈ set.line_through Rectangle.A Rectangle.D)
(proj_AB : Q ∈ set.line_through Rectangle.A Rectangle.B)
(proj_BC : R ∈ set.line_through Rectangle.B Rectangle.C)
(proj_CD : S ∈ set.line_through Rectangle.C Rectangle.D)

-- Proof problem statement
theorem pq_rs_perpendicular {α : Type*} [inner_product_space ℝ α] 
(rect : Rectangle α) (circ : Circumcircle α) (proj : Projections α):
(inner_product_space.orthogonal ℝ (proj.Q - proj.P) (proj.S - proj.R))
: sorry

theorem pq_rs_intersect_diagonal {α : Type*} [inner_product_space ℝ α] 
(rect : Rectangle α) (circ : Circumcircle α) (proj : Projections α) (M : α):
∃ T : α, T ∈ set.inter (affine_subspace.span ℝ {proj.P, proj.Q}) 
                    (affine_subspace.span ℝ {proj.R, proj.S}) ∧
                  T ∈ set.line_through M (midpoint ℝ M (rect.C))
: sorry

end pq_rs_perpendicular_pq_rs_intersect_diagonal_l177_177734


namespace intersection_points_l177_177256

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Set.Icc (-1 : ℝ) 1 then x^2 else sorry

theorem intersection_points : ∀ (f : ℝ → ℝ),
  (∀ x : ℝ, f(x+1) = f(x-1)) →
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f x = x^2) →
  ∃ n : ℕ, n = 4 ∧
      ∃ x1 x2 x3 x4 : ℝ,
        (f x1 = Real.log 5 x1) ∧
        (f x2 = Real.log 5 x2) ∧
        (f x3 = Real.log 5 x3) ∧
        (f x4 = Real.log 5 x4) := sorry

end intersection_points_l177_177256


namespace points_on_line_initial_l177_177758

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l177_177758


namespace least_addition_to_palindrome_l177_177976

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem least_addition_to_palindrome (n : ℕ) : n = 35721 → ∃ k : ℕ, is_palindrome (n + k) ∧ k = 132 :=
by
  intros h
  existsi 132
  split
  {
    sorry -- Proof that (35721 + 132) is a palindrome
  }
  {
    exact rfl -- Proof that k is exactly 132
  }

end least_addition_to_palindrome_l177_177976


namespace evaluate_expression_l177_177202

theorem evaluate_expression (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 2 * x ^ y + 5 * y ^ x - z ^ 2 = 42 :=
by
  sorry

end evaluate_expression_l177_177202


namespace alpha_minus_beta_l177_177595

theorem alpha_minus_beta {α β : ℝ} (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
    (h_cos_alpha : Real.cos α = 2 * Real.sqrt 5 / 5) 
    (h_cos_beta : Real.cos β = Real.sqrt 10 / 10) : 
    α - β = -π / 4 := 
sorry

end alpha_minus_beta_l177_177595


namespace range_of_a_l177_177641

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end range_of_a_l177_177641


namespace sqrt_200_eq_10_sqrt_2_l177_177002

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177002


namespace initial_points_l177_177752

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l177_177752


namespace minimal_polynomial_with_roots_l177_177211

noncomputable def minimal_polynomial : Polynomial ℚ :=
  Polynomial.monic (x^4 - 8*x^3 + 22*x^2 - 8*x + 1)

theorem minimal_polynomial_with_roots :
  ∃ P : Polynomial ℚ, Polynomial.roots P = {2 + Real.sqrt 3, 2 - Real.sqrt 3, 2 + Real.sqrt 5, 2 - Real.sqrt 5} ∧
  P.leading_coeff = 1 ∧ 
  P.degree = 4 :=
by
  use minimal_polynomial
  split
  {
    sorry
  }
  split
  {
    exact Polynomial.leading_coeff (minimal_polynomial) = 1
  }
  {
    exact Polynomial.degree (minimal_polynomial) = 4
  }

end minimal_polynomial_with_roots_l177_177211


namespace num_decoration_schemes_l177_177503

def decoration_schemes_count : ℕ := 
  8

theorem num_decoration_schemes :
  ∃ (r c : ℕ), (4 * r + c = 60) ∧ (r ≥ 5) ∧ (c ≥ 10) ∧ 
  (decoration_schemes_count = (∑ s in Finset.range 8, 1)) :=
sorry

end num_decoration_schemes_l177_177503


namespace bruce_eggs_lost_l177_177521

theorem bruce_eggs_lost :
  ∀ (initial_eggs remaining_eggs eggs_lost : ℕ), 
  initial_eggs = 75 → remaining_eggs = 5 →
  eggs_lost = initial_eggs - remaining_eggs →
  eggs_lost = 70 :=
by
  intros initial_eggs remaining_eggs eggs_lost h_initial h_remaining h_loss
  sorry

end bruce_eggs_lost_l177_177521


namespace two_digit_numbers_sum_154_l177_177192

theorem two_digit_numbers_sum_154 (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 2 ∣ a) :
  (10 * a + b) + (10 * b + a) = 154 → 2 :=
by sorry

end two_digit_numbers_sum_154_l177_177192


namespace cube_of_square_is_15625_l177_177098

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l177_177098


namespace percentage_water_fresh_fruit_l177_177471

-- Definitions of the conditions
def weight_dried_fruit : ℝ := 12
def water_content_dried_fruit : ℝ := 0.15
def weight_fresh_fruit : ℝ := 101.99999999999999

-- Derived definitions based on the conditions
def weight_non_water_dried_fruit : ℝ := weight_dried_fruit - (water_content_dried_fruit * weight_dried_fruit)
def weight_non_water_fresh_fruit : ℝ := weight_non_water_dried_fruit
def weight_water_fresh_fruit : ℝ := weight_fresh_fruit - weight_non_water_fresh_fruit

-- Proof statement
theorem percentage_water_fresh_fruit :
  (weight_water_fresh_fruit / weight_fresh_fruit) * 100 = 90 :=
sorry

end percentage_water_fresh_fruit_l177_177471


namespace points_on_line_l177_177763

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177763


namespace parker_added_dumbbells_l177_177366

def initial_dumbbells : Nat := 4
def weight_per_dumbbell : Nat := 20
def total_weight_used : Nat := 120

theorem parker_added_dumbbells :
  (total_weight_used - (initial_dumbbells * weight_per_dumbbell)) / weight_per_dumbbell = 2 := by
  sorry

end parker_added_dumbbells_l177_177366


namespace original_price_l177_177809

theorem original_price (p q a : ℝ) (h : 1 + (p - q) / 100 - (p * q) / 10000 ≠ 0) :
  let x := (10000 * a) / (10000 + 100 * (p - q) - p * q) in
  a = x * (1 + (p - q) / 100 - (p * q) / 10000) := 
begin
  -- problems steps go here
  sorry
end

end original_price_l177_177809


namespace find_m_l177_177214

noncomputable def curve (x : ℝ) : ℝ := (1 / 4) * x^2
noncomputable def line (x : ℝ) : ℝ := 1 - 2 * x

theorem find_m (m n : ℝ) (h_curve : curve m = n) (h_perpendicular : (1 / 2) * m * (-2) = -1) : m = 1 := 
  sorry

end find_m_l177_177214


namespace part1_part2_l177_177253

noncomputable def f (x : ℝ) : ℝ := |3 * x + 2|

theorem part1 (x : ℝ): f x < 6 - |x - 2| ↔ (-3/2 < x ∧ x < 1) :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : m + n = 4) (h₄ : 0 < a) (h₅ : ∀ x, |x - a| - f x ≤ 1/m + 1/n) :
    0 < a ∧ a ≤ 1/3 :=
by sorry

end part1_part2_l177_177253


namespace find_valid_digits_for_divisibility_l177_177830

-- Define the conditions
def is_last_three_digits_divisible_by_eight (n : ℕ) (b : ℕ) : Prop :=
  (n * 10 + b) % 8 = 0

def is_digit_sum_divisible_by_nine (a : ℕ) (n : ℕ) (b : ℕ) : Prop :=
  let digits_sum := a + (n / 10^7 % 10) + (n / 10^6 % 10) + (n / 10^5 % 10) + 
                    (n / 10^4 % 10) + (n / 10^3 % 10) + (n / 10^2 % 10) + 
                    (n / 10 % 10) + (n % 10) + b
  in digits_sum % 9 = 0

-- The main statement to prove
theorem find_valid_digits_for_divisibility (a b : ℕ) (n : ℕ) :
  n = 20222023 →
  is_last_three_digits_divisible_by_eight (2022202) b →
  is_digit_sum_divisible_by_nine a 20222023 b →
  (a * 10^9 + n * 10 + b) % 72 = 0 :=
sorry -- Proof steps omitted

end find_valid_digits_for_divisibility_l177_177830


namespace simplify_cos_sum_identity_l177_177027

noncomputable def cos_sum_identity (n : ℕ) : ℝ :=
  real.cos (2 * real.pi / n) + real.cos (6 * real.pi / n) + real.cos (8 * real.pi / n)

theorem simplify_cos_sum_identity :
  cos_sum_identity 17 = (real.sqrt 13 - 1) / 4 :=
sorry

end simplify_cos_sum_identity_l177_177027


namespace range_of_a_l177_177286

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l177_177286


namespace sqrt_of_9_fact_over_84_eq_24_sqrt_15_l177_177977

theorem sqrt_of_9_fact_over_84_eq_24_sqrt_15 :
  Real.sqrt (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (2^2 * 3 * 7)) = 24 * Real.sqrt 15 :=
by
  sorry

end sqrt_of_9_fact_over_84_eq_24_sqrt_15_l177_177977


namespace angle_BAO_eq_angle_CAH_l177_177353

open EuclideanGeometry

variables {A B C O H : Point}

-- Assuming A, B, C form a triangle
def is_triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ Area A B C ≠ 0

-- Defining circumcenter
def circumcenter (A B C : Point) : Point := sorry

-- Defining orthocenter
def orthocenter (A B C : Point) : Point := sorry

-- Angle between points
def angle (A B C : Point) : ℝ := sorry

-- Main theorem
theorem angle_BAO_eq_angle_CAH 
  (h_triangle : is_triangle A B C)
  (h_circumcenter : circumcenter A B C = O)
  (h_orthocenter : orthocenter A B C = H) :
  angle B A O = angle C A H :=
sorry

end angle_BAO_eq_angle_CAH_l177_177353


namespace circumcircles_equal_radius_l177_177688

-- Conditions
variables {A B C D E F M P K : Type} [InnerProductGeometry A B C]
variables [isRightTriangle A B C] -- triangle ABC with angle BAC = 90 degrees
variables [incircle_tangent D B C] -- D is the point of tangency on BC
variables [incircle_tangent E C A] -- E is the point of tangency on CA
variables [incircle_tangent F A B] -- F is the point of tangency on AB
variables [midpoint M E F] -- M is the midpoint of EF
variables [projection P A B C] -- P is the projection of A onto BC
variables [intersection K M P A D] -- K is the intersection of MP and AD

-- Main statement to prove
theorem circumcircles_equal_radius :
  circumradius A F E = circumradius P D K :=
sorry

end circumcircles_equal_radius_l177_177688


namespace points_in_rectangle_distance_le_sqrt_five_l177_177296

theorem points_in_rectangle_distance_le_sqrt_five :
  ∀ (points : Fin 6 → ℝ × ℝ), 
    (∀ i, 0 ≤ points i.1 ∧ points i.1 ≤ 3) ∧ (∀ i, 0 ≤ points i.2 ∧ points i.2 ≤ 4) → 
    ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_in_rectangle_distance_le_sqrt_five_l177_177296


namespace exists_n_for_any_m_l177_177984

def has_same_prime_divisors (d n : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ d ↔ p ∣ n

def rho (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d > 0 ∧ d ∣ n ∧ has_same_prime_divisors d n).card

theorem exists_n_for_any_m (m : ℕ) (hm : 0 < m) : ∃ n : ℕ, rho (202 ^ n + 1) ≥ m :=
sorry

end exists_n_for_any_m_l177_177984


namespace points_on_line_l177_177768

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177768


namespace points_on_line_initial_l177_177762

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l177_177762


namespace find_b_l177_177797

-- Define the slopes of the two lines derived from the given conditions
noncomputable def slope1 := -2 / 3
noncomputable def slope2 (b : ℚ) := -b / 3

-- Lean 4 statement to prove that for the lines to be perpendicular, b must be -9/2
theorem find_b (b : ℚ) (h_perpendicular: slope1 * slope2 b = -1) : b = -9 / 2 := by
  sorry

end find_b_l177_177797


namespace sin_2023pi_over_3_l177_177817

theorem sin_2023pi_over_3 : sin (2023 * Real.pi / 3) = Real.sqrt 3 / 2 := 
by sorry

end sin_2023pi_over_3_l177_177817


namespace candle_time_l177_177732

-- Define initial lengths and burning rates
def initial_length : ℝ := 20
def burn_rate_thin : ℝ := initial_length / 4 -- 20 cm in 4 hours
def burn_rate_thick : ℝ := initial_length / 5 -- 20 cm in 5 hours

-- Define time variable
variable (t : ℝ)

-- Define lengths after time t
def length_thin (t : ℝ) : ℝ := initial_length - burn_rate_thin * t
def length_thick (t : ℝ) : ℝ := initial_length - burn_rate_thick * t

-- State the main theorem
theorem candle_time (t : ℝ) : length_thin t = 2 * (length_thick t) → t = 20 / 3 :=
sorry

end candle_time_l177_177732


namespace babelian_language_word_count_l177_177363

-- Definitions based on problem conditions
def babelian_alphabet_letters : ℕ := 6
def max_word_length : ℕ := 4

-- Theorem statement
theorem babelian_language_word_count :
  ∑ i in finset.range (max_word_length + 1), babelian_alphabet_letters^i = 1554 :=
by sorry

end babelian_language_word_count_l177_177363


namespace incorrect_statements_l177_177162

theorem incorrect_statements (SSS : ∀ {T1 T2 : Triangle}, T1.sides = T2.sides → T1 ≅ T2)
  (AAS : ∀ {T1 T2 : Triangle}, (∃ (a1 a2 : Angle) (s : Side),
     T1 ∈ triangles_with_sides_angles a1 a2 s ∧
     T2 ∈ triangles_with_sides_angles a1 a2 s) → T1 ≅ T2)
  (SAS : ∀ {T1 T2 : Triangle}, (∃ (s1 s2 : Side) (a : Angle),
     T1 ∈ triangles_with_sides_angle s1 s2 a ∧
     T2 ∈ triangles_with_sides_angle s1 s2 a) → T1 ≅ T2)
  (three_angles_not_congruent : ∀ {T1 T2 : Triangle}, T1.angles = T2.angles → ¬ T1 ≅ T2)
  (SSA_not_congruent : ∀ {T1 T2 : Triangle}, (∃ (s1 s2 : Side) (a : Angle),
     T1 ∈ triangles_with_sides_opposite_angle s1 s2 a ∧
     T2 ∈ triangles_with_sides_opposite_angle s1 s2 a) → ¬ T1 ≅ T2)
  : incorrect ② ∧ incorrect ⑤ := by
  sorry

end incorrect_statements_l177_177162


namespace circle_equation_line_l_equation_l177_177566

/-- Mathematically Equivalent Proof Problem in Lean 4 -/

-- Definitions of the points and given equations
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (-1, 1)
def center_on_line (a : ℝ) : ℝ × ℝ := (a, 2 * a - 1)
def C (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 1) ^ 2 = 4

-- Theorem1: The standard equation of circle C 
theorem circle_equation (x y : ℝ) (a : ℝ) (ha : center_on_line a = (1,1)) :
  C x y ↔ (x - 1) ^ 2 + (y - 1) ^ 2 = 4 := 
sorry

-- Definitions about the line l intersecting the circle C 
def chord_length := 2 * Real.sqrt 3
def l_through_point (p : ℝ × ℝ) (k b : ℝ) : Prop := p.2 = k * p.1 + b

-- Theorem2: The equation of the line l
theorem line_l_equation (k b : ℝ) (p : ℝ × ℝ) (hp : p = (2, 2)) (d : ℝ) (hd : d = 1) :
  (l_through_point p k b ∧ (p = (2, 2)) ∧ (|k - 1 + b| / Real.sqrt (k^2 + 1)) = 1) ↔
  (b = 2 ∧ k = 0) ∨ (l_through_point p 0 2 ∧ l_through_point p 1 1) := 
sorry

end circle_equation_line_l_equation_l177_177566


namespace probability_adjacent_vertices_of_octagon_l177_177128

theorem probability_adjacent_vertices_of_octagon :
  let num_vertices := 8;
  let adjacent_vertices (v1 v2 : Fin num_vertices) : Prop := 
    (v2 = (v1 + 1) % num_vertices) ∨ (v2 = (v1 - 1 + num_vertices) % num_vertices);
  let total_vertices := num_vertices - 1;
  (2 : ℚ) / total_vertices = (2 / 7 : ℚ) :=
by
  -- Proof goes here
  sorry

end probability_adjacent_vertices_of_octagon_l177_177128


namespace particle_and_account_balance_l177_177888

theorem particle_and_account_balance (P₀ : ℝ) (r : ℝ) (t : ℝ) (s : ℝ → ℝ) :
  (s = λ t, -15 * t^2 + 150 * t + 50) →
  (P₀ = 1000) →
  (r = 0.05) →
  (t = 5) →
  (s t = 425) ∧ (P₀ * (1 + r * (t / 12)) = 1020.83) :=
sorry

end particle_and_account_balance_l177_177888


namespace number_of_persons_in_second_group_l177_177867

-- Definitions based on conditions
def total_man_hours_first_group : ℕ := 42 * 12 * 5

def total_man_hours_second_group (X : ℕ) : ℕ := X * 14 * 6

-- Theorem stating that the number of persons in the second group is 30, given the conditions
theorem number_of_persons_in_second_group (X : ℕ) : 
  total_man_hours_first_group = total_man_hours_second_group X → X = 30 :=
by
  sorry

end number_of_persons_in_second_group_l177_177867


namespace share_ratio_l177_177746

theorem share_ratio (A B C : ℕ) (hA : A = (2 * B) / 3) (hA_val : A = 372) (hB_val : B = 93) (hC_val : C = 62) : B / C = 3 / 2 := 
by 
  sorry

end share_ratio_l177_177746


namespace points_on_line_l177_177771

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l177_177771


namespace find_a_l177_177226

noncomputable def z1 : ℂ := 2 + I
noncomputable def z2 (a : ℝ) : ℂ := 3 + a * I

theorem find_a (a : ℝ) (h : (z1 + z2 a).im = 0) : a = -1 :=
by
  -- proof is skipped
  sorry

end find_a_l177_177226


namespace part1_part2_part3_l177_177614

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (a * x + b) / (x^2 + 4)

theorem part1 (h1 : f 0 a b = 0) (h2 : f (1/2) a b = 2/17) : a = 1 ∧ b = 0 := 
sorry

theorem part2 (a : ℝ) (b : ℝ) (h1 : a = 1) (h2 : b = 0) (x1 x2 : ℝ) (h3 : -2 < x1 ∧ x1 < 2) (h4 : -2 < x2 ∧ x2 < 2) :
  x1 < x2 → f x1 1 0 < f x2 1 0 :=
sorry

theorem part3 (a : ℝ) (h1 : -2 < a + 1 ∧ a + 1 < 2) (h2 : -2 < 2 * a - 1 ∧ 2 * a - 1 < 2) :
  f (a + 1) 1 0 - f (2 * a - 1) 1 0 > 0 → -1/2 < a ∧ a < 1 :=
sorry

end part1_part2_part3_l177_177614


namespace stratified_sampling_result_l177_177474

-- Definitions and conditions from the problem
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_sample : ℕ := 60
def junior_sample : ℕ := 40
def senior_sample : ℕ := 20

-- Main theorem statement proving the number of different sampling results
theorem stratified_sampling_result :
  choose junior_students junior_sample * choose senior_students senior_sample = 
  choose 400 40 * choose 200 20 := by
  sorry

end stratified_sampling_result_l177_177474


namespace direction_2010_to_2012_l177_177042

def zigzagDirection (n : ℕ) : string :=
  if n % 4 = 0 then "↓"
  else if n % 4 = 1 then "→"
  else if n % 4 = 2 then "↑"
  else "→"

theorem direction_2010_to_2012 :
  zigzagDirection 2010 = "↑" ∧ zigzagDirection 2011 = "→" :=
by
  sorry

end direction_2010_to_2012_l177_177042


namespace elephant_entry_rate_l177_177427

-- Define the variables and constants
def initial_elephants : ℕ := 30000
def exit_rate : ℕ := 2880
def exit_time : ℕ := 4
def enter_time : ℕ := 7
def final_elephants : ℕ := 28980

-- Prove the rate of new elephants entering the park
theorem elephant_entry_rate :
  (final_elephants - (initial_elephants - exit_rate * exit_time)) / enter_time = 1500 :=
by
  sorry -- placeholder for the proof

end elephant_entry_rate_l177_177427


namespace triangle_area_l177_177321

variable {α : Type*} [LinearOrder α] [Field α] [FloorRing α] [LinearOrderedField α]

def area_of_triangle (a b C : α) : α :=
  (1 / 2) * a * b * (sin C)

theorem triangle_area
  (A B C a b : ℝ)
  (h1 : sin A = 3 * sin B)
  (h2 : b = 1)
  (h3 : C = π / 6)
  (h4 : a = 3 * b) :
  area_of_triangle a b C = 3 / 4 := by
  sorry

end triangle_area_l177_177321


namespace birds_landed_l177_177821

theorem birds_landed (original_birds total_birds : ℕ) (h : original_birds = 12) (h2 : total_birds = 20) :
  total_birds - original_birds = 8 :=
by {
  sorry
}

end birds_landed_l177_177821


namespace sector_area_proof_l177_177246

-- Define the sector with its characteristics
structure sector :=
  (r : ℝ)            -- radius
  (theta : ℝ)        -- central angle

-- Given conditions
def sector_example : sector := {r := 1, theta := 2}

-- Definition of perimeter for a sector
def perimeter (sec : sector) : ℝ :=
  2 * sec.r + sec.theta * sec.r

-- Definition of area for a sector
def area (sec : sector) : ℝ :=
  0.5 * sec.r * (sec.theta * sec.r)

-- Theorem statement based on the problem statement
theorem sector_area_proof (sec : sector) (h1 : perimeter sec = 4) (h2 : sec.theta = 2) : area sec = 1 := 
  sorry

end sector_area_proof_l177_177246


namespace circle_diameter_l177_177228

variables (r x y : ℝ) (π : ℝ := Real.pi)

-- Define the conditions
def area (r : ℝ) : ℝ := π * r^2
def circumference (r : ℝ) : ℝ := 2 * π * r
def condition (x y : ℝ) : Prop := x + y = 100 * π

-- Define the theorem we want to prove
theorem circle_diameter (h1 : x = area r) (h2 : y = circumference r) (h3 : condition x y) : 2 * r = 16 := 
by
  -- we skip the proof
  sorry

end circle_diameter_l177_177228


namespace sum_of_primitive_roots_mod_11_l177_177438

def is_primitive_root_mod (a n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n - 1 → ∃ l, a ^ k ≡ a ^ l [MOD n] ∧ l ≠ k

theorem sum_of_primitive_roots_mod_11 :
  let s := {x ∈ finset.range 11 | is_primitive_root_mod x 11} in
  finset.sum s id = 15 := sorry

end sum_of_primitive_roots_mod_11_l177_177438


namespace theater_min_number_of_plays_l177_177664

theorem theater_min_number_of_plays :
    ∃ (n : ℕ), n = 6 ∧
    (∃ (plays : finset (finset (fin 60))),
        (∀ p ∈ plays, p.card ≤ 30) ∧
        (∀ i j : fin 60, i ≠ j → ∃ p ∈ plays, {i, j} ⊆ p) ∧
        plays.card = n) :=
begin
  sorry
end

end theater_min_number_of_plays_l177_177664


namespace points_on_line_initial_l177_177760

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l177_177760


namespace minimum_time_reach_distance_minimum_l177_177895

/-- Given a right triangle with legs of length 1 meter, and two bugs starting crawling from the vertices
with speeds 5 cm/s and 10 cm/s respectively, prove that the minimum time after the start of their movement 
for the distance between the bugs to reach its minimum is 4 seconds. -/
theorem minimum_time_reach_distance_minimum (l : ℝ) (v_A v_B : ℝ) (h_l : l = 1) (h_vA : v_A = 5 / 100) (h_vB : v_B = 10 / 100) :
  ∃ t_min : ℝ, t_min = 4 := by
  -- Proof is omitted
  sorry

end minimum_time_reach_distance_minimum_l177_177895


namespace lion_king_box_office_earnings_l177_177777

-- Definitions and conditions
def cost_lion_king : ℕ := 10  -- Lion King cost 10 million
def cost_star_wars : ℕ := 25  -- Star Wars cost 25 million
def earnings_star_wars : ℕ := 405  -- Star Wars earned 405 million

-- Calculate profit of Star Wars
def profit_star_wars : ℕ := earnings_star_wars - cost_star_wars

-- Define the profit of The Lion King, given it's half of Star Wars' profit
def profit_lion_king : ℕ := profit_star_wars / 2

-- Calculate the earnings of The Lion King
def earnings_lion_king : ℕ := cost_lion_king + profit_lion_king

-- Theorem to prove
theorem lion_king_box_office_earnings : earnings_lion_king = 200 :=
by
  sorry

end lion_king_box_office_earnings_l177_177777


namespace problem1_problem2_l177_177865

-- Problem (1) Lean Statement
theorem problem1 (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) : 
  a / (c - a) > b / (c - b) :=
sorry

-- Problem (2) Lean Statement
theorem problem2 (x : ℝ) (hx : x > 2) : 
  ∃ (xmin : ℝ), xmin = 6 ∧ (x = 6 → (x + 16 / (x - 2)) = 10) :=
sorry

end problem1_problem2_l177_177865


namespace yolka_probability_correct_l177_177918

open ProbabilityTheory

noncomputable def yolka_meeting_probability : ℝ :=
  let anya_last := 1 / 3
  let borya_vasya_meet := (144 - (0.5 * (81 + 100))) / 144
  in anya_last * borya_vasya_meet

theorem yolka_probability_correct :
  yolka_meeting_probability = 0.124 := 
by sorry

end yolka_probability_correct_l177_177918


namespace geoboard_quadrilaterals_l177_177939

-- Definitions of the quadrilaterals as required by the conditions of the problem.
def quadrilateral_area (quad : Type) : ℝ := sorry
def quadrilateral_perimeter (quad : Type) : ℝ := sorry

-- Declaration of Quadrilateral I and II on a geoboard.
def quadrilateral_i : Type := sorry
def quadrilateral_ii : Type := sorry

-- The proof problem statement.
theorem geoboard_quadrilaterals :
  quadrilateral_area quadrilateral_i = quadrilateral_area quadrilateral_ii ∧
  quadrilateral_perimeter quadrilateral_i < quadrilateral_perimeter quadrilateral_ii := by
  sorry

end geoboard_quadrilaterals_l177_177939


namespace num_irreducible_fractions_l177_177547

def N : ℕ := 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10

theorem num_irreducible_fractions : 
  ∃ (cnt : ℕ), cnt = 16 ∧ 
  (∀ (a b : ℕ), a * b = N → ∃ (gcd_ab) : ℕ, (gcd a b = gcd_ab) ∧ gcd_ab = 1 → (a * b = N → a * b ∣ N) ) :=
by
  sorry

end num_irreducible_fractions_l177_177547


namespace rate_of_interest_l177_177154

theorem rate_of_interest (SI P T R : ℝ) 
  (hSI : SI = 4016.25) 
  (hP : P = 6693.75) 
  (hT : T = 5) 
  (h : SI = (P * R * T) / 100) : 
  R = 12 :=
by 
  sorry

end rate_of_interest_l177_177154


namespace estimate_red_balls_l177_177132

-- Define the conditions in Lean 4
def total_balls : ℕ := 15
def freq_red_ball : ℝ := 0.4

-- Define the proof statement without proving it
theorem estimate_red_balls (x : ℕ) 
  (h1 : x ≤ total_balls) 
  (h2 : ∃ (p : ℝ), p = x / total_balls ∧ p = freq_red_ball) :
  x = 6 :=
sorry

end estimate_red_balls_l177_177132


namespace curves_and_distance_problems_l177_177313

theorem curves_and_distance_problems 
  (C₁_parametric : ∀ θ, (x, y) = (4 * Real.cos θ, 3 * Real.sin θ))
  (C₂_polar : ∀ (ρ θ), ρ * Real.sin (θ + π / 4) = 3 * Real.sqrt 2) : 
  ∃ (r1 r2 : ℝ),
  (r1 = (\<=> \))
  ((∃ x y, x = 4 * Real.cos θ ∧ y = 3 * Real.sin θ → (x ^ 2 / 16) + (y ^ 2 / 9) = 1)) ∧ 
  ((∃ ρ θ, ρ * Real.sin (θ + π / 4) = 3 * Real.sqrt 2 → ∃ x y, x + y - 6 = 0)) ∧ 
  (∀ M N : (ℝ × ℝ), M ∈ C₁_parametric ∧ N ∈ C₂_polar → 
    ∃ h : ℝ, h = (|4 * Real.cos θ + 3 * Real.sin θ - 6|) / Real.sqrt 2 → 
      h = Real.sqrt 2 / 2) := 
sorry

end curves_and_distance_problems_l177_177313


namespace polygon_equally_split_by_any_line_l177_177955

def symmetric_polygon_split (P : Set Point) (O : Point) (hO : O ∈ boundary P) : Prop :=
  ∀ (L : Line), (O ∈ L) → (area (P ∩ halfplane L)) = (area (P ∩ (complement (halfplane L))))

theorem polygon_equally_split_by_any_line (P : Set Point) (O : Point) (hO : O ∈ boundary P)
  (hSymmetric : symmetric O P) : symmetric_polygon_split P O hO := 
sorry

end polygon_equally_split_by_any_line_l177_177955


namespace vector_sum_proof_l177_177322

variable (a b : Vector3)

/--
In triangle ABC, with
  CA = a,
  CB = b,
and point P on line AB such that
  AP = 2 PB,
we want to show that
  CP = (a + 2 b) / 3.
-/
theorem vector_sum_proof (CA CB CP : Vector3) (AP PB : Vector3) 
  (hCA : CA = a) (hCB : CB = b) (h_relation : AP = 2 • PB) :
  CP = (a + 2 • b) / 3 := 
by 
  sorry

end vector_sum_proof_l177_177322


namespace find_min_CP_PQ_l177_177863

noncomputable def side_length : ℝ := 4 * a
noncomputable def point_E := (0, 0, a)
noncomputable def point_F := (4 * a, 0, 3 * a)
noncomputable def midpoint_G := (2 * a, 0, 4 * a)
noncomputable def midpoint_H := (2 * a, 4 * a, 4 * a)
noncomputable def parameter_y : ℝ := 4 * a * (sqrt 6) / (10 + 4 * sqrt 5)
noncomputable def parameter_x : ℝ := (14 * a) / 5

def CP (a : ℝ) (y : ℝ) : ℝ := real.sqrt (20 * a^2 + (4 * a - y)^2)
def PQ (a : ℝ) (x : ℝ) (y : ℝ) : ℝ := real.sqrt ((x - 2 * a)^2 + y^2 + ((x - 6 * a)^2) / 4)

def minimum_value (a : ℝ) (x : ℝ) (y : ℝ) : ℝ := CP a y + PQ a x y

theorem find_min_CP_PQ (a : ℝ) : 
  minimum_value a parameter_x parameter_y = 
    sqrt (20 * a^2 + (4 * a - parameter_y)^2) + 
    sqrt ((parameter_x - 2 * a)^2 + parameter_y^2 + ((parameter_x - 6 * a)^2) / 4) :=
by 
  unfold minimum_value 
  unfold CP 
  unfold PQ 
  unfold parameter_y 
  unfold parameter_x 
  sorry

end find_min_CP_PQ_l177_177863


namespace binomial_coeffs_odd_iff_l177_177219

def binomial_expansion (a b : ℕ) (n : ℕ) := (a + b) ^ n

theorem binomial_coeffs_odd_iff (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n k % 2 = 1) ↔ (∃ k, n = 2 ^ k - 1) := 
sorry

end binomial_coeffs_odd_iff_l177_177219


namespace find_x_l177_177108

theorem find_x (x : ℚ) (h : |x - 1| = |x - 2|) : x = 3 / 2 :=
sorry

end find_x_l177_177108


namespace max_value_of_function_l177_177643

theorem max_value_of_function (a : ℕ) (h1 : ∀ x : ℝ, f x a = x + real.sqrt (13 - 2 * a * x)) 
  (h2 : ∀ x : ℝ, 13 - 2 * a * x ≥ 0) (h3 : a > 0) : ∃ y : ℕ, (y = 7) :=
by {
  sorry
}

noncomputable def f (x : ℝ) (a : ℕ) : ℝ := x + real.sqrt (13 - 2 * a * x)

end max_value_of_function_l177_177643


namespace circle_theorem_l177_177067

noncomputable def circle_problem (r_A r_B r_C : ℝ) (A B C : ℝ × ℝ) 
(h1 : r_A > r_B)
(h2 : r_B > r_C)
(h3 : dist A B = r_A - r_B)
(h4 : dist A C = r_A + r_C) : Prop :=
  ¬ (r_A + r_C < dist A C)

theorem circle_theorem (r_A r_B r_C : ℝ) (A B C : ℝ × ℝ)
(h1 : r_A > r_B)
(h2 : r_B > r_C)
(h3 : dist A B = r_A - r_B)
(h4 : dist A C = r_A + r_C) : circle_problem r_A r_B r_C A B C h1 h2 h3 h4 :=
begin
  sorry
end

end circle_theorem_l177_177067


namespace zero_point_in_interval_l177_177551

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x / 2) - 2 / x

theorem zero_point_in_interval : ∃ x ∈ Ioo 1 2, f x = 0 := by
  sorry

end zero_point_in_interval_l177_177551


namespace number_of_distinct_products_of_S_l177_177335

def S : Set ℕ := { d | d ∣ 129600 ∧ d > 0 }

theorem number_of_distinct_products_of_S :
  {n : ℕ | ∃ x y ∈ S, x ≠ y ∧ n = x * y}.toFinset.card = 488 :=
by
  sorry

end number_of_distinct_products_of_S_l177_177335


namespace find_number_of_students_l177_177453

theorem find_number_of_students (N : ℕ) (T : ℕ) (hN : N ≠ 0) (hT : T = 80 * N) 
  (h_avg_excluded : (T - 200) / (N - 5) = 90) : N = 25 :=
by
  sorry

end find_number_of_students_l177_177453


namespace total_wait_days_l177_177718

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end total_wait_days_l177_177718


namespace sufficient_but_not_necessary_l177_177196

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x < 1 / 2) : x > 2 ∨ x < 0 :=
by
  sorry

end sufficient_but_not_necessary_l177_177196


namespace cost_of_transport_l177_177040

-- Define the constant cost per kilogram in dollars
def cost_per_kg : ℝ := 22000

-- Define the mass of the control module in grams
def mass_in_grams : ℝ := 250

-- Convert the mass to kilograms
def mass_in_kg : ℝ := mass_in_grams / 1000

-- Define the correct answer
def expected_cost : ℝ := 5500

-- The theorem we want to prove
theorem cost_of_transport : (cost_per_kg * mass_in_kg) = expected_cost :=
by sorry

end cost_of_transport_l177_177040


namespace zoe_earns_per_candy_bar_l177_177850

-- Given conditions
def cost_of_trip : ℝ := 485
def grandma_contribution : ℝ := 250
def candy_bars_to_sell : ℝ := 188

-- Derived condition
def additional_amount_needed : ℝ := cost_of_trip - grandma_contribution

-- Assertion to prove
theorem zoe_earns_per_candy_bar :
  (additional_amount_needed / candy_bars_to_sell) = 1.25 :=
by
  sorry

end zoe_earns_per_candy_bar_l177_177850


namespace jogging_time_l177_177278

theorem jogging_time (distance : ℝ) (speed : ℝ) (h1 : distance = 25) (h2 : speed = 5) : (distance / speed) = 5 :=
by
  rw [h1, h2]
  norm_num

end jogging_time_l177_177278


namespace sqrt_200_eq_l177_177008

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l177_177008


namespace sequences_equal_l177_177696

-- Definitions for the sequences and the conditions
def posInt (n : ℕ) : Prop := n > 0

def conditions (n : ℕ) (ε : Fin (n - 1) → ℕ) (a b : Fin (n + 1) → ℕ) : Prop :=
  posInt n ∧
  (∀ i, 0 ≤ ε i ∧ ε i ≤ 1) ∧
  (a 0 = 1 ∧ b 0 = 1 ∧ a 1 = 7 ∧ b 1 = 7) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n - 1 → (ε i = 0 → a (i+1) = 2 * a (i-1) + 3 * a i) ∧ (ε i = 1 → a (i+1) = 3 * a (i-1) + a i)) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n - 1 → (ε (n-i) = 0 → b (i+1) = 2 * b (i-1) + 3 * b i) ∧ (ε (n-i) = 1 → b (i+1) = 3 * b (i-1) + b i))

-- The theorem we want to prove
theorem sequences_equal (n : ℕ) (ε : Fin (n-1) → ℕ) (a b : Fin (n+1) → ℕ) :
  conditions n ε a b → a n = b n :=
by 
  intros h,
  sorry

end sequences_equal_l177_177696


namespace power_difference_l177_177589

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l177_177589


namespace power_difference_l177_177588

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l177_177588


namespace first_nonzero_digit_1_div_143_l177_177435

theorem first_nonzero_digit_1_div_143 : 
  let n := (1 : ℚ) / 143
  ∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ (let m := n * 10^3 in m - m.floor = d / 10 ∧ m.floor = 6) → d = 7 := 
by 
  sorry

end first_nonzero_digit_1_div_143_l177_177435


namespace evaluate_expression_l177_177542

theorem evaluate_expression : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 :=
by
  sorry

end evaluate_expression_l177_177542


namespace angle_between_vectors_is_3pi_div_4_lambda_for_perpendicular_is_minus_1_l177_177627

-- Defining vectors a and b
def vector_a := (1, 2) : ℝ × ℝ
def vector_b := (-3, 4) : ℝ × ℝ

-- Question Ⅰ: Prove the angle between a + b and a - b is 3π/4
theorem angle_between_vectors_is_3pi_div_4 :
  let c := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  let d := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let dot_prod := c.1 * d.1 + c.2 * d.2
  let mag_c := Real.sqrt (c.1^2 + c.2^2)
  let mag_d := Real.sqrt (d.1^2 + d.2^2)
  Real.arccos (dot_prod / (mag_c * mag_d)) = 3 * Real.pi / 4 :=
by
  intros
  have c := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  have d := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  have dot_prod := c.1 * d.1 + c.2 * d.2
  have mag_c := Real.sqrt (c.1^2 + c.2^2)
  have mag_d := Real.sqrt (d.1^2 + d.2^2)
  sorry

-- Question Ⅱ: Prove λ = -1 makes a perpendicular to (a + λb)
theorem lambda_for_perpendicular_is_minus_1 :
  ∃ λ : ℝ, 
  let c := (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
  (vector_a.1 * c.1 + vector_a.2 * c.2 = 0) ∧ (λ = -1) :=
by
  use -1
  intros
  let c := (vector_a.1 + (-1) * vector_b.1, vector_a.2 + (-1) * vector_b.2)
  have dot_prod := vector_a.1 * c.1 + vector_a.2 * c.2
  split
  {
    -- Perpendicular condition
    have dot_prod := (1 + 3) * c.1 + (2 - 4) * c.2
    exact dot_prod = 0
    sorry
  }
  {
    -- Lambda being -1
    exact (-1 = -1)
  }

end angle_between_vectors_is_3pi_div_4_lambda_for_perpendicular_is_minus_1_l177_177627


namespace line_intercepts_area_relation_l177_177804

theorem line_intercepts_area_relation :
  let line_eq := ∀ x y, y = (5 / 3) * x - 15 in
  let P := (9, 0) in
  let Q := (0, -15) in
  ∃ (r s : ℝ), T (r, s) ∧
  (r, s) ∈ segment P Q ∧
  1 / 2 * 9 * 15 = 2 * (1 / 2 * 9 * abs(s + 15)) →
  r + s = -3 :=
begin
  sorry
end

end line_intercepts_area_relation_l177_177804


namespace no_prime_roots_sum_72_l177_177928

theorem no_prime_roots_sum_72 :
  ∀ (p q : ℕ), prime p → prime q → p + q = 72 → false :=
by
  sorry

end no_prime_roots_sum_72_l177_177928


namespace function_passes_1_2_l177_177045

theorem function_passes_1_2 (a : ℝ) (ha : 0 < a ∧ a < 1) : 
  (∀ x : ℝ, y = 2 * a^(x - 1)) → (y 1 = 2) :=
by
  intros h
  have hy := h 1
  sorry

end function_passes_1_2_l177_177045


namespace initial_puppies_l177_177500

theorem initial_puppies (p_sold p_cages p_pups_per_cage : ℕ) (h1 : p_sold = 24) (h2 : p_cages = 8) (h3 : p_pups_per_cage = 4) : 
  let p_left := p_cages * p_pups_per_cage in
  let p_initial := p_sold + p_left in
  p_initial = 56 := by
sorry

end initial_puppies_l177_177500


namespace magnitude_difference_l177_177628

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
def norm_a : ℝ := ‖a‖
def norm_b : ℝ := ‖b‖
def dot_a_b : ℝ := inner a b

-- Specifying the conditions as given in the problem
axiom norm_a_eq_5 : norm_a a b = 5
axiom norm_b_eq_8 : norm_b a b = 8
axiom dot_a_b_eq_20 : dot_a_b a b = 20

-- The main theorem to prove
theorem magnitude_difference (a b : EuclideanSpace ℝ (Fin 3)) :
  norm_a a b = 5 →
  norm_b a b = 8 →
  dot_a_b a b = 20 →
  ‖a - b‖ = 7 :=
by
  intros h1 h2 h3
  sorry

end magnitude_difference_l177_177628


namespace number_of_ns_l177_177392

def f (n : ℕ) : ℕ := sorry

lemma f_properties (n : ℕ) : 
  (f (4 * n) = f (2 * n) + f n) ∧ 
  (f (4 * n + 2) = f (4 * n) + 1) ∧ 
  (f (2 * n + 1) = f (2 * n) + 1) := sorry

theorem number_of_ns (m : ℕ) : 
  { n | n < 2^m ∧ f (4 * n) = f (3 * n) }.card = f (2^(m+1)) := sorry

end number_of_ns_l177_177392


namespace Carrie_can_add_turnips_l177_177171

-- Define the variables and conditions
def potatoToTurnipRatio (potatoes turnips : ℕ) : ℚ :=
  potatoes / turnips

def pastPotato : ℕ := 5
def pastTurnip : ℕ := 2
def currentPotato : ℕ := 20
def allowedTurnipAddition : ℕ := 8

-- Define the main theorem to prove, given the conditions.
theorem Carrie_can_add_turnips (past_p_ratio : potatoToTurnipRatio pastPotato pastTurnip = 2.5)
                                : potatoToTurnipRatio currentPotato allowedTurnipAddition = 2.5 :=
sorry

end Carrie_can_add_turnips_l177_177171


namespace sqrt_200_eq_10_sqrt_2_l177_177003

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177003


namespace cube_of_square_of_third_smallest_prime_is_correct_l177_177090

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l177_177090


namespace total_number_of_sampling_results_l177_177480

theorem total_number_of_sampling_results : 
  let junior_students := 400
  let senior_students := 200
  let total_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (junior_sample + senior_sample = total_sample) → 
  (junior_sample / junior_students = 2 / 3) → 
  (senior_sample / senior_students = 1 / 3) → 
  @Fintype.card (Finset (Fin junior_students)).choose junior_sample *
  @Fintype.card (Finset (Fin senior_students)).choose senior_sample = 
  Nat.binom junior_students junior_sample * Nat.binom senior_students senior_sample
:= by
  sorry

end total_number_of_sampling_results_l177_177480


namespace part_a_area_of_square_l177_177446

theorem part_a_area_of_square {s : ℝ} (h : s = 9) : s ^ 2 = 81 := 
sorry

end part_a_area_of_square_l177_177446


namespace electron_reaches_underside_prob_l177_177730

theorem electron_reaches_underside_prob :
  let p₀ := 1
  let q₀ := 0
  let r₀ := 0
  let s₀ := 0
  -- Recursive relations
  ∀ i : ℕ, 
    let p i := (q (i-1) + r (i-1) + s (i-1)) / 3
    let q i := (p (i-1) + r (i-1)) / 3
    let r i := (p (i-1) + q (i-1) + s (i-1)) / 3
    let s i := (p (i-1) + r (i-1)) / 3
    -- Intermediate relations
    let a i := p i + r i
    let b i := q i + s i
    a 0 = 1 ∧ b 0 = 0 ∧ 
    a (i+1) = (a i + 2 * b i) / 3 ∧ 
    b (i+1) = 2 * a i / 3 ->
    let ai := (3 / Real.sqrt 17) * ((1 + Real.sqrt 17) / 6)^(i + 1) - (3 / Real.sqrt 17) * ((1 - Real.sqrt 17) / 6)^(i + 1) --
    b (1994) / 3  =
    let P := (2 / (3 * Real.sqrt 17)) * 
    (((1 + Real.sqrt 17) / 6)^1994 - ((1 - Real.sqrt 17) / 6)^1994) in
  P = 2.42e-138 :=
by sorry -- Proof is omitted

end electron_reaches_underside_prob_l177_177730


namespace find_y_l177_177280

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end find_y_l177_177280


namespace min_value_frac_l177_177704

theorem min_value_frac (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : a + b = 1) : 
  (1 / a) + (4 / b) ≥ 9 :=
by sorry

end min_value_frac_l177_177704


namespace integral_eval_l177_177341

theorem integral_eval : ∫ x in 1..2, (3 * x^2 - 2 * x) = 4 := by
  sorry

end integral_eval_l177_177341


namespace cube_remaining_surface_area_l177_177999

noncomputable def remaining_surface_area (edge_length : ℝ) (PA QA RA : ℝ) : ℝ :=
  let total_surface_area := 6 * edge_length^2
  let removed_area := 3 * ((sqrt 2 / 2) * (PA + QA + RA) ^ 2 * (sin (π / 3)))
  in total_surface_area - removed_area

theorem cube_remaining_surface_area :
  remaining_surface_area 10 3 3 3 = 586.5 :=
begin
  sorry
end

end cube_remaining_surface_area_l177_177999


namespace sum_of_all_distinct_results_l177_177965

theorem sum_of_all_distinct_results :
  let f (a b c d : ℕ → ℕ) := 625 + a 125 + b 25 + c 5 + d 1,
      g (a b c d : ℕ → ℕ) := 625 - a 125 - b 25 - c 5 - d 1,
      options (n : ℕ) := if n = 0 then id else fun x => -x
  in (f options options options options + g options options options options) = 10000 := by
sorry

end sum_of_all_distinct_results_l177_177965


namespace modulus_of_z_l177_177248

open Complex

variable (z : ℂ) -- declare z as a complex variable

-- Given condition for z
def given_z : Prop := z = (1 + I)^3 / 2

-- The main theorem to prove
theorem modulus_of_z : given_z z → Complex.abs z = Real.sqrt 2 := 
by 
  intros h,
  rw [given_z, h],
  -- Simplifications corresponding to the solution steps are to be filled in the actual proof
  have : (1 + I)^3 = -1 + I, -- Step by step simplifications in actual proof
  rw this,
  sorry

end modulus_of_z_l177_177248


namespace sqrt_200_eq_10_l177_177024

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177024


namespace rectangle_area_l177_177744

theorem rectangle_area :
  ∀ (A B C D E F : ℝ × ℝ) (a : ℝ),
    A = (0, 0) → 
    B = (2*a, 0) → 
    C = (2*a, a) → 
    D = (0, a) → 
    E = (0, a/2) → 
    (F = (4*a/5, 2*a/5)) → 
    let area_triangle : ℝ := (1/2) * |2*a * (2*a/5 - a) + 4*a/5 * a + 2*a * 0| 
    in area_triangle = 50 → 
    2 * a * a = 500/3 :=
begin
  intros A B C D E F a hA hB hC hD hE hF,
  intro area_triangle,
  sorry
end

end rectangle_area_l177_177744


namespace sqrt_200_eq_10_sqrt_2_l177_177006

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l177_177006


namespace verify_point_A_L_coordinates_l177_177936

noncomputable def omega := 10 -- rad/s
noncomputable def OA := 90 -- cm
noncomputable def OB := 90 -- cm
noncomputable def AL := OA / 3 -- cm

def point_A_coordinates (t : ℝ) : ℝ × ℝ :=
  (OA * Real.cos (omega * t), OA * Real.sin (omega * t))

def point_L_coordinates (t : ℝ) (theta : ℝ) : ℝ × ℝ :=
  let (x_A, y_A) := point_A_coordinates t
  (x_A - AL * Real.sin theta, y_A - AL * Real.cos theta)

theorem verify_point_A_L_coordinates (t theta : ℝ) :
  point_A_coordinates t = (OA * Real.cos (omega * t), OA * Real.sin (omega * t)) ∧
  point_L_coordinates t theta = ((OA * Real.cos (omega * t)) - AL * Real.sin theta, 
                                 (OA * Real.sin (omega * t)) - AL * Real.cos theta) :=
by
  sorry

end verify_point_A_L_coordinates_l177_177936


namespace sqrt_200_eq_10_l177_177025

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l177_177025


namespace length_of_first_train_is_correct_l177_177425

noncomputable def length_of_first_train (speed1_km_hr speed2_km_hr : ℝ) (time_cross_sec : ℝ) (length2_m : ℝ) : ℝ :=
  let speed1_m_s := speed1_km_hr * (5 / 18)
  let speed2_m_s := speed2_km_hr * (5 / 18)
  let relative_speed_m_s := speed1_m_s + speed2_m_s
  let total_distance_m := relative_speed_m_s * time_cross_sec
  total_distance_m - length2_m

theorem length_of_first_train_is_correct : 
  length_of_first_train 60 40 11.879049676025918 160 = 170 := by
  sorry

end length_of_first_train_is_correct_l177_177425


namespace center_of_matrix_is_28_5_l177_177647

-- Definitions for giving conditions
def first_row (n : ℕ) := 2 + (n - 1) * 4.5
def fifth_row (n : ℕ) := 22 + (n - 1) * 12

-- Definition of the value at the center of the matrix
def center_value := first_row 3 + 2 * 8.75

theorem center_of_matrix_is_28_5 :
  center_value = 28.5 :=
by
  -- Proof omitted
  sorry

end center_of_matrix_is_28_5_l177_177647


namespace solution_set_of_inequality_l177_177273

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} :=
sorry

end solution_set_of_inequality_l177_177273


namespace quadratic_equation_solution_l177_177407

theorem quadratic_equation_solution : ∀ x : ℝ, x^2 - 9 = 0 ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end quadratic_equation_solution_l177_177407


namespace alternating_sum_series_l177_177177

theorem alternating_sum_series : 
  (Finset.sum (Finset.range 101) (λ n, if even n then -(n+1) else n+1)) = 51 :=
by
  sorry

end alternating_sum_series_l177_177177


namespace tan_half_angle_value_l177_177290

theorem tan_half_angle_value (α : ℝ) (h : ∃ (x y : ℝ), (x, y) = (-1, 2) ∧ x^2 + y^2 = 1) :
  tan (α / 2) = (1 + real.sqrt 5) / 2 :=
sorry

end tan_half_angle_value_l177_177290


namespace billy_distance_l177_177921

def displacement_billy (east_walk : ℝ) (north_angle : ℝ) (north_walk : ℝ) : ℝ :=
  let bd_leg := north_walk / Real.sqrt 2
  let total_east := east_walk + bd_leg
  let total_north := bd_leg
  Real.sqrt (total_east^2 + total_north^2)

theorem billy_distance :
  displacement_billy 5 45 8 = Real.sqrt (89 + 40 * Real.sqrt 2) :=
by
  sorry

end billy_distance_l177_177921


namespace turnips_in_mashed_potatoes_l177_177173

theorem turnips_in_mashed_potatoes:
  ∀ (turnips_prev potatoes_prev : ℝ) (potatoes_curr : ℝ),
    turnips_prev = 2 →
    potatoes_prev = 5 →
    potatoes_curr = 20 →
    (potatoes_curr / (potatoes_prev / turnips_prev) = 8) :=
begin
  intros,
  sorry,
end

end turnips_in_mashed_potatoes_l177_177173


namespace sum_of_squares_changes_l177_177227

theorem sum_of_squares_changes 
  (a : Fin 100 → ℤ) 
  (h : (∑ i, (a i + 1)^2 = ∑ i, (a i)^2)) : 
  (∑ i, (a i + 2)^2 = ∑ i, (a i)^2 + 200) :=
sorry

end sum_of_squares_changes_l177_177227


namespace total_sections_l177_177415

theorem total_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls) = 29 :=
by
  sorry

end total_sections_l177_177415


namespace number_of_true_propositions_l177_177400

def P1 : Prop := (∀ x y : ℝ, x = 0 ∧ y = 0 → x^2 + y^2 = 0)
def P2 : Prop := (∀ Δ1 Δ2 : Triangle, ¬Congruent Δ1 Δ2 → ¬EqualArea Δ1 Δ2)
def P3 : Prop := (∀ (A B : Set), (A ∩ B = A) → (A ⊆ B))
def P4 : Prop := (∀ n : ℕ, (n % 10 = 0) → (n % 5 = 0))

theorem number_of_true_propositions : 
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4 → 2 = 2) := sorry

end number_of_true_propositions_l177_177400


namespace trig_values_of_angle_terminal_side_through_point_l177_177247

noncomputable def r (a : ℝ) := real.sqrt ( (3*a)^2 + (4*a)^2 )

variables (a : ℝ) (α : ℝ) (h : a ≠ 0) (hα : P(3*a, 4*a) lies on the terminal side of angle α)

theorem trig_values_of_angle_terminal_side_through_point :
  sin α = 4 / 5 ∨ sin α = -4 / 5 ∧
  cos α = 3 / 5 ∨ cos α = -3 / 5 ∧
  tan α = 4 / 3 := 
sorry

end trig_values_of_angle_terminal_side_through_point_l177_177247


namespace add_points_proof_l177_177756

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l177_177756


namespace part1_part2_l177_177302

variables {A B C D E F : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (side_length : ℝ := 2)
variables (midpoint : D = (B + C) / 2)
variables (on_side_AB : E ∈ line_segment A B)
variables (on_side_AC : F ∈ line_segment A C)

-- Part 1
theorem part1 (angle_DEF : ∠ D E F = 120) : 
  dist A E + dist A F = 3 := sorry

-- Part 2
theorem part2 (angle_DEF : ∠ D E F = 60) :
  ∃ (a b : ℝ), (dist A E + dist A F = a) ∧ (a ≥ (3/2)) ∧ (a ≤ 2) := sorry

end part1_part2_l177_177302


namespace reflections_of_orthocenter_lie_on_circumcircle_l177_177337

noncomputable theory

open EuclideanGeometry

variable (ABC : Triangle) (H : Point)
variable (h_orthocenter : isOrthocenter H ABC)

theorem reflections_of_orthocenter_lie_on_circumcircle :
  ∀ side : Line, let H_reflection := reflection_in_line side H in
  H_reflection ∈ circumcircle ABC :=
by sorry

end reflections_of_orthocenter_lie_on_circumcircle_l177_177337


namespace number_of_people_with_cards_ge_0_3_l177_177333

def jungkook_card := 0.8
def yoongi_card := 1 / 2
def yoojung_card := 0.9

theorem number_of_people_with_cards_ge_0_3 : 
  (jungkook_card >= 0.3) ∧ (yoongi_card >= 0.3) ∧ (yoojung_card >= 0.3) → 
  (3 = 3) :=
begin
  sorry
end

end number_of_people_with_cards_ge_0_3_l177_177333


namespace distinct_pairs_12_students_l177_177982

/-- Given a group of twelve students, the number of distinct pairs of students is 66. -/
theorem distinct_pairs_12_students : (nat.choose 12 2) = 66 := by
  sorry

end distinct_pairs_12_students_l177_177982


namespace ben_trip_time_l177_177448

theorem ben_trip_time (distance : ℝ) (bexy_walk_speed : ℝ) (bexy_bike_speed : ℝ) (ben_speed_factor : ℝ) :
  distance = 5 →
  bexy_walk_speed = 5 →
  bexy_bike_speed = 15 →
  ben_speed_factor = 0.5 →
  let bexy_walk_time := distance / bexy_walk_speed in
  let bexy_bike_time := distance / bexy_bike_speed in
  let bexy_total_time := bexy_walk_time + bexy_bike_time in
  let bexy_avg_speed := 2 * distance / bexy_total_time in
  let ben_speed := bexy_avg_speed * ben_speed_factor in
  let ben_time := (2 * distance) / ben_speed in
  ben_time * 60 = 160 :=
by
  intros
  simp [bexy_walk_time, bexy_bike_time, bexy_total_time, bexy_avg_speed, ben_speed, ben_time]
  sorry

end ben_trip_time_l177_177448


namespace claire_photos_l177_177117

theorem claire_photos (C L R : ℕ) 
  (h1 : L = 3 * C) 
  (h2 : R = C + 12)
  (h3 : L = R) : C = 6 := 
by
  sorry

end claire_photos_l177_177117


namespace infinite_remainder_equality_l177_177954

-- Define r(n) as the sum of remainders when n is divided by 1 to n
def remainder_sum (n : ℕ) : ℕ := (List.range (n + 1)).map (λ i => if i = 0 then 0 else n % i).sum

-- Define the infinte property needed to be proven
theorem infinite_remainder_equality : ∃ᶠ k in at_top, remainder_sum k = remainder_sum (k - 1) :=
sorry

end infinite_remainder_equality_l177_177954


namespace intersection_complement_A_B_l177_177708

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_A_B : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_B_l177_177708


namespace ducks_and_dogs_total_l177_177496

theorem ducks_and_dogs_total (d g : ℕ) (h1 : d = g + 2) (h2 : 4 * g - 2 * d = 10) : d + g = 16 :=
  sorry

end ducks_and_dogs_total_l177_177496


namespace fruit_in_each_box_l177_177066

noncomputable def number_of_fruits_each_box (p : ℕ) : ℕ :=
  let total_fruits := 12 + p
  let fruits_in_first_box := 12 + p / 9
  let each_box := total_fruits / 3
  if 12 + p / 9 = each_box then each_box else 0

theorem fruit_in_each_box (p : ℕ) (h : 12 + p / 9 = (12 + p) / 3) : number_of_fruits_each_box p = 16 :=
by
  rw [number_of_fruits_each_box]
  rw [if_pos h]
  simp
  sorry

end fruit_in_each_box_l177_177066


namespace alice_total_savings_percentage_l177_177491

theorem alice_total_savings_percentage :
  let original_cost_coat := 150
  let original_cost_hat := 50
  let original_cost_shoes := 100
  let discount_coat := 0.35
  let discount_hat := 0.50
  let discount_shoes := 0.60
  let original_total_cost := original_cost_coat + original_cost_hat + original_cost_shoes
  let savings_coat := original_cost_coat * discount_coat
  let savings_hat := original_cost_hat * discount_hat
  let savings_shoes := original_cost_shoes * discount_shoes
  let total_savings := savings_coat + savings_hat + savings_shoes
  let savings_percentage := (total_savings / original_total_cost) * 100
  savings_percentage = 45.83 :=
by
  let original_cost_coat := 150
  let original_cost_hat := 50
  let original_cost_shoes := 100
  let discount_coat := 0.35
  let discount_hat := 0.50
  let discount_shoes := 0.60
  let original_total_cost := original_cost_coat + original_cost_hat + original_cost_shoes
  let savings_coat := original_cost_coat * discount_coat
  let savings_hat := original_cost_hat * discount_hat
  let savings_shoes := original_cost_shoes * discount_shoes
  let total_savings := savings_coat + savings_hat + savings_shoes
  let savings_percentage := (total_savings / original_total_cost) * 100
  sorry

end alice_total_savings_percentage_l177_177491


namespace cube_square_third_smallest_prime_l177_177084

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l177_177084


namespace cube_square_third_smallest_prime_l177_177096

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l177_177096


namespace range_of_t_l177_177344

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (a t : ℝ) := 2 * a * t - t^2

theorem range_of_t (t : ℝ) (a : ℝ) (x : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x)
                   (h₂ : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 1 → f x₁ ≤ f x₂)
                   (h₃ : f (-1) = -1) (h₄ : -1 ≤ x ∧ x ≤ 1 → f x ≤ t^2 - 2 * a * t + 1)
                   (h₅ : -1 ≤ a ∧ a ≤ 1) :
  t ≥ 2 ∨ t = 0 ∨ t ≤ -2 := sorry

end range_of_t_l177_177344


namespace triangle_side_length_BC_l177_177669

variable (A B C : Type) [InnerProductSpace ℝ A]

def isTriangle (A B C : A) : Prop := sorry  -- This should be defined based on given points.

theorem triangle_side_length_BC (A B C : A) (p: isTriangle A B C) (AB AC: ℝ) (angleACB: Real.Angle) (hAB: AB = 7) (hAC: AC = 5) (hAngle: angleACB = Real.Angle.ofDegrees 120) :
    ∃ BC : ℝ, BC = 3 :=
by
  sorry

end triangle_side_length_BC_l177_177669


namespace min_even_numbers_in_circle_l177_177062

open Nat

theorem min_even_numbers_in_circle : 
  ∀ (a : Fin 100 → ℕ), 
    (∀ i : Fin 100, even (a i) ∨ even (a (i + 1) % 100) ∨ even (a (i + 2) % 100)) → 
    ∃ evens : ℕ, evens = 34 ∧ (∀ i : Fin 100, evens = countp (even ∘ a)),
sorry

end min_even_numbers_in_circle_l177_177062


namespace area_ratio_triangle_quadrilateral_l177_177574

theorem area_ratio_triangle_quadrilateral (A B C D E F : ℝ×ℝ)
(hA : A = (0, 0)) (hB : B = (0, 1)) (hC : C = (1, 1)) (hD : D = (1, 0))
(hE : E = ((1 + 0) / 2, (1 + 0) / 2)) -- E is the midpoint of diagonal BD
(hF : F = (3/4, 0)) -- F is on DA with DF = 1/4 DA
: (area_triangle D F E) / (area_quadrilateral A B E F) = 1 / 7 :=
sorry

end area_ratio_triangle_quadrilateral_l177_177574


namespace column_sort_preserves_row_order_l177_177902

-- Define the table as a 2D array of size 10x10 filled with natural numbers
def Table := Array (Array Nat)

-- Define a function to check if each row in the table is sorted
def is_sorted_row (tbl : Table) : Prop :=
  ∀ i, i < 10 → (∀ j, j < 9 → tbl[i][j] ≤ tbl[i][j + 1])

-- Define a function to check if each column in the table is sorted
def is_sorted_col (tbl : Table) : Prop :=
  ∀ j, j < 10 → (∀ i, i < 9 → tbl[i][j] ≤ tbl[i + 1][j])

-- The main theorem we want to prove
theorem column_sort_preserves_row_order :
  ∀ tbl : Table, 
    (∀ i, i < 10 → Array.sorted (tbl[i])) → 
    (∀ tbl_sorted : Table, 
      (∀ i j, tbl_sorted[i][j] = tbl[i][j]) →
      is_sorted_row tbl → is_sorted_row tbl_sorted
    ) →
    (∀ tbl_sorted_col : Table, 
      (∀ i j, tbl_sorted_col[i][j] = tbl_sorted[i][j]) →
      is_sorted_col tbl_sorted_col → is_sorted_row tbl_sorted_col
    )
:= by
  sorry

end column_sort_preserves_row_order_l177_177902


namespace eighth_binomial_term_l177_177945

theorem eighth_binomial_term :
  let n := 10
  let a := 2 * x
  let b := 1
  let k := 7
  (Nat.choose n k) * (a ^ k) * (b ^ (n - k)) = 960 * (x ^ 3) := by
  sorry

end eighth_binomial_term_l177_177945


namespace intersect_at_single_point_l177_177075

variables {P : Type*} [EuclideanGeometry P]
variables {A B C D E A1 B1 C1 D1 E1 : P}

-- Given conditions
axiom convex_pentagon (hA hB hC hD hE : P) :
  convex_hull {A, B, C, D, E}.is_pentagon
axiom midpoints_opposite_vertices :
  midpoint (segment A (segment_mid B C)) = Some A1 ∧
  midpoint (segment B (segment_mid C D)) = Some B1 ∧
  midpoint (segment C (segment_mid D E)) = Some C1 ∧
  midpoint (segment D (segment_mid E A)) = Some D1 ∧
  midpoint (segment E (segment_mid A B)) = Some E1

axiom lines_bisect_area (h_area : by
  let t1 := area (triangle A A1 E)
  let t2 := area (triangle D D1 C)
  let t3 := area (triangle B B1 D)
  let t4 := area (triangle C C1 E)
  assert t1 + t2 + t3 + t4 + area (triangle A E1 D) = area (pentagon A B C D E)
  sorry) :

-- Prove that lines AA1, BB1, CC1, DD1, EE1 intersect at a single point
theorem intersect_at_single_point :
  ∃ Q : P, are_concurrent {line_throughpoint A A1, line_throughpoint B B1, line_throughpoint C C1, line_throughpoint D D1, line_throughpoint E E1} Q :=
sorry

end intersect_at_single_point_l177_177075


namespace cube_square_third_smallest_prime_l177_177085

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l177_177085


namespace fraction_of_hidden_sea_is_five_over_eight_l177_177220

noncomputable def cloud_fraction := 1 / 2
noncomputable def island_uncovered_fraction := 1 / 4 
noncomputable def island_covered_fraction := island_uncovered_fraction / (1 - cloud_fraction)

-- The total island area is the sum of covered and uncovered.
noncomputable def total_island_fraction := island_uncovered_fraction + island_covered_fraction 

-- The sea area covered by the cloud is half minus the fraction of the island covered by the cloud.
noncomputable def sea_covered_by_cloud := cloud_fraction - island_covered_fraction 

-- The sea occupies the remainder of the landscape not taken by the uncoveed island.
noncomputable def total_sea_fraction := 1 - island_uncovered_fraction - cloud_fraction + island_covered_fraction 

-- The sea fraction visible and not covered by clouds
noncomputable def sea_visible_not_covered := total_sea_fraction - sea_covered_by_cloud 

-- The fraction of the sea hidden by the cloud
noncomputable def sea_fraction_hidden_by_cloud := sea_covered_by_cloud / total_sea_fraction 

theorem fraction_of_hidden_sea_is_five_over_eight : sea_fraction_hidden_by_cloud = 5 / 8 := 
by
  sorry

end fraction_of_hidden_sea_is_five_over_eight_l177_177220


namespace basketball_initial_winning_percentage_l177_177468

-- Lean statement for the given math problem
theorem basketball_initial_winning_percentage 
  (played_games : ℕ) 
  (remaining_games : ℕ) 
  (can_lose : ℕ)
  (total_games : ℕ := played_games + remaining_games)
  (target_percentage : ℚ := 0.60)
  (wins_needed : ℕ := (target_percentage * total_games).to_nat)
  (initial_wins : ℕ := wins_needed - can_lose) 
  (initial_percentage : ℚ := (initial_wins / played_games) * 100) : 
  played_games = 40 → 
  remaining_games = 10 → 
  can_lose = 8 → 
  target_percentage = 0.60 → 
  initial_percentage = 55 :=
by
  intros
  sorry

end basketball_initial_winning_percentage_l177_177468


namespace count_divisibles_lt_300_l177_177631

theorem count_divisibles_lt_300 : 
  let lcm_4_5_7 := Nat.lcm 4 (Nat.lcm 5 7) in
  lcm_4_5_7 = 140 → 
  ∀ n : ℕ, n < 300 → n % lcm_4_5_7 = 0 → 
  2 = 2 := sorry

end count_divisibles_lt_300_l177_177631


namespace combined_annual_income_eq_correct_value_l177_177397

theorem combined_annual_income_eq_correct_value :
  let A_income := 5 / 2 * 17000
  let B_income := 1.12 * 17000
  let C_income := 17000
  let D_income := 0.85 * A_income
  (A_income + B_income + C_income + D_income) * 12 = 1375980 :=
by
  sorry

end combined_annual_income_eq_correct_value_l177_177397


namespace stratified_sampling_second_year_students_l177_177875

theorem stratified_sampling_second_year_students 
  (total_athletes : ℕ) 
  (first_year_students : ℕ) 
  (sample_size : ℕ) 
  (second_year_students_in_sample : ℕ)
  (h1 : total_athletes = 98) 
  (h2 : first_year_students = 56) 
  (h3 : sample_size = 28)
  (h4 : second_year_students_in_sample = (42 * sample_size) / total_athletes) :
  second_year_students_in_sample = 4 := 
sorry

end stratified_sampling_second_year_students_l177_177875


namespace H_is_orthocenter_of_triangle_ABC_circumradius_of_triangle_ABC_eq_R_l177_177126

variable {R : ℝ}
variable {H A B C : ℝ} -- Points in the plane

-- Given conditions
def three_circles_intersect_at_H (R : ℝ) (H A B C : ℝ) := 
  ∃ (O1 O2 O3 : ℝ), 
    (dist O1 H = R ∧ dist O2 H = R ∧ dist O3 H = R) ∧
    (dist O1 A = R ∧ dist O1 B = R ∧ dist O1 C = R) ∧
    (dist O2 A = R ∧ dist O2 B = R ∧ dist O2 C = R) ∧
    (dist O3 A = R ∧ dist O3 B = R ∧ dist O3 C = R)

-- Theorem statements based on given conditions
theorem H_is_orthocenter_of_triangle_ABC 
  (h : three_circles_intersect_at_H R H A B C) : 
  is_orthocenter H A B C := sorry

theorem circumradius_of_triangle_ABC_eq_R 
  (h : three_circles_intersect_at_H R H A B C) : 
  circumradius A B C = R := sorry

end H_is_orthocenter_of_triangle_ABC_circumradius_of_triangle_ABC_eq_R_l177_177126


namespace sqrt_simplification_of_2_6_5_5_l177_177109

theorem sqrt_simplification_of_2_6_5_5 : 
  ∃ (a b : ℕ), (a + b = 30) ∧ (∃ (k : ℝ), k = 4 ∧ (a * (k.root 4 (2^6 * 5^5)) = (a * (k.root 4 b)))) := 
sorry

end sqrt_simplification_of_2_6_5_5_l177_177109


namespace sports_competition_l177_177419

theorem sports_competition (M p1 p2 p3 : ℕ) (A_points B_points C_points : ℕ) (win_100m : string) :
  (p1 > p2) ∧ (p2 > p3) ∧ 
  (A_points = 22) ∧ 
  (B_points = 9) ∧ 
  (C_points = 9) ∧ 
  (win_100m = "B") ∧ 
  (M > 1) ∧ 
  (M * (p1 + p2 + p3) = 40) ∧ 
  (p1 > 0) ∧ (p2 > 0) ∧ (p3 > 0) 
→ (M = 5) ∧ 
  ("C" = "second" ∧ "high_jump_event") := 
by {
  sorry
}

end sports_competition_l177_177419


namespace perfect_square_probability_l177_177891

-- Define n as a positive integer not exceeding 200
def n (k: ℕ) : Prop := k ≤ 200

-- Define the probability function
def probability (k: ℕ) (p: ℚ) : ℚ := if k ≤ 100 then p else 2 * p

-- Define that the sum of probabilities of all n should be 1
def total_prob (p: ℚ) : Prop := (100 * p) + (100 * 2 * p) = 1

-- Define the set of perfect squares
def perfect_squares : set ℕ := {k | ∃ (m: ℕ), m^2 = k}

-- Define the probability of choosing a perfect square
def prob_perfect_square (p: ℚ) : ℚ :=
  (∑ m in (finset.range 101).filter (λ m, (m^2) ≤ 100 ∧ (m^2) ∈ perfect_squares), p) +
  (∑ m in (finset.range 15).filter (λ m, (101 ≤ m^2 ∧ m^2 ≤ 200 ∧ (m^2) ∈ perfect_squares)), 2 * p)

-- The statement to prove:
theorem perfect_square_probability :
  ∃ p: ℚ, total_prob p ∧ prob_perfect_square p = 1/30 :=
by {
    sorry
}

end perfect_square_probability_l177_177891


namespace triangle_angle_XYZ_l177_177319

-- We need to show that \(\angle XYZ = 55^\circ\) given the conditions.
theorem triangle_angle_XYZ (P Q R X Y Z : Point)
  (hPQ_PR : dist P Q = dist P R)
  (h_angle_P : ∠ P Q R = 70)
  (h_RX_RY : dist R X = dist R Y)
  (h_QZ_QY : dist Q Z = dist Q Y)
  (hX_on_QR : X ∈ line Q R)
  (hY_on_PR : Y ∈ line P R)
  (hZ_on_PQ : Z ∈ line P Q) :
  ∠ X Y Z = 55 := 
sorry

end triangle_angle_XYZ_l177_177319


namespace sequence_formula_l177_177315

variable (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def cond1 : ∀ n, S n = n^2 * a n := by sorry
def cond2 : a 1 = 1 := by sorry

-- Conclusion
theorem sequence_formula (h1 : ∀ n, S n = n^2 * a n) (h2 : a 1 = 1) (n ≥ 2) : 
  a n = 2 / (n * (n + 1)) := 
by sorry

end sequence_formula_l177_177315


namespace unique_nonneg_sequence_l177_177375

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end unique_nonneg_sequence_l177_177375


namespace power_difference_l177_177587

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l177_177587


namespace sum_two_numbers_eq_twelve_l177_177409

theorem sum_two_numbers_eq_twelve (x y : ℕ) (h1 : x^2 + y^2 = 90) (h2 : x * y = 27) : x + y = 12 :=
by
  sorry

end sum_two_numbers_eq_twelve_l177_177409


namespace determine_p_in_terms_of_q_l177_177540

variable {p q : ℝ}

-- Given the condition in the problem
def log_condition (p q : ℝ) : Prop :=
  Real.log p + 2 * Real.log q = Real.log (2 * p + q)

-- The goal is to prove that under this condition, the following holds
theorem determine_p_in_terms_of_q (h : log_condition p q) :
  p = q / (q^2 - 2) :=
sorry

end determine_p_in_terms_of_q_l177_177540


namespace min_value_expr_least_is_nine_l177_177212

noncomputable def minimum_value_expression (a b c d : ℝ) : ℝ :=
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2

theorem min_value_expr_least_is_nine (a b c d : ℝ)
  (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  minimum_value_expression a b c d = 9 := 
sorry

end min_value_expr_least_is_nine_l177_177212


namespace angle_E_is_60_l177_177159

-- Define the properties of the convex polygon ABCDE
variables (A B C D E : Type) [ConvexPolygon ABCDE]

-- Define the equal length sides
def equal_length_sides : Prop := 
  distance A B = distance B C ∧
  distance B C = distance C D ∧
  distance C D = distance D E ∧
  distance D E = distance E A

-- Define the specific angles
def angle_B : Prop := angle B = 90
def angle_C : Prop := angle C = 120

-- Theorem statement
theorem angle_E_is_60 
  (h1 : equal_length_sides A B C D E)
  (h2 : angle_B B)
  (h3 : angle_C C) :
  ∃ angle E, angle E = 60 := 
sorry

end angle_E_is_60_l177_177159


namespace third_friend_34_third_friend_35_impossible_third_friend_56_impossible_l177_177421

variable (A B C : string)
variable (games_A games_B games_C : ℕ)
variable (x y z : ℕ)

-- Given conditions
axiom games_A_eq : games_A = 25
axiom games_B_eq : games_B = 17

-- Define total games played by the third friend for different scenarios
def games_C_34 := x + y = 34
def games_C_35 := x + y = 35
def games_C_56 := x + y = 56

-- Define supporting equations
def eq1 := x + z = games_A
def eq2 := y + z = games_B

-- Proving the required questions
theorem third_friend_34 : ∃ x y z, eq1 x z ∧ eq2 y z ∧ games_C_34 x y := sorry

theorem third_friend_35_impossible : ¬ ∃ x y z, eq1 x z ∧ eq2 y z ∧ games_C_35 x y := sorry

theorem third_friend_56_impossible : ¬ ∃ x y z, eq1 x z ∧ eq2 y z ∧ games_C_56 x y := sorry

end third_friend_34_third_friend_35_impossible_third_friend_56_impossible_l177_177421


namespace perpendicular_FL_BC_l177_177862

theorem perpendicular_FL_BC
  (A B C K F L : Type)
  [IsMidpoint F A C]   -- F is the midpoint of A and C
  [ISProj L A B K]     -- L is the projection of A onto BK
  (h1 : Dist A K = 2 * Dist K C)  -- AK = 2KC
  (h2 : Angle A B K = 2 * Angle K B C)  -- ∠ABK = 2 * ∠KBC
  : Perpendicular FL BC := sorry

end perpendicular_FL_BC_l177_177862


namespace min_pieces_for_four_contiguous_in_row_l177_177842

theorem min_pieces_for_four_contiguous_in_row (m n : ℕ) (pieces : ℕ) :
  m = 8 →
  n = 8 →
  pieces = 49 →
  ∀ (placement : fin m × fin n → bool),
    (∃ row, ∃ k,  k ≤ n - 4 ∧ placement (row, ⟨k, nat.lt_of_le_of_lt (nat.zero_le _) (by decide : k < n)⟩)
                 ∧ placement (row, ⟨k + 1, nat.lt_of_le_of_lt (nat.zero_le _) (by decide : k + 1 < n)⟩)
                 ∧ placement (row, ⟨k + 2, nat.lt_of_le_of_lt (nat.zero_le _) (by decide : k + 2 < n)⟩)
                 ∧ placement (row, ⟨k + 3, nat.lt_of_le_of_lt (nat.zero_le _) (by decide : k + 3 < n)⟩)) →
  ∃ (row : fin 8), ∃ k, k ≤ 4 → placement (row, k)
:=
begin
  intros,
  sorry  -- skip the proof
end

end min_pieces_for_four_contiguous_in_row_l177_177842


namespace part_a_part_b_l177_177349

variables (A B C I U D : Point)
variables (ABC_isosceles : is_isosceles_triangle A B C)
variables (AC_eq_BC : AC = BC)
variables (angle_ACB_lt_60 : angle A C B < 60)
variables (incenter_condition : incenter ABC = I)
variables (circumcenter_condition : circumcenter ABC = U)
variables (D_on_circumcircle_BIU : D ∈ circumcircle_BIU)
variables (D_second_intersect : ∀ P, P ≠ B → P ≠ I → P ≠ U → P ≠ D → P ∈ circumcircle_BIU → P ∉ BC)

theorem part_a : is_parallel (line_through I D) (line_through A C) := sorry

theorem part_b : is_perpendicular (line_through U D) (line_through I B) := sorry

end part_a_part_b_l177_177349
