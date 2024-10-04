import Mathlib

namespace tom_tim_ratio_l668_668883

variable (T M : ℝ)
variable h1 : T + M = 12
variable h2 : T + 1.3 * M = 15

theorem tom_tim_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 := by
  sorry

end tom_tim_ratio_l668_668883


namespace car_truck_ratio_l668_668036

theorem car_truck_ratio (total_vehicles trucks cars : ℕ)
  (h1 : total_vehicles = 300)
  (h2 : trucks = 100)
  (h3 : cars + trucks = total_vehicles)
  (h4 : ∃ (k : ℕ), cars = k * trucks) : 
  cars / trucks = 2 :=
by
  sorry

end car_truck_ratio_l668_668036


namespace Meadowood_problem_l668_668365

theorem Meadowood_problem (s h : ℕ) : ¬(26 * s + 3 * h = 58) :=
sorry

end Meadowood_problem_l668_668365


namespace vasya_no_purchase_days_l668_668946

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668946


namespace convert_binary_to_decimal_l668_668998

noncomputable def binary_to_decimal : ℕ → ℚ
| 0 := 0
| n := if n % 10 == 1 then binary_to_decimal (n / 10) * 2 + 1 else binary_to_decimal (n / 10) * 2

theorem convert_binary_to_decimal (n : ℚ) : n = 1 * 2^2 + 1 * 2^1 + 1 * 2^0 + 1 * 2^(-1) + 1 * 2^(-2) → n = 7.75 :=
by
  sorry

end convert_binary_to_decimal_l668_668998


namespace sunzi_carriage_l668_668151

theorem sunzi_carriage (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  ((Three people share a carriage, leaving two carriages empty) ∧ (Two people share a carriage, leaving nine people walking)) := sorry

end sunzi_carriage_l668_668151


namespace number_of_friends_l668_668538

-- Let n be the number of friends
-- Given the conditions:
-- 1. 9 chicken wings initially.
-- 2. 7 more chicken wings cooked.
-- 3. Each friend gets 4 chicken wings.

theorem number_of_friends :
  let initial_wings := 9
  let additional_wings := 7
  let wings_per_friend := 4
  let total_wings := initial_wings + additional_wings
  let n := total_wings / wings_per_friend
  n = 4 :=
by
  sorry

end number_of_friends_l668_668538


namespace volume_of_regular_tetrahedral_pyramid_l668_668183

-- Given conditions as variables
def height := 11  -- in cm
def area := 210  -- in cm^2

-- Required volume to verify
def volume := 770  -- in cm^3

theorem volume_of_regular_tetrahedral_pyramid (m a V : ℝ) 
  (h₀ : m = height)
  (h₁ : a = area)
  (h₂ : V = volume) :
  V = (1.0 / 3.0) * a * m := sorry

end volume_of_regular_tetrahedral_pyramid_l668_668183


namespace cary_calories_burn_per_mile_l668_668562

variables (c : ℕ) -- We assume c is a natural number for simplicity, can adjust if needed
variables (h_walk : ℕ) (h_candy : ℕ) (h_deficit : ℕ)

-- Define the conditions
def walk_miles : ℕ := 3
def candy_calories : ℕ := 200
def calorie_deficit : ℕ := 250

-- Define the condition relationship
 theorem cary_calories_burn_per_mile (h_walk : h_walk = walk_miles) (h_candy : h_candy = candy_calories) (h_deficit : h_deficit = calorie_deficit) : 
 c = 150 := 
by {
  have h1 : 3 * c - 200 = 250, from sorry,
  have h2 : 3 * c = 450, from sorry,
  have h3 : c = 150, from sorry,
  exact h3,
}

end cary_calories_burn_per_mile_l668_668562


namespace I1_is_incenter_I2_is_excenter_l668_668023

variables {A B C D I1 I2 E F M : Type}

-- Given the conditions
def height_from_A (A B C D : Type) : Prop :=
  true  -- AD is the height from A to BC in ΔABC, where AB < AC
  
def incenters (ABD ACD : Type) (I1 I2 : Type) : Prop :=
  true  -- I1 and I2 are the incenters of ΔABD and ΔACD respectively

def circumcircle_intersections (A I1 I2 AB AC E F : Type) : Prop :=
  true  -- Circumcircle of ΔAI1I2 intersects AB at E and AC at F

def EF_intersects_BC_at (E F BC M : Type) : Prop :=
  true  -- EF intersects BC at M

-- To be proved
theorem I1_is_incenter_I2_is_excenter (A B C D I1 I2 E F M : Type)
  (h1 : height_from_A A B C D)
  (h2 : incenters A B D A C D I1 I2)
  (h3 : circumcircle_intersections A I1 I2 A B A C E F)
  (h4 : EF_intersects_BC_at E F B C M) :
  (true : Prop) :=
sorry

end I1_is_incenter_I2_is_excenter_l668_668023


namespace quadratic_k_value_l668_668271

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end quadratic_k_value_l668_668271


namespace total_cost_to_replace_floor_l668_668478

def removal_cost : ℝ := 50
def cost_per_sqft : ℝ := 1.25
def room_dimensions : (ℝ × ℝ) := (8, 7)

theorem total_cost_to_replace_floor :
  removal_cost + (cost_per_sqft * (room_dimensions.1 * room_dimensions.2)) = 120 := by
  sorry

end total_cost_to_replace_floor_l668_668478


namespace neg_70kg_represents_subtract_70kg_l668_668733

theorem neg_70kg_represents_subtract_70kg (add_30kg : Int) (concept_opposite : ∀ (x : Int), x = -(-x)) :
  -70 = -70 := 
by
  sorry

end neg_70kg_represents_subtract_70kg_l668_668733


namespace cos_chebyshev_sin_chebyshev_l668_668888

-- Chebyshev polynomials of the first kind
def chebyshev_first : ℕ → (ℝ → ℝ) 
| 0       := λ x, 1
| 1       := λ x, x
| (n + 1) := λ x, 2 * x * chebyshev_first n x - chebyshev_first (n - 1) x

-- Chebyshev polynomials of the second kind
def chebyshev_second : ℕ → (ℝ → ℝ)
| 0       := λ x, 1
| 1       := λ x, 2 * x
| (n + 1) := λ x, 2 * x * chebyshev_second n x - chebyshev_second (n - 1) x

-- Theorem to prove the required identities using De Moivre's theorem
theorem cos_chebyshev (n : ℕ) (x : ℝ) : 
    cos (n * x) = chebyshev_first n (cos x) := by sorry

theorem sin_chebyshev (n : ℕ) (x : ℝ) : 
    sin (n * x) = sin x * chebyshev_second (n - 1) (cos x) := by sorry

end cos_chebyshev_sin_chebyshev_l668_668888


namespace cylinder_oil_depth_l668_668184

noncomputable def cylinder_oil_depth_upright (h_cylinder : ℝ) (d_base : ℝ) (h_oil_flat : ℝ) : ℝ :=
  let r := d_base / 2
  let theta := 2 * real.acos ((r - h_oil_flat) / r)
  let A_segment := (r^2 / 2) * (theta - real.sin theta)
  let A_circle := real.pi * r^2
  let fraction_covered := A_segment / A_circle
  h_cylinder * fraction_covered

theorem cylinder_oil_depth (h_cylinder : ℝ) (d_base : ℝ) (h_oil_flat : ℝ) :
  h_cylinder = 20 ∧ d_base = 6 ∧ h_oil_flat = 4 →
  abs (cylinder_oil_depth_upright h_cylinder d_base h_oil_flat - 2.2) < 0.1 :=
by
  intros h_assumption
  -- Assertion: under given assumptions, the depth of the oil when the cylinder is upright is approximately 2.2 feet.
  sorry

end cylinder_oil_depth_l668_668184


namespace greatest_b_value_for_integer_solution_eq_l668_668824

theorem greatest_b_value_for_integer_solution_eq : ∀ (b : ℤ), (∃ (x : ℤ), x^2 + b * x = -20) → b > 0 → b ≤ 21 :=
by
  sorry

end greatest_b_value_for_integer_solution_eq_l668_668824


namespace exists_set_M0_l668_668722

variable (I : Type)
variable (f : Set I → I)
variable (h_f_unique : ∀ (A : Set I), Set.card A = 19 → (∀ (x y : I), (x ∈ A ∧ y ∈ A ∧ f A = x ∧ f A = y) → x = y))
variable (h_f_friend : ∀ (A : Set I), Set.card A = 19 → f A ∈ I)

theorem exists_set_M0 (I : Set I) (h_I : Set.card I = 40) : 
  ∃ (M0 : Set I), Set.card M0 = 20 ∧ ∀ a ∈ M0, f (M0 \ {a}) ≠ a :=
by sorry

end exists_set_M0_l668_668722


namespace range_of_k_for_monotonic_increase_l668_668308

theorem range_of_k_for_monotonic_increase : 
  ∀ k : ℝ, (∀ x ∈ Icc (5 : ℝ) (20 : ℝ), 8 * x - k ≥ 0) → k ∈ Iic (40 : ℝ) :=
by
  intros k h
  have h_vertex : k / 8 ≤ 5, from sorry
  have h_range : k ≤ 40, from sorry
  exact ⟨h_range⟩

end range_of_k_for_monotonic_increase_l668_668308


namespace stratified_sample_size_is_correct_l668_668473

def workshop_A_produces : ℕ := 120
def workshop_B_produces : ℕ := 90
def workshop_C_produces : ℕ := 60
def sample_from_C : ℕ := 4

def total_products : ℕ := workshop_A_produces + workshop_B_produces + workshop_C_produces

noncomputable def sampling_ratio : ℚ := (sample_from_C:ℚ) / (workshop_C_produces:ℚ)

noncomputable def sample_size : ℚ := total_products * sampling_ratio

theorem stratified_sample_size_is_correct :
  sample_size = 18 := by
  sorry

end stratified_sample_size_is_correct_l668_668473


namespace cone_volume_correct_l668_668965

variables (r m n p h : ℝ)

def volume_cone (r h : ℝ) : ℝ := (1/3) * real.pi * r^2 * h

theorem cone_volume_correct :
  ∃ h, m + n > 0 ∧ m + p > 0 ∧
  let V_cylinder := real.pi * r^2 * m in
  let V_rise_tip_down := real.pi * r^2 * (m + n) in
  let V_rise_base_down := real.pi * r^2 * (m + p) in
  volume_cone r h = (1 / 3) * real.pi * r^2 * h :=
begin
  sorry
end

end cone_volume_correct_l668_668965


namespace find_sum_of_angles_l668_668644

open Real

namespace math_problem

theorem find_sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : cos (α - β / 2) = sqrt 3 / 2)
  (h2 : sin (α / 2 - β) = -1 / 2) : α + β = 2 * π / 3 :=
sorry

end math_problem

end find_sum_of_angles_l668_668644


namespace wendy_adds_18_gallons_l668_668118

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end wendy_adds_18_gallons_l668_668118


namespace clips_ratio_l668_668033

def clips (April May: Nat) : Prop :=
  April = 48 ∧ April + May = 72 → (48 / (72 - 48)) = 2

theorem clips_ratio : clips 48 (72 - 48) :=
by
  sorry

end clips_ratio_l668_668033


namespace smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l668_668577

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l668_668577


namespace sum_difference_formula_l668_668567

def setA (n : ℕ) (x : ℕ) : Prop :=
  nat.bitcount x % 2 = 1

def setB (n : ℕ) (x : ℕ) : Prop :=
  nat.bitcount x % 2 = 0

def S (n r : ℕ) : ℤ :=
  ∑ x in finset.filter (setA n) (finset.range (2^n)), x^r -
  ∑ y in finset.filter (setB n) (finset.range (2^n)), y^r

theorem sum_difference_formula (n : ℕ) : 
  S n n = (-1)^(n-1) * nat.factorial n * 2^((n*(n-1))/2) :=
sorry

end sum_difference_formula_l668_668567


namespace truthful_dwarfs_count_l668_668603

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l668_668603


namespace common_factor_polynomials_l668_668042

-- Define the two polynomials
def poly1 (x y z : ℝ) := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
def poly2 (x y z : ℝ) := 6 * x^4 * y * z^2

-- Define the common factor
def common_factor (x y z : ℝ) := 3 * x^2 * y * z

-- The statement to prove that the common factor of poly1 and poly2 is 3 * x^2 * y * z
theorem common_factor_polynomials (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (poly1 x y z) = (f x y z) * (common_factor x y z) ∧
                          (poly2 x y z) = (f x y z) * (common_factor x y z) :=
sorry

end common_factor_polynomials_l668_668042


namespace vasya_days_without_purchase_l668_668925

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668925


namespace estimate_value_l668_668039

def squares (n : ℕ) : ℕ := n * n

theorem estimate_value : 
  (∏ n in finset.range(2016), (squares (n+2) - 1)) / (∏ n in finset.range(2016), squares(n+1)) = 1 / 2 := 
sorry

end estimate_value_l668_668039


namespace Sue_total_travel_time_to_SanFrancisco_l668_668063

theorem Sue_total_travel_time_to_SanFrancisco (T : ℝ) 
  (hT : T = 24) :
  let time_NewOrleans_to_NewYork := (3 / 4) * T
  let layover := 16
  let total_time := time_NewOrleans_to_NewYork + layover + T
  in total_time = 58 :=
by
  sorry

end Sue_total_travel_time_to_SanFrancisco_l668_668063


namespace z_share_in_profit_l668_668884

noncomputable def investment_share (investment : ℕ) (months : ℕ) : ℕ := investment * months

noncomputable def profit_share (profit : ℕ) (share : ℚ) : ℚ := (profit : ℚ) * share

theorem z_share_in_profit 
  (investment_X : ℕ := 36000)
  (investment_Y : ℕ := 42000)
  (investment_Z : ℕ := 48000)
  (months_X : ℕ := 12)
  (months_Y : ℕ := 12)
  (months_Z : ℕ := 8)
  (total_profit : ℕ := 14300) :
  profit_share total_profit (investment_share investment_Z months_Z / 
            (investment_share investment_X months_X + 
             investment_share investment_Y months_Y + 
             investment_share investment_Z months_Z)) = 2600 := 
by
  sorry

end z_share_in_profit_l668_668884


namespace weight_of_melted_mixture_l668_668887

def ratio_zinc_copper_silver := (9 : ℕ, 11 : ℕ, 7 : ℕ)
def weight_zinc := (27 : ℕ)

theorem weight_of_melted_mixture (r_z : ℕ) (r_c : ℕ) (r_s : ℕ) (wz : ℕ) (total_parts : ℕ) (weight_per_part : ℕ) :
  ratio_zinc_copper_silver = (r_z, r_c, r_s) ∧ 
  weight_zinc = wz ∧ 
  total_parts = r_z + r_c + r_s ∧ 
  weight_per_part = wz / r_z →
  r_z + r_c + r_s = 27 ∧ wz / r_z = 3 ∧ total_parts * weight_per_part = 81 :=
begin
  assume h,
  rcases h with ⟨rfl, rfl, rfl, rfl⟩,
  simp,
  split,
  { refl },
  { split; simp,
    sorry }
end

end weight_of_melted_mixture_l668_668887


namespace domain_transformation_l668_668675

noncomputable def f : ℝ → ℝ := sorry -- Define the function f (details not provided)

def domain_f_x_plus_1 : set ℝ := set.Ico (-1 : ℝ) 0
def domain_f_2x : set ℝ := set.Ico (0 : ℝ) (1/2)

theorem domain_transformation 
  (h : ∀ x, x ∈ domain_f_x_plus_1 → (x + 1) ∈ domain_f_x_plus_1) :
  ∀ x, x ∈ domain_f_2x → (2 * x) ∈ domain_f_x_plus_1 :=
sorry

end domain_transformation_l668_668675


namespace common_sum_4x4_matrix_l668_668068

theorem common_sum_4x4_matrix :
  ∀ (S : Fin 4 → Fin 4 → ℤ), 
  (∀ i : Fin 4, ∑ j, S i j = ∑ j, S j i) → 
  (∀ i : Fin 4, ∑ j, S i j) = 14 :=
begin
  -- Define the set of integers from -4 to 11
  let nums := list.range (11 - -4 + 1),
  let nums := list.map (λ x, x - 4) nums,
  have h_len : nums.length = 16, -- 16 integers in total

  -- Partition numbers into a 4x4 grid
  let S := λ i j:Fin 4, nums.nth_le (i.val * 4 + j.val) sorry,

  -- Sum of all numbers from -4 to 11 should sum to 56
  have h_sum : nums.sum = 56,

  -- Common sum for each row, column, and diagonal
  -- Since all sums are supposed to be equal to the total sum divided by 4
  have h_sums_eq : ∀ i, ∑ j, S i j = 14,
  sorry,
end

end common_sum_4x4_matrix_l668_668068


namespace total_present_ages_l668_668155

variable (P Q P' Q' : ℕ)

-- Condition 1: 6 years ago, \( p \) was half of \( q \) in age.
axiom cond1 : P = Q / 2

-- Condition 2: The ratio of their present ages is 3:4.
axiom cond2 : (P + 6) * 4 = (Q + 6) * 3

-- We need to prove: the total of their present ages is 21
theorem total_present_ages : P' + Q' = 21 :=
by
  -- We already have the variables and axioms in the context, so we just need to state the goal
  sorry

end total_present_ages_l668_668155


namespace octagon_cannot_cover_floor_l668_668880

def interior_angle (n : ℕ) : ℝ :=
  if h : n ≥ 3 then (180 * (n - 2) : ℝ) / n else 0

def can_cover_floor (n : ℕ) : Prop :=
  360 % interior_angle n = 0

theorem octagon_cannot_cover_floor : ¬ can_cover_floor 8 :=
by
  have h_angle : interior_angle 8 = 135 := rfl -- or compute directly
  have h_div : 360 % 135 ≠ 0 := by -- factual mathematical assertion
    norm_num
  exact h_div

end octagon_cannot_cover_floor_l668_668880


namespace third_month_sale_l668_668537

theorem third_month_sale (sale1 sale2 sale4 sale5 sale6 avg total : ℕ)
  (h1 : sale1 = 6400)
  (h2 : sale2 = 7000)
  (h4 : sale4 = 7200)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 5100)
  (h_avg : avg = 6500)
  (h_total : total = avg * 6) :
  (total - (sale1 + sale2 + sale4 + sale5 + sale6) = 6800) :=
by
  rw [h1, h2, h4, h5, h6, h_avg, h_total]
  sorry

end third_month_sale_l668_668537


namespace number_of_truthful_gnomes_l668_668609

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l668_668609


namespace polar_curve_line_and_circle_l668_668440

theorem polar_curve_line_and_circle {ρ θ : ℝ} :
  (ρ * real.sin θ = real.sin (2 * θ)) →
  ((∃ ρ, θ = 0 ∧ ρ ∈ set.univ) ∨ (ρ^2 = 2 * ρ * real.cos θ ↔ (∃ x y, (x - 1)^2 + y^2 = 1))) :=
by
  sorry

end polar_curve_line_and_circle_l668_668440


namespace regular_polygon_sides_l668_668179

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = 150 ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l668_668179


namespace quadrilateral_is_square_l668_668572

variables {A B C D O : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] 
variables [normed_add_comm_group A] [normed_add_comm_group B] [normed_add_comm_group C] [normed_add_comm_group D] [normed_add_comm_group O]
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D] [inner_product_space ℝ O]

def is_square (A B C D : Type*) :=
  dist A B = dist B C ∧
  dist B C = dist C D ∧
  dist C D = dist D A ∧
  dist A C = dist B D ∧
  (A.2.1 - O.2.1) ⬝ (B.2.1 - O.2.1) = 0

theorem quadrilateral_is_square
  (A B C D O : Type*)
  [α : metric_space A] 
  [β : metric_space B] 
  [γ : metric_space C] 
  [δ : metric_space D] 
  [ω : metric_space O]
  [normed_add_comm_group A] [normed_add_comm_group B] [normed_add_comm_group C] [normed_add_comm_group D] [normed_add_comm_group O]
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D] [inner_product_space ℝ O]
  (AO_eq_CO : dist A O = dist C O)
  (BO_eq_DO : dist B O = dist D O)
  (AC_perp_BD : dist A C = dist B D ∧ (A.2.1 - O.2.1) ⬝ (B.2.1 - O.2.1) = 0)
  : is_square A B C D :=
by
  sorry

end quadrilateral_is_square_l668_668572


namespace radius_of_circle_l668_668743

theorem radius_of_circle (A B C M : Point)
  (h_angleA : angle A B C = 120)
  (h_AC : distance A C = 1)
  (h_BC : distance B C = sqrt (7))
  (h_BM_altitude : ∃ (M: Point), BM ⟂ AC) :
  radius_of_circle_through_A_and_M_tangent_at_M
    (circle_through_M_B_C_with_tangency_at_M) = sqrt (7) / 4 :=
sorry

end radius_of_circle_l668_668743


namespace max_students_with_extra_credit_l668_668780

theorem max_students_with_extra_credit (n : ℕ := 150) (scores : Fin n → ℕ) :
  (∀ i : Fin n, scores i ≤ 10) →
  (∃ i : Fin n, scores i = 1) →
  1 ≤ ∑ i, scores i / n.toReal :=
  (∃ k : ℕ, k < n ∧ (∀ j : Fin n, scores j > ∑ i, scores i / n.toReal) →
            (k = 149)) :=
begin
  sorry,
end

end max_students_with_extra_credit_l668_668780


namespace triangle_position_after_five_movements_l668_668845

/-- 
  Given a square initially with a solid triangle positioned at the bottom,
  after rolling clockwise around a fixed regular octagon for five movements,
  the solid triangle will be positioned on the right side.
-/
theorem triangle_position_after_five_movements 
  (initial_position : string) 
  (movements : ℕ) 
  (octagon_interior_angle : ℕ) 
  (square_rotation_per_movement : ℕ) 
  (total_movements : ℕ) : 
  (initial_position = "bottom" ∧ movements = 5 ∧ octagon_interior_angle = 135 ∧ square_rotation_per_movement = 270 ∧ total_movements = 5) → 
  "position_of_triangle_after_movements" = "right" := 
by
  sorry

end triangle_position_after_five_movements_l668_668845


namespace product_of_roots_cubic_l668_668769

theorem product_of_roots_cubic :
  (p q r : ℂ) (hpq : (Polynomial.eval (polynomial.C p * polynomial.C q * polynomial.C r) (polynomial.X^3 - 9 * polynomial.X^2 + 5 * polynomial.X - 15)) = 0) :
  (p * q * r = 5) :=
sorry

end product_of_roots_cubic_l668_668769


namespace Tanika_total_boxes_sold_l668_668435

theorem Tanika_total_boxes_sold:
  let friday_boxes := 60
  let saturday_boxes := friday_boxes + 0.5 * friday_boxes
  let sunday_boxes := saturday_boxes - 0.3 * saturday_boxes
  friday_boxes + saturday_boxes + sunday_boxes = 213 :=
by
  sorry

end Tanika_total_boxes_sold_l668_668435


namespace loss_percentage_is_five_l668_668517

/-- Definitions -/
def original_price : ℝ := 490
def sold_price : ℝ := 465.50
def loss_amount : ℝ := original_price - sold_price

/-- Theorem -/
theorem loss_percentage_is_five :
  (loss_amount / original_price) * 100 = 5 :=
by
  sorry

end loss_percentage_is_five_l668_668517


namespace louisa_first_day_distance_l668_668417

theorem louisa_first_day_distance :
  ∃ x : ℕ, 
    let speed := 50,
        distance_second_day := 350,
        time_second_day := distance_second_day / speed,
        time_first_day := time_second_day - 3
    in x = speed * time_first_day ∧ x = 200 := 
by {
  -- This is just a statement, the proof is omitted.
  sorry
}

end louisa_first_day_distance_l668_668417


namespace correct_side_length_triangle_AOB_correct_equation_line_l_correct_number_of_lines_l668_668650

-- Define the conditions
structure GeometricConditions where
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  l : ℝ → ℝ
  parabola : ℝ → ℝ
  circle : ℝ × ℝ → ℝ

-- The distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Condition about parabola
def parabola (x y : ℝ) := y^2 = 4*x

-- Condition about circle (r > 0)
def circle (x y r : ℝ) := (x - 5)^2 + y^2 = r^2

-- Midpoint definition
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def problem_1 (A B : ℝ × ℝ) (O : ℝ × ℝ) (hAOB : (distance A B) = (distance O A) ∧ (distance O A) = (distance O B)) : ℝ :=
  distance A B

noncomputable def problem_2 (r : ℝ) (equation_l : ℝ → ℝ) : Prop :=
  r = 4 → (equation_l = fun y ↦ 1) ∨ (equation_l = fun y ↦ 9)

noncomputable def problem_3 (r : ℝ) : ℕ :=
  if (2 < r ∧ r < 4) then 4
  else if ((0 < r ∧ r ≤ 2) ∨ (4 ≤ r ∧ r < 5)) then 2
  else if (5 ≤ r) then 1
  else 0

-- Statements
theorem correct_side_length_triangle_AOB :
  ∀ (A B O : ℝ × ℝ) (hAOB : (distance A B) = (distance O A) ∧ (distance O A) = (distance O B)),
    problem_1 A B O hAOB = 8 * real.sqrt 3 := sorry

theorem correct_equation_line_l :
  ∀ (r : ℝ) (l : ℝ → ℝ),
    problem_2 r l := sorry

theorem correct_number_of_lines :
  ∀ (r : ℝ),
    problem_3 r = 
      if (2 < r ∧ r < 4) then 4
      else if ((0 < r ∧ r ≤ 2) ∨ (4 ≤ r ∧ r < 5)) then 2
      else if (5 ≤ r) then 1
      else 0 := sorry

end correct_side_length_triangle_AOB_correct_equation_line_l_correct_number_of_lines_l668_668650


namespace dot_product_zero_l668_668698

noncomputable def vec_a : ℝ^3 := sorry
noncomputable def vec_b : ℝ^3 := sorry

variables (angle_between_a_b : real.angle vec_a vec_b = π * (2 / 3))
variables (norm_a : ∥vec_a∥ = 4)
variables (norm_b : ∥vec_b∥ = 4)

theorem dot_product_zero : vec_b ⬝ (2 • vec_a + vec_b) = 0 := 
by {
  sorry
}

end dot_product_zero_l668_668698


namespace lyka_saving_per_week_l668_668407

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l668_668407


namespace least_clock_equiv_square_l668_668781

-- Define what it means for two times to be clock equivalent
def clock_equiv (a b : ℕ) : Prop :=
  (a - b) % 12 = 0

-- Define the target proof problem
theorem least_clock_equiv_square (h : ℕ) (h > 10) : (∃ k, k > 10 ∧ clock_equiv k (k * k) ∧ k = 12) :=
by
  -- The proof is omitted
  sorry

end least_clock_equiv_square_l668_668781


namespace uniformity_of_scores_l668_668049

/- Problem statement:
  Randomly select 10 students from class A and class B to participate in an English oral test. 
  The variances of their test scores are S1^2 = 13.2 and S2^2 = 26.26, respectively. 
  Then, we show that the scores of the 10 students from class A are more uniform than 
  those of the 10 students from class B.
-/

theorem uniformity_of_scores (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : 
    13.2 < 26.26 := 
by 
  sorry

end uniformity_of_scores_l668_668049


namespace number_of_valid_permutations_l668_668994

def nine_digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_valid_permutation (a : List ℕ) : Prop :=
  (a.perm nine_digits) ∧ (∃ x y : ℕ, 
    x = (a.dropLast 5).sum ∧ y = (a.drop 5).sum ∧
    (x + y = 45) ∧ (x - y) % 11 = 0)

theorem number_of_valid_permutations :
  {a : List ℕ // is_valid_permutation a}.card = 31680 := 
sorry

end number_of_valid_permutations_l668_668994


namespace number_of_truthful_dwarfs_l668_668582

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l668_668582


namespace planar_graph_inequality_l668_668787

theorem planar_graph_inequality (G : Type) [graph G] : planar G → ∃ F E : ℕ, faces G = F ∧ edges G = E → 2 * E ≥ 3 * F :=
by
  intros G planarG
  sorry

end planar_graph_inequality_l668_668787


namespace remainder_division_problem_l668_668873

theorem remainder_division_problem : 
  let a := 3^303 + 303 in
  let b := 3^151 + 3^75 + 1 in
  a % b = 294 := by
  sorry

end remainder_division_problem_l668_668873


namespace largest_consecutive_odd_sum_650_l668_668846

theorem largest_consecutive_odd_sum_650:
  (∑ k in finset.range 25, 2 * (k + 1)) = 650 →
  ∃ x : ℤ, x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1 ∧ (x + 6) % 2 = 1 ∧
    x + (x + 2) + (x + 4) + (x + 6) = 650 ∧
    max (max x (x + 2)) (max (x + 4) (x + 6)) = 165 :=
begin
  sorry
end

end largest_consecutive_odd_sum_650_l668_668846


namespace tan_alpha_value_l668_668307

noncomputable def f (x : ℝ) := 3 * Real.sin x + 4 * Real.cos x

theorem tan_alpha_value (α : ℝ) (h : ∀ x : ℝ, f x ≥ f α) : Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l668_668307


namespace socks_ratio_l668_668010

theorem socks_ratio (b : ℕ) (x : ℕ) :
  let cost_orig := 4 * (2 * x) + b * x in
  let cost_inter := b * (2 * x) + 4 * x in
  cost_inter = 3 * cost_orig / 2 → 4 / b = 1 / 4 :=
by
  intros cost_orig cost_inter h
  have h1 : cost_orig = 8 * x + b * x := rfl
  have h2 : cost_inter = 2 * b * x + 4 * x := rfl
  rw [h1, h2] at h
  sorry

end socks_ratio_l668_668010


namespace probability_harold_betty_two_common_books_l668_668416

open scoped BigOperators
  
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k 

def probability_two_common_books : ℚ := 
  let total_ways := (binomial 10 5) * (binomial 10 5)
  let successful_outcomes := (binomial 10 2) * (binomial 8 3) * (binomial 5 3)
  successful_outcomes / total_ways

theorem probability_harold_betty_two_common_books :
  probability_two_common_books = 25 / 63 := by
  sorry

end probability_harold_betty_two_common_books_l668_668416


namespace meeting_point_l668_668774

-- Coordinates of Mark and Sandy
def mark : ℝ × ℝ := (1, 10)
def sandy : ℝ × ℝ := (-5, 0)

-- Midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Proof statement
theorem meeting_point :
  midpoint mark sandy = (-2, 5) :=
by
  -- Here would be the proof steps
  sorry

end meeting_point_l668_668774


namespace find_sample_size_l668_668225

theorem find_sample_size :
  ∀ (n : ℕ), 
    (∃ x : ℝ,
      2 * x + 3 * x + 4 * x + 6 * x + 4 * x + x = 1 ∧
      2 * n * x + 3 * n * x + 4 * n * x = 27) →
    n = 60 :=
by
  intro n
  rintro ⟨x, h1, h2⟩
  sorry

end find_sample_size_l668_668225


namespace find_people_and_carriages_l668_668144

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end find_people_and_carriages_l668_668144


namespace gcd_proof_l668_668759

theorem gcd_proof :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 33 ∧ Nat.lcm a b = 90 ∧ Nat.gcd a b = 3 :=
sorry

end gcd_proof_l668_668759


namespace wendy_total_gas_to_add_l668_668117

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end wendy_total_gas_to_add_l668_668117


namespace ellipse_eccentricity_l668_668303

theorem ellipse_eccentricity (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c = b)
  (h4 : a = Real.sqrt 2 * c)
  : ∃ e : ℝ, e = c / a ∧ e = Real.sqrt 2 / 2 := 
by
  use c / a
  split
  · sorry
  · sorry

end ellipse_eccentricity_l668_668303


namespace Bingley_bracelets_l668_668556

theorem Bingley_bracelets 
  (B : ℕ) (K : ℕ) 
  (h_B : B = 5) (h_K : K = 16) :
  (B + (K / 4) / 3) - (B + (K / 4) / 3) / 3 = 4 :=
by
  -- Bingley's initial bracelets
  have h1 : B = 5 := h_B,
  
  -- Kelly's initial bracelets
  have h2 : K = 16 := h_K,

  -- Kelly gives a fourth of her bracelets
  have h3 : K / 4 = 4,
  
  -- Kelly gives in sets of 3
  have h4 : (K / 4) / 3 = 4 / 3,
  
  -- Bingley's total after receiving bracelets from Kelly
  have h5 : (B + (K / 4) / 3) = 6,
  
  -- Bingley gives a third to his sister
  have h6 : (B + (K / 4) / 3) / 3 = 2,

  -- Bingley's remaining bracelets
  exact h5 - h6

end Bingley_bracelets_l668_668556


namespace adiabatic_compression_work_eq_l668_668721

variable (V0 V1 p0 k : ℝ) (h : 1 ≠ k)

def work_of_adiabatic_compression : ℝ :=
  (p0 * V0 / (k - 1)) * ((V0 / V1) ^ (k - 1) - 1)

theorem adiabatic_compression_work_eq :
  W = work_of_adiabatic_compression V0 V1 p0 k :=
by
  sorry

end adiabatic_compression_work_eq_l668_668721


namespace intersection_A_B_range_m_l668_668319

-- Definitions for Sets A, B, and C
def SetA : Set ℝ := { x | -2 ≤ x ∧ x < 5 }
def SetB : Set ℝ := { x | 3 * x - 5 ≥ x - 1 }
def SetC (m : ℝ) : Set ℝ := { x | -x + m > 0 }

-- Problem 1: Prove \( A \cap B = \{ x \mid 2 \leq x < 5 \} \)
theorem intersection_A_B : SetA ∩ SetB = { x : ℝ | 2 ≤ x ∧ x < 5 } :=
by
  sorry

-- Problem 2: Prove \( m \in [5, +\infty) \) given \( A \cup C = C \)
theorem range_m (m : ℝ) : (SetA ∪ SetC m = SetC m) → m ∈ Set.Ici 5 :=
by
  sorry

end intersection_A_B_range_m_l668_668319


namespace input_for_output_16_l668_668105

theorem input_for_output_16 (x : ℝ) (y : ℝ) : 
  (y = (if x < 0 then (x + 1)^2 else (x - 1)^2)) → 
  y = 16 → 
  (x = 5 ∨ x = -5) :=
by sorry

end input_for_output_16_l668_668105


namespace alcohol_percentage_in_new_mixture_l668_668154

theorem alcohol_percentage_in_new_mixture :
  let total_volume_original := 20
  let percent_alcohol := 20
  let percent_water := 100 - percent_alcohol
  let volume_alcohol := total_volume_original * (percent_alcohol / 100)
  let volume_water_original := total_volume_original - volume_alcohol
  let volume_water_added := 3
  let total_volume_new := total_volume_original + volume_water_added
  let percent_alcohol_new := (volume_alcohol / total_volume_new) * 100
  percent_alcohol_new ≈ 17.39 :=
by
  sorry

end alcohol_percentage_in_new_mixture_l668_668154


namespace dwarfs_truthful_count_l668_668587

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l668_668587


namespace greatest_integer_l668_668475

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 2) (h3 : ∃ l : ℕ, n = 8 * l - 4) : n = 124 :=
by
  sorry

end greatest_integer_l668_668475


namespace remainder_mod_5_l668_668244

-- Definitions based on the conditions
def is_arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) (last_term : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (a + (i - 1) * d) = last_term

def sequence_mod_5_eq_4 (a : ℕ) (d : ℕ) (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (a + (i - 1) * d) % 5 = 4

-- Given conditions in Lean
def a := 4
def d := 10
def n := 20
def last_term := 194

-- Proof statement in Lean
theorem remainder_mod_5 :
  is_arithmetic_sequence a d n last_term ∧ sequence_mod_5_eq_4 a d n →
  (Finset.prod (Finset.range n.succ) (λ i, a + i * d)) % 5 = 1 := 
by sorry

end remainder_mod_5_l668_668244


namespace chocolate_cost_proof_l668_668217

/-- The initial amount of money Dan has. -/
def initial_amount : ℕ := 7

/-- The cost of the candy bar. -/
def candy_bar_cost : ℕ := 2

/-- The remaining amount of money Dan has after the purchases. -/
def remaining_amount : ℕ := 2

/-- The cost of the chocolate. -/
def chocolate_cost : ℕ := initial_amount - candy_bar_cost - remaining_amount

/-- Expected cost of the chocolate. -/
def expected_chocolate_cost : ℕ := 3

/-- Prove that the cost of the chocolate equals the expected cost. -/
theorem chocolate_cost_proof : chocolate_cost = expected_chocolate_cost :=
by
  sorry

end chocolate_cost_proof_l668_668217


namespace symmetric_coordinates_l668_668819

def symmetric_point_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-P.1, P.2, -P.3)

theorem symmetric_coordinates (P : ℝ × ℝ × ℝ) :
  symmetric_point_y_axis (2, -1, 4) = (-2, -1, -4) := by
  sorry

end symmetric_coordinates_l668_668819


namespace shakes_indeterminable_l668_668984

theorem shakes_indeterminable (B S C x : ℝ) (h1 : 3 * B + 7 * S + C = 120) (h2 : 4 * B + x * S + C = 164.50) : ¬ (∃ B S C, ∀ x, 4 * B + x * S + C = 164.50) → false := 
by 
  sorry

end shakes_indeterminable_l668_668984


namespace range_of_a_nonempty_l668_668396

noncomputable def A (a : ℝ) (x : ℝ) : Prop := x^2 + x + a ≤ 0
noncomputable def B (a : ℝ) (x : ℝ) : Prop := x^2 - x + 2*a*x - 1 < 0
noncomputable def C (a : ℝ) (x : ℝ) : Prop := a ≤ x ∧ x ≤ 4*a - 9

theorem range_of_a_nonempty :
  (∃ x, A a x) ∨ (∃ x, B a x) ∨ (∃ x, C a x) ↔ a ∈ (Iio (5/8) ∪ Ici 3) :=
sorry

end range_of_a_nonempty_l668_668396


namespace probability_floor_log3_l668_668050

open Real

def probability_floor_log3_eq (x y : ℝ) : ℝ :=
  if x ∈ (0, 1/2) ∧ y ∈ (0, 1/2) then sorry else 0

theorem probability_floor_log3 (x y : ℝ) (hx : x ∈ (0, 1/2)) (hy : y ∈ (0, 1/2)) :
  probability_floor_log3_eq x y = 25 / 324 :=
sorry

end probability_floor_log3_l668_668050


namespace arithmetic_sequence_l668_668356

theorem arithmetic_sequence (
  a1 : ℤ,
  d : ℤ,
  S : ℤ,
  n : ℤ
) (h_a1 : a1 = 13)
  (h_d : d = -2)
  (h_S : S = 40)
  (h_sum : S = n * a1 + n * (n - 1) * d / 2) :
  n = 4 ∨ n = 10 :=
by
  sorry

end arithmetic_sequence_l668_668356


namespace lyka_saving_per_week_l668_668409

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l668_668409


namespace infinite_series_value_l668_668993

theorem infinite_series_value :
  (∑' n in (Set.Ici 3 : Set ℕ), ((n^4 + 4*n^2 + 10*n + 10) / (3^n * (n^4 + 4)))) = 3 :=
sorry

end infinite_series_value_l668_668993


namespace height_of_taller_tree_l668_668463

theorem height_of_taller_tree 
  (h : ℝ) 
  (ratio_condition : (h - 20) / h = 2 / 3) : 
  h = 60 := 
by 
  sorry

end height_of_taller_tree_l668_668463


namespace compute_H_5_times_l668_668174

noncomputable def H : ℝ → ℝ
| 2 := -2
| -2 := 6
| 6 := 6
| _ := 0  -- we use 0 as a default value for undefined points for the purpose of this statement

theorem compute_H_5_times :
  H (H (H (H (H 2)))) = 6 :=
by {
  simp only [H],
  -- H(2) = -2
  -- H(-2) = 6
  -- H(6) = 6
  sorry
}

end compute_H_5_times_l668_668174


namespace years_digits_arithmetic_progression_l668_668224

noncomputable def valid_years_in_century : List ℕ :=
  [1881, 1894]

theorem years_digits_arithmetic_progression :
  ∀ (year : ℕ), year = 1800 + 10 * x + y ∧ 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 
  let d1 := 8 - 1 in
  let d2 := x - 8 in
  let d3 := y - x in
  (d1, d2, d3) dictates an arithmetic sequence →
    year ∈ valid_years_in_century
:= sorry

end years_digits_arithmetic_progression_l668_668224


namespace average_speed_bike_l668_668611

theorem average_speed_bike (t_goal : ℚ) (d_swim r_swim : ℚ) (d_run r_run : ℚ) (d_bike r_bike : ℚ) :
  t_goal = 1.75 →
  d_swim = 1 / 3 ∧ r_swim = 1.5 →
  d_run = 2.5 ∧ r_run = 8 →
  d_bike = 12 →
  r_bike = 1728 / 175 :=
by
  intros h_goal h_swim h_run h_bike
  sorry

end average_speed_bike_l668_668611


namespace prob_f_is_meaningful_l668_668427

open set

noncomputable def f (x : ℝ) : ℝ := real.log (1 - x) + real.sqrt (x + 2)

theorem prob_f_is_meaningful :
  let I := Icc (-3 : ℝ) 3 in
  let M := {x : ℝ | 1 - x > 0 ∧ x + 2 ≥ 0 ∧ x ∈ I} in
  measure_theory.measure_space.measure ((real.volume.restrict I).to_outer_measure M.to_measurable_set) = 1 / 2 :=
by
    sorry

end prob_f_is_meaningful_l668_668427


namespace line_length_l668_668079

theorem line_length (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := 
by
  sorry

end line_length_l668_668079


namespace polynomial_nonnegative_decomposition_l668_668522

theorem polynomial_nonnegative_decomposition
  (p : RealPolynomial)
  (h : ∀ x ≥ 0, p.eval x ≥ 0) :
  ∃ A B : RealPolynomial, p = A^2 + X * B^2 :=
sorry

end polynomial_nonnegative_decomposition_l668_668522


namespace smallest_integer_n_exists_l668_668504

-- Define the conditions
def lcm_gcd_correct_division (a b : ℕ) : Prop :=
  (lcm a b) / (gcd a b) = 44

-- Define the main problem
theorem smallest_integer_n_exists : ∃ n : ℕ, lcm_gcd_correct_division 60 n ∧ 
  (∀ k : ℕ, lcm_gcd_correct_division 60 k → k ≥ n) :=
begin
  sorry
end

end smallest_integer_n_exists_l668_668504


namespace g_of_3_l668_668312

def f (x : ℝ) : ℝ := 1 - 2 * x

def g (x : ℝ) : ℝ := if x ≠ 0 then (x^2 - 1) / x^2 else 0

theorem g_of_3 : g 3 = 0 := 
  by 
  sorry

end g_of_3_l668_668312


namespace total_students_l668_668431

-- Definitions based on the conditions:
def yoongi_left : ℕ := 7
def yoongi_right : ℕ := 5

-- Theorem statement that proves the total number of students given the conditions
theorem total_students (y_left y_right : ℕ) : y_left = yoongi_left -> y_right = yoongi_right -> (y_left + y_right - 1) = 11 := 
by
  intros h1 h2
  rw [h1, h2]
  sorry

end total_students_l668_668431


namespace find_PB_l668_668757

theorem find_PB (PA PT AB PB : ℝ) : PA = 5 → PT = AB - 2 * PA → PA < PB → PA * PB = PT^2 → PB = 30 :=
begin
  intros,
  sorry,
end

end find_PB_l668_668757


namespace vasya_days_without_purchase_l668_668916

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668916


namespace triangle_ratios_l668_668368

variable {A B C D E F : Type*}
variable {BC_ratio AE_ratio : ℚ}

-- Conditions given in the problem
def D_divides_BC (D B C : Type*) := BC_ratio = 2 / 3
def E_divides_AE (E A B : Type*) := AE_ratio = 1 / 3
def F_intersection (A D E C F : Type*) := (true) -- F is defined as the intersection, implying it exists and is unique.

-- Proof statement
theorem triangle_ratios (BC_ratio AE_ratio : ℚ) (hD_BC : D_divides_BC D B C) (hE_AE : E_divides_AE E A B)
    (hF : F_intersection A D E C F) :
    (EF FC : ℚ) (AF FD : ℚ) (h_ratio_EF_FC : EF / FC = 3 / 2) (h_ratio_AF_FD : AF / FD = 3):
    (EF / FC) + (AF / FD) = 9 / 2 :=
by
  sorry

end triangle_ratios_l668_668368


namespace dwarfs_truthful_count_l668_668586

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l668_668586


namespace muffin_expense_l668_668250

theorem muffin_expense (B D : ℝ) 
    (h1 : D = 0.90 * B) 
    (h2 : B = D + 15) : 
    B + D = 285 := 
    sorry

end muffin_expense_l668_668250


namespace question1_question2_l668_668697

variables (a b : ℝ → ℝ → ℝ)
variables (θ : ℝ)
variables (parallel : a ≠ 0 ∧ b ≠ 0 ∧ ∃ k : ℝ, a = k • b)
variables (perpendicular : a ≠ 0 ∧ b ≠ 0 ∧ (a - b) ⬝ a = 0)

def magnitude (v : ℝ → ℝ → ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def question1_answer : ℝ :=
if parallel then ±(magnitude a * magnitude b) else 0

noncomputable def question2_answer : ℝ :=
if perpendicular then real.arccos (1 / real.sqrt 2) else 0

theorem question1 : 
  magnitude a = 1 → magnitude b = real.sqrt 2 → 
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ k : ℝ, a = k • b → 
  a ⬝ b = ±real.sqrt 2 :=
sorry

theorem question2 : 
  magnitude a = 1 → magnitude b = real.sqrt 2 → 
  a ≠ 0 ∧ b ≠ 0 ∧ (a - b) ⬝ a = 0 → 
  θ = real.degrees 45 :=
sorry

end question1_question2_l668_668697


namespace part_a_circumcircle_through_A_part_b_orthocenter_on_BC_l668_668382

noncomputable theory

variables {A B C X : Type} [EuclideanSpace Type One] (triangle_ABC : Triangle A B C)
  (X_on_BC : X ∈ Line B C) -- X is an arbitrary point on BC
  (T : Triangle,
    formed_by_bisectors : (AngleBisector.triangle_AngleBisector_triangle_ABC_ABC A ∧
                           AngleBisector.triangle_AngleBisector_triangle_ABC_ACB C ∧
                           AngleBisector.triangle_AngleBisector_triangle_ABC_AXC X))

-- Part (a)
theorem part_a_circumcircle_through_A :
  CircumscribedCircle T A := sorry

-- Part (b)
theorem part_b_orthocenter_on_BC :
  Orthocenter_lying_on_BC T :=
  sorry

end part_a_circumcircle_through_A_part_b_orthocenter_on_BC_l668_668382


namespace no_segment_equal_angle_in_regular_pentagon_l668_668746

theorem no_segment_equal_angle_in_regular_pentagon (P : Type)
  [f : fintype P] [decidable_eq P] (hP : ∀ (A B C : P), 
  ∃ (c : P), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
  is_regular_pentagon P ∧ (∃ (X Y : P), segment_viewed_from_all_vertices_same_angle (X Y : segment P) A B C) → False):
  False :=
sorry

end no_segment_equal_angle_in_regular_pentagon_l668_668746


namespace polynomial_has_real_root_l668_668652

noncomputable def P : Polynomial ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ) (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
variables (h_eq : ∀ x : ℝ, P.eval (a1 * x + b1) + P.eval (a2 * x + b2) = P.eval (a3 * x + b3))

theorem polynomial_has_real_root : ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l668_668652


namespace v_2008_is_7979_l668_668022

-- Defining the sequence based on the problem conditions
def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0
  else let k := ((-1 + Int.sqrt(1 + 8 * n)) / 2).to_nat + 1 in
       let offset := n - (k * (k - 1)) / 2 in
       2 * k ^ 2 + k + 1 + (offset - 1) * 4

-- Proving v_{2008} == 7979
theorem v_2008_is_7979 : sequence 2007 = 7979 := 
  sorry

end v_2008_is_7979_l668_668022


namespace noah_large_paintings_last_month_l668_668034

-- problem definitions
def large_painting_price : ℕ := 60
def small_painting_price : ℕ := 30
def small_paintings_sold_last_month : ℕ := 4
def sales_this_month : ℕ := 1200

-- to be proven
theorem noah_large_paintings_last_month (L : ℕ) (last_month_sales_eq : large_painting_price * L + small_painting_price * small_paintings_sold_last_month = S) 
   (this_month_sales_eq : 2 * S = sales_this_month) :
  L = 8 :=
sorry

end noah_large_paintings_last_month_l668_668034


namespace regular_polygon_sides_l668_668180

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = 150 ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l668_668180


namespace binary_to_decimal_l668_668999

theorem binary_to_decimal : let binary := "111.11" in binary.to_decimal = 7.75 := sorry

end binary_to_decimal_l668_668999


namespace triangle_side_ratio_l668_668850

theorem triangle_side_ratio (α β γ : ℝ) (a b c : ℝ) 
  (h_angles_sum : α + β + γ = π)
  (h_tangent_ratio : (tan α) / (tan β) = 1/2 ∧ (tan β) / (tan γ) = 2/3):
  a / b = sqrt(5) / (2 * sqrt(2)) ∧ b / c = (2 * sqrt(2)) / 3 ∧ a / c = sqrt(5) / 3 :=
sorry

end triangle_side_ratio_l668_668850


namespace age_equivalence_l668_668323

variable (x : ℕ)

theorem age_equivalence : ∃ x : ℕ, 60 + x = 35 + x + 11 + x ∧ x = 14 :=
by
  sorry

end age_equivalence_l668_668323


namespace frac_not_suff_nec_l668_668264

theorem frac_not_suff_nec {a b : ℝ} (hab : a / b > 1) : 
  ¬ ((∀ a b : ℝ, a / b > 1 → a > b) ∧ (∀ a b : ℝ, a > b → a / b > 1)) :=
sorry

end frac_not_suff_nec_l668_668264


namespace sunscreen_cost_l668_668862

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end sunscreen_cost_l668_668862


namespace value_of_x_l668_668892

-- Define the conditions
variable (C S x : ℝ)
variable (h1 : 20 * C = x * S)
variable (h2 : (S - C) / C * 100 = 25)

-- Define the statement to be proved
theorem value_of_x : x = 16 :=
by
  sorry

end value_of_x_l668_668892


namespace correct_calculation_is_B_l668_668125

theorem correct_calculation_is_B : 
  (∃ (x : ℝ), (x = (∛((1:ℝ)/8) ∧ x = (1/2)) ∨ (x = ∛((-8)^2) ∧ x = 4) ∨
  (x = ∛((-3)^3) ∧ x = 3) ∨ (x = -∛(-(2)^3) ∧ x = -2)) ∧ 
  x = ∛((-8)^2) ∧ x = 4) :=
by
  sorry

end correct_calculation_is_B_l668_668125


namespace cyclist_rate_l668_668169

theorem cyclist_rate 
  (rate_hiker : ℝ := 4)
  (wait_time_1 : ℝ := 5 / 60)
  (wait_time_2 : ℝ := 10.000000000000002 / 60)
  (hiker_distance : ℝ := rate_hiker * wait_time_2)
  (cyclist_distance : ℝ := hiker_distance)
  (cyclist_rate := cyclist_distance / wait_time_1) :
  cyclist_rate = 8 := by 
sorry

end cyclist_rate_l668_668169


namespace vasya_days_without_purchase_l668_668915

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668915


namespace relationship_among_abc_l668_668388

namespace ProofProblem

noncomputable def a : ℝ := 2 ^ (-0.6)
noncomputable def b : ℝ := 0.5 ^ 3.1
noncomputable def c : ℝ := Real.sin (5 * Real.pi / 6)

theorem relationship_among_abc : b < c ∧ c < a :=
by
  sorry

end ProofProblem

end relationship_among_abc_l668_668388


namespace number_of_truthful_dwarfs_l668_668584

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l668_668584


namespace owner_overtakes_thief_l668_668546

theorem owner_overtakes_thief :
  let thief_speed_initial := 45 -- kmph
  let discovery_time := 0.5 -- hours
  let owner_speed := 50 -- kmph
  let mud_road_speed := 35 -- kmph
  let mud_road_distance := 30 -- km
  let speed_bumps_speed := 40 -- kmph
  let speed_bumps_distance := 5 -- km
  let traffic_speed := 30 -- kmph
  let head_start_distance := thief_speed_initial * discovery_time
  let mud_road_time := mud_road_distance / mud_road_speed
  let speed_bumps_time := speed_bumps_distance / speed_bumps_speed
  let total_distance_before_traffic := mud_road_distance + speed_bumps_distance
  let total_time_before_traffic := mud_road_time + speed_bumps_time
  let distance_owner_travelled := owner_speed * total_time_before_traffic
  head_start_distance + total_distance_before_traffic < distance_owner_travelled →
  discovery_time + total_time_before_traffic = 1.482 :=
by sorry


end owner_overtakes_thief_l668_668546


namespace isosceles_triangles_with_perimeter_27_count_l668_668705

theorem isosceles_triangles_with_perimeter_27_count :
  ∃ n, (∀ (a : ℕ), 7 ≤ a ∧ a ≤ 13 → ∃ (b : ℕ), b = 27 - 2*a ∧ b < 2*a) ∧ n = 7 :=
sorry

end isosceles_triangles_with_perimeter_27_count_l668_668705


namespace enclosed_area_of_line_l668_668666

theorem enclosed_area_of_line {θ : ℝ} (h : θ ∈ [0, 2*real.pi]) : 
  (let l (θ : ℝ) := λ (x y : ℝ), x * real.cos θ + y * real.sin θ = 4 + real.sqrt 2 * real.sin (θ + real.pi / 4) in
   let area := 16 * real.pi in
   ∃ (x y : ℝ) r, (l θ x y) ∧ r = 4 ∧ r^2 * real.pi = area) :=
sorry

end enclosed_area_of_line_l668_668666


namespace real_solutions_count_l668_668326

theorem real_solutions_count : 
    ∃ y : ℝ, ((2 * y + 5)^2 - 7 = -abs (3 * y + 1)) → 
             (((2 * y + 5)^2 - 7 = -abs (3 * y + 1)) → 
              ((2 * y + 5)^2 ≥ 7) → 
              4 * y^2 + 20 * y + 18 = abs (3 * y + 1)^2) → 
              2 * y^2 - 7 * y - 17 = 0) → 
             sorry := sorry

end real_solutions_count_l668_668326


namespace cannot_divide_polygon_50_equal_parts_l668_668784

open Set

theorem cannot_divide_polygon_50_equal_parts
  (P : Set (ℤ × ℤ)) -- represents the polygon on the grid
  (area_P : ∀ P, ∑ (x, y) in P, 1 = 100) -- area of the polygon is 100 cells
  (div_2 : ∃ P₁ P₂, is_partition P (P₁ ∪ P₂) ∧ ∑ (x, y) in P₁, 1 = 50 ∧ ∑ (x, y) in P₂, 1 = 50)
  (div_25 : ∃ Ps : Finset (Set (ℤ × ℤ)), Ps.card = 25 ∧ ∀ Q ∈ Ps, ∑ (x, y) in Q, 1 = 4)
  : ¬ (∃ Ps' : Finset (Set (ℤ × ℤ)), Ps'.card = 50 ∧ ∀ Q ∈ Ps', ∑ (x, y) in Q, 1 = 2) :=
sorry

end cannot_divide_polygon_50_equal_parts_l668_668784


namespace count_not_divide_g_l668_668762

def proper_divisors (n : ℕ) : finset ℕ :=
  (finset.range n).filter (λ i, i > 0 ∧ n % i = 0)

noncomputable def g (n : ℕ) : ℕ :=
  (proper_divisors n).prod id

def not_divide_g (n : ℕ) : Prop :=
  ¬ n ∣ g n

theorem count_not_divide_g (lower upper : ℕ) (h_range : 2 ≤ lower) (h_upper : upper ≤ 100) :
  (finset.range (upper + 1)).filter (λ n, (lower ≤ n) ∧ not_divide_g n).card = 29 :=
by
  sorry

end count_not_divide_g_l668_668762


namespace min_triangles_cover_l668_668120

theorem min_triangles_cover (s₁ s₂ : ℝ) (h₁ : s₁ = 1) (h₂ : s₂ = 10) :
  100 = (s₂ / s₁) ^ 2 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end min_triangles_cover_l668_668120


namespace find_z_l668_668673

def z : ℂ := (4 + real.sqrt 2) + 2 * (1 - real.sqrt 2) * complex.I

theorem find_z (z_condition : ℂ → Prop) : 
  z_condition (z := z) := 
sorry

namespace complex_number

open complex

example : ∃ z : ℂ, z / (2 + I) = 2 - real.sqrt 2 * I ∧ z = (4 + real.sqrt 2) + 2 * (1 - real.sqrt 2) * I :=
by
  use (4 + real.sqrt 2) + 2 * (1 - real.sqrt 2) * I
  split
  { -- First part is to verify the given condition
    sorry },
  { -- Second part is to check the given result
    refl }
end complex_number

end find_z_l668_668673


namespace rate_of_interest_l668_668806

theorem rate_of_interest (
  interest_earned : ℝ,
  total_amount : ℝ,
  years : ℕ,
  principal_amount : ℝ,
) : 
  interest_earned = 420 →
  total_amount = 2419.999999999998 →
  years = 2 →
  principal_amount = total_amount - interest_earned →
  real.rpow  (1 + 10 / 100) years = total_amount / principal_amount :=
sorry

end rate_of_interest_l668_668806


namespace find_x_in_vector_equation_l668_668669

theorem find_x_in_vector_equation 
  (O A B C P : Type)
  [AddGroup V] [Module ℝ V] [InnerProductSpace ℝ V]
  (hP : ∃ a b c : ℝ, a + b + c = 1 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ P = a • A + b • B + c • C)
  (x : ℝ)
  (h_op : O = (1/2) • A + (1/3) • B + x • C) :
  x = 1/6 :=
sorry

end find_x_in_vector_equation_l668_668669


namespace initial_sugar_amount_l668_668832

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end initial_sugar_amount_l668_668832


namespace tablet_screen_area_difference_l668_668139

theorem tablet_screen_area_difference (d1 d2 : ℝ) (A1 A2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 7) :
  A1 - A2 = 7.5 :=
by
  -- Note: The proof is omitted as the prompt requires only the statement.
  sorry

end tablet_screen_area_difference_l668_668139


namespace fibonacci_b_eq_7_l668_668094

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => sequence (n + 1) + sequence (n + 2)

theorem fibonacci_b_eq_7 :
  (∑ i in Finset.range 10, sequence (i + 1)) = 11 * sequence 7 := by
  sorry

end fibonacci_b_eq_7_l668_668094


namespace range_of_k_l668_668689

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (Real.exp x / x) + k * ((Real.log x) - x)

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f(x, k) = f x k → x = 1) →
  (k ≤ Real.exp 1) := 
by
  -- Here we include the necessary reasoning to complete the proof.
  sorry

end range_of_k_l668_668689


namespace marble_arrangements_count_l668_668886

-- Definitions
def marbles : Finset String := {"Aggie", "Bumblebee", "Steelie", "Tiger", "Clearie"}
def not_adjacent (x y : String) (lst : List String) : Prop := ¬(lst.indexOf x + 1 = lst.indexOf y ∨ lst.indexOf y + 1 = lst.indexOf x)
def valid_arrangements (lst : List String) : Prop := 
  lst.toFinset = marbles ∧ not_adjacent "Steelie" "Tiger" lst ∧ not_adjacent "Bumblebee" "Clearie" lst

-- Theorem Statement
theorem marble_arrangements_count : 
  (Finset.filter valid_arrangements (Finset.permutations marbles.toList)).card = 72 := 
  by sorry

end marble_arrangements_count_l668_668886


namespace number_of_truthful_dwarfs_l668_668597

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l668_668597


namespace intersection_M_N_l668_668694

def M : set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : set ℝ := {x | -2 < x ∧ x ≤ 4}

theorem intersection_M_N :
  M ∩ N = {-1, 3} :=
sorry

end intersection_M_N_l668_668694


namespace test_scores_l668_668192

theorem test_scores (n : ℕ) (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → a i ≠ a j) → -- All students scored different points
  (∑ i in Finset.range n, a i = 119) → -- Total sum is 119
  (∑ i in Finset.range 3, a i = 23) → -- Sum of three smallest scores is 23
  (∑ i in (Finset.range n).filter (λ k, k ≥ n - 3), a i = 49) → -- Sum of three largest scores is 49
  n = 10 ∧ a (n - 1) = 18 := -- The number of students is 10 and the top score is 18
by
  sorry

end test_scores_l668_668192


namespace temperature_equivalence_l668_668418

theorem temperature_equivalence :
  ∃ T : ℝ, T = 1.8 * T + 32 ∧ T = -40 :=
begin
  use -40,
  split,
  { convert (eq.trans (show (-40 : ℝ) = 1.8 * (-40) + 32, by norm_num) rfl); sorry },
  { norm_num }
end

end temperature_equivalence_l668_668418


namespace find_a_l668_668219

def distance_from_point_to_line (P : ℝ × ℝ) (a1 b1 c1 : ℝ) : ℝ :=
  |a1 * P.1 + b1 * P.2 + c1| / Math.sqrt (a1^2 + b1^2)

def distance_curve_to_line (f : ℝ → ℝ) (g : ℝ → ℝ) (h1 h2 : ℝ × ℝ) : ℝ :=
  let d := distance_from_point_to_line h1.fst h1.snd g h2.fst h2.snd 1
  d - f

theorem find_a (a : ℝ) :
  let center_C2 := (0, -4)
  let radius_C2 := real.sqrt 2
  let distance_C2 := distance_from_point_to_line center_C2 1 (-1) 0 - radius_C2
  distance_curve_to_line (λ x, x^2 + a) (λ x, x) (1, 1) (distance_C2, distance_C2) = real.sqrt 2
  → a = 9/4 :=
sorry -- proof here

end find_a_l668_668219


namespace circle_center_radius_l668_668685

def center_and_radius_of_circle (x y : ℝ) : Prop :=
  ∃ h k r, x^2 + y^2 + 2 * x - 4 * y - 4 = 0 ∧ h = -1 ∧ k = 2 ∧ r = 3

theorem circle_center_radius :
  ∃ h k r, (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y - 4 = 0 → h = -1 ∧ k = 2 ∧ r = 3) :=
by {
  sorry,
}

end circle_center_radius_l668_668685


namespace evaluate_expression_l668_668875

theorem evaluate_expression :
  (( (3 + 1)⁻¹ + 1)⁻¹ + 1 - 1)⁻¹ + 1 = 9 / 4 :=
by sorry

end evaluate_expression_l668_668875


namespace area_of_triangle_formed_by_lines_and_vertical_line_l668_668481

theorem area_of_triangle_formed_by_lines_and_vertical_line
  (m1 m2 : ℝ)
  (intersect : ℝ × ℝ)
  (vertical_x : ℝ)
  (h_m1 : m1 = 1 / 4)
  (h_m2 : m2 = 5 / 4)
  (h_intersect : intersect = (1, 1))
  (h_vertical_x : vertical_x = 5) :
  let line1 (x : ℝ) := m1 * (x - 1) + 1 in
  let line2 (x : ℝ) := m2 * (x - 1) + 1 in
  let y1 := line1 vertical_x in
  let y2 := line2 vertical_x in
  let triangle_base := vertical_x - intersect.1 in
  let triangle_height := y2 - y1 in
  1 / 2 * triangle_base * triangle_height = 8 :=
by
  sorry

end area_of_triangle_formed_by_lines_and_vertical_line_l668_668481


namespace prob_15_heartsuit_25_l668_668214

variable {α : Type _}
variable [LinearOrderedField α] -- We require a mathematical context that supports positive real numbers and their properties.

-- Define the operation heartsuit
def heartsuit (x y : α) : α

-- Define the necessary conditions
axiom heartsuit_prop1 (x y : α) (h_x : 0 < x) (h_y : 0 < y) : 
  heartsuit (x * y) y = x * (heartsuit y y)

axiom heartsuit_prop2 (x : α) (h_x : 0 < x) : 
  heartsuit (heartsuit x 1) x = heartsuit x 1

axiom heartsuit_base : 
  heartsuit 1 1 = (1 : α)

-- Now state the problem:
theorem prob_15_heartsuit_25 : heartsuit 15 25 = (375 : α) := 
sorry

end prob_15_heartsuit_25_l668_668214


namespace find_n_in_arithmetic_sequence_l668_668284

theorem find_n_in_arithmetic_sequence 
  (a : ℕ → ℕ)
  (a_1 : ℕ)
  (d : ℕ) 
  (a_n : ℕ) 
  (n : ℕ)
  (h₀ : a_1 = 11)
  (h₁ : d = 2)
  (h₂ : a n = a_1 + (n - 1) * d)
  (h₃ : a n = 2009) :
  n = 1000 := 
by
  -- The proof steps would go here
  sorry

end find_n_in_arithmetic_sequence_l668_668284


namespace recipe_flour_total_l668_668775

theorem recipe_flour_total (flour_added : ℕ) (flour_to_add : ℕ) : 
    flour_added = 6 → flour_to_add = 4 → (flour_added + flour_to_add) = 10 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end recipe_flour_total_l668_668775


namespace intersection_shape_is_rectangle_l668_668573

noncomputable def curve1 (x y : ℝ) : Prop := x * y = 16
noncomputable def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 34

theorem intersection_shape_is_rectangle (x y : ℝ) :
  (curve1 x y ∧ curve2 x y) → 
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    (curve1 p1.1 p1.2 ∧ curve1 p2.1 p2.2 ∧ curve1 p3.1 p3.2 ∧ curve1 p4.1 p4.2) ∧
    (curve2 p1.1 p1.2 ∧ curve2 p2.1 p2.2 ∧ curve2 p3.1 p3.2 ∧ curve2 p4.1 p4.2) ∧ 
    (dist p1 p2 = dist p3 p4 ∧ dist p2 p3 = dist p4 p1) ∧ 
    (∃ m : ℝ, p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.1 ≠ m ∧ p2.1 ≠ m) := sorry

end intersection_shape_is_rectangle_l668_668573


namespace number_of_n_S_S_n_l668_668632

noncomputable def S (n : ℕ) : ℕ :=
  let lg_n := (Real.log (n : ℝ) / Real.log 10).floor.toNat
  let base := n / 10^lg_n
  let rest := n % 10^lg_n
  base + 10 * rest

theorem number_of_n_S_S_n (h : 1 ≤ n ∧ n ≤ 5000) :
  (S (S n) = n) → ((List.range 5000).countp (λ n => S (S n) = n) = 135) :=
sorry

end number_of_n_S_S_n_l668_668632


namespace equal_angles_AOC1_MPB1_l668_668423

-- Definitions of points and setup
variables {A B C B1 C1 P O M : Type}

-- Required conditions and assumptions as hypotheses
variables (triangle_ABC : Triangle A B C)
variables (point_B1_on_AC : LiesOn B1 AC)
variables (point_C1_on_AB : LiesOn C1 AB)
variables (intersection_at_P : Intersects (Segment B B1) (Segment C C1) P)
variables (incenter_O : Incenter O (Triangle A B1 C1))
variables (M_touchpoint : Touchpoint O B1 C1 M)
variables (OP_perpendicular_BB1 : Perpendicular (LineThrough O P) (LineThrough B B1))

-- Formal statement of the theorem to be proved
theorem equal_angles_AOC1_MPB1
  (triangle_ABC : Triangle A B C)
  (point_B1_on_AC : LiesOn B1 AC)
  (point_C1_on_AB : LiesOn C1 AB)
  (intersection_at_P : Intersects (Segment B B1) (Segment C C1) P)
  (incenter_O : Incenter O (Triangle A B1 C1))
  (M_touchpoint : Touchpoint O B1 C1 M)
  (OP_perpendicular_BB1 : Perpendicular (LineThrough O P) (LineThrough B B1)) :
  Angle A O C1 = Angle M P B1 := 
  sorry

end equal_angles_AOC1_MPB1_l668_668423


namespace Tom_marble_choices_l668_668479

theorem Tom_marble_choices :
  let total_marbles := 18
  let special_colors := 4
  let choose_one_from_special := (Nat.choose special_colors 1)
  let remaining_marbles := total_marbles - special_colors
  let choose_remaining := (Nat.choose remaining_marbles 5)
  choose_one_from_special * choose_remaining = 8008
:= sorry

end Tom_marble_choices_l668_668479


namespace total_students_in_school_l668_668348

noncomputable def total_students (girls boys : ℕ) (ratio_girls boys_ratio : ℕ) : ℕ :=
  let parts := ratio_girls + boys_ratio
  let students_per_part := girls / ratio_girls
  students_per_part * parts

theorem total_students_in_school (girls : ℕ) (ratio_girls boys_ratio : ℕ) (h1 : ratio_girls = 5) (h2 : boys_ratio = 8) (h3 : girls = 160) :
  total_students girls boys_ratio ratio_girls = 416 :=
  by
  -- proof would go here
  sorry

end total_students_in_school_l668_668348


namespace curve_is_parabola_l668_668619

theorem curve_is_parabola :
  ∀ (r θ : ℝ) (x y : ℝ),
  r = 1 / (1 - real.sin θ) →
  x = r * real.cos θ →
  y = r * real.sin θ →
  x^2 = 2 * y + 1 :=
by
  intros r θ x y h1 h2 h3
  sorry

end curve_is_parabola_l668_668619


namespace length_of_segment_AB_l668_668626

theorem length_of_segment_AB :
  let A := (sqrt 2, -1)
  let B := (sqrt 2, 1)
  let hyperbola := λ (x y : ℝ), x^2 - y^2 = 1
  let line := λ x, x = sqrt 2
  (hyperbola A.1 A.2) ∧ (hyperbola B.1 B.2) ∧ (line A.1) ∧ (line B.1) →
  |B.2 - A.2| = 2 :=
by 
  intros A B hyperbola line h
  sorry

end length_of_segment_AB_l668_668626


namespace student_total_score_l668_668352

-- Define the conditions.
def correct_answer_marks : ℕ := 4
def wrong_answer_penalty : ℕ := 2
def total_questions : ℕ := 150
def correct_answers : ℕ := 120

-- We need to prove that the total score is 420.
theorem student_total_score :
  let incorrect_answers := total_questions - correct_answers in
  let total_positive_marks := correct_answers * correct_answer_marks in
  let total_negative_marks := incorrect_answers * wrong_answer_penalty in
  total_positive_marks - total_negative_marks = 420 := by
  sorry

end student_total_score_l668_668352


namespace adam_goats_l668_668190

def number_of_goats_adam (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  (c = 13) ∧ (c = b - 6) ∧ (b = 2 * a + 5)

theorem adam_goats {a b c : ℕ} (hc : c = 13) (h1 : c = b - 6) (h2 : b = 2 * a + 5) :
  a = 7 :=
by
  have h3 : b = 19 := by
    rw [h1, hc]
    simp
  have h4 : 2 * a + 5 = 19 := by
    rw [←h3, h2]
  have h5 : 2 * a = 14 := by
    rw [h4, add_comm] at *
    apply eq_sub_iff_add_eq.mpr
    exact eq_of_sub_eq h4
  have h6 : a = 7 := eq_of_mul_eq_mul_left (by norm_num) h5
  exact h6

end adam_goats_l668_668190


namespace cost_of_600_candies_l668_668961

-- Definitions based on conditions
def costOfBox : ℕ := 6       -- The cost of one box of 25 candies in dollars
def boxSize   : ℕ := 25      -- The number of candies in one box
def cost (n : ℕ) : ℕ := (n / boxSize) * costOfBox -- The cost function for n candies

-- Theorem to be proven
theorem cost_of_600_candies : cost 600 = 144 :=
by sorry

end cost_of_600_candies_l668_668961


namespace simplify_ratios_32_48_simplify_ratios_015_3_simplify_ratios_11_12_3_8_l668_668055

noncomputable def simplify_ratios (a b : ℚ) : ℚ × ℚ :=
  let d := gcd a.natAbs b.natAbs
  (a / d, b / d)

theorem simplify_ratios_32_48 : simplify_ratios (32 : ℚ) 48 = (2, 3) := by
  sorry

theorem simplify_ratios_015_3 : simplify_ratios (0.15 : ℚ) 3 = (1, 20) := by
  sorry

theorem simplify_ratios_11_12_3_8 : simplify_ratios (11 / 12) (3 / 8) = (22, 9) := by
  sorry

end simplify_ratios_32_48_simplify_ratios_015_3_simplify_ratios_11_12_3_8_l668_668055


namespace total_cost_is_21_l668_668233

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end total_cost_is_21_l668_668233


namespace correct_operation_l668_668129

theorem correct_operation (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 :=
by
  sorry

end correct_operation_l668_668129


namespace largest_palindrome_not_five_digit_l668_668623

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def largest_three_digit_palindrome_not_five_digit_product_with_101 : ℕ :=
  979

theorem largest_palindrome_not_five_digit :
  ∃ n : ℕ, n = largest_three_digit_palindrome_not_five_digit_product_with_101 ∧
           n > 99 ∧ n < 1000 ∧ 
           is_palindrome n ∧
           ¬ is_palindrome (101 * n) ∧
           ∀ m, m > 99 → m < 1000 → is_palindrome m → ¬ is_palindrome (101 * m) → m ≤ n :=
begin
  use 979,
  split,
  { refl },
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { show is_palindrome 979, sorry },
  split,
  { show ¬ is_palindrome (101 * 979), sorry },
  { intros m hm1 hm2 hm3 hm4,
    show m ≤ 979, 
    sorry }
end

end largest_palindrome_not_five_digit_l668_668623


namespace transportation_charges_l668_668793

theorem transportation_charges 
  (purchase_price repair_cost selling_price : ℕ)
  (h_purchase : purchase_price = 11000)
  (h_repair : repair_cost = 5000)
  (h_selling : selling_price = 25500)
  (profit_rate : ℚ := 1.5) :
  ∃ T : ℕ, T = 1000 :=
by
  let total_cost_before_transport := purchase_price + repair_cost
  have h1: total_cost_before_transport = 16000 := by
    rw [h_purchase, h_repair]
    norm_num
  let total_cost_including_transport := total_cost_before_transport + T
  have h2: selling_price = profit_rate * total_cost_including_transport := by
    rw [h_selling]
    sorry
  have h3: 25500 = profit_rate * (total_cost_before_transport + T) := by
    rw [h1, h2]
    sorry
  use T
  sorry

end transportation_charges_l668_668793


namespace curve_equivalence_l668_668731

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 / Real.sqrt 5 * t, Real.sqrt 5 + 2 / Real.sqrt 5 * t)

def polar_curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos (2 * θ) + 4 = 0

def cartesian_curve_C (x y : ℝ) : Prop :=
  y^2 - x^2 = 4

def point_A : ℝ × ℝ := (0, Real.sqrt 5)

theorem curve_equivalence :
  (∀ (ρ θ : ℝ), polar_curve_C ρ θ ↔ cartesian_curve_C (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∀ (t1 t2 : ℝ), (∃ t1 t2 : ℝ, parametric_line t1 ∈ {p : ℝ × ℝ | cartesian_curve_C p.1 p.2} ∧ 
                                    parametric_line t2 ∈ {p : ℝ × ℝ | cartesian_curve_C p.1 p.2}) → 
                                    (Real.abs (1 / |t1| + 1 / |t2|) = 4)) := 
sorry

end curve_equivalence_l668_668731


namespace largest_prime_factor_of_3913_l668_668486

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), nat.prime p ∧ p ∣ 3913 ∧ (∀ q, nat.prime q ∧ q ∣ 3913 → q ≤ p) ∧ p = 43 :=
sorry

end largest_prime_factor_of_3913_l668_668486


namespace hyperbola_asymptotes_eq_l668_668211

theorem hyperbola_asymptotes_eq (M : ℝ) :
  (4 / 3 = 5 / Real.sqrt M) → M = 225 / 16 :=
by
  intro h
  sorry

end hyperbola_asymptotes_eq_l668_668211


namespace polynomial_factorization_l668_668127

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := 
by sorry

end polynomial_factorization_l668_668127


namespace percentage_graded_on_Tuesday_l668_668779

def total_exams : ℕ := 120
def percentage_graded_on_Monday : ℝ := 0.60
def exams_left_on_Wednesday : ℕ := 12

theorem percentage_graded_on_Tuesday :
  let exams_graded_on_Monday := total_exams * percentage_graded_on_Monday
  let exams_remaining_after_Monday := total_exams - exams_graded_on_Monday.to_nat
  let exams_graded_before_Wednesday := exams_remaining_after_Monday - exams_left_on_Wednesday
  (exams_graded_before_Wednesday.to_nat : ℕ) = 36 ->
  (exams_graded_before_Wednesday.to_nat : ℝ) / exams_remaining_after_Monday.to_nat * 100 = 75 :=
by
  sorry

end percentage_graded_on_Tuesday_l668_668779


namespace largest_prime_factor_of_3913_l668_668489

def is_prime (n : ℕ) := nat.prime n

def prime_factors_3913 := {17, 2, 5, 23}

theorem largest_prime_factor_of_3913 
  (h1 : is_prime 17)
  (h2 : is_prime 2)
  (h3 : is_prime 5)
  (h4 : is_prime 23)
  (h5 : 3913 = 17 * 2 * 5 * 23) : 
  (23 ∈ prime_factors_3913 ∧ ∀ x ∈ prime_factors_3913, x ≤ 23) :=
  by 
    sorry

end largest_prime_factor_of_3913_l668_668489


namespace inequality_solution_set_l668_668682

theorem inequality_solution_set
  (a b c m n : ℝ) (h : a ≠ 0) 
  (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ m < x ∧ x < n)
  (h2 : 0 < m)
  (h3 : ∀ x : ℝ, cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) :
  (cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) := 
sorry

end inequality_solution_set_l668_668682


namespace first_team_cups_l668_668428

theorem first_team_cups (total_cups : ℕ) (second_team : ℕ) (third_team : ℕ) :
  total_cups = 280 → second_team = 120 → third_team = 70 → total_cups - (second_team + third_team) = 90 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end first_team_cups_l668_668428


namespace hypotenuse_of_right_angled_triangle_with_incircle_radius_four_and_angle_forty_five_deg_l668_668543

theorem hypotenuse_of_right_angled_triangle_with_incircle_radius_four_and_angle_forty_five_deg 
  (r : ℝ) (h : ℝ) (x : ℝ) 
  (h_r : r = 4) 
  (h_angle : ∠BAC = 90) 
  (h_b : ∠ABC = 45)
  (h_radius_formula : r = (x + x - h) / 2) 
  (h_isosceles : x = 8 + 4 * Real.sqrt 2) : 
  h = 8 + 8 * Real.sqrt 2 :=
by sorry

end hypotenuse_of_right_angled_triangle_with_incircle_radius_four_and_angle_forty_five_deg_l668_668543


namespace q_implies_not_p_l668_668021

-- Define the conditions p and q
def p (x : ℝ) := x < -1
def q (x : ℝ) := x^2 - x - 2 > 0

-- Prove that q implies ¬p
theorem q_implies_not_p (x : ℝ) : q x → ¬ p x := by
  intros hq hp
  -- Provide the steps of logic here
  sorry

end q_implies_not_p_l668_668021


namespace quadratic_k_value_l668_668272

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end quadratic_k_value_l668_668272


namespace ravish_maximum_marks_l668_668891

theorem ravish_maximum_marks (M : ℝ) (h_pass : 0.40 * M = 80) : M = 200 :=
sorry

end ravish_maximum_marks_l668_668891


namespace percent_decrease_is_correct_l668_668755

def original_price : ℝ := 100
def sale_price : ℝ := 20
def decrease_in_price : ℝ := original_price - sale_price
def percent_decrease : ℝ := (decrease_in_price / original_price) * 100

theorem percent_decrease_is_correct :
  percent_decrease = 80 := by
  sorry

end percent_decrease_is_correct_l668_668755


namespace sum_of_integers_l668_668889

theorem sum_of_integers (m t : ℕ) 
  (hm : m = (list.sum (list.filter (λ n => n % 2 = 1) (list.range (111 + 1)))))
  (ht : t = (list.sum (list.filter (λ n => n % 2 = 0) (list.range (50 + 1))))) :
  m + t = 3786 :=
by
  sorry

end sum_of_integers_l668_668889


namespace compute_BG_l668_668014

noncomputable theory

-- Declare the variables according to the problem statement
variables {A B C D E F G H : Type}
variables [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Segment A B E] [Segment D F H]

-- Define conditions including properties of the octahedron and segments
def regular_octahedron (A B C D E F : Point) (edge_length : ℝ) := 
  -- define the unit side length conditions of the regular octahedron
  sorry 

def point_on_segment (S : Segment A E) (G : Point) := 
  -- define that G lies on segment BE
  sorry 

def planes_divide_into_three_equal_volumes (A D G B C H : Point) :=
  -- define that planes AGD and BCH divide the octahedron into three pieces of equal volume
  sorry 

-- The theorem stating the final result
theorem compute_BG 
  (h_octahedron : regular_octahedron A B C D E F 1)
  (h_point_G : point_on_segment BE G)
  (h_planes : planes_divide_into_three_equal_volumes A D G B C H) :
  BG = (9 - real.sqrt 57) / 6 :=
begin
  sorry 
end

end compute_BG_l668_668014


namespace find_smaller_number_l668_668848

theorem find_smaller_number (a b : ℕ) 
  (h1 : a + b = 15) 
  (h2 : 3 * (a - b) = 21) : b = 4 :=
by
  sorry

end find_smaller_number_l668_668848


namespace translation_of_cos_function_l668_668086

theorem translation_of_cos_function (φ : ℝ) (h₀ : 0 < φ) (h₁ : φ < π)
    (h₂ : ∀ x, cos (2 * x - π / 6) = cos (2 * x + 2 * φ - π / 6)) :
    φ = π / 4 :=
by
  have h : 2 * φ - π / 6 = π / 3 := sorry -- use equating arguments here
  have hφ : φ = π / 4 := sorry -- solve for φ here
  exact hφ

end translation_of_cos_function_l668_668086


namespace range_of_m_l668_668306

theorem range_of_m (m : ℝ) : (∃ f : ℝ → ℝ, 
  f = λ x, x^3 + m * x^2 + (m + 6) * x + 1 ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f' a = 0 ∧ f' b = 0)) ↔ (m < -3 ∨ m > 6) := 
sorry

end range_of_m_l668_668306


namespace rachel_served_correctly_l668_668048

variable wage : ℝ
variable total_earnings : ℝ
variable tip_per_person : ℝ

theorem rachel_served_correctly :
    wage = 12 ∧ total_earnings = 37 ∧ tip_per_person = 1.25 → 
    let tips := total_earnings - wage in
    let people_served := tips / tip_per_person in
    people_served = 20 :=
begin
  sorry
end

end rachel_served_correctly_l668_668048


namespace sum_of_multiples_of_20_and_14_up_to_2014_l668_668551

theorem sum_of_multiples_of_20_and_14_up_to_2014 : 
  (∑ n in (Finset.filter (λ x, (x % 20 = 0) ∧ (x % 14 = 0)) (Finset.range 2015)), n) = 14700 :=
by sorry

end sum_of_multiples_of_20_and_14_up_to_2014_l668_668551


namespace peaches_problem_l668_668960

theorem peaches_problem :
  ∃ d : ℕ, (∃ r u : ℕ, 
    r = 4 + 2 * d - if d ≥ 3 then 3 else 0 ∧ 
    u = 14 - 2 * d ∧ 
    r = u + 7) ∧ d = 5 :=
by 
  sorry

end peaches_problem_l668_668960


namespace log_base_250_2662sqrt10_l668_668959

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variables (a b : ℝ)
variables (h1 : log_base 50 55 = a) (h2 : log_base 55 20 = b)

theorem log_base_250_2662sqrt10 : log_base 250 (2662 * Real.sqrt 10) = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) :=
by
  sorry

end log_base_250_2662sqrt10_l668_668959


namespace initial_volume_proof_l668_668528

noncomputable def initial_volume (V : ℝ) :=
  let new_volume := V + 5 + 15 in
  let initial_jasmine := 0.10 * V in
  let new_jasmine := initial_jasmine + 5 in
  (0.13 * new_volume = new_jasmine) → (V = 80)

theorem initial_volume_proof : ∃ V : ℝ, initial_volume V :=
begin
  use 80,
  dsimp [initial_volume],
  linarith,
end

end initial_volume_proof_l668_668528


namespace change_for_50_cents_using_only_pennies_and_dimes_l668_668708

theorem change_for_50_cents_using_only_pennies_and_dimes : ∃ n : ℕ, n = 4 ∧ -- existence of the number n which is the count of ways to make change
  (∃ f : ℕ → ℕ,  -- existence of a function f which defines the number of ways for each possible number of dimes
    (∀ k : ℕ, f 0 = 1) ∧ -- case where no dimes are used
    (∀ k : ℕ, (1 ≤ k) → k ≤ 4 → f k = 1) ∧ -- cases where 1 to 4 dimes are used
    (∀ k : ℕ, (k ≠ 0) ∧ ((k < 1) ∨ (k > 4)) → f k = 0)) -- cases where using more than 4 dimes is impossible
 := by
sorrry

end change_for_50_cents_using_only_pennies_and_dimes_l668_668708


namespace find_values_l668_668640

theorem find_values (a b: ℝ) (h1: a > b) (h2: b > 1)
  (h3: Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h4: a^b = b^a) :
  a = 4 ∧ b = 2 := 
sorry

end find_values_l668_668640


namespace vasya_days_without_purchase_l668_668919

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668919


namespace negation_of_proposition_l668_668337

theorem negation_of_proposition :
  (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) → (∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) :=
sorry

end negation_of_proposition_l668_668337


namespace vasya_no_purchase_days_l668_668950

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668950


namespace vasya_days_l668_668905

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668905


namespace sign_of_fsum_l668_668648

variable {f : ℝ → ℝ}

theorem sign_of_fsum 
  (h1 : ∀ x : ℝ, f(-x) = -f(x + 4))
  (h2 : ∀ x : ℝ, (2 < x) → (∃ y, f(y) > f(x) ∧ (y < x)))
  (hx1x2 : ℝ) (hx2x1 : ℝ)
  (hx1x2_sum : hx1x2 + hx2x1 < 4)
  (hprod : (hx1x2 - 2) * (hx2x1 - 2) < 0) : f hx1x2 + f hx2x1 < 0 := 
sorry

end sign_of_fsum_l668_668648


namespace relationship_among_a_b_c_l668_668678

noncomputable def f : ℝ → ℝ := sorry

theorem relationship_among_a_b_c
  (h1 : ∀ x, f (1 - x) = f (1 + x))
  (h2 : ∀ x ∈ Ioo (-∞ : ℝ) 0, f (x) + x * deriv f x < 0) :
  let a := (sin (1 / 2)) * f (sin (1 / 2)),
      b := (log 2) * f (log 2),
      c := 2 * f (log (1 / (2 * 1 / 4))) in
  a > b ∧ b > c :=
begin
  sorry
end

end relationship_among_a_b_c_l668_668678


namespace exists_quadratics_l668_668226

-- Defining the quadratic polynomials
def P1(x : ℝ) : ℝ := (x - 3)^2 - 1
def P2(x : ℝ) : ℝ := x^2 - 1
def P3(x : ℝ) : ℝ := (x + 3)^2 - 1

-- Statement to prove
theorem exists_quadratics : 
  (∀ P : (ℝ → ℝ), (∃ a b c : ℝ, P = (λ x : ℝ, a*x^2 + b*x + c) ∧ (b^2 - 4*a*c > 0))) ∧
  (∀ Q : (ℝ → ℝ), Q = (P1 + P2) → (let (a,b,c) := Q in b^2 - 4*a*c < 0)) ∧
  (∀ Q : (ℝ → ℝ), Q = (P1 + P3) → (let (a,b,c) := Q in b^2 - 4*a*c < 0)) ∧
  (∀ Q : (ℝ → ℝ), Q = (P2 + P3) → (let (a,b,c) := Q in b^2 - 4*a*c < 0)) :=
begin
    sorry
end

end exists_quadratics_l668_668226


namespace vasya_days_without_purchases_l668_668898

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668898


namespace alpha_arctan_l668_668019

open Real

theorem alpha_arctan {α : ℝ} (h1 : α ∈ Set.Ioo 0 (π/4)) (h2 : tan (α + (π/4)) = 2 * cos (2 * α)) : 
  α = arctan (2 - sqrt 3) := by
  sorry

end alpha_arctan_l668_668019


namespace oleum_mixture_composition_l668_668141

theorem oleum_mixture_composition :
  let mixture1 := (24.0, 0.85, 0.05, 0.10)
  let mixture2 := (120.0, 0.55, 0.05, 0.40)
  let mixture3 := (32.0, 0.30, 0.20, 0.50)
  let mixture4 := (16.0, 0.60, 0.35, 0.05)

  let mass_total := 24.0 + 120.0 + 32.0 + 16.0
  let h2so4_mass_total :=
    (24.0 * 0.85) + (120.0 * 0.55) + (32.0 * 0.30) + (16.0 * 0.60)
  let so3_mass_total :=
    (24.0 * 0.05) + (120.0 * 0.05) + (32.0 * 0.20) + (16.0 * 0.35)
  let h2s2o7_mass_total :=
    (24.0 * 0.10) + (120.0 * 0.40) + (32.0 * 0.50) + (16.0 * 0.05)
  let h2so4_percentage := (h2so4_mass_total / mass_total) * 100
  let so3_percentage := (so3_mass_total / mass_total) * 100
  let h2s2o7_percentage := (h2s2o7_mass_total / mass_total) * 100

  h2so4_percentage ≈ 55.0 ∧ so3_percentage = 10.0 ∧ h2s2o7_percentage ≈ 35.0
:
  sorry

end oleum_mixture_composition_l668_668141


namespace max_product_of_arithmetic_sequence_l668_668667

noncomputable def max_a1_a20 (a : ℕ → ℝ) : ℝ :=
  if h : (∀ n, ∃ d : ℝ, a(n) = a(0) + n * d) ∧ a(20) > 0 ∧ a(1) > 0 ∧ (20 * a(0) + 190 * (a(1) - a(0)) = 100) then
    max_a1_a20 := max (a(1) * a(20))
  else 0

theorem max_product_of_arithmetic_sequence : 
  ∀ a : ℕ → ℝ, (∀ n, ∃ d : ℝ, a(n) = a(0) + n * d) ∧ 
  a(1) > 0 ∧ 
  a(20) > 0 ∧ 
  (20 * a(0) + 190 * (a(1) - a(0)) = 100) → 
  max_a1_a20 a = 25 :=
begin
  sorry
end

end max_product_of_arithmetic_sequence_l668_668667


namespace find_point_P_l668_668304

def f (x : ℝ) : ℝ := x^4 - 2 * x

def tangent_line_perpendicular (x y : ℝ) : Prop :=
  (f x) = y ∧ (4 * x^3 - 2 = 2)

theorem find_point_P :
  ∃ (x y : ℝ), tangent_line_perpendicular x y ∧ x = 1 ∧ y = -1 :=
sorry

end find_point_P_l668_668304


namespace vasya_days_without_purchase_l668_668926

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668926


namespace integer_valued_polynomial_l668_668047

theorem integer_valued_polynomial {m : ℕ} (f : ℤ → ℤ) :
  (∃ k : ℤ, ∀ i : ℕ, i ≤ m → f (k + i) ∈ ℤ) →
  ∀ x : ℤ, f x ∈ ℤ :=
by
  sorry

end integer_valued_polynomial_l668_668047


namespace number_of_truthful_dwarfs_l668_668580

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l668_668580


namespace test_scores_l668_668191

theorem test_scores (n : ℕ) (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → a i ≠ a j) → -- All students scored different points
  (∑ i in Finset.range n, a i = 119) → -- Total sum is 119
  (∑ i in Finset.range 3, a i = 23) → -- Sum of three smallest scores is 23
  (∑ i in (Finset.range n).filter (λ k, k ≥ n - 3), a i = 49) → -- Sum of three largest scores is 49
  n = 10 ∧ a (n - 1) = 18 := -- The number of students is 10 and the top score is 18
by
  sorry

end test_scores_l668_668191


namespace problem_parts_l668_668717

open Real

-- Define the quadratic equation from the problem.
def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

-- Given conditions
axiom lg_roots : ∃ (m n : ℝ), quadratic 2 (-4) 1 (log 10 m) = 0 ∧ quadratic 2 (-4) 1 (log 10 n) = 0

-- Using Vieta's theorem directly in conclusions
theorem problem_parts (m n : ℝ) (H1 : quadratic 2 (-4) 1 (log 10 m) = 0) (H2 : quadratic 2 (-4) 1 (log 10 n) = 0) :
  m * n = 100 ∧ (log n m + log m n = 6) :=
by
  sorry

end problem_parts_l668_668717


namespace ratio_of_amount_spent_on_movies_to_weekly_allowance_l668_668008

-- Define weekly allowance
def weekly_allowance : ℕ := 10

-- Define final amount after all transactions
def final_amount : ℕ := 11

-- Define earnings from washing the car
def earnings : ℕ := 6

-- Define amount left before washing the car
def amount_left_before_wash : ℕ := final_amount - earnings

-- Define amount spent on movies
def amount_spent_on_movies : ℕ := weekly_allowance - amount_left_before_wash

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Prove the required ratio
theorem ratio_of_amount_spent_on_movies_to_weekly_allowance :
  ratio amount_spent_on_movies weekly_allowance = 1 / 2 :=
by
  sorry

end ratio_of_amount_spent_on_movies_to_weekly_allowance_l668_668008


namespace parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l668_668692

open Real

theorem parabola_tangent_perpendicular_m_eq_one (k : ℝ) (hk : k > 0) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + m) ∧ (y₂ = k * x₂ + m) ∧ ((x₁ / 2) * (x₂ / 2) = -1)) → m = 1 :=
sorry

theorem parabola_min_MF_NF (k : ℝ) (hk : k > 0) :
  (m = 2) → 
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + 2) ∧ (y₂ = k * x₂ + 2) ∧ |(y₁ + 1) * (y₂ + 1)| ≥ 9) :=
sorry

end parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l668_668692


namespace vasya_purchase_l668_668935

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668935


namespace line_equation_exists_l668_668621

theorem line_equation_exists 
  (a b : ℝ) 
  (ha_pos: a > 0)
  (hb_pos: b > 0)
  (h_area: 1 / 2 * a * b = 2) 
  (h_diff: a - b = 3 ∨ b - a = 3) : 
  (∀ x y : ℝ, (x + 4 * y = 4 ∧ (x / a + y / b = 1)) ∨ (4 * x + y = 4 ∧ (x / a + y / b = 1))) :=
sorry

end line_equation_exists_l668_668621


namespace simplify_expr_l668_668056

-- Define the expression
def expr := |-4^2 + 7|

-- State the theorem
theorem simplify_expr : expr = 9 :=
by sorry

end simplify_expr_l668_668056


namespace count_valid_subsets_l668_668213

open Finset

noncomputable def set := {2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_valid_subset (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ 7 ∈ s ∧ s.sum id = 18

theorem count_valid_subsets : 
  (set.filter is_valid_subset).card = 3 :=
by
  sorry

end count_valid_subsets_l668_668213


namespace find_S6_l668_668020

noncomputable def a (n : ℕ) : ℝ := sorry -- the nth term of the geometric sequence
def S (n : ℕ) : ℝ := sorry -- the sum of the first n terms of the geometric sequence

axiom geom_seq (n : ℕ) : a (n + 1) = r * a n

axiom sum_3 : S 3 = 3
axiom sum_8_5 : S 8 - S 5 = -96

theorem find_S6 : S 6 = -21 :=
sorry

end find_S6_l668_668020


namespace telescoping_series_sum_l668_668630

noncomputable def H (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), 1 / (i + 1 : ℝ)

noncomputable def series_term (n : ℕ) : ℝ :=
  (n / (n + 1 : ℝ)) * ((1 / H n) - (1 / H (n + 1)))

theorem telescoping_series_sum :
  ∑' n, series_term n = 1 / 2 :=
sorry

end telescoping_series_sum_l668_668630


namespace arithmetic_sequences_count_l668_668258

theorem arithmetic_sequences_count :
  (∃ (count : ℕ),
    count = (2 * ∑ d in finset.Icc 1 6, (20 - 3 * d)) 
    ∧ count = 114) := 
begin
  use 114,
  split,
  { rw nat.mul_comm,
    rw nat.mul_assoc,
    rw finset.sum_Icc_bot,
    norm_num },
  { refl },
end

end arithmetic_sequences_count_l668_668258


namespace count_random_events_l668_668195

def is_random_event {α : Type} (e : α → Prop) : Prop :=
  ∃ x, ∃ y, x ≠ y ∧ (e x ↔ True) ∧ (e y ↔ False)

def event1 : Prop := is_random_event (λ _, true) -- Throwing two dice twice in a row and getting 2 points both times
def event2 : Prop := ¬ is_random_event (λ _, true) -- Pear falling down
def event3 : Prop := is_random_event (λ _, true) -- Someone winning the lottery
def event4 : Prop := is_random_event (λ _, true) -- Having one daughter already, then having a boy the second time
def event5 : Prop := ¬ is_random_event (λ _, true) -- Water boiling at 90°C under standard atmospheric pressure

theorem count_random_events : event1 ∧ ¬ event2 ∧ event3 ∧ event4 ∧ ¬ event5 → (3 : Nat) := sorry

end count_random_events_l668_668195


namespace waiter_total_customers_l668_668142

theorem waiter_total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) (tables_eq : tables = 6) (women_eq : women_per_table = 3) (men_eq : men_per_table = 5) :
  tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end waiter_total_customers_l668_668142


namespace polynomial_identity_l668_668433

theorem polynomial_identity (g : ℝ → ℝ) : (∀ (x : ℝ), g(x)^2 = 9x^2 - 12x + 4) → (∀ (x : ℝ), g(x) = 3x - 2 ∨ g(x) = -3x + 2) :=
by
  assume h : ∀ (x : ℝ), g(x)^2 = 9x^2 - 12x + 4
  sorry

end polynomial_identity_l668_668433


namespace sum_of_abs_of_coefficients_l668_668665

theorem sum_of_abs_of_coefficients :
  ∃ a_0 a_2 a_4 a_1 a_3 a_5 : ℤ, 
    ((2*x - 1)^5 + (x + 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) ∧
    (|a_0| + |a_2| + |a_4| = 110) :=
by
  sorry

end sum_of_abs_of_coefficients_l668_668665


namespace value_of_k_l668_668084

theorem value_of_k (k : ℝ) (hk : 0 < k) :
  (∀ x ∈ set.Icc 4 6, (k / (x - 2)) ≤ 1) → k = 2 :=
by
  intro h
  -- The proof will go here
  sorry

end value_of_k_l668_668084


namespace impossible_segment_in_pentagon_l668_668752

theorem impossible_segment_in_pentagon :
  ∀ (P : Type) [EuclideanSpace P] [RegularPentagon P],
    ¬ ∃ (XY : Segment P), is_inside_pentagon XY ∧ 
      (∀ (v : Vertex P), ∃ (α : Angle), is_seen_from v α XY) :=
begin
  sorry
end

end impossible_segment_in_pentagon_l668_668752


namespace johns_new_weekly_earnings_l668_668374

-- Define the original weekly earnings and the raise percentage
def original_weekly_earnings : ℝ := 60
def raise_percentage : ℝ := 16.666666666666664 / 100

-- Define the raise amount based on the given conditions
def raise_amount : ℝ := original_weekly_earnings * raise_percentage

-- Define the new weekly earnings after the raise
def new_weekly_earnings : ℝ := original_weekly_earnings + raise_amount

-- Prove that the new weekly earnings are $70 given the conditions
theorem johns_new_weekly_earnings : new_weekly_earnings = 70 := by
  sorry  -- Proof omitted

end johns_new_weekly_earnings_l668_668374


namespace points_in_S_n_cannot_be_partitioned_into_fewer_than_n_paths_l668_668024

def S_n (n : ℕ) : set (ℤ × ℤ) :=
  { p | abs p.fst + abs (p.snd + 1 / 2) < n }

noncomputable def is_path (n : ℕ) (p : list (ℤ × ℤ)) : Prop :=
  p.length > 1 ∧ ∀ i ∈ list.range (p.length - 1), 
  ((p.nth (i + 1)).get_or_else (0,0)) ≠ ((p.nth i).get_or_else (0,0)) ∧ 
  int.dist ((p.nth (i + 1)).get_or_else (0,0)) ((p.nth i).get_or_else (0,0)) = 1

def min_paths (n : ℕ) : ℕ :=
  - sorry 

theorem points_in_S_n_cannot_be_partitioned_into_fewer_than_n_paths (n : ℕ) (h : n > 0) :
  (min_paths n) ≥ n :=
sorry

end points_in_S_n_cannot_be_partitioned_into_fewer_than_n_paths_l668_668024


namespace arithmetic_series_modulo_l668_668231

theorem arithmetic_series_modulo (
  a d : ℤ)
  (h_a : a = 2)
  (h_d : d = 5)
  (h_n : ∀ n, 0 ≤ n ∧ n < 17 → (102 = a + (n-1) * d))
  : ∑ i in (range 21), (a + i * d) % 17 = 11 := by
    sorry

end arithmetic_series_modulo_l668_668231


namespace annual_interest_rate_l668_668030

-- Defining the problem conditions
def investment_initial := 2000
def investment_final := 18000
def time_years := 28
noncomputable def tripling_time (r : ℝ) := 112 / r
noncomputable def tripling_periods (r : ℝ) := time_years / tripling_time(r)

-- Our goal is to prove this statement
theorem annual_interest_rate (r : ℝ) (h1 : tripling_time r ≠ 0) :
  investment_final = investment_initial * (3 ^ tripling_periods r) → r = 8 :=
by
  sorry

end annual_interest_rate_l668_668030


namespace number_of_truthful_gnomes_l668_668606

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l668_668606


namespace angle_bisectors_intersect_on_AC_l668_668269

-- Define a convex quadrilateral ABCD that is not a deltoid
variables {A B C D : Type} [EuclideanGeometry.quadrilateral A B C D]
noncomputable def is_not_deltoid (A B C D : Type) [EuclideanGeometry.quadrilateral A B C D] : Prop := 
¬ (∃ p : Type, EuclideanGeometry.is_symmetric_with_respect_to_diagonal p A B C D ∧ EuclideanGeometry.is_diagonal BD)

-- Define angle bisectors intersection conditions
variables (M : Point) (BD : Diagonal)

-- Assume angle bisectors of angles A and C intersect at M on BD
axiom angle_bisectors_intersect_on_BD (h₁ : EuclideanGeometry.angle_bisector_intersects A C BD M)

-- Prove the angle bisectors of angles B and D intersect on AC
theorem angle_bisectors_intersect_on_AC 
  (ABCD_is_convex : EuclideanGeometry.is_convex_quadrilateral A B C D)
  (ABCD_not_deltoid : is_not_deltoid A B C D)
  (angle_bisectors_AC_intersect_BD : angle_bisectors_intersect_on_BD A B C D M BD) :
  ∃ N : Point, EuclideanGeometry.angle_bisector_intersects B D AC N :=
sorry

end angle_bisectors_intersect_on_AC_l668_668269


namespace range_of_a_l668_668085

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Ici (-3 : ℝ), (sqrt (x + 3) + (1 / (a * x + 2)) > 0)) ↔ (0 < a ∧ a < (2 / 3)) := by
  sorry

end range_of_a_l668_668085


namespace lyka_saving_per_week_l668_668408

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l668_668408


namespace vasya_no_purchase_days_l668_668945

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668945


namespace stephen_number_correct_l668_668062

noncomputable def Stephen_number : ℕ :=
  let m := 11880 in
  m

theorem stephen_number_correct (m : ℕ) 
  (h1 : 216 ∣ m) 
  (h2 : 55 ∣ m) 
  (h3 : 9000 < m ∧ m < 15000) : 
  m = 11880 :=
by
  sorry

end stephen_number_correct_l668_668062


namespace perpendicularity_condition_l668_668649

variables {α : Type*} [LinearOrderedField α]
variables (l : Line α) (P : Plane α)

def perpendicular_to_two_intersecting_lines (l : Line α) (α : Plane α) : Prop :=
  ∃ (a b : Line α), a ≠ b ∧ a ∩ b ≠ ∅ ∧ a ∈ α ∧ b ∈ α ∧ line_perpendicular l a ∧ line_perpendicular l b

def perpendicular_to_plane (l : Line α) (α : Plane α) : Prop :=
  ∀ (m : Line α), m ∈ α → line_perpendicular l m

theorem perpendicularity_condition (l : Line α) (α : Plane α) :
  (perpendicular_to_two_intersecting_lines l α) ↔ (perpendicular_to_plane l α) := sorry

end perpendicularity_condition_l668_668649


namespace election_majority_l668_668727

theorem election_majority (V W L : ℕ) (P : Real) : 
    P = 0.70 ∧ 
    V = 435 ∧ 
    W = Real.toNat (P * V) ∧ 
    L = Real.toNat ((1 - P) * V) ∧ 
    (∀ x, x = Real.toNat _ -> W = 304) ∧ 
    (∀ y, y = Real.toNat _ -> L = 131) ->
    W - L = 173 :=
by
  sorry

end election_majority_l668_668727


namespace red_car_speed_correct_l668_668863

noncomputable def red_car_speed : ℝ :=
  40

theorem red_car_speed_correct (black_car_speed red_car_distance : ℝ) (overtake_time : ℝ)
  (H1 : black_car_speed = 50)
  (H2 : red_car_distance = 30)
  (H3 : overtake_time = 3) :
  let red_car_speed := (black_car_speed * overtake_time - red_car_distance) / overtake_time in
  red_car_speed = 40 := by
{
  simp [H1, H2, H3],
  linarith,
}

end red_car_speed_correct_l668_668863


namespace smallest_positive_angle_l668_668671

-- Definitions of the given conditions
def point_p : ℝ × ℝ := (Real.sqrt 3, -1)
def point_in_fourth_quadrant : Prop := point_p.fst > 0 ∧ point_p.snd < 0

-- Mathematical proofs
theorem smallest_positive_angle :
  point_in_fourth_quadrant → ∃ α : ℝ, α = 11 * Real.pi / 6 ∧ (α > 0) := by
  assume h : point_in_fourth_quadrant
  exists (11 * Real.pi / 6)
  constructor
  { 
    -- The proof that this is the angle. (We skip the proof.)
    sorry 
  }
  {
    -- The proof that the angle is positive.
    linarith
  }

end smallest_positive_angle_l668_668671


namespace prob_at_least_one_A_or_B_l668_668579

open ProbabilityTheory

noncomputable def prob_A : ℝ := 1 / 3
noncomputable def prob_B : ℝ := 1 / 4

axiom independent_events : ProbEventsIndependent (λ ω, A ω) (λ ω, B ω)

theorem prob_at_least_one_A_or_B : P (λ ω, A ω ∨ B ω) = 1 / 2 :=
by
  have prob_not_A : P (λ ω, ¬ A ω) = 1 - prob_A := sorry
  have prob_not_B : P (λ ω, ¬ B ω) = 1 - prob_B := sorry
  have prob_not_A_and_not_B : P (λ ω, ¬ A ω ∧ ¬ B ω) = (1 - prob_A) * (1 - prob_B) := sorry
  have prob_not_A_and_not_B_val : P (λ ω, ¬ A ω ∧ ¬ B ω) = 1 / 2 := by
    rw [prob_not_A_and_not_B]
    norm_num
  calc
    P (λ ω, A ω ∨ B ω)
      = 1 - P(λ ω, ¬ (A ω ∨ B ω)) := by sorry
      = 1 - P(λ ω, ¬ A ω ∧ ¬ B ω) := by sorry
      = 1 - 1 / 2 := by exact eq.symm prob_not_A_and_not_B_val
      = 1 / 2 := by norm_num

end prob_at_least_one_A_or_B_l668_668579


namespace subsequence_length_mn_plus_one_l668_668044

theorem subsequence_length_mn_plus_one
  {α : Type*} [LinearOrder α]
  (a : Fin (m * n + 1) → α)
  (m n : ℕ) :
  ∃ (s : Finset (Fin (m * n + 1))), 
    (s.card = m + 1 ∧ s.pairwise (λ i j, i < j → a i < a j)) ∨ 
    (s.card = n + 1 ∧ s.pairwise (λ i j, i < j → a j < a i)) :=
by
  sorry

end subsequence_length_mn_plus_one_l668_668044


namespace solve_x_l668_668330

theorem solve_x (x : ℝ) (h : (x + 1) ^ 2 = 9) : x = 2 ∨ x = -4 :=
sorry

end solve_x_l668_668330


namespace christy_tanya_spending_ratio_l668_668991

theorem christy_tanya_spending_ratio :
  ∃ (C T : ℝ), 
  T = 340 ∧
  C + T = 1020 ∧
  (C / T) = 2 :=
by
  use 680, 340
  split
  · refl
  split
  · norm_num
  · norm_num

end christy_tanya_spending_ratio_l668_668991


namespace vasya_days_l668_668906

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668906


namespace find_discount_percentage_l668_668976

noncomputable def discount_percentage (P B S : ℝ) (H1 : B = P * (1 - D / 100)) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : ℝ :=
D

theorem find_discount_percentage (P B S : ℝ) (H1 : B = P * (1 - (60 / 100))) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : 
  discount_percentage P B S H1 H2 H3 = 60 := sorry

end find_discount_percentage_l668_668976


namespace sum_of_squares_of_roots_l668_668222

theorem sum_of_squares_of_roots (x₁ x₂ : ℚ) (h : 6 * x₁^2 - 9 * x₁ + 5 = 0 ∧ 6 * x₂^2 - 9 * x₂ + 5 = 0 ∧ x₁ ≠ x₂) : x₁^2 + x₂^2 = 7 / 12 :=
by
  -- Since we are only required to write the statement, we leave the proof as sorry
  sorry

end sum_of_squares_of_roots_l668_668222


namespace factorial_division_example_l668_668292

theorem factorial_division_example : (10! / (5! * 2!)) = 15120 := 
by
  sorry

end factorial_division_example_l668_668292


namespace sin_half_alpha_range_l668_668298

variable {α : ℝ}

-- Conditions
def first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def sin_gt_cos (α : ℝ) : Prop := sin (α / 2) > cos (α / 2)

-- Statement
theorem sin_half_alpha_range (h1 : first_quadrant α) (h2 : sin_gt_cos α) : 
  ∃ (x : ℝ), x = sin (α / 2) ∧ (√2 / 2 < x ∧ x < 1) := sorry

end sin_half_alpha_range_l668_668298


namespace solve_trig_equation_l668_668208

noncomputable def smallest_positive_angle : ℝ :=
  (1/4) * Real.arcsin(2/9)

theorem solve_trig_equation :
  ∃ x : ℝ, 0 < x ∧ x = smallest_positive_angle ∧ 9 * Real.sin x * Real.cos x ^ 7 - 9 * Real.sin x ^ 7 * Real.cos x = 1 :=
by
  sorry

end solve_trig_equation_l668_668208


namespace parabola_directrix_l668_668442

theorem parabola_directrix (y x : ℝ) (h : y^2 = -4 * x) : x = 1 :=
sorry

end parabola_directrix_l668_668442


namespace highest_power_of_3_dividing_N_l668_668069

noncomputable def N : ℕ := 737271 /* Concatenate appropriately */ 3031

theorem highest_power_of_3_dividing_N :
  ∀ k : ℕ, 3^k ∣ N → k = 0 :=
by sorry

end highest_power_of_3_dividing_N_l668_668069


namespace dwarfs_truthful_count_l668_668590

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l668_668590


namespace net_marble_change_l668_668375

/-- Josh's initial number of marbles. -/
def initial_marbles : ℕ := 20

/-- Number of marbles Josh lost. -/
def lost_marbles : ℕ := 16

/-- Number of marbles Josh found. -/
def found_marbles : ℕ := 8

/-- Number of marbles Josh traded away. -/
def traded_away_marbles : ℕ := 5

/-- Number of marbles Josh received in a trade. -/
def received_in_trade_marbles : ℕ := 9

/-- Number of marbles Josh gave away. -/
def gave_away_marbles : ℕ := 3

/-- Number of marbles Josh received from his cousin. -/
def received_from_cousin_marbles : ℕ := 4

/-- Final number of marbles Josh has after all transactions. -/
def final_marbles : ℕ :=
  initial_marbles - lost_marbles + found_marbles - traded_away_marbles + received_in_trade_marbles
  - gave_away_marbles + received_from_cousin_marbles

theorem net_marble_change : (final_marbles : ℤ) - (initial_marbles : ℤ) = -3 := 
by
  sorry

end net_marble_change_l668_668375


namespace suresh_work_hours_l668_668434

theorem suresh_work_hours (x : ℝ) (h : x / 15 + 8 / 20 = 1) : x = 9 :=
by 
    sorry

end suresh_work_hours_l668_668434


namespace min_value_of_f_l668_668315

def f (x y z : ℝ) : ℝ := x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

theorem min_value_of_f (a b c x y z : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z) 
  (hcybz : c * y + b * z = a) 
  (hazcx : a * z + c * x = b) 
  (hbxay : b * x + a * y = c) : 
  f x y z = 1 / 2 :=
by 
  sorry

end min_value_of_f_l668_668315


namespace base_b_correct_l668_668249

theorem base_b_correct : ∃ b : ℕ, 7 < b ∧ 8 < b ∧ b = 9 ∧ b > 8 :=
by
  use 9
  split; sorry

end base_b_correct_l668_668249


namespace subtraction_example_l668_668203

theorem subtraction_example : -1 - 3 = -4 := 
  sorry

end subtraction_example_l668_668203


namespace find_A_satisfy_3A_multiple_of_8_l668_668464

theorem find_A_satisfy_3A_multiple_of_8 (A : ℕ) (h : 0 ≤ A ∧ A < 10) : 8 ∣ (30 + A) ↔ A = 2 := 
by
  sorry

end find_A_satisfy_3A_multiple_of_8_l668_668464


namespace expectation_comparison_l668_668436

-- Definitions for passing probabilities of events for athletes A and B
def p1 : ℝ := 4 / 5
def p2_A : ℝ := 3 / 4
def p3_A : ℝ := 2 / 3
def p2_B : ℝ := 5 / 8
def p3_B : ℝ := 3 / 4

-- Calculation of passing probabilities
def P_A1 : ℝ := p1 * p1
def P_A2 : ℝ := p2_A * p2_A
def P_A3 : ℝ := (3.choose 2) * (p3_A ^ 2) * (1 - p3_A) + (p3_A ^ 3)

def P_B1 : ℝ := p1 * p1
def P_B2 : ℝ := p2_B * p2_B
def P_B3 : ℝ := (3.choose 2) * (p3_B ^ 2) * (1 - p3_B) + (p3_B ^ 3)

-- Expected values
def E_A (p1 p2 p3 : ℝ) : ℝ := (16 / 25) * p1 + (9 / 25) * p2 + (4 / 15) * p3
def E_B (p1 p2 p3 : ℝ) : ℝ := (16 / 25) * p1 + (1 / 4) * p2 + (27 / 128) * p3

-- Proof statement
theorem expectation_comparison (p1 p2 p3 : ℝ) (hp1 : p1 > 0) (hp2 : p2 > 0) (hp3 : p3 > 0) :
  E_A p1 p2 p3 > E_B p1 p2 p3 :=
by sorry

end expectation_comparison_l668_668436


namespace PC_eq_QC_l668_668474

variables {A B C K L M N Q P : Point}

-- Assuming definitions for a Point and a Circle exist and are imported above.
-- Let's define the conditions given in the problem.

-- Definition of 'midpoint'
def is_midpoint (C A B : Point) : Prop :=
  sorry

-- Definition of 'chord passing through a point'
def chord_passes_through (C : Point) (Chord : Set Point) : Prop :=
  sorry

-- Conditions
axiom midpoint_C : is_midpoint C A B
axiom K_in_same_side_as_M : /*Definition that K and M are on the same side of AB*/
axiom KL_passes_through_C : chord_passes_through C {K, L}
axiom MN_passes_through_C : chord_passes_through C {M, N}
axiom Q_is_intersection_of_KN_with_AB : /*Definition that Q is the intersection of KN with AB*/
axiom P_is_intersection_of_ML_with_AB : /*Definition that P is the intersection of ML with AB*/

-- Statement to be proven
theorem PC_eq_QC : dist P C = dist Q C :=
by sorry

end PC_eq_QC_l668_668474


namespace impossible_segment_in_pentagon_l668_668751

theorem impossible_segment_in_pentagon :
  ∀ (P : Type) [EuclideanSpace P] [RegularPentagon P],
    ¬ ∃ (XY : Segment P), is_inside_pentagon XY ∧ 
      (∀ (v : Vertex P), ∃ (α : Angle), is_seen_from v α XY) :=
begin
  sorry
end

end impossible_segment_in_pentagon_l668_668751


namespace isosceles_triangles_with_perimeter_27_l668_668704

def is_valid_isosceles_triangle (a b : ℕ) : Prop :=
  2 * a + b = 27 ∧ 2 * a > b

theorem isosceles_triangles_with_perimeter_27 :
  { t : ℕ × ℕ // is_valid_isosceles_triangle t.1 t.2 }.card = 7 := 
sorry

end isosceles_triangles_with_perimeter_27_l668_668704


namespace maximize_S_l668_668647

noncomputable def S (x : ℝ) : ℝ :=
  x * (180 - (3 / x) + (400 / x^2))

theorem maximize_S :
  let x := 40
  let y := 45
  xy = 1800 → S x = x * (180 - (3 / x) + (400 / x^2)) → 
  S x ≤ S 40 :=
by
  simp
  sorry

end maximize_S_l668_668647


namespace vasya_days_l668_668910

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668910


namespace min_k_for_very_good_list_l668_668970

-- Definitions based on the problem statement
def is_good (l : List ℕ) : Prop := l.maximum?.get_or_else 0 = l.maximum?.map (λ x => (l.filter (λ y => y = x)).length).get_or_else 0

def is_very_good (l : List ℕ) : Prop := ∀ (s : List ℕ), s ≠ [] → s ⊆ l → (is_good s)

-- The question we want to prove
theorem min_k_for_very_good_list (n : ℕ) (k : ℕ) (l : List ℕ) :
  (length l = 2019) → (∀ x ∈ l, x ≤ k) →
  (is_very_good l) → k = 11 :=
sorry

end min_k_for_very_good_list_l668_668970


namespace circle_radius_is_sqrt_34_l668_668565

-- Circle problem definitions
variables {C₁ C₂ : Type} [MetricSpace C₁] [MetricSpace C₂]

-- Given conditions
variables (O X Y Z : C₁)
variables (XZ OZ YZ : ℝ)
variables (r : ℝ)  -- radius of C₁

-- Assume the conditions provided in the problem
def conditions : Prop :=
  (XZ = 15) ∧
  (OZ = 17) ∧
  (YZ = 8)

-- Problem statement
theorem circle_radius_is_sqrt_34
  (h : conditions XZ OZ YZ r) :
  r = Real.sqrt 34 := by
  sorry

end circle_radius_is_sqrt_34_l668_668565


namespace ratio_concentrate_to_water_l668_668189

theorem ratio_concentrate_to_water (c w : ℕ) (r_servings : ℕ) (oz_concentrate_per_can : ℕ) (oz_serving : ℕ) 
    (mix_ratio : ℕ) (gcd_cw : ℕ) 
    (h1 : mix_ratio = 3) 
    (h2 : r_servings = 320) 
    (h3 : oz_concentrate_per_can = 12) 
    (h4 : oz_serving = 6) 
    (h5 : c = 40) 
    (h6 : w = c * mix_ratio) 
    (h7 : gcd gcd_cw c w = 40) : 
    (c / gcd_cw) : (w / gcd_cw) = 1 : 3 :=
by
  skip_proof_using sorry -- proof is not actually required per instructions

end ratio_concentrate_to_water_l668_668189


namespace vasya_days_without_purchase_l668_668920

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668920


namespace min_sum_of_product_2004_l668_668455

theorem min_sum_of_product_2004 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
    (hxyz : x * y * z = 2004) : x + y + z ≥ 174 ∧ ∃ (a b c : ℕ), a * b * c = 2004 ∧ a + b + c = 174 :=
by sorry

end min_sum_of_product_2004_l668_668455


namespace fraction_girls_at_event_l668_668100

theorem fraction_girls_at_event :
  let mapl_students := 320
  let mapl_boys := (3 * (mapl_students / 8))
  let mapl_girls := (5 * (mapl_students / 8))
  let brook_students := 240
  let brook_boys := (5 * (brook_students / 8))
  let brook_girls := (3 * (brook_students / 8))
  let pine_students := 400
  let pine_boys := (pine_students / 2)
  let pine_girls := (pine_students / 2)
  let total_girls := mapl_girls + brook_girls + pine_girls
  let total_students := mapl_students + brook_students + pine_students
  in total_girls / total_students = (35 / 69) := sorry

end fraction_girls_at_event_l668_668100


namespace largest_prime_factor_of_3913_l668_668491

def is_prime (n : ℕ) := nat.prime n

def prime_factors_3913 := {17, 2, 5, 23}

theorem largest_prime_factor_of_3913 
  (h1 : is_prime 17)
  (h2 : is_prime 2)
  (h3 : is_prime 5)
  (h4 : is_prime 23)
  (h5 : 3913 = 17 * 2 * 5 * 23) : 
  (23 ∈ prime_factors_3913 ∧ ∀ x ∈ prime_factors_3913, x ≤ 23) :=
  by 
    sorry

end largest_prime_factor_of_3913_l668_668491


namespace problem_1_monotonicity_extreme_values_problem_2_unique_intersection_problem_3_geometric_progression_l668_668265

noncomputable def f(x : ℝ) := x / Real.exp x
noncomputable def g(x : ℝ) := Real.log x / x

theorem problem_1_monotonicity_extreme_values :
  (∀ x < 1, 0 < deriv f x) ∧
  (∀ x > 1, deriv f x < 0) ∧
  (f 1 = 1 / Real.exp 1) ∧
  (∀ x < exp 1, 0 < deriv g x) ∧
  (∀ x > exp 1, deriv g x < 0) ∧
  (g (exp 1) = 1 / exp 1) :=
by sorry
  
theorem problem_2_unique_intersection :
  ∃! x, f x = g x :=
by sorry

theorem problem_3_geometric_progression (a : ℝ) (h : 0 < a ∧ a < 1 / Real.exp 1)
  (x1 x2 x3 : ℝ) (hx1_lt : x1 < x2) (hx2_lt : x2 < x3)
  (h_intersections : f x1 = a ∧ f x2 = a ∧ g x3 = a) :
  x1 * x3 = x2^2 :=
by sorry

end problem_1_monotonicity_extreme_values_problem_2_unique_intersection_problem_3_geometric_progression_l668_668265


namespace ratio_water_to_orange_juice_l668_668035

variable (O W : ℝ)

-- Conditions:
-- 1. Amount of orange juice is O for both days.
-- 2. Amount of water is W on the first day and 2W on the second day.
-- 3. Price per glass is $0.60 on the first day and $0.40 on the second day.

theorem ratio_water_to_orange_juice 
  (h : (O + W) * 0.60 = (O + 2 * W) * 0.40) : 
  W / O = 1 := 
by 
  -- The proof is skipped
  sorry

end ratio_water_to_orange_juice_l668_668035


namespace find_distance_MA_l668_668314

noncomputable def point := (ℝ × ℝ)

def M : point := (-3, -1)
def A : point := (1, 1)

def distance (p1 p2 : point) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem find_distance_MA : distance M A = 2 * real.sqrt 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_MA_l668_668314


namespace vasya_did_not_buy_anything_days_l668_668938

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668938


namespace tangent_line_equation_at_1_1_l668_668082

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_equation_at_1_1 :
  tangent_line f (1,1) = (λ x y : ℝ, x + y - 2 = 0) :=
sorry

end tangent_line_equation_at_1_1_l668_668082


namespace density_transformation_l668_668693

variable {X Y : Type} [MeasurableSpace X] [MeasurableSpace Y]

variables (f : ℝ → ℝ) (a b : ℝ)

theorem density_transformation (f_density : ∀ x, a < x ∧ x < b → 0 ≤ f x)
  (Y_eq_3X : ∀ (x : ℝ), a < x ∧ x < b → Y = 3 * x) :
  (∀ y, 3 * a < y ∧ y < 3 * b → 0 ≤ (1 / 3) * f (y / 3)) :=
by
  sorry

end density_transformation_l668_668693


namespace total_amount_invested_l668_668344

-- Define the conditions and specify the correct answer
theorem total_amount_invested (x y : ℝ) (h8 : y = 600) 
  (h_income_diff : 0.10 * (x - 600) - 0.08 * 600 = 92) : 
  x + y = 2000 := sorry

end total_amount_invested_l668_668344


namespace sum_of_squares_of_geometric_sequence_l668_668361

theorem sum_of_squares_of_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : ∀ n, (∑ i in Finset.range n, a i) = 2^n - 1) :
  (∑ i in Finset.range n, (a i)^2) = (1 / 3) * (4^n - 1) := 
sorry

end sum_of_squares_of_geometric_sequence_l668_668361


namespace triangle_is_right_angled_l668_668719

theorem triangle_is_right_angled
  (a b c : ℝ)
  (A B C : ℝ)
  (cos_A : ℝ)
  (triangle_condition : cos_A = b / c)
  (cosine_rule_A : cos_A = (b^2 + c^2 - a^2) / (2 * b * c)) :
  c^2 = a^2 + b^2 :=
begin
  sorry
end

end triangle_is_right_angled_l668_668719


namespace cannot_cover_with_L_shape_l668_668531

/-! 
# Mathematical problem
Given an 8x8 grid with one corner 2x2 square removed,
prove that it is impossible to cover the remaining 60 squares
using exactly 15 L-shaped pieces, each shaped like a small "L".
-/

-- Definition of an 8x8 grid
def grid8x8 := fin 8 × fin 8

-- Definition of a 2x2 square
def square2x2 := fin 2 × fin 2

-- Function to generate the checkerboard pattern value at a given position
def checkerboard_value (i j : nat) : int :=
  if (i + j) % 2 == 0 then 1 else -1

-- Definition of "L-shaped" piece (one example of L-shape)
structure L_shape := 
  (p1 p2 p3 p4 : fin 8 × fin 8)
  (cond : p1 = (p2.1, p2.2.next) ∧ p2 = (p3.1, p3.2.next) ∧ p4 = (p1.1.next, p1.2))

-- Lean statement of the problem
theorem cannot_cover_with_L_shape :
  ¬ (∃ (S : set (L_shape)), S.card = 15 ∧
      ∀ l ∈ S, (∀ p ∈ [l.p1, l.p2, l.p3, l.p4], p ∉ square2x2) ∧ 
      ∀ p ≠ l ∈ S, disjoint [l.p1, l.p2, l.p3, l.p4] p ∧ 
      (∃ x ∈ grid8x8, ∀ p ∈ S, p ∈ [l.p1, l.p2, l.p3, l.p4])) :=
by
  sorry

end cannot_cover_with_L_shape_l668_668531


namespace number_of_truthful_gnomes_l668_668607

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l668_668607


namespace no_solution_to_system_l668_668239

open Real

theorem no_solution_to_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^(1/3) - y^(1/3) - z^(1/3) = 64) ∧ (x^(1/4) - y^(1/4) - z^(1/4) = 32) ∧ (x^(1/6) - y^(1/6) - z^(1/6) = 8) → False := by
  sorry

end no_solution_to_system_l668_668239


namespace negations_true_of_BD_l668_668200

def is_parallelogram_rhombus : Prop :=
  ∃ p : parallelogram, is_rhombus p

def exists_x_in_R : Prop :=
  ∃ x : ℝ, x^2 - 3 * x + 3 < 0

def forall_x_in_R : Prop :=
  ∀ x : ℝ, |x| + x^2 ≥ 0

def quad_eq_has_real_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - a * x + 1 = 0) → real_roots x

theorem negations_true_of_BD :
  (¬exists_x_in_R) ∧ (¬quad_eq_has_real_solutions a) :=
sorry

end negations_true_of_BD_l668_668200


namespace ellipse_equation_triangle_area_range_l668_668658

-- Condition statements

def ellipse (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def parabola (x y : ℝ) : Prop :=
  y^2 = x

def focal_length (c : ℝ) : Prop :=
  c = 2

def foci_positions (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (2, 0) ∧ F2 = (-2, 0)

def passes_through (C1 C2 : ℝ × ℝ → Prop) (F2 : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P ∈ C1 ∧ P ∈ C2 ∧ F2 ∈ C1

def line_no_intersect_parabola (l : ℝ → ℝ) (C2 : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C2 x y → l y ≠ x

-- Question (I)

theorem ellipse_equation (a b : ℝ) (x y : ℝ) :
  (focal_length 2) ∧ (foci_positions (2, 0) (-2, 0)) ∧
  (passes_through (ellipse a b) parabola (-2, 0)) →
  ∃ a b, a = 2 * sqrt 2 ∧ b = 2 ∧ (x^2 / (2 * sqrt 2)^2 + y^2 / 4 = 1) :=
sorry

-- Question (II)

theorem triangle_area_range (t x1 x2 y1 y2 a b : ℝ) :
  (t^2 < 8) ∧ (a = 2 * sqrt 2) ∧ (b = 2) ∧
  (line_no_intersect_parabola (λ y, t * y - 2) parabola) →
  let AB := ∀ y1 y2, sqrt (1 + t^2) * abs (y1 - y2) in
  let d := 4 / sqrt (t^2 + 1) in
  abs ( (1 / 2) * AB * d ) ∈ set.Ioo (8 * sqrt 2 / 5) (4 * sqrt 2) :=
sorry

end ellipse_equation_triangle_area_range_l668_668658


namespace solution_set_inequality_l668_668680

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x

axiom mono_increasing (x y : ℝ) (hxy : 0 < x ∧ x < y) : f x < f y

axiom f_2_eq_0 : f 2 = 0

theorem solution_set_inequality :
  { x : ℝ | (x - 1) * f x < 0 } = { x : ℝ | -2 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_inequality_l668_668680


namespace trigonometric_identity_l668_668288

variable {α : Real}
variable (h : Real.cos α = -2 / 3)

theorem trigonometric_identity : 
  (Real.cos α = -2 / 3) → 
  (Real.cos (4 * Real.pi - α) * Real.sin (-α) / 
  (Real.sin (Real.pi / 2 + α) * Real.tan (Real.pi - α)) = Real.cos α) :=
by
  intro h
  sorry

end trigonometric_identity_l668_668288


namespace linear_regression_slope_l668_668651

theorem linear_regression_slope {x y : ℝ} (h : y = 2 - x) :
  ∀ x', (y = 2 - x') → (y - (2 - (x' + 1))) = 1 :=
by
  intro x' h'
  have : y - (2 - (x' + 1)) = y - (2 - x' - 1) := by sorry
  rw h at this
  rw h' at this
  exact this

end linear_regression_slope_l668_668651


namespace transaction_result_l668_668160

-- Definitions derived from conditions in a)
def SellingPriceCar : ℝ := 20000
def SellingPriceMotorcycle : ℝ := 10000
def CostPriceCar : ℝ := (4 / 3) * SellingPriceCar
def CostPriceMotorcycle : ℝ := (4 / 5) * SellingPriceMotorcycle
def TotalCost : ℝ := CostPriceCar + CostPriceMotorcycle
def TotalSellingPrice : ℝ := SellingPriceCar + SellingPriceMotorcycle

-- Lean statement to be proved: TotalCost is \$4667 more than TotalSellingPrice
theorem transaction_result :
  TotalCost - TotalSellingPrice = 4667 := by
  sorry

end transaction_result_l668_668160


namespace sum_of_digits_of_products_of_76eights_and_76fives_l668_668206

theorem sum_of_digits_of_products_of_76eights_and_76fives :
  let n1 := (List.repeat 8 76).foldl (λ acc d, acc * 10 + d) 0
  let n2 := (List.repeat 5 76).foldl (λ acc d, acc * 10 + d) 0
  (Nat.digits 10 (n1 * n2)).sum = 304 :=
by
  sorry

end sum_of_digits_of_products_of_76eights_and_76fives_l668_668206


namespace degree_measure_angle_EFB_l668_668360

-- Definitions based on conditions:
def is_square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  ∠ABC = 90 ∧ ∠BCD = 90 ∧ ∠CDA = 90 ∧ ∠DAB = 90

def same_half_plane (A B E : Point) : Prop :=
  ∠ABE = 150

def lies_on_segment (F : Point) (B C : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (1 - t) • B + t • C

def is_isosceles_triangle (B E F : Point) : Prop :=
  dist B E = dist B F

-- Lean statement of the mathematically equivalent proof problem:
theorem degree_measure_angle_EFB
  (A B C D E F : Point)
  (hsquare : is_square A B C D)
  (hhalfplane : same_half_plane A B E)
  (hlies_on : lies_on_segment F B C)
  (hisosceles : is_isosceles_triangle B E F) :
  ∠EFB = 60 := sorry

end degree_measure_angle_EFB_l668_668360


namespace sum_of_consecutive_odds_l668_668837

theorem sum_of_consecutive_odds (a : ℤ) (h : (a - 2) * a * (a + 2) = 9177) : (a - 2) + a + (a + 2) = 63 := 
sorry

end sum_of_consecutive_odds_l668_668837


namespace vector_subtraction_magnitude_l668_668322

noncomputable def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def vector_dot (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem vector_subtraction_magnitude (a b : ℝ × ℝ × ℝ) (ha : vector_norm a = 2) (hb : vector_norm b = 3) (θ : real.angle) (hθ : θ = real.pi / 3) :
  vector_norm (a - b) = real.sqrt 7 :=
by
  sorry

end vector_subtraction_magnitude_l668_668322


namespace proof_problem_l668_668198

-- Proposition B: ∃ x ∈ ℝ, x^2 - 3*x + 3 < 0
def propB : Prop := ∃ x : ℝ, x^2 - 3 * x + 3 < 0

-- Proposition D: ∀ x ∈ ℝ, x^2 - a*x + 1 = 0 has real solutions
def propD (a : ℝ) : Prop := ∀ x : ℝ, ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0

-- Negation of Proposition B: ∀ x ∈ ℝ, x^2 - 3 * x + 3 ≥ 0
def neg_propB : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 ≥ 0

-- Negation of Proposition D: ∃ a ∈ ℝ, ∃ x ∈ ℝ, ∄ (x1 x2 : ℝ), x^2 - a * x + 1 = 0
def neg_propD : Prop := ∃ a : ℝ, ∀ x : ℝ, ¬ ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0 

-- The main theorem combining the results based on the solutions.
theorem proof_problem : neg_propB ∧ neg_propD :=
by
  sorry

end proof_problem_l668_668198


namespace election_votes_l668_668470

theorem election_votes
  (V : ℕ)  -- Total number of votes
  (winner_votes_share : ℝ)  -- Share of winner's votes
  (vote_margin : ℕ)  -- Margin by which the winner won
  (h1 : winner_votes_share = 0.62)  -- Winner received 62% of votes
  (h2 : winner_votes_share * V - (1 - winner_votes_share) * V = vote_margin)  -- Winner won by 408 votes
  (h3 : vote_margin = 408)  -- Margin of votes
  : winner_votes_share * V = 1054 :=
by
  -- The proof would go here
  sorry

end election_votes_l668_668470


namespace P_and_Q_are_real_l668_668561

-- Definitions for polynomials and real/complex coefficients
variables {R : Type*} [field R] (P Q : polynomial R)

-- Conditions from a)
axiom is_real_polynomial : polynomial R → Prop
axiom is_complex_polynomial : polynomial R → Prop
axiom composition_is_real : is_real_polynomial (P.comp Q)
axiom leading_coeff_real : (Q.leading_coeff : R) ∈ ℝ
axiom constant_term_real : (Q.coeff 0 : R) ∈ ℝ

-- Statement to prove in c)
theorem P_and_Q_are_real :
  is_real_polynomial P ∧ is_real_polynomial Q :=
sorry

end P_and_Q_are_real_l668_668561


namespace curve_constant_width_l668_668786

-- Defining the conditions of the problem in Lean
def curve (C : Type) (points : set ℝ) (h : ℝ) : Prop :=
  ∀ (L : ℝ → Prop), (∃! (P : ℝ), P ∈ points ∧ L P)

def supporting_lines (L : ℝ → Prop) (C : Type) (points : set ℝ) : Prop :=
  ∃! (P : ℝ), P ∈ points ∧ L P

def parallel_supporting_lines (L1 L2 : ℝ → Prop) (C : Type) (points : set ℝ) (h : ℝ) : Prop :=
  supporting_lines L1 C points ∧ supporting_lines L2 C points ∧
  ∀ (A B : ℝ), A ∈ points ∧ B ∈ points ∧ L1 A ∧ L2 B → dist A B = h

def perpendicular_chord (L1 L2 : ℝ → Prop) (A B : ℝ) (h : ℝ) : Prop :=
  L1 A ∧ L2 B ∧ dist A B = h ∧ abs (A - B) = h

-- Lean theorem to be proved
theorem curve_constant_width (C : Type) (points : set ℝ) (h : ℝ) :
  (∀ (L : ℝ → Prop), supporting_lines L C points) →
  (∀ (L1 L2 : ℝ → Prop) (A B : ℝ),
    parallel_supporting_lines L1 L2 C points h →
    (∃ (A B : ℝ), L1 A ∧ L2 B ∧ perpendicular_chord L1 L2 A B h)) :=
sorry

end curve_constant_width_l668_668786


namespace number_of_truthful_dwarfs_l668_668581

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l668_668581


namespace proof_g_neg1_l668_668677

noncomputable def f : ℝ → ℝ := sorry -- Assume existence due to conditions
noncomputable def g (x : ℝ) := f(x) + 2
def y (x : ℝ) := f(x) + x^2 + x

theorem proof_g_neg1 :
  (∀ x : ℝ, y(-x) = -y(x)) ∧ (f 1 = 1) ∧ (g (-1) = f (-1) + 2) → g (-1) = -1 :=
sorry

end proof_g_neg1_l668_668677


namespace num_subsets_of_M_l668_668091

-- Define the set M using the given condition x^2 - 2x + 1 = 0
def M : Set ℝ := { x | x^2 - 2 * x + 1 = 0 }

-- State the theorem that proves the number of subsets of M is equal to 2
theorem num_subsets_of_M : (M.to_finset.powerset.card = 2) :=
by
  sorry

end num_subsets_of_M_l668_668091


namespace sunzi_carriage_l668_668149

theorem sunzi_carriage (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  ((Three people share a carriage, leaving two carriages empty) ∧ (Two people share a carriage, leaving nine people walking)) := sorry

end sunzi_carriage_l668_668149


namespace al_original_portion_l668_668550

theorem al_original_portion (a b c : ℝ) (h1 : a + b + c = 1200) (h2 : 0.75 * a + 2 * b + 2 * c = 1800) : a = 480 :=
by
  sorry

end al_original_portion_l668_668550


namespace vasya_did_not_buy_anything_days_l668_668936

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668936


namespace number_of_truthful_dwarfs_l668_668583

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l668_668583


namespace factorize_expression_find_xy_l668_668103

-- Problem 1: Factorizing the quadratic expression
theorem factorize_expression (x : ℝ) : 
  x^2 - 120 * x + 3456 = (x - 48) * (x - 72) :=
sorry

-- Problem 2: Finding the product xy from the given equation
theorem find_xy (x y : ℝ) (h : x^2 + y^2 + 8 * x - 12 * y + 52 = 0) : 
  x * y = -24 :=
sorry

end factorize_expression_find_xy_l668_668103


namespace eccentricity_of_hyperbola_l668_668027

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) : ℝ :=
  (1 + b^2 / a^2) ^ (1/2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 h3 = Real.sqrt 3 := 
by
  unfold hyperbola_eccentricity
  rw [h3]
  simp
  sorry

end eccentricity_of_hyperbola_l668_668027


namespace book_pages_l668_668007

theorem book_pages (books sheets pages_per_sheet pages_per_book : ℕ)
  (hbooks : books = 2)
  (hpages_per_sheet : pages_per_sheet = 8)
  (hsheets : sheets = 150)
  (htotal_pages : pages_per_sheet * sheets = 1200)
  (hpages_per_book : pages_per_book = 1200 / books) :
  pages_per_book = 600 :=
by
  -- Proof goes here
  sorry

end book_pages_l668_668007


namespace xiao_wang_original_plan_l668_668510

theorem xiao_wang_original_plan (p d1 extra_pages : ℕ) (original_days : ℝ) (x : ℝ) 
  (h1 : p = 200)
  (h2 : d1 = 5)
  (h3 : extra_pages = 5)
  (h4 : original_days = p / x)
  (h5 : original_days - 1 = d1 + (p - (d1 * x)) / (x + extra_pages)) :
  x = 20 := 
  sorry

end xiao_wang_original_plan_l668_668510


namespace overall_gain_percentage_l668_668172

def cost_price_A : ℝ := 4000
def cost_price_B : ℝ := 6000
def cost_price_C : ℝ := 8000

def selling_price_A : ℝ := 4500
def selling_price_B : ℝ := 6500
def selling_price_C : ℝ := 8500

def total_cost_price : ℝ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℝ := selling_price_A + selling_price_B + selling_price_C

def total_gain : ℝ := total_selling_price - total_cost_price

def gain_percentage : ℝ := (total_gain / total_cost_price) * 100

theorem overall_gain_percentage :
  gain_percentage = 8.33 := by
  sorry

end overall_gain_percentage_l668_668172


namespace vasya_purchase_l668_668933

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668933


namespace weight_shaina_receives_l668_668012

namespace ChocolateProblem

-- Definitions based on conditions
def total_chocolate : ℚ := 60 / 7
def piles : ℚ := 5
def weight_per_pile : ℚ := total_chocolate / piles
def shaina_piles : ℚ := 2

-- Proposition to represent the question and correct answer
theorem weight_shaina_receives : 
  (weight_per_pile * shaina_piles) = 24 / 7 := 
by
  sorry

end ChocolateProblem

end weight_shaina_receives_l668_668012


namespace sum_reciprocal_sums_l668_668317

def a_n (n : ℕ) : ℕ := n

def S_n (n : ℕ) : ℚ := n * (n + 1) / 2

theorem sum_reciprocal_sums :
  (∑ k in Finset.range 100, 1 / S_n (k + 1)) = 200 / 101 := by
  sorry

end sum_reciprocal_sums_l668_668317


namespace number_of_truthful_dwarfs_l668_668596

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l668_668596


namespace vasya_days_without_purchase_l668_668923

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668923


namespace LykaSavings_l668_668410

-- Define the given values and the properties
def totalCost : ℝ := 160
def amountWithLyka : ℝ := 40
def averageWeeksPerMonth : ℝ := 4.33
def numberOfMonths : ℝ := 2

-- Define the remaining amount Lyka needs
def remainingAmount : ℝ := totalCost - amountWithLyka

-- Define the number of weeks in the saving period
def numberOfWeeks : ℝ := numberOfMonths * averageWeeksPerMonth

-- Define the weekly saving amount
def weeklySaving : ℝ := remainingAmount / numberOfWeeks

-- State the theorem to be proved
theorem LykaSavings :  weeklySaving ≈ 13.86 :=
by
  -- Proof steps (omitted)
  sorry

end LykaSavings_l668_668410


namespace problem_part_I_problem_part_II_l668_668309

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt(a * x^2 - 2 * a * x + 1)

theorem problem_part_I (a : ℝ) : 0 ≤ a → a ≤ 1 :=
by
  sorry

theorem problem_part_II (m : ℝ) :
  ∃ (a : ℝ) (h : a = 1),
  ∀ (x1 : ℝ) (h1 : -1 ≤ x1 ∧ x1 ≤ 1),
  ∃ (x2 : ℝ),
  9^x1 + 9^(-x1) + m * (3^x1 - 3^(-x1)) - 1 ≥ f 1 x2 →
  -2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end problem_part_I_problem_part_II_l668_668309


namespace g_value_range_l668_668395

noncomputable def g (x y z : ℝ) : ℝ :=
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_value_range (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  (3/2 : ℝ) ≤ g x y z ∧ g x y z ≤ (3 : ℝ) / 2 := 
sorry

end g_value_range_l668_668395


namespace find_f_2007_l668_668444

/-- Given a function f such that for all positive x and y, f(x * y) = f(x) + f(y), and given
f(1) = 1, prove that f(2007) = -1. -/

constant f : ℝ → ℝ

axiom f_property : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_initial_condition : f 1 = 1

theorem find_f_2007 : f 2007 = -1 :=
sorry

end find_f_2007_l668_668444


namespace first_exceed_500_on_tuesday_l668_668377

def deposits (n : ℕ) : ℕ :=
  if n = 0 then 3
  else if n = 1 then 3
  else if n % 2 = 0 then deposits (n - 2) * 3
  else deposits (n - 1)

def cumulative_sum (n : ℕ) : ℕ :=
  (list.range (n + 1)).sum deposits

theorem first_exceed_500_on_tuesday :
  ∃ n, cumulative_sum n > 500 ∧ n = 9 ∧ (n % 7 = 2) :=
by
  sorry

end first_exceed_500_on_tuesday_l668_668377


namespace inequality_not_true_l668_668687

noncomputable def f (x : ℝ) : ℝ := (1 / real.exp(1)) ^ x + real.log x

variables (a b c x0 : ℝ)
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variables (h_order : a < b) (h_order' : b < c)
variables (h_prod : f a * f b * f c > 0)
variables (h_root : f x0 = 0)

theorem inequality_not_true : ¬ (x0 > c) :=
sorry

end inequality_not_true_l668_668687


namespace PQCD_is_parallelogram_l668_668953

variable {α : ℝ} (hα : α > 1)

variables (A B C D P Q : ℂ)
variable (hABDC : AB ∥ DC)
variable (hRatio1 : AP / AC = (α - 1) / (α + 1))
variable (hRatio2 : BQ / BD = (α - 1) / (α + 1))

theorem PQCD_is_parallelogram
  (hABDC : AB ∥ DC)
  (hRatio1 : AP / AC = (α - 1) / (α + 1))
  (hRatio2 : BQ / BD = (α - 1) / (α + 1))
  : is_parallelogram PQ CD :=
sorry

end PQCD_is_parallelogram_l668_668953


namespace Carson_returned_l668_668009

theorem Carson_returned :
  ∀ (initial_oranges ate_oranges stolen_oranges final_oranges : ℕ), 
  initial_oranges = 60 →
  ate_oranges = 10 →
  stolen_oranges = (initial_oranges - ate_oranges) / 2 →
  final_oranges = 30 →
  final_oranges = (initial_oranges - ate_oranges - stolen_oranges) + 5 :=
by 
  sorry

end Carson_returned_l668_668009


namespace vasya_days_l668_668904

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668904


namespace symmetry_of_shifted_sine_function_l668_668798

noncomputable def shifted_sine_function (x : ℝ) : ℝ :=
  sin (x + π / 6 + π)

theorem symmetry_of_shifted_sine_function :
  ∀ x : ℝ, shifted_sine_function x = shifted_sine_function (π / 3 - x) :=
sorry

end symmetry_of_shifted_sine_function_l668_668798


namespace vasya_days_without_purchase_l668_668918

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668918


namespace x1_mul_x2_l668_668754

open Real

theorem x1_mul_x2 (x1 x2 : ℝ) (h1 : x1 + x2 = 2 * sqrt 1703) (h2 : abs (x1 - x2) = 90) : x1 * x2 = -322 := by
  sorry

end x1_mul_x2_l668_668754


namespace tan_cot_identity_l668_668523

variable {α β : ℝ}

def cot (x : ℝ) : ℝ := 1 / Real.tan x
def tan2alpha := Real.tan (2 * α)
def tan3beta := Real.tan (3 * β)

theorem tan_cot_identity (h : cot (2 * α) = 1 / tan2alpha ∧ cot (3 * β) = 1 / tan3beta) :
    (tan2alpha + cot (3 * β)) / (cot (2 * α) + tan3beta) = tan2alpha / tan3beta :=
by
  sorry

end tan_cot_identity_l668_668523


namespace limit_solution_l668_668560

noncomputable def limit_problem :=
  limit (λ x : ℝ, (6 - 5 / cos x) ^ (cot x)^2) 0 (𝓝 0) = exp (-5 / 2)

theorem limit_solution : limit_problem := 
  sorry

end limit_solution_l668_668560


namespace find_inverse_l668_668448

noncomputable def inverse_function (x : ℝ) (hx : 1/3 < x ∧ x ≤ 1) : ℝ :=
  - Real.sqrt (1 + Real.log x / Real.log 3)

theorem find_inverse :
  (∀ x : ℝ, -1 ≤ x ∧ x < 0 → (inverse_function (3^(x^2 - 1)) ⟨Real.rpow_pos_of_pos (by linarith) _, calc (3^(x^2 - 1)) < 1 : by sorry, _⟩ = x)) :=
by
  sorry

end find_inverse_l668_668448


namespace k_h_neg3_eq_15_l668_668393

-- Define the function h
def h (x : Int) : Int := 5 * x^2 - 12

-- Given: k(h(3)) = 15
axiom k_h3_eq_15 : k (h 3) = 15

-- Prove that k(h(-3)) = 15
theorem k_h_neg3_eq_15 : k (h (-3)) = 15 :=
by
  have h3 : h 3 = 33 := by rfl
  have h_neg3 : h (-3) = 33 := by rfl
  rw [h_neg3, k_h3_eq_15]
  sorry -- placeholder to indicate further steps are needed to complete the proof

end k_h_neg3_eq_15_l668_668393


namespace length_PJ_l668_668005

open Real

noncomputable def triangle_PQR (P Q R : Point) (PQ QR RP : ℝ) :=
  PQ = 17 ∧ QR = 15 ∧ RP = 8

noncomputable def incenter_J (P Q R J K L M : Point) (PQ QR RP : ℝ) :=
  triangle_PQR P Q R PQ QR RP ∧
  touches_incenter PQ QR RP J K L M

theorem length_PJ {P Q R J K L M : Point} (PQ QR RP : ℝ) :
  triangle_PQR P Q R PQ QR RP →
  touches_incenter PQ QR RP J K L M →
  length (P, J) = sqrt 34 :=
by
  sorry

end length_PJ_l668_668005


namespace coloring_probability_l668_668353

-- Definition of the problem and its conditions
def num_cells := 16
def num_diags := 2
def chosen_diags := 7

-- Define the probability
noncomputable def prob_coloring_correct : ℚ :=
  (num_diags ^ chosen_diags : ℚ) / (num_diags ^ num_cells)

-- The Lean theorem statement
theorem coloring_probability : prob_coloring_correct = 1 / 512 := 
by 
  unfold prob_coloring_correct
  -- The proof steps would follow here (omitted)
  sorry

end coloring_probability_l668_668353


namespace value_of_a7_l668_668289

noncomputable def arithmetic_sequence (f : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, f (n+1) - f n = d

variable {a : ℕ → ℝ}
variable (h_arith : arithmetic_sequence (λ n, 1 / a n))
variable (h_a1 : a 1 = 1)
variable (h_a4 : a 4 = 4)

theorem value_of_a7 : a 7 = -2 :=
by
  sorry

end value_of_a7_l668_668289


namespace vasya_days_without_purchase_l668_668924

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668924


namespace probability_sum_9_is_three_twentieths_l668_668484

def t : Finset ℕ := {2, 3, 4, 5}
def b : Finset ℕ := {4, 5, 6, 7, 8}

def numberOfPairsWithSum9 : ℕ := (t.product b).filter (λ pair => pair.1 + pair.2 = 9).card

def totalNumberOfPairs : ℕ := t.card * b.card

def probabilityOfSum9 (numPairs : ℕ) (totalPairs : ℕ) : ℚ :=
  numPairs / totalPairs

theorem probability_sum_9_is_three_twentieths :
  probabilityOfSum9 numberOfPairsWithSum9 totalNumberOfPairs = 3 / 20 :=
by
  sorry

end probability_sum_9_is_three_twentieths_l668_668484


namespace arc_length_independent_of_O_l668_668553

theorem arc_length_independent_of_O (A B C O D E F : Point)
  (h_isosceles : distance B A = distance B C)
  (h_parallel_line : Line.parallel (Line.mk A C) (Line.mk B O))
  (h_point_on_line : Point.on_line O (Line.mk B O))
  (h_circle_tangent : Circle.tangent (Circle.mk O A) (Line.mk A C) D)
  (h_circle_intersects_AB : Circle.intersects (Circle.mk O A) (Line.mk B A) E)
  (h_circle_intersects_BC : Circle.intersects (Circle.mk O A) (Line.mk B C) F) :
  arc_length (arc EDF) = arc_length (arc E'D'F') :=
sorry

end arc_length_independent_of_O_l668_668553


namespace integral_evaluation_l668_668895

noncomputable def indefinite_integral : ℝ → ℝ :=
  λ x, ↑∫ (a : ℝ), (a ^ 3 + 6 * a ^ 2 + 10 * a + 12)/(a - 2) * (a + 2) ^ 3 -- Terminated the RHS to avoid making this too complex for this snippet.

theorem integral_evaluation :
  ∫ (a : ℝ), (a ^ 3 + 6 * a ^ 2 + 10 * a + 12)/(a - 2) * (a + 2) ^ 3 = 
    λ x, ln (abs (x - 2)) + 1/(x+2)^2 + C :=
sorry

end integral_evaluation_l668_668895


namespace first_class_fraction_walk_l668_668110

-- Define the variables for speed
def walking_speed := 4
def bus_speed_loaded := 40
def bus_speed_empty := 50

-- Define variable for total distance A to B
variable (AB : ℝ)

-- Define the fractional distance walked by the first class
def fraction_walked_by_first_class (AC : ℝ) (CB : ℝ) : ℝ :=
  AC / AB

def distance_walked_by_first_class (AC : ℝ) : ℝ :=
  AC

theorem first_class_fraction_walk :
  ∃ AC CB : ℝ, (walking_speed = 4) ∧
                (bus_speed_loaded = 40) ∧
                (bus_speed_empty = 50) ∧
                (AC + CB = AB) ∧
                (CB = (1 / 6) * AB) ∧
                (fraction_walked_by_first_class AC CB = 1 / 7) :=
sorry

end first_class_fraction_walk_l668_668110


namespace jacks_speed_is_7_l668_668370

-- Define the constants and speeds as given in conditions
def initial_distance : ℝ := 150
def christina_speed : ℝ := 8
def lindy_speed : ℝ := 10
def lindy_total_distance : ℝ := 100

-- Hypothesis stating when the three meet
theorem jacks_speed_is_7 :
  ∃ (jack_speed : ℝ), (∃ (time : ℝ), 
    time = lindy_total_distance / lindy_speed
    ∧ christina_speed * time + jack_speed * time = initial_distance) 
  → jack_speed = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end jacks_speed_is_7_l668_668370


namespace retrievers_count_l668_668372

-- Definitions of given conditions
def huskies := 5
def pitbulls := 2
def retrievers := Nat
def husky_pups := 3
def pitbull_pups := 3
def retriever_extra_pups := 2
def total_pups_excess := 30

-- Equation derived from the problem conditions
def total_pups (G : Nat) := huskies * husky_pups + pitbulls * pitbull_pups + G * (husky_pups + retriever_extra_pups)
def total_adults (G : Nat) := huskies + pitbulls + G

theorem retrievers_count : ∃ G : Nat, G = 4 ∧ total_pups G = total_adults G + total_pups_excess :=
by
  sorry

end retrievers_count_l668_668372


namespace gcd_45736_123456_l668_668241

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 :=
by sorry

end gcd_45736_123456_l668_668241


namespace part1_part2_l668_668336

def constant_tangent_function (f : ℝ -> ℝ) :=
  ∀ k b : ℝ, ∀ x_0 : ℝ, f(x_0) + k * x_0 + b = k * x_0 + b ∧ 
                        ∀ x_0 : ℝ, has_deriv_at f x_0 (f' x_0 + k = k)

noncomputable def f1 : ℝ -> ℝ := fun x => x^3
noncomputable def f2 (m : ℝ) : ℝ -> ℝ := fun x => 0.5 * ((Real.exp x - x - 1) * Real.exp x) + m

theorem part1 : constant_tangent_function f1 :=
  sorry

theorem part2 (m : ℝ) : constant_tangent_function (f2 m) → -1/8 < m ∧ m ≤ 0 :=
  sorry

end part1_part2_l668_668336


namespace simplify_division_l668_668989

theorem simplify_division (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := 
by sorry

end simplify_division_l668_668989


namespace vasya_did_not_buy_anything_days_l668_668942

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668942


namespace cost_of_45_lilies_l668_668554

-- Defining the conditions
def price_per_lily (n : ℕ) : ℝ :=
  if n <= 30 then 2
  else 1.8

-- Stating the problem in Lean 4
theorem cost_of_45_lilies :
  price_per_lily 15 * 15 = 30 → (price_per_lily 45 * 45 = 81) :=
by
  intro h
  sorry

end cost_of_45_lilies_l668_668554


namespace digit_place_value_ratio_l668_668362

theorem digit_place_value_ratio :
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  digit_6_value / digit_5_value = 10000 :=
by
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  sorry

end digit_place_value_ratio_l668_668362


namespace num_marked_cells_at_least_num_cells_in_one_square_l668_668952

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end num_marked_cells_at_least_num_cells_in_one_square_l668_668952


namespace sum_fraction_inequality_l668_668756

theorem sum_fraction_inequality (M : ℕ) (hM : 0 < M) (f : Fin (M + 2) → Fin (M + 2)) 
  (hf : Function.Bijective f) :
  (∑ n in Finset.range M, 1 / (f n + f (n + 1))) > M / (M + 3) := 
sorry

end sum_fraction_inequality_l668_668756


namespace largest_central_rectangle_area_l668_668357

theorem largest_central_rectangle_area :
  let a b c d e f : ℝ in 
  (a + b + c = 23) ∧ 
  (d + e + f = 23) ∧
  (∃ (a d : ℝ), a * d = 13) ∧ 
  (∃ (b f : ℝ), b * f = 111) ∧ 
  (∃ (c e : ℝ), c * e = 37) ∧ 
  (∃ (a f : ℝ), a * f = 123) → 
  b * e ≤ 180 :=
begin
  sorry
end

end largest_central_rectangle_area_l668_668357


namespace parallelogram_area_l668_668514

theorem parallelogram_area (base height : ℕ) (hbase : base = 30) (hheight : height = 12) :
  base * height = 360 :=
by
  rw [hbase, hheight]
  norm_num

end parallelogram_area_l668_668514


namespace vasya_days_without_purchases_l668_668896

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668896


namespace range_of_a_l668_668668

theorem range_of_a (a : ℝ) : (∀ b : ℝ, ∀ x : ℝ, 3 * x^2 ≠ 3 * a - 1) ↔ a ∈ set.Iio (1 / 3) :=
by sorry

end range_of_a_l668_668668


namespace total_papers_delivered_l668_668038

-- Definitions based on given conditions
def papers_saturday : ℕ := 45
def papers_sunday : ℕ := 65
def total_papers := papers_saturday + papers_sunday

-- The statement we need to prove
theorem total_papers_delivered : total_papers = 110 := by
  -- Proof steps would go here
  sorry

end total_papers_delivered_l668_668038


namespace pow_neg_l668_668205

theorem pow_neg (x : ℝ) (n : ℤ) (h : x ≠ 0) : x ^ -n = 1 / (x ^ n) :=
by sorry

example : (2 : ℝ) ^ (-3 : ℤ) = 1 / (2 ^ 3) :=
by rw pow_neg 2 3 (by norm_num)

end pow_neg_l668_668205


namespace width_of_larger_cuboid_l668_668533

theorem width_of_larger_cuboid
    (length_larger : ℝ)
    (width_larger : ℝ)
    (height_larger : ℝ)
    (length_smaller : ℝ)
    (width_smaller : ℝ)
    (height_smaller : ℝ)
    (num_smaller : ℕ)
    (volume_larger : ℝ)
    (volume_smaller : ℝ)
    (divided_into : Real) :
    length_larger = 12 → height_larger = 10 →
    length_smaller = 5 → width_smaller = 3 → height_smaller = 2 →
    num_smaller = 56 →
    volume_smaller = length_smaller * width_smaller * height_smaller →
    volume_larger = num_smaller * volume_smaller →
    volume_larger = length_larger * width_larger * height_larger →
    divided_into = volume_larger / (length_larger * height_larger) →
    width_larger = 14 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end width_of_larger_cuboid_l668_668533


namespace range_of_m_l668_668696

noncomputable def vector_a (n : ℝ) (x : ℝ) : ℝ × ℝ := (n + 2, n - (Real.cos x)^2)
noncomputable def vector_b (m : ℝ) (x : ℝ) : ℝ × ℝ := (m, m / 2 + Real.sin x)

theorem range_of_m (m n x : ℝ) (h : vector_a n x = (2:ℝ) • vector_b m x) : 0 ≤ m ∧ m ≤ 4 :=
by {
  unfold vector_a vector_b at h,
  sorry
}

end range_of_m_l668_668696


namespace angle_BAM_eq_angle_CKM_l668_668829

variable {A B C K M : Type}
variable [EuclideanGeometry A B C K M]

-- Definitions corresponding to the conditions
def right_triangle_ABC (hABC : Type) : Prop :=
  ∃ (C : hABC), ∠C = 90° 

def midpoint (K : Type) (A B : Type) :=
  ∃ (K : A × B), K = (A + B) / 2

def point_on_segment_ratio (M : Type) (B C : Type) : Prop :=
  ∃ (x : ℝ), BM = 2 * MC

-- The theorem stating the question and the conditions
theorem angle_BAM_eq_angle_CKM
  (hABC : triangle A B C) (hC : ∠ C = 90°)
  (hK : midpoint K A B) (hM : point_on_segment_ratio M B C) :
  ∠BAM = ∠CKM :=
sorry

end angle_BAM_eq_angle_CKM_l668_668829


namespace fraction_value_l668_668852

-- Define the constants
def eight := 8
def four := 4

-- Statement to prove
theorem fraction_value : (eight + four) / (eight - four) = 3 := 
by
  sorry

end fraction_value_l668_668852


namespace simplify_expression1_simplify_expression2_l668_668053

-- Problem (1)
theorem simplify_expression1 :
  (0.008)^(1/3) + (sqrt 2 - real.pi)^0 - (125/64)^(-1/3) = 2/5 :=
sorry

-- Problem (2)
theorem simplify_expression2 :
  ( (real.log 2 / real.log 3 + real.log 2 / (real.log 9))
    * (real.log 3 / (real.log 4) + real.log 3 / (real.log 8)) )
  / (real.log 600 - (1/2) * real.log 0.036 - (1/2) * real.log 0.1) ≈ 0.2412 :=
sorry

end simplify_expression1_simplify_expression2_l668_668053


namespace p_investment_l668_668890

-- Define parameters for the investments and profit ratio
variables (Q P : ℝ)
-- Q is defined as 45,000
def condition1 := Q = 45000
-- Profit is divided in the ratio 2:3
def condition2 := P / Q = 2 / 3

-- The theorem to prove
theorem p_investment (h1 : condition1) (h2 : condition2) : P = 30000 :=
sorry

end p_investment_l668_668890


namespace suresh_and_wife_meet_time_l668_668186

noncomputable def suresh_speed_flat : ℝ := 4.5 * 1000 / 60  -- 75 m/min
noncomputable def suresh_speed_downhill : ℝ := suresh_speed_flat * 1.10  -- 82.5 m/min
noncomputable def suresh_speed_uphill : ℝ := suresh_speed_flat * 0.85  -- 63.75 m/min
noncomputable def suresh_average_speed : ℝ := (suresh_speed_downhill + suresh_speed_uphill) / 2  -- 73.125 m/min

noncomputable def wife_speed_flat : ℝ := 3.75 * 1000 / 60  -- 62.5 m/min
noncomputable def wife_speed_downhill : ℝ := wife_speed_flat * 1.07  -- 66.875 m/min
noncomputable def wife_speed_uphill : ℝ := wife_speed_flat * 0.90  -- 56.25 m/min
noncomputable def wife_average_speed : ℝ := (wife_speed_downhill + wife_speed_uphill) / 2  -- 61.5625 m/min

noncomputable def combined_speed : ℝ := suresh_average_speed + wife_average_speed  -- 134.6875 m/min
noncomputable def track_circumference : ℝ := 726  -- meters
noncomputable def time_to_meet : ℝ := track_circumference / combined_speed  -- 5.39 minutes

theorem suresh_and_wife_meet_time :
  time_to_meet ≈ 5.39 := 
sorry

end suresh_and_wife_meet_time_l668_668186


namespace sector_area_135_deg_20_cm_l668_668672

def area_of_sector (θ : ℝ) (r : ℝ) : ℝ := (θ * real.pi * r^2) / 360

theorem sector_area_135_deg_20_cm : 
  area_of_sector 135 20 = 150 * real.pi := 
by 
  sorry

end sector_area_135_deg_20_cm_l668_668672


namespace shortest_distance_between_circles_l668_668493

theorem shortest_distance_between_circles :
    ∀ (x y : ℝ), (circle1 : x^2 - 8 * x + y^2 + 6 * y = 8) →
                 (circle2 : x^2 + 6 * x + y^2 - 2 * y = 1) →
    shortest_distance circle1 circle2 = 0.68 :=
by
  -- Proof to be provided
  sorry

end shortest_distance_between_circles_l668_668493


namespace probability_of_correct_dialing_l668_668956

theorem probability_of_correct_dialing : 
  let possible_first_segments := {3086, 3089, 3098}
  let possible_last_digits := {0, 1, 2, 5}
  let num_possible_first_segments := 3
  let num_permutations_last_digits := 24
  let total_combinations := num_possible_first_segments * num_permutations_last_digits
  let correct_combinations := 1
  let probability := (correct_combinations : ℚ) / total_combinations
  probability = 1 / 72 :=
by
  trivial

end probability_of_correct_dialing_l668_668956


namespace sequence_product_l668_668232

def sequence_term (n : ℕ) : ℚ :=
if n % 2 = 0 then 1 / (2^(2 * (n / 2 + 1))) 
else 2 ^ (2 * (n / 2 + 1) + 1)

def product_sequence : ℚ :=
let terms := list.map sequence_term (list.range 10)
terms.prod

theorem sequence_product : product_sequence = 32 := sorry

end sequence_product_l668_668232


namespace difference_between_x_and_y_l668_668333

theorem difference_between_x_and_y 
  (x y : ℕ) 
  (h1 : 3 ^ x * 4 ^ y = 531441) 
  (h2 : x = 12) : x - y = 12 := 
by 
  sorry

end difference_between_x_and_y_l668_668333


namespace external_bisector_angle_is_50_l668_668006

-- Given the conditions
variable (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variable (α : RealAngle)
variable (AB_extension_internal_bisector_angle : RealAngle)

#check AB_extension_internal_bisector_angle
#check α

def angle_of_external_bisector (α : RealAngle) (AB_extension_internal_bisector_angle : RealAngle) : RealAngle := sorry

-- Internal bisector forms an angle of 40 degrees with AB
axiom internal_bisector_angle : AB_extension_internal_bisector_angle = 40
-- Angle alpha is calculated as twice of the internal bisector angle
axiom angle_Alpha : α = 2 * AB_extension_internal_bisector_angle

-- The angle formed by the external bisector with the extension of side AB is 50 degrees
theorem external_bisector_angle_is_50 (α : RealAngle) (AB_extension_internal_bisector_angle : RealAngle) : 
  angle_of_external_bisector α AB_extension_internal_bisector_angle = 50 := sorry

end external_bisector_angle_is_50_l668_668006


namespace part1_part2_l668_668266

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (π + α)) / (Real.tan (π - α))

theorem part1 : f (5 * π / 6) = -Real.sqrt 3 / 2 := 
  by sorry

theorem part2 (α : ℝ) (h : f (π / 3 - α) = 2 / 3) : Real.sin (α + π / 6) = 2 / 3 := 
  by sorry

end part1_part2_l668_668266


namespace quadratic_inequality_solution_l668_668430

theorem quadratic_inequality_solution (x : ℝ) :
  (-3 * x^2 + 8 * x + 3 > 0) ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l668_668430


namespace fraction_of_menu_vegan_soy_free_l668_668201

def num_vegan_dishes : Nat := 6
def fraction_menu_vegan : ℚ := 1 / 4
def num_vegan_dishes_with_soy : Nat := 4

def num_vegan_soy_free_dishes : Nat := num_vegan_dishes - num_vegan_dishes_with_soy
def fraction_vegan_soy_free : ℚ := num_vegan_soy_free_dishes / num_vegan_dishes
def fraction_menu_vegan_soy_free : ℚ := fraction_vegan_soy_free * fraction_menu_vegan

theorem fraction_of_menu_vegan_soy_free :
  fraction_menu_vegan_soy_free = 1 / 12 := by
  sorry

end fraction_of_menu_vegan_soy_free_l668_668201


namespace circumcenter_is_point_equidistant_from_vertices_l668_668835

theorem circumcenter_is_point_equidistant_from_vertices (triangle : Type) [inhabited triangle] :
  (∃ (circumcenter : triangle), 
     ∀ (vertex : triangle), distance circumcenter vertex = radius) ↔
  intersection_of_perpendicular_bisectors triangle = circumcenter :=
sorry

end circumcenter_is_point_equidistant_from_vertices_l668_668835


namespace find_p_l668_668966

-- Define X as a binomial random variable B(n, p)
def is_binomial (X : ℕ → ℕ) (n : ℕ) (p : ℚ) : Prop :=
  ∀ k, X k = binomial_pmf n p k

-- Define the expectation of X
def E (X : ℕ → ℕ) [Fintype (X)] : ℚ :=
  ∑ k, (X k) * (k / (∑ k, X k))

-- Define the variance of X
def D (X : ℕ → ℕ) [Fintype (X)] : ℚ :=
  E ((λ x, (x - E X) ^ 2) ∘ X)

-- The given conditions
variables (n : ℕ) (p : ℚ)
variables (X : ℕ → ℕ)
hypothesis (h1 : is_binomial X n p)
hypothesis (h2 : E X = 4)
hypothesis (h3 : D X = 3)

-- The statement to be proved
theorem find_p : p = 1 / 4 :=
sorry

end find_p_l668_668966


namespace sum_in_base3_l668_668758

def is_valid_digit (d : ℕ) : Prop := d = 0 ∨ d = 1 ∨ d = 2

def valid_five_digit_numbers : set (list ℕ) :=
  {l | l.length = 5 ∧ (∀ x ∈ l, is_valid_digit x) ∧ (l.head = some 1 ∨ l.head = some 2)}

noncomputable def convert_to_base3 (l : list ℕ) : ℕ :=
  l.enum_from 0 |>.foldl (λ acc ⟨i, d⟩ => acc + d * 3^i) 0

noncomputable def sum_of_elements_in_T : ℕ :=
  valid_five_digit_numbers.to_finset.sum convert_to_base3

theorem sum_in_base3 :
  nat.to_digits 3 (sum_of_elements_in_T) = [1, 1, 2, 2, 1, 1, 2] :=
sorry

end sum_in_base3_l668_668758


namespace least_n_product_exceeds_one_million_l668_668316

noncomputable def product_exceeds_one_million (n : ℕ) : Prop :=
  (∏ i in finset.range n, 2^((i+1) / 13 : ℝ)) > 1000000

theorem least_n_product_exceeds_one_million :
  ∃ n : ℕ, product_exceeds_one_million n ∧ ∀ m < n, ¬product_exceeds_one_million m :=
begin
  sorry
end

end least_n_product_exceeds_one_million_l668_668316


namespace matrix_power_15_l668_668286

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ 0, -1,  0;
      1,  0,  0;
      0,  0,  1]

-- Define what we want to prove
theorem matrix_power_15 :
  B^15 = !![ 0,  1,  0;
            -1,  0,  0;
             0,  0,  1] :=
sorry

end matrix_power_15_l668_668286


namespace photographs_taken_l668_668610

theorem photographs_taken (P : ℝ) (h : P + 0.80 * P = 180) : P = 100 :=
by sorry

end photographs_taken_l668_668610


namespace dance_troupe_minimum_members_l668_668167

theorem dance_troupe_minimum_members :
  ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 9 = 0 ∧ n % 12 = 0 ∧ n % 5 = 0 ∧ n = 180 :=
begin
  use 180,
  split,
  { norm_num }, -- Prove that 180 > 0
  split,
  { norm_num }, -- Prove that 180 % 6 = 0
  split,
  { norm_num }, -- Prove that 180 % 9 = 0
  split,
  { norm_num }, -- Prove that 180 % 12 = 0
  split,
  { norm_num }, -- Prove that 180 % 5 = 0
  { norm_num }, -- Prove that 180 = 180
end

end dance_troupe_minimum_members_l668_668167


namespace vasya_days_without_purchase_l668_668912

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668912


namespace volume_tetrahedron_l668_668975

noncomputable def calculateVolume (R : ℝ) (AB AC BC : ℝ) : ℝ :=
  let cosACB := (AC^2 + BC^2 - AB^2) / (2 * AC * BC)
  let angleACB := real.acos cosACB
  let circumradius := (AC * BC / real.sin angleACB) / 2
  let distanceFromOtoABC := √(R^2 - circumradius^2)
  let areaOfABC := real.sqrt (circumradius^2 - (AB ^ 2) / 4) / 2
  let volumeOABC := (1 / 3) * areaOfABC * distanceFromOtoABC
  volumeOABC

theorem volume_tetrahedron {O : Point} {A B C : Point} (R : ℝ) (hR : R = 13)
    (hAB : dist A B = 12 * real.sqrt 3) 
    (hAC : dist A C = 12) 
    (hBC : dist B C = 12) :
    calculateVolume R 12 12 (12 * real.sqrt 3) = 60 * real.sqrt 3 :=
by
  delta calculateVolume
  simp [hR, hAB, hAC, hBC]
  sorry

end volume_tetrahedron_l668_668975


namespace triangle_angles_l668_668282

-- Definitions used in conditions
def is_triangle (A B C : ℝ) := A + B + C = 180 -- Sum of angles in a triangle is 180 degrees
def is_isosceles (A B C : ℝ) := A = B ∨ B = C ∨ A = C -- Triangle is isosceles

-- Main theorem statement
theorem triangle_angles (A B C : ℝ) (hB : B = 120) 
    (h1 : ∃ D, is_isosceles B D C ∧ is_triangle A B C) :
    (A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) :=
begin
  sorry
end

end triangle_angles_l668_668282


namespace equilateral_iff_rotation_l668_668046

noncomputable def is_equilateral {A B C : Point} (ABC : Triangle A B C) : Prop :=
(side_length A B = side_length B C) ∧ 
(side_length B C = side_length C A)

noncomputable def rotation_condition {A B C : Point} (rotation: Rotation A 60) : Prop :=
rotation B = C

theorem equilateral_iff_rotation (A B C: Point) (ABC: Triangle A B C) (rotation: Rotation A 60):
  is_equilateral ABC ↔ rotation_condition rotation := 
sorry

end equilateral_iff_rotation_l668_668046


namespace lines_form_at_least_100_triangles_l668_668267

theorem lines_form_at_least_100_triangles (n : ℕ) (hn : n = 300) :
  ∀ (lines : set (set (ℝ×ℝ))), -- assuming lines is a set of sets of points representing lines
  (∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → ∀ x ∈ l₁, ∀ y ∈ l₂, x ≠ y) ∧ -- no two lines are parallel
  (∀ l₁ l₂ l₃ ∈ lines, l₁ ≠ l₂ → l₁ ≠ l₃ → l₂ ≠ l₃ → -- no three lines intersect at a single point
    ∀ p ∈ l₁ ∩ l₂, ∀ q ∈ l₁ ∩ l₃, ∀ r ∈ l₂ ∩ l₃, p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  ∃ t : ℕ, t ≥ 100 ∧ ∃ triangles : set (set (ℝ×ℝ)), -- at least 100 triangles
  t = triangles.card ∧
  (∀ triangle ∈ triangles, ∃ a b c ∈ ⋃₀ lines, -- each triangle has three distinct points from the lines
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (a, b, c) ∈ lines ∧ 
    (∃ l₁ l₂ l₃ ∈ lines, {a, b, c} ⊆ l₁ ∪ l₂ ∪ l₃)) :=
sorry

end lines_form_at_least_100_triangles_l668_668267


namespace train_crossing_man_time_l668_668327

theorem train_crossing_man_time (man_speed_kmh : ℝ) (train_speed_kmh : ℝ) (train_length_m : ℝ)
  (man_speed_conv : man_speed_kmh = 3)
  (train_speed_conv : train_speed_kmh = 63) 
  (train_length_conv : train_length_m = 700) : 
  let man_speed_ms := man_speed_kmh * 1000 / 3600 in
  let train_speed_ms := train_speed_kmh * 1000 / 3600 in
  let relative_speed_ms := train_speed_ms - man_speed_ms in
  let crossing_time := train_length_m / relative_speed_ms in
  crossing_time = 42 :=
by
  simp [man_speed_conv, train_speed_conv, train_length_conv]
  sorry

end train_crossing_man_time_l668_668327


namespace number_8437d_multiple_of_6_l668_668811

theorem number_8437d_multiple_of_6 (d : ℕ) (h1 : d = 2 ∨ d = 8) :
  (84370 + d) % 6 = 0 :=
by
  cases h1 with
  | inl h2 =>
    rw [h2]
    norm_num
  | inr h2 =>
    rw [h2]
    norm_num

end number_8437d_multiple_of_6_l668_668811


namespace instantaneous_velocity_at_t_eq_2_l668_668825

def motion_equation (t : ℝ) := (1/3) * t^3 + 3

theorem instantaneous_velocity_at_t_eq_2 : 
  (deriv motion_equation 2) = 4 :=
by
  sorry

end instantaneous_velocity_at_t_eq_2_l668_668825


namespace greatest_power_of_3_dividing_factorial_l668_668516

theorem greatest_power_of_3_dividing_factorial (k : ℕ) (h1 : ∀ m : ℕ, 3^m ∣ (nat.factorial 15) ↔ m ≤ k) (h2 : 0 < k) : k = 8 :=
sorry

end greatest_power_of_3_dividing_factorial_l668_668516


namespace find_a12_l668_668684

def a : ℕ → ℤ
noncomputable def arithmetic_sequence (a : ℕ → ℤ) :=
∀ (n m : ℕ), (n < m) → a (m+1) - a m = a (n+1) - a n

axiom condition_1 {a : ℕ → ℤ} (arithmetic_sequence : ∀ (n m : ℕ), (n < m) → a (m+1) - a m = a (n+1) - a n) :
  a 7 + a 9 = 16

axiom condition_2 {a : ℕ → ℤ} (arithmetic_sequence : ∀ (n m : ℕ), (n < m) → a (m+1) - a m = a (n+1) - a n) :
  a 4 = 1

theorem find_a12 (a : ℕ → ℤ) (h1 : ∀ n m, (n < m) → a (m + 1) - a m = a (n + 1) - a n)
  (cond1 : a 7 + a 9 = 16)
  (cond2 : a 4 = 1) :
  a 12 = 15 :=
sorry

end find_a12_l668_668684


namespace miranda_heels_cost_l668_668777

theorem miranda_heels_cost (months_saved : ℕ) (savings_per_month : ℕ) (gift_from_sister : ℕ) 
  (h1 : months_saved = 3) (h2 : savings_per_month = 70) (h3 : gift_from_sister = 50) : 
  months_saved * savings_per_month + gift_from_sister = 260 := 
by
  sorry

end miranda_heels_cost_l668_668777


namespace correct_system_of_equations_l668_668146

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l668_668146


namespace sunscreen_cost_l668_668861

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end sunscreen_cost_l668_668861


namespace smallest_possible_n_l668_668497

theorem smallest_possible_n (n : ℕ) (h : n > 0) 
  (h_condition : Nat.lcm 60 n / Nat.gcd 60 n = 44) : n = 165 := by
  sorry

end smallest_possible_n_l668_668497


namespace num_possible_diagonal_AC_l668_668253

-- Given points A, B, C, and D forming a quadrilateral, we define:
def point (x y : ℝ) := (x, y)

def A := point 0 0
def B := point 4 6
def C := point 11 2
def D := point 6 (-7)

-- Define the lengths of the sides of the quadrilateral based on distance formula
def length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def AB := length A B  -- 6
def BC := length B C  -- 9
def CD := length C D  -- 14
def DA := length D A  -- 10

-- Define the length of diagonal AC
def AC := length A C

-- Problem Statement:
theorem num_possible_diagonal_AC : ∃ n : ℕ, n = 12 := by
  sorry

end num_possible_diagonal_AC_l668_668253


namespace number_of_photos_to_form_square_without_overlapping_l668_668328

theorem number_of_photos_to_form_square_without_overlapping 
  (width length : ℕ)
  (h_width : width = 12)
  (h_length : length = 15) : 
  let lcm_value := Nat.lcm width length in
  let area_square := lcm_value * lcm_value in
  let area_photo := width * length in
  area_square / area_photo = 20 :=
by
  -- Definitions
  let width := 12
  let length := 15
  let lcm_value := Nat.lcm width length
  let area_square := lcm_value * lcm_value
  let area_photo := width * length

  -- Proof
  -- Prove LCM(12, 15) = 60
  have lcm_eq : lcm_value = 60 := 
    by 
      simp [Nat.lcm]; norm_num
   
   -- Prove (60^2 / (12 * 15) = 20)
  have result : area_square / area_photo = 20 := 
    by 
      rw [lcm_eq, Nat.mul_div_cancel, Nat.pow_two]
      norm_num

  exact result

end number_of_photos_to_form_square_without_overlapping_l668_668328


namespace tan_subtraction_cos_double_angle_l668_668662

variables (θ : ℝ)
axiom tan_theta_eq_two : tan θ = 2

theorem tan_subtraction :
  tan (π/4 - θ) = -1/3 :=
by
  have h := tan_theta_eq_two
  sorry

theorem cos_double_angle :
  cos (2 * θ) = -3/5 :=
by
  have h := tan_theta_eq_two
  sorry

end tan_subtraction_cos_double_angle_l668_668662


namespace salary_based_on_tax_l668_668877

theorem salary_based_on_tax (salary tax paid_tax excess_800 excess_500 excess_500_2000 : ℤ) 
    (h1 : excess_800 = salary - 800)
    (h2 : excess_500 = min excess_800 500)
    (h3 : excess_500_2000 = excess_800 - excess_500)
    (h4 : paid_tax = (excess_500 * 5 / 100) + (excess_500_2000 * 10 / 100))
    (h5 : paid_tax = 80) :
  salary = 1850 := by
  sorry

end salary_based_on_tax_l668_668877


namespace limit_a_n_l668_668383

noncomputable def f₁ (x : ℝ) (a₁ b₁ c₁ : ℝ) : ℝ := x^3 + a₁ * x^2 + b₁ * x + c₁

theorem limit_a_n (α β γ : ℝ) (hαβγ : α > β ∧ β > γ ∧ γ > 0)
 (h₁ : ∀ x, f₁ x (-(α + β + γ)) (α*β + α*γ + β*γ) (-(α*β*γ)) = 0) : 
  (lim (λ n, real.sqrt (2^(n-1) (-aₙ α β γ))) at_top) = α :=
sorry

end limit_a_n_l668_668383


namespace expression_value_l668_668876

theorem expression_value :
    (2.502 + 0.064)^2 - ((2.502 - 0.064)^2) / (2.502 * 0.064) = 4.002 :=
by
  -- the proof goes here
  sorry

end expression_value_l668_668876


namespace sum_of_circle_areas_l668_668526

theorem sum_of_circle_areas (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) : 
  π * r^2 + π * s^2 + π * t^2 = 56 * π := 
by 
  sorry

end sum_of_circle_areas_l668_668526


namespace truthful_dwarfs_count_l668_668598

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l668_668598


namespace isosceles_triangles_with_perimeter_27_count_l668_668706

theorem isosceles_triangles_with_perimeter_27_count :
  ∃ n, (∀ (a : ℕ), 7 ≤ a ∧ a ≤ 13 → ∃ (b : ℕ), b = 27 - 2*a ∧ b < 2*a) ∧ n = 7 :=
sorry

end isosceles_triangles_with_perimeter_27_count_l668_668706


namespace regular_polygon_sides_l668_668181

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l668_668181


namespace expression_value_l668_668772

theorem expression_value : 
  let x := 1 + Real.sqrt 2
  let y := x + 1
  let z := x - 1
  y^2 * z^4 - 4 * y^3 * z^3 + 6 * y^2 * z^2 + 4 * y = -120 - 92 * Real.sqrt 2 :=
by
  let x := 1 + Real.sqrt 2
  let y := x + 1
  let z := x - 1
  calc
    y^2 * z^4 - 4 * y^3 * z^3 + 6 * y^2 * z^2 + 4 * y = -120 - 92 * Real.sqrt 2 : sorry

end expression_value_l668_668772


namespace general_term_sequence_l668_668277

-- Define the sequence {a_n}
def seq (n : ℕ) : ℕ → ℕ
| 0 := 2
| (n+1) := 2 * seq n

-- Define the sum S_n of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
nat.rec_on n 0 (λ n sum, sum + seq n)

-- Define the conditions
def is_arithmetic_mean (S a : ℕ) : Prop := a = (S + 2) / 2
def sum_condition (S a : ℕ) : Prop := S = 2 * a - 2

-- The final theorem to be proved
theorem general_term_sequence (n : ℕ) : seq n = 2 ^ n :=
by sorry

end general_term_sequence_l668_668277


namespace number_of_truthful_dwarfs_l668_668592

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l668_668592


namespace minimum_value_of_F_l668_668340

theorem minimum_value_of_F (f g : ℝ → ℝ) (a b : ℝ) (h_odd_f : ∀ x, f (-x) = -f x) 
  (h_odd_g : ∀ x, g (-x) = -g x) (h_max_F : ∃ x > 0, a * f x + b * g x + 3 = 10) 
  : ∃ x < 0, a * f x + b * g x + 3 = -4 := 
sorry

end minimum_value_of_F_l668_668340


namespace ellipse_eqn_length_MN_eq_range_y0_eq_l668_668686

noncomputable def ellipse_equation (C : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, C x y ↔ (x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0 ∧ F = (1, 0) ∧ A = (2, 0))

noncomputable def line_through_F (l : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, l x = x - 1 ∧ l(F.1) = F.2

noncomputable def length_segment_MN (C : ℝ → ℝ → Prop) (l : ℝ → ℝ) : ℝ :=
  let points := {p : ℝ × ℝ | ∃ x : ℝ, (p.1 = x ∧ p.2 = l x ∧ C p.1 p.2)} in
  let M := points.min in
  let N := points.max in
  real.sqrt (1 + 1) * real.abs (M.1 - N.1)

noncomputable def range_y0 (l : ℝ → ℝ) (C : ℝ → ℝ → Prop) : set ℝ :=
  {y | ∃ x1 x2 M x', 
    (x1 ≠ x2 ∧ x' = (4 * (l (1))^2)/(4 * (l (1))^2 + 3) ∧ 
    M = (x', l (x') - 1/(l(1)) * x') ∧ 
    y = (l(1) / (4 * (l(1))^2 + 3) ∨ 
        - (l(1) / (4 * (l(1))^2 + 3)))}

theorem ellipse_eqn : ∃ C : ℝ → ℝ → Prop, ellipse_equation C 2 (real.sqrt 3) :=
sorry

theorem length_MN_eq : ∀ l : ℝ → ℝ, line_through_F l → length_segment_MN C l = 24 / 7 :=
sorry

theorem range_y0_eq : ∀ l : ℝ → ℝ, line_through_F l → range_y0 l C = Icc (- real.sqrt 3 / 12) (real.sqrt 3 / 12) :=
sorry

end ellipse_eqn_length_MN_eq_range_y0_eq_l668_668686


namespace expected_value_L_n_ge_sqrt_n_l668_668868

noncomputable def L_n (σ : List ℕ) : ℕ := 
  -- Placeholder for the actual definition of the length of the longest increasing subsequence
  sorry

theorem expected_value_L_n_ge_sqrt_n (n : ℕ) [Fact (0 < n)] :
  let σs := {σ : List ℕ // σ.perm_of_list.to_finset = Finset.range (n + 1)} in
  ∑ σ in σs, L_n σ / σs.card.to_real >= Real.sqrt n :=
sorry

end expected_value_L_n_ge_sqrt_n_l668_668868


namespace truthful_dwarfs_count_l668_668602

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l668_668602


namespace train_crossing_time_l668_668137

theorem train_crossing_time (length_of_train : ℕ) (speed_km_per_hr : ℕ) (conversion_factor : ℚ) :
  length_of_train = 55 →
  speed_km_per_hr = 36 →
  conversion_factor = (5 : ℚ) / 18 →
  (length_of_train / (speed_km_per_hr * conversion_factor)) = 5.5 :=
by
  assume h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_crossing_time_l668_668137


namespace bird_families_left_near_mountain_l668_668131

def total_bird_families : ℕ := 85
def bird_families_flew_to_africa : ℕ := 23
def bird_families_flew_to_asia : ℕ := 37

theorem bird_families_left_near_mountain : total_bird_families - (bird_families_flew_to_africa + bird_families_flew_to_asia) = 25 := by
  sorry

end bird_families_left_near_mountain_l668_668131


namespace greatest_error_sum_smallest_error_diff_l668_668437

-- Define the approximate numbers with their errors
variables {a b c : ℝ}
variables {α β γ : ℝ}

-- Condition: errors for sum
def approx_sum (a b c : ℝ) (α β γ : ℝ) : ℝ := (a + b + c) ± (α + β + γ)

-- Condition: errors for difference
def approx_diff (a b : ℝ) (α β : ℝ) : ℝ := (a - b) ± (α - β)

-- Statement for the greatest error in the sum
theorem greatest_error_sum:
  ∀ (a b c : ℝ) (α β γ : ℝ), 
    abs ((a + b + c) + (α + β + γ)) = α + β + γ :=
begin
  sorry
end

-- Statement for the smallest error in the difference
theorem smallest_error_diff:
  ∀ (a b : ℝ) (α β : ℝ), 
    α = β → abs ((a - b) ± (α - β)) = α - β :=
begin
  sorry
end

end greatest_error_sum_smallest_error_diff_l668_668437


namespace integral_of_exp_over_circle_l668_668987

open Complex

theorem integral_of_exp_over_circle :
  (∮ z in C(0, 1), (exp (2 * z)) / (z + (π * I / 2)) ^ 2) = -4 * π * I :=
sorry

end integral_of_exp_over_circle_l668_668987


namespace truthful_dwarfs_count_l668_668599

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l668_668599


namespace determine_Sets_l668_668571

variable (S : Set Point)

def constraints (S : Set Point) : Prop :=
  ∀ {A B C D : Point}, A ∈ S → B ∈ S → C ∈ S → D ∈ S → A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (∃ (circle : Circle), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∨
  ∃ (line : Line), A ∈ line ∧ B ∈ line ∧ C ∈ line ∧ D ∈ line)

theorem determine_Sets (S : Set Point) (h : constraints S):
  (∃ (circle : Circle), ∀ (p : Point), p ∈ S → p ∈ circle) ∨
  (∃ (A B C D : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧
  ∀ E ∈ Set.Points where E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D → ∃ (P : Point), P ∈ Set.inter (Line.through A B) (Line.through C D) ∨
  P ∈ Set.inter (Line.through A C) (Line.through B D) ∨ P ∈ Set.inter (Line.through A D) (Line.through B C)) ∨
  (∃ (line : Line), ∀ (p1 : Point), p1 ∈ S → p1 ∈ line ∧ ∃ p2, p2 ∈ S ∧ p2 ∉ line) := sorry

end determine_Sets_l668_668571


namespace symmetric_ln_2_minus_x_l668_668979

theorem symmetric_ln_2_minus_x :
  ∀ x : ℝ, ∃ f : ℝ → ℝ, (f x = Real.log x) ∧ (f (2 - x) = Real.log (2 - x)) :=
by
  intro x
  use Real.log
  split
  . exact rfl
  . exact rfl
  sorry

end symmetric_ln_2_minus_x_l668_668979


namespace angle_QSR_72_degrees_l668_668741

theorem angle_QSR_72_degrees (P Q R S M N O : Type)
  [DecidableEq P] [DecidableEq Q] [DecidableEq R] [DecidableEq S]
  [DecidableEq M] [DecidableEq N] [DecidableEq O]
  (h_altitudes : ∀ (PQR : Type), (∀ x ∈ [PM, QN, RO], x intersects at S))
  (h_angle_PQR : angle P Q R = 53)
  (h_angle_PRQ : angle P R Q = 19)
  : angle Q S R = 72 := 
sorry

end angle_QSR_72_degrees_l668_668741


namespace smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l668_668576

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l668_668576


namespace heads_not_consecutive_probability_l668_668107

theorem heads_not_consecutive_probability :
  (∃ n m : ℕ, n = 2^4 ∧ m = 1 + Nat.choose 4 1 + Nat.choose 3 2 ∧ (m / n : ℚ) = 1 / 2) :=
by
  use 16     -- n
  use 8      -- m
  sorry

end heads_not_consecutive_probability_l668_668107


namespace molecular_weight_of_compound_l668_668872

def num_atoms_C : ℕ := 6
def num_atoms_H : ℕ := 8
def num_atoms_O : ℕ := 7

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  nC * wC + nH * wH + nO * wO

theorem molecular_weight_of_compound :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 192.124 :=
by
  sorry

end molecular_weight_of_compound_l668_668872


namespace stamp_cost_91_2_grams_l668_668878

theorem stamp_cost_91_2_grams : ∀ (weight : ℝ), 
  (weight = 91.2) →
  (∀ (w : ℝ), 0 ≤ w ∧ w ≤ 20 → 0.8) →
  (∀ (w : ℝ), 20 < w ∧ w ≤ 40 → 1.6) →
  (∀ (w : ℝ), 40 < w ∧ w ≤ 60 → 2.4) →
  ∃ (cost : ℝ), cost = 4 :=
sorry

end stamp_cost_91_2_grams_l668_668878


namespace sum_of_integers_l668_668859

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 343000)
  (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
  a + b + c = 476 :=
by
  sorry

end sum_of_integers_l668_668859


namespace regression_prediction_8_days_latest_adjustment_day_l668_668228
open Real

-- Definitions based on given conditions
def b : ℝ := 25.2 / 28 
def ln_k : ℝ := -1.2 - (25.2 / 28) * 4
def k : ℝ := exp ln_k
def regression_eqn (t : ℝ) : ℝ := k * exp (b * t)
def predicted_sinking_8 := regression_eqn 8

-- Proving the required statements
theorem regression_prediction_8_days :
  regression_eqn 8 = exp 2.4 := 
by
  -- Given calculations derive exp(2.4)
  sorry

def sinking_rate (t : ℝ) := b * k * exp (b * t)

theorem latest_adjustment_day :
  (∃ n : ℝ, sinking_rate n > 27 ∧ n = 9) :=
by
  -- From given conditions derive n = 9 to meet the sinking rate condition
  have rate_bound : sinking_rate n > 27 :=
    sorry
  -- Given rate_bound and calculations derive n = 9
  sorry

end regression_prediction_8_days_latest_adjustment_day_l668_668228


namespace Tim_age_l668_668106

theorem Tim_age (T t : ℕ) (h1 : T = 22) (h2 : T = 2 * t + 6) : t = 8 := by
  sorry

end Tim_age_l668_668106


namespace min_sales_week_is_fourth_sales_amount_week3_total_profit_preferred_scheme_for_7_pieces_l668_668165

-- Defining conditions
def cost_per_piece : ℝ := 3.5
def selling_price_per_piece : ℝ := 6.0
def promotional_prices : ℕ → ℝ
| 1 := 4.5
| 2 := 5.0
| 3 := 5.5
| 4 := 6.0
| _ := 6.0  -- Assuming any week beyond the fourth has no further promotion

def standard_sales_per_week : ℕ := 200
def actual_sales : ℕ → ℤ
| 1 := 28
| 2 := 16
| 3 := -6
| 4 := -12
| _ := 0

-- Problem Proofs

-- Part 1: Minimum sales week (week 4) and sales amount in week 3
theorem min_sales_week_is_fourth : ∀ w, (w = 4) → actual_sales 4 = min (actual_sales 1) (min (actual_sales 2) (min (actual_sales 3) (actual_sales 4)))
  sorry

theorem sales_amount_week3 : ∀ (sales : ℕ), (sales = 194) → (sales : ℝ) * promotional_prices 3 = 1067
  sorry

-- Part 2: Total profit for the four weeks
theorem total_profit : ∀ (profit : ℕ), (profit = (228 + 324 + 388 + 470)) → profit = 1410
  sorry

-- Part 3: Preferred discount scheme (Scheme One)
def scheme_one_profit (pieces : ℕ) : ℝ := (selling_price_per_piece - cost_per_piece - 0.3) * pieces
def scheme_two_profit (pieces : ℕ) : ℝ := 
  let full_price_pieces := min pieces 3
  let discount_price_pieces := max (pieces - 3) 0
  in (full_price_pieces * selling_price_per_piece + discount_price_pieces * (selling_price_per_piece * 0.9)) - (cost_per_piece * pieces)

theorem preferred_scheme_for_7_pieces : scheme_one_profit 7 > scheme_two_profit 7
  sorry

end min_sales_week_is_fourth_sales_amount_week3_total_profit_preferred_scheme_for_7_pieces_l668_668165


namespace correct_system_of_equations_l668_668148

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l668_668148


namespace median_age_could_be_16_l668_668849

open Classical

theorem median_age_could_be_16 (F_16 : ℕ) :
  let total_members := 5 + 7 + 13 + F_16 in
  13 < total_members →
  total_members % 2 = 0 →
  F_16 > 25 - 13 →
  16 ∈ {(total_members / 2).natFloor, (total_members / 2).natCeil} → 
  true :=
by {
  intro F_16,
  intro total_members ht hmod hF,
  sorry
}

end median_age_could_be_16_l668_668849


namespace sum_of_T_l668_668209

def is_repeating_decimal (x : ℝ) : Prop :=
  ∃ (a b c : ℤ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ x = (a * 100 + b * 10 + c) / 999

def T : set ℝ := {x | is_repeating_decimal x}

theorem sum_of_T : ∑ x in T, x = 500 := 
  sorry

end sum_of_T_l668_668209


namespace PQ_squared_l668_668347

structure Quadrilateral :=
(E F G H : Point)
(convex : Convex EF GH)
(EF_eq_FG : EF.dist_to FG = 15)
(GH_eq_HE : GH.dist_to HE = 20)
(angle_H_eq_90 : ∠ H = 90)
(midpoint_P : P.isMidpoint F G)
(midpoint_Q : Q.isMidpoint H E)

theorem PQ_squared (EFGH : Quadrilateral) : 
  let P := midpoint EFGH.F EFGH.G
  let Q := midpoint EFGH.H EFGH.E
  P.dist_to Q ^ 2 = 156.25 := 
sorry

end PQ_squared_l668_668347


namespace square_grid_tiling_t_shaped_l668_668634

theorem square_grid_tiling_t_shaped (n : ℕ) (h : n ≥ 1) :
  (∃ tile_fn : (ℕ → ℕ → bool), (∀ i j : ℕ, ij ∈ (list.range n).product (list.range n) → {tile_fn i j = tt ∨ tile_fn i j = ff)) → -- Valid tiling function
    (∀ i j : ℕ, tile_fn i j → -- Tiling does not overlap
      (tile_fn (i + 1) j ∧ tile_fn (i + 2) j ∧ tile_fn (i + 1) (j - 1))) → -- Tiling covers 4 squares
    (∀ i j : ℕ, tile_fn i j → (i + 1) < n ∧ (j + 1) < n) → -- No overflow
  (∃ tiles : ℕ → ℕ → bool, wholetiling grid n (n - 1))
) ↔ n % 4 = 0 :=
sorry

end square_grid_tiling_t_shaped_l668_668634


namespace solve_for_A_l668_668799

theorem solve_for_A (x y : ℝ) (A : ℝ) 
  (h_eq : 2.343 * A = (sqrt (4 * (x - sqrt y) + y / x) * 
                       sqrt (9 * x^2 + 6 * (2 * y * x^3)^(1/3) + (4 * y^2)^(1/3))) / 
                      (6 * x^2 + 2 * (2 * y * x^3)^(1/3) - 3 * sqrt (y * x^2) - (4 * y^5)^(1/6)))
  : (y > 4 * x^2 → A = -1 / sqrt x) ∧ (0 ≤ y ∧ y < 4 * x^2 → A = 1 / sqrt x) := 
sorry

end solve_for_A_l668_668799


namespace compute_expression_l668_668990

-- Given conditions
lemma sin_60_is_sqrt3_over_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
lemma pow_zero_of_any_number : ∀ x : Real, x ^ 0 = 1 := by {
  intro x, rw Real.zero_rpow, norm_num
}
lemma cube_root_of_27 : Real.cbrt 27 = 3 := sorry
lemma reciprocal_of_half : (1 / 2)⁻¹ = 2 := sorry

-- The theorem to prove
theorem compute_expression : 
  2 * (Real.sin (60 * Real.pi / 180)) + (3.14 - Real.pi)^0 - Real.cbrt 27 + (1 / 2)⁻¹ = Real.sqrt 3 := 
by {
  have h1 := sin_60_is_sqrt3_over_2,
  have h2 := pow_zero_of_any_number (3.14 - Real.pi),
  have h3 := cube_root_of_27,
  have h4 := reciprocal_of_half,
  rw [h1, h2, h3, h4],
  norm_num,
}

end compute_expression_l668_668990


namespace height_of_old_lamp_l668_668369

theorem height_of_old_lamp (height_new_lamp : ℝ) (height_difference : ℝ) (h : height_new_lamp = 2.33) (h_diff : height_difference = 1.33) : 
  (height_new_lamp - height_difference) = 1.00 :=
by
  have height_new : height_new_lamp = 2.33 := h
  have height_diff : height_difference = 1.33 := h_diff
  sorry

end height_of_old_lamp_l668_668369


namespace elmer_eats_more_l668_668419

def penelope_intake : ℝ := 20
def greta_intake : ℝ := penelope_intake / 10
def milton_intake : ℝ := greta_intake / 100
def elmer_intake : ℝ := milton_intake * 4000
def rosie_intake : ℝ := greta_intake * 3
def carl_intake : ℝ := penelope_intake / 2

def total_others_intake : ℝ :=
  penelope_intake + greta_intake + milton_intake + rosie_intake + carl_intake

def elmer_diff_others : ℝ :=
  elmer_intake - total_others_intake

theorem elmer_eats_more : elmer_diff_others = 41.98 :=
by
  sorry

end elmer_eats_more_l668_668419


namespace mean_difference_incorrect_actual_l668_668814

theorem mean_difference_incorrect_actual :
  ∀ (T : ℝ), 
    let mean_actual := (T + 110000) / 1200
    let mean_incorrect := (T + 220000) / 1200
  in
    mean_incorrect - mean_actual = 91.67 :=
by
  intros
  let mean_actual := (T + 110000) / 1200
  let mean_incorrect := (T + 220000) / 1200
  have h : (T + 220000) / 1200 - (T + 110000) / 1200 = 91.67 := by sorry
  exact h

end mean_difference_incorrect_actual_l668_668814


namespace donut_circumference_ratio_l668_668090

theorem donut_circumference_ratio
  (r_inner r_outer : ℝ)
  (h_inner : r_inner = 2)
  (h_outer : r_outer = 6) :
  (2 * Real.pi * r_outer) / (2 * Real.pi * r_inner) = 3 :=
by
  rw [h_inner, h_outer]
  simp
  sorry

end donut_circumference_ratio_l668_668090


namespace range_of_a_l668_668311

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Ioc 0 1 → f a x ≥ g x) → a ∈ Set.Ici (-2) :=
begin
  sorry
end

end range_of_a_l668_668311


namespace number_of_truthful_gnomes_l668_668605

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l668_668605


namespace solution_set_equivalence_l668_668844

def solution_set_inequality (x : ℝ) : Prop :=
  abs (x - 1) + abs x < 3

theorem solution_set_equivalence :
  { x : ℝ | solution_set_inequality x } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_equivalence_l668_668844


namespace sum_a_b_eq_neg2_l668_668404

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : f a + f b = 20) : a + b = -2 :=
by
  sorry

end sum_a_b_eq_neg2_l668_668404


namespace tan_equals_three_l668_668262

variable (α : ℝ)

theorem tan_equals_three : 
  (Real.tan α = 3) → (1 / (Real.sin α * Real.sin α + 2 * Real.sin α * Real.cos α) = 2 / 3) :=
by
  intro h
  sorry

end tan_equals_three_l668_668262


namespace vasya_did_not_buy_anything_days_l668_668939

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668939


namespace right_angled_triangles_with_area_eq_perimeter_l668_668242

theorem right_angled_triangles_with_area_eq_perimeter :
  ∃ n : ℕ, n = 2 ∧ (∀ (a b c : ℕ), a^2 + b^2 = c^2 → (1 / 2 : ℚ) * a * b = a + b + c → 
    ((a = 5 ∧ b = 12 ∧ c = 13) ∨ 
     (a = 6 ∧ b = 8 ∧ c = 10) ∨ 
     (a = 8 ∧ b = 6 ∧ c = 10) ∨ 
     (a = 12 ∧ b = 5 ∧ c = 13))) :=
begin
  sorry
end

end right_angled_triangles_with_area_eq_perimeter_l668_668242


namespace ratio_of_sums_is_7_over_8_l668_668297

noncomputable def ratio_of_sums (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
    (h1 : a^2 + b^2 + c^2 = 49) 
    (h2 : x^2 + y^2 + z^2 = 64) 
    (h3 : ax + by + cz = 56) : ℝ :=
(a + b + c) / (x + y + z)

theorem ratio_of_sums_is_7_over_8
    (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : a^2 + b^2 + c^2 = 49) 
    (h2 : x^2 + y^2 + z^2 = 64) 
    (h3 : ax + by + cz = 56) :
    ratio_of_sums a b c x y z ha hb hc hx hy hz h1 h2 h3 = 7 / 8 :=
sorry

end ratio_of_sums_is_7_over_8_l668_668297


namespace probability_of_rolling_sevens_l668_668168

theorem probability_of_rolling_sevens :
  let p := (1 : ℚ) / 4
  let q := (3 : ℚ) / 4
  (7 * p^6 * q + p^7 = 22 / 16384) :=
by
  let p := (1 : ℚ) / 4
  let q := (3 : ℚ) / 4
  let k1 := 7 * p^6 * q
  let k2 := p^7
  have h1 : k1 = (21 : ℚ) / 16384 := by sorry
  have h2 : k2 = (1 : ℚ) / 16384 := by sorry
  have h := h1 + h2
  show (k1 + k2 = 22 / 16384)
  exact h

end probability_of_rolling_sevens_l668_668168


namespace vasya_no_purchase_days_l668_668947

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668947


namespace vasya_days_l668_668909

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668909


namespace find_a_minus_b_l668_668390

theorem find_a_minus_b (a b : ℝ) 
  (h1 : ∀ x, f(x) = a * x + b)
  (h2 : ∀ x, g(x) = -4 * x + 3)
  (h3 : ∀ x, h(x) = f(g(x)))
  (h4 : ∀ x, h⁻¹(x) = x + 3) :
  a - b = 2 := 
sorry

end find_a_minus_b_l668_668390


namespace factorial_division_example_l668_668293

theorem factorial_division_example : (10! / (5! * 2!)) = 15120 := 
by
  sorry

end factorial_division_example_l668_668293


namespace solve_x_for_fraction_l668_668800

theorem solve_x_for_fraction :
  ∃ x : ℝ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 14.6 :=
by
  sorry

end solve_x_for_fraction_l668_668800


namespace dwarfs_truthful_count_l668_668589

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l668_668589


namespace infinitely_many_n_satisfiability_l668_668251

theorem infinitely_many_n_satisfiability :
  ∃ (S : Set ℤ), (∀ n ∈ S, (Real.sqrt (n + 1) ≤ Real.sqrt (3 * n + 2)) ∧ (Real.sqrt (3 * n + 2) < Real.sqrt (4 * n - 1))) ∧
  (S = { n | n ≥ 4 }) :=
begin
  sorry
end

end infinitely_many_n_satisfiability_l668_668251


namespace exists_three_pieces_form_triangle_l668_668187

theorem exists_three_pieces_form_triangle (a b c d e : ℝ) (h1 : a + b + c + d + e = 200) 
  (h2 : a ≥ 17) (h3 : b ≥ 17) (h4 : c ≥ 17) (h5 : d ≥ 17) (h6 : e ≥ 17) : 
  ∃ x y z, x + y > z ∧ x + z > y ∧ y + z > x :=
by 
  sorry

end exists_three_pieces_form_triangle_l668_668187


namespace B_cycling_speed_l668_668548

theorem B_cycling_speed (v : ℝ) : 
  (∀ (t : ℝ), 10 * t + 30 = B_start_distance) ∧ 
  (B_start_distance = 60) ∧ 
  (t = 3) →
  v = 20 :=
sorry

end B_cycling_speed_l668_668548


namespace sum_of_variables_l668_668808

theorem sum_of_variables (a b c d : ℤ)
  (h1 : a - b + 2 * c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 20 :=
by
  sorry

end sum_of_variables_l668_668808


namespace impossible_to_all_zero_l668_668115

theorem impossible_to_all_zero (e : Fin 6 → ℕ) :
  (∀ i, e i > 0) → ¬ (∃ steps : List (Fin 4), 
  ∀ step ∈ steps,
    ∀ x ∈ [0, 1, 2], ∃ a b c : ℕ, 
      (a = abs (b - c) ∨ b = abs (c - a) ∨ c = a + b ∧ 
      (∀ i, e i = 0))) :=
by sorry

end impossible_to_all_zero_l668_668115


namespace quadrilateral_inscribed_properties_l668_668175

-- Define the circle and the inscribed quadrilateral
structure Circle where
  center : Point
  radius : ℝ

structure Quadrilateral where 
  A B C D : Point

structure Tangent where
  circle : Circle
  point : Point
  line : Line

-- Define the tangents at opposite vertices intersecting at points M and N
structure TangentsIntersectAtPoints where
  circle : Circle
  quad : Quadrilateral
  tangent1 : Tangent
  tangent2 : Tangent
  M : Point
  N : Point

-- Define the statement for opposite sides intersecting on line MN
def opposite_sides_intersect (circle : Circle) (quad : Quadrilateral) (M N : Point) : Prop := 
  ∃ P Q : Point, (line P Q) = line M N ∧ 
    (intersects ⟨quad.A, quad.B⟩ ⟨quad.C, quad.D⟩ P) ∧
    (intersects ⟨quad.B, quad.C⟩ ⟨quad.D, quad.A⟩ Q)

-- Define the statement for intersection point dividing segment MN in the ratio MA:ND
def divides_in_ratio_MA_ND (M N intersection_point : Point) (quad : Quadrilateral) : Prop :=
  (distance M intersection_point) / (distance intersection_point N) = 
  (distance M quad.A) / (distance N quad.D)

-- Putting everything in a theorem statement
theorem quadrilateral_inscribed_properties (circle : Circle) (quad : Quadrilateral)
    (tangents : TangentsIntersectAtPoints) : 
    (opposite_sides_intersect circle quad tangents.M tangents.N) ∧
    (∃ intersection_point : Point, divides_in_ratio_MA_ND tangents.M tangents.N intersection_point quad) := 
  sorry

end quadrilateral_inscribed_properties_l668_668175


namespace min_varphi_symmetry_l668_668108

theorem min_varphi_symmetry (ϕ : ℝ) (hϕ : ϕ > 0) : 
  ∃ varphi_min, varphi_min = 3 * Real.pi / 8 ∧ f x = 2 * sin (2 * x + Real.pi / 4) ∧
  (∀ x, g x = 2 * sin (4 * x - 2 * ϕ + Real.pi / 4)) ∧ 
  (∀ k ∈ ℤ, g (Real.pi / 4) = g (k * Real.pi + Real.pi / 2) ∨ 
  g (Real.pi / 4) = g (k * Real.pi + Real.pi / 2)) :=
by
  sorry

end min_varphi_symmetry_l668_668108


namespace nice_count_bound_l668_668766

def is_nice (k: ℕ) (ℓ: ℕ) : Prop := ∃ m: ℕ, k! + ℓ = m^2

theorem nice_count_bound (ℓ n : ℕ) (hℓ_pos : 0 < ℓ) (hn_ge_ℓ : ℓ ≤ n) :
  (∃ s : finset ℕ, s.card ≤ n^2 - n + ℓ ∧ ∀ k ∈ s, is_nice k ℓ) :=
sorry

end nice_count_bound_l668_668766


namespace train_blue_boxcars_l668_668792

theorem train_blue_boxcars (B : ℕ) :
  let black_boxcar_capacity := 4000
  let blue_boxcar_capacity := 2 * black_boxcar_capacity
  let red_boxcar_capacity := 3 * blue_boxcar_capacity
  let total_capacity := 132000
  let black_boxcars := 7
  let red_boxcars := 3
  total_capacity = (black_boxcar_capacity * black_boxcars) + (blue_boxcar_capacity * B) + (red_boxcar_capacity * red_boxcars)
  → B = 4 :=
by {
  sorry,
}

end train_blue_boxcars_l668_668792


namespace multiple_proof_l668_668986

noncomputable def K := 185  -- Given KJ's stamps
noncomputable def AJ := 370  -- Given AJ's stamps
noncomputable def total_stamps := 930  -- Given total amount

-- Using the conditions to find C
noncomputable def stamps_of_three := AJ + K  -- Total stamps of KJ and AJ
noncomputable def C := total_stamps - stamps_of_three

-- Stating the equivalence we need to prove
theorem multiple_proof : ∃ M: ℕ, M * K + 5 = C := by
  -- The solution proof here if required
  existsi 2
  sorry  -- proof to be completed

end multiple_proof_l668_668986


namespace num_solutions_system_eqns_l668_668830

open Real

theorem num_solutions_system_eqns :
  (∃ x y : ℝ, y = exp (abs x) - exp 1 ∧ abs (abs x - abs y) = 1) ↔ 6 := 
sorry

end num_solutions_system_eqns_l668_668830


namespace snow_volume_l668_668842

theorem snow_volume
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 15)
  (h_width : width = 3)
  (h_depth : depth = 0.6) :
  length * width * depth = 27 := 
by
  -- placeholder for proof
  sorry

end snow_volume_l668_668842


namespace sum_inverse_a_eq_20_11_l668_668570

noncomputable def ceil (x : ℝ) : ℤ :=
  if x - x.floor = 0 then x.floor
  else x.floor + 1

noncomputable def f (x : ℝ) : ℝ := x * (ceil x)

def A (n : ℕ) : set ℝ :=
  {f x | x ∈ set.Ioc 0 n} 

noncomputable def a (n : ℕ) : ℕ :=
  (A n).toFinset.card

theorem sum_inverse_a_eq_20_11 : 
  (∑ i in (finset.range 10).image (λ i, i + 1), (1 : ℝ) / a i) = 20 / 11 :=
by
  sorry

end sum_inverse_a_eq_20_11_l668_668570


namespace area_of_transformed_parallelogram_l668_668071

variables (u v : ℝ^3)

-- Given condition
def area_12 (h : ℝ) : Prop := ‖u × v‖ = h

-- The proof goal
theorem area_of_transformed_parallelogram (h12 : area_12 u v 12) :
  ‖((3 : ℝ) • u + (4 : ℝ) • v) × ((2 : ℝ) • u - (6 : ℝ) • v)‖ = 312 :=
begin
  sorry
end

end area_of_transformed_parallelogram_l668_668071


namespace vasya_purchase_l668_668932

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668932


namespace pyramidal_integral_value_l668_668114

noncomputable def pyramidal_integral : ℝ :=
  ∫ x in 0..1, ∫ y in 0..(1 - x), ∫ z in 0..(1 - x - y), x * (y^9) * (z^8) * ((1 - x - y - z)^4) ∂z ∂y ∂x

theorem pyramidal_integral_value :
  pyramidal_integral = (nat.factorial 9 * nat.factorial 8 * nat.factorial 4) / nat.factorial 25 := by
  sorry

end pyramidal_integral_value_l668_668114


namespace smallest_positive_n_l668_668732

-- Definitions representing the problem conditions
noncomputable def first_term_positive (a₁ : ℝ) : Prop := a₁ > 0
noncomputable def common_difference_negative (d : ℝ) : Prop := d < 0
noncomputable def sum_representation (S : ℕ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  S n = a * n^2 + b * n
noncomputable def axis_of_symmetry (a b : ℝ) : Prop :=
  -b / (2 * a) = 10
noncomputable def S_nineteen_positive (S : ℕ → ℝ) : Prop :=
  S 19 > 0
noncomputable def S_twenty_negative (S : ℕ → ℝ) : Prop :=
  S 20 < 0

-- The statement to prove
theorem smallest_positive_n (a b : ℝ) (S : ℕ → ℝ) :
  first_term_positive (S 1) →
  common_difference_negative d →
  sum_representation S a b →
  axis_of_symmetry a b →
  S_nineteen_positive S →
  S_twenty_negative S →
  (S 1 ≤ S 19 ∧ S 1 > 0) ∨ (S 19 ≤ S 1 ∧ S 19 > 0) :=
sorry

end smallest_positive_n_l668_668732


namespace vasya_days_without_purchases_l668_668897

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668897


namespace wendy_total_gas_to_add_l668_668116

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end wendy_total_gas_to_add_l668_668116


namespace liquid_level_ratio_l668_668113

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end liquid_level_ratio_l668_668113


namespace largest_mersenne_prime_less_than_500_l668_668220

-- Define what it means for a number to be prime
def is_prime (p : ℕ) : Prop :=
p > 1 ∧ ∀ (n : ℕ), n > 1 ∧ n < p → ¬ (p % n = 0)

-- Define what a Mersenne prime is
def is_mersenne_prime (m : ℕ) : Prop :=
∃ n : ℕ, is_prime n ∧ m = 2^n - 1

-- We state the main theorem we want to prove
theorem largest_mersenne_prime_less_than_500 : ∀ (m : ℕ), is_mersenne_prime m ∧ m < 500 → m ≤ 127 :=
by 
  sorry

end largest_mersenne_prime_less_than_500_l668_668220


namespace triangle_octagon_side_ratio_l668_668178

theorem triangle_octagon_side_ratio
    (s_t s_o : ℝ)
    (area_triangle : ℝ := (s_t^2 * real.sqrt 3) / 4)
    (area_octagon : ℝ := 2 * real.sqrt 2 * s_o^2)
    (equal_areas : area_triangle = area_octagon) :
    s_t = s_o * 2 * real.sqrt (real.sqrt 2) := by
sorry

end triangle_octagon_side_ratio_l668_668178


namespace picking_at_least_one_good_is_certain_l668_668256

theorem picking_at_least_one_good_is_certain :
  ∀ (products : Finset ℕ), 
  (∀ p ∈ products, p ≥ 1) → products.card = 12 →
  let good_products := (products.filter (λ p, p ≤ 10)).card in
  let defective_products := (products.filter (λ p, p > 10)).card in
  good_products = 10 ∧ defective_products = 2 →
  ∀ (chosen : Finset ℕ), chosen.card = 3 →
  ∃ (g ∈ chosen), g ≤ 10 :=
by 
  intros products h_card h_product_card h_partition chosen h_chosen_card,
  sorry

end picking_at_least_one_good_is_certain_l668_668256


namespace peytons_children_l668_668421

theorem peytons_children (C : ℕ) (juice_per_week : ℕ) (weeks_in_school_year : ℕ) (total_juice_boxes : ℕ) 
  (h1 : juice_per_week = 5) 
  (h2 : weeks_in_school_year = 25) 
  (h3 : total_juice_boxes = 375)
  (h4 : C * (juice_per_week * weeks_in_school_year) = total_juice_boxes) 
  : C = 3 :=
sorry

end peytons_children_l668_668421


namespace divide_garden_l668_668636

/-- A garden is represented by a 4x4 grid where positions indicate tree locations. -/
structure Garden :=
  (match_count : ℕ)
  (tree_positions : List (ℕ × ℕ))
  (enclosed_by : ℕ)

def garden_example : Garden :=
{ match_count := 12,
  tree_positions := [(0, 0), (0, 1), (0, 2), (0, 3),
                     (1, 0), (1, 2), (1, 3), (2, 1),
                     (2, 2), (2, 3), (3, 1), (3, 2)],
  enclosed_by := 16 }

theorem divide_garden (g : Garden) :
  g.match_count = 12 ∧ g.tree_positions.length = 12 ∧ g.enclosed_by = 16 →
  ∃ partitions : list (list (ℕ × ℕ)), 
    partitions.length = 4 ∧
    ∀ part ∈ partitions, part.length = 3 ∧ 
    (∃ (match_lines : ℕ), match_lines = 12) :=
by
  sorry

end divide_garden_l668_668636


namespace ratio_of_daily_wages_l668_668077

-- Definitions for daily wages and conditions
def daily_wage_man : ℝ := sorry
def daily_wage_woman : ℝ := sorry

axiom condition_for_men (M : ℝ) : 16 * M * 25 = 14400
axiom condition_for_women (W : ℝ) : 40 * W * 30 = 21600

-- Theorem statement for the ratio of daily wages
theorem ratio_of_daily_wages 
  (M : ℝ) (W : ℝ) 
  (hM : 16 * M * 25 = 14400) 
  (hW : 40 * W * 30 = 21600) :
  M / W = 2 := 
  sorry

end ratio_of_daily_wages_l668_668077


namespace area_enclosed_by_curves_l668_668816

theorem area_enclosed_by_curves :
  ∫ x in 0..1, (x^2 - x^3) = 1 / 12 :=
by
  sorry

end area_enclosed_by_curves_l668_668816


namespace largest_number_in_sequence_is_48_l668_668709

theorem largest_number_in_sequence_is_48 
    (a_1 a_2 a_3 a_4 a_5 a_6 : ℕ) 
    (h1 : 0 < a_1) 
    (h2 : a_1 < a_2 ∧ a_2 < a_3 ∧ a_3 < a_4 ∧ a_4 < a_5 ∧ a_5 < a_6)
    (h3 : ∃ k_1 k_2 k_3 k_4 k_5 : ℕ, k_1 > 1 ∧ k_2 > 1 ∧ k_3 > 1 ∧ k_4 > 1 ∧ k_5 > 1 ∧ 
          a_2 = k_1 * a_1 ∧ a_3 = k_2 * a_2 ∧ a_4 = k_3 * a_3 ∧ a_5 = k_4 * a_4 ∧ a_6 = k_5 * a_5)
    (h4 : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 79) 
    : a_6 = 48 := 
by 
    sorry

end largest_number_in_sequence_is_48_l668_668709


namespace frequency_count_l668_668631

theorem frequency_count (n : ℕ) (f : ℝ) (h1 : n = 500) (h2 : f = 0.3) : n * f = 150 := 
by
  rw [h1, h2]
  norm_num
  sorry

end frequency_count_l668_668631


namespace inversely_proportional_ratios_l668_668065

theorem inversely_proportional_ratios (x y x₁ x₂ y₁ y₂ : ℝ) (hx_inv : ∀ x y, x * y = 1)
  (hx_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end inversely_proportional_ratios_l668_668065


namespace find_k_l668_668701

noncomputable def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_k
  (k : ℝ)
  (a b c : ℝ × ℝ)
  (ha : a = vec 3 1)
  (hb : b = vec 1 3)
  (hc : c = vec k (-2))
  (h_perp : dot_product (vec (a.1 - c.1) (a.2 - c.2)) (vec (a.1 - b.1) (a.2 - b.2)) = 0) :
  k = 0 :=
sorry

end find_k_l668_668701


namespace cube_volume_l668_668826

theorem cube_volume (d : ℝ) (h : d = 6 * Real.sqrt 2) : 
  ∃ v : ℝ, v = 48 * Real.sqrt 6 := by
  let s := d / Real.sqrt 3
  let volume := s ^ 3
  use volume
  /- Proof of the volume calculation is omitted. -/
  sorry

end cube_volume_l668_668826


namespace sum_of_n_consecutive_integers_l668_668633

theorem sum_of_n_consecutive_integers (n : ℕ) (hn : n > 0) : 
  (∃ k : ℤ, (∑ i in (Finset.range n), k + i) = (n : ℤ)) ↔ odd n := 
sorry

end sum_of_n_consecutive_integers_l668_668633


namespace value_of_b_f_is_monotonically_decreasing_range_of_k_l668_668656

-- Given conditions and function
def f (x : ℝ) := (-2^x + b) / (2^(x + 1) + 2)

-- Condition for odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)

-- Function properties and conditions
variable (b : ℝ)
variable (t : ℝ)
variable (k : ℝ)

-- Proof statements
theorem value_of_b (h_odd : is_odd_function f) : b = 1 := sorry

theorem f_is_monotonically_decreasing (h_b : b = 1) : ∀ x1 x2, x1 < x2 → f x1 > f x2 := sorry

theorem range_of_k 
  (h_b : b = 1) 
  (h_odd : is_odd_function f) 
  (h_monotone : ∀ x1 x2, x1 < x2 → f x1 > f x2) 
  (h_ineq : ∀ t : ℝ, f((t^2 - 2*t)) + f(2*t^2 - k) < 0) : k < -1/3 :=
sorry

end value_of_b_f_is_monotonically_decreasing_range_of_k_l668_668656


namespace age_ordered_youngest_to_oldest_l668_668425

variable (M Q S : Nat)

theorem age_ordered_youngest_to_oldest 
  (h1 : M = Q ∨ S = Q)
  (h2 : M ≥ Q)
  (h3 : S ≤ Q) : S = Q ∧ M > Q :=
by 
  sorry

end age_ordered_youngest_to_oldest_l668_668425


namespace parabola_vertex_l668_668980

theorem parabola_vertex (x : ℝ) : 
  (∀ A B C D : Type,
    (A = y = x^2 + 2) →
    (B = y = x^2 - 2) →
    (C = y = (x + 2)^2) →
    (D = y = (x - 2)^2) →
    true) → (∃ (a : ℝ), y = a * (x + 2)^2 ∧ ⟨-2,0⟩)
  where (y : ℝ)
:=
sorry

end parabola_vertex_l668_668980


namespace b_should_pay_l668_668135

-- Definitions for the number of horses and their duration in months
def horses_of_a := 12
def months_of_a := 8

def horses_of_b := 16
def months_of_b := 9

def horses_of_c := 18
def months_of_c := 6

-- Total rent
def total_rent := 870

-- Shares in horse-months for each person
def share_of_a := horses_of_a * months_of_a
def share_of_b := horses_of_b * months_of_b
def share_of_c := horses_of_c * months_of_c

-- Total share in horse-months
def total_share := share_of_a + share_of_b + share_of_c

-- Fraction for b
def fraction_for_b := share_of_b / total_share

-- Amount b should pay
def amount_for_b := total_rent * fraction_for_b

-- Theorem to verify the amount b should pay
theorem b_should_pay : amount_for_b = 360 := by
  -- The steps of the proof would go here
  sorry

end b_should_pay_l668_668135


namespace circle_containing_n_points_l668_668381

-- Definitions based on given conditions
variables (n : ℕ) (points : finset (ℝ × ℝ))
-- points are either not collinear and any four not concyclic
-- not three points are collinear
def not_collinear (points : finset (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ∈ points → p2 ∈ points → p3 ∈ points →
  ¬ collinear {p1, p2, p3}

-- not four points are concyclic
def not_concyclic (points : finset (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 p4 : (ℝ × ℝ)), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
  ¬ concyclic {p1, p2, p3, p4}

-- Formal theorem statement
theorem circle_containing_n_points
  (h1 : points.card = 2 * n + 3)
  (h2 : not_collinear points)
  (h3 : not_concyclic points) :
  ∃ (c : set (ℝ × ℝ)), ∃ (ps : finset (ℝ × ℝ)), ps ⊆ points ∧ ps.card = 3 ∧ 
  ((λ p, (∃ X, circle(X) container ps)) ≠ ∅ ∧ ((λ p, (circle p ∈ ps).card = n)) :=
  sorry

end circle_containing_n_points_l668_668381


namespace problem_1_problem_2_problem_3_l668_668341

theorem problem_1 (a : ℕ → ℕ) (f : ℕ → ℕ) (b : ℕ → ℕ) : (∀ n, a n = n^2) → (∀ m, f m = m^2) → (b 1 = 1 ∧ b 2 = 2 ∧ b 3 = 3) :=
by
  sorry

theorem problem_2 (a : ℕ → ℕ) (f : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ): 
  (∀ n, a n = 2 * n) → 
  (∀ m, f m = m) → 
  (∀ m, (S m = if m % 2 = 1 then (m^2 - 1) / 4 else m^2 / 4) :=
by
  sorry

theorem problem_3 (a : ℕ → ℕ) (f : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ) (A : ℕ) :
  (∀ n, a n = 2 * n) → 
  (∀ m, f m = A * m^3) → 
  A ∈ ℕ^* → 
  b 3 = 10 → 
  (∀ m, b m = b 1 + (m - 1) * d) → 
  d = 3 :=
by
  sorry

end problem_1_problem_2_problem_3_l668_668341


namespace number_of_truthful_dwarfs_l668_668594

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l668_668594


namespace sum_of_possible_values_of_d_l668_668162

theorem sum_of_possible_values_of_d :
  let d (n : ℕ) := (Nat.log 16 n) + 1
  let lower_bound : ℕ := 256
  let upper_bound : ℕ := 1023
  (∀ n : ℕ, (lower_bound ≤ n ∧ n ≤ upper_bound) → d n = 3) →
  ∑ (d_val : ℕ) in {d n | lower_bound ≤ n ∧ n ≤ upper_bound}, d_val = 3 :=
by
  sorry

end sum_of_possible_values_of_d_l668_668162


namespace at_least_one_bug_has_crawled_more_than_three_l668_668855

noncomputable def insect_distance_proof : Prop :=
  ∀ (A B C : ℝ × ℝ), 
  let r := 2 
  ∃ (x y z : ℝ), 
    (x = (A.1 ^ 2 + A.2 ^ 2) ^ 0.5 ∧ y = (B.1 ^ 2 + B.2 ^ 2) ^ 0.5 ∧ z = (C.1 ^ 2 + C.2 ^ 2) ^ 0.5) ∧
    (x + y ≥ A.1 - B.1 ∧ y + z ≥ B.1 - C.1 ∧ z + x ≥ C.1 - A.1) ∧
    (r = 2) → 
    (x > 3 ∨ y > 3 ∨ z > 3)

theorem at_least_one_bug_has_crawled_more_than_three : insect_distance_proof :=
sorry

end at_least_one_bug_has_crawled_more_than_three_l668_668855


namespace area_of_trapezoid_efba_l668_668955

variable (A B C D E F : Type)
variable [rect : ∀ (p : A) (q : B) (r : C) (s : D), Rectangle p q r s]
variable [point_e : E]
variable [point_f : F]
variable (length_ad : ℝ) (length_bc : ℝ) (area_abcd : ℝ)
variable (ae_ratio bf_ratio : ℝ)

noncomputable def area_trapezoid (ae : ℝ) (bf : ℝ) (length_ab length_bc : ℝ) : ℝ :=
  let area_rect : ℝ := ae * length_ab
  let area_triangle_ae : ℝ := (1 / 2) * ae * (length_ad - ae)
  let area_triangle_bf : ℝ := (1 / 2) * bf * (length_bc - bf)
  area_rect + area_triangle_ae + area_triangle_bf

theorem area_of_trapezoid_efba (length_ab length_bc : ℝ)
  (area_abcd : length_ab * length_bc = 24)
  (ae_ratio : 1 / 4)
  (ae : ae_ratio * length_ad)
  (bf_ratio : 1 / 4)
  (bf : bf_ratio * length_bc) :
  area_trapezoid ae bf length_ab length_bc = 9 :=
by 
  -- The actual proof of the calculation steps will go here
  sorry

end area_of_trapezoid_efba_l668_668955


namespace find_integer_a_l668_668471

theorem find_integer_a (a : ℤ) : (∃ x : ℕ, a * x = 3) ↔ a = 1 ∨ a = 3 :=
by
  sorry

end find_integer_a_l668_668471


namespace trailing_zeros_15_factorial_base_15_l668_668988

theorem trailing_zeros_15_factorial_base_15 : trailing_zeros_base 15 (factorial 15) = 3 := 
by
    -- Definitions and conditions derived from the problem
    def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
    def trailing_zeros_base (b n : ℕ) : ℕ := sorry  -- Placeholder definition

    -- Proof statement translated to Lean
    sorry

end trailing_zeros_15_factorial_base_15_l668_668988


namespace perimeter_increase_ratio_of_sides_l668_668037

def width_increase (a : ℝ) : ℝ := 1.1 * a
def length_increase (b : ℝ) : ℝ := 1.2 * b
def original_perimeter (a b : ℝ) : ℝ := 2 * (a + b)
def new_perimeter (a b : ℝ) : ℝ := 2 * (1.1 * a + 1.2 * b)

theorem perimeter_increase : ∀ a b : ℝ, 
  (a > 0) → (b > 0) → 
  (new_perimeter a b - original_perimeter a b) / (original_perimeter a b) * 100 < 20 := 
by
  sorry

theorem ratio_of_sides (a b : ℝ) (h : new_perimeter a b = 1.18 * original_perimeter a b) : a / b = 1 / 4 := 
by
  sorry

end perimeter_increase_ratio_of_sides_l668_668037


namespace total_bins_correct_l668_668578

def total_bins (soup vegetables pasta : ℝ) : ℝ :=
  soup + vegetables + pasta

theorem total_bins_correct : total_bins 0.12 0.12 0.5 = 0.74 :=
  by
    sorry

end total_bins_correct_l668_668578


namespace common_ratio_geometric_sequence_l668_668002
-- Import the required library

-- Define the necessary hypothesis (conditions)
variable {a : ℕ → ℝ} -- a(n) is the general term of the sequence
variable {q : ℝ} -- q is the common ratio of the sequence

-- Given conditions: The first term and the product of the first 5 terms
axiom a1 : a 1 = 1
axiom T5 : (∏ i in finset.range 5, a i) = 1024

-- Definition of geometric sequence
axiom geometric_sequence : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

-- Main statement to prove
theorem common_ratio_geometric_sequence : q = 2 ∨ q = -2 := sorry

end common_ratio_geometric_sequence_l668_668002


namespace number_of_students_and_top_scorer_l668_668193

-- Definitions from conditions
def distinct_positive (seq : List ℕ) : Prop :=
  (∀ i, 1 ≤ seq[i]) ∧ (seq.nodup)

variable {n : ℕ} {a : ℕ → ℕ}

axiom sum_all_scores : (∑ i in Finset.range n, a i) = 119
axiom sum_smallest_three : (a 0 + a 1 + a 2) = 23
axiom sum_largest_three : (a (n-3) + a (n-2) + a (n-1)) = 49
axiom distinct_scores : distinct_positive (List.ofFn a n)

-- Mathematical equivalent proof statement
theorem number_of_students_and_top_scorer :
  ∃ (n : ℕ) (a_n : ℕ), (dist_scores_and_sum_properties n a) ∧ n = 10 ∧ a (n-1) = 18 :=
sorry

-- We assume a definition for our problem's constraints for collating all properties
def dist_scores_and_sum_properties (n : ℕ) (a : ℕ → ℕ) : Prop :=
  distinct_positive (List.ofFn a n) ∧
  (∑ i in Finset.range n, a i = 119) ∧
  (a 0 + a 1 + a 2 = 23) ∧
  (a (n-3) + a (n-2) + a (n-1) = 49)

end number_of_students_and_top_scorer_l668_668193


namespace scheduling_subjects_l668_668707

theorem scheduling_subjects (periods subjects : ℕ)
  (h_periods : periods = 7)
  (h_subjects : subjects = 4) :
  ∃ (ways : ℕ), ways = Nat.choose periods subjects * subjects.factorial ∧ ways = 840 :=
by
  use Nat.choose 7 4 * 4.factorial
  split
  {
    rw [h_periods, h_subjects],
    norm_num
  }
  {
    rw [h_subjects],
    norm_num
  }
  sorry

end scheduling_subjects_l668_668707


namespace minimum_score_to_win_l668_668338

namespace CompetitionPoints

-- Define points awarded for each position
def points_first : ℕ := 5
def points_second : ℕ := 3
def points_third : ℕ := 1

-- Define the number of competitions
def competitions : ℕ := 3

-- Total points in one competition
def total_points_one_competition : ℕ := points_first + points_second + points_third

-- Total points in all competitions
def total_points_all_competitions : ℕ := total_points_one_competition * competitions

theorem minimum_score_to_win : ∃ m : ℕ, m = 13 ∧ (∀ s : ℕ, s < 13 → ¬ ∃ c1 c2 c3 : ℕ, 
  c1 ≤ competitions ∧ c2 ≤ competitions ∧ c3 ≤ competitions ∧ 
  ((c1 * points_first) + (c2 * points_second) + (c3 * points_third)) = s) :=
by {
  sorry
}

end CompetitionPoints

end minimum_score_to_win_l668_668338


namespace stone_hitting_ground_time_l668_668441

noncomputable def equation (s : ℝ) : ℝ := -4.5 * s^2 - 12 * s + 48

theorem stone_hitting_ground_time :
  ∃ s : ℝ, equation s = 0 ∧ s = (-8 + 16 * Real.sqrt 7) / 6 :=
by
  sorry

end stone_hitting_ground_time_l668_668441


namespace age_of_jerry_l668_668414

variable (M J : ℕ)

theorem age_of_jerry (h1 : M = 2 * J - 5) (h2 : M = 19) : J = 12 := by
  sorry

end age_of_jerry_l668_668414


namespace dwarfs_truthful_count_l668_668588

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l668_668588


namespace Tim_eats_91_pickle_slices_l668_668796

theorem Tim_eats_91_pickle_slices :
  let Sammy := 25
  let Tammy := 3 * Sammy
  let Ron := Tammy - 0.15 * Tammy
  let Amy := Sammy + 0.50 * Sammy
  let CombinedTotal := Ron + Amy
  let Tim := CombinedTotal - 0.10 * CombinedTotal
  Tim = 91 :=
by
  admit

end Tim_eats_91_pickle_slices_l668_668796


namespace regular_polygon_sides_l668_668182

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l668_668182


namespace lady_speed_in_square_field_l668_668815

noncomputable def sqrt (x : ℝ) : ℝ := sorry

theorem lady_speed_in_square_field :
  let area := 7201;
  let time := 6.0008333333333335;
  let side := sqrt area;
  let diagonal := sqrt (2 * side^2);
  let distance_km := diagonal / 1000;
  let speed := distance_km / time
  in speed = 0.02000138888888889 :=
by
  sorry

end lady_speed_in_square_field_l668_668815


namespace QO_perpendicular_to_BC_l668_668764

open EuclideanGeometry

variables (A B C M N Q P O : Point)
variables (hABC : Triangle A B C)
variables (hM : Median A M B C)
variables (hN : AngleBisector A N B C)
variables (hQ_perp : Perpendicular N Q (Line.mk N A))
variables (hQ_meet : Meet Q (Line.mk N A) (Line.mk M A))
variables (hP_perp : Perpendicular N P (Line.mk N A))
variables (hP_meet : Meet P (Line.mk N A) (Line.mk B A))
variables (hO_perp : Perpendicular P O (Line.mk P B))
variables (hO_meet : Meet O (Line.mk P B) (LineProduced.mk A N))

theorem QO_perpendicular_to_BC :
  Perpendicular (Line.mk Q O) (Line.mk B C) :=
sorry

end QO_perpendicular_to_BC_l668_668764


namespace sequence_formula_l668_668013

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n →  1 / a (n + 1) = 1 / a n + 1) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by {
  sorry
}

end sequence_formula_l668_668013


namespace volume_of_pyramid_l668_668072

-- We define the relevant parameters and conditions:
variables (SA SB SC SD A B C D : Point)
           (rhombus_height : ℝ) (sphere_radius : ℝ)
           (center_to_line_AC_dist : ℝ)
           (AB_length : ℝ)

-- Specify the conditions based on the problem:
def pyramid_base_is_rhombus (A B C D : Point) : Prop :=
  -- Define properties of rhombus ABCD with acute angle at A
  rhombus A B C D ∧ angle A < 90

def height_of_rhombus := rhombus_height = 4
def intersection_projection := (intersection_diagonals A B C D) = orthogonal_projection S (plane A B C D)
def sphere_touches_faces := sphere_of_radius sphere_radius ∧
                           sphere_touches_planes S A B C A B C D
def distance_center_to_AC := center_to_line_AC_dist = (2 * sqrt 2 / 3) * AB_length
def side_length_AB := AB_length

-- Prove the volume of the pyramid SABCD equals to 8 * sqrt 2
theorem volume_of_pyramid 
  (pyramid_base_is_rhombus A B C D)
  (height_of_rhombus)
  (intersection_projection)
  (sphere_touches_faces)
  (distance_center_to_AC) :
  volume (pyramid S A B C D) = 8 * sqrt 2 :=
sorry

end volume_of_pyramid_l668_668072


namespace solution_eq1_solution_eq2_l668_668802

-- Equation (1)
theorem solution_eq1 (x : ℝ) :
    3 * x * (x - 1) = 2 * (x - 1) ↔ x = 1 ∨ x = 2 / 3 :=
by sorry

-- Equation (2)
theorem solution_eq2 (x : ℝ) :
    x^2 - 6*x + 6 = 0 ↔ x = 3 + real.sqrt 3 ∨ x = 3 - real.sqrt 3 :=
by sorry

end solution_eq1_solution_eq2_l668_668802


namespace k_h_neg3_eq_15_l668_668392

-- Define the function h
def h (x : Int) : Int := 5 * x^2 - 12

-- Given: k(h(3)) = 15
axiom k_h3_eq_15 : k (h 3) = 15

-- Prove that k(h(-3)) = 15
theorem k_h_neg3_eq_15 : k (h (-3)) = 15 :=
by
  have h3 : h 3 = 33 := by rfl
  have h_neg3 : h (-3) = 33 := by rfl
  rw [h_neg3, k_h3_eq_15]
  sorry -- placeholder to indicate further steps are needed to complete the proof

end k_h_neg3_eq_15_l668_668392


namespace cube_root_floor_equality_l668_668785

theorem cube_root_floor_equality (n : ℕ) : 
  (⌊(n : ℝ)^(1/3) + (n+1 : ℝ)^(1/3)⌋ : ℝ) = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end cube_root_floor_equality_l668_668785


namespace determine_m_l668_668691

def f (x : ℝ) : ℝ := |2 * x - 1|

def f_n : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := f ∘ f_n n

def g (m : ℕ) (x : ℝ) : ℝ := f_n m x - x

theorem determine_m : ∃ m : ℕ, ( ∀ x : ℝ, g m x = 0 → (x = 1/9 ∨ x = 1/7 ∨ x = 3/9 ∨ x = 3/7 ∨ x = 5/9 ∨ x = 5/7 ∨ x = 7/9 ∨ x = 1)) ∧ m = 4 :=
sorry

end determine_m_l668_668691


namespace solve_equation_l668_668060

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) :
  (x / (x - 1) - 2 / x = 1) ↔ x = 2 :=
sorry

end solve_equation_l668_668060


namespace find_standard_equation_of_ellipse_l668_668248

noncomputable def ellipse_equation (a c b : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∨ (y^2 / a^2 + x^2 / b^2 = 1)

theorem find_standard_equation_of_ellipse (h1 : 2 * a = 12) (h2 : c / a = 1 / 3) :
  ellipse_equation 6 2 4 :=
by
  -- We are proving that given the conditions, the standard equation of the ellipse is as stated
  sorry

end find_standard_equation_of_ellipse_l668_668248


namespace spring_unextended_state_l668_668097

theorem spring_unextended_state {m k g a b t: ℝ} (h1: mg/k = 1) (h2: b = 1 - 2a):
  (∀ t, m ≠ 0 ∧ k ≠ 0 ∧ g ≠ 0 → (a * exp (-2 * t) + b * exp (-t) + m * g / k = 0) → 
       interval (1 + sqrt 3 / 2) 2 a) := 
begin
  sorry,
end

end spring_unextended_state_l668_668097


namespace find_ratio_max_value_sin_cos_l668_668770

-- Definitions and conditions
def x : ℝ := sorry
def y : ℝ := sorry
def k := (Real.tan (9 * Real.pi / 20) * Real.cos (Real.pi / 5) - Real.sin (Real.pi / 5)) / (Real.cos (Real.pi / 5) + Real.tan (9 * Real.pi / 20) * Real.sin (Real.pi / 5))

-- Non-zero conditions
axiom x_nonzero : x ≠ 0
axiom y_nonzero : y ≠ 0

-- Given equation
axiom given_equation :
  (x * Real.sin (Real.pi / 5) + y * Real.cos (Real.pi / 5)) / (x * Real.cos (Real.pi / 5) - y * Real.sin (Real.pi / 5)) = Real.tan (9 * Real.pi / 20)

-- First problem: value of y / x
theorem find_ratio : y / x = k := 
sorry

-- Second problem context
variables (A B C : ℝ)
axiom triangle_ABC : ∀ (A B C : ℝ), A + B + C = Real.pi
axiom tan_C : Real.tan C = y / x

-- Second problem: maximum value of sin 2A + 2 cos B
theorem max_value_sin_cos : ∃ M, M = Real.sin (2 * A) + 2 * Real.cos B := 
sorry

end find_ratio_max_value_sin_cos_l668_668770


namespace triangle_side_centroid_distance_l668_668742

theorem triangle_side_centroid_distance (α β γ λ μ ν : ℝ) (h_triangle: is_triangle α β γ) :
  (λ^2 + μ^2 + ν^2 = (1/3)^2 * (|α^2 + β^2 + γ^2|^2)) ∧ 
  (λ^2 + μ^2 + ν^2 > 0) → 
  (λ^2 + μ^2 + ν^2 ≠ 0) → 
  (λ ≠ 0) → 
  (μ ≠ 0) → 
  (ν ≠ 0) → 
  ((α^2 + β^2 + γ^2) / (λ^2 + μ^2 + ν^2) = 6) :=
begin
  sorry
end

end triangle_side_centroid_distance_l668_668742


namespace smallest_number_hides_range_121_to_2121_l668_668405

def hides (A B : ℕ) : Prop := ∃ (subseq : List ℕ), List.foldr (λ x acc, 10 * acc + x) 0 subseq = B ∧ List.isSubsequenceOf subseq (List.ofFn (λ i, (A / 10^i) % 10) (Nat.digits 10 A).length)

theorem smallest_number_hides_range_121_to_2121 : ∃ (A : ℕ), 
  A = 1201345678921 ∧
  ∀ B ∈ (set.range (121, 2121 + 1)), hides A B :=
sorry

end smallest_number_hides_range_121_to_2121_l668_668405


namespace limit_solution_l668_668559

noncomputable def limit_problem :=
  limit (λ x : ℝ, (6 - 5 / cos x) ^ (cot x)^2) 0 (𝓝 0) = exp (-5 / 2)

theorem limit_solution : limit_problem := 
  sorry

end limit_solution_l668_668559


namespace no_zeros_of_g_l668_668291

variable (f : ℝ → ℝ)
variable (f' f'' : ℝ → ℝ)
variable [Differentiable ℝ f]
variable [Differentiable ℝ f']
variable [∀ x, DifferentiableAt ℝ f'' x]

theorem no_zeros_of_g 
  (h : ∀ x ≠ 0, f'' x + f x / x > 0) :
  ∀ x ≠ 0, (f x + 1 / x) ≠ 0 := 
  sorry

end no_zeros_of_g_l668_668291


namespace vasya_purchase_l668_668928

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668928


namespace problem1_problem2_l668_668310

-- Problem 1: Prove that the solution to f(x) <= 0 for a = -2 is [1, +∞)
theorem problem1 (x : ℝ) : (|x + 2| - 2 * x - 1 ≤ 0) ↔ (1 ≤ x) := sorry

-- Problem 2: Prove that the range of m such that there exists x ∈ ℝ satisfying f(x) + |x + 2| ≤ m for a = 1 is m ≥ 0
theorem problem2 (m : ℝ) : (∃ x : ℝ, |x - 1| - 2 * x - 1 + |x + 2| ≤ m) ↔ (0 ≤ m) := sorry

end problem1_problem2_l668_668310


namespace triangle_angle_A_triangle_area_l668_668739

-- Problem 1: Prove angle A = 45° given the conditions
theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 3) (h2 : C = 60) (h3 : a = sqrt 6) :
  A = 45 := sorry

-- Problem 2: Prove the area of the triangle equals 3√3 / 2 given the conditions
theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 3) (h2 : C = 60) (h3 : a = 2 * b) :
  (1/2 * a * b * sin (C * Real.pi / 180)) = 3 * sqrt 3 / 2 := sorry

end triangle_angle_A_triangle_area_l668_668739


namespace arrangement_books_l668_668424

/--
Theorem: The number of ways to arrange eleven books on a shelf, consisting of three Arabic books, four German books, and four Spanish books, where the Arabic books must stay together, is 2,177,280.
-/
theorem arrangement_books (arabic german spanish : ℕ) (h_arabic : arabic = 3) (h_german : german = 4) (h_spanish : spanish = 4) : 
  (∃ arrangement : ℕ, arrangement = (9.factorial * arabic.factorial) ∧ arrangement = 2177280) :=
by
  use (9.factorial * arabic.factorial)
  split
  · simp [h_arabic, h_german, h_spanish]
  · simp [h_arabic]
  done

end arrangement_books_l668_668424


namespace number_of_students_and_top_scorer_l668_668194

-- Definitions from conditions
def distinct_positive (seq : List ℕ) : Prop :=
  (∀ i, 1 ≤ seq[i]) ∧ (seq.nodup)

variable {n : ℕ} {a : ℕ → ℕ}

axiom sum_all_scores : (∑ i in Finset.range n, a i) = 119
axiom sum_smallest_three : (a 0 + a 1 + a 2) = 23
axiom sum_largest_three : (a (n-3) + a (n-2) + a (n-1)) = 49
axiom distinct_scores : distinct_positive (List.ofFn a n)

-- Mathematical equivalent proof statement
theorem number_of_students_and_top_scorer :
  ∃ (n : ℕ) (a_n : ℕ), (dist_scores_and_sum_properties n a) ∧ n = 10 ∧ a (n-1) = 18 :=
sorry

-- We assume a definition for our problem's constraints for collating all properties
def dist_scores_and_sum_properties (n : ℕ) (a : ℕ → ℕ) : Prop :=
  distinct_positive (List.ofFn a n) ∧
  (∑ i in Finset.range n, a i = 119) ∧
  (a 0 + a 1 + a 2 = 23) ∧
  (a (n-3) + a (n-2) + a (n-1) = 49)

end number_of_students_and_top_scorer_l668_668194


namespace complex_no_first_quadrant_l668_668075

open Complex

theorem complex_no_first_quadrant (m : ℝ) : 
  (let z := (m - 2 * I) / (1 + 2 * I) in (z.re > 0 ∧ z.im > 0) → False) :=
by
  sorry

end complex_no_first_quadrant_l668_668075


namespace sufficient_but_not_necessary_condition_for_intersection_l668_668714

def A (m : ℕ) : set ℕ := {1, m^2}
def B : set ℕ := {2, 4}

theorem sufficient_but_not_necessary_condition_for_intersection (m : ℕ) : 
  (A m ∩ B = {4} → m = 2) ∧ (A 2 ∩ B = {4}) :=
by 
  sorry

end sufficient_but_not_necessary_condition_for_intersection_l668_668714


namespace find_numbers_l668_668320

theorem find_numbers :
  ∃ a d : ℝ, 
    ((a - d) + a + (a + d) = 12) ∧ 
    ((a - d) * a * (a + d) = 48) ∧
    (a = 4) ∧ 
    (d = -2) ∧ 
    (a - d = 6) ∧ 
    (a + d = 2) :=
by
  sorry

end find_numbers_l668_668320


namespace exist_consecutive_days_20_games_l668_668962

theorem exist_consecutive_days_20_games 
  (a : ℕ → ℕ)
  (h_daily : ∀ n, a (n + 1) - a n ≥ 1)
  (h_weekly : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ a j - a i = 20 := by 
  sorry

end exist_consecutive_days_20_games_l668_668962


namespace alpha_norm_is_two_l668_668387

-- Define what it means for two numbers to be conjugate complex
def conjugate_complex (α β : ℂ) : Prop :=
  β = complex.conj α

-- Given conditions
variables (α β : ℂ)
hypothesis (h_conj: conjugate_complex α β)
hypothesis (h_diff_norm: abs (α - β) = 2)
hypothesis (h_diff_real: abs (α - β) ∈ ℝ)

-- Prove the main question
theorem alpha_norm_is_two : abs α = 2 :=
by
  sorry

end alpha_norm_is_two_l668_668387


namespace sum_of_numbers_l668_668138

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149)
  (h2 : ab + bc + ca = 70) : 
  a + b + c = 17 :=
sorry

end sum_of_numbers_l668_668138


namespace geometric_series_sum_l668_668476

theorem geometric_series_sum :
  (∑ k in Finset.range (2023 + 1), 5^k) = (5^2024 - 1) / 4 :=
by
  sorry

end geometric_series_sum_l668_668476


namespace find_A_l668_668018

theorem find_A (A M C : ℕ) (h1 : A ≤ 9) (h2 : M ≤ 9) (h3 : C ≤ 9)
    (h4 : (100 * A + 10 * M + C) * 2 * (A + M + C + 1) = 4010) : A = 4 := 
begin
    sorry
end

end find_A_l668_668018


namespace bruce_bhishma_meet_time_l668_668557

theorem bruce_bhishma_meet_time :
  ∀ (track_length : ℕ) (speed_bruce speed_bhishma : ℕ),
    track_length = 600 → speed_bruce = 30 → speed_bhishma = 20 →
    speed_bruce > speed_bhishma →
    (track_length / (speed_bruce - speed_bhishma)) = 60 :=
by
  intros track_length speed_bruce speed_bhishma h_track h_bruce h_bhish h_compare
  rw [h_track, h_bruce, h_bhish]
  norm_num
  sorry  -- Placeholder for the actual proof

end bruce_bhishma_meet_time_l668_668557


namespace negation_P_l668_668452

-- Define the original proposition P
def P (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- State the negation of P
theorem negation_P : ∀ (a b : ℝ), (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end negation_P_l668_668452


namespace ellipse_standard_eq_exists_fixed_point_l668_668654

noncomputable def standard_equation_of_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

noncomputable def line_intersects_ellipse (k m a b : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m ∧ x^2 / a^2 + y^2 / b^2 = 1

noncomputable def circle_with_diameter_AB (A B D : (ℝ × ℝ)) : Prop :=
  let (x₁, y₁) := A in
  let (x₂, y₂) := B in
  let (x₃, y₃) := D in
  let k_AB := if x₁ ≠ x₂ then (y₂ - y₁) / (x₂ - x₁) else 0 in
  let k_AD := if x₁ ≠ x₃ then (y₃ - y₁) / (x₃ - x₁) else 0 in
  let k_BD := if x₂ ≠ x₃ then (y₃ - y₂) / (x₃ - x₂) else 0 in
  k_AD * k_BD = -1

theorem ellipse_standard_eq_exists_fixed_point :
  ∃ (a b : ℝ), 
      (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) 
      ∧ (a + sqrt (a^2 - b^2) = 3) 
      ∧ (a - sqrt (a^2 - b^2) = 1)
      ∧ let f := (1/2 : ℝ) in 
          ∀ k m : ℝ, ∀ (A B : (ℝ × ℝ)),
            (line_intersects_ellipse k m a b A.fst A.snd ∧ line_intersects_ellipse k m a b B.fst B.snd)
            → circle_with_diameter_AB A B (2, 0) 
            → (line_intersects_ellipse k m a b (f,0) ) :=
begin
  sorry
end

end ellipse_standard_eq_exists_fixed_point_l668_668654


namespace value_of_x_when_z_is_32_l668_668511

variables {x y z k : ℝ}
variable (m n : ℝ)

def directly_proportional (x y : ℝ) (m : ℝ) := x = m * y^2
def inversely_proportional (y z : ℝ) (n : ℝ) := y = n / z^2

-- Our main proof goal
theorem value_of_x_when_z_is_32 (h1 : directly_proportional x y m) 
  (h2 : inversely_proportional y z n) (h3 : z = 8) (hx : x = 5) : 
  x = 5 / 256 :=
by
  let k := x * z^4
  have k_value : k = 20480 := by sorry
  have x_new : x = k / z^4 := by sorry
  have z_new : z = 32 := by sorry
  have x_final : x = 5 / 256 := by sorry
  exact x_final

end value_of_x_when_z_is_32_l668_668511


namespace smallest_integer_of_lcm_gcd_l668_668500

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l668_668500


namespace distinct_digit_sum_equation_l668_668995

theorem distinct_digit_sum_equation :
  ∃ (F O R T Y S I X : ℕ), 
    F ≠ O ∧ F ≠ R ∧ F ≠ T ∧ F ≠ Y ∧ F ≠ S ∧ F ≠ I ∧ F ≠ X ∧ 
    O ≠ R ∧ O ≠ T ∧ O ≠ Y ∧ O ≠ S ∧ O ≠ I ∧ O ≠ X ∧ 
    R ≠ T ∧ R ≠ Y ∧ R ≠ S ∧ R ≠ I ∧ R ≠ X ∧ 
    T ≠ Y ∧ T ≠ S ∧ T ≠ I ∧ T ≠ X ∧ 
    Y ≠ S ∧ Y ≠ I ∧ Y ≠ X ∧ 
    S ≠ I ∧ S ≠ X ∧ 
    I ≠ X ∧ 
    FORTY = 10000 * F + 1000 * O + 100 * R + 10 * T + Y ∧ 
    TEN = 100 * T + 10 * E + N ∧ 
    SIXTY = 10000 * S + 1000 * I + 100 * X + 10 * T + Y ∧ 
    FORTY + TEN + TEN = SIXTY ∧ 
    SIXTY = 31486 :=
sorry

end distinct_digit_sum_equation_l668_668995


namespace probability_A_will_receive_2_awards_l668_668092

def classes := Fin 4
def awards := 8

-- The number of ways to distribute 4 remaining awards to 4 classes
noncomputable def total_distributions : ℕ :=
  Nat.choose (awards - 4 + 4 - 1) (4 - 1)

-- The number of ways when class A receives exactly 2 awards
noncomputable def favorable_distributions : ℕ :=
  Nat.choose (2 + 3 - 1) (4 - 1)

-- The probability that class A receives exactly 2 out of 8 awards
noncomputable def probability_A_receives_2_awards : ℚ :=
  favorable_distributions / total_distributions

theorem probability_A_will_receive_2_awards :
  probability_A_receives_2_awards = 2 / 7 := by
  sorry

end probability_A_will_receive_2_awards_l668_668092


namespace cos_double_angle_l668_668331

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_l668_668331


namespace value_of_f_composition_l668_668761

noncomputable def f (x : ℝ) : ℝ :=
if x % 2 < 0 then
  -4 * x^2 + 1
else if x % 2 < 1 then
  x + 7 / 4
else
  f (x - 2 * real.floor(0.5 * (x + 1)))

-- Lean proof problem statement
theorem value_of_f_composition : f (f (3 / 2)) = 7 / 4 := 
by 
  -- This will be the place for formal proof steps
  sorry

end value_of_f_composition_l668_668761


namespace LykaSavings_l668_668412

-- Define the given values and the properties
def totalCost : ℝ := 160
def amountWithLyka : ℝ := 40
def averageWeeksPerMonth : ℝ := 4.33
def numberOfMonths : ℝ := 2

-- Define the remaining amount Lyka needs
def remainingAmount : ℝ := totalCost - amountWithLyka

-- Define the number of weeks in the saving period
def numberOfWeeks : ℝ := numberOfMonths * averageWeeksPerMonth

-- Define the weekly saving amount
def weeklySaving : ℝ := remainingAmount / numberOfWeeks

-- State the theorem to be proved
theorem LykaSavings :  weeklySaving ≈ 13.86 :=
by
  -- Proof steps (omitted)
  sorry

end LykaSavings_l668_668412


namespace ratio_x_share_to_total_profit_l668_668972

-- Definitions based on the conditions
def total_profit : ℝ := 500
def difference_in_shares : ℝ := 100

-- Given equalities based on the conditions
def x_share (a b : ℝ) : ℝ := (a / (a + b)) * total_profit
def y_share (a b : ℝ) : ℝ := (b / (a + b)) * total_profit

-- The statement to prove
theorem ratio_x_share_to_total_profit (a b : ℝ) (h : 500 * (a - b) / (a + b) = 100) : 
  (x_share a b) / total_profit = 3 / 5 :=
sorry

end ratio_x_share_to_total_profit_l668_668972


namespace centroid_vector_sum_zero_l668_668765

universe u

structure Point (α : Type u) :=
(x y : α)

variables {α : Type u} [AddGroup α]

/-- Definition of vector from point a to point b -/ 
def vector (a b : Point α) : Point α :=
⟨b.x - a.x, b.y - a.y⟩

/-- The centroid of a triangle ABC is the intersection of its medians -/
def centroid (A B C : Point α) : Point α :=
⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

/-- Define Midpoints: midpoints of triangle sides -/
def midpoint (A B : Point α) : Point α :=
⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

variables (A B C : Point α) 
noncomputable def M := centroid A B C
noncomputable def A1 := midpoint B C
noncomputable def B1 := midpoint A C
noncomputable def C1 := midpoint A B

noncomputable def vec_MA := vector M A
noncomputable def vec_MB := vector M B
noncomputable def vec_MC := vector M C

theorem centroid_vector_sum_zero :
  vec_MA + vec_MB + vec_MC = ⟨0, 0⟩ :=
sorry

end centroid_vector_sum_zero_l668_668765


namespace digits_in_first_2002_even_numbers_l668_668495

theorem digits_in_first_2002_even_numbers :
  let one_digit := 4
  let two_digit := ((98 - 10) / 2 + 1) * 2
  let three_digit := ((998 - 100) / 2 + 1) * 3
  let four_digit := ((4004 - 1000) / 2 + 1) * 4
  one_digit + two_digit + three_digit + four_digit = 7456 := by
  -- Let us define each segment and compute their total contribution
  have h_one : one_digit = 4 := rfl
  have h_two : two_digit = (((98 - 10) / 2 + 1) * 2) := by norm_num
  have h_three : three_digit = (((998 - 100) / 2 + 1) * 3) := by norm_num
  have h_four : four_digit = (((4004 - 1000) / 2 + 1) * 4) := by norm_num
  have total_sum : one_digit + two_digit + three_digit + four_digit = 
    4 + 90 + 1350 + 6012 := by norm_num
  show one_digit + two_digit + three_digit + four_digit = 7456 from
  total_sum ▸ by norm_num

end digits_in_first_2002_even_numbers_l668_668495


namespace sin_shift_right_pi_l668_668104

theorem sin_shift_right_pi (x : ℝ) : sin (x - π) = sin x := by
  sorry

end sin_shift_right_pi_l668_668104


namespace all_columns_covered_at_362_l668_668973

def S : ℕ → ℕ
| 0     := 1
| (n+1) := S n + (2 * n + 1)

def column (n : ℕ) : ℕ := (S n - 1) % 10 + 1

theorem all_columns_covered_at_362 :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → ∃ n ≤ 20, column n = k  :=
by {
  sorry
}

end all_columns_covered_at_362_l668_668973


namespace vasya_purchase_l668_668930

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668930


namespace toys_lost_l668_668985

theorem toys_lost (initial_toys found_in_closet total_after_finding : ℕ) 
  (h1 : initial_toys = 40) 
  (h2 : found_in_closet = 9) 
  (h3 : total_after_finding = 43) : 
  initial_toys - (total_after_finding - found_in_closet) = 9 :=
by 
  sorry

end toys_lost_l668_668985


namespace four_bags_remainder_l668_668235

theorem four_bags_remainder (n : ℤ) (hn : n % 11 = 5) : (4 * n) % 11 = 9 := 
by
  sorry

end four_bags_remainder_l668_668235


namespace range_of_t_l668_668278

noncomputable def S (n : ℕ) : ℝ := n^2 + 2 * n
noncomputable def a (n : ℕ) : ℝ := 2 * n + 1
noncomputable def b (n : ℕ) : ℝ := (2 * n + 1) * (2 * n + 3) * Real.cos ((n + 1) * Real.pi)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range(n+1), (b i)

theorem range_of_t (t : ℝ) : (∀ n : ℕ, 0 < n → (T n) ≥ t * n^2) ↔ t ≤ -5 := sorry

end range_of_t_l668_668278


namespace probability_two_faces_no_faces_l668_668532

theorem probability_two_faces_no_faces :
  let side_length := 5
  let total_cubes := side_length ^ 3
  let painted_faces := 2 * (side_length ^ 2)
  let two_painted_faces := 16
  let no_painted_faces := total_cubes - painted_faces + two_painted_faces
  (two_painted_faces = 16) →
  (no_painted_faces = 91) →
  -- Total ways to choose 2 cubes from 125
  let total_ways := (total_cubes * (total_cubes - 1)) / 2
  -- Ways to choose 1 cube with 2 painted faces and 1 with no painted faces
  let successful_ways := two_painted_faces * no_painted_faces
  (successful_ways = 1456) →
  (total_ways = 7750) →
  -- The desired probability
  let probability := successful_ways / (total_ways : ℝ)
  probability = 4 / 21 :=
by
  intros side_length total_cubes painted_faces two_painted_faces no_painted_faces h1 h2 total_ways successful_ways h3 h4 probability
  sorry

end probability_two_faces_no_faces_l668_668532


namespace share_cost_equally_l668_668568

variable (P Q R : ℝ) (hR : R = 3 * Q - 2 * P) (hP_lt_Q : P < Q)

theorem share_cost_equally :
  (Javier_should_pay : ℝ) := 
  ∃ (Javier_should_pay : ℝ), Javier_should_pay = (2 * Q - P) / 2 :=
sorry

end share_cost_equally_l668_668568


namespace inverse_prop_l668_668040

theorem inverse_prop (a b : ℝ) : (a > b) → (|a| > |b|) :=
sorry

end inverse_prop_l668_668040


namespace mangoes_purchased_l668_668552

-- Define the known values
def grapes_kg : ℕ := 11
def grape_rate_per_kg : ℕ := 98
def total_payment : ℕ := 1428
def mango_rate_per_kg : ℕ := 50

-- Define the theorem to be proved
theorem mangoes_purchased :
  let cost_of_grapes := grapes_kg * grape_rate_per_kg,
      total_amount_spent_on_mangoes := total_payment - cost_of_grapes,
      mangoes_kg := total_amount_spent_on_mangoes / mango_rate_per_kg
  in mangoes_kg = 7 := 
by {
  let cost_of_grapes := grapes_kg * grape_rate_per_kg,
  let total_amount_spent_on_mangoes := total_payment - cost_of_grapes,
  let mangoes_kg := total_amount_spent_on_mangoes / mango_rate_per_kg,
  have h: mangoes_kg = 7,
  { calc
      mangoes_kg = (1428 - (11 * 98)) / 50 : by rfl
              ... = 7 : by norm_num },
  exact h,
  sorry
}

end mangoes_purchased_l668_668552


namespace find_values_of_symbols_l668_668140

theorem find_values_of_symbols (a b : ℕ) (h1 : a + b + b = 55) (h2 : a + b = 40) : b = 15 ∧ a = 25 :=
  by
    sorry

end find_values_of_symbols_l668_668140


namespace length_of_BC_l668_668367

open Real

-- Definitions of the conditions
variables {AB CA BC : ℝ}
variables (A B C : Type)

def right_triangle (A B C : Type) := true

def tan_B := CA / AB = 4 / 3
def given_AB := AB = 3
def pythagorean_theorem := BC = sqrt (AB^2 + CA^2)

-- Statement to prove
theorem length_of_BC (h1 : right_triangle A B C) (h2 : tan_B) (h3 : given_AB) : BC = 5 :=
sorry

end length_of_BC_l668_668367


namespace probability_half_of_26_parts_l668_668836

noncomputable def probability_of_half_top_grade_parts 
  (p : ℝ) (n : ℕ) (m : ℕ) : ℝ :=
  let q := 1 - p in
  let np := n * p in
  let npq := np * q in
  let std_dev := Real.sqrt npq in
  let normalized_diff := (m - np) / std_dev in
  Real.ofRat (StdNormalDistribution.cdf normalized_diff) / std_dev

theorem probability_half_of_26_parts :
  probability_of_half_top_grade_parts 0.4 26 13 ≈ 0.093 :=
by sorry

end probability_half_of_26_parts_l668_668836


namespace sum_of_arithmetic_progressions_l668_668064

variable {α : Type*} [AddCommMonoid α] [MulAction ℤ α]

def arithmetic_prog (a d : α) (n : ℕ) : α :=
  a + (n - 1) • d

theorem sum_of_arithmetic_progressions
  (a₁ b₁ c₁ d_a d_b d_c : α)
  (h₁ : a₁ + b₁ + c₁ = 0)
  (h₂ : (a₁ + d_a) + (b₁ + d_b) + (c₁ + d_c) = 1) :
  arithmetic_prog a₁ d_a 2014 + arithmetic_prog b₁ d_b 2014 + arithmetic_prog c₁ d_c 2014 = 2013 := 
sorry

end sum_of_arithmetic_progressions_l668_668064


namespace initial_sugar_weight_l668_668834

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end initial_sugar_weight_l668_668834


namespace vasya_no_purchase_days_l668_668948

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668948


namespace prove_race_result_l668_668894

-- Define the positions as elements of finite set {1, 2, 3, 4}
def positions := {1, 2, 3, 4}

-- Assume the rankings are injective mappings from the participants to positions
variable {A B C D : ℕ}

-- Conditions translated into Lean
def condition1 := C = 4 → A = 2
def condition2 := A = 2 → B = 1
def condition3 := (A < C ∧ B < C) ∨ (A > C ∧ B > C)
def condition4 := (A > D ∧ B < D) ∨ (A < D ∧ B > D)

-- Combining all conditions into one
def all_conditions := 
  (C ∈ positions) ∧ 
  (D ∈ positions) ∧ 
  C ≠ D ∧
  (A ∈ positions) ∧ 
  (B ∈ positions) ∧ 
  (A ≠ B) ∧ 
  (A ≠ C) ∧ 
  (A ≠ D) ∧
  (B ≠ C) ∧ 
  (B ≠ D) ∧
  condition1 ∧ 
  condition2 ∧ 
  condition3 ∧ 
  condition4

-- The main theorem: under the given conditions, the ranking is {B, B, C, D} = 4, 2, 1, 3
theorem prove_race_result : all_conditions → (B = 4 ∧ B = 2 ∧ C = 1 ∧ D = 3) := 
by sorry

end prove_race_result_l668_668894


namespace total_rock_needed_l668_668161

theorem total_rock_needed (a b : ℕ) (h₁ : a = 8) (h₂ : b = 8) : a + b = 16 :=
by
  sorry

end total_rock_needed_l668_668161


namespace number_of_arrangements_of_six_students_l668_668101

/-- A and B cannot stand together -/
noncomputable def arrangements_A_B_not_together (n: ℕ) (A B: ℕ) : ℕ :=
  if n = 6 then 480 else 0

theorem number_of_arrangements_of_six_students :
  arrangements_A_B_not_together 6 1 2 = 480 :=
sorry

end number_of_arrangements_of_six_students_l668_668101


namespace dwarfs_truthful_count_l668_668591

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l668_668591


namespace segment_in_regular_pentagon_l668_668749

/--
  Problem: 
  Prove that it is impossible to place a segment inside a regular pentagon
  so that it is seen from all vertices at the same angle.
-/
theorem segment_in_regular_pentagon (P : Type) [regular_pentagon P] :
  ¬ ∃ (XY : segment P), (∀ (v : P), ∃ (angle : angle P), seen_from v XY = angle) :=
by
  -- Proof goes here
  sorry

end segment_in_regular_pentagon_l668_668749


namespace continuous_extension_possible_l668_668015

open Set Rat Real

noncomputable def f (I : Set ℝ) (hI : IsOpen I) (hI_nonempty : I.Nonempty) (f : {q : ℚ // q ∈ I} → ℝ) :=
  ∀ (x y : {q : ℚ // q ∈ I}),
  4 * f ⟨(3 * (x : ℝ) + (y : ℝ)) / 4, sorry⟩ +
  4 * f ⟨((x : ℝ) + 3 * (y : ℝ)) / 4, sorry⟩ ≤
  f x + 6 * f ⟨((x : ℝ) + (y : ℝ)) / 2, sorry⟩ + f y

theorem continuous_extension_possible
  (I : Set ℝ) (hI : IsOpen I) (hI_nonempty : I.Nonempty)
  (f : {q : ℚ // q ∈ I} → ℝ)
  (H : ∀ (x y : {q : ℚ // q ∈ I}),
    4 * f ⟨(3 * (x : ℝ) + (y : ℝ)) / 4, sorry⟩ +
    4 * f ⟨((x : ℝ) + 3 * (y : ℝ)) / 4, sorry⟩ ≤
    f x + 6 * f ⟨((x : ℝ) + (y : ℝ)) / 2, sorry⟩ + f y)
  : ∃ (g : ℝ → ℝ), ContinuousOn g I ∧ ∀ q ∈ I ∩ (↑)''(Set.univ : Set ℚ), g q = f ⟨q, sorry⟩ :=
sorry

end continuous_extension_possible_l668_668015


namespace infinite_nonprime_divisible_l668_668789

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬ nat.prime n

theorem infinite_nonprime_divisible : ∃ᶠ n in at_top, ¬(nat.prime n ∧ n > 0) ∧ n ∣ (7^(n-1) - 3^(n-1)) := 
begin
  sorry
end

end infinite_nonprime_divisible_l668_668789


namespace condition_neither_sufficient_nor_necessary_l668_668954

variable (a b : ℝ)

theorem condition_neither_sufficient_nor_necessary 
    (h1 : ∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2))
    (h2 : ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b)) :
  ¬((a > b) ↔ (a^2 > b^2)) :=
sorry

end condition_neither_sufficient_nor_necessary_l668_668954


namespace sequence_is_1_over_n_l668_668364

noncomputable def sequence : ℕ → ℝ 
| 0     := 1   -- Define a1 (starting from n=1 in index 0)
| (n+1) := sequence n / (1 + sequence n)

theorem sequence_is_1_over_n (n : ℕ) (h : n > 0) : sequence n = 1 / n :=
by 
  sorry

end sequence_is_1_over_n_l668_668364


namespace min_shortest_side_length_triangle_l668_668281

theorem min_shortest_side_length_triangle 
  (h_alt1 : AD = 3) 
  (h_alt2 : BE = 4) 
  (h_alt3 : CF = 5) 
  (sides_int : ∀ (x : ℝ), x ∈ {BC, CA, AB} → x ∈ ℤ) : 
  ∃ (x : ℝ), (x ∈ {BC, CA, AB} ∧ x = 12) := 
sorry

end min_shortest_side_length_triangle_l668_668281


namespace num_of_valid_three_digit_numbers_l668_668564

theorem num_of_valid_three_digit_numbers :
  let digits := {0, 2}
  let odd_digits := {1, 3, 5}
  let is_valid (n : Nat) : Prop :=
    let d0 := n % 10
    let d1 := (n / 10) % 10
    let d2 := n / 100 
    d2 ∈ digits ∪ odd_digits ∧ d1 ∈ digits ∪ odd_digits ∧ d0 ∈ odd_digits ∧
    d0 ≠ d1 ∧ d1 ≠ d2 ∧ d2 ≠ d0 ∧ d2 ≠ 0
  { n // is_valid n }.card = 18 :=
sorry

end num_of_valid_three_digit_numbers_l668_668564


namespace expected_value_8_sided_die_l668_668869

-- Define the set of outcomes for an 8-sided die
def outcomes : finset ℕ := finset.range 9

-- Define the probability of each outcome
def probability (x : ℕ) : ℝ := if x ∈ outcomes.to_set then 1 / 8 else 0

-- Define the expected value calculation
noncomputable def expected_value : ℝ :=
  ∑ x in outcomes, (x : ℝ) * probability x

-- State the theorem we want to prove
theorem expected_value_8_sided_die : expected_value = 4.5 :=
sorry

end expected_value_8_sided_die_l668_668869


namespace negations_true_of_BD_l668_668199

def is_parallelogram_rhombus : Prop :=
  ∃ p : parallelogram, is_rhombus p

def exists_x_in_R : Prop :=
  ∃ x : ℝ, x^2 - 3 * x + 3 < 0

def forall_x_in_R : Prop :=
  ∀ x : ℝ, |x| + x^2 ≥ 0

def quad_eq_has_real_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - a * x + 1 = 0) → real_roots x

theorem negations_true_of_BD :
  (¬exists_x_in_R) ∧ (¬quad_eq_has_real_solutions a) :=
sorry

end negations_true_of_BD_l668_668199


namespace find_geometric_sequence_l668_668617

def geometric_sequence (b1 b2 b3 b4 : ℤ) :=
  ∃ q : ℤ, b2 = b1 * q ∧ b3 = b1 * q^2 ∧ b4 = b1 * q^3

theorem find_geometric_sequence :
  ∃ b1 b2 b3 b4 : ℤ, 
    geometric_sequence b1 b2 b3 b4 ∧
    (b1 + b4 = -49) ∧
    (b2 + b3 = 14) ∧ 
    ((b1, b2, b3, b4) = (7, -14, 28, -56) ∨ (b1, b2, b3, b4) = (-56, 28, -14, 7)) :=
by
  sorry

end find_geometric_sequence_l668_668617


namespace find_a_for_unextended_spring_twice_l668_668099

theorem find_a_for_unextended_spring_twice (a : ℝ) (h1 : (0 < a) ∧ (a < 2)) :
  (4*a^2 - 8*a + 1 ≥ 0) → (1 + 2*a > 2) → (1 + a - 2*a - 1 < 0) → ((0 < (2 * a - 1) / (2 * a)) ∧ ((2 * a - 1) / (2 * a) < 1)) → (a ∈ (1 + (real.sqrt 3) / 2, 2)) :=
by
  intros h2 h3 h4 h5
  sorry  

end find_a_for_unextended_spring_twice_l668_668099


namespace seq_composite_l668_668458

-- Define the sequence recurrence relation
def seq (a : ℕ → ℕ) : Prop :=
  ∀ (k : ℕ), k ≥ 1 → a (k+2) = a (k+1) * a k + 1

-- Prove that for k ≥ 9, a_k - 22 is composite
theorem seq_composite (a : ℕ → ℕ) (h_seq : seq a) :
  ∀ (k : ℕ), k ≥ 9 → ∃ d, d > 1 ∧ d < a k ∧ d ∣ (a k - 22) :=
by
  sorry

end seq_composite_l668_668458


namespace quadratic_root_property_l668_668274

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end quadratic_root_property_l668_668274


namespace smallest_n_has_1000_solutions_l668_668399

-- Define the function f as the absolute value of sine of pi times x
def f (x : ℝ) : ℝ := |Real.sin (Real.pi * x)|

-- Define the theorem to prove that the smallest positive integer n that makes nf(xf(x)) = x
-- have at least 1000 real solutions is n = 500
theorem smallest_n_has_1000_solutions : ∃ (n : ℕ), (∀ (x : ℝ), nf(xf(x)) = x) → (n >= 500) := sorry

end smallest_n_has_1000_solutions_l668_668399


namespace statement_A_statement_C_l668_668507

noncomputable def normal_xi : measure_theory.probability_measure ℝ :=
measure_theory.probability_measure.mk' (λ s, if s ≤ 4 then 0.79 else 0.21)

theorem statement_A (σ : ℝ) :
  ∀ (ξ : ℝ → ℝ), (∀ x, ξ x = real.normal 1 σ) →
  ∀ x, ξ x = normal_xi.to_fun x :=
by sorry

theorem statement_C :
  ∀ (ξ : ℝ → ℝ), (∀ x, ξ x = real.binomial 4 (1/4)) →
  real.expectation ξ = 1 :=
by sorry

end statement_A_statement_C_l668_668507


namespace possible_integer_roots_l668_668439

noncomputable theory

-- Definitions for the polynomial and integer coefficients
def polynomial := {f : ℤ[X] // degree f = 5}
def integer_roots (p : polynomial) : ℕ := 
  @finsupp.count ℤ _ _ p.roots multiset.to_fide.list pred (to_list p.roots)

-- Levi asks for the possible values of n, where it is the number of integer roots
theorem possible_integer_roots (p : polynomial) : 
  ∃ (n : ℕ), n ∈ {0, 1, 2, 3, 4, 5} ∧ integer_roots p = n :=
sorry

end possible_integer_roots_l668_668439


namespace proof_problem_l668_668197

-- Proposition B: ∃ x ∈ ℝ, x^2 - 3*x + 3 < 0
def propB : Prop := ∃ x : ℝ, x^2 - 3 * x + 3 < 0

-- Proposition D: ∀ x ∈ ℝ, x^2 - a*x + 1 = 0 has real solutions
def propD (a : ℝ) : Prop := ∀ x : ℝ, ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0

-- Negation of Proposition B: ∀ x ∈ ℝ, x^2 - 3 * x + 3 ≥ 0
def neg_propB : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 ≥ 0

-- Negation of Proposition D: ∃ a ∈ ℝ, ∃ x ∈ ℝ, ∄ (x1 x2 : ℝ), x^2 - a * x + 1 = 0
def neg_propD : Prop := ∃ a : ℝ, ∀ x : ℝ, ¬ ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0 

-- The main theorem combining the results based on the solutions.
theorem proof_problem : neg_propB ∧ neg_propD :=
by
  sorry

end proof_problem_l668_668197


namespace vasya_no_purchase_days_l668_668949

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668949


namespace continuous_odd_function_integral_zero_continuous_even_function_integral_relation_continuous_positive_function_integral_positive_continuous_function_integral_not_always_positive_l668_668196

-- Define properties of functions and integrals
theorem continuous_odd_function_integral_zero {f : ℝ → ℝ} (h_cont : continuous f) (h_odd : ∀ x, f(-x) = -f(x)) (a : ℝ) :
  ∫ x, f x in -a..a = 0 := by
sorry

theorem continuous_even_function_integral_relation {f : ℝ → ℝ} (h_cont : continuous f) (h_even : ∀ x, f(-x) = f(x)) (a : ℝ) :
  ∫ x, f x in -a..a = 2 * ∫ x, f x in 0..a := by
sorry

theorem continuous_positive_function_integral_positive {f : ℝ → ℝ} (h_cont : continuous_on f (set.Icc a b)) (h_pos : ∀ x ∈ set.Icc a b, f x > 0) :
  ∫ x in set.Icc a b, f x > 0 := by
sorry

theorem continuous_function_integral_not_always_positive {f : ℝ → ℝ} 
  (h_cont : continuous_on f (set.Icc a b)) (h_integral_pos : ∫ x in a..b, f x > 0) : 
  ¬ (∀ x ∈ set.Icc a b, f x > 0) := by
  -- Provide counterexample:
  let counter_example := λ x, x^3
  have h1 : continuous_on counter_example (set.Icc (-1 : ℝ) 2) := by
    -- Proof of continuity
    sorry
  have h2 : ∫ x in -1..2, counter_example x > 0 := by
    -- Proof of integral value being positive
    sorry
  have h3 : ¬ (∀ x ∈ set.Icc (-1 : ℝ) 2, counter_example x > 0) := by
    -- Proof that counter example function is not positive always
    sorry
  exact ⟨-1, 2, h1, h2, h3⟩

end continuous_odd_function_integral_zero_continuous_even_function_integral_relation_continuous_positive_function_integral_positive_continuous_function_integral_not_always_positive_l668_668196


namespace Gwen_avg_speed_trip_l668_668515

theorem Gwen_avg_speed_trip : 
  ∀ (d1 d2 s1 s2 t1 t2 : ℝ), 
  d1 = 40 → d2 = 40 → s1 = 15 → s2 = 30 →
  d1 / s1 = t1 → d2 / s2 = t2 →
  (d1 + d2) / (t1 + t2) = 20 :=
by 
  intros d1 d2 s1 s2 t1 t2 hd1 hd2 hs1 hs2 ht1 ht2
  sorry

end Gwen_avg_speed_trip_l668_668515


namespace inverse_equals_self_l668_668379

def f (l x : ℝ) := (3 * x + 4) / (l * x - 5)

theorem inverse_equals_self (l : ℝ) :
  (∀ x, ∃ y, f l y = x) →
  l ∈ set.Ioo (-∞ : ℝ) (25 / 4 : ℝ) ∪ set.Ioo (25 / 4 : ℝ) (∞ : ℝ)  :=
begin
  sorry
end

end inverse_equals_self_l668_668379


namespace fractions_order_l668_668128

theorem fractions_order :
  let f1 := (18 : ℚ) / 14
  let f2 := (16 : ℚ) / 12
  let f3 := (20 : ℚ) / 16
  f3 < f1 ∧ f1 < f2 :=
by {
  sorry
}

end fractions_order_l668_668128


namespace log_expression_solution_unique_l668_668223

theorem log_expression_solution_unique (x : ℝ) (h : x = Real.log (50 + 3 * x) / Real.log 2) : x = 8 :=
begin
  sorry
end

end log_expression_solution_unique_l668_668223


namespace sum_bound_for_k_l668_668432

variable {x : ℕ → ℝ}

axiom seq_positive (n : ℕ) : 0 < x n
axiom seq_nonincreasing (n m : ℕ) : n ≤ m → x n ≥ x m
axiom special_sum_bound (n : ℕ) : (∑ i in finset.range n, x (i+1)^2 / (i+1)) ≤ 1

theorem sum_bound_for_k (k : ℕ) : (∑ i in finset.range k, x (i+1) / (i+1)) ≤ 3 := sorry

end sum_bound_for_k_l668_668432


namespace john_collects_crabs_l668_668011

-- Definitions for the conditions
def baskets_per_week : ℕ := 3
def crabs_per_basket : ℕ := 4
def price_per_crab : ℕ := 3
def total_income : ℕ := 72

-- Definition for the question
def times_per_week_to_collect (baskets_per_week crabs_per_basket price_per_crab total_income : ℕ) : ℕ :=
  (total_income / price_per_crab) / (baskets_per_week * crabs_per_basket)

-- The theorem statement
theorem john_collects_crabs (h1 : baskets_per_week = 3) (h2 : crabs_per_basket = 4) (h3 : price_per_crab = 3) (h4 : total_income = 72) :
  times_per_week_to_collect baskets_per_week crabs_per_basket price_per_crab total_income = 2 :=
by
  sorry

end john_collects_crabs_l668_668011


namespace sum_fraction_result_l668_668394

noncomputable theory

open Complex

theorem sum_fraction_result (x : ℂ) (h1 : x^2009 = 1) (h2 : x ≠ 1) :
  (∑ k in Finset.range 2008, x^(2 * (k + 1)) / (x^(k + 1) - 1)) = 1003 :=
  sorry

end sum_fraction_result_l668_668394


namespace percentage_markup_is_20_l668_668454

-- Definitions of the conditions
def Cp : ℝ := 7000
def Sp : ℝ := 8400

-- Definition of markup and percentage markup
def Markup := Sp - Cp
def PercentageMarkup := (Markup / Cp) * 100

-- Statement of the problem
theorem percentage_markup_is_20 : PercentageMarkup = 20 := by
  sorry

end percentage_markup_is_20_l668_668454


namespace total_area_painted_correct_l668_668159

def width := 12 -- width of the barn (yd)
def length := 15 -- length of the barn (yd)
def height := 8 -- height of the barn (yd)
def door_width := 4 -- width of the door (yd)
def door_height := 3 -- height of the door (yd)

def area_wall_with_door := (width * height) - (door_width * door_height)
def area_wall_without_door := width * height
def area_first_pair_walls := area_wall_with_door + area_wall_without_door

def area_second_pair_walls := (length * height) * 2
def area_ceiling := width * length

def total_area_walls := area_first_pair_walls + area_second_pair_walls
def total_area_inside_outside := total_area_walls * 2
def total_area_to_be_painted := total_area_inside_outside + area_ceiling

theorem total_area_painted_correct : total_area_to_be_painted = 1020 := by
  sorry

end total_area_painted_correct_l668_668159


namespace no_calls_days_l668_668031

noncomputable def days_with_no_calls (days total_days : ℕ) : ℕ := sorry

theorem no_calls_days :
  days_with_no_calls 365 (by simp [finset.range]) = 146 :=
begin
  -- Conditions stated in conditions a)
  have calls_every_3_days := 3,
  have calls_every_4_days := 4,
  have calls_every_5_days := 5,
  
  -- All three grandchildren called her on December 31, 2016.
  -- The problem asks about the next year which is 365 days.

  sorry -- proof will be developed here
end

end no_calls_days_l668_668031


namespace checkerboard_pattern_possible_l668_668354

theorem checkerboard_pattern_possible (n m : ℕ) : 
  (∀ i j : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ m → 1) →
  (∃ f : ℕ → ℕ → ℤ, (∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ m → f i j = 1) ∨ (∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ m → f i j = -1)) ↔ 
  (n % 4 = 0 ∧ m % 4 = 0) :=
by
  sorry

end checkerboard_pattern_possible_l668_668354


namespace min_value_f_l668_668285

noncomputable def f (a b x : ℝ) : ℝ :=
  a * x ^ 3 + b * real.log (x + real.sqrt (1 + x ^ 2)) + 3

theorem min_value_f (a b : ℝ) 
  (h_max_f_neg : ∀ x : ℝ, x < 0 → f a b x ≤ 10) :
  ∃ x : ℝ, x > 0 ∧ f a b x = -4 :=
by
  sorry

end min_value_f_l668_668285


namespace factorial_division_l668_668295

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end factorial_division_l668_668295


namespace algebra_expression_evaluation_l668_668663

theorem algebra_expression_evaluation (a b c d e : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : e < 0) 
  (h4 : abs e = 1) : 
  (-a * b) ^ 2009 - (c + d) ^ 2010 - e ^ 2011 = 0 := by 
  sorry

end algebra_expression_evaluation_l668_668663


namespace find_f_2010_l668_668391

def odd_function_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def periodicity_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 3) = -f(x)

theorem find_f_2010
  (f : ℝ → ℝ)
  (h1 : odd_function_on_reals f)
  (h2 : periodicity_3 f) :
  f 2010 = 0 :=
sorry

end find_f_2010_l668_668391


namespace expected_games_value_l668_668521

noncomputable def probability_A_wins := 2 / 3 : ℚ
noncomputable def probability_B_wins := 1 - probability_A_wins

def ends_match_condition (x : ℕ) : Prop :=
  ∃ (i : ℕ), (x_i < x_{i + 1}) ∧ (x > i) ∧ (∃ (p_i : ℚ), p_i = probability_A_wins^(x_i - i) * probability_B_wins^(x_i - i))

noncomputable def expected_number_of_games : ℚ :=
  ∑ i in (finset.range (x_i + 1)), (x_i * p_i)

theorem expected_games_value (S : ℚ) 
  (hA : probability_A_wins = 2 / 3) 
  (hB : probability_B_wins = 1 / 3) 
  (hx : ends_match_condition x_i) 
  (hp : ∑ i in (finset.range (x_i + 1)), (x_i * p_i)) : 
  S = 18 / 5 :=
  begin
    sorry
  end

end expected_games_value_l668_668521


namespace calculate_y_when_x_is_neg2_l668_668790

def conditional_program (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 3
  else if x > 0 then
    -2 * x + 5
  else
    0

theorem calculate_y_when_x_is_neg2 : conditional_program (-2) = -1 :=
by
  sorry

end calculate_y_when_x_is_neg2_l668_668790


namespace sum_inequality_condition_l668_668760

variables {n : ℕ} (a b : Fin n → ℝ)

theorem sum_inequality_condition :
  (∀ (x : Fin n → ℝ), (∀ i j : Fin n, i < j → x i ≤ x j) → (∑ i, a i * x i) ≤ (∑ i, b i * x i)) ↔
    ((∀ k : Fin n, k.1 < n - 1 → (Finset.range (k.1 + 1)).sum (λ i, a i) ≥ (Finset.range (k.1 + 1)).sum (λ i, b i))
    ∧ (Finset.range n).sum a = (Finset.range n).sum b) :=
sorry

end sum_inequality_condition_l668_668760


namespace inscribed_sphere_radius_l668_668530

theorem inscribed_sphere_radius 
  (base_radius height liquid_height : ℝ) 
  (hr₁ : base_radius = 15) 
  (hr₂ : height = 30) 
  (hr₃ : liquid_height = 10) :
  ∃ (radius : ℝ), radius = 10 := 
by
  use 10
  sorry

end inscribed_sphere_radius_l668_668530


namespace factorization_correct_l668_668126

theorem factorization_correct (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) :=
by
  sorry

end factorization_correct_l668_668126


namespace find_angle_B_find_area_l668_668738

variable {A B C : ℝ}
variable {a b c : ℝ}

def condition1 : Prop := (2 * Real.sin B * Real.cos C = 2 * Real.sin A * Real.cos C + 2 * Real.cos C^2 - Real.sin C)

def condition2 : Prop := (a = 1)
def condition3 : Prop := (b = sqrt 7)
def condition4 : Prop := (a = 1)
def condition5 : Prop := (b = sqrt 7)

-- Proving the measure of angle B
theorem find_angle_B : condition1 -> condition2 -> condition3 -> B = Real.pi / 3 :=
by
  sorry

-- Proving the area of triangle ABC
theorem find_area : condition1 -> condition4 -> condition5 -> 
                   ∃ c, (c > 0) ∧ (b^2 = a^2 + c^2 - 2*a*c*(1/2)) ∧ 
                   (1/2)*a*c*(sqrt 3/2) = (3*sqrt 3)/4 :=
by
  sorry

end find_angle_B_find_area_l668_668738


namespace closest_ratio_adults_children_l668_668812

theorem closest_ratio_adults_children :
  ∃ (a c : ℕ), 25 * a + 15 * c = 1950 ∧ a ≥ 1 ∧ c ≥ 1 ∧ a / c = 24 / 25 := sorry

end closest_ratio_adults_children_l668_668812


namespace mix_substrates_equal_parts_l668_668483

-- Define the ratios for "Orchid-1" and "Orchid-2"
def orchid1_bark (x : ℝ) : ℝ := 3 * x
def orchid1_peat (x : ℝ) : ℝ := 2 * x
def orchid1_sand (x : ℝ) : ℝ := x

def orchid2_bark (y : ℝ) : ℝ := y / 2
def orchid2_peat (y : ℝ) : ℝ := y
def orchid2_sand (y : ℝ) : ℝ := 3 * y / 2

-- Mixing ratios resulting in equal parts
theorem mix_substrates_equal_parts (a b : ℝ) (x y : ℝ) :
  3 * a + b / 2 = 2 * a + b ∧
  2 * a + b = a + 3 * b / 2 →
  a = b :=
by
  intros h,
  sorry

end mix_substrates_equal_parts_l668_668483


namespace sequence_product_1009_l668_668492

-- Define the sequence
def sequence_term (k : Nat) : ℚ := (2 * k + 2) / (2 * k + 1)

-- Define the product of sequence from 1 to n
def sequence_product (n : Nat) : ℚ :=
  List.prod (List.map sequence_term (List.range (n + 1)))

theorem sequence_product_1009 :
  sequence_product 1009 = 673.3333333333333 (i.e., 673.(3 repeating)):
  sorry

end sequence_product_1009_l668_668492


namespace jacob_younger_than_michael_l668_668371

variables (J M : ℕ)

theorem jacob_younger_than_michael (h1 : M + 9 = 2 * (J + 9)) (h2 : J = 5) : M - J = 14 :=
by
  -- Insert proof steps here
  sorry

end jacob_younger_than_michael_l668_668371


namespace smallest_integer_of_lcm_gcd_l668_668499

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l668_668499


namespace yvette_sundae_cost_l668_668885

noncomputable def cost_friends : ℝ := 7.50 + 10.00 + 8.50
noncomputable def final_bill : ℝ := 42.00
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def tip_amount : ℝ := tip_percentage * final_bill

theorem yvette_sundae_cost : 
  final_bill - (cost_friends + tip_amount) = 7.60 := by
  sorry

end yvette_sundae_cost_l668_668885


namespace seq_1000_eq_2098_l668_668726

-- Define the sequence a_n
def seq (n : ℕ) : ℤ := sorry

-- Initial conditions
axiom a1 : seq 1 = 100
axiom a2 : seq 2 = 101

-- Recurrence relation condition
axiom recurrence_relation : ∀ n : ℕ, 1 ≤ n → seq n + seq (n+1) + seq (n+2) = 2 * ↑n + 3

-- Main theorem to prove
theorem seq_1000_eq_2098 : seq 1000 = 2098 :=
by {
  sorry
}

end seq_1000_eq_2098_l668_668726


namespace vasya_purchase_l668_668931

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668931


namespace cos_equation_solution_range_l668_668771

theorem cos_equation_solution_range (t : ℝ) (h_t : 0 ≤ t ∧ t ≤ π) :
  (∃ x : ℝ, cos (x + t) = 1 - cos x) ↔ t ∈ set.Icc 0 (2 * π / 3) :=
by sorry

end cos_equation_solution_range_l668_668771


namespace largest_prime_factor_of_3913_l668_668487

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), nat.prime p ∧ p ∣ 3913 ∧ (∀ q, nat.prime q ∧ q ∣ 3913 → q ≤ p) ∧ p = 43 :=
sorry

end largest_prime_factor_of_3913_l668_668487


namespace figure_perimeter_l668_668803

theorem figure_perimeter (H_segments : ℕ) (V_segments : ℕ) :
  H_segments = 16 ∧ V_segments = 10 → H_segments + V_segments = 26 := 
by {
  intro h,
  cases h with H_eq V_eq,
  rw [H_eq, V_eq],
  norm_num,
  sorry
}

end figure_perimeter_l668_668803


namespace tangent_line_length_l668_668109

theorem tangent_line_length 
  (A B C : Type) 
  (d_A : A)
  (d_B : B)
  (r_A r_B : ℝ)
  (distance_AB : ℝ)
  (tangent_condition : distance_AB = r_A + r_B)
  (tangent_line : Type) 
  (intersection_point : tangent_line → C) 
  (touches_circles : ∀ t : tangent_line, 
    ∃ p1 p2 : C, ∃ tangency_A tangency_B : Prop, 
      (tangency_A ∧ tangency_B)) : 
  let BC_length : ℝ := 44 / 3 in BC_length = 44 / 3 := 
sorry

end tangent_line_length_l668_668109


namespace simplest_square_root_l668_668881

theorem simplest_square_root (A B C D : ℝ) 
  (hA : A = real.sqrt (1 / 3)) 
  (hB : B = real.sqrt 0.5) 
  (hC : C = real.sqrt 6) 
  (hD : D = real.sqrt 16) : C = real.sqrt 6 :=
by
  -- to be filled with the proof
  sorry

end simplest_square_root_l668_668881


namespace minimum_value_l668_668627

variable (θ : ℝ)

theorem minimum_value : ∃ θ: ℝ, ∀ θ ∈ set.univ, 
  (1 / (2 - real.cos θ ^ 2) + 1 / (2 - real.sin θ ^ 2)) ≥ 4 / 3 :=
sorry

end minimum_value_l668_668627


namespace similarity_of_triangle_l668_668509

noncomputable def side_length (AB BC AC : ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 1 → (AB, BC, AC) = (k * AB, k * BC, k * AC)

theorem similarity_of_triangle (AB BC AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : AC > 0) :
  side_length (2 * AB) (2 * BC) (2 * AC) = side_length AB BC AC :=
by sorry

end similarity_of_triangle_l668_668509


namespace poultry_prices_l668_668534

theorem poultry_prices :
  ∃ (c d g : ℕ), 3 * c + d = 2 * g ∧ c + 2 * d + 3 * g = 25 ∧ c = 2 ∧ d = 4 ∧ g = 5 :=
by
  use 2, 4, 5
  split
  · calc
      3 * 2 + 4 = 6 + 4 := by rfl
      _ = 2 * 5 := by rfl
  split
  · calc
      2 + 2 * 4 + 3 * 5 = 2 + 2 * 4 + 15 := by rfl
      _ = 2 + 8 + 15 := by rfl
      _ = 25 := by rfl

  repeat { constructor <|> rfl }

end poultry_prices_l668_668534


namespace initial_sugar_weight_l668_668833

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end initial_sugar_weight_l668_668833


namespace segment_in_regular_pentagon_l668_668748

/--
  Problem: 
  Prove that it is impossible to place a segment inside a regular pentagon
  so that it is seen from all vertices at the same angle.
-/
theorem segment_in_regular_pentagon (P : Type) [regular_pentagon P] :
  ¬ ∃ (XY : segment P), (∀ (v : P), ∃ (angle : angle P), seen_from v XY = angle) :=
by
  -- Proof goes here
  sorry

end segment_in_regular_pentagon_l668_668748


namespace painting_ways_correct_l668_668525

noncomputable def num_ways_to_paint : ℕ :=
  let red := 1
  let green_or_blue := 2
  let total_ways_case1 := red
  let total_ways_case2 := (green_or_blue ^ 4)
  let total_ways_case3 := green_or_blue ^ 3
  let total_ways_case4 := green_or_blue ^ 2
  let total_ways_case5 := green_or_blue
  let total_ways_case6 := red
  total_ways_case1 + total_ways_case2 + total_ways_case3 + total_ways_case4 + total_ways_case5 + total_ways_case6

theorem painting_ways_correct : num_ways_to_paint = 32 :=
  by
  sorry

end painting_ways_correct_l668_668525


namespace trapezoid_GHCD_area_triangle_EGH_area_l668_668737

noncomputable def trapezoid_area (a b height : ℝ) : ℝ :=
  (a + b) / 2 * height

noncomputable def triangle_area (base height : ℝ) : ℝ :=
  0.5 * base * height

theorem trapezoid_GHCD_area :
  ∀ (AB CD height : ℝ),
    AB = 10 →
    CD = 24 →
    height = 14 →
    let GH := (AB + CD) / 2 in
    let new_height := height / 2 in
    trapezoid_area GH CD new_height = 143.5 :=
by
  intros AB CD height hAB hCD hHeight GH new_height
  rw [hAB, hCD, hHeight]
  have hGH : GH = (10 + 24) / 2 := by sorry
  have hNewHeight : new_height = 7 := by sorry
  rw [hGH, hNewHeight]
  exact sorry

theorem triangle_EGH_area :
  ∀ (GH : ℝ),
    GH = 17 →
    let base := GH / 2 in
    let height := 14 in
    triangle_area base height = 59.5 :=
by
  intros GH hGH base height
  rw [hGH]
  have hBase : base = 8.5 := by sorry
  have hHeight : height = 14 := by sorry
  rw [hBase, hHeight]
  exact sorry

end trapezoid_GHCD_area_triangle_EGH_area_l668_668737


namespace total_pure_acid_amount_l668_668494

theorem total_pure_acid_amount :
  let acid1 := 0.40 * 6
  let acid2 := 0.35 * 4
  let acid3 := 0.55 * 3
  acid1 + acid2 + acid3 = 5.45 :=
by
  let acid1 := 0.40 * 6
  let acid2 := 0.35 * 4
  let acid3 := 0.55 * 3
  have h1 : acid1 + acid2 + acid3 = 2.4 + 1.4 + 1.65 := by rfl
  have h2 : 2.4 + 1.4 + 1.65 = 5.45 := by norm_num
  rw [h1, h2]
  exact rfl

end total_pure_acid_amount_l668_668494


namespace point_k_outside_and_distance_KB_l668_668386

noncomputable def L : Point := intersection CE DF
noncomputable def K : Point := L + (AC - 3 * BC)

theorem point_k_outside_and_distance_KB
    (ABCDEF : regular_hexagon)
    (side_length : 2)
    (L_is_intersection : L = intersection CE DF)
    (K_definition : K = L + (AC - 3 * BC)) :
    (K is_outside ABCDEF) ∧ (distance K B = (2 * sqrt 3) / 3) :=
sorry

end point_k_outside_and_distance_KB_l668_668386


namespace inscribed_sphere_radius_proof_l668_668460

-- Let a be the side length of the base of the pyramid.
variables (a : ℝ)

-- The pyramid is a regular quadrangular pyramid with a base angle of 45 degrees.
-- Assume the lateral faces form an angle of 45 degrees with the base.
def inscribed_sphere_radius (a : ℝ) : ℝ :=
  a * (Real.sqrt 2 - 1) / 2

theorem inscribed_sphere_radius_proof :
  ∀ (a : ℝ), a > 0 → 
  let r := inscribed_sphere_radius a 
  in r = a * (Real.sqrt 2 - 1) / 2 :=
by
  intro a ha
  dsimp [inscribed_sphere_radius]
  sorry

end inscribed_sphere_radius_proof_l668_668460


namespace minimize_p_for_repeating_decimal_l668_668718

noncomputable def repeating_decimal_as_fraction : ℚ :=
  2 / 11

theorem minimize_p_for_repeating_decimal :
  ∀ p q : ℕ, p ≠ 0 ∧ q ≠ 0 ∧ (0.181818181818181818 = (p : ℚ) / q) ∧ (nat.gcd p q = 1 ∧ ∀ p1 q1 : ℕ, (p1 : ℚ) / q1 = 0.181818181818181818 → q1 < q → nat.gcd p1 q1 ≠ 1) → p = 2 := 
by { sorry }

end minimize_p_for_repeating_decimal_l668_668718


namespace volume_remaining_cube_l668_668964

theorem volume_remaining_cube (a : ℝ) (original_volume vertex_cube_volume : ℝ) (number_of_vertices : ℕ) :
  original_volume = a^3 → 
  vertex_cube_volume = 1 → 
  number_of_vertices = 8 → 
  a = 3 →
  original_volume - (number_of_vertices * vertex_cube_volume) = 19 := 
by
  sorry

end volume_remaining_cube_l668_668964


namespace vasya_days_without_purchase_l668_668917

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668917


namespace sum_of_coefficients_of_polynomial_l668_668660

-- Definition conditions
variables (α β : ℂ)
def polynomial_sum_is_one := (α + β) = 1
def polynomial_product_is_one := (α * β) = 1

-- Theorem statement
theorem sum_of_coefficients_of_polynomial :
  polynomial_sum_is_one α β →
  polynomial_product_is_one α β →
  ∑ c in (polynomial.coeff (α ^ 2005 + β ^ 2005)), c = -1 :=
by
  intros h_sum h_product
  sorry

end sum_of_coefficients_of_polynomial_l668_668660


namespace inscribed_triangle_inequality_l668_668524

theorem inscribed_triangle_inequality 
  (d : ℝ) (A B C : EuclideanGeometry.Point) 
  (h_circle : EuclideanGeometry.is_circle (segment A B) (2 * d))
  (h_thales : EuclideanGeometry.sits_on_circle C (segment A B))
  (h_not_on_diameter : C ≠ A ∧ C ≠ B) :
  let s := EuclideanGeometry.distance A C + EuclideanGeometry.distance B C
  in s^2 ≤ 8 * d^2 :=
sorry

end inscribed_triangle_inequality_l668_668524


namespace largest_L_conditioned_l668_668982

theorem largest_L_conditioned (
  let sum_config1 (x : ℕ) := x + (x - 7) + (x - 8),
  let sum_config2 (x : ℕ) := x + (x - 6) + (x - 7),
  let sum_config3 (x : ℕ) := x + (x - 1) + (x - 7),
  let sum_config4 (x : ℕ) := x + (x - 1) + (x - 8)
) : ∃ (x : ℕ), (sum_config1 x = 2015 ∨ sum_config2 x = 2015 ∨ sum_config3 x = 2015 ∨ sum_config4 x = 2015) ∧ x = 676 := 
by 
  sorry

end largest_L_conditioned_l668_668982


namespace systematic_sampling_l668_668853

theorem systematic_sampling :
  ∃ (l : List ℕ), l = [3, 13, 23, 33, 43] ∧ (∀ i, i < l.length - 1 → l[i+1] = l[i] + 10) :=
by
  sorry

end systematic_sampling_l668_668853


namespace expression_has_square_l668_668227

theorem expression_has_square (N : ℕ) : ∃ N : ℕ, N = 10^3968 - 10^1985 + 25 :=
begin
  use N,
  sorry,
end

end expression_has_square_l668_668227


namespace count_9s_in_subtraction_l668_668004

def count_digit_9 (n : Nat) : Nat :=
  (Int.ofNat n).digits 10 |>.filter (λ d => d = 9) |>.length

theorem count_9s_in_subtraction : count_digit_9 (10000000000 - 101011) = 7 :=
by
  sorry

end count_9s_in_subtraction_l668_668004


namespace ratio_345_iff_arithmetic_sequence_l668_668300

-- Define the variables and the context
variables (a b c : ℕ) -- assuming non-negative integers for simplicity
variable (k : ℕ) -- scaling factor for the 3:4:5 ratio
variable (d : ℕ) -- common difference in the arithmetic sequence

-- Conditions given
def isRightAngledTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ a < b ∧ b < c

def is345Ratio (a b c : ℕ) : Prop :=
  ∃ k, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k

def formsArithmeticSequence (a b c : ℕ) : Prop :=
  ∃ d, b = a + d ∧ c = b + d 

-- The statement to prove: sufficiency and necessity
theorem ratio_345_iff_arithmetic_sequence 
  (h_triangle : isRightAngledTriangle a b c) :
  (is345Ratio a b c ↔ formsArithmeticSequence a b c) :=
sorry

end ratio_345_iff_arithmetic_sequence_l668_668300


namespace num_palindromic_strings_first_1000_l668_668725

noncomputable def reversal (s : String) : String := List.foldl (flip String.append) "" s.toList.reverse

noncomputable def sequence : ℕ → String
| 1      := "A"
| 2      := "B"
| (n+3)  := sequence (n+2) ++ reversal (sequence (n+1))

def is_palindrome (s : String) : Bool :=
s = reversal s

def count_palindromes_up_to (n : ℕ) : ℕ :=
(nat.list.range n).countp (fun i => is_palindrome (sequence (i + 1)))

theorem num_palindromic_strings_first_1000 : count_palindromes_up_to 1000 = 667 :=
sorry

end num_palindromic_strings_first_1000_l668_668725


namespace smallest_integer_n_exists_l668_668502

-- Define the conditions
def lcm_gcd_correct_division (a b : ℕ) : Prop :=
  (lcm a b) / (gcd a b) = 44

-- Define the main problem
theorem smallest_integer_n_exists : ∃ n : ℕ, lcm_gcd_correct_division 60 n ∧ 
  (∀ k : ℕ, lcm_gcd_correct_division 60 k → k ≥ n) :=
begin
  sorry
end

end smallest_integer_n_exists_l668_668502


namespace reading_time_difference_l668_668132

theorem reading_time_difference (x_pages_per_hour : ℕ) (m_pages_per_hour : ℕ) (book_pages : ℕ)
  (hx : x_pages_per_hour = 120) (hm : m_pages_per_hour = 60) (hb : book_pages = 300) :
  let time_difference_minutes := ((book_pages / m_pages_per_hour) - (book_pages / x_pages_per_hour)) * 60
  in time_difference_minutes = 150 := by
  sorry

end reading_time_difference_l668_668132


namespace correct_system_of_equations_l668_668147

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l668_668147


namespace largest_n_sum_below_million_l668_668363

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_n_sum_below_million :
  ∃ n : ℕ, (∀ k : ℕ, k < n → binomial (2 * k) k)
   ∧ (∑ k in Finset.range n, binomial (2 * k) k < 1000000) 
   ∧ (¬ ∑ k in Finset.range (n+1), binomial (2 * k) k < 1000000) := 
sorry

end largest_n_sum_below_million_l668_668363


namespace no_segment_equal_angle_in_regular_pentagon_l668_668745

theorem no_segment_equal_angle_in_regular_pentagon (P : Type)
  [f : fintype P] [decidable_eq P] (hP : ∀ (A B C : P), 
  ∃ (c : P), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
  is_regular_pentagon P ∧ (∃ (X Y : P), segment_viewed_from_all_vertices_same_angle (X Y : segment P) A B C) → False):
  False :=
sorry

end no_segment_equal_angle_in_regular_pentagon_l668_668745


namespace remaining_volume_proof_l668_668820

def cube_side_length : ℝ := 5
def cylinder_radius : ℝ := 1.5
def cube_volume : ℝ := cube_side_length^3
def cylinder_height : ℝ := cube_side_length
def cylinder_volume : ℝ := π * cylinder_radius^2 * cylinder_height
def remaining_volume : ℝ := cube_volume - cylinder_volume

theorem remaining_volume_proof : remaining_volume = 125 - 11.25 * π :=
by 
  calc remaining_volume = cube_volume - cylinder_volume : rfl
                   ... = 125 - 11.25 * π : sorry

end remaining_volume_proof_l668_668820


namespace intersection_A_B_l668_668287

def A (x : ℝ) : Prop := ∃ y, y = Real.log (-x^2 - 2*x + 8) ∧ -x^2 - 2*x + 8 > 0
def B (x : ℝ) : Prop := Real.log x / Real.log 2 < 1 ∧ x > 0

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l668_668287


namespace proof_problem_l668_668653

-- Definitions for sequences a_n and b_n
def a (n : ℕ+) : ℕ 
| 1 := 1
| 2 := 4
| (n+2) := 3 * a (n + 1) - 2 * a n

def b (n : ℕ+) : ℕ := a (n + 1) - a n

-- Definitions for sum S_n
def S (n : ℕ+) : ℕ := ∑ i in finset.range n, a (i + 1)

-- Theorem to prove all the assertions
theorem proof_problem : 
  (∀ n : ℕ+, b n = 3 * 2^(n-1)) ∧
  (∀ n : ℕ+, n ≥ 1 → a n = 3 * 2^(n-1) - 2) ∧
  (∃ n : ℕ+, S n > 21 - 2 * n) :=
by 
  sorry

end proof_problem_l668_668653


namespace probability_multiple_of_100_l668_668343

-- Definitions of the problem
def set_of_numbers : List ℕ := [5, 10, 15, 20, 25, 30, 35]

def is_multiple_of_100 (x : ℕ) : Prop :=
  x % 100 = 0

-- Definition capturing the condition of multiplying two distinct members
def prob_product_multiple_of_100 : ℚ :=
  let pairs := (set_of_numbers.product set_of_numbers).filter (fun pair => pair.1 ≠ pair.2)
  let count_valid_pairs := pairs.filter (fun pair => is_multiple_of_100 (pair.1 * pair.2)).length
  (count_valid_pairs : ℚ) / (pairs.length : ℚ)

-- The theorem statement
theorem probability_multiple_of_100 :
  prob_product_multiple_of_100 = 1 / 7 :=
sorry

end probability_multiple_of_100_l668_668343


namespace incorrect_statement_D_l668_668505

-- Definitions for the conditions
def statement_A : Prop := ∀ (c: Prop) (body: Prop), 
  (while c do body) = (if c then (body; while c do body) else ())

def statement_B : Prop := ∀ (c: Prop) (body: Prop), 
  ¬c → (while c do body) = ()

def statement_C : Prop := Prop -- placeholder for the correct definition that the While statement is also called a while-type loop.

def statement_D : Prop := Prop -- placeholder for the incorrect definition that a while-type loop is sometimes also referred to as a "post-test" loop.

-- The proof problem is to show that statement_D is incorrect
theorem incorrect_statement_D (A : statement_A) (B : statement_B) (C : statement_C) : ¬statement_D := 
  sorry

end incorrect_statement_D_l668_668505


namespace savings_percentage_l668_668378

-- Define the conditions
variables (S : ℝ) -- last year's salary
def last_year_savings := 0.06 * S
def this_year_salary := 1.10 * S
def this_year_savings := 0.10 * this_year_salary

-- Define the statement to be proven
theorem savings_percentage (S : ℝ) (hS : S ≠ 0) :
  (this_year_savings S / last_year_savings S) * 100 = 183.33 :=
by
  unfold last_year_savings this_year_savings this_year_salary
  calc
    (0.10 * (1.10 * S) / (0.06 * S)) * 100
      = (0.11 * S / 0.06 * S) * 100   : by rw [mul_div_mul_left _ _ hS, ←mul_assoc]
  ... = (0.11 / 0.06) * 100           : by rw div_eq_div_iff
  ... = (11 / 6) * 100                : by norm_num
  ... = 183.33                        : by norm_num
  sorry

end savings_percentage_l668_668378


namespace calculate_g_neg_one_l668_668400

noncomputable def f (p q r : ℝ) : Polynomial ℝ := Polynomial.X ^ 3 - p * Polynomial.X ^ 2 + q * Polynomial.X - r

noncomputable def g (a b c : ℝ) : Polynomial ℝ :=
  (Polynomial.X - Polynomial.C (1 / a)) * (Polynomial.X - Polynomial.C (1 / b)) * (Polynomial.X - Polynomial.C (1 / c))

theorem calculate_g_neg_one (p q r : ℝ) (h_pos : 0 < p ∧ 0 < q ∧ 0 < r)
  (h_ineq : p < q ∧ q < r)
  (a b c : ℝ)
  (h_roots_f : (Polynomial.X - Polynomial.C a) * (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) =
    f p q r)
  (h_roots_g : (Polynomial.X - Polynomial.C (1 / a)) * (Polynomial.X - Polynomial.C (1 / b)) * (Polynomial.X - Polynomial.C (1 / c)) =
    g a b c) :
  g a b c .eval (-1) = (1 + p + q - r) / (-r) :=
sorry

end calculate_g_neg_one_l668_668400


namespace ellipse_range_2x_plus_y_l668_668041

theorem ellipse_range_2x_plus_y (x y : ℝ) (h : x^2 / 4 + y^2 = 1) : 
  -real.sqrt 17 ≤ 2 * x + y ∧ 2 * x + y ≤ real.sqrt 17 :=
sorry

end ellipse_range_2x_plus_y_l668_668041


namespace f_increasing_f_odd_function_l668_668688

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : ∀ (x1 x2 : ℝ), x1 < x2 → f a x1 < f a x2 :=
by
  sorry

theorem f_odd_function (a : ℝ) : f a 0 = 0 → (a = 1) :=
by
  sorry

end f_increasing_f_odd_function_l668_668688


namespace find_smaller_integer_l668_668095

theorem find_smaller_integer
  (x y : ℤ)
  (h1 : x + y = 30)
  (h2 : 2 * y = 5 * x - 10) :
  x = 10 :=
by
  -- proof would go here
  sorry

end find_smaller_integer_l668_668095


namespace range_of_quadratic_function_in_interval_l668_668840

noncomputable def quadratic_function : ℝ → ℝ := λ x, x^2 - 4*x + 3

theorem range_of_quadratic_function_in_interval :
  set.range (quadratic_function ∘ (λ x, x)) (set.Icc 1 4) = set.Icc (-1) 3 :=
sorry

end range_of_quadratic_function_in_interval_l668_668840


namespace max_apples_discarded_l668_668157

theorem max_apples_discarded (n : ℕ) : n % 7 ≤ 6 := by
  sorry

end max_apples_discarded_l668_668157


namespace bus_speed_excluding_stoppages_l668_668614

-- Definitions of the conditions
def speed_including_stoppages : ℝ := 42
def stoppage_time_per_hour_minutes : ℝ := 9.6
def stoppage_time_per_hour_hours : ℝ := stoppage_time_per_hour_minutes / 60
def moving_time_per_hour : ℝ := 1 - stoppage_time_per_hour_hours
def distance_covered_including_stoppages : ℝ := speed_including_stoppages * 1

-- Goal to prove
def speed_excluding_stoppages := distance_covered_including_stoppages / moving_time_per_hour

theorem bus_speed_excluding_stoppages : speed_excluding_stoppages = 50 := by
  sorry

end bus_speed_excluding_stoppages_l668_668614


namespace find_ab_solve_inequality_l668_668313

theorem find_ab (a b : ℝ) (h₁ : b/a = -2) (h₂ : 3/a = -3) : a = -1 ∧ b = 2 := 
sorry

theorem solve_inequality (x : ℝ) (h_a : a = -1) (h_b : b = 2) 
  : log b (2 * x - 1) ≤ 1 / (2^a) → 1 / 2 < x ∧ x ≤ 5 / 2 := 
sorry

end find_ab_solve_inequality_l668_668313


namespace largest_prime_factor_of_3913_l668_668488

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), nat.prime p ∧ p ∣ 3913 ∧ (∀ q, nat.prime q ∧ q ∣ 3913 → q ≤ p) ∧ p = 43 :=
sorry

end largest_prime_factor_of_3913_l668_668488


namespace sequence_properties_l668_668318

noncomputable def a (n : ℕ) : ℕ := 3 * n - 1

def S (n : ℕ) : ℕ := n * (2 + 3 * n - 1) / 2

theorem sequence_properties :
  a 5 + a 7 = 34 ∧ ∀ n, S n = (3 * n ^ 2 + n) / 2 :=
by
  sorry

end sequence_properties_l668_668318


namespace new_triangle_shape_l668_668276

theorem new_triangle_shape
  (a b c h : ℝ)
  (h₀ : a^2 + b^2 = c^2)
  (h₁ : a * b = c * h) :
  (c + h)^2 = (a + b)^2 + h^2 → right_triangle (c + h) (a + b) h := sorry

end new_triangle_shape_l668_668276


namespace liquid_level_ratio_l668_668112

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end liquid_level_ratio_l668_668112


namespace angle_between_a_and_d_l668_668465

variables (a b d : ℝ^3) (φ : ℝ)

-- Conditions
def norm_a : ∥a∥ = 1 := sorry
def norm_b : ∥b∥ = 1 := sorry
def norm_d : ∥d∥ = 3 := sorry
def vector_property : a × (a × d) + 2 • b = 0 := sorry

-- Statement
theorem angle_between_a_and_d : (φ = Real.arccos (sqrt 5 / 3)) ∨ (φ = Real.arccos (-sqrt 5 / 3)) :=
sorry

end angle_between_a_and_d_l668_668465


namespace largest_and_smallest_multiples_of_12_l668_668870

theorem largest_and_smallest_multiples_of_12 (k : ℤ) (n₁ n₂ : ℤ) (h₁ : k = -150) (h₂ : n₁ = -156) (h₃ : n₂ = -144) :
  (∃ m1 : ℤ, m1 * 12 = n₁ ∧ n₁ < k) ∧ (¬ (∃ m2 : ℤ, m2 * 12 = n₂ ∧ n₂ > k ∧ ∃ m2' : ℤ, m2' * 12 > k ∧ m2' * 12 < n₂)) :=
by
  sorry

end largest_and_smallest_multiples_of_12_l668_668870


namespace select_team_ways_l668_668032

theorem select_team_ways : 
    (∃ (boys girls total_students team_size : ℕ), boys = 7 ∧ girls = 9 ∧ total_students = boys + girls ∧ team_size = 5 ∧ 
    nat.choose total_students team_size = 4368) :=
  by
    use 7 -- boys
    use 9 -- girls
    use 16 -- total_students = boys + girls
    use 5 -- team_size
    simp
    sorry

end select_team_ways_l668_668032


namespace good_oranges_B_fraction_l668_668794

-- Define the conditions: 
/-- Total number of trees Salaria has -/
def total_trees : ℕ := 10

/-- Trees of type A -/
def tree_A_count : ℕ := total_trees / 2

/-- Trees of type B -/
def tree_B_count : ℕ := total_trees / 2

/-- Oranges per tree from tree A -/
def oranges_per_tree_A : ℕ := 10

/-- Fraction of good oranges from tree A -/
def fraction_good_oranges_A : ℝ := 0.6

/-- Good oranges from tree A per tree -/
def good_oranges_per_tree_A : ℝ := oranges_per_tree_A * fraction_good_oranges_A

/-- Total good oranges from tree A -/
def total_good_oranges_A : ℝ := tree_A_count * good_oranges_per_tree_A

/-- Total good oranges Salaria gets per month -/
def total_good_oranges : ℝ := 55

/-- Total good oranges from tree B -/
def total_good_oranges_B : ℝ := total_good_oranges - total_good_oranges_A

/-- Oranges per tree from tree B -/
def oranges_per_tree_B : ℕ := 15

/-- Total oranges from tree B -/
def total_oranges_B : ℝ := tree_B_count * oranges_per_tree_B 

/-- Fraction of good oranges from tree B -/
def fraction_good_oranges_B : ℝ := total_good_oranges_B / total_oranges_B

theorem good_oranges_B_fraction : fraction_good_oranges_B = (1/3) := 
by 
  -- Placeholder proof
  sorry

end good_oranges_B_fraction_l668_668794


namespace average_earning_week_l668_668817

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ) 
  (h1 : (D1 + D2 + D3 + D4) / 4 = 18)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 13) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 22.86 := 
by 
  sorry

end average_earning_week_l668_668817


namespace total_cost_of_vitamins_l668_668376

-- Definitions based on the conditions
def original_price : ℝ := 15.00
def discount_percentage : ℝ := 0.20
def coupon_value : ℝ := 2.00
def num_coupons : ℕ := 3
def num_bottles : ℕ := 3

-- Lean statement to prove the final cost
theorem total_cost_of_vitamins
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (coupon_value : ℝ)
  (num_coupons : ℕ)
  (num_bottles : ℕ)
  (discounted_price_per_bottle : ℝ := original_price * (1 - discount_percentage))
  (total_coupon_value : ℝ := coupon_value * num_coupons)
  (total_cost_before_coupons : ℝ := discounted_price_per_bottle * num_bottles) :
  (total_cost_before_coupons - total_coupon_value) = 30.00 :=
by
  sorry

end total_cost_of_vitamins_l668_668376


namespace ellipse_eccentricity_equilateral_triangle_l668_668713

theorem ellipse_eccentricity_equilateral_triangle
  (c a : ℝ) (h : c / a = 1 / 2) : eccentricity = 1 / 2 :=
by
  -- Proof goes here, we add sorry to skip proof content
  sorry

end ellipse_eccentricity_equilateral_triangle_l668_668713


namespace product_zero_probability_l668_668066

noncomputable def probability_product_is_zero : ℚ :=
  let S := [-3, -1, 0, 0, 2, 5]
  let total_ways := 15 -- Calculated as 6 choose 2 taking into account repetition
  let favorable_ways := 8 -- Calculated as (2 choose 1) * (4 choose 1)
  favorable_ways / total_ways

theorem product_zero_probability : probability_product_is_zero = 8 / 15 := by
  sorry

end product_zero_probability_l668_668066


namespace intersection_points_l668_668299

-- Definition of curve C by the polar equation
def curve_C (ρ : ℝ) (θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Definition of line l by the polar equation
def line_l (ρ : ℝ) (θ : ℝ) (m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = m

-- Proof statement that line l intersects curve C exactly once for specific values of m
theorem intersection_points (m : ℝ) : 
  (∀ ρ θ, curve_C ρ θ → line_l ρ θ m → ρ = 0 ∧ θ = 0) ↔ (m = -1/2 ∨ m = 3/2) :=
by
  sorry

end intersection_points_l668_668299


namespace total_marks_eq_300_second_candidate_percentage_l668_668527

-- Defining the conditions
def percentage_marks (total_marks : ℕ) : ℕ := 40
def fail_by (fail_marks : ℕ) : ℕ := 40
def passing_marks : ℕ := 160

-- The number of total marks in the exam computed from conditions
theorem total_marks_eq_300 : ∃ T, 0.40 * T = 120 :=
by
  use 300
  sorry

-- The percentage of marks the second candidate gets
theorem second_candidate_percentage : ∃ percent, percent = (180 / 300) * 100 :=
by
  use 60
  sorry

end total_marks_eq_300_second_candidate_percentage_l668_668527


namespace inradius_semicircle_relation_l668_668351

theorem inradius_semicircle_relation 
  (a b c : ℝ)
  (h_acute: a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (S : ℝ)
  (p : ℝ)
  (r : ℝ)
  (ra rb rc : ℝ)
  (h_def_semi_perim : p = (a + b + c) / 2)
  (h_area : S = p * r)
  (h_ra : ra = (2 * S) / (b + c))
  (h_rb : rb = (2 * S) / (a + c))
  (h_rc : rc = (2 * S) / (a + b)) :
  2 / r = 1 / ra + 1 / rb + 1 / rc :=
by
  sorry

end inradius_semicircle_relation_l668_668351


namespace problem001_l668_668807

theorem problem001
  (y : ℝ)
  (h : sqrt (4 + sqrt (3 * y - 5)) = sqrt 10) :
  y = 41 / 3 := by
  sorry

end problem001_l668_668807


namespace find_s_l668_668067

variable {t s : Real}

theorem find_s (h1 : t = 8 * s^2) (h2 : t = 4) : s = Real.sqrt 2 / 2 :=
by
  sorry

end find_s_l668_668067


namespace find_domain_f_sqrt_x_squared_minus_1_l668_668674

-- Define the conditions and proof problem statement
def domain_f_sqrt_x_squared_minus_1 (domain_f_2_pow_x : Set ℝ) : Set ℝ :=
  { x : ℝ | (-sqrt 17 < x ∧ x < -sqrt 5) ∨ (sqrt 5 < x ∧ x < sqrt 17) }

-- The initial given condition
axiom domain_f_2_pow_x : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

-- The theorem stating the equivalence with the given conditions
theorem find_domain_f_sqrt_x_squared_minus_1 :
  domain_f_sqrt_x_squared_minus_1 domain_f_2_pow_x = { x : ℝ | (-sqrt 17 < x ∧ x < -sqrt 5) ∨ (sqrt 5 < x ∧ x < sqrt 17) } :=
sorry

end find_domain_f_sqrt_x_squared_minus_1_l668_668674


namespace radius_of_circle_with_square_and_chord_l668_668243

theorem radius_of_circle_with_square_and_chord :
  ∃ (r : ℝ), 
    (∀ (chord_length square_side_length : ℝ), chord_length = 6 ∧ square_side_length = 2 → 
    (r = Real.sqrt 10)) :=
by
  sorry

end radius_of_circle_with_square_and_chord_l668_668243


namespace solve_absolute_value_equation_l668_668059

theorem solve_absolute_value_equation (y : ℝ) :
  (|y - 8| + 3 * y = 11) → (y = 1.5) :=
by
  sorry

end solve_absolute_value_equation_l668_668059


namespace interval_contains_root_l668_668447

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 / x

theorem interval_contains_root :
  ∃ ξ ∈ Ioo (3 / 2 : ℝ) 2, f ξ = 0 :=
begin
  sorry
end

end interval_contains_root_l668_668447


namespace sequence_value_a2017_l668_668457

noncomputable def a : ℕ → ℝ
| 1 := Real.sqrt 3
| (n + 1) := ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem sequence_value_a2017 : a 2017 = 3024 + Real.sqrt 3 := 
sorry

end sequence_value_a2017_l668_668457


namespace douglas_won_46_percent_in_county_Y_l668_668728

theorem douglas_won_46_percent_in_county_Y (V : ℝ) (P : ℝ) :
  let V_x := 2 * V,
      V_y := V,
      D_t := 0.58 * (V_x + V_y),
      D_x := 0.64 * V_x,
      D_y := (P / 100) * V_y in
  D_x + D_y = D_t → P = 46 :=
by
  dsimp,
  intro h,
  sorry

end douglas_won_46_percent_in_county_Y_l668_668728


namespace vasya_days_l668_668908

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668908


namespace isosceles_triangles_with_perimeter_27_l668_668703

def is_valid_isosceles_triangle (a b : ℕ) : Prop :=
  2 * a + b = 27 ∧ 2 * a > b

theorem isosceles_triangles_with_perimeter_27 :
  { t : ℕ × ℕ // is_valid_isosceles_triangle t.1 t.2 }.card = 7 := 
sorry

end isosceles_triangles_with_perimeter_27_l668_668703


namespace log_expression_evaluation_l668_668711

theorem log_expression_evaluation (k : ℝ) (hk : 0 < k) : 
  log 210 / log 10 + log k / log 10 - log 56 / log 10 + log 40 / log 10 - log 120 / log 10 + log 25 / log 10 = 2 := 
by
  sorry

end log_expression_evaluation_l668_668711


namespace sum_of_numbers_l668_668102

-- Define the conditions
variables (a b : ℝ) (r d : ℝ)
def geometric_progression := a = 3 * r ∧ b = 3 * r^2
def arithmetic_progression := b = a + d ∧ 9 = b + d

-- Define the problem as proving the sum of a and b
theorem sum_of_numbers (h1 : geometric_progression a b r)
                       (h2 : arithmetic_progression a b d) : 
  a + b = 45 / 4 :=
sorry

end sum_of_numbers_l668_668102


namespace series_equality_l668_668866

theorem series_equality (n: ℕ) : 
  (∑ i in range(n), (2*i+1)*(2*i+2)^2 - (2*i+2)*(2*i+3)^2) = -n*(n+1)*(4*n+3) :=
sorry

end series_equality_l668_668866


namespace common_ratio_of_arithmetic_sequence_l668_668283

theorem common_ratio_of_arithmetic_sequence (S_odd S_even : ℤ) (q : ℤ) 
  (h1 : S_odd + S_even = -240) (h2 : S_odd - S_even = 80) 
  (h3 : q = S_even / S_odd) : q = 2 := 
  sorry

end common_ratio_of_arithmetic_sequence_l668_668283


namespace vasya_did_not_buy_anything_days_l668_668943

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668943


namespace MN_perpendicular_PQ_l668_668401

noncomputable theory
open_locale classical

variables {A B C D M N P Q : Point}
variables (h_parallel: is_parallelogram A B C D)
variables (h_touch1: is_inscribed_triangle_circle_touch A B D M N)
variables (h_touch2: is_inscribed_triangle_circle_touch A C D P Q)

theorem MN_perpendicular_PQ : is_perpendicular (line_through M N) (line_through P Q) :=
sorry

end MN_perpendicular_PQ_l668_668401


namespace problem_1_problem_2_l668_668321

-- Define the vectors a and b
def vector_a (x : ℝ) (m : ℝ) := (Real.sin x, m * Real.cos x)
def vector_b := (3 : ℝ, -1 : ℝ)

-- Problem 1: Prove that 2 * sin^2(x) - 3 * cos^2(x) = 3/2 given the conditions
theorem problem_1 (x : ℝ) (h1 : vector_a x 1 = (Real.sin x, Real.cos x))
  (h_parallel : (Real.sin x, Real.cos x) = (k * 3, k * -1)) : 
  2 * (Real.sin x)^2 - 3 * (Real.cos x)^2 = 3 / 2 := 
sorry 

-- Problem 2: Prove the range of f(2x) on [pi/8, 2pi/3] given the conditions
theorem problem_2 (x m : ℝ) (h1 : vector_a x m = (Real.sin x, m * Real.cos x))
  (h_symmetric : f : ℝ → ℝ := λ x, vector_a x m.1 * vector_b.1 + vector_a x m.2 * vector_b.2
  (∀ x, f(x) = f(2 * π/3 - x))
  : 
(∀ x ∈ [π / 8, 2 * π/3], f(2 * x) ∈ [ -sqrt(3), 2 * sqrt(3) ] ∨ f(2 * x) ∈ [ -2 * sqrt(3), sqrt(3) ]) :=
sorry

end problem_1_problem_2_l668_668321


namespace factorial_division_l668_668294

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end factorial_division_l668_668294


namespace smallest_value_of_a_l668_668026

theorem smallest_value_of_a (a b c d : ℤ) (h1 : (a - 2 * b) > 0) (h2 : (b - 3 * c) > 0) (h3 : (c - 4 * d) > 0) (h4 : d > 100) : a ≥ 2433 := sorry

end smallest_value_of_a_l668_668026


namespace ordering_of_f_values_l668_668301

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 ^ x + 1 else (1 / 3) ^ x + 1

def even_function (x : ℝ) : Prop :=
  f (-x) = f x

def a := 2 ^ (4 / 3)
def b := 4 ^ (2 / 5)
def c := 25 ^ (1 / 3)

theorem ordering_of_f_values :
  even_function f →
  (∀ x < 0, f x = 3 ^ x + 1) →
  a = 2 ^ (4 / 3) →
  b = 4 ^ (2 / 5) →
  c = 25 ^ (1 / 3) →
  f c < f a ∧ f a < f b :=
by
  sorry

end ordering_of_f_values_l668_668301


namespace problem_equivalence_l668_668690

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def quadratic (m : ℝ) : ℝ → ℝ := λ x, m * x^2 + x - 1

theorem problem_equivalence :
  ∀ (m : ℝ), is_monotonically_increasing (quadratic m) (-1) (real.top) ↔ 0 ≤ m ∧ m ≤ (1 / 2) := 
sorry

end problem_equivalence_l668_668690


namespace sum_nine_terms_l668_668847

-- Definitions required based on conditions provided in Step a)
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- The arithmetic sequence condition is encapsulated here
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The definition of S_n being the sum of the first n terms
def sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- The given condition from the problem
def given_condition (a : ℕ → ℝ) : Prop :=
  2 * a 8 = 6 + a 1

-- The proof statement to show S_9 = 54 given the above conditions
theorem sum_nine_terms (h_arith : is_arithmetic_sequence a d)
                        (h_sum : sum_first_n a S) 
                        (h_given : given_condition a): 
                        S 9 = 54 :=
  by sorry

end sum_nine_terms_l668_668847


namespace find_nine_digit_number_l668_668616

-- Definitions based on the conditions
def valid_nine_digit_number (n : ℕ) : Prop :=
  let a₁a₂a₃ := n // 1000000 in
  let b₁b₂b₃ := (n % 1000000) // 1000 in
  let a₁a₂a₃′ := n % 1000 in
  a₁a₂a₃ = a₁a₂a₃′ ∧ b₁b₂b₃ = 2 * a₁a₂a₃ ∧ a₁a₂a₃ > 99 ∧ a₁a₂a₃ < 1000

def is_product_of_squares_of_primes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
    n = (p₁ * p₁) * (p₂ * p₂) * (p₃ * p₃) * (p₄ * p₄)

-- The theorem statement
theorem find_nine_digit_number :
  ∃ n : ℕ, valid_nine_digit_number n ∧ is_product_of_squares_of_primes n ∧ (n = 289578289 ∨ n = 361722361) :=
sorry

end find_nine_digit_number_l668_668616


namespace probability_even_product_l668_668257

theorem probability_even_product :
  let S := {1, 2, 3, 4} in
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b } in
  let even_product_pairs := { p ∈ pairs | (p.1 * p.2) % 2 = 0 } in
  (↑(even_product_pairs.card) / ↑(pairs.card) : ℝ) = 5 / 6 :=
by
  let S := {1, 2, 3, 4}
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b }
  let even_product_pairs := { p ∈ pairs | (p.1 * p.2) % 2 = 0 }
  sorry

end probability_even_product_l668_668257


namespace how_many_correct_statements_l668_668978

-- Definitions from conditions
def derivative1 : Prop := (∀ x : ℝ, (Real.sin x)' = -Real.cos x)
def derivative2 : Prop := (∀ x : ℝ, (Real.inv x)' = Real.inv (x^2))
def derivative3 : Prop := (∀ x : ℝ, (Real.log 3 x)' = inv (3 * Real.log (Real.exp 1) x))
def derivative4 : Prop := (∀ x : ℝ, (Real.log (Real.exp 1) x)' = inv x)

-- The proof problem
theorem how_many_correct_statements :
  ¬derivative1 ∧ ¬derivative2 ∧ ¬derivative3 ∧ derivative4 :=
sorry

end how_many_correct_statements_l668_668978


namespace LykaSavings_l668_668411

-- Define the given values and the properties
def totalCost : ℝ := 160
def amountWithLyka : ℝ := 40
def averageWeeksPerMonth : ℝ := 4.33
def numberOfMonths : ℝ := 2

-- Define the remaining amount Lyka needs
def remainingAmount : ℝ := totalCost - amountWithLyka

-- Define the number of weeks in the saving period
def numberOfWeeks : ℝ := numberOfMonths * averageWeeksPerMonth

-- Define the weekly saving amount
def weeklySaving : ℝ := remainingAmount / numberOfWeeks

-- State the theorem to be proved
theorem LykaSavings :  weeklySaving ≈ 13.86 :=
by
  -- Proof steps (omitted)
  sorry

end LykaSavings_l668_668411


namespace coefficient_x2_expansion_l668_668074

theorem coefficient_x2_expansion :
  let f := λ x : ℂ, (x - (1 / complex.sqrt x))^8 in
  (∃ c : ℂ, c * x^2 = @ereal.get ℂ ℝ_domain 
  (series.coeff (λ x, complex.coeff (x - 1/complex.sqrt x)^8) 2)) ↔ c = 70 :=
by
  sorry

end coefficient_x2_expansion_l668_668074


namespace area_cross_section_l668_668459

-- Define the pyramid and relevant points and measurements
structure Pyramid (a b : ℝ) :=
(base_side : ℝ := a)
(lateral_edge : ℝ := b)

-- Problem statement in Lean
theorem area_cross_section (a b : ℝ) :
  let pyramid := Pyramid.mk a b in
  ∃ area: ℝ, area = (5 * a * b * Real.sqrt 2) / 16 :=
by
  let pyramid := Pyramid.mk a b
  exists (5 * a * b * Real.sqrt 2 / 16)
  sorry

end area_cross_section_l668_668459


namespace simple_interest_rate_l668_668712

theorem simple_interest_rate (P : ℝ) (T : ℝ) (A : ℝ) (R : ℝ) (h : A = 3 * P) (h1 : T = 12) (h2 : A - P = (P * R * T) / 100) :
  R = 16.67 :=
by sorry

end simple_interest_rate_l668_668712


namespace exists_xy_interval_l668_668788

theorem exists_xy_interval (a b : ℝ) : 
  ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |x * y - a * x - b * y| ≥ 1 / 3 :=
sorry

end exists_xy_interval_l668_668788


namespace pentagon_perimeter_l668_668358

theorem pentagon_perimeter (A B C D E : Type) 
  (ABE BCE CDE : A → B → C → D → E → Prop)
  (right_angled_ABE : ∀ (A B E : ℝ), ABE A B E → angle AEB = 45)
  (right_angled_BCE : ∀ (B C E : ℝ), BCE B C E → angle BEC = 60)
  (right_angled_CDE : ∀ (C D E : ℝ), CDE C D E → angle CED = 45)
  (AE : ℝ) (hAE : AE = 40)
  (AB : ℝ) (hAB : AB = AE)
  (BC : ℝ) (hBC : BC = (40 * Real.sqrt 3) / 3)
  (CD : ℝ) (hCD : CD = 20)
  (DE : ℝ) (hDE : DE = 20) :
  40 + (40 * Real.sqrt 3) / 3 + 20 + 20 + 40 = 140 + (40 * Real.sqrt 3) / 3 := 
sorry

end pentagon_perimeter_l668_668358


namespace vasya_days_without_purchase_l668_668927

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668927


namespace vasya_no_purchase_days_l668_668951

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668951


namespace polynomial_positivity_intervals_l668_668212

theorem polynomial_positivity_intervals (x : ℝ) :
  ((x + 1) * (x - 1) * (x - 3) > 0) ↔ ((-1 < x ∧ x < 1) ∨ (3 < x)) :=
begin
  sorry -- proof not required
end

end polynomial_positivity_intervals_l668_668212


namespace sphere_volume_eq_div7sqrt14div3pi_l668_668450

theorem sphere_volume_eq_div7sqrt14div3pi :
  let l := 3
      w := 2
      h := 1
      diagonal := Real.sqrt (l^2 + w^2 + h^2)
      R := diagonal / 2
      V := (4 / 3) * Real.pi * R^3
  in V = (7 * Real.sqrt 14 / 3) * Real.pi :=
by
  let l := 3
  let w := 2
  let h := 1
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let R := diagonal / 2
  let V := (4 / 3) * Real.pi * R^3
  have h1: diagonal = Real.sqrt 14 := by sorry
  have h2: R = Real.sqrt 14 / 2 := by sorry
  have h3: V = 7 * Real.sqrt 14 / 3 * Real.pi := by sorry
  exact h3

end sphere_volume_eq_div7sqrt14div3pi_l668_668450


namespace max_profit_l668_668967

def fixed_cost : ℝ := 2.5
def selling_price_per_thousand_units : ℝ := 50

def additional_cost (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

def revenue (x : ℝ) : ℝ :=
  selling_price_per_thousand_units * x

def annual_profit (x : ℝ) : ℝ :=
  revenue(x) - fixed_cost - additional_cost(x)

theorem max_profit :
  ∃ x : ℝ, x = 100 ∧ annual_profit 100 = 1000 :=
by
  sorry

end max_profit_l668_668967


namespace find_ordered_pairs_l668_668052

theorem find_ordered_pairs (w l : ℕ) (hl_pos : l > 0) (hw_pos : w > 0) (h_area : w * l = 18) :
  (w, l) ∈ {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
sorry

end find_ordered_pairs_l668_668052


namespace solution_to_equation_l668_668058

theorem solution_to_equation:
  ∃ (x : ℝ), x = Real.log 4 / Real.log 4 ∧ (4^x - 4 * 3^x + 9 = 0) :=
by
  use Real.log 4 / Real.log 4
  split
  { refl }
  { sorry }

end solution_to_equation_l668_668058


namespace hyperbola_correct_eqn_l668_668676

open Real

def hyperbola_eqn (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

theorem hyperbola_correct_eqn (e c a b x y : ℝ)
  (h_eccentricity : e = 2)
  (h_foci_distance : c = 4)
  (h_major_axis_half_length : a = 2)
  (h_minor_axis_half_length_square : b^2 = c^2 - a^2) :
  hyperbola_eqn x y :=
by
  sorry

end hyperbola_correct_eqn_l668_668676


namespace quadratic_inequality_solution_l668_668254

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40 * z + 340 ≤ 4 ↔ 12 ≤ z ∧ z ≤ 28 := by 
  sorry

end quadratic_inequality_solution_l668_668254


namespace vasya_did_not_buy_anything_days_l668_668941

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668941


namespace num_arrangements_l668_668854

theorem num_arrangements : 
  let grid := λ (i j : ℕ), 
    if i = 3 ∧ j = 3 then 'A' 
    else if i = 1 ∧ j = 2 then 'B' 
    else '_'
  in 
  ∃ (arrangements : list (ℕ × ℕ → char)), 
    (∀ arr ∈ arrangements,
      (∀ i, (∃ j, arr (i, j) = 'A') ∧ (∃ j, arr (i, j) = 'B') ∧ (∃ j, arr (i, j) = 'C')) ∧ 
      (∀ j, (∃ i, arr (i, j) = 'A') ∧ (∃ i, arr (i, j) = 'B') ∧ (∃ i, arr (i, j) = 'C')) ∧
      arr (3,3) = 'A' ∧ arr (1,2) = 'B') ∧
    arrangements.length = 4 :=
sorry

end num_arrangements_l668_668854


namespace find_vectors_l668_668969

open Matrix

noncomputable def r (a d : Matrix (Fin 3) (Fin 1) ℝ) (t : ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  a + t • d

def matrix1 := !![2; 6; 16] : Matrix (Fin 3) (Fin 1) ℝ
def matrix2 := !![0; -1; -2] : Matrix (Fin 3) (Fin 1) ℝ
def matrix3 := !![-2; -8; -18] : Matrix (Fin 3) (Fin 1) ℝ
def zero_time_vector := !![2/3; 4/3; 4] : Matrix (Fin 3) (Fin 1) ℝ
def five_time_vector := !![-8; -19; -26] : Matrix (Fin 3) (Fin 1) ℝ

theorem find_vectors (a d : Matrix (Fin 3) (Fin 1) ℝ) :
  (r a d (-2) = matrix1 ∧
   r a d 1 = matrix2 ∧
   r a d 4 = matrix3) →
  (r a d 0 = zero_time_vector ∧
   r a d 5 = five_time_vector) :=
by
  sorry

end find_vectors_l668_668969


namespace emily_annual_income_l668_668720

variables {q I : ℝ}

theorem emily_annual_income (h1 : (0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) = ((q + 0.75) * 0.01 * I)) : 
  I = 40000 := 
by
  sorry

end emily_annual_income_l668_668720


namespace prob_two_blue_balls_l668_668729

-- Ball and Urn Definitions
def total_balls : ℕ := 10
def blue_balls_initial : ℕ := 6
def red_balls_initial : ℕ := 4

-- Probabilities
def prob_blue_first_draw : ℚ := blue_balls_initial / total_balls
def prob_blue_second_draw_given_first_blue : ℚ :=
  (blue_balls_initial - 1) / (total_balls - 1)

-- Resulting Probability
def prob_both_blue : ℚ := prob_blue_first_draw * prob_blue_second_draw_given_first_blue

-- Statement to Prove
theorem prob_two_blue_balls :
  prob_both_blue = 1 / 3 :=
by
  sorry

end prob_two_blue_balls_l668_668729


namespace convert_binary_to_decimal_l668_668997

noncomputable def binary_to_decimal : ℕ → ℚ
| 0 := 0
| n := if n % 10 == 1 then binary_to_decimal (n / 10) * 2 + 1 else binary_to_decimal (n / 10) * 2

theorem convert_binary_to_decimal (n : ℚ) : n = 1 * 2^2 + 1 * 2^1 + 1 * 2^0 + 1 * 2^(-1) + 1 * 2^(-2) → n = 7.75 :=
by
  sorry

end convert_binary_to_decimal_l668_668997


namespace draw_balls_l668_668958

theorem draw_balls :
  let balls := 16
  let choices := [16, 15, 14, 13]
  (choices.foldr (*) 1) = 43680 :=
by
  let balls := 16
  let choices := [16, 15, 14, 13]
  sorry

end draw_balls_l668_668958


namespace vasya_days_without_purchases_l668_668900

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668900


namespace find_a_for_unextended_spring_twice_l668_668098

theorem find_a_for_unextended_spring_twice (a : ℝ) (h1 : (0 < a) ∧ (a < 2)) :
  (4*a^2 - 8*a + 1 ≥ 0) → (1 + 2*a > 2) → (1 + a - 2*a - 1 < 0) → ((0 < (2 * a - 1) / (2 * a)) ∧ ((2 * a - 1) / (2 * a) < 1)) → (a ∈ (1 + (real.sqrt 3) / 2, 2)) :=
by
  intros h2 h3 h4 h5
  sorry  

end find_a_for_unextended_spring_twice_l668_668098


namespace graph_description_l668_668130

theorem graph_description : 
  ∀ (x y : ℝ), (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ (l1 l2 l3 : ℝ → ℝ), 
    l1 = (λ x, -x) ∧
    l2 = (λ x, x) ∧
    l3 = (λ x, -x - 2) ∧
    ∀ (p1 p2 : ℝ × ℝ), (p1 ∈ {x | l1 x = l2 x} ∧ p2 ∈ {x | l1 x = l3 x} ∨ p1 ∈ {x | l2 x = l3 x}) → p1 ≠ p2) :=
sorry

end graph_description_l668_668130


namespace shaded_percentage_l668_668123

-- Define the properties of the 6x6 grid and the checkerboard pattern
def is_shaded (i j : ℕ) : Prop :=
  (i + j) % 2 = 1

def total_squares := 6 * 6

def shaded_squares : ℕ :=
  (List.natRange 6).bind (λ i => (List.natRange 6).filter (is_shaded i)).length

theorem shaded_percentage :
  (shaded_squares : ℚ) / total_squares * 100 = 50 := by
  sorry

end shaded_percentage_l668_668123


namespace number_of_truthful_dwarfs_l668_668595

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l668_668595


namespace max_abs_diff_eq_4_div_27_l668_668558

def f (x : ℝ) := x ^ 2
def g (x : ℝ) := x ^ 3

theorem max_abs_diff_eq_4_div_27 :
  ∃ x ∈ Icc (0 : ℝ) 1, |f x - g x| = (4 / 27) ∧ ∀ y ∈ Icc (0 : ℝ) 1, |f y - g y| ≤ (4 / 27) :=
by
  sorry

end max_abs_diff_eq_4_div_27_l668_668558


namespace dice_probability_exactly_two_threes_is_approx_0_l668_668230

def dice_probability := 
  let numDice := 8
  let target := 2
  let successProb := 1 / 6
  let failProb := 5 / 6
  let binomial := Nat.choose numDice target
  let successTerm := successProb ^ target
  let failTerm := failProb ^ (numDice - target)
  binomial * successTerm * failTerm

theorem dice_probability_exactly_two_threes_is_approx_0.780 :
  abs (dice_probability - 0.780) < 0.001 :=
by sorry

end dice_probability_exactly_two_threes_is_approx_0_l668_668230


namespace alternating_exponents_inequality_l668_668335

theorem alternating_exponents_inequality :
  let exp2 := (2, 3)
  let expr1 := nat.rec 2 (λ _ e, e ^ exp2.snd) (fin.mk 99 sorry)
  let expr2 := nat.rec 3 (λ _ e, e ^ exp2.fst) (fin.mk 99 sorry)
  expr1 > expr2 := 
sorry

end alternating_exponents_inequality_l668_668335


namespace sixth_ingot_positions_l668_668744

def position1 := 1
def position2 := 161
def position3 := (1 + 161) / 2
def position4 := (1 + (1 + 161) / 2) / 2
def position5 := ((1 + (1 + 161) / 2) / 2 + (1 + 161) / 2) / 2

def possible_sixth_positions := {21, 101, 141}
def storage_positions := {position1, position2, position3, position4, position5}

theorem sixth_ingot_positions :
  ∃ x ∈ possible_sixth_positions, ∀ y ∈ storage_positions, abs (x - y) ≥ 20 :=
sorry

end sixth_ingot_positions_l668_668744


namespace total_pumpkins_l668_668797

-- Define the number of pumpkins grown by Sandy and Mike
def pumpkinsSandy : ℕ := 51
def pumpkinsMike : ℕ := 23

-- Prove that their total is 74
theorem total_pumpkins : pumpkinsSandy + pumpkinsMike = 74 := by
  sorry

end total_pumpkins_l668_668797


namespace determine_m_l668_668659

def P : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def M (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}

theorem determine_m (m : ℝ) : M m ⊆ P ↔ m ∈ {1/2, 1/3, 0} := by
  sorry

end determine_m_l668_668659


namespace marble_slab_cut_l668_668156

def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

def greatest_and_least_perimeter_difference : ℝ :=
  let rect1 := (4.5, 6) -- 4.5 x 6 pieces
  let rect2 := (3, 9)   -- 3 x 9 pieces
  let rect3 := (3, 4.5) -- 3 x 4.5 pieces
  let perim1 := rectangle_perimeter rect1.1 rect1.2
  let perim2 := rectangle_perimeter rect2.1 rect2.2
  let perim3 := rectangle_perimeter rect3.1 rect3.2
  max perim1 (max perim2 perim3) - min perim1 (min perim2 perim3)

theorem marble_slab_cut :
  greatest_and_least_perimeter_difference = 9 :=
sorry

end marble_slab_cut_l668_668156


namespace sin_double_angle_l668_668261

open Real

theorem sin_double_angle
  {α : ℝ} (h1: tan α = -1/2) (h2: 0 < α ∧ α < π) :
  sin (2 * α) = -4/5 :=
sorry

end sin_double_angle_l668_668261


namespace vasya_days_without_purchase_l668_668922

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668922


namespace tangent_slope_angle_l668_668679

open Real

theorem tangent_slope_angle (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_lim : tendsto (λ Δx : ℝ, (f (1 + Δx) - f 1) / Δx) (𝓝 0) (𝓝 1)) :
  ∃ θ : ℝ, θ = 45 ∧ tan θ = 1 :=
by
  -- This is a placeholder proof to ensure the statement is syntactically correct
  sorry

end tangent_slope_angle_l668_668679


namespace polyhedron_faces_vertices_l668_668279

theorem polyhedron_faces_vertices (E : ℕ) (F : ℕ) (V : ℕ) 
  (hE : E = 30) 
  (h_faces : ∀ (f : ℕ), f = 5) 
  (h_polyhedron : ∀ (f : ℕ), f ∈ (30 / 2)) :
  F = 12 ∧ V = 20 := 
by
  sorry

end polyhedron_faces_vertices_l668_668279


namespace sum_of_integers_l668_668858

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 343000)
  (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
  a + b + c = 476 :=
by
  sorry

end sum_of_integers_l668_668858


namespace max_pieces_l668_668871

namespace CakeProblem

-- Define the dimensions of the cake and the pieces.
def cake_side : ℕ := 16
def piece_side : ℕ := 4

-- Define the areas of the cake and the pieces.
def cake_area : ℕ := cake_side * cake_side
def piece_area : ℕ := piece_side * piece_side

-- State the main problem to prove.
theorem max_pieces : cake_area / piece_area = 16 :=
by
  -- The proof is omitted.
  sorry

end CakeProblem

end max_pieces_l668_668871


namespace vasya_days_without_purchases_l668_668903

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668903


namespace vasya_did_not_buy_anything_days_l668_668937

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668937


namespace runners_meet_at_1200_l668_668860

def runners_meet (t : ℕ) : Prop :=
  let s1 := 0
  let s2 := 100
  let s3 := 300
  let v1 := 5
  let v2 := 5.5
  let v3 := 6
  let track_length := 600
  (5 * t % track_length) = ((100 + 5.5 * t) % track_length) ∧
  (5 * t % track_length) = ((300 + 6 * t) % track_length)

theorem runners_meet_at_1200 : ∃ t, runners_meet t ∧ t = 1200 := 
by
  sorry

end runners_meet_at_1200_l668_668860


namespace sum_of_digits_of_n_equals_10_l668_668809

theorem sum_of_digits_of_n_equals_10 (n : ℕ) (h1 : 0 < n) (h2 : (n+1)! + (n+2)! = n! * 474) : 
  n.digits.sum = 10 :=
sorry

end sum_of_digits_of_n_equals_10_l668_668809


namespace points_concyclic_C_D_G_H_l668_668734

-- Define points and conditions
variables {A B O C D G H : Point}
variables {α β : Angle}

-- Given conditions
axiom acute_angle_AOB : acute (∠ A O B)
axiom angle_AOC_eq_angle_DOB : ∠ A O C = ∠ D O B
axiom perpendicular_D_to_OA_at_G : is_perpendicular D G O A
axiom perpendicular_C_to_OB_at_H : is_perpendicular C H O B

-- Main statement to prove
theorem points_concyclic_C_D_G_H :
  concyclic {C, D, G, H} :=
by sorry

end points_concyclic_C_D_G_H_l668_668734


namespace mustang_ratio_l668_668051

variables (F S M : ℝ)

theorem mustang_ratio (hF : F = 240) (hS : S = 12) (hRelation : S = (1 / 2) * M) : (M / F) = (1 / 10) :=
by {
  have hM : M = 2 * S,
  { rw ←hRelation,
    ring, },
  rw [hM, hS, hF],
  norm_num,
}

end mustang_ratio_l668_668051


namespace product_of_areas_eq_square_of_volume_l668_668177

-- define the dimensions of the prism
variables (x y z : ℝ)

-- define the areas of the faces as conditions
def top_area := x * y
def back_area := y * z
def lateral_face_area := z * x

-- define the product of the areas of the top, back, and one lateral face
def product_of_areas := (top_area x y) * (back_area y z) * (lateral_face_area z x)

-- define the volume of the prism
def volume := x * y * z

-- theorem to prove: product of areas equals square of the volume
theorem product_of_areas_eq_square_of_volume 
  (ht: top_area x y = x * y)
  (hb: back_area y z = y * z)
  (hl: lateral_face_area z x = z * x) :
  product_of_areas x y z = (volume x y z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l668_668177


namespace a_1414_value_l668_668093
noncomputable theory

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1111
  else if n = 2 then 1212
  else if n = 3 then 1313
  else |seq (n - 1) - seq (n - 2)| + |seq (n - 2) - seq (n - 3)|

theorem a_1414_value :
  seq 1414 = 1 :=
sorry

end a_1414_value_l668_668093


namespace angle_between_focus_and_minor_axis_l668_668070

-- Define the conditions for the ellipse
def condition_1 (a c : ℝ) : Prop := ∀ (d : ℝ), d = 2 * a^2 / c
def condition_2 (c : ℝ) : Prop := ∀ (f : ℝ), f = 2 * c

-- The proof problem statement in Lean 4
theorem angle_between_focus_and_minor_axis 
  (a c : ℝ) (h1 : condition_1 a c) (h2 : condition_2 c) : 
  ∠ (line_focus_minor a c) (major_axis a c) = 90 :=
sorry

end angle_between_focus_and_minor_axis_l668_668070


namespace number_of_truthful_gnomes_l668_668608

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l668_668608


namespace vasya_purchase_l668_668934

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668934


namespace speed_in_still_water_correct_l668_668170

variables (upstream_speed downstream_speed : ℝ)
def speed_in_still_water (u d : ℝ) := (u + d) / 2

theorem speed_in_still_water_correct : 
  upstream_speed = 35 ∧ downstream_speed = 45 → speed_in_still_water upstream_speed downstream_speed = 40 :=
by
  intros h
  cases h with hu hd
  rw [hu, hd]
  unfold speed_in_still_water
  norm_num

end speed_in_still_water_correct_l668_668170


namespace Traci_trip_fraction_l668_668783

theorem Traci_trip_fraction :
  let total_distance := 600
  let first_stop_distance := total_distance / 3
  let remaining_distance_after_first_stop := total_distance - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  (distance_between_stops / remaining_distance_after_first_stop) = 1 / 4 :=
by
  let total_distance := 600
  let first_stop_distance := 600 / 3
  let remaining_distance_after_first_stop := 600 - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  have h1 : total_distance = 600 := by exact rfl
  have h2 : first_stop_distance = 200 := by norm_num [first_stop_distance]
  have h3 : remaining_distance_after_first_stop = 400 := by norm_num [remaining_distance_after_first_stop]
  have h4 : distance_between_stops = 100 := by norm_num [distance_between_stops]
  show (distance_between_stops / remaining_distance_after_first_stop) = 1/4
  -- Proof omitted
  sorry

end Traci_trip_fraction_l668_668783


namespace number_of_two_digit_numbers_satisfying_condition_l668_668210

-- Definitions: two-digit number and conditions involving it
def is_two_digit_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def satisfies_condition (a b : ℕ) : Prop :=
  (13 * a + 4 * b) % 10 = 7

-- Main theorem: Prove the number of pairs (a, b) satisfying the conditions is 8
theorem number_of_two_digit_numbers_satisfying_condition : 
  (finset.univ.filter (λ p : ℕ × ℕ, 
    is_two_digit_pair p.1 p.2 ∧ satisfies_condition p.1 p.2)).card = 8 :=
sorry

end number_of_two_digit_numbers_satisfying_condition_l668_668210


namespace no_integer_solutions_l668_668045

def equation1 (x y : ℤ) : Prop := x^6 + x^3 + x^3 * y + y = 147^157
def equation2 (x y z : ℤ) : Prop := x^3 + x^3 * y + y^2 + y + z^9 = 157 ^ 1177

theorem no_integer_solutions : ¬ ∃ x y z : ℤ, equation1 x y ∧ equation2 x y z :=
by
  assume h : ∃ x y z : ℤ, equation1 x y ∧ equation2 x y z
  sorry

end no_integer_solutions_l668_668045


namespace solve_equation_l668_668801

theorem solve_equation (x : ℝ) (h : x = 1) : 
  (x^2 + 3 * x + 4) / (x^2 - 3 * x + 2) = x + 6 :=
by
  rw h
  norm_num

end solve_equation_l668_668801


namespace flag_raising_arrangements_l668_668529

theorem flag_raising_arrangements (n_first_year_classes : ℕ) (n_second_year_classes : ℕ)
  (h1 : n_first_year_classes = 8) (h2 : n_second_year_classes = 6) :
  n_first_year_classes + n_second_year_classes = 14 :=
by
  rw [h1, h2]
  rfl

end flag_raising_arrangements_l668_668529


namespace coordinates_of_A_l668_668730

-- Definitions based on conditions
def origin : ℝ × ℝ := (0, 0)
def similarity_ratio : ℝ := 2
def point_A : ℝ × ℝ := (2, 3)
def point_A' (P : ℝ × ℝ) : Prop :=
  P = (similarity_ratio * point_A.1, similarity_ratio * point_A.2) ∨
  P = (-similarity_ratio * point_A.1, -similarity_ratio * point_A.2)

-- Statement of the theorem
theorem coordinates_of_A' :
  ∃ P : ℝ × ℝ, point_A' P :=
by
  use (4, 6)
  left
  sorry

end coordinates_of_A_l668_668730


namespace original_cost_of_car_l668_668426

theorem original_cost_of_car (C : ℝ) 
  (repair_cost : ℝ := 15000)
  (selling_price : ℝ := 64900)
  (profit_percent : ℝ := 13.859649122807017) :
  C = 43837.21 :=
by
  have h1 : C + repair_cost = selling_price - (selling_price - (C + repair_cost)) := by sorry
  have h2 : profit_percent / 100 = (selling_price - (C + repair_cost)) / C := by sorry
  have h3 : C = 43837.21 := by sorry
  exact h3

end original_cost_of_car_l668_668426


namespace repeating_decimal_to_fraction_l668_668237

theorem repeating_decimal_to_fraction : 
∀ (x : ℝ), x = 4 + (0.0036 / (1 - 0.01)) → x = 144/33 :=
by
  intro x hx
  -- This is a placeholder where the conversion proof would go.
  sorry

end repeating_decimal_to_fraction_l668_668237


namespace product_eq_neg_one_l668_668540

theorem product_eq_neg_one (m b : ℚ) (hm : m = -2 / 3) (hb : b = 3 / 2) : m * b = -1 :=
by
  rw [hm, hb]
  sorry

end product_eq_neg_one_l668_668540


namespace find_number_of_soldiers_joining_l668_668136

-- Definition of the initial conditions
def initial_soldiers : ℕ := 1200
def initial_consumption_per_soldier : ℚ := 3
def initial_days : ℕ := 30

-- Definition of the conditions after additional soldiers join
def new_consumption_per_soldier : ℚ := 2.5
def new_days : ℕ := 25

-- The number of soldiers joining (to be proven)
def soldiers_joining : ℚ := 528

-- The statement to be proven
theorem find_number_of_soldiers_joining :
  let total_provisions := initial_soldiers * initial_consumption_per_soldier * initial_days in
  total_provisions = (initial_soldiers + soldiers_joining) * new_consumption_per_soldier * new_days :=
by
  -- here should be the proof steps
  sorry

end find_number_of_soldiers_joining_l668_668136


namespace complex_square_l668_668332

theorem complex_square (z : ℂ) (i : ℂ) (h : i ^ 2 = -1) (hz : z = 5 + 2 * i) : z ^ 2 = 21 + 20 * i := by
  sorry

end complex_square_l668_668332


namespace nonneg_sets_property_l668_668238

open Set Nat

theorem nonneg_sets_property (A : Set ℕ) :
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔
  (A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = { n | 0 ≤ n }) :=
sorry

end nonneg_sets_property_l668_668238


namespace period_f_interval_decreasing_value_of_a_l668_668699

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x) + 2, cos x)
def vec_n (x : ℝ) : ℝ × ℝ := (1, 2 * cos x)
def f (x : ℝ) : ℝ := (vec_m x).fst * (vec_n x).fst + (vec_m x).snd * (vec_n x).snd
def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ := 0.5 * b * c * sin A
def cos_law (a b c : ℝ) (A : ℝ) : ℝ := b^2 + c^2 - 2 * b * c * cos A

-- 1) Prove the smallest positive period of f(x) is π and interval of monotonic decreasing
theorem period_f : ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem interval_decreasing : ∀ k : ℤ, ∀ x : ℝ, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 →
  ∀ y : ℝ, k * π + π / 6 ≤ y ∧ y ≤ k * π + 2 * π / 3 → x < y → f x > f y :=
by sorry

-- 2) Prove the value of a in triangle ABC
theorem value_of_a (A : ℝ) (b : ℝ) (area : ℝ) (a : ℝ) :
  f A = 4 ∧ b = 1 ∧ area = sqrt 3 / 2 → a = sqrt 3 :=
by
  intros h
  have h1 : f A = 4 := h.1
  have h2 : b = 1 := h.2.1
  have h3 : area = sqrt 3 / 2 := h.2.2
  sorry

end period_f_interval_decreasing_value_of_a_l668_668699


namespace percentage_increase_from_retail_to_customer_l668_668536

variable {C : ℝ} (hC : C > 0)

def retail_price := 1.40 * C
def customer_price := 1.82 * C

theorem percentage_increase_from_retail_to_customer :
  100 * (customer_price C - retail_price C) / retail_price C = 30 :=
by
  sorry

end percentage_increase_from_retail_to_customer_l668_668536


namespace square_position_2023_l668_668443

def initial_square := "ABCD"
def transformation_sequence :=
  ["ABCD", "DABC", "CBAD", "DCBA"]

def find_square_position (n : ℕ) : String :=
  transformation_sequence[(n % 3 + 1) % 4]

theorem square_position_2023 :
  find_square_position 2023 = "DABC" :=
by
  sorry

end square_position_2023_l668_668443


namespace find_z_modulus_ω_l668_668646

noncomputable def z (a: ℝ) : ℂ := a + complex.I

theorem find_z (a: ℝ) (h_imaginary: ∃ c: ℝ, ((1 + 2 * complex.I) * z a) = c * complex.I) : z a = 2 + complex.I := by
  sorry

noncomputable def ω : ℂ := (2 + complex.I) / (2 - complex.I)

theorem modulus_ω : complex.abs ω = 1 := by
  sorry

end find_z_modulus_ω_l668_668646


namespace find_roots_l668_668398

-- Given conditions
def f (x: ℝ) : ℝ := -3 * (x - 5)^2 + 45 * (x - 5) - 108

-- Define the roots given the conditions
theorem find_roots (x1 x2: ℝ) (hf: ∀ x, f x = -3 * x^2 + 45 * x - 108) :
  (f x1 = 0 ∧ f x2 = 0) ∧ (x1 = 7 ∧ x2 = -2) :=
by
  -- skipping proof steps using sorry
  sorry

end find_roots_l668_668398


namespace find_a_b_max_b_over_2_plus_a_l668_668346

-- Definitions of the conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables h1 : c = 2
variables h2 : C = π / 3
variables h3 : 1 / 2 * a * b * Real.sin C = Real.sqrt 3 

-- The first part of the theorem
theorem find_a_b 
  (h4 : a^2 + b^2 - a * b = 4) 
  (h5 : a * b = 4) : a = 2 ∧ b = 2 :=
sorry

-- Additional definitions and properties for the second part
variables h6 : ∀ B : ℝ, (2 * a * b * (1 / 2) * Real.sin B = Real.sqrt 3 → B = acos (1 - (a^2 + b^2 - c^2) / (2 * a * b))) 
variables {R : ℝ} 

-- The second part of the theorem
theorem max_b_over_2_plus_a : (2 * R * ((Real.sin B) / 2 + Real.sin (B + π / 3))) = (2 * Real.sqrt 21 / 3) :=
sorry

end find_a_b_max_b_over_2_plus_a_l668_668346


namespace david_completes_square_l668_668218

theorem david_completes_square (x : ℝ) :
  let p := -2 in
  let q := 104 in
  (4 * x^2 - 16 * x - 400 = 0 → (x + p)^2 = q) :=
by
  sorry

end david_completes_square_l668_668218


namespace correct_relationship_not_in_options_l668_668329

/-!
Given that P (A + B) = 1, prove that the correct relationship between events A and B
is not necessarily mutually exclusive, complementary, or other specified conditions.
-/

variables (A B : Prop)

-- Define the probability function P
def P (X : Prop) : Prop := sorry -- Placeholder definition, since we are not providing the proof

theorem correct_relationship_not_in_options
  (h : P (A ∨ B) = 1) :
  ¬ (P (A ∨ B) = P A + P B ∨ P (A ∨ B) = P A + P B - P (A ∧ B)) :=
begin
  sorry -- Placeholder for the logic to prove the theorem
end

end correct_relationship_not_in_options_l668_668329


namespace smallest_positive_period_of_f_l668_668247

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

theorem smallest_positive_period_of_f :
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi / 2 :=
by
  sorry

end smallest_positive_period_of_f_l668_668247


namespace fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l668_668542

theorem fourth_vertex_of_regular_tetrahedron_exists_and_is_unique :
  ∃ (x y z : ℤ),
    (x, y, z) ≠ (1, 2, 3) ∧ (x, y, z) ≠ (5, 3, 2) ∧ (x, y, z) ≠ (4, 2, 6) ∧
    (x - 1)^2 + (y - 2)^2 + (z - 3)^2 = 18 ∧
    (x - 5)^2 + (y - 3)^2 + (z - 2)^2 = 18 ∧
    (x - 4)^2 + (y - 2)^2 + (z - 6)^2 = 18 ∧
    (x, y, z) = (2, 3, 5) :=
by
  -- Proof goes here
  sorry

end fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l668_668542


namespace survey_students_count_l668_668350

noncomputable def total_students : ℕ :=
  let A := 250
  let B := 100
  let AB := 0.20 * A
  let only_A := A - AB
  let only_B := B - AB
  (only_A + only_B + AB).to_nat

theorem survey_students_count :
  (∀ A B AB, AB = 0.20 * A → AB = 0.50 * B → (A - AB) - (B - AB) = 150 → 
   A = 250 ∧ B = 100 ∧ AB = 50 ∧ (A - AB) + (B - AB) + AB = 300) :=
sorry

end survey_students_count_l668_668350


namespace players_no_collision_valid_choice_count_l668_668637

-- Define the vertices and players
inductive Vertex : Type
| A1 | A2 | A3 | A4
deriving DecidableEq, Repr

inductive Player : Type
| P1 | P2 | P3 | P4
deriving DecidableEq, Repr

-- Function to represent a player's choice of vertex
def PlayerChoice := Player → Vertex

-- Condition where no two players can choose the same vertex
def validChoices (choices : PlayerChoice) : Prop :=
  ∀ (p1 p2 : Player), p1 ≠ p2 → choices p1 ≠ choices p2

-- Counting the valid player choices
noncomputable def countValidChoices : Nat :=
  -- Since the players P1, P2, P3, P4 are fixed at vertices A1, A2, A3, A4 respectively,
  -- we need to count the number of ways such that none of the choices lead to a collision.
  11 -- Based on the problem's solution

-- The main theorem statement
theorem players_no_collision_valid_choice_count :
  ∃ (choices : PlayerChoice), validChoices choices ∧ countValidChoices = 11 :=
begin
  sorry
end

end players_no_collision_valid_choice_count_l668_668637


namespace no_segment_equal_angle_in_regular_pentagon_l668_668747

theorem no_segment_equal_angle_in_regular_pentagon (P : Type)
  [f : fintype P] [decidable_eq P] (hP : ∀ (A B C : P), 
  ∃ (c : P), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
  is_regular_pentagon P ∧ (∃ (X Y : P), segment_viewed_from_all_vertices_same_angle (X Y : segment P) A B C) → False):
  False :=
sorry

end no_segment_equal_angle_in_regular_pentagon_l668_668747


namespace trig_identity_l668_668043

noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem trig_identity (α β γ : ℝ) 
  (h₁ : cos γ = cos α * cos β) :
  tan ((γ + α) / 2) * tan ((γ - α) / 2) = tan (β / 2) ^ 2 := 
sorry

end trig_identity_l668_668043


namespace flashlights_distribution_equal_l668_668957

open Classical

noncomputable def lit_flashlights (a b : ℕ) : Prop :=
  ∃ k : ℕ, a - k = b ∧ 2 * k = a + b

theorem flashlights_distribution_equal (litA unlitA litB unlitB : ℕ) :
  litA = 100 ∧ unlitA = 100 ∧ litB = 100 ∧ unlitB = 100 →
  (∃ x y, lit_flashlights (litA + litB + x - y) (unlitA + unlitB - x + y)) :=
by
  intros h
  have h_eq : litA = 100 ∧ unlitA = 100 ∧ litB = 100 ∧ unlitB = 100 := h
  use 100, 100
  rw [←nat.add_sub_assoc h_eq.1.le, ←nat.add_sub_assoc h_eq.3.le]
  rw [add_comm (litA + litB), add_comm (unlitA + unlitB)]
  apply lit_flashlights
  sorry

end flashlights_distribution_equal_l668_668957


namespace monomial_properties_l668_668290

theorem monomial_properties:
  (∃ a b : ℝ, |a - 2| + (b + 3)^2 = 0 ∧ 
              (∀ x y : ℝ, 
                let coeff := -5 * Real.pi,
                let degree := a - b + 1 in 
                coeff = -5 * Real.pi ∧ degree = 6)) :=
by {
  use [2, -3],
  simp, 
  intros,
  simp,
  split,
  {
    refl,
  },
  {
    simp, 
  sorry,
  }
}

end monomial_properties_l668_668290


namespace number_of_paths_2012_l668_668655

def f (n : ℕ) : ℕ := (nat.factorial (n - 1))

theorem number_of_paths_2012 : f 2012 = nat.factorial 2011 :=
by sorry

end number_of_paths_2012_l668_668655


namespace total_inheritance_money_l668_668795

-- Defining the conditions
def number_of_inheritors : ℕ := 5
def amount_per_person : ℕ := 105500

-- The proof problem
theorem total_inheritance_money :
  number_of_inheritors * amount_per_person = 527500 :=
by sorry

end total_inheritance_money_l668_668795


namespace count_possible_pairs_l668_668373

noncomputable def is_prime (n : ℕ) : Prop :=
nat.prime n

def valid_pair (d n : ℕ) : Prop :=
∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ b > a ∧
30 + n = 10 * a + b ∧
d + n = 10 * b + a ∧
is_prime (10 * b + a)

theorem count_possible_pairs :
  { (d, n) : ℕ × ℕ | valid_pair d n } = { (54, 17), (75, 8) } :=
by sorry

end count_possible_pairs_l668_668373


namespace probability_more_heads_than_tails_l668_668334

theorem probability_more_heads_than_tails :
  (probability_more_heads_than_tails (num_flips := 8)) = 93 / 256 :=
by
  -- the definition and proof will be inserted here
  sorry

def probability_more_heads_than_tails (num_flips : ℕ) : ℚ :=
  let total_outcomes := 2 ^ num_flips
  let equal_heads_tails := 
    (nat.choose num_flips (num_flips / 2) : ℚ) / total_outcomes
  (1 - equal_heads_tails) / 2

end probability_more_heads_than_tails_l668_668334


namespace quadratic_root_property_l668_668273

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end quadratic_root_property_l668_668273


namespace local_max_gt_zero_implies_a_lt_neg1_l668_668389

noncomputable def f (a x : ℝ) : ℝ := exp x + a * x

theorem local_max_gt_zero_implies_a_lt_neg1 {a : ℝ} :
  (∃ x : ℝ, f a x = exp x + a * x ∧ (∀ ε > 0, exp x + a * (x + ε) ≤ exp x + a * x) ∧ x > 0 ) → a < -1 :=
by
  sorry

end local_max_gt_zero_implies_a_lt_neg1_l668_668389


namespace probability_team_B_wins_second_game_l668_668810

theorem probability_team_B_wins_second_game
  (team_A_wins_series : ∀ sequence : list (bool), 
                        (sequence.count true = 3 ∧
                         sequence.count false < 3) → 
                        sequence ∈ valid_sequences)
  (team_B_wins_third : ∀ sequence : list (bool), 
                       (sequence.length ≥ 3 ∧ 
                        sequence.nth 2 = some false) → 
                       sequence ∈ valid_sequences)
  (equal_prob_each_game : ∀ sequence : list (bool), 
                          sequence.probability = (0.5)^(sequence.length))
  (no_ties : ∀ sequence : list (bool), 
             sequence.nth_last(n) = some(false) ∨ 
             sequence.nth_last(n) = some(true))
  (independence : ∀ sequence : list (bool), 
                  ∀ sequence' : list (bool), 
                  sequence ∩ sequence' = ∅ → 
                  (sequence ∪ sequence').probability = 
                  sequence.probability * sequence'.probability) :
  (probability (sequence : list (bool)) (team_B_wins_second sequence) = 1 / 3) :=
sorry

end probability_team_B_wins_second_game_l668_668810


namespace EF_parallel_AB_l668_668422

open EuclideanGeometry

-- Define the square ABCD and the points K, L, M, N lying on the sides AB, BC, CD, and DA respectively
variables {A B C D K L M N E F : Point} (ABCD: square A B C D)
          (HK : lies_on K (segment A B)) (HL : lies_on L (segment B C))
          (HM : lies_on M (segment C D)) (HN : lies_on N (segment D A))

-- Declare that K, L, M, and N form a square
variables (square_KLMN : square K L M N)

-- Points E and F are defined as follows
variables (HE : intersection (line D K) (line N M) E)
          (HF : intersection (line K C) (line L M) F)

-- We need to prove that E F is parallel to the line segment A B
theorem EF_parallel_AB : parallel (line E F) (line A B) :=
  sorry

end EF_parallel_AB_l668_668422


namespace teds_age_l668_668555

theorem teds_age (s t : ℕ) (h1 : t = 3 * s - 20) (h2 : t + s = 76) : t = 52 :=
by
  sorry

end teds_age_l668_668555


namespace incorrect_option_l668_668791

theorem incorrect_option (a b c d e : ℝ) : 
  a = -1 / 2 ∧ b = (real.sqrt 3) / 2 ∧ d = a + b ∧ e = -(1 + real.sqrt 3) / 2 → d ≠ e :=
by
  intro h
  sorry

end incorrect_option_l668_668791


namespace quadratic_graph_characteristics_l668_668723

theorem quadratic_graph_characteristics (a b : ℝ) (h1 : a ≠ 0) :
  let c := -b^2 / (4 * a)
  in
    if a > 0 then 
      ∃ v : ℝ, ∀ x : ℝ, ax^2 + bx + c ≥ v
    else if a < 0 then 
      ∃ v : ℝ, ∀ x : ℝ, ax^2 + bx + c ≤ v
    else 
      false :=
begin
  sorry
end

end quadratic_graph_characteristics_l668_668723


namespace solution_set_f_x_minus_2_l668_668968

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 4 else -(x^2 - 4)

theorem solution_set_f_x_minus_2 :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | (0 < x ∧ x < 2) ∨ 4 < x} :=
by
  sorry

end solution_set_f_x_minus_2_l668_668968


namespace truthful_dwarfs_count_l668_668600

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l668_668600


namespace scientific_notation_of_361000000_l668_668236

theorem scientific_notation_of_361000000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ abs a) ∧ (abs a < 10) ∧ (361000000 = a * 10^n) ∧ (a = 3.61) ∧ (n = 8) :=
sorry

end scientific_notation_of_361000000_l668_668236


namespace marj_leftover_money_l668_668413

def total_initial_money (num_20_bills num_10_bills num_5_bills num_1_bills : ℕ) (coins : ℝ) : ℝ :=
  num_20_bills * 20 + num_10_bills * 10 + num_5_bills * 5 + num_1_bills * 1 + coins

def total_expenses (cake_cost gift_cost donation : ℝ) : ℝ :=
  cake_cost + gift_cost + donation

theorem marj_leftover_money 
  (num_20_bills : ℕ) (num_10_bills : ℕ) (num_5_bills : ℕ) (num_1_bills : ℕ)
  (coins : ℝ) (cake_cost : ℝ) (gift_cost : ℝ) (donation : ℝ) :
  total_initial_money num_20_bills num_10_bills num_5_bills num_1_bills coins = 81.50 →
  total_expenses cake_cost gift_cost donation = 35.50 →
  (total_initial_money num_20_bills num_10_bills num_5_bills num_1_bills coins - 
   total_expenses cake_cost gift_cost donation = 46.00) :=
by
  intros h_initial h_expenses
  rw [h_initial, h_expenses]
  norm_num

-- Parameter values for Marj's case
#eval marj_leftover_money 2 2 3 2 4.50 (-17.50) 12.70 5.30 sorry sorry

end marj_leftover_money_l668_668413


namespace compute_xy_l668_668482

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 198) : xy = 5 :=
by
  sorry

end compute_xy_l668_668482


namespace bank_record_withdrawal_l668_668345

def deposit (x : ℤ) := x
def withdraw (x : ℤ) := -x

theorem bank_record_withdrawal : withdraw 500 = -500 :=
by
  sorry

end bank_record_withdrawal_l668_668345


namespace alpha_eq_beta_eq_gamma_l668_668385

theorem alpha_eq_beta_eq_gamma
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (α β γ : ℂ)
  (hα : |α| = 1)
  (hβ : |β| = 1)
  (hγ : |γ| = 1)
  (h1 : x + y + z = 0)
  (h2 : α * x + β * y + γ * z = 0) :
  α = β ∧ β = γ := sorry

end alpha_eq_beta_eq_gamma_l668_668385


namespace common_internal_tangent_length_l668_668818

theorem common_internal_tangent_length (d r₁ r₂ : ℝ) (h₀ : d = 50) (h₁ : r₁ = 10) (h₂ : r₂ = 7) :
  sqrt (d^2 - (r₁ + r₂)^2) = sqrt 2211 :=
by 
  rw [h₀, h₁, h₂]
  sorry

end common_internal_tangent_length_l668_668818


namespace blue_chips_count_l668_668164

variable (T : ℕ) (blue_chips : ℕ) (white_chips : ℕ) (green_chips : ℕ)

-- Conditions
def condition1 : Prop := blue_chips = (T / 10)
def condition2 : Prop := white_chips = (T / 2)
def condition3 : Prop := green_chips = 12
def condition4 : Prop := blue_chips + white_chips + green_chips = T

-- Proof problem
theorem blue_chips_count (h1 : condition1 T blue_chips)
                          (h2 : condition2 T white_chips)
                          (h3 : condition3 green_chips)
                          (h4 : condition4 T blue_chips white_chips green_chips) :
  blue_chips = 3 :=
sorry

end blue_chips_count_l668_668164


namespace find_least_n_for_1000_pairs_l668_668625

-- Define the number of diagonal pairs function
def diagonal_pairs (n : ℕ) : ℕ :=
  have h : n % 2 = 0 := sorry, /- n must be even -/
  let m := n / 2
  m * (m - 1) ^ 2 / 2

-- Define the property to check for at least 1000 pairs
def has_at_least_1000_pairs (n : ℕ) : Prop :=
  diagonal_pairs n ≥ 1000

-- Main theorem statement
theorem find_least_n_for_1000_pairs :
  ∃ n, n > 0 ∧ has_at_least_1000_pairs n ∧ ∀ k, k > 0 → has_at_least_1000_pairs k → n ≤ k :=
  sorry

end find_least_n_for_1000_pairs_l668_668625


namespace no_n_satisfies_Tn_property_l668_668017

theorem no_n_satisfies_Tn_property :
  ∀ (n : ℕ), ∃ (a b : ℕ), a ≠ b ∧ a ∈ {11 * (k + h) + 10 * (n^k + n^h) | k h : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 1 ≤ h ∧ h ≤ 10} ∧ b ∈ {11 * (k + h) + 10 * (n^k + n^h) | k h : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 1 ≤ h ∧ h ≤ 10} ∧ a % 110 = b % 110 :=
by
  sorry

end no_n_satisfies_Tn_property_l668_668017


namespace segment_in_regular_pentagon_l668_668750

/--
  Problem: 
  Prove that it is impossible to place a segment inside a regular pentagon
  so that it is seen from all vertices at the same angle.
-/
theorem segment_in_regular_pentagon (P : Type) [regular_pentagon P] :
  ¬ ∃ (XY : segment P), (∀ (v : P), ∃ (angle : angle P), seen_from v XY = angle) :=
by
  -- Proof goes here
  sorry

end segment_in_regular_pentagon_l668_668750


namespace projectile_reaches_40ft_l668_668081

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  ((-b + real.sqrt (b*b - 4*a*c)) / (2*a), (-b - real.sqrt (b*b - 4*a*c)) / (2*a))

theorem projectile_reaches_40ft :
  ∃ t : ℝ, (-16 * t^2 + 50 * t = 40) ∧ (t > 0) :=
  by
    let a := -16
    let b := 50
    let c := -40
    let (t1, t2) := quadratic_roots a b c
    use t1
    split
    sorry -- Proof of -16 * t1^2 + 50 * t1 = 40
    sorry -- Proof that t1 > 0

end projectile_reaches_40ft_l668_668081


namespace prove_area_of_triangle_abc_l668_668078

open Real

noncomputable def area_triangle_abc : Real :=
    let radius := 3
    let ab := 2 * radius
    let bd := 4
    let ad := ab + bd -- This is actually 6 not 10. Correction
    let ed := 6
    let angle_ade := 30
    let cos_angle_ade := sqrt 3 / 2
    let ae := sqrt (ad^2 + ed^2 - 2 * ad * ed * cos_angle_ade)
    let ea := ae
    let ec := ed^2 / ea
    let ac := ea - ec
    let ab_sq := ab^2
    let ac_sq := ac^2
    let bc := sqrt (ab_sq - ac_sq)
    (1 / 2) * bc * ac

theorem prove_area_of_triangle_abc
  (radius : Real)
  (ab : Real := 2 * radius)
  (bd : Real := 4)
  (ad : Real := ab + bd)
  (ed : Real := 6)
  (cos_angle_ade : Real := sqrt 3 / 2)
  (ae := sqrt (ad^2 + ed^2 - 2 * ad * ed * cos_angle_ade))
  (ea := ae)
  (ec := ed^2 / ea)
  (ac := ea - ec)
  (ab_sq := ab^2)
  (ac_sq := ac^2)
  (bc := sqrt (ab_sq - ac_sq))
  (area := (1 / 2) * bc * ac) :
  area = _ := (sorry : Real)

end prove_area_of_triangle_abc_l668_668078


namespace terminating_decimal_expansion_of_13_over_320_l668_668629

theorem terminating_decimal_expansion_of_13_over_320 : ∃ (b : ℕ) (a : ℚ), (13 : ℚ) / 320 = a / 10 ^ b ∧ a / 10 ^ b = 0.650 :=
by
  sorry

end terminating_decimal_expansion_of_13_over_320_l668_668629


namespace sum_of_roots_l668_668874

theorem sum_of_roots (a b c : ℝ) (h : 6 * a ^ 3 - 7 * a ^ 2 + 2 * a = 0 ∧ 
                                   6 * b ^ 3 - 7 * b ^ 2 + 2 * b = 0 ∧ 
                                   6 * c ^ 3 - 7 * c ^ 2 + 2 * c = 0 ∧ 
                                   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    a + b + c = 7 / 6 :=
sorry

end sum_of_roots_l668_668874


namespace peter_initial_erasers_l668_668420

theorem peter_initial_erasers (E : ℕ) (h : E + 3 = 11) : E = 8 :=
by {
  sorry
}

end peter_initial_erasers_l668_668420


namespace not_odd_function_l668_668843

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x ^ 2 + 1)

theorem not_odd_function : ¬ is_odd_function f := by
  sorry

end not_odd_function_l668_668843


namespace number_of_truthful_gnomes_l668_668604

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l668_668604


namespace lisa_box_min_height_l668_668773

def is_valid_box (x : ℝ) : Prop :=
  2 * x^2 + 4 * x * (x + 3) ≥ 90

def height (x : ℝ) : ℝ :=
  x + 3

theorem lisa_box_min_height : ∃ x : ℝ, is_valid_box x ∧ height x = 6 :=
by
  sorry

end lisa_box_min_height_l668_668773


namespace mutually_exclusive_not_contradictory_l668_668638

-- Definitions for the conditions
def bag := { "red1", "red2", "black1", "black2" }

-- Define what is meant by choosing two balls
def two_ball_draws (s : finset string) : finset (finset string) :=
  finset.powersetLen 2 s

-- Define each event explicitly based on problem description
def event_A (s : finset string) : Prop :=
  "black1" ∈ s ∧ "black2" ∈ s

def event_B (s : finset string) : Prop :=
  ("black1" ∈ s ∨ "black2" ∈ s) ∧ ("red1" ∈ s ∨ "red2" ∈ s)

def event_C (s : finset string) : Prop :=
  s.filter (λ (ball : string), ball.starts_with "black").card = 1 ∧ s.card = 2

def event_D (s : finset string) : Prop :=
  ("black1" ∈ s ∨ "black2" ∈ s) ∧ ("red1" ∈ s ∧ "red2" ∈ s)

-- Conditions based on the described context
def condition := two_ball_draws bag

-- Proving that Event C is mutually exclusive but not contradictory
theorem mutually_exclusive_not_contradictory :
  ∃ s ∈ condition, event_C s :=
by {
  sorry 
}

end mutually_exclusive_not_contradictory_l668_668638


namespace b_120_is_correct_l668_668569

def seq (n : ℕ) : ℚ
| 1 => 2
| 2 => 1/3
| n+1 => (2 - 3 * seq n) / (3 * seq (n-1))

theorem b_120_is_correct : seq 120 = 5 / 6 := by
  sorry

end b_120_is_correct_l668_668569


namespace marbles_left_l668_668882

def initial_marbles : ℕ := 100
def percent_t_to_Theresa : ℕ := 25
def percent_t_to_Elliot : ℕ := 10

theorem marbles_left (w t e : ℕ) (h_w : w = initial_marbles)
                                 (h_t : t = percent_t_to_Theresa)
                                 (h_e : e = percent_t_to_Elliot) : w - ((t * w) / 100 + (e * w) / 100) = 65 :=
by
  rw [h_w, h_t, h_e]
  sorry

end marbles_left_l668_668882


namespace stephanie_store_visits_l668_668804

theorem stephanie_store_visits (oranges_per_visit total_oranges : ℕ) 
  (h1 : oranges_per_visit = 2)
  (h2 : total_oranges = 16) : 
  total_oranges / oranges_per_visit = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end stephanie_store_visits_l668_668804


namespace constant_S13_l668_668000

-- Define the arithmetic sequence and its sum function
def arithmetic_sequence (a1 d : ℕ → ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Define the constant condition
def condition (a1 d : ℕ → ℝ) : Prop :=
  a1 2 + a1 4 + a1 15 = k -- arbitrary constant k

theorem constant_S13 (a1 d : ℕ → ℝ) (h : condition a1 d) : 
  ∃ (k : ℝ), sum_arithmetic_sequence a1 d 13 = k :=
by
  sorry

end constant_S13_l668_668000


namespace wendy_adds_18_gallons_l668_668119

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end wendy_adds_18_gallons_l668_668119


namespace least_number_modular_l668_668624

theorem least_number_modular 
  (n : ℕ)
  (h1 : n % 34 = 4)
  (h2 : n % 48 = 6)
  (h3 : n % 5 = 2) : n = 4082 :=
by
  sorry

end least_number_modular_l668_668624


namespace necessary_and_sufficient_condition_l668_668716

variables {f g : ℝ → ℝ}

theorem necessary_and_sufficient_condition (f g : ℝ → ℝ)
  (hdom : ∀ x : ℝ, true)
  (hst : ∀ y : ℝ, true) :
  (∀ x : ℝ, f x > g x) ↔ (∀ x : ℝ, ¬ (x ∈ {x : ℝ | f x ≤ g x})) :=
by sorry

end necessary_and_sufficient_condition_l668_668716


namespace smallest_possible_n_l668_668498

theorem smallest_possible_n (n : ℕ) (h : n > 0) 
  (h_condition : Nat.lcm 60 n / Nat.gcd 60 n = 44) : n = 165 := by
  sorry

end smallest_possible_n_l668_668498


namespace find_m_l668_668083

-- Define the function f(x)
def f (a m : ℝ) (x : ℝ) : ℝ := a^(x^2 + 2 * x - 3) + m

-- Conditions and question in Lean 4 statement
theorem find_m (a : ℝ) (m : ℝ) (h_a : a > 1) (h_f : f a m 1 = 10) : m = 9 :=
by
  -- Proof omitted
  sorry

end find_m_l668_668083


namespace angle_of_inclination_l1_eq_45_l668_668302

noncomputable def slope (m : ℝ) := m
noncomputable def angle_of_inclination (k : ℝ) := real.arctan k * (180 / real.pi)

theorem angle_of_inclination_l1_eq_45 :
  ∀ (l1 l2 : ℝ → ℝ)
    (h : slope (l1 1 - l1 0) = slope (l2 1 - l2 0))
    (h_eq : ∀ x, l2 x = x - 2),
  angle_of_inclination (l1 1 - l1 0) = 45 :=
by
  intros l1 l2 h h_eq
  -- Proof steps will be filled in here
  sorry

end angle_of_inclination_l1_eq_45_l668_668302


namespace domain_of_f_l668_668080

variable (x : ℝ)

def f (x : ℝ) : ℝ := sqrt (2 * x^2 + x - 3) + log 3 (3 + 2 * x - x^2)

theorem domain_of_f :
  {y : ℝ | ∃ x, f x = y} = set.Ico 1 3 :=
by sorry

end domain_of_f_l668_668080


namespace students_minus_rabbits_l668_668229

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the number of classrooms
def number_of_classrooms : ℕ := 5

-- Define the total number of students and rabbits
def total_students : ℕ := students_per_classroom * number_of_classrooms
def total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms

-- The main statement to prove
theorem students_minus_rabbits :
  total_students - total_rabbits = 105 :=
by
  sorry

end students_minus_rabbits_l668_668229


namespace lambda_six_ge_sqrt_three_l668_668641

open Real
open Set

noncomputable def lambda (n : ℕ) (P : Fin n → Point) : ℝ :=
  let dists := {dist p1 p2 | p1 p2 in P}
  let maxDist := Sup dists
  let minDist := Inf dists
  maxDist / minDist

theorem lambda_six_ge_sqrt_three
  (P : Fin 6 → Point)
  (hP : ∀ i j, i ≠ j → P i ≠ P j) : 
  lambda 6 P ≥ sqrt 3 :=
  sorry

end lambda_six_ge_sqrt_three_l668_668641


namespace percentage_deposited_l668_668429

theorem percentage_deposited (amount_deposited income : ℝ) 
  (h1 : amount_deposited = 2500) (h2 : income = 10000) : 
  (amount_deposited / income) * 100 = 25 :=
by
  have amount_deposited_val : amount_deposited = 2500 := h1
  have income_val : income = 10000 := h2
  sorry

end percentage_deposited_l668_668429


namespace trajectory_of_point_l668_668715

theorem trajectory_of_point 
  (P : ℝ × ℝ) 
  (h1 : abs (P.1 - 4) + P.2^2 - 1 = abs (P.1 + 5)) : 
  P.2^2 = 16 * P.1 := 
sorry

end trajectory_of_point_l668_668715


namespace vasya_did_not_buy_anything_days_l668_668940

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l668_668940


namespace initial_sugar_amount_l668_668831

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end initial_sugar_amount_l668_668831


namespace sum_of_pairwise_rel_prime_integers_l668_668856

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end sum_of_pairwise_rel_prime_integers_l668_668856


namespace correct_root_calculation_l668_668879

theorem correct_root_calculation : (sqrt 12 / sqrt 2) = sqrt 6 :=
by
  sorry

end correct_root_calculation_l668_668879


namespace pyramid_volume_in_unit_cube_l668_668547

noncomputable def base_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem pyramid_volume_in_unit_cube : 
  let s := Real.sqrt 2 / 2
  let height := 1
  pyramid_volume (base_area s) height = Real.sqrt 3 / 24 :=
by
  sorry

end pyramid_volume_in_unit_cube_l668_668547


namespace pentagon_quadrilateral_sum_of_angles_l668_668822

   theorem pentagon_quadrilateral_sum_of_angles
     (exterior_angle_pentagon : ℕ := 72)
     (interior_angle_pentagon : ℕ := 108)
     (sum_interior_angles_quadrilateral : ℕ := 360)
     (reflex_angle : ℕ := 252) :
     (sum_interior_angles_quadrilateral - reflex_angle = interior_angle_pentagon) :=
   by
     sorry
   
end pentagon_quadrilateral_sum_of_angles_l668_668822


namespace mike_ride_equals_42_l668_668415

-- Define the costs as per the conditions
def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie : ℝ := 2.50 + 5.00 + 0.25 * 22

-- State the theorem that needs to be proved
theorem mike_ride_equals_42 : ∃ M : ℕ, cost_mike M = cost_annie ∧ M = 42 :=
by
  sorry

end mike_ride_equals_42_l668_668415


namespace ratio_of_circle_area_to_triangle_area_l668_668349

theorem ratio_of_circle_area_to_triangle_area (b h : ℝ) (hb : b > 0) (hh : h > 0) :
  let c := real.sqrt (b^2 + h^2),
      s := (b + h + c) / 2,
      A := (1 / 2) * b * h,
      r := A / s,
      circle_area := real.pi * r^2,
      triangle_area := A,
      ratio := 2 * real.pi * (b^2 * h^2) / ((b + h + real.sqrt (b^2 + h^2))^2 * b * h)
  in ratio = circle_area / triangle_area :=
by
  sorry

end ratio_of_circle_area_to_triangle_area_l668_668349


namespace magic_square_sum_l668_668003

-- Given conditions
def magic_square (S : ℕ) (a b c d e : ℕ) :=
  (30 + b + 27 = S) ∧
  (30 + 33 + a = S) ∧
  (33 + c + d = S) ∧
  (a + 18 + e = S) ∧
  (30 + c + e = S)

-- Prove that the sum a + d is 38 given the sums of the 3x3 magic square are equivalent
theorem magic_square_sum (a b c d e S : ℕ) (h : magic_square S a b c d e) : a + d = 38 :=
  sorry

end magic_square_sum_l668_668003


namespace machine_A_produces_7_sprockets_per_hour_l668_668519

theorem machine_A_produces_7_sprockets_per_hour
    (A B : ℝ)
    (h1 : B = 1.10 * A)
    (h2 : ∃ t : ℝ, 770 = A * (t + 10) ∧ 770 = B * t) : 
    A = 7 := 
by 
    sorry

end machine_A_produces_7_sprockets_per_hour_l668_668519


namespace correct_geometry_statement_to_extend_segment_l668_668506

-- Definitions
inductive GeometryStatement
| ExtendRay (O A : Point) : GeometryStatement
| ExtendLine (A B : Point) : GeometryStatement
| ExtendSegment (A B : Point) : GeometryStatement
| ConstructEqualLine (A B C D : Point) : GeometryStatement

open GeometryStatement

-- Conditions
def ray_has_endpoint_and_extends_infinitely (O A : Point) : Prop :=
  True

def line_extends_infinitely_in_both_directions (A B : Point) : Prop :=
  True

def segment_has_two_endpoints (A B : Point) : Prop :=
  True

def construct_equal_line_segment (A B C D : Point) : Prop :=
  AB = CD

-- Theorem statement claiming that the correct option is to extend the segment AB
theorem correct_geometry_statement_to_extend_segment (O A B C D : Point) :
  (∀ P, (ExtendRay O A = P) → false) ∧
  (∀ P, (ExtendLine A B = P) → false) ∧
  (∀ P, (ConstructEqualLine A B C D = P) → false) ∧
  (ExtendSegment A B = ExtendSegment A B) :=
begin
  sorry
end

end correct_geometry_statement_to_extend_segment_l668_668506


namespace rhombus_area_l668_668520

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 12) (h2 : d2 = 15) : d1 * d2 / 2 = 90 := by
  rw [h1, h2]
  norm_num
  sorry

end rhombus_area_l668_668520


namespace closest_point_l668_668628

theorem closest_point 
  (x y z : ℝ) 
  (h_plane : 3 * x - 4 * y + 5 * z = 30)
  (A : ℝ × ℝ × ℝ := (1, 2, 3)) 
  (P : ℝ × ℝ × ℝ := (x, y, z)) :
  P = (11 / 5, 2 / 5, 5) := 
sorry

end closest_point_l668_668628


namespace inverse_of_matrix_l668_668622

noncomputable def matrix_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 5], ![1, 3]]

def inverse_matrix (m : Matrix (Fin 2) (Fin 2) ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, -5], ![-1, 2]]

theorem inverse_of_matrix :
  let det := (matrix_inverse 0 0) * (matrix_inverse 1 1) - (matrix_inverse 0 1) * (matrix_inverse 1 0) in
  det ≠ 0 →
  (matrix_inverse.mul (inverse_matrix matrix_inverse) = 1 ∧ inverse_matrix matrix_inverse.mul matrix_inverse = 1) :=
by
  sorry

end inverse_of_matrix_l668_668622


namespace vasya_days_l668_668907

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668907


namespace vasya_days_without_purchase_l668_668914

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668914


namespace find_N_l668_668549

theorem find_N (N : ℕ) (h : 7.5 < N / 3 ∧ N / 3 < 8) : N = 23 :=
by
  sorry

end find_N_l668_668549


namespace fraction_inequality_quadratic_no_real_solution_range_l668_668508

theorem fraction_inequality {a b m : ℝ} (h₀ : a > b) (h₁ : b > 0) (h₂ : m > 0) : 
  (b / a) < (b + m) / (a + m) :=
by sorry

theorem quadratic_no_real_solution_range (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 4 * x + m = 0) ↔ m ∈ set.Ioi 4 :=
by sorry

end fraction_inequality_quadratic_no_real_solution_range_l668_668508


namespace problem1_problem2_l668_668683

-- First declare the necessary conditions and objects:
def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ := (range (n+1)).map a \sum λ i => a i
def f (x : ℚ) : ℚ := log x (1/3)
def b_n (a : ℕ → ℚ) (n : ℕ) : ℚ := (range (n+1)).map (λ i => f (a i)) \sum λ i => f (a i)
def a_n (n : ℕ) : ℚ := (1 / 3) ^ n

-- Prove that a_n satisfies the given condition for S_n:
theorem problem1 (n : ℕ) : S_n a_n n = 1/2 * (1 - a_n n) := sorry

-- Given the function f(x) and the sequence a_n, relate it to b_n and find T_n:
theorem problem2 (n : ℕ) : ∑ i in (range (n+1)), 1 / b_n a_n i = 2 * n / (n + 1) := sorry

end problem1_problem2_l668_668683


namespace smallest_possible_n_l668_668496

theorem smallest_possible_n (n : ℕ) (h : n > 0) 
  (h_condition : Nat.lcm 60 n / Nat.gcd 60 n = 44) : n = 165 := by
  sorry

end smallest_possible_n_l668_668496


namespace train_platform_pass_time_l668_668134

-- Definition of the initial conditions
def train_length : ℝ := 250
def pole_pass_time : ℝ := 10
def platform_length : ℝ := 1250

-- Derived definitions
def train_speed : ℝ := train_length / pole_pass_time
def total_distance : ℝ := train_length + platform_length

-- The statement to be proven
theorem train_platform_pass_time : 
  (total_distance / train_speed) = 60 := 
by 
  -- This is where the proof would be placed
  sorry

end train_platform_pass_time_l668_668134


namespace same_terminal_angle_l668_668813

theorem same_terminal_angle (k : ℤ) :
  ∃ α : ℝ, α = k * 360 + 40 :=
by
  sorry

end same_terminal_angle_l668_668813


namespace arc_length_sector_l668_668355

theorem arc_length_sector
  (O A B : Type)
  (radius: ℝ)
  (angle: ℝ)
  (length_arc_AB: ℝ)
  (h_radius: radius = 3)
  (h_angle: angle = 120) :
  length_arc_AB = (angle / 360) * 2 * π * radius :=
by
  rw [h_radius, h_angle]
  simp
  sorry

end arc_length_sector_l668_668355


namespace number_of_truthful_dwarfs_l668_668585

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l668_668585


namespace problem_pqr_l668_668268

theorem problem_pqr
  (p q r : ℤ)
  (f g : ℤ[X])
  (h_f : f = X^4 + 4*X^3 + 6*p*X^2 + 4*q*X + r)
  (h_g : g = X^3 + 3*X^2 + 9*X + 3)
  (h_div : g ∣ f) :
  (p + q) * r = 15 := 
sorry

end problem_pqr_l668_668268


namespace initial_children_count_l668_668469

theorem initial_children_count (initial_children : ℕ) (girls_came : ℕ) (boys_left : ℕ) (final_children : ℕ) :
  girls_came = 24 →
  boys_left = 31 →
  final_children = 78 →
  initial_children + girls_came - boys_left = final_children →
  initial_children = 85 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end initial_children_count_l668_668469


namespace john_spent_on_candy_l668_668202

theorem john_spent_on_candy (X : ℝ) (H1 : X = 150) :
  (X / 10 : ℝ) = 10 :=
by
  have h1 : (X/2 + X/3 + X/10) = 28*X/30 := sorry
  have h2 : X - (28*X/30) = X/15 := sorry
  rw [H1 at h2]
  norm_num
  rw [h2]
  norm_num
  sorry

end john_spent_on_candy_l668_668202


namespace vasya_purchase_l668_668929

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l668_668929


namespace num_all_three_restaurants_l668_668513

-- Definitions corresponding to conditions
def num_employees : ℕ := 39
def num_family_buffet : ℕ := 15
def num_dining_room : ℕ := 18
def num_snack_bar : ℕ := 12
def num_two_restaurants : ℕ := 4

-- Main statement
theorem num_all_three_restaurants : ∃ x : ℕ, x = 2 ∧ 
  num_employees = num_family_buffet + num_dining_room + num_snack_bar - num_two_restaurants - 2 * x + x := 
by 
  use 2
  split
  { refl }
  { 
    simp [num_employees, num_family_buffet, num_dining_room, num_snack_bar, num_two_restaurants]
    sorry 
  }

end num_all_three_restaurants_l668_668513


namespace area_of_triangle_DEF_is_correct_l668_668618

-- Definitions
def DE := (8 : ℝ) / (Real.sqrt 3)
def DF := (8 : ℝ)

-- This involves naming the hypotenuse based on the properties.
def EF := 2 * DE

-- Calculate the area
def area_triangle_DEF : ℝ := (1 / 2) * DE * DF

-- Main theorem statement, which is to be proved.
theorem area_of_triangle_DEF_is_correct :
  area_triangle_DEF = 32 * Real.sqrt 3 / 3 := 
sorry

end area_of_triangle_DEF_is_correct_l668_668618


namespace metal_waste_is_three_halves_area_l668_668176

theorem metal_waste_is_three_halves_area (w : ℝ) :
  let area_rectangle := 2 * w^2
    let area_circle := (Real.pi * (w / 2)^2) 
    let a := (w * Real.sqrt 2 / 2)
    let area_square := (w * Real.sqrt 2 / 2)^2 / 2
  in 2 * w^2 - (Real.pi * (w / 2)^2) + (Real.pi * (w / 2)^2) - (w^2 / 2) = 3 / 2 * 2 * w^2 := 
by {
  -- Definitions as given by conditions
  let area_rectangle := 2 * w^2
  let area_circle := (Real.pi * (w / 2)^2)
  let a := (w * Real.sqrt 2 / 2)
  let area_square := (w * Real.sqrt 2 / 2)^2 / 2
  
  -- the statement simplifies and calculates the waste
  have waste : 2 * w^2 - (Real.pi * (w / 2)^2) + (Real.pi * (w / 2)^2) - (w^2 / 2) = 3 / 2 * 2 * w^2,
  sorry,
  
  exact waste,
}

end metal_waste_is_three_halves_area_l668_668176


namespace projected_percent_increase_l668_668029

theorem projected_percent_increase (R : ℝ) (p : ℝ) 
  (h1 : 0.7 * R = R * 0.7) 
  (h2 : 0.7 * R = 0.5 * (R + p * R)) : 
  p = 0.4 :=
by
  sorry

end projected_percent_increase_l668_668029


namespace algebraic_expression_value_l668_668260

theorem algebraic_expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2 * x + 1) * (2 * x - 1) + x * (x - 1) = 1 :=
by
  sorry

end algebraic_expression_value_l668_668260


namespace choir_member_count_l668_668456

theorem choir_member_count (original new : ℕ) (h₀ : original = 36) (h₁ : new = 9) : original + new = 45 := 
by
  rw [h₀, h₁]
  norm_num
  sorry

end choir_member_count_l668_668456


namespace measure_angle_BAC_l668_668963

/-- A circle centered at O is circumscribed about ΔABC, with the given angles AOB and BOC. -/
theorem measure_angle_BAC (A B C O : Type*) [fintype O]
  (h₁ : ∠ A O B = 130) (h₂ : ∠ B O C = 90) : ∠ B A C = 45 :=
sorry

end measure_angle_BAC_l668_668963


namespace problem_solution_l668_668661

variable (α : ℝ)
-- Condition: α in the first quadrant (0 < α < π/2)
variable (h1 : 0 < α ∧ α < Real.pi / 2)
-- Condition: sin α + cos α = sqrt 2
variable (h2 : Real.sin α + Real.cos α = Real.sqrt 2)

theorem problem_solution : Real.tan α + Real.cos α / Real.sin α = 2 :=
by
  sorry

end problem_solution_l668_668661


namespace optimal_chart_is_line_chart_l668_668477

def Chart : Type := 
| Bar
| Line
| Pie
| None

axiom represent_changes (c : Chart) : Prop

def reflect_changes_temperature := ∀ (c : Chart), represent_changes c → c = Chart.Line

theorem optimal_chart_is_line_chart :
  reflect_changes_temperature Chart.Line :=
by
  intros c H_c
  sorry

end optimal_chart_is_line_chart_l668_668477


namespace candy_factory_days_l668_668073

noncomputable def candies_per_hour := 50
noncomputable def total_candies := 4000
noncomputable def working_hours_per_day := 10
noncomputable def total_hours_needed := total_candies / candies_per_hour
noncomputable def total_days_needed := total_hours_needed / working_hours_per_day

theorem candy_factory_days :
  total_days_needed = 8 := 
by
  -- (Proof steps will be filled here)
  sorry

end candy_factory_days_l668_668073


namespace find_people_and_carriages_l668_668143

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end find_people_and_carriages_l668_668143


namespace ratio_of_numbers_l668_668111

theorem ratio_of_numbers (A B : ℕ) (h_lcm : Nat.lcm A B = 48) (h_hcf : Nat.gcd A B = 4) : A / 4 = 3 ∧ B / 4 = 4 :=
sorry

end ratio_of_numbers_l668_668111


namespace find_angle_APC_l668_668001

-- Definitions of angles and distance between the centers of the semicircles
def semicircle_SAR := {angle_AS : ℝ} (h_as : angle_AS = 72)
def semicircle_RBT := {angle_BT : ℝ} (h_bt : angle_BT = 45)
def distance_centers := {d : ℝ}

-- Tangents and lines properties
def tangent_PA := {PA_tangent : Prop}
def tangent_PC := {PC_tangent : Prop}
def line_SRT := {SRT_line : Prop}

-- Prove that angle APC is 117 degrees.
theorem find_angle_APC (semicircle_SAR : Prop) (semicircle_RBT : Prop) (distance_centers : Prop)
(tangent_PA : Prop) (tangent_PC : Prop) (line_SRT : Prop) :
∃ (angle_APC : ℝ), angle_APC = 117 :=
by sorry

end find_angle_APC_l668_668001


namespace average_speed_l668_668893

-- Definitions based on the conditions from part a
def distance_first_hour : ℕ := 90
def distance_second_hour : ℕ := 30
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Main theorem stating the average speed
theorem average_speed :
  (distance_first_hour + distance_second_hour) / (time_first_hour + time_second_hour) = 60 :=
by
  -- proof goes here
  sorry

end average_speed_l668_668893


namespace intersection_polar_coordinates_l668_668735

-- Define parametric functions for C1 and C2
def C1x (θ : ℝ) := sqrt 5 * cos θ
def C1y (θ : ℝ) := sqrt 5 * sin θ

def C2x (t : ℝ) := sqrt 5 - (sqrt 2 / 2) * t
def C2y (t : ℝ) := - (sqrt 2 / 2) * t

-- Lean theorem stating the proof problem
theorem intersection_polar_coordinates : 
  (∃ θ t : ℝ, C1x θ = C2x t ∧ C1y θ = C2y t ∧ 
    ((sqrt ((C1x θ)^2 + (C1y θ)^2) = 5 ∧ atan2 (C1y θ) (C1x θ) = 3 * pi / 2) 
    ∨ (sqrt ((C1x θ)^2 + (C1y θ)^2) = 5 ∧ atan2 (C1y θ) (C1x θ) = 0))) :=
sorry

end intersection_polar_coordinates_l668_668735


namespace minimize_radii_correct_l668_668280

variable {P : Type} [Plane P]
variable (A B C : P)

noncomputable def minimize_radii (M : P) : Prop :=
  let r1 := circumcircle_radius (triangle A B M)
  let r2 := circumcircle_radius (triangle B C M)
  r1 + r2 = min (fun m => circumcircle_radius (triangle A B m) + circumcircle_radius (triangle B C m))

def is_foot_of_perpendicular (M : P) : Prop :=
  perpendicularly_projected B M C

theorem minimize_radii_correct (M : P) :
  minimize_radii A B C M ↔ is_foot_of_perpendicular B M C :=
by
  -- Proof skipped
  sorry

end minimize_radii_correct_l668_668280


namespace total_cost_is_21_l668_668234

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end total_cost_is_21_l668_668234


namespace triangle_trig_values_l668_668366

theorem triangle_trig_values (a b : ℝ) (A : ℝ)
  (h_a : a = 10) (h_b : b = 8) (h_A : A = 60 * Real.pi / 180) :
  (Real.sin (angle_of_sides a b 60) = 2 * Real.sqrt 3 / 5) ∧
  (Real.cos (angle_of_sides a b 60 + angle_of_sides a b 60) = (6 - Real.sqrt 13) / 10) :=
by {
  sorry
}

end triangle_trig_values_l668_668366


namespace required_bricks_l668_668325

-- Definitions based on the problem descriptions
def brick_length := 25 -- in cm
def brick_height := 11.25 -- in cm
def brick_width := 6 -- in cm

def wall_length := 400 -- in cm (converted from 4 m)
def wall_height := 200 -- in cm (converted from 2 m)
def wall_width := 25 -- in cm

-- Definition to calculate the volumes
def volume_brick := brick_length * brick_height * brick_width
def volume_wall := wall_length * wall_height * wall_width

-- Proof statement: number of bricks required
theorem required_bricks :
  (volume_wall / volume_brick).ceil = 1186 := 
by
  sorry

end required_bricks_l668_668325


namespace system1_solution_system2_solution_l668_668061

theorem system1_solution (x y : ℤ) 
  (h1 : x = 2 * y - 1) 
  (h2 : 3 * x + 4 * y = 17) : 
  x = 3 ∧ y = 2 :=
by 
  sorry

theorem system2_solution (x y : ℤ) 
  (h1 : 2 * x - y = 0) 
  (h2 : 3 * x - 2 * y = 5) : 
  x = -5 ∧ y = -10 := 
by 
  sorry

end system1_solution_system2_solution_l668_668061


namespace minimal_sticks_l668_668768

structure Sieve (n : ℕ) (h : n ≥ 2) :=
  (removed_cells : Fin n → Fin n)
  (distinct_rows : Function.Injective removed_cells)
  (distinct_cols : Function.Injective (fun (i : Fin n) => removed_cells i))

def stick (k : ℕ) := k > 0

theorem minimal_sticks (n : ℕ) (h : n ≥ 2) (A : Sieve n h) : 
  let mA := 2 * n - 2 in true :=
sorry

end minimal_sticks_l668_668768


namespace FGH_supermarkets_count_l668_668468

theorem FGH_supermarkets_count (C US_ca total : ℕ) (h1 : total = 60) (h2 : US_ca = C + 22) (h3 : total = C + US_ca) :
  US_ca = 41 :=
by
  -- Definitions and conditions
  let C := 19
  have h4 : total = 2 * C + 22, by sorry
  have h5 : total = 60, by sorry
  have h6 : 2 * C + 22 = 60, by sorry
  have h7 : C = 19, by sorry
  -- Concluding
  have US_ca_val : US_ca = C + 22, by sorry
  have US_ca := 19 + 22, by sorry
  sorry

end FGH_supermarkets_count_l668_668468


namespace sin_double_angle_identity_l668_668681

theorem sin_double_angle_identity (α : ℝ) : 
  let P := (real.cos α, real.sin α) in
  P.2 = 2 * P.1 → real.sin (2 * α) = 4 / 5 :=
by
  intros
  sorry

end sin_double_angle_identity_l668_668681


namespace vasya_days_without_purchases_l668_668902

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668902


namespace balance_weights_l668_668841

def pair_sum {α : Type*} (l : List α) [Add α] : List (α × α) :=
  l.zip l.tail

theorem balance_weights (w : Fin 100 → ℝ) (h : ∀ i j, |w i - w j| ≤ 20) :
  ∃ (l r : Finset (Fin 100)), l.card = 50 ∧ r.card = 50 ∧
  |(l.sum w - r.sum w)| ≤ 20 :=
sorry

end balance_weights_l668_668841


namespace first_player_win_l668_668974

theorem first_player_win (n : ℕ) (h : n ≥ 4) :
  ∃ strategy : (fin n → option ℕ) → (fin n → option ℕ), 
    (∀ chips : fin n → option ℕ,
      (chips (fin.of_nat (n-2)) = some 1) ∧
      (chips (fin.of_nat (n-1)) = some 1) ∧
      (chips (fin.of_nat n) = some 1) →
      (∃m : ℕ, strategy chips = chips) ) → 
  (∀ (opponent_strategy : (fin n → option ℕ) → (fin n → option ℕ)) (chips : fin n → option ℕ),
    (chips (fin.of_nat (n-2)) = some 1) ∧
    (chips (fin.of_nat (n-1)) = some 1) ∧
    (chips (fin.of_nat n) = some 1) →
    (∃ m : ℕ, opponent_strategy chips ≠ chips) →
    ∃ chips', strategy chips = chips' ∧ 
    (∀ m : ℕ, ∃ k : ℕ, chips' (fin.of_nat k)  ≠ chips (fin.of_nat m) )
) :=
sorry

end first_player_win_l668_668974


namespace problem_statement_l668_668402

-- Define the value z
noncomputable def z : ℂ := (1 - complex.I) / real.sqrt 2

-- Define the target expression we need to prove equals 36
def target_expr : ℂ := 
  let sumZ := (finset.range 6).sum (λ m, z ^ (2 * m + 1) ^ 2)
  let sumInvZ := (finset.range 6).sum (λ m, 1 / (z ^ (2 * m + 1) ^ 2))
  sumZ * sumInvZ

theorem problem_statement : target_expr = 36 := by
  sorry

end problem_statement_l668_668402


namespace part1_part2_l668_668645

noncomputable def radius_distance : ℝ :=
  (abs (2 * real.sqrt 2 - (- real.sqrt 2))) / real.sqrt ((3 - 2)^2 + 1^2)

noncomputable def circle_N_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 9

noncomputable def symmetrical_point_C (x y : ℝ) : Prop :=
  (x, y) = (-5, -2)

noncomputable def tangent_circle_C_equation (x y : ℝ) : Prop :=
  (x + 5)^2 + (y + 2)^2 = 49

theorem part1 (x y : ℝ) : (radius_distance = 3) → circle_N_equation x y := by sorry

theorem part2 (x y : ℝ) : symmetrical_point_C x y → tangent_circle_C_equation x y := by sorry

end part1_part2_l668_668645


namespace find_people_and_carriages_l668_668145

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end find_people_and_carriages_l668_668145


namespace problem_thre_is_15_and_10_percent_l668_668472

theorem problem_thre_is_15_and_10_percent (x y : ℝ) 
  (h1 : 3 = 0.15 * x) 
  (h2 : 3 = 0.10 * y) : 
  x - y = -10 := 
by 
  sorry

end problem_thre_is_15_and_10_percent_l668_668472


namespace sin_cos_identity_l668_668152

theorem sin_cos_identity :
  sin (70 * real.pi / 180) * sin (10 * real.pi / 180) + cos (10 * real.pi / 180) * cos (70 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l668_668152


namespace find_side_length_l668_668539

theorem find_side_length
  (n : ℕ) 
  (h : (6 * n^2) / (6 * n^3) = 1 / 3) : 
  n = 3 := 
by
  sorry

end find_side_length_l668_668539


namespace find_lambda_l668_668700

def vector (α : Type*) [Add α] [Mul α] := α × α

def dot_product {α : Type*} [Add α] [Mul α] [AddGroup α] (v1 v2 : vector α) : α :=
v1.1 * v2.1 + v1.2 * v2.2

variables (λ : ℝ)

def a : vector ℝ := (2, 0)
def b : vector ℝ := (1, 2)
def c : vector ℝ := (1, -2)

theorem find_lambda (h : dot_product (a - λ • b) c = 0) : λ = -2 / 3 :=
by sorry

end find_lambda_l668_668700


namespace screening_methods_l668_668535

theorem screening_methods (units : ℕ) (docs : ℕ) (h_units : units = 4) (h_docs : docs = 4) :
  (docs ^ units) = 4^4 := by
  rw [h_units, h_docs]
  sorry

end screening_methods_l668_668535


namespace speed_of_car_in_second_hour_l668_668461

theorem speed_of_car_in_second_hour
(speed_in_first_hour : ℝ)
(average_speed : ℝ)
(total_time : ℝ)
(speed_in_second_hour : ℝ)
(h1 : speed_in_first_hour = 100)
(h2 : average_speed = 65)
(h3 : total_time = 2)
(h4 : average_speed = (speed_in_first_hour + speed_in_second_hour) / total_time) :
  speed_in_second_hour = 30 :=
by {
  sorry
}

end speed_of_car_in_second_hour_l668_668461


namespace sqrt_nested_eq_five_l668_668612

theorem sqrt_nested_eq_five {x : ℝ} (h : x = Real.sqrt (15 + x)) : x = 5 :=
sorry

end sqrt_nested_eq_five_l668_668612


namespace construct_triangle_l668_668216

-- Define the line l and points E, F
variable {Field : Type*} [Field K] [Inhabited K]
variable (l : line K) (E F : point K)

-- Define and state the theorem to construct triangle ABC
theorem construct_triangle 
  (h1 : ∃ A C : point K, A ≠ C ∧ A ∈ l ∧ C ∈ l) 
  (h2 : ∃ B : point K, 
          ∃ A C : point K, 
          A ≠ C ∧ A ∈ l ∧ C ∈ l ∧
          E ≠ B ∧ F ≠ B ∧
          collinear {A, B, E} ∧ collinear {B, C, F}) : 
          ∃ (A B C : point K), 
          A ≠ B ∧ B ≠ C ∧ collinear {A, C} ∧ 
          collinear {A, B, E} ∧ collinear {B, C, F} := 
begin
  sorry
end

end construct_triangle_l668_668216


namespace sufficient_but_not_necessary_l668_668657

variable (x : ℝ)

def p := abs x > 1
def q := x < -2

theorem sufficient_but_not_necessary :
  (p x → q x) ∧ ¬ (q x → p x) :=
by
  split
  · intro h
    cases abs_lt.1 (lt_trans (neg_lt_neg_iff.2 h) zero_lt_one)
    exact lt_trans h (neg_lt_self zero_lt_one)
  · intro h
    have : (q x → p x) → (q x → false) → false,
    from λ h1 h2, h2 (λ h3, h3 (h1 h3)),
    exact this 
      (λ h1 : q x → p x, 
        if h3 : x = 2 then h3.symm ▸ absurd (h1 (by norm_num)) (by norm_num) 
        else false.elim (ne_of_lt h (ne_of_lt (lt_trans (neg_lt_neg_iff.2 h1 _) zero_lt_one))))
      (λ h, h)
  sorry 

end sufficient_but_not_necessary_l668_668657


namespace v1_correct_l668_668867

-- Define the polynomial function
def f (x : ℝ) : ℝ := 3 * x ^ 4 + 2 * x ^ 2 + x + 4

-- Using Horner's method
def f_horner (x : ℝ) : ℝ := (((3 * x + 0) * x + 2) * x + 1) * x + 4

-- Define the intermediate value v1 for x = 10 using Horner's method
def v1 (x : ℝ) : ℝ := 3 * x + 0

theorem v1_correct : v1 10 = 30 :=
by
  -- beginning of the proof
  unfold v1 -- expanding the definition of v1
  simp -- performing the simplification
  -- end of the proof
  exact rfl

end v1_correct_l668_668867


namespace pq_relationship_l668_668642

variable (a : ℝ)

def p : Prop := (ax-1)*(x-1) > 0 = (1/a, 1)
def q : Prop := a < 1/2

theorem pq_relationship : (p a → q a) ∧ ¬(q a → p a) := by
  sorry

end pq_relationship_l668_668642


namespace vertical_asymptotes_count_eq_two_l668_668221

theorem vertical_asymptotes_count_eq_two :
  let f := λ x : ℝ, (2 * x - 2) / (x^2 + 10 * x - 24) in
  ∃ (asymptotes : Finset ℝ), asymptotes.card = 2 ∧
    ∀ (x : ℝ), x ∈ asymptotes ↔ (x^2 + 10 * x - 24 = 0 ∧ ¬ (2 * x - 2 = 0)) :=
by
  let f := λ x : ℝ, (2 * x - 2) / (x^2 + 10 * x - 24)
  sorry

end vertical_asymptotes_count_eq_two_l668_668221


namespace inscribed_circle_distance_l668_668449

theorem inscribed_circle_distance (AC BC : ℤ) (hAC : AC = 36) (hBC : BC = 48) :
  let AB := Real.sqrt (AC^2 + BC^2)
      r := (AC + BC - AB) / 2
      CH := (AC * BC) / AB
      CP := CH - r
      OC := r * Real.sqrt 2
      OP := Real.sqrt (OC^2 - CP^2)
  in OP = 12 / 5 :=
by
  sorry

end inscribed_circle_distance_l668_668449


namespace largest_prime_factor_of_3913_l668_668490

def is_prime (n : ℕ) := nat.prime n

def prime_factors_3913 := {17, 2, 5, 23}

theorem largest_prime_factor_of_3913 
  (h1 : is_prime 17)
  (h2 : is_prime 2)
  (h3 : is_prime 5)
  (h4 : is_prime 23)
  (h5 : 3913 = 17 * 2 * 5 * 23) : 
  (23 ∈ prime_factors_3913 ∧ ∀ x ∈ prime_factors_3913, x ≤ 23) :=
  by 
    sorry

end largest_prime_factor_of_3913_l668_668490


namespace max_value_f1_on_interval_range_of_a_g_increasing_l668_668275

noncomputable def f1 (x : ℝ) : ℝ := 2 * x^2 + x + 2

theorem max_value_f1_on_interval : 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f1 x ≤ 5) ∧ 
  (∃ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ f1 x = 5) :=
sorry

noncomputable def f2 (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Set.Icc (1 : ℝ) (2 : ℝ) → f2 a x / x ≥ 2) → a ≥ 1 :=
sorry

noncomputable def g (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a + (1 - (a-1) * x^2) / x

theorem g_increasing (a : ℝ) : 
  (∀ x1 x2, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → g a x1 < g a x2) → a ≥ 1 / 16 :=
sorry

end max_value_f1_on_interval_range_of_a_g_increasing_l668_668275


namespace smallest_n_satisfying_congruence_l668_668574

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end smallest_n_satisfying_congruence_l668_668574


namespace events_complementary_l668_668977

open Set

theorem events_complementary : 
  let A := {1, 3, 5}
  let B := {1, 2, 3}
  let C := {4, 5, 6}
  let S := {1, 2, 3, 4, 5, 6}
  (B ∩ C = ∅) ∧ (B ∪ C = S) :=
by
  let A := {1, 3, 5}
  let B := {1, 2, 3}
  let C := {4, 5, 6}
  let S := {1, 2, 3, 4, 5, 6}
  have h₁ : B ∩ C = ∅ := by sorry
  have h₂ : B ∪ C = S := by sorry
  exact ⟨h₁, h₂⟩
  sorry

end events_complementary_l668_668977


namespace inverse_contrapositive_l668_668087

theorem inverse_contrapositive (α : ℝ) :
  (α = π / 4 → tan α = 1) ↔ (tan α ≠ 1 → α ≠ π / 4) :=
sorry

end inverse_contrapositive_l668_668087


namespace area_ratio_l668_668480

-- Define the points O, P, and X
variables {O P X : Point}

-- Define the distances
variables (r1 r2 : ℝ)

-- Define the two circles as having the same center O
def circle1 : Circle := Circle.mk O r1
def circle2 : Circle := Circle.mk O r2

-- Given condition that point X is two-thirds the way from O to P
def segment_OP := Segment.mk O P
def point_X_on_segment_OP : IsPointOnSegment X segment_OP := sorry

-- Given that OX = 2/3 OP
axiom distance_OX_to_OP : dist O X = 2 / 3 * dist O P

-- Area function defined for circles
def circle_area (r : ℝ) : ℝ := π * r^2

-- The statement to prove
theorem area_ratio : circle_area (dist O X) / circle_area (dist O P) = 4 / 9 := sorry

end area_ratio_l668_668480


namespace solve_abs_inequality_l668_668245

theorem solve_abs_inequality (x : ℝ) : 
  (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := 
by sorry

end solve_abs_inequality_l668_668245


namespace sunzi_carriage_l668_668150

theorem sunzi_carriage (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  ((Three people share a carriage, leaving two carriages empty) ∧ (Two people share a carriage, leaving nine people walking)) := sorry

end sunzi_carriage_l668_668150


namespace room_width_l668_668089

theorem room_width (length : ℕ) (total_cost : ℕ) (cost_per_sqm : ℕ) : ℚ :=
  let area := total_cost / cost_per_sqm
  let width := area / length
  width

example : room_width 9 38475 900 = 4.75 := by
  sorry

end room_width_l668_668089


namespace percentage_shaded_area_l668_668088

def num_of_smaller_triangles (total_triangles: ℕ) (shaded_triangles: ℕ) :=
  shaded_triangles / total_triangles

theorem percentage_shaded_area (total_triangles: ℕ) (shaded_triangles: ℕ) (h_total: total_triangles = 25) (h_shaded: shaded_triangles = 22) :
  num_of_smaller_triangles total_triangles shaded_triangles * 100 = 88 :=
by
  rw [h_total, h_shaded]
  have h: num_of_smaller_triangles 25 22 = 88 / 100 := by norm_num
  rw [h]
  norm_num

end percentage_shaded_area_l668_668088


namespace distance_between_centers_l668_668864

theorem distance_between_centers (r₁ r₂ d c : ℝ) (r₁_pos : r₁ = 10) (r₂_pos : r₂ = 6) (d_pos : d = 20) :
  c = real.sqrt (d^2 + (r₁ - r₂)^2) := 
by 
  have h : c = real.sqrt (20^2 + (10 - 6)^2) := sorry,
  have h_simplified : c = 4 * real.sqrt 26 := by 
    -- Proof omitted: Exact simplification corresponds to the conditions
    sorry,
  exact h_simplified

end distance_between_centers_l668_668864


namespace log_simplification_l668_668054

-- Define the variables and their conditions
variables (a b m : ℝ)
hypothesis (h : m^2 = a^2 - b^2)

-- Prove the simplified expression is 0
theorem log_simplification : (log (a + b) m + log (a - b) m - 2 * log (a + b) m * log (a - b) m) = 0 :=
sorry

end log_simplification_l668_668054


namespace perpendicular_iff_line_passes_l668_668270

open Set
open Classical

variable {p a b : ℝ} (hyp_p_positive : 0 < p)

def parabola : Set (ℝ × ℝ) := { A | A.2^2 = 2 * p * A.1 }

def fixed_point (A : ℝ × ℝ) := A ∈ parabola
def point_M : ℝ × ℝ := (2 * p + a, -b)
def line_passes_through (P Q : ℝ × ℝ) (M : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, ∀ t : ℝ, ((t * (P.1 - Q.1) + P.1, t * (P.2 - Q.2) + P.2) = M)

def perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.snd - b) / (P.fst - a) * (Q.snd - b) / (Q.fst - a) = -1

theorem perpendicular_iff_line_passes (A P Q : ℝ × ℝ) (hypA : fixed_point A) :
  perpendicular P Q ↔ line_passes_through P Q point_M :=
sorry

end perpendicular_iff_line_passes_l668_668270


namespace number_of_points_in_circle_radius_2_satisfying_cond_l668_668763

theorem number_of_points_in_circle_radius_2_satisfying_cond {n : ℝ} :
  ∀ (P : ℝ × ℝ), (dist P A)^2 + (dist P B)^2 = 5 →
  ∃ infinity_points, ∀ (P_in : ℝ × ℝ), P_in ∈ infinite_points ∧ dist P_in (0, 0) ≤ 2 := sorry

end number_of_points_in_circle_radius_2_satisfying_cond_l668_668763


namespace cyclic_inequality_l668_668512

variables {a b c : ℝ}

theorem cyclic_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (ab / (a + b + 2 * c) + bc / (b + c + 2 * a) + ca / (c + a + 2 * b)) ≤ (a + b + c) / 4 :=
sorry

end cyclic_inequality_l668_668512


namespace twenty_five_percent_greater_l668_668518

theorem twenty_five_percent_greater (x : ℕ) (h : x = (88 + (88 * 25) / 100)) : x = 110 :=
sorry

end twenty_five_percent_greater_l668_668518


namespace number_of_truthful_dwarfs_l668_668593

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l668_668593


namespace nine_point_circle_tangent_l668_668397

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def ninePointCircle (A B C : Point) : Circle := sorry
noncomputable def incircle (A B C : Point) : Circle := sorry
noncomputable def excircles (A B C : Point) : Set Circle := sorry

structure Triangle (P Q R : Point)

def isTangents (C1 C2 : Circle) : Prop := sorry

theorem nine_point_circle_tangent
  (A B C : Point) (H : Point)
  (hH : H = orthocenter A B C) :
  let nine_PC := ninePointCircle A B C
  in (isTangents nine_PC (incircle A H B)) ∧ 
     (∃ ex_ABCs, ex_ABCs = excircles A H B ∧ ∀ ex, ex ∈ ex_ABCs → isTangents nine_PC ex) ∧
     (isTangents nine_PC (incircle B H C)) ∧ 
     (∃ ex_BHCs, ex_BHCs = excircles B H C ∧ ∀ ex, ex ∈ ex_BHCs → isTangents nine_PC ex) ∧
     (isTangents nine_PC (incircle C H A)) ∧ 
     (∃ ex_CHAs, ex_CHAs = excircles C H A ∧ ∀ ex, ex ∈ ex_CHAs → isTangents nine_PC ex) :=
sorry

end nine_point_circle_tangent_l668_668397


namespace distance_between_vertices_of_hyperbola_l668_668620

def hyperbola_distance : ℝ :=
  let a := real.sqrt 16
  let b := real.sqrt 4
  2 * a

theorem distance_between_vertices_of_hyperbola :
    ∀ (x y : ℝ), x^2 / 16 - y^2 / 4 = 1 → hyperbola_distance = 8 :=
begin
  intros x y h,
  sorry
end

end distance_between_vertices_of_hyperbola_l668_668620


namespace smallest_n_satisfying_congruence_l668_668575

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end smallest_n_satisfying_congruence_l668_668575


namespace ball_radius_l668_668158

theorem ball_radius
  (diameter_hole : ℝ)
  (depth_hole : ℝ)
  (r : ℝ)
  (hole_radius : ℝ := diameter_hole / 2)
  (eq1 : diameter_hole = 24)
  (eq2 : depth_hole = 8)
  (pythagorean : hole_radius^2 + (r - depth_hole)^2 = r^2):
  r = 13 :=
by
-- NOTE: proof context
-- hole_radius = 12 (half of diameter_hole)
-- pythagorean equation to simplify to 13
suffices : hole_radius = 12, from
  sorry

end ball_radius_l668_668158


namespace books_taken_out_on_friday_l668_668466

theorem books_taken_out_on_friday :
  ∀ (x y z w : ℕ) (books_after_tuesday books_after_thursday k : ℕ),
  x = 235 →
  y = 227 →
  z = 56 →
  w = 29 →
  books_after_tuesday = x - y →
  books_after_thursday = books_after_tuesday + z →
  k = books_after_thursday - w →
  k = 35 :=
by
  intros x y z w books_after_tuesday books_after_thursday k
  intro hx hy hz hw hbt hbn hfr
  rw [hx, hy, hz, hw, hbt, hbn, hfr]
  sorry

end books_taken_out_on_friday_l668_668466


namespace mandy_books_ratio_l668_668028

theorem mandy_books_ratio (X : ℕ) :
  let pages_at_6 := 8 
  let pages_at_12 := pages_at_6 * 5
  let pages_at_20 := pages_at_12 * X
  let pages_present := pages_at_20 * 4
  pages_present = 480 → (X = 3) → ratio (pages_at_20) (pages_at_12) = 3 :=
by
  intros
  sorry

end mandy_books_ratio_l668_668028


namespace Johnny_is_8_l668_668124

-- Define Johnny's current age
def johnnys_age (x : ℕ) : Prop :=
  x + 2 = 2 * (x - 3)

theorem Johnny_is_8 (x : ℕ) (h : johnnys_age x) : x = 8 :=
sorry

end Johnny_is_8_l668_668124


namespace tournament_scheduling_ways_l668_668485

/-- Representing the main conditions -/
structure Tournament :=
(players_per_university : ℕ)
(games_per_pair : ℕ)
(total_rounds : ℕ)
(games_per_round : ℕ)
(no_repeat_in_consecutive_rounds : Bool)

def westlake_eastgate_tournament : Tournament := {
  players_per_university := 3,
  games_per_pair := 2,
  total_rounds := 6,
  games_per_round := 3,
  no_repeat_in_consecutive_rounds := true
}

/-- Main theorem to prove the number of ways to schedule the tournament -/
theorem tournament_scheduling_ways (t : Tournament) :
  t = westlake_eastgate_tournament → 1296 :=
by
  intros
  sorry

end tournament_scheduling_ways_l668_668485


namespace total_workers_l668_668438

-- Define the conditions as variables and assumptions
variables {W N : ℕ}

-- Define the conditions
def avg_salary_all_workers (W : ℕ) : Prop := W * 8000
def avg_salary_tech (N : ℕ) : Prop := 7 * 12000
def avg_salary_non_tech (N : ℕ) : Prop := N * 6000

-- Write the equations derived from the conditions
def salary_equation (W N : ℕ) := (7 * 12000) + (N * 6000) = W * 8000
def workers_equation (W N : ℕ) := W = 7 + N

-- State the theorem to be proven: that W must be 21 given the equations
theorem total_workers (W N : ℕ) (h1 : salary_equation W N) (h2 : workers_equation W N) : W = 21 :=
sorry

end total_workers_l668_668438


namespace proof_FC_bisects__l668_668380

open EuclideanGeometry

-- Lets define the original conditions

variables {A B C D E F M : Point} 
variables (hABC : IsRightTriangle A B C)
variables (hM : isMidpoint M A B)
variables (hD : onLine D A C ∧ Distance A D = 2 * Distance D C)
variables (hE : onLine E B C ∧ Distance B E = 2 * Distance E C)
variables (hF : collinear [A, E, F] ∧ collinear [D, M, F])

theorem proof_FC_bisects_∠DFE:
  angle Bisects (F, C, E) (F, D, E) := by
  sorry

end proof_FC_bisects__l668_668380


namespace divides_all_l668_668384

theorem divides_all (p : ℕ) (a b c : ℤ) 
  (h_prime : p.prime) 
  (h_odd : 2 < p) 
  (h1 : p ∣ a ^ 2023 + b ^ 2023) 
  (h2 : p ∣ b ^ 2024 + c ^ 2024) 
  (h3 : p ∣ a ^ 2025 + c ^ 2025) : 
  p ∣ a ∧ p ∣ b ∧ p ∣ c :=
sorry

end divides_all_l668_668384


namespace impossible_segment_in_pentagon_l668_668753

theorem impossible_segment_in_pentagon :
  ∀ (P : Type) [EuclideanSpace P] [RegularPentagon P],
    ¬ ∃ (XY : Segment P), is_inside_pentagon XY ∧ 
      (∀ (v : Vertex P), ∃ (α : Angle), is_seen_from v α XY) :=
begin
  sorry
end

end impossible_segment_in_pentagon_l668_668753


namespace value_of_expression_l668_668710

-- Define the variables and conditions
variables (x y : ℝ)
axiom h1 : x + 2 * y = 4
axiom h2 : x * y = -8

-- Define the statement to be proven
theorem value_of_expression : x^2 + 4 * y^2 = 48 := 
by
  sorry

end value_of_expression_l668_668710


namespace net_population_increase_one_day_l668_668724

-- Definitions of rates and hours
def peak_hours_birth_rate : ℝ := 7 / 2 -- people per second
def off_peak_hours_birth_rate : ℝ := 3 / 2 -- people per second
def peak_hours_death_rate : ℝ := 1 / 2 -- people per second
def off_peak_hours_death_rate : ℝ := 2 / 2 -- people per second
def peak_hours_in_migration_rate : ℝ := 1 / 4 -- people per second
def off_peak_hours_out_migration_rate : ℝ := 1 / 6 -- people per second
def hours_per_day : ℝ := 24
def hours_peak : ℝ := 12
def seconds_per_hour : ℝ := 3600

-- Calculation of net increase
def net_population_increase (hours_peak : ℝ) (seconds_per_hour : ℝ) : ℝ :=
  let peak_births := peak_hours_birth_rate * hours_peak * seconds_per_hour
  let peak_deaths := peak_hours_death_rate * hours_peak * seconds_per_hour
  let peak_in_migration := peak_hours_in_migration_rate * hours_peak * seconds_per_hour
  let off_peak_births := off_peak_hours_birth_rate * hours_peak * seconds_per_hour
  let off_peak_deaths := off_peak_hours_death_rate * hours_peak * seconds_per_hour
  let off_peak_out_migration := off_peak_hours_out_migration_rate * hours_peak * seconds_per_hour
  let net_peak_hours := peak_births - peak_deaths + peak_in_migration
  let net_off_peak_hours := off_peak_births - off_peak_deaths - off_peak_out_migration
  net_peak_hours + net_off_peak_hours

-- Formal statement of the theorem
theorem net_population_increase_one_day : net_population_increase hours_peak seconds_per_hour = 154800 :=
by
  sorry

end net_population_increase_one_day_l668_668724


namespace num_arrangements_correct_l668_668204

def num_arrangements : ℕ :=
  let n := 7 -- number of staff and days
  let a_possibilities := 5 -- A has 5 possible days (not May 1st or May 7th)
  let b_possibilities := 5 -- B has 5 possible days (not May 7th)
  (P 6 6) + (a_possibilities * b_possibilities * (P 5 5))

theorem num_arrangements_correct : num_arrangements = 3720 := 
  by 
  sorry

end num_arrangements_correct_l668_668204


namespace minimum_distance_l668_668670

theorem minimum_distance (x y : ℝ) (h : x - y - 1 = 0) : (x - 2)^2 + (y - 2)^2 ≥ 1 / 2 :=
sorry

end minimum_distance_l668_668670


namespace value_of_x_squared_plus_reciprocal_squared_l668_668025

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 0 < x) (h : x + 1/x = Real.sqrt 2020) : x^2 + 1/x^2 = 2018 :=
sorry

end value_of_x_squared_plus_reciprocal_squared_l668_668025


namespace sum_of_coeff_l668_668359

theorem sum_of_coeff (x y : ℕ) (n : ℕ) (h : 2 * x + y = 3) : (2 * x + y) ^ n = 3^n := 
by
  sorry

end sum_of_coeff_l668_668359


namespace prime_p_and_p_squared_plus_2_prime_l668_668133

theorem prime_p_and_p_squared_plus_2_prime (p : ℕ) (hp : Nat.prime p) (hp2_plus_2_prime : Nat.prime (p^2 + 2)) : p = 3 :=
sorry

end prime_p_and_p_squared_plus_2_prime_l668_668133


namespace vasya_days_without_purchases_l668_668901

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668901


namespace four_by_four_grid_arrangements_l668_668635

theorem four_by_four_grid_arrangements :
  let n := 4 in
  let letters := ["A", "B", "C", "D"] in
  let grid := matrix (fin n) (fin n) (option string) in
  (forall i : fin n, ∃! j : fin n, grid i j = some "A") ∧ -- each row has exactly one A
  (forall j : fin n, ∃! i : fin n, grid i j = some "A") ∧ -- each column has exactly one A
  grid 0 0 = some "A" ∧
  (forall i : fin n, grid i i = some "A") → -- A's are placed in the diagonal
  (Σ' t : set (matrix (fin n) (fin n) (option string)), 
    (∀ g ∈ t, 
      (∀ i, ∃! j, g i j = some "B") ∧
      (∀ j, ∃! i, g i j = some "B") ∧
      (∀ i, ∃! j, g i j = some "C") ∧
      (∀ j, ∃! i, g i j = some "C") ∧
      (∀ i, ∃! j, g i j = some "D") ∧
      (∀ j, ∃! i, g i j = some "D")
    )
  ).card = 144 :=
by sorry


end four_by_four_grid_arrangements_l668_668635


namespace sum_of_pairwise_rel_prime_integers_l668_668857

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end sum_of_pairwise_rel_prime_integers_l668_668857


namespace contrapositive_iff_l668_668076

theorem contrapositive_iff (a b : ℝ) :
  (a^2 - b^2 = 0 → a = b) ↔ (a ≠ b → a^2 - b^2 ≠ 0) :=
by
  sorry

end contrapositive_iff_l668_668076


namespace diminishing_allocation_proof_l668_668851

noncomputable def diminishing_allocation_problem : Prop :=
  ∃ (a b m : ℝ), 
  a = 0.2 ∧
  b * (1 - a)^2 = 80 ∧
  b * (1 - a) + b * (1 - a)^3 = 164 ∧
  b + 80 + 164 = m ∧
  m = 369

theorem diminishing_allocation_proof : diminishing_allocation_problem :=
by
  sorry

end diminishing_allocation_proof_l668_668851


namespace positive_divisors_eight_fold_application_l668_668767

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem positive_divisors_eight_fold_application (x : ℝ) (h : x = 5 / 2) :
  let fx := (λ n, Nat.floor (f^[n] x)) in 
  Nat.find_divisors (fx 8) = 6562 :=
by sorry

end positive_divisors_eight_fold_application_l668_668767


namespace exists_same_ties_exists_same_losses_white_l668_668983

-- Condition: each player plays every other player twice, and all players have the same number of points
variable {n : ℕ} (num_players : Fin n) (num_matches : Fin (2 * (n - 1)))

-- Defining players and their number of ties and losses with white pieces
def player (p : Fin n) : Type := (ties : Fin (2 * (n - 1)), losses_white : Fin n)

-- Every player has the same total score
axiom total_score_equal : ∀ p : Fin n, score p = score (0 : Fin n)

-- Points system
def score (p : Fin n) : ℚ := victories p + 0.5 * ties p

-- Tied games scenario
theorem exists_same_ties : ∃ p₁ p₂ : Fin n, p₁ ≠ p₂ ∧ ties p₁ = ties p₂ := 
by sorry

-- Lost games when playing with white pieces scenario
theorem exists_same_losses_white : ∃ p₁ p₂ : Fin n, p₁ ≠ p₂ ∧ losses_white p₁ = losses_white p₂ := 
by sorry

end exists_same_ties_exists_same_losses_white_l668_668983


namespace orange_beads_in_necklace_l668_668163

theorem orange_beads_in_necklace (O : ℕ) : 
    (∀ g w o : ℕ, g = 9 ∧ w = 6 ∧ ∃ t : ℕ, t = 45 ∧ 5 * (g + w + O) = 5 * (9 + 6 + O) ∧ 
    ∃ n : ℕ, n = 5 ∧ n * (45) =
    n * (5 * O)) → O = 9 :=
by
  sorry

end orange_beads_in_necklace_l668_668163


namespace smallest_n_l668_668121

theorem smallest_n (n : ℕ) (h : 5 * n % 26 = 220 % 26) : n = 18 :=
by
  -- Initial congruence simplification
  have h1 : 220 % 26 = 12 := by norm_num
  rw [h1] at h
  -- Reformulation of the problem
  have h2 : 5 * n % 26 = 12 := h
  -- Conclude the smallest n
  sorry

end smallest_n_l668_668121


namespace crate_weight_l668_668545

variable (C : ℝ)
variable (carton_weight : ℝ := 3)
variable (total_weight : ℝ := 96)
variable (num_crates : ℝ := 12)
variable (num_cartons : ℝ := 16)

theorem crate_weight :
  (num_crates * C + num_cartons * carton_weight = total_weight) → (C = 4) :=
by
  sorry

end crate_weight_l668_668545


namespace trigonometric_expression_l668_668263

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end trigonometric_expression_l668_668263


namespace smallest_integer_n_exists_l668_668503

-- Define the conditions
def lcm_gcd_correct_division (a b : ℕ) : Prop :=
  (lcm a b) / (gcd a b) = 44

-- Define the main problem
theorem smallest_integer_n_exists : ∃ n : ℕ, lcm_gcd_correct_division 60 n ∧ 
  (∀ k : ℕ, lcm_gcd_correct_division 60 k → k ≥ n) :=
begin
  sorry
end

end smallest_integer_n_exists_l668_668503


namespace vasya_days_l668_668911

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l668_668911


namespace equal_real_roots_imp_a_eq_4_l668_668339

theorem equal_real_roots_imp_a_eq_4 : 
  ∀ (a: ℝ), (∀ x: ℝ, (x^2 + a * x + 4 = 0 -> x ∈ ℝ)) ->
  (x^2 + a * x + 4 = 0) ∧ (a^2 - 4 * 1 * 4 = 0 ) -> a = 4 :=
begin
  sorry
end

end equal_real_roots_imp_a_eq_4_l668_668339


namespace elberta_has_l668_668324

-- Define the amounts in the problem
def granny_smith : ℝ := 45
def anjou : ℝ := 1 / 4 * granny_smith
def elberta : ℝ := anjou + 5

-- State the problem as a theorem
theorem elberta_has (h : elberta = 16.25) : true :=
by
  -- This proof is omitted
  sorry

end elberta_has_l668_668324


namespace heavy_tailed_permutations_count_l668_668171

def is_heavy_tailed (p : Perm (Fin 5)) : Prop :=
  p 0 + p 1 < p 3 + p 4

theorem heavy_tailed_permutations_count :
  (univ.filter is_heavy_tailed).card = 48 :=
by sorry

end heavy_tailed_permutations_count_l668_668171


namespace correlation_signs_l668_668821

section CorrelationProblem

variables {X Y U V : list ℝ}

-- Given data points
def X_vals : list ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y_vals : list ℝ := [1, 2, 3, 4, 5]
def U_vals : list ℝ := [10, 11.3, 11.8, 12.5, 13]
def V_vals : list ℝ := [5, 4, 3, 2, 1]

-- Definitions of correlation coefficients
noncomputable def r1 := correlation X_vals Y_vals
noncomputable def r2 := correlation U_vals V_vals

-- The problem statement: proof that r2 < 0 < r1
theorem correlation_signs :
  r2 < 0 ∧ 0 < r1 := 
sorry

end CorrelationProblem

end correlation_signs_l668_668821


namespace truthful_dwarfs_count_l668_668601

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l668_668601


namespace count_visible_factor_numbers_151_200_l668_668971

def isVisibleFactorNumber (n : Nat) : Prop :=
  ∀ d : Nat, d ∈ (Nat.digits 10 n).filter (λ x => x ≠ 0) → n % d = 0

theorem count_visible_factor_numbers_151_200 :
  (Finset.filter isVisibleFactorNumber (Finset.range 200 \ Finset.range 151)).card = 15 :=
by
  sorry

end count_visible_factor_numbers_151_200_l668_668971


namespace vasya_days_without_purchase_l668_668913

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l668_668913


namespace correct_arrangements_count_l668_668467

noncomputable def count_arrangements : ℕ :=
  let num_male_volunteers := 4
  let num_female_volunteers := 2
  let num_elderly_people := 2
  let total_volunteers := num_male_volunteers + num_female_volunteers
  let total_elements := num_male_volunteers + num_elderly_people - 1  -- treating elderly as one element
  let A_n_k := λ n k, Nat.factorial n / Nat.factorial (n - k)
  A_n_k total_elements total_elements * A_n_k num_elderly_people num_elderly_people * A_n_k (total_elements + 1) num_female_volunteers

theorem correct_arrangements_count : count_arrangements = 7200 := by
  sorry

end correct_arrangements_count_l668_668467


namespace max_min_distance_between_lines_l668_668403

noncomputable def distance_between_lines (a b : ℝ) : ℝ :=
  |a - b| / real.sqrt 2

theorem max_min_distance_between_lines :
  (∀ (a b c: ℝ), a + b = -1 ∧ ab = c ∧ 0 ≤ c ∧ c ≤ 1 →
    distance_between_lines a b ≤ 1 / real.sqrt 2) ∧
  (∀ (a b c: ℝ), a + b = -1 ∧ ab = c ∧ 0 ≤ c ∧ c ≤ 1 →
    distance_between_lines a b) :=
begin
  sorry
end

end max_min_distance_between_lines_l668_668403


namespace cheryl_material_used_and_cost_l668_668563

-- Define the fractions needed
def material_1 : ℚ := 5 / 11
def material_2 : ℚ := 2 / 3
def material_3 : ℚ := 7 / 15
def leftover : ℚ := 25 / 55

-- Define the discount
def discount : ℚ := 0.15

-- Prove the total amount of material used and percentage paid
theorem cheryl_material_used_and_cost :
  let total_bought := material_1 + material_2 + material_3,
      used_material := total_bought - leftover,
      cost_paid_percentage := 100 * (1 - discount)
  in
  used_material = 187 / 165 ∧ cost_paid_percentage = 85 :=
  by
    sorry

end cheryl_material_used_and_cost_l668_668563


namespace who_applied_l668_668981

-- Definitions for the problem
def students := {A, B, C}
def applied (s : students) : Prop
def statement_A := ¬ applied C
def statement_B := applied A
def statement_C := statement_A

-- Only one student is lying
axiom only_one_lying : ∃ x : students, (∀ s : students, s ≠ x → truthful s) ∧ lying x

-- Defining the properties of a truthful and lying statement
def truthful (s : students) : Prop :=
  match s with
  | A => statement_A
  | B => statement_B
  | C => statement_C

-- Defining the lying student's property
def lying (s : students) : Prop := ¬ truthful s

-- Proof problem: Proving B is the applicant
theorem who_applied : applied B :=
by {
  sorry -- The proof needs to be filled in
}

end who_applied_l668_668981


namespace find_distance_l668_668173

-- Definitions based on given conditions
def speed : ℝ := 40 -- in km/hr
def time : ℝ := 6 -- in hours

-- Theorem statement
theorem find_distance (speed : ℝ) (time : ℝ) : speed = 40 → time = 6 → speed * time = 240 :=
by
  intros h1 h2
  rw [h1, h2]
  -- skipping the proof with sorry
  sorry

end find_distance_l668_668173


namespace vasya_days_without_purchases_l668_668899

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l668_668899


namespace f_is_odd_function_l668_668695

noncomputable def op_1 (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)
noncomputable def op_2 (a b : ℝ) : ℝ := real.sqrt ((a - b)^2)

noncomputable def f (x : ℝ) : ℝ := (op_1 2 x) / ((op_2 x 2) - 2)

theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x := 
by 
  sorry

end f_is_odd_function_l668_668695


namespace num_valid_n_l668_668252

theorem num_valid_n (f : ℕ → ℕ) : 
  (∑ n in Finset.filter (λ n, n ≤ 1000 ∧ ¬(f n % 5 = 0)) (Finset.range 1001), 1) = 18 :=
by
  let f (n : ℕ) := Int.floor (1004 / n) + Int.floor (1005 / n) + Int.floor (1006 / n)
  sorry

end num_valid_n_l668_668252


namespace circle_through_two_points_tangent_to_line_l668_668215

theorem circle_through_two_points_tangent_to_line 
  (A B : ℝ × ℝ) (l : ℝ → ℝ) :
  (∃ M : ℝ × ℝ, ∃ X : ℝ × ℝ, (M ∈ line_segment A B) ∧ (X ∈ l) ∧ (M ∈ l) ∧ 
  (dist M X = √(dist A M * dist B M))) ↔ ¬(parallel (line A B) l) :=
begin
  sorry
end

end circle_through_two_points_tangent_to_line_l668_668215


namespace problem_proof_l668_668296

noncomputable def triangle_expression (a b c : ℝ) (A B C : ℝ) : ℝ :=
  b^2 * (Real.cos (C / 2))^2 + c^2 * (Real.cos (B / 2))^2 + 
  2 * b * c * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2)

theorem problem_proof (a b c A B C : ℝ) (h1 : a + b + c = 16) : 
  triangle_expression a b c A B C = 64 := 
sorry

end problem_proof_l668_668296


namespace count_possible_values_of_n_l668_668451

-- Mathematically equivalent proof problem translated to Lean statement

open Real

theorem count_possible_values_of_n :
  ∃ (n_set : Set ℕ), (∀ n ∈ n_set, log 20 + log n > log 45 ∧ log 45 + log n > log 20 ∧ log 20 + log 45 > log n) ∧
  n_set.count = 897 := by
  sorry

end count_possible_values_of_n_l668_668451


namespace total_time_to_climb_first_seven_flights_l668_668207

theorem total_time_to_climb_first_seven_flights :
  ∀ (a d : ℕ), a = 25 → d = 7 → 
  (7 * (2 * a + 6 * d)) / 2 = 322 :=
by
  intros a d ha hd
  rw [ha, hd]
  sorry

end total_time_to_climb_first_seven_flights_l668_668207


namespace smallest_integer_of_lcm_gcd_l668_668501

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l668_668501


namespace range_of_a_l668_668639

theorem range_of_a (a : ℝ) : ({x : ℝ | a - 4 < x ∧ x < a + 4} ⊆ {x : ℝ | 1 < x ∧ x < 3}) → (-1 ≤ a ∧ a ≤ 5) := by
  sorry

end range_of_a_l668_668639


namespace smallest_n_for_area_gt_3000_l668_668566

noncomputable def complex_triangle_area (n : ℕ) : ℝ :=
let z1 := (n : ℂ) + complex.I,
    z2 := z1 ^ 2,
    z3 := z1 ^ 4 in
1 / 2 * abs (
  (re z1 * im z2 + re z2 * im z3 + re z3 * im z1) -
  (im z1 * re z2 + im z2 * re z3 + im z3 * re z1)
)

theorem smallest_n_for_area_gt_3000 : ∃ n : ℕ, 0 < n ∧ complex_triangle_area n > 3000 ∧ ∀ m : ℕ, 0 < m ∧ complex_triangle_area m > 3000 → n ≤ m :=
⟨20, by sorry, by sorry, by sorry⟩

end smallest_n_for_area_gt_3000_l668_668566


namespace right_triangle_area_l668_668828

theorem right_triangle_area (a b : ℝ) (h1 : b = 10) (θ : ℝ) (h2 : θ = Real.pi / 6) 
                            (area : ℝ) (h3 : area = 0.5 * (10 * b) * Real.sqrt(3)) : 
                            area = (50 * Real.sqrt(3)) / 3 :=
by
  sorry

end right_triangle_area_l668_668828


namespace distance_between_l1_and_l2_l668_668823

noncomputable def distance_between_parallel_lines := 
  let A1 := 9
  let B1 := 12
  let C1 := -6
  let A2 := 9
  let B2 := 12
  let C2 := -10
  real.abs (C2 - C1) / real.sqrt (A1^2 + B1^2)

theorem distance_between_l1_and_l2 :
  distance_between_parallel_lines = 4 / 15 :=
by
  sorry

end distance_between_l1_and_l2_l668_668823


namespace largest_root_range_l668_668996

theorem largest_root_range :
  ∀ (b_2 b_1 b_0 : ℝ), 
    (|b_2| < 3) ∧ (|b_1| < 3) ∧ (|b_0| < 3) 
    → ∃ r ∈ Ioo 3.5 5, ∃ x : ℝ, x^3 + b_2 * x^2 + b_1 * x + b_0 + x^4 = 0 :=
by
  intros b_2 b_1 b_0 h
  sorry

end largest_root_range_l668_668996


namespace sum_of_divisors_of_119_l668_668122

theorem sum_of_divisors_of_119 : ∑ d in (Finset.filter (λ x, 119 ∣ x) (Finset.range (119 + 1))), d = 144 :=
by
  sorry

end sum_of_divisors_of_119_l668_668122


namespace f_is_monotonically_decreasing_tangent_line_l668_668305

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x + (2 * Real.pi / 3))

-- Define the condition for monotonicity
theorem f_is_monotonically_decreasing (x : ℝ) (h : 0 < x ∧ x < (5 * Real.pi) / 12) :
  f'(x) < 0 :=
sorry

-- Define the condition for the tangent line
theorem tangent_line (l : ℝ → ℝ) :
  l = (λ x, (-x + (Real.sqrt 3 / 2))) ∧ 
  ∃ (x₀ : ℝ), f'(x₀) = -1 :=
sorry

end f_is_monotonically_decreasing_tangent_line_l668_668305


namespace transform_proof_l668_668240

-- Given the original piecewise function
def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-3 : ℝ) 0 then -2 - x
  else if x ∈ Set.Icc 0 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x ∈ Set.Icc 2 3 then 2 * (x - 2)
  else 0 -- Assuming the function is 0 outside the given intervals

-- The transformed function:
def g (x : ℝ) := -f (x / 3) - 3

theorem transform_proof :
  (∀ x, x ∈ Set.Icc (-9:ℝ) 0  → g x = (x / 3 - 1)) ∧
  (∀ x, x ∈ Set.Icc 0 6      → g x = - (Real.sqrt (4 - (x / 3 - 2)^2 ) - 2) - 3) ∧
  (∀ x, x ∈ Set.Icc 6 9      → g x = -2 * (x / 3 - 2) - 3) :=
  sorry

end transform_proof_l668_668240


namespace find_n_for_conditions_l668_668255

noncomputable def valid_n (n : ℕ) : Prop :=
  let a_n := 7 * n - 3 in
  (∃ k : ℤ, a_n = 5 * k) ∧ (∀ m : ℤ, a_n ≠ 3 * m)

theorem find_n_for_conditions :
  ∃ t : ℕ, t ≠ 3 * m - 1 ∧ valid_n (5 * t - 1) :=
sorry

end find_n_for_conditions_l668_668255


namespace rise_water_level_l668_668166

noncomputable def volume_cube (edge : ℝ) : ℝ :=
  edge ^ 3

noncomputable def base_area_rect_vessel (length width : ℝ) : ℝ :=
  length * width

noncomputable def rise_in_water_level (V_cube A_base : ℝ) : ℝ :=
  V_cube / A_base

theorem rise_water_level (edge length width : ℝ) (V_cube : ℝ) (A_base : ℝ) :
  edge = 12 →
  length = 20 →
  width = 15 →
  V_cube = volume_cube edge →
  A_base = base_area_rect_vessel length width →
  rise_in_water_level V_cube A_base = 5.76 :=
by
  intros h_edge h_length h_width h_V_cube h_A_base
  rw [h_edge, h_length, h_width, h_V_cube, h_A_base]
  norm_num
  sorry

end rise_water_level_l668_668166


namespace circle_equation_l668_668462

-- Definitions based on the conditions
def center_on_x_axis (a b r : ℝ) := b = 0
def tangent_at_point (a b r : ℝ) := (b - 1) / a = -1/2

-- Proof statement
theorem circle_equation (a b r : ℝ) (h1: center_on_x_axis a b r) (h2: tangent_at_point a b r) :
    ∃ (a b r : ℝ), (x - a)^2 + y^2 = r^2 ∧ a = 2 ∧ b = 0 ∧ r^2 = 5 :=
by 
  sorry

end circle_equation_l668_668462


namespace vasya_no_purchase_days_l668_668944

theorem vasya_no_purchase_days :
  ∃ (x y z w : ℕ), x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_no_purchase_days_l668_668944


namespace problem_1_part_1_proof_problem_1_part_2_proof_l668_668736

noncomputable def problem_1_part_1 : Real :=
  2 * Real.sqrt 2 + (Real.sqrt 6) / 2

theorem problem_1_part_1_proof:
  let θ₀ := 3 * Real.pi / 4
  let ρ_A := 4 * Real.cos θ₀
  let ρ_B := Real.sqrt 3 * Real.sin θ₀
  |ρ_A - ρ_B| = 2 * Real.sqrt 2 + (Real.sqrt 6) / 2 :=
  sorry

theorem problem_1_part_2_proof :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x - (Real.sqrt 3)/2 * y = 0) :=
  sorry

end problem_1_part_1_proof_problem_1_part_2_proof_l668_668736


namespace range_of_a_l668_668453

theorem range_of_a (a x y : ℝ) (h1 : 77 * a = (2 * x + 2 * y) / 2) (h2 : Real.sqrt (abs a) = Real.sqrt (x * y)) :
  a ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
sorry

end range_of_a_l668_668453


namespace prove_exponentiation_l668_668643

theorem prove_exponentiation (x y : ℝ) (h : abs (x + 2) + (y - 3)^2 = 0) : x ^ y = -8 :=
by
  have h1 : x + 2 = 0 := sorry
  have h2 : y - 3 = 0 := sorry
  have x_val : x = -2 := sorry  
  have y_val : y = 3 := sorry  
  show x ^ y = -8, from sorry

end prove_exponentiation_l668_668643


namespace vasya_days_without_purchase_l668_668921

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l668_668921


namespace smallest_k_that_square_sum_div_by_250_l668_668246

theorem smallest_k_that_square_sum_div_by_250 
: ∃ (k : ℕ), 
  (1^2 + 2^2 + ... + k^2) % 250 = 0 ∧ 
  k = 375 := by
  sorry

end smallest_k_that_square_sum_div_by_250_l668_668246


namespace count_intersections_l668_668445

-- Define the functions involved
def f₁ (x : ℝ) : ℝ := Real.log x / Real.log 2
def f₂ (x : ℝ) : ℝ := 1 / f₁ x -- using log_x 2 = 1 / log_2 x
def f₃ (x : ℝ) : ℝ := -f₁ x -- using log_{1/2} x = - log_2 x
def f₄ (x : ℝ) : ℝ := -f₂ x -- using log_x (1/2) = - 1 / log_2 x
def f₅ (x : ℝ) : ℝ := Real.sqrt (f₁ x)

-- Define the predicate that counts intersections
def intersects_at_least_two (x : ℝ) (y : ℝ) : Prop :=
  (y = f₁ x ∨ y = f₂ x ∨ y = f₃ x ∨ y = f₄ x ∨ y = f₅ x)
  ∧ ((y = f₁ x ∧ (y = f₂ x ∨ y = f₃ x ∨ y = f₄ x ∨ y = f₅ x))
  ∨ (y = f₂ x ∧ (y = f₁ x ∨ y = f₃ x ∨ y = f₄ x ∨ y = f₅ x))
  ∨ (y = f₃ x ∧ (y = f₁ x ∨ y = f₂ x ∨ y = f₄ x ∨ y = f₅ x))
  ∨ (y = f₄ x ∧ (y = f₁ x ∨ y = f₂ x ∨ y = f₃ x ∨ y = f₅ x))
  ∨ (y = f₅ x ∧ (y = f₁ x ∨ y = f₂ x ∨ y = f₃ x ∨ y = f₄ x)))

-- Define the main statement
theorem count_intersections : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ 
  ∧ 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ 
  ∧ (∃ y₁, intersects_at_least_two x₁ y₁) 
  ∧ (∃ y₂, intersects_at_least_two x₂ y₂)
  ∧ (∃ y₃, intersects_at_least_two x₃ y₃)) :=
sorry

end count_intersections_l668_668445


namespace odometer_reading_before_lunch_l668_668778
variable {Real : Type*} [AddGroupWithOne Real] -- Ensuring the type Real has necessary algebraic structure

theorem odometer_reading_before_lunch
  (odometer_before_trip : Real)
  (miles_traveled : Real)
  (odometer_reading_lunch : Real) :
  odometer_before_trip = 212.3 →
  miles_traveled = 159.7 →
  odometer_reading_lunch = odometer_before_trip + miles_traveled →
  odometer_reading_lunch = 372.0 := by
  intros
  sorry

end odometer_reading_before_lunch_l668_668778


namespace log_8_128_eq_7_over_3_l668_668613

theorem log_8_128_eq_7_over_3 : log 8 128 = 7 / 3 :=
by {
  -- Given conditions
  have h1 : 8 = 2 ^ 3 := by sorry,
  have h2 : 128 = 2 ^ 7 := by sorry,
  -- Based on these conditions and the properties of logarithms, we can derive the result
  sorry
}

end log_8_128_eq_7_over_3_l668_668613


namespace wire_cut_perimeter_equal_l668_668188

theorem wire_cut_perimeter_equal (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 4 * (a / 4) = 8 * (b / 8)) :
  a / b = 1 :=
sorry

end wire_cut_perimeter_equal_l668_668188


namespace spring_unextended_state_l668_668096

theorem spring_unextended_state {m k g a b t: ℝ} (h1: mg/k = 1) (h2: b = 1 - 2a):
  (∀ t, m ≠ 0 ∧ k ≠ 0 ∧ g ≠ 0 → (a * exp (-2 * t) + b * exp (-t) + m * g / k = 0) → 
       interval (1 + sqrt 3 / 2) 2 a) := 
begin
  sorry,
end

end spring_unextended_state_l668_668096


namespace find_k_l668_668827

theorem find_k (k : ℝ) (h_perimeter : |-1/k| + 1 + sqrt ((-1/k)^2 + 1) = 6) : k = 5 / 12 ∨ k = -5 / 12 :=
sorry

end find_k_l668_668827


namespace blackboard_area_difference_l668_668776

theorem blackboard_area_difference : 
  let area_square := 8 * 8 in
  let area_rectangle := 10 * 5 in
  area_square - area_rectangle = 14 := 
by
  let area_square := 8 * 8
  let area_rectangle := 10 * 5
  sorry

end blackboard_area_difference_l668_668776


namespace person_who_announced_17_picked_14_l668_668615

-- Define that we have 15 people in a circle
constant n : ℕ := 15

-- Define the picked numbers as a sequence a_i
noncomputable def a : ℕ → ℤ

-- Define the announced numbers
noncomputable def announced_numbers (i : ℕ) : ℤ :=
  let avg_neigh := (a ((i - 1) % n) + a ((i + 1) % n)) / 2
  in avg_neigh + 3

-- Define the known announced numbers sequence from 7 to 21
noncomputable def known_announced_numbers : ℕ → ℤ
| 0 => 7
| 1 => 8
| 2 => 9
| 3 => 10
| 4 => 11
| 5 => 12
| 6 => 13
| 7 => 14
| 8 => 15
| 9 => 16
| 10 => 17
| 11 => 18
| 12 => 19
| 13 => 20
| 14 => 21
| _ => 0 -- This is just a fallback case for the definition

-- State the problem as a theorem: the person who announced 17 (index 10) picked 14
theorem person_who_announced_17_picked_14 :
  a 10 = 14 := by
  sorry

end person_who_announced_17_picked_14_l668_668615


namespace determinant_of_A_l668_668992

-- Define the matrix
def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℤ
def A : matrix_2x2 := ![
  [8, 4],
  [-2, 3]
]

-- State the theorem
theorem determinant_of_A : Matrix.det A = 32 :=
by
  -- Placeholder for the proof
  sorry

end determinant_of_A_l668_668992


namespace racing_distance_l668_668865

-- Given definitions
def firstFriendPace (D : ℝ) := D / 21
def secondFriendPace (D : ℝ) := D / 24
def combinedTime (D : ℝ) : ℝ := 
  (5 / (firstFriendPace D)) + (5 / (secondFriendPace D))

-- Theorem stating the problem, with the desired proof
theorem racing_distance : ∃ D : ℝ, combinedTime D = 75 ∧ D = 3 := 
sorry

end racing_distance_l668_668865


namespace equivalence_of_congruence_and_divisibility_l668_668016

theorem equivalence_of_congruence_and_divisibility (n : ℕ) (h_ge_2 : n ≥ 2) :
  (∀ x : ℕ, Nat.coprime x n → x^6 % n = 1) ↔ n ∣ 504 := by
  sorry

end equivalence_of_congruence_and_divisibility_l668_668016


namespace shaded_region_area_calculation_l668_668544

noncomputable def calculate_area_of_shaded_region (num_squares : ℕ) (total_diagonal_length : ℝ) (area : ℝ) : Prop :=
  let d := total_diagonal_length / (num_squares : ℝ).sqrt in
  let area_of_one_square := (d^2) / 2 in
  num_squares = 24 ∧ total_diagonal_length = 10 ∧ area = 24 * area_of_one_square → area = 50

-- The proof statement
theorem shaded_region_area_calculation : calculate_area_of_shaded_region 24 10 50 :=
sorry

end shaded_region_area_calculation_l668_668544


namespace chessboard_division_impossible_l668_668782

theorem chessboard_division_impossible :
  ¬ ∃ (cuts : fin 13 → set (ℝ × ℝ)), 
    (∀ part, cardinal.mk (set.center_of_squares part) ≤ 1 ∧ set.covered parts)
    → (set.division_of_board 8 8 parts = 13) :=
sorry

end chessboard_division_impossible_l668_668782


namespace common_difference_greater_than_30000_l668_668153

theorem common_difference_greater_than_30000
  (a : ℕ) 
  (d : ℕ) 
  (h_arith_prog : ∀ n : ℕ, n < 15 → prime (a + n * d))
  (h_divisible : d % 2 = 0 ∧ d % 3 = 0 ∧ d % 5 = 0 ∧ d % 7 = 0 ∧ d % 11 = 0 ∧ d % 13 = 0) :
  d > 30000 := 
sorry

end common_difference_greater_than_30000_l668_668153


namespace problem1_problem2_l668_668702

-- Definitions
def vector := (ℝ × ℝ)
def vector_magnitude (v : vector) : ℝ := real.sqrt (v.1^2 + v.2^2)
def dot_product (v w : vector) : ℝ := v.1 * w.1 + v.2 * w.2
def parallel (v w : vector) : Prop := v.1 * w.2 = v.2 * w.1
def perpendicular (v w : vector) : Prop := dot_product v w = 0
def angle (u v : vector) : ℝ :=
  real.arccos ((dot_product u v) / (vector_magnitude u * vector_magnitude v))

-- Given vectors a, b, and c
noncomputable def a : vector := (1, 2)
noncomputable def c : vector := if true then (2, 4) else (-2, -4) -- Placeholders
noncomputable def b : vector := (0, 0) -- Placeholder

-- Problem 1
theorem problem1 : 
  vector_magnitude c = 2 * real.sqrt 5 → parallel c a → (c = (2, 4) ∨ c = (-2, -4)) :=
by sorry

-- Problem 2
theorem problem2 :
  vector_magnitude b = real.sqrt(5) / 2 → 
  perpendicular (a + (2, 1) • b) (2 • a - (1, 1) • b) →
  angle a b = real.pi :=
by sorry

end problem1_problem2_l668_668702


namespace right_triangle_sides_l668_668185

theorem right_triangle_sides {a b c : ℕ} (h1 : a * (b + 2) = 150) (h2 : a^2 + b^2 = c^2) (h3 : a + (1 / 2 : ℤ) * (a * b) = 75) :
  (a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
sorry

end right_triangle_sides_l668_668185


namespace radius_of_circle_S_l668_668740

theorem radius_of_circle_S (DE DF EF : ℝ) (rR : ℝ) (no_point_outside : Prop) :
  DE = 120 → DF = 120 → EF = 70 → rR = 20 →
  ∃ p q r : ℕ, r = 41 ∧ ∀ x, distinct_prime_factors r x →
  (radius_of_circle_S = (p - q * real.sqrt r) →
  (p + q * r = 255)) :=
by
  intros DE_eq DF_eq EF_eq rR_eq
  use 55, 5, 41
  split
  { exact sorry }
  { intros _ _ 
    exact sorry }

end radius_of_circle_S_l668_668740


namespace stephanie_store_visits_l668_668805

theorem stephanie_store_visits (oranges_per_visit total_oranges : ℕ) 
  (h1 : oranges_per_visit = 2)
  (h2 : total_oranges = 16) : 
  total_oranges / oranges_per_visit = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end stephanie_store_visits_l668_668805


namespace maple_is_taller_l668_668406

def pine_tree_height : ℚ := 13 + 1/4
def maple_tree_height : ℚ := 20 + 1/2
def height_difference : ℚ := maple_tree_height - pine_tree_height

theorem maple_is_taller : height_difference = 7 + 1/4 := by
  sorry

end maple_is_taller_l668_668406


namespace recovery_period_related_to_treatment_plan_distribution_and_expectation_of_X_recovery_after_7_days_l668_668446

-- Problem 1
theorem recovery_period_related_to_treatment_plan
  (a b c d n : ℕ)
  (chi_square : ℚ)
  (chi_square_critical : ℚ) :
  (a = 10) → (b = 45) → (c = 20) → (d = 30) → (n = 105) →
  (chi_square = (n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d)))) →
  (chi_square_critical = 3.841) →
  chi_square > chi_square_critical :=
by sorry

-- Problem 2
theorem distribution_and_expectation_of_X
  (P_X : ℕ → ℚ)
  (expectation : ℚ) :
  (P_X 0 = 1/6) → (P_X 1 = 1/2) → (P_X 2 = 3/10) → (P_X 3 = 1/30) →
  (expectation = 0 * 1/6 + 1 * 1/2 + 2 * 3/10 + 3 * 1/30) :=
by sorry

-- Problem 3
theorem recovery_after_7_days
  (Y : ℝ → ℝ)
  (P : ℝ → ℝ) :
  (∀ x, Y x = pdf_normal 5 1 x) →
  (P (Y 7) < 0.95) :=
by sorry

end recovery_period_related_to_treatment_plan_distribution_and_expectation_of_X_recovery_after_7_days_l668_668446


namespace radius_of_circle_l668_668839

def circle_eq_def (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

theorem radius_of_circle {x y r : ℝ} (h : circle_eq_def x y) : r = 3 := 
by
  -- Proof skipped
  sorry

end radius_of_circle_l668_668839


namespace solve_equation_l668_668057

theorem solve_equation :
  ∃ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 2) ∧ (x ≈ 4.695 ∨ x ≈ -3.195) :=
sorry

end solve_equation_l668_668057


namespace find_a_l668_668664

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (3^(x) + 1) / log 3 + 1/2 * a * x

theorem find_a (a : ℝ) (h_even : ∀ x : ℝ, f a x = f a (-x)) : a = -1 :=
by
  sorry

end find_a_l668_668664


namespace crease_length_l668_668541

theorem crease_length (A B C : Point)
  (h_triangle : is_right_triangle A B C 6 8 10)
  (folded : fold_point C A B = A) : 
  length_of_crease A C B = 20 / 3 :=
sorry

end crease_length_l668_668541


namespace fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l668_668342

theorem fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes :
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) -> n = 45 :=
by
  sorry

end fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l668_668342


namespace square_DF_length_l668_668259

def square_length_DF (a b : ℝ) : ℝ :=
  (1 / 2) * (a + Real.sqrt (a ^ 2 + b ^ 2) - Real.sqrt (b ^ 2 + 2 * a * Real.sqrt (a ^ 2 + b ^ 2) - 2 * a ^ 2))

theorem square_DF_length (a b : ℝ) :  
  let DF := (1 / 2) * (a + Real.sqrt (a ^ 2 + b ^ 2) - Real.sqrt (b ^ 2 + 2 * a * Real.sqrt (a ^ 2 + b ^ 2) - 2 * a ^ 2)) in 
  DF = square_length_DF a b :=
sorry

end square_DF_length_l668_668259


namespace largest_sum_product_l668_668838

theorem largest_sum_product (p q : ℕ) (h1 : p * q = 100) (h2 : 0 < p) (h3 : 0 < q) : p + q ≤ 101 :=
sorry

end largest_sum_product_l668_668838
