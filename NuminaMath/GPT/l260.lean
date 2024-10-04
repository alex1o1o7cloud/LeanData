import Mathlib

namespace bisect_angle_l260_260157

variables {A B C D M N Q : Type}
variables [parallelogram A B C D]
variables (M : A → B) (N : B → C)
variables (hM : M ≠ A ∧ M ≠ B) (hN : N ≠ B ∧ N ≠ C)
variables (hAMNC : distance A M = distance N C)
variables (AN CM : line_segment A B) (CM : line_segment C D)
variables (Q_intersection : intersection_point (line_through A N) (line_through C M) Q)

theorem bisect_angle (hM_N_conditions: (distance A M = distance N C)) :
  bisects_angle D Q (∠ADC) := sorry

end bisect_angle_l260_260157


namespace find_n_l260_260017

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end find_n_l260_260017


namespace remainder_of_sum_modulo_9_l260_260026

theorem remainder_of_sum_modulo_9 : 
  (8230 + 8231 + 8232 + 8233 + 8234 + 8235) % 9 = 0 := by
  sorry

end remainder_of_sum_modulo_9_l260_260026


namespace pizza_toppings_l260_260349

theorem pizza_toppings (n : ℕ) (h : n = 8) : 
  (1 + n + (n * (n - 1) / 2) = 37) :=
by
  rw h
  sorry

end pizza_toppings_l260_260349


namespace bacteria_lifespan_scientific_notation_l260_260620

theorem bacteria_lifespan_scientific_notation 
  (lifespan : ℝ) (h : lifespan = 0.000012) : lifespan = 1.2 * 10^(-5) := 
by 
  sorry

end bacteria_lifespan_scientific_notation_l260_260620


namespace least_three_digit_multiple_of_8_l260_260303

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l260_260303


namespace calc_elderly_employees_in_sampled_group_l260_260697

-- Definitions
variables {total_employees young_employees sampled_young elderly middle_aged : ℕ}
variables (ratio_sampled_elderly : ℚ)

-- Given conditions
def conditions : Prop :=
  total_employees = 430 ∧
  young_employees = 160 ∧
  sampled_young = 32 ∧
  middle_aged = 2 * elderly ∧
  elderly * 3 + young_employees = total_employees ∧
  ratio_sampled_elderly = (elderly : ℚ) / total_employees

-- Question: Calculate the number of elderly employees in the sample
def question : Prop :=
  ∃ (sampled_elderly : ℕ), ratio_sampled_elderly * sampled_young = 18 / 32

-- The theorem statement to be proved
theorem calc_elderly_employees_in_sampled_group :
  conditions →
  ∃ sampled_elderly : ℕ, ratio_sampled_elderly * sampled_young = 18 / 32 :=
begin
  sorry
end

end calc_elderly_employees_in_sampled_group_l260_260697


namespace paige_winter_clothing_l260_260322

theorem paige_winter_clothing :
  ∀ (boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ),
  boxes = 6 → scarves_per_box = 5 → mittens_per_box = 5 →
  (boxes * (scarves_per_box + mittens_per_box) = 60) :=
begin
  intros boxes scarves_per_box mittens_per_box h_boxes h_scarves h_mittens,
  rw [h_boxes, h_scarves, h_mittens],
  norm_num,
  exact rfl,
end

end paige_winter_clothing_l260_260322


namespace find_a6_l260_260522

def seq (n : ℕ) : ℤ :=
  if n = 1 then 2
  else if n = 2 then 5
  else seq (n + 1) + seq n

theorem find_a6 : seq 6 = -3 := by
  sorry

end find_a6_l260_260522


namespace intriguing_quadruples_count_l260_260384

-- Define the concept of an intriguing ordered quadruple
def is_intriguing_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d > b + c

-- State the theorem
theorem intriguing_quadruples_count : 
  { (a, b, c, d) : ℕ × ℕ × ℕ × ℕ // is_intriguing_quadruple a b c d }.to_finset.card = 420 :=
  sorry

end intriguing_quadruples_count_l260_260384


namespace harvest_weeks_l260_260568

/-- Lewis earns $403 every week during a certain number of weeks of harvest. 
If he has to pay $49 rent every week, and he earns $93,899 during the harvest season, 
we need to prove that the number of weeks in the harvest season is 265. --/
theorem harvest_weeks 
  (E : ℕ) (R : ℕ) (T : ℕ) (W : ℕ) 
  (hE : E = 403) (hR : R = 49) (hT : T = 93899) 
  (hW : W = 265) : 
  W = (T / (E - R)) := 
by sorry

end harvest_weeks_l260_260568


namespace solve_system_of_equations_l260_260237

theorem solve_system_of_equations :
    ∃ x y : ℚ, 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13 / 38 ∧ y = -4 / 19 :=
by
  sorry

end solve_system_of_equations_l260_260237


namespace power_function_decreasing_n_value_l260_260470

theorem power_function_decreasing_n_value (n : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < x → f x = (n^2 + 2 * n - 2) * x^(n^2 - 3 * n)) →
  (∀ x y : ℝ, 0 < x ∧ 0 < y → x < y → f y < f x) →
  n = 1 := 
by
  sorry

end power_function_decreasing_n_value_l260_260470


namespace divisors_of_64n4_l260_260034

theorem divisors_of_64n4 (n : ℕ) (hn : 0 < n) (hdiv : ∃ d, d = (120 * n^3) ∧ d.divisors.card = 120) : (64 * n^4).divisors.card = 375 := 
by 
  sorry

end divisors_of_64n4_l260_260034


namespace axis_of_symmetry_translated_sin_l260_260141

theorem axis_of_symmetry_translated_sin (k : ℤ) :
  let f := λ x, Real.sin (2 * x + π / 3)
  in ∃ x, f x = Real.sin (k * π + π / 2) ↔ x = (k * π / 2) + (π / 12) := 
sorry

end axis_of_symmetry_translated_sin_l260_260141


namespace EF_bisects_angle_CFD_l260_260712

open EuclideanGeometry

variables {l : Line} {τ : Circle} {C D : Point} {A B E F : Point}

/-- Proof that EF bisects ∠CFD given specific geometric conditions -/
theorem EF_bisects_angle_CFD 
  (h1 : is_on_tau τ C)
  (h2 : is_on_tau τ D)
  (h3 : is_tangent τ C l B)
  (h4 : is_tangent τ D l A)
  (h5 : is_between_center A B)
  (h6 : E = intersection (Line.mk A C) (Line.mk B D))
  (h7 : F = foot_of_perpendicular E l) : 
  bisects_angle E F C D :=
sorry

/-- Definitions for the required geometric properties -/
def is_on_tau (τ : Circle) (P : Point) : Prop := ∃ r : ℝ, dist τ.center P = r

def is_tangent (τ : Circle) (P: Point) (l : Line) (Q : Point) : Prop := 
  is_on_tau τ P ∧ ⊥ (Line.mk P τ.center) l ∧ Q ∈ l

def is_between_center (centre : Point) (A B : Point) : Prop := 
  dist A centre < dist B centre

def foot_of_perpendicular (P : Point) (l : Line) : Point := 
  l.foot P

def bisects_angle (E F C D : Point) : Prop :=
  ∠EFC = ∠EFD

end EF_bisects_angle_CFD_l260_260712


namespace danny_lost_66_bottle_caps_l260_260381

theorem danny_lost_66_bottle_caps (initial_caps : ℕ) (current_caps : ℕ) (h_initial : initial_caps = 91) (h_current : current_caps = 25) : 
  initial_caps - current_caps = 66 :=
by
  /* Math operations and statement directly based on the condition provided, concluding with the required proof */
  sorry

end danny_lost_66_bottle_caps_l260_260381


namespace number_of_divisors_64n4_l260_260032

theorem number_of_divisors_64n4 
  (n : ℕ) 
  (h1 : (factors (120 * n^3)).length = 120) 
  (h2 : 120.nat_factors.prod * (n^3).nat_factors.prod = (120 * n^3)) :
  (factors (64 * n^4)).length = 675 := 
sorry

end number_of_divisors_64n4_l260_260032


namespace proportionality_problem_l260_260602

noncomputable def find_x (z w : ℝ) (k : ℝ) : ℝ :=
  k / (z^(3/2) * w^2)

theorem proportionality_problem :
  ∃ k : ℝ, 
    (find_x 16 2 k = 5) ∧
    (find_x 64 4 k = 5 / 32) :=
by
  sorry

end proportionality_problem_l260_260602


namespace triangle_construction_exists_l260_260000

noncomputable def median (A B C D : ℝ) : ℝ := sorry

theorem triangle_construction_exists
  (α : ℝ) (AB_AC_diff : ℝ) (s_a : ℝ) :
  ∃ (A B C : Point),
  ∠A = α ∧
  |dist A B - dist A C| = AB_AC_diff ∧
  median A B C = s_a :=
begin
  -- Proof that such a triangle exists and is unique would go here.
  sorry,
end

end triangle_construction_exists_l260_260000


namespace tallest_giraffe_difference_l260_260633

theorem tallest_giraffe_difference :
  let tallest_giraffe := 96
  let shortest_giraffe := 68
  in tallest_giraffe - shortest_giraffe = 28 :=
by
  sorry

end tallest_giraffe_difference_l260_260633


namespace sum_of_all_four_numbers_is_zero_l260_260513

theorem sum_of_all_four_numbers_is_zero 
  {a b c d : ℝ}
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b = c + d)
  (h_prod : a * c = b * d) 
  : a + b + c + d = 0 := 
by
  sorry

end sum_of_all_four_numbers_is_zero_l260_260513


namespace min_value_of_f_l260_260386

def max (a b : ℝ) : ℝ := if a ≥ b then a else b

def f (x : ℝ) : ℝ := max (-x - 1) (x^2 - 2*x - 3)

theorem min_value_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) → y ≥ -3 :=
begin
  sorry
end

end min_value_of_f_l260_260386


namespace tangent_30_degrees_l260_260814

theorem tangent_30_degrees (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) (hA : ∃ α : ℝ, α = 30 ∧ (y / x) = Real.tan (π / 6)) :
  y / x = Real.sqrt 3 / 3 :=
by
  sorry

end tangent_30_degrees_l260_260814


namespace circumradius_triangle_ABC_l260_260042

open EuclideanGeometry

noncomputable def radiusOfCircumcircleOfTriangleABC : Real :=
  let r := 2 * Real.sqrt 5
  let C := Point.mk 0 0
  let O := Point.mk r 0
  let A := Point.mk r (r)
  let B := Segment.mk O C ∩ Circle.mk O r
  let F := Perpendicular.mk B (Line.mk C O) ∩ Line.mk A C
  if BF = 2 then
    let circumradius := Circumcircle.radius (Triangle.mk A B C)
    circumradius
  else
    0

theorem circumradius_triangle_ABC :
  radiusOfCircumcircleOfTriangleABC = Real.sqrt 30 / 2 := by
  -- The proof will be done here.
  sorry

end circumradius_triangle_ABC_l260_260042


namespace harly_shelter_dog_count_l260_260842

noncomputable def dogs_remaining_at_end : ℕ := 
  let initial_dogs : ℕ := 100
  let adopted_out_dogs : ℕ := 60 * initial_dogs / 100
  let dogs_after_return : ℕ := initial_dogs - adopted_out_dogs + 8
  let harly_breedA_dogs : ℕ := 30 * dogs_after_return / 100
  let other_shelter_dogs : ℕ := 50
  let other_breedA_dogs : ℕ := 40 * other_shelter_dogs / 100
  let combined_breedA_dogs : ℕ := harly_breedA_dogs + other_breedA_dogs
  let adopted_breedA_dogs : ℕ := 70 * combined_breedA_dogs / 100
  let remaining_harly_breedA_dogs : ℕ := harly_breedA_dogs - adopted_breedA_dogs
  dogs_after_return - harly_breedA_dogs

theorem harly_shelter_dog_count : dogs_remaining_at_end = 34 := 
  by 
    let harly_dogs := 100
    let adopted_out := 60 * harly_dogs / 100
    let remaining_after_return := harly_dogs - adopted_out + 8
    let harly_breedA := 30 * remaining_after_return / 100
    let other_breedA := 20
    let combined_breedA := harly_breedA + other_breedA
    let adopted_breedA := 70 * combined_breedA / 100
    let remaining_harly_breedA := 0
    have total_remaining := remaining_after_return - harly_breedA
    guard_hypothesis total_remaining == 34

end harly_shelter_dog_count_l260_260842


namespace distance_from_point_to_line_l260_260772

def point : ℝ × ℝ × ℝ := (0, -1, 4)

def line (s : ℝ) : (ℝ × ℝ × ℝ) := (-3 + 4 * s, 2 + s, 5 - 3 * s)

def direction_vector : ℝ × ℝ × ℝ := (4, 1, -3)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

theorem distance_from_point_to_line :
  ∃ s : ℝ, distance (0, -1, 4) (line s) = 35 / 13 := by
sorry

end distance_from_point_to_line_l260_260772


namespace non_congruent_triangles_with_perimeter_12_l260_260125

theorem non_congruent_triangles_with_perimeter_12 :
  ∃ (S : finset (ℤ × ℤ × ℤ)), S.card = 2 ∧ ∀ (a b c : ℤ), (a, b, c) ∈ S →
  a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ c < a + b :=
sorry

end non_congruent_triangles_with_perimeter_12_l260_260125


namespace least_positive_three_digit_multiple_of_8_l260_260293

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l260_260293


namespace net_effect_is_correct_l260_260496

variables (P Q : ℝ)

def original_sale_value (P Q : ℝ) : ℝ := P * Q

def new_sale_value (P Q : ℝ) : ℝ := (0.82 * P) * (1.88 * Q)

def net_effect (P Q : ℝ) : ℝ := ((new_sale_value P Q) / (original_sale_value P Q) - 1) * 100

theorem net_effect_is_correct (P Q : ℝ) : net_effect P Q = 54.26 :=
by
  sorry

end net_effect_is_correct_l260_260496


namespace rest_of_customers_bought_20_l260_260722

/-
Let's define the number of melons sold by the stand, number of customers who bought one and three melons, and total number of melons bought by these customers.
-/

def total_melons_sold : ℕ := 46
def customers_bought_one : ℕ := 17
def customers_bought_three : ℕ := 3

def melons_bought_by_those_bought_one := customers_bought_one * 1
def melons_bought_by_those_bought_three := customers_bought_three * 3

def remaining_melons := total_melons_sold - (melons_bought_by_those_bought_one + melons_bought_by_those_bought_three)

-- Now we state the theorem that the number of melons bought by the rest of the customers is 20 
theorem rest_of_customers_bought_20 :
  remaining_melons = 20 :=
by
  -- Skip the proof with 'sorry'
  sorry

end rest_of_customers_bought_20_l260_260722


namespace sum_of_first_eight_terms_l260_260080

theorem sum_of_first_eight_terms (a : ℝ) (r : ℝ) 
  (h1 : r = 2) (h2 : a * (1 + 2 + 4 + 8) = 1) :
  a * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 17 :=
by
  -- sorry is used to skip the proof
  sorry

end sum_of_first_eight_terms_l260_260080


namespace count_even_numbers_l260_260971

theorem count_even_numbers : 
  let digits := {1, 2, 3, 4, 5} in 
  let is_valid_permutation (p : List ℕ) :=
    p.length = 5 ∧
    p.all (λ x, x ∈ digits) ∧
    p.nodup ∧
    (\(p : List ℕ) => p.getLast! ∈ {2, 4}) p ∧
    ∀ (i : ℕ), i < 4 → ¬(p.nthLe! i = 1 ∧ p.nthLe! (i + 1) = 5 ∨
                      p.nthLe! i = 5 ∧ p.nthLe! (i + 1) = 1) in
  ∃ (count : ℕ), count = 24 ∧ (count = (List.permutations [1, 2, 3, 4, 5])
    .countp is_valid_permutation) :=
begin
  let count := (List.permutations [1, 2, 3, 4, 5])
    .countp is_valid_permutation,
  use count,
  split,
  { sorry },       -- Prove count = 24
  { rfl },        -- count = actual count computation
end

end count_even_numbers_l260_260971


namespace norma_cards_lost_l260_260918

theorem norma_cards_lost (original_cards : ℕ) (current_cards : ℕ) (cards_lost : ℕ)
  (h1 : original_cards = 88) (h2 : current_cards = 18) :
  original_cards - current_cards = cards_lost →
  cards_lost = 70 := by
  sorry

end norma_cards_lost_l260_260918


namespace find_a2_plus_b2_l260_260851

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 :=
by
  sorry

end find_a2_plus_b2_l260_260851


namespace klay_to_draymond_ratio_l260_260516

-- Let us define the points earned by each player
def draymond_points : ℕ := 12
def curry_points : ℕ := 2 * draymond_points
def kelly_points : ℕ := 9
def durant_points : ℕ := 2 * kelly_points

-- Total points of the Golden State Team
def total_points_team : ℕ := 69

theorem klay_to_draymond_ratio :
  ∃ klay_points : ℕ,
    klay_points = total_points_team - (draymond_points + curry_points + kelly_points + durant_points) ∧
    klay_points / draymond_points = 1 / 2 :=
by
  sorry

end klay_to_draymond_ratio_l260_260516


namespace complex_number_sum_product_real_a_l260_260493

theorem complex_number_sum_product_real_a :
  ∀ (a b : ℝ), (-1 + a * Complex.i) + (b - Complex.i) ∈ ℝ ∧ (-1 + a * Complex.i) * (b - Complex.i) ∈ ℝ → a = 1 :=
by
  intros a b h
  have h1 : Complex.i = 0 by sorry
  exact sorry

end complex_number_sum_product_real_a_l260_260493


namespace area_of_WIN_sector_is_correct_l260_260699

-- Definitions of the given conditions
def radius : ℝ := 8
def probability_of_win : ℝ := 1 / 4

-- State the theorem for the area of the WIN sector
theorem area_of_WIN_sector_is_correct : 
  let total_area := Real.pi * radius^2
  let win_sector_area := probability_of_win * total_area
  win_sector_area = 16 * Real.pi := 
by
  sorry

end area_of_WIN_sector_is_correct_l260_260699


namespace gabby_l260_260046

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end gabby_l260_260046


namespace product_wavelengths_eq_n_cbrt_mn2_l260_260269

variable (m n : ℝ)

noncomputable def common_ratio (m n : ℝ) := (n / m)^(1/3)

noncomputable def wavelength_jiazhong (m n : ℝ) := (m^2 * n)^(1/3)
noncomputable def wavelength_nanlu (m n : ℝ) := (n^4 / m)^(1/3)

theorem product_wavelengths_eq_n_cbrt_mn2
  (h : n = m * (common_ratio m n)^3) :
  (wavelength_jiazhong m n) * (wavelength_nanlu m n) = n * (m * n^2)^(1/3) :=
by
  sorry

end product_wavelengths_eq_n_cbrt_mn2_l260_260269


namespace find_n_l260_260018

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end find_n_l260_260018


namespace num_odd_digits_base4_of_345_l260_260417

/-- The number of odd digits in the base-4 representation of 345₁₀ is 4. -/
theorem num_odd_digits_base4_of_345 : 
  let base4_repr := Nat.digits 4 345 in
  (base4_repr.filter (λ d, d % 2 = 1)).length = 4 := by
  sorry

end num_odd_digits_base4_of_345_l260_260417


namespace election_upper_bound_l260_260155

theorem election_upper_bound (p : ℕ) (h_pos : p > 0) :
  (∀ (S : Finset (Finset (Fin p))), (∀ a b ∈ S, a ≠ b → ¬ Disjoint a b) ∧ (∀ a b ∈ S, a = b → a = b) → S.card ≤ 2 ^ (p - 1)) :=
begin
  sorry
end

end election_upper_bound_l260_260155


namespace find_digits_l260_260389

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l260_260389


namespace child_playing_time_l260_260598

theorem child_playing_time (n : ℕ) (t : ℕ) (total_time : ℕ) (h1 : n = 6) (h2 : total_time = 90) (h3 : t = 2 * total_time) : t / n = 30 :=
by
  -- Total number of children is 6
  have h4 : t = 2 * total_time, from h3,
  -- Simplify to find the child-minutes
  have h5 : t = 180, by rw [h4, h2, mul_comm 2 90, mul_comm 90 2],
  -- Divide the total child-minutes by the number of children
  have h6 : 180 / 6 = 30, from nat.div_self,
  have h7 : t / n = 30, from h6,
  -- Hence, each child plays for 30 minutes
  exact h7

end child_playing_time_l260_260598


namespace probability_of_green_ball_l260_260380

def container_X := (5, 7)  -- (red balls, green balls)
def container_Y := (7, 5)  -- (red balls, green balls)
def container_Z := (7, 5)  -- (red balls, green balls)

def total_balls (container : ℕ × ℕ) : ℕ := container.1 + container.2

def probability_green (container : ℕ × ℕ) : ℚ := 
  (container.2 : ℚ) / total_balls container

noncomputable def probability_green_from_random_selection : ℚ :=
  (1 / 3) * probability_green container_X +
  (1 / 3) * probability_green container_Y +
  (1 / 3) * probability_green container_Z

theorem probability_of_green_ball :
  probability_green_from_random_selection = 17 / 36 :=
sorry

end probability_of_green_ball_l260_260380


namespace max_value_z_l260_260912

def max_z (x y : ℝ) : ℝ := x + 2 * y - (1 / x)

theorem max_value_z :
  ∃ (x y : ℝ), (x - y ≤ 0) ∧ (4 * x - y ≥ 0) ∧ (x + y ≤ 3) ∧ (max_z x y = 4) :=
by
  sorry

end max_value_z_l260_260912


namespace old_selling_price_l260_260674

theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) :
  C + 0.10 * C = 88 :=
by
  sorry

end old_selling_price_l260_260674


namespace find_x_l260_260424

-- Definition of logarithm in Lean
noncomputable def log (b a: ℝ) : ℝ := Real.log a / Real.log b

-- Problem statement in Lean
theorem find_x (x : ℝ) (h : log 64 4 = 1 / 3) : log x 8 = 1 / 3 → x = 512 :=
by sorry

end find_x_l260_260424


namespace problem_part_I_problem_part_II_l260_260455

variable (a b : ℕ → ℤ)

def arithmetic_seq := ∀ n, a n = 3 * n - 2
def sum_first_n_arithmetic_seq := ∀ (n : ℕ), (finset.range n).sum (λ i, a (i + 1)) = (3 * n * (n + 1) / 2) - (n / 2)

def geometric_seq := ∀ (n : ℕ), b n = (-2)^(n-1)
def alternating_sum_geometric_seq := ∀ (n : ℕ), (finset.range n).sum (λ i, (-1)^i * b (i + 1)) = 2^(2 * n) - 1

theorem problem_part_I (ha1 : a 1 = 1)
                      (hb1 : b 1 = 1)
                      (hb2 : 2 * b 2 + b 3 = 0)
                      (ha1a3 : a 1 + a 3 = 2 * b 3) :
  arithmetic_seq a ∧ sum_first_n_arithmetic_seq a :=
sorry

theorem problem_part_II (ha1 : a 1 = 1)
                        (hb1 : b 1 = 1)
                        (hb2 : 2 * b 2 + b 3 = 0)
                        (ha1a3 : a 1 + a 3 = 2 * b 3) :
  geometric_seq b ∧ alternating_sum_geometric_seq b :=
sorry

end problem_part_I_problem_part_II_l260_260455


namespace range_of_a_l260_260050

noncomputable def domain_f (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + a ≥ 0
noncomputable def range_g (a : ℝ) : Prop := ∀ x : ℝ, x ≤ 2 → 2^x - a ∈ Set.Ioi (0 : ℝ)

theorem range_of_a (a : ℝ) : (domain_f a ∨ range_g a) ∧ ¬(domain_f a ∧ range_g a) → (a ≥ 1 ∨ a ≤ 0) := by
  sorry

end range_of_a_l260_260050


namespace maximize_profit_correct_l260_260148

def maximize_profit (a b : ℝ) : ℝ :=
  -b / (2 * a)

theorem maximize_profit_correct : ∀ (p : ℝ → ℝ) (x : ℝ),
  (∀ x, p x = -25 * x ^ 2 + 7500 * x) →
  maximize_profit (-25) 7500 = 150 :=
by
  intros p x hp
  unfold maximize_profit
  rw hp
  -- leave as sorry here to skip the proof
  sorry

end maximize_profit_correct_l260_260148


namespace factorize_difference_of_squares_l260_260406

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 6 = (a + real.sqrt 6) * (a - real.sqrt 6) :=
by
  sorry

end factorize_difference_of_squares_l260_260406


namespace time_b_used_l260_260673

noncomputable def time_b_used_for_proof : ℚ :=
  let C : ℚ := 1
  let C_a : ℚ := 1 / 4 * C
  let t_a : ℚ := 15
  let p_a : ℚ := 1 / 3
  let p_b : ℚ := 2 / 3
  let ratio : ℚ := (C_a * t_a) / ((C - C_a) * (t_a * p_a / p_b))
  t_a * p_a / p_b

theorem time_b_used : time_b_used_for_proof = 10 / 3 := by
  sorry

end time_b_used_l260_260673


namespace reflect_H_across_x_axis_and_y_eq_x_minus_1_l260_260221

theorem reflect_H_across_x_axis_and_y_eq_x_minus_1 :
  let H := (5, 0)
  let H' := (H.1, -H.2)
  let H''_translated := (H'.1 - 1, H'.2 + 1)
  let H'' := (H''_translated.2, H''_translated.1)
  in (H''.1, H''.2 - 1) = (1, 4) :=
by 
  apply sorry

end reflect_H_across_x_axis_and_y_eq_x_minus_1_l260_260221


namespace sum_of_x_satisfying_property_l260_260549

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_on_pos (f : ℝ → ℝ) := ∀ x y, 0 < x → x < y → f x ≤ f y

theorem sum_of_x_satisfying_property (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : is_monotonic_on_pos f)
  (h_cont : continuous f) :
  ∑ x in {x | f x = f ((x+1)/(2*x+4))}.to_finset = -4 :=
begin
  sorry
end

end sum_of_x_satisfying_property_l260_260549


namespace exist_four_good_sequences_l260_260511

noncomputable def grid_sequence (n : ℕ) (n_pos : n > 1) : Prop :=
  ∃ (seq : fin (2 * n) → ℤ), 
    (∑ i, |seq i| ≠ 0) ∧ 
    (∀ i, |seq i| ≤ n) ∧
    (∀ i : fin n, ∑ j in finset.range (2 * n), ite (j % 2 = 0) (seq (fin.cast_add₀ i j)) (seq (fin.cast_add₀ i j)) = 0)

theorem exist_four_good_sequences (n : ℕ) (n_pos : n > 1) : ∃ (seqs : fin 4 → (fin (2 * n) → ℤ)), 
  (∀ k : fin 4, ∑ i, |(seqs k) i| ≠ 0) ∧ 
  (∀ k : fin 4, ∀ i, |(seqs k) i| ≤ n) ∧
  (∀ k : fin 4, ∀ i : fin n, ∑ j in finset.range (2 * n), ite (j % 2 = 0) ((seqs k) (fin.cast_add₀ i j)) ((seqs k) (fin.cast_add₀ i j)) = 0) ∧
  ∀ (i j : fin 4), i ≠ j → seqs i ≠ seqs j := sorry

end exist_four_good_sequences_l260_260511


namespace num_non_congruent_triangles_with_perimeter_12_l260_260121

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l260_260121


namespace digit_ends_with_l260_260392

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l260_260392


namespace adjacent_product_negative_l260_260163

noncomputable def a_seq : ℕ → ℚ
| 0 => 15
| (n+1) => (a_seq n) - (2 / 3)

theorem adjacent_product_negative :
  ∃ n : ℕ, a_seq 22 * a_seq 23 < 0 :=
by
  -- From the conditions, it is known that a_seq satisfies the recursive definition
  --
  -- We seek to prove that a_seq 22 * a_seq 23 < 0
  sorry

end adjacent_product_negative_l260_260163


namespace find_complex_z_l260_260811

-- Setting up the required complex numbers and condition
def complex_number_satisfies_condition (z : ℂ) : Prop :=
  (1 + 2 * complex.i) * complex.conj(z) = 4 + 3 * complex.i

-- The main statement proving that, given the condition, z is equal to 2 + i
theorem find_complex_z (z : ℂ) (h : complex_number_satisfies_condition z) : z = 2 + complex.i :=
  sorry

end find_complex_z_l260_260811


namespace parallelogram_side_length_l260_260346

theorem parallelogram_side_length (a a1 b b1 x y : ℝ) :
  (a + a1 < 2 * x ∧ b + b1 < 2 * y) →
  (∃ (P1 P2 P3 : Type) (inscribed1 : P2 ⊆ P1)
    (inscribed2 : P3 ⊆ P2) 
    (parallel_sides : ∀ (s1 s2 : ℝ), s1 - s2 = 0 → s1 = s2) 
    (side_len_P3_a side_len_P3_b side_len_P1_a side_len_P1_b : ℝ),
      side_len_P3_a = a + a1 ∧ 
      side_len_P3_b = b + b1 ∧ 
      side_len_P1_a = a + a1 + 2 * x ∧ 
      side_len_P1_b = b + b1 + 2 * y) → 
  (2 * x ≤ a + a1 ∨ 2 * y ≤ b + b1) :=
begin
  sorry
end

end parallelogram_side_length_l260_260346


namespace find_n_l260_260065

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (n : ℕ)

def isArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sumTo (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem find_n 
  (h_arith : isArithmeticSeq a)
  (h_a2 : a 2 = 2) 
  (h_S_diff : ∀ n, n > 3 → S n - S (n - 3) = 54)
  (h_Sn : S n = 100)
  : n = 10 := 
by
  sorry

end find_n_l260_260065


namespace remainder_is_83_l260_260688

-- Define the condition: the values for the division
def value1 : ℤ := 2021
def value2 : ℤ := 102

-- State the theorem: remainder when 2021 is divided by 102 is 83
theorem remainder_is_83 : value1 % value2 = 83 := by
  sorry

end remainder_is_83_l260_260688


namespace abs_x_minus_one_iff_x_in_interval_l260_260266

theorem abs_x_minus_one_iff_x_in_interval (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 :=
by
  sorry

end abs_x_minus_one_iff_x_in_interval_l260_260266


namespace polygon_sides_l260_260079

theorem polygon_sides (h : ∀ (θ : ℕ), θ = 108) : ∃ n : ℕ, n = 5 :=
by
  sorry

end polygon_sides_l260_260079


namespace trajectory_is_ellipse_l260_260623

/-- Define fixed points F1, F2, and a moving point M -/
def F1 : ℝ × ℝ := (0, 0)  -- Assume F1 is at the origin for simplicity
def F2 : ℝ × ℝ := (6, 0)  -- Place F2 on the x-axis 6 units apart from F1

/-- Define the distance function between two points -/
def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

/-- The condition that |F1F2| = 6 -/
def condition1 := dist F1 F2 = 6

/-- The condition that the sum of distances |MF1| + |MF2| = 8 -/
def condition2 (M : ℝ × ℝ) := dist M F1 + dist M F2 = 8

/-- The trajectory of M must be an ellipse, as per the given conditions -/
theorem trajectory_is_ellipse (M : ℝ × ℝ) : (condition1 ∧ condition2 M) → 
  exists a b c : ℝ, (∀ m, dist m F1 + dist m F2 = 8 ↔ 
    (m.1 - c) ^ 2 / a ^ 2 + m.2 ^ 2 / b ^ 2 = 1) := 
by
  sorry

end trajectory_is_ellipse_l260_260623


namespace certain_number_is_five_hundred_l260_260490

theorem certain_number_is_five_hundred (x : ℝ) (h : 0.60 * x = 0.50 * 600) : x = 500 := 
by sorry

end certain_number_is_five_hundred_l260_260490


namespace bhanu_house_rent_eq_seventy_l260_260679

-- Assuming Bhanu's income is a positive real number:
variable (income : Real) (Rs : Real)
-- Assuming Bhanu spends 30% of his income on petrol:
variable (percent_petrol : Real := 0.30)
-- Assuming Bhanu spends Rs. 300 on petrol:
variable (amount_petrol : Real := 300)
-- Assuming Bhanu spends 10% of the remaining income on house rent:
variable (percent_house_rent : Real := 0.10)

-- Definition of total income based on given conditions:
def total_income := amount_petrol / percent_petrol
-- Definition of remaining income after petrol expenditure:
def remaining_income := total_income - amount_petrol
-- Definition of house rent expenditure based on remaining income:
def house_rent := remaining_income * percent_house_rent

-- The goal is to prove that Bhanu's house rent expenditure is Rs. 70:
theorem bhanu_house_rent_eq_seventy : house_rent = 70 := by 
  sorry

end bhanu_house_rent_eq_seventy_l260_260679


namespace cake_shop_problem_l260_260858

theorem cake_shop_problem :
  ∃ (N n K : ℕ), (N - n * K = 6) ∧ (N = (n - 1) * 8 + 1) ∧ (N = 97) :=
by
  sorry

end cake_shop_problem_l260_260858


namespace f_neg_l260_260192

-- Define the function f and its properties
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else sorry

-- Define the property of f being an odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of f for non-negative x
axiom f_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

-- The theorem to be proven
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := by
  sorry

end f_neg_l260_260192


namespace intersection_of_A_and_B_l260_260839

def A := {1, 2, 3, 4}
def B := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} :=
by
  sorry

end intersection_of_A_and_B_l260_260839


namespace sufficient_not_necessary_l260_260906

theorem sufficient_not_necessary (x y : ℝ) : (x > |y|) → (x > y ∧ ¬ (x > y → x > |y|)) :=
by
  sorry

end sufficient_not_necessary_l260_260906


namespace horse_fertilizer_production_l260_260174

theorem horse_fertilizer_production:
  (horses acres gallons_per_acre acres_per_day days total_fertilizer_per_day total_fertilizer_per_days : ℕ)
  (Hhorses : horses = 80)
  (Hacres : acres = 20)
  (Hgallons_per_acre : gallons_per_acre = 400)
  (Hacres_per_day : acres_per_day = 4)
  (Hdays : days = 25)
  (Htotal_fertilizer_per_day : total_fertilizer_per_day = acres_per_day * gallons_per_acre)
  (Htotal_fertilizer_per_days : total_fertilizer_per_days = total_fertilizer_per_day * days)
  : (total_fertilizer_per_days / (horses * days) = 20) :=
by
  sorry

end horse_fertilizer_production_l260_260174


namespace largest_option_is_B_l260_260104

noncomputable def sqrt := Real.sqrt
noncomputable def sqrt_fourth := λ x : ℝ, x ^ (1 / 4)

def option_A := sqrt (sqrt_fourth (7 * 8))
def option_B := sqrt (sqrt_fourth (8 * sqrt_fourth 7))
def option_C := sqrt (sqrt_fourth (7 * sqrt_fourth 8))
def option_D := sqrt_fourth (7 * sqrt (8))
def option_E := sqrt_fourth (8 * sqrt (7))

theorem largest_option_is_B :
  max option_A (max option_B (max option_C (max option_D option_E))) = option_B :=
sorry

end largest_option_is_B_l260_260104


namespace two_pow_pos_not_exists_two_pow_le_zero_l260_260638

theorem two_pow_pos (x : ℝ) : 2 ^ x > 0 :=
sorry

theorem not_exists_two_pow_le_zero :
  ¬ ∃ x_0 : ℝ, 2 ^ x_0 ≤ 0 :=
by {
  intro h,
  cases h with x0 hx0,
  have h2 : 2 ^ x0 > 0 := two_pow_pos x0,
  linarith,
}

end two_pow_pos_not_exists_two_pow_le_zero_l260_260638


namespace arithmetic_sequence_terms_l260_260630

theorem arithmetic_sequence_terms
  (a : ℕ → ℝ)
  (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 20)
  (h2 : a (n-2) + a (n-1) + a n = 130)
  (h3 : (n * (a 1 + a n)) / 2 = 200) :
  n = 8 := 
sorry

end arithmetic_sequence_terms_l260_260630


namespace num_non_congruent_triangles_with_perimeter_12_l260_260120

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l260_260120


namespace coords_of_point_wrt_origin_l260_260954

-- Define the given point
def given_point : ℝ × ℝ := (1, -2)

-- Goal: Prove that the coordinates of the given point with respect to the origin are (1, -2)
theorem coords_of_point_wrt_origin (p : ℝ × ℝ) (hp : p = given_point) : p = (1, -2) :=
by
  rw hp
  rfl

end coords_of_point_wrt_origin_l260_260954


namespace value_equation_l260_260866

noncomputable def quarter_value := 25
noncomputable def dime_value := 10
noncomputable def half_dollar_value := 50

theorem value_equation (n : ℕ) :
  25 * quarter_value + 20 * dime_value = 15 * quarter_value + 10 * dime_value + n * half_dollar_value → 
  n = 7 :=
by
  sorry

end value_equation_l260_260866


namespace find_lambda_l260_260475

open LinearAlgebra

def vector_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_lambda (λ : ℝ) (h : vector_parallel (4, 2 * λ - 1) (3, -λ - 2)) : λ = -1 / 2 :=
by
  sorry

end find_lambda_l260_260475


namespace angle_ADE_is_135_l260_260867

noncomputable def ∆ABC := Type
variables (A B C D E : ∆ABC)
variables (DE BC BE CE AC AB AD : ℝ)
variables (triangle_ABC : ∆ABC)
variables (ADE : ℝ)

-- Given conditions
axiom condition1 : AC - AB = (sqrt 2) / 2 * BC
axiom condition2 : AD = AB
axiom condition3 : DE ⊥ BC
axiom condition4 : 3 * BE = CE

-- To prove: ∠ADE = 135°
theorem angle_ADE_is_135 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  angle ADE = 135 := 
sorry

end angle_ADE_is_135_l260_260867


namespace edge_length_in_mm_l260_260845

-- Definitions based on conditions
def cube_volume (a : ℝ) : ℝ := a^3

axiom volume_of_dice : cube_volume 2 = 8

-- Statement of the theorem to be proved
theorem edge_length_in_mm : ∃ (a : ℝ), cube_volume a = 8 ∧ a * 10 = 20 := sorry

end edge_length_in_mm_l260_260845


namespace ratio_of_area_to_breadth_l260_260245

theorem ratio_of_area_to_breadth (b l A : ℝ) (h₁ : b = 10) (h₂ : l - b = 10) (h₃ : A = l * b) : A / b = 20 := 
by
  sorry

end ratio_of_area_to_breadth_l260_260245


namespace part_a_solution_l260_260685

def P (n : ℕ) : ℕ := (n.digits 10).prod

def part_a_question (n : ℕ) : Prop := 1 ≤ n ∧ n < 1000 ∧ P(n) = 12

def part_a_answer : list ℕ := [26, 62, 34, 43, 126, 162, 216, 261, 612, 621, 134, 143, 314, 341, 413, 431, 223, 232, 322]

theorem part_a_solution : ∀ n : ℕ, part_a_question n ↔ n ∈ part_a_answer := by
  sorry

end part_a_solution_l260_260685


namespace tangent_line_at_1_2_l260_260810

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then exp (-x - 1) - x else exp (x - 1) + x

theorem tangent_line_at_1_2 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, x ≤ 0 → f x = exp (-x - 1) - x) →
  ∃ m b : ℝ, (∀ x : ℝ, x = 1 → f' x = m) ∧ (f 1 = 2) ∧ (∀ x : ℝ, y = m * x + b) ∧ y = 2 * x :=
by
  sorry

end tangent_line_at_1_2_l260_260810


namespace next_equalities_from_conditions_l260_260319

-- Definitions of the equality conditions
def eq1 : Prop := 3^2 + 4^2 = 5^2
def eq2 : Prop := 10^2 + 11^2 + 12^2 = 13^2 + 14^2
def eq3 : Prop := 21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2
def eq4 : Prop := 36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2

-- The next equalities we want to prove
def eq5 : Prop := 55^2 + 56^2 + 57^2 + 58^2 + 59^2 + 60^2 = 61^2 + 62^2 + 63^2 + 64^2 + 65^2
def eq6 : Prop := 78^2 + 79^2 + 80^2 + 81^2 + 82^2 + 83^2 + 84^2 = 85^2 + 86^2 + 87^2 + 88^2 + 89^2 + 90^2

theorem next_equalities_from_conditions : eq1 → eq2 → eq3 → eq4 → (eq5 ∧ eq6) :=
by
  sorry

end next_equalities_from_conditions_l260_260319


namespace foci_of_ellipse_l260_260607

def ellipse_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ (y = 12 ∨ y = -12))

theorem foci_of_ellipse :
  ∀ (x y : ℝ), (x^2)/25 + (y^2)/169 = 1 → ellipse_focus x y :=
by
  intros x y h
  sorry

end foci_of_ellipse_l260_260607


namespace number_of_drawing_methods_l260_260277

-- Define the main problem as a Lean statement
theorem number_of_drawing_methods : 
  let students := {A, B, C, D} in
  let cards := {A, B, C, D} in
  ∃ f : cards → students, (∀ s ∈ students, f s ≠ s) ∧ (function.bijective f) ∧ 
  (∃! n, number_of_drawing_methods = 9) := 
sorry

end number_of_drawing_methods_l260_260277


namespace poly_square_of_binomial_l260_260666

theorem poly_square_of_binomial (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := 
by 
  sorry

end poly_square_of_binomial_l260_260666


namespace area_KLMN_eq_ZM_ZK_plus_ZL_ZN_l260_260208

noncomputable theory
open_locale classical

variables (A B C D P Q R S Z l_A l_B l_C l_D E F G H K L M N : Type)
variables [convex_quadrilateral A B C D]
variables (angle_bisector_A : bisects_angle A)
variables (angle_bisector_B : bisects_angle B)
variables (angle_bisector_C : bisects_angle C)
variables (angle_bisector_D : bisects_angle D)
variables (intersect_P : intersection (angle_bisector_A) (angle_bisector_B) P)
variables (intersect_Q : intersection (angle_bisector_A) (angle_bisector_D) Q)
variables (intersect_R : intersection (angle_bisector_C) (angle_bisector_D) R)
variables (intersect_S : intersection (angle_bisector_C) (angle_bisector_B) S)
variables (distinct_points : is_distinct P Q R S)
variables (perpendicular_PR_QS : meets_perpendicularly PR QS Z)
variables (exterior_bisector_A : exterior_bisects_angle l_A A)
variables (exterior_bisector_B : exterior_bisects_angle l_B B)
variables (exterior_bisector_C : exterior_bisects_angle l_C C)
variables (exterior_bisector_D : exterior_bisects_angle l_D D)
variables (intersect_E : intersection l_A l_B E)
variables (intersect_F : intersection l_B l_C F)
variables (intersect_G : intersection l_C l_D G)
variables (intersect_H : intersection l_D l_A H)
variables (midpoint_K : midpoint FG K)
variables (midpoint_L : midpoint GH L)
variables (midpoint_M : midpoint HE M)
variables (midpoint_N : midpoint EF N)

theorem area_KLMN_eq_ZM_ZK_plus_ZL_ZN
  (angle_bisector_A_ab : bisects_angle A B)
  (angle_bisector_D_cd : bisects_angle C D)
  (rectangle_KLMN : rectangle K L M N) :
  area K L M N = ZM * ZK + ZL * ZN :=
sorry

end area_KLMN_eq_ZM_ZK_plus_ZL_ZN_l260_260208


namespace people_visited_both_l260_260151

theorem people_visited_both (total iceland norway neither both : ℕ) (h_total: total = 100) (h_iceland: iceland = 55) (h_norway: norway = 43) (h_neither: neither = 63)
  (h_both_def: both = iceland + norway - (total - neither)) :
  both = 61 :=
by 
  rw [h_total, h_iceland, h_norway, h_neither] at h_both_def
  simp at h_both_def
  exact h_both_def

end people_visited_both_l260_260151


namespace general_term_l260_260838

noncomputable def a : ℕ → ℝ
| 0 := 1
| 1 := 1
| 2 := 1
| (n+3) := (a (n+1) * a (n+2) + 1) / a n

theorem general_term (n : ℕ) :
  (∃ k : ℕ, n = 2 * k - 1 ∧
    a n = (9 - 5 * real.sqrt 3) / 6 * (2 + real.sqrt 3) ^ k +
          (9 + 5 * real.sqrt 3) / 6 * (2 - real.sqrt 3) ^ k) ∨
  (∃ k : ℕ, n = 2 * k ∧
    a n = (2 - real.sqrt 3) / 2 * (2 + real.sqrt 3) ^ k +
          (2 + real.sqrt 3) / 2 * (2 - real.sqrt 3) ^ k) := sorry

end general_term_l260_260838


namespace average_cookies_in_package_l260_260753

noncomputable def averageCookiesPerPackage :
  (packagesCounted : ℕ) → 
  (cookiesInCountedPackage : Fin packagesCounted → ℕ) → 
  (totalPackages : ℕ) → 
  (uncountedPackages : ℕ) → 
  Real
| 8, cookiesInCountedPackage, 10, 2 =>
    let totalCounted := cookiesInCountedPackage 0 + cookiesInCountedPackage 1 + cookiesInCountedPackage 2 + cookiesInCountedPackage 3 + cookiesInCountedPackage 4 + cookiesInCountedPackage 5 + cookiesInCountedPackage 6 + cookiesInCountedPackage 7
    let totalMissing := (totalCounted / 8) * 2
    let totalCookies := totalCounted + totalMissing
    totalCookies / 10

theorem average_cookies_in_package :
  averageCookiesPerPackage 8 (λ i, match i.1 with
                                  | 0 => 9
                                  | 1 => 11
                                  | 2 => 12
                                  | 3 => 14
                                  | 4 => 16
                                  | 5 => 17
                                  | 6 => 18
                                  | 7 => 21
                                  | _ => 0) 10 2 = 14.75 :=
by
  sorry

end average_cookies_in_package_l260_260753


namespace max_volume_of_rotated_triangle_l260_260628

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def volume (a b c : ℝ) : ℝ := 
  let s := semi_perimeter a b c in
  (4 * Real.pi / (3 * min a (min b c))) * s * (s - a) * (s - b) * (s - c)

-- Problem statement in Lean 4
theorem max_volume_of_rotated_triangle (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let s := semi_perimeter a b c in 
  volume a b c = (4 * Real.pi / (3 * min a (min b c))) * s * (s - a) * (s - b) * (s - c) :=
by
  sorry

end max_volume_of_rotated_triangle_l260_260628


namespace problem_statement_l260_260450

variables (a b c : Type) [Line a] [Line b] [Line c]
variables (α β : Type) [Plane α] [Plane β]

theorem problem_statement (ha_perp_α : a ⊥ α) (hα_par_β : α ∥ β) : a ⊥ β :=
sorry

end problem_statement_l260_260450


namespace product_of_first_two_terms_arithmetic_sequence_l260_260253

noncomputable def arithmetic_sequence_product (a_1 d a_5 : ℝ) := 
  let a_2 := a_1 + d in 
  a_1 * a_2

theorem product_of_first_two_terms_arithmetic_sequence :
  ∀ (a_1 d a_5 : ℝ), (a_5 = a_1 + 4 * d) ∧ (a_5 = 15) ∧ (d = 2) →
  arithmetic_sequence_product a_1 d a_5 = 63 := by
  intros a_1 d a_5 h
  sorry

end product_of_first_two_terms_arithmetic_sequence_l260_260253


namespace no_integer_b_satisfies_conditions_l260_260250

theorem no_integer_b_satisfies_conditions :
  ¬ ∃ b : ℕ, b^6 ≤ 196 ∧ 196 < b^7 :=
by
  sorry

end no_integer_b_satisfies_conditions_l260_260250


namespace non_congruent_triangles_perimeter_12_l260_260114

theorem non_congruent_triangles_perimeter_12 :
  ∃ S : finset (ℕ × ℕ × ℕ), S.card = 5 ∧ ∀ (abc ∈ S), 
  let (a, b, c) := abc in 
    a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    ∀ (abc' ∈ S), abc' ≠ abc → abc ≠ (λ t, (t.2.2, t.2.1, t.1)) abc' :=
by sorry

end non_congruent_triangles_perimeter_12_l260_260114


namespace discount_percentage_l260_260705

theorem discount_percentage 
  (C : ℝ) (S : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : C = 48)
  (h2 : 0.60 * S = C)
  (h3 : P = 16)
  (h4 : P = S - SP)
  (h5 : SP = 80 - 16)
  (h6 : S = 80) :
  (S - SP) / S * 100 = 20 := by
sorry

end discount_percentage_l260_260705


namespace abc_proof_l260_260486

noncomputable def abc_value (a b c : ℝ) : ℝ :=
  a * b * c

theorem abc_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b = 24 * (3 ^ (1 / 3)))
  (h5 : a * c = 40 * (3 ^ (1 / 3)))
  (h6 : b * c = 16 * (3 ^ (1 / 3))) : 
  abc_value a b c = 96 * (15 ^ (1 / 2)) :=
sorry

end abc_proof_l260_260486


namespace non_congruent_triangles_with_perimeter_12_l260_260122

theorem non_congruent_triangles_with_perimeter_12 :
  ∃ (S : finset (ℤ × ℤ × ℤ)), S.card = 2 ∧ ∀ (a b c : ℤ), (a, b, c) ∈ S →
  a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ c < a + b :=
sorry

end non_congruent_triangles_with_perimeter_12_l260_260122


namespace angle_ATK_eq_angle_BTK_l260_260900

noncomputable theory

-- Define the circles and the points
variables {Γ ω : Type*} [circle Γ T_in_Γ] [circle ω T_in_ω]
          {A B K T : Type*} [point A_on_Γ] [point B_on_Γ] [point K_on_ω] [point T_on_Γ] [point T_on_ω]

-- Homothety between the circles, centered at T
variable (h : homothety_cent(Γ, ω, T))

-- Tangent condition
variable (AB_tangent_to_ω_at_K : tangent_to K ω)

-- Being tangent to ω at point K
variable (tangent_at_K : tangent(ω, AB))

-- The goal
theorem angle_ATK_eq_angle_BTK :
  ∠ ATK = ∠ BTK :=
by
  sorry

end angle_ATK_eq_angle_BTK_l260_260900


namespace least_positive_three_digit_multiple_of_8_l260_260298

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l260_260298


namespace marsha_first_package_miles_l260_260210

noncomputable def total_distance (x : ℝ) : ℝ := x + 28 + 14

noncomputable def earnings (x : ℝ) : ℝ := total_distance x * 2

theorem marsha_first_package_miles : ∃ x : ℝ, earnings x = 104 ∧ x = 10 :=
by
  use 10
  sorry

end marsha_first_package_miles_l260_260210


namespace alice_weekly_walk_distance_l260_260009

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end alice_weekly_walk_distance_l260_260009


namespace prudence_sleep_in_4_weeks_l260_260041

theorem prudence_sleep_in_4_weeks :
  let hours_per_night_from_sun_to_thu := 6
      nights_from_sun_to_thu := 5
      hours_per_night_fri_and_sat := 9
      nights_fri_and_sat := 2
      nap_hours_per_day_on_sat_and_sun := 1
      nap_days_on_sat_and_sun := 2
      weeks := 4
  in
  (nights_from_sun_to_thu * hours_per_night_from_sun_to_thu +
   nights_fri_and_sat * hours_per_night_fri_and_sat +
   nap_days_on_sat_and_sun * nap_hours_per_day_on_sat_and_sun) * weeks = 200 :=
by
  sorry

end prudence_sleep_in_4_weeks_l260_260041


namespace no_infinite_set_exists_l260_260402

variable {S : Set ℕ} -- We assume S is a set of natural numbers

def satisfies_divisibility_condition (a b : ℕ) : Prop :=
  (a^2 + b^2 - a * b) ∣ (a * b)^2

theorem no_infinite_set_exists (h1 : Infinite S)
  (h2 : ∀ (a b : ℕ), a ∈ S → b ∈ S → satisfies_divisibility_condition a b) : false :=
  sorry

end no_infinite_set_exists_l260_260402


namespace sea_creatures_lost_l260_260478

theorem sea_creatures_lost (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34) 
  (h2 : seashells = 21) 
  (h3 : snails = 29) 
  (h4 : items_left = 59) : 
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l260_260478


namespace least_months_for_tripling_debt_l260_260530

theorem least_months_for_tripling_debt (P : ℝ) (r : ℝ) (t : ℕ) : P = 1500 → r = 0.06 → (3 * P < P * (1 + r) ^ t) → t ≥ 20 :=
by
  intros hP hr hI
  rw [hP, hr] at hI
  norm_num at hI
  sorry

end least_months_for_tripling_debt_l260_260530


namespace no_eulerian_path_possible_l260_260683

-- Define the notion of a Graph
structure Graph (V : Type) :=
  (edges : V → V → Prop)

-- Define Euler's theorem conditions
def has_eulerian_path (G : Graph ℕ) :=
  ∃ (u v : ℕ), 
    u ≠ v ∧ 
    (G.edges u) % 2 = 1 ∧ 
    (G.edges v) % 2 = 1 ∧
    ∀ x, x ≠ u ∧ x ≠ v → (G.edges x) % 2 = 0

-- Problem statement to be proved
theorem no_eulerian_path_possible : 
  ¬ (∃ G : Graph ℕ, 
      (∀ v : ℕ, (G.edges v) = 3) ∧ 
      has_eulerian_path G) :=
sorry

end no_eulerian_path_possible_l260_260683


namespace range_of_a_l260_260069

variable {a : ℝ}

def proposition_p (a : ℝ) : Prop := ∀ x ≥ 1, deriv (λ x, x^2 - 3*a*x + 4) x > 0
def proposition_q (a : ℝ) : Prop := ∀ x > 0, deriv (λ x, (2*a - 1)^x) x < 0

theorem range_of_a (h : ¬ (proposition_p a ∧ proposition_q a)) :
  a ≤ 1/2 ∨ a > 2/3 :=
sorry

end range_of_a_l260_260069


namespace sin_double_angle_l260_260785

theorem sin_double_angle (theta : ℝ) 
  (h : Real.sin (theta + Real.pi / 4) = 2 / 5) :
  Real.sin (2 * theta) = -17 / 25 := by
  sorry

end sin_double_angle_l260_260785


namespace digit_after_decimal_l260_260198

theorem digit_after_decimal (n : ℕ) : 
  (Nat.floor (10 * (Real.sqrt (n^2 + n) - Nat.floor (Real.sqrt (n^2 + n))))) = 4 :=
by
  sorry

end digit_after_decimal_l260_260198


namespace find_max_min_range_of_a_l260_260094

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 / 8 - log x

theorem find_max_min :
  let min_value := f 2
      max_value := f 1 in
  (min_value = 1 / 2 - log 2) ∧ (max_value = 1 / 8) ∧
  (∀ x ∈ set.Icc 1 3, f x ≤ max_value ∧ f x ≥ min_value) :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 3, ∀ t ∈ set.Icc 0 2, f x < 4 - a * t) →
  a < 31 / 16 :=
by
  sorry

end find_max_min_range_of_a_l260_260094


namespace intersection_A_B_l260_260474

def set_A : set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }
def set_B : set (ℝ × ℝ) := { p | 2 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 2 }

theorem intersection_A_B : set_A ∩ set_B = {(2, 1)} := 
by sorry

end intersection_A_B_l260_260474


namespace csc_neg_420_eq_neg_2sqrt3_div_3_l260_260407

theorem csc_neg_420_eq_neg_2sqrt3_div_3 :
  ∀ (θ : ℝ), (csc θ = 1 / sin θ) → (sin (θ + 360) = sin θ) → (sin (-θ) = -sin θ) → sin 60 = sqrt 3 / 2 → csc -420 = -2 * sqrt 3 / 3 :=
by
  intros θ h1 h2 h3 h4
  -- rest of the proof steps will go here.
  sorry

end csc_neg_420_eq_neg_2sqrt3_div_3_l260_260407


namespace max_value_expr_l260_260426

theorem max_value_expr (x : ℝ) : 
  ( x ^ 6 / (x ^ 12 + 3 * x ^ 8 - 6 * x ^ 6 + 12 * x ^ 4 + 36) <= 1/18 ) :=
by
  sorry

end max_value_expr_l260_260426


namespace value_of_abcg_defh_l260_260315

theorem value_of_abcg_defh
  (a b c d e f g h: ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4) :
  abcg / defh = 5 / 48 :=
by
  sorry

end value_of_abcg_defh_l260_260315


namespace non_congruent_triangles_with_perimeter_12_l260_260123

theorem non_congruent_triangles_with_perimeter_12 :
  ∃ (S : finset (ℤ × ℤ × ℤ)), S.card = 2 ∧ ∀ (a b c : ℤ), (a, b, c) ∈ S →
  a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ c < a + b :=
sorry

end non_congruent_triangles_with_perimeter_12_l260_260123


namespace complex_quadrant_l260_260884

theorem complex_quadrant : 
  ∀ z : ℂ, z = (4 + 3 * complex.i) / (1 + complex.i) → 
  (z.re > 0 ∧ z.im < 0) :=
by
  assume z
  assume h : z = (4 + 3 * complex.i) / (1 + complex.i)
  sorry

end complex_quadrant_l260_260884


namespace f_neg_a_eq_2a2_minus_M_l260_260827

theorem f_neg_a_eq_2a2_minus_M (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + log (x + sqrt (x^2 + 1))) (a M : ℝ) (ha : f a = M) :
  f (-a) = 2 * a^2 - M :=
by
  sorry

end f_neg_a_eq_2a2_minus_M_l260_260827


namespace sum_of_digits_0_to_999_l260_260661

theorem sum_of_digits_0_to_999 : 
  (∑ n in Finset.range 1000, (n / 100) + ((n / 10) % 10) + (n % 10) = 13500) := 
  sorry

end sum_of_digits_0_to_999_l260_260661


namespace count_integers_sum_of_consecutive_odd_l260_260479

theorem count_integers_sum_of_consecutive_odd (N : ℕ) :
  (∀ N < 1000, ∃ (f : ℕ → ℕ), 
     (∀ j : ℕ, (∃ (i : ℕ), N = j * (2 * i + j)) → 
      ((f j = 1) ∨ 
       (f (N / j) = 1) ∧ 
       (N / j).mod 2 = j.mod 2)) ∧ 
      ∑ j : ℕ, if (∃ (i : ℕ), N = j * (2 * i + j)) 
                then (1 : ℕ) 
                else 0) = 5) → 
       N < 1000 → (14 : ℕ) :=
by sorry

end count_integers_sum_of_consecutive_odd_l260_260479


namespace range_of_m_l260_260097

noncomputable def f (x m : ℝ) : ℝ := -x^2 + 6*x + m
noncomputable def g (x : ℝ) : ℝ :=  2 * Real.sin (2 * x + Real.pi / 3)

theorem range_of_m :
  (∀ x0 ∈ Set.Icc (0 : ℝ) (Real.pi / 4), ∃ x1 x2 ∈ Set.Icc (-1 : ℝ) 3, f x1 m ≤ g x0 ∧ g x0 ≤ f x2) →
  (-7 ≤ m ∧ m ≤ 8) :=
by
  sorry

end range_of_m_l260_260097


namespace sqrt_sum_ineq_l260_260436

open Real

theorem sqrt_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 1) :
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) + a + b + c > 3 := by
  sorry

end sqrt_sum_ineq_l260_260436


namespace vasya_has_greater_expected_area_l260_260740

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l260_260740


namespace range_of_g_interval_l260_260262

noncomputable def g (x : ℝ) := 2 / (2 + 4 * x^2)

theorem range_of_g_interval (a b : ℝ) (h : set.Ioo a 1 = set.range g) : a + b = 1 := by
  have h0 : a = 0 := by
    sorry
  have h1 : b = 1 := by
    sorry
  rw [h0, h1]
  exact add_zero 1

end range_of_g_interval_l260_260262


namespace domain_of_sqrt_fraction_is_l260_260614

def domain_of_sqrt_fraction (y : ℝ → ℝ) : Set ℝ :=
  { x | y x = sqrt ((x - 1) / (x + 2)) }

theorem domain_of_sqrt_fraction_is :
  domain_of_sqrt_fraction (λ x, sqrt ((x - 1) / (x + 2))) = { x | x < -2 ∨ x ≥ 1 } :=
by {
  -- Constraints: 
  -- 1. (x - 1) / (x + 2) >= 0
  -- 2. x + 2 ≠ 0
  sorry
}

end domain_of_sqrt_fraction_is_l260_260614


namespace area_difference_of_ABC_l260_260580

-- Define the geometric entities and conditions
variables (A B C O : Point)
variables (OB : Real) (OA : Real) (OC : Real)
hypothesis hOB : OB = 3
hypothesis hOA : OA = 4
hypothesis hOC : OC = 5

-- The main proof goal
theorem area_difference_of_ABC
    (equilateral_triangle : is_equilateral_triangle A B C)
    (O_inside_ABC : is_inside O (triangle A B C)) :
    area_of_triangle A O C - area_of_triangle B O C = 7 * Real.sqrt 3 / 4 :=
sorry

end area_difference_of_ABC_l260_260580


namespace hotel_charge_decrease_l260_260947

theorem hotel_charge_decrease 
  (G R P : ℝ)
  (h1 : R = 1.60 * G)
  (h2 : P = 0.50 * R) :
  (G - P) / G * 100 = 20 := by
sorry

end hotel_charge_decrease_l260_260947


namespace honor_students_count_l260_260981

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l260_260981


namespace weight_equivalence_l260_260517

variable (Δ ◯ ▢ : ℝ)

theorem weight_equivalence
  (h1 : 5 * Δ = 3 * ◯)
  (h2 : ◯ = Δ + 2 * ▢) :
  Δ + ◯ = 3 * ▢ :=
by
  sorry

end weight_equivalence_l260_260517


namespace factor_81_minus_4y4_l260_260768

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by 
    sorry

end factor_81_minus_4y4_l260_260768


namespace find_value_of_2a_minus_b_l260_260940

noncomputable def f (a b : ℚ) (x : ℚ) : ℚ := a * x + b
noncomputable def g (x : ℚ) : ℚ := -4 * x + 6
noncomputable def h (a b : ℚ) (x : ℚ) : ℚ := f a b (g x)
noncomputable def h_inv (x : ℚ) : ℚ := x + 9

theorem find_value_of_2a_minus_b (a b : ℚ) (ha : ∀ x : ℚ, h a b x = x - 9) : 2 * a - b = 7 := by
  have h_def : h a b = λ x, -4 * a * x + 6 * a + b := by
    sorry
    
  have ha_eq : (λ x, -4 * a * x + 6 * a + b) = (λ x, x - 9) := by
    ext x
    exact ha x
    
  -- Solve the system of equations implied by ha_eq
  have eq1 : 6 * a + b = -9 := by
    sorry
    
  have eq2 : 2 * a + b = -8 := by
    sorry
    
  -- Calculate a and b
  have a_val : a = -1/4 := by
    sorry
    
  have b_val : b = -15/2 := by
    sorry
    
  -- Find value of 2a - b
  have result : 2 * a - b = 7 := by
    sorry
    
  exact result

end find_value_of_2a_minus_b_l260_260940


namespace count_integers_with_distinct_digits_in_order_l260_260108

theorem count_integers_with_distinct_digits_in_order :
  ∃ n : ℕ, 
    (n = 40) ∧ 
    ∀ x : ℕ, 
      (3100 ≤ x) ∧ 
      (x < 3500) ∧ 
      (distinct_digits x) ∧ 
      (increasing_digits x)
  := sorry

end count_integers_with_distinct_digits_in_order_l260_260108


namespace real_number_unique_l260_260502

variable (a x : ℝ)

theorem real_number_unique (h1 : (a + 3) * (a + 3) = x)
  (h2 : (2 * a - 9) * (2 * a - 9) = x) : x = 25 := by
  sorry

end real_number_unique_l260_260502


namespace inequality_proof_l260_260898

noncomputable def a (x1 x2 x3 x4 x5 : ℝ) := x1 + x2 + x3 + x4 + x5
noncomputable def b (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 + x3 * x4 + x3 * x5 + x4 * x5
noncomputable def c (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 + x1 * x2 * x4 + x1 * x2 * x5 + x1 * x3 * x4 + x1 * x3 * x5 + x1 * x4 * x5 + x2 * x3 * x4 + x2 * x3 * x5 + x2 * x4 * x5 + x3 * x4 * x5
noncomputable def d (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 * x4 + x1 * x2 * x3 * x5 + x1 * x2 * x4 * x5 + x1 * x3 * x4 * x5 + x2 * x3 * x4 * x5

theorem inequality_proof (x1 x2 x3 x4 x5 : ℝ) (hx1x2x3x4x5 : x1 * x2 * x3 * x4 * x5 = 1) :
  (1 / a x1 x2 x3 x4 x5) + (1 / b x1 x2 x3 x4 x5) + (1 / c x1 x2 x3 x4 x5) + (1 / d x1 x2 x3 x4 x5) ≤ 3 / 5 := 
sorry

end inequality_proof_l260_260898


namespace sin_C_perimeter_obtuse_C_l260_260169

variable {α : Type*} [LinearOrderedRing α] [RealField α] [PlaneGeometry α]

-- Define the conditions of the problem
def cos_two_b : α := -1/2
def side_c : α := 8
def side_b : α := 7

-- Define the questions to be proven
-- Part Ⅰ: Prove \sin C
theorem sin_C (a b c : α) (B : α) (h_cos2b : cos_two_b = -1/2)
  (h_c : c = 8) (h_b : b = 7) : 
  sin (angleBToC B) = 4 * sqrt 3 / 7 := sorry

-- Part Ⅱ: If angle C is obtuse, prove perimeter
theorem perimeter_obtuse_C (a b c : α) (C : α) (h_cosC : cos C = -1/7)
  (h_c : c = 8) (h_b : b = 7) : 
  a + b + c = 18 := sorry

end sin_C_perimeter_obtuse_C_l260_260169


namespace projection_of_a_onto_b_l260_260088

noncomputable def vector_projection (a b : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
  let norm (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)
  let cos_theta := (dot_product a b) / (norm a * norm b)

  cos_theta * norm a

theorem projection_of_a_onto_b 
  (a b : ℝ × ℝ × ℝ)
  (ha : (norm a) = 1)
  (hb : (norm b) = 1)
  (hab : dot_product (a + b) b = 3 / 2) :
  vector_projection a b = 1 / 2 :=
by
  sorry

end projection_of_a_onto_b_l260_260088


namespace magicalNumbers_l260_260335

/-
Definition of a Canadian function
-/
def isCanadianFunction (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → Nat.gcd (f (f x)) (f (x + y)) = Nat.gcd x y

/-
Define a magical number as a number m such that f(m) = m for all Canadian functions f.
-/
def isMagical (m : ℕ) : Prop :=
  ∀ f : ℕ → ℕ, isCanadianFunction f → f m = m

/-
Main theorem: all magical numbers are exactly the positive integers which are not prime powers
-/
theorem magicalNumbers :
  ∀ m : ℕ, m > 0 → (isMagical m ↔ ¬∃ p k : ℕ, Nat.prime p ∧ k > 0 ∧ m = p^k) := 
sorry

end magicalNumbers_l260_260335


namespace domain_of_expression_l260_260014

theorem domain_of_expression (x : ℝ) :
  (1 ≤ x ∧ x < 6) ↔ (∃ y : ℝ, y = (x-1) ∧ y = (6-x) ∧ 0 ≤ y) :=
sorry

end domain_of_expression_l260_260014


namespace closest_integer_proof_l260_260491

noncomputable def closest_integer (x y : ℝ) : ℤ := Int.floor (x - y + 1 + 0.5)

theorem closest_integer_proof (x y : ℝ) (h1 : |x| + y = 5) (h2 : |x| * y + x^3 - 2 = 0) : 
  closest_integer x y = -3 := 
by 
suffices : x - y + 1 ≈ -3
-- Additional logical steps would go here to show the ≈ part
sorry

end closest_integer_proof_l260_260491


namespace least_positive_three_digit_multiple_of_8_l260_260294

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l260_260294


namespace S_is_positive_integers_l260_260196

noncomputable def S : set ℝ := sorry 

theorem S_is_positive_integers:
  (1 ∈ S) ∧ (∀ x y : ℝ, x ∈ S → y ∈ S → x + y ∈ S ∧ x * y ∈ S) ∧ 
  (∃ P ⊆ S, ∀ s ∈ (S \ {1}), ∃! t : multiset P, (t.prod = s)) → 
  S = {n | n ∈ ℕ ∧ n > 0} :=
sorry

end S_is_positive_integers_l260_260196


namespace texts_sent_on_Tuesday_l260_260919

theorem texts_sent_on_Tuesday (total_texts monday_texts : Nat) (texts_each_monday : Nat)
  (h_monday : texts_each_monday = 5)
  (h_total : total_texts = 40)
  (h_monday_total : monday_texts = 2 * texts_each_monday) :
  total_texts - monday_texts = 30 := by
  sorry

end texts_sent_on_Tuesday_l260_260919


namespace prove_cost_l260_260629

-- Define Variables
variables (p n e : ℝ)

-- Define Conditions
def condition1 : 10 * p + 12 * n + 6 * e = 5.50 := sorry
def condition2 : 6 * p + 4 * n + 3 * e = 2.40 := sorry

-- Define the expression we want to prove
def cost : ℝ := 20 * p + 15 * n + 9 * e

-- The statement we need to prove
theorem prove_cost : condition1 → condition2 → cost = 8.95 :=
begin
  sorry
end

end prove_cost_l260_260629


namespace problem1_minimum_value_of_g_problem2_slope_condition_l260_260826

open Real

noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (x : ℝ) (a : ℝ) := log x + a * x ^ 2 - 3 * x

theorem problem1_minimum_value_of_g :
  ∃ a : ℝ, g 1 1 = -2 ∧ (∀ x > 0, ∃ c : ℝ, g' c = 0) :=
sorry

theorem problem2_slope_condition (k x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  (∃ k : ℝ, ∀ x₁ x₂ : ℝ, x₁ < x₂ → log x₂ - log x₁ = k * (x₂ - x₁)) → 
  1/x₂ < k ∧ k < 1/x₁ :=
sorry

end problem1_minimum_value_of_g_problem2_slope_condition_l260_260826


namespace non_congruent_triangles_perimeter_12_l260_260117

theorem non_congruent_triangles_perimeter_12 :
  ∃ S : finset (ℕ × ℕ × ℕ), S.card = 5 ∧ ∀ (abc ∈ S), 
  let (a, b, c) := abc in 
    a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    ∀ (abc' ∈ S), abc' ≠ abc → abc ≠ (λ t, (t.2.2, t.2.1, t.1)) abc' :=
by sorry

end non_congruent_triangles_perimeter_12_l260_260117


namespace sequence_general_term_l260_260062

theorem sequence_general_term (c : ℕ) :
  ∃ x : ℕ → ℕ, 
    (x 1 = c) ∧ 
    (∀ n, n ≥ 2 → x n = x (n-1) + (⌊(2 * x (n-1) - (n + 2)) / n⌋ : ℕ) + 1) ∧
    ((c % 3 = 0 → ∀ n, x n = ((c - 1) / 6) * (n + 1) * (n + 2) + 1) ∧
     (c % 3 = 1 → ∀ n, x n = ((c - 2) / 6) * (n + 1) * (n + 2) + n + 1) ∧
     (c % 3 = 2 → ∀ n, x n = ((c - 3) / 6) * (n + 1) * (n + 2) + (⌊(n + 2) ^ 2 / 4⌋ : ℕ) + 1)) :=
by {
  sorry
}

end sequence_general_term_l260_260062


namespace honor_students_count_l260_260989

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l260_260989


namespace vasya_expected_area_greater_l260_260750

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l260_260750


namespace natural_numbers_product_and_sum_l260_260172

theorem natural_numbers_product_and_sum (
    n : ℕ → ℕ
) (hprod : (∀ i : fin 6, i ≠ 5 → n i = [2, 2, 2, 3, 19].nth i))
  (hsum : (∑ i, n i^2) = 456) : (∏ i, n i) = 456 :=
by
  sorry

end natural_numbers_product_and_sum_l260_260172


namespace complex_inequality_l260_260791

theorem complex_inequality (m : ℝ) (i : ℂ) (h : i * i = -1) (h_imag_unit : i ≠ 0) :
  (1 - 2 * i) / (m - i) > 0 → m = 1 / 2 :=
by
  sorry

end complex_inequality_l260_260791


namespace find_value_added_l260_260857

theorem find_value_added :
  ∀ (n x : ℤ), (2 * n + x = 8 * n - 4) → (n = 4) → (x = 20) :=
by
  intros n x h1 h2
  sorry

end find_value_added_l260_260857


namespace cartons_in_a_case_l260_260695

-- Definitions based on problem conditions
def numberOfBoxesInCarton (c : ℕ) (b : ℕ) : ℕ := c * b * 300
def paperClipsInTwoCases (c : ℕ) (b : ℕ) : ℕ := 2 * numberOfBoxesInCarton c b

-- Condition from problem statement: paperClipsInTwoCases c b = 600
theorem cartons_in_a_case 
  (c b : ℕ) 
  (h1 : paperClipsInTwoCases c b = 600) 
  (h2 : b ≥ 1) : 
  c = 1 := 
by
  -- Proof will be provided here
  sorry

end cartons_in_a_case_l260_260695


namespace marbles_lost_correct_l260_260531

-- Define the initial number of marbles
def initial_marbles : ℕ := 16

-- Define the current number of marbles
def current_marbles : ℕ := 9

-- Define the number of marbles lost
def marbles_lost (initial current : ℕ) : ℕ := initial - current

-- State the proof problem: Given the conditions, prove the number of marbles lost is 7
theorem marbles_lost_correct : marbles_lost initial_marbles current_marbles = 7 := by
  sorry

end marbles_lost_correct_l260_260531


namespace roots_of_equation_l260_260976

theorem roots_of_equation (x : ℝ) : (x - 3) ^ 2 = 4 ↔ (x = 5 ∨ x = 1) := by
  sorry

end roots_of_equation_l260_260976


namespace relationship_PX_PY_l260_260915

open Real

-- Definitions of the lines and points
variables {XY AB : ℝ → ℝ} -- Lines in the plane
variables {A B P X Y : ℝ} -- Points on the lines

-- Conditions 
def line_condition_1 : Prop := XY ≠ AB
def point_condition_1 : Prop := (P - A) = 2 * (B - P)
def perpendicular_condition : Prop := ∀ x, (X - A) * (XY x) = 0
def angle_condition : Prop := ∀ y, (Y - B) = 45 * (π / 180)

-- Prove relationship between PX and PY
theorem relationship_PX_PY (XY AB : ℝ → ℝ) (A B P X Y : ℝ)
  (h_line_1 : line_condition_1)
  (h_point_1 : point_condition_1)
  (h_perp : perpendicular_condition)
  (h_angle : angle_condition) : 
  ∃ (PX PY : ℝ), (PX = PY) ∨ (PX ≠ PY) :=
by
  sorry

end relationship_PX_PY_l260_260915


namespace general_form_l260_260797

open Nat

def seq (a : ℕ → ℕ) (a_1 : ℕ) (a_rec : ℕ → ℕ → Prop) : Prop :=
  a 1 = a_1 ∧ ∀ n ≥ 2, a_rec n (a n)

theorem general_form (a : ℕ → ℕ) :
  seq a 2 (λ n a_n, a n = 2 * a (n - 1) - 1) → (∀ n, a n = 2 ^ (n - 1) + 1) :=
by
  intros h
  sorry

end general_form_l260_260797


namespace problem_statement_l260_260030

-- Define the sum of divisors function for a positive integer n
def sigma (n : ℕ) : ℕ :=
  (1 to n).filter (λ d, n % d = 0).sum

-- Define the function f(n)
def f (n : ℕ) : ℚ :=
  (sigma n + n) / n

-- State the problem
theorem problem_statement : f(540) - f(180) = 7 / 90 :=
by
  sorry

end problem_statement_l260_260030


namespace possible_lengths_of_c_l260_260057

-- Definitions of the given conditions
variables (a b c : ℝ) (S : ℝ)
variables (h₁ : a = 4)
variables (h₂ : b = 5)
variables (h₃ : S = 5 * Real.sqrt 3)

-- The main theorem stating the possible lengths of c
theorem possible_lengths_of_c : c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
  sorry

end possible_lengths_of_c_l260_260057


namespace age_problem_l260_260484

theorem age_problem (x y : ℕ) 
  (h1 : 3 * x = 4 * y) 
  (h2 : 3 * y - x = 140) : x = 112 ∧ y = 84 := 
by 
  sorry

end age_problem_l260_260484


namespace find_lambda_l260_260106

-- Definitions of vectors a and b
def vec_a : ℝ × ℝ := (2, 3)
def vec_b (λ : ℝ) : ℝ × ℝ := (λ, 1)

-- Definition of the operation parallel
def is_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Definition of the condition given in the problem
def condition (λ : ℝ) : Prop :=
  is_parallel (vec_a.1 - λ, vec_a.2 - 1) (2 * vec_a.1 + λ, 2 * vec_a.2 + 1)

-- Main theorem statement
theorem find_lambda : ∃ λ : ℝ, condition λ ∧ λ = 2 / 3 :=
by {
  sorry
}

end find_lambda_l260_260106


namespace part1_part2_l260_260466

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 9 → x ∈ Set.Icc (-2 : ℝ) 4 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) →
  (a ∈ Set.Icc (19 / 4) (7 : ℝ)) :=
sorry

end part1_part2_l260_260466


namespace greater_expected_area_l260_260735

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l260_260735


namespace reflection_maps_A_B_to_A_l260_260668

variable (A A' B B' : ℝ × ℝ)

-- Definitions of the given points
def A := (-3, 2)
def A' := (2, -3)
def B := (-2, 5)
def B' := (5, -2)

-- Definition of the reflection in the line y = x
def reflect_in_y_eq_x (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.2, P.1)

-- Theorem stating that reflection in the line y = x maps A to A' and B to B'
theorem reflection_maps_A_B_to_A'_B' :
  reflect_in_y_eq_x A = A' ∧ reflect_in_y_eq_x B = B' :=
by 
  sorry

end reflection_maps_A_B_to_A_l260_260668


namespace fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l260_260227

def fractional_eq_solution_1 (x : ℝ) : Prop :=
  x + 5 / x = -6

theorem fractional_eq_solutions_1 : fractional_eq_solution_1 (-1) ∧ fractional_eq_solution_1 (-5) := sorry

def fractional_eq_solution_2 (x : ℝ) : Prop :=
  x - 3 / x = 4

theorem fractional_eq_reciprocal_sum
  (m n : ℝ) (h₀ : fractional_eq_solution_2 m) (h₁ : fractional_eq_solution_2 n) :
  m * n = -3 → m + n = 4 → (1 / m + 1 / n = -4 / 3) := sorry

def fractional_eq_solution_3 (x : ℝ) (a : ℝ) : Prop :=
  x + (a^2 + 2 * a) / (x + 1) = 2 * a + 1

theorem fractional_eq_solution_diff_square (a : ℝ) (h₀ : a ≠ 0)
  (x1 x2 : ℝ) (hx1 : fractional_eq_solution_3 x1 a) (hx2 : fractional_eq_solution_3 x2 a) :
  x1 + 1 = a → x2 + 1 = a + 2 → (x1 - x2) ^ 2 = 4 := sorry

end fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l260_260227


namespace honor_students_count_l260_260991

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260991


namespace factor_expression_l260_260043

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l260_260043


namespace probability_C1_selected_probability_at_most_one_A1_B1_selected_l260_260270

def students : Type := {x // x = "A1" ∨ x = "A2" ∨ x = "A3" ∨ x = "B1" ∨ x = "B2" ∨ x = "C1" ∨ x = "C2"}

def excel_math (s : students) : Prop := s.val = "A1" ∨ s.val = "A2" ∨ s.val = "A3"
def excel_physics (s : students) : Prop := s.val = "B1" ∨ s.val = "B2"
def excel_chemistry (s : students) : Prop := s.val = "C1" ∨ s.val = "C2"

def team : Type := { t : students × students × students // excel_math t.1 ∧ excel_physics t.2 ∧ excel_chemistry t.3 }

def sample_space : finset team :=
  { val := [("A1", "B1", "C1"), ("A1", "B1", "C2"), ("A1", "B2", "C1"), ("A1", "B2", "C2"),
            ("A2", "B1", "C1"), ("A2", "B1", "C2"), ("A2", "B2", "C1"), ("A2", "B2", "C2"),
            ("A3", "B1", "C1"), ("A3", "B1", "C2"), ("A3", "B2", "C1"), ("A3", "B2", "C2")].to_finset,
    nodup := sorry }

def event_C1_selected : finset team :=
  { val := [("A1", "B1", "C1"), ("A1", "B2", "C1"), ("A2", "B1", "C1"), ("A2", "B2", "C1"),
            ("A3", "B1", "C1"), ("A3", "B2", "C1")].to_finset,
    nodup := sorry }

def event_A1_B1_selected : finset team :=
  { val := [("A1", "B1", "C1"), ("A1", "B1", "C2")].to_finset,
    nodup := sorry }

theorem probability_C1_selected : 
  (event_C1_selected.card : ℚ) / (sample_space.card : ℚ) = 1 / 2 :=
by sorry

theorem probability_at_most_one_A1_B1_selected : 
  ((sample_space.card - event_A1_B1_selected.card) : ℚ) / (sample_space.card : ℚ) = 5 / 6 :=
by sorry

end probability_C1_selected_probability_at_most_one_A1_B1_selected_l260_260270


namespace savings_by_buying_together_l260_260715

-- Definitions
def cost_per_iPhone : ℝ := 600
def num_iPhones_individual : ℝ := 1  -- Each person buys individually
def num_iPhones_together : ℝ := 3  -- All three buy together
def discount_rate : ℝ := 0.05
def total_cost_without_discount (n : ℝ) : ℝ := n * cost_per_iPhone
def discount (total_cost : ℝ) : ℝ := total_cost * discount_rate
def total_cost_with_discount (total_cost : ℝ) : ℝ := total_cost - discount(total_cost)

-- Theorem: Calculate savings by buying together
theorem savings_by_buying_together :
  total_cost_without_discount num_iPhones_together -
  total_cost_with_discount (total_cost_without_discount num_iPhones_together) = 90 :=
by {
  sorry
}

end savings_by_buying_together_l260_260715


namespace problem_statement_l260_260077

noncomputable def a : ℝ := 2.68 * 0.74
noncomputable def b : ℝ := a^2 + Real.cos a

theorem problem_statement : b = 2.96535 := 
by 
  sorry

end problem_statement_l260_260077


namespace honor_students_count_l260_260987

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l260_260987


namespace HaroldAdrienne_meet_distance_l260_260577

theorem HaroldAdrienne_meet_distance :
  let adrienne_speed := 3
  let harold_speed := adrienne_speed + 1
  let christine_speed := harold_speed - 0.5
  let adrienne_time := 1
  let t := 3 in
  (adrienne_speed * (adrienne_time + t) = 12) ∧
  (christine_speed * t ≠ 12) := by
  sorry

end HaroldAdrienne_meet_distance_l260_260577


namespace multiple_of_shirt_cost_l260_260670

theorem multiple_of_shirt_cost (S C M : ℕ) (h1 : S = 97) (h2 : C = 300 - S)
  (h3 : C = M * S + 9) : M = 2 :=
by
  -- The proof will be filled in here
  sorry

end multiple_of_shirt_cost_l260_260670


namespace white_paint_amount_l260_260570

theorem white_paint_amount (total_blue_paint additional_blue_paint total_mix blue_parts red_parts white_parts green_parts : ℕ) 
    (h_ratio: blue_parts = 7 ∧ red_parts = 2 ∧ white_parts = 1 ∧ green_parts = 1)
    (total_blue_paint_eq: total_blue_paint = 140)
    (max_total_mix: additional_blue_paint ≤ 220 - total_blue_paint) 
    : (white_parts * (total_blue_paint / blue_parts)) = 20 := 
by 
  sorry

end white_paint_amount_l260_260570


namespace domain_of_f_l260_260963

open Function

def f (x : ℝ) : ℝ := 1 / (Real.sqrt (2 - x)) + Real.log (1 + x)

theorem domain_of_f :
  { x : ℝ | (2 - x > 0) ∧ (1 + x > 0) } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  funext x
  apply propext
  constructor
  · intro h
    cases h with h1 h2
    constructor
    · linarith
    · linarith
  · intro h
    cases h with h1 h2
    constructor
    · linarith
    · linarith

end domain_of_f_l260_260963


namespace find_a9_l260_260073

variable (S : ℕ → ℚ) (a : ℕ → ℚ) (n : ℕ) (d : ℚ)

-- Conditions
axiom sum_first_six : S 6 = 3
axiom sum_first_eleven : S 11 = 18
axiom Sn_definition : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n)
axiom arithmetic_sequence : ∀ n, a (n + 1) = a 1 + n * d

-- Problem statement
theorem find_a9 : a 9 = 3 := sorry

end find_a9_l260_260073


namespace find_f2_l260_260256

noncomputable def f : ℝ → ℝ := sorry

axiom function_property : ∀ (x : ℝ), f (2^x) + x * f (2^(-x)) = 1

theorem find_f2 : f 2 = 0 :=
by
  sorry

end find_f2_l260_260256


namespace maximum_value_OB_OA_l260_260887

open Real

noncomputable def C1_polar (ρ θ : ℝ) : Prop :=
  ρ * (cos θ + sin θ) = 4

noncomputable def C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * cos θ

def maximum_OB_OA (α : ℝ) : ℝ :=
  (1 / 4) * (sqrt 2 + 1)

theorem maximum_value_OB_OA :
  ∀ (A B : ℝ → ℝ → Prop) (θ α : ℝ),
    C1_polar (A ρ2 θ) α → C2_polar (B ρ1 θ) α →
    ∃ ρ1 ρ2 : ℝ,
      ρ1 = (4 / (cos α + sin α)) ∧ ρ2 = 2 * cos α ∧
      (∀ α, -π / 4 < α ∧ α < π / 2 → 
      (ρ2 / ρ1 <= maximum_OB_OA α)) :=
sorry

end maximum_value_OB_OA_l260_260887


namespace greater_expected_area_l260_260733

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l260_260733


namespace magnitude_diff_eq_sqrt_13_l260_260056

variables {α : Type*} [inner_product_space ℝ α]
open_locale real_inner_product_space

-- Given conditions
variables (a b : α)
variables (h_a : ∥a∥ = 2) (h_b : ∥b∥ = 3) (h_perp : inner a b = 0)

-- The theorem to prove
theorem magnitude_diff_eq_sqrt_13 : ∥b - a∥ = real.sqrt 13 :=
by
  -- Skip the proof
  sorry

end magnitude_diff_eq_sqrt_13_l260_260056


namespace find_vector_op_l260_260560

-- Definition of the vectors and their properties
variables (a b : ℝ) (θ : ℝ)

-- Conditions
def unit_vector_a : Prop := abs a = 1
def magnitude_b : Prop := abs b = real.sqrt 2
def magnitude_diff : Prop := abs (a - b) = 1
def angle_between_vectors (θ : ℝ) : Prop := cos θ = (real.sqrt 2) / 2 ∧ sin θ = (real.sqrt 2) / 2

-- The operation definition
def vector_op (a b θ: ℝ) : ℝ := abs (a * sin θ + b * cos θ)

-- The theorem statement with the conditions and result
theorem find_vector_op
  (ha : unit_vector_a a)
  (hb : magnitude_b b)
  (hd : magnitude_diff a b)
  (ht : angle_between_vectors θ) :
  vector_op a b θ = (real.sqrt 10) / 2 :=
sorry

end find_vector_op_l260_260560


namespace x_value_l260_260459

noncomputable def find_x (a x y : ℝ) (ha : a > 1) (hmax_y : y = sqrt 2) : Prop :=
  log a x + 2 * log x a + log x y = -3 → x = 1/8

-- Lean theorem statement
theorem x_value (a x y : ℝ) (ha : a > 1) (hmax_y : y = sqrt 2) 
  (h_eq : log a x + 2 * log x a + log x y = -3) : x = 1/8 :=
sorry

end x_value_l260_260459


namespace vasya_has_greater_expected_area_l260_260738

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l260_260738


namespace identify_element_l260_260331

-- Definitions based on conditions and problem statement
def molecular_weight := 84
def atomic_weight_fluoride := 19.00
def number_of_fluoride_atoms := 3
def total_weight_fluoride := number_of_fluoride_atoms * atomic_weight_fluoride
def atomic_weight_aluminum := 26.98

-- The main statement we want to prove
theorem identify_element (atomic_weight_al : ℝ) (compound_weight : ℝ) 
  (weight_fluoride : ℝ) (weight_aluminum : ℝ) 
  (num_fluoride_atoms : ℝ) (atomic_fluoride : ℝ) :
  compound_weight = molecular_weight →
  atomic_weight_fluoride = atomic_fluoride →
  num_fluoride_atoms = number_of_fluoride_atoms →
  weight_fluoride = num_fluoride_atoms * atomic_weight_fluoride →
  weight_aluminum = compound_weight - weight_fluoride →
  abs (weight_aluminum - atomic_weight_aluminum) < 0.5 →
  atomic_weight_aluminium = atomic_weight_al :=
by
  intros h_compound_weight h_atomic_weight_fluoride h_num_fluoride_atoms h_weight_fluoride h_weight_aluminum h_close_al
  sorry

end identify_element_l260_260331


namespace exists_acute_edge_l260_260585

-- Define a polyhedron and necessary properties
structure Polyhedron :=
  (vertices : Type)
  (faces : set (set vertices))
  (is_triangle : ∀ f ∈ faces, ∃ a b c, f = {a, b, c})

-- Define the edge property
def is_acutely_adjacently_summed (poly : Polyhedron) : Prop :=
  ∃ e : set poly.vertices, 
  (∃ (f1 f2 ∈ poly.faces), 
    e.subset f1 ∧ e.subset f2 ∧ (∃ a b,
    e = {a, b} ∧ 
    ∀ α β, α ∈ f1 ∧ β ∈ f2 ∧ α ≠ β → ⦃ angle a α β ⦄ < 180))

-- Prove the existence of the desired edge
theorem exists_acute_edge (poly : Polyhedron) (h : ∀ f ∈ poly.faces, ∃ a b c, f = {a, b, c}) : 
  is_acutely_adjacently_summed poly :=
sorry

end exists_acute_edge_l260_260585


namespace amicable_numbers_l260_260864

/--
  **Theorem**:
  Given \(n = 2\), the numbers \(A = 220\) and \(B = 284\) are amicable if the prime numbers are defined as follows:
  - \(p = 3 \cdot 2^n - 1 = 11\)
  - \(q = 3 \cdot 2^{n-1} - 1 = 5\)
  - \(r = 9 \cdot 2^{2n-1} - 1 = 71\)
  - \(A = 2^n \cdot p \cdot q = 220\)
  - \(B = 2^n \cdot r = 284\)
  Amicable numbers are defined as pairs of numbers such that each is equal to the sum of the proper divisors of the other.
-/
theorem amicable_numbers :
  ∀ (n : ℕ), n = 2 →
  (let p := 3 * 2^n - 1 in
   let q := 3 * 2^(n-1) - 1 in
   let r := 9 * 2^(2n-1) - 1 in
   let A := 2^n * p * q in
   let B := 2^n * r in
   (p = 11) ∧ (q = 5) ∧ (r = 71) ∧ (A = 220) ∧ (B = 284) ∧
   (∑ d in (divisors A).erase A, d = B) ∧ (∑ d in (divisors B).erase B, d = A)) := 
by
  intros n hn
  let p := 3 * 2^n - 1
  let q := 3 * 2^(n-1) - 1
  let r := 9 * 2^(2n-1) - 1
  let A := 2^n * p * q
  let B := 2^n * r
  sorry

end amicable_numbers_l260_260864


namespace projection_theorem_proof_l260_260224

variable {R : Type} [LinearOrderedField R]
variable (a b c : R)
variable (A B C : ℝ → R)

def cos := (x : ℝ) → R

axiom projection_theorem : 
  (a = b * cos C A + c * cos B C) ∧
  (b = c * cos A B + a * cos C A) ∧
  (c = a * cos B C + b * cos A B)

theorem projection_theorem_proof : 
  (a = b * cos C A + c * cos B C) ∧
  (b = c * cos A B + a * cos C A) ∧
  (c = a * cos B C + b * cos A B) := 
begin
  -- given these conditions
  exact projection_theorem,
end

end projection_theorem_proof_l260_260224


namespace vasya_has_greater_area_l260_260744

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l260_260744


namespace binomial_coefficient_expansion_l260_260085

theorem binomial_coefficient_expansion (n : ℕ) (x : ℝ) :
  (let M := (5 - 1) ^ n
   let N := 2 ^ n
   M - N = 240) →
  (let T := ∑ r in finset.range (n + 1), (-1) ^ r * (nat.choose n r) * 5 ^ (n - r) * x ^ (n - (3 * r / 2))
   T = x) →
  ((-1) ^ 2 * (nat.choose n 2) * 5 ^ (4 - 2) = 150) :=
begin
  sorry
end

end binomial_coefficient_expansion_l260_260085


namespace find_x_value_l260_260956

theorem find_x_value 
  (P Q R S O : Type) 
  (PO : ℝ) (OQ : ℝ) (RO : ℝ) (OS : ℝ) (RS : ℝ) (x : ℝ)
  (PR QS : P → Q → Prop)
  (intersect_at_O : PR → QS → Prop) :
  PR ∧ QS ∧ intersect_at_O PR QS ∧
  PO = 4 ∧ OQ = 10 ∧ RO = 4 ∧ OS = 5 ∧ RS = 8
  → x = 9 * Real.sqrt 2 :=
by
  sorry

end find_x_value_l260_260956


namespace vasya_expected_area_greater_l260_260746

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l260_260746


namespace Dickens_birth_day_l260_260943

theorem Dickens_birth_day : 
  (∀ (y : ℕ), y > 1812 → y ≤ 2022 → ((y % 4 = 0 ∧ y % 100 ≠ 0) ∨ y % 400 = 0) ∧ (y % 4 ≠ 0 ∨ (y % 100 = 0 ∧ y % 400 ≠ 0) → regular_year)) →
  (day_of_week 2022 2 7 = Tuesday) →
  (day_of_week 1812 2 7 = Saturday) :=
by
  sorry

end Dickens_birth_day_l260_260943


namespace proof_problem_l260_260669

section Problem

-- Define vectors and points
variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b A B C O : V)

-- Problem B
def condition_B : Prop :=
  a = (1 : ℝ) • (1, 3) ∧ a - b = (-1 : ℝ) • (-1, -3) ∧ ∃ k : ℝ, 0 < k ∧ b = k • a

-- Problem C
def condition_C : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ a - b ≠ 0) ∧ 
  (O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ A ≠ B ∧ B ≠ C) ∧
  a - 3 • b + 2 • C = 0 ∧
  ∃ k : ℝ, k > 0 ∧ abs (A - B) / abs (B - C) = 2

-- Final statement
def is_correct := (∃ k : ℝ, 0 < k ∧ b = k • a) ∧
                   (∃ k : ℝ, k > 0 ∧ abs (A - B) / abs (B - C) = 2)

theorem proof_problem :
  is_correct :=
  sorry

end Problem

end proof_problem_l260_260669


namespace compute_series_l260_260189

noncomputable def infinite_series (a b : ℝ) (h : a > b) : ℝ :=
  let c := a - b in
  ∑' (n : ℕ) (0 < n) , (1 / (n * c * (n * c + c)))

theorem compute_series (a b : ℝ) (h : a > b):
  infinite_series a b h = 1 / ((a - b) * b) :=
by 
  sorry

end compute_series_l260_260189


namespace selling_price_of_book_l260_260689

theorem selling_price_of_book (cost_price : ℝ) (profit_percentage : ℝ) (profit : ℝ) (selling_price : ℝ) 
  (h₁ : cost_price = 60) 
  (h₂ : profit_percentage = 25) 
  (h₃ : profit = (profit_percentage / 100) * cost_price) 
  (h₄ : selling_price = cost_price + profit) : 
  selling_price = 75 := 
by
  sorry

end selling_price_of_book_l260_260689


namespace first_term_of_arithmetic_sequence_l260_260445

theorem first_term_of_arithmetic_sequence :
  ∃ (a_1 : ℤ), ∀ (d n : ℤ), d = 3 / 4 ∧ n = 30 ∧ a_n = 63 / 4 → a_1 = -6 := by
  sorry

end first_term_of_arithmetic_sequence_l260_260445


namespace probability_of_convex_quadrilateral_l260_260766

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_convex_quadrilateral :
  let num_points := 8
  let total_chords := binomial num_points 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
by
  -- definitions
  let num_points := 8
  let total_chords := binomial 8 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  
  -- assertion of result
  have h : (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
    sorry
  exact h

end probability_of_convex_quadrilateral_l260_260766


namespace number_of_blue_faces_l260_260704

theorem number_of_blue_faces (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end number_of_blue_faces_l260_260704


namespace continuous_function_properties_l260_260910

theorem continuous_function_properties (f : ℝ → ℝ) 
  (hf_cont : Continuous f)
  (hf_eq : ∀ x y : ℝ, f(x + y) * f(x - y) = f x ^ 2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x ≠ 0) := 
by
  sorry

end continuous_function_properties_l260_260910


namespace find_N_l260_260204

theorem find_N (N : ℕ) (h_pos : N > 0) (h_small_factors : 1 + 3 = 4) 
  (h_large_factors : N + N / 3 = 204) : N = 153 :=
  by sorry

end find_N_l260_260204


namespace SUVs_purchased_l260_260288

theorem SUVs_purchased (x : ℕ) (hToyota : ℕ) (hHonda : ℕ) (hNissan : ℕ) 
  (hRatioToyota : hToyota = 7 * x) 
  (hRatioHonda : hHonda = 5 * x) 
  (hRatioNissan : hNissan = 3 * x) 
  (hToyotaSUV : ℕ) (hHondaSUV : ℕ) (hNissanSUV : ℕ) 
  (hToyotaSUV_num : hToyotaSUV = (50 * hToyota) / 100) 
  (hHondaSUV_num : hHondaSUV = (40 * hHonda) / 100) 
  (hNissanSUV_num : hNissanSUV = (30 * hNissan) / 100) : 
  hToyotaSUV + hHondaSUV + hNissanSUV = 64 := 
by
  sorry

end SUVs_purchased_l260_260288


namespace inverse_of_k_l260_260547

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l260_260547


namespace vasya_has_greater_area_l260_260742

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l260_260742


namespace smallest_positive_integer_l260_260657

theorem smallest_positive_integer (
  a : ℕ
) : 
  (a ≡ 5 [MOD 6]) ∧ (a ≡ 7 [MOD 8]) → a = 23 :=
by sorry

end smallest_positive_integer_l260_260657


namespace XBYBprime_has_incircle_l260_260909

-- Definitions based on given conditions
variables {A A' B C C' B' X Y : Type}
          [convex_hexagon AA'BCCB']
          [tangent_incircle_triangle AC (incircle (triangle A'B'C'))]
          [tangent_incircle_triangle A'C' (incircle (triangle ABC))]
          [meeting_point AB A'B' X]
          [meeting_point BC B'C' Y]
          [convex_quad XBYB']

-- Theorem statement
theorem XBYBprime_has_incircle : has_incircle (quadrilateral XBYB') :=
by { sorry }

end XBYBprime_has_incircle_l260_260909


namespace simplify_root_l260_260663

theorem simplify_root (a b : ℕ) (h : a * (b ^ (1 / 4 : ℝ)) = (2 ^ (7 / 4 : ℝ)) * (3 ^ (3 / 4 : ℝ))) : 
  a + b = 218 := 
sorry

end simplify_root_l260_260663


namespace percentage_of_male_employees_is_65_l260_260881

def total_employees : ℕ := 6400
def males_below_50 : ℕ := 3120
def percentage_males_at_least_50 : ℝ := 0.25

noncomputable def percentage_male_employees : ℝ :=
  let M := (males_below_50 / (1 - percentage_males_at_least_50)) in
  (M / total_employees) * 100

theorem percentage_of_male_employees_is_65 :
  percentage_male_employees = 65 :=
by
  sorry

end percentage_of_male_employees_is_65_l260_260881


namespace number_a_eq_223_l260_260958

theorem number_a_eq_223 (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end number_a_eq_223_l260_260958


namespace k_inverse_k_inv_is_inverse_l260_260544

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x - 4
def k (x : ℝ) : ℝ := f (g x)

def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem k_inverse {x : ℝ} : k_inv (k x) = x :=
by
  sorry

theorem k_inv_is_inverse {x y : ℝ} : k_inv (y) = x ↔ y = k(x) :=
by
  sorry

end k_inverse_k_inv_is_inverse_l260_260544


namespace monotonicity_f_intersection_point_inequality_ln_sum_l260_260093

/-- Part (1): Monotonicity of the function f(x) = 2a ln(x) - x^2 + a -/
theorem monotonicity_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → (a ≤ 0 → Deriv (λ x, 2 * a * log x - x ^ 2 + a) x < 0) ∧ 
  (a > 0 → (forall y: ℝ, 0 < y ∧ y < sqrt a → Deriv (λ y, 2 * a * log y - y ^ 2 + a) y > 0) ∧ 
  (∀ z : ℝ, z > sqrt a → Deriv (λ z, 2 * a * log z - z ^ 2 + a) z < 0))) := sorry

/-- Part (2): Intersection Point x0 is less than the arithmetic mean of x1 and x2 -/
theorem intersection_point (x1 x2 : ℝ) (a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2)
  (h4 : f x1 = 0) (h5 : f x2 = 0) (hx0 : x0 = (x1 + x2) / 2):
  let x0 := (x1 + x2) / 2 in x0 < (x1 + x2) / 2 := sorry

/-- Part (3): Prove the inequality ln(n+1) < 1/2 + sum (1/(i+1)) for n ∈ ℕ*-/
theorem inequality_ln_sum (n : ℕ) (hn : 0 < n) : 
  log (n + 1) < 1 / 2 + ∑ i in range n, 1 / (i + 2) := sorry

end monotonicity_f_intersection_point_inequality_ln_sum_l260_260093


namespace coords_of_point_wrt_origin_l260_260953

-- Define the given point
def given_point : ℝ × ℝ := (1, -2)

-- Goal: Prove that the coordinates of the given point with respect to the origin are (1, -2)
theorem coords_of_point_wrt_origin (p : ℝ × ℝ) (hp : p = given_point) : p = (1, -2) :=
by
  rw hp
  rfl

end coords_of_point_wrt_origin_l260_260953


namespace honor_students_count_l260_260996

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260996


namespace find_integer_n_l260_260015

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end find_integer_n_l260_260015


namespace find_larger_number_l260_260631

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 :=
  sorry

end find_larger_number_l260_260631


namespace num_complementary_sets_eq_117_l260_260405

structure Card :=
(shape : Type)
(color : Type)
(shade : Type)

def deck_condition: Prop := 
  ∃ (deck : List Card), 
  deck.length = 27 ∧
  ∀ c1 c2 c3, c1 ∈ deck ∧ c2 ∈ deck ∧ c3 ∈ deck →
  (c1.shape ≠ c2.shape ∨ c2.shape ≠ c3.shape ∨ c1.shape = c3.shape) ∧
  (c1.color ≠ c2.color ∨ c2.color ≠ c3.color ∨ c1.color = c3.color) ∧
  (c1.shade ≠ c2.shade ∨ c2.shade ≠ c3.shade ∨ c1.shade = c3.shade)

theorem num_complementary_sets_eq_117 :
  deck_condition → ∃ sets : List (List Card), sets.length = 117 := sorry

end num_complementary_sets_eq_117_l260_260405


namespace geometric_sequence_formula_sum_first_n_terms_l260_260885

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 + a 3 + a 4 = 28 ∧ q > 1

def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 + 2 = (a 2 + a 4) / 2

-- Problem 1: General formula for the sequence {a_n}
theorem geometric_sequence_formula (a : ℕ → ℝ) (q : ℝ)
  (h1 : is_geometric_sequence a q) (h2 : condition1 a q) (h3 : condition2 a q) :
  ∀ n, a n = 2 ^ n :=
sorry

-- Problem 2: Sum of the first n terms for T_n
def b (n : ℕ) : ℝ := Real.log2 (a (n + 5))

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

def T (n : ℕ) : ℝ := ∑ i in Finset.range n, S i / n

theorem sum_first_n_terms (h1 : ∀ n, a n = 2 ^ n) :
  ∀ n, T n = (n^2 + 23*n) / 4 :=
sorry

end geometric_sequence_formula_sum_first_n_terms_l260_260885


namespace circle_tangent_to_fixed_circles_l260_260160

variable {Ω : Type*} [EuclideanGeometry Ω]

-- Define the points and their properties
variable (A B C H O N M : Ω)
variable (ω ω₁ ω₂ : Circle Ω)

-- Define the conditions
variable (HA : isOrthocenter H (triangle A B C))
variable (circA : isPointOnCircle A ω)
variable (circO : isCircumcenter O (triangle A B C))
variable (midN : isMidpoint N A H)
variable (midM : isMidpoint M B C)

-- Define the target fixed circles centered at M with specific radii
variable (circ_ω₁ : isCircle ω₁ M (radius ω - OM))
variable (circ_ω₂ : isCircle ω₂ M (radius ω + OM))

-- The main statement to be proved
theorem circle_tangent_to_fixed_circles :
  ∀ (A: Ω), isPointOnCircle A ω →
            tangent_to_circle (circle (diameter A H)) ω₁ ∧
            tangent_to_circle (circle (diameter A H)) ω₂
  := by
    sorry

end circle_tangent_to_fixed_circles_l260_260160


namespace variance_after_scaling_l260_260443

def original_variance (ξ : ℝ → ℝ) (D : ℝ) : Prop :=
  D = S^2

theorem variance_after_scaling (D : ℝ) (S : ℝ) (ξ : ℝ → ℝ) (h : D = S^2) : 
  D * 100 = 100 * S^2 :=
by
  rw [h]
  sorry

end variance_after_scaling_l260_260443


namespace divide_figure_into_equal_areas_l260_260821

/-- A given figure composed of two rectangles with right angles. We need to prove that
    a line passing through the centers of symmetry of both rectangles divides the figure into
    two polygons of equal area. -/
theorem divide_figure_into_equal_areas (R1 R2 : Type*) [rectangles_with_right_angles R1] 
  [rectangles_with_right_angles R2] (hs1 : center_symmetry R1) (hs2 : center_symmetry R2) :
  ∃ (line : Type*), divides_into_equal_areas R1 R2 line := 
begin
  sorry
end

end divide_figure_into_equal_areas_l260_260821


namespace explicit_quadratic_formula_l260_260442

def quadratic_function_formula (a b c : ℝ) (y : ℝ → ℝ) : Prop :=
  ∀ x, y x = a * x^2 + b * x + c

variable {a b c y y1 y2 y3 : ℝ}
hypothesis ha : a > 0
hypothesis hb : b > 0
hypothesis hc1 : y 0 = y1
hypothesis hc2 : y 1 = y2
hypothesis hc3 : y (-1) = y3
hypothesis h_conditions : y1^2 = 1 ∧ y2^2 = 1 ∧ y3^2 = 1

theorem explicit_quadratic_formula :
  quadratic_function_formula a b c y →
  (∀ x, y x = x^2 + x - 1) :=
by
  intro h
  sorry

end explicit_quadratic_formula_l260_260442


namespace vasya_has_greater_area_l260_260741

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l260_260741


namespace range_of_a_l260_260092

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then (1/2)^x else Real.logb 2 x

theorem range_of_a (a : ℝ) (h : f a ≥ 2) : a ∈ Set.Iic (-1) ∪ Set.Ici 4 :=
by {
  sorry
}

end range_of_a_l260_260092


namespace three_digit_number_is_473_l260_260355

theorem three_digit_number_is_473 (x y z : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) (h5 : 0 ≤ z) (h6 : z ≤ 9)
  (h7 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 99)
  (h8 : x + y + z = 14)
  (h9 : x + z = y) : 100 * x + 10 * y + z = 473 :=
by
  sorry

end three_digit_number_is_473_l260_260355


namespace tetrahedron_surface_area_correct_l260_260087

noncomputable def tetrahedron_surface_area (AB BC AC volume : ℝ)
    (incenter_as_orthogonal_projection : Prop) : ℝ :=
if (AB = 7) ∧ (BC = 8) ∧ (AC = 9) ∧ (volume = 40) ∧ incenter_as_orthogonal_projection then 
  60 + 12 * Real.sqrt 5 
else 
  0

theorem tetrahedron_surface_area_correct :
  tetrahedron_surface_area 7 8 9 40
    (∃ (D : ℝ), ∀ (A B C : ℝ),
       orthogonal_projection (plane_spanned_by_points A B C) D = incenter_of_triangle A B C) =
  60 + 12 * Real.sqrt 5 :=
sorry

end tetrahedron_surface_area_correct_l260_260087


namespace percent_employed_females_in_employed_population_l260_260167

def percent_employed (population: ℝ) : ℝ := 0.64 * population
def percent_employed_males (population: ℝ) : ℝ := 0.50 * population
def percent_employed_females (population: ℝ) : ℝ := percent_employed population - percent_employed_males population

theorem percent_employed_females_in_employed_population (population: ℝ) : 
  (percent_employed_females population / percent_employed population) * 100 = 21.875 :=
by
  sorry

end percent_employed_females_in_employed_population_l260_260167


namespace total_paint_needed_l260_260153

-- Define the dimensions of the classroom
def length := 15 -- feet
def width := 12  -- feet
def height := 10 -- feet
def blackboards_and_windows_area := 80 -- square feet

-- Define the number of classrooms
def classrooms := 4

-- Define the areas of the walls of a single classroom
def length_walls_area := 2 * (length * height)
def width_walls_area := 2 * (width * height)

-- Define the total walls area in one classroom
def total_walls_area (length_walls_area width_walls_area : ℕ) := length_walls_area + width_walls_area

-- Define the paintable area in one classroom
def paintable_area_one_classroom (total_walls_area blackboards_and_windows_area : ℕ) := total_walls_area - blackboards_and_windows_area

-- Define the paintable area for all classrooms
def total_paintable_area (paintable_area_one_classroom classrooms : ℕ) := paintable_area_one_classroom * classrooms

-- Main theorem: calculating the total paintable area
theorem total_paint_needed : 
  total_paintable_area (paintable_area_one_classroom (total_walls_area length_walls_area width_walls_area) blackboards_and_windows_area) classrooms = 1840 := by
  sorry

end total_paint_needed_l260_260153


namespace find_a_l260_260769

noncomputable def has_exactly_one_solution_in_x (a : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 + 2*a*x + a + 5| = 3 → x = -a

theorem find_a (a : ℝ) : has_exactly_one_solution_in_x a ↔ (a = 4 ∨ a = -2) :=
by
  sorry

end find_a_l260_260769


namespace find_n_l260_260819

theorem find_n (n : ℕ) (M N : ℕ) (hM : M = 4 ^ n) (hN : N = 2 ^ n) (h : M - N = 992) : n = 5 :=
sorry

end find_n_l260_260819


namespace evaluate_expression_l260_260309

theorem evaluate_expression :
  (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 :=
by
  sorry

end evaluate_expression_l260_260309


namespace sum_arithmetic_geometric_sequence_l260_260084

theorem sum_arithmetic_geometric_sequence (a_1 : ℕ) (a_4 : ℕ) (b_2 : ℕ) (b_5 : ℕ) (n : ℕ) 
  (h₁ : a_1 = 2) (h₂ : a_4 = 8) (h₃ : b_2 = 4) (h₄ : b_5 = 32) :
  let a : ℕ → ℕ := λ n, 2 * n,
      b : ℕ → ℕ := λ n, 2^n,
      S_n : ℕ := (finset.range n).sum (λ k, a (k+1) + b (k+1)) in
  S_n = n^2 + n + 2^(n+1) - 2 := 
sorry

end sum_arithmetic_geometric_sequence_l260_260084


namespace path_length_pq_l260_260264

theorem path_length_pq (PQ_length : ℝ) (hPQ : PQ_length = 73) : 
  let segments := 3 * PQ_length in 
  segments = 219 :=
by
  /- Add the proof here -/
  sorry

end path_length_pq_l260_260264


namespace meaning_of_poverty_l260_260681

theorem meaning_of_poverty (s : String) : s = "poverty" ↔ s = "poverty" := sorry

end meaning_of_poverty_l260_260681


namespace integer_not_natural_l260_260618

theorem integer_not_natural (n : ℕ) (a : ℝ) (b : ℝ) (x y z : ℝ) 
  (h₁ : x = (1 + a) ^ n) 
  (h₂ : y = (1 - a) ^ n) 
  (h₃ : z = a): 
  ∃ k : ℤ, (x - y) / z = ↑k ∧ (k < 0 ∨ k ≠ 0) :=
by 
  sorry

end integer_not_natural_l260_260618


namespace max_ab_value_l260_260447

noncomputable def max_ab (a b : ℝ) (h : 0 < a ∧ 0 < b ∧ (1 / ((2 * a + b) * b) + 2 / ((2 * b + a) * a) = 1)) : ℝ :=
  2 - (2 * Real.sqrt 2) / 3

theorem max_ab_value (a b : ℝ) (h : 0 < a ∧ 0 < b ∧ 1 / ((2 * a + b) * b) + 2 / ((2 * b + a) * a) = 1) :
  ab = max_ab a b :=
sorry

end max_ab_value_l260_260447


namespace distance_between_vertices_l260_260552

def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, (a * x * x) + (b * x) + c)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p1.1 - p2.1
  let dy := p1.2 - p2.2
  real.sqrt (dx * dx + dy * dy)

theorem distance_between_vertices :
  let A := vertex 1 6 5
  let B := vertex 1 (-8) 20
  distance A B = real.sqrt 113 :=
by
  sorry

end distance_between_vertices_l260_260552


namespace odd_function_expression_l260_260326

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -x + 1 else -(x + 1)

theorem odd_function_expression {x : ℝ} (hx : x < 0) :
  f x = -x - 1 :=
by unfold f; split_ifs; sorry

end odd_function_expression_l260_260326


namespace determine_truth_and_criminal_l260_260920

open Classical

-- Define individuals
inductive Person
| A | B | C | D

open Person

-- Define statements as functions with corresponding truths
def statement_A (truths : Person → Prop) : Prop := 
  ¬ truths A

def statement_B (truths : Person → Prop) : Prop :=
  ¬ truths D

def statement_C (truths : Person → Prop) : Prop :=
  ¬ truths B

def statement_D (truths : Person → Prop) : Prop :=
  ¬ truths B

-- Define the main conditions
def main_condition (truths : Person → Prop) : Prop :=
  (statement_A truths ∨ statement_B truths ∨ statement_C truths ∨ statement_D truths) ∧
  (∀ p : Person, (p = A → truths A) ∧ (p = B → truths B) ∧ (p = C → truths C) ∧ (p = D → truths D)) ∧
  (∃! p : Person, truths p)

-- Theorem to determine who is telling the truth and who is the criminal
theorem determine_truth_and_criminal : ∃ p : Person, p = A ∧ ∃ t : Person, t = D ∧ main_condition (λ p, p = t) :=
by
  sorry

end determine_truth_and_criminal_l260_260920


namespace sqrt_factorial_product_l260_260308

theorem sqrt_factorial_product : sqrt (fact 3 * fact 3) = 6 := by
  sorry

end sqrt_factorial_product_l260_260308


namespace diameter_of_inscribed_circle_l260_260651

theorem diameter_of_inscribed_circle (AB AC BC : ℝ) (h1 : AB = 13) (h2 : AC = 5) (h3 : BC = 12) :
  let s := (AB + AC + BC) / 2 in
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
  let r := K / s in
  let d := 2 * r in
  d = 4 :=
by
  sorry

end diameter_of_inscribed_circle_l260_260651


namespace Jiyeon_biggest_number_l260_260527

def isLargestNumber (n : ℕ) (digits : list ℕ) : Prop :=
  ∃ (perm : list ℕ), perm.perm digits ∧ (perm.foldl (λ acc d, 10 * acc + d) 0 = n) ∧
  ∀ (m : ℕ) (perm' : list ℕ), perm'.perm digits → (perm'.foldl (λ acc d, 10 * acc + d) 0 ≤ n)

theorem Jiyeon_biggest_number :
  isLargestNumber 541 [1, 4, 5] :=
sorry

end Jiyeon_biggest_number_l260_260527


namespace number_of_pots_l260_260636

theorem number_of_pots (f s t : ℕ) (total : ℕ) (n : ℕ) : 
  (∀ P, f = 53 ∧ s = 181 ∧ total = 109044 ∧ t = f + s ∧ total = P * t) → n = 466 :=
by
  intro h
  obtain ⟨hf, hs, htotal, ht, heq⟩ := h 466
  have h1 : f + s = 53 + 181 := by rw [hf, hs]
  have h2 : 234 = 53 + 181 := by norm_num
  have h3 : t = 234 := by rw [t, h1, h2]
  have h4 : 109044 = 466 * t := by rw [htotal, ht]
  have h5 : 109044 = 466 * 234 := by norm_num
  rw [← h3, h5] at h4
  rw [← h4]
  exact rfl

end number_of_pots_l260_260636


namespace intersection_eq_l260_260472

noncomputable def A : Set ℕ := {1, 2, 3, 4}
noncomputable def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_eq : A ∩ B = {2, 3, 4} := 
by
  sorry

end intersection_eq_l260_260472


namespace speed_ratio_l260_260359

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end speed_ratio_l260_260359


namespace isosceles_triangle_perimeter_l260_260865

-- Define an isosceles triangle where the sides can either be (6cm, 6cm, 3cm) or (3cm, 3cm, 6cm)
structure IsoscelesTriangle (a b c : ℕ) : Prop :=
(is_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
(is_isosceles : a = b ∨ a = c ∨ b = c)

-- The theorem proving the valid perimeter given the problem's conditions
theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), IsoscelesTriangle a b c ∧ (a = 6 ∨ b = 6 ∨ c = 6) ∧ (a = 3 ∨ b = 3 ∨ c = 3) → a + b + c = 15 :=
begin
  sorry
end

end isosceles_triangle_perimeter_l260_260865


namespace initial_oranges_per_tree_l260_260005

theorem initial_oranges_per_tree (x : ℕ) (h1 : 8 * (5 * x - 2 * x) / 5 = 960) : x = 200 :=
sorry

end initial_oranges_per_tree_l260_260005


namespace external_bisector_of_triangle_ABC_at_B_AE_eq_BE_plus_BC_l260_260216

-- Definitions and conditions
variables (A B C D E : Type*)
variables (circle : Set Point)
variables [IsCircle circle]
variables (on_circle : ∀ {P : Point}, P ∈ {A, B, C, D, E} → P ∈ circle)
variables (arc_AC_midpoint : IsMidpointOfArc D A C)
variables (arc_DC_contains_B_not_A : B ∈ Arc D C ∧ A ∉ Arc D C)
variables (E_projection : OrthogonalProjection D (Line A B E))

-- Theorem statements
theorem external_bisector_of_triangle_ABC_at_B :
  IsExternalBisector (Line B D) (triangle A B C) B := sorry

theorem AE_eq_BE_plus_BC :
  Distance A E = Distance B E + Distance B C := sorry

end external_bisector_of_triangle_ABC_at_B_AE_eq_BE_plus_BC_l260_260216


namespace vitya_wins_with_perfect_play_l260_260177

/-- Kolya and Vitya are playing a marking game on an infinite grid. 
Starting with Kolya, they alternately mark vertices at the intersections 
of vertical and horizontal lines. Starting from Kolya's second turn, 
all marked vertices must lie on the vertices of a convex polygon. The 
player who cannot make a valid move loses. Prove that Vitya has a 
winning strategy with perfect play. -/
theorem vitya_wins_with_perfect_play : 
  ∃ (strategy : ℕ → ℕ × ℕ), (∀ n, convex_hull {strategy i | i < n}.pair) ∧
  (∀ n, ∃ m, strategy m ∉ convex_hull {strategy i | i < n}.pair) :=
sorry

end vitya_wins_with_perfect_play_l260_260177


namespace ratio_surface_area_l260_260339

theorem ratio_surface_area (α : ℝ) (R : ℝ) (hα : α ≠ 0 ∧ α ≠ π / 2) : 
  let S_full := 6 * R^2 * Real.sqrt 3 * (3 + 4 * (Real.cot α)^2),
      S_sphere := 4 * Real.pi * R^2 
  in S_full / S_sphere = (3 * Real.sqrt 3 / (2 * Real.pi)) * (3 + 4 * (Real.cot α)^2) := by
sory

end ratio_surface_area_l260_260339


namespace find_cos_of_angle_in_third_quadrant_l260_260853

theorem find_cos_of_angle_in_third_quadrant (B : Real) (hB_quad : ∀ θ, θ = B → 180° < θ ∧ θ < 270°) (h_sin : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end find_cos_of_angle_in_third_quadrant_l260_260853


namespace simplify_expression_l260_260597

-- Given variables a, b, c are real numbers
variables {a b c : ℝ}

-- Declare our theorem with the given conditions
theorem simplify_expression (h : a ≠ b + c) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  (\left(\frac{1}{a * b} + \frac{1}{a * c} + \frac{1}{b * c}\right) * \frac{a * b}{a^2 - (b + c)^2}) = \frac{1}{c * (a - b - c)} :=
sorry

end simplify_expression_l260_260597


namespace number_of_points_in_star_polygon_l260_260377

theorem number_of_points_in_star_polygon :
  ∀ (n : ℕ) (D C : ℕ),
    (∀ i : ℕ, i < n → C = D - 15) →
    n * (D - (D - 15)) = 360 → n = 24 :=
by
  intros n D C h1 h2
  sorry

end number_of_points_in_star_polygon_l260_260377


namespace rad_to_deg_conversion_l260_260327

theorem rad_to_deg_conversion (pi_rad_eq_deg: Real): 
  (pi_rad_eq_deg = 180) -> (-23 / 12 * pi_rad_eq_deg = -345) := by {
  intro H,
  sorry
}

end rad_to_deg_conversion_l260_260327


namespace new_figure_perimeter_equals_5_l260_260939

-- Defining the side length of the square and the equilateral triangle
def side_length : ℝ := 1

-- Defining the perimeter of the new figure
def new_figure_perimeter : ℝ := 3 * side_length + 2 * side_length

-- Statement: The perimeter of the new figure equals 5
theorem new_figure_perimeter_equals_5 :
  new_figure_perimeter = 5 := by
  sorry

end new_figure_perimeter_equals_5_l260_260939


namespace tel_aviv_rain_days_l260_260946

-- Define the conditions
def chance_of_rain : ℝ := 0.5
def days_considered : ℕ := 6
def given_probability : ℝ := 0.234375

-- Helper function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function P(X = k)
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- The main theorem to prove
theorem tel_aviv_rain_days :
  ∃ k, binomial_probability days_considered k chance_of_rain = given_probability ∧ k = 2 := by
  sorry

end tel_aviv_rain_days_l260_260946


namespace polynomial_factorization_l260_260973

theorem polynomial_factorization (p q : ℝ) :
  (∃ r s : ℝ, px^4 + qx^3 + 40x^2 - 20x + 10 = (5 * x^2 - 4 * x + 2) * (r * x^2 + s * x + 5))
  → p = 112.5 ∧ q = -52.5 :=
by {
  sorry
}

end polynomial_factorization_l260_260973


namespace cone_volume_is_correct_l260_260944

-- Define the base area and height of the cone
def base_area (A : ℝ) := A = 30
def height (h : ℝ) := h = 6

-- Define the volume formula for a cone
def cone_volume (A h V : ℝ) := V = (1 / 3) * A * h

-- The statement we need to prove
theorem cone_volume_is_correct (A h V : ℝ) (h_base_area : base_area A) (h_height : height h) : cone_volume A h V → V = 60 :=
by
  sorry

end cone_volume_is_correct_l260_260944


namespace paulina_value_l260_260528

theorem paulina_value (x : ℝ) (hx : 0 < x ∧ x < 90) (hsinx : sin x = 1 / 2) :
  x = 30 ∧ cos x = sqrt 3 / 2 :=
by
  sorry

end paulina_value_l260_260528


namespace min_route_length_5x5_l260_260948

-- Definition of the grid and its properties
def grid : Type := Fin 5 × Fin 5

-- Define a function to calculate the minimum route length
noncomputable def min_route_length (grid_size : ℕ) : ℕ :=
  if h : grid_size = 5 then 68 else 0

-- The proof problem statement
theorem min_route_length_5x5 : min_route_length 5 = 68 :=
by
  -- Skipping the actual proof
  sorry

end min_route_length_5x5_l260_260948


namespace diameter_Γ_2007_l260_260902

noncomputable def parabola (x : ℝ) : ℝ := x^2

def circle_tangent_to_parabola (d : ℝ) : Prop :=
∃ x : ℝ, parabola x = 0 ∧ d = 1

def circle_sequence_tangent (Γ : ℕ → ℝ) (parabola : ℝ → ℝ) : Prop :=
Γ 1 = 1 ∧ ∀ n : ℕ, n > 0 → 
    (∃ d : ℝ, ∃ s : ℝ, ∃ x : ℝ, 
      Γ (n + 1) = 1 + 2 * real.sqrt (s) ∧ 
      s = ∑ i in finset.range (n + 1), Γ (i + 1) ∧
      parabola x = (Γ n / 2) ^ 2)

theorem diameter_Γ_2007 :
  ∀ Γ : ℕ → ℝ, circle_sequence_tangent Γ parabola →
    Γ 2007 = 4013 :=
begin
  intros Γ hΓ,
  sorry
end

end diameter_Γ_2007_l260_260902


namespace min_triples_count_l260_260554

theorem min_triples_count (a : Fin 9 → ℝ) (m : ℝ)
  (h_avg : (∑ i, a i) / 9 = m) :
  ∃ A : ℕ, A = 28 ∧
  (∀ i j k, 1 ≤ i → i < j → j < k → k ≤ 9 → a i + a j + a k ≥ 3 * m) :=
  sorry

end min_triples_count_l260_260554


namespace meeting_point_distance_l260_260286

noncomputable def meeting_distance_to_Hillcreast : ℝ :=
  let d := 120
  let v_h := 5
  let v_s := λ t, 4 + (t / 2)
  let person_H := λ t, v_h * t
  let sum_series := λ t, sum (range (t / 2)) (λ n, 2 * (4 + n))
  let person_S := λ t, if even t then sum_series t else sum_series (t - 1) + (4 + ((t - 1) / 2))
  let t := some (argmin (λ t, abs (person_H t + person_S t - d)) (range 100)) -- upper bound for practical purposes
  let dist_to_H := min (abs (d - v_h * t)) (abs (person_H t - (d - person_S t)))
  dist_to_H

theorem meeting_point_distance:
  ∃ ε, ε > 0 ∧ abs (meeting_distance_to_Hillcreast - 10.73) < ε :=
by
  sorry

end meeting_point_distance_l260_260286


namespace problem_is_even_and_satisfies_eqn_l260_260446

variable (f : ℝ → ℝ)

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

theorem problem_is_even_and_satisfies_eqn :
  is_even f →
  (∀ x : ℝ, f (x + 2) = x * f x) →
  f 1 = 0 :=
by
  intros h_even h_eqn
  specialize h_eqn (-1)
  have h1 : f 1 = (-1) * f (-1) := by exact h_eqn
  tactic.assumption sorry

end problem_is_even_and_satisfies_eqn_l260_260446


namespace construct_point_inside_triangle_l260_260523

variables {A B C K : Type}
variables [A ∈ Triangle ABC] [B ∈ Triangle ABC] [C ∈ Triangle ABC] [K ∈ Triangle ABC]
variables [is_acute_triangle ABC]
variables {angle_KBA angle_KAB angle_KBC angle_KCB : ℝ}

def valid_point (A B C K : Type) (angle_KBA angle_KAB angle_KBC angle_KCB : ℝ)
    [A ∈ Triangle ABC] [B ∈ Triangle ABC] [C ∈ Triangle ABC] [K ∈ Triangle ABC]
    [is_acute_triangle ABC] :
    Prop :=
  (angle_KBA = 2 * angle_KAB) ∧ (angle_KBC = 2 * angle_KCB)

theorem construct_point_inside_triangle {A B C K : Type}
    [is_acute_triangle ABC]
    {angle_KBA angle_KAB angle_KBC angle_KCB : ℝ}
    (h1 : angle_KBA = 2 * angle_KAB)
    (h2 : angle_KBC = 2 * angle_KCB)
    : valid_point A B C K angle_KBA angle_KAB angle_KBC angle_KCB :=
  by sorry

end construct_point_inside_triangle_l260_260523


namespace angle_bisector_median_inequality_l260_260555

-- Definitions for the given conditions
variables {α : Type*}
variables (a b c s s3 f1 f2 : ℝ)
variables [triangle : a + b > c] [triangle : a + c > b] [triangle : b + c > a]

-- The Lean statement of the problem
theorem angle_bisector_median_inequality 
  (h1 : f1 = sqrt ( b * c * (1 - (a/(b + c))^2 ) ))
  (h2 : f2 = sqrt ( a * b * (1 - (c/(a + b))^2 ) ))
  (h3 : s3 = sqrt ( 2 * a^2 + 2 * c^2 - b^2 ) / 2)
  (h4 : s = (a + b + c) / 2) :
  f1 + f2 + s3 ≤ sqrt 3 * s := 
sorry

end angle_bisector_median_inequality_l260_260555


namespace coordinates_with_respect_to_origin_l260_260951

theorem coordinates_with_respect_to_origin :
  ∀ x y : ℝ, (x = 1 ∧ y = -2) → (x, y) = (1, -2) :=
by
  intros x y h
  cases h with hx hy
  simp [hx, hy]
  sorry

end coordinates_with_respect_to_origin_l260_260951


namespace m_value_for_fractional_equation_l260_260965

theorem m_value_for_fractional_equation (m : ℤ) :
  (∃ x : ℤ, x ≠ -2 ∧ (x - 5) / (x + 2) = m / (x + 2)) → m = -7 :=
by
  intro h
  cases h with x hx
  sorry

end m_value_for_fractional_equation_l260_260965


namespace james_coins_value_l260_260173

theorem james_coins_value (p n : ℕ) (h1 : p + n = 15) (h2 : p = n + 2) : 5 * n + p = 38 :=
by
  -- We start with the given equations
  have h3 : p = n + 2 := h2
  have h4 : (n + 2) + n = 15 := by rw [h3, h1]
  have h5 : 2 * n + 2 = 15 := by linarith
  have h6 : 2 * n = 13 := by linarith
  have h7 : n = 6 := by linarith
  have h8 : p = 8 := by rw [h3, h7]
  -- Now we calculate the total value of the coins
  have value := 5 * n + p
  rw [h7, h8]
  exact rfl


end james_coins_value_l260_260173


namespace number_of_zeros_of_h_l260_260053

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 3 - x^2
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem number_of_zeros_of_h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 = 0 ∧ h x2 = 0 ∧ ∀ x, h x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_zeros_of_h_l260_260053


namespace sum_of_values_sum_of_x_values_l260_260774

theorem sum_of_values (x : ℝ) :
  (2 ^ (x^2 - 3*x - 2) = 4 ^ (x - 4)) → (x = 2 ∨ x = 3) :=
sorry

theorem sum_of_x_values :
  (∑ x in ({2, 3} : Finset ℝ), x = 5) :=
sorry

end sum_of_values_sum_of_x_values_l260_260774


namespace top_card_is_5_or_king_l260_260717

-- Define the number of cards in a deck
def total_cards : ℕ := 52

-- Define the number of 5s in a deck
def number_of_5s : ℕ := 4

-- Define the number of Kings in a deck
def number_of_kings : ℕ := 4

-- Define the number of favorable outcomes (cards that are either 5 or King)
def favorable_outcomes : ℕ := number_of_5s + number_of_kings

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_cards

-- Theorem: The probability that the top card is either a 5 or a King is 2/13
theorem top_card_is_5_or_king (h_total_cards : total_cards = 52)
    (h_number_of_5s : number_of_5s = 4)
    (h_number_of_kings : number_of_kings = 4) :
    probability = 2 / 13 := by
  -- Proof would go here
  sorry

end top_card_is_5_or_king_l260_260717


namespace minimum_value_of_expression_l260_260907

noncomputable def min_value_expr (x y z : ℝ) : ℝ := (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1)

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) :
  min_value_expr x y z = 24 * Real.sqrt 2 :=
sorry

end minimum_value_of_expression_l260_260907


namespace non_congruent_triangles_perimeter_12_l260_260116

theorem non_congruent_triangles_perimeter_12 :
  ∃ S : finset (ℕ × ℕ × ℕ), S.card = 5 ∧ ∀ (abc ∈ S), 
  let (a, b, c) := abc in 
    a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    ∀ (abc' ∈ S), abc' ≠ abc → abc ≠ (λ t, (t.2.2, t.2.1, t.1)) abc' :=
by sorry

end non_congruent_triangles_perimeter_12_l260_260116


namespace quadratic_inequality_for_all_x_l260_260863

theorem quadratic_inequality_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 + a) * x^2 - a * x + 1 > 0) ↔ (-4 / 3 < a ∧ a < -1) ∨ a = 0 :=
sorry

end quadratic_inequality_for_all_x_l260_260863


namespace greater_expected_area_l260_260734

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l260_260734


namespace probability_knows_cpp_l260_260878

theorem probability_knows_cpp :
  (∀ (total_employees : ℕ) (p_cpp : ℚ),
    total_employees = 600 →
    p_cpp = 3 / 10 →
    (180 / total_employees) = 0.3) :=
by
  intros total_employees p_cpp htotal hcpp
  rw [htotal, hcpp]
  simp
  norm_num
  sorry

end probability_knows_cpp_l260_260878


namespace number_of_non_congruent_triangles_l260_260112

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_congruent_triangles_with_perimeter_12 : ℕ :=
  { (a, b, c) | a ≤ b ∧ b ≤ c ∧ a + b + c = 12 ∧ is_triangle a b c }.to_finset.card

theorem number_of_non_congruent_triangles : non_congruent_triangles_with_perimeter_12 = 2 := sorry

end number_of_non_congruent_triangles_l260_260112


namespace least_positive_three_digit_multiple_of_8_l260_260295

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l260_260295


namespace diameter_of_circle_l260_260609

noncomputable def area : ℝ := 63.61725123519331
noncomputable def pi_approx : ℝ := 3.14159

theorem diameter_of_circle (A : ℝ) (pi : ℝ) (h : A = 63.61725123519331) (hpi : pi = pi_approx) :
  ∃ d : ℝ, d ≈ 9 :=
by
  let r := real.sqrt (A / pi)
  let d := 2 * r
  have : d ≈ 9 := sorry
  exact ⟨d, this⟩

end diameter_of_circle_l260_260609


namespace ratio_twitter_to_combined_l260_260209

noncomputable def instagram_followers := 240
noncomputable def facebook_followers := 500
noncomputable def twitter_followers := (3840 - 240 - 500 - (3 * twitter_followers) - (3 * twitter_followers + 510)) / 7
noncomputable def tikTok_followers := 3 * twitter_followers
noncomputable def youTube_followers := tikTok_followers + 510
noncomputable def combined_followers := instagram_followers + facebook_followers

theorem ratio_twitter_to_combined : twitter_followers / combined_followers = 1 / 2 :=
by
  sorry

end ratio_twitter_to_combined_l260_260209


namespace greater_expected_area_l260_260732

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l260_260732


namespace count_angles_same_terminal_side_l260_260161
noncomputable def same_terminal_side_count : ℤ :=
  let angles := {β : ℝ | ∃ k : ℤ, β = (-5 * Real.pi / 4) + (2 * k * Real.pi)}
  let interval := Set.Ioo (-Real.pi) (4 * Real.pi)
  Set.card {β | β ∈ angles ∧ β ∈ interval}

theorem count_angles_same_terminal_side :
  same_terminal_side_count = 2 :=
  sorry

end count_angles_same_terminal_side_l260_260161


namespace ineq_prove_l260_260180

theorem ineq_prove 
  (a b c d e : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_product : a * b * c * d * e = 1) :
  (d * e / (a * (b + 1)) + e * a / (b * (c + 1)) + a * b / (c * (d + 1)) + b * c / (d * (e + 1)) + c * d / (e * (a + 1)) ≥ 5 / 2) :=
begin
  sorry
end

end ineq_prove_l260_260180


namespace mod_equiv_1043_36_mod_equiv_1_10_l260_260397

open Int

-- Define the integers involved
def a : ℤ := -1043
def m1 : ℕ := 36
def m2 : ℕ := 10

-- Theorems to prove modulo equivalence
theorem mod_equiv_1043_36 : a % m1 = 1 := by
  sorry

theorem mod_equiv_1_10 : 1 % m2 = 1 := by
  sorry

end mod_equiv_1043_36_mod_equiv_1_10_l260_260397


namespace k_inverse_k_inv_is_inverse_l260_260545

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x - 4
def k (x : ℝ) : ℝ := f (g x)

def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem k_inverse {x : ℝ} : k_inv (k x) = x :=
by
  sorry

theorem k_inv_is_inverse {x y : ℝ} : k_inv (y) = x ↔ y = k(x) :=
by
  sorry

end k_inverse_k_inv_is_inverse_l260_260545


namespace trajectory_eq_line_eq_l260_260441

-- Define the problem conditions
noncomputable def M (x y : ℝ) : Prop := true
def M1 : ℝ × ℝ := (26, 1)
def M2 : ℝ × ℝ := (2, 1)
def ratio := 5

-- Define the problem 1: Find the trajectory equation
theorem trajectory_eq (x y : ℝ) (h : sqrt ((x - 26)^2 + (y - 1)^2) / sqrt ((x - 2)^2 + (y - 1)^2) = 5) :
  (x - 1)^2 + (y - 1)^2 = 25 :=
  sorry

-- Define the problem 2: Equation of the line
theorem line_eq (x y : ℝ) (h1 : (x - 1)^2 + (y - 1)^2 = 25)
  (hx : x = -2 ∨ (5 * x - 12 * y + 46 = 0)) :
  ∃ (l : ℝ → Prop), 
    (∀ (x y : ℝ), l(x y) → segment_length l M(-2, 3) = 8) :=
  sorry

end trajectory_eq_line_eq_l260_260441


namespace neg_or_implication_l260_260137

theorem neg_or_implication {p q : Prop} : ¬(p ∨ q) → (¬p ∧ ¬q) :=
by
  intros h
  sorry

end neg_or_implication_l260_260137


namespace angle_of_inclination_l260_260762

-- Define the angle of inclination problem in Lean 4
theorem angle_of_inclination (a : ℝ) : 
  ∃ α : ℝ, 0 ≤ α ∧ α < real.pi ∧ real.arctan (-real.sqrt 3 / 3) = α ∧ α = 5 * real.pi / 6 := 
sorry

end angle_of_inclination_l260_260762


namespace greater_expected_area_vasya_l260_260729

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l260_260729


namespace divide_19_degree_angle_into_19_parts_l260_260764

noncomputable def divide_angle (θ : ℝ) (n : ℕ) : Prop :=
  exists (draw : ℕ → ℝ), (∀ i, 0 ≤ draw i ∧ draw i < θ) ∧
    ∑ i in range n, (draw i) = θ / n

theorem divide_19_degree_angle_into_19_parts :
  divide_angle 19 19 :=
sorry

end divide_19_degree_angle_into_19_parts_l260_260764


namespace find_years_of_compound_interest_l260_260771

noncomputable def compound_interest_years (P : ℝ) (r : ℝ) (CI : ℝ) (A : ℝ) : ℕ :=
  let n : ℕ := 1
  let t : ℝ := t in
  if A = P * (1 + r / n)^(n * t) then t.toInt else 0

theorem find_years_of_compound_interest :
  let P : ℝ := 1200
  let r : ℝ := 0.2
  let CI : ℝ := 240
  let A : ℝ := P + CI
  compound_interest_years P r CI A = 1 :=
by
  sorry

end find_years_of_compound_interest_l260_260771


namespace distance_after_shift_is_13_l260_260412

-- Define the points before the origin shift
def point1_before : ℝ × ℝ := (-2, -6)
def point2_before : ℝ × ℝ := (10, -1)

-- Define the shift
def shift : ℝ × ℝ := (3, 4)

-- Define the points after the origin shift
def point1_after : ℝ × ℝ := (point1_before.1 - shift.1, point1_before.2 - shift.2)
def point2_after : ℝ × ℝ := (point2_before.1 - shift.1, point2_before.2 - shift.2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove the distance between the adjusted points is 13
theorem distance_after_shift_is_13 : distance point1_after point2_after = 13 :=
by
  sorry

end distance_after_shift_is_13_l260_260412


namespace gas_cost_shared_among_friends_l260_260029

theorem gas_cost_shared_among_friends :
  ∀ (x : ℝ), 
    (∀ (initial_friends new_friends : ℕ),
      initial_friends = 5 ∧ new_friends = 2 ∧ 
      (x / initial_friends - 15 = x / (initial_friends + new_friends)) → 
      x = 262.50
    ) :=
begin
  intros x initial_friends new_friends h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h3 with h5 h6,
  subst h5, subst h6,
  have h4 := h4,
  sorry
end

end gas_cost_shared_among_friends_l260_260029


namespace mass_percentage_Ba_in_BaI2_l260_260020

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 : 
  (molar_mass_Ba / molar_mass_BaI2) * 100 = 35.11 := 
  by 
    -- implementing the proof here would demonstrate that (137.33 / 391.13) * 100 = 35.11
    sorry

end mass_percentage_Ba_in_BaI2_l260_260020


namespace range_of_m_l260_260072

open Set

theorem range_of_m {m : ℝ} :
  (let A := {x | 0 < x ∧ x < 4}
   in let B := {2, m}
   in (A ∩ B).to_finset.powerset.card = 4) ↔ (0 < m ∧ m < 4 ∧ m ≠ 2) :=
by
  intros
  sorry

end range_of_m_l260_260072


namespace parabola_points_relation_l260_260581

theorem parabola_points_relation (c y1 y2 y3 : ℝ)
  (h1 : y1 = -(-2)^2 - 2*(-2) + c)
  (h2 : y2 = -(0)^2 - 2*(0) + c)
  (h3 : y3 = -(1)^2 - 2*(1) + c) :
  y1 = y2 ∧ y2 > y3 :=
by
  sorry

end parabola_points_relation_l260_260581


namespace check_construction_l260_260601

noncomputable def construction_area_error (r : ℝ) : ℝ :=
  let α_desired := 54 * (pi / 180)
  let α_actual := atan (3 / 2)
  let angle_discrepancy := α_actual - α_desired
  angle_discrepancy * (r^2 * pi / (2 * pi))

theorem check_construction (r : ℝ) :
  construction_area_error r = 0.0064 * r^2 * pi :=
sorry

end check_construction_l260_260601


namespace train_speed_l260_260356

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) (h_length : length = 975) (h_time : time = 48) (h_speed : speed = length / time * 3.6) : 
  speed = 73.125 := 
by 
  sorry

end train_speed_l260_260356


namespace roots_quadratic_eq_a2_b2_l260_260316

theorem roots_quadratic_eq_a2_b2 (a b : ℝ) (h1 : a^2 - 5 * a + 5 = 0) (h2 : b^2 - 5 * b + 5 = 0) : a^2 + b^2 = 15 :=
by
  sorry

end roots_quadratic_eq_a2_b2_l260_260316


namespace problem_statement_l260_260806

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0}) :
  a^2017 + b^2017 = -1 := by
  sorry

end problem_statement_l260_260806


namespace sum_k_to_n_eq_l260_260283

theorem sum_k_to_n_eq (n : ℕ) : 
  (∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k * Nat.choose n k / ((k + 2) * (k + 3) * (k + 4))) = 
  1 / (2 * ((n + 3) * (n + 4))) :=
sorry

end sum_k_to_n_eq_l260_260283


namespace lambda_value_l260_260784

-- Given conditions: vector a and vector b
def a : ℝ × ℝ := (3, 1)
def b (λ : ℝ) : ℝ × ℝ := (2, λ)

-- Definition of parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Proof statement
theorem lambda_value (λ : ℝ) (h : parallel a (b λ)) : λ = 2 / 3 := by
  sorry

end lambda_value_l260_260784


namespace percentage_error_in_square_area_l260_260725

theorem percentage_error_in_square_area (s : ℝ) (h : s > 0) : 
  let measured_side := 1.05 * s,
      actual_area := s^2,
      calculated_area := (measured_side)^2,
      error_area := calculated_area - actual_area,
      percentage_error := (error_area / actual_area) * 100
  in percentage_error = 10.25 :=
by
  sorry

end percentage_error_in_square_area_l260_260725


namespace find_larger_number_l260_260610

-- Define the conditions
variables (L S : ℕ)
axiom condition1 : L - S = 1365
axiom condition2 : L = 6 * S + 35

-- State the theorem
theorem find_larger_number : L = 1631 :=
by
  sorry

end find_larger_number_l260_260610


namespace robin_total_distance_l260_260961

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end robin_total_distance_l260_260961


namespace find_fx_l260_260861

theorem find_fx (f : ℕ → ℕ) (x : ℕ) (h : f(x + 1) = x^2 - 5) : f(x) = x^2 - 2x - 4 := 
by sorry

end find_fx_l260_260861


namespace num_odd_digits_base4_of_345_l260_260416

/-- The number of odd digits in the base-4 representation of 345₁₀ is 4. -/
theorem num_odd_digits_base4_of_345 : 
  let base4_repr := Nat.digits 4 345 in
  (base4_repr.filter (λ d, d % 2 = 1)).length = 4 := by
  sorry

end num_odd_digits_base4_of_345_l260_260416


namespace tom_profit_l260_260640

-- Define the initial conditions
def initial_investment : ℕ := 20 * 3
def revenue_from_selling : ℕ := 10 * 4
def value_of_remaining_shares : ℕ := 10 * 6
def total_amount : ℕ := revenue_from_selling + value_of_remaining_shares

-- We claim that the profit Tom makes is 40 dollars
theorem tom_profit : (total_amount - initial_investment) = 40 := by
  sorry

end tom_profit_l260_260640


namespace second_class_students_count_l260_260247

theorem second_class_students_count 
    (x : ℕ)
    (h1 : 12 * 40 = 480)
    (h2 : ∀ x, x * 60 = 60 * x)
    (h3 : (12 + x) * 54 = 480 + 60 * x) : 
    x = 28 :=
by
  sorry

end second_class_students_count_l260_260247


namespace solution_exists_l260_260154

theorem solution_exists (grid : Fin 8 → Fin 8 → Bool)
  (h_black_squares : (∑ i j, if grid i j then 1 else 0) = 12) :
  ∃ (rows : Finset (Fin 8)) (cols : Finset (Fin 8)), 
  rows.card = 4 ∧ cols.card = 4 ∧
  ∀ i j, grid i j → i ∈ rows ∧ j ∈ cols :=
begin
  sorry
end

end solution_exists_l260_260154


namespace coefficient_of_x_in_first_exponent_l260_260849

theorem coefficient_of_x_in_first_exponent (a : ℝ) (x : ℝ) (h : 4^(a * x + 2) = 16^(3 * x - 1)) (hx : x = 1) : a = 2 :=
by
  sorry

end coefficient_of_x_in_first_exponent_l260_260849


namespace correct_statements_for_sequence_l260_260781

theorem correct_statements_for_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  -- Statement 1
  (S_n = n^2 + n → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 2
  (S_n = 2^n - 1 → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1)) ∧
  -- Statement 3
  (∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1) → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 4
  (¬(∀ n, n ≥ 2 → a n^2 = a (n + 1) * a (n - 1) → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1))) :=
sorry

end correct_statements_for_sequence_l260_260781


namespace pairwise_partitions_l260_260271

-- Conditions: there are 8 people equally spaced around a circle,
-- and each person knows exactly 4 others: the two adjacent ones and 
-- the two who are two spaces away.
def know_relation (i j : ℕ) : Prop := 
  (i ≡ (j + 1) % 8) ∨ (i ≡ (j - 1) % 8) ∨ (i ≡ (j + 2) % 8) ∨ (i ≡ (j - 2) % 8)

-- Proof problem: Prove there are exactly 10 ways to split the 8 people 
-- into 4 pairs such that the members of each pair know each other.
theorem pairwise_partitions : 
  (number_of_valid_pairings know_relation 8 4) = 10 :=
sorry

end pairwise_partitions_l260_260271


namespace tan_of_angle_subtraction_l260_260049

theorem tan_of_angle_subtraction (a : ℝ) (h : Real.tan (a + Real.pi / 4) = 1 / 7) : Real.tan a = -3 / 4 :=
by
  sorry

end tan_of_angle_subtraction_l260_260049


namespace minute_hand_length_l260_260218

theorem minute_hand_length 
  (arc_length : ℝ) (r : ℝ) (h : arc_length = 20 * (2 * Real.pi / 60) * r) :
  r = 1/2 :=
  sorry

end minute_hand_length_l260_260218


namespace sum_of_interior_angles_l260_260701

theorem sum_of_interior_angles (n : ℕ) (hn : n ≥ 6) :
  let S := 180 * (n - 2)
  in Σ (interior_angle : ℝ), True = (interior_angle == (180 * (n - 2))) := 
by
  sorry

end sum_of_interior_angles_l260_260701


namespace check_statements_l260_260228

def point_3D := (ℝ, ℝ, ℝ)

def P : point_3D := (1, 2, 3)

def distance (p1 p2 : point_3D) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def midpoint (p1 p2 : point_3D) : point_3D :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def symmetric_about_x_axis (p : point_3D) : point_3D :=
  (-p.1, p.2, p.3)

def symmetric_about_origin (p : point_3D) : point_3D :=
  (-p.1, -p.2, -p.3)

def symmetric_about_xy_plane (p : point_3D) : point_3D :=
  (p.1, p.2, -p.3)

theorem check_statements :
  let d := distance P (0,0,0) in
  let mp := midpoint P (0,0,0) in
  let sx := symmetric_about_x_axis P in
  let so := symmetric_about_origin P in
  let sp := symmetric_about_xy_plane P in
  (d ≠ real.sqrt 13) ∧
  (mp = (0.5, 1, 1.5)) ∧
  (sx = (-1, 2, 3)) ∧
  (so = (-1, -2, -3)) ∧
  (sp = (1, 2, -3)) →
  2 = 2 :=
by
  intro h
  exact rfl

end check_statements_l260_260228


namespace digit_ends_with_l260_260393

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l260_260393


namespace sequence_mod_l260_260383

def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ ∀ n ≥ 1, a n = ∑ k in Finset.range n, Nat.choose n k * a k

theorem sequence_mod (a : ℕ → ℕ) (p m q r : ℕ) [Fact p.prime] (h_seq : sequence a) (hm : m > 0) :
  a (p ^ m * q + r) % p ^ m = a (p ^ (m - 1) * q + r) % p ^ m :=
by
  sorry

end sequence_mod_l260_260383


namespace minimize_max_value_l260_260966

theorem minimize_max_value (A B : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ (3 / 2) * π → F x = |cos x ^ 2 + 2 * sin x * cos x - sin x ^ 2 + A * x + B|) →
  ∃ (A B : ℝ), A = 0 ∧ B = 0 ∧ (∀ x, 0 ≤ x ∧ x ≤ (3 / 2) * π → |cos x ^ 2 + 2 * sin x * cos x - sin x ^ 2 + A * x + B| ≤ √2) :=
begin
  sorry
end

end minimize_max_value_l260_260966


namespace sum_of_roots_l260_260913

variable {f : ℝ → ℝ}

open Real

def is_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(3 + x) = f(3 - x)

def has_exact_six_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ s : Finset ℝ, s.card = 6 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x y ∈ s, x ≠ y → x ≠ y

theorem sum_of_roots (hf_symm : is_symmetric f)
    (hf_roots : has_exact_six_distinct_real_roots f) : ∑ x in (hf_roots.some), x = 18 :=
  sorry

end sum_of_roots_l260_260913


namespace sqrt_inequality_l260_260584

open Real

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z) 
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  sqrt (x + y + z) ≥ sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) :=
sorry

end sqrt_inequality_l260_260584


namespace vasya_has_greater_expected_area_l260_260739

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l260_260739


namespace trains_crossing_time_l260_260318

noncomputable def time_to_cross_opposite_direction
  (length : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  let S1 := length / time1
  let S2 := length / time2
  let Sr := S1 + S2
  let D := 2 * length
  D / Sr

theorem trains_crossing_time :
  time_to_cross_opposite_direction 120 10 18 ≈ 12.85 :=
by
  sorry

end trains_crossing_time_l260_260318


namespace volume_difference_l260_260425

theorem volume_difference (x1 x2 x3 Vmin Vmax : ℝ)
  (hx1 : 0.5 < x1 ∧ x1 < 1.5)
  (hx2 : 0.5 < x2 ∧ x2 < 1.5)
  (hx3 : 2016.5 < x3 ∧ x3 < 2017.5)
  (rV : 2017 = Nat.floor (x1 * x2 * x3))
  : abs (Vmax - Vmin) = 4035 := 
sorry

end volume_difference_l260_260425


namespace number_of_non_congruent_triangles_l260_260111

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_congruent_triangles_with_perimeter_12 : ℕ :=
  { (a, b, c) | a ≤ b ∧ b ≤ c ∧ a + b + c = 12 ∧ is_triangle a b c }.to_finset.card

theorem number_of_non_congruent_triangles : non_congruent_triangles_with_perimeter_12 = 2 := sorry

end number_of_non_congruent_triangles_l260_260111


namespace pondFishEstimate_l260_260916

noncomputable def estimateTotalFish (initialFishMarked : ℕ) (caughtFishTenDaysLater : ℕ) (markedFishCaught : ℕ) : ℕ :=
  initialFishMarked * caughtFishTenDaysLater / markedFishCaught

theorem pondFishEstimate
    (initialFishMarked : ℕ)
    (caughtFishTenDaysLater : ℕ)
    (markedFishCaught : ℕ)
    (h1 : initialFishMarked = 30)
    (h2 : caughtFishTenDaysLater = 50)
    (h3 : markedFishCaught = 2) :
    estimateTotalFish initialFishMarked caughtFishTenDaysLater markedFishCaught = 750 := by
  sorry

end pondFishEstimate_l260_260916


namespace gcd_2835_9150_l260_260652

theorem gcd_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end gcd_2835_9150_l260_260652


namespace bounded_polyhedron_volume_l260_260875

noncomputable theory

variables {a : ℝ}

def is_right_prism
  (A B C A1 B1 C1 M N K L : Point) 
  (prism_edge : ℝ) : Prop := 
  is_regular_prism A B C A1 B1 C1 ∧ side_length A B = prism_edge ∧ height A A1 = prism_edge

theorem bounded_polyhedron_volume (A B C A1 B1 C1 : Point) (a : ℝ):
  is_right_prism A B C A1 B1 C1 a →
  prism_bound_polyhedron_volume A B C A1 B1 C1 = (9*a^3*real.sqrt 3) / 4 :=
begin
  sorry
end

end bounded_polyhedron_volume_l260_260875


namespace find_distance_l260_260341

-- Defining the conditions
def man_rowing_speed_in_still_water : ℝ := 8
def current_velocity : ℝ := 2
def round_trip_time : ℝ := 2
def wind_resistance_factor : ℝ := 0.1

-- Effective speed in still water considering wind resistance
def effective_speed_in_still_water : ℝ := man_rowing_speed_in_still_water * (1 - wind_resistance_factor)

-- Effective speeds downstream and upstream
def downstream_speed : ℝ := effective_speed_in_still_water + current_velocity
def upstream_speed : ℝ := effective_speed_in_still_water - current_velocity

-- Times taken for downstream and upstream rowing
def t1 (D : ℝ) : ℝ := D / downstream_speed
def t2 (D : ℝ) : ℝ := D / upstream_speed

-- Total round trip time equation
theorem find_distance (D : ℝ) :
  t1 D + t2 D = round_trip_time → D = 6.65 :=
by {
  sorry
}

end find_distance_l260_260341


namespace reflection_segment_length_l260_260099

/-- Given a point A at coordinates (-3, -2, 4), B is the reflection of A about the origin,
and C is the reflection of A about the yOz plane. Prove that the length of segment BC is 4√5. -/
theorem reflection_segment_length :
  let A := (-3, -2, 4 : ℝ × ℝ × ℝ)
  let B := (3, 2, -4 : ℝ × ℝ × ℝ)
  let C := (3, -2, 4 : ℝ × ℝ × ℝ)
  dist B C = 4 * Real.sqrt 5 := by
  sorry

end reflection_segment_length_l260_260099


namespace part1_part2_l260_260468

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 :
  (∀ x, f x ≤ 9) → ∀ x, x ∈ (Icc (-2 : ℝ) (4 : ℝ)) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) ↔ a ∈ Icc (19 / 4 : ℝ) (7 : ℝ) :=
by
  sorry

end part1_part2_l260_260468


namespace find_f_2017_l260_260860

noncomputable def f : ℝ → ℝ := sorry

axiom function_condition (a b : ℝ) : 3 * f((a + 2 * b) / 3) = f(a) + 2 * f(b)

axiom f_one : f(1) = 1
axiom f_four : f(4) = 7

theorem find_f_2017 : f(2017) = 4033 := sorry

end find_f_2017_l260_260860


namespace seating_arrangements_l260_260284

-- Definitions of conditions based on problem statement
def total_chairs : ℕ := 12
def total_couples : ℕ := 6

-- Conditions about seating
def alternating_seating : Prop := ∀ (n : ℕ) (hn : n < total_chairs), 
  (n % 2 = 0 → ∃ m : ℕ, m < total_couples ∧ chair n belongs to man m) ∧ 
  (n % 2 = 1 → ∃ w : ℕ, w < total_couples ∧ chair n belongs to woman w)

def not_next_to_or_across_spouse : Prop := ∀ (n m : ℕ) (hn : n < total_couples) (hm : m < total_couples),
  spouse_of n ≠ m ∧ 
  ¬(chair_of_spouse m = (chair_of_n + 1) % total_chairs ∨ chair_of_spouse m = (chair_of_n - 1 + total_chairs) % total_chairs ∨ 
    chair_of_spouse m = (chair_of_n + total_chairs / 2) % total_chairs)

def not_next_to_same_profession : Prop := ∀ (n : ℕ) (hn : n < total_chairs),
  (n % 2 = 0 → ¬(∃ m : ℕ, m < total_couples ∧ (chair n + 1) % total_chairs = chair_of_man m ∨ 
    (chair n - 1 + total_chairs) % total_chairs = chair_of_man m)) ∧ 
  (n % 2 = 1 → ¬(∃ w : ℕ, w < total_couples ∧ (chair n + 1) % total_chairs = chair_of_woman w ∨ 
    (chair n - 1 + total_chairs) % total_chairs = chair_of_woman w))

-- Proof statement
theorem seating_arrangements : alternating_seating ∧ not_next_to_or_across_spouse ∧ not_next_to_same_profession →
  ∃ n : ℕ, n = 2880 :=
sorry

end seating_arrangements_l260_260284


namespace propositions_truth_count_l260_260807

theorem propositions_truth_count (a b c : ℝ) : 
  (∃ n : ℕ, n = 2 ∧ 
    (
      ((a > b) → (ac^2 > bc^2)) ∧ 
      ((ac^2 > bc^2) → (a > b)) ∧ 
      (¬(a > b) → ¬(ac^2 > bc^2)) ∧ 
      (¬(ac^2 > bc^2) → ¬(a > b))
    )
  )
:= sorry

end propositions_truth_count_l260_260807


namespace obtuse_triangles_in_17_gon_l260_260129

noncomputable def number_of_obtuse_triangles (n : ℕ): ℕ := 
  if h : n ≥ 3 then (n * (n - 1) * (n - 2)) / 6 else 0

theorem obtuse_triangles_in_17_gon : number_of_obtuse_triangles 17 = 476 := sorry

end obtuse_triangles_in_17_gon_l260_260129


namespace train_time_to_pass_platform_l260_260720

noncomputable def train_length : ℝ := 360
noncomputable def platform_length : ℝ := 140
noncomputable def train_speed_km_per_hr : ℝ := 45

noncomputable def train_speed_m_per_s : ℝ :=
  train_speed_km_per_hr * (1000 / 3600)

noncomputable def total_distance : ℝ :=
  train_length + platform_length

theorem train_time_to_pass_platform :
  (total_distance / train_speed_m_per_s) = 40 := by
  sorry

end train_time_to_pass_platform_l260_260720


namespace greater_expected_area_vasya_l260_260726

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l260_260726


namespace complex_modulus_calc_l260_260089

noncomputable def z (i : ℂ) := (2 * i) / (1 - i)

theorem complex_modulus_calc (z : ℂ) (i : ℂ) (hi : i^2 = -1) (h : z * (1 - i) = 2 * i) : 
  |z| = Real.sqrt 2 :=
by
  sorry

end complex_modulus_calc_l260_260089


namespace binomial_square_a_value_l260_260763

theorem binomial_square_a_value (a : ℚ) :
  (∀ x : ℚ, (9 * x^2 + 15 * x + a) = (3 * x + 5 / 2) ^ 2) → a = 25 / 4 :=
by
  intro h
  have h1 := h 0
  rw [pow_two, add_assoc, mul_assoc, mul_comm] at h1
  simp at h1
  sorry

end binomial_square_a_value_l260_260763


namespace coordinates_with_respect_to_origin_l260_260952

theorem coordinates_with_respect_to_origin :
  ∀ x y : ℝ, (x = 1 ∧ y = -2) → (x, y) = (1, -2) :=
by
  intros x y h
  cases h with hx hy
  simp [hx, hy]
  sorry

end coordinates_with_respect_to_origin_l260_260952


namespace range_of_a_l260_260823

theorem range_of_a (a : ℝ) :
  (∀ x, (3 ≤ x → 2*a*x + 4 ≤ 2*a*(x+1) + 4) ∧ (2 < x ∧ x < 3 → (a + (2*a + 2)/(x-2) ≤ a + (2*a + 2)/(x-1))) ) →
  -1 < a ∧ a ≤ -2/3 :=
by
  intros h
  sorry

end range_of_a_l260_260823


namespace exterior_angle_FGH_l260_260229

noncomputable def sum_of_interior_angles (n: ℕ) : ℕ :=
  180 * (n - 2)

noncomputable def regular_polygon_interior_angle (n: ℕ) : ℕ :=
  sum_of_interior_angles n / n

theorem exterior_angle_FGH :
  let hex_interior_angle := regular_polygon_interior_angle 6,
      oct_interior_angle := regular_polygon_interior_angle 8,
      FAG_exterior_angle := 360 - hex_interior_angle - oct_interior_angle
  in FAG_exterior_angle = 105 := 
by
  sorry

end exterior_angle_FGH_l260_260229


namespace mark_more_than_kate_by_100_l260_260222

variable (Pat Kate Mark : ℕ)
axiom total_hours : Pat + Kate + Mark = 180
axiom pat_twice_as_kate : Pat = 2 * Kate
axiom pat_third_of_mark : Pat = Mark / 3

theorem mark_more_than_kate_by_100 : Mark - Kate = 100 :=
by
  sorry

end mark_more_than_kate_by_100_l260_260222


namespace solve_for_x_l260_260236

theorem solve_for_x :
  ∀ (x : ℝ), (4 * x - 5) / (5 * x - 10) = 3 / 4 → x = -10 :=
by
  intros x h
  have h1 : 4 * (4 * x - 5) = 3 * (5 * x - 10) := by
    rw [eq_div_iff (by norm_num : (5 : ℝ) * x - 10 ≠ 0)] at h
    exact (mul_eq_mul_right_iff.mp h).2
  norm_num at h1
  linarith

end solve_for_x_l260_260236


namespace triangle_count_l260_260635

theorem triangle_count (points : Finset Point) (h_points : points.card = 7)
  (collinear_points : Finset Point) (h_collinear : collinear_points.card = 4)
  (h_no_other_line : ∀ (l : Line) (h_l : ∀ p ∈ collinear_points, p ∈ l), collinear_points = {p ∈ points | p ∈ l }) :
  ∀ (triangle_points : Finset Point), 
    triangle_points.card = 3 → triangle_points ⊆ points → 
    (∀ (l : Line), ¬ ∀ p ∈ triangle_points, p ∈ l) → 
    (Finset.card (points.'choose 3)) - (Finset.card (collinear_points.'choose 3)) = 31 := 
by
  sorry

end triangle_count_l260_260635


namespace area_sum_of_parallelograms_l260_260379

-- Definitions of the geometrical elements and the problem conditions
variable {Point : Type}
variable [AffineSpace Point ℝ]

-- Assume some basic geometrical objects and conditions
variables (A B C D E F G H I J : Point)
variables {ℓ₁ ℓ₂ : Line Point} (parallelogram_ABDE parallelogram_BCFG parallelogram_CAIJ : set Point)

-- Definitions for the parallelogram constructions and properties
def is_triangle (A B C : Point) : Prop := 
abc_triangle

def is_parallelogram (ℓ₁ ℓ₂ : Line Point) (P1 P2 P3 P4 : Point) : Prop :=
parallel_and_equal_length ℓ₁ ℓ₂ P1 P2 ∧ 
parallel_and_equal_length ℓ₁ ℓ₂ P3 P4 ∧ 
P1 ≠ P2 ∧ P3 ≠ P4

-- Define the areas based on given geometric objects
def area (A B C D : Point) : ℝ := parallelogram_area_formula A B C D

-- Main theorem statement
theorem area_sum_of_parallelograms 
  (h_triangle : is_triangle A B C)
  (h_parallelogram_ABDE : is_parallelogram ℓ₁ ℓ₂ A B D E)
  (h_parallelogram_BCFG : is_parallelogram ℓ₁ ℓ₂ B C F G)
  (h_intersection : intersect ℓ₁ linE linF = some H)
  (h_parallelogram_CAIJ : is_parallelogram ℓ₂ ℓ₃ C A I J)
  (h_parallel_equal_AIBH : parallel_and_equal_length ℓ₂ ℓ₃ A I B H) :
  area C A I J = area A B D E + area B C F G := sorry

end area_sum_of_parallelograms_l260_260379


namespace rational_unique_index_l260_260627

open Rational

noncomputable def x_seq : ℕ → ℚ
| 1       := 1
| (2*n)   := 1 + x_seq n
| (2*n+1) := 1 / x_seq (2*n)

theorem rational_unique_index (r : ℚ) (h_pos : 0 < r) :
  ∃! n : ℕ, r = x_seq n :=
by
  sorry

end rational_unique_index_l260_260627


namespace find_a_l260_260083

-- Definitions and conditions
variables {z a : ℂ}

-- Pure imaginary number definition
def isPureImaginary (z : ℂ) : Prop := z.re = 0

-- The main theorem to prove
theorem find_a (h : (1 - complex.i) * z = 1 + a * complex.i) (hz : isPureImaginary z) : a = 1 :=
by
  sorry

end find_a_l260_260083


namespace domain_of_g_l260_260003

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

theorem domain_of_g : {x : ℝ | g x ∈ ℝ} = set.Icc (-1 : ℝ) (1 : ℝ) \ {0 : ℝ} := by
  sorry

end domain_of_g_l260_260003


namespace find_triples_l260_260975

variable (x y z : ℝ) (a b c : ℝ)

-- Definitions according to the problem
def eq1 := 4 * x^2 - 2 * x - 30 * y * z = 25 * y^2 + 5 * y + 12 * x * z
def eq2 := 25 * y^2 + 5 * y + 12 * x * z = 9 * z^2 - 3 * z - 20 * x * y

def def_a := a = 2 * x + 5 * y
def def_b := b = 3 * z + 5 * y
def def_c := c = 3 * z - 2 * x

theorem find_triples 
  (h₁ : eq1)
  (h₂ : eq2)
  (h₃ : def_a)
  (h₄ : def_b)
  (h₅ : def_c) :
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = 1) ∨
  (a = 1 ∧ b = 0 ∧ c = -1) ∨
  (a = -1 ∧ b = -1 ∧ c = 0) :=
sorry

end find_triples_l260_260975


namespace photos_ratio_l260_260220

theorem photos_ratio (L R C : ℕ) (h1 : R = L) (h2 : C = 12) (h3 : R = C + 24) :
  L / C = 3 :=
by 
  sorry

end photos_ratio_l260_260220


namespace ginger_children_cakes_l260_260427

theorem ginger_children_cakes (C : ℕ) (h₁ : ∀ n : ℕ, n = 10) 
  (h₂ : ∀ c : ℕ, c = 4 * C) 
  (h₃ : ∀ h : ℕ, h = 6) 
  (h₄ : ∀ p : ℕ, p = 2)
  (cakes_per_year : ∀ t : ℕ, t = C * 4 + 6 + 2) 
  : C = 2 :=
begin
  have annual_cakes : ∀ y : ℕ, y = 16 := by sorry,
  sorry
end

end ginger_children_cakes_l260_260427


namespace tan_derivative_thm_cot_derivative_thm_l260_260582

noncomputable def tan_derivative (x : ℝ) (k : ℤ) (h : x ≠ (2 * k + 1) * Real.pi / 2) : Prop :=
  HasDerivAt tan (1 / (Real.cos x)^2) x

noncomputable def cot_derivative (x : ℝ) (k : ℤ) (h : x ≠ k * Real.pi) : Prop :=
  HasDerivAt cot (-(1 / (Real.sin x)^2)) x

-- To state theorems
theorem tan_derivative_thm (x : ℝ) (k : ℤ) (h : x ≠ (2 * k + 1) * Real.pi / 2) : tan_derivative x k h := 
sorry

theorem cot_derivative_thm (x : ℝ) (k : ℤ) (h : x ≠ k * Real.pi) : cot_derivative x k h := 
sorry

end tan_derivative_thm_cot_derivative_thm_l260_260582


namespace ratio_of_volumes_l260_260506

-- Define the properties of the cube and octahedron
def volumeOfCube (side : ℝ) : ℝ :=
  side ^ 3

def volumeOfOctahedron (side : ℝ) : ℝ :=
  let base_area := (side * Real.sqrt 2) ^ 2
  let height := side / 2
  (2 * (1 / 3 * base_area * height))

-- Given problem conditions
def side_length : ℝ := 2

-- Prove that the ratio of the volumes is 1/6 and the sum of the numerator and denominator is 7
theorem ratio_of_volumes : ∃ n d : ℕ, n / (d : ℝ) = volumeOfOctahedron side_length / volumeOfCube side_length ∧ n + d = 7 :=
by
  sorry

end ratio_of_volumes_l260_260506


namespace probability_geometric_progression_dice_l260_260783

theorem probability_geometric_progression_dice :
  let total_outcomes := 6^4 in
  let favorable_sets := [([1, 1, 2, 4], 12), ([1, 2, 2, 4], 12), ([1, 2, 4, 4], 12)] in
  let total_favorable_outcomes := favorable_sets.foldl (λ acc pair, acc + pair.2) 0 in
  total_favorable_outcomes / total_outcomes = 1 / 36 :=
by
  sorry

end probability_geometric_progression_dice_l260_260783


namespace triangle_side_lengths_l260_260974

-- Define the problem
variables {r: ℝ} (h_a h_b h_c a b c : ℝ)
variable (sum_of_heights : h_a + h_b + h_c = 13)
variable (r_value : r = 4 / 3)
variable (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4)

-- Define the theorem to be proven
theorem triangle_side_lengths (h_a h_b h_c : ℝ)
  (sum_of_heights : h_a + h_b + h_c = 13) 
  (r_value : r = 4 / 3)
  (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4) :
  (a, b, c) = (32 / Real.sqrt 15, 24 / Real.sqrt 15, 16 / Real.sqrt 15) := 
sorry

end triangle_side_lengths_l260_260974


namespace perfect_squares_between_2_and_20_l260_260846

-- Defining the conditions and problem statement
theorem perfect_squares_between_2_and_20 : 
  ∃ n, n = 3 ∧ ∀ m, (2 < m ∧ m < 20 ∧ ∃ k, k * k = m) ↔ m = 4 ∨ m = 9 ∨ m = 16 :=
by {
  -- Start the proof process
  sorry -- Placeholder for the proof
}

end perfect_squares_between_2_and_20_l260_260846


namespace orthocenter_locus_circle_reflection_l260_260897

theorem orthocenter_locus_circle_reflection (Γ A B : Type) [MetricSpace Γ] [Circle Γ A] [Circle Γ B]
  (hAB_not_diametric : ¬(A = B ∧ distance A B = diameter Γ))
  (P : Γ) (hP_not_AB : P ≠ A ∧ P ≠ B) :
  let H := orthocenter A B P in
  ∀ P ∈ Γ, locus_of H = ((reflection Γ AB) \ {A, B}) := 
sorry

end orthocenter_locus_circle_reflection_l260_260897


namespace evaluate_product_of_powers_of_i_l260_260007

noncomputable def i : ℂ := complex.I

lemma powers_of_i_cyclic (n : ℤ) : i ^ (n + 4) = i ^ n := by
  rw [complex.I_pow_add, complex.I_pow_four];
  ring

theorem evaluate_product_of_powers_of_i :
  (i ^ 15) * (i ^ 135) = -1 := by
  have h1 : i ^ 15 = i ^ 3 := by
    rw [←int.mod_add_div 15 4, powers_of_i_cyclic];
    norm_num
  have h2 : i ^ 135 = i ^ 3 := by
    rw [←int.mod_add_div 135 4, powers_of_i_cyclic];
    norm_num
  rw [h1, h2];
  norm_num;
  exact complex.I_pow_three

#eval evaluate_product_of_powers_of_i  -- for testing purpose

end evaluate_product_of_powers_of_i_l260_260007


namespace budget_per_friend_l260_260526

-- Definitions for conditions
def total_budget : ℕ := 100
def parents_gift_cost : ℕ := 14
def number_of_parents : ℕ := 2
def number_of_friends : ℕ := 8

-- Statement to prove
theorem budget_per_friend :
  (total_budget - number_of_parents * parents_gift_cost) / number_of_friends = 9 :=
by
  sorry

end budget_per_friend_l260_260526


namespace number_of_boys_l260_260682

variable (x y : ℕ)

theorem number_of_boys (h1 : x + y = 900) (h2 : y = (x / 100) * 900) : x = 90 :=
by
  sorry

end number_of_boys_l260_260682


namespace prove_b_div_c_equals_one_l260_260852

theorem prove_b_div_c_equals_one
  (a b c d : ℕ)
  (h_a : a > 0 ∧ a < 4)
  (h_b : b > 0 ∧ b < 4)
  (h_c : c > 0 ∧ c < 4)
  (h_d : d > 0 ∧ d < 4)
  (h_eq : 4^a + 3^b + 2^c + 1^d = 78) :
  b / c = 1 :=
by
  sorry

end prove_b_div_c_equals_one_l260_260852


namespace honor_students_count_l260_260985

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l260_260985


namespace sum_of_ages_l260_260892

theorem sum_of_ages (J L : ℕ) (h1 : J = L + 8) (h2 : J + 5 = 3 * (L - 6)) : (J + L) = 39 :=
by {
  -- Proof steps would go here, but are omitted for this task per instructions
  sorry
}

end sum_of_ages_l260_260892


namespace solve_equation_l260_260409

theorem solve_equation (x : ℝ) (h : x ≥ 1) :
  (sqrt (x + 2 - 2 * sqrt (x - 1)) + 
  sqrt (x + 5 - 3 * sqrt (x - 1)) = 2) ↔ 
  (2 ≤ x ∧ x ≤ 5) :=
by sorry

end solve_equation_l260_260409


namespace find_divisor_l260_260273

theorem find_divisor : exists d : ℕ, 
  (∀ x : ℕ, x ≥ 10 ∧ x ≤ 1000000 → x % d = 0) ∧ 
  (10 + 999990 * d/111110 = 1000000) ∧
  d = 9 := by
  sorry

end find_divisor_l260_260273


namespace periodic_points_dense_in_interval_l260_260928

-- Definitions and conditions from problem
variable {f : ℝ → ℝ}
variable (h_cont : ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1)
variable (h_continuous: ContinuousOn f (Icc 0 1))
variable (h_top_trans : ∀ U V : Set ℝ, IsOpen U → IsOpen V → (U ∩ Icc 0 1).Nonempty → (V ∩ Icc 0 1).Nonempty → ∃ n : ℕ, (Function.iterate f n '' U ∩ V).Nonempty)

-- Statement to prove
theorem periodic_points_dense_in_interval :
  Dense {x : ℝ | ∃ n : ℕ, Function.iterate f n x = x ∧ 0 ≤ x ∧ x ≤ 1} :=
sorry

end periodic_points_dense_in_interval_l260_260928


namespace max_value_of_expression_l260_260833

noncomputable def expression (x y z : ℝ) := sin (x - y) + sin (y - z) + sin (z - x)

theorem max_value_of_expression :
  ∀ x y z ∈ set.Icc (0 : ℝ) (real.pi / 2), expression x y z ≤ real.sqrt 2 - 1 :=
sorry

end max_value_of_expression_l260_260833


namespace bags_sold_on_Thursday_l260_260176

theorem bags_sold_on_Thursday 
    (total_bags : ℕ) (sold_Monday : ℕ) (sold_Tuesday : ℕ) (sold_Wednesday : ℕ) (sold_Friday : ℕ) (percent_not_sold : ℕ) :
    total_bags = 600 →
    sold_Monday = 25 →
    sold_Tuesday = 70 →
    sold_Wednesday = 100 →
    sold_Friday = 145 →
    percent_not_sold = 25 →
    ∃ (sold_Thursday : ℕ), sold_Thursday = 110 :=
by
  sorry

end bags_sold_on_Thursday_l260_260176


namespace smallest_number_people_divisible_by_18_and_42_l260_260656

theorem smallest_number_people_divisible_by_18_and_42 : ∃ x : ℕ, (x % 18 = 0 ∧ x % 42 = 0) ∧ x = 126 :=
by
  existsi 126
  simp [Nat.lcm]
  sorry

end smallest_number_people_divisible_by_18_and_42_l260_260656


namespace negation_of_universal_proposition_l260_260970

variable (f : ℕ+ → ℕ+)

theorem negation_of_universal_proposition :
  (¬ (∀ n : ℕ+, (f n ∈ ℕ+ ∧ f n ≤ n))) ↔ (∃ n0 : ℕ+, (f n0 ∉ ℕ+) ∨ (f n0 > n0)) :=
  sorry

end negation_of_universal_proposition_l260_260970


namespace honor_students_count_l260_260992

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260992


namespace alyssa_puppies_l260_260363

-- Definitions from the problem conditions
def initial_puppies (P x : ℕ) : ℕ := P + x

-- Lean 4 Statement of the problem
theorem alyssa_puppies (P x : ℕ) (given_aw: 7 = 7) (remaining: 5 = 5) :
  initial_puppies P x = 12 :=
sorry

end alyssa_puppies_l260_260363


namespace Jackie_apples_count_l260_260724

variable (Adam_apples Jackie_apples : ℕ)
variable (h1 : Adam_apples = 10)
variable (h2 : Adam_apples = Jackie_apples + 8)

theorem Jackie_apples_count : Jackie_apples = 2 := by
  sorry

end Jackie_apples_count_l260_260724


namespace dvd_blu_ratio_l260_260706

theorem dvd_blu_ratio (D B : ℕ) (h1 : D + B = 378) (h2 : (D : ℚ) / (B - 4 : ℚ) = 9 / 2) :
  D / Nat.gcd D B = 51 ∧ B / Nat.gcd D B = 12 :=
by
  sorry

end dvd_blu_ratio_l260_260706


namespace p_cycling_speed_l260_260751

-- J starts walking at 6 kmph at 12:00
def start_time : ℕ := 12 * 60  -- time in minutes for convenience
def j_speed : ℤ := 6  -- in kmph
def j_start_time : ℕ := start_time  -- 12:00 in minutes

-- P starts cycling at 13:30
def p_start_time : ℕ := (13 * 60) + 30  -- time in minutes for convenience

-- They are at their respective positions at 19:30
def end_time : ℕ := (19 * 60) + 30  -- time in minutes for convenience

-- At 19:30, J is 3 km behind P
def j_behind_p_distance : ℤ := 3  -- in kilometers

-- Prove that P's cycling speed = 8 kmph
theorem p_cycling_speed {p_speed : ℤ} :
  j_start_time = start_time →
  p_start_time = (13 * 60) + 30 →
  end_time = (19 * 60) + 30 →
  j_speed = 6 →
  j_behind_p_distance = 3 →
  p_speed = 8 :=
by
  sorry

end p_cycling_speed_l260_260751


namespace determine_digit_z_l260_260395

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l260_260395


namespace pirate_rick_dig_time_l260_260579

/-- Given the initial conditions and total sand and mud displacement, 
prove that the time required for Pirate Rick to dig up his treasure 
upon return is approximately 6.5625 hours. -/
theorem pirate_rick_dig_time :
  let initial_digging_time_hrs := 4
  let initial_sand_ft := 8
  let sand_remaining_after_storm_ft := initial_sand_ft / 2
  let sand_after_tsunami_ft := sand_remaining_after_storm_ft + 2
  let sand_after_earthquake_ft := sand_after_tsunami_ft + 1.5
  let total_sand_and_mud_ft := sand_after_earthquake_ft + 3
  let original_digging_speed_ft_per_hr := initial_sand_ft / initial_digging_time_hrs
  let digging_speed_after_conditions_ft_per_hr := original_digging_speed_ft_per_hr * 0.80
  let time_to_dig_ft := total_sand_and_mud_ft / digging_speed_after_conditions_ft_per_hr
  (time_to_dig_ft ≈ 6.5625) :=
  by
    -- Proof omitted
    sorry

end pirate_rick_dig_time_l260_260579


namespace factorization_of_cubic_polynomial_l260_260012

-- Define the elements and the problem
variable (a : ℝ)

theorem factorization_of_cubic_polynomial :
  a^3 - 3 * a = a * (a + Real.sqrt 3) * (a - Real.sqrt 3) := by
  sorry

end factorization_of_cubic_polynomial_l260_260012


namespace average_remaining_after_trim_l260_260267

-- Definition of the given seven scores.
variable (scores : List ℕ) (h : scores.length = 7)

-- Definition that removing the highest and lowest scores yields a specific average
theorem average_remaining_after_trim (h : scores.length = 7)
  (remaining_scores : List ℕ := scores.erase (List.minimum scores))
  (remaining_scores_trimmed : List ℕ := remaining_scores.erase (List.maximum remaining_scores))
  (h_trimmed_length : remaining_scores_trimmed.length = 5) :
  (List.sum remaining_scores_trimmed) / 5 = 85 :=
sorry

end average_remaining_after_trim_l260_260267


namespace range_of_m_l260_260140

noncomputable def single_valued (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x ∈ set.Icc a b, (b - a) * (f' x) = f b - f a

theorem range_of_m (a m : ℝ) :
  (∃! x ∈ set.Icc 0 a, (a - 0) * ((3 * x^2 - 2 * x)) = (a^3 - a^2 + m - m)) →
  (a > 1/2) →
  (∃! x ∈ set.Icc 0 a, (x^3 - x^2 + m) = 0) →
  (-1 ≤ m ∧ m < 4 / 27) :=
sorry

end range_of_m_l260_260140


namespace center_of_intersection_circle_on_circumcircle_l260_260586

-- Definitions for the problem
variables {A B C : Point}
def circumcenter (k : Circle) (A B C : Point) : Point := sorry
def incenter (A B C : Point) : Point := sorry

-- Problem statement in Lean 4
theorem center_of_intersection_circle_on_circumcircle (k k1 : Circle) (O1 : Point)
  (h1 : k = circumcircle A B C)
  (h2 : k1 = Circle.mk A (incenter A B C) C)
  (h3 : O1 = center k1) :
  O1 ∈ k :=
  sorry

end center_of_intersection_circle_on_circumcircle_l260_260586


namespace ellipse_eccentricity_l260_260794

theorem ellipse_eccentricity
  (a b : ℝ)
  (h₁ : 0 < b)
  (h₂ : b < a)
  (h₃ : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (h₄ : ∀ (x y : ℝ), ∃ m n : ℝ, m = √(a^2 - b^2) ∧ n = b ∧ (y^2 / m^2 - x^2 / n^2 = 1))
  (h₅ : ∀ y, ∃ x : ℝ, y = ± x ∧ (x^2 / a^2 + y^2 / b^2 = 1))
  (h_square : ∀ x y : ℝ, |y| = |x|):
  eccentricity (conic_section.ellipse a b) = √2 / 2 := sorry

end ellipse_eccentricity_l260_260794


namespace count_A_l260_260103

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A (s : Finset ℕ) : Prop := s.card = 4 ∧ s ⊆ U
def complement_A (s : Finset ℕ) : Finset ℕ := U \ s

theorem count_A (a b : Finset ℕ) (ha : A a) (hb : A b) :
  (∑ x in a, x) < (∑ y in (complement_A a), y) → 
  Finset.count A {x ∈ (Finset.powersetLen 4 U) | 
  (∑ x in x, x) < (∑ y in (complement_A x), y)} = 31 :=
by 
  sorry

end count_A_l260_260103


namespace simplify_and_evaluate_l260_260596

theorem simplify_and_evaluate (a b : ℝ) (h : a - 2 * b = -1) :
  -3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -8 :=
by
  sorry

end simplify_and_evaluate_l260_260596


namespace C_share_of_profit_l260_260678

variable (A B C P Rs_36000 k : ℝ)

-- Definitions as per the conditions given in the problem statement.
def investment_A := 24000
def investment_B := 32000
def investment_C := 36000
def total_profit := 92000
def C_Share := 36000

-- The Lean statement without the proof as requested.
theorem C_share_of_profit 
  (h_A : investment_A = 24000)
  (h_B : investment_B = 32000)
  (h_C : investment_C = 36000)
  (h_P : total_profit = 92000)
  (h_C_share : C_Share = 36000)
  : C_Share = (investment_C / k) / ((investment_A / k) + (investment_B / k) + (investment_C / k)) * total_profit := 
sorry

end C_share_of_profit_l260_260678


namespace trapezoid_perimeter_l260_260304

noncomputable def perimeter_of_trapezoid (AB CD BC AD AP DQ : ℕ) : ℕ :=
  AB + BC + CD + AD

theorem trapezoid_perimeter (AB CD BC AP DQ : ℕ) (hBC : BC = 50) (hAP : AP = 18) (hDQ : DQ = 7) :
  perimeter_of_trapezoid AB CD BC (AP + BC + DQ) AP DQ = 180 :=
by 
  unfold perimeter_of_trapezoid
  rw [hBC, hAP, hDQ]
  -- sorry to skip the proof
  sorry

end trapezoid_perimeter_l260_260304


namespace hydrogen_to_oxygen_ratio_l260_260646

theorem hydrogen_to_oxygen_ratio (total_mass_water mass_hydrogen mass_oxygen : ℝ) 
(h1 : total_mass_water = 117)
(h2 : mass_hydrogen = 13)
(h3 : mass_oxygen = total_mass_water - mass_hydrogen) :
(mass_hydrogen / mass_oxygen) = 1 / 8 := 
sorry

end hydrogen_to_oxygen_ratio_l260_260646


namespace num_non_congruent_triangles_with_perimeter_12_l260_260119

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l260_260119


namespace roots_of_quadratic_l260_260979

theorem roots_of_quadratic :
  ∀ x : ℝ, (x - 3) ^ 2 = 4 → x = 5 ∨ x = 1 :=
begin
  intros x hx,
  have h_pos : x - 3 = 2 ∨ x - 3 = -2,
  { rw [eq_comm, pow_two] at hx,
    rwa sqr_eq_iff_abs_eq at hx, },
  cases h_pos with h1 h2,
  { left,
    linarith, },
  { right,
    linarith, },
end

end roots_of_quadratic_l260_260979


namespace sinC_calculation_maxArea_calculation_l260_260868

noncomputable def sinC_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  Real.sin C

theorem sinC_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2) 
  (h4 : Real.sin B = Real.sqrt 5 / 3) : 
  sinC_given_sides_and_angles A B C a b c h1 h2 h3 = 2 / 3 := by sorry

noncomputable def maxArea_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem maxArea_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2)
  (h4 : Real.sin B = Real.sqrt 5 / 3) 
  (h5 : a * c ≤ 15 / 2) : 
  maxArea_given_sides_and_angles A B C a b c h1 h2 h3 = 5 * Real.sqrt 5 / 4 := by sorry

end sinC_calculation_maxArea_calculation_l260_260868


namespace passengers_at_second_stop_l260_260626

theorem passengers_at_second_stop :
  let total_seats := 23 * 4 in
  let initial_passengers := 16 in
  let first_board := 15 in
  let first_disembark := 3 in
  let second_disembark := 10 in
  let empty_seats_after_second_stop := 57 in
  ∃ (x : ℕ), 
    let initial_empty_seats := total_seats - initial_passengers in
    let empty_seats_after_first_stop := initial_empty_seats - (first_board - first_disembark) in
    empty_seats_after_first_stop - x + second_disembark = empty_seats_after_second_stop ∧ x = 17 :=
begin
  let total_seats := 23 * 4,
  let initial_passengers := 16,
  let first_board := 15,
  let first_disembark := 3,
  let second_disembark := 10,
  let empty_seats_after_second_stop := 57,
  let initial_empty_seats := total_seats - initial_passengers,
  let empty_seats_after_first_stop := initial_empty_seats - (first_board - first_disembark),
  use (empty_seats_after_first_stop + second_disembark - empty_seats_after_second_stop),
  split,
  { 
    change empty_seats_after_first_stop - (empty_seats_after_first_stop + second_disembark - empty_seats_after_second_stop) + second_disembark = empty_seats_after_second_stop,
    ring,
  },
  {
    change 64 - 17 + 10 = 57,
    refl,
  },
end

end passengers_at_second_stop_l260_260626


namespace geometric_sequence_increasing_iff_l260_260188

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem geometric_sequence_increasing_iff 
  (ha : is_geometric_sequence a q) 
  (h : a 0 < a 1 ∧ a 1 < a 2) : 
  is_increasing_sequence a ↔ (a 0 < a 1 ∧ a 1 < a 2) := 
sorry

end geometric_sequence_increasing_iff_l260_260188


namespace other_root_of_quadratic_l260_260453

theorem other_root_of_quadratic (m : ℝ) :
  (IsRoot (λ x : ℝ, x^2 + m * x - 4) 1) → (IsRoot (λ x : ℝ, x^2 + m * x - 4) (-4 / 1)) :=
sorry

end other_root_of_quadratic_l260_260453


namespace angle_between_lines_l260_260165

-- Define the geometric setup
variables (A B C D E F O : Point)
variable (prism : TriangularPrism A B C D)
variable (midpoint_E : E = midpoint A C)
variable (midpoint_F : F = midpoint A D)
variable (centroid_O : O = centroid B C D)
variable (length_one : ∀ (X Y : Point), (edge X Y ∈ edges prism) → length (edge X Y) = 1)

-- Define the vectors and the angle between them
variables (v_BE v_FO : Vector)
variable (BE : Vector)
variable (FO : Vector)
variable (dot_product : dot v_BE v_FO = (length v_BE) * (length v_FO) * (AngleCos v_BE v_FO))
variable (cos_theta : AngleCos v_BE v_FO = frac 5 18 * sqrt 3)

-- Main theorem
theorem angle_between_lines :
  angle_between_lines \(BE F O\) = arccos (frac 5 18 * sqrt 3) :=
sorry

end angle_between_lines_l260_260165


namespace circus_tent_capacity_l260_260278

theorem circus_tent_capacity (total_sections : ℕ) (total_capacity : ℕ)
  (equal_capacity : Prop) (total_sections = 4) (total_capacity = 984)
  (equal_capacity = (total_capacity / total_sections = 246)) :
  total_capacity / total_sections = 246 := 
by
  sorry

end circus_tent_capacity_l260_260278


namespace exists_x0_in_interval_l260_260908

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_x0_in_interval :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 4 ∧ f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
sorry

end exists_x0_in_interval_l260_260908


namespace find_k_from_ternary_l260_260144

theorem find_k_from_ternary :
  ∃ (k : ℕ), k > 0 ∧ (1 * 3^3 + k * 3^2 + 2 = 35) ∧ (k = 2) :=
by
  use 2
  split
  · sorry -- Prove that \( k > 0 \)
  · split
    · sorry -- Prove that \( 1 * 3^3 + 2 * 3^2 + 2 = 35 \)
    · refl -- Prove that \( k = 2 \)

end find_k_from_ternary_l260_260144


namespace seq_formula_l260_260836

noncomputable def seq : ℕ → ℝ
| 0 := -1 -- defined for consistency with definition on natural numbers
| (n + 1) := seq n + 1 + 2 * real.sqrt (1 + seq n)

theorem seq_formula (n : ℕ) (hn : n > 0) : seq n = n^2 - 1 :=
by
  sorry

end seq_formula_l260_260836


namespace evaluate_f_l260_260489

def f (x : ℝ) : ℝ := x^2 + 4*x - 3

theorem evaluate_f (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 :=
by 
  -- The proof is omitted
  sorry

end evaluate_f_l260_260489


namespace alice_weekly_walk_distance_l260_260008

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end alice_weekly_walk_distance_l260_260008


namespace lewis_earnings_l260_260567

theorem lewis_earnings :
  ∀ (weekly_earnings rent_per_week weeks_in_harvest : ℕ),
    weekly_earnings = 403 →
    rent_per_week = 49 →
    weeks_in_harvest = 233 →
    weekly_earnings * weeks_in_harvest - rent_per_week * weeks_in_harvest = 82782 :=
by {
  intros,
  sorry
}

end lewis_earnings_l260_260567


namespace complex_number_sum_product_real_a_l260_260492

theorem complex_number_sum_product_real_a :
  ∀ (a b : ℝ), (-1 + a * Complex.i) + (b - Complex.i) ∈ ℝ ∧ (-1 + a * Complex.i) * (b - Complex.i) ∈ ℝ → a = 1 :=
by
  intros a b h
  have h1 : Complex.i = 0 by sorry
  exact sorry

end complex_number_sum_product_real_a_l260_260492


namespace area_of_triangle_l260_260883

open Complex

theorem area_of_triangle (z : ℂ) (hnd : z ≠ 0 ∧ z^3 ≠ 1) (htri : (z^2 - z).abs = (z^5 - z^2).abs ∧ arg (z^5 - z^2) - arg (z^2 - z) = 2 * π / 3) :
  (sqrt 3)/4 := by
sorry

end area_of_triangle_l260_260883


namespace sum_of_possible_n_values_l260_260205

def f (x n : ℝ) : ℝ :=
if x < n then x^2 + 4 * x + 1 else 3 * x + 7

theorem sum_of_possible_n_values : 
  (∃ n : ℝ, ∀ x : ℝ, 
    (x < n → f x n = x^2 + 4 * x + 1) ∧ 
    (x ≥ n → f x n = 3 * x + 7) ∧ 
    continuous_at (λ x : ℝ, f x n) n) → 
  (let roots := {n : ℝ | n^2 + n - 6 = 0} in 
  ∑ x in roots, x = -1) :=
sorry

end sum_of_possible_n_values_l260_260205


namespace factorization_of_expression_l260_260617

open Polynomial

theorem factorization_of_expression (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^3 * b + a^3 * c + a^2 * b^2 + a^2 * b * c + a^2 * c^2 + a * b^3 + a * b * c^2 + a * c^3 + b^3 * c + b^2 * c^2 + b * c^3) := by
  sorry

end factorization_of_expression_l260_260617


namespace Alexander_max_investment_l260_260752

theorem Alexander_max_investment (principal : ℝ) (time_single : ℝ) (time_double : ℝ) (price_2022 : ℝ) : 
  principal = 70 ∧ time_single = 1.1 ∧ time_double = 1.08 ∧ price_2022 = 100 → 
  (principal * time_single * 1.05 ≤ price_2022) ∧ (principal * time_double * time_double ≤ price_2022) :=
by
  intros h
  rcases h with ⟨hp, hs, hd, p2022⟩
  have h1 : principal * time_single * 1.05 = 70 * 1.1 * 1.05, by rw [hp, hs]
  have h2 : principal * time_double * time_double = 70 * 1.08 * 1.08, by rw [hp, hd]
  have ha1 : 70 * 1.1 * 1.05 = 80.85 := by norm_num
  have ha2 : 70 * 1.08 * 1.08 = 81.648 := by norm_num
  rw [h1, ha1, p2022]
  rw [h2, ha2, p2022]
  split
  norm_num
  norm_num

end Alexander_max_investment_l260_260752


namespace log_geometric_sequence_value_l260_260512

theorem log_geometric_sequence_value :
  ∀ (a : ℕ → ℤ) (b : ℕ → ℤ),
  (∃ (d : ℤ), d ≠ 0 ∧ ∀ n : ℕ, a n = a 1 + ↑(n - 1) * d) →
  2 * a 3 - a 7 ^ 2 + 2 * a 11 = 0 →
  (∃ r : ℤ, r ≠ 0 ∧ ∀ m n : ℕ, b m * b n = b (m + n) * r ^ (m - n)) →
  b 7 = a 7 →
  (log 2 (b 6 * b 8) : ℤ) = 6 :=
by
  intros a b ha_seq ha_eq hb_geoseq hb_cond
  sorry

end log_geometric_sequence_value_l260_260512


namespace number_of_valid_numbers_l260_260756

-- Define the condition of the digits and the problem constraints
def is_valid (n : ℕ) : Prop := 
  let digits := [1, 2, 3] in
  (∀ d ∈ digits, count_digit d n > 0) ∧
  (count_digit n > 4) ∧ 
  (∀ i < 3, n[i] ≠ n[i+1])

-- Function to count occurrences of digit in the number
def count_digit (d : ℕ) (n : ℕ) : ℕ := sorry

-- Function to count the number of digits (doesn't need to use digits 0-9 only)
def num_digits (n : ℕ) : ℕ := sorry

-- The actual theorem proving that the conditions match exactly 18 valid numbers.
theorem number_of_valid_numbers : ∃ n : ℕ, is_valid n ∧ n = 18 := sorry

end number_of_valid_numbers_l260_260756


namespace star_difference_l260_260847

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by
  sorry

end star_difference_l260_260847


namespace most_cookies_at_one_time_l260_260765

theorem most_cookies_at_one_time (C : ℕ) (hC : 0 < C) :
  let dongwoo := C / 7
  let minyeop := C / 8
  let seongjeong := C / 5
  seongjeong > dongwoo ∧ seongjeong > minyeop :=
by
  have hd : dongwoo = C / 7 := rfl
  have hm : minyeop = C / 8 := rfl
  have hs : seongjeong = C / 5 := rfl
  sorry

end most_cookies_at_one_time_l260_260765


namespace min_people_wearing_both_l260_260877

theorem min_people_wearing_both (n : ℕ) (h1 : n % 3 = 0)
  (h_gloves : ∃ g, g = n / 3 ∧ g = 1) (h_hats : ∃ h, h = (2 * n) / 3 ∧ h = 2) :
  ∃ x, x = 0 := by
  sorry

end min_people_wearing_both_l260_260877


namespace y1_y2_ratio_inversely_proportional_l260_260244

theorem y1_y2_ratio_inversely_proportional
  (x y : ℝ)
  (h_inv_prop : ∃ k, ∀ x y, x * y = k)
  (x1 x2 y1 y2 : ℝ)
  (hx1_ne_zero : x1 ≠ 0) 
  (hx2_ne_zero : x2 ≠ 0)
  (hy1_ne_zero : y1 ≠ 0) 
  (hy2_ne_zero : y2 ≠ 0)
  (h_ratio : x1 / x2 = 3 / 5)
  (h_corr : x1 * y1 = x2 * y2) :
  y1 / y2 = 5 / 3 := 
sorry

end y1_y2_ratio_inversely_proportional_l260_260244


namespace determine_weights_of_balls_l260_260290

theorem determine_weights_of_balls (A B C D E m1 m2 m3 m4 m5 m6 m7 m8 m9 : ℝ)
  (h1 : m1 = A)
  (h2 : m2 = B)
  (h3 : m3 = C)
  (h4 : m4 = A + D)
  (h5 : m5 = A + E)
  (h6 : m6 = B + D)
  (h7 : m7 = B + E)
  (h8 : m8 = C + D)
  (h9 : m9 = C + E) :
  ∃ (A' B' C' D' E' : ℝ), 
    ((A' = A ∨ B' = B ∨ C' = C ∨ D' = D ∨ E' = E) ∧
     (A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ E' ∧
      B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ E' ∧
      C' ≠ D' ∧ C' ≠ E' ∧
      D' ≠ E')) :=
sorry

end determine_weights_of_balls_l260_260290


namespace max_sum_arithmetic_sequence_l260_260799

theorem max_sum_arithmetic_sequence {d a1 c : ℝ} (h1 : d < 0) (h2 : ∀ x, 0 ≤ x ∧ x ≤ 22 → 0 ≤ (d / 2) * x^2 + (a1 - d / 2) * x + c) : ∃ n : ℕ, n = 11 ∧ ∀ m < 11, sum (range m) (λ i, a1 + i * d) < sum (range 11) (λ i, a1 + i * d) := by
  sorry

end max_sum_arithmetic_sequence_l260_260799


namespace part_I_part_II_l260_260281

-- Definition of constants and given table entries for Part (I)
def n : ℕ := 30
def n11 : ℕ := 8
def n22 : ℕ := 18
def n12 : ℕ := 2
def n21 : ℕ := 2
def n1+ : ℕ := 10
def n2+ : ℕ := 20
def n+1 : ℕ := 10
def n+2 : ℕ := 20

-- Definition of chi-square critical value for 99% confidence
def chi_square_critical_value : ℝ := 6.635

-- Chi-square statistic
def chi_square := (n * (n11 * n22 - n12 * n21)^2) / (n1+ * n2+ * n+1 * n+2 : ℕ)

-- Part (I)
theorem part_I : chi_square > chi_square_critical_value := sorry

-- Part (II)

-- Probability of a student being excellent and regressed to textbook
def P_A : ℝ := 18 / 30

-- Distribution probabilities
def P_X0 : ℝ := (1 - P_A)^3
def P_X1 : ℝ := 3 * P_A * (1 - P_A)^2
def P_X2 : ℝ := 3 * (P_A)^2 * (1 - P_A)
def P_X3 : ℝ := (P_A)^3

-- Distribution table and expectation
def distribution_table : List ℝ := [P_X0, P_X1, P_X2, P_X3]

def E_X : ℝ := 0 * P_X0 + 1 * P_X1 + 2 * P_X2 + 3 * P_X3

-- The expected value E(X)
theorem part_II : E_X = 9 / 5 := sorry

end part_I_part_II_l260_260281


namespace correct_options_l260_260464

variable {ω : ℝ} (ϕ x : ℝ)
variable (f g : ℝ → ℝ)

noncomputable def f (x : ℝ) : ℝ := Math.sin (ω * x + ϕ)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 12)

theorem correct_options (hω_pos : ω > 0)
    (hϕ_bound : |ϕ| < π / 2)
    (hx1x2_extreme : ∃ (x₁ x₂ : ℝ), f' x₁ = 0 ∧ f' x₂ = 0 ∧ |x₁ - x₂| = π/2)
    (hsymmetry : (∀ x, f (π/3 - x) = f (π/3 + x))) :
    (ϕ = -π / 6) ∧ (∀ x, g (π/2 + x) + g (π/2 - x) = 0) :=
sorry

end correct_options_l260_260464


namespace conjugate_of_square_of_one_plus_i_l260_260808

def i := Complex.I

theorem conjugate_of_square_of_one_plus_i : 
  Complex.conj ((1 + i) ^ 2) = -2 * i :=
by
  sorry

end conjugate_of_square_of_one_plus_i_l260_260808


namespace number_of_non_congruent_triangles_l260_260113

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_congruent_triangles_with_perimeter_12 : ℕ :=
  { (a, b, c) | a ≤ b ∧ b ≤ c ∧ a + b + c = 12 ∧ is_triangle a b c }.to_finset.card

theorem number_of_non_congruent_triangles : non_congruent_triangles_with_perimeter_12 = 2 := sorry

end number_of_non_congruent_triangles_l260_260113


namespace total_amount_proof_l260_260707

-- Definitions of the base 8 numbers
def silks_base8 := 5267
def stones_base8 := 6712
def spices_base8 := 327

-- Conversion function from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ := sorry -- Assume this function converts a base 8 number to base 10

-- Converted values
def silks_base10 := base8_to_base10 silks_base8
def stones_base10 := base8_to_base10 stones_base8
def spices_base10 := base8_to_base10 spices_base8

-- Total amount calculation in base 10
def total_amount_base10 := silks_base10 + stones_base10 + spices_base10

-- The theorem that we want to prove
theorem total_amount_proof : total_amount_base10 = 6488 :=
by
  -- The proof is omitted here.
  sorry

end total_amount_proof_l260_260707


namespace intersection_area_l260_260508

namespace RectanglePolygons

variables (P : Type) [polygon : Polygon P]

def area (p : P) : ℝ := sorry

axiom polygon_area : ∀ (p : P), area p = 1

structure Rect := (width height : ℝ)
instance : HasArea Rect := ⟨λ r, r.width * r.height⟩

/-- a list of nine polygons inside a given rectangle --/
def nine_polygons (r : Rect) (polygons : list P) : Prop :=
  r.width * r.height = 5 ∧ polygons.length = 9 ∧ ∀ p ∈ polygons, area p = 1

theorem intersection_area (r : Rect) (polygons : list P) (h : nine_polygons r polygons) :
  ∃ p1 p2 ∈ polygons, p1 ≠ p2 ∧ area (p1 ∩ p2) ≥ 1 / 9 := sorry

end RectanglePolygons

end intersection_area_l260_260508


namespace condition_for_absolute_value_l260_260051

-- Define the conditions and the theorem
theorem condition_for_absolute_value (a x : ℝ) (h : a ≠ 0) : 
  (x ∈ {-a, a} ↔ |x| = a) ↔ false :=
sorry

end condition_for_absolute_value_l260_260051


namespace non_congruent_triangles_with_perimeter_12_l260_260124

theorem non_congruent_triangles_with_perimeter_12 :
  ∃ (S : finset (ℤ × ℤ × ℤ)), S.card = 2 ∧ ∀ (a b c : ℤ), (a, b, c) ∈ S →
  a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ c < a + b :=
sorry

end non_congruent_triangles_with_perimeter_12_l260_260124


namespace sum_of_roots_l260_260398

   theorem sum_of_roots : 
     let a := 2
     let b := 7
     let c := 3
     let roots := (-b / a : ℝ)
     roots = -3.5 :=
   by
     sorry
   
end sum_of_roots_l260_260398


namespace binomial_expansion_coefficients_equal_l260_260605

theorem binomial_expansion_coefficients_equal (n : ℕ) (h : n ≥ 6)
  (h_eq : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : n = 7 := by
  sorry

end binomial_expansion_coefficients_equal_l260_260605


namespace least_three_digit_multiple_of_8_l260_260300

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l260_260300


namespace sum_of_digits_0_to_999_l260_260660

theorem sum_of_digits_0_to_999 : 
  (∑ n in Finset.range 1000, (n / 100) + ((n / 10) % 10) + (n % 10) = 13500) := 
  sorry

end sum_of_digits_0_to_999_l260_260660


namespace honor_students_count_l260_260993

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260993


namespace range_of_values_for_x2_plus_y2_l260_260059

theorem range_of_values_for_x2_plus_y2 (x y : ℝ) (z : ℂ) (hz : z = x + y * complex.I) (h : complex.abs (z - (3 + 4 * complex.I)) = 1) : 
  16 ≤ x*x + y*y ∧ x*x + y*y ≤ 36 :=
sorry

end range_of_values_for_x2_plus_y2_l260_260059


namespace phone_number_count_l260_260578

/-- 
  The number of valid 7-digit telephone numbers, where all digits are distinct, 
  the first four are in descending order, and the last four are in ascending order, 
  is 840. 
-/
theorem phone_number_count : 
  let valid_phone_number_count := (Nat.choose 10 7) * 7
  in valid_phone_number_count = 840 := 
by
  sorry

end phone_number_count_l260_260578


namespace greater_expected_area_l260_260731

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l260_260731


namespace tangent_line_eq_zeros_of_g_min_value_of_p_l260_260822

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 1)
noncomputable def g (x p : ℝ) : ℝ := x - Real.log (x - p)

theorem tangent_line_eq :
  (f' : ℝ → ℝ) (1 / 3).1 = -9 / 10 
  ∧ f (1 / 3) = 3
  ∧ (λ x, f (1 / 3) + f' (1 / 3) * (x - 1 / 3))
    = λ x, -9 / 10 * x + 33 / 10 := 
sorry

theorem zeros_of_g (p : ℝ) :
  if p < -1 then g_{(p) has two zeros} 
  else if p = -1 then g_{(p) has one zero} 
  else g_{(p) has no zeros} := 
sorry

theorem min_value_of_p (a : ℕ → ℝ) (p : ℝ) :
  (∀ n, 0 < a n ∧ a n ≤ 3)
  ∧ (∑ k in finset.range 2015, a k = 2015 / 3) 
  ∧ ∀ x ∈ set.Ioi p, ∑ k in finset.range 2015, f(a k) x ≤ g x p 
  → p ≥ 6044 :=
sorry

end tangent_line_eq_zeros_of_g_min_value_of_p_l260_260822


namespace complex_modulus_l260_260437

open Complex

theorem complex_modulus (z : ℂ) (h : (z - 1) * (2 + I) = 5 * I) : abs (conj z + I) = Real.sqrt 5 :=
sorry

end complex_modulus_l260_260437


namespace line_integral_helix_l260_260372

noncomputable def helix_curve (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * Real.cos (4 * t), 2 * Real.sin (4 * t), 6 * t)

noncomputable def integrand (x y z : ℝ) (dx dy dz : ℝ) : ℝ :=
  (2 * x + 4 * z) * dx + (2 * y + 2 * z) * dy - 12 * z * dz

theorem line_integral_helix :
  (∫ t in 0..(2 * Real.pi), integrand
    (2 * Real.cos (4 * t))
    (2 * Real.sin (4 * t))
    (6 * t)
    (-8 * Real.sin (4 * t))
    (8 * Real.cos (4 * t))
    6) = -864 * Real.pi ^ 2 :=
by
  sorry

end line_integral_helix_l260_260372


namespace we_need_more_information_to_determine_burgers_l260_260368

noncomputable def number_of_burgers := sorry

theorem we_need_more_information_to_determine_burgers 
  (b s c : ℝ) (n : ℝ) 
  (h1 : n * b + 7 * s + c = 120) 
  (h2 : 4 * b + 10 * s + c = 158.5) : 
  false :=
sorry

end we_need_more_information_to_determine_burgers_l260_260368


namespace area_of_triangle_l260_260185

open Real

def vector2 := Matrix (Fin 2) (Fin 1) ℝ

def a : vector2 := ![![4], ![-1]]
def b : vector2 := ![![2], ![3]]

def triangle_area (a b : vector2) : ℝ :=
  let two_a : vector2 := ![![2 * a 0 0], ![2 * a 1 0]]
  let det := (two_a 0 0 * b 1 0 - two_a 1 0 * b 0 0)
  abs (det / 2)

theorem area_of_triangle : triangle_area a b = 14 := by
  sorry

end area_of_triangle_l260_260185


namespace monthly_rent_calc_l260_260969

def monthly_rent (length width annual_rent_per_sq_ft : ℕ) : ℕ :=
  (length * width * annual_rent_per_sq_ft) / 12

theorem monthly_rent_calc :
  monthly_rent 10 8 360 = 2400 := 
  sorry

end monthly_rent_calc_l260_260969


namespace remainder_when_sum_divided_by_100_l260_260183

/-- Let S be the sum of all positive integers n such that n^2 + 8n - 1225 is a perfect square.
    Prove that the remainder when S is divided by 100 is 58. -/
theorem remainder_when_sum_divided_by_100 :
  let S := (Finset.filter (λ n : ℕ => ∃ m : ℤ, (n : ℤ)^2 + 8 * (n : ℤ) - 1225 = m^2) (Finset.range 2000)).sum id
  in S % 100 = 58 :=
by 
  sorry

end remainder_when_sum_divided_by_100_l260_260183


namespace minimize_wire_length_l260_260289

theorem minimize_wire_length :
  ∃ (x : ℝ), (x > 0) ∧ (2 * (x + 4 / x) = 8) :=
by
  sorry

end minimize_wire_length_l260_260289


namespace min_distance_sum_min_distance_sum_nonconvex_l260_260677

/-- Given a quadrilateral ABCD, prove the following:
    For convex quadrilateral, the point M (intersection of diagonals) minimizes the sum of distances to the vertices.
    For non-convex quadrilateral with ∠D > 180°, the vertex D minimizes the sum of distances to the vertices. 
-/
theorem min_distance_sum (A B C D : Point) (is_convex : QuadrilateralConvex A B C D) :
  let M := diagonal_intersection A B C D in
  (∀ X, distance X A + distance X B + distance X C + distance X D ≥ distance M A + distance M B + distance M C + distance M D) :=
by sorry

theorem min_distance_sum_nonconvex (A B C D : Point) (is_nonconvex : QuadrilateralNonConvex A B C D) (angle_cond : ∠D > π) :
  ∀ X, distance X A + distance X B + distance X C + distance X D ≥ distance D A + distance D B + distance D C + 0 :=
by sorry

end min_distance_sum_min_distance_sum_nonconvex_l260_260677


namespace monotonically_increasing_intervals_range_of_f_on_interval_l260_260095

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∃ (a b : ℝ), a = -Real.pi / 12 + k * Real.pi ∧ b = 5 * Real.pi / 12 + k * Real.pi ∧ 
    (∀ x : ℝ, a ≤ x ∧ x ≤ b → Real.deriv f x > 0) := sorry

theorem range_of_f_on_interval : 
  ∃ a b : ℝ, a = 0 ∧ b = Real.pi / 2 ∧ ∀ y : ℝ, y ∈ set.range (λ x, f x) ↔ - Real.sqrt 3 / 2 ≤ y ∧ y ≤ 1 := sorry

end monotonically_increasing_intervals_range_of_f_on_interval_l260_260095


namespace total_hours_played_l260_260214

-- Definitions based on conditions
def Nathan_hours_per_day : ℕ := 3
def Nathan_weeks : ℕ := 2
def days_per_week : ℕ := 7

def Tobias_hours_per_day : ℕ := 5
def Tobias_weeks : ℕ := 1

-- Calculating total hours
def Nathan_total_hours := Nathan_hours_per_day * days_per_week * Nathan_weeks
def Tobias_total_hours := Tobias_hours_per_day * days_per_week * Tobias_weeks

-- Theorem statement
theorem total_hours_played : Nathan_total_hours + Tobias_total_hours = 77 := by
  -- Proof would go here
  sorry

end total_hours_played_l260_260214


namespace proof_combination_l260_260401

open Classical

theorem proof_combination :
  (∃ x : ℝ, x^3 < 1) ∧ (¬ ∃ x : ℚ, x^2 = 2) ∧ (¬ ∀ x : ℕ, x^3 > x^2) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by
  have h1 : ∃ x : ℝ, x^3 < 1 := sorry
  have h2 : ¬ ∃ x : ℚ, x^2 = 2 := sorry
  have h3 : ¬ ∀ x : ℕ, x^3 > x^2 := sorry
  have h4 : ∀ x : ℝ, x^2 + 1 > 0 := sorry
  exact ⟨h1, h2, h3, h4⟩

end proof_combination_l260_260401


namespace sum_of_quarter_circles_arcs_l260_260608

theorem sum_of_quarter_circles_arcs (D : ℝ) (n : ℕ) :
  ∃ (N : ℕ), ∀ n ≥ N, (n : ℝ) * (π * D / (4 * n) = D) :=
sorry

end sum_of_quarter_circles_arcs_l260_260608


namespace ellipse_properties_l260_260066

theorem ellipse_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (ecc : a = 2 * b) 
  (tangent_line : ∀ (x y : ℝ), x - y + real.sqrt 6 = 0 → |0 - 0 + real.sqrt 6| / real.sqrt 2 = b) :
  (std_eq : ∀ x y, (x^2 / 4 + y^2 / 3 = 1)) ∧
  (∀ (k m : ℝ) (A B : ℝ × ℝ), 
    A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m ∧
    (A.1^2 / 4 + A.2^2 / 3 = 1) ∧
    (B.1^2 / 4 + B.2^2 / 3 = 1) ∧ 
    (d : dist (0, 0) (line k m) = |m| / real.sqrt (1 + k^2)) ∧
    (k_A_B : k * OA * OB = - 3 / 4),
    let area_Δ = 1 / 2 * (line_dist A B) * d in
    area_Δ = real.sqrt 3)
:= 
begin
  -- Proof for the theorem
  sorry
end

end ellipse_properties_l260_260066


namespace digit_ends_with_l260_260391

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l260_260391


namespace degree_of_g_l260_260240

variable {R : Type*} [CommRing R]

/-- Given polynomials f and g, suppose h(x) = f(g(x)) + g(x). -/
def h (f g : R[X]) : R[X] := λ x, f (g x) + g x

/-- The degree of h(x) is 6, and the degree of f(x) is 3. -/
theorem degree_of_g (f g : R[X]) (h : R[X]) 
  (hf : degree f = 3)
  (hh : degree (λ x, f (g x) + g x) = 6) : 
  degree g = 2 :=
sorry

end degree_of_g_l260_260240


namespace find_cos_of_angle_in_third_quadrant_l260_260854

theorem find_cos_of_angle_in_third_quadrant (B : Real) (hB_quad : ∀ θ, θ = B → 180° < θ ∧ θ < 270°) (h_sin : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end find_cos_of_angle_in_third_quadrant_l260_260854


namespace logs_currently_have_l260_260693

-- Definitions based on conditions
def total_woodblocks_needed : ℕ := 80
def woodblocks_per_log : ℕ := 5
def more_logs_needed : ℕ := 8

-- The equivalent math proof problem statement
theorem logs_currently_have (current_logs : ℕ) :
  current_logs * woodblocks_per_log = total_woodblocks_needed - (more_logs_needed * woodblocks_per_log) → 
  current_logs = 8 :=
begin
  -- Proof not provided
  sorry
end

end logs_currently_have_l260_260693


namespace vec_op_result_l260_260565

variables {θ : ℝ} (a b : ℝ ^ 3)

def unit_vector (v : ℝ ^ 3) : Prop := ∥v∥ = 1
def vec_mag_sqrt2 (v : ℝ ^ 3) : Prop := ∥v∥ = real.sqrt 2
def vec_diff_mag1 (u v : ℝ ^ 3) : Prop := ∥u - v∥ = 1
def vector_op (a b : ℝ ^ 3) (θ : ℝ) : ℝ := ∥a * real.sin θ + b * real.cos θ∥

theorem vec_op_result (a b : ℝ ^ 3) (θ : ℝ)
  (ha : unit_vector a)
  (hb : vec_mag_sqrt2 b)
  (habdiff : vec_diff_mag1 a b) :
  vector_op a b θ = real.sqrt 10 / 2 :=
sorry

end vec_op_result_l260_260565


namespace y1_y2_ratio_inversely_proportional_l260_260243

theorem y1_y2_ratio_inversely_proportional
  (x y : ℝ)
  (h_inv_prop : ∃ k, ∀ x y, x * y = k)
  (x1 x2 y1 y2 : ℝ)
  (hx1_ne_zero : x1 ≠ 0) 
  (hx2_ne_zero : x2 ≠ 0)
  (hy1_ne_zero : y1 ≠ 0) 
  (hy2_ne_zero : y2 ≠ 0)
  (h_ratio : x1 / x2 = 3 / 5)
  (h_corr : x1 * y1 = x2 * y2) :
  y1 / y2 = 5 / 3 := 
sorry

end y1_y2_ratio_inversely_proportional_l260_260243


namespace find_digits_l260_260390

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l260_260390


namespace final_price_percentage_l260_260714

variable (P : ℝ)

def sale_price_80 := 0.8 * P
def sale_price_10 := sale_price_80 - 0.1 * sale_price_80
def sale_price_5 := sale_price_10 - 0.05 * sale_price_10
def final_price := sale_price_5 - 0.15 * sale_price_5

theorem final_price_percentage (P : ℝ) : final_price P = 0.5814 * P := by
  sorry

end final_price_percentage_l260_260714


namespace avg_speed_correct_l260_260692

noncomputable def avg_speed_excl_break (
  total_distance : ℝ := 390,
  total_time : ℝ := 4,
  speed1 : ℝ := 90,
  time1 : ℝ := 1,
  speed2 : ℝ := 70,
  time2 : ℝ := 1.5,
  break_time : ℝ := 0.5,
  speed3 : ℝ := 110
) : ℝ :=
let driving_time := total_time - break_time in
let remaining_time := driving_time - (time1 + time2) in
let distance1 := speed1 * time1 in
let distance2 := speed2 * time2 in
let distance3 := speed3 * remaining_time in
let total_driving_distance := distance1 + distance2 + distance3 in
total_driving_distance / driving_time

theorem avg_speed_correct :
  avg_speed_excl_break () = 87.14 :=
by
  sorry

end avg_speed_correct_l260_260692


namespace degree_of_polynomial_product_l260_260197

theorem degree_of_polynomial_product (h j : polynomial ℝ) 
    (h_deg : h.degree = 3) 
    (j_deg : j.degree = 5) : 
    (h.comp (X^4) * j.comp (X^3)).degree = 27 :=
by sorry

end degree_of_polynomial_product_l260_260197


namespace days_with_equal_fridays_and_sundays_l260_260342

theorem days_with_equal_fridays_and_sundays 
  (days_in_month : ℕ) 
  (days_in_week : ℕ) 
  (complete_weeks : ℕ) 
  (extra_days : ℕ) 
  (num_fridays num_sundays : ℕ): 
  (days_in_month = 30) → 
  (days_in_week = 7) → 
  (complete_weeks * days_in_week + extra_days = days_in_month) → 
  (complete_weeks = 4) → 
  (extra_days = 2) → 
  (num_fridays = num_sundays) → 
  (∃ (start_day : ℕ), start_day ∈ {0, 1, 2}): 
  num_fridays = num_sundays :=
by
  sorry

end days_with_equal_fridays_and_sundays_l260_260342


namespace determine_vector_Q_l260_260899

theorem determine_vector_Q (C D Q : ℝ) (h : ∃ k : ℝ, k > 0 ∧ k * C + D = 5 * Q) :
  ∃ (m n : ℝ), m = 1 / 5 ∧ n = 4 / 5 ∧ Q = m * C + n * D :=
by {
  simp only [],
  sorry
}

end determine_vector_Q_l260_260899


namespace hanoi_tower_l260_260320

noncomputable def move_all_disks (n : ℕ) : Prop := 
  ∀ (A B C : Type), 
  (∃ (move : A → B), move = sorry) ∧ -- Only one disk can be moved
  (∃ (can_place : A → A → Prop), can_place = sorry) -- A disk cannot be placed on top of a smaller disk 
  → ∃ (u_n : ℕ), u_n = 2^n - 1 -- Formula for minimum number of steps

theorem hanoi_tower : ∀ n : ℕ, move_all_disks n :=
by sorry

end hanoi_tower_l260_260320


namespace train_length_l260_260357

theorem train_length (t1 t2 : ℝ) (platform_length : ℝ) (L : ℝ) : 
  (t1 = 8) → (t2 = 39) → (platform_length = 1162.5) → 
  let V := L / t1 in 
  let L_crosses_platform := L + platform_length in 
  let time_to_cross_platform := V * t2 in 
  L_crosses_platform = time_to_cross_platform → 
  L = 300 :=
by
  intros h1 h2 h3 V L_crosses_platform time_to_cross_platform heq
  sorry

end train_length_l260_260357


namespace jumpy_implies_not_green_l260_260590

variables (Lizard : Type)
variables (IsJumpy IsGreen CanSing CanDance : Lizard → Prop)

-- Conditions given in the problem
axiom jumpy_implies_can_sing : ∀ l, IsJumpy l → CanSing l
axiom green_implies_cannot_dance : ∀ l, IsGreen l → ¬ CanDance l
axiom cannot_dance_implies_cannot_sing : ∀ l, ¬ CanDance l → ¬ CanSing l

theorem jumpy_implies_not_green (l : Lizard) : IsJumpy l → ¬ IsGreen l :=
by
  sorry

end jumpy_implies_not_green_l260_260590


namespace xiao_ming_actual_sleep_time_l260_260964

def required_sleep_time : ℝ := 9
def recorded_excess_sleep_time : ℝ := 0.4
def actual_sleep_time (required : ℝ) (excess : ℝ) : ℝ := required + excess

theorem xiao_ming_actual_sleep_time :
  actual_sleep_time required_sleep_time recorded_excess_sleep_time = 9.4 := 
by
  sorry

end xiao_ming_actual_sleep_time_l260_260964


namespace stick_lengths_l260_260276

-- Math conditions given in the problem
variables (S L : ℕ)
def long_stick_condition : Prop := L = S + 18
def sum_condition : Prop := S + L = 30

-- Proof that the correct answers satisfy the conditions
theorem stick_lengths (S L : ℕ) (h1 : long_stick_condition S L) (h2 : sum_condition S L) :
  S = 6 ∧ L = 24 ∧ (L / S = 4) :=
by
  subst h1
  subst h2
  sorry

end stick_lengths_l260_260276


namespace train_length_l260_260645

noncomputable def length_of_each_train (v_A v_B t : ℝ) : ℝ :=
  v_A - v_B / 3.6 / t * 2

theorem train_length {L : ℝ} 
  (equal_length : Prop)
  (v_A : ℝ := 42)
  (v_B : ℝ := 36)
  (t : ℝ := 36)
  (non_uniform_deceleration : Prop)
  (higher_weight_A : Prop)
  (fluctuating_wind_resistance : Prop) :
  length_of_each_train v_A v_B t = L → L ≈ 30.06 :=
begin
  intro h,
  calc 
    length_of_each_train v_A v_B t 
      = (42 - 36) * 1000 / 3600 * 36 / 2 : by sorry
      ... = (6 * 1000 / 3600 * 36 / 2) : by sorry
      ... ≈ 30.06 : by sorry,
end

end train_length_l260_260645


namespace sum_of_digits_from_0_to_999_l260_260659

theorem sum_of_digits_from_0_to_999 : 
  (Finset.sum (Finset.range 1000) (λ n, (n.to_digits 10).sum)) = 13500 :=
by sorry

end sum_of_digits_from_0_to_999_l260_260659


namespace not_necessarily_heavier_l260_260880

/--
In a zoo, there are 10 elephants. It is known that if any four elephants stand on the left pan and any three on the right pan, the left pan will weigh more. If five elephants stand on the left pan and four on the right pan, the left pan does not necessarily weigh more.
-/
theorem not_necessarily_heavier (E : Fin 10 → ℝ) (H : ∀ (L : Finset (Fin 10)) (R : Finset (Fin 10)), L.card = 4 → R.card = 3 → L ≠ R → L.sum E > R.sum E) :
  ∃ (L' R' : Finset (Fin 10)), L'.card = 5 ∧ R'.card = 4 ∧ L'.sum E ≤ R'.sum E :=
by
  sorry

end not_necessarily_heavier_l260_260880


namespace solution_set_for_inequality_a1_range_of_a_for_solution_l260_260054

def f (x a : ℝ) : ℝ := |x + a| + |x|

theorem solution_set_for_inequality_a1 :
  {x : ℝ | f x 1 < 3} = set.Ioo (-2) 1 :=
sorry

theorem range_of_a_for_solution :
  (∃ x : ℝ, f x a < 3) → -3 < a ∧ a < 3 :=
sorry

end solution_set_for_inequality_a1_range_of_a_for_solution_l260_260054


namespace increase_in_average_weight_l260_260150

def initial_average_weight (A : ℝ) : Prop :=
  ∀ (n : ℕ), n = 10 → (10 * A) = (∑ i in (finRange 10), (λ i, avg_weight i)) ∧
             ∃ i ∈ (finRange 10), avg_weight i = 65 ∧
             (∃ new_weight, new_weight = 128 ∧
                            (10 * A - 65 + new_weight) / 10 = A + 6.3)

theorem increase_in_average_weight (A : ℝ) :
  ∀ (n : ℕ) (new_weight : ℝ),
  n = 10 → 
  new_weight = 128 → 
  (∃ (replaced_weight : ℝ), replaced_weight = 65 ∧
                             (10 * A - replaced_weight + new_weight) / 10 = A + 6.3) → 
  (new_weight - replaced_weight) / 10 = 6.3 :=
by
  intros n new_weight h1 h2 h3
  sorry

end increase_in_average_weight_l260_260150


namespace question_selection_l260_260709

theorem question_selection :
  (nat.choose 12 8) * (nat.choose 10 5) * (nat.choose 8 3) = 13838400 := 
by 
  sorry

end question_selection_l260_260709


namespace sum_of_an_series_l260_260052

theorem sum_of_an_series : (∑ k in Finset.range 9, k * (k + 1)) = 330 := 
sorry

end sum_of_an_series_l260_260052


namespace midpoint_of_A_double_prime_C_double_prime_l260_260261

open Function

-- Definitions of points and transformations
def Point := ℝ × ℝ

-- Initial points
def A : Point := (3, 2)
def D : Point := (4, 6)
def C : Point := (7, 2)

-- Translation by (−3, 4)
def translate (p : Point) : Point := (p.1 - 3, p.2 + 4)

-- Rotation by 90 degrees clockwise around origin
def rotate_90_clockwise (p : Point) : Point := (p.2, -p.1)

-- Combine translation and rotation around A'
def A_prime : Point := translate A
def D_prime : Point := translate D
def C_prime : Point := translate C

def A_double_prime : Point := A_prime
def D_double_prime : Point := (A_prime.1 + (rotate_90_clockwise (D_prime.1 - A_prime.1, D_prime.2 - A_prime.2)).1, A_prime.2 + (rotate_90_clockwise (D_prime.1 - A_prime.1, D_prime.2 - A_prime.2)).2)
def C_double_prime : Point := (A_prime.1 + (rotate_90_clockwise (C_prime.1 - A_prime.1, C_prime.2 - A_prime.2)).1, A_prime.2 + (rotate_90_clockwise (C_prime.1 - A_prime.1, C_prime.2 - A_prime.2)).2)

-- Midpoint calculation
def midpoint (p1 p2 : Point) : Point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- The theorem to be proved
theorem midpoint_of_A_double_prime_C_double_prime : 
  midpoint A_double_prime C_double_prime = (0, 4) :=
sorry

end midpoint_of_A_double_prime_C_double_prime_l260_260261


namespace sara_spent_on_bought_movie_l260_260574

-- Define the costs involved
def cost_ticket : ℝ := 10.62
def cost_rent : ℝ := 1.59
def total_spent : ℝ := 36.78

-- Define the quantity of tickets
def number_of_tickets : ℝ := 2

-- Define the total cost on tickets
def cost_on_tickets : ℝ := cost_ticket * number_of_tickets

-- Define the total cost on tickets and rented movie
def cost_on_tickets_and_rent : ℝ := cost_on_tickets + cost_rent

-- Define the total amount spent on buying the movie
def cost_bought_movie : ℝ := total_spent - cost_on_tickets_and_rent

-- The statement we need to prove
theorem sara_spent_on_bought_movie : cost_bought_movie = 13.95 :=
by
  sorry

end sara_spent_on_bought_movie_l260_260574


namespace incorrect_conclusion_l260_260780

def y (x : ℝ) : ℝ := -2 * x + 3

theorem incorrect_conclusion : ∀ (x : ℝ), y x = 0 → x ≠ 0 := 
by
  sorry

end incorrect_conclusion_l260_260780


namespace Robin_total_distance_walked_l260_260960

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end Robin_total_distance_walked_l260_260960


namespace domain_of_f_l260_260616

-- Definitions of the conditions and the function
def domain : Set ℝ := {x | 1 ≤ x ∧ x < 2}

def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.log (2 - x)

-- Prove that the domain of the function f is [1, 2)
theorem domain_of_f :
  ∀ x : ℝ, (x ∈ domain ↔ 1 ≤ x ∧ x < 2) :=
by {
  intros x,
  split,
  {
    intro h,
    exact h,
  },
  {
    intro h,
    exact h,
  }
}

end domain_of_f_l260_260616


namespace greater_expected_area_vasya_l260_260728

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l260_260728


namespace problem_solution_l260_260186

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = q * a n

def arithmetic_sequence (a b c : ℝ) : Prop :=
  (a + c) / 2 = b

theorem problem_solution :
  ∃ (a : ℕ → ℝ) (q : ℝ), geometric_sequence a q ∧
  (a 0 + a 1 + a 2 = 7) ∧
  arithmetic_sequence (a 0 + 3) (3 * a 1) (a 2 + 4) ∧
  (∀ n, a n = 2 ^ n) ∧
  ∃ (b : ℕ → ℝ), (∀ n, b n = a n + real.log (a n)) ∧
  ∀ n, (finset.range n).sum b = 2 ^ n - 1 + (n * (n - 1) / 2) * real.log 2 :=
sorry

end problem_solution_l260_260186


namespace find_UW_squared_l260_260158

noncomputable def square (a: ℝ) := a * a

variables {PQRS: RealPlane} -- RealPlane represents a plane with real coordinates
variables {P Q R S T M U V W X: RealPlane.Point}

-- Conditions from problem
axiom square_PQRS : PQRS.is_square P Q R S
axiom T_on_PQ : T ∈ P.line_through Q
axiom M_on_PS : M ∈ P.line_through S
axiom PT_PM_eq : dist P T = dist P M
axiom U_on_QR : U ∈ Q.line_through R
axiom V_on_RS : V ∈ R.line_through S
axiom W_on_TM : W ∈ T.line_through M
axiom X_on_TM : X ∈ T.line_through M
axiom UW_perp_TM : ∠ U W (T.line_through M) = π/2
axiom VX_perp_TM : ∠ V X (T.line_through M) = π/2
axiom area_PTM : ∆ P T M = 2
axiom area_QUWT : quadrilateral_area Q U W T = 2
axiom area_SMXV : quadrilateral_area S M X V = 2
axiom area_URVWX : pentagon_area U R V W X = 2

-- Statement to prove
theorem find_UW_squared : 
  ∃ (a: ℝ), a = square (dist U W) ∧ a = 20 - 8 * real.sqrt 2 := sorry

end find_UW_squared_l260_260158


namespace constant_term_in_binomial_expansion_l260_260519

theorem constant_term_in_binomial_expansion :
  let x := (√[3] x + 1/(2*x))^8 in
  (∃ (r : ℕ), (x^(8 - 4*r/3) = 0) ∧ (r = 2) ∧ 
  let term := (1/(2^r)) * (C(8, r)) in term = 7) :=
sorry

end constant_term_in_binomial_expansion_l260_260519


namespace find_pqr_l260_260239

noncomputable theory

-- Definitions based on the problem conditions
def square_center (O : ℝ) (A B C D : ℝ) := (A + B + C + D)/4 = O
def square_side (A B : ℝ) : ℝ := B - A
def tan_addition_rule (E O F : ℝ) := Real.tan (O - E) + Real.tan (O - F) = 1

-- Given values from conditions
def AB_length := 600
def EF_length := 300
def angle_45 := Real.pi / 4

-- Variables as per the definitions
variables (p q r: ℝ) (x y : ℝ) (A B C D E F O G: ℝ)

-- Building the problem conditions
def conditions : Prop :=
  square_center O A B C D ∧
  square_side A B = 600 ∧
  E < F ∧ EF_length = F - E ∧
  Real.angle O E F = angle_45 ∧
  
  -- Given or derived relations
  EF_length = 300 ∧
  x > y ∧
  x + y = 300 ∧
  x * y = 45000 ∧

  -- Calculate given BF in terms of p, q, r
  ((A + B)/2 - (E + F)/2) = p + q * Real.sqrt r ∧
  r = 3 ∧ q = 50 ∧ p = 150

-- Proof statement
theorem find_pqr : conditions p q r x y A B C D E F O G → p + q + r = 203 :=
by
  sorry -- proof skipped

end find_pqr_l260_260239


namespace solve_inequality_l260_260336

theorem solve_inequality (f : ℝ → ℝ) (n : ℕ) (a : ℝ) (h_pos : 0 < n) (h_a_neg : a < 0)
  (h_add : ∀ x y : ℝ, f (x + y) = f x + f y)
  (h_neg : ∀ x : ℝ, 0 < x → f x < 0) :
  let sol := if a < -Real.sqrt n then {x : ℝ | x > n / a ∨ x < a}
             else if a = -Real.sqrt n then {x : ℝ | x ≠ -Real.sqrt n}
             else {x : ℝ | x > a ∨ x < n / a} in
  (∀ x : ℝ, (1 / n) * f (a * x^2) - f x > (1 / n) * f (a^2 * x) - f a ↔ x ∈ sol) :=
by
  sorry

end solve_inequality_l260_260336


namespace train_speed_comparison_l260_260613

variables (V_A V_B : ℝ)

open Classical

theorem train_speed_comparison
  (distance_AB : ℝ)
  (h_distance : distance_AB = 360)
  (h_time_limit : V_A ≤ 72)
  (h_meeting_time : 3 * V_A + 2 * V_B > 360) :
  V_B > V_A :=
by {
  sorry
}

end train_speed_comparison_l260_260613


namespace fewest_handshakes_organizer_l260_260870

theorem fewest_handshakes_organizer (n k : ℕ) (h : k < n) 
  (total_handshakes: n*(n-1)/2 + k = 406) :
  k = 0 :=
sorry

end fewest_handshakes_organizer_l260_260870


namespace simplify_cos18_minus_cos54_l260_260235

noncomputable def cos_54 : ℝ := 2 * (cos 27)^2 - 1
noncomputable def cos_27 : ℝ := sqrt ((1 + cos_54) / 2)
noncomputable def cos_18 : ℝ := 1 - 2 * (sin 9)^2
noncomputable def sin_9 : ℝ := sqrt ((1 - cos_18) / 2)

theorem simplify_cos18_minus_cos54 : (cos 18 - cos 54) = 0 :=
by
  have h_cos_54 : cos 54 = cos_54 := by sorry
  have h_cos_27 : cos 27 = cos_27 := by sorry
  have h_cos_18 : cos 18 = cos_18 := by sorry
  have h_sin_9 : sin 9 = sin_9 := by sorry
  sorry

end simplify_cos18_minus_cos54_l260_260235


namespace problem_statement_l260_260063

noncomputable def f : ℝ → ℝ
| x => x^2 + 1

theorem problem_statement :
  (∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  f 0 = 1 ∧
  f (-2) = 5 ∧
  ∀ x : ℝ, x ∈ Icc (-2 : ℝ) (1 : ℝ) → (f x ≥ 1 ∧ f x ≤ 5) ∧
  (∃ x : ℝ, x ∈ Icc (-2 : ℝ) 1 ∧ f x = 1) ∧
  (∃ x : ℝ, x ∈ Icc (-2 : ℝ) 1 ∧ f x = 5)

end problem_statement_l260_260063


namespace neg_proposition_p_l260_260804

variable {x : ℝ}

def proposition_p : Prop := ∀ x ≥ 0, x^3 - 1 ≥ 0

theorem neg_proposition_p : ¬ proposition_p ↔ ∃ x ≥ 0, x^3 - 1 < 0 :=
by sorry

end neg_proposition_p_l260_260804


namespace honor_students_count_l260_260999

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260999


namespace regression_total_sum_of_squares_l260_260603

variables (y : Fin 10 → ℝ) (y_hat : Fin 10 → ℝ)
variables (residual_sum_of_squares : ℝ) 

-- Given conditions
def R_squared := 0.95
def RSS := 120.53

-- The total sum of squares is what we need to prove
noncomputable def total_sum_of_squares := 2410.6

-- Statement to prove
theorem regression_total_sum_of_squares :
  1 - RSS / total_sum_of_squares = R_squared := by
sorry

end regression_total_sum_of_squares_l260_260603


namespace vertex_of_parabola_l260_260400

theorem vertex_of_parabola (x y : ℝ) : (y^2 - 4 * y + 3 * x + 7 = 0) → (x, y) = (-1, 2) :=
by
  sorry

end vertex_of_parabola_l260_260400


namespace satisfy_conditions_l260_260002

variable (x : ℝ)

theorem satisfy_conditions :
  (3 * x^2 + 4 * x - 9 < 0) ∧ (x ≥ -2) ↔ (-2 ≤ x ∧ x < 1) := by
  sorry

end satisfy_conditions_l260_260002


namespace smallest_two_digit_prime_with_reversed_composite_and_tens_digit_two_l260_260028

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n % 10
  let d2 := n / 10
  10 * d1 + d2

theorem smallest_two_digit_prime_with_reversed_composite_and_tens_digit_two :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (n / 10 = 2) ∧ is_composite (reverse_digits n) ∧ ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ is_prime m ∧ (m / 10 = 2) ∧ is_composite (reverse_digits m) → n ≤ m :=
  ∃ n, n = 23
sorry

end smallest_two_digit_prime_with_reversed_composite_and_tens_digit_two_l260_260028


namespace find_t_l260_260107

variable (t : ℝ)

def m := (t + 1, 1 : ℝ)
def n := (t + 2, 2 : ℝ)

def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_t (h : dot_product (add_vectors (m t) (n t)) (sub_vectors (m t) (n t)) = 0) : t = -3 :=
sorry

end find_t_l260_260107


namespace solve_for_x_l260_260311

theorem solve_for_x (x : ℝ) (h : (2 * x - 3) ^ (x + 3) = 1) : 
  x = -3 ∨ x = 2 ∨ x = 1 := 
sorry

end solve_for_x_l260_260311


namespace hyperbola_center_l260_260338

theorem hyperbola_center (F1 F2 : ℂ) (F1_x F1_y F2_x F2_y : ℝ) :
  F1 = (F1_x + F1_y * I) →
  F2 = (F2_x + F2_y * I) →
  F1_x = 2 → F1_y = 3 →
  F2_x = 8 → F2_y = 6 →
  ( (F1_x + F2_x) / 2, (F1_y + F2_y) / 2 ) = (5 : ℝ, 4.5 : ℝ) :=
by {
  intros hF1 hF2 hF1x hF1y hF2x hF2y,
  simp [hF1x, hF1y, hF2x, hF2y],
  norm_num,
  exact ⟨rfl, rfl⟩
}

end hyperbola_center_l260_260338


namespace determine_k_value_l260_260779

-- Given problem definition
def base_k_eq_0_12k (k : ℕ) : Prop :=
  0.\overline{12}_k = 0.121212..._k ∧ nat.positive k

-- The repeating base-k representation of the fraction 3/28 is 0.\overline{12}_k
def repeat_base_k (k : ℕ) : Prop :=
  ∃ k, k > 0 ∧ base_k_eq_0_12k k

-- The value of k is determined to be 10
theorem determine_k_value : ∀ k : ℕ, repeat_base_k k → k = 10 := 
by 
  intros,
  sorry -- Proof will be provided here

end determine_k_value_l260_260779


namespace number_of_solutions_l260_260481

theorem number_of_solutions :
  ∃ (s : Finset ℕ), (∀ x ∈ s, ⌊ x / 10 ⌋ = ⌊ x / 11 ⌋ + 1) ∧ s.card = 110 :=
by
  sorry

end number_of_solutions_l260_260481


namespace correct_operation_l260_260665

-- Definitions of conditions
def option_A : Prop := sqrt 2 + sqrt 3 = sqrt 5
def option_B : Prop := (2 * a) ^ 3 = 8 * a ^ 3
def option_C : Prop := a ^ 8 / a ^ 4 = a ^ 2
def option_D : Prop := (a - b) ^ 2 = a ^ 2 - b ^ 2

-- Statement to prove that only option B is correct
theorem correct_operation (a b : ℝ) : option_B :=
by
  sorry

end correct_operation_l260_260665


namespace estimate_expr_l260_260404

theorem estimate_expr : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end estimate_expr_l260_260404


namespace end_balance_l260_260348

noncomputable def future_value_annuity_due (P r e n : ℝ) : ℝ :=
  P * ((e^n - 1) * e / (e - 1))

noncomputable def present_value_annuity_due (P r e n : ℝ) : ℝ :=
  P * ((1 - (e^(-n))) * e^(-1) / (e - 1))

theorem end_balance (P r n : ℝ) :
  let e := 1 + r
  let A10 := future_value_annuity_due P r e n
  let W10 := present_value_annuity_due P r e n
  A10 - W10 = 6243.12 := 
begin
  let P := 500,
  let r := 0.04,
  let e := 1.04,
  let n := 10,
  sorry
end

end end_balance_l260_260348


namespace domain_of_f_l260_260615

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.tan x - 1) + Real.sqrt (1 - x^2)

theorem domain_of_f : {x : ℝ | 0 ≤ Real.tan x - 1 ∧ 0 ≤ 1 - x^2} = Icc (Real.pi/4) 1 :=
by
  sorry

end domain_of_f_l260_260615


namespace greatest_sundays_in_56_days_l260_260291

theorem greatest_sundays_in_56_days (days_in_first: ℕ) (days_in_week: ℕ) (sundays_in_week: ℕ) : ℕ :=
by 
  -- Given conditions
  have days_in_first := 56
  have days_in_week := 7
  have sundays_in_week := 1

  -- Conclusion
  let num_weeks := days_in_first / days_in_week

  -- Answer
  exact num_weeks * sundays_in_week

-- This theorem establishes that the greatest number of Sundays in 56 days is indeed 8.
-- Proof: The number of Sundays in 56 days is given by the number of weeks (which is 8) times the number of Sundays per week (which is 1).

example : greatest_sundays_in_56_days 56 7 1 = 8 := 
by 
  unfold greatest_sundays_in_56_days
  exact rfl

end greatest_sundays_in_56_days_l260_260291


namespace larger_number_is_eight_l260_260139

variable {x y : ℝ}

theorem larger_number_is_eight (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l260_260139


namespace bounded_sequence_iff_l260_260202

theorem bounded_sequence_iff (x : ℕ → ℝ) (h : ∀ n, x (n + 1) = (n^2 + 1) * x n ^ 2 / (x n ^ 3 + n^2)) :
  (∃ C, ∀ n, x n < C) ↔ (0 < x 0 ∧ x 0 ≤ (Real.sqrt 5 - 1) / 2) ∨ x 0 ≥ 1 := sorry

end bounded_sequence_iff_l260_260202


namespace similar_triangles_angle_perpendicular_l260_260313

-- Part (a)
theorem similar_triangles
  (A B C A1 B1 M : Point)
  (h1 : Circle_through C intersects BC at A1)
  (h2 : Circle_through C intersects AC at B1)
  (h3 : Circle_through C intersects Circumcircle ABC at M) :
  Similar_triangle (Triangle A B1 M) (Triangle B A1 M) :=
sorry

-- Part (b)
theorem angle_perpendicular
  (A B C A1 B1 M O : Point)
  (s : length_segment AA1 = semiperimeter ABC)
  (t : length_segment BB1 = semiperimeter ABC)
  (h4 : CM parallel to Line A1 B1)
  (h5 : O = Incenter ABC) :
  Angle C M O = 90 :=
sorry

end similar_triangles_angle_perpendicular_l260_260313


namespace correct_derivatives_count_l260_260260

def is_correct_derivative_1 := (∀ x: ℝ, deriv (λ x, 3^x) x = 3^x * log 3)
def is_correct_derivative_2 := (∀ x: ℝ, deriv log x / log 2 = 1 / (x * log 2))
def is_correct_derivative_3 := (∀ x: ℝ, deriv (λ x, exp x) x = exp x)
def is_correct_derivative_4 := (∀ x: ℝ, deriv (λ x, 1 / log x) x = x)
def is_correct_derivative_5 := (∀ x: ℝ, deriv (λ x, x * exp(2 * x)) x = exp(2 * x) + 2 * x * exp(2 * x))

theorem correct_derivatives_count : 
  (¬ is_correct_derivative_1 ∧ is_correct_derivative_2 ∧ is_correct_derivative_3 ∧ ¬ is_correct_derivative_4 ∧ is_correct_derivative_5) → 
  3 =
  (if is_correct_derivative_1 then 1 else 0) + 
  (if is_correct_derivative_2 then 1 else 0) + 
  (if is_correct_derivative_3 then 1 else 0) + 
  (if is_correct_derivative_4 then 1 else 0) + 
  (if is_correct_derivative_5 then 1 else 0) := 
by
  sorry

end correct_derivatives_count_l260_260260


namespace pyramid_volume_l260_260708

noncomputable def volume_of_pyramid (a b h : ℝ) : ℝ :=
  (1 / 3) * (a * b) * h

theorem pyramid_volume :
  let a := 7
  let b := 9
  let edge := 15
  let h := real.sqrt (edge^2 - ((real.sqrt (a^2 + b^2)) / 2)^2)
  volume_of_pyramid a b h = 84 * real.sqrt 10 :=
by
  have base := 7
  have height := 9
  have side_length := 15
  have base_area := 63
  have center_to_corner := real.sqrt (base^2 + height^2) / 2
  have pyramid_height := real.sqrt (side_length^2 - center_to_corner^2)
  have volume := (1 / 3) * base_area * pyramid_height
  exact Eq.symm sorry

end pyramid_volume_l260_260708


namespace simplify_complex_expression_l260_260930

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l260_260930


namespace proof_problem_l260_260367

noncomputable def problem_statement : Prop :=
  let A_initial_pos : ℝ := 50    -- A is 50 feet above T
  let B_initial_pos : ℝ := -100  -- T is 100 feet above B, so B's position is -100 relative to T
  let T_speed : ℝ := v           -- T flies horizontally at speed v
  let A_speed : ℝ := k * v       -- A flies directly towards B at speed k*v
  let B_speed : ℝ := 2 * v       -- B flies directly towards T at speed 2v
  let dist_T : ℝ := 200 / 3      -- Given in solution steps
  let dist_A : ℝ := 96.2         -- Derived in solution, distance traveled by A
  let dist_B : ℝ := 133.3        -- Derived in solution, distance traveled by B
  let A_speed_ratio : ℝ := 1.443 -- Derived in solution, speed ratio of A to T
  (dist_T = 200 / 3) ∧ (dist_A = 96.2) ∧ (dist_B = 133.3) ∧ (k = 1.443)
  
theorem proof_problem : problem_statement :=
  by
  -- Proof omitted, just the statement required
  sorry

end proof_problem_l260_260367


namespace general_formula_l260_260164

def sequence_a (n : ℕ) : ℕ :=
by sorry

def partial_sum (n : ℕ) : ℕ :=
by sorry

axiom base_case : partial_sum 1 = 5

axiom recurrence_relation (n : ℕ) (h : 2 ≤ n) : partial_sum (n - 1) = sequence_a n

theorem general_formula (n : ℕ) : partial_sum n = 5 * 2^(n-1) :=
by
-- Proof will be provided here
sorry

end general_formula_l260_260164


namespace vector_sum_to_M_l260_260551

-- Define properties of points and vectors
variables {A B C D M O : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup O] 
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D] [Module ℝ M] [Module ℝ O] 
  [AddCommGroup (A →₀ ℝ)] [AddCommGroup (B →₀ ℝ)] [AddCommGroup (C →₀ ℝ)] [AddCommGroup (D →₀ ℝ)] 
  [AddCommGroup (M →₀ ℝ)] [AddCommGroup (O →₀ ℝ)] 

-- Define the coordinates
variables (OA OB OC OD OM : A)
variables (quadrilateral : Prop)

-- Conditions
axiom intersection_M : M = (A + B + C + D) / 2
axiom point_O_not_M : O ≠ M
axiom quadrilateral_condition : quadrilateral → (M = (A + B + C + D) / 2)

-- Proof statement
theorem vector_sum_to_M {A B C D M O : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] 
  [AddCommGroup M] [AddCommGroup O] [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D] 
  [Module ℝ M] [Module ℝ O] [AddCommGroup (A →₀ ℝ)] [AddCommGroup (B →₀ ℝ)] [AddCommGroup (C →₀ ℝ)] 
  [AddCommGroup (D →₀ ℝ)] [AddCommGroup (M →₀ ℝ)] [AddCommGroup (O →₀ ℝ)] (OA OB OC OD OM : A) 
  (quadrilateral : Prop) (intersection_M : M = (A + B + C + D) / 2) (point_O_not_M : O ≠ M) 
  (quadrilateral_condition : quadrilateral → (M = (A + B + C + D) / 2)) :
  \(\overrightarrow{OA} + \overrightarrow{OB} + \overrightarrow{OC} + \overrightarrow{OD} = 4 \overrightarrow{OM} \) :=
sorry

end vector_sum_to_M_l260_260551


namespace simplify_and_evaluate_l260_260934

theorem simplify_and_evaluate :
  let x := -1
  let y := 2
  (2 * x + y) ^ 2 + (x + y) * (x - y) - x ^ 2 = -4 :=
by
  let x := -1
  let y := 2
  calc
    (2 * x + y) ^ 2 + (x + y) * (x - y) - x ^ 2 = ((2 * x + y) ^ 2 + (x + y) * (x - y)) - x ^ 2 : by rw add_sub_assoc
    ... = ((2 * x + y) ^ 2 + (x + y) * (x - y)) - x ^ 2 : by sorry -- additional completion here.
    ... = -4 : by sorry -- completing the proof steps.

end simplify_and_evaluate_l260_260934


namespace max_d_7_digit_multiple_of_33_l260_260013

theorem max_d_7_digit_multiple_of_33 :
  ∃ d e : ℕ, (5 * 10^6 + d * 10^5 + 5 * 10^4 + 2 * 10^3 + 2 * 10^2 + e * 10 + 1) % 33 = 0 ∧
            15 + d + e % 3 = 0 ∧
            (8 + d - e) % 11 = 0 ∧
            d ≤ 9 ∧ e ≤ 9 ∧
            ∀ d' e' : ℕ, (5 * 10^6 + d' * 10^5 + 5 * 10^4 + 2 * 10^3 + 2 * 10^2 + e' * 10 + 1) % 33 = 0 ∧ 
                      15 + d' + e' % 3 = 0 ∧ 
                      (8 + d' - e') % 11 = 0 ∧ 
                      d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d :=
  ∃ d e : ℕ, (5 000 000 + d * 100 000 + 50 000 + 2000 + 200 + e * 10 + 1) % 33 = 0 ∧
            (15 + d + e) % 3 = 0 ∧
            (8 + d - e) % 11 = 0 ∧
            d ≤ 9 ∧ e ≤ 9 ∧
            ∀ d' e' : ℕ, (5 000 000 + d' * 100 000 + 50 000 + 2000 + 200 + e' * 10 + 1) % 33 = 0 ∧ 
                      (15 + d' + e') % 3 = 0 ∧ 
                      (8 + d' - e') % 11 = 0 ∧ 
                      d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d := by
  sorry

end max_d_7_digit_multiple_of_33_l260_260013


namespace deputy_leader_handshakes_l260_260279

theorem deputy_leader_handshakes {n : ℕ} (countries : Finset (Fin n)) (leaders : countries → ℕ) (deputies : countries → ℕ)
  (handshakes : (countries ∪ deputies) → Finset (countries ∪ deputies))
  (H_no_leader_deputy_shake : ∀ c ∈ countries, ¬ (leaders c ∈ handshakes (deputies c)))
  (H_distinct_handshakes : ∀ (l ∈ countries) (p q ∈ (countries ∪ deputies)), (handshakes p).card ≠ (handshakes q).card)
  (illyrian_leader : ℕ) (illyrian_deputy : ℕ) (H_illyrian_members: illyrian_leader ∈ leaders ∧ illyrian_deputy ∈ deputies) :
  (handshakes illyrian_deputy).card = 33 := by
  sorry

end deputy_leader_handshakes_l260_260279


namespace total_days_spent_on_islands_l260_260894

-- Define the conditions and question in Lean 4
def first_expedition_A_weeks := 3
def second_expedition_A_weeks := first_expedition_A_weeks + 2
def last_expedition_A_weeks := second_expedition_A_weeks * 2

def first_expedition_B_weeks := 5
def second_expedition_B_weeks := first_expedition_B_weeks - 3
def last_expedition_B_weeks := first_expedition_B_weeks

def total_weeks_on_island_A := first_expedition_A_weeks + second_expedition_A_weeks + last_expedition_A_weeks
def total_weeks_on_island_B := first_expedition_B_weeks + second_expedition_B_weeks + last_expedition_B_weeks

def total_weeks := total_weeks_on_island_A + total_weeks_on_island_B
def total_days := total_weeks * 7

theorem total_days_spent_on_islands : total_days = 210 :=
by
  -- We skip the proof part
  sorry

end total_days_spent_on_islands_l260_260894


namespace range_m_l260_260828

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x - 1

noncomputable def g (x m : ℝ) : ℝ := f(x) - m * (Real.exp (2 * x) / x)

theorem range_m (m : ℝ) : 
  (∀ x ≥ 1, deriv (g x m) ≤ 0) ↔ m ≥ 2 / Real.exp 2 := 
by 
  sorry

end range_m_l260_260828


namespace central_conic_has_two_foci_and_directrix_circles_l260_260594

theorem central_conic_has_two_foci_and_directrix_circles
  (F₁ F₂ : Type) (directrix₁ directrix₂ : Type) (M : Type)
  (conic_section : M → Prop)
  (distance : F₂ → M → ℝ)
  (directrix_distance : M → directrix₁ → ℝ)
  (point_on_conic : M → F₂ → directrix₁ → Prop) :
  (∀ (M : M), point_on_conic M F₂ directrix₁ →
    distance F₂ M = directrix_distance M directrix₁) →
  ∃ directrix' : Type, (∀ M : M, 
    distance F₁ M + directrix_distance M directrix' = distance F₂ M + directrix_distance M directrix₁) :=
sorry

end central_conic_has_two_foci_and_directrix_circles_l260_260594


namespace prudence_sleep_4_weeks_equals_200_l260_260039

-- Conditions
def sunday_to_thursday_sleep := 6 
def friday_saturday_sleep := 9 
def nap := 1 

-- Number of days in the mentioned periods per week
def sunday_to_thursday_days := 5
def friday_saturday_days := 2
def nap_days := 2

-- Calculate total sleep per week
def total_sleep_per_week : Nat :=
  (sunday_to_thursday_days * sunday_to_thursday_sleep) +
  (friday_saturday_days * friday_saturday_sleep) +
  (nap_days * nap)

-- Calculate total sleep in 4 weeks
def total_sleep_in_4_weeks : Nat :=
  4 * total_sleep_per_week

theorem prudence_sleep_4_weeks_equals_200 : total_sleep_in_4_weeks = 200 := by
  sorry

end prudence_sleep_4_weeks_equals_200_l260_260039


namespace original_price_of_dish_l260_260529

theorem original_price_of_dish (P : ℝ) (h1 : ∃ P, John's_payment = (0.9 * P) + (0.15 * P))
                               (h2 : ∃ P, Jane's_payment = (0.9 * P) + (0.135 * P))
                               (h3 : John's_payment = Jane's_payment + 0.51) : P = 34 := by
  -- John's Payment
  let John's_payment := (0.9 * P) + (0.15 * P)
  -- Jane's Payment
  let Jane's_payment := (0.9 * P) + (0.135 * P)
  -- Condition that John paid $0.51 more than Jane
  have h3 : John's_payment = Jane's_payment + 0.51 := sorry
  -- From the given conditions, we need to prove P = 34
  sorry

end original_price_of_dish_l260_260529


namespace complex_modulus_l260_260439

theorem complex_modulus:
  (∃ (z : ℂ), (z - 1) * (2 + complex.i) = 5 * complex.i) →
  ∀ (z : ℂ), (z - 1) * (2 + complex.i) = 5 * complex.i → |conj z + complex.i| = real.sqrt 5 :=
by
  intro h,
  sorry

end complex_modulus_l260_260439


namespace find_value_l260_260075

theorem find_value (a b : ℝ) (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) : 2 * a^100 - 3 * b⁻¹ = 3 :=
by sorry

end find_value_l260_260075


namespace sphere_diameter_l260_260957

theorem sphere_diameter (r : ℝ) (V : ℝ) (threeV : ℝ) (a b : ℕ) :
  (∀ (r : ℝ), r = 5 →
  V = (4 / 3) * π * r^3 →
  threeV = 3 * V →
  D = 2 * (3 * V * 3 / (4 * π))^(1 / 3) →
  D = a * b^(1 / 3) →
  a = 10 ∧ b = 3) →
  a + b = 13 :=
by
  intros
  sorry

end sphere_diameter_l260_260957


namespace containers_distribution_l260_260272

theorem containers_distribution :
  ∃ (n m k : ℕ),
    n + 10 * m + 50 * k = 500 ∧
    n + m + k = 100 ∧
    n = 60 ∧
    m = 39 ∧
    k = 1 :=
by {
  let n := 60,
  let m := 39,
  let k := 1,
  use [n, m, k],
  simp,
  exact and.intro rfl (and.intro rfl (and.intro rfl rfl))
}

end containers_distribution_l260_260272


namespace sum_of_digits_from_0_to_999_l260_260658

theorem sum_of_digits_from_0_to_999 : 
  (Finset.sum (Finset.range 1000) (λ n, (n.to_digits 10).sum)) = 13500 :=
by sorry

end sum_of_digits_from_0_to_999_l260_260658


namespace quadrilateral_area_is_sum_l260_260194

noncomputable theory

open_locale classical

variables (A B C D P Q X Y Z : Type) [metric_space ℝ]

-- Rectangle property assumed
variables [is_rectangle A B C D]
variables [is_point_on_segment P A B]
variables [is_point_on_segment Q B C]

-- Conditions on intersection points
variables [intersect_at (A, Q) (D, P) X]
variables [intersect_at (A, Q) (C, P) Y]
variables [intersect_at (C, P) (D, Q) Z]

-- Area assumptions
variables (area_APX area_CQZ area_BPYQ area_DXYZ : ℝ)

-- Given known areas
variables (a b c : ℝ)
variables (h1 : area_APX = a)
variables (h2 : area_CQZ = b)
variables (h3 : area_BPYQ = c)

-- Statement to prove
theorem quadrilateral_area_is_sum (h : area_DXYZ = a + b + c) : area_DXYZ = a + b + c := 
by  
  sorry

end quadrilateral_area_is_sum_l260_260194


namespace odd_primes_less_than_2047_and_coprime_l260_260713

noncomputable def sequence_def (n : ℕ) (a : ℕ → ℕ) : ℕ := |a (n + 1) - a n|

def a1 : ℕ := 2047

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

def a2_candidates (a2 : ℕ) : Prop := 
  a2 < 2047 ∧ 
  is_prime a2 ∧ 
  a1.gcd a2 = 1

theorem odd_primes_less_than_2047_and_coprime :
  ∃ s : Finset ℕ, (∀ p ∈ s, a2_candidates p) ∧ s.card = 300 := 
by
  sorry

end odd_primes_less_than_2047_and_coprime_l260_260713


namespace density_function_lim_l260_260927

noncomputable def density_function (x : ℝ) : ℝ :=
  ∑' (n : ℕ), (Set.indicator (Set.Icc n (n + 2^(-n : ℝ))) (λ _, 1) x)

theorem density_function_lim :
  (0 = lim_x (density_function x, x → ∞) < limsup_x (density_function x, x → ∞) = 1) ∧
  (∀ᵐ (x : ℝ) (measure_theory.measure_space ℝ), tendsto (λ n, density_function (x + n)) at_top (𝓝 0)) :=
sorry

end density_function_lim_l260_260927


namespace gcd_three_numbers_l260_260414

def a : ℕ := 8650
def b : ℕ := 11570
def c : ℕ := 28980

theorem gcd_three_numbers : Nat.gcd (Nat.gcd a b) c = 10 :=
by 
  sorry

end gcd_three_numbers_l260_260414


namespace find_d_not_unique_solution_l260_260776

variable {x y k d : ℝ}

-- Definitions of the conditions
def eq1 (d : ℝ) (x y : ℝ) := 4 * (3 * x + 4 * y) = d
def eq2 (k : ℝ) (x y : ℝ) := k * x + 12 * y = 30

-- The theorem we need to prove
theorem find_d_not_unique_solution (h1: eq1 d x y) (h2: eq2 k x y) (h3 : ¬ ∃! (x y : ℝ), eq1 d x y ∧ eq2 k x y) : d = 40 := 
by
  sorry

end find_d_not_unique_solution_l260_260776


namespace alpha_plus_beta_eq_pi_div_4_l260_260131

variable {α β : ℝ}

theorem alpha_plus_beta_eq_pi_div_4 (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : tan α = 1 / 7) (h4 : tan β = 3 / 4) : α + β = π / 4 :=
by
  sorry

end alpha_plus_beta_eq_pi_div_4_l260_260131


namespace correct_params_l260_260968

variables {x y t: ℝ}
def line (x y : ℝ) : Prop := y = (7 / 4) * x - 11 / 2

def param_A : Prop := 
  ∃ t, 
    (x = -2 + 4 * t) ∧
    (y = -9 + 7 * t) ∧
    line (x) (y)

def param_B : Prop := 
  ∃ t, 
    (x = 8 + 8 * t) ∧
    (y = 3 + 14 * t) ∧
    line (x) (y)

def param_C : Prop := 
  ∃ t, 
    (x = 3 + (1 / 2) * t) ∧
    (y = - (1 / 4) + t) ∧
    line (x) (y)

def param_D : Prop := 
  ∃ t, 
    (x = 2 + 2 * t) ∧
    (y = 1 + (7 / 4) * t) ∧
    line (x) (y)

def param_E : Prop := 
  ∃ t, 
    (x = 0 + 8 * t) ∧
    (y = - (11 / 2) + (-14) * t) ∧
    line (x) (y)

theorem correct_params : param_A ∧ param_B ∧ param_C ∧ ¬ param_D ∧ ¬ param_E := by
  sorry

end correct_params_l260_260968


namespace average_book_width_l260_260895

theorem average_book_width :
  let widths := [3, 0.75, 1.5, 2.5, 0.5, 7, 8]
  (∑ w in widths, w) / widths.length = 3.32 := 
by
  sorry

end average_book_width_l260_260895


namespace find_a_value_l260_260494

variable (a b : ℝ)

def z1 : ℂ := -1 + a * complex.I
def z2 : ℂ := b - complex.I

theorem find_a_value (h1 : (z1 a) + (z2 b) ∈ ℝ) (h2 : (z1 a) * (z2 b) ∈ ℝ) : a = 1 :=
by
  sorry

end find_a_value_l260_260494


namespace number_of_odd_digits_in_base4_of_345_l260_260419

-- Define the conditions and the proof problem
theorem number_of_odd_digits_in_base4_of_345 :
  let n := 345 in
  let base4_representation := 10521 in
  let odd_digits := [1, 5, 1] in
  odd_digits.length = 3 :=
by
  -- Definitions
  let n := 345
  let base4_representation := 10521
  let odd_digits := [1, 5, 1]
  -- Expected result
  have h : odd_digits.length = 3 := by rfl
  exact h

end number_of_odd_digits_in_base4_of_345_l260_260419


namespace parabola_directrix_eq_neg2_l260_260859

-- Definitions based on conditions
def ellipse_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0

def parabola_directrix (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ ∃ x, x = -p / 2

theorem parabola_directrix_eq_neg2 (p : ℝ) (hp : p > 0) :
  (∀ (x y : ℝ), ellipse_focus 9 5 x y → parabola_directrix p x y) →
  (∃ x y : ℝ, parabola_directrix p x y → x = -2) :=
sorry

end parabola_directrix_eq_neg2_l260_260859


namespace geometric_seq_condition_l260_260872

/-- In a geometric sequence with common ratio q, sum of the first n terms S_n.
  Given q > 0, show that it is a necessary condition for {S_n} to be an increasing sequence,
  but not a sufficient condition. -/
theorem geometric_seq_condition (a1 q : ℝ) (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h1 : q > 0) : 
  (∀ n, S n < S (n + 1)) ↔ a1 > 0 :=
sorry

end geometric_seq_condition_l260_260872


namespace ratio_largest_to_sum_l260_260378

theorem ratio_largest_to_sum (n : ℕ) :
  let S := 2^15 in
  let sum := (1 - 2^15) / (1 - 2) in
  ((S : ℝ) / ((sum - S): ℝ)) = 1 :=
by
  let S := 2^15
  let sum := 2^15 - 1
  have h1 : (S : ℝ) = (2^15 : ℝ) := rfl
  have h2 : (sum : ℝ) = (2^15 - 1 : ℝ) := rfl
  rw [h1, h2]
  -- Skip proof
  sorry

end ratio_largest_to_sum_l260_260378


namespace avoid_forbidden_forest_l260_260477

def harry_coords : (ℝ × ℝ) := (15, -4)
def sandy_coords : (ℝ × ℝ) := (7, 10)
def forbidden_forest_center : (ℝ × ℝ) := ((harry_coords.1 + sandy_coords.1) / 2, (harry_coords.2 + sandy_coords.2) / 2)
def forbidden_forest_radius : ℝ := 1
def desired_meeting_point : (ℝ × ℝ) := (12, 3)

theorem avoid_forbidden_forest :
  dist forbidden_forest_center desired_meeting_point > forbidden_forest_radius := sorry

end avoid_forbidden_forest_l260_260477


namespace smallest_sum_of_abcd_l260_260255

theorem smallest_sum_of_abcd (a b c d : ℕ) (habcd : a * b * c * d = 10!) : 
  a + b + c + d ≥ 175 :=
sorry

end smallest_sum_of_abcd_l260_260255


namespace prime_for_all_k_l260_260556

theorem prime_for_all_k (n : ℤ) (h₁ : n ≥ 2)
  (h₂ : ∀ k : ℤ, 0 ≤ k ∧ k ≤ ⌊√(n / 3)⌋ → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℤ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by 
  sorry

end prime_for_all_k_l260_260556


namespace desks_per_row_calc_l260_260754

theorem desks_per_row_calc :
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  (total_desks / 4 = 6) :=
by
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  show total_desks / 4 = 6
  sorry

end desks_per_row_calc_l260_260754


namespace count_five_digit_numbers_divisible_by_5_no_repetition_l260_260844

/--
Proof that the number of five-digit numbers that are divisible by 5 and do not contain repeating digits is 5712.
-/
theorem count_five_digit_numbers_divisible_by_5_no_repetition :
  let total := 
    let last_digit_zero := 9 * 8 * 7 * 6
    let last_digit_five := 8 * 8 * 7 * 6
    last_digit_zero + last_digit_five
  in total = 5712 :=
by
  sorry

end count_five_digit_numbers_divisible_by_5_no_repetition_l260_260844


namespace circumradius_triangle_ACL_l260_260876

-- Given conditions
variables (a : ℝ) (α : ℝ)
variables (C : ∠C = 90) -- Angle C is a right angle
variables (L : Point) -- Point L on line MN

-- Theorem Statement
theorem circumradius_triangle_ACL (h1 : triangle_right_angle C)
(h2 : BC = a)
(h3 : midpoint M AB)
(h4 : midpoint N BC)
(h5 : angle_bisector A L MN)
(h6 : ∠CBL = α) :
  radius_circumcircle_ACL = a / (2 * sin(2 * α)) := sorry

end circumradius_triangle_ACL_l260_260876


namespace max_length_is_3sqrt2_l260_260805

noncomputable def max_vector_length (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : ℝ :=
  let OP₁ := (Real.cos θ, Real.sin θ)
  let OP₂ := (2 + Real.sin θ, 2 - Real.cos θ)
  let P₁P₂ := (OP₂.1 - OP₁.1, OP₂.2 - OP₁.2)
  Real.sqrt ((P₁P₂.1)^2 + (P₁P₂.2)^2)

theorem max_length_is_3sqrt2 : ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → max_vector_length θ sorry = 3 * Real.sqrt 2 := 
sorry

end max_length_is_3sqrt2_l260_260805


namespace maximum_value_achieved_l260_260691

theorem maximum_value_achieved (initial_value : ℝ) (n : ℕ) (h_init : initial_value = 1) (h_n : n = 2001) :
  ∃ f : fin n → (ℝ → ℝ), (∀ i : fin n, f i = sin ∨ f i = cos) ∧ 
  (let result := (list.range n).foldl (λ acc i, (f i) acc) initial_value in result = 1) :=
by
  sorry

end maximum_value_achieved_l260_260691


namespace non_congruent_triangles_perimeter_12_l260_260115

theorem non_congruent_triangles_perimeter_12 :
  ∃ S : finset (ℕ × ℕ × ℕ), S.card = 5 ∧ ∀ (abc ∈ S), 
  let (a, b, c) := abc in 
    a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    ∀ (abc' ∈ S), abc' ≠ abc → abc ≠ (λ t, (t.2.2, t.2.1, t.1)) abc' :=
by sorry

end non_congruent_triangles_perimeter_12_l260_260115


namespace number_of_females_l260_260882

-- Definitions
variable (F : ℕ) -- ℕ = Natural numbers, ensuring F is a non-negative integer
variable (h_male : ℕ := 2 * F)
variable (h_total : F + 2 * F = 18)
variable (h_female_pos : F > 0)

-- Theorem
theorem number_of_females (F : ℕ) (h_male : ℕ := 2 * F) (h_total : F + 2 * F = 18) (h_female_pos : F > 0) : F = 6 := 
by 
  sorry

end number_of_females_l260_260882


namespace min_value_expression_l260_260096

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem min_value_expression (a b : ℝ) (h1 : b > 0) (h2 : f a b 1 = 3) :
  ∃ x, x = (4 / (a - 1) + 1 / b) ∧ x = 9 / 2 :=
by
  sorry

end min_value_expression_l260_260096


namespace honor_students_count_l260_260982

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l260_260982


namespace inverse_of_composed_function_l260_260541

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l260_260541


namespace length_of_internal_tangent_l260_260625

theorem length_of_internal_tangent (r1 r2 : ℝ) 
    (h_r1 : r1 = 2) 
    (h_r2 : r2 = 4) 
    (h_perpendicular_tangents : true) : 
    ∃ d : ℝ, d = r1 + r2 ∧ d = 6 :=
by
  use (r1 + r2)
  split
  case left => 
    sorry
  case right => 
    sorry

end length_of_internal_tangent_l260_260625


namespace original_price_lawn_chair_l260_260340

theorem original_price_lawn_chair:
  ∀ (P : ℝ), 
    (59.95 : ℝ) = P * 0.7498436522826767 →
    P ≈ 79.93 := 
by
  sorry

end original_price_lawn_chair_l260_260340


namespace part1_part2_l260_260465

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 9 → x ∈ Set.Icc (-2 : ℝ) 4 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) →
  (a ∈ Set.Icc (19 / 4) (7 : ℝ)) :=
sorry

end part1_part2_l260_260465


namespace ratio_of_times_l260_260329

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem ratio_of_times (T_A T_B : ℝ) (hA : T_A = 12) 
  (hCombinedWorkRate : work_rate T_A + work_rate T_B = 0.25) :
  T_B = 6 → T_B / T_A = 1 / 2 :=
by
  intros hTB
  rw [hA, hTB]
  norm_num
  sorry

end ratio_of_times_l260_260329


namespace value_of_4d_minus_c_l260_260252

-- Conditions
def quadratic_expr : ℝ → ℝ :=
  λ x, x^2 - 18 * x + 72
def c : ℕ := 12 -- given by factorization (x - 12)(x - 6)
def d : ℕ := 6  -- given by factorization (x - 12)(x - 6)

-- Condition that c > d
def c_gt_d : c > d :=
  by decide

-- The final statement to prove
theorem value_of_4d_minus_c : 4 * d - c = 12 :=
  sorry

end value_of_4d_minus_c_l260_260252


namespace common_point_through_PQ_l260_260193

open Real

theorem common_point_through_PQ
  (c : ℝ) (hc : 0 < c) : 
  ∃ R : ℝ, ∀ X Y P Q : ℝ × ℝ, 
    (X ∈ {p : ℝ × ℝ | p.1 = 1}) → 
    (Y ∈ {p : ℝ × ℝ | p.1 = 1}) → 
    (X ≠ Y) → 
    (X.2 * Y.2 = c / 4) → 
    (P, Q ∈ ({a : ℝ × ℝ | dist a (0, 0) = 1})) → 
    (∃ t : ℝ, (X + t * (B - X)).1 = P) → 
    (∃ t : ℝ, (Y + t * (B - Y)).1 = Q) →
    (∃ R : ℝ, (P ≠ Q) → (∃ x : ℝ, (x, 0) = R)) :=
by
  sorry


end common_point_through_PQ_l260_260193


namespace chimney_bricks_l260_260370

theorem chimney_bricks :
  ∃ (h : ℕ),
    (h / 8 + h / 12 - 12) * 6 = h :=
begin
  sorry
end

#eval chimney_bricks -- Here 288 should be the solution.

end chimney_bricks_l260_260370


namespace acid_solution_l260_260280

theorem acid_solution (m x : ℝ) (h_m_gt_30 : m > 30) :
  let initial_acid := m * (m / 100)
      final_acid := (m - 15) * (m + x) / 100
  in initial_acid = final_acid → x = 15 * m / (m - 15) :=
by {
  let initial_acid := m * (m / 100),
  let final_acid := (m - 15) * (m + x) / 100,
  assume h : initial_acid = final_acid,
  sorry
}

end acid_solution_l260_260280


namespace alice_total_distance_correct_l260_260011

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end alice_total_distance_correct_l260_260011


namespace sqrt_five_eq_l260_260583

theorem sqrt_five_eq (m n a b c d : ℤ)
  (h : m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5)) :
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) := by
  sorry

end sqrt_five_eq_l260_260583


namespace range_a_l260_260433

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*x + a
noncomputable def g (a x : ℝ) : ℝ := x^2 + (2*a - 3)*x + 1

theorem range_a (a : ℝ) :
  ((∃ x ∈ Set.Ioo 1 2, f a x = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0) → False →
  ((∃ x ∈ Set.Ioo 1 2, f a x = 0) ∨ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0) →
  a ∈ Set.Icc (0 : ℝ) 1 ∨ a ∈ Set.Icc (5/2 : ℝ) ∞ ∨ a ∈ Set.Icc 0 (1/2) ∨ a ∈ Set.Icc (-∞) (0 : ℝ) :=
by
  sorry

end range_a_l260_260433


namespace common_ratio_of_gp_l260_260498

variable (a r : ℝ)

def Sn (n : ℕ) (a r : ℝ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem common_ratio_of_gp
  (h : Sn 6 a r / Sn 3 a r = 217) :
  r = 6 :=
  by sorry

end common_ratio_of_gp_l260_260498


namespace max_value_of_g_l260_260385

def g : ℕ → ℕ
| n := if n < 15 then n + 15 else g (n - 3)

theorem max_value_of_g : ∀ n : ℕ, g n ≤ 29 :=
begin
  sorry
end

example : ∃ n : ℕ, g n = 29 :=
begin
  use 14,
  -- prove that g(14) = 29
  sorry
end

end max_value_of_g_l260_260385


namespace larger_factor_of_lcm_l260_260967

-- Define the conditions in Lean
def hcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

noncomputable def A := 414
def hcf_val := 23
def factor_one := 13

-- Define the question and answer in Lean
theorem larger_factor_of_lcm (B : ℕ) (X : ℕ) (hcf A B = hcf_val) (lcm A B = hcf_val * factor_one * X) : X = 18 := by 
sorry

end larger_factor_of_lcm_l260_260967


namespace digit_property_l260_260711

/-- 
  A sealed envelope contains a card with a digit from 0 to 9 on it.
  Four of the following statements are true, and the other one is false.
  I. The digit is not 0.
  II. The digit is an even number.
  III. The digit is 5.
  IV. The digit is not 6.
  V. The digit is less than 7.
  
  We need to prove that statement II (the digit is an even number) must necessarily be false.
-/
theorem digit_property : 
  ∃ (digit : ℕ), digit ∈ (0:9).finset ∧ 
  ((digit ≠ 0) ↔ (true, true, true, true, false)) ∧
  ((digit % 2 = 0) ↔ (true, true, false, true, true)) ∧
  ((digit = 5) ↔ (true, false, false, false, false)) ∧
  ((digit ≠ 6) ↔ (true, true, false, false, false)) ∧
  ((digit < 7) ↔ (true, true, true, false, true)) ∧
  (statement_false (digit % 2 = 0))
:= sorry

end digit_property_l260_260711


namespace AE_div_BC_eq_sqrt3_l260_260152

theorem AE_div_BC_eq_sqrt3 (s : ℝ) :  
  let BC := s,
  let altitude := s * sqrt 3 / 2,
  let AE := 2 * altitude in
  AE / BC = sqrt 3 :=
by
  sorry

end AE_div_BC_eq_sqrt3_l260_260152


namespace boys_neither_happy_nor_sad_l260_260215

theorem boys_neither_happy_nor_sad : 
  (∀ children total happy sad neither boys girls happy_boys sad_girls : ℕ,
    total = 60 →
    happy = 30 →
    sad = 10 →
    neither = 20 →
    boys = 19 →
    girls = 41 →
    happy_boys = 6 →
    sad_girls = 4 →
    (boys - (happy_boys + (sad - sad_girls))) = 7) :=
by
  intros children total happy sad neither boys girls happy_boys sad_girls
  sorry

end boys_neither_happy_nor_sad_l260_260215


namespace area_solution_l260_260387

noncomputable def area_of_figure (x y : ℝ) : ℝ :=
  if (9 - x^2 - y^2 - 2y ≥ 0 ∧ y ≤ 0)
  then 10 * Real.pi + 3 - 10 * Real.atan 3
  else 0

theorem area_solution :
  ∀ x y : ℝ, (|9 - x^2 - y^2 - 2y| + |-2y| = 9 - x^2 - y^2 - 4y) →
             (9 - x^2 - y^2 - 2y ≥ 0 ∧ y ≤ 0) →
             area_of_figure x y = 10 * Real.pi + 3 - 10 * Real.atan 3 :=
by
  intros x y h1 h2
  sorry

end area_solution_l260_260387


namespace probability_factor_less_than_8_l260_260305

/--
The probability that a randomly drawn positive factor of 90 is less than 8 is 5/12.
-/
theorem probability_factor_less_than_8 :
  let n := 90
  let factors_less_than_8 := {d : ℕ | d ∣ n ∧ d < 8}
  let total_factors := {d : ℕ | d ∣ n}
  (factors_less_than_8.to_finset.card : ℚ) / total_factors.to_finset.card = 5 / 12 :=
by
  sorry

end probability_factor_less_than_8_l260_260305


namespace problem_1_proof_problem_2_proof_problem_3_proof_l260_260795

noncomputable def problem_1 (α : ℝ) (R : ℝ) : ℝ :=
l := 10 * (Real.pi / 3)
l

theorem problem_1_proof (α : ℝ) (R : ℝ) (h1 : α = 60) (h2 : R = 10) : 
l = (10 * Real.pi / 3) :=
by {
  rw h1,
  rw h2,
  exact l
}

noncomputable def problem_2 (P : ℝ) (R : ℝ) : ℝ :=
let α : ℝ := 2
α

theorem problem_2_proof (R : ℝ) (P : ℝ) (h1 : P = 20) : 
α = 2 :=
by {
  rw h1,
  exact α
}

noncomputable def problem_3 (α : ℝ) (R : ℝ) : ℝ :=
let S_segment : ℝ := ((1 / 2) * ((2 * Real.pi) / 3) * 2) - ((1 / 2) * (2 ^ 2) * (Real.sin (Real.pi / 3)))
S_segment

theorem problem_3_proof (α : ℝ) (R : ℝ) (h1 : α = Real.pi / 3) (h2 : R = 2) : 
S_segment = ((2 * Real.pi) / 3) - Real.sqrt 3 :=
by {
  rw h1,
  rw h2,
  exact S_segment
}

end problem_1_proof_problem_2_proof_problem_3_proof_l260_260795


namespace probability_exactly_one_zero_l260_260825

noncomputable def f (x a b : ℝ) : ℝ := 1/2 * x^3 + a * x - b

lemma increasing_condition {a b : ℝ} (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → f x a b ≥ f (-1) a b := sorry

lemma zero_in_interval {a b : ℝ} (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) :
  (f (-1) a b) * (f 1 a b) ≤ 0 :=
begin
  have h_minus1 := -1/2 + a - b,
  have h_1 := 1/2 + a - b,
  calc
    (h_minus1) * (h_1) ≤ 0 : sorry
end

theorem probability_exactly_one_zero :
  ∃ p : ℝ, p = 7 / 8 := sorry

end probability_exactly_one_zero_l260_260825


namespace union_of_sets_l260_260071

def setA : set ℝ := { x | x^2 + x - 2 < 0 }
def setB : set ℝ := { x | x > 0 }
def unionAB : set ℝ := { x | x > -2 }

theorem union_of_sets : setA ∪ setB = unionAB :=
sorry

end union_of_sets_l260_260071


namespace max_runs_ideal_case_l260_260317

-- Defining the conditions:
def max_runs_per_ball : ℕ := 6
def balls_per_over : ℕ := 6
def overs_in_match : ℕ := 20

-- Defining the maximum runs calculation:
def max_runs_per_over : ℕ := max_runs_per_ball * balls_per_over
def max_runs_in_match : ℕ := max_runs_per_over * overs_in_match

-- Statement of the mathematical proof:
theorem max_runs_ideal_case (max_runs_per_ball = 6) 
                            (balls_per_over = 6) 
                            (overs_in_match = 20) : 
                            max_runs_in_match = 720 := by
  -- Proof to be filled
  sorry

end max_runs_ideal_case_l260_260317


namespace symmetric_point_example_l260_260955

/-- Definition of a point in the 2D plane. -/
structure Point :=
(x : ℝ)
(y : ℝ)

/-- Definition of a line in the 2D plane with the general form Ax + By + C = 0. -/
structure Line :=
(A : ℝ)
(B : ℝ)
(C : ℝ)

/-- Symmetric point about a line. -/
def symmetric_point (M : Point) (l : Line) : Point :=
  let x := 2 * l.A * l.C / (l.A^2 + l.B^2) + 2 * M.x * l.B / (l.A^2 + l.B^2) - M.x
  let y := 2 * l.B * (-l.C) / (l.A^2 + l.B^2) + 2 * M.y * l.A / (l.A^2 + l.B^2) - M.y
  { x := x, y := y }

/-- Test problem: Prove that the symmetric point of a given point about a given line is another given point. -/
theorem symmetric_point_example :
  let M := {x := -1, y := 1} : Point
  let l := {A := 1, B := -1, C := -1} : Line
  symmetric_point M l = {x := 2, y := -2} :=
by
  sorry

end symmetric_point_example_l260_260955


namespace not_all_conditions_ensure_congruence_l260_260145

variables {A B C A' B' C' : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space A'] [metric_space B'] [metric_space C']
variables {AB BC AC A'B' B'C' A'C' : ℝ}
variables {angleA angleB angleC angleA' angleB' angleC' : ℝ}

-- Sum of square sides to determine congruence
def congruent_triangles (AB BC AC A'B' B'C' A'C' : ℝ) (angleA angleB angleC angleA' angleB' angleC' : ℝ) : Prop :=
    (AB = A'B') ∧ (BC = B'C') ∧ (AC = A'C') ∧
    (angleA = angleA') ∧ (angleB = angleB') ∧ (angleC = angleC')

-- Prove that set of conditions (1, 3, 5) cannot ensure congruence
theorem not_all_conditions_ensure_congruence :
    ¬ congruent_triangles AB BC AC A'B' B'C' A'C' angleA angleB angleC angleA' angleB' angleC' :=
begin
    -- Specific conditions that do not ensure congruence
    have h1 : AB = A'B' := sorry,
    have h3 : AC = A'C' := sorry,
    have h5 : angleB = angleB' := sorry,
    sorry
end

end not_all_conditions_ensure_congruence_l260_260145


namespace inclination_angle_l260_260612

theorem inclination_angle (n : ℝ × ℝ) (h : n = (1, 1)) : 
  let θ := Real.arctan (n.2 / n.1) in Real.toDegrees θ = 45 :=
by
  -- Assuming n = (1, 1)
  have : n = (1, 1) := h
  sorry

end inclination_angle_l260_260612


namespace max_regions_by_11_rays_l260_260518

theorem max_regions_by_11_rays : 
  let n := 11 
  in (n^2 - n + 2) / 2 = 56 :=
by
  let n := 11
  sorry

end max_regions_by_11_rays_l260_260518


namespace divisors_of_64n4_l260_260033

theorem divisors_of_64n4 (n : ℕ) (hn : 0 < n) (hdiv : ∃ d, d = (120 * n^3) ∧ d.divisors.card = 120) : (64 * n^4).divisors.card = 375 := 
by 
  sorry

end divisors_of_64n4_l260_260033


namespace problem_statement_l260_260535

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem problem_statement :
  ∃ (A B C : ℝ × ℝ), 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (∃ a b : ℝ, A = (a, 2 * a^2) ∧ B = (b, 2 * b^2) ∧ a ≠ b) ∧ 
  (B.2 = A.2) ∧ 
  (C.1 = 0 ∧ ∃ y : ℝ, C = (0, y)) ∧ 
  (∃ h : ℝ, h = (A.1 - B.1).abs * ((y - 2 * (a^2)).abs) / 2 ∧ h = 1024) ∧ 
  sum_of_digits (nat_abs y) = 13 := 
sorry

end problem_statement_l260_260535


namespace N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l260_260576

open Nat -- Natural numbers framework

-- Definitions for game conditions would go here. We assume them to be defined as:
-- structure GameCondition (N : ℕ) :=
-- (players_take_turns_to_circle_numbers_from_1_to_N : Prop)
-- (any_two_circled_numbers_must_be_coprime : Prop)
-- (a_number_cannot_be_circled_twice : Prop)
-- (player_who_cannot_move_loses : Prop)

inductive Player
| first
| second

-- Definitions indicating which player wins for a given N
def first_player_wins (N : ℕ) : Prop := sorry
def second_player_wins (N : ℕ) : Prop := sorry

-- For N = 10
theorem N_10_first_player_wins : first_player_wins 10 := sorry

-- For N = 12
theorem N_12_first_player_wins : first_player_wins 12 := sorry

-- For N = 15
theorem N_15_second_player_wins : second_player_wins 15 := sorry

-- For N = 30
theorem N_30_first_player_wins : first_player_wins 30 := sorry

end N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l260_260576


namespace daily_rate_problem_l260_260532

noncomputable def daily_rate : ℝ := 126.19 -- Correct answer

theorem daily_rate_problem
  (days : ℕ := 14)
  (pet_fee : ℝ := 100)
  (service_fee_rate : ℝ := 0.20)
  (security_deposit : ℝ := 1110)
  (deposit_rate : ℝ := 0.50)
  (x : ℝ) : x = daily_rate :=
by
  have total_cost := days * x + pet_fee + service_fee_rate * (days * x)
  have total_cost_with_fees := days * x * (1 + service_fee_rate) + pet_fee
  have security_deposit_cost := deposit_rate * total_cost_with_fees
  have eq_security : security_deposit_cost = security_deposit := sorry
  sorry

end daily_rate_problem_l260_260532


namespace calculate_f_g_of_1_l260_260135

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem calculate_f_g_of_1 : f (g 1) = 39 :=
by
  -- Enable quick skippable proof with 'sorry'
  sorry

end calculate_f_g_of_1_l260_260135


namespace lindsey_exercise_bands_l260_260569

theorem lindsey_exercise_bands (x : ℕ) 
  (h1 : ∀ n, n = 5 * x) 
  (h2 : ∀ m, m = 10 * x) 
  (h3 : ∀ d, d = m + 10) 
  (h4 : d = 30) : 
  x = 2 := 
by 
  sorry

end lindsey_exercise_bands_l260_260569


namespace max_value_of_expression_l260_260832

noncomputable def expression (x y z : ℝ) := sin (x - y) + sin (y - z) + sin (z - x)

theorem max_value_of_expression :
  ∀ x y z ∈ set.Icc (0 : ℝ) (real.pi / 2), expression x y z ≤ real.sqrt 2 - 1 :=
sorry

end max_value_of_expression_l260_260832


namespace least_three_digit_multiple_of_8_l260_260302

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l260_260302


namespace max_l_l260_260179

open Finset

noncomputable def l (σ : Perm (Fin n)) : ℕ :=
Finset.min' (univ.image (λ i : Fin (n - 1), abs (σ (i + 1) - σ i))) sorry

theorem max_l (n : ℕ) (hn : 2 ≤ n) : 
  ∀ σ : Perm (Fin n), l σ ≤ 2 :=
  sorry

end max_l_l260_260179


namespace min_flight_routes_l260_260702

theorem min_flight_routes (k : ℕ) (a : Fin k → ℕ) :
  let n := (Finset.univ : Finset (Fin k)).sum a in
  (n ^ (k - 2)) * ∏ i, (n - a i) ^ (a i - 1) = 
  (n ^ (k-2)) * ∏ i in Finset.univ, (n - a i) ^ (a i - 1) :=
  sorry

end min_flight_routes_l260_260702


namespace angle_FEA_45_l260_260203

open EuclideanGeometry

-- Define the square ABCD with the required properties
variables {A B C D E F : Point}
variables [is_square A B C D] [is_midpoint E B D]
variable [is_intersection F (line_through C E) (line_through A D)]

theorem angle_FEA_45 :
  angle F E A = 45 := 
  sorry

end angle_FEA_45_l260_260203


namespace maximum_sine_sum_l260_260834

open Real

theorem maximum_sine_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : x ≤ π / 2) (hz : 0 ≤ y) (hw : y ≤ π / 2) (hv : 0 ≤ z) (hu : z ≤ π / 2) :
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x y z : ℝ, 0 ≤ x → x ≤ π / 2 → 0 ≤ y → y ≤ π / 2 → 0 ≤ z → z ≤ π / 2 → 
  sin (x - y) + sin (y - z) + sin (z - x) ≤ M :=
by
  sorry

end maximum_sine_sum_l260_260834


namespace work_days_for_A_l260_260850

/-- If A is thrice as fast as B and together they can do a work in 15 days, A alone can do the work in 20 days. -/
theorem work_days_for_A (Wb : ℕ) (Wa : ℕ) (H_wa : Wa = 3 * Wb) (H_total : (Wa + Wb) * 15 = Wa * 20) : A_work_days = 20 :=
by
  sorry

end work_days_for_A_l260_260850


namespace inverse_of_composed_function_l260_260540

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l260_260540


namespace range_of_a_l260_260130

theorem range_of_a (m a : ℝ) (h1 : m < a) (h2 : m ≤ -1) : a > -1 :=
by sorry

end range_of_a_l260_260130


namespace inserted_threes_divisible_by_19_l260_260225

theorem inserted_threes_divisible_by_19 (n : ℕ) : ∃ k, ∃ m : ℕ, m >= 0 ∧ 
    (k = 120 * 10 ^ n + ∑ i in (Finset.range m), 3 * 10 ^ (i + 2) + 8) ∧
    19 ∣ k := sorry

end inserted_threes_divisible_by_19_l260_260225


namespace equilateral_octagon_side_length_l260_260874

theorem equilateral_octagon_side_length (WZ ZY AW BZ k m n : ℝ) 
(h1 : WZ = 10) (h2 : ZY = 15) (h3 : AW = BZ) (h4 : AW < 5) 
(h5 : ∀(A B C D E F G H : ℝ), equilateral Octagon A B C D E F G H) 
(h6 : n ≠ 0 ∧ ∀ (p : ℕ), prime p → ¬(p^2 ∣ n)) : 
  k + m + n = 41 := 
sorry

end equilateral_octagon_side_length_l260_260874


namespace complex_modulus_l260_260438

open Complex

theorem complex_modulus (z : ℂ) (h : (z - 1) * (2 + I) = 5 * I) : abs (conj z + I) = Real.sqrt 5 :=
sorry

end complex_modulus_l260_260438


namespace parabola_intersects_x_axis_l260_260082

theorem parabola_intersects_x_axis (a c : ℝ) (h : ∃ h : ℝ, h = -1 ∧ (a * h^2 + h + c = 0)) : a + c = 1 :=
by
  obtain ⟨h, ⟨h_eq, H⟩⟩ := h
  have : h = -1 := h_eq
  rw [this, neg_sq] at H
  simp at H
  rw add_comm at H
  exact H
where neg_sq is by simp only [neg_sq, one_mul] 

end parabola_intersects_x_axis_l260_260082


namespace triangle_CKB_area_l260_260219

theorem triangle_CKB_area (A B C K : Point) (b : ℝ) (β : ℝ) 
  (hABC_right : angle A B C = 90) 
  (hAC_diameter : ∃ circle, Circle.circleContains circle (Line.line A C) ∧ diameter circle = b)
  (hCircleIntersects : ∃ circle, Circle.circleContains circle (Line.line A B) ∧ intersects circle K)
  (hAC_b : distance A C = b) 
  (hAngle_ABC: angle A B C = β) :
  areaTriangle C K B = (1/2) * b^2 * (Real.cos β * Real.cos β * Real.cot β) :=
by
  sorry

end triangle_CKB_area_l260_260219


namespace M_eq_N_l260_260101

def M : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = (5/6) * Real.pi + 2 * k * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = -(7/6) * Real.pi + 2 * k * Real.pi}

theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l260_260101


namespace monotonic_increasing_iff_monotonic_decreasing_on_interval_l260_260191

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 - a * x - 1

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ a ≤ 0 :=
by 
  sorry

theorem monotonic_decreasing_on_interval (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → ∀ y : ℝ, -1 < y ∧ y < 1 → x < y → f y a < f x a) ↔ 3 ≤ a :=
by 
  sorry

end monotonic_increasing_iff_monotonic_decreasing_on_interval_l260_260191


namespace parallelogram_area_l260_260410

-- Definitions for given conditions
def base : ℝ := 24
def slant_height : ℝ := 26
def angle_deg : ℝ := 40
def cos40 := Real.cos (angle_deg * Real.pi/180)
def height := cos40 * slant_height

-- Statement to prove the area of the parallelogram
theorem parallelogram_area : 
  abs (base * height - 478) < 0.01 := 
by 
  sorry

end parallelogram_area_l260_260410


namespace polynomial_remainder_l260_260420

theorem polynomial_remainder (P Q : Polynomial ℤ) (n k : ℤ)
  (hP : P = \(x => x^(6 * n) + x^(5 * n) + x^(4 * n) + x^(3 * n) + x^(2 * n) + x^n + 1\))
  (hQ : Q = \(x => x^6 + x^5 + x^4 + x^3 + x^2 + x + 1\))
  (h_n_multiple_7 : n = 7 * k) :
  (P % Q) = 7 :=
sorry

end polynomial_remainder_l260_260420


namespace true_statements_among_propositions_l260_260364

-- Definitions of the conditions
def quadrilateral_with_equal_diagonals_bisect_each_other (Q : Type) (is_quadrilateral : Q → Prop) (diags_equal_bisect : Q → Prop) : Prop :=
  ∀ q : Q, is_quadrilateral q → diags_equal_bisect q → True

def quadrilateral_with_perpendicular_diagonals (Q : Type) (is_quadrilateral : Q → Prop) (diags_perpendicular : Q → Prop) : Prop :=
  ∀ q : Q, is_quadrilateral q → diags_perpendicular q → True

def quadrilateral_with_bisecting_diagonals (Q : Type) (is_quadrilateral : Q → Prop) (diags_bisect : Q → Prop) : Prop :=
  ∀ q : Q, is_quadrilateral q → diags_bisect q → True

def rhombus_with_equal_diagonals (Q : Type) (is_rhombus : Q → Prop) (diags_equal : Q → Prop) : Prop :=
  ∀ q : Q, is_rhombus q → diags_equal q → True

-- Proof problem to show the mathematical equivalence for the set of true statements
theorem true_statements_among_propositions (Q : Type)
  (is_quadrilateral : Q → Prop) (is_rhombus : Q → Prop)
  (diags_equal_bisect : Q → Prop) (diags_perpendicular : Q → Prop)
  (diags_bisect : Q → Prop) (diags_equal : Q → Prop)
  (rectangle : Q → Prop) (parallelogram : Q → Prop) (square : Q → Prop) : Prop :=
  (quadrilateral_with_equal_diagonals_bisect_each_other Q is_quadrilateral diags_equal_bisect →
   rectangle Q) ∧
  (¬ quadrilateral_with_perpendicular_diagonals Q is_quadrilateral diags_perpendicular → 
   ¬ is_rhombus Q) ∧
  (quadrilateral_with_bisecting_diagonals Q is_quadrilateral diags_bisect →
   parallelogram Q) ∧
  (rhombus_with_equal_diagonals Q is_rhombus diags_equal →
   square Q) :=
sorry

end true_statements_among_propositions_l260_260364


namespace area_ratio_of_triangles_l260_260515

-- Definitions of points and segments
structure Trapezium :=
  (A B C D E F P Q : ℝ × ℝ)
  (AB_length : dist A B = 6)
  (CD_length : dist C D = 9)
  (height : abs (A.snd - C.snd) = 4 ∧ abs (B.snd - D.snd) = 4)
  (ratio_BE_EF_FD : ∃ k : ℝ, 0 < k ∧ dist B E = 2 * k ∧ dist E F = 3 * k ∧ dist F D = 5 * k)
  (intersections_PQ : ∃ P Q : ℝ × ℝ, line_through A E P ∧ line_through A F Q ∧ on_line_segment B C P ∧ on_line_segment B C Q)

-- Theorem statement
theorem area_ratio_of_triangles (trapezium : Trapezium) :
  ratio (triangle_area trapezium.A trapezium.P trapezium.B) (triangle_area trapezium.B trapezium.P trapezium.C) = 1 :=
sorry

end area_ratio_of_triangles_l260_260515


namespace regression_y_value_l260_260064

theorem regression_y_value (b : ℝ) (x : ℝ) (y : ℝ) 
  (mean_x : ℝ) (mean_y : ℝ) 
  (reg_eq : y = b * x + 0.2)
  (mean_x_value : mean_x = 4)
  (mean_y_value : mean_y = 5) :
  (let b := (mean_y - 0.2) / mean_x in b * 2 + 0.2 = 2.6) :=
by
  sorry

end regression_y_value_l260_260064


namespace inequality_proof_l260_260182

variable {f : ℝ → ℝ}
variable {a b c : ℝ}

theorem inequality_proof 
  (h_deriv : ∀ x > 0, has_deriv_at ℝ (deriv (deriv (deriv f))) f x)
  (h_pos : ∀ x > 0, deriv (deriv (deriv f)) x > 0)
  (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) :
  f (a^2 + b^2 + c^2) + 2 * f (a*b + b*c + c*a) 
  >= f (a^2 + 2*b*c) + f (b^2 + 2*c*a) + f (c^2 + 2*a*b) := 
sorry

end inequality_proof_l260_260182


namespace isosceles_right_triangle_area_parabola_l260_260365

noncomputable def area_of_isosceles_right_triangle_parabola : ℝ :=
  let y_squared_eq_4x := λ (x y : ℝ), y^2 = 4 * x in
  let inscribed_in_parabola := ∀ {x₁ y₁ x₂ y₂ : ℝ}, 
    y_squared_eq_4x x₁ y₁ ∧ y_squared_eq_4x x₂ y₂ in
  let isosceles_right_triangle := 
    ∀ {x₁ y₁ x₂ y₂ : ℝ},
    (x₁ - 0)^2 + (y₁ - 0)^2 = (x₂ - 0)^2 + (y₂ - 0)^2 ∧ 
    y₁ = y₂ ∨ y₁ = -y₂ in
  let oa_perp_ob := ∀ {x₁ y₁ x₂ y₂ : ℝ},
    (x₁, 0) = (0, y₁ * y₁ / 4) ∧ (0, y₁ * y₁ / 4) = (x₂, 0) in
  1 / 2 * 4 * 8

theorem isosceles_right_triangle_area_parabola : 
  area_of_isosceles_right_triangle_parabola = 16 :=
by sorry

end isosceles_right_triangle_area_parabola_l260_260365


namespace negation_of_square_zero_l260_260622

variable {m : ℝ}

def is_positive (m : ℝ) : Prop := m > 0
def square_is_zero (m : ℝ) : Prop := m^2 = 0

theorem negation_of_square_zero (h : ∀ m, is_positive m → square_is_zero m) :
  ∀ m, ¬ is_positive m → ¬ square_is_zero m := 
sorry

end negation_of_square_zero_l260_260622


namespace subtracted_amount_l260_260330

theorem subtracted_amount (A N : ℝ) (h₁ : N = 200) (h₂ : 0.95 * N - A = 178) : A = 12 :=
by
  sorry

end subtracted_amount_l260_260330


namespace gabby_fruit_total_l260_260044

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end gabby_fruit_total_l260_260044


namespace seating_possible_around_table_l260_260891

theorem seating_possible_around_table (n : ℕ) : 
  ∃ (hamiltonian_cycle_decomposition : list (list ℕ)), 
    (∀ cycle ∈ hamiltonian_cycle_decomposition, is_hamiltonian_cycle cycle (2 * n + 1)) ∧
    edge_disjoint_decomposition hamiltonian_cycle_decomposition (complete_graph_edges (2 * n + 1)) := 
sorry

-- Definitions for Hamiltonian cycle and edge decomposition:

def is_hamiltonian_cycle (cycle : list ℕ) (num_vertices : ℕ) : Prop := 
  cycle.length = num_vertices + 1 ∧ 
  (∀ (i : ℕ), i < num_vertices → cycle.nth i ≠ cycle.nth ((i + 1) % num_vertices)) ∧ 
  cycle.nth 0 = cycle.nth num_vertices

def complete_graph_edges (num_vertices : ℕ) : list (ℕ × ℕ) := 
  (list.fin_range num_vertices).product (list.fin_range num_vertices) -- this generates all possible pairs (a (vertex) × another vertex); need to add a filter ≠ self

def edge_disjoint_decomposition (decomposition : list (list ℕ)) (edges : list (ℕ × ℕ)) : Prop :=
  (∀ cycle ∈ decomposition, pairwise_disjoint_edges cycle edges) ∧
  union_of_all_edges decomposition = edges

def union_of_all_edges (cycles : list (list ℕ)) : list (ℕ × ℕ) := 
  cycles.join

def pairwise_disjoint_edges (cycle : list ℕ) (edges : list (ℕ × ℕ)) : Prop := 
  cycle.pairwise (≠)

end seating_possible_around_table_l260_260891


namespace option_A_is_not_valid_l260_260667

universe u

def is_valid_mapping {α β : Type u} (A : set α) (B : set β) (f : α → β) : Prop :=
∀ x ∈ A, f x ∈ B

def option_A : Prop :=
¬ is_valid_mapping (set.univ : set ℝ) (set.univ : set ℝ) (λ x : ℝ, x⁻¹)

def option_B : Prop :=
is_valid_mapping (set.univ : set ℝ) (set.univ : set ℝ) (abs : ℝ → ℝ)

def option_C : Prop :=
is_valid_mapping (set.Ioi 0) (set.univ : set ℝ) (λ x : ℝ, x^2)

def option_D : Prop :=
is_valid_mapping ({x : ℝ | x < π / 2 ∧ x > 0} : set ℝ) (set.Ioo 0 1) (Real.sin)

theorem option_A_is_not_valid : option_A :=
by
  unfold option_A is_valid_mapping
  intro h
  specialize h 0 ()
  simp at h
  exact h

end option_A_is_not_valid_l260_260667


namespace find_vector_op_l260_260561

-- Definition of the vectors and their properties
variables (a b : ℝ) (θ : ℝ)

-- Conditions
def unit_vector_a : Prop := abs a = 1
def magnitude_b : Prop := abs b = real.sqrt 2
def magnitude_diff : Prop := abs (a - b) = 1
def angle_between_vectors (θ : ℝ) : Prop := cos θ = (real.sqrt 2) / 2 ∧ sin θ = (real.sqrt 2) / 2

-- The operation definition
def vector_op (a b θ: ℝ) : ℝ := abs (a * sin θ + b * cos θ)

-- The theorem statement with the conditions and result
theorem find_vector_op
  (ha : unit_vector_a a)
  (hb : magnitude_b b)
  (hd : magnitude_diff a b)
  (ht : angle_between_vectors θ) :
  vector_op a b θ = (real.sqrt 10) / 2 :=
sorry

end find_vector_op_l260_260561


namespace unit_square_points_dist_greater_half_l260_260879

theorem unit_square_points_dist_greater_half : 
  ∀ s : set (ℝ × ℝ), (∀ p1 p2 ∈ s, p1 ≠ p2 → dist p1 p2 > 0.5) → s.finite ∧ s.card ≤ 8 := 
by 
  sorry

end unit_square_points_dist_greater_half_l260_260879


namespace digit_makes_57A2_divisible_by_9_l260_260649

theorem digit_makes_57A2_divisible_by_9 (A : ℕ) (h : 0 ≤ A ∧ A ≤ 9) : 
  (5 + 7 + A + 2) % 9 = 0 ↔ A = 4 :=
by
  sorry

end digit_makes_57A2_divisible_by_9_l260_260649


namespace mean_combined_rel_n_m_l260_260621

-- Definitions based on the conditions
variables {n m : ℕ}
variables {x y : ℝ}
variables (α : ℝ) (x̄ ȳ z̄ : ℝ)
variables (hx : x̄ ≠ ȳ)
variables (hα : 0 < α ∧ α < 1 / 2)

-- Mean definitions for the samples
noncomputable def sample_mean_x (n : ℕ) (x : ℝ) : ℝ := x̄
noncomputable def sample_mean_y (m : ℕ) (y : ℝ) : ℝ := ȳ

-- Mean of the combined sample
noncomputable def combined_mean (n m : ℕ) (x y : ℝ) : ℝ := α * x̄ + (1 - α) * ȳ

-- Lean statement
theorem mean_combined_rel_n_m
    (hx : x̄ ≠ ȳ) (hα : 0 < α ∧ α < 1 / 2)
    (hn : sample_mean_x n x = x̄) (hm : sample_mean_y m y = ȳ) :
    combined_mean n m x y = (n • x̄ + m • ȳ) / (n + m) → n < m :=
begin
  sorry
end

end mean_combined_rel_n_m_l260_260621


namespace polar_to_rect_eq_circle_line_intersection_distance_l260_260162

-- Definitions for parametric equation of line l and polar equation of circle C
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (2 + (real.sqrt 2) / 2 * t, 1 + (real.sqrt 2) / 2 * t)

def polar_circle (rho theta : ℝ) : Prop :=
  rho = 6 * real.cos theta

-- Problem (1): Convert polar equation to rectangular coordinates and prove the equation.
theorem polar_to_rect_eq :
  ∀ (rho theta : ℝ), polar_circle rho theta → (∀ x y : ℝ, x = rho * real.cos theta ∧ y = rho * real.sin theta → (x - 3) ^ 2 + y ^ 2 = 9) :=
  sorry

-- Problem (2): Given the intersection points, prove the distance between points A and B.
theorem circle_line_intersection_distance :
  ∀ t1 t2 : ℝ,
  ((parametric_line t1).fst - 3) ^ 2 + (parametric_line t1).snd ^ 2 = 9 ∧
  ((parametric_line t2).fst - 3) ^ 2 + (parametric_line t2).snd ^ 2 = 9 →
  real.abs (t1 - t2) = 2 * real.sqrt 7 :=
  sorry

end polar_to_rect_eq_circle_line_intersection_distance_l260_260162


namespace f_15_equals_227_l260_260488

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem f_15_equals_227 : f 15 = 227 := by
  sorry

end f_15_equals_227_l260_260488


namespace cone_height_l260_260632

-- Define the given conditions and problem statement
def surface_area (r l : ℝ) : ℝ := π * r * (r + l)
def central_angle : ℝ := 120 * π / 180 -- 120 degrees in radians

theorem cone_height (r l h : ℝ) (h_cond : 0 < r)
  (surface_area_eq : surface_area r l = π)
  (slant_height_eq : l = 3 * r)
  (height_eq : h = real.sqrt (l^2 - r^2)) :
  h = real.sqrt 2 :=
by
  -- The proof will be filled in here
  sorry

end cone_height_l260_260632


namespace total_birds_in_tree_l260_260593

theorem total_birds_in_tree (bluebirds cardinals swallows : ℕ) 
  (h1 : swallows = 2) 
  (h2 : swallows = bluebirds / 2) 
  (h3 : cardinals = 3 * bluebirds) : 
  swallows + bluebirds + cardinals = 18 := 
by 
  sorry

end total_birds_in_tree_l260_260593


namespace part1_part2_l260_260788

-- Part 1: Proving range of \(a\) for function \(\(f(x)\)

theorem part1 (a : ℝ) : 
  (∀ x > 1, 2 * x^2 - 2 * a * x + 1 ≥ 0) ↔ (a ∈ Iic (3 / 2)) :=
sorry

-- Part 2: Finding max and min values for A=3/2
theorem part2 (f : ℝ → ℝ) :
  (∀ x ∈ Icc 0 2, f x = (2 / 3) * x^3 - (3 / 2) * x^2 + x + 1) → 
  (∀ x, x = 0 ∨ x = 2 → f(x) = 1 ∨ f(x) = 7 / 3) := 
sorry

end part1_part2_l260_260788


namespace spheres_in_cone_l260_260533
noncomputable def possible_values_of_n 
  (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a > 0) (h3 : b >0) : set ℕ :=
{n | n = 7 ∨ n = 8 ∨ n = 9}

theorem spheres_in_cone 
  (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a > 0) (h3 : b > 0)
  (n : ℕ) (n_additional_spheres : 
    ∀ i, 0 ≤ i < n → 
    (∃ r : ℝ, 
      let sin_psi := fun b r => (b - r) / (b + r)
      in ∃ ψ : ℝ, sin(ψ) = sin_psi b r ∧ (sin_ψ b r > 0) ∧ 
          let sin_phi := (a - b) / (a + b)
          ∃ φ R, 
            sin(φ) = sin_phi ∧ 
            R = (b + r) * 
                (sin(φ + ψ)) ∧
            (R > 0) ∧ 
            let k := sqrt(r) * sqrt(a) * sqrt(b) * sqrt(a) in 
              (r > 0) ∧ 
              (φ > 0) ∧ 
              (ψ > 0) ∧ 
              ((r / R) = sin(Real.pi / n)) ∧ 
              (k = (a * b)))
    ) : n ∈ possible_values_of_n a b h1 h2 h3 := 
sorry

end spheres_in_cone_l260_260533


namespace round_358_to_nearest_hundred_million_l260_260230

def interpret_rounding (x : ℝ) : ℝ :=
  if x % 100000000 < 50000000 then
    x - x % 100000000
  else
    x - x % 100000000 + 100000000

theorem round_358_to_nearest_hundred_million : interpret_rounding (35.8 * 10^8) = 3.6 * 10^9 :=
by
  sorry

end round_358_to_nearest_hundred_million_l260_260230


namespace relationship_between_D_and_A_l260_260070

variable {A B C D : Prop}

theorem relationship_between_D_and_A
  (h1 : A → B)
  (h2 : B → C)
  (h3 : D ↔ C) :
  (A → D) ∧ ¬(D → A) :=
by
sorry

end relationship_between_D_and_A_l260_260070


namespace f_has_extrema_iff_l260_260091

variable {a : ℝ}

def f (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

theorem f_has_extrema_iff :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv f x1 = 0 ∧ deriv f x2 = 0) ↔ a ∈ (Iio (-3) ∪ Ioi 6) :=
by
  -- sorry is used to skip the proof
  sorry

end f_has_extrema_iff_l260_260091


namespace cos_third_quadrant_l260_260856

theorem cos_third_quadrant (B : ℝ) (h1 : angle_in_third_quadrant B) (h2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

/-- You would need to define what it means for an angle to lie in the third quadrant. -/
def angle_in_third_quadrant (B : ℝ) : Prop := π < B ∧ B < 3 * π / 2

end cos_third_quadrant_l260_260856


namespace pizza_area_increase_l260_260589

noncomputable def area (r : ℝ) := Real.pi * r^2 

theorem pizza_area_increase (r1 r2 : ℝ) (h1 : r1 = 8) (h2 : r2 = 10) :
  (area r2 - area r1) / area r1 * 100 = 56.25 := 
by
  -- Uses preconditions
  have A1 := calc area r1 = Real.pi * 8^2 : by rw [h1, area]
  have A2 := calc area r2 = Real.pi * 10^2 : by rw [h2, area]
  sorry

end pizza_area_increase_l260_260589


namespace complex_modulus_l260_260440

theorem complex_modulus:
  (∃ (z : ℂ), (z - 1) * (2 + complex.i) = 5 * complex.i) →
  ∀ (z : ℂ), (z - 1) * (2 + complex.i) = 5 * complex.i → |conj z + complex.i| = real.sqrt 5 :=
by
  intro h,
  sorry

end complex_modulus_l260_260440


namespace determine_digit_z_l260_260394

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l260_260394


namespace length_QR_l260_260074

theorem length_QR (QP QR : ℝ) (h1 : cos Q = 3 / 5) (h2 : QP = 15) : QR = 25 :=
by
  sorry

end length_QR_l260_260074


namespace least_positive_three_digit_multiple_of_8_l260_260297

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l260_260297


namespace honor_students_count_l260_260995

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260995


namespace angle_A_is_pi_div_3_area_of_triangle_l260_260538

variables {A B C a b c : ℝ}

-- Problem 1: Prove angle A is π/3 given the specified trigonometric condition.
theorem angle_A_is_pi_div_3 
  (h1 : b * (Real.sin B - Real.sin C) + (c - a) * (Real.sin A + Real.sin C) = 0) :
  A = π / 3 :=
  sorry

-- Problem 2: Prove the area of triangle ABC is (3 + sqrt 3) / 4 given a = sqrt 3 and sin C = (1 + sqrt 3) / 2 * sin B.
theorem area_of_triangle 
  (ha : a = Real.sqrt 3) 
  (hsin : Real.sin C = (1 + Real.sqrt 3) / 2 * Real.sin B) :
  let s := 1/2 * a * b * Real.sin C in
  s = (3 + Real.sqrt 3) / 4 :=
  sorry

end angle_A_is_pi_div_3_area_of_triangle_l260_260538


namespace find_abc_l260_260905

noncomputable theory
open Polynomial

def t_k (k : ℕ) : ℕ := 
  if k = 0 then 3
  else if k = 1 then 6
  else if k = 2 then 14
  else 0  -- Placeholder, recurrence relation will be defined next

def recurrence_relation (a b c : ℕ) : Prop :=
  ∀ k ≥ 2, t_k (k + 1) = a * t_k k + b * t_k (k - 1) + c * t_k (k - 2)

theorem find_abc :
  ∃ a b c : ℕ, recurrence_relation a b c ∧ (a + b + c = 5) :=
begin
  -- Existence proof for a, b, and c that satisfy the recurrence relation and sum to 5
  sorry
end

end find_abc_l260_260905


namespace probability_one_in_solution_set_l260_260344

theorem probability_one_in_solution_set (a : ℝ) (h : a ∈ set.Icc (-2 : ℝ) 4) :
  let prob := ∫ (x : ℝ) in set.Icc (-2 : ℝ) 4, indicator_fn (λ a, a > -1 ∧ a < 2) a in
  (prob / (4 - (-2))) = 1 / 2 :=
by sorry

end probability_one_in_solution_set_l260_260344


namespace question1_question2_l260_260462

-- Conditions translated to Lean
axiom cond1 : ∀ x y : ℝ, f (x + y) - f y = (x + 2 * y - 2) * x
axiom cond2 : f 1 = 0

-- Definition of the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1)^2

-- Question 1: Proof that given the conditions, f(x) = (x - 1)^2
theorem question1 : ∀ x : ℝ, f x = (x - 1)^2 :=
by sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := (f x - 2 * x) / x

-- Question 2: Proof of the range of k
theorem question2 : ∀ k : ℝ, (∀ x : ℝ, x ∈ Icc (-2) 2 → g (2^x) - k * 2^x ≤ 0) → k ≥ 1 :=
by sorry

end question1_question2_l260_260462


namespace isosceles_triangle_perimeter_l260_260800

theorem isosceles_triangle_perimeter
  (x y : ℝ)
  (h : |x - 3| + (y - 1)^2 = 0)
  (isosceles_triangle : ∃ a b c, (a = x ∧ b = x ∧ c = y) ∨ (a = x ∧ b = y ∧ c = y) ∨ (a = y ∧ b = y ∧ c = x)):
  ∃ perimeter : ℝ, perimeter = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l260_260800


namespace count_expressible_integers_l260_260126

def g (x : ℝ) : ℤ := 
  Int.floor (3 * x) + Int.floor (6 * x) + Int.floor (9 * x) + Int.floor (12 * x) + Int.floor (15 * x)

theorem count_expressible_integers : 
  { n : ℤ | n > 0 ∧ n <= 1200 ∧ (∃ x : ℝ, g x = n) }.card = 780 := 
by
  sorry

end count_expressible_integers_l260_260126


namespace altitudes_circum_inradius_bound_l260_260798

namespace Triangle

variables (A B C : Type) [IsTriangle A B C] 
          (ha hb hc R r : ℝ) 
          [IsAcuteTriangle A B C] 
          [HasAltitude A B C ha hb hc] 
          [HasCircumradius A B C R] 
          [HasInradius A B C r]

theorem altitudes_circum_inradius_bound :
  min ha (min hb hc) ≤ R + r ∧ R + r ≤ max ha (max hb hc) :=
sorry

end Triangle

end altitudes_circum_inradius_bound_l260_260798


namespace suresh_wifes_speed_l260_260575

-- Define conditions
def circumference_of_track : ℝ := 0.726 -- track circumference in kilometers
def suresh_speed : ℝ := 4.5 -- Suresh's speed in km/hr
def meeting_time_in_hours : ℝ := 0.088 -- time till they meet in hours

-- Define the question and expected answer
theorem suresh_wifes_speed : ∃ (V : ℝ), V = 3.75 :=
  by
    -- Let Distance_covered_by_both = circumference_of_track
    let Distance_covered_by_suresh : ℝ := suresh_speed * meeting_time_in_hours
    let Distance_covered_by_suresh_wife : ℝ := circumference_of_track - Distance_covered_by_suresh
    let suresh_wifes_speed : ℝ := Distance_covered_by_suresh_wife / meeting_time_in_hours
    -- Expected answer
    existsi suresh_wifes_speed
    sorry

end suresh_wifes_speed_l260_260575


namespace hyperbola_distance_to_directrix_l260_260456

def hyperbola_has_distance_to_directrix_one (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  let c := 4 in
  let asymptote_ratio := b / a = Real.sqrt 3 in
  let focal_to_center_relation := c * c = a * a + b * b in
  let directrix_distance := (a * a) / c = 1 in
  asymptote_ratio ∧ focal_to_center_relation ∧ directrix_distance

theorem hyperbola_distance_to_directrix :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ hyperbola_has_distance_to_directrix_one a b :=
by
  use 2, 2 * Real.sqrt 3
  sorry

end hyperbola_distance_to_directrix_l260_260456


namespace line_intersects_ellipse_l260_260263

theorem line_intersects_ellipse (k : ℝ) : 
  ∃ x y : ℝ, y = k * x + 1 - k ∧ (x = 1 ∧ y = 1) ∧ (x^2 / 9 + y^2 / 4 < 1) :=
by
  -- Definitions based on conditions
  let x := 1
  let y := 1
  have line_eq : y = k * x + 1 - k := by sorry
  have point_inside_ellipse : x^2 / 9 + y^2 / 4 < 1 := by sorry
  
  -- Required proof based on correct answer
  use [x, y]
  exact ⟨line_eq, ⟨rfl, rfl⟩, point_inside_ellipse⟩

end line_intersects_ellipse_l260_260263


namespace relationship_abc_l260_260068

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (a b c : ℝ)

-- f(x) is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- f(x) is increasing on ℝ
axiom increasing_f : ∀ x y : ℝ, x < y → f x < f y

-- g(x) = x * f(x)
def g_def (x : ℝ) : ℝ := x * f x

-- Define a = g(-√5)
def a_def : ℝ := g (-sqrt 5)

-- Define b = g(2 ^ 0.8)
def b_def : ℝ := g (2 ^ 0.8)

-- Define c = g(3)
def c_def : ℝ := g 3

-- Prove the required relationship between a, b, and c
theorem relationship_abc :
  a = g_def (-sqrt 5) ∧
  b = g_def (2 ^ 0.8) ∧
  c = g_def 3 ∧
  (∀ x : ℝ, g x = g_def x) →
  (b < a ∧ a < c) :=
sorry

end relationship_abc_l260_260068


namespace frog_cannot_reach_1_0_l260_260703

theorem frog_cannot_reach_1_0 :
  ¬ ∃ (n : ℕ) (f : ℕ → ℤ × ℤ), f 0 = (0, 0) ∧ f n = (1, 0) ∧
    ∀ k < n, let ⟨x, y⟩ := f k, ⟨x', y'⟩ := f (k + 1) in (x' - x)^2 + (y' - y)^2 = 49 :=
sorry

end frog_cannot_reach_1_0_l260_260703


namespace cistern_leak_time_l260_260675

-- Definitions from conditions
def fill_rate_normal := 1 / 8 -- cistern per hour
def fill_rate_leak := 1 / 10 -- cistern per hour

-- We are to prove the rate at which the cistern empties and the time to empty when full.
theorem cistern_leak_time :
  let leak_rate := fill_rate_normal - fill_rate_leak in
  let time_to_empty := 1 / leak_rate in
  time_to_empty = 40 := by
  sorry

end cistern_leak_time_l260_260675


namespace cos_diff_simplify_l260_260233

theorem cos_diff_simplify (x : ℝ) (y : ℝ) (h1 : x = Real.cos (Real.pi / 10)) (h2 : y = Real.cos (3 * Real.pi / 10)) : 
  x - y = 4 * x * (1 - x^2) := 
sorry

end cos_diff_simplify_l260_260233


namespace determine_values_x_l260_260980

-- Statement only, no proof required.
-- Given conditions translated to Lean
noncomputable def right_triangle_sides (x : ℝ) : Prop :=
  (0 < x ∧ x < 90) ∧ (sin x + sin (3 * x) > sin (5 * x)) ∧
  (sin x + sin (5 * x) > sin (3 * x)) ∧ (sin (3 * x) + sin (5 * x) > sin x) ∧
  (one_of_non_right_angles_eq := 3 * x) ∧
  (x * π / 180) ∈ {10, 30, 50} 

-- Main theorem: Determine all possible values of x
theorem determine_values_x (x : ℝ) : 
  right_triangle_sides x → 
  x ∈ {10, 30, 50} :=
by
 sorry

end determine_values_x_l260_260980


namespace range_a_plus_2b_l260_260461

-- Define the function f(x) = |log x|
def f (x : ℝ) : ℝ := abs (Real.log x)
  
-- Main theorem statement
theorem range_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 3 < a + 2 * b :=
by
  -- Proof needs to be provided
  sorry

end range_a_plus_2b_l260_260461


namespace find_B_and_ac_val_l260_260510

variables {A B C a b c : ℝ}
variables (m n : Vector ℝ)
variables [AcuteTriangle A B C]
variables (h1 : m = (2 * sin (A + C), - sqrt 3))
variables (h2 : n = (1 - 2 * cos (B / 2) ^ 2, cos (2 * B)))
variables (h3 : dotProduct m n = 0)

theorem find_B_and_ac_val 
  (h4 : sin A * sin C = sin B ^ 2)
  (ha : sideA A = a)
  (hb : sideB B = b)
  (hc : sideC C = c) : 
  B = (π / 3) ∧ (a - c) = 0 := by
sory

end find_B_and_ac_val_l260_260510


namespace proof_problem_l260_260100

noncomputable def p (a : ℝ) : Prop :=
∀ x : ℝ, x^2 + a * x + a^2 ≥ 0

noncomputable def q : Prop :=
∃ x₀ : ℕ, 0 < x₀ ∧ 2 * x₀^2 - 1 ≤ 0

theorem proof_problem (a : ℝ) (hp : p a) (hq : q) : p a ∨ q :=
by
  sorry

end proof_problem_l260_260100


namespace candy_distribution_l260_260758

theorem candy_distribution (candy : ℕ) (people : ℕ) (hcandy : candy = 30) (hpeople : people = 5) :
  ∃ k : ℕ, candy - k = people * (candy / people) ∧ k = 0 := 
by
  sorry

end candy_distribution_l260_260758


namespace population_definition_l260_260282

variable (students : Type) (weights : students → ℝ) (sample : Fin 50 → students)
variable (total_students : Fin 300 → students)
variable (is_selected : students → Prop)

theorem population_definition :
    (∀ s, is_selected s ↔ ∃ i, sample i = s) →
    (population = {w : ℝ | ∃ s, w = weights s}) ↔
    (population = {w : ℝ | ∃ s, w = weights s ∧ ∃ i, total_students i = s}) := by
  sorry

end population_definition_l260_260282


namespace find_side_length_l260_260694

theorem find_side_length (s : ℝ) :
  (let original_area := s * s in
   let removed_area := 4 * (0.09 * original_area) in
   let remaining_area := original_area - removed_area in
   remaining_area = 256) →
  s = 20 :=
by
  intro h
  have h1 : 0.64 * (s * s) = 256 := by
    rw [←h]
    simp [original_area, removed_area, remaining_area]
  sorry

end find_side_length_l260_260694


namespace probability_hitting_target_is_0_point_7_l260_260454

noncomputable def probability_hitting_target : ℝ :=
  let P_A := 3 / 5
  let P_notA := 2 / 5
  let P_B_given_A := 0.9
  let P_B_given_notA := 0.4
  P_A * P_B_given_A + P_notA * P_B_given_notA

theorem probability_hitting_target_is_0_point_7 :
  probability_hitting_target = 0.7 := by
  sorry

end probability_hitting_target_is_0_point_7_l260_260454


namespace events_not_mutually_exclusive_l260_260696
noncomputable def Students := ℕ

def male_students : Students := 1
def female_students : Students := 2

def select_two_students (total : Students) : List (List ℕ) := 
  List.combinations (List.range (total + 1)) 2

def event_A (selection : List ℕ) : Prop := 0 ∈ selection
def event_B (selection : List ℕ) : Prop := 1 ∈ selection ∨ 2 ∈ selection

theorem events_not_mutually_exclusive : 
  ∃ (selection : List ℕ), event_A selection ∧ event_B selection :=
by
  let total_students := male_students + female_students
  let selections := select_two_students total_students
  have h : ∃ (s : List ℕ), event_A s ∧ event_B s := sorry
  exact h

end events_not_mutually_exclusive_l260_260696


namespace construct_triangle_max_area_l260_260818

noncomputable theory

variables (A0 B0 C0 A' B' C' : Type*)

def acute_triangle (T : Type*) := sorry -- define some property to represent acute-angled triangles

def internal_point (P Q R : Type*) (pt : Type*) :=
  sorry -- define some property to represent pt being an internal point of segment PQ

def similar (T1 T2 : Type*) :=
  sorry -- define similarity between two triangles

def largest_area (T : Type*) :=
  sorry -- define the property of having the largest area among such triangles

theorem construct_triangle_max_area
  (A0 B0 C0 A' B' C' T : Type*)
  (hA0 : acute_triangle A0)
  (hB0 : acute_triangle B0)
  (hC0 : acute_triangle C0)
  (hC0_in_AB : internal_point A B C0)
  (hA0_in_BC : internal_point B C A0)
  (hB0_in_CA : internal_point C A B0)
  (h_similar : similar T A') :
  ∃ (A B C : Type*), similar (triangle A B C) (triangle A' B' C') ∧ largest_area (triangle A B C) :=
begin
  sorry
end

end construct_triangle_max_area_l260_260818


namespace clique_cluque_claque_inequality_l260_260149

-- Define the necessary conditions
def city := Type

structure CityGraph (city: Type) :=
(bus_flight: city × city → Prop)
(plane_flight: city × city → Prop)
(bus_plane_relation: ∀ c1 c2 : city, bus_flight (c1, c2) ∨ plane_flight (c1, c2))

def is_clique {city : Type} (G : CityGraph city) (S : set city) : Prop :=
∀ (c1 c2 : city), c1 ≠ c2 → c1 ∈ S → c2 ∈ S → G.plane_flight (c1, c2)

def is_cluque {city : Type} (G : CityGraph city) (S : set city) : Prop :=
is_clique G S ∧ ∃ n, ∀ c1 c2 : city, c1 ≠ c2 → c1 ∈ S → c2 ∈ S → (number_of_bus_routes G c1 = n ∧ number_of_bus_routes G c2 = n)

def number_of_bus_routes {city : Type} (G : CityGraph city) (c : city) : ℕ :=
{c' : city | G.bus_flight (c, c')}.card

def is_claque {city : Type} (G : CityGraph city) (S : set city) : Prop :=
is_clique G S ∧ ∀ (c1 c2 : city), c1 ≠ c2 → c1 ∈ S → c2 ∈ S → number_of_bus_routes G c1 ≠ number_of_bus_routes G c2

-- Define the theorem
theorem clique_cluque_claque_inequality {city : Type} (G : CityGraph city) (C : set city) :
  is_clique G C →
  ∃ cluque claque : set city, is_cluque G cluque ∧ is_claque G claque ∧ C.card ≤ cluque.card * claque.card :=
sorry

end clique_cluque_claque_inequality_l260_260149


namespace part_I_part_II_l260_260090

noncomputable section

def f (x a : ℝ) : ℝ := |x + a| + |x - (1 / a)|

theorem part_I (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -5/2 ∨ x ≥ 5/2 := by
  sorry

theorem part_II (a m : ℝ) (h : ∀ x : ℝ, f x a ≥ |m - 1|) : -1 ≤ m ∧ m ≤ 3 := by
  sorry

end part_I_part_II_l260_260090


namespace count_definitive_quadratic_eqs_l260_260820

-- Define the conditions, given each equation explicitly in Lean format
def eq1 (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq2 (x : ℝ) : Prop := x^2 + 1/x - 1 = 0
def eq3 (t : ℝ) : Prop := 0.01 * t^2 = 1
def eq4 (x : ℝ) : Prop := x * (x^2 + x - 1) = x^3
def eq5 (x : ℝ) : Prop := x^2 + 2 * x = x^2 - 1

-- The theorem statement to prove the number of definitively quadratic equations
theorem count_definitive_quadratic_eqs :
  (∃ (a b c : ℝ), eq1 a b c 1 ∧ a ≠ 0) +
  (eq2 1) +
  (eq3 1) +
  (eq4 1) +
  (eq5 1) = 2 := 
sorry

end count_definitive_quadratic_eqs_l260_260820


namespace impossible_to_divide_1980_numbers_l260_260171

theorem impossible_to_divide_1980_numbers :
  let sum_1_to_1980 := (1980 * 1981) / 2 in
  let group_sum := 4 * s + 60 in
  ∃ s, group_sum = sum_1_to_1980 → False := 
by 
  sorry

end impossible_to_divide_1980_numbers_l260_260171


namespace stock_percentage_correct_l260_260770

noncomputable def percentage_of_stock (annual_income investment_amount stock_price : ℝ) : ℝ :=
  (annual_income * stock_price / investment_amount) / stock_price * 100

theorem stock_percentage_correct :
  percentage_of_stock 2500 6800 136 ≈ 36.76 :=
by
  sorry

end stock_percentage_correct_l260_260770


namespace vasya_has_greater_area_l260_260743

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l260_260743


namespace complement_A_in_U_l260_260102

variable (U A : Set ℤ) -- Define the sets U and A

-- Set the universal set U and the set A as given in the problem
def U := {-1, 0, 1}
def A := {0, 1}

-- The theorem to prove the complement of A in U is {-1}
theorem complement_A_in_U : U \ A = {-1} := by
    sorry

end complement_A_in_U_l260_260102


namespace equal_distances_l260_260505

open EuclideanGeometry

theorem equal_distances 
  (A B C D : Point)
  (convex : ConvexQuadrilateral A B C D)
  (angle_ADB_is_double_angle_ACB : ∠ A D B = 2 * ∠ A C B)
  (angle_BDC_is_double_angle_BAC : ∠ B D C = 2 * ∠ B A C) :
  Distance A D = Distance C D :=
sorry

end equal_distances_l260_260505


namespace mother_younger_than_father_l260_260147

variable (total_age : ℕ) (father_age : ℕ) (brother_age : ℕ) (sister_age : ℕ) (kaydence_age : ℕ) (mother_age : ℕ)

noncomputable def family_data : Prop :=
  total_age = 200 ∧
  father_age = 60 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  kaydence_age = 12 ∧
  mother_age = total_age - (father_age + brother_age + sister_age + kaydence_age)

theorem mother_younger_than_father :
  family_data total_age father_age brother_age sister_age kaydence_age mother_age →
  father_age - mother_age = 2 :=
sorry

end mother_younger_than_father_l260_260147


namespace honor_students_count_l260_260983

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l260_260983


namespace find_pencils_per_package_l260_260923

variables (pensPerPackage pencilsPerPackage totalPens totalPencils : ℕ)

/- Conditions -/
def package_of_pens := pensPerPackage = 12
def same_number_of_pencils_as_pens := totalPens = totalPencils
def philip_can_buy_60_pens := totalPens = 60

/- Statement -/
theorem find_pencils_per_package
  (h1: package_of_pens)
  (h2: same_number_of_pencils_as_pens)
  (h3: philip_can_buy_60_pens) :
  pencilsPerPackage = 12 :=
sorry

end find_pencils_per_package_l260_260923


namespace range_of_a_l260_260076

def f(x : ℝ) (a : ℝ) := x^3 + a*x^2 + (a+6)*x + 1

theorem range_of_a (a : ℝ) :
  (∃ x_max x_min, (∃ x, f a x = x_max) ∧ (∃ x, f a x = x_min)) 
  → a ∈ set.Ioo (-∞ : ℝ) (-3) ∪ set.Ioo (6 : ℝ) (∞ : ℝ) :=
by
  sorry

end range_of_a_l260_260076


namespace part1_part2_l260_260207

-- Condition 1
def f (theta : ℝ) : ℝ := (real.sqrt 3) * real.sin theta + real.cos theta

-- Condition 2, specifying point coordinates
def P : ℝ × ℝ := (real.sqrt 3 / 2, 1 / 2)

-- 1. Prove f(θ) = sqrt(3) if θ satisfies sin(θ) = 1/2 and cos(θ) = sqrt(3)/2
theorem part1 (theta : ℝ) (h1 : real.sin theta = 1 / 2) (h2 : real.cos theta = real.sqrt 3 / 2) : f(theta) = real.sqrt 3 := by
  sorry

-- 2. Prove θ = π/3 if f(θ) = 2
theorem part2 (theta : ℝ) (h : f(theta) = 2) (h_interval : 0 ≤ theta ∧ theta ≤ real.pi) : theta = real.pi / 3 := by
  sorry

end part1_part2_l260_260207


namespace range_of_a_l260_260469

open Real

def f (x : ℝ) : ℝ := exp x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g a x₁ ∧ f x₂ = g a x₂ ∧ x₁ * x₂ = 1) ↔
    (0 < a ∧ a < 1) ∨ (a > 1) :=
by
  sorry

end range_of_a_l260_260469


namespace alice_total_distance_correct_l260_260010

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end alice_total_distance_correct_l260_260010


namespace min_dist_l260_260159

noncomputable def polar_equations_c1 (theta: ℝ) : ℝ := 2 * Real.cos theta
noncomputable def polar_equations_c2 (alpha: ℝ) : ℝ^2 := 8 / (1 + Real.sin² alpha)
noncomputable def distance_OA (alpha: ℝ) : ℝ^2 := 4 * (Real.cos² alpha)
noncomputable def distance_OB (alpha: ℝ) : ℝ^2 := 8 / (1 + Real.sin² alpha)

theorem min_dist (alpha : ℝ) (h1 : 0 < alpha ∧ alpha < Real.pi / 2) :
  ∃ (answer : ℝ), answer = 8 * Real.sqrt 2 - 8 ∧
  ∀ a b, (|distance_OB(alpha)|^2 - |distance_OA(alpha)|^2) = answer :=
sorry

end min_dist_l260_260159


namespace problem1_ap_problem2_op_l260_260803

-- Definitions of points, vectors, and the unit circle condition
variables {V : Type} [inner_product_space ℝ V]
variables (O A B P : V)

-- Non-collinear condition
def non_collinear (V : Type) [inner_product_space ℝ V] {A B : V} : Prop :=
¬ collinear ℝ {A, B}

-- The proof problem conditions
variables (h_non_collinear : non_collinear V A B)
variables (h_ap_eq_2pb : ∃ (r s : ℝ), (P = r • B + s • A ∧ 2 • (1 - r) = r))
variables (h_op_eq_maob : ∃ (m : ℝ), (P = m • A + B))
variables (h_parallelogram : ∃ A B O P : V, O + A = P + B ∧ O + B = A + P)

-- Statement 1: Prove r + s = 0 given the expression for AP
theorem problem1_ap (r s : ℝ) : ∀ (A B P : V),
  (P = r • B + s • A ∧ 2 • (1 - r) = r) → r + s = 0  :=
by sorry

-- Statement 2: Prove m = -1 given the expression for OP and the parallelogram condition
theorem problem2_op (m : ℝ) : ∀ (A B P : V),
  (P = m • A + B) ∧ (O + A = P + B ∧ O + B = A + P) → m = -1 :=
by sorry

end problem1_ap_problem2_op_l260_260803


namespace simplify_cos18_minus_cos54_l260_260234

noncomputable def cos_54 : ℝ := 2 * (cos 27)^2 - 1
noncomputable def cos_27 : ℝ := sqrt ((1 + cos_54) / 2)
noncomputable def cos_18 : ℝ := 1 - 2 * (sin 9)^2
noncomputable def sin_9 : ℝ := sqrt ((1 - cos_18) / 2)

theorem simplify_cos18_minus_cos54 : (cos 18 - cos 54) = 0 :=
by
  have h_cos_54 : cos 54 = cos_54 := by sorry
  have h_cos_27 : cos 27 = cos_27 := by sorry
  have h_cos_18 : cos 18 = cos_18 := by sorry
  have h_sin_9 : sin 9 = sin_9 := by sorry
  sorry

end simplify_cos18_minus_cos54_l260_260234


namespace value_of_expression_l260_260132

theorem value_of_expression (x : ℕ) (h : x = 3) : 2 * x + 3 = 9 :=
by 
  sorry

end value_of_expression_l260_260132


namespace smallest_integer_n_satisfying_inequality_l260_260306

theorem smallest_integer_n_satisfying_inequality :
  ∃ n : ℤ, n^2 - 13 * n + 36 ≤ 0 ∧ (∀ m : ℤ, m^2 - 13 * m + 36 ≤ 0 → m ≥ n) ∧ n = 4 := 
by
  sorry

end smallest_integer_n_satisfying_inequality_l260_260306


namespace Lance_must_read_today_l260_260178

def total_pages : ℕ := 100
def pages_read_yesterday : ℕ := 35
def pages_read_tomorrow : ℕ := 27

noncomputable def pages_read_today : ℕ :=
  pages_read_yesterday - 5

noncomputable def pages_left_today : ℕ :=
  total_pages - (pages_read_yesterday + pages_read_today + pages_read_tomorrow)

theorem Lance_must_read_today :
  pages_read_today + pages_left_today = 38 :=
by 
  rw [pages_read_today, pages_left_today, pages_read_yesterday, pages_read_tomorrow, total_pages]
  simp
  sorry

end Lance_must_read_today_l260_260178


namespace main_theorem_l260_260686

noncomputable def exists_coprime_integers (a b p : ℤ) : Prop :=
  ∃ (m n : ℤ), Int.gcd m n = 1 ∧ p ∣ (a * m + b * n)

theorem main_theorem (a b p : ℤ) : exists_coprime_integers a b p := 
  sorry

end main_theorem_l260_260686


namespace sum_nearest_integer_l260_260767

open BigOperators

theorem sum_nearest_integer :
  (500 * ∑ n in Finset.range 19997 \ Finset.range 3, 1 / ((n + 1 - 3) * (n + 1 + 3)) : ℚ).round = 153 :=
by
  sorry

end sum_nearest_integer_l260_260767


namespace sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l260_260449

theorem sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_β : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π / 4 := sorry

end sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l260_260449


namespace num_integers_satisfying_inequality_l260_260109

theorem num_integers_satisfying_inequality : 
  ∃ (xs : Finset ℤ), (∀ x ∈ xs, -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9) ∧ xs.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l260_260109


namespace if_a_gt_abs_b_then_a2_gt_b2_l260_260904

theorem if_a_gt_abs_b_then_a2_gt_b2 (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by sorry

end if_a_gt_abs_b_then_a2_gt_b2_l260_260904


namespace sum_first_11_terms_is_22_l260_260444

-- Define the arithmetic sequence and its related properties
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 0 + n * d

-- Define the roots' condition
def roots_condition (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∃ b c : ℝ, b = a 5 ∧ c = a 7 ∧ b * b - 4 * b + k = 0 ∧ c * c - 4 * c + k = 0

-- Define the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

-- Define the goal statement
theorem sum_first_11_terms_is_22 (a : ℕ → ℝ) (k : ℝ) (h_seq : arithmetic_sequence a) (h_root_cond : roots_condition a k) : 
  sum_first_n_terms a 11 = 22 :=
sorry

end sum_first_11_terms_is_22_l260_260444


namespace solve_inequality_l260_260937

theorem solve_inequality (x : ℝ) (h : x > -2) : 
  (log 3 ((x + 2) * (x + 4)) + log (1 / 3) (x + 2) < (1 / 2) * log 3 7) ↔ (-2 < x ∧ x < 3) := 
sorry

end solve_inequality_l260_260937


namespace min_value_l260_260434

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
(hgeom : 3^x * 3^(3*y) = 3) : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 3^x * 3^(3 * y) = 3 ∧  1 / x + 1 / (3 * y) = 4) :=
by
  sorry

end min_value_l260_260434


namespace grocer_decaf_percentage_l260_260337

theorem grocer_decaf_percentage :
  let total_weight_700 := 700
  let type_A_weight := 0.4 * total_weight_700
  let type_A_decaf := 0.3 * type_A_weight
  let type_B_weight := 0.35 * total_weight_700
  let type_B_decaf := 0.5 * type_B_weight
  let type_C_weight := 0.25 * total_weight_700
  let type_C_decaf := 0.6 * type_C_weight
  let additional_type_C := 150
  let new_total_weight := total_weight_700 + additional_type_C
  let new_type_C_weight := type_C_weight + additional_type_C
  let new_type_C_decaf := 0.6 * new_type_C_weight
  let total_decaf := type_A_decaf + type_B_decaf + new_type_C_decaf
  let decaf_percentage := (total_decaf / new_total_weight) * 100
  decaf_percentage ≈ 47.24 :=
by
  sorry

end grocer_decaf_percentage_l260_260337


namespace general_term_a_n_sum_T_n_l260_260184

section
  variable {n : ℕ}

  -- Given conditions
  def S_n (n : ℕ) : ℝ := (1 / 2) * 3 ^ n + (3 / 2)
  def a_n (n : ℕ) : ℝ :=
    if n = 1 then 3 else 3 ^ (n - 1)
  def b_n (n : ℕ) : ℝ :=
    if a_n n = 3 then (1 / 3) else (n - 1) / 3 ^ (n - 1)

  -- Main theorems to prove
  theorem general_term_a_n (n : ℕ) : a_n n = if n = 1 then 3 else 3 ^ (n - 1) :=
    sorry
    
  theorem sum_T_n (n : ℕ) : 
    let T_n := (finset.range n).sum (λ i, b_n (i + 1)) in
    T_n = (13 / 12) - (2 * n + 1) / (4 * 3 ^ (n - 1)) :=
    sorry
end

end general_term_a_n_sum_T_n_l260_260184


namespace solution_set_of_inequality_l260_260537

-- Definition of the greatest integer function
def greatest_integer (x : ℝ) : ℤ := Int.floor x

-- The mathematical statement to be proven
theorem solution_set_of_inequality (x : ℝ) : ((greatest_integer x)^2 - 5 * greatest_integer x + 6 ≤ 0) ↔ (2 ≤ x ∧ x < 4) :=
by
  sorry

end solution_set_of_inequality_l260_260537


namespace number_of_nickels_l260_260698

theorem number_of_nickels (n d : ℕ) (h1 : n + d = 70) (h2 : 0.05 * n + 0.10 * d = 5.55) : n = 29 :=
by
  sorry

end number_of_nickels_l260_260698


namespace main_diagonal_distinct_l260_260514

open Matrix

-- Define the problem in Lean 4
theorem main_diagonal_distinct
  (n : ℕ) 
  (a : Matrix (Fin (2 * n + 1)) (Fin (2 * n + 1)) ℕ)
  (sym : ∀ i j, a i j = a j i)
  (rows : ∀ i, ∃! perm : Fin (2 * n + 1) → Fin (2 * n + 1), ∀ j, a i j = perm j + 1 ∧ perm j + 1 ∈ Fin (2 * n + 1))
  (cols : ∀ j, ∃! perm : Fin (2 * n + 1) → Fin (2 * n + 1), ∀ i, a i j = perm i + 1 ∧ perm i + 1 ∈ Fin (2 * n + 1)):
  ∀ i j, i ≠ j → a i i ≠ a j j :=
by
  sorry

end main_diagonal_distinct_l260_260514


namespace closed_curve_area_l260_260949

noncomputable def enclosedArea (num_arcs : ℕ) (arc_length : ℝ) (side_length : ℝ) : ℝ :=
  let r := arc_length / Real.pi
  let sector_area := (arc_length / (2 * Real.pi)) * Real.pi * r^2
  let total_sector_area := num_arcs * sector_area
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  octagon_area + total_sector_area

theorem closed_curve_area :
  enclosedArea 12 π 3 = 54 + 54 * Real.sqrt 2 + 6 * π :=
by
  sorry

end closed_curve_area_l260_260949


namespace factorial_inequalities_1_factorial_inequalities_2_l260_260924

noncomputable def e := 2.71828 -- This is an approximation

theorem factorial_inequalities_1 (n : ℕ) (h : n > 6) : 
  (n / e) ^ n < n! ∧ n! < n * (n / e) ^ n := 
sorry

theorem factorial_inequalities_2 (a1 a2 : ℝ) (h1 : a1 < e) (h2 : e < a2) :
  ∃ N : ℕ, ∀ n : ℕ, (n > N) → (n / a1) ^ n > n! ∧ n! > (n / a2) ^ n := 
sorry

end factorial_inequalities_1_factorial_inequalities_2_l260_260924


namespace number_of_odd_digits_in_base4_of_345_l260_260418

-- Define the conditions and the proof problem
theorem number_of_odd_digits_in_base4_of_345 :
  let n := 345 in
  let base4_representation := 10521 in
  let odd_digits := [1, 5, 1] in
  odd_digits.length = 3 :=
by
  -- Definitions
  let n := 345
  let base4_representation := 10521
  let odd_digits := [1, 5, 1]
  -- Expected result
  have h : odd_digits.length = 3 := by rfl
  exact h

end number_of_odd_digits_in_base4_of_345_l260_260418


namespace honor_students_count_l260_260998

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260998


namespace gabby_fruit_total_l260_260045

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end gabby_fruit_total_l260_260045


namespace count_non_decreasing_maps_l260_260415

-- Define the set and the mapping condition
def is_non_decreasing_map (f : Fin 3 → Fin 5) : Prop :=
  ∀ (i j : Fin 3), i < j → f i ≤ f j

-- Define the problem as a theorem statement
theorem count_non_decreasing_maps :
  (Fin 3 → Fin 5) → Prop :=
  λ f, ∃! (f : Fin 3 → Fin 5), is_non_decreasing_map f :=
  35 :=
sorry

end count_non_decreasing_maps_l260_260415


namespace range_of_a_l260_260816

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end range_of_a_l260_260816


namespace maxwell_walking_speed_l260_260571

theorem maxwell_walking_speed :
  ∀ (distance_between_homes : ℕ)
    (brad_speed : ℕ)
    (middle_travel_maxwell : ℕ)
    (middle_distance : ℕ),
    distance_between_homes = 36 →
    brad_speed = 4 →
    middle_travel_maxwell = 12 →
    middle_distance = 18 →
    (middle_travel_maxwell : ℕ) / (8 : ℕ) = (middle_distance - middle_travel_maxwell) / brad_speed :=
  sorry

end maxwell_walking_speed_l260_260571


namespace inequality_problem_l260_260647

theorem inequality_problem (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by 
  sorry

end inequality_problem_l260_260647


namespace sum_of_squares_bound_l260_260553

variable {p : ℕ} {n M r : ℝ} {q : ℕ}
variable (a : Fin p → ℝ)

open Real

noncomputable def decreasing_seq (a : Fin p → ℝ) : Prop :=
  ∀ i j : Fin p, i ≤ j → a i ≥ a j

theorem sum_of_squares_bound (h_seq : decreasing_seq a) 
    (h_sum : (∑ i, a i) = n) (h_nonneg : ∀ i, 0 ≤ a i)
    (h_le_M : ∀ i, a i ≤ M) (h_split : n = ↑q * M + r)
    (h_q_pos : 0 < q) (h_r_bound : 0 ≤ r ∧ r < M) :
    (∑ i, a i ^ 2) ≤ q * M ^ 2 + r ^ 2 :=
by
  sorry

end sum_of_squares_bound_l260_260553


namespace vec_op_result_l260_260563

variables {θ : ℝ} (a b : ℝ ^ 3)

def unit_vector (v : ℝ ^ 3) : Prop := ∥v∥ = 1
def vec_mag_sqrt2 (v : ℝ ^ 3) : Prop := ∥v∥ = real.sqrt 2
def vec_diff_mag1 (u v : ℝ ^ 3) : Prop := ∥u - v∥ = 1
def vector_op (a b : ℝ ^ 3) (θ : ℝ) : ℝ := ∥a * real.sin θ + b * real.cos θ∥

theorem vec_op_result (a b : ℝ ^ 3) (θ : ℝ)
  (ha : unit_vector a)
  (hb : vec_mag_sqrt2 b)
  (habdiff : vec_diff_mag1 a b) :
  vector_op a b θ = real.sqrt 10 / 2 :=
sorry

end vec_op_result_l260_260563


namespace solve_system_l260_260600

theorem solve_system (x y z : ℝ) 
  (h1 : x^3 - y = 6)
  (h2 : y^3 - z = 6)
  (h3 : z^3 - x = 6) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_system_l260_260600


namespace circles_intersection_equality_l260_260644

noncomputable def circle (center : Point) (radius : ℝ) : set Point := sorry

variable 
  (X Y : set Point)
  (A B C D E O_Y : Point)
  (R_X : ℝ)
  (h_X : circle O_X R_X = X)
  (h_Y : circle O_Y R_Y = Y)
  (h_intersect : A ∈ X ∧ A ∈ Y ∧ B ∈ X ∧ B ∈ Y)
  (h_center : O_Y ∈ X)
  (h_C_on_X : C ∈ X)
  (h_D_on_Y : D ∈ Y)
  (h_collinear : line_through C B D)
  (h_E_on_Y : E ∈ Y)
  (h_parallel : parallel (line_through D E) (line_through A C))

theorem circles_intersection_equality 
  (h_AE_eq_AB : dist A E = dist A B) :
  True :=
sorry

end circles_intersection_equality_l260_260644


namespace cos_exp_rel_l260_260146

theorem cos_exp_rel (x : ℝ) : cos x = (exp (complex.I * x) + exp (-complex.I * x)) / 2 :=
by
  sorry

end cos_exp_rel_l260_260146


namespace product_of_odd_integers_l260_260654

theorem product_of_odd_integers :
  let odd_factorial_product := ∏ i in finset.filter (λ x : ℕ, x % 2 = 1) (finset.range 10000), i
  in odd_factorial_product = 10000.factorial / (2^5000 * 5000.factorial) :=
by
  sorry 

end product_of_odd_integers_l260_260654


namespace securities_investors_in_equities_l260_260403

-- Definitions
def total_investors := 100
def num_equities := 80
def num_equities_and_securities := 25

-- Proof Statement
theorem securities_investors_in_equities :
  ∀ (total_investors num_equities num_equities_and_securities : ℕ),
  num_equities = 80 →
  num_equities_and_securities = 25 →
  ∃ num_securities : ℕ, num_securities = num_equities_and_securities :=
by
  intros total_investors num_equities num_equities_and_securities H1 H2
  use num_equities_and_securities
  exact H2

end securities_investors_in_equities_l260_260403


namespace line_of_intersection_line_of_intersection_parallel_l260_260945

-- Define the structure for points and lines
structure Point := (x y z : ℝ)

-- Definitions of planes
def plane (A B C : Point) : Set Point := {
  P | ∃ λ μ : ℝ, P = ⟨A.x + λ * (B.x - A.x) + μ * (C.x - A.x), 
                      A.y + λ * (B.y - A.y) + μ * (C.y - A.y), 
                      A.z + λ * (B.z - A.z) + μ * (C.z - A.z)⟩
}

-- Defining pyramid and the intersection logic
noncomputable def pyramid_base (S A B C D : Point) : Set Point :=
  plane A B S ∩ plane C D S

-- Prove line of intersection based on conditions
theorem line_of_intersection (S A B C D M : Point) 
(h_intersect : ∃ M, M ∈ line A B ∧ M ∈ line C D)
: ∃ l: Set Point, l = line S M ∨ ∃ l: Set Point, (l = { P : Point | ∃ λ: ℝ, P = ⟨S.x + λ * (A.x - S.x), S.y + λ * (A.y - S.y), S.z + λ * (A.z - S.z) ⟩ }) :=
begin
  sorry
end

theorem line_of_intersection_parallel (S A B C D : Point) 
(h_parallel: parallel A B C D)
: ∃ l: Set Point, l = { P : Point | ∃ λ: ℝ, P = ⟨S.x + λ * (A.x - S.x), S.y + λ * (A.y - S.y), S.z + λ * (A.z - S.z) ⟩ } :=
begin
  sorry
end

end line_of_intersection_line_of_intersection_parallel_l260_260945


namespace BradSpeed_is_correct_l260_260211

noncomputable def BradSpeed (MaxwellSpeed BradStartDelay InitialDistance MeetupTime : ℝ) : ℝ :=
  let DistanceWalkedByMaxwell := MaxwellSpeed * MeetupTime
  let DistanceToBeCoveredByBrad := InitialDistance - DistanceWalkedByMaxwell
  -- Time taken by Brad
  let TimeTakenByBrad := MeetupTime - BradStartDelay
  -- Brad's running speed
  DistanceToBeCoveredByBrad / TimeTakenByBrad

theorem BradSpeed_is_correct :
  ∀ (MaxwellSpeed BradStartDelay InitialDistance MeetupTime : ℝ), 
    MaxwellSpeed = 4 → 
    BradStartDelay = 1 → 
    InitialDistance = 34 → 
    MeetupTime = 4 → 
      BradSpeed MaxwellSpeed BradStartDelay InitialDistance MeetupTime = 6 :=
by
  intros MaxwellSpeed BradStartDelay InitialDistance MeetupTime h1 h2 h3 h4
  rw [BradSpeed]
  rw [h1, h2, h3, h4]
  simp only [Real.smul_eq_mul, sub_self, add_zero]
  rw [show 4 * 4 = 16, from rfl]
  rw [show 34 - 16 = 18, from rfl]
  rw [show 4 - 1 = 3, from rfl]
  exact rfl

end BradSpeed_is_correct_l260_260211


namespace tan_alpha_plus_pi_over_4_l260_260786

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos (2 * α), Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (1, 2 * Real.sin α - 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi)
    (h3 : dot_product (vec_a α) (vec_b α) = 0) :
    Real.tan (α + Real.pi / 4) = -1 := sorry

end tan_alpha_plus_pi_over_4_l260_260786


namespace A_speed_ratio_B_speed_l260_260360

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end A_speed_ratio_B_speed_l260_260360


namespace inverse_of_k_l260_260546

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l260_260546


namespace work_completion_days_l260_260328

theorem work_completion_days (A B : ℕ) (hB : B = 12) (work_together_days : ℕ) (work_together : work_together_days = 3) (work_alone_days : ℕ) (work_alone : work_alone_days = 3) : 
  (1 / A + 1 / B) * 3 + (1 / B) * 3 = 1 → A = 6 := 
by 
  intro h
  sorry

end work_completion_days_l260_260328


namespace transformed_polynomial_roots_l260_260078

theorem transformed_polynomial_roots :
  (∀ a b c d : ℝ, 
    Polynomial.eval a (Polynomial.X^4 - 16 * Polynomial.X - 2) = 0
    ∧ Polynomial.eval b (Polynomial.X^4 - 16 * Polynomial.X - 2) = 0
    ∧ Polynomial.eval c (Polynomial.X^4 - 16 * Polynomial.X - 2) = 0
    ∧ Polynomial.eval d (Polynomial.X^4 - 16 * Polynomial.X - 2) = 0) →
    Polynomial.eval (a + b) (Polynomial.X^4 - 16 * Polynomial.X + 2) = 0
    ∧ Polynomial.eval (a + c) (Polynomial.X^4 - 16 * Polynomial.X + 2) = 0
    ∧ Polynomial.eval (b + c) (Polynomial.X^4 - 16 * Polynomial.X + 2) = 0
    ∧ Polynomial.eval (b + d) (Polynomial.X^4 - 16 * Polynomial.X + 2) = 0 :=
by
  sorry

end transformed_polynomial_roots_l260_260078


namespace min_value_of_f_l260_260022

noncomputable def f (x : ℝ) : ℝ :=
  x^2 / (x - 3)

theorem min_value_of_f : ∀ x > 3, f x ≥ 12 :=
by
  sorry

end min_value_of_f_l260_260022


namespace range_of_x_inequality_l260_260463

def f (x : ℝ) : ℝ := (1/2)^(abs x) - 1 / (1 + real.logb (1/2) (1 + abs x))

theorem range_of_x_inequality :
  { x : ℝ | f x > f (2 * x - 1) } = { x : ℝ | x < -1 } ∪ { x : ℝ | -1 < x ∧ x < 1/3 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end range_of_x_inequality_l260_260463


namespace miscellaneous_expenses_correct_l260_260362

-- Definitions of the given conditions
def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def saved : ℝ := 2350

-- Assertion to be proved
theorem miscellaneous_expenses_correct :
  let salary := saved / 0.10 in
  let total_spent := rent + milk + groceries + education + petrol in
  let misc_expenses := salary - (total_spent + saved) in
  misc_expenses = 5650 := by
{
  sorry
}

end miscellaneous_expenses_correct_l260_260362


namespace alcohol_reduction_l260_260483

theorem alcohol_reduction (V_initial : ℝ) (C_initial : ℝ) (V_final : ℝ) (C_final : ℝ) :
    V_initial = 10 → C_initial = 20 → V_final = 40 →
    C_final = (2 / V_final) * 100 →
    (C_initial - C_final) / C_initial * 100 = 75 :=
by
  intros hV_initial hC_initial hV_final hC_final
  rw [hV_initial, hC_initial, hV_final, hC_final]
  norm_num
  sorry

end alcohol_reduction_l260_260483


namespace candy_left_l260_260642

-- Definitions of the problem conditions
def num_houses : ℕ := 15
def candies_per_house : ℕ := 8
def num_people : ℕ := 3
def candies_per_person_eaten : ℕ := 6

-- The theorem statement
theorem candy_left (houses : ℕ) (candies_per_house : ℕ) (people : ℕ) (candies_eaten_per_person : ℕ) :
  houses * candies_per_house - people * candies_eaten_per_person = 102 :=
by 
  -- proof goes here
  sorry

-- Given values
#eval candy_left num_houses candies_per_house num_people candies_per_person_eaten

end candy_left_l260_260642


namespace range_of_m_l260_260428

theorem range_of_m (m : ℝ) (h : (m^2 + m) ^ (3 / 5) ≤ (3 - m) ^ (3 / 5)) : 
  -3 ≤ m ∧ m ≤ 1 :=
by { sorry }

end range_of_m_l260_260428


namespace solve_problem_l260_260888

noncomputable def problem_statement (a c B b : ℝ) : Prop :=
  (a * c = 8) ∧
  (a + c = 7) ∧
  (B = π / 3) →
  b^2 = a^2 + c^2 - 2 * a * c * real.cos B →
  b = 5

theorem solve_problem :
  ∃ (a c b : ℝ), problem_statement a c (π / 3) b :=
by
  use [3, 4, 5]
  sorry

end solve_problem_l260_260888


namespace sum_of_endpoints_l260_260170

noncomputable def triangle_side_length (PQ QR PR QS PS : ℝ) (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = (PS / QS)) : ℝ :=
  if 4 < PR ∧ PR < 18 then 4 + 18 else 0

theorem sum_of_endpoints {PQ PR QS PS : ℝ} (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = ( PS / QS)) :
  triangle_side_length PQ 0 PR QS PS h1 h2 h3 = 22 := by
  sorry

end sum_of_endpoints_l260_260170


namespace honor_students_count_l260_260984

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l260_260984


namespace part1_part2_part3_l260_260830

def f (x : ℝ) := 6 / x
def g (x : ℝ) := x^2 + 1

theorem part1 (x : ℝ) :
  f (g x) = 6 / (x^2 + 1) :=
by sorry

theorem part2 (k : ℝ) :
  (∀ x : ℝ, 6 / (x^2 + 1) ≥ k - 7 * x^2) ↔ k ≤ 6 :=
by sorry

theorem part3 (a : ℝ) :
  (∃! n1 n2 n3 : ℕ, n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧
  6 / ((n1 : ℝ)^2 + 1) > a / (n1 : ℝ) ∧
  6 / ((n2 : ℝ)^2 + 1) > a / (n2 : ℝ) ∧
  6 / ((n3 : ℝ)^2 + 1) > a / (n3 : ℝ) ∧
  n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3) ↔
  a ∈ set.Icc (24/17 : ℝ) (9/5 : ℝ) :=
by sorry

end part1_part2_part3_l260_260830


namespace max_area_rectangle_min_area_rectangle_l260_260435

theorem max_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k) 
  : (n - 1 + 2^(2*n)) * (4 * 2^(2*(n-1)) - 1/3) = 1/3 * (4^n - 1) * (4^n + n - 1) := sorry

theorem min_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k)
  : (2^n - 1)^2 = 4 * (2^n - 1)^2 := sorry

end max_area_rectangle_min_area_rectangle_l260_260435


namespace sum_of_arithmetic_sequence_l260_260485

variable {S : ℕ → ℕ}

def isArithmeticSum (S : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ n, S n = n * (2 * a + (n - 1) * d ) / 2

theorem sum_of_arithmetic_sequence :
  isArithmeticSum S →
  S 8 - S 4 = 12 →
  S 12 = 36 :=
by
  intros
  sorry

end sum_of_arithmetic_sequence_l260_260485


namespace determine_digit_z_l260_260396

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l260_260396


namespace sum_of_remaining_numbers_bounded_l260_260869

theorem sum_of_remaining_numbers_bounded (n : ℕ) 
    (a b : Fin n → ℝ) 
    (h_pos_a : ∀ i, a i > 0) 
    (h_pos_b : ∀ i, b i > 0) 
    (h_sum : ∀ i, a i + b i = 1) :
    ∃ (s : Fin n → Bool), 
        ((∑ i, if s i then a i else 0) ≤ (n + 1) / 4) ∧ 
        ((∑ i, if s i then 0 else b i) ≤ (n + 1) / 4) :=
sorry

end sum_of_remaining_numbers_bounded_l260_260869


namespace angle_A_45_degrees_l260_260504

variables {a b c : ℝ} {S : ℝ} {A : ℝ}

def area (a b c : ℝ) (S : ℝ) (A : ℝ) := S = 1/4 * (b^2 + c^2 - a^2)

theorem angle_A_45_degrees 
    (a b c : ℝ) 
    (S : ℝ)
    (h : area a b c S)
    (hA : S = 1/2 * b * c * real.sin A) :
    A = 45 :=
by
  -- the proof is omitted
  sorry

end angle_A_45_degrees_l260_260504


namespace coefficient_x3y2_is_3_l260_260411

-- Given expression
def expr := 4 * (x ^ 3 * y ^ 2 - 2 * x ^ 4 * y ^ 3) + 
            3 * (x ^ 2 * y - x ^ 3 * y ^ 2) - 
            (5 * x ^ 4 * y ^ 3 - 2 * x ^ 3 * y ^ 2)

-- The main theorem to prove
theorem coefficient_x3y2_is_3 : 
  (coef (expand expr) (x ^ 3 * y ^ 2) = 3) := 
by 
  sorry

end coefficient_x3y2_is_3_l260_260411


namespace ratio_of_areas_l260_260168

variable (A B C D E : Type) 
variable (AB CD : ℝ)
variable [hAB : AB = 9]
variable [hCD : CD = 16]
variable [hCondition : Trapezoid A B C D]

noncomputable def area_ratio (E A B C D : Type) (AB CD : ℝ) [hAB : AB = 9] [hCD : CD = 16] [hCondition : Trapezoid A B C D] : ℝ :=
  (area (triangle E A B)) / (area (trapezoid A B C D))

theorem ratio_of_areas :
  area_ratio E A B C D AB CD = 81 / 175 :=
by
  sorry

end ratio_of_areas_l260_260168


namespace max_f_on_sphere_l260_260021

noncomputable def f (x y z : ℝ) : ℝ := 3 * x + 5 * y - z

theorem max_f_on_sphere :
  ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 1 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → f x y z ≤ sqrt 35 :=
sorry

end max_f_on_sphere_l260_260021


namespace andrena_has_more_dolls_than_debelyn_l260_260382

-- Definitions based on the given conditions
def initial_dolls_debelyn := 20
def initial_gift_debelyn_to_andrena := 2

def initial_dolls_christel := 24
def gift_christel_to_andrena := 5
def gift_christel_to_belissa := 3

def initial_dolls_belissa := 15
def gift_belissa_to_andrena := 4

-- Final number of dolls after exchanges
def final_dolls_debelyn := initial_dolls_debelyn - initial_gift_debelyn_to_andrena
def final_dolls_christel := initial_dolls_christel - gift_christel_to_andrena - gift_christel_to_belissa
def final_dolls_belissa := initial_dolls_belissa - gift_belissa_to_andrena + gift_christel_to_belissa
def final_dolls_andrena := initial_gift_debelyn_to_andrena + gift_christel_to_andrena + gift_belissa_to_andrena

-- Additional conditions
def andrena_more_than_christel := final_dolls_andrena = final_dolls_christel + 2
def belissa_equals_debelyn := final_dolls_belissa = final_dolls_debelyn

-- Proof Statement
theorem andrena_has_more_dolls_than_debelyn :
  andrena_more_than_christel →
  belissa_equals_debelyn →
  final_dolls_andrena - final_dolls_debelyn = 4 :=
by
  sorry

end andrena_has_more_dolls_than_debelyn_l260_260382


namespace vasya_has_greater_area_l260_260745

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l260_260745


namespace geom_seq_necessary_not_sufficient_l260_260187

theorem geom_seq_necessary_not_sufficient (a1 : ℝ) (q : ℝ) (h1 : 0 < a1) :
  (∀ n : ℕ, n > 0 → a1 * q^(2*n - 1) + a1 * q^(2*n) < 0) ↔ q < 0 :=
sorry

end geom_seq_necessary_not_sufficient_l260_260187


namespace complex_multiplication_complex_division_l260_260374

open Complex

theorem complex_multiplication :
  (1 - 2*Complex.i) * (3 + 4*Complex.i) * (-2 + Complex.i) = 12 + 9*Complex.i :=
by
  sorry

theorem complex_division :
  (1 + 2*Complex.i) / (3 - 4*Complex.i) = -1/5 + 2/5*Complex.i :=
by
  sorry

end complex_multiplication_complex_division_l260_260374


namespace sum_of_real_solutions_eq_l260_260422

theorem sum_of_real_solutions_eq :
  let f1 (x : ℝ) := (x - 3) / (x^2 + 5*x + 3)
  let f2 (x : ℝ) := (x - 4) / (x^2 - 8*x + 2)
  (f1 = f2) → (∑ x in (roots_of_polynomial (12 * x^2 - 43 * x + 6)), x) = 43 / 12 :=
by
  sorry

end sum_of_real_solutions_eq_l260_260422


namespace eight_digit_number_divisible_by_101_l260_260350

def repeat_twice (x : ℕ) : ℕ := 100 * x + x

theorem eight_digit_number_divisible_by_101 (ef gh ij kl : ℕ) 
  (hef : ef < 100) (hgh : gh < 100) (hij : ij < 100) (hkl : kl < 100) :
  (100010001 * repeat_twice ef + 1000010 * repeat_twice gh + 10010 * repeat_twice ij + 10 * repeat_twice kl) % 101 = 0 := sorry

end eight_digit_number_divisible_by_101_l260_260350


namespace range_of_x_for_sqrt_meaningful_l260_260500

theorem range_of_x_for_sqrt_meaningful (x : ℝ) (h : x + 2 ≥ 0) : x ≥ -2 :=
by {
  sorry
}

end range_of_x_for_sqrt_meaningful_l260_260500


namespace lattice_points_on_sphere_at_distance_5_with_x_1_l260_260166

theorem lattice_points_on_sphere_at_distance_5_with_x_1 :
  let points := [(1, 0, 4), (1, 0, -4), (1, 4, 0), (1, -4, 0),
                 (1, 2, 4), (1, 2, -4), (1, -2, 4), (1, -2, -4),
                 (1, 4, 2), (1, 4, -2), (1, -4, 2), (1, -4, -2),
                 (1, 2, 2), (1, 2, -2), (1, -2, 2), (1, -2, -2)]
  (hs : ∀ y z, (1, y, z) ∈ points → 1^2 + y^2 + z^2 = 25) →
  24 = points.length :=
sorry

end lattice_points_on_sphere_at_distance_5_with_x_1_l260_260166


namespace grandpa_tomatoes_before_vacation_l260_260156

theorem grandpa_tomatoes_before_vacation 
  (tomatoes_after_vacation : ℕ) 
  (growth_factor : ℕ) 
  (actual_number : ℕ) 
  (h1 : growth_factor = 100) 
  (h2 : tomatoes_after_vacation = 3564) 
  (h3 : actual_number = tomatoes_after_vacation / growth_factor) : 
  actual_number = 36 := 
by
  -- Here would be the step-by-step proof, but we use sorry to skip it
  sorry

end grandpa_tomatoes_before_vacation_l260_260156


namespace find_square_sum_l260_260133

theorem find_square_sum (x y z : ℝ)
  (h1 : x^2 - 6 * y = 10)
  (h2 : y^2 - 8 * z = -18)
  (h3 : z^2 - 10 * x = -40) :
  x^2 + y^2 + z^2 = 50 :=
sorry

end find_square_sum_l260_260133


namespace lithium_carbonate_price_in_august_l260_260268

def mean (xs : List ℚ) : ℚ := (xs.sum) / (xs.length)

def regression_slope (xs ys : List ℚ) (mean_x mean_y : ℚ) : ℚ := 
  (List.zipWith List.cons xs ys).sum / 
  (List.zipWith List.cons xs xs).sum

theorem lithium_carbonate_price_in_august :
  let month_codes := [1, 2, 3, 4, 5]
  let prices := [0.5, 0.7, 1, 1.2, 1.6]
  let mean_x := mean month_codes
  let mean_y := mean prices
  let regression_eq := λ (x : ℚ), regression_slope month_codes prices mean_x mean_y * x + 0.19
  regression_eq 8 = 2.35 := by
  sorry

end lithium_carbonate_price_in_august_l260_260268


namespace maximum_sine_sum_l260_260835

open Real

theorem maximum_sine_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : x ≤ π / 2) (hz : 0 ≤ y) (hw : y ≤ π / 2) (hv : 0 ≤ z) (hu : z ≤ π / 2) :
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x y z : ℝ, 0 ≤ x → x ≤ π / 2 → 0 ≤ y → y ≤ π / 2 → 0 ≤ z → z ≤ π / 2 → 
  sin (x - y) + sin (y - z) + sin (z - x) ≤ M :=
by
  sorry

end maximum_sine_sum_l260_260835


namespace gabby_l260_260047

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end gabby_l260_260047


namespace value_of_a3_l260_260257

def a_n (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem value_of_a3 : a_n 3 = -10 :=
by
  -- The proof would go here.
  sorry

end value_of_a3_l260_260257


namespace pyramid_total_surface_area_l260_260373

theorem pyramid_total_surface_area :
  ∀ (s h : ℝ), s = 8 → h = 10 →
  6 * (1/2 * s * (Real.sqrt (h^2 - (s/2)^2))) = 48 * Real.sqrt 21 :=
by
  intros s h s_eq h_eq
  rw [s_eq, h_eq]
  sorry

end pyramid_total_surface_area_l260_260373


namespace robin_total_distance_l260_260962

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end robin_total_distance_l260_260962


namespace theta_values_l260_260105

noncomputable theory

open Real

def is_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem theta_values (theta : ℝ) :
  is_perpendicular (cos (42 * π / 180), sin (42 * π / 180)) (cos theta, sin theta) ↔
  ∃ k : ℤ, theta = 132 * π / 180 + k * 180 * π / 180 :=
by
  sorry

end theta_values_l260_260105


namespace andy_questions_wrong_l260_260509

variable (a b c d : ℕ)

theorem andy_questions_wrong
  (h1 : a + b = c + d)
  (h2 : a + d = b + c + 6)
  (h3 : c = 7)
  (h4 : d = 9) :
  a = 10 :=
by {
  sorry  -- Proof would go here
}

end andy_questions_wrong_l260_260509


namespace field_trip_vans_l260_260213

-- Define the number of students and adults
def students := 12
def adults := 3

-- Define the capacity of each van
def van_capacity := 5

-- Total number of people
def total_people := students + adults

-- Calculate the number of vans needed
def vans_needed := (total_people + van_capacity - 1) / van_capacity  -- For rounding up division

theorem field_trip_vans : vans_needed = 3 :=
by
  -- Calculation and proof would go here
  sorry

end field_trip_vans_l260_260213


namespace topsoil_cost_l260_260641

theorem topsoil_cost :
  let cubic_yard_to_cubic_foot := 27
  let cubic_feet_in_5_cubic_yards := 5 * cubic_yard_to_cubic_foot
  let cost_per_cubic_foot := 6
  let total_cost := cubic_feet_in_5_cubic_yards * cost_per_cubic_foot
  total_cost = 810 :=
by
  sorry

end topsoil_cost_l260_260641


namespace pizza_slices_l260_260369

theorem pizza_slices (number_of_people : ℕ) (slices_per_person : ℕ) (number_of_pizzas : ℕ) :
  number_of_people = 10 → 
  slices_per_person = 2 → 
  number_of_pizzas = 5 → 
  (number_of_people * slices_per_person) / number_of_pizzas = 4 := 
by 
  intros h_people h_slices h_pizzas 
  rw [h_people, h_slices, h_pizzas]
  norm_num
  sorry

end pizza_slices_l260_260369


namespace synthetic_method_for_cosine_identity_l260_260521

theorem synthetic_method_for_cosine_identity (θ : ℝ) : 
  let lhs := cos (4 * θ) - sin (4 * θ)
  let step1 := (cos (2 * θ) + sin (2 * θ)) * (cos (2 * θ) - sin (2 * θ))
  let step2 := cos (2 * θ) - sin (2 * θ)
  lhs = cos (2 * θ) 
  → step1 = step2 
  → step2 = cos (2 * θ)
  → (step1 = step2 ∧ step2 = cos (2 * θ) → lhs = cos (2 * θ))
  → true
:= 
by
  intros
  sorry

end synthetic_method_for_cosine_identity_l260_260521


namespace problem_statement_l260_260559

variables (a b : ℝ^3) (θ : ℝ)
def a_unit_vector : Prop := ‖a‖ = 1
def b_magnitude : Prop := ‖b‖ = real.sqrt 2
def a_minus_b_magnitude : Prop := ‖a - b‖ = 1
def a_boxplus_b : ℝ := ‖a * real.sin θ + b * real.cos θ‖

theorem problem_statement (h1 : a_unit_vector a)
                         (h2 : b_magnitude b)
                         (h3 : a_minus_b_magnitude a b) :
  a_boxplus_b a b θ = real.sqrt 10 / 2 :=
sorry

end problem_statement_l260_260559


namespace problem_statement_l260_260558

variables (a b : ℝ^3) (θ : ℝ)
def a_unit_vector : Prop := ‖a‖ = 1
def b_magnitude : Prop := ‖b‖ = real.sqrt 2
def a_minus_b_magnitude : Prop := ‖a - b‖ = 1
def a_boxplus_b : ℝ := ‖a * real.sin θ + b * real.cos θ‖

theorem problem_statement (h1 : a_unit_vector a)
                         (h2 : b_magnitude b)
                         (h3 : a_minus_b_magnitude a b) :
  a_boxplus_b a b θ = real.sqrt 10 / 2 :=
sorry

end problem_statement_l260_260558


namespace min_cuts_for_5x5_square_l260_260653

-- Prove that the minimum number of straight cuts required to divide a 5 x 5 square 
-- into unit squares, given pieces can be rearranged, is 6.
theorem min_cuts_for_5x5_square : 
  ∃ n : ℕ, n = 6 ∧ (∀ cuts : ℕ, (cuts < n) → ¬(can_divide_into_unit_squares 5 5 cuts)) :=
sorry

-- Define can_divide_into_unit_squares
def can_divide_into_unit_squares (width : ℕ) (height : ℕ) (cuts : ℕ) : Prop :=
sorry

end min_cuts_for_5x5_square_l260_260653


namespace charcoal_needed_total_l260_260175

def charcoal_total
 (c1 : ℝ := 2) (w1 : ℝ := 30) (v1 : ℝ := 900)
 (c2 : ℝ := 3) (w2 : ℝ := 50) (v2 : ℝ := 1150)
 (c3 : ℝ := 4) (w3 : ℝ := 80) (v3 : ℝ := 1615)
 (c4 : ℝ := 2.3) (w4 : ℝ := 25) (v4 : ℝ := 675)
 (c5 : ℝ := 5.5) (w5 : ℝ := 115) (v5 : ℝ := 1930) : ℝ :=
 (c1 / w1 * v1) + (c2 / w2 * v2) + (c3 / w3 * v3) + (c4 / w4 * v4) + (c5 / w5 * v5)

theorem charcoal_needed_total :
  charcoal_total = 363.28 :=
by
  -- This is where the proof would go
  sorry

end charcoal_needed_total_l260_260175


namespace pool_fill_time_l260_260921

theorem pool_fill_time
  (faster_pipe_time : ℝ) (slower_pipe_factor : ℝ)
  (H1 : faster_pipe_time = 9) 
  (H2 : slower_pipe_factor = 1.25) : 
  (faster_pipe_time * (1 + slower_pipe_factor) / (faster_pipe_time + faster_pipe_time/slower_pipe_factor)) = 5 :=
by
  sorry

end pool_fill_time_l260_260921


namespace highest_weekly_sales_is_60_l260_260700

/-- 
Given that a convenience store sold 300 bags of chips in a month,
and the following weekly sales pattern:
1. In the first week, 20 bags were sold.
2. In the second week, there was a 2-for-1 promotion, tripling the sales to 60 bags.
3. In the third week, a 10% discount doubled the sales to 40 bags.
4. In the fourth week, sales returned to the first week's number, 20 bags.
Prove that the number of bags of chips sold during the week with the highest demand is 60.
-/
theorem highest_weekly_sales_is_60 
  (total_sales : ℕ)
  (week1_sales : ℕ)
  (week2_sales : ℕ)
  (week3_sales : ℕ)
  (week4_sales : ℕ)
  (h_total : total_sales = 300)
  (h_week1 : week1_sales = 20)
  (h_week2 : week2_sales = 3 * week1_sales)
  (h_week3 : week3_sales = 2 * week1_sales)
  (h_week4 : week4_sales = week1_sales) :
  max (max week1_sales week2_sales) (max week3_sales week4_sales) = 60 := 
sorry

end highest_weekly_sales_is_60_l260_260700


namespace length_of_congruent_l260_260249

-- Define the base of the isosceles triangle
def base (DEF : Triangle) : ℝ := 30

-- Define the area of the isosceles triangle
def area (DEF : Triangle) : ℝ := 96

-- Define the length of one of the congruent sides
noncomputable def length_of_congruent_side (DEF : Triangle) : ℝ :=
  sqrt ( (base DEF / 2)^2 + (2 * (area DEF / base DEF))^2 )

-- Lean 4 statement to be proved
theorem length_of_congruent sides_of_isosceles_triangle
  (DEF : Triangle) 
  (h1 : base DEF = 30) 
  (h2 : area DEF = 96) : 
  length_of_congruent_side DEF = 16.31 :=
by 
  sorry

end length_of_congruent_l260_260249


namespace coef_x4y3_l260_260950

theorem coef_x4y3 (x y : ℝ) : 
  coeff (expand (x^2 - x + y)^5) (x^4 * y^3) = 10 := sorry

end coef_x4y3_l260_260950


namespace san_antonio_bus_passes_4_austin_buses_l260_260371

theorem san_antonio_bus_passes_4_austin_buses :
  ∀ (hourly_austin_buses : ℕ → ℕ) (every_50_minute_san_antonio_buses : ℕ → ℕ) (trip_time : ℕ),
    (∀ h : ℕ, hourly_austin_buses (h) = (h * 60)) →
    (∀ m : ℕ, every_50_minute_san_antonio_buses (m) = (m * 60 + 50)) →
    trip_time = 240 →
    ∃ num_buses_passed : ℕ, num_buses_passed = 4 :=
by
  sorry

end san_antonio_bus_passes_4_austin_buses_l260_260371


namespace evaluate_expression_at_1_l260_260929

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x

theorem evaluate_expression_at_1 :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 ∧ x ≠ -2 ∧ x ≠ 0 ∧ x ≠ 2 → simplify_expression 1 = 10 :=
by
  intros x hx
  apply hx.1
  sorry

end evaluate_expression_at_1_l260_260929


namespace log_bn_an_log_ban_an1_log_bn_an1_log_bn_a_l260_260226

variable (a b n : ℝ)

-- Statement for Proof 1
theorem log_bn_an : log (a * real.log n) = (log a + real.log b) / (1 + real.log b) :=
sorry

-- Statement for Proof 2
theorem log_ban_an1 : log ((a) * real.log (a ^ n)) = (n + 1) * real.log a / (1 + n * real.log a) :=
sorry

-- Statement for Proof 3
theorem log_bn_an1 : log ((a) * real.log (a ^ n) = (real.log a + n)) / (1 + n) :=
sorry

-- Statement for Proof 4
theorem log_bn_a : log (((a) * b ^ n)) = 1 / (real.log a / (1 + n * real.log a)) :=
sorry

end log_bn_an_log_ban_an1_log_bn_an1_log_bn_a_l260_260226


namespace num_mappings_l260_260841

noncomputable def count_functions {n k : ℕ} (h1 : 1 ≤ n) (h2 : n / 2 ≤ k) (h3 : k ≤ n) : ℕ :=
  n! / ((n - k)! * (2 * k - n)!)

theorem num_mappings (Z : Finset ℕ) (f : ℕ → ℕ) (n k : ℕ) (hZ : ∀ x, x ∈ Z → 1 ≤ x ∧ x ≤ n)
  (hk1 : n / 2 ≤ k) (hk2 : k ≤ n) (f_idempotent : ∀ x, f (f x) = f x)
  (f_range_k : ∃! S : Finset ℕ, S.card = k ∧ ∀ x, x ∈ S ↔ ∃ y, y ∈ Z ∧ f y = x)
  (at_most_two : ∀ y ∈ ∃(!) S : Finset ℕ, S.card = k ∧ ∀ x, x ∈ S ↔ ∃ y, y ∈ Z ∧ f y = x),
  (∃! S : Finset ℕ, Set.card S = 1 → Set.card (S = k) → Set.filter (λ y : ℕ, ∃ x ∈ Z, f x = y) = 2 → 2 * sets.card (S) - x)) 
  : ∃ n f k ∃ count, count = n := sorry

end num_mappings_l260_260841


namespace pedestrian_meets_cart_at_10_40_l260_260347

-- Declare all participants' movements with constant speeds
variables (cyclist pedestrian cart car : ℝ)
variables (t : ℝ) -- Time in hours

-- Conditions based on the problem
axiom cyclist_overtakes_pedestrian_at_10 (h1 : t = 10) : cyclist := pedestrian
axiom pedestrian_meets_car_at_11 (h2 : t = 11) : pedestrian := car
axiom equal_time_intervals (t_L_t_C t_C_t_K : ℝ) :
  t_L_t_C = t_C_t_K ∧ t_L_t_C > 0 ∧ t_C_t_K > 0

-- Conclusion we want to prove
theorem pedestrian_meets_cart_at_10_40 : t = 10 + (2/3 : ℝ) := sorry

end pedestrian_meets_cart_at_10_40_l260_260347


namespace find_x_l260_260248

-- Define the condition as a theorem
theorem find_x (x : ℝ) (h : (1 + 3 + x) / 3 = 3) : x = 5 :=
by
  sorry  -- Placeholder for the proof

end find_x_l260_260248


namespace fertilizer_needed_per_acre_l260_260893

-- Definitions for the conditions
def horse_daily_fertilizer : ℕ := 5 -- Each horse produces 5 gallons of fertilizer per day.
def horses : ℕ := 80 -- Janet has 80 horses.
def days : ℕ := 25 -- It takes 25 days until all her fields are fertilized.
def total_acres : ℕ := 20 -- Janet's farmland is 20 acres.

-- Calculated intermediate values
def total_fertilizer : ℕ := horse_daily_fertilizer * horses * days -- Total fertilizer produced
def fertilizer_per_acre : ℕ := total_fertilizer / total_acres -- Fertilizer needed per acre

-- Theorem to prove
theorem fertilizer_needed_per_acre : fertilizer_per_acre = 500 := by
  sorry

end fertilizer_needed_per_acre_l260_260893


namespace question1_question2_l260_260824

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3) * cos x

theorem question1 (x : ℝ) (k : ℤ) : 
  f x = 0 ↔ (x = k * π - π / 3 ∨ x = -π / 2 + k * π) := sorry

noncomputable def g (x : ℝ) : ℝ := sin (x / 2 - π / 3) + √3 / 2

theorem question2 (x : ℝ) (k : ℤ) :
  ∃ I : Set ℝ, 
    (I = Set.Icc (4 * k * π - π / 3) (4 * k * π + 5 * π / 3)) ∧ 
    (∀ x ∈ I, ∀ y ∈ I, x < y → g x < g y) := sorry

end question1_question2_l260_260824


namespace eccentricity_of_ellipse_l260_260067

theorem eccentricity_of_ellipse (a b c : ℝ) (e : ℝ)
    (h1 : a > b) (h2 : b > 0) (h3 : c = real.sqrt (a^2 - b^2)) : 
    (∃ x y: ℝ, (2 * x - 2 * c = 0) ∧ (2 * y = -b^2 / a) ∧ (x^2 * 4 / a^2 + (y^2) * 4 / b^2 = 1)) →
    e = c / a :=
begin
    intros,
    rw ← real.sqrt_mul_self at h3,
    exact sorry
end

end eccentricity_of_ellipse_l260_260067


namespace domain_g_l260_260789

def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.sqrt (3 - x)

noncomputable def g (x : ℝ) : ℝ := f (x + 1) / (x - 1)

theorem domain_g :
  {x : ℝ | -2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2 } =
  {x | ∃ y ∈ Icc (-1:ℝ) 3, y = x + 1} ∩ {x | x ≠ 1} :=
by {
  sorry
}

end domain_g_l260_260789


namespace projectile_height_reaches_49_feet_l260_260251

theorem projectile_height_reaches_49_feet (t : ℝ) : 
  ∃ t : ℝ, (t ≈ 0.6) ∧ (-20 * t^2 + 90 * t = 49) :=
sorry

end projectile_height_reaches_49_feet_l260_260251


namespace juliet_younger_than_ralph_l260_260896

-- Define the ages of Juliet, Maggie, and Ralph as integers
variables (J M R : ℤ)

-- Conditions from the problem
def condition1 : Prop := J = M + 3        -- Juliet is 3 years older than Maggie
def condition2 : Prop := J = 10           -- Juliet is 10 years old
def condition3 : Prop := M + R = 19       -- The sum of Maggie's and Ralph's ages is 19

-- Statement to prove: Juliet is 2 years younger than Ralph
theorem juliet_younger_than_ralph (J M R : ℤ) (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  R - J = 2 :=
sorry

end juliet_younger_than_ralph_l260_260896


namespace largest_three_digit_number_divisible_by_12_and_its_digits_l260_260019

theorem largest_three_digit_number_divisible_by_12_and_its_digits :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ (digits 10 n).to_finset, d ≠ 0 ∧ n % d = 0) ∧ n % 12 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (∀ d ∈ (digits 10 m).to_finset, d ≠ 0 ∧ m % d = 0) ∧ m % 12 = 0 → m ≤ n :=
begin
  use 864,
  sorry
end

end largest_three_digit_number_divisible_by_12_and_its_digits_l260_260019


namespace smallest_possible_value_of_a_l260_260539

noncomputable def smallest_possible_a : ℕ :=
  let a := 1680
  in if
    ∀ P : ℚ[X], 
      (P.eval 1 = ↑a ∧ P.eval 3 = ↑a ∧ P.eval 5 = ↑a ∧ P.eval 7 = ↑a ∧ P.eval 9 = ↑a) ∧ 
      (P.eval 2 = -↑a ∧ P.eval 4 = -↑a ∧ P.eval 6 = -↑a ∧ P.eval 8 = -↑a ∧ P.eval 10 = -↑a) 
      then true else false
  then a else 0

theorem smallest_possible_value_of_a : smallest_possible_a = 1680 :=
sorry

end smallest_possible_value_of_a_l260_260539


namespace simplify_complex_expression_l260_260932

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l260_260932


namespace number_of_non_congruent_triangles_l260_260110

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_congruent_triangles_with_perimeter_12 : ℕ :=
  { (a, b, c) | a ≤ b ∧ b ≤ c ∧ a + b + c = 12 ∧ is_triangle a b c }.to_finset.card

theorem number_of_non_congruent_triangles : non_congruent_triangles_with_perimeter_12 = 2 := sorry

end number_of_non_congruent_triangles_l260_260110


namespace thirteen_power_1997_tens_digit_l260_260399

def tens_digit (n : ℕ) := (n / 10) % 10

theorem thirteen_power_1997_tens_digit :
  tens_digit (13 ^ 1997 % 100) = 5 := by
  sorry

end thirteen_power_1997_tens_digit_l260_260399


namespace vasya_expected_area_greater_l260_260749

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l260_260749


namespace linear_or_constant_l260_260471

noncomputable def bernsteinBasis (n k : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n k) * (x^k) * ((1 - x)^(n - k))

noncomputable def f (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), a k * bernsteinBasis n k x

theorem linear_or_constant (a : ℕ → ℝ) (h : ∀ i, 1 ≤ i -> a (i - 1) + a (i + 1) = 2 * a i) (n : ℕ) :
  ∃ c d : ℝ, ∀ x : ℝ, f a n x = c + d * n * x :=
begin
  sorry
end

end linear_or_constant_l260_260471


namespace expression_parity_l260_260200

variable (o n c : ℕ)

def is_odd (x : ℕ) : Prop := ∃ k, x = 2 * k + 1

theorem expression_parity (ho : is_odd o) (hc : is_odd c) : 
  (o^2 + n * o + c) % 2 = 0 :=
  sorry

end expression_parity_l260_260200


namespace pool_fill_time_l260_260591

def pool_capacity : ℕ := 32000
def gallons_per_minute_per_hose : ℕ := 3
def number_of_hoses : ℕ := 3
def minutes_per_hour : ℕ := 60

theorem pool_fill_time : 
  (pool_capacity.toRat / ((gallons_per_minute_per_hose * number_of_hoses * minutes_per_hour).toRat)).round = 59 :=
by
  sorry

end pool_fill_time_l260_260591


namespace proof_problem_l260_260664

def A_condition : Prop := ∀ x : ℕ, x ∈ ℚ
def B_condition : Prop := ¬ ∃ x : {n // 0 < n}, x.val ^ 2 - 3 < 0
def C_condition : ∀ a b c : ℝ, 
  (a^2 + b^2 = c^2 ↔ is_right_triangle a b c)
def D_condition : Prop := ∀ x : ℝ, x > 0 → x^2 - 3 > 0 ↔ ∃ x : ℝ, x > 0 ∧ x^2 - 3 ≤ 0

theorem proof_problem : B_condition = false ∧ C_condition = false := 
sorry

end proof_problem_l260_260664


namespace Seohyeon_l260_260525

-- Define the distances in their respective units
def d_Kunwoo_km : ℝ := 3.97
def d_Seohyeon_m : ℝ := 4028

-- Convert Kunwoo's distance to meters
def d_Kunwoo_m : ℝ := d_Kunwoo_km * 1000

-- The main theorem we need to prove
theorem Seohyeon's_distance_longer_than_Kunwoo's :
  d_Seohyeon_m > d_Kunwoo_m :=
by
  sorry

end Seohyeon_l260_260525


namespace distance_from_point_to_line_l260_260413

noncomputable def distance_point_to_line
(point : ℝ × ℝ × ℝ)
(line_point direction_vector : ℝ × ℝ × ℝ)
: ℝ :=
let \(\begin{pmatrix} x_0, y_0, z_0 \end{pmatrix}\) := point in
let \(\begin{pmatrix} x, y, z \end{pmatrix}\) := line_point in
let \(\begin{pmatrix} u, v, w \end{pmatrix}\) := direction_vector in
real.sqrt ((x - x_0 - 11/34 * u)^2 + (y - y_0 - 11/34 * v)^2 + (z - z_0 + 11/34 * w)^2)

theorem distance_from_point_to_line : 
  distance_point_to_line (2, 3, 4) (4, 5, 5) (4, 3, -3) = real.sqrt 5.44 :=
by
  sorry

end distance_from_point_to_line_l260_260413


namespace vasya_expected_area_greater_l260_260748

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l260_260748


namespace distinct_real_numbers_ab_sum_is_four_l260_260431

theorem distinct_real_numbers_ab_sum_is_four
  (a b : ℝ)
  (h_distinct : a ≠ b)
  (h_M : M = {a^2 - 4 * a, -1})
  (h_N : N = {b^2 - 4 * b + 1, -2})
  (h_mapping : ∀ x ∈ M, x ∈ N) :
  a + b = 4 :=
sorry

end distinct_real_numbers_ab_sum_is_four_l260_260431


namespace floor_sequence_square_eq_l260_260231

def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 2
  else sequence_a (n - 1) / (n - 1) + (n - 1) / sequence_a (n - 1)

theorem floor_sequence_square_eq (n : ℕ) (hn : n ≥ 4) : 
  ∀ (n : ℕ), n ≥ 4 → (Nat.floor (sequence_a n ^ 2 : ℝ)) = n := sorry

end floor_sequence_square_eq_l260_260231


namespace parabola_equation_and_AB_length_l260_260345

-- Definitions
def parabola_vertex_origin_focus_y_axis := ∀ (x y : ℝ), x^2 = 4 * (2 * y)
def line_l := ∀ (x y : ℝ), y = 2 * x + 1

-- Problem statement
theorem parabola_equation_and_AB_length:
  (∀ (x y : ℝ), parabola_vertex_origin_focus_y_axis x y → x^2 = 4 * y) ∧
  ( ∃ (x1 x2 y1 y2 : ℝ), 
    parabola_vertex_origin_focus_y_axis x1 y1 ∧ 
    parabola_vertex_origin_focus_y_axis x2 y2 ∧ 
    line_l x1 y1 ∧ 
    line_l x2 y2 ∧ 
    y1 + y2 + 2 = 20) :=
sorry

end parabola_equation_and_AB_length_l260_260345


namespace variance_of_binomial_distribution_l260_260497

noncomputable def variance_binomial :=
  let n := 8
  let p := 0.7
  n * p * (1 - p)

theorem variance_of_binomial_distribution : variance_binomial = 1.68 :=
  by
  sorry

end variance_of_binomial_distribution_l260_260497


namespace price_of_water_margin_comics_l260_260375

-- Define the conditions
variables (x : ℕ) (y : ℕ)

-- Condition 1: Price relationship
def price_relationship : Prop := y = x + 60

-- Condition 2: Total expenditure on Romance of the Three Kingdoms comic books
def total_expenditure_romance_three_kingdoms : Prop := 60 * (y / 60) = 3600

-- Condition 3: Total expenditure on Water Margin comic books
def total_expenditure_water_margin : Prop := 120 * (x / 120) = 4800

-- Condition 4: Number of sets relationship
def number_of_sets_relationship : Prop := y = (4800 / x) / 2

-- The main statement to prove
theorem price_of_water_margin_comics (x : ℕ) (h1: price_relationship x (x + 60))
  (h2: total_expenditure_romance_three_kingdoms x)
  (h3: total_expenditure_water_margin x)
  (h4: number_of_sets_relationship x (x + 60)) : x = 120 :=
sorry

end price_of_water_margin_comics_l260_260375


namespace min_sum_of_digits_of_S_l260_260285

def distinct_digits (n : ℕ) : Prop :=
  let digits := (List.dedup (Nat.digits 10 n))
  digits.length = (Nat.digits 10 n).length

def three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).foldl (λ acc x => acc + x) 0

theorem min_sum_of_digits_of_S (a b : ℕ) (S : ℕ) :
  three_digit_number a →
  three_digit_number b →
  distinct_digits a →
  distinct_digits b →
  (Nat.digits 10 a).head!.getD 0 < 5 →
  (Nat.digits 10 a).head!.getD 0 ≠ 0 →
  a + b = S →
  sum_of_digits S = 15 := sorry

end min_sum_of_digits_of_S_l260_260285


namespace binomial_expansion_equality_l260_260324

theorem binomial_expansion_equality (x : ℝ) : 
  (x-1)^4 - 4*x*(x-1)^3 + 6*(x^2)*(x-1)^2 - 4*(x^3)*(x-1)*x^4 = 1 := 
by 
  sorry 

end binomial_expansion_equality_l260_260324


namespace simplify_complex_expression_l260_260931

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l260_260931


namespace find_digits_l260_260388

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l260_260388


namespace f_maximum_value_of_negative_x_l260_260452

noncomputable def f : ℝ → ℝ := sorry  -- Function definition to satisfy the conditions

theorem f_maximum_value_of_negative_x
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_diff : ∀ x : ℝ, x > 0 → x ≠ 1 → (x - 1) * f' x > 0)
  (h_value : f 1 = 2) :
  ∀ x : ℝ, x < 0 → f x ≤ -2 :=
sorry

end f_maximum_value_of_negative_x_l260_260452


namespace nancy_carrots_next_day_l260_260212

-- Definitions based on conditions
def carrots_picked_on_first_day : Nat := 12
def carrots_thrown_out : Nat := 2
def total_carrots_after_two_days : Nat := 31

-- Problem statement
theorem nancy_carrots_next_day :
  let carrots_left_after_first_day := carrots_picked_on_first_day - carrots_thrown_out
  let carrots_picked_next_day := total_carrots_after_two_days - carrots_left_after_first_day
  carrots_picked_next_day = 21 :=
by
  sorry

end nancy_carrots_next_day_l260_260212


namespace modulus_of_z_is_2_l260_260566

def z : ℂ :=
  ((6 : ℂ) + (4 : ℂ) * complex.I) / ((2 : ℂ) - (3 : ℂ) * complex.I)

theorem modulus_of_z_is_2 : complex.abs z = 2 := by
  sorry

end modulus_of_z_is_2_l260_260566


namespace sector_central_angle_l260_260457

-- Defining the problem as a theorem in Lean 4
theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 4) (h2 : (1 / 2) * r^2 * θ = 1) : θ = 2 :=
by
  sorry

end sector_central_angle_l260_260457


namespace vector_v_satisfies_conditions_l260_260901

def vec3 : Type := (ℝ × ℝ × ℝ)

def cross_product (u v : vec3) : vec3 :=
  (u.2.2 * v.3 - u.3 * v.2.2,
   u.3 * v.1 - u.1 * v.3,
   u.1 * v.2.2 - u.2.2 * v.1
   )

def a : vec3 := (3, 2, 1)
def b : vec3 := (1, -1, 2)
def v : vec3 := (4, 1, 3)

theorem vector_v_satisfies_conditions :
  cross_product v a = cross_product b a ∧ cross_product v b = cross_product a b :=
by
  sorry

end vector_v_satisfies_conditions_l260_260901


namespace prime_dates_in_2008_l260_260001

noncomputable def num_prime_dates_2008 : Nat := 52

theorem prime_dates_in_2008 : 
  let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_months_days := [(2, 29), (3, 31), (5, 31), (7, 31), (11, 30)]
  -- Count the prime days for each month considering the list
  let prime_day_count (days : Nat) := (prime_days.filter (λ d => d <= days)).length
  -- Sum the counts for each prime month
  (prime_months_days.map (λ (m, days) => prime_day_count days)).sum = num_prime_dates_2008 :=
by
  sorry

end prime_dates_in_2008_l260_260001


namespace equivalent_fractions_l260_260201

variable {x y a c : ℝ}

theorem equivalent_fractions (h_nonzero_c : c ≠ 0) (h_transform : x = (a / c) * y) :
  (x + a) / (y + c) = a / c :=
by
  sorry

end equivalent_fractions_l260_260201


namespace card_probability_l260_260354

theorem card_probability (n m : ℕ)
  (h1 : (finset.range 44).card = 44)
  (h2 : 2 * 4 = 8)
  (total_ways : 44.choose 2 = 946)
  (pairs_remaining : 10 * nat.choose 4 2 = 60)
  (prob : (2 + 2) + 60 = 62)
  (simplified : nat.succ 30 * nat.succ 472 = 473 * 31)
  (rat_prod_eq : ((62 : ℚ) / 946) = (31 : ℚ) / 473)
  : (31 + 473 = 504) :=
begin
  sorry
end

end card_probability_l260_260354


namespace greater_expected_area_vasya_l260_260730

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l260_260730


namespace num_people_is_8_l260_260938

-- Define the known conditions
def bill_amt : ℝ := 314.16
def person_amt : ℝ := 34.91
def total_amt : ℝ := 314.19

-- Prove that the number of people is 8
theorem num_people_is_8 : ∃ num_people : ℕ, num_people = total_amt / person_amt ∧ num_people = 8 :=
by
  sorry

end num_people_is_8_l260_260938


namespace expression_decrease_l260_260520

theorem expression_decrease (x y : ℝ) : 
  let x' := 0.8 * x in
  let y' := 0.7 * y in
  let original_value := x * y^2 in
  let new_value := x' * y'^2 in
  new_value = 0.392 * original_value :=
by
  sorry

end expression_decrease_l260_260520


namespace find_ratio_CQ_QE_l260_260503

-- Let's define the setup
variables (A B C D E Q : Type) [geometry A B C] [line_segment CE] [line_segment AD] 

-- Define the ratios as given in the problem
def ratio_CD_DB (CD DB : ℝ) : Prop := CD / DB = 5 / 3
def ratio_AE_EB (AE EB : ℝ) : Prop := AE / EB = 7 / 3

-- Assume the intersection point Q
variable (is_intersection : ∀ {A B C D E Q : Type}, is_intersection_point CE AD Q)

-- Now state what we need to prove
theorem find_ratio_CQ_QE (CD DB AE EB CQ QE : ℝ) 
  (h1 : ratio_CD_DB CD DB) 
  (h2 : ratio_AE_EB AE EB) 
  (h3 : is_intersection) 
  : CQ / QE = 7 / 3 :=
sorry

end find_ratio_CQ_QE_l260_260503


namespace option_A_option_B_option_C_option_D_l260_260310

theorem option_A (a : ℝ) (ha : a > 0) : 2^a > 1 := 
by 
  sorry

theorem option_B (x y : ℝ) (h : x^2 + y^2 = 0) : x = 0 ∧ y = 0 := 
by 
  sorry

theorem option_C (a b c : ℝ) (h : b^2 = a * c) : ¬(a, b, c form_geom_seq) := 
by 
  sorry
  where 
    form_geom_seq := ∀ (a b c : ℝ), (b / a = c / b) ∨ (a = b ∧ b = c)

theorem option_D (a b c : ℝ) (h : a + c = 2 * b) : a, b, c form_arith_seq := 
by 
  sorry
  where
    form_arith_seq := ∀ (a b c : ℝ), b - a = c - b

end option_A_option_B_option_C_option_D_l260_260310


namespace infinitely_many_integers_not_of_form_l260_260926

theorem infinitely_many_integers_not_of_form (p : ℕ) (hp : Nat.Prime p) :
  ∃ᶠ n : ℕ in at_top, ∀ (a b c d : ℤ),
    (p^a - p^b) / (p^c - p^d) ∉ ℤ :=
begin
  sorry
end

end infinitely_many_integers_not_of_form_l260_260926


namespace amount_of_loan_l260_260588

theorem amount_of_loan (P R T SI : ℝ) (hR : R = 6) (hT : T = 6) (hSI : SI = 432) :
  SI = (P * R * T) / 100 → P = 1200 :=
by
  intro h
  sorry

end amount_of_loan_l260_260588


namespace find_ratio_of_y_l260_260242

theorem find_ratio_of_y (x y c x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x * y = c)
  (h2 : x₁ ≠ 0)
  (h3 : x₂ ≠ 0) 
  (h4 : y₁ ≠ 0)
  (h5 : y₂ ≠ 0)
  (h6 : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end find_ratio_of_y_l260_260242


namespace max_non_managers_l260_260680

theorem max_non_managers (x : ℕ) (h : (7:ℚ) / 32 < 9 / x) : x = 41 := sorry

end max_non_managers_l260_260680


namespace exists_infinite_coprime_sequence_l260_260524

noncomputable def sequence : ℕ → ℕ 
| 0       => 1
| (n + 1) => (3 * sequence n)! + 1

def gcd_coprime (a b : ℕ) : Prop :=
gcd a b = 1

def sequence_property : Prop :=
  ∀ i j p q r : ℕ, i ≠ j ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p →
  gcd_coprime ((sequence i) + (sequence j)) ((sequence p) + (sequence q) + (sequence r))

theorem exists_infinite_coprime_sequence : ∃ f : ℕ → ℕ, (∀ (n : ℕ), f n = sequence n) ∧ sequence_property :=
sorry

end exists_infinite_coprime_sequence_l260_260524


namespace complete_the_square_d_l260_260376

theorem complete_the_square_d (x : ℝ) (h : x^2 + 6 * x + 5 = 0) : ∃ d : ℝ, (x + 3)^2 = d ∧ d = 4 :=
by
  sorry

end complete_the_square_d_l260_260376


namespace ratio_of_volumes_l260_260655

theorem ratio_of_volumes (rC hC rD hD : ℝ) (h1 : rC = 10) (h2 : hC = 25) (h3 : rD = 25) (h4 : hD = 10) : 
  (1/3 * Real.pi * rC^2 * hC) / (1/3 * Real.pi * rD^2 * hD) = 2 / 5 :=
by
  sorry

end ratio_of_volumes_l260_260655


namespace exists_constant_C_inequality_for_difference_l260_260058

theorem exists_constant_C (a : ℕ → ℝ) (C : ℝ) (hC : 0 < C) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a n ≤ C * n^2) := sorry

theorem inequality_for_difference (a : ℕ → ℝ) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a (n + 1) - a n ≤ 4 * n + 3) := sorry

end exists_constant_C_inequality_for_difference_l260_260058


namespace solve_for_x_l260_260136

theorem solve_for_x (x : ℝ) (h : 3 / (x + 10) = 1 / (2 * x)) : x = 2 :=
sorry

end solve_for_x_l260_260136


namespace min_a_geq_neg_five_half_l260_260862

noncomputable def min_value_a (a : ℝ) :=
  ∀ x ∈ Ioo (0 : ℝ) (1 / 2 : ℝ), x^2 + a * x + 1 ≥ 0

theorem min_a_geq_neg_five_half : inf {a | min_value_a a} = -5 / 2 := 
sorry

end min_a_geq_neg_five_half_l260_260862


namespace range_of_m_l260_260190

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - 2 * m * x

def monotonically_decreasing_on {α : Type*} [Preorder α] (f : α → ℝ) (s : Set α) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Ici (1 : ℝ), (f x m)' ≤ 0) ↔ m ≥ 1 / 2 :=
by
  sorry

end range_of_m_l260_260190


namespace interest_rate_proof_l260_260217

def interest_rate_paid (purchase_price down_payment monthly_payment n_payments : ℝ) : ℝ :=
  let total_paid := down_payment + (monthly_payment * n_payments)
  let interest_amount := total_paid - purchase_price
  (interest_amount / purchase_price) * 100

theorem interest_rate_proof :
  interest_rate_paid 127 27 10 12 ≈ 15.7 :=
by
  sorry

end interest_rate_proof_l260_260217


namespace part1_part2_l260_260801

open Complex

theorem part1 {m : ℝ} : m + (m^2 + 2) * I = 0 -> m = 0 :=
by sorry

theorem part2 {m : ℝ} (h : (m + I)^2 - 2 * (m + I) + 2 = 0) :
    (let z1 := m + I
     let z2 := 2 + m * I
     im ((z2 / z1) : ℂ) = -1 / 2) :=
by sorry

end part1_part2_l260_260801


namespace sequence_general_term_correct_l260_260086

open Nat

def S (n : ℕ) : ℤ := 3 * (n : ℤ) * (n : ℤ) - 2 * (n : ℤ) + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2
  else 6 * (n : ℤ) - 5

theorem sequence_general_term_correct : ∀ n, (S n - S (n - 1) = a n) :=
by
  intros
  sorry

end sequence_general_term_correct_l260_260086


namespace find_real_number_a_l260_260840

theorem find_real_number_a (a : ℝ) :
  ({0, -1, 2 * a} = {a - 1, -abs a, a + 1}) → (a = 1 ∨ a = -1) :=
by
  sorry

end find_real_number_a_l260_260840


namespace vec_op_result_l260_260564

variables {θ : ℝ} (a b : ℝ ^ 3)

def unit_vector (v : ℝ ^ 3) : Prop := ∥v∥ = 1
def vec_mag_sqrt2 (v : ℝ ^ 3) : Prop := ∥v∥ = real.sqrt 2
def vec_diff_mag1 (u v : ℝ ^ 3) : Prop := ∥u - v∥ = 1
def vector_op (a b : ℝ ^ 3) (θ : ℝ) : ℝ := ∥a * real.sin θ + b * real.cos θ∥

theorem vec_op_result (a b : ℝ ^ 3) (θ : ℝ)
  (ha : unit_vector a)
  (hb : vec_mag_sqrt2 b)
  (habdiff : vec_diff_mag1 a b) :
  vector_op a b θ = real.sqrt 10 / 2 :=
sorry

end vec_op_result_l260_260564


namespace find_natural_numbers_l260_260760

theorem find_natural_numbers (x y z : ℕ) (hx : x ≤ y) (hy : y ≤ z) : 
    (1 + 1 / x) * (1 + 1 / y) * (1 + 1 / z) = 3 
    → (x = 1 ∧ y = 3 ∧ z = 8) 
    ∨ (x = 1 ∧ y = 4 ∧ z = 5) 
    ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
sorry

end find_natural_numbers_l260_260760


namespace cone_volume_equivalent_to_surface_area_l260_260587

noncomputable def radius : Type :=
  ℝ

noncomputable def height : Type :=
  ℝ

noncomputable def slantHeight : Type :=
  ℝ

noncomputable def distanceFromBaseToSlantHeight : Type :=
  ℝ

noncomputable def lateralSurfaceArea (r: radius) (l: slantHeight) :=
  π * r * l

noncomputable def coneVolume (r: radius) (h: height) :=
  (1/3) * π * r^2 * h

theorem cone_volume_equivalent_to_surface_area (r : radius) (h : height) (l : slantHeight) (d : distanceFromBaseToSlantHeight) (S : ℝ) :
  S = lateralSurfaceArea r l →
  coneVolume r h = (1/3) * d * S :=
sorry

end cone_volume_equivalent_to_surface_area_l260_260587


namespace new_average_mark_l260_260246

theorem new_average_mark (average_mark : ℕ) (average_excluded : ℕ) (total_students : ℕ) (excluded_students: ℕ)
    (h1 : average_mark = 90)
    (h2 : average_excluded = 45)
    (h3 : total_students = 20)
    (h4 : excluded_students = 2) :
  ((total_students * average_mark - excluded_students * average_excluded) / (total_students - excluded_students)) = 95 := by
  sorry

end new_average_mark_l260_260246


namespace tangent_line_parallel_extreme_values_l260_260829

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

theorem tangent_line_parallel (a : ℝ) (h : (1 : ℝ) - 1 + a / Real.exp 1 = 0) : a = Real.exp 1 := 
by sorry

theorem extreme_values (a : ℝ) :
  (a ≤ 0 → ∀ x, ¬(∃ local_min x (f a x) ∨ ∃ local_max x (f a x))) ∧
  (a > 0 → (∃ x, x = Real.log a ∧ ∀ y, y ≠ x → f a y < f a x ∧ f a y ≠ f a x)) :=
by sorry

end tangent_line_parallel_extreme_values_l260_260829


namespace base5_product_is_correct_l260_260650

-- Definitions for the problem context
def base5_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 528 then [4, 1, 0, 0, 3] else []

-- Theorem to prove the base-5 multiplication result
theorem base5_product_is_correct :
  base10_to_base5 (base5_to_base10 2 3 1 * base5_to_base10 1 3 0) = [4, 1, 0, 0, 3] :=
by
  sorry

end base5_product_is_correct_l260_260650


namespace plant_trees_l260_260672

theorem plant_trees :
  let w := 27.9
  let d := 3.1
  Int.floor (w / d) = 9 :=
by
  sorry

end plant_trees_l260_260672


namespace number_of_triangles_not_exceeding_ten_with_integer_side_lengths_and_unequal_sides_l260_260972

theorem number_of_triangles_not_exceeding_ten_with_integer_side_lengths_and_unequal_sides :
  let triangles := { (a, b, c) | a + b + c ≤ 10 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a ≠ b ∧ b ≠ c } in
  ∃ (n : ℕ), n = 11 ∧ (triangles.card = n) :=
by
  sorry

end number_of_triangles_not_exceeding_ten_with_integer_side_lengths_and_unequal_sides_l260_260972


namespace honor_students_count_l260_260988

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l260_260988


namespace Robin_total_distance_walked_l260_260959

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end Robin_total_distance_walked_l260_260959


namespace students_scoring_between_55_and_65_l260_260871

theorem students_scoring_between_55_and_65 (total_students : ℕ) (students_scored_at_least_55 : ℕ)
    (students_scored_at_most_65 : ℕ) (h1 : students_scored_at_least_55 = 0.55 * total_students)
    (h2 : students_scored_at_most_65 = 0.65 * total_students) :
    (students_scored_at_most_65 - (total_students - students_scored_at_least_55)) = 0.20 * total_students :=
by
  sorry

end students_scoring_between_55_and_65_l260_260871


namespace front_crawl_speed_l260_260637
   
   def swim_condition := 
     ∃ F : ℝ, -- Speed of front crawl in yards per minute
     (∃ t₁ t₂ d₁ d₂ : ℝ, -- t₁ is time for front crawl, t₂ is time for breaststroke, d₁ and d₂ are distances
               t₁ = 8 ∧
               t₂ = 4 ∧
               d₁ = t₁ * F ∧
               d₂ = t₂ * 35 ∧
               d₁ + d₂ = 500 ∧
               t₁ + t₂ = 12) ∧
     F = 45
   
   theorem front_crawl_speed : swim_condition :=
     by
       sorry -- Proof goes here, with given conditions satisfying F = 45
   
end front_crawl_speed_l260_260637


namespace find_vector_op_l260_260562

-- Definition of the vectors and their properties
variables (a b : ℝ) (θ : ℝ)

-- Conditions
def unit_vector_a : Prop := abs a = 1
def magnitude_b : Prop := abs b = real.sqrt 2
def magnitude_diff : Prop := abs (a - b) = 1
def angle_between_vectors (θ : ℝ) : Prop := cos θ = (real.sqrt 2) / 2 ∧ sin θ = (real.sqrt 2) / 2

-- The operation definition
def vector_op (a b θ: ℝ) : ℝ := abs (a * sin θ + b * cos θ)

-- The theorem statement with the conditions and result
theorem find_vector_op
  (ha : unit_vector_a a)
  (hb : magnitude_b b)
  (hd : magnitude_diff a b)
  (ht : angle_between_vectors θ) :
  vector_op a b θ = (real.sqrt 10) / 2 :=
sorry

end find_vector_op_l260_260562


namespace honor_students_count_l260_260990

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l260_260990


namespace num_non_congruent_triangles_with_perimeter_12_l260_260118

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l260_260118


namespace speed_ratio_l260_260358

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end speed_ratio_l260_260358


namespace smallest_positive_integer_square_mean_l260_260027

theorem smallest_positive_integer_square_mean :
  ∃ (n : ℕ), n > 1 ∧ (∃ (m : ℕ), (let sn := (∑ i in Finset.range (n+1), i^2) in (sn / n) = m^2) ∧ 
             (∀ (k : ℕ), k > 1 ∧ (∃ (m' : ℕ), (let sk := (∑ i in Finset.range (k+1), i^2) in (sk / k) = m'^2)) → 337 ≤ k)) :=
sorry

end smallest_positive_integer_square_mean_l260_260027


namespace sequence_a_1000_l260_260796

theorem sequence_a_1000 (a : ℕ → ℕ)
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n, a (n + 1) = 3 * a n - 2 * a (n - 1)) : 
  a 1000 = 2^1000 - 1 := 
sorry

end sequence_a_1000_l260_260796


namespace orthogonal_vectors_y_value_l260_260004

theorem orthogonal_vectors_y_value (y : ℝ) :
  (3 : ℝ) * (-1) + (4 : ℝ) * y = 0 → y = 3 / 4 :=
by
  sorry

end orthogonal_vectors_y_value_l260_260004


namespace find_divisor_l260_260142

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 34) / y = 2) : y = 10 :=
by
  sorry

end find_divisor_l260_260142


namespace simultaneous_eq_solvable_l260_260035

theorem simultaneous_eq_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 4 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_eq_solvable_l260_260035


namespace unique_solution_value_l260_260782

theorem unique_solution_value (k : ℝ) :
  (∃ x : ℝ, x^2 = 2 * x + k ∧ ∀ y : ℝ, y^2 = 2 * y + k → y = x) ↔ k = -1 := 
by
  sorry

end unique_solution_value_l260_260782


namespace base_8_digits_of_2147_l260_260843

theorem base_8_digits_of_2147 : ∀ n : ℕ, n = 2147 → ∃ d : ℕ, d = 4 ∧ nat.digits 8 n = d :=
by
  intro n
  intro h
  rw h
  existsi 4
  split
  refl
  sorry -- This is where the proof would go

end base_8_digits_of_2147_l260_260843


namespace adam_earning_per_lawn_l260_260723

theorem adam_earning_per_lawn 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : forgotten_lawns = 8) 
  (h3 : total_earnings = 36) : 
  total_earnings / (total_lawns - forgotten_lawns) = 9 :=
by
  sorry

end adam_earning_per_lawn_l260_260723


namespace discount_correct_l260_260718

noncomputable def discount_percentage (W : ℝ) : ℝ :=
  let R := W * 1.6428571428571428
  let S := W * 1.40
  (R - S) / R * 100

theorem discount_correct (W : ℝ) : discount_percentage W ≈ 14.77 :=
by
  sorry

end discount_correct_l260_260718


namespace cos_third_quadrant_l260_260855

theorem cos_third_quadrant (B : ℝ) (h1 : angle_in_third_quadrant B) (h2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

/-- You would need to define what it means for an angle to lie in the third quadrant. -/
def angle_in_third_quadrant (B : ℝ) : Prop := π < B ∧ B < 3 * π / 2

end cos_third_quadrant_l260_260855


namespace combined_mean_is_254_over_15_l260_260942

noncomputable def combined_mean_of_sets 
  (mean₁ : ℝ) (n₁ : ℕ) 
  (mean₂ : ℝ) (n₂ : ℕ) : ℝ :=
  (mean₁ * n₁ + mean₂ * n₂) / (n₁ + n₂)

theorem combined_mean_is_254_over_15 :
  combined_mean_of_sets 18 7 16 8 = (254 : ℝ) / 15 :=
by
  sorry

end combined_mean_is_254_over_15_l260_260942


namespace complex_fraction_equivalence_l260_260606

/-- The complex number 2 / (1 - i) is equal to 1 + i. -/
theorem complex_fraction_equivalence : (2 : ℂ) / (1 - (I : ℂ)) = 1 + (I : ℂ) := by
  sorry

end complex_fraction_equivalence_l260_260606


namespace packages_per_box_l260_260275

theorem packages_per_box (P : ℕ) (h1 : 192 > 0) (h2 : 2 > 0) (total_soaps : 2304 > 0) (h : 2 * P * 192 = 2304) : P = 6 :=
by
  sorry

end packages_per_box_l260_260275


namespace range_of_product_l260_260448

variables (a b : ℝ)

def condition1 := |a| ≤ 1
def condition2 := |a + b| ≤ 1

theorem range_of_product : 
  condition1 a ∧ condition2 a b →
  ∃ m M : ℝ, m = -2 ∧ M = 9/4 ∧
  ∀ x : ℝ, x = (a+1)*(b+1) → -2 ≤ x ∧ x ≤ 9/4 :=
begin
  sorry
end

end range_of_product_l260_260448


namespace find_ratio_of_y_l260_260241

theorem find_ratio_of_y (x y c x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x * y = c)
  (h2 : x₁ ≠ 0)
  (h3 : x₂ ≠ 0) 
  (h4 : y₁ ≠ 0)
  (h5 : y₂ ≠ 0)
  (h6 : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end find_ratio_of_y_l260_260241


namespace part1_part2_l260_260467

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 :
  (∀ x, f x ≤ 9) → ∀ x, x ∈ (Icc (-2 : ℝ) (4 : ℝ)) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) ↔ a ∈ Icc (19 / 4 : ℝ) (7 : ℝ) :=
by
  sorry

end part1_part2_l260_260467


namespace asymptotes_holes_sum_value_l260_260886

def f (x : ℝ) := (x^2 - 2*x + 1) / (x^3 - 3*x^2 + 2*x)

theorem asymptotes_holes_sum_value : 
  let a := 1 in -- number of holes
  let b := 2 in -- number of vertical asymptotes
  let c := 1 in -- number of horizontal asymptotes
  let d := 0 in -- number of oblique asymptotes
  a^2 + 2*b^2 + 3*c^2 + 4*d^2 = 12 :=
by
  sorry

end asymptotes_holes_sum_value_l260_260886


namespace solve_equation_l260_260599

theorem solve_equation (x : ℝ) (h : (x - 3) / 2 - (2 * x) / 3 = 1) : x = -15 := 
by 
  sorry

end solve_equation_l260_260599


namespace cistern_leak_empty_time_l260_260312

/-- 
A cistern which could be filled in 9 hours takes 1 hour more to be filled owing to a 
leak in its bottom. Prove that if the cistern is full, the leak will empty it in 90 hours.
-/
theorem cistern_leak_empty_time :
  (rate_fill : ℝ) 
  (rate_combined : ℝ)
  (rate_leak : ℝ) :
  rate_fill = 1 / 9 → 
  rate_combined = 1 / 10 → 
  rate_fill - rate_leak = rate_combined → 
  (rate_leak = 1 / 90) → 
  (time_to_empty : ℝ)  := 
begin
  sorry
end

end cistern_leak_empty_time_l260_260312


namespace a_beats_b_by_meters_l260_260873

theorem a_beats_b_by_meters :
  ∀ (A B : Type) (dist_km : ℝ) (time_a : ℝ) (time_diff : ℝ),
    dist_km = 1 ∧ time_a = 190 ∧ time_diff = 10 → 
    A_beats_B_by := dist_km * 1000 * time_diff / time_a 
    A_beats_B_by ≈ 52.63 :=
begin
  intros A B dist_km time_a time_diff h,
  cases h with hd ht,
  have ha : dist_km = 1, from hd.left,
  have hb : time_a = 190, from ht.left,
  have hc : time_diff = 10, from ht.right,
  sorry
end

end a_beats_b_by_meters_l260_260873


namespace greater_expected_area_vasya_l260_260727

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l260_260727


namespace f_eq_four_or_seven_l260_260534

noncomputable def f (a b : ℕ) : ℚ := (a^2 + a * b + b^2) / (a * b - 1)

theorem f_eq_four_or_seven (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : a * b ≠ 1) : 
  f a b = 4 ∨ f a b = 7 := 
sorry

end f_eq_four_or_seven_l260_260534


namespace probability_point_between_C_and_E_l260_260223

noncomputable def length_between_points (total_length : ℝ) (ratio : ℝ) : ℝ :=
ratio * total_length

theorem probability_point_between_C_and_E
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : C < E)
  (h3 : B - A = 4 * (D - A))
  (h4 : B - A = 8 * (B - C))
  (h5 : B - E = 2 * (E - C)) :
  (E - C) / (B - A) = 1 / 24 :=
by 
  sorry

end probability_point_between_C_and_E_l260_260223


namespace final_price_is_correct_l260_260624

-- Define the initial price of the book
def initial_price : ℝ := 400

-- Define the price transformations
def price_after_first_decrease (p : ℝ) : ℝ := p * (1 - 12.5 / 100)
def price_after_first_increase (p : ℝ) : ℝ := p * (1 + 30 / 100)
def price_after_second_decrease (p : ℝ) : ℝ := p * (1 - 20 / 100)
def price_after_second_increase (p : ℝ) : ℝ := p * (1 + 50 / 100)

-- Compute the final price after all transformations
def final_price : ℝ :=
  let p1 := price_after_first_decrease initial_price
  let p2 := price_after_first_increase p1
  let p3 := price_after_second_decrease p2
  price_after_second_increase p3

-- Theorem that proves the final price is $546
theorem final_price_is_correct : final_price = 546 := by
  sorry

end final_price_is_correct_l260_260624


namespace prove_problem_statement_l260_260903

noncomputable def problem_statement (ω : ℂ) (b : ℕ → ℝ) (n : ℕ) : Prop :=
  (ω ^ 3 = 1) ∧ (ω ≠ 1) ∧
  (∑ i in Finset.range n, (1 / (b i + ω)) = (3 + 4 * Complex.I)) →
  (∑ i in Finset.range n, ((3 * (b i : ℂ) - 2) / ((b i : ℂ) ^ 2 - (b i : ℂ) + 1)) = 6)

theorem prove_problem_statement {ω : ℂ} {b : ℕ → ℝ} {n : ℕ} : problem_statement ω b n :=
begin
  sorry
end

end prove_problem_statement_l260_260903


namespace longest_side_of_rectangular_solid_l260_260619

theorem longest_side_of_rectangular_solid 
  (x y z : ℝ) 
  (h1 : x * y = 20) 
  (h2 : y * z = 15) 
  (h3 : x * z = 12) 
  (h4 : x * y * z = 60) : 
  max (max x y) z = 10 := 
by sorry

end longest_side_of_rectangular_solid_l260_260619


namespace last_letter_of_95th_word_in_permutations_of_GRAPE_l260_260604

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def permutations_of_grape : List String := List.permutations "GRAPE".toList |>.map String.ofList

def lexicographic_order (l : List String) : List String := l.mergeSort (· < ·)

theorem last_letter_of_95th_word_in_permutations_of_GRAPE :
  (lexicographic_order permutations_of_grape).get! 94 |>.getLast! 'x' = 'E' :=
sorry

end last_letter_of_95th_word_in_permutations_of_GRAPE_l260_260604


namespace total_payment_l260_260006

namespace DiscountProblem

def item_price_with_discount (x : ℝ) : ℝ :=
  if x <= 600 then 
    x
  else if x <= 900 then 
    x * 0.8 
  else 
    900 * 0.8 + (x - 900) * 0.6

theorem total_payment (x y : ℝ) (h1 : x = 560) (h2 : y = 640) :
  item_price_with_discount x + item_price_with_discount y = 996 ∨ item_price_with_discount x + item_price_with_discount y = 1080 :=
by
  sorry

end DiscountProblem

end total_payment_l260_260006


namespace find_m_plus_n_l260_260432

noncomputable def math_problem (m n : ℕ) (A B C : Set ℕ) :=
  A = {1, n} ∧ B = {2, 4, m} ∧ C = {c | ∃ x ∈ A, ∃ y ∈ B, c = x * y} ∧
  C.card = 6 ∧ ∑ c in C, c = 42

theorem find_m_plus_n (m n : ℕ) (A B C : Set ℕ) (h : math_problem m n A B C) :
  m + n = 6 :=
sorry

end find_m_plus_n_l260_260432


namespace honor_students_count_l260_260986

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l260_260986


namespace total_number_of_valid_subsets_l260_260592

open Set Finset Nat

noncomputable def valid_subsets : Finset (Finset ℕ) :=
  (powerset (range 11)).filter 
  (λ s, s.card = 5 ∧ ∀ a b ∈ s, a ≠ b → a + b ≠ 11)

theorem total_number_of_valid_subsets :
  valid_subsets.card = 32 :=
sorry

end total_number_of_valid_subsets_l260_260592


namespace grasshopper_jump_avoid_M_l260_260199

-- Definitions and theorems related to the problem
structure GrasshopperProblem where
  n : ℕ
  a : Fin (n+2) → ℕ
  h_distinct : (Function.Injective a.toFun) -- ensuring all a_i are distinct
  s : ℕ := Finset.univ.sum a.toFun -- s = sum of all a_i
  M : Fin n → ℕ
  h_M_range : ∀ i, 0 < M i ∧ M i < s

theorem grasshopper_jump_avoid_M (p : GrasshopperProblem) : 
  ∃ (perm : Fin (p.n + 2) → Fin (p.n + 2)), 
    (∀ i, ((Finset.univ.sum (λ j, p.a (perm j)) 
            : ℕ)) ≠ p.M i) := 
by
  sorry

end grasshopper_jump_avoid_M_l260_260199


namespace incenter_squared_distance_sum_l260_260195

variable {R : Type*} [OrderedRing R]

structure Point :=
(x : R)
(y : R)

structure Triangle :=
(A B C : Point)
(a b c : R)

-- Define the squared distance between two points
def squared_distance (P Q : Point) : R :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- The main theorem statement
theorem incenter_squared_distance_sum {T : Triangle} (P Q : Point)
  (hQ : Q = Point.mk 
            ((T.a * T.A.x + T.b * T.B.x + T.c * T.C.x) / (T.a + T.b + T.c))
            ((T.a * T.A.y + T.b * T.B.y + T.c * T.C.y) / (T.a + T.b + T.c))) :
  T.a * squared_distance P T.A + T.b * squared_distance P T.B + T.c * squared_distance P T.C =
  T.a * squared_distance Q T.A + T.b * squared_distance Q T.B + T.c * squared_distance Q T.C +
  (T.a + T.b + T.c) * squared_distance Q P := by
  sorry

end incenter_squared_distance_sum_l260_260195


namespace volume_and_surface_area_of_revolution_l260_260353

noncomputable def volume_of_solid_of_revolution (a : ℝ) (phi : ℝ) : ℝ :=
  π * a ^ 3 * Real.sqrt 2 * Real.sin (phi + Real.pi / 4)

noncomputable def surface_area_of_solid_of_revolution (a : ℝ) (phi : ℝ) : ℝ :=
  4 * π * a ^ 2 * Real.sqrt 2 * Real.sin (phi + Real.pi / 4)

theorem volume_and_surface_area_of_revolution (a : ℝ) (phi : ℝ) (h₀ : 0 ≤ phi) (h₁ : phi ≤ Real.pi / 4) :
  volume_of_solid_of_revolution a phi = π * a ^ 3 * Real.sqrt 2 * Real.sin (phi + Real.pi / 4) ∧
  surface_area_of_solid_of_revolution a phi = 4 * π * a ^ 2 * Real.sqrt 2 * Real.sin (phi + Real.pi / 4) :=
by
  sorry

end volume_and_surface_area_of_revolution_l260_260353


namespace m_range_positive_solution_l260_260143

theorem m_range_positive_solution (m : ℝ) : (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) := by
  sorry

end m_range_positive_solution_l260_260143


namespace area_of_triangle_ABC_l260_260889

theorem area_of_triangle_ABC 
  {A B C M N P : Type*} 
  [IsTriangle A B C] 
  [∈ M (Segment A C)] 
  [∈ N (Segment B C)] 
  [∈ P (Segment M N)] 
  (h1 : ∃ λ : ℝ, (|AM| / |MC| = λ) ∧ (|CN| / |NB| = λ) ∧ (|MP| / |PN| = λ))
  (T Q : ℝ) 
  (area_AMP : Area A M P T)
  (area_BNP : Area B N P Q) :
  Area A B C ((T^(1/3) + Q^(1/3))^3) := 
by 
  sorry

end area_of_triangle_ABC_l260_260889


namespace tan_alpha_eq_two_imp_inv_sin_double_angle_l260_260430

theorem tan_alpha_eq_two_imp_inv_sin_double_angle (α : ℝ) (h : Real.tan α = 2) : 
  (1 / Real.sin (2 * α)) = 5 / 4 :=
by
  sorry

end tan_alpha_eq_two_imp_inv_sin_double_angle_l260_260430


namespace simplification_of_expression_l260_260487

variable {a b : ℚ}

theorem simplification_of_expression (h1a : a ≠ 0) (h1b : b ≠ 0) (h2 : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ( (3 * a)⁻¹ - (b / 3)⁻¹ ) = -(a * b)⁻¹ := 
sorry

end simplification_of_expression_l260_260487


namespace trapezoid_ratio_l260_260351

theorem trapezoid_ratio
  (ABCD : Type) [geometric_figure ABCD]
  (M N : point ABCD)
  (A B C D : point ABCD)
  (h1 : on_segment A B M ∧ segment_length A M = 2 * segment_length M B)
  (h2 : on_segment C D N ∧ divides_segment_in_three_times_area M N ABCD)
  (h3 : segment_length B C = segment_length A D / 2) :
  segment_length C N / segment_length D N = 3 / 29 :=
by
  sorry

end trapezoid_ratio_l260_260351


namespace interval_of_segmentation_l260_260639

-- Define the population size and sample size as constants.
def population_size : ℕ := 2000
def sample_size : ℕ := 40

-- State the theorem for the interval of segmentation.
theorem interval_of_segmentation :
  population_size / sample_size = 50 :=
sorry

end interval_of_segmentation_l260_260639


namespace as_share_of_total_profit_l260_260507

-- Define variables and constants
def capital : ℝ := 1    -- Total capital
def total_profit : ℝ := 2300
def A_invest_ratio : ℝ := 1/6
def A_time_ratio : ℝ := 1/6
def B_invest_ratio : ℝ := 1/3
def B_time_ratio : ℝ := 1/3
def remaining_invest_ratio : ℝ := 1 - (A_invest_ratio + B_invest_ratio)
def total_time : ℝ := 1 -- Consider C for the whole time

-- Define the capital-time investments for A, B, and C
def A_capital_time := (A_invest_ratio * capital) * A_time_ratio
def B_capital_time := (B_invest_ratio * capital) * B_time_ratio
def C_capital_time := (remaining_invest_ratio * capital) * total_time

-- Sum the total capital-time investments
def total_capital_time := A_capital_time + B_capital_time + C_capital_time

-- Calculate A's share of the total profit
def A_share := (A_capital_time / total_capital_time) * total_profit

-- The formal problem statement to prove
theorem as_share_of_total_profit : A_share = 100 :=
by
  sorry

end as_share_of_total_profit_l260_260507


namespace minimized_parabola_area_l260_260061

theorem minimized_parabola_area :
  ∃ (m n : ℝ), 4 + 2 * m + n = -1 ∧ 
              ∃ (a b : ℝ), a * b = n ∧ 
                           a + b = -m ∧ 
                           ∀ P : ℝ × ℝ, P = (-m / 2, - (m^2 - 2 * m - 5) / 4) → 
                                        y = x^2 + m * x + n → 
                                        y = x^2 - 4 * x + 3 := 
begin
  sorry
end

end minimized_parabola_area_l260_260061


namespace sum_of_factorials_of_odd_integers_less_than_500_l260_260662

theorem sum_of_factorials_of_odd_integers_less_than_500 :
  (∑ k in Finset.range 250, fact (2*k + 1)) = (∑ k in (Finset.range 500).filter (λ n, n % 2 = 1), fact n) :=
by
  sorry

end sum_of_factorials_of_odd_integers_less_than_500_l260_260662


namespace roots_of_quadratic_l260_260978

theorem roots_of_quadratic :
  ∀ x : ℝ, (x - 3) ^ 2 = 4 → x = 5 ∨ x = 1 :=
begin
  intros x hx,
  have h_pos : x - 3 = 2 ∨ x - 3 = -2,
  { rw [eq_comm, pow_two] at hx,
    rwa sqr_eq_iff_abs_eq at hx, },
  cases h_pos with h1 h2,
  { left,
    linarith, },
  { right,
    linarith, },
end

end roots_of_quadratic_l260_260978


namespace projection_of_vector_l260_260024

open Real EuclideanSpace

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b

theorem projection_of_vector : 
  vector_projection (6, -3) (3, 0) = (6, 0) := 
by 
  sorry

end projection_of_vector_l260_260024


namespace sin_double_angle_l260_260429

noncomputable def sin_double_angle_identity (sin_alpha : ℝ) (h₀ : sin_alpha = -4/5) (α : ℝ) (h₁ : α ∈ set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) : ℝ :=
  by sorry

theorem sin_double_angle (sin_alpha α : ℝ)
  (h₀ : sin_alpha = -4/5)
  (h₁ : α ∈ set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) :
  sin (2 * α) = -24/25 :=
sorry

end sin_double_angle_l260_260429


namespace quadratic_common_root_l260_260081

noncomputable def common_root (k : ℝ) : ℝ :=
  if k = 6 then arbitrary else
  if (k - 6) ≠ 0 then 1 else arbitrary

theorem quadratic_common_root (k : ℝ) (x : ℝ)
  (h1 : x^2 - k * x - 7 = 0)
  (h2 : x^2 - 6 * x - (k + 1) = 0) :
  k = -6 ∧ x = 1 :=
begin
  by_cases hk : k = 6,
  { -- if k = 6, equations are identical
    sorry
  },
  { -- if k ≠ 6
    have hx : x = 1 := by sorry,
    have hk' : k = -6 := by sorry,
    exact ⟨hk', hx⟩,
  }
end

end quadratic_common_root_l260_260081


namespace problem_statement_l260_260258

-- Define the power function f and the property that it is odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_cond : f 3 < f 2)

-- The statement we need to prove
theorem problem_statement : f (-3) > f (-2) := by
  sorry

end problem_statement_l260_260258


namespace greatest_integer_less_than_pi_minus_five_l260_260423

theorem greatest_integer_less_than_pi_minus_five : (nat.floor (Real.pi - 5) = 2) :=
by
  sorry

end greatest_integer_less_than_pi_minus_five_l260_260423


namespace carnations_in_last_three_bouquets_l260_260643

/--
Trevor buys six bouquets of carnations.
In the first bouquet, there are 9.5 carnations.
In the second bouquet, there are 14.25 carnations.
In the third bouquet, there are 18.75 carnations.
The average number of carnations in all six bouquets is 16.
Prove that the total number of carnations in the fourth, fifth, and sixth bouquets combined is 53.5.
-/
theorem carnations_in_last_three_bouquets:
  let bouquet1 := 9.5
  let bouquet2 := 14.25
  let bouquet3 := 18.75
  let total_bouquets := 6
  let average_per_bouquet := 16
  let total_carnations := average_per_bouquet * total_bouquets
  let remaining_carnations := total_carnations - (bouquet1 + bouquet2 + bouquet3)
  remaining_carnations = 53.5 :=
by
  sorry

end carnations_in_last_three_bouquets_l260_260643


namespace slope_of_line_l_l260_260812

-- Definitions of the given conditions
def lineA_intersects : Prop := ∃ x_A : ℝ, A = (x_A, 2)
def lineB_intersects : Prop := ∃ y_B : ℝ, B = (m, n) ∧ m - n = 1
def midpoint_condition : Prop := ∀ A B : (ℝ × ℝ), (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1

-- Using the properties to define the points A and B
def points_A_and_B (A B : ℝ × ℝ) : Prop :=
  (∃ x_A, A = (x_A, 2)) ∧
  (∃ m n, B = (m, n) ∧ m - n = 1) ∧
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1)

-- The main theorem to prove the slope of line l
theorem slope_of_line_l (A B : ℝ × ℝ) (h : points_A_and_B A B) : 
  ∃ k : ℝ, k = 3 / 5 :=
by
  sorry -- No proof needed

end slope_of_line_l_l260_260812


namespace lucas_seq_mod_50_l260_260831

def lucas_seq : ℕ → ℕ
| 0       => 2
| 1       => 5
| (n + 2) => lucas_seq n + lucas_seq (n + 1)

theorem lucas_seq_mod_50 : lucas_seq 49 % 5 = 0 := 
by
  sorry

end lucas_seq_mod_50_l260_260831


namespace prudence_sleep_in_4_weeks_l260_260040

theorem prudence_sleep_in_4_weeks :
  let hours_per_night_from_sun_to_thu := 6
      nights_from_sun_to_thu := 5
      hours_per_night_fri_and_sat := 9
      nights_fri_and_sat := 2
      nap_hours_per_day_on_sat_and_sun := 1
      nap_days_on_sat_and_sun := 2
      weeks := 4
  in
  (nights_from_sun_to_thu * hours_per_night_from_sun_to_thu +
   nights_fri_and_sat * hours_per_night_fri_and_sat +
   nap_days_on_sat_and_sun * nap_hours_per_day_on_sat_and_sun) * weeks = 200 :=
by
  sorry

end prudence_sleep_in_4_weeks_l260_260040


namespace degree_measure_angle_Q_l260_260595

theorem degree_measure_angle_Q (ABCDEFGHIJ : Type*) [decagon : regular_decagon ABCDEFGHIJ]
                                (A B C D E F G H I J Q : Point)
                                (HAF_extends : extends_to A F Q)
                                (HCD_extends : extends_to C D Q) :
  measure_angle Q = 72 :=
sorry

end degree_measure_angle_Q_l260_260595


namespace convertibles_count_l260_260922

def storeA_total := 40
def storeA_regular := 0.33 * storeA_total
def storeA_trucks := 0.10 * storeA_total
def storeA_sedans := 0.15 * storeA_total
def storeA_sports := 0.25 * storeA_total
def storeA_suvs := 0.05 * storeA_total

def storeB_total := 50
def storeB_regular := 0.40 * storeB_total
def storeB_trucks := 0.05 * storeB_total
def storeB_sedans := 0.20 * storeB_total
def storeB_sports := 0.15 * storeB_total
def storeB_suvs := 0.10 * storeB_total

def storeC_total := 35
def storeC_regular := 0.29 * storeC_total
def storeC_trucks := 0.20 * storeC_total
def storeC_sedans := 0.11 * storeC_total
def storeC_sports := 0.17 * storeC_total
def storeC_suvs := 0.07 * storeC_total

def total_cars := 125

def total_non_convertibles := 
  storeA_regular + storeA_trucks + storeA_sedans + storeA_sports + storeA_suvs +
  storeB_regular + storeB_trucks + storeB_sedans + storeB_sports + storeB_suvs +
  storeC_regular + storeC_trucks + storeC_sedans + storeC_sports + storeC_suvs

def convertibles := total_cars - total_non_convertibles

theorem convertibles_count : convertibles = 15 := by
  sorry

end convertibles_count_l260_260922


namespace perfect_square_factors_count_l260_260127

theorem perfect_square_factors_count : 
  let P := (2^12) * (3^15) * (7^7)
  in ∃ (n : ℕ), (n = 224) ∧ (∀ (d : ℕ), d ∣ P → ∃ (k : ℕ), d = k^2 → d ≠ 0) :=
begin
  sorry
end

end perfect_square_factors_count_l260_260127


namespace parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l260_260813

variables {Point Line Plane : Type}
variables (α β : Plane) (ℓ m : Line) (point_on_line_ℓ : Point) (point_on_line_m : Point)

-- Definitions of conditions
def line_perpendicular_to_plane (ℓ : Line) (α : Plane) : Prop := sorry
def line_contained_in_plane (m : Line) (β : Plane) : Prop := sorry
def planes_parallel (α β : Plane) : Prop := sorry
def line_perpendicular_to_line (ℓ m : Line) : Prop := sorry

axiom h1 : line_perpendicular_to_plane ℓ α
axiom h2 : line_contained_in_plane m β

-- Statement of the proof problem
theorem parallel_planes_sufficient_not_necessary_for_perpendicular_lines : 
  (planes_parallel α β → line_perpendicular_to_line ℓ m) ∧ 
  ¬ (line_perpendicular_to_line ℓ m → planes_parallel α β) :=
  sorry

end parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l260_260813


namespace sum_f_ge_zero_l260_260181

noncomputable def f : ℤ → ℝ := sorry
axiom f_cond : ∀ x y z : ℤ, x + y + z = 0 → f x + f y + f z ≥ 0

theorem sum_f_ge_zero : 
  f (-2017) + f (-2016) + ... + f 2016 + f 2017 ≥ 0 :=
sorry

end sum_f_ge_zero_l260_260181


namespace honor_students_count_l260_260997

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260997


namespace solve_inequality_f2_minimum_f5_l260_260055

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := ∑ i in finset.range n, |x - (i + 1)|

theorem solve_inequality_f2 (x : ℝ) : f_n 2 x < x + 1 ↔ (2 / 3 < x ∧ x < 4) :=
begin
  sorry,
end

theorem minimum_f5 (x : ℝ) : ∃ (x_min : ℝ), x_min = 3 ∧ ∀ (x : ℝ), f_n 5 x ≥ f_n 5 x_min ∧ f_n 5 x_min = 6 :=
begin
  sorry,
end

end solve_inequality_f2_minimum_f5_l260_260055


namespace least_three_digit_multiple_of_8_l260_260301

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l260_260301


namespace dilation_result_l260_260611

variable (w : ℂ) (k : ℝ) (z0 : ℂ)
variable (h_w : w = 1 + 2 * complex.I)
variable (h_k : k = 2)
variable (h_z0 : z0 = 3 + complex.I)

theorem dilation_result : ∃ z : ℂ, z - w = k * (z0 - w) ∧ z = 5 := by
  existsi (5 : ℂ)
  rw [h_w, h_k, h_z0]
  simp
  sorry

end dilation_result_l260_260611


namespace sum_g_approximation_l260_260550

-- Define the function g(n) as the integer closest to the 5th root of n
noncomputable def g (n : ℕ) : ℕ :=
  Real.toNat (Real.cbrt n ^ (1 / 5 : ℝ))

-- Statement of the problem
theorem sum_g_approximation :
  ∑ k in Finset.range 5000, (1 / (g k : ℝ)) = 5222.5 := 
sorry

end sum_g_approximation_l260_260550


namespace probability_sum_21_l260_260333

-- Conditions: Define the faces of both dice
def die1_faces : Set ℕ := {n | 1 ≤ n ∧ n ≤ 19} ∪ {0}
def die2_faces : Set ℕ := {n | (1 ≤ n ∧ n ≤ 11) ∨ (13 ≤ n ∧ n ≤ 20)} ∪ {0}

-- Define what it means to have a fair die roll with the given faces
def fair_die_roll (faces : Set ℕ) : Set (ℕ × ℕ) := 
  {p | p.1 ∈ faces ∧ p.2 ∈ faces}

-- Question: Prove the probability of sum 21 is 1/25 given the conditions
theorem probability_sum_21 :
  (finset.filter (λ p : ℕ × ℕ, p.1 + p.2 = 21)
    (finset.product (finset.filter (λ x, x ∈ die1_faces) finset.univ)
                    (finset.filter (λ y, y ∈ die2_faces) finset.univ))).card.to_rat /
  (finset.product (finset.filter (λ x, x ∈ die1_faces) finset.univ)
                  (finset.filter (λ y, y ∈ die2_faces) finset.univ)).card.to_rat = 1 / 25 :=
sorry

end probability_sum_21_l260_260333


namespace inequality_solution_l260_260036

theorem inequality_solution (x : ℝ) (hx1 : x ≥ -1/2) (hx2 : x ≠ 0) :
  (4 * x^2 / (1 - Real.sqrt (1 + 2 * x))^2 < 2 * x + 9) ↔ 
  (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45/8) :=
by
  sorry

end inequality_solution_l260_260036


namespace vasya_expected_area_greater_l260_260747

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l260_260747


namespace digit_difference_base3_vs_base5_and_base8_l260_260761

def num_digits_in_base (n b : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log n / Nat.log b + 1

theorem digit_difference_base3_vs_base5_and_base8 (n : ℕ) (h : n = 1357) :
  num_digits_in_base n 3 - (num_digits_in_base n 5 + num_digits_in_base n 8) = -2 :=
by
  have h3 : num_digits_in_base 1357 3 = 7 := by sorry
  have h5 : num_digits_in_base 1357 5 = 5 := by sorry
  have h8 : num_digits_in_base 1357 8 = 4 := by sorry
  rw [h] at *
  simp only [h3, h5, h8]
  norm_num

end digit_difference_base3_vs_base5_and_base8_l260_260761


namespace f_inequality_l260_260815

variable {f : ℝ → ℝ}

-- Defining the conditions as hypotheses
hypothesis (h1 : ∀ x, f x = 1)
hypothesis (h2 : ∀ x, f' x < 1 / 2)

-- Define the inequality to be proved
theorem f_inequality (x : ℝ) (hx : x ∈ [-1, 1]) : f x < x^2 / 2 + 1 / 2 :=
sorry

end f_inequality_l260_260815


namespace parabola_line_non_intersect_l260_260536

def P (x : ℝ) : ℝ := x^2 + 3 * x + 1
def Q : ℝ × ℝ := (10, 50)

def line_through_Q_with_slope (m x : ℝ) : ℝ := m * (x - Q.1) + Q.2

theorem parabola_line_non_intersect (r s : ℝ) (h : ∀ m, (r < m ∧ m < s) ↔ (∀ x, 
  x^2 + (3 - m) * x + (10 * m - 49) ≠ 0)) : r + s = 46 := 
sorry

end parabola_line_non_intersect_l260_260536


namespace find_a_value_l260_260495

variable (a b : ℝ)

def z1 : ℂ := -1 + a * complex.I
def z2 : ℂ := b - complex.I

theorem find_a_value (h1 : (z1 a) + (z2 b) ∈ ℝ) (h2 : (z1 a) * (z2 b) ∈ ℝ) : a = 1 :=
by
  sorry

end find_a_value_l260_260495


namespace compound_difference_l260_260476

noncomputable def monthly_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let monthly_rate := annual_rate / 12
  let periods := 12 * years
  principal * (1 + monthly_rate) ^ periods

noncomputable def semi_annual_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let semi_annual_rate := annual_rate / 2
  let periods := 2 * years
  principal * (1 + semi_annual_rate) ^ periods

theorem compound_difference (principal : ℝ) (annual_rate : ℝ) (years : ℝ) :
  monthly_compound_amount principal annual_rate years - semi_annual_compound_amount principal annual_rate years = 23.36 :=
by
  let principal := 8000
  let annual_rate := 0.08
  let years := 3
  sorry

end compound_difference_l260_260476


namespace count_sets_of_non_negative_integers_l260_260757

theorem count_sets_of_non_negative_integers :
  let A := { x : ℕ // ∃ x1 x2 x3 x4, x = x1 + x2 + x3 + x4 ∧ x1 >= 0 ∧ x2 >= 0 ∧ x3 >= 0 ∧ x4 >= 0 } in
  ∃ x1 x2 x3 x4 : ℕ, x1 + x2 + x3 + x4 = 36 → 9139 :=
by
  sorry

end count_sets_of_non_negative_integers_l260_260757


namespace find_speed_of_stream_l260_260676

-- Definitions of the conditions:
def downstream_equation (b s : ℝ) : Prop := b + s = 60
def upstream_equation (b s : ℝ) : Prop := b - s = 30

-- Theorem stating the speed of the stream given the conditions:
theorem find_speed_of_stream (b s : ℝ) (h1 : downstream_equation b s) (h2 : upstream_equation b s) : s = 15 := 
sorry

end find_speed_of_stream_l260_260676


namespace find_solution_l260_260499

-- Define the setup for the problem
variables (k x y : ℝ)

-- Conditions from the problem
def cond1 : Prop := x - y = 9 * k
def cond2 : Prop := x + y = 5 * k
def cond3 : Prop := 2 * x + 3 * y = 8

-- Proof statement combining all conditions to show the values of k, x, and y that satisfy them
theorem find_solution :
  cond1 k x y →
  cond2 k x y →
  cond3 x y →
  k = 1 ∧ x = 7 ∧ y = -2 := by
  sorry

end find_solution_l260_260499


namespace solve_for_x_l260_260936

theorem solve_for_x (x : ℝ) (h : (9 + 1/x)^(1/3) = -2) : x = -1/17 :=
by
  sorry

end solve_for_x_l260_260936


namespace prudence_sleep_4_weeks_equals_200_l260_260038

-- Conditions
def sunday_to_thursday_sleep := 6 
def friday_saturday_sleep := 9 
def nap := 1 

-- Number of days in the mentioned periods per week
def sunday_to_thursday_days := 5
def friday_saturday_days := 2
def nap_days := 2

-- Calculate total sleep per week
def total_sleep_per_week : Nat :=
  (sunday_to_thursday_days * sunday_to_thursday_sleep) +
  (friday_saturday_days * friday_saturday_sleep) +
  (nap_days * nap)

-- Calculate total sleep in 4 weeks
def total_sleep_in_4_weeks : Nat :=
  4 * total_sleep_per_week

theorem prudence_sleep_4_weeks_equals_200 : total_sleep_in_4_weeks = 200 := by
  sorry

end prudence_sleep_4_weeks_equals_200_l260_260038


namespace shaded_area_calculation_l260_260366

-- Define the grid and the side length conditions
def grid_size : ℕ := 5 * 4
def side_length : ℕ := 1
def total_squares : ℕ := 5 * 4

-- Define the area of one small square
def area_of_square (side: ℕ) : ℕ := side * side

-- Define the shaded region in terms of number of small squares fully or partially occupied
def shaded_squares : ℕ := 11

-- By analyzing the grid based on given conditions, prove that the area of the shaded region is 11
theorem shaded_area_calculation : (shaded_squares * side_length * side_length) = 11 := sorry

end shaded_area_calculation_l260_260366


namespace sequence_count_21_l260_260480

-- Define the conditions and the problem
def valid_sequence (n : ℕ) : ℕ :=
  if n = 21 then 114 else sorry

theorem sequence_count_21 : valid_sequence 21 = 114 :=
  by sorry

end sequence_count_21_l260_260480


namespace sum_of_first_100_terms_l260_260787

theorem sum_of_first_100_terms :
  let a_n (n : ℕ) := 2 / (n * (n + 1)) in
  let S_100 := ∑ n in Finset.range 100, a_n (n + 1) in
  S_100 = 200 / 101 :=
begin
  sorry
end

end sum_of_first_100_terms_l260_260787


namespace vasya_has_greater_expected_area_l260_260736

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l260_260736


namespace geometric_sequence_a5_l260_260817

-- Definitions based on the conditions:
variable {a : ℕ → ℝ} -- the sequence {a_n}
variable (q : ℝ) -- the common ratio of the geometric sequence

-- The sequence is geometric and terms are given:
axiom seq_geom (n m : ℕ) : a n = a 0 * q ^ n
axiom a_3_is_neg4 : a 3 = -4
axiom a_7_is_neg16 : a 7 = -16

-- The specific theorem we are proving:
theorem geometric_sequence_a5 :
  a 5 = -8 :=
by {
  sorry
}

end geometric_sequence_a5_l260_260817


namespace transformation_matrix_correct_l260_260307

-- Definitions of the rotation and scaling matrices
def rotation_matrix_90ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

def scaling_matrix_3 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3, 0; 0, 3]

-- The target transformation matrix
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -3; 3, 0]

-- The theorem we need to prove
theorem transformation_matrix_correct :
  scaling_matrix_3 ⬝ rotation_matrix_90ccw = transformation_matrix :=
begin
  sorry
end

end transformation_matrix_correct_l260_260307


namespace honor_students_count_l260_260994

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l260_260994


namespace exists_subset_sum_condition_l260_260206

open Real

theorem exists_subset_sum_condition {n : ℕ} {r : Fin n → ℝ} :
  ∃ I : Finset (Fin n), 
    (∀ i : Fin (n - 2), ∃ s : Finset (Fin n), s ⊆ [i, i + 1, i + 2] ∧ (s ∩ I).card ≤ 2) ∧
    abs (I.sum (λ i, r i)) ≥ (1 / 6) * (Finset.univ.sum (λ i, abs (r i))) :=
sorry

end exists_subset_sum_condition_l260_260206


namespace part1_part2_l260_260451
noncomputable theory

-- Definition of the triangle and the given conditions
variables {a b c A B C : ℝ}

-- Given sides and angles relationship
axiom sides_and_angles_opposite : (a, b, c, A, B, C) = (a, b, c, A, B, C)

-- Conditions given in the problem
axiom condition_1 : 2 * Real.sin (7 * Real.pi / 6) * Real.sin ((Real.pi / 6) + C) + Real.cos C = -1 / 2
axiom condition_2 : c = Real.sqrt 13
axiom condition_3 : (1 / 2) * a * b * (Real.sqrt 3 / 2) = 3 * Real.sqrt 3

-- Proof goals
theorem part1 : C = Real.pi / 3 := sorry

theorem part2 (h1 : C = Real.pi / 3) (h2 : c = Real.sqrt 13) (h3 : (1 / 2) * a * b * (Real.sqrt 3 / 2) = 3 * Real.sqrt 3) :
  Real.sin A + Real.sin B = 7 * Real.sqrt 39 / 26 := sorry

end part1_part2_l260_260451


namespace cars_with_all_seats_occupied_l260_260274

/-- Given that there are 18 cars in a train and various conditions about free seats,
    prove that the number of cars with all seats occupied is 13. --/
theorem cars_with_all_seats_occupied :
  ∀ (x y : ℕ), 
   let n := 18 in
   (1/2) * x + (1/3) * y = 2 ∧
   x + y ≤ n → 
   n - x - y = 13 :=
by
  intros x y cond,
  let n := 18,
  sorry

end cars_with_all_seats_occupied_l260_260274


namespace surface_area_of_shape_l260_260332

-- Definitions from conditions
def circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 3
def line (k x y : ℝ) : Prop := k * x - y - 1 = 0

-- The proof problem statement
theorem surface_area_of_shape (k : ℝ) (h : ∃ x y : ℝ, circle x y ∧ line k x y) :
  ∃ S : ℝ, S = 12 * Real.pi :=
sorry

end surface_area_of_shape_l260_260332


namespace least_positive_three_digit_multiple_of_8_l260_260292

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l260_260292


namespace evaluate_expression_at_x_l260_260933

-- Declare the original expression and state the conditions.
def expression (x : ℝ) := (1 - (2 / (x - 1))) / ((x^2 - 6 * x + 9) / (x^2 - 1))

-- State the simplified form.
def simplified_expression (x : ℝ) := (x + 1) / (x - 3)

-- Prove that the expression evaluates correctly under the given conditions.
theorem evaluate_expression_at_x :
  (1 ≠ 2) ∧ (x ≠ -1) ∧ (x ≠ 1) ∧ (x ≠ 3) ∧ (x = 2) → expression x = -3 :=
by
  intros h_conditions
  simp only [expression, simplified_expression]
  sorry

end evaluate_expression_at_x_l260_260933


namespace problem_statement_l260_260138

variable {P : ℕ → Prop}

theorem problem_statement
  (h1 : ∀ k, P k → P (k + 1))
  (h2 : ¬P 4)
  (n : ℕ) (hn : 1 ≤ n → n ≤ 4 → n ∈ Set.Icc 1 4) :
  ¬P n :=
by
  sorry

end problem_statement_l260_260138


namespace arithmetic_sum_l260_260501

variables {a d : ℝ}

theorem arithmetic_sum (h : 15 * a + 105 * d = 90) : 2 * a + 14 * d = 12 :=
sorry

end arithmetic_sum_l260_260501


namespace problem_statement_l260_260793

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 / 2 * x^2 + m * x + 7 / 2
noncomputable def g' (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := f (x + 1) - g' x

theorem problem_statement (m : ℝ) (m_lt_zero : m < 0) :
  (∀ x, (l : ℝ → ℝ) derives from graphs of f and g and abscissa of tangent on f is 1 →
    ∃ y, l y = y - 1 ∧ m = -2 ) ∧ 
  (∃ max_h, max_h = 2) ∧
  (∀ x, c : ℝ, ln(x + 1) < x + c → c ≥ 0) := 
sorry

end problem_statement_l260_260793


namespace similar_triangles_ratios_eq_l260_260925

open Complex

theorem similar_triangles_ratios_eq {a b c a' b' c' : ℂ}
  (h : ∃ (k : ℂ) (θ : ℝ), k ≠ 0 ∧ (b - a = k * exp(θ * I) * (b' - a')) ∧ (c - a = k * exp(θ * I) * (c' - a'))) :
  (b - a) / (c - a) = (b' - a') / (c' - a') := by
  sorry

end similar_triangles_ratios_eq_l260_260925


namespace avg_rate_of_change_is_correct_l260_260914

def f (x : ℝ) : ℝ := x^2 - 1

theorem avg_rate_of_change_is_correct :
  let Δx := 1.1 - 1 in
  let Δy := f 1.1 - f 1 in
  (Δy / Δx) = 2.1 :=
by
  sorry

end avg_rate_of_change_is_correct_l260_260914


namespace probability_mod_3_of_N_power_12_eq_1_l260_260343

theorem probability_mod_3_of_N_power_12_eq_1 :
  (probability (λ N : ℕ, N ∈ set.Icc 1 1500 ∧ (N^12 % 3 = 1)) 
               (λ N : ℕ, N ∈ set.Icc 1 1500) = 2/3) :=
sorry

end probability_mod_3_of_N_power_12_eq_1_l260_260343


namespace median_is_3_l260_260259

def list_of_children : List ℕ := [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6]

theorem median_is_3 : median list_of_children = 3 := 
by 
  sorry

end median_is_3_l260_260259


namespace cos_equivalent_angle_l260_260773

theorem cos_equivalent_angle (n : ℝ) (h1 : -180 ≤ n ∧ n ≤ 180) : 
  (cos (n * real.pi / 180) = cos (430 * real.pi / 180)) → (n = 70 ∨ n = -70) :=
sorry

end cos_equivalent_angle_l260_260773


namespace distance_between_poles_l260_260352

-- Definitions for the problem's conditions
def length : ℕ := 60
def width : ℕ := 50
def number_of_poles : ℕ := 44

-- The statement we want to prove
theorem distance_between_poles :
  let perimeter := 2 * (length + width)
  let number_of_gaps := number_of_poles - 1
  let distance := (perimeter : ℚ) / (number_of_gaps : ℚ)
  distance ≈ 5.116 :=
by
  sorry

end distance_between_poles_l260_260352


namespace slope_of_tangent_line_at_one_l260_260265

def f (x : ℝ) := x * Real.exp x

theorem slope_of_tangent_line_at_one :
  Deriv f 1 = 2 * Real.exp 1 :=
sorry

end slope_of_tangent_line_at_one_l260_260265


namespace find_x_log_eq_l260_260408

theorem find_x_log_eq (x : ℝ) : log 64 (3 * x - 2) = -1/3 → x = 3/4 :=
by sorry

end find_x_log_eq_l260_260408


namespace savings_by_buying_together_l260_260716

-- Definitions
def cost_per_iPhone : ℝ := 600
def num_iPhones_individual : ℝ := 1  -- Each person buys individually
def num_iPhones_together : ℝ := 3  -- All three buy together
def discount_rate : ℝ := 0.05
def total_cost_without_discount (n : ℝ) : ℝ := n * cost_per_iPhone
def discount (total_cost : ℝ) : ℝ := total_cost * discount_rate
def total_cost_with_discount (total_cost : ℝ) : ℝ := total_cost - discount(total_cost)

-- Theorem: Calculate savings by buying together
theorem savings_by_buying_together :
  total_cost_without_discount num_iPhones_together -
  total_cost_with_discount (total_cost_without_discount num_iPhones_together) = 90 :=
by {
  sorry
}

end savings_by_buying_together_l260_260716


namespace buses_needed_40_buses_needed_30_l260_260671

-- Define the number of students
def number_of_students : ℕ := 186

-- Define the function to calculate minimum buses needed
def min_buses_needed (n : ℕ) : ℕ := (number_of_students + n - 1) / n

-- Theorem statements for the specific cases
theorem buses_needed_40 : min_buses_needed 40 = 5 := 
by 
  sorry

theorem buses_needed_30 : min_buses_needed 30 = 7 := 
by 
  sorry

end buses_needed_40_buses_needed_30_l260_260671


namespace constant_compositions_count_l260_260482

def is_constant {α β : Type*} (f : α → β) : Prop :=
∀ a b : α, f a = f b

def number_of_constant_compositions : ℕ :=
4^2010 - 2^2010

theorem constant_compositions_count :
  ∃ (fns : ℕ → (ℕ → ℕ) → Prop), 
  ∀ (f : ℕ), fns f ({0,1} → {0,1}) → is_constant (f 2010 2009) →
  number_of_constant_compositions = 4^2010 - 2^2010 :=
sorry

end constant_compositions_count_l260_260482


namespace cos_diff_simplify_l260_260232

theorem cos_diff_simplify (x : ℝ) (y : ℝ) (h1 : x = Real.cos (Real.pi / 10)) (h2 : y = Real.cos (3 * Real.pi / 10)) : 
  x - y = 4 * x * (1 - x^2) := 
sorry

end cos_diff_simplify_l260_260232


namespace diameter_of_circle_l260_260325

theorem diameter_of_circle (a c : ℝ) (h_ne : a ≠ c) 
  (tangent_at_A : tangent_point_circle A D) 
  (tangent_at_C : tangent_point_circle C E) 
  (diameter_AB : diameter A B) 
  (meet_at_P : meets_tangent_lines A D C E P) 
  : diameter = real.sqrt (a^2 + c^2) := 
sorry

end diameter_of_circle_l260_260325


namespace factorize_expression_polygon_sides_l260_260323

-- Problem 1: Factorize 2x^3 - 8x
theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Find the number of sides of a polygon with interior angle sum 1080 degrees
theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end factorize_expression_polygon_sides_l260_260323


namespace number_of_subsets_l260_260128

theorem number_of_subsets :
  { B : set ℕ | {1, 2, 3} ⊆ B ∧ B ⊆ {1, 2, 3, 4, 5} }.to_finset.card = 4 := 
by
  sorry

end number_of_subsets_l260_260128


namespace perpendicular_planes_parallel_l260_260473

-- Define the lines m and n, and planes alpha and beta
def Line := Unit
def Plane := Unit

-- Define perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

-- The main theorem statement: If m ⊥ α and m ⊥ β, then α ∥ β
theorem perpendicular_planes_parallel (m : Line) (α β : Plane)
  (h₁ : perpendicular m α) (h₂ : perpendicular m β) : parallel α β :=
sorry

end perpendicular_planes_parallel_l260_260473


namespace roots_of_equation_l260_260977

theorem roots_of_equation (x : ℝ) : (x - 3) ^ 2 = 4 ↔ (x = 5 ∨ x = 1) := by
  sorry

end roots_of_equation_l260_260977


namespace nine_pointed_star_sum_of_tips_angles_l260_260573

theorem nine_pointed_star_sum_of_tips_angles 
  (points : Finset ℝ) 
  (circumference : ℝ)
  (h1 : points.card = 9) 
  (h2 : ∀ (p1 p2 : ℝ), p1 ∈ points ∧ p2 ∈ points → dist p1 p2 = circumference / 9)
  (pattern : ∀ (p : ℝ), p ∈ points → ∃ (p' : ℝ), p' ∈ points ∧ dist p p' = 4 * (circumference / 9)) :
  (∑ p in points, angle_at_tip p) = 720 :=
sorry

end nine_pointed_star_sum_of_tips_angles_l260_260573


namespace count_pairs_ab_l260_260777

theorem count_pairs_ab : 
    (∃ ab : ℝ × ℝ, ∀ xy : ℤ × ℤ,
        let x := xy.1, y := xy.2 in
        (ab.1 * x + ab.2 * y = 1 ∧ (x - 1)^2 + (y - 1)^2 = 50)) → 36 :=
begin
  sorry
end

end count_pairs_ab_l260_260777


namespace chemistry_books_count_l260_260634

theorem chemistry_books_count
  (C : ℕ)
  (h1 : 10.choose 2 = 45) -- Here 45 is the combination 10C2
  (h2 : 45 * (C.choose 2) = 1260) : 
  C = 8 := 
sorry

end chemistry_books_count_l260_260634


namespace k_inverse_k_inv_is_inverse_l260_260543

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x - 4
def k (x : ℝ) : ℝ := f (g x)

def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem k_inverse {x : ℝ} : k_inv (k x) = x :=
by
  sorry

theorem k_inv_is_inverse {x y : ℝ} : k_inv (y) = x ↔ y = k(x) :=
by
  sorry

end k_inverse_k_inv_is_inverse_l260_260543


namespace perimeters_equal_l260_260911

open_locale euclidean_geometry

noncomputable theory
open affine

-- Definitions for the given problem:
variables (A B C T M N X Y : Point ℝ)
variables [nonempty (affine_triangle ℝ)]
variables [affine.equilateral (triangle.mk A B C)]

-- Given conditions
def T_on_AC : affine.line ℝ A C.ray T := sorry
def M_on_AB_arc : affine.circumcircle (triangle.mk A B C) M := sorry
def N_on_BC_arc : affine.circumcircle (triangle.mk A B C) N := sorry
def MT_parallel_BC : affine.parallel ℝ T M C B := sorry
def NT_parallel_AB : affine.parallel ℝ T N A B := sorry
def AN_MT_intersect_X : affine.intersect (affine.line ℝ A N) (affine.line ℝ M T) X := sorry
def CM_NT_intersect_Y : affine.intersect (affine.line ℝ C M) (affine.line ℝ N T) Y := sorry

-- Prove that the perimeters of the polygons AXYC and XMBNY are equal.
theorem perimeters_equal :
  (perimeter (polygon.mk (A::X::Y::C::[])))
    = (perimeter (polygon.mk (X::M::B::N::Y::[]))) :=
sorry

end perimeters_equal_l260_260911


namespace vasya_has_greater_expected_area_l260_260737

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l260_260737


namespace problem_solution_l260_260684

-- conditions given in the problem
def quadratic_sequence (a b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n+1) = c n ∧ a n + a (n+1) = n

-- general term formula for the sequence {a_n}
def a_n_general_term (a : ℕ → ℝ) : Prop :=
  ∀ n, (odd n → a n = (n + 1) / 2) ∧ (even n → a n = (n - 2) / 2)

-- general term formula for the sequence {b_n}
def b_n_general_term (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = n^2 - n

-- sum of the first n terms of {b_n}
def sum_b_n_first_n_terms (T : ℕ → ℝ) : Prop :=
  ∀ n, T n = (n * (n-1) * (n+1)) / 3

-- proving the given conditions imply the formulas' correctness
theorem problem_solution (a b : ℕ → ℝ) (c T : ℕ → ℝ) :
  quadratic_sequence a b c →
  (a 1 = 1) →
  a_n_general_term a →
  (∀ n, b n = c (2*n - 1) → b_n_general_term b) →
  (∀ n, T n = ∑ k in range n, b k → sum_b_n_first_n_terms T) :=
by intros; sorry

end problem_solution_l260_260684


namespace inverse_of_k_l260_260548

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l260_260548


namespace probability_odd_sum_highest_below_10_l260_260690

theorem probability_odd_sum_highest_below_10 : 
  let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  in ∃ favorable_outcomes total_outcomes, 
     favorable_outcomes / total_outcomes = 65 / 126 :=
sorry

end probability_odd_sum_highest_below_10_l260_260690


namespace count_five_digit_even_numbers_l260_260037

theorem count_five_digit_even_numbers : 
  let digits := [0, 1, 2, 3, 7]
  let A_4_4 := 24 -- permutations of 4 out of 4 
  let C_3_1A_3_3 := 18 -- choices and permutations for unit digit 2
  (∑ d in digits.filter (λ d => d % 2 = 0), d) = 24 + 18
  ∨ sorry :=
  42

end count_five_digit_even_numbers_l260_260037


namespace cone_sphere_intersection_l260_260321

variables (r m : ℝ)

theorem cone_sphere_intersection (h₁ : m > 0) (h₂ : r > 0) :
  ∃ x : ℝ, x = (m * r^2) / (2 * r^2 + m^2) :=
begin
  use (m * r^2) / (2 * r^2 + m^2),
  sorry
end

end cone_sphere_intersection_l260_260321


namespace fourth_vertex_exists_l260_260460

structure Point :=
  (x : ℚ)
  (y : ℚ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  let M_AC := Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2)
  let M_BD := Point.mk ((B.x + D.x) / 2) ((B.y + D.y) / 2)
  is_midpoint M_AC A C ∧ is_midpoint M_BD B D ∧ M_AC = M_BD

theorem fourth_vertex_exists (A B C : Point) (hA : A = ⟨-1, 0⟩) (hB : B = ⟨3, 0⟩) (hC : C = ⟨1, -5⟩) :
  ∃ D : Point, (D = ⟨1, 5⟩ ∨ D = ⟨-3, -5⟩) ∧ is_parallelogram A B C D :=
by
  sorry

end fourth_vertex_exists_l260_260460


namespace A_speed_ratio_B_speed_l260_260361

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end A_speed_ratio_B_speed_l260_260361


namespace unit_vectors_collinear_with_a_l260_260775

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def collinear_unit_vectors (a : ℝ × ℝ × ℝ) : set (ℝ × ℝ × ℝ) :=
  {e | e = (1 / magnitude a) • a ∨ e = -(1 / magnitude a) • a}

theorem unit_vectors_collinear_with_a :
  let a : ℝ × ℝ × ℝ := (1, -2, 2) in
  collinear_unit_vectors a =
    { (1 / 3, -2 / 3, 2 / 3), (-1 / 3, 2 / 3, -2 / 3) } :=
by
  sorry


end unit_vectors_collinear_with_a_l260_260775


namespace part1_part2_l260_260719

-- Define point and circle
structure Point where
  x : ℝ
  y : ℝ

def Circle (r : ℝ) : Prop := r > 0

-- Define Q and condition for QD length
def Q : Point := { x := -2, y := Real.sqrt 21 }

noncomputable def dist (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

def QD_length (C_r : ℝ) (O : Point) : Prop :=
  dist Q O = Real.sqrt (dist Q O^2 - 16)

-- Define the tangent line equation for the second part
def tangent_line (a b : ℝ) : Prop := a > 0 ∧ b > 0

noncomputable def OM_magnitude (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- The proof problem statements
theorem part1 (O : Point) (C_r : ℝ) (hC : Circle C_r) (hQD : QD_length C_r O) : C_r = 3 :=
  sorry

theorem part2 (a b : ℝ) (hline : tangent_line a b) : OM_magnitude a b ≥ 6 :=
  sorry

end part1_part2_l260_260719


namespace coprime_unique_residues_non_coprime_same_residue_l260_260687

-- Part (a)

theorem coprime_unique_residues (m k : ℕ) (h : m.gcd k = 1) : 
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∀ (i : Fin m) (j : Fin k), 
      ∀ (i' : Fin m) (j' : Fin k), 
        (i, j) ≠ (i', j') → (a i * b j) % (m * k) ≠ (a i' * b j') % (m * k) := 
sorry

-- Part (b)

theorem non_coprime_same_residue (m k : ℕ) (h : m.gcd k > 1) : 
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∃ (i : Fin m) (j : Fin k) (i' : Fin m) (j' : Fin k), 
      (i, j) ≠ (i', j') ∧ (a i * b j) % (m * k) = (a i' * b j') % (m * k) := 
sorry

end coprime_unique_residues_non_coprime_same_residue_l260_260687


namespace find_b_vector_l260_260048

-- Define input vectors a, b, and their sum.
def vec_a : ℝ × ℝ × ℝ := (1, -2, 1)
def vec_b : ℝ × ℝ × ℝ := (-2, 4, -2)
def vec_sum : ℝ × ℝ × ℝ := (-1, 2, -1)

-- The theorem statement to prove that b is calculated correctly.
theorem find_b_vector :
  vec_a + vec_b = vec_sum →
  vec_b = (-2, 4, -2) :=
by
  sorry

end find_b_vector_l260_260048


namespace simplify_expression_correct_l260_260935

def simplify_expression (x : ℝ) : Prop :=
  2 * x - 3 * (2 - x) + 4 * (2 + 3 * x) - 5 * (1 - 2 * x) = 27 * x - 3

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
by
  sorry

end simplify_expression_correct_l260_260935


namespace amount_beef_correct_l260_260572

variable (cost_pasta_per_kg cost_beef_per_kg cost_sauce_per_jar cost_quesadilla money_left : ℝ)
variable (amount_pasta amount_beef amount_sauce : ℕ)

-- Conditions
def cost_pasta := amount_pasta * cost_pasta_per_kg
def cost_sauce := amount_sauce * cost_sauce_per_jar
def cost_quesadilla_total := cost_quesadilla
def total_spent := cost_pasta + cost_sauce + cost_quesadilla_total
def money_for_beef := money_left - total_spent
def amount_beef := money_for_beef / cost_beef_per_kg

-- Theorem to prove
theorem amount_beef_correct :
  amount_pasta = 2 →
  cost_pasta_per_kg = 1.5 →
  amount_sauce = 2 →
  cost_sauce_per_jar = 2 →
  cost_quesadilla = 6 →
  money_left = 15 →
  cost_beef_per_kg = 8 →
  amount_beef = 0.25 :=
by
  intros
  simp [amount_beef, money_for_beef, total_spent, cost_pasta, cost_sauce, cost_quesadilla_total]
  sorry

end amount_beef_correct_l260_260572


namespace increasing_distance_paths_product_l260_260755

open Real

-- Define the distance function for the grid
def distance (p1 p2 : ℤ × ℤ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the increasing distance path property
def increasing_distance_path (points : List (ℤ × ℤ)) : Prop :=
  ∀ i j, i < j → distance (points[i]) (points[i + 1]) < distance (points[j]) (points[j + 1])

-- Given conditions: 5x5 grid points
def grid_points : List (ℤ × ℤ) :=
  List.product (List.range 5) (List.range 5)

-- Number of maximum points in an increasing distance path (m)
def m : ℕ := 16

-- Number of such paths (r)
def r : ℕ := 8

-- Prove that m * r = 128
theorem increasing_distance_paths_product : m * r = 128 :=
by
  -- Define your conditions and constraints
  have m_def : m = 16 := by rfl
  have r_def : r = 8 := by rfl
  rw [m_def, r_def]
  -- Simplify the product
  exact Nat.mul_comm 16 8 ▸ rfl
  sorry -- The actual combinatorial proof goes here

end increasing_distance_paths_product_l260_260755


namespace perimeter_of_rectangle_B_eq_140_l260_260254

theorem perimeter_of_rectangle_B_eq_140 (a : ℕ)
    (P_A : ℕ) (P_B : ℕ)
    (ha_side_length : 8 * a = 112)
    (h_rectangle_A : 2 * a + 2 * (3 * a) = P_A)
    (h_rectangle_B : 2 * a + 2 * (4 * a) = P_B)
    : P_B = 140 :=
begin
  have h_a := (by linarith : a = 14),
  sorry,
end

end perimeter_of_rectangle_B_eq_140_l260_260254


namespace initial_women_count_l260_260238

-- Let W be the amount of work one woman can do in one day
definition work_done_by_woman := W

-- Let C be the amount of work one child can do in one day
definition work_done_by_child := C

-- Condition 1: Some women can complete the work in 5 days. Assume x women.
definition initial_women := x

-- Condition 2: 10 children take 10 days to complete the same work.
definition total_work_by_children := 10 * 10 * C

-- Condition 3: 5 women and 10 children together take 5 days to complete the work.
definition total_work_by_mixed_group := 5 * 5 * W + 5 * 10 * C

theorem initial_women_count :
  (5 * initial_women * work_done_by_woman = total_work_by_children) →
  (total_work_by_mixed_group = 5 * initial_women * work_done_by_woman) →
  initial_women = 10 := sorry

end initial_women_count_l260_260238


namespace problem_statement_l260_260557

variables (a b : ℝ^3) (θ : ℝ)
def a_unit_vector : Prop := ‖a‖ = 1
def b_magnitude : Prop := ‖b‖ = real.sqrt 2
def a_minus_b_magnitude : Prop := ‖a - b‖ = 1
def a_boxplus_b : ℝ := ‖a * real.sin θ + b * real.cos θ‖

theorem problem_statement (h1 : a_unit_vector a)
                         (h2 : b_magnitude b)
                         (h3 : a_minus_b_magnitude a b) :
  a_boxplus_b a b θ = real.sqrt 10 / 2 :=
sorry

end problem_statement_l260_260557


namespace least_positive_three_digit_multiple_of_8_l260_260296

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l260_260296


namespace not_integer_division_l260_260134

def P : ℕ := 1
def Q : ℕ := 2

theorem not_integer_division : ¬ (∃ (n : ℤ), (P : ℤ) / (Q : ℤ) = n) := by
sorry

end not_integer_division_l260_260134


namespace odometer_correct_l260_260334

def faultyOdometerReading := "003006"

def interpretReading (s : String) : Nat :=
  let base9List := [3, 0, 0, 5]
  List.foldl (λ acc x, acc * 9 + x) 0 base9List

def milesTraveled : Nat := 2192

theorem odometer_correct :
  interpretReading faultyOdometerReading = milesTraveled :=
by
  rw [interpretReading, faultyOdometerReading]
  sorry

end odometer_correct_l260_260334


namespace trains_cross_time_l260_260287

-- Define conditions
def length_train1 : ℝ := 50  -- in meters
def length_train2 : ℝ := 120  -- in meters
def speed_train1 : ℝ := 60 * (5/18)  -- in meters per second (converted from 60 km/hr)
def speed_train2 : ℝ := 40 * (5/18)  -- in meters per second (converted from 40 km/hr)

-- Prove the time for them to cross each other
theorem trains_cross_time : 
  let relative_speed := speed_train1 + speed_train2 in
  let total_length := length_train1 + length_train2 in
  let crossing_time := total_length / relative_speed in
  crossing_time ≈ 6.12 := sorry

end trains_cross_time_l260_260287


namespace milk_production_group_B_l260_260941

theorem milk_production_group_B (a b c d e : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_pos_d : d > 0) (h_pos_e : e > 0) :
  ((1.2 * b * d * e) / (a * c)) = 1.2 * (b / (a * c)) * d * e := 
by
  sorry

end milk_production_group_B_l260_260941


namespace least_positive_three_digit_multiple_of_8_l260_260299

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l260_260299


namespace initial_money_l260_260648

theorem initial_money (M : ℝ) 
  (H_clothes : 0.2 * M)
  (H_grocery : 0.15 * M)
  (H_electronics : 0.10 * M)
  (H_dining : 0.05 * M)
  (H_left : M - (0.2 * M + 0.15 * M + 0.10 * M + 0.05 * M) = 15700) :
  M = 31400 :=
begin
  sorry
end

end initial_money_l260_260648


namespace marie_erasers_count_l260_260917

theorem marie_erasers_count (initial_erasers : ℕ) (lost_erasers : ℕ) 
  (h_initial : initial_erasers = 95) (h_lost : lost_erasers = 42) : 
  initial_erasers - lost_erasers = 53 := 
by 
  rw [h_initial, h_lost]
  exact rfl

end marie_erasers_count_l260_260917


namespace area_ratio_of_pentagons_l260_260025

open Real

-- Define the regular pentagon F1 and the inner pentagon F2
variable (F1 F2 : Type)
variable [regular_pentagon F1]
variable [inner_pentagon F2 F1]

-- Given conditions: F1 is a regular pentagon, F2 is formed by intersection of all diagonals of F1
def regular_pentagon (P : Type) : Prop := 
  ∃ (s : ℝ), ∀ (x y : P), x ≠ y → dist x y = s

def inner_pentagon (P Q : Type) : Prop := 
  ∀ (x y : P), x ≠ y → ∃ (a b : Q), x ∈ P ∧ y ∈ P

-- Question: Prove the ratio of the areas of F1 to F2
theorem area_ratio_of_pentagons :
  ∃ (r : ℝ), r = (cos 72 / cos 36) ^ 2 :=
sorry

end area_ratio_of_pentagons_l260_260025


namespace Dawns_hourly_income_l260_260759

theorem Dawns_hourly_income :
  let sketches_time := 1.5 * 12 in
  let painting_time := 2 * 12 in
  let finishing_time := 0.5 * 12 in
  let total_time := sketches_time + painting_time + finishing_time in
  let total_earnings := 3600 + 1200 + 300 in
  total_earnings / total_time = 106.25 :=
by
  let sketches_time := 1.5 * 12
  let painting_time := 2 * 12
  let finishing_time := 0.5 * 12
  let total_time := sketches_time + painting_time + finishing_time
  let total_earnings := 3600 + 1200 + 300
  have h1 : total_time = 48 := sorry
  have h2 : total_earnings = 5100 := sorry
  calc
    total_earnings / total_time = 5100 / 48 : by rw [h1, h2]
    ... = 106.25 : sorry

end Dawns_hourly_income_l260_260759


namespace smallest_prime_with_composite_reverse_l260_260421

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m : Nat, m > 1 ∧ m < n ∧ n % m = 0

def reverse_digits (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem smallest_prime_with_composite_reverse :
  ∃ (n : Nat), 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (n / 10 = 3) ∧ is_composite (reverse_digits n) ∧
  (∀ m : Nat, 10 ≤ m ∧ m < n ∧ (m / 10 = 3) ∧ is_prime m → ¬is_composite (reverse_digits m)) :=
by
  sorry

end smallest_prime_with_composite_reverse_l260_260421


namespace octagon_edge_length_from_pentagon_l260_260710

noncomputable def regular_pentagon_edge_length : ℝ := 16
def num_of_pentagon_edges : ℕ := 5
def num_of_octagon_edges : ℕ := 8

theorem octagon_edge_length_from_pentagon (total_length_thread : ℝ) :
  total_length_thread = num_of_pentagon_edges * regular_pentagon_edge_length →
  (total_length_thread / num_of_octagon_edges) = 10 :=
by
  intro h
  sorry

end octagon_edge_length_from_pentagon_l260_260710


namespace projection_of_2a_plus_b_l260_260809

variables {ℝ : Type*} [inner_product_space ℝ]
variables (a b : ℝ)
variables (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : inner a b = 0)

theorem projection_of_2a_plus_b (a b : ℝ) (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : inner a b = 0) :
  (inner (2 • a + b) (a + b)) / norm (a + b) = 3 * real.sqrt 2 / 2 :=
by sorry

end projection_of_2a_plus_b_l260_260809


namespace find_integer_n_l260_260016

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end find_integer_n_l260_260016


namespace number_of_divisors_64n4_l260_260031

theorem number_of_divisors_64n4 
  (n : ℕ) 
  (h1 : (factors (120 * n^3)).length = 120) 
  (h2 : 120.nat_factors.prod * (n^3).nat_factors.prod = (120 * n^3)) :
  (factors (64 * n^4)).length = 675 := 
sorry

end number_of_divisors_64n4_l260_260031


namespace fourth_function_form_l260_260802

variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)
variable (hf : Function.LeftInverse f_inv f)
variable (hf_inv : Function.RightInverse f_inv f)

theorem fourth_function_form :
  (∀ x, y = (-(f (-x - 1)) + 2) ↔ y = f_inv (x + 2) + 1 ↔ -(x + y) = 0) :=
  sorry

end fourth_function_form_l260_260802


namespace train_crosses_bridge_l260_260721

def L_bridge := 1500 -- length of the bridge in meters
def t_lamp := 20 -- time to cross the lamp post in seconds
def L_train := 600 -- length of the train in meters

theorem train_crosses_bridge : 
  let v := L_train / t_lamp in
  let L_total := L_train + L_bridge in
  let t_bridge := L_total / v in
  t_bridge = 70 := by
  sorry

end train_crosses_bridge_l260_260721


namespace sum_first_2016_terms_eq_zero_l260_260837

theorem sum_first_2016_terms_eq_zero
  (a : ℕ → ℤ)
  (h1 : a 1 = 2008)
  (h2 : a 2 = 2009)
  (h_seq : ∀ n, a (n + 1) = a n + a (n + 2)) :
  (∑ i in Finset.range 2016, a i) = 0 :=
sorry

end sum_first_2016_terms_eq_zero_l260_260837


namespace symmetric_point_origin_l260_260098

theorem symmetric_point_origin (A : ℝ × ℝ × ℝ) (hA : A = (-3, 1, 4)) :
    let B := (3, -1, -4) in B = (-A.1, -A.2, -A.3) :=
by
  have hB : (3, -1, -4) = (-(-3), -(1), -(4)) by sorry
  exact hB

end symmetric_point_origin_l260_260098


namespace g_not_even_g_not_odd_g_neither_even_nor_odd_l260_260890

def g (x : ℝ) : ℝ := 3^(x^2 + sin x - 3) - |x|

theorem g_not_even : ¬ (∀ (x : ℝ), g x = g (-x)) := 
by sorry

theorem g_not_odd : ¬ (∀ (x : ℝ), g (-x) = - g x) := 
by sorry

theorem g_neither_even_nor_odd : ¬ (∀ (x : ℝ), g x = g (-x)) ∧ ¬ (∀ (x : ℝ), g (-x) = - g x) := 
by sorry

end g_not_even_g_not_odd_g_neither_even_nor_odd_l260_260890


namespace integral_f_eq_neg_one_third_l260_260790

theorem integral_f_eq_neg_one_third (f : ℝ → ℝ) 
  (h : ∀ x, f(x) = x^2 + 2 * ∫ y in 0..1, f(y))
  : ∫ x in 0..1, f(x) = -1/3 :=
sorry

end integral_f_eq_neg_one_third_l260_260790


namespace complement_domain_l260_260458

noncomputable def M : Set ℝ := { x | x ≤ 2 }

theorem complement_domain :
  ∀ x : ℝ, x ∈ (Set.univ \ M) ↔ x ∈ Set.Ioo 2 ℝ :=
by
  intro x
  rw Set.mem_diff
  simp only [M, Set.mem_set_of_eq, Set.mem_univ, true_and, Set.Ioo, Set.mem_Ioo]
  split
  · intro h1
    cases h1
    tauto
  · intro h1
    tauto
  sorry

end complement_domain_l260_260458


namespace problem_statement_l260_260060

noncomputable def R (k : ℕ) : ℕ := 
  Nat.choose k 3 + Nat.choose k 2 + Nat.choose k 1 + Nat.choose k 0

def M (n : ℕ) : ℕ :=
  Nat.find (λ k, R k ≥ n)

def m (n : ℕ) : ℕ := n - 1

theorem problem_statement : M 200 * m 200 = 2189 := by
  sorry

end problem_statement_l260_260060


namespace find_ab_l260_260023

noncomputable def p (a : ℝ) : Polynomial ℝ := Polynomial.Coeff.monomial 3 1 + Polynomial.Coeff.monomial 2 a + Polynomial.Coeff.monomial 1 17 + Polynomial.Coeff.monomial 0 12
noncomputable def q (b : ℝ) : Polynomial ℝ := Polynomial.Coeff.monomial 3 1 + Polynomial.Coeff.monomial 2 b + Polynomial.Coeff.monomial 1 23 + Polynomial.Coeff.monomial 0 15

theorem find_ab : 
  ∃ (a b : ℝ), ((p a).roots.filter (λ r, r ≠ 0)).card = 3 ∧ 
              ((q b).roots.filter (λ r, r ≠ 0)).card = 3 ∧ 
              ((p a).roots.filter (λ r, r ≠ 0)).filter (λ r, (q b).isRoot r) ≥ 2 ∧ 
              a = -10 ∧ b = -11 :=
  sorry

end find_ab_l260_260023


namespace inverse_of_composed_function_l260_260542

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l260_260542


namespace min_value_condition_l260_260848

noncomputable def min_value (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + 3 * y = 5 * x * y) : ℝ :=
  3 * x + 4 * y

theorem min_value_condition (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + 3 * y = 5 * x * y) :
    min_value x y h = 5 :=
begin
  sorry
end

end min_value_condition_l260_260848


namespace inequality_proof_l260_260792

noncomputable def x : ℝ := Real.exp (-1/2)
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 3

theorem inequality_proof : z > x ∧ x > y := by
  -- Conditions defined as follows:
  -- x = exp(-1/2)
  -- y = log(2) / log(5)
  -- z = log(3)
  -- To be proved:
  -- z > x > y
  sorry

end inequality_proof_l260_260792


namespace axis_symmetry_101_gon_l260_260314

variable (vertices : Set ℝ × ℝ)
variable (axis : Set ℝ × ℝ)

theorem axis_symmetry_101_gon
  (h_convex : Convex vertices)
  (h_101gon : vertices.card = 101)
  (h_symmetric  : SymmetricAlong axis vertices) :
  ∃ v ∈ vertices, v ∈ axis :=
sorry

end axis_symmetry_101_gon_l260_260314


namespace count_g_100_eq_18_l260_260778

-- Define g_1
def g_1 (n : ℕ) : ℕ :=
  3 * (Nat.divisors n).card

-- Define the g_j sequence
def g : ℕ → ℕ → ℕ
| 1, n := g_1 n
| (j + 1), n := g 1 (g j n)

-- Define the main theorem statement
theorem count_g_100_eq_18 : 
  (Finset.univ.filter (λ n : Fin 101, g 100 n = 18)).card = 18 :=
sorry

end count_g_100_eq_18_l260_260778
