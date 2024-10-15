import Mathlib

namespace NUMINAMATH_GPT_find_k_parallel_lines_l413_41352

theorem find_k_parallel_lines (k : ℝ) : 
  (∀ x y, (k - 1) * x + y + 2 = 0 → 
            (8 * x + (k + 1) * y + k - 1 = 0 → False)) → 
  k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_parallel_lines_l413_41352


namespace NUMINAMATH_GPT_polygon_sides_l413_41363

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l413_41363


namespace NUMINAMATH_GPT_uncovered_area_l413_41331

def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side : ℕ := 4

theorem uncovered_area (height width side : ℕ) (h : height = shoebox_height) (w : width = shoebox_width) (s : side = block_side) :
  (width * height) - (side * side) = 8 :=
by
  rw [h, w, s]
  -- Area of shoebox bottom = width * height
  -- Area of square block = side * side
  -- Uncovered area = (width * height) - (side * side)
  -- Therefore, (6 * 4) - (4 * 4) = 24 - 16 = 8
  sorry

end NUMINAMATH_GPT_uncovered_area_l413_41331


namespace NUMINAMATH_GPT_not_hyperbola_condition_l413_41374

theorem not_hyperbola_condition (m : ℝ) (x y : ℝ) (h1 : 1 ≤ m) (h2 : m ≤ 3) :
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m) :=
sorry

end NUMINAMATH_GPT_not_hyperbola_condition_l413_41374


namespace NUMINAMATH_GPT_find_n_for_perfect_square_l413_41335

theorem find_n_for_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℤ), n^2 + 5 * n + 13 = m^2 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_for_perfect_square_l413_41335


namespace NUMINAMATH_GPT_loom_weaving_rate_l413_41383

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) 
    (h1 : total_cloth = 25) (h2 : total_time = 195.3125) : 
    total_cloth / total_time = 0.128 :=
sorry

end NUMINAMATH_GPT_loom_weaving_rate_l413_41383


namespace NUMINAMATH_GPT_quadratic_unique_solution_pair_l413_41373

theorem quadratic_unique_solution_pair (a c : ℝ) (h₁ : a + c = 12) (h₂ : a < c) (h₃ : a * c = 9) :
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_unique_solution_pair_l413_41373


namespace NUMINAMATH_GPT_flour_already_added_l413_41367

theorem flour_already_added (sugar flour salt additional_flour : ℕ) 
  (h1 : sugar = 9) 
  (h2 : flour = 14) 
  (h3 : salt = 40)
  (h4 : additional_flour = sugar + 1) : 
  flour - additional_flour = 4 :=
by
  sorry

end NUMINAMATH_GPT_flour_already_added_l413_41367


namespace NUMINAMATH_GPT_find_k_l413_41347

theorem find_k (k : ℝ) : 
  let a := 6
  let b := 25
  let root := (-25 - Real.sqrt 369) / 12
  6 * root^2 + 25 * root + k = 0 → k = 32 / 3 :=
sorry

end NUMINAMATH_GPT_find_k_l413_41347


namespace NUMINAMATH_GPT_find_k_l413_41322

theorem find_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (h : k * x - y = 3) : k = 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l413_41322


namespace NUMINAMATH_GPT_probability_all_choose_paper_l413_41392

-- Given conditions
def probability_choice_is_paper := 1 / 3

-- The theorem to be proved
theorem probability_all_choose_paper :
  probability_choice_is_paper ^ 3 = 1 / 27 :=
sorry

end NUMINAMATH_GPT_probability_all_choose_paper_l413_41392


namespace NUMINAMATH_GPT_razors_blades_equation_l413_41389

/-- Given the number of razors sold x,
each razor sold brings a profit of 30 yuan,
each blade sold incurs a loss of 0.5 yuan,
the number of blades sold is twice the number of razors sold,
and the total profit from these two products is 5800 yuan,
prove that the linear equation is -0.5 * 2 * x + 30 * x = 5800 -/
theorem razors_blades_equation (x : ℝ) :
  -0.5 * 2 * x + 30 * x = 5800 := 
sorry

end NUMINAMATH_GPT_razors_blades_equation_l413_41389


namespace NUMINAMATH_GPT_thyme_pots_count_l413_41390

theorem thyme_pots_count
  (basil_pots : ℕ := 3)
  (rosemary_pots : ℕ := 9)
  (leaves_per_basil_pot : ℕ := 4)
  (leaves_per_rosemary_pot : ℕ := 18)
  (leaves_per_thyme_pot : ℕ := 30)
  (total_leaves : ℕ := 354)
  : (total_leaves - (basil_pots * leaves_per_basil_pot + rosemary_pots * leaves_per_rosemary_pot)) / leaves_per_thyme_pot = 6 :=
by
  sorry

end NUMINAMATH_GPT_thyme_pots_count_l413_41390


namespace NUMINAMATH_GPT_polynomial_simplification_l413_41378

theorem polynomial_simplification (y : ℤ) : 
  (2 * y - 1) * (4 * y ^ 10 + 2 * y ^ 9 + 4 * y ^ 8 + 2 * y ^ 7) = 8 * y ^ 11 + 6 * y ^ 9 - 2 * y ^ 7 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l413_41378


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l413_41354

-- Problem 1
theorem problem1_solution (x y : ℝ) : (2 * x - y = 3) ∧ (x + y = 3) ↔ (x = 2 ∧ y = 1) := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) : (x / 4 + y / 3 = 3) ∧ (3 * x - 2 * (y - 1) = 11) ↔ (x = 6 ∧ y = 9 / 2) := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l413_41354


namespace NUMINAMATH_GPT_max_value_m_l413_41398

noncomputable def exists_triangle_with_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_value_m (a b c : ℝ) (m : ℝ) (h1 : 0 < m) (h2 : abc ≤ 1/4) (h3 : 1/(a^2) + 1/(b^2) + 1/(c^2) < m) :
  m ≤ 9 ↔ exists_triangle_with_sides a b c :=
sorry

end NUMINAMATH_GPT_max_value_m_l413_41398


namespace NUMINAMATH_GPT_koschei_coins_l413_41310

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end NUMINAMATH_GPT_koschei_coins_l413_41310


namespace NUMINAMATH_GPT_min_value_of_M_l413_41315

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem min_value_of_M (M : ℝ) (hM : M = Real.sqrt 2) :
  ∀ (a b c : ℝ), a > M → b > M → c > M → a^2 + b^2 = c^2 → 
  (f a) + (f b) > f c ∧ (f a) + (f c) > f b ∧ (f b) + (f c) > f a :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_M_l413_41315


namespace NUMINAMATH_GPT_intersection_of_sets_l413_41329

noncomputable def A : Set ℝ := { x | x^2 - 1 > 0 }
noncomputable def B : Set ℝ := { x | Real.log x / Real.log 2 > 0 }

theorem intersection_of_sets :
  A ∩ B = { x | x > 1 } :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_sets_l413_41329


namespace NUMINAMATH_GPT_runners_speed_ratio_l413_41323

/-- Two runners, 20 miles apart, start at the same time, aiming to meet. 
    If they run in the same direction, they meet in 5 hours. 
    If they run towards each other, they meet in 1 hour.
    Prove that the ratio of the speed of the faster runner to the slower runner is 3/2. -/
theorem runners_speed_ratio (v1 v2 : ℝ) (h1 : v1 > v2)
  (h2 : 20 = 5 * (v1 - v2)) 
  (h3 : 20 = (v1 + v2)) : 
  v1 / v2 = 3 / 2 :=
sorry

end NUMINAMATH_GPT_runners_speed_ratio_l413_41323


namespace NUMINAMATH_GPT_boys_in_classroom_l413_41370

-- Definitions of the conditions
def total_children := 45
def girls_fraction := 1 / 3

-- The theorem we want to prove
theorem boys_in_classroom : (2 / 3) * total_children = 30 := by
  sorry

end NUMINAMATH_GPT_boys_in_classroom_l413_41370


namespace NUMINAMATH_GPT_symmetric_line_eq_l413_41384

-- Define the original line equation
def original_line (x: ℝ) : ℝ := -2 * x - 3

-- Define the symmetric line with respect to y-axis
def symmetric_line (x: ℝ) : ℝ := 2 * x - 3

-- The theorem stating the symmetric line with respect to the y-axis
theorem symmetric_line_eq : (∀ x: ℝ, original_line (-x) = symmetric_line x) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l413_41384


namespace NUMINAMATH_GPT_minimize_z_l413_41388

theorem minimize_z (x y : ℝ) (h1 : 2 * x - y ≥ 0) (h2 : y ≥ x) (h3 : y ≥ -x + 2) :
  ∃ (x y : ℝ), (z = 2 * x + y) ∧ z = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_z_l413_41388


namespace NUMINAMATH_GPT_circle_center_radius_proof_l413_41338

noncomputable def circle_center_radius (x y : ℝ) :=
  x^2 + y^2 - 4*x + 2*y + 2 = 0

theorem circle_center_radius_proof :
  ∀ x y : ℝ, circle_center_radius x y ↔ ((x - 2)^2 + (y + 1)^2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_proof_l413_41338


namespace NUMINAMATH_GPT_area_of_rectangle_l413_41336

def length : ℝ := 0.5
def width : ℝ := 0.24

theorem area_of_rectangle :
  length * width = 0.12 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l413_41336


namespace NUMINAMATH_GPT_trapezium_area_l413_41364

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_area_l413_41364


namespace NUMINAMATH_GPT_triangle_segments_l413_41320

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_segments (a : ℕ) (h : a > 0) :
  ¬ triangle_inequality 1 2 3 ∧
  ¬ triangle_inequality 4 5 10 ∧
  triangle_inequality 5 10 13 ∧
  ¬ triangle_inequality (2 * a) (3 * a) (6 * a) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_triangle_segments_l413_41320


namespace NUMINAMATH_GPT_cost_unit_pen_max_profit_and_quantity_l413_41317

noncomputable def cost_pen_A : ℝ := 5
noncomputable def cost_pen_B : ℝ := 10
noncomputable def profit_pen_A : ℝ := 2
noncomputable def profit_pen_B : ℝ := 3
noncomputable def spent_on_A : ℝ := 400
noncomputable def spent_on_B : ℝ := 800
noncomputable def total_pens : ℝ := 300

theorem cost_unit_pen : (spent_on_A / cost_pen_A) = (spent_on_B / (cost_pen_A + 5)) := by
  sorry

theorem max_profit_and_quantity
    (xa xb : ℝ)
    (h1 : xa ≥ 4 * xb)
    (h2 : xa + xb = total_pens)
    : ∃ (wa : ℝ), wa = 2 * xa + 3 * xb ∧ xa = 240 ∧ xb = 60 ∧ wa = 660 := by
  sorry

end NUMINAMATH_GPT_cost_unit_pen_max_profit_and_quantity_l413_41317


namespace NUMINAMATH_GPT_unit_digit_calc_l413_41316

theorem unit_digit_calc : (8 * 19 * 1981 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_unit_digit_calc_l413_41316


namespace NUMINAMATH_GPT_range_of_t_circle_largest_area_eq_point_P_inside_circle_l413_41360

open Real

-- Defining the given equation representing the trajectory of a point on a circle
def circle_eq (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16 * t^4 + 9 = 0

-- Problem 1: Proving the range of t
theorem range_of_t : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → -1/7 < t ∧ t < 1 :=
sorry

-- Problem 2: Proving the equation of the circle with the largest area
theorem circle_largest_area_eq : 
  ∃ t : ℝ, t = 3/7 ∧ (∀ x y : ℝ, circle_eq x y (3/7)) → 
  ∀ x y : ℝ, (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Problem 3: Proving the range of t for point P to be inside the circle
theorem point_P_inside_circle : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → 
  (0 < t ∧ t < 3/4) :=
sorry

end NUMINAMATH_GPT_range_of_t_circle_largest_area_eq_point_P_inside_circle_l413_41360


namespace NUMINAMATH_GPT_quadratic_value_at_two_l413_41381

open Real

-- Define the conditions
variables (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 + a * x + b

-- State the proof problem
theorem quadratic_value_at_two (h₀ : f a b (f a b 0) = 0) (h₁ : f a b (f a b 1) = 0) (h₂ : f a b 0 ≠ f a b 1) :
  f a b 2 = 2 := 
sorry

end NUMINAMATH_GPT_quadratic_value_at_two_l413_41381


namespace NUMINAMATH_GPT_sum_digits_10_pow_85_minus_85_l413_41345

-- Define the function that computes the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

-- Define the specific problem for n = 10^85 - 85
theorem sum_digits_10_pow_85_minus_85 : 
  sum_of_digits (10^85 - 85) = 753 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_10_pow_85_minus_85_l413_41345


namespace NUMINAMATH_GPT_domain_of_f_l413_41376

def domain_f := {x : ℝ | 2 * x - 3 > 0}

theorem domain_of_f : ∀ x : ℝ, x ∈ domain_f ↔ x > 3 / 2 := 
by
  intro x
  simp [domain_f]
  sorry

end NUMINAMATH_GPT_domain_of_f_l413_41376


namespace NUMINAMATH_GPT_volume_of_regular_tetrahedron_with_edge_length_1_l413_41350

-- We define the concepts needed: regular tetrahedron, edge length, and volume.
open Real

noncomputable def volume_of_regular_tetrahedron (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * a^2
  let height := sqrt (a^2 - (a * (sqrt 3 / 3))^2)
  (1 / 3) * base_area * height

-- The problem statement and our goal to prove:
theorem volume_of_regular_tetrahedron_with_edge_length_1 :
  volume_of_regular_tetrahedron 1 = sqrt 2 / 12 := sorry

end NUMINAMATH_GPT_volume_of_regular_tetrahedron_with_edge_length_1_l413_41350


namespace NUMINAMATH_GPT_number_smaller_than_neg3_exists_l413_41344

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end NUMINAMATH_GPT_number_smaller_than_neg3_exists_l413_41344


namespace NUMINAMATH_GPT_decodeMINT_l413_41359

def charToDigit (c : Char) : Option Nat :=
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'M' => some 5
  | 'I' => some 6
  | 'N' => some 7
  | 'D' => some 8
  | 'S' => some 9
  | _   => none

def decodeWord (word : String) : Option Nat :=
  let digitsOption := word.toList.map charToDigit
  if digitsOption.all Option.isSome then
    let digits := digitsOption.map Option.get!
    some (digits.foldl (λ acc d => 10 * acc + d) 0)
  else
    none

theorem decodeMINT : decodeWord "MINT" = some 5674 := by
  sorry

end NUMINAMATH_GPT_decodeMINT_l413_41359


namespace NUMINAMATH_GPT_mod_remainder_7_10_20_3_20_l413_41318

theorem mod_remainder_7_10_20_3_20 : (7 * 10^20 + 3^20) % 9 = 7 := sorry

end NUMINAMATH_GPT_mod_remainder_7_10_20_3_20_l413_41318


namespace NUMINAMATH_GPT_sequence_infinite_pos_neg_l413_41361

theorem sequence_infinite_pos_neg (a : ℕ → ℝ)
  (h : ∀ k : ℕ, a (k + 1) = (k * a k + 1) / (k - a k)) :
  ∃ (P N : ℕ → Prop), (∀ n, P n ↔ 0 < a n) ∧ (∀ n, N n ↔ a n < 0) ∧ 
  (∀ m, ∃ n, n > m ∧ P n) ∧ (∀ m, ∃ n, n > m ∧ N n) := 
sorry

end NUMINAMATH_GPT_sequence_infinite_pos_neg_l413_41361


namespace NUMINAMATH_GPT_sum_of_ages_l413_41358

-- Definitions based on conditions
def age_relation1 (a b c : ℕ) : Prop := a = 20 + b + c
def age_relation2 (a b c : ℕ) : Prop := a^2 = 2000 + (b + c)^2

-- The statement to be proven
theorem sum_of_ages (a b c : ℕ) (h1 : age_relation1 a b c) (h2 : age_relation2 a b c) : a + b + c = 80 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l413_41358


namespace NUMINAMATH_GPT_seating_arrangements_l413_41385

-- Number of ways to arrange a block of n items
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Groups
def dodgers : ℕ := 4
def marlins : ℕ := 3
def phillies : ℕ := 2

-- Total number of players
def total_players : ℕ := dodgers + marlins + phillies

-- Number of ways to arrange the blocks
def blocks_arrangements : ℕ := factorial 3

-- Internal arrangements within each block
def dodgers_arrangements : ℕ := factorial dodgers
def marlins_arrangements : ℕ := factorial marlins
def phillies_arrangements : ℕ := factorial phillies

-- Total number of ways to seat the players
def total_arrangements : ℕ :=
  blocks_arrangements * dodgers_arrangements * marlins_arrangements * phillies_arrangements

-- Prove that the total arrangements is 1728
theorem seating_arrangements : total_arrangements = 1728 := by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l413_41385


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l413_41324

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2*(m-3)*x + 16) = (x + a)^2) ↔ (m = 7 ∨ m = -1) := 
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l413_41324


namespace NUMINAMATH_GPT_eggs_in_box_l413_41319

-- Given conditions as definitions in Lean 4
def initial_eggs : ℕ := 7
def additional_whole_eggs : ℕ := 3

-- The proof statement
theorem eggs_in_box : initial_eggs + additional_whole_eggs = 10 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_eggs_in_box_l413_41319


namespace NUMINAMATH_GPT_value_of_m_l413_41307

theorem value_of_m (m x : ℝ) (h : x - 4 ≠ 0) (hx_pos : x > 0) 
  (eqn : m / (x - 4) - (1 - x) / (4 - x) = 0) : m = 3 := 
by
  sorry

end NUMINAMATH_GPT_value_of_m_l413_41307


namespace NUMINAMATH_GPT_sum_of_common_ratios_l413_41302

noncomputable def geometric_sequence (m x : ℝ) : ℝ × ℝ × ℝ := (m, m * x, m * x^2)

theorem sum_of_common_ratios
  (m x y : ℝ)
  (h1 : x ≠ y)
  (h2 : m ≠ 0)
  (h3 : ∃ c3 c2 d3 d2 : ℝ, geometric_sequence m x = (m, c2, c3) ∧ geometric_sequence m y = (m, d2, d3) ∧ c3 - d3 = 3 * (c2 - d2)) :
  x + y = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l413_41302


namespace NUMINAMATH_GPT_pyramid_value_l413_41330

theorem pyramid_value (a b c d e f : ℕ) (h_b : b = 6) (h_d : d = 20) (h_prod1 : d = b * (20 / b)) (h_prod2 : e = (20 / b) * c) (h_prod3 : f = c * (72 / c)) : a = b * c → a = 54 :=
by 
  -- Assuming the proof would assert the calculations done in the solution.
  sorry

end NUMINAMATH_GPT_pyramid_value_l413_41330


namespace NUMINAMATH_GPT_find_pairs_l413_41340

theorem find_pairs (p q : ℤ) (a b : ℤ) :
  (p^2 - 4 * q = a^2) ∧ (q^2 - 4 * p = b^2) ↔ 
    (p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l413_41340


namespace NUMINAMATH_GPT_problem_prove_divisibility_l413_41306

theorem problem_prove_divisibility (n : ℕ) : 11 ∣ (5^(2*n) + 3^(n+2) + 3^n) :=
sorry

end NUMINAMATH_GPT_problem_prove_divisibility_l413_41306


namespace NUMINAMATH_GPT_dividend_is_2160_l413_41321

theorem dividend_is_2160 (d q r : ℕ) (h₁ : d = 2016 + d) (h₂ : q = 15) (h₃ : r = 0) : d = 2160 :=
by
  sorry

end NUMINAMATH_GPT_dividend_is_2160_l413_41321


namespace NUMINAMATH_GPT_proof_problem_l413_41309

theorem proof_problem
  (a b c : ℂ)
  (h1 : ac / (a + b) + ba / (b + c) + cb / (c + a) = -4)
  (h2 : bc / (a + b) + ca / (b + c) + ab / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 7 := 
sorry

end NUMINAMATH_GPT_proof_problem_l413_41309


namespace NUMINAMATH_GPT_prove_s90_zero_l413_41369

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 0) + (n * (n - 1) * (a 1 - a 0)) / 2)

theorem prove_s90_zero (a : ℕ → ℕ) (h_arith : is_arithmetic_sequence a) (h : sum_of_first_n_terms a 30 = sum_of_first_n_terms a 60) :
  sum_of_first_n_terms a 90 = 0 :=
sorry

end NUMINAMATH_GPT_prove_s90_zero_l413_41369


namespace NUMINAMATH_GPT_smallest_number_increased_by_3_divisible_l413_41343

theorem smallest_number_increased_by_3_divisible (n : ℤ) 
    (h1 : (n + 3) % 18 = 0)
    (h2 : (n + 3) % 70 = 0)
    (h3 : (n + 3) % 25 = 0)
    (h4 : (n + 3) % 21 = 0) : 
    n = 3147 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_increased_by_3_divisible_l413_41343


namespace NUMINAMATH_GPT_price_of_cork_l413_41333

theorem price_of_cork (C : ℝ) 
  (h₁ : ∃ (bottle_with_cork bottle_without_cork : ℝ), bottle_with_cork = 2.10 ∧ bottle_without_cork = C + 2.00 ∧ bottle_with_cork = C + bottle_without_cork) :
  C = 0.05 :=
by
  obtain ⟨bottle_with_cork, bottle_without_cork, hwc, hwoc, ht⟩ := h₁
  sorry

end NUMINAMATH_GPT_price_of_cork_l413_41333


namespace NUMINAMATH_GPT_solution_set_of_inequality_l413_41337

theorem solution_set_of_inequality (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l413_41337


namespace NUMINAMATH_GPT_factorization_correct_l413_41371

theorem factorization_correct (x : ℝ) :
    x^2 - 3 * x - 4 = (x + 1) * (x - 4) :=
  sorry

end NUMINAMATH_GPT_factorization_correct_l413_41371


namespace NUMINAMATH_GPT_current_number_of_women_is_24_l413_41301

-- Define initial person counts based on the given ratio and an arbitrary factor x.
variables (x : ℕ)
def M_initial := 4 * x
def W_initial := 5 * x
def C_initial := 3 * x
def E_initial := 2 * x

-- Define the changes that happened to the room.
def men_after_entry := M_initial x + 2
def women_after_leaving := W_initial x - 3
def women_after_doubling := 2 * women_after_leaving x
def children_after_leaving := C_initial x - 5
def elderly_after_leaving := E_initial x - 3

-- Define the current counts after all changes.
def men_current := 14
def children_current := 7
def elderly_current := 6

-- Prove that the current number of women is 24.
theorem current_number_of_women_is_24 :
  men_after_entry x = men_current ∧
  children_after_leaving x = children_current ∧
  elderly_after_leaving x = elderly_current →
  women_after_doubling x = 24 :=
by
  sorry

end NUMINAMATH_GPT_current_number_of_women_is_24_l413_41301


namespace NUMINAMATH_GPT_dog_bones_l413_41304

theorem dog_bones (initial_bones found_bones : ℕ) (h₁ : initial_bones = 15) (h₂ : found_bones = 8) : initial_bones + found_bones = 23 := by
  sorry

end NUMINAMATH_GPT_dog_bones_l413_41304


namespace NUMINAMATH_GPT_complement_of_irreducible_proper_fraction_is_irreducible_l413_41339

theorem complement_of_irreducible_proper_fraction_is_irreducible 
  (a b : ℤ) (h0 : 0 < a) (h1 : a < b) (h2 : Int.gcd a b = 1) : Int.gcd (b - a) b = 1 :=
sorry

end NUMINAMATH_GPT_complement_of_irreducible_proper_fraction_is_irreducible_l413_41339


namespace NUMINAMATH_GPT_nearest_integer_pow_l413_41312

noncomputable def nearest_integer_to_power : ℤ := 
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_pow : nearest_integer_to_power = 7414 := 
  by
    unfold nearest_integer_to_power
    sorry -- Proof skipped

end NUMINAMATH_GPT_nearest_integer_pow_l413_41312


namespace NUMINAMATH_GPT_find_f_0_plus_f_neg_1_l413_41314

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - x^2 else
if x < 0 then -(2^(-x) - (-x)^2) else 0

theorem find_f_0_plus_f_neg_1 : f 0 + f (-1) = -1 := by
  sorry

end NUMINAMATH_GPT_find_f_0_plus_f_neg_1_l413_41314


namespace NUMINAMATH_GPT_onion_pieces_per_student_l413_41362

theorem onion_pieces_per_student (total_pizzas : ℕ) (slices_per_pizza : ℕ)
  (cheese_pieces_leftover : ℕ) (onion_pieces_leftover : ℕ) (students : ℕ) (cheese_per_student : ℕ)
  (h1 : total_pizzas = 6) (h2 : slices_per_pizza = 18) (h3 : cheese_pieces_leftover = 8) (h4 : onion_pieces_leftover = 4)
  (h5 : students = 32) (h6 : cheese_per_student = 2) :
  ((total_pizzas * slices_per_pizza) - cheese_pieces_leftover - onion_pieces_leftover - (students * cheese_per_student)) / students = 1 := 
by
  sorry

end NUMINAMATH_GPT_onion_pieces_per_student_l413_41362


namespace NUMINAMATH_GPT_pow_mod_remainder_l413_41379

theorem pow_mod_remainder :
  (2^2013 % 11) = 8 :=
sorry

end NUMINAMATH_GPT_pow_mod_remainder_l413_41379


namespace NUMINAMATH_GPT_length_of_tank_l413_41342

namespace TankProblem

def field_length : ℝ := 90
def field_breadth : ℝ := 50
def field_area : ℝ := field_length * field_breadth

def tank_breadth : ℝ := 20
def tank_depth : ℝ := 4

def earth_volume (L : ℝ) : ℝ := L * tank_breadth * tank_depth

def remaining_field_area (L : ℝ) : ℝ := field_area - L * tank_breadth

def height_increase : ℝ := 0.5

theorem length_of_tank (L : ℝ) :
  earth_volume L = remaining_field_area L * height_increase →
  L = 25 :=
by
  sorry

end TankProblem

end NUMINAMATH_GPT_length_of_tank_l413_41342


namespace NUMINAMATH_GPT_find_a_plus_b_l413_41357

open Function

theorem find_a_plus_b (a b : ℝ) (f g h : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x - b)
  (h_g : ∀ x, g x = -4 * x - 1)
  (h_h : ∀ x, h x = f (g x))
  (h_h_inv : ∀ y, h⁻¹ y = y + 9) :
  a + b = -9 := 
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l413_41357


namespace NUMINAMATH_GPT_all_cells_equal_l413_41372

-- Define the infinite grid
def Grid := ℕ → ℕ → ℕ

-- Define the condition on the grid values
def is_min_mean_grid (g : Grid) : Prop :=
  ∀ i j : ℕ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

-- Main theorem
theorem all_cells_equal (g : Grid) (h : is_min_mean_grid g) : ∃ a : ℕ, ∀ i j : ℕ, g i j = a := 
sorry

end NUMINAMATH_GPT_all_cells_equal_l413_41372


namespace NUMINAMATH_GPT_sin_sum_identity_l413_41393

theorem sin_sum_identity 
  (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) : 
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_sum_identity_l413_41393


namespace NUMINAMATH_GPT_total_words_story_l413_41382

def words_per_line : ℕ := 10
def lines_per_page : ℕ := 20
def pages_filled : ℚ := 1.5
def words_left : ℕ := 100

theorem total_words_story : 
    words_per_line * lines_per_page * pages_filled + words_left = 400 := 
by
sorry

end NUMINAMATH_GPT_total_words_story_l413_41382


namespace NUMINAMATH_GPT_somu_age_to_father_age_ratio_l413_41311

theorem somu_age_to_father_age_ratio
  (S : ℕ) (F : ℕ)
  (h1 : S = 10)
  (h2 : S - 5 = (1/5) * (F - 5)) :
  S / F = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_somu_age_to_father_age_ratio_l413_41311


namespace NUMINAMATH_GPT_ticket_queue_correct_l413_41334

-- Define the conditions
noncomputable def ticket_queue_count (m n : ℕ) (h : n ≥ m) : ℕ :=
  (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))

-- State the theorem
theorem ticket_queue_correct (m n : ℕ) (h : n ≥ m) :
  ticket_queue_count m n h = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_ticket_queue_correct_l413_41334


namespace NUMINAMATH_GPT_function_range_of_roots_l413_41375

theorem function_range_of_roots (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : a > 1 := 
sorry

end NUMINAMATH_GPT_function_range_of_roots_l413_41375


namespace NUMINAMATH_GPT_determine_sanity_l413_41399

-- Defining the conditions for sanity based on responses to a specific question

-- Define possible responses
inductive Response
| ball : Response
| yes : Response

-- Define sanity based on logical interpretation of an illogical question
def is_sane (response : Response) : Prop :=
  response = Response.ball

-- The theorem stating asking the specific question determines sanity
theorem determine_sanity (response : Response) : is_sane response ↔ response = Response.ball :=
by
  sorry

end NUMINAMATH_GPT_determine_sanity_l413_41399


namespace NUMINAMATH_GPT_regression_correlation_relation_l413_41326

variable (b r : ℝ)

theorem regression_correlation_relation (h : b = 0) : r = 0 := 
sorry

end NUMINAMATH_GPT_regression_correlation_relation_l413_41326


namespace NUMINAMATH_GPT_intersection_A_B_union_A_B_complement_intersection_A_B_l413_41353

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def A_inter_B : Set ℝ := { x | 2 ≤ x ∧ x < 6 }
def A_union_B : Set ℝ := { x | 1 < x ∧ x ≤ 8 }
def A_compl_inter_B : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = A_inter_B := by
  sorry

theorem union_A_B :
  A ∪ B = A_union_B := by
  sorry

theorem complement_intersection_A_B :
  (Aᶜ ∩ B) = A_compl_inter_B := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_union_A_B_complement_intersection_A_B_l413_41353


namespace NUMINAMATH_GPT_smallest_candies_value_l413_41394

def smallest_valid_n := ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 9 = 2 ∧ n % 7 = 5 ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 9 = 2 ∧ m % 7 = 5 → n ≤ m

theorem smallest_candies_value : ∃ n : ℕ, smallest_valid_n ∧ n = 101 := 
by {
  sorry  
}

end NUMINAMATH_GPT_smallest_candies_value_l413_41394


namespace NUMINAMATH_GPT_find_x_l413_41327

variable (x : ℝ)
variable (l : ℝ) (w : ℝ)

def length := 4 * x + 1
def width := x + 7

theorem find_x (h1 : l = length x) (h2 : w = width x) (h3 : l * w = 2 * (2 * l + 2 * w)) :
  x = (-9 + Real.sqrt 481) / 8 :=
by
  subst_vars
  sorry

end NUMINAMATH_GPT_find_x_l413_41327


namespace NUMINAMATH_GPT_division_multiplication_result_l413_41300

theorem division_multiplication_result :
  (7.5 / 6) * 12 = 15 := by
  sorry

end NUMINAMATH_GPT_division_multiplication_result_l413_41300


namespace NUMINAMATH_GPT_towel_length_decrease_l413_41341

theorem towel_length_decrease (L B : ℝ) (HL1: L > 0) (HB1: B > 0)
  (length_percent_decr : ℝ) (breadth_decr : B' = 0.8 * B) 
  (area_decr : (L' * B') = 0.64 * (L * B)) :
  (L' = 0.8 * L) ∧ (length_percent_decrease = 20) := by
  sorry

end NUMINAMATH_GPT_towel_length_decrease_l413_41341


namespace NUMINAMATH_GPT_find_daily_wage_c_l413_41356

noncomputable def daily_wage_c (total_earning : ℕ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (days_d : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) (ratio_d : ℕ) : ℝ :=
  let total_ratio := days_a * ratio_a + days_b * ratio_b + days_c * ratio_c + days_d * ratio_d
  let x := total_earning / total_ratio
  ratio_c * x

theorem find_daily_wage_c :
  daily_wage_c 3780 6 9 4 12 3 4 5 7 = 119.60 :=
by
  sorry

end NUMINAMATH_GPT_find_daily_wage_c_l413_41356


namespace NUMINAMATH_GPT_age_ratio_l413_41397
open Nat

theorem age_ratio (B A x : ℕ) (h1 : B - 4 = 2 * (A - 4)) 
                                (h2 : B - 8 = 3 * (A - 8)) 
                                (h3 : (B + x) / (A + x) = 3 / 2) : 
                                x = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l413_41397


namespace NUMINAMATH_GPT_find_dividend_l413_41308

theorem find_dividend (D Q R dividend : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) (h4 : dividend = D * Q + R) :
  dividend = 5336 :=
by
  -- We will complete the proof using the provided conditions
  sorry

end NUMINAMATH_GPT_find_dividend_l413_41308


namespace NUMINAMATH_GPT_usual_time_catch_bus_l413_41396

variable (S T T' : ℝ)

theorem usual_time_catch_bus (h1 : T' = T + 6)
  (h2 : S * T = (4 / 5) * S * T') : T = 24 := by
  sorry

end NUMINAMATH_GPT_usual_time_catch_bus_l413_41396


namespace NUMINAMATH_GPT_inverse_function_coeff_ratio_l413_41395

noncomputable def f_inv_coeff_ratio : ℝ :=
  let f (x : ℝ) := (2 * x - 1) / (x + 5)
  let a := 5
  let b := 1
  let c := -1
  let d := 2
  a / c

theorem inverse_function_coeff_ratio :
  f_inv_coeff_ratio = -5 := 
by
  sorry

end NUMINAMATH_GPT_inverse_function_coeff_ratio_l413_41395


namespace NUMINAMATH_GPT_winning_ticket_probability_l413_41387

open BigOperators

-- Calculate n choose k
def choose (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def probability_PowerBall := (1 : ℚ) / 30
def probability_LuckyBalls := (1 : ℚ) / choose 49 6

-- Theorem to prove the result
theorem winning_ticket_probability :
  probability_PowerBall * probability_LuckyBalls = (1 : ℚ) / 419514480 := by
  sorry

end NUMINAMATH_GPT_winning_ticket_probability_l413_41387


namespace NUMINAMATH_GPT_intersection_correct_union_correct_l413_41325

variable (U A B : Set Nat)

def U_set : U = {1, 2, 3, 4, 5, 6} := by sorry
def A_set : A = {2, 4, 5} := by sorry
def B_set : B = {1, 2, 5} := by sorry

theorem intersection_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∩ B) = {2, 5} := by sorry

theorem union_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∪ (U \ B)) = {2, 3, 4, 5, 6} := by sorry

end NUMINAMATH_GPT_intersection_correct_union_correct_l413_41325


namespace NUMINAMATH_GPT_length_of_DG_l413_41380

theorem length_of_DG {AB BC DG DF : ℝ} (h1 : AB = 8) (h2 : BC = 10) (h3 : DG = DF) 
  (h4 : 1/5 * (AB * BC) = 1/2 * DG^2) : DG = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_length_of_DG_l413_41380


namespace NUMINAMATH_GPT_cos_a3_value_l413_41332

theorem cos_a3_value (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 1 + a 3 + a 5 = Real.pi) : 
  Real.cos (a 3) = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_a3_value_l413_41332


namespace NUMINAMATH_GPT_thickness_relation_l413_41377

noncomputable def a : ℝ := (1/3) * Real.sin (1/2)
noncomputable def b : ℝ := (1/2) * Real.sin (1/3)
noncomputable def c : ℝ := (1/3) * Real.cos (7/8)

theorem thickness_relation : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_thickness_relation_l413_41377


namespace NUMINAMATH_GPT_geometric_mean_4_16_l413_41348

theorem geometric_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
sorry

end NUMINAMATH_GPT_geometric_mean_4_16_l413_41348


namespace NUMINAMATH_GPT_lcm_hcf_product_l413_41313

theorem lcm_hcf_product (lcm hcf a b : ℕ) (hlcm : lcm = 2310) (hhcf : hcf = 30) (ha : a = 330) (eq : lcm * hcf = a * b) : b = 210 :=
by {
  sorry
}

end NUMINAMATH_GPT_lcm_hcf_product_l413_41313


namespace NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l413_41391

-- Definitions corresponding to the conditions
def isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a = b ∧ A + B + C = 180) ∧ A = 40 -- Isosceles and sum of angles is 180° with apex angle A = 40°

-- The theorem to be proven
theorem base_angle_of_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  isosceles_triangle a b c A B C → B = 70 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l413_41391


namespace NUMINAMATH_GPT_bacteria_eradication_time_l413_41349

noncomputable def infected_bacteria (n : ℕ) : ℕ := n

theorem bacteria_eradication_time (n : ℕ) : ∃ t : ℕ, t = n ∧ (∃ infect: ℕ → ℕ, ∀ t < n, infect t ≤ n ∧ infect n = n ∧ (∀ k < n, infect k = 2^(n-k))) :=
by sorry

end NUMINAMATH_GPT_bacteria_eradication_time_l413_41349


namespace NUMINAMATH_GPT_tub_drain_time_l413_41303

theorem tub_drain_time (time_for_five_sevenths : ℝ)
  (time_for_five_sevenths_eq_four : time_for_five_sevenths = 4) :
  let rate := time_for_five_sevenths / (5 / 7)
  let time_for_two_sevenths := 2 * rate
  time_for_two_sevenths = 11.2 := by
  -- Definitions and initial conditions
  sorry

end NUMINAMATH_GPT_tub_drain_time_l413_41303


namespace NUMINAMATH_GPT_inequality_l413_41368

theorem inequality (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_l413_41368


namespace NUMINAMATH_GPT_point_exists_if_square_or_rhombus_l413_41365

-- Definitions to state the problem
structure Point (α : Type*) := (x : α) (y : α)
structure Rectangle (α : Type*) := (A B C D : Point α)

-- Definition of equidistant property
def isEquidistant (α : Type*) [LinearOrderedField α] (P : Point α) (R : Rectangle α) : Prop :=
  let d1 := abs (P.y - R.A.y)
  let d2 := abs (P.y - R.C.y)
  let d3 := abs (P.x - R.A.x)
  let d4 := abs (P.x - R.B.x)
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4

-- Theorem stating the problem
theorem point_exists_if_square_or_rhombus {α : Type*} [LinearOrderedField α]
  (R : Rectangle α) : 
  (∃ P : Point α, isEquidistant α P R) ↔ 
  (∃ (a b : α), (a ≠ b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b) ∨ 
                (a = b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b)) :=
sorry

end NUMINAMATH_GPT_point_exists_if_square_or_rhombus_l413_41365


namespace NUMINAMATH_GPT_layers_tall_l413_41355

def total_cards (n_d c_d : ℕ) : ℕ := n_d * c_d
def layers (total c_l : ℕ) : ℕ := total / c_l

theorem layers_tall (n_d c_d c_l : ℕ) (hn_d : n_d = 16) (hc_d : c_d = 52) (hc_l : c_l = 26) : 
  layers (total_cards n_d c_d) c_l = 32 := by
  sorry

end NUMINAMATH_GPT_layers_tall_l413_41355


namespace NUMINAMATH_GPT_percentage_dogs_movies_l413_41351

-- Definitions from conditions
def total_students : ℕ := 30
def students_preferring_dogs_videogames : ℕ := total_students / 2
def students_preferring_dogs : ℕ := 18
def students_preferring_dogs_movies : ℕ := students_preferring_dogs - students_preferring_dogs_videogames

-- Theorem statement
theorem percentage_dogs_movies : (students_preferring_dogs_movies * 100 / total_students) = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_dogs_movies_l413_41351


namespace NUMINAMATH_GPT_line_through_points_a_plus_b_l413_41366

theorem line_through_points_a_plus_b :
  ∃ a b : ℝ, (∀ x y : ℝ, (y = a * x + b) → ((x, y) = (6, 7)) ∨ ((x, y) = (10, 23))) ∧ (a + b = -13) :=
sorry

end NUMINAMATH_GPT_line_through_points_a_plus_b_l413_41366


namespace NUMINAMATH_GPT_wood_cost_l413_41305

theorem wood_cost (C : ℝ) (h1 : 20 * 15 = 300) (h2 : 300 - C = 200) : C = 100 :=
by
  -- The proof is to be filled here, but it is currently skipped with 'sorry'.
  sorry

end NUMINAMATH_GPT_wood_cost_l413_41305


namespace NUMINAMATH_GPT_parallel_line_through_P_perpendicular_line_through_P_l413_41386

-- Define point P
def P := (-4, 2)

-- Define line l
def l (x y : ℝ) := 3 * x - 2 * y - 7 = 0

-- Define the equation of the line parallel to l that passes through P
def parallel_line (x y : ℝ) := 3 * x - 2 * y + 16 = 0

-- Define the equation of the line perpendicular to l that passes through P
def perpendicular_line (x y : ℝ) := 2 * x + 3 * y + 2 = 0

-- Theorem 1: Prove that parallel_line is the equation of the line passing through P and parallel to l
theorem parallel_line_through_P :
  ∀ (x y : ℝ), 
    (parallel_line x y → x = -4 ∧ y = 2) :=
sorry

-- Theorem 2: Prove that perpendicular_line is the equation of the line passing through P and perpendicular to l
theorem perpendicular_line_through_P :
  ∀ (x y : ℝ), 
    (perpendicular_line x y → x = -4 ∧ y = 2) :=
sorry

end NUMINAMATH_GPT_parallel_line_through_P_perpendicular_line_through_P_l413_41386


namespace NUMINAMATH_GPT_triangle_side_lengths_l413_41346

variable {c z m : ℕ}

axiom condition1 : 3 * c + z + m = 43
axiom condition2 : c + z + 3 * m = 35
axiom condition3 : 2 * (c + z + m) = 46

theorem triangle_side_lengths : c = 10 ∧ z = 7 ∧ m = 6 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l413_41346


namespace NUMINAMATH_GPT_projections_relationship_l413_41328

theorem projections_relationship (a b r : ℝ) (h : r ≠ 0) :
  (∃ α β : ℝ, a = r * Real.cos α ∧ b = r * Real.cos β ∧ (Real.cos α)^2 + (Real.cos β)^2 = 1) → (a^2 + b^2 = r^2) :=
by
  sorry

end NUMINAMATH_GPT_projections_relationship_l413_41328
