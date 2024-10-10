import Mathlib

namespace derivative_x_squared_cos_x_l2923_292338

theorem derivative_x_squared_cos_x :
  let y : ℝ → ℝ := λ x ↦ x^2 * Real.cos x
  deriv y = λ x ↦ 2 * x * Real.cos x - x^2 * Real.sin x := by
sorry

end derivative_x_squared_cos_x_l2923_292338


namespace fraction_simplification_l2923_292311

theorem fraction_simplification (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end fraction_simplification_l2923_292311


namespace total_distance_travelled_l2923_292346

theorem total_distance_travelled (distance_by_land distance_by_sea : ℕ) 
  (h1 : distance_by_land = 451)
  (h2 : distance_by_sea = 150) : 
  distance_by_land + distance_by_sea = 601 := by
sorry

end total_distance_travelled_l2923_292346


namespace at_least_one_zero_one_is_zero_l2923_292393

theorem at_least_one_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → False := by
  sorry

theorem one_is_zero (a b : ℝ) : a = 0 ∨ b = 0 := by
  sorry

end at_least_one_zero_one_is_zero_l2923_292393


namespace rationalize_denominator_l2923_292396

theorem rationalize_denominator (x : ℝ) (hx : x > 0) :
  (1 : ℝ) / (x^(1/3) + (27 : ℝ)^(1/3)) = (4 : ℝ)^(1/3) / (2 + 3 * (4 : ℝ)^(1/3)) :=
by sorry

end rationalize_denominator_l2923_292396


namespace fraction_equality_implies_sum_l2923_292351

theorem fraction_equality_implies_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 992) / (x^2 + 56*x - 3168)) →
  α + β = 82 := by
sorry

end fraction_equality_implies_sum_l2923_292351


namespace subset_conditions_l2923_292365

/-- Given sets A and B, prove the conditions for m when A is a proper subset of B -/
theorem subset_conditions (m : ℝ) : 
  let A : Set ℝ := {3, m^2}
  let B : Set ℝ := {1, 3, 2*m-1}
  (A ⊂ B) → (m^2 ≠ 1 ∧ m^2 ≠ 2*m-1 ∧ m^2 ≠ 3) :=
by sorry

end subset_conditions_l2923_292365


namespace algebraic_expression_value_l2923_292322

theorem algebraic_expression_value (a b : ℝ) (h : 5*a + 3*b = -4) :
  -8 - 2*(a + b) - 4*(2*a + b) = 0 := by
  sorry

end algebraic_expression_value_l2923_292322


namespace amy_balloon_count_l2923_292383

/-- The number of balloons James has -/
def james_balloons : ℕ := 1222

/-- The difference between James' and Amy's balloon counts -/
def difference : ℕ := 208

/-- Amy's balloon count -/
def amy_balloons : ℕ := james_balloons - difference

theorem amy_balloon_count : amy_balloons = 1014 := by
  sorry

end amy_balloon_count_l2923_292383


namespace sum_of_three_smallest_solutions_l2923_292358

def is_solution (x : ℝ) : Prop :=
  x > 0 ∧ x - ⌊x⌋ = 1 / (⌊x⌋^2)

def smallest_solutions : Set ℝ :=
  {x | is_solution x ∧ ∀ y, is_solution y → x ≤ y}

theorem sum_of_three_smallest_solutions :
  ∃ (a b c : ℝ), a ∈ smallest_solutions ∧ b ∈ smallest_solutions ∧ c ∈ smallest_solutions ∧
  (∀ x ∈ smallest_solutions, x = a ∨ x = b ∨ x = c) ∧
  a + b + c = 9 + 17/36 :=
sorry

end sum_of_three_smallest_solutions_l2923_292358


namespace square_ratio_problem_l2923_292330

theorem square_ratio_problem (A₁ A₂ : ℝ) (s₁ s₂ : ℝ) :
  A₁ / A₂ = 300 / 147 →
  A₁ = s₁^2 →
  A₂ = s₂^2 →
  4 * s₁ = 60 →
  s₁ / s₂ = 10 / 7 ∧ s₂ = 21 / 2 :=
by sorry

end square_ratio_problem_l2923_292330


namespace imaginary_part_of_complex_fraction_l2923_292300

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (15 * Complex.I) / (3 + 4 * Complex.I)
  Complex.im z = 9 / 5 := by sorry

end imaginary_part_of_complex_fraction_l2923_292300


namespace square_perimeter_l2923_292369

theorem square_perimeter (side_length : ℝ) (h : side_length = 13) : 
  4 * side_length = 52 := by
  sorry

end square_perimeter_l2923_292369


namespace lego_count_l2923_292320

theorem lego_count (initial : Nat) (lost : Nat) (remaining : Nat) : 
  initial = 380 → lost = 57 → remaining = initial - lost → remaining = 323 := by
  sorry

end lego_count_l2923_292320


namespace triangle_inequality_violation_l2923_292394

theorem triangle_inequality_violation (a b c : ℝ) : 
  a = 1 ∧ b = 2 ∧ c = 7 → ¬(a + b > c ∧ b + c > a ∧ c + a > b) := by
  sorry

end triangle_inequality_violation_l2923_292394


namespace ellipse_major_axis_length_l2923_292375

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorAxisLongerRatio : ℝ) : ℝ :=
  2 * cylinderRadius * (1 + majorAxisLongerRatio)

/-- Theorem stating the length of the major axis of the ellipse --/
theorem ellipse_major_axis_length :
  majorAxisLength 2 0.3 = 5.2 := by sorry

end ellipse_major_axis_length_l2923_292375


namespace ratio_to_twelve_l2923_292359

theorem ratio_to_twelve : ∃ x : ℝ, (5 : ℝ) / 1 = x / 12 ∧ x = 60 := by
  sorry

end ratio_to_twelve_l2923_292359


namespace expected_balls_original_value_l2923_292350

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The number of interchanges performed -/
def num_interchanges : ℕ := 4

/-- The probability of a specific ball being chosen for an interchange -/
def prob_chosen : ℚ := 2 / 7

/-- The probability of a ball being in its original position after the interchanges -/
def prob_original_position : ℚ :=
  (1 - prob_chosen) ^ num_interchanges +
  (num_interchanges.choose 2) * prob_chosen ^ 2 * (1 - prob_chosen) ^ 2 +
  prob_chosen ^ num_interchanges

/-- The expected number of balls in their original positions -/
def expected_balls_original : ℚ := num_balls * prob_original_position

theorem expected_balls_original_value :
  expected_balls_original = 3.61 := by sorry

end expected_balls_original_value_l2923_292350


namespace probability_at_least_one_woman_l2923_292345

def total_people : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def selected : ℕ := 5

theorem probability_at_least_one_woman :
  let p := 1 - (num_men.choose selected / total_people.choose selected)
  p = 917 / 1001 := by sorry

end probability_at_least_one_woman_l2923_292345


namespace negation_of_universal_proposition_l2923_292356

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x < 0)) ↔ (∃ x : ℝ, x^2 - x ≥ 0) := by sorry

end negation_of_universal_proposition_l2923_292356


namespace quadratic_roots_pure_imaginary_l2923_292321

theorem quadratic_roots_pure_imaginary (m : ℂ) (h : m.re = 0 ∧ m.im ≠ 0) :
  ∃ (z₁ z₂ : ℂ), z₁.re = 0 ∧ z₂.re = 0 ∧ 
  8 * z₁^2 + 4 * Complex.I * z₁ - m = 0 ∧
  8 * z₂^2 + 4 * Complex.I * z₂ - m = 0 :=
sorry

end quadratic_roots_pure_imaginary_l2923_292321


namespace hypotenuse_length_of_special_triangle_l2923_292334

theorem hypotenuse_length_of_special_triangle : 
  ∀ (a b c : ℝ), 
  (a^2 - 17*a + 60 = 0) → 
  (b^2 - 17*b + 60 = 0) → 
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = 13 := by
sorry

end hypotenuse_length_of_special_triangle_l2923_292334


namespace hyperbola_ratio_range_l2923_292390

/-- A hyperbola with foci F₁ and F₂, and a point G satisfying specific conditions -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  G : ℝ × ℝ
  hC : G.1^2 / a^2 - G.2^2 / b^2 = 1
  hG : Real.sqrt ((G.1 - F₁.1)^2 + (G.2 - F₁.2)^2) = 7 * Real.sqrt ((G.1 - F₂.1)^2 + (G.2 - F₂.2)^2)

/-- The range of b/a for a hyperbola satisfying the given conditions -/
theorem hyperbola_ratio_range (h : Hyperbola) : 0 < h.b / h.a ∧ h.b / h.a ≤ Real.sqrt 7 / 3 := by
  sorry

end hyperbola_ratio_range_l2923_292390


namespace marias_candy_l2923_292318

/-- The number of candy pieces Maria ate -/
def pieces_eaten : ℕ := 64

/-- The number of candy pieces Maria has left -/
def pieces_left : ℕ := 3

/-- The initial number of candy pieces Maria had -/
def initial_pieces : ℕ := pieces_eaten + pieces_left

theorem marias_candy : initial_pieces = 67 := by
  sorry

end marias_candy_l2923_292318


namespace largest_sphere_on_torus_l2923_292361

/-- Represents a torus generated by revolving a circle about the z-axis -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  circle_center : ℝ × ℝ × ℝ
  circle_radius : ℝ

/-- Represents a sphere centered on the z-axis -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Checks if a sphere touches the horizontal plane -/
def touches_plane (s : Sphere) : Prop :=
  s.center.2.2 = s.radius

/-- Checks if a sphere touches the top of a torus -/
def touches_torus (t : Torus) (s : Sphere) : Prop :=
  (t.circle_center.1 - s.center.1) ^ 2 + (t.circle_center.2.2 - s.center.2.2) ^ 2 = (s.radius + t.circle_radius) ^ 2

theorem largest_sphere_on_torus (t : Torus) (s : Sphere) :
  t.inner_radius = 3 ∧
  t.outer_radius = 5 ∧
  t.circle_center = (4, 0, 1) ∧
  t.circle_radius = 1 ∧
  s.center.1 = 0 ∧
  s.center.2.1 = 0 ∧
  touches_plane s ∧
  touches_torus t s →
  s.radius = 4 :=
sorry

end largest_sphere_on_torus_l2923_292361


namespace paulas_remaining_money_l2923_292370

/-- Calculates the remaining money after shopping given the initial amount, 
    number of shirts, cost per shirt, and cost of pants. -/
def remaining_money (initial : ℕ) (num_shirts : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) : ℕ :=
  initial - (num_shirts * shirt_cost + pants_cost)

/-- Theorem stating that Paula's remaining money after shopping is $74 -/
theorem paulas_remaining_money :
  remaining_money 109 2 11 13 = 74 := by
  sorry

end paulas_remaining_money_l2923_292370


namespace roof_collapse_time_l2923_292343

/-- The number of days it takes for Bill's roof to collapse under the weight of leaves -/
def days_to_collapse (roof_capacity : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : ℕ :=
  (roof_capacity * leaves_per_pound) / leaves_per_day

/-- Theorem stating that it takes 5000 days for Bill's roof to collapse -/
theorem roof_collapse_time :
  days_to_collapse 500 100 1000 = 5000 := by
  sorry

end roof_collapse_time_l2923_292343


namespace sum_remainder_mod_seven_l2923_292324

theorem sum_remainder_mod_seven (n : ℤ) : 
  ((7 - n) + (n + 3) + n^2) % 7 = (3 + n^2) % 7 := by
  sorry

end sum_remainder_mod_seven_l2923_292324


namespace percentage_added_to_a_l2923_292331

-- Define the ratio of a to b
def ratio_a_b : ℚ := 4 / 5

-- Define the percentage decrease for m
def decrease_percent : ℚ := 80

-- Define the ratio of m to x
def ratio_m_x : ℚ := 1 / 7

-- Define the function to calculate x given a and P
def x_from_a (a : ℚ) (P : ℚ) : ℚ := a * (1 + P / 100)

-- Define the function to calculate m given b
def m_from_b (b : ℚ) : ℚ := b * (1 - decrease_percent / 100)

-- Theorem statement
theorem percentage_added_to_a (a b : ℚ) (P : ℚ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = ratio_a_b) 
  (h4 : m_from_b b / x_from_a a P = ratio_m_x) : P = 75 := by
  sorry

end percentage_added_to_a_l2923_292331


namespace place_mat_length_l2923_292347

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  r = 6 ∧ n = 8 ∧ w = 1 ∧ 
  (∀ i : Fin n, ∃ p₁ p₂ : ℝ × ℝ, 
    (p₁.1 - r)^2 + p₁.2^2 = r^2 ∧
    (p₂.1 - r)^2 + p₂.2^2 = r^2 ∧
    (p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2 = w^2) ∧
  (∀ i : Fin n, ∃ q₁ q₂ : ℝ × ℝ,
    (q₁.1 - r)^2 + q₁.2^2 < r^2 ∧
    (q₂.1 - r)^2 + q₂.2^2 < r^2 ∧
    (q₂.1 - q₁.1)^2 + (q₂.2 - q₁.2)^2 = y^2 ∧
    (∃ j : Fin n, j ≠ i ∧ (q₂.1 = q₁.1 ∨ q₂.2 = q₁.2))) →
  y = 3 * Real.sqrt (2 - Real.sqrt 2) := by
sorry

end place_mat_length_l2923_292347


namespace rsa_factorization_l2923_292353

theorem rsa_factorization :
  ∃ (p q : ℕ), 
    p.Prime ∧ 
    q.Prime ∧ 
    p * q = 400000001 ∧ 
    p = 19801 ∧ 
    q = 20201 := by
  sorry

end rsa_factorization_l2923_292353


namespace cracker_difference_l2923_292382

theorem cracker_difference (marcus_crackers : ℕ) (nicholas_crackers : ℕ) : 
  marcus_crackers = 27 →
  nicholas_crackers = 15 →
  ∃ (mona_crackers : ℕ), 
    marcus_crackers = 3 * mona_crackers ∧
    nicholas_crackers = mona_crackers + 6 :=
by
  sorry

end cracker_difference_l2923_292382


namespace sector_max_area_l2923_292363

/-- Given a sector with perimeter 60 cm, its maximum area is 225 cm² -/
theorem sector_max_area (r : ℝ) (l : ℝ) (S : ℝ → ℝ) :
  (0 < r) → (r < 30) →
  (l + 2 * r = 60) →
  (S = λ r => (1 / 2) * l * r) →
  (∀ r', S r' ≤ 225) ∧ (∃ r', S r' = 225) :=
sorry

end sector_max_area_l2923_292363


namespace marshmallow_roasting_l2923_292302

/-- The number of marshmallows Joe's dad has -/
def dads_marshmallows : ℕ := 21

/-- The number of marshmallows Joe has -/
def joes_marshmallows : ℕ := 4 * dads_marshmallows

/-- The number of marshmallows Joe's dad roasts -/
def dads_roasted : ℕ := dads_marshmallows / 3

/-- The number of marshmallows Joe roasts -/
def joes_roasted : ℕ := joes_marshmallows / 2

/-- The total number of marshmallows roasted by Joe and his dad -/
def total_roasted : ℕ := dads_roasted + joes_roasted

theorem marshmallow_roasting :
  total_roasted = 49 := by
  sorry

end marshmallow_roasting_l2923_292302


namespace units_digit_of_4589_pow_1276_l2923_292355

theorem units_digit_of_4589_pow_1276 : ∃ n : ℕ, 4589^1276 ≡ 1 [ZMOD 10] :=
by sorry

end units_digit_of_4589_pow_1276_l2923_292355


namespace matthias_balls_without_holes_l2923_292332

/-- The number of balls without holes in Matthias' collection -/
def balls_without_holes (total_soccer : ℕ) (total_basketball : ℕ) (soccer_with_holes : ℕ) (basketball_with_holes : ℕ) : ℕ :=
  (total_soccer - soccer_with_holes) + (total_basketball - basketball_with_holes)

/-- Theorem stating the total number of balls without holes in Matthias' collection -/
theorem matthias_balls_without_holes :
  balls_without_holes 180 75 125 49 = 81 := by
  sorry

end matthias_balls_without_holes_l2923_292332


namespace max_gold_tokens_l2923_292348

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange booth --/
structure Booth where
  red_in : ℕ
  blue_in : ℕ
  red_out : ℕ
  blue_out : ℕ
  gold_out : ℕ

/-- Checks if an exchange is possible at a given booth --/
def canExchange (tokens : TokenCount) (booth : Booth) : Bool :=
  tokens.red ≥ booth.red_in ∧ tokens.blue ≥ booth.blue_in

/-- Performs an exchange at a given booth --/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.red_in + booth.red_out,
    blue := tokens.blue - booth.blue_in + booth.blue_out,
    gold := tokens.gold + booth.gold_out }

/-- The main theorem to prove --/
theorem max_gold_tokens : ∃ (final : TokenCount),
  let initial := TokenCount.mk 100 100 0
  let booth1 := Booth.mk 3 0 0 2 1
  let booth2 := Booth.mk 0 4 2 0 1
  (¬ canExchange final booth1 ∧ ¬ canExchange final booth2) ∧
  final.gold = 133 ∧
  (∀ (other : TokenCount),
    (¬ canExchange other booth1 ∧ ¬ canExchange other booth2) →
    other.gold ≤ final.gold) :=
sorry

end max_gold_tokens_l2923_292348


namespace v2_value_for_f_at_2_l2923_292388

def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x + 2 * x^2 - x + 5

def qin_jiushao_v2 (a b c d e : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * x + c

theorem v2_value_for_f_at_2 :
  let a := 2
  let b := 3
  let c := 0
  qin_jiushao_v2 a b c 5 (-4) 2 = 14 := by sorry

end v2_value_for_f_at_2_l2923_292388


namespace total_lines_through_centers_l2923_292301

/-- The size of the cube --/
def cube_size : Nat := 2008

/-- The number of lines parallel to the edges of the cube --/
def parallel_lines : Nat := cube_size * cube_size * 3

/-- The number of diagonal lines within the planes --/
def diagonal_lines : Nat := cube_size * 2 * 3

/-- The number of space diagonals of the cube --/
def space_diagonals : Nat := 4

/-- Theorem stating the total number of lines passing through the centers of exactly 2008 unit cubes in a 2008 x 2008 x 2008 cube --/
theorem total_lines_through_centers (cube_size : Nat) (h : cube_size = 2008) :
  parallel_lines + diagonal_lines + space_diagonals = 12115300 := by
  sorry

#eval parallel_lines + diagonal_lines + space_diagonals

end total_lines_through_centers_l2923_292301


namespace smallest_product_l2923_292364

def S : Finset Int := {-10, -3, 0, 4, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y ≤ a * b ∧ x * y = -60 :=
by sorry

end smallest_product_l2923_292364


namespace system_solution_no_solution_l2923_292360

-- Problem 1
theorem system_solution (x y : ℝ) :
  x - y = 8 ∧ 3*x + y = 12 → x = 5 ∧ y = -3 := by sorry

-- Problem 2
theorem no_solution (x : ℝ) :
  x ≠ 1 → 3 / (x - 1) - (x + 2) / (x * (x - 1)) ≠ 0 := by sorry

end system_solution_no_solution_l2923_292360


namespace steel_to_tin_ratio_l2923_292392

-- Define the masses of the bars
def mass_copper : ℝ := 90
def mass_steel : ℝ := mass_copper + 20

-- Define the total mass of all bars
def total_mass : ℝ := 5100

-- Define the number of bars of each type
def num_bars : ℕ := 20

-- Theorem statement
theorem steel_to_tin_ratio : 
  ∃ (mass_tin : ℝ), 
    mass_tin > 0 ∧ 
    num_bars * (mass_steel + mass_copper + mass_tin) = total_mass ∧ 
    mass_steel / mass_tin = 2 := by
  sorry

end steel_to_tin_ratio_l2923_292392


namespace four_digit_divisibility_l2923_292395

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem four_digit_divisibility (p q : ℕ) : 
  is_two_digit_prime p ∧ 
  is_two_digit_prime q ∧ 
  p ≠ q ∧
  (100 * p + q) % ((p + q) / 2) = 0 ∧ 
  (100 * q + p) % ((p + q) / 2) = 0 →
  ({p, q} : Set ℕ) = {13, 53} ∨ 
  ({p, q} : Set ℕ) = {19, 47} ∨ 
  ({p, q} : Set ℕ) = {23, 43} ∨ 
  ({p, q} : Set ℕ) = {29, 37} :=
by sorry

end four_digit_divisibility_l2923_292395


namespace gcd_factorial_problem_l2923_292344

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l2923_292344


namespace inequality_solution_set_l2923_292349

theorem inequality_solution_set : 
  {x : ℝ | -x^2 - x + 6 > 0} = Set.Ioo (-3 : ℝ) 2 := by sorry

end inequality_solution_set_l2923_292349


namespace gcd_of_B_is_two_l2923_292342

def B : Set ℕ := {n | ∃ k : ℕ, n = k + (k + 1) + (k + 2) + (k + 3)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ b ∈ B, d ∣ b) ∧ (∀ m : ℕ, (∀ b ∈ B, m ∣ b) → m ∣ d) ∧ d = 2 := by
sorry

end gcd_of_B_is_two_l2923_292342


namespace cube_root_monotone_l2923_292340

theorem cube_root_monotone (a b : ℝ) (h : a ≤ b) : a ^ (1/3) ≤ b ^ (1/3) := by
  sorry

end cube_root_monotone_l2923_292340


namespace polynomial_root_property_l2923_292352

/-- A polynomial of degree 4 with real coefficients -/
def PolynomialDegree4 (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- The derivative of a polynomial of degree 4 -/
def DerivativePolynomialDegree4 (a b c : ℝ) (x : ℝ) : ℝ := 4*x^3 + 3*a*x^2 + 2*b*x + c

theorem polynomial_root_property (a b c d : ℝ) :
  let f := PolynomialDegree4 a b c d
  let f' := DerivativePolynomialDegree4 a b c
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (∃ w x y z : ℝ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z ∧
    f w = 0 ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) ∨
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = 0 ∧ f y = 0 ∧ f z = 0 ∧
    (f' x = 0 ∨ f' y = 0 ∨ f' z = 0)) := by
  sorry

end polynomial_root_property_l2923_292352


namespace square_sum_given_sum_and_product_l2923_292387

theorem square_sum_given_sum_and_product (a b : ℝ) : a + b = 8 → a * b = -2 → a^2 + b^2 = 68 := by
  sorry

end square_sum_given_sum_and_product_l2923_292387


namespace not_all_structure_diagrams_are_tree_shaped_l2923_292336

/-- Represents a structure diagram -/
structure StructureDiagram where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Property: Elements show conceptual subordination and logical sequence -/
def shows_conceptual_subordination (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Can reflect relationships and overall characteristics -/
def reflects_relationships (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Can reflect system details thoroughly -/
def reflects_details (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Is tree-shaped -/
def is_tree_shaped (sd : StructureDiagram) : Prop :=
  sorry

/-- Theorem: Not all structure diagrams are tree-shaped -/
theorem not_all_structure_diagrams_are_tree_shaped :
  ¬ (∀ sd : StructureDiagram, is_tree_shaped sd) :=
sorry

end not_all_structure_diagrams_are_tree_shaped_l2923_292336


namespace linear_equation_solution_l2923_292385

theorem linear_equation_solution :
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 ∧ x = -30 := by sorry

end linear_equation_solution_l2923_292385


namespace hyperbola_eccentricity_range_l2923_292333

/-- The eccentricity range of a hyperbola -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  let c := Real.sqrt (a^2 + b^2)
  let d := a * b / c
  d ≥ Real.sqrt 2 / 3 * c →
  Real.sqrt 6 / 2 ≤ e ∧ e ≤ Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_range_l2923_292333


namespace cosine_product_theorem_l2923_292376

theorem cosine_product_theorem :
  (1 + Real.cos (π / 10)) * (1 + Real.cos (3 * π / 10)) *
  (1 + Real.cos (7 * π / 10)) * (1 + Real.cos (9 * π / 10)) =
  (3 - Real.sqrt 5) / 32 := by
sorry

end cosine_product_theorem_l2923_292376


namespace price_ratio_theorem_l2923_292315

theorem price_ratio_theorem (CP : ℝ) (SP1 SP2 : ℝ) 
  (h1 : SP1 = CP * (1 + 0.2))
  (h2 : SP2 = CP * (1 - 0.2)) :
  SP2 / SP1 = 2 / 3 := by
sorry

end price_ratio_theorem_l2923_292315


namespace tg_arccos_leq_sin_arctg_l2923_292367

theorem tg_arccos_leq_sin_arctg (x : ℝ) : 
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.tan (Real.arccos x) ≤ Real.sin (Real.arctan x) ↔ 
   x ∈ Set.Icc (-(Real.sqrt (Real.sqrt (1/2)))) 0 ∪ Set.Icc (Real.sqrt (Real.sqrt (1/2))) 1) :=
by sorry

end tg_arccos_leq_sin_arctg_l2923_292367


namespace gemma_pizza_payment_l2923_292304

/-- The amount of money Gemma gave for her pizza order -/
def amount_given (num_pizzas : ℕ) (price_per_pizza : ℕ) (tip : ℕ) (change : ℕ) : ℕ :=
  num_pizzas * price_per_pizza + tip + change

/-- Proof that Gemma gave $50 for her pizza order -/
theorem gemma_pizza_payment :
  amount_given 4 10 5 5 = 50 := by
  sorry

end gemma_pizza_payment_l2923_292304


namespace complex_number_in_fourth_quadrant_l2923_292305

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 + Complex.I) * (1 - Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l2923_292305


namespace worker_completion_time_l2923_292368

/-- Given two workers who can complete a job together in a certain time,
    and one worker's individual completion time, find the other worker's time. -/
theorem worker_completion_time
  (total_time : ℝ)
  (together_time : ℝ)
  (b_time : ℝ)
  (h1 : together_time > 0)
  (h2 : b_time > 0)
  (h3 : total_time > 0)
  (h4 : 1 / together_time = 1 / total_time + 1 / b_time) :
  total_time = 15 :=
sorry

end worker_completion_time_l2923_292368


namespace domino_arrangements_count_l2923_292310

structure Grid :=
  (rows : Nat)
  (cols : Nat)

structure Domino :=
  (length : Nat)
  (width : Nat)

def count_arrangements (g : Grid) (d : Domino) (num_dominoes : Nat) : Nat :=
  Nat.choose (g.rows + g.cols - 2) (g.cols - 1)

theorem domino_arrangements_count (g : Grid) (d : Domino) (num_dominoes : Nat) :
  g.rows = 6 → g.cols = 4 → d.length = 2 → d.width = 1 → num_dominoes = 4 →
  count_arrangements g d num_dominoes = 126 := by
  sorry

end domino_arrangements_count_l2923_292310


namespace randys_trees_l2923_292327

/-- Proves that Randy has 5 less coconut trees compared to half the number of mango trees -/
theorem randys_trees (mango_trees : ℕ) (total_trees : ℕ) (coconut_trees : ℕ) : 
  mango_trees = 60 →
  total_trees = 85 →
  coconut_trees = total_trees - mango_trees →
  coconut_trees < mango_trees / 2 →
  mango_trees / 2 - coconut_trees = 5 := by
sorry

end randys_trees_l2923_292327


namespace square_plus_product_equals_twelve_plus_two_sqrt_six_l2923_292389

theorem square_plus_product_equals_twelve_plus_two_sqrt_six :
  ∀ a b : ℝ,
  a = Real.sqrt 6 + 1 →
  b = Real.sqrt 6 - 1 →
  a^2 + a*b = 12 + 2 * Real.sqrt 6 := by
  sorry

end square_plus_product_equals_twelve_plus_two_sqrt_six_l2923_292389


namespace robins_bracelet_cost_l2923_292354

/-- Represents a friend's name -/
inductive Friend
| jessica
| tori
| lily
| patrice

/-- Returns the number of letters in a friend's name -/
def nameLength (f : Friend) : Nat :=
  match f with
  | .jessica => 7
  | .tori => 4
  | .lily => 4
  | .patrice => 7

/-- The cost of a single bracelet in dollars -/
def braceletCost : Nat := 2

/-- The list of Robin's friends -/
def friendsList : List Friend := [Friend.jessica, Friend.tori, Friend.lily, Friend.patrice]

/-- Theorem: The total cost for Robin's bracelets is $44 -/
theorem robins_bracelet_cost : 
  (friendsList.map nameLength).sum * braceletCost = 44 := by
  sorry


end robins_bracelet_cost_l2923_292354


namespace quadratic_factorization_l2923_292339

theorem quadratic_factorization (x : ℝ) : x^2 + 2*x - 3 = (x + 3) * (x - 1) := by
  sorry

end quadratic_factorization_l2923_292339


namespace quadratic_rewrite_l2923_292337

-- Define the quadratic expression
def quadratic (k : ℝ) : ℝ := 8 * k^2 - 16 * k + 28

-- Define the completed square form
def completed_square (k a b c : ℝ) : ℝ := a * (k + b)^2 + c

-- Theorem statement
theorem quadratic_rewrite :
  ∃ (a b c : ℝ), 
    (∀ k, quadratic k = completed_square k a b c) ∧ 
    (c / b = -20) := by
  sorry

end quadratic_rewrite_l2923_292337


namespace range_of_m_l2923_292303

/-- Proposition p: The equation x^2+mx+1=0 has exactly two distinct negative roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  ∀ x : ℝ, x^2 + m*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)

/-- Proposition q: The inequality 3^x-m+1≤0 has a real solution -/
def q (m : ℝ) : Prop :=
  ∃ x : ℝ, 3^x - m + 1 ≤ 0

/-- The range of m given the conditions -/
theorem range_of_m :
  ∀ m : ℝ, (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (1 < m ∧ m ≤ 2) :=
sorry

end range_of_m_l2923_292303


namespace symmetric_point_x_axis_l2923_292380

/-- Given a point P(2, 5), its symmetric point with respect to the x-axis has coordinates (2, -5) -/
theorem symmetric_point_x_axis : 
  let P : ℝ × ℝ := (2, 5)
  let symmetric_point := (P.1, -P.2)
  symmetric_point = (2, -5) := by
sorry

end symmetric_point_x_axis_l2923_292380


namespace train_length_calculation_l2923_292328

/-- Prove that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 46 km/hr and the slower train at 36 km/hr,
    if the faster train completely passes the slower train in 18 seconds,
    then the length of each train is 25 meters. -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 18 →
  (faster_speed - slower_speed) * (5 / 18) * passing_time = 2 * train_length →
  train_length = 25 := by
  sorry


end train_length_calculation_l2923_292328


namespace yumi_counting_l2923_292335

def reduce_number (start : ℕ) (amount : ℕ) (times : ℕ) : ℕ :=
  start - amount * times

theorem yumi_counting :
  reduce_number 320 10 4 = 280 := by
  sorry

end yumi_counting_l2923_292335


namespace probability_square_or_triangle_l2923_292357

theorem probability_square_or_triangle :
  let total_figures : ℕ := 10
  let num_triangles : ℕ := 4
  let num_squares : ℕ := 3
  let num_circles : ℕ := 3
  let favorable_outcomes : ℕ := num_triangles + num_squares
  (favorable_outcomes : ℚ) / total_figures = 7 / 10 :=
by sorry

end probability_square_or_triangle_l2923_292357


namespace inverse_of_complex_expression_l2923_292309

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_complex_expression :
  i ^ 2 = -1 →
  (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by sorry

end inverse_of_complex_expression_l2923_292309


namespace initial_squares_step_increase_squares_after_five_steps_l2923_292378

/-- The number of squares after n steps in the square subdivision process -/
def num_squares (n : ℕ) : ℕ := 5 + 4 * n

/-- The square subdivision process starts with 5 squares -/
theorem initial_squares : num_squares 0 = 5 := by sorry

/-- Each step adds 4 new squares -/
theorem step_increase (n : ℕ) : num_squares (n + 1) = num_squares n + 4 := by sorry

/-- The number of squares after 5 steps is 25 -/
theorem squares_after_five_steps : num_squares 5 = 25 := by sorry

end initial_squares_step_increase_squares_after_five_steps_l2923_292378


namespace area_ratio_theorem_l2923_292397

-- Define the triangle AEF
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function
def area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

-- Define the parallel relation
def parallel : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry

-- Define the cyclic property
def is_cyclic : Quadrilateral → Prop := sorry

-- Define the distance function
def distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (AEF : Triangle) (ABCD : Quadrilateral) :
  distance AEF.B AEF.C = 20 →
  distance AEF.A AEF.B = 21 →
  distance AEF.A AEF.C = 21 →
  parallel ABCD.B ABCD.D AEF.B AEF.C →
  is_cyclic ABCD →
  distance ABCD.B ABCD.C = 3 →
  distance ABCD.C ABCD.D = 4 →
  (area ABCD.A ABCD.B ABCD.C + area ABCD.A ABCD.C ABCD.D) / area AEF.A AEF.B AEF.C = 49 / 400 :=
by sorry

end area_ratio_theorem_l2923_292397


namespace cakes_per_person_l2923_292399

theorem cakes_per_person (total_cakes : ℕ) (num_friends : ℕ) (h1 : total_cakes = 8) (h2 : num_friends = 4) :
  total_cakes / num_friends = 2 := by
  sorry

end cakes_per_person_l2923_292399


namespace binary_1100_equals_12_l2923_292319

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1100_equals_12 :
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end binary_1100_equals_12_l2923_292319


namespace total_students_count_l2923_292314

/-- The number of students who wish to go on a scavenger hunting trip -/
def scavenger_students : ℕ := 4000

/-- The number of students who wish to go on a skiing trip -/
def skiing_students : ℕ := 2 * scavenger_students

/-- The total number of students who wish to go on either trip -/
def total_students : ℕ := scavenger_students + skiing_students

theorem total_students_count : total_students = 12000 := by
  sorry

end total_students_count_l2923_292314


namespace cone_syrup_amount_l2923_292377

/-- The amount of chocolate syrup used on each shake in ounces -/
def syrup_per_shake : ℕ := 4

/-- The number of shakes sold -/
def num_shakes : ℕ := 2

/-- The number of cones sold -/
def num_cones : ℕ := 1

/-- The total amount of chocolate syrup used in ounces -/
def total_syrup : ℕ := 14

/-- The amount of chocolate syrup used on each cone in ounces -/
def syrup_per_cone : ℕ := total_syrup - (syrup_per_shake * num_shakes)

theorem cone_syrup_amount : syrup_per_cone = 6 := by
  sorry

end cone_syrup_amount_l2923_292377


namespace hill_climbing_speed_l2923_292341

/-- Proves that the average speed while climbing is 2.625 km/h given the conditions of the journey -/
theorem hill_climbing_speed 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) 
  (total_average_speed : ℝ) 
  (h1 : uphill_time = 4)
  (h2 : downhill_time = 2)
  (h3 : total_average_speed = 3.5) : 
  (total_average_speed * (uphill_time + downhill_time)) / (2 * uphill_time) = 2.625 := by
  sorry

#eval (3.5 * (4 + 2)) / (2 * 4)  -- This should evaluate to 2.625

end hill_climbing_speed_l2923_292341


namespace green_marbles_after_replacement_l2923_292307

/-- Represents the number of marbles of each color in a jar -/
structure MarbleJar where
  red : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ
  purple : ℕ
  white : ℕ

/-- Calculates the total number of marbles in the jar -/
def totalMarbles (jar : MarbleJar) : ℕ :=
  jar.red + jar.green + jar.blue + jar.yellow + jar.purple + jar.white

/-- Represents the percentage of each color in the jar -/
structure MarblePercentages where
  red : ℚ
  green : ℚ
  blue : ℚ
  yellow : ℚ
  purple : ℚ

/-- Theorem stating the final number of green marbles after replacement -/
theorem green_marbles_after_replacement (jar : MarbleJar) (percentages : MarblePercentages) :
  percentages.red = 25 / 100 →
  percentages.green = 15 / 100 →
  percentages.blue = 20 / 100 →
  percentages.yellow = 10 / 100 →
  percentages.purple = 15 / 100 →
  jar.white = 35 →
  (jar.red : ℚ) / (totalMarbles jar : ℚ) = percentages.red →
  (jar.green : ℚ) / (totalMarbles jar : ℚ) = percentages.green →
  (jar.blue : ℚ) / (totalMarbles jar : ℚ) = percentages.blue →
  (jar.yellow : ℚ) / (totalMarbles jar : ℚ) = percentages.yellow →
  (jar.purple : ℚ) / (totalMarbles jar : ℚ) = percentages.purple →
  (jar.white : ℚ) / (totalMarbles jar : ℚ) = 1 - (percentages.red + percentages.green + percentages.blue + percentages.yellow + percentages.purple) →
  jar.green + jar.red / 3 = 55 := by
  sorry

end green_marbles_after_replacement_l2923_292307


namespace min_value_sum_reciprocals_l2923_292391

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 1.5 := by
  sorry

end min_value_sum_reciprocals_l2923_292391


namespace probability_of_selection_X_l2923_292379

theorem probability_of_selection_X 
  (prob_Y : ℝ) 
  (prob_X_and_Y : ℝ) 
  (h1 : prob_Y = 2/5) 
  (h2 : prob_X_and_Y = 0.13333333333333333) :
  ∃ (prob_X : ℝ), prob_X = 1/3 ∧ prob_X_and_Y = prob_X * prob_Y :=
by
  sorry

end probability_of_selection_X_l2923_292379


namespace perfect_square_trinomial_l2923_292362

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 1 = (x - a)^2) → (k = 2 ∨ k = -2) := by
  sorry

end perfect_square_trinomial_l2923_292362


namespace five_balls_three_boxes_l2923_292316

/-- Represents the number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 21 := by
  sorry

end five_balls_three_boxes_l2923_292316


namespace biker_bob_route_l2923_292371

theorem biker_bob_route (distance_AB : Real) (net_west : Real) (initial_north : Real) :
  distance_AB = 20.615528128088304 →
  net_west = 5 →
  initial_north = 15 →
  ∃ x : Real, 
    x ≥ 0 ∧ 
    (x + initial_north)^2 + net_west^2 = distance_AB^2 ∧ 
    abs (x - 5.021531) < 0.000001 := by
  sorry

#check biker_bob_route

end biker_bob_route_l2923_292371


namespace inequality_and_equality_condition_l2923_292308

theorem inequality_and_equality_condition (x y : ℝ) :
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x ∧
  (5 * x^2 + y^2 + 1 = 4 * x * y + 2 * x ↔ x = 1 ∧ y = 2) :=
by sorry

end inequality_and_equality_condition_l2923_292308


namespace max_remainder_and_dividend_l2923_292384

theorem max_remainder_and_dividend (star : ℕ) (triangle : ℕ) :
  star / 7 = 102 ∧ star % 7 = triangle →
  triangle ≤ 6 ∧
  (triangle = 6 → star = 720) :=
by sorry

end max_remainder_and_dividend_l2923_292384


namespace minimum_point_of_translated_graph_l2923_292366

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 3| - 8

-- Theorem statement
theorem minimum_point_of_translated_graph :
  ∀ x : ℝ, f x ≥ f 3 ∧ f 3 = -8 :=
sorry

end minimum_point_of_translated_graph_l2923_292366


namespace tan_630_undefined_l2923_292374

theorem tan_630_undefined :
  ¬∃ (x : ℝ), Real.tan (630 * π / 180) = x :=
by
  sorry


end tan_630_undefined_l2923_292374


namespace nested_fraction_evaluation_l2923_292326

theorem nested_fraction_evaluation : 
  1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by sorry

end nested_fraction_evaluation_l2923_292326


namespace age_problem_l2923_292323

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 :=
by sorry

end age_problem_l2923_292323


namespace matches_for_512_players_l2923_292317

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_players : ℕ
  matches_eliminate_one : Bool

/-- The number of matches needed to determine the winner in a single-elimination tournament -/
def matches_needed (t : Tournament) : ℕ :=
  t.num_players - 1

/-- Theorem stating the number of matches needed for a 512-player single-elimination tournament -/
theorem matches_for_512_players (t : Tournament) 
  (h1 : t.num_players = 512) 
  (h2 : t.matches_eliminate_one = true) : 
  matches_needed t = 511 := by
  sorry

end matches_for_512_players_l2923_292317


namespace factorization_equality_l2923_292329

theorem factorization_equality (y a : ℝ) : 3*y*a^2 - 6*y*a + 3*y = 3*y*(a-1)^2 := by
  sorry

end factorization_equality_l2923_292329


namespace min_value_expression_l2923_292381

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  ∃ (min : ℝ), min = 162 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x * y * z = 27 →
    x^2 + 6*x*y + 9*y^2 + 3*z^2 ≥ min :=
sorry

end min_value_expression_l2923_292381


namespace cubic_roots_sum_l2923_292312

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/11) := by
sorry

end cubic_roots_sum_l2923_292312


namespace max_dot_product_regular_octagon_l2923_292398

/-- Regular octagon with side length 1 -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, 
    (i.val + 1) % 8 = j.val → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 1

/-- Vector between two points -/
def vector (A : RegularOctagon) (i j : Fin 8) : ℝ × ℝ :=
  ((A.vertices j).1 - (A.vertices i).1, (A.vertices j).2 - (A.vertices i).2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem max_dot_product_regular_octagon (A : RegularOctagon) :
  ∃ (i j : Fin 8), ∀ (k l : Fin 8),
    dot_product (vector A k l) (vector A 0 1) ≤ dot_product (vector A i j) (vector A 0 1) ∧
    dot_product (vector A i j) (vector A 0 1) = Real.sqrt 2 + 1 := by
  sorry

end max_dot_product_regular_octagon_l2923_292398


namespace largest_solution_of_equation_l2923_292372

theorem largest_solution_of_equation (x : ℝ) :
  (((15 * x^2 - 40 * x + 16) / (4 * x - 3)) + 3 * x = 7 * x + 2) →
  x ≤ -14 + Real.sqrt 218 :=
by sorry

end largest_solution_of_equation_l2923_292372


namespace correct_num_technicians_l2923_292386

/-- The number of technicians in a workshop. -/
def num_technicians : ℕ := 7

/-- The total number of workers in the workshop. -/
def total_workers : ℕ := 42

/-- The average salary of all workers in the workshop. -/
def avg_salary_all : ℕ := 8000

/-- The average salary of technicians in the workshop. -/
def avg_salary_technicians : ℕ := 18000

/-- The average salary of non-technician workers in the workshop. -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the workshop conditions. -/
theorem correct_num_technicians :
  num_technicians = 7 ∧
  num_technicians + (total_workers - num_technicians) = total_workers ∧
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = avg_salary_all :=
by sorry

end correct_num_technicians_l2923_292386


namespace correct_train_sequence_l2923_292325

-- Define the actions
inductive TrainAction
  | BuyTicket
  | WaitForTrain
  | CheckTicket
  | BoardTrain
  | RepairTrain

-- Define a sequence of actions
def ActionSequence := List TrainAction

-- Define the possible sequences
def sequenceA : ActionSequence := [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.CheckTicket, TrainAction.BoardTrain]
def sequenceB : ActionSequence := [TrainAction.WaitForTrain, TrainAction.BuyTicket, TrainAction.BoardTrain, TrainAction.CheckTicket]
def sequenceC : ActionSequence := [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.BoardTrain, TrainAction.CheckTicket]
def sequenceD : ActionSequence := [TrainAction.RepairTrain, TrainAction.BuyTicket, TrainAction.CheckTicket, TrainAction.BoardTrain]

-- Define the correct sequence
def correctSequence : ActionSequence := sequenceA

-- Theorem stating that sequenceA is the correct sequence
theorem correct_train_sequence : correctSequence = sequenceA := by
  sorry


end correct_train_sequence_l2923_292325


namespace homogeneous_polynomial_theorem_l2923_292373

variable {n : ℕ}

-- Define a homogeneous polynomial of degree n
def IsHomogeneousPolynomial (f : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (x y t : ℝ), f (t * x) (t * y) = t^n * f x y

theorem homogeneous_polynomial_theorem (f : ℝ → ℝ → ℝ) (n : ℕ) 
  (h1 : IsHomogeneousPolynomial f n)
  (h2 : f 1 0 = 1)
  (h3 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0) :
  ∀ (x y : ℝ), f x y = (x - 2*y) * (x + y)^(n - 1) := by
  sorry

end homogeneous_polynomial_theorem_l2923_292373


namespace sum_is_composite_l2923_292313

theorem sum_is_composite (a b : ℕ) (h : 31 * a = 54 * b) : ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ a + b = k * m := by
  sorry

end sum_is_composite_l2923_292313


namespace geometric_sequence_problem_l2923_292306

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The statement of the problem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 ^ 2 + 6 * a 2 + 2 = 0 →
  a 16 ^ 2 + 6 * a 16 + 2 = 0 →
  (a 2 * a 16 / a 9 = Real.sqrt 2 ∨ a 2 * a 16 / a 9 = -Real.sqrt 2) :=
by sorry

end geometric_sequence_problem_l2923_292306
