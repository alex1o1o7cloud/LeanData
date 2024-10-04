import Mathlib

namespace area_of_right_triangle_l143_143767

def point_intersection (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = g x

def is_right_triangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3 in
    (x1 = x2 ∧ y2 = y3) ∨ (y1 = y2 ∧ x2 = x3)

def base_length : ℝ := 5
def height_length : ℝ := 5

theorem area_of_right_triangle : 
  let p1 := (-5, -5)
  let p2 := (-5, 0)
  let p3 := (0, 0)
  is_right_triangle p1 p2 p3 ∧ point_intersection (λ x, x) (λ x, -5) (-5) →
  (1 / 2 : ℝ) * base_length * height_length = 12.5 := 
by
  sorry

end area_of_right_triangle_l143_143767


namespace max_value_of_func_l143_143948

open Real

noncomputable def func (x : ℝ) : ℝ :=
  tan (x + 2/3 * π) - tan (x + π / 6) + cos (x + π / 6)

theorem max_value_of_func: 
  ∃ (x : ℝ), x ∈ Icc (-(5/12) * π) (-(π/3)) ∧ 
    func x = 11/6 * √3 :=
by
  sorry

end max_value_of_func_l143_143948


namespace total_marbles_is_correct_l143_143522

variables (r : ℝ) (b g : ℝ)

def blue_marbles (r : ℝ) := r / 1.30
def green_marbles (r : ℝ) := 1.50 * r
def total_marbles (r : ℝ) := r + (blue_marbles r) + (green_marbles r)

theorem total_marbles_is_correct (r : ℝ) :
  total_marbles r = (85 / 26) * r := 
sorry

end total_marbles_is_correct_l143_143522


namespace side_length_of_square_l143_143333

theorem side_length_of_square (A : ℝ) (π : ℝ) (h1 : A = 78.53981633974483) (h2 : π = 3.141592653589793) :
  let r := (A / π).sqrt in
  let diameter := 2 * r in
  diameter = 10 :=
by sorry

end side_length_of_square_l143_143333


namespace min_groups_l143_143847

def twin_siblings := ({A1, A2}, {B1, B2}, {C1, C2}, {D1, D2}, {E1, E2} : set (Set ℕ))

def separates_twins (groups: List (Set ℕ)) := ∀ {x y}, (x ∈ y) → ∃ group, group ∈ groups ∧ x ∈ group ∧ y ∉ group

def one_common_group (groups: List (Set ℕ)) := ∀ {x y}, x ≠ y → ∃! group, group ∈ groups ∧ x ∈ group ∧ y ∈ group

def involved_in_two_groups (groups: List (Set ℕ)) := ∃ x, (∃ group1 ∈ groups, x ∈ group1) ∧ (∃ group2 ∈ groups, x ∈ group2) ∧ (group1 ≠ group2)

theorem min_groups (k : ℕ):
  separates_twins groups → one_common_group groups → involved_in_two_groups groups → k = 14 :=
by
  sorry

end min_groups_l143_143847


namespace compensation_problem_l143_143968

namespace CompensationProof

variables (a b c : ℝ)

def geometric_seq_with_ratio_1_by_2 (a b c : ℝ) : Prop :=
  c = (1/2) * b ∧ b = (1/2) * a

def total_compensation_eq (a b c : ℝ) : Prop :=
  4 * c + 2 * b + a = 50

theorem compensation_problem :
  total_compensation_eq a b c ∧ geometric_seq_with_ratio_1_by_2 a b c → c = 50 / 7 :=
sorry

end CompensationProof

end compensation_problem_l143_143968


namespace find_r_p_q_l143_143892

theorem find_r_p_q (p q r : ℕ) (x : ℝ) (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r) (h3 : Nat.coprime p q) :
  r + p + q = 5 :=
sorry

end find_r_p_q_l143_143892


namespace rationalize_denominator_l143_143688

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143688


namespace find_principal_l143_143354

noncomputable def principal_amount (A: ℝ) (P: ℝ) (r: ℝ) (n: ℕ) : Prop :=
  A = P * (1 + r)^n

theorem find_principal :
  ∃ P : ℝ, 
    (principal_amount 8000 P 0.157625 2) ∧ 
    (principal_amount 9261 P 0.157625 3) :=
begin
  use 5967.79,
  split,
  {
    rw [principal_amount],
    sorry
  },
  {
    rw [principal_amount],
    sorry
  }
end

end find_principal_l143_143354


namespace cross_section_circle_example_l143_143750

-- Definitions of geometric figures
inductive GeomFigure
| cone
| cylinder

-- Definition of cross-section being a circle
def is_circle (c : GeomFigure) : Prop :=
  match c with
  | GeomFigure.cone => true -- Simplified representation for the condition
  | GeomFigure.cylinder => true

-- Theorem to prove the intersection results in a circular cross-section
theorem cross_section_circle_example (c1 c2 : GeomFigure) (h1 : is_circle c1) (h2 : is_circle c2) : 
  (c1 = GeomFigure.cone ∨ c1 = GeomFigure.cylinder) ∧ (c2 = GeomFigure.cone ∨ c2 = GeomFigure.cylinder) :=
by
  exact ⟨or.inl rfl, or.inr rfl⟩

end cross_section_circle_example_l143_143750


namespace fixed_points_quadratic1_min_fixed_point_ratio_range_of_a_l143_143850

-- Definition of a fixed point for a quadratic function
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- (1) Prove that the quadratic function y = x^2 - x - 3 has fixed points at x = -1 and x = 3
theorem fixed_points_quadratic1 :
  {x : ℝ | is_fixed_point (λ x, x^2 - x - 3) x} = {-1, 3} :=
by sorry

-- (2) Prove that for the quadratic function y = 2x^2 - (3 + a)x + a - 1 
-- having two distinct positive fixed points x1 and x2, the minimum value of 
-- (x1 / x2) + (x2 / x1) is 8
theorem min_fixed_point_ratio (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ is_fixed_point (λ x, 2*x^2 - (3 + a)*x + a - 1) x1 ∧ is_fixed_point (λ x, 2*x^2 - (3 + a)*x + a - 1) x2) →
  4 * a - 7 ≥ 0 ∧ (4 * a - 7 < 7 ∨ (4 * a - 1) ≤ 68) :=
by sorry

-- (3) Prove that for the quadratic function y = ax^2 + (b + 1)x + (b - 1)
-- always having a fixed point for any real number b, the range of a is (0, 1]
theorem range_of_a (a : ℝ) :
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point (λ x, a*x^2 + (b + 1)*x + (b - 1)) x) ↔ (0 < a ∧ a ≤ 1) :=
by sorry

end fixed_points_quadratic1_min_fixed_point_ratio_range_of_a_l143_143850


namespace simplify_expr_l143_143002

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) :
  (3/4) * (8/(x^2) + 12*x - 5) = 6/(x^2) + 9*x - 15/4 := by
  sorry

end simplify_expr_l143_143002


namespace no_unique_subset_sum_l143_143853

open Set

def M : Set ℚ := { x : ℚ | 0 < x ∧ x < 1 }

theorem no_unique_subset_sum {A : Set ℚ} (hA : A ⊆ M) :
  ¬ ∀ x ∈ M, ∃ (a : List ℚ), (∀ x ∈ a, x ∈ A ∧ List.Pairwise (≠) a) ∧ List.sum a = x := 
sorry

end no_unique_subset_sum_l143_143853


namespace exists_root_in_interval_l143_143806

noncomputable def f (x : ℝ) : ℝ := real.exp x + 4 * x - 3

theorem exists_root_in_interval : ∃ x ∈ set.Ioo 0 (1 / 2), f x = 0 := by
  sorry

end exists_root_in_interval_l143_143806


namespace equal_perimeters_of_triangles_l143_143782

-- Define the setup of the circles and triangle
variables {C A B : Point}
variables {ω Γ : Circle}
variables [∀ ω, isInscribedAtAngle ω C]
variables [∀ Γ, passesThrough Γ C ∧ touchesExternally Γ ω ∧ meetsAtSides Γ C A B]

-- Define the theorem
theorem equal_perimeters_of_triangles :
  ∀ (ω : Circle) (Γ : Circle) (C A B : Point),
    isInscribedAtAngle ω C →
    passesThrough Γ C →
    touchesExternally Γ ω →
    meetsAtSides Γ C A B →
    perimeter (triangle ABC) = perimeter (triangle ABC) :=
sorry

end equal_perimeters_of_triangles_l143_143782


namespace carla_gas_cost_l143_143380

theorem carla_gas_cost:
  let distance_grocery := 8
  let distance_school := 6
  let distance_bank := 12
  let distance_practice := 9
  let distance_dinner := 15
  let distance_home := 2 * distance_practice
  let total_distance := distance_grocery + distance_school + distance_bank + distance_practice + distance_dinner + distance_home
  let miles_per_gallon := 25
  let price_per_gallon_first := 2.35
  let price_per_gallon_second := 2.65
  let total_gallons := total_distance / miles_per_gallon
  let gallons_per_fill_up := total_gallons / 2
  let cost_first := gallons_per_fill_up * price_per_gallon_first
  let cost_second := gallons_per_fill_up * price_per_gallon_second
  let total_cost := cost_first + cost_second
  total_cost = 6.80 :=
by sorry

end carla_gas_cost_l143_143380


namespace cube_surface_area_given_sphere_volume_l143_143862

noncomputable def surface_area_cube (V : ℝ) : ℝ :=
  let R := ((3 * V) / (4 * π))^(1/3)
  let a := R * (2/√3)
  6 * a^2

theorem cube_surface_area_given_sphere_volume :
  surface_area_cube (9 * π / 2) = 18 :=
by
  sorry

end cube_surface_area_given_sphere_volume_l143_143862


namespace fliers_left_next_day_l143_143307

def total_fliers : Nat := 10000
def morning_fraction : ℚ := 1 / 5
def afternoon_fraction : ℚ := 1 / 4
def evening_fraction : ℚ := 1 / 3

def fliers_remaining (total : Nat) (morning_frac afternoon_frac evening_frac : ℚ) : Nat :=
  let morning_sent := morning_frac * total
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := afternoon_frac * remaining_after_morning.natAbs
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent
  let evening_sent := evening_frac * remaining_after_afternoon.natAbs
  remaining_after_afternoon - evening_sent

theorem fliers_left_next_day : fliers_remaining total_fliers morning_fraction afternoon_fraction evening_fraction = 4000 := sorry

end fliers_left_next_day_l143_143307


namespace isosceles_right_triangle_perimeter_area_l143_143952

theorem isosceles_right_triangle_perimeter_area 
    (A B C : Type*) [metric_space A] [metric_space B] [metric_space C]
    (H : metric_space.angle A B C = π / 2)
    (h_isosceles : isosceles_triangle A B C)
    (h_AB : metric_space.dist A B = 10) :
    let x := metric_space.dist A C in
    let y := metric_space.dist B C in
    x = y ∧
    2 * x ^ 2 = 100 ∧
    (metric_space.dist A B + metric_space.dist A C + metric_space.dist B C = 10 + 10 * real.sqrt 2) ∧
    ((metric_space.dist A C * metric_space.dist B C / 2) = 25) :=
by sorry

end isosceles_right_triangle_perimeter_area_l143_143952


namespace monotonic_intervals_a_range_l143_143898

section
variables {x a : ℝ} (f g : ℝ → ℝ)

-- Define the functions f(x) and g(x)
def f (x : ℝ) : ℝ := a * x^2 - log x
def g (x : ℝ) : ℝ := exp x - a * x

-- Conditions
axiom ha : a ∈ ℝ
axiom hfg : ∀ x > 0, f x * g x > 0

theorem monotonic_intervals (hp : deriv f 1 > -1) :
  ∀ (a > 0), 
    ((∀ x > sqrt (1 / (2 * a)), deriv f x > 0) 
    ∧ (∀ x, 0 < x ∧ x < sqrt (1 / (2 * a)) → deriv f x < 0)) := sorry

theorem a_range : (1 / (2 * exp 1) < a) ∧ (a < exp 1) := sorry

end

end monotonic_intervals_a_range_l143_143898


namespace minimum_value_a_plus_b_plus_c_l143_143442

theorem minimum_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 4 * b + 7 * c ≤ 2 * a * b * c) : a + b + c ≥ 15 / 2 :=
by
  sorry

end minimum_value_a_plus_b_plus_c_l143_143442


namespace soybeans_to_oil_kg_l143_143771

-- Define initial data
def kgSoybeansToTofu : ℕ := 3
def kgSoybeansToOil : ℕ := 6
def kgTofuCostPerKg : ℕ := 3
def kgOilCostPerKg : ℕ := 15
def batchSoybeansKg : ℕ := 460
def totalRevenue : ℕ := 1800

-- Define problem statement
theorem soybeans_to_oil_kg (x y : ℕ) (h : x + y = batchSoybeansKg) 
  (hRevenue : 3 * kgTofuCostPerKg * x + (kgOilCostPerKg * y) / (kgSoybeansToOil) = totalRevenue) : 
  y = 360 :=
sorry

end soybeans_to_oil_kg_l143_143771


namespace candies_taken_by_boys_invariant_l143_143531

theorem candies_taken_by_boys_invariant (k : ℕ) (total_candies : ℕ) 
    (is_boy : Fin k.succ -> Bool) : 
    total_candies = 1000 -> 
    ∀ queue1 queue2 : List (Fin k.succ),
    Permutation queue1 queue2 ->
    (let candies_taken_by_boys := 
        queue1.foldl (λ C ch =>
            let taken := if is_boy ch then (Int.ceil (C / (k - queue1.indexOf ch + 1) : ℝ)) else (Int.floor (C / (k - queue1.indexOf ch + 1) : ℝ)) in
            C - taken) 1000 in
     let boys_in_queue := queue1.filter (λ ch => is_boy ch) in
     boys_in_queue.foldl (λ acc ch => acc + (if is_boy ch then (Int.ceil (total_candies / (k - queue1.indexOf ch + 1) : ℝ)) else (Int.floor (total_candies / (k - queue1.indexOf ch + 1) : ℝ)))) 0
    )
    = 
    (let candies_taken_by_boys := 
        queue2.foldl (λ C ch =>
            let taken := if is_boy ch then (Int.ceil (C / (k - queue2.indexOf ch + 1) : ℝ)) else (Int.floor (C / (k - queue2.indexOf ch + 1) : ℝ)) in
            C - taken) 1000 in
     let boys_in_queue := queue2.filter (λ ch => is_boy ch) in
     boys_in_queue.foldl (λ acc ch => acc + (if is_boy ch then (Int.ceil (total_candies / (k - queue2.indexOf ch + 1) : ℝ)) else (Int.floor (total_candies / (k - queue2.indexOf ch + 1) : ℝ)))) 0
    ) := sorry

end candies_taken_by_boys_invariant_l143_143531


namespace arith_seq_and_sum_l143_143438

-- Given conditions
def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def S_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
∑ i in range(n + 1), a i

noncomputable def a_seq := λ n : ℕ, 2 * n - 1

def b_seq (a : ℕ → ℤ) (n : ℕ) : ℝ :=
1 / ((a n) * (a (n + 1)))

def T_n (a : ℕ → ℤ) (n : ℕ) : ℝ :=
∑ i in range(n + 1), b_seq a i

-- The main theorem we want to prove
theorem arith_seq_and_sum 
  (a : ℕ → ℤ) (S_9 : S_n a 8 = 81)
  (a_3_5 : a 2 + a 4 = 14) : 
  is_arithmetic_seq a ∧ a = a_seq ∧ T_n a_seq n = n / (2n + 1) :=
by
  sorry

end arith_seq_and_sum_l143_143438


namespace measure_angle_BAG_l143_143528

variable (A B C D E F G : Type) [DecidableEq α]

-- condition 1: ABCDEF is a regular hexagon
def regular_hexagon (A B C D E F : Type) : Prop := 
  ∀ (x y : Type), x ∈ {A, B, C, D, E, F} → y ∈ {A, B, C, D, E, F} 
  → (x ≠ y  → ∃ (angle: ℝ), angle = 120 / 180 * Math.pi )

-- condition 2: each interior angle of the regular hexagon measures 120 degrees
def each_interior_angle_120_deg (A B C D E F : Type) : Prop := 
  ∀ (a b c : Type), (a, b, c ∈ {A, B, C, D, E, F}) → 
  (∠ a b c = 120 / 180 * Math.pi) 

-- condition 3: diagonals AC and BD intersect at point G
def diagonals_intersect_G (A B C D E F G : Type) : Prop :=
  ∃ G, (G = AC ∩ BD)

-- statement to prove: the measure of angle BAG == 30 degrees
theorem measure_angle_BAG (A B C D E F : Type) (G : Type)
  (h1 : regular_hexagon A B C D E F) 
  (h2 : each_interior_angle_120_deg A B C D E F)
  (h3 : diagonals_intersect_G A B C D E F G) : 
  (∠ B A G = 30 / 180 * Math.pi) :=
sorry

end measure_angle_BAG_l143_143528


namespace sum_of_divisors_of_24_l143_143286

theorem sum_of_divisors_of_24 : ∑ d in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), d = 60 := by
  sorry

end sum_of_divisors_of_24_l143_143286


namespace arithmetic_sequence_first_term_l143_143995

-- Define the sum of first n terms T_n of the arithmetic sequence
def T (n : ℕ) (b : ℝ) : ℝ := n * (2 * b + (n - 1) * 5) / 2

-- The main theorem we need to prove
theorem arithmetic_sequence_first_term (b : ℝ) (k : ℝ)
  (h : ∀ n : ℕ, n > 0 → T (3 * n) b / T n b = k) : b = 5 / 2 :=
by
  sorry

end arithmetic_sequence_first_term_l143_143995


namespace b_is_arithmetic_general_formula_a_sum_formula_l143_143999

/-- Defining the sequences a_n and b_n respectively -/
def a_seq : ℕ → ℕ
| 0       := 4
| (n + 1) := 2 * a_seq n + 2^(n + 1)

def b_seq (n : ℕ) : ℕ := a_seq n / 2^n

/-- Problem 1: Prove that b_seq is an arithmetic sequence -/
theorem b_is_arithmetic :
  ∀ n : ℕ, b_seq (n + 1) = b_seq n + 1 := by
  sorry

/-- Problem 2a: Find the general formula for a_seq -/
theorem general_formula_a :
  ∀ n : ℕ, a_seq n = (n + 1) * 2^n := by
  sorry

/-- Problem 2b: Prove the formula for the sum of the first n terms of a_seq -/
def sum_a (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a_seq (i + 1))

theorem sum_formula :
  ∀ n : ℕ, sum_a n = n * 2^(n + 1) := by
  sorry

end b_is_arithmetic_general_formula_a_sum_formula_l143_143999


namespace not_perfect_square_l143_143941

theorem not_perfect_square
  (a b c d : ℤ)
  (A : ℤ := 2 * (a - 2 * b + c)^4 + 2 * (b - 2 * c + a)^4 + 2 * (c - 2 * a + b)^4)
  (B : ℤ := d * (d + 1) * (d + 2) * (d + 3) + 1) :
  ¬ ∃ k : ℤ, (√A + 1) ^ 2 + B = k ^ 2 :=
sorry

end not_perfect_square_l143_143941


namespace number_of_terms_in_arithmetic_sequence_l143_143938

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (-5 + (k - 1) * 5 = 65)) := by
  sorry

end number_of_terms_in_arithmetic_sequence_l143_143938


namespace inequality_abc_l143_143560

theorem inequality_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1/a + 1/(b * c)) * (1/b + 1/(c * a)) * (1/c + 1/(a * b)) ≥ 1728 :=
by sorry

end inequality_abc_l143_143560


namespace olympic_pair_count_l143_143344

def is_olympic_pair (m n : ℕ) : Prop :=
  ∀ k, (m % 10^(k+1) // 10^k + n % 10^(k+1) // 10^k) < 10

theorem olympic_pair_count : 
  (finset.filter (λ p : ℕ × ℕ, p.1 + p.2 = 2008 ∧ is_olympic_pair p.1 p.2) ((finset.Icc 0 2008).product (finset.Icc 0 2008))).card = 27 := 
sorry

end olympic_pair_count_l143_143344


namespace number_of_pairs_l143_143991

theorem number_of_pairs (n : ℕ) (hn : 0 < n) :
  let count := 2 * n^3 + 7 * n^2 + 7 * n + 3 in
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^2 - y^2 = 10^2 * 30^(2 * n)) ∧ ¬ (∃ k : ℕ, k^2 = count) :=
begin
  let count := 2 * n^3 + 7 * n^2 + 7 * n + 3,
  have := sorry,
end

end number_of_pairs_l143_143991


namespace car_mileage_l143_143328

theorem car_mileage (distance : ℕ) (gallons : ℕ) (mileage : ℕ) 
  (h1 : distance = 120) 
  (h2 : gallons = 3) :
  mileage = distance / gallons := by
  sorry

example : car_mileage 120 3 40 := by
  apply car_mileage
  rfl
  rfl

end car_mileage_l143_143328


namespace correct_calculation_l143_143326

theorem correct_calculation (x : ℕ) (h : 637 = x + 238) : x - 382 = 17 :=
by
  sorry

end correct_calculation_l143_143326


namespace shapes_fit_exactly_l143_143716

-- Conditions: Shapes are drawn on a piece of paper and folded along a central bold line
def shapes_drawn_on_paper := true
def paper_folded_along_central_line := true

-- Define the main proof problem
theorem shapes_fit_exactly : shapes_drawn_on_paper ∧ paper_folded_along_central_line → 
  number_of_shapes_fitting_exactly_on_top = 3 :=
by
  intros h
  sorry

end shapes_fit_exactly_l143_143716


namespace range_of_expression_l143_143993

noncomputable def expression (a b c d : ℝ) : ℝ :=
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + 
  Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)

theorem range_of_expression (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2)
  (h3 : 0 ≤ b) (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2)
  (h7 : 0 ≤ d) (h8 : d ≤ 2) :
  4 * Real.sqrt 2 ≤ expression a b c d ∧ expression a b c d ≤ 16 :=
by
  sorry

end range_of_expression_l143_143993


namespace tom_made_money_correct_l143_143259

-- Define constants for flour, salt, promotion cost, ticket price, and tickets sold
def flour_needed : ℕ := 500
def flour_bag_size : ℕ := 50
def flour_bag_cost : ℕ := 20
def salt_needed : ℕ := 10
def salt_cost_per_pound : ℚ := 0.2
def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 500

-- Compute how much money Tom made
def money_made : ℤ :=
  let flour_bags := flour_needed / flour_bag_size
  let total_flour_cost := flour_bags * flour_bag_cost
  let total_salt_cost := salt_needed * salt_cost_per_pound
  let total_cost := total_flour_cost + total_salt_cost + promotion_cost
  let total_revenue := tickets_sold * ticket_price
  total_revenue - total_cost

-- The theorem statement
theorem tom_made_money_correct :
  money_made = 8798 := by
  sorry

end tom_made_money_correct_l143_143259


namespace min_value_of_a_l143_143579

variables (a b c d : ℕ)

-- Conditions
def conditions : Prop :=
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2004 ∧
  a^2 - b^2 + c^2 - d^2 = 2004

-- Theorem: minimum value of a
theorem min_value_of_a (h : conditions a b c d) : a = 503 :=
sorry

end min_value_of_a_l143_143579


namespace largest_whole_number_n_satisfying_inequality_l143_143273

theorem largest_whole_number_n_satisfying_inequality :
  ∃ (n : ℕ), (∀ (m : ℕ), (frac1 := 1 / 4) (frac2 := n / 8) (frac3 := 1 / 8)
  frac1 + frac2 + frac3 < 1 → n ≤ m) ∧ n = 4 := 
by
  sorry

end largest_whole_number_n_satisfying_inequality_l143_143273


namespace lemonade_price_l143_143210

-- Defining the conditions
def total_days := 10
def hot_days := 4
def regular_days := total_days - hot_days
def sold_cups_each_day := 32
def total_profit := 200
def hot_day_price_increase := 1.25
def revenue_regular_day (P : ℚ) := sold_cups_each_day * P
def revenue_hot_day (P : ℚ) := sold_cups_each_day * hot_day_price_increase * P
def total_revenue (P : ℚ) := regular_days * revenue_regular_day P + hot_days * revenue_hot_day P

-- Proving that the cost of 1 cup of lemonade is 25/44 dollars
theorem lemonade_price : 
  (∃ P : ℚ, total_revenue P = 200) → (∃ P : ℚ, P = 25 / 44) :=
by
  -- Skipping the complete proof
  sorry

end lemonade_price_l143_143210


namespace barycenter_of_triangle_l143_143440

-- Define the problem data and conditions
variables {A B C G O A₁ B₁ C₁ : Type} [nonempty (triangle A B C)] 

-- Definitions related to barycenter, circumcenter, and perpendicular bisectors
def isCircumcenter (O : Type) (A B C : Type) : Prop := sorry
def isBarycenter (G : Type) (A B C : Type) : Prop := sorry
def perpendicularBisector (P Q : Type) : Type := sorry
def intersection (l₁ l₂ : Type) : Type := sorry
def centroid (P Q R : Type) : Type := sorry

-- Problem Statement
theorem barycenter_of_triangle (hO : isCircumcenter O A B C) 
                              (hG : isBarycenter G A B C)
                              (hA₁ : A₁ = intersection (perpendicularBisector G A) (perpendicularBisector G B))
                              (hB₁ : B₁ = intersection (perpendicularBisector G B) (perpendicularBisector G C))
                              (hC₁ : C₁ = intersection (perpendicularBisector G C) (perpendicularBisector G A)) : 
    centroid O A₁ B₁ C₁ := sorry

end barycenter_of_triangle_l143_143440


namespace union_A_B_complement_A_intersect_B_l143_143474

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def U_def : U = {x | -2 < x ∧ x < 12} := 
begin
  sorry
end

def A_def : A = {x | 3 ≤ x ∧ x < 7} := 
begin
  sorry
end

def B_def : B = {x | 2 < x ∧ x < 10} := 
begin
  sorry
end

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
begin
  sorry
end

theorem complement_A_intersect_B : (U \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
begin
  sorry
end

end union_A_B_complement_A_intersect_B_l143_143474


namespace tobias_downloads_l143_143741

theorem tobias_downloads : 
  ∀ (m : ℕ), (∀ (price_per_app total_spent : ℝ), 
  price_per_app = 2.00 + 2.00 * 0.10 ∧ 
  total_spent = 52.80 → 
  m = total_spent / price_per_app) → 
  m = 24 := 
  sorry

end tobias_downloads_l143_143741


namespace principal_amount_is_1200_l143_143227

-- Define the given conditions
def simple_interest (P : ℝ) : ℝ := 0.10 * P
def compound_interest (P : ℝ) : ℝ := 0.1025 * P

-- Define given difference
def interest_difference (P : ℝ) : ℝ := compound_interest P - simple_interest P

-- The main goal is to prove that the principal amount P that satisfies the difference condition is 1200
theorem principal_amount_is_1200 : ∃ P : ℝ, interest_difference P = 3 ∧ P = 1200 :=
by
  sorry -- Proof to be completed

end principal_amount_is_1200_l143_143227


namespace events_A_B_equal_prob_l143_143513

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l143_143513


namespace coefficient_a6_in_expansion_l143_143060

theorem coefficient_a6_in_expansion:
  ∀ (a : ℤ), 
    (x + 1) ^ 10 = ∑ i in (finset.range 11), 
      (a_i * (x - 1) ^ i) → 
    a_6 = (nat.choose 10 4) * 2 ^ 4 := 
by sorry

end coefficient_a6_in_expansion_l143_143060


namespace largest_n_permutation_divisible_l143_143054

theorem largest_n_permutation_divisible :
  ∃ n : ℕ, (∀ (s : fin n.succ -> Nat), 
            (∀ i j : fin n.succ, i ≠ j → ∃ (d : list ℕ), 
              s i = list.foldr (λ x y, x + 10 * y) 0 d ∧ 
              s j = list.foldr (λ x y, x + 10 * y) 0 d.reverse) →
            ∃ x : Nat, 
              (s 0 = x ∧ ∀ k : fin n.succ, x ∣ s k)) ∧ 
          (∀ m : ℕ, (m > n → ¬(∀ (s : fin m.succ -> Nat), 
            (∀ i j : fin m.succ, i ≠ j → ∃ (d : list ℕ), 
              s i = list.foldr (λ x y, x + 10 * y) 0 d ∧ 
              s j = list.foldr (λ x y, x + 10 * y) 0 d.reverse) →
            ∃ x : Nat, 
              (s 0 = x ∧ ∀ k : fin m.succ, x ∣ s k)))) ∧ n = 7 :=
begin
  sorry
end

end largest_n_permutation_divisible_l143_143054


namespace distance_from_midpoints_l143_143974

variables {A B C M O K L : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space O] [metric_space K] [metric_space L]
variables [inhabited A] [inhabited B] [inhabited C] [inhabited M] [inhabited O] [inhabited K] [inhabited L]
variables {triangle : Type}

structure centroid (M : triangle → Prop) :=
  (is_centroid : ∀ (Δ : triangle), M Δ → ∃ (G : A), G)

structure incenter (O : triangle → Prop) :=
  (is_incenter : ∀ (Δ : triangle), O Δ → ∃ (I : B), I)

structure midpoint (K L : triangle → Prop) :=
  (is_midpoint : ∀ (Δ : triangle), K Δ → L Δ → ∃ (MPK MPL : C), MPK MPL)

def parallel (line1 line2 : Prop) : Prop := sorry

def distance (p q : Type) : Type := sorry

theorem distance_from_midpoints {A B C M O K L : Type}
  (hM : centroid (λ Δ, Δ = Δ))
  (hO : incenter (λ Δ, Δ = Δ))
  (hK : midpoint (λ Δ, Δ = Δ))
  (hL : midpoint (λ Δ, Δ = Δ))
  (h_parallel : parallel (λ Δ, OM) (λ Δ, BC))
  : distance O K = distance O L :=
sorry

end distance_from_midpoints_l143_143974


namespace rationalize_sqrt_5_over_12_l143_143641

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143641


namespace original_acid_percentage_l143_143756

theorem original_acid_percentage (a w : ℕ) (h1 : a / (a + w + 2) = 0.25) (h2 : (a + 2) / (a + w + 4) = 0.40) : 
  a / (a + w) * 100 = 33.33 := 
sorry

end original_acid_percentage_l143_143756


namespace rationalize_sqrt_fraction_l143_143616

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143616


namespace product_of_first_2012_terms_l143_143929

noncomputable def a : ℕ → ℚ
| 0       := 2
| (n + 1) := (1 + a n) / (1 - a n)

theorem product_of_first_2012_terms : (∏ n in finset.range 2012, a n) = 1 := 
sorry

end product_of_first_2012_terms_l143_143929


namespace sin_C_max_value_l143_143055

noncomputable def sin_angle_C_max (a b c : ℝ)
  (D_midpoint : ∀(A C D : ℝ × ℝ × ℝ), D = midpoint A C)
  (dot_product_eq : let BD := vector_sum (vector_scale 1/2 (vector_sub BD AC)) 
                     in dot_product BD AC = 25/2) : ℝ :=
  sin (angle_of_triangle a b c max_C)

-- Statement to prove
theorem sin_C_max_value
  (a b c : ℝ)
  (D_midpoint : ∀(A C D : ℝ × ℝ × ℝ), D = midpoint A C)
  (dot_product_eq : let BD := vector_sum (vector_scale 1/2 (vector_sub BD AC)) 
                      in dot_product BD AC = 25/2)
  (a_eq : a = 7) :
  sin_angle_C_max a b c D_midpoint dot_product_eq = 2 * sqrt 6 / 7 :=
by sorry

end sin_C_max_value_l143_143055


namespace general_term_a_sum_S_19_l143_143437

variable (a : ℕ → ℚ) (S : ℕ → ℚ) (b : ℕ → ℚ) (S_b : ℕ → ℚ)

-- Conditions
axiom a_2_plus_a_6_eq_6 : a 2 + a 6 = 6
axiom S_5_eq_35_div_3 : S 5 = 35 / 3

-- Given conditions on b_n
axiom b1_eq_3 : b 1 = 3
axiom b_def : ∀ n : ℕ, n ≥ 2 → b n = 1 / (a (n - 1) * a n)

-- Sum definitions
axiom S_def : ∀ n : ℕ, S n = ∑ i in Finset.range n, a i
axiom S_b_def : ∀ n : ℕ, S_b n = ∑ i in Finset.range n, b i

-- Statements to prove
theorem general_term_a :
  (∀ n : ℕ, a n = (2 * n + 1) / 3) :=
sorry

theorem sum_S_19 : 
  S_b 19 = 323 / 38 :=
sorry

end general_term_a_sum_S_19_l143_143437


namespace imaginary_part_of_z_l143_143044

theorem imaginary_part_of_z (z : ℂ) (h : z * (complex.I + 1) + complex.I = 1 + 3 * complex.I) : complex.im z = 1 / 2 := by
  sorry

end imaginary_part_of_z_l143_143044


namespace max_area_cross_section_of_prism_l143_143350

noncomputable def prism_vertex_A : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def prism_vertex_B : ℝ × ℝ × ℝ := (-3, 0, 0)
noncomputable def prism_vertex_C : ℝ × ℝ × ℝ := (0, 3 * Real.sqrt 3, 0)
noncomputable def plane_eq (x y z : ℝ) : ℝ := 2 * x - 3 * y + 6 * z

-- Statement
theorem max_area_cross_section_of_prism (h : ℝ) (A B C : ℝ × ℝ × ℝ)
  (plane : ℝ → ℝ → ℝ → ℝ) (cond_h : h = 5)
  (cond_A : A = prism_vertex_A) (cond_B : B = prism_vertex_B) 
  (cond_C : C = prism_vertex_C) (cond_plane : ∀ x y z, plane x y z = 2 * x - 3 * y + 6 * z - 30) : 
  ∃ cross_section : ℝ, cross_section = 0 :=
by
  sorry

end max_area_cross_section_of_prism_l143_143350


namespace largest_digit_for_5178M_divisible_by_six_l143_143272

theorem largest_digit_for_5178M_divisible_by_six : 
  ∃ M : ℕ, (∃ M : ℕ, 0 ≤ M ∧ M < 10 ∧ even M ∧ (21 + M) % 3 = 0) ∧ ∀ M' : ℕ, (0 ≤ M' ∧ M' < 10 ∧ even M' ∧ (21 + M') % 3 = 0) → M' ≤ M :=
begin
  use 6,
  split,
  { existsi 6,
    repeat {split},
    repeat {linarith},
    { exact even_bit0 3 },
    { norm_num, }},
  { intros M' hM',
    cases hM' with hM'1 hM'',
    cases hM'' with hM'2 hM''',
    cases hM''' with hM'3 hM'4,
    transitivity,
    { exact hM'2 },
    { linarith, }},
end

end largest_digit_for_5178M_divisible_by_six_l143_143272


namespace angle_ADB_is_45_degrees_l143_143971

open Real EuclideanGeometry

def convex_pentagon (A B C D E : Point) : Prop :=
  ConvexPolygon A B C D E ∧
  ∠A B C = 90 ∧
  ∠B C D = 90 ∧
  ∠D A E = 90 ∧
  Inscribable A B C D E

theorem angle_ADB_is_45_degrees 
{A B C D E : Point} 
(h_convex : convex_pentagon A B C D E) 
: ∠A D B = 45 := 
sorry

end angle_ADB_is_45_degrees_l143_143971


namespace smallest_number_l143_143364

theorem smallest_number (a b c d : ℚ) (h₀ : a = 0) (h₁ : b = -3) (h₂ : c = 1/3) (h₃ : d = 1) :
  min (min a (min b c)) d = -3 := 
by 
  rw [h₀, h₁, h₂, h₃]
  norm_num
  sorry

end smallest_number_l143_143364


namespace rationalize_sqrt_5_over_12_l143_143645

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143645


namespace valid_numbers_eq_l143_143008

-- Definition of the number representation
def is_valid_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999 ∧
  ∃ (a b c : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    x = 100 * a + 10 * b + c ∧
    x = a^3 + b^3 + c^3

-- The theorem to prove
theorem valid_numbers_eq : 
  {x : ℕ | is_valid_number x} = {153, 407} :=
by
  sorry

end valid_numbers_eq_l143_143008


namespace radian_measure_sector_l143_143105

theorem radian_measure_sector (r l : ℝ) (h1 : 2 * r + l = 12) (h2 : (1 / 2) * l * r = 8) :
  l / r = 1 ∨ l / r = 4 := by
  sorry

end radian_measure_sector_l143_143105


namespace counting_nines_in_winter_l143_143731

def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month = 12 then 31
  else if month = 1 then 31
  else if month = 2 then 28
  else 0

def total_days (dec_21 : ℕ) (jan : ℕ) (feb_10 : ℕ) : ℕ :=
  dec_21 + jan + feb_10

def nines_period (total_days : ℕ) : ℕ × ℕ :=
  (total_days / 9, total_days % 9)

theorem counting_nines_in_winter :
  let dec_21 := 31 - 21 + 1 in
  let jan := 31 in
  let feb_10 := 10 in
  total_days dec_21 jan feb_10 = 52 →
  nines_period 52 = (6, 7) :=
by {
  sorry
}

end counting_nines_in_winter_l143_143731


namespace collinear_A₁_F_B_iff_q_eq_4_l143_143468

open Real

theorem collinear_A₁_F_B_iff_q_eq_4
  (m q : ℝ) (h_m : m ≠ 0)
  (A B : ℝ × ℝ)
  (h_A : 3 * (m * A.snd + q)^2 + 4 * A.snd^2 = 12)
  (h_B : 3 * (m * B.snd + q)^2 + 4 * B.snd^2 = 12)
  (A₁ : ℝ × ℝ := (A.fst, -A.snd))
  (F : ℝ × ℝ := (1, 0)) :
  ((q = 4) ↔ (∃ k : ℝ, k * (F.fst - A₁.fst) = F.snd - A₁.snd ∧ k * (B.fst - F.fst) = B.snd - F.snd)) :=
sorry

end collinear_A₁_F_B_iff_q_eq_4_l143_143468


namespace quadratic_equation_m_value_l143_143583

-- Definition of the quadratic equation having exactly one solution with the given parameters
def quadratic_equation_has_one_solution (a b c : ℚ) : Prop :=
  b^2 - 4 * a * c = 0

-- Given constants in the problem
def a : ℚ := 3
def b : ℚ := -7

-- The value of m we aim to prove
def m_correct : ℚ := 49 / 12

-- The theorem stating the problem
theorem quadratic_equation_m_value (m : ℚ) (h : quadratic_equation_has_one_solution a b m) : m = m_correct :=
  sorry

end quadratic_equation_m_value_l143_143583


namespace solution_unique_for_alpha_neg_one_l143_143562

noncomputable def alpha : ℝ := sorry

axiom alpha_nonzero : alpha ≠ 0

def functional_eqn (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (x + y)) = f (x + y) + f (x) * f (y) + alpha * x * y

theorem solution_unique_for_alpha_neg_one (f : ℝ → ℝ) :
  (alpha = -1 → (∀ x : ℝ, f x = x)) ∧ (alpha ≠ -1 → ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, functional_eqn f x y) :=
sorry

end solution_unique_for_alpha_neg_one_l143_143562


namespace α_conjugateβ_α_minus_β_l143_143174

open Complex

noncomputable def α (a b : ℝ) : ℂ := ⟨a, b⟩
noncomputable def β (a b : ℝ) : ℂ := ⟨a, -b⟩

theorem α_conjugateβ_α_minus_β (a b : ℝ) (hαβ : α a b - β a b = ⟨0, 2 * b⟩)
  (h2s3 : (2 * b).abs = 2 * Real.sqrt 3)
  (hα_β2_real : ((α a b) / (β a b)^2).im = 0) :
  Complex.abs (α a b) = 2 :=
by
  have hαβ_condition : α a b - β a b = ⟨0, 2 * b⟩ := hαβ
  have h2b_abs : (abs 2 * b) = 2 * abs b := by sorry
  have h2s3_rearrangement: 2 * (abs b) = 2 * Real.sqrt 3 := h2s3
  have hαβ_conclusion : abs b = Real.sqrt 3 := by sorry
  have hα_β2_real_condition : ((α a b) / (β a b)^2).im = 0 := hα_β2_real
  have hreal_conj_implies: α a b = ⟨a, b⟩ := by sorry
  show Complex.abs (α a b) = 2 from by sorry

end α_conjugateβ_α_minus_β_l143_143174


namespace standard_equation_of_ellipse_l143_143909

theorem standard_equation_of_ellipse : 
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  (∀ e : ℝ, e = 1/2 → 
  (∀ F : ℝ, F = 2 * Real.sqrt 3 → 
   (a ^ 2 = b ^ 2 + (e * a) ^ 2) → 
   (∀ x y : ℝ, x = 4 → y = 2 * Real.sqrt 3 → 
    (a = x ∧ b = y → 
     (∀ X Y : ℝ, X = 16 → Y = 12 → 
      ∀ x y : ℝ, x^2 / X + y^2 / Y = 1)))))) :=
begin
  sorry
end

end standard_equation_of_ellipse_l143_143909


namespace similar_rectangles_division_can_be_divided_into_unequal_similar_rectangles_l143_143819

theorem similar_rectangles_division (a b x : ℝ) (h : 0 < b)
  (h_inequality : 0 < a - 2 * b) :
  ∃ x, 0 < x ∧ x(a - x) = b^2 :=
sorry

theorem can_be_divided_into_unequal_similar_rectangles (a b : ℝ) : a > 2 * b ↔ 
  ∃ x, 0 < x ∧ x * (a - x) = b^2 :=
begin
  split,
  { intro h,
    use (a - sqrt (a^2 - 4 * b^2)) / 2,
    split,
    { -- prove 0 < x
      sorry },
    { -- prove x * (a - x) = b^2
      sorry }
  },
  { rintros ⟨x, hx1, hx2⟩,
    -- prove a > 2 * b from hx1 and hx2
    sorry }
end

end similar_rectangles_division_can_be_divided_into_unequal_similar_rectangles_l143_143819


namespace axis_of_symmetry_find_b_value_using_cosine_rule_l143_143919
noncomputable theory

def f (x : ℝ) : ℝ := sqrt(3) * sin x * cos x - cos x ^ 2 - 1 / 2

def g (x : ℝ) : ℝ := sin (x + π / 6) - 1

axiom B_in_the_range : π / 6 < B ∧ B < 7 * π / 6

axiom g_B_eq_0 : g B = 0

axiom sides_of_triangle : a = 2 ∧ c = 4

theorem axis_of_symmetry :
∀ k : ℤ, ∃ x : ℝ, f(x) = f(-x) 
	line 9             := sorry

theorem find_b_value_using_cosine_rule :
b = 2 * sqrt 3 :=
begin
	sorry
end

end axis_of_symmetry_find_b_value_using_cosine_rule_l143_143919


namespace rationalize_denominator_l143_143694

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143694


namespace largest_percent_error_l143_143988

theorem largest_percent_error (D : ℝ) (E : ℝ) (π : ℝ):
  D = 50 →
  E = 0.30 →
  π = Real.pi →
  let actual_area := π * (D / 2) ^ 2 in
  let min_diameter := D - D * E in
  let max_diameter := D + D * E in
  let min_area := π * (min_diameter / 2) ^ 2 in
  let max_area := π * (max_diameter / 2) ^ 2 in
  let min_error := ((actual_area - min_area) / actual_area) * 100 in
  let max_error := ((max_area - actual_area) / actual_area) * 100 in
  max_error = 69 :=
sorry

end largest_percent_error_l143_143988


namespace min_value_abs_function_l143_143112

theorem min_value_abs_function (a : ℝ) (f : ℝ → ℝ) (h : f = (λ x, |x + 1| + 2 * |x - a|)) :
  (∃ x, f x = 5) → (a = -6 ∨ a = 4) :=
by
  sorry

end min_value_abs_function_l143_143112


namespace distinguishable_boxes_indistinguishable_boxes_l143_143598

noncomputable def f (n r : ℕ) : ℕ :=
  r^n - (r.choose 1) * (r-1)^n + (r.choose 2) * (r-2)^n - ∑ i in finset.range r, (-1) ^ (i + 1) * (r.choose (i + 1)) * (r - (i + 1))^n

noncomputable def S (n r : ℕ) : ℕ :=
  (1 / r.factorial) * f n r

-- Theorem for the case of distinguishable boxes
theorem distinguishable_boxes (n r : ℕ) (h: n ≥ r) : f n r = r^n - (r.choose 1) * (r-1)^n + (r.choose 2) * (r-2)^n - ∑ i in finset.range r, (-1) ^ (i + 1) * (r.choose (i + 1)) * (r - (i + 1))^n :=
by sorry

-- Theorem for the case of indistinguishable boxes
theorem indistinguishable_boxes (n r : ℕ) (h: n ≥ r) : S n r = (1 / r.factorial) * (r^n - (r.choose 1) * (r-1)^n + (r.choose 2) * (r-2)^n - ∑ i in finset.range r, (-1) ^ (i + 1) * (r.choose (i + 1)) * (r - (i + 1))^n) :=
by sorry

end distinguishable_boxes_indistinguishable_boxes_l143_143598


namespace sum_divisors_24_l143_143276

theorem sum_divisors_24 :
  (∑ n in Finset.filter (λ n => 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_divisors_24_l143_143276


namespace value_of_a_l143_143945

theorem value_of_a (a x y : ℤ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - 3 * y = 1) : a = 2 := by
  sorry

end value_of_a_l143_143945


namespace find_linear_function_and_unit_price_l143_143973

def linear_function (k b x : ℝ) : ℝ := k * x + b

def profit (cost_price : ℝ) (selling_price : ℝ) (sales_volume : ℝ) : ℝ := 
  (selling_price - cost_price) * sales_volume

theorem find_linear_function_and_unit_price
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1 = 20) (h2 : y1 = 200)
  (h3 : x2 = 25) (h4 : y2 = 150)
  (h5 : x3 = 30) (h6 : y3 = 100)
  (cost_price := 10) (desired_profit := 2160) :
  ∃ k b x : ℝ, 
    (linear_function k b x1 = y1) ∧ 
    (linear_function k b x2 = y2) ∧ 
    (profit cost_price x (linear_function k b x) = desired_profit) ∧ 
    (linear_function k b x = -10 * x + 400) ∧ 
    (x = 22) :=
by
  sorry

end find_linear_function_and_unit_price_l143_143973


namespace smallest_resolvable_debt_l143_143266

theorem smallest_resolvable_debt : ∃ (D : ℕ), D > 0 ∧ (∀ p g : ℤ, D ≠ 400 * p + 250 * g) ∧ D = 50 := 
by {
  sorry -- proof omitted
}

end smallest_resolvable_debt_l143_143266


namespace pool_water_amount_correct_l143_143381

noncomputable def water_in_pool_after_ten_hours : ℝ :=
  let h1 := 8
  let h2_3 := 10 * 2
  let h4_5 := 14 * 2
  let h6 := 12
  let h7 := 12 - 8
  let h8 := 12 - 18
  let h9 := 12 - 24
  let h10 := 6
  h1 + h2_3 + h4_5 + h6 + h7 + h8 + h9 + h10

theorem pool_water_amount_correct :
  water_in_pool_after_ten_hours = 60 := 
sorry

end pool_water_amount_correct_l143_143381


namespace solve_for_m_l143_143852

def A := {x : ℝ | x^2 + 3*x - 10 ≤ 0}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem solve_for_m (m : ℝ) (h : B m ⊆ A) : m < 2 :=
by
  sorry

end solve_for_m_l143_143852


namespace cos_C_value_l143_143122

theorem cos_C_value {A B C : ℝ} {a b c k : ℝ} 
  (h1 : sin A / sin B = 3 / 2) 
  (h2 : sin A / sin C = 3 / 4)
  (h3 : a = 3 * k)
  (h4 : b = 2 * k)
  (h5 : c = 4 * k) :
  cos C = - 1 / 4 :=
by
  sorry

end cos_C_value_l143_143122


namespace smallest_positive_period_find_sides_of_triangle_l143_143074

-- Define vectors and function f
def vec_a (x : ℝ) : EuclideanSpace ℝ (Fin 2) := ![2 * Real.cos x, 1]
def vec_b (x : ℝ) : EuclideanSpace ℝ (Fin 2) := ![Real.cos x, Real.sqrt 3 * Real.sin (2 * x)]
def f (x : ℝ) : ℝ := inner (vec_a x) (vec_b x)

-- Smallest positive period of f(x)
theorem smallest_positive_period : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi := sorry

-- Sides of ∆ABC given specific conditions
def sides_are_valid (b c : ℝ) :=
  b + c = 3 ∧ b * c = 2 ∧ b > c

theorem find_sides_of_triangle (a A : ℝ) (h1 : a = Real.sqrt 3) (h2 : 2 = 2 * Real.sin (2 * A + Real.pi / 6) + 1) :
  ∃ b c, sides_are_valid b c ∧ b = 2 ∧ c = 1 := sorry

end smallest_positive_period_find_sides_of_triangle_l143_143074


namespace power_function_expression_l143_143109

theorem power_function_expression (a : ℝ) (f : ℝ → ℝ) (h : ∀ x: ℝ, f x = x^a)
(h_point : f 4 = 1 / 2) :
f = (λ x, x^(-1 / 2)) ∧ ∀ x, x > 0 → f x = x^(-1 / 2) :=
by
  sorry

end power_function_expression_l143_143109


namespace remove_denominators_l143_143755

theorem remove_denominators (x : ℝ) : (1 / 2 - (x - 1) / 3 = 1) → (3 - 2 * (x - 1) = 6) :=
by
  intro h
  sorry

end remove_denominators_l143_143755


namespace jessica_probability_at_least_two_correct_l143_143985

open Set
open Function
open Probability

noncomputable def jessica_quiz_probability : ℚ :=
  let p_wrong := (2 / 3 : ℚ)
  let p_correct := (1 / 3 : ℚ)
  let p0 := (p_wrong ^ 6) -- Probability of getting exactly 0 correct
  let p1 := (6 * p_correct * (p_wrong ^ 5)) -- Probability of getting exactly 1 correct
  1 - (p0 + p1)

theorem jessica_probability_at_least_two_correct :
  jessica_quiz_probability = 473 / 729 := by
  sorry

end jessica_probability_at_least_two_correct_l143_143985


namespace total_clients_l143_143189

theorem total_clients (V K B N : Nat) (hV : V = 7) (hK : K = 8) (hB : B = 3) (hN : N = 18) :
    V + K - B + N = 30 := by
  sorry

end total_clients_l143_143189


namespace rationalize_sqrt_5_over_12_l143_143646

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143646


namespace rationalize_sqrt_5_over_12_l143_143649

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143649


namespace b_2019_eq_1_l143_143452

def INT (x : ℝ) : ℤ := ⌊x⌋  -- Defines the integer part function

def a (n : ℕ) : ℤ := INT (1 / 7 * 2^n)

def b : ℕ → ℤ
| 0     := a 1  -- Let's assume ℕ* starts from 1
| (n+1) := a (n + 1) - 2 * a n

theorem b_2019_eq_1 : b 2019 = 1 := by
  sorry

end b_2019_eq_1_l143_143452


namespace ab_gt_6_l143_143860

noncomputable def F (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
noncomputable def G (a b : ℝ) (x : ℝ) : ℝ := x^2 + b * x + a
noncomputable def FG (a b : ℝ) (x : ℝ) : ℝ := F a b (G a b x)
noncomputable def GF (a b : ℝ) (x : ℝ) : ℝ := G a b (F a b x)

theorem ab_gt_6 (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hFG_real : ∀ x : ℝ, (FG a b x).real_roots) (hGF_real : ∀ x : ℝ, (GF a b x).real_roots) : 
  6 < a ∧ 6 < b :=
sorry

end ab_gt_6_l143_143860


namespace cylinder_lateral_surface_area_l143_143062

-- Define the given values
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Define the lateral surface area of the cylinder given base radius and slant height
def lateral_surface_area (r : ℝ) (l : ℝ) : ℝ := 2 * Real.pi * r * l

-- Theorem: Prove the lateral surface area given the conditions.
theorem cylinder_lateral_surface_area :
  lateral_surface_area radius slant_height = 30 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l143_143062


namespace simplify_expression_l143_143219

theorem simplify_expression (n : ℕ) : 
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by sorry

end simplify_expression_l143_143219


namespace compute_expression_l143_143823

theorem compute_expression :
  (143 + 29) * 2 + 25 + 13 = 382 :=
by 
  sorry

end compute_expression_l143_143823


namespace sum_of_positive_divisors_of_24_l143_143290

theorem sum_of_positive_divisors_of_24 : ∑ n in {n : ℕ | n > 0 ∧ (n+24) % n = 0}, n = 60 := 
by sorry

end sum_of_positive_divisors_of_24_l143_143290


namespace sample_size_is_correct_l143_143352

-- Define the conditions
def num_classes := 40
def students_per_class := 50
def selected_students := 150

-- Define the statement to prove the sample size
theorem sample_size_is_correct : selected_students = 150 := by 
  -- Proof is skipped with sorry
  sorry

end sample_size_is_correct_l143_143352


namespace cricket_team_members_l143_143739

theorem cricket_team_members (avg_whole_team: ℕ) (captain_age: ℕ) (wicket_keeper_age: ℕ) 
(remaining_avg_age: ℕ) (n: ℕ):
avg_whole_team = 23 →
captain_age = 25 →
wicket_keeper_age = 30 →
remaining_avg_age = 22 →
(n * avg_whole_team - captain_age - wicket_keeper_age = (n - 2) * remaining_avg_age) →
n = 11 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cricket_team_members_l143_143739


namespace count_even_numbers_l143_143936

theorem count_even_numbers : 
  ∃ n : ℕ, n = 199 ∧ ∀ m : ℕ, (302 ≤ m ∧ m < 700 ∧ m % 2 = 0) → 
    151 ≤ ((m - 300) / 2) ∧ ((m - 300) / 2) ≤ 349 :=
sorry

end count_even_numbers_l143_143936


namespace area_NPQ_l143_143121

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_NPQ :
  let X : ℝ × ℝ := (0, 0)
  let Y : ℝ × ℝ := (20, 0)
  let Z : ℝ × ℝ := (15.2, 13)
  let P : ℝ × ℝ := (10, 6.5) -- Circumcenter
  let Q : ℝ × ℝ := (7.7, 5.1) -- Incenter
  let N : ℝ × ℝ := (15.2, 9.5) -- Excircle center
  in triangle_area N P Q = 49.21 :=
by
  -- sorry is used to skip the proof.
  sorry

end area_NPQ_l143_143121


namespace circle_definition_l143_143779

theorem circle_definition (plane : Type) [metric_space plane] (O : plane) (r : ℝ) :
  ∀ (A : plane), dist A O = r ↔ A ∈ {P | dist P O = r} :=
by
  sorry

end circle_definition_l143_143779


namespace robert_total_amount_l143_143702

theorem robert_total_amount (T : ℝ) 
  (hrm : 100) 
  (hm : 125) 
  (hc : 0.10 * T) 
  (h_expense : T = 100 + 125 + 0.10 * T) : 
  T = 250 := 
by 
  sorry

end robert_total_amount_l143_143702


namespace area_of_triangle_l143_143485

theorem area_of_triangle : 
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := {x : ℝ | f x = 0}
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  in area = 168 :=
by
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := {x | f x = 0}
  let y_intercept := f 0
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  sorry

end area_of_triangle_l143_143485


namespace rationalize_denominator_l143_143679

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143679


namespace rationalize_denominator_l143_143698

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143698


namespace average_marks_l143_143252

variable (M P C : ℤ)

-- Conditions
axiom h1 : M + P = 50
axiom h2 : C = P + 20

-- Theorem statement
theorem average_marks : (M + C) / 2 = 35 := by
  sorry

end average_marks_l143_143252


namespace sum_of_second_largest_and_third_smallest_eq_1434_l143_143202

open List

noncomputable def digits := [1, 6, 8]

-- Generate all permutations of 3 elements from the digits list
noncomputable def all_three_digit_numbers (l : List ℕ) : List ℕ :=
  (perm l).map (λ x, 100 * x.nthLe 0 (by simp) + 10 * x.nthLe 1 (by simp) + x.nthLe 2 (by simp))

def second_largest_and_third_smallest_sum (l : List ℕ) : ℕ :=
  let sorted_l := l.qsort (≤)
  sorted_l.getNth! (sorted_l.length - 2) + sorted_l.getNth! 2

theorem sum_of_second_largest_and_third_smallest_eq_1434 : 
  second_largest_and_third_smallest_sum (all_three_digit_numbers digits) = 1434 := 
sorry

end sum_of_second_largest_and_third_smallest_eq_1434_l143_143202


namespace line_circle_separate_l143_143470

noncomputable theory

def polar_circle (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

def parametric_line (t x y : ℝ) : Prop :=
  x = Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t + 4 * Real.sqrt 2

def center (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 2

def distance (x0 y0 a b : ℝ) : ℝ :=
  abs (a * x0 + b * y0) / Real.sqrt (a^2 + b^2)

theorem line_circle_separate : 
  (∃ (x y : ℝ), center x y) ∧ 
  (∀ t, parametric_line t 1 (-1)) ∧
  (distance 1 (-1) 1 (-4 * Real.sqrt 2) = 4 - Real.sqrt 2 ∧ 4 - Real.sqrt 2 > Real.sqrt 2) :=
begin
  sorry
end

end line_circle_separate_l143_143470


namespace amanda_car_round_trip_time_l143_143802

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end amanda_car_round_trip_time_l143_143802


namespace find_cos_beta_l143_143905

noncomputable def α : ℝ := sorry  -- Placeholder, α will be inferred from the provided conditions
noncomputable def β : ℝ := sorry  -- Placeholder, β will be inferred from the provided conditions
noncomputable def cos_alpha : ℝ := 1 / 3
noncomputable def sin_sum_alpha_beta : ℝ := -3 / 5

-- The main theorem to prove
theorem find_cos_beta 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : π / 2 < β ∧ β < π)
  (h_cos_alpha : cos α = cos_alpha)
  (h_sin_sum_alpha_beta : sin (α + β) = sin_sum_alpha_beta) :
  cos β = - (4 + 6 * sqrt 2) / 15 := 
sorry

end find_cos_beta_l143_143905


namespace greatest_prime_divisor_sum_of_digits_eq_14_l143_143020

theorem greatest_prime_divisor_sum_of_digits_eq_14 (n : ℕ) (h1 : n = 32767) : 
  let p := Nat.greatestPrimeDivisor n in 
  (p.digits.sum = 14) :=
by
  sorry

end greatest_prime_divisor_sum_of_digits_eq_14_l143_143020


namespace rationalize_sqrt_fraction_l143_143618

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143618


namespace func1_max_min_func2_max_min_l143_143015

noncomputable def func1 (x : ℝ) : ℝ := 2 * Real.sin x - 3
noncomputable def func2 (x : ℝ) : ℝ := (7/4 : ℝ) + Real.sin x - (Real.sin x) ^ 2

theorem func1_max_min : (∀ x : ℝ, func1 x ≤ -1) ∧ (∃ x : ℝ, func1 x = -1) ∧ (∀ x : ℝ, func1 x ≥ -5) ∧ (∃ x : ℝ, func1 x = -5)  :=
by
  sorry

theorem func2_max_min : (∀ x : ℝ, func2 x ≤ 2) ∧ (∃ x : ℝ, func2 x = 2) ∧ (∀ x : ℝ, func2 x ≥ 7 / 4) ∧ (∃ x : ℝ, func2 x = 7 / 4) :=
by
  sorry

end func1_max_min_func2_max_min_l143_143015


namespace rationalize_denominator_l143_143697

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143697


namespace implicit_major_premise_l143_143126

def shooter_data : Type := Type
def scores : shooter_data → list ℝ := sorry
def standard_deviation (data : list ℝ) : ℝ := sorry
def more_stable_than (a b : shooter_data) : Prop := sorry

-- Let's define the shooters
def A : shooter_data := sorry
def B : shooter_data := sorry

-- Given conditions
axiom shots_A : ∃ s : list ℝ, s.length = 10 ∧ scores A = s
axiom shots_B : ∃ s : list ℝ, s.length = 10 ∧ scores B = s
axiom std_dev_comparison : standard_deviation (scores A) > standard_deviation (scores B)
axiom coach_conclusion : more_stable_than B A

-- We need to prove this
theorem implicit_major_premise : 
  (∀ data1 data2 : shooter_data, standard_deviation (scores data1) > standard_deviation (scores data2) → ¬more_stable_than data1 data2) := 
sorry

end implicit_major_premise_l143_143126


namespace material_point_motion_l143_143341

theorem material_point_motion (C k : ℝ) (s : ℝ → ℝ) :
  (∀ t, (s t).ln = k * t + C) ∧
  s 0 = 1 ∧
  s 2 = Real.exp 1 →
  s = λ t, Real.exp (t / 2) :=
by
  -- proof here
  sorry

end material_point_motion_l143_143341


namespace parabola_line_intersect_solutions_count_l143_143030

theorem parabola_line_intersect_solutions_count :
  ∃ b1 b2 : ℝ, (b1 ≠ b2 ∧ (b1^2 - b1 - 3 = 0) ∧ (b2^2 - b2 - 3 = 0)) :=
by
  sorry

end parabola_line_intersect_solutions_count_l143_143030


namespace eccentricity_of_ellipse_l143_143228

theorem eccentricity_of_ellipse {x y : ℝ} (h : x^2 / 2 + y^2 = 1) : 
  let a := sqrt 2,
      c := 1,
      e := c / a
  in e = sqrt 2 / 2 :=
  sorry

end eccentricity_of_ellipse_l143_143228


namespace proof_statements_l143_143163

variable {a b c : ℝ} (ha : a > b) (hb : b > 1) (hc : c < 0)

-- Define statements according to the given problem
def stmt1 := c / a > c / b
def stmt2 := a^c < b^c
def stmt3 := Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a

theorem proof_statements : stmt1 ha hb hc ∧ stmt2 ha hb hc ∧ stmt3 ha hb hc :=
by
  unfold stmt1 stmt2 stmt3
  sorry -- Proof of the theorem is omitted

end proof_statements_l143_143163


namespace verify_true_distance_and_error_l143_143135

def true_distance (pm ps pl : ℕ) : ℝ :=
  3.85 * pm + 1.05 * ps + 0.025 * pl

def first_surveyor_distance := true_distance 4 4 18
def second_surveyor_distance := true_distance 3 2 43
def third_surveyor_distance := true_distance 6 1 1

def true_distance_output := ((first_surveyor_distance + second_surveyor_distance + third_surveyor_distance) / 3)

theorem verify_true_distance_and_error :
  true_distance_output = 19.45 ∧ |first_surveyor_distance - 19.45| = 4.125 :=
by
  sorry

end verify_true_distance_and_error_l143_143135


namespace b_share_1800_l143_143309

theorem b_share_1800 (total_money : ℤ) (ratio_a ratio_b ratio_c : ℕ) (h : total_money = 5400 ∧ ratio_a = 2 ∧ ratio_b = 3 ∧ ratio_c = 4) : 
  let total_parts := ratio_a + ratio_b + ratio_c in
  let value_per_part := total_money / total_parts in
  let b_share := value_per_part * ratio_b in
  b_share = 1800 :=
by 
  intros total_parts value_per_part b_share
  sorry

end b_share_1800_l143_143309


namespace parallelepiped_properties_l143_143536

/-
Given:
- CO : ℝ, the projection of the lateral edge onto the base plane, is 5 dm
- C₁O : ℝ, the height of the parallelepiped, is 12 dm
- Sₘₙₖₗ : ℝ, the area of the rhombus section is 24 dm²
- KM : ℝ, one diagonal of the rhombus length is 8 dm

Prove:
- The lateral surface area is 260 dm²
- The volume of the parallelepiped is 312 dm³
-/

theorem parallelepiped_properties 
  (CO C₁O Sₘₙₖₗ KM : ℝ)
  (hCO : CO = 5)
  (hC₁O : C₁O = 12)
  (hSₘₙₖₗ : Sₘₙₖₗ = 24)
  (hKM : KM = 8)
: 
  (lateral_surface_area : ℝ) (volume : ℝ)
  (hlat : lateral_surface_area = 260)
  (hvol : volume = 312) := 
  by
  sorry

end parallelepiped_properties_l143_143536


namespace rationalize_sqrt_5_over_12_l143_143642

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143642


namespace find_x_l143_143879

theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_factors : ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧ (7^n + 1) = p1 * p2 * p3 ∧ (p1 = 2 ∨ p2 = 2 ∨ p3 = 2) ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
  7^n + 1 = 16808 :=
sorry

end find_x_l143_143879


namespace claire_wins_probability_l143_143254

/-- Define the conditions of the game and prove that, with both players playing optimally, the probability that Claire wins is 43/192. -/
theorem claire_wins_probability : 
  (probability_that_claire_wins :=
    let digit_choices_random : Finset Int := {1, 2, 3, 4, 5, 6}
    let sum_of_digits_divisible_by_3 : Int → Prop := λ n, n % 3 = 0
    let last_digit_even : Int → Prop := λ n, n % 2 = 0
    in (∃ sum, sum_of_digits_divisible_by_3 sum ∧ last_digit_even sum) → (43 / 192 : ℚ)
  ) := sorry

end claire_wins_probability_l143_143254


namespace jake_reaches_ground_later_by_2_seconds_l143_143812

noncomputable def start_floor : ℕ := 12
noncomputable def steps_per_floor : ℕ := 25
noncomputable def jake_steps_per_second : ℕ := 3
noncomputable def elevator_B_time : ℕ := 90

noncomputable def total_steps_jake := (start_floor - 1) * steps_per_floor
noncomputable def time_jake := (total_steps_jake + jake_steps_per_second - 1) / jake_steps_per_second
noncomputable def time_difference := time_jake - elevator_B_time

theorem jake_reaches_ground_later_by_2_seconds :
  time_difference = 2 := by
  sorry

end jake_reaches_ground_later_by_2_seconds_l143_143812


namespace rationalize_denominator_l143_143681

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143681


namespace probabilities_equal_l143_143504

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l143_143504


namespace tangram_area_is_correct_l143_143101

-- Definitions for the problem conditions and pieces
def tangram (total_area : ℝ) : Prop :=
  total_area = 1 -- Assuming the total tangram set area is normalized to 1 for simplicity

def piece_area (index : ℕ) (area : ℝ) : Prop :=
  match index with
  | 2 => area = 1 / 8
  | 4 => area = 1 / 16
  | 7 => area = 2 / 16
  | _ => True -- Undefined areas for other pieces for now

-- The theorem to prove the question (conditions => correct answers)
theorem tangram_area_is_correct :
  ∀ (total_area : ℝ),
  tangram total_area →
  piece_area 2 (1 / 8) ∧ (piece_area 4 (1 / 16) ∧ piece_area 7 (2 / 16) → 1 / 16 + 2 / 16 = 3 / 16) :=
by
  intros total_area h_total_area
  split
  {
    -- Prove the area of the second piece
    sorry
  }
  {
    intros h4 h7
    split
    {
      -- Prove the area of the fourth piece
      sorry
    }
    {
      -- Prove the area of the seventh piece
      sorry
      -- Prove the sum of the areas of the fourth and seventh pieces
      sorry
    }
  }

end tangram_area_is_correct_l143_143101


namespace rationalize_denominator_l143_143655

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143655


namespace gasoline_reduction_percentage_l143_143498

theorem gasoline_reduction_percentage : 
  ∀ (original_price : ℝ) (original_qty : ℝ) 
    (price_increase_percent : ℝ) (spending_increase_percent : ℝ),
    price_increase_percent = 0.78 →
    spending_increase_percent = 0.32 →
    original_price > 0 →
    original_qty > 0 →
  let new_price := original_price * (1 + price_increase_percent),
      new_budget := original_qty * original_price * (1 + spending_increase_percent),
      new_qty := new_budget / new_price,
      reduction_percent := ((original_qty - new_qty) / original_qty) * 100 
  in reduction_percent ≈ 25.84 :=
by
  intros original_price original_qty price_increase_percent spending_increase_percent
  simp only []
  sorry

end gasoline_reduction_percentage_l143_143498


namespace total_subjects_is_41_l143_143194

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end total_subjects_is_41_l143_143194


namespace det_B2_minus_3B_l143_143990

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 4], ![3, 2]]

theorem det_B2_minus_3B : 
  det (B ⬝ B - 3 • B) = 88 :=
by
  sorry

end det_B2_minus_3B_l143_143990


namespace smallest_positive_period_l143_143108

-- Define a predicate for a function to have a period
def is_periodic {α : Type*} [AddGroup α] (f : α → ℝ) (T : α) : Prop :=
  ∀ x, f (x) = f (x - T)

-- The actual problem statement
theorem smallest_positive_period {f : ℝ → ℝ} 
  (h : ∀ x : ℝ, f (3 * x) = f (3 * x - 3 / 2)) : 
  is_periodic f (1 / 2) ∧ 
  ¬ (∃ T : ℝ, 0 < T ∧ T < 1 / 2 ∧ is_periodic f T) :=
by
  sorry

end smallest_positive_period_l143_143108


namespace problem1_problem2_l143_143446

-- Problem 1
theorem problem1 (a : ℝ) (h : a = log 4 3) : 2^(a) + 2^(-a) = (4 * sqrt 3) / 3 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : log 2 (9^(x-1) - 5) = log 2 (3^(x-1) - 2) + 2) : x = 2 :=
by sorry

end problem1_problem2_l143_143446


namespace paper_string_area_l143_143258

theorem paper_string_area (side len overlap : ℝ) (n : ℕ) (h_side : side = 30) 
                          (h_len : len = 30) (h_overlap : overlap = 7) (h_n : n = 6) :
  let area_one_sheet := side * len
  let effective_len := side - overlap
  let total_length := len + effective_len * (n - 1)
  let width := side
  let area := total_length * width
  area = 4350 := 
by
  sorry

end paper_string_area_l143_143258


namespace minimum_value_x_min_achieved_at_6_l143_143302

theorem minimum_value_x (x : ℝ) : (x^2 - 12 * x + 35) ≥ -1 :=
by sorry

theorem min_achieved_at_6 : (∀ x, (x = 6 ↔ x^2 - 12 * x + 35 = -1)
:= by sorry

end minimum_value_x_min_achieved_at_6_l143_143302


namespace angle_between_PQ_and_CD_is_90_degrees_l143_143319

theorem angle_between_PQ_and_CD_is_90_degrees
  (A B C D P E F Q : Point)
  (h_inscribed : IsInscribedQuadrilateral A B C D)
  (h_intersect : IntersectAtDiagonals A C B D P)
  (h_obtuse : Angle A P B > 90)
  (h_midpoints : Midpoint E A D ∧ Midpoint F B C)
  (h_perpendiculars : Perpendicular E A C ∧ Perpendicular F B D)
  (h_intersect_perp : IntersectAtPerpendiculars E F Q) :
  Angle (Line P Q) (Line C D) = 90 :=
begin
  sorry
end

end angle_between_PQ_and_CD_is_90_degrees_l143_143319


namespace probability_is_1_div_21_l143_143084

/-- Let S be the set of integers from 1 to 105, inclusive. -/
def S : set ℕ := { x | 1 ≤ x ∧ x ≤ 105 }

/-- Define a quadratic equation where the coefficient a is from S. -/
def has_integer_roots (a : ℕ) : Prop := 
  ∃ m n : ℤ, m + n = -a ∧ m * n = 6 * a

/-- Compute the number of elements in S. -/
def S_size : ℕ := set.card S

/-- Compute the number of elements a in S such that the quadratic equation has integer roots. -/
def count_valid_a : ℕ := 
  set.card { a ∈ S | has_integer_roots a }

/-- Calculate the probability that a randomly chosen a from S results in the quadratic equation having integer roots. -/
def probability : ℚ := count_valid_a / S_size

/-- Prove that this probability is 1/21. -/
theorem probability_is_1_div_21 : probability = 1 / 21 :=
  sorry

end probability_is_1_div_21_l143_143084


namespace sum_of_positive_divisors_of_24_l143_143293

theorem sum_of_positive_divisors_of_24 : ∑ n in {n : ℕ | n > 0 ∧ (n+24) % n = 0}, n = 60 := 
by sorry

end sum_of_positive_divisors_of_24_l143_143293


namespace duration_of_investment_l143_143346

-- Define the constants as given in the conditions
def Principal : ℝ := 7200
def Rate : ℝ := 17.5
def SimpleInterest : ℝ := 3150

-- Define the time variable we want to prove
def Time : ℝ := 2.5

-- Prove that the calculated time matches the expected value
theorem duration_of_investment :
  SimpleInterest = (Principal * Rate * Time) / 100 :=
sorry

end duration_of_investment_l143_143346


namespace tips_fraction_l143_143129

variable (S : ℚ)

def week1_tips : ℚ := 11 / 4 * S
def week2_tips : ℚ := 7 / 3 * S
def total_income_salary : ℚ := 2 * S
def total_tips : ℚ := week1_tips S + week2_tips S
def total_income : ℚ := total_income_salary S + total_tips S

theorem tips_fraction :
  total_tips S / total_income S = 61 / 85 :=
by
  sorry

end tips_fraction_l143_143129


namespace angle_relationships_l143_143722

variables {P : Type} [MetricSpace P]

-- Define the points and their relationships
variables (A B C D S K L X Y : P)

-- Define the conditions in the problem
variables (is_trapezoid : ∃ h1 : ∃ (AB CD : Line), AB.parallel CD)
variables (AB_CD_parallel : parallel AB CD)
variables (AB_CD_meet_S : ∃ (AB_CD : Trapezoid), AB_CD.S = S)
variables (bisector_ASC : is_bisector_of_angle K L (angle A S C))
variables (X_on_SK : X ∈ (segment S K))
variables (Y_on_SL_extension : Y ∈ (extension_of_segment S L))
variables (angle_condition : angle A X C - angle A Y C = angle A S C)

theorem angle_relationships :
  angle B X D - angle B Y D = angle B S D :=
sorry

end angle_relationships_l143_143722


namespace part_1_part_2_l143_143916

-- Define the function f and g
def f (x m : ℝ) : ℝ := sqrt (abs (x + 1) + abs (x - 3) - m)
def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

-- Part (Ⅰ): Prove that m ≤ 4 given f(x) has domain ℝ
theorem part_1 (m : ℝ) (h : ∀ x : ℝ, f x m ≥ 0) : m ≤ 4 :=
sorry

-- Part (Ⅱ): Prove the minimum value of 7a + 4b is 9/4 given the equation
theorem part_2 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_eqn : 2 / (3 * a + b) + 1 / (a + 2 * b) = 4) :
  7 * a + 4 * b = 9 / 4 :=
sorry

end part_1_part_2_l143_143916


namespace option_B_equals_six_l143_143757

theorem option_B_equals_six :
  (3 - (-3)) = 6 :=
by
  sorry

end option_B_equals_six_l143_143757


namespace missing_number_l143_143455

theorem missing_number (n : ℝ) (h : (0.0088 * 4.5) / (0.05 * n * 0.008) = 990) : n = 0.1 :=
sorry

end missing_number_l143_143455


namespace carl_stamps_l143_143379

theorem carl_stamps : 
  ∀ (kevin_stamps : ℕ) (extra_stamps : ℕ), 
  kevin_stamps = 57 ∧ extra_stamps = 32 → 
  kevin_stamps + extra_stamps = 89 :=
by 
  intros kevin_stamps extra_stamps h,
  cases h,
  rw [h_left, h_right],
  rfl

end carl_stamps_l143_143379


namespace n_even_sequences_count_l143_143882

noncomputable def number_n_even_sequences (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n-2) - 2^((n-2) / 2)
  else
    2^(n-2) - 2^((n-3) / 2)

theorem n_even_sequences_count (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k = number_n_even_sequences n :=
begin
  use number_n_even_sequences n,
  refl,
end

end n_even_sequences_count_l143_143882


namespace min_value_of_expression_l143_143029

variable (a b : ℝ)

theorem min_value_of_expression (h : b ≠ 0) : 
  ∃ (a b : ℝ), (a^2 + b^2 + a / b + 1 / b^2) = Real.sqrt 3 :=
sorry

end min_value_of_expression_l143_143029


namespace candies_taken_by_boys_invariant_l143_143532

theorem candies_taken_by_boys_invariant (k : ℕ) (total_candies : ℕ) 
    (is_boy : Fin k.succ -> Bool) : 
    total_candies = 1000 -> 
    ∀ queue1 queue2 : List (Fin k.succ),
    Permutation queue1 queue2 ->
    (let candies_taken_by_boys := 
        queue1.foldl (λ C ch =>
            let taken := if is_boy ch then (Int.ceil (C / (k - queue1.indexOf ch + 1) : ℝ)) else (Int.floor (C / (k - queue1.indexOf ch + 1) : ℝ)) in
            C - taken) 1000 in
     let boys_in_queue := queue1.filter (λ ch => is_boy ch) in
     boys_in_queue.foldl (λ acc ch => acc + (if is_boy ch then (Int.ceil (total_candies / (k - queue1.indexOf ch + 1) : ℝ)) else (Int.floor (total_candies / (k - queue1.indexOf ch + 1) : ℝ)))) 0
    )
    = 
    (let candies_taken_by_boys := 
        queue2.foldl (λ C ch =>
            let taken := if is_boy ch then (Int.ceil (C / (k - queue2.indexOf ch + 1) : ℝ)) else (Int.floor (C / (k - queue2.indexOf ch + 1) : ℝ)) in
            C - taken) 1000 in
     let boys_in_queue := queue2.filter (λ ch => is_boy ch) in
     boys_in_queue.foldl (λ acc ch => acc + (if is_boy ch then (Int.ceil (total_candies / (k - queue2.indexOf ch + 1) : ℝ)) else (Int.floor (total_candies / (k - queue2.indexOf ch + 1) : ℝ)))) 0
    ) := sorry

end candies_taken_by_boys_invariant_l143_143532


namespace second_die_sides_l143_143146

theorem second_die_sides (p : ℚ) (n : ℕ) (h1 : p = 0.023809523809523808) (h2 : n ≠ 0) :
  let first_die_sides := 6
  let probability := (1 : ℚ) / first_die_sides * (1 : ℚ) / n
  probability = p → n = 7 :=
by
  intro h
  sorry

end second_die_sides_l143_143146


namespace rationalize_denominator_l143_143692

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143692


namespace max_value_l143_143306

-- Define the weights and values of gemstones
def weight_sapphire : ℕ := 6
def value_sapphire : ℕ := 15
def weight_ruby : ℕ := 3
def value_ruby : ℕ := 9
def weight_diamond : ℕ := 2
def value_diamond : ℕ := 5

-- Define the weight capacity
def max_weight : ℕ := 24

-- Define the availability constraint
def min_availability : ℕ := 10

-- The goal is to prove that the maximum value is 72
theorem max_value : ∃ (num_sapphire num_ruby num_diamond : ℕ),
  num_sapphire >= min_availability ∧
  num_ruby >= min_availability ∧
  num_diamond >= min_availability ∧
  num_sapphire * weight_sapphire + num_ruby * weight_ruby + num_diamond * weight_diamond ≤ max_weight ∧
  num_sapphire * value_sapphire + num_ruby * value_ruby + num_diamond * value_diamond = 72 :=
by sorry

end max_value_l143_143306


namespace magnitude_of_conjugate_l143_143970

def Z := Complex
def Z_constraint (z : Z) : Prop := z * (3 + 4 * Complex.i) = 7 + Complex.i

theorem magnitude_of_conjugate (z : Z) (hz : Z_constraint z) : Complex.abs (Complex.conj z) = Real.sqrt 2 :=
by
  sorry

end magnitude_of_conjugate_l143_143970


namespace parallel_alpha_neither_sufficient_nor_necessary_perpendicular_beta_l143_143901

-- Definitions based on conditions
variables {Point Line Plane : Type}
variables (m : Line) (α β : Plane)
variables [IsPerpendicular α β] [IsLine m] 

-- Question: Prove that "$m \parallel \alpha$" is neither sufficient nor necessary condition for "$m \perp \beta$"
theorem parallel_alpha_neither_sufficient_nor_necessary_perpendicular_beta :
  (m ∥ α) ⇔ ¬(m ⊥ β) := 
sorry

end parallel_alpha_neither_sufficient_nor_necessary_perpendicular_beta_l143_143901


namespace hexagon_of_interior_angle_120_l143_143787

theorem hexagon_of_interior_angle_120 (P : Type) [polygon P] 
  (h1 : ∀ (η : angle), η ∈ interior_angles P → η = 120 * (π / 180)) 
  (h2 : ∀ (η : angle), η ∈ interior_angles P → (interior_angle η + exterior_angle η) = π)
  (h3 : ∑ (η : angle) in exterior_angles P, η = 2 * π) : 
  num_sides P = 6 := 
sorry

end hexagon_of_interior_angle_120_l143_143787


namespace _l143_143559

variable {P D E A T : Type*} [plane_geometry]

-- Definition of points and conditions
noncomputable def triangle (A B C : point) : Prop := 
B ≠ C ∧ A ≠ B ∧ A ≠ C

noncomputable def isosceles (A B C : point) (AB AC : length) : Prop :=
AB = AC

noncomputable def points_on_seg (p1 p2 : point) (l : segment) : Prop :=
p1 ∈ l ∧ p2 ∈ l

noncomputable def circumcircle (Δ : triangle) (circle : circ) : Prop :=
circle ∈ Δ

noncomputable def main_theorem 
  (A B C D E T P : point)
  (AB AC : length)
  (circ_abc : circ)
  (ΔABC : triangle A B C)
  (isosceles_ABC : isosceles A B C AB AC)
  (points_on_AB_AC : points_on_seg D E (seg AB AC))
  (DE_eq_AC : length DE = AC)
  (DE_meet_circ_abc : meet DE circ_abc T)
  (P_on_AT : points_on_seg P AT (seg A T))
  : Prop :=
  (length PD + length PE = length AT) ↔ P ∈ circumcircle (ΔADE A D E)


end _l143_143559


namespace units_digit_of_sum_of_factorials_upto_8_l143_143829

def factorial (n : ℕ) : ℕ := 
if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials_upto (n : ℕ) : ℕ :=
(1 to n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10 

theorem units_digit_of_sum_of_factorials_upto_8 : 
  units_digit (sum_of_factorials_upto 8) = 3 := by
  sorry

end units_digit_of_sum_of_factorials_upto_8_l143_143829


namespace possible_integer_roots_l143_143828

def polynomial : Polynomial ℤ := Polynomial.X^4 + 2 * Polynomial.X^3 - Polynomial.X^2 + 3 * Polynomial.X - 30

theorem possible_integer_roots :
  {x : ℤ | polynomial.eval x = 0} ⊆ {±1, ±2, ±3, ±5, ±6, ±10, ±15, ±30} := sorry

end possible_integer_roots_l143_143828


namespace correct_difference_is_nine_l143_143760

-- Define the conditions
def misunderstood_number : ℕ := 35
def actual_number : ℕ := 53
def incorrect_difference : ℕ := 27

-- Define the two-digit number based on Yoongi's incorrect calculation
def original_number : ℕ := misunderstood_number + incorrect_difference

-- State the theorem
theorem correct_difference_is_nine : (original_number - actual_number) = 9 :=
by
  -- Proof steps go here
  sorry

end correct_difference_is_nine_l143_143760


namespace rationalize_sqrt_fraction_denom_l143_143605

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143605


namespace equalize_expenses_l143_143742

variable {x y : ℝ} 

theorem equalize_expenses (h : x > y) : (x + y) / 2 - y = (x - y) / 2 :=
by sorry

end equalize_expenses_l143_143742


namespace solve_for_x_l143_143490

noncomputable def x : ℝ :=
  let cond1 := (Real.log 2 (x^3)) - (Real.log (1 / 2) x) = 8
  4

theorem solve_for_x (h : (Real.log 2 (x^3)) - (Real.log (1 / 2) x) = 8) :
  x = 4 :=
by
  sorry

end solve_for_x_l143_143490


namespace find_quotient_l143_143957

theorem find_quotient (divisor remainder dividend : ℕ) (h_divisor : divisor = 24) (h_remainder : remainder = 5) (h_dividend : dividend = 1565) : 
  (dividend - remainder) / divisor = 65 :=
by
  sorry

end find_quotient_l143_143957


namespace fraction_of_green_knights_who_are_magical_is_l143_143958

noncomputable def magical_fraction : ℚ := 9 / 25

theorem fraction_of_green_knights_who_are_magical_is:
  (total_knights green_knights yellow_knights magical_knights : ℕ)
  (h1 : green_knights = total_knights / 3)
  (h2 : magical_knights = total_knights / 5)
  (h3 : ∀ (p q : ℕ), green_knights * p / q / 3 = yellow_knights * p / (3 * q)) :
  (green_knights * magical_fraction) = magical_knights := by
  sorry

end fraction_of_green_knights_who_are_magical_is_l143_143958


namespace number_of_triangles_in_triangular_grid_l143_143832

theorem number_of_triangles_in_triangular_grid :
  let grid := triangular_grid 4 4 in 
  count_triangles grid = 20 :=
by 
  sorry

end number_of_triangles_in_triangular_grid_l143_143832


namespace solve_cubic_eq_l143_143220

theorem solve_cubic_eq (x : ℂ) :
  (x - 2)^3 + (x - 6)^3 = 0 ↔ (x = 4 ∨ x = 4 + 2*complex.I*real.sqrt(3) ∨ x = 4 - 2*complex.I*real.sqrt(3)) :=
by
  sorry

end solve_cubic_eq_l143_143220


namespace find_line_eq_l143_143894

def point (α : ℝ) : ℝ × ℝ :=
  (2 * real.cos α, 2 * real.sin α)

def point_B (α : ℝ) : ℝ × ℝ :=
  (real.cos α - real.sqrt 3 * real.sin α, real.sin α + real.sqrt 3 * real.cos α)

def vector_OP (α : ℝ) : ℝ × ℝ :=
  let A := point α
  let B := point_B α
  (A.1 + B.1, A.2 + B.2)

def max_angle_line_eq (α : ℝ) : ℝ × ℝ :=
  (4, 0)

theorem find_line_eq (α : ℝ) :
  let P := vector_OP α
  ∃ k : ℝ, k = real.sqrt 3 ∧
  (P.2 - (k * ( P.1 - 4)) = 0 ∨ P.2 + (k * ( P.1 - 4)) = 0) :=
sorry

end find_line_eq_l143_143894


namespace domain_is_all_real_l143_143824

-- Definitions and conditions
def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 18

def domain_of_f (x : ℝ) : Prop := ∃ (y : ℝ), y = 1 / (⌊quadratic_expression x⌋)

-- Theorem statement
theorem domain_is_all_real : ∀ x : ℝ, domain_of_f x :=
by
  sorry

end domain_is_all_real_l143_143824


namespace power_mod_result_l143_143274

theorem power_mod_result :
  9^1002 % 50 = 1 := by
  sorry

end power_mod_result_l143_143274


namespace assign_roles_l143_143024

-- Define the number of people and the required roles
def num_people := 6
def num_cooks := 3
def num_table_setter := 1
def num_cleaners := 2

-- Prove the number of ways to assign these roles
theorem assign_roles : 
  nat.choose num_people num_cooks * 
  nat.choose (num_people - num_cooks) num_table_setter *
  nat.choose ((num_people - num_cooks) - num_table_setter) num_cleaners = 60 := 
by
  sorry

end assign_roles_l143_143024


namespace temperature_fraction_l143_143964

def current_temperature : ℤ := 84
def temperature_decrease : ℤ := 21

theorem temperature_fraction :
  (current_temperature - temperature_decrease) = (3 * current_temperature / 4) := 
by
  sorry

end temperature_fraction_l143_143964


namespace ninth_term_arith_seq_l143_143251

theorem ninth_term_arith_seq (a d : ℤ) (h1 : a + 2 * d = 25) (h2 : a + 5 * d = 31) : a + 8 * d = 37 :=
sorry

end ninth_term_arith_seq_l143_143251


namespace probability_equality_l143_143507

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l143_143507


namespace evaluate_expression_l143_143000

variables (a b c : ℝ)

theorem evaluate_expression :
  (3 * a^2 + 3 * b^2 - 5 * c^2 + 6 * a * b) / (4 * a^2 + 4 * c^2 - 6 * b^2 + 8 * a * c)
  = ((a + b + real.sqrt 5 * c) * (a + b - real.sqrt 5 * c)) / ((2 * (a + c) + real.sqrt 6 * b) * (2 * (a + c) - real.sqrt 6 * b)) :=
sorry

end evaluate_expression_l143_143000


namespace find_m_n_l143_143264

variables (A B C K : Type)
variables (AB BC CA : ℝ)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K]

-- Declare that AB, BC, CA are segments with lengths 7, 8, and 9 respectively
def length_AB := 7
def length_BC := 8
def length_CA := 9

-- Define circles ω₁ and ω₂
variables (ω₁ ω₂ : Circle)

-- Circles passing through specific points and being tangent at specific points
axiom circle_ω₁ : ω₁.passes_through B ∧ ω₁.tangent_at_point_on_line A C
axiom circle_ω₂ : ω₂.passes_through C ∧ ω₂.tangent_at_point_on_line A B

-- Intersection point of circles ω₁ and ω₂ not equal to A
axiom K_intersection : ω₁ ∩ ω₂ = {K} ∪ {A}

-- The theorem to be proved
theorem find_m_n (h : length (segment AK) = 9 / 2) : ∃ m n : ℕ, gcd m n = 1 ∧ m + n = 11 := begin
  sorry
end

end find_m_n_l143_143264


namespace intersecting_lines_sum_constant_l143_143238

theorem intersecting_lines_sum_constant
  (c d : ℝ)
  (h1 : 3 = (1 / 3) * 3 + c)
  (h2 : 3 = (1 / 3) * 3 + d) :
  c + d = 4 :=
by
  sorry

end intersecting_lines_sum_constant_l143_143238


namespace simplify_fraction_l143_143218

theorem simplify_fraction :
  (1 / (3 / (Real.sqrt 5 + 2) + 4 / (Real.sqrt 7 - 2))) = (3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10)) :=
sorry

end simplify_fraction_l143_143218


namespace simplify_sqrt_fraction_simplify_series_sum_l143_143003

-- Lean statement for part 1 of the problem
theorem simplify_sqrt_fraction (n : ℕ) (hn : 0 < n) : 
  (1 / (Real.sqrt n - Real.sqrt (n - 1))) = Real.sqrt n + Real.sqrt (n - 1) := 
sorry

-- Lean statement for part 2 of the problem
theorem simplify_series_sum : 
  (∑ k in Finset.range 99, (1 / (Real.sqrt k + Real.sqrt (k + 1)))) = 12 := 
sorry

end simplify_sqrt_fraction_simplify_series_sum_l143_143003


namespace solution_set_p_range_of_a_l143_143428

-- Define the predicates
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x a : ℝ) : Prop := |x - 3| < a

-- The solution set A corresponding to p
def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- The interval B for q
def B (a : ℝ) : set ℝ := {x | 3 - a < x ∧ x < 3 + a}

-- a > 0 condition
def a_pos (a : ℝ) : Prop := a > 0

-- a > 4 condition to be derived based on A ⊂ B
def a_gt_4 (a : ℝ) : Prop := a > 4

-- Main theorem statements to be proven
theorem solution_set_p (x : ℝ) : p x ↔ x ∈ A := by sorry

theorem range_of_a (a : ℝ) : A ⊂ B a → a_gt_4 a := by sorry

end solution_set_p_range_of_a_l143_143428


namespace factorial_fraction_l143_143384

theorem factorial_fraction : (8.factorial + 9.factorial + 10.factorial) / 7.factorial = 800 :=
by
  sorry

end factorial_fraction_l143_143384


namespace addition_problem_l143_143967

theorem addition_problem (m n p q : ℕ) (Hm : m = 2) (Hn : 2 + n + 7 + 5 = 20) (Hp : 1 + 6 + p + 8 = 24) (Hq : 3 + 2 + q = 12) (Hpositives : 0 < m ∧ 0 < n ∧ 0 < p ∧ 0 < q) :
  m + n + p + q = 24 :=
sorry

end addition_problem_l143_143967


namespace tangent_line_at_2_6_tangent_line_through_origin_l143_143918

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_at_2_6 : 
  ∃ m b, (∀ x, f' x = 3*x^2 + 1) ∧ (f' 2 = m) ∧ ((13 : ℝ) = 13) ∧ ((y : ℝ - 6 : ℝ = m * (x : ℝ - 2 : ℝ)) = ((13 * x) - y - 20 : ℝ = 0)) :=
by
  sorry

theorem tangent_line_through_origin :
  ∃ x₀ m, 
    (f' x₀ = 3 * x₀^2 + 1) ∧ 
    (13 = 13) ∧ 
    ((y - (x₀^3 + x₀ - 16) = m * (x - x₀)) = 
      ((y = 13 * x) ∧ 
      (tangent_point = (-2, -26) = (x₀ = -2 : ℝ ∧ 
        f(-2 : ℝ) = -26))) :=
by
  sorry

end tangent_line_at_2_6_tangent_line_through_origin_l143_143918


namespace rationalize_sqrt_fraction_l143_143670

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143670


namespace fill_time_with_conditions_l143_143348

-- Define rates as constants
def pipeA_rate := 1 / 10
def pipeB_rate := 1 / 6
def pipeC_rate := 1 / 5
def tarp_factor := 1 / 2
def leak_rate := 1 / 15

-- Define effective fill rate taking into account the tarp and leak
def effective_fill_rate := ((pipeA_rate + pipeB_rate + pipeC_rate) * tarp_factor) - leak_rate

-- Define the required time to fill the pool
def required_time := 1 / effective_fill_rate

theorem fill_time_with_conditions :
  required_time = 6 :=
by
  sorry

end fill_time_with_conditions_l143_143348


namespace grade_representation_l143_143096

theorem grade_representation :
  (8, 1) = (8, 1) :=
by
  sorry

end grade_representation_l143_143096


namespace runners_meet_again_l143_143256

theorem runners_meet_again : ∃ t : ℕ, t = 800 ∧ 
  (∃ k1 : ℤ, (5.5 * t - 5.0 * t) = 400 * k1) ∧
  (∃ k2 : ℤ, (6.0 * t - 5.5 * t) = 400 * k2) ∧
  (∃ k3 : ℤ, (6.0 * t - 5.0 * t) = 400 * k3) :=
sorry

end runners_meet_again_l143_143256


namespace simplify_expression_l143_143213

theorem simplify_expression : (84 / 1764) * 21 = 1 / 2 := 
by {
  -- Factor 84 and 1764
  have h84 : 84 = 2^2 * 3 * 7 := by norm_num,
  have h1764 : 1764 = 2^2 * 3 * 7^2 := by norm_num,
  
  -- Perform the simplification
  calc
    (84 / 1764) * 21
        = (2^2 * 3 * 7 / (2^2 * 3 * 7^2)) * 21 : by rw [h84, h1764]
    ... = (7 / 7^2) * 21                : by { field_simp, ring }
    ... = (1 / 7) * 21                   : by { simp }
    ... = 1/2                                   : by norm_num
}

end simplify_expression_l143_143213


namespace rita_canoe_trip_distance_l143_143701

theorem rita_canoe_trip_distance 
  (D : ℝ)
  (h_upstream : ∃ t1, t1 = D / 3)
  (h_downstream : ∃ t2, t2 = D / 9)
  (h_total_time : ∃ t1 t2, t1 + t2 = 8) :
  D = 18 :=
by
  sorry

end rita_canoe_trip_distance_l143_143701


namespace find_m_and_n_l143_143118

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end find_m_and_n_l143_143118


namespace shots_radius_l143_143483

theorem shots_radius (R r : ℝ) (hR : R = 3) (n : ℕ) (hn : n = 27) :
  (R ≥ 3 * r) → r ≤ 1 :=
by {
  -- We assume the hypothesis R = 3 and n = 27
  intros hab,
  rw hR at hab,
  rw hab,
  exact le_of_eq hn,
  sorry
}

end shots_radius_l143_143483


namespace ratio_e_a_l143_143944

theorem ratio_e_a (a b c d e : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4) :
  e / a = 8 / 15 := 
by
  sorry

end ratio_e_a_l143_143944


namespace m_range_proof_l143_143857

theorem m_range_proof
  (x m : ℝ)
  (A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10})
  (B : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m ∧ 0 < m})
  (not_p_nec_not_q : ¬ ∃ x ∈ A, ¬ ∃ (x : ℝ) ∈ B, true) :
  m ≥ 9 :=
by
  sorry

end m_range_proof_l143_143857


namespace area_of_triangle_l143_143723

theorem area_of_triangle
  (m_line : ∀ x y : ℝ, 3 * x - y + 2 = 0)
  (l_line : ∀ x y : ℝ, 3 * x + y + 2 = 0) : 
  (∃ x : ℝ, m_line x 0) ∧
  (∃ y : ℝ, m_line 0 y) ∧
  (∃ x : ℝ, l_line x 0) ∧
  (∃ y : ℝ, l_line 0 y) ∧
  (∃ area : ℝ, area = 4 / 3) :=
sorry

end area_of_triangle_l143_143723


namespace solid_not_necessarily_frustum_l143_143790

structure Solid :=
  (faces : Set Face)
  (bounded_by_parallel_planes : ∃ (plane1 plane2 : Plane), plane1 ≠ plane2 ∧ ∀ (face : Face), face ∈ faces → face_is_trapezoid face)

def face_is_trapezoid (face : Face) : Prop := sorry  -- assuming we have a way to recognize trapezoid faces

def is_frustum (solid : Solid) : Prop := sorry -- assuming we have a way to recognize a frustum

theorem solid_not_necessarily_frustum (S : Solid) (H : S.bounded_by_parallel_planes) : ¬ is_frustum S :=
by
  sorry

end solid_not_necessarily_frustum_l143_143790


namespace find_f_at_1_l143_143111

-- Let's define the inverse function
def f_inv (x : ℝ) : ℝ := 3^(x + 1)

-- Translate the problem to verifying the value of the original function f when x = 1
theorem find_f_at_1 : ∃ f : ℝ → ℝ, (∀ x : ℝ, f_inv (f x) = x) ∧ f 1 = -1 :=
begin
  -- The proof goes here.
  sorry
end

end find_f_at_1_l143_143111


namespace log_sum_tangent_angles_zero_l143_143837

noncomputable def log_sum_tangent_angles : ℝ :=
∑ i in (finset.range 45).map (λ i, i + 1).to_finset, real.log (real.tan (i : ℝ * (π / 180)))

theorem log_sum_tangent_angles_zero : log_sum_tangent_angles = 0 := 
by
  sorry

end log_sum_tangent_angles_zero_l143_143837


namespace monotonic_increasing_intervals_min_max_values_l143_143077

-- Define the function f(x) with the given conditions
def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x - π / 4)

-- Define the interval where the function is monotonically increasing
theorem monotonic_increasing_intervals (k : ℤ) : 
  ∃ I : set ℝ, I = set.Icc (-(π / 8 : ℝ) + (k : ℝ) * π) ((3 * π / 8 : ℝ) + (k : ℝ) * π) ∧ 
  ∀ x y ∈ I, x < y → f x < f y := sorry

-- Define the minimum and maximum values on the interval [π/8, 3π/4]
theorem min_max_values : 
  ∀ x ∈ set.Icc (π / 8 : ℝ) (3 * π / 4 : ℝ), 
  f x ∈ set.Icc (-(1 : ℝ)) (sqrt 2) ∧ 
  (f (π / 8 : ℝ) = 0 ∧ 
   f (3 * π / 8 : ℝ) = sqrt 2 ∧ 
   f (3 * π / 4 : ℝ) = -1) := sorry

end monotonic_increasing_intervals_min_max_values_l143_143077


namespace log_base_5_12_l143_143854

theorem log_base_5_12 (a b : ℝ) (h1 : log 5 2 = a) (h2 : 5 ^ b = 3) : log 5 12 = 2 * a + b :=
by
  sorry

end log_base_5_12_l143_143854


namespace number_of_true_propositions_l143_143113

variable (x : ℝ)

def original_proposition (x : ℝ) : Prop := (x = 5) → (x^2 - 8 * x + 15 = 0)
def converse_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 = 0) → (x = 5)
def inverse_proposition (x : ℝ) : Prop := (x ≠ 5) → (x^2 - 8 * x + 15 ≠ 0)
def contrapositive_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 ≠ 0) → (x ≠ 5)

theorem number_of_true_propositions : 
  (original_proposition x ∧ contrapositive_proposition x) ∧
  ¬(converse_proposition x) ∧ ¬(inverse_proposition x) ↔ true := sorry

end number_of_true_propositions_l143_143113


namespace solve_m_n_l143_143119

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end solve_m_n_l143_143119


namespace ingrid_bake_percent_l143_143552

def irin_ratio : ℝ := 9.18
def ingrid_ratio : ℝ := 5.17
def nell_ratio : ℝ := 2.05
def total_cookies : ℝ := 148

def total_ratio : ℝ := irin_ratio + ingrid_ratio + nell_ratio
def ingrid_cookies : ℝ := (ingrid_ratio / total_ratio) * total_cookies
def ingrid_percentage : ℝ := (ingrid_cookies / total_cookies) * 100

theorem ingrid_bake_percent :
  ingrid_percentage ≈ 31.76 :=
sorry

end ingrid_bake_percent_l143_143552


namespace distance_between_consecutive_trees_l143_143773

theorem distance_between_consecutive_trees :
  ∀ (yard_length : ℝ) (num_trees : ℕ),
    yard_length = 375 ∧ num_trees = 37 →
      yard_length / (num_trees - 1) = 375 / 36 :=
begin
  intros yard_length num_trees h,
  cases h with h_yard h_trees,
  rw [h_yard, h_trees],
  norm_num,
end

end distance_between_consecutive_trees_l143_143773


namespace paula_travel_fraction_l143_143184

theorem paula_travel_fraction :
  ∀ (f : ℚ), 
    (∀ (L_time P_time travel_total : ℚ), 
      L_time = 70 →
      P_time = 70 * f →
      travel_total = 504 →
      (L_time + 5 * L_time + P_time + P_time = travel_total) →
      f = 3/5) :=
by
  sorry

end paula_travel_fraction_l143_143184


namespace hall_breadth_15_l143_143336

noncomputable def hall_breadth (length : ℕ) (stone_length_dm stone_breadth_dm stones_required : ℕ) : ℕ :=
  let length_m : ℕ := length
  let stone_length_m : ℚ := stone_length_dm / 10
  let stone_breadth_m : ℚ := stone_breadth_dm / 10
  let area_per_stone_m2 : ℚ := stone_length_m * stone_breadth_m
  let total_area_m2 : ℚ := stones_required * area_per_stone_m2
  let breadth_m : ℚ := total_area_m2 / length
  breadth_m.natAbs

theorem hall_breadth_15 :
  hall_breadth 36 2 5 5400 = 15 :=
by
  sorry

end hall_breadth_15_l143_143336


namespace rationalize_sqrt_fraction_denom_l143_143604

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143604


namespace ratio_of_angles_in_triangle_l143_143151

noncomputable def ratio_of_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  (BC / (AB - BC)) = (AB + BC) / AC

theorem ratio_of_angles_in_triangle (A B C : ℝ) (a b c : ℝ)
  (h : ratio_of_angles A B C a b c) :
  ∠A : ∠C = 1 : 2 := 
sorry

end ratio_of_angles_in_triangle_l143_143151


namespace events_A_B_equal_prob_l143_143514

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l143_143514


namespace arithmetic_to_geometric_and_sum_l143_143424

/-- Given f(x) = log_a x (a > 0, a ≠ 1), 
and the sequence f(a₁), f(a₂), f(a₃), ... , f(a_n), ... is an 
arithmetic sequence with the first term being 4 and the common difference being 2,
prove (I) that the sequence {a_n} forms a geometric sequence, 
and (II) when a = sqrt(2), the sum of the first n terms of the sequence {b_n},
where b_n = a_n f(a_n), is S_n = n * 2^(n+3).
-/
theorem arithmetic_to_geometric_and_sum 
  (a : Real) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) 
  (f : Real → Real) (h_f : ∀ x, f x = Real.log x / Real.log a) 
  (a_n : ℕ → Real) (h_seq : ∀ n, f (a_n n) = (2 : Real) * n + 2) 
  (b_n : ℕ → Real := λ n, a_n n * f (a_n n)) 
  (S_n : ℕ → Real := λ n, Finset.sum (Finset.range n) (λ k, b_n (k + 1)))
  :
  -- Part (I): Show that {a_n} forms a geometric sequence.
  (∃ r : Real, ∀ n, a_n (n + 1) = r * a_n n) ∧ 
  -- Part (II): When a = sqrt(2), show that S_n = n * 2^(n+3).
  let a_sqrt_2 := Real.sqrt 2 in
  (a = a_sqrt_2 → ∀ n, S_n n = n * 2^(n + 3)) :=
sorry

end arithmetic_to_geometric_and_sum_l143_143424


namespace determine_function_l143_143395

noncomputable def f (x : ℝ) : ℝ := 1 - x^2 / 2

theorem determine_function :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1) →
  (∀ x, f x = 1 - x^2 / 2) := by
  sorry

end determine_function_l143_143395


namespace polygon_sides_doubled_increase_diagonals_l143_143496

theorem polygon_sides_doubled_increase_diagonals :
  ∃ n : ℕ, (n * (2 * n - 3) - (n * (n - 3) / 2) = 45) ∧ n = 6 :=
by
  use 6
  constructor
  · -- Left hand side of the equation
    have h1: 6 * (2 * 6 - 3) = 6 * 9 := by rfl
    have h2: 6 * (6 - 3) / 2 = 9 := by norm_num
    rw [h1, h2]
    norm_num
  · rfl

end polygon_sides_doubled_increase_diagonals_l143_143496


namespace rationalize_denominator_l143_143685

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143685


namespace mileage_entries_l143_143315

theorem mileage_entries (n : Nat) (h : n = 8) : ∑ i in Finset.range n, i = 28 := by
  sorry

end mileage_entries_l143_143315


namespace vector_scalar_sub_l143_143383

def a : ℝ × ℝ := (3, -9)
def b : ℝ × ℝ := (2, -8)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_scalar_sub:
  scalar1 • a - scalar2 • b = (6, -12) := by
  sorry

end vector_scalar_sub_l143_143383


namespace technicians_in_workshop_l143_143962

theorem technicians_in_workshop :
  (∃ T R: ℕ, T + R = 42 ∧ 8000 * 42 = 18000 * T + 6000 * R) → ∃ T: ℕ, T = 7 :=
by
  sorry

end technicians_in_workshop_l143_143962


namespace count_squares_with_given_ones_digit_l143_143091

-- Definitions for the conditions.
def ends_in_1_or_9 (n : ℕ) : Prop := ∃ k, (k % 10 = 1 ∨ k % 10 = 9) ∧ k^2 = n
def ends_in_2_or_8 (n : ℕ) : Prop := ∃ k, (k % 10 = 2 ∨ k % 10 = 8) ∧ k^2 = n
def ends_in_3_or_7 (n : ℕ) : Prop := ∃ k, (k % 10 = 3 ∨ k % 10 = 7) ∧ k^2 = n

-- Main statement combining all conditions and giving the final answer.
theorem count_squares_with_given_ones_digit :
  let squares := {n : ℕ | n < 500 ∧ 
                  (  ends_in_1_or_9 n
                  ∨ ends_in_2_or_8 n
                  ∨ ends_in_3_or_7 n)} in 
  squares.to_finset.card = 13 :=
by sorry

end count_squares_with_given_ones_digit_l143_143091


namespace multiply_102_98_square_99_l143_143818

theorem multiply_102_98 : 102 * 98 = 9996 :=
by
  have h1 : (100 + 2) * (100 - 2) = 100^2 - 2^2 := by rw [add_mul, mul_sub, add_sub, mul_self, mul_self, sub_self]
  calc
    102 * 98 = (100 + 2) * (100 - 2) : by rw [add_sub]
    ... = 100^2 - 2^2 : by rw [h1]
    ... = 10000 - 4 : by rw [pow_two, pow_two]
    ... = 9996 : by norm_num

theorem square_99 : 99^2 = 9801 :=
by
  have h2 : (100 - 1)^2 = 100^2 - 2 * 100 * 1 + 1^2 := by rw [sub_sq, mul_two, one_mul, add_subitory]
  calc
    99^2 = (100 - 1)^2 : by rw [sub_sq]
    ... = 100^2 - 2 * 100 * 1 + 1^2 : by rw [h2]
    ... = 10000 - 200 + 1 : by rw [pow_two, one_pow, sub_add_cancel]
    ... = 9801 : by norm_num


end multiply_102_98_square_99_l143_143818


namespace circles_intersect_l143_143729

noncomputable def circle_center_and_radius (a b: ℝ) : ℝ × ℝ × ℝ :=
  let r: ℝ := real.sqrt (a^2 + b^2)
  ((r - a, r - b), r)

theorem circles_intersect
(C1_eq : ∀ x y : ℝ, x^2 + y^2 - 4*x = 0)
(C2_eq : ∀ x y : ℝ, x^2 + y^2 + 2*y = 0) :
  let c1 := circle_center_and_radius 4 0,
      c2 := circle_center_and_radius 0 2,
      d := real.sqrt ((c1.1.1 - c2.1.1)^2 + (c1.1.2 - c2.1.2)^2),
      R := c1.2,
      r := c2.2
  in R - r < d ∧ d < R + r :=
by 
  sorry

end circles_intersect_l143_143729


namespace find_x_l143_143878

theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_factors : ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧ (7^n + 1) = p1 * p2 * p3 ∧ (p1 = 2 ∨ p2 = 2 ∨ p3 = 2) ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
  7^n + 1 = 16808 :=
sorry

end find_x_l143_143878


namespace rationalize_denominator_l143_143603

theorem rationalize_denominator :
  (3 : ℝ) / Real.sqrt 48 = Real.sqrt 3 / 4 :=
by
  sorry

end rationalize_denominator_l143_143603


namespace probability_equality_l143_143509

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l143_143509


namespace brad_balloons_total_l143_143373

theorem brad_balloons_total (red_balloons : ℕ) (green_balloons : ℕ) (h_red : red_balloons = 8) (h_green : green_balloons = 9) :
  red_balloons + green_balloons = 17 :=
by
  rw [h_red, h_green]
  norm_num
  sorry

end brad_balloons_total_l143_143373


namespace num_solutions_l143_143769

theorem num_solutions (h : ∀ n : ℕ, (1 ≤ n ∧ n ≤ 455) → n^3 % 455 = 1) : 
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ (1 ≤ n ∧ n ≤ 455) ∧ n^3 % 455 = 1) ∧ s.card = 9) :=
sorry

end num_solutions_l143_143769


namespace card_arrangement_impossible_l143_143476

theorem card_arrangement_impossible :
  ¬ ∃ (arrangement : list ℕ),
    (arrangement.length = 20) ∧
    (multiset.card arrangement.to_multiset = 20) ∧
    (∀ d ∈ (finset.range 10), list.count d arrangement = 2) ∧
    (∀ d ∈ (finset.range 10), 
      ∃ i j, i < j ∧ arrangement.nth i = some d ∧ arrangement.nth j = some d ∧ j - i - 1 = d) :=
by sorry

end card_arrangement_impossible_l143_143476


namespace ratio_part_to_whole_l143_143597

variable (N : ℝ)

theorem ratio_part_to_whole :
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  0.4 * N = 120 →
  (10 / ((1 / 3) * (2 / 5) * N) = 1 / 4) :=
by
  intros h1 h2
  sorry

end ratio_part_to_whole_l143_143597


namespace arrange_books_correct_l143_143537

def math_books : Nat := 4
def history_books : Nat := 4

def arrangements (m h : Nat) : Nat := sorry

theorem arrange_books_correct :
  arrangements math_books history_books = 576 := sorry

end arrange_books_correct_l143_143537


namespace three_element_subset_sum_compare_l143_143709

/-- For the set S = {1, 2, ..., 63}, prove that the number of 3-element 
subsets whose sum is greater than 95 is greater than the number of those whose sum is less than 95. -/
theorem three_element_subset_sum_compare :
  let S := Finset.range 64 in
  ∃ f : Finset ℕ ≃ Finset ℕ,
  (∀ a b c ∈ S, a < b < c → f (Finset.mk [a, b, c]) = Finset.mk [64-a, 64-b, 64-c]) ∧
  (∃ N1 N2 ∈ Finset.powersetLen 3 S,
   (N1.sum id > 95 ∧ N2.sum id < 95) ∧ N1 > N2) :=
by
  sorry

end three_element_subset_sum_compare_l143_143709


namespace tangent_line_at_1_2_l143_143720

noncomputable def curve (x : ℝ) : ℝ := x^2 + 1/x

theorem tangent_line_at_1_2 :
  let x0 := 1
  let y0 := curve x0
  let derivative_at : ℝ := deriv curve x0
  let tangent_line (x y : ℝ) := derivative_at * (x - x0) = y - y0
  tangent_line = (λ x y, x - y + 1 = 0) :=
sorry

end tangent_line_at_1_2_l143_143720


namespace sum_A_inter_B_eq_15_l143_143891

def A : Set ℕ := {4, 5, 6}
def B : Set ℕ := {1, 2, 3}
def A_inter B : Set ℕ := {x | ∃ m n, m ∈ A ∧ n ∈ B ∧ x = m - n}

theorem sum_A_inter_B_eq_15 : (∑ x in A_inter B, x) = 15 := sorry

end sum_A_inter_B_eq_15_l143_143891


namespace sum_of_divisors_of_24_l143_143281

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range (24 + 1))), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l143_143281


namespace lateral_surface_area_prism_l143_143883

def side_length_rhombus : ℝ := 2
def side_diagonal_length : ℝ := 2 * Real.sqrt 3
def lateral_edge : ℝ := Real.sqrt((2 * Real.sqrt 3)^2 - (2)^2)

theorem lateral_surface_area_prism : 
  let S := 4 * side_length_rhombus * lateral_edge 
  in S = 16 * Real.sqrt 2 :=
by
  let side_length := side_length_rhombus
  let diagonal := side_diagonal_length
  let height := Real.sqrt((diagonal)^2 - (side_length)^2)
  let S := 4 * side_length * height
  have h1 : height = 2 * Real.sqrt 2 := by sorry -- Following the computation from the solution
  rw h1
  have h2 : S = 16 * Real.sqrt 2 := by sorry
  exact h2

end lateral_surface_area_prism_l143_143883


namespace problem_statement_l143_143209

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 :=
sorry

end problem_statement_l143_143209


namespace petya_wins_second_race_l143_143200

theorem petya_wins_second_race 
  (v_P v_V : ℝ) -- Petya's and Vasya's speeds
  (h1 : v_V = 0.9 * v_P) -- Condition from the first race
  (d_P d_V : ℝ) -- Distances covered by Petya and Vasya in the first race
  (h2 : d_P = 100) -- Petya covers 100 meters
  (h3 : d_V = 90) -- Vasya covers 90 meters
  (start_diff : ℝ) -- Initial distance difference in the second race
  (h4 : start_diff = 10) -- Petya starts 10 meters behind Vasya
  (race_length : ℝ) -- Total race length
  (h5 : race_length = 100) -- The race is 100 meters long
  : (v_P * (race_length / v_P) - v_V * (race_length / v_P)) = 1 :=
by
  sorry

end petya_wins_second_race_l143_143200


namespace marble_problem_l143_143359

theorem marble_problem:
  (∀ marbles : ℕ, marbles = 100 → 
  (∀ white_marble_percent : ℝ, white_marble_percent = 0.20 → 
  (∀ total_earnings : ℝ, total_earnings = 14 → 
  (∀ earnings_per_white_marble : ℝ, earnings_per_white_marble = 0.05 → 
  (∀ earnings_per_black_marble : ℝ, earnings_per_black_marble = 0.10 → 
  (∀ earnings_per_colored_marble : ℝ, earnings_per_colored_marble = 0.20 → 
  (∀ (B : ℝ), 
  1.00 + 10 * B + 20 * (0.80 - B) = 14 → 
  B = 0.30))))))

end marble_problem_l143_143359


namespace zero_of_f_l143_143580

noncomputable def f (x : ℝ) : ℝ := 2^x - 4

theorem zero_of_f : f 2 = 0 :=
by
  sorry

end zero_of_f_l143_143580


namespace twentieth_number_base_5_l143_143523

/-- The problem is to prove that the twentieth number in the base 5 number system is 40_5. -/
theorem twentieth_number_base_5 :
  let base := 5 in
  let twentieth_in_decimal := 20 in
  let twentieth_in_base_5 := "40" in
  convert_base_10_to_base_b twentieth_in_decimal base = some twentieth_in_base_5 :=
by
  sorry

end twentieth_number_base_5_l143_143523


namespace tan_2alpha_eq_neg_4_over_3_l143_143069

theorem tan_2alpha_eq_neg_4_over_3 (α : ℝ) 
  (h_parallel : (sin α) / (cos α) = 2) 
  : tan (2 * α) = -4 / 3 :=
sorry

end tan_2alpha_eq_neg_4_over_3_l143_143069


namespace zero_sum_of_squares_l143_143602

theorem zero_sum_of_squares {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_l143_143602


namespace evaluate_g_at_4_l143_143492

def g (x : ℕ) : ℕ := 5 * x - 2

theorem evaluate_g_at_4 : g 4 = 18 := by
  sorry

end evaluate_g_at_4_l143_143492


namespace required_integer_l143_143719

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 + d2 + d3 + d4 = sum

def middle_digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  d2 + d3 = sum

def thousands_minus_units (n : ℕ) (diff : ℕ) : Prop :=
  let d1 := n / 1000
  let d4 := n % 10
  d1 - d4 = diff

def divisible_by (n : ℕ) (d : ℕ) : Prop :=
  n % d = 0

theorem required_integer : 
  ∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    digits_sum_to n 18 ∧ 
    middle_digits_sum_to n 9 ∧ 
    thousands_minus_units n 3 ∧ 
    divisible_by n 9 ∧ 
    n = 6453 :=
by
  sorry

end required_integer_l143_143719


namespace rationalize_denominator_l143_143657

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143657


namespace number_of_solutions_in_interval_l143_143417

noncomputable def equation_to_solve (x : Real) : Real :=
  Real.tan(2 * x) + Real.tan(x)^2 + Real.sin(x)^3 + Real.sin(2 * x)^4

theorem number_of_solutions_in_interval :
  let num_solutions : ℕ := sorry in
  num_solutions = 3 :=
by
  -- Define the interval
  let interval : Set Real := {x | -Real.pi ≤ x ∧ x ≤ Real.pi}
  -- Define the equation
  have equation_satisfied : ∀ x ∈ interval, equation_to_solve x = 0 := sorry
  -- Insert the result of numerical or graphical analysis
  have num_solutions : ℕ := sorry
  -- Prove the number of solutions within the interval
  exact sorry

end number_of_solutions_in_interval_l143_143417


namespace log_base_half_of_32_l143_143838

theorem log_base_half_of_32 :
  ∀ x : ℝ, ( (1 / 2) ^ x = 32 ) → x = -5 :=
by
  intro x h
  have h1 : (1 : ℝ) / 2 = 2⁻¹ := by norm_num
  have h2 : (2⁻¹) ^ x = 32 := by rw [h1, h]
  have h3 : (2⁻¹) ^ x = 2 ^ 5 := by rw [h2, ←eq_comm]
  have h4 : 2 ^ (-x) = 2 ^ 5 := by rw [pow_neg, h3]
  have h5 : -x = 5 := pow_inj (exp_pos 2) h4
  exact eq_neg_iff_eq_neg.mp (eq.symm h5)

end log_base_half_of_32_l143_143838


namespace heptagon_not_octagon_slice_l143_143809

theorem heptagon_not_octagon_slice (H : heptagon) : 
  ¬ ∃ (s : slice), is_octagon s := 
sorry

end heptagon_not_octagon_slice_l143_143809


namespace sum_of_digits_of_N_l143_143390

-- Define the sequence
def sequence_500 : list ℕ := list.map (λ n, 10^n - 1) (list.range 500)

-- Define N as the sum of the sequence
def N : ℕ := list.sum sequence_500

-- State the theorem: the sum of the digits of N is 4477
theorem sum_of_digits_of_N : (list.sum (digits N) = 4477) :=
sorry

end sum_of_digits_of_N_l143_143390


namespace evaluate_expression_l143_143405

theorem evaluate_expression : (-1 : ℤ)^(3^3) + (1 : ℤ)^(3^3) = 0 := 
by
  sorry

end evaluate_expression_l143_143405


namespace rationalize_sqrt_fraction_l143_143672

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143672


namespace shaded_area_l143_143596

theorem shaded_area (A1 A2 A3 A4 : ℕ) (h1 : A1 = 20) (h2 : A2 = 40) (h3 : A3 = 48) (h4 : A4 = 42) : 
  A1 + A2 + A3 + A4 = 150 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end shaded_area_l143_143596


namespace roots_quadratic_sum_l143_143115

theorem roots_quadratic_sum (a b : ℝ) (h1 : (-2) + (-(1/4)) = -b/a)
  (h2 : -2 * (-(1/4)) = -2/a) : a + b = -13 := by
  sorry

end roots_quadratic_sum_l143_143115


namespace next_perfect_square_l143_143733

theorem next_perfect_square (x : ℤ) (h : ∃ k : ℤ, x = k^2) : ∃ z : ℤ, z = x + 2 * Int.sqrt x + 1 :=
by
  sorry

end next_perfect_square_l143_143733


namespace incorrect_statement_D_l143_143305

theorem incorrect_statement_D :
  (∀ (x : ℝ) (y : ℝ) (m : ℝ) (n : ℝ),
     (¬ (∃ c : ℝ, c * (π * (m * n) / 5) = (π * (m * n) / 5) ∧ c = 1/5)) :=
begin
  assume x y m n,
  intro h,
  cases h with c hc,
  have h_coeff : c = π / 5,
  { rw [hc.1] at hc,
    sorry },
  have : c ≠ 1/5,
  { sorry },
  exact this hc.2,
end

end incorrect_statement_D_l143_143305


namespace purple_chip_value_l143_143520

theorem purple_chip_value 
  (x : ℕ)
  (blue_chip_value : 1 = 1)
  (green_chip_value : 5 = 5)
  (red_chip_value : 11 = 11)
  (purple_chip_condition1 : x > 5)
  (purple_chip_condition2 : x < 11)
  (product_of_points : ∃ b g p r, (b = 1 ∨ b = 1) ∧ (g = 5 ∨ g = 5) ∧ (p = x ∨ p = x) ∧ (r = 11 ∨ r = 11) ∧ b * g * p * r = 28160) : 
  x = 7 :=
sorry

end purple_chip_value_l143_143520


namespace power_function_m_eq_4_l143_143457

theorem power_function_m_eq_4 (m : ℝ) :
  (m^2 - 3*m - 3 = 1) → m = 4 :=
by
  sorry

end power_function_m_eq_4_l143_143457


namespace distinct_bead_arrangements_l143_143132

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n-1)

theorem distinct_bead_arrangements : factorial 8 / (8 * 2) = 2520 := 
  by sorry

end distinct_bead_arrangements_l143_143132


namespace ellipse_and_parabola_equation_line_ap_equation_l143_143582

variable (a b c p : ℝ)
variable (F A P Q B D : Real × Real)
variable (m : ℝ)

-- Conditions
axiom ellipse_eq : a > b ∧ b > 0 ∧ (F.1, F.2, A.1, A.2, P.1, P.2, Q.1, Q.2, B.1, B.2, D.1, D.2)
axiom eccentricity : c / a = 1/2
axiom parabola_focus : A.1 ^ 2 = 2 * p * A.1
axiom focus_directrix_distance : F.1 + 1/2 = 0 -- x = -1 is directrix
axiom symmetric_points : (P.1 = -1 ∧ Q.1 = -1) ∧ (-P.2 = Q.2)
axiom AP_intersection : B ≠ A ∧ (∃ m, P.2 = -2 / m ∧ Q.2 = 2 / m)
axiom BQ_intersection : ∃ y, y = 0 ∧ (P.1, P.2) = ((2 - 3 * m^2)/(3 * m^2 + 2), 0)
axiom triangle_area : (1/2) * (6 * m^2 / (3 * m^2 + 2) * 2 / m) = √6/2

-- Proof
theorem ellipse_and_parabola_equation :
  (eccentricity ∧ parabola_focus ∧ focus_directrix_distance ∧ symmetric_points ∧ AP_intersection ∧ BQ_intersection ∧ triangle_area) →
    (x^2 + 4*y^2/3 = 1 ∧ y^2 = 4*x) :=
sorry

theorem line_ap_equation :
  (eccentricity ∧ parabola_focus ∧ focus_directrix_distance ∧ symmetric_points ∧ AP_intersection ∧ BQ_intersection ∧ triangle_area) →
    (3*x + √6*y - 3 = 0 ∨ 3*x - √6*y - 3 = 0) :=
sorry

end ellipse_and_parabola_equation_line_ap_equation_l143_143582


namespace student_rewards_l143_143123

theorem student_rewards :
  ∃ a d : ℝ, 
    (2 * a - 6 * d = 40) ∧ 
    (2 * a - 13 * d = 30) ∧ 
    (a - 2 * d = 21.4286) ∧
    (a - 4 * d = 18.5714) ∧
    (a - 5 * d = 17.1429) ∧
    (a - 8 * d = 12.8571) :=
begin
  sorry
end

end student_rewards_l143_143123


namespace rationalize_sqrt_fraction_l143_143617

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143617


namespace find_x_l143_143845

theorem find_x (x : ℕ) (a : ℕ) (h₁: a = 450) (h₂: (15^x * 8^3) / 256 = a) : x = 2 :=
by
  sorry

end find_x_l143_143845


namespace curve_represents_two_lines_l143_143225

theorem curve_represents_two_lines : ∀ x y : ℝ, x^2 + x * y = x ↔ (x = 0 ∨ x + y - 1 = 0) := 
begin
  intros x y,
  split,
  { intro h,
    -- Show that x^2 + xy = x implies x = 0 or x + y = 1
    rw [←eq_sub_iff_add_eq, mul_comm] at h,
    factor h,  -- factor the polynomial
    },
  { rintros (hx | hy),
    { rw hx,
      ring, },
    { rw [←eq_sub_of_add_eq hy, mul_comm],
      ring, }
  }
end

end curve_represents_two_lines_l143_143225


namespace area_shaded_smaller_dodecagon_area_in_circle_l143_143311

-- Part (a) statement
theorem area_shaded_smaller (dodecagon_area : ℝ) (shaded_area : ℝ) 
  (h : shaded_area = (1 / 12) * dodecagon_area) :
  shaded_area = dodecagon_area / 12 :=
sorry

-- Part (b) statement
theorem dodecagon_area_in_circle (r : ℝ) (A : ℝ) 
  (h : r = 1) (h' : A = (1 / 2) * 12 * r ^ 2 * Real.sin (2 * Real.pi / 12)) :
  A = 3 :=
sorry

end area_shaded_smaller_dodecagon_area_in_circle_l143_143311


namespace harmonic_terminating_sum_l143_143393

noncomputable def harmonic (n : ℕ) : ℚ :=
  (List.range n).map (λ i, (1 : ℚ) / (i + 1)).sum

def is_terminating (q : ℚ) : Prop :=
  q.denom.gcd(2 ^ (q.denom.log2 + 1)) = q.denom

def sum_terminating_harmonics : ℚ :=
  (List.range 6).filter (λ n, is_terminating (harmonic (n + 1))).map (λ n, harmonic (n + 1)).sum

theorem harmonic_terminating_sum :
  let S := sum_terminating_harmonics in
  ∃ (m n : ℕ), S = m / n ∧ nat.coprime m n ∧ 100 * m + n = 9920 :=
by
  let S := sum_terminating_harmonics
  existsi 99
  existsi 20
  split
  {exact sorry}
  split
  {exact sorry}
  {exact sorry}

end harmonic_terminating_sum_l143_143393


namespace complement_set_l143_143085

open Set

variable (x : ℝ)

def U : Set ℝ := @univ ℝ _
def A : Set ℝ := { x | (x + 1) / (x - 2) ≤ 0 }
def complement_U_A : Set ℝ := U \ A

theorem complement_set :
  complement_U_A = { x | x < -1 ∨ x ≥ 2 } :=
by
  sorry

end complement_set_l143_143085


namespace rationalize_denominator_l143_143696

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143696


namespace petya_second_race_finishes_first_l143_143198

variable (t v_P v_V : ℝ)
variable (h1 : v_P * t = 100)
variable (h2 : v_V * t = 90)
variable (d : ℝ)

theorem petya_second_race_finishes_first :
  v_V = 0.9 * v_P ∧
  d * v_P = 10 + d * (0.9 * v_P) →
  ∃ t2 : ℝ, t2 = 100 / v_P ∧ (v_V * t2 = 90) →
  ∃ t3 : ℝ, t3 = t2 + d / 10 ∧ (d * v_P = 100) →
  v_P * d / 10 - v_V * d / 10 = 1 :=
by
  sorry

end petya_second_race_finishes_first_l143_143198


namespace michael_sixth_time_l143_143590

theorem michael_sixth_time (a1 a2 a3 a4 a5 : ℕ) (m : ℕ) :
  a1 = 102 → a2 = 104 → a3 = 110 → a4 = 116 → a5 = 118 → m = 109 →
  ∃ x : ℕ, 
    let sorted_times := list.sort (≤) [a1, a2, x, a3, a4, a5] in
    (sorted_times.nth 2).get_or_else 0 + (sorted_times.nth 3).get_or_else 0 = 2 * m ∧
    x = 108 :=
by
  intros h1 h2 h3 h4 h5 hm
  existsi 108
  have : list.sort (≤) [102, 104, 108, 110, 116, 118] = [102, 104, 108, 110, 116, 118], from sorry
  split
  · rw this
    simp
  · refl

end michael_sixth_time_l143_143590


namespace digital_root_eq_one_iff_all_digits_eq_one_l143_143942

-- Define the sequence as per the problem statement
def sequence (a : Nat) : List Nat :=
  List.iterate (λ x, if x < 10 then x else List.foldl (*) 1 (Nat.digits 10 x)) a

-- Digital root definition as per the problem statement
def digitalRoot (a : Nat) : Nat :=
  List.head! (sequence a).reverse

-- Theorem to be proved: The digital root of a positive integer n equals 1 
-- if and only if all the digits of n are 1.
theorem digital_root_eq_one_iff_all_digits_eq_one (n : Nat) (h : n > 0) : 
  digitalRoot n = 1 ↔ ∀ d ∈ Nat.digits 10 n, d = 1 :=
by
  sorry

end digital_root_eq_one_iff_all_digits_eq_one_l143_143942


namespace range_of_a_l143_143928

variable (a : ℝ)

def proposition_p (a : ℝ) : Prop := 0 < a ∧ a < 1

def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + a > 0 ∧ 1 - 4 * a^2 < 0

theorem range_of_a : (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
  (0 < a ∧ a ≤ 1/2 ∨ a ≥ 1) := 
by
  sorry

end range_of_a_l143_143928


namespace value_calculation_l143_143734

theorem value_calculation :
  6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 :=
by
  sorry

end value_calculation_l143_143734


namespace triangle_ABC_angles_l143_143318

theorem triangle_ABC_angles 
(points_mid_arcs : ∀ {A B C : Point} {O : Circle} {X Y : Point}, 
  midpoint O A B X ∧ midpoint O B C Y) 
(angle_ABC_2_ACB : ∀ {A B C : Point}, ∠ABC = 2 * ∠ACB)
(angle_bisector_BL : ∀ {A B C : Point} {L : Point}, bisects B A C L)
(angle_XLY_90 : ∀ {X L Y : Point}, ∠XLY = 90) :
  ∃ {A B C : Point}, ∠A = 45 ∧ ∠B = 90 ∧ ∠C = 45 := by 
  sorry

end triangle_ABC_angles_l143_143318


namespace parabola_point_min_distance_l143_143907

theorem parabola_point_min_distance (m n : ℝ) (h : n = - (1 / 4) * m^2) :
  ∃ P : ℝ, (∀ p q : ℝ, P = sqrt(m^2 + (n + 1)^2) + sqrt((m - 4)^2 + (n + 5)^2) → P = 6) := 
sorry

end parabola_point_min_distance_l143_143907


namespace cosine_of_angle_BHD_l143_143136

-- Definitions of the given conditions
def angle_DHG := 30
def angle_FHB := 60
def CD_value := 1

-- The main theorem stating the result
theorem cosine_of_angle_BHD (h1 : angle_DHG = 30) (h2 : angle_FHB = 60) (h3 : CD = CD_value) :
  cos(angle BHD) = sqrt(6) / 8 :=
sorry

end cosine_of_angle_BHD_l143_143136


namespace sequence_positive_and_divisible_l143_143247

theorem sequence_positive_and_divisible:
  ∃ (a : ℕ → ℕ), 
    (a 1 = 2) ∧ (a 2 = 500) ∧ (a 3 = 2000) ∧ 
    (∀ n ≥ 2, (a (n + 2) + a (n + 1)) * a (n - 1) = a (n + 1) * (a (n + 1) + a (n - 1))) ∧ 
    (∀ n, a n > 0) ∧ 
    (2 ^ 2000 ∣ a 2000) := 
sorry

end sequence_positive_and_divisible_l143_143247


namespace arithmetic_mean_of_primes_l143_143011

theorem arithmetic_mean_of_primes (l : List ℕ) (p1 : l = [21, 23, 25, 27, 29]) :
  let primes := (l.filter Nat.Prime) in
  primes = [23, 29] ∧ (primes.sum) / primes.length = 26 :=
by
  sorry

end arithmetic_mean_of_primes_l143_143011


namespace seq_general_formula_seq_sum_l143_143046

variable {n : ℕ}

/- Condition: the sequence {a_n} has sum S_n, and sqrt(S_n) is the arithmetic mean of 1 and a_n -/
axiom S_n (n : ℕ) : ℝ
axiom a_n (n : ℕ) : ℝ
axiom mean_condition (n : ℕ) : Real.sqrt (S_n n) = (a_n n + 1) / 2

/- Task (I): Prove that a_n = 2n - 1 -/
theorem seq_general_formula (n : ℕ) (a_n S_n : ℕ → ℝ)
  (mean_condition : ∀ n, Real.sqrt (S_n n) = (a_n n + 1) / 2) :
  a_n n = 2 * n - 1 :=
sorry

/- Task (II): Given a_n = 2n - 1, prove T_n = 1 - 1/(2n + 1) -/
def T_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in Finset.range n, 2 / (a_n i * a_n (i + 1))

theorem seq_sum (n : ℕ) (a_n : ℕ → ℝ) (h : ∀ n, a_n n = 2 * n - 1) :
  T_n a_n n = 1 - 1 / (2 * n + 1) :=
sorry

end seq_general_formula_seq_sum_l143_143046


namespace sin_square_alpha_minus_pi_div_4_l143_143445

theorem sin_square_alpha_minus_pi_div_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α - Real.pi / 4) ^ 2 = 1 / 6 := 
sorry

end sin_square_alpha_minus_pi_div_4_l143_143445


namespace sum_of_divisors_of_24_eq_60_l143_143298

theorem sum_of_divisors_of_24_eq_60 : 
  (∑ n in { n | n ∣ 24 ∧ 0 < n }.toFinset, n) = 60 := 
sorry

end sum_of_divisors_of_24_eq_60_l143_143298


namespace sum_a1_a3_a5_l143_143922

open Nat

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 16 ∧ ∀ n > 0, a (n + 1) = a n / 2

theorem sum_a1_a3_a5 :
  ∃ (a : ℕ → ℝ), sequence a ∧ (a 1 + a 3 + a 5 = 21) :=
by
  sorry

end sum_a1_a3_a5_l143_143922


namespace sqrt8_sub_sqrt2_eq_sqrt2_l143_143816

theorem sqrt8_sub_sqrt2_eq_sqrt2 : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt8_sub_sqrt2_eq_sqrt2_l143_143816


namespace tangent_length_correct_l143_143794

theorem tangent_length_correct :
  let P := (1 : ℝ, -2 : ℝ)
  let center := (-1 : ℝ, 1 : ℝ)
  let radius := 2 : ℝ
  let distance_P_C := Real.sqrt ((1 + 1)^2 + (-2 - 1)^2)
  let tangent_length := Real.sqrt (distance_P_C^2 - radius^2)
  tangent_length = 3 :=
by
  let P := (1, -2)
  let center := (-1, 1)
  let radius := 2
  let distance_P_C := Real.sqrt ((1 + 1)^2 + (-2 - 1)^2)
  let tangent_length := Real.sqrt (distance_P_C^2 - radius^2)
  sorry

end tangent_length_correct_l143_143794


namespace paper_clips_distribution_l143_143584

theorem paper_clips_distribution (P c b : ℕ) (hP : P = 81) (hc : c = 9) (hb : b = P / c) : b = 9 :=
by
  rw [hP, hc] at hb
  simp at hb
  exact hb

end paper_clips_distribution_l143_143584


namespace cards_exchanged_l143_143371

theorem cards_exchanged (x : ℕ) (h : x * (x - 1) = 1980) : x * (x - 1) = 1980 :=
by sorry

end cards_exchanged_l143_143371


namespace find_length_FD_l143_143138

-- Define the paper with a square ABCD of side length 8 cm
def isSquare (A B C D : Point) : Prop :=
  dist A B = 8 ∧ dist B C = 8 ∧ dist C D = 8 ∧ dist D A = 8 ∧ ∠ A B C = 90 ∧ ∠ B C D = 90 ∧ ∠ C D A = 90 ∧ ∠ D A B = 90

-- Define E one-third of the way along AD from D towards A
def oneThirdAD (A D E : Point) : Prop :=
  dist D E = 8 / 3 ∧ dist A E = 16 / 3

-- Define FD as the length we need to find
def lengthFD (F D : Point) (x : ℝ) : Prop :=
  dist F D = x

-- Proof problem statement
theorem find_length_FD (A B C D E F G : Point)
  (hSquare : isSquare A B C D)
  (hOneThirdAD : oneThirdAD A D E)
  (hCrease : F ∈ line_CD) :
  ∃ x : ℝ, lengthFD F D (32 / 9) :=
by
  sorry

end find_length_FD_l143_143138


namespace trapezoid_height_ratios_l143_143235

theorem trapezoid_height_ratios (A B C D O M N K L : ℝ) (h : ℝ) (h_AD : D = 2 * B) 
  (h_OK : K = h / 3) (h_OL : L = (2 * h) / 3) :
  (K / h = 1 / 3) ∧ (L / h = 2 / 3) := by
  sorry

end trapezoid_height_ratios_l143_143235


namespace number_of_distinct_slopes_l143_143212

open Finset

-- We define the set of available numbers
def S : Finset ℕ := {1, 3, 5, 7, 9}

-- We define the concept of distinct pairs from elements of the set
def distinctPairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  filter (λ p, p.1 ≠ p.2) (s.product s)

-- We define the slope function
def slope (a b : ℕ) : ℚ := -(a : ℚ) / b

-- We define a set of all possible slopes
def allSlopes : Finset ℚ :=
  distinctPairs S |>.image (λ p, slope p.1 p.2)

-- The final proof statement
theorem number_of_distinct_slopes : allSlopes.card = 18 :=
by
  sorry

end number_of_distinct_slopes_l143_143212


namespace infinitely_many_primes_congruent_3_mod_4_l143_143710

def is_congruent_3_mod_4 (p : ℕ) : Prop :=
  p % 4 = 3

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

def S (p : ℕ) : Prop :=
  is_prime p ∧ is_congruent_3_mod_4 p

theorem infinitely_many_primes_congruent_3_mod_4 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ S p :=
sorry

end infinitely_many_primes_congruent_3_mod_4_l143_143710


namespace locus_of_M_l143_143203

-- Let's define the required entities first
variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] {k : ℝ} {C : α}
variables (A B : α) (r : ℝ) (hc : ∀ P : α, dist C P = r)
variables (M : α)

def midpoint (X Y : α) : α :=
(X + Y) / 2

-- Assume A lies on the given circle
def on_circle (A : α) : Prop :=
dist C A = r

-- The main theorem to be proved
theorem locus_of_M (hA : on_circle A) :
  ∃ (B : α), dist B A = r ∧ ∀ M, dist B M = r ∧ M ≠ A → dist C ((A + M) / 2) = r :=
sorry

end locus_of_M_l143_143203


namespace compareNumbers_l143_143314

theorem compareNumbers : 0.5 - (1 / 6) = (1 / 3) := 
by 
    have h : (1 : ℝ) / 6 = 0.16666666666666666 := by norm_num
    have h' : (0.5 : ℝ) - 0.16666666666666666 = 0.3333333333333333 := by norm_num
    have h'' : 0.3333333333333333 = 1 / 3 := by norm_num
    exact eq.trans (eq.trans h') h''

end compareNumbers_l143_143314


namespace mass_percentage_C_in_H2CO3_l143_143842

theorem mass_percentage_C_in_H2CO3 
  (H_mass : Float := 1.008) 
  (C_mass : Float := 12.01) 
  (O_mass : Float := 16.00) 
  (mass_percentage : Float := (C_mass / (2 * H_mass + C_mass + 3 * O_mass) * 100)) : 
  mass_percentage ≈ 19.36 := 
by 
  -- The proof will go here
  sorry

end mass_percentage_C_in_H2CO3_l143_143842


namespace sufficient_to_perpendicular_l143_143900

variable {Point : Type} [TopologicalSpace Point]

structure Line (P : Type) := 
  (contains : P → Prop)
  (exists_two_points : ∃ a b, a ≠ b ∧ contains a ∧ contains b)

structure Plane (P : Type) :=
  (contains : P → Prop)
  (exists_three_points : ∃ a b c, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ contains a ∧ contains b ∧ contains c)

structure Geometry :=
  (perpendicular : Line Point → Plane Point → Prop)
  (parallel : Line Point → Line Point → Prop)

variable [G : Geometry]

open Geometry

variables {m n : Line Point} {α β : Plane Point}

theorem sufficient_to_perpendicular (h1: G.parallel m n) (h2: G.perpendicular n β) : G.perpendicular m β :=
sorry

end sufficient_to_perpendicular_l143_143900


namespace point_transformation_l143_143065

theorem point_transformation : ∀ (P : ℝ×ℝ), P = (1, -2) → P = (-1, 2) :=
by
  sorry

end point_transformation_l143_143065


namespace min_sum_of_labels_l143_143332

theorem min_sum_of_labels : 
  ∃ (r : Fin 8 → Fin 8), (∀ i j, i ≠ j → r i ≠ r j) ∧ (∑ i, 1 / (i + 1 : ℕ) + (r i).val + 1) = 8 / 9 :=
by
  sorry

end min_sum_of_labels_l143_143332


namespace nat_exists_six_divisors_and_sum_3500_l143_143409

theorem nat_exists_six_divisors_and_sum_3500 (n : ℕ) : 
  (count_divisors n = 6 ∧ sum_divisors n = 3500) ↔ n = 1996 :=
sorry

end nat_exists_six_divisors_and_sum_3500_l143_143409


namespace arrangement_problem_l143_143834


open Function

noncomputable def valid_arrangements_count : ℕ :=
  let S := {σ : Fin 10 → Fin 10 | 
    (∀ i, 1 ≤ i ∧ i ≤ 5 → σ i > σ (2*i)) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 4 → σ i > σ (2*i + 1)) } in
  Fintype.card S

theorem arrangement_problem : valid_arrangements_count = 840 := 
  by
  sorry

end arrangement_problem_l143_143834


namespace translate_method_for_intersecting_xaxis_and_distance_one_l143_143540

theorem translate_method_for_intersecting_xaxis_and_distance_one :
  ∃ t : ℝ, t = -4 ∧
  ∀ x y : ℝ, y = (x - 2009) * (x - 2008) + 4 ∧ y - t = 0 →
  abs ((2009) - (2008)) = 1 :=
by
  sorry

end translate_method_for_intersecting_xaxis_and_distance_one_l143_143540


namespace number_of_correct_conclusions_l143_143071

theorem number_of_correct_conclusions {m n : ℝ} (h_curve : m * x^2 + n * y^2 = 1) :
    ((m > n ∧ n > 0 ∧ is_ellipse_with_foci_on_y_axis m n) ∨
     (m = n ∧ n > 0 ∧ is_circle_with_radius_sqrt_n m n) ∨
     (m * n < 0 ∧ is_hyperbola_with_asymptotes m n) ∨
     (m = 0 ∧ n > 0 ∧ consists_two_straight_lines n)) = 3 := 
    sorry

end number_of_correct_conclusions_l143_143071


namespace four_digit_integers_with_1_3_7_are_81_l143_143937

theorem four_digit_integers_with_1_3_7_are_81 : 
  ∃ n : ℕ, n = 81 ∧ ∀ x : ℕ, x ∈ { d | ∀ i < 4, d.digit_at i ∈ {1, 3, 7} } → x > 999 ∧ x < 10000 :=
begin
  sorry
end

end four_digit_integers_with_1_3_7_are_81_l143_143937


namespace perp_condition_l143_143154

variables {ω1 ω2 : Type*} [circle ω1] [circle ω2]
variables {A1 A2 B1 B2 : Type*}
variables {is_tangent : (Type* → Type* → Prop)}
variables {internal_tangent : is_tangent A1 ω1 ∧ is_tangent A2 ω2}
variables {external_tangent : is_tangent B1 ω1 ∧ is_tangent B2 ω2}
variables {are_non_intersecting : ¬(intersect ω1 ω2)}
variables {A1B2_eq_A2B1 : dist A1 B2 = dist A2 B1}

theorem perp_condition : A1B2_eq_A2B1 → is_perpendicular (A1, B2) (A2, B1) :=
sorry

end perp_condition_l143_143154


namespace platform_length_l143_143762

noncomputable def train_length := 420 -- length of the train in meters
noncomputable def time_to_cross_platform := 60 -- time to cross the platform in seconds
noncomputable def time_to_cross_pole := 30 -- time to cross the signal pole in seconds

theorem platform_length :
  ∃ L, L = 420 ∧ train_length / time_to_cross_pole = train_length / time_to_cross_platform * (train_length + L) / time_to_cross_platform :=
by
  use 420
  sorry

end platform_length_l143_143762


namespace point_on_circle_l143_143966

noncomputable def distance_from_origin (x : ℝ) (y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem point_on_circle : distance_from_origin (-3) 4 = 5 := by
  sorry

end point_on_circle_l143_143966


namespace falling_body_time_l143_143089

theorem falling_body_time (g : ℝ) (h_g : g = 9.808) (d : ℝ) (t1 : ℝ) (h_d : d = 49.34) (h_t1 : t1 = 1.3) : 
  ∃ t : ℝ, (1 / 2 * g * (t + t1)^2 - 1 / 2 * g * t^2 = d) → t = 7.088 :=
by 
  use 7.088
  intros h
  sorry

end falling_body_time_l143_143089


namespace rationalize_sqrt_fraction_denom_l143_143611

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143611


namespace marvin_arvin_ratio_l143_143587

theorem marvin_arvin_ratio :
  ∃ M : ℕ, 
  let ratio := M / 40 in 
  (40 + M + 2 * 40 + 2 * M = 480) ∧
  ratio = 3 :=
begin
  -- Existence of M
  use 120,
  dsimp only,
  split,
  -- Equation condition
  {
    linarith,
  },
  -- Ratio condition
  {
    exact (120 / 40 : ℕ) = 3,
  },
end

end marvin_arvin_ratio_l143_143587


namespace rationalize_sqrt_fraction_l143_143667

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143667


namespace tilings_remainder_l143_143327

def num_ways_partitioned_tiling_with_colors
  (edges parts : ℕ) : ℕ :=
  Nat.choose edges (parts - 1)

noncomputable def pow_sum (x y : ℕ) (k : ℕ) : ℕ :=
  x^k - 3 * (y^k) + 3

theorem tilings_remainder (k : ℕ) (H : k = 9) :
  let tiles := [3, 4, 5, 6, 7, 8, 9] in
  let ways := List.map (num_ways_partitioned_tiling_with_colors 8) tiles in
  let colorings := List.map (λ k, pow_sum 3 2 k) tiles in
  let products := List.zipWith (λ a b, a * b) ways colorings in
  let N := List.sum products in
  N % 1000 = 838 := 
by
  sorry

end tilings_remainder_l143_143327


namespace inequality_proof_l143_143037

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
    a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by
  sorry

end inequality_proof_l143_143037


namespace rationalize_sqrt_fraction_l143_143622

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143622


namespace sum_of_divisors_of_24_eq_60_l143_143299

theorem sum_of_divisors_of_24_eq_60 : 
  (∑ n in { n | n ∣ 24 ∧ 0 < n }.toFinset, n) = 60 := 
sorry

end sum_of_divisors_of_24_eq_60_l143_143299


namespace circle_graph_rectangle_representation_l143_143042

theorem circle_graph_rectangle_representation (x : ℕ) :
  let White := 6 * x
  let Gray := 2 * x
  let Black := x
  let Blue := x
  (White = 6 ∧ Gray = 2 ∧ Black = 1 ∧ Blue = 1) ↔ option_A (White Gray Black Blue) :=
by
  -- proof steps will be here
  sorry

end circle_graph_rectangle_representation_l143_143042


namespace trajectory_of_midpoint_l143_143066

theorem trajectory_of_midpoint (x y : ℝ) (A B : ℝ × ℝ) 
  (hB : B = (4, 0)) (hA_on_circle : (A.1)^2 + (A.2)^2 = 4)
  (hM : ((x, y) = ( (A.1 + B.1)/2, (A.2 + B.2)/2))) :
  (x - 2)^2 + y^2 = 1 :=
sorry

end trajectory_of_midpoint_l143_143066


namespace sum_of_divisors_of_24_l143_143288

theorem sum_of_divisors_of_24 : ∑ d in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), d = 60 := by
  sorry

end sum_of_divisors_of_24_l143_143288


namespace find_b_compare_f_l143_143081

-- Definition from conditions
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Prove that b = 4
theorem find_b (b c : ℝ) (h : ∀ x : ℝ, f (2 + x) b c = f (2 - x) b c) : b = 4 :=
sorry

-- Part 2: Prove the comparison of f(\frac{5}{4}) and f(-a^2 - a + 1)
theorem compare_f (c : ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (2 + x) 4 c = f (2 - x) 4 c) (h₂ : f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c) :
f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c := 
sorry

end find_b_compare_f_l143_143081


namespace part1_distance_part2_equation_l143_143467

noncomputable section

-- Define the conditions for Part 1
def hyperbola_C1 (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = 1

-- Define the point M(3, t) existing on hyperbola C₁
def point_on_hyperbola_C1 (t : ℝ) : Prop := hyperbola_C1 3 t

-- Define the right focus of hyperbola C1
def right_focus_C1 : ℝ × ℝ := (4, 0)

-- Part 1: Distance from point M to the right focus
theorem part1_distance (t : ℝ) (h : point_on_hyperbola_C1 t) :  
  let distance := Real.sqrt ((3 - 4)^2 + (t - 0)^2)
  distance = 4 := sorry

-- Define the conditions for Part 2
def hyperbola_C2 (x y : ℝ) (m : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = m

-- Define the point (-3, 2√6) existing on hyperbola C₂
def point_on_hyperbola_C2 (m : ℝ) : Prop := hyperbola_C2 (-3) (2 * Real.sqrt 6) m

-- Part 2: The standard equation of hyperbola C₂
theorem part2_equation (h : point_on_hyperbola_C2 (1/4)) : 
  ∀ (x y : ℝ), hyperbola_C2 x y (1/4) ↔ (x^2 - (y^2 / 3) = 1) := sorry

end part1_distance_part2_equation_l143_143467


namespace probability_of_longer_piece_l143_143351

noncomputable def probability_longer_piece_triple_shorter (C : ℝ) : ℝ :=
  if 0 ≤ C ∧ C ≤ 1 then
    if C ≤ 1 / 4 ∨ C ≥ 3 / 4 then 1 / 2 else 0
  else
    0

theorem probability_of_longer_piece (C : ℝ) (h1 : 0 ≤ C) (h2 : C ≤ 1) 
  (h3 : C ≤ 1 / 4 ∨ C ≥ 3 / 4 ∨ C ∈ Icc 1 3 / 4 ∪ Icc 3 / 4 1) :
  probability_longer_piece_triple_shorter C = 1 / 2 :=
  sorry

end probability_of_longer_piece_l143_143351


namespace correct_statements_l143_143758

-- Define the statements
def statement1 : Prop := ∀ (f : ℝ → ℝ), (det : ℝ → ℝ) → (∀ x, f x = det x)
def statement2 : Prop := ∀ (x y : ℝ), (residual_plot : ℝ × ℝ) → y = snd residual_plot
def statement3 : Prop := ¬(∀ (x y : ℝ), (corr : ℝ × ℝ) → (∀ x y, x = y))
def statement4 : Prop := complex.conj ⟨-1, 1⟩ = ⟨-1, -1⟩

-- Prove that statements 1, 2, and 4 are correct
theorem correct_statements : statement1 ∧ statement2 ∧ statement4 := by
  sorry

end correct_statements_l143_143758


namespace product_of_g_at_roots_l143_143573

noncomputable def f (x : ℝ) : ℝ := x^5 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 2
noncomputable def roots : List ℝ := sorry -- To indicate the list of roots x_1, x_2, x_3, x_4, x_5 of the polynomial f(x)

theorem product_of_g_at_roots :
  (roots.map g).prod = -23 := sorry

end product_of_g_at_roots_l143_143573


namespace problem_1_intervals_problem_2_triangle_l143_143480

def vec_m (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.sin x)
def vec_n (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
def f (x : ℝ) : ℝ := vec_m x.1 * vec_n x.1 + vec_m x.2 * vec_n x.2

theorem problem_1_intervals :
  -- Prove the intervals of monotonic increase for f(x) on [0, 3π/2] are [π/6, 2π/3] and [7π/6, 3π/2]

theorem problem_2_triangle (A a b c : ℝ) :
  (0 < A ∧ A < Real.pi / 2) ∧
  (f A + Real.sin (2 * A - Real.pi / 6) = 1) ∧
  (b + c = 7) ∧
  (1 / 2 * b * c * Real.sin A = 2 * Real.sqrt 3) →
  a = 5

end problem_1_intervals_problem_2_triangle_l143_143480


namespace sum_roots_inverses_l143_143564

noncomputable def polynomial_roots : set ℂ := {
  z | is_root (λ x => x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x - 680) z
}

noncomputable def problem_sum (roots : set ℂ) := ∑ z in roots, (1 / (1 - z))

theorem sum_roots_inverses (roots : set ℂ) (h : roots = polynomial_roots) :
    problem_sum roots = 55 / 679 :=
by {
  sorry
}

end sum_roots_inverses_l143_143564


namespace geometric_series_correct_statements_l143_143083

theorem geometric_series_correct_statements :
  let a := (3 : ℝ)
  let r := (1 / 4 : ℝ)
  let S := ∑' n, a * r^n
  ∃ L, is_limit S L ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, |a * r^n| < ε) :=
by
  let a := (3 : ℝ)
  let r := (1 / 4 : ℝ)
  let S := ∑' n, a * r^n
  let L := a / (1 - r)
  have h1 : is_limit S L := by sorry
  have h2 : (∀ ε > 0, ∃ N, ∀ n ≥ N, |a * r^n| < ε) := by sorry
  exact ⟨L, h1, h2⟩

end geometric_series_correct_statements_l143_143083


namespace expected_students_at_end_of_four_months_l143_143338

noncomputable def expected_students (initial : ℕ) : ℕ :=
let after_first_month := let new_students := initial * 30 / 100 in
                         let total_first := initial + new_students in
                         let drop_first := new_students * 25 / 100 in
                           total_first - drop_first in
let after_exchange_festival := let international := 5 in
                               let total_exchange := after_first_month + international in
                               let drop_exchange := international * 10 / 100 in
                                 total_exchange - drop_exchange in
let after_open_house := let new_open := after_exchange_festival * 25 / 100 in
                        let total_open := after_exchange_festival + new_open in
                        let drop_open := 3 in
                          total_open - drop_open in
let after_language_club_joins := let new_club := after_open_house * 50 / 100 in
                                 let total_club := after_open_house + new_club in
                                 let drop_club := new_club * 20 / 100 in
                                  total_club - drop_club in
let after_graduation_third_month := let graduated := after_language_club_joins * 2 / 3 in
                                    after_language_club_joins - graduated in
let after_fourth_month_campaign := let new_fourth := after_graduation_third_month * 20 / 100 in
                                   let total_fourth := after_graduation_third_month + new_fourth in
                                   let drop_fourth := new_fourth * 30 / 100 in
                                     total_fourth - drop_fourth in
let after_final_graduation := let graduated_final := after_fourth_month_campaign * 10 / 100 in
                              after_fourth_month_campaign - graduated_final in
  after_final_graduation

theorem expected_students_at_end_of_four_months : expected_students 8 = 6 :=
sorry

end expected_students_at_end_of_four_months_l143_143338


namespace sum_of_positive_divisors_of_24_l143_143294

theorem sum_of_positive_divisors_of_24 : ∑ n in {n : ℕ | n > 0 ∧ (n+24) % n = 0}, n = 60 := 
by sorry

end sum_of_positive_divisors_of_24_l143_143294


namespace convex_polygon_diagonals_l143_143781

theorem convex_polygon_diagonals (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → convex_poly.angle_interior n i = 150)
  (h2 : convex_poly.is_convex n) :
  convex_poly.num_diagonals n = 54 :=
sorry

end convex_polygon_diagonals_l143_143781


namespace area_triangle_LEF_l143_143545

variables
  (O L A B E F : Point)
  (circle_O : Circle O 10)
  (chord_EF : Chord O E F)
  (length_EF : length(E, F) = 12)
  (parallel_EF_LB : parallel line(E, F) line(L, B))
  (length_LA : length(L, A) = 20)
  (collinear_LAOB : collinear [L, A, O, B])
  (above_EF_AB : above_line(E, F) (line(A, B)))
  (LE : Segment L E)
  (LF : Segment L F)

theorem area_triangle_LEF : area (triangle L E F) = 48 := 
by
  sorry

end area_triangle_LEF_l143_143545


namespace sequence_condition_l143_143056

theorem sequence_condition
  (n m : ℕ)
  (h_nm : n > m)
  (i : ℕ → ℕ)
  (hi : ∀ j, 1 ≤ j → j ≤ m → 1 ≤ i j ∧ i j ≤ n ∧ (∀ k, j < k → k ≤ m → i j < i k))
  (x : ℕ → ℝ)
  (hx_sum : ∑ i in Finset.range (n + 1), x i = 0)
  (hx_order : ∀ j k, j < k → x j < x k)
  (hi_condition : ∀ j, 1 ≤ j → j ≤ m → i j ≥ (j * n / m)) :
  ∑ k in Finset.range (m + 1), x (i k) > 0 := by
  sorry

end sequence_condition_l143_143056


namespace AngeliCandies_l143_143367

def CandyProblem : Prop :=
  ∃ (C B G : ℕ), 
    (1/3 : ℝ) * C = 3 * (B : ℝ) ∧
    (2/3 : ℝ) * C = 2 * (G : ℝ) ∧
    (B + G = 40) ∧ 
    C = 144

theorem AngeliCandies :
  CandyProblem :=
sorry

end AngeliCandies_l143_143367


namespace minimum_value_of_reciprocals_is_four_l143_143039

theorem minimum_value_of_reciprocals_is_four (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ x : ℝ, x > 0 ∧ (1 / e * x + m + 1 = log x - n + 2) ∧ (1 / e = 1 / x)) :
  (1 / m + 1 / n) = 4 :=
by sorry

end minimum_value_of_reciprocals_is_four_l143_143039


namespace problem1_range_problem2_range_l143_143770

theorem problem1_range (x y : ℝ) (h : y = 2*|x-1| - |x-4|) : -3 ≤ y := sorry

theorem problem2_range (x a : ℝ) (h : ∀ x, 2*|x-1| - |x-a| ≥ -1) : 0 ≤ a ∧ a ≤ 2 := sorry

end problem1_range_problem2_range_l143_143770


namespace part1_tangent_line_max_min_values_l143_143921

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x
def tangent_line_at (a : ℝ) (x y : ℝ) : ℝ := 9 * x + y - 4

theorem part1 (a : ℝ) : f' a 1 = -9 → a = -6 :=
by
  sorry

theorem tangent_line (a : ℝ) (x y : ℝ) : a = -6 → f a 1 = -5 → tangent_line_at a 1 (-5) = 0 :=
by
  sorry

def interval := Set.Icc (-5 : ℝ) 5

theorem max_min_values (a : ℝ) : a = -6 →
  (∀ x ∈ interval, f a (-5) = -275 ∨ f a 0 = 0 ∨ f a 4 = -32 ∨ f a 5 = -25) →
  (∀ x ∈ interval, f a x ≤ 0 ∧ f a x ≥ -275) :=
by
  sorry

end part1_tangent_line_max_min_values_l143_143921


namespace exists_k_L_constant_over_range_find_m_for_distinct_L_l143_143170

-- Define L(n) representing the least common multiple of {1, 2, ..., n}
def L (n : ℕ) : ℕ := Nat.lcmList (List.range (n + 1))

-- Proof for problem (i)
theorem exists_k_L_constant_over_range :
  ∃ k : ℕ, ∀ i, 0 ≤ i ∧ i ≤ 2000 → L (k + i) = L k := by
  sorry

-- Proof for problem (ii)
theorem find_m_for_distinct_L :
  ∃ m : ℕ, ∀ i, i = 0 ∨ i = 1 ∨ i = 2 → L (m + i) ≠ L (m + i + 1) ∧ 
  (m = 1 ∨ m = 2 ∨ m = 6) := by
  sorry

end exists_k_L_constant_over_range_find_m_for_distinct_L_l143_143170


namespace find_x_l143_143876

theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_factors : ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧ (7^n + 1) = p1 * p2 * p3 ∧ (p1 = 2 ∨ p2 = 2 ∨ p3 = 2) ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
  7^n + 1 = 16808 :=
sorry

end find_x_l143_143876


namespace isosceles_triangle_complex_l143_143169

noncomputable def complex_roots : Prop := sorry

theorem isosceles_triangle_complex (a b z1 z2 : ℂ) (h1 : z1 ≠ 0) (h2 : z2 ≠ 0)
  (h3 : z2 = complex.exp (real.pi * complex.I / 4) * z1) 
  (h4 : ∀ z : ℂ, z^2 + a * z + b = 0 → z = z1 ∨ z = z2)
  (h5 : 0 ≠ z1 ∧ 0 ≠ z2 ∧ z1 ≠ z2 → (abs (z1 - 0) = abs (z2 - 0) 
       ∧ ¬(abs (z1 - 0) = abs (z2 - z1)))):
  a^2 / b = 4 + 4 * real.sqrt 2 := by
  sorry

end isosceles_triangle_complex_l143_143169


namespace inequality_solution_l143_143411

noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) * (x - 4) / (x + 2)

theorem inequality_solution :
  {x : ℝ | (2 * x - 5) * (x - 4) / (x + 2) ≥ 0} = Iio (-2) ∪ Ici 4 := 
by
  sorry

end inequality_solution_l143_143411


namespace rationalize_sqrt_fraction_l143_143636

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143636


namespace perimeter_triangle_FPQ_l143_143053

variables (b k m : Real) (h_b : b > 0) (h_k : k < 0) (h_m : m > 0)

def ellipse_C (x y : Real) : Prop := (x^2) / 3 + y^2 / (b^2) = 1
def circle_O (x y : Real) : Prop := x^2 + y^2 = 1
def line_l (x y : Real) : Prop := y = k * x + m

-- Claim the perimeter of ∆FPQ is 2√3
theorem perimeter_triangle_FPQ (F P Q : Real × Real)
  (h_inter1 : ellipse_C F P ∧ circle_O P ∧ line_l F P)
  (h_inter2 : ellipse_C F Q ∧ circle_O Q ∧ line_l F Q)
  (h_tangent : circle_O F P ∧ circle_O F Q) :
  2 := 2 * Real.sqrt 3
sorry

end perimeter_triangle_FPQ_l143_143053


namespace find_angle_APB_l143_143972

theorem find_angle_APB 
  (O1 O2 P A B R S T : Point)
  (h_PA_tangent : tangent_to_semi_circle P A S R)
  (h_PB_tangent : tangent_to_semi_circle P B R T)
  (h_SRT_line : collinear S R T)
  (arc_AS_40 : arc_angle A S = 40)
  (arc_BT_120 : arc_angle B T = 120)
  : angle A P B = 160 :=
sorry

end find_angle_APB_l143_143972


namespace minimum_cross_section_area_l143_143224

noncomputable theory

open Real

variable (h : ℝ) (α β : ℝ)
variable (TA TC : Triangle ℝ)
variable (BD : Line ℝ)
variable (base_plane : Plane ℝ)
variable (pyramid_plane : Plane ℝ)

axiom pyramid_definition :
  IsPyramidWithRectBase TA TC BD h α β base_plane pyramid_plane

theorem minimum_cross_section_area :
  α = π / 6 → β = π / 3 →
  MinimumAreaOfCrossSection BD TA TC h α β = h^2 * sqrt 3 / 8 :=
by
  sorry

end minimum_cross_section_area_l143_143224


namespace range_of_x_l143_143890

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Theorem statement
theorem range_of_x (x : ℝ) : (¬ q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  sorry

end range_of_x_l143_143890


namespace correct_statements_l143_143459

-- Defining the function f(x)
def f (x : ℝ) (b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Derivative of the function f(x)
def f' (x : ℝ) (b c : ℝ) : ℝ := 3*x^2 + 2*b*x + c

-- Defining the conditions
def condition1 (b c d k : ℝ) : Prop :=
  (k ∈ (-∞, 0) ∪ (4, ∞)) → (∃! x : ℝ, f x b c d = k)

def condition2 (b c d k : ℝ) : Prop :=
  (k ∈ (0, 4)) → (∃ x1 x2 x3 : ℝ, (f x1 b c d = k) ∧ (f x2 b c d = k) ∧ (f x3 b c d = k) ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)

-- Evaluating the statements
def statement1 (b c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, f' x1 b c = 0 ∧ f' x2 b c = 0 ∧ x1 ≠ x2

def statement2 (b c : ℝ) : Prop :=
  ∃ x1 x2 x3 : ℝ, f' x1 b c = 0 ∧ f' x2 b c = 0 ∧ f' x3 b c = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3

def statement3 (b c d : ℝ) : Prop :=
  ∃ x : ℝ, f x b c d = 4 ∧ f' x b c = 0

def statement4 (b c d : ℝ) : Prop :=
  ∃ x : ℝ, f x b c d = 0 ∧ f' x b c = 0

-- The main proposition to prove the correct statements
theorem correct_statements (b c d : ℝ) :
  (∀ k, condition1 b c d k) → (∀ k, condition2 b c d k) →
  statement1 b c ∧ ¬ statement2 b c ∧ statement3 b c d ∧ statement4 b c d :=
sorry

end correct_statements_l143_143459


namespace rationalize_denominator_l143_143690

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143690


namespace rationalize_sqrt_fraction_l143_143668

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143668


namespace find_x_l143_143870

theorem find_x (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p1 * p2 * p3 = 11 * (7^n + 1) ∧ p1.prime ∧ p2.prime ∧ p3.prime): 
  7^n + 1 = 16808 :=
begin
  sorry
end

end find_x_l143_143870


namespace permutation_exists_for_even_n_l143_143561

open Finset

theorem permutation_exists_for_even_n (n : ℕ) (h_even : n % 2 = 0) (h_pos : 0 < n) : 
  ∃ (x : Fin n → Fin n), 
    (∀ i : Fin n, 
      (x (i + 1) = 2 * x i) ∨
      (x (i + 1) = 2 * x i - 1) ∨
      (x (i + 1) = 2 * x i - n) ∨
      (x (i + 1) = 2 * x i - n - 1)) ∧
    (∀ i j : Fin n, i ≠ j → x i ≠ x j) :=
sorry

end permutation_exists_for_even_n_l143_143561


namespace rationalize_denominator_l143_143680

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143680


namespace sum_of_three_positive_integers_l143_143538

theorem sum_of_three_positive_integers (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, k = (n - 1) * (n - 2) / 2 := 
sorry

end sum_of_three_positive_integers_l143_143538


namespace rationalize_denominator_l143_143656

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143656


namespace sum_of_coordinates_is_1_5_l143_143242

-- The points (12, -7) and (-6, 4) are the endpoints of a diameter of a circle.
def point1 : ℝ × ℝ := (12, -7)
def point2 : ℝ × ℝ := (-6, 4)

-- Define the center of the circle as the midpoint of the diameter.
def center : ℝ × ℝ := ((point1.1 + point2.1) / 2, (point1.2 + point2.2) / 2)

-- Define the sum of the coordinates of the center.
def sum_of_coordinates_of_center : ℝ := center.1 + center.2

-- Prove that the sum of the coordinates of the center is 1.5.
theorem sum_of_coordinates_is_1_5 : sum_of_coordinates_of_center = 1.5 := by
  sorry

end sum_of_coordinates_is_1_5_l143_143242


namespace students_number_l143_143253

theorem students_number (C P S : ℕ) : C = 315 ∧ 121 + C = P * S -> S = 4 := by
  sorry

end students_number_l143_143253


namespace tree_height_at_year_3_l143_143797

theorem tree_height_at_year_3 :
  ∃ h₃ : ℕ, h₃ = 27 ∧
  (∃ h₇ h₆ h₅ h₄ : ℕ,
   h₇ = 648 ∧
   h₆ = h₇ / 2 ∧
   h₅ = h₆ / 2 ∧
   h₄ = h₅ / 2 ∧
   h₄ = 3 * h₃) :=
by
  sorry

end tree_height_at_year_3_l143_143797


namespace max_cards_Jasmine_can_buy_l143_143147

-- Define the conditions
def money_jasmine_has : ℝ := 9.20
def cost_per_card : ℝ := 1.05

-- State the theorem
theorem max_cards_Jasmine_can_buy : ∃ n : ℕ, n = 8 ∧ (cost_per_card * n ≤ money_jasmine_has) :=
by
  use 8
  split
  · refl
  · exact sorry

end max_cards_Jasmine_can_buy_l143_143147


namespace parallel_line_segment_length_l143_143236

theorem parallel_line_segment_length (AB : ℝ) (S : ℝ) (x : ℝ) 
  (h1 : AB = 36) 
  (h2 : S = (S / 2) * 2)
  (h3 : x / AB = (↑(1 : ℝ) / 2 * S / S) ^ (1 / 2)) : 
  x = 18 * Real.sqrt 2 :=
by 
    sorry 

end parallel_line_segment_length_l143_143236


namespace sequence_count_l143_143482

theorem sequence_count (n : ℕ) : (∃ x : ℕ, x = 15 → 
  let c := (∀ s : list bool, s.length = 15 → 
    ((∃ k : ℕ, (1 ≤ k ∧ k ≤ 14) ∧ ∃ pos : ℕ, pos + k ≤ s.length ∧ ∀ i, pos ≤ i → i < pos + k → s.nth i = some tt) ∨ 
    (∃ k : ℕ, (1 ≤ k ∧ k ≤ 14) ∧ ∃ pos : ℕ, pos + k ≤ s.length ∧ ∀ i, pos ≤ i → i < pos + k → s.nth i = some ff) 
    → ¬(∃ k1 k2, k1 = 15 ∨ k2 = 15))) 
  in c → x = 238) := 
sorry

end sequence_count_l143_143482


namespace find_RT_l143_143976

noncomputable def triangle_def := ℝ

-- Conditions
variables (DE DF EF : triangle_def)
variables (S J R T : triangle_def) -- points on the triangle related to the constructions
variables (D E F : triangle_def) -- vertices of the triangle

-- Given lengths of the sides of triangle
axiom de_length : DE = 135
axiom df_length : DF = 132
axiom ef_length : EF = 123

-- Problem statement
theorem find_RT : RT = 60 := by
  -- the conditions and steps leading to RT = 60 would be here
  sorry

end find_RT_l143_143976


namespace tylenol_mg_per_tablet_l143_143186

noncomputable def dose_intervals : ℕ := 3  -- Mark takes Tylenol 3 times
noncomputable def total_mg : ℕ := 3000     -- Total intake in milligrams
noncomputable def tablets_per_dose : ℕ := 2  -- Number of tablets per dose

noncomputable def tablet_mg : ℕ :=
  total_mg / dose_intervals / tablets_per_dose

theorem tylenol_mg_per_tablet : tablet_mg = 500 := by
  sorry

end tylenol_mg_per_tablet_l143_143186


namespace angle_between_vectors_l143_143949

variables {G : Type*} [inner_product_space ℝ G] (a b : G)

theorem angle_between_vectors 
  (h₁ : ∥a∥ = 3 * ∥b∥) 
  (h₂ : inner (2 • a + 3 • b) b = 0) : 
  real.angle a b = 2 * real.pi / 3 :=
by sorry

end angle_between_vectors_l143_143949


namespace monotonic_increase_interval_l143_143234

noncomputable def f (x : ℝ) := (x - 3) * Real.exp x

theorem monotonic_increase_interval (x : ℝ) :
  (∃ c : ℝ, c > 2) → (∀ x, 2 < x → 0 < (x - 2) * Real.exp x → f' x > 0) :=
sorry

end monotonic_increase_interval_l143_143234


namespace part1_not_three_equal_root_equation_l143_143904

variables x : ℝ

def is_three_equal_root_equation (a b c : ℝ) : Prop :=
∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) ∧ (x1 = (1 / 3) * x2)

theorem part1_not_three_equal_root_equation : 
  ¬ is_three_equal_root_equation 1 (-8) 11 :=
by {
  sorry
}

end part1_not_three_equal_root_equation_l143_143904


namespace intersection_is_correct_complement_is_correct_l143_143475

open Set

variable {U : Set ℝ} (A B : Set ℝ)

-- Define the universal set U
def U_def : Set ℝ := { x | 1 < x ∧ x < 7 }

-- Define set A
def A_def : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

-- Define set B using the simplified condition from the inequality
def B_def : Set ℝ := { x | x ≥ 3 }

-- Proof statement that A ∩ B is as specified
theorem intersection_is_correct :
  (A_def ∩ B_def) = { x : ℝ | 3 ≤ x ∧ x < 5 } := by
  sorry

-- Proof statement for the complement of A relative to U
theorem complement_is_correct :
  (U_def \ A_def) = { x : ℝ | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7) } := by
  sorry

end intersection_is_correct_complement_is_correct_l143_143475


namespace sequence_term_divisible_by_n_l143_143204

theorem sequence_term_divisible_by_n (n : ℕ) (hn1 : 1 < n) (hn_odd : n % 2 = 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ n ∣ (2^k - 1) :=
by
  sorry

end sequence_term_divisible_by_n_l143_143204


namespace exists_least_n_l143_143565

noncomputable def a : ℕ → ℕ
| 5 := 5
| (n + 1) := 70 * a n + 2 * (n + 1)

theorem exists_least_n : ∃ n > 5, (a n) % 77 = 0 ∧ ∀ m > 5, (m < n) → (a m) % 77 ≠ 0 :=
begin
  use 7,
  split,
  { norm_num },
  split,
  { rw a, norm_num },
  { intros m h_m h_le,
    cases m,
    { linarith },
    { cases m,
      { linarith },
      { rw a,
        stochastic,
        integration
        library_search },
    { cases m,
      { rw a,
        constant_h 
        lib,
        card_length }} }
end

end exists_least_n_l143_143565


namespace power_function_correct_statement_l143_143361

theorem power_function_correct_statement :
  (∀ α : ℝ, 0 < α → y = x^α →
    (y = x^α → y treats 0 ~ x) ∧
    ((∀ x : ℝ, 0 ≤ x → y = x^α → x treats 0 ~ 1 → y treats 0 ~ 1)) ∧
    (∀ α : ℝ, α = 1 ∨ α = 3 ∨ α = 1 / 2 → ∀ x : ℝ, x ≥ 0 → differentiable x^α ∧
    ((∀ α : ℝ, α = -1 → ∀ x : ℝ, x > 0 → deriv (x^α) < 0) ∧ (∀ α : ℝ, α = -1 → ∀ x : ℝ, x < 0 → deriv (x^α) < -1))) 
  := sorry

end power_function_correct_statement_l143_143361


namespace rationalize_sqrt_fraction_l143_143638

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143638


namespace compute_cosine_B_l143_143143

theorem compute_cosine_B (A B C : Type) 
    [DecidableEq A] [DecidableEq B] [DecidableEq C] 
    (AC : Real) (AB : Real) (BC : Real) (h1 : AC = Real.sqrt 34) (h2 : AB = 5) (h3 : BC = 3) 
    (right_triangle : ∠ A = 90) : 
    Real.cos (∠ B) = Real.sqrt 34 / 3 := 
    by sorry

end compute_cosine_B_l143_143143


namespace solve_for_x_l143_143792

noncomputable theory

def problem_statement (x y : ℝ) : Prop :=
  y = 3 ∧ (2 * x) ^ y - 152 = 102 → x = real.cbrt(254) / 2

-- Provide the main theorem
theorem solve_for_x : ∃ x : ℝ, problem_statement x 3 :=
begin
  use real.cbrt(254) / 2,
  unfold problem_statement,
  split,
  { refl },
  { norm_num,
    suffices : (2 * (real.cbrt 254 / 2)) ^ 3 = 254,
    { linarith },
    { field_simp,
      ring_exp_eq,
      exact real.rpow_nat_cast _ 3 } }
end

#check solve_for_x

end solve_for_x_l143_143792


namespace factory_produces_6400_toys_per_week_l143_143334

-- Definition of worker productivity per day
def toys_per_day : ℝ := 2133.3333333333335

-- Definition of workdays per week
def workdays_per_week : ℕ := 3

-- Definition of total toys produced per week
def toys_per_week : ℝ := toys_per_day * workdays_per_week

-- Theorem stating the total number of toys produced per week
theorem factory_produces_6400_toys_per_week : toys_per_week = 6400 :=
by
  sorry

end factory_produces_6400_toys_per_week_l143_143334


namespace find_angle_l143_143521

-- Define the context for triangle
variables {A M : Type} [triangle : Triangle A M]

-- Assume that 'b', 'c', and 'f_alpha' are lengths and 'α' is the angle we need to find
variables (b c f_alpha : ℝ) -- lengths of sides and angle bisector
variables (α : ℝ) -- given angle

-- Provide the condition that relates the sides and the distance of angle bisector
def condition := (b - c = 2 * f_alpha)

-- Formalize the proof problem: show that α = 60 degrees
theorem find_angle (h : condition b c f_alpha) : 
  α = 60 :=
sorry

end find_angle_l143_143521


namespace total_tiles_to_be_replaced_l143_143358

-- Define the given conditions
def horizontal_paths : List ℕ := [30, 50, 30, 20, 20, 50]
def vertical_paths : List ℕ := [20, 50, 20, 50, 50]
def intersections : ℕ := List.sum [2, 3, 3, 4, 4]

-- Problem statement: Prove that the total number of tiles to be replaced is 374
theorem total_tiles_to_be_replaced : List.sum horizontal_paths + List.sum vertical_paths - intersections = 374 := 
by sorry

end total_tiles_to_be_replaced_l143_143358


namespace rationalize_sqrt_fraction_denom_l143_143606

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143606


namespace sum_divisors_24_l143_143278

theorem sum_divisors_24 :
  (∑ n in Finset.filter (λ n => 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_divisors_24_l143_143278


namespace number_of_dimes_l143_143347

theorem number_of_dimes (Q D : ℕ) 
  (h1 : 0.25 * Q + 0.10 * D = 39.50 ) 
  (h2 : Q + D = 200) :
  D = 70 :=
sorry

end number_of_dimes_l143_143347


namespace rationalize_sqrt_5_over_12_l143_143648

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143648


namespace distance_between_points_l143_143815

theorem distance_between_points :
  let x1 := 3
      y1 := -2
      z1 := 0
      x2 := 7
      y2 := 4
      z2 := -3
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) = sqrt 61 :=
by
  sorry

end distance_between_points_l143_143815


namespace family_members_count_l143_143149

variable (F : ℕ) -- Number of other family members

def annual_cost_per_person : ℕ := 4000 + 12 * 1000
def john_total_cost_for_family (F : ℕ) : ℕ := (F + 1) * annual_cost_per_person / 2

theorem family_members_count :
  john_total_cost_for_family F = 32000 → F = 3 := by
  sorry

end family_members_count_l143_143149


namespace arithmetic_sequence_property_Sn_property_sum_property_l143_143047

open Nat

noncomputable def seq_a (n : ℕ) : ℕ :=
if h : n > 0 then 2 * n else 0

def Sn (n : ℕ) : ℕ := ∑ i in finset.range n, seq_a i

theorem arithmetic_sequence_property :
  (∀ n ∈ ℕ, seq_a (n + 1) * (seq_a (n + 1) - 2) = seq_a n * (seq_a n + 2)) :=
sorry

theorem Sn_property (hS3 : Sn 3 = 12) : seq_a 1 = 2 :=
sorry

def seq_b (n : ℕ) : ℚ :=
1 / (seq_a n * seq_a (n + 1))

def Tn (n : ℕ) : ℚ :=
∑ i in finset.range n, seq_b i

theorem sum_property (h : (∀ n ∈ ℕ, seq_a (n + 1) * (seq_a (n + 1) - 2) = seq_a n * (seq_a n + 2)) 
                      (seq_a 1 = 2)) : 
  Tn = (n : ℚ) / (4 * (n + 1)) :=
sorry

end arithmetic_sequence_property_Sn_property_sum_property_l143_143047


namespace max_salary_difference_2002_employees_l143_143736

theorem max_salary_difference_2002_employees 
  (n : ℕ)(h_n : n = 2002)
  (salaries : Fin n → ℕ)
  (h_distinct : Function.Injective salaries)
  (diff : Fin n → Fin n → ℕ)
  (h_diff : ∀ i : Fin n, diff i (i + 1) = abs (salaries (i + 1) - salaries i))
  (h_diff_range : ∀ i : Fin n, diff i (i + 1) ∈ {2, 3})
  : ∃ a b : Fin n, abs (salaries a - salaries b) = 3002 :=
by
  sorry

end max_salary_difference_2002_employees_l143_143736


namespace geometric_sequence_product_l143_143439

variable {α : Type*} [Group α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∀ n k : ℕ, (∃ r : α, a (n + k) = a n * r^k)

noncomputable def a_n : ℕ → ℝ := sorry

axiom h1 : is_geometric_sequence a_n
axiom h2 : a_n 3 * a_n 4 = 2

theorem geometric_sequence_product : a_n 1 * a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 = 8 :=
  sorry

end geometric_sequence_product_l143_143439


namespace gary_stickers_l143_143033

theorem gary_stickers (initial_stickers : ℕ) (given_to_Lucy : ℕ) (given_to_Alex : ℕ) :
  initial_stickers = 99 → given_to_Lucy = 42 → given_to_Alex = 26 → (initial_stickers - (given_to_Lucy + given_to_Alex)) = 31 :=
by
  intros h_initial h_Lucy h_Alex
  rw [h_initial, h_Lucy, h_Alex]
  sorry

end gary_stickers_l143_143033


namespace preimage_of_4_neg_2_eq_1_3_l143_143453

def mapping (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem preimage_of_4_neg_2_eq_1_3 : ∃ x y : ℝ, mapping x y = (4, -2) ∧ (x = 1) ∧ (y = 3) :=
by 
  sorry

end preimage_of_4_neg_2_eq_1_3_l143_143453


namespace min_value_of_expression_l143_143028

variable (a b : ℝ)

theorem min_value_of_expression (h : b ≠ 0) : 
  ∃ (a b : ℝ), (a^2 + b^2 + a / b + 1 / b^2) = Real.sqrt 3 :=
sorry

end min_value_of_expression_l143_143028


namespace cost_price_for_fabrics_l143_143355

noncomputable def total_cost_price (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  selling_price - (meters_sold * profit_per_meter)

noncomputable def cost_price_per_meter (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  total_cost_price meters_sold selling_price profit_per_meter / meters_sold

theorem cost_price_for_fabrics :
  cost_price_per_meter 45 6000 12 = 121.33 ∧
  cost_price_per_meter 60 10800 15 = 165 ∧
  cost_price_per_meter 30 3900 10 = 120 :=
by
  sorry

end cost_price_for_fabrics_l143_143355


namespace rationalize_sqrt_fraction_l143_143669

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143669


namespace impossible_to_get_105_piles_l143_143141

-- Definitions for initial conditions
def initial_piles : list ℕ := [51, 49, 5]

/-- 
  Given the operations of combining any two piles into one or dividing a pile with an even number of stones
  into two equal piles, it is impossible to get 105 piles each containing exactly one stone from the initial
  piles [51, 49, 5].
-/
theorem impossible_to_get_105_piles (initial_piles = [51, 49, 5]) : 
  ¬ ∃ piles : list ℕ, (∀ pile ∈ piles, pile = 1) ∧ piles.length = 105 := 
sorry

end impossible_to_get_105_piles_l143_143141


namespace minimum_value_l143_143164

theorem minimum_value {a b c : ℝ} (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * b * c = 1 / 2) :
  ∃ x, x = a^2 + 4 * a * b + 9 * b^2 + 8 * b * c + 3 * c^2 ∧ x = 13.5 :=
sorry

end minimum_value_l143_143164


namespace Amanda_car_round_trip_time_l143_143801

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end Amanda_car_round_trip_time_l143_143801


namespace friendly_sequences_exist_l143_143707

theorem friendly_sequences_exist :
  ∃ (a b : ℕ → ℕ), 
    (∀ n, a n = 2^(n-1)) ∧ 
    (∀ n, b n = 2*n - 1) ∧ 
    (∀ k : ℕ, ∃ (i j : ℕ), k = a i * b j) :=
by
  sorry

end friendly_sequences_exist_l143_143707


namespace sum_odd_product_even_l143_143601

theorem sum_odd_product_even (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) 
                             (h2 : ∃ m : ℤ, b = 2 * m + 1) 
                             (h3 : ∃ n : ℤ, a + b = 2 * n + 1) : 
  ∃ p : ℤ, a * b = 2 * p := 
  sorry

end sum_odd_product_even_l143_143601


namespace range_of_a_l143_143038

def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x >= 0 then -(x - a)^2 else -x^2 - 2 * x - 3 + a

theorem range_of_a (a : ℝ) (H : ∀ x, f a x ≤ f a 0) : -2 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l143_143038


namespace combined_molecular_weight_l143_143271

-- Define atomic masses of elements
def atomic_mass_Ca : Float := 40.08
def atomic_mass_Br : Float := 79.904
def atomic_mass_Sr : Float := 87.62
def atomic_mass_Cl : Float := 35.453

-- Define number of moles for each compound
def moles_CaBr2 : Float := 4
def moles_SrCl2 : Float := 3

-- Define molar masses of compounds
def molar_mass_CaBr2 : Float := atomic_mass_Ca + 2 * atomic_mass_Br
def molar_mass_SrCl2 : Float := atomic_mass_Sr + 2 * atomic_mass_Cl

-- Define total mass calculation for each compound
def total_mass_CaBr2 : Float := moles_CaBr2 * molar_mass_CaBr2
def total_mass_SrCl2 : Float := moles_SrCl2 * molar_mass_SrCl2

-- Prove the combined molecular weight
theorem combined_molecular_weight :
  total_mass_CaBr2 + total_mass_SrCl2 = 1275.13 :=
  by
    -- The proof will be here
    sorry

end combined_molecular_weight_l143_143271


namespace vector_AD_length_l143_143908

open Real EuclideanSpace

noncomputable def problem_statement
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC : ℝ) (AD : ℝ) : Prop :=
  angle_mn = π / 6 ∧ 
  norm_m = sqrt 3 ∧ 
  norm_n = 2 ∧ 
  AB = 2 * m + 2 * n ∧ 
  AC = 2 * m - 6 * n ∧ 
  AD = 2 * m - 2 * n ∧
  sqrt ((AD) * (AD)) = 2

theorem vector_AD_length 
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC AD : ℝ) :
  problem_statement m n angle_mn norm_m norm_n AB AC AD :=
by
  unfold problem_statement
  sorry

end vector_AD_length_l143_143908


namespace range_of_f_l143_143754

def f (x : ℝ) : ℝ := (x^2 + 1) / x^2

theorem range_of_f :
  set.range f = {y : ℝ | y > 1} :=
by sorry

end range_of_f_l143_143754


namespace count_perfect_squares_with_specific_digits_l143_143939

theorem count_perfect_squares_with_specific_digits 
    (n : ℕ) : 
    let is_perfect_square (k : ℕ) := ∃ m : ℕ, m * m = k in
    let ones_digit (m : ℕ) := m % 10 in
    (∀ k, k < n → is_perfect_square k → 
    (ones_digit k = 1 ∨ ones_digit k = 4 ∨ ones_digit k = 9) = 
    (∃ count, count = 42)) :=
begin
  sorry
end

end count_perfect_squares_with_specific_digits_l143_143939


namespace sum_of_coordinates_of_D_is_12_l143_143599

theorem sum_of_coordinates_of_D_is_12 :
  (exists (x y : ℝ), (5 = (11 + x) / 2) ∧ (9 = (5 + y) / 2) ∧ (x + y = 12)) :=
by
  sorry

end sum_of_coordinates_of_D_is_12_l143_143599


namespace find_n_for_volume_l143_143595

noncomputable def volume_factor (n : ℕ) : ℝ := (2*n + 1) / (2*n)

noncomputable def cumulative_volume (n : ℕ) : ℝ :=
  ∏ i in finset.range (n + 1), volume_factor (i + 1)

theorem find_n_for_volume :
  ∃ n : ℕ, cumulative_volume n = 150 := sorry

end find_n_for_volume_l143_143595


namespace shorter_piece_length_l143_143774

theorem shorter_piece_length (P : ℝ) (Q : ℝ) (h1 : P + Q = 68) (h2 : Q = P + 12) : P = 28 := 
by
  sorry

end shorter_piece_length_l143_143774


namespace necessary_condition_for_x_greater_than_2_l143_143903

-- Define the real number x
variable (x : ℝ)

-- The proof statement
theorem necessary_condition_for_x_greater_than_2 : (x > 2) → (x > 1) :=
by sorry

end necessary_condition_for_x_greater_than_2_l143_143903


namespace rationalize_sqrt_fraction_l143_143633

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143633


namespace event_probabilities_equal_l143_143515

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l143_143515


namespace complex_magnitude_l143_143578

theorem complex_magnitude :
  ∀ z : ℂ, z = ((1 - I) / (1 + I)) + 2 * I → complex.abs z = 1 :=
by
  intro z h
  calc
    z = ((1 - I) / (1 + I) + 2 * I) : by { rw h }
    -- Continue proof steps if needed, omitted with sorry to skip the proof
    sorry

end complex_magnitude_l143_143578


namespace problem_solution_l143_143896

noncomputable def a_n (n : ℕ) : ℕ := 3 * n - 1

noncomputable def b_n (n : ℕ) : ℚ := (1 / 3)^(n - 1)

noncomputable def c_n (n : ℕ) : ℚ := a_n n * b_n n

noncomputable def T_n (n : ℕ) : ℚ := ∑ k in finset.range(n + 1), c_n k

theorem problem_solution (n : ℕ) :
  a_n n = 3 * n - 1 ∧
  b_n n = (1 / 3)^ (n - 1) ∧
  T_n n = 7 / 2 - (n + 7 / 6) * (1 / 3)^(n - 1) := 
by
  sorry

end problem_solution_l143_143896


namespace find_a_and_monotonic_intervals_x2_plus_x3_less_than_zero_l143_143075

noncomputable def f (x a : ℝ) : ℝ := (x - a) * real.exp (x + 1) - (1/2) * x^2

theorem find_a_and_monotonic_intervals :
  (∃ a : ℝ, a = 1) ∧ 
  (∃ I1 I2 I3 : set ℝ, 
    (∀ x, x ∈ I1 → f x 1 < f (x + 1) 1) ∧ 
    (∀ x, x ∈ I2 → f x 1 > f (x + 1) 1) ∧ 
    (∀ x, x ∈ I3 → f x 1 < f (x + 1) 1) ∧ 
    I1 = {x | x < -1} ∧ 
    I2 = {x | -1 < x ∧ x < 0} ∧ 
    I3 = {x | x > 0}) :=
sorry

theorem x2_plus_x3_less_than_zero (x₁ x₂ x₃ : ℝ)
  (h₁ : x₁ < x₂) (h₂ : x₂ < x₃)
  (hfx1 : f x₁ 1 = f x₂ 1) (hfx2 : f x₂ 1 = f x₃ 1) :
  x₂ + x₃ < 0 :=
sorry

end find_a_and_monotonic_intervals_x2_plus_x3_less_than_zero_l143_143075


namespace find_other_number_l143_143223

-- Definitions for the given conditions
def A : ℕ := 500
def LCM : ℕ := 3000
def HCF : ℕ := 100

-- Theorem statement: If A = 500, LCM(A, B) = 3000, and HCF(A, B) = 100, then B = 600.
theorem find_other_number (B : ℕ) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 :=
by
  sorry

end find_other_number_l143_143223


namespace value_of_t_l143_143034

theorem value_of_t (x y t : ℝ) (hx : 2^x = t) (hy : 7^y = t) (hxy : 1/x + 1/y = 2) : t = Real.sqrt 14 :=
by
  sorry

end value_of_t_l143_143034


namespace natalie_blueberry_bushes_l143_143836

-- Definitions of the conditions
def bushes_yield_containers (bushes containers : ℕ) : Prop :=
  containers = bushes * 7

def containers_exchange_zucchinis (containers zucchinis : ℕ) : Prop :=
  zucchinis = containers * 3 / 7

-- Theorem statement
theorem natalie_blueberry_bushes (zucchinis_needed : ℕ) (zucchinis_per_trade containers_per_trade bushes_per_container : ℕ) 
  (h1 : zucchinis_per_trade = 3) (h2 : containers_per_trade = 7) (h3 : bushes_per_container = 7) 
  (h4 : zucchinis_needed = 63) : 
  ∃ bushes_needed : ℕ, bushes_needed = 21 := 
by
  sorry

end natalie_blueberry_bushes_l143_143836


namespace mode_is_41_l143_143331

def sales_data : List (ℕ × ℕ) := [(39, 10), (40, 14), (41, 25), (42, 13), (43, s)]

def mode (data : List (ℕ × ℕ)) : ℕ :=
  data.maximumBy (λ x => x.2)

theorem mode_is_41 (s : ℕ) (h : s ≤ 25) : mode [(39, 10), (40, 14), (41, 25), (42, 13), (43, s)] = 41 :=
  by
    sorry

end mode_is_41_l143_143331


namespace matrix_power_trace_l143_143320

variable {k n : ℕ}
variable (A : Matrix (Fin n) (Fin n) ℝ)
variable [IsSymm A]
variable (h_even_k : ∃ l : ℕ, k = 2 * l)
variable (h_condition : (Matrix.trace (A ^ k)) ^ (k + 1) = (Matrix.trace (A ^ (k + 1))) ^ k)

theorem matrix_power_trace (h_even_k : ∃ l : ℕ, k = 2 * l) 
  (h_condition : (Matrix.trace (A ^ k)) ^ (k + 1) = (Matrix.trace (A ^ (k + 1))) ^ k) :
  A ^ n = Matrix.trace A • A ^ (n - 1) :=
sorry

end matrix_power_trace_l143_143320


namespace term_50_is_273_l143_143388

-- Define the concept of a term being a sum of distinct powers of 3 up to 3^5
def is_sum_of_distinct_powers_of_three (n : ℕ) : Prop :=
  ∃ (a b c d e f : bool), 
    (n = ((if a then 1 else 0) * 3^0 + 
          (if b then 1 else 0) * 3^1 + 
          (if c then 1 else 0) * 3^2 + 
          (if d then 1 else 0) * 3^3 + 
          (if e then 1 else 0) * 3^4 + 
          (if f then 1 else 0) * 3^5)) 

-- Define the sequence as a list of natural numbers
noncomputable def sequence : ℕ → ℕ
| 0       := 0
| (n + 1) := Nat.find (λ m : ℕ, is_sum_of_distinct_powers_of_three m ∧ m > sequence n)

-- Prove that the 50th term of the sequence is 273
theorem term_50_is_273 : sequence 50 = 273 := sorry

end term_50_is_273_l143_143388


namespace angle_A_is_60_degrees_l143_143086

variables {A B C : Type*} [nonempty A] [nonempty B] [nonempty C]
variables (triangle_ABC : triangle A B C)
variables (O : excenter triangle_ABC B C)
variables (O_1 : reflection O B C)

theorem angle_A_is_60_degrees (h : lies_on_circumcircle O_1 triangle_ABC) :
  ∠BAC = 60 :=
sorry

end angle_A_is_60_degrees_l143_143086


namespace triangle_similarity_medians_iff_sq_sides_arith_seq_l143_143206

variable {a b c : ℝ}

theorem triangle_similarity_medians_iff_sq_sides_arith_seq
  (h : a^2 + b^2 = 2 * c^2 ∨ b^2 + c^2 = 2 * a^2 ∨ c^2 + a^2 = 2 * b^2) :
  ∃ k : ℝ, k = √3 / 2 ∧
    (m_a = √(3 / 4) * a ∧ m_b = √(3 / 4) * b ∧ m_c = √(3 / 4) * c) :=
sorry

end triangle_similarity_medians_iff_sq_sides_arith_seq_l143_143206


namespace each_worker_paid_40_l143_143786

variable (n_orchids : ℕ) (price_per_orchid : ℕ)
variable (n_money_plants : ℕ) (price_per_money_plant : ℕ)
variable (new_pots_cost : ℕ) (leftover_money : ℕ)
variable (n_workers : ℕ)

noncomputable def total_earnings : ℤ :=
  n_orchids * price_per_orchid + n_money_plants * price_per_money_plant

noncomputable def total_spent : ℤ :=
  new_pots_cost + leftover_money

noncomputable def amount_paid_to_workers : ℤ :=
  total_earnings n_orchids price_per_orchid n_money_plants price_per_money_plant - 
  total_spent new_pots_cost leftover_money

noncomputable def amount_paid_to_each_worker : ℤ :=
  amount_paid_to_workers n_orchids price_per_orchid n_money_plants price_per_money_plant 
    new_pots_cost leftover_money / n_workers

theorem each_worker_paid_40 :
  amount_paid_to_each_worker 20 50 15 25 150 1145 2 = 40 := by
  sorry

end each_worker_paid_40_l143_143786


namespace calculate_expression_l143_143377

theorem calculate_expression : 
  let a := (-1 : Int) ^ 2023
  let b := (-8 : Int) / (-4)
  let c := abs (-5)
  a + b - c = -4 := 
by
  sorry

end calculate_expression_l143_143377


namespace x0_y0_sum_eq_31_l143_143321

theorem x0_y0_sum_eq_31 :
  ∃ x0 y0 : ℕ, (0 ≤ x0 ∧ x0 < 37) ∧ (0 ≤ y0 ∧ y0 < 37) ∧ 
  (2 * x0 ≡ 1 [MOD 37]) ∧ (3 * y0 ≡ 36 [MOD 37]) ∧ 
  (x0 + y0 = 31) :=
sorry

end x0_y0_sum_eq_31_l143_143321


namespace find_m_and_n_l143_143117

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end find_m_and_n_l143_143117


namespace rationalize_sqrt_fraction_l143_143664

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143664


namespace sum_digits_greatest_prime_divisor_l143_143300

open Nat

theorem sum_digits_greatest_prime_divisor (n : ℕ) (h : n = 32767) : 
  sum_digits (greatest_prime_divisor n) = 7 :=
by
  sorry

def greatest_prime_divisor (n : ℕ) : ℕ :=
  if h : n > 1 then
    (factors n).filter prime |> list.maximum' 1
  else
    0

def sum_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_digits (n / 10)

end sum_digits_greatest_prime_divisor_l143_143300


namespace halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l143_143843

theorem halfway_between_one_sixth_and_one_twelfth_is_one_eighth : 
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := 
by
  sorry

end halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l143_143843


namespace find_g_l143_143491

theorem find_g (f g : ℝ[X]) (h₀ : f + g = 3 * X^2 - 1) (h₁ : f = X^4 - X^2 - 3) : g = -X^4 + 4 * X^2 + 2 :=
by
  sorry

end find_g_l143_143491


namespace amplitude_of_cosine_function_is_3_l143_143813

variable (a b : ℝ)
variable (h_a : a > 0)
variable (h_b : b > 0)
variable (h_max : ∀ x : ℝ, a * Real.cos (b * x) ≤ 3)
variable (h_cycle : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (∀ x : ℝ, a * Real.cos (b * (x + 2 * Real.pi)) = a * Real.cos (b * x)))

theorem amplitude_of_cosine_function_is_3 :
  a = 3 :=
sorry

end amplitude_of_cosine_function_is_3_l143_143813


namespace greatest_possible_percentage_of_both_services_l143_143325

theorem greatest_possible_percentage_of_both_services (p₁ p₂ : ℝ) (h₁ : p₁ = 0.45) (h₂ : p₂ = 0.70) :
  ∃ p : ℝ, p = 0.45 := by
  use p₁
  rw [h₁]
  sorry

end greatest_possible_percentage_of_both_services_l143_143325


namespace translated_upwards_2_units_l143_143745

theorem translated_upwards_2_units (x : ℝ) : (x + 2 > 0) → (x > -2) :=
by 
  intros h
  exact sorry

end translated_upwards_2_units_l143_143745


namespace num_distinct_products_l143_143040

def A : Set Int := {2, 3, 7}
def B : Set Int := {-31, -24, 4}
def distinct_products (A B : Set Int) : Set Int := 
  {z | ∃ a ∈ A, ∃ b ∈ B, z = a * b}

theorem num_distinct_products : 
  (distinct_products A B).card = 9 := 
by
  sorry

end num_distinct_products_l143_143040


namespace simplified_expression_l143_143711

def simplify_expression (x : ℝ) : ℝ :=
  sqrt (1 + ( (x^4 - x^2) / (2 * x^2) )^2)

theorem simplified_expression (x : ℝ) : 
  simplify_expression x = sqrt ((x^8 - 2 * x^6 + 5 * x^4) / (4 * x^4)) :=
by
  sorry

end simplified_expression_l143_143711


namespace sin_is_symmetric_about_origin_l143_143363

-- Define the four functions
def f1 (x : ℝ) : ℝ := Real.log x
def f2 (x : ℝ) : ℝ := Real.cos x
def f3 (x : ℝ) : ℝ := abs x
def f4 (x : ℝ) : ℝ := Real.sin x

-- Define what it means for a function to be symmetric about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- Proof problem statement in Lean 4
theorem sin_is_symmetric_about_origin : symmetric_about_origin f4 := by
  sorry

end sin_is_symmetric_about_origin_l143_143363


namespace rationalize_sqrt_fraction_l143_143625

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143625


namespace parabola_x_intercepts_difference_l143_143880

theorem parabola_x_intercepts_difference :
  ∀ (a b c x1 x2 : ℝ), 
  (y = a * x ^ 2 + b * x + c) → 
  (vertex (a * x ^ 2 + b * x + c) = (3, -9)) →
  (a * (5 - 3) ^ 2 - 9 = 7) →
  (has_root (a * x ^ 2 + b * x + c) x1) →
  (has_root (a * x ^ 2 + b * x + c) x2) →
  (x1 = 4.5) → 
  (x2 = 1.5) →
  (|x1 - x2| = 3) :=
begin
  sorry
end

end parabola_x_intercepts_difference_l143_143880


namespace factorize_expression_l143_143005

-- Define the variables a and b
variables (a b : ℝ)

-- State the theorem
theorem factorize_expression : 5*a^2*b - 20*b^3 = 5*b*(a + 2*b)*(a - 2*b) :=
by sorry

end factorize_expression_l143_143005


namespace number_of_arrangements_l143_143712

theorem number_of_arrangements (A B : Type) (individuals : Fin 6 → Type)
  (adjacent_condition : ∃ (i : Fin 5), individuals i = B ∧ individuals (i + 1) = A) :
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end number_of_arrangements_l143_143712


namespace hotel_key_distribution_l143_143784

theorem hotel_key_distribution 
  (rooms : ℕ) 
  (guests : ℕ) 
  (returning_guests : ℕ) 
  (h_rooms : rooms = 90) 
  (h_guests : guests = 100) 
  (h_returning_guests : returning_guests = 90) : 
  ∃ (keys : ℕ), keys >= 990 := 
by
  have h1 : rooms = 90 := h_rooms
  have h2 : guests = 100 := h_guests
  have h3 : returning_guests = 90 := h_returning_guests
  existsi (990 : ℕ)
  apply nat.le_refl
  sorry

end hotel_key_distribution_l143_143784


namespace rationalize_denominator_l143_143687

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143687


namespace rationalize_denominator_l143_143699

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143699


namespace calculate_r_when_n_is_3_l143_143173

theorem calculate_r_when_n_is_3 : 
  ∀ (r s n : ℕ), r = 4^s - s → s = 3^n + 2 → n = 3 → r = 4^29 - 29 :=
by 
  intros r s n h1 h2 h3
  sorry

end calculate_r_when_n_is_3_l143_143173


namespace min_sum_of_squares_of_roots_l143_143572

/-- 
Mathematically equivalent proof problem:
Let \( x_1 \) and \( x_2 \) be the two real roots of the equation \( 2x^2 - 4mx + 2m^2 + 3m - 2 = 0 \).
Prove that the minimum value of \( x_1^2 + x_2^2 \) occurs at \( m = \frac{2}{3} \) and this minimum value is \( \frac{8}{9} \).
--/
theorem min_sum_of_squares_of_roots :
  ∃ (m : ℝ), (m = 2 / 3) ∧ 
    ∀ (x₁ x₂ : ℝ), 
      (2 * x₁^2 - 4 * m * x₁ + 2 * m^2 + 3 * m - 2 = 0) ∧ 
      (2 * x₂^2 - 4 * m * x₂ + 2 * m^2 + 3 * m - 2 = 0) → 
        x₁^2 + x₂^2 = 8 / 9 :=
begin
  sorry
end

end min_sum_of_squares_of_roots_l143_143572


namespace events_A_B_equal_prob_l143_143510

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l143_143510


namespace count_valid_3x3_grids_l143_143752

def isValidGrid (grid : List (List ℕ)) : Prop :=
  (∀ i j k : ℕ, (i < 3) ∧ (j < 3) ∧ (k < 3) ∧ (i ≠ j) ∧ (j ≠ k) ∧ (i ≠ k) →
    (grid[i][0] + grid[j][1] + grid[k][2] = 15))

theorem count_valid_3x3_grids : ∃ count : ℕ, (count = 72) ∧ (∃ grid : List (List ℕ),
  (length grid = 3) ∧ (∀ row : List ℕ, row ∈ grid → length row = 3) ∧ 
  (∀ n : ℕ, (1 ≤ n ∧ n ≤ 9) → ∃! row : List ℕ, row ∈ grid ∧ n ∈ row) ∧
  isValidGrid grid ) :=
sorry

end count_valid_3x3_grids_l143_143752


namespace sum_consecutive_not_power_of_two_l143_143820

theorem sum_consecutive_not_power_of_two :
  ∀ n k : ℕ, ∀ x : ℕ, n > 0 → k > 0 → (n * (n + 2 * k - 1)) / 2 ≠ 2 ^ x := by
  sorry

end sum_consecutive_not_power_of_two_l143_143820


namespace speed_upstream_l143_143732

-- Conditions definitions
def speed_of_boat_still_water : ℕ := 50
def speed_of_current : ℕ := 20

-- Theorem stating the problem
theorem speed_upstream : (speed_of_boat_still_water - speed_of_current = 30) :=
by
  -- Proof is omitted
  sorry

end speed_upstream_l143_143732


namespace cone_radius_l143_143068

theorem cone_radius (r l : ℝ) 
  (surface_area_eq : π * r^2 + π * r * l = 12 * π)
  (net_is_semicircle : π * l = 2 * π * r) : 
  r = 2 :=
by
  sorry

end cone_radius_l143_143068


namespace find_x_l143_143867

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l143_143867


namespace factorization_correct_l143_143408

theorem factorization_correct {x : ℝ} : (x - 15)^2 = x^2 - 30*x + 225 :=
by
  sorry

end factorization_correct_l143_143408


namespace triangle_is_right_triangle_l143_143924

theorem triangle_is_right_triangle
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (tangent_condition : IsTangent (λ p q : ℝ => a * p + b * q + c) (λ p q : ℝ => p^2 + q^2 - 1)) :
  triangle.is_right_triangle (abs a) (abs b) (abs c) := 
sorry

end triangle_is_right_triangle_l143_143924


namespace expansion_coeff_l143_143094

theorem expansion_coeff (a b : ℝ) (x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x^2 + a^5 * x^5) :
  b = 40 :=
sorry

end expansion_coeff_l143_143094


namespace Isaiah_types_more_than_Micah_l143_143589

theorem Isaiah_types_more_than_Micah :
  let minutes_in_hour := 60
  let hours_in_day := 24
  let days_in_week := 7
  let minutes_in_week := minutes_in_hour * hours_in_day * days_in_week
  let Micah_words_per_minute := 35
  let Isaiah_words_per_minute := 120
  let Micah_words_per_week := Micah_words_per_minute * minutes_in_week
  let Isaiah_words_per_week := Isaiah_words_per_minute * minutes_in_week
  Isaiah_words_per_week - Micah_words_per_week = 856,800 := 
by
  sorry

end Isaiah_types_more_than_Micah_l143_143589


namespace rationalize_sqrt_fraction_denom_l143_143614

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143614


namespace angle_bisectors_intersect_equiv_l143_143708

variables {A B C D : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (a b c d : A)

-- Define a convex quadrilateral and angle bisectors intersections
def convex_quadrilateral (a b c d : A) :=
convex a b c d ∧ intersect (angle_bisector a c) (line b d) ∧ intersect (angle_bisector b d) (line a c)

-- Define the proportion condition
def proportion_condition (a b c d : A) :=
(b.distance a)/(c.distance b) = (d.distance a)/(d.distance c)

-- Proving the equivalence of angle bisectors intersection condition
theorem angle_bisectors_intersect_equiv
(conv_quad : convex_quadrilateral a b c d) :
(intersect (angle_bisector a c) (line b d) ↔ intersect (angle_bisector b d) (line a c))
∧ proportion_condition a b c d :=
sorry

end angle_bisectors_intersect_equiv_l143_143708


namespace prob_BD_gt_8_is_0_l143_143535

theorem prob_BD_gt_8_is_0
  (A B C P D : Point)
  (h_tri_ABC : isosceles_right_triangle A B C)
  (h_angle_BAC : ∠BAC = 90)
  (h_side_AB : dist A B = 12)
  (h_random_P : P ∈ interior_triangle A B C)
  (h_extend_BP_meet_AC : ∃ D, line BP ∩ line AC = {D}) :
  probability (BD > 8) = 0 :=
sorry

end prob_BD_gt_8_is_0_l143_143535


namespace exists_triangle_with_smallest_angle_l143_143031

theorem exists_triangle_with_smallest_angle (P : Fin 6 → ℝ × ℝ) (h_nocollinear : ∀ i j k : Fin 6, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (P i) (P j) (P k)) :
  ∃ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 30 ∧
    (∃ a b c : ℝ × ℝ, P i = a ∧ P j = b ∧ P k = c ∧ ∃ α β γ: ℝ, ∠ (a, b, c) = θ ∨ ∠ (b, c, a) = θ ∨ ∠ (c, a, b) = θ) :=
sorry

end exists_triangle_with_smallest_angle_l143_143031


namespace problem_solution_l143_143998

noncomputable def smaller_root_x (p q : ℤ) : ℝ :=
  let x := (9 - Real.sqrt 77) / 6
  x ^ 3 - (p - Real.sqrt q)

theorem problem_solution :
  ∃ (p q : ℤ),
    (smaller_root_x p q = 0) ∧ (p + q = 26) :=
begin
  sorry
end

end problem_solution_l143_143998


namespace probability_equality_l143_143506

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l143_143506


namespace smallest_n_satisfies_l143_143082

noncomputable def a : ℕ+ → ℚ
| 1 := 10
| (n+1) := -1 / 2 * a n + 3 / 2

def b (n : ℕ+) : ℚ := a n - 1

def S : ℕ+ → ℚ
| 1 := b 1
| (n+1) := S n + b (n + 1)

theorem smallest_n_satisfies :
  ∃ n : ℕ, n > 0 ∧ |S n - 6| < 1 / 170 ∧ ∀ m : ℕ, m > 0 → m < n → ¬(|S m - 6| < 1 / 170) :=
begin
  sorry
end

end smallest_n_satisfies_l143_143082


namespace find_comparison_speed_l143_143775

-- Declare the known constant car speed
def car_speed_kph : ℕ := 80

-- Given that it takes 5 seconds longer to travel 1 km at car_speed_kph than at speed v
def time_difference := 5  -- seconds

-- Convert the car speed from km/hour to km/second
def car_speed_kps : ℝ := car_speed_kph / 3600.0

-- Time to travel 1 km at car speed
def time_at_car_speed : ℝ := 1.0 / car_speed_kps

-- Unknown speed v in km/hour
variable (v : ℝ)

-- Time to travel 1 km at speed v
def time_at_v : ℝ := 1.0 / (v / 3600.0)

-- Equation based on the given condition
def equation (v : ℝ) : Prop := time_at_car_speed = time_at_v + time_difference

-- The statement to prove the speed v is 90 km/hour
theorem find_comparison_speed (v : ℝ) (h: equation v) : v = 90 :=
sorry

end find_comparison_speed_l143_143775


namespace maximum_sum_of_squares_l143_143057

theorem maximum_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 20 :=
sorry

end maximum_sum_of_squares_l143_143057


namespace mean_median_difference_l143_143959

theorem mean_median_difference :
  let scores := [
    (0.15 : ℝ, 60 : ℝ),
    (0.25 : ℝ, 75 : ℝ),
    (0.35 : ℝ, 85 : ℝ),
    (0.20 : ℝ, 95 : ℝ),
    (0.05 : ℝ, 110 : ℝ)
  ] in
  let mean := (scores.map (λ (p : ℝ × ℝ), p.1 * p.2)).sum in
  let sorted_scores := List.sort (≤) (scores.map (λ (p : ℝ × ℝ), p.2)) in
  let median := sorted_scores[sorted_scores.length / 2] in
  (median - mean) = 3 :=
by
  -- Skipping proof steps for now
  sorry

end mean_median_difference_l143_143959


namespace cyclic_quadrilateral_arithmetic_progression_exists_cyclic_quadrilateral_geometric_progression_not_exists_l143_143378

theorem cyclic_quadrilateral_arithmetic_progression_exists (α d : ℝ) (h1 : 0 < d) :
  ∃ (α : ℝ), is_cyclic_quadrilateral (α, α + d, α + 2 * d, α + 3 * d) :=
sorry

theorem cyclic_quadrilateral_geometric_progression_not_exists (α r : ℝ) (h1 : r ≠ 1) :
  ¬(∃ (α : ℝ), is_cyclic_quadrilateral (α, α * r, α * r^2, α * r^3)) :=
sorry

end cyclic_quadrilateral_arithmetic_progression_exists_cyclic_quadrilateral_geometric_progression_not_exists_l143_143378


namespace transformation_correct_l143_143950

noncomputable def transform_sin_function (x : ℝ) : ℝ :=
  sin (2 * x)

noncomputable def transformed_sin_function (x : ℝ) : ℝ :=
  sin (x + π / 4)

theorem transformation_correct :
  ∀ y : ℝ, 
  (∃ x : ℝ, transform_sin_function x = y) ↔ 
  (∃ x : ℝ, transformed_sin_function (2 * (x - π / 8)) = y) :=
sorry

end transformation_correct_l143_143950


namespace remaining_statues_weight_l143_143088

-- Conditions
def initial_marble := 80
def weight_first_statue := 10
def weight_second_statue := 18
def weight_discarded_marble := 22

-- Question as a Lean statement
theorem remaining_statues_weight:
  let remaining_marble := initial_marble - (weight_first_statue + weight_second_statue + weight_discarded_marble) in
  let weight_each_remaining_statue := remaining_marble / 2 in
  weight_each_remaining_statue = 15 :=
by
  sorry

end remaining_statues_weight_l143_143088


namespace exists_natural_number_plane_assignment_l143_143553

theorem exists_natural_number_plane_assignment :
  ∃ f : ℤ × ℤ → ℕ,
    (∀ n : ℕ, ∃ p : ℤ × ℤ, f p = n) ∧
    (∀ (a b c : ℤ) (h₁ : a ≠ 0 ∨ b ≠ 0) (h₂ : c ≠ 0),
       ∀ p₁ p₂ : ℤ × ℤ,
       a * p₁.1 + b * p₁.2 = c → a * p₂.1 + b * p₂.2 = c →
       ∃ d : ℤ, f p₁ = f (p₁.1 + d * b, p₁.2 - d * a) ∧ f p₂ = f (p₂.1 + d * b, p₂.2 - d * a)) :=
by {
  sorry
}

end exists_natural_number_plane_assignment_l143_143553


namespace invalid_votes_percentage_l143_143130

theorem invalid_votes_percentage (V B A total_valid_votes invalid_votes : ℕ) (h1 : V = 7720)
  (h2 : B = 2509) (h3 : A = B + 15 * V / 100) (h4 : total_valid_votes = A + B)
  (h5 : invalid_votes = V - total_valid_votes) :
  (invalid_votes * 100 / V : ℝ) ≈ 19.979 := 
by sorry

end invalid_votes_percentage_l143_143130


namespace sum_of_divisors_of_24_l143_143283

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range (24 + 1))), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l143_143283


namespace rationalize_denominator_l143_143661

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143661


namespace max_g_value_range_of_m_compare_values_l143_143923

-- Proof for maximum value of g(x)
theorem max_g_value :
  ∀ x > 0, ln (x) - x + 2 ≤ 1 := 
sorry

-- Proof for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x ≥ 1, m * ln x ≥ (x - 1) / (x + 1)) → m ≥ 1 / 2 :=
sorry

-- Proof for comparing f(tan(alpha)) and -cos(2*alpha)
theorem compare_values (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let x := tan α in 
  if hα1 : α < π / 4 then ln x < -cos (2 * α)
  else if hα2 : α = π / 4 then ln x = -cos (2 * α)
  else ln x > -cos (2 * α) :=
sorry

end max_g_value_range_of_m_compare_values_l143_143923


namespace rationalize_denominator_l143_143662

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143662


namespace dihedral_angle_prism_l143_143269

noncomputable def dihedral_angle_AB (A B C A₁ B₁ C₁: Point) (cylinder : Cylinder) : ℝ :=
  if h : (** A and B lie on the axis of the cylinder **) ∧
         (** C, A₁, B₁, and C₁ lie on the lateral surface of the cylinder **) ∧
         (** The base of the prism is a regular polygon **)
  then 120
  else 0

theorem dihedral_angle_prism (A B C A₁ B₁ C₁: Point) (cylinder : Cylinder) :
  ((** A and B lie on the axis of the cylinder **) ∧
   (** C, A₁, B₁, and C₁ lie on the lateral surface of the cylinder **) ∧
   (** The base of the prism is a regular polygon **))
  → dihedral_angle_AB A B C A₁ B₁ C₁ cylinder = 120 :=
by { sorry }

end dihedral_angle_prism_l143_143269


namespace function_value_proof_l143_143179

theorem function_value_proof (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : ∀ x, f (x + 1) = -f (-x + 1))
    (h2 : ∀ x, f (x + 2) = f (-x + 2))
    (h3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b)
    (h4 : ∀ x y : ℝ, x - y - 3 = 0)
    : f (9/2) = 5/4 := by
  sorry

end function_value_proof_l143_143179


namespace rationalize_sqrt_fraction_l143_143639

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143639


namespace determine_function_l143_143831

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (1/2) * (x^2 + (1/x)) else 0

theorem determine_function (f: ℝ → ℝ) (h : ∀ x ≠ 0, (1/x) * f (-x) + f (1/x) = x ) :
  ∀ x ≠ 0, f x = (1/2) * (x^2 + (1/x)) :=
by
  sorry

end determine_function_l143_143831


namespace quadratic_no_real_roots_l143_143107

theorem quadratic_no_real_roots (m : ℝ) (h : ∀ x : ℝ, x^2 - m * x + 1 ≠ 0) : m = 0 :=
by
  sorry

end quadratic_no_real_roots_l143_143107


namespace math_problem_l143_143955

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.cos (Real.pi / 2 - x) - 
  Real.sqrt 3 * Real.sin (Real.pi + x) * Real.cos x + 
  Real.sin (Real.pi / 2 + x) * Real.cos x

theorem math_problem
  (a b c A B C : ℝ)
  (h1 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h2 : (a - c) * (a + c) = b * (b - c))
  (h3 : 0 < A ∧ A < Real.pi)
  (h4 : 0 < C ∧ C < Real.pi)
  (hB : B = Real.pi - A - C) : 
  ∃ T, T = Real.pi ∧ (f B = 5 / 2) := 
sorry

end math_problem_l143_143955


namespace lowest_degree_polynomial_l143_143014

def is_polynomial (p : ℤ[X]) :=
  ∃ (a : ℤ) (n l k : ℤ), p = a * (X - 1)^l * (X + 1)^k ∧ l + 2 * k = n

theorem lowest_degree_polynomial
  (p : ℤ[X]) 
  (h1 : ∀ c ∈ p.coefficients.toFinset, c ∈ ℤ)  -- The coefficients of \( p(x) \) are integers.
  (h2 : ∃ q : list (ℤ[X]), p = q.prod ∧ ∀ p ∈ q, p.degree = 1) -- \( p(x) \) can be factored into a product of first-degree polynomials.
  (h3 : ∀ x ∈ p.root_set ℚ, x ∈ ℤ)  -- The roots of \( p(x) \) are integers.
  (h4 : p.eval 0 = -1) -- \( p(0) = -1 \).
  (h5 : p.eval 3 = 128) -- \( p(3) = 128 \).
  :
  p = X^4 + 2*X^3 - 2*X - 1 ∧ ∀ q, q satisfies h1, h2, h3, h4, h5 → q.degree ≥ 4 := 
begin
  sorry
end

end lowest_degree_polynomial_l143_143014


namespace total_subjects_l143_143191

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end total_subjects_l143_143191


namespace max_xy_l143_143167

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) : 
  x * y = 25.92 := 
sorry

end max_xy_l143_143167


namespace rationalize_denominator_l143_143677

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143677


namespace sin_double_angle_identity_l143_143095

theorem sin_double_angle_identity (alpha : ℝ) (h : Real.cos (Real.pi / 4 - alpha) = -4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l143_143095


namespace rationalize_sqrt_5_over_12_l143_143647

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143647


namespace total_rounds_in_game_l143_143125

open Nat

theorem total_rounds_in_game 
  (initial_tokens_A : Nat := 20)
  (initial_tokens_B : Nat := 19)
  (initial_tokens_C : Nat := 18)
  (round_loss : Nat := 6)
  (round_gain : Nat := 2)
  (game_end_condition : ∀ (a b c : Nat), a = 0 ∨ b = 0 ∨ c = 0) : 
  ∃ (rounds : Nat), rounds = 28 := 
begin
  sorry
end

end total_rounds_in_game_l143_143125


namespace cond_prob_l143_143159

noncomputable theory

open probability_theory

variables {Ω : Type*} {P : measure Ω} [probability_measure P]

def binomial_pmf (n : ℕ) (p : ℝ) : pmf (fin (n + 1)) :=
pmf.of_finset (finset.range (n + 1))
  (λ k, nat.choose n k * p ^ k * (1 - p) ^ (n - k))
  (by {
    simp_rw [nat.cast_mul, ← nnreal.coe_pow, ← nnreal.coe_mul, pmf.coe_of_finset,
    finset.sum_fin_eq_sum_finset],
    norm_cast,
    exact nat.mul_sum_binom.symm (1 - p) p (by linarith),
  })

variables {n p : ℕ} [hp : fact (0 < p ∧ p < 1)]

def X : pmf (fin (n + 1)) := binomial_pmf n p
def Y : pmf (fin (n + 1)) := binomial_pmf n p

theorem cond_prob (X Y : pmf (fin (n + 1))) (m k : ℕ) 
  (h : k ≤ m ∧ m - k ≤ n ∧ k ≤ n) :
  (X.bind (λ x, (Y.bind (λ y, if x + y = m then pmf.pure (x, y) else ⊥)))).support 
  = { (k, m - k) } →
  P[X = k | X + Y = m] = ↑(nat.choose n k) * ↑(nat.choose n (m - k)) / ↑(nat.choose (2 * n) m) :=
sorry

end cond_prob_l143_143159


namespace triangle_median_altitude_l143_143142

theorem triangle_median_altitude
    (A B C G P : Point)
    (hAB : dist A B = 7)
    (hAC : dist A C = 24)
    (hBC : dist B C = 25)
    (is_centroid : centroid A B C G)
    (is_altitude : foot G (line B C) = P) :
    dist G P = 56 / 25 := 
sorry

end triangle_median_altitude_l143_143142


namespace find_A_B_l143_143007

theorem find_A_B (A B : ℚ) :
  (∀ x : ℚ, 6 * x + 3 = A * (x + 4) + B * (x - 12)) →
  (x^2 - 8 * x - 48 = (x - 12) * (x + 4)) →
  (A = 75 / 16) ∧ (B = 21 / 16) :=
by
  intros h1 h2
  have hA : A = 75 / 16, by sorry
  have hB : B = 21 / 16, by sorry
  exact ⟨hA, hB⟩

end find_A_B_l143_143007


namespace solution_set_for_quadratic_inequality_l143_143114

theorem solution_set_for_quadratic_inequality (a m : ℝ) (h1 : a > 0) (h2 : ∀ x, ax^2 - 6x + a^2 < 0 ↔ 1 < x ∧ x < m) : m = 2 :=
sorry

end solution_set_for_quadratic_inequality_l143_143114


namespace total_capsules_sold_in_2_weeks_l143_143934

-- Define the conditions as constants
def Earnings100mgPerWeek := 80
def CostPer100mgCapsule := 5
def Earnings500mgPerWeek := 60
def CostPer500mgCapsule := 2

-- Theorem to prove the total number of capsules sold in 2 weeks
theorem total_capsules_sold_in_2_weeks : 
  (Earnings100mgPerWeek / CostPer100mgCapsule) * 2 + (Earnings500mgPerWeek / CostPer500mgCapsule) * 2 = 92 :=
by
  sorry

end total_capsules_sold_in_2_weeks_l143_143934


namespace find_a_l143_143525

variable a : ℕ

-- Define the side lengths of the triangle
def side_lengths := (a, a + 1, a + 2)

-- Triangle \(ABC\) is obtuse
def is_obtuse_triangle (a : ℕ) (b : ℕ) (c : ℕ) : Prop := 
  c^2 > a^2 + b^2

-- Define that \(a \le b \le c\) (although implicitly, we follow the given condition a ≤ a + 1 ≤ a + 2)
def triangle_abc_is_obtuse : Prop := is_obtuse_triangle a (a + 1) (a + 2)

theorem find_a (h : triangle_abc_is_obtuse) : a = 2 :=
  sorry

end find_a_l143_143525


namespace average_age_of_town_population_l143_143534

theorem average_age_of_town_population
  (children adults : ℕ)
  (ratio_condition : 3 * adults = 2 * children)
  (avg_age_children : ℕ := 10)
  (avg_age_adults : ℕ := 40) :
  ((10 * children + 40 * adults) / (children + adults) = 22) :=
by
  sorry

end average_age_of_town_population_l143_143534


namespace part1_part2_part3_l143_143461

def f (x : ℝ) := (1 + (Real.sqrt 3) * Real.tan x) * (Real.cos x) ^ 2

theorem part1 (α : ℝ) (h1 : α ∈ (Set.Ioo (π / 2) π)) (h2 : Real.sin α = (Real.sqrt 6) / 3) : 
  f α = (1 - Real.sqrt 6) / 3 := 
by sorry

theorem part2 : 
  (Set.Ioo 0 π).subset (Set.restrict (fun x => f x) (Set.Ioo 0 (2 * π))).domain := 
by sorry

theorem part3 : 
  Set.range (fun x => f x) = Set.Icc (-1 / 2) (3 / 2) := 
by sorry

end part1_part2_part3_l143_143461


namespace rationalize_sqrt_fraction_l143_143637

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143637


namespace average_speed_stan_l143_143717

theorem average_speed_stan (d1 d2 : ℝ) (h1 h2 rest : ℝ) (total_distance total_time : ℝ) (avg_speed : ℝ) :
  d1 = 350 → 
  d2 = 400 → 
  h1 = 6 → 
  h2 = 7 → 
  rest = 0.5 → 
  total_distance = d1 + d2 → 
  total_time = h1 + h2 + rest → 
  avg_speed = total_distance / total_time → 
  avg_speed = 55.56 :=
by 
  intros h_d1 h_d2 h_h1 h_h2 h_rest h_total_distance h_total_time h_avg_speed
  sorry

end average_speed_stan_l143_143717


namespace decreasing_function_l143_143804

theorem decreasing_function :
  (∀ x > 0, -|x-1| > -|x-1 + 1|) ∧
  (∀ x > 0, x^2 - 2*x + 4 > (x+1)^2 - 2*(x+1) + 4) ∧
  (∀ x > 0, ln (x+2) > ln ((x+1)+2)) ∧
  (∀ x > 0, (1 / 2)^x < (1 / 2)^(x+1)) :=
by sorry

end decreasing_function_l143_143804


namespace lines_parallel_l143_143911

theorem lines_parallel (a : ℝ) :
  (ax + 3y + 2a = 0) ∧ (2x + (a+1)y - 2 = 0) →
  (a = 2 ∨ a = -3) :=
by
  sorry

end lines_parallel_l143_143911


namespace smallest_n_l143_143567

theorem smallest_n (n : ℕ) (h1 : n > 0) (h2 : n % 2010 = 0) (h3 : (∃! d : ℕ, d < 10 ∧ even_digit d ∧ in_digits_of_n n d)) : n = 311550 :=
sorry

def even_digit (d : ℕ) : Prop :=
d % 2 = 0

def in_digits_of_n (n d : ℕ) : Prop :=
∃ k, d = (n / (10 ^ k)) % 10

end smallest_n_l143_143567


namespace hillary_climbing_rate_l143_143935

theorem hillary_climbing_rate (H : ℝ) (start_time : ℝ := 6) (pass_time : ℝ := 12) : 
  (4700 - 700) / (pass_time - start_time) = 2000 / 3 → H = 2000 / 3 :=
by
  intro h1
  exact h1

end hillary_climbing_rate_l143_143935


namespace quarters_given_l143_143703

theorem quarters_given (initial_quarters : ℕ) (final_quarters : ℕ) : 
    initial_quarters = 783 → final_quarters = 1054 → final_quarters - initial_quarters = 271 :=
by
  assume h1 : initial_quarters = 783
  assume h2 : final_quarters = 1054
  rw [h1, h2]
  exact rfl

# This line will be skipped as per the user instruction to write the statement only without proof.

end quarters_given_l143_143703


namespace complex_point_coordinates_l143_143137

theorem complex_point_coordinates :
  let z : ℂ := (2 + 3 * complex.I) / complex.I
  (z.re, z.im) = (3, -2) := sorry

end complex_point_coordinates_l143_143137


namespace simplify_fraction_l143_143216

-- Define the original expressions
def expr1 := 3 / (Real.sqrt 5 + 2)
def expr2 := 4 / (Real.sqrt 7 - 2)

-- State the mathematical problem.
theorem simplify_fraction :
  (1 / (expr1 + expr2)) =
  ((9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) / 
  ((9 * Real.sqrt 5 + 4 * Real.sqrt 7) ^ 2 - 100)) :=
by sorry

end simplify_fraction_l143_143216


namespace josef_timothy_game_l143_143987

theorem josef_timothy_game :
  {n : ℕ | n > 0 ∧ n ≤ 500 ∧ 500 % n = 0}.card = 12 :=
by
  sorry

end josef_timothy_game_l143_143987


namespace rationalize_sqrt_fraction_l143_143665

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143665


namespace rationalize_denominator_l143_143654

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143654


namespace largest_number_among_l143_143808

theorem largest_number_among (π: ℝ) (sqrt_2: ℝ) (neg_2: ℝ) (three: ℝ)
  (h1: 3.14 ≤ π)
  (h2: 1 < sqrt_2 ∧ sqrt_2 < 2)
  (h3: neg_2 < 1)
  (h4: 3 < π) :
  (neg_2 < sqrt_2) ∧ (sqrt_2 < 3) ∧ (3 < π) :=
by {
  sorry
}

end largest_number_among_l143_143808


namespace domain_of_f_l143_143414

noncomputable def f (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (2 * x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = f x} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_f_l143_143414


namespace train_speed_first_part_l143_143356

variables (x V : ℝ)

def train_time_first_part (x V : ℝ) : ℝ := x / V
def train_time_second_part (x : ℝ) : ℝ := 2 * x / 20
def total_train_time (x : ℝ) : ℝ := 3 * x / 27

theorem train_speed_first_part :
  train_time_first_part x V + train_time_second_part x = total_train_time x → V = 90 := by sorry

end train_speed_first_part_l143_143356


namespace find_x_l143_143874

open Nat

def has_three_distinct_prime_factors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a * b * c

theorem find_x (n : ℕ) (h₁ : Odd n) (h₂ : 7^n + 1 = x)
  (h₃ : has_three_distinct_prime_factors x) (h₄ : 11 ∣ x) : x = 16808 := by
  sorry

end find_x_l143_143874


namespace num_wave_numbers_eq_sixteen_l143_143345

def is_wave_number (a b c d e : ℕ) : Prop :=
  (b > a) ∧ (b > c) ∧ (d > c) ∧ (d > e)

def without_repetition (x y z w v: ℕ) : Prop :=
  list.nodup [x, y, z, w, v]

def wave_numbers : set (ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {p | let (a, b, c, d, e) := p in is_wave_number a b c d e ∧ without_repetition a b c d e}

theorem num_wave_numbers_eq_sixteen :
  finset.card (finset.filter (λ p : ℕ × ℕ × ℕ × ℕ × ℕ, ∃ a b c d e, p = (a, b, c, d, e) ∧
    p ∈ wave_numbers) (finset.univ : finset (ℕ × ℕ × ℕ × ℕ × ℕ))) = 16 :=
sorry

end num_wave_numbers_eq_sixteen_l143_143345


namespace repeating_decimal_0_42_153_as_fraction_l143_143303

noncomputable theory

def repeating_decimal_as_fraction (d : ℚ) := ∃ x : ℕ, d = x / 99900

theorem repeating_decimal_0_42_153_as_fraction :
  repeating_decimal_as_fraction (42 / 100 + 153 / 99900) := by
  sorry

end repeating_decimal_0_42_153_as_fraction_l143_143303


namespace probability_equality_l143_143508

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l143_143508


namespace digit_agreement_l143_143730

theorem digit_agreement (N : ℕ) (abcd : ℕ) (h1 : N % 10000 = abcd) (h2 : N ^ 2 % 10000 = abcd) (h3 : ∃ a b c d, abcd = a * 1000 + b * 100 + c * 10 + d ∧ a ≠ 0) : abcd / 10 = 937 := sorry

end digit_agreement_l143_143730


namespace support_percentage_l143_143961

theorem support_percentage (men women : ℕ) (support_men_percentage support_women_percentage : ℝ) 
(men_support women_support total_support : ℕ)
(hmen : men = 150) 
(hwomen : women = 850) 
(hsupport_men_percentage : support_men_percentage = 0.55) 
(hsupport_women_percentage : support_women_percentage = 0.70) 
(hmen_support : men_support = 83) 
(hwomen_support : women_support = 595)
(htotal_support : total_support = men_support + women_support) :
  ((total_support : ℝ) / (men + women) * 100) = 68 :=
by
  -- Insert the proof here to verify each step of the calculation and rounding
  sorry

end support_percentage_l143_143961


namespace find_largest_x_and_compute_ratio_l143_143013

theorem find_largest_x_and_compute_ratio (a b c d : ℤ) (h : x = (a + b * Real.sqrt c) / d)
   (cond : (5 * x / 7) + 1 = 3 / x) : a * c * d / b = -70 :=
by
  sorry

end find_largest_x_and_compute_ratio_l143_143013


namespace sum_of_divisors_of_24_eq_60_l143_143297

theorem sum_of_divisors_of_24_eq_60 : 
  (∑ n in { n | n ∣ 24 ∧ 0 < n }.toFinset, n) = 60 := 
sorry

end sum_of_divisors_of_24_eq_60_l143_143297


namespace S9_value_l143_143061

variables (a_n : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
variable [arithmetic : ∀ n, a_n = λ n, a_n 1 + (n - 1) * d]

-- Given conditions
axiom a_3_7_condition : a_n 3 + a_n 7 = 8
axiom sum_formula : ∀ n, S n = (n / 2) * (a_n 1 + a_n n)

-- Objective to prove
theorem S9_value : S 9 = 36 :=
by
  -- To complete the proof here. Adding sorry to move on.
  sorry

end S9_value_l143_143061


namespace count_valid_configurations_l143_143221

/-- We have an L-shaped figure made of four squares. We can add one of the four lettered squares (A, B, C, D). -/
def L_shaped_figure : Type :=
  { figure : set (ℕ × ℕ) // 
    (∃ a b c d: ℕ × ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧
    figure = {a, b, c, d} ∧
    (∃ u v, u ≠ v ∧ (a = (0, 0) ∧ b = (0, 1) ∧ c = (1, 0) ∧ d = (1, 1)) ∨
    (a = (0, 0) ∧ b = (0, 1) ∧ c = (0, 2) ∧ d = (1, 1)) ∨
    (a = (0, 0) ∧ b = (1, 0) ∧ c = (1, 1) ∧ d = (2, 1)))) }

/-- A configuration is valid if it can be folded into a cubical box -/
def valid_configuration (figure : L_shaped_figure) (additional_square : ℕ × ℕ) : Prop :=
  -- define the rules for a valid configuration
  sorry

/-- The number of resulting figures that can be folded into a fully enclosed cubical box is 3. -/
theorem count_valid_configurations : (∀ f : L_shaped_figure, (∃ asq : ℕ × ℕ, valid_configuration f asq)) → 
                                ∃ valid_count : ℕ, valid_count = 3 :=
by
  -- set the proper configurations and prove that it can only be 3 valid count
  sorry

end count_valid_configurations_l143_143221


namespace rationalize_denominator_l143_143660

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143660


namespace sin_2alpha_value_l143_143427

variable (α : Real)

theorem sin_2alpha_value (h : (1 - tan α) / (1 + tan α) = 3 - 2 * Real.sqrt 2) :
  sin (2 * α) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2alpha_value_l143_143427


namespace sum_of_abscissas_l143_143447

noncomputable def f (x : ℝ) : ℝ := sorry -- f is such that f(x+1) is an odd function
noncomputable def g (x : ℝ) : ℝ := real.log (real.sqrt (x ^ 2 - 2 * x + 2) - x + 1)

theorem sum_of_abscissas (x1 x2 x3 x4 x5 : ℝ) (h1 : f (x1) = g (x1)) (h2 : f (x2) = g (x2)) (h3 : f (x3) = g (x3)) (h4 : f (x4) = g (x4)) (h5 : f (x5) = g (x5)) :
  x1 + x2 + x3 + x4 + x5 = 5 :=
sorry

end sum_of_abscissas_l143_143447


namespace degree_le_of_lt_eventually_l143_143575

open Polynomial

theorem degree_le_of_lt_eventually {P Q : Polynomial ℝ} (h_exists : ∃ N : ℝ, ∀ x : ℝ, x > N → P.eval x < Q.eval x) :
  P.degree ≤ Q.degree :=
sorry

end degree_le_of_lt_eventually_l143_143575


namespace proj_norm_l143_143477

variable (v w : Vec)
variable norm_v : norm v = 5
variable norm_w : norm w = 8
variable dot_v_w : dot v w = 20

theorem proj_norm (v w : Vec) (norm_v : norm v = 5)
  (norm_w : norm w = 8) (dot_v_w : dot v w = 20) :
  (norm (proj w v) = 2.5) :=
sorry

end proj_norm_l143_143477


namespace find_x_l143_143869

theorem find_x (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p1 * p2 * p3 = 11 * (7^n + 1) ∧ p1.prime ∧ p2.prime ∧ p3.prime): 
  7^n + 1 = 16808 :=
begin
  sorry
end

end find_x_l143_143869


namespace sum_of_divisors_of_24_l143_143285

theorem sum_of_divisors_of_24 : ∑ d in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), d = 60 := by
  sorry

end sum_of_divisors_of_24_l143_143285


namespace tangent_line_parallel_x_axis_at_one_g_concave_l143_143073

noncomputable def f (x : ℝ) (a : ℝ) := log x - 1/x + a*x
noncomputable def f' (x : ℝ) (a : ℝ) := 1/x - 1/(x*x) + a

theorem tangent_line_parallel_x_axis_at_one (a : ℝ) (h : f'(1, a) = 0) : a = 0 := by
  calc
    f'(1, 0) = 1 - 1 + a : by simp [f']; ring
    ...     = a : by simp [f']; ring
    ...     = 0 : h

lemma monotonicity_conditions (x : ℝ) (a : ℝ) (h_pos : 0 < x) : 
  f' x a > 0 ↔ a ≥ 0 ∨ (a < 0 ∧ (a*x^2 + x + 1) > 0) := by
  sorry

noncomputable def g (x : ℝ) (a : ℝ) := x*x*f' x a

theorem g_concave (a : ℝ) (h_pos : 0 < a) :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → 
  g((x1 + x2) / 2, a) ≤ (g(x1, a) + g(x2, a)) / 2 := by
  calc
    g((x1 + x2) / 2, a) = a * ((x1 + x2)/2)^2 + ((x1 + x2)/2) + 1 : by simp [g, f']; ring
    ... ≤ (a * x1^2 + x1 + 1 + (a * x2^2 + x2 + 1)) / 2 : by linarith
    ... = (g(x1, a) + g(x2, a)) / 2 : by simp [g]; ring

end tangent_line_parallel_x_axis_at_one_g_concave_l143_143073


namespace pattern_C_forms_tetrahedron_l143_143851

-- Definition of the patterns; these are the conditions provided.
inductive Pattern
| A
| B
| C
| D

-- The main statement/request
theorem pattern_C_forms_tetrahedron :
  ∃ p: Pattern, p = Pattern.C ↔ fold_to_tetrahedron p :=
sorry

-- Auxiliary definition as needed
def fold_to_tetrahedron (p: Pattern) : Prop :=
  match p with
  | Pattern.A => false
  | Pattern.B => false
  | Pattern.C => true
  | Pattern.D => false

end pattern_C_forms_tetrahedron_l143_143851


namespace smallest_m_for_triangle_sides_l143_143885

noncomputable def is_triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_m_for_triangle_sides (a b c : ℝ) (h : is_triangle_sides a b c) :
  (a^2 + c^2) / (b + c)^2 < 1 / 2 := sorry

end smallest_m_for_triangle_sides_l143_143885


namespace rationalize_denominator_l143_143663

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143663


namespace statistical_measures_change_l143_143533

theorem statistical_measures_change
  (scores : Fin 9 → ℝ) :
  (∃ (new_mean new_range new_variance : ℝ),
    scores.mean ≠ new_mean ∧ scores.range ≠ new_range ∧ scores.variance ≠ new_variance) 
  ∧ scores.median = (Array.remove_indices (Array.ofFn scores)).median := 
sorry

end statistical_measures_change_l143_143533


namespace multiple_of_6_units_digit_2_less_150_count_l143_143092

theorem multiple_of_6_units_digit_2_less_150_count : 
  let multiples := {n : ℕ | ∃ k : ℕ, n = 6 * k ∧ n < 150 ∧ n % 10 = 2} in 
  multiples.to_finset.card = 2 :=
by
  sorry

end multiple_of_6_units_digit_2_less_150_count_l143_143092


namespace average_delivery_fee_l143_143329

theorem average_delivery_fee
  (p1 p2 : ℝ)
  (fee1 fee2 : ℝ)
  (h1 : p1 = 0.7)
  (h2 : p2 = 0.3)
  (h3 : fee1 = 4)
  (h4 : fee2 = 6)
  (h_sum : p1 + p2 = 1):
  (p1 * fee1 + p2 * fee2) = 4.6 :=
by
  rw [h1, h2, h3, h4]
  field_simp
  norm_num
  rw h_sum
  norm_num

end average_delivery_fee_l143_143329


namespace area_ratio_PQR_ABC_l143_143975

variables {A B C D E F P Q R : Type}
variables [incidence_geometry A] [incidence_geometry B] [incidence_geometry C]
variables [incidence_geometry D] [incidence_geometry E] [incidence_geometry F]
variables [incidence_geometry P] [incidence_geometry Q] [incidence_geometry R]

noncomputable def ratio_BD_DC : ℝ := 3 / 2
noncomputable def ratio_CE_EA : ℝ := 3 / 2
noncomputable def ratio_AF_FB : ℝ := 3 / 2

-- Setting up the conditions
axiom condition_BD_DC : ∀ (BD DC : ℝ), ratio_BD_DC = BD / DC
axiom condition_CE_EA : ∀ (CE EA : ℝ), ratio_CE_EA = CE / EA
axiom condition_AF_FB : ∀ (AF FB : ℝ), ratio_AF_FB = AF / FB

-- Main statement
theorem area_ratio_PQR_ABC : 
  -- If D, E, F are on sides BC, CA, AB respectively with given point-division ratios
  -- And lines AD, BE, CF intersect at P, Q, R respectively
  -- Then the area ratio [PQR] / [ABC] = 1 / 4
  (∃ (D E F P Q R : Type),
     condition_BD_DC ∧ condition_CE_EA ∧ condition_AF_FB ∧
     (AD P) ∧ (BE Q) ∧ (CF R)) →
  (area PQR / area ABC = 1 / 4) :=
sorry

end area_ratio_PQR_ABC_l143_143975


namespace johnny_practice_l143_143150

variable (P : ℕ) -- Current amount of practice in days
variable (h : P = 40) -- Given condition translating Johnny's practice amount
variable (d : ℕ) -- Additional days needed

theorem johnny_practice : d = 80 :=
by
  have goal : 3 * P = P + d := sorry
  have initial_condition : P = 40 := sorry
  have required : d = 3 * 40 - 40 := sorry
  sorry

end johnny_practice_l143_143150


namespace rationalize_sqrt_fraction_l143_143630

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143630


namespace cricket_bat_cost_price_USD_l143_143789

noncomputable def costPriceInUSD : ℝ :=
let
    exchangeGBPToEUR : ℝ := 1.15,
    exchangeGBPToUSD : ℝ := 0.72,
    dFinalPriceInEUR : ℝ := 310, -- price D paid in Euros
    dDiscount : ℝ := 0.07,
    cProfit : ℝ := 0.30,
    cDiscount : ℝ := 0.10,
    bProfit : ℝ := 0.25,
    aProfit : ℝ := 0.20,
    aDiscount : ℝ := 0.05
in
    let dPriceBeforeDiscount : ℝ := dFinalPriceInEUR / (1 - dDiscount),
        cPriceBeforeProfit : ℝ := dPriceBeforeDiscount / (1 + cProfit),
        cPriceInGBP : ℝ := cPriceBeforeProfit / exchangeGBPToEUR,
        bPriceBeforeDiscount : ℝ := cPriceInGBP / (1 - cDiscount),
        bPriceBeforeProfit : ℝ := bPriceBeforeDiscount / (1 + bProfit),
        aPriceBeforeDiscount : ℝ := bPriceBeforeProfit / (1 - aDiscount),
        aCostPriceInGBP : ℝ := aPriceBeforeDiscount / (1 + aProfit),
        aCostPriceInUSD : ℝ := aCostPriceInGBP / exchangeGBPToUSD
    in aCostPriceInUSD

theorem cricket_bat_cost_price_USD : costPriceInUSD ≈ 241.50 := sorry

end cricket_bat_cost_price_USD_l143_143789


namespace treasure_in_box_2_l143_143981

def Box := Fin 5  -- We have 5 boxes indexed from 0 to 4.

-- Conditions: Box materials. True means cedar, False means sandalwood.
def is_cedar (n : Box) : Prop :=
  n = 0 ∨ n = 3 ∨ n = 4

-- Box inscriptions.
def inscription_1 (treasure_box : Box) : Prop :=
  treasure_box = 0 ∨ treasure_box = 3

def inscription_2 (treasure_box : Box) : Prop := 
  treasure_box = 0

def inscription_3 (treasure_box : Box) : Prop := 
  treasure_box = 2 ∨ treasure_box = 4

def inscription_4 (treasure_box : Box) : Prop := 
  treasure_box = 3 ∨ treasure_box = 4

def inscription_5 (statements : Box → Prop) : Prop :=
  ∀ n : Box, n ≠ 4 → ¬statements n

-- The number of false statements on cedar boxes is equal to the number of false statements on sandalwood boxes.
def equal_false_statements (conditions : Box → Prop) : Prop :=
  (∑ n in (Finset.filter is_cedar Finset.univ), if conditions n then 0 else 1) =
  (∑ n in (Finset.filter (λ n, ¬is_cedar n) Finset.univ), if conditions n then 0 else 1)

-- The main proof problem.
theorem treasure_in_box_2 : ∃ treasure_box : Box, treasure_box = 1 ∧ 
  (∃ (conditions : Box → Prop),
    conditions 0 ↔ inscription_1 treasure_box ∧
    conditions 1 ↔ inscription_2 treasure_box ∧
    conditions 2 ↔ inscription_3 treasure_box ∧
    conditions 3 ↔ inscription_4 treasure_box ∧
    conditions 4 ↔ inscription_5 conditions ∧
    equal_false_statements (λ n, ¬conditions n)) :=
begin
  sorry -- Proof goes here
end

end treasure_in_box_2_l143_143981


namespace man_rowed_distance_downstream_l143_143339

def distance_rowed_downstream : ℕ := 72

variables (V_s : ℕ) (t_upstream : ℕ) (d_upstream : ℕ) (t_downstream : ℕ) (D : ℕ)

axiom upstream_time : t_upstream = 3
axiom downstream_time : t_downstream = 3
axiom stream_speed : V_s = 7
axiom upstream_distance : d_upstream = 30
axiom effective_speed_upstream : V_b - V_s = d_upstream / t_upstream
axiom effective_speed_downstream : V_b + V_s = D / t_downstream

theorem man_rowed_distance_downstream :
  D = distance_rowed_downstream :=
sorry

end man_rowed_distance_downstream_l143_143339


namespace hyperbola_equation_l143_143906

theorem hyperbola_equation (a b : ℝ) (h1 : ∃ a b : ℝ, (∀ x y, x^2 / a^2 - y^2 / b^2 = 1)) 
  (h2 : ∃ F : ℝ × ℝ, F = (2, 0)) (h3 : ∀ x, y = x * sqrt(3) ∨ y = -x * sqrt(3)) : 
  ∃ a b : ℝ, x^2 - (y^2 / 3) = 1 :=
by
  sorry

end hyperbola_equation_l143_143906


namespace distance_A_B_minimum_distance_d_l143_143925

noncomputable def line (t : ℝ) : ℝ × ℝ :=
  (1 + 1/2 * t, sqrt 3 / 2 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (1/2 * Real.cos θ, sqrt 3 / 2 * Real.sin θ)

theorem distance_A_B : 
  let (A_x, A_y) := (1 : ℝ, 0 : ℝ),
      (B_x, B_y) := (1/2 : ℝ, - sqrt 3 / 2 : ℝ) in
  Real.sqrt (((A_x - B_x)^2) + (A_y - B_y)^2) = 1 :=
by
  sorry

theorem minimum_distance_d :
  let d (θ : ℝ) : ℝ := 
    abs (sqrt 3 / 2 * (Real.cos θ - Real.sin θ - 2)) / Real.sqrt 4,
      min_d := sqrt 6 / 4 * (sqrt 2 - 1) in
  ∀ theta : ℝ, d theta ≥ min_d :=
by
  sorry

end distance_A_B_minimum_distance_d_l143_143925


namespace seq_value_is_minus_30_l143_143434

open Nat  -- Open the natural numbers namespace

noncomputable def seq_condition (a : ℕ → ℤ) :=
  ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem seq_value_is_minus_30 (a : ℕ → ℤ) (h_seq : seq_condition a) (h_a2 : a 2 = -6) :
  a 10 = -30 :=
by 
  sorry

end seq_value_is_minus_30_l143_143434


namespace one_equation_does_not_pass_origin_l143_143090

def passes_through_origin (eq : ℝ → ℝ) : Prop := eq 0 = 0

def equation1 (x : ℝ) : ℝ := x^4 + 1
def equation2 (x : ℝ) : ℝ := x^4 + x
def equation3 (x : ℝ) : ℝ := x^4 + x^2
def equation4 (x : ℝ) : ℝ := x^4 + x^3

theorem one_equation_does_not_pass_origin :
  (¬ passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  ¬ passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  ¬ passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  ¬ passes_through_origin equation4) :=
sorry

end one_equation_does_not_pass_origin_l143_143090


namespace trucks_count_l143_143706

theorem trucks_count (x : ℕ) : 
  (∀ k : ℕ, (3 * k + 16 = 5 * (k - 1) + y ∧ 0 < y ∧ y < 5) → x = k) → x = 9 :=
by
  intros h
  have h1 := h 9
  have h2 : 3 * 9 + 16 = 5 * (9 - 1) + 4 := by norm_num
  have h3 : 0 < 4 ∧ 4 < 5 := by norm_num
  apply h1
  split
  exact h2
  exact h3

end trucks_count_l143_143706


namespace rationalize_denominator_l143_143653

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143653


namespace proj_norm_l143_143478

variable (v w : Vec)
variable norm_v : norm v = 5
variable norm_w : norm w = 8
variable dot_v_w : dot v w = 20

theorem proj_norm (v w : Vec) (norm_v : norm v = 5)
  (norm_w : norm w = 8) (dot_v_w : dot v w = 20) :
  (norm (proj w v) = 2.5) :=
sorry

end proj_norm_l143_143478


namespace angle_B_is_80_l143_143527

-- Given conditions
variables (A B C D x : ℝ) (h1 : A = 60) (h2 : B = 2 * C) (h3 : D = 2 * C - x) (h4 : x > 0)
variables (EPA EC EPD ED : ℝ) (h5 : EPA = 30) (h6 : EC = 30) (h7 : ∠EPD = 20)

-- Proving the angle B
theorem angle_B_is_80 (h8 : A + B + C + D = 360) : B = 80 :=
by
  sorry

end angle_B_is_80_l143_143527


namespace odd_function_shift_l143_143233

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  2 * sin (ω * x - π / 6)

theorem odd_function_shift (ω : ℝ) (h₁ : 0 < ω) (h₂ : ω < 3)
  (h₃ : f (π / 3) ω = 0) : 
  ∀ x, f (x + π / 3) (1 / 2) = 2 * sin (1 / 2 * x) :=
sorry

end odd_function_shift_l143_143233


namespace least_number_of_marbles_divisible_by_2_3_4_5_6_7_l143_143353

theorem least_number_of_marbles_divisible_by_2_3_4_5_6_7 : 
  ∃ n : ℕ, (∀ k ∈ [2, 3, 4, 5, 6, 7], k ∣ n) ∧ n = 420 :=
  by sorry

end least_number_of_marbles_divisible_by_2_3_4_5_6_7_l143_143353


namespace event_probabilities_equal_l143_143518

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l143_143518


namespace perimeter_one_face_of_cube_is_24_l143_143241

noncomputable def cube_volume : ℝ := 216
def perimeter_of_face_of_cube (V : ℝ) : ℝ := 4 * (V^(1/3) : ℝ)

theorem perimeter_one_face_of_cube_is_24 :
  perimeter_of_face_of_cube cube_volume = 24 := 
by
  -- This proof will invoke the calculation shown in the problem.
  sorry

end perimeter_one_face_of_cube_is_24_l143_143241


namespace hyperbola_focal_length_l143_143231

open Real

theorem hyperbola_focal_length :
  let a := sqrt 2
  let b := sqrt 3
  let c := sqrt (a^2 + b^2) in
  2 * c = 2 * sqrt 5 := 
by 
  sorry

end hyperbola_focal_length_l143_143231


namespace maximum_OD_OE_OF_l143_143175

-- Definitions for the problem
variable {A B C O D E F : Point}
variable [unit_circle : Circle A B C O]
variable [inside_triangle : Inside O (Triangle A B C)]
variable [projection_D : Projection O (Line B C) D]
variable [projection_E : Projection O (Line C A) E]
variable [projection_F : Projection O (Line A B) F]
variable [perpendicular_OD : Perpendicular O D (Line B C)]
variable [perpendicular_OE : Perpendicular O E (Line C A)]
variable [perpendicular_OF : Perpendicular O F (Line A B)]

-- Statement of the theorem
theorem maximum_OD_OE_OF : OD + OE + OF ≤ 3 / 2 := sorry

end maximum_OD_OE_OF_l143_143175


namespace corrected_mean_l143_143766

theorem corrected_mean (n : ℕ) (mean incorrect_observation correct_observation : ℝ) (h_n : n = 50) (h_mean : mean = 32) (h_incorrect : incorrect_observation = 23) (h_correct : correct_observation = 48) : 
  (mean * n + (correct_observation - incorrect_observation)) / n = 32.5 := 
by 
  sorry

end corrected_mean_l143_143766


namespace initial_mixture_volume_l143_143100

theorem initial_mixture_volume (x : ℝ) (hx1 : 0.10 * x + 10 = 0.28 * (x + 10)) : x = 40 :=
by
  sorry

end initial_mixture_volume_l143_143100


namespace scientists_lose_radio_contact_time_l143_143749

theorem scientists_lose_radio_contact_time :
  ∀ (v1 v2 R : ℝ) (T : ℝ),
    v1 = 20 →
    v2 = 30 →
    R = 125 →
    T = R / (v1 + v2) →
    T = 2.5 :=
by
  intros v1 v2 R T h_v1 h_v2 h_R h_T
  rw [h_v1, h_v2] at h_T
  rw h_R at h_T
  norm_num at h_T
  exact h_T

end scientists_lose_radio_contact_time_l143_143749


namespace union_A_B_complement_A_intersection_B_l143_143472

-- Definitions of sets A and B as subsets of real numbers
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Prove the union of A and B
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 10} :=
by {
  sorry
}

-- Complement of A in the universal set of real numbers
def complement_A : Set ℝ := {x | x < 1 ∨ x ≥ 7}

-- Prove the intersection of complement of A with B
theorem complement_A_intersection_B : complement_A ∩ B = {x | 7 ≤ x ∧ x < 10} :=
by {
  sorry
}

end union_A_B_complement_A_intersection_B_l143_143472


namespace coprime_sum_product_l143_143176

theorem coprime_sum_product (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a + b) (a * b) = 1 := by
  sorry

end coprime_sum_product_l143_143176


namespace ineq_a_2n_l143_143394

noncomputable def a : ℕ → ℕ
| 1 => 1
| (2 * n) := a (2 * n - 1) + a n
| (2 * n + 1) := a (2 * n)
| _ := 0

theorem ineq_a_2n : ∀ n : ℕ, n > 0 → a (2^n) > 2^((n^2) / 4) := 
by 
  intro n hn
  -- base case and inductive step go here
  sorry

end ineq_a_2n_l143_143394


namespace rationalize_sqrt_fraction_denom_l143_143613

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143613


namespace rationalize_sqrt_5_over_12_l143_143650

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143650


namespace arthur_bakes_muffins_l143_143810

-- Definitions of the conditions
def james_muffins : ℚ := 9.58333333299999
def multiplier : ℚ := 12.0

-- Statement of the problem
theorem arthur_bakes_muffins : 
  abs (multiplier * james_muffins - 115) < 1 :=
by
  sorry

end arthur_bakes_muffins_l143_143810


namespace triangle_ratio_l143_143550

theorem triangle_ratio (A B C D E : Type) [PlaneTriangle A B C] 
  (angleA : angle A = 60) (angleB : angle B = 45) 
  (D_on_AB : is_on_line D A B) (angleADE : angle D A E = 45)
  (area_div : area (triangle A D E) = area (triangle B C D E)) :
  ratio AD AB = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2) :=
sorry

end triangle_ratio_l143_143550


namespace rationalize_sqrt_fraction_denom_l143_143615

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143615


namespace surface_area_of_pyramid_l143_143063

def base_side_length : ℝ := 2
def lateral_edge_length : ℝ := 2

def base_area := base_side_length^2
def lateral_face_area := (sqrt 3 / 4) * lateral_edge_length^2
def total_surface_area := base_area + 4 * lateral_face_area

theorem surface_area_of_pyramid : total_surface_area = 4 + 4 * sqrt 3 :=
by
  -- Base area calculation
  have h_base_area : base_area = 4 := by
    simp [base_area, base_side_length]
  
  -- Lateral face area calculation
  have h_face_area : lateral_face_area = sqrt 3 := by
    simp [lateral_face_area, lateral_edge_length]

  -- Total surface area calculation
  have h_total_area : total_surface_area = 4 + 4 * sqrt 3 := by
    simp [total_surface_area, h_base_area, h_face_area]
  
  exact h_total_area

end surface_area_of_pyramid_l143_143063


namespace line_exists_l143_143178

theorem line_exists (x y x' y' : ℝ)
  (h1 : x' = 3 * x + 2 * y + 1)
  (h2 : y' = x + 4 * y - 3) : 
  (∃ A B C : ℝ, A * x + B * y + C = 0 ∧ A * x' + B * y' + C = 0 ∧ 
  ((A = 1 ∧ B = -1 ∧ C = 4) ∨ (A = 4 ∧ B = -8 ∧ C = -5))) :=
sorry

end line_exists_l143_143178


namespace total_candies_by_boys_invariant_l143_143529

theorem total_candies_by_boys_invariant (k : ℕ) (candies : ℕ) (children : List Bool) :
  candies = 1000 →
  children.length = k →
  ∀ child_order1 child_order2 : List Bool,
    (∀ (i : ℕ), (i < k → (child_order1.nth i = children.nth i ∨ child_order1.nth i = children.nth (k - i - 1)))
    → (child_order2.nth i = children.nth i ∨ child_order2.nth i = children.nth (k - i - 1)))) →
  let round_up (x : ℕ) (d : ℕ) := if x % d = 0 then x / d else x / d + 1,
      round_down (x : ℕ) (d : ℕ) := x / d in
  let take_candies := λ (order : List Bool) (total : ℕ) (n : ℕ), 
    List.foldl (λ (acc : ℕ × ℕ) (child_type : Bool), 
       if child_type then (acc.1 + round_up acc.2 n, acc.2 - round_up acc.2 n) 
       else (acc.1, acc.2 - round_down acc.2 n)) (0, total) order in
  let total_boys_candies := λ (order : List Bool), (take_candies order candies k).1 in
  total_boys_candies child_order1 = total_boys_candies child_order2 :=
by
  intros h_candies h_length h_perm
  sorry


end total_candies_by_boys_invariant_l143_143529


namespace lucas_seventh_score_l143_143556

open Set

def average_is_integer (scores : List ℕ) : Prop :=
  (scores.sum / scores.length : ℚ).den = 1

theorem lucas_seventh_score (S : Fin 8 → ℕ)
  (hS_range : ∀ i, S i ∈ (range 13).map (+ 88))
  (hS_distinct : ∀ i j, S i = S j → i = j)
  (h_avg_integer : ∀ (n : Fin 8), average_is_integer (S 0 ..n))
  (h_eighth_test : S 7 = 94) :
  S 6 = 100 := sorry

end lucas_seventh_score_l143_143556


namespace cos_half_alpha_l143_143855

theorem cos_half_alpha (α : ℝ) (h1 : sin α = (4 / 9) * sqrt 2) (h2 : π / 2 < α ∧ α < π) : 
  cos (α / 2) = 1 / 3 := 
by 
  sorry

end cos_half_alpha_l143_143855


namespace inequality_proof_l143_143558

theorem inequality_proof
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_pos : ∀ k, 0 < x k)
  (x_succ : ℝ := ∑ i, x i) :
  (∑ k : Fin n, Real.sqrt (x k * (x_succ - x k)))
  ≤ Real.sqrt (∑ k : Fin n, x_succ * (x_succ - x k)) :=
by
  sorry

end inequality_proof_l143_143558


namespace sin_omega_decreasing_iff_omega_bound_l143_143080

theorem sin_omega_decreasing_iff_omega_bound (ω : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Ioo (-π/2) (π/2) → x2 ∈ Ioo (-π/2) (π/2) → x1 < x2 → sin (ω * x1) > sin (ω * x2)) ↔ (-1 ≤ ω ∧ ω < 0) :=
sorry

end sin_omega_decreasing_iff_omega_bound_l143_143080


namespace factor_example_solve_equation_example_l143_143322

-- Factorization proof problem
theorem factor_example (m a b : ℝ) : 
  (m * a ^ 2 - 4 * m * b ^ 2) = m * (a + 2 * b) * (a - 2 * b) :=
sorry

-- Solving the equation proof problem
theorem solve_equation_example (x : ℝ) (hx1: x ≠ 2) (hx2: x ≠ 0) : 
  (1 / (x - 2) = 3 / x) ↔ x = 3 :=
sorry

end factor_example_solve_equation_example_l143_143322


namespace line_parallel_x_axis_l143_143996

noncomputable def f (x α : ℝ) : ℝ :=
  cos x ^ 2 + cos (x + α) ^ 2 - 2 * cos α * cos x * cos (x + α)

theorem line_parallel_x_axis (α : ℝ) (h : ∀ k : ℤ, α ≠ k * Real.pi) : 
  ∀ x1 x2 : ℝ, f x1 α = f x2 α :=
begin
  sorry
end

end line_parallel_x_axis_l143_143996


namespace speed_of_faster_train_l143_143267

theorem speed_of_faster_train
  (length_each_train : ℕ)
  (length_in_meters : length_each_train = 50)
  (speed_slower_train_kmh : ℝ)
  (speed_slower : speed_slower_train_kmh = 36)
  (pass_time_seconds : ℕ)
  (pass_time : pass_time_seconds = 36) :
  ∃ speed_faster_train_kmh, speed_faster_train_kmh = 46 :=
by
  sorry

end speed_of_faster_train_l143_143267


namespace gcd_79625_51575_l143_143415

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 :=
by
  sorry

end gcd_79625_51575_l143_143415


namespace Doug_lost_marbles_l143_143403

theorem Doug_lost_marbles (D E L : ℕ) 
    (h1 : E = D + 22) 
    (h2 : E = D - L + 30) 
    : L = 8 := by
  sorry

end Doug_lost_marbles_l143_143403


namespace locus_of_intersections_is_radical_axis_l143_143932

theorem locus_of_intersections_is_radical_axis
  (circle1 circle2 : Circle)
  (A B : Point)
  (H_tangent1 : Tangent circle1 A)
  (H_tangent2 : Tangent circle2 B)
  (X Y : Point)
  (H_touch1 : TouchesAt circle1 circle2 X)
  (H_touch2 : TouchesAt circle1 circle2 Y)
  (H_same_way : SameWayTouch X Y) :
  is_radical_axis (locus_of_intersections (fun P => ∃ (AX := line_through A X) (BY := line_through B Y), intersect AX BY P)) :=
by sorry

end locus_of_intersections_is_radical_axis_l143_143932


namespace sum_of_three_digit_numbers_l143_143418

theorem sum_of_three_digit_numbers : 
  let digits := {0, 1, 2, 3}
  ∃ S : Finset ℕ, (∀ n ∈ S, 
    let d1 := n / 100
    let d2 := (n / 10) % 10
    let d3 := n % 10
    n / 1000 = 0 ∧ d1 ∈ {1, 2, 3} ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ d3 ≠ 0 ∧ 
  S.sum id = 2544) := 
by
  sorry

end sum_of_three_digit_numbers_l143_143418


namespace find_a_l143_143473

-- Define the sets A and B and their union
variables (a : ℕ)
def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a^2}
def C : Set ℕ := {0, 1, 2, 3, 9}

-- Define the condition and prove that it implies a = 3
theorem find_a (h : A a ∪ B a = C) : a = 3 := 
by
  sorry

end find_a_l143_143473


namespace find_k_at_intersection_l143_143844

theorem find_k_at_intersection : 
  ∃ k : ℝ, ∀ x y : ℝ, (3 * x + y = k) ∧ (-0.75 * x + y = 25) ∧ (x = -6.3) → k = 1.375 :=
by
  use 1.375
  intros x y h
  cases h with h1 hxy
  cases hxy with h2 h3
  sorry

end find_k_at_intersection_l143_143844


namespace part1_part2_l143_143387

open Complex

theorem part1 : (1 : ℂ) * (1 + 2 * I)^2 = -3 + 4 * I := 
sorry

theorem part2 : 
    ( ( (1 + I) / (1 - I) )^6 + ( (sqrt 2 + sqrt 3 * I) / (sqrt 3 - sqrt 2 * I) ) ) = 
    -1 + (sqrt 6) / 5 + ((sqrt 3) + (sqrt 2)) / 5 * I :=
sorry

end part1_part2_l143_143387


namespace sqrt_expression_l143_143406

theorem sqrt_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2 * x + 1)) :=
by
  sorry

end sqrt_expression_l143_143406


namespace incorrect_tan_sign_l143_143399

theorem incorrect_tan_sign : ¬ (tan (Real.pi * 170 / 180) > 0) :=
by
  have angle170_quad2: 90 < 170 ∧ 170 < 180 := sorry
  have tan170_negative: tan (Real.pi * 170 / 180) < 0 := sorry
  sorry

end incorrect_tan_sign_l143_143399


namespace sum_c_n_lt_six_l143_143435

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℝ := a_n n / b_n n

def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, c_n (i + 1)

theorem sum_c_n_lt_six (n : ℕ) : T_n n < 6 := by
  sorry

end sum_c_n_lt_six_l143_143435


namespace train_crosses_platform_in_time_l143_143796

theorem train_crosses_platform_in_time :
  (train_length platform_length : ℝ) (train_speed_kmh conversion_factor_kmh_to_ms : ℝ) (time_taken : ℝ)
  (h1 : train_length = 200)
  (h2 : platform_length = 175.03)
  (h3 : train_speed_kmh = 54)
  (h4 : conversion_factor_kmh_to_ms = 1000 / 3600)
  (total_distance : ℝ)
  (h5 : total_distance = train_length + platform_length)
  (train_speed_ms : ℝ)
  (h6 : train_speed_ms = train_speed_kmh * conversion_factor_kmh_to_ms)
  (h7 : time_taken = total_distance / train_speed_ms) :
  time_taken = 25.002 :=
sorry

end train_crosses_platform_in_time_l143_143796


namespace eccentricity_of_ellipse_l143_143157

/-- Given an ellipse with equation x^2/a^2 + y^2/b^2 = 1 (a > b > 0),
    with left vertex A and right focus F. A line passing through A 
    with a slope of 30 degrees intersects the ellipse at B. If BF 
    is perpendicular to AF, then the eccentricity of the ellipse is (3 - √3) / 3. -/
theorem eccentricity_of_ellipse
  (a b c : ℝ) (h : a > b) (hb : b > 0)
  (h1 : ∀ x : ℝ, (x/a)^2 + (x/b)^2 = 1)
  (A : ℝ → ℝ) (FA : ℝ → ℝ)
  (B : ℝ → ℝ) (BF : ℝ → ℝ)
  (F : ℝ → ℝ)
  (h2 : ∀ p : ℝ, B(p) = p → F(p+c) = p) 
  (h3 : c = a * e)
  (h4 : e = (3 - real.sqrt 3)/3) :
  e = (3 - real.sqrt 3)/3 :=
sorry

end eccentricity_of_ellipse_l143_143157


namespace trapezoid_area_l143_143547

theorem trapezoid_area (AD BC: ℝ) (triangle_ABD: ℝ) (alpha beta: ℝ)
    (hAD_parallel_BC: AD ∥ BC)
    (hMidsegment: EF = (AD + BC) / 2)
    (hAreaRatio: (area AEFD / area EBCF) = (real.sqrt(3) + 1) / (3 - real.sqrt(3)))
    (hArea_ABD: area (triangle_ABD) = real.sqrt(3)) :
    area (trapezoid_ABCD) = 2 :=
  sorry

end trapezoid_area_l143_143547


namespace emma_age_l143_143195

variables (O N L E : ℕ)

def oliver_eq : Prop := O = N - 5
def nancy_eq : Prop := N = L + 6
def emma_eq : Prop := E = L + 4
def oliver_age : Prop := O = 16

theorem emma_age :
  oliver_eq O N ∧ nancy_eq N L ∧ emma_eq E L ∧ oliver_age O → E = 19 :=
by
  sorry

end emma_age_l143_143195


namespace eighth_perfect_square_l143_143248

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := nat.floor (a n + real.sqrt (a n))

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem eighth_perfect_square :
  ∃ n, is_perfect_square (a n) ∧ ∃ k : ℕ, k = 64 ∧ ∀ m < n, ¬ is_perfect_square (a m) :=
sorry

end eighth_perfect_square_l143_143248


namespace perfect_cubes_count_l143_143481

theorem perfect_cubes_count : 
  let lower_bound := 2^8 + 1 in
  let upper_bound := 2^10 + 1 in
  (∃ c : ℕ, c^3 ≥ lower_bound ∧ c^3 ≤ upper_bound) ∧ 
  (finset.Icc 7 10).card = 4 :=
by
  sorry

end perfect_cubes_count_l143_143481


namespace find_x_l143_143868

theorem find_x (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p1 * p2 * p3 = 11 * (7^n + 1) ∧ p1.prime ∧ p2.prime ∧ p3.prime): 
  7^n + 1 = 16808 :=
begin
  sorry
end

end find_x_l143_143868


namespace sum_divisors_24_l143_143275

theorem sum_divisors_24 :
  (∑ n in Finset.filter (λ n => 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_divisors_24_l143_143275


namespace shift_to_obtain_g_from_f_l143_143464

noncomputable theory
open Real

def f (x : ℝ) := sin (2 * x + π / 4)

def g (x : ℝ) := cos (2 * x)

theorem shift_to_obtain_g_from_f :
  ∀ x : ℝ, f (x - π / 4) = g x :=
by
  intro x
  simp [f, g, sin_add, cos_eq_sin_add_pi_div_two]
  ring
  simp [cos_pi_div_two]
  sorry

end shift_to_obtain_g_from_f_l143_143464


namespace rationalize_denominator_l143_143684

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143684


namespace rajesh_test_scores_l143_143208

theorem rajesh_test_scores :
  ∃ (scores : List ℕ), 
    (scores = [95, 92, 89, 84, 82, 78, 75]) ∧ 
    (∀ x ∈ scores, x ≤ 95) ∧ 
    List.Sum scores = 595 ∧ 
    (∃ firstFour lastThree, 
      firstFour = [75, 84, 92, 78] ∧
      lastThree = [95, 89, 82] ∧
      List.Sum firstFour = 329 ∧
      List.Sum lastThree = 266 ∧
      ∀ (x y : ℕ), x ∈ firstFour → y ∈ lastThree → x ≠ y) :=
by
  sorry

end rajesh_test_scores_l143_143208


namespace rationalize_sqrt_fraction_l143_143666

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143666


namespace solve_for_y_identity_l143_143714

theorem solve_for_y_identity : 
  ∀ y : ℚ, 7 * (4 * y + 3) - 3 = -3 * (2 - 5 * y) → y = -24 / 13 :=
by
  assume y : ℚ,
  assume h : 7 * (4 * y + 3) - 3 = -3 * (2 - 5 * y),
  sorry

end solve_for_y_identity_l143_143714


namespace proposition_D_l143_143304

theorem proposition_D (a b : ℝ) (h : a > b > 0) : a^2 > b^2 := 
by 
  sorry

end proposition_D_l143_143304


namespace ice_cream_flavors_l143_143486

theorem ice_cream_flavors (n k : ℕ) (h_n : n = 4) (h_k : k = 4) :
  (nat.choose (n + k - 1) (k - 1)) = 35 :=
by {
  rw [h_n, h_k],
  norm_num,
  sorry
}

end ice_cream_flavors_l143_143486


namespace no_hobbits_present_l143_143592

theorem no_hobbits_present
  (n : ℕ)                      -- number of participants
  (h₀ : n > 20)                -- more than 20 participants present
  (humans elves : ℕ → ℕ)       -- functions returning the number of humans and elves in subsets.
  (h₁ : ∀ (s : finset ℕ), s.card = 15 → humans s ≥ 4)    -- at least 4 humans in any group of 15 participants
  (h₂ : ∀ (s : finset ℕ), s.card = 15 → elves s ≥ 5)     -- at least 5 elves in any group of 15 participants
  : ∀ (x : ℕ), x ∈ (range n) → ¬ (x = n+1) :=            -- n+1 refers to the hobbit  
begin
  sorry,
end

end no_hobbits_present_l143_143592


namespace find_m_find_t_range_l143_143079

noncomputable def f (x m : ℝ) : ℝ := |x - m| - 3

-- Statement (I): Prove that m = 1 given the solution set of f(x) ≥ 0 is (-∞, -2] ∪ [4, +∞).
theorem find_m 
  (sol_set : Set ℝ := {x | f x m ≥ 0}) 
  (H : sol_set = (Iio (-2) ∪ Ici 4)) :
  m = 1 :=
  sorry

-- Statement (II): Prove that t ≤ -2 given that there exists x ∈ ℝ such that f(x) ≥ t + |2 - x|.
theorem find_t_range 
  (t : ℝ)
  (H : ∃ x : ℝ, f x 1 ≥ t + |2 - x|) :
  t ≤ -2 :=
  sorry

end find_m_find_t_range_l143_143079


namespace range_of_k_intersecting_AB_l143_143441

-- Definitions of points and the line equation
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (-2, -1)
def l (k : ℝ) (x : ℝ) : ℝ := k * (x - 2) + 1

-- Proof statement asserting the range of k
theorem range_of_k_intersecting_AB :
  ∃ k : ℝ, -2 ≤ k ∧ k ≤ 0.5 ∧ ∀ x, l k x ∈ closed_segment ℝ (1,3) (-2,-1) :=
sorry

end range_of_k_intersecting_AB_l143_143441


namespace mayar_distance_when_meeting_l143_143188

theorem mayar_distance_when_meeting (initial_distance : ℕ) (rosie_speed mayar_speed meet_distance_rosie meet_distance_mayar : ℕ) 
  (h1 : initial_distance = 90) 
  (h2 : mayar_speed = 2 * rosie_speed) 
  (h3 : meet_distance_rosie + meet_distance_mayar = initial_distance) 
  (h4 : meet_distance_mayar = 2 * meet_distance_rosie) : 
  meet_distance_mayar = 60 := 
by 
  rw [h4, ← add_assoc] at h3
  rw [←two_mul, mul_comm 2, mul_assoc] at h3
  linarith

end mayar_distance_when_meeting_l143_143188


namespace area_of_R_is_4sqrt3_l143_143389

-- Definitions for the problem conditions
def square_ABCD (A B C D : Point) (side_length : ℝ) : Prop :=
  side_length = 4 ∧ 
  dist A B = side_length ∧ 
  dist B C = side_length ∧ 
  dist C D = side_length ∧ 
  dist D A = side_length ∧ 
  ∠A B C = 90 ∧ 
  ∠B C D = 90 ∧ 
  ∠C D A = 90 ∧ 
  ∠D A B = 90

def secondary_square_BEFG (B E F G : Point) : Prop :=
  ∠E B F = 120 ∧ 
  dist B E = dist E F ∧ 
  dist E F = dist F G ∧ 
  dist F G = dist G B

def region_R (R : Set Point) (A B C D : Point) : Prop :=
  ∀ P ∈ R, dist P B < min (dist P A) (min (dist P C) (dist P D))

-- The main proof statement, testing the equivalence
theorem area_of_R_is_4sqrt3
  (A B C D E F G : Point)
  (R : Set Point) 
  (h_ABCD : square_ABCD A B C D 4) 
  (h_BEFG : secondary_square_BEFG B E F G) 
  (h_R : region_R R A B C D) : 
  area R = 4 * sqrt 3 := 
sorry

end area_of_R_is_4sqrt3_l143_143389


namespace valid_orderings_of_colored_houses_num_valid_orderings_is_2_l143_143270

noncomputable def num_valid_orderings (houses : List String) : ℕ :=
  if houses = ["G", "B", "Y", "O", "R"] ∨ houses = ["R", "G", "B", "Y", "O"] then 2 else 0

theorem valid_orderings_of_colored_houses : num_valid_orderings ["G", "B", "Y", "O", "R"] = 2 :=
by {
  -- Conditions:
  -- 1. Orange house (O) before the Red house (R)
  -- 2. Blue house (B) before Yellow house (Y) but after Green house (G)
  -- 3. Green house (G) is not next to Blue house (B)
  -- 4. Yellow house (Y) is not next to Orange house (O)

  -- Valid combinations are ["G", "B", "Y", "O", "R"] and ["R", "G", "B", "Y", "O"]
  sorry
}

theorem num_valid_orderings_is_2 : ∃ houses : List String, num_valid_orderings houses = 2 :=
by {
  -- The correct answer is 2
  use ["G", "B", "Y", "O", "R"],
  exact valid_orderings_of_colored_houses
}

end valid_orderings_of_colored_houses_num_valid_orderings_is_2_l143_143270


namespace sum_of_divisors_of_24_l143_143280

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range (24 + 1))), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l143_143280


namespace angle_QSR_eq_70_l143_143543

-- Define the given conditions
variable (P Q R S T : Type) [geometry : EuclideanGeometry P Q R S T]
open EuclideanGeometry

def angle_PQS : Angle PQS := 120 
def angle_QRS : Angle QRS := 50 
def angle_QST : Angle QST := 30

-- Prove that the measure of angle QSR is 70 degrees
theorem angle_QSR_eq_70 :
  (angle QSR).measure = 70 :=
sorry

end angle_QSR_eq_70_l143_143543


namespace apples_remaining_l143_143585

theorem apples_remaining (total_apples : ℕ) 
  (first_day_ratio : ℚ) 
  (second_day_multiple : ℕ) 
  (third_day_extra : ℕ) 
  (first_day_picked second_day_picked third_day_picked total_picked remaining_apples : ℕ) :
  total_apples = 200 →
  first_day_ratio = 1/5 →
  second_day_multiple = 2 →
  third_day_extra = 20 →
  first_day_picked = total_apples * first_day_ratio.to_nat →
  second_day_picked = first_day_picked * second_day_multiple →
  third_day_picked = first_day_picked + third_day_extra →
  total_picked = first_day_picked + second_day_picked + third_day_picked →
  remaining_apples = total_apples - total_picked →
  remaining_apples = 20 :=
by
  sorry

end apples_remaining_l143_143585


namespace problem_l143_143429

theorem problem (x y : ℝ) (h1 : 2^x = 3) (h2 : log 2 5 = y) : x + y = log 2 15 := 
sorry

end problem_l143_143429


namespace smallest_b_no_lattice_points_line_l143_143391

theorem smallest_b_no_lattice_points_line :
  ∃ (b : ℚ), (b = 52 / 151) ∧ 
    (∀ (m : ℚ), (1 / 3 < m) → (m < b) → 
      ∀ (x : ℤ), (0 < x ∧ x ≤ 150 → ∀ (y : ℤ), y ≠ m * x + 3)) :=
by
  sorry

end smallest_b_no_lattice_points_line_l143_143391


namespace value_of_expression_l143_143301

theorem value_of_expression : 3 + (-3 : ℚ) ^ -3 = 80 / 27 := 
  sorry

end value_of_expression_l143_143301


namespace rationalize_denominator_l143_143682

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143682


namespace find_cows_l143_143124

-- Define the number of ducks (D) and cows (C)
variables (D C : ℕ)

-- Define the main condition given in the problem
def legs_eq_condition (D C : ℕ) : Prop :=
  2 * D + 4 * C = 2 * (D + C) + 36

-- State the theorem we wish to prove
theorem find_cows (D C : ℕ) (h : legs_eq_condition D C) : C = 18 :=
sorry

end find_cows_l143_143124


namespace find_q_in_terms_of_p_and_r_l143_143746

variables (LM LX XN p r q NM LN: Real)
variables (Jenny Vicky: Point → Point → ℝ → Bool)

def right_angled_triangle_LMN : Prop := true

def LM_is_r : Prop := LM = r
def LX_is_p : Prop := LX = p
def XN_is_q : Prop := XN = q

def Jenny_walks := Jenny X M p
def Vicky_walks := Vicky X N q

def Jenny_Vicky_same_speed := ∀ t: ℝ, Jenny t = Vicky t

theorem find_q_in_terms_of_p_and_r :
  (right_angled_triangle_LMN ) →
  (LM_is_r ) →
  (LX_is_p ) →
  (XN_is_q ) →
  (Jenny_walks ) →
  (Vicky_walks) →
  (Jenny_Vicky_same_speed ) →
  q = (p * r) / (2 * p + r)
:= by
  sorry

end find_q_in_terms_of_p_and_r_l143_143746


namespace int_pair_divides_pow_l143_143839

theorem int_pair_divides_pow (a n : ℤ) : 
  n ∣ ((a + 1)^n - a^n) ↔ n = 1 := 
begin
  sorry
end

end int_pair_divides_pow_l143_143839


namespace ratio_area_ABD_to_ABC_l143_143128

variable {T : ℝ}
variable {A B C D : Type*} [triangle ABC] [point D]

/- Given conditions:
1. Triangle ABC has area T.
2. Point D is the midpoint of side BC.
3. Triangle ABD is formed by connecting point A to D.
-/

theorem ratio_area_ABD_to_ABC (hT : area ABC = T) 
    (hD_mid : is_midpoint D B C) :
    area ABD = T / 2 :=
by
  sorry

end ratio_area_ABD_to_ABC_l143_143128


namespace largest_p_n2_largest_p_n5_largest_p_general_l143_143841

theorem largest_p_n2 (x1 x2 : ℝ): 
  x1^2 + x2^2 ≥ 2 * x1 * x2 := 
  sorry

theorem largest_p_n5 (x1 x2 x3 x4 x5 : ℝ) :
  x1^2 + x2^2 + x3^2 + x4^2 + x5^2 ≥ (2 / Real.sqrt 3) * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5) := 
  sorry

theorem largest_p_general (n : ℕ) (h : 2 ≤ n) (x : Fin n → ℝ) :
  (∑ i, x i ^ 2) ≥ 2 * (∑ i in Finset.range (n - 1), x i * x (i + 1)) := 
  sorry

end largest_p_n2_largest_p_n5_largest_p_general_l143_143841


namespace num_two_digit_integers_l143_143456

theorem num_two_digit_integers :
  let digits := {1, 3, 5, 8, 9}
  ∃ (count : Nat), count = 20 ∧ count = (digits.card × (digits.card - 1)) := 
by
  let digits := {1, 3, 5, 8, 9}
  let count := digits.card * (digits.card - 1)
  exists count
  have : count = 20,
  sorry
  exact ⟨this, rfl⟩

end num_two_digit_integers_l143_143456


namespace find_x_l143_143871

theorem find_x (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p1 * p2 * p3 = 11 * (7^n + 1) ∧ p1.prime ∧ p2.prime ∧ p3.prime): 
  7^n + 1 = 16808 :=
begin
  sorry
end

end find_x_l143_143871


namespace angle_equality_l143_143546

theorem angle_equality
  (A B C D P E: Type)
  [geometry E]
  (hP: on_point E P)
  (hED: on_line P (line_through E D))
  (hPCB_ACD : ∠(line_through P C) (line_through C B) = ∠(line_through A C) (line_through C D))
  (circABD: is_circumcircle (triangle A B D))
  (hE: intersects_on (circABD) (line_through A C) = E) :
  ∠(line_through A E) (line_through E D) = ∠(line_through P E) (line_through E B) :=
begin
  sorry
end

end angle_equality_l143_143546


namespace lila_will_have_21_tulips_l143_143245

def tulip_orchid_ratio := 3 / 4

def initial_orchids := 16

def added_orchids := 12

def total_orchids : ℕ := initial_orchids + added_orchids

def groups_of_orchids : ℕ := total_orchids / 4

def total_tulips : ℕ := 3 * groups_of_orchids

theorem lila_will_have_21_tulips :
  total_tulips = 21 := by
  sorry

end lila_will_have_21_tulips_l143_143245


namespace avg_sequence_eq_cot_one_l143_143205

theorem avg_sequence_eq_cot_one :
  (∑ n in (finset.filter (λ n : ℕ, n % 2 = 0) (finset.range 181)), n * real.sin (real.pi * n / 180)) / 90 = real.cot (real.pi / 180) :=
sorry

end avg_sequence_eq_cot_one_l143_143205


namespace rolls_for_mode_of_two_l143_143093

theorem rolls_for_mode_of_two (n : ℕ) (p : ℚ := 1/6) (m0 : ℕ := 32) : 
  (n : ℚ) * p - (1 - p) ≤ m0 ∧ m0 ≤ (n : ℚ) * p + p ↔ 191 ≤ n ∧ n ≤ 197 := 
by
  sorry

end rolls_for_mode_of_two_l143_143093


namespace rationalize_sqrt_fraction_l143_143621

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143621


namespace school_fee_is_correct_l143_143183

def denomination_value (count : Nat) (value : Float) : Float := count * value

noncomputable def total_given_by_mother : Float :=
  denomination_value 2 100 + denomination_value 1 50 + denomination_value 5 20 + 
  denomination_value 3 10 + denomination_value 4 5 + denomination_value 6 0.25 + 
  denomination_value 10 0.10 + denomination_value 5 0.05

noncomputable def total_given_by_father : Float :=
  denomination_value 3 100 + denomination_value 4 50 + denomination_value 2 20 + 
  denomination_value 1 10 + denomination_value 6 5 + denomination_value 8 0.25 + 
  denomination_value 7 0.10 + denomination_value 3 0.05

noncomputable def total_school_fee : Float :=
  total_given_by_mother + total_given_by_father

theorem school_fee_is_correct : total_school_fee = 985.60 := by
  sorry

end school_fee_is_correct_l143_143183


namespace option_d_is_negative_integer_option_d_is_integer_negative_integer_condition_l143_143807

theorem option_d_is_negative_integer (x : Int) (hx : x = -2) : x < 0 := by
  sorry

-- Adding condition that negative integers are less than zero.
theorem option_d_is_integer : Int := -2

theorem negative_integer_condition (x : Int) (hx : x < 0) : x ∈ Int := by
  sorry

end option_d_is_negative_integer_option_d_is_integer_negative_integer_condition_l143_143807


namespace Amanda_car_round_trip_time_l143_143800

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end Amanda_car_round_trip_time_l143_143800


namespace sum_bin_to_dec_l143_143249

def bin_to_dec (n : ℕ) : ℕ :=
  nat.binaryRecOn n 0 (λ b d f, d + (bit b 0).toNat * (f * 2) + bit b 0).toNat

-- Define the binary numbers
def bin_101 : ℕ := 0b101
def bin_110 : ℕ := 0b110

-- Define the corresponding decimal numbers through conversion
def dec_101 : ℕ := bin_to_dec bin_101
def dec_110 : ℕ := bin_to_dec bin_110

-- Prove that the sum of their decimal equivalents is correct
theorem sum_bin_to_dec :
  dec_101 + dec_110 = 11 :=
by sorry

end sum_bin_to_dec_l143_143249


namespace fractionSpentAtArcade_l143_143555

/-- John's weekly allowance in dollars. -/
def weeklyAllowance : ℝ := 1.50

/-- Fraction of the allowance John spent at the arcade. -/
variable (f : ℝ)

/-- Remaining allowance after spending at the arcade. -/
def remainingAfterArcade := (1 - f) * weeklyAllowance

/-- John's spending at the toy store which is 1/3 of the remaining allowance. -/
def spentAtToyStore := (1 / 3) * remainingAfterArcade f

/-- Remaining allowance after spending at the toy store. -/
def remainingAfterToyStore := remainingAfterArcade f - spentAtToyStore f

/-- Final constraint that John spent his last $0.40 at the candy store. -/
def finalSpendingConstraint := remainingAfterToyStore f = 0.40

/-- John spent 3/5 of his allowance at the arcade. -/
theorem fractionSpentAtArcade : f = 3 / 5 :=
by
  -- proof goes here
  sorry

end fractionSpentAtArcade_l143_143555


namespace inequality_solution_l143_143009

noncomputable def solution_set : Set ℝ := {x | 0.5 ≤ x ∧ x ≤ 1}

theorem inequality_solution (x : ℝ) : 
  (solution_set x) ↔ (frac x (x - 1) + frac (x + 1) (2 * x) ≥ 5 / 2) := by
  sorry

end inequality_solution_l143_143009


namespace compute_result_l143_143386

theorem compute_result : (300000 * 200000) / 100000 = 600000 := by
  sorry

end compute_result_l143_143386


namespace max_pairwise_sum_l143_143761

theorem max_pairwise_sum :
  ∃ (a b c d x y : ℕ), 
    (x = a + b ∨ x = a + c ∨ x = a + d ∨ x = b + c ∨ x = b + d ∨ x = c + d) ∧
    (y = a + b ∨ y = a + c ∨ y = a + d ∨ y = b + c ∨ y = b + d ∨ y = c + d) ∧
    x ≠ y ∧
    ({a + b, a + c, a + d, b + c, b + d, c + d} ⊆ {189, 320, 287, 264, x, y}) ∧
    (x + y ≤ 761) :=
sorry

end max_pairwise_sum_l143_143761


namespace cube_root_series_l143_143817

theorem cube_root_series :
  ( ( (8 : ℝ) * ∑ k in (Finset.range 1000).map (Finset.natEmb) , (k : ℝ)^3) 
    / (27 * ∑ k in (Finset.range 1000).map (Finset.natEmb), (k : ℝ)^3)) ^ (1/3: ℝ) = 2 / 3 := 
by {
  sorry
}

end cube_root_series_l143_143817


namespace find_treasure_in_box_2_l143_143979

def box_number := {n : ℕ // 1 ≤ n ∧ n ≤ 5}

def is_cedar (n : box_number) : Prop :=
  n = 1 ∨ n = 4 ∨ n = 5

def is_sandalwood (n : box_number) : Prop :=
  n = 2 ∨ n = 3

def statement (n : box_number) : Prop :=
  match n with
  | 1 => (treasure 1 ∨ treasure 4)
  | 2 => treasure 1
  | 3 => (treasure 3 ∨ treasure 5)
  | 4 => ¬(treasure 1 ∨ treasure 2 ∨ treasure 3 ∨ treasure 4)
  | 5 => ∀ j, j ≠ 5 → ¬ statement j

def false_count_equals : Prop :=
  ∑ i in {1, 4, 5}.to_finset, (if statement i then 0 else 1) =
  ∑ i in {2, 3}.to_finset, (if statement i then 0 else 1)

def treasure (n : box_number) : Prop

theorem find_treasure_in_box_2 : treasure 2 :=
by {
  -- creating the necessary structures
  sorry
}

end find_treasure_in_box_2_l143_143979


namespace melanie_trip_distance_l143_143190

theorem melanie_trip_distance (x : ℚ) 
  (h1 : x / 4 + 36 + x / 3 = x) : x = 432/5 := 
begin
  sorry
end

end melanie_trip_distance_l143_143190


namespace problem_math_I_problem_math_II_l143_143466

noncomputable def f (x a : ℝ) : ℝ := exp x * (x^2 - a * x + a)
noncomputable def p (x a : ℝ) : ℝ := f x a - x^2

theorem problem_math_I (a : ℝ) : 
  (∀ x ∈ Icc (1:ℝ) 2, deriv (f x) a ≥ 0) ↔ a < 4 := 
by 
  sorry

theorem problem_math_II (a : ℝ) : 
  (∀ x, deriv (p x) a ≥ 0 → deriv (p x) a = 0 → x = 0) ↔ a < 0 := 
by 
  sorry

end problem_math_I_problem_math_II_l143_143466


namespace x_y_result_l143_143902

noncomputable def x_y_value (x y : ℝ) : ℝ := x + y

theorem x_y_result (x y : ℝ) 
  (h1 : x + Real.cos y = 3009) 
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x_y_value x y = 3009 + Real.pi / 2 :=
by
  sorry

end x_y_result_l143_143902


namespace probabilities_equal_l143_143503

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l143_143503


namespace total_subjects_l143_143192

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end total_subjects_l143_143192


namespace hyperbola_eccentricity_l143_143893

-- Define the conditions
variables {F1 F2 P : Type*}
variables (h1 : ∃ (F1 F2 : P), ∃ (P : P), ∠(F1, P, F2) = 60)
variables (h2 : ∀ (P : P), |P − F1| = 3 * |P − F2|)

-- Define the goal
theorem hyperbola_eccentricity : eccentricity = sqrt(7) / 2 :=
sorry

end hyperbola_eccentricity_l143_143893


namespace range_of_a_l143_143019

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
by sorry

end range_of_a_l143_143019


namespace maximize_profit_l143_143340

noncomputable def profit (a : ℝ) (x : ℝ) : ℝ :=
  16 - ((4 / (x + 1)) + x)

theorem maximize_profit (a : ℝ) (h_a_pos : 0 <= a) :
  (∃ x : ℝ, 0 <= x ∧ x <= a ∧ profit a x = 16 - ((4 / (x + 1 : ℝ)) + x)) ∧
  (if 1 <= a then x = 1 else x = a) :=
begin
  have h_profit : ∀ x, profit a x = 16 - ((4 / (x + 1 : ℝ)) + x),
  { intros x,
    rw profit },
  split,
  { use [x],
    split; sorry },
  split_ifs,
  { use [1],
    split; sorry },
  { use [a],
    split; sorry }
end

end maximize_profit_l143_143340


namespace solve_f_eq_1_l143_143463

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else |real.log x / real.log 2|

theorem solve_f_eq_1 :
  {x : ℝ | f x = 1} = {0, 1/2, 2} :=
by
  sorry

end solve_f_eq_1_l143_143463


namespace right_triangle_cos_l143_143548

open Real

theorem right_triangle_cos (ABC : Triangle) (C_right_angle : ∠ABC.C = π / 2) (BD : ℝ) (BD_value : BD = (17 ^ 3 : ℝ))
  (integer_side_lengths : ∀ s : LineSegment, s ∈ ABC.sides → ∃ (n : ℤ), (↑n : ℝ) = s.length)
  : ∃ m n : ℕ, m.gcd n = 1 ∧ acos (ABC.∠BC.angle_cos) = ↑17 / ↑145 ∧ m + n = 162 :=
by
  sorry

end right_triangle_cos_l143_143548


namespace rationalize_sqrt_fraction_l143_143624

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143624


namespace triangle_ratio_l143_143549

theorem triangle_ratio
  (P Q R M N L S : Point)
  (hM : M ∈ segment P Q)
  (hN : N ∈ segment P R)
  (hPM : dist P M = 2)
  (hMQ : dist M Q = 6)
  (hPN : dist P N = 3)
  (hNR : dist N R = 9)
  (hBisector : is_angle_bisector P S Q R)
  (hL : L ∈ segment M N)
  (hPL_on_PS : L ∈ line PS) :
  ∃ (k : ℚ), k = 1 / 4 ∧ dist P L / dist P S = k := sorry

end triangle_ratio_l143_143549


namespace percent_less_than_m_add_d1_l143_143524

variable (m d1 : ℝ)
variable (γ₁ : ℝ := 0.5)
variable (γ₂ : ℝ := 3)

-- Conditions
/-- A large population distribution is symmetric around the mean m. -/
axiom symmetric_distribution : ∀ x, P(m + x) = P(m - x)

/-- 36% of the distribution lies within one standard deviation d1 of the mean. -/
axiom thirty_six_percent_within_one_std : P(m - d1 ≤ X ∧ X ≤ m + d1) = 0.36

/-- 60% of the distribution lies within two standard deviations d2 of the mean. -/
axiom sixty_percent_within_two_std : P(m - d2 ≤ X ∧ X ≤ m + d2) = 0.60

/-- The skewness γ₁ is 0.5. -/
axiom skewness_value : γ₁ = 0.5

/-- The kurtosis γ₂ is 3. -/
axiom kurtosis_value : γ₂ = 3

-- Question to prove as a theorem
/-- The percentage of the distribution less than m + d1 is 68%. -/
theorem percent_less_than_m_add_d1 : P(X < m + d1) = 0.68 :=
by
  sorry

end percent_less_than_m_add_d1_l143_143524


namespace main_statement_l143_143581

def f (x : ℚ) : ℚ := sorry

theorem main_statement :
  (∀ (m n : ℚ), abs (f (m + n) - f m) ≤ m / n) →
  (∀ k : ℕ, 0 < k →
    (∑ i in finset.range k, abs (f (2^k) - f (2^i)) ≤ (k * (k-1)) / 2)) :=
by {
  sorry
}

end main_statement_l143_143581


namespace rationalize_sqrt_5_over_12_l143_143640

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143640


namespace solve_m_n_l143_143120

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end solve_m_n_l143_143120


namespace more_white_animals_than_cats_l143_143811

theorem more_white_animals_than_cats (C W WC : ℕ) 
  (h1 : WC = C / 3) 
  (h2 : WC = W / 6) : W = 2 * C :=
by {
  sorry
}

end more_white_animals_than_cats_l143_143811


namespace trajectory_equation_of_point_l143_143433

theorem trajectory_equation_of_point
  (x y a b : ℝ)
  (h₀ : a = (3 / 2) * x)
  (h₁ : b = 3 * y)
  (h₂ : x > 0)
  (h₃ : y > 0)
  (h₄ : λ O Q AB, (-x, y) • (- (3 / 2) * x, 3 * y) = 1) :
  (3 / 2) * x^2 + 3 * y^2 = 1 :=
by
  sorry

end trajectory_equation_of_point_l143_143433


namespace find_angle_GYD_l143_143542

theorem find_angle_GYD (AB CD : Line) (BXG : Angle) (BG : Line) (GYD : Angle) 
  (h1 : parallel AB CD) 
  (h2 : measure BXG = 135) 
  (h3 : on_same_side_of_transversal BXG GYD BG AB CD) :
  measure GYD = 45 := 
sorry

end find_angle_GYD_l143_143542


namespace relationship_between_y1_y2_y3_l143_143497

variable (y1 y2 y3 : ℚ)

def A (x : ℚ) := -4 / x

def B := y1 = A (-1)

def C := y2 = A 2

def D := y3 = A 3

theorem relationship_between_y1_y2_y3 (h1 : B) (h2 : C) (h3 : D) : y1 > y3 ∧ y3 > y2 :=
by {
  sorry
}

end relationship_between_y1_y2_y3_l143_143497


namespace possible_values_of_n_l143_143571

theorem possible_values_of_n (x1 x2 : ℤ) (n : ℤ) (h1 : x1 * x2 = 36) (h2 : x1 + x2 = n) :
  ∃ s, s = {37, 20, 15, 13, 12, -37, -20, -15, -13, -12} ∧ s.toFinset.card = 10 :=
by
  sorry

end possible_values_of_n_l143_143571


namespace z1_z2_in_quadrant_IV_l143_143914

def quadrant (z : ℂ) : String :=
if (Re z > 0) ∧ (Im z > 0) then "Quadrant I"
else if (Re z < 0) ∧ (Im z > 0) then "Quadrant II"
else if (Re z < 0) ∧ (Im z < 0) then "Quadrant III"
else if (Re z > 0) ∧ (Im z < 0) then "Quadrant IV"
else "On the axis"

theorem z1_z2_in_quadrant_IV :
  let z1 := (1 : ℂ) + (-3 * I)
  let z2 := (3 : ℂ) + (2 * I)
  quadrant (z1 + z2) = "Quadrant IV" :=
by
  let z1 := (1 : ℂ) + (-3 * I)
  let z2 := (3 : ℂ) + (2 * I)
  sorry

end z1_z2_in_quadrant_IV_l143_143914


namespace ellipse_problem_l143_143072

noncomputable def ellipse_equation (a b : ℝ) (h : a > b) : Prop :=
∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def prove_relation (A1 Q : ℝ) (MN : ℝ) : Prop :=
2 * |A1 - Q| = |MN|

variable (a b c : ℝ)
variable (h_a_b : a > b)
variable (h_eccentricity : (c / a) = (Real.sqrt 6 / 3))
variable (F1 : (ℝ × ℝ))
variable (F2 : (ℝ × ℝ))
variable (R : ℝ × ℝ := (2 * Real.sqrt 2, Real.sqrt 6))
variable (h_perp_bisector : F2 = ((-1, 0)) → F2 = ((1, 0)))

theorem ellipse_problem (h_ellipse_eqn : ellipse_equation a b h_a_b)
    (real_a : a = Real.sqrt 3)
    (real_b : b = 1)
    (result_eqn : (x y: ℝ) -> (x^2 / 3 + y^2 = 1))
    (h_A1_A2 : A1_x = -Real.sqrt 3 ∧ A2_x = Real.sqrt 3)
    (P : (ℝ × ℝ)) (h_P_line : P.1 = -2 * Real.sqrt 3 ∧ P.2 ≠ 0)
    (N M Q : (ℝ × ℝ))
    (h_P_A1_N : Line_through P A1 ∧ Line_through P A2)
    (h_midpoint : midpoint (N, M) = Q)
    (result_prove : 2 * |A1_x - Q_x)| = |MN|) :
    (result_eqn ∧ result_prove) :=
sorry

end ellipse_problem_l143_143072


namespace fixed_point_of_symmetric_line_l143_143104

/-- If line l1: y = k(x - 4) is symmetric to line l2 about the point (2,1), then line l2 always passes through a fixed point. -/
theorem fixed_point_of_symmetric_line
  (k : ℝ)
  (l1 : ∀ x : ℝ, ℝ) (l1_def : ∀ x, l1 x = k * (x - 4))
  (symm_point : Prod ℝ ℝ) (symm_point_def : symm_point = (2, 1))
  (l2 : ∀ x : ℝ, ℝ)
  (symmetric : ∀ x y, l1 x = y ↔ l2 (2 * 2 - x) = 2 * 1 - y) :
  ∃ x y, (x, y) = (0, 2) ∧ ∃ y, l2 y = y := 
begin
  sorry,
end

end fixed_point_of_symmetric_line_l143_143104


namespace sum_of_divisors_of_24_l143_143287

theorem sum_of_divisors_of_24 : ∑ d in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), d = 60 := by
  sorry

end sum_of_divisors_of_24_l143_143287


namespace sin_eq_log_solutions_l143_143016

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := Real.sin x
def f2 (x : ℝ) : ℝ := Real.log x

-- The main statement that there are exactly 3 solutions to sin x = log x for x > 0
theorem sin_eq_log_solutions :
  {x : ℝ | x > 0 ∧ f1 x = f2 x}.finite ∧ {x : ℝ | x > 0 ∧ f1 x = f2 x}.to_finset.card = 3 :=
by
  sorry

end sin_eq_log_solutions_l143_143016


namespace number_of_special_sets_l143_143385

theorem number_of_special_sets : 
  (∃ S : finset ℕ, (∀ x ∈ S, x < 16) ∧ (∀ x ∈ S, (2 * x % 16) ∈ S)) → 
  ∃ (n : ℕ), n = 678 := 
sorry

end number_of_special_sets_l143_143385


namespace find_a_l143_143076

noncomputable def f (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + Real.log x

theorem find_a (h_max : ∃ (x : Set.Icc (0 : ℝ) 1), f (-Real.exp 1) x = -1) : 
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≤ -1) → a = -Real.exp 1 :=
sorry

end find_a_l143_143076


namespace transmission_time_l143_143404

theorem transmission_time :
  let regular_blocks := 70
  let large_blocks := 30
  let chunks_per_regular_block := 800
  let chunks_per_large_block := 1600
  let channel_rate := 200
  let total_chunks := (regular_blocks * chunks_per_regular_block) + (large_blocks * chunks_per_large_block)
  let total_time_seconds := total_chunks / channel_rate
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 8.67 := 
by 
  sorry

end transmission_time_l143_143404


namespace first_three_good_power_numbers_not_good_power_number_50_smallest_good_power_number_gt_70_infinitely_many_good_power_numbers_l143_143471

def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then 1 else 2^(n/2)

def S (n : ℕ) : ℕ :=
(n+1).sum sequence

-- Definition of "good power number" of given sequence
def is_good_power_number (m : ℕ) : Prop :=
∃ p : ℕ, S m = 2^p

-- (I) Prove that the first 3 "good power numbers" are 1, 2, 3
theorem first_three_good_power_numbers :
  is_good_power_number 1 ∧ is_good_power_number 2 ∧ is_good_power_number 3 :=
sorry

-- (II) Prove that 50 is not a "good power number"
theorem not_good_power_number_50 :
  ¬ is_good_power_number 50 :=
sorry

-- (III)(i) Prove that the smallest "good power number" m that satisfies m > 70 is 95
theorem smallest_good_power_number_gt_70 :
  ∃ m : ℕ, m > 70 ∧ is_good_power_number m ∧ ∀ n, n > 70 → is_good_power_number n → m ≤ n :=
sorry

-- (III)(ii) Prove that there are infinitely many "good power numbers"
theorem infinitely_many_good_power_numbers :
  ∃ (f : ℕ → ℕ), ∀ n, is_good_power_number (f n) :=
sorry

end first_three_good_power_numbers_not_good_power_number_50_smallest_good_power_number_gt_70_infinitely_many_good_power_numbers_l143_143471


namespace tracy_initial_candies_l143_143263

theorem tracy_initial_candies (y : ℕ) 
  (condition1 : y - y / 4 = y * 3 / 4)
  (condition2 : y * 3 / 4 - (y * 3 / 4) / 3 = y / 2)
  (condition3 : y / 2 - 24 = y / 2 - 12 - 12)
  (condition4 : y / 2 - 24 - 4 = 2) : 
  y = 60 :=
by sorry

end tracy_initial_candies_l143_143263


namespace evaluate_expression_l143_143099

theorem evaluate_expression (x : ℕ) (h : x = 5) : 2 * x ^ 2 + 5 = 55 := by
  sorry

end evaluate_expression_l143_143099


namespace number_of_p_element_subsets_l143_143557

theorem number_of_p_element_subsets (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  let A := {x | x ∈ Finset.range (2 * p) ∧ 1 ≤ x + 1 ∧ x + 1 ≤ 2 * p} in
      (∑ B in (Finset.powersetLen p A), (if ((∑ x in B, x) % p = 0) then 1 else 0).toNat) = 
        2 + (Nat.choose (2 * p) p - 2) / p := 
  sorry

end number_of_p_element_subsets_l143_143557


namespace compare_real_numbers_l143_143372

theorem compare_real_numbers (a b : ℝ) : (a > b) ∨ (a = b) ∨ (a < b) :=
sorry

end compare_real_numbers_l143_143372


namespace a_is_positive_int_l143_143856

def a (k n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | n+1   => k * (a k n + 1) + (k + 1) * a k n + 2 * (nat.sqrt (k * (k + 1) * a k n * (a k n + 1)))

theorem a_is_positive_int (k : ℕ) (h : k > 0) (n : ℕ) (hn : n ≥ 1) : a k n > 0 :=
  by
  sorry

end a_is_positive_int_l143_143856


namespace find_real_root_l143_143376

noncomputable def lg (x : ℝ) : ℝ := Real.log10 x

def f (x : ℝ) : ℝ := 2 - x - lg x

theorem find_real_root : ∃ x0 : ℝ, f x0 = 0 ∧ abs (x0 - 1.755581) < 0.000001 :=
by
  sorry

end find_real_root_l143_143376


namespace find_x10_l143_143168

theorem find_x10 (x : ℕ → ℝ) :
  x 1 = 1 ∧ x 2 = 1 ∧ (∀ n ≥ 2, x (n + 1) = (x n * x (n - 1)) / (x n + x (n - 1))) →
  x 10 = 1 / 55 :=
by sorry

end find_x10_l143_143168


namespace solution_set_I_range_of_a_II_l143_143920

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2*x - 1|

theorem solution_set_I (x : ℝ) (a : ℝ) (h : a = 2) :
  f x a ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem range_of_a_II (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end solution_set_I_range_of_a_II_l143_143920


namespace sum_of_floor_sqrt_l143_143825

theorem sum_of_floor_sqrt : ∑ i in Finset.range 25, Int.floor (Real.sqrt (i + 1)) = 75 := by
  sorry

end sum_of_floor_sqrt_l143_143825


namespace express_nineteen_in_base_3_l143_143004

theorem express_nineteen_in_base_3 :
  nat.to_digits 3 19 = [2, 0, 1] :=
by
  sorry

end express_nineteen_in_base_3_l143_143004


namespace pure_imaginary_z_z_fourth_quadrant_z_fraction_l143_143070

-- Proof Problem for Question 1
theorem pure_imaginary_z (m : ℝ) (z : ℂ) (h : z = (m-1) * (m+2) + (m-1) * complex.I) :
  ((complex.re z = 0) ∧ (complex.im z ≠ 0)) → m = -2 :=
by sorry

-- Proof Problem for Question 2
theorem z_fourth_quadrant (m : ℝ) (z : ℂ) (h : z = (m-1) * (m+2) + (m-1) * complex.I) :
  ((complex.re z > 0) ∧ (complex.im z < 0)) → m < -2 :=
by sorry

-- Proof Problem for Question 3
theorem z_fraction (m : ℝ) (a b : ℝ) (z : ℂ) (h : z = (m-1) * (m+2) + (m-1) * complex.I) :
  m = 2 → (a + b = 8 / 5) :=
by sorry

end pure_imaginary_z_z_fourth_quadrant_z_fraction_l143_143070


namespace symmetry_center_of_f_max_min_values_of_f_l143_143462

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x)

theorem symmetry_center_of_f :
  ∃ k : ℤ, ∀ x, f(x) = 2 * Real.cos x * (Real.sin x - Real.cos x) →
  symmetry_center f = (k * π / 2 + π / 8, -1) :=
sorry

theorem max_min_values_of_f : 
  let a := π / 8
  let b := 3 * π / 4
  ∃ x : set.Icc a b, 
    (∀ (x ∈ set.Icc a b), f(a) ≤ f(x) ∧ f(x) ≤ f(b)) ∧
    (f(a) = √2 - 1 ∧ f(b) = -2 ∧ f(3 * π / 8) = -√2 - 1) :=
sorry

end symmetry_center_of_f_max_min_values_of_f_l143_143462


namespace angles_in_order_l143_143926

-- α1, α2, α3 are real numbers representing the angles of inclination of lines
variable (α1 α2 α3 : ℝ)

-- Conditions given in the problem
axiom tan_α1 : Real.tan α1 = 1
axiom tan_α2 : Real.tan α2 = -1
axiom tan_α3 : Real.tan α3 = -2

-- Theorem to prove
theorem angles_in_order : α1 < α3 ∧ α3 < α2 := 
by
  sorry

end angles_in_order_l143_143926


namespace divisor_product_to_number_l143_143180

theorem divisor_product_to_number (n : ℕ) (h : ∏ d in (nat.divisors n), d = 2^4 * 3^12) : n = 54 :=
sorry

end divisor_product_to_number_l143_143180


namespace greatest_divisor_form_p_plus_1_l143_143992

theorem greatest_divisor_form_p_plus_1 (n : ℕ) (hn : 0 < n):
  (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → 6 ∣ (p + 1)) ∧
  (∀ d : ℕ, (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → d ∣ (p + 1)) → d ≤ 6) :=
by {
  sorry
}

end greatest_divisor_form_p_plus_1_l143_143992


namespace sum_of_positive_divisors_of_24_l143_143291

theorem sum_of_positive_divisors_of_24 : ∑ n in {n : ℕ | n > 0 ∧ (n+24) % n = 0}, n = 60 := 
by sorry

end sum_of_positive_divisors_of_24_l143_143291


namespace area_of_triangle_XYZ_l143_143134

theorem area_of_triangle_XYZ (X Y Z : Type) [IsTriangle XYZ] (h_right_triangle : is_right_triangle XYZ)
    (h_isosceles : angle X = angle Z) (h_hypotenuse : XY = 8) : 
    area XYZ = 16 := 
by 
  sorry

end area_of_triangle_XYZ_l143_143134


namespace parrot_age_is_24_l143_143743

variable (cat_age : ℝ) (rabbit_age : ℝ) (dog_age : ℝ) (parrot_age : ℝ)

def ages (cat_age rabbit_age dog_age parrot_age : ℝ) : Prop :=
  cat_age = 8 ∧
  rabbit_age = cat_age / 2 ∧
  dog_age = rabbit_age * 3 ∧
  parrot_age = cat_age + rabbit_age + dog_age

theorem parrot_age_is_24 (cat_age rabbit_age dog_age parrot_age : ℝ) :
  ages cat_age rabbit_age dog_age parrot_age → parrot_age = 24 :=
by
  intro h
  sorry

end parrot_age_is_24_l143_143743


namespace population_increase_l143_143032

/-
Given the following conditions:
- From t=0 to t=1, the population increased by 5%.
- From t=1 to t=2, the population increased by 10%.
- From t=2 to t=3, the population increased by 15%.

Prove that the total percentage increase from t=0 to t=3 is 33.18%.
-/

theorem population_increase (t : ℝ) (P : ℝ) : 
  let scale_factor := (1 + 0.05) * (1 + 0.10) * (1 + 0.15),
      total_increase := scale_factor - 1 in
  (total_increase * 100) = 33.18 :=
by
  sorry

end population_increase_l143_143032


namespace range_of_S_n_l143_143165

noncomputable def f (x : ℝ) : ℝ := sorry

def a_n (n : ℕ) : ℝ := 
  if n = 0 then 0 
  else if n = 1 then 1 / 2 
  else f n

def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n (i + 1)

theorem range_of_S_n :
  (∀ x y : ℝ, f x * f y = f (x + y)) →
  (f 1 = 1 / 2) →
  ∀ n : ℕ, 
    1 ≤ n → 
    S_n n ∈ (set.Ico (1 / 2) 1) := sorry

end range_of_S_n_l143_143165


namespace range_of_a_l143_143036

def is_ellipse (a : ℝ) : Prop :=
  2 * a > 0 ∧ 3 * a - 6 > 0 ∧ 2 * a < 3 * a - 6

def discriminant_neg (a : ℝ) : Prop :=
  a^2 + 8 * a - 48 < 0

def p (a : ℝ) : Prop := is_ellipse a
def q (a : ℝ) : Prop := discriminant_neg a

theorem range_of_a (a : ℝ) : p a ∧ q a → 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l143_143036


namespace temperature_difference_l143_143594

theorem temperature_difference (H L : ℝ) (hH : H = 8) (hL : L = -2) :
  H - L = 10 :=
by
  rw [hH, hL]
  norm_num

end temperature_difference_l143_143594


namespace simplify_fraction_l143_143217

theorem simplify_fraction :
  (1 / (3 / (Real.sqrt 5 + 2) + 4 / (Real.sqrt 7 - 2))) = (3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10)) :=
sorry

end simplify_fraction_l143_143217


namespace rationalize_denominator_l143_143676

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143676


namespace trains_pass_each_other_in_6_seconds_l143_143316

noncomputable def time_to_pass_each_other (speed1 speed2 : ℕ) (length1 length2 : ℕ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5.0/18.0) in
  let total_length := length1 + length2 in
  total_length / relative_speed

theorem trains_pass_each_other_in_6_seconds :
  time_to_pass_each_other 80 70 150 100 = 6 :=
by
  sorry

end trains_pass_each_other_in_6_seconds_l143_143316


namespace simplify_expression_l143_143222

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : (3 * b - 3 - 5 * b) / 3 = - (2 / 3) * b - 1 :=
by
  sorry

end simplify_expression_l143_143222


namespace lawn_remaining_fraction_l143_143588

-- Define the conditions
def mary_mowing_rate : ℝ := 1 / 3
def tom_mowing_rate : ℝ := 1 / 6
def tom_mowing_time_alone : ℝ := 3
def mary_and_tom_mowing_time : ℝ := 1

-- Define the statement to be proved
theorem lawn_remaining_fraction (mary_mowing_rate tom_mowing_rate tom_mowing_time_alone mary_and_tom_mowing_time : ℝ) : 
  mary_mowing_rate = 1 / 3 → 
  tom_mowing_rate = 1 / 6 →
  tom_mowing_time_alone = 3 →
  mary_and_tom_mowing_time = 1 →
  (1 - ((tom_mowing_rate * tom_mowing_time_alone) + mary_and_tom_mowing_time * (tom_mowing_rate + mary_mowing_rate))) = 0 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end lawn_remaining_fraction_l143_143588


namespace range_of_g_l143_143375

def g (A : ℝ) : ℝ :=
  (sin A * (4 * cos A^2 + cos A^4 + 2 * sin A^2 + 2 * sin A^2 * cos A^2)) / 
  (tan A * (1 / cos A - 2 * sin A * tan A + 1))

theorem range_of_g (A : ℝ) (h : ∀ n : ℤ, A ≠ n * π / 2) : g A = 2 :=
  by sorry

end range_of_g_l143_143375


namespace exists_non_prime_value_of_P_nonconstant_l143_143158

noncomputable def P : Polynomial ℤ :=
  sorry -- assuming existence of such non-constant polynomial

def non_prime_value_exists (P : Polynomial ℤ) : Prop :=
  ∃ n : ℤ, ¬ nat.prime (P.eval (n^2 + 2020))

theorem exists_non_prime_value_of_P_nonconstant (h_nonconstant : ¬ P.degree ≤ 0) : non_prime_value_exists P :=
by
  sorry

end exists_non_prime_value_of_P_nonconstant_l143_143158


namespace rationalize_sqrt_fraction_l143_143619

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143619


namespace rationalize_sqrt_5_over_12_l143_143643

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143643


namespace rationalize_sqrt_5_over_12_l143_143651

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143651


namespace rationalize_sqrt_fraction_l143_143632

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143632


namespace quadratic_inequality_solution_l143_143715

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x - 8 ≤ 0 ↔ -4/3 ≤ x ∧ x ≤ 2 :=
sorry

end quadratic_inequality_solution_l143_143715


namespace tangent_line_at_1_eq_a_minus_1_monotonicity_of_f_l143_143078

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := real.sqrt x - a * real.log (x + 1)

-- Part (1): Tangent line at x = 1 when a = -1
theorem tangent_line_at_1_eq_a_minus_1 :
  (∀ x, f x (-1) = real.sqrt x + real.log (x + 1)) →
  let f' (x : ℝ) : ℝ := (1 / (2 * real.sqrt x)) + (1 / (x + 1)) in
  let x := 1 in
  f' x = 1 ∧ f x (-1) = 1 + real.log 2 →
  (∀ (y : ℝ), y = f'[x] * (x - 1) + f (1) (-1) → y = x + real.log 2) :=
  sorry

-- Part (2): Monotonicity of f(x)
theorem monotonicity_of_f:
  ∀ (a : ℝ),
  (∀ x, f x a = real.sqrt x - a * real.log (x + 1)) →
  
  (if a ≤ 1 then 
    ∀ x y, (0 ≤ x) → (0 ≤ y) → (x ≤ y) → (f x a ≤ f y a)
  else
    let x1 := (a - real.sqrt(a^2 - 1))^2 in
    let x2 := (a + real.sqrt(a^2 - 1))^2 in
    ∀ x y, 
      ((0 ≤ x) → (0 ≤ y) → (x < x1 → y < x1 → f x a ≥ f y a))
    ∧ ((x1 ≤ x) → (x < x2) → ((0 ≤ y) → (y < x1) → f x a ≥ f y a))
    ∧ ((x2 ≤ x) → (x ≤ y) → f x a ≥ f y a)) :=
  sorry

end tangent_line_at_1_eq_a_minus_1_monotonicity_of_f_l143_143078


namespace total_candies_by_boys_invariant_l143_143530

theorem total_candies_by_boys_invariant (k : ℕ) (candies : ℕ) (children : List Bool) :
  candies = 1000 →
  children.length = k →
  ∀ child_order1 child_order2 : List Bool,
    (∀ (i : ℕ), (i < k → (child_order1.nth i = children.nth i ∨ child_order1.nth i = children.nth (k - i - 1)))
    → (child_order2.nth i = children.nth i ∨ child_order2.nth i = children.nth (k - i - 1)))) →
  let round_up (x : ℕ) (d : ℕ) := if x % d = 0 then x / d else x / d + 1,
      round_down (x : ℕ) (d : ℕ) := x / d in
  let take_candies := λ (order : List Bool) (total : ℕ) (n : ℕ), 
    List.foldl (λ (acc : ℕ × ℕ) (child_type : Bool), 
       if child_type then (acc.1 + round_up acc.2 n, acc.2 - round_up acc.2 n) 
       else (acc.1, acc.2 - round_down acc.2 n)) (0, total) order in
  let total_boys_candies := λ (order : List Bool), (take_candies order candies k).1 in
  total_boys_candies child_order1 = total_boys_candies child_order2 :=
by
  intros h_candies h_length h_perm
  sorry


end total_candies_by_boys_invariant_l143_143530


namespace radioactive_numbers_count_l143_143705

def is_prime_factor (n p : ℕ) : Prop :=
  (p.prime) ∧ (p ∣ n)

def is_radioactive (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime_factor n p ∧ (p > nat.sqrt n)

def all_prime_factors_lt_30 (n : ℕ) : Prop :=
  ∀ p, is_prime_factor n p → p < 30

theorem radioactive_numbers_count : 
  (∑ p in (finset.filter (λ p, p.prime) (finset.range 30)), (p - 1)) = 119 := 
by sorry

end radioactive_numbers_count_l143_143705


namespace union_sets_S_T_l143_143058

open Set Int

def S : Set Int := { s : Int | ∃ n : Int, s = 2 * n + 1 }
def T : Set Int := { t : Int | ∃ n : Int, t = 4 * n + 1 }

theorem union_sets_S_T : S ∪ T = S := 
by sorry

end union_sets_S_T_l143_143058


namespace find_g_2022_l143_143006

def g : ℝ → ℝ := sorry -- This is pre-defined to say there exists such a function

theorem find_g_2022 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y)) :
  g 2022 = 4086462 :=
sorry

end find_g_2022_l143_143006


namespace overall_percentage_reduction_is_32_l143_143330

-- Define the initial price as a percentage (100%)
def initial_price : ℝ := 100

-- Define the first reduction percentage
def first_reduction : ℝ := 15

-- Define the second reduction percentage
def second_reduction : ℝ := 20

-- Compute the intermediate price after the first reduction
def intermediate_price : ℝ := initial_price - ((first_reduction / 100) * initial_price)

-- Compute the second reduction amount
def second_reduction_amount : ℝ := (second_reduction / 100) * intermediate_price

-- Compute the final price after the second reduction
def final_price : ℝ := intermediate_price - second_reduction_amount

-- Compute the overall percentage reduction
def overall_reduction : ℝ := initial_price - final_price

-- Theorem to prove the overall percentage reduction is 32%
theorem overall_percentage_reduction_is_32 : overall_reduction = 32 := by
  sorry

end overall_percentage_reduction_is_32_l143_143330


namespace rationalize_sqrt_fraction_l143_143628

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143628


namespace rationalize_denominator_l143_143652

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143652


namespace greatest_integer_not_exceeding_1000x_l143_143798

-- Define the conditions.
def edge_length : ℝ := 2
def shadow_area_excluding_cube : ℝ := 300

-- Variables for position of the light source and total shadow area.
def x : ℝ := (edge_length + 1) / 9
def shadow_area_total : ℝ := shadow_area_excluding_cube + (edge_length * edge_length)

-- Define the main theorem that translates the problem into a proof.
theorem greatest_integer_not_exceeding_1000x : 
  greatest_integer (1000 * ((√19) + 1) / 9) = 706 := 
  by sorry

end greatest_integer_not_exceeding_1000x_l143_143798


namespace ratio_of_b_to_a_l143_143098

noncomputable def positive_reals := {x : ℝ // 0 < x}

theorem ratio_of_b_to_a (a b : positive_reals) (n : ℕ) (h_n : n = 4)
    (h_eq : (a.val + b.val * complex.I)^n = (a.val - b.val * complex.I)^n) :
    b.val / a.val = 1 :=
by
  sorry

end ratio_of_b_to_a_l143_143098


namespace sqrt_difference_l143_143360

theorem sqrt_difference : sqrt 45 - sqrt 20 = sqrt 5 := by
  sorry

end sqrt_difference_l143_143360


namespace rationalize_sqrt_fraction_l143_143627

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143627


namespace events_A_B_equal_prob_l143_143511

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l143_143511


namespace lines_passing_through_four_points_l143_143127

-- Definitions for the problem conditions
def is_valid_point (i j k : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 ∧ 1 ≤ k ∧ k ≤ 4

-- Statement to prove the number of lines passing through four different points is 76
theorem lines_passing_through_four_points : 
  (∃ (l : set (ℕ × ℕ × ℕ)), 
    (∀ p ∈ l, is_valid_point p.1 p.2.1 p.2.2) ∧ 
    l.card = 4) → 
    ∑ l in some_set_containing_lines, 1 = 76 := 
sorry

end lines_passing_through_four_points_l143_143127


namespace geometric_sequence_fourth_term_l143_143499

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℕ)
  (r : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₅ = 2187)
  (h₃ : a₅ = a₁ * r ^ 4) :
  a₁ * r ^ 3 = 2187 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l143_143499


namespace rationalize_denominator_l143_143693

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143693


namespace minimum_value_expression_l143_143026

theorem minimum_value_expression (a b : ℝ) (hne : b ≠ 0) :
    ∃ a b, a^2 + b^2 + a / b + 1 / b^2 = sqrt 3 :=
sorry

end minimum_value_expression_l143_143026


namespace area_of_triangle_formed_by_tangents_l143_143265

/-- Two circles of radii R and r touch externally. The area of the triangle formed by 
the three common tangents to the circles in terms of R and r is given by 
    2 * (R * r)^(3/2) / (R - r) -/
theorem area_of_triangle_formed_by_tangents (R r : ℝ) (hR : 0 < R) (hr : 0 < r) (h : R ≠ r) :
  let area_tangent_triangle := (2 * (R * r)^(3/2)) / (R - r)
  in area_tangent_triangle = (2 * (R * r)^(3/2)) / (R - r)
:= by
  sorry

end area_of_triangle_formed_by_tangents_l143_143265


namespace dimensions_multiple_of_three_l143_143788

theorem dimensions_multiple_of_three (a b c : ℤ) (h : a * b * c = (a + 1) * (b + 1) * (c - 2)) :
  (a % 3 = 0) ∨ (b % 3 = 0) ∨ (c % 3 = 0) :=
sorry

end dimensions_multiple_of_three_l143_143788


namespace sum_of_positive_divisors_of_24_l143_143292

theorem sum_of_positive_divisors_of_24 : ∑ n in {n : ℕ | n > 0 ∧ (n+24) % n = 0}, n = 60 := 
by sorry

end sum_of_positive_divisors_of_24_l143_143292


namespace rationalize_sqrt_fraction_l143_143634

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143634


namespace find_g_l143_143232

noncomputable def g : ℝ → ℝ := sorry

theorem find_g :
  (g 1 = 2) ∧ (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) ↔ (∀ x : ℝ, g x = 2 * (4^x - 3^x)) := 
by
  sorry

end find_g_l143_143232


namespace simplify_fraction_l143_143215

-- Define the original expressions
def expr1 := 3 / (Real.sqrt 5 + 2)
def expr2 := 4 / (Real.sqrt 7 - 2)

-- State the mathematical problem.
theorem simplify_fraction :
  (1 / (expr1 + expr2)) =
  ((9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) / 
  ((9 * Real.sqrt 5 + 4 * Real.sqrt 7) ^ 2 - 100)) :=
by sorry

end simplify_fraction_l143_143215


namespace no_possible_arrangement_l143_143982

open Finset

-- Define the set of weights from 1 to 100
def weights := (range 100).map (λ n, n + 1)

-- Statement of the problem
theorem no_possible_arrangement :
  ∀ (piles : Finset (Finset ℕ)),
    (∀ p ∈ piles, (∀ w ∈ p, w ∈ weights) ∧ p.card = 10) →
    (∀ p₁ p₂ ∈ piles, p₁ ≠ p₂ → p₁.sum < p₂.sum → p₁.card > p₂.card) →
    piles.card = 10 →
    False :=
by
  -- Proof skipped, as instructed
  sorry

end no_possible_arrangement_l143_143982


namespace cos_double_angle_identity_l143_143946

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_double_angle_identity_l143_143946


namespace petya_wins_second_race_l143_143201

theorem petya_wins_second_race 
  (v_P v_V : ℝ) -- Petya's and Vasya's speeds
  (h1 : v_V = 0.9 * v_P) -- Condition from the first race
  (d_P d_V : ℝ) -- Distances covered by Petya and Vasya in the first race
  (h2 : d_P = 100) -- Petya covers 100 meters
  (h3 : d_V = 90) -- Vasya covers 90 meters
  (start_diff : ℝ) -- Initial distance difference in the second race
  (h4 : start_diff = 10) -- Petya starts 10 meters behind Vasya
  (race_length : ℝ) -- Total race length
  (h5 : race_length = 100) -- The race is 100 meters long
  : (v_P * (race_length / v_P) - v_V * (race_length / v_P)) = 1 :=
by
  sorry

end petya_wins_second_race_l143_143201


namespace cabinets_and_perimeter_l143_143554

theorem cabinets_and_perimeter :
  ∀ (original_cabinets : ℕ) (install_factor : ℕ) (num_counters : ℕ) 
    (cabinets_L_1 cabinets_L_2 cabinets_L_3 removed_cabinets cabinet_height total_cabinets perimeter : ℕ),
    original_cabinets = 3 →
    install_factor = 2 →
    num_counters = 4 →
    cabinets_L_1 = 3 →
    cabinets_L_2 = 5 →
    cabinets_L_3 = 7 →
    removed_cabinets = 2 →
    cabinet_height = 2 →
    total_cabinets = (original_cabinets * install_factor * num_counters) + 
                     (cabinets_L_1 + cabinets_L_2 + cabinets_L_3) - removed_cabinets →
    perimeter = (cabinets_L_1 * cabinet_height) +
                (cabinets_L_3 * cabinet_height) +
                2 * (cabinets_L_2 * cabinet_height) →
    total_cabinets = 37 ∧
    perimeter = 40 :=
by
  intros
  sorry

end cabinets_and_perimeter_l143_143554


namespace find_x_l143_143873

open Nat

def has_three_distinct_prime_factors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a * b * c

theorem find_x (n : ℕ) (h₁ : Odd n) (h₂ : 7^n + 1 = x)
  (h₃ : has_three_distinct_prime_factors x) (h₄ : 11 ∣ x) : x = 16808 := by
  sorry

end find_x_l143_143873


namespace sum_of_divisors_of_24_eq_60_l143_143296

theorem sum_of_divisors_of_24_eq_60 : 
  (∑ n in { n | n ∣ 24 ∧ 0 < n }.toFinset, n) = 60 := 
sorry

end sum_of_divisors_of_24_eq_60_l143_143296


namespace factorization_a4_plus_4_l143_143721

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 - 2*a + 2) * (a^2 + 2*a + 2) :=
by sorry

end factorization_a4_plus_4_l143_143721


namespace overall_average_of_marks_l143_143718

theorem overall_average_of_marks (n total_boys passed_boys failed_boys avg_passed avg_failed : ℕ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 105)
  (h3 : failed_boys = 15)
  (h4 : total_boys = passed_boys + failed_boys)
  (h5 : avg_passed = 39)
  (h6 : avg_failed = 15) :
  ((passed_boys * avg_passed + failed_boys * avg_failed) / total_boys = 36) :=
by
  sorry

end overall_average_of_marks_l143_143718


namespace fiona_total_evaluations_l143_143846

theorem fiona_total_evaluations :
  ∀ n k : ℕ, (n = 13) ∧ (k = 2) → 3 * (Nat.choose n k) = 234 :=
by
  intros n k h
  cases h
  sorry

end fiona_total_evaluations_l143_143846


namespace good_arrangement_exists_for_1983_gon_l143_143140

-- Define the concept of a regular n-gon
structure Polygon (n : ℕ) :=
  (vertices : Fin n → ℕ)

-- Define the concept of a "good" arrangement with respect to an axis of symmetry
def is_good_arrangement (n : ℕ) (p : Polygon n) : Prop :=
  ∀ axis : Fin n, let (left, right) := axis_partition n axis
  in ∀ i : Fin (n/2), p.vertices (left i) > p.vertices (right i)

-- Helper function to compute the symmetric partition of vertices
def axis_partition (n : ℕ) (axis : Fin n) : (Fin (n/2) → Fin n) × (Fin (n/2) → Fin n) :=
  sorry

-- The theorem to prove
theorem good_arrangement_exists_for_1983_gon :
  ∃ p : Polygon 1983, is_good_arrangement 1983 p :=
sorry

end good_arrangement_exists_for_1983_gon_l143_143140


namespace rationalize_sqrt_fraction_denom_l143_143612

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143612


namespace percentage_saved_last_year_l143_143989

variable (S P : ℝ)

theorem percentage_saved_last_year :
  (P / 100) * S = 0.05 * 1.20 * S → P = 6 := by
  intro h
  have h1 : (P / 100) * S = (0.06 * S) := by
    rw [mul_comm]
    norm_num
    rw [← mul_assoc]
    exact h
  have h2 : (P / 100) = 0.06 := by
    rw [mul_right_eq_self] at h1
    exact h1
  have h3 : P = 0.06 * 100 := by
    rw [eq_div_iff] at h2
    exact h2
  norm_num at h3
  exact h3

end percentage_saved_last_year_l143_143989


namespace problem_statement_l143_143374

-- Define the given exponentials as complex numbers on the unit circle

def e1 := Complex.exp (11 * Real.pi * Complex.I / 60)
def e2 := Complex.exp (23 * Real.pi * Complex.I / 60)
def e3 := Complex.exp (35 * Real.pi * Complex.I / 60)
def e4 := Complex.exp (47 * Real.pi * Complex.I / 60)
def e5 := Complex.exp (59 * Real.pi * Complex.I / 60)
def S := e1 + e2 + e3 + e4 + e5

-- Define the expected resulting argument of the sum S
def θ := 7 * Real.pi / 12

-- The theorem that the argument of S equals θ.
theorem problem_statement : Complex.arg S = θ := 
by sorry

end problem_statement_l143_143374


namespace k_values_l143_143162

-- Definitions of the given conditions
variables {V : Type*} [inner_product_space ℝ V] -- Define a real inner product space
variables {a b c : V}
variables (k : ℝ)

-- Let a, b, and c be unit vectors
axiom a_unit : ∥a∥ = 1
axiom b_unit : ∥b∥ = 1
axiom c_unit : ∥c∥ = 1

-- Given conditions
axiom a_dot_b_zero : ⟪a, b⟫ = 0
axiom a_dot_c_zero : ⟪a, c⟫ = 0

-- The angle between b and c is π/3
axiom angle_b_c : real.angle b c = real.pi / 3

-- The goal
theorem k_values : ∃ k : ℝ, (a = k • (b × c)) ∧ (k = (2 * real.sqrt 3) / 3 ∨ k = -(2 * real.sqrt 3) / 3) :=
sorry

end k_values_l143_143162


namespace find_BT_equation_l143_143025

-- Define ellipse and points
def is_on_ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 9) = 1

def focus : ℝ × ℝ := (4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define points A, B, and C
variables {x1 y1 x2 y2 : ℝ}
def A := (x1, y1)
def B : ℝ × ℝ := (4, 9 / 5)
def C := (x2, y2)

-- Define conditions
def points_on_ellipse : Prop := is_on_ellipse x1 y1 ∧ is_on_ellipse 4 (9 / 5) ∧ is_on_ellipse x2 y2

def distances_form_arith_seq : Prop := 
  let dA := distance A focus
  let dB := distance B focus
  let dC := distance C focus
  dB = (dA + dC) / 2

def midpoint_AC : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define perpendicular bisector of AC
def is_on_perp_bisector (T : ℝ × ℝ) :
  T.2 = 0 ∧ 
  (T.1 - midpoint_AC.1) * (x1 - x2) + midpoint_AC.2 * (y1 - y2) = 0

-- Define line BT and its equation
def line_eq (m : ℝ) (x1 y1 : ℝ) : ℝ → ℝ := fun x => m * (x - x1) + y1

def line_BT_eq : Prop :=
  ∃ m x0,
    is_on_perp_bisector (x0, 0) ∧
    let BT := line_eq m 4 (9 / 5)
    25 * (BT x0) - 20 * (BT 4) = 64

-- Proof problem statement
theorem find_BT_equation : 
  points_on_ellipse ∧ distances_form_arith_seq → line_BT_eq := by
  sorry

end find_BT_equation_l143_143025


namespace positional_relationship_l143_143450

-- Definitions of the necessary conditions and the corresponding statement
universe u

variables {Point Line Plane : Type u} [HasMem Point Line] [HasMem Point Plane] [HasMem Line Plane]

def parallel_planes (α β : Plane) : Prop :=
  ∀ (p₁ : Point), p₁ ∈ α → ∀ (p₂ : Point), p₂ ∈ β → p₁ ≠ p₂

def contained_in (l : Line) (α : Plane) : Prop :=
  ∀ (p : Point), p ∈ l → p ∈ α

def parallel_or_skew (a b : Line) : Prop :=
  (∀ (p₁ : Point), p₁ ∈ a → ∀ (p₂ : Point), p₂ ∈ b → p₁ ≠ p₂) ∨
    (∀ (p₁ : Point), p₁ ∈ a → ∀ (p₂ : Point), p₂ ∈ b → ¬ (p₁, p₂) (Set.inter)

theorem positional_relationship {α β : Plane} {a b : Line} :
  parallel_planes α β →
  contained_in a α →
  contained_in b β →
  parallel_or_skew a b :=
sorry

end positional_relationship_l143_143450


namespace max_triangle_area_l143_143064

-- Define the conditions and the problem
theorem max_triangle_area (R : ℝ) (hR : 1 ≤ R) (b c : ℝ) (hb : b = 1) (hc : c = 1) :
    ∃ A : ℝ, 0 < A ∧ A < π ∧ 
    let sin_A := Real.sin A;
    let area := (1/2) * b * c * sin_A 
    in area = sqrt(3)/4 := 
by
  sorry

end max_triangle_area_l143_143064


namespace box_height_proof_l143_143785

-- The problem and assumptions
variable (length width height : ℝ)
variable (cube_volume : ℝ := 9)
variable (num_cubes : ℕ := 42)
variable (total_volume : ℝ := num_cubes * cube_volume)

-- Definitions based on the conditions
def box_volume : ℝ := length * width * height

-- Conditions assumed about the box
axiom length_def : length = 7
axiom width_def : width = 18
axiom volume_def : total_volume = box_volume

-- The theorem to prove
theorem box_height_proof : height = 3 :=
  by
    sorry

end box_height_proof_l143_143785


namespace find_x_l143_143872

open Nat

def has_three_distinct_prime_factors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a * b * c

theorem find_x (n : ℕ) (h₁ : Odd n) (h₂ : 7^n + 1 = x)
  (h₃ : has_three_distinct_prime_factors x) (h₄ : 11 ∣ x) : x = 16808 := by
  sorry

end find_x_l143_143872


namespace circle_radius_l143_143431

theorem circle_radius (A B C O : Type) (AB AC : ℝ) (OA : ℝ) (r : ℝ) 
  (h1 : AB * AC = 60)
  (h2 : OA = 8) 
  (h3 : (8 + r) * (8 - r) = 60) : r = 2 :=
sorry

end circle_radius_l143_143431


namespace pb_pc_lt_ad_l143_143551

-- Needed definitions for points, distances, and angles
variables {Point : Type*} [coordinate_space Point]

def distance (a b : Point) : ℝ := sorry
def angle (a b c : Point) : real.angle := sorry

-- Convex quadrilateral properties
variables (A B C D P : Point)
variables (h_convex : convex_quadrilateral A B C D)
variables (h_AB_eq_CD : distance A B = distance C D)
variables (h_angles_sum : angle P B A + angle P C D = 180)

-- Goal: Prove that PB + PC < AD
theorem pb_pc_lt_ad (h_convex : convex_quadrilateral A B C D) 
                     (h_AB_eq_CD : distance A B = distance C D)
                     (h_angles_sum : angle P B A + angle P C D = 180) :
  distance P B + distance P C < distance A D := 
sorry

end pb_pc_lt_ad_l143_143551


namespace petrol_price_l143_143310

theorem petrol_price (P : ℝ) (h : 0.9 * P = 0.9 * P) : (250 / (0.9 * P) - 250 / P = 5) → P = 5.56 :=
by
  sorry

end petrol_price_l143_143310


namespace domain_of_f_l143_143833

theorem domain_of_f (x : ℝ) :
  (-115 ≤ x ∧ x ≤ 6) ↔ 
  (4 - real.sqrt (5 - real.sqrt (6 - x)) ≥ 0 ∧
   real.sqrt (6 - x) ≥ 0) :=
by
  sorry

end domain_of_f_l143_143833


namespace final_number_less_than_one_over_n_l143_143240

theorem final_number_less_than_one_over_n (n : ℕ) (h : n > 0) :
  (∀ (s : ℝ) (numbers : list ℝ), 
    numbers = list.range (4^n) ++ 1 →
    (∀ a b ∈ numbers, 
      let c := (a * b) / (Real.sqrt (2 * a ^ 2 + 2 * b ^ 2)) in
      numbers = (list.erase numbers a).erase b ++ [c]) →
    list.length numbers = 1 →
    s < 1 / n) := sorry

end final_number_less_than_one_over_n_l143_143240


namespace rationalize_sqrt_fraction_denom_l143_143608

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143608


namespace minimum_value_expression_l143_143027

theorem minimum_value_expression (a b : ℝ) (hne : b ≠ 0) :
    ∃ a b, a^2 + b^2 + a / b + 1 / b^2 = sqrt 3 :=
sorry

end minimum_value_expression_l143_143027


namespace fifth_segment_student_l143_143268

variable (N : ℕ) (n : ℕ) (second_segment_student : ℕ)

def sampling_interval (N n : ℕ) : ℕ := N / n

def initial_student (second_segment_student interval : ℕ) : ℕ := second_segment_student - interval

def student_number (initial_student interval : ℕ) (segment : ℕ) : ℕ :=
  initial_student + (segment - 1) * interval

theorem fifth_segment_student (N n : ℕ) (second_segment_student : ℕ) (hN : N = 700) (hn : n = 50) (hsecond : second_segment_student = 20) :
  student_number (initial_student second_segment_student (sampling_interval N n)) (sampling_interval N n) 5 = 62 := by
  sorry

end fifth_segment_student_l143_143268


namespace sum_n_4_to_12_l143_143001

theorem sum_n_4_to_12 :
  (∑ n in finset.range (12 + 1), if n >= 4 then n * (1 - (1 / (n - 1))) else 0) = 74.5 :=
by sorry

end sum_n_4_to_12_l143_143001


namespace standard_equation_of_ellipse_maximum_area_triangle_OAB_is_1_l143_143886

-- Given:
def ellipse_passes_through_focus_parabola (a b : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ), (F.1 = 0 ∧ F.2 = 1) ∧ (a > b) ∧ (b > 0) ∧
  (F.2 = b) ∧ (let c := Real.sqrt (a^2 - b^2) in 2 * c^2 = 6)

-- Proving:
theorem standard_equation_of_ellipse (a b : ℝ) (h : ellipse_passes_through_focus_parabola a b) :
  a = 2 ∧ b = 1 ∧ (a > b) ∧ (b > 0) :=
sorry

-- Given:
def max_area_of_triangle_OAB (C : ℝ × ℝ → Prop) (P tangent_to parabola_and_line : C tanget M) : Prop :=
  ∃ (x0 : ℝ), 0 < x0^2 ∧ x0^2 < 8 + 4 * Real.sqrt 5 ∧
  let d := (x0^2) / (2 * Real.sqrt (x0^2 + 4)) in
  let |AB| := (Real.sqrt ((4 + x0^2) / (4 * (1 + x0^2)))) in
  let area := (d * |AB|) / 2 in area ≤ 1

-- Proving:
theorem maximum_area_triangle_OAB_is_1 (C : ℝ × ℝ → Prop) (P tangent_to parabola_and_line : C tanget M)
  (h : max_area_of_triangle_OAB C P) :
  ∃ (x0 : ℝ), x0^2 = 4 + 2 * Real.sqrt 6 ∧
  let d := (x0^2) / (2 * Real.sqrt (x0^2 + 4)) in
  let |AB| := (Real.sqrt ((4 + x0^2) / (4 * (1 + x0^2)))) in
  let area := (d * |AB|) / 2 in area = 1 :=
sorry

end standard_equation_of_ellipse_maximum_area_triangle_OAB_is_1_l143_143886


namespace fair_dice_game_l143_143308

theorem fair_dice_game : 
  let outcomes := [(x, y) | x <- [1,2,3,4,5,6], y <- [1,2,3,4,5,6]] in
  let odd_sum := [(x, y) | (x, y) ∈ outcomes, (x + y) % 2 = 1] in
  let even_sum := [(x, y) | (x, y) ∈ outcomes, (x + y) % 2 = 0] in
  probability (odd_sum.length.to_real / outcomes.length.to_real) = probability (even_sum.length.to_real / outcomes.length.to_real) :=
sorry

end fair_dice_game_l143_143308


namespace rationalize_denominator_l143_143678

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143678


namespace transform_equation_l143_143744

theorem transform_equation (x : ℝ) (h₁ : x ≠ 3 / 2) (h₂ : 5 - 3 * x = 1) :
  x = 4 / 3 :=
sorry

end transform_equation_l143_143744


namespace rationalize_denominator_l143_143659

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143659


namespace find_x_l143_143865

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l143_143865


namespace non_parallel_lines_implies_unique_solution_l143_143793

variable (a1 b1 c1 a2 b2 c2 : ℝ)

def system_of_equations (x y : ℝ) := a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

def lines_not_parallel := a1 * b2 ≠ a2 * b1

theorem non_parallel_lines_implies_unique_solution :
  lines_not_parallel a1 b1 a2 b2 → ∃! (x y : ℝ), system_of_equations a1 b1 c1 a2 b2 c2 x y :=
sorry

end non_parallel_lines_implies_unique_solution_l143_143793


namespace imaginary_part_of_z_l143_143895

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem imaginary_part_of_z (θ : ℝ) (hz : is_pure_imaginary (sin (2 * θ) - 1 + complex.I * (sqrt 2 * cos θ - 1))) :
  (sqrt 2 * cos θ - 1) = -2 :=
by
  sorry

end imaginary_part_of_z_l143_143895


namespace hyperbola_imaginary_axis_length_l143_143067

theorem hyperbola_imaginary_axis_length (m : ℝ) (h_length : mx^2 + y^2 = 1) (h_length_imag_axis : 2) (h_m_neg : m < 0) : m = -1 :=
by
  sorry

end hyperbola_imaginary_axis_length_l143_143067


namespace power_function_determined_l143_143912

-- Definitions based on conditions
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Lean statement of the proof problem
theorem power_function_determined (h : power_function α (real.sqrt 3 / 3) = real.sqrt 3 / 9) :
  ∃ α, α = 3 ∧ (∀ x, power_function α x = x ^ 3) :=
by
  sorry

end power_function_determined_l143_143912


namespace event_probabilities_equal_l143_143519

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l143_143519


namespace angle_AMC_165_degrees_l143_143574

theorem angle_AMC_165_degrees {O A B C M : Point}
  (hO_center : is_circumcenter O A B C)
  (hO_opposite_B : O ∈ line_through A C ∧ B ∉ line_through A C) 
  (h_angle_AOC : ∠AOC = 60) 
  (hM_incenter : is_incenter M A B C) :
  ∠AMC = 165 := 
by 
parity
/-
This means proving the angle ∠AMC equals 165 degrees given:
1. O is the center of the circumcircle of triangle ABC
2. O and B lie on opposite sides of the line through A and C
3. The angle AOC equals 60 degrees
4. M is the incenter of triangle ABC.
-/
sorry -- proof would go here

end angle_AMC_165_degrees_l143_143574


namespace lionel_distance_walked_when_met_l143_143182

theorem lionel_distance_walked_when_met (distance_between : ℕ) (lionel_speed : ℕ) (walt_speed : ℕ) (advance_time : ℕ) 
(h1 : distance_between = 48) 
(h2 : lionel_speed = 2) 
(h3 : walt_speed = 6) 
(h4 : advance_time = 2) : 
  ∃ D : ℕ, D = 15 :=
by
  sorry

end lionel_distance_walked_when_met_l143_143182


namespace arc_length_correct_l143_143451

noncomputable def chord_length := 2
noncomputable def central_angle := 2
noncomputable def half_chord_length := 1
noncomputable def radius := 1 / Real.sin 1
noncomputable def arc_length := 2 * radius

theorem arc_length_correct :
  arc_length = 2 / Real.sin 1 := by
sorry

end arc_length_correct_l143_143451


namespace length_MN_eq_a_l143_143740

-- Define an acute-angled triangle ABC with given sides
variables {A B C M N : Type} [acute_angled_triangle ABC]

-- Parallel line through A parallel to side BC with length a
variable (a : ℝ)
variable (h_parallel : parallel A BC)

-- Circles constructed on diameters AB and AC intersecting at M and N respectively
variable (h_circle_AB : is_diameter_circle A B M)
variable (h_circle_AC : is_diameter_circle A C N)

-- Given the conditions, prove MN = a
theorem length_MN_eq_a 
  (h_AM_parallel : M ∈ A ∧ parallel AM BC)
  (h_AN_parallel : N ∈ A ∧ parallel AN BC)
  (h_MN_intersect : intersect AM h_circle_AB M)
  (h_NN_intersect : intersect AN h_circle_AC N)
  : length MN = a :=
sorry

end length_MN_eq_a_l143_143740


namespace no_real_x_satisfies_matrix_eq_l143_143494

theorem no_real_x_satisfies_matrix_eq :
  ∀ (x : ℝ),
  det (Matrix.of (λ i j, if (i, j) = (0, 0) then (3 * x) else
                            if (i, j) = (0, 1) then 2 else
                            if (i, j) = (1, 0) then (6 * x) else x)) ≠ 6 :=
by {
  sorry
}

end no_real_x_satisfies_matrix_eq_l143_143494


namespace remainder_correct_l143_143401

def P : Polynomial ℤ := X^4 + 3*X^2 - 4
def D : Polynomial ℤ := X^2 - 3

theorem remainder_correct : (P % D) = 14 :=
by
  sorry

end remainder_correct_l143_143401


namespace g_inverse_sum_eq_58_l143_143827

def g (x : ℝ) : ℝ :=
if x < 5 then x - 3 else sqrt (x - 1)

def a_inv (y : ℝ) : ℝ := y + 3
def b_inv (y : ℝ) : ℝ := y^2 + 1

theorem g_inverse_sum_eq_58 : 
  (a_inv (-6) + a_inv (-5) + a_inv (-4) + a_inv (-3) + a_inv (-2) + a_inv (-1) + a_inv (0) + a_inv (1)
  + b_inv (2) + b_inv (3) + b_inv (4) + b_inv (5)) = 58 :=
sorry

end g_inverse_sum_eq_58_l143_143827


namespace solve_n_m_l143_143444

noncomputable def exponents_of_linear_equation (n m : ℕ) (x y : ℝ) : Prop :=
2 * x ^ (n - 3) - (1 / 3) * y ^ (2 * m + 1) = 0

theorem solve_n_m (n m : ℕ) (x y : ℝ) (h_linear : exponents_of_linear_equation n m x y) :
  n ^ m = 1 :=
sorry

end solve_n_m_l143_143444


namespace rationalize_denominator_l143_143686

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143686


namespace S_n_formula_l143_143577

noncomputable def S : ℕ → ℝ
| 0       := 0
| (n + 1) := a (n + 1) + S n

noncomputable def a : ℕ → ℝ
| 0        := 0  -- a_0 is not defined, use 0 for simplicity
| 1        := -1
| (n + 2)  := S (n + 1) * S (n + 2)

theorem S_n_formula (n : ℕ) (hn : n ≠ 0) : S n = - (1:ℝ) / n := by 
  sorry

end S_n_formula_l143_143577


namespace cube_difference_l143_143593

theorem cube_difference (n : ℕ) (h: 0 < n) : (n + 1)^3 - n^3 = 3 * n^2 + 3 * n + 1 := 
sorry

end cube_difference_l143_143593


namespace integral_solution_l143_143023

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..1, ( sqrt(1 - (x - 1) ^ 2) - x ^ 2 )

theorem integral_solution : integral_problem = (Real.pi / 4) - (1 / 3) :=
by
  sorry

end integral_solution_l143_143023


namespace range_of_a_l143_143495

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, (0 < x ∧ x < 1) → 2^(1/x) > x^a

theorem range_of_a (a : ℝ) : satisfies_inequality a ↔ a > -Real.exp(1) * Real.log 2 := by
  sorry

end range_of_a_l143_143495


namespace find_n_solution_l143_143021

theorem find_n_solution (n : ℚ) (h : (2 / (n+2)) + (4 / (n+2)) + (n / (n+2)) = 4) : n = -2 / 3 := 
by 
  sorry

end find_n_solution_l143_143021


namespace time_to_fill_bottle_l143_143357

-- Definitions
def flow_rate := 500 / 6 -- mL per second
def volume := 250 -- mL

-- Target theorem
theorem time_to_fill_bottle (r : ℝ) (v : ℝ) (t : ℝ) (h : r = flow_rate) (h2 : v = volume) : t = 3 :=
by
  sorry

end time_to_fill_bottle_l143_143357


namespace rationalize_sqrt_fraction_l143_143674

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143674


namespace cost_of_fencing_approx_l143_143764

def diameter : ℝ := 28
def rate_per_meter : ℝ := 1.50
noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d
noncomputable def total_cost (d : ℝ) (rate : ℝ) : ℝ := rate * circumference d

theorem cost_of_fencing_approx :
  total_cost diameter rate_per_meter ≈ 131.95 :=
sorry

end cost_of_fencing_approx_l143_143764


namespace find_x_l143_143877

theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_factors : ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧ (7^n + 1) = p1 * p2 * p3 ∧ (p1 = 2 ∨ p2 = 2 ∨ p3 = 2) ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
  7^n + 1 = 16808 :=
sorry

end find_x_l143_143877


namespace find_x_l143_143864

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l143_143864


namespace parabola_focus_conditions_l143_143881

/-- Given a parabola with its vertex at the origin, the following are the conditions:
 1. The focus is on the y-axis
 2. The focus is on the x-axis
 3. The distance from the point on the parabola with an x-coordinate of 1 to the focus is 6
 4. The length of the latus rectum is 5
 5. A perpendicular line is drawn from the origin to a certain line passing through the focus, and the foot of the perpendicular has coordinates (2,1). 
Prove that the conditions that make the equation of the parabola y^2 = 10x are conditions 2 and 5 -/
theorem parabola_focus_conditions 
(Condition_2 : ∃ x : ℝ, focus_eq : (x, 0))
(Condition_5 : (∃ x y : ℝ, (2,1) := (x, y) ∧ (x ≠ 0 ∧ y ≠ 0))) :
y^2 = 10x ↔ (Condition_2 ∧ Condition_5) :=
sorry

end parabola_focus_conditions_l143_143881


namespace find_e_value_l143_143425

-- Define constants a, b, c, d, and e
variables (a b c d e : ℝ)

-- Theorem statement
theorem find_e_value (h1 : (2 : ℝ)^7 * a + (2 : ℝ)^5 * b + (2 : ℝ)^3 * c + 2 * d + e = 23)
                     (h2 : ((-2) : ℝ)^7 * a + ((-2) : ℝ)^5 * b + ((-2) : ℝ)^3 * c + ((-2) : ℝ) * d + e = -35) :
  e = -6 :=
sorry

end find_e_value_l143_143425


namespace solution_count_l143_143484

open Real

noncomputable def count_solutions (a b : ℝ) (f : ℝ → ℝ) :=
  { θ | θ ∈ Ioo a b ∧ f θ = 0 }.toFinset.card

-- Definitions for the trigonometric identities and interval
def g (θ : ℝ) : ℝ :=
  sin (2 * θ) - cos (2 * θ) - (sqrt 6 / 2)

def interval : Set ℝ := Set.Ioo (-π / 2) (π / 2)

-- The proof statement
theorem solution_count :
  count_solutions (-π / 2) (π / 2) g = 4 :=
sorry

end solution_count_l143_143484


namespace var_of_or_l143_143927

theorem var_of_or (p q : Prop) (h : ¬ (p ∧ q)) : (p ∨ q = true) ∨ (p ∨ q = false) :=
by
  sorry

end var_of_or_l143_143927


namespace maximize_distance_of_tangent_point_l143_143196

theorem maximize_distance_of_tangent_point (c : ℝ) (r : ℝ) (h1 : c > 1) (h2 : 0 ≤ r ∧ r ≤ c - 1) :
  r = -3 / 4 + sqrt (c^2 / 2 + 1 / 16) :=
sorry

end maximize_distance_of_tangent_point_l143_143196


namespace efficiency_of_worker_p_more_than_q_l143_143799

noncomputable def worker_p_rate : ℚ := 1 / 22
noncomputable def combined_rate : ℚ := 1 / 12

theorem efficiency_of_worker_p_more_than_q
  (W_p : ℚ) (W_q : ℚ)
  (h1 : W_p = worker_p_rate)
  (h2 : W_p + W_q = combined_rate) : (W_p / W_q) = 6 / 5 :=
by
  sorry

end efficiency_of_worker_p_more_than_q_l143_143799


namespace rationalize_sqrt_fraction_denom_l143_143609

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143609


namespace rationalize_sqrt_fraction_l143_143623

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143623


namespace unique_solution_to_equation_l143_143017

theorem unique_solution_to_equation (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) 
  (h : 1 + 5^x = 2^y + 2^z * 5^t) : (x, y, z, t) = (2, 4, 1, 1) := 
sorry

end unique_solution_to_equation_l143_143017


namespace problem_statement_l143_143413

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let CA := (A.1 - C.1, A.2 - C.2)
  let CB := (B.1 - C.1, B.2 - C.2)
  (1 / 2) * abs (CA.1 * CB.2 - CA.2 * CB.1)

theorem problem_statement : 
  let A := (2, 3)
  let B := (-3, -7)
  let C := (4, -2) 
  area_of_triangle A B C = 22.5 := 
by 
  -- proof skipped
  sorry

end problem_statement_l143_143413


namespace alpha_interval_l143_143432

open Real

theorem alpha_interval (α : ℝ) (h0: 0 ≤ α) (h1: α < 2 * π) (h2: sin α - cos α > 0) (h3: tan α > 0) :
  (π / 4 < α ∧ α < π / 2) ∨ (π < α ∧ α < 5 * π / 4) :=
begin
  sorry
end

end alpha_interval_l143_143432


namespace regression_unit_change_l143_143048

theorem regression_unit_change (x : ℝ) :
  let y := 3 - 5*x,
      y' := 3 - 5*(x + 1) in
  y' = y - 5 :=
by
  sorry

end regression_unit_change_l143_143048


namespace find_sin_theta_l143_143563

variables {a b c : ℝ^3} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variables (h1 : ¬ (∃ k : ℝ, a = k • b)) (h2 : ¬ (∃ k : ℝ, a = k • c)) (h3 : ¬ (∃ k : ℝ, b = k • c))
variables (angle_bc : ℝ) (θ : angle_bc = real.angle b c)

theorem find_sin_theta
  (h : (a × b) × c = (1/4) * ∥b∥ * ∥c∥ • a) : real.sin θ = (sqrt 15) / 4 :=
by sorry

end find_sin_theta_l143_143563


namespace minimum_y0_plus_PQ_l143_143887

noncomputable def dist (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

def parabola (x : ℝ) : ℝ := (x ^ 2) / 4

theorem minimum_y0_plus_PQ :
  let Q := (2 * real.sqrt 2, 0),
      y (x : ℝ) := x ^ 2 / 4,
      P (x₀ y₀ : ℝ) := y₀ = y x₀,
      PQ (x₀ y₀ : ℝ) := dist (x₀, y₀) Q in
  ∀ x₀ y₀ : ℝ, y₀ = y x₀ → (y₀ + PQ x₀ y₀) ≥ 1 :=
by
  sorry

end minimum_y0_plus_PQ_l143_143887


namespace fourth_number_second_set_l143_143724

theorem fourth_number_second_set :
  (∃ (x y : ℕ), (28 + x + 42 + 78 + 104) / 5 = 90 ∧ (128 + 255 + 511 + y + x) / 5 = 423 ∧ x = 198) →
  (y = 1023) :=
by
  sorry

end fourth_number_second_set_l143_143724


namespace probabilities_equal_l143_143500

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l143_143500


namespace four_digit_parity_monotonic_count_l143_143365

/-- A function that determines whether a digit is odd. -/
def isOdd (digit : ℕ) : Prop := digit % 2 = 1

/-- A function that determines whether a digit is even. -/
def isEven (digit : ℕ) : Prop := digit % 2 = 0

/-- A function that checks if a given list of digits represents a parity-monotonic integer. -/
def isParityMonotonic (digits : List ℕ) : Prop :=
  ∀ (i : ℕ), (i < digits.length - 1) →
  (isOdd (digits[i]) → digits[i] < digits[i + 1]) ∧
  (isEven (digits[i]) → digits[i] > digits[i + 1])

/-- The number of four-digit parity-monotonic integers is 576. -/
theorem four_digit_parity_monotonic_count : 
  ∃ (count : ℕ), count = 576 ∧ ∀ (digits : List ℕ),
    digits.length = 4 → 
    (isParityMonotonic digits ↔ digits ∈ List.range (10^4)) → count = 576 := sorry

end four_digit_parity_monotonic_count_l143_143365


namespace smaller_angle_parallelogram_l143_143956

theorem smaller_angle_parallelogram (x : ℕ) (h1 : ∀ a b : ℕ, a ≠ b ∧ a + b = 180) (h2 : ∃ y : ℕ, y = x + 70) : x = 55 :=
by
  sorry

end smaller_angle_parallelogram_l143_143956


namespace function_properties_l143_143465

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  √3 * sin (ω * x) + 2 * sin (ω * x / 2)^2

theorem function_properties (ω : ℝ) (h₀ : 13 / 6 ≤ ω ∧ ω < 19 / 6) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π) → 
    (f ω x = 2 * sin (ω * x - π / 6) + 1)
    ∧ (2 <= ⌊ω⌋ ∨ ⌊ω⌋ < 3)
    ∧ (∀ x, (0 < x ∧ x < π / 4) → f ω x ≠ 2)
    ∧ (∀ x, (0 < x ∧ x < π / 6) → f ω x > 0)) :=
by
  sorry

end function_properties_l143_143465


namespace probabilities_equal_l143_143501

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l143_143501


namespace intervals_of_increase_l143_143059

noncomputable theory

open Real

/-- Given f(x) = 2A cos^2(ωx + φ) - A with A > 0, ω > 0, and 0 < φ < π/2,
and given that the line x = π/3 and the point (π/12, 0)
are respectively a symmetry axis and symmetry center,
we prove that the function f(x) is increasing on the intervals
[kπ - 2π/3, kπ - π/6] for k ∈ ℤ. -/
theorem intervals_of_increase
  (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 2)
  (hx_sym : ∀ x, f (x) = f (π/3 - x)) 
  (hpt_sym : f (π/12) = 0) :
  ∀ k : ℤ, ∃ f : ℝ → ℝ,
    (∀ x, f x = 2 * A * cos(ω * x + φ) ^ 2 - A) ∧
    ∀ x : ℝ, interval (k * π - 2 * π / 3) (k * π - π / 6) (f x) :=
sorry

end intervals_of_increase_l143_143059


namespace sequence_problem_l143_143139

-- Given sequence
variable (P Q R S T U V : ℤ)

-- Given conditions
variable (hR : R = 7)
variable (hPQ : P + Q + R = 21)
variable (hQS : Q + R + S = 21)
variable (hST : R + S + T = 21)
variable (hTU : S + T + U = 21)
variable (hUV : T + U + V = 21)

theorem sequence_problem : P + V = 14 := by
  sorry

end sequence_problem_l143_143139


namespace avg_age_adults_l143_143526

-- Given conditions
def num_members : ℕ := 50
def avg_age_members : ℕ := 20
def num_girls : ℕ := 25
def num_boys : ℕ := 20
def num_adults : ℕ := 5
def avg_age_girls : ℕ := 18
def avg_age_boys : ℕ := 22

-- Prove that the average age of the adults is 22 years
theorem avg_age_adults :
  (num_members * avg_age_members - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_adults = 22 :=
by 
  sorry

end avg_age_adults_l143_143526


namespace max_area_triangle_ABC_l143_143448

theorem max_area_triangle_ABC :
  ∀ (a b x0 y0 : ℝ),
  (3 ≤ y0) →
  (0 < x0) →
  (0 < a) →
  (0 < b) →
  (0 < y0) →
  (x0^2 + y0^2 - 2*y0 ≤ 8) →
  (∃ (AB_length : ℝ), AB_length = abs (a - b) / (y0 - 2) * sqrt (x0^2 + y0^2 - 2*y0)) →
  (∃ (S : ℝ), S = (1/2) * y0 * abs (a - b) / (y0 - 2) * sqrt (x0^2 + y0^2 - 2*y0)) →
    S ≤ 6 * sqrt 2
:= sorry

end max_area_triangle_ABC_l143_143448


namespace cathy_worked_hours_l143_143821

noncomputable def hours_per_week (cathy_hours total_weeks : ℕ) : ℕ :=
  cathy_hours / total_weeks

theorem cathy_worked_hours (cathy_hours total_weeks : ℕ)
  (h_cathy_hours : cathy_hours = 180)
  (h_total_weeks : total_weeks = 9) :
  hours_per_week cathy_hours total_weeks = 20 :=
by {
  rw [hours_per_week, h_cathy_hours, h_total_weeks],
  norm_num,
  sorry
}

end cathy_worked_hours_l143_143821


namespace range_of_a_l143_143237

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (cubic_function x1 = a) ∧ (cubic_function x2 = a) ∧ (cubic_function x3 = a)) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l143_143237


namespace exists_digit_sum_divisible_l143_143172

-- Definitions for the problem
def is_valid_digit_sum (m k : ℕ) : Prop :=
  m.digits.sum = k

def divides (n m : ℕ) : Prop :=
  n ∣ m

theorem exists_digit_sum_divisible (n k : ℕ) (h1 : 3 ∣ n = false) (h2 : k ≥ n) :
  ∃ m : ℕ, is_valid_digit_sum m k ∧ divides n m :=
sorry

end exists_digit_sum_divisible_l143_143172


namespace rationalize_denominator_l143_143691

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143691


namespace g_function_properties_l143_143171

-- Define the function g and the functional equation
theorem g_function_properties (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g(g(x) + y) = g(x + y) + x * g(y) - 2 * x * y - x + 2) :
  let n := 1 in let s := 3 in n * s = 3 := by
  -- Proof omitted
  sorry

end g_function_properties_l143_143171


namespace range_of_a_l143_143469

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + (1 - a) * x

theorem range_of_a (a : ℝ) (h1 : f a 1 = 1) (h2 : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a x| ≤ 1) : 
  a ∈ Set.Icc (-(1/2) : ℝ) 4 :=
begin
  sorry
end

end range_of_a_l143_143469


namespace sum_of_divisors_of_24_eq_60_l143_143295

theorem sum_of_divisors_of_24_eq_60 : 
  (∑ n in { n | n ∣ 24 ∧ 0 < n }.toFinset, n) = 60 := 
sorry

end sum_of_divisors_of_24_eq_60_l143_143295


namespace transformed_function_l143_143460

-- Define the necessary functions and constants
noncomputable def f (x : ℝ) : ℝ := √3 * sin ((π / 2 : ℝ) * x)

theorem transformed_function :
  ∀ (x : ℝ), g x = f (x - (1 / 3 : ℝ)) :=
begin
  intro x,
  simp [f],
  have h1 : (π / 2 : ℝ) * (x - 1 / 3) = (π / 2 : ℝ) * x - π / 6 :=
    by ring,
  rw h1,
  refl,
end

end transformed_function_l143_143460


namespace modulus_of_z_l143_143861

open Complex

theorem modulus_of_z (z : ℂ) (h : z * ⟨0, 1⟩ = ⟨2, 1⟩) : abs z = Real.sqrt 5 :=
by
  sorry

end modulus_of_z_l143_143861


namespace geometric_sequence_s6_s4_l143_143863

section GeometricSequence

variables {a : ℕ → ℝ} {a1 : ℝ} {q : ℝ}
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = a1 * (1 - q^(n + 1)) / (1 - q))
variable (h_ratio : S 4 / S 2 = 3)

theorem geometric_sequence_s6_s4 :
  S 6 / S 4 = 7 / 3 :=
sorry

end GeometricSequence

end geometric_sequence_s6_s4_l143_143863


namespace probability_distances_l143_143051

noncomputable def probability_distance_ge_one (length : ℝ) : ℝ :=
  let vertices := {(0, 0), (0, length), (length, 0), (length, length)}
  let center := (length / 2, length / 2)
  let points := vertices.insert center
  let distances := {dist p q | p ∈ points, q ∈ points, p ≠ q}
  let favorable := distances.filter (λ d, d ≥ length).card
  favorable / distances.card

theorem probability_distances :
  probability_distance_ge_one 1 = 3 / 5 :=
by sorry

end probability_distances_l143_143051


namespace sections_in_orchard_l143_143230

-- Conditions: Farmers harvest 45 sacks from each section daily, 360 sacks are harvested daily
def harvest_sacks_per_section : ℕ := 45
def total_sacks_harvested_daily : ℕ := 360

-- Statement: Prove that the number of sections is 8 given the conditions
theorem sections_in_orchard (h1 : harvest_sacks_per_section = 45) (h2 : total_sacks_harvested_daily = 360) :
  total_sacks_harvested_daily / harvest_sacks_per_section = 8 :=
sorry

end sections_in_orchard_l143_143230


namespace pieces_by_first_team_correct_l143_143422

-- Define the number of pieces required.
def total_pieces : ℕ := 500

-- Define the number of pieces made by the second team.
def pieces_by_second_team : ℕ := 131

-- Define the number of pieces made by the third team.
def pieces_by_third_team : ℕ := 180

-- Define the number of pieces made by the first team.
def pieces_by_first_team : ℕ := total_pieces - (pieces_by_second_team + pieces_by_third_team)

-- Statement to prove
theorem pieces_by_first_team_correct : pieces_by_first_team = 189 := 
by 
  -- Proof to be filled in
  sorry

end pieces_by_first_team_correct_l143_143422


namespace basic_terms_divisible_by_4_l143_143436

def is_basic_term (a : ℤ) (grid : ℕ × ℕ → ℤ) (r c : ℕ) : Prop :=
  a = ∏ i in finset.range r, grid (i, c)

theorem basic_terms_divisible_by_4 (n : ℕ) (grid : fin n.succ.succ.succ.succ → fin n.succ.succ.succ.succ → ℤ)
    (h1 : ∀ i j, grid i j = 1 ∨ grid i j = -1) :
  (∑ σ : equiv.perm (fin n.succ.succ.succ.succ), ∏ i, grid i (σ i)) % 4 = 0 := sorry

end basic_terms_divisible_by_4_l143_143436


namespace find_real_solutions_l143_143410

theorem find_real_solutions (x : ℝ) :
  x^4 + (2 - x)^4 - 4 = 42 ↔ 
  x = 1 + sqrt (-3 + sqrt 124 / 2) ∨ x = 1 - sqrt (-3 + sqrt 124 / 2) :=
by
  sorry

end find_real_solutions_l143_143410


namespace amanda_car_round_trip_time_l143_143803

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end amanda_car_round_trip_time_l143_143803


namespace different_colors_ways_same_color_ways_l143_143738

def bag_A : Finset (ℕ × string) := {(1, "Red")}
def bag_B : Finset (ℕ × string) := {(2, "White1"), (3, "White2")}
def bag_C : Finset (ℕ × string) := {(4, "Yellow1"), (5, "Yellow2"), (6, "Yellow3")}

def balls := bag_A ∪ bag_B ∪ bag_C
def draw_two_balls := (balls ×ˢ balls).filter (λ b, b.fst ≠ b.snd)

def different_colors := draw_two_balls.filter 
  (λ b, (b.fst.snd ≠ b.snd.snd) &&
  ((bag_A ∪ bag_B ∪ bag_C).contains b.fst) &&
  ((bag_A ∪ bag_B ∪ bag_C).contains b.snd))

def same_color := draw_two_balls.filter 
  (λ b, (b.fst.snd = b.snd.snd) &&
  ((bag_A ∪ bag_B ∪ bag_C).contains b.fst) &&
  ((bag_A ∪ bag_B ∪ bag_C).contains b.snd))

theorem different_colors_ways : different_colors.card = 11 := by sorry
  
theorem same_color_ways : same_color.card = 4 := by sorry

end different_colors_ways_same_color_ways_l143_143738


namespace chord_length_of_intercepted_circle_and_line_l143_143397

theorem chord_length_of_intercepted_circle_and_line :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 4 * x + 4 * y + 6 = 0) ∧ (x - y - 5 = 0) →
  ∃ l : ℝ, l = sqrt 6 := sorry

end chord_length_of_intercepted_circle_and_line_l143_143397


namespace inequality_correct_l143_143493

-- Theorem: For all real numbers x and y, if x ≥ y, then x² + y² ≥ 2xy.
theorem inequality_correct (x y : ℝ) (h : x ≥ y) : x^2 + y^2 ≥ 2 * x * y := 
by {
  -- Placeholder for the proof
  sorry
}

end inequality_correct_l143_143493


namespace find_a1_range_l143_143997

noncomputable def arithmetic_sequence (a d : Real) (n : ℕ) : Real :=
  a + (n - 1) * d

/-- 
Let {a_n} be an arithmetic sequence satisfying the given equation,
with common difference d in (-1, 0).
If the sum of the first n terms S_n reaches its maximum value only when n=8,
the range of the first term a_1 is (7π/6, 4π/3).
-/
theorem find_a1_range (a_1 d : ℝ) (h1 : -1 < d ∧ d < 0)
    (h2 : (∃ a d, ∀ (n : ℕ), a_n = arithmetic_sequence a d n)
      (h3 : (∑ i in Finset.range 8 + 1, arithmetic_sequence a_1 d  S_n reaches its maximum value only when n = 8)
  :
    (7 * Real.pi / 6 < a_1) ∧ (a_1 < 4 * Real.pi / 3) :=
begin
  sorry,
end

end find_a1_range_l143_143997


namespace probabilities_equal_l143_143502

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l143_143502


namespace T_n_formula_l143_143969

-- Definitions
def a₁ : ℚ := 1 / 3
def S_n (n : ℕ) : ℚ := n * (a₁ + a₁ * (n - 1)) / 2
def b₁ : ℚ := 1
def b (n : ℕ) : ℚ := 3 ^ (n - 1)
def c (n : ℕ) : ℚ := (n : ℚ) * 3 ^ (n - 2)
def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, c (i + 1)

-- Main statement to prove
theorem T_n_formula (n : ℕ) : 
  T_n n = (2 * n - 1) / 4 * 3 ^ (n - 1) + 1 / 12 :=
sorry

end T_n_formula_l143_143969


namespace chemist_salt_solution_l143_143763

theorem chemist_salt_solution (x : ℝ) 
  (hx : 0.60 * x = 0.20 * (1 + x)) : x = 0.5 :=
sorry

end chemist_salt_solution_l143_143763


namespace percentage_difference_liliane_alice_l143_143984

theorem percentage_difference_liliane_alice :
  let J := 200
  let L := 1.30 * J
  let A := 1.15 * J
  (L - A) / A * 100 = 13.04 :=
by
  sorry

end percentage_difference_liliane_alice_l143_143984


namespace cos_angle_DBC_l143_143244

-- Define the problem conditions and question in Lean
variables (A B C D : Type)
variables [triangle A B C]
variables (radius : ℝ)
variables (AC BC : ℝ)
variables (point_D : Line A B)
variables (dist_D_AC dist_D_BC : ℝ)

-- Specify the conditions
axiom radius_eq_4 : radius = 4
axiom AC_eq_BC : AC = BC
axiom point_D_on_AB : D ∈ point_D
axiom dist_D_AC_eq_11 : dist_D_AC = 11
axiom dist_D_BC_eq_3 : dist_D_BC = 3

-- State the theorem
theorem cos_angle_DBC : cos_angle D B C = 3 / 4 :=
by
  sorry

end cos_angle_DBC_l143_143244


namespace proof_problem_l143_143479

variables {V : Type*} [inner_product_space ℝ V]

variables (a b c : V) 

-- Conditions
def is_unit_vector (v : V) : Prop := ⟪v, v⟫ = 1
def are_mutually_perpendicular_unit_vectors (v w : V) : Prop := is_unit_vector v ∧ is_unit_vector w ∧ ⟪v, w⟫ = 0
def satisfies_dot_product_conditions (c a b : V) : Prop := ⟪c, a⟫ = -1 ∧ ⟪c, b⟫ = -1

-- Theorem to prove
theorem proof_problem 
  (h1 : are_mutually_perpendicular_unit_vectors a b)
  (h2 : satisfies_dot_product_conditions c a b) :
  ⟪3 • a - b + 5 • c, b⟫ = -6 :=
sorry

end proof_problem_l143_143479


namespace rootiful_set_eq_integers_l143_143753

def T : Set ℤ := { n | ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = 2^a - 2^b }

def rootiful (S : Set ℤ) : Prop :=
  ∀ (n : ℕ) (a : Fin (n + 1) → ℤ), (∀ x : ℤ, (Finset.range (n + 1)).sum (λ i, a i * x^i) = 0 → x ∈ S)

theorem rootiful_set_eq_integers (S : Set ℤ) (h1 : ∀ n : ℕ, a : Fin (n + 1) → ℤ, (∀ x : ℤ, (Finset.range (n + 1)).sum (λ i, a i * x^i) = 0 → x ∈ S))
  (h2 : T ⊆ S) : S = Set.univ :=
sorry

end rootiful_set_eq_integers_l143_143753


namespace a_plus_b_l143_143994

-- Define the sets A and B according to the given conditions
def A : Set ℝ := {x | x^2 - 2x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

-- Given the conditions A ∪ B = ℝ and A ∩ B = (3, 4], prove a + b = -7
theorem a_plus_b (a b : ℝ) (h1 : A ∪ B a b = Set.univ) (h2 : A ∩ B a b = { x : ℝ | 3 < x ∧ x ≤ 4 }) :
  a + b = -7 :=
by
  sorry

end a_plus_b_l143_143994


namespace find_wind_speed_l143_143759

-- Definitions from conditions
def speed_with_wind (j w : ℝ) := (j + w) * 6 = 3000
def speed_against_wind (j w : ℝ) := (j - w) * 9 = 3000

-- Theorem to prove the wind speed is 83.335 mph
theorem find_wind_speed (j w : ℝ) (h1 : speed_with_wind j w) (h2 : speed_against_wind j w) : w = 83.335 :=
by 
  -- Here we would prove the theorem using the given conditions
  sorry

end find_wind_speed_l143_143759


namespace carpet_needed_for_room_l143_143349

theorem carpet_needed_for_room
  (length_feet : ℕ) (width_feet : ℕ)
  (area_conversion_factor : ℕ)
  (length_given : length_feet = 12)
  (width_given : width_feet = 6)
  (conversion_given : area_conversion_factor = 9) :
  (length_feet * width_feet) / area_conversion_factor = 8 := 
by
  sorry

end carpet_needed_for_room_l143_143349


namespace glucose_solution_volume_l143_143783

theorem glucose_solution_volume
  (h1 : 6.75 / 45 = 15 / x) :
  x = 100 :=
by
  sorry

end glucose_solution_volume_l143_143783


namespace dot_product_equivalence_l143_143933

variables {u v : ℝ^3}

-- Define the conditions
def norm_u : ℝ := 5
def norm_v : ℝ := 2

-- State the theorem to be proved
theorem dot_product_equivalence (hu : ‖u‖ = norm_u) (hv : ‖v‖ = norm_v) : 
  (u + v) • (u - v) = 21 :=
by
  sorry

end dot_product_equivalence_l143_143933


namespace division_of_polynomials_l143_143018

def polynomial_division (p q : Polynomial ℚ) (q_ne_zero : q ≠ 0) : Polynomial ℚ × Polynomial ℚ :=
  p.divModByMonic q

theorem division_of_polynomials : 
  polynomial_division (5 * (X ^ 3) - 4 * (X ^ 2) + 6 * X - 9) (X - 1) (Polynomial.X_sub_C_ne_zero 1) = (5 * (X ^ 2) + X + 7, -2) :=
by
  sorry

end division_of_polynomials_l143_143018


namespace cos_neg_eq_cos_l143_143035

theorem cos_neg_eq_cos (theta : ℝ) : cos (-theta) = cos theta := 
begin
  exact cos_neg theta,
end

end cos_neg_eq_cos_l143_143035


namespace number_of_hours_sold_l143_143369

def packs_per_hour_peak := 6
def packs_per_hour_low := 4
def price_per_pack := 60
def extra_revenue := 1800

def revenue_per_hour_peak := packs_per_hour_peak * price_per_pack
def revenue_per_hour_low := packs_per_hour_low * price_per_pack
def revenue_diff_per_hour := revenue_per_hour_peak - revenue_per_hour_low

theorem number_of_hours_sold (h : ℕ) 
  (h_eq : revenue_diff_per_hour * h = extra_revenue) : 
  h = 15 :=
by
  -- skip proof
  sorry

end number_of_hours_sold_l143_143369


namespace max_fraction_l143_143443

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  1 + y / x ≤ -2 :=
sorry

end max_fraction_l143_143443


namespace rationalize_sqrt_fraction_l143_143671

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143671


namespace percentage_increase_in_surface_area_l143_143768

def cube_surface_area (L : ℝ) : ℝ := 6 * L^2

theorem percentage_increase_in_surface_area (L : ℝ) (hL : L > 0) : 
  let new_edge_length := 1.60 * L in
  let original_surface_area := cube_surface_area L in
  let new_surface_area := cube_surface_area new_edge_length in
  let increase_in_area := new_surface_area - original_surface_area in
  let percentage_increase := (increase_in_area / original_surface_area) * 100 in
  percentage_increase = 156 :=
by
  -- Proceed with the proof
  sorry

end percentage_increase_in_surface_area_l143_143768


namespace solve_for_x_l143_143713

theorem solve_for_x (x : ℝ) : (∛(3 - 1 / x^2) = -4) → (x = 1 / Real.sqrt 67 ∨ x = -1 / Real.sqrt 67) :=
by
  sorry

end solve_for_x_l143_143713


namespace pq_aq_ratio_l143_143943

-- Define the conditions for the perpendicular diameters and angle
def Circle (Q : Type) := ∃ (O A B C D P : Q), 
  (is_center O ∧ is_diameter A B ∧ is_diameter C D ∧ is_perpendicular AB CD ∧ (is_on_line P AQ) ∧ 
   angle Q P C = 45)

-- Define the lengths and ratio
def ratio_PQ_AQ (Q : Type) [Circle Q] : Q :=
  let ⟨O, A, B, C, D, P, hO, hAB, hCD, h_perp, hP_on_AQ, h_ang⟩ := Circle.existence Q in
  ∃ R PQ AQ : ℝ, (R = radius Q) ∧ (PQ = segment P Q) ∧ (AQ = segment A Q) ∧ (PQ / AQ = sqrt 2 / 2)

-- The main theorem to be proven
theorem pq_aq_ratio (Q : Type) [Circle Q] : 
  ratio_PQ_AQ Q :=
sorry

end pq_aq_ratio_l143_143943


namespace orthocenter_incenter_equilateral_triangle_iff_l143_143954

-- Given conditions
variables {A B C D E F O : Point}
variables (circumcircle : Circle) (triangleABC : Triangle)
variables (angle_bisector_A : Line) (angle_bisector_B : Line) (angle_bisector_C : Line)

-- Theorems to be proven
theorem orthocenter_incenter (hD : D ∈ circumcircle ∧ D ∈ (angle_bisector_A ∩ circumcircle))
                               (hE : E ∈ circumcircle ∧ E ∈ (angle_bisector_B ∩ circumcircle))
                               (hF : F ∈ circumcircle ∧ F ∈ (angle_bisector_C ∩ circumcircle)) :
  orthocenter (Triangle.mk D E F) = incenter (Triangle.mk A B C) :=
sorry

theorem equilateral_triangle_iff (hD : D ∈ circumcircle ∧ D ∈ (angle_bisector_A ∩ circumcircle))
                                 (hE : E ∈ circumcircle ∧ E ∈ (angle_bisector_B ∩ circumcircle))
                                 (hF : F ∈ circumcircle ∧ F ∈ (angle_bisector_C ∩ circumcircle))
                                 (vector_sum_condition : (vector (AD) + vector (BE) + vector (CF) = 0)) :
  (Triangle.mk A B C).is_equilateral :=
sorry

end orthocenter_incenter_equilateral_triangle_iff_l143_143954


namespace minimum_notes_to_determine_prize_location_l143_143735

/--
There are 100 boxes, numbered from 1 to 100. A prize is hidden in one of the boxes, 
and the host knows its location. The viewer can send the host a batch of notes 
with questions that require a "yes" or "no" answer. The host shuffles the notes 
in the batch and, without announcing the questions aloud, honestly answers 
all of them. Prove that the minimum number of notes that need to be sent to 
definitely determine where the prize is located is 99.
-/
theorem minimum_notes_to_determine_prize_location : 
  ∀ (boxes : Fin 100 → Prop) (prize_location : ∃ i : Fin 100, boxes i) 
    (batch_size : Nat), 
  (batch_size + 1) ≥ 100 → batch_size = 99 :=
by
  sorry

end minimum_notes_to_determine_prize_location_l143_143735


namespace minimum_cells_25_l143_143728

/--
Given a plane divided into unit cells, each cell painted in one of two colors, 
prove that the minimum possible number of cells in a figure consisting of entire cells 
which contains each of the 16 possible colored 2x2 squares is 25.
-/
theorem minimum_cells_25 (cells : ℕ) : 
  ∃ (figure : ℕ), (∃ (colors : ℕ → ℕ → Prop), colors = (λ x y, x < 2 ∧ y < 2)) ∧
  (∀ (square : ℕ → ℕ → bool), (cells >= 25) → figure = cells) :=
sorry

end minimum_cells_25_l143_143728


namespace relative_frequency_defective_books_l143_143243

theorem relative_frequency_defective_books 
  (N_defective : ℤ) (N_total : ℤ)
  (h_defective : N_defective = 5)
  (h_total : N_total = 100) :
  (N_defective : ℚ) / N_total = 0.05 := by
  sorry

end relative_frequency_defective_books_l143_143243


namespace projection_of_a_onto_b_l143_143087

variables (a b : Vector ℝ)
variables (norm_a : ‖a‖ = 3)
variables (norm_b : ‖b‖ = 4)
variables (cos_theta : inner a b = (3/4) * 3 * 4)

theorem projection_of_a_onto_b :
  proj b a = (9 / 16) • b := by
  sorry

end projection_of_a_onto_b_l143_143087


namespace complex_number_properties_l143_143043

theorem complex_number_properties (z : ℂ) (h : z^2 = 3 + 4 * Complex.I) : 
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_number_properties_l143_143043


namespace intersection_P_Q_l143_143931

def P : Set ℤ := { x | Int.abs (x - 1) < 2 }
def Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_P_Q : P ∩ Q = {0, 1, 2} :=
by
  sorry

end intersection_P_Q_l143_143931


namespace sum_of_divisors_of_24_l143_143282

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range (24 + 1))), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l143_143282


namespace history_book_cost_l143_143751

theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) (history_book_cost : ℕ) : 
  total_books = 90 ∧ math_book_cost = 4 ∧ total_price = 390 ∧ math_books = 60 →
  history_book_cost = 5 :=
by
  intro h,
  cases h with ht h1,
  cases h1 with hmc h2,
  cases h2 with tp hg,
  sorry

end history_book_cost_l143_143751


namespace count_incorrect_statements_l143_143362

-- Definitions of the statements as conditions in the problem
def stmt1 : Prop := ¬(∅ ∈ ({1, 2, 3} : Set ℕ))
def stmt2 : Prop := ∅ ⊆ ({0} : Set ℕ)
def stmt3 : Prop := ({0, 1, 2} : Set ℕ) ⊆ ({1, 2, 0} : Set ℕ)
def stmt4 : Prop := ¬(0 ∈ (∅ : Set ℕ))
def stmt5 : Prop := (0 : Set ℕ).inter ∅ = ∅

-- Definition of the total number of incorrect statements
def number_of_incorrect_statements : ℕ :=
  let incorrect_statements := [stmt1, stmt4, stmt5] |>.count(λ stmt => ¬stmt)
  incorrect_statements

-- The main theorem to prove
theorem count_incorrect_statements : number_of_incorrect_statements = 3 := by
  sorry

end count_incorrect_statements_l143_143362


namespace arithmetic_sequence_terms_l143_143913

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a 0 + a 1 + a 2 = 4) 
  (h2 : a (n-3) + a (n-2) + a (n-1) = 7) 
  (h3 : (n * (a 0 + a (n-1)) / 2) = 22) : 
  n = 12 :=
sorry

end arithmetic_sequence_terms_l143_143913


namespace total_worth_of_stock_l143_143131

theorem total_worth_of_stock (x y : ℕ) (cheap_cost expensive_cost : ℝ) 
  (h1 : y = 21) (h2 : x + y = 22)
  (h3 : expensive_cost = 10) (h4 : cheap_cost = 2.5) :
  (x * expensive_cost + y * cheap_cost) = 62.5 :=
by
  sorry

end total_worth_of_stock_l143_143131


namespace sum_of_divisors_of_24_l143_143289

theorem sum_of_divisors_of_24 : ∑ d in Finset.filter (λ n, 24 % n = 0) (Finset.range 25), d = 60 := by
  sorry

end sum_of_divisors_of_24_l143_143289


namespace dave_pieces_l143_143392

theorem dave_pieces (boxes_bought : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) 
  (h₁ : boxes_bought = 12) (h₂ : boxes_given = 5) (h₃ : pieces_per_box = 3) : 
  boxes_bought - boxes_given * pieces_per_box = 21 :=
by
  sorry

end dave_pieces_l143_143392


namespace relationship_among_a_b_and_ab_l143_143897

noncomputable def a : ℝ := Real.log 0.4 / Real.log 0.2
noncomputable def b : ℝ := 1 - (1 / (Real.log 4 / Real.log 10))

theorem relationship_among_a_b_and_ab : a * b < a + b ∧ a + b < 0 := by
  sorry

end relationship_among_a_b_and_ab_l143_143897


namespace median_line_equation_altitude_line_equation_l143_143965

-- Define the coordinates of the vertices of the triangle
def A : ℝ × ℝ := (2, -2)
def B : ℝ × ℝ := (6, 6)
def C : ℝ × ℝ := (0, 6)

-- Part (I): Proving the equation of the median line CM
theorem median_line_equation :
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  M = (4, 2) →
  ∀ x y : ℝ, (y - 2) = -1 * (x - 4) → x + y - 6 = 0 :=
sorry

-- Part (II): Proving the equation of the altitude on side AB
theorem altitude_line_equation :
  let m_AB : ℝ := (B.2 - A.2) / (B.1 - A.1) in
  m_AB = 2 →
  let m_altitude : ℝ := -1 / m_AB in
  m_altitude = -1/2 →
  ∀ x y : ℝ, (y - C.2) = -1/2 * x → x + 2*y - 12 = 0 :=
sorry

end median_line_equation_altitude_line_equation_l143_143965


namespace sum_divisors_24_l143_143277

theorem sum_divisors_24 :
  (∑ n in Finset.filter (λ n => 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_divisors_24_l143_143277


namespace thirty_percent_less_is_one_fourth_more_l143_143255

-- Define thirty percent less than 100
def thirty_percent_less_of_100 : ℝ := 100 - 0.30 * 100

-- Define one-fourth more than a number
def one_fourth_more (n : ℝ) : ℝ := (5 / 4) * n

-- The main statement of the problem
theorem thirty_percent_less_is_one_fourth_more :
  (∃ n : ℝ, one_fourth_more n = thirty_percent_less_of_100) ↔ ∃ n : ℝ, n = 56 :=
by
  sorry

end thirty_percent_less_is_one_fourth_more_l143_143255


namespace sum_of_cubes_of_nonneg_rationals_l143_143155

theorem sum_of_cubes_of_nonneg_rationals (n : ℤ) (h1 : n > 1) (h2 : ∃ a b : ℚ, a^3 + b^3 = n) :
  ∃ c d : ℚ, c ≥ 0 ∧ d ≥ 0 ∧ c^3 + d^3 = n :=
sorry

end sum_of_cubes_of_nonneg_rationals_l143_143155


namespace rationalize_denominator_l143_143695

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143695


namespace min_fraction_sum_is_15_l143_143181

theorem min_fraction_sum_is_15
  (A B C D : ℕ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_nonzero_int : ∃ k : ℤ, k ≠ 0 ∧ (A + B : ℤ) = k * (C + D))
  : C + D = 15 :=
sorry

end min_fraction_sum_is_15_l143_143181


namespace paul_spent_on_rackets_l143_143197

theorem paul_spent_on_rackets :
  let full_price : ℝ := 60
  let first_racket_discount : ℝ := 0.20 * full_price
  let second_racket_discount : ℝ := 0.50 * full_price
  let first_racket_price : ℝ := full_price - first_racket_discount
  let second_racket_price : ℝ := full_price - second_racket_discount
  first_racket_price + second_racket_price = 78 :=
begin
  sorry
end

end paul_spent_on_rackets_l143_143197


namespace tom_profit_calculation_l143_143262

theorem tom_profit_calculation :
  let flour_needed := 500
  let flour_per_bag := 50
  let flour_bag_cost := 20
  let salt_needed := 10
  let salt_cost_per_pound := 0.2
  let promotion_cost := 1000
  let tickets_sold := 500
  let ticket_price := 20

  let flour_bags := flour_needed / flour_per_bag
  let cost_flour := flour_bags * flour_bag_cost
  let cost_salt := salt_needed * salt_cost_per_pound
  let total_expenses := cost_flour + cost_salt + promotion_cost
  let total_revenue := tickets_sold * ticket_price

  let profit := total_revenue - total_expenses

  profit = 8798 := by
  sorry

end tom_profit_calculation_l143_143262


namespace find_line_properties_l143_143791

-- Define the points P and Q
def P : ℝ × ℝ := (2, 0)
def Q : ℝ × ℝ := (-2, 4 * Real.sqrt 3 / 3)

-- Distance function from a point to a line
def distToLine (a b c : ℝ) (x y : ℝ) : ℝ :=
  Real.abs (a * x + b * y + c) / Real.sqrt (a ^ 2 + b ^ 2)

-- Condition: distance from Q to the line is 4, line passes through P
def validLine (a b c : ℝ) : Prop :=
  distToLine a b c (fst Q) (snd Q) = 4 ∧ a * fst P + b * snd P + c = 0

-- Proof goal: The line should have specific angle of inclination and equation
theorem find_line_properties (a b c : ℝ) (h : validLine a b c) :
  (b = 0 ∧ a = 1 ∧ c = -2) ∨
  (a = 1 ∧ b = -Real.sqrt 3 ∧ c = -2) := by
  sorry

end find_line_properties_l143_143791


namespace sqrt_pos_neg_l143_143407

noncomputable def sqrt_16 : ℝ := Real.sqrt 16

theorem sqrt_pos_neg (h : sqrt_16 = 4) : (± sqrt_16) = (± 4) :=
sorry

end sqrt_pos_neg_l143_143407


namespace amusement_park_ticket_cost_l143_143335

theorem amusement_park_ticket_cost (T_adult T_child : ℕ) (num_children num_adults : ℕ) 
  (h1 : T_adult = 15) (h2 : T_child = 8) 
  (h3 : num_children = 15) (h4 : num_adults = 25 + num_children) :
  num_adults * T_adult + num_children * T_child = 720 :=
by
  sorry

end amusement_park_ticket_cost_l143_143335


namespace total_team_formation_plans_l143_143423

def numberOfWaysToChooseDoctors (m f : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose m (k - 1) * Nat.choose f 1) +
  (Nat.choose m 1 * Nat.choose f (k - 1))

theorem total_team_formation_plans :
  let m := 5
  let f := 4
  let total := 3
  numberOfWaysToChooseDoctors m f total = 70 :=
by
  let m := 5
  let f := 4
  let total := 3
  unfold numberOfWaysToChooseDoctors
  sorry

end total_team_formation_plans_l143_143423


namespace six_digit_special_number_exists_l143_143541

theorem six_digit_special_number_exists :
  ∃ n : ℕ, n = 924741 ∧
            (∃ a0 a1 a2 a3 a4 a5 : ℕ,
               n = a5 * 10^5 + a4 * 10^4 + a3 * 10^3 + a2 * 10^2 + a1 * 10 + a0 ∧
               0 ≤ a0 ∧ a0 < 10 ∧ 0 ≤ a1 ∧ a1 < 10 ∧ 0 ≤ a2 ∧ a2 < 10 ∧
               0 ≤ a3 ∧ a3 < 10 ∧ 0 ≤ a4 ∧ a4 < 10 ∧ 0 ≤ a5 ∧ a5 < 10 ∧ 
               (a0 = 7 ∨ a1 = 7 ∨ a2 = 7 ∨ a3 = 7 ∨ a4 = 7 ∨ a5 = 7) ∧
               (n % 9 = 0) ∧
               (∀ (i j : ℕ), i ≠ j ∧ i < 6 ∧ j < i →
                 let new_n := a0 * 10^(if 0=i then j else if 0=j then i else 0) +
                               a1 * 10^(if 1=i then j else if 1=j then i else 1) +
                               a2 * 10^(if 2=i then j else if 2=j then i else 2) +
                               a3 * 10^(if 3=i then j else if 3=j then i else 3) +
                               a4 * 10^(if 4=i then j else if 4=j then i else 4) +
                               a5 * 10^(if 5=i then j else if 5=j then i else 5)
                 in (abs (n - new_n) % 2525 = 0 ∨ abs (n - new_n) % 2168 = 0 ∨
                     abs (n - new_n) % 4375 = 0 ∨ abs (n - new_n) % 6875 = 0)))
:= sorry

end six_digit_special_number_exists_l143_143541


namespace sum_of_cube_faces_l143_143826

theorem sum_of_cube_faces :
  ∃ (a b c d e f : ℕ), 
    (a = 12) ∧ 
    (b = a + 3) ∧ 
    (c = b + 3) ∧ 
    (d = c + 3) ∧ 
    (e = d + 3) ∧ 
    (f = e + 3) ∧ 
    (a + f = 39) ∧ 
    (b + e = 39) ∧ 
    (c + d = 39) ∧ 
    (a + b + c + d + e + f = 117) :=
by
  let a := 12
  let b := a + 3
  let c := b + 3
  let d := c + 3
  let e := d + 3
  let f := e + 3
  have h1 : a + f = 39 := sorry
  have h2 : b + e = 39 := sorry
  have h3 : c + d = 39 := sorry
  have sum : a + b + c + d + e + f = 117 := sorry
  exact ⟨a, b, c, d, e, f, rfl, rfl, rfl, rfl, rfl, rfl, h1, h2, h3, sum⟩

end sum_of_cube_faces_l143_143826


namespace range_of_a_l143_143106

variable (a x : Real)

theorem range_of_a (h1 : sin x ^ 2 + 3 * a ^ 2 * cos x - 2 * a ^ 2 * (3 * a - 2) - 1 = 0) :
  -1 ≤ cos x ∧ cos x ≤ 1 → -1 / 2 ≤ a ∧ a ≤ 1 := by
sorry

end range_of_a_l143_143106


namespace number_of_x0_values_l143_143041

noncomputable def x_seq (x0 : ℝ) : ℕ → ℝ
| 0     := x0
| (n+1) := if 2 * (x_seq n) < 1 then 2 * (x_seq n) else 2 * (x_seq n) - 1

theorem number_of_x0_values : 
  (∃ (xs : set ℝ) (h : ∀ x0 ∈ xs, 0 ≤ x0 ∧ x0 < 1 ∧ x_seq x0 5 = x0), xs.countable ∧ xs.finite ∧ xs.size = 31) :=
sorry

end number_of_x0_values_l143_143041


namespace find_x_l143_143875

open Nat

def has_three_distinct_prime_factors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a * b * c

theorem find_x (n : ℕ) (h₁ : Odd n) (h₂ : 7^n + 1 = x)
  (h₃ : has_three_distinct_prime_factors x) (h₄ : 11 ∣ x) : x = 16808 := by
  sorry

end find_x_l143_143875


namespace petya_second_race_finishes_first_l143_143199

variable (t v_P v_V : ℝ)
variable (h1 : v_P * t = 100)
variable (h2 : v_V * t = 90)
variable (d : ℝ)

theorem petya_second_race_finishes_first :
  v_V = 0.9 * v_P ∧
  d * v_P = 10 + d * (0.9 * v_P) →
  ∃ t2 : ℝ, t2 = 100 / v_P ∧ (v_V * t2 = 90) →
  ∃ t3 : ℝ, t3 = t2 + d / 10 ∧ (d * v_P = 100) →
  v_P * d / 10 - v_V * d / 10 = 1 :=
by
  sorry

end petya_second_race_finishes_first_l143_143199


namespace rationalize_denominator_l143_143658

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143658


namespace sin_double_angle_is_correct_l143_143103

theorem sin_double_angle_is_correct (θ : ℝ) 
  (h1 : cos θ + sin θ = 7 / 5)
  (h2 : cos θ - sin θ = 1 / 5) : 
  sin (2 * θ) = 48 / 25 :=
by sorry

end sin_double_angle_is_correct_l143_143103


namespace bridge_length_is_correct_l143_143765

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let total_distance := speed_ms * time_sec
  total_distance - length_of_train

theorem bridge_length_is_correct :
  length_of_bridge 160 45 30 = 215 := by
  sorry

end bridge_length_is_correct_l143_143765


namespace concentration_sequences_and_min_operations_l143_143778

theorem concentration_sequences_and_min_operations :
  (a_1 = 1.55 ∧ b_1 = 0.65) ∧
  (∀ n ≥ 1, a_n - b_n = 0.9 * (1 / 2)^(n - 1)) ∧
  (∃ n, 0.9 * (1 / 2)^(n - 1) < 0.01 ∧ n = 8) :=
by
  sorry

end concentration_sequences_and_min_operations_l143_143778


namespace find_B_angle_l143_143977

noncomputable def triangle_AB_alphas : Type _ :=
  {b : ℝ | b = sqrt 2}

noncomputable def triangle_BC_betas : Type _ :=
  {c : ℝ | c = sqrt 3}

theorem find_B_angle
  (A : Real)
  (alpha : triangle_AB_alphas)
  (beta : triangle_BC_betas) :
  A = Real.pi / 4 ∧ alpha = sqrt 2 ∧ beta = sqrt 3 → 
  sin (Real.to_rad 45) * sqrt 3 / sqrt 2 = sqrt 3 / 2 ∨ 
  1 / 2 = sqrt 3 / 2 :=
sorry

end find_B_angle_l143_143977


namespace correct_sampling_methods_l143_143207

theorem correct_sampling_methods :
  (let num_balls := 1000
   let red_box := 500
   let blue_box := 200
   let yellow_box := 300
   let sample_balls := 100
   let num_students := 20
   let selected_students := 3
   let q1_method := "stratified"
   let q2_method := "simple_random"
   q1_method = "stratified" ∧ q2_method = "simple_random") := sorry

end correct_sampling_methods_l143_143207


namespace sum_absolute_difference_eq_n_squared_l143_143835

namespace myproof

theorem sum_absolute_difference_eq_n_squared (n : ℕ) (a b : Fin n → ℕ) (h₁ : ∀ i j, i < j → a i < a j) (h₂ : ∀ i j, i < j → b i > b j) (h₃ : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n) (h₄ : ∀ i, 1 ≤ b i ∧ b i ≤ 2 * n) (h₅ : ∀ i j, a i ≠ a j ∧ b i ≠ b j) (h₆ : ∀ i j, a i ≠ b j) :
  (Finset.univ.sum (λ i, abs (a i - b i))) = n ^ 2 := 
begin
  sorry
end

end myproof

end sum_absolute_difference_eq_n_squared_l143_143835


namespace triangles_even_l143_143050

def point : Type := ℝ × ℝ

def no_three_collinear (S : set point) : Prop :=
  ∀ P1 P2 P3 ∈ S, P1 ≠ P2 → P2 ≠ P3 → P1 ≠ P3 →
  ¬ collinear {P1, P2, P3}

def count_triangles (S : set point) (P : point) : ℕ :=
  set.card {T | ∃ x y ∈ S, x ≠ y ∧ x ≠ P ∧ y ≠ P ∧ T = {P, x, y}}

theorem triangles_even (S : set point) (hS : S.card = 9 ∧ no_three_collinear S) :
  ∀ P ∈ S, even (count_triangles S P) :=
begin
  sorry
end

end triangles_even_l143_143050


namespace sin_expression_eq_neg_half_l143_143214

theorem sin_expression_eq_neg_half : 
  sin (119 : ℝ) * sin (181 : ℝ) - sin (91 : ℝ) * sin (29 : ℝ) = -1 / 2 :=
sorry

end sin_expression_eq_neg_half_l143_143214


namespace beaver_stores_60_carrots_l143_143420

theorem beaver_stores_60_carrots (b r : ℕ) (h1 : 4 * b = 5 * r) (h2 : b = r + 3) : 4 * b = 60 :=
by
  sorry

end beaver_stores_60_carrots_l143_143420


namespace john_newspaper_percentage_less_l143_143148

theorem john_newspaper_percentage_less
  (total_newspapers : ℕ)
  (selling_price : ℝ)
  (percentage_sold : ℝ)
  (profit : ℝ)
  (total_cost : ℝ)
  (cost_per_newspaper : ℝ)
  (percentage_less : ℝ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : percentage_sold = 0.80)
  (h4 : profit = 550)
  (h5 : total_cost = 800 - profit)
  (h6 : cost_per_newspaper = total_cost / total_newspapers)
  (h7 : percentage_less = ((selling_price - cost_per_newspaper) / selling_price) * 100) :
  percentage_less = 75 :=
by
  sorry

end john_newspaper_percentage_less_l143_143148


namespace distance_zero_l143_143888

-- Define points A and B
def A := (-2 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 2 : ℝ)

-- Assuming point M satisfies the condition that vectors AM and MB are orthogonal
def orthogonal_condition (M : ℝ × ℝ) : Prop :=
  let (m, n) := M
  (m + 2) * (-m) + n * (2 - n) = 0

-- Distance from any point to a line
def point_to_line_distance (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := P
  (abs (a * x + b * y + c)) / (sqrt (a * a + b * b))

-- Proof statement
theorem distance_zero (M : ℝ × ℝ) (hM : orthogonal_condition M) :
  point_to_line_distance M 1 (-1) (-2) = 0 :=
sorry

end distance_zero_l143_143888


namespace apple_tree_yield_l143_143586

theorem apple_tree_yield (A : ℝ) 
    (h1 : Magdalena_picks_day1 = A / 5)
    (h2 : Magdalena_picks_day2 = 2 * (A / 5))
    (h3 : Magdalena_picks_day3 = (A / 5) + 20)
    (h4 : remaining_apples = 20)
    (total_picked : Magdalena_picks_day1 + Magdalena_picks_day2 + Magdalena_picks_day3 + remaining_apples = A)
    : A = 200 :=
by
    sorry

end apple_tree_yield_l143_143586


namespace abs_ineq_solution_range_l143_143951

theorem abs_ineq_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 :=
by
  sorry

end abs_ineq_solution_range_l143_143951


namespace rhombus_area_l143_143960

theorem rhombus_area (A B C D : ℝ × ℝ)
  (hA : A = (0, 5.5)) (hB : B = (8, 0)) (hC : C = (0, -5.5)) (hD : D = (-8, 0)) :
  let d1 := dist A C,
      d2 := dist B D
  in (d1 * d2) / 2 = 88 :=
by
  sorry

end rhombus_area_l143_143960


namespace locus_G_eq_P_on_fixed_circle_l143_143045

open Real 

structure Point := (x : ℝ) (y : ℝ)

def is_on_locus_G (G : Point) : Prop := 
  G.x^2 / 16 + G.y^2 = 1

def perpendicular_tangents_condition (P : Point) (C : (Point → Prop)) : Prop :=
  (∀ k₁ k₂ : ℝ, C (mk (P.x + k₁ * P.y) (P.y - k₁ * P.x)) ∧ C (mk (P.x + k₂ * P.y) (P.y - k₂ * P.x)) → 
    k₁ * k₂ = -1) → 
    P.x^2 + P.y^2 = 17

theorem locus_G_eq :
  ∀ (A B G : Point), 
    (dist A B = 4) 
    ∧ (A.x + 2 * A.y = 0 ∨ A.x = -2 * A.y )
    ∧ (B.x + 2 * B.y = 0 ∨ B.x = -2 * B.y )
    ∧ (G.x = (A.x + B.x) / 2 ∧ G.y = (A.y + B.y) / 2) → 
    is_on_locus_G G
:= 
  sorry

theorem P_on_fixed_circle :
  ∀ (P : Point), 
    (perpendicular_tangents_condition P (λ Q, Q.x^2 / 16 + Q.y^2 = 1)) → 
    P.x^2 + P.y^2 = 17
:= 
  sorry

end locus_G_eq_P_on_fixed_circle_l143_143045


namespace increase_coefficients_l143_143983

variables (x y z k1 k2 : ℝ)
variable h1 : x ≠ 0
variable h2 : y ≠ 0
variable h3 : k1 ≠ k2

-- Conditions
variable h4 : y - z = k1 * x
variable h5 : z - x = k2 * y
variable h6 : z = 3 * (x - y)

theorem increase_coefficients (h4 : y - z = k1 * x) (h5 : z - x = k2 * y) (h6 : z = 3 * (x - y))
 (h3 : k1 ≠ k2) (h1 : x ≠ 0) (h2 : y ≠ 0) :
  (k1 + 3) * (k2 + 3) = 8 := 
by 
  sorry

end increase_coefficients_l143_143983


namespace transformed_system_solution_l143_143116

theorem transformed_system_solution 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * 3 + b1 * 4 = c1)
  (h2 : a2 * 3 + b2 * 4 = c2) :
  (3 * a1 * 5 + 4 * b1 * 5 = 5 * c1) ∧ (3 * a2 * 5 + 4 * b2 * 5 = 5 * c2) :=
by 
  sorry

end transformed_system_solution_l143_143116


namespace min_value_expression_l143_143097

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 3) :
  ∃ x : ℝ, (x = (a^2 + b^2 + 22) / (a + b)) ∧ (x = 8) :=
by
  sorry

end min_value_expression_l143_143097


namespace selection_methods_at_least_one_girl_l143_143780

/-
We are given a total of 6 people (4 boys and 2 girls), and we want to select 4 of them such that there is at least one girl in the selection.
We need to prove that the number of different selection methods is 14.
-/

theorem selection_methods_at_least_one_girl :
  let boys := 4
  let girls := 2
  let total_people := 6
  ∃ n : ℕ, (n = 14) ↔ ∃ (select4 : finset (finset fin_ℕ)),
    select4.card = 4 ∧ 
    (∃ g : fin_ℕ, g < girls → ∃ s ∈ select4, s ⊆ (finset.range total_people) ∧ g ∈ s) :=
sorry

end selection_methods_at_least_one_girl_l143_143780


namespace solution_of_equation_l143_143160

-- Definitions based on the conditions
def floor (x : ℝ) : ℤ := Real.floor x
def frac_part (x : ℝ) : ℝ := x - (floor x)

-- The main theorem to prove
theorem solution_of_equation (x : ℝ) 
  (h_eq : (floor x)^4 + (frac_part x)^4 + x^4 = 2048) :
  x = -3 - Real.sqrt 5 := 
sorry

end solution_of_equation_l143_143160


namespace min_value_of_expression_l143_143570

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 
  ∃ m : ℝ, (∀ y : ℝ, 0 < y → 3 * y^4 + 6 * y^(-3) ≥ m) ∧ m = 9 :=
by
  sorry

end min_value_of_expression_l143_143570


namespace find_a_b_and_intervals_and_extremes_l143_143458

-- Define the function
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 + (a - 1) * x ^ 2 + 48 * (a - 2) * x + b

-- Asserts symmetry about the origin implies f is odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- State the theorem for the given conditions
theorem find_a_b_and_intervals_and_extremes (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = a * x ^ 3 + (a - 1) * x ^ 2 + 48 * (a - 2) * x + b) →
  is_odd_function f →
  a = 1 ∧ b = 0 ∧
  -- Redefine the updated function with a == 1 and b == 0
  let f (x : ℝ) := x ^ 3 - 48 * x in
  (∀ x, f' x = 3 * x ^ 2 - 48) ∧
  (∃ x1 x2, f' x1 = 0 ∧ x1 = -4 ∧ f' x2 = 0 ∧ x2 = 4) ∧
  (∀ x, if -4 < x ∧ x < 4 then f' x < 0 else if x < -4 ∨ 4 < x then f' x > 0 else True) ∧
  (f (-4) = 128 ∧ f (4) = -128) := sorry

end find_a_b_and_intervals_and_extremes_l143_143458


namespace Luke_carrying_capacity_l143_143185

theorem Luke_carrying_capacity
  (trays_table1 : ℕ) (trays_table2 : ℕ) (total_trays : ℕ) (trips : ℕ) 
  (h1 : trays_table1 = 20) (h2 : trays_table2 = 16) (h3 : total_trays = trays_table1 + trays_table2) (h4 : trips = 9) :
  (total_trays / trips) = 4 :=
by
  have h5 : 20 + 16 = 36 := by norm_num
  rw [h1, h2] at h3
  rw h5 at h3
  have h6 : 36 = total_trays := by exact h3
  rw h6
  rw h4
  norm_num

end Luke_carrying_capacity_l143_143185


namespace restaurant_meal_cost_l143_143368

def cost_of_group_meal (total_people : Nat) (kids : Nat) (adult_meal_cost : Nat) : Nat :=
  let adults := total_people - kids
  adults * adult_meal_cost

theorem restaurant_meal_cost :
  cost_of_group_meal 9 2 2 = 14 := by
  sorry

end restaurant_meal_cost_l143_143368


namespace common_tangents_l143_143239

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y = 0
def circle2_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

theorem common_tangents :
  ∃ (n : ℕ), n = 4 ∧ 
    (∀ (L : ℝ → ℝ → Prop), 
      (∀ x y, L x y → circle1_eqn x y ∧ circle2_eqn x y) → n = 4) := 
sorry

end common_tangents_l143_143239


namespace area_PQR_l143_143382

noncomputable def equilateral_triangle (A B C : Point) : Prop :=
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist A C = 1)

noncomputable def compute_triangle_area (A B C : Point) : ℝ :=
  let base := dist A B
  let height := dist (midpoint A B) C
  (1 / 2) * base * height

theorem area_PQR (M N O A B C P Q R : Point)
  (h1 : ∀ A B, (dist A B > 0)) -- all points are distinct
  (h2 : CenteredCircle ω1 M)
  (h3 : CenteredCircle ω2 N)
  (h4 : CenteredCircle ω3 O)
  (h5 : Tangent ω2 ω3 A)
  (h6 : Tangent ω3 ω1 B)
  (h7 : Tangent ω1 ω2 C)
  (h8 : IntersectsLine MO ω3 P)
  (h9 : IntersectsLine MO ω1 Q)
  (h10 : IntersectsLine AP ω2 R)
  (h11 : equilateral_triangle A B C)
  : compute_triangle_area P Q R = sqrt 3 :=
sorry

end area_PQR_l143_143382


namespace find_x_l143_143866

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l143_143866


namespace inequality_holds_for_all_real_numbers_l143_143110

theorem inequality_holds_for_all_real_numbers (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (k ∈ Set.Icc (-3 : ℝ) 0) := 
sorry

end inequality_holds_for_all_real_numbers_l143_143110


namespace problem1_problem2_l143_143568

def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6*x - 72 <= 0) ∧ (x^2 + x - 6 > 0)

theorem problem1 (x : ℝ) (a : ℝ) (h : a = -1): (p x a ∨ q x) → (-6 ≤ x ∧ x < -3) ∨ (1 < x ∧ x ≤ 12) :=
sorry

theorem problem2 (a : ℝ): (¬ ∃ x : ℝ, p x a) → (¬ ∃ x : ℝ, q x) → (-4 ≤ a ∧ a ≤ -2) :=
sorry

end problem1_problem2_l143_143568


namespace sqrt_b2_ac_lt_sqrt3a_l143_143366

variable (a b c : ℝ)

-- Given conditions
hypothesis ha_gt_b : a > b
hypothesis hb_gt_c : b > c
hypothesis habc_eq_zero : a + b + c = 0
hypothesis hab_gt_zero : (a - b) * (a - c) > 0

-- The proof objective
theorem sqrt_b2_ac_lt_sqrt3a : sqrt (b^2 - a * c) < sqrt 3 * a :=
by
  sorry

end sqrt_b2_ac_lt_sqrt3a_l143_143366


namespace sum_of_divisors_of_24_l143_143284

theorem sum_of_divisors_of_24 : 
  (∑ n in (Finset.filter (λ n, 24 % n = 0) (Finset.range (24 + 1))), n) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l143_143284


namespace tom_profit_calculation_l143_143261

theorem tom_profit_calculation :
  let flour_needed := 500
  let flour_per_bag := 50
  let flour_bag_cost := 20
  let salt_needed := 10
  let salt_cost_per_pound := 0.2
  let promotion_cost := 1000
  let tickets_sold := 500
  let ticket_price := 20

  let flour_bags := flour_needed / flour_per_bag
  let cost_flour := flour_bags * flour_bag_cost
  let cost_salt := salt_needed * salt_cost_per_pound
  let total_expenses := cost_flour + cost_salt + promotion_cost
  let total_revenue := tickets_sold * ticket_price

  let profit := total_revenue - total_expenses

  profit = 8798 := by
  sorry

end tom_profit_calculation_l143_143261


namespace rationalize_sqrt_fraction_l143_143631

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143631


namespace inequality_of_ab_bc_ca_l143_143426

theorem inequality_of_ab_bc_ca (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^4 + b^4 + c^4 = 3) : 
  (1 / (4 - a * b)) + (1 / (4 - b * c)) + (1 / (4 - c * a)) ≤ 1 :=
by
  sorry

end inequality_of_ab_bc_ca_l143_143426


namespace percentage_of_women_in_study_group_l143_143777

theorem percentage_of_women_in_study_group
  (W : ℝ) -- percentage of women in decimal form
  (h1 : 0 < W ∧ W ≤ 1) -- percentage of women should be between 0 and 1
  (h2 : 0.4 * W = 0.32) -- 40 percent of women are lawyers, and probability is 0.32
  : W = 0.8 :=
  sorry

end percentage_of_women_in_study_group_l143_143777


namespace rationalize_denominator_l143_143689

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l143_143689


namespace three_digit_values_satisfying_property_l143_143161

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def double_sum_of_digits_eq_4 (x : ℕ) : Prop :=
    sum_of_digits (sum_of_digits x) = 4

theorem three_digit_values_satisfying_property : {x // 100 ≤ x ∧ x ≤ 999 ∧ double_sum_of_digits_eq_4 x}.card = 30 := sorry

end three_digit_values_satisfying_property_l143_143161


namespace complex_number_solution_l143_143012

theorem complex_number_solution (a b : ℝ) (z : ℂ) :
  z = a + b * I →
  (a - 2) ^ 2 + b ^ 2 = 25 →
  (a + 4) ^ 2 + b ^ 2 = 25 →
  a ^ 2 + (b - 2) ^ 2 = 25 →
  z = -1 - 4 * I :=
sorry

end complex_number_solution_l143_143012


namespace tom_made_money_correct_l143_143260

-- Define constants for flour, salt, promotion cost, ticket price, and tickets sold
def flour_needed : ℕ := 500
def flour_bag_size : ℕ := 50
def flour_bag_cost : ℕ := 20
def salt_needed : ℕ := 10
def salt_cost_per_pound : ℚ := 0.2
def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 500

-- Compute how much money Tom made
def money_made : ℤ :=
  let flour_bags := flour_needed / flour_bag_size
  let total_flour_cost := flour_bags * flour_bag_cost
  let total_salt_cost := salt_needed * salt_cost_per_pound
  let total_cost := total_flour_cost + total_salt_cost + promotion_cost
  let total_revenue := tickets_sold * ticket_price
  total_revenue - total_cost

-- The theorem statement
theorem tom_made_money_correct :
  money_made = 8798 := by
  sorry

end tom_made_money_correct_l143_143260


namespace geom_sequence_sum_l143_143454

noncomputable def f (x : ℝ) : ℝ := 2 / (1 + x^2)

theorem geom_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ r ≠ 1, ∀ n, a (n+1) = r * a n)
    (h_log : real.log10 (a 1) + real.log10 (a 2019) = 0) :
    (∑ i in finset.range 2019, f (a (i+1))) = 2019 :=
by
  sorry

end geom_sequence_sum_l143_143454


namespace john_makes_200_profit_l143_143986

noncomputable def john_profit (num_woodburnings : ℕ) (price_per_woodburning : ℕ) (cost_of_wood : ℕ) : ℕ :=
  (num_woodburnings * price_per_woodburning) - cost_of_wood

theorem john_makes_200_profit :
  john_profit 20 15 100 = 200 :=
by
  sorry

end john_makes_200_profit_l143_143986


namespace exists_monochromatic_triangle_l143_143747

-- We define the vertices of P and the coloring property
inductive Vertex
| A (n : Fin 7)
| B1
| B2

open Vertex

-- Define adjacency function for vertices
def adjacent : Vertex → Vertex → Prop
| A n, A m => (n ≠ m ∧ ((n + 1) % 7 = m ∨ (m + 1) % 7 = n))
| B1, B2 => false
| B2, B1 => false
| _, _ => true

-- Define coloring of edges
inductive Colour
| red
| blue

open Colour

-- Function to check if three vertices form a monochromatic triangle
def monochromatic (col : (Vertex × Vertex) → Colour) (v1 v2 v3 : Vertex) : Prop :=
(col(v1, v2) = col(v2, v3)) ∧ (col(v2, v3) = col(v3, v1)) ∧ (col(v1, v2) = col(v3, v1))

-- The theorem statement
theorem exists_monochromatic_triangle (col : (Σ v1 v2, (¬adjacent v1 v2)) → Colour) :
  ∃ v1 v2 v3 : Vertex, ¬adjacent v1 v2 ∧ ¬adjacent v2 v3 ∧ ¬adjacent v1 v3 ∧ monochromatic col v1 v2 v3 := sorry

end exists_monochromatic_triangle_l143_143747


namespace sum_divisors_24_l143_143279

theorem sum_divisors_24 :
  (∑ n in Finset.filter (λ n => 24 % n = 0) (Finset.range 25), n) = 60 :=
by
  sorry

end sum_divisors_24_l143_143279


namespace max_value_of_f_on_I_l143_143022

-- Define the function y = x + 2 * cos x
def f (x : ℝ) : ℝ := x + 2 * Real.cos x

-- Define the interval [0, π/2]
def I : Set ℝ := Set.Icc 0 (Real.pi / 2)

-- Define the problem statement
theorem max_value_of_f_on_I : ∃ x ∈ I, ∀ y ∈ I, f x ≥ f y :=
  sorry

end max_value_of_f_on_I_l143_143022


namespace perpendicular_length_GH_from_centroid_l143_143211

theorem perpendicular_length_GH_from_centroid
  (A B C D E F G : ℝ)
  -- Conditions for distances from vertices to the line RS
  (hAD : AD = 12)
  (hBE : BE = 12)
  (hCF : CF = 18)
  -- Define the coordinates based on the vertical distances to line RS
  (yA : A = 12)
  (yB : B = 12)
  (yC : C = 18)
  -- Define the centroid G of triangle ABC based on the average of the y-coordinates
  (yG : G = (A + B + C) / 3)
  : G = 14 :=
by
  sorry

end perpendicular_length_GH_from_centroid_l143_143211


namespace proof_problem_l143_143910

variable {ℝ : Type*} [Real ℝ]

noncomputable def f : ℝ → ℝ
noncomputable def g (x : ℝ) : ℝ := (f' x)

theorem proof_problem (hf_sym : ∀ x, f(x) = f(-2 - x))
  (hg_odd: ∀ x, g(3 + 2*x) = - g(-3 - 2*x)) :
  (g (-1) = 0) ∧ 
  (g 2023 + g (-2025) = 0) ∧ 
  (g 3 = 0) ∧ 
  (g 2023 = 0) :=
  sorry

end proof_problem_l143_143910


namespace exists_divisor_of_others_l143_143153

open Set

-- Given conditions as definitions
def condition1 (S : Set ℕ) : Prop :=
  ∀ (a b c d ∈ S),
  a ∣ b ∧ a ∣ c ∧ a ∣ d ∨ a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

-- Main theorem statement
theorem exists_divisor_of_others 
  (S : Set ℕ) 
  (hS1 : cardinal.mk S = 100)
  (hS2 : ∀ (a b c d ∈ S), a ∣ b ∧ a ∣ c ∧ a ∣ d ∨ a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c) 
: ∃ a ∈ S, ∀ b ∈ S, b ≠ a → a ∣ b :=
sorry

end exists_divisor_of_others_l143_143153


namespace triangle_XYZ_l143_143144

noncomputable def XY_value (XY YZ XZ : ℝ) (X_angle : ℝ) (tanZ sinY : ℝ) := XY = 40 * real.sqrt 2 / 3

theorem triangle_XYZ
  (XY YZ XZ : ℝ) (X_angle : ℝ)
  (h1 : YZ = 20)
  (h2 : X_angle = real.pi / 2)
  (h3 : tanZ = 3 * sinY)
  (h4 : tanZ = XY / XZ)
  (h5 : sinY = XY / YZ) :
  XY_value XY YZ XZ X_angle tanZ sinY :=
begin
  unfold XY_value,
  sorry
end

end triangle_XYZ_l143_143144


namespace triangle_angle_contradiction_l143_143600

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α < 60) (h3 : β < 60) (h4 : γ < 60) : false := 
sorry

end triangle_angle_contradiction_l143_143600


namespace angle_same_terminal_side_l143_143324

theorem angle_same_terminal_side (k : ℤ) : 
  ∃ (θ : ℤ), θ = -324 ∧ 
    ∀ α : ℤ, α = 36 + k * 360 → 
            ( (α % 360 = θ % 360) ∨ (α % 360 + 360 = θ % 360) ∨ (θ % 360 + 360 = α % 360)) :=
by
  sorry

end angle_same_terminal_side_l143_143324


namespace sandy_red_balloons_l143_143704

-- Define the given conditions
def sara_red_balloons : ℕ := 31
def total_red_balloons : ℕ := 55

-- Formulate the theorem
theorem sandy_red_balloons : ∃ sandy_red_balloons : ℕ, sandy_red_balloons = 24 :=
by
  -- Using the condition that Sara and Sandy together have 55 red balloons.
  let sandy_red_balloons := total_red_balloons - sara_red_balloons
  -- Since 55 - 31 = 24, we conclude sandy has 24 red balloons.
  existsi sandy_red_balloons
  -- Prove the expected equality
  show sandy_red_balloons = 24 from sorry

end sandy_red_balloons_l143_143704


namespace proof_hamburgers_sold_and_revenue_l143_143187

def monday_before_6 := 48
def monday_additional := 28
def monday_price := 3.50

def tuesday_before_6 := 36
def tuesday_additional := 15
def tuesday_price := 4.25

def wednesday_before_6 := 52
def wednesday_additional := 21
def wednesday_price := 3.75

theorem proof_hamburgers_sold_and_revenue :
  let monday_after_6 := monday_before_6 + monday_additional,
      tuesday_after_6 := tuesday_before_6 + tuesday_additional,
      wednesday_after_6 := wednesday_before_6 + wednesday_additional,
      total_hamburgers := monday_after_6 + tuesday_after_6 + wednesday_after_6,
      total_revenue := (monday_after_6 * monday_price) + 
                       (tuesday_after_6 * tuesday_price) + 
                       (wednesday_after_6 * wednesday_price) 
  in total_hamburgers = 200 ∧ total_revenue = 756.50 :=
by
  sorry

end proof_hamburgers_sold_and_revenue_l143_143187


namespace particle_reaches_origin_final_sum_l143_143343

noncomputable def particle_probability : ℚ :=
  let prob := 1 / 3^8 in
  63 * prob

theorem particle_reaches_origin :
  particle_probability = 63 / 3^8 :=
by 
  sorry

theorem final_sum :
  (63 + 8 : ℕ) = 71 :=
by 
  sorry

end particle_reaches_origin_final_sum_l143_143343


namespace rationalize_sqrt_fraction_denom_l143_143610

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143610


namespace minimum_F_l143_143539

noncomputable def F (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + (0.5 * x)

theorem minimum_F : ∃ x : ℝ, x ≥ 0 ∧ F x = 57.5 ∧ ∀ y ≥ 0, F y ≥ F x := by
  use 55
  sorry

end minimum_F_l143_143539


namespace determine_cos_A_given_tan_sec_eq_three_l143_143488

theorem determine_cos_A_given_tan_sec_eq_three (A : ℝ) (h : Real.tan A + Real.sec A = 3) : Real.cos A = 3 / 5 :=
sorry

end determine_cos_A_given_tan_sec_eq_three_l143_143488


namespace polynomials_equal_l143_143576

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

def complexRoots (p : Polynomial ℝ) : set ℂ := {z : ℂ | IsRoot p z}
def rootsEqual (p q : Polynomial ℝ) : Prop := complexRoots p = complexRoots q

theorem polynomials_equal (P Q : Polynomial ℝ)
  (hP_nonconst : ¬(IsMonic P) ∧ ¬(IsMonic Q))
  (hRoots : rootsEqual P Q)
  (hRoots_minus_one : rootsEqual (P - 1) (Q - 1)) :
  P = Q :=
sorry

end polynomials_equal_l143_143576


namespace walk_back_length_l143_143940

def is_prime (n : ℕ) : Prop := sorry -- Placeholder for the definition of prime numbers
def is_composite (n : ℕ) : Prop := sorry -- Placeholder for the definition of composite numbers
def is_perfect_square (n : ℕ) : Prop := sorry -- Placeholder for the definition of perfect squares

def step_count (move : ℕ) : ℤ :=
  if is_prime move then 1
  else if is_perfect_square move then 3
  else if is_composite move then (-2)
  else 0

def total_steps : ℤ := (List.range' 2 49).sum step_count

theorem walk_back_length : total_steps = -23 :=
  sorry -- Proof required

end walk_back_length_l143_143940


namespace treasure_in_box_2_l143_143980

def Box := Fin 5  -- We have 5 boxes indexed from 0 to 4.

-- Conditions: Box materials. True means cedar, False means sandalwood.
def is_cedar (n : Box) : Prop :=
  n = 0 ∨ n = 3 ∨ n = 4

-- Box inscriptions.
def inscription_1 (treasure_box : Box) : Prop :=
  treasure_box = 0 ∨ treasure_box = 3

def inscription_2 (treasure_box : Box) : Prop := 
  treasure_box = 0

def inscription_3 (treasure_box : Box) : Prop := 
  treasure_box = 2 ∨ treasure_box = 4

def inscription_4 (treasure_box : Box) : Prop := 
  treasure_box = 3 ∨ treasure_box = 4

def inscription_5 (statements : Box → Prop) : Prop :=
  ∀ n : Box, n ≠ 4 → ¬statements n

-- The number of false statements on cedar boxes is equal to the number of false statements on sandalwood boxes.
def equal_false_statements (conditions : Box → Prop) : Prop :=
  (∑ n in (Finset.filter is_cedar Finset.univ), if conditions n then 0 else 1) =
  (∑ n in (Finset.filter (λ n, ¬is_cedar n) Finset.univ), if conditions n then 0 else 1)

-- The main proof problem.
theorem treasure_in_box_2 : ∃ treasure_box : Box, treasure_box = 1 ∧ 
  (∃ (conditions : Box → Prop),
    conditions 0 ↔ inscription_1 treasure_box ∧
    conditions 1 ↔ inscription_2 treasure_box ∧
    conditions 2 ↔ inscription_3 treasure_box ∧
    conditions 3 ↔ inscription_4 treasure_box ∧
    conditions 4 ↔ inscription_5 conditions ∧
    equal_false_statements (λ n, ¬conditions n)) :=
begin
  sorry -- Proof goes here
end

end treasure_in_box_2_l143_143980


namespace problem_solution_l143_143947

variable (α : ℝ)

/-- If $\sin\alpha = 2\cos\alpha$, then the function $f(x) = 2^x - \tan\alpha$ satisfies $f(0) = -1$. -/
theorem problem_solution (h : Real.sin α = 2 * Real.cos α) : (2^0 - Real.tan α) = -1 := by
  sorry

end problem_solution_l143_143947


namespace function_equality_l143_143805

theorem function_equality :
  (∀ x : ℝ, y = x - 1 → y = t - 1) ∧
  ¬ (∀ x : ℝ, y = x - 1 → y = sqrt (x^2 - 2*x + 1)) ∧
  ¬ (∀ x : ℝ, y = x - 1 → y = (x^2 - 1) / (x + 1)) ∧
  ¬ (∀ x : ℝ, y = x - 1 → y = - sqrt ((x - 1)^2)) :=
sorry

end function_equality_l143_143805


namespace points_on_circle_l143_143849

theorem points_on_circle (t : ℝ) : ∃ x y : ℝ, x = Real.cos t ∧ y = Real.sin t ∧ x^2 + y^2 = 1 :=
by
  sorry

end points_on_circle_l143_143849


namespace correct_operations_count_l143_143725

theorem correct_operations_count :
  (|2023| = 2023) ∧
  (2023^0 = 1) ∧
  (2023^{-1} = 1/2023) ∧
  (Real.sqrt (2023^2) = 2023) ↔
  4 = 4 := 
by
  sorry

end correct_operations_count_l143_143725


namespace sugar_cubes_left_l143_143591

theorem sugar_cubes_left (h w d : ℕ) (hd1 : w * d = 77) (hd2 : h * d = 55) :
  (h - 1) * w * (d - 1) = 300 ∨ (h - 1) * w * (d - 1) = 0 :=
by
  sorry

end sugar_cubes_left_l143_143591


namespace event_probabilities_equal_l143_143517

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l143_143517


namespace smallest_recording_zero_l143_143776

noncomputable def smallest_recording (V : Fin 5 → ℝ) : ℝ :=
  if hV : (∀ i, V i ∈ {0, 0.2, 0.4, ... , 9.8, 10}) ∧ (∑ i, V i = 32) then
    sorry

theorem smallest_recording_zero (V : Fin 5 → ℝ) (h1 : ∀ i, V i ∈ {0, 0.2, 0.4, ..., 9.8, 10}) 
    (h2 : (∑ i, V i) = 32) : ∃ i, V i = 0 :=
sorry

end smallest_recording_zero_l143_143776


namespace quadratic_b_value_l143_143370

theorem quadratic_b_value (b n : ℝ) (h : b < 0) 
  (H1 : ∀ x : ℝ, x^2 + b * x + (1 / 4) = (x + n) ^ 2 + (1 / 16)) : b = -√3 / 2 :=
by {
  sorry
}

end quadratic_b_value_l143_143370


namespace largest_n_l143_143416

def is_solution (n : ℕ) : Prop :=
  ∃ (X : Fin n → Finset ℕ), ∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n →
    (X ⟨a-1, sorry⟩ ∪ X ⟨b-1, sorry⟩ ∪ X ⟨c-1, sorry⟩).card = Int.ceil (Real.sqrt (a * b * c))

theorem largest_n : ∃ (n : ℕ), is_solution n ∧ ∀ m, (is_solution m → m ≤ 4) :=
by
  exists 4
  split
  . sorry -- Proof of is_solution 4
  . intro m h
    by_contradiction
    sorry -- Proof that larger m violates the conditions

end largest_n_l143_143416


namespace min_value_expression_l143_143449

theorem min_value_expression (a d b c : ℝ) (habd : a ≥ 0 ∧ d ≥ 0) (hbc : b > 0 ∧ c > 0) (h_cond : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expression_l143_143449


namespace largest_possible_integer_smallest_possible_integer_l143_143814

theorem largest_possible_integer : 3 * (15 + 20 / 4 + 1) = 63 := by
  sorry

theorem smallest_possible_integer : (3 * 15 + 20) / (4 + 1) = 13 := by
  sorry

end largest_possible_integer_smallest_possible_integer_l143_143814


namespace range_of_m_l143_143569

def has_distinct_positive_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0

def no_real_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c) < 0

def p (m : ℝ) : Prop :=
  has_distinct_positive_roots 1 (2 * m) 1

def q (m : ℝ) : Prop :=
  no_real_roots 1 (2 * (m - 2)) (-3 * m + 10)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (m ∈ set.Iic (-2) ∪ set.Ico (-1) 3) := sorry

end range_of_m_l143_143569


namespace distribute_books_l143_143400

theorem distribute_books :
  ∃ (n : ℕ), 
  (n = 216) ∧ 
  ∃ (books : Fin 4 → ℕ), 
  (∑ i, books i = 16) ∧  -- sum of books each student gets
  (∀ i, books i > 0) ∧   -- each student gets at least one book
  (∀ (i j : Fin 4), i ≠ j → books i ≠ books j)  -- each student gets a different number of books
  :=
by
  use 216
  constructor
  · rfl
  sorry

end distribute_books_l143_143400


namespace rationalize_sqrt_fraction_l143_143629

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143629


namespace taller_pot_shadow_length_l143_143748

theorem taller_pot_shadow_length :
  ∀ (h1 h2 s1 θ1 θ2: ℝ), θ1 = 45 ∧ θ2 = 60 ∧ h1 = 20 ∧ s1 = 10 ∧ h2 = 40 →
  let s := h2 / Real.tan (Real.pi * θ2 / 180) in
  s ≈ 23.09 := by
  sorry

end taller_pot_shadow_length_l143_143748


namespace divide_triangle_area_l143_143884

theorem divide_triangle_area (A B C : (ℝ × ℝ)) (a : ℝ):
  A = (1,3) ∧ B = (1,0) ∧ C = (10,0) ∧
  ( let total_area := (1/2) * real.abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
    in let left_area := (1/2) * (a - A.1) * A.2
    in let right_area := total_area - left_area 
    in left_area = 3 * right_area )
  → a = 7.75 :=
by
  intros h
  cases h with ha hb
  sorry

end divide_triangle_area_l143_143884


namespace events_A_B_equal_prob_l143_143512

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l143_143512


namespace keychain_arrangements_l143_143963

theorem keychain_arrangements :
  let keys := ["House", "Car", "Mailbox", "Bike", "Key1", "Key2"]
  -- treat ("House", "Car") and ("Mailbox", "Bike") as single units
  let units := [("House", "Car"), ("Mailbox", "Bike"), ("Key1"), ("Key2")]
  -- number of unique ways to arrange these units accounting for rotation and reflection
  (factorial 3 / 2) = 3 :=
by
  let keys := ["House", "Car", "Mailbox", "Bike", "Key1", "Key2"]
  let units := [("House", "Car"), ("Mailbox", "Bike"), ("Key1"), ("Key2")]
  calc (factorial 3) / 2 = (6 / 2) : by sorry
                     ... = 3 : by sorry

end keychain_arrangements_l143_143963


namespace range_of_m_l143_143930

open Set

variable (m : ℝ)
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 9}
def complementR (A : Set ℝ) : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem range_of_m (H : (complementR A) ∩ (B m) = B m) : m ≤ -11 ∨ m ≥ 3 :=
by
  sorry

end range_of_m_l143_143930


namespace main_l143_143430

noncomputable def a_n (n : ℕ) : ℚ :=
  (1 / 4) * (1 / 4)^(n - 1)

noncomputable def b_n (n : ℕ) : ℚ :=
  3 * (Real.log_base (1 / 4) (a_n n)) - 2

noncomputable def c_n (n : ℕ) : ℚ :=
  a_n n * b_n n

def question1 : Prop :=
  ∃ a d : ℚ, ∀ n : ℕ, b_n n = a + (n - 1) * d

def question2 (n : ℕ) : Prop :=
  (∑ k in range n, c_n k) = (2 / 3 - (3 * n + 2) * (1 / 4)^n / 3)

def question3 (m : ℚ) : Prop :=
  (∀ n : ℕ, c_n n ≤ ((1 / 4) * m^2 + m - 1)) → (1 ≤ m ∨ m ≤ -5)

-- You can now write the following statement:
theorem main : question1 ∧ question2 ∧ question3 := sorry

end main_l143_143430


namespace problem1_problem2_problem3_l143_143917

noncomputable 
def f (x : ℝ) : ℝ := Real.exp x

theorem problem1 
  (a b : ℝ)
  (h1 : f 1 = a) 
  (h2 : b = 0) : f x = Real.exp x :=
sorry

theorem problem2 
  (k : ℝ) 
  (h : ∀ x : ℝ, f x ≥ k * x) : 0 ≤ k ∧ k ≤ Real.exp 1 :=
sorry

theorem problem3 
  (t : ℝ)
  (h : t ≤ 2) : ∀ x : ℝ, f x > t + Real.log x :=
sorry

end problem1_problem2_problem3_l143_143917


namespace general_term_of_sequence_l143_143049

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  sorry -- the recurrence relation will go here, but we'll skip its implementation

theorem general_term_of_sequence :
  ∀ n : ℕ, n ≥ 1 → a n = 3 - (2 / n) :=
by sorry

end general_term_of_sequence_l143_143049


namespace triangle_area_def_l143_143145

-- Stating the conditions and the proof problem.
theorem triangle_area_def (DE EF DF : ℝ) (h1 : DE = 30) (h2 : EF = 30) (h3 : DF = 40) : 
  let s := (DE + EF + DF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF)) in
  area = 447 := 
sorry

end triangle_area_def_l143_143145


namespace valid_function_rule_l143_143250

def M : Set ℤ := {-2, 1, 2, 4}
def N : Set ℤ := {1, 2, 4, 16}
def rule (x : ℤ) : ℤ := 2 ^ (abs x)

theorem valid_function_rule : ∀ x ∈ M, rule x ∈ N :=
by
  intro x hx
  simp only [M, N, rule, Set.mem_setOf, Set.mem_insert_iff, Set.mem_singleton_iff, abs_eq_self, abs_neg]
  fin_cases hx <;> native.dec_trivial

end valid_function_rule_l143_143250


namespace solution_set_l143_143566

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 4 * x + 6 else -x + 6

theorem solution_set (S : Set ℝ) :
  S = {x | f x < 3} ↔ (S = {x | -3 < x ∧ x < -1} ∪ {x | 3 < x}) :=
sorry

end solution_set_l143_143566


namespace solution_exists_l143_143010

noncomputable def has_solution (a : ℝ) : Prop :=
∃ x y : ℝ, (x^2 + y^2 = 1 ∧ (|x - 0.5| + |y| - a) / (sqrt 3 * y - x) = 0)

theorem solution_exists (a : ℝ) :
  a = 0.5 ∨ a = sqrt 3 / 2 ∨ a = 1.5 ∨ a = sqrt 2 + 0.5 ∨ a = sqrt 3 / 2 + 1 ↔ has_solution a :=
sorry

end solution_exists_l143_143010


namespace find_ellipse_equation_find_min_area_of_triangle_l143_143052

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := 
  (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1

def point := ℝ × ℝ

/-- Conditions for the ellipse -/
variables (a b : ℝ) (M1 M2 M3 : point)

axiom h_a_gt_b_gt_0 : a > b ∧ b > 0
axiom h_eccentricity : (sqrt 2 / 2 : ℝ) = sqrt ((a ^ 2 - b ^ 2) / a ^ 2)
axiom h_M1 : M1 = (-2, sqrt 2)
axiom h_M2 : M2 = (2, -sqrt 2)
axiom h_M3 : M3 = (sqrt 2, sqrt 3 / 2)
axiom two_points_on_ellipse : ellipse_eq a b (fst M1) (snd M1) ∧ ellipse_eq a b (fst M2) (snd M2) ∧ ¬ellipse_eq a b (fst M3) (snd M3)

/-- Proving ellipse equation -/
theorem find_ellipse_equation : ellipse_eq 2sqrt 2 2 x y ↔
  ellipse_eq 8 4 x y :=
sorry

/-- Proving minimum area of triangle EPQ -/
variables (E F P Q : point)
axiom h_E : E = (0, 2)
axiom h_F : F = (2sqrt 2, 0)
axiom h_line : ∀ x y, y = x - 4

theorem find_min_area_of_triangle : 
  ∀ A B : point,
  (A ≠ E ∧ B ≠ E ∧ 
  ¬ (A = top_vertex ∨ A = right_focus) ∧ 
  ¬ (B = top_vertex ∨ B = right_focus) ∧ 
  intersects (line_through F) C A B ∧ 
  intersects (line E A) line_PQ P ∧ 
  intersects (line E B) line_PQ Q) → 
  area_triangle E P Q = 36/5 :=
sorry

end find_ellipse_equation_find_min_area_of_triangle_l143_143052


namespace rationalize_sqrt_5_over_12_l143_143644

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l143_143644


namespace solve_for_x_l143_143487

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 8) (h2 : 2 * x + 3 * y = 1) : x = 2 := 
by 
  sorry

end solve_for_x_l143_143487


namespace find_length_of_CE_l143_143544

theorem find_length_of_CE
  (triangle_ABE_right : ∀ A B E : Type, ∃ (angle_AEB : Real), angle_AEB = 45)
  (triangle_BCE_right : ∀ B C E : Type, ∃ (angle_BEC : Real), angle_BEC = 45)
  (triangle_CDE_right : ∀ C D E : Type, ∃ (angle_CED : Real), angle_CED = 45)
  (AE_is_32 : 32 = 32) :
  ∃ (CE : ℝ), CE = 16 * Real.sqrt 2 :=
by
  sorry

end find_length_of_CE_l143_143544


namespace train_crossing_time_l143_143313

-- Definitions based on conditions
def length_of_train : ℝ := 110  -- length in meters
def length_of_bridge : ℝ := 132 -- length in meters
def speed_in_kmph : ℝ := 36     -- speed in kilometers per hour

-- Convert speed to meters per second
def speed_in_mps (speed_in_kmph : ℝ) : ℝ := speed_in_kmph * (1000 / 3600)

-- Total distance to cross
def total_distance (length_of_train length_of_bridge : ℝ) : ℝ :=
  length_of_train + length_of_bridge

-- Time taken to cross the bridge
def time_to_cross (total_distance speed_in_mps : ℝ) : ℝ :=
  total_distance / speed_in_mps

-- Theorem statement
theorem train_crossing_time :
  time_to_cross (total_distance 110 132) (speed_in_mps 36) = 24.2 :=
by
  sorry

end train_crossing_time_l143_143313


namespace probability_equality_l143_143505

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l143_143505


namespace circle_center_eq_circle_center_is_1_3_2_l143_143396

-- Define the problem: Given the equation of the circle, prove the center is (1, 3/2)
theorem circle_center_eq (x y : ℝ) :
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0 ↔ (x - 1)^2 + (y - 3/2)^2 = 3 := sorry

-- Prove that the center of the circle from the given equation is (1, 3/2)
theorem circle_center_is_1_3_2 :
  ∃ x y : ℝ, (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0) ∧ (x = 1) ∧ (y = 3 / 2) := sorry

end circle_center_eq_circle_center_is_1_3_2_l143_143396


namespace modulus_of_z_l143_143899

-- Given conditions
variable (z : ℂ) (hz : (1 - z) / (1 + z) = 2 * Complex.i)

-- Proof statement: 
theorem modulus_of_z (z : ℂ) (hz : (1 - z) / (1 + z) = 2 * Complex.i) : Complex.abs z = 1 :=
sorry

end modulus_of_z_l143_143899


namespace mother_kept_one_third_l143_143342

-- Define the problem conditions
def total_sweets : ℕ := 27
def eldest_sweets : ℕ := 8
def youngest_sweets : ℕ := eldest_sweets / 2
def second_sweets : ℕ := 6
def total_children_sweets : ℕ := eldest_sweets + youngest_sweets + second_sweets
def sweets_mother_kept : ℕ := total_sweets - total_children_sweets
def fraction_mother_kept : ℚ := sweets_mother_kept / total_sweets

-- Prove the fraction of sweets the mother kept
theorem mother_kept_one_third : fraction_mother_kept = 1 / 3 := 
  by
    sorry

end mother_kept_one_third_l143_143342


namespace ages_total_l143_143772

theorem ages_total (P Q : ℕ) (h1 : P - 8 = (1 / 2) * (Q - 8)) (h2 : P / Q = 3 / 4) : P + Q = 28 :=
by
  sorry

end ages_total_l143_143772


namespace quad_eq_sols_l143_143229

theorem quad_eq_sols (a b : ℕ) (ha : a = 131) (hb : b = 7) :
  (∃ x : ℝ, x^2 + 14 * x = 82 ∧ x = Real.sqrt a - b) → a + b = 138 :=
by
  sorry

end quad_eq_sols_l143_143229


namespace value_of_expression_l143_143858

theorem value_of_expression (x y : ℝ) (h₁ : x * y = 3) (h₂ : x + y = 4) : x ^ 2 + y ^ 2 - 3 * x * y = 1 := 
by
  sorry

end value_of_expression_l143_143858


namespace rationalize_sqrt_fraction_l143_143675

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143675


namespace trader_profit_l143_143795

theorem trader_profit (P : ℝ) (h₀ : P > 0) : 
  let buying_price := 0.90 * P
  let selling_price := 1.80 * buying_price
  (selling_price - P) / P * 100 = 62 :=
by
  let buying_price := 0.90 * P
  let selling_price := 1.80 * buying_price
  have h₁ : (selling_price - P) / P * 100 = ((1.62 * P - P) / P) * 100 := by sorry
  have h₂ : ((1.62 * P - P) / P) * 100 = ((0.62 * P) / P) * 100 := by sorry
  have h₃ : ((0.62 * P) / P) * 100 = 0.62 * 100 := by sorry
  have h₄ : 0.62 * 100 = 62 := by sorry
  exact h₄

end trader_profit_l143_143795


namespace sequence_constant_and_square_l143_143177

theorem sequence_constant_and_square
  (a : ℕ → ℤ)
  (h : ∀ (n k : ℕ), 0 < k → (∃ m : ℤ, (a n + a (n + 1) + ... + a (n + k - 1)) = m * k^2)) :
  ∃ c : ℤ, ∀ i : ℕ, a i = c := sorry

end sequence_constant_and_square_l143_143177


namespace quadrilateral_opposite_sides_equal_l143_143727

theorem quadrilateral_opposite_sides_equal {A B C D : Point} : 
  (angle A B C = angle C D A) ∧ (angle B C D = angle D A B) → 
  dist A B = dist C D ∧ dist A D = dist B C :=
begin
  sorry
end

end quadrilateral_opposite_sides_equal_l143_143727


namespace rectangle_geometry_l143_143133

theorem rectangle_geometry (A B C D M N P Q : Type*)
(rectangle_ABCD : Rectangle ABCD)
(midpoint_M : M = midpoint A D)
(midpoint_N : N = midpoint B C)
(P_on_CD_extension : P extends CD)
(Q_intersect_PM_AC : Q = intersection (line PM) (line AC)) :
∠ QNM = ∠ MNP := 
sorry

end rectangle_geometry_l143_143133


namespace mutually_incongruent_mod_iff_power_of_2_l143_143421

-- Define the sum of the first k natural numbers
def sum_nat (k : ℕ) : ℕ := (k * (k + 1)) / 2

-- Define the set S
def set_S (n : ℕ) : Set ℕ := {0} ∪ {sum_nat k | k : ℕ, k < n}

theorem mutually_incongruent_mod_iff_power_of_2 (n : ℕ) (hn : 0 < n):
  (∀ (a b ∈ set_S n), a ≠ b → a % n ≠ b % n) ↔ ∃ (k : ℕ), n = 2^k :=
by
  sorry

end mutually_incongruent_mod_iff_power_of_2_l143_143421


namespace two_a_minus_b_values_l143_143859

theorem two_a_minus_b_values (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 5) (h3 : |a + b| = -(a + b)) :
  (2 * a - b = 13) ∨ (2 * a - b = -3) :=
sorry

end two_a_minus_b_values_l143_143859


namespace lower_right_square_is_one_l143_143257

open Matrix

def initial_grid : Matrix (Fin 4) (Fin 4) (Option ℕ) := ![
  ![(some 1), none, none, (some 4)],
  ![none, (some 2), none, none],
  ![none, none, (some 3), none],
  ![none, none, none, none]
]

def is_valid_grid (m : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  ∀ (i : Fin 4), (Finset.univ.image (m i)).card = 4 ∧ (Finset.univ.image (λ j, m j i)).card = 4

theorem lower_right_square_is_one (m : Matrix (Fin 4) (Fin 4) ℕ) (h_valid : is_valid_grid m)
  (h_initial : ∀ i j, initial_grid i j = some x → m i j = x) :
  m ⟨3, sorry⟩ ⟨3, sorry⟩ = 1 :=
sorry

end lower_right_square_is_one_l143_143257


namespace find_smaller_number_l143_143419

theorem find_smaller_number (x : ℕ) (hx : x + 4 * x = 45) : x = 9 :=
by
  sorry

end find_smaller_number_l143_143419


namespace rationalize_sqrt_fraction_l143_143620

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143620


namespace find_y_l143_143889

noncomputable def pointA : prod ℝ ℝ := (-3, 10)
noncomputable def pointB (y : ℝ) : prod ℝ ℝ := (5, y)
noncomputable def slope (A B : prod ℝ ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

theorem find_y (y : ℝ) :
  slope pointA (pointB y) = -4 / 3 ↔ y = -2 / 3 :=
by
  sorry

end find_y_l143_143889


namespace rationalize_denominator_l143_143683

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l143_143683


namespace rationalize_sqrt_fraction_l143_143635

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l143_143635


namespace total_payment_correct_l143_143822

-- Define the prices of different apples.
def price_small_apple : ℝ := 1.5
def price_medium_apple : ℝ := 2.0
def price_big_apple : ℝ := 3.0

-- Define the quantities of apples bought by Donny.
def quantity_small_apples : ℕ := 6
def quantity_medium_apples : ℕ := 6
def quantity_big_apples : ℕ := 8

-- Define the conditions.
def discount_medium_apples_threshold : ℕ := 5
def discount_medium_apples_rate : ℝ := 0.20
def tax_rate : ℝ := 0.10
def big_apple_special_offer_count : ℕ := 3
def big_apple_special_offer_discount_rate : ℝ := 0.50

-- Step function to calculate discount and total cost.
noncomputable def total_cost : ℝ :=
  let cost_small := quantity_small_apples * price_small_apple
  let cost_medium := quantity_medium_apples * price_medium_apple
  let discount_medium := if quantity_medium_apples > discount_medium_apples_threshold 
                         then cost_medium * discount_medium_apples_rate else 0
  let cost_medium_after_discount := cost_medium - discount_medium
  let cost_big := quantity_big_apples * price_big_apple
  let discount_big := (quantity_big_apples / big_apple_special_offer_count) * 
                       (price_big_apple * big_apple_special_offer_discount_rate)
  let cost_big_after_discount := cost_big - discount_big
  let total_cost_before_tax := cost_small + cost_medium_after_discount + cost_big_after_discount
  let tax := total_cost_before_tax * tax_rate
  total_cost_before_tax + tax

-- Define the expected total payment.
def expected_total_payment : ℝ := 43.56

-- The theorem statement: Prove that total_cost equals the expected total payment.
theorem total_payment_correct : total_cost = expected_total_payment := sorry

end total_payment_correct_l143_143822


namespace parabola_focus_directrix_distance_l143_143337

theorem parabola_focus_directrix_distance {a : ℝ} (h₀ : a > 0):
  (∃ (b : ℝ), ∃ (x1 x2 : ℝ), (x1 + x2 = 1 / a) ∧ (1 / (2 * a) = 1)) → 
  (1 / (2 * a) / 2 = 1 / 4) :=
by
  sorry

end parabola_focus_directrix_distance_l143_143337


namespace product_approximation_l143_143323

theorem product_approximation :
  2.46 * 8.163 * (5.17 + 4.829) ≈ 200 :=
by sorry

end product_approximation_l143_143323


namespace min_area_circle_l143_143830

def f (x : ℝ) : ℝ := 1 + x - (x^2)/2 + (x^3)/3 - ... - (x^2016)/2016

def F (x : ℝ) : ℝ := f (x + 4)

/-- 
The minimum value of the area of the circle given by x^2 + y^2 = b - a where all zeros of 
F(x) = f(x + 4) lie in the interval (a, b) with a, b ∈ ℤ and a < b 
is π.
-/
theorem min_area_circle (a b : ℤ) (h₁ : ∀ x ∈ Ioo (0 : ℝ) 1, F x = 0 → (↑a : ℝ) < x ∧ x < (↑b : ℝ)) (h₂ : a < b) :
  (b - a : ℝ) = 1 → π =
  real.pi :=
by sorry

end min_area_circle_l143_143830


namespace total_subjects_is_41_l143_143193

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end total_subjects_is_41_l143_143193


namespace log_sum_of_geometric_sequence_l143_143848

noncomputable theory

variable (a : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n+1) = a n * r

def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

theorem log_sum_of_geometric_sequence 
  (ha : is_geometric_sequence a) 
  (hp : all_positive a)
  (h : a 1007 * a 1012 + a 1008 * a 1011 = 18) :
  (finset.range 2018).sum (λ n, real.logb 3 (a n)) = 2018 :=
by
sorry

end log_sum_of_geometric_sequence_l143_143848


namespace number_equal_to_its_opposite_is_zero_l143_143726

theorem number_equal_to_its_opposite_is_zero :
  ∀ (x : ℝ), x = -x → x = 0 :=
by
  assume x hx
  sorry

end number_equal_to_its_opposite_is_zero_l143_143726


namespace find_pairs_of_integers_l143_143398

theorem find_pairs_of_integers
  (n p : ℕ) (h_p_prime : Prime p) (h_n_pos : 0 < n) 
  (h_neg_p_prime : Prime (-p)) (h_neg_n_le_2p : -n ≤ 2 * p)
  (h_divisibility : n ^ (p - 1) ∣ -((p - 1) ^ n) + 1) : 
  (n = 1 ∧ Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by
  sorry

end find_pairs_of_integers_l143_143398


namespace inverse_of_expr_l143_143166

-- Define the imaginary unit and its properties
def imaginary_unit (j : ℂ) : Prop := j^2 = -1

-- Define the expression we want to find the inverse for
def expr (j : ℂ) := (3 * j - (1 / 3) * j⁻¹)

-- Statement asserting the inverse of the expression
theorem inverse_of_expr (j : ℂ) (h : imaginary_unit j) :
  (expr j)⁻¹ = - (3 * j / 10) :=
sorry

end inverse_of_expr_l143_143166


namespace cos_x_value_f_B_range_l143_143317

-- Definition of vectors m and n
def m (x : ℝ) := (Real.cos (x / 2), -1 : ℝ)
def n (x : ℝ) := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)

-- Definition of the function f
def f (x : ℝ) := (m x).1 * (n x).1 + (m x).2 * (n x).2 + 1

-- Prove the value of cos x given conditions
theorem cos_x_value (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 2) (h2 : f x = 11 / 10) : 
(Real.cos x = (4 * Real.sqrt 3 - 3) / 10) :=
sorry

-- Range of f(B)
theorem f_B_range (a b c : ℝ) (A B : ℝ) (h : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) :
(0 < f B ∧ f B ≤ 1 / 2) :=
sorry

end cos_x_value_f_B_range_l143_143317


namespace region_area_l143_143700

def Rhombus := { A B C D : Point // 
  distance A B = distance B C ∧
  distance B C = distance C D ∧
  distance C D = distance D A ∧
  (∠ B A D).degrees = 150 ∨ (∠ B C D).degrees = 150 ∧
  distance A B = 3 }

theorem region_area (R : Set Point) (ABC : Rhombus)
  (Hclose : ∀ P ∈ R, distance P ABC.B < min (distance P ABC.A) (distance P ABC.C) (distance P ABC.D)):
  area R = 0.2 := 
sorry

end region_area_l143_143700


namespace rationalize_sqrt_fraction_l143_143626

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l143_143626


namespace uniform_convergence_to_square_l143_143152

variable (A : ℝ) (y₁ : ℝ → ℝ) (y_n : ℕ → ℝ → ℝ)
variable (y₁_cont : ∀ x ∈ set.Icc 0 A, continuous_at y₁ x)
variable (y₁_pos : ∀ x ∈ set.Icc 0 A, 0 < y₁ x)
variable (recurrence : ∀ (n : ℕ) (x ∈ set.Icc 0 A), y_n (n + 1) x = 2 * ∫ t in 0..x, sqrt (y_n n t))

theorem uniform_convergence_to_square :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, ∀ x ∈ set.Icc 0 A, abs (y_n n x - x^2) < ε := 
sorry

end uniform_convergence_to_square_l143_143152


namespace number_of_correct_propositions_l143_143915

noncomputable def arithmetic_sequence (a n : ℕ → ℝ) := ∃ d, ∀ n, a (n + 1) = a n + d
noncomputable def geometric_sequence (a n : ℕ → ℝ) := ∃ r, ∀ n, a (n + 1) = a n * r

def sum_seq (a : ℕ → ℝ) (n : ℕ) := ∑ i in range (n + 1), a i

noncomputable def seq_prop_1 (a : ℕ → ℝ) :=
  (arithmetic_sequence a) →
  collinear [(10 : ℕ, sum_seq a 10 / 10), 
             (100 : ℕ, sum_seq a 100 / 100), 
             (110 : ℕ, sum_seq a 110 / 110)]
 
noncomputable def seq_prop_2 (a : ℕ → ℝ) :=
  (geometric_sequence a) →
  ∀ m : ℕ+, geometric_sequence (λ n, [sum_seq a m, 
                                       sum_seq a (2 * m) - sum_seq a m, 
                                       sum_seq a (3 * m) - sum_seq a (2 * m)].nth n.succ)

noncomputable def seq_prop_3 (a : ℕ → ℝ) :=
  (geometric_sequence a) →
  (∀ n : ℕ+, (n, sum_seq a n) ∈ set_of (λ (p : ℕ × ℝ), ∃ b r, b ≠ 0 ∧ b ≠ 1 ∧ p.snd = b ^ p.fst + r)) →
  ∃ (r : ℝ), r = -1

noncomputable def seq_prop_4 (a : ℕ → ℝ) :=
  (a 1 = 2) → 
  (∀ n : ℕ, a (n + 1) - a n = 2 ^ n) →
  ∀ n, sum_seq a n = 2^(n+1) - 2

theorem number_of_correct_propositions :
  (∃ a1 a2 a3 a4 : ℕ → ℝ, seq_prop_1 a1 ∧ seq_prop_2 a2 ∧ seq_prop_3 a3 ∧ seq_prop_4 a4)
  → 3 = 3 :=
by sorry

end number_of_correct_propositions_l143_143915


namespace rationalize_sqrt_fraction_denom_l143_143607

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l143_143607


namespace ridges_per_record_is_60_l143_143737

-- Define the given conditions
def cases : Nat := 4
def shelves_per_case : Nat := 3
def records_per_shelf : Nat := 20
def full_rate : Float := 0.60
def total_ridges : Nat := 8640

-- Define the total records and actual number of records
def total_records : Nat := cases * shelves_per_case * records_per_shelf
def actual_records : Nat := Float.toNat (full_rate * total_records.toFloat)

-- Lean statement to prove ridges per record is 60
theorem ridges_per_record_is_60 : (total_ridges / actual_records) = 60 := by
  sorry

end ridges_per_record_is_60_l143_143737


namespace event_probabilities_equal_l143_143516

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l143_143516


namespace min_abs_diff_l143_143489

theorem min_abs_diff
  (x y : ℝ)
  (h : log 4 (x + 2 * y) + log 4 (x - 2 * y) = 1) :
  ∃ x y, x > 0 ∧ y > 0 ∧ ( |x| - |y| = sqrt 3) :=
  sorry

end min_abs_diff_l143_143489


namespace find_triples_l143_143412

theorem find_triples (a b n : ℕ) (m : ℕ) (A B C : ℕ) (h : a^3 = b^2 + 2^n) :
  (∃ m ≥ 0, (a, b, n) = (2^(2*m) * A, 2^(3*m) * B, C + 6*m) ∧
  (A, B, C) ∈ {(2, 2, 2), (3, 5, 1), (5, 11, 2)}) :=
begin
  sorry -- Proof is omitted as per the instruction.
end

end find_triples_l143_143412


namespace heidi_paint_fraction_l143_143102

theorem heidi_paint_fraction (total_time : ℕ) (partial_time : ℕ) (fraction_painted : ℚ) (h1 : total_time = 60) (h2 : partial_time = 12) : fraction_painted = 1 / 5 :=
by
  rw [h1, h2]
  suffices (12 : ℚ) / 60 = 1 / 5 by
    exact this
  sorry

end heidi_paint_fraction_l143_143102


namespace max_value_sqrt_abc_l143_143156

theorem max_value_sqrt_abc :
  ∀ (a b : ℝ) (c : ℤ), 
  0 ≤ a → a ≤ 1 → 
  0 ≤ b → b ≤ 1 → 
  (c = 0 ∨ c = 1) →
  sqrt (a * b * c) + sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 :=
by
  intros a b c ha1 ha2 hb1 hb2 hc
  sorry

end max_value_sqrt_abc_l143_143156


namespace intercepts_of_line_l143_143840

theorem intercepts_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) :
  (∃ x_intercept : ℝ, x_intercept = 7 ∧ (4 * x_intercept + 7 * 0 = 28)) ∧
  (∃ y_intercept : ℝ, y_intercept = 4 ∧ (4 * 0 + 7 * y_intercept = 28)) :=
by
  sorry

end intercepts_of_line_l143_143840


namespace rhombus_diagonal_l143_143226

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 20) (h2 : area = 170) :
  (area = (d1 * d2) / 2) → d2 = 17 :=
by
  sorry

end rhombus_diagonal_l143_143226


namespace length_cd_l143_143246

noncomputable def volume_of_points_within_4_units_of_cd (L : ℝ) : ℝ :=
  let radius := 4
  let volume_hemispheres := 2 * (4 / 3) * π * radius^3 / 2
  let volume_cylinder := π * radius^2 * L
  volume_hemispheres + volume_cylinder

theorem length_cd (L : ℝ) (h : volume_of_points_within_4_units_of_cd L = 448 * π) : L = 68 / 3 := 
by 
  sorry

end length_cd_l143_143246


namespace find_treasure_in_box_2_l143_143978

def box_number := {n : ℕ // 1 ≤ n ∧ n ≤ 5}

def is_cedar (n : box_number) : Prop :=
  n = 1 ∨ n = 4 ∨ n = 5

def is_sandalwood (n : box_number) : Prop :=
  n = 2 ∨ n = 3

def statement (n : box_number) : Prop :=
  match n with
  | 1 => (treasure 1 ∨ treasure 4)
  | 2 => treasure 1
  | 3 => (treasure 3 ∨ treasure 5)
  | 4 => ¬(treasure 1 ∨ treasure 2 ∨ treasure 3 ∨ treasure 4)
  | 5 => ∀ j, j ≠ 5 → ¬ statement j

def false_count_equals : Prop :=
  ∑ i in {1, 4, 5}.to_finset, (if statement i then 0 else 1) =
  ∑ i in {2, 3}.to_finset, (if statement i then 0 else 1)

def treasure (n : box_number) : Prop

theorem find_treasure_in_box_2 : treasure 2 :=
by {
  -- creating the necessary structures
  sorry
}

end find_treasure_in_box_2_l143_143978


namespace avg_temperature_correct_l143_143402

theorem avg_temperature_correct :
  let week1 := [55, 62, 58, 65, 54, 60, 56]
  let week2 := [70, 74, 71, 77, 64, 68, 72]
  let week3 := [82, 85, 89, 73, 65, 63, 67]
  let week4 := [75, 72, 60, 57, 50, 55, 58]
  let additional := [69, 67, 70]
  let total_temperatures := week1 ++ week2 ++ week3 ++ week4 ++ additional
  let total_sum := List.sum total_temperatures
  let total_days := total_temperatures.length
  let avg_temp := total_sum / total_days
  avg_temp ≈ 66.55 :=
by
  sorry

end avg_temperature_correct_l143_143402


namespace ratio_CD_BD_l143_143953

open_locale classical

variables 
  (A B C D E T : Type) 
  [metric_space A] [add_comm_monoid B] [module ℝ B] 
  [metric_space C] [add_comm_monoid D] [module ℝ D]
  [metric_space E] [add_comm_monoid F] [module ℝ E]
  [metric_space T] [add_comm_monoid U] [module ℝ T]

variables
  {A B C : Point}
  {D : Point}
  {E : Point}
  {T : Point}

variables (AD : segment A D) (BE : segment B E) (BC : segment B C) (AC : segment A C) (CD : segment C D) (BD : segment B D)

noncomputable def ratio_AT_DT := 2
noncomputable def ratio_BT_ET := 3

theorem ratio_CD_BD :
  let at_dt : ℝ := 2 in
  let bt_et : ℝ := 3 in
  at_dt = ratio_AT_DT →
  bt_et = ratio_BT_ET →
  ∃ (x y : Point), 
  (x ∈ line_from_to A D) ∧ 
  (y ∈ line_from_to B E) ∧ 
  T = intersection_point (line_from_to A D) (line_from_to B E) → 
  ∃ (ratio_CD_BD : ℝ), ratio_CD_BD = 3/5 :=
by {
  sorry
}

end ratio_CD_BD_l143_143953


namespace work_days_l143_143312

theorem work_days (d_b d_c total_earnings w_c d_a : ℕ) (hb : d_b = 9) (hc : d_c = 4) (he : total_earnings = 1702) (hwc : w_c = 115) (h_ratio : ∃ (x : ℕ), 5 * x = w_c):
  3 * h_ratio.some * d_a + 4 * h_ratio.some * d_b + w_c * d_c = total_earnings → d_a = 6 := 
by
  sorry

end work_days_l143_143312


namespace rationalize_sqrt_fraction_l143_143673

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l143_143673
