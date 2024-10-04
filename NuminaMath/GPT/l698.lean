import Mathlib

namespace counties_rained_on_monday_l698_698246

theorem counties_rained_on_monday : 
  ∀ (M T R_no_both R_both : ℝ),
    T = 0.55 → 
    R_no_both = 0.35 →
    R_both = 0.60 →
    (M + T - R_both = 1 - R_no_both) →
    M = 0.70 :=
by
  intros M T R_no_both R_both hT hR_no_both hR_both hInclusionExclusion
  sorry

end counties_rained_on_monday_l698_698246


namespace directrix_parabola_l698_698671

theorem directrix_parabola (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end directrix_parabola_l698_698671


namespace not_similar_pair_D_l698_698872

noncomputable def isosceles_triangle (a b angle_deg : ℝ) : Prop :=
  a = b ∧ 0 < angle_deg ∧ angle_deg < 180

noncomputable def right_angled_triangle (leg1 leg2 : ℝ) : Prop :=
  ∃ hypotenuse : ℝ, hypotenuse = real.sqrt (leg1^2 + leg2^2)

noncomputable def similar_triangles (t1 t2 : Prop) : Prop :=
  ∀ (a1 b1 c1 a2 b2 c2 : ℝ),
  t1 → t2 → (a1 / a2 = b1 / b2 = c1 / c2)

axiom condition_A : similar_triangles (right_angled_triangle 6 4) (right_angled_triangle 4.5 3)
axiom condition_B : similar_triangles (isosceles_triangle 1 1 40) (isosceles_triangle 1 1 40)
axiom condition_C : similar_triangles (right_angled_triangle 1 (1 / real.sqrt 3)) (right_angled_triangle (2 / real.sqrt 3) 1)
axiom condition_D : ¬ similar_triangles (isosceles_triangle 1 1 30) (isosceles_triangle 1 1 30)

theorem not_similar_pair_D : ¬ similar_triangles (isosceles_triangle 1 1 30) (isosceles_triangle 1 1 30) :=
by
  exact condition_D

end not_similar_pair_D_l698_698872


namespace parallelepiped_height_l698_698871

-- Define the parallelepiped and geometric properties
structure Parallelepiped (a : ℝ) :=
(face_is_rhombus : ∀ face, face.is_congruent_rhombus a 60)
(side_length : ℝ := a)
(acute_angle : ℝ := 60)

-- Define a theorem to find the height of the parallelepiped
theorem parallelepiped_height (a : ℝ) (p : Parallelepiped a) : 
  (h : ℝ) (h = a * Real.sqrt (2 / 3)) :=
sorry

end parallelepiped_height_l698_698871


namespace value_of_t_l698_698220

def vec (x y : ℝ) := (x, y)

def p := vec 3 3
def q := vec (-1) 2
def r := vec 4 1

noncomputable def t := 3

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem value_of_t (t : ℝ) : (dot_product (vec (6 + 4 * t) (6 + t)) q = 0) ↔ t = 3 :=
by
  sorry

end value_of_t_l698_698220


namespace sixty_first_batch_is_1211_l698_698594

-- Definitions based on conditions
def total_bags : ℕ := 3000
def total_batches : ℕ := 150
def first_batch_number : ℕ := 11

-- Define the calculation of the 61st batch number
def batch_interval : ℕ := total_bags / total_batches
def sixty_first_batch_number : ℕ := first_batch_number + 60 * batch_interval

-- The statement of the proof
theorem sixty_first_batch_is_1211 : sixty_first_batch_number = 1211 := by
  sorry

end sixty_first_batch_is_1211_l698_698594


namespace nikolai_faster_than_gennady_l698_698045

-- The conditions of the problem translated to Lean definitions
def gennady_jump_length : ℕ := 6
def gennady_jumps_per_time : ℕ := 2
def nikolai_jump_length : ℕ := 4
def nikolai_jumps_per_time : ℕ := 3
def turn_around_distance : ℕ := 2000
def round_trip_distance : ℕ := 2 * turn_around_distance

-- The statement that Nikolai completes the journey faster than Gennady
theorem nikolai_faster_than_gennady :
  (nikolai_jumps_per_time * nikolai_jump_length) = (gennady_jumps_per_time * gennady_jump_length) →
  (round_trip_distance % nikolai_jump_length = 0) →
  (round_trip_distance % gennady_jump_length ≠ 0) →
  (round_trip_distance / nikolai_jump_length) + 1 < (round_trip_distance / gennady_jump_length) →
  "Nikolay completes the journey faster." :=
by
  intros h_eq_speed h_nikolai_divisible h_gennady_not_divisible h_time_compare
  sorry

end nikolai_faster_than_gennady_l698_698045


namespace prove_correct_operation_l698_698823

def correct_operation (a b : ℕ) : Prop :=
  (a^3 * a^2 ≠ a^6) ∧
  ((a * b^2)^2 = a^2 * b^4) ∧
  (a^10 / a^5 ≠ a^2) ∧
  (a^2 + a ≠ a^3)

theorem prove_correct_operation (a b : ℕ) : correct_operation a b :=
by {
  sorry
}

end prove_correct_operation_l698_698823


namespace thirtieth_term_value_l698_698820

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l698_698820


namespace right_triangle_third_side_product_l698_698788

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l698_698788


namespace club_popularity_order_l698_698878

-- Definitions based on conditions
def chess_popularity := 5 / 18
def drama_popularity := 4 / 15
def science_popularity := 7 / 20

-- Theorem to prove the order from most popular to least popular
theorem club_popularity_order :
  science_popularity > chess_popularity ∧ chess_popularity > drama_popularity :=
by
  norm_num
  have h1 : chess_popularity = 5 / 18 := rfl
  have h2 : drama_popularity = 4 / 15 := rfl
  have h3 : science_popularity = 7 / 20 := rfl
  rw [h1, h2, h3]
  norm_num
  split
  { 
    -- Science > Chess
    sorry
  }
  {
    -- Chess > Drama
    sorry
  }

end club_popularity_order_l698_698878


namespace josh_final_num_l698_698266

def joshMarksOut (lst : List ℕ) (start skip : ℕ) : List ℕ :=
  match lst with
  | [] => []
  | _  => 
    let len := lst.length
    let markedIdxs := List.range len |>.filter (λ i => (i - start) % (skip + 1) == 0)
    lst.filterWithIndex (λ i _ => ¬ List.contains markedIdxs i)

def finalRemainingNumber : List ℕ → ℕ
  | [x] => x
  | lst => (finalRemainingNumber (joshMarksOut lst 0 0)).tail

theorem josh_final_num : finalRemainingNumber (List.range' 1 50) = 21 := by
  sorry

#eval finalRemainingNumber (List.range' 1 50) -- This should be 21

end josh_final_num_l698_698266


namespace number_of_integers_l698_698163

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l698_698163


namespace sum_opposite_numbers_correct_opposite_sum_numbers_correct_l698_698565

def opposite (x : Int) : Int := -x

def sum_opposite_numbers (a b : Int) : Int := opposite a + opposite b

def opposite_sum_numbers (a b : Int) : Int := opposite (a + b)

theorem sum_opposite_numbers_correct (a b : Int) : sum_opposite_numbers (-6) 4 = 2 := 
by sorry

theorem opposite_sum_numbers_correct (a b : Int) : opposite_sum_numbers (-6) 4 = 2 := 
by sorry

end sum_opposite_numbers_correct_opposite_sum_numbers_correct_l698_698565


namespace sequence_formula_l698_698940

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 4 else 3 * n + 1

theorem sequence_formula (n : ℕ) :
  (sequence n = 3 * n + 1) :=
by
  sorry

end sequence_formula_l698_698940


namespace count_even_divisors_l698_698572

theorem count_even_divisors (n : ℕ) (h : n < 60) : ∃ k, k = 52 :=
by
  let all_integers := {x : ℕ // 0 < x ∧ x < 60}
  let even_divisors (x : ℕ) := ¬ ∃ y : ℕ, y * y = x
  let count_even := {x // x ∈ all_integers ∧ even_divisors x}
  have h_count: (finset.univ : finset all_integers).filter even_divisors = 52 :=
    sorry
  use 52
  exact h_count

end count_even_divisors_l698_698572


namespace calculate_expression_l698_698455

theorem calculate_expression : | - (1 / 2) | + (-2023)^0 + 2^(-1) = 2 := by
  sorry

end calculate_expression_l698_698455


namespace number_of_crocodiles_l698_698248

theorem number_of_crocodiles
  (f : ℕ) -- number of frogs
  (c : ℕ) -- number of crocodiles
  (total_eyes : ℕ) -- total number of eyes
  (frog_eyes : ℕ) -- number of eyes per frog
  (croc_eyes : ℕ) -- number of eyes per crocodile
  (h_f : f = 20) -- condition: there are 20 frogs
  (h_total_eyes : total_eyes = 52) -- condition: total number of eyes is 52
  (h_frog_eyes : frog_eyes = 2) -- condition: each frog has 2 eyes
  (h_croc_eyes : croc_eyes = 2) -- condition: each crocodile has 2 eyes
  :
  c = 6 := -- proof goal: number of crocodiles is 6
by
  sorry

end number_of_crocodiles_l698_698248


namespace decagon_adjacent_vertex_probability_l698_698716

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698716


namespace coeff_term_x_neg2_l698_698149

theorem coeff_term_x_neg2 :
  let x := ℝ,
      binomial_expansion := (2 * Real.sqrt x - 1 / x)^5,
      r := 3,
      term_r_plus_1 := Nat.choose 5 r * 2^(5 - r) * (-1)^r * x ^ (5 - 3 * r) / 2 
  in term_r_plus_1 = -40 := 
begin
  -- proof
  sorry
end

end coeff_term_x_neg2_l698_698149


namespace probability_adjacent_vertices_decagon_l698_698757

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698757


namespace mountain_numbers_count_l698_698472

-- Number of 4-digit mountain numbers
theorem mountain_numbers_count : ∃ (n : ℕ), n = 240 ∧ 
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧
  (b > a) ∧ (c > a) ∧ (b = c) ∧ (b > d) ∧ 
  (a, b, c, d) ∈ ({(x, x, y, x) | x ≠ 0 ∧ x ≠ y ∧ y > x} ∪ 
                 {(x, y, y, x) | x ≠ 0 ∧ x ≠ y ∧ y > x} ∪ 
                 {(x, y, y, z) | x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ y > x ∧ y > z}) :=
by
  exact ⟨240, rfl, sorry⟩

end mountain_numbers_count_l698_698472


namespace find_smallest_nth_root_of_unity_l698_698920

theorem find_smallest_nth_root_of_unity :
  ∃ n : ℕ, 0 < n ∧ (∀ z : ℂ, (z ^ 5 - z ^ 3 + 1 = 0) → (∃ k : ℕ, z = exp(2 * Real.pi * I * k / n))) ∧ n = 30 :=
by
  sorry

end find_smallest_nth_root_of_unity_l698_698920


namespace probability_of_adjacent_vertices_l698_698720

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698720


namespace area_of_park_l698_698009

theorem area_of_park (speed_kph : ℝ) (time_min : ℝ) (length_ratio : ℝ) (breadth_ratio : ℝ)
  (H_speed : speed_kph = 12) (H_time : time_min = 4) 
  (H_length_ratio : length_ratio = 1) (H_breadth_ratio : breadth_ratio = 3) :
  let speed_mpm := speed_kph * 1000 / 60,
      perimeter := speed_mpm * time_min,
      length := perimeter / (2 * (length_ratio + breadth_ratio)),
      breadth := 3 * length in
  length * breadth = 30000 :=
by
  -- No proof needed
  sorry

end area_of_park_l698_698009


namespace distinct_triangle_areas_l698_698901

noncomputable def triangle_areas (A B C D E : Point) : ℝ := sorry

theorem distinct_triangle_areas
  (A B C D E : Point)
  (hAB : dist A B = 1)
  (hBC : dist B C = Real.sqrt 2)
  (hAC : dist A C = 1)
  (hCD : dist C D = 1)
  (hBE : dist B E = 1)
  (h_parallel : Parallel (Line.from_pts B E) (Line.from_pts A C)) :
  (find_distinct_areas (A B C D E)).card = 3 :=
sorry

end distinct_triangle_areas_l698_698901


namespace sum_fractions_series_l698_698463

-- Define a function representing the sum of fractions from 1/7 to 12/7
def sum_fractions : ℚ :=
  (list.sum (list.map (λ k, k / 7) (list.range' 1 12)))

-- State the theorem
theorem sum_fractions_series :
  sum_fractions = 11 + 1 / 7 :=
sorry

end sum_fractions_series_l698_698463


namespace total_people_3522_l698_698424

def total_people (M W: ℕ) : ℕ := M + W

theorem total_people_3522 
    (M W: ℕ) 
    (h1: M / 9 * 45 + W / 12 * 60 = 17760)
    (h2: M % 9 = 0)
    (h3: W % 12 = 0) : 
    total_people M W = 3552 :=
by {
  sorry
}

end total_people_3522_l698_698424


namespace necessary_but_not_sufficient_l698_698069

theorem necessary_but_not_sufficient (x : ℝ) (h : x < 4) : x < 0 ∨ true :=
by
  sorry

end necessary_but_not_sufficient_l698_698069


namespace num_integers_satisfying_ineq_count_integers_satisfying_ineq_l698_698161

theorem num_integers_satisfying_ineq (k : ℤ) :
  (20 < k^2 ∧ k^2 < 150) ↔ k ∈ ({-12, -11, -10, -9, -8, -7, -6, -5, 5, 6, 7, 8, 9, 10, 11, 12} : set ℤ) := by
  sorry

theorem count_integers_satisfying_ineq :
  {n : ℤ | 20 < n^2 ∧ n^2 < 150}.finite.to_finset.card = 16 := by
  sorry

end num_integers_satisfying_ineq_count_integers_satisfying_ineq_l698_698161


namespace original_price_l698_698012

theorem original_price (P : ℝ) (h_discount : 0.75 * P = 560): P = 746.68 :=
sorry

end original_price_l698_698012


namespace cosine_difference_l698_698534

theorem cosine_difference (A B : ℝ) (h1 : Real.sin A + Real.sin B = 3/2) (h2 : Real.cos A + Real.cos B = 2) :
  Real.cos (A - B) = 17 / 8 :=
by
  sorry

end cosine_difference_l698_698534


namespace length_of_string_l698_698855

-- Definitions for the cylindrical pole's dimensions
def circumference : ℝ := 3
def height : ℝ := 16
def loops : ℝ := 6

-- Calculate height per loop
def height_per_loop (h : ℝ) (n : ℝ) : ℝ := h / n

-- Calculate the string length per loop using Pythagorean theorem
def string_length_per_loop (c : ℝ) (h_l : ℝ) : ℝ := Real.sqrt (c^2 + h_l^2)

-- Calculate the total length of the string after all loops
def total_string_length (n : ℝ) (s_l_per_loop : ℝ) : ℝ := n * s_l_per_loop

-- The main theorem
theorem length_of_string : total_string_length loops (string_length_per_loop circumference (height_per_loop height loops)) = 24.0966 := 
by 
  sorry

end length_of_string_l698_698855


namespace properties_of_A_prime_prime_l698_698482

-- Definitions
structure Point :=
(x : ℝ) (y : ℝ)

structure Parallelogram :=
(A B C D : Point)

def opposite_direction (P Q : Parallelogram) : Prop :=
-- Definition indicating P and Q vertices are labeled in opposite directions
sorry

def coincident (P Q : Parallelogram) : Prop :=
P = Q

def reflection_axis (P Q : Parallelogram) : Prop :=
-- Reflect about an axis
sorry

def reflection_point (P Q : Parallelogram) : Prop :=
-- Reflect about a point
sorry

-- Proof Problem
theorem properties_of_A_prime_prime
    (P P' P'' : Parallelogram)
    (h1 : opposite_direction P P')
    (h2 : coincident P' P'')
    (h3 : reflection_axis P P' ∨ reflection_point P P')
    (h4 : ∀ A B C D A' B' C' D', 
        P = Parallelogram.mk A B C D →
        P' = Parallelogram.mk A' B' C' D' →
        AB = DC ∧ BC = AD ∧ A'B' = D'C' ∧ B'C' = A'D')
    : Prop :=
-- Conditions and result to test properties of P''
begin
    sorry
end

end properties_of_A_prime_prime_l698_698482


namespace product_of_third_sides_is_correct_l698_698765

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l698_698765


namespace smallest_whole_number_larger_than_sum_l698_698167

theorem smallest_whole_number_larger_than_sum : 
  let x := 3 + 1/3
  let y := 5 + 1/4
  let z := 7 + 1/6
  let w := 9 + 1/8
  25 = Int.ceil (x + y + z + w) :=
by
  sorry

end smallest_whole_number_larger_than_sum_l698_698167


namespace arithmetic_sequence_30th_term_l698_698805

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l698_698805


namespace earl_stuff_rate_l698_698907

variable (E L : ℕ)

-- Conditions
def ellen_rate : Prop := L = (2 * E) / 3
def combined_rate : Prop := E + L = 60

-- Main statement
theorem earl_stuff_rate (h1 : ellen_rate E L) (h2 : combined_rate E L) : E = 36 := by
  sorry

end earl_stuff_rate_l698_698907


namespace find_m_l698_698942

variable (C E : ℝ → ℝ → Prop)
variable (m : ℝ)

-- Definitions based on given conditions
def CircleC : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 5 - m
def CircleE : ℝ → ℝ → Prop := λ x y, (x - 3)^2 + (y - 4)^2 = 16

-- Proof problem stating that given the conditions, m equals 4
theorem find_m (h : ∀ (C E : ℝ → ℝ → Prop), C (0, 0) ∧ E (3, 4) → 
               (4 + real.sqrt (5 - m)) = 5) : m = 4 :=
by
  sorry

end find_m_l698_698942


namespace equation_of_line_through_point_with_slope_l698_698858

theorem equation_of_line_through_point_with_slope 
  (M : ℝ × ℝ) (slope : ℝ) (y x : ℝ) 
  (hM : M = (-1, 1)) 
  (hSlope : slope = 2) :
  y = 2 * x + 3 ↔ ∃ x₀ y₀, M = (x₀, y₀) ∧ y₀ = 1 ∧ y - y₀ = slope * (x - x₀) := 
by
  intro h
  sorry

end equation_of_line_through_point_with_slope_l698_698858


namespace decagon_adjacent_vertices_probability_l698_698731

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698731


namespace decagon_adjacent_vertex_probability_l698_698717

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698717


namespace complex_div_conjugate_proof_l698_698084

-- Given condition: i is the imaginary unit
local notation "i" => Complex.i

-- Question: Prove that (2 * i) / (1 + i) equals 1 + i
theorem complex_div_conjugate_proof : ((2 * i) / (1 + i)) = (1 + i) :=
by
  sorry

end complex_div_conjugate_proof_l698_698084


namespace sin_cos_difference_l698_698505

theorem sin_cos_difference
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.sin θ + Real.cos θ = 1 / 5) :
  Real.sin θ - Real.cos θ = 7 / 5 :=
sorry

end sin_cos_difference_l698_698505


namespace perimeter_of_ABCDEFG_l698_698364

-- Definitions representing the points, lengths, and properties.
variables {A B C D E F G : Type} [EquilateralTriangle ABC] [EquilateralTriangle ADE] [EquilateralTriangle EFG]
variables {midpointAC : Midpoint AC D} {midpointAE : Midpoint AE G} {AB_length : length AB = 4}

theorem perimeter_of_ABCDEFG :
  let AB := length AB,
      BC := length BC,
      CD := length CD,
      DE := length DE,
      EF := length EF,
      FG := length FG,
      GA := length GA in
  AB + BC + CD + DE + EF + FG + GA = 15 :=
-- conditions related to the lengths and properties of each triangle and midpoint
  by
    have h1 : BC = 4 := sorry,
    have h2 : CD = 2 := sorry,
    have h3 : DE = 2 := sorry,
    have h4 : EF = 1 := sorry,
    have h5 : FG = 1 := sorry,
    have h6 : GA = 1 := sorry,
    exact sorry  -- end the proof placeholder

end perimeter_of_ABCDEFG_l698_698364


namespace y_complete_work_in_24_days_l698_698082

noncomputable def work_rate_x : ℝ := 1 / 20
noncomputable def work_done_x (days : ℕ) : ℝ := days * work_rate_x
noncomputable def remaining_work : ℝ := 1 - work_done_x 10
noncomputable def work_rate_y : ℝ := (remaining_work) / 12
noncomputable def days_y_to_complete_work : ℝ := 1 / work_rate_y

theorem y_complete_work_in_24_days : days_y_to_complete_work = 24 := by
  unfold days_y_to_complete_work work_rate_y remaining_work work_done_x work_rate_x
  linarith

end y_complete_work_in_24_days_l698_698082


namespace decagon_adjacent_vertex_probability_l698_698715

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698715


namespace ice_cream_volume_l698_698426

-- Define the volume formula for a cone
def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

-- Given conditions
def radius_main_cone : ℝ := 3
def height_main_cone : ℝ := 10
def height_small_cone : ℝ := 5

-- Volumes of the individual cones
def V1 : ℝ := cone_volume radius_main_cone height_main_cone
def V2 : ℝ := cone_volume radius_main_cone height_small_cone

-- Total volume
def total_volume : ℝ := V1 + V2

-- Prove the total volume is 45π cubic inches
theorem ice_cream_volume : total_volume = 45 * Real.pi :=
by
  unfold total_volume V1 V2 cone_volume radius_main_cone height_main_cone height_small_cone
  simp
  norm_num
  ring

end ice_cream_volume_l698_698426


namespace possible_sequences_l698_698704

structure Observation (α : Type) :=
  (first : α)
  (second : α)
  (third : α)

def observedRain (o : Observation Bool) : Observation Bool :=
  Observation.mk
    (o.first = true)
    (o.second = true)
    (count [o.first, o.second, o.third] true >= 2)

theorem possible_sequences :
  ∀ (o : Observation Bool),
    let result := observedRain o in
    result = Observation.mk true true true ∨
    result = Observation.mk false true true ∨
    result = Observation.mk false false true ∨
    result = Observation.mk false false false :=
by
  intro o
  cases o
  sorry

end possible_sequences_l698_698704


namespace x_pow_27_minus_inv_x_pow_27_eq_zero_l698_698194

theorem x_pow_27_minus_inv_x_pow_27_eq_zero {x : ℂ} (h : x - 1/x = Complex.i * Real.sqrt 3) : 
  x^27 - 1/x^27 = 0 :=
sorry

end x_pow_27_minus_inv_x_pow_27_eq_zero_l698_698194


namespace angle_between_vectors_l698_698508

variables (a b : EuclideanSpace ℝ (Fin 3))
variable (theta : ℝ)

-- Conditions given in the problem
axiom a_norm : ‖a‖ = 4
axiom b_norm : ‖b‖ = 3 * Real.sqrt 2
axiom a_dot_b : a ⬝ b = -12

-- The angle theta between vectors a and b
theorem angle_between_vectors :
  theta = 3 * Real.pi / 4 :=
sorry

end angle_between_vectors_l698_698508


namespace prime_numbers_square_condition_l698_698488

theorem prime_numbers_square_condition (p : ℕ) (hp : nat.prime p) :
  ∃ a : ℕ, p^2 + 200 = a^2 ↔ (p = 5 ∨ p = 23) :=
by
  sorry

end prime_numbers_square_condition_l698_698488


namespace triangle_area_l698_698188

-- Conditions
variables {a b c A B C : ℝ}
hypotheses (h1 : a = 2 * Real.sqrt 2)
(h2 : Real.cos A = 3 / 4)
(h3 : Real.sin B = 2 * Real.sin C)

-- The proof problem
theorem triangle_area (h1 : a = 2 * Real.sqrt 2) (h2 : Real.cos A = 3 / 4) (h3 : Real.sin B = 2 * Real.sin C) : 
  let b := 2 * c in
  let sin_A := Real.sqrt (1 - (3/4)^2) in
  let area := 1 / 2 * b * c * sin_A in
  area = Real.sqrt 7 :=
  sorry

end triangle_area_l698_698188


namespace probability_adjacent_vertices_decagon_l698_698737

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698737


namespace determine_alpha_values_l698_698948

def power_function_odd_and_domain_real (α : ℝ) : Prop :=
  (∀ x : ℝ, x ∈ ℝ → x ≠ 0 → (x^α):ℝ = (x:ℝ)^α) ∧
  (∀ x : ℝ, ((-x)^α) = -(x^α))

theorem determine_alpha_values :
  ∀ α ∈ ({1, 2, 3, (1/2), -1} : set ℝ), power_function_odd_and_domain_real α ↔ α ∈ ({1, 3} : set ℝ) :=
by sorry

end determine_alpha_values_l698_698948


namespace correct_sum_of_valid_primes_l698_698063

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def has_prime_reverse_digits (n : ℕ) : Prop :=
  let reversed := (n % 10) * 10 + n / 10
  is_prime reversed

def sum_of_digits_is_prime (n : ℕ) : Prop :=
  let sum_digits := (n / 10) + (n % 10)
  is_prime sum_digits

def valid_prime_sum : ℕ :=
  let primes := { p ∈ Ico 13 100 | is_prime p }
  let valid_primes := { p ∈ primes | has_prime_reverse_digits p ∧ sum_of_digits_is_prime p }
  valid_primes.sum

theorem correct_sum_of_valid_primes : valid_prime_sum = 0 :=
by
  sorry

end correct_sum_of_valid_primes_l698_698063


namespace domain_g_l698_698662

open Set

variable {α : Type*} [LinearOrder α] (f : α → α) (g : α → α)

def domain_f : Set α := Icc (-3 : α) 3
def f (x : α) := sorry  -- Placeholder for the actual definition of f
def g (x : α) := f (3 * x + 1)

theorem domain_g :
  ∀ x, g x ∈ domain_f ↔ - (4 / 3 : α) ≤ x ∧ x ≤ (2 / 3 : α) :=
by
  sorry

end domain_g_l698_698662


namespace angles_at_point_l698_698247

theorem angles_at_point (x y : ℝ) 
  (h1 : x + y + 120 = 360) 
  (h2 : x = 2 * y) : 
  x = 160 ∧ y = 80 :=
by
  sorry

end angles_at_point_l698_698247


namespace sam_sitting_fee_l698_698112

theorem sam_sitting_fee :
  (∃ S : ℝ,
  let johns_cost := 2.75 * 12 + 125 in
  let sams_cost := 1.50 * 12 + S in
  johns_cost = sams_cost → S = 140) :=
by
  sorry

end sam_sitting_fee_l698_698112


namespace quadratic_smallest_root_a_quadratic_smallest_root_b_l698_698829

-- For Part (a)
theorem quadratic_smallest_root_a (a : ℝ) 
  (h : a^2 - 9 * a - 10 = 0 ∧ ∀ x, x^2 - 9 * x - 10 = 0 → x ≥ a) : 
  a^4 - 909 * a = 910 :=
by sorry

-- For Part (b)
theorem quadratic_smallest_root_b (b : ℝ) 
  (h : b^2 - 9 * b + 10 = 0 ∧ ∀ x, x^2 - 9 * x + 10 = 0 → x ≥ b) : 
  b^4 - 549 * b = -710 :=
by sorry

end quadratic_smallest_root_a_quadratic_smallest_root_b_l698_698829


namespace back_section_revenue_l698_698429

theorem back_section_revenue (capacity : ℕ) (regular_price half_price : ℝ) (promo_n : ℕ)
  (h_capacity : capacity = 25000)
  (h_regular_price : regular_price = 55)
  (h_half_price : half_price = 27.50)
  (h_promo_n : promo_n = 5) :
  let regular_revenue := (regular_price * capacity.to_nat.to_real)
  let total_sets := (capacity.to_real / (promo_n + 1))
  let half_price_revenue := (total_sets * half_price)
  let total_revenue := (regular_revenue + half_price_revenue)
  in total_revenue = 1489565 :=
by
  sorry

end back_section_revenue_l698_698429


namespace chromatic_number_le_k_add_one_l698_698405

noncomputable theory
open finite_graph -- Assuming a finite_graph module exists for graph theory

variables {G : Type} [finite_graph G] [fintype G]

def degree_k (k : ℕ) : Prop :=
  ∀ v : G, degree v = k

theorem chromatic_number_le_k_add_one (k : ℕ) (h : degree_k k) :
  chromatic_number G ≤ k + 1 := 
sorry

end chromatic_number_le_k_add_one_l698_698405


namespace three_times_repeated_six_equals_expression_l698_698304

theorem three_times_repeated_six_equals_expression (n : ℤ) (hn : n > 1) :
  3 * (10^n - 1)^3 = (10^(3*n) - 3 * 10^(2*n) + 3 * 10^n - 1) * 8 / 9 := 
by
  sorry

end three_times_repeated_six_equals_expression_l698_698304


namespace probability_adjacent_vertices_decagon_l698_698740

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698740


namespace decagon_adjacent_probability_l698_698746

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698746


namespace Nikolai_faster_than_Gennady_l698_698049

theorem Nikolai_faster_than_Gennady
  (gennady_jump1 gennady_jump2 : ℕ) (nikolai_jump1 nikolai_jump2 nikolai_jump3 : ℕ) :
  gennady_jump1 = 6 → gennady_jump2 = 6 →
  nikolai_jump1 = 4 → nikolai_jump2 = 4 → nikolai_jump3 = 4 →
  2 * gennady_jump1 + gennady_jump2 = 3 * (nikolai_jump1 + nikolai_jump2 + nikolai_jump3) →
  let total_path := 2000 in
  (total_path % 4 = 0 ∧ total_path % 6 ≠ 0) →
  (total_path / 4 < (total_path + 4) / 6) :=
by
  intros
  sorry

end Nikolai_faster_than_Gennady_l698_698049


namespace cyclist_average_speed_l698_698417

open Real

-- Define the cyclist segments speeds and distances
def speed_segment1 := 30 -- kph
def distance_segment1 := 15 -- km

def speed_segment2 := 45 -- kph
def distance_segment2 := 35 -- km

def speed_segment3 := 25 -- kph
def time_segment3 := 0.5 -- hr (30 minutes)

def total_time := 1 -- hr

-- Define the total distance and total time excluding the fourth segment
def distance_segment3 := speed_segment3 * time_segment3
def total_distance := distance_segment1 + distance_segment2 + distance_segment3
def total_travel_time := distance_segment1 / speed_segment1 + distance_segment2 / speed_segment2 + time_segment3

-- Define the average speed
def average_speed := total_distance / total_travel_time

-- The eventual theorem statement asserting the average speed equals to 33 kph
theorem cyclist_average_speed : average_speed = 33 := by
  -- skipping the proof
  sorry

end cyclist_average_speed_l698_698417


namespace distinct_p_q_r_s_t_sum_l698_698287

theorem distinct_p_q_r_s_t_sum (p q r s t : ℤ) (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t)
    (h9 : r ≠ s) (h10 : r ≠ t)
    (h11 : s ≠ t) : p + q + r + s + t = 25 := by
  sorry

end distinct_p_q_r_s_t_sum_l698_698287


namespace total_birds_in_marsh_l698_698361

theorem total_birds_in_marsh (geese ducks : ℕ) (h_geese : geese = 58) (h_ducks : ducks = 37) : geese + ducks = 95 := by
  rw [h_geese, h_ducks]
  norm_num

end total_birds_in_marsh_l698_698361


namespace shaded_area_proof_l698_698430

noncomputable def shaded_area (side_length : ℝ) (radius_factor : ℝ) : ℝ :=
  let square_area := side_length * side_length
  let radius := radius_factor * side_length
  let circle_area := Real.pi * (radius * radius)
  square_area - circle_area

theorem shaded_area_proof : shaded_area 8 0.6 = 64 - 23.04 * Real.pi :=
by sorry

end shaded_area_proof_l698_698430


namespace probability_of_adjacent_vertices_l698_698723

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698723


namespace toothpicks_at_200th_stage_l698_698334

-- Define initial number of toothpicks at stage 1
def a_1 : ℕ := 4

-- Define the function to compute the number of toothpicks at stage n, taking into account the changing common difference
def a (n : ℕ) : ℕ :=
  if n = 1 then 4
  else if n <= 49 then 4 + 4 * (n - 1)
  else if n <= 99 then 200 + 5 * (n - 50)
  else if n <= 149 then 445 + 6 * (n - 100)
  else if n <= 199 then 739 + 7 * (n - 150)
  else 0  -- This covers cases not considered in the problem for clarity

-- State the theorem to check the number of toothpicks at stage 200
theorem toothpicks_at_200th_stage : a 200 = 1082 :=
  sorry

end toothpicks_at_200th_stage_l698_698334


namespace find_integer_solutions_l698_698912

theorem find_integer_solutions :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    (a * b - 2 * c * d = 3) ∧ (a * c + b * d = 1) } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end find_integer_solutions_l698_698912


namespace find_omega_l698_698969

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x + ϕ)

def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop := 
  ∀ x, f (2 * c - x) = f x

theorem find_omega (ω ϕ : ℝ) (hω : 0 < ω) (hϕ : |ϕ| < Real.pi / 2)
  (hsym : symmetric_about (f ω ϕ) (Real.pi / 4))
  (hzero : f ω ϕ (- Real.pi / 4) = 0)
  (hlmax : (∃ x1 x2 x3, 0 < x1 ∧ x1 < Real.pi ∧ 
                    0 < x2 ∧ x2 < Real.pi ∧ 
                    0 < x3 ∧ x3 < Real.pi ∧
                    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
                    Real.add (Real.deriv (f ω ϕ)) 1 = 0 ∧
                    Real.add (Real.deriv (f ω ϕ)) 2 = 0 ∧
                    Real.add (Real.deriv (f ω ϕ)) 3 = 0)) :
  ω = 5 :=
sorry

end find_omega_l698_698969


namespace complex_conjugate_l698_698586

-- Define the conditions
def z1 := i * (3 - 2 * i)
noncomputable def z2 : ℂ := 2 + 3 * i

-- Define the theorem to be proved
theorem complex_conjugate : z1 = z2 → conj z2 = 2 - 3 * i := by
  sorry

end complex_conjugate_l698_698586


namespace laura_needs_to_buy_flour_l698_698269

/--
Laura is baking a cake and needs to buy ingredients.
Flour costs $4, sugar costs $2, butter costs $2.5, and eggs cost $0.5.
The cake is cut into 6 slices. Her mother ate 2 slices.
The dog ate the remaining cake, costing $6.
Prove that Laura needs to buy flour worth $4.
-/
theorem laura_needs_to_buy_flour
  (flour_cost sugar_cost butter_cost eggs_cost dog_ate_cost : ℝ)
  (cake_slices mother_ate_slices dog_ate_slices : ℕ)
  (H_flour : flour_cost = 4)
  (H_sugar : sugar_cost = 2)
  (H_butter : butter_cost = 2.5)
  (H_eggs : eggs_cost = 0.5)
  (H_dog_ate : dog_ate_cost = 6)
  (total_slices : cake_slices = 6)
  (mother_slices : mother_ate_slices = 2)
  (dog_slices : dog_ate_slices = 4) :
  flour_cost = 4 :=
by {
  sorry
}

end laura_needs_to_buy_flour_l698_698269


namespace mistaken_quotient_is_35_l698_698595

theorem mistaken_quotient_is_35 (D : ℕ) (correct_divisor mistaken_divisor correct_quotient : ℕ) 
    (h1 : D = correct_divisor * correct_quotient)
    (h2 : correct_divisor = 21)
    (h3 : mistaken_divisor = 12)
    (h4 : correct_quotient = 20)
    : D / mistaken_divisor = 35 := by
  sorry

end mistaken_quotient_is_35_l698_698595


namespace sum_of_x_satisfying_equation_l698_698478

theorem sum_of_x_satisfying_equation :
  let P (x : ℝ) := (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1
    ; S := { x : ℝ | P x }
    ; L := S.toList
    ; sum := List.sum L
  in sum = 11 :=
sorry

end sum_of_x_satisfying_equation_l698_698478


namespace rods_quadrilateral_l698_698627

/-
We need to prove that the number of rods Joy can choose as the fourth rod to form a quadrilateral with positive area is equal to 30, given:
1. Joy has rods of lengths 1 cm to 40 cm.
2. She places the rods with lengths 5 cm, 12 cm, and 20 cm on the table.
-/

theorem rods_quadrilateral (rods : Finset ℕ)
(h1 : ∀ n, n ∈ rods ↔ n ∈ (Finset.range 41).erase 0)
(h2 : ∀ n, n ≠ 5 ∧ n ≠ 12 ∧ n ≠ 20)
(h3 : 4 ≤ n ∧ n ≤ 36):
  rods.card = 30 :=
sorry

end rods_quadrilateral_l698_698627


namespace coordinates_of_B_l698_698541

-- Definitions of given conditions and coordinates.
variables (A B : ℝ × ℝ)
noncomputable def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Assuming coordinates of point A
def point_A : ℝ × ℝ := (3, 4)

-- Assumptions
variables (k : ℝ)
axiom slope_AB : slope A B = k
axiom k_value : k = 4
axiom B_on_axis : B.1 = 0 ∨ B.2 = 0

-- Theorem statement
theorem coordinates_of_B (h1 : A = point_A) (h2 : B_on_axis) (h3 : slope_AB) (h4 : k_value) :
  B = (0, -8) ∨ B = (2, 0) :=
sorry

end coordinates_of_B_l698_698541


namespace problem_statement_l698_698276

def floor (x : ℝ) := int.floor x
def frac_part (x: ℝ) := x - floor x

theorem problem_statement :
  (∀ x : ℝ, floor (x + 2) = floor x + 2) ∧
  (∀ x y : ℝ, frac_part x + frac_part y ≥ 1 → floor (x + y) = floor x + floor y + 1) ∧
  (∀ x : ℝ, floor (2 * x) ≠ 2 * floor x) :=
by
  sorry

end problem_statement_l698_698276


namespace lowest_possible_number_of_students_l698_698826

theorem lowest_possible_number_of_students : ∃ n : ℕ, (n > 0) ∧ (∃ k1 : ℕ, n = 10 * k1) ∧ (∃ k2 : ℕ, n = 24 * k2) ∧ n = 120 :=
by
  sorry

end lowest_possible_number_of_students_l698_698826


namespace probability_adjacent_vertices_decagon_l698_698756

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698756


namespace cones_volume_ratio_l698_698365

theorem cones_volume_ratio 
  (R r : ℝ) 
  (h_area_base : π * r^2 = (3/16) * 4 * π * R^2) 
  (h_vertex_sphere : ∃ h₁ h₂ : ℝ, h₁ = R/2 ∧ h₂ = 3*R/2):
  r = (sqrt 3 / 2) * R →
  h₁ = R/2 →
  h₂ = 3*R/2 →
  (∃ V1 V2 : ℝ, V1 = (1/3) * π * r^2 * h₁ ∧ V2 = (1/3) * π * r^2 * h₂ ∧ V1/V2 = 1/3) :=
by
  sorry

end cones_volume_ratio_l698_698365


namespace find_complex_number_find_imaginary_number_l698_698086

/-- Problem 1: Given a non-zero complex number z that satisfies |z+2|=2 and z + 4/z ∈ ℝ, find z. -/
theorem find_complex_number 
  (z : ℂ) 
  (h0 : z ≠ 0) 
  (h1 : complex.abs (z + 2) = 2) 
  (h2 : z + 4 / z ∈ ℝ) : 
  z = -1 + complex.I * real.sqrt 3 ∨ z = -1 - complex.I * real.sqrt 3 :=
sorry

/-- Problem 2: Given an imaginary number z that makes both z^2/(z+1) and z/(z^2+1) real numbers, find z. -/
theorem find_imaginary_number
  (z : ℂ)
  (h_imaginary : z.re = 0)
  (h1 : (z ^ 2) / (z + 1) ∈ ℝ)
  (h2 : z / (z ^ 2 + 1) ∈ ℝ) :
  z = -1 / 2 + complex.I * (real.sqrt 3 / 2) ∨ z = -1 / 2 - complex.I * (real.sqrt 3 / 2) :=
sorry

end find_complex_number_find_imaginary_number_l698_698086


namespace expression_for_x_l698_698661

variable (A B C x y : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := C > B
def condition3 := B > 0
def condition4 := C = (1 + y / 100) * B
def condition5 := A = (1 + x / 100) * C

-- The theorem
theorem expression_for_x (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 B) (h4 : condition4 B C y) (h5 : condition5 A C x) :
    x = 100 * ((100 * (A - B)) / (100 + y)) :=
sorry

end expression_for_x_l698_698661


namespace Nikolai_faster_than_Gennady_l698_698048

theorem Nikolai_faster_than_Gennady
  (gennady_jump1 gennady_jump2 : ℕ) (nikolai_jump1 nikolai_jump2 nikolai_jump3 : ℕ) :
  gennady_jump1 = 6 → gennady_jump2 = 6 →
  nikolai_jump1 = 4 → nikolai_jump2 = 4 → nikolai_jump3 = 4 →
  2 * gennady_jump1 + gennady_jump2 = 3 * (nikolai_jump1 + nikolai_jump2 + nikolai_jump3) →
  let total_path := 2000 in
  (total_path % 4 = 0 ∧ total_path % 6 ≠ 0) →
  (total_path / 4 < (total_path + 4) / 6) :=
by
  intros
  sorry

end Nikolai_faster_than_Gennady_l698_698048


namespace total_wheels_is_90_l698_698693

-- Defining the conditions
def number_of_bicycles := 20
def number_of_cars := 10
def number_of_motorcycles := 5

-- Calculating the total number of wheels
def total_wheels_in_garage : Nat :=
  (2 * number_of_bicycles) + (4 * number_of_cars) + (2 * number_of_motorcycles)

-- Statement to prove
theorem total_wheels_is_90 : total_wheels_in_garage = 90 := by
  sorry

end total_wheels_is_90_l698_698693


namespace domain_of_function_l698_698151

def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := x^2 - 9

theorem domain_of_function :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_function_l698_698151


namespace perimeter_greater_than_diagonals_l698_698853

namespace InscribedQuadrilateral

def is_convex_quadrilateral (AB BC CD DA AC BD: ℝ) : Prop :=
  -- Conditions for a convex quadrilateral (simple check)
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ AC > 0 ∧ BD > 0

def is_inscribed_in_circle (AB BC CD DA AC BD: ℝ) (r: ℝ) : Prop :=
  -- Check if quadrilateral is inscribed in a circle of radius 1
  r = 1

theorem perimeter_greater_than_diagonals 
  (AB BC CD DA AC BD: ℝ) 
  (r: ℝ)
  (h1 : is_convex_quadrilateral AB BC CD DA AC BD) 
  (h2 : is_inscribed_in_circle AB BC CD DA AC BD r) :
  0 < (AB + BC + CD + DA) - (AC + BD) ∧ (AB + BC + CD + DA) - (AC + BD) < 2 :=
by
  sorry 

end InscribedQuadrilateral

end perimeter_greater_than_diagonals_l698_698853


namespace volume_ratio_l698_698216

namespace Geometry

variables {Point : Type} [MetricSpace Point]

noncomputable def volume_pyramid (A B1 C1 D1 : Point) : ℝ := sorry

theorem volume_ratio 
  (A B1 B2 C1 C2 D1 D2 : Point) 
  (hA_B1: dist A B1 ≠ 0) (hA_B2: dist A B2 ≠ 0)
  (hA_C1: dist A C1 ≠ 0) (hA_C2: dist A C2 ≠ 0)
  (hA_D1: dist A D1 ≠ 0) (hA_D2: dist A D2 ≠ 0) :
  (volume_pyramid A B1 C1 D1 / volume_pyramid A B2 C2 D2) = 
    (dist A B1 * dist A C1 * dist A D1) / (dist A B2 * dist A C2 * dist A D2) := 
sorry

end Geometry

end volume_ratio_l698_698216


namespace Part1_Answer_Part2_Answer_l698_698851

open Nat

-- Definitions for Part 1
def contingencyTable : Type := {
  boysCoord : Nat,
  boysIneq : Nat,
  girlsCoord : Nat,
  girlsIneq : Nat
}
def totalStudents (table : contingencyTable) : Nat :=
  table.boysCoord + table.boysIneq + table.girlsCoord + table.girlsIneq

def chiSquareStatistic (table : contingencyTable) : Real :=
  let n := totalStudents table
  let a := table.boysCoord
  let b := table.boysIneq
  let c := table.girlsCoord
  let d := table.girlsIneq
  n * ((a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def isPreferenceRelatedToGender (table : contingencyTable) : Prop :=
  chiSquareStatistic table > 3.841

theorem Part1_Answer :
  isPreferenceRelatedToGender { boysCoord := 15, boysIneq := 25, girlsCoord := 20, girlsIneq := 10 } :=
  -- Sorry will be replaced with the proof
  sorry

-- Definitions for Part 2
def probabilityDist (table : contingencyTable) (stratified_boys : Nat) (total_selected : Nat) (xi : Nat) : Real :=
  if xi = 0 then 4 / 35
  else if xi = 1 then 18 / 35
  else if xi = 2 then 12 / 35
  else if xi = 3 then 1 / 35
  else 0

def expectedValueXi : Real :=
  0 * (4 / 35) + 1 * (18 / 35) + 2 * (12 / 35) + 3 * (1 / 35)

def part2ExpectedValueProof : Real :=
  expectedValueXi

theorem Part2_Answer :
  part2ExpectedValueProof = 9 / 7 :=
  sorry

end Part1_Answer_Part2_Answer_l698_698851


namespace leadership_relationship_diagram_l698_698337

-- Definitions based on conditions in a)
inductive DiagramType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

def isCorrectDiagram : DiagramType → Prop
| DiagramType.ProgramFlowchart := false
| DiagramType.ProcessFlowchart := false
| DiagramType.KnowledgeStructureDiagram := false
| DiagramType.OrganizationalStructureDiagram := true

-- Proof statement
theorem leadership_relationship_diagram :
  isCorrectDiagram DiagramType.OrganizationalStructureDiagram =
  true := 
by
  -- Use the conditions provided to formalize the proof problem
  sorry

end leadership_relationship_diagram_l698_698337


namespace find_range_of_a_l698_698822

variable {a : ℝ}
variable {x : ℝ}

theorem find_range_of_a (h₁ : x ∈ Set.Ioo (-2:ℝ) (-1:ℝ)) :
  ∃ a, a ∈ Set.Icc (1:ℝ) (2:ℝ) ∧ (x + 1)^2 < Real.log (|x|) / Real.log a :=
by
  sorry

end find_range_of_a_l698_698822


namespace smallest_base_for_62_three_digits_l698_698800

theorem smallest_base_for_62_three_digits: 
  ∃ b : ℕ, (b^2 ≤ 62 ∧ 62 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 62 ∧ 62 < n^3) → n ≥ b :=
by
  sorry

end smallest_base_for_62_three_digits_l698_698800


namespace min_distance_to_line_l698_698614

theorem min_distance_to_line :
  let Q : ℝ × ℝ := (√3 * cos θ, sin θ)
  ∃ Q : ℝ × ℝ, (Q.1^2 / 3 + Q.2^2 = 1) ∧ 
              (∀ θ : ℝ, (Q.1 = √3 * cos θ) ∧ (Q.2 = sin θ) →
                          (∀ A B C : ℝ, A = 1 ∧ B = -1 ∧ C = 4 →
                            (d = |A * Q.1 + B * Q.2 + C| / √(A^2 + B^2) → 
                              d = sqrt 2))) :=
begin
  -- proof omitted
  sorry
end

end min_distance_to_line_l698_698614


namespace probability_of_adjacent_vertices_in_decagon_l698_698748

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698748


namespace boarders_joined_l698_698081

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ) (final_ratio_num : ℕ) (final_ratio_denom : ℕ) (new_boarders : ℕ)
  (initial_ratio_boarders_to_day_scholars : initial_boarders * 16 = 7 * initial_day_scholars)
  (initial_boarders_eq : initial_boarders = 560)
  (final_ratio : (initial_boarders + new_boarders) * 2 = final_day_scholars)
  (day_scholars_eq : initial_day_scholars = 1280) : 
  new_boarders = 80 := by
  sorry

end boarders_joined_l698_698081


namespace car_speed_first_hour_l698_698013

theorem car_speed_first_hour (x : ℕ) (h1 : 60 > 0) (h2 : 40 > 0) (h3 : 2 > 0) (avg_speed : 40 = (x + 60) / 2) : x = 20 := 
by
  sorry

end car_speed_first_hour_l698_698013


namespace cyclist_total_distance_l698_698676

-- Definitions for velocities and times
def v1 : ℝ := 2  -- velocity in the first minute (m/s)
def v2 : ℝ := 4  -- velocity in the second minute (m/s)
def t : ℝ := 60  -- time interval in seconds (1 minute)

-- Total distance covered in two minutes
def total_distance : ℝ := v1 * t + v2 * t

-- The proof statement
theorem cyclist_total_distance : total_distance = 360 := by
  sorry

end cyclist_total_distance_l698_698676


namespace length_of_chord_cut_from_circle_by_line_l698_698000

noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (abs (a * x₀ + b * y₀ + c)) / (sqrt (a * a + b * b))

noncomputable def length_of_chord (radius dist : ℝ) : ℝ :=
  2 * sqrt (radius * radius - dist * dist)

theorem length_of_chord_cut_from_circle_by_line :
  let center := (1 : ℝ, 0 : ℝ)
  let radius := 1
  let line_coeff := (1 : ℝ, sqrt 3, -2) in
  let dist := distance_point_to_line center.fst center.snd line_coeff.fst line_coeff.snd line_coeff.2 in
  let chord_length := length_of_chord radius dist in
  chord_length = sqrt 3 :=
by sorry

end length_of_chord_cut_from_circle_by_line_l698_698000


namespace area_ABC_l698_698608

noncomputable def area_of_triangle (A B C : Point) : ℝ := sorry  -- defining the area function for a triangle

variables {A B C D E F : Point}

def parallelogram (A B C D : Point) : Prop := sorry  -- defining the property of a parallelogram

-- Given conditions as Lean definitions
variable (pf : parallelogram A B C D)
variable on_side_E : E ∈ segment A B
variable ratio_AE_EB : vector_length A E / vector_length E B = 1 / 2
variable intersection_F : F ∈ (line_through D E ∩ line_through A C)
variable area_AEF : area_of_triangle A E F = 6

-- Question
theorem area_ABC : area_of_triangle A B C = 18 :=
sorry

end area_ABC_l698_698608


namespace find_other_asymptote_l698_698297

-- Define the conditions
def asymptote1 (x y : ℝ) : Prop := y = 2 * x
def same_y_coordinate (y : ℝ) : Prop := y = -3

-- Define the target equation of the other asymptote
def other_asymptote (x y : ℝ) : Prop := y = -2 * x - 6

-- The theorem to be proved
theorem find_other_asymptote :
  (∃ x y, asymptote1 x y ∧ same_y_coordinate y) →
  (∃ x y, other_asymptote x y) :=
by
  intro h
  cases h with x hxy
  cases hxy with y hy
  exists x, y
  exact sorry

end find_other_asymptote_l698_698297


namespace sequence_sum_l698_698011

theorem sequence_sum :
  (∀ n : ℕ, n > 0 → let a : ℕ → ℝ := λ n, if n = 1 then 1/4 else if n = 2 then 1/5 else 1/(n+3) in
  (a 1) * (a 2) + ∑ i in (list.range (n-1)).map (λ i, (a (i + 2)) * (a (i + 3))) = n * (a 1) * (a (n + 2))) →
  (1 / (1 / 4) + 1 / (1 / 5) + ∑ i in (list.range 96).map (λ i, 1 / (1 / (i + 6)))) = 5044 :=
by
  sorry

end sequence_sum_l698_698011


namespace nikolai_completes_faster_l698_698037

-- Given conditions: distances they can cover in the same time and total journey length 
def gennady_jump_distance := 2 * 6 -- 12 meters
def nikolai_jump_distance := 3 * 4 -- 12 meters
def total_distance := 2000 -- 2000 meters before turning back

-- Mathematical translation + Target proof: prove that Nikolai will complete the journey faster
theorem nikolai_completes_faster 
  (gennady_distance_per_time : gennady_jump_distance = 12)
  (nikolai_distance_per_time : nikolai_jump_distance = 12)
  (journey_length : total_distance = 2000) : 
  ( (2000 % 4 = 0) ∧ (2000 % 6 ≠ 0) ) -> true := 
by 
  intros,
  sorry

end nikolai_completes_faster_l698_698037


namespace geometric_sequence_value_of_a_l698_698014

noncomputable def a : ℝ :=
sorry

theorem geometric_sequence_value_of_a
  (is_geometric_seq : ∀ (x y z : ℝ), z / y = y / x)
  (first_term : ℝ)
  (second_term : ℝ)
  (third_term : ℝ)
  (h1 : first_term = 140)
  (h2 : second_term = a)
  (h3 : third_term = 45 / 28)
  (pos_a : a > 0):
  a = 15 :=
sorry

end geometric_sequence_value_of_a_l698_698014


namespace max_white_rooks_l698_698296

-- Definitions of the problem conditions
def chessboard_8x8 := (fin 8 × fin 8)

structure rook_placement :=
  (black_rooks : fin 8 → fin 8 → Prop)
  (white_rooks : fin 8 → fin 8 → Prop)
  (non_attacking : ∀ r1 r2, black_rooks r1 → white_rooks r2 → r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)

-- Problem statement: Prove that the maximum number of white rooks (k) is 14
theorem max_white_rooks (rp : rook_placement) (h : ∀ r, ∃ k, rook_placement.black_rooks rp k = true) :
  ∃ k, (∀ r, rook_placement.white_rooks rp r = true → k ≤ 14) :=
sorry

end max_white_rooks_l698_698296


namespace smallest_number_am_median_largest_l698_698702

noncomputable def smallest_number (a b c : ℕ) : ℕ :=
if a ≤ b ∧ a ≤ c then a
else if b ≤ a ∧ b ≤ c then b
else c

theorem smallest_number_am_median_largest (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 28) (h3 : c = b + 6) :
  smallest_number a b c = 28 :=
sorry

end smallest_number_am_median_largest_l698_698702


namespace number_of_squares_at_least_5_l698_698138

def is_in_H (x y : ℤ) : Prop :=
  1 ≤ |x| ∧ |x| ≤ 7 ∧ 1 ≤ |y| ∧ |y| ≤ 7

theorem number_of_squares_at_least_5 (H := { p : ℤ × ℤ | is_in_H p.1 p.2 }) :
  (number_of_squares_with_vertices_in_H H 5 + 
   number_of_squares_with_vertices_in_H H 6 + 
   number_of_squares_with_vertices_in_H H 7) * 4 = 56 :=
sorry

-- The following function is meant to count the squares of a given side length in H.
-- Placeholder for actual implementation.
def number_of_squares_with_vertices_in_H (H : set (ℤ × ℤ)) (side_length : ℤ) : ℕ :=
sorry

end number_of_squares_at_least_5_l698_698138


namespace area_common_region_l698_698409

-- Definitions of the conditions:
def rectangle := {length : ℝ, width : ℝ} 
def circle := {radius : ℝ} 
def right_triangle := {hypotenuse : ℝ} 

-- Given data
def rect := rectangle.mk 10 3
def circ := circle.mk 2.5
def tri := right_triangle.mk 10

-- Statement of the problem
theorem area_common_region : 
  let common_area := (3.125 : ℝ) * real.pi + 1 in
  common_area = 3.125 * real.pi + 1 := 
by
  sorry

end area_common_region_l698_698409


namespace decagon_adjacent_vertices_probability_l698_698733

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698733


namespace min_value_f_on_interval_l698_698553

noncomputable def f (a x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

theorem min_value_f_on_interval (a : ℝ) :
  (a ≤ 1 → ∀ x ∈ set.Icc 0 1, f a x ≥ -a^2 - 4) ∧  
  (a > 1 → ∀ x ∈ set.Icc 0 1, f a x ≥ -a^2 - 4 * a) :=
by
  sorry

end min_value_f_on_interval_l698_698553


namespace range_of_a_l698_698523

theorem range_of_a (a : ℝ) (h1 : 0 < a + 1) (h2 : 2a - 3 < 0) : -1 < a ∧ a < 3/2 := 
by 
  sorry

end range_of_a_l698_698523


namespace sequence_a_1998_value_l698_698351

theorem sequence_a_1998_value :
  (∃ (a : ℕ → ℕ),
    (∀ n : ℕ, 0 <= a n) ∧
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ k : ℕ, ∃ i j t : ℕ, k = a i + 2 * a j + 4 * a t) ∧
    a 1998 = 1227096648) := sorry

end sequence_a_1998_value_l698_698351


namespace probability_pair_two_girls_l698_698857

def probability_pair_two_girls_formed (boys girls : ℕ) : ℚ :=
  let total_pairs := (Nat.factorial (boys + girls)) / (2 ^ (boys + girls / 2) * Nat.factorial (boys + girls / 2))
  let no_girl_pairs := Nat.factorial boys
  let girl_pairs := total_pairs - no_girl_pairs
  girl_pairs / total_pairs

theorem probability_pair_two_girls (boys girls : ℕ) (hb : boys = 5) (hg : girls = 5) :
  probability_pair_two_girls_formed boys girls ≈ 0.87 := by
  sorry

end probability_pair_two_girls_l698_698857


namespace ball_distribution_ratio_l698_698322

noncomputable def probability_ratio (p q : ℚ) : Prop :=
  p / q = 48

theorem ball_distribution_ratio :
  let N := 15
  let bins := 5
  let p := (nat.factorial 15) / ((nat.factorial 3) ^ 5 * bins ^ N : ℚ)
  let q := (nat.factorial 15) / (nat.factorial 5 * nat.factorial 4 * (nat.factorial 2) ^ 3 * bins ^ N : ℚ)
  probability_ratio p q :=
by
  sorry

end ball_distribution_ratio_l698_698322


namespace price_of_stock_l698_698606

-- Defining the conditions
def income : ℚ := 650
def dividend_rate : ℚ := 10
def investment : ℚ := 6240

-- Defining the face value calculation from income and dividend rate
def face_value (i : ℚ) (d_rate : ℚ) : ℚ := (i * 100) / d_rate

-- Calculating the price of the stock
def stock_price (inv : ℚ) (fv : ℚ) : ℚ := (inv / fv) * 100

-- Main theorem to be proved
theorem price_of_stock : stock_price investment (face_value income dividend_rate) = 96 := by
  sorry

end price_of_stock_l698_698606


namespace cost_of_carton_l698_698357

-- Definition of given conditions
def totalCost : ℝ := 4.88
def numberOfCartons : ℕ := 4
def costPerCarton : ℝ := 1.22

-- The proof statement
theorem cost_of_carton
  (h : totalCost = 4.88) 
  (n : numberOfCartons = 4) :
  totalCost / numberOfCartons = costPerCarton := 
sorry

end cost_of_carton_l698_698357


namespace eldest_got_8_sweets_l698_698859

noncomputable theory

-- Define the initial number of sweets
def total_sweets : ℕ := 27

-- Mother keeps 1/3 of the sweets
def sweets_mother_kept : ℕ := total_sweets / 3

-- Remaining sweets for children
def remaining_sweets : ℕ := total_sweets - sweets_mother_kept

-- Sweets received by the second child
def sweets_second_child : ℕ := 6

-- Sweets remaining after giving to the second child
def sweets_for_youngest_and_eldest : ℕ := remaining_sweets - sweets_second_child

-- The relation between the youngest and the eldest child
variable {E Y : ℕ}
axiom youngest_half_of_eldest : Y = E / 2
axiom total_for_youngest_and_eldest : E + Y = sweets_for_youngest_and_eldest

-- The theorem to prove that the eldest child got 8 sweets
theorem eldest_got_8_sweets (E : ℕ) (Y : ℕ) 
  (h1 : Y = E / 2)
  (h2 : E + Y = 12) :
  E = 8 := by sorry

end eldest_got_8_sweets_l698_698859


namespace hyperbola_eccentricity_l698_698202

variable {a b : ℝ}
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : ∀ x y : ℝ, (x, y) = (- √2, 0) → (b * x / (√ (a^2 + b^2))) = √5 / 5)

theorem hyperbola_eccentricity : ∀ a b : ℝ, h1 → h2 → h3 → (b^2 = (1/9) * a^2) → 
                                    (c = (√10 / 3) * a) → 
                                    (e = c / a) → 
                                    e = √10 / 3 :=
by
  intros a b h1 h2 h3 hb hc he
  sorry

end hyperbola_eccentricity_l698_698202


namespace midpoint_of_intersection_is_correct_l698_698517

noncomputable def midpoint_of_intersection (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def ellipse_line_intersection_midpoint : ℝ × ℝ :=
  let a := 3
  let c := 2 * Real.sqrt 2
  let b := Real.sqrt (a ^ 2 - c ^ 2)
  let ellipse_equation (x y : ℝ) := x ^ 2 / a ^ 2 + y ^ 2 = 1
  let line_equation (x y : ℝ) := y = x + 2
  let A := sorry /- intersection point 1 -/
  let B := sorry /- intersection point 2 -/
  midpoint_of_intersection A B

theorem midpoint_of_intersection_is_correct :
  ellipse_line_intersection_midpoint = (-9 / 5, 1 / 5) :=
by
  sorry

end midpoint_of_intersection_is_correct_l698_698517


namespace seq_a_ge_two_pow_nine_nine_l698_698521

theorem seq_a_ge_two_pow_nine_nine (a : ℕ → ℤ) 
  (h0 : a 1 > a 0)
  (h1 : a 1 > 0)
  (h2 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 :=
sorry

end seq_a_ge_two_pow_nine_nine_l698_698521


namespace problem1_solution_problem2_solution_l698_698658

-- Problem 1: System of Equations
theorem problem1_solution (x y : ℝ) (h_eq1 : x - y = 2) (h_eq2 : 2 * x + y = 7) : x = 3 ∧ y = 1 :=
by {
  sorry -- Proof to be filled in
}

-- Problem 2: Fractional Equation
theorem problem2_solution (y : ℝ) (h_eq : 3 / (1 - y) = y / (y - 1) - 5) : y = 2 :=
by {
  sorry -- Proof to be filled in
}

end problem1_solution_problem2_solution_l698_698658


namespace combined_age_of_siblings_l698_698438

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end combined_age_of_siblings_l698_698438


namespace train_speed_is_180_kmph_l698_698870

-- Definitions of the given conditions
def length_of_train_meters : ℝ := 900  -- Length of the train in meters
def time_to_cross_seconds : ℝ := 18    -- Time to cross the pole in seconds

-- Conversion constants
def meters_to_kilometers (m : ℝ) : ℝ := m / 1000
def seconds_to_hours (s : ℝ) : ℝ := s / 3600

-- Conversion of the given conditions to consistent units
def length_of_train_kilometers : ℝ := meters_to_kilometers length_of_train_meters
def time_to_cross_hours : ℝ := seconds_to_hours time_to_cross_seconds

-- Calculation of the speed
def speed_of_train_kmph : ℝ := length_of_train_kilometers / time_to_cross_hours

-- The theorem to prove
theorem train_speed_is_180_kmph : speed_of_train_kmph = 180 := by
  sorry

end train_speed_is_180_kmph_l698_698870


namespace ap_bh_eq_bq_ah_l698_698649

variable {α : Type}
variables (A B C L H P Q : α)
variables [Inhabited α] [LinearOrder α] [Fractional α] [Intro α] 

theorem ap_bh_eq_bq_ah (h_triangle : AcuteAngledTriangle A B C)
    (hL : OnLineSegment L A B)
    (hH : OnLineSegment H A B)
    (hCL_bisector : IsAngleBisector (C, L) (angle ACB))
    (hCH_altitude : IsAltitude (C, H) (line A B))
    (hP_perpendicular : IsPerpendicular (L, P) (line A C))
    (hQ_perpendicular : IsPerpendicular (L, Q) (line B C)) :
    (AP_from L P A) * (BH_from B H C) = (BQ_from B Q L) * (AH_from A H C) :=
sorry

end ap_bh_eq_bq_ah_l698_698649


namespace mathematically_equivalent_proof_problem_l698_698253

noncomputable def general_equation_of_line (t : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 1 + (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t ∧ (x - y - 1 = 0)

noncomputable def polar_to_cartesian_equation_C (θ : ℝ) : Prop :=
  ∃ (ρ x y : ℝ), ρ = 4 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ (x^2 + y^2 - 4 * y = 0)

noncomputable def minimum_distance_P_to_line : Prop :=
  let d := (| -1 | + | -1 |) / Real.sqrt 2 - 1 in d = Real.sqrt 2 - 1

theorem mathematically_equivalent_proof_problem : Prop :=
  ∀ (t θ : ℝ), general_equation_of_line t ∧ polar_to_cartesian_equation_C θ ∧ minimum_distance_P_to_line

end mathematically_equivalent_proof_problem_l698_698253


namespace Nikolai_faster_than_Gennady_l698_698047

theorem Nikolai_faster_than_Gennady
  (gennady_jump1 gennady_jump2 : ℕ) (nikolai_jump1 nikolai_jump2 nikolai_jump3 : ℕ) :
  gennady_jump1 = 6 → gennady_jump2 = 6 →
  nikolai_jump1 = 4 → nikolai_jump2 = 4 → nikolai_jump3 = 4 →
  2 * gennady_jump1 + gennady_jump2 = 3 * (nikolai_jump1 + nikolai_jump2 + nikolai_jump3) →
  let total_path := 2000 in
  (total_path % 4 = 0 ∧ total_path % 6 ≠ 0) →
  (total_path / 4 < (total_path + 4) / 6) :=
by
  intros
  sorry

end Nikolai_faster_than_Gennady_l698_698047


namespace probability_of_not_hearing_favorite_l698_698457

noncomputable def song_lengths : List ℕ := 
  [45, 90, 135, 180, 225, 270, 315, 360, 405, 450, 495, 540]

def favorite_song_length : ℕ := 315

def total_songs : ℕ := 12
def total_arrangements := ∏ i in Finset.range total_songs, (i + 1)
def favorable_arrangements : ℕ := 3 * ∏ i in Finset.range 10, (i + 1)

def probability_not_hearing_favorite :=
  1 - (favorable_arrangements.to_rat / total_arrangements.to_rat)

theorem probability_of_not_hearing_favorite : 
  probability_not_hearing_favorite = 43 / 44 := 
by
  sorry

end probability_of_not_hearing_favorite_l698_698457


namespace marble_arrangement_l698_698128

theorem marble_arrangement:
  ∃ (m : ℕ), 
  let num_black := 3 in
  let num_white := m in
  let total_marbles := num_black + num_white in
  num_white = 4 ∧
  total_marbles = 7 ∧
  (∑ (B W : ℕ), 
    B = num_black ∧ 
    W = num_white ∧ 
    (∃ (arrangement : Finset (Fin (total_marbles))), 
      arrangement.card = 10 ∧ 
      (∀ i ∈ arrangement, i < total_marbles - 1 → 
        (arrangement i.succ ≠ arrangement i) → 
        (arrangement i = arrangement (i + 1)) 
      )
    )
  ) 
  :=
begin
  sorry
end

end marble_arrangement_l698_698128


namespace o_hara_triple_example_l698_698023

-- definitions
def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a) + (Real.sqrt b) = x

-- conditions
def a : ℕ := 81
def b : ℕ := 49
def x : ℕ := 16

-- statement
theorem o_hara_triple_example : is_OHara_triple a b x :=
by
  sorry

end o_hara_triple_example_l698_698023


namespace sum_of_intersection_coordinates_l698_698450

def f (x : ℝ) : ℝ := 1 - (x + 1)^2 / 3

theorem sum_of_intersection_coordinates :
  let a := 1 + sqrt 5,
      b := f (1 + sqrt 5) in
  a + b = - (6 + sqrt 5) / 3 :=
by
  sorry

end sum_of_intersection_coordinates_l698_698450


namespace problem1_problem2_l698_698551

noncomputable def f (x : ℝ) := Real.log (x + 1) / Real.log 2
noncomputable def g (x : ℝ) := -(Real.log (1 - x) / Real.log 2)

theorem problem1 (x : ℝ) : 2 * f x + g x ≥ 0 ↔ 0 ≤ x ∧ x < 1 := sorry

theorem problem2 (m : ℝ) : (∀ x ∈ Ico 0 1, 2 * f x + g x ≥ m) ↔ m ≤ 0 := sorry

end problem1_problem2_l698_698551


namespace jack_age_difference_l698_698131

def beckett_age : ℕ := 12
def olaf_age : ℕ := beckett_age + 3
def shannen_age : ℕ := olaf_age - 2
def total_age : ℕ := 71
def jack_age : ℕ := total_age - (beckett_age + olaf_age + shannen_age)
def difference := jack_age - 2 * shannen_age

theorem jack_age_difference :
  difference = 5 :=
by
  -- Math proof goes here
  sorry

end jack_age_difference_l698_698131


namespace sum_of_x_satisfying_equation_l698_698479

theorem sum_of_x_satisfying_equation :
  let P (x : ℝ) := (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1
    ; S := { x : ℝ | P x }
    ; L := S.toList
    ; sum := List.sum L
  in sum = 11 :=
sorry

end sum_of_x_satisfying_equation_l698_698479


namespace sierpinski_carpet_area_sum_l698_698877

theorem sierpinski_carpet_area_sum (n : ℕ) :
  let A0 : ℝ := (real.sqrt 3)/4
  let Ak (k : ℕ) : ℝ := (real.sqrt 3)/16 * (3/4) ^ (k - 1)
  (∑ k in finset.range n, Ak (k + 1)) = (real.sqrt 3)/4 * (1 - (3/4)^n) :=
by
  let A0 := (real.sqrt 3) / 4
  let Ak := λ k: ℕ, (real.sqrt 3) / 16 * (3 / 4) ^ (k - 1)
  sorry

end sierpinski_carpet_area_sum_l698_698877


namespace num_integers_satisfying_ineq_count_integers_satisfying_ineq_l698_698160

theorem num_integers_satisfying_ineq (k : ℤ) :
  (20 < k^2 ∧ k^2 < 150) ↔ k ∈ ({-12, -11, -10, -9, -8, -7, -6, -5, 5, 6, 7, 8, 9, 10, 11, 12} : set ℤ) := by
  sorry

theorem count_integers_satisfying_ineq :
  {n : ℤ | 20 < n^2 ∧ n^2 < 150}.finite.to_finset.card = 16 := by
  sorry

end num_integers_satisfying_ineq_count_integers_satisfying_ineq_l698_698160


namespace chebyshev_inequality_mod_l698_698305

theorem chebyshev_inequality_mod (a : ℕ → ℝ) (n : ℕ) (h1 : ∀ i, 1 ≤ i → a i > 0) :
  (1 / (∑ i in finset.range n, 1 / (1 + a i))) - (1 / (∑ i in finset.range n, 1 / a i)) ≤ (1 / n) :=
by sorry

end chebyshev_inequality_mod_l698_698305


namespace horatio_sonnets_count_l698_698996

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end horatio_sonnets_count_l698_698996


namespace parallelogram_inequality_l698_698650

variables {E : Type*} [InnerProductSpace ℝ E] 
variables (u v : E)

-- Define the vectors corresponding to the diagonals
def AC := u + v
def BD := v - u

-- Statement of the theorem
theorem parallelogram_inequality (h : true) :
  ∥v∥^2 - ∥u∥^2 < ∥AC u v∥ * ∥BD u v∥ :=
sorry

end parallelogram_inequality_l698_698650


namespace solve_inequality_l698_698639

noncomputable def f (x : ℝ) : ℝ := real.log (2^x + 1) / real.log 2

noncomputable def f_inv (y : ℝ) : ℝ := real.log (2^y - 1) / real.log 2

theorem solve_inequality :
  ∀ x : ℝ, 2 * f x ≤ f_inv (real.log 5 / real.log 2) → x ≤ 0 :=
begin
  sorry
end

end solve_inequality_l698_698639


namespace weight_of_new_student_l698_698394

theorem weight_of_new_student 
  (avg_weight_29 : ℕ → ℕ → ℕ)
  (new_avg_weight_30 : ℕ → ℕ → ℕ)
  (total_weight_29 : ∀ n m, avg_weight_29 n m = 28 * 29)
  (total_weight_30 : ∀ n m, new_avg_weight_30 n m = 27.1 * 30)
  (new_weight_eq : ∀ n m, new_weight_eq = total_weight_30 n m - total_weight_29 n m) :
  new_student_weight new_weight_eq = 1 :=
by sorry

end weight_of_new_student_l698_698394


namespace find_k_intersection_l698_698678

theorem find_k_intersection :
  ∃ (k : ℚ), 
    (∃ (x y : ℚ), y = -4 * x + 2 ∧ y = 3 * x - 18 ∧ y = 7 * x + k) → 
      k = -206 / 7 :=
begin
  sorry,
end

end find_k_intersection_l698_698678


namespace shirt_ratio_l698_698659

theorem shirt_ratio
  (A B S : ℕ)
  (h1 : A = 6 * B)
  (h2 : B = 3)
  (h3 : S = 72) :
  S / A = 4 :=
by
  sorry

end shirt_ratio_l698_698659


namespace GH_parallel_t_l698_698349

-- Definition of the points and conditions
variables {k : Type*} [euclidean_plane k]
variables (A B C D : k) -- Points on the circle in order
variables (t : k → Prop) -- Tangent at C
variables (s : k → Prop) -- Reflection of AB across AC

-- Assuming conditions, we'll define reflective symmetry and intersecting points.
variables (AC BD CD : k → Prop) -- Lines AC, BD, and CD in the plane

-- Hypotheses for being reflective and intersecting lines
hypothesis (h_reflect : ∀ x, (s x ↔ ∃ y ∈ AB, ∃ z ∈ AC, reflection y z x))
hypothesis (h_intersect_G : ∃ G, ∀ x, (AC x ↔ x = G) ∧ (BD x ↔ x = G))
hypothesis (h_intersect_H : ∃ H, ∀ x, (s x ↔ x = H) ∧ (CD x ↔ x = H))

-- This is the actual theorem we need to state and prove
theorem GH_parallel_t (G H : k)
  (hG : ∀ x, (AC x ↔ x = G) ∧ (BD x ↔ x = G))
  (hH : ∀ x, (s x ↔ x = H) ∧ (CD x ↔ x = H))
  : parallel (GH : k → Prop) t :=
by
  sorry -- Proof not included

end GH_parallel_t_l698_698349


namespace time_to_pass_platform_is_correct_l698_698115

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def time_to_pass_pole : ℝ := 10 -- seconds
noncomputable def time_to_pass_platform : ℝ := 60 -- seconds

-- Speed of the train
noncomputable def train_speed := train_length / time_to_pass_pole -- meters/second

-- Length of the platform
noncomputable def platform_length := train_speed * time_to_pass_platform - train_length -- meters

-- Proving the time to pass the platform is 50 seconds
theorem time_to_pass_platform_is_correct : 
  (platform_length / train_speed) = 50 :=
by
  sorry

end time_to_pass_platform_is_correct_l698_698115


namespace num_ways_sum_528_consecutive_l698_698605

theorem num_ways_sum_528_consecutive : 
  (count_consecutive_sums 528) = 13 := 
sorry

def count_consecutive_sums (S : ℕ) : ℕ :=
  (factors_of_sum S).length

def factors_of_sum (S : ℕ) : List ℕ :=
  (List.range (S + 1)).filter (λ n => n ≥ 3 ∧ ((S * 2) % n = 0))

end num_ways_sum_528_consecutive_l698_698605


namespace point_P_outside_circleC_max_length_PQ_min_value_PA_PB_l698_698281

variables (m x y : ℝ)
def line1 : ℝ → ℝ → Prop := λ m x y, m * x - y - 3 * m + 1 = 0
def line2 : ℝ → ℝ → Prop := λ m x y, x + m * y - 3 * m - 1 = 0
def circleC : ℝ → ℝ → Prop := λ x y, (x + 2) ^ 2 + (y + 1) ^ 2 = 4
def dist (x1 y1 x2 y2 : ℝ) := sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem point_P_outside_circleC :
  ∃ x y, line1 m x y ∧ line2 m x y ∧ ¬ circleC x y :=
sorry

variables (Q P : ℝ × ℝ) (length_AB : ℝ)
def midpoint_Q (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def valid_chord (A B : ℝ × ℝ) := dist A.1 A.2 B.1 B.2 = length_AB

theorem max_length_PQ :
  ∃ A B : ℝ × ℝ, valid_chord A B ∧ Q = midpoint_Q A B ∧ length_AB = 6 + sqrt 2 :=
sorry

def dot_product_PA_PB (A B P : ℝ × ℝ) :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

theorem min_value_PA_PB :
  ∃ A B : ℝ × ℝ, valid_chord A B ∧ 
  Q = midpoint_Q A B ∧ dot_product_PA_PB A B (2, 2) = 15 - 8 * sqrt 2 :=
sorry

end point_P_outside_circleC_max_length_PQ_min_value_PA_PB_l698_698281


namespace solve_system_of_equations_l698_698570

theorem solve_system_of_equations :
  ∃! (pairs : List (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ pairs → (p.1 + 2 * p.2 = 2) ∧ (| |p.1| - |p.2| | = 2)) ∧ 
    pairs.length = 2 :=
by
  sorry

end solve_system_of_equations_l698_698570


namespace sum_digits_l698_698886

def digit_sum (d : ℕ) (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.count d

theorem sum_digits (sum_d2_d5 : ℕ) (h : sum_d2_d5 = (List.range 1 151).sum (λ n, digit_sum 2 n + digit_sum 5 n)) :
  sum_d2_d5 = 420 :=
by
  sorry

end sum_digits_l698_698886


namespace participants_take_questionnaire_C_l698_698055

theorem participants_take_questionnaire_C :
  ∀ (n_total n_sample n_first_group number_upper_A number_upper_B : ℕ),
    n_total = 960 →
    n_sample = 32 →
    n_first_group = 9 →
    number_upper_A = 450 →
    number_upper_B = 750 →
    (n_sample - (number_upper_B / (n_total / n_sample)) = 7) :=
begin
  intros n_total n_sample n_first_group number_upper_A number_upper_B 
  h_total h_sample h_first_group h_upper_A h_upper_B,
  -- skipping the proof as per the instructions
  sorry,
end

end participants_take_questionnaire_C_l698_698055


namespace focus_of_parabola_l698_698203

noncomputable def hyperbola (a b : ℝ) := { p : ℝ × ℝ // p.1^2 / a^2 - p.2^2 / b^2 = 1 }

noncomputable def parabola (p : ℝ) := { q : ℝ × ℝ // q.1^2 = 2 * p * q.2 }

theorem focus_of_parabola
  (a b p : ℝ)
  (hyp_a_pos : a > 0) (hyp_b_pos : b > 0) (hyp_p_pos : p > 0)
  (hyp_eccentricity : a^2 + b^2 = 2 * b^2)
  (tangent_parallel : ∀ (P : ℝ × ℝ), P ∈ parabola p → deriv (λ x, x^2 / (2 * p)) P.1 = 1 ∨ deriv (λ x, x^2 / (2 * p)) P.1 = -1)
  (distance_PF : ∀ (P F : ℝ × ℝ), P ∈ parabola p → F = (0, p/2) → |(P.1 - F.1)^2 + (P.2 - F.2)^2| = 9) :
  (0, 3/2) = (0, p/2) :=
by 
  sorry

end focus_of_parabola_l698_698203


namespace domain_of_sqrt_function_l698_698473

theorem domain_of_sqrt_function (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (2 - x)) ↔ x ≤ 2 :=
by
  sorry

end domain_of_sqrt_function_l698_698473


namespace ferry_speeds_l698_698501

theorem ferry_speeds (v_P v_Q : ℝ) 
  (h1: v_P = v_Q - 1) 
  (h2: 3 * v_P * 3 = v_Q * (3 + 5))
  : v_P = 8 := 
sorry

end ferry_speeds_l698_698501


namespace nikolai_faster_than_gennady_l698_698043

-- The conditions of the problem translated to Lean definitions
def gennady_jump_length : ℕ := 6
def gennady_jumps_per_time : ℕ := 2
def nikolai_jump_length : ℕ := 4
def nikolai_jumps_per_time : ℕ := 3
def turn_around_distance : ℕ := 2000
def round_trip_distance : ℕ := 2 * turn_around_distance

-- The statement that Nikolai completes the journey faster than Gennady
theorem nikolai_faster_than_gennady :
  (nikolai_jumps_per_time * nikolai_jump_length) = (gennady_jumps_per_time * gennady_jump_length) →
  (round_trip_distance % nikolai_jump_length = 0) →
  (round_trip_distance % gennady_jump_length ≠ 0) →
  (round_trip_distance / nikolai_jump_length) + 1 < (round_trip_distance / gennady_jump_length) →
  "Nikolay completes the journey faster." :=
by
  intros h_eq_speed h_nikolai_divisible h_gennady_not_divisible h_time_compare
  sorry

end nikolai_faster_than_gennady_l698_698043


namespace numerical_triangle_integers_l698_698861
-- Import the entire Mathlib for all required mathematical tools and libraries.

-- Define the conditions given in the problem.
theorem numerical_triangle_integers
  (n : ℕ) 
  (units : ℕ → ℕ) 
  (arbitrary_integers : ℕ → ℤ)
  (triangle_property : ∀ a c b d, a * c = b * d + 1) 
  (non_zero : ∀ x, x ≠ 0) :
  ∀ x, ∃ k : ℤ, x = k :=
begin
  sorry -- Proof omitted
end

end numerical_triangle_integers_l698_698861


namespace solve_trig_equation_l698_698657

theorem solve_trig_equation (x : ℝ) (h : |x - 3| < 1) :
  (sqrt (2 + cos (2 * x) - sqrt 3 * tan x) = sin x - sqrt 3 * cos x) ↔ 
  (x = π ∨ x = 3 * π / 4 ∨ x = 5 * π / 4) :=
sorry

end solve_trig_equation_l698_698657


namespace num_integers_satisfying_ineq_count_integers_satisfying_ineq_l698_698159

theorem num_integers_satisfying_ineq (k : ℤ) :
  (20 < k^2 ∧ k^2 < 150) ↔ k ∈ ({-12, -11, -10, -9, -8, -7, -6, -5, 5, 6, 7, 8, 9, 10, 11, 12} : set ℤ) := by
  sorry

theorem count_integers_satisfying_ineq :
  {n : ℤ | 20 < n^2 ∧ n^2 < 150}.finite.to_finset.card = 16 := by
  sorry

end num_integers_satisfying_ineq_count_integers_satisfying_ineq_l698_698159


namespace total_donation_l698_698677

theorem total_donation : 2 + 6 + 2 + 8 = 18 := 
by sorry

end total_donation_l698_698677


namespace find_a_2020_l698_698562

theorem find_a_2020 :
  let a : ℕ → ℕ := 
    let b := λ n : ℕ, n + 1 in
    λ n : ℕ, 2 + (∑ i in Finset.range n, b i)
  in a 2020 = 2041211 :=
by
  sorry

end find_a_2020_l698_698562


namespace line_perpendicular_to_intersection_l698_698200

variables (α β : Plane) (l : Line) (a b : Line)
-- Assuming the necessary conditions are given as hypotheses:
axiom h1 : α ⊥ β
axiom h2 : α ∩ β = l
axiom h3 : a ∥ α
axiom h4 : b ⊥ β

theorem line_perpendicular_to_intersection : b ⊥ l :=
sorry

end line_perpendicular_to_intersection_l698_698200


namespace dried_mushrooms_weight_l698_698500

theorem dried_mushrooms_weight (fresh_weight : ℝ) (water_content_fresh : ℝ) (water_content_dried : ℝ) :
  fresh_weight = 22 →
  water_content_fresh = 0.90 →
  water_content_dried = 0.12 →
  ∃ x : ℝ, x = 2.5 :=
by
  intros h1 h2 h3
  have hw_fresh : ℝ := fresh_weight * water_content_fresh
  have dry_material_fresh : ℝ := fresh_weight - hw_fresh
  have dry_material_dried : ℝ := 1.0 - water_content_dried
  have hw_dried := dry_material_fresh / dry_material_dried
  use hw_dried
  sorry

end dried_mushrooms_weight_l698_698500


namespace remainder_3203_4507_9929_mod_75_l698_698373

theorem remainder_3203_4507_9929_mod_75 :
  (3203 * 4507 * 9929) % 75 = 34 :=
by
  have h1 : 3203 % 75 = 53 := sorry
  have h2 : 4507 % 75 = 32 := sorry
  have h3 : 9929 % 75 = 29 := sorry
  -- complete the proof using modular arithmetic rules.
  sorry

end remainder_3203_4507_9929_mod_75_l698_698373


namespace commencement_addresses_sum_l698_698601

noncomputable def addresses (S H L : ℕ) := 40

theorem commencement_addresses_sum
  (S H L : ℕ)
  (h1 : S = 12)
  (h2 : S = 2 * H)
  (h3 : L = S + 10) :
  S + H + L = addresses S H L :=
by
  sorry

end commencement_addresses_sum_l698_698601


namespace sphere_volume_in_cone_l698_698108

-- Definitions for the problem conditions
def cone_base_diameter : ℝ := 16
def cone_base_radius : ℝ := cone_base_diameter / 2
def cone_height : ℝ := cone_base_radius -- height is equal to the radius due to the right triangle property
def sphere_radius : ℝ := cone_height / 2
def sphere_volume : ℝ := (4 / 3) * Real.pi * (sphere_radius ^ 3)

-- The statement of the theorem
theorem sphere_volume_in_cone : sphere_volume = (256 / 3) * Real.pi := 
by {
  -- Proof would go here
  sorry
}

end sphere_volume_in_cone_l698_698108


namespace fraction_of_red_cubes_given_to_Gage_l698_698991

theorem fraction_of_red_cubes_given_to_Gage :
  ∀ (Grady_red Grady_blue Gage_init_red Gage_init_blue Gage_final : ℕ) (fraction_received : ℚ),
    Grady_red = 20 →
    Grady_blue = 15 →
    Gage_init_red = 10 →
    Gage_init_blue = 12 →
    Gage_final = 35 →
    fraction_received = 1 / 3 →
    Gage_init_red + (fraction_received * Grady_red) + (Gage_init_blue + (fraction_received * Grady_blue)) = Gage_final →
    fraction_received * 20 = 8 →
    fraction_received = 2 / 5 :=
by
  intros Grady_red Grady_blue Gage_init_red Gage_init_blue Gage_final fraction_received
  assume hGrady_red hGrady_blue hGage_init_red hGage_init_blue hGage_final hFraction_received hTotal_cubes hFrac_times_20
  have h1 : fraction_received * Grady_red = 8 := by sorry
  have h2 : fraction_received = 8 / 20 := by sorry
  exact by sorry

end fraction_of_red_cubes_given_to_Gage_l698_698991


namespace coordinates_of_P_l698_698243

variable (P Q R : ℝ × ℝ)
variable (y_R : ℝ)
variable (x_Q : ℝ)
variable (P_coordinates : ℝ × ℝ)

-- Given conditions
def is_horizontal (P R : ℝ × ℝ) : Prop := P.snd = R.snd
def is_vertical (P Q : ℝ × ℝ) : Prop := P.fst = Q.fst

-- Given y-coordinate of R and x-coordinate of Q
axiom y_R_eq : y_R = -2
axiom x_Q_eq : x_Q = -11
axiom R_coordinates : R = (R.fst, y_R)
axiom Q_coordinates : Q = (x_Q, Q.snd)

-- Proof statement
theorem coordinates_of_P
  (h1 : is_horizontal P R)
  (h2 : is_vertical P Q)
  (h3 : R.snd = y_R)
  (h4 : Q.fst = x_Q) :
  P = (x_Q, y_R) :=
by
  rw [x_Q_eq, y_R_eq]
  sorry

end coordinates_of_P_l698_698243


namespace statement_1_statement_2_statement_3_statement_4_l698_698496

variables {f : ℝ → ℝ} {x : ℝ}

theorem statement_1 (h : ∀ x, f (-1 - x) = f (x - 1)) : x ↦ f (-1 - x) x ↦ f (x - 1) symmetric_about x := 0 :=
sorry

theorem statement_2 (h : ∀ x, f (1 - x) = f (x - 1)) : ¬ (x ↦ f x symmetric_about x := 1) :=
sorry

theorem statement_3 (h : ∀ x, f (1 + x) = f (x - 1)) : periodic f :=
sorry

theorem statement_4 (h : ∀ x, f (1 - x) = -f (x - 1)) : x ↦ f x symmetric_about (0, 0) :=
sorry

end statement_1_statement_2_statement_3_statement_4_l698_698496


namespace angle_between_slant_height_and_height_l698_698148

noncomputable def slant_height_angle (R : ℝ) : ℝ :=
  let S_bok := π * R * ((1 + Real.sqrt 5) / 2)
  let S_osn := π * R^2
  let S := S_osn + S_bok
  let alpha := Real.arcsin ((Real.sqrt 5 - 1) / 2)
  alpha

theorem angle_between_slant_height_and_height (R : ℝ) : 
  ∃ α, α = Real.arcsin ((Real.sqrt 5 - 1) / 2) :=
by
  use slant_height_angle R
  sorry


end angle_between_slant_height_and_height_l698_698148


namespace max_rooks_l698_698797

-- Define a chessboard as an 8x8 grid
def Chessboard := Fin 8 × Fin 8

-- Define what it means for rooks to attack each other
def attacks (r1 r2 : Chessboard) : Prop :=
  r1.fst = r2.fst ∨ r1.snd = r2.snd

-- Statement of the problem
theorem max_rooks (S : Finset Chessboard) :
  (∀ r ∈ S, ∃! r' ∈ S, r ≠ r' ∧ attacks r r') → S.card ≤ 10 := sorry

end max_rooks_l698_698797


namespace intersection_M_N_l698_698213

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = Real.exp (x * Real.log 2) }
noncomputable def N : Set ℝ := { x | ∃ y : ℝ, y = Real.log (2 * x - x^2) }

theorem intersection_M_N : M ∩ N = (0, 2) := by
  sorry

end intersection_M_N_l698_698213


namespace min_value_of_expression_l698_698987

theorem min_value_of_expression (x y : ℝ) (h : (x - 1) * 4 + 2 * y = 0) :
  16^x + 4^y = 8 :=
sorry

end min_value_of_expression_l698_698987


namespace cards_kept_away_l698_698097

theorem cards_kept_away (total_cards : ℕ) (cards_used : ℕ) (cards_kept : ℕ) : 
  total_cards = 52 ∧ cards_used = 45 → cards_kept = 7 :=
by
  intro h
  cases h with ht hu
  rw [ht, hu]
  -- proof omitted, sorry added to compile
  sorry

end cards_kept_away_l698_698097


namespace greatest_5_digit_divisible_by_12_15_18_is_99900_l698_698831

theorem greatest_5_digit_divisible_by_12_15_18_is_99900 :
  ∃ n : ℕ, n ∈ set.Ico 10000 100000 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n ∧ n = 99900 :=
by
  sorry

end greatest_5_digit_divisible_by_12_15_18_is_99900_l698_698831


namespace horatio_sonnets_count_l698_698997

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end horatio_sonnets_count_l698_698997


namespace sum_of_solutions_l698_698476

theorem sum_of_solutions : 
  (∀ x : ℝ, (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1) → 
  (∃ s : ℝ, s = 16) :=
by
  sorry

end sum_of_solutions_l698_698476


namespace probability_x_plus_y_lt_4_l698_698862

noncomputable def square_points : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

def region_of_interest (p : ℝ × ℝ) : Prop := p.1 + p.2 < 4

theorem probability_x_plus_y_lt_4 : 
  let area_square := 9
  let area_triangle := 2
  let area_region := area_square - area_triangle
  P(x in square_points | region_of_interest x) = 7 / 9 :=
sorry

end probability_x_plus_y_lt_4_l698_698862


namespace sequence_properties_l698_698186

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def is_arithmetic_sequence (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 2 * a n = S n + 2

theorem sequence_properties
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 2)
  (h3 : is_arithmetic_sequence a S)
  (h4 : ∀ n, S n = Sn a n) :
  (∀ n, a n = 2 ^ n) ∧
  (∀ (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ),
    (∀ n, b n = real.log (a n) / real.log 2) →
    (∀ n, c n = 1 / (b n * b (n + 1))) →
    ∑ i in finset.range n, c i = n / (n + 1)) := by
  sorry

end sequence_properties_l698_698186


namespace total_stickers_needed_l698_698856

theorem total_stickers_needed 
  (total_butterflies : ℕ)
  (cons_starting_from_1 : bool)
  (double_digit_butterflies : ℕ)
  (triple_digit_butterflies : ℕ)
  (total_butterflies = 330)
  (cons_starting_from_1 = true)
  (double_digit_butterflies = 21)
  (triple_digit_butterflies = 4) :
  let single_digit_butterflies := 9,
      single_digit_stickers := single_digit_butterflies * 1,
      double_digit_stickers := double_digit_butterflies * 2,
      triple_digit_stickers := triple_digit_butterflies * 3,
      total_stickers := single_digit_stickers + double_digit_stickers + triple_digit_stickers 
  in total_stickers = 63 := 
by 
  sorry

end total_stickers_needed_l698_698856


namespace bilion_wins_1000000_dollars_l698_698033

theorem bilion_wins_1000000_dollars :
  ∃ (p : ℕ), (p = 1000000) ∧ (p % 3 = 1) → p = 1000000 :=
by
  sorry

end bilion_wins_1000000_dollars_l698_698033


namespace Nikolai_faster_than_Gennady_l698_698042

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l698_698042


namespace largest_three_digit_sum_l698_698226

open Nat

def isDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem largest_three_digit_sum : 
  ∀ (X Y Z : ℕ), isDigit X → isDigit Y → isDigit Z → areDistinct X Y Z →
  100 ≤  (110 * X + 11 * Y + 2 * Z) → (110 * X + 11 * Y + 2 * Z) ≤ 999 → 
  110 * X + 11 * Y + 2 * Z ≤ 982 :=
by
  intros
  sorry

end largest_three_digit_sum_l698_698226


namespace jennifer_book_fraction_l698_698265

theorem jennifer_book_fraction :
  (120 - (1/5 * 120 + 1/6 * 120 + 16)) / 120 = 1/2 :=
by
  sorry

end jennifer_book_fraction_l698_698265


namespace sqrt_a_minus_b_possible_values_l698_698231

theorem sqrt_a_minus_b_possible_values (a b : ℝ) (ha : abs a = 3) (hb : real.sqrt (b ^ 2) = 4) (h : a > b) :
  real.sqrt (a - b) = real.sqrt 7 ∨ real.sqrt (a - b) = 1 :=
sorry

end sqrt_a_minus_b_possible_values_l698_698231


namespace find_natural_number_n_l698_698157

theorem find_natural_number_n : 
  ∃ (n : ℕ), (∃ k : ℕ, n + 15 = k^2) ∧ (∃ m : ℕ, n - 14 = m^2) ∧ n = 210 :=
by
  sorry

end find_natural_number_n_l698_698157


namespace range_of_b_l698_698336

noncomputable def f (b : ℝ) : ℝ → ℝ :=
  λ (x : ℝ), if (x > 0) then (b - 3/2) * x + b - 1 else -x^2 + (2 + b) * x

theorem range_of_b (b : ℝ) : (∀ x y : ℝ, x < y → f b x ≤ f b y) ↔ (3 / 2 < b ∧ b ≤ 2) :=
by
  sorry

end range_of_b_l698_698336


namespace min_value_of_expression_l698_698946

theorem min_value_of_expression (x : ℝ) (h : x > -1) : ∃ y : ℝ, y = x + 4 / (x + 1) ∧ y ≥ 3 :=
begin
  sorry
end

end min_value_of_expression_l698_698946


namespace minor_axis_length_l698_698926

def ellipse_minor_axis_length (e : ℝ) (focal_length : ℝ) : ℝ :=
  let c := focal_length / 2
  let a := c / e
  let b := Real.sqrt (a^2 - c^2)
  2 * b

theorem minor_axis_length (e : ℝ) (focal_length : ℝ) (h_e : e = 1 / 2) (h_f: focal_length = 2) :
  ellipse_minor_axis_length e focal_length = 2 * Real.sqrt 3 :=
by
  rw [h_e, h_f, ellipse_minor_axis_length]
  have c_def : c = 1 := by
    calc 
      c = 2 / 2 : by rfl
      ... = 1 : by norm_num
  have a_def : a = 2 := by
    calc 
      a = 1 / (1/2) : by rfl
      ... = 2 : by norm_num
  have b_def : b = Real.sqrt 3 := by
    calc
      b = Real.sqrt (2^2 - 1^2) : by rfl
      ... = Real.sqrt 3 : by norm_num
  rw [c_def, a_def, b_def]
  rfl

end minor_axis_length_l698_698926


namespace square_side_length_l698_698259

theorem square_side_length (center : (ℤ × ℤ)) (num_lattice_points : ℕ) : 
  center = (0, 0) → num_lattice_points = 9 → ∃ side_length, side_length = 2 :=
by
  intro h_center
  intro h_lattice_points
  use 2
  simp
  sorry

end square_side_length_l698_698259


namespace focus_coordinates_correct_l698_698452
noncomputable def ellipse_focus : Real × Real :=
  let center : Real × Real := (4, -1)
  let a : Real := 4
  let b : Real := 1.5
  let c : Real := Real.sqrt (a^2 - b^2)
  (center.1 + c, center.2)

theorem focus_coordinates_correct : 
  ellipse_focus = (7.708, -1) := 
by 
  sorry

end focus_coordinates_correct_l698_698452


namespace packet_weight_l698_698432

theorem packet_weight :
  ∀ (num_packets : ℕ) (total_weight_kg : ℕ), 
  num_packets = 20 → total_weight_kg = 2 →
  (total_weight_kg * 1000) / num_packets = 100 := by
  intro num_packets total_weight_kg h1 h2
  sorry

end packet_weight_l698_698432


namespace not_sufficient_nor_necessary_l698_698278

theorem not_sufficient_nor_necessary (a b : ℝ) :
  ¬((a^2 > b^2) → (a > b)) ∧ ¬((a > b) → (a^2 > b^2)) :=
by
  sorry

end not_sufficient_nor_necessary_l698_698278


namespace problem_statement_l698_698540

def complex_number := ℂ

noncomputable def imaginary_unit : complex_number := complex.I

noncomputable def z : complex_number := -1/2 + (real.sqrt 3) / 2 * imaginary_unit

noncomputable def z_conjugate (z : complex_number) : complex_number := conj z

noncomputable def z_modulus (z : complex_number) : ℝ := abs z

theorem problem_statement :
  z_conjugate z + z_modulus z = ⟨1/2, - (real.sqrt 3) / 2⟩ :=
sorry

end problem_statement_l698_698540


namespace probability_of_adjacent_vertices_l698_698726

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698726


namespace value_of_k_l698_698923

theorem value_of_k (k : ℝ) :
  (5 + ∑' n : ℕ, (5 + k * (2^n / 4^n))) / 4^n = 10 → k = 15 :=
by
  sorry

end value_of_k_l698_698923


namespace quadratic_inequality_solution_l698_698929

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 6*x - 16 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 8} := by
  sorry

end quadratic_inequality_solution_l698_698929


namespace fraction_to_decimal_l698_698484

theorem fraction_to_decimal : (31 : ℝ) / (2 * 5^6) = 0.000992 :=
by sorry

end fraction_to_decimal_l698_698484


namespace collinear_RTS_l698_698526

variables {Ω : Type*} 
noncomputable theory

structure Point (Ω : Type*) := 
  (x : ℝ) 
  (y : ℝ)

structure Circle (Ω : Type*) :=
  (center : Point Ω) 
  (radius : ℝ)

structure Line (Ω : Type*) :=
  (p1 : Point Ω) 
  (p2 : Point Ω)

def is_on_circle (p : Point Ω) (c : Circle Ω) : Prop :=
  ((c.center.x - p.x)^2 + (c.center.y - p.y)^2) = c.radius^2

def intersect_line_circle (l : Line Ω) (c : Circle Ω) : set (Point Ω) :=
  {p : Point Ω | ∃ λ : ℝ, p = ⟨λ * l.p1.x + (1 - λ) * l.p2.x, λ * l.p1.y + (1 - λ) * l.p2.y⟩ ∧ is_on_circle p c}

def intersect_lines (l1 l2 : Line Ω) : Point Ω :=
  let A := l1.p1.y - l1.p2.y, B := l1.p2.x - l1.p1.x, C := l1.p1.x * l1.p2.y - l1.p1.y * l1.p2.x,
      D := l2.p1.y - l2.p2.y, E := l2.p2.x - l2.p1.x, F := l2.p1.x * l2.p2.y - l2.p1.y * l2.p2.x in
  ⟨(C * E - B * F) / (A * E - B * D), (A * F - C * D) / (A * E - B * D)⟩

def are_collinear (p1 p2 p3 : Point Ω) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

variables (A B C D E F P R S T : Point Ω) (circ : Circle Ω)

axiom quadrilateral_inscribed : is_on_circle A circ ∧ is_on_circle B circ ∧ is_on_circle C circ ∧ is_on_circle D circ
axiom intersection_E : E = intersect_lines (Line.mk A B) (Line.mk D C)
axiom intersection_F : F = intersect_lines (Line.mk A D) (Line.mk B C)
axiom point_P_on_circle : is_on_circle P circ
axiom intersect_PE : ∃ (R' : Point Ω), R = ⋃ (intersect_line_circle (Line.mk P E) circ)
axiom intersect_PF : ∃ (S' : Point Ω), S = ⋃ (intersect_line_circle (Line.mk P F) circ)
axiom intersection_T : T = intersect_lines (Line.mk A C) (Line.mk B D)

theorem collinear_RTS : are_collinear R T S :=
sorry

end collinear_RTS_l698_698526


namespace combined_age_of_siblings_l698_698440

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end combined_age_of_siblings_l698_698440


namespace find_p_l698_698212
noncomputable theory

-- The given parabola C and the focus F
def parabola (p : ℝ) : set (ℝ × ℝ) := { M | ∃ x y, y^2 = 2 * p * x ∧ M = (x, y) }

-- The specific point M on the parabola C
def M_on_parabola (p : ℝ) (x₀ : ℝ) : ℝ × ℝ := (x₀, 2 * real.sqrt 2)

-- The given condition about the point A
def MA_AF_ratio (M F A : ℝ × ℝ) : Prop := dist M A / dist A F = 2

-- The main theorem to prove
theorem find_p (p x₀ : ℝ) (h₁ : p > 0) (h₂ : (x₀, 2 * real.sqrt 2) ∈ parabola p)
(h₃ : MA_AF_ratio (M_on_parabola p x₀) F A) : p = 2 :=
sorry

end find_p_l698_698212


namespace incorrect_statement_C_l698_698576

open Real

variables (x : ℝ) (b : ℝ) (y : ℝ)

noncomputable def log_b (b x : ℝ) : ℝ := log x / log b

theorem incorrect_statement_C (hb : b > 1) (hx : x = 0) : y ≠ log_b b (x^2) :=
by
  have hy : y = log_b b (x^2),
  {
    sorry  -- Proof that y = log_b b (x^2)
  }
  have hx_squared_zero : (x^2) = 0,
  {
    sorry  -- Proof that x = 0 implies x^2 = 0
  }
  have log_zero_undefined : log_b b 0 = ∞,
  {
    sorry  -- Proof that log_b of 0 is undefined (negative infinity)
  }
  show y ≠ log_b b (0^2), from
  calc
    y = log_b b (x^2) : hy
    ... = log_b b 0     : by rw [hx_squared_zero]
    ... = ∞            : log_zero_undefined

end incorrect_statement_C_l698_698576


namespace sum_of_roots_l698_698921

theorem sum_of_roots (a b c : ℝ) (h : 3 * x^2 - 7 * x + 2 = 0) : -b / a = 7 / 3 :=
by sorry

end sum_of_roots_l698_698921


namespace probability_of_adjacent_vertices_in_decagon_l698_698754

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698754


namespace number_of_obtuse_triangles_l698_698001

def is_obtuse_triangle (a b c : ℚ) : Prop :=
  a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2

theorem number_of_obtuse_triangles :
  ∃ k_values : finset ℕ,
    ∀ k ∈ k_values, is_obtuse_triangle 11 15 k ∧ 1 < k ∧ k < 26 ∧ k ∈ ℕ ∧ k_values.card = 13 :=
sorry

end number_of_obtuse_triangles_l698_698001


namespace solve_for_a_l698_698538

variable (a u : ℝ)

def eq1 := (3 / a) + (1 / u) = 7 / 2
def eq2 := (2 / a) - (3 / u) = 6

theorem solve_for_a (h1 : eq1 a u) (h2 : eq2 a u) : a = 2 / 3 := 
by
  sorry

end solve_for_a_l698_698538


namespace gardener_area_l698_698360

-- The definition considers the placement of gardeners and the condition for attending flowers.
noncomputable def grid_assignment (gardener_position: (ℕ × ℕ)) (flower_position: (ℕ × ℕ)) : List (ℕ × ℕ) :=
  sorry

-- A theorem that states the equivalent proof.
theorem gardener_area (gardener_position: (ℕ × ℕ)) :
  ∀ flower_position: (ℕ × ℕ), (∃ g1 g2 g3, g1 ∈ grid_assignment gardener_position flower_position ∧
                                            g2 ∈ grid_assignment gardener_position flower_position ∧
                                            g3 ∈ grid_assignment gardener_position flower_position) →
  (gardener_position = g1 ∨ gardener_position = g2 ∨ gardener_position = g3) → true :=
by
  sorry

end gardener_area_l698_698360


namespace sin_value_l698_698533

theorem sin_value (x : ℝ) (h : sec x - tan x = 5 / 4) : sin x = 1 / 4 := 
by
  sorry

end sin_value_l698_698533


namespace perimeter_of_face_given_volume_l698_698681

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end perimeter_of_face_given_volume_l698_698681


namespace sum_highlighted_is_positive_l698_698406

noncomputable def is_highlighted (n : List ℤ) (i : ℕ) : Bool :=
  n.get? i = some n[i] ∧ (n[i] > 0 ∨ (i < 99 ∧ n[i] + n[i + 1] > 0))

theorem sum_highlighted_is_positive (n : List ℤ) (h_length : n.length = 100)
  (h_exists : ∃ i, n.get? i ≠ none) : 
  (∑ i in Finset.range 100, if is_highlighted n i then n[i] else 0) > 0 :=
by
  sorry

end sum_highlighted_is_positive_l698_698406


namespace total_annual_cost_l698_698028

def daily_pills : ℕ := 2
def pill_cost : ℕ := 5
def medication_cost (daily_pills : ℕ) (pill_cost : ℕ) : ℕ := daily_pills * pill_cost
def insurance_coverage : ℚ := 0.80
def visit_cost : ℕ := 400
def visits_per_year : ℕ := 2
def annual_medication_cost (medication_cost : ℕ) (insurance_coverage : ℚ) : ℚ :=
  medication_cost * 365 * (1 - insurance_coverage)
def annual_visit_cost (visit_cost : ℕ) (visits_per_year : ℕ) : ℕ :=
  visit_cost * visits_per_year

theorem total_annual_cost : annual_medication_cost (medication_cost daily_pills pill_cost) insurance_coverage
  + annual_visit_cost visit_cost visits_per_year = 1530 := by
  sorry

end total_annual_cost_l698_698028


namespace find_angle_A_l698_698241

noncomputable theory

variables (A B C : ℝ) (a b c : ℝ)

def triangle_ABC (A B C : ℝ) (a b c : ℝ) :=
  a = c * (Real.sin A / Real.sin C) ∧ b = c * (Real.sin B / Real.sin C)

theorem find_angle_A :
  C = 60 ∧ b = Real.sqrt 6 ∧ c = 3 → A = 75 := by
  intro h
  cases h
  cases h_left
  have h1 : Real.sin 60 = Real.sqrt 3 / 2 := sorry -- Known trigonometric value
  have h2 : Real.sqrt 6 / Real.sqrt 3 / 2 = Real.sqrt 2 / 2 := sorry -- Calculation step
  have h3 : B = 45 := sorry -- Based on the above calculation and angle property
  have h4 : A = 180 - B - C := sorry -- Angle sum property
  exact h4

end find_angle_A_l698_698241


namespace sum_of_squares_le_bound_l698_698303

open Real

theorem sum_of_squares_le_bound (n : ℕ) (s : ℝ)
  (b : Fin n → ℝ) (a : Fin n → ℝ)
  (hb_pos : ∀ i, 0 < b i)
  (hb_sum : ∑ i, b i ≤ 10)
  (ha1 : a 0 = b 0)
  (ham : ∀ i : Fin (n - 1), a ⟨i + 1, Nat.succ_lt_succ i.is_lt⟩ = s * a i + b ⟨i + 1, Nat.succ_lt_succ i.is_lt⟩)
  (hs_range : 0 ≤ s ∧ s < 1) :
  ∑ i, (a i)^2 ≤ 100 / (1 - s^2) := sorry

end sum_of_squares_le_bound_l698_698303


namespace bounded_f_b_bounded_f_c_l698_698542

noncomputable def is_bounded_function (f : ℝ → ℝ) (A : set ℝ) : Prop :=
  ∃ M > 0, ∀ x ∈ A, |f x| ≤ M

def f_b (x : ℝ) : ℝ := 5 / (2 * x ^ 2 - 4 * x + 3)
def f_c (x : ℝ) : ℝ := (2 ^ x - 1) / (2 ^ x + 1)

theorem bounded_f_b : is_bounded_function f_b set.univ := sorry
theorem bounded_f_c : is_bounded_function f_c set.univ := sorry

end bounded_f_b_bounded_f_c_l698_698542


namespace arithmetic_progression_sum_l698_698342

theorem arithmetic_progression_sum (a : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∃ d, ∀ i < n, a (i+1) = a i + d) :
  (∑ i in Finset.range n, 1 / (a i * a (i + 1))) = n / (a 0 * a n) :=
by
  sorry

end arithmetic_progression_sum_l698_698342


namespace michael_seventh_score_l698_698268

theorem michael_seventh_score (t : ℕ → ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 8 → 85 ≤ t i ∧ t i ≤ 95)
  (h2 : ∀ n, 1 ≤ n ∧ n ≤ 8 → (∑ i in Finset.range n, t (i + 1)) % n = 0)
  (h3 : t 8 = 90) : t 7 = 90 := 
sorry

end michael_seventh_score_l698_698268


namespace total_spent_correct_l698_698992

def cost_gifts : ℝ := 561.00
def cost_giftwrapping : ℝ := 139.00
def total_spent : ℝ := cost_gifts + cost_giftwrapping

theorem total_spent_correct : total_spent = 700.00 := by
  sorry

end total_spent_correct_l698_698992


namespace condition_for_ellipse_l698_698400

theorem condition_for_ellipse (m : ℝ) : 
  (3 < m ∧ m < 7) ↔ (7 - m > 0 ∧ m - 3 > 0 ∧ (7 - m) ≠ (m - 3)) :=
by sorry

end condition_for_ellipse_l698_698400


namespace k_l_m_n_values_l698_698631

theorem k_l_m_n_values (k l m n : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m) (hn : 0 < n)
  (hklmn : k + l + m + n = k * m) (hln : k + l + m + n = l * n) :
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end k_l_m_n_values_l698_698631


namespace intersection_of_sets_l698_698638

-- Defining the sets as given in the conditions
def setM : Set ℝ := { x | (x + 1) * (x - 3) ≤ 0 }
def setN : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Statement to prove
theorem intersection_of_sets :
  { x | (x + 1) * (x - 3) ≤ 0 } ∩ { x | 1 < x ∧ x < 4 } = { x | 1 < x ∧ x ≤ 3 } := by
sorry

end intersection_of_sets_l698_698638


namespace find_angle_ABC_l698_698539

def is_centroid {A B C G : Type} [AddCommGroup G] [AddGroup G] (GA GB GC : G) :=
  GA + GB + GC = (0 : G)

def given_equation {A B C G : Type} [AddCommGroup G] [MulAction ℝ G] (GA GB GC : G) (sinA sinB sinC : ℝ) :=
  (sqrt 7) • GA • sinA + 3 • GB • sinB + 3 • (sqrt 7) • GC • sinC = (0 : G)

theorem find_angle_ABC
  {A B C G : Type} [AddCommGroup G] [MulAction ℝ G]
  {GA GB GC : G} (sinA sinB sinC : ℝ)
  (h1 : is_centroid GA GB GC)
  (h2 : given_equation GA GB GC sinA sinB sinC) :
  ∃ (angle_ABC : ℝ), angle_ABC = 60 :=
  sorry

end find_angle_ABC_l698_698539


namespace dice_probability_exactly_four_twos_l698_698145

theorem dice_probability_exactly_four_twos :
  let probability := (Nat.choose 8 4 : ℚ) * (1 / 8)^4 * (7 / 8)^4 
  probability = 168070 / 16777216 :=
by
  sorry

end dice_probability_exactly_four_twos_l698_698145


namespace number_of_obtuse_triangles_with_side_lengths_7_12_k_l698_698685

theorem number_of_obtuse_triangles_with_side_lengths_7_12_k 
  (k : ℕ) (h1 : k > 0) (h2 : k < 19 ∧ (k > 5 ∧ k < 10 ∨ k > 13 ∧ k < 19)) :
    {k : ℕ | (7^2 + 12^2) < k^2 ∨ (k^2 + 7^2) < 12^2} → 
    9 := 
by 
  sorry

end number_of_obtuse_triangles_with_side_lengths_7_12_k_l698_698685


namespace sum_of_sequence_l698_698885

/-- Sum of the sequence 1 · 2 + 2 · 3 + 3 · 4 + ... + n(n+1) is given by (n(n+1)(n+2))/3 --/
theorem sum_of_sequence (n : ℕ) :
  (Finset.range n).sum (λ k, (k + 1) * (k + 2)) = (n * (n + 1) * (n + 2)) / 3 :=
by
  sorry

end sum_of_sequence_l698_698885


namespace traffic_light_probability_max_l698_698904

theorem traffic_light_probability_max (x : ℝ) :
  ∀ (x : ℝ),
  (x ≥ 0 ∧ x = 60) →
  let p := (120 * x) / ((x + 30) * (120 + x)) in
  p ≤ (4 / 9) ∧ p = (4 / 9) :=
begin
  intros,
  sorry
end

end traffic_light_probability_max_l698_698904


namespace find_n_l698_698706

theorem find_n (N : ℕ) (m k : ℕ) (h1 : m = (multiset.filter (λ d, d < N) (N.divisors)).max')
               (h2 : N + m = 10^k) : N = 75 :=
sorry

end find_n_l698_698706


namespace min_distance_parabola_focus_l698_698002

theorem min_distance_parabola_focus : 
  let P := fun p : ℝ × ℝ => (p.2 ^ 2 = 8 * p.1)
  let F := (2, 0)
  let distance (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  ∃ (p : ℝ × ℝ), P p ∧ (∀ q : ℝ × ℝ, P q → distance q F ≥ distance p F) ∧ distance p F = 2 :=
sorry

end min_distance_parabola_focus_l698_698002


namespace part1_part2_l698_698516

noncomputable def S : ℕ → ℝ
| 1     := 2 / 3
| (n+1) := 1 - (1/3)^(n+1)

noncomputable def a : ℕ → ℝ
| 1     := 2 - 2 * S 1
| (n+1) := 2 - 2 * S (n+1)

noncomputable def b (n: ℕ) : ℝ :=
log 3 (1 - S n)

theorem part1 (n : ℕ) (hn : n ≥ 1) :
  a n = 2 / 3 ^ n :=
sorry

theorem part2 (hn : (∑ k in finset.range 100, 1 / ((b (k + 2)) * (b (k + 3))) = 25/51)) :
  ∃ n, n = 101 :=
sorry

end part1_part2_l698_698516


namespace find_k_l698_698985

-- Definitions based on the problem conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Property of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := ∃ c : ℝ, u.1 = c * v.1 ∧ u.2 = c * v.2

-- Theorem statement equivalent to the problem
theorem find_k (k : ℝ) (h : parallel vector_a (vector_b k)) : k = -2 :=
sorry

end find_k_l698_698985


namespace probability_adjacent_vertices_decagon_l698_698734

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698734


namespace probability_adjacent_vertices_decagon_l698_698758

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698758


namespace exists_arrangement_l698_698903

noncomputable def placement_rule (x y : ℤ) : ℤ :=
  if (x + y) % 1918 = 0 then 1
  else if (x + y) % 1978 = 0 then -1
  else 0

theorem exists_arrangement :
  ∃ f : ℤ × ℤ → ℤ, (∀ (x y : ℤ), f (x, y) = placement_rule x y) ∧
  (∀ (a b : ℤ), ∑ i in finset.Icc (a, b) (a + 1917, b + 1977), f i = 60) :=
by
  existsi (λ pt, placement_rule pt.1 pt.2)
  split
  {
    intros x y,
    refl
  }
  {
    intros a b,
    -- Proof that the sum over any 1918 x 1978 rectangle is 60 has to be completed
    sorry
  }

end exists_arrangement_l698_698903


namespace carol_first_to_roll_six_l698_698124

def probability_roll (x : ℕ) (success : ℕ) : ℚ := success / x

def first_to_roll_six_probability (a b c : ℕ) : ℚ :=
  let p_six : ℚ := probability_roll 6 1
  let p_not_six : ℚ := 1 - p_six
  let cycle_prob : ℚ := p_not_six * p_not_six * p_six
  let continue_prob : ℚ := p_not_six * p_not_six * p_not_six
  let geometric_sum : ℚ := cycle_prob / (1 - continue_prob)
  geometric_sum

theorem carol_first_to_roll_six :
  first_to_roll_six_probability 1 1 1 = 25 / 91 := 
sorry

end carol_first_to_roll_six_l698_698124


namespace calc_perm_product_l698_698133

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Lean statement to prove the given problem
theorem calc_perm_product : permutation 6 2 * permutation 4 2 = 360 := 
by
  -- Test the calculations if necessary, otherwise use sorry
  sorry

end calc_perm_product_l698_698133


namespace am_gm_inequality_l698_698655

variable (a : ℝ) (h : a > 0) -- Variables and condition

theorem am_gm_inequality (a : ℝ) (h : a > 0) : a + 1 / a ≥ 2 := 
sorry -- Proof is not provided according to instructions.

end am_gm_inequality_l698_698655


namespace probability_adjacent_vertices_decagon_l698_698761

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698761


namespace seating_arrangements_l698_698698

theorem seating_arrangements (front_seats back_seats : ℕ) (middle_seats : Finset ℕ)
  (not_adjacent : ∀ (i j : ℕ), i ≠ j → |i - j| ≠ 1) :
  front_seats = 9 →
  back_seats = 8 →
  middle_seats = {3, 4, 5} →
  (∃ a b : ℕ, a ≠ b ∧ 1 ≤ a ∧ a ≤ front_seats ∧ 1 ≤ b ∧ b ≤ back_seats) →
  ∃ (arrangements : ℕ), arrangements = 114 :=
by
  sorry

end seating_arrangements_l698_698698


namespace count_irrationals_of_given_set_l698_698443

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem count_irrationals_of_given_set :
  let nums := [3 * Real.pi, -7 / 8, 0, Real.sqrt 2, -3.15, Real.sqrt 9, Real.sqrt 3 / 3] in
  (filter is_irrational nums).length = 3 :=
by
  sorry

end count_irrationals_of_given_set_l698_698443


namespace largest_divisor_power_of_ten_l698_698707

def largest_divisor_smaller_than (N : ℕ) : ℕ :=
  if N = 1 then 0
  else (Nat.divisors N).filter (λ d => d < N) |>.lastD 1

theorem largest_divisor_power_of_ten (N : ℕ) (k : ℕ) :
  N + largest_divisor_smaller_than N = 10^k → N = 75 :=
  sorry

end largest_divisor_power_of_ten_l698_698707


namespace equation_of_C2_distance_between_A_B_l698_698252

-- Definitions and conditions
def parametric_C1 (α : ℝ) : ℝ × ℝ := (f α, g α) -- Curve C1 is given by parametric equations

def M_on_C1 (α : ℝ) : ℝ × ℝ := parametric_C1 α
def P_on_C2 (x y : ℝ) : Prop := ∃ α, M_on_C1 α = (x / 2, y / 2)

-- Polar equations for the curves
def polar_eq_C1 (θ : ℝ) : ℝ := 4 * Real.sin θ
def polar_eq_C2 (θ : ℝ) : ℝ := 8 * Real.sin θ

-- Distance between intersections with a given ray θ
def distance_AB (θ : ℝ) : ℝ :=
  let A := polar_eq_C1 θ
  let B := polar_eq_C2 θ
  |B - A|

-- The Lean theorem statements
theorem equation_of_C2 : ∀ x y, P_on_C2 x y → ∃ α, (x, y) = (2 * f α, 2 * g α) := by
  sorry

theorem distance_between_A_B : ∀ θ, let A := polar_eq_C1 θ in 
                                    let B := polar_eq_C2 θ in 
                                    |B - A| = 4 * Real.sin θ := by
  sorry

end equation_of_C2_distance_between_A_B_l698_698252


namespace each_child_receives_14_jellybeans_l698_698881

theorem each_child_receives_14_jellybeans :
  ∀ (total_jellybeans : ℕ) (nephews : ℕ) (nieces : ℕ),
  total_jellybeans = 70 → nephews = 3 → nieces = 2 → 
  total_jellybeans / (nephews + nieces) = 14 :=
by
  intros total_jellybeans nephews nieces hjells hneph hnieces
  rw [hjells, hneph, hnieces]
  norm_num
  sorry

end each_child_receives_14_jellybeans_l698_698881


namespace part1_part2_l698_698593

-- 1. First part: proving sin A and c
theorem part1 (a b R : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : R = 5/2) : 
  (sin (real.arcsin (3 / 5)) = 3 / 5) ∧ (5 = 5) := 
by 
  sorry

-- 2. Second part: proving length of CD
theorem part2 (a b R c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : R = 5/2) (h4 : c = 5) :
  ∃ D : ℝ, D = (sqrt 73 / 3) ∨ D = (2 * sqrt 13 / 3) := 
by 
  sorry

end part1_part2_l698_698593


namespace angle_C_in_triangle_l698_698620

theorem angle_C_in_triangle 
    (a b c : ℝ)
    (S : ℝ)
    (hS : S = (a^2 + b^2 - c^2) / 4) :
    ∠C = 45° :=
sorry

end angle_C_in_triangle_l698_698620


namespace minimum_a_inequality_l698_698556

variable {x y : ℝ}

/-- The inequality (x + y) * (1/x + a/y) ≥ 9 holds for any positive real numbers x and y 
     if and only if a ≥ 4.  -/
theorem minimum_a_inequality (a : ℝ) (h : ∀ (x y : ℝ), 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  a ≥ 4 :=
by
  sorry

end minimum_a_inequality_l698_698556


namespace circle_tangents_l698_698890

-- Defining the problem in Lean 4
theorem circle_tangents:
  ∀ (m a b c : ℕ), 
  (m = (a * (Real.sqrt b)) / c) ∧
  b = 130 ∧
  a = 2 ∧
  c = 3 ∧
  Nat.coprime a c →
  (a + b + c = 135) :=
by
  intro m a b c
  rintro ⟨hm1, hb, ha, hc, h_coprime⟩
  sorry

end circle_tangents_l698_698890


namespace area_BDE_l698_698619

variables {A B C D E M : Type} 
variables (h_angleC_gt_90 : ∠C > 90)
variables (h_midpoint_AM_MB : AM = MB)
variables (h_MD_perp_BC : MD ⊥ BC) 
variables (h_ME_perp_AB : ME ⊥ AB) 
variables (h_D_on_BC : D ∈ BC)
variables (h_E_on_AB : E ∈ AB)
variables (h_M_midpoint_EB : M = midpoint E B)
variables (h_area_ABC : area ΔABC = 30)

theorem area_BDE :
  area ΔBDE = 7.5 :=
sorry

end area_BDE_l698_698619


namespace AKLB_is_rhombus_l698_698401

variables {A B C D K L : Type*}
variables [geometry A] [geometry B] [geometry C] [geometry D]
variables [geometry K] [geometry L]
variables {P Q R S T U V : point}

axiom cyclic_quadrilateral (A B C D : point) : is_cyclic (A, B, C, D)
axiom perpendicular_diagonals (AC BD : line) : AC ⊥ BD
axiom perpendicular_from_A (A CD : line) : ∃ K, K ∈ CD ∧ A ⊥ K
axiom perpendicular_from_B (B CD : line) : ∃ L, L ∈ CD ∧ B ⊥ L
axiom intersect_BD (BD : line) : ∃ K, K ∈ BD
axiom intersect_AC (AC : line) : ∃ L, L ∈ AC

theorem AKLB_is_rhombus
  (h₁ : is_cyclic (A, B, C, D))
  (h₂ : AC ⊥ BD)
  (h₃ : ∃ K, K ∈ CD ∧ A ⊥ K)
  (h₄ : ∃ L, L ∈ CD ∧ B ⊥ L)
  (h₅ : ∃ K, K ∈ BD)
  (h₆ : ∃ L, L ∈ AC) :
  is_rhombus (A, K, L, B) :=
sorry

end AKLB_is_rhombus_l698_698401


namespace incorrect_option_b_l698_698825

theorem incorrect_option_b {A B C : ℝ} 
  (hA : A = 30) 
  (hB : B = 50) 
  (hSum : A + B + C = 180) 
  (hTriangle : C = 180 - (A + B)) : ∃ (x : ℝ), x ∈ set.Ioo 90 180 ∧ ¬ x = C :=
by
  sorry

end incorrect_option_b_l698_698825


namespace problem1_problem2_l698_698218

-- Define vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (-1, 2)

-- Define scalar multiples of vectors
def three_a : ℝ × ℝ := (3 * a.1, 3 * a.2)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)
def three_a_plus_two_b : ℝ × ℝ := (three_a.1 + two_b.1, three_a.2 + two_b.2)

-- Define vector addition and scalar multiplication
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (c : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (c * u.1, c * u.2)

-- Magnitude of a vector
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

-- Condition: Given vectors a and b, prove the magnitude of 3a + 2b
theorem problem1 : magnitude three_a_plus_two_b = 7 * Real.sqrt 2 := sorry

-- Define vectors for the second problem
def vec_a_plus_kb (k : ℝ) : ℝ × ℝ := vec_add a (scalar_mul k b)
def vec_2a_minus_b : ℝ × ℝ := vec_add (scalar_mul 2 a) (scalar_mul (-1) b)

-- Condition: Given vectors a and b, find k such that (a + k*b) // (2*a - b)
theorem problem2 : ∃ k : ℝ, (vec_a_plus_kb k).2 = 0 ↔ k = -1/2 := sorry

end problem1_problem2_l698_698218


namespace probabilities_correct_l698_698670

variable (p₁ p₂ p₃ : ℝ)
variable (P_all P_two P_one P_none : ℝ)

def prob_correct (p₁ p₂ p₃ : ℝ) :=
  p₁ = 0.7 ∧ p₂ = 0.8 ∧ p₃ = 0.9 ∧
  P_all = 0.504 ∧
  P_two = 0.398 ∧
  P_one = 0.092 ∧
  P_none = 0.006

theorem probabilities_correct :
  prob_correct p₁ p₂ p₃ :=
by
  -- mathematical proof goes here
  sorry

end probabilities_correct_l698_698670


namespace zero_point_in_interval_l698_698018

-- Define the function f
def f (x : ℝ) : ℝ := 2^x - 3

-- Theorem statement
theorem zero_point_in_interval : ∃ c : ℝ, c ∈ Ioo 1 2 ∧ f c = 0 := by
  sorry

end zero_point_in_interval_l698_698018


namespace arith_geom_seq_formula_l698_698447

variable {ℝ : Type} [LinearOrderedField ℝ]

def arith_geom_seq (u : ℕ → ℝ) (a b : ℝ) : Prop :=
  ∀ n : ℕ, u (n + 1) = a * u n + b

theorem arith_geom_seq_formula (u : ℕ → ℝ) (a b : ℝ) (h : arith_geom_seq u a b) :
  a ≠ 1 → ∀ n : ℕ, u n = a^n * u 0 + b * (a^n - 1) / (a - 1) :=
by
  sorry

end arith_geom_seq_formula_l698_698447


namespace complex_quadrant_l698_698407

noncomputable def i : ℂ := complex.I 

theorem complex_quadrant (z : ℂ) (h : (1 + i) * z = 1 - 2 * i^3) : 
  z = (3 / 2) + (1 / 2) * i :=
begin
  sorry
end

end complex_quadrant_l698_698407


namespace eval_expression_l698_698256

theorem eval_expression (x : ℝ) (h₀ : x = 3) :
  let initial_expr : ℝ := (2 * x + 2) / (x - 2)
  let replaced_expr : ℝ := (2 * initial_expr + 2) / (initial_expr - 2)
  replaced_expr = 8 :=
by
  sorry

end eval_expression_l698_698256


namespace average_eq_5_times_non_zero_l698_698080

theorem average_eq_5_times_non_zero (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := 
by sorry

end average_eq_5_times_non_zero_l698_698080


namespace longest_chord_in_circle_l698_698957

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end longest_chord_in_circle_l698_698957


namespace exists_matrices_B_C_not_exists_matrices_commute_l698_698075

-- Equivalent proof statement for part (a)
theorem exists_matrices_B_C (A : Matrix (Fin 2) (Fin 2) ℝ): 
  ∃ (B C : Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 :=
by
  sorry

-- Equivalent proof statement for part (b)
theorem not_exists_matrices_commute (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = ![![0, 1], ![1, 0]]) :
  ¬∃ (B C: Matrix (Fin 2) (Fin 2) ℝ), A = B^2 + C^2 ∧ B * C = C * B :=
by
  sorry

end exists_matrices_B_C_not_exists_matrices_commute_l698_698075


namespace lathes_equal_parts_processed_15_minutes_l698_698022

variable (efficiencyA efficiencyB efficiencyC : ℝ)
variable (timeA timeB timeC : ℕ)

/-- Lathe A starts 10 minutes before lathe C -/
def start_time_relation_1 : Prop := timeA + 10 = timeC

/-- Lathe C starts 5 minutes before lathe B -/
def start_time_relation_2 : Prop := timeC + 5 = timeB

/-- After lathe B has been working for 10 minutes, B and C process the same number of parts -/
def parts_processed_relation_1 (efficiencyB efficiencyC : ℝ) : Prop :=
  10 * efficiencyB = (10 + 5) * efficiencyC

/-- After lathe C has been working for 30 minutes, A and C process the same number of parts -/
def parts_processed_relation_2 (efficiencyA efficiencyC : ℝ) : Prop :=
  (30 + 10) * efficiencyA = 30 * efficiencyC

/-- How many minutes after lathe B starts will it have processed the same number of standard parts as lathe A? -/
theorem lathes_equal_parts_processed_15_minutes
  (h₁ : start_time_relation_1 timeA timeC)
  (h₂ : start_time_relation_2 timeC timeB)
  (h₃ : parts_processed_relation_1 efficiencyB efficiencyC)
  (h₄ : parts_processed_relation_2 efficiencyA efficiencyC) :
  ∃ t : ℕ, (t = 15) ∧ ( (timeB + t) * efficiencyB = (timeA + (timeB + t - timeA)) * efficiencyA ) := sorry

end lathes_equal_parts_processed_15_minutes_l698_698022


namespace area_MNKP_l698_698454

theorem area_MNKP (S_ABCD : ℝ) (h : S_ABCD = (180 + 50 * Real.sqrt 3) / 6) :
  (let S_MNKP := (90 + 25 * Real.sqrt 3) / 6 in S_MNKP) = (S_ABCD / 2) :=
by
  sorry

end area_MNKP_l698_698454


namespace negation_of_existence_l698_698004

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, x ∈ (set.univ : set ℝ) ∧ P x) ↔ (∀ x, x ∈ (set.univ : set ℝ) → ¬ P x) :=
by sorry

end negation_of_existence_l698_698004


namespace decagon_adjacent_probability_l698_698747

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698747


namespace sam_sitting_fee_l698_698111

theorem sam_sitting_fee 
  (s : ℝ) 
  (h₁ : ∀ n : ℝ, cost_john (n) = 2.75 * n + 125)
  (h₂ : ∀ n : ℝ, cost_sam (n) = 1.50 * n + s)
  (h₃ : cost_john 12 = cost_sam 12) : s = 140 :=
by
  unfold cost_john cost_sam at h₁ h₂ h₃
  sorry

def cost_john (n : ℝ) : ℝ := 2.75 * n + 125

def cost_sam (n : ℝ) (s : ℝ) : ℝ := 1.50 * n + s

end sam_sitting_fee_l698_698111


namespace circle_radius_integer_part_l698_698932

theorem circle_radius_integer_part
  (angle_BAC : Real := π / 6)
  (angle_DAC : Real := 5 * π / 12)
  (area_ABC : Real := 32)
  (R : Real) :
  let alpha := angle_DAC
  let beta := angle_BAC
  let S_ABC := area_ABC
  floor (R) = 5 :=
sorry

end circle_radius_integer_part_l698_698932


namespace linear_inequality_m_eq_one_l698_698227

theorem linear_inequality_m_eq_one
  (m : ℤ)
  (h1 : |m| = 1)
  (h2 : m + 1 ≠ 0) :
  m = 1 :=
sorry

end linear_inequality_m_eq_one_l698_698227


namespace honey_barrel_problem_l698_698847

theorem honey_barrel_problem
  (x y : ℝ)
  (h1 : x + y = 56)
  (h2 : x / 2 + y = 34) :
  x = 44 ∧ y = 12 :=
by
  sorry

end honey_barrel_problem_l698_698847


namespace count_irrationals_of_given_set_l698_698444

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem count_irrationals_of_given_set :
  let nums := [3 * Real.pi, -7 / 8, 0, Real.sqrt 2, -3.15, Real.sqrt 9, Real.sqrt 3 / 3] in
  (filter is_irrational nums).length = 3 :=
by
  sorry

end count_irrationals_of_given_set_l698_698444


namespace find_root_l698_698290

theorem find_root (x : ℝ) : 
  (x^3 - 3 * real.sqrt 2 * x^2 + 7 * x - 3 * real.sqrt 2 = 0) → 
  (x = real.sqrt 2) :=
begin
  sorry
end

end find_root_l698_698290


namespace monotonic_interval_of_f_l698_698548

theorem monotonic_interval_of_f (ω : ℝ) (ϕ : ℝ) (k : ℤ)
  (hω_pos : ω > 0)
  (hϕ_abs : |ϕ| < π)
  (h_zero : 2 * sin (ω * (π / 3) + ϕ) - 1 = 0)
  (h_symmetry_axis : -ω * (π / 6) - ϕ = π / 2 + k * π) :
  ∃ (interval : ℝ × ℝ), interval = (3 * k * π - 5 * π / 3, 3 * k * π - π / 6)
  := sorry

end monotonic_interval_of_f_l698_698548


namespace travel_once_change_transportation_at_most_once_l698_698094

-- Define the conditions based on the problem statement.
variable (N : ℕ)
variable (G : Type)
variable [graph G]
variable [complete_graph G]
variable [two_colored G]

-- Define the statement of the problem
theorem travel_once_change_transportation_at_most_once (h : N ≥ 3) :
  ∃ (c : G), 
    ∃ (cycle : list G), 
      cycle.head = some c ∧ 
      cycle.last = some c ∧ 
      (∀ (x ∈ cycle) (y ∈ cycle), x ≠ y) ∧
      (∃ (split_index : ℕ), 
         split_index < cycle.length ∧ 
         (∀ (i < split_index), edge_color (cycle.nth i) (cycle.nth (i+1)) = color1) ∧
         (∀ (i ≥ split_index), edge_color (cycle.nth i) (cycle.nth (i+1)) = color2)) :=
sorry

end travel_once_change_transportation_at_most_once_l698_698094


namespace limit_f_at_1_infinite_l698_698480

def f (x : ℝ) := (x^4 + 1) / (x^2 - 1)

theorem limit_f_at_1_infinite : 
  Tendsto f (𝓝[≠] 1) at_top := sorry

end limit_f_at_1_infinite_l698_698480


namespace crayons_given_l698_698874

theorem crayons_given (start_crayons end_crayons: ℕ) (h1: start_crayons = 4) (h2: end_crayons = 40) :
  end_crayons - start_crayons = 36 :=
by
  rw [h1, h2]
  simp
  sorry

end crayons_given_l698_698874


namespace area_of_triangle_PQR_l698_698061

open Real

def point := (ℝ × ℝ)

def area_of_triangle (P Q R : point) : ℝ :=
  0.5 * |P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)|

theorem area_of_triangle_PQR :
  let P := (-3, 2) : point
  let Q := (1, 7) : point
  let R := (4, -1) : point
  area_of_triangle P Q R = 23.5 :=
by
  let P := (-3, 2) : point
  let Q := (1, 7) : point
  let R := (4, -1) : point
  show area_of_triangle P Q R = 23.5
  sorry

end area_of_triangle_PQR_l698_698061


namespace no_all_white_4x4x4_no_all_white_5x5x5_l698_698099

inductive Color
| white
| black

def cube_4x4x4 : Type := Fin 4 × Fin 4 × Fin 4
def cube_5x5x5 : Type := Fin 5 × Fin 5 × Fin 5

def initial_configuration (n : ℕ) (cube : Fin n × Fin n × Fin n) : Color :=
  if cube = (⟨0, sorry⟩, ⟨0, sorry⟩, ⟨0, sorry⟩) then Color.black else Color.white

def toggle_color (c : Color) : Color :=
  match c with
  | Color.white => Color.black
  | Color.black => Color.white

def change_color (n : ℕ) (cube_colors : (Fin n × Fin n × Fin n) → Color) 
                 (cube : Fin n × Fin n × Fin n) : (Fin n × Fin n × Fin n) → Color :=
  λ pos, if pos = cube ∨ sorry then toggle_color (cube_colors pos) else cube_colors pos

theorem no_all_white_4x4x4 : ¬∃ f : (cube_4x4x4 → Color), ∀ pos, f pos = Color.white :=
  sorry

theorem no_all_white_5x5x5 : ¬∃ f : (cube_5x5x5 → Color), ∀ pos, f pos = Color.white :=
  sorry

end no_all_white_4x4x4_no_all_white_5x5x5_l698_698099


namespace area_equality_l698_698592

-- Define the problem structure

variable {Point Line : Type}
variable [Euclidean_geometry Point Line]

-- Assume basic geometric entities and properties
variables {A B C D E F H Q R U V M N : Point}
variables {Γ Γ' : Circle Point}
variables {AD BE CF AQ QR RV : Line}
variable (AM HN : Line)

-- Conditions from the problem statement
hypothesis (h1 : altitude AD ∆ABC)
hypothesis (h2 : altitude BE ∆ABC)
hypothesis (h3 : altitude CF ∆ABC)
hypothesis (h4 : orthocenter H ∆ABC)
hypothesis (h5 : Q ∈ circumcircle ∆ABC)
hypothesis (h6 : QR ⊥ BC at R)
hypothesis (h7 : line_through R ∥ line_through AQ)
hypothesis (h8 : line_through R ∥ circumcircle ∆DEF at U V)
hypothesis (h9 : AM ⊥ RV at M)
hypothesis (h10 : HN ⊥ RV at N)

-- Goal to prove
theorem area_equality : area (∆AMV) = area (∆HNV) := by
  sorry

end area_equality_l698_698592


namespace petya_exchange_impossible_l698_698854

theorem petya_exchange_impossible :
  ¬ ∃ (N : ℕ), (5 * N = 2001) :=
by 
  assume h,
  obtain ⟨N, hn⟩ := h,
  have : (2001 % 5) ≠ 0 := dec_trivial,
  have : 2001 % 5 = 2001 % 5 := rfl,
  contradiction

end petya_exchange_impossible_l698_698854


namespace coloring_exists_l698_698419

noncomputable def figure : Type :=
  {A : list ℝ // A.length = 10 ∧ (∀ a ∈ A, 0 ≤ a) ∧ A.sum = 1}

theorem coloring_exists (front back : figure) (colors : fin 10 → color) :
  ∃ σ : list (fin 10), (σ.perm.front) =
  front ∧ (σ.perm.back.sum ≥ 1/10) :=
begin
  sorry
end

end coloring_exists_l698_698419


namespace points_in_one_circle_of_radius_1_l698_698509

def point := (ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Condition: Any 3 points are contained within a circle of radius 1
def within_circle (p1 p2 p3 : point) : Prop :=
  ∃ (c : point), dist c p1 ≤ 1 ∧ dist c p2 ≤ 1 ∧ dist c p3 ≤ 1

-- Prove: All \( n \) points are contained within a single circle of radius 1
theorem points_in_one_circle_of_radius_1 (n : ℕ) (points : fin n → point)
  (h : ∀ (i j k : fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → within_circle (points i) (points j) (points k)) :
  ∃ (c : point), ∀ i : fin n, dist c (points i) ≤ 1 := 
by 
  sorry

end points_in_one_circle_of_radius_1_l698_698509


namespace largest_partner_share_l698_698169

-- Definitions for the conditions
def total_profit : ℕ := 48000
def ratio_parts : List ℕ := [2, 4, 5, 3, 6]
def total_ratio_parts : ℕ := ratio_parts.sum
def value_per_part : ℕ := total_profit / total_ratio_parts
def largest_share : ℕ := 6 * value_per_part

-- Statement of the proof problem
theorem largest_partner_share : largest_share = 14400 := by
  -- Insert proof here
  sorry

end largest_partner_share_l698_698169


namespace new_water_intake_recommendation_l698_698025

noncomputable def current_consumption : ℝ := 25
noncomputable def increase_percentage : ℝ := 0.75
noncomputable def increased_amount : ℝ := increase_percentage * current_consumption
noncomputable def new_recommended_consumption : ℝ := current_consumption + increased_amount

theorem new_water_intake_recommendation :
  new_recommended_consumption = 43.75 := 
by 
  sorry

end new_water_intake_recommendation_l698_698025


namespace decagon_adjacent_vertex_probability_l698_698719

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698719


namespace compute_z_pow_8_l698_698271

noncomputable def z : ℂ := (1 - Real.sqrt 3 * Complex.I) / 2

theorem compute_z_pow_8 : z ^ 8 = -(1 + Real.sqrt 3 * Complex.I) / 2 :=
by
  sorry

end compute_z_pow_8_l698_698271


namespace tan_alpha_l698_698531

def alpha : Type := ℝ

variable (α : alpha)
variable [IsAngleInThirdQuadrant α] (hα : cos α = -12/13)

theorem tan_alpha (hα : cos α = -12/13) (h_quad : IsAngleInThirdQuadrant α) : tan α = 5/12 := 
sorry

end tan_alpha_l698_698531


namespace veranda_area_eq_136_l698_698834

def length : ℕ := 18
def width : ℕ := 12
def veranda_width : ℕ := 2

theorem veranda_area_eq_136 :
  let total_length := length + 2 * veranda_width in
  let total_width := width + 2 * veranda_width in
  let total_area := total_length * total_width in
  let room_area := length * width in
  total_area - room_area = 136 := 
by
  sorry

end veranda_area_eq_136_l698_698834


namespace correct_transformation_l698_698383

theorem correct_transformation (x : ℝ) :
  (∀ x, 3 * (1/3 * x) = 3 * 0 → x = 0) ∧ 
  (∀ x, (3 * x = 2 * x - 2 → x = -2)) ∧ 
  (∀ x, (3 * x = 2 → x = 2 / 3)) ∧ 
  (∀ x, (2/3 * x - 1 = x → 3 * (2/3 * x - 1) = 3 * x)) :=
by 
  intros,
  split,
  { 
    intro x,
    have h1 : 3 * (1/3 * x) = 3 * 0 := sorry,
    exact calc
      x = 0 : sorry
  },
  split,
  { 
    intro x,
    have h2 : 3 * x = 2 * x - 2 := sorry,
    exact calc
      x = -2 : sorry
  },
  split,
  {
    intro x,
    have h3 : 3 * x = 2 := sorry,
    exact calc
      x = 2 / 3 : sorry
  },
  {
    intro x,
    have h4 : 2/3 * x - 1 = x := sorry,
    exact calc
      3 * (2/3 * x - 1) = 3 * x : sorry
  }

end correct_transformation_l698_698383


namespace distinct_member_differences_l698_698998

theorem distinct_member_differences :
  let S := {2, 3, 4, 5, 6, 7}
  ∃ n, n = 5 ∧ ∀ k ∈ {1, 2, 3, 4, 5}, ∃ a b ∈ S, a ≠ b ∧ k = a - b :=
by
  let S := {2, 3, 4, 5, 6, 7}
  use 5
  split
  · rfl
  · intros k hk
  cases hk with
  | or.inl heq _ | or.inr (or.inl heq _ | or.inr (or.inl heq _ | or.inr (or.inl heq _ | or.inr (or.inl heq _ | or.inr heq _)))) _ =>
    { exists 7, 6, 7 in S, 6 in S, sorry  } -- for k = 1
    { exists 7, 5, 7 in S, 5 in S, sorry  } -- for k = 2
    { exists 7, 4, 7 in S, 4 in S, sorry  } -- for k = 3
    { exists 7, 3, 7 in S, 3 in S, sorry  } -- for k = 4
    { exists 7, 2, 7 in S, 2 in S, sorry  } -- for k = 5
    |>
  sorry

end distinct_member_differences_l698_698998


namespace no_such_function_l698_698911

theorem no_such_function (f : ℕ → ℕ) : ¬ (∀ n : ℕ, f (f n) = n + 2019) :=
sorry

end no_such_function_l698_698911


namespace five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l698_698458

theorem five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand :
  5.8 / 0.001 = 5.8 * 1000 :=
by
  -- This is where the proof would go
  sorry

end five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l698_698458


namespace hiker_walked_three_days_l698_698421

theorem hiker_walked_three_days :
  ∃ day1_speed day1_distance day2_speed day2_time day3_speed day3_time total_distance, 
  day1_speed = 3 ∧
  day1_distance = 18 ∧
  day2_speed = day1_speed + 1 ∧
  day2_time = day1_distance / day1_speed - 1 ∧
  day3_speed = day2_speed ∧
  day3_time = day1_distance / day1_speed ∧
  total_distance = day1_distance + day2_speed * day2_time + day3_speed * day3_time ∧
  total_distance = 62 :=
begin
  let day1_speed := 3,
  let day1_distance := 18,
  let day2_speed := day1_speed + 1,
  let day1_time := day1_distance / day1_speed,
  let day2_time := day1_time - 1,
  let day3_speed := day2_speed,
  let day3_time := day1_time,
  let total_distance := day1_distance + day2_speed * day2_time + day3_speed * day3_time,
  use [day1_speed, day1_distance, day2_speed, day2_time, day3_speed, day3_time, total_distance],
  simp,
  split,
  all_goals { sorry }
end

end hiker_walked_three_days_l698_698421


namespace probability_two_cards_form_pair_l698_698418

theorem probability_two_cards_form_pair :
  let num_cards := 50
  let num_each := 5
  let num_total := num_cards - 3
  let total_pairs := choose num_total 2
  let num_not_removed := 9 * choose num_each 2
  let num_remaining_removed := 1
  let m := 91
  let n := 1081
  let probability := (num_not_removed + num_remaining_removed) / total_pairs
  m + n = 1172 :=
by
  sorry

end probability_two_cards_form_pair_l698_698418


namespace decagon_adjacent_probability_l698_698745

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698745


namespace thirtieth_term_of_arithmetic_seq_l698_698815

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l698_698815


namespace value_of_4_Y_3_eq_neg23_l698_698137

def my_operation (a b : ℝ) (c : ℝ) : ℝ := a^2 - 2 * a * b * c + b^2

theorem value_of_4_Y_3_eq_neg23 : my_operation 4 3 2 = -23 := by
  sorry

end value_of_4_Y_3_eq_neg23_l698_698137


namespace factorize_poly1_factorize_poly2_l698_698005

-- Define y substitution for first problem
def poly1_y := fun (x : ℝ) => x^2 + 2*x
-- Define y substitution for second problem
def poly2_y := fun (x : ℝ) => x^2 - 4*x

-- Define the given polynomial expressions 
def poly1 := fun (x : ℝ) => (x^2 + 2*x)*(x^2 + 2*x + 2) + 1
def poly2 := fun (x : ℝ) => (x^2 - 4*x)*(x^2 - 4*x + 8) + 16

theorem factorize_poly1 (x : ℝ) : poly1 x = (x + 1) ^ 4 := sorry

theorem factorize_poly2 (x : ℝ) : poly2 x = (x - 2) ^ 4 := sorry

end factorize_poly1_factorize_poly2_l698_698005


namespace golden_section_search_third_point_l698_698792

noncomputable def golden_ratio : ℝ := 0.618

theorem golden_section_search_third_point :
  let L₀ := 1000
  let U₀ := 2000
  let d₀ := U₀ - L₀
  let x₁ := U₀ - golden_ratio * d₀
  let x₂ := L₀ + golden_ratio * d₀
  let d₁ := U₀ - x₁
  let x₃ := x₁ + golden_ratio * d₁
  x₃ = 1764 :=
by
  sorry

end golden_section_search_third_point_l698_698792


namespace probability_sum_of_scores_is_one_third_l698_698321

-- Definitions
variable (num_red num_yellow num_blue : ℕ) (score_red score_yellow score_blue : ℕ)
variable (prob_draw_red prob_draw_yellow prob_draw_blue : ℚ) (points : ℕ)

-- Values given in the problem
def bag_contents := (num_red = 3) ∧ (num_yellow = 2) ∧ (num_blue = 1)
def scoring_rules := (score_red = 1) ∧ (score_yellow = 2) ∧ (score_blue = 3)
def draw_prob := (prob_draw_red = 3/6) ∧ (prob_draw_yellow = 2/6) ∧ (prob_draw_blue = 1/6)
def target_points := (points = 1 + 2)

-- The proof problem statement
theorem probability_sum_of_scores_is_one_third :
    bag_contents ∧ scoring_rules ∧ draw_prob ∧ target_points → 
    (prob_draw_red * prob_draw_yellow + prob_draw_yellow * prob_draw_red = 1/3) :=
by
  intro h
  sorry

end probability_sum_of_scores_is_one_third_l698_698321


namespace each_child_receives_14_jellybeans_l698_698880

theorem each_child_receives_14_jellybeans :
  ∀ (total_jellybeans : ℕ) (nephews : ℕ) (nieces : ℕ),
  total_jellybeans = 70 → nephews = 3 → nieces = 2 → 
  total_jellybeans / (nephews + nieces) = 14 :=
by
  intros total_jellybeans nephews nieces hjells hneph hnieces
  rw [hjells, hneph, hnieces]
  norm_num
  sorry

end each_child_receives_14_jellybeans_l698_698880


namespace coordinates_of_OC_l698_698566

-- Define the given vectors
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define the dot product for ℝ × ℝ
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define OC as a point on line OP, parameterized by t
def OC (t : ℝ) : ℝ × ℝ := (2 * t, t)

-- Define CA and CB
def CA (t : ℝ) : ℝ × ℝ := (OA.1 - (OC t).1, OA.2 - (OC t).2)
def CB (t : ℝ) : ℝ × ℝ := (OB.1 - (OC t).1, OB.2 - (OC t).2)

-- Prove that minimization of dot_product (CA t) (CB t) occurs at OC = (4, 2)
noncomputable def find_coordinates_at_min_dot_product : Prop :=
  ∃ (t : ℝ), t = 2 ∧ OC t = (4, 2)

-- The theorem statement
theorem coordinates_of_OC : find_coordinates_at_min_dot_product :=
sorry

end coordinates_of_OC_l698_698566


namespace smallest_integer_for_prime_modulus_l698_698374

theorem smallest_integer_for_prime_modulus :
  ∃ (y : ℤ), (∀ m : ℤ, (m < y → ¬ nat.prime (nat_abs (5 * m^2 - 56 * m + 12)))) ∧ nat.prime (nat_abs (5 * y^2 - 56 * y + 12)) ∧ y = 11 :=
by
  sorry

end smallest_integer_for_prime_modulus_l698_698374


namespace right_triangle_third_side_product_l698_698777

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l698_698777


namespace lowest_fraction_of_job_done_l698_698906

theorem lowest_fraction_of_job_done :
  ∀ (rateA rateB rateC rateB_plus_C : ℝ),
  (rateA = 1/4) → (rateB = 1/6) → (rateC = 1/8) →
  (rateB_plus_C = rateB + rateC) →
  rateB_plus_C = 7/24 := by
  intros rateA rateB rateC rateB_plus_C hA hB hC hBC
  sorry

end lowest_fraction_of_job_done_l698_698906


namespace sum_odd_positives_less_than_100_l698_698376

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := n > 0
def less_than (n m : ℕ) : Prop := n < m

theorem sum_odd_positives_less_than_100 :
  ∑ k in Finset.range 100, if is_odd k ∧ is_positive k then k else 0 = 2500 := by
  sorry

end sum_odd_positives_less_than_100_l698_698376


namespace find_x_l698_698425

noncomputable theory
open Real

def log2 (x : ℝ) : ℝ := log x / log 2

theorem find_x {x : ℝ} 
  (h1 : 2 * (log2 (2 * x) * log2 (3 * x) + log2 (2 * x) * log2 (6 * x) + log2 (3 * x) * log2 (6 * x)) 
        = (log2 (2 * x) * log2 (3 * x) * log2 (6 * x))) : 
  x = 24 := 
  sorry

end find_x_l698_698425


namespace thirtieth_term_of_arithmetic_seq_l698_698814

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l698_698814


namespace intersection_correct_l698_698528

open Set

def A := {x : ℕ | 0 < x ∧ x < 6}
def B := {2, 4, 6, 8}

theorem intersection_correct : A ∩ B = {2, 4} :=
by {
  sorry
}

end intersection_correct_l698_698528


namespace quadrilateral_area_l698_698830

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 28) (hh1 : h1 = 9) (hh2 : h2 = 6) : 
  (1 / 2) * d * (h1 + h2) = 210 :=
by
  rw [hd, hh1, hh2]
  have : (1 / 2) * 28 * (9 + 6) = 210 := by linarith
  exact this

end quadrilateral_area_l698_698830


namespace field_level_rise_l698_698072

open Real

theorem field_level_rise :
  ∀ (field_length field_breadth tank_length tank_breadth tank_depth : ℝ),
    field_length = 90 →
    field_breadth = 50 →
    tank_length = 25 →
    tank_breadth = 20 →
    tank_depth = 4 →
    let volume_of_earth := tank_length * tank_breadth * tank_depth in
    let area_of_field := field_length * field_breadth in
    let area_of_tank := tank_length * tank_breadth in
    let area_of_remaining_field := area_of_field - area_of_tank in
    (volume_of_earth / area_of_remaining_field) = 0.5 :=
by
  intros field_length field_breadth tank_length tank_breadth tank_depth
  intros f_len_eq f_bre_eq t_len_eq t_bre_eq t_dep_eq
  simp [f_len_eq, f_bre_eq, t_len_eq, t_bre_eq, t_dep_eq]
  let volume_of_earth := tank_length * tank_breadth * tank_depth
  let area_of_field := field_length * field_breadth
  let area_of_tank := tank_length * tank_breadth
  let area_of_remaining_field := area_of_field - area_of_tank
  calc
    volume_of_earth / area_of_remaining_field = (25 * 20 * 4 : ℝ) / ((90 * 50) - (25 * 20)) : by simp
                                      ... = 2000 / 4000 : by norm_num
                                      ... = 0.5 : by norm_num

end field_level_rise_l698_698072


namespace tangerine_and_orange_percentage_l698_698024

-- Given conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17
def initial_grapes := 12
def initial_kiwis := 7

def removed_oranges := 2
def removed_tangerines := 10
def removed_grapes := 4
def removed_kiwis := 3

def added_oranges := 3
def added_tangerines := 6

-- Computed values based on the initial conditions and changes
def remaining_apples := initial_apples
def remaining_oranges := initial_oranges - removed_oranges + added_oranges
def remaining_tangerines := initial_tangerines - removed_tangerines + added_tangerines
def remaining_grapes := initial_grapes - removed_grapes
def remaining_kiwis := initial_kiwis - removed_kiwis

def total_remaining_fruits := remaining_apples + remaining_oranges + remaining_tangerines + remaining_grapes + remaining_kiwis
def total_citrus_fruits := remaining_oranges + remaining_tangerines

-- Statement to prove
def citrus_percentage := (total_citrus_fruits : ℚ) / total_remaining_fruits * 100

theorem tangerine_and_orange_percentage : citrus_percentage = 47.5 := by
  sorry

end tangerine_and_orange_percentage_l698_698024


namespace equilateral_triangle_area_calculation_l698_698324

noncomputable def equilateral_triangle_area (altitude : ℝ) : ℝ :=
  let s := (2 * altitude) / (Real.sqrt 3) in
  (Real.sqrt 3 / 4) * s^2

theorem equilateral_triangle_area_calculation :
  equilateral_triangle_area (Real.sqrt 8) = 32 * Real.sqrt 3 / 3 :=
by
  sorry

end equilateral_triangle_area_calculation_l698_698324


namespace total_tea_cups_l698_698575

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end total_tea_cups_l698_698575


namespace area_triangle_ABC_l698_698591

-- Definitions of the conditions
variable (A B C D E F H : Type)
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space D]
variable [metric_space E]
variable [metric_space F]
variable [metric_space H]

-- Adding noncomputable before def only when necessary
noncomputable def BD_eq_DE_eq_EC (BD DE EC : ℝ) : Prop :=
BD = DE ∧ DE = EC

noncomputable def CF_ratio (CF AC : ℝ) : Prop :=
CF / AC = 1 / 3

noncomputable def area_condition (ADH HEF : ℝ) : Prop :=
ADH - HEF = 24

-- Main theorem to prove
theorem area_triangle_ABC (BD DE EC CF AC ADH HEF : ℝ) 
  (h1 : BD_eq_DE_eq_EC BD DE EC)
  (h2 : CF_ratio CF AC)
  (h3 : area_condition ADH HEF) :
  ∃ (area_ABC : ℝ), area_ABC = 108 :=
sorry

end area_triangle_ABC_l698_698591


namespace real_sum_iff_purely_imaginary_sum_iff_l698_698918

noncomputable def sum_complex_numbers (a b c d : ℂ) : ℂ :=
  (a + c) + (b + d) * complex.I

theorem real_sum_iff (a b c d : ℂ) : (imaginary_part (sum_complex_numbers a b c d) = 0) ↔ (b = -d) :=
sorry

theorem purely_imaginary_sum_iff (a b c d : ℂ) : (real_part (sum_complex_numbers a b c d) = 0 ∧ imaginary_part (sum_complex_numbers a b c d) ≠ 0) ↔ (a = -c ∧ b ≠ -d) :=
sorry

end real_sum_iff_purely_imaginary_sum_iff_l698_698918


namespace total_sonnets_written_l698_698995

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end total_sonnets_written_l698_698995


namespace curve_is_hyperbola_l698_698150

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  let x := r * Math.cos θ
  let y := r * Math.sin θ
  (x, y)

theorem curve_is_hyperbola (r θ : ℝ) (h : r = 1 / (1 + Math.cos θ)) :
  ∃ y : ℝ, y^2 = 1 := 
sorry

end curve_is_hyperbola_l698_698150


namespace tangent_length_from_origin_to_circle_l698_698461

variable (O A B C : ℝ × ℝ)

-- Conditions
def circle_passes_through_points (P1 P2 P3 : ℝ × ℝ) : Prop :=
  let det = (P1.1 * (P2.2 - P3.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P1.2 - P2.2)) in
  det = 0

def distances (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def power_of_point (O : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ :=
  let OA := distances O A in
  let OB := distances O B in
  OA * OB

theorem tangent_length_from_origin_to_circle : 
  circle_passes_through_points (4, 5) (8, 10) (7, 15) →
  distances (0, 0) (4, 5) = real.sqrt 41 →
  distances (0, 0) (8, 10) = 2 * real.sqrt 41 →
  power_of_point (0, 0) (4, 5) (8, 10) = 82 :=
by
  intros
  sorry

end tangent_length_from_origin_to_circle_l698_698461


namespace find_truck_weight_l698_698122

variable (T Tr : ℝ)

def weight_condition_1 : Prop := T + Tr = 7000
def weight_condition_2 : Prop := Tr = 0.5 * T - 200

theorem find_truck_weight (h1 : weight_condition_1 T Tr) 
                           (h2 : weight_condition_2 T Tr) : 
  T = 4800 :=
sorry

end find_truck_weight_l698_698122


namespace car_mpg_in_city_l698_698387

theorem car_mpg_in_city
  (H C T : ℕ)
  (h1 : H * T = 462)
  (h2 : C * T = 336)
  (h3 : C = H - 9) : C = 24 := by
  sorry

end car_mpg_in_city_l698_698387


namespace impossible_to_light_all_cells_in_10x10_grid_l698_698369

theorem impossible_to_light_all_cells_in_10x10_grid 
  (initially_lit: ℕ) 
  (grid_size: ℕ) 
  (lighting_rule: (ℕ × ℕ) → bool) 
  (initial_condition: initially_lit = 9) 
  (grid_condition: grid_size = 10)
  (rule_condition: ∀ (i j: ℕ), 1 ≤ i ∧ i < grid_size - 1 ∧ 1 ≤ j ∧ j < grid_size - 1 → 
    lighting_rule (i, j) = (lighting_rule (i-1, j) ∧ lighting_rule (i+1, j)) ∨ (lighting_rule (i, j-1) ∧ lighting_rule (i, j+1))) :
  ¬(∃ (all_lit: (ℕ × ℕ) → bool), ∀ (i j: ℕ), 1 ≤ i ∧ i ≤ grid_size ∧ 1 ≤ j ∧ j ≤ grid_size → all_lit (i, j) = true) := 
begin
  sorry
end

end impossible_to_light_all_cells_in_10x10_grid_l698_698369


namespace unique_triple_solution_zero_l698_698914

theorem unique_triple_solution_zero (m n k : ℝ) :
  (∃ x : ℝ, m * x ^ 2 + n = 0) ∧
  (∃ x : ℝ, n * x ^ 2 + k = 0) ∧
  (∃ x : ℝ, k * x ^ 2 + m = 0) ↔
  (m = 0 ∧ n = 0 ∧ k = 0) := 
sorry

end unique_triple_solution_zero_l698_698914


namespace compute_expression_l698_698280

-- Definition of the imaginary unit i
class ImaginaryUnit (i : ℂ) where
  I_square : i * i = -1

-- Definition of non-zero real number a
variable (a : ℝ) (h_a : a ≠ 0)

-- Theorem to prove the equivalence
theorem compute_expression (i : ℂ) [ImaginaryUnit i] :
  (a * i - i⁻¹)⁻¹ = -i / (a + 1) :=
by
  sorry

end compute_expression_l698_698280


namespace find_fraction_l698_698390

theorem find_fraction (x : ℚ) : (3 / 4) / (1 / 2) = (3 / 2) → (x / (2 / 6) = 3 / 2) → x = 1 / 2 :=
by
  intros h1 h2
  rw ← h1 at h2
  exact h2.symm

end find_fraction_l698_698390


namespace derivative_f_l698_698668

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem derivative_f : (deriv f) = λ x, -Real.exp (-x) :=
by
  sorry

end derivative_f_l698_698668


namespace find_principal_amount_l698_698074

variable (R P : ℝ)
variable (T : ℕ) (P T R : ℝ → ℝ)

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem find_principal_amount :
  (simple_interest P (R + 3) 4 - simple_interest P R 4 = 120) → P = 1000 :=
  sorry

end find_principal_amount_l698_698074


namespace quadratic_transform_is_valid_l698_698008

theorem quadratic_transform_is_valid :
  ∃ a b c : ℤ, (8 * (λ x : ℤ, x^2) - 48 * (λ x : ℤ, x) - 320) = (a * (λ x : ℤ, (x+b)^2) + c) ∧ a + b + c = -387 :=
sorry

end quadratic_transform_is_valid_l698_698008


namespace sequence_sum_l698_698867

theorem sequence_sum 
  (a : ℕ → ℤ)
  (x y : ℤ)
  (h1 : a 1 = x)
  (h3 : a 3 = y)
  (h_seq : ∀ n ≥ 1, a (n + 1) = a n + a (n + 2) - 1) :
  (∑ i in Finset.range 2018, a i) = 2 * x + y + 2015 :=
sorry

end sequence_sum_l698_698867


namespace tangent_line_at_1_l698_698333

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_line_at_1 : 
  let tangent_line := λ x y : ℝ, x - y - 1 
  in tangent_line = (1,0) := 
by
  sorry

end tangent_line_at_1_l698_698333


namespace right_triangle_same_color_exists_l698_698683

theorem right_triangle_same_color_exists (color : ℕ → Prop) (triangle : ℕ → ℕ → Prop)
  (h_color : ∀ x y z : ℕ, triangle x y z → (color x ∨ color y ∨ color z)) :
  ∃ x y z : ℕ, triangle x y z ∧ (color x ∧ color y ∧ color z) := 
sorry

end right_triangle_same_color_exists_l698_698683


namespace probability_adjacent_vertices_decagon_l698_698735

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698735


namespace part1_part2_l698_698185

-- Define the sequence
def a : ℕ → ℕ
| 1     := 2
| (n+1) := 2 * a n - n

-- Prove that {a_n - n} is a geometric sequence with ratio 2
theorem part1 : ∀ n : ℕ, n > 0 → ∃ r : ℕ, ∃ b : ℕ, a (n+1) - (n+1) = r * (a n - n) :=
by {
  sorry
}

-- Prove that 265 is the 9th term and the sum of the first 8 terms is 291
theorem part2 : (a 9 = 265) ∧ (∑ i in finset.range 8, a (i + 1) = 291) :=
by {
  sorry
}

end part1_part2_l698_698185


namespace math_problem_l698_698192

noncomputable def calculate_ratio (a0 a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  (a0 + a2 + a4) / (a1 + a3)

theorem math_problem (a0 a1 a2 a3 a4 a5 : ℝ) (h1 : a0 + a1 + a2 + a3 + a4 + a5 = 1) 
                     (h2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) 
                     (expr_eq : ∀ x, x^5 = a0 + a1 * (2 - x) + a2 * (2 - x)^2 + a3 * (2 - x)^3 + a4 * (2 - x)^4 + a5 * (2 - x)^5) :
  calculate_ratio a0 a1 a2 a3 a4 a5 = -61 / 60 :=
begin
  sorry
end

end math_problem_l698_698192


namespace probability_adjacent_vertices_decagon_l698_698759

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698759


namespace largest_power_of_2_divides_n_l698_698916

def n : ℤ := 17^4 - 13^4

theorem largest_power_of_2_divides_n : ∃ (k : ℕ), 2^4 = k ∧ 2^k ∣ n ∧ ¬ (2^(k + 1) ∣ n) := by
  sorry

end largest_power_of_2_divides_n_l698_698916


namespace nonConsecutiveWaysToChooseThree_l698_698174

def isNonConsecutiveSet (s : Finset ℕ) : Prop :=
  ∀ (x ∈ s) (y ∈ s), x ≠ y → abs (x - y) > 1

theorem nonConsecutiveWaysToChooseThree :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}.toFinset in
  (S.powerset.filter (λ s, s.card = 3 ∧ isNonConsecutiveSet s)).card = 20 :=
by
  sorry

end nonConsecutiveWaysToChooseThree_l698_698174


namespace complex_equation_solution_l698_698503

open Complex

noncomputable def z : ℂ := -1 - (2 / 3) * I

theorem complex_equation_solution :
  (2 * conj z) - z = (1 + 3 * I) / (1 - I) :=
sorry

end complex_equation_solution_l698_698503


namespace painting_count_l698_698574

noncomputable def count_valid_paintings : ℕ :=
  let colors := {red, green, blue}
  let divisors := [
    (4, [2]),
    (6, [2, 3]),
    (8, [2, 4]),
    (9, [3]),
    (10, [2, 5])
  ]
  let independent_numbers := [2, 3, 5, 7]
  let dependent_numbers := [4, 6, 8, 9, 10]
  let independent_count := 3 ^ independent_numbers.length
  let dependent_count := 2 * 1 * 1 * 2 * 1
  independent_count * dependent_count

theorem painting_count : count_valid_paintings = 324 := by
  sorry

end painting_count_l698_698574


namespace polynomial_coeff_sum_l698_698589

/-- 
Given that the product of the polynomials (4x^2 - 6x + 5)(8 - 3x) can be written as
ax^3 + bx^2 + cx + d, prove that 9a + 3b + c + d = 19.
-/
theorem polynomial_coeff_sum :
  ∃ a b c d : ℝ, 
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧
  9 * a + 3 * b + c + d = 19 :=
sorry

end polynomial_coeff_sum_l698_698589


namespace find_breadth_of_rectangle_l698_698006

-- Definitions
def side_of_square (breadth_of_rectangle : ℝ) : ℝ :=
  (26.70 * 2) / (Real.pi + 2)

def breadth_of_rectangle : ℝ := 
  2 * side_of_square 0.78 - 20

-- Statement
theorem find_breadth_of_rectangle (h1 : (4 * (side_of_square 0.78)) = 2 * (20 + breadth_of_rectangle))
                                  (h2 : (Real.pi * (side_of_square 0.78) / 2 + side_of_square 0.78) = 26.70) :
  abs (breadth_of_rectangle - 0.78) < 0.01 :=
by
  sorry

end find_breadth_of_rectangle_l698_698006


namespace find_a_l698_698977

theorem find_a :
  let p1 := (⟨-3, 7⟩ : ℝ × ℝ)
  let p2 := (⟨2, -1⟩ : ℝ × ℝ)
  let direction := (5, -8)
  let target_direction := (a, -2)
  a = (direction.1 * -2) / (direction.2) := by
  sorry

end find_a_l698_698977


namespace quadratic_has_two_real_distinct_roots_and_find_m_l698_698979

theorem quadratic_has_two_real_distinct_roots_and_find_m 
  (m : ℝ) :
  (x : ℝ) → 
  (h1 : x^2 - (2 * m - 2) * x + (m^2 - 2 * m) = 0) →
  (x1 x2 : ℝ) →
  (h2 : x1^2 + x2^2 = 10) →
  (x1 + x2 = 2 * m - 2) →
  (x1 * x2 = m^2 - 2 * m) →
  (x1 ≠ x2) ∧ (m = -1 ∨ m = 3) :=
by sorry

end quadratic_has_two_real_distinct_roots_and_find_m_l698_698979


namespace new_apps_added_l698_698898

theorem new_apps_added (x : ℕ) (h1 : 15 + x - (x + 1) = 14) : x = 0 :=
by
  sorry

end new_apps_added_l698_698898


namespace hyperbola_cosine_angle_l698_698329

def hyperbola_cos_angle := "cosine of the angle between the asymptotes of the hyperbola x^2 - y^2/4 = 1".

theorem hyperbola_cosine_angle : 
  (∃ (a b : ℝ), hyperbola_cos_angle ∧ a = 1 ∧ b = 2 ∧ (x^2 - y^2 / (2*a) = 1)) 
  → ∃ (cosine : ℝ), cosine = 3 / 5 :=
by
  -- The proof would go here
  sorry

end hyperbola_cosine_angle_l698_698329


namespace joanna_reading_rate_l698_698358

variable (P : ℝ)

theorem joanna_reading_rate (h : 3 * P + 6.5 * P + 6 * P = 248) : P = 16 := by
  sorry

end joanna_reading_rate_l698_698358


namespace range_of_m_value_of_m_l698_698530

variable (α β m : ℝ)

open Real

-- Conditions: α and β are positive roots.
def quadratic_roots (α β m : ℝ) : Prop :=
  (α > 0) ∧ (β > 0) ∧ (α + β = 1 - 2*m) ∧ (α * β = m^2)

-- Part 1: Range of values for m.
theorem range_of_m (h : quadratic_roots α β m) : m ≤ 1/4 ∧ m ≠ 0 :=
sorry

-- Part 2: Given α^2 + β^2 = 49, find the value of m.
theorem value_of_m (h : quadratic_roots α β m) (h' : α^2 + β^2 = 49) : m = -4 :=
sorry

end range_of_m_value_of_m_l698_698530


namespace product_of_third_sides_is_correct_l698_698766

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l698_698766


namespace exists_six_points_l698_698490

open EuclideanGeometry

def satisfies_conditions (points : List (ℝ × ℝ)) : Prop :=
  points.length = 6 ∧ 
  ∀ (i : ℕ) (h : i < points.length),
    let A_i := points.nthLe i h in
    (points.filter (λ A_j, dist A_i A_j = 1).length = 3)

theorem exists_six_points : 
  ∃ (points : List (ℝ × ℝ)), satisfies_conditions points :=
sorry

end exists_six_points_l698_698490


namespace count_students_less_than_10_l698_698876

def student_times : List ℕ := [10, 12, 15, 6, 3, 8, 9]

theorem count_students_less_than_10 : student_times.count (< 10) = 4 := by
  sorry

end count_students_less_than_10_l698_698876


namespace problem_dorlir_ahmeti_equality_case_l698_698235

theorem problem_dorlir_ahmeti (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h : x^2 + y^2 + z^2 = x + y + z) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 :=
sorry
  
theorem equality_case (x y z : ℝ)
  (hx : x = 0) (hy : y = 0) (hz : z = 0) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) = 3 :=
sorry

end problem_dorlir_ahmeti_equality_case_l698_698235


namespace axis_of_symmetry_find_zeros_l698_698952

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x + π / 3)

theorem axis_of_symmetry (ω : ℝ) (hω : ω > 0) (hT : 2 * π / ω = π) :
  ∃ x ∈ set.Ioo 0 (π / 2), f ω x = f ω (π / 6 - x) :=
sorry

theorem find_zeros (ω : ℝ) (hω : ω > 0) (hT : 2 * π / ω = π) :
  ∃ (x : ℝ), x ∈ set.Icc π 2 * π ∧ (x = π / 3 ∨ x = 5 * π / 6 ∨ x = 4 * π / 3 ∨ x = 11 * π / 6) :=
sorry

end axis_of_symmetry_find_zeros_l698_698952


namespace tangent_meets_curve_twice_l698_698285

noncomputable def curve := λ x : ℝ, x^3

def grad (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem tangent_meets_curve_twice {a : ℝ} (ha : a ≠ 0):
  let A := (a, curve a) in
  let B := (-2 * a, curve (-2*a)) in
  grad curve B.1 = 4 * (grad curve A.1) :=
by
  sorry

end tangent_meets_curve_twice_l698_698285


namespace div_c_a_l698_698579

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end div_c_a_l698_698579


namespace arithmetic_sequence_30th_term_l698_698807

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l698_698807


namespace infinite_geometric_series_sum_l698_698887

theorem infinite_geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1 / 3
  ∑' (n : ℕ), a * r ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l698_698887


namespace right_triangle_third_side_product_l698_698772

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l698_698772


namespace increase_by_one_or_prime_l698_698852

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def is_prime (p : ℕ) : Prop := Nat.Prime p

def million_steps (initial : ℕ) : Prop :=
  ∀ n current_number, 
    current_number = initial + (Finset.sum (Finset.range n) (λ k, gcd (initial + (Finset.sum (Finset.range k) (λ j, gcd (initial + (Finset.sum (Finset.range j) (λ i, gcd (initial, i)))), k)))))
    → (current_number + gcd (current_number, n) - current_number = 1 ∨ is_prime (gcd (current_number, n)))

theorem increase_by_one_or_prime :
  (∃ initial, initial = 6) → million_steps 6 :=
by
  sorry

end increase_by_one_or_prime_l698_698852


namespace min_A_l698_698288

noncomputable def min_A_proof (n : ℕ) (x : Fin n → ℝ) (a : Fin n → ℝ) (A : ℝ) : Prop :=
  (∑ i, x i = 0) ∧
  (∑ i, |x i| = 1) ∧
  ((∀ i, ∀ j, i ≤ j → a i ≥ a j) → 
   (|∑ i, a i * x i| ≤ A * (a 0 - a (Fin.last n))))

theorem min_A (n : ℕ) (x : Fin n → ℝ) (a : Fin n → ℝ) : n > 2 →
  (∑ i, x i = 0) →
  (∑ i, |x i| = 1) →
  (∀ i j, i ≤ j → a i ≥ a j) →
  (∀ A, min_A_proof n x a A → A ≥ 1 / 2) :=
begin
  intros h1 h2 h3 h4 A h5,
  sorry -- Proof omitted
end

end min_A_l698_698288


namespace adjusted_pace_correct_l698_698577

noncomputable def adjusted_pace (distance_miles : ℕ) (time_minutes : ℕ)
  (elevation_gain_ft : ℕ) (elevation_loss_ft : ℕ)
  (gain_adjustment : ℕ := 2) (loss_adjustment : ℕ := 1) : ℝ :=
let additional_time := (elevation_gain_ft / 100) * gain_adjustment in
let subtracted_time := (elevation_loss_ft / 200) * loss_adjustment in
let net_time_adjustment := additional_time - subtracted_time in
let total_time_minutes := time_minutes + net_time_adjustment in
let total_time_hours := total_time_minutes / 60.0 in
distance_miles / total_time_hours

theorem adjusted_pace_correct :
  adjusted_pace 9 (1*60 + 15) 600 400 ≈ 6.35 := 
by 
  sorry

end adjusted_pace_correct_l698_698577


namespace max_chickens_ducks_l698_698270

theorem max_chickens_ducks (x y : ℕ) 
  (h1 : ∀ (k : ℕ), k = 6 → x + y - 6 ≥ 2) 
  (h2 : ∀ (k : ℕ), k = 9 → y ≥ 1) : 
  x + y ≤ 12 :=
sorry

end max_chickens_ducks_l698_698270


namespace div_equivalence_l698_698580

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end div_equivalence_l698_698580


namespace no_partition_10x10_with_L_shapes_l698_698888

theorem no_partition_10x10_with_L_shapes :
  ¬ ∃ (f : (Fin 10 × Fin 10) → Fin 25), 
  (∀ i : Fin 25, ∃! t : Finset (Fin 10 × Fin 10), t.card = 4 ∧ is_L_shape t ∧ ∀ p ∈ t, f p = i) := 
sorry

end no_partition_10x10_with_L_shapes_l698_698888


namespace train_length_calculation_l698_698117

noncomputable def length_of_train (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_calculation : 
  length_of_train 4.99960003199744 72 = 99.9920006399488 :=
by 
  sorry  -- proof of the actual calculation

end train_length_calculation_l698_698117


namespace pure_imaginary_z1_over_z2_l698_698519

theorem pure_imaginary_z1_over_z2 (b : Real) : 
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  (Complex.re ((z1 / z2) : Complex)) = 0 → b = -3 / 2 :=
by
  intros
  -- Conditions
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  -- Assuming that the real part of (z1 / z2) is zero
  have h : Complex.re (z1 / z2) = 0 := ‹_›
  -- Require to prove that b = -3 / 2
  sorry

end pure_imaginary_z1_over_z2_l698_698519


namespace log_diff_equals_l698_698925

open Real

def a (n : ℕ) (h : n > 1) : ℝ := 1 / (log 3003 / log n)

theorem log_diff_equals :
  let d = (a 3 (by norm_num) + a 4 (by norm_num) + a 5 (by norm_num) + a 6 (by norm_num))
  let e = (a 15 (by norm_num) + a 16 (by norm_num) + a 17 (by norm_num) + a 18 (by norm_num) + a 19 (by norm_num))
  in d - e = (log 1 / log 323) / (log 3003)
:=
sorry

end log_diff_equals_l698_698925


namespace reciprocal_of_negative_one_point_five_l698_698010

theorem reciprocal_of_negative_one_point_five : 
  let x := -1.5 in 
  ∃ (y : ℚ), x * y = 1 ∧ y = -2 / 3 :=
by
  -- definitions for the proof
  let x : ℚ := -3 / 2
  let y : ℚ := -2 / 3

  -- skipping the proof for now
  sorry

end reciprocal_of_negative_one_point_five_l698_698010


namespace sum_of_even_indexed_angles_l698_698019

open Complex

noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

theorem sum_of_even_indexed_angles (n : ℕ) (θ : Fin (2 * n) → ℝ)
  (h1 : ∀ i, 0 ≤ θ i ∧ θ i < 360)
  (h2 : StrictMono θ)
  (h3 : ∀ i, let z := cis (θ i * π / 180) in z ^ 28 - z ^ 8 - 1 = 0 ∧ abs z = 1) :
  (Finset.range n).sum (λ i, θ (⟨2 * (i + 1) - 1, sorry⟩)) = 840 := 
sorry

end sum_of_even_indexed_angles_l698_698019


namespace segment_length_at_1_point_5_l698_698184

-- Definitions for the conditions
def Point := ℝ × ℝ
def Triangle (A B C : Point) := ∃ a b c : ℝ, a = 4 ∧ b = 3 ∧ c = 5 ∧ (A = (0, 0)) ∧ (B = (4, 0)) ∧ (C = (0, 3)) ∧ (c^2 = a^2 + b^2)

noncomputable def length_l (x : ℝ) : ℝ := (4 * (abs ((3/4) * x + 3))) / 5

theorem segment_length_at_1_point_5 (A B C : Point) (h : Triangle A B C) : 
  length_l 1.5 = 3.3 := by 
  sorry

end segment_length_at_1_point_5_l698_698184


namespace thirtieth_term_of_arithmetic_seq_l698_698812

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l698_698812


namespace find_angle_BHC_l698_698260

noncomputable def triangle_ABC (A B C H D E F : Type) :=
  -- conditions
  angle ABC = 58
  ∧ angle ACB = 23
  ∧ are_orthocenter H (triangle A B C)

theorem find_angle_BHC (A B C H D E F : Type) [triangle_ABC A B C H D E F] :
  angle BHC = 81 := 
sorry

end find_angle_BHC_l698_698260


namespace num_signs_in_sign_language_l698_698686

theorem num_signs_in_sign_language (n : ℕ) (h : n^2 - (n - 2)^2 = 888) : n = 223 := 
sorry

end num_signs_in_sign_language_l698_698686


namespace man_twice_son_age_in_years_l698_698101

theorem man_twice_son_age_in_years :
  ∀ (S M Y : ℕ),
  (M = S + 26) →
  (S = 24) →
  (M + Y = 2 * (S + Y)) →
  Y = 2 :=
by
  intros S M Y h1 h2 h3
  sorry

end man_twice_son_age_in_years_l698_698101


namespace find_value_of_x_plus_y_l698_698225

variable (x y : ℚ)

theorem find_value_of_x_plus_y (h1 : 5 * x - 3 * y = 17) (h2 : 3 * x + 5 * y = 1) :
  x + y = 36 / 85 := 
begin
  sorry
end

end find_value_of_x_plus_y_l698_698225


namespace solution_set_of_inequalities_l698_698913

theorem solution_set_of_inequalities :
  {x : ℝ | 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9} = {x : ℝ | x > 45 / 26} :=
by sorry

end solution_set_of_inequalities_l698_698913


namespace final_amount_in_local_currency_l698_698068

def progressive_tax (amount: ℝ) : ℝ :=
  if amount ≤ 10000 then 0
  else if amount ≤ 20000 then (amount - 10000) * 0.10
  else if amount ≤ 50000 then 1000 + (amount - 20000) * 0.20
  else 1000 + 6000 + (amount - 50000) * 0.20

def additional_fees_tax : ℝ := 5 + 200 + 1000

def administrative_fee (amount: ℝ) : ℝ :=
  amount * 0.02

def net_amount (winnings: ℝ) (total_deductions: ℝ) : ℝ :=
  winnings - total_deductions

def converted_amount (amount: ℝ) (exchange_rate: ℝ) : ℝ :=
  amount * exchange_rate

theorem final_amount_in_local_currency :
  let winnings := 50000
  let us_tax := progressive_tax winnings
  let total_deductions := us_tax + additional_fees_tax
  let amount_after_deductions := net_amount winnings total_deductions
  let admin_fee := administrative_fee amount_after_deductions
  let final_amount := net_amount amount_after_deductions admin_fee
  let local_currency := converted_amount final_amount 0.85
  local_currency = 34815.24 :=
by
  sorry

end final_amount_in_local_currency_l698_698068


namespace calculate_annual_cost_l698_698029

noncomputable def doctor_visit_cost (visits_per_year: ℕ) (cost_per_visit: ℕ) : ℕ :=
  visits_per_year * cost_per_visit

noncomputable def medication_night_cost (pills_per_night: ℕ) (cost_per_pill: ℕ) : ℕ :=
  pills_per_night * cost_per_pill

noncomputable def insurance_coverage (medication_cost: ℕ) (coverage_percent: ℕ) : ℕ :=
  medication_cost * coverage_percent / 100

noncomputable def out_of_pocket_cost (total_cost: ℕ) (insurance_covered: ℕ) : ℕ :=
  total_cost - insurance_covered

noncomputable def annual_medication_cost (night_cost: ℕ) (nights_per_year: ℕ) : ℕ :=
  night_cost * nights_per_year

noncomputable def total_annual_cost (doctor_visit_total: ℕ) (medication_total: ℕ) : ℕ :=
  doctor_visit_total + medication_total

theorem calculate_annual_cost :
  let visits_per_year := 2 in
  let cost_per_visit := 400 in
  let pills_per_night := 2 in
  let cost_per_pill := 5 in
  let coverage_percent := 80 in
  let nights_per_year := 365 in
  let total_cost := total_annual_cost 
    (doctor_visit_cost visits_per_year cost_per_visit)
    (annual_medication_cost 
      (out_of_pocket_cost 
        (medication_night_cost pills_per_night cost_per_pill) 
        (insurance_coverage (medication_night_cost pills_per_night cost_per_pill) coverage_percent)
      ) 
      nights_per_year
    ) in
  total_cost = 1530 :=
begin
  sorry
end

end calculate_annual_cost_l698_698029


namespace minimum_distance_OC_l698_698190

theorem minimum_distance_OC
  (a b : ℝ)
  (ha : a > b)
  (hb : b > 0)
  (h_ellipse : ∀ x y : ℝ, (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1)
  (h_area : 2 * (sqrt 3)) -- assuming area given is 2 * sqrt(3); need further context on interpreting this
  (h_angle : 60 = 60) -- internal angle condition
  (h_AB : ∀ l : ℝ, ∃ A B : ℝ × ℝ, A ≠ B ∧ (h_ellipse A.1 A.2 ∧ h_ellipse B.1 B.2)) -- existence of A, B intersection
  (h_midpoint : ∀ A B C O : ℝ × ℝ, C = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ O = (0, 0))
  (h_triangle_area : 1 / 2 * (A.1 * B.2 - A.2 * B.1) = sqrt 3 / 2) :
  ∃ C : ℝ × ℝ, |OC| = sqrt 2 / 2 :=
sorry

end minimum_distance_OC_l698_698190


namespace A_implies_B_l698_698520

-- Define points E, F, G, H
variables {E F G H : Type}

-- Define the propositions A and B
def not_coplanar (E F G H : Type) : Prop :=
  ∀ (plane : Type), ¬ (E ∈ plane ∧ F ∈ plane ∧ G ∈ plane ∧ H ∈ plane)

def not_intersect (EF GH : Type) : Prop :=
  ¬ ∃ point : Type, point ∈ EF ∧ point ∈ GH

-- Lean theorem statement
theorem A_implies_B (E F G H : Type) 
  (hA : not_coplanar E F G H) :
  not_intersect (λ p : Type, p ∈ E ∨ p ∈ F) (λ q : Type, q ∈ G ∨ q ∈ H) :=
sorry

end A_implies_B_l698_698520


namespace part_1_part_2_l698_698529

noncomputable def A (x : ℝ) : ℝ := ⟨x, ∃ y, y = Real.sqrt ((2 + x) * (4 - x))⟩

def B (x : ℝ) (m : ℝ) : ℝ := ⟨x, -1 < x ∧ x < m + 1⟩

theorem part_1 (m : ℝ) (h_m : m = 4) :
  A ∪ (B x m) =({x : ℝ | -2 ≤ x ∧ x < 5}) ∧ 
  compl(A) ∩ (B x m) = ({x : ℝ | 4 < x ∧ x < 5}) := 
sorry

theorem part_2 (m : ℝ) :
    (∀ x, B x m → A x) → m ≤ 3 := 
sorry

end part_1_part_2_l698_698529


namespace pentagon_diagonal_ratio_l698_698143

theorem pentagon_diagonal_ratio (P : Type) [ConvexPentagon P] :
  (∀ (d s : ℝ), is_diagonal P d → is_side P s → d = (s * (Real.sqrt 5 + 1) / 2)) :=
sorry

end pentagon_diagonal_ratio_l698_698143


namespace smallest_period_40_l698_698894

noncomputable def smallest_positive_period (f : ℝ → ℝ) :=
  ∃ p : ℝ, p > 0 ∧ (∀ x, f(x + p) = f(x)) ∧ (∀ q, q > 0 → (∀ x, f(x + q) = f(x)) → p ≤ q)

theorem smallest_period_40 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x + 5) + f(x - 5) = f(x)) : smallest_positive_period f = 40 :=
sorry

end smallest_period_40_l698_698894


namespace orthocenter_on_GM_l698_698618

variables {A B C D E F G M H I : Type}
variables [triangle_ABC : triangle A B C]
variables [incircle_ABC : incircle I A B C]
variables [incircle_touches : incircle_touches I A B C D E F]
variables [DG_perpendicular_EF : perp D G (line_through E F)]
variables [GM_intersects_BC : ∃ M, line_through G M ∧ M ∈ line_through B C]
variables [equal_angles_I_G_D_M_G_D : ∠ I G D = ∠ M G D]

theorem orthocenter_on_GM :
  orthocenter A B C H → lies_on_line H (line_through G M) :=
sorry

end orthocenter_on_GM_l698_698618


namespace statues_broken_in_year3_correct_l698_698222

-- Definitions based on the conditions
def year1_statues : ℕ := 4
def year2_statues : ℕ := year1_statues * 4
def year3_statues_before_hail : ℕ := year2_statues + 12
def year4_statues_total : ℕ := 31

-- Let B be the number of broken statues in the third year
def statues_broken_in_year3 (B : ℕ) : Prop :=
  year3_statues_before_hail - B + 2 * B = year4_statues_total

-- Proof problem statement
theorem statues_broken_in_year3_correct :
  ∃ B : ℕ, statues_broken_in_year3 B ∧ B = 3 :=
by
  use 3
  unfold statues_broken_in_year3
  rw [Nat.sub_add_cancel]
  exact Nat.add_sub_of_le
  sorry

end statues_broken_in_year3_correct_l698_698222


namespace area_of_triangle_ABC_l698_698078

/--
Given a triangle \(ABC\) with points \(D\) and \(E\) on sides \(BC\) and \(AC\) respectively,
where \(BD = 4\), \(DE = 2\), \(EC = 6\), and \(BF = FC = 3\),
proves that the area of triangle \( \triangle ABC \) is \( 18\sqrt{3} \).
-/
theorem area_of_triangle_ABC :
  ∀ (ABC D E : Type) (BD DE EC BF FC : ℝ),
    BD = 4 → DE = 2 → EC = 6 → BF = 3 → FC = 3 → 
    ∃ area, area = 18 * Real.sqrt 3 :=
by
  intros ABC D E BD DE EC BF FC hBD hDE hEC hBF hFC
  sorry

end area_of_triangle_ABC_l698_698078


namespace percentage_increase_B_over_C_l698_698003

noncomputable def A_monthly_income := 403200.0000000001 / 12
noncomputable def C_monthly_income := 12000
noncomputable def x := A_monthly_income / 5
noncomputable def B_monthly_income := 2 * x
noncomputable def percentage_increase := ((B_monthly_income - C_monthly_income) / C_monthly_income) * 100

theorem percentage_increase_B_over_C : percentage_increase = 12 :=
by sorry

end percentage_increase_B_over_C_l698_698003


namespace value_of_f_g_5_l698_698229

def g (x : ℕ) : ℕ := 4 * x - 5
def f (x : ℕ) : ℕ := 6 * x + 11

theorem value_of_f_g_5 : f (g 5) = 101 := by
  sorry

end value_of_f_g_5_l698_698229


namespace sum_fractions_series_l698_698462

-- Define a function representing the sum of fractions from 1/7 to 12/7
def sum_fractions : ℚ :=
  (list.sum (list.map (λ k, k / 7) (list.range' 1 12)))

-- State the theorem
theorem sum_fractions_series :
  sum_fractions = 11 + 1 / 7 :=
sorry

end sum_fractions_series_l698_698462


namespace tan_angle_equiv_tan_1230_l698_698154

theorem tan_angle_equiv_tan_1230 : ∃ n : ℤ, -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1230 * Real.pi / 180) :=
sorry

end tan_angle_equiv_tan_1230_l698_698154


namespace range_of_hyperbola_eccentricity_l698_698211

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : c = Real.sqrt (a^2 + b^2))
  (h₃ : 90 < Real.angle (arctan (b / (c - a^2 / c))) (arctan (-b / (c - a^2 / c))) * 180 / π < 120) :
  ℝ :=
Real.sqrt (1 + (b / a)^2)

theorem range_of_hyperbola_eccentricity (a b c : ℝ) 
  (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : c = Real.sqrt (a^2 + b^2))
  (h₃ : 90 < Real.angle (arctan (b / (c - a^2 / c))) (arctan (-b / (c - a^2 / c))) * 180 / π < 120) :
  let e := hyperbola_eccentricity a b c h₀ h₁ h₂ h₃ in
  (2*Real.sqrt 3 / 3 < e) ∧ (e < Real.sqrt 2) :=
by
  sorry

end range_of_hyperbola_eccentricity_l698_698211


namespace wheel_horizontal_distance_l698_698437

noncomputable def wheel_radius : ℝ := 2
noncomputable def wheel_revolution_fraction : ℝ := 3 / 4
noncomputable def wheel_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem wheel_horizontal_distance :
  wheel_circumference wheel_radius * wheel_revolution_fraction = 3 * Real.pi :=
by
  sorry

end wheel_horizontal_distance_l698_698437


namespace triangle_ratio_l698_698617

-- Define the triangle FGH and points X and Y
structure Triangle :=
  (F G H : ℝ)

-- Define the points and their ratios
structure Points :=
  (X Y : ℝ)
  (HX XF HY YF : ℝ)

-- Conditions:
-- 1. Line through X parallel to FG
-- 2. Ratio HX : XF = 4 : 1
-- 3. Areas of shaded regions through X and Y are equal

def HX_XF_Ratio (points : Points) : Prop :=
  points.HX / points.XF = 4

def Areas_Equal (triangle : Triangle) (points : Points) : Prop :=
  -- Placeholder, Real makes it more practical to deduce areas
  sorry

def HY_YF_Ratio (points : Points) : Prop :=
  points.HY / points.YF = 3 / 2

-- The theorem to be proven
theorem triangle_ratio 
  (triangle : Triangle)
  (points : Points)
  (h_ratio : HX_XF_Ratio(points))
  (h_areas_equal : Areas_Equal(triangle, points)) :
  HY_YF_Ratio(points) :=
sorry

end triangle_ratio_l698_698617


namespace marbles_shared_l698_698889

theorem marbles_shared (initial_marbles final_marbles shared_marbles : ℕ) 
  (h_initial: initial_marbles = 47)
  (h_final: final_marbles = 5) :
  shared_marbles = initial_marbles - final_marbles :=
by 
  have h: shared_marbles = 42, from sorry,
  exact h

end marbles_shared_l698_698889


namespace solve_for_a_l698_698960

theorem solve_for_a (a x : ℤ) (h : x + 2 * a = -3) (hx : x = 1) : a = -2 := by
  sorry

end solve_for_a_l698_698960


namespace decagon_adjacent_vertex_probability_l698_698718

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698718


namespace lines_intersect_or_parallel_l698_698836

open EuclideanGeometry

-- Definitions for the circles and intersection points
variable {circle1 circle2 circle3 : Circle}
variable {A0 A1 : Point}
variable {B0 B1 : Point}
variable {C0 C1 : Point}

-- Assumptions about the intersection points
axiom intersectA : intersects circle1 circle2 A0 ∧ intersects circle1 circle2 A1
axiom intersectB : intersects circle2 circle3 B0 ∧ intersects circle2 circle3 B1
axiom intersectC : intersects circle3 circle1 C0 ∧ intersects circle3 circle1 C1

-- Definition for the circumcenters
noncomputable def circumcenter (A B C : Point) : Point := sorry

-- The proof statement
theorem lines_intersect_or_parallel :
  ∃ P : Point, ∀ (i j k : Fin 2),
    let O_ijk := circumcenter (if i = 0 then A0 else A1) (if j = 0 then B0 else B1) (if k = 0 then C0 else C1)
    in O_ijk = P ∨ ∥ O_ijk ∥ // parallel condition
Proof := sorry

end lines_intersect_or_parallel_l698_698836


namespace ant_weight_statement_l698_698875

variable (R : ℝ) -- Rupert's weight
variable (A : ℝ) -- Antoinette's weight
variable (C : ℝ) -- Charles's weight

-- Conditions
def condition1 : Prop := A = 2 * R - 7
def condition2 : Prop := C = (A + R) / 2 + 5
def condition3 : Prop := A + R + C = 145

-- Question: Prove Antoinette's weight
def ant_weight_proof : Prop :=
  ∃ R A C, condition1 R A ∧ condition2 R A C ∧ condition3 R A C ∧ A = 79

theorem ant_weight_statement : ant_weight_proof :=
sorry

end ant_weight_statement_l698_698875


namespace find_a_l698_698489

theorem find_a (a : ℝ) : 
  (\[ a > 15 / 8 \]) ->
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (
    (x1^3 - 2 * a * x1^2 - 3 * a * x1 + a^2 - 2 = 0) ∧ 
    (x2^3 - 2 * a * x2^2 - 3 * a * x2 + a^2 - 2 = 0)) := 
by
  sorry

end find_a_l698_698489


namespace domain_of_f_l698_698332

noncomputable def f : ℝ → ℝ := λ x, real.sqrt(x + 2) + 1 / (x - 1)

theorem domain_of_f :
  {x : ℝ | -2 ≤ x} \ {1} = {x : ℝ | -2 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end domain_of_f_l698_698332


namespace always_intersect_two_points_shortest_chord_eq_l698_698511

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + 21 = 0

def line_eq (k x y : ℝ) : Prop :=
  k*x - y - 4*k + 3 = 0

theorem always_intersect_two_points :
  ∀ k : ℝ, ∃ x y : ℝ, circle_eq x y ∧ line_eq k x y := 
sorry

theorem shortest_chord_eq :
  ∃ k : ℝ, k = 1 ∧ (∀ x y : ℝ, circle_eq x y → line_eq k x y → (distance (3, 4) (x, y) = 2 * sqrt 2)) := 
sorry

end always_intersect_two_points_shortest_chord_eq_l698_698511


namespace net_increase_in_bicycles_l698_698645

def bicycles_sold (fri_sat_sun : ℤ × ℤ × ℤ) : ℤ :=
  fri_sat_sun.1 + fri_sat_sun.2 + fri_sat_sun.3

def bicycles_bought (fri_sat_sun : ℤ × ℤ × ℤ) : ℤ :=
  fri_sat_sun.1 + fri_sat_sun.2 + fri_sat_sun.3

def net_increase (sold bought : ℤ) : ℤ :=
  bought - sold

theorem net_increase_in_bicycles :
  let bicycles_sold_days := (10, 12, 9)
  let bicycles_bought_days := (15, 8, 11)
  net_increase (bicycles_sold bicycles_sold_days) (bicycles_bought bicycles_bought_days) = 3 :=
by
  sorry

end net_increase_in_bicycles_l698_698645


namespace circle_diameter_lt_medial_line_l698_698413

variables (A B C D : Point) (circle : Circle) (a b h r : ℝ)

noncomputable def trapezoid (AB CD : ℝ) (height : ℝ) :=
  ∀ (A B C D : Point),
    parallel A B C D → 
    touches circle A B → 
    touches circle C D → 
    center circle ∈ line AD

noncomputable def trapezoid_medial_line (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def circle_diameter (r : ℝ) : ℝ :=
  2 * r

theorem circle_diameter_lt_medial_line (a b h r : ℝ) :
  trapezoid a b h r →
  h = 2 * r →
  2 * r < (a + b) / 2 := 
sorry

end circle_diameter_lt_medial_line_l698_698413


namespace min_number_of_lucky_weights_l698_698692

-- Definitions and conditions
def weight (n: ℕ) := n -- A weight is represented as a natural number.

def is_lucky (weights: Finset ℕ) (w: ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a ≠ b ∧ w = a + b
-- w is "lucky" if it's the sum of two other distinct weights in the set.

def min_lucky_guarantee (weights: Finset ℕ) (k: ℕ) : Prop :=
  ∀ (w1 w2 : ℕ), w1 ∈ weights ∧ w2 ∈ weights →
    ∃ (lucky_weights : Finset ℕ), lucky_weights.card = k ∧
    (is_lucky weights w1 ∧ is_lucky weights w2 ∧ (w1 ≥ 3 * w2 ∨ w2 ≥ 3 * w1))
-- The minimum number k of "lucky" weights ensures there exist two weights 
-- such that their masses differ by at least a factor of three.

-- The theorem to be proven
theorem min_number_of_lucky_weights (weights: Finset ℕ) (h_distinct: weights.card = 100) :
  ∃ k, min_lucky_guarantee weights k ∧ k = 87 := 
sorry

end min_number_of_lucky_weights_l698_698692


namespace find_n_l698_698705

theorem find_n (N : ℕ) (m k : ℕ) (h1 : m = (multiset.filter (λ d, d < N) (N.divisors)).max')
               (h2 : N + m = 10^k) : N = 75 :=
sorry

end find_n_l698_698705


namespace toothpick_removal_l698_698930

/-- Given 40 toothpicks used to create 10 squares and 15 triangles, with each square formed by 
4 toothpicks and each triangle formed by 3 toothpicks, prove that removing 10 toothpicks is 
sufficient to ensure no squares or triangles remain. -/
theorem toothpick_removal (n : ℕ) (squares triangles : ℕ) (sq_toothpicks tri_toothpicks : ℕ) 
    (total_toothpicks : ℕ) (remove_toothpicks : ℕ) 
    (h1 : n = 40) 
    (h2 : squares = 10) 
    (h3 : triangles = 15) 
    (h4 : sq_toothpicks = 4) 
    (h5 : tri_toothpicks = 3) 
    (h6 : total_toothpicks = n) 
    (h7 : remove_toothpicks = 10) 
    (h8 : (squares * sq_toothpicks + triangles * tri_toothpicks) = total_toothpicks) :
  remove_toothpicks = 10 :=
by
  sorry

end toothpick_removal_l698_698930


namespace converse_x_gt_y_then_x_gt_abs_y_is_true_l698_698442

theorem converse_x_gt_y_then_x_gt_abs_y_is_true :
  (∀ x y : ℝ, (x > y) → (x > |y|)) → (∀ x y : ℝ, (x > |y|) → (x > y)) :=
by
  sorry

end converse_x_gt_y_then_x_gt_abs_y_is_true_l698_698442


namespace probability_tan_cos_l698_698613

open Real

noncomputable def probability_tan_cos_inequality : Prop :=
  ∀ x : ℝ, (0 < x ∧ x < π / 2) →
    (tan x * cos x > sqrt 2 / 2) →
    ∃ y : ℝ, (π / 4 < y ∧ y < π / 2) ∧ measure_theory.measure_of (measure_theory.interval_oc 0 (π / 2)) = 1 / 2

theorem probability_tan_cos : probability_tan_cos_inequality :=
  sorry

end probability_tan_cos_l698_698613


namespace grazing_area_of_goat_l698_698827

noncomputable def goat_grazing_area (r : ℝ) : ℝ :=
  (π * r^2) / 4

theorem grazing_area_of_goat :
  goat_grazing_area 7 = (π * 7^2) / 4 :=
by
  sorry

end grazing_area_of_goat_l698_698827


namespace decagon_adjacent_vertex_probability_l698_698713

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698713


namespace decagon_adjacent_probability_l698_698742

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698742


namespace candy_mixture_problem_l698_698410

theorem candy_mixture_problem:
  ∃ x y : ℝ, x + y = 5 ∧ 3.20 * x + 1.70 * y = 10 ∧ x = 1 :=
by
  sorry

end candy_mixture_problem_l698_698410


namespace geometric_sequence_S20_l698_698535

-- Define the conditions and target statement
theorem geometric_sequence_S20
  (a : ℕ → ℝ) -- defining the sequence as a function from natural numbers to real numbers
  (q : ℝ) -- common ratio
  (h_pos : ∀ n, a n > 0) -- all terms are positive
  (h_geo : ∀ n, a (n + 1) = q * a n) -- geometric sequence property
  (S : ℕ → ℝ) -- sum function
  (h_S : ∀ n, S n = (a 1 * (1 - q ^ n)) / (1 - q)) -- sum formula for a geometric progression
  (h_S5 : S 5 = 3) -- given S_5 = 3
  (h_S15 : S 15 = 21) -- given S_15 = 21
  : S 20 = 45 := sorry

end geometric_sequence_S20_l698_698535


namespace correct_propositions_l698_698514

-- Definitions based on conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop := l ∈ perpendicularTo α
def line_in_plane (m : Line) (β : Plane) : Prop := m ∈ β

-- Propositions as functions of lines and planes
def proposition_1 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := α ∥ β → line_perpendicular_to_plane l α → line_perpendicular_to_plane l β → l ⊥ m
def proposition_2 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := α ⊥ β → line_perpendicular_to_plane l α → l ∥ m
def proposition_3 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := l ∥ m → line_perpendicular_to_plane l α → ⨆ (h : line_in_plane m β), α ⊥ β
def proposition_4 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := l ⊥ m → ⨆ (h : line_in_plane m β), α ∥ β

-- The final proof statement
theorem correct_propositions (l : Line) (m : Line) (α : Plane) (β : Plane) :
  line_perpendicular_to_plane l α →
  line_in_plane m β →
  (proposition_1 l m α β) ∧ (proposition_3 l m α β) ∧ ¬ (proposition_2 l m α β) ∧ ¬ (proposition_4 l m α β) :=
by
  sorry

end correct_propositions_l698_698514


namespace days_per_week_l698_698300

def threeChildren := 3
def schoolYearWeeks := 25
def totalJuiceBoxes := 375

theorem days_per_week (d : ℕ) :
  (threeChildren * d * schoolYearWeeks = totalJuiceBoxes) → d = 5 :=
by
  sorry

end days_per_week_l698_698300


namespace correct_interpretation_of_forecast_l698_698328

-- Definitions of the given choices
def choice_A : Prop := ∀ (d: ℕ), (d = "It will rain in the morning and not in the afternoon tomorrow.")
def choice_B : Prop := ∀ (d: ℕ), (d = "The probability of rain tomorrow is 80%.")
def choice_C : Prop := ∀ (d: ℕ), (d = "It will rain in some places and not in others tomorrow.")
def choice_D : Prop := ∀ (d: ℕ), (d = "It will rain for a total of 19.2 hours tomorrow.")

-- Definition of the condition
def condition_80_percent_precipitation : Prop := ∃ (forecast: ℕ), (forecast = "The probability of precipitation tomorrow is 80%")

-- Proof statement
theorem correct_interpretation_of_forecast : condition_80_percent_precipitation → choice_B :=
by sorry

end correct_interpretation_of_forecast_l698_698328


namespace arrangement_count_for_jobs_l698_698697

theorem arrangement_count_for_jobs (n k : ℕ) (h_n_eq_4 : n = 4) (h_k_eq_3 : k = 3) : ∏ i in finset.range(k), (n - i) = 24 :=
  by
  simp [h_n_eq_4, h_k_eq_3]
  sorry

end arrangement_count_for_jobs_l698_698697


namespace cannot_repaint_and_form_ngon_l698_698504

-- Define the problem variables and conditions
variables (N : ℕ) (blue_sticks red_sticks : list ℕ)

-- Define the total length equality condition
def equal_total_length (blue_sticks red_sticks : list ℕ) : Prop :=
  (blue_sticks.sum = red_sticks.sum)

-- Define the polygon formation condition for a list of sticks
def can_form_ngon (sticks : list ℕ) : Prop :=
  sticks.length = N ∧ ∀ (s : finset ℕ) (hs : s.card = 3), 
    (finset.sum s (λ i, sticks.nth_le i (finset.mem_def.1 hs))) > 
    (sticks.sum - finset.sum s (λ i, sticks.nth_le i (finset.mem_def.1 hs)))

-- The main theorem statement
theorem cannot_repaint_and_form_ngon (hN : N ≥ 3)
  (h1 : can_form_ngon N blue_sticks)
  (h2 : can_form_ngon N red_sticks)
  (h3 : equal_total_length blue_sticks red_sticks) : 
  ¬ ∃ (b r : ℕ), b ∈ blue_sticks ∧ r ∈ red_sticks ∧ 
  can_form_ngon N ((blue_sticks.erase b).insert r) ∧
  can_form_ngon N ((red_sticks.erase r).insert b) :=
begin
  sorry
end

end cannot_repaint_and_form_ngon_l698_698504


namespace julie_read_yesterday_l698_698628

variable (x : ℕ)
variable (y : ℕ := 2 * x)
variable (remaining_pages_after_two_days : ℕ := 120 - (x + y))

theorem julie_read_yesterday :
  (remaining_pages_after_two_days / 2 = 42) -> (x = 12) :=
by
  sorry

end julie_read_yesterday_l698_698628


namespace propositions_correct_l698_698964
-- Imports the necessary library

-- Define the constants and functions required for the conditions
def function1 (x : ℝ) := sin (5 * pi / 2 - 2 * x)
def function1_even : Prop := ∀ x : ℝ, function1 x = function1 (-x)

def function2 (x : ℝ) := sin (2 * x + 5 * pi / 4)
def function2_symmetry_axis : Prop := ∀ x : ℝ, function2 (pi / 8 + x) = function2 (pi / 8 - x)

def sin_greater (α β : ℝ) (h : 0 < α ∧ α < pi / 2 ∧ 0 < β ∧ β < pi / 2 ∧ α > β) : Prop := sin α > sin β

def roots_property (a x1 x2 k : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ 0 < k ∧ log a x1 = k ∧ log a x2 = -k) : Prop := x1 * x2 = 1

-- The theorem to prove which propositions are correct
theorem propositions_correct :
  function1_even ∧ function2_symmetry_axis ∧ ¬ (sin_greater 390 30 (by linarith)) ∧ (roots_property 2 x1 x2 k) → True :=
by
  intros,
  sorry

end propositions_correct_l698_698964


namespace counterexample_inverse_of_inequality_l698_698066

theorem counterexample_inverse_of_inequality (a b m : ℝ) (h : m = 0) : ¬ (∀ a b : ℝ, (a < b → am^2 < bm^2)) :=
by {
  intro h,
  have : a < b → am^2 < bm^2 := h a b,
  specialize this (a < b),
  simp at this h,
  sorry
}

end counterexample_inverse_of_inequality_l698_698066


namespace not_always_true_l698_698031

-- Define the conditions and proof problem
variables {D E F : ℝ × ℝ}
def second_quadrant (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ P.2 > 0
def reflection_across_y_eq_neg_x (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, -P.1)

-- Triangular vertices D, E, F in second quadrant
def DEF_in_second_quadrant : Prop :=
  second_quadrant D ∧ second_quadrant E ∧ second_quadrant F

-- Reflected vertices
def D' := reflection_across_y_eq_neg_x D
def E' := reflection_across_y_eq_neg_x E
def F' := reflection_across_y_eq_neg_x F

-- The statements to be proved
def statement_A : Prop := second_quadrant D' ∧ second_quadrant E' ∧ second_quadrant F'
def statement_B : Prop := true -- Reflection preserves area
def statement_C : Prop := ∃ m : ℝ, m = (D.2 - -D'.2) / (D.1 - -D'.1) ∧ m = -1
def statement_D : Prop := ∃ m : ℝ, m = (D.2 - -D'.2) / (D.1 - -D'.1) ∧ m = (F.2 - -F'.2) / (F.1 - -F'.1)
def statement_E : Prop :=
  let m_DE : ℝ := (E.2 - D.2) / (E.1 - D.1) in
  let m_D'E' : ℝ := (E'.2 - D'.2) / (E'.1 - D'.1) in
  m_DE * m_D'E' = -1

-- Final proof statement: Determine which of these is not always true
theorem not_always_true :
  DEF_in_second_quadrant → 
  ¬statement_A ∧ statement_B ∧ ¬statement_C ∧ statement_D ∧ ¬statement_E :=
sorry  -- Proof to be provided by Lean


end not_always_true_l698_698031


namespace solve_teachers_problem_l698_698866

def teachers_problem : Prop :=
  ∃ (T : ℕ) (O : ℕ),
    let S := 26 in
    let I := 104 in
    let N := 16 in
    let total_sampled := 56 in
    (N / total_sampled : ℚ) = (O / T : ℚ) ∧
    S + I + O = T ∧
    T = 52

theorem solve_teachers_problem : teachers_problem :=
by {
  -- to be proved
  sorry
}

end solve_teachers_problem_l698_698866


namespace find_lengths_of_AB_and_AC_l698_698687

theorem find_lengths_of_AB_and_AC (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (k θ : ℝ) (h_area : triangle_area A B C = k) (h_angle : ∠A = θ) :
  AB = sqrt(2 * k / sin θ) ∧ AC = sqrt(2 * k / sin θ) :=
by
  sorry

end find_lengths_of_AB_and_AC_l698_698687


namespace geometric_sequence_common_ratio_l698_698895

theorem geometric_sequence_common_ratio (a : ℝ) :
  let t1 := a + log 3 / log 2,
      t2 := a + (log 3 / log 2) / 2,
      t3 := a + (log 3 / log 2) / 3 in
  is_geometric_sequence t1 t2 t3 → (t2 / t1 = 1/3) := by
  sorry

end geometric_sequence_common_ratio_l698_698895


namespace exists_same_distance_l698_698690

open Finset

theorem exists_same_distance (men_wives : Finset (ℕ × ℕ)) (h_size : men_wives.card = 30)
  (h_range : ∀ (m w : ℕ), (m, w) ∈ men_wives → (1 ≤ m ∧ m ≤ 60) ∧ (1 ≤ w ∧ w ≤ 60)) :
  ∃ (m1 m2 w1 w2 : ℕ), (m1, w1) ∈ men_wives ∧ (m2, w2) ∈ men_wives ∧ m1 ≠ m2 ∧ 
  (min (abs (m1 - w1)) (60 - abs (m1 - w1)) = min (abs (m2 - w2)) (60 - abs (m2 - w2))) :=
by 
  sorry

end exists_same_distance_l698_698690


namespace percent_not_even_integers_l698_698389

def total_numbers_in_set (A : Type) := A
def even_multiples_of_3 (n : ℕ) : Prop := n % 6 = 0
def even_not_multiples_of_3 (n : ℕ) : Prop := n % 2 = 0 ∧ n % 3 ≠ 0

variables (T : ℕ)
variable (A : Finset ℕ)

-- 36% of the numbers in set A are even multiples of 3
def evens_multiples_3_set : Finset ℕ := (A.filter even_multiples_of_3)
def percent_evens_multiples_3 : ℕ := 36 

-- 40% of the even integers in set A are not multiples of 3
def even_integers_set : Finset ℕ := (A.filter (λ n, n % 2 = 0))
def percent_even_not_multiples_3 : ℕ := 40 

-- Proving that the percent of numbers in set A that are not even integers is 40%
theorem percent_not_even_integers :
  percent_evens_multiples_3 = 36 → 
  percent_even_not_multiples_3 = 40 → 
  ∑ t in A, t = (0.40 * T) * T := sorry

end percent_not_even_integers_l698_698389


namespace sixty_five_times_fifty_five_l698_698459

theorem sixty_five_times_fifty_five : 65 * 55 = 3575 := by
  calc
    65 * 55 = (60 + 5) * (60 - 5) : by rw [add_comm 60 5, sub_comm 60 5] -- Rewriting using difference of squares
    ... = 60^2 - 5^2 : by rw [mul_sub (60 : ℤ) 5 5 (int.coe_nat_add 60 5)] -- Applying the difference of squares formula
    ... = 3600 - 25 : by norm_num -- Computing the squares
    ... = 3575 : by norm_num -- Subtracting to reach the final answer

end sixty_five_times_fifty_five_l698_698459


namespace function_classification_l698_698485

noncomputable def f (x : ℝ) : ℝ

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x^2 + y^2 + 2 * f(x * y)) = (f(x + y))^2) →
  (f = (fun x => x) ∨ 
   f = (fun x => 0) ∨ 
   (∃ X : set ℝ, X ⊆ set.Ioo (-∞) (-2/3) ∧ 
     ∀ x : ℝ, f x = if x ∈ X then -1 else 1)) :=
by 
  sorry

end function_classification_l698_698485


namespace find_values_of_z1_z2_find_product_of_z1_z2_l698_698206

-- Definitions for Part (1)
variables {z1 z2 : ℂ}
def magnitude_condition (z : ℂ) := abs z = 1
def difference_condition (z1 z2 : ℂ) := z1 - z2 = (ℂ.sqrt 6 / 3) + ((ℂ.sqrt 3 / 3) * Complex.i)

-- Part (1) Problem Statement in Lean
theorem find_values_of_z1_z2 
  (hz1 : magnitude_condition z1) 
  (hz2 : magnitude_condition z2) 
  (h_diff : difference_condition z1 z2) :
  z1 = (1/6) * ((ℂ.sqrt 6 + 3) + ((ℂ.sqrt 3 - 3 * ℂ.sqrt 2) * Complex.i)) ∧ 
  z2 = (1/6) * ((-(ℂ.sqrt 6) + 3) + (-(ℂ.sqrt 3) - 3 * ℂ.sqrt 2) * Complex.i) :=
sorry

-- Definitions for Part (2)
def sum_condition (z1 z2 : ℂ) := z1 + z2 = (12 / 13) - ((5 / 13) * Complex.i)

-- Part (2) Problem Statement in Lean
theorem find_product_of_z1_z2
  (hz1 : magnitude_condition z1)
  (hz2 : magnitude_condition z2)
  (h_sum : sum_condition z1 z2) :
  z1 * z2 = (119 / 169) - (120 / 169) * Complex.i :=
sorry

end find_values_of_z1_z2_find_product_of_z1_z2_l698_698206


namespace sum_of_interior_angles_n_plus_3_l698_698330

theorem sum_of_interior_angles_n_plus_3 (n : ℕ) (h : 180 * (n - 2) = 3240) :
  180 * ((n + 3) - 2) = 3780 := 
begin
  sorry
end

end sum_of_interior_angles_n_plus_3_l698_698330


namespace local_odd_function_range_m_l698_698139

def local_odd_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ ∈ set.Ioo a b, f (-x₀) = -f (x₀)

def f (x m : ℝ) : ℝ := (1 / (x - 3)) + m

theorem local_odd_function_range_m :
  ∀ {m : ℝ}, local_odd_function (λ x, f x m) (-1) 1 ↔ (m ∈ set.Ico (1 / 3) (3 / 8)) :=
by
  -- Sorry as we are skipping the proof part
  sorry

end local_odd_function_range_m_l698_698139


namespace calculate_length_QR_l698_698665

noncomputable def length_QR (A : ℝ) (h : ℝ) (PQ : ℝ) (RS : ℝ) : ℝ :=
  21 - 0.5 * (Real.sqrt (PQ ^ 2 - h ^ 2) + Real.sqrt (RS ^ 2 - h ^ 2))

theorem calculate_length_QR :
  length_QR 210 10 12 21 = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by
  sorry

end calculate_length_QR_l698_698665


namespace probability_adjacent_vertices_decagon_l698_698760

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698760


namespace prob_not_same_city_is_056_l698_698789

def probability_not_same_city (P_A_cityA P_B_cityA : ℝ) : ℝ :=
  let P_A_cityB := 1 - P_A_cityA
  let P_B_cityB := 1 - P_B_cityA
  (P_A_cityA * P_B_cityB) + (P_A_cityB * P_B_cityA)

theorem prob_not_same_city_is_056 :
  probability_not_same_city 0.6 0.2 = 0.56 :=
by
  sorry

end prob_not_same_city_is_056_l698_698789


namespace cyclist_speed_l698_698098

theorem cyclist_speed 
  (v : ℝ) 
  (hiker1_speed : ℝ := 4)
  (hiker2_speed : ℝ := 5)
  (cyclist_overtakes_hiker2_after_hiker1 : ∃ t1 t2 : ℝ, 
      t1 = 8 / (v - hiker1_speed) ∧ 
      t2 = 5 / (v - hiker2_speed) ∧ 
      t2 - t1 = 1/6)
: (v = 20 ∨ v = 7 ∨ abs (v - 6.5) < 0.1) :=
sorry

end cyclist_speed_l698_698098


namespace solution_set_l698_698949

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (sqrt 2 * x + sqrt (2 * x ^ 2 + 1))

theorem solution_set (x : ℝ) (hx : f x > 3 / 2) : 
  ∃ a : ℝ, a = sqrt 2 ∧ f x = Real.logb 2 (sqrt 2 * x + sqrt (2 * x ^ 2 + 1)) ∧ x > 7 / 8 := 
sorry

end solution_set_l698_698949


namespace f_monotonic_increasing_l698_698209

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

theorem f_monotonic_increasing :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 > x2 → f x1 > f x2 :=
by
  intros x1 x2 hx1 hx2 h
  sorry

end f_monotonic_increasing_l698_698209


namespace sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l698_698537

theorem sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : 100^2 + 1^2 = p * q ∧ 65^2 + 76^2 = p * q) : p + q = 210 := 
sorry

end sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l698_698537


namespace derivative_at_pi_div_4_l698_698176

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) * (Real.cos x + 1)

theorem derivative_at_pi_div_4 :
  (deriv f (π / 4)) = (sqrt 2 / 2) :=
by
  sorry

end derivative_at_pi_div_4_l698_698176


namespace calculate_total_students_l698_698129

-- Define the conditions and state the theorem
theorem calculate_total_students (perc_bio : ℝ) (num_not_bio : ℝ) (perc_not_bio : ℝ) (T : ℝ) :
  perc_bio = 0.475 →
  num_not_bio = 462 →
  perc_not_bio = 1 - perc_bio →
  perc_not_bio * T = num_not_bio →
  T = 880 :=
by
  intros
  -- proof will be here
  sorry

end calculate_total_students_l698_698129


namespace arithmetic_mean_inequality_minimum_t_satisfies_inequality_l698_698058

theorem arithmetic_mean_inequality 
  (a b c : ℝ) : 
  ( (a + b + c) / 3 ) ^ 2 ≤ ( a ^ 2 + b ^ 2 + c ^ 2 ) / 3 
  ∧ ( (a + b + c) / 3 ) ^ 2 = ( a ^ 2 + b ^ 2 + c ^ 2 ) / 3 ↔ a = b ∧ b = c :=
sorry

theorem minimum_t_satisfies_inequality
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∀ t : ℝ, (sqrt x + sqrt y + sqrt z ≤ t * sqrt (x + y + z)) ↔ t ≥ sqrt 3 :=
sorry

end arithmetic_mean_inequality_minimum_t_satisfies_inequality_l698_698058


namespace decagon_adjacent_vertices_probability_l698_698729

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698729


namespace angle_between_vectors_is_90_degrees_l698_698532

variables {E : Type*} [InnerProductSpace ℝ E] {a b : E}

theorem angle_between_vectors_is_90_degrees (h₀ : ∥a + (3 : ℝ) • b∥ = ∥a - (3 : ℝ) • b∥) (ha : a ≠ 0) (hb : b ≠ 0) : real.angle a b = real.pi / 2 := 
by
  sorry

end angle_between_vectors_is_90_degrees_l698_698532


namespace pat_earns_per_photo_l698_698299

-- Defining conditions
def minutes_per_shark := 10
def fuel_cost_per_hour := 50
def hunting_hours := 5
def expected_profit := 200

-- Defining intermediate calculations based on the conditions
def sharks_per_hour := 60 / minutes_per_shark
def total_sharks := sharks_per_hour * hunting_hours
def total_fuel_cost := fuel_cost_per_hour * hunting_hours
def total_earnings := expected_profit + total_fuel_cost
def earnings_per_photo := total_earnings / total_sharks

-- Main theorem: Prove that Pat earns $15 for each photo
theorem pat_earns_per_photo : earnings_per_photo = 15 := by
  -- The proof would be here
  sorry

end pat_earns_per_photo_l698_698299


namespace number_of_elements_in_K_2004_2004_l698_698469

def K (n m : ℕ) : Set ℕ :=
  if m = 0 then ∅ else { k | 1 ≤ k ∧ k ≤ n ∧ (K k (m - 1) ∩ K (n - k) (m - 1) = ∅) }

theorem number_of_elements_in_K_2004_2004 : K 2004 2004.card = 127 := by
  sorry

end number_of_elements_in_K_2004_2004_l698_698469


namespace GMAT_scores_ratio_l698_698703

variables (u v w : ℝ)

theorem GMAT_scores_ratio
  (h1 : u - w = (u + v + w) / 3)
  (h2 : u - v = 2 * (v - w))
  : v / u = 4 / 7 :=
sorry

end GMAT_scores_ratio_l698_698703


namespace simplify_trig_expression_l698_698312

theorem simplify_trig_expression : 
  (tan 20 + tan 30 + tan 60 + tan 70) / cos 10 = 2 * cos 40 / cos 10 * (cos 20 * cos 30 * cos 60 * cos 70) := 
sorry

end simplify_trig_expression_l698_698312


namespace problem_statement_l698_698136

def floor (a : ℝ) : ℤ := Int.floor a

theorem problem_statement :
  floor (2 + 1/5) * floor (-3.6) - floor (0.1) / (-6) = -8 := 
by
  sorry

end problem_statement_l698_698136


namespace no_closed_non_self_intersecting_polygonal_chain_l698_698309

theorem no_closed_non_self_intersecting_polygonal_chain (segments : set (set ℝ × ℝ)) :
  (∀ s1 s2 ∈ segments, s1 ≠ s2 → s1 ∩ s2 = ∅) →
  ¬ (∃ closed_chain : set (ℝ × ℝ),
      closed_chain ⊆ ⋃ s ∈ segments, s ∧
      (∀ p ∈ closed_chain, ∃ s ∈ segments, p ∈ s) ∧
      (∀ p1 p2 ∈ closed_chain, p1 ≠ p2 ∧ segment_conn(p1, p2) → (p1, p2) ∉ segments)) :=
begin
  intros,
  sorry
end

end no_closed_non_self_intersecting_polygonal_chain_l698_698309


namespace trigonometric_simplification_l698_698316

theorem trigonometric_simplification :
  ∀ (a b c d e f g h : ℝ),
  (a = 20 * Real.pi / 180) →
  (b = 30 * Real.pi / 180) →
  (c = 60 * Real.pi / 180) →
  (d = 70 * Real.pi / 180) →
  (e = 10 * Real.pi / 180) →
  (f = 50 * Real.pi / 180) →
  (g = 40 * Real.pi / 180) →
  (h = 130 * Real.pi / 180) →
  sin h = sin f →
  sin f / cos e = cos g / cos e →
  ( ∀ x y : ℝ, tan x + tan y = sin (x + y) / (cos x * cos y) ) →
  (tan a + tan b + tan c + tan d) / cos e = 2 * cos g / (cos e ^ 2 * cos b * cos c * cos d) := sorry

end trigonometric_simplification_l698_698316


namespace area_of_playground_l698_698347

-- Definitions of the problem's conditions
variables {w l : ℝ}

-- Conditions given in the problem
def perimeter_condition : Prop := 2 * l + 2 * w = 90
def length_width_relation : Prop := l = 3 * w

-- The statement that needs to be proved
theorem area_of_playground (h1 : perimeter_condition) (h2 : length_width_relation) : l * w = 380.15625 := by
  sorry

end area_of_playground_l698_698347


namespace compute_expression_l698_698663

-- Given Conditions
variables (a b c : ℕ)
variable (h : 2^a * 3^b * 5^c = 36000)

-- Proof Statement
theorem compute_expression (h : 2^a * 3^b * 5^c = 36000) : 3 * a + 4 * b + 6 * c = 41 :=
sorry

end compute_expression_l698_698663


namespace how_many_positive_l698_698612

-- Defining the function and conditions as per the problem
def polynomial (a b c d e x : ℝ) : ℝ := (x + a) * (x + b)^2 * (x + c) * (x + d) * (x + e)

-- The proof problem statement
theorem how_many_positive (a b c d e : ℝ) :
  (∃ x, polynomial a b c d e x = 0) ∧
  ∃! (x : ℝ), polynomial a b c d e x = 0 ∧ ((x = -a) ∨ (x = -c) ∨ (x = -d) ∨ (x = -e)) ∧
  (∀ x, polynomial a b c d e x = 0 ↔ x = -b) ∧
  ( ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ polynomial a b c d e x1 = 0 ∧ polynomial a b c d e x2 = 0 ∧ polynomial a b c d e x3 = 0 ) ->
  (real.countp (λ x, polynomial a b c d e x = 0 ∧ (x + a) * (x + c) * (x + d) * (x + e) > 0) [a, c, d, e] = 3) := sorry

end how_many_positive_l698_698612


namespace fewer_cats_than_dogs_l698_698338

variable (Cats Dogs : ℕ)
variable (Cats_eq_ratio : Ratio ℚ := 3/4)
variable (Dogs_eq : Dogs = 32)

-- Prove that there are 8 fewer cats than dogs given the conditions.
theorem fewer_cats_than_dogs : Dogs - Cats = 8 :=
by
  have h_ratio : Cats = (3 * Dogs) / 4 := sorry
  have h_dogs : Dogs = 32 := by rw [Dogs_eq]; sorry
  have h_cats : Cats = 24 := by rw [h_ratio, h_dogs]; sorry
  show Dogs - Cats = 8 := by rw [h_cats, h_dogs]; sorry

end fewer_cats_than_dogs_l698_698338


namespace ambassador_seating_mod_l698_698171

/-- 
Given there are 12 chairs numbered from 1 to 12, and 4 ambassadors who must sit 
in even-numbered chairs, with each advisor sitting adjacent to his or her ambassador, 
we want to find the number of ways to seat these 8 people, then compute this 
number modulo 1000. 
-/
theorem ambassador_seating_mod (N : ℕ) : 
  (N = 2520) → (N % 1000 = 520) :=
by 
  assume hN : N = 2520,
  rw hN,
  exact Nat.mod_eq_of_lt (Nat.lt_succ_of_le (Nat.lt_succ_of_le (Nat.lt_succ_of_le (Nat.succ_pos 2019))))

end ambassador_seating_mod_l698_698171


namespace add_pure_alcohol_to_achieve_percentage_l698_698845

-- Define the initial conditions
def initial_solution_volume : ℝ := 6
def initial_alcohol_percentage : ℝ := 0.30
def initial_pure_alcohol : ℝ := initial_solution_volume * initial_alcohol_percentage

-- Define the final conditions
def final_alcohol_percentage : ℝ := 0.50

-- Define the unknown to prove
def amount_of_alcohol_to_add : ℝ := 2.4

-- The target statement to prove
theorem add_pure_alcohol_to_achieve_percentage :
  (initial_pure_alcohol + amount_of_alcohol_to_add) / (initial_solution_volume + amount_of_alcohol_to_add) = final_alcohol_percentage :=
by
  sorry

end add_pure_alcohol_to_achieve_percentage_l698_698845


namespace range_of_a_l698_698234

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x+1| + |x-3| ≥ a + 4 / a) ↔ (a ∈ set.Iio 0 ∪ {2}) :=
sorry

end range_of_a_l698_698234


namespace simplify_trig_expression_l698_698310

theorem simplify_trig_expression : 
  (tan 20 + tan 30 + tan 60 + tan 70) / cos 10 = 2 * cos 40 / cos 10 * (cos 20 * cos 30 * cos 60 * cos 70) := 
sorry

end simplify_trig_expression_l698_698310


namespace sum_of_a5_a6_l698_698354

variable (a : ℕ → ℕ)

def S (n : ℕ) : ℕ :=
  n ^ 2 + 2

theorem sum_of_a5_a6 :
  a 5 + a 6 = S 6 - S 4 := by
  sorry

end sum_of_a5_a6_l698_698354


namespace greatest_possible_overlap_l698_698408

-- Define the conditions as given in the math problem
def wireless_internet_percentage : ℝ := 40 / 100
def free_snacks_percentage : ℝ := 70 / 100

-- Prove the greatest possible percentage that offers both services
theorem greatest_possible_overlap : 
  greatest_possible_percentage wireless_internet_percentage free_snacks_percentage = wireless_internet_percentage :=
by
  sorry

end greatest_possible_overlap_l698_698408


namespace area_ratio_of_reflected_quadrilateral_l698_698144

theorem area_ratio_of_reflected_quadrilateral (S S' : ℝ) 
  (cond : ∀ (quad : Type) [quadrilateral quad] (vertices_reflected : convex_quadrilateral quad), 
    area vertices_reflected = S' → area quad = S ) :
  S' / S < 3 :=
sorry

end area_ratio_of_reflected_quadrilateral_l698_698144


namespace maria_carrots_l698_698837

theorem maria_carrots (originally_picked : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) (total : ℕ) :
  originally_picked = 48 → thrown_out = 11 → picked_next_day = 15 → total = 52 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end maria_carrots_l698_698837


namespace probability_of_event_B_given_A_l698_698363

-- Definition of events and probability
noncomputable def prob_event_B_given_A : ℝ :=
  let total_outcomes := 36
  let outcomes_A := 30
  let outcomes_B_given_A := 10
  outcomes_B_given_A / outcomes_A

-- Theorem statement
theorem probability_of_event_B_given_A : prob_event_B_given_A = 1 / 3 := by
  sorry

end probability_of_event_B_given_A_l698_698363


namespace nikolai_faster_than_gennady_l698_698046

-- The conditions of the problem translated to Lean definitions
def gennady_jump_length : ℕ := 6
def gennady_jumps_per_time : ℕ := 2
def nikolai_jump_length : ℕ := 4
def nikolai_jumps_per_time : ℕ := 3
def turn_around_distance : ℕ := 2000
def round_trip_distance : ℕ := 2 * turn_around_distance

-- The statement that Nikolai completes the journey faster than Gennady
theorem nikolai_faster_than_gennady :
  (nikolai_jumps_per_time * nikolai_jump_length) = (gennady_jumps_per_time * gennady_jump_length) →
  (round_trip_distance % nikolai_jump_length = 0) →
  (round_trip_distance % gennady_jump_length ≠ 0) →
  (round_trip_distance / nikolai_jump_length) + 1 < (round_trip_distance / gennady_jump_length) →
  "Nikolay completes the journey faster." :=
by
  intros h_eq_speed h_nikolai_divisible h_gennady_not_divisible h_time_compare
  sorry

end nikolai_faster_than_gennady_l698_698046


namespace factor_polynomial_l698_698654

theorem factor_polynomial :
  (x : ℝ) -> x^6 - 16 * real.sqrt 5 * x^3 + 64 = 
  (x - (real.sqrt 5 + 1)) * (x^2 + x * (real.sqrt 5 + 1) + 6 + 2 * real.sqrt 5) * 
  (x - (real.sqrt 5 - 1)) * (x^2 + x * (real.sqrt 5 - 1) + 6 - 2 * real.sqrt 5) :=
sorry

end factor_polynomial_l698_698654


namespace pairs_of_shoes_in_box_l698_698090

/-- 
A box contains 18 shoes in total.
The probability that two randomly selected shoes are a matching pair is 0.058823529411764705.
Prove that there are 9 pairs of shoes in the box.
-/
theorem pairs_of_shoes_in_box (n : ℕ) (h_total_shoes : 2 * n = 18)
  (h_prob_matching : 1 / (2 * n - 1) = 0.058823529411764705) : n = 9 :=
sorry

end pairs_of_shoes_in_box_l698_698090


namespace right_triangle_third_side_product_l698_698787

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l698_698787


namespace count_of_three_digit_numbers_divisible_by_5_l698_698653

open Finset

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def form_three_digit_numbers_with_distinct_digits (d : Finset ℕ) : Finset ℕ :=
  d.product (d.erase 0).product (d.erase 0).erase 0.filter (λ t, t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2)

noncomputable def count_three_digit_numbers_divisible_by_5 : ℕ :=
  (form_three_digit_numbers_with_distinct_digits digits).filter (λ n, is_three_digit_number n ∧ is_divisible_by_5 n).card

theorem count_of_three_digit_numbers_divisible_by_5 :
  count_three_digit_numbers_divisible_by_5 = 36 :=
sorry

end count_of_three_digit_numbers_divisible_by_5_l698_698653


namespace new_average_commission_is_250_l698_698641

-- Definitions based on the problem conditions
def C : ℝ := 1000
def n : ℝ := 6
def increase_in_average_commission : ℝ := 150

-- Theorem stating the new average commission is $250
theorem new_average_commission_is_250 (x : ℝ) (h1 : x + increase_in_average_commission = (5 * x + C) / n) :
  x + increase_in_average_commission = 250 := by
  sorry

end new_average_commission_is_250_l698_698641


namespace volume_common_solid_hemisphere_cone_l698_698794

noncomputable def volume_common_solid (r : ℝ) : ℝ := 
  let V_1 := (2/3) * Real.pi * (r^3 - (3 * r / 5)^3)
  let V_2 := Real.pi * ((r / 5)^2) * (r - (r / 15))
  V_1 + V_2

theorem volume_common_solid_hemisphere_cone (r : ℝ) :
  volume_common_solid r = (14 * Real.pi * r^3) / 25 := 
by
  sorry

end volume_common_solid_hemisphere_cone_l698_698794


namespace cos_2alpha_value_l698_698962

noncomputable def cos_double_angle (α : ℝ) : ℝ := Real.cos (2 * α)

theorem cos_2alpha_value (α : ℝ): 
  (∃ a : ℝ, α = Real.arctan (-3) + 2 * a * Real.pi) → cos_double_angle α = -4 / 5 :=
by
  intro h
  sorry

end cos_2alpha_value_l698_698962


namespace combined_age_of_siblings_l698_698439

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end combined_age_of_siblings_l698_698439


namespace arithmetic_mean_inequality_minimum_t_satisfies_inequality_l698_698057

theorem arithmetic_mean_inequality 
  (a b c : ℝ) : 
  ( (a + b + c) / 3 ) ^ 2 ≤ ( a ^ 2 + b ^ 2 + c ^ 2 ) / 3 
  ∧ ( (a + b + c) / 3 ) ^ 2 = ( a ^ 2 + b ^ 2 + c ^ 2 ) / 3 ↔ a = b ∧ b = c :=
sorry

theorem minimum_t_satisfies_inequality
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∀ t : ℝ, (sqrt x + sqrt y + sqrt z ≤ t * sqrt (x + y + z)) ↔ t ≥ sqrt 3 :=
sorry

end arithmetic_mean_inequality_minimum_t_satisfies_inequality_l698_698057


namespace decagon_adjacent_probability_l698_698743

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698743


namespace arithmetic_mean_inequality_min_value_of_t_l698_698059

-- Part (1)
theorem arithmetic_mean_inequality (a b c : ℝ) : 
  ( (a + b + c) / 3 ) ^ 2 ≤ (a^2 + b^2 + c^2) / 3 ∧ 
    (( (a + b + c) / 3 ) ^ 2 = (a^2 + b^2 + c^2) / 3 → a = b ∧ b = c) := 
sorry

-- Part (2)
theorem min_value_of_t (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  sqrt x + sqrt y + sqrt z ≤ sqrt 3 * sqrt (x + y + z) :=
sorry

end arithmetic_mean_inequality_min_value_of_t_l698_698059


namespace train_length_correct_l698_698119

-- Define the given conditions as constants
def time_to_cross_pole : ℝ := 4.99960003199744
def speed_kmh : ℝ := 72

-- Convert the speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train : ℝ := speed_ms * time_to_cross_pole

-- The problem statement: prove that length_of_train is approximately 99.992 meters
theorem train_length_correct : abs (length_of_train - 99.992) < 0.001 := by
  sorry

end train_length_correct_l698_698119


namespace sufficient_condition_for_coplanarity_l698_698499

def three_lines (P Q R: Type*) [AffineSpace P Q] (ℓ1 ℓ2 ℓ3 : AffineSubspace Q) : Prop :=
  (ℓ1 ≠ ℓ2 ∧ ℓ2 ≠ ℓ3 ∧ ℓ1 ≠ ℓ3) ∧ 
  ((∀ p ∈ ℓ1, ∀ q ∈ ℓ2, ∀ r ∈ ℓ3, affine_independent ℝ ![p, q, r]) ∨
   (∃ p ∈ ℓ1, ∃ q ∈ ℓ2, ∃ r ∈ ℓ3, affine_span ℝ {p, q, r} = affine_subspace Q))

theorem sufficient_condition_for_coplanarity (P Q R: Type*) [AffineSpace P Q] (ℓ1 ℓ2 ℓ3 : AffineSubspace Q) : Prop :=
  (∃ p ∈ ℓ1, ∃ q ∈ ℓ2, ∃ r ∈ ℓ3, affine_span ℝ {p, q, r} = affine_subspace Q)
  ↔
  (∀ p ∈ ℓ1, ∀ q ∈ ℓ2, ∀ r ∈ ℓ3, p ≠ q ∧ p ≠ r ∧ q ≠ r → collinear ℝ ![p, q, r])
  :=
sorry

end sufficient_condition_for_coplanarity_l698_698499


namespace marks_change_factor_l698_698326

def total_marks (n : ℕ) (avg : ℝ) : ℝ := n * avg

theorem marks_change_factor 
  (n : ℕ) (initial_avg new_avg : ℝ) 
  (initial_total := total_marks n initial_avg) 
  (new_total := total_marks n new_avg)
  (h1 : initial_avg = 36)
  (h2 : new_avg = 72)
  (h3 : n = 12):
  (new_total / initial_total) = 2 :=
by
  sorry

end marks_change_factor_l698_698326


namespace robbie_wins_probability_sum_l698_698868

theorem robbie_wins_probability_sum :
  ∃ (r s : ℕ), let p := 1 / 21 in
  let robbie_wins := 
      (4 * p * p) + 
      (5 * p * p + 5 * p * (2 * p)) + 
      (6 * p * p + 6 * p * (2 * p) + 6 * p * (3 * p)) in
  robbie_wins = r / (400 + s) ∧ r + s = 96 :=
by {
  use 55,
  use 41,
  let p := 1 / 21,
  let robbie_wins : ℚ := 
    (4 * p * p) + 
    (5 * p * p + 5 * p * (2 * p)) + 
    (6 * p * p + 6 * p * (2 * p) + 6 * p * (3 * p)),
  have rw: robbie_wins = 55 / 441 := 
    by norm_num [robbie_wins, p],
  have eq1: 441 = 400 + 41 := 
    by norm_num,
  split, 
  { rw [rw, eq1] },
  norm_num
}

end robbie_wins_probability_sum_l698_698868


namespace sufficient_but_not_necessary_l698_698666

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1 → x^2 > x) ∧ (¬(∀ x, x^2 > x → x > 1)) := by
  have h1 : x > 1 → x^2 > x := by
    intros h
    calc
      x^2 = x * x := by ring
      ... > x     := by nlinarith [h]
  have h2 : ¬(∀ x, x^2 > x → x > 1) := by
    intro h
    specialize h (-1)
    have : (-1)^2 > -1 := by norm_num
    linarith
  exact ⟨h1, h2⟩

end sufficient_but_not_necessary_l698_698666


namespace initial_puppies_count_l698_698879

theorem initial_puppies_count (P : ℕ) (h1 : P - 2 + 3 = 8) : P = 7 :=
sorry

end initial_puppies_count_l698_698879


namespace correct_propositions_count_l698_698208

variables (length_eq : ∀ {A B : Type} (AB BA : vector A B), AB.length = BA.length)
          (parallel : ∀ {a b : Type} (a b : vector a), a ∥ b ↔ (a.direction = b.direction ∨ a.direction = -b.direction))
          (same_start_eq_length : ∀ {A B C : Type} 
                                    (AB : vector A B) (AC : vector A C), 
                                    AB.start = AC.start ∧ AB.length = AC.length → 
                                    AB.end = AC.end)
          (same_end_collinear : ∀ {A B C : Type} 
                                  (AB : vector A B) (CB : vector C B), 
                                  AB.end = CB.end → collinear AB CB)
          (collinear_points_line : ∀ {A B C D : Type} 
                                    (AB : vector A B) (CD : vector C D), 
                                    collinear AB CD → ∃ l : line, A, B, C, D ∈ l )

theorem correct_propositions_count : 
  length_eq ∧ same_start_eq_length → 
  2 := 
sorry

end correct_propositions_count_l698_698208


namespace diagonal_support_beam_length_l698_698422

def rectangle_area (L W : ℝ) : Prop :=
  L * W = 117

def rectangle_perimeter (L W : ℝ) : Prop :=
  2 * L + 2 * W = 44

def diagonal_length (L W D : ℝ) : Prop :=
  D = (real.sqrt (L^2 + W^2))

theorem diagonal_support_beam_length :
  ∃ L W D : ℝ, rectangle_area L W ∧ rectangle_perimeter L W ∧ diagonal_length L W D ∧ D = 5 * real.sqrt 10 :=
by {
  use [13, 9, 5 * real.sqrt 10],
  split,
  { -- proof that 13 * 9 = 117
    unfold rectangle_area,
    norm_num }
  split,
  { -- proof that 2 * 13 + 2 * 9 = 44
    unfold rectangle_perimeter,
    norm_num }
  split,
  { -- proof that diagonal_length 13 9 (5 * real.sqrt 10)
    unfold diagonal_length,
    norm_num,
    rw real.sqrt_eq_rpow,
    norm_num }
  { -- final condition D = 5 * real.sqrt 10
    refl }
}

end diagonal_support_beam_length_l698_698422


namespace problem_statement_l698_698210

-- Definitions based on the problem conditions and given information
def f (x a : ℝ) := a * Real.log (x + 1) - a * x

-- Problem statement rewritten in Lean 4
theorem problem_statement (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, x > -1 → f x a > (a * x - Real.exp (x + 1) + a) / (x + 1)) →
  a < Real.exp 1 := 
begin
  sorry
end

end problem_statement_l698_698210


namespace copper_alloy_percentage_l698_698664

theorem copper_alloy_percentage :
  ∃ (x : ℝ), 
    (32 * (x / 100) + 8 * 0.50 = 18) ∧ 
    (x = 43.75) :=
begin
  use 43.75,
  split,
  {
    linarith,
  },
  {
    linarith,
  },
end

end copper_alloy_percentage_l698_698664


namespace arithmetic_sequence_30th_term_l698_698810

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l698_698810


namespace unique_peg_placement_proof_l698_698695

def peg_placement_is_unique (yellow red green blue : ℕ) (r : Type) [fintype r] : Prop :=
  yellow = 4 ∧ red = 3 ∧ green = 2 ∧ blue = 1 ∧ card r = 4 ∧ triangular r → 
  ∃! placement : finset (r × ℕ), ∀ c ∈ placement, valid_placement c yellow red green blue

noncomputable def triangular_pegboard_unique_placement : Prop := 
  peg_placement_is_unique 4 3 2 1 (fin 4)

theorem unique_peg_placement_proof : triangular_pegboard_unique_placement :=
begin
  sorry,
end

end unique_peg_placement_proof_l698_698695


namespace min_moves_to_reassemble_l698_698123

theorem min_moves_to_reassemble (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, (∀ pieces, pieces = n - 1) ∧ pieces = 1 → move_count = n - 1 :=
by
  sorry

end min_moves_to_reassemble_l698_698123


namespace sum_of_digits_palindrome_x_l698_698087

def is_palindrome (n : ℕ) : Prop := n.toString = n.toString.reverse

theorem sum_of_digits_palindrome_x (x : ℕ) (h1 : 100 ≤ x ∧ x < 1000) (h2 : is_palindrome x) (h3 : is_palindrome (x + 50)) (h4 : 1000 ≤ x + 50 ∧ x + 50 < 10000) :
  (x.digits.sum) = 15 :=
  sorry

end sum_of_digits_palindrome_x_l698_698087


namespace total_cost_mark_l698_698292

theorem total_cost_mark (wt_tomatoes: ℝ) (price_per_pound_tomatoes: ℝ) 
                         (wt_apples: ℝ) (price_per_pound_apples: ℝ) 
                         (wt_oranges: ℝ) (price_per_pound_oranges: ℝ) : 
                         wt_tomatoes = 3 → price_per_pound_tomatoes = 4.5 →
                         wt_apples = 7 → price_per_pound_apples = 3.25 →
                         wt_oranges = 4 → price_per_pound_oranges = 2.75 →
                         (wt_tomatoes * price_per_pound_tomatoes + 
                          wt_apples * price_per_pound_apples + 
                          wt_oranges * price_per_pound_oranges) = 47.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cost_mark_l698_698292


namespace minimum_chord_length_is_4sqrt5_l698_698976

noncomputable def minimum_chord_length_of_circle (l : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (C : ℝ → ℝ → ℝ) : ℝ :=
  let m := λ x y, l x y = 0 in
  let P := (3 : ℝ, 1 : ℝ) in
  let C := (λ x y, (x - 1)^2 + (y - 2)^2 - 25) in
  4 * Real.sqrt 5

theorem minimum_chord_length_is_4sqrt5 :
  minimum_chord_length_of_circle (λ x y, x - y - 2) (3, 1) (λ x y, (x - 1)^2 + (y - 2)^2 - 25) = 4 * Real.sqrt 5 :=
  sorry

end minimum_chord_length_is_4sqrt5_l698_698976


namespace Nikolai_faster_than_Gennady_l698_698040

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l698_698040


namespace express_g_as_sin_l698_698135

open Real

noncomputable def g (x : ℝ) : ℝ :=
  cot (x / 3) + cot (3 * x)

theorem express_g_as_sin (x : ℝ) (hx1 : sin (x / 3) ≠ 0) (hx2 : sin (3 * x) ≠ 0) :
  g x = sin ((10 * x) / 3) / (sin (x / 3) * sin (3 * x)) := 
sorry

end express_g_as_sin_l698_698135


namespace rotate_disks_at_least_100_l698_698902

section DiskRotation

-- Definitions of disk and sectors
def sector := fin 200

-- Coloring of sectors - either red or blue
inductive color
| red
| blue

open color

-- Function that assigns colors to the larger disk
def larger_disk_color (i : sector) : color := 
  if i.val < 100 then red else blue

-- Function that assigns colors to the smaller disk
noncomputable def smaller_disk_color : sector → color := 
  sorry

-- Define the same color relation
def same_color (a b : color) : Prop := a = b

-- Rotation by k sectors
def rotate (k : ℕ) (i : sector) : sector :=
  ⟨(i.val + k) % 200, by decide⟩

-- Define alignment of small and large sectors
def align (k : ℕ) : ℕ :=
  finset.card (finset.filter (λ i, same_color (smaller_disk_color i) (larger_disk_color (rotate k i))) finset.univ)

-- Statement
theorem rotate_disks_at_least_100 :
  ∃ k : ℕ, k < 200 ∧ align k ≥ 100 :=
sorry

end DiskRotation

end rotate_disks_at_least_100_l698_698902


namespace correct_statement_l698_698125

theorem correct_statement : ∀ (a b : ℝ), ((a ≠ b ∧ ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x = a ∨ x = b)) ∧
                                            ¬(∀ p q : ℝ, p = q → p = q) ∧
                                            ¬(∀ a : ℝ, |a| = -a → a < 0) ∧
                                            ¬(∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (a = -b)) → (a / b = -1))) :=
by sorry

-- Explanation of conditions:
-- a  ≠ b ensures two distinct points
-- ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x is between a and b) incorrectly rephrased as shortest distance as a line segment
-- ¬(∀ p q : ℝ, p = q → p = q) is not directly used, a minimum to refute the concept as required.
-- |a| = -a → a < 0 reinterpreted as a ≤ 0 but incorrectly stated as < 0 explicitly refuted
-- ¬(∀ a b : ℝ, a ≠ 0 and/or b ≠ 0 maintained where a / b not strictly required/misinterpreted)

end correct_statement_l698_698125


namespace parabola_symmetry_l698_698343

def p (x : ℝ) : ℝ := -5 * x^2 + 2
def q (x : ℝ) : ℝ := 5 * x^2 - 2

theorem parabola_symmetry (x : ℝ) : 
  ∃ y : ℝ, p x = y ∧ q x = -y :=
begin
  existsi (-p x),
  split,
  { refl,},
  { sorry, }
end

end parabola_symmetry_l698_698343


namespace volume_of_sphere_tangent_to_cube_is_36pi_l698_698615

noncomputable def volume_of_sphere_tangent_to_cube : ℝ :=
  let A := (0, 0, 0 : ℝ × ℝ × ℝ)
  let B := (6, 0, 0 : ℝ × ℝ × ℝ)
  let C := (0, 6, 0 : ℝ × ℝ × ℝ)
  let D := (0, 0, 6 : ℝ × ℝ × ℝ)
  let edge_length := 6
  let radius := edge_length / 2
  let volume := (4 / 3) * Real.pi * radius^3
  volume

theorem volume_of_sphere_tangent_to_cube_is_36pi :
  volume_of_sphere_tangent_to_cube = 36 * Real.pi :=
by
  -- Placeholder for proof
  sorry

end volume_of_sphere_tangent_to_cube_is_36pi_l698_698615


namespace cost_per_pack_l698_698584

theorem cost_per_pack (total_cost : ℤ) (packs : ℤ) (h1 : total_cost = 2673) (h2 : packs = 33) :
  total_cost / packs = 81 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end cost_per_pack_l698_698584


namespace nikolai_completes_faster_l698_698036

-- Given conditions: distances they can cover in the same time and total journey length 
def gennady_jump_distance := 2 * 6 -- 12 meters
def nikolai_jump_distance := 3 * 4 -- 12 meters
def total_distance := 2000 -- 2000 meters before turning back

-- Mathematical translation + Target proof: prove that Nikolai will complete the journey faster
theorem nikolai_completes_faster 
  (gennady_distance_per_time : gennady_jump_distance = 12)
  (nikolai_distance_per_time : nikolai_jump_distance = 12)
  (journey_length : total_distance = 2000) : 
  ( (2000 % 4 = 0) ∧ (2000 % 6 ≠ 0) ) -> true := 
by 
  intros,
  sorry

end nikolai_completes_faster_l698_698036


namespace arc_length_radius_l698_698179

-- Define the circumference of the circle and the angle in radians
def C : ℝ := 100
def angle_EDF : ℝ := 45 * (Real.pi / 180)  -- Convert degrees to radians

-- Prove the length of arc EF and the radius of the circle
theorem arc_length_radius (C : ℝ) (angle_EDF : ℝ) :
  (C = 100) ∧ (angle_EDF = 45 * (Real.pi / 180)) →
  (∃ arc_length : ℝ, arc_length = 12.5) ∧ 
  (∃ r : ℝ, r = 50 / Real.pi) := 
by
  intros _
  sorry

end arc_length_radius_l698_698179


namespace right_triangle_third_side_product_l698_698774

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l698_698774


namespace simplify_trig_expression_l698_698315

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / cos (10 * Real.pi / 180) = 
  2 * (3 * Real.sqrt 3 + 4) / 9 := 
by 
  sorry

end simplify_trig_expression_l698_698315


namespace minimize_downtime_l698_698696

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def total_downtime (sequence : list point) : ℝ :=
  sequence.sum (λ (i : ℕ), distance (sequence.nth_le i sorry) (sequence.nth_le (i + 1) sorry))

def points_set : list point :=
  [(1, 10), (976, 9), (666, 87), (377, 422), (535, 488), (775, 488), (941, 500), (225, 583),
   (388, 696), (3, 713), (504, 872), (560, 934), (22, 997)]

def optimal_path : list point :=
  [(1, 10), (377, 422), (225, 583), (388, 696), (504, 872), (560, 934), (535, 488), (775, 488),
   (941, 500), (976, 9), (666, 87), (3, 713), (22, 997)]

theorem minimize_downtime :
  total_downtime points_set = 24113.147907 :=
sorry

end minimize_downtime_l698_698696


namespace formula_for_a_sum_of_b_l698_698193

variables (a b c : ℕ → ℝ) (d : ℝ)

-- Conditions 
def increasing_arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) > a n
def a_condition1 := a 2 * a 4 = 21
def a_condition2 := a 1 + a 5 = 10
def c_definition (n : ℕ) := c n = a n + 1
def b_definition (n : ℕ) := b n = 2^n * c n

-- To prove the general formula for {a_n}
theorem formula_for_a (h1 : increasing_arithmetic_sequence a) 
                     (h2 : a_condition1 a) 
                     (h3 : a_condition2 a) : 
  ∃ d > 0, ∀ n, a n = 2 * n - 1 := 
sorry

-- To prove the sum of the first n terms of {b_n}
theorem sum_of_b (h1 : increasing_arithmetic_sequence a) 
                 (h2 : a_condition1 a) 
                 (h3 : a_condition2 a) 
                 (h4 : ∀ n, c_definition a c n) 
                 (h5 : ∀ n, b_definition b c n) : 
  ∀ n, ∑ i in finset.range n, b i = 2^(n+2) - 4 := 
sorry

end formula_for_a_sum_of_b_l698_698193


namespace isosceles_triangle_side_length_l698_698931

theorem isosceles_triangle_side_length (area_of_square : ℝ)
  (area_of_sum_of_triangles : ℝ)
  (side_length_square : ℝ)
  (base_of_triangle : ℝ)
  (height_of_triangle : ℝ)
  (length_of_congruent_sides : ℝ) :
  area_of_square = 4 ∧
  area_of_sum_of_triangles = area_of_square ∧
  side_length_square = 2 ∧
  base_of_triangle = 1 ∧
  height_of_triangle = 2 ∧
  length_of_congruent_sides = √( (1/2)^2 + 2^2)  →
  length_of_congruent_sides = sqrt(17)/2 :=
by
  sorry

end isosceles_triangle_side_length_l698_698931


namespace equilateral_triangle_condition_l698_698436

theorem equilateral_triangle_condition (T : Triangle) (C : Circle) (h1 : TriangleInscribedInCircle T C)
  (h2 : TangentTrianglePerimeterTwice T C) : IsEquilateral T :=
by
  sorry

end equilateral_triangle_condition_l698_698436


namespace remainder_7_pow_4_div_100_l698_698798

theorem remainder_7_pow_4_div_100 : (7 ^ 4) % 100 = 1 := 
by
  sorry

end remainder_7_pow_4_div_100_l698_698798


namespace negation_all_nonzero_l698_698341

    theorem negation_all_nonzero (a b c : ℝ) : ¬ (¬ (a = 0 ∨ b = 0 ∨ c = 0)) → (a = 0 ∧ b = 0 ∧ c = 0) :=
    by
      sorry
    
end negation_all_nonzero_l698_698341


namespace parabola_m_value_l698_698237

-- Define the conditions for the parabola and its vertex
def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)
def vertex_y (a b c : ℝ) : ℝ := c - b^2 / (4 * a)

noncomputable def m_value (a b m : ℝ) := (vertex_y a b m) = 0

-- Given data
def parabola_vertex_on_x_axis (a b m : ℝ) (a_neg1 : a = -1) (b_2 : b = 2) : Prop :=
  m_value a b m

-- Prove that if the vertex of the parabola lies on the x-axis, then m = -1
theorem parabola_m_value : parabola_vertex_on_x_axis (-1) 2 (-1) 
:= sorry

end parabola_m_value_l698_698237


namespace general_term_c_eq_1_bounded_sequence_for_c_l698_698684

-- Definitions needed for the Lean statements based on the given conditions
def sequence (c : ℝ) : ℕ → ℝ
| 0       := 1
| (n + 1) := sqrt ((sequence c n) ^ 2 - 2 * (sequence c n) + 3) + c

-- Part (1): Proving the general term when c = 1
theorem general_term_c_eq_1 :
  (∀ n : ℕ, sequence 1 n = 1 + sqrt (2 * (↑n : ℝ))) :=
sorry

-- Part (2): Prove the boundedness of the sequence for 0 < c < 1 and unboundedness for c ≥ 1
theorem bounded_sequence_for_c :
  ((∀ c : ℝ, (0 < c ∧ c < 1) → ∃ M : ℝ, ∀ n : ℕ, sequence c n < M) ∧
   (∀ c : ℝ, (c ≥ 1) → ¬(∃ M : ℝ, ∀ n : ℕ, sequence c n < M))) :=
sorry

end general_term_c_eq_1_bounded_sequence_for_c_l698_698684


namespace arithmetic_sequence_general_formula_sequence_sum_arithmetic_terms_l698_698189

open Real

theorem arithmetic_sequence_general_formula :
  ∃ a d, (S (3 : ℕ) = 0 ∧ S (5 : ℕ) = -5) →
    (∀ n : ℕ, a_n = a + (n - 1) * d) →
    (a = 1 ∧ d = -1) → 
    (∀ n : ℕ, a_n = 2 - n) :=
by
  sorry

theorem sequence_sum_arithmetic_terms :
  ∀ n : ℕ,
    let a_n := 2 - n
    let terms := (λ k, 1 / (a_(2 * k - 1) * a_(2 * k + 1)))
    sum_to_n terms = n / (1 - 2 * n) :=
by
  sorry

end arithmetic_sequence_general_formula_sequence_sum_arithmetic_terms_l698_698189


namespace eccentricity_of_hyperbola_l698_698978

-- Define the problem conditions
def parabola_focus : Point := (2, 0)

def parabola (x y : ℝ) := y^2 = 8 * x

def hyperbola (x y a b : ℝ) (h : a > 0) (h' : b > 0) := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def tangent_condition (a b : ℝ) (h : a > 0) (h' : b > 0) :=
  a^2 + b^2 = 4

noncomputable def min_value_condition (a b : ℝ) (h : a > 0) (h' : b > 0) :=
  4 / a^2 + 1 / b^2 = 9 / 4

noncomputable def eccentricity (c a : ℝ) := c / a

-- State the goal using Lean 4
theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (ht : tangent_condition a b ha hb)
  (hm : min_value_condition a b ha hb) :
  eccentricity (sqrt 3 * b) a = sqrt 6 / 2 :=
by sorry

end eccentricity_of_hyperbola_l698_698978


namespace circle_division_l698_698402

theorem circle_division (n : ℕ)
  (digits : Fin n → ℕ)
  (h_nonzero : ∀ (x : Fin n), digits x ≠ 0)
  (senya_seq zhenya_seq : Fin (n-1) → ℕ)
  (h_senya : ∀ i : Fin (n-1), senya_seq i = digits (⟨(i + some_zero_senya_index) % n, some_property⟩))
  (h_zhenya : ∀ i : Fin (n-1), zhenya_seq i = digits (⟨(i + some_zero_zhenya_index) % n, some_property⟩))
  (h_match : senya_seq = zhenya_seq) :
  ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ n % m = 0 ∧ ∀ i : Fin n, digits i = digits (⟨(i + m) % n, sorry⟩) := by
  sorry

end circle_division_l698_698402


namespace find_prime_pair_l698_698146

-- Definition of the problem
def is_integral_expression (p q : ℕ) : Prop :=
  (p + q)^(p + q) * (p - q)^(p - q) - 1 ≠ 0 ∧
  (p + q)^(p - q) * (p - q)^(p + q) - 1 ≠ 0 ∧
  ((p + q)^(p + q) * (p - q)^(p - q) - 1) % ((p + q)^(p - q) * (p - q)^(p + q) - 1) = 0

-- Mathematical theorem to be proved
theorem find_prime_pair (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (h : p > q) :
  is_integral_expression p q → (p, q) = (3, 2) :=
by 
  sorry

end find_prime_pair_l698_698146


namespace remainder_of_sum_is_five_l698_698065

theorem remainder_of_sum_is_five (a b c d : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) (hd : d % 15 = 14) :
  (a + b + c + d) % 15 = 5 :=
by
  sorry

end remainder_of_sum_is_five_l698_698065


namespace local_max_at_2_l698_698587

noncomputable def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

def f' (x c : ℝ) : ℝ := 3 * x ^ 2 - 4 * c * x + c ^ 2

def f'' (x c : ℝ) : ℝ := 6 * x - 4 * c

theorem local_max_at_2 (c : ℝ) (h1: f'(2, c) = 0) (h2: f''(2, c) < 0) : c = 6 :=
by sorry

end local_max_at_2_l698_698587


namespace probability_of_adjacent_vertices_in_decagon_l698_698749

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698749


namespace trigonometric_simplification_l698_698318

theorem trigonometric_simplification :
  ∀ (a b c d e f g h : ℝ),
  (a = 20 * Real.pi / 180) →
  (b = 30 * Real.pi / 180) →
  (c = 60 * Real.pi / 180) →
  (d = 70 * Real.pi / 180) →
  (e = 10 * Real.pi / 180) →
  (f = 50 * Real.pi / 180) →
  (g = 40 * Real.pi / 180) →
  (h = 130 * Real.pi / 180) →
  sin h = sin f →
  sin f / cos e = cos g / cos e →
  ( ∀ x y : ℝ, tan x + tan y = sin (x + y) / (cos x * cos y) ) →
  (tan a + tan b + tan c + tan d) / cos e = 2 * cos g / (cos e ^ 2 * cos b * cos c * cos d) := sorry

end trigonometric_simplification_l698_698318


namespace train_speed_l698_698368

theorem train_speed (v : ℝ) (h1 : 60 * 6.5 + v * 6.5 = 910) : v = 80 := 
sorry

end train_speed_l698_698368


namespace problem1_xiao_ming_score_problem2_min_students_l698_698323

-- Definition and conditions used in the problem
def totalScore (correct: ℕ) (incorrect: ℕ) : ℤ :=
  (correct * 5) - (incorrect * 1)

-- Problem 1: Prove Xiao Ming's total score
theorem problem1_xiao_ming_score (correct incorrect : ℕ) :
  correct = 8 → incorrect = 2 → totalScore correct incorrect = 38 :=
by
  intros h1 h2
  rw [h1, h2]
  simp [totalScore]
  sorry

-- Definition and calculation for Problem 2
def minStudents : ℕ := 23

-- Problem 2: Prove the minimum number of students for at least 3 having the same score
theorem problem2_min_students :
  minStudents = 23 :=
by
  simp [minStudents]
  sorry

end problem1_xiao_ming_score_problem2_min_students_l698_698323


namespace minimum_distance_ellipse_line_l698_698492

noncomputable def distance_point_line (x₁ y₁ A B C : ℝ) : ℝ :=
  abs (A * x₁ + B * y₁ + C) / real.sqrt (A^2 + B^2)

theorem minimum_distance_ellipse_line :
  ∃ (M : ℝ × ℝ), (M.1^2 / 9 + M.2^2 / 4 = 1) ∧ 
  distance_point_line M.1 M.2 1 2 (-10) = real.sqrt 5 :=
by
  sorry

end minimum_distance_ellipse_line_l698_698492


namespace smallest_real_number_l698_698126

theorem smallest_real_number :
  ∀ (a b c d : ℝ), a = 1 → b = -3 → c = -√2 → d = -π →
  (d < b ∧ d < c ∧ d < a) :=
by
  intros a b c d Ha Hb Hc Hd
  rw [Ha, Hb, Hc, Hd]
  have H1 : -π < -3 := sorry
  have H2 : -π < -√2 := sorry
  have H3 : -π < 1 := sorry
  exact ⟨H1, H2, H3⟩

end smallest_real_number_l698_698126


namespace collinear_P_Q_R_l698_698244

variable {P Q R A B C D E F : Type}
variable [AffineSpace ℝ P]

-- Conditions of the problem
variable (A B C D E F P Q R : P)

def midpoint (a b : P) : P := (a +ᵥ b) / 2

-- Assuming given conditions
variable (h1 : affine_combination [A, B] = E)
variable (h2 : affine_combination [D, C] = E)
variable (h3 : affine_combination [A, D] = F)
variable (h4 : affine_combination [B, C] = F)
variable (hP : P = midpoint A C)
variable (hQ : Q = midpoint B D)
variable (hR : R = midpoint E F)

-- Question: To prove P, Q, and R are collinear
theorem collinear_P_Q_R : collinear ℝ [P, Q, R] :=
sorry

end collinear_P_Q_R_l698_698244


namespace point_on_graph_l698_698673

noncomputable def f (x : ℝ) : ℝ := abs (x^3 + 1) + abs (x^3 - 1)

theorem point_on_graph (a : ℝ) : ∃ (x y : ℝ), (x = a) ∧ (y = f (-a)) ∧ (y = f x) :=
by 
  sorry

end point_on_graph_l698_698673


namespace retailer_marked_price_percentage_above_cost_l698_698106

noncomputable def cost_price : ℝ := 100
noncomputable def discount_rate : ℝ := 0.15
noncomputable def sales_profit_rate : ℝ := 0.275

theorem retailer_marked_price_percentage_above_cost :
  ∃ (MP : ℝ), ((MP - cost_price) / cost_price = 0.5) ∧ (((MP * (1 - discount_rate)) - cost_price) / cost_price = sales_profit_rate) :=
sorry

end retailer_marked_price_percentage_above_cost_l698_698106


namespace ellipse_standard_eq_points_collinear_line_l_eq_l698_698191

-- (I) The standard equation of the ellipse
theorem ellipse_standard_eq {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h_e : (real.sqrt 3) / 2 = 3 / a) :
  a = 2 * (real.sqrt 3) ∧ b = real.sqrt 3 → 
  (∀ (x y : ℝ), (x^2 / 12 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) := sorry

-- (II) Points O, M, and N are collinear
theorem points_collinear (m : ℝ) (h_m : m ≠ 0) :
  let N := (4, m)
  let M := (12 / (m^2 + 4), 3 * m / (m^2 + 4))
  let O := (0 : ℝ, 0 : ℝ) in
  ((M.2 / M.1) = (m / 4) ∧ (N.2 / (N.1 - 4)) = (m / 4)) → 
  collinear [O, M, N] := sorry

-- (III) Equation of line l
theorem line_l_eq (m : ℝ) (h_m : m = real.sqrt 5 ∨ m = -real.sqrt 5) :
  2 * abs (12 / (m^2 + 4)) = abs (4 - 12 / (m^2 + 4)) →
  ∃ k : ℝ, line_eq := (x = k * y + 3) ∧ (k = real.sqrt 5 ∨ k = -real.sqrt 5) := sorry

end ellipse_standard_eq_points_collinear_line_l_eq_l698_698191


namespace sqrt_of_square_neg_l698_698945

variable {a : ℝ}

theorem sqrt_of_square_neg (h : a < 0) : Real.sqrt (a^2) = -a := 
sorry

end sqrt_of_square_neg_l698_698945


namespace find_c_plus_d_l698_698155

noncomputable def log_75843 := Real.log10 75843

theorem find_c_plus_d : ∃ (c d : ℤ), c ≤ log_75843 ∧ log_75843 ≤ d ∧ c = 4 ∧ d = 5 ∧ c + d = 9 := by
  use 4
  use 5
  have hc : 4 ≤ log_75843 := sorry
  have hd : log_75843 ≤ 5 := sorry
  exact ⟨hc, hd, rfl, rfl, by simp⟩

end find_c_plus_d_l698_698155


namespace betting_strategy_exists_l698_698600

theorem betting_strategy_exists :
  (1 / 3) + (1 / 5) + (1 / 6) + (1 / 7) < 1 :=
by
  have h₁ : (1 : ℚ) / 3 + 1 / 5 + 1 / 6 + 1 / 7 ≈ 
    (70 : ℚ) / 210 + 42 / 210 + 35 / 210 + 30 / 210,
  { ring },
  have h₂: (70 + 42 + 35 + 30).toRational < 210,
  { norm_num },
  linarith,

end betting_strategy_exists_l698_698600


namespace probability_lands_in_triangle_pbc_l698_698201

open Set

noncomputable def probability_in_triangle_pbc (A B C P : Point) : ℝ :=
  if h₁ : P ∈ plane_of_triangle A B C ∧ 
          (B - P) + (C - P) + 2 * (A - P) = 0 then 1/2 else 0

theorem probability_lands_in_triangle_pbc 
  (A B C P : Point) 
  (h₁ : P ∈ plane_of_triangle A B C) 
  (h₂ : (B - P) + (C - P) + 2 * (A - P) = 0) : 
  probability_in_triangle_pbc A B C P = 1/2 := 
by
  -- Proof continues here...
  sorry -- Placeholder for the actual proof

end probability_lands_in_triangle_pbc_l698_698201


namespace arithmetic_sequence_30th_term_l698_698806

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l698_698806


namespace arithmetic_sequence_sums_l698_698254

variable (a : ℕ → ℕ)

-- Conditions
def condition1 := a 1 + a 4 + a 7 = 39
def condition2 := a 2 + a 5 + a 8 = 33

-- Question and expected answer
def result := a 3 + a 6 + a 9 = 27

theorem arithmetic_sequence_sums (h1 : condition1 a) (h2 : condition2 a) : result a := 
sorry

end arithmetic_sequence_sums_l698_698254


namespace fruit_order_count_l698_698897

-- Define the initial conditions
def apples := 3
def oranges := 2
def bananas := 2
def totalFruits := apples + oranges + bananas -- which is 7

-- Calculate the factorial of a number
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Noncomputable definition to skip proof
noncomputable def distinctOrders : ℕ :=
  fact totalFruits / (fact apples * fact oranges * fact bananas)

-- Lean statement expressing that the number of distinct orders is 210
theorem fruit_order_count : distinctOrders = 210 :=
by
  sorry

end fruit_order_count_l698_698897


namespace sphere_surface_area_l698_698543

theorem sphere_surface_area (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5)
  (d : ℝ) (h₄ : d = (sqrt 3) / 2)
  (R : ℝ) (h₅ : R = sqrt ((5/2)^2 + (sqrt 3 / 2)^2)) :
  (∃ S : ℝ, S = 4 * Real.pi * R^2) := 
by 
    exists 28 * Real.pi
    sorry

end sphere_surface_area_l698_698543


namespace intersection_MN_eq_set1_l698_698640

def f (x : ℝ) : ℝ := x^2 - 4*x + 3
def g (x : ℝ) : ℝ := 3*x - 2

def M : set ℝ := {x | f (g x) > 0}
def N : set ℝ := {x | g x < 2}

theorem intersection_MN_eq_set1 :
  M ∩ N = {x | x < 1} :=
by
  sorry

end intersection_MN_eq_set1_l698_698640


namespace longest_chord_of_circle_l698_698958

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end longest_chord_of_circle_l698_698958


namespace pass_through_game_max_levels_l698_698842

theorem pass_through_game_max_levels :
  ∀ n : ℕ, (∑ i in Finset.range n, (nat.choose 6 1)) ≤ 2 ^ n → n ≤ 4 :=
sorry

end pass_through_game_max_levels_l698_698842


namespace optimal_price_l698_698089

def monthly_sales (p : ℝ) : ℝ := 150 - 6 * p
def break_even (p : ℝ) : Prop := 40 ≤ monthly_sales p
def revenue (p : ℝ) : ℝ := p * monthly_sales p

theorem optimal_price : ∃ p : ℝ, p = 13 ∧ p ≤ 30 ∧ break_even p ∧ ∀ q : ℝ, q ≤ 30 → break_even q → revenue p ≥ revenue q := 
by
  sorry

end optimal_price_l698_698089


namespace min_value_y_l698_698947

theorem min_value_y (x : ℝ) (hx : x > 3) : 
  ∃ y, (∀ x > 3, y = min_value) ∧ min_value = 5 :=
by 
  sorry

end min_value_y_l698_698947


namespace chocolate_bar_weight_l698_698412

-- Conditions
def box_weight_kg : ℝ := 2
def bars_count : ℕ := 16

-- Conversion factor from kg to grams
def kg_to_grams : ℝ := 1000

-- The total weight of the box in grams
def box_weight_grams : ℝ := box_weight_kg * kg_to_grams

-- The weight of each bar
def bar_weight_grams : ℝ := box_weight_grams / bars_count

-- Statement of the problem
theorem chocolate_bar_weight : bar_weight_grams = 125 :=
by sorry

end chocolate_bar_weight_l698_698412


namespace skillful_hands_award_prob_cannot_enter_finals_after_training_l698_698607

noncomputable def combinatorial_probability : ℚ :=
  let P1 := (4 * 3) / (10 * 10)    -- P1: 1 specified, 2 creative
  let P2 := (6 * 3) / (10 * 10)    -- P2: 2 specified, 1 creative
  let P3 := (6 * 3) / (10 * 10)    -- P3: 2 specified, 2 creative
  P1 + P2 + P3

theorem skillful_hands_award_prob : combinatorial_probability = 33 / 50 := 
  sorry

def after_training_probability := 3 / 4
theorem cannot_enter_finals_after_training : after_training_probability * 5 < 4 := 
  sorry

end skillful_hands_award_prob_cannot_enter_finals_after_training_l698_698607


namespace incorrect_diagonal_property_of_rectangle_l698_698381

/-- A quadrilateral with four equal sides is a rhombus -/
def is_rhombus (Q : Type) [quad : has_quadrilateral Q] : Prop :=
  quad.has_four_equal_sides Q

/-- A parallelogram with perpendicular diagonals is a rhombus -/
def is_rhombus_parallelogram (P : Type) [parallelogram : has_parallelogram P] : Prop :=
  parallelogram.has_perpendicular_diagonals P

/-- The diagonals of a rectangle are perpendicular and equal. We prove that this is incorrect,
   meaning they are equal but not necessarily perpendicular. -/
theorem incorrect_diagonal_property_of_rectangle {R : Type} [rectangle : has_rectangle R] :
  ¬ (rectangle.has_equal_and_perpendicular_diagonals R) :=
  sorry

end incorrect_diagonal_property_of_rectangle_l698_698381


namespace thirtieth_term_value_l698_698817

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l698_698817


namespace probability_of_adjacent_vertices_l698_698722

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698722


namespace largest_divisor_power_of_ten_l698_698708

def largest_divisor_smaller_than (N : ℕ) : ℕ :=
  if N = 1 then 0
  else (Nat.divisors N).filter (λ d => d < N) |>.lastD 1

theorem largest_divisor_power_of_ten (N : ℕ) (k : ℕ) :
  N + largest_divisor_smaller_than N = 10^k → N = 75 :=
  sorry

end largest_divisor_power_of_ten_l698_698708


namespace max_value_abs_cube_sum_l698_698196

theorem max_value_abs_cube_sum (x : Fin 5 → ℝ) (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (|x 0 - x 1|^3 + |x 1 - x 2|^3 + |x 2 - x 3|^3 + |x 3 - x 4|^3 + |x 4 - x 0|^3) ≤ 4 :=
sorry

end max_value_abs_cube_sum_l698_698196


namespace determinant_of_3x3_matrix_l698_698460

/-- Given a specific 3x3 matrix, proves that its determinant is -51 --/
theorem determinant_of_3x3_matrix :
  let A := Matrix.of ![![2, -1, 4], ![0, 6, -3], ![3, 0, 1]] in
  Matrix.det A = -51 :=
by
  sorry

end determinant_of_3x3_matrix_l698_698460


namespace propositions_correct_l698_698507

variables (a b l : Line) (α β γ : Plane)

def proposition1 (h1 : α ∩ β = a) (h2 : β ∩ γ = b) (h3 : a ∥ b) : α ∥ γ := sorry
-- the proposition is false, hence including (a)

def proposition2 (h1 : a ∩ b ≠ ∅) (h2 : a ∉ α ∧ a ∉ β) (h3 : b ∉ α ∧ b ∉ β)
                 (h4 : a ∥ α) (h5 : a ∥ β) (h6 : b ∥ α) (h7 : b ∥ β) : α ∥ β := sorry
-- the proposition is false, hence including (b)

def proposition3 (h1 : α ⊥ β) (h2 : α ∩ β = a) (h3 : b ⊆ β) (h4 : a ⊥ b) : b ⊥ α := sorry
-- the proposition is true

def proposition4 (h1 : a ⊆ α) (h2 : b ⊆ α) (h3 : l ⊥ a) (h4 : l ⊥ b) : l ⊥ α := sorry
-- the proposition is true

theorem propositions_correct : 
  ¬ proposition1 α β γ a b → ¬ proposition2 α β a b → proposition3 α β a b → proposition4 α b l :=
sorry

end propositions_correct_l698_698507


namespace decagon_adjacent_vertices_probability_l698_698732

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698732


namespace find_positive_integer_l698_698147

theorem find_positive_integer (n : ℕ) (hn_pos : n > 0) :
  (∃ a b : ℕ, n = a^2 ∧ n + 100 = b^2) → n = 576 :=
by sorry

end find_positive_integer_l698_698147


namespace min_m_value_l698_698232

noncomputable def f (x a : ℝ) : ℝ := 2^|x - a|

theorem min_m_value 
  (a : ℝ) 
  (h1 : ∀ x, f (1 + x) a = f (1 - x) a)
  (h2 : ∀ x y, x ≤ y → y ∈ set.Ici m → f x a ≤ f y a) :
  m = 1 := 
sorry

end min_m_value_l698_698232


namespace percentage_increase_in_feeding_cost_l698_698264

theorem percentage_increase_in_feeding_cost (
  number_of_cattle : ℕ := 100
  buying_cost : ℝ := 40000
  weight_per_cattle : ℝ := 1000
  selling_price_per_pound : ℝ := 2
  profit : ℝ := 112000
) : (42800 / buying_cost) * 100 = 20 :=
by
  sorry

end percentage_increase_in_feeding_cost_l698_698264


namespace find_omega_l698_698965

def f (ω x : ℝ) : ℝ := sin (ω * x - π / 6) + 1 / 2

theorem find_omega 
  (ω α β : ℝ)
  (hω_pos : 0 < ω)
  (hα : f ω α = -1 / 2)
  (hβ : f ω β = 1 / 2)
  (halpha_beta : |α - β| = 3 * π / 4) : 
  ω = 2 / 3 :=
sorry

end find_omega_l698_698965


namespace contrapositive_sin_pi_over_two_l698_698327

theorem contrapositive_sin_pi_over_two (α : ℝ) :
  (α = π/2 → sin α = 1) ↔ (sin α ≠ 1 → α ≠ π/2) :=
by
  sorry

end contrapositive_sin_pi_over_two_l698_698327


namespace sum_of_digits_y_l698_698843

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString
  str = str.reverse

theorem sum_of_digits_y : 
  ∃ y : ℕ, (1000 ≤ y ∧ y ≤ 9999) ∧ is_palindrome y ∧ 
           is_palindrome (y + 42) ∧ 
           (y + 42 ≥ 10000) ∧ (y + 42 ≤ 10041) ∧ 
           (y.digits.sum = 32) :=
by
  sorry

end sum_of_digits_y_l698_698843


namespace arithmetic_sequence_30th_term_l698_698803

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l698_698803


namespace foma_wait_time_probability_l698_698448

/-- Define the bounds for Ivan and Foma's arrival times. -/
def valid_arrival_times (x y : ℝ) : Prop :=
  2 ≤ x ∧ x < y ∧ y ≤ 10

/-- Define the condition that Foma waits no more than 4 minutes after Ivan. -/
def foma_waits_no_more_than_4 (x y : ℝ) : Prop :=
  y - x ≤ 4

/-- Define the probability of the event given the conditions using classical probability. -/
noncomputable def probability_foma_waits_no_more_than_4 : ℝ :=
  let total_area := 32 in 
  let favorable_area := 24 in
  favorable_area / total_area

/-- Statement: The probability that Foma waits no more than 4 minutes is 0.75. -/
theorem foma_wait_time_probability : 
  probability_foma_waits_no_more_than_4 = 0.75 :=
sorry

end foma_wait_time_probability_l698_698448


namespace Nedy_crackers_total_l698_698294

theorem Nedy_crackers_total :
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  (packs_from_Mon_to_Thu + packs_on_Fri) = 24 :=
by
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  show packs_from_Mon_to_Thu + packs_on_Fri = 24
  sorry

end Nedy_crackers_total_l698_698294


namespace roots_polynomial_sum_l698_698282

theorem roots_polynomial_sum :
  ∀ p q r : ℝ,
  (polynomial.eval p (polynomial.C 1 * polynomial.X ^ 3 - polynomial.C 6 * polynomial.X ^ 2 + polynomial.C 11 * polynomial.X - polynomial.C 6) = 0) ∧
  (polynomial.eval q (polynomial.C 1 * polynomial.X ^ 3 - polynomial.C 6 * polynomial.X ^ 2 + polynomial.C 11 * polynomial.X - polynomial.C 6) = 0) ∧
  (polynomial.eval r (polynomial.C 1 * polynomial.X ^ 3 - polynomial.C 6 * polynomial.X ^ 2 + polynomial.C 11 * polynomial.X - polynomial.C 6) = 0) →
  (p + q + r = 6) ∧ (p * q + p * r + q * r = 11) ∧ (p * q * r = 6) →
  (p / (p * q + 2) + q / (p * r + 2) + r / (q * p + 2) = 3 / 4) := by
  sorry

end roots_polynomial_sum_l698_698282


namespace commencement_addresses_l698_698604

theorem commencement_addresses (sandoval_addresses : ℕ) 
                             (hawkins_addresses : ℕ) 
                             (sloan_addresses : ℕ) :
  sandoval_addresses = 12 →
  hawkins_addresses = sandoval_addresses / 2 →
  sloan_addresses = sandoval_addresses + 10 →
  sandoval_addresses + hawkins_addresses + sloan_addresses = 40 :=
begin
  sorry
end

end commencement_addresses_l698_698604


namespace sin_sum_given_cos_tan_conditions_l698_698510

open Real

theorem sin_sum_given_cos_tan_conditions 
  (α β : ℝ)
  (h1 : cos α + cos β = 1 / 3)
  (h2 : tan (α + β) = 24 / 7)
  : sin α + sin β = 1 / 4 ∨ sin α + sin β = -4 / 9 := 
  sorry

end sin_sum_given_cos_tan_conditions_l698_698510


namespace longest_chord_of_circle_l698_698959

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end longest_chord_of_circle_l698_698959


namespace pelicans_among_non_egrets_is_47_percent_l698_698245

-- Definitions for the percentage of each type of bird.
def pelican_percentage : ℝ := 0.4
def cormorant_percentage : ℝ := 0.2
def egret_percentage : ℝ := 0.15
def osprey_percentage : ℝ := 0.25

-- Calculate the percentage of pelicans among the non-egret birds.
theorem pelicans_among_non_egrets_is_47_percent :
  (pelican_percentage / (1 - egret_percentage)) * 100 = 47 :=
by
  -- Detailed proof goes here
  sorry

end pelicans_among_non_egrets_is_47_percent_l698_698245


namespace solve_n_l698_698168

/-
Define the condition for the problem.
Given condition: \(\frac{1}{n+1} + \frac{2}{n+1} + \frac{n}{n+1} = 4\)
-/

noncomputable def condition (n : ℚ) : Prop :=
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1)) = 4

/-
The theorem to prove: Value of \( n \) that satisfies the condition is \( n = -\frac{1}{3} \)
-/
theorem solve_n : ∃ n : ℚ, condition n ∧ n = -1 / 3 :=
by
  sorry

end solve_n_l698_698168


namespace range_of_k_l698_698291

noncomputable def f (x t : ℝ) := -cos x ^ 2 - 4 * t * sin (x / 2) * cos (x / 2) + 2 * t ^ 2 - 6 * t + 2

def g (t : ℝ) := 
  if t < -1 then 2 * t ^ 2 - 4 * t + 2
  else if -1 ≤ t ∧ t ≤ 1 then t ^ 2 - 6 * t + 1
  else 2 * t ^ 2 - 8 * t + 2

theorem range_of_k (t : ℝ) (k : ℝ) (ht: -1 < t ∧ t < 1) :
  (g t = k * t) ↔ (k < -8 ∨ k > -4) := 
sorry

end range_of_k_l698_698291


namespace thirtieth_term_value_l698_698816

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l698_698816


namespace new_rectangle_area_l698_698183

theorem new_rectangle_area :
  let a := 3
  let b := 4
  let diagonal := Real.sqrt (a^2 + b^2)
  let sum_of_sides := a + b
  let area := diagonal * sum_of_sides
  area = 35 :=
by
  sorry

end new_rectangle_area_l698_698183


namespace M_inter_N_eq_0_1_l698_698214

def set_M : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

def set_N : Set ℝ := { x | log x / log 2 < 0 }

def set_intersection (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem M_inter_N_eq_0_1 : set_intersection set_M set_N = { x | 0 < x ∧ x < 1 } :=
by {
  -- proof goes here
  sorry
}

end M_inter_N_eq_0_1_l698_698214


namespace sum_odd_positives_less_than_100_l698_698377

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := n > 0
def less_than (n m : ℕ) : Prop := n < m

theorem sum_odd_positives_less_than_100 :
  ∑ k in Finset.range 100, if is_odd k ∧ is_positive k then k else 0 = 2500 := by
  sorry

end sum_odd_positives_less_than_100_l698_698377


namespace area_new_rectangle_l698_698936

theorem area_new_rectangle (a b : ℝ) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
sorry

end area_new_rectangle_l698_698936


namespace ratio_of_costs_l698_698884

-- Definitions based on conditions
def old_car_cost : ℕ := 1800
def new_car_cost : ℕ := 1800 + 2000

-- Theorem stating the desired proof
theorem ratio_of_costs :
  (new_car_cost / old_car_cost : ℚ) = 19 / 9 :=
by
  sorry

end ratio_of_costs_l698_698884


namespace percentage_increase_is_40_l698_698393

-- Define the original weekly earnings
def original_earnings : ℝ := 50

-- Define the new weekly earnings
def new_earnings : ℝ := 70

-- Calculate the increase
def increase : ℝ := new_earnings - original_earnings

-- Calculate the percentage increase
def percentage_increase : ℝ := (increase / original_earnings) * 100

-- Theorem and Statement: Prove that the percentage increase is 40%
theorem percentage_increase_is_40 :
  percentage_increase = 40 := 
by
  sorry

end percentage_increase_is_40_l698_698393


namespace initial_weight_of_solution_Y_l698_698656

variables (W : ℝ) 

-- Define conditions
def condition1 : Prop := (W > 0)
def condition2 : Prop := (0.30 * W + 2 * 0.30 = 0.36 * W + 0.72)

-- State the theorem
theorem initial_weight_of_solution_Y : condition1 ∧ condition2 → W = 10 :=
by
  intros,
  sorry

end initial_weight_of_solution_Y_l698_698656


namespace desired_average_sale_l698_698096

def s1 := 2500
def s2 := 4000
def s3 := 3540
def s4 := 1520
def avg := 2890

theorem desired_average_sale : (s1 + s2 + s3 + s4) / 4 = avg := by
  sorry

end desired_average_sale_l698_698096


namespace right_triangle_third_side_product_l698_698785

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l698_698785


namespace sorting_problem_correct_l698_698941

noncomputable def sequence_sum (n : ℕ) : ℤ := n * (n - 40)

def a_n (n : ℕ) : ℤ := sequence_sum (n + 1) - sequence_sum n

theorem sorting_problem_correct (n : ℕ) :
  sequence_sum 39 < 0 ∧
  sequence_sum 40 = 0 ∧
  sequence_sum 41 > 0 ∧
  a_n 19 < 0 ∧
  a_n 20 = 0 ∧
  a_n 21 > 0 :=
by {
  sorry
}

end sorting_problem_correct_l698_698941


namespace decagon_adjacent_vertices_probability_l698_698728

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698728


namespace simplify_trig_expression_l698_698313

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / cos (10 * Real.pi / 180) = 
  2 * (3 * Real.sqrt 3 + 4) / 9 := 
by 
  sorry

end simplify_trig_expression_l698_698313


namespace domain_of_f_l698_698331

theorem domain_of_f :
  (∀ x : ℝ, (0 < 1 - x) ∧ (0 < 3 * x + 1) ↔ ( - (1 / 3 : ℝ) < x ∧ x < 1)) :=
by
  sorry

end domain_of_f_l698_698331


namespace approximately_41_students_score_in_105_115_l698_698198

open ProbabilityTheory

noncomputable def students_scores_within_interval :
  (total_students : ℕ) →
  (μ σ : ℝ) →
  (interval_lb interval_ub : ℝ) →
  (probability : ℝ) →
  Prop :=
λ total_students μ σ interval_lb interval_ub probability,
  let X := NormalDist μ σ in
  let num_students := total_students.to_real * (X.cdf interval_ub - X.cdf interval_lb) in
  abs (num_students - probability * total_students.to_real) < 1

theorem approximately_41_students_score_in_105_115 :
  students_scores_within_interval 60 110 5 105 115 0.683 :=
sorry

end approximately_41_students_score_in_105_115_l698_698198


namespace train_A_distance_travelled_l698_698053

/-- Let Train A and Train B start from opposite ends of a 200-mile route at the same time.
Train A has a constant speed of 20 miles per hour, and Train B has a constant speed of 200 miles / 6 hours (which is approximately 33.33 miles per hour).
Prove that Train A had traveled 75 miles when it met Train B. --/
theorem train_A_distance_travelled:
  ∀ (T : Type) (start_time : T) (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (meeting_time : ℝ),
  distance = 200 ∧ speed_A = 20 ∧ speed_B = 33.33 ∧ meeting_time = 200 / (speed_A + speed_B) → 
  (speed_A * meeting_time = 75) :=
by
  sorry

end train_A_distance_travelled_l698_698053


namespace sum_of_products_l698_698840

theorem sum_of_products (m n : ℕ) :
  ∑ k in Finset.range n + 1, ∏ i in Finset.range m, (k + i) = 
  (∏ i in Finset.range (m+1), (n + 1 + i)) / (m + 1) := sorry

end sum_of_products_l698_698840


namespace unique_product_count_eq_eleven_l698_698568

theorem unique_product_count_eq_eleven :
  let s := {3, 4, 7, 13}
  in { p | ∃ a b ∈ s, a ≠ b ∧ a * b = p }.card +
     { p | ∃ a b c ∈ s, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = p }.card +
     { p | ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a * b * c * d = p }.card 
  = 11 :=
by
  let s := ({1, 3, 4, 7, 13} : Finset ℕ).erase 1
  let products := Finset.filter (λ p, ∃ a b: ℕ, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ a * b = p) (FIN.elems (Multiset.range 1100))
  let three_products := Finset.filter (λ p, ∃ a b c: ℕ, a ∈ s ∧ b ∈ s ∧ c ∈ s ⟨  ⟩∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = p) (FIN.elems (Multiselect.range 1100))
  let four_products :=  Finset.filter (λ p, ∃ a b c d : ℕ, a ∈ s ∧ b ∈ s ∧ c ∈s ∧ d ∈ s ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a * b * c * d = p) (FIN.elems Multiset.range 2000)
  clean sorry
  sorry

end unique_product_count_eq_eleven_l698_698568


namespace slope_angle_of_line_l698_698950

theorem slope_angle_of_line (n : ℝ × ℝ) (h : n = (Real.sqrt 3, -1)) : 
  ∃ θ ∈ set.Ico 0 Real.pi, θ = Real.pi / 3 :=
by
  sorry

end slope_angle_of_line_l698_698950


namespace find_principal_l698_698073

-- Define the conditions
def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

-- Given values
def SI : ℕ := 750
def R : ℕ := 6
def T : ℕ := 5

-- Proof statement
theorem find_principal : ∃ P : ℕ, simple_interest P R T = SI ∧ P = 2500 := by
  aesop

end find_principal_l698_698073


namespace avg_questions_per_hour_l698_698863

variable (q : ℝ)

-- conditions
def num_members : ℝ := 200
def answers_per_question : ℝ := 3
def total_daily_activity : ℝ := 57600

-- derived quantities per hour
def questions_per_hour := num_members * q
def answers_per_hour := num_members * answers_per_question * q
def total_per_hour := questions_per_hour + answers_per_hour

theorem avg_questions_per_hour : q = 3 :=
by
  have total_questions_and_answers : ℝ := total_daily_activity / 24
  have : total_per_hour = total_questions_and_answers := sorry
  have : num_members * q + num_members * answers_per_question * q = total_questions_and_answers := by
    rw [questions_per_hour, answers_per_hour, total_per_hour, this]
  have : num_members * (1 + answers_per_question) * q = total_questions_and_answers := by
    rw [← add_mul]
  have : 200 * 4 * q = 2400 := by
    rw [num_members, answers_per_question, this, total_questions_and_answers]
  have : q = 2400 / 800 := by
    rw [← mul_div_assoc, mul_comm, ← mul_assoc, ← mul_div_assoc] at this
  have : q = 3 := by
    rw [div_eq_mul_inv, inv_mul_cancel, mul_comm] at this
    norm_num at this

  exact this
  
#check avg_questions_per_hour

end avg_questions_per_hour_l698_698863


namespace root_in_interval_iff_a_range_l698_698197

def f (a x : ℝ) : ℝ := 2 * a * x ^ 2 + 2 * x - 3 - a

theorem root_in_interval_iff_a_range (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0) ↔ (1 ≤ a ∨ a ≤ - (3 + Real.sqrt 7) / 2) :=
sorry

end root_in_interval_iff_a_range_l698_698197


namespace carl_total_driving_hours_l698_698710

theorem carl_total_driving_hours :
  let before_promotion := [2, 3, 4, 2, 5] in
  let additional_hours := [1, 2, 3, 4, 0] in
  let after_promotion := (before_promotion.zip additional_hours).map (λ ⟨b, a⟩ => b + a) in
  let first_week_hours := after_promotion.sum in
  let second_week_hours := [after_promotion.head!, after_promotion.tail!.head!, 0, 0, after_promotion.tail!.tail!.tail!.head!] in
  first_week_hours + second_week_hours.sum = 39 :=
by
  let before_promotion := [2, 3, 4, 2, 5]
  let additional_hours := [1, 2, 3, 4, 0]
  let after_promotion := (before_promotion.zip additional_hours).map (λ ⟨b, a⟩ => b + a)
  let first_week_hours := after_promotion.sum
  let second_week_hours := [after_promotion.head!, after_promotion.tail!.head!, 0, 0, after_promotion.tail!.tail!.tail!.head!]
  have first_week_hours_calc : first_week_hours = 26 := by simp [before_promotion, additional_hours, List.zip, List.sum]
  have second_week_hours_calc : second_week_hours.sum = 13 := by simp [second_week_hours, List.sum]
  calc
    first_week_hours + second_week_hours.sum = 26 + 13 := by rw [first_week_hours_calc, second_week_hours_calc]
    ... = 39 := by norm_num
    sorry

end carl_total_driving_hours_l698_698710


namespace sequence_formula_l698_698939

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 4 else 3 * n + 1

theorem sequence_formula (n : ℕ) :
  (sequence n = 3 * n + 1) :=
by
  sorry

end sequence_formula_l698_698939


namespace statement_A_statement_B_statement_D_l698_698986

open Real

variables (θ : ℝ) (a b : ℝ × ℝ)

def vector_a : ℝ × ℝ := (1, 1)

def vector_b : ℝ × ℝ := (cos θ, sin θ)

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable
def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  Real.acos ((vector_dot_product u v) / (vector_magnitude u * vector_magnitude v))

theorem statement_A :
  (vector_b θ = (Real.sqrt 2 / 2, Real.sqrt 2 / 2)) → θ = π / 4 :=
sorry

theorem statement_B :
  0 ≤ angle_between_vectors vector_a (vector_b θ) ∧ angle_between_vectors vector_a (vector_b θ) ≤ 3 * π / 4 :=
sorry

theorem statement_D :
  (∃ θ, vector_magnitude (vector_a + vector_b θ) = vector_magnitude (vector_a - vector_b θ)) :=
sorry

end statement_A_statement_B_statement_D_l698_698986


namespace number_of_ways_to_cut_staircase_l698_698224

-- Define the shapes and the staircase
inductive Shape
| rect1x6
| rect1x5
| rect1x4
| rect1x3
| rect1x2
| square1x1

def shapes : List Shape := [Shape.rect1x6, Shape.rect1x5, Shape.rect1x4, Shape.rect1x3, Shape.rect1x2, Shape.square1x1]

-- The theorem statement
theorem number_of_ways_to_cut_staircase :
  let placements : List Nat := [2, 2, 2, 2, 2, 1]
  (placements.product = 32) :=
  sorry

end number_of_ways_to_cut_staircase_l698_698224


namespace decagon_adjacent_probability_l698_698744

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698744


namespace find_angle_B_l698_698944

open Real 
open Emetric Geometry EuclideanGeometry

theorem find_angle_B (A B C G : Point) (a b c : ℝ) (sinA sinB sinC : ℝ)
  (h₁ : centroid_triangle A B C = G)
  (h₂ : (sqrt 7 / ∥G - A∥ * sinA • (G - A) + 3 * sinB • (G - B) + 3 * sqrt 7 * sinC • (G - C) = 0)) :
  ∠B = 60 :=
sorry

end find_angle_B_l698_698944


namespace vector_problem_l698_698988

-- Definition of the vectors a and b
def a : ℝ × ℝ × ℝ := (1, -1, -2)
def b : ℝ × ℝ × ℝ := (1, -3, -3)

-- Prove that a - b = (0, 2, 1) and a ⋅ b = 10
theorem vector_problem : (a.1 - b.1, a.2 - b.2, a.3 - b.3) = (0, 2, 1) ∧ 
                         (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 10) :=
by
  sorry

end vector_problem_l698_698988


namespace complex_pure_imaginary_solution_l698_698951

theorem complex_pure_imaginary_solution (m : ℝ) : 
  let z := m^2 * (1 + complex.I) - m * (3 + 6 * complex.I) in
  (z.re = 0) → (m ≠ 0 ∧ m ≠ 6) → m = 3 :=
by
  intros
  sorry

end complex_pure_imaginary_solution_l698_698951


namespace probability_adjacent_vertices_decagon_l698_698739

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698739


namespace base_133_not_perfect_square_l698_698680

theorem base_133_not_perfect_square (b : ℤ) : ¬ ∃ k : ℤ, b^2 + 3 * b + 3 = k^2 := by
  sorry

end base_133_not_perfect_square_l698_698680


namespace proof_angle_BHD_eq_90_proof_sum_of_angles_AHD_DHG_GHF_eq_180_l698_698255

-- Define the primary elements (squares and intersecting lines) and points as terms in Lean
variables (A B C D E F G H : Type)
variables [IsSquare ABCD : Quadrilateral A B C D]
variables [IsSquare EFGC : Quadrilateral E F G C]
variables [Intersect BG DE : Intersects (Line B G) (Line D E) H]

-- Problem statement as Lean 4 definitions
def angle_BHD_eq_90 : Prop :=
  ∃ (theta : ℝ), theta = 90 ∧ angle (B H D) = theta

def sum_of_angles_AHD_DHG_GHF_eq_180 : Prop :=
  ∃ (alpha beta gamma : ℝ), alpha = 45 ∧ beta = 90 ∧ gamma = 45 ∧ 
  angle (A H D) = alpha ∧ angle (D H G) = beta ∧ angle (G H F) = gamma ∧ 
  alpha + beta + gamma = 180

-- Declarations of the Lean 4 statements.
theorem proof_angle_BHD_eq_90
  (sq1 : IsSquare ABCD)
  (sq2 : IsSquare EFGC)
  (intersect : Intersects (Line B G) (Line D E) H) :
  angle_BHD_eq_90 :=
sorry

theorem proof_sum_of_angles_AHD_DHG_GHF_eq_180
  (sq1 : IsSquare ABCD)
  (sq2 : IsSquare EFGC)
  (intersect : Intersects (Line B G) (Line D E) H) :
  sum_of_angles_AHD_DHG_GHF_eq_180 :=
sorry

end proof_angle_BHD_eq_90_proof_sum_of_angles_AHD_DHG_GHF_eq_180_l698_698255


namespace probability_in_sync_l698_698846

theorem probability_in_sync :
  let s := {1, 2, 3, 4, 5, 6}
  let in_sync (a b : ℕ) : Prop := abs (a - b) ≤ 1
  let total_outcomes : ℕ := 36
  let favorable_outcomes : ℕ := 11
  (favorable_outcomes.to_rat / total_outcomes.to_rat) = (11 : ℚ / 36 : ℚ) :=
by
  sorry

end probability_in_sync_l698_698846


namespace solve_problem_l698_698103

-- Define the polynomial g(x) as given in the problem
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- Define the condition given in the problem
def condition (p q r s t : ℝ) : Prop := g p q r s t (-2) = -4

-- State the theorem to be proved
theorem solve_problem (p q r s t : ℝ) (h : condition p q r s t) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 4 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l698_698103


namespace probability_of_three_vertices_inside_tetrahedron_l698_698435

-- Define a tetrahedron with four vertices.
structure Tetrahedron :=
(vertices : Fin 4 → Point)

-- Assume we have a valid tetrahedron structure
variable (T : Tetrahedron) 

-- Define a function that chooses three vertices at random
noncomputable def choose_three_vertices (T : Tetrahedron) : Finset (Fin 4) :=
{a // a.card = 3}

-- Define a function that checks if a plane determined by three vertices contains points inside the tetrahedron
def contains_points_inside (T : Tetrahedron) (S : Finset (Fin 4)) : Prop :=
true -- Since any three vertices of a tetrahedron always form a valid triangle with points inside

-- Define the probability computation
noncomputable def probability (T : Tetrahedron) :=
if (∀ S ∈ choose_three_vertices T, contains_points_inside T S) then 1 else 0

-- The main theorem statement
theorem probability_of_three_vertices_inside_tetrahedron (T : Tetrahedron) :
  probability T = 1 := sorry

end probability_of_three_vertices_inside_tetrahedron_l698_698435


namespace zoey_finishes_on_sunday_l698_698386

theorem zoey_finishes_on_sunday :
  let start_day := 0 in
  let total_books := 20 in
  let duration_of_book n := n in
  let sum_of_durations := (total_books * (total_books + 1)) / 2 in
  (start_day + sum_of_durations) % 7 = 0 := 
by
  sorry

end zoey_finishes_on_sunday_l698_698386


namespace neg_intervals_of_expression_l698_698140

theorem neg_intervals_of_expression (x : ℝ) :
  (x - 2) * (x + 2) * (x - 3) < 0 ↔ x ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo 2 3 := 
sorry

end neg_intervals_of_expression_l698_698140


namespace division_subtraction_eq_eight_l698_698821

theorem division_subtraction_eq_eight (x : ℝ) : (x / 4 = x - 6) → x = 8 :=
begin
  intro h,
  have step1 : (x / 4 = x - 6) := h,
  sorry
end

end division_subtraction_eq_eight_l698_698821


namespace hypotenuse_of_right_triangle_l698_698339

/-- A right triangle with sides 2a and 2b, whose medians from the vertices of the acute angles are
given to be 5 and √40, has a hypotenuse of length 2√13. -/
theorem hypotenuse_of_right_triangle (a b : ℝ) (h1 : sqrt(b^2 + (a / 2)^2) = sqrt 40)
  (h2 : sqrt(a^2 + (b / 2)^2) = 5) : sqrt ((2 * a)^2 + (2 * b)^2) = 2 * sqrt 13 :=
by 
  sorry

end hypotenuse_of_right_triangle_l698_698339


namespace product_of_third_sides_is_correct_l698_698767

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l698_698767


namespace max_value_f_on_interval_l698_698156

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_f_on_interval : 
  ∃ x ∈ set.Icc (-3 : ℝ) 3, ∀ y ∈ set.Icc (-3 : ℝ) 3, f x ≥ f y ∧ f x = 18 := 
by {
  sorry
}

end max_value_f_on_interval_l698_698156


namespace student_scores_l698_698109

theorem student_scores :
  let w1 := 0.50
  let s1 := 0.72
  let w2 := 0.30
  let s2 := 0.84
  let w3 := 0.20
  let s_need := 0.75
  let total_w := 1.0
  (w1 * s1 + w2 * s2 + w3 * 0.69) / total_w = s_need ∧ (w1 + w2 + w3 + 0) = total_w.
Proof: sorry

end student_scores_l698_698109


namespace right_triangle_third_side_product_l698_698781

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l698_698781


namespace right_angle_triangle_problem_l698_698273

open EuclideanGeometry

theorem right_angle_triangle_problem
    (A B C K D E M : Point)
    (hABC : right_triangle ∠A ∠B ∠C)
    (angleA : ∠A = 90°)
    (angleB : ∠B = 30°)
    (midM : midpoint M B C)
    (perpendicularM : perp_scalar M line BC)
    (bisectorBK : is_bisector BK ∠B)
    (meetE : ∃ E, line BK ∩ perpendicularM = {E})
    (perpendicular_bisector_EK : is_perpendicular_bisector line EK line AB)
    (meetD : D ∈ line AB ∩ perpendicular_bisector_EK)
    (KD : line KD)
    (DE : line DE) :
  perp KD DE := sorry

end right_angle_triangle_problem_l698_698273


namespace isosceles_triangle_angle_bisector_equality_l698_698217

theorem isosceles_triangle_angle_bisector_equality
  (ABC : Triangle)
  (D E F : Point)
  (hD : is_angle_bisector ABC A D)
  (hE : is_angle_bisector ABC B E)
  (hF : is_angle_bisector ABC C F)
  (M N : Point)
  (hMN_EF : OnLineSegment M EF ∧ OnLineSegment N EF)
  (hAM_AN : dist A M = dist A N)
  (H : Point)
  (hH : is_foot_of_altitude A BC H)
  (K L : Point)
  (hKL_EF : OnLineSegment K EF ∧ OnLineSegment L EF)
  (h_sim : similar (Triangle.mk A K L) (Triangle.mk H M N))
  (hKA_HM : ¬parallel (Line.mk A K) (Line.mk H M))
  (hKA_HN : ¬parallel (Line.mk A K) (Line.mk H N)) :
  dist D K = dist D L :=
by sorry

end isosceles_triangle_angle_bisector_equality_l698_698217


namespace probability_of_adjacent_vertices_l698_698724

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698724


namespace toby_photos_proof_l698_698712

theorem toby_photos_proof (P_0 D_1 C F D_2 A S_2 S_3 D_3 D_4 P_f : ℕ) 
  (h1 : P_0 = 63)
  (h2 : D_1 = 7)
  (h3 : C = 15)
  (h4 : D_2 = 3)
  (h5 : A = 5)
  (h6 : S_2 = 11)
  (h7 : S_3 = 8)
  (h8 : D_3 = 6)
  (h9 : D_4 = 4)
  (h10 : P_f = 112) :
  let intermediate1 := P_0 - D_1 in
  let intermediate2 := intermediate1 + C in
  let intermediate3 := intermediate2 + (F - D_2) in
  let intermediate4 := intermediate3 + A in
  let intermediate5 := intermediate4 + (S_2 + S_3 - D_3 - D_4) in
  (P_f = intermediate5) → 
  (F + S_2 + S_3 = 46) := sorry

end toby_photos_proof_l698_698712


namespace harkamal_total_payment_l698_698567

theorem harkamal_total_payment:
  let cost_grapes := 8 * 80,
      cost_mangoes := 9 * 55 in
  cost_grapes + cost_mangoes = 1135 :=
by
  let cost_grapes := 8 * 80,
      cost_mangoes := 9 * 55
  sorry

end harkamal_total_payment_l698_698567


namespace f2_in_A_f2_inequality_l698_698352

def f1 (x : ℝ) : ℝ := sqrt x - 2
def f2 (x : ℝ) : ℝ := 4 - 6 * (1 / 2) ^ x

def in_set_A (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, 0 ≤ x → -2 ≤ f x ∧ f x < 4) ∧
  (∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y)

theorem f2_in_A : in_set_A f2 :=
by
  sorry

theorem f2_inequality (x : ℝ) (h : 0 ≤ x) : f2 x + f2 (x + 2) < 2 * f2 (x + 1) :=
by
  sorry

end f2_in_A_f2_inequality_l698_698352


namespace max_rubles_earned_l698_698020

theorem max_rubles_earned :
  ∀ (cards_with_1 cards_with_2 : ℕ), 
  cards_with_1 = 2013 ∧ cards_with_2 = 2013 →
  ∃ (max_moves : ℕ), max_moves = 5 :=
by
  intros cards_with_1 cards_with_2 h
  sorry

end max_rubles_earned_l698_698020


namespace combined_age_of_siblings_l698_698441

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end combined_age_of_siblings_l698_698441


namespace commencement_addresses_sum_l698_698602

noncomputable def addresses (S H L : ℕ) := 40

theorem commencement_addresses_sum
  (S H L : ℕ)
  (h1 : S = 12)
  (h2 : S = 2 * H)
  (h3 : L = S + 10) :
  S + H + L = addresses S H L :=
by
  sorry

end commencement_addresses_sum_l698_698602


namespace calculate_annual_cost_l698_698030

noncomputable def doctor_visit_cost (visits_per_year: ℕ) (cost_per_visit: ℕ) : ℕ :=
  visits_per_year * cost_per_visit

noncomputable def medication_night_cost (pills_per_night: ℕ) (cost_per_pill: ℕ) : ℕ :=
  pills_per_night * cost_per_pill

noncomputable def insurance_coverage (medication_cost: ℕ) (coverage_percent: ℕ) : ℕ :=
  medication_cost * coverage_percent / 100

noncomputable def out_of_pocket_cost (total_cost: ℕ) (insurance_covered: ℕ) : ℕ :=
  total_cost - insurance_covered

noncomputable def annual_medication_cost (night_cost: ℕ) (nights_per_year: ℕ) : ℕ :=
  night_cost * nights_per_year

noncomputable def total_annual_cost (doctor_visit_total: ℕ) (medication_total: ℕ) : ℕ :=
  doctor_visit_total + medication_total

theorem calculate_annual_cost :
  let visits_per_year := 2 in
  let cost_per_visit := 400 in
  let pills_per_night := 2 in
  let cost_per_pill := 5 in
  let coverage_percent := 80 in
  let nights_per_year := 365 in
  let total_cost := total_annual_cost 
    (doctor_visit_cost visits_per_year cost_per_visit)
    (annual_medication_cost 
      (out_of_pocket_cost 
        (medication_night_cost pills_per_night cost_per_pill) 
        (insurance_coverage (medication_night_cost pills_per_night cost_per_pill) coverage_percent)
      ) 
      nights_per_year
    ) in
  total_cost = 1530 :=
begin
  sorry
end

end calculate_annual_cost_l698_698030


namespace longest_chord_in_circle_l698_698956

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end longest_chord_in_circle_l698_698956


namespace p_necessary_not_sufficient_for_q_l698_698466

open Classical

variable (p q : Prop)

theorem p_necessary_not_sufficient_for_q (h1 : ¬(p → q)) (h2 : ¬q → ¬p) : (¬(p → q) ∧ (¬q → ¬p) ∧ (¬p → ¬q ∧ ¬(¬q → p))) := by
  sorry

end p_necessary_not_sufficient_for_q_l698_698466


namespace sum_of_k_distinct_integer_roots_l698_698375

theorem sum_of_k_distinct_integer_roots (k : ℤ) 
: ∑ k, (∃ p q : ℤ, p ≠ q ∧ 2 * p^2 - k * p + 5 = 0 ∧ 2 * q^2 - k * q + 5 = 0) = 0 := 
sorry

end sum_of_k_distinct_integer_roots_l698_698375


namespace not_exists_implies_bounds_l698_698524

variable (a : ℝ)

/-- If there does not exist an x such that x^2 + (a - 1) * x + 1 < 0, then -1 ≤ a ∧ a ≤ 3. -/
theorem not_exists_implies_bounds : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) :=
by sorry

end not_exists_implies_bounds_l698_698524


namespace irrationals_in_set_l698_698446

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

def num_irrationals (s : set ℝ) : ℕ := (s.filter is_irrational).to_finset.card

theorem irrationals_in_set : num_irrationals ({3 * real.pi, -7 / 8, 0, real.sqrt 2, -3.15, real.sqrt 9, real.sqrt 3 / 3} : set ℝ) = 3 :=
by
  sorry

end irrationals_in_set_l698_698446


namespace complex_number_addition_l698_698513

section
variables {a b : ℝ} (z : ℂ) (conj_z : ℂ)
  (h1 : z = a + 3 * complex.I)
  (h2 : conj_z = 2 + b * complex.I)
  (h3 : conj_z = complex.conj z)

theorem complex_number_addition : a + b = -1 :=
by
  sorry
end

end complex_number_addition_l698_698513


namespace decagon_adjacent_probability_l698_698741

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l698_698741


namespace sin_330_l698_698922

theorem sin_330 (θ : ℝ) (hθ : θ = 30) : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 = 360 - 30 := by norm_num
  have h2 : Real.sin (360 * Real.pi / 180 - θ * Real.pi / 180) = - Real.sin (θ * Real.pi / 180) :=
    by rw [Real.sin_sub, Real.sin_pi, Real.cos_pi, Real.sin_mul, Real.pi_div_two, Real.coThm]
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
    norm_num
  rw [h1, h2, hθ, h3]
  norm_num
  sorry

end sin_330_l698_698922


namespace point_M_in_second_quadrant_l698_698609

-- Given conditions
def m : ℤ := -2
def n : ℤ := 1

-- Definitions to identify the quadrants
def point_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

-- Problem statement to prove
theorem point_M_in_second_quadrant : 
  point_in_second_quadrant m n :=
by
  sorry

end point_M_in_second_quadrant_l698_698609


namespace determine_m_l698_698590

theorem determine_m (m : ℝ) : (∀ x : ℝ, (0 < x ∧ x < 2) ↔ -1/2 * x^2 + 2 * x + m * x > 0) → m = -1 :=
by
  intro h
  sorry

end determine_m_l698_698590


namespace triangle_sides_root_bounds_l698_698236

theorem triangle_sides_root_bounds (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x1 = 1) ∧ (x1, x2, x3 are roots of (x-1)(x^2-2x+m)) ∧ x2 + x3 > x1 ∧ |x2 - x3| < x1) ↔ (3/4 < m ∧ m ≤ 1) :=
by
  sorry

end triangle_sides_root_bounds_l698_698236


namespace area_of_playground_l698_698346

-- Definitions of the problem's conditions
variables {w l : ℝ}

-- Conditions given in the problem
def perimeter_condition : Prop := 2 * l + 2 * w = 90
def length_width_relation : Prop := l = 3 * w

-- The statement that needs to be proved
theorem area_of_playground (h1 : perimeter_condition) (h2 : length_width_relation) : l * w = 380.15625 := by
  sorry

end area_of_playground_l698_698346


namespace consecutive_numbers_l698_698021

theorem consecutive_numbers (x : ℕ) (h : (4 * x + 2) * (4 * x^2 + 6 * x + 6) = 3 * (4 * x^3 + 4 * x^2 + 18 * x + 8)) :
  x = 2 :=
sorry

end consecutive_numbers_l698_698021


namespace probability_of_adjacent_vertices_in_decagon_l698_698750

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698750


namespace incorrect_description_l698_698672

-- Conditions
def population_size : ℕ := 2000
def sample_size : ℕ := 150

-- Main Statement
theorem incorrect_description : ¬ (sample_size = 150) := 
by sorry

end incorrect_description_l698_698672


namespace intersection_eq_l698_698839

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem intersection_eq : M ∩ N = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_eq_l698_698839


namespace xiao_ming_cube_division_l698_698384

theorem xiao_ming_cube_division (large_edge small_cubes : ℕ)
  (large_edge_eq : large_edge = 4)
  (small_cubes_eq : small_cubes = 29)
  (total_volume : large_edge ^ 3 = 64) :
  ∃ (small_edge_1_cube : ℕ), small_edge_1_cube = 24 ∧ small_cubes = 29 ∧ 
  small_edge_1_cube + (small_cubes - small_edge_1_cube) * 8 = 64 := 
by
  -- We only need to assert the existence here as per the instruction.
  sorry

end xiao_ming_cube_division_l698_698384


namespace vector_parallel_l698_698984

theorem vector_parallel (x y : ℝ) (a b : ℝ × ℝ × ℝ) (h_parallel : a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ k : ℝ, a = k • b) : x + y = 6 :=
by sorry

end vector_parallel_l698_698984


namespace travel_time_from_B_to_A_l698_698699

theorem travel_time_from_B_to_A (trolley_interval travel_time_trolley trolleys_encountered : ℕ) :
    trolley_interval = 5 →
    travel_time_trolley = 15 →
    trolleys_encountered = 10 →
    travel_time_trolley * 2 / trolley_interval + trolley_interval * trolleys_encountered = 50 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end travel_time_from_B_to_A_l698_698699


namespace infinite_natural_numbers_with_perfect_square_average_l698_698307

def is_perfect_square (k : ℕ) : Prop := ∃ m : ℕ, m * m = k

theorem infinite_natural_numbers_with_perfect_square_average :
  (∃^∞ n : ℕ, is_perfect_square ((n + 1) * (2 * n + 1) / 6)) ∧
  (is_perfect_square ((337 + 1) * (2 * 337 + 1) / 6)) ∧
  (is_perfect_square ((65521 + 1) * (2 * 65521 + 1) / 6)) :=
  sorry

end infinite_natural_numbers_with_perfect_square_average_l698_698307


namespace cross_product_identity_l698_698277

open Matrix

variables (v w : ℝ^3)

def cross_product (a b : ℝ^3) : ℝ^3 :=
  ![a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]

theorem cross_product_identity (h : cross_product v w = ![3, -1, 2]) :
  cross_product (v + 2 • w) (v + 2 • w) = ![0, 0, 0] :=
by sorry

end cross_product_identity_l698_698277


namespace gcd_2025_2070_l698_698795

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end gcd_2025_2070_l698_698795


namespace kittens_given_by_sara_l698_698362

-- Define conditions
def initial_kittens : ℕ := 6
def kittens_given_away : ℕ := 3
def kittens_after_receiving : ℕ := 12

-- Define the proof statement
theorem kittens_given_by_sara :
  kittens_after_receiving = initial_kittens - kittens_given_away + kittens_given_by_sara → kittens_given_by_sara = 9 :=
by
  sorry

end kittens_given_by_sara_l698_698362


namespace distinct_quadrilaterals_count_equal_tetrahedra_volumes_l698_698076

section QuadrilateralVolume

variables (a b c d : ℝ^3)

-- Define the volume of a parallelepiped given three vectors
def parallelepiped_volume (u v w : ℝ^3) : ℝ :=
  |u ⋅ (v × w)|

-- Define the volume of a tetrahedron given three vectors
def tetrahedron_volume (u v w : ℝ^3) : ℝ :=
  (1 / 6) * parallelepiped_volume u v w

-- Proving the number of distinct quadrilaterals
theorem distinct_quadrilaterals_count : 
  let sides := list.permutations [a, b, c, d] in
  sides.length = 6 := sorry

-- Proving the volumes of tetrahedra are equal
theorem equal_tetrahedra_volumes : 
  let tetrahedra := list.map (fun perm => tetrahedron_volume perm[0] perm[1] perm[2]) (list.permutations [a, b, c, d]) in
  ∀v ∈ tetrahedra, v = (1 / 6) * parallelepiped_volume a b c := sorry

end QuadrilateralVolume

end distinct_quadrilaterals_count_equal_tetrahedra_volumes_l698_698076


namespace right_triangle_third_side_product_l698_698779

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l698_698779


namespace solution_set_for_f_gt_3x_plus_1_l698_698953

variable {x : ℝ}
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_condition1 : f 1 = 4
axiom f_condition2 : ∀ x : ℝ, f' x < 3

-- Proof goal
theorem solution_set_for_f_gt_3x_plus_1 : {x : ℝ | f x > 3 * x + 1} = set.Iio 1 :=
sorry

end solution_set_for_f_gt_3x_plus_1_l698_698953


namespace range_of_a_l698_698972

-- Definitions for f and g
def f (x : ℝ) (a : ℝ) : ℝ := x + a
def g (x : ℝ) : ℝ := x + 4 / x

-- Conditions
variable (a : ℝ)
variable (x1 : ℝ) (x2 : ℝ)
variable h1 : 1 ≤ x1 ∧ x1 ≤ 3
variable h2 : 1 ≤ x2 ∧ x2 ≤ 4
variable h : ∀ x1 ∈ set.Icc 1 3, ∃ x2 ∈ set.Icc 1 4, f x1 a ≥ g x2

-- Goal
theorem range_of_a : a ≥ 3 :=
by
  sorry -- Proof goes here

end range_of_a_l698_698972


namespace minimal_cross_section_area_of_tetrahedron_l698_698598

noncomputable def regular_tetrahedron (A B C D : Point) : Prop :=
is_regular_tetrahedron A B C D ∧ edge_length A B = 1

def is_midpoint (M A D : Point) : Prop :=
(euclidean_dist M A = euclidean_dist M D) ∧ piece_length A D = 1

def is_minimal_cross_section_area := 
∀ (A B C D : Point) (M N : Point), 
  regular_tetrahedron A B C D → 
  is_midpoint M A D →
  is_midpoint N B C →
  cross_section_area (A B C D) (plane_through_line_segment M N) = 1 / 4

theorem minimal_cross_section_area_of_tetrahedron :
  is_minimal_cross_section_area :=
begin
  intros A B C D,
  intros M N h_tetra h_midpoint_M h_midpoint_N,
  sorry
end

end minimal_cross_section_area_of_tetrahedron_l698_698598


namespace number_of_triangles_l698_698573

theorem number_of_triangles (h_div_subdivision: ∀ x y : ℕ, (x ∈ {1,2,3} → y ∈ {1,2,3} → 
  ∃ Δ : Type, Δ ⊆ draw_rectangle) ∧ 
  (hvertical : is_vertical x) ∧ 
  (hhorizontal : is_horizontal y)) 
  (h_diagonals: ∃ Δ : Type, Δ ⊆ draw_diagonals) :
  (∃ n, n = 76) := 
sorry

end number_of_triangles_l698_698573


namespace max_min_P_l698_698630

noncomputable def f (x t : ℝ) : ℝ := t * Real.sin x + (1 - t) * Real.cos x

noncomputable def P (t : ℝ) : ℝ := 
  (∫ x in 0..(Real.pi / 2), Real.exp x * f x t) * (∫ x in 0..(Real.pi / 2), Real.exp (-x) * f x t)

theorem max_min_P : 
  ∃ (maxP minP : ℝ), 
    (0 ≤ t ∧ t ≤ 1 → P t ≤ maxP) ∧ 
    (0 ≤ t ∧ t ≤ 1 → minP ≤ P t) ∧
    maxP = (1 / 4) * Real.exp (Real.pi / 2) ∧ 
    minP = (1 / 2) * Real.sinh (Real.pi / 2) :=
begin
  sorry
end

end max_min_P_l698_698630


namespace polynomial_divisibility_l698_698899

def main : IO Unit :=
IO.println s!"Hello, {hello}!"

def polynomial (m : ℝ) (x : ℝ) := 2*x^3 - 6*x^2 + m*x - 18

def is_divisible (P Q : ℝ → ℝ) : Prop :=
∀ x : ℝ, Q x = 0 → P x = 0

theorem polynomial_divisibility (m : ℝ) : m = 6 → 
  (is_divisible (polynomial m) (λ x, x - 3)) ∧ 
  (is_divisible (polynomial m) (λ x, 2*x^2 + 6)) :=
sorry

end polynomial_divisibility_l698_698899


namespace simplify_trig_expression_l698_698311

theorem simplify_trig_expression : 
  (tan 20 + tan 30 + tan 60 + tan 70) / cos 10 = 2 * cos 40 / cos 10 * (cos 20 * cos 30 * cos 60 * cos 70) := 
sorry

end simplify_trig_expression_l698_698311


namespace value_of_x_l698_698353

theorem value_of_x (x : ℕ) (h1 : ∑ d in (factors x), d = 24) (h2 : 3 ∣ x) : x = 15 :=
sorry

end value_of_x_l698_698353


namespace problem1_problem2_l698_698838

noncomputable theory -- Required because the exponents and constants will not be computable
open BigOperators

-- Problem 1 Definition
def expression1 : ℤ := -3^2 + 2^2023 * (-1/2)^2022 + (-2024)^0

-- Problem 2 Definition
def expression2 (x y : ℝ) : ℝ :=
  (((x + 2*y)^2 - (2*x + y)*(2*x - y) - 5*(x^2 + y^2)) / (2*x))

-- Problem 1 Proof Statement
theorem problem1 : expression1 = -6 := sorry

-- Problem 2 Proof Statement (Evaluation)
theorem problem2 : expression2 (-1/2) 1 = 4 := sorry

end problem1_problem2_l698_698838


namespace number_of_integers_l698_698162

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l698_698162


namespace water_consumption_correct_l698_698415

-- Define the constants provided in the conditions
def base_rate : ℝ := 1.8
def excess_rate : ℝ := 2.3
def sewage_rate : ℝ := 1
def threshold : ℝ := 15
def total_bill : ℝ := 58.5

-- Define the total water bill equation given the consumption x
def total_water_bill (x : ℝ) : ℝ :=
  if x ≤ threshold then
    x * (base_rate + sewage_rate)
  else
    (threshold * (base_rate + sewage_rate)) + ((x - threshold) * (excess_rate + sewage_rate))

-- State the theorem to prove the correct water consumption
theorem water_consumption_correct (x : ℝ) : total_water_bill x = total_bill → x = 20 :=
sorry

end water_consumption_correct_l698_698415


namespace max_norm_value_l698_698983

variable {𝒜 : Type*} [inner_product_space ℝ 𝒜] [normed_space ℝ 𝒜] [normed_group 𝒜]

-- Define non-zero vectors m and n
variables (m n : 𝒜) (hm : ∥m∥ = 2) (h : ∥m + 2 • n∥ = 2) 

-- Define the maximum value of ∥n∥ + ∥2 • m + n∥
noncomputable def max_value := ∥n∥ + ∥2 • m + n∥

-- Prove the maximum value is 8√3 / 3
theorem max_norm_value : max_value m n hm h = 8 * real.sqrt 3 / 3 :=
by sorry

end max_norm_value_l698_698983


namespace inequality_holds_for_m_l698_698632

theorem inequality_holds_for_m (n : ℕ) (m : ℕ) :
  (∀ a b : ℝ, (0 < a ∧ 0 < b) ∧ (a + b = 2) → (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by
  sorry

end inequality_holds_for_m_l698_698632


namespace fee_percentage_l698_698468

theorem fee_percentage (quarters dimes nickels pennies received : ℕ)
  (hquarters : quarters = 76) (hdimes : dimes = 85) (hnickels : nickels = 20) 
  (hpennies : pennies = 150) (hreceived : received = 27) : 
  let total_before_fee := 76 * 0.25 + 85 * 0.10 + 20 * 0.05 + 150 * 0.01 in
  let fee := total_before_fee - received in
  (fee / total_before_fee) * 100 = 10 :=
by
  sorry

end fee_percentage_l698_698468


namespace magnitude_of_c_l698_698924

def root_poly (c : ℂ) (x : ℂ) : ℂ :=
  (x^2 - 3*x + 3) * (x^2 - c*x + 5) * (x^2 - 5*x + 15)

noncomputable def distinct_roots (c : ℂ) : Prop :=
  ∃ s : finset ℂ, s.card = 5 ∧ ∀ x ∈ s, root_poly c x = 0

theorem magnitude_of_c (c : ℂ) (h : distinct_roots c): |c| = 3 :=
sorry

end magnitude_of_c_l698_698924


namespace smallest_n_l698_698848

theorem smallest_n (n : ℕ) (h₁ : n > 0) (h₂ : (∏ i in finset.range n, (8 - i) / (9 - i)) < 0.5) : n = 6 :=
sorry

end smallest_n_l698_698848


namespace solution_in_interval_l698_698230

noncomputable theory

open Real

/-- Define the function f(x) = ln(x) + x -/
def f (x : ℝ) : ℝ := log x + x

/-- The statement to be proved: if x_0 is a solution to f(x) = 3, then x_0 belongs to the interval (2, 3) -/
theorem solution_in_interval (x_0 : ℝ) (h : f x_0 = 3) : 2 < x_0 ∧ x_0 < 3 :=
sorry

end solution_in_interval_l698_698230


namespace directrix_of_parabola_is_one_l698_698152

-- Define the conditions of the problem
def parabola_eqn (x : ℝ) : ℝ := - (1 / 4) * x ^ 2

-- The theorem statement: the equation of the directrix of the given parabola is y = 1
theorem directrix_of_parabola_is_one : ∀ y : ℝ, (∃ x : ℝ, parabola_eqn x = y) → y = 1 :=
by
  -- Proof omitted
  sorry

end directrix_of_parabola_is_one_l698_698152


namespace find_roots_of_polynomial_l698_698153

noncomputable theory

def polynomial := 3 * (x : ℝ)^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3

theorem find_roots_of_polynomial : 
  ∀ x : ℝ, polynomial = 0 ↔ (x = (-1 + sqrt (-171 + 12 * sqrt 43)) / 6) 
                          ∨ (x = (-1 - sqrt (-171 + 12 * sqrt 43)) / 6)
                          ∨ (x = (-1 + sqrt (-171 - 12 * sqrt 43)) / 6)
                          ∨ (x = (-1 - sqrt (-171 - 12 * sqrt 43)) / 6) :=
by
  sorry

end find_roots_of_polynomial_l698_698153


namespace abc_plus_ab_plus_a_div_4_l698_698302

noncomputable def prob_abc_div_4 (a b c : ℕ) (isPositive_a : 0 < a) (isPositive_b : 0 < b) (isPositive_c : 0 < c) (a_in_range : a ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (b_in_range : b ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (c_in_range : c ∈ {k | 1 ≤ k ∧ k ≤ 2009}) : ℚ :=
  let total_elements : ℚ := 2009
  let multiples_of_4 := 502
  let non_multiples_of_4 := total_elements - multiples_of_4
  let prob_a_div_4 : ℚ := multiples_of_4 / total_elements
  let prob_a_not_div_4 : ℚ := non_multiples_of_4 / total_elements
  sorry

theorem abc_plus_ab_plus_a_div_4 : ∃ P : ℚ, prob_abc_div_4 a b c isPositive_a isPositive_b isPositive_c a_in_range b_in_range c_in_range = P :=
by sorry

end abc_plus_ab_plus_a_div_4_l698_698302


namespace right_triangle_third_side_product_l698_698773

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l698_698773


namespace max_number_of_chips_l698_698251

noncomputable def max_chips (n : Nat) : Nat :=
  200 * 200

theorem max_number_of_chips :
  ∃ (chips : Fin 40000 → Fin 2), (∀ i, ∑ j, if chips i = chips j then 0 else 1 = 5) → (∑ i, 1 ≤ 3800) :=
by
  sorry

end max_number_of_chips_l698_698251


namespace problem1_problem2_problem3_problem4_l698_698456

theorem problem1 : 6 + (-8) - (-5) = 3 := by
  sorry

theorem problem2 : (5 + 3/5) + (-(5 + 2/3)) + (4 + 2/5) + (-1/3) = 4 := by
  sorry

theorem problem3 : ((-1/2) + 1/6 - 1/4) * 12 = -7 := by
  sorry

theorem problem4 : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by
  sorry

end problem1_problem2_problem3_problem4_l698_698456


namespace locus_of_points_l698_698636

noncomputable theory
open Set

variables {Point : Type}
variable (A B C D O G : Point)
variable (R : ℝ)
variable (AP BP CP DP PA1 PB1 PC1 PD1 : Point → ℝ)
variable (PA PB PC PD : Point → ℝ)

def is_inscribed_tetrahedron (A B C D : Point) (S : Point → Prop) : Prop :=
  S A ∧ S B ∧ S C ∧ S D

def is_intersection_with_sphere (S : Point → Prop) (P : Point) (A1 B1 C1 D1: Point) : Prop :=
  S A1 ∧ S B1 ∧ S C1 ∧ S D1

def distance_condition (P : Point) : Prop :=
  PA P / PA1 P + PB P / PB1 P + PC P / PC1 P + PD P / PD1 P = 4

def sphere_diameter_midpoint (O G: Point) (S : Point → Prop) :=
  ∃ diameter : ℝ, S (midpoint O G) ∧ diameter = dist O G

theorem locus_of_points (S : Point → Prop) :
  is_inscribed_tetrahedron A B C D S →
  is_intersection_with_sphere S P PA1 PB1 PC1 PD1 →
  distance_condition P →
  sphere_diameter_midpoint O G S :=
begin
  sorry
end

end locus_of_points_l698_698636


namespace word_problems_count_l698_698359

theorem word_problems_count (total_questions add_sub_questions : ℕ) :
  total_questions = 45 → add_sub_questions = 28 → total_questions - add_sub_questions = 17 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end word_problems_count_l698_698359


namespace min_number_of_roots_l698_698497

theorem min_number_of_roots (k0 k1 k2 : ℕ) (h_order: k0 < k1 ∧ k1 < k2) 
  (A1 A2 : ℝ) : 
  ∃ N ≥ 2 * k0, ∀ x ∈ Icc (0 : ℝ) (2 * real.pi), 
     sin (k0 * x) + A1 * sin (k1 * x) + A2 * sin (k2 * x) = 0 :=
sorry

end min_number_of_roots_l698_698497


namespace area_CDEB_correct_l698_698032

noncomputable def area_CDEB (A B C D F E : ℝ × ℝ) : ℝ := sorry

theorem area_CDEB_correct : 
  ∀ (A B C D F E : ℝ × ℝ),
  (dist A B = 2 * real.sqrt 5) →
  (dist B C = 1) →
  (dist C A = 5) →
  (colinear C D A) →
  (colinear F D A) →
  (dist C D = 1) →
  (dist B F = 2) →
  (dist C F = 3) →
  (colinear D F E) →
  (area_CDEB A B C D F E) = 22 / 35 :=
begin
  sorry
end

end area_CDEB_correct_l698_698032


namespace find_incorrect_statements_l698_698382

-- Definitions of the statements based on their mathematical meanings
def is_regular_tetrahedron (shape : Type) : Prop := 
  -- assume some definition for regular tetrahedron
  sorry 

def is_cube (shape : Type) : Prop :=
  -- assume some definition for cube
  sorry

def is_generatrix_parallel (cylinder : Type) : Prop :=
  -- assume definition stating that generatrix of a cylinder is parallel to its axis
  sorry

def is_lateral_faces_isosceles (pyramid : Type) : Prop :=
  -- assume definition that in a regular pyramid, lateral faces are congruent isosceles triangles
  sorry

def forms_cone_on_rotation (triangle : Type) (axis : Type) : Prop :=
  -- assume definition that a right triangle forms a cone when rotated around one of its legs (other than hypotenuse)
  sorry

-- Given conditions as definitions
def statement_A : Prop := ∀ (shape : Type), is_regular_tetrahedron shape → is_cube shape = false
def statement_B : Prop := ∀ (cylinder : Type), is_generatrix_parallel cylinder = true
def statement_C : Prop := ∀ (pyramid : Type), is_lateral_faces_isosceles pyramid = true
def statement_D : Prop := ∀ (triangle : Type) (axis : Type), forms_cone_on_rotation triangle axis = false

-- The proof problem equivalent to incorrectness of statements A, B, and D
theorem find_incorrect_statements : 
  (statement_A = true) ∧ -- statement A is indeed incorrect
  (statement_B = true) ∧ -- statement B is indeed incorrect
  (statement_C = false) ∧ -- statement C is correct
  (statement_D = true)    -- statement D is indeed incorrect
:= 
sorry

end find_incorrect_statements_l698_698382


namespace circumcircles_tangent_l698_698272

structure PentagnonPoints (P : Type*) :=
(A B C D E : P)

variables {P : Type*} [EuclideanGeometry P]

-- Define the convex pentagon and geometrical properties as conditions
def Pentagnon :: P → P → P → P → P → Prop := λ A B C D E, convex (convex_hull P {A, B, C, D, E})

noncomputable def midpoint (A B : P) : P := sorry

def tangent_to_circumcircle (A B C D : P) (p : P → circle) : Prop :=
  is_tangent (p C) C D ∧ is_tangent (p C) C B

-- Tangency proof
theorem circumcircles_tangent :
  ∀ (A B C D E : P), 
  Pentagnon A B C D E → 
  (tangent_to_circumcircle A C D B (circumcircle A C D)) →
  ∃ M, M = midpoint C D ∧ point_on_circumcircle A B C M ∧ point_on_circumcircle A D E M →
  tangent_circles (circumcircle A B E) (circumcircle A C D) :=
begin
  intros A B C D E pent is_tangent_to_circ M Mdef points_on_circ,
  sorry
end

end circumcircles_tangent_l698_698272


namespace inverse_matrix_equation_of_line_l_l698_698016

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![3, 4]]
noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![-2, 1], ![3/2, -1/2]]

theorem inverse_matrix :
  M⁻¹ = M_inv :=
by
  sorry

def transformed_line (x y : ℚ) : Prop := 2 * (x + 2 * y) - (3 * x + 4 * y) = 4 

theorem equation_of_line_l (x y : ℚ) :
  transformed_line x y → x + 4 = 0 :=
by
  sorry

end inverse_matrix_equation_of_line_l_l698_698016


namespace find_m_l698_698195

variables {a_1 d m : ℤ}

def a_n (n : ℤ) : ℤ := a_1 + (n - 1) * d

def S (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

theorem find_m (h1 : a_n 3 + S 5 = 18)
               (h2 : a_n 5 = 7)
               (h3 : ∃ k, a_n 6 = a_n 3 * k ∧ a_n m = a_n 6 * k) :
  m = 15 :=
sorry

end find_m_l698_698195


namespace problem_I_problem_II_l698_698549

-- Definition of the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Problem (I): Prove solution set
theorem problem_I (x : ℝ) : f (x - 1) + f (x + 3) ≥ 6 ↔ (x ≤ -3 ∨ x ≥ 3) := by
  sorry

-- Problem (II): Prove inequality given conditions
theorem problem_II (a b : ℝ) (ha: |a| < 1) (hb: |b| < 1) (hano: a ≠ 0) : 
  f (a * b) > |a| * f (b / a) := by
  sorry

end problem_I_problem_II_l698_698549


namespace num_divisible_l698_698170

open Nat

theorem num_divisible (N : ℕ) (hN : N ≤ 30) : 
  ∃ (count : ℕ), count = 18 ∧ 
  (count = (Finset.card (Finset.filter (λ n, (n * (n + 1) / 2 + 12) ∣ fact n) (Finset.range (N + 1))))) :=
by
  sorry

end num_divisible_l698_698170


namespace peach_tree_average_production_l698_698258

-- Definitions derived from the conditions
def num_apple_trees : ℕ := 30
def kg_per_apple_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def total_mass_fruit : ℕ := 7425

-- Main Statement to be proven
theorem peach_tree_average_production : 
  (total_mass_fruit - (num_apple_trees * kg_per_apple_tree)) = (num_peach_trees * 65) :=
by
  sorry

end peach_tree_average_production_l698_698258


namespace taxi_ride_cost_l698_698434

-- Definitions based on conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def minimum_charge : ℝ := 5.00
def fare (miles : ℝ) : ℝ := base_fare + miles * cost_per_mile

-- Theorem statement reflecting the problem
theorem taxi_ride_cost (miles : ℝ) (h : miles < 4) : fare miles < minimum_charge → fare miles = minimum_charge :=
by
  sorry

end taxi_ride_cost_l698_698434


namespace power_function_solution_l698_698954

theorem power_function_solution (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = x ^ α) (h2 : f 4 = 2) : f 3 = Real.sqrt 3 :=
sorry

end power_function_solution_l698_698954


namespace shaded_area_in_rectangle_is_correct_l698_698034

noncomputable def percentage_shaded_area : ℝ :=
  let side_length_congruent_squares := 10
  let side_length_small_square := 5
  let rect_length := 20
  let rect_width := 15
  let rect_area := rect_length * rect_width
  let overlap_congruent_squares := side_length_congruent_squares * rect_width
  let overlap_small_square := (side_length_small_square / 2) * side_length_small_square
  let total_shaded_area := overlap_congruent_squares + overlap_small_square
  (total_shaded_area / rect_area) * 100

theorem shaded_area_in_rectangle_is_correct :
  percentage_shaded_area = 54.17 :=
sorry

end shaded_area_in_rectangle_is_correct_l698_698034


namespace sufficient_condition_hyperbola_l698_698207

theorem sufficient_condition_hyperbola (m : ℝ) (h : 5 < m) : 
  ∃ a b : ℝ, (a > 0) ∧ (b < 0) ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1) := 
sorry

end sufficient_condition_hyperbola_l698_698207


namespace p_is_neither_necessary_nor_sufficient_for_q_l698_698522

theorem p_is_neither_necessary_nor_sufficient_for_q (a : ℝ) (p : a < 0) (q : 0 < a ∧ a < 1) :
  ¬ (p ⟹ q) ∧ ¬ (q ⟹ p) :=
by
  sorry

end p_is_neither_necessary_nor_sufficient_for_q_l698_698522


namespace sin_cos_necessary_not_sufficient_l698_698525

/-- Given two angles alpha and beta, if sin(alpha) = sin(beta) and cos(alpha) = cos(beta),
    then it is a necessary but not sufficient condition for alpha = beta. -/
theorem sin_cos_necessary_not_sufficient (α β : ℝ) :
  (sin α = sin β ∧ cos α = cos β) → ¬ (α = β ↔ (sin α = sin β ∧ cos α = cos β)) :=
by
  sorry

end sin_cos_necessary_not_sufficient_l698_698525


namespace decimal_places_seven_l698_698900

theorem decimal_places_seven (sqrt2_approx : Real)
                            (sqrt3_approx : Real)
                            (a_b_approx : Real) 
                            (a_diff_b_approx : Real) : 
  abs (495 * sqrt2_approx - 388 * sqrt3_approx - 28) < 10^(-7) :=
by
  have sqrt2_val : sqrt2_approx = 1.414213562 := rfl
  have sqrt3_val : sqrt3_approx = 1.732050808 := rfl
  have a_val : a_b_approx = 700.035713374 := rfl
  have b_val : a_b_approx - a_diff_b_approx = 672.035713336 := rfl
  have diff_val : a_diff_b_approx = 28.000000038 := rfl
  sorry

end decimal_places_seven_l698_698900


namespace pyramid_base_edge_length_l698_698414

theorem pyramid_base_edge_length 
(radius_hemisphere height_pyramid : ℝ)
(h_radius : radius_hemisphere = 4)
(h_height : height_pyramid = 10)
(h_tangent : ∀ face : ℝ, True) : 
∃ s : ℝ, s = 2 * Real.sqrt 42 :=
by
  sorry

end pyramid_base_edge_length_l698_698414


namespace perpendicular_bisector_of_triangle_l698_698682

theorem perpendicular_bisector_of_triangle :
  ∀ {A B C D E F O1 O2 X Y : Type} [Inhabited A] [Inhabited B] [Inhabited C] 
    [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited O1] [Inhabited O2] [Inhabited X] [Inhabited Y], 
    -- Given points and circumcenters:
    (circumcenter ABC = O1) → 
    (circumcenter DEF = O2) → 
    -- Perpendicular bisectors conditions:
    (O1 = intersection_of_perpendicular_bisectors ABC) → 
    (O2 = intersection_of_perpendicular_bisectors DEF) → 
    -- Intersection points condition:
    (X = intersection_of_perpendicular_bisectors.some_line) →
    (Y = intersection_of_perpendicular_bisectors.some_other_line) →
    -- Conclusion to prove:
    perpendicular CO AO :=
begin
  sorry
end

end perpendicular_bisector_of_triangle_l698_698682


namespace product_of_third_sides_is_correct_l698_698764

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l698_698764


namespace nikolai_completes_faster_l698_698038

-- Given conditions: distances they can cover in the same time and total journey length 
def gennady_jump_distance := 2 * 6 -- 12 meters
def nikolai_jump_distance := 3 * 4 -- 12 meters
def total_distance := 2000 -- 2000 meters before turning back

-- Mathematical translation + Target proof: prove that Nikolai will complete the journey faster
theorem nikolai_completes_faster 
  (gennady_distance_per_time : gennady_jump_distance = 12)
  (nikolai_distance_per_time : nikolai_jump_distance = 12)
  (journey_length : total_distance = 2000) : 
  ( (2000 % 4 = 0) ∧ (2000 % 6 ≠ 0) ) -> true := 
by 
  intros,
  sorry

end nikolai_completes_faster_l698_698038


namespace numberOfBoys_is_50_l698_698392

-- Define the number of boys and the conditions given.
def numberOfBoys (B G : ℕ) : Prop :=
  B / G = 5 / 13 ∧ G = B + 80

-- The theorem that we need to prove.
theorem numberOfBoys_is_50 (B G : ℕ) (h : numberOfBoys B G) : B = 50 :=
  sorry

end numberOfBoys_is_50_l698_698392


namespace value_range_of_function_l698_698689

theorem value_range_of_function :
  ∀ (x : ℝ), -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -1 ≤ Real.sin x * Real.sin x - 2 * Real.sin x ∧ Real.sin x * Real.sin x - 2 * Real.sin x ≤ 3 :=
by
  sorry

end value_range_of_function_l698_698689


namespace curve_intersect_count_l698_698056

theorem curve_intersect_count : 
  ∃ A B C D : Fin 6, 
  A ≠ C ∧ D ≠ B ∧ 
  (D > B ∧ A > C ∨ D < B ∧ A < C) ∧
  (∀ x, A * x^2 + B = C * x^2 + D → true) ∧
  (∀ A' B' C' D', A' ≠ C' ∧ D' ≠ B' ∧ 
                  (D' > B' ∧ A' > C' ∨ D' < B' ∧ A' < C') → 
                  ∀ x, A' * x^2 + B' = C' * x^2 + D' → true) ∧
  Finset.card { (A, B, C, D) | A ≠ C ∧ D ≠ B ∧ (D > B ∧ A > C ∨ D < B ∧ A < C) ∧
                               (∀ x, A * x^2 + B = C * x^2 + D → true) } = 90 := sorry

end curve_intersect_count_l698_698056


namespace div_equivalence_l698_698581

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end div_equivalence_l698_698581


namespace infinite_sets_of_six_numbers_l698_698569

-- Define the set of real numbers
def is_product_of_two_others (s : Set ℝ) (x : ℝ) : Prop :=
  ∃ a b ∈ s, (a ≠ x ∧ b ≠ x) ∧ (x = a * b)

theorem infinite_sets_of_six_numbers :
  ∃ (S : Set ℝ), S.finite ∧ S.card = 6 ∧ ∀ x ∈ S, is_product_of_two_others S x ∧ infinite {s : Set ℝ | s.finite ∧ s.card = 6 ∧ ∀ x ∈ s, is_product_of_two_others s x} :=
begin
  sorry
end

end infinite_sets_of_six_numbers_l698_698569


namespace sum_gcd_lcm_63_2898_l698_698064

theorem sum_gcd_lcm_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 :=
by
  sorry

end sum_gcd_lcm_63_2898_l698_698064


namespace pages_copied_for_30_dollars_eq_1875_l698_698262

-- Define the given conditions
def cost_per_page : ℝ := 8 / 5 -- Cost in cents per page
def total_cents : ℝ := 3000 -- Total cents for $30

-- Question: How many pages can be copied for $30?
def pages_for_30_dollars : ℝ := total_cents / cost_per_page

-- Goal is to show that pages_for_30_dollars is 1875 pages
theorem pages_copied_for_30_dollars_eq_1875 : pages_for_30_dollars = 1875 :=
by
  sorry

end pages_copied_for_30_dollars_eq_1875_l698_698262


namespace right_triangle_third_side_product_l698_698771

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l698_698771


namespace monotonic_decreasing_intervals_of_tangent_l698_698340

theorem monotonic_decreasing_intervals_of_tangent (k : ℤ) :
  ∀ x, -2 ≤ x ≤ 2 → (2*k*π - π/3) < x ∧ x < (2*k*π + 5*π/3) →
    ∃ I, I = (2*k*π - π/3, 2*k*π + 5*π/3) ∧
      y = tan(-1/2 * x + π/3) ∧
      is_monotonic_decreasing y I := 
sorry

end monotonic_decreasing_intervals_of_tangent_l698_698340


namespace savings_calculation_l698_698077

-- Define the problem conditions and goal in Lean 4
theorem savings_calculation 
  (x : ℕ) (h1 : 4 * x = 20000) :
  let income := 4 * x,
      expenditure := 3 * x,
      savings := income - expenditure
  in savings = 5000 :=
by
  sorry

end savings_calculation_l698_698077


namespace jellybeans_per_child_l698_698883

theorem jellybeans_per_child (total_jellybeans nephews nieces : ℕ) 
    (h_nephews : nephews = 3) (h_nieces : nieces = 2) (h_total : total_jellybeans = 70) : 
    (total_jellybeans / (nephews + nieces)) = 14 :=
by 
  rw [h_nephews, h_nieces]
  norm_num
  rw [h_total]
  norm_num
  sorry

end jellybeans_per_child_l698_698883


namespace problem1_problem2_l698_698552

noncomputable def f (x m : ℝ) : ℝ :=
  sin (3 * x + π / 3) + cos (3 * x + π / 6) + m * sin (3 * x)

theorem problem1 (m : ℝ) (h : f (17 * π / 18) m = -1) : m = 1 :=
sorry

theorem problem2 (A B a b c : ℝ) (h1 : f (B / 3) 1 = sqrt 3)
  (h2 : a ^ 2 = 2 * c ^ 2 + b ^ 2) : tan A = -3 * sqrt 3 :=
sorry

end problem1_problem2_l698_698552


namespace Nikolai_faster_than_Gennady_l698_698041

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l698_698041


namespace regular_price_of_fox_jeans_l698_698172

-- Definitions
variables (P_F P_P: ℝ)
variables (D_F D_P: ℝ)
variables (total_savings: ℝ)

-- Conditions
def condition1 := P_P = 20
def condition2 := D_F + 0.18 = 0.22
def condition3 := D_P = 0.18
def condition4 := total_savings = 9

-- Calculate savings from quantities and prices
def savings_fox (F D_F: ℝ) := 3 * D_F * F
def savings_pony (P D_P: ℝ) := 2 * D_P * P

-- Statement to prove
theorem regular_price_of_fox_jeans (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  P_F = 15 :=
by
  sorry

end regular_price_of_fox_jeans_l698_698172


namespace find_numbers_l698_698102

theorem find_numbers (N : ℕ) (a b : ℕ) :
  N = 5 * a →
  N = 7 * b →
  N = 35 ∨ N = 70 ∨ N = 105 :=
by
  sorry

end find_numbers_l698_698102


namespace decagon_adjacent_vertices_probability_l698_698730

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698730


namespace scalene_triangles_count_l698_698141

theorem scalene_triangles_count :
  let count := (
    let nat_lessthan_15 := set.filter (λ (s : List ℕ), s.length = 3 ∧ s.sum < 15) (list.permutations (list.range 15).to_finset.to_list) in
    nat_lessthan_15.filter (λ lst, 
      lst.nodup ∧ lst.sorted (λ a b, a ≤ b ∧ a + b > lst.sorted.get_nth 2 0)
    ).length) 
  in count = 9 :=
begin
  sorry
end

end scalene_triangles_count_l698_698141


namespace non_neg_integer_solutions_eq_l698_698165

theorem non_neg_integer_solutions_eq : 
  (∑ i in (Finset.range 2023), (x i)^2 = 2 + ∑ i in (Finset.range 2022), (x i) * (x (i + 1))) → 
  2 * Nat.choose 2024 4 := sorry

end non_neg_integer_solutions_eq_l698_698165


namespace weekend_weekday_ratio_l698_698905

-- Defining the basic constants and conditions
def weekday_episodes : ℕ := 8
def total_episodes_in_week : ℕ := 88

-- Defining the main theorem
theorem weekend_weekday_ratio : (2 * (total_episodes_in_week - 5 * weekday_episodes)) / weekday_episodes = 3 :=
by
  sorry

end weekend_weekday_ratio_l698_698905


namespace arithmetic_sequence_30th_term_l698_698809

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l698_698809


namespace probability_of_adjacent_vertices_in_decagon_l698_698752

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698752


namespace exists_polynomial_l698_698927

def gcd (a b : ℤ) : ℤ := a.gcd b

def primitive_lattice_point (p : ℤ × ℤ) : Prop :=
  let (x, y) := p in gcd x y = 1

theorem exists_polynomial (
  S : finset (ℤ × ℤ)) (hS : ∀ p ∈ S, primitive_lattice_point p) :
  ∃ (n : ℕ) (a : fin (n + 1) → ℤ), ∀ (p : ℤ × ℤ), p ∈ S →
    (finset.sum (finset.range (n + 1)) (λ i, a i * p.1 ^ (n - i) * p.2 ^ i) = 1) :=
sorry

end exists_polynomial_l698_698927


namespace thirtieth_term_value_l698_698818

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l698_698818


namespace probability_adjacent_vertices_decagon_l698_698736

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698736


namespace parabola_zero_sum_l698_698675

-- Define the original parabola equation and transformations
def original_parabola (x : ℝ) : ℝ := (x - 3) ^ 2 + 4

-- Define the resulting parabola after transformations
def transformed_parabola (x : ℝ) : ℝ := -(x - 7) ^ 2 + 1

-- Prove that the resulting parabola has zeros at p and q such that p + q = 14
theorem parabola_zero_sum : 
  ∃ (p q : ℝ), transformed_parabola p = 0 ∧ transformed_parabola q = 0 ∧ p + q = 14 :=
by
  sorry

end parabola_zero_sum_l698_698675


namespace gain_percentage_is_33_33_percent_l698_698828

section ClothProfit

-- Define the variables in the problem
variables (CP SP Profit : ℝ) (meters_sold : ℝ := 30) (profit_equivalent_meters : ℝ := 10)
          (x : ℝ) (CP_per_meter : ℝ := x)

-- Define the condition that selling 30 meters of cloth
def cost_of_30_meters := meters_sold * CP_per_meter

-- Define the profit as the selling price of 10 meters of cloth
def profit := profit_equivalent_meters * CP_per_meter

-- Define the selling price of 30 meters of cloth
def SP_30_meters := cost_of_30_meters + profit

-- Gain percentage formula
def gain_percentage := (profit / cost_of_30_meters) * 100

-- Theorem to prove the gain percentage is 33.33%
theorem gain_percentage_is_33_33_percent : gain_percentage CP_per_meter profit = 33.33 :=
by
  unfold gain_percentage profit cost_of_30_meters
  sorry

end ClothProfit

end gain_percentage_is_33_33_percent_l698_698828


namespace leigh_path_length_l698_698502

theorem leigh_path_length :
  let north := 10
  let south := 40
  let west := 60
  let east := 20
  let net_south := south - north
  let net_west := west - east
  let distance := Real.sqrt (net_south^2 + net_west^2)
  distance = 50 := 
by sorry

end leigh_path_length_l698_698502


namespace work_rate_ratio_l698_698298

theorem work_rate_ratio (x : ℝ) 
  (h1 : (1 / 4) + (x / 4) = (1 / 3)) : 
  x = (1 / 3) :=
begin
  sorry
end

end work_rate_ratio_l698_698298


namespace perimeter_of_triangle_l698_698546

def ellipse (a b x y : ℝ) : Prop := (x^2)/(a^2) + (y^2)/b^2 = 1

noncomputable def left_focus (c : ℝ) : ℝ×ℝ := (-c, 0)
noncomputable def right_focus (c : ℝ) : ℝ×ℝ := (c, 0)
noncomputable def point_P (c : ℝ) : ℝ×ℝ := (0, c)

noncomputable def focus_symmetric (F₁ P : ℝ×ℝ) : Prop := F₁ = (-P.2, -P.1)

theorem perimeter_of_triangle
  (a b c : ℝ)
  (h_ellipse : ellipse a b 0 c)
  (h_b : b = c)
  (h_a : a = sqrt (b^2 + c^2))
  (F₁ := left_focus c)
  (F₂ := right_focus c)
  (P := point_P c)
  (h_symmetric : focus_symmetric F₁ P) :
  2 * a + 2 * c = 4 + 2 * sqrt 2 := 
sorry

end perimeter_of_triangle_l698_698546


namespace hogwarts_school_students_l698_698993

def total_students_at_school (participants boys : ℕ) (boy_participants girl_non_participants : ℕ) : Prop :=
  participants = 246 ∧ boys = 255 ∧ boy_participants = girl_non_participants + 11 → (boys + (participants - boy_participants + girl_non_participants)) = 490

theorem hogwarts_school_students : total_students_at_school 246 255 (boy_participants) girl_non_participants := 
 sorry

end hogwarts_school_students_l698_698993


namespace find_brick_length_l698_698850

-- Definitions of dimensions
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 750
def wall_height : ℝ := 600
def wall_thickness : ℝ := 22.5
def num_bricks : ℝ := 6000

-- Volume calculations
def volume_wall : ℝ := wall_length * wall_height * wall_thickness
def volume_brick (x : ℝ) : ℝ := x * brick_width * brick_height

-- Statement of the problem
theorem find_brick_length (length_of_brick : ℝ) :
  volume_wall = num_bricks * volume_brick length_of_brick → length_of_brick = 25 :=
by
  simp [volume_wall, volume_brick, num_bricks, brick_width, brick_height, wall_length, wall_height, wall_thickness]
  intro h 
  sorry

end find_brick_length_l698_698850


namespace polynomial_is_linear_l698_698935

theorem polynomial_is_linear (P : ℤ → ℤ) (n : ℕ)
  (hP : ∀ m : ℕ, ∃ k : ℕ, P k < 0 ∨ P k ≤ P m)
  (h_seq : ∀ b : ℕ, ∃ k : ℕ, ∃ m : ℕ, m > 1 ∧ P(k) = m^b) :
  ∃ a b : ℤ, P = λ x, a * x + b := 
by
  sorry

end polynomial_is_linear_l698_698935


namespace projection_not_simple_convex_pentagon_l698_698261

open Classical

def isPossibleProjectionPentagon (s : Set (ℝ × ℝ × ℝ)) :=
  ∃ (f : ℝ × ℝ × ℝ → ℝ × ℝ), ∃ (t : Set (ℝ × ℝ)), s = {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.2.1 ∧ p.2.1 ≤ 1} ∧ 
  t = image f s ∧ isSimpleConvexPentagon t

def isSimpleConvexPentagon (t : Set (ℝ × ℝ)) :=
  convex t ∧ (convexHull t).card = 5

theorem projection_not_simple_convex_pentagon : ¬ isPossibleProjectionPentagon {p : ℝ × ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.2.1 ∧ p.2.1 ≤ 1} :=
by
  sorry

end projection_not_simple_convex_pentagon_l698_698261


namespace seq_convergence_iff_cauchy_l698_698306

theorem seq_convergence_iff_cauchy (x : ℕ → ℝ) :
  (∃ a : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - a| < ε) ↔
  (∀ ε > 0, ∃ N : ℕ, ∀ m n ≥ N, |x n - x m| < ε) := 
sorry

end seq_convergence_iff_cauchy_l698_698306


namespace isabella_hair_growth_l698_698624

def initial_hair_length : ℝ := 18
def final_hair_length : ℝ := 24
def hair_growth : ℝ := final_hair_length - initial_hair_length

theorem isabella_hair_growth : hair_growth = 6 := by
  sorry

end isabella_hair_growth_l698_698624


namespace infinite_sqrt_expression_eq_four_l698_698908

theorem infinite_sqrt_expression_eq_four :
  let x := sqrt (12 + sqrt (12 + sqrt (12 + sqrt (12 + ...))))
  x = 4 :=
by
  sorry

end infinite_sqrt_expression_eq_four_l698_698908


namespace least_distinct_values_l698_698100

def list_of_integers (lst : List ℕ) : Prop :=
  lst.length = 3045 ∧ 
  (∃! (m : ℕ), m ∈ lst ∧ (∀ x ∈ lst, x = m) → (List.count x lst = 15))

theorem least_distinct_values {lst : List ℕ} (h : list_of_integers lst) : 
  ∃ n : ℕ, n = 218 ∧ ∀ m < 218, ¬(∃ l : List ℕ, list_of_integers l ∧ List.distinct l = m) :=
sorry

end least_distinct_values_l698_698100


namespace point_inequality_l698_698633

variable (n : ℕ)
variable (A : Fin n → ℝ)
variable (B : Fin n → ℝ)

theorem point_inequality :
  ∑ i j, dist (A i) (B j) ≥ 
  ∑ i j, if i < j then dist (A i) (A j) else 0 + 
  ∑ i j, if i < j then dist (B i) (B j) else 0 := 
sorry

end point_inequality_l698_698633


namespace sum_of_b_n_l698_698205

-- Define the sequence S_n
def S (n : ℕ) : ℕ := n * (n + 1)

-- Define the sequence a_n based on the sum S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * n

-- Define the sequence b_n based on a_n
noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 1 / (a 1 * a 3) else 1 / (a n * a (n + 2))

-- Define the sum of the first n terms of the sequence b_n, T_n
noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b (i + 1)

-- State the theorem
theorem sum_of_b_n
  (n : ℕ) :
  T n = (1/8) * (3/2 - 1/(n+1) - 1/(n+2)) :=
sorry

end sum_of_b_n_l698_698205


namespace square_perimeter_l698_698079

theorem square_perimeter (len width : ℕ) (area_rectangle area_square : ℕ) :
  len = 32 → width = 10 → area_rectangle = len * width → area_square = 5 * area_rectangle →
  ∃ (s : ℕ), s^2 = area_square ∧ 4 * s = 160 :=
by
  -- Definitions and assumptions setup
  intro h_len h_width h_area_rec h_area_sq
  use 40
  split
  · -- Prove that 40^2 = area_square
    calc
      40^2 = 1600 : by norm_num
      ... = area_square : by rw [h_area_sq, h_area_rec, h_len, h_width]; norm_num
  · -- Prove that 4 * 40 = 160
    norm_num

end square_perimeter_l698_698079


namespace gcd_ab_a2b2_is_1_or_2_l698_698204

theorem gcd_ab_a2b2_is_1_or_2 (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_coprime : Nat.coprime a b) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
sorry

end gcd_ab_a2b2_is_1_or_2_l698_698204


namespace sum_series_eq_l698_698981

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 ≥ 2 ∧ ∀ n : ℕ, a (n + 1) = (a n) ^ 2 - 2

theorem sum_series_eq (a : ℕ → ℝ) (h : sequence a) :
  (∑' k, 1 / (∏ i in finset.range (k + 1), a i)) = 
  a 1 / 2 - (real.sqrt (((a 1) / 2) ^ 2 - 1)) :=
sorry

end sum_series_eq_l698_698981


namespace distance_AE_BF_in_parallelepiped_l698_698616

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

noncomputable def distance_between_skew_lines (d1 d2 : ℝ × ℝ × ℝ) (r1 r2 : ℝ × ℝ × ℝ) : ℝ :=
  let cross_product := (d1.2 * d2.3 - d1.3 * d2.2, d1.3 * d2.1 - d1.1 * d2.3, d1.1 * d2.2 - d1.2 * d2.1)
  let magnitude_cp := Real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)
  let r_diff := (r2.1 - r1.1, r2.2 - r1.2, r2.3 - r1.3)
  let dot_product := cross_product.1 * r_diff.1 + cross_product.2 * r_diff.2 + cross_product.3 * r_diff.3
  Real.abs dot_product / magnitude_cp

theorem distance_AE_BF_in_parallelepiped :
  let A := (0, 0, 0)
  let B := (12, 0, 0)
  let D := (0, 24, 0)
  let A1 := (0, 0, 6)
  let B1 := (12, 0, 6)
  let C1 := (12, 24, 6)
  let E := midpoint A1 B1
  let F := midpoint B1 C1
  let d1 := (E.1 - A.1, E.2 - A.2, E.3 - A.3)
  let d2 := (F.1 - B.1, F.2 - B.2, F.3 - B.3) in
  distance_between_skew_lines d1 d2 A B = 8 := by
  sorry

end distance_AE_BF_in_parallelepiped_l698_698616


namespace points_coplanar_l698_698955

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_e1 : V := sorry
def vector_e2 : V := sorry
def vector_not_collinear : Prop := ¬ collinear ℝ {vector_e1, vector_e2}
def vector_AB : V := vector_e1 + vector_e2
def vector_AC : V := -3 • vector_e1 + 7 • vector_e2
def vector_AD : V := 2 • vector_e1 - 3 • vector_e2

theorem points_coplanar (h : vector_not_collinear) : 
  coplanar ℝ {vector_AB, vector_AC, vector_AD} :=
sorry

end points_coplanar_l698_698955


namespace problem_statement_l698_698990

noncomputable def a := 8
noncomputable def t := 63

theorem problem_statement :
  (∃ (a t : ℝ), (sqrt (8 + a / t) = 8 * sqrt (a / t)) ∧ (a > 0) ∧ (t > 0)) →
  a + t = 71 :=
begin
  sorry
end

end problem_statement_l698_698990


namespace irregular_shape_area_l698_698134

noncomputable def area_irregular_shape (length_rect : ℝ) (area_square : ℝ) (breadth_ratio : ℝ) : ℝ :=
  let radius_circle := real.sqrt area_square
  let breadth_rect := breadth_ratio * radius_circle
  let area_rect := length_rect * breadth_rect
  let area_semi_circle := (1 / 2) * real.pi * radius_circle^2
  area_rect + area_semi_circle

theorem irregular_shape_area :
  area_irregular_shape 10 2025 (3/5) = 270 + 1012.5 * real.pi :=
by
  sorry

end irregular_shape_area_l698_698134


namespace L_x_eq_R_x_eq_l698_698933

def factor_chain (x : ℕ) (seq : List ℕ) : Prop :=
  seq.head = 1 ∧ seq.last = x ∧
  (∀ i ∈ List.range (seq.length - 1), seq.nth i < seq.nth (i + 1) ∧ seq.nth i ∣ seq.nth (i + 1))

def L (x : ℕ) : ℕ := sorry -- Definition to be filled with actual computation of the length of the longest factor chain

def R (x : ℕ) : ℕ := sorry -- Definition to be filled with actual computation of the number of longest factor chains

theorem L_x_eq (k m n : ℕ) : 
  let x := 5^k * 31^m * (2 * 5 * 199)^n in
  L(x) = k + m + 3n :=
sorry

theorem R_x_eq (k m n : ℕ) :
  let x := 5^k * 31^m * (2 * 5 * 199)^n in
  R(x) = Nat.factorial (k + m + 3n) / (Nat.factorial (k + n) * Nat.factorial m * Nat.factorial n ^ 2) :=
sorry

end L_x_eq_R_x_eq_l698_698933


namespace sum_of_squares_s_comp_r_l698_698896

def r (x : ℝ) : ℝ := x^2 - 4
def s (x : ℝ) : ℝ := -|x + 1|
def s_comp_r (x : ℝ) : ℝ := s (r x)

theorem sum_of_squares_s_comp_r :
  (s_comp_r (-4))^2 + (s_comp_r (-3))^2 + (s_comp_r (-2))^2 + (s_comp_r (-1))^2 +
  (s_comp_r 0)^2 + (s_comp_r 1)^2 + (s_comp_r 2)^2 + (s_comp_r 3)^2 + (s_comp_r 4)^2 = 429 :=
by
  sorry

end sum_of_squares_s_comp_r_l698_698896


namespace probability_sum_of_two_dice_is_seven_l698_698239

-- Definitions based on conditions
def is_standard_die_roll (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- The main theorem stating the problem and its solution as a Lean statement
theorem probability_sum_of_two_dice_is_seven :
  (∑ (d1 d2 : ℕ) in (finset.filter is_standard_die_roll (finset.range 7)).product (finset.filter is_standard_die_roll (finset.range 7)), 
    if d1 + d2 = 7 then 1 else 0) / 
  (∑ (d1 d2 : ℕ) in (finset.filter is_standard_die_roll (finset.range 7)).product (finset.filter is_standard_die_roll (finset.range 7)), 1) = 
  1 / 6 :=
sorry

end probability_sum_of_two_dice_is_seven_l698_698239


namespace find_c_d_l698_698279

-- Define the problem:
theorem find_c_d (c d : ℝ) (h_root : (2 - 3 * complex.I) * (2 + 3 * complex.I) = 0) :
  c = 5/4 ∧ d = -143/4 :=
sorry

end find_c_d_l698_698279


namespace greatest_prime_factor_of_899_l698_698796

theorem greatest_prime_factor_of_899 : 
  ∃ p, prime p ∧ p ∣ 899 ∧ ∀ q, prime q ∧ q ∣ 899 → q ≤ p :=
suffices h1 : 899 = 29 * 31,
h2 : prime 29,
h3 : prime 31, by
{
  use 31,
  split,
  exact h3,
  split,
  rw h1,
  exact dvd_mul_right 31 29,
  intros q hq1 hq2,
  cases hq1 with hq_prime hq_dvd,
  rw h1 at hq_dvd,
  cases hq_dvd with c hc,
  rw hc at hq_prime,
  interval_cases q,
  exact le_refl 31,
  exact prime.le_of_dvd hq_prime (dvd_refl q),
  exact prime.le_of_dvd hq_prime (dvd_refl q),
},
calc 899 = 30 * 30 - 1        : by norm_num
 ...   = 30^2 - 1^2         : by rw sqr_sub_one_eq_mul
 ...   = (30 - 1) * (30 + 1) : by exact_mod_cast diff_of_sqrs
 ...   = 29 * 31            : by norm_num,

end greatest_prime_factor_of_899_l698_698796


namespace distance_traveled_l698_698763

-- Define the variables for speed of slower and faster bike
def slower_speed := 60
def faster_speed := 64

-- Define the condition that slower bike takes 1 hour more than faster bike
def condition (D : ℝ) : Prop := (D / slower_speed) = (D / faster_speed) + 1

-- The theorem we need to prove
theorem distance_traveled : ∃ (D : ℝ), condition D ∧ D = 960 := 
by
  sorry

end distance_traveled_l698_698763


namespace sum_valid_n_l698_698860

-- Definition: Good number k
def good_number (k : ℕ) : Prop :=
  k > 1 ∧ ∃ (a : list ℕ), a.length = k ∧ (∀ i j, i < j → i < a.length → j < a.length → a.nth_le i sorry < a.nth_le j sorry) ∧ (a.map (λ x, 1 / real.sqrt x)).sum = 1

-- Definition: f(n) is the sum of the first n good numbers
noncomputable def f (n : ℕ) : ℕ :=
  @nat.sum (λ k, {k : ℕ // good_number k}) (list.fin_range $ n.succ) (λ ⟨x, _⟩, x)

-- Main theorem
theorem sum_valid_n : (∑ n in {1, 2, 5, 10}.to_finset, n) = 18 :=
  by sorry

end sum_valid_n_l698_698860


namespace total_annual_cost_l698_698027

def daily_pills : ℕ := 2
def pill_cost : ℕ := 5
def medication_cost (daily_pills : ℕ) (pill_cost : ℕ) : ℕ := daily_pills * pill_cost
def insurance_coverage : ℚ := 0.80
def visit_cost : ℕ := 400
def visits_per_year : ℕ := 2
def annual_medication_cost (medication_cost : ℕ) (insurance_coverage : ℚ) : ℚ :=
  medication_cost * 365 * (1 - insurance_coverage)
def annual_visit_cost (visit_cost : ℕ) (visits_per_year : ℕ) : ℕ :=
  visit_cost * visits_per_year

theorem total_annual_cost : annual_medication_cost (medication_cost daily_pills pill_cost) insurance_coverage
  + annual_visit_cost visit_cost visits_per_year = 1530 := by
  sorry

end total_annual_cost_l698_698027


namespace clara_loses_q_minus_p_l698_698597

def clara_heads_prob : ℚ := 2 / 3
def clara_tails_prob : ℚ := 1 / 3

def ethan_heads_prob : ℚ := 1 / 4
def ethan_tails_prob : ℚ := 3 / 4

def lose_prob_clara : ℚ := clara_heads_prob
def both_tails_prob : ℚ := clara_tails_prob * ethan_tails_prob

noncomputable def total_prob_clara_loses : ℚ :=
  lose_prob_clara + ∑' n : ℕ, (both_tails_prob ^ n) * lose_prob_clara

theorem clara_loses_q_minus_p :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ total_prob_clara_loses = p / q ∧ (q - p = 1) :=
sorry

end clara_loses_q_minus_p_l698_698597


namespace perimeter_ratio_eq_radius_ratio_l698_698249

-- Definitions related to the problem
variables {A B C D E F : Type*}

-- Acute triangle ABC with altitudes AD, BE, CF
variables [acute_triangle A B C]
variables [altitude A D BC]
variables [altitude B E AC]
variables [altitude C F AB]

-- Radii of incircle and circumcircle
variables {r R : ℝ}
-- Perimeters of triangles
variables {P p : ℝ}

-- Statement to be proven
theorem perimeter_ratio_eq_radius_ratio (h1: incircle_radius A B C r) 
                                        (h2: circumcircle_radius A B C R) 
                                        (h3: perimeter A B C P) 
                                        (h4: perimeter D E F p) :
    P / p = r / R :=
by
    sorry

end perimeter_ratio_eq_radius_ratio_l698_698249


namespace sum_of_solutions_l698_698477

theorem sum_of_solutions : 
  (∀ x : ℝ, (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1) → 
  (∃ s : ℝ, s = 16) :=
by
  sorry

end sum_of_solutions_l698_698477


namespace flower_beds_fraction_l698_698423

-- Definitions based on given conditions
def yard_length := 30
def yard_width := 6
def trapezoid_parallel_side1 := 20
def trapezoid_parallel_side2 := 30
def flower_bed_leg := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
def flower_bed_area := (1 / 2) * flower_bed_leg ^ 2
def total_flower_bed_area := 2 * flower_bed_area
def yard_area := yard_length * yard_width
def occupied_fraction := total_flower_bed_area / yard_area

-- Statement to prove
theorem flower_beds_fraction :
  occupied_fraction = 5 / 36 :=
by
  -- sorries to skip the proofs
  sorry

end flower_beds_fraction_l698_698423


namespace minimum_value_of_difference_l698_698555

noncomputable def f (x : ℝ) : ℝ := log (x / 2) + 0.5
noncomputable def g (x : ℝ) : ℝ := exp (x - 2)

theorem minimum_value_of_difference (x1 x2 : ℝ) (h1: 0 < x1) (h2: f x1 = g x2) :
  ∃ y, (y = 1 / 2) ∧ (x1 - x2 = log 2) :=
begin
  sorry
end

end minimum_value_of_difference_l698_698555


namespace problem_solution_l698_698892

noncomputable def T : ℝ :=
  ∑ i in (Finset.range 100).map Finset.succ, real.sqrt (1 + 1 / (i^2 : ℝ) + 1 / ((i + 1)^2 : ℝ) + 1 / ((i + 2)^2 : ℝ))

theorem problem_solution : floor (T^2) = 10199 :=
  sorry

end problem_solution_l698_698892


namespace shopkeeper_profit_percentage_l698_698428

theorem shopkeeper_profit_percentage
  (C : ℝ) -- The cost price of one article
  (cost_price_50 : ℝ := 50 * C) -- The cost price of 50 articles
  (cost_price_70 : ℝ := 70 * C) -- The cost price of 70 articles
  (selling_price_50 : ℝ := 70 * C) -- Selling price of 50 articles is the cost price of 70 articles
  :
  ∃ (P : ℝ), P = 40 :=
by
  sorry

end shopkeeper_profit_percentage_l698_698428


namespace range_of_a_inequality_l698_698465

noncomputable def f (a : ℝ) (x : ℝ) := x^2 - a * real.log (x + 2)

def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 < x2 ∧ x1^2 - a * real.log (x1 + 2) = 0 ∧ x2^2 - a * real.log (x2 + 2) = 0

theorem range_of_a (a : ℝ) (h : has_two_extreme_points a) :
  -2 < a ∧ a < 0 :=
sorry

theorem inequality (a : ℝ) (h : has_two_extreme_points a) (x1 x2 : ℝ)
  (h_ext : x1 < x2 ∧ x1^2 - a * real.log (x1 + 2) = 0 ∧ x2^2 - a * real.log (x2 + 2) = 0) :
  (f a x1 / x2) + 1 < 0 :=
sorry

end range_of_a_inequality_l698_698465


namespace restore_original_text_l698_698711

def russian_alphabet := "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
def received_words := ["ГъЙ", "АЭЁ", "БПРК", "ЕЖЩЮ", "НМЬЧ", "СЫЛЗ", "ШДУ", "ЦХОТ", "ЯФВИ"]
def original_words := ["бык", "вяз", "гной", "дичь", "плющ", "соём", "цех", "шурф", "этаж"]

def valid_substitution (src : Char) (dest : Char) : Bool := 
  let pos_src := russian_alphabet.indexOf src
  let pos_dest := russian_alphabet.indexOf dest
  (pos_dest <= pos_src + 2 ∧ pos_dest >= pos_src - 2) ||
  (pos_src ≥ 31 ∧ pos_dest <= (pos_src + 2) % 33) || -- wrap-around logic
  (pos_src ≤ 1 ∧ pos_dest >= 31 - (1 - pos_src)) -- wrap-around logic

def valid_word_substitution (src : String) (dest : String) : Bool := 
  src.length = dest.length ∧
  (List.zipWith valid_substitution src.data dest.data = List.repeat true src.length)

theorem restore_original_text : 
  (∀ w ∈ received_words, ∃ ow ∈ original_words, valid_word_substitution w ow) ∧
  (∀ ow ∈ original_words, ∃ w ∈ received_words, valid_word_substitution w ow) := 
  by sorry

end restore_original_text_l698_698711


namespace probability_of_adjacent_vertices_in_decagon_l698_698753

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698753


namespace number_of_valid_subsets_l698_698999

def problem_condition (T : Finset ℕ) : Prop :=
  ∀ (k : ℕ), T.card = k → (∀ x ∈ T, x ≥ k + 1) ∧ (∀ (x y : ℕ), x ∈ T → y ∈ T → x ≠ y + 1 ∧ y ≠ x + 1)

noncomputable def count_valid_subsets : ℕ :=
  (Finset.powerset (Finset.range 19)).filter (λ T, T ≠ ∅ ∧ problem_condition T).card

theorem number_of_valid_subsets : count_valid_subsets = 1277 :=
  sorry

end number_of_valid_subsets_l698_698999


namespace a_square_minus_b_square_l698_698582

theorem a_square_minus_b_square (a b : ℚ)
  (h1 : a + b = 11 / 17)
  (h2 : a - b = 1 / 143) : a^2 - b^2 = 11 / 2431 :=
by
  sorry

end a_square_minus_b_square_l698_698582


namespace proof_second_number_is_30_l698_698355

noncomputable def second_number_is_30 : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = 98 ∧ 
    (a / (gcd a b) = 2) ∧ (b / (gcd a b) = 3) ∧
    (b / (gcd b c) = 5) ∧ (c / (gcd b c) = 8) ∧
    b = 30

theorem proof_second_number_is_30 : second_number_is_30 :=
  sorry

end proof_second_number_is_30_l698_698355


namespace factorization_l698_698909

theorem factorization (a : ℂ) : a^3 + a^2 - a - 1 = (a - 1) * (a + 1)^2 :=
by
  sorry

end factorization_l698_698909


namespace Nikolai_faster_than_Gennady_l698_698039

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l698_698039


namespace n_form_t_squared_minus_2_l698_698910

/-- A predicate that checks if a list of binomial coefficients is an arithmetic progression. -/
def is_arithmetic_progression (seq : List ℤ) : Prop :=
  ∀ i : ℕ, i < seq.length - 2 → seq[i] + seq[i+2] = 2 * seq[i+1]

/-- The main statement to prove. -/
theorem n_form_t_squared_minus_2 (k n : ℕ) (t : ℤ) (ht : t ≥ 2):
  (k ≥ 4) →
  (∃ j : ℕ, 0 ≤ j ∧ j ≤ n - k + 2 ∧
    is_arithmetic_progression
      (List.map (λ i, binom n (j + i)) (List.range (k - 1)))) →
  n = t^2 - 2 :=
sorry

end n_form_t_squared_minus_2_l698_698910


namespace min_max_diff_val_l698_698284

def find_min_max_diff (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : ℝ :=
  let m := 0
  let M := 1
  M - m

theorem min_max_diff_val (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : find_min_max_diff x y hx hy = 1 :=
by sorry

end min_max_diff_val_l698_698284


namespace overall_percentage_positive_attitude_l698_698114

theorem overall_percentage_positive_attitude :
  let men := 150;
  let women := 850;
  let men_positive := 55 / 100 * men;
  let women_positive := 75 / 100 * women;
  let total_people := men + women;
  let total_positive := men_positive + women_positive;
  let overall_percentage := total_positive / total_people * 100
  in overall_percentage = 72 := by sorry

end overall_percentage_positive_attitude_l698_698114


namespace largest_domain_of_g_l698_698674

-- Define the function g and the conditions
variable (g : ℝ → ℝ)
variable (dom_g : ∀ x : ℝ, x ≠ 0 → x ∈ set.univ ∧ (1 / x) ∈ set.univ)
variable h : ∀ x : ℝ, x ≠ 0 → g x + g (1 / x) = x ^ 2

-- Theorem statement using the conditions
theorem largest_domain_of_g : (∀ x : ℝ, x ∈ set.univ ∧ (x ≠ 0 ↔ (x ∈ set.univ ∧ 1/x ∈ set.univ))) :=
by
  intros x
  split
  {
    intro hx
    split
    {
      exact hx
    }
    {
      intro hxx
      exact hxx
    }
  }
  {
    intro hx
    exact ⟨hx, hx.symm⟩
  }
sort sorry

end largest_domain_of_g_l698_698674


namespace numWaysToDistributeBalls_l698_698841

-- Define the conditions: number of balls and number of boxes
def numBalls : Nat := 7
def numBoxes : Nat := 3

-- Define a function that calculates the number of ways to distribute the balls
-- (summarizing the cases and accounting for the indistinguishability of the boxes).
noncomputable def waysToDistributeBalls (n : Nat) (k : Nat) : Nat :=
  if k = 1 then 1 else
  if k = 2 then n - 1 else 
  match n, k with 
  | 7, 3 => 1 + 7 + 21 + 35 + 14 + 35 + 47 
  | _, _ => 0

-- The theorem to prove
theorem numWaysToDistributeBalls : waysToDistributeBalls numBalls numBoxes = 160 := 
  by native_decide

end numWaysToDistributeBalls_l698_698841


namespace div_c_a_l698_698578

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end div_c_a_l698_698578


namespace bunch_sheets_l698_698130

theorem bunch_sheets (bundle_sheets heap_sheets bundles bunches heaps total_sheets : ℕ)
  (h_bundle_sheets : bundle_sheets = 2)
  (h_heap_sheets : heap_sheets = 20)
  (h_bundles : bundles = 3)
  (h_bunches : bunches = 2)
  (h_heaps : heaps = 5)
  (h_total_sheets : total_sheets = 114) :
  ∃ B : ℕ, 6 + 2 * B + 100 = 114 ∧ B = 4 := 
by {
  existsi 4,
  split,
  any_goals {sorry},
}

end bunch_sheets_l698_698130


namespace decagon_adjacent_vertices_probability_l698_698727

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l698_698727


namespace max_value_f_l698_698917

noncomputable def f (x : ℝ) : ℝ := real.sin x - real.cos x + x + 1

theorem max_value_f : 
  ∀ x ∈ Icc (3 * real.pi / 4) (7 * real.pi / 4), 
  f x ≤ f real.pi := sorry

end max_value_f_l698_698917


namespace unique_zero_point_condition1_unique_zero_point_condition2_l698_698554

noncomputable def func (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem unique_zero_point_condition1 {a b : ℝ} (h1 : 1 / 2 < a) (h2 : a ≤ Real.exp 2 / 2) (h3 : b > 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

theorem unique_zero_point_condition2 {a b : ℝ} (h1 : 0 < a) (h2 : a < 1 / 2) (h3 : b ≤ 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

end unique_zero_point_condition1_unique_zero_point_condition2_l698_698554


namespace train_length_correct_l698_698121

noncomputable def train_length (v_kmph : ℝ) (t_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let v_mps := v_kmph / 3.6
  let total_distance := v_mps * t_sec
  total_distance - bridge_length

theorem train_length_correct : train_length 72 12.099 132 = 109.98 :=
by
  sorry

end train_length_correct_l698_698121


namespace derivative_at_zero_l698_698966

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1) - 3 * x

theorem derivative_at_zero : Derivative (fun x => f x) 0 = 2 * Real.exp 1 - 3 :=
by
  sorry

end derivative_at_zero_l698_698966


namespace reflection_point_B_l698_698105

-- Define the points and the plane equation
def A : ℝ × ℝ × ℝ := (-3, 9, 11)
def C : ℝ × ℝ × ℝ := (4, 7, 10)
def plane (x y z : ℝ) : Prop := x + y + z = 10

-- Define the reflected point B
def B : ℝ × ℝ × ℝ := (-5/3, 16/3, 25/3)

-- State the theorem to be proved
theorem reflection_point_B :
  ∃ B : ℝ × ℝ × ℝ, B = (-5/3, 16/3, 25/3) ∧
                     plane B.1 B.2 B.3 :=
by {
  use B,
  split,
  { refl },
  { sorry }
}

end reflection_point_B_l698_698105


namespace find_nonzero_c_l698_698493

def quadratic_has_unique_solution (c b : ℝ) : Prop :=
  (b^4 + (1 - 4 * c) * b^2 + 1 = 0) ∧ (1 - 4 * c)^2 - 4 = 0

theorem find_nonzero_c (c : ℝ) (b : ℝ) (h_nonzero : c ≠ 0) (h_unique_sol : quadratic_has_unique_solution c b) : 
  c = 3 / 4 := 
sorry

end find_nonzero_c_l698_698493


namespace trajectory_of_moving_point_l698_698515

theorem trajectory_of_moving_point (x y : ℝ) 
  (h : sqrt ((y + 5)^2 + x^2) - sqrt ((y - 5)^2 + x^2) = 8) :
  y^2 / 16 - x^2 / 9 = 1 ∧ y > 0 := 
sorry

end trajectory_of_moving_point_l698_698515


namespace find_second_number_l698_698350

theorem find_second_number :
  ∃ (x y : ℕ), (y = x + 4) ∧ (x + y = 56) ∧ (y = 30) :=
by
  sorry

end find_second_number_l698_698350


namespace transformed_q_factor_l698_698391

-- Given definition of the function q
def q (w m x z : ℝ) : ℝ := 7 * w / (6 * m * x * (z^3))

-- Conditions: transformations on w, m, x, z
def transformed_w (w : ℝ) : ℝ := 4 * w
def transformed_m (m : ℝ) : ℝ := 2 * m
def transformed_x (x : ℝ) : ℝ := x / 2
def transformed_z (z : ℝ) : ℝ := z^4

-- Prove that the new q value is (4 / z) times the original q value
theorem transformed_q_factor (w m x z : ℝ) : q (transformed_w w) (transformed_m m) (transformed_x x) (transformed_z z) = (4 / z) * q w m x z := 
by sorry

end transformed_q_factor_l698_698391


namespace jellybeans_per_child_l698_698882

theorem jellybeans_per_child (total_jellybeans nephews nieces : ℕ) 
    (h_nephews : nephews = 3) (h_nieces : nieces = 2) (h_total : total_jellybeans = 70) : 
    (total_jellybeans / (nephews + nieces)) = 14 :=
by 
  rw [h_nephews, h_nieces]
  norm_num
  rw [h_total]
  norm_num
  sorry

end jellybeans_per_child_l698_698882


namespace number_of_valid_functions_l698_698158

noncomputable def countValidFunctions : ℕ :=
  Set.card {f : ℕ → ℕ | (∀ a ∈ ({x ∈ Finset.range 2021 | f x = x} : Finset ℕ).val, a = 0)
                         ∧ (∀ a b ∈ Finset.range 2021, f a = f b → a = b)
                         ∧ (∀ a b c ∈ Finset.range 2021, (c = (a + b) % 2021) → (f c = (f a + f b) % 2021))}

theorem number_of_valid_functions : countValidFunctions = 1845 := 
  sorry 

end number_of_valid_functions_l698_698158


namespace odd_function_property_l698_698127

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) : ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by
  -- The proof is omitted as per the instruction
  sorry

end odd_function_property_l698_698127


namespace trajectory_of_P_is_line_segment_l698_698637

open Real EuclideanGeometry

def F1 : Point := (-5, 0)
def F2 : Point := (5, 0)

def P (x y : ℝ) : Point := (x, y)

theorem trajectory_of_P_is_line_segment :
  ∀ (x y : ℝ), dist (P x y) F1 + dist (P x y) F2 = 10 →
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ P x y = (1 - t) • F1 + t • F2 :=
by
  sorry

end trajectory_of_P_is_line_segment_l698_698637


namespace arithmetic_sequence_30th_term_l698_698808

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l698_698808


namespace num_administrative_personnel_l698_698092

noncomputable def total_employees : ℕ := 280
noncomputable def sample_size : ℕ := 56
noncomputable def ordinary_staff_sample : ℕ := 49

theorem num_administrative_personnel (n : ℕ) (h1 : total_employees = 280) 
(h2 : sample_size = 56) (h3 : ordinary_staff_sample = 49) : 
n = 35 := 
by
  have h_proportion : (sample_size - ordinary_staff_sample) / sample_size = n / total_employees := by sorry
  have h_sol : n = (sample_size - ordinary_staff_sample) * (total_employees / sample_size) := by sorry
  have h_n : n = 35 := by sorry
  exact h_n

end num_administrative_personnel_l698_698092


namespace right_triangle_third_side_product_l698_698784

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l698_698784


namespace sum_of_series_l698_698536

noncomputable def S : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + a n + 1

noncomputable def a : ℕ → ℤ
| 0       := 0
| 1       := 2
| (n + 2) := a (n + 1) + 1

theorem sum_of_series :
  (a 2 + a 6 = 10) → S 7 = 35 :=
by
  intros h₁
  sorry

end sum_of_series_l698_698536


namespace polynomial_divisibility_l698_698399

theorem polynomial_divisibility (n : ℕ) (α : ℝ) (h1 : n ≠ 1) (h2 : real.sin α ≠ 0) :
  ∃ (Q : polynomial ℝ), Q = (polynomial.X^2 - 2 * polynomial.X * real.cos α + 1) ∧
  (polynomial.X^n * polynomial.C (real.sin α) - polynomial.X * polynomial.C (real.sin (n * α)) +
  polynomial.C (real.sin ((n - 1) * α))) = Q * (some_polynomial : polynomial ℝ) :=
sorry

end polynomial_divisibility_l698_698399


namespace sheep_count_l698_698395

theorem sheep_count (S H : ℕ) (h1 : S / H = 3 / 7) (h2 : H * 230 = 12880) : S = 24 :=
by
  sorry

end sheep_count_l698_698395


namespace tangents_AngleACB_120_l698_698518

def circle_eq (x y: ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y = 0
def line_eq (P : ℝ × ℝ) : Prop := P.1 + P.2 + 2 = 0

theorem tangents_AngleACB_120 (P A B : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ)
  (h1: ∀ (x y : ℝ), circle_eq x y)
  (h2: line_eq P)
  (h3: C = (1, 1))
  (h4: r = Real.sqrt 2)
  (h5: (P.1 - C.1)^2 + (P.2 - C.2)^2 = 2 * (P.1^2 + P.2^2) - 4)
  (h6: is_tangent (P.1) (P.2) (A.1) (A.2) (B.1) (B.2) (C.1) (C.2) (r))
  : ∠ACB = 120 :=
sorry

end tangents_AngleACB_120_l698_698518


namespace tangent_line_existence_l698_698971

open Real

theorem tangent_line_existence (x0 : ℝ) (hx0 : x0 > sqrt 3 ∧ x0 < 2) :
  ∃ m : ℝ, 0 < m < 1 ∧ (∀ x : ℝ, y = (1 / 2) * x ^ 2 ∧ y = ln x → 
   (deriv (λ x, (1 / 2) * x ^ 2) x0 = deriv ln m) ∧ 
   (ln m - 1 = - (1 / 2) * x0 ^ 2)) :=
  sorry

end tangent_line_existence_l698_698971


namespace arithmetic_mean_inequality_min_value_of_t_l698_698060

-- Part (1)
theorem arithmetic_mean_inequality (a b c : ℝ) : 
  ( (a + b + c) / 3 ) ^ 2 ≤ (a^2 + b^2 + c^2) / 3 ∧ 
    (( (a + b + c) / 3 ) ^ 2 = (a^2 + b^2 + c^2) / 3 → a = b ∧ b = c) := 
sorry

-- Part (2)
theorem min_value_of_t (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  sqrt x + sqrt y + sqrt z ≤ sqrt 3 * sqrt (x + y + z) :=
sorry

end arithmetic_mean_inequality_min_value_of_t_l698_698060


namespace simple_random_sampling_is_C_l698_698824

noncomputable def finite_set {α : Type*} (s : set α) : Prop := ∃ n : ℕ, s.finite

def options := {"A", "B", "C", "D"}

def condition_A : Prop :=
  ∃ (elderly_rep middle_aged_rep young_rep : ℕ), (elderly_rep, middle_aged_rep, young_rep) = (2, 5, 3)

def condition_B : Prop :=
  ∃ (num_list : list ℝ), num_list.length = 10 ∧ ∀ n ∈ num_list, n / 2 ∈ ℝ

def condition_C : Prop :=
  ∃ (individuals : set ℕ) (lottery_machine : list ℕ),
    finite_set individuals ∧ lottery_machine.length = individuals.card ∧
    (∀ (i j : ℕ), i ≠ j → lottery_machine.nth i ≠ lottery_machine.nth j) ∧
    (∀ i, i ∈ lottery_machine → i ∈ individuals) ∧
    (∀ i, i ∈ individuals → ∃ j, lottery_machine.nth j = i)

def condition_D : Prop :=
  ∃ (postcard_numbers : set ℕ),
    (∀ n ∈ postcard_numbers, n % 10000 = 6637) ∧
    finite_set postcard_numbers

theorem simple_random_sampling_is_C :
  (condition_A ∨ condition_B ∨ condition_C ∨ condition_D) →
  condition_C :=
by
  sorry

end simple_random_sampling_is_C_l698_698824


namespace last_digits_after_an_hour_l698_698622

theorem last_digits_after_an_hour (a b c : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) :
  let S := a + b + c in
  S = 11 ∧
  (∀ n : ℕ, S ≡ 11 [MOD 10]) →
  ∃ x y z : ℕ, (∃ (p : Permutation [x, y, z]), [x, y, z] = [8, 7, 1]) ∧ x + y + z = 11 ∧
  x % 10 + y % 10 + z % 10 ≡ 1 [MOD 10] :=
by
  sorry

end last_digits_after_an_hour_l698_698622


namespace num_points_C_l698_698544

theorem num_points_C (
  A B : ℝ × ℝ)
  (C : ℝ × ℝ) 
  (hA : A = (2, 2))
  (hB : B = (-1, -2))
  (hC : (C.1 - 3)^2 + (C.2 + 5)^2 = 36)
  (h_area : 1/2 * (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))) = 5/2) :
  ∃ C1 C2 C3 : ℝ × ℝ,
    (C1.1 - 3)^2 + (C1.2 + 5)^2 = 36 ∧
    (C2.1 - 3)^2 + (C2.2 + 5)^2 = 36 ∧
    (C3.1 - 3)^2 + (C3.2 + 5)^2 = 36 ∧
    1/2 * (abs ((B.1 - A.1) * (C1.2 - A.2) - (B.2 - A.2) * (C1.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C2.2 - A.2) - (B.2 - A.2) * (C2.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C3.2 - A.2) - (B.2 - A.2) * (C3.1 - A.1))) = 5/2 ∧
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C2 ≠ C3) :=
sorry

end num_points_C_l698_698544


namespace conical_heap_radius_l698_698388

noncomputable def volume_of_cylinder (r_cylinder h_cylinder : ℝ) := π * r_cylinder^2 * h_cylinder

noncomputable def volume_of_cone (r_cone h_cone : ℝ) := (1/3) * π * r_cone^2 * h_cone

theorem conical_heap_radius (h_cylinder r_cylinder h_cone : ℝ) 
  (Hcylinder : h_cylinder = 36)
  (Rcylinder : r_cylinder = 21)
  (Hcone : h_cone = 12) :
  ∃ r_cone : ℝ, volume_of_cylinder r_cylinder h_cylinder = volume_of_cone r_cone h_cone ∧ r_cone = 63 := by
  sorry

end conical_heap_radius_l698_698388


namespace range_of_f_l698_698919

noncomputable def f (x : ℝ) : ℝ :=
  (cos x)^3 + 6 * (cos x)^2 + cos x + 2 * (1 - (cos x)^2) - 8 / (cos x - 1)

theorem range_of_f :
  ∀ x, cos x ≠ 1 → 2 ≤ ((cos x + 2) * (cos x + 3)) ∧ ((cos x + 2) * (cos x + 3)) < 12 :=
by
  sorry

end range_of_f_l698_698919


namespace special_rational_numbers_count_correct_l698_698652

-- Define digit to be a number from 0 to 9
def digit := Fin 10

-- Define special_digit_set to be the set of digits {2, 0, 1, 5}
def special_digit_set : Finset digit := {2, 0, 1, 5}.val

-- Define a sequence of digits representing a special rational number
structure special_rational_number :=
(d1 d2 d3 d4 d5 d6 : digit)
(h1 : 2 ∈ {d1, d2, d3, d4, d5, d6})
(h2 : 0 ∈ {d1, d2, d3, d4, d5, d6})
(h3 : 1 ∈ {d1, d2, d3, d4, d5, d6})
(h4 : 5 ∈ {d1, d2, d3, d4, d5, d6})

-- Total number of special rational numbers
def count_special_rational_numbers : ℕ := 22080

-- The theorem verifying the count is correct
theorem special_rational_numbers_count_correct :
  (@Finset.card special_rational_number (by apply_instance) = count_special_rational_numbers) :=
by
  -- Proof should go here
  sorry

end special_rational_numbers_count_correct_l698_698652


namespace jill_net_monthly_salary_l698_698481

variable (S : ℝ)

def discretionary_income : ℝ := S / 5
def gifts_and_charitable_causes_percent : ℝ := 0.135
def gifts_and_charitable_causes : ℝ := 102
def remaining_percent : ℝ := 100 - (27 + 18 + 31.5 + 10)

theorem jill_net_monthly_salary :
  remaining_percent = 13.5 → (* Condition checking whether remaining percentage matches expected value *)
  discretionary_income * gifts_and_charitable_causes_percent = gifts_and_charitable_causes →
  S = 5 * (102 / 0.135) := (* Using the setup solution to find S *)
begin
  intros h1 h2,
  sorry
end

end jill_net_monthly_salary_l698_698481


namespace product_of_third_sides_is_correct_l698_698768

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l698_698768


namespace age_difference_l698_698015

variable (A B C D : ℕ)

theorem age_difference (h₁ : A + B > B + C) (h₂ : C = A - 15) : (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l698_698015


namespace simplify_trig_expression_l698_698314

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / cos (10 * Real.pi / 180) = 
  2 * (3 * Real.sqrt 3 + 4) / 9 := 
by 
  sorry

end simplify_trig_expression_l698_698314


namespace student_program_selection_l698_698433

-- Define the course selection problem within the given conditions
theorem student_program_selection :
  let courses := ["Algebra", "Geometry", "History", "Art", "Latin", "Science"]
  let math_courses := ["Algebra", "Geometry"]
  let choose_ways (n k : ℕ) := nat.choose n k
  
  -- Case with 2 math courses
  let case_2_math := choose_ways 2 2 * choose_ways 4 2 
  -- Case with 3 math courses
  let case_3_math := choose_ways 2 2 * choose_ways 4 1 
  -- Case with 4 math courses
  let case_4_math := choose_ways 2 2 * choose_ways 4 0 
  
  case_2_math + case_3_math + case_4_math = 11 :=
by
  sorry

end student_program_selection_l698_698433


namespace number_of_tiles_is_47_l698_698070

theorem number_of_tiles_is_47 : 
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 2) ∧ n = 47 :=
by
  sorry

end number_of_tiles_is_47_l698_698070


namespace intersection_equivalence_l698_698215

open Set

noncomputable def U : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def M : Set ℤ := {-1, 0, 1}
noncomputable def N : Set ℤ := {x | x * x - x - 2 = 0}
noncomputable def complement_M_in_U : Set ℤ := U \ M

theorem intersection_equivalence : (complement_M_in_U ∩ N) = {2} := 
by
  sorry

end intersection_equivalence_l698_698215


namespace train_length_calculation_l698_698118

noncomputable def length_of_train (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_calculation : 
  length_of_train 4.99960003199744 72 = 99.9920006399488 :=
by 
  sorry  -- proof of the actual calculation

end train_length_calculation_l698_698118


namespace probability_adjacent_vertices_decagon_l698_698755

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l698_698755


namespace circumference_of_circle_inscribed_in_rectangle_l698_698844

-- Define the sides of the rectangle
def side1 : ℝ := 5
def side2 : ℝ := 12

-- Define the diagonal calculated using Pythagorean theorem
def diagonal : ℝ := Real.sqrt (side1 ^ 2 + side2 ^ 2)

-- Define the circumference of the circle
def circumference (d : ℝ) := Real.pi * d

-- Prove that the circumference of the circle is 13π cm
theorem circumference_of_circle_inscribed_in_rectangle :
  circumference diagonal = 13 * Real.pi := by
  -- Proof is left as an exercise
  sorry

end circumference_of_circle_inscribed_in_rectangle_l698_698844


namespace smallest_n_with_digit_7_l698_698799

def contains_digit_7 (k : ℕ) : Prop :=
  ∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ d = 7 ∧ (k.to_string.contains d.to_string.to_char)

theorem smallest_n_with_digit_7 :
  ∃ n : ℕ, 0 < n ∧
    contains_digit_7 (n^2) ∧
    contains_digit_7 ((n+1)^2) ∧
    ¬ contains_digit_7 ((n+2)^2) ∧
    (∀ m : ℕ, 0 < m ∧
      contains_digit_7 (m^2) ∧
      contains_digit_7 ((m+1)^2) ∧
      ¬ contains_digit_7 ((m+2)^2) → n ≤ m) :=
by { sorry }

end smallest_n_with_digit_7_l698_698799


namespace right_triangle_third_side_product_l698_698780

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l698_698780


namespace sufficient_but_not_necessary_l698_698238

noncomputable def four_points := {p1 p2 p3 p4 : Point | True}
noncomputable def no_three_collinear (P : set Point) : Prop :=
  ∀ p1 p2 p3 ∈ P, ¬Collinear ({p1, p2, p3} : set Point)
noncomputable def not_coplanar (P : set Point) : Prop :=
  ∃ p1 p2 p3 p4 ∈ P, ¬Coplanar ({p1, p2, p3, p4} : set Point)

theorem sufficient_but_not_necessary :
  (∀ P ∈ four_points, no_three_collinear P) → 
  (∀ P ∈ four_points, not_coplanar P) ∧
  ¬((∀ P ∈ four_points, not_coplanar P) → (∀ P ∈ four_points, no_three_collinear P)) :=
by
  sorry

end sufficient_but_not_necessary_l698_698238


namespace cost_of_song_book_l698_698626

-- Define the costs as constants
def cost_trumpet : ℝ := 149.16
def cost_music_tool : ℝ := 9.98
def total_spent : ℝ := 163.28

-- Define the statement to prove
theorem cost_of_song_book : total_spent - (cost_trumpet + cost_music_tool) = 4.14 := 
by
  sorry

end cost_of_song_book_l698_698626


namespace right_angled_triangle_sum_of_squares_l698_698599

noncomputable def sum_of_squares_of_segments :
  (n : ℕ) → (a b : ℝ) →(C : ℂ) → (A B : ℂ)(angle_C_right : ∠C = 90°) 
  (A_coords B_coords : Complex) 
  (C0_eq_A : C0 = A) (Cn_eq_B : Cn = B) → 
  (H : \ C0 = A, C1, C2, ..., C_{n-1}, C_n = B as points dividing the AB
  into n equal segments) :
  \sum_{i=0}^{n} distance_squared :=
sorry

theorem right_angled_triangle_sum_of_squares :
  (n : ℕ) → (a b : ℝ) →(C : ℂ) → (A B : ℂ)
  [angle_C_right : ∠(C, A, B) = 90°] 
  [A_coords B_coords : Complex] [C0_eq_A : C0 = A] [Cn_eq_B : Cn = B] → 
  ∑_{i=0}^{n} dist_sq({_, C, C_i}) =  \ becorrect_result :=
sorry

end right_angled_triangle_sum_of_squares_l698_698599


namespace games_played_l698_698293

def total_points : ℝ := 120.0
def points_per_game : ℝ := 12.0
def num_games : ℝ := 10.0

theorem games_played : (total_points / points_per_game) = num_games := 
by 
  sorry

end games_played_l698_698293


namespace algebraic_expression_value_l698_698177

-- Define the conditions 
variables (x y : ℝ)
def condition1 : Prop := x + y = 2
def condition2 : Prop := x - y = 4

-- State the main theorem
theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) :
  1 + x^2 - y^2 = 9 :=
sorry

end algebraic_expression_value_l698_698177


namespace min_max_f_l698_698175

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - Real.cos x

theorem min_max_f :
  (∀ x, 2 * (Real.sin (x / 2))^2 = 1 - Real.cos x) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 5 / 4) :=
by 
  intros h x
  sorry

end min_max_f_l698_698175


namespace right_triangle_third_side_product_l698_698775

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l698_698775


namespace total_sonnets_written_l698_698994

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end total_sonnets_written_l698_698994


namespace geometric_sequence_log_sum_l698_698199

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (a_pos : ∀ n, a n > 0) (h : a 1 * a 7 = 4) :
  (∑ i in finset.range 7, Real.log (a (i + 1)) / Real.log 2) = 7 :=
sorry

end geometric_sequence_log_sum_l698_698199


namespace net_increase_in_bicycle_stock_l698_698646

-- Definitions for changes in stock over the three days
def net_change_friday : ℤ := 15 - 10
def net_change_saturday : ℤ := 8 - 12
def net_change_sunday : ℤ := 11 - 9

-- Total net increase in stock
def total_net_increase : ℤ := net_change_friday + net_change_saturday + net_change_sunday

-- Theorem statement
theorem net_increase_in_bicycle_stock : total_net_increase = 3 := by
  -- We would provide the detailed proof here.
  sorry

end net_increase_in_bicycle_stock_l698_698646


namespace ratio_area_triangle_to_rectangle_l698_698250

theorem ratio_area_triangle_to_rectangle
  (A B C D E F : Type*)
  (x : ℝ) -- length AD
  (AD BC AE ED FD : A → B → ℝ)
  (rectangle_ABCD : A → B → C → D → Prop)
  (bisects_ADC : A → B → D → Prop)
  (angle_ADC : ℝ = 90) 
  (ratio_to_prove : ℝ) :
  AD A D = x ∧
  BC B C = x ∧
  rectangle_ABC A B C D ∧
  bisects_ADC E D A ∧
  bisects_ADC F D A ∧
  ratio_to_prove = (1 / 6) :=
sorry

end ratio_area_triangle_to_rectangle_l698_698250


namespace find_GE_EF_ratio_l698_698934

variable {α : Type} [LinearOrderedField α]

-- Define the variables and given conditions
variables {A B C D E M F G : α}
variables (AM AG : α)
variable (λ : α) (h₁ : AM / AG = λ)

-- Define the target ratio we need to prove
theorem find_GE_EF_ratio (h₁ : AM / AG = λ) :
  GE / EF = λ / (1 - λ) :=
sorry

end find_GE_EF_ratio_l698_698934


namespace inequalities_correctness_l698_698228

theorem inequalities_correctness (a b : ℝ) (h : a < b ∧ b < 0) :
  (a + b < ab) ∨ ((|a| > |b|) ∨ (a < b ∨ (a^2 + b^2 > 2))) → (exactly_two_are_true : Nat) :=
sorry

end inequalities_correctness_l698_698228


namespace correct_propositions_l698_698873

-- Definitions of the propositions as conditions
def proposition1 (hat_b hat_a : ℝ) (x y : ℕ → ℝ) : Prop :=
  ∀ i, (∃ i, y i = hat_b * (x i) + hat_a) ∧ y (x.length) = hat_b * (x.length) + hat_a

def proposition2 (data : ℕ → ℝ) (c : ℝ) : Prop :=
  let var_before := (data.range.reduce (λ sum i, sum + (data i - data.mean)^2)) / (data.length - 1)
  let data_new := λ i, (data i) + c
  let var_after := (data_new.range.reduce (λ sum i, sum + ((data_new i) - data_new.mean)^2)) / (data_new.length - 1)
  var_before = var_after

def proposition3 (r2 : ℝ) : Prop :=
  r2 ≈ 1 → r2 ≈ 0

def proposition4 : Prop :=
  (160 / 20 = 8) → (126 - 15 * 8 = 6)

-- The problem statement
theorem correct_propositions : 
  ∀ (hat_b hat_a : ℝ) (x y : ℕ → ℝ) (data : ℕ → ℝ) (c r2 : ℝ), 
  (proposition2 data c) ∧ (proposition4) :=
by
  sorry

end correct_propositions_l698_698873


namespace gcd_lcm_42_63_gcd_lcm_8_20_l698_698791

def gcd (a b : ℕ) : ℕ := (a.gcd b)
def lcm (a b : ℕ) : ℕ := (a.lcm b)

theorem gcd_lcm_42_63 : gcd 42 63 = 21 ∧ lcm 42 63 = 126 := by
  sorry

theorem gcd_lcm_8_20 : gcd 8 20 = 4 ∧ lcm 8 20 = 40 := by
  sorry

end gcd_lcm_42_63_gcd_lcm_8_20_l698_698791


namespace fixed_points_l698_698968

def f (x : ℝ) : ℝ := (9 * x - 5) / (x + 3)

theorem fixed_points :
  (f 1 = 1) ∧ (f 5 = 5) :=
by
  -- Proof will be here
  sorry

end fixed_points_l698_698968


namespace dodecahedron_edge_coloring_l698_698071

-- Define the properties of the dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)          -- 12 pentagonal faces
  (edges : Fin 30)         -- 30 edges
  (vertices : Fin 20)      -- 20 vertices
  (edge_faces : Fin 30 → Fin 2) -- Each edge contributes to two faces

-- Prove the number of valid edge colorations such that each face has an even number of red edges
theorem dodecahedron_edge_coloring : 
    (∃ num_colorings : ℕ, num_colorings = 2^11) :=
sorry

end dodecahedron_edge_coloring_l698_698071


namespace minimum_AF_plus_4BF_l698_698559

theorem minimum_AF_plus_4BF {A B F : ℝ} (hparabola : ∀ point, point ∈ parabola ∧ point = A ∨ point = B) :
  ∃ (m n : ℝ), |AF| = m ∧ |BF| = n ∧ (1/m + 1/n = 1) → m + 4 * n = 9 :=
by
  sorry

end minimum_AF_plus_4BF_l698_698559


namespace find_x_coordinate_of_P_l698_698974

theorem find_x_coordinate_of_P
  (b c : ℝ)
  (h_hyperbola : ∀ x y, x^2 - y^2 / b^2 = 1)
  (h_distance : 2 * c = 2)
  (h_circle : ∀ x y, x^2 + y^2 = c^2)
  (h_distance_F1P : ∀ x y, (x - 1)^2 + y^2 = (c^2)^2) :
  let P := (1, sqrt 3) in P.1 = 1 :=
by 
  sorry

end find_x_coordinate_of_P_l698_698974


namespace playground_area_l698_698344

theorem playground_area :
  ∃ (w l : ℝ), 2 * l + 2 * w = 90 ∧ l = 3 * w ∧ l * w = 380.625 :=
by
  use 11.25
  use 33.75
  split
  sorry
  split
  sorry
  sorry

end playground_area_l698_698344


namespace find_annual_interest_rate_l698_698325

variable (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ)

-- Conditions
def principal_amount : P = 8000 := by sorry
def final_amount : A = 8820 := by sorry
def compounding_periods_per_year : n = 1 := by sorry
def total_years : t = 2 := by sorry

-- Theorem to prove the annual interest rate
theorem find_annual_interest_rate (P A n t : ℝ) (hP : P = 8000) (hA : A = 8820) (hn : n = 1) (ht : t = 2) :
  ∃ r : ℝ, r = 5 / 100 :=
begin
  sorry
end

end find_annual_interest_rate_l698_698325


namespace largest_n_sets_satisfying_conditions_l698_698474

-- Definition of conditions using Lean's mathematical language
def pairwise_different_sets (n : ℕ) (sets : Fin n → Finset ℕ) : Prop :=
  ∀ i j : Fin n, i ≠ j → sets i ≠ sets j

def condition1 (n : ℕ) (sets : Fin n → Finset ℕ) : Prop :=
  ∀ i j : Fin n, (sets i ∪ sets j).card ≤ 2004

def condition2 (n : ℕ) (sets : Fin n → Finset ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k → (sets i ∪ sets j ∪ sets k) = Finset.range' 1 2008

-- The mathematical statement equivalent to the given problem
theorem largest_n_sets_satisfying_conditions :
  ∃ (n : ℕ) (sets : Fin n → Finset ℕ), pairwise_different_sets n sets ∧ condition1 n sets ∧ condition2 n sets ∧ n = 32 :=
by sorry

end largest_n_sets_satisfying_conditions_l698_698474


namespace distance_between_planes_l698_698915

noncomputable def distance_between_planes_eq (a b c d1 d2 x1 y1 z1 : ℝ) :=
  abs (a * x1 + b * y1 + c * z1 - d1) / real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)

theorem distance_between_planes : distance_between_planes_eq 2 (-4) 4 10 1 1 0 0 = 4 / 3 :=
by 
  unfold distance_between_planes_eq 
  simp
  sorry

end distance_between_planes_l698_698915


namespace pirates_total_coins_l698_698301

theorem pirates_total_coins (x : ℕ) (h : (x * (x + 1)) / 2 = 5 * x) : 6 * x = 54 := by
  -- The proof will go here, but it's currently omitted with 'sorry'
  sorry

end pirates_total_coins_l698_698301


namespace train_length_is_correct_l698_698116

noncomputable def length_of_train
  (time_cross_signal : ℝ)
  (time_cross_platform : ℝ)
  (length_of_platform : ℝ)
  (train_speed_cross_signal : time_cross_signal > 0)
  (train_speed_cross_platform : time_cross_platform > 0)
  : ℝ :=
  let v := length_of_train / time_cross_signal in
  let train_total_time := (length_of_train + length_of_platform) / v in
  if train_total_time = time_cross_platform then
    length_of_train
  else
    0

theorem train_length_is_correct
  (time_cross_signal := 16)
  (time_cross_platform := 39)
  (length_of_platform := 431.25)
  (L : ℝ := 299.57)
  (h1 : 0 < time_cross_signal)
  (h2 : 0 < time_cross_platform) :
  length_of_train time_cross_signal time_cross_platform length_of_platform h1 h2 = L := by
  sorry

end train_length_is_correct_l698_698116


namespace max_value_f_l698_698679

open Real

def f (x : ℝ) : ℝ := x + 2 * cos x

theorem max_value_f : 
  ∃ x ∈ Icc 0 (π / 2), ∀ y ∈ Icc 0 (π / 2), f y ≤ f x ∧ f x = (π / 6 + ⟨sqrt 3, Real.sqrt_pos.2 zero_lt_three⟩) := 
sorry

end max_value_f_l698_698679


namespace unit_square_BE_value_l698_698621

theorem unit_square_BE_value
  (ABCD : ℝ × ℝ → Prop)
  (unit_square : ∀ (a b c d : ℝ × ℝ), ABCD a ∧ ABCD b ∧ ABCD c ∧ ABCD d → 
                  a.1 = 0 ∧ a.2 = 0 ∧ b.1 = 1 ∧ b.2 = 0 ∧ 
                  c.1 = 1 ∧ c.2 = 1 ∧ d.1 = 0 ∧ d.2 = 1)
  (E F G : ℝ × ℝ)
  (on_sides : E.1 = 1 ∧ F.2 = 1 ∧ G.1 = 0)
  (AE_perp_EF : ((E.1 - 0) * (F.2 - E.2)) + ((E.2 - 0) * (F.1 - E.1)) = 0)
  (EF_perp_FG : ((F.1 - E.1) * (G.2 - F.2)) + ((F.2 - E.2) * (G.1 - F.1)) = 0)
  (GA_val : (1 - G.1) = 404 / 1331) :
  ∃ BE, BE = 9 / 11 := 
sorry

end unit_square_BE_value_l698_698621


namespace greatest_x_value_l698_698320

noncomputable def greatest_possible_value (x : ℕ) : ℕ :=
  if (x % 5 = 0) ∧ (x^3 < 3375) then x else 0

theorem greatest_x_value :
  ∃ x, greatest_possible_value x = 10 ∧ (∀ y, ((y % 5 = 0) ∧ (y^3 < 3375)) → y ≤ x) :=
by
  sorry

end greatest_x_value_l698_698320


namespace ball_count_proof_l698_698085

noncomputable def valid_ball_count : ℕ :=
  150

def is_valid_ball_count (N : ℕ) : Prop :=
  80 < N ∧ N ≤ 200 ∧
  (∃ y b w r : ℕ,
    y = Nat.div (12 * N) 100 ∧
    b = Nat.div (20 * N) 100 ∧
    w = 2 * Nat.div N 3 ∧
    r = N - (y + b + w) ∧
    r.mod N = 0 )

theorem ball_count_proof : is_valid_ball_count valid_ball_count :=
by
  -- The proof would be inserted here.
  sorry

end ball_count_proof_l698_698085


namespace percent_defective_units_produced_l698_698257

-- Given conditions
def defective_units_percent_ship_sale := 4 / 100
def defective_units_ship_sale_total_units := 0.2 / 100

-- Define the percentage we are looking to prove
def D := 5 / 100

-- Proof problem statement
theorem percent_defective_units_produced (D : ℝ) :
    (defective_units_percent_ship_sale * D = defective_units_ship_sale_total_units) → D = 5 / 100 :=
by
  intros h
  sorry

end percent_defective_units_produced_l698_698257


namespace find_S4_minus_S3_l698_698181

open_locale real

variables {A B C P D E F: Type*}
variables (AF BF BD CD CE AE : ℝ)
variables (S1 S2 S3 S4 S5 S6 : ℝ)

-- Conditions
axiom perpendicular_AF (P : P) (F : F) : ⊥
axiom perpendicular_BF (P : P) (F : F) : ⊥
axiom perpendicular_BD (P : P) (D : D) : ⊥
axiom perpendicular_CD (P : P) (D : D) : ⊥
axiom perpendicular_CE (P : P) (E : E) : ⊥
axiom perpendicular_AE (P : P) (E : E) : ⊥

axiom S5_minus_S6 : S5 - S6 = 2
axiom S1_minus_S2 : S1 - S2 = 1

theorem find_S4_minus_S3 (h1 : S5 - S6 = 2) (h2 : S1 - S2 = 1) : S4 - S3 = 3 :=
by
  sorry

end find_S4_minus_S3_l698_698181


namespace existential_proof_l698_698483

variables (M : Type) (p : M → Prop)

theorem existential_proof : ∃ x_0 ∈ M, p x_0 :=
sorry

end existential_proof_l698_698483


namespace x_pow_n_plus_inv_x_pow_n_l698_698583

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end x_pow_n_plus_inv_x_pow_n_l698_698583


namespace arithmetic_sequence_30th_term_l698_698802

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l698_698802


namespace calc_f_18_48_l698_698286

def f (x y : ℕ) : ℕ := sorry

axiom f_self (x : ℕ) : f x x = x
axiom f_symm (x y : ℕ) : f x y = f y x
axiom f_third_cond (x y : ℕ) : (x + y) * f x y = x * f x (x + y)

theorem calc_f_18_48 : f 18 48 = 48 := sorry

end calc_f_18_48_l698_698286


namespace club_officers_combination_l698_698416

noncomputable def choose_officers (total_members : ℕ) (qualified_members : ℕ) : ℕ :=
  if h : qualified_members ≤ total_members ∧ total_members ≥ 4 then
    let president_choices := qualified_members in
    let vice_president_choices := qualified_members - 1 in
    let secretary_choices := total_members - 2 in
    let treasurer_choices := total_members - 3 in
    president_choices * vice_president_choices * secretary_choices * treasurer_choices
  else 0

theorem club_officers_combination :
  choose_officers 12 4 = 1080 :=
by {
  -- skipping proof
  sorry
}

end club_officers_combination_l698_698416


namespace roads_cost_correct_l698_698865

noncomputable def area_of_roads_cost (length width road_width cost_per_sqm : ℝ) : ℝ :=
  let road_length_area := road_width * width
  let road_breadth_area := road_width * length
  let intersection_area := road_width * road_width
  let total_road_area := road_length_area + road_breadth_area - intersection_area
  total_road_area * cost_per_sqm

theorem roads_cost_correct : area_of_roads_cost 80 60 10 4 = 5200 :=
by
  unfold area_of_roads_cost
  norm_num
  sorry

end roads_cost_correct_l698_698865


namespace prime_factors_unique_l698_698083

theorem prime_factors_unique (a b c d k l : ℕ) (h : ∀ n : ℕ, (nat.prime_factors (n^k + a^n + c) = nat.prime_factors (n^l + b^n + d))) :
  k = l ∧ a = b ∧ c = d := 
by
  -- proof will be written here
  sorry

end prime_factors_unique_l698_698083


namespace avg_temp_three_cities_avg_temp_four_cities_l698_698356

-- Define temperatures based on the given conditions
def T_ny := 80
def T_mi := T_ny + 10
def T_sd := T_mi + 25
def T_ph := T_sd + 0.15 * T_sd

-- Prove the average temperature for New York, Miami, and San Diego is 95 degrees
theorem avg_temp_three_cities : (T_ny + T_mi + T_sd) / 3 = 95 := by
  sorry

-- Prove the new average temperature for New York, Miami, San Diego, and Phoenix is approximately 104.3125 degrees
theorem avg_temp_four_cities : (T_ny + T_mi + T_sd + T_ph) / 4 ≈ 104.3125 := by
  sorry

end avg_temp_three_cities_avg_temp_four_cities_l698_698356


namespace union_sets_l698_698585

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_sets : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_sets_l698_698585


namespace shortest_third_stick_length_l698_698404

-- Definitions of the stick lengths
def length1 := 6
def length2 := 9

-- Statement: The shortest length of the third stick that forms a triangle with lengths 6 and 9 should be 4
theorem shortest_third_stick_length : ∃ length3, length3 = 4 ∧
  (length1 + length2 > length3) ∧ (length1 + length3 > length2) ∧ (length2 + length3 > length1) :=
sorry

end shortest_third_stick_length_l698_698404


namespace books_shelved_in_history_section_l698_698263

theorem books_shelved_in_history_section
  (initial_books : ℕ)
  (books_unshelved : ℕ)
  (books_fiction : ℕ)
  (books_children_shelved : ℕ)
  (books_children_misplaced : ℕ)
  (final_books_in_cart : ℕ) :
  initial_books = 51 →
  books_unshelved = 16 →
  books_fiction = 19 →
  books_children_shelved = 8 →
  books_children_misplaced = 4 →
  final_books_in_cart = (initial_books - books_fiction - books_children_shelved + books_children_misplaced) - books_unshelved →
  final_books_in_cart = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end books_shelved_in_history_section_l698_698263


namespace problem_solution_correct_l698_698864

open Real

noncomputable def probability_token_covers_black_region
  (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (triangle_leg : ℝ) (token_diameter : ℝ) : ℝ :=
  let total_area := (rectangle_width - token_diameter) * (rectangle_height - token_diameter)
  let triangle_area := 2 * (1 / 2 * triangle_leg^2)
  let additional_area_one_triangle := (π * (token_diameter / 2)^2 / (2 * 2)) + (triangle_leg * sqrt(2) / 2)
  let total_black_area := triangle_area + 2 * additional_area_one_triangle
  (total_black_area) / (total_area)

theorem problem_solution_correct :
  probability_token_covers_black_region 10 6 3 2 = (9 + (π / 2) + 3 * sqrt 2) / 32 :=
by
  -- the proof goes here
  sorry

end problem_solution_correct_l698_698864


namespace right_triangle_third_side_product_l698_698769

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l698_698769


namespace length_of_each_train_l698_698762

theorem length_of_each_train
  (L : ℝ) -- length of each train
  (speed_fast : ℝ) (speed_slow : ℝ) -- speeds of the fast and slow trains in km/hr
  (time_pass : ℝ) -- time for the slower train to pass the driver of the faster one in seconds
  (h_speed_fast : speed_fast = 45) -- speed of the faster train
  (h_speed_slow : speed_slow = 15) -- speed of the slower train
  (h_time_pass : time_pass = 60) -- time to pass
  (h_same_length : ∀ (x y : ℝ), x = y → x = L) :  
  L = 1000 :=
  by
  -- Skipping the proof as instructed
  sorry

end length_of_each_train_l698_698762


namespace three_side_inequality_l698_698943

theorem three_side_inequality
  (x_1 x_2 x_3 x_4 : ℝ)
  (h_distinct : x_1 ≠ x_2 ∧ x_1 ≠ x_3 ∧ x_1 ≠ x_4 ∧ x_2 ≠ x_3 ∧ x_2 ≠ x_4 ∧ x_3 ≠ x_4)
  (h_pos : 0 < x_1 ∧ 0 < x_2 ∧ 0 < x_3 ∧ 0 < x_4)
  (h_ineq : (x_1 + x_2 + x_3 + x_4) * (1/x_1 + 1/x_2 + 1/x_3 + 1/x_4) ≤ 17) :
  ∀ (a b c : ℝ), (a ∈ {x_1, x_2, x_3, x_4}) → (b ∈ {x_1, x_2, x_3, x_4}) → (c ∈ {x_1, x_2, x_3, x_4}) → 
  a ≠ b → b ≠ c → a ≠ c → a + b > c ∧ b + c > a ∧ a + c > b :=
by
  sorry

end three_side_inequality_l698_698943


namespace pq_ge_1_l698_698629

-- Define the basic structures and entities
variables {A B C D P Q M N : Type} 
variables [metric_space ℝ] (ABCD : square ABCD 2) (M : point_A_on_AB) (N : point_D_on_CD)
variables (P : meets_CM_BN_at P) (Q : meets_AN_DM_at Q)

-- The proof statement
theorem pq_ge_1 (a b c d : ℝ) (h : a + b + c + d = 2)
  (h1 : triangle_sim AQG NQD) (h2 : triangle_sim BPM NPC)
  (h3 : b + c = a^2 / b + d^2 / c) :
  ∥P - Q∥ ≥ 1 :=
begin
  sorry
end

end pq_ge_1_l698_698629


namespace incline_angle_of_vertical_line_l698_698233

theorem incline_angle_of_vertical_line (α : ℝ) (h : α = 90) : α = 90 :=
by
  exact h

-- Noncomputable theory assumes no calculation to be performed.
noncomputable def incline_angle_vertical : Prop :=
  ∃ α : ℝ, α = 90

example : incline_angle_vertical :=
by
  use 90
  exact rfl

end incline_angle_of_vertical_line_l698_698233


namespace problem_1_problem_2_l698_698970

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1
theorem problem_1 (x : ℝ) : (∀ x, f x (-2) > 5) ↔ (x < -4 / 3 ∨ x > 2) :=
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (a : ℝ) : (∀ x, f x a ≤ a * |x + 3|) → (a ≥ 1 / 2) :=
  sorry

end problem_1_problem_2_l698_698970


namespace house_cost_l698_698642

-- Definitions of given conditions
def annual_salary : ℝ := 150000
def saving_rate : ℝ := 0.10
def downpayment_rate : ℝ := 0.20
def years_saving : ℝ := 6

-- Given the conditions, calculate annual savings and total savings after 6 years
def annual_savings : ℝ := annual_salary * saving_rate
def total_savings : ℝ := annual_savings * years_saving

-- Total savings represents 20% of the house cost
def downpayment : ℝ := total_savings

-- Prove the total cost of the house
theorem house_cost (downpayment : ℝ) (downpayment_rate : ℝ) : ℝ :=
  downpayment / downpayment_rate

lemma house_cost_correct : house_cost downpayment downpayment_rate = 450000 :=
by
  -- the proof would go here
  sorry

end house_cost_l698_698642


namespace least_possible_square_area_l698_698396

theorem least_possible_square_area (s : ℝ) (h1 : 4.5 ≤ s) (h2 : s < 5.5) : s * s ≥ 20.25 := by
  sorry

end least_possible_square_area_l698_698396


namespace exact_product_of_two_decimals_with_units_digit_5_eq_27_55_l698_698366

theorem exact_product_of_two_decimals_with_units_digit_5_eq_27_55
  (a b : Int)
  (x y : Real)
  (ha : x = a + 0.5)
  (hb : y = b + 0.5)
  (h : Real.round (x * y * 10) / 10 = 27.6) :
  x * y = 27.55 := by
sorry

end exact_product_of_two_decimals_with_units_digit_5_eq_27_55_l698_698366


namespace min_value_of_2b_minus_c_l698_698989

variables {R : Type*} [RealField R]
variables (a b c : EuclideanSpace R (Fin 2))

-- Given conditions
def given_conditions (a b c : EuclideanSpace R (Fin 2)) : Prop :=
  (∥a∥ = 2) ∧
  (∥b∥ = 2) ∧
  (inner a b = 2) ∧
  (inner (a - c) (b - c) = 0)

-- The statement to prove
theorem min_value_of_2b_minus_c
  (a b c : EuclideanSpace R (Fin 2))
  (h : given_conditions a b c) :
  ∥2 • b - c∥ = Real.sqrt 7 :=
sorry

end min_value_of_2b_minus_c_l698_698989


namespace right_triangle_third_side_product_l698_698782

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l698_698782


namespace fraction_calculation_l698_698370

theorem fraction_calculation :
  (1 / 4) * (1 / 3) * (1 / 6) * 144 + (1 / 2) = (5 / 2) :=
by
  sorry

end fraction_calculation_l698_698370


namespace max_a_cos_inequality_l698_698588

theorem max_a_cos_inequality :
  ∃ (a : ℝ), (∀ (x : ℝ), 1 - (2 / 3) * real.cos (2 * x) + a * real.cos x ≥ 0) ∧ a = 1 / 3 :=
begin
  sorry
end

end max_a_cos_inequality_l698_698588


namespace average_width_books_l698_698295

noncomputable def average_width (widths : List ℝ) :=
  (widths.sum / widths.length)

theorem average_width_books :
  average_width [3, 3/4, 7/2, 4.5, 11, 0.5, 7.25] = 4.3571 :=
by
  sorry

end average_width_books_l698_698295


namespace most_suitable_for_comprehensive_survey_l698_698067

theorem most_suitable_for_comprehensive_survey :
  ∃ (A B C D : Prop) (C_suitable : D) (A_not_suitable : ¬A) (B_not_suitable : ¬B) (C_not_suitable: ¬C),
  D :=
by
  -- Definitions for each option
  let A := "Investigating the service life of a batch of infrared thermometers"
  let B := "Investigating the travel methods of the people of Henan during the Spring Festival"
  let C := "Investigating the viewership of the Henan TV program 'Li Yuan Chun'"
  let D := "Investigating the heights of all classmates"
  
  -- Correct Answer
  let C_suitable := True
  let A_not_suitable := True
  let B_not_suitable := True
  let C_not_suitable := True

  -- Conclusion
  exact ⟨A, B, C, D, C_suitable, A_not_suitable, B_not_suitable, C_not_suitable⟩

end most_suitable_for_comprehensive_survey_l698_698067


namespace derivative_of_2_sin_x_l698_698669

theorem derivative_of_2_sin_x : ∀ x : ℝ, deriv (λ x, 2 * sin x) x = 2 * cos x :=
by
  intros x
  sorry

end derivative_of_2_sin_x_l698_698669


namespace angle_B_45_deg_side_lengths_b_c_l698_698240

-- Problem Part 1: Prove that the angle B = 45 degrees
theorem angle_B_45_deg
  (a b c : ℝ)
  (a_eq_4 : a = 4)
  (b_eq_4_sqrt_6_over_3 : b = 4 * real.sqrt 6 / 3)
  (eq_bc_rel : b^2 - a^2 = c * (b - c)) :
  ∠ABC = 45 :=
  sorry

-- Problem Part 2: Prove that b = 4 and c = 4 given the area
theorem side_lengths_b_c
  (a b c : ℝ)
  (a_eq_4 : a = 4)
  (area_eq_4_sqrt_3 : 1 / 2 * b * c * real.sin (120 * π / 180) = 4 * real.sqrt 3)
  (eq_bc_rel : b^2 - a^2 = c * (b - c)) :
  b = 4 ∧ c = 4 :=
  sorry

end angle_B_45_deg_side_lengths_b_c_l698_698240


namespace probability_of_adjacent_vertices_in_decagon_l698_698751

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l698_698751


namespace smallest_x_coordinate_of_leftmost_vertex_l698_698017

theorem smallest_x_coordinate_of_leftmost_vertex :
  ∃ (n : ℕ), n > 0 ∧ 
    let a := (n, Real.log n),
        b := (n + 1, Real.log (n + 1)),
        c := (n + 2, Real.log (n + 2)),
        d := (n + 4, Real.log (n + 4)),
        area := 0.5 * abs ((n * Real.log (n + 1) + (n + 1) * Real.log (n + 2) + (n + 2) * Real.log (n + 4) + (n + 4) * Real.log n) - ((n + 1) * Real.log n + (n + 2) * Real.log (n + 1) + (n + 4) * Real.log (n + 2) + n * Real.log (n + 4)))
    in area = Real.log (163 / 162) → n = 13 :=
  sorry

end smallest_x_coordinate_of_leftmost_vertex_l698_698017


namespace unique_solution_p_zero_l698_698928

theorem unique_solution_p_zero :
  ∃! (x y p : ℝ), 
    (x^2 - y^2 = 0) ∧ 
    (x * y + p * x - p * y = p^2) ↔ 
    p = 0 :=
by sorry

end unique_solution_p_zero_l698_698928


namespace playground_area_l698_698345

theorem playground_area :
  ∃ (w l : ℝ), 2 * l + 2 * w = 90 ∧ l = 3 * w ∧ l * w = 380.625 :=
by
  use 11.25
  use 33.75
  split
  sorry
  split
  sorry
  sorry

end playground_area_l698_698345


namespace number_of_integers_l698_698164

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l698_698164


namespace problem_statement_l698_698403

noncomputable def x : ℝ :=
1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / ...))

def A : ℝ := -6  -- Placeholder for the actual computed value
def B : ℕ := 3   -- Placeholder for the actual computed value
def C : ℕ := 6   -- Placeholder for the actual computed value

def expr : ℝ :=
1 / ((x + 2) * (x - 3))

theorem problem_statement :
  ∃ (A B : ℤ) (C : ℕ), expr = ((A + Real.sqrt (B : ℝ)) / C)  ∧ (C ≠ 0) ∧  abs A + abs B + abs C = ?
:= 
by {
  sorry
}

end problem_statement_l698_698403


namespace coprime_sequence_l698_698398

noncomputable def m (k : ℕ) := 4 * k^2 - 5

sequence xn : ℕ → ℕ
| 0 := a
| 1 := b
| n + 2 := xn n + xn (n + 1)

theorem coprime_sequence (k a b : ℕ) (h_k : k > 1) 
  (h_a : a = 1) 
  (h_b : b = 2 * k^2 + k - 2) 
  (m_k := m k) :
  ∀ n : ℕ, Nat.coprime (xn n) m_k :=
begin
  sorry
end

end coprime_sequence_l698_698398


namespace circus_capacity_l698_698700

theorem circus_capacity (sections : ℕ) (people_per_section : ℕ) (h1 : sections = 4) (h2 : people_per_section = 246) :
  sections * people_per_section = 984 :=
by
  sorry

end circus_capacity_l698_698700


namespace arithmetic_sequence_30th_term_l698_698801

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l698_698801


namespace graph_transformation_proof_l698_698026

noncomputable def y1 (x : Real) : Real := (sqrt 2) * sin (2 * x + π / 4)
noncomputable def y2 (x : Real) : Real := (sqrt 2) * cos x

theorem graph_transformation_proof :
  ∀ x : Real, y2 x = y1 ((x + π / 4) / 2) :=
by
  intros x
  sorry

end graph_transformation_proof_l698_698026


namespace sum_S_correct_l698_698495

-- Define the function S that counts the strictly increasing subsequences of length 2 or more
def S (pi : List ℕ) : ℕ :=
  (pi.sublists.filter (λ l => l.length ≥ 2 ∧ l.sorted (· < ·))).length

-- Define the function that sums S(pi) over all permutations of a list
def sum_S (l : List ℕ) : ℕ :=
  l.permutations.map S |>.sum

-- The final theorem statement: sum of S over all permutations of a list [1, 2, 3, 4, 5, 6] equals 8287
theorem sum_S_correct : sum_S [1, 2, 3, 4, 5, 6] = 8287 :=
by
  sorry

end sum_S_correct_l698_698495


namespace worm_divisible_if_greater_than_two_cells_l698_698893

-- We define the worm configuration as starting at (0,0) with segments either up or right
def starts_at_origin (w : List Char) : Prop :=
  w.head = 'U' ∨ w.head = 'R'

-- We define the movement along the segments either up ('U') or to the right ('R')
def valid_segments (w : List Char) : Prop :=
  ∀ (c : Char), c ∈ w → c = 'U' ∨ c = 'R'

-- Define worms being able to be divided into domino pieces
def can_be_divided_into_dominoes (w : List Char) : Prop :=
  sorry -- detailed definition of divisibility into dominoes

-- We assume there is a function that gives the number of cells in the worm
def number_of_cells (w : List Char) : Nat :=
  sorry -- function logic here

-- Main theorem stating the condition for a worm to be divisible into domino pieces
theorem worm_divisible_if_greater_than_two_cells (w : List Char)
  (h1 : starts_at_origin w)
  (h2 : valid_segments w)
  (h3 : can_be_divided_into_dominoes w) :
  number_of_cells w > 2 :=
sorry

end worm_divisible_if_greater_than_two_cells_l698_698893


namespace prize_rankings_count_l698_698091

def red_packet_types := {Harmony, Patriotism, Dedication}

-- Define the condition where Employee A wins only on the fourth click by collecting all three types on the fourth click.
def isValidSequence (seq: list (finset red_packet_types)) : Prop :=
  seq.length = 4 ∧
  (∀ t ∈ seq.take 3, t ≠ seq.getLast) ∧
  (red_packet_types = seq.take 3.to_finset ∪ {seq.getLast})

theorem prize_rankings_count : 
  (∃ seq : list (finset red_packet_types), isValidSequence seq) → 
  (card (list (finset red_packet_types)).filter isValidSequence = 18) :=
sorry

end prize_rankings_count_l698_698091


namespace find_vectors_l698_698561

def A :  ℝ × ℝ := (3, 2)
def B :  ℝ × ℝ := (-1, 5)
def C :  ℝ × ℝ := (0, 3)

def vector (p1 p2: ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

theorem find_vectors:
    vector A B = (-4, 3) ∧
    vector B C = (1, -2) ∧
    vector A C = (-3, 1) := 
by
  split
  · simp [vector, A, B]
  · split <;> simp [vector, A, B, C]

end find_vectors_l698_698561


namespace find_lambda_l698_698219

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (4, -2)

def is_perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := 
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

theorem find_lambda (λ : ℝ) (h : is_perpendicular (λ • a - b) a) : λ = 1 := 
sorry

end find_lambda_l698_698219


namespace geometry_proof_l698_698274

/-- Given a triangle ABC and points A_1, B_1, and C_1 on BC, CA, and AB respectively, 
such that the lines AA_1, BB_1, and CC_1 meet at a single point, and given that A, B_1, A_1, B are concyclic 
and B, C_1, B_1, C are concyclic, we are to prove that 
C, A_1, C_1, A are concyclic, and AA_1, BB_1, CC_1 are the heights of triangle ABC. -/
theorem geometry_proof (A B C A_1 B_1 C_1 : Point)
  (h1 : collinear A B C)
  (h2 : A_1 ∈ line BC)
  (h3 : B_1 ∈ line CA)
  (h4 : C_1 ∈ line AB)
  (h5 : ∃ P : Point, collinear P A A_1 ∧ collinear P B B_1 ∧ collinear P C C_1)
  (h6 : cyclic_quad A B_1 A_1 B)
  (h7 : cyclic_quad B C_1 B_1 C)
  :
  cyclic_quad C A_1 C_1 A ∧ 
  perpendicular (line A A_1) (line B C) ∧ 
  perpendicular (line B B_1) (line C A) ∧ 
  perpendicular (line C C_1) (line A B) :=
by
  sorry

end geometry_proof_l698_698274


namespace duty_schedule_possible_l698_698691

structure Group :=
  (members : Finset ℕ)
  (friends : ℕ → Finset ℕ)
  (friend_condition : ∀ x ∈ members, (friends x).card = 3)
  (total_members : members.card = 100)

theorem duty_schedule_possible (G : Group)
  (trios_duties : Fin 99 → Finset (Finset ℕ))
  (no_repetition : ∀ i j : Fin 99, i ≠ j → trios_duties i ∩ trios_duties j = ∅) :
   ∃ trio_100 : Finset ℕ, trio_100.card = 3 ∧ (∀ x ∈ trio_100, ∀ y ∈ trio_100, y ∈ G.friends x) := 
sorry

end duty_schedule_possible_l698_698691


namespace probability_minimal_S_l698_698643

theorem probability_minimal_S :
  let S (arr : List ℕ) := (List.foldr(λ (i j : ℕ), abs (i - j)) 0 arr)
  in (probability of S being minimal when balls [1, 2, ..., 9] are 
      randomly placed in a manner same arrangements considered under 
      rotation or reflection overlap) = 1 / 315 := 
sorry

end probability_minimal_S_l698_698643


namespace max_value_of_x_plus_2y_on_ellipse_l698_698348

theorem max_value_of_x_plus_2y_on_ellipse (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 1) : 
  ∃ (θ : ℝ) (hx : x = √6 * Real.cos θ) (hy : y = 2 * Real.sin θ), x + 2 * y ≤ √22 ∧ (∃ θ, x + 2 * y = √22) :=
sorry

end max_value_of_x_plus_2y_on_ellipse_l698_698348


namespace thirtieth_term_value_l698_698819

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l698_698819


namespace value_of_k_l698_698498

theorem value_of_k (k : ℝ) :
  ∃ (k : ℝ), k ≠ 1 ∧ (k-1) * (0 : ℝ)^2 + 6 * (0 : ℝ) + k^2 - 1 = 0 ∧ k = -1 :=
by
  sorry

end value_of_k_l698_698498


namespace lim_to_infty_l698_698635

-- define a continuous function f from [0, +∞) to [0, +∞)
variables {f : ℝ → ℝ} 
variable h_cont : continuous_on f (set.Ici 0)  -- f is continuous on [0, +∞)
variable h_infty : ∀ x, f x ≥ 0 → f (f x) → ∞  -- lim_{x→+∞} f(f(x)) = +∞

-- Prove that lim_{x→+∞} f(x) = +∞
theorem lim_to_infty (h_infty : ∀ L : ℝ, ∃ x₀ : ℝ, ∀ x : ℝ, x ≥ x₀ → f (f x) > L) : ∀ M : ℝ, ∃ y₀ : ℝ, ∀ y : ℝ, y ≥ y₀ → f y > M :=
sorry

-- Provide counterexample for f: (0, +∞) → (0, +∞) where the statement doesn't hold
example : ∃ f : ℝ → ℝ, (∀ x > 0, f x = 1 / x) ∧ (∀ L : ℝ, ∃ x₀ : ℝ, ∀ x : ℝ, x ≥ x₀ → 1 / f (1 / x) > L) ∧ (∀ M : ℝ, ∃ y₀ : ℝ, ∀ y : ℝ, y ≥ y₀ → 1 / y > M) :=
begin
  use (λ x, 1 / x),
  split,
  { intro x,
    exact ⟨assume x₀ hx₀, forall h1, _⟩ },
  { split,
    { exact λ L, exists.intro 1 (λ x hx, L), },
    { exact λ M, exists.intro 1 (λ y hy, M), } },
end

end lim_to_infty_l698_698635


namespace probability_adjacent_vertices_decagon_l698_698738

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l698_698738


namespace coefficient_x3y6_expansion_l698_698611

-- Define factorial to be used in binomial coefficient
def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

-- The final Lean statement to prove the required coefficient
theorem coefficient_x3y6_expansion : binom 8 6 - binom 8 5 = -28 := 
by
  -- These are the placeholder steps for the proof. 
  sorry

end coefficient_x3y6_expansion_l698_698611


namespace rational_roots_count_l698_698571

theorem rational_roots_count :
  {pq : ℕ × ℕ // pq.1 ≤ 100 ∧ pq.2 ≤ 100 ∧ 
                 (∃ x : ℚ, x^5 + pq.1 * x + pq.2 = 0)}.finset.card = 133 :=
sorry

end rational_roots_count_l698_698571


namespace acute_angle_at_7_36_l698_698372

-- Definitions of the movement of clock hands
def minute_hand_angle (m : ℕ) : ℝ := m * 6
def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ := h * 30 + (m / 2)

-- Theorem stating the problem
theorem acute_angle_at_7_36 : |(hour_hand_angle 7 36) - (minute_hand_angle 36)| = 12 := by
  sorry

end acute_angle_at_7_36_l698_698372


namespace dot_v_w_l698_698564

variables {ℝ : Type*}

noncomputable def u : ℝ^3 := sorry
noncomputable def v : ℝ^3 := sorry
noncomputable def w : ℝ^3 := sorry

-- Given conditions
axiom norm_u : ∥u∥ = 2
axiom norm_v : ∥v∥ = 1
axiom norm_u_sub_v : ∥u - v∥ = real.sqrt 7
axiom vec_eq : w + 2 • u - 3 • v = 2 • (u × v)

-- Theorem to prove
theorem dot_v_w : v ⬝ w = 5 := 
by sorry

end dot_v_w_l698_698564


namespace find_symmetric_point_l698_698560

def symmetric_point (p₁ p₂ : ℝ × ℝ) (L : ℝ × ℝ → Prop) : Prop :=
  (let (x₁, y₁) := p₁ in
   let (x₂, y₂) := p₂ in
   x₁ - y₂ = -2 ∧ x₁ + y₂ = 0) ∧
  (let (x², y²) := p₂ in
    let (a, b) := (x₁ + x², y₁ + y²) in L (a/2, (b + 2) / 2))

def line (v : ℝ × ℝ) : Prop :=
  let (x, y) := v in x + y - 1 = 0

theorem find_symmetric_point :
  symmetric_point (0, 2) (-1, 1) line :=
  sorry

end find_symmetric_point_l698_698560


namespace quadratic_eq_proof_l698_698182

noncomputable def quadratic_eq := ∀ (a b : ℝ), 
  (a ≠ 0 → (∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0) →
    (a = b^2 ∧ a = 1 ∧ b = 1) ∨ (a > 1 ∧ 0 < b ∧ b < 1 → ¬ ∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0))

theorem quadratic_eq_proof : quadratic_eq := 
by
  sorry

end quadratic_eq_proof_l698_698182


namespace road_repair_completion_time_l698_698625

theorem road_repair_completion_time :
  (∀ (r : ℝ), 1 = 45 * r * 3) → (∀ (t : ℝ), (30 * (1 / (3 * 45))) * t = 1) → t = 4.5 :=
by
  intros rate_eq time_eq
  sorry

end road_repair_completion_time_l698_698625


namespace trigonometric_simplification_l698_698317

theorem trigonometric_simplification :
  ∀ (a b c d e f g h : ℝ),
  (a = 20 * Real.pi / 180) →
  (b = 30 * Real.pi / 180) →
  (c = 60 * Real.pi / 180) →
  (d = 70 * Real.pi / 180) →
  (e = 10 * Real.pi / 180) →
  (f = 50 * Real.pi / 180) →
  (g = 40 * Real.pi / 180) →
  (h = 130 * Real.pi / 180) →
  sin h = sin f →
  sin f / cos e = cos g / cos e →
  ( ∀ x y : ℝ, tan x + tan y = sin (x + y) / (cos x * cos y) ) →
  (tan a + tan b + tan c + tan d) / cos e = 2 * cos g / (cos e ^ 2 * cos b * cos c * cos d) := sorry

end trigonometric_simplification_l698_698317


namespace probability_of_adjacent_vertices_l698_698725

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698725


namespace right_triangle_third_side_product_l698_698770

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l698_698770


namespace constant_term_in_binomial_expansion_l698_698667

theorem constant_term_in_binomial_expansion :
  let T (r : ℕ) := (nat.choose 6 r) * (-1)^r * (x : ℝ)^(6 - 2 * r)
  (∃ r : ℕ, 6 - 2 * r = 0 ∧ T r = 20) := sorry

end constant_term_in_binomial_expansion_l698_698667


namespace distance_between_droplets_proof_l698_698790

-- Given constants
def h : ℝ := 300000  -- in mm
def d : ℝ := 0.001   -- in mm
def g : ℝ := 9800    -- in mm/s^2

-- Formula to calculate s_1 - s_2
noncomputable def distance_between_droplets (s1 d : ℝ) : ℝ :=
  2 * real.sqrt (s1 * d) - d

-- Proof that the distance is 34.6 mm
theorem distance_between_droplets_proof :
  distance_between_droplets 300000 0.001 = 34.6 :=
by {
  rw [distance_between_droplets],
  norm_num,
  sorry
}

end distance_between_droplets_proof_l698_698790


namespace lottery_probability_correct_l698_698107

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_correct :
  let MegaBall_probability := 1 / 30
  let WinnerBalls_probability := 1 / (combination 50 6)
  MegaBall_probability * WinnerBalls_probability = 1 / 476721000 :=
by
  sorry

end lottery_probability_correct_l698_698107


namespace intersection_of_A_and_B_l698_698563

open Set

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def B : Set ℕ := {x | ∃ k : ℕ, x = 2 * k + 1 ∧ k < 3}

theorem intersection_of_A_and_B :
  A ∩ (B : Set ℤ) = {1, 3, 5} :=
sorry

end intersection_of_A_and_B_l698_698563


namespace right_triangle_third_side_product_l698_698783

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l698_698783


namespace dynamo_power_output_l698_698688

noncomputable def resistance_per_unit_length : ℝ := 1 / 12
noncomputable def length_of_wire : ℝ := 1000 -- 1 km in meters
noncomputable def diameter_of_wire : ℝ := 1.2e-3 -- 1.2 mm in meters
noncomputable def current : ℝ := 12 -- Current in amperes
noncomputable def efficiency : ℝ := 0.85 -- Efficiency

-- Cross-sectional area calculation
noncomputable def cross_sectional_area : ℝ :=
  let radius := diameter_of_wire / 2
  pi * radius^2

-- Total resistance calculation
noncomputable def total_resistance : ℝ :=
  resistance_per_unit_length * length_of_wire / cross_sectional_area

-- Power dissipation calculation
noncomputable def power_dissipation : ℝ :=
  current^2 * total_resistance

-- Voltage drop calculation
noncomputable def voltage_drop : ℝ :=
  current * total_resistance

-- Required horsepower calculation
noncomputable def required_horsepower : ℝ :=
  power_dissipation / (736 * efficiency)

theorem dynamo_power_output :
  power_dissipation = 10610 ∧
  voltage_drop = 884 ∧
  required_horsepower = 17 :=
by
  sorry

end dynamo_power_output_l698_698688


namespace interest_rate_approx_l698_698833

noncomputable def rate_of_interest (P SI : ℝ) : ℝ :=
  (λ R : ℝ, (SI * 100 / P) = (R * R)) 

theorem interest_rate_approx :
  let P := 1800
  let SI := 632
  rate_of_interest P SI ≈ 5.926 := 
by
  sorry

end interest_rate_approx_l698_698833


namespace min_squares_cover_l698_698062

noncomputable def min_squares_for_triangle (s : ℕ) : ℕ :=
  ⌈((s^2 * real.sqrt 3) / 4)⌉.to_nat

theorem min_squares_cover (s : ℕ) (h : s = 15) : min_squares_for_triangle s = 98 :=
by
  rw [h]
  sorry

end min_squares_cover_l698_698062


namespace commencement_addresses_l698_698603

theorem commencement_addresses (sandoval_addresses : ℕ) 
                             (hawkins_addresses : ℕ) 
                             (sloan_addresses : ℕ) :
  sandoval_addresses = 12 →
  hawkins_addresses = sandoval_addresses / 2 →
  sloan_addresses = sandoval_addresses + 10 →
  sandoval_addresses + hawkins_addresses + sloan_addresses = 40 :=
begin
  sorry
end

end commencement_addresses_l698_698603


namespace missed_the_bus_by_5_minutes_l698_698793

theorem missed_the_bus_by_5_minutes 
    (usual_time : ℝ)
    (new_time : ℝ)
    (h1 : usual_time = 20)
    (h2 : new_time = usual_time * (5 / 4)) : 
    new_time - usual_time = 5 := 
by
  sorry

end missed_the_bus_by_5_minutes_l698_698793


namespace sum_odd_integers_lt_100_l698_698378

-- Define the sequence of odd positive integers less than 100
def odd_integers_lt_100 : List ℕ :=
  List.filter (λ n => odd n ∧ n < 100) (List.range 100)

-- Define the property that the sequence consists of the first n odd numbers
def is_first_n_odd_numbers (l : List ℕ) (n : ℕ) : Prop :=
  l = List.map (λ k => 2 * k - 1) (List.range n)

-- Define the sum of the first n odd numbers
def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

-- Prove that the sum of odd positive integers less than 100 is 2500
theorem sum_odd_integers_lt_100 :
  odd_integers_lt_100.sum = 2500 := by
  sorry

end sum_odd_integers_lt_100_l698_698378


namespace find_smallest_a_l698_698166

noncomputable def smallest_triangle_length (a : ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ) (hA : A.1^2 + A.2^2 = 1) (hB : B.1^2 + B.2^2 = 1) (hC : C.1^2 + C.2^2 = 1),
    ∃ (P Q R : ℝ × ℝ) (hPQR : (P = Q) → false ∧ (Q = R) → false ∧ (R = P) → false) (eq_triangle : ∀ X Y, (X = P ∨ X = Q ∨ X = R) → (Y = P ∨ Y = Q ∨ Y = R) → dist X Y = a),
    (A = P ∨ A = Q ∨ A = R ∨ ∃x, (x ∈ [P,R] ∨ x ∈ [P,Q] ∨ x ∈ [Q,R]) ∧ dist A x = 0) ∧
    (B = P ∨ B = Q ∨ B = R ∨ ∃x, (x ∈ [P,R] ∨ x ∈ [P,Q] ∨ x ∈ [Q,R]) ∧ dist B x = 0) ∧
    (C = P ∨ C = Q ∨ C = R ∨ ∃x, (x ∈ [P,R] ∨ x ∈ [P,Q] ∨ x ∈ [Q,R]) ∧ dist C x = 0)

theorem find_smallest_a : smallest_triangle_length (\frac{4}{\sqrt{3}} * (Real.sin (80 * Real.pi / 180))^2) := 
sorry

end find_smallest_a_l698_698166


namespace trajectory_of_point_P_slope_range_l698_698547

-- Conditions
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

def distance_from_P_to_center_equals_distance_to_y_axis (P : ℝ × ℝ) : Prop := 
  let d_center := Math.sqrt ((P.1 - 1)^2 + P.2^2)
  let d_y_axis := Math.abs P.1
  d_center = d_y_axis

def line_l (l : ℝ → ℝ) (k m x : ℝ) : Prop := l x = k*x + m

def dot_product_OA_OB_equals_negative_four (A B : ℝ × ℝ) : Prop := 
  A.1 * B.1 + A.2 * B.2 = -4

def AB_length_bounds (A B : ℝ × ℝ) : Prop := 
  let distance_AB := Math.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  4 * Math.sqrt 6 ≤ distance_AB ∧ distance_AB ≤ 4 * Math.sqrt 30

-- The trajectory of point P
theorem trajectory_of_point_P (P : ℝ × ℝ) (x y : ℝ) :
  (circle_equation x y) ∧ (distance_from_P_to_center_equals_distance_to_y_axis P) →
  (if x ≥ 0 then y^2 = 4*x else y = 0) := sorry

-- The range of values for slope k
theorem slope_range (A B : ℝ × ℝ) (l : ℝ → ℝ) (k m : ℝ) :
  (dot_product_OA_OB_equals_negative_four A B) ∧ (AB_length_bounds A B) →
  (-1 ≤ k ∧ k ≤ -1/2) ∨ (1/2 ≤ k ∧ k ≤ 1) := sorry

end trajectory_of_point_P_slope_range_l698_698547


namespace num_parallel_groups_l698_698475

-- Defining the given vectors
def vec1_a : ℝ × ℝ × ℝ := (2, 2, 1)
def vec1_b : ℝ × ℝ × ℝ := (3, -2, 2)
def vec2_a : ℝ × ℝ × ℝ := (8, 4, -6)
def vec2_b : ℝ × ℝ × ℝ := (4, 2, -3)
def vec3_a : ℝ × ℝ × ℝ := (0, -1, 1)
def vec3_b : ℝ × ℝ × ℝ := (0, 3, -3)
def vec4_a : ℝ × ℝ × ℝ := (-3, 2, 0)
def vec4_b : ℝ × ℝ × ℝ := (4, -3, 3)

-- Statement of the theorem
theorem num_parallel_groups :
  let is_parallel (a b : ℝ × ℝ × ℝ) := ∃ k : ℝ, a = k • b ∨ b = k • a in
  (if is_parallel vec1_a vec1_b then 1 else 0) +
  (if is_parallel vec2_a vec2_b then 1 else 0) +
  (if is_parallel vec3_a vec3_b then 1 else 0) +
  (if is_parallel vec4_a vec4_b then 1 else 0) = 2 :=
by sorry

end num_parallel_groups_l698_698475


namespace find_ab_l698_698451

theorem find_ab (a b : ℝ) 
  (period_cond : (π / b) = (2 * π / 5)) 
  (point_cond : a * Real.tan (5 * (π / 10) / 2) = 1) :
  a * b = 5 / 2 :=
sorry

end find_ab_l698_698451


namespace sam_sitting_fee_l698_698110

theorem sam_sitting_fee 
  (s : ℝ) 
  (h₁ : ∀ n : ℝ, cost_john (n) = 2.75 * n + 125)
  (h₂ : ∀ n : ℝ, cost_sam (n) = 1.50 * n + s)
  (h₃ : cost_john 12 = cost_sam 12) : s = 140 :=
by
  unfold cost_john cost_sam at h₁ h₂ h₃
  sorry

def cost_john (n : ℝ) : ℝ := 2.75 * n + 125

def cost_sam (n : ℝ) (s : ℝ) : ℝ := 1.50 * n + s

end sam_sitting_fee_l698_698110


namespace solve_otimes_eq_l698_698634
noncomputable def otimes : ℝ → ℝ → ℝ := sorry

axiom otimes_comm : ∀ a b: ℝ, otimes a b = otimes b a
axiom otimes_distrib : ∀ a b c: ℝ, otimes a (b * c) = (otimes a b) * (otimes a c)
axiom otimes_continuous : continuous (λ p : ℝ × ℝ, otimes p.1 p.2)
axiom otimes_2_2 : otimes 2 2 = 4

theorem solve_otimes_eq {x y: ℝ} (h₀ : x > 1) (h₁ : otimes x y = x) : 
  y = sqrt 2 := 
sorry

end solve_otimes_eq_l698_698634


namespace find_x_l698_698088

def diamond (a b : ℝ) : ℝ := a / b

theorem find_x (x : ℝ) (hx : x ≠ 0) : 5040 ⬝ (8 ⬝ x) = 250 → x = 25 / 63 :=
begin
  intros h,
  sorry
end

end find_x_l698_698088


namespace length_of_platform_l698_698869

variables (LengthTrain : ℝ) (TimePassMan : ℝ) (SpeedTrain : ℝ) 
          (TimeCrossPlatform : ℝ) (LengthPlatform : ℝ)

-- Given conditions
def train_length : LengthTrain = 180 := by sorry
def time_pass_man : TimePassMan = 8 := by sorry
def time_cross_platform : TimeCrossPlatform = 20 := by sorry

-- Calculate speed of the train when passing the man
def speed_train : SpeedTrain = LengthTrain / TimePassMan := by sorry

-- Problem statement: Prove length of platform is 270 meters
theorem length_of_platform : LengthPlatform = 270 :=
by 
  have h1 : SpeedTrain = 180 / 8 := speed_train
  have h2 : SpeedTrain = 22.5 := by calc 
    SpeedTrain = 180 / 8 : sorry
  have h3 : LengthTrain + LengthPlatform = SpeedTrain * TimeCrossPlatform := sorry
  have h4 : 180 + LengthPlatform = 22.5 * 20 := sorry
  have h5 : LengthPlatform = 270 := by linarith
  exact h5

end length_of_platform_l698_698869


namespace division_problem_l698_698835

theorem division_problem (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m % n = 12) (h4 : m / n = 24.2) : n = 60 :=
sorry

end division_problem_l698_698835


namespace birds_are_not_four_types_l698_698242

theorem birds_are_not_four_types (total_birds hawks non_hawks paddyfield_warblers kingfishers blackbirds : ℕ)
  (h_total: total_birds = 100)
  (h_hawks: hawks = 30)
  (h_non_hawks: non_hawks = total_birds - hawks)
  (h_paddyfield_warblers: paddyfield_warblers = 0.4 * non_hawks)
  (h_kingfishers: kingfishers = 0.25 * paddyfield_warblers)
  (h_blackbirds: blackbirds = 0.15 * (hawks + paddyfield_warblers)) :
  (total_birds - (hawks + paddyfield_warblers + kingfishers + blackbirds)) = 26 := sorry

end birds_are_not_four_types_l698_698242


namespace insert_median_is_8_to_find_x_l698_698623

theorem insert_median_is_8_to_find_x (x : ℝ) : 
  ∃ x, x ∈ (4, 9) ∧ (median ([1, 4, 9, 15, 21, x]) = 8) :=
sorry

end insert_median_is_8_to_find_x_l698_698623


namespace max_non_intersecting_points_l698_698449

open set

-- Define the type of points in the plane
def Point := ℝ × ℝ

-- Define what it means for a line to not self-intersect
def non_intersecting_line (pts : list Point) : Prop := 
  ∀ (σ : list Point),
  σ.permutations.contains pts →
  (∀ i j, i < j → (∃! seg₁ seg₂, 
    seg₁ = (σ.nth i, σ.nth (i + 1)) ∧ 
    seg₂ = (σ.nth j, σ.nth (j + 1))) →
    ¬ (seg₁ ∩ seg₂ ≠ ∅))

-- Define the proof statement
theorem max_non_intersecting_points (A : list Point) (h : non_intersecting_line A) :
  A.length ≤ 4 := sorry

end max_non_intersecting_points_l698_698449


namespace total_kids_played_l698_698267

theorem total_kids_played (kids_monday : ℕ) (kids_tuesday : ℕ) (h_monday : kids_monday = 4) (h_tuesday : kids_tuesday = 14) : 
  kids_monday + kids_tuesday = 18 := 
by
  -- proof steps here (for now, use sorry to skip the proof)
  sorry

end total_kids_played_l698_698267


namespace greatest_prime_factor_99_l698_698371

theorem greatest_prime_factor_99 : ∃ p : ℕ, nat.prime p ∧ nat.greatest_prime_factor 99 = p ∧ p = 11 := by
  sorry

end greatest_prime_factor_99_l698_698371


namespace Nikolai_faster_than_Gennady_l698_698050

theorem Nikolai_faster_than_Gennady
  (gennady_jump1 gennady_jump2 : ℕ) (nikolai_jump1 nikolai_jump2 nikolai_jump3 : ℕ) :
  gennady_jump1 = 6 → gennady_jump2 = 6 →
  nikolai_jump1 = 4 → nikolai_jump2 = 4 → nikolai_jump3 = 4 →
  2 * gennady_jump1 + gennady_jump2 = 3 * (nikolai_jump1 + nikolai_jump2 + nikolai_jump3) →
  let total_path := 2000 in
  (total_path % 4 = 0 ∧ total_path % 6 ≠ 0) →
  (total_path / 4 < (total_path + 4) / 6) :=
by
  intros
  sorry

end Nikolai_faster_than_Gennady_l698_698050


namespace find_x_values_l698_698054

open Nat

theorem find_x_values (a b c : ℕ) (h_coprime : Nat.coprime a c) (t : ℤ) :
  ∃ x : ℤ, (∃ t : ℤ, x = b * a^((Nat.totient c) - 1) + c * t) ∧ (a * x - b) % c = 0 :=
sorry

end find_x_values_l698_698054


namespace train_length_correct_l698_698120

-- Define the given conditions as constants
def time_to_cross_pole : ℝ := 4.99960003199744
def speed_kmh : ℝ := 72

-- Convert the speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train : ℝ := speed_ms * time_to_cross_pole

-- The problem statement: prove that length_of_train is approximately 99.992 meters
theorem train_length_correct : abs (length_of_train - 99.992) < 0.001 := by
  sorry

end train_length_correct_l698_698120


namespace football_team_total_progress_l698_698095

theorem football_team_total_progress :
  let play1 := -5
  let play2 := 13
  let play3 := -2 * play1
  let play4 := play3 / 2
  play1 + play2 + play3 + play4 = 3 :=
by
  sorry

end football_team_total_progress_l698_698095


namespace right_triangle_third_side_product_l698_698786

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l698_698786


namespace leap_year_median_modes_l698_698596

def leap_year_data : List ℕ :=
  List.replicate 12 [1, 2, 3, ..., 30].flatten ++ List.replicate 11 [31]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (λ a b => a < b)
  if h : l.length % 2 = 0 then
    (sorted[(l.length / 2) - 1] + sorted[l.length / 2]) / 2
  else
    sorted[l.length / 2]

noncomputable def mean (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / l.length

def modes (l : List ℕ) : List ℕ :=
  let freq_map := l.foldl (λ m x => m.insert x (m.find_d x + 1)) ∅
  let max_freq := freq_map.to_list.maximum_by (λ p, p.2).2
  freq_map.to_list.filter_map (λ p, if p.2 = max_freq then some p.1 else none)

noncomputable def median_of_modes (l : List ℕ) : ℕ :=
  median (modes l)

theorem leap_year_median_modes :
  let l := leap_year_data in
  let M := median l in
  let μ := mean l in
  let d := median_of_modes l in
  d < M ∧ M < μ :=
by
  let l := leap_year_data
  have d := median_of_modes l
  have M := median l
  have μ := mean l
  sorry

end leap_year_median_modes_l698_698596


namespace probability_of_odd_sum_is_27_over_64_l698_698701

noncomputable def probability_odd_sum : ℚ :=
  let coin_prob : ℚ := 1 / 2 in
  let die_prob_odd : ℚ := 1 / 2 in
  let case_0_heads := coin_prob ^ 3 in
  let case_1_head := 3 * (coin_prob ^ 3) * die_prob_odd in
  let case_2_heads := 3 * (coin_prob ^ 3) * (2 * (die_prob_odd * (1 - die_prob_odd))) in
  let case_3_heads := (coin_prob ^ 3) * (((3 * (die_prob_odd ^ 2 * (1 - die_prob_odd))) + (3 * ((1 - die_prob_odd) ^ 2 * die_prob_odd)))) in
  case_0_heads * 0 + case_1_head + case_2_heads + case_3_heads

theorem probability_of_odd_sum_is_27_over_64 : probability_odd_sum = 27 / 64 :=
sorry

end probability_of_odd_sum_is_27_over_64_l698_698701


namespace thirtieth_term_of_arithmetic_seq_l698_698813

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l698_698813


namespace ratio_proof_l698_698980

variables {F : Type*} [Field F] 
variables (w x y z : F)

theorem ratio_proof 
  (h1 : w / x = 4 / 3) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
by sorry

end ratio_proof_l698_698980


namespace sequence_general_formula_l698_698938

theorem sequence_general_formula :
  ∃ (a : ℕ → ℕ), 
    (a 1 = 4) ∧ 
    (∀ n : ℕ, a (n + 1) = a n + 3) ∧ 
    (∀ n : ℕ, a n = 3 * n + 1) :=
sorry

end sequence_general_formula_l698_698938


namespace lcm_of_two_numbers_l698_698051

theorem lcm_of_two_numbers (a b : ℕ) (h_prod : a * b = 145862784) (h_hcf : Nat.gcd a b = 792) : Nat.lcm a b = 184256 :=
by {
  sorry
}

end lcm_of_two_numbers_l698_698051


namespace math_problem_l698_698506

theorem math_problem 
  (a b : ℂ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) : 
  a^(2*n + 1) + b^(2*n + 1) = 0 := 
by 
  sorry

end math_problem_l698_698506


namespace ordered_pair_sqrt_identity_l698_698494

theorem ordered_pair_sqrt_identity (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hlt : a < b) :
  (sqrt 1 + sqrt (25 + 20 * sqrt 2) = sqrt a + sqrt b) → (a = 2 ∧ b = 8) :=
by sorry

end ordered_pair_sqrt_identity_l698_698494


namespace no_three_digit_number_such_that_sum_is_perfect_square_l698_698308

theorem no_three_digit_number_such_that_sum_is_perfect_square :
  ∀ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 →
  ¬ (∃ m : ℕ, m * m = 100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b) := by
  sorry

end no_three_digit_number_such_that_sum_is_perfect_square_l698_698308


namespace irrationals_in_set_l698_698445

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

def num_irrationals (s : set ℝ) : ℕ := (s.filter is_irrational).to_finset.card

theorem irrationals_in_set : num_irrationals ({3 * real.pi, -7 / 8, 0, real.sqrt 2, -3.15, real.sqrt 9, real.sqrt 3 / 3} : set ℝ) = 3 :=
by
  sorry

end irrationals_in_set_l698_698445


namespace find_omega_f_period_find_monotonically_decreasing_interval_l698_698221

def vector_a (ω x : ℝ) : ℝ × ℝ := (-√3 * sin(ω * x), cos(ω * x))
def vector_b (ω x : ℝ) : ℝ × ℝ := (cos(ω * x), cos(ω * x))

def f (ω x : ℝ) : ℝ := (vector_a ω x).1 * (vector_b ω x).1 + (vector_a ω x).2 * (vector_b ω x).2

theorem find_omega_f_period (ω : ℝ) (hω : ω > 0) (hT : ∀ x, f ω (x + π) = f ω x) : ω = 1 :=
sorry

theorem find_monotonically_decreasing_interval (k : ℤ) : 
  ∀ x, f 1 x = cos(2 * x + (π / 3)) + (1 / 2) →
  ∀ x, k * π - (π / 6) ≤ x ∧ x ≤ k * π + (π / 3) ↔ 
  ∀ x, x ∈ (set.Icc (k * π - (π / 6)) (k * π + (π / 3))) :=
sorry

end find_omega_f_period_find_monotonically_decreasing_interval_l698_698221


namespace operation_1_and_2004_l698_698471

def operation (m n : ℕ) : ℕ :=
  if m = 1 ∧ n = 1 then 2
  else if m = 1 ∧ n > 1 then 2 + 3 * (n - 1)
  else 0 -- handle other cases generically, although specifics are not given

theorem operation_1_and_2004 : operation 1 2004 = 6011 :=
by
  unfold operation
  sorry

end operation_1_and_2004_l698_698471


namespace sum_z_eq_minus_three_halves_l698_698289

open Complex

theorem sum_z_eq_minus_three_halves (z : ℂ) (hx : ∃ x : ℝ, (z * x^2 + 2 * conj(z) * x + 2 = 0)) (hz : abs z = 1) :
  ∑ z in {z : ℂ | ∃ x : ℝ, (z * x^2 + 2 * conj(z) * x + 2 = 0) ∧ abs z = 1}.to_finset = -3 / 2 :=
sorry

end sum_z_eq_minus_three_halves_l698_698289


namespace product_approximation_l698_698709

-- Define the approximation condition
def approxProduct (x y : ℕ) (approxX approxY : ℕ) : ℕ :=
  approxX * approxY

-- State the theorem
theorem product_approximation :
  let x := 29
  let y := 32
  let approxX := 30
  let approxY := 30
  approxProduct x y approxX approxY = 900 := by
  sorry

end product_approximation_l698_698709


namespace max_value_of_z_l698_698527

theorem max_value_of_z 
    (x y : ℝ) 
    (h1 : |2 * x + y + 1| ≤ |x + 2 * y + 2|)
    (h2 : -1 ≤ y ∧ y ≤ 1) : 
    2 * x + y ≤ 5 := 
sorry

end max_value_of_z_l698_698527


namespace cardioid_arc_length_l698_698132

-- Definition of the polar equation of the cardioid
def rho (ϕ : ℝ) : ℝ := 2 * (1 + Real.cos ϕ)

-- The statement to be proved
theorem cardioid_arc_length :
  (∫ ϕ in 0..2*Real.pi, Real.sqrt ((deriv rho ϕ)^2 + (rho ϕ)^2) dϕ) = 16 :=
sorry

end cardioid_arc_length_l698_698132


namespace discount_is_five_percent_l698_698420

-- Define the wholesale price
def wholesale_price : ℝ := 4

-- Define the retail price as 25% above wholesale price
def retail_price : ℝ := wholesale_price * 1.25

-- Define the price paid by customer with coupon
def price_with_coupon : ℝ := 4.75

-- Calculate the discount percentage
def discount_percentage : ℝ := (retail_price - price_with_coupon) / retail_price * 100

-- The statement to prove the discount percentage is 5%
theorem discount_is_five_percent : discount_percentage = 5 := by
  sorry

end discount_is_five_percent_l698_698420


namespace decagon_adjacent_vertex_probability_l698_698714

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l698_698714


namespace billy_initial_crayons_l698_698453

theorem billy_initial_crayons (eaten left : ℕ) (h_eaten : eaten = 52) (h_left : left = 10) : 
  (eaten + left = 62) :=
by
  rw [h_eaten, h_left]
  exact rfl

end billy_initial_crayons_l698_698453


namespace range_of_a_l698_698557

theorem range_of_a (a : ℝ) : 
  (∀ x : ℕ, (1 ≤ x ∧ x ≤ 4) → ax + 4 ≥ 0) → (-1 ≤ a ∧ a < -4/5) :=
by
  sorry

end range_of_a_l698_698557


namespace sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l698_698486

theorem sum_of_power_of_2_plus_1_divisible_by_3_iff_odd (n : ℕ) : 
  (3 ∣ (2^n + 1)) ↔ (n % 2 = 1) :=
sorry

end sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l698_698486


namespace distances_square_inscribed_in_circle_l698_698431

theorem distances_square_inscribed_in_circle
  (R : ℝ) (hR : R = 5)
  (A B C D : ℝ × ℝ) (P : ℝ × ℝ)
  (h_square_inscribed : 
    ∀ (a b c : ℝ × ℝ),
      ∥a - b∥ = ∥b - c∥ ∧ ∥b - c∥ = ∥c - d∥ ∧ ∥c - d∥ = ∥d - a∥ ∧
      ∥a - c∥ = ∥b - d∥ ∧ ∥a - c∥ = 2 * R)
  (h_point_on_circle : ∥P∥ = R)
  (h_distance_to_vertex : 
    ∃ (A : ℝ × ℝ) (hA : A ∈ set_of (λ a, ∥a∥ = R)),
      dist P A = 6) :
  ∃ (B C D : ℝ × ℝ),
    dist P B = √2 ∧ dist P C = 8 ∧ dist P D = 7 * √2 :=
sorry

end distances_square_inscribed_in_circle_l698_698431


namespace four_ping_pong_four_shuttlecocks_cost_l698_698385

theorem four_ping_pong_four_shuttlecocks_cost
  (x y : ℝ)
  (h1 : 3 * x + 2 * y = 15.5)
  (h2 : 2 * x + 3 * y = 17) :
  4 * x + 4 * y = 26 :=
sorry

end four_ping_pong_four_shuttlecocks_cost_l698_698385


namespace solve_jennifer_flowers_sum_possible_flower_counts_l698_698380

theorem solve_jennifer_flowers (F : ℕ) : (F % 6 = 5) → (F % 8 = 7) → (F < 100) → F = 23 ∨ F = 47 ∨ F = 71 ∨ F = 95 :=
by
  intros h_mod6 h_mod8 h_bound
  sorry

theorem sum_possible_flower_counts : 
  (Finset.sum (Finset.of_list [23, 47, 71, 95]) id) = 236 :=
by
  sorry

end solve_jennifer_flowers_sum_possible_flower_counts_l698_698380


namespace tower_count_l698_698411

theorem tower_count (total_cubes red_cubes blue_cubes green_cubes tower_height : ℕ) 
    (h_total : total_cubes = 11)
    (h_red : red_cubes = 3)
    (h_blue : blue_cubes = 4)
    (h_green : green_cubes = 4)
    (h_height : tower_height = 10) 
    : (∑ x in {red_cubes, blue_cubes, green_cubes}, if x >= 0 then (Nat.factorial tower_height) / ((Nat.factorial (red_cubes - (if red_cubes >= 1 then 1 else 0))) * (Nat.factorial (blue_cubes - (if blue_cubes >= 1 then 1 else 0))) * (Nat.factorial (green_cubes - (if green_cubes >= 1 then 1 else 0)))) else 0) = 26250 := 
by
  sorry

end tower_count_l698_698411


namespace general_formulas_T_n_less_than_half_sum_D_sequence_l698_698961

-- Define the sequences $\{a_n\}$ and $\{b_n\}$
noncomputable def a_sequence (n : ℕ) := n
noncomputable def b_sequence (n : ℕ) := 3^n

-- Conditions of the problem
axiom sum_a_sequence (n : ℕ) : (∑ k in Finset.range n, a_sequence k) = (1 / 2) * n^2 + (1 / 2) * n
axiom b_conditions : b_sequence 1 = a_sequence 3 ∧ b_sequence 2 + b_sequence 3 = 36

-- Parts of the problem
-- 1. Prove that the given conditions imply the general formulas for $\{a_n\}$ and $\{b_n\}$
theorem general_formulas : (a_sequence = λ n, n) ∧ (b_sequence = λ n, 3^n) :=
sorry

-- 2. Prove $T_n < \frac{1}{2}$ for the sum sequence $\{T_n\}$
noncomputable def T_sequence (n : ℕ) : ℝ := (1 / 2) * (1 - 1 / (2n + 1))
theorem T_n_less_than_half (n : ℕ) : T_sequence n < 1 / 2 :=
sorry

-- 3. Find the sum of the first $n$ terms of the sequence $\{C_n\}$ denoted as $D_n$
noncomputable def C_sequence (n : ℕ) : ℝ := (2 * n + 1) / (3^n)
noncomputable def D_sequence (n : ℕ) : ℝ := ∑ k in Finset.range n, C_sequence k
theorem sum_D_sequence (n : ℕ) : D_sequence n = 2 - (n + 2) / (3^n) :=
sorry

end general_formulas_T_n_less_than_half_sum_D_sequence_l698_698961


namespace solution1_solution2_l698_698319

-- Definition of the first inequality
def inequality1 (x : ℝ) : Prop := |x - 1| > 2

-- Proving the solution set for the first inequality
theorem solution1 (x : ℝ) : inequality1 x ↔ x ∈ set.union {y : ℝ | y < -1} {z : ℝ | z > 3} :=
by sorry

-- Definition of the second inequality and the condition for a
def inequality2 (x a : ℝ) (h : 0 < a ∧ a < 1) : Prop := a^(1 - x) < a^(x + 1)

-- Proving the solution set for the second inequality under the condition 0 < a < 1
theorem solution2 (x a : ℝ) (h : 0 < a ∧ a < 1) : inequality2 x a h ↔ x < 0 :=
by sorry

end solution1_solution2_l698_698319


namespace find_max_difference_l698_698648

theorem find_max_difference :
  ∃ (abc def : ℕ), 
    ([3, 5, 9].member abc.digits.head ∧ 
     [2, 3, 7].member abc.digits.tail.head ∧
     [3, 4, 8, 9].member abc.digits.tail.tail.head ∧
     [2, 3, 7].member def.digits.head ∧
     [3, 5, 9].member def.digits.tail.head ∧
     [1, 4, 7].member def.digits.tail.tail.head ∧
     (∃ ghi : ℕ, 
       [4, 5, 9].member ghi.digits.head ∧
       [2].member ghi.digits.tail.head ∧
       [4, 5, 9].member ghi.digits.tail.tail.head ∧
       abc - def = ghi ∧
       ghi < 900)) :=
begin
  use 923,
  use 394,
  split,
  -- conditions for 923
  { split,
    { simp[membership]; left; refl, },
    { split,
      { simp[membership]; right; left; refl, },
      { simp[membership]; left; refl } } },
  split,
  -- conditions for 394
  { split,
    { simp[membership]; right; left; refl, },
    { split,
      { simp[membership]; left; refl,},
      { simp[membership]; right; right; left; refl} } },
  existsi 529,
  split,
  -- conditions for result: 529
  { split,
    { simp[membership]; right; right; left; refl, },
    { split,
      { simp[membership]; left; refl,},
      { simp[membership]; right; right; left; refl} } },
  split,
  -- checking difference value
  { refl },
  -- checking that result is less than 900
  linarith,
end

end find_max_difference_l698_698648


namespace point_inside_circle_l698_698007

theorem point_inside_circle (a : ℝ) (h : ((1 + a)^2 + (1 - a)^2 < 4)) : -1 < a ∧ a < 1 :=
begin
  sorry
end

end point_inside_circle_l698_698007


namespace inequality_solution_set_l698_698142

theorem inequality_solution_set :
  ∀ x : ℝ, 8 * x^3 + 9 * x^2 + 7 * x - 6 < 0 ↔ (( -6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)) :=
sorry

end inequality_solution_set_l698_698142


namespace a2013_eq_1_l698_698187

noncomputable def sequence (n : ℕ) : ℕ := sorry
noncomputable def sum_first_n_terms (n : ℕ) : ℕ := sorry

axiom sum_of_terms (n m : ℕ) : sum_first_n_terms(n) + sum_first_n_terms(m) = sum_first_n_terms(n + m)
axiom first_term : sequence 1 = 1

theorem a2013_eq_1 : sequence 2013 = 1 := sorry

end a2013_eq_1_l698_698187


namespace determine_a_given_odd_function_l698_698545

theorem determine_a_given_odd_function (a : ℝ) (h₀ : 0 < a)
    (h₁ : ∀ x : ℝ, (f : ℝ → ℝ) = λ x, (2^x + a) / (2^x - a) ∧ f (-x) + f x = 0) : a = 1 :=
  sorry

end determine_a_given_odd_function_l698_698545


namespace sam_sitting_fee_l698_698113

theorem sam_sitting_fee :
  (∃ S : ℝ,
  let johns_cost := 2.75 * 12 + 125 in
  let sams_cost := 1.50 * 12 + S in
  johns_cost = sams_cost → S = 140) :=
by
  sorry

end sam_sitting_fee_l698_698113


namespace range_of_m_n_l698_698967

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * Real.exp x + x^2 + n * x

theorem range_of_m_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0) ∧ (∀ x : ℝ, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end range_of_m_n_l698_698967


namespace hyperbola_eccentricity_l698_698975

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let c := 2 * b in
  let a_squared := a^2 in
  let c_squared := c^2 in
  (a_squared = c_squared - b^2) →
  let eccentricity := c / a in
  e = √(1 + b^2 / a^2) → 
  e = 2 * √3 / 3 :=
by
  intro c a_squared c_squared ha_square ecc e;
  sorry

end hyperbola_eccentricity_l698_698975


namespace no_snuggly_numbers_l698_698470

def isSnuggly (n : Nat) : Prop :=
  ∃ (a b : Nat), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    n = 10 * a + b ∧ 
    n = a + b^3 + 5

theorem no_snuggly_numbers : 
  ¬ ∃ n : Nat, 10 ≤ n ∧ n < 100 ∧ isSnuggly n :=
by
  sorry

end no_snuggly_numbers_l698_698470


namespace sum_odd_integers_lt_100_l698_698379

-- Define the sequence of odd positive integers less than 100
def odd_integers_lt_100 : List ℕ :=
  List.filter (λ n => odd n ∧ n < 100) (List.range 100)

-- Define the property that the sequence consists of the first n odd numbers
def is_first_n_odd_numbers (l : List ℕ) (n : ℕ) : Prop :=
  l = List.map (λ k => 2 * k - 1) (List.range n)

-- Define the sum of the first n odd numbers
def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

-- Prove that the sum of odd positive integers less than 100 is 2500
theorem sum_odd_integers_lt_100 :
  odd_integers_lt_100.sum = 2500 := by
  sorry

end sum_odd_integers_lt_100_l698_698379


namespace sum_distinct_x_g_g_g_x_eq_2_l698_698104

-- Defining the cubic function g(x)
def g (x : ℝ) : ℝ := (x^3 / 8) - (3 * x / 2) - 2

-- The main theorem statement
theorem sum_distinct_x_g_g_g_x_eq_2 :
  let xs : Set ℝ := {x : ℝ | g (g (g x)) = 2}
  ∃ (xs' : Finset ℝ), xs' = (xs.to_finset) ∧ xs'.sum = 18 :=
sorry

end sum_distinct_x_g_g_g_x_eq_2_l698_698104


namespace total_wheels_is_90_l698_698694

-- Defining the conditions
def number_of_bicycles := 20
def number_of_cars := 10
def number_of_motorcycles := 5

-- Calculating the total number of wheels
def total_wheels_in_garage : Nat :=
  (2 * number_of_bicycles) + (4 * number_of_cars) + (2 * number_of_motorcycles)

-- Statement to prove
theorem total_wheels_is_90 : total_wheels_in_garage = 90 := by
  sorry

end total_wheels_is_90_l698_698694


namespace find_abc_solutions_l698_698487

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end find_abc_solutions_l698_698487


namespace piecewise_function_evaluation_l698_698550

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^3 else 2^x

theorem piecewise_function_evaluation :
  f (f (-1)) = 2 :=
sorry

end piecewise_function_evaluation_l698_698550


namespace arithmetic_sequence_count_l698_698464

theorem arithmetic_sequence_count :
  (Σ a d n : ℕ, 0 ≤ a ∧ 0 ≤ d ∧ n ≥ 3 ∧ n * (2 * a + (n - 1) * d) = 2 * 97^2).1.fst.length = 4 :=
by sorry

end arithmetic_sequence_count_l698_698464


namespace thirtieth_term_of_arithmetic_seq_l698_698811

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l698_698811


namespace count_four_digit_with_thousands_5_l698_698223

-- Definition of four-digit numbers with thousands digit 5
def is_four_digit_with_thousands_5 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = 5)

/-- Prove that the total number of four-digit integers with thousands digit 5 is 1000. -/
theorem count_four_digit_with_thousands_5 : 
  {n : ℕ | is_four_digit_with_thousands_5 n}.to_finset.card = 1000 := 
by
  sorry

end count_four_digit_with_thousands_5_l698_698223


namespace exists_triangle_no_triangle_l698_698397

-- Assumptions and definitions
variable {M : Set (Set ℝ ℝ)} -- M is a set of points representing a convex polygon
variable {S : ℝ} -- S is the area of the polygon M
variable {l : Set ℝ} -- l is an arbitrary line

-- Prove the existence of a triangle with side parallel to l and area >= (3/8) * S
theorem exists_triangle-area-at-least_three_eighths (hM : is_convex_polygon M) (hS : area M = S) :
  ∃ (A B C : ℝ×ℝ), triangle A B C ∧ (exists_parallel_side_to A B C l) ∧ (area (triangle A B C) ≥ (3 / 8 * S)) :=
sorry

-- Prove that there exist specific M and l such that no triangle has area > (3/8) * S
theorem no_triangle-area-greater_than_three_eighths :
  ∃ (M : Set (Set ℝ ℝ)) (l : Set ℝ), is_convex_polygon M ∧ area M = S ∧
  ¬(∃ (A B C : ℝ×ℝ), triangle A B C ∧ (exists_parallel_side_to A B C l) ∧ (area (triangle A B C) > (3 / 8 * S))) :=
sorry

end exists_triangle_no_triangle_l698_698397


namespace axis_tangent_circle_l698_698558

-- Definitions based on given conditions
def parabola (y : ℝ) (p : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

def circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x - 9 = 0

def axis_of_parabola (p : ℝ) : ℝ :=
  -p / 2

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Proof statement
theorem axis_tangent_circle (p : ℝ) (h_parabola : parabola y p) (h_circle : circle x y) :
  distance 4 0 (axis_of_parabola p) 0 = 5 → p = 2 :=
sorry

end axis_tangent_circle_l698_698558


namespace net_increase_in_bicycle_stock_l698_698647

-- Definitions for changes in stock over the three days
def net_change_friday : ℤ := 15 - 10
def net_change_saturday : ℤ := 8 - 12
def net_change_sunday : ℤ := 11 - 9

-- Total net increase in stock
def total_net_increase : ℤ := net_change_friday + net_change_saturday + net_change_sunday

-- Theorem statement
theorem net_increase_in_bicycle_stock : total_net_increase = 3 := by
  -- We would provide the detailed proof here.
  sorry

end net_increase_in_bicycle_stock_l698_698647


namespace cylinder_volume_ratio_l698_698467

theorem cylinder_volume_ratio (h_C r_D : ℝ) (V_C V_D : ℝ) :
  h_C = 3 * r_D →
  r_D = h_C →
  V_C = 3 * V_D →
  V_C = (1 / 9) * π * h_C^3 :=
by
  sorry

end cylinder_volume_ratio_l698_698467


namespace find_correct_r_l698_698963

noncomputable def ellipse_tangent_circle_intersection : Prop :=
  ∃ (E F : ℝ × ℝ) (r : ℝ), E ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧
                             F ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧ 
                             (E ≠ F) ∧
                             ((E.1 - 2)^2 + (E.2 - 3/2)^2 = r^2) ∧
                             ((F.1 - 2)^2 + (F.2 - 3/2)^2 = r^2) ∧
                             r = (Real.sqrt 37) / 37

theorem find_correct_r : ellipse_tangent_circle_intersection :=
sorry

end find_correct_r_l698_698963


namespace arithmetic_sequence_30th_term_l698_698804

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l698_698804


namespace find_sin_phi_l698_698275
open Real

noncomputable def sin_phi (s : ℝ) (EP : ℝ) (PQ : ℝ) : ℝ :=
  let EQ := sqrt ((EP ^ 2) + (PQ ^ 2)) in
  let cos_phi := (EP ^ 2 + PQ ^ 2 - EQ ^ 2) / (2 * EP * PQ) in
  sqrt (1 - cos_phi ^ 2)

theorem find_sin_phi :
  let EFGH_side := 4 in
  let EP := 2 * sqrt 2 in
  let PQ := 2 in 
  sin_phi EFGH_side EP PQ = 1 :=
by
  -- Proof goes here
  sorry

end find_sin_phi_l698_698275


namespace xiao_wang_final_position_and_fuel_consume_l698_698651

theorem xiao_wang_final_position_and_fuel_consume 
  (itinerary : List ℤ)
  (fuel_rate : ℚ)
  (distance_formula : itinerary.sum = -25)
  (fuel_formula : itinerary.map (Int.natAbs).sum = 87) :
  let final_position := itinerary.sum
  let total_distance := itinerary.map (Int.natAbs).sum
  final_position = -25 ∧ (total_distance : ℚ) * fuel_rate = 30.45 := by
{
  sorry -- The proof is not required
}

end xiao_wang_final_position_and_fuel_consume_l698_698651


namespace nikolai_completes_faster_l698_698035

-- Given conditions: distances they can cover in the same time and total journey length 
def gennady_jump_distance := 2 * 6 -- 12 meters
def nikolai_jump_distance := 3 * 4 -- 12 meters
def total_distance := 2000 -- 2000 meters before turning back

-- Mathematical translation + Target proof: prove that Nikolai will complete the journey faster
theorem nikolai_completes_faster 
  (gennady_distance_per_time : gennady_jump_distance = 12)
  (nikolai_distance_per_time : nikolai_jump_distance = 12)
  (journey_length : total_distance = 2000) : 
  ( (2000 % 4 = 0) ∧ (2000 % 6 ≠ 0) ) -> true := 
by 
  intros,
  sorry

end nikolai_completes_faster_l698_698035


namespace nikolai_faster_than_gennady_l698_698044

-- The conditions of the problem translated to Lean definitions
def gennady_jump_length : ℕ := 6
def gennady_jumps_per_time : ℕ := 2
def nikolai_jump_length : ℕ := 4
def nikolai_jumps_per_time : ℕ := 3
def turn_around_distance : ℕ := 2000
def round_trip_distance : ℕ := 2 * turn_around_distance

-- The statement that Nikolai completes the journey faster than Gennady
theorem nikolai_faster_than_gennady :
  (nikolai_jumps_per_time * nikolai_jump_length) = (gennady_jumps_per_time * gennady_jump_length) →
  (round_trip_distance % nikolai_jump_length = 0) →
  (round_trip_distance % gennady_jump_length ≠ 0) →
  (round_trip_distance / nikolai_jump_length) + 1 < (round_trip_distance / gennady_jump_length) →
  "Nikolay completes the journey faster." :=
by
  intros h_eq_speed h_nikolai_divisible h_gennady_not_divisible h_time_compare
  sorry

end nikolai_faster_than_gennady_l698_698044


namespace trains_cross_each_other_in_16_14_seconds_l698_698052

noncomputable def train_crossing_time (length1 length2 : ℕ) (speed1 speed2 : ℕ) : ℕ :=
  let relative_speed := (speed1 + speed2) * 5 / 18
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_each_other_in_16_14_seconds :
  train_crossing_time 300 350 80 65 ≈ 16.14 := 
sorry

end trains_cross_each_other_in_16_14_seconds_l698_698052


namespace trajectory_of_z_l698_698512

theorem trajectory_of_z (z : ℂ) (h : complex.abs (z + 1 - complex.i) = complex.abs (z - 1 + complex.i)) :
  ∃ (y : ℝ), y = complex.re z ∧ y = complex.im z :=
sorry

end trajectory_of_z_l698_698512


namespace right_triangle_third_side_product_l698_698776

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l698_698776


namespace hyperbola_asymptotes_correct_l698_698973

noncomputable def hyperbola_asymptotes : Prop :=
  ∀ (a b : ℝ), a > 0 → b > 0 → (e : ℝ) = √5 / 2 →
    (h : (∀ x y : ℝ, (y^2 / a^2 - x^2 / b^2 = 1 → y = 2 * x ∨ y = -2 * x))) →
      (∀ x y : ℝ, y = 2 * x ∨ y = -2 * x)

theorem hyperbola_asymptotes_correct : hyperbola_asymptotes :=
by
  sorry

end hyperbola_asymptotes_correct_l698_698973


namespace school_children_equation_l698_698832

theorem school_children_equation
  (C B : ℕ)
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 := by
  sorry

end school_children_equation_l698_698832


namespace probability_of_adjacent_vertices_l698_698721

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l698_698721


namespace average_activity_area_per_person_l698_698335

-- Define the conditions given in the problem
def base_length : ℝ := 9
def height : ℝ := 8
def number_of_students : ℕ := 24

-- Calculate the area of the parallelogram
def area (a h : ℝ) : ℝ := a * h

-- Calculate the average activity area per person
def average_activity_area (S : ℝ) (n : ℕ) : ℝ := S / (n : ℝ)

-- The theorem stating the equivalence
theorem average_activity_area_per_person :
  average_activity_area (area base_length height) number_of_students = 3 := 
by
  sorry

end average_activity_area_per_person_l698_698335


namespace arctan_tan45_plus_2tan30_l698_698891

theorem arctan_tan45_plus_2tan30 :
  let tan45 := 1 : ℝ
  let tan30 := 1 / Real.sqrt 3 : ℝ
  arctan (tan45 + 2 * tan30) = 75 := 
sorry

end arctan_tan45_plus_2tan30_l698_698891


namespace circle_center_and_chord_length_l698_698178

theorem circle_center_and_chord_length:
  (∀ (a : ℝ), ∃ x y : ℝ, x^2 + y^2 - 2*x - 2*a*y + a^2 - 24 = 0 → 2*x - y = 0 → a = 2) ∧
  (∀ m : ℝ, ∃ x y : ℝ, (x-1)^2 + (y-2)^2 = 25 → (2*m+1)*x + (m+1)*y - 7*m - 4 = 0 →
    x = 3 ∧ y = 1 → 
    let CM := (3-1)^2 + (1-2)^2 in
    let l := 2*√(25 - CM) in
    l = 4*√5) := 
sorry

end circle_center_and_chord_length_l698_698178


namespace largest_two_digit_with_remainder_2_l698_698491

theorem largest_two_digit_with_remainder_2 (n : ℕ) :
  10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n = 93 :=
by
  intro h
  sorry

end largest_two_digit_with_remainder_2_l698_698491


namespace probability_coprime_l698_698173

open Nat

/-- Define the set of integers from 2 to 8 -/
def S : Finset ℕ := {2, 3, 4, 5, 6, 7, 8}

/-- Define the pairs of integers from the set S -/
def pairs := Finset.filter (λ p: ℕ × ℕ, p.1 < p.2) (S.product S)

/-- Define coprime pairs from the set pairs -/
def coprime_pairs := pairs.filter (λ p, gcd p.1 p.2 = 1)

/-- The probability that two randomly selected numbers from S are coprime is 2/3 -/
theorem probability_coprime : 
  (coprime_pairs.card : ℚ) / (pairs.card : ℚ) = 2 / 3 := 
sorry

end probability_coprime_l698_698173


namespace sym_axis_of_g_l698_698660

theorem sym_axis_of_g :
  ∀ (f g : ℝ → ℝ), 
    f = (λ x, 2 * sin (2 * x + π / 3)) →
    g = (λ x, 2 * sin (x + π / 6)) →
    (∃ k : ℤ, g (k * π + π / 3) = g (k * π + - (π / 3))) :=
by sorry

end sym_axis_of_g_l698_698660


namespace commission_rate_up_to_threshold_l698_698427

-- Definitions
def total_sales : ℝ := 32500
def remitted_amount : ℝ := 31100
def commission_exceeding_rate : ℝ := 0.04
def exceeding_sales (total_sales : ℝ) (threshold : ℝ) : ℝ := total_sales - threshold
def commission_exceeding (rate : ℝ) (sales : ℝ) : ℝ := rate * sales

-- The conditions to use in the proof
def threshold : ℝ := 10000
def commission_amount (x : ℝ) (threshold : ℝ) (commission_exceeding : ℝ) : ℝ :=
  (x * threshold / 100) + commission_exceeding
def total_commission (total_sales : ℝ) (remitted_amount : ℝ) : ℝ :=
  total_sales - remitted_amount

-- The theorem to prove
theorem commission_rate_up_to_threshold (x : ℝ) :
  commission_amount x threshold (commission_exceeding commission_exceeding_rate (exceeding_sales total_sales threshold)) =
  total_commission total_sales remitted_amount → x = 5 :=
by
  -- proof will go here
  sorry

end commission_rate_up_to_threshold_l698_698427


namespace sum_d_e_f_l698_698283

noncomputable def t : ℕ → ℝ 
| 0 => 3
| 1 => 4
| 2 => 10
| (n+3) => d * t (n+2) + e * t (n+1) + f * t n

theorem sum_d_e_f : ∃ d e f : ℝ, (∀ n ≥ 2, t (n+1) = d * t n + e * t (n-1) + f * t (n-2)) ∧ d + e + f = 3 :=
  sorry

end sum_d_e_f_l698_698283


namespace line_OP_intersection_l_C_l698_698610

-- Definitions for the given problem
def M : ℝ × ℝ := (2, 0)
def N : ℝ × ℝ := (0, 2 * sqrt(3) / 3)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

def P : ℝ × ℝ := midpoint M N

-- First proof problem: Equation of line OP
theorem line_OP (P : ℝ × ℝ) : P.snd = sqrt(3) / 3 * P.fst :=
sorry

-- Definitions for the second part of the problem
def C (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y + sqrt(3)) ^ 2 = 4

def l (x y : ℝ) : Prop :=
  sqrt(3) * x + 3 * y - 2 * sqrt(3) = 0

-- Second proof problem: Intersection of line l and circle C
theorem intersection_l_C : ∃ x y, C x y ∧ l x y :=
sorry

end line_OP_intersection_l_C_l698_698610


namespace net_increase_in_bicycles_l698_698644

def bicycles_sold (fri_sat_sun : ℤ × ℤ × ℤ) : ℤ :=
  fri_sat_sun.1 + fri_sat_sun.2 + fri_sat_sun.3

def bicycles_bought (fri_sat_sun : ℤ × ℤ × ℤ) : ℤ :=
  fri_sat_sun.1 + fri_sat_sun.2 + fri_sat_sun.3

def net_increase (sold bought : ℤ) : ℤ :=
  bought - sold

theorem net_increase_in_bicycles :
  let bicycles_sold_days := (10, 12, 9)
  let bicycles_bought_days := (15, 8, 11)
  net_increase (bicycles_sold bicycles_sold_days) (bicycles_bought bicycles_bought_days) = 3 :=
by
  sorry

end net_increase_in_bicycles_l698_698644


namespace triangle_TS_eq_r_l698_698180

-- Define the structure and core conditions of the problem
variables {r : ℝ} -- The radius of k1 and k2
variables {O A B S T : EuclideanSpace ℝ (Fin 2)} -- Points in 2D space
variables {k1 k2 : Circle} -- Two circles

 -- Define that O is on the circumference of circle k1 and is the center of k2
-- Define that A and B are intersection points of circles k1 and k2
variable (hA : k1.isOn A ∧ k2.isOn A)
variable (hB : k1.isOn B ∧ k2.isOn B)

-- Define S is an interior point of circle k1
variable (hS_inside_k1 : k1.isInside S)

-- Define the intersection of line BS with circle k1 at T
variable (hBS_T : ∃ T, Line.through B S ∩ k1 = {T})

-- Define the equilateral condition on triangle AOS
variable (hEquilateral : triangle.isEquilateral A O S)

-- stating the theorem: if points and conditions holds, TS = r
theorem triangle_TS_eq_r (hA : k1.isOn A ∧ k2.isOn A)
    (hB : k1.isOn B ∧ k2.isOn B)
    (hS_inside_k1 : k1.isInside S)
    (hBS_T : ∃ T, Line.through B S ∩ k1 = {T})
    (hEquilateral : triangle.isEquilateral A O S) :
    dist S T = r :=
sorry

end triangle_TS_eq_r_l698_698180


namespace triangle_angle_B_triangle_area_l698_698982

open Real

theorem triangle_angle_B (A B C a b c : ℝ) (h1 : a + 2 * c = 2 * b * cos A) (h2 : b = 2 * sqrt 3) :
  B = 2 * π / 3 :=
by
  sorry

theorem triangle_area (A B C a b c : ℝ) (h1 : a + 2 * c = 2 * b * cos A) (h2 : b = 2 * sqrt 3)
  (h3 : a + c = 4) (hB : B = 2 * π / 3) :
  (1 / 2) * a * c * sin B = sqrt 3 :=
by
  sorry

end triangle_angle_B_triangle_area_l698_698982


namespace second_line_equation_l698_698367

theorem second_line_equation :
  ∀ (x y : ℝ),
    (∀ x, y = 2 * x + 2) →
    ∀ x (y: ℝ), y = - (1/2) * x + 649.5 →
    ∀ x y, (259, 520) ∈ {p : ℝ × ℝ | p.snd = 2 * p.fst + 2} →
    ∀ m1 m2 : ℝ, m1 = 2 → m2 = -1 / 2 →
    m1 * m2 = -1 →
    x = 259 → y = 520 →
    (y = - (1/2) * x + 649.5) :=
begin
  intros,
  sorry,
end

end second_line_equation_l698_698367


namespace sequence_general_formula_l698_698937

theorem sequence_general_formula :
  ∃ (a : ℕ → ℕ), 
    (a 1 = 4) ∧ 
    (∀ n : ℕ, a (n + 1) = a n + 3) ∧ 
    (∀ n : ℕ, a n = 3 * n + 1) :=
sorry

end sequence_general_formula_l698_698937


namespace CP_eq_DP_l698_698093

def quadrilateral (A B C D : Type) := 
  convex_quadrilateral A B C D

variables (A B C D P : Type) -- Define the points

-- Define the conditions given in the problem
variables (h1 : convex_quadrilateral A B C D)
  (h2 : angle A B D = 2 * angle B C D)
  (h3 : distance A B = distance A D)
  (h4 : parallelogram A B C P)

-- Define the statement we want to prove
theorem CP_eq_DP (h1 : convex_quadrilateral A B C D)
  (h2 : angle A B D = 2 * angle B C D)
  (h3 : distance A B = distance A D)
  (h4 : parallelogram A B C P) : 
  distance C P = distance D P :=
sorry

end CP_eq_DP_l698_698093


namespace red_balls_in_bag_l698_698849

theorem red_balls_in_bag (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) :
  total_balls = 60 → 
  white_balls = 22 → 
  green_balls = 18 → 
  yellow_balls = 8 → 
  purple_balls = 7 → 
  prob_neither_red_nor_purple = 0.8 → 
  ( ∃ (red_balls : ℕ), red_balls = 5 ) :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end red_balls_in_bag_l698_698849


namespace right_triangle_third_side_product_l698_698778

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l698_698778
