import Mathlib

namespace NUMINAMATH_CALUDE_extended_box_volume_sum_l3456_345683

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def volume_extended_box (b : Box) : ℝ := sorry

/-- Checks if two natural numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop := sorry

theorem extended_box_volume_sum (b : Box) (m n p : ℕ) :
  b.length = 3 ∧ b.width = 4 ∧ b.height = 5 →
  volume_extended_box b = (m : ℝ) + (n : ℝ) * Real.pi / (p : ℝ) →
  m > 0 ∧ n > 0 ∧ p > 0 →
  are_relatively_prime n p →
  m + n + p = 505 := by
  sorry

end NUMINAMATH_CALUDE_extended_box_volume_sum_l3456_345683


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l3456_345637

theorem power_mod_seventeen : 7^1985 % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l3456_345637


namespace NUMINAMATH_CALUDE_area_constant_circle_final_equation_minimum_distance_l3456_345618

noncomputable section

variable (t : ℝ)
variable (h : t ≠ 0)

def C : ℝ × ℝ := (t, 2/t)
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2*t, 0)
def B : ℝ × ℝ := (0, 4/t)

def circle_equation (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2/t)^2 = t^2 + 4/t^2

def line_equation (x y : ℝ) : Prop :=
  2*x + y - 4 = 0

def line_l_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

theorem area_constant :
  (1/2) * |2*t| * |4/t| = 4 :=
sorry

theorem circle_final_equation (x y : ℝ) :
  (∃ M N : ℝ × ℝ, 
    circle_equation t x y ∧ 
    line_equation (M.1) (M.2) ∧ 
    line_equation (N.1) (N.2) ∧
    (M.1 - O.1)^2 + (M.2 - O.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2) →
  (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

theorem minimum_distance (h_pos : t > 0) :
  let B : ℝ × ℝ := (0, 2)
  ∃ P Q : ℝ × ℝ,
    line_l_equation P.1 P.2 ∧
    circle_equation t Q.1 Q.2 ∧
    (∀ P' Q' : ℝ × ℝ, 
      line_l_equation P'.1 P'.2 → 
      circle_equation t Q'.1 Q'.2 →
      Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤
      Real.sqrt ((P'.1 - B.1)^2 + (P'.2 - B.2)^2) + Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 5 ∧
    P.1 = -4/3 ∧ P.2 = -2/3 :=
sorry

end NUMINAMATH_CALUDE_area_constant_circle_final_equation_minimum_distance_l3456_345618


namespace NUMINAMATH_CALUDE_gathering_handshakes_l3456_345699

/-- The number of gremlins at the gathering -/
def num_gremlins : ℕ := 25

/-- The number of imps at the gathering -/
def num_imps : ℕ := 20

/-- The number of imps willing to shake hands with gremlins -/
def num_imps_shaking : ℕ := 10

/-- The number of gremlins each participating imp shakes hands with -/
def gremlin_per_imp : ℕ := 15

/-- Calculate the number of handshakes among gremlins -/
def gremlin_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the number of handshakes between gremlins and imps -/
def gremlin_imp_handshakes : ℕ := num_imps_shaking * gremlin_per_imp

/-- The total number of handshakes at the gathering -/
def total_handshakes : ℕ := gremlin_handshakes num_gremlins + gremlin_imp_handshakes

theorem gathering_handshakes : total_handshakes = 450 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l3456_345699


namespace NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l3456_345629

/-- Given four consecutive digits p, q, r, s, the sum of pqrs and srqp is divisible by 1111 -/
theorem consecutive_digits_sum_divisibility (p : ℕ) (h1 : p < 7) :
  ∃ (k : ℕ), 1000 * p + 100 * (p + 1) + 10 * (p + 2) + (p + 3) +
             1000 * (p + 3) + 100 * (p + 2) + 10 * (p + 1) + p = 1111 * k := by
  sorry

#check consecutive_digits_sum_divisibility

end NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l3456_345629


namespace NUMINAMATH_CALUDE_gcd_g_x_eq_six_l3456_345635

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(11*x+7)*(3*x+5)

theorem gcd_g_x_eq_six (x : ℤ) (h : 18432 ∣ x) : 
  Nat.gcd (g x).natAbs x.natAbs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_eq_six_l3456_345635


namespace NUMINAMATH_CALUDE_isabel_paper_calculation_l3456_345694

/-- The number of pieces of paper Isabel bought -/
def total_paper : ℕ := 900

/-- The number of pieces of paper Isabel used -/
def used_paper : ℕ := 156

/-- The number of pieces of paper Isabel has left -/
def remaining_paper : ℕ := total_paper - used_paper

theorem isabel_paper_calculation :
  remaining_paper = 744 :=
sorry

end NUMINAMATH_CALUDE_isabel_paper_calculation_l3456_345694


namespace NUMINAMATH_CALUDE_marble_probability_l3456_345627

theorem marble_probability (a b c : ℕ) : 
  a + b + c = 97 →
  (a * (a - 1) + b * (b - 1) + c * (c - 1)) / (97 * 96) = 5 / 12 →
  (a^2 + b^2 + c^2) / 97^2 = 41 / 97 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l3456_345627


namespace NUMINAMATH_CALUDE_trigonometric_product_square_root_l3456_345692

theorem trigonometric_product_square_root : 
  let f (x : ℝ) := 512 * x^3 - 1152 * x^2 + 576 * x - 27
  (f (Real.sin (π / 9)^2) = 0) ∧ 
  (f (Real.sin (2 * π / 9)^2) = 0) ∧ 
  (f (Real.sin (4 * π / 9)^2) = 0) →
  Real.sqrt ((3 - Real.sin (π / 9)^2) * (3 - Real.sin (2 * π / 9)^2) * (3 - Real.sin (4 * π / 9)^2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_square_root_l3456_345692


namespace NUMINAMATH_CALUDE_B_equals_roster_l3456_345626

def A : Set Int := {-2, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_roster : B = {4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_B_equals_roster_l3456_345626


namespace NUMINAMATH_CALUDE_perfect_square_m_l3456_345620

theorem perfect_square_m (l m n : ℕ+) (p : ℕ) (h_prime : Prime p) 
  (h_perfect_square : ∃ k : ℕ, p^(2*l.val - 1) * m.val * (m.val * n.val + 1)^2 + m.val^2 = k^2) :
  ∃ r : ℕ, m.val = r^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_m_l3456_345620


namespace NUMINAMATH_CALUDE_base_2_representation_of_56_l3456_345685

/-- Represents a natural number in base 2 as a list of bits (least significant bit first) -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

/-- Converts a list of bits (least significant bit first) to a natural number -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem base_2_representation_of_56 :
  toBinary 56 = [false, false, false, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_base_2_representation_of_56_l3456_345685


namespace NUMINAMATH_CALUDE_adjacent_supplementary_not_always_complementary_l3456_345614

-- Define supplementary angles
def supplementary (α β : Real) : Prop := α + β = 180

-- Define complementary angles
def complementary (α β : Real) : Prop := α + β = 90

-- Define adjacent angles
def adjacent (α β : Real) : Prop := ∃ (γ : Real), α + β + γ = 360

-- Theorem statement
theorem adjacent_supplementary_not_always_complementary :
  ¬ ∀ (α β : Real), (adjacent α β ∧ supplementary α β) → complementary α β :=
sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_not_always_complementary_l3456_345614


namespace NUMINAMATH_CALUDE_cost_of_thousand_pieces_l3456_345698

/-- The cost in dollars of purchasing a given number of pieces of gum -/
def gum_cost (pieces : ℕ) : ℚ :=
  if pieces ≤ 500 then
    (pieces : ℚ) / 100
  else
    (500 : ℚ) / 100 + ((pieces - 500 : ℚ) * 8) / 1000

/-- The cost of 1000 pieces of gum is $9.00 -/
theorem cost_of_thousand_pieces : gum_cost 1000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_thousand_pieces_l3456_345698


namespace NUMINAMATH_CALUDE_equation_solution_l3456_345649

theorem equation_solution (x y : ℝ) 
  (h1 : x + 2 ≠ 0) 
  (h2 : x - y + 1 ≠ 0) 
  (h3 : (x - y) / (x + 2) = y / (x - y + 1)) : 
  x = (y - 1 + Real.sqrt (-3 * y^2 + 10 * y + 1)) / 2 ∨ 
  x = (y - 1 - Real.sqrt (-3 * y^2 + 10 * y + 1)) / 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3456_345649


namespace NUMINAMATH_CALUDE_min_sum_squares_l3456_345674

theorem min_sum_squares (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, (a + 2) / x = a * x + 2 * b + 1) →
  (∀ c d : ℝ, (∃ x ∈ Set.Icc 3 4, (c + 2) / x = c * x + 2 * d + 1) → c^2 + d^2 ≥ 1/100) ∧
  (∃ c d : ℝ, (∃ x ∈ Set.Icc 3 4, (c + 2) / x = c * x + 2 * d + 1) ∧ c^2 + d^2 = 1/100) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3456_345674


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3456_345610

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3456_345610


namespace NUMINAMATH_CALUDE_marbles_in_larger_container_l3456_345638

/-- Given that a container with a volume of 24 cm³ can hold 75 marbles,
    prove that a container with a volume of 72 cm³ can hold 225 marbles,
    assuming the ratio of marbles to volume is constant. -/
theorem marbles_in_larger_container (v₁ v₂ : ℝ) (m₁ m₂ : ℕ) 
    (h₁ : v₁ = 24) (h₂ : m₁ = 75) (h₃ : v₂ = 72) :
    (m₁ : ℝ) / v₁ = m₂ / v₂ → m₂ = 225 := by
  sorry

end NUMINAMATH_CALUDE_marbles_in_larger_container_l3456_345638


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l3456_345693

theorem chocolate_box_problem (total_bars : ℕ) (bars_per_small_box : ℕ) (h1 : total_bars = 500) (h2 : bars_per_small_box = 25) :
  total_bars / bars_per_small_box = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l3456_345693


namespace NUMINAMATH_CALUDE_shifted_parabola_vertex_l3456_345656

/-- Given a parabola y = -2x^2 + 1 shifted 1 unit left and 3 units up, its vertex is at (-1, 4) -/
theorem shifted_parabola_vertex (x y : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -2 * x^2 + 1
  let g : ℝ → ℝ := λ x ↦ f (x + 1) + 3
  g x = y ∧ ∀ t, g t ≤ y → (x = -1 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_shifted_parabola_vertex_l3456_345656


namespace NUMINAMATH_CALUDE_pyarelal_loss_calculation_l3456_345606

/-- Calculates Pyarelal's share of the loss given the total loss and the ratio of investments -/
def pyarelal_loss (total_loss : ℚ) (ashok_ratio : ℚ) (pyarelal_ratio : ℚ) : ℚ :=
  (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss

/-- Proves that Pyarelal's loss is 1080 given the conditions of the problem -/
theorem pyarelal_loss_calculation :
  let total_loss : ℚ := 1200
  let ashok_ratio : ℚ := 1
  let pyarelal_ratio : ℚ := 9
  pyarelal_loss total_loss ashok_ratio pyarelal_ratio = 1080 := by
  sorry

#eval pyarelal_loss 1200 1 9

end NUMINAMATH_CALUDE_pyarelal_loss_calculation_l3456_345606


namespace NUMINAMATH_CALUDE_anna_meal_cost_difference_l3456_345678

theorem anna_meal_cost_difference : 
  let bagel_cost : ℚ := 95/100
  let orange_juice_cost : ℚ := 85/100
  let sandwich_cost : ℚ := 465/100
  let milk_cost : ℚ := 115/100
  let breakfast_cost := bagel_cost + orange_juice_cost
  let lunch_cost := sandwich_cost + milk_cost
  lunch_cost - breakfast_cost = 4
  := by sorry

end NUMINAMATH_CALUDE_anna_meal_cost_difference_l3456_345678


namespace NUMINAMATH_CALUDE_maddie_monday_viewing_l3456_345660

/-- The number of minutes Maddie watched TV on Monday -/
def monday_minutes (total_episodes : ℕ) (episode_length : ℕ) (thursday_minutes : ℕ) (friday_episodes : ℕ) (weekend_minutes : ℕ) : ℕ :=
  total_episodes * episode_length - (thursday_minutes + friday_episodes * episode_length + weekend_minutes)

theorem maddie_monday_viewing : 
  monday_minutes 8 44 21 2 105 = 138 := by
  sorry

end NUMINAMATH_CALUDE_maddie_monday_viewing_l3456_345660


namespace NUMINAMATH_CALUDE_divisibility_condition_l3456_345673

theorem divisibility_condition (n : ℕ) : 
  (∃ k : ℤ, (7 * n + 5 : ℤ) = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3456_345673


namespace NUMINAMATH_CALUDE_matthews_crackers_l3456_345658

theorem matthews_crackers (initial_cakes : ℕ) (num_friends : ℕ) (cakes_eaten_per_person : ℕ)
  (h1 : initial_cakes = 30)
  (h2 : num_friends = 2)
  (h3 : cakes_eaten_per_person = 15)
  : initial_cakes = num_friends * cakes_eaten_per_person :=
by
  sorry

#check matthews_crackers

end NUMINAMATH_CALUDE_matthews_crackers_l3456_345658


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3456_345663

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + k = 0) ↔ k = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3456_345663


namespace NUMINAMATH_CALUDE_vote_intersection_l3456_345680

theorem vote_intersection (U A B : Finset Int) (h1 : U.card = 300) 
  (h2 : A.card = 230) (h3 : B.card = 190) (h4 : (U \ A).card + (U \ B).card - U.card = 40) :
  (A ∩ B).card = 160 := by
  sorry

end NUMINAMATH_CALUDE_vote_intersection_l3456_345680


namespace NUMINAMATH_CALUDE_root_in_interval_l3456_345642

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  ∃ (root : ℝ), root ∈ Set.Icc 2 2.5 ∧ f root = 0 :=
by
  have h1 : f 2 < 0 := by sorry
  have h2 : f 2.5 > 0 := by sorry
  have h3 : f 3 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3456_345642


namespace NUMINAMATH_CALUDE_perpendicular_lines_plane_theorem_l3456_345604

/-- Represents a plane in 3D space -/
structure Plane :=
  (α : Type*)

/-- Represents a line in 3D space -/
structure Line :=
  (l : Type*)

/-- Indicates that a line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Indicates that a line is perpendicular to another line -/
def perpendicular_to_line (l1 l2 : Line) : Prop :=
  sorry

/-- Indicates that a line is in a plane -/
def line_in_plane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Indicates that a line is outside a plane -/
def line_outside_plane (l : Line) (α : Plane) : Prop :=
  sorry

theorem perpendicular_lines_plane_theorem 
  (α : Plane) (a b l : Line) 
  (h1 : a ≠ b)
  (h2 : line_in_plane a α)
  (h3 : line_in_plane b α)
  (h4 : line_outside_plane l α) :
  (∀ (α : Plane) (l : Line), perpendicular_to_plane l α → 
    perpendicular_to_line l a ∧ perpendicular_to_line l b) ∧
  (∃ (α : Plane) (a b l : Line), 
    perpendicular_to_line l a ∧ perpendicular_to_line l b ∧
    ¬perpendicular_to_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_plane_theorem_l3456_345604


namespace NUMINAMATH_CALUDE_student_count_l3456_345648

theorem student_count (rank_right rank_left : ℕ) 
  (h1 : rank_right = 13) 
  (h2 : rank_left = 8) : 
  rank_right + rank_left - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3456_345648


namespace NUMINAMATH_CALUDE_remainder_492381_div_6_l3456_345602

theorem remainder_492381_div_6 : 492381 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_492381_div_6_l3456_345602


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_682_l3456_345657

theorem sin_n_equals_cos_682 :
  ∃ n : ℤ, -120 ≤ n ∧ n ≤ 120 ∧ Real.sin (n * π / 180) = Real.cos (682 * π / 180) ∧ n = 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_682_l3456_345657


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l3456_345624

theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ x ≠ 1/2 ∧ 2/(x-1) = m/(2*x-1)) ↔ 
  (m > 4 ∨ m < 2) ∧ m ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l3456_345624


namespace NUMINAMATH_CALUDE_avg_f_value_l3456_345615

/-- A function that counts the number of multiples of p in the partial sums of a permutation -/
def f (p : ℕ) (π : Fin p → Fin p) : ℕ := sorry

/-- The average value of f over all permutations -/
def avg_f (p : ℕ) : ℚ := sorry

theorem avg_f_value (p : ℕ) (h : p.Prime) (h2 : p > 2) :
  avg_f p = 2 - 1 / p := by sorry

end NUMINAMATH_CALUDE_avg_f_value_l3456_345615


namespace NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l3456_345697

-- Define a triangle
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_of_angles : angle1 + angle2 + angle3 = 180
  positive_angles : angle1 > 0 ∧ angle2 > 0 ∧ angle3 > 0

-- Define an obtuse angle
def is_obtuse (angle : Real) : Prop := angle > 90

-- Theorem statement
theorem at_most_one_obtuse_angle (t : Triangle) : 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle2) ∧ 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle3) ∧ 
  ¬(is_obtuse t.angle2 ∧ is_obtuse t.angle3) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l3456_345697


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l3456_345634

def M : ℕ := 18 * 18 * 56 * 165

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 62 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l3456_345634


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3456_345628

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 18 / (9 - x ^ (1/4))) ↔ (x = 81 ∨ x = 1296) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3456_345628


namespace NUMINAMATH_CALUDE_max_area_of_region_S_l3456_345684

/-- Represents a circle in a plane -/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents a configuration of four circles tangent to a line -/
structure CircleConfiguration where
  circles : Fin 4 → Circle
  tangent_point : ℝ × ℝ
  radii_correct : ∀ i, (circles i).radius ∈ ({2, 4, 6, 8} : Set ℝ)
  tangent_to_line : ∀ i, (circles i).center.2 = tangent_point.2

/-- The area of the region inside exactly one of the circles -/
def areaOfRegionS (config : CircleConfiguration) : ℝ := sorry

theorem max_area_of_region_S :
  ∃ (config : CircleConfiguration),
    ∀ (other_config : CircleConfiguration),
      areaOfRegionS config ≥ areaOfRegionS other_config ∧
      areaOfRegionS config = 84 * Real.pi := by sorry

end NUMINAMATH_CALUDE_max_area_of_region_S_l3456_345684


namespace NUMINAMATH_CALUDE_first_recipe_cups_l3456_345622

/-- Represents the amount of soy sauce in various units --/
structure SoySauce where
  bottles : ℕ
  ounces : ℕ
  cups : ℕ

/-- Conversion factors and recipe requirements --/
def bottleSize : ℕ := 16 -- ounces per bottle
def ouncesPerCup : ℕ := 8
def recipe2Cups : ℕ := 1
def recipe3Cups : ℕ := 3
def totalBottles : ℕ := 3

/-- The main theorem to prove --/
theorem first_recipe_cups (sauce : SoySauce) : 
  sauce.bottles = totalBottles → 
  sauce.ounces = sauce.bottles * bottleSize → 
  sauce.cups = sauce.ounces / ouncesPerCup →
  sauce.cups = recipe2Cups + recipe3Cups + 2 :=
by sorry

end NUMINAMATH_CALUDE_first_recipe_cups_l3456_345622


namespace NUMINAMATH_CALUDE_count_valid_m_l3456_345695

theorem count_valid_m : ∃! (S : Finset ℤ), 
  (∀ m ∈ S, (∀ x : ℝ, (3 - 3*x < x - 5 ∧ x - m > -1) ↔ x > 2) ∧ 
             (∃ x : ℕ+, (2*x - m) / 3 = 1)) ∧
  S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_valid_m_l3456_345695


namespace NUMINAMATH_CALUDE_fourth_term_is_seven_l3456_345609

/-- An arithmetic sequence with sum of first 7 terms equal to 49 -/
structure ArithmeticSequence where
  /-- The nth term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- Property: S_7 = 49 -/
  sum_7 : S 7 = 49
  /-- Property: S_n is the sum of first n terms -/
  sum_property : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

/-- The 4th term of the arithmetic sequence is 7 -/
theorem fourth_term_is_seven (seq : ArithmeticSequence) : seq.a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_seven_l3456_345609


namespace NUMINAMATH_CALUDE_kamal_biology_mark_l3456_345630

/-- Given Kamal's marks in 5 subjects with a known average, prove his Biology mark. -/
theorem kamal_biology_mark (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) 
  (biology : ℕ) (average : ℕ) (h1 : english = 66) (h2 : mathematics = 65) (h3 : physics = 77) 
  (h4 : chemistry = 62) (h5 : average = 69) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 75 := by
  sorry

end NUMINAMATH_CALUDE_kamal_biology_mark_l3456_345630


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_min_value_l3456_345677

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set (x : ℝ) : f x ≤ x + 1 ↔ 1 ≤ x ∧ x ≤ 5 := by sorry

-- Theorem 2: Inequality proof
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 / (a + 1) + b^2 / (b + 1) ≥ 1 := by sorry

-- Theorem to show that the minimum value of f(x) is 2
theorem min_value : ∀ x : ℝ, f x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_min_value_l3456_345677


namespace NUMINAMATH_CALUDE_derek_savings_and_expenses_l3456_345661

theorem derek_savings_and_expenses :
  let geometric_sum := (2 : ℝ) * (1 - 2^12) / (1 - 2)
  let arithmetic_sum := 12 / 2 * (2 * 3 + (12 - 1) * 2)
  geometric_sum - arithmetic_sum = 8022 := by
  sorry

end NUMINAMATH_CALUDE_derek_savings_and_expenses_l3456_345661


namespace NUMINAMATH_CALUDE_xy_value_l3456_345613

theorem xy_value (x y : ℝ) : (Complex.I : ℂ).re * x + (Complex.I : ℂ).im * y = 2 → x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3456_345613


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3456_345601

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3456_345601


namespace NUMINAMATH_CALUDE_power_sum_inequality_l3456_345689

theorem power_sum_inequality (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l3456_345689


namespace NUMINAMATH_CALUDE_b_equals_seven_l3456_345659

-- Define the functions f and F
def f (a : ℝ) (x : ℝ) : ℝ := x - a

def F (x y : ℝ) : ℝ := y^2 + x

-- Define b as F(3, f(4))
def b (a : ℝ) : ℝ := F 3 (f a 4)

-- Theorem to prove
theorem b_equals_seven (a : ℝ) : b a = 7 := by
  sorry

end NUMINAMATH_CALUDE_b_equals_seven_l3456_345659


namespace NUMINAMATH_CALUDE_imaginary_number_condition_l3456_345681

theorem imaginary_number_condition (m : ℝ) : 
  let z : ℂ := (m + Complex.I) / (1 + m * Complex.I)
  z.re = 0 ∧ z.im ≠ 0 → m = 0 := by
sorry

end NUMINAMATH_CALUDE_imaginary_number_condition_l3456_345681


namespace NUMINAMATH_CALUDE_min_value_theorem_l3456_345644

theorem min_value_theorem (a b c d : ℝ) :
  (|b + a^2 - 4 * Real.log a| + |2 * c - d + 2| = 0) →
  ∃ (min_value : ℝ), (∀ (a' b' c' d' : ℝ), 
    (|b' + a'^2 - 4 * Real.log a'| + |2 * c' - d' + 2| = 0) →
    ((a' - c')^2 + (b' - d')^2 ≥ min_value)) ∧
  min_value = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3456_345644


namespace NUMINAMATH_CALUDE_money_problem_l3456_345651

theorem money_problem (M : ℚ) : 
  (3/4 * (2/3 * (2/3 * M + 10) + 20) = M) → M = 30 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3456_345651


namespace NUMINAMATH_CALUDE_sunday_calorie_intake_theorem_l3456_345676

/-- Calculates John's calorie intake for Sunday given his meal structure and calorie content --/
def sunday_calorie_intake (breakfast_calories : ℝ) (morning_snack_addition : ℝ) 
  (lunch_percentage : ℝ) (afternoon_snack_reduction : ℝ) (dinner_multiplier : ℝ) 
  (energy_drink_calories : ℝ) : ℝ :=
  let lunch_calories := breakfast_calories * (1 + lunch_percentage)
  let afternoon_snack_calories := lunch_calories * (1 - afternoon_snack_reduction)
  let dinner_calories := lunch_calories * dinner_multiplier
  let weekday_calories := breakfast_calories + (breakfast_calories + morning_snack_addition) + 
                          lunch_calories + afternoon_snack_calories + dinner_calories
  let energy_drinks_calories := 2 * energy_drink_calories
  weekday_calories + energy_drinks_calories

theorem sunday_calorie_intake_theorem :
  sunday_calorie_intake 500 150 0.25 0.30 2 220 = 3402.5 := by
  sorry

end NUMINAMATH_CALUDE_sunday_calorie_intake_theorem_l3456_345676


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l3456_345672

/-- Represents a digit in the set {1, 2, 3, 4, 5, 6} -/
def Digit := Fin 6

/-- Represents the multiplication problem AB × C = DEF -/
def IsValidMultiplication (a b c d e f : Digit) : Prop :=
  (a.val + 1) * 10 + (b.val + 1) = (d.val + 1) * 100 + (e.val + 1) * 10 + (f.val + 1)

/-- All digits are distinct -/
def AreDistinct (a b c d e f : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

theorem multiplication_puzzle :
  ∀ (a b c d e f : Digit),
    IsValidMultiplication a b c d e f →
    AreDistinct a b c d e f →
    c.val = 2 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l3456_345672


namespace NUMINAMATH_CALUDE_logan_desired_amount_left_l3456_345671

/-- Represents Logan's financial situation and goal --/
structure LoganFinances where
  current_income : ℕ
  rent_expense : ℕ
  groceries_expense : ℕ
  gas_expense : ℕ
  income_increase : ℕ

/-- Calculates the desired amount left each year for Logan --/
def desired_amount_left (f : LoganFinances) : ℕ :=
  (f.current_income + f.income_increase) - (f.rent_expense + f.groceries_expense + f.gas_expense)

/-- Theorem stating the desired amount left each year for Logan --/
theorem logan_desired_amount_left :
  let f : LoganFinances := {
    current_income := 65000,
    rent_expense := 20000,
    groceries_expense := 5000,
    gas_expense := 8000,
    income_increase := 10000
  }
  desired_amount_left f = 42000 := by
  sorry


end NUMINAMATH_CALUDE_logan_desired_amount_left_l3456_345671


namespace NUMINAMATH_CALUDE_number_of_factors_of_M_l3456_345647

def M : ℕ := 58^6 + 6*58^5 + 15*58^4 + 20*58^3 + 15*58^2 + 6*58 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (λ x => M % x = 0) (Finset.range (M + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_M_l3456_345647


namespace NUMINAMATH_CALUDE_troll_difference_l3456_345682

/-- The number of trolls hiding under the bridge -/
def T : ℕ := 18

/-- The number of trolls hiding by the path in the forest -/
def forest_trolls : ℕ := 6

/-- The number of trolls hiding in the plains -/
def plains_trolls : ℕ := T / 2

/-- The total number of trolls -/
def total_trolls : ℕ := 33

theorem troll_difference : 
  4 * forest_trolls - T = 6 ∧ 
  forest_trolls + T + plains_trolls = total_trolls :=
sorry

end NUMINAMATH_CALUDE_troll_difference_l3456_345682


namespace NUMINAMATH_CALUDE_probability_three_digit_l3456_345667

def set_start : ℕ := 60
def set_end : ℕ := 1000

def three_digit_start : ℕ := 100
def three_digit_end : ℕ := 999

def total_numbers : ℕ := set_end - set_start + 1
def three_digit_numbers : ℕ := three_digit_end - (three_digit_start - 1)

theorem probability_three_digit :
  (three_digit_numbers : ℚ) / total_numbers = 901 / 941 := by sorry

end NUMINAMATH_CALUDE_probability_three_digit_l3456_345667


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_logarithmic_inequality_l3456_345679

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) :=
by sorry

theorem negation_of_logarithmic_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x + x - 1 ≤ 0) ↔
  (∀ x : ℝ, x > 0 → Real.log x + x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_logarithmic_inequality_l3456_345679


namespace NUMINAMATH_CALUDE_jills_peaches_l3456_345631

theorem jills_peaches (steven_peaches jake_peaches jill_peaches : ℕ) : 
  steven_peaches = 19 →
  jake_peaches = steven_peaches - 18 →
  steven_peaches = jill_peaches + 13 →
  jill_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_jills_peaches_l3456_345631


namespace NUMINAMATH_CALUDE_factorization_quadratic_l3456_345639

theorem factorization_quadratic (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_quadratic_l3456_345639


namespace NUMINAMATH_CALUDE_sum_of_5_and_8_l3456_345664

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_5_and_8_l3456_345664


namespace NUMINAMATH_CALUDE_xy_difference_l3456_345666

theorem xy_difference (x y : ℝ) (h : 10 * x^2 - 16 * x * y + 8 * y^2 + 6 * x - 4 * y + 1 = 0) : 
  x - y = -0.25 := by
sorry

end NUMINAMATH_CALUDE_xy_difference_l3456_345666


namespace NUMINAMATH_CALUDE_bus_journey_time_l3456_345655

/-- Calculates the total time for a bus journey with two different speeds -/
theorem bus_journey_time (total_distance : ℝ) (speed1 speed2 : ℝ) (distance1 : ℝ) : 
  total_distance = 250 →
  speed1 = 40 →
  speed2 = 60 →
  distance1 = 148 →
  (distance1 / speed1) + ((total_distance - distance1) / speed2) = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_time_l3456_345655


namespace NUMINAMATH_CALUDE_exists_same_direction_interval_l3456_345605

/-- Represents a periodic motion on a line segment -/
structure PeriodicMotion where
  period : ℝ
  direction : ℝ → Bool  -- True for one direction, False for the opposite

/-- Theorem: Given three periodic motions with periods 12, 6, and 4 minutes,
    there always exists a 1-minute interval where all motions are in the same direction -/
theorem exists_same_direction_interval
  (m1 : PeriodicMotion)
  (m2 : PeriodicMotion)
  (m3 : PeriodicMotion)
  (h1 : m1.period = 12)
  (h2 : m2.period = 6)
  (h3 : m3.period = 4)
  : ∃ (t : ℝ), ∀ (s : ℝ), 0 ≤ s ∧ s ≤ 1 →
    (m1.direction (t + s) = m2.direction (t + s)) ∧
    (m2.direction (t + s) = m3.direction (t + s)) :=
sorry

end NUMINAMATH_CALUDE_exists_same_direction_interval_l3456_345605


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l3456_345690

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- Define the set difference operation
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetric_difference (M N : Set ℝ) : Set ℝ := (set_difference M N) ∪ (set_difference N M)

-- State the theorem
theorem symmetric_difference_A_B :
  symmetric_difference A B = {x | x ≥ 0 ∨ x < -9/4} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l3456_345690


namespace NUMINAMATH_CALUDE_x_value_l3456_345608

theorem x_value (y : ℝ) (h1 : 2 * x - y = 14) (h2 : y = 2) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3456_345608


namespace NUMINAMATH_CALUDE_number_of_phone_repairs_l3456_345696

/-- Represents the repair shop scenario -/
def repair_shop (phone_repairs : ℕ) : Prop :=
  let phone_cost : ℕ := 11
  let laptop_cost : ℕ := 15
  let computer_cost : ℕ := 18
  let laptop_repairs : ℕ := 2
  let computer_repairs : ℕ := 2
  let total_earnings : ℕ := 121
  phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs = total_earnings

/-- Theorem stating that the number of phone repairs is 5 -/
theorem number_of_phone_repairs : ∃ (phone_repairs : ℕ), phone_repairs = 5 ∧ repair_shop phone_repairs :=
by sorry

end NUMINAMATH_CALUDE_number_of_phone_repairs_l3456_345696


namespace NUMINAMATH_CALUDE_x_squared_y_plus_xy_squared_l3456_345645

theorem x_squared_y_plus_xy_squared (x y : ℝ) :
  x = Real.sqrt 3 + Real.sqrt 2 →
  y = Real.sqrt 3 - Real.sqrt 2 →
  x^2 * y + x * y^2 = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_x_squared_y_plus_xy_squared_l3456_345645


namespace NUMINAMATH_CALUDE_height_difference_l3456_345675

theorem height_difference (tallest_height shortest_height : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : shortest_height = 68.25) :
  tallest_height - shortest_height = 9.5 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l3456_345675


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l3456_345617

theorem complex_in_fourth_quadrant (m : ℝ) (z : ℂ) 
  (h1 : m < 1) 
  (h2 : z = 2 + (m - 1) * Complex.I) : 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l3456_345617


namespace NUMINAMATH_CALUDE_mixture_ratio_l3456_345603

def mixture (initial_water : ℝ) : Prop :=
  let initial_alcohol : ℝ := 10
  let added_water : ℝ := 10
  let new_ratio_alcohol : ℝ := 2
  let new_ratio_water : ℝ := 7
  (initial_alcohol / (initial_water + added_water) = new_ratio_alcohol / new_ratio_water) ∧
  (initial_alcohol / initial_water = 2 / 5)

theorem mixture_ratio : ∃ (initial_water : ℝ), mixture initial_water :=
  sorry

end NUMINAMATH_CALUDE_mixture_ratio_l3456_345603


namespace NUMINAMATH_CALUDE_double_age_in_three_years_l3456_345669

/-- The number of years from now when Tully will be twice as old as Kate -/
def years_until_double_age (tully_age_last_year : ℕ) (kate_age_now : ℕ) : ℕ :=
  3

theorem double_age_in_three_years (tully_age_last_year kate_age_now : ℕ) 
  (h1 : tully_age_last_year = 60) (h2 : kate_age_now = 29) :
  years_until_double_age tully_age_last_year kate_age_now = 3 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_three_years_l3456_345669


namespace NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_remainder_at_most_15_exists_number_for_remainder_l3456_345668

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Sum of digits of a two-digit number -/
def sumOfDigits (n : TwoDigitNumber) : ℕ :=
  n.val / 10 + n.val % 10

/-- Theorem 1: There exists a two-digit number divisible by the sum of its digits -/
theorem exists_divisible_by_sum_of_digits :
  ∃ n : TwoDigitNumber, n.val % (sumOfDigits n) = 0 :=
sorry

/-- Theorem 2: The remainder when a two-digit number is divided by the sum of its digits is at most 15 -/
theorem remainder_at_most_15 (n : TwoDigitNumber) :
  n.val % (sumOfDigits n) ≤ 15 :=
sorry

/-- Theorem 3: For any remainder r ≤ 12, there exists a two-digit number that produces that remainder -/
theorem exists_number_for_remainder (r : ℕ) (h : r ≤ 12) :
  ∃ n : TwoDigitNumber, n.val % (sumOfDigits n) = r :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_remainder_at_most_15_exists_number_for_remainder_l3456_345668


namespace NUMINAMATH_CALUDE_car_speed_l3456_345646

theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 624) 
    (h2 : time = 3) 
    (h3 : speed = distance / time) : speed = 208 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3456_345646


namespace NUMINAMATH_CALUDE_unique_valid_number_l3456_345619

def is_valid_number (n : ℕ) : Prop :=
  -- n is a four-digit number
  1000 ≤ n ∧ n < 10000 ∧
  -- n can be divided into two two-digit numbers
  let x := n / 100
  let y := n % 100
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧
  -- Adding a 0 to the end of the first two-digit number and adding it to the product of the two two-digit numbers equals the original four-digit number
  10 * x + x * y = n ∧
  -- The unit digit of the original number is 5
  n % 10 = 5

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 1995 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3456_345619


namespace NUMINAMATH_CALUDE_centroid_maximizes_min_area_ratio_l3456_345632

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Calculates the area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- Calculates the centroid of a triangle -/
def Triangle.centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Represents a line in 2D space -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Calculates the minimum area ratio when a triangle is divided by a line through a point -/
def minAreaRatio (t : Triangle) (p : ℝ × ℝ) : ℝ := sorry

theorem centroid_maximizes_min_area_ratio (t : Triangle) :
  ∀ p : ℝ × ℝ, minAreaRatio t (t.centroid) ≥ minAreaRatio t p :=
sorry

end NUMINAMATH_CALUDE_centroid_maximizes_min_area_ratio_l3456_345632


namespace NUMINAMATH_CALUDE_booklet_word_count_l3456_345621

theorem booklet_word_count (words_per_page : ℕ) : 
  words_per_page ≤ 150 →
  (120 * words_per_page) % 221 = 172 →
  words_per_page = 114 := by
sorry

end NUMINAMATH_CALUDE_booklet_word_count_l3456_345621


namespace NUMINAMATH_CALUDE_christmas_to_birthday_ratio_l3456_345641

def total_presents : ℕ := 90
def christmas_presents : ℕ := 60

theorem christmas_to_birthday_ratio :
  (christmas_presents : ℚ) / (total_presents - christmas_presents : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_christmas_to_birthday_ratio_l3456_345641


namespace NUMINAMATH_CALUDE_unique_positive_number_l3456_345662

theorem unique_positive_number : ∃! (n : ℝ), n > 0 ∧ (1/5 * n) * (1/7 * n) = n := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l3456_345662


namespace NUMINAMATH_CALUDE_words_per_page_l3456_345612

/-- Calculates the number of words per page in books Sarah is reading --/
theorem words_per_page
  (reading_speed : ℕ)  -- Sarah's reading speed in words per minute
  (reading_time : ℕ)   -- Total reading time in hours
  (num_books : ℕ)      -- Number of books Sarah plans to read
  (pages_per_book : ℕ) -- Number of pages in each book
  (h1 : reading_speed = 40)
  (h2 : reading_time = 20)
  (h3 : num_books = 6)
  (h4 : pages_per_book = 80)
  : (reading_speed * 60 * reading_time) / (num_books * pages_per_book) = 100 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l3456_345612


namespace NUMINAMATH_CALUDE_vector_collinearity_l3456_345687

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c, then k = -1 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 1)) 
    (hb : b = (-1, 0)) 
    (hc : c = (2, 1)) 
    (hcollinear : ∃ (t : ℝ), t • c = k • a + b) : 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3456_345687


namespace NUMINAMATH_CALUDE_game_draw_probability_l3456_345688

theorem game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) :
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_game_draw_probability_l3456_345688


namespace NUMINAMATH_CALUDE_root_zero_iff_m_neg_three_l3456_345686

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + (3*m - 1) * x + m^2 - 9

/-- Theorem: One root of the quadratic equation is 0 iff m = -3 -/
theorem root_zero_iff_m_neg_three :
  ∀ m : ℝ, (∃ x : ℝ, quadratic_equation m x = 0 ∧ x = 0) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_root_zero_iff_m_neg_three_l3456_345686


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l3456_345623

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) 
  (h3 : 1 ≤ a - 2*b) (h4 : a - 2*b ≤ 3) : 
  (-11/3 ≤ a + 3*b) ∧ (a + 3*b ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l3456_345623


namespace NUMINAMATH_CALUDE_total_carrots_l3456_345654

theorem total_carrots (sandy sam sarah : ℕ) 
  (h1 : sandy = 6) 
  (h2 : sam = 3) 
  (h3 : sarah = 5) : 
  sandy + sam + sarah = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l3456_345654


namespace NUMINAMATH_CALUDE_number_of_possible_values_l3456_345600

theorem number_of_possible_values (m n k a b : ℕ+) :
  ((1 + a.val : ℕ) * n.val^2 - 4 * (m.val + a.val) * n.val + 4 * m.val^2 + 4 * a.val + b.val * (k.val - 1)^2 < 3) →
  (∃ (s : Finset ℕ), s = {x | ∃ (m' n' k' : ℕ+), 
    ((1 + a.val : ℕ) * n'.val^2 - 4 * (m'.val + a.val) * n'.val + 4 * m'.val^2 + 4 * a.val + b.val * (k'.val - 1)^2 < 3) ∧
    x = m'.val + n'.val + k'.val} ∧ 
  s.card = 4) :=
sorry

end NUMINAMATH_CALUDE_number_of_possible_values_l3456_345600


namespace NUMINAMATH_CALUDE_feathers_count_l3456_345643

/-- The number of animals in the first group -/
def group1_animals : ℕ := 934

/-- The number of feathers in crowns for the first group -/
def group1_feathers : ℕ := 7

/-- The number of animals in the second group -/
def group2_animals : ℕ := 425

/-- The number of colored feathers in crowns for the second group -/
def group2_colored_feathers : ℕ := 7

/-- The number of golden feathers in crowns for the second group -/
def group2_golden_feathers : ℕ := 5

/-- The number of animals in the third group -/
def group3_animals : ℕ := 289

/-- The number of colored feathers in crowns for the third group -/
def group3_colored_feathers : ℕ := 4

/-- The number of golden feathers in crowns for the third group -/
def group3_golden_feathers : ℕ := 10

/-- The total number of feathers needed for all animals -/
def total_feathers : ℕ := 15684

theorem feathers_count :
  group1_animals * group1_feathers +
  group2_animals * (group2_colored_feathers + group2_golden_feathers) +
  group3_animals * (group3_colored_feathers + group3_golden_feathers) =
  total_feathers := by
  sorry

end NUMINAMATH_CALUDE_feathers_count_l3456_345643


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l3456_345652

-- Define the cubic polynomial whose roots are a, b, c
def cubic (x : ℝ) := x^3 + 4*x^2 + 6*x + 8

-- Define the properties of P
def P_properties (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  cubic a = 0 ∧ cubic b = 0 ∧ cubic c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- Define the specific polynomial we want to prove is equal to P
def target_poly (x : ℝ) := 2*x^3 + 7*x^2 + 11*x + 12

-- The main theorem
theorem cubic_polynomial_uniqueness :
  ∀ (P : ℝ → ℝ) (a b c : ℝ),
  P_properties P a b c →
  (∀ x, P x = target_poly x) :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l3456_345652


namespace NUMINAMATH_CALUDE_shirt_probabilities_l3456_345633

structure ShirtPack :=
  (m_shirts : ℕ)
  (count : ℕ)

def total_packs : ℕ := 50

def shirt_distribution : List ShirtPack := [
  ⟨0, 7⟩, ⟨1, 3⟩, ⟨4, 10⟩, ⟨5, 15⟩, ⟨7, 5⟩, ⟨9, 4⟩, ⟨10, 3⟩, ⟨11, 3⟩
]

def count_packs (pred : ShirtPack → Bool) : ℕ :=
  (shirt_distribution.filter pred).foldl (λ acc pack => acc + pack.count) 0

theorem shirt_probabilities :
  (count_packs (λ pack => pack.m_shirts = 0) : ℚ) / total_packs = 7 / 50 ∧
  (count_packs (λ pack => pack.m_shirts < 7) : ℚ) / total_packs = 7 / 10 ∧
  (count_packs (λ pack => pack.m_shirts > 9) : ℚ) / total_packs = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_shirt_probabilities_l3456_345633


namespace NUMINAMATH_CALUDE_square_of_threes_and_four_exist_three_digits_for_infinite_squares_l3456_345640

/-- Represents a number with n threes followed by a four -/
def number_with_threes_and_four (n : ℕ) : ℕ :=
  (3 * (10^n - 1) / 9) * 10 + 4

/-- Represents a number with n+1 ones, followed by n fives, and ending with a six -/
def number_with_ones_fives_and_six (n : ℕ) : ℕ :=
  (10^(2*n + 2) - 1) / 9 * 10^n * 5 + 6

/-- Theorem stating that the square of number_with_threes_and_four is equal to number_with_ones_fives_and_six -/
theorem square_of_threes_and_four (n : ℕ) :
  (number_with_threes_and_four n)^2 = number_with_ones_fives_and_six n := by
  sorry

/-- Corollary stating that there exist three non-zero digits that can be used to form
    an infinite number of decimal representations of squares of different integers -/
theorem exist_three_digits_for_infinite_squares :
  ∃ (d₁ d₂ d₃ : ℕ), d₁ ≠ 0 ∧ d₂ ≠ 0 ∧ d₃ ≠ 0 ∧
    ∀ (n : ℕ), ∃ (m : ℕ), m^2 = number_with_ones_fives_and_six n ∧
    (∀ (k : ℕ), k < n → number_with_ones_fives_and_six k ≠ number_with_ones_fives_and_six n) := by
  sorry

end NUMINAMATH_CALUDE_square_of_threes_and_four_exist_three_digits_for_infinite_squares_l3456_345640


namespace NUMINAMATH_CALUDE_two_point_form_equation_l3456_345625

/-- Two-point form equation of a line passing through two points -/
theorem two_point_form_equation (x1 y1 x2 y2 : ℝ) :
  let A : ℝ × ℝ := (x1, y1)
  let B : ℝ × ℝ := (x2, y2)
  x1 = 5 ∧ y1 = 6 ∧ x2 = -1 ∧ y2 = 2 →
  ∀ (x y : ℝ), (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1) ↔
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) :=
by sorry

end NUMINAMATH_CALUDE_two_point_form_equation_l3456_345625


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3456_345650

theorem min_value_of_expression (c d : ℤ) (h : c > d) :
  (c + 2*d) / (c - d) + (c - d) / (c + 2*d) ≥ 2 ∧
  ∃ (c' d' : ℤ), c' > d' ∧ (c' + 2*d') / (c' - d') + (c' - d') / (c' + 2*d') = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3456_345650


namespace NUMINAMATH_CALUDE_unselected_probability_l3456_345653

/-- The type representing a selection of five consecutive integers from a circle of 10 numbers -/
def Selection := Fin 10

/-- The type representing the choices of four people -/
def Choices := Fin 4 → Selection

/-- The probability that there exists a number not selected by any of the four people -/
def probability_unselected (choices : Choices) : ℚ :=
  sorry

/-- The main theorem stating the probability of an unselected number -/
theorem unselected_probability :
  ∃ (p : ℚ), (∀ (choices : Choices), probability_unselected choices = p) ∧ 10000 * p = 3690 :=
sorry

end NUMINAMATH_CALUDE_unselected_probability_l3456_345653


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3456_345607

theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 2) * x^(m^2 - 2) - m*x + 1 = a*x^2 + b*x + c) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3456_345607


namespace NUMINAMATH_CALUDE_max_moves_le_four_max_moves_21x21_max_moves_20x21_l3456_345616

/-- Represents a rectangular grid with lights -/
structure Grid (m n : ℕ) where
  lights : Fin m → Fin n → Bool

/-- A move in the game -/
structure Move (m n : ℕ) where
  line : (Fin m × Fin n) → Bool
  affects_light : ∀ (i : Fin m) (j : Fin n), line (i, j) = false

/-- The game state -/
structure GameState (m n : ℕ) where
  grid : Grid m n
  moves : List (Move m n)

/-- The maximum number of moves for any rectangular grid is at most 4 -/
theorem max_moves_le_four (m n : ℕ) (g : GameState m n) :
  g.moves.length ≤ 4 :=
sorry

/-- For a 21x21 square grid, the maximum number of moves is 3 -/
theorem max_moves_21x21 (g : GameState 21 21) :
  g.moves.length ≤ 3 :=
sorry

/-- For a 20x21 rectangular grid, the maximum number of moves is 4 -/
theorem max_moves_20x21 (g : GameState 20 21) :
  g.moves.length ≤ 4 ∧ ∃ (g' : GameState 20 21), g'.moves.length = 4 :=
sorry

end NUMINAMATH_CALUDE_max_moves_le_four_max_moves_21x21_max_moves_20x21_l3456_345616


namespace NUMINAMATH_CALUDE_five_dollar_four_equals_85_l3456_345636

/-- Custom operation $\$$ defined as a $ b = a(2b + 1) + 2ab -/
def dollar_op (a b : ℕ) : ℕ := a * (2 * b + 1) + 2 * a * b

/-- Theorem stating that 5 $ 4 = 85 -/
theorem five_dollar_four_equals_85 : dollar_op 5 4 = 85 := by
  sorry

end NUMINAMATH_CALUDE_five_dollar_four_equals_85_l3456_345636


namespace NUMINAMATH_CALUDE_bundle_sheets_value_l3456_345670

/-- The number of sheets in a bundle -/
def bundle_sheets : ℕ := 2

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def bunch_sheets : ℕ := 4

/-- The number of sheets in a heap -/
def heap_sheets : ℕ := 20

/-- The total number of sheets removed -/
def total_sheets : ℕ := 114

theorem bundle_sheets_value :
  colored_bundles * bundle_sheets + white_bunches * bunch_sheets + scrap_heaps * heap_sheets = total_sheets :=
by sorry

end NUMINAMATH_CALUDE_bundle_sheets_value_l3456_345670


namespace NUMINAMATH_CALUDE_christmas_decorations_distribution_l3456_345611

/-- The number of decorations in each box -/
def decorations_per_box : ℕ := 10

/-- The total number of decorations handed out -/
def total_decorations : ℕ := 120

/-- The number of families who received a box of decorations -/
def num_families : ℕ := 11

theorem christmas_decorations_distribution :
  decorations_per_box * (num_families + 1) = total_decorations :=
sorry

end NUMINAMATH_CALUDE_christmas_decorations_distribution_l3456_345611


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3456_345691

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b = a(cosC + (√3/3)sinC), a = √3, and c = 1, then C = π/6 -/
theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  b = a * (Real.cos C + (Real.sqrt 3 / 3) * Real.sin C) →
  a = Real.sqrt 3 →
  c = 1 →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3456_345691


namespace NUMINAMATH_CALUDE_unit_digit_of_sum_powers_l3456_345665

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def unitDigit (n : ℕ) : ℕ := n % 10

theorem unit_digit_of_sum_powers (a b c : ℕ) :
  unitDigit (a^(sumFactorials a) + b^(sumFactorials b) + c^(sumFactorials c)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_unit_digit_of_sum_powers_l3456_345665
