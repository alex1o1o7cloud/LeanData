import Mathlib

namespace NUMINAMATH_CALUDE_equal_integers_from_equation_l875_87525

/-- The least prime divisor of a positive integer greater than 1 -/
def least_prime_divisor (m : ℕ) : ℕ :=
  Nat.minFac m

theorem equal_integers_from_equation (a b : ℕ) 
  (ha : a > 1) (hb : b > 1)
  (h : a^2 + b = least_prime_divisor a + (least_prime_divisor b)^2) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_equal_integers_from_equation_l875_87525


namespace NUMINAMATH_CALUDE_three_zeros_range_l875_87590

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

-- Define the property of having 3 zeros
def has_three_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

-- Theorem statement
theorem three_zeros_range :
  ∀ a : ℝ, has_three_zeros a ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_range_l875_87590


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l875_87568

/-- Given a quadratic function f(x) = 3x^2 + 2x - 5, when shifted 6 units to the right,
    the resulting function g(x) = ax^2 + bx + c has coefficients that sum to 60. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, (3 * (x - 6)^2 + 2 * (x - 6) - 5) = (a * x^2 + b * x + c)) →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l875_87568


namespace NUMINAMATH_CALUDE_candy_store_lollipops_l875_87534

/-- The number of milliliters of food coloring used for each lollipop -/
def lollipop_coloring : ℕ := 5

/-- The number of milliliters of food coloring used for each hard candy -/
def hard_candy_coloring : ℕ := 20

/-- The number of hard candies made -/
def hard_candies_made : ℕ := 5

/-- The total amount of food coloring used in milliliters -/
def total_coloring_used : ℕ := 600

/-- The number of lollipops made -/
def lollipops_made : ℕ := 100

theorem candy_store_lollipops :
  lollipops_made * lollipop_coloring + hard_candies_made * hard_candy_coloring = total_coloring_used :=
by sorry

end NUMINAMATH_CALUDE_candy_store_lollipops_l875_87534


namespace NUMINAMATH_CALUDE_triangle_cosine_rule_l875_87514

/-- Given a triangle ABC where 6 sin A = 4 sin B = 3 sin C, prove that cos C = -1/4 -/
theorem triangle_cosine_rule (A B C : ℝ) (h : 6 * Real.sin A = 4 * Real.sin B ∧ 4 * Real.sin B = 3 * Real.sin C) :
  Real.cos C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_rule_l875_87514


namespace NUMINAMATH_CALUDE_radical_axis_intersection_squared_distance_l875_87586

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : Real)
  (BC : Real)
  (CA : Real)

-- Define the incircle and its touchpoints
structure Incircle :=
  (I : ℝ × ℝ)
  (M N D : ℝ × ℝ)

-- Define point K
def K (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry

-- Define circumcircles of triangles MAN and KID
def CircumcircleMAN (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry
def CircumcircleKID (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry

-- Define the radical axis
def RadicalAxis (c1 c2 : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

-- Define L₁ and L₂
def L₁ (t : Triangle) (ra : ℝ × ℝ → Prop) : ℝ × ℝ := sorry
def L₂ (t : Triangle) (ra : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

-- Distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem radical_axis_intersection_squared_distance
  (t : Triangle)
  (inc : Incircle)
  (h1 : t.AB = 36)
  (h2 : t.BC = 48)
  (h3 : t.CA = 60)
  (h4 : inc.M = sorry)  -- Point where incircle touches AB
  (h5 : inc.N = sorry)  -- Point where incircle touches AC
  (h6 : inc.D = sorry)  -- Point where incircle touches BC
  :
  let k := K t inc
  let c1 := CircumcircleMAN t inc
  let c2 := CircumcircleKID t inc
  let ra := RadicalAxis c1 c2
  let l1 := L₁ t ra
  let l2 := L₂ t ra
  (distance l1 l2)^2 = 720 := by sorry

end NUMINAMATH_CALUDE_radical_axis_intersection_squared_distance_l875_87586


namespace NUMINAMATH_CALUDE_chapter_length_l875_87569

theorem chapter_length (pages_per_chapter : ℕ) 
  (h1 : 10 * pages_per_chapter + 20 + 2 * pages_per_chapter = 500) :
  pages_per_chapter = 40 := by
  sorry

end NUMINAMATH_CALUDE_chapter_length_l875_87569


namespace NUMINAMATH_CALUDE_subset_implies_range_l875_87561

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

-- Define set M parameterized by a
def M (a : ℝ) : Set ℝ := {x | x^2 - (a+2)*x + 2*a ≤ 0}

-- Theorem statement
theorem subset_implies_range (a : ℝ) : M a ⊆ A ↔ a ∈ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_range_l875_87561


namespace NUMINAMATH_CALUDE_det_A_eq_cube_l875_87506

/-- The matrix A as defined in the problem -/
def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1 + x^2 - y^2 - z^2, 2*(x*y + z), 2*(z*x - y);
    2*(x*y - z), 1 + y^2 - z^2 - x^2, 2*(y*z + x);
    2*(z*x + y), 2*(y*z - x), 1 + z^2 - x^2 - y^2]

/-- The theorem stating that the determinant of A is equal to (1 + x^2 + y^2 + z^2)^3 -/
theorem det_A_eq_cube (x y z : ℝ) : 
  Matrix.det (A x y z) = (1 + x^2 + y^2 + z^2)^3 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_cube_l875_87506


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l875_87574

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l875_87574


namespace NUMINAMATH_CALUDE_function_composition_sqrt2_l875_87564

theorem function_composition_sqrt2 (a : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) :
  (∀ x, f x = a * x^2 - Real.sqrt 2) →
  f (f (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_sqrt2_l875_87564


namespace NUMINAMATH_CALUDE_option_2_cheaper_for_42_options_equal_at_45_unique_equality_at_45_l875_87524

-- Define the ticket price
def ticket_price : ℕ := 30

-- Define the discount rates
def discount_rate_1 : ℚ := 0.2
def discount_rate_2 : ℚ := 0.1

-- Define the number of free tickets in Option 2
def free_tickets : ℕ := 5

-- Function to calculate cost for Option 1
def cost_option_1 (students : ℕ) : ℚ :=
  (students : ℚ) * ticket_price * (1 - discount_rate_1)

-- Function to calculate cost for Option 2
def cost_option_2 (students : ℕ) : ℚ :=
  ((students - free_tickets) : ℚ) * ticket_price * (1 - discount_rate_2)

-- Theorem 1: For 42 students, Option 2 is cheaper
theorem option_2_cheaper_for_42 : cost_option_2 42 < cost_option_1 42 := by sorry

-- Theorem 2: Both options are equal when there are 45 students
theorem options_equal_at_45 : cost_option_1 45 = cost_option_2 45 := by sorry

-- Theorem 3: 45 is the only number of students (> 40) where both options are equal
theorem unique_equality_at_45 :
  ∀ n : ℕ, n > 40 → cost_option_1 n = cost_option_2 n → n = 45 := by sorry

end NUMINAMATH_CALUDE_option_2_cheaper_for_42_options_equal_at_45_unique_equality_at_45_l875_87524


namespace NUMINAMATH_CALUDE_quadratic_relation_l875_87507

/-- A quadratic function of the form y = -x² + 2x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- The y-coordinate of point P₁ -/
def y₁ (c : ℝ) : ℝ := f c (-1)

/-- The y-coordinate of point P₂ -/
def y₂ (c : ℝ) : ℝ := f c 2

/-- The y-coordinate of point P₃ -/
def y₃ (c : ℝ) : ℝ := f c 5

theorem quadratic_relation (c : ℝ) : y₂ c > y₁ c ∧ y₁ c > y₃ c :=
sorry

end NUMINAMATH_CALUDE_quadratic_relation_l875_87507


namespace NUMINAMATH_CALUDE_quilt_cost_calculation_l875_87566

/-- The cost of a rectangular quilt -/
def quilt_cost (length width cost_per_sqft : ℝ) : ℝ :=
  length * width * cost_per_sqft

/-- Theorem: The cost of a 7ft by 8ft quilt at $40 per square foot is $2240 -/
theorem quilt_cost_calculation :
  quilt_cost 7 8 40 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_calculation_l875_87566


namespace NUMINAMATH_CALUDE_board_structure_count_l875_87503

/-- The number of ways to structure a corporate board -/
def board_structures (n : ℕ) : ℕ :=
  let president_choices := n
  let vp_choices := n - 1
  let remaining_after_vps := n - 3
  let dh_choices_vp1 := remaining_after_vps.choose 3
  let dh_choices_vp2 := (remaining_after_vps - 3).choose 3
  president_choices * (vp_choices * (vp_choices - 1)) * dh_choices_vp1 * dh_choices_vp2

/-- Theorem stating the number of ways to structure a 13-member board -/
theorem board_structure_count :
  board_structures 13 = 655920 := by
  sorry

end NUMINAMATH_CALUDE_board_structure_count_l875_87503


namespace NUMINAMATH_CALUDE_dedekind_cut_property_l875_87578

-- Define a Dedekind cut
def DedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧ 
  (M ∩ N = ∅) ∧ 
  (∀ x ∈ M, ∀ y ∈ N, x < y) ∧
  M.Nonempty ∧ 
  N.Nonempty

-- Theorem stating the impossibility of M having a largest element and N having a smallest element
theorem dedekind_cut_property (M N : Set ℚ) (h : DedekindCut M N) :
  ¬(∃ (m : ℚ), m ∈ M ∧ ∀ x ∈ M, x ≤ m) ∨ ¬(∃ (n : ℚ), n ∈ N ∧ ∀ y ∈ N, n ≤ y) :=
sorry

end NUMINAMATH_CALUDE_dedekind_cut_property_l875_87578


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l875_87520

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + 2*x) * (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 253 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l875_87520


namespace NUMINAMATH_CALUDE_probability_one_has_no_growth_pie_l875_87543

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := total_pies - growth_pies
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ := 7/10

theorem probability_one_has_no_growth_pie :
  (1 : ℚ) - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given : ℚ) = probability_no_growth_pie :=
sorry

end NUMINAMATH_CALUDE_probability_one_has_no_growth_pie_l875_87543


namespace NUMINAMATH_CALUDE_probability_three_students_l875_87583

/-- The probability of having students participate on both Saturday and Sunday -/
def probability_both_days (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (2^n - 2) / 2^n

theorem probability_three_students :
  probability_both_days 3 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_students_l875_87583


namespace NUMINAMATH_CALUDE_pie_slices_left_l875_87511

theorem pie_slices_left (total_slices : ℕ) (half_given : ℚ) (quarter_given : ℚ) : 
  total_slices = 8 → half_given = 1/2 → quarter_given = 1/4 → 
  total_slices - (half_given * total_slices + quarter_given * total_slices) = 2 := by
sorry

end NUMINAMATH_CALUDE_pie_slices_left_l875_87511


namespace NUMINAMATH_CALUDE_prob_two_queens_or_two_aces_value_l875_87598

-- Define the deck
def total_cards : ℕ := 52
def num_aces : ℕ := 4
def num_queens : ℕ := 4

-- Define the probability function
noncomputable def prob_two_queens_or_two_aces : ℚ :=
  let two_queens := (num_queens.choose 2) * ((total_cards - num_queens).choose 1)
  let two_aces := (num_aces.choose 2) * ((total_cards - num_aces).choose 1)
  let three_aces := num_aces.choose 3
  (two_queens + two_aces + three_aces) / (total_cards.choose 3)

-- State the theorem
theorem prob_two_queens_or_two_aces_value : 
  prob_two_queens_or_two_aces = 29 / 1105 := by sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_two_aces_value_l875_87598


namespace NUMINAMATH_CALUDE_min_distance_to_line_l875_87577

/-- Given that 5x + 12y = 60, the minimum value of √(x² + y²) is 60/13 -/
theorem min_distance_to_line (x y : ℝ) (h : 5 * x + 12 * y = 60) :
  ∃ (min_val : ℝ), min_val = 60 / 13 ∧ 
  ∀ (x' y' : ℝ), 5 * x' + 12 * y' = 60 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l875_87577


namespace NUMINAMATH_CALUDE_frank_lawn_money_l875_87521

/-- The amount of money Frank made mowing lawns -/
def lawn_money : ℕ := 19

/-- The cost of mower blades -/
def blade_cost : ℕ := 11

/-- The number of games Frank could buy -/
def num_games : ℕ := 4

/-- The cost of each game -/
def game_cost : ℕ := 2

/-- Theorem stating that the money Frank made mowing lawns is correct -/
theorem frank_lawn_money :
  lawn_money = blade_cost + num_games * game_cost :=
by sorry

end NUMINAMATH_CALUDE_frank_lawn_money_l875_87521


namespace NUMINAMATH_CALUDE_point_C_coordinates_l875_87581

-- Define the points A and B
def A : ℝ × ℝ := (-2, -1)
def B : ℝ × ℝ := (4, 9)

-- Define the condition for point C
def is_point_C (C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  C.1 = A.1 + t * (B.1 - A.1) ∧
  C.2 = A.2 + t * (B.2 - A.2) ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 16 * ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Theorem statement
theorem point_C_coordinates :
  ∃ C : ℝ × ℝ, is_point_C C ∧ C = (-0.8, 1) :=
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l875_87581


namespace NUMINAMATH_CALUDE_thousand_factorial_zeroes_l875_87563

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- n! is the product of integers from 1 to n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem thousand_factorial_zeroes :
  trailingZeroes 1000 = 249 :=
sorry

end NUMINAMATH_CALUDE_thousand_factorial_zeroes_l875_87563


namespace NUMINAMATH_CALUDE_jet_distance_l875_87558

/-- Given a jet that travels 580 miles in 2 hours, prove that it will travel 2900 miles in 10 hours. -/
theorem jet_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) 
    (h1 : distance = 580) 
    (h2 : time = 2) 
    (h3 : new_time = 10) : 
  (distance / time) * new_time = 2900 := by
  sorry

end NUMINAMATH_CALUDE_jet_distance_l875_87558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_24th_term_l875_87509

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 7 and the 10th term is 27,
    the 24th term is 67. -/
theorem arithmetic_sequence_24th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 7)
  (h_10th : a 10 = 27) :
  a 24 = 67 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_24th_term_l875_87509


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l875_87592

theorem quadratic_equation_unique_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ↔ 
  (c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l875_87592


namespace NUMINAMATH_CALUDE_base_nine_to_ten_l875_87587

theorem base_nine_to_ten : 
  (3 * 9^3 + 7 * 9^2 + 2 * 9^1 + 5 * 9^0) = 2777 := by
  sorry

end NUMINAMATH_CALUDE_base_nine_to_ten_l875_87587


namespace NUMINAMATH_CALUDE_dog_reachable_area_is_8pi_l875_87516

/-- The area a dog can reach when tethered to a vertex of a regular hexagonal doghouse -/
def dogReachableArea (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  -- Define the area calculation here
  sorry

/-- Theorem stating the area a dog can reach for the given conditions -/
theorem dog_reachable_area_is_8pi :
  dogReachableArea 2 3 = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_reachable_area_is_8pi_l875_87516


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l875_87584

/-- Represents a parabola in the form y = a(x-h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 4 ∧ p.k = 3 →
  let p' := shift p 4 (-4)
  p'.a * X ^ 2 + p'.a * p'.h ^ 2 - 2 * p'.a * p'.h * X + p'.k = 3 * X ^ 2 - 1 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l875_87584


namespace NUMINAMATH_CALUDE_remainder_mod_88_l875_87556

theorem remainder_mod_88 : (1 - 90) ^ 10 ≡ 1 [MOD 88] := by sorry

end NUMINAMATH_CALUDE_remainder_mod_88_l875_87556


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l875_87585

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(61*y)) :
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l875_87585


namespace NUMINAMATH_CALUDE_train_probabilities_l875_87595

/-- Three independent events with given probabilities -/
structure ThreeIndependentEvents where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ
  p1_in_range : 0 ≤ p1 ∧ p1 ≤ 1
  p2_in_range : 0 ≤ p2 ∧ p2 ≤ 1
  p3_in_range : 0 ≤ p3 ∧ p3 ≤ 1

/-- The probability of exactly two events occurring -/
def prob_exactly_two (e : ThreeIndependentEvents) : ℝ :=
  e.p1 * e.p2 * (1 - e.p3) + e.p1 * (1 - e.p2) * e.p3 + (1 - e.p1) * e.p2 * e.p3

/-- The probability of at least one event occurring -/
def prob_at_least_one (e : ThreeIndependentEvents) : ℝ :=
  1 - (1 - e.p1) * (1 - e.p2) * (1 - e.p3)

/-- Theorem stating the probabilities for the given scenario -/
theorem train_probabilities (e : ThreeIndependentEvents) 
  (h1 : e.p1 = 0.8) (h2 : e.p2 = 0.7) (h3 : e.p3 = 0.9) : 
  prob_exactly_two e = 0.398 ∧ prob_at_least_one e = 0.994 := by
  sorry

end NUMINAMATH_CALUDE_train_probabilities_l875_87595


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l875_87540

theorem sum_of_squares_of_roots (a b c : ℂ) : 
  (2 * a^3 - a^2 + 4*a + 10 = 0) → 
  (2 * b^3 - b^2 + 4*b + 10 = 0) → 
  (2 * c^3 - c^2 + 4*c + 10 = 0) → 
  a^2 + b^2 + c^2 = -15/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l875_87540


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l875_87549

theorem x_range_for_quadratic_inequality (x : ℝ) :
  (∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4)*x + 4 - 2*a > 0) →
  x ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l875_87549


namespace NUMINAMATH_CALUDE_lucas_siblings_product_l875_87518

/-- A family with Lauren and Lucas as members -/
structure Family where
  lauren_sisters : ℕ
  lauren_brothers : ℕ
  lucas : Member

/-- A member of the family -/
inductive Member
  | Lauren
  | Lucas
  | OtherSister
  | OtherBrother

/-- The number of sisters Lucas has in the family -/
def lucas_sisters (f : Family) : ℕ :=
  f.lauren_sisters + 1

/-- The number of brothers Lucas has in the family -/
def lucas_brothers (f : Family) : ℕ :=
  f.lauren_brothers - 1

theorem lucas_siblings_product (f : Family) 
  (h1 : f.lauren_sisters = 4)
  (h2 : f.lauren_brothers = 7)
  (h3 : f.lucas = Member.Lucas) :
  lucas_sisters f * lucas_brothers f = 35 := by
  sorry

end NUMINAMATH_CALUDE_lucas_siblings_product_l875_87518


namespace NUMINAMATH_CALUDE_option2_more_cost_effective_l875_87536

/-- The cost of a pair of badminton rackets in dollars -/
def racket_cost : ℕ := 100

/-- The cost of a box of shuttlecocks in dollars -/
def shuttlecock_cost : ℕ := 20

/-- The number of pairs of badminton rackets the school wants to buy -/
def racket_pairs : ℕ := 10

/-- The number of boxes of shuttlecocks the school wants to buy -/
def shuttlecock_boxes : ℕ := 60

/-- The cost of Option 1 in dollars -/
def option1_cost (x : ℕ) : ℕ := 20 * x + 800

/-- The cost of Option 2 in dollars -/
def option2_cost (x : ℕ) : ℕ := 18 * x + 900

/-- Theorem stating that Option 2 is more cost-effective when x = 60 -/
theorem option2_more_cost_effective :
  shuttlecock_boxes > 10 →
  option1_cost shuttlecock_boxes > option2_cost shuttlecock_boxes :=
by sorry

end NUMINAMATH_CALUDE_option2_more_cost_effective_l875_87536


namespace NUMINAMATH_CALUDE_rower_round_trip_time_l875_87537

/-- Proves that the total time to row to Big Rock and back is 1 hour -/
theorem rower_round_trip_time
  (rower_speed : ℝ)
  (river_speed : ℝ)
  (distance : ℝ)
  (h1 : rower_speed = 7)
  (h2 : river_speed = 2)
  (h3 : distance = 3.2142857142857144)
  : (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_rower_round_trip_time_l875_87537


namespace NUMINAMATH_CALUDE_digits_after_decimal_point_l875_87560

theorem digits_after_decimal_point : ∃ (n : ℕ), 
  (5^7 : ℚ) / (10^5 * 15625) = (1 : ℚ) / 10^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_digits_after_decimal_point_l875_87560


namespace NUMINAMATH_CALUDE_uncle_bob_can_park_l875_87528

-- Define the number of total spaces and parked cars
def total_spaces : ℕ := 18
def parked_cars : ℕ := 14

-- Define a function to calculate the number of ways to distribute n items into k groups
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

-- Define the probability of Uncle Bob finding a parking space
def uncle_bob_parking_probability : ℚ :=
  1 - (stars_and_bars 7 5 : ℚ) / (Nat.choose total_spaces parked_cars : ℚ)

-- Theorem statement
theorem uncle_bob_can_park : 
  uncle_bob_parking_probability = 91 / 102 :=
sorry

end NUMINAMATH_CALUDE_uncle_bob_can_park_l875_87528


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l875_87579

/-- Given a circle with diameter endpoints (2, -3) and (8, 5), 
    prove that its center is at (5, 1) and its radius is 5. -/
theorem circle_center_and_radius : 
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (8, 5)
  let center : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  let radius : ℝ := Real.sqrt ((center.1 - a.1)^2 + (center.2 - a.2)^2)
  center = (5, 1) ∧ radius = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l875_87579


namespace NUMINAMATH_CALUDE_jack_morning_letters_l875_87589

def morning_letters (afternoon_letters : ℕ) (difference : ℕ) : ℕ :=
  afternoon_letters + difference

theorem jack_morning_letters :
  morning_letters 7 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_morning_letters_l875_87589


namespace NUMINAMATH_CALUDE_jamie_earnings_l875_87526

def total_earnings (hourly_rate : ℝ) (days_per_week : ℕ) (hours_per_day : ℕ) (weeks_worked : ℕ) : ℝ :=
  hourly_rate * (days_per_week * hours_per_day * weeks_worked)

theorem jamie_earnings : 
  let hourly_rate : ℝ := 10
  let days_per_week : ℕ := 2
  let hours_per_day : ℕ := 3
  let weeks_worked : ℕ := 6
  total_earnings hourly_rate days_per_week hours_per_day weeks_worked = 360 := by
  sorry

end NUMINAMATH_CALUDE_jamie_earnings_l875_87526


namespace NUMINAMATH_CALUDE_no_prime_solution_to_equation_l875_87530

theorem no_prime_solution_to_equation :
  ∀ p q r s t : ℕ, 
    Prime p → Prime q → Prime r → Prime s → Prime t →
    p^2 + q^2 ≠ r^2 + s^2 + t^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_to_equation_l875_87530


namespace NUMINAMATH_CALUDE_interest_rate_problem_l875_87594

/-- Given a total sum and a second part, calculates the interest rate of the first part
    such that the interest on the first part for 8 years equals the interest on the second part for 3 years at 5% --/
def calculate_interest_rate (total_sum : ℚ) (second_part : ℚ) : ℚ :=
  let first_part := total_sum - second_part
  let second_part_interest := second_part * 5 * 3 / 100
  second_part_interest * 100 / (first_part * 8)

theorem interest_rate_problem (total_sum : ℚ) (second_part : ℚ) 
  (h1 : total_sum = 2769)
  (h2 : second_part = 1704) :
  calculate_interest_rate total_sum second_part = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l875_87594


namespace NUMINAMATH_CALUDE_jo_stair_climbing_l875_87546

/-- The number of ways to climb n stairs, taking 1, 2, or 3 stairs at a time -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

/-- The number of stairs Jo climbs -/
def totalStairs : ℕ := 8

theorem jo_stair_climbing :
  climbStairs totalStairs = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_jo_stair_climbing_l875_87546


namespace NUMINAMATH_CALUDE_total_copies_is_7050_l875_87522

/-- The total number of copies made by four copy machines in 30 minutes -/
def total_copies : ℕ :=
  let machine1 := 35 * 30
  let machine2 := 65 * 30
  let machine3 := 50 * 15 + 80 * 15
  let machine4 := 90 * 10 + 60 * 20
  machine1 + machine2 + machine3 + machine4

/-- Theorem stating that the total number of copies made by the four machines in 30 minutes is 7050 -/
theorem total_copies_is_7050 : total_copies = 7050 := by
  sorry

end NUMINAMATH_CALUDE_total_copies_is_7050_l875_87522


namespace NUMINAMATH_CALUDE_proportion_property_l875_87567

theorem proportion_property (a b c d : ℝ) (h : a / b = c / d) : b * c - a * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_proportion_property_l875_87567


namespace NUMINAMATH_CALUDE_couscous_shipment_l875_87523

theorem couscous_shipment (first_shipment second_shipment num_dishes couscous_per_dish : ℕ)
  (h1 : first_shipment = 7)
  (h2 : second_shipment = 13)
  (h3 : num_dishes = 13)
  (h4 : couscous_per_dish = 5) :
  let total_used := num_dishes * couscous_per_dish
  let first_two_shipments := first_shipment + second_shipment
  total_used - first_two_shipments = 45 := by
    sorry

end NUMINAMATH_CALUDE_couscous_shipment_l875_87523


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l875_87572

def i : ℂ := Complex.I

theorem imaginary_part_of_pure_imaginary_z (a : ℝ) :
  let z : ℂ := a + 15 / (3 - 4 * i)
  (z.re = 0) → z.im = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l875_87572


namespace NUMINAMATH_CALUDE_yellow_balls_after_loss_l875_87580

theorem yellow_balls_after_loss (initial_total : ℕ) (current_total : ℕ) (blue : ℕ) (lost : ℕ) : 
  initial_total = 120 →
  current_total = 110 →
  blue = 15 →
  lost = 10 →
  let red := 3 * blue
  let green := red + blue
  let yellow := initial_total - (red + blue + green)
  yellow = 0 := by sorry

end NUMINAMATH_CALUDE_yellow_balls_after_loss_l875_87580


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l875_87501

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_02 : ℚ := 2/99

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_02 = 35/99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l875_87501


namespace NUMINAMATH_CALUDE_shelves_needed_l875_87576

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 46)
  (h2 : books_taken = 10)
  (h3 : books_per_shelf = 4)
  (h4 : books_per_shelf > 0) :
  (total_books - books_taken) / books_per_shelf = 9 :=
by sorry

end NUMINAMATH_CALUDE_shelves_needed_l875_87576


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l875_87547

/-- Given a triangle ABC with angle ratio ∠A:∠B:∠C = 3:4:5, it cannot be concluded that ABC is a right triangle. -/
theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / (A + B + C) = 3 / 12 ∧ B / (A + B + C) = 4 / 12 ∧ C / (A + B + C) = 5 / 12) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l875_87547


namespace NUMINAMATH_CALUDE_offspring_different_genes_l875_87553

structure Eukaryote where
  genes : Set String

def sexualReproduction (parent1 parent2 : Eukaryote) : Eukaryote :=
  sorry

theorem offspring_different_genes (parent1 parent2 : Eukaryote) :
  let offspring := sexualReproduction parent1 parent2
  ∃ (gene : String), (gene ∈ offspring.genes ∧ gene ∉ parent1.genes) ∨
                     (gene ∈ offspring.genes ∧ gene ∉ parent2.genes) :=
  sorry

end NUMINAMATH_CALUDE_offspring_different_genes_l875_87553


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l875_87512

/-- Given a 9x16 rectangle that is cut into two congruent quadrilaterals
    which can be repositioned to form a square, the side length z of
    one quadrilateral is 12. -/
theorem quadrilateral_side_length (z : ℝ) : z = 12 :=
  let rectangle_area : ℝ := 9 * 16
  let square_side : ℝ := Real.sqrt rectangle_area
  sorry


end NUMINAMATH_CALUDE_quadrilateral_side_length_l875_87512


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l875_87551

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l875_87551


namespace NUMINAMATH_CALUDE_complex_cube_root_l875_87591

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a - ↑b * Complex.I) ^ 3 = 27 - 27 * Complex.I →
  ↑a - ↑b * Complex.I = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l875_87591


namespace NUMINAMATH_CALUDE_unique_assignment_l875_87515

-- Define the type for tables
inductive Table
| T1 | T2 | T3 | T4

-- Define the type for students
inductive Student
| Albert | Bogdan | Vadim | Denis

-- Define a function to represent the assignment of tables to students
def assignment : Student → Table
| Student.Albert => Table.T4
| Student.Bogdan => Table.T2
| Student.Vadim => Table.T1
| Student.Denis => Table.T3

-- Define a predicate for table intersection
def intersects (t1 t2 : Table) : Prop := sorry

-- Albert and Bogdan colored some cells
axiom albert_bogdan_colored : ∀ (t : Table), t ≠ Table.T1 → intersects (assignment Student.Albert) t ∨ intersects (assignment Student.Bogdan) t

-- Vadim's table doesn't intersect with Albert's or Bogdan's
axiom vadim_condition : ¬(intersects (assignment Student.Vadim) (assignment Student.Albert)) ∧ 
                        ¬(intersects (assignment Student.Vadim) (assignment Student.Bogdan))

-- Denis's table doesn't intersect with Bogdan's or Vadim's
axiom denis_condition : ¬(intersects (assignment Student.Denis) (assignment Student.Bogdan)) ∧ 
                        ¬(intersects (assignment Student.Denis) (assignment Student.Vadim))

-- Theorem stating that the given assignment is the only valid solution
theorem unique_assignment : 
  ∀ (f : Student → Table), 
    (∀ (s1 s2 : Student), s1 ≠ s2 → f s1 ≠ f s2) →
    (∀ (t : Table), t ≠ Table.T1 → intersects (f Student.Albert) t ∨ intersects (f Student.Bogdan) t) →
    (¬(intersects (f Student.Vadim) (f Student.Albert)) ∧ ¬(intersects (f Student.Vadim) (f Student.Bogdan))) →
    (¬(intersects (f Student.Denis) (f Student.Bogdan)) ∧ ¬(intersects (f Student.Denis) (f Student.Vadim))) →
    f = assignment := by sorry

end NUMINAMATH_CALUDE_unique_assignment_l875_87515


namespace NUMINAMATH_CALUDE_divisibility_by_1897_l875_87552

theorem divisibility_by_1897 (n : ℕ) : 
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1897_l875_87552


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l875_87559

theorem sqrt_product_equality : Real.sqrt 50 * Real.sqrt 18 * Real.sqrt 8 = 60 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l875_87559


namespace NUMINAMATH_CALUDE_polynomial_simplification_l875_87539

theorem polynomial_simplification (x : ℝ) : 
  (2*x^2 + 3*x + 7)*(x + 1) - (x + 1)*(x^2 + 4*x - 63) + (3*x - 14)*(x + 1)*(x + 5) = 4*x^3 + 4*x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l875_87539


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l875_87582

theorem abs_sum_inequality (x : ℝ) : |x - 2| + |x + 3| < 7 ↔ -6 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l875_87582


namespace NUMINAMATH_CALUDE_even_times_odd_is_odd_l875_87599

variable (f g : ℝ → ℝ)

-- Define even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem even_times_odd_is_odd (hf : IsEven f) (hg : IsOdd g) : IsOdd (fun x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_even_times_odd_is_odd_l875_87599


namespace NUMINAMATH_CALUDE_systematic_sampling_l875_87502

theorem systematic_sampling (population : ℕ) (sample_size : ℕ) 
  (h_pop : population = 1650) (h_sample : sample_size = 35) :
  ∃ (removed : ℕ) (segments : ℕ) (per_segment : ℕ),
    removed = 5 ∧ 
    segments = sample_size ∧
    per_segment = 47 ∧
    (population - removed) = segments * per_segment :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l875_87502


namespace NUMINAMATH_CALUDE_dot_product_AP_BP_l875_87571

/-- The dot product of vectors AP and BP, where P is a point on a specific ellipse satisfying certain conditions. -/
theorem dot_product_AP_BP : ∃ (x y : ℝ), 
  (x^2 / 12 + y^2 / 16 = 1) ∧ 
  (((x - 0)^2 + (y - (-2))^2).sqrt - ((x - 0)^2 + (y - 2)^2).sqrt = 2) →
  (x * x + (y + 2) * (y - 2) = 9) := by
sorry

end NUMINAMATH_CALUDE_dot_product_AP_BP_l875_87571


namespace NUMINAMATH_CALUDE_cars_to_double_earnings_l875_87575

def base_salary : ℕ := 1000
def commission_per_car : ℕ := 200
def january_earnings : ℕ := 1800

theorem cars_to_double_earnings : 
  ∃ (february_cars : ℕ), 
    base_salary + february_cars * commission_per_car = 2 * january_earnings ∧ 
    february_cars = 13 :=
by sorry

end NUMINAMATH_CALUDE_cars_to_double_earnings_l875_87575


namespace NUMINAMATH_CALUDE_binomial_18_10_l875_87570

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 8008) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l875_87570


namespace NUMINAMATH_CALUDE_first_discount_percentage_l875_87538

/-- Given an initial price of 400, a final price of 240 after two discounts,
    where the second discount is 20%, prove that the first discount is 25%. -/
theorem first_discount_percentage 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) : 
  initial_price = 400 →
  final_price = 240 →
  second_discount = 20 →
  ∃ (first_discount : ℝ),
    first_discount = 25 ∧ 
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l875_87538


namespace NUMINAMATH_CALUDE_total_students_calculation_l875_87555

/-- Represents a high school with three years of students -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Represents a sample taken from the high school -/
structure Sample where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- The theorem stating the conditions and the conclusion about the total number of students -/
theorem total_students_calculation (school : HighSchool) (sample : Sample) :
  school.second_year = 300 →
  sample.first_year = 20 →
  sample.third_year = 10 →
  sample.first_year + sample.second_year + sample.third_year = 45 →
  (sample.first_year : ℚ) / sample.third_year = 2 →
  (sample.first_year : ℚ) / school.first_year = 
    (sample.second_year : ℚ) / school.second_year →
  (sample.second_year : ℚ) / school.second_year = 
    (sample.third_year : ℚ) / school.third_year →
  school.first_year + school.second_year + school.third_year = 900 := by
  sorry


end NUMINAMATH_CALUDE_total_students_calculation_l875_87555


namespace NUMINAMATH_CALUDE_triangle_properties_l875_87519

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.hasAltitude (t : Triangle) (l : Line) : Prop :=
  l.a * t.A.1 + l.b * t.A.2 + l.c = 0

def Triangle.hasAngleBisector (t : Triangle) (l : Line) : Prop :=
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0

theorem triangle_properties (t : Triangle) (altitude : Line) (bisector : Line) :
  t.A = (1, 1) →
  altitude = { a := 3, b := 1, c := -12 } →
  bisector = { a := 1, b := -2, c := 4 } →
  t.hasAltitude altitude →
  t.hasAngleBisector bisector →
  t.B = (-8, -2) ∧
  (∃ (l : Line), l = { a := 9, b := -13, c := 46 } ∧ 
    l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
    l.a * t.C.1 + l.b * t.C.2 + l.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l875_87519


namespace NUMINAMATH_CALUDE_S_subset_T_l875_87565

-- Define set S
def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}

-- Define set T
def T : Set ℕ := {x | ∃ n : ℕ, x = 3*n}

-- Theorem stating S is a subset of T
theorem S_subset_T : S ⊆ T := by
  sorry

end NUMINAMATH_CALUDE_S_subset_T_l875_87565


namespace NUMINAMATH_CALUDE_mod_equation_l875_87533

theorem mod_equation (m : ℕ) (h1 : m < 37) (h2 : (4 * m) % 37 = 1) :
  (3^m)^2 % 37 - 3 % 37 = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_equation_l875_87533


namespace NUMINAMATH_CALUDE_festival_expense_sharing_l875_87517

theorem festival_expense_sharing 
  (C D X : ℝ) 
  (h1 : C > D) 
  (h2 : C > 0) 
  (h3 : D > 0) 
  (h4 : X > 0) :
  let total_expense := C + D + X
  let alex_share := (2/3) * total_expense
  let morgan_share := (1/3) * total_expense
  let alex_paid := C + X/2
  let morgan_paid := D + X/2
  morgan_share - morgan_paid = (1/3)*C - (2/3)*D + X := by
sorry

end NUMINAMATH_CALUDE_festival_expense_sharing_l875_87517


namespace NUMINAMATH_CALUDE_triangle_reconstruction_l875_87562

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle) : Point := sorry

/-- Represents the foot of altitude from C to AB -/
def altitudeFootC (t : Triangle) : Point := sorry

/-- Represents the C-excenter of a triangle -/
def excenterC (t : Triangle) : Point := sorry

/-- Theorem: Given the incenter, foot of altitude from C, and C-excenter, 
    a unique triangle can be reconstructed -/
theorem triangle_reconstruction 
  (I : Point) (H : Point) (I_c : Point) : 
  ∃! (t : Triangle), 
    incenter t = I ∧ 
    altitudeFootC t = H ∧ 
    excenterC t = I_c := by sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_l875_87562


namespace NUMINAMATH_CALUDE_sequence_sum_product_l875_87545

def sequence_property (α β γ : ℕ) (a b : ℕ → ℕ) : Prop :=
  (α < γ) ∧
  (α * γ = β^2 + 1) ∧
  (a 0 = 1) ∧ (b 0 = 1) ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n)

theorem sequence_sum_product (α β γ : ℕ) (a b : ℕ → ℕ) 
  (h : sequence_property α β γ a b) :
  ∀ m n, a (m + n) + b (m + n) = a m * a n + b m * b n :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_product_l875_87545


namespace NUMINAMATH_CALUDE_alice_winning_equivalence_l875_87596

/-- The game constant k, which is greater than 2 -/
def k : ℕ := sorry

/-- Definition of Alice-winning number -/
def is_alice_winning (n : ℕ) : Prop := sorry

/-- The radical of a number n with respect to k -/
def radical (n : ℕ) : ℕ := sorry

theorem alice_winning_equivalence (l l' : ℕ) 
  (h : ∀ p : ℕ, p.Prime → p ≤ k → (p ∣ l ↔ p ∣ l')) : 
  is_alice_winning l ↔ is_alice_winning l' := by sorry

end NUMINAMATH_CALUDE_alice_winning_equivalence_l875_87596


namespace NUMINAMATH_CALUDE_jellybean_count_l875_87532

/-- The number of jellybeans in a bag with specific color distributions -/
def total_jellybeans (black green orange red yellow : ℕ) : ℕ :=
  black + green + orange + red + yellow

/-- Theorem stating the total number of jellybeans in the bag -/
theorem jellybean_count : ∃ (black green orange red yellow : ℕ),
  black = 8 ∧
  green = black + 4 ∧
  orange = green - 5 ∧
  red = orange + 3 ∧
  yellow = black - 2 ∧
  total_jellybeans black green orange red yellow = 43 := by
  sorry


end NUMINAMATH_CALUDE_jellybean_count_l875_87532


namespace NUMINAMATH_CALUDE_grid_shading_l875_87542

/-- Given a 4 × 5 grid with 3 squares already shaded, 
    prove that 7 additional squares need to be shaded 
    to have half of all squares shaded. -/
theorem grid_shading (grid_width : Nat) (grid_height : Nat) 
  (total_squares : Nat) (already_shaded : Nat) (half_squares : Nat) 
  (additional_squares : Nat) : 
  grid_width = 4 → 
  grid_height = 5 → 
  total_squares = grid_width * grid_height →
  already_shaded = 3 →
  half_squares = total_squares / 2 →
  additional_squares = half_squares - already_shaded →
  additional_squares = 7 := by
sorry


end NUMINAMATH_CALUDE_grid_shading_l875_87542


namespace NUMINAMATH_CALUDE_sum_of_powers_inequality_l875_87557

theorem sum_of_powers_inequality (x : ℝ) (hx : x > 0) :
  1 + x + x^2 + x^3 + x^4 ≥ 5 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_inequality_l875_87557


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l875_87527

-- Define the triangle ABC and points D, E, P
variable (A B C D E P : ℝ × ℝ)

-- Define the conditions
def D_on_BC_extended (A B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = B + t • (C - B)

def E_on_AC (A C E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = A + t • (C - A)

def BD_DC_ratio (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ D = B + (2/3) • (C - B)

def AE_EC_ratio (A C E : ℝ × ℝ) : Prop :=
  E = A + (2/3) • (C - A)

def P_on_BE (B E P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = B + t • (E - B)

def P_on_AD (A D P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (D - A)

-- State the theorem
theorem intersection_point_coordinates
  (h1 : D_on_BC_extended A B C D)
  (h2 : E_on_AC A C E)
  (h3 : BD_DC_ratio B C D)
  (h4 : AE_EC_ratio A C E)
  (h5 : P_on_BE B E P)
  (h6 : P_on_AD A D P) :
  P = (1/3) • A + (1/2) • B + (1/6) • C :=
sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l875_87527


namespace NUMINAMATH_CALUDE_diaries_calculation_l875_87554

theorem diaries_calculation (initial_diaries : ℕ) : initial_diaries = 8 →
  (initial_diaries + 2 * initial_diaries) * 3 / 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_diaries_calculation_l875_87554


namespace NUMINAMATH_CALUDE_correct_time_per_lap_l875_87500

/-- The time in minutes for one lap around the playground -/
def time_per_lap : ℝ := 19.2

/-- The number of laps cycled -/
def num_laps : ℕ := 5

/-- The total time in minutes for cycling the given number of laps -/
def total_time : ℝ := 96

theorem correct_time_per_lap : 
  time_per_lap * num_laps = total_time := by sorry

end NUMINAMATH_CALUDE_correct_time_per_lap_l875_87500


namespace NUMINAMATH_CALUDE_square_sum_equals_three_l875_87550

theorem square_sum_equals_three (a b : ℝ) 
  (h : a^4 + b^4 = a^2 - 2*a^2*b^2 + b^2 + 6) : 
  a^2 + b^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_l875_87550


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l875_87541

/-- Represents the juice extraction information for a fruit type -/
structure FruitJuice where
  fruitCount : ℕ
  juiceAmount : ℕ

/-- Represents the blend composition -/
structure Blend where
  pearCount : ℕ
  orangeCount : ℕ

def calculateJuicePercentage (pearJuice : FruitJuice) (orangeJuice : FruitJuice) (blend : Blend) : ℚ :=
  let pearJuiceRate := pearJuice.juiceAmount / pearJuice.fruitCount
  let orangeJuiceRate := orangeJuice.juiceAmount / orangeJuice.fruitCount
  let totalPearJuice := pearJuiceRate * blend.pearCount
  let totalOrangeJuice := orangeJuiceRate * blend.orangeCount
  let totalJuice := totalPearJuice + totalOrangeJuice
  totalOrangeJuice / totalJuice

theorem orange_juice_percentage
  (pearJuice : FruitJuice)
  (orangeJuice : FruitJuice)
  (blend : Blend)
  (h1 : pearJuice.fruitCount = 5 ∧ pearJuice.juiceAmount = 10)
  (h2 : orangeJuice.fruitCount = 4 ∧ orangeJuice.juiceAmount = 12)
  (h3 : blend.pearCount = 9 ∧ blend.orangeCount = 6) :
  calculateJuicePercentage pearJuice orangeJuice blend = 1/2 := by
  sorry

#eval calculateJuicePercentage ⟨5, 10⟩ ⟨4, 12⟩ ⟨9, 6⟩

end NUMINAMATH_CALUDE_orange_juice_percentage_l875_87541


namespace NUMINAMATH_CALUDE_math_book_cost_l875_87544

/-- Proves that the cost of each math book is $4 given the conditions of the book purchase problem. -/
theorem math_book_cost (total_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) (math_books : ℕ) :
  total_books = 80 →
  history_book_cost = 5 →
  total_cost = 390 →
  math_books = 10 →
  (total_books - math_books) * history_book_cost + math_books * 4 = total_cost :=
by
  sorry

#check math_book_cost

end NUMINAMATH_CALUDE_math_book_cost_l875_87544


namespace NUMINAMATH_CALUDE_solution_approximation_l875_87513

theorem solution_approximation : ∃ x : ℝ, 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 0.225) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l875_87513


namespace NUMINAMATH_CALUDE_sequence_b_is_geometric_progression_l875_87535

def sequence_a (a : ℝ) (n : ℕ) : ℝ := 
  if n = 1 then a else 3 * (4 ^ (n - 1)) + 2 * (a - 4) * (3 ^ (n - 2))

def sum_S (a : ℝ) (n : ℕ) : ℝ := 
  (4 ^ n) + (a - 4) * (3 ^ (n - 1))

def sequence_b (a : ℝ) (n : ℕ) : ℝ := 
  sum_S a n - (4 ^ n)

theorem sequence_b_is_geometric_progression (a : ℝ) (h : a ≠ 4) :
  ∀ n : ℕ, n ≥ 1 → sequence_b a (n + 1) = 3 * sequence_b a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_b_is_geometric_progression_l875_87535


namespace NUMINAMATH_CALUDE_orangeade_pricing_l875_87508

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℚ
  day : Nat

/-- Represents the volume of orangeade produced on a given day -/
structure OrangeadeVolume where
  volume : ℚ
  day : Nat

def orangeade_revenue (price : OrangeadePrice) (volume : OrangeadeVolume) : ℚ :=
  price.price * volume.volume

theorem orangeade_pricing
  (day1_volume : OrangeadeVolume)
  (day2_volume : OrangeadeVolume)
  (day1_price : OrangeadePrice)
  (day2_price : OrangeadePrice)
  (h_day1 : day1_volume.day = 1)
  (h_day2 : day2_volume.day = 2)
  (h_volume_ratio : day2_volume.volume = (3/2) * day1_volume.volume)
  (h_equal_revenue : orangeade_revenue day1_price day1_volume = orangeade_revenue day2_price day2_volume)
  (h_day2_price : day2_price.price = 2/5)
  : day1_price.price = 3/5 := by
  sorry

#check orangeade_pricing

end NUMINAMATH_CALUDE_orangeade_pricing_l875_87508


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l875_87597

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 20) :
  (1 / a + 1 / b) ≥ (1 / 5 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l875_87597


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l875_87593

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l875_87593


namespace NUMINAMATH_CALUDE_no_valid_gnomon_tiling_l875_87529

/-- A gnomon is a figure formed by removing one unit square from a 2x2 square -/
def Gnomon : Type := Unit

/-- Represents a tiling of an m × n rectangle with gnomons -/
def GnomonTiling (m n : ℕ) := Unit

/-- Predicate to check if a tiling satisfies the no-rectangle condition -/
def NoRectangleCondition (tiling : GnomonTiling m n) : Prop := sorry

/-- Predicate to check if a tiling satisfies the no-four-vertex condition -/
def NoFourVertexCondition (tiling : GnomonTiling m n) : Prop := sorry

theorem no_valid_gnomon_tiling (m n : ℕ) :
  ¬∃ (tiling : GnomonTiling m n), NoRectangleCondition tiling ∧ NoFourVertexCondition tiling := by
  sorry

end NUMINAMATH_CALUDE_no_valid_gnomon_tiling_l875_87529


namespace NUMINAMATH_CALUDE_purely_imaginary_value_l875_87505

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_value (a : ℝ) :
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 1)
  is_purely_imaginary z → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_value_l875_87505


namespace NUMINAMATH_CALUDE_stratified_sampling_c_l875_87573

/-- Represents the number of individuals in each sample -/
structure SampleSizes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The ratio of individuals in samples A, B, and C -/
def sample_ratio : SampleSizes := { A := 5, B := 3, C := 2 }

/-- The total sample size for stratified sampling -/
def total_sample_size : ℕ := 100

/-- Calculates the number of individuals to be drawn from a specific sample -/
def stratified_sample_size (ratio : ℕ) : ℕ :=
  (total_sample_size * ratio) / (sample_ratio.A + sample_ratio.B + sample_ratio.C)

theorem stratified_sampling_c :
  stratified_sample_size sample_ratio.C = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_c_l875_87573


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l875_87588

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (1 + m) 3 ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l875_87588


namespace NUMINAMATH_CALUDE_jane_test_probability_l875_87531

theorem jane_test_probability (pass_prob : ℚ) (h : pass_prob = 4/7) :
  1 - pass_prob = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_jane_test_probability_l875_87531


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l875_87510

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem greatest_x_given_lcm (x : ℕ) :
  lcm x 15 21 = 105 → x ≤ 105 ∧ ∃ y : ℕ, lcm y 15 21 = 105 ∧ y = 105 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l875_87510


namespace NUMINAMATH_CALUDE_WXYZ_perimeter_l875_87504

/-- Represents a rectangle with a perimeter --/
structure Rectangle where
  perimeter : ℕ

/-- Represents the large rectangle WXYZ --/
def WXYZ : Rectangle := sorry

/-- The four smaller rectangles that WXYZ is divided into --/
def smallRectangles : Fin 4 → Rectangle := sorry

/-- The sum of perimeters of diagonally opposite rectangles equals the perimeter of WXYZ --/
axiom perimeter_sum (i j : Fin 4) (h : i.val + j.val = 3) : 
  (smallRectangles i).perimeter + (smallRectangles j).perimeter = WXYZ.perimeter

/-- The perimeters of three of the smaller rectangles --/
axiom known_perimeters : 
  ∃ (i j k : Fin 4) (h : i ≠ j ∧ j ≠ k ∧ i ≠ k),
    (smallRectangles i).perimeter = 11 ∧
    (smallRectangles j).perimeter = 16 ∧
    (smallRectangles k).perimeter = 19

/-- The perimeter of the fourth rectangle is between 11 and 19 --/
axiom fourth_perimeter :
  ∃ (l : Fin 4), ∀ (i : Fin 4), 
    (smallRectangles i).perimeter ≠ 11 → 
    (smallRectangles i).perimeter ≠ 16 → 
    (smallRectangles i).perimeter ≠ 19 →
    11 < (smallRectangles l).perimeter ∧ (smallRectangles l).perimeter < 19

/-- The perimeter of WXYZ is 30 --/
theorem WXYZ_perimeter : WXYZ.perimeter = 30 := by sorry

end NUMINAMATH_CALUDE_WXYZ_perimeter_l875_87504


namespace NUMINAMATH_CALUDE_vehicle_speeds_l875_87548

/-- A structure representing a vehicle with its speed -/
structure Vehicle where
  speed : ℝ
  speed_pos : speed > 0

/-- The problem setup -/
def VehicleProblem (v₁ v₄ : ℝ) : Prop :=
  v₁ > 0 ∧ v₄ > 0 ∧ v₁ > v₄

/-- The theorem statement -/
theorem vehicle_speeds (v₁ v₄ : ℝ) (h : VehicleProblem v₁ v₄) :
  ∃ (v₂ v₃ : ℝ),
    v₂ = 3 * v₁ * v₄ / (2 * v₄ + v₁) ∧
    v₃ = 3 * v₁ * v₄ / (v₄ + 2 * v₁) ∧
    v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > v₄ :=
  sorry

end NUMINAMATH_CALUDE_vehicle_speeds_l875_87548
