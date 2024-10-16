import Mathlib

namespace NUMINAMATH_CALUDE_m_geq_n_l820_82066

theorem m_geq_n (a b : ℝ) : 
  let M := a^2 + 12*a - 4*b
  let N := 4*a - 20 - b^2
  M ≥ N := by
sorry

end NUMINAMATH_CALUDE_m_geq_n_l820_82066


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l820_82062

theorem ceiling_floor_difference : 
  ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l820_82062


namespace NUMINAMATH_CALUDE_consecutive_sums_not_prime_l820_82004

theorem consecutive_sums_not_prime (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ k < 5*n + 10 ∧ (5*n + 10) % k = 0) ∧
  (∃ k : ℕ, k > 1 ∧ k < 5*n^2 + 10 ∧ (5*n^2 + 10) % k = 0) := by
  sorry

#check consecutive_sums_not_prime

end NUMINAMATH_CALUDE_consecutive_sums_not_prime_l820_82004


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l820_82077

/-- For a parabola with equation y^2 = 4x, the distance from its focus to its directrix is 2 -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), y^2 = 4*x → ∃ (f d : ℝ × ℝ), 
    (f.1 = 1 ∧ f.2 = 0) ∧ -- focus coordinates
    (d.1 = -1 ∧ ∀ t, d.2 = t) ∧ -- directrix equation
    (f.1 - d.1 = 2) -- distance between focus and directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l820_82077


namespace NUMINAMATH_CALUDE_max_type_A_machines_l820_82028

/-- The cost of a type A machine in millions of yuan -/
def cost_A : ℕ := 7

/-- The cost of a type B machine in millions of yuan -/
def cost_B : ℕ := 5

/-- The total number of machines to be purchased -/
def total_machines : ℕ := 6

/-- The maximum budget in millions of yuan -/
def max_budget : ℕ := 34

/-- Condition: Cost of 3 type A machines and 2 type B machines is 31 million yuan -/
axiom condition1 : 3 * cost_A + 2 * cost_B = 31

/-- Condition: One type A machine costs 2 million yuan more than one type B machine -/
axiom condition2 : cost_A = cost_B + 2

/-- Theorem: The maximum number of type A machines that can be purchased within the budget is 2 -/
theorem max_type_A_machines : 
  ∀ m : ℕ, m ≤ total_machines → m * cost_A + (total_machines - m) * cost_B ≤ max_budget → m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_type_A_machines_l820_82028


namespace NUMINAMATH_CALUDE_initial_mat_weavers_l820_82006

theorem initial_mat_weavers : ℕ :=
  let initial_weavers : ℕ := sorry
  let initial_mats : ℕ := 4
  let initial_days : ℕ := 4
  let second_weavers : ℕ := 14
  let second_mats : ℕ := 49
  let second_days : ℕ := 14

  have h1 : initial_weavers * initial_days * second_mats = second_weavers * second_days * initial_mats := by sorry

  have h2 : initial_weavers = 4 := by sorry

  4


end NUMINAMATH_CALUDE_initial_mat_weavers_l820_82006


namespace NUMINAMATH_CALUDE_inequality_proof_l820_82080

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l820_82080


namespace NUMINAMATH_CALUDE_extremum_implies_zero_derivative_l820_82033

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a point to be an extremum
def is_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, |y - x| < 1 → f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem extremum_implies_zero_derivative (f : ℝ → ℝ) (x : ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : is_extremum f x) : 
  deriv f x = 0 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_zero_derivative_l820_82033


namespace NUMINAMATH_CALUDE_BC_equals_2AB_l820_82055

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (4, 3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

theorem BC_equals_2AB : vector_BC = (2 * vector_AB.1, 2 * vector_AB.2) := by
  sorry

end NUMINAMATH_CALUDE_BC_equals_2AB_l820_82055


namespace NUMINAMATH_CALUDE_trig_simplification_l820_82098

theorem trig_simplification (x : ℝ) : 
  (1 + Real.sin (3 * x) - Real.cos (3 * x)) / (1 + Real.sin (3 * x) + Real.cos (3 * x)) = 
  (1 + 3 * (Real.sin x + Real.cos x) - 4 * (Real.sin x ^ 3 + Real.cos x ^ 3)) / 
  (1 + 3 * (Real.sin x - Real.cos x) - 4 * (Real.sin x ^ 3 - Real.cos x ^ 3)) := by
sorry

end NUMINAMATH_CALUDE_trig_simplification_l820_82098


namespace NUMINAMATH_CALUDE_union_of_sets_l820_82057

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l820_82057


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l820_82030

theorem restaurant_glasses_count :
  -- Define the number of glasses in each box type
  let small_box_glasses : ℕ := 12
  let large_box_glasses : ℕ := 16
  -- Define the difference in number of boxes
  let box_difference : ℕ := 16
  -- Define the average number of glasses per box
  let average_glasses : ℚ := 15
  -- Define variables for the number of each type of box
  ∀ (small_boxes large_boxes : ℕ),
  -- Condition: There are 16 more large boxes than small boxes
  large_boxes = small_boxes + box_difference →
  -- Condition: The average number of glasses per box is 15
  (small_box_glasses * small_boxes + large_box_glasses * large_boxes : ℚ) / 
    (small_boxes + large_boxes : ℚ) = average_glasses →
  -- Conclusion: The total number of glasses is 480
  small_box_glasses * small_boxes + large_box_glasses * large_boxes = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l820_82030


namespace NUMINAMATH_CALUDE_unique_m_for_power_function_l820_82011

/-- A function f is a power function if it has the form f(x) = ax^b for some constants a and b, where a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^b

/-- A function f is increasing on (0, +∞) if for all x₁, x₂ > 0, x₁ < x₂ implies f(x₁) < f(x₂) -/
def is_increasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂

/-- The main theorem -/
theorem unique_m_for_power_function :
  ∃! m : ℝ, 
    is_power_function (fun x ↦ (m^2 - m - 1) * x^m) ∧
    is_increasing_on_positive_reals (fun x ↦ (m^2 - m - 1) * x^m) ∧
    m = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_for_power_function_l820_82011


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l820_82083

def manuscript_cost (total_pages : ℕ) (once_revised : ℕ) (twice_revised : ℕ) (twice_revised_set : ℕ) 
                    (thrice_revised : ℕ) (thrice_revised_sets : ℕ) : ℕ :=
  let initial_cost := total_pages * 5
  let once_revised_cost := once_revised * 3
  let twice_revised_cost := (twice_revised - twice_revised_set) * 3 * 2 + twice_revised_set * 3 * 2 + 10
  let thrice_revised_cost := (thrice_revised - thrice_revised_sets * 10) * 3 * 3 + thrice_revised_sets * 15
  initial_cost + once_revised_cost + twice_revised_cost + thrice_revised_cost

theorem manuscript_typing_cost :
  manuscript_cost 200 50 70 10 40 2 = 1730 :=
by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l820_82083


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l820_82021

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l820_82021


namespace NUMINAMATH_CALUDE_exist_x_y_sequences_l820_82076

def sequence_a : ℕ → ℚ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * sequence_a (n + 1) - sequence_a n

theorem exist_x_y_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n, 
    sequence_a n = (y n ^ 2 + 7 : ℚ) / (x n - y n : ℚ) ∧
    x n > y n ∧ 
    x n > 0 ∧ 
    y n > 0 :=
by sorry

end NUMINAMATH_CALUDE_exist_x_y_sequences_l820_82076


namespace NUMINAMATH_CALUDE_square_root_of_two_l820_82067

theorem square_root_of_two :
  ∀ x : ℝ, x^2 = 2 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_two_l820_82067


namespace NUMINAMATH_CALUDE_reciprocal_problem_l820_82045

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the smallest composite number
def smallest_composite : ℕ := 4

theorem reciprocal_problem :
  (reciprocal 0.8 = 5/4) ∧
  (reciprocal (1/4) = smallest_composite) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l820_82045


namespace NUMINAMATH_CALUDE_line_properties_l820_82088

/-- Represents a line in the form x = my + 1 --/
structure Line where
  m : ℝ

/-- The point (1, 0) is on the line --/
def point_on_line (l : Line) : Prop :=
  1 = l.m * 0 + 1

/-- The area of the triangle formed by the line and the axes when m = 2 --/
def triangle_area (l : Line) : Prop :=
  l.m = 2 → (1 / 2 : ℝ) * 1 * (1 / 2) = (1 / 4 : ℝ)

/-- Main theorem stating that both properties hold for any line of the form x = my + 1 --/
theorem line_properties (l : Line) : point_on_line l ∧ triangle_area l := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l820_82088


namespace NUMINAMATH_CALUDE_complement_union_equals_ge_one_l820_82075

open Set

def M : Set ℝ := {x | (x + 3) / (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

theorem complement_union_equals_ge_one : 
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_ge_one_l820_82075


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l820_82027

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_entrance_cost :
  total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l820_82027


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_inverses_squared_l820_82049

theorem quadratic_roots_sum_of_inverses_squared (p q : ℝ) : 
  (3 * p^2 - 5 * p + 2 = 0) → 
  (3 * q^2 - 5 * q + 2 = 0) → 
  (1 / p^2 + 1 / q^2 = 13 / 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_inverses_squared_l820_82049


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l820_82084

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a2 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l820_82084


namespace NUMINAMATH_CALUDE_card_collection_difference_l820_82038

theorem card_collection_difference (total : ℕ) (baseball : ℕ) (football : ℕ) 
  (h1 : total = 125)
  (h2 : baseball = 95)
  (h3 : total = baseball + football)
  (h4 : ∃ k : ℕ, baseball = 3 * football + k) :
  baseball - 3 * football = 5 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_difference_l820_82038


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l820_82065

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ c d : ℕ+, c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l820_82065


namespace NUMINAMATH_CALUDE_wire_problem_l820_82096

theorem wire_problem (total_length : ℝ) (num_parts : ℕ) (used_parts : ℕ) : 
  total_length = 50 ∧ 
  num_parts = 5 ∧ 
  used_parts = 3 → 
  total_length - (total_length / num_parts) * used_parts = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_problem_l820_82096


namespace NUMINAMATH_CALUDE_cos_sum_max_min_points_l820_82063

/-- Given a function f(x) = cos(2x) + sin(x), prove that the cosine of the sum of
    the abscissas of its maximum and minimum points equals 1/4. -/
theorem cos_sum_max_min_points (f : ℝ → ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, f x = Real.cos (2 * x) + Real.sin x) →
  (∀ x, f x ≤ f x₁) →
  (∀ x, f x ≥ f x₂) →
  Real.cos (x₁ + x₂) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_max_min_points_l820_82063


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l820_82090

theorem complex_arithmetic_equality : 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001 = 76802 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l820_82090


namespace NUMINAMATH_CALUDE_father_daughter_ages_l820_82002

/-- Represents the ages of a father and daughter at present and in the future. -/
structure FamilyAges where
  daughter_now : ℕ
  father_now : ℕ
  daughter_future : ℕ
  father_future : ℕ

/-- The conditions given in the problem. -/
def age_conditions (ages : FamilyAges) : Prop :=
  ages.father_now = 5 * ages.daughter_now ∧
  ages.daughter_future = ages.daughter_now + 30 ∧
  ages.father_future = ages.father_now + 30 ∧
  ages.father_future = 3 * ages.daughter_future

/-- The theorem stating the solution to the problem. -/
theorem father_daughter_ages :
  ∃ (ages : FamilyAges), age_conditions ages ∧ ages.daughter_now = 30 ∧ ages.father_now = 150 := by
  sorry

end NUMINAMATH_CALUDE_father_daughter_ages_l820_82002


namespace NUMINAMATH_CALUDE_locus_of_centers_l820_82089

/-- Circle C1 with equation x^2 + y^2 = 1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C2 with equation (x - 2)^2 + y^2 = 25 -/
def C2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25

/-- A circle is externally tangent to C1 if the distance between their centers equals the sum of their radii -/
def externally_tangent_C1 (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C2 if the distance between their centers equals the difference of their radii -/
def internally_tangent_C2 (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (5 - r)^2

/-- The main theorem: the locus of centers (a,b) of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C1 a b r ∧ internally_tangent_C2 a b r) → 
  3 * a^2 + b^2 + 44 * a + 121 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l820_82089


namespace NUMINAMATH_CALUDE_log_equation_holds_l820_82097

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l820_82097


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l820_82019

-- Define 100!
def factorial_100 : ℕ := Nat.factorial 100

-- Define the function to get the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_of_100_factorial :
  last_two_nonzero_digits (factorial_100 / (10^24)) = 76 := by
  sorry

#eval last_two_nonzero_digits (factorial_100 / (10^24))

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l820_82019


namespace NUMINAMATH_CALUDE_polynomial_inequality_implies_upper_bound_l820_82047

theorem polynomial_inequality_implies_upper_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, x^3 + x^2 + a < 0) → a < -12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_implies_upper_bound_l820_82047


namespace NUMINAMATH_CALUDE_function_composition_identity_l820_82029

/-- Given a function f(x) = (2ax - b) / (3cx + d) where b ≠ 0, d ≠ 0, abcd ≠ 0,
    and f(f(x)) = x for all x in the domain of f, there exist real numbers b and c
    such that 3a - 2d = -4.5c - 4b -/
theorem function_composition_identity (a b c d : ℝ) : 
  b ≠ 0 → d ≠ 0 → a * b * c * d ≠ 0 → 
  (∀ x, (2 * a * ((2 * a * x - b) / (3 * c * x + d)) - b) / 
        (3 * c * ((2 * a * x - b) / (3 * c * x + d)) + d) = x) →
  ∃ (b c : ℝ), 3 * a - 2 * d = -4.5 * c - 4 * b :=
by sorry

end NUMINAMATH_CALUDE_function_composition_identity_l820_82029


namespace NUMINAMATH_CALUDE_hallway_tiles_l820_82095

/-- Calculates the total number of tiles used in a rectangular hallway with specific tiling patterns. -/
def total_tiles (length width : ℕ) : ℕ :=
  let outer_border := 2 * (length - 2) + 2 * (width - 2) + 4
  let second_border := 2 * ((length - 4) / 2) + 2 * ((width - 4) / 2)
  let inner_area := ((length - 6) * (width - 6)) / 9
  outer_border + second_border + inner_area

/-- Theorem stating that the total number of tiles used in a 20x30 foot rectangular hallway
    with specific tiling patterns is 175. -/
theorem hallway_tiles : total_tiles 30 20 = 175 := by
  sorry

end NUMINAMATH_CALUDE_hallway_tiles_l820_82095


namespace NUMINAMATH_CALUDE_bracelet_large_beads_l820_82024

/-- Proves the number of large beads per bracelet given the problem conditions -/
theorem bracelet_large_beads (total_beads : ℕ) (num_bracelets : ℕ) : 
  total_beads = 528 →
  num_bracelets = 11 →
  ∃ (large_beads_per_bracelet : ℕ),
    large_beads_per_bracelet * num_bracelets = total_beads / 2 ∧
    large_beads_per_bracelet = 24 := by
  sorry

#check bracelet_large_beads

end NUMINAMATH_CALUDE_bracelet_large_beads_l820_82024


namespace NUMINAMATH_CALUDE_complex_simplification_l820_82092

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  3 * (4 - 2*i) + 2*i*(3 + 2*i) - (1 + i)*(2 - i) = 5 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l820_82092


namespace NUMINAMATH_CALUDE_houses_in_block_l820_82073

/-- Given a block where each house receives 32 pieces of junk mail and
    the entire block receives 640 pieces of junk mail, prove that
    there are 20 houses in the block. -/
theorem houses_in_block (mail_per_house : ℕ) (mail_per_block : ℕ)
    (h1 : mail_per_house = 32)
    (h2 : mail_per_block = 640) :
    mail_per_block / mail_per_house = 20 := by
  sorry

end NUMINAMATH_CALUDE_houses_in_block_l820_82073


namespace NUMINAMATH_CALUDE_washing_machines_removed_count_l820_82018

/-- Represents the number of washing machines removed from a shipping container --/
def washing_machines_removed (crates boxes_per_crate machines_per_box machines_removed_per_box : ℕ) : ℕ :=
  crates * boxes_per_crate * machines_removed_per_box

/-- Theorem stating the number of washing machines removed from the shipping container --/
theorem washing_machines_removed_count : 
  washing_machines_removed 10 6 4 1 = 60 := by
  sorry

#eval washing_machines_removed 10 6 4 1

end NUMINAMATH_CALUDE_washing_machines_removed_count_l820_82018


namespace NUMINAMATH_CALUDE_expand_and_simplify_l820_82094

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5*x - 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l820_82094


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l820_82099

def complex_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant :
  let z : ℂ := 2 * Complex.I / (2 - Complex.I)
  second_quadrant (complex_point z) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l820_82099


namespace NUMINAMATH_CALUDE_parabola_standard_form_l820_82007

/-- A parabola with vertex at the origin and axis of symmetry x = -2 has the standard form equation y² = 8x -/
theorem parabola_standard_form (p : ℝ) (h : p / 2 = 2) :
  ∀ x y : ℝ, y^2 = 8 * x ↔ y^2 = 2 * p * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_form_l820_82007


namespace NUMINAMATH_CALUDE_reaction_stoichiometry_l820_82068

-- Define the chemical species
def CaO : Type := Unit
def H2O : Type := Unit
def Ca_OH_2 : Type := Unit

-- Define the reaction
def reaction (cao : CaO) (h2o : H2O) : Ca_OH_2 := sorry

-- Define the number of moles
def moles : Type → ℝ := sorry

-- Theorem statement
theorem reaction_stoichiometry :
  ∀ (cao : CaO) (h2o : H2O),
    moles CaO = 1 →
    moles Ca_OH_2 = 1 →
    moles H2O = 1 :=
by sorry

end NUMINAMATH_CALUDE_reaction_stoichiometry_l820_82068


namespace NUMINAMATH_CALUDE_additional_workers_for_earlier_completion_l820_82012

/-- Calculates the number of additional workers needed to complete a task earlier -/
def additional_workers (original_days : ℕ) (actual_days : ℕ) (original_workers : ℕ) : ℕ :=
  ⌊(original_workers * original_days / actual_days - original_workers : ℚ)⌋.toNat

/-- Proves that 6 additional workers are needed to complete the task 3 days earlier -/
theorem additional_workers_for_earlier_completion :
  additional_workers 10 7 15 = 6 := by
  sorry

#eval additional_workers 10 7 15

end NUMINAMATH_CALUDE_additional_workers_for_earlier_completion_l820_82012


namespace NUMINAMATH_CALUDE_parallel_line_through_point_A_l820_82059

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the parallel line passing through point A
def parallel_line (x y : ℝ) : Prop := 2 * x - y - 3 = 0

theorem parallel_line_through_point_A :
  (parallel_line point_A.1 point_A.2) ∧
  (∀ (x y : ℝ), parallel_line x y → given_line x y → x = y) ∧
  (∃ (m b : ℝ), ∀ (x y : ℝ), parallel_line x y ↔ y = m * x + b) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_A_l820_82059


namespace NUMINAMATH_CALUDE_completing_square_proof_l820_82001

theorem completing_square_proof (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_proof_l820_82001


namespace NUMINAMATH_CALUDE_prob_different_colors_l820_82093

/-- The probability of drawing two balls of different colors from a box containing 3 red balls and 2 yellow balls. -/
theorem prob_different_colors (total : ℕ) (red : ℕ) (yellow : ℕ) : 
  total = 5 → red = 3 → yellow = 2 → 
  (red.choose 1 * yellow.choose 1 : ℚ) / total.choose 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_l820_82093


namespace NUMINAMATH_CALUDE_equation_solutions_l820_82058

def solution_set : Set (ℤ × ℤ) :=
  {(0, -4), (0, 8), (-2, 0), (-4, 8), (-6, 6), (0, 0), (-10, 4)}

def satisfies_equation (x y : ℤ) : Prop :=
  x + y ≠ 0 ∧ (x - y)^2 / (x + y) = x - y + 6

theorem equation_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l820_82058


namespace NUMINAMATH_CALUDE_max_crates_on_trailer_l820_82044

/-- The maximum number of crates a trailer can carry given weight constraints -/
theorem max_crates_on_trailer (min_crate_weight max_total_weight : ℕ) 
  (h1 : min_crate_weight ≥ 120)
  (h2 : max_total_weight = 720) :
  (max_total_weight / min_crate_weight : ℕ) = 6 := by
  sorry

#check max_crates_on_trailer

end NUMINAMATH_CALUDE_max_crates_on_trailer_l820_82044


namespace NUMINAMATH_CALUDE_product_digits_concatenated_digits_l820_82014

-- Part (a)
theorem product_digits (A B : ℕ) (hA : 10^5 < A ∧ A < 2*10^5) (hB : 10^9 < B ∧ B < 2*10^9) :
  10^14 < A * B ∧ A * B < 10^15 :=
sorry

-- Part (b)
theorem concatenated_digits :
  ∃ (x y : ℕ), (10^(x-1) < 2^2016 ∧ 2^2016 < 10^x) ∧
               (10^(y-1) < 5^2016 ∧ 5^2016 < 10^y) ∧
               x + y = 2017 :=
sorry

end NUMINAMATH_CALUDE_product_digits_concatenated_digits_l820_82014


namespace NUMINAMATH_CALUDE_tan_theta_equals_sqrt_three_over_five_l820_82003

theorem tan_theta_equals_sqrt_three_over_five (θ : Real) : 
  2 * Real.sin (θ + π/3) = 3 * Real.sin (π/3 - θ) → 
  Real.tan θ = Real.sqrt 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_equals_sqrt_three_over_five_l820_82003


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l820_82082

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 179)
  (h2 : a*b + b*c + a*c = 131) :
  a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l820_82082


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l820_82022

/-- The equation of the trajectory of the midpoint of a line segment -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) : 
  y₁ = 2 * x₁^2 + 1 →  -- P is on the curve y = 2x^2 + 1
  x = (x₁ + 0) / 2 →   -- x-coordinate of midpoint
  y = (y₁ + (-1)) / 2  -- y-coordinate of midpoint
  → y = 4 * x^2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l820_82022


namespace NUMINAMATH_CALUDE_expected_participants_2005_l820_82048

/-- The number of participants after n years, given an initial population and growth rate -/
def participants (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

/-- Theorem: Given 500 initial participants in 2000 and 20% annual growth,
    the expected number of participants in 2005 is 1244 -/
theorem expected_participants_2005 :
  ⌊participants 500 (1/5) 5⌋ = 1244 := by
  sorry

end NUMINAMATH_CALUDE_expected_participants_2005_l820_82048


namespace NUMINAMATH_CALUDE_custard_pie_pieces_l820_82051

/-- Proves that the number of pieces a custard pie is cut into is 6, given the conditions of the bakery problem. -/
theorem custard_pie_pieces : ℕ :=
  let pumpkin_pieces : ℕ := 8
  let pumpkin_price : ℕ := 5
  let custard_price : ℕ := 6
  let pumpkin_pies_sold : ℕ := 4
  let custard_pies_sold : ℕ := 5
  let total_revenue : ℕ := 340

  have h1 : pumpkin_pieces * pumpkin_price * pumpkin_pies_sold + custard_price * custard_pies_sold * custard_pie_pieces = total_revenue := by sorry

  custard_pie_pieces
where
  custard_pie_pieces : ℕ := 6

#check custard_pie_pieces

end NUMINAMATH_CALUDE_custard_pie_pieces_l820_82051


namespace NUMINAMATH_CALUDE_seating_arrangements_5_total_arrangements_l820_82052

/-- Defines the number of seating arrangements for n people -/
def seating_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => seating_arrangements (n + 1) + seating_arrangements n

/-- Theorem stating that the number of seating arrangements for 5 people is 8 -/
theorem seating_arrangements_5 : seating_arrangements 5 = 8 := by sorry

/-- Theorem stating that the total number of arrangements for two independent groups of 5 is 64 -/
theorem total_arrangements : seating_arrangements 5 * seating_arrangements 5 = 64 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_5_total_arrangements_l820_82052


namespace NUMINAMATH_CALUDE_mistaken_calculation_l820_82023

theorem mistaken_calculation (x : ℝ) (h : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l820_82023


namespace NUMINAMATH_CALUDE_inequality_proof_l820_82026

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l820_82026


namespace NUMINAMATH_CALUDE_locus_of_point_P_l820_82056

/-- The locus of point P where a moving circle M with diameter PF₁ is tangent internally 
    to a fixed circle C -/
theorem locus_of_point_P (n m : ℝ) (h_positive : 0 < n ∧ n < m) :
  ∃ (locus : ℝ × ℝ → Prop),
    (∀ (P : ℝ × ℝ), locus P ↔ 
      (P.1^2 / m^2 + P.2^2 / (m^2 - n^2) = 1)) ∧
    (∀ (P : ℝ × ℝ), locus P → 
      ∃ (M : ℝ × ℝ),
        -- M is the center of the moving circle
        M = ((P.1 - (-n)) / 2, P.2 / 2) ∧
        -- M is internally tangent to the fixed circle C
        ((M.1^2 + M.2^2)^(1/2) + ((M.1 - (-n))^2 + M.2^2)^(1/2) = m) ∧
        -- PF₁ is a diameter of the moving circle
        (P.1 - (-n))^2 + P.2^2 = (2 * ((M.1 - (-n))^2 + M.2^2)^(1/2))^2) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l820_82056


namespace NUMINAMATH_CALUDE_magnitude_2a_plus_b_l820_82070

variable (a b : ℝ × ℝ)

theorem magnitude_2a_plus_b (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) 
  (h3 : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
  ‖2 • a + b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_2a_plus_b_l820_82070


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_of_special_triangle_l820_82054

/-- Given a triangle ABC where side b = 2√3 and angles A, B, C form an arithmetic sequence,
    the radius of the circumscribed circle is 2. -/
theorem circumscribed_circle_radius_of_special_triangle (A B C : Real) (a b c : Real) :
  b = 2 * Real.sqrt 3 →
  ∃ (d : Real), B = (A + C) / 2 ∧ A + d = B ∧ B + d = C →
  A + B + C = Real.pi →
  2 * Real.sin B = b / 2 →
  2 = 2 * Real.sin B / b * 2 * Real.sqrt 3 := by
  sorry

#check circumscribed_circle_radius_of_special_triangle

end NUMINAMATH_CALUDE_circumscribed_circle_radius_of_special_triangle_l820_82054


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l820_82031

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 8)

-- Define the altitude equation
def altitude_eq (x y : ℝ) : Prop := 6 * x - y - 24 = 0

-- Define the median equation
def median_eq (x y : ℝ) : Prop := y = -15/2 * x + 30

-- Theorem statement
theorem triangle_altitude_and_median :
  (∀ x y : ℝ, altitude_eq x y ↔ 
    (x - A.1) * (B.2 - C.2) = (y - A.2) * (B.1 - C.1) ∧ 
    (x - A.1) * (B.1 - C.1) + (y - A.2) * (B.2 - C.2) = 0) ∧
  (∀ x y : ℝ, median_eq x y ↔ 
    2 * (y - A.2) * (B.1 - C.1) = (x - A.1) * (B.2 + C.2 - 2 * A.2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l820_82031


namespace NUMINAMATH_CALUDE_circle_equation_l820_82081

/-- 
Given a circle with radius 2, center on the positive x-axis, and tangent to the y-axis,
prove that its equation is x^2 + y^2 - 4x = 0.
-/
theorem circle_equation (x y : ℝ) : 
  ∃ (h : ℝ), h > 0 ∧ 
  (∀ (a b : ℝ), (a - h)^2 + b^2 = 4 → a ≥ 0) ∧
  (∃ (c : ℝ), c^2 = 4 ∧ (h - 0)^2 + c^2 = 4) →
  x^2 + y^2 - 4*x = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l820_82081


namespace NUMINAMATH_CALUDE_x_value_proof_l820_82037

theorem x_value_proof (x : ℝ) (h : 3/4 - 1/2 = 4/x) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l820_82037


namespace NUMINAMATH_CALUDE_unique_plane_through_three_points_perpendicular_line_implies_parallel_planes_parallel_to_plane_not_implies_parallel_lines_perpendicular_to_plane_implies_parallel_lines_l820_82046

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (collinear : Point → Point → Point → Prop)
variable (on_plane : Point → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Proposition A
theorem unique_plane_through_three_points 
  (p q r : Point) (h : ¬ collinear p q r) :
  ∃! π : Plane, on_plane p π ∧ on_plane q π ∧ on_plane r π :=
sorry

-- Proposition B
theorem perpendicular_line_implies_parallel_planes 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular m β) :
  parallel_planes α β :=
sorry

-- Proposition C (negation)
theorem parallel_to_plane_not_implies_parallel_lines 
  (m n : Line) (α : Plane) :
  parallel_line_plane m α ∧ parallel_line_plane n α → 
  ¬ (parallel_lines m n → True) :=
sorry

-- Proposition D
theorem perpendicular_to_plane_implies_parallel_lines 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_three_points_perpendicular_line_implies_parallel_planes_parallel_to_plane_not_implies_parallel_lines_perpendicular_to_plane_implies_parallel_lines_l820_82046


namespace NUMINAMATH_CALUDE_jug_fills_ten_large_glasses_l820_82091

/-- Represents the volume of a glass -/
structure Glass :=
  (volume : ℚ)

/-- Represents a jug with a certain capacity -/
structure Jug :=
  (capacity : ℚ)

/-- Represents the problem setup -/
structure JugProblem :=
  (small_glass : Glass)
  (large_glass : Glass)
  (jug : Jug)
  (condition1 : 9 * small_glass.volume + 4 * large_glass.volume = jug.capacity)
  (condition2 : 6 * small_glass.volume + 6 * large_glass.volume = jug.capacity)

theorem jug_fills_ten_large_glasses (problem : JugProblem) :
  problem.jug.capacity = 10 * problem.large_glass.volume :=
sorry

end NUMINAMATH_CALUDE_jug_fills_ten_large_glasses_l820_82091


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l820_82071

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/49
  let a₃ : ℚ := 64/343
  (a₂ / a₁ = 4/7) ∧ (a₃ / a₂ = 4/7) → 
  ∃ (r : ℚ), ∀ (n : ℕ), n ≥ 1 → 
    (4/7) * (4/7)^(n-1) = (4/7) * r^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l820_82071


namespace NUMINAMATH_CALUDE_pqr_plus_xyz_eq_zero_l820_82087

theorem pqr_plus_xyz_eq_zero 
  (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := by sorry

end NUMINAMATH_CALUDE_pqr_plus_xyz_eq_zero_l820_82087


namespace NUMINAMATH_CALUDE_divisors_of_20_factorial_l820_82015

theorem divisors_of_20_factorial : (Nat.divisors (Nat.factorial 20)).card = 41040 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_20_factorial_l820_82015


namespace NUMINAMATH_CALUDE_ramanujan_identity_l820_82060

theorem ramanujan_identity : ∃ (p q r p₁ q₁ r₁ : ℕ), 
  p ≠ q ∧ p ≠ r ∧ p ≠ p₁ ∧ p ≠ q₁ ∧ p ≠ r₁ ∧
  q ≠ r ∧ q ≠ p₁ ∧ q ≠ q₁ ∧ q ≠ r₁ ∧
  r ≠ p₁ ∧ r ≠ q₁ ∧ r ≠ r₁ ∧
  p₁ ≠ q₁ ∧ p₁ ≠ r₁ ∧
  q₁ ≠ r₁ ∧
  p^2 + q^2 + r^2 = p₁^2 + q₁^2 + r₁^2 ∧
  p^4 + q^4 + r^4 = p₁^4 + q₁^4 + r₁^4 := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_identity_l820_82060


namespace NUMINAMATH_CALUDE_train_passes_jogger_l820_82005

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 260 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 38 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l820_82005


namespace NUMINAMATH_CALUDE_consecutive_product_and_fourth_power_properties_l820_82025

theorem consecutive_product_and_fourth_power_properties (c d m n : ℕ) : 
  (c * (c + 1) ≠ d * (d + 2)) ∧ 
  (m^4 + (m + 1)^4 ≠ n^2 + (n + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_and_fourth_power_properties_l820_82025


namespace NUMINAMATH_CALUDE_square_nonnegative_l820_82034

theorem square_nonnegative (x : ℚ) : x^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_nonnegative_l820_82034


namespace NUMINAMATH_CALUDE_food_price_consumption_reduction_l820_82061

theorem food_price_consumption_reduction (initial_price : ℝ) (h : initial_price > 0) :
  let price_increase_factor := 1.5
  let consumption_reduction_factor := 2/3
  initial_price * price_increase_factor * consumption_reduction_factor = initial_price :=
by sorry

end NUMINAMATH_CALUDE_food_price_consumption_reduction_l820_82061


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l820_82039

/-- The line equation kx - y - 2k + 3 = 0 is tangent to the circle x^2 + (y + 1)^2 = 4 if and only if k = 3/4 -/
theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) ↔ k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l820_82039


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l820_82000

theorem polynomial_equality_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℝ, 512 * x^3 + 125 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 6410 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l820_82000


namespace NUMINAMATH_CALUDE_square_difference_sum_l820_82072

theorem square_difference_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l820_82072


namespace NUMINAMATH_CALUDE_largest_element_complement_A_intersect_B_l820_82042

def I : Set ℤ := {x | 1 ≤ x ∧ x ≤ 100}
def A : Set ℤ := {m ∈ I | ∃ k : ℤ, m = 2 * k + 1}
def B : Set ℤ := {n ∈ I | ∃ k : ℤ, n = 3 * k}

theorem largest_element_complement_A_intersect_B :
  ∃ x : ℤ, x ∈ (I \ A) ∩ B ∧ x = 96 ∧ ∀ y ∈ (I \ A) ∩ B, y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_element_complement_A_intersect_B_l820_82042


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l820_82017

/-- The volume of a rectangular prism with length:width:height ratio of 4:3:1 and height √2 cm is 24√2 cm³ -/
theorem rectangular_prism_volume (height : ℝ) (length width : ℝ) : 
  height = Real.sqrt 2 →
  length = 4 * height →
  width = 3 * height →
  length * width * height = 24 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l820_82017


namespace NUMINAMATH_CALUDE_max_prime_factors_l820_82086

theorem max_prime_factors (x y : ℕ+) 
  (h_gcd : (Nat.gcd x y).factors.length = 5)
  (h_lcm : (Nat.lcm x y).factors.length = 20)
  (h_fewer : (x : ℕ).factors.length < (y : ℕ).factors.length) :
  (x : ℕ).factors.length ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_prime_factors_l820_82086


namespace NUMINAMATH_CALUDE_same_color_probability_l820_82010

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def selected_plates : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates selected_plates : ℚ) / (Nat.choose total_plates selected_plates) = 4 / 33 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l820_82010


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l820_82078

def complex_is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  complex_is_pure_imaginary ((a + 6 * Complex.I) / (3 - Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l820_82078


namespace NUMINAMATH_CALUDE_no_positive_abc_equality_l820_82013

theorem no_positive_abc_equality : ¬∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = a * b + a * c + b * c ∧ a * b + a * c + b * c = a * b * c := by
  sorry

end NUMINAMATH_CALUDE_no_positive_abc_equality_l820_82013


namespace NUMINAMATH_CALUDE_four_students_three_teams_l820_82040

/-- The number of ways students can sign up for sports teams -/
def signup_ways (num_students : ℕ) (num_teams : ℕ) : ℕ :=
  num_teams ^ num_students

/-- Theorem: 4 students signing up for 3 teams results in 3^4 ways -/
theorem four_students_three_teams :
  signup_ways 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_teams_l820_82040


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l820_82050

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  ((n - 2) * 180 : ℝ) / n = 160 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l820_82050


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l820_82020

/-- Proves that for a journey of given distance and original time,
    the average speed required to complete the same journey in a 
    multiple of the original time is as calculated. -/
theorem journey_speed_calculation 
  (distance : ℝ) 
  (original_time : ℝ) 
  (time_multiplier : ℝ) 
  (h1 : distance = 378) 
  (h2 : original_time = 6) 
  (h3 : time_multiplier = 3/2) :
  distance / (original_time * time_multiplier) = 42 :=
by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_journey_speed_calculation_l820_82020


namespace NUMINAMATH_CALUDE_billy_sleep_problem_l820_82032

theorem billy_sleep_problem (x : ℝ) : 
  x + (x + 2) + (x + 2) / 2 + 3 * ((x + 2) / 2) = 30 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_billy_sleep_problem_l820_82032


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l820_82053

theorem partial_fraction_decomposition :
  ∀ (x : ℝ), x ≠ 10 → x ≠ -2 →
  (6 * x - 4) / (x^2 - 8 * x - 20) = 
  (14 / 3) / (x - 10) + (4 / 3) / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l820_82053


namespace NUMINAMATH_CALUDE_jackson_score_l820_82064

/-- Given a basketball team's scoring information, calculate Jackson's score. -/
theorem jackson_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) (h1 : total_score = 72) (h2 : other_players = 7) (h3 : avg_score = 6) : total_score - other_players * avg_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_jackson_score_l820_82064


namespace NUMINAMATH_CALUDE_division_remainder_problem_l820_82016

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1345 → 
  L = 1596 → 
  L / S = 6 → 
  L % S = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l820_82016


namespace NUMINAMATH_CALUDE_band_arrangement_minimum_band_size_l820_82085

theorem band_arrangement (n : ℕ) : n > 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 → n ≥ 168 := by
  sorry

theorem minimum_band_size : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 ∧ n = 168 := by
  sorry

end NUMINAMATH_CALUDE_band_arrangement_minimum_band_size_l820_82085


namespace NUMINAMATH_CALUDE_expression_evaluation_l820_82069

theorem expression_evaluation (x : ℚ) (h : x = 1/2) : 
  (1 + x) * (1 - x) + x * (x + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l820_82069


namespace NUMINAMATH_CALUDE_decimal_23_to_binary_binary_to_decimal_23_l820_82036

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_23_to_binary :
  toBinary 23 = [true, true, true, false, true] :=
sorry

theorem binary_to_decimal_23 :
  fromBinary [true, true, true, false, true] = 23 :=
sorry

end NUMINAMATH_CALUDE_decimal_23_to_binary_binary_to_decimal_23_l820_82036


namespace NUMINAMATH_CALUDE_circular_film_radius_l820_82008

/-- The radius of a circular film formed by pouring a cylindrical container of liquid onto water -/
theorem circular_film_radius 
  (h : ℝ) -- height of the cylindrical container
  (d : ℝ) -- diameter of the cylindrical container
  (t : ℝ) -- thickness of the resulting circular film
  (h_pos : h > 0)
  (d_pos : d > 0)
  (t_pos : t > 0)
  (h_val : h = 10)
  (d_val : d = 5)
  (t_val : t = 0.2) :
  ∃ (r : ℝ), r^2 = 312.5 ∧ π * (d/2)^2 * h = π * r^2 * t :=
by sorry

end NUMINAMATH_CALUDE_circular_film_radius_l820_82008


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l820_82009

/-- The range of 'a' for which the ellipse x^2 + 4(y - a)^2 = 4 and the parabola x^2 = 2y intersect -/
theorem ellipse_parabola_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l820_82009


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l820_82035

/-- Given a geometric sequence {a_n} where a_1 and a_10 are the roots of 2x^2 + 5x + 1 = 0,
    prove that a_4 * a_7 = 1/2 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (2 * (a 1)^2 + 5 * (a 1) + 1 = 0) →       -- a_1 is a root
  (2 * (a 10)^2 + 5 * (a 10) + 1 = 0) →     -- a_10 is a root
  a 4 * a 7 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l820_82035


namespace NUMINAMATH_CALUDE_symmetric_f_max_value_l820_82079

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f(x) about x = -2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b (-2 - x) = f a b (-2 + x)

/-- The theorem stating that if f(x) is symmetric about x = -2, its maximum value is 16 -/
theorem symmetric_f_max_value (a b : ℝ) (h : is_symmetric a b) :
  ∃ x, f a b x = 16 ∧ ∀ y, f a b y ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_symmetric_f_max_value_l820_82079


namespace NUMINAMATH_CALUDE_max_value_constraint_l820_82041

theorem max_value_constraint (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 3 ∧ ∀ (a b : ℝ), 3 * a^2 + 4 * b^2 = 12 → 3 * a + 2 * b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l820_82041


namespace NUMINAMATH_CALUDE_star_four_three_l820_82043

def star (a b : ℕ) : ℕ := 3 * a^2 + 5 * b

theorem star_four_three : star 4 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l820_82043


namespace NUMINAMATH_CALUDE_problem_statement_l820_82074

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12)
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) :
  a + b^2 + c^3 = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l820_82074
