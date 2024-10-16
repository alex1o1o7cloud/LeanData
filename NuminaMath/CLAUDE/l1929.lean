import Mathlib

namespace NUMINAMATH_CALUDE_double_quarter_four_percent_l1929_192912

theorem double_quarter_four_percent : 
  (4 / 100 / 4 * 2 : ℝ) = 0.02 := by sorry

end NUMINAMATH_CALUDE_double_quarter_four_percent_l1929_192912


namespace NUMINAMATH_CALUDE_endpoint_from_midpoint_and_other_endpoint_l1929_192923

theorem endpoint_from_midpoint_and_other_endpoint :
  ∀ (x y : ℝ),
  (3 : ℝ) = (7 + x) / 2 →
  (2 : ℝ) = (-4 + y) / 2 →
  (x, y) = (-1, 8) := by
sorry

end NUMINAMATH_CALUDE_endpoint_from_midpoint_and_other_endpoint_l1929_192923


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1929_192917

/-- The area of the union of a square with side length 12 and a circle with radius 12
    centered at one of the square's vertices is equal to 144 + 108π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let quarter_circle_area : ℝ := circle_area / 4
  square_area + circle_area - quarter_circle_area = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1929_192917


namespace NUMINAMATH_CALUDE_max_t_value_l1929_192981

theorem max_t_value (f : ℝ → ℝ) (a : ℝ) (t : ℝ) : 
  (∀ x : ℝ, f x = (x + 1)^2) →
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ t → f (x + a) ≤ 2*x - 4) →
  t ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l1929_192981


namespace NUMINAMATH_CALUDE_enclosing_rectangle_exists_l1929_192968

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ vertices
  area : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- Checks if a polygon is enclosed within a rectangle -/
def enclosed (p : ConvexPolygon) (r : Rectangle) : Prop :=
  ∀ v ∈ p.vertices, 
    r.bottom_left.1 ≤ v.1 ∧ v.1 ≤ r.top_right.1 ∧
    r.bottom_left.2 ≤ v.2 ∧ v.2 ≤ r.top_right.2

/-- Calculates the area of a rectangle -/
def rectangle_area (r : Rectangle) : ℝ :=
  (r.top_right.1 - r.bottom_left.1) * (r.top_right.2 - r.bottom_left.2)

/-- The main theorem -/
theorem enclosing_rectangle_exists (p : ConvexPolygon) (h : p.area = 1) :
  ∃ r : Rectangle, enclosed p r ∧ rectangle_area r ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_enclosing_rectangle_exists_l1929_192968


namespace NUMINAMATH_CALUDE_boot_pairing_l1929_192971

theorem boot_pairing (total_boots : ℕ) (left_boots right_boots : ℕ) (size_count : ℕ) :
  total_boots = 600 →
  left_boots = 300 →
  right_boots = 300 →
  size_count = 3 →
  total_boots = left_boots + right_boots →
  ∃ (valid_pairs : ℕ), valid_pairs ≥ 100 ∧ 
    ∃ (size_41 size_42 size_43 : ℕ),
      size_41 + size_42 + size_43 = total_boots ∧
      size_41 = size_42 ∧ size_42 = size_43 ∧
      (∀ (size : ℕ), size ∈ [size_41, size_42, size_43] → 
        ∃ (left_count right_count : ℕ), 
          left_count + right_count = size ∧
          left_count ≤ left_boots ∧
          right_count ≤ right_boots) :=
by sorry


end NUMINAMATH_CALUDE_boot_pairing_l1929_192971


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_of_1_157_l1929_192928

theorem first_nonzero_digit_after_decimal_of_1_157 : ∃ (n : ℕ) (d : ℕ), 
  0 < d ∧ d < 10 ∧ 
  (1000 : ℚ) / 157 = 6 + (d : ℚ) / 10 + (n : ℚ) / 100 ∧ 
  d = 3 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_of_1_157_l1929_192928


namespace NUMINAMATH_CALUDE_beth_crayons_l1929_192991

/-- Given the number of crayon packs, crayons per pack, and extra crayons,
    calculate the total number of crayons Beth has. -/
def total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) : ℕ :=
  packs * crayons_per_pack + extra_crayons

/-- Prove that Beth has 46 crayons in total. -/
theorem beth_crayons : total_crayons 4 10 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l1929_192991


namespace NUMINAMATH_CALUDE_class_size_from_average_change_l1929_192950

theorem class_size_from_average_change 
  (original_mark : ℕ) 
  (incorrect_mark : ℕ)
  (mark_difference : ℕ)
  (average_increase : ℚ) :
  incorrect_mark = original_mark + mark_difference →
  mark_difference = 20 →
  average_increase = 1/2 →
  (mark_difference : ℚ) / (class_size : ℕ) = average_increase →
  class_size = 40 := by
sorry

end NUMINAMATH_CALUDE_class_size_from_average_change_l1929_192950


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1929_192998

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 - a^2 = 1 →
  b * c = 1 →
  Real.cos B * Real.cos C = -1/8 →
  a + b + c = Real.sqrt 2 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1929_192998


namespace NUMINAMATH_CALUDE_chocolate_division_l1929_192930

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) :
  total_chocolate = 72 / 7 →
  num_piles = 6 →
  piles_for_shaina = 2 →
  (total_chocolate / num_piles) * piles_for_shaina = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1929_192930


namespace NUMINAMATH_CALUDE_equation_solution_l1929_192935

theorem equation_solution : ∃ y : ℝ, (16 : ℝ) ^ (2 * y - 4) = (1 / 4 : ℝ) ^ (5 - y) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1929_192935


namespace NUMINAMATH_CALUDE_insulation_cost_for_given_tank_l1929_192999

/-- Calculates the surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulation_cost (l w h cost_per_sqft : ℝ) : ℝ :=
  surface_area l w h * cost_per_sqft

/-- Theorem: The cost of insulating a rectangular tank with given dimensions -/
theorem insulation_cost_for_given_tank :
  insulation_cost 7 3 2 20 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_insulation_cost_for_given_tank_l1929_192999


namespace NUMINAMATH_CALUDE_equation_solution_l1929_192975

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 3)^2 * (5 + x)
  {x : ℝ | f x = 0} = {0, 3, -5} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1929_192975


namespace NUMINAMATH_CALUDE_point_on_line_value_l1929_192947

theorem point_on_line_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l1929_192947


namespace NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l1929_192901

theorem cos_theta_plus_pi_fourth (θ : Real) :
  (3 : Real) = 5 * Real.cos θ ∧ (-4 : Real) = 5 * Real.sin θ →
  Real.cos (θ + π/4) = 7 * Real.sqrt 2 / 10 := by sorry

end NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l1929_192901


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l1929_192948

/-- The number of candy pieces Katie and her sister have left after eating some on Halloween night -/
theorem halloween_candy_theorem (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : 
  katie_candy = 8 → sister_candy = 23 → eaten_candy = 8 →
  katie_candy + sister_candy - eaten_candy = 23 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l1929_192948


namespace NUMINAMATH_CALUDE_cube_sphere_surface_area_ratio_l1929_192959

-- Define a cube with an inscribed sphere
structure CubeWithInscribedSphere where
  edge_length : ℝ
  sphere_radius : ℝ
  h_diameter : sphere_radius * 2 = edge_length

-- Theorem statement
theorem cube_sphere_surface_area_ratio 
  (c : CubeWithInscribedSphere) : 
  (6 * c.edge_length^2) / (4 * Real.pi * c.sphere_radius^2) = 6 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_surface_area_ratio_l1929_192959


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1929_192979

-- Define the condition p
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- Define the condition q
def q (x a : ℝ) : Prop := x ≤ a

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1929_192979


namespace NUMINAMATH_CALUDE_allison_total_items_l1929_192986

/-- Represents the number of craft items bought by a person -/
structure CraftItems where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- The problem setup -/
def craftProblem (marie allison : CraftItems) : Prop :=
  allison.glueSticks = marie.glueSticks + 8 ∧
  marie.constructionPaper = 6 * allison.constructionPaper ∧
  marie.glueSticks = 15 ∧
  marie.constructionPaper = 30

/-- The theorem to prove -/
theorem allison_total_items (marie allison : CraftItems) 
  (h : craftProblem marie allison) : 
  allison.glueSticks + allison.constructionPaper = 28 := by
  sorry


end NUMINAMATH_CALUDE_allison_total_items_l1929_192986


namespace NUMINAMATH_CALUDE_inequality_proof_l1929_192940

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1929_192940


namespace NUMINAMATH_CALUDE_binary_to_base_4_conversion_l1929_192906

-- Define the binary number
def binary_num : ℕ := 11011011

-- Define the base 4 number
def base_4_num : ℕ := 3123

-- Theorem stating the equality of the binary and base 4 representations
theorem binary_to_base_4_conversion :
  (binary_num.digits 2).foldl (λ acc d => 2 * acc + d) 0 =
  (base_4_num.digits 4).foldl (λ acc d => 4 * acc + d) 0 :=
by sorry

end NUMINAMATH_CALUDE_binary_to_base_4_conversion_l1929_192906


namespace NUMINAMATH_CALUDE_root_power_sums_equal_l1929_192994

-- Define the polynomial
def p (x : ℂ) : ℂ := x^3 + 2*x^2 + 3*x + 4

-- Define the sum of nth powers of roots
def S (n : ℕ) : ℂ := sorry

theorem root_power_sums_equal :
  S 1 = -2 ∧ S 2 = -2 ∧ S 3 = -2 := by sorry

end NUMINAMATH_CALUDE_root_power_sums_equal_l1929_192994


namespace NUMINAMATH_CALUDE_gum_distribution_l1929_192916

theorem gum_distribution (num_cousins : ℕ) (gum_per_cousin : ℕ) : 
  num_cousins = 4 → gum_per_cousin = 5 → num_cousins * gum_per_cousin = 20 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l1929_192916


namespace NUMINAMATH_CALUDE_handshake_procedure_solution_l1929_192927

/-- The remainder function modulo 5251 -/
def r_5251 (t : ℤ) : ℤ := t % 5251

/-- The function F(t) = r_5251(t^3) -/
def F (t : ℤ) : ℤ := r_5251 (t^3)

/-- Theorem stating the unique solution for x and y -/
theorem handshake_procedure_solution :
  ∃! (x y : ℤ),
    0 ≤ x ∧ x ≤ 5250 ∧
    0 ≤ y ∧ y ≤ 5250 ∧
    F x = 506 ∧
    F (x + 1) = 519 ∧
    F y = 229 ∧
    F (y + 1) = 231 ∧
    x = 102 ∧
    y = 72 := by
  sorry

end NUMINAMATH_CALUDE_handshake_procedure_solution_l1929_192927


namespace NUMINAMATH_CALUDE_john_biking_distance_l1929_192942

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem john_biking_distance :
  base7ToBase10 [2, 5, 6, 3] = 1360 := by
  sorry

end NUMINAMATH_CALUDE_john_biking_distance_l1929_192942


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_zero_l1929_192909

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define a function with exactly four zeros
def HasFourZeros (f : ℝ → ℝ) : Prop := ∃ x₁ x₂ x₃ x₄, 
  (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0) ∧
  (∀ x, f x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

theorem sum_of_zeros_is_zero (f : ℝ → ℝ) 
  (heven : EvenFunction f) (hzeros : HasFourZeros f) : 
  ∃ x₁ x₂ x₃ x₄, f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ x₁ + x₂ + x₃ + x₄ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_zero_l1929_192909


namespace NUMINAMATH_CALUDE_girls_not_attending_college_percentage_l1929_192929

theorem girls_not_attending_college_percentage
  (total_boys : ℕ)
  (total_girls : ℕ)
  (boys_not_attending_percentage : ℚ)
  (total_attending_percentage : ℚ)
  (h1 : total_boys = 300)
  (h2 : total_girls = 240)
  (h3 : boys_not_attending_percentage = 30 / 100)
  (h4 : total_attending_percentage = 70 / 100)
  : (↑(total_girls - (total_boys + total_girls) * total_attending_percentage + total_boys * boys_not_attending_percentage) / total_girls : ℚ) = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_girls_not_attending_college_percentage_l1929_192929


namespace NUMINAMATH_CALUDE_prob_odd_die_roll_l1929_192992

/-- The number of possible outcomes when rolling a die -/
def total_outcomes : ℕ := 6

/-- The number of favorable outcomes (odd numbers) when rolling a die -/
def favorable_outcomes : ℕ := 3

/-- The probability of an event in a finite sample space -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem: The probability of rolling an odd number on a standard six-sided die is 1/2 -/
theorem prob_odd_die_roll : probability favorable_outcomes total_outcomes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_die_roll_l1929_192992


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1929_192962

theorem quadratic_rewrite_ratio (k : ℝ) : 
  ∃ (d r s : ℝ), 5 * k^2 - 6 * k + 15 = d * (k + r)^2 + s ∧ s / r = -22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1929_192962


namespace NUMINAMATH_CALUDE_divisor_sum_not_divides_l1929_192952

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n

/-- The set of proper divisors of a natural number -/
def ProperDivisors (n : ℕ) : Set ℕ := {d : ℕ | d ∣ n ∧ 1 < d ∧ d < n}

/-- The set of remaining divisors after removing the smaller of each pair -/
def RemainingDivisors (n : ℕ) : Set ℕ :=
  {d ∈ ProperDivisors n | d ≥ n / d}

theorem divisor_sum_not_divides (a b : ℕ) (ha : IsComposite a) (hb : IsComposite b) :
  ∀ (c : ℕ) (d : ℕ), c ∈ RemainingDivisors a → d ∈ RemainingDivisors b →
    ¬((c + d) ∣ (a + b)) := by
  sorry

#check divisor_sum_not_divides

end NUMINAMATH_CALUDE_divisor_sum_not_divides_l1929_192952


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_with_mean_70_l1929_192905

theorem max_ratio_two_digit_integers_with_mean_70 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 70 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 70 →
  x / y ≤ 99 / 41 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_with_mean_70_l1929_192905


namespace NUMINAMATH_CALUDE_log_equation_solution_l1929_192945

theorem log_equation_solution (x : ℝ) :
  0 < x ∧ x ≠ 1 ∧ x < 10 →
  (1 + 2 * (Real.log 2 / Real.log x) * (Real.log (10 - x) / Real.log 4) = 2 / (Real.log x / Real.log 4)) ↔
  (x = 2 ∨ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1929_192945


namespace NUMINAMATH_CALUDE_hash_difference_l1929_192903

def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 6 5) - (hash 5 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1929_192903


namespace NUMINAMATH_CALUDE_brothers_to_madelines_money_ratio_l1929_192960

theorem brothers_to_madelines_money_ratio (madelines_money : ℕ) (total_money : ℕ) : 
  madelines_money = 48 →
  total_money = 72 →
  (total_money - madelines_money) * 2 = madelines_money :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_to_madelines_money_ratio_l1929_192960


namespace NUMINAMATH_CALUDE_arrangements_with_conditions_l1929_192946

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def arrangements_of_n (n : ℕ) : ℕ := factorial n

def arrangements_with_left_end (n : ℕ) : ℕ := factorial (n - 1)

def arrangements_adjacent (n : ℕ) : ℕ := 2 * factorial (n - 1)

def arrangements_left_end_and_adjacent (n : ℕ) : ℕ := factorial (n - 2)

theorem arrangements_with_conditions (n : ℕ) (h : n = 5) : 
  arrangements_of_n n - arrangements_with_left_end n - arrangements_adjacent n + arrangements_left_end_and_adjacent n = 54 :=
sorry

end NUMINAMATH_CALUDE_arrangements_with_conditions_l1929_192946


namespace NUMINAMATH_CALUDE_total_sharks_l1929_192926

theorem total_sharks (newport_sharks : ℕ) (dana_point_sharks : ℕ) : 
  newport_sharks = 22 → 
  dana_point_sharks = 4 * newport_sharks → 
  newport_sharks + dana_point_sharks = 110 := by
sorry

end NUMINAMATH_CALUDE_total_sharks_l1929_192926


namespace NUMINAMATH_CALUDE_fraction_equality_l1929_192969

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1929_192969


namespace NUMINAMATH_CALUDE_error_percentage_l1929_192941

theorem error_percentage (y : ℝ) (h : y > 0) : 
  (|5 * y - y / 4| / (5 * y)) * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_error_percentage_l1929_192941


namespace NUMINAMATH_CALUDE_solution_set_implies_a_l1929_192956

theorem solution_set_implies_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 6 ≤ 0 ↔ 2 ≤ x ∧ x ≤ 3) → a = -5 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_l1929_192956


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1929_192934

/-- The function f(x) = a^(x-2015) + 2015 passes through the point (2015, 2016) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 2015) + 2015
  f 2015 = 2016 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1929_192934


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l1929_192908

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of f being increasing on (-∞, -1]
def increasing_on_neg_infinity_to_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ -1 → f x < f y

-- State the theorem
theorem even_increasing_inequality 
  (h_even : is_even f) 
  (h_incr : increasing_on_neg_infinity_to_neg_one f) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l1929_192908


namespace NUMINAMATH_CALUDE_remainder_meters_after_marathons_l1929_192957

/-- The length of a marathon in kilometers -/
def marathon_length : ℝ := 42.195

/-- The number of marathons run -/
def num_marathons : ℕ := 15

/-- The number of meters in a kilometer -/
def meters_per_km : ℕ := 1000

/-- The total distance in kilometers -/
def total_distance : ℝ := marathon_length * num_marathons

theorem remainder_meters_after_marathons :
  ∃ (k : ℕ) (m : ℕ), 
    total_distance = k + (m : ℝ) / meters_per_km ∧ 
    m < meters_per_km ∧ 
    m = 925 := by sorry

end NUMINAMATH_CALUDE_remainder_meters_after_marathons_l1929_192957


namespace NUMINAMATH_CALUDE_coles_fence_payment_l1929_192953

/-- Calculates Cole's payment for fencing his backyard -/
theorem coles_fence_payment
  (side_length : ℝ)
  (back_length : ℝ)
  (fence_cost_per_foot : ℝ)
  (back_neighbor_contribution_ratio : ℝ)
  (left_neighbor_contribution_ratio : ℝ)
  (h1 : side_length = 9)
  (h2 : back_length = 18)
  (h3 : fence_cost_per_foot = 3)
  (h4 : back_neighbor_contribution_ratio = 1/2)
  (h5 : left_neighbor_contribution_ratio = 1/3) :
  side_length * 2 + back_length * fence_cost_per_foot -
  (back_length * back_neighbor_contribution_ratio * fence_cost_per_foot +
   side_length * left_neighbor_contribution_ratio * fence_cost_per_foot) = 72 :=
by sorry

end NUMINAMATH_CALUDE_coles_fence_payment_l1929_192953


namespace NUMINAMATH_CALUDE_sum_of_three_times_m_and_half_n_square_diff_minus_square_sum_l1929_192980

-- Part 1
theorem sum_of_three_times_m_and_half_n (m n : ℝ) :
  3 * m + (1/2) * n = 3 * m + (1/2) * n := by sorry

-- Part 2
theorem square_diff_minus_square_sum (a b : ℝ) :
  (a - b)^2 - (a + b)^2 = (a - b)^2 - (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_times_m_and_half_n_square_diff_minus_square_sum_l1929_192980


namespace NUMINAMATH_CALUDE_divisibility_from_point_distribution_l1929_192915

theorem divisibility_from_point_distribution (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_k_le_n : k ≤ n)
  (points : Finset ℝ) (h_card : points.card = n)
  (h_divisible : ∀ x ∈ points, (points.filter (λ y => |y - x| ≤ 1)).card % k = 0) :
  k ∣ n := by
sorry

end NUMINAMATH_CALUDE_divisibility_from_point_distribution_l1929_192915


namespace NUMINAMATH_CALUDE_rectangle_circles_radii_sum_l1929_192965

theorem rectangle_circles_radii_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * b^2 + a^2) / (4 * b) + (4 * a^2 + b^2) / (4 * a) ≥ 5 * (a + b) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circles_radii_sum_l1929_192965


namespace NUMINAMATH_CALUDE_length_XX₁_l1929_192919

-- Define the triangles and circle
def triangle_DEF (DE DF : ℝ) : Prop := DE = 7 ∧ DF = 3
def inscribed_circle (F₁ : ℝ × ℝ) : Prop := sorry  -- Details of circle inscription

-- Define the second triangle XYZ
def triangle_XYZ (XY XZ : ℝ) (F₁E F₁D : ℝ) : Prop :=
  XY = F₁E ∧ XZ = F₁D

-- Define the angle bisector and point X₁
def angle_bisector (X₁ : ℝ × ℝ) : Prop := sorry  -- Details of angle bisector

-- Main theorem
theorem length_XX₁ (DE DF : ℝ) (F₁ : ℝ × ℝ) (XY XZ : ℝ) (X₁ : ℝ × ℝ) :
  triangle_DEF DE DF →
  inscribed_circle F₁ →
  triangle_XYZ XY XZ (Real.sqrt 10 - 2) (Real.sqrt 10 - 2) →
  angle_bisector X₁ →
  ∃ (XX₁ : ℝ), XX₁ = 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_length_XX₁_l1929_192919


namespace NUMINAMATH_CALUDE_complement_of_60_degrees_l1929_192990

def angle : ℝ := 60

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem complement_of_60_degrees :
  complement angle = 30 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_60_degrees_l1929_192990


namespace NUMINAMATH_CALUDE_exponent_division_l1929_192939

theorem exponent_division (a : ℝ) : a^10 / a^9 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1929_192939


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1929_192943

/-- Given the prices of adult and child tickets, the total number of tickets sold,
    and the total revenue, prove the number of adult tickets sold. -/
theorem adult_tickets_sold
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100)
  : ∃ (adult_tickets : ℕ),
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧
    adult_tickets = 500 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l1929_192943


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l1929_192996

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, domain x → f_inv (f x) = x) ∧
    (∀ y, ∃ x, domain x ∧ f x = y → f_inv y = x) ∧
    f_inv 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l1929_192996


namespace NUMINAMATH_CALUDE_remainder_17_pow_33_mod_7_l1929_192900

theorem remainder_17_pow_33_mod_7 : 17^33 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_17_pow_33_mod_7_l1929_192900


namespace NUMINAMATH_CALUDE_min_value_expression_l1929_192985

theorem min_value_expression (x : ℝ) :
  x ≥ 0 →
  (1 + x^2) / (1 + x) ≥ -2 + 2 * Real.sqrt 2 ∧
  ∃ y : ℝ, y ≥ 0 ∧ (1 + y^2) / (1 + y) = -2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1929_192985


namespace NUMINAMATH_CALUDE_some_number_value_l1929_192970

theorem some_number_value (x : ℚ) :
  (3 / 5 : ℚ) * ((2 / 3 + 3 / 8) / x) - 1 / 16 = 0.24999999999999994 →
  x = 48 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1929_192970


namespace NUMINAMATH_CALUDE_distance_from_origin_l1929_192958

theorem distance_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1929_192958


namespace NUMINAMATH_CALUDE_potato_cooking_time_l1929_192955

theorem potato_cooking_time 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_potatoes = 16) 
  (h2 : cooked_potatoes = 7) 
  (h3 : remaining_time = 45) :
  remaining_time / (total_potatoes - cooked_potatoes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l1929_192955


namespace NUMINAMATH_CALUDE_inscribed_square_area_is_2210_l1929_192987

/-- Represents a triangle with an inscribed square -/
structure TriangleWithInscribedSquare where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed in the triangle -/
  is_inscribed : square_side > 0

/-- The area of the inscribed square in the given triangle -/
def inscribed_square_area (t : TriangleWithInscribedSquare) : ℝ :=
  t.square_side^2

/-- Theorem: The area of the inscribed square is 2210 when PQ = 34 and PR = 65 -/
theorem inscribed_square_area_is_2210
    (t : TriangleWithInscribedSquare)
    (h_pq : t.pq = 34)
    (h_pr : t.pr = 65) :
    inscribed_square_area t = 2210 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_is_2210_l1929_192987


namespace NUMINAMATH_CALUDE_prob_A_and_B_selected_is_three_tenths_l1929_192936

def total_students : ℕ := 5
def students_to_select : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (total_students - students_to_select + 1 : ℚ) / (total_students.choose students_to_select)

theorem prob_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_and_B_selected_is_three_tenths_l1929_192936


namespace NUMINAMATH_CALUDE_clothes_transport_expenditure_l1929_192918

/-- Calculates the monthly amount spent on clothes and transport given the yearly savings --/
def monthly_clothes_transport (yearly_savings : ℕ) : ℕ :=
  let monthly_savings := yearly_savings / 12
  let monthly_salary := monthly_savings * 5
  monthly_salary / 5

/-- Theorem stating that given the conditions in the problem, 
    the monthly amount spent on clothes and transport is 4038 --/
theorem clothes_transport_expenditure :
  monthly_clothes_transport 48456 = 4038 := by
  sorry

#eval monthly_clothes_transport 48456

end NUMINAMATH_CALUDE_clothes_transport_expenditure_l1929_192918


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1929_192993

theorem divisibility_by_five (a : ℤ) : 
  5 ∣ (a^3 + 3*a + 1) ↔ a % 5 = 1 ∨ a % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1929_192993


namespace NUMINAMATH_CALUDE_flower_arrangement_theorem_l1929_192921

/-- Represents a flower arrangement on a square -/
structure FlowerArrangement where
  corners : ℕ  -- number of flowers at each corner
  midpoints : ℕ  -- number of flowers at each midpoint

/-- The total number of flowers in the arrangement -/
def total_flowers (arrangement : FlowerArrangement) : ℕ :=
  4 * arrangement.corners + 4 * arrangement.midpoints

/-- The number of flowers seen on each side of the square -/
def flowers_per_side (arrangement : FlowerArrangement) : ℕ :=
  2 * arrangement.corners + arrangement.midpoints

theorem flower_arrangement_theorem :
  (∃ (arr : FlowerArrangement), 
    flowers_per_side arr = 9 ∧ 
    total_flowers arr = 36 ∧ 
    (∀ (other : FlowerArrangement), flowers_per_side other = 9 → total_flowers other ≤ 36)) ∧
  (∃ (arr : FlowerArrangement), 
    flowers_per_side arr = 12 ∧ 
    total_flowers arr = 24 ∧ 
    (∀ (other : FlowerArrangement), flowers_per_side other = 12 → total_flowers other ≥ 24)) :=
by sorry

end NUMINAMATH_CALUDE_flower_arrangement_theorem_l1929_192921


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1929_192902

theorem quadratic_root_property (a : ℝ) : 
  (a^2 + 3*a - 5 = 0) → (-a^2 - 3*a = -5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1929_192902


namespace NUMINAMATH_CALUDE_min_value_expression_l1929_192954

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
  (2 * x^2 + 1) / (x * y) - 2 ≥ min_val ∧
  (2 * a^2 + 1) / (a * b) - 2 = min_val ↔ a = (Real.sqrt 3 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1929_192954


namespace NUMINAMATH_CALUDE_square_less_than_triple_l1929_192977

theorem square_less_than_triple (n : ℤ) : n^2 < 3*n ↔ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l1929_192977


namespace NUMINAMATH_CALUDE_beaded_corset_cost_l1929_192937

/-- The number of rows of purple beads -/
def purple_rows : ℕ := 50

/-- The number of beads per row of purple beads -/
def purple_beads_per_row : ℕ := 20

/-- The number of rows of blue beads -/
def blue_rows : ℕ := 40

/-- The number of beads per row of blue beads -/
def blue_beads_per_row : ℕ := 18

/-- The number of gold beads -/
def gold_beads : ℕ := 80

/-- The cost of beads in dollars per 10 beads -/
def cost_per_10_beads : ℚ := 1

/-- The total cost of all beads in dollars -/
def total_cost : ℚ := 180

theorem beaded_corset_cost :
  (purple_rows * purple_beads_per_row + blue_rows * blue_beads_per_row + gold_beads) / 10 * cost_per_10_beads = total_cost := by
  sorry

end NUMINAMATH_CALUDE_beaded_corset_cost_l1929_192937


namespace NUMINAMATH_CALUDE_kim_payment_amount_l1929_192924

def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_percentage : ℝ := 0.2
def change_received : ℝ := 5

theorem kim_payment_amount :
  let total_before_tip := meal_cost + drink_cost
  let tip := tip_percentage * total_before_tip
  let total_with_tip := total_before_tip + tip
  let payment_amount := total_with_tip + change_received
  payment_amount = 20 := by sorry

end NUMINAMATH_CALUDE_kim_payment_amount_l1929_192924


namespace NUMINAMATH_CALUDE_circle_condition_l1929_192966

theorem circle_condition (m : ℝ) : 
  (∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ 
  (m < 1/4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1929_192966


namespace NUMINAMATH_CALUDE_dumpling_selection_probability_l1929_192931

/-- The number of dumplings of each kind in the pot -/
def dumplings_per_kind : ℕ := 5

/-- The number of different kinds of dumplings -/
def kinds_of_dumplings : ℕ := 3

/-- The total number of dumplings in the pot -/
def total_dumplings : ℕ := dumplings_per_kind * kinds_of_dumplings

/-- The number of dumplings to be selected -/
def selected_dumplings : ℕ := 4

/-- The probability of selecting at least one dumpling of each kind -/
def probability_at_least_one_of_each : ℚ := 50 / 91

theorem dumpling_selection_probability :
  (Nat.choose total_dumplings selected_dumplings *
   probability_at_least_one_of_each : ℚ) =
  (Nat.choose kinds_of_dumplings 1 *
   Nat.choose dumplings_per_kind 2 *
   Nat.choose dumplings_per_kind 1 *
   Nat.choose dumplings_per_kind 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_dumpling_selection_probability_l1929_192931


namespace NUMINAMATH_CALUDE_non_officers_count_l1929_192938

/-- Prove the number of non-officers in an office given salary information -/
theorem non_officers_count (avg_salary : ℝ) (officer_salary : ℝ) (non_officer_salary : ℝ) 
  (officer_count : ℕ) (h1 : avg_salary = 120) (h2 : officer_salary = 430) 
  (h3 : non_officer_salary = 110) (h4 : officer_count = 15) : 
  ∃ (non_officer_count : ℕ), 
    avg_salary * (officer_count + non_officer_count) = 
    officer_salary * officer_count + non_officer_salary * non_officer_count ∧ 
    non_officer_count = 465 := by
  sorry

end NUMINAMATH_CALUDE_non_officers_count_l1929_192938


namespace NUMINAMATH_CALUDE_sum_of_one_third_and_two_thirds_equals_one_l1929_192972

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  (n : ℚ) / 9

theorem sum_of_one_third_and_two_thirds_equals_one :
  RepeatingDecimal 3 + RepeatingDecimal 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_one_third_and_two_thirds_equals_one_l1929_192972


namespace NUMINAMATH_CALUDE_cost_of_3000_pencils_l1929_192910

/-- The cost of purchasing a given number of pencils with a bulk discount. -/
def cost_of_pencils (box_size : ℕ) (box_price : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) (num_pencils : ℕ) : ℚ :=
  let unit_price := box_price / box_size
  let total_price := unit_price * num_pencils
  if num_pencils > discount_threshold then
    total_price * (1 - discount_rate)
  else
    total_price

/-- Theorem stating that the cost of 3000 pencils is $675 given the problem conditions. -/
theorem cost_of_3000_pencils :
  cost_of_pencils 200 50 1000 (1/10) 3000 = 675 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_3000_pencils_l1929_192910


namespace NUMINAMATH_CALUDE_G_difference_l1929_192925

/-- G is defined as the infinite repeating decimal 0.737373... -/
def G : ℚ := 73 / 99

/-- The difference between the denominator and numerator of G when expressed as a fraction in lowest terms -/
def difference : ℕ := 99 - 73

theorem G_difference : difference = 26 := by sorry

end NUMINAMATH_CALUDE_G_difference_l1929_192925


namespace NUMINAMATH_CALUDE_rectangle_tiling_exists_l1929_192982

/-- A tiling of a rectangle using two layers of 1 × 2 bricks -/
structure Tiling (n m : ℕ) :=
  (layer1 : Fin n → Fin (2*m) → Bool)
  (layer2 : Fin n → Fin (2*m) → Bool)

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (n m : ℕ) (t : Tiling n m) : Prop :=
  (∀ i j, t.layer1 i j ∨ t.layer2 i j) ∧ 
  (∀ i j, ¬(t.layer1 i j ∧ t.layer2 i j))

/-- Main theorem: A valid tiling exists for any rectangle n × 2m where n > 1 -/
theorem rectangle_tiling_exists (n m : ℕ) (h : n > 1) : 
  ∃ t : Tiling n m, is_valid_tiling n m t :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_exists_l1929_192982


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1929_192904

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), ∀ x, f x = (x + 1) * (x / 2 - a / Real.sqrt x)^6) →
  (∃ (g : ℝ → ℝ), ∀ x, g x = (x + 1) * (x / 2 - a / Real.sqrt x)^6 ∧ 
    (∃ c, c = 60 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |g x - c| < ε))) →
  a = 2 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1929_192904


namespace NUMINAMATH_CALUDE_store_purchase_total_l1929_192949

/-- Calculate the total amount spent at the store -/
theorem store_purchase_total (initial_backpack_price initial_binder_price : ℚ)
  (backpack_increase binder_decrease : ℚ)
  (backpack_discount binder_deal sales_tax : ℚ)
  (num_binders : ℕ) :
  let new_backpack_price := initial_backpack_price + backpack_increase
  let new_binder_price := initial_binder_price - binder_decrease
  let discounted_backpack_price := new_backpack_price * (1 - backpack_discount)
  let binders_to_pay := (num_binders + 1) / 2
  let total_binder_price := new_binder_price * binders_to_pay
  let subtotal := discounted_backpack_price + total_binder_price
  let total_with_tax := subtotal * (1 + sales_tax)
  initial_backpack_price = 50 ∧
  initial_binder_price = 20 ∧
  backpack_increase = 5 ∧
  binder_decrease = 2 ∧
  backpack_discount = 0.1 ∧
  sales_tax = 0.06 ∧
  num_binders = 3 →
  total_with_tax = 90.63 :=
by sorry

end NUMINAMATH_CALUDE_store_purchase_total_l1929_192949


namespace NUMINAMATH_CALUDE_find_a_l1929_192984

def U (a : ℝ) : Set ℝ := {3, 7, a^2 - 2*a - 3}

def A (a : ℝ) : Set ℝ := {7, |a - 7|}

theorem find_a : ∃ a : ℝ, (U a \ A a = {5}) ∧ (A a ⊆ U a) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1929_192984


namespace NUMINAMATH_CALUDE_smallest_number_with_three_prime_factors_ge_10_l1929_192983

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_three_prime_factors (n : ℕ) : Prop := sorry

def all_prime_factors_ge_10 (n : ℕ) : Prop := sorry

theorem smallest_number_with_three_prime_factors_ge_10 :
  ∀ n : ℕ, (has_exactly_three_prime_factors n ∧ all_prime_factors_ge_10 n) → n ≥ 2431 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_three_prime_factors_ge_10_l1929_192983


namespace NUMINAMATH_CALUDE_work_left_fraction_l1929_192961

/-- The fraction of work left after two workers work together for a certain number of days -/
def fractionLeft (daysA : ℕ) (daysB : ℕ) (daysTogether : ℕ) : ℚ :=
  1 - (daysTogether : ℚ) * (1 / daysA + 1 / daysB)

/-- Theorem stating that if A can complete the job in 15 days and B in 20 days, 
    working together for 4 days leaves 8/15 of the work -/
theorem work_left_fraction :
  fractionLeft 15 20 4 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l1929_192961


namespace NUMINAMATH_CALUDE_proposition_correctness_l1929_192997

theorem proposition_correctness : 
  (∃ (S : Finset (Prop)), 
    S.card = 4 ∧ 
    (∃ (incorrect : Finset (Prop)), 
      incorrect ⊆ S ∧ 
      incorrect.card = 2 ∧
      (∀ p ∈ S, p ∈ incorrect ↔ ¬p) ∧
      (∃ p ∈ S, p = (∀ (p q : Prop), p ∨ q → p ∧ q)) ∧
      (∃ p ∈ S, p = (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧ 
                   (∃ y : ℝ, y^2 - 4*y - 5 > 0 ∧ y ≤ 5)) ∧
      (∃ p ∈ S, p = ((¬∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0))) ∧
      (∃ p ∈ S, p = (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → x^2 - 3*x + 2 ≠ 0)))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_correctness_l1929_192997


namespace NUMINAMATH_CALUDE_quadratic_bounded_values_l1929_192920

/-- A quadratic function f(x) = ax^2 + bx + c where a > 100 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_bounded_values (a b c : ℝ) (ha : a > 100) :
  ∃ (n : ℕ), n ≤ 2 ∧
  ∀ (S : Finset ℤ), (∀ x ∈ S, |QuadraticFunction a b c x| ≤ 50) →
  Finset.card S ≤ n :=
sorry

end NUMINAMATH_CALUDE_quadratic_bounded_values_l1929_192920


namespace NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l1929_192914

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x < 2}

theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l1929_192914


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1929_192967

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents two concentric circles -/
structure ConcentricCircles where
  inner : Circle
  outer : Circle
  h : inner.radius < outer.radius

/-- Represents three circles tangent to two concentric circles and to each other -/
structure TangentCircles (cc : ConcentricCircles) where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  tangent_to_concentric : 
    c1.radius = cc.outer.radius - cc.inner.radius ∧
    c2.radius = cc.outer.radius - cc.inner.radius ∧
    c3.radius = cc.outer.radius - cc.inner.radius
  tangent_to_each_other : True  -- This is a simplification, as we can't easily express tangency

/-- The main theorem: If three circles are tangent to two concentric circles and to each other,
    then the ratio of the radii of the concentric circles is 3 -/
theorem concentric_circles_ratio 
  (cc : ConcentricCircles) 
  (tc : TangentCircles cc) : 
  cc.outer.radius / cc.inner.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1929_192967


namespace NUMINAMATH_CALUDE_multiples_equality_l1929_192922

def c : ℕ := (Finset.filter (fun n => 12 ∣ n ∧ n < 60) (Finset.range 60)).card

def d : ℕ := (Finset.filter (fun n => 3 ∣ n ∧ 4 ∣ n ∧ n < 60) (Finset.range 60)).card

theorem multiples_equality : (c - d)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiples_equality_l1929_192922


namespace NUMINAMATH_CALUDE_clare_bought_four_loaves_l1929_192995

def clares_bread_purchase (initial_money : ℕ) (milk_cartons : ℕ) (bread_cost : ℕ) (milk_cost : ℕ) (money_left : ℕ) : ℕ :=
  ((initial_money - money_left) - (milk_cartons * milk_cost)) / bread_cost

theorem clare_bought_four_loaves :
  clares_bread_purchase 47 2 2 2 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_clare_bought_four_loaves_l1929_192995


namespace NUMINAMATH_CALUDE_snow_cone_price_is_0_875_l1929_192964

/-- Calculates the price of a snow cone given the conditions of Todd's snow-cone stand. -/
def snow_cone_price (borrowed : ℚ) (repay : ℚ) (ingredients_cost : ℚ) (num_sold : ℕ) (leftover : ℚ) : ℚ :=
  (repay + leftover) / num_sold

/-- Proves that the price of each snow cone is $0.875 under the given conditions. -/
theorem snow_cone_price_is_0_875 :
  snow_cone_price 100 110 75 200 65 = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_snow_cone_price_is_0_875_l1929_192964


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1929_192989

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1929_192989


namespace NUMINAMATH_CALUDE_inner_circles_radii_l1929_192976

/-- An isosceles triangle with a 120° angle and an inscribed circle of radius R -/
structure IsoscelesTriangle120 where
  R : ℝ
  R_pos : R > 0

/-- Two equal circles inside the triangle that touch each other,
    where each circle touches one leg of the triangle and the inscribed circle -/
structure InnerCircles (t : IsoscelesTriangle120) where
  radius : ℝ
  radius_pos : radius > 0

/-- The theorem stating the possible radii of the inner circles -/
theorem inner_circles_radii (t : IsoscelesTriangle120) (c : InnerCircles t) :
  c.radius = t.R / 3 ∨ c.radius = (3 - 2 * Real.sqrt 2) / 3 * t.R :=
by sorry

end NUMINAMATH_CALUDE_inner_circles_radii_l1929_192976


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1929_192911

theorem quadratic_roots_relation (m n p q : ℤ) (r₁ r₂ : ℝ) : 
  (r₁^2 - m*r₁ + n = 0 ∧ r₂^2 - m*r₂ + n = 0) →
  (r₁^4 - p*r₁^2 + q = 0 ∧ r₂^4 - p*r₂^2 + q = 0) →
  p = m^2 - 2*n :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1929_192911


namespace NUMINAMATH_CALUDE_range_of_a_l1929_192974

-- Define the propositions p, q, and r
def p (x : ℝ) : Prop := (x - 3) * (x + 1) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0
def r (x a : ℝ) : Prop := a < x ∧ x < 2 * a

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, (p x ∧ q x) → r x a) →
  (3 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1929_192974


namespace NUMINAMATH_CALUDE_factorial_ratio_eleven_nine_l1929_192944

theorem factorial_ratio_eleven_nine : Nat.factorial 11 / Nat.factorial 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_eleven_nine_l1929_192944


namespace NUMINAMATH_CALUDE_blue_marbles_count_l1929_192907

theorem blue_marbles_count (yellow green black : ℕ) (total : ℕ) (prob_black : ℚ) :
  yellow = 12 →
  green = 5 →
  black = 1 →
  prob_black = 1 / 28 →
  total = yellow + green + black + (total - yellow - green - black) →
  prob_black = black / total →
  (total - yellow - green - black) = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l1929_192907


namespace NUMINAMATH_CALUDE_nested_root_equation_l1929_192951

theorem nested_root_equation (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → M^(1/d + 1/(d*e) + 1/(d*e*f)) = M^(17/24)) →
  e = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_root_equation_l1929_192951


namespace NUMINAMATH_CALUDE_macey_savings_l1929_192933

/-- The amount Macey has already saved is equal to the cost of the shirt minus the amount she will save in the next 3 weeks. -/
theorem macey_savings (shirt_cost : ℝ) (weeks_left : ℕ) (weekly_savings : ℝ) 
  (h1 : shirt_cost = 3)
  (h2 : weeks_left = 3)
  (h3 : weekly_savings = 0.5) :
  shirt_cost - (weeks_left : ℝ) * weekly_savings = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_macey_savings_l1929_192933


namespace NUMINAMATH_CALUDE_divide_fractions_three_sevenths_div_two_and_half_l1929_192988

theorem divide_fractions (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem three_sevenths_div_two_and_half :
  (3 : ℚ) / 7 / (5 / 2) = 6 / 35 := by sorry

end NUMINAMATH_CALUDE_divide_fractions_three_sevenths_div_two_and_half_l1929_192988


namespace NUMINAMATH_CALUDE_seashells_found_joan_seashells_l1929_192963

theorem seashells_found (given_to_mike : ℕ) (has_now : ℕ) : ℕ :=
  given_to_mike + has_now

theorem joan_seashells : seashells_found 63 16 = 79 := by
  sorry

end NUMINAMATH_CALUDE_seashells_found_joan_seashells_l1929_192963


namespace NUMINAMATH_CALUDE_valid_schedules_l1929_192932

/-- Number of periods in a day -/
def total_periods : ℕ := 8

/-- Number of periods in the morning -/
def morning_periods : ℕ := 5

/-- Number of periods in the afternoon -/
def afternoon_periods : ℕ := 3

/-- Number of classes to teach -/
def classes_to_teach : ℕ := 3

/-- Calculate the number of ways to arrange n items taken k at a time -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of valid teaching schedules -/
theorem valid_schedules : 
  arrange total_periods classes_to_teach - 
  (morning_periods * arrange morning_periods classes_to_teach) - 
  arrange afternoon_periods classes_to_teach = 312 := by sorry

end NUMINAMATH_CALUDE_valid_schedules_l1929_192932


namespace NUMINAMATH_CALUDE_opposite_property_opposite_of_neg_two_l1929_192913

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_property (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -2 is 2 -/
theorem opposite_of_neg_two :
  opposite (-2 : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_property_opposite_of_neg_two_l1929_192913


namespace NUMINAMATH_CALUDE_expression_independent_of_alpha_l1929_192973

theorem expression_independent_of_alpha :
  ∀ α : ℝ, 
    Real.sin (250 * π / 180 + α) * Real.cos (200 * π / 180 - α) - 
    Real.cos (240 * π / 180) * Real.cos (220 * π / 180 - 2 * α) = 
    1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_independent_of_alpha_l1929_192973


namespace NUMINAMATH_CALUDE_brads_balloons_l1929_192978

/-- Brad's balloon count problem -/
theorem brads_balloons (red : ℕ) (green : ℕ) 
  (h1 : red = 8) 
  (h2 : green = 9) : 
  red + green = 17 := by
  sorry

end NUMINAMATH_CALUDE_brads_balloons_l1929_192978
