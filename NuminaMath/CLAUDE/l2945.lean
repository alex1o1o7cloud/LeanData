import Mathlib

namespace NUMINAMATH_CALUDE_triplet_equality_l2945_294531

theorem triplet_equality (p q r s t u v : ℤ) : 
  -- Formulation 1
  (q + r + (p + r) + (2*p + 2*q + r) = r + (p + 2*q + r) + (2*p + q + r)) ∧
  ((q + r)^2 + (p + r)^2 + (2*p + 2*q + r)^2 = r^2 + (p + 2*q + r)^2 + (2*p + q + r)^2) ∧
  -- Formulation 2
  (u*v = s*t → 
    (s + t + (u + v) = u + v + (s + t)) ∧
    (s^2 + t^2 + (u + v)^2 = u^2 + v^2 + (s + t)^2)) := by
  sorry

end NUMINAMATH_CALUDE_triplet_equality_l2945_294531


namespace NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l2945_294588

theorem cos_squared_pi_fourth_minus_alpha (α : ℝ) 
  (h : Real.sin α - Real.cos α = 4/3) : 
  Real.cos (π/4 - α)^2 = 1/9 := by sorry

end NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l2945_294588


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2945_294577

theorem arctan_equation_solution :
  ∃ x : ℝ, x > 0 ∧ Real.arctan (1/x) + Real.arctan (1/x^2) + Real.arctan (1/x^3) = π/4 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2945_294577


namespace NUMINAMATH_CALUDE_direction_vector_of_determinant_line_l2945_294515

/-- Given a line in 2D space defined by the determinant equation |x y; 2 1| = 3,
    prove that (-2, -1) is a direction vector of this line. -/
theorem direction_vector_of_determinant_line :
  let line := {(x, y) : ℝ × ℝ | x - 2*y = 3}
  ((-2 : ℝ), -1) ∈ {v : ℝ × ℝ | ∃ (t : ℝ), ∀ (p q : ℝ × ℝ), p ∈ line → q ∈ line → ∃ (s : ℝ), q.1 - p.1 = s * v.1 ∧ q.2 - p.2 = s * v.2} :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_of_determinant_line_l2945_294515


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l2945_294533

theorem estimate_larger_than_original 
  (x y a b ε : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x > y) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hab : a > b) 
  (hε : ε > 0) : 
  (a * x + ε) - (b * y - ε) > a * x - b * y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l2945_294533


namespace NUMINAMATH_CALUDE_candy_groups_l2945_294595

theorem candy_groups (total_candies : ℕ) (group_size : ℕ) (h1 : total_candies = 30) (h2 : group_size = 3) :
  total_candies / group_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_groups_l2945_294595


namespace NUMINAMATH_CALUDE_marble_241_is_blue_l2945_294536

/-- Represents the color of a marble -/
inductive MarbleColor
| Blue
| Red
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 14 with
  | 0 | 1 | 2 | 3 | 4 | 5 => MarbleColor.Blue
  | 6 | 7 | 8 | 9 | 10 => MarbleColor.Red
  | _ => MarbleColor.Green

/-- Theorem: The 241st marble in the sequence is blue -/
theorem marble_241_is_blue : marbleColor 241 = MarbleColor.Blue := by
  sorry

end NUMINAMATH_CALUDE_marble_241_is_blue_l2945_294536


namespace NUMINAMATH_CALUDE_additional_cakes_count_l2945_294555

/-- Represents the number of cakes Baker initially made -/
def initial_cakes : ℕ := 62

/-- Represents the number of cakes Baker sold -/
def sold_cakes : ℕ := 144

/-- Represents the number of cakes Baker still has -/
def remaining_cakes : ℕ := 67

/-- Theorem stating the number of additional cakes Baker made -/
theorem additional_cakes_count : 
  ∃ x : ℕ, initial_cakes + x - sold_cakes = remaining_cakes ∧ x = 149 := by
  sorry

end NUMINAMATH_CALUDE_additional_cakes_count_l2945_294555


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2945_294520

/-- The trajectory of the center of a moving circle externally tangent to a fixed circle and the y-axis -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    -- The moving circle is externally tangent to (x-2)^2 + y^2 = 1
    ((x - 2)^2 + y^2 = (r + 1)^2) ∧ 
    -- The moving circle is tangent to the y-axis
    (x = r)) →
  y^2 = 6*x - 3 := by
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2945_294520


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2945_294574

/-- Given that the solution set of the inequality (ax)/(x-1) > 1 is (1, 2), prove that a = 1/2 --/
theorem inequality_solution_implies_a_value (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) ↔ (a * x) / (x - 1) > 1) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2945_294574


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_divisibility_l2945_294522

theorem prime_sum_of_squares_divisibility (p : ℕ) (h_prime : Prime p) 
  (h_sum : ∃ a : ℕ, 2 * p = a^2 + (a+1)^2 + (a+2)^2 + (a+3)^2) : 
  36 ∣ (p - 7) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_divisibility_l2945_294522


namespace NUMINAMATH_CALUDE_equation_solution_l2945_294590

theorem equation_solution (m : ℤ) : 
  (∃ x : ℕ+, 2 * m * x - 8 = (m + 2) * x) → 
  m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2945_294590


namespace NUMINAMATH_CALUDE_integer_fraction_sum_l2945_294509

theorem integer_fraction_sum (n : ℕ) : n > 0 →
  (∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    x + y + z = 0 ∧ 
    (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = (1 : ℚ) / n) ↔ 
  ∃ (k : ℕ), n = 2 * k ∧ k > 0 :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_sum_l2945_294509


namespace NUMINAMATH_CALUDE_probability_yellow_second_marble_l2945_294500

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def probDrawColor (bag : BagContents) (color : String) : ℚ :=
  if color = bag.color1 then
    bag.count1 / (bag.count1 + bag.count2)
  else if color = bag.color2 then
    bag.count2 / (bag.count1 + bag.count2)
  else
    0

theorem probability_yellow_second_marble 
  (bagX : BagContents)
  (bagY : BagContents)
  (bagZ : BagContents)
  (h1 : bagX = { color1 := "white", count1 := 5, color2 := "black", count2 := 5 })
  (h2 : bagY = { color1 := "yellow", count1 := 8, color2 := "blue", count2 := 2 })
  (h3 : bagZ = { color1 := "yellow", count1 := 3, color2 := "blue", count2 := 4 })
  : probDrawColor bagX "white" * probDrawColor bagY "yellow" +
    probDrawColor bagX "black" * probDrawColor bagZ "yellow" = 43/70 := by
  sorry


end NUMINAMATH_CALUDE_probability_yellow_second_marble_l2945_294500


namespace NUMINAMATH_CALUDE_absolute_value_squared_l2945_294585

theorem absolute_value_squared (a b : ℝ) : |a| < b → a^2 < b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_squared_l2945_294585


namespace NUMINAMATH_CALUDE_time_to_bernards_house_l2945_294564

/-- Given June's biking rate and the distance to Bernard's house, prove the time to bike there --/
theorem time_to_bernards_house 
  (distance_to_julia : ℝ) 
  (time_to_julia : ℝ) 
  (distance_to_bernard : ℝ) 
  (h1 : distance_to_julia = 2) 
  (h2 : time_to_julia = 8) 
  (h3 : distance_to_bernard = 6) : 
  (time_to_julia / distance_to_julia) * distance_to_bernard = 24 := by
  sorry

end NUMINAMATH_CALUDE_time_to_bernards_house_l2945_294564


namespace NUMINAMATH_CALUDE_log_product_equation_l2945_294532

theorem log_product_equation (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 10) = 4 → x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equation_l2945_294532


namespace NUMINAMATH_CALUDE_intersection_point_l2945_294506

/-- The point of intersection of two lines in a 2D plane. -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- First line: y = 2x -/
def line1 (p : IntersectionPoint) : Prop := p.y = 2 * p.x

/-- Second line: x + y = 3 -/
def line2 (p : IntersectionPoint) : Prop := p.x + p.y = 3

/-- The intersection point of the two lines -/
def intersection : IntersectionPoint := ⟨1, 2⟩

/-- Theorem: The point (1, 2) is the unique intersection of the lines y = 2x and x + y = 3 -/
theorem intersection_point :
  line1 intersection ∧ line2 intersection ∧
  ∀ p : IntersectionPoint, line1 p ∧ line2 p → p = intersection :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l2945_294506


namespace NUMINAMATH_CALUDE_original_number_proof_l2945_294503

theorem original_number_proof (n k : ℕ) : 
  (n + k = 3200) → 
  (k ≥ 0) →
  (k < 8) →
  (3200 % 8 = 0) →
  ((n + k) % 8 = 0) →
  (∀ m : ℕ, m < k → (n + m) % 8 ≠ 0) →
  n = 3199 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2945_294503


namespace NUMINAMATH_CALUDE_incorrect_number_correction_l2945_294562

theorem incorrect_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_num : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 16)
  (h3 : incorrect_num = 25)
  (h4 : correct_avg = 17) :
  let correct_num := incorrect_num - (n * correct_avg - n * incorrect_avg)
  correct_num = 15 := by sorry

end NUMINAMATH_CALUDE_incorrect_number_correction_l2945_294562


namespace NUMINAMATH_CALUDE_abc_value_l2945_294535

theorem abc_value (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_prod_eq : b * c + c * a + a * b = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 10) : 
  a * b * c = 2 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2945_294535


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2945_294587

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 4 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_focal_property :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧  -- A and B are on the ellipse
  (∃ (t : ℝ), B = F2 + t • (A - F2)) ∧  -- A, B, and F2 are collinear
  ‖A - B‖ = 8 →  -- Distance between A and B is 8
  ‖A - F1‖ + ‖B - F1‖ = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2945_294587


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l2945_294541

def third_smallest_prime : ℕ := 5

theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l2945_294541


namespace NUMINAMATH_CALUDE_fraction_equations_l2945_294517

theorem fraction_equations : 
  (5 / 6 - 2 / 3 = 1 / 6) ∧
  (1 / 2 + 1 / 4 = 3 / 4) ∧
  (9 / 7 - 7 / 21 = 17 / 21) ∧
  (4 / 8 - 1 / 4 = 3 / 8) := by
sorry

end NUMINAMATH_CALUDE_fraction_equations_l2945_294517


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2945_294558

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (47 * π / 180) * Real.sin (103 * π / 180) +
  Real.sin (43 * π / 180) * Real.cos (77 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2945_294558


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_sum_l2945_294530

theorem floor_sqrt_sum_eq_floor_sqrt_sum (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_sum_l2945_294530


namespace NUMINAMATH_CALUDE_guitar_savings_l2945_294582

/-- The suggested retail price of the guitar -/
def suggested_price : ℝ := 1000

/-- The discount percentage offered by Guitar Center -/
def gc_discount : ℝ := 0.15

/-- The shipping fee charged by Guitar Center -/
def gc_shipping : ℝ := 100

/-- The discount percentage offered by Sweetwater -/
def sw_discount : ℝ := 0.10

/-- The cost of the guitar at Guitar Center -/
def gc_cost : ℝ := suggested_price * (1 - gc_discount) + gc_shipping

/-- The cost of the guitar at Sweetwater -/
def sw_cost : ℝ := suggested_price * (1 - sw_discount)

/-- The savings when buying from the cheaper store (Sweetwater) -/
theorem guitar_savings : gc_cost - sw_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_guitar_savings_l2945_294582


namespace NUMINAMATH_CALUDE_color_p_gon_l2945_294578

theorem color_p_gon (p a : ℕ) (hp : Nat.Prime p) :
  let total_colorings := a^p
  let monochromatic_colorings := a
  let distinct_non_monochromatic := (total_colorings - monochromatic_colorings) / p
  distinct_non_monochromatic + monochromatic_colorings = (a^p - a) / p + a := by
  sorry

end NUMINAMATH_CALUDE_color_p_gon_l2945_294578


namespace NUMINAMATH_CALUDE_fourth_derivative_of_f_l2945_294571

open Real

noncomputable def f (x : ℝ) : ℝ := exp (1 - 2*x) * sin (2 + 3*x)

theorem fourth_derivative_of_f (x : ℝ) :
  (deriv^[4] f) x = -119 * exp (1 - 2*x) * sin (2 + 3*x) + 120 * exp (1 - 2*x) * cos (2 + 3*x) :=
by sorry

end NUMINAMATH_CALUDE_fourth_derivative_of_f_l2945_294571


namespace NUMINAMATH_CALUDE_marble_transfer_result_l2945_294529

/-- Represents the marble transfer game between A and B -/
def marbleTransfer (a b n : ℕ) : Prop :=
  -- Initial conditions
  b < a ∧
  -- After 2n transfers, A has b marbles
  -- The ratio of initial marbles (a) to final marbles (b) is given by the formula
  (a : ℚ) / b = (2 * (4^n + 1)) / (1 - 4^n)

/-- Theorem stating the result of the marble transfer game -/
theorem marble_transfer_result {a b n : ℕ} (h : marbleTransfer a b n) :
  (a : ℚ) / b = (2 * (4^n + 1)) / (1 - 4^n) :=
by
  sorry

#check marble_transfer_result

end NUMINAMATH_CALUDE_marble_transfer_result_l2945_294529


namespace NUMINAMATH_CALUDE_nine_triangles_perimeter_l2945_294547

theorem nine_triangles_perimeter (large_perimeter : ℝ) (num_small_triangles : ℕ) 
  (h1 : large_perimeter = 120)
  (h2 : num_small_triangles = 9) :
  ∃ (small_perimeter : ℝ), 
    small_perimeter * num_small_triangles = large_perimeter ∧ 
    small_perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_nine_triangles_perimeter_l2945_294547


namespace NUMINAMATH_CALUDE_items_sold_increase_after_discount_l2945_294591

/-- Theorem: Increase in items sold after discount
  If a store offers a 10% discount on all items and their gross income increases by 3.5%,
  then the number of items sold increases by 15%.
-/
theorem items_sold_increase_after_discount (P N : ℝ) (N' : ℝ) :
  P > 0 → N > 0 →
  (0.9 * P * N' = 1.035 * P * N) →
  (N' - N) / N * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_items_sold_increase_after_discount_l2945_294591


namespace NUMINAMATH_CALUDE_sin_315_degrees_l2945_294514

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l2945_294514


namespace NUMINAMATH_CALUDE_primitive_pythagorean_triple_parity_l2945_294556

theorem primitive_pythagorean_triple_parity (a b c : ℕ+) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : Nat.gcd a.val (Nat.gcd b.val c.val) = 1) :
  (Even a.val ∧ Odd b.val) ∨ (Odd a.val ∧ Even b.val) := by
sorry

end NUMINAMATH_CALUDE_primitive_pythagorean_triple_parity_l2945_294556


namespace NUMINAMATH_CALUDE_students_behind_yoongi_l2945_294575

/-- Given a line of students, calculates the number of students behind a specific student -/
def studentsInBack (totalStudents : ℕ) (studentsBetween : ℕ) : ℕ :=
  totalStudents - (studentsBetween + 2)

theorem students_behind_yoongi :
  let totalStudents : ℕ := 20
  let studentsBetween : ℕ := 5
  studentsInBack totalStudents studentsBetween = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_l2945_294575


namespace NUMINAMATH_CALUDE_light_glow_theorem_l2945_294597

/-- The number of times a light glows in a given time interval -/
def glowCount (interval : ℕ) (period : ℕ) : ℕ :=
  interval / period

/-- The number of times all lights glow simultaneously in a given time interval -/
def simultaneousGlowCount (interval : ℕ) (periodA periodB periodC : ℕ) : ℕ :=
  interval / (lcm (lcm periodA periodB) periodC)

theorem light_glow_theorem (totalInterval : ℕ) (periodA periodB periodC : ℕ)
    (h1 : totalInterval = 4969)
    (h2 : periodA = 18)
    (h3 : periodB = 24)
    (h4 : periodC = 30) :
    glowCount totalInterval periodA = 276 ∧
    glowCount totalInterval periodB = 207 ∧
    glowCount totalInterval periodC = 165 ∧
    simultaneousGlowCount totalInterval periodA periodB periodC = 13 := by
  sorry

end NUMINAMATH_CALUDE_light_glow_theorem_l2945_294597


namespace NUMINAMATH_CALUDE_buyer_count_solution_l2945_294592

/-- The number of buyers in a grocery store over three days -/
structure BuyerCount where
  dayBeforeYesterday : ℕ
  yesterday : ℕ
  today : ℕ

/-- Conditions for the buyer count problem -/
def BuyerCountProblem (b : BuyerCount) : Prop :=
  b.today = b.yesterday + 40 ∧
  b.yesterday = b.dayBeforeYesterday / 2 ∧
  b.dayBeforeYesterday + b.yesterday + b.today = 140

theorem buyer_count_solution :
  ∃ b : BuyerCount, BuyerCountProblem b ∧ b.dayBeforeYesterday = 67 := by
  sorry

end NUMINAMATH_CALUDE_buyer_count_solution_l2945_294592


namespace NUMINAMATH_CALUDE_cos_300_degrees_l2945_294543

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l2945_294543


namespace NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l2945_294551

theorem opposite_solutions_value_of_m : ∀ (x y m : ℝ),
  (3 * x + 5 * y = 2) →
  (2 * x + 7 * y = m - 18) →
  (x = -y) →
  m = 23 := by
  sorry

end NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l2945_294551


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2945_294542

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 4 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ min)) ∧
  (1/a + 1/b = 4 ↔ a = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2945_294542


namespace NUMINAMATH_CALUDE_exactly_eighteen_pairs_l2945_294521

/-- Predicate to check if a pair of natural numbers satisfies the given conditions -/
def satisfies_conditions (a b : ℕ) : Prop :=
  (b ∣ (5 * a - 3)) ∧ (a ∣ (5 * b - 1))

/-- The number of pairs of natural numbers satisfying the conditions -/
def number_of_pairs : ℕ := 18

/-- Theorem stating that there are exactly 18 pairs satisfying the conditions -/
theorem exactly_eighteen_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = number_of_pairs ∧
    ∀ (pair : ℕ × ℕ), pair ∈ s ↔ satisfies_conditions pair.1 pair.2) :=
sorry

end NUMINAMATH_CALUDE_exactly_eighteen_pairs_l2945_294521


namespace NUMINAMATH_CALUDE_y_range_for_x_condition_l2945_294538

theorem y_range_for_x_condition (x y : ℝ) : 
  (4 * x + y = 1) → ((-1 < x ∧ x ≤ 2) ↔ (-7 ≤ y ∧ y < -3)) := by
  sorry

end NUMINAMATH_CALUDE_y_range_for_x_condition_l2945_294538


namespace NUMINAMATH_CALUDE_incorrect_proposition_l2945_294510

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem incorrect_proposition
  (m n : Line) (α β : Plane)
  (h1 : parallel m α)
  (h2 : perpendicular n β)
  (h3 : perpendicular_planes α β) :
  ¬ (parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l2945_294510


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_number_l2945_294553

theorem smallest_consecutive_even_number (n : ℕ) : 
  (n % 2 = 0) →  -- n is even
  (n + (n + 2) + (n + 4) = 162) →  -- sum of three consecutive even numbers is 162
  n = 52 :=  -- the smallest number is 52
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_number_l2945_294553


namespace NUMINAMATH_CALUDE_inequality_proof_l2945_294549

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * (x - y - 1) + 1 / (x^2 - 2*x*y + y^2) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2945_294549


namespace NUMINAMATH_CALUDE_rational_cosine_summands_l2945_294540

theorem rational_cosine_summands (x : ℝ) 
  (h_S : ∃ q : ℚ, q = Real.sin (64 * x) + Real.sin (65 * x))
  (h_C : ∃ q : ℚ, q = Real.cos (64 * x) + Real.cos (65 * x)) :
  ∃ (q1 q2 : ℚ), q1 = Real.cos (64 * x) ∧ q2 = Real.cos (65 * x) :=
sorry

end NUMINAMATH_CALUDE_rational_cosine_summands_l2945_294540


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l2945_294580

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the larger base -/
  largerBase : ℝ
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The left segment of the larger base divided by the height -/
  leftSegment : ℝ
  /-- The right segment of the larger base divided by the height -/
  rightSegment : ℝ
  /-- The larger base is positive -/
  largerBase_pos : 0 < largerBase
  /-- The smaller base is positive -/
  smallerBase_pos : 0 < smallerBase
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The sum of segments equals the larger base -/
  segment_sum : leftSegment + rightSegment = largerBase
  /-- The ratio of segments is 2:3 -/
  segment_ratio : leftSegment / rightSegment = 2 / 3

/-- 
If the height of an isosceles trapezoid divides the larger base into segments 
with a ratio of 2:3, then the ratio of the larger base to the smaller base is 5:1
-/
theorem isosceles_trapezoid_base_ratio (t : IsoscelesTrapezoid) : 
  t.largerBase / t.smallerBase = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l2945_294580


namespace NUMINAMATH_CALUDE_negation_equivalence_l2945_294599

theorem negation_equivalence :
  (¬ ∃ (x y : ℝ), 2*x + 3*y + 3 < 0) ↔ (∀ (x y : ℝ), 2*x + 3*y + 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2945_294599


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l2945_294560

theorem field_trip_girls_fraction (total_students : ℕ) (h_positive : total_students > 0) :
  let girls : ℚ := total_students / 2
  let boys : ℚ := total_students / 2
  let girls_on_trip : ℚ := (4 / 5) * girls
  let boys_on_trip : ℚ := (3 / 4) * boys
  let total_on_trip : ℚ := girls_on_trip + boys_on_trip
  (girls_on_trip / total_on_trip) = 16 / 31 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l2945_294560


namespace NUMINAMATH_CALUDE_six_to_six_sum_l2945_294534

theorem six_to_six_sum : (6^6 : ℕ) + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 := by
  sorry

end NUMINAMATH_CALUDE_six_to_six_sum_l2945_294534


namespace NUMINAMATH_CALUDE_ellipse_properties_l2945_294525

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop :=
  c = 2 * Real.sqrt 3

-- Define the intersection points with y-axis
def y_intersections (b : ℝ) : Prop :=
  b = 1

-- Define the standard form of the ellipse
def standard_form (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 3 / 2

-- Define the range of x-coordinate for point P
def x_range (x : ℝ) : Prop :=
  24 / 13 < x ∧ x ≤ 2

-- Define the maximum value of |EF|
def max_ef (ef : ℝ) : Prop :=
  ef = 1

theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  focal_distance c ∧
  y_intersections b →
  (∃ x y, ellipse x y a b ∧ standard_form x y) ∧
  (∃ e, eccentricity e) ∧
  (∃ x, x_range x) ∧
  (∃ ef, max_ef ef) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2945_294525


namespace NUMINAMATH_CALUDE_translated_point_sum_zero_l2945_294501

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate (p : Point) (dx dy : ℝ) : Point :=
  (p.1 + dx, p.2 + dy)

theorem translated_point_sum_zero :
  let A : Point := (-1, 2)
  let B : Point := translate (translate A 1 0) 0 (-2)
  B.1 + B.2 = 0 := by sorry

end NUMINAMATH_CALUDE_translated_point_sum_zero_l2945_294501


namespace NUMINAMATH_CALUDE_fence_coloring_theorem_l2945_294579

/-- A coloring of a fence is valid if any two boards separated by exactly 2, 3, or 5 boards
    are painted in different colors. -/
def is_valid_coloring (coloring : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, (coloring i ≠ coloring (i + 3)) ∧
           (coloring i ≠ coloring (i + 4)) ∧
           (coloring i ≠ coloring (i + 6))

/-- The minimum number of colors required to paint the fence -/
def min_colors : ℕ := 3

theorem fence_coloring_theorem :
  (∃ coloring : ℕ → ℕ, is_valid_coloring coloring ∧ (∀ i : ℕ, coloring i < min_colors)) ∧
  (∀ n : ℕ, n < min_colors → ¬∃ coloring : ℕ → ℕ, is_valid_coloring coloring ∧ (∀ i : ℕ, coloring i < n)) :=
sorry

end NUMINAMATH_CALUDE_fence_coloring_theorem_l2945_294579


namespace NUMINAMATH_CALUDE_symmetric_circle_l2945_294516

/-- Given a circle C1 with equation (x+2)^2+(y-1)^2=5,
    prove that its symmetric circle C2 with respect to the origin (0,0)
    has the equation (x-2)^2+(y+1)^2=5 -/
theorem symmetric_circle (x y : ℝ) :
  (∀ x y, (x + 2)^2 + (y - 1)^2 = 5) →
  (∃ C2 : Set (ℝ × ℝ), C2 = {(x, y) | (x - 2)^2 + (y + 1)^2 = 5} ∧
    ∀ (p : ℝ × ℝ), p ∈ C2 ↔ (-p.1, -p.2) ∈ {(x, y) | (x + 2)^2 + (y - 1)^2 = 5}) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l2945_294516


namespace NUMINAMATH_CALUDE_farmer_loss_l2945_294584

/-- Represents the total weight of onions in pounds -/
def total_weight : ℝ := 100

/-- Represents the market price per pound of onions in dollars -/
def market_price : ℝ := 3

/-- Represents the dealer's price per pound for both leaves and whites in dollars -/
def dealer_price : ℝ := 1.5

/-- Theorem stating the farmer's loss -/
theorem farmer_loss : 
  total_weight * market_price - total_weight * dealer_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_farmer_loss_l2945_294584


namespace NUMINAMATH_CALUDE_system_solution_l2945_294581

theorem system_solution :
  ∃ (x y : ℚ),
    (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧
    (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
    (x = -3/4) ∧ (y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2945_294581


namespace NUMINAMATH_CALUDE_line_slope_from_parametric_equation_l2945_294550

/-- Given a line l with parametric equations x = 1 - (3/5)t and y = (4/5)t,
    prove that the slope of the line is -4/3 -/
theorem line_slope_from_parametric_equation :
  ∀ (l : ℝ → ℝ × ℝ),
  (∀ t, l t = (1 - 3/5 * t, 4/5 * t)) →
  (∃ m b, ∀ x y, (x, y) ∈ Set.range l → y = m * x + b) →
  (∃ m b, ∀ x y, (x, y) ∈ Set.range l → y = m * x + b ∧ m = -4/3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_from_parametric_equation_l2945_294550


namespace NUMINAMATH_CALUDE_negative_abs_of_negative_one_l2945_294566

theorem negative_abs_of_negative_one : -|-1| = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_of_negative_one_l2945_294566


namespace NUMINAMATH_CALUDE_tied_rope_length_l2945_294548

/-- Calculates the length of a rope made by tying multiple shorter ropes together. -/
def ropeLength (n : ℕ) (ropeLength : ℕ) (knotReduction : ℕ) : ℕ :=
  n * ropeLength - (n - 1) * knotReduction

/-- Proves that tying 64 ropes of 25 cm each, with 3 cm reduction per knot, results in a 1411 cm rope. -/
theorem tied_rope_length :
  ropeLength 64 25 3 = 1411 := by
  sorry

end NUMINAMATH_CALUDE_tied_rope_length_l2945_294548


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2945_294589

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25             -- Shorter leg length
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2945_294589


namespace NUMINAMATH_CALUDE_independence_and_polynomial_value_l2945_294512

/-- The algebraic expression is independent of x -/
def is_independent_of_x (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (2 - 2*b) * x^2 + (a + 3) * x - 6*y + 7 = -6*y + 7

/-- The value of the polynomial given a and b -/
def polynomial_value (a b : ℝ) : ℝ :=
  3*(a^2 - 2*a*b - b^2) - (4*a^2 + a*b + b^2)

theorem independence_and_polynomial_value :
  ∃ a b : ℝ, is_independent_of_x a b ∧ a = -3 ∧ b = 1 ∧ polynomial_value a b = 8 := by
  sorry

end NUMINAMATH_CALUDE_independence_and_polynomial_value_l2945_294512


namespace NUMINAMATH_CALUDE_miss_one_out_of_three_l2945_294507

def free_throw_probability : ℝ := 0.9

theorem miss_one_out_of_three (p : ℝ) (hp : p = free_throw_probability) :
  p * p * (1 - p) + p * (1 - p) * p + (1 - p) * p * p = 0.243 := by
  sorry

end NUMINAMATH_CALUDE_miss_one_out_of_three_l2945_294507


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2945_294563

/-- The constant term in the expansion of (x + 1/x)^4 -/
def constant_term : ℕ := 6

/-- Represents a geometric sequence -/
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n * a m

theorem geometric_sequence_product
  (a : ℕ → ℕ)
  (h_geo : geometric_sequence a)
  (h_a5 : a 5 = constant_term) :
  a 3 * a 7 = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2945_294563


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l2945_294586

theorem sufficient_condition_implies_a_range :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) →
  a ∈ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l2945_294586


namespace NUMINAMATH_CALUDE_regular_15gon_symmetry_sum_l2945_294546

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry (in degrees) for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

/-- Theorem: For a regular 15-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 39 -/
theorem regular_15gon_symmetry_sum :
  ∀ (p : RegularPolygon 15),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 39 := by
  sorry

end NUMINAMATH_CALUDE_regular_15gon_symmetry_sum_l2945_294546


namespace NUMINAMATH_CALUDE_equation_solution_l2945_294508

theorem equation_solution :
  ∃ x : ℝ, (4 / 7) * (1 / 9) * x = 14 ∧ x = 220.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2945_294508


namespace NUMINAMATH_CALUDE_min_value_abc_l2945_294569

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c = 4 * (a + b)) : 
  a + b + c ≥ 8 ∧ (a + b + c = 8 ↔ a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l2945_294569


namespace NUMINAMATH_CALUDE_evaluate_expression_l2945_294528

theorem evaluate_expression : (3^10 + 3^7) / (3^10 - 3^7) = 14/13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2945_294528


namespace NUMINAMATH_CALUDE_problem_solution_l2945_294567

theorem problem_solution (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2*y = 10) : x = 26/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2945_294567


namespace NUMINAMATH_CALUDE_bus_problem_l2945_294539

/-- Calculates the final number of people on a bus given initial count and changes -/
def final_bus_count (initial : ℕ) (getting_on : ℕ) (getting_off : ℕ) : ℕ :=
  initial + getting_on - getting_off

/-- Theorem stating that given 22 initial people, 4 getting on, and 8 getting off, 
    the final count is 18 -/
theorem bus_problem : final_bus_count 22 4 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2945_294539


namespace NUMINAMATH_CALUDE_simplify_expressions_l2945_294552

open Real

theorem simplify_expressions (θ : ℝ) :
  (sqrt (1 - 2 * sin (135 * π / 180) * cos (135 * π / 180))) / 
  (sin (135 * π / 180) + sqrt (1 - sin (135 * π / 180) ^ 2)) = 1 ∧
  (sin (θ - 5 * π) * cos (-π / 2 - θ) * cos (8 * π - θ)) / 
  (sin (θ - 3 * π / 2) * sin (-θ - 4 * π)) = -sin (θ - 5 * π) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2945_294552


namespace NUMINAMATH_CALUDE_coreys_weekend_goal_l2945_294561

/-- Corey's goal for the number of golf balls to find every weekend -/
def coreys_goal (saturday_balls sunday_balls remaining_balls : ℕ) : ℕ :=
  saturday_balls + sunday_balls + remaining_balls

/-- Theorem stating Corey's goal for the number of golf balls to find every weekend -/
theorem coreys_weekend_goal :
  coreys_goal 16 18 14 = 48 := by
  sorry

end NUMINAMATH_CALUDE_coreys_weekend_goal_l2945_294561


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2945_294504

theorem system_of_equations_solution (a b c d e f g : ℚ) : 
  a + b + c + d + e = 1 →
  b + c + d + e + f = 2 →
  c + d + e + f + g = 3 →
  d + e + f + g + a = 4 →
  e + f + g + a + b = 5 →
  f + g + a + b + c = 6 →
  g + a + b + c + d = 7 →
  g = 13/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2945_294504


namespace NUMINAMATH_CALUDE_paper_fold_sum_l2945_294568

-- Define the fold line
def fold_line (x y : ℝ) : Prop := y = x

-- Define the mapping of points
def maps_to (x1 y1 x2 y2 : ℝ) : Prop :=
  fold_line ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  (y2 - y1) = -(x2 - x1)

-- Main theorem
theorem paper_fold_sum (m n : ℝ) :
  maps_to 0 5 5 0 →  -- (0,5) maps to (5,0)
  maps_to 8 4 m n →  -- (8,4) maps to (m,n)
  m + n = 12 := by
sorry

end NUMINAMATH_CALUDE_paper_fold_sum_l2945_294568


namespace NUMINAMATH_CALUDE_fourth_term_is_54_l2945_294598

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  is_positive : ∀ n, a n > 0
  is_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q
  first_term : a 1 = 2
  arithmetic_mean : a 2 + 4 = (a 1 + a 3) / 2

/-- The fourth term of the special geometric sequence is 54 -/
theorem fourth_term_is_54 (seq : SpecialGeometricSequence) : seq.a 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_54_l2945_294598


namespace NUMINAMATH_CALUDE_function_increasing_l2945_294544

theorem function_increasing (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁) : 
  StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_l2945_294544


namespace NUMINAMATH_CALUDE_jonessas_take_home_pay_l2945_294576

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jonessa's take-home pay is $450 -/
theorem jonessas_take_home_pay :
  let totalPay : ℝ := 500
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 450 := by
sorry

end NUMINAMATH_CALUDE_jonessas_take_home_pay_l2945_294576


namespace NUMINAMATH_CALUDE_noahs_sales_ratio_l2945_294554

/-- Noah's painting sales problem -/
theorem noahs_sales_ratio :
  let large_price : ℕ := 60
  let small_price : ℕ := 30
  let last_month_large : ℕ := 8
  let last_month_small : ℕ := 4
  let this_month_sales : ℕ := 1200
  let last_month_sales : ℕ := large_price * last_month_large + small_price * last_month_small
  (this_month_sales : ℚ) / (last_month_sales : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_noahs_sales_ratio_l2945_294554


namespace NUMINAMATH_CALUDE_expand_product_l2945_294518

theorem expand_product (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = 3 / y - 6 * y^3 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2945_294518


namespace NUMINAMATH_CALUDE_group_size_l2945_294526

/-- An international group consisting of Chinese, Americans, and Australians -/
structure InternationalGroup where
  chinese : ℕ
  americans : ℕ
  australians : ℕ

/-- The total number of people in the group -/
def InternationalGroup.total (group : InternationalGroup) : ℕ :=
  group.chinese + group.americans + group.australians

theorem group_size (group : InternationalGroup) 
  (h1 : group.chinese = 22)
  (h2 : group.americans = 16)
  (h3 : group.australians = 11) :
  group.total = 49 := by
  sorry

#check group_size

end NUMINAMATH_CALUDE_group_size_l2945_294526


namespace NUMINAMATH_CALUDE_shopping_mall_pricing_l2945_294545

/-- Shopping mall pricing problem -/
theorem shopping_mall_pricing
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_monthly_sales : ℝ)
  (sales_increase_rate : ℝ)
  (target_monthly_profit : ℝ)
  (h1 : purchase_price = 280)
  (h2 : initial_selling_price = 360)
  (h3 : initial_monthly_sales = 60)
  (h4 : sales_increase_rate = 5)
  (h5 : target_monthly_profit = 7200) :
  ∃ (price_reduction : ℝ),
    price_reduction = 60 ∧
    (initial_selling_price - price_reduction - purchase_price) *
    (initial_monthly_sales + sales_increase_rate * price_reduction) =
    target_monthly_profit :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_pricing_l2945_294545


namespace NUMINAMATH_CALUDE_magnitude_of_b_l2945_294527

def vector_a : ℝ × ℝ := (1, -2)
def vector_sum : ℝ × ℝ := (0, 2)

def vector_b : ℝ × ℝ := (vector_sum.1 - vector_a.1, vector_sum.2 - vector_a.2)

theorem magnitude_of_b : Real.sqrt ((vector_b.1)^2 + (vector_b.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l2945_294527


namespace NUMINAMATH_CALUDE_unicorn_to_witch_ratio_l2945_294519

/-- Represents the number of votes for each cake type -/
structure CakeVotes where
  unicorn : ℕ
  witch : ℕ
  dragon : ℕ

/-- The conditions of the baking contest voting -/
def baking_contest (votes : CakeVotes) : Prop :=
  votes.dragon = votes.witch + 25 ∧
  votes.witch = 7 ∧
  votes.unicorn + votes.witch + votes.dragon = 60

theorem unicorn_to_witch_ratio (votes : CakeVotes) 
  (h : baking_contest votes) : 
  votes.unicorn / votes.witch = 3 :=
sorry

end NUMINAMATH_CALUDE_unicorn_to_witch_ratio_l2945_294519


namespace NUMINAMATH_CALUDE_mildred_blocks_l2945_294596

/-- The number of blocks Mildred ends up with -/
def total_blocks (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Mildred's total blocks is the sum of initial and found blocks -/
theorem mildred_blocks (initial : ℕ) (found : ℕ) :
  total_blocks initial found = initial + found := by
  sorry

end NUMINAMATH_CALUDE_mildred_blocks_l2945_294596


namespace NUMINAMATH_CALUDE_blood_expires_same_day_l2945_294523

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The factorial of 8 -/
def blood_expiration_seconds : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- The day a unit of blood expires when donated at noon -/
def blood_expiration_day (donation_day : ℕ) : ℕ :=
  donation_day + (blood_expiration_seconds / seconds_per_day)

theorem blood_expires_same_day (donation_day : ℕ) :
  blood_expiration_day donation_day = donation_day := by
  sorry

end NUMINAMATH_CALUDE_blood_expires_same_day_l2945_294523


namespace NUMINAMATH_CALUDE_elise_remaining_money_l2945_294565

/-- Calculates the remaining money in dollars for Elise --/
def remaining_money (initial_amount : ℝ) (saved_euros : ℝ) (euro_to_dollar : ℝ) 
                    (comic_cost : ℝ) (puzzle_cost_pounds : ℝ) (pound_to_dollar : ℝ) : ℝ :=
  initial_amount + saved_euros * euro_to_dollar - comic_cost - puzzle_cost_pounds * pound_to_dollar

/-- Theorem stating that Elise's remaining money is $1.04 --/
theorem elise_remaining_money :
  remaining_money 8 11 1.18 2 13 1.38 = 1.04 := by
  sorry

end NUMINAMATH_CALUDE_elise_remaining_money_l2945_294565


namespace NUMINAMATH_CALUDE_prob_hit_135_prob_hit_exactly_3_l2945_294511

-- Define the probability of hitting the target
def hit_probability : ℚ := 3 / 5

-- Define the number of shots
def num_shots : ℕ := 5

-- Theorem for the first part
theorem prob_hit_135 : 
  (hit_probability * (1 - hit_probability) * hit_probability * (1 - hit_probability) * hit_probability) = 108 / 3125 := by
  sorry

-- Theorem for the second part
theorem prob_hit_exactly_3 :
  (Nat.choose num_shots 3 : ℚ) * hit_probability ^ 3 * (1 - hit_probability) ^ 2 = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_prob_hit_135_prob_hit_exactly_3_l2945_294511


namespace NUMINAMATH_CALUDE_multiply_25_26_8_multiply_divide_340_40_17_sum_products_15_l2945_294502

-- Part 1
theorem multiply_25_26_8 : 25 * 26 * 8 = 5200 := by sorry

-- Part 2
theorem multiply_divide_340_40_17 : 340 * 40 / 17 = 800 := by sorry

-- Part 3
theorem sum_products_15 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := by sorry

end NUMINAMATH_CALUDE_multiply_25_26_8_multiply_divide_340_40_17_sum_products_15_l2945_294502


namespace NUMINAMATH_CALUDE_car_sale_profit_percentage_l2945_294513

theorem car_sale_profit_percentage (P : ℝ) : 
  let buying_price := 0.80 * P
  let selling_price := 1.16 * P
  ((selling_price - buying_price) / buying_price) * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_car_sale_profit_percentage_l2945_294513


namespace NUMINAMATH_CALUDE_inequality_range_l2945_294570

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 1 - 2 * x^2) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2945_294570


namespace NUMINAMATH_CALUDE_permutations_theorem_l2945_294524

def alphabet_size : ℕ := 26

def excluded_words : List String := ["dog", "god", "gum", "depth", "thing"]

def permutations_without_substrings (n : ℕ) (words : List String) : ℕ :=
  n.factorial - 3 * (n - 2).factorial + 3 * (n - 6).factorial + 2 * (n - 7).factorial - (n - 9).factorial

theorem permutations_theorem :
  permutations_without_substrings alphabet_size excluded_words =
  alphabet_size.factorial - 3 * (alphabet_size - 2).factorial + 3 * (alphabet_size - 6).factorial +
  2 * (alphabet_size - 7).factorial - (alphabet_size - 9).factorial :=
by sorry

end NUMINAMATH_CALUDE_permutations_theorem_l2945_294524


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2945_294557

/-- The x-intercept of a line is the point where the line crosses the x-axis (i.e., where y = 0) -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  let x := c / a
  (x, 0)

/-- The line equation is in the form ax + by = c -/
def line_equation (a b c : ℚ) (x y : ℚ) : Prop :=
  a * x + b * y = c

theorem x_intercept_of_line :
  x_intercept 5 (-7) 35 = (7, 0) ∧
  line_equation 5 (-7) 35 (x_intercept 5 (-7) 35).1 (x_intercept 5 (-7) 35).2 :=
sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2945_294557


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2945_294573

theorem complex_fraction_equality : (3 - I) / (1 - I) = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2945_294573


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l2945_294559

/-- Proves that for the equation x^2 + y^2 - 2x + 2k + 3 = 0 to represent a circle,
    k must be in the range (-∞, -1). -/
theorem circle_equation_k_range :
  ∀ (k : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*x + 2*k + 3 = 0 ∧ 
    ∃ (h r : ℝ), ∀ (x' y' : ℝ), (x' - h)^2 + (y' - r)^2 = (x - h)^2 + (y - r)^2) 
  ↔ k < -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l2945_294559


namespace NUMINAMATH_CALUDE_merchant_discount_percentage_l2945_294593

/-- Calculates the discount percentage for a merchant's pricing strategy -/
theorem merchant_discount_percentage
  (markup_percentage : ℝ)
  (profit_percentage : ℝ)
  (h_markup : markup_percentage = 50)
  (h_profit : profit_percentage = 35)
  : ∃ (discount_percentage : ℝ),
    discount_percentage = 10 ∧
    (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = 1 + profit_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_merchant_discount_percentage_l2945_294593


namespace NUMINAMATH_CALUDE_inequality_solution_l2945_294572

theorem inequality_solution (n : Int) :
  n ∈ ({-1, 0, 1, 2, 3} : Set Int) →
  ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n) ↔ (n = -1 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2945_294572


namespace NUMINAMATH_CALUDE_flagpole_height_l2945_294537

/-- Given a flagpole that breaks and folds over in half, with its tip 2 feet above the ground
    and the break point 7 feet from the base, prove that its original height was 16 feet. -/
theorem flagpole_height (H : ℝ) : 
  (H - 7 - 2 = 7) →  -- The folded part equals the standing part
  (H = 16) :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l2945_294537


namespace NUMINAMATH_CALUDE_inequality_implies_range_l2945_294583

theorem inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, a * Real.sin x - Real.cos x ^ 2 ≤ 3) →
  -3 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l2945_294583


namespace NUMINAMATH_CALUDE_square_difference_area_l2945_294505

theorem square_difference_area (a b : ℝ) (h : a > b) :
  (a ^ 2 - b ^ 2 : ℝ) = (Real.sqrt (a ^ 2 - b ^ 2)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_area_l2945_294505


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2945_294594

def tank_capacity (oil_bought : ℕ) (oil_in_tank : ℕ) : ℕ :=
  oil_bought + oil_in_tank

theorem tank_capacity_proof (oil_bought : ℕ) (oil_in_tank : ℕ) 
  (h1 : oil_bought = 728) (h2 : oil_in_tank = 24) : 
  tank_capacity oil_bought oil_in_tank = 752 := by
  sorry

#check tank_capacity_proof

end NUMINAMATH_CALUDE_tank_capacity_proof_l2945_294594
