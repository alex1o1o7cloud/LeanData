import Mathlib

namespace NUMINAMATH_CALUDE_flashlight_problem_l899_89979

/-- Represents the minimum number of attempts needed to guarantee a flashlight lights up -/
def min_attempts (total_batteries : ℕ) (good_batteries : ℕ) : ℕ :=
  if total_batteries = 2 * good_batteries - 1 
  then good_batteries + 1
  else if total_batteries = 2 * good_batteries 
  then good_batteries + 3
  else 0  -- undefined for other cases

/-- Theorem for the flashlight problem -/
theorem flashlight_problem (n : ℕ) (h : n > 2) :
  (min_attempts (2 * n + 1) (n + 1) = n + 2) ∧
  (min_attempts (2 * n) n = n + 3) := by
  sorry

#check flashlight_problem

end NUMINAMATH_CALUDE_flashlight_problem_l899_89979


namespace NUMINAMATH_CALUDE_tangent_line_problem_l899_89970

/-- Given a curve y = x^3 + ax + b and a line y = kx + 1 that is tangent to this curve at the point (1, 3), 
    the value of a - b is equal to -4. -/
theorem tangent_line_problem (a b k : ℝ) : 
  (∀ x, x^3 + a*x + b = k*x + 1 → x = 1) →  -- The line is tangent to the curve
  3^3 + a*3 + b = k*3 + 1 →                 -- The point (1, 3) lies on the curve
  3 = k*1 + 1 →                             -- The point (1, 3) lies on the line
  a - b = -4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l899_89970


namespace NUMINAMATH_CALUDE_bipin_chandan_age_ratio_l899_89933

/-- Proves that the ratio of Bipin's age to Chandan's age after 10 years is 2:1 -/
theorem bipin_chandan_age_ratio :
  let alok_age : ℕ := 5
  let bipin_age : ℕ := 6 * alok_age
  let chandan_age : ℕ := 7 + 3
  let bipin_future_age : ℕ := bipin_age + 10
  let chandan_future_age : ℕ := chandan_age + 10
  (bipin_future_age : ℚ) / chandan_future_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_bipin_chandan_age_ratio_l899_89933


namespace NUMINAMATH_CALUDE_remaining_cards_l899_89990

-- Define the initial number of baseball cards Mike has
def initial_cards : ℕ := 87

-- Define the number of cards Sam bought
def bought_cards : ℕ := 13

-- Theorem stating that Mike's remaining cards is the difference between initial and bought
theorem remaining_cards : initial_cards - bought_cards = 74 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cards_l899_89990


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l899_89964

/-- Given a line passing through points (1, -7) and (k, 19) that is parallel to 3x + 4y = 12, prove that k = -101/3 -/
theorem parallel_line_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (m * 1 + b = -7) ∧ (m * k + b = 19) ∧ (m = -3/4)) → 
  k = -101/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l899_89964


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l899_89962

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age : ℕ) (final_people : ℕ) (final_average : ℚ) : 
  initial_people = 7 →
  initial_average = 28 →
  leaving_age = 20 →
  final_people = 6 →
  final_average = 29 →
  (initial_people : ℚ) * initial_average - leaving_age = final_people * final_average := by
  sorry

#check average_age_after_leaving

end NUMINAMATH_CALUDE_average_age_after_leaving_l899_89962


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l899_89957

theorem min_value_quadratic_form (x y : ℤ) (h : (x, y) ≠ (0, 0)) :
  |5 * x^2 + 11 * x * y - 5 * y^2| ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l899_89957


namespace NUMINAMATH_CALUDE_snail_distance_is_29_l899_89967

def snail_path : List ℤ := [3, -5, 8, 0]

def distance (a b : ℤ) : ℤ := Int.natAbs (b - a)

def total_distance (path : List ℤ) : ℤ :=
  (List.zip path path.tail).foldl (fun acc (a, b) => acc + distance a b) 0

theorem snail_distance_is_29 : total_distance snail_path = 29 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_is_29_l899_89967


namespace NUMINAMATH_CALUDE_robin_extra_drinks_l899_89971

/-- Calculates the number of extra drinks given the quantities bought and consumed --/
def extra_drinks (sodas_bought energy_bought smoothies_bought
                  sodas_drunk energy_drunk smoothies_drunk : ℕ) : ℕ :=
  (sodas_bought + energy_bought + smoothies_bought) -
  (sodas_drunk + energy_drunk + smoothies_drunk)

/-- Theorem stating that Robin has 32 extra drinks --/
theorem robin_extra_drinks :
  extra_drinks 22 15 12 6 9 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_robin_extra_drinks_l899_89971


namespace NUMINAMATH_CALUDE_cube_root_simplification_l899_89921

theorem cube_root_simplification :
  (72^3 + 108^3 + 144^3 : ℝ)^(1/3) = 36 * 99^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l899_89921


namespace NUMINAMATH_CALUDE_parallelogram_area_l899_89985

/-- Parallelogram EFGH with given side lengths and diagonal -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  EH : ℝ
  is_parallelogram : EF > 0 ∧ FG > 0 ∧ EH > 0

/-- The area of the parallelogram EFGH -/
def area (p : Parallelogram) : ℝ :=
  p.EF * p.FG

/-- Theorem: The area of parallelogram EFGH is 1200 -/
theorem parallelogram_area (p : Parallelogram) 
  (h1 : p.EF = 40) 
  (h2 : p.FG = 30) 
  (h3 : p.EH = 50) : 
  area p = 1200 := by
  sorry

#check parallelogram_area

end NUMINAMATH_CALUDE_parallelogram_area_l899_89985


namespace NUMINAMATH_CALUDE_total_food_eaten_l899_89996

/-- The amount of food Ella's dog eats relative to Ella -/
def dog_food_ratio : ℕ := 4

/-- The number of days -/
def days : ℕ := 10

/-- The amount of food Ella eats per day (in pounds) -/
def ella_food_per_day : ℕ := 20

/-- The total amount of food eaten by Ella and her dog (in pounds) -/
def total_food : ℕ := days * ella_food_per_day * (1 + dog_food_ratio)

theorem total_food_eaten :
  total_food = 1000 := by sorry

end NUMINAMATH_CALUDE_total_food_eaten_l899_89996


namespace NUMINAMATH_CALUDE_largest_b_value_l899_89984

theorem largest_b_value (b : ℚ) (h : (3 * b + 7) * (b - 2) = 8 * b) : 
  b ≤ 7 / 2 ∧ ∃ (b₀ : ℚ), (3 * b₀ + 7) * (b₀ - 2) = 8 * b₀ ∧ b₀ = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_b_value_l899_89984


namespace NUMINAMATH_CALUDE_main_theorem_l899_89974

/-- Proposition p -/
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0

/-- Proposition q -/
def q (x m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

/-- Set A: values of x that satisfy p -/
def A : Set ℝ := {x | p x}

/-- Set B: values of x that satisfy q -/
def B (m : ℝ) : Set ℝ := {x | q x m}

/-- Main theorem: If ¬q is a sufficient but not necessary condition for ¬p,
    then m > 1 or m < -2 -/
theorem main_theorem (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, p x ∧ q x m) →
  m > 1 ∨ m < -2 :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l899_89974


namespace NUMINAMATH_CALUDE_not_all_equilateral_triangles_have_same_perimeter_l899_89912

-- Define an equilateral triangle
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Properties of equilateral triangles
def EquilateralTriangle.isEquiangular (t : EquilateralTriangle) : Prop :=
  -- All angles are 60 degrees
  true

def EquilateralTriangle.isIsosceles (t : EquilateralTriangle) : Prop :=
  -- At least two sides are equal (all sides are equal in this case)
  true

def EquilateralTriangle.isRegularPolygon (t : EquilateralTriangle) : Prop :=
  -- All sides equal and all angles equal
  true

def EquilateralTriangle.isSimilarTo (t1 t2 : EquilateralTriangle) : Prop :=
  -- All equilateral triangles are similar
  true

def EquilateralTriangle.perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.sideLength

-- Theorem to prove
theorem not_all_equilateral_triangles_have_same_perimeter :
  ∃ t1 t2 : EquilateralTriangle, t1.perimeter ≠ t2.perimeter ∧
    t1.isEquiangular ∧ t2.isEquiangular ∧
    t1.isIsosceles ∧ t2.isIsosceles ∧
    t1.isRegularPolygon ∧ t2.isRegularPolygon ∧
    t1.isSimilarTo t2 :=
  sorry

end NUMINAMATH_CALUDE_not_all_equilateral_triangles_have_same_perimeter_l899_89912


namespace NUMINAMATH_CALUDE_set_problem_l899_89947

def I (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}
def A : Set ℝ := {5}

theorem set_problem (x y : ℝ) (C : Set ℝ) : 
  C ⊆ I x → C \ A = {2, y} → 
  ((x = -4 ∧ y = 3) ∨ (x = 2 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_set_problem_l899_89947


namespace NUMINAMATH_CALUDE_octagon_inner_square_area_l899_89905

/-- The area of the square formed by connecting every second vertex of a regular octagon with side length 10 -/
theorem octagon_inner_square_area : 
  let side_length : ℝ := 10
  let diagonal_length : ℝ := side_length * (1 + Real.sqrt 2)
  let inner_square_area : ℝ := diagonal_length ^ 2
  inner_square_area = 300 + 200 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_octagon_inner_square_area_l899_89905


namespace NUMINAMATH_CALUDE_discount_difference_l899_89959

/-- The cover price of the book in cents -/
def cover_price : ℕ := 3000

/-- The percentage discount as a fraction -/
def percent_discount : ℚ := 1/4

/-- The fixed discount in cents -/
def fixed_discount : ℕ := 500

/-- Applies the percentage discount followed by the fixed discount -/
def percent_then_fixed (price : ℕ) : ℚ :=
  (price : ℚ) * (1 - percent_discount) - fixed_discount

/-- Applies the fixed discount followed by the percentage discount -/
def fixed_then_percent (price : ℕ) : ℚ :=
  ((price : ℚ) - fixed_discount) * (1 - percent_discount)

theorem discount_difference :
  fixed_then_percent cover_price - percent_then_fixed cover_price = 125 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l899_89959


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l899_89930

theorem smallest_four_digit_congruence : ∃ (x : ℕ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 7 ≡ 10 [ZMOD 8] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 35]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20] ∧
   3 * x + 7 ≡ 10 [ZMOD 8] ∧
   -3 * x + 2 ≡ 2 * x [ZMOD 35]) ∧
  x = 1009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l899_89930


namespace NUMINAMATH_CALUDE_thirteenth_row_sum_l899_89956

def row_sum (n : ℕ) : ℕ :=
  3 * 2^(n-1)

theorem thirteenth_row_sum :
  row_sum 13 = 12288 :=
by sorry

end NUMINAMATH_CALUDE_thirteenth_row_sum_l899_89956


namespace NUMINAMATH_CALUDE_initial_fliers_count_l899_89988

theorem initial_fliers_count (morning_fraction : ℚ) (afternoon_fraction : ℚ) (remaining_fliers : ℕ) : 
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  remaining_fliers = 1500 →
  ∃ initial_fliers : ℕ, 
    initial_fliers = 2500 ∧
    (1 - morning_fraction) * (1 - afternoon_fraction) * initial_fliers = remaining_fliers :=
by
  sorry

end NUMINAMATH_CALUDE_initial_fliers_count_l899_89988


namespace NUMINAMATH_CALUDE_whole_number_between_l899_89936

theorem whole_number_between : ∀ M : ℤ, (5.5 < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 6) → M = 23 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_l899_89936


namespace NUMINAMATH_CALUDE_ellie_distance_after_six_steps_l899_89951

/-- The distance Ellie walks after n steps, starting from 0 and aiming for a target 5 meters away,
    walking 1/4 of the remaining distance with each step. -/
def ellieDistance (n : ℕ) : ℚ :=
  5 * (1 - (3/4)^n)

/-- Theorem stating that after 6 steps, Ellie has walked 16835/4096 meters. -/
theorem ellie_distance_after_six_steps :
  ellieDistance 6 = 16835 / 4096 := by
  sorry


end NUMINAMATH_CALUDE_ellie_distance_after_six_steps_l899_89951


namespace NUMINAMATH_CALUDE_perpendicular_planes_l899_89995

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b c : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h3 : perpendicular a α)
  (h4 : subset b β)
  (h5 : parallel a b) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l899_89995


namespace NUMINAMATH_CALUDE_sin_90_degrees_l899_89955

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l899_89955


namespace NUMINAMATH_CALUDE_percentage_loss_calculation_l899_89941

theorem percentage_loss_calculation (cost_price selling_price : ℝ) :
  cost_price = 1600 →
  selling_price = 1440 →
  (cost_price - selling_price) / cost_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_calculation_l899_89941


namespace NUMINAMATH_CALUDE_problem_solution_l899_89910

-- Define the sets A and B
def A (a b c : ℝ) : Prop := a^2 - b*c - 8*a + 7 = 0
def B (a b c : ℝ) : Prop := b^2 + c^2 + b*c - b*a + b = 0

-- Define the function y
def y (a b c : ℝ) : ℝ := a*b + b*c + a*c

-- Theorem statement
theorem problem_solution :
  ∃ (a b c : ℝ), A a b c ∧ B a b c →
  (∀ a : ℝ, (∃ b c : ℝ, A a b c ∧ B a b c) → 1 ≤ a ∧ a ≤ 9) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, 
    A a₁ b₁ c₁ ∧ B a₁ b₁ c₁ ∧ A a₂ b₂ c₂ ∧ B a₂ b₂ c₂ ∧
    y a₁ b₁ c₁ = 88 ∧ y a₂ b₂ c₂ = -56 ∧
    ∀ a b c : ℝ, A a b c ∧ B a b c → -56 ≤ y a b c ∧ y a b c ≤ 88) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l899_89910


namespace NUMINAMATH_CALUDE_circle_area_sum_l899_89911

/-- The sum of areas of an infinite series of circles with specific radii -/
theorem circle_area_sum : 
  let r₀ : ℝ := 2  -- Initial radius
  let ratio : ℝ := 1/3  -- Ratio between subsequent radii
  let area_sum : ℝ := ∑' n, π * (r₀ * ratio^n)^2  -- Sum of areas
  area_sum = 9*π/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_sum_l899_89911


namespace NUMINAMATH_CALUDE_job_completion_time_l899_89900

/-- 
Given two people who can complete a job independently in 10 and 15 days respectively,
this theorem proves that they can complete the job together in 6 days.
-/
theorem job_completion_time 
  (ram_time : ℝ) 
  (gohul_time : ℝ) 
  (h1 : ram_time = 10) 
  (h2 : gohul_time = 15) : 
  (ram_time * gohul_time) / (ram_time + gohul_time) = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l899_89900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l899_89991

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (d : ℚ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 7 * a 11 = 6)
  (h3 : a 4 + a 14 = 5) :
  d = 1/4 ∨ d = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l899_89991


namespace NUMINAMATH_CALUDE_joeys_route_length_l899_89901

/-- Given a round trip with total time 1 hour and average speed 3 miles/hour,
    prove that the one-way distance is 1.5 miles. -/
theorem joeys_route_length (total_time : ℝ) (avg_speed : ℝ) (one_way_distance : ℝ) :
  total_time = 1 →
  avg_speed = 3 →
  one_way_distance = avg_speed * total_time / 2 →
  one_way_distance = 1.5 := by
  sorry

#check joeys_route_length

end NUMINAMATH_CALUDE_joeys_route_length_l899_89901


namespace NUMINAMATH_CALUDE_existence_of_same_sum_opposite_signs_l899_89913

theorem existence_of_same_sum_opposite_signs :
  ∃ (y : ℝ) (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ x₁^4 + x₁^5 = y ∧ x₂^4 + x₂^5 = y :=
by sorry

end NUMINAMATH_CALUDE_existence_of_same_sum_opposite_signs_l899_89913


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l899_89975

/-- The longest side of a triangle with vertices at (1,1), (4,5), and (7,1) has a length of 6 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ), 
  a = (1, 1) ∧ b = (4, 5) ∧ c = (7, 1) ∧
  ∀ (d : ℝ), d = max (dist a b) (max (dist b c) (dist c a)) → d = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l899_89975


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l899_89915

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℝ, 5*x + 2*y = 25 ∧ 3*x + 4*y = 15 ∧ x = 5 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l899_89915


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l899_89903

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) →
  (b^3 - 15*b^2 + 22*b - 8 = 0) →
  (c^3 - 15*c^2 + 22*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 181/9) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l899_89903


namespace NUMINAMATH_CALUDE_smallest_satisfying_both_properties_l899_89963

def is_sum_of_five_fourth_powers (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
    n = a^4 + b^4 + c^4 + d^4 + e^4

def is_sum_of_six_consecutive_integers (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)

theorem smallest_satisfying_both_properties : 
  ∀ n : ℕ, n < 2019 → ¬(is_sum_of_five_fourth_powers n ∧ is_sum_of_six_consecutive_integers n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_both_properties_l899_89963


namespace NUMINAMATH_CALUDE_quadratic_factorization_l899_89969

theorem quadratic_factorization (C E : ℤ) :
  (∀ x, 20 * x^2 - 87 * x + 91 = (C * x - 13) * (E * x - 7)) →
  C * E + C = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l899_89969


namespace NUMINAMATH_CALUDE_decimal_34_to_binary_binary_to_decimal_34_l899_89934

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_34_to_binary :
  toBinary 34 = [false, true, false, false, false, true] :=
by sorry

theorem binary_to_decimal_34 :
  fromBinary [false, true, false, false, false, true] = 34 :=
by sorry

end NUMINAMATH_CALUDE_decimal_34_to_binary_binary_to_decimal_34_l899_89934


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l899_89960

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 5 ∧ c * x + y = 7 ∧ x > 3 ∧ y > 1) ↔ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l899_89960


namespace NUMINAMATH_CALUDE_power_of_power_three_l899_89992

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l899_89992


namespace NUMINAMATH_CALUDE_william_car_wash_body_time_l899_89977

/-- Represents the time William spends washing vehicles -/
def WilliamCarWash :=
  {time_body : ℕ //
    ∃ (time_normal time_suv : ℕ),
      time_normal = time_body + 17 ∧
      time_suv = 2 * time_normal ∧
      2 * time_normal + time_suv = 96}

/-- Theorem stating that William spends 7 minutes washing the car body -/
theorem william_car_wash_body_time :
  ∀ w : WilliamCarWash, w.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_william_car_wash_body_time_l899_89977


namespace NUMINAMATH_CALUDE_factorization_equality_l899_89940

theorem factorization_equality (m : ℝ) : m^2 * (m - 1) + 4 * (1 - m) = (m - 1) * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l899_89940


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l899_89982

theorem number_puzzle_solution : ∃ x : ℤ, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l899_89982


namespace NUMINAMATH_CALUDE_original_celery_cost_l899_89978

def original_order : ℝ := 25
def new_tomatoes : ℝ := 2.20
def old_tomatoes : ℝ := 0.99
def new_lettuce : ℝ := 1.75
def old_lettuce : ℝ := 1.00
def new_celery : ℝ := 2.00
def delivery_tip : ℝ := 8.00
def new_total : ℝ := 35

theorem original_celery_cost :
  ∃ (old_celery : ℝ),
    old_celery = 0.04 ∧
    original_order = old_tomatoes + old_lettuce + old_celery ∧
    new_total = new_tomatoes + new_lettuce + new_celery + delivery_tip :=
by sorry

end NUMINAMATH_CALUDE_original_celery_cost_l899_89978


namespace NUMINAMATH_CALUDE_inequality_solution_l899_89997

theorem inequality_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x - 1) / 2 + 1 ≥ 2 * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l899_89997


namespace NUMINAMATH_CALUDE_power_function_exponent_l899_89909

/-- A power function passing through (1/4, 1/2) has exponent 1/2 -/
theorem power_function_exponent (m : ℝ) (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = m * x^a) ∧ f (1/4) = 1/2) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_function_exponent_l899_89909


namespace NUMINAMATH_CALUDE_nikitas_claim_incorrect_l899_89993

theorem nikitas_claim_incorrect : ¬∃ (x y : ℕ), 5 * (x - y) = 49 := by
  sorry

end NUMINAMATH_CALUDE_nikitas_claim_incorrect_l899_89993


namespace NUMINAMATH_CALUDE_product_of_specific_difference_and_cube_difference_l899_89922

theorem product_of_specific_difference_and_cube_difference
  (x y : ℝ) (h1 : x - y = 4) (h2 : x^3 - y^3 = 28) : x * y = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_difference_and_cube_difference_l899_89922


namespace NUMINAMATH_CALUDE_factorization_identity_l899_89945

theorem factorization_identity (a b : ℝ) : a^2 - 2*a*b = a*(a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l899_89945


namespace NUMINAMATH_CALUDE_T_divisibility_l899_89917

-- Define the set T
def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

-- Theorem statement
theorem T_divisibility :
  (∀ x ∈ T, ¬(5 ∣ x)) ∧ (∃ x ∈ T, 7 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_T_divisibility_l899_89917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l899_89980

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2 : a 2 = 3)
  (h_6 : a 6 = 7) :
  a 11 = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l899_89980


namespace NUMINAMATH_CALUDE_max_houses_buildable_l899_89928

def houses_buildable (sinks doors windows toilets : ℕ) : ℕ :=
  min (sinks / 6) (min (doors / 4) (min (windows / 8) (toilets / 3)))

theorem max_houses_buildable :
  houses_buildable 266 424 608 219 = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_houses_buildable_l899_89928


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l899_89952

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  bottom_base : ℝ
  top_base : ℝ
  side : ℝ

/-- The diagonal of an isosceles trapezoid -/
def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specific isosceles trapezoid is 12√3 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := {
    bottom_base := 24,
    top_base := 12,
    side := 12
  }
  diagonal t = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l899_89952


namespace NUMINAMATH_CALUDE_female_officers_count_l899_89948

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 210 →
  female_on_duty_ratio = 2/3 →
  female_ratio = 24/100 →
  ∃ (total_female : ℕ), total_female = 583 ∧ 
    (↑total_on_duty * female_on_duty_ratio : ℚ) = (↑total_female * female_ratio : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l899_89948


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l899_89973

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := a + (Complex.I - 1) / (1 + Complex.I)
  (z.re = z.im) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l899_89973


namespace NUMINAMATH_CALUDE_not_third_PSU_l899_89907

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the ordering relation for runners
def beats (a b : Runner) : Prop := sorry

-- Define the conditions
axiom P_beats_Q : beats Runner.P Runner.Q
axiom Q_beats_R : beats Runner.Q Runner.R
axiom T_beats_S : beats Runner.T Runner.S
axiom T_beats_U : beats Runner.T Runner.U
axiom U_after_P_before_Q : beats Runner.P Runner.U ∧ beats Runner.U Runner.Q

-- Define what it means to finish third
def finishes_third (r : Runner) : Prop := sorry

-- Theorem statement
theorem not_third_PSU : 
  ¬(finishes_third Runner.P) ∧ 
  ¬(finishes_third Runner.S) ∧ 
  ¬(finishes_third Runner.U) := by sorry

end NUMINAMATH_CALUDE_not_third_PSU_l899_89907


namespace NUMINAMATH_CALUDE_minimum_score_for_average_increase_miguel_minimum_score_l899_89918

def current_scores : List ℕ := [92, 88, 76, 84, 90]
def desired_increase : ℕ := 4

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem minimum_score_for_average_increase 
  (scores : List ℕ) 
  (increase : ℕ) 
  (min_score : ℕ) : Prop :=
  let current_avg := average scores
  let new_scores := scores ++ [min_score]
  let new_avg := average new_scores
  new_avg ≥ current_avg + increase ∧
  ∀ (score : ℕ), score < min_score → 
    average (scores ++ [score]) < current_avg + increase

theorem miguel_minimum_score : 
  minimum_score_for_average_increase current_scores desired_increase 110 := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_for_average_increase_miguel_minimum_score_l899_89918


namespace NUMINAMATH_CALUDE_not_divisible_by_6_and_11_l899_89954

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  (n - 1) - (n - 1) / a - (n - 1) / b + (n - 1) / (a * b)

theorem not_divisible_by_6_and_11 :
  count_not_divisible 1500 6 11 = 1136 := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_6_and_11_l899_89954


namespace NUMINAMATH_CALUDE_wendy_first_day_miles_l899_89926

theorem wendy_first_day_miles (total_miles second_day_miles third_day_miles : ℕ) 
  (h1 : total_miles = 493)
  (h2 : second_day_miles = 223)
  (h3 : third_day_miles = 145) :
  total_miles - (second_day_miles + third_day_miles) = 125 := by
  sorry

end NUMINAMATH_CALUDE_wendy_first_day_miles_l899_89926


namespace NUMINAMATH_CALUDE_lorry_speed_l899_89932

/-- Calculates the speed of a lorry crossing a bridge -/
theorem lorry_speed (lorry_length bridge_length : ℝ) (crossing_time : ℝ) :
  lorry_length = 200 →
  bridge_length = 200 →
  crossing_time = 17.998560115190784 →
  ∃ (speed : ℝ), abs (speed - 80) < 0.01 ∧ 
  speed = (lorry_length + bridge_length) / crossing_time * 3.6 := by
  sorry

#check lorry_speed

end NUMINAMATH_CALUDE_lorry_speed_l899_89932


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l899_89998

/-- The cost of pencils and notebooks given specific quantities -/
structure StationeryCost where
  pencil_cost : ℝ
  notebook_cost : ℝ

/-- The conditions from the problem -/
def problem_conditions (c : StationeryCost) : Prop :=
  4 * c.pencil_cost + 5 * c.notebook_cost = 3.35 ∧
  6 * c.pencil_cost + 4 * c.notebook_cost = 3.16

/-- The theorem to prove -/
theorem stationery_cost_theorem (c : StationeryCost) :
  problem_conditions c →
  20 * c.pencil_cost + 13 * c.notebook_cost = 10.29 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l899_89998


namespace NUMINAMATH_CALUDE_f_minimum_at_three_halves_l899_89968

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem f_minimum_at_three_halves :
  ∃ (y : ℝ), ∀ (x : ℝ), f (3/2) ≤ f x :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_three_halves_l899_89968


namespace NUMINAMATH_CALUDE_justice_plants_l899_89904

theorem justice_plants (ferns_and_palms : ℕ) (desired_total : ℕ) (additional_needed : ℕ) 
  (h1 : ferns_and_palms = 8)
  (h2 : desired_total = 24)
  (h3 : additional_needed = 9) :
  desired_total - additional_needed - ferns_and_palms = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_justice_plants_l899_89904


namespace NUMINAMATH_CALUDE_gianna_savings_l899_89925

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Proves that saving $39 every day for 365 days results in $14,235 total savings -/
theorem gianna_savings : totalSavings 39 365 = 14235 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_l899_89925


namespace NUMINAMATH_CALUDE_solution_set_l899_89914

/-- A function f : ℝ → ℝ satisfying certain properties -/
axiom f : ℝ → ℝ

/-- The derivative of f -/
axiom f' : ℝ → ℝ

/-- f(x-1) is an odd function -/
axiom f_odd : ∀ x, f ((-x) - 1) = -f (x - 1)

/-- For x < -1, (x+1)[f(x) + (x+1)f'(x)] < 0 -/
axiom f_property : ∀ x, x < -1 → (x + 1) * (f x + (x + 1) * f' x) < 0

/-- The solution set for xf(x-1) > f(0) is (-1, 1) -/
theorem solution_set : 
  {x : ℝ | x * f (x - 1) > f 0} = Set.Ioo (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_l899_89914


namespace NUMINAMATH_CALUDE_nails_per_paw_is_four_l899_89902

/-- The number of nails on one paw of a dog -/
def nails_per_paw : ℕ := sorry

/-- The total number of trimmed nails -/
def total_nails : ℕ := 164

/-- The number of dogs with three legs -/
def three_legged_dogs : ℕ := 3

/-- Theorem stating that the number of nails on one paw of a dog is 4 -/
theorem nails_per_paw_is_four : nails_per_paw = 4 := by sorry

end NUMINAMATH_CALUDE_nails_per_paw_is_four_l899_89902


namespace NUMINAMATH_CALUDE_sequence_bound_l899_89994

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j, i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l899_89994


namespace NUMINAMATH_CALUDE_rahul_work_time_l899_89981

theorem rahul_work_time (meena_time : ℝ) (combined_time : ℝ) (rahul_time : ℝ) : 
  meena_time = 10 →
  combined_time = 10 / 3 →
  1 / rahul_time + 1 / meena_time = 1 / combined_time →
  rahul_time = 5 := by
sorry

end NUMINAMATH_CALUDE_rahul_work_time_l899_89981


namespace NUMINAMATH_CALUDE_log_expression_simplification_l899_89923

theorem log_expression_simplification 
  (p q r s t z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + 2 * Real.log (r / s) - Real.log (p * t / (s * z)) = 
  Real.log (r * z / (s * t)) := by
sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l899_89923


namespace NUMINAMATH_CALUDE_limit_of_a_l899_89942

def a (n : ℕ) : ℚ := (2 * n + 3) / (n + 5)

theorem limit_of_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_l899_89942


namespace NUMINAMATH_CALUDE_creative_arts_academy_painting_paradox_l899_89916

theorem creative_arts_academy_painting_paradox :
  let total_students : ℝ := 100
  let enjoy_painting_ratio : ℝ := 0.7
  let dont_enjoy_painting_ratio : ℝ := 1 - enjoy_painting_ratio
  let enjoy_but_negate_ratio : ℝ := 0.25
  let dont_enjoy_but_affirm_ratio : ℝ := 0.15

  let enjoy_painting : ℝ := total_students * enjoy_painting_ratio
  let dont_enjoy_painting : ℝ := total_students * dont_enjoy_painting_ratio
  
  let enjoy_but_negate : ℝ := enjoy_painting * enjoy_but_negate_ratio
  let dont_enjoy_but_affirm : ℝ := dont_enjoy_painting * dont_enjoy_but_affirm_ratio
  
  let total_claim_dislike : ℝ := enjoy_but_negate + (dont_enjoy_painting - dont_enjoy_but_affirm)
  
  (enjoy_but_negate / total_claim_dislike) * 100 = 40.698 :=
by sorry

end NUMINAMATH_CALUDE_creative_arts_academy_painting_paradox_l899_89916


namespace NUMINAMATH_CALUDE_function_positivity_condition_l899_89931

theorem function_positivity_condition (m : ℝ) : 
  (∀ x : ℝ, max (2*m*x^2 - 2*(4-m)*x + 1) (m*x) > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end NUMINAMATH_CALUDE_function_positivity_condition_l899_89931


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l899_89906

/-- Given a geometric sequence {a_n} where a_2 = 8 and a_5 = 64, the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 2 = 8 →                     -- Given condition
  a 5 = 64 →                    -- Given condition
  q = 2 :=                      -- Conclusion to prove
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l899_89906


namespace NUMINAMATH_CALUDE_math_club_team_selection_l899_89966

/-- The number of ways to select a team from a math club --/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (team_boys : ℕ) (team_girls : ℕ) 
  (experienced_boys : ℕ) (experienced_girls : ℕ) : ℕ :=
  (Nat.choose (total_boys - experienced_boys) (team_boys - experienced_boys)) * 
  (Nat.choose (total_girls - experienced_girls) (team_girls - experienced_girls))

/-- Theorem: The number of ways to select the team is 540 --/
theorem math_club_team_selection :
  select_team 7 10 6 3 3 1 1 = 540 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l899_89966


namespace NUMINAMATH_CALUDE_outbound_speed_l899_89961

/-- Proves that given a round trip of 2 hours, with an outbound journey of 70 minutes
    and a return journey at 105 km/h, the outbound journey speed is 75 km/h -/
theorem outbound_speed (total_time : Real) (outbound_time : Real) (return_speed : Real) :
  total_time = 2 →
  outbound_time = 70 / 60 →
  return_speed = 105 →
  (total_time - outbound_time) * return_speed = outbound_time * 75 := by
  sorry

#check outbound_speed

end NUMINAMATH_CALUDE_outbound_speed_l899_89961


namespace NUMINAMATH_CALUDE_triangle_transformation_l899_89939

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def transform (p : ℝ × ℝ) : ℝ × ℝ := 
  reflect_y (rotate_180 (reflect_x p))

theorem triangle_transformation :
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (4, 1)
  let C : ℝ × ℝ := (2, 3)
  (transform A = A) ∧ (transform B = B) ∧ (transform C = C) :=
sorry

end NUMINAMATH_CALUDE_triangle_transformation_l899_89939


namespace NUMINAMATH_CALUDE_blue_cards_count_l899_89935

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℚ) (blue_cards : ℕ) : 
  red_cards = 8 →
  blue_prob = 6/10 →
  (blue_cards : ℚ) / (blue_cards + red_cards) = blue_prob →
  blue_cards = 12 := by
sorry

end NUMINAMATH_CALUDE_blue_cards_count_l899_89935


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l899_89999

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (n.cast * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := sorry

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧  -- arithmetic sequence property
  a 7 = 4 ∧                                 -- given condition
  a 19 = 2 * a 9 ∧                          -- given condition
  (∀ n : ℕ, a n = (1 + n.cast) / 2) ∧       -- general formula for a_n
  (∀ n : ℕ, S n = (2 * n.cast) / (n.cast + 1)) -- sum of first n terms of b_n
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l899_89999


namespace NUMINAMATH_CALUDE_percentage_loss_l899_89919

/-- Calculate the percentage of loss in a sale transaction -/
theorem percentage_loss (cost_price selling_price : ℚ) (h1 : cost_price = 1800) (h2 : selling_price = 1620) :
  (cost_price - selling_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_l899_89919


namespace NUMINAMATH_CALUDE_product_difference_theorem_l899_89944

theorem product_difference_theorem (number value : ℕ) (h1 : number = 15) (h2 : value = 13) :
  number * value - number = 180 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_theorem_l899_89944


namespace NUMINAMATH_CALUDE_part_one_part_two_l899_89929

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m + 1

-- Part I
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : Set.Iic (-2) ∪ Set.Ici 2 = {x | f m (x - 3) ≥ 0}) : 
  m = 3 := by sorry

-- Part II
theorem part_two : 
  {t : ℝ | ∃ x, |x + 3| - 2 ≥ |2*x - 1| - t^2 + 5/2*t} = 
  Set.Iic 1 ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l899_89929


namespace NUMINAMATH_CALUDE_cs_majors_consecutive_probability_l899_89908

def total_people : ℕ := 12
def cs_majors : ℕ := 5
def chem_majors : ℕ := 4
def lit_majors : ℕ := 3

theorem cs_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let consecutive_arrangements := Nat.factorial (total_people - cs_majors) * Nat.factorial cs_majors
  (consecutive_arrangements : ℚ) / total_arrangements = 1 / 66 := by
  sorry

end NUMINAMATH_CALUDE_cs_majors_consecutive_probability_l899_89908


namespace NUMINAMATH_CALUDE_simplify_expression_l899_89953

theorem simplify_expression (x y : ℝ) : 3*x + 5*x + 7*x + 2*y = 15*x + 2*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l899_89953


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l899_89938

/-- Calculates the profit percentage given cost price, marked price, and discount percentage. -/
def profit_percentage (cost_price marked_price discount_percent : ℚ) : ℚ :=
  let discount := (discount_percent / 100) * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that for the given conditions, the profit percentage is 25%. -/
theorem profit_percentage_is_25_percent :
  profit_percentage 95 125 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l899_89938


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l899_89927

-- Define the propositions
variable (P Q : Prop)

-- Define the original implication
def original_statement : Prop := P → Q

-- Define the contrapositive
def contrapositive : Prop := ¬Q → ¬P

-- Define the disjunction form
def disjunction_form : Prop := ¬P ∨ Q

-- Theorem stating the equivalence of the three forms
theorem equivalence_of_statements :
  (original_statement P Q) ↔ (contrapositive P Q) ∧ (disjunction_form P Q) :=
sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l899_89927


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_range_l899_89943

/-- The range of k for which a hyperbola and parabola have at most two intersections -/
theorem hyperbola_parabola_intersection_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 - y^2 + 1 = 0 ∧ y^2 = (k - 1) * x →
    (∃! p q : ℝ × ℝ, (p.1^2 - p.2^2 + 1 = 0 ∧ p.2^2 = (k - 1) * p.1) ∧
                     (q.1^2 - q.2^2 + 1 = 0 ∧ q.2^2 = (k - 1) * q.1) ∧
                     p ≠ q) ∨
    (∃! p : ℝ × ℝ, p.1^2 - p.2^2 + 1 = 0 ∧ p.2^2 = (k - 1) * p.1) ∨
    (∀ x y : ℝ, x^2 - y^2 + 1 ≠ 0 ∨ y^2 ≠ (k - 1) * x)) →
  -1 ≤ k ∧ k < 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_range_l899_89943


namespace NUMINAMATH_CALUDE_game_ends_in_37_rounds_l899_89958

/-- Represents the state of the game at any point --/
structure GameState where
  a : ℕ  -- tokens of player A
  b : ℕ  -- tokens of player B
  c : ℕ  -- tokens of player C

/-- Represents a single round of the game --/
def playRound (state : GameState) : GameState :=
  if state.a ≥ state.b ∧ state.a ≥ state.c then
    { a := state.a - 3, b := state.b + 1, c := state.c + 1 }
  else if state.b ≥ state.a ∧ state.b ≥ state.c then
    { a := state.a + 1, b := state.b - 3, c := state.c + 1 }
  else
    { a := state.a + 1, b := state.b + 1, c := state.c - 3 }

/-- Checks if the game has ended (any player has 0 tokens) --/
def gameEnded (state : GameState) : Bool :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- Plays the game for a given number of rounds --/
def playGame (initialState : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initialState
  | n + 1 => playRound (playGame initialState n)

/-- The main theorem to prove --/
theorem game_ends_in_37_rounds :
  let initialState : GameState := { a := 15, b := 14, c := 13 }
  let finalState := playGame initialState 37
  gameEnded finalState ∧ ¬gameEnded (playGame initialState 36) := by
  sorry

#check game_ends_in_37_rounds

end NUMINAMATH_CALUDE_game_ends_in_37_rounds_l899_89958


namespace NUMINAMATH_CALUDE_video_game_expenditure_l899_89986

theorem video_game_expenditure (total : ℝ) (books snacks movies video_games : ℝ) : 
  total = 50 ∧ 
  books = (1/4) * total ∧ 
  snacks = (1/5) * total ∧ 
  movies = (2/5) * total ∧ 
  total = books + snacks + movies + video_games 
  → video_games = 7.5 := by
sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l899_89986


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l899_89950

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ 12 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 1 ∧ y > 1 ∧
  (x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) < 12 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l899_89950


namespace NUMINAMATH_CALUDE_cricketer_average_score_l899_89946

theorem cricketer_average_score 
  (initial_average : ℝ) 
  (runs_19th_inning : ℝ) 
  (average_increase : ℝ) : 
  runs_19th_inning = 96 →
  average_increase = 4 →
  (18 * initial_average + runs_19th_inning) / 19 = initial_average + average_increase →
  (18 * initial_average + runs_19th_inning) / 19 = 24 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l899_89946


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l899_89983

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 7 / 2 →  -- The ratio of the measures is 7:2
  max a b - min a b = 100 :=  -- The positive difference is 100°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l899_89983


namespace NUMINAMATH_CALUDE_two_digit_numbers_count_l899_89924

/-- The number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then Nat.factorial n / Nat.factorial (n - r) else 0

/-- The set of digits used -/
def digits : Finset ℕ := {1, 2, 3, 4, 5}

theorem two_digit_numbers_count : permutations (Finset.card digits) 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_count_l899_89924


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_3_l899_89965

/-- A quadratic function of the form y = (x - m)^2 - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 - 1

/-- The function decreases as x increases when x ≤ 3 -/
def decreasing_for_x_leq_3 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≤ x₂ → x₂ ≤ 3 → f m x₁ ≥ f m x₂

/-- If the quadratic function y = (x - m)^2 - 1 decreases as x increases when x ≤ 3,
    then m ≥ 3 -/
theorem quadratic_decreasing_implies_m_geq_3 (m : ℝ) :
  decreasing_for_x_leq_3 m → m ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_3_l899_89965


namespace NUMINAMATH_CALUDE_employee_pay_l899_89920

theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = 638 →
  x = 1.2 * y →
  total_pay = x + y →
  y = 290 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l899_89920


namespace NUMINAMATH_CALUDE_no_perfect_squares_l899_89987

theorem no_perfect_squares (x y : ℕ+) : 
  ¬(∃ (a b : ℕ), (x^2 + y + 2 : ℕ) = a^2 ∧ (y^2 + 4*x : ℕ) = b^2) :=
sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l899_89987


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l899_89949

/-- A color representation --/
inductive Color
| Black
| White

/-- A 3 × 7 grid where each cell is colored either black or white --/
def Grid := Fin 3 → Fin 7 → Color

/-- A rectangle in the grid, represented by its top-left and bottom-right corners --/
structure Rectangle where
  top_left : Fin 3 × Fin 7
  bottom_right : Fin 3 × Fin 7

/-- Check if a rectangle has all corners of the same color --/
def has_same_color_corners (g : Grid) (r : Rectangle) : Prop :=
  let (t, l) := r.top_left
  let (b, r) := r.bottom_right
  g t l = g t r ∧ g t l = g b l ∧ g t l = g b r

/-- Main theorem: There exists a rectangle with all corners of the same color --/
theorem exists_same_color_rectangle (g : Grid) : 
  ∃ r : Rectangle, has_same_color_corners g r := by sorry

end NUMINAMATH_CALUDE_exists_same_color_rectangle_l899_89949


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l899_89989

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 80)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 295) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l899_89989


namespace NUMINAMATH_CALUDE_y_divisibility_l899_89937

def y : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464 + 8192

theorem y_divisibility :
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  (∃ k : ℕ, y = 32 * k) ∧
  (∃ k : ℕ, y = 64 * k) :=
sorry

end NUMINAMATH_CALUDE_y_divisibility_l899_89937


namespace NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l899_89972

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The number of rows in the triangular array -/
def N : ℕ := sorry

theorem triangular_array_sum_of_digits : 
  (triangular_sum N = 5050) → (sum_of_digits N = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l899_89972


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l899_89976

theorem min_value_sum_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a ≥ 12 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a = 12 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l899_89976
