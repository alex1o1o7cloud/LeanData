import Mathlib

namespace triangle_special_angle_l280_28086

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a^2 + b^2 = c^2 - √2ab, then angle C = 3π/4 -/
theorem triangle_special_angle (a b c : ℝ) (h : a^2 + b^2 = c^2 - Real.sqrt 2 * a * b) :
  let angle_C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_C = 3 * π / 4 := by
  sorry

end triangle_special_angle_l280_28086


namespace parabola_chord_midpoint_to_directrix_l280_28016

/-- Given a parabola y² = 4x and a chord AB of length 7 intersecting the parabola at points A(x₁, y₁) and B(x₂, y₂),
    the distance from the midpoint M of the chord to the parabola's directrix is 7/2. -/
theorem parabola_chord_midpoint_to_directrix
  (x₁ y₁ x₂ y₂ : ℝ) 
  (on_parabola_A : y₁^2 = 4*x₁)
  (on_parabola_B : y₂^2 = 4*x₂)
  (chord_length : Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 7) :
  let midpoint_x := (x₁ + x₂) / 2
  (midpoint_x + 1) = 7/2 := by sorry

end parabola_chord_midpoint_to_directrix_l280_28016


namespace system_of_equations_solution_l280_28082

theorem system_of_equations_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + x*y + y^2 = 108)
  (h_eq2 : y^2 + y*z + z^2 = 49)
  (h_eq3 : z^2 + x*z + x^2 = 157) :
  x*y + y*z + x*z = 104 := by
sorry

end system_of_equations_solution_l280_28082


namespace incorrect_operation_l280_28018

theorem incorrect_operation : (4 + 5)^2 ≠ 4^2 + 5^2 := by sorry

end incorrect_operation_l280_28018


namespace train_speed_l280_28031

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 2500) (h2 : time = 100) :
  length / time = 25 := by
  sorry

end train_speed_l280_28031


namespace all_integers_are_cute_l280_28024

/-- An integer is cute if it can be written as a^2 + b^3 + c^3 + d^5 for some integers a, b, c, and d. -/
def IsCute (n : ℤ) : Prop :=
  ∃ a b c d : ℤ, n = a^2 + b^3 + c^3 + d^5

/-- All integers are cute. -/
theorem all_integers_are_cute : ∀ n : ℤ, IsCute n := by
  sorry


end all_integers_are_cute_l280_28024


namespace telephone_answered_probability_l280_28000

theorem telephone_answered_probability :
  let p1 : ℝ := 0.1  -- Probability of answering at first ring
  let p2 : ℝ := 0.3  -- Probability of answering at second ring
  let p3 : ℝ := 0.4  -- Probability of answering at third ring
  let p4 : ℝ := 0.1  -- Probability of answering at fourth ring
  1 - (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 0.9
  := by sorry

end telephone_answered_probability_l280_28000


namespace range_of_a_range_of_b_l280_28087

-- Define propositions
def p (a : ℝ) : Prop := ∀ x, 2^x + 1 ≥ a

def q (a : ℝ) : Prop := ∀ x, a * x^2 - x + a > 0

def m (a b : ℝ) : Prop := ∃ x, x^2 + b*x + a = 0

-- Theorem for part (1)
theorem range_of_a : 
  (∃ a, p a ∧ q a) → (∀ a, p a ∧ q a → a > 1/2 ∧ a ≤ 1) :=
sorry

-- Theorem for part (2)
theorem range_of_b :
  (∀ a b, (¬p a → ¬m a b) ∧ ¬(m a b → ¬p a)) →
  (∀ b, (∃ a, ¬p a ∧ m a b) → b > -2 ∧ b < 2) :=
sorry

end range_of_a_range_of_b_l280_28087


namespace three_from_seven_combination_l280_28008

theorem three_from_seven_combination : Nat.choose 7 3 = 35 := by
  sorry

end three_from_seven_combination_l280_28008


namespace sum_of_squared_digits_l280_28062

/-- The number of digits in 222222222 -/
def n : ℕ := 9

/-- The number whose square we're considering -/
def num : ℕ := 222222222

/-- Function to calculate the sum of digits of a number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem sum_of_squared_digits : sum_of_digits (num ^ 2) = 162 := by sorry

end sum_of_squared_digits_l280_28062


namespace g_of_fifty_l280_28063

/-- A function g satisfying g(xy) = xg(y) for all real x and y, and g(1) = 30 -/
def g : ℝ → ℝ :=
  fun x => x * 30

/-- Theorem stating that g(50) = 1500 -/
theorem g_of_fifty : g 50 = 1500 := by
  sorry

end g_of_fifty_l280_28063


namespace stadium_length_conversion_l280_28021

/-- Conversion factor from feet to yards -/
def feet_per_yard : ℚ := 3

/-- Length of the stadium in feet -/
def stadium_length_feet : ℚ := 183

/-- Length of the stadium in yards -/
def stadium_length_yards : ℚ := stadium_length_feet / feet_per_yard

theorem stadium_length_conversion :
  stadium_length_yards = 61 := by
  sorry

end stadium_length_conversion_l280_28021


namespace greatest_difference_multiple_of_five_l280_28014

theorem greatest_difference_multiple_of_five : ∀ a b : ℕ,
  (a < 10) →
  (b < 10) →
  (700 + 10 * a + b) % 5 = 0 →
  ((a + b) % 5 = 0) →
  ∃ c d : ℕ,
    (c < 10) ∧
    (d < 10) ∧
    (700 + 10 * c + d) % 5 = 0 ∧
    ((c + d) % 5 = 0) ∧
    (∀ e f : ℕ,
      (e < 10) →
      (f < 10) →
      (700 + 10 * e + f) % 5 = 0 →
      ((e + f) % 5 = 0) →
      (a + b) - (c + d) ≤ (e + f) - (c + d)) ∧
    (a + b) - (c + d) = 10 :=
by sorry

end greatest_difference_multiple_of_five_l280_28014


namespace solve_equation_l280_28076

theorem solve_equation : ∃ x : ℝ, 10 * x - (2 * 1.5 / 0.3) = 50 ∧ x = 6 := by
  sorry

end solve_equation_l280_28076


namespace sum_divisible_by_1987_l280_28003

def odd_product : ℕ := (List.range 993).foldl (λ acc i => acc * (2 * i + 1)) 1

def even_product : ℕ := (List.range 993).foldl (λ acc i => acc * (2 * i + 2)) 1

theorem sum_divisible_by_1987 : 
  ∃ k : ℤ, (odd_product : ℤ) + (even_product : ℤ) = 1987 * k := by
  sorry

end sum_divisible_by_1987_l280_28003


namespace min_dot_product_l280_28058

open Real

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the fixed point M
def M : ℝ × ℝ := (1, 0)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the vector from M to a point P
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - M.1, P.2 - M.2)

theorem min_dot_product :
  ∃ (min : ℝ),
    (∀ (A B : ℝ × ℝ),
      ellipse A.1 A.2 →
      ellipse B.1 B.2 →
      dot_product (vector_MP A) (vector_MP B) = 0 →
      dot_product (vector_MP A) (A.1 - B.1, A.2 - B.2) ≥ min) ∧
    (∃ (A B : ℝ × ℝ),
      ellipse A.1 A.2 ∧
      ellipse B.1 B.2 ∧
      dot_product (vector_MP A) (vector_MP B) = 0 ∧
      dot_product (vector_MP A) (A.1 - B.1, A.2 - B.2) = min) ∧
    min = 2/3 :=
by sorry

end min_dot_product_l280_28058


namespace students_with_all_pets_l280_28068

theorem students_with_all_pets (total_students : ℕ) 
  (dog_ratio : ℚ) (cat_ratio : ℚ) (other_pets : ℕ) (no_pets : ℕ)
  (only_dogs : ℕ) (dogs_and_other : ℕ) (only_cats : ℕ) :
  total_students = 40 →
  dog_ratio = 1/2 →
  cat_ratio = 2/5 →
  other_pets = 8 →
  no_pets = 7 →
  only_dogs = 12 →
  dogs_and_other = 3 →
  only_cats = 11 →
  ∃ (all_pets : ℕ),
    all_pets = 5 ∧
    total_students * dog_ratio = only_dogs + dogs_and_other + all_pets ∧
    total_students * cat_ratio = only_cats + all_pets ∧
    other_pets = dogs_and_other + all_pets ∧
    total_students - no_pets = only_dogs + dogs_and_other + only_cats + all_pets :=
by sorry

end students_with_all_pets_l280_28068


namespace min_value_h_positive_m_l280_28053

/-- The minimum value of ax - ln x for x > 0 and a ≥ 1 is 1 + ln a -/
theorem min_value_h (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, a * x - Real.log x ≥ 1 + Real.log a := by sorry

/-- For all x > 0 and a ≥ 1, ax - ln(x + 1) > 0 -/
theorem positive_m (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, a * x - Real.log (x + 1) > 0 := by sorry

end min_value_h_positive_m_l280_28053


namespace derivative_inequality_implies_function_inequality_l280_28061

theorem derivative_inequality_implies_function_inequality 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x > 0, deriv f x - f x / x > 0) → 3 * f 4 > 4 * f 3 := by
  sorry

end derivative_inequality_implies_function_inequality_l280_28061


namespace intersection_of_A_and_B_l280_28033

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}
def B : Set ℝ := {y | y > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l280_28033


namespace egg_processing_plant_l280_28027

theorem egg_processing_plant (E : ℕ) : 
  (96 : ℚ) / 100 * E + (4 : ℚ) / 100 * E = E → -- Original ratio
  ((96 : ℚ) / 100 * E + 12) / E = (99 : ℚ) / 100 → -- New ratio with 12 additional accepted eggs
  E = 400 := by
sorry

end egg_processing_plant_l280_28027


namespace cafe_sign_white_area_l280_28073

/-- Represents a rectangular sign with painted letters -/
structure Sign :=
  (width : ℕ)
  (height : ℕ)
  (c_area : ℕ)
  (a_area : ℕ)
  (f_area : ℕ)
  (e_area : ℕ)

/-- Calculates the white area of the sign -/
def white_area (s : Sign) : ℕ :=
  s.width * s.height - (s.c_area + s.a_area + s.f_area + s.e_area)

/-- Theorem stating that the white area of the given sign is 66 square units -/
theorem cafe_sign_white_area :
  ∃ (s : Sign),
    s.width = 6 ∧
    s.height = 18 ∧
    s.c_area = 11 ∧
    s.a_area = 10 ∧
    s.f_area = 12 ∧
    s.e_area = 9 ∧
    white_area s = 66 :=
sorry

end cafe_sign_white_area_l280_28073


namespace arccos_equation_solution_l280_28032

theorem arccos_equation_solution (x : ℝ) : 
  Real.arccos (3 * x) - Real.arccos (2 * x) = π / 6 →
  x = 1 / (2 * Real.sqrt (12 - 6 * Real.sqrt 3)) ∨
  x = -1 / (2 * Real.sqrt (12 - 6 * Real.sqrt 3)) :=
by sorry

end arccos_equation_solution_l280_28032


namespace min_swaps_to_reverse_l280_28022

/-- Represents a strip of cells containing tokens -/
def Strip := Fin 100 → ℕ

/-- Reverses the order of tokens in the strip -/
def reverse (s : Strip) : Strip :=
  fun i => s (99 - i)

/-- Represents a swap operation -/
inductive Swap
  | adjacent : Fin 100 → Swap
  | free : Fin 96 → Swap

/-- Applies a swap operation to a strip -/
def applySwap (s : Strip) (swap : Swap) : Strip :=
  match swap with
  | Swap.adjacent i => 
      if i < 99 then
        fun j => if j = i then s (i+1) 
                 else if j = i+1 then s i
                 else s j
      else s
  | Swap.free i => 
      fun j => if j = i then s (i+4)
               else if j = i+4 then s i
               else s j

/-- A sequence of swap operations -/
def SwapSequence := List Swap

/-- Applies a sequence of swaps to a strip -/
def applySwaps (s : Strip) : SwapSequence → Strip
  | [] => s
  | (swap :: rest) => applySwaps (applySwap s swap) rest

/-- Counts the number of adjacent swaps in a sequence -/
def countAdjacentSwaps : SwapSequence → ℕ
  | [] => 0
  | (Swap.adjacent _ :: rest) => 1 + countAdjacentSwaps rest
  | (_ :: rest) => countAdjacentSwaps rest

/-- The main theorem: proving that 50 adjacent swaps are required to reverse the strip -/
theorem min_swaps_to_reverse (s : Strip) : 
  (∃ swaps : SwapSequence, applySwaps s swaps = reverse s) → 
  (∃ minSwaps : SwapSequence, 
    applySwaps s minSwaps = reverse s ∧ 
    countAdjacentSwaps minSwaps = 50 ∧
    ∀ swaps : SwapSequence, applySwaps s swaps = reverse s → 
      countAdjacentSwaps minSwaps ≤ countAdjacentSwaps swaps) :=
by sorry

end min_swaps_to_reverse_l280_28022


namespace rectangle_area_error_percentage_l280_28005

/-- Given a rectangle where one side is measured 8% in excess and the other side is measured 5% in deficit, 
    the error percentage in the calculated area is 2.6%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.08 * L) (h2 : W' = 0.95 * W) :
  (L' * W' - L * W) / (L * W) * 100 = 2.6 := by
  sorry

end rectangle_area_error_percentage_l280_28005


namespace regression_line_not_most_points_l280_28090

/-- A type representing a scatter plot of data points. -/
structure ScatterPlot where
  points : Set (ℝ × ℝ)

/-- A type representing a line in 2D space. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The regression line for a given scatter plot. -/
noncomputable def regressionLine (plot : ScatterPlot) : Line :=
  sorry

/-- The number of points a line passes through in a scatter plot. -/
def pointsPassed (line : Line) (plot : ScatterPlot) : ℕ :=
  sorry

/-- The statement that the regression line passes through the most points. -/
def regressionLinePassesMostPoints (plot : ScatterPlot) : Prop :=
  ∀ l : Line, pointsPassed (regressionLine plot) plot ≥ pointsPassed l plot

/-- Theorem stating that the regression line does not necessarily pass through the most points. -/
theorem regression_line_not_most_points :
  ∃ plot : ScatterPlot, ¬(regressionLinePassesMostPoints plot) :=
sorry

end regression_line_not_most_points_l280_28090


namespace hyperbola_eccentricity_l280_28030

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the line x = a²/c intersects its asymptotes at points A and B,
    and triangle ABF is a right-angled triangle (where F is the right focus),
    then the eccentricity of the hyperbola is √2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  A.1 = a^2 / c →
  B.1 = a^2 / c →
  F.1 = c →
  F.2 = 0 →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
  c / a = Real.sqrt 2 :=
by sorry

end hyperbola_eccentricity_l280_28030


namespace largest_consecutive_composite_l280_28084

theorem largest_consecutive_composite : ∃ (n : ℕ), 
  (n < 50) ∧ 
  (n ≥ 10) ∧ 
  (∀ i ∈ Finset.range 10, ¬(Nat.Prime (n - i))) ∧
  (∀ m : ℕ, m > n → ¬(∀ i ∈ Finset.range 10, ¬(Nat.Prime (m - i)))) :=
by sorry

end largest_consecutive_composite_l280_28084


namespace greatest_three_digit_multiple_of_17_l280_28074

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l280_28074


namespace ratio_odd_even_divisors_l280_28017

def M : ℕ := 36 * 36 * 98 * 210

-- Sum of odd divisors
def sum_odd_divisors (n : ℕ) : ℕ := sorry

-- Sum of even divisors
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 60 = sum_even_divisors M := by sorry

end ratio_odd_even_divisors_l280_28017


namespace parallel_lines_imply_a_eq_3_l280_28075

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0

/-- The second line equation: 3x + (a - 1)y = a - 7 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

/-- The theorem stating that if the two lines are parallel, then a = 3 -/
theorem parallel_lines_imply_a_eq_3 :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 :=
by sorry

end parallel_lines_imply_a_eq_3_l280_28075


namespace divisible_by_three_l280_28046

theorem divisible_by_three (A B : ℤ) (h : A > B) :
  ∃ x : ℤ, (x = A ∨ x = B ∨ x = A + B ∨ x = A - B) ∧ x % 3 = 0 := by
  sorry

end divisible_by_three_l280_28046


namespace even_sum_probability_l280_28037

def wheel1_sections : ℕ := 6
def wheel1_even_sections : ℕ := 2
def wheel1_odd_sections : ℕ := 4

def wheel2_sections : ℕ := 4
def wheel2_even_sections : ℕ := 1
def wheel2_odd_sections : ℕ := 3

theorem even_sum_probability :
  let p_even_sum := (wheel1_even_sections / wheel1_sections) * (wheel2_even_sections / wheel2_sections) +
                    (wheel1_odd_sections / wheel1_sections) * (wheel2_odd_sections / wheel2_sections)
  p_even_sum = 7 / 12 := by
  sorry

end even_sum_probability_l280_28037


namespace triangle_angle_60_degrees_l280_28070

theorem triangle_angle_60_degrees (A B C : Real) (hABC : A + B + C = Real.pi)
  (h_eq : Real.sin A ^ 2 - Real.sin C ^ 2 + Real.sin B ^ 2 = Real.sin A * Real.sin B) :
  C = Real.pi / 3 := by
  sorry

end triangle_angle_60_degrees_l280_28070


namespace max_b_value_l280_28047

theorem max_b_value (b : ℕ+) (x : ℤ) (h : x^2 + b*x = -21) : b ≤ 22 := by
  sorry

end max_b_value_l280_28047


namespace lcm_of_12_and_15_l280_28009

theorem lcm_of_12_and_15 : 
  let a := 12
  let b := 15
  let hcf := 3
  let lcm := Nat.lcm a b
  lcm = 60 := by
  sorry

end lcm_of_12_and_15_l280_28009


namespace clock_strikes_ten_l280_28004

/-- A clock that strikes at regular intervals -/
structure StrikingClock where
  /-- The time it takes to complete a given number of strikes -/
  strike_time : ℕ → ℝ
  /-- The number of strikes at a given hour -/
  strikes_at_hour : ℕ → ℕ

/-- Our specific clock that takes 7 seconds to strike 7 times at 7 o'clock -/
def our_clock : StrikingClock where
  strike_time := fun n => if n = 7 then 7 else 0  -- We only know about 7 strikes
  strikes_at_hour := fun h => if h = 7 then 7 else 0  -- We only know about 7 o'clock

/-- The theorem stating that our clock takes 10.5 seconds to strike 10 times -/
theorem clock_strikes_ten (c : StrikingClock) (h : c.strike_time 7 = 7) :
  c.strike_time 10 = 10.5 := by
  sorry

#check clock_strikes_ten our_clock (by rfl)

end clock_strikes_ten_l280_28004


namespace r_fourth_plus_inverse_r_fourth_l280_28097

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by sorry

end r_fourth_plus_inverse_r_fourth_l280_28097


namespace book_pages_proof_l280_28042

/-- Calculates the number of digits used to number pages from 1 to n -/
def digits_used (n : ℕ) : ℕ := sorry

/-- The number of pages in the book -/
def num_pages : ℕ := 155

/-- The total number of digits used to number all pages -/
def total_digits : ℕ := 357

theorem book_pages_proof : digits_used num_pages = total_digits := by sorry

end book_pages_proof_l280_28042


namespace integer_roots_of_polynomial_l280_28079

def polynomial (x : ℤ) : ℤ := x^3 - 2*x^2 + 3*x - 17

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-17, -1, 1, 17} := by sorry

end integer_roots_of_polynomial_l280_28079


namespace candy_distribution_l280_28083

theorem candy_distribution (total_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ) : 
  total_candies = 90 →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  (total_candies / 3 : ℕ) % lollipops_per_boy = 0 →
  ((2 * total_candies / 3) : ℕ) % candy_canes_per_girl = 0 →
  (total_candies / 3 / lollipops_per_boy : ℕ) + 
  ((2 * total_candies / 3) / candy_canes_per_girl : ℕ) = 40 :=
by sorry

end candy_distribution_l280_28083


namespace equation_solution_l280_28039

theorem equation_solution (x : ℝ) (h : 9 - 16/x + 9/x^2 = 0) : 3/x = 3 := by
  sorry

end equation_solution_l280_28039


namespace polynomial_expansion_properties_l280_28001

theorem polynomial_expansion_properties 
  (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) : 
  (a₀ = -1) ∧ (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) := by
  sorry

end polynomial_expansion_properties_l280_28001


namespace man_work_days_l280_28029

theorem man_work_days (man_son_days : ℝ) (son_days : ℝ) (man_days : ℝ) : 
  man_son_days = 4 → son_days = 20 → man_days = 5 := by
  sorry

end man_work_days_l280_28029


namespace calculate_expression_l280_28060

theorem calculate_expression : 3000 * (3000^2999 - 3000^2998) = 3000^2999 := by
  sorry

end calculate_expression_l280_28060


namespace terminal_side_of_half_angle_l280_28077

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 135) ∨
  (∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 315)

theorem terminal_side_of_half_angle (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by sorry

end terminal_side_of_half_angle_l280_28077


namespace sqrt_five_squared_l280_28092

theorem sqrt_five_squared : (Real.sqrt 5) ^ 2 = 5 := by
  sorry

end sqrt_five_squared_l280_28092


namespace cone_base_radius_l280_28023

/-- Given a sector with radius 5 and central angle 144°, prove that when wrapped into a cone, 
    the radius of the base of the cone is 2. -/
theorem cone_base_radius (r : ℝ) (θ : ℝ) : 
  r = 5 → θ = 144 → (θ / 360) * (2 * π * r) = 2 * π * 2 := by
  sorry

end cone_base_radius_l280_28023


namespace union_of_M_and_N_l280_28040

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end union_of_M_and_N_l280_28040


namespace parabola_vertex_l280_28043

/-- The parabola defined by y = -x^2 + cx + d -/
def parabola (c d : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + d

/-- The solution set of the inequality -x^2 + cx + d ≤ 0 -/
def solution_set (c d : ℝ) : Set ℝ := {x | x ∈ Set.Icc (-6) (-1) ∨ x ∈ Set.Ici 4}

theorem parabola_vertex (c d : ℝ) :
  (solution_set c d = {x | x ∈ Set.Icc (-6) (-1) ∨ x ∈ Set.Ici 4}) →
  (∃ (x y : ℝ), x = 7/2 ∧ y = -171/4 ∧
    ∀ (t : ℝ), parabola c d t ≤ parabola c d x) :=
by sorry

end parabola_vertex_l280_28043


namespace f_inequality_range_l280_28089

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : f (2*x) > f (x - 3) ↔ x < -3 ∨ x > 1 := by
  sorry

end f_inequality_range_l280_28089


namespace certain_number_problem_l280_28095

theorem certain_number_problem (x : ℝ) (y : ℝ) (h1 : x = 3) 
  (h2 : (x + 1) / (x + y) = (x + y) / (x + 13)) : y = 5 := by
  sorry

end certain_number_problem_l280_28095


namespace complex_fraction_evaluation_l280_28072

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 2 := by
  sorry

end complex_fraction_evaluation_l280_28072


namespace rain_probability_l280_28055

/-- The probability of rain on Friday -/
def prob_friday : ℝ := 0.7

/-- The probability of rain on Saturday -/
def prob_saturday : ℝ := 0.5

/-- The probability of rain on Sunday -/
def prob_sunday : ℝ := 0.3

/-- The events are independent -/
axiom independence : True

/-- The probability of rain on all three days -/
def prob_all_days : ℝ := prob_friday * prob_saturday * prob_sunday

/-- Theorem: The probability of rain on all three days is 10.5% -/
theorem rain_probability : prob_all_days = 0.105 := by
  sorry

end rain_probability_l280_28055


namespace cake_chord_length_squared_l280_28013

theorem cake_chord_length_squared (d : ℝ) (n : ℕ) (l : ℝ) : 
  d = 18 → n = 4 → l = (d / 2) * Real.sqrt 2 → l^2 = 162 := by
  sorry

end cake_chord_length_squared_l280_28013


namespace viewers_scientific_notation_l280_28050

/-- Represents 1 billion -/
def billion : ℝ := 1000000000

/-- The number of viewers who watched the Spring Festival Gala live broadcast -/
def viewers : ℝ := 1.173 * billion

/-- Theorem stating that the number of viewers in billions is equal to its scientific notation -/
theorem viewers_scientific_notation : viewers = 1.173 * (10 : ℝ)^9 := by
  sorry

end viewers_scientific_notation_l280_28050


namespace canoe_rowing_probability_l280_28067

/-- The probability of rowing a canoe given certain conditions on oar functionality and weather -/
theorem canoe_rowing_probability :
  let p_left_works : ℚ := 3/5  -- Probability left oar works
  let p_right_works : ℚ := 3/5  -- Probability right oar works
  let p_weather : ℚ := 1/4  -- Probability of adverse weather
  let p_oar_works_in_weather (p : ℚ) : ℚ := 1 - 2 * (1 - p)  -- Probability oar works in adverse weather
  
  let p_both_work_no_weather : ℚ := p_left_works * p_right_works
  let p_both_work_weather : ℚ := p_oar_works_in_weather p_left_works * p_oar_works_in_weather p_right_works
  
  let p_row : ℚ := p_both_work_no_weather * (1 - p_weather) + p_both_work_weather * p_weather

  p_row = 7/25 := by sorry

end canoe_rowing_probability_l280_28067


namespace min_c_value_l280_28056

/-- Given natural numbers a, b, c where a < b < c, and a system of equations with exactly one solution,
    prove that the minimum possible value of c is 1018. -/
theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
    (h3 : ∃! (x y : ℝ), 2 * x + y = 2035 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1018 ∧ ∃ (a' b' : ℕ), a' < b' ∧ b' < 1018 ∧
    ∃! (x y : ℝ), 2 * x + y = 2035 ∧ y = |x - a'| + |x - b'| + |x - 1018| :=
by sorry

end min_c_value_l280_28056


namespace total_odd_green_and_red_marbles_l280_28034

/-- Represents a person's marble collection --/
structure MarbleCollection where
  green : Nat
  red : Nat
  blue : Nat

/-- Counts odd numbers of green and red marbles --/
def countOddGreenAndRed (mc : MarbleCollection) : Nat :=
  (if mc.green % 2 = 1 then mc.green else 0) +
  (if mc.red % 2 = 1 then mc.red else 0)

theorem total_odd_green_and_red_marbles :
  let sara := MarbleCollection.mk 3 5 6
  let tom := MarbleCollection.mk 4 7 2
  let lisa := MarbleCollection.mk 5 3 7
  countOddGreenAndRed sara + countOddGreenAndRed tom + countOddGreenAndRed lisa = 23 := by
  sorry

end total_odd_green_and_red_marbles_l280_28034


namespace campers_fed_specific_l280_28059

/-- The number of campers that can be fed given the caught fish --/
def campers_fed (trout_weight : ℕ) (bass_count bass_weight : ℕ) (salmon_count salmon_weight : ℕ) (consumption_per_camper : ℕ) : ℕ :=
  (trout_weight + bass_count * bass_weight + salmon_count * salmon_weight) / consumption_per_camper

/-- Theorem stating the number of campers that can be fed given the specific fishing scenario --/
theorem campers_fed_specific : campers_fed 8 6 2 2 12 2 = 22 := by
  sorry

end campers_fed_specific_l280_28059


namespace complex_sum_of_powers_i_l280_28035

theorem complex_sum_of_powers_i (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end complex_sum_of_powers_i_l280_28035


namespace odd_cube_plus_linear_plus_constant_l280_28051

theorem odd_cube_plus_linear_plus_constant (o n m : ℤ) 
  (ho : ∃ k : ℤ, o = 2*k + 1) : 
  Odd (o^3 + n*o + m) ↔ Even m := by
  sorry

end odd_cube_plus_linear_plus_constant_l280_28051


namespace greatest_whole_number_satisfying_inequalities_l280_28025

theorem greatest_whole_number_satisfying_inequalities :
  ∃ (n : ℕ), n = 1 ∧
  (∀ (x : ℝ), (x > n → ¬(3 * x - 5 < 1 - x ∧ 2 * x + 4 ≤ 8))) ∧
  (3 * n - 5 < 1 - n ∧ 2 * n + 4 ≤ 8) :=
by sorry

end greatest_whole_number_satisfying_inequalities_l280_28025


namespace ashton_pencils_left_l280_28041

/-- The number of pencils Ashton has left after giving some away -/
def pencils_left (initial_boxes : ℕ) (pencils_per_box : ℕ) (given_to_brother : ℕ) (given_to_friends : ℕ) : ℕ :=
  initial_boxes * pencils_per_box - given_to_brother - given_to_friends

/-- Theorem stating that Ashton has 24 pencils left -/
theorem ashton_pencils_left : pencils_left 3 14 6 12 = 24 := by
  sorry

end ashton_pencils_left_l280_28041


namespace smallest_prime_factor_of_1729_l280_28006

theorem smallest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → p ≤ q :=
by sorry

end smallest_prime_factor_of_1729_l280_28006


namespace probability_one_match_l280_28096

/-- Represents the two topics that can be chosen. -/
inductive Topic : Type
  | A : Topic
  | B : Topic

/-- Represents a selection of topics by the three teachers. -/
def Selection := Topic × Topic × Topic

/-- The set of all possible selections. -/
def allSelections : Finset Selection := sorry

/-- Predicate for selections where exactly one male and the female choose the same topic. -/
def exactlyOneMatch (s : Selection) : Prop := sorry

/-- The set of selections where exactly one male and the female choose the same topic. -/
def matchingSelections : Finset Selection := sorry

/-- Theorem stating that the probability of exactly one male and the female choosing the same topic is 1/2. -/
theorem probability_one_match :
  (matchingSelections.card : ℚ) / allSelections.card = 1 / 2 := by sorry

end probability_one_match_l280_28096


namespace employees_using_public_transportation_l280_28038

theorem employees_using_public_transportation 
  (total_employees : ℕ) 
  (drive_percentage : ℚ) 
  (public_transport_fraction : ℚ) :
  total_employees = 100 →
  drive_percentage = 60 / 100 →
  public_transport_fraction = 1 / 2 →
  (total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction = 20 := by
  sorry

end employees_using_public_transportation_l280_28038


namespace small_circle_radius_l280_28065

theorem small_circle_radius (R : ℝ) (h : R = 5) :
  let d := Real.sqrt (2 * R^2)
  let r := (d - 2*R) / 2
  r = (Real.sqrt 200 - 10) / 2 := by sorry

end small_circle_radius_l280_28065


namespace fraction_sum_equality_l280_28091

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (40 - a) + b / (75 - b) + c / (85 - c) = 8) :
  8 / (40 - a) + 15 / (75 - b) + 17 / (85 - c) = 40 := by
  sorry

end fraction_sum_equality_l280_28091


namespace representative_selection_counts_l280_28064

def num_boys : Nat := 5
def num_girls : Nat := 3
def num_representatives : Nat := 5
def num_subjects : Nat := 5

theorem representative_selection_counts :
  let scenario1 := (num_girls.choose 1) * (num_boys.choose 4) * (num_representatives.factorial) +
                   (num_girls.choose 2) * (num_boys.choose 3) * (num_representatives.factorial)
  let scenario2 := ((num_boys + num_girls - 1).choose (num_representatives - 1)) * ((num_representatives - 1).factorial)
  let scenario3 := ((num_boys + num_girls - 1).choose (num_representatives - 1)) * ((num_representatives - 1).factorial) * (num_subjects - 1)
  let scenario4 := ((num_boys + num_girls - 2).choose (num_representatives - 2)) * ((num_representatives - 2).factorial) * (num_subjects - 1)
  (∃ (count1 count2 count3 count4 : Nat),
    count1 = scenario1 ∧
    count2 = scenario2 ∧
    count3 = scenario3 ∧
    count4 = scenario4) := by sorry

end representative_selection_counts_l280_28064


namespace point_on_line_l280_28098

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨3, 0⟩
  let p2 : Point := ⟨11, 4⟩
  let p3 : Point := ⟨19, 8⟩
  collinear p1 p2 p3 := by sorry

end point_on_line_l280_28098


namespace quadratic_function_uniqueness_l280_28071

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_uniqueness
  (f : ℝ → ℝ)
  (h_quad : is_quadratic f)
  (h_solution_set : ∀ x, f x < 0 ↔ 0 < x ∧ x < 5)
  (h_max_value : ∀ x ∈ Set.Icc (-1) 4, f x ≤ 12)
  (h_attains_max : ∃ x ∈ Set.Icc (-1) 4, f x = 12) :
  ∀ x, f x = 2 * x^2 - 10 * x :=
sorry

end quadratic_function_uniqueness_l280_28071


namespace complex_modulus_problem_l280_28078

theorem complex_modulus_problem (z : ℂ) :
  (2017 * z - 25) / (z - 2017) = (3 : ℂ) + 4 * I →
  Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l280_28078


namespace comprehensive_survey_suitable_for_grade_8_1_l280_28036

/-- Represents a type of survey -/
inductive SurveyType
| Sampling
| Comprehensive

/-- Represents a population to be surveyed -/
structure Population where
  size : ℕ
  accessibility : Bool
  variability : Bool

/-- Determines if a survey type is suitable for a given population -/
def is_suitable (st : SurveyType) (p : Population) : Prop :=
  match st with
  | SurveyType.Sampling => p.size > 1000 ∨ p.accessibility = false ∨ p.variability = true
  | SurveyType.Comprehensive => p.size ≤ 1000 ∧ p.accessibility = true ∧ p.variability = false

/-- Represents the population of Grade 8 (1) students in a certain school -/
def grade_8_1_population : Population :=
  { size := 50,  -- Assuming a typical class size
    accessibility := true,
    variability := false }

/-- Theorem stating that a comprehensive survey is suitable for the Grade 8 (1) population -/
theorem comprehensive_survey_suitable_for_grade_8_1 :
  is_suitable SurveyType.Comprehensive grade_8_1_population :=
by
  sorry


end comprehensive_survey_suitable_for_grade_8_1_l280_28036


namespace max_f_sum_l280_28010

/-- A permutation of 4n letters consisting of n occurrences each of A, B, C, and D -/
def Permutation (n : ℕ) := Fin (4 * n) → Fin 4

/-- The number of B's to the right of each A in the permutation -/
def f_AB (σ : Permutation n) : ℕ := sorry

/-- The number of C's to the right of each B in the permutation -/
def f_BC (σ : Permutation n) : ℕ := sorry

/-- The number of D's to the right of each C in the permutation -/
def f_CD (σ : Permutation n) : ℕ := sorry

/-- The number of A's to the right of each D in the permutation -/
def f_DA (σ : Permutation n) : ℕ := sorry

/-- The sum of f_AB, f_BC, f_CD, and f_DA for a given permutation -/
def f_sum (σ : Permutation n) : ℕ := f_AB σ + f_BC σ + f_CD σ + f_DA σ

theorem max_f_sum (n : ℕ) : (∀ σ : Permutation n, f_sum σ ≤ 3 * n^2) ∧ (∃ σ : Permutation n, f_sum σ = 3 * n^2) := by sorry

end max_f_sum_l280_28010


namespace multiple_with_binary_digits_l280_28066

theorem multiple_with_binary_digits (n : ℕ+) : ∃ m : ℕ,
  (n : ℕ) ∣ m ∧
  (Nat.digits 2 m).length ≤ n ∧
  ∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1 := by
  sorry

end multiple_with_binary_digits_l280_28066


namespace sin_negative_945_degrees_l280_28093

theorem sin_negative_945_degrees : Real.sin ((-945 : ℝ) * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_negative_945_degrees_l280_28093


namespace equivalent_discount_l280_28026

theorem equivalent_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  original_price = 50 →
  first_discount = 0.3 →
  second_discount = 0.4 →
  let discounted_price := original_price * (1 - first_discount)
  let final_price := discounted_price * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 0.58 := by
sorry

end equivalent_discount_l280_28026


namespace fraction_product_value_l280_28052

/-- The product of fractions from 8/4 to 2008/2004 following the pattern (4n+4)/(4n) -/
def fraction_product : ℚ :=
  (2008 : ℚ) / 4

theorem fraction_product_value : fraction_product = 502 := by
  sorry

end fraction_product_value_l280_28052


namespace average_b_c_l280_28099

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : a - c = 80) : 
  (b + c) / 2 = 70 := by
  sorry

end average_b_c_l280_28099


namespace choose_two_from_three_l280_28094

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n.choose k = 3 :=
  by
  -- Assume n = 3 and k = 2
  have h1 : n = 3 := by sorry
  have h2 : k = 2 := by sorry
  
  -- Define the number of interest groups
  let num_groups : ℕ := 3
  
  -- Define the number of groups to choose
  let groups_to_choose : ℕ := 2
  
  -- Assert that n and k match our problem
  have h3 : n = num_groups := by rw [h1]
  have h4 : k = groups_to_choose := by rw [h2]
  
  -- Prove that choosing 2 from 3 equals 3
  sorry

end choose_two_from_three_l280_28094


namespace portraits_not_taken_l280_28028

theorem portraits_not_taken (total_students : ℕ) (before_lunch : ℕ) (after_lunch : ℕ) : 
  total_students = 24 → 
  before_lunch = total_students / 3 →
  after_lunch = 10 →
  total_students - (before_lunch + after_lunch) = 6 := by
sorry

end portraits_not_taken_l280_28028


namespace sqrt_two_squared_l280_28069

theorem sqrt_two_squared : (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_two_squared_l280_28069


namespace remainder_987654_div_8_l280_28048

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end remainder_987654_div_8_l280_28048


namespace tanC_over_tanA_max_tanB_l280_28044

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + 2*t.b^2 = t.c^2

-- Theorem 1: If the condition is satisfied, then tan C / tan A = -3
theorem tanC_over_tanA (t : Triangle) (h : satisfiesCondition t) :
  Real.tan t.C / Real.tan t.A = -3 :=
sorry

-- Theorem 2: If the condition is satisfied, then the maximum value of tan B is √3/3
theorem max_tanB (t : Triangle) (h : satisfiesCondition t) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 3 ∧ Real.tan t.B ≤ max :=
sorry

end tanC_over_tanA_max_tanB_l280_28044


namespace special_function_value_l280_28007

/-- A monotonic function on (0, +∞) satisfying f(f(x) - 1/x) = 2 for all x > 0 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ 0 < y ∧ x < y → f x < f y) ∧ 
  (∀ x, 0 < x → f (f x - 1/x) = 2)

/-- Theorem stating that for a special function f, f(1/5) = 6 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f (1/5) = 6 := by
  sorry

end special_function_value_l280_28007


namespace investment_duration_theorem_l280_28049

def initial_investment : ℝ := 2000
def interest_rate_1 : ℝ := 0.08
def interest_rate_2 : ℝ := 0.12
def final_value : ℝ := 6620
def years_at_rate_1 : ℕ := 2

def investment_equation (t : ℝ) : Prop :=
  initial_investment * (1 + interest_rate_1) ^ years_at_rate_1 * (1 + interest_rate_2) ^ (t - years_at_rate_1) = final_value

theorem investment_duration_theorem :
  ∃ t : ℕ, (∀ s : ℝ, investment_equation s → t ≥ ⌈s⌉) ∧ investment_equation (t : ℝ) := by
  sorry

end investment_duration_theorem_l280_28049


namespace max_t_value_max_t_is_negative_one_l280_28085

open Real

noncomputable def f (x : ℝ) : ℝ := log x / (x + 1)

theorem max_t_value (t : ℝ) :
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t / x > log x / (x - 1)) →
  t ≤ -1 :=
by sorry

theorem max_t_is_negative_one :
  ∃ t : ℝ, t = -1 ∧
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t / x > log x / (x - 1)) ∧
  (∀ t' : ℝ, (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t' / x > log x / (x - 1)) → t' ≤ t) :=
by sorry

end max_t_value_max_t_is_negative_one_l280_28085


namespace parallel_line_theorem_perpendicular_lines_theorem_l280_28011

-- Define the line l1
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the parallel line l2
def l2_parallel (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

-- Define the perpendicular lines l2
def l2_perp_pos (x y : ℝ) : Prop := 4 * x - 3 * y + 4 * Real.sqrt 6 = 0
def l2_perp_neg (x y : ℝ) : Prop := 4 * x - 3 * y - 4 * Real.sqrt 6 = 0

-- Theorem for parallel line
theorem parallel_line_theorem :
  (∀ x y, l2_parallel x y ↔ ∃ k, 3 * x + 4 * y = k) ∧
  l2_parallel (-1) 3 := by sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines_theorem :
  (∀ x y, (l2_perp_pos x y ∨ l2_perp_neg x y) → 
    (3 * 4 + 4 * (-3) = 0)) ∧
  (∀ x y, l2_perp_pos x y → 
    (1/2 * |x| * |y| = 4 ∧ 4 * x = 0 → y = Real.sqrt 6 ∧ 3 * y = 0 → x = 4/3 * Real.sqrt 6)) ∧
  (∀ x y, l2_perp_neg x y → 
    (1/2 * |x| * |y| = 4 ∧ 4 * x = 0 → y = Real.sqrt 6 ∧ 3 * y = 0 → x = 4/3 * Real.sqrt 6)) := by sorry

end parallel_line_theorem_perpendicular_lines_theorem_l280_28011


namespace no_real_some_complex_solutions_l280_28088

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x + 1)^2
def equation2 (x y : ℂ) : Prop := x * y^2 + y = 1

-- Theorem statement
theorem no_real_some_complex_solutions :
  (∀ x y : ℝ, ¬(equation1 x y ∧ equation2 x y)) ∧
  (∃ x y : ℂ, equation1 x y ∧ equation2 x y) :=
sorry

end no_real_some_complex_solutions_l280_28088


namespace f_2004_equals_2003_l280_28019

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_2004_equals_2003 
  (f g : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_relation : ∀ x, g x = f (x - 1))
  (h_g1 : g 1 = 2003) :
  f 2004 = 2003 := by
  sorry

end f_2004_equals_2003_l280_28019


namespace hash_twelve_six_l280_28081

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the theorem
theorem hash_twelve_six :
  (∀ r s : ℝ, hash r 0 = r) →
  (∀ r s : ℝ, hash r s = hash s r) →
  (∀ r s : ℝ, hash (r + 2) s = hash r s + 2 * s + 2) →
  hash 12 6 = 168 :=
by
  sorry

end hash_twelve_six_l280_28081


namespace triangle_pairs_theorem_l280_28054

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle_pair (t1 t2 : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t1
  let (d, e, f) := t2
  is_triangle a b c ∧ is_triangle d e f ∧ a + b + c + d + e + f = 16

theorem triangle_pairs_theorem :
  ∀ t1 t2 : ℕ × ℕ × ℕ,
  valid_triangle_pair t1 t2 →
  ((t1 = (4, 4, 3) ∧ t2 = (1, 2, 2)) ∨
   (t1 = (4, 4, 2) ∧ t2 = (2, 2, 2)) ∨
   (t1 = (4, 4, 1) ∧ t2 = (3, 2, 2)) ∨
   (t1 = (4, 4, 1) ∧ t2 = (3, 3, 1)) ∨
   (t2 = (4, 4, 3) ∧ t1 = (1, 2, 2)) ∨
   (t2 = (4, 4, 2) ∧ t1 = (2, 2, 2)) ∨
   (t2 = (4, 4, 1) ∧ t1 = (3, 2, 2)) ∨
   (t2 = (4, 4, 1) ∧ t1 = (3, 3, 1))) :=
by sorry

end triangle_pairs_theorem_l280_28054


namespace franks_final_score_l280_28012

/-- Calculates the final score in a trivia competition given the number of correct and incorrect answers in each half. -/
def final_score (first_half_correct first_half_incorrect second_half_correct second_half_incorrect : ℕ) : ℤ :=
  let points_per_correct : ℤ := 3
  let points_per_incorrect : ℤ := -1
  (first_half_correct * points_per_correct + first_half_incorrect * points_per_incorrect) +
  (second_half_correct * points_per_correct + second_half_incorrect * points_per_incorrect)

/-- Theorem stating that Frank's final score in the trivia competition is 39 points. -/
theorem franks_final_score :
  final_score 6 4 10 5 = 39 := by
  sorry

end franks_final_score_l280_28012


namespace box_dimensions_sum_l280_28020

-- Define the dimensions of the box
variable (P Q R : ℝ)

-- Define the conditions
def condition1 : Prop := P * Q = 30
def condition2 : Prop := P * R = 50
def condition3 : Prop := Q * R = 90

-- Theorem statement
theorem box_dimensions_sum 
  (h1 : condition1 P Q)
  (h2 : condition2 P R)
  (h3 : condition3 Q R) :
  P + Q + R = 18 * Real.sqrt 1.5 := by
  sorry

end box_dimensions_sum_l280_28020


namespace value_of_c_l280_28002

theorem value_of_c (a c : ℝ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end value_of_c_l280_28002


namespace cherry_tree_leaves_l280_28057

/-- The number of cherry trees originally planned to be planted -/
def original_plan : ℕ := 7

/-- The actual number of cherry trees planted -/
def actual_trees : ℕ := 2 * original_plan

/-- The number of leaves each tree drops -/
def leaves_per_tree : ℕ := 100

/-- The total number of leaves falling from all cherry trees -/
def total_leaves : ℕ := actual_trees * leaves_per_tree

theorem cherry_tree_leaves : total_leaves = 1400 := by
  sorry

end cherry_tree_leaves_l280_28057


namespace min_product_of_three_distinct_l280_28080

def S : Finset Int := {-10, -5, -3, 0, 4, 6, 9}

theorem min_product_of_three_distinct (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≤ x * y * z :=
by sorry

end min_product_of_three_distinct_l280_28080


namespace turtle_distribution_theorem_l280_28015

/-- The ratio of turtles received by Marion, Martha, and Martin -/
def turtle_ratio : Fin 3 → ℕ
| 0 => 3  -- Marion
| 1 => 2  -- Martha
| 2 => 1  -- Martin

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The total number of turtles received by all three -/
def total_turtles : ℕ := martha_turtles * (turtle_ratio 0 + turtle_ratio 1 + turtle_ratio 2) / turtle_ratio 1

theorem turtle_distribution_theorem : total_turtles = 120 := by
  sorry

end turtle_distribution_theorem_l280_28015


namespace money_distribution_problem_l280_28045

/-- The number of people in the money distribution problem -/
def num_people : ℕ := 195

/-- The amount of coins the first person receives -/
def first_person_coins : ℕ := 3

/-- The amount of coins each person receives after redistribution -/
def redistribution_coins : ℕ := 100

theorem money_distribution_problem :
  ∃ (n : ℕ), n = num_people ∧
  first_person_coins * n + (n * (n - 1)) / 2 = redistribution_coins * n :=
by sorry

end money_distribution_problem_l280_28045
