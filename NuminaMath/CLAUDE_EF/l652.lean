import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_after_one_second_l652_65219

noncomputable section

-- Define the square
def square_side_length : ℝ := 4

-- Define the speeds of P and Q
def speed_P : ℝ := 1
def speed_Q : ℝ := 2

-- Define the time
def time : ℝ := 1

-- Define the positions of P and Q after 1 second
def position_P (t : ℝ) : ℝ × ℝ := 
  if t * speed_P ≤ square_side_length then (0, t * speed_P)
  else if t * speed_P ≤ 2 * square_side_length then (t * speed_P - square_side_length, square_side_length)
  else if t * speed_P ≤ 3 * square_side_length then (square_side_length, 3 * square_side_length - t * speed_P)
  else (4 * square_side_length - t * speed_P, 0)

def position_Q (t : ℝ) : ℝ × ℝ := 
  if t * speed_Q ≤ square_side_length then (t * speed_Q, square_side_length)
  else if t * speed_Q ≤ 2 * square_side_length then (square_side_length, 2 * square_side_length - t * speed_Q)
  else if t * speed_Q ≤ 3 * square_side_length then (3 * square_side_length - t * speed_Q, 0)
  else (0, t * speed_Q - 3 * square_side_length)

-- Define the position of N
def position_N : ℝ × ℝ := (square_side_length, square_side_length / 2)

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem area_after_one_second :
  triangle_area position_N (position_P time) (position_Q time) = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_after_one_second_l652_65219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_maximum_l652_65210

open Real

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (sin (π/4 + x) - sin (π/4 - x)) * sin (π/3 + x)

-- Define the set of x-values where the maximum occurs
def max_x_set : Set ℝ := {x | ∃ k : ℤ, x = k * π + π/3}

-- Theorem statement
theorem y_maximum :
  (∀ x : ℝ, y x ≤ 3 * sqrt 2 / 4) ∧
  (∀ x : ℝ, x ∈ max_x_set → y x = 3 * sqrt 2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_maximum_l652_65210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_proof_l652_65265

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℚ
  water : ℚ

/-- The capacity of the can in liters -/
def canCapacity : ℚ := 30

/-- The amount of milk added in liters -/
def milkAdded : ℚ := 10

/-- Calculates the ratio of milk to water -/
def milkWaterRatio (contents : CanContents) : ℚ := contents.milk / contents.water

/-- The initial contents of the can -/
def initialContents : CanContents := { milk := 4, water := 3 }

/-- The final contents of the can after adding milk -/
def finalContents : CanContents := { milk := initialContents.milk + milkAdded, water := initialContents.water }

theorem can_capacity_proof :
  (milkWaterRatio initialContents = 4/3) →
  (milkWaterRatio finalContents = 5/2) →
  (finalContents.milk = initialContents.milk + milkAdded) →
  (finalContents.water = initialContents.water) →
  (finalContents.milk + finalContents.water = canCapacity) →
  (canCapacity = 30) := by
  sorry

#eval canCapacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_proof_l652_65265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l652_65240

theorem book_price_change (P : ℝ) (h : P > 0) : 
  P * (1 - 0.15) * (1 + 0.10) = P * 0.935 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l652_65240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_equal_digit_sums_l652_65218

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 695 is the smallest positive integer n such that s(n) = 20 and s(n+864) = 20 -/
theorem smallest_n_with_equal_digit_sums :
  ∃ n, n = 695 ∧ (∀ m < n, ¬(sum_of_digits m = 20 ∧ sum_of_digits (m + 864) = 20)) ∧
    sum_of_digits n = 20 ∧ sum_of_digits (n + 864) = 20 := by
  sorry

#check smallest_n_with_equal_digit_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_equal_digit_sums_l652_65218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_is_673_l652_65232

/-- Represents the earnings from a fundraiser car wash activity over three days -/
structure FundraiserEarnings where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ
  friday_earnings : friday = 147
  saturday_earnings : saturday = 2 * friday + 7
  sunday_earnings : sunday = friday + 78

/-- Theorem stating that the total earnings over three days is $673 -/
theorem total_earnings_is_673 (e : FundraiserEarnings) :
  e.friday + e.saturday + e.sunday = 673 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_is_673_l652_65232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_even_and_increasing_l652_65208

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2
noncomputable def g (x : ℝ) : ℝ := 2^(|x|)

-- Theorem statement
theorem f_and_g_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, g (-x) = g x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y → g x < g y) :=
by
  sorry

#check f_and_g_even_and_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_even_and_increasing_l652_65208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l652_65246

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 4), -1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 4), Real.cos (x / 4) ^ 2)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  f A = -1/2 →
  a = 2 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (∃ (S : ℝ), S = 1/2 * b * c * Real.sin A ∧
              ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ S) →
  ∃ (S : ℝ), S = Real.sqrt 3 ∧
             ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l652_65246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_center_properties_l652_65289

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point O
variable (O : ℝ × ℝ)

-- Define vectors OA, OB, OC
def OA (A O : ℝ × ℝ) : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)
def OB (B O : ℝ × ℝ) : ℝ × ℝ := (B.1 - O.1, B.2 - O.2)
def OC (C O : ℝ × ℝ) : ℝ × ℝ := (C.1 - O.1, C.2 - O.2)

-- Define coefficients m, n, p
variable (m n p : ℝ)

-- Define the condition m⃗OA + n⃗OB + p⃗OC = 0⃗
def condition (m n p : ℝ) (A B C O : ℝ × ℝ) : Prop :=
  (m * (OA A O).1 + n * (OB B O).1 + p * (OC C O).1 = 0) ∧
  (m * (OA A O).2 + n * (OB B O).2 + p * (OC C O).2 = 0)

-- Define predicates for the four statements
def is_inside (O A B C : ℝ × ℝ) : Prop := sorry
def is_centroid (O A B C : ℝ × ℝ) : Prop := sorry
def is_incenter (O A B C : ℝ × ℝ) : Prop := sorry
def is_circumcenter (O A B C : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem triangle_center_properties (A B C O : ℝ × ℝ) (m n p : ℝ) :
  condition m n p A B C O →
  (is_inside O A B C ∧ is_centroid O A B C ∧ is_incenter O A B C) ∨
  (is_inside O A B C ∧ is_centroid O A B C ∧ is_circumcenter O A B C) ∨
  (is_inside O A B C ∧ is_incenter O A B C ∧ is_circumcenter O A B C) ∨
  (is_centroid O A B C ∧ is_incenter O A B C ∧ is_circumcenter O A B C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_center_properties_l652_65289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_l652_65290

/-- A dilation that transforms a circle -/
structure CircleDilation where
  original_center : ℝ × ℝ
  original_radius : ℝ
  dilated_center : ℝ × ℝ
  dilated_radius : ℝ

/-- The distance moved by a point under a dilation -/
noncomputable def distance_moved (d : CircleDilation) (p : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating the distance moved by the origin under the specific dilation -/
theorem origin_movement (d : CircleDilation) 
  (h1 : d.original_center = (3, 3))
  (h2 : d.original_radius = 3)
  (h3 : d.dilated_center = (7, 9))
  (h4 : d.dilated_radius = 4.5) :
  distance_moved d (0, 0) = 1.5 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_l652_65290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_possible_l652_65280

/-- A triangle with two sides of length 4 and 10 can have a third side of length 11 -/
theorem triangle_third_side_possible (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  |a - b| < c ∧ |b - c| < a ∧ |c - a| < b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_possible_l652_65280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l652_65272

/-- Given an angle α in the fourth quadrant with its terminal side passing through P(4, y) and sin α = y/5, prove that tan α = -3/4 -/
theorem tan_alpha_fourth_quadrant (α : ℝ) (y : ℝ) 
  (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) -- α is in the fourth quadrant
  (h2 : Real.sin α = y / 5) -- sin α = y/5
  (h3 : (4 : ℝ) = 4 * Real.cos α) -- terminal side passes through (4, y)
  (h4 : y = 5 * Real.sin α) -- terminal side passes through (4, y)
  : Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l652_65272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l652_65285

open Real

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 3)

noncomputable def l (x : ℝ) : ℝ := -x + 2 * π / 3

theorem intersection_count : 
  ∃! x : ℝ, f x = l x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l652_65285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l652_65212

-- Define the number of subsets function
def n (S : Finset α) : ℕ := 2^(Finset.card S)

-- Define the theorem
theorem min_intersection_size 
  (A B C : Finset ℕ)
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : Finset.card A = 100)
  (h3 : Finset.card B = 100) :
  97 ≤ Finset.card (A ∩ B ∩ C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l652_65212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_flash_time_l652_65277

/-- Proves that a light flashing every 6 seconds will flash 600 times in exactly 1 hour -/
theorem light_flash_time (flash_interval flash_count seconds_per_hour : ℝ) : 
  flash_interval * flash_count = seconds_per_hour →
  flash_interval = 6 →
  flash_count = 600 →
  seconds_per_hour = 3600 →
  (flash_interval * flash_count) / seconds_per_hour = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_flash_time_l652_65277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_eq_l652_65268

open Real

theorem smallest_positive_solution_tan_eq (x : ℝ) : 
  (x > 0 ∧ tan (2*x) + tan (5*x) = 1 / cos (5*x) ∧ 
   ∀ y, y > 0 ∧ tan (2*y) + tan (5*y) = 1 / cos (5*y) → x ≤ y) → 
  x = π / 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_eq_l652_65268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_triangle_area_l652_65296

-- Define the ellipse parameters
def major_axis_length : ℝ := 10
def focus_distance : ℝ := 3

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (focus_distance, 0)
def F₂ : ℝ × ℝ := (-focus_distance, 0)

-- Define P as an endpoint of the minor axis
def P : ℝ × ℝ := (0, 4)

-- Theorem for the standard equation of the ellipse
theorem ellipse_equation : 
  ∀ x y : ℝ, ellipse x y ↔ x^2 / 25 + y^2 / 16 = 1 := by
  sorry

-- Theorem for the area of triangle F₁PF₂
theorem triangle_area : 
  (1 / 2) * ‖F₁ - F₂‖ * P.2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_triangle_area_l652_65296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l652_65226

noncomputable section

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Condition that a point (x, y) is on the parabola -/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  p.y_coord x = y

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (-p.b / (2 * p.a), p.y_coord (-p.b / (2 * p.a)))

/-- Condition that a parabola has a vertical axis of symmetry -/
def Parabola.has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  ∀ x y : ℝ, p.contains_point x y → p.contains_point (2 * p.vertex.1 - x) y

theorem parabola_coefficients :
  ∀ p : Parabola,
  p.vertex = (3, 2) →
  p.has_vertical_axis_of_symmetry →
  p.contains_point 1 0 →
  (p.a, p.b, p.c) = (-1/2, 3, -5/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficients_l652_65226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arith_seq_sum_l652_65200

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: For an arithmetic sequence with S_7 = 14, a_3 + a_5 = 4 -/
theorem arith_seq_sum (seq : ArithmeticSequence) (h : S seq 7 = 14) :
  seq.a 3 + seq.a 5 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arith_seq_sum_l652_65200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_11_degrees_l652_65252

/-- Represents the volume of a gas as a function of temperature -/
noncomputable def gas_volume (initial_temp : ℝ) (initial_volume : ℝ) (temp : ℝ) : ℝ :=
  initial_volume + (5 / 4) * (temp - initial_temp)

/-- Theorem: The volume of the gas at 11° is 17.5 cubic centimeters -/
theorem gas_volume_at_11_degrees 
  (initial_temp : ℝ) 
  (initial_volume : ℝ) 
  (h1 : initial_temp = 25) 
  (h2 : initial_volume = 35) :
  gas_volume initial_temp initial_volume 11 = 17.5 := by
  sorry

#check gas_volume_at_11_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_11_degrees_l652_65252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l652_65247

-- Define the function f
noncomputable def f (a : ℝ) : ℝ := ∫ x in (Set.Icc 0 1), 2 * a * x^2 - a^2 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), (∀ (a : ℝ), f a ≤ max) ∧ (∃ (a : ℝ), f a = max) ∧ (max = 2/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l652_65247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l652_65222

theorem trigonometric_equation_solution :
  ∀ x : ℝ, Real.sin x ^ 2 - 2 * Real.sin x * Real.cos x = 3 * Real.cos x ^ 2 ↔
  (∃ k : ℤ, x = -Real.pi / 4 + Real.pi * (k : ℝ)) ∨ 
  (∃ n : ℤ, x = Real.arctan 3 + Real.pi * (n : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l652_65222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l652_65249

noncomputable def z : ℂ := (1 - Complex.I * Real.sqrt 2) / Complex.I

theorem z_in_third_quadrant : 
  Real.sign z.re = -1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l652_65249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l652_65284

noncomputable section

-- Define the complex numbers
def i : ℂ := Complex.I

-- Define z₁ and z₂
noncomputable def z₁ : ℂ := ((-1 : ℂ) + 3*i) / ((1 : ℂ) + 2*i)
noncomputable def z₂ : ℂ := 1 + (1 + i)^10

-- Define points A and B
noncomputable def A : ℂ := z₁
noncomputable def B : ℂ := z₂

-- Theorem statement
theorem distance_AB : Complex.abs (B - A) = 31 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l652_65284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_theorem_l652_65257

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (4, 0)

-- Define the slope of the asymptotes
noncomputable def asymptote_slope : ℝ := Real.sqrt 3 / 3

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -(Real.sqrt 3) * x + 4 * Real.sqrt 3

-- Theorem statement
theorem hyperbola_line_theorem :
  ∀ (x y : ℝ),
  hyperbola x y →
  (∃ (m : ℝ), m * asymptote_slope = -1) →
  line_l (right_focus.fst) (right_focus.snd) →
  line_l x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_theorem_l652_65257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l652_65262

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine
| counterfeit

/-- Represents the result of a weighing -/
inductive WeighingResult
| balanced
| leftHeavier
| rightHeavier

/-- Represents a two-pan balance -/
def Balance := List Coin → List Coin → WeighingResult

/-- A strategy for identifying the counterfeit coin -/
def IdentificationStrategy := List Coin → Balance → Fin 13

/-- Theorem: It's possible to identify the counterfeit coin in three weighings -/
theorem counterfeit_coin_identification
  (coins : List Coin)
  (h_count : coins.length = 13)
  (h_one_counterfeit : ∃! c, c ∈ coins ∧ c = Coin.counterfeit)
  (balance : Balance)
  : ∃ (strategy : IdentificationStrategy),
    ∀ (counterfeit_index : Fin 13),
    coins[counterfeit_index.val]? = some Coin.counterfeit →
    ∃ (w1 w2 w3 : WeighingResult),
    strategy coins balance = counterfeit_index :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l652_65262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l652_65203

-- Define the function f on the interval (1,+∞)
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the conditions
axiom f_domain : ∀ x > 1, f x ≠ 0
axiom f'_condition : ∀ x > 1, x * (f' x) * Real.log x > f x
axiom f_value : f (Real.exp 2) = 2

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | f (Real.exp x) < x}

-- State the theorem
theorem solution_set_eq :
  solution_set = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l652_65203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_4theta_from_exp_l652_65209

theorem cos_4theta_from_exp (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (1 + 2 * Complex.I) / Real.sqrt 5 →
  Real.cos (4 * θ) = -7 / 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_4theta_from_exp_l652_65209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_sum_set_l652_65228

def sum_set (A B : Set ℕ) : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ B ∧ x = a + b}

def target_set : Set ℕ := Finset.range 2021

theorem least_k_for_sum_set : 
  (∃ (k : ℕ) (A B : Finset ℕ), 
    A.card = k ∧ 
    B.card = 2 * k ∧ 
    sum_set A B = target_set) ∧ 
  (∀ (j : ℕ), j < 32 → 
    ¬∃ (A B : Finset ℕ), 
      A.card = j ∧ 
      B.card = 2 * j ∧ 
      sum_set A B = target_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_sum_set_l652_65228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_equals_210_factorial_l652_65211

theorem factorial_sum_equals_210_factorial (n : ℕ) : 
  Nat.factorial (n+1) + Nat.factorial (n+2) + Nat.factorial (n+3) = Nat.factorial n * 210 ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_equals_210_factorial_l652_65211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_value_of_a_in_triangle_abc_l652_65227

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * sin (π / 2 - x) - 2 * cos (π + x) * cos x + 2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

theorem value_of_a_in_triangle_abc (A B C : ℝ) (a b c : ℝ) :
  f A = 4 →
  b = 1 →
  (1/2) * b * c * sin A = sqrt 3 / 2 →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A →
  a = sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_value_of_a_in_triangle_abc_l652_65227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_representation_exists_l652_65260

noncomputable def g (x : ℝ) : ℝ := Real.tan (x / 3) - Real.tan x

theorem g_representation_exists :
  ∃ j : ℝ, ∀ x : ℝ, 
    Real.sin (x / 3) ≠ 0 → Real.sin x ≠ 0 → 
    g x = Real.sin (j * x) / (Real.sin (x / 3) * Real.sin x) ∧ j = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_representation_exists_l652_65260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_compose_8_eq_4_l652_65276

-- Define the function f
def f (n : ℕ) : ℕ := sorry

-- Define the nth_sqrt2_digit function
def nth_sqrt2_digit (n : ℕ) : ℕ := sorry

-- Axiom: f(n) is the nth digit after the decimal point of √2
axiom f_def : ∀ n, f n = nth_sqrt2_digit n

-- Axiom: The first 8 digits after the decimal point of √2 are 41421356
axiom sqrt2_digits : 
  (nth_sqrt2_digit 1 = 4) ∧ 
  (nth_sqrt2_digit 2 = 1) ∧ 
  (nth_sqrt2_digit 3 = 4) ∧ 
  (nth_sqrt2_digit 4 = 2) ∧ 
  (nth_sqrt2_digit 5 = 1) ∧ 
  (nth_sqrt2_digit 6 = 3) ∧ 
  (nth_sqrt2_digit 7 = 5) ∧ 
  (nth_sqrt2_digit 8 = 6)

-- Define the n-fold composition of f
def f_compose : ℕ → (ℕ → ℕ)
| 0 => id
| n + 1 => f ∘ (f_compose n)

-- The theorem to prove
theorem f_2016_compose_8_eq_4 : (f_compose 2016) 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_compose_8_eq_4_l652_65276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l652_65238

/-- An ellipse with center at the origin, major to minor axis ratio of 2:1, and one focus at (0, -2) -/
structure Ellipse where
  /-- The ratio of major to minor axis is 2:1 -/
  axis_ratio : ℚ
  axis_ratio_eq : axis_ratio = 2 / 1
  /-- One focus is at (0, -2) -/
  focus_y : ℚ
  focus_y_eq : focus_y = -2

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt 3 / 2

/-- The standard equation of the ellipse -/
def standard_equation (x y : ℝ) : Prop :=
  y^2 / (16/3) + x^2 / (4/3) = 1

theorem ellipse_properties (e : Ellipse) :
  eccentricity e = Real.sqrt 3 / 2 ∧
  ∀ x y, standard_equation x y ↔ y^2 / (16/3) + x^2 / (4/3) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l652_65238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l652_65294

def M : Set ℝ := {m | -3 < m ∧ m < 2}
def N : Set ℝ := {n | n ∈ Set.range (fun i : ℕ => (i : ℝ)) ∧ -1 < n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l652_65294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l652_65229

def Hyperbola (P : ℝ × ℝ) : Prop :=
  (P.1 ^ 2) - (P.2 ^ 2 / 3) = 1

def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

noncomputable def DistanceSum (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 + 2) ^ 2 + P.2 ^ 2) + Real.sqrt ((P.1 - 2) ^ 2 + P.2 ^ 2)

def DotProduct (P : ℝ × ℝ) : ℝ :=
  (P.1 + 2) * (P.1 - 2) + P.2 * P.2

theorem hyperbola_dot_product (P : ℝ × ℝ) :
  Hyperbola P ∧ P.1 > 0 ∧ DistanceSum P = 10 → DotProduct P = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l652_65229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l652_65243

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (4 * x - 6)

theorem vertical_asymptote_of_f :
  ∃ (x : ℝ), x = 3/2 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (y : ℝ), 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l652_65243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_participants_count_l652_65202

/-- Represents the donation information for a grade --/
structure GradeDonation where
  total : ℝ
  participants : ℕ

/-- Proves that given the donation conditions, the number of participants in each grade is correct --/
theorem donation_participants_count 
  (eighth : GradeDonation)
  (ninth : GradeDonation)
  (h1 : eighth.total = 4800)
  (h2 : ninth.total = 5000)
  (h3 : ninth.participants = eighth.participants + 20)
  (h4 : eighth.total / (eighth.participants : ℝ) = ninth.total / (ninth.participants : ℝ)) :
  eighth.participants = 480 ∧ ninth.participants = 500 := by
  sorry

#check donation_participants_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_participants_count_l652_65202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knicks_knacks_knocks_conversion_l652_65233

/-- Given the conversion rates between knicks, knacks, and knocks,
    prove that 49 knocks are equal to 70/3 knicks. -/
theorem knicks_knacks_knocks_conversion :
  ∀ (knicks_to_knacks knacks_to_knocks : ℚ),
  knicks_to_knacks = 3 / 5 →
  knacks_to_knocks = 7 / 2 →
  49 * (knicks_to_knacks * knacks_to_knocks) = 70 / 3 := by
  intros knicks_to_knacks knacks_to_knocks h1 h2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knicks_knacks_knocks_conversion_l652_65233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l652_65259

/-- The function f(x) defined as a*ln(x) + (1/2)*x^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2

/-- Theorem stating the range of a given the conditions --/
theorem range_of_a (a : ℝ) (h_a_pos : a > 0) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) →
  a ∈ Set.Ici 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l652_65259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_real_root_l652_65207

-- Define the function f(y) = 5 + (3^(y+1) - 7) / 3^(y^2)
noncomputable def f (y : ℝ) : ℝ := 5 + (3^(y+1) - 7) / 3^(y^2)

-- Define the minimum value of f(y) over [-1, 1]
noncomputable def f_min : ℝ := 0.5925003438768

-- Define the maximum value of f(y) over [-1, 1]
noncomputable def f_max : ℝ := 17/3

-- State the theorem
theorem equation_has_real_root (a : ℝ) :
  (∃ x : ℝ, 3^(Real.cos (2*x) + 1) - (a - 5) * 3^(Real.cos (2*x))^2 = 7) ↔ 
  (f_min ≤ a ∧ a ≤ f_max) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_real_root_l652_65207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_in_B_is_16_25_l652_65235

noncomputable def square_A_area : ℝ := 25
noncomputable def square_B_perimeter : ℝ := 12

noncomputable def square_A_side : ℝ := Real.sqrt square_A_area
noncomputable def square_B_side : ℝ := square_B_perimeter / 4

noncomputable def square_B_area : ℝ := square_B_side ^ 2
noncomputable def area_difference : ℝ := square_A_area - square_B_area

noncomputable def probability_not_in_B : ℝ := area_difference / square_A_area

theorem probability_not_in_B_is_16_25 :
  probability_not_in_B = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_in_B_is_16_25_l652_65235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_theorem_sufficient_not_necessary_negation_existential_conjunction_false_l652_65204

-- 1. Contrapositive
theorem contrapositive_theorem (x : ℝ) :
  (x^2 - 1 = 0 → x^2 = 1) ↔ (x^2 ≠ 1 → x^2 - 1 ≠ 0) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∀ x : ℝ, x = 1 → x^2 = x) ∧ (∃ x : ℝ, x^2 = x ∧ x ≠ 1) := by sorry

-- 3. Negation of existential statement
theorem negation_existential :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ)^x > 0) := by sorry

-- 4. Conjunction of propositions
theorem conjunction_false :
  ∃ p q : Prop, (p ∧ q = False) ∧ ¬(p = False ∧ q = False) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_theorem_sufficient_not_necessary_negation_existential_conjunction_false_l652_65204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_box_sand_capacity_l652_65236

/-- Represents the dimensions and sand capacity of a box -/
structure Box where
  height : ℚ
  width : ℚ
  length : ℚ
  sand_capacity : ℚ

/-- Calculates the volume of a box -/
def box_volume (b : Box) : ℚ := b.height * b.width * b.length

/-- Given the dimensions and sand capacity of the first box, 
    calculates the sand capacity of a box with the given volume -/
def sand_capacity_for_volume (first_box : Box) (volume : ℚ) : ℚ :=
  (volume / box_volume first_box) * first_box.sand_capacity

/-- Theorem stating that the third box can hold 90 grams of sand -/
theorem third_box_sand_capacity :
  let first_box : Box := { height := 1, width := 2, length := 4, sand_capacity := 30 }
  let second_box : Box := { height := 3 * first_box.height, 
                            width := 2 * first_box.width, 
                            length := first_box.length,
                            sand_capacity := 0 }
  let third_box : Box := { height := second_box.height, 
                           width := second_box.width / 2, 
                           length := second_box.length,
                           sand_capacity := 0 }
  sand_capacity_for_volume first_box (box_volume third_box) = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_box_sand_capacity_l652_65236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_cosine_sum_l652_65279

-- Define a triangle with sides forming a geometric sequence
structure GeometricTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  geometric_sequence : b^2 = a * c
  angle_sum : A + B + C = Real.pi
  sine_law : a / (Real.sin A) = b / (Real.sin B)

-- State the theorem
theorem geometric_triangle_cosine_sum 
  (t : GeometricTriangle) : 
  Real.cos (2 * t.B) + Real.cos t.B + Real.cos (t.A - t.C) = 
  Real.cos t.B + Real.cos (t.A - t.C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_cosine_sum_l652_65279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_equal_intercepts_line_through_point_with_double_inclination_l652_65291

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def equalIntercepts (l : Line2D) : Prop :=
  l.a * l.c = -l.b * l.c

noncomputable def angleOfInclination (l : Line2D) : ℝ :=
  Real.arctan (-l.a / l.b)

theorem line_through_point_with_equal_intercepts 
  (p : Point2D) 
  (h : p.x = 3 ∧ p.y = 2) :
  ∃ l : Line2D, pointOnLine p l ∧ equalIntercepts l ∧
  ((l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5)) := by
  sorry

theorem line_through_point_with_double_inclination 
  (a : Point2D) 
  (h : a.x = -1 ∧ a.y = -3) :
  ∃ l : Line2D, pointOnLine a l ∧ 
  angleOfInclination l = 2 * angleOfInclination { a := 1, b := -1, c := 0 } ∧
  l.a = 1 ∧ l.b = 0 ∧ l.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_equal_intercepts_line_through_point_with_double_inclination_l652_65291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandth_term_is_45_l652_65269

/-- The sequence where each positive integer n is repeated exactly n times -/
def our_sequence (n : ℕ) : ℕ :=
  (Nat.sqrt (2 * n + 1/4 : ℚ).ceil.toNat) + 1

/-- The 1000th term of the sequence is 45 -/
theorem thousandth_term_is_45 : our_sequence 1000 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandth_term_is_45_l652_65269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_of_f_l652_65242

/-- A cubic function -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The second derivative of a cubic function -/
def cubic_second_derivative (a b : ℝ) (x : ℝ) : ℝ := 6 * a * x + 2 * b

/-- The inflection point of a cubic function -/
def inflection_point (f : ℝ → ℝ) (x : ℝ) : ℝ × ℝ := (x, f x)

/-- The center of symmetry of a cubic function -/
noncomputable def center_of_symmetry (f : ℝ → ℝ) : ℝ × ℝ := sorry

/-- The specific cubic function in the problem -/
noncomputable def f (x : ℝ) : ℝ := x^3 - 3/2 * x^2 + 3 * x - 1/4

theorem center_of_symmetry_of_f :
  center_of_symmetry f = (1/2, 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_of_f_l652_65242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l652_65267

theorem inequality_equivalence (x : ℝ) : 
  x^2 - 3 * Real.sqrt (x^2 + 3) ≤ 1 ↔ -Real.sqrt 13 ≤ x ∧ x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l652_65267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l652_65234

def M : Set ℝ := {x : ℝ | x^2 ≤ x}
def N : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) < 1}

theorem intersection_complement_equality : M ∩ Nᶜ = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l652_65234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_exists_2x2_square_l652_65281

/-- Represents a rectangle on the board -/
structure Rectangle where
  x : Fin 8
  y : Fin 8
  width : Nat
  height : Nat

/-- Represents the 8x8 board -/
def Board := Fin 8 → Fin 8 → Bool

/-- Checks if a 2x2 square exists at the given coordinates -/
def exists_2x2_square (board : Board) (x y : Fin 8) : Prop :=
  board x y ∧ board x (y + 1) ∧ board (x + 1) y ∧ board (x + 1) (y + 1)

/-- The main theorem to be proved -/
theorem always_exists_2x2_square 
  (initial_board : Board) 
  (cut_rectangles : List Rectangle) 
  (h1 : ∀ x y, initial_board x y = true) 
  (h2 : cut_rectangles.length = 8) 
  (h3 : ∀ r ∈ cut_rectangles, r.width = 2 ∧ r.height = 1) :
  ∃ x y, exists_2x2_square 
    (fun x y ↦ initial_board x y ∧ 
      ∀ r ∈ cut_rectangles, ¬(x ≥ r.x ∧ x < r.x + r.width ∧ y ≥ r.y ∧ y < r.height)) 
    x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_exists_2x2_square_l652_65281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_of_N_l652_65216

/-- Represents the number of smaller triangles in the large triangle -/
def num_triangles : ℕ := 13

/-- Represents the number of available colors -/
def num_colors : ℕ := 3

/-- Represents the number of ways to color the triangles -/
def N : ℕ := 5184

/-- Theorem stating that the number of positive integer divisors of N is 35 -/
theorem num_divisors_of_N : 
  (Finset.filter (fun d ↦ N % d = 0) (Finset.range (N + 1))).card = 35 := by
  sorry

/-- Lemma: N is equal to 2^6 * 3^4 -/
lemma N_factorization : N = 2^6 * 3^4 := by
  rfl

/-- Lemma: The number of divisors of 2^6 * 3^4 is (6+1)*(4+1) = 35 -/
lemma num_divisors_calculation :
  (Finset.filter (fun d ↦ (2^6 * 3^4) % d = 0) (Finset.range ((2^6 * 3^4) + 1))).card = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_of_N_l652_65216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l652_65251

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)

-- Define the transformed reference function
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + 5 * Real.pi / 6) + 1

-- Theorem statement
theorem function_equivalence : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l652_65251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l652_65213

open Polynomial

-- Define the polynomials g and e
noncomputable def g : ℝ[X] := 3 * X^4 + 9 * X^3 - 7 * X^2 + 2 * X + 5
noncomputable def e : ℝ[X] := X^2 + 2 * X - 3

-- State the theorem
theorem polynomial_division_result :
  ∃ (s t : ℝ[X]), g = s * e + t ∧ degree t < degree e → eval 1 s + eval (-1) t = -22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_result_l652_65213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l652_65237

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem tangent_slope_at_origin :
  (deriv f) 0 = 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l652_65237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p4_wins_fourth_is_p4_l652_65278

/-- Represents a player in the Jai Alai game -/
inductive Player : Type
  | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8

/-- The game of Jai Alai with 8 players -/
structure JaiAlaiGame where
  players : List Player
  winner_score : Nat
  total_score : Nat

/-- The specific game instance we're analyzing -/
def our_game : JaiAlaiGame :=
  { players := [Player.P1, Player.P2, Player.P3, Player.P4, Player.P5, Player.P6, Player.P7, Player.P8]
  , winner_score := 7
  , total_score := 37 }

/-- The theorem stating that P4 wins the game -/
theorem p4_wins (g : JaiAlaiGame) (h1 : g = our_game) : 
  ∃ (x r : Nat), x + 7 * r + 7 = g.total_score ∧ x = 2 ∧ r = 4 := by
  sorry

/-- Helper function to get the fourth player -/
def fourth_player (g : JaiAlaiGame) : Option Player :=
  g.players.get? 3

/-- Theorem stating that the fourth player is P4 -/
theorem fourth_is_p4 (g : JaiAlaiGame) (h1 : g = our_game) :
  fourth_player g = some Player.P4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p4_wins_fourth_is_p4_l652_65278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_difference_specific_fill_time_difference_l652_65239

/-- Represents a water tank with a given capacity and inflow rate -/
structure Tank where
  capacity : ℝ
  inflow_rate : ℝ

/-- Calculates the time needed to fill a tank -/
noncomputable def fill_time (tank : Tank) : ℝ :=
  tank.capacity / tank.inflow_rate

/-- Proves that the difference in filling time between two tanks with the same capacity
    but different inflow rates is equal to the difference of their individual filling times -/
theorem fill_time_difference (tank_a tank_b : Tank) 
    (h_capacity : tank_a.capacity = tank_b.capacity) :
    fill_time tank_a - fill_time tank_b = 
    tank_a.capacity * (1 / tank_a.inflow_rate - 1 / tank_b.inflow_rate) := by
  sorry

/-- Proves that for the specific case of two 20-liter tanks with inflow rates of 2 and 4 liters per hour,
    the difference in filling time is 5 hours -/
theorem specific_fill_time_difference :
    let tank_a : Tank := ⟨20, 2⟩
    let tank_b : Tank := ⟨20, 4⟩
    fill_time tank_a - fill_time tank_b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_difference_specific_fill_time_difference_l652_65239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_A_complement_A_intersect_B_range_of_a_l652_65274

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5) - 1 / Real.sqrt (8 - x)

-- Define set A (domain of f)
def A : Set ℝ := {x | 5 ≤ x ∧ x < 8}

-- Define set B
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ 3 < n ∧ n < 11}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

theorem domain_of_f_is_A : Set.range f = A := by sorry

theorem complement_A_intersect_B : 
  (Set.compl A ∩ B) = {4, 8, 9, 10} := by sorry

theorem range_of_a (h : A ∪ C a = Set.univ) : 
  5 ≤ a ∧ a < 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_A_complement_A_intersect_B_range_of_a_l652_65274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_extrema_l652_65230

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def distance_to_line (p : Point) (l : Line) : ℝ := 
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

noncomputable def sum_of_distances (A B : Point) (l : Line) : ℝ :=
  distance_to_line A l + distance_to_line B l

def is_altitude (l : Line) (A B O : Point) : Prop := sorry
def is_side (l : Line) (A B O : Point) : Prop := sorry
def is_perpendicular_to_median (l : Line) (A B O : Point) : Prop := sorry
def is_parallel_to_side (l : Line) (A B O : Point) : Prop := sorry

def collinear (A B C : Point) : Prop := 
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

-- Theorem statement
theorem distance_sum_extrema 
  (A B O : Point) 
  (h_not_collinear : ¬ collinear A B O) :
  ∃ (l_max l_min : Line),
    (l_max.a * O.x + l_max.b * O.y + l_max.c = 0) ∧
    (l_min.a * O.x + l_min.b * O.y + l_min.c = 0) ∧
    (∀ l : Line, l.a * O.x + l.b * O.y + l.c = 0 →
      sum_of_distances A B l ≤ sum_of_distances A B l_max) ∧
    (∀ l : Line, l.a * O.x + l.b * O.y + l.c = 0 →
      sum_of_distances A B l_min ≤ sum_of_distances A B l) ∧
    (is_altitude l_max A B O ∨ is_side l_max A B O) ∧
    (is_perpendicular_to_median l_min A B O ∨ is_parallel_to_side l_min A B O) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_extrema_l652_65230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l652_65221

noncomputable def IsEquilateralTriangle (triangle : ℂ × ℂ × ℂ) : Prop :=
  let (a, b, c) := triangle
  Complex.abs (a - b) = Complex.abs (b - c) ∧ 
  Complex.abs (b - c) = Complex.abs (c - a) ∧
  Complex.abs (c - a) = Complex.abs (a - b)

theorem equilateral_triangle_lambda (z : ℂ) (l : ℝ) 
  (h1 : Complex.abs z = 3)
  (h2 : l > 1)
  (h3 : IsEquilateralTriangle (z, z^2, l • z)) : 
  l = (1 + Real.sqrt 33) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l652_65221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_at_origin_l652_65275

noncomputable section

-- Define the lines l₁ and l₂
def l₁ : ℝ → ℝ := λ _ => 2
def l₂ : ℝ → ℝ := λ _ => 4

-- Define the exponential functions
def f₁ : ℝ → ℝ := λ x => (3 : ℝ)^x
def f₂ : ℝ → ℝ := λ x => (5 : ℝ)^x

-- Define the intersection points
def A : ℝ × ℝ := (Real.log 2 / Real.log 3, 2)
def B : ℝ × ℝ := (Real.log 4 / Real.log 3, 4)
def C : ℝ × ℝ := (Real.log 2 / Real.log 5, 2)
def D : ℝ × ℝ := (Real.log 4 / Real.log 5, 4)

-- Define the lines AB and CD
def line_AB : ℝ → ℝ := λ x => (B.2 - A.2) / (B.1 - A.1) * (x - A.1) + A.2
def line_CD : ℝ → ℝ := λ x => (D.2 - C.2) / (D.1 - C.1) * (x - C.1) + C.2

theorem intersection_at_origin :
  ∃ (x y : ℝ), x = 0 ∧ y = 0 ∧ line_AB x = y ∧ line_CD x = y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_at_origin_l652_65275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l652_65205

theorem theta_value (θ : Real) 
  (h1 : Real.sin (π + θ) = -Real.sqrt 3 * Real.cos (2 * π - θ))
  (h2 : abs θ < π / 2) : 
  θ = π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l652_65205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_8_factorial_l652_65206

theorem number_of_divisors_8_factorial : 
  (Finset.card (Nat.divisors (Nat.factorial 8))) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_8_factorial_l652_65206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_distance_l652_65273

/-- The distance between intersection points of an ellipse and a parabola with shared focus -/
theorem ellipse_parabola_intersection_distance : 
  ∀ (e p : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ e ↔ x^2/36 + y^2/16 = 1) →  -- Ellipse equation
  (∃ (f : ℝ × ℝ), f ∈ e ∧ f ∈ p) →  -- Shared focus
  (∃ (d : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ d ↔ x = 2*Real.sqrt 5) ∧  -- Directrix of parabola
    (∀ (x y : ℝ), (x, y) ∈ p ↔ (x + Real.sqrt 5)^2 = 5*y^2)) →  -- Parabola equation
  (∃! (i₁ i₂ : ℝ × ℝ), i₁ ∈ e ∧ i₁ ∈ p ∧ i₂ ∈ e ∧ i₂ ∈ p ∧ i₁ ≠ i₂) →  -- Two intersection points
  ∃ (i₁ i₂ : ℝ × ℝ), i₁ ∈ e ∧ i₁ ∈ p ∧ i₂ ∈ e ∧ i₂ ∈ p ∧ 
    Real.sqrt ((i₁.1 - i₂.1)^2 + (i₁.2 - i₂.2)^2) = 2*Real.sqrt 140/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_distance_l652_65273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l652_65264

/-- The complex number z is defined as 2/(1-i) - 2i³, where i is the imaginary unit -/
noncomputable def z : ℂ := 2 / (1 - Complex.I) - 2 * Complex.I^3

/-- A complex number is in the first quadrant if its real part is positive and its imaginary part is positive -/
def is_in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

theorem z_in_first_quadrant : is_in_first_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l652_65264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_of_tangent_roots_l652_65297

theorem cos_difference_of_tangent_roots (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi) 
  (h2 : 0 < β ∧ β < Real.pi)
  (h3 : (Real.tan α)^2 + 3*(Real.tan α) + 1 = 0)
  (h4 : (Real.tan β)^2 + 3*(Real.tan β) + 1 = 0) :
  Real.cos (α - β) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_of_tangent_roots_l652_65297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_discard_rate_l652_65244

noncomputable def vendor_pattern (initial_pears : ℝ) (first_day_discard_rate : ℝ) : ℝ :=
  let remaining_after_first_sale := 0.2 * initial_pears
  let discarded_first_day := first_day_discard_rate * remaining_after_first_sale
  let remaining_after_first_discard := remaining_after_first_sale - discarded_first_day
  let remaining_after_second_sale := 0.2 * remaining_after_first_discard
  (discarded_first_day + remaining_after_second_sale) / initial_pears

theorem vendor_discard_rate :
  ∀ initial_pears : ℝ, initial_pears > 0 →
    ∃ first_day_discard_rate : ℝ,
      0 ≤ first_day_discard_rate ∧ first_day_discard_rate ≤ 1 ∧
      vendor_pattern initial_pears first_day_discard_rate = 0.12 ∧
      first_day_discard_rate = 0.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_discard_rate_l652_65244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_screen_area_difference_18_16_l652_65220

/-- The area of a square screen given its diagonal length -/
noncomputable def screen_area (diagonal : ℝ) : ℝ := diagonal^2 / 2

/-- The difference in area between two square screens given their diagonal lengths -/
noncomputable def screen_area_difference (diagonal1 diagonal2 : ℝ) : ℝ :=
  screen_area diagonal1 - screen_area diagonal2

theorem screen_area_difference_18_16 :
  screen_area_difference 18 16 = 34 := by
  -- Unfold the definitions
  unfold screen_area_difference
  unfold screen_area
  -- Simplify the expression
  simp [pow_two]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_screen_area_difference_18_16_l652_65220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l652_65270

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  -- The area of the trapezoid
  area : ℝ
  -- Assumption that the area is positive
  area_pos : area > 0

/-- The height of an isosceles trapezoid with perpendicular diagonals -/
noncomputable def trapezoidHeight (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt t.area

/-- Theorem: The height of an isosceles trapezoid with perpendicular diagonals
    is equal to the square root of its area -/
theorem isosceles_trapezoid_height (t : IsoscelesTrapezoid) :
  trapezoidHeight t = Real.sqrt t.area := by
  -- Unfold the definition of trapezoidHeight
  unfold trapezoidHeight
  -- The result follows immediately from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l652_65270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_remainder_61_7_l652_65295

def divisors_with_remainder (m n r : ℕ) : Finset ℕ :=
  (Finset.range m).filter (fun d => d > r ∧ m % d = r)

theorem count_divisors_with_remainder_61_7 :
  (divisors_with_remainder 61 61 7).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_remainder_61_7_l652_65295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_l652_65250

/-- Given a bond with face value, interest rate on face value, and interest rate as a percentage of selling price, calculate the selling price of the bond. -/
theorem bond_selling_price (face_value : ℝ) (interest_rate_face : ℝ) (interest_rate_selling : ℝ) :
  face_value = 5000 →
  interest_rate_face = 0.07 →
  interest_rate_selling = 0.065 →
  ∃ (selling_price : ℝ), 
    (face_value * interest_rate_face) = (selling_price * interest_rate_selling) ∧
    (abs (selling_price - 5384.62) < 0.01) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_l652_65250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_is_correct_l652_65288

noncomputable section

-- Define the currency exchange rates
def usd_per_eur : ℝ := 1.1
def usd_per_gbp : ℝ := 1.4

-- Define the costs and tax rates for each item
def item1_cost : ℝ := 40
def item1_tax_rate : ℝ := 0.05
def item1_discount_rate : ℝ := 0.10

def item2_cost : ℝ := 70
def item2_tax_rate : ℝ := 0.08
def item2_coupon : ℝ := 5

def item3_cost : ℝ := 100
def item3_tax_rate : ℝ := 0.06
def item3_discount_rate : ℝ := 0.15

-- Define the number of people splitting the cost
def num_people : ℕ := 5

-- Define the function to calculate the final cost per person
noncomputable def final_cost_per_person : ℝ :=
  let item1_final := item1_cost * (1 + item1_tax_rate) * (1 - item1_discount_rate)
  let item2_final := (item2_cost * (1 + item2_tax_rate) - item2_coupon) * usd_per_eur
  let item3_final := item3_cost * (1 + item3_tax_rate) * (1 - item3_discount_rate) * usd_per_gbp
  let total_cost := item1_final + item2_final + item3_final
  total_cost / num_people

-- Theorem to prove
theorem final_cost_is_correct : 
  final_cost_per_person = 48.32 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_is_correct_l652_65288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_geometric_proofs_count_l652_65241

/-- A class with two elective topics --/
structure ClassInfo where
  total_students : ℕ
  geometric_proofs : Finset ℕ
  polar_coordinates : Finset ℕ

/-- The conditions of the problem --/
def problem_conditions (c : ClassInfo) : Prop :=
  c.total_students = 54 ∧
  (c.geometric_proofs ∪ c.polar_coordinates).card = c.total_students ∧
  (c.geometric_proofs ∩ c.polar_coordinates).card = 6 ∧
  c.polar_coordinates.card = c.geometric_proofs.card + 8

/-- The theorem to prove --/
theorem only_geometric_proofs_count (c : ClassInfo) 
  (h : problem_conditions c) : 
  (c.geometric_proofs \ c.polar_coordinates).card = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_geometric_proofs_count_l652_65241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l652_65215

open Set Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- State the theorem
theorem solution_set_theorem (h1 : ∀ x ∈ domain, x^2 * (deriv f x) + 1 > 0) 
                             (h2 : f 1 = 5) : 
  {x ∈ domain | f x < 1/x + 4} = Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l652_65215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l652_65283

open Real Set

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * cos (2 * x - π / 4)

-- Define the domain
def domain : Set ℝ := Icc (-π / 2) (π / 2)

-- State the theorem
theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), a = -3 * π / 8 ∧ b = π / 8 ∧
  (∀ x y, x ∈ domain → y ∈ domain → a ≤ x → x < y → y ≤ b → f x < f y) ∧
  (∀ c d, c < a ∨ b < d → ∃ x y, x ∈ domain ∧ y ∈ domain ∧ c ≤ x ∧ x < y ∧ y ≤ d ∧ f y ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l652_65283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosines_l652_65282

theorem right_triangle_cosines (D E F : Real) (h_right : D = 90) (h_de : E = 9) (h_df : F = 15) :
  let cos_F := (3 * Real.sqrt 34) / 34
  let cos_D := 0
  (Real.cos F = cos_F) ∧ (Real.cos D = cos_D) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosines_l652_65282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_holds_for_2013_l652_65293

theorem proposition_holds_for_2013 (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, ¬P (k + 1) → ¬P k)
  (h2 : P 2012) : 
  P 2013 := by
  by_contra h
  have h3 : ¬P 2012 := h1 2012 h
  contradiction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_holds_for_2013_l652_65293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polyhedron_space_diagonals_l652_65254

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A specific convex polyhedron Q has 335 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_polyhedron_space_diagonals_l652_65254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt2_over_2_sufficient_not_necessary_l652_65223

theorem sin_sqrt2_over_2_sufficient_not_necessary :
  (∃ α : ℝ, Real.sin α = Real.sqrt 2 / 2 → Real.cos (2 * α) = 0) ∧
  (∃ α : ℝ, Real.cos (2 * α) = 0 ∧ Real.sin α ≠ Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt2_over_2_sufficient_not_necessary_l652_65223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_line_symmetry_half_line_no_symmetry_line_segment_symmetry_l652_65224

-- Define the types for our geometric objects
def InfiniteLine : Type := ℝ → Prop
def HalfLine : Type := ℝ → Prop
def LineSegment : Type := ℝ × ℝ

-- Define what it means to be a center of symmetry
def IsCenterOfSymmetry (line : InfiniteLine) (point : ℝ) : Prop :=
  ∀ p : ℝ, line p → ∃ p' : ℝ, line p' ∧ point = (p + p') / 2

-- Theorem for infinite line
theorem infinite_line_symmetry (line : InfiniteLine) :
  ∀ point : ℝ, line point → IsCenterOfSymmetry line point := by
  sorry

-- Theorem for half-line
theorem half_line_no_symmetry (halfLine : HalfLine) :
  ¬∃ point : ℝ, halfLine point ∧ IsCenterOfSymmetry halfLine point := by
  sorry

-- Define midpoint of a line segment
noncomputable def Midpoint (segment : LineSegment) : ℝ :=
  (segment.1 + segment.2) / 2

-- Theorem for line segment
theorem line_segment_symmetry (segment : LineSegment) :
  IsCenterOfSymmetry (λ x => x ≥ segment.1 ∧ x ≤ segment.2) (Midpoint segment) ∧
  ∀ point : ℝ, point ≠ Midpoint segment →
    ¬IsCenterOfSymmetry (λ x => x ≥ segment.1 ∧ x ≤ segment.2) point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_line_symmetry_half_line_no_symmetry_line_segment_symmetry_l652_65224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_mileage_proof_l652_65287

/-- Calculates the mileage of a car in kilometers per gallon -/
noncomputable def mileage (distance : ℝ) (gasoline : ℝ) : ℝ :=
  distance / gasoline

theorem car_mileage_proof (distance : ℝ) (gasoline : ℝ) 
  (h1 : distance = 190) 
  (h2 : gasoline = 4.75) : 
  mileage distance gasoline = 40 := by
  -- Unfold the definition of mileage
  unfold mileage
  -- Rewrite using the hypotheses
  rw [h1, h2]
  -- Perform the division
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_mileage_proof_l652_65287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_false_l652_65266

theorem inverse_prop_false : 
  ¬(∀ θ : ℝ, (π / 2 < θ ∧ θ < π) → 
    (Real.sin θ * (1 - 2 * (Real.cos (θ / 2))^2) ≤ 0 → 
      ¬(π / 2 < θ ∧ θ < π))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_false_l652_65266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_ratio_l652_65201

/-- Represents the partnership profit distribution problem -/
theorem partnership_profit_ratio 
  (total_profit : ℕ) 
  (ramesh_xyz_ratio : ℚ) 
  (rajeev_share : ℕ) 
  (h1 : total_profit = 36000)
  (h2 : ramesh_xyz_ratio = 5 / 4)
  (h3 : rajeev_share = 12000) :
  ∃ (xyz_share : ℕ), 
    (xyz_share : ℚ) / rajeev_share = 8 / 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_ratio_l652_65201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l652_65256

-- Define the ⊙ operator
def odot (a b : ℕ) : ℕ := a^3 - b

-- State the theorem
theorem odot_calculation :
  (5^(odot 6 23 : ℕ)) - (2^(odot 8 17 : ℕ)) = 5^193 - 2^495 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l652_65256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_calculation_l652_65286

/-- Calculates the actual distance between two cities given the map distance and scale. -/
noncomputable def actual_distance (map_distance : ℝ) (scale_inches : ℝ) (scale_miles : ℝ) : ℝ :=
  map_distance * (scale_miles / scale_inches)

/-- Theorem: Given a map distance of 20 inches between two cities and a scale where 0.5 inches
    represents 10 miles, the actual distance between the cities is 400 miles. -/
theorem city_distance_calculation :
  let map_distance := (20 : ℝ)
  let scale_inches := (0.5 : ℝ)
  let scale_miles := (10 : ℝ)
  actual_distance map_distance scale_inches scale_miles = 400 := by
  -- Unfold the definition of actual_distance
  unfold actual_distance
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_calculation_l652_65286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_comparison_l652_65292

/-- Represents a square divided into regions -/
structure DividedSquare where
  totalArea : ℝ
  shadedArea : ℝ

/-- Square I divided by diagonals -/
noncomputable def squareI : DividedSquare :=
  { totalArea := 1
  , shadedArea := 1/2 }

/-- Square II divided by midpoint lines -/
noncomputable def squareII : DividedSquare :=
  { totalArea := 1
  , shadedArea := 1/4 }

/-- Square III divided by diagonals and midpoint lines -/
noncomputable def squareIII : DividedSquare :=
  { totalArea := 1
  , shadedArea := 1/2 }

/-- Theorem stating the relationship between shaded areas -/
theorem shaded_area_comparison :
  squareI.shadedArea ≠ squareII.shadedArea ∧
  squareII.shadedArea ≠ squareIII.shadedArea ∧
  squareI.shadedArea = squareIII.shadedArea := by
  sorry

#check shaded_area_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_comparison_l652_65292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l652_65261

theorem tan_double_angle_special_case (α : Real) 
  (h1 : α > Real.pi ∧ α < 2*Real.pi) 
  (h2 : Real.cos α = -Real.sqrt 5 / 5) : 
  Real.tan (2*α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l652_65261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_through_point_l652_65255

/-- If the terminal side of angle α passes through point (1,-2), then tan α = -2 -/
theorem tan_alpha_through_point (α : ℝ) :
  (∃ (t : ℝ), t > 0 ∧ t * (Real.cos α) = 1 ∧ t * (Real.sin α) = -2) →
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_through_point_l652_65255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_bias_l652_65225

-- Define the total number of balls
def total_balls : ℕ := 100

-- Define the set of ball numbers
def ball_numbers : Finset ℕ := Finset.range total_balls

-- Define the set of odd-numbered balls
def odd_balls : Finset ℕ := ball_numbers.filter (fun n => n % 2 = 1)

-- Define the set of even-numbered balls
def even_balls : Finset ℕ := ball_numbers.filter (fun n => n % 2 = 0)

-- Define the probability of selecting an odd-numbered ball
def prob_odd : ℚ := 2/3

-- Theorem stating that the selection process is biased towards odd numbers
theorem selection_bias (h : prob_odd = 2/3) :
  ∃ (bias : ℚ), bias > 1 ∧ 
  (bias * odd_balls.card : ℚ) / ball_numbers.card = prob_odd :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_bias_l652_65225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_g_max_min_ab_value_l652_65253

/-- The function to be minimized -/
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

/-- The function whose max and min values we're interested in -/
def g (x : ℝ) : ℝ := |x - 3| - |x - 2| + |x - 1|

/-- Theorem for the first part of the problem -/
theorem f_min_g_max_min :
  ∃ x₀ : ℝ, (∀ x : ℝ, f x₀ ≤ f x) →
  (∃ x₁ : ℝ, f x₁ = f x₀ ∧ g x₁ = 2) ∧
  (∃ x₂ : ℝ, f x₂ = f x₀ ∧ g x₂ = 1) ∧
  (∀ x : ℝ, f x = f x₀ → 1 ≤ g x ∧ g x ≤ 2) :=
sorry

/-- Theorem for the second part of the problem -/
theorem ab_value (a b : ℤ) (ha : a > 0) (hb : b > 0) :
  |b - 2| + b - 2 = 0 →
  |a - b| + a - b = 0 →
  a * b = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_g_max_min_ab_value_l652_65253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coefficient_binomial_expansion_l652_65214

theorem max_coefficient_binomial_expansion (n : ℕ) 
  (h : Nat.choose n 1 + Nat.choose n 2 = 21) : 
  Finset.sup (Finset.range (n + 1)) (λ i => Nat.choose n i) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coefficient_binomial_expansion_l652_65214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisibility_and_product_l652_65248

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem four_digit_divisibility_and_product :
  let numbers := [3412, 3459, 3471, 3582, 3615]
  (∃! n, n ∈ numbers ∧ ¬is_divisible_by_3 n) ∧
  (∀ n, n ∈ numbers → ¬is_divisible_by_3 n → units_digit n * tens_digit n = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisibility_and_product_l652_65248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l652_65298

noncomputable def sequence_a (n : ℕ) : ℝ := n

noncomputable def sum_s (n : ℕ) : ℝ := (1/2 : ℝ) * n^2 + (1/2 : ℝ) * n

noncomputable def sequence_T (n : ℕ) : ℝ := (3/4 : ℝ) - (1/2 : ℝ) * ((1 / (n + 1 : ℝ)) + (1 / (n + 2 : ℝ)))

theorem sequence_properties (n : ℕ) :
  (∀ k : ℕ, sum_s k = (1/2 : ℝ) * k^2 + (1/2 : ℝ) * k) →
  (sequence_a n = n) ∧
  (sequence_T n = (3/4 : ℝ) - (1/2 : ℝ) * ((1 / (n + 1 : ℝ)) + (1 / (n + 2 : ℝ)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l652_65298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromes_l652_65217

/-- A digit is a natural number from 0 to 9. -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- A 5-digit palindrome is a number in the form abcba where a, b, c are digits and a ≠ 0. -/
def FiveDigitPalindrome : Type :=
  { n : ℕ // ∃ (a b c : Digit), a.val ≠ 0 ∧ n = a.val * 10000 + b.val * 1000 + c.val * 100 + b.val * 10 + a.val }

instance : Fintype FiveDigitPalindrome := by
  sorry

/-- The number of 5-digit palindromes is 900. -/
theorem count_five_digit_palindromes : Fintype.card FiveDigitPalindrome = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_palindromes_l652_65217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_F_value_l652_65245

-- Define the cube structure
structure Cube where
  vertices : Fin 8 → ℚ  -- Changed to ℚ (rational numbers) for computability

-- Define the average function
def average (c : Cube) (v : Fin 8) : ℚ :=
  match v with
  | 0 => (c.vertices 1 + c.vertices 3 + c.vertices 4) / 3  -- A
  | 1 => (c.vertices 0 + c.vertices 2 + c.vertices 5) / 3  -- B
  | 2 => (c.vertices 1 + c.vertices 3 + c.vertices 6) / 3  -- C
  | 3 => (c.vertices 0 + c.vertices 2 + c.vertices 7) / 3  -- D
  | 4 => (c.vertices 0 + c.vertices 5 + c.vertices 7) / 3  -- E
  | 5 => (c.vertices 1 + c.vertices 4 + c.vertices 6) / 3  -- F
  | 6 => (c.vertices 2 + c.vertices 5 + c.vertices 7) / 3  -- G
  | 7 => (c.vertices 3 + c.vertices 4 + c.vertices 6) / 3  -- H

-- Theorem statement
theorem vertex_F_value (c : Cube) 
  (h0 : average c 0 = 1)
  (h1 : average c 1 = 2)
  (h2 : average c 2 = 3)
  (h3 : average c 3 = 4)
  (h4 : average c 4 = 5)
  (h5 : average c 5 = 6)
  (h6 : average c 6 = 7)
  (h7 : average c 7 = 8) :
  c.vertices 5 = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_F_value_l652_65245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_property_l652_65231

/-- A function that returns the digits of a three-digit number -/
def digits (n : Nat) : Fin 3 → Nat :=
  fun i => (n / (10 ^ (2 - i.val))) % 10

/-- A function that returns the number formed by erasing the first digit -/
def erase_first (n : Nat) : Nat :=
  n % 100

/-- A function that returns the number formed by erasing the middle digit -/
def erase_middle (n : Nat) : Nat :=
  (n / 100) * 10 + n % 10

theorem three_digit_number_property (n : Nat) :
  (100 ≤ n ∧ n < 1000) →
  (7 * erase_first n = n ↔ n = 350) ∧
  (6 * erase_middle n = n ↔ n = 108) := by
  sorry

#check three_digit_number_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_property_l652_65231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l652_65299

def is_valid_subset (M : Finset Nat) : Prop :=
  M ⊆ Finset.range 2008 ∧
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M →
    (a ∣ b) ∨ (b ∣ c) ∨ (c ∣ a) ∨ (a ∣ c) ∨ (b ∣ a) ∨ (c ∣ b)

theorem max_valid_subset_size :
  ∃ M : Finset Nat, is_valid_subset M ∧ M.card = 21 ∧
  ∀ N : Finset Nat, is_valid_subset N → N.card ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l652_65299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_with_100_unit_dist_points_l652_65258

/-- The main theorem -/
theorem exists_set_with_100_unit_dist_points 
  {α : Type*} [MetricSpace α] : 
  ∃ (S : Set α), Set.Finite S ∧ 
    ∀ p ∈ S, ∃! (points : Finset α), 
      points.card = 100 ∧ 
      ∀ q ∈ points, q ∈ S ∧ q ≠ p ∧ dist p q = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_with_100_unit_dist_points_l652_65258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_g_to_f_l652_65271

noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem transform_g_to_f :
  (∀ x, f x = g (2 * (x - Real.pi / 8))) ∧
  (∀ x, f x = g (2 * x - Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_g_to_f_l652_65271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l652_65263

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 10 * (Real.cos t) ^ 3
noncomputable def y (t : ℝ) : ℝ := 10 * (Real.sin t) ^ 3

-- Define the domain
def domain : Set ℝ := { t | 0 ≤ t ∧ t ≤ Real.pi / 2 }

-- State the theorem
theorem arc_length_of_curve :
  ∫ t in domain, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l652_65263
