import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_triangle_l849_84949

theorem golden_ratio_triangle (a b c : ℝ) (θ : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/a)^2 + (1/b)^2 = (1/c)^2 →
  θ = min (Real.arcsin (a/c)) (Real.arcsin (b/c)) →
  Real.sin θ = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_triangle_l849_84949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l849_84991

open Polynomial

theorem polynomial_division_remainder : ∃ (q : Polynomial ℂ), 
  X^60 + 2*X^45 + 3*X^30 + 4*X^15 + (5 : Polynomial ℂ) = 
  (X^5 + X^4 + X^3 + X^2 + X + 1) * q + 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l849_84991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l849_84948

theorem subset_count (k : ℕ) (h : k ≥ 2) :
  let S := Finset.range (2 * k)
  (Finset.filter (λ M : Finset ℕ ↦
    M.Nonempty ∧
    (∀ x ∈ M, x < 2 * k) ∧
    (∀ x ∈ M, (2 * k - x - 1) ∈ M)) (Finset.powerset S)).card = 2^k - 1 := by
  sorry

#check subset_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_l849_84948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_function_value_l849_84937

/-- Given a function f(x) = tan(ωx + π/3) where ω > 0, and the distance between
    consecutive intersection points of f(x) and y = 2016 is 3π, then f(π) = -√3. -/
theorem intersection_points_imply_function_value (ω : ℝ) (h₁ : ω > 0) :
  (∃ x₁ x₂ : ℝ, x₂ - x₁ = 3 * π ∧
    Real.tan (ω * x₁ + π / 3) = 2016 ∧
    Real.tan (ω * x₂ + π / 3) = 2016) →
  Real.tan (ω * π + π / 3) = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_function_value_l849_84937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_nine_l849_84958

noncomputable section

-- Define the point P in the first quadrant
variable (a b : ℝ)

-- Define the conditions
def first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0
def on_line (a b : ℝ) : Prop := a + 2 * b - 1 = 0

-- Define the expression
def f (a b : ℝ) : ℝ := 4 / (a + b) + 1 / b

-- State the theorem
theorem min_value_is_nine : 
  ∀ a b : ℝ, first_quadrant a b → on_line a b → f a b ≥ 9 :=
by
  -- The proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_nine_l849_84958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_value_l849_84985

/-- 
Given an acute triangle ABC with:
- sin A = 2√2/3
- side a = 2
- area of triangle ABC = √2
Prove that side b = √3
-/
theorem side_b_value (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  Real.sin A = 2 * Real.sqrt 2 / 3 →
  a = 2 →
  S = Real.sqrt 2 →
  S = 1/2 * b * c * Real.sin A →  -- Area formula
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine rule
  b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_value_l849_84985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_eye_count_l849_84960

/-- The total number of eyes given specific counts of boys, girls, cats, and spiders with varying eye counts. -/
theorem total_eye_count : 
  let num_boys := 23
  let num_one_eyed_boys := 2
  let num_girls := 18
  let num_one_eyed_girls := 3
  let num_cats := 10
  let num_one_eyed_cats := 2
  let num_spiders := 5
  let num_six_eyed_spiders := 1
  
  let boys_eyes := (num_boys - num_one_eyed_boys) * 2 + num_one_eyed_boys
  let girls_eyes := (num_girls - num_one_eyed_girls) * 2 + num_one_eyed_girls
  let cats_eyes := (num_cats - num_one_eyed_cats) * 2 + num_one_eyed_cats
  let spiders_eyes := (num_spiders - num_six_eyed_spiders) * 8 + num_six_eyed_spiders * 6

  boys_eyes + girls_eyes + cats_eyes + spiders_eyes = 133 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_eye_count_l849_84960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l849_84972

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (a b : Line) 
  (α β : Plane) 
  (h_distinct_lines : a ≠ b) 
  (h_distinct_planes : α ≠ β) :
  (∀ (a : Line) (α β : Plane), perpendicular a α → perpendicular a β → planeParallel α β) ∧
  (∀ (α β : Plane), planePerpendicular α β → ∃ (γ : Plane), planePerpendicular γ α ∧ planePerpendicular γ β) ∧
  (∀ (α β : Plane), planePerpendicular α β → ∃ (l : Line), perpendicular l α ∧ lineParallelPlane l β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l849_84972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_equals_64_l849_84954

/-- The function f(x) = 2x^2 + a/x, where a is a constant -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 + a / x

/-- The constant a is defined such that f(-1) = -30 -/
def a : ℝ := 32

theorem sum_of_max_min_equals_64 :
  let f_a := f a
  let max_val := sSup {y | ∃ x ∈ Set.Icc 1 4, f_a x = y}
  let min_val := sInf {y | ∃ x ∈ Set.Icc 1 4, f_a x = y}
  max_val + min_val = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_equals_64_l849_84954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_system_unique_solution_l849_84915

/-- Represents a 3x3 system of linear equations with specific coefficient properties -/
structure SpecialSystem where
  a : Fin 3 → Fin 3 → ℝ
  diagonal_positive : ∀ i, a i i > 0
  off_diagonal_negative : ∀ i j, i ≠ j → a i j < 0
  sum_positive : ∀ i, (Finset.sum (Finset.range 3) (λ j ↦ a i j)) > 0

/-- The solution to the system of equations -/
noncomputable def solution (sys : SpecialSystem) : Fin 3 → ℝ := 
  λ _ ↦ 0 -- We define the solution as the zero vector

/-- Theorem stating that the only solution to the special system is the zero vector -/
theorem special_system_unique_solution (sys : SpecialSystem) :
  solution sys = λ _ ↦ 0 := by
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_system_unique_solution_l849_84915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_condition_l849_84930

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

-- State the theorem
theorem f_sum_condition (x : ℝ) : f x + f (x - 1/2) > 1 ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_condition_l849_84930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_points_l849_84981

def zach_points : ℝ := 42.0
def total_points : ℝ := 63

theorem ben_points : total_points - zach_points = 21 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_points_l849_84981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_tamika_greater_carlos_l849_84918

def tamika_set : Finset ℕ := {5, 6, 7}
def carlos_set : Finset ℕ := {2, 4, 8}

def tamika_sums : Finset ℕ := {11, 12, 13}
def carlos_products : Finset ℕ := {8, 16, 32}

def total_outcomes : ℕ := (tamika_sums.card * carlos_products.card)
def favorable_outcomes : ℕ := 3

theorem probability_tamika_greater_carlos :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_tamika_greater_carlos_l849_84918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_not_divisible_l849_84944

def sequence_x (n : ℕ) : ℕ :=
  match n with
  | 0 => 2022
  | n + 1 => 7 * sequence_x n + 5

def is_not_divisible_by_seven (n m : ℕ) : Prop :=
  ¬ (7 ∣ Nat.choose (sequence_x n) m)

theorem max_m_not_divisible : ∀ n : ℕ, is_not_divisible_by_seven n 404 ∧
  ∀ k : ℕ, k > 404 → ∃ n : ℕ, ¬ (is_not_divisible_by_seven n k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_not_divisible_l849_84944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_breadth_is_six_l849_84967

/-- Calculates the breadth of a room given its length, carpet width, cost per meter, and total cost. -/
noncomputable def room_breadth (room_length : ℝ) (carpet_width : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let total_carpet_length := total_cost / (cost_per_meter / 100)
  let num_strips := total_carpet_length / room_length
  num_strips * (carpet_width / 100)

/-- Theorem stating that for a room 15 meters long, carpeted with a 75 cm wide carpet 
    at a cost of 30 paise per meter, and a total cost of Rs. 36, the breadth of the room is 6 meters. -/
theorem room_breadth_is_six :
  room_breadth 15 75 30 3600 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_breadth_is_six_l849_84967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_equals_28_l849_84989

/-- Represents a term in the sum, with a coefficient (1 or -1) and an exponent --/
structure Term where
  coeff : Int
  exp : Nat
  coeff_valid : coeff = 1 ∨ coeff = -1

/-- The sum of terms equals 2022 --/
def sum_equals_2022 (terms : List Term) : Prop :=
  (terms.map (fun t => t.coeff * 3^t.exp)).sum = 2022

/-- The exponents are strictly decreasing --/
def exponents_decreasing (terms : List Term) : Prop :=
  terms.zip (terms.tail) |>.all (fun (t1, t2) => t1.exp > t2.exp)

/-- The main theorem --/
theorem sum_of_exponents_equals_28 (terms : List Term) 
    (h_sum : sum_equals_2022 terms)
    (h_decreasing : exponents_decreasing terms) :
    (terms.map (fun t => t.exp)).sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_equals_28_l849_84989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l849_84979

theorem tan_sum_given_tan_cot_sum (a b : ℝ) 
  (h1 : Real.tan a + Real.tan b = 15)
  (h2 : (Real.tan a)⁻¹ + (Real.tan b)⁻¹ = 20) : 
  Real.tan (a + b) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l849_84979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l849_84970

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x + (1-a) * x + 1

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x < y → x < a → y > a → (f a x > f a y)) ∧
  (a = 1 → ∀ x > 0, f a x ≤ x * (Real.exp x - 1) + (1/2) * x^2 - 2 * Real.log x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l849_84970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_properties_l849_84975

variable {G : Type*} [Group G]
variable (g : G)

-- Define the map φᵍ
def phi (n : ℤ) : G := g ^ n

-- Statement of the theorem
theorem phi_properties :
  -- φᵍ is a group homomorphism
  (∀ n m : ℤ, phi g (n + m) = phi g n * phi g m) ∧
  -- Kernel properties
  ((∀ n : ℤ, g ^ n = 1 → n = 0) →  -- g has infinite order
    (∀ n : ℤ, phi g n = 1 ↔ n = 0)) ∧
  (∃ k : ℕ, k > 0 ∧ g ^ k = 1 →  -- g has finite order
    ∃ n : ℕ, n > 0 ∧ ∀ m : ℤ, phi g m = 1 ↔ ∃ k : ℤ, m = n * k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_properties_l849_84975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_at_four_and_eight_l849_84936

/-- The distance function of a particle moving in a straight line -/
noncomputable def distance (t : ℝ) : ℝ := (1/3) * t^3 - 6 * t^2 + 32 * t

/-- The velocity function of the particle -/
noncomputable def velocity (t : ℝ) : ℝ := deriv distance t

theorem velocity_zero_at_four_and_eight :
  velocity 4 = 0 ∧ velocity 8 = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_at_four_and_eight_l849_84936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l849_84913

theorem triangle_side_values (A B C : ℝ) (a b c : ℝ) :
  let m : ℝ × ℝ := (2 * (Real.cos C)^2, Real.sqrt 3)
  let n : ℝ × ℝ := (1, Real.sin (2 * C))
  let f : ℝ → ℝ := λ x ↦ m.1 * n.1 + m.2 * n.2
  f C = 3 ∧
  c = 1 ∧
  a * b = 2 * Real.sqrt 3 ∧
  a > b ∧
  a^2 + b^2 - 2 * a * b * Real.cos C = c^2 →
  a = 2 ∧ b = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l849_84913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_geq_one_l849_84921

theorem negation_of_exists_sin_geq_one :
  (¬ ∃ x : ℝ, Real.sin x ≥ 1) ↔ (∀ x : ℝ, Real.sin x < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_geq_one_l849_84921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_feed_cost_l849_84976

theorem expensive_feed_cost
  (total_mixed : ℝ)
  (mixed_cost : ℝ)
  (cheap_cost : ℝ)
  (cheap_amount : ℝ)
  (h1 : total_mixed = 35)
  (h2 : mixed_cost = 0.36)
  (h3 : cheap_cost = 0.18)
  (h4 : cheap_amount = 17) :
  let expensive_amount := total_mixed - cheap_amount
  let total_value := total_mixed * mixed_cost
  let cheap_value := cheap_amount * cheap_cost
  let expensive_value := total_value - cheap_value
  expensive_value / expensive_amount = 0.53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_feed_cost_l849_84976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l849_84990

/-- Given two 2D vectors a and b, if the angle between them is π/6, prove that the y-coordinate of b is √3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b.1 = 3 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 6 →
  b.2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l849_84990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_after_operations_l849_84932

def operation (a b : ℕ) : ℕ := a * b + a + b

def initial_numbers : List ℕ := List.range' 1 20

theorem final_number_after_operations :
  ∃ (sequence : List (ℕ × ℕ)),
    sequence.length = 19 ∧
    (sequence.foldl
      (λ (numbers : List ℕ) (pair : ℕ × ℕ) =>
        let a := pair.fst
        let b := pair.snd
        numbers.filter (λ x => x ≠ a ∧ x ≠ b) ++ [operation a b])
      initial_numbers).head? = some (Nat.factorial 21 - 1) := by
  sorry

#check final_number_after_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_after_operations_l849_84932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_arrangement_exists_l849_84995

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square on the plane -/
structure Square where
  center : Point
  side_length : ℝ

/-- Checks if two squares overlap -/
def squares_overlap (s1 s2 : Square) : Prop := sorry

/-- Checks if a square covers at least one point of another square -/
def square_covers_point (s1 s2 : Square) : Prop := sorry

/-- The black square on the plane -/
def black_square : Square := sorry

/-- A list of seven square tiles -/
def tiles : List Square := sorry

/-- Theorem stating that it's possible to arrange the tiles as required -/
theorem tile_arrangement_exists : 
  ∃ (arrangement : List Square), 
    arrangement.length = 7 ∧ 
    (∀ t, t ∈ arrangement → square_covers_point t black_square) ∧
    (∀ t1 t2, t1 ∈ arrangement → t2 ∈ arrangement → t1 ≠ t2 → ¬squares_overlap t1 t2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_arrangement_exists_l849_84995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_has_199_eggs_l849_84974

/-- Represents a crate of eggs with its count and price -/
structure EggCrate where
  count : Nat
  price : Rat

/-- Represents Michael's egg inventory -/
structure MichaelsEggs where
  tuesday : List EggCrate
  wednesday_given : List EggCrate
  thursday_bought : List EggCrate
  friday_sold : List EggCrate

/-- Calculate the number of eggs Michael has after all transactions -/
def remaining_eggs (inventory : MichaelsEggs) : Nat :=
  let initial_count := inventory.tuesday.map EggCrate.count |>.sum
  let given_count := inventory.wednesday_given.map EggCrate.count |>.sum
  let bought_count := inventory.thursday_bought.map EggCrate.count |>.sum
  let sold_count := inventory.friday_sold.map EggCrate.count |>.sum
  initial_count - given_count + bought_count - sold_count

/-- The main theorem stating Michael has 199 eggs remaining -/
theorem michael_has_199_eggs (inventory : MichaelsEggs) 
  (h_tuesday : inventory.tuesday = 
    [EggCrate.mk 24 12, EggCrate.mk 28 14, EggCrate.mk 32 16, 
     EggCrate.mk 36 18, EggCrate.mk 40 20, EggCrate.mk 44 22])
  (h_wednesday : inventory.wednesday_given = 
    [EggCrate.mk 28 14, EggCrate.mk 32 16, EggCrate.mk 40 20])
  (h_thursday : inventory.thursday_bought = 
    [EggCrate.mk 50 20, EggCrate.mk 45 18, EggCrate.mk 55 22, EggCrate.mk 60 24])
  (h_friday : inventory.friday_sold = 
    [EggCrate.mk 60 15, EggCrate.mk 55 10]) :
  remaining_eggs inventory = 199 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_has_199_eggs_l849_84974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l849_84978

/-- Given a triangle ABC with acute angle A, prove that if √3b = 2a*sin(B), 
    then angle A = 60°, and if a = 7 and the area is 10√3, then b^2 + c^2 = 89 -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A → A < π/2 →  -- A is acute
  Real.sqrt 3 * b = 2 * a * Real.sin B →  -- Given condition
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S = 10 * Real.sqrt 3) →  -- Area condition
  a = 7 →  -- Given value of a
  A = π/3 ∧ b^2 + c^2 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l849_84978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_15_l849_84923

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of degrees in a circle -/
def degrees_in_circle : ℝ := 360

/-- The angle moved by the hour hand in one hour -/
noncomputable def hour_hand_angle_per_hour : ℝ := degrees_in_circle / clock_hours

/-- The angle moved by the minute hand in one minute -/
noncomputable def minute_hand_angle_per_minute : ℝ := degrees_in_circle / minutes_per_hour

/-- The position of the hour hand at 8:15 -/
noncomputable def hour_hand_position : ℝ :=
  8 * hour_hand_angle_per_hour + 15 * (hour_hand_angle_per_hour / minutes_per_hour)

/-- The position of the minute hand at 8:15 -/
noncomputable def minute_hand_position : ℝ := 15 * minute_hand_angle_per_minute

/-- The absolute difference between the hour hand and minute hand positions -/
noncomputable def angle_difference : ℝ := |hour_hand_position - minute_hand_position|

/-- The smaller angle between the hour hand and minute hand -/
noncomputable def smaller_angle : ℝ := min angle_difference (degrees_in_circle - angle_difference)

theorem clock_angle_at_8_15 : smaller_angle = 157.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_15_l849_84923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_possibilities_no_one_or_three_tangents_l849_84953

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles -/
def CommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- Two circles with different radii -/
axiom different_radii (c1 c2 : Circle) : c1.radius ≠ c2.radius

/-- The possible number of common tangents -/
theorem common_tangents_possibilities (c1 c2 : Circle) : 
  (CommonTangents c1 c2 = 0) ∨ 
  (CommonTangents c1 c2 = 2) ∨ 
  (CommonTangents c1 c2 = 4) := by
  sorry

/-- There cannot be exactly 1 or 3 common tangents -/
theorem no_one_or_three_tangents (c1 c2 : Circle) : 
  (CommonTangents c1 c2 ≠ 1) ∧ 
  (CommonTangents c1 c2 ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_possibilities_no_one_or_three_tangents_l849_84953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_properties_l849_84935

variable (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)

/-- Given 6 ordered real numbers -/
def ordered_samples : Prop :=
  x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ ∧ x₄ ≤ x₅ ∧ x₅ ≤ x₆

/-- Median of 4 numbers -/
noncomputable def median4 (a b c d : ℝ) : ℝ := (b + c) / 2

/-- Median of 6 numbers -/
noncomputable def median6 (a b c d e f : ℝ) : ℝ := (c + d) / 2

/-- Range of 4 numbers -/
def range4 (a b c d : ℝ) : ℝ := d - a

/-- Range of 6 numbers -/
def range6 (a b c d e f : ℝ) : ℝ := f - a

theorem sample_properties (h : ordered_samples x₁ x₂ x₃ x₄ x₅ x₆) :
  (median4 x₂ x₃ x₄ x₅ = median6 x₁ x₂ x₃ x₄ x₅ x₆) ∧
  (range4 x₂ x₃ x₄ x₅ ≤ range6 x₁ x₂ x₃ x₄ x₅ x₆) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_properties_l849_84935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l849_84911

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 2*x - 45

-- State the theorem
theorem g_equality (a : ℝ) : 
  a < 0 → (g (g (g 11)) = g (g (g a)) ↔ a = -34) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l849_84911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residue_of_n_l849_84942

theorem residue_of_n (m n k : ℕ) 
  (hm : m > 0) (hn : n > 0) (hk : k > 0)
  (h1 : m^2 + 1 = 2*n^2) 
  (h2 : 2*m^2 + 1 = 11*k^2) : 
  n % 17 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residue_of_n_l849_84942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeremy_school_distance_l849_84982

/-- The distance from Jeremy's home to school in miles -/
noncomputable def distance : ℝ := sorry

/-- The speed during rush hour in miles per hour -/
noncomputable def rush_hour_speed : ℝ := sorry

/-- The travel time during rush hour in hours -/
def rush_hour_time : ℚ := 1/2

/-- The travel time without traffic in hours -/
def no_traffic_time : ℚ := 1/4

/-- The speed without traffic in miles per hour -/
noncomputable def no_traffic_speed : ℝ := rush_hour_speed + 12

theorem jeremy_school_distance : 
  distance = rush_hour_speed * (rush_hour_time : ℝ) ∧
  distance = no_traffic_speed * (no_traffic_time : ℝ) ∧
  distance = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeremy_school_distance_l849_84982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l849_84912

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x) - 2 * Real.sqrt 3 * (Real.sin (ω * x / 2))^2 + Real.sqrt 3

theorem min_value_of_f (ω : ℝ) (h_ω : ω > 0) 
  (h_intersect : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ x₂ - x₁ = Real.pi / 2) :
  ∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f ω x_min = -Real.sqrt 3 ∧ 
    ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f ω x ≥ -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l849_84912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l849_84931

/-- The distance between Masha's and Misha's homes -/
noncomputable def distance_between_homes : ℝ := 3

/-- Masha's original speed -/
noncomputable def v_m : ℝ := sorry

/-- Misha's original speed -/
noncomputable def v_i : ℝ := sorry

/-- Time taken in both scenarios -/
noncomputable def t : ℝ := sorry

/-- First scenario: distance traveled by Masha -/
noncomputable def masha_distance_1 : ℝ := 1

/-- First scenario: distance traveled by Misha -/
noncomputable def misha_distance_1 : ℝ := distance_between_homes - 1

/-- Second scenario: Masha's new speed -/
noncomputable def masha_speed_2 : ℝ := 2 * v_m

/-- Second scenario: Misha's new speed -/
noncomputable def misha_speed_2 : ℝ := v_i / 2

/-- Second scenario: distance traveled by Masha -/
noncomputable def masha_distance_2 : ℝ := distance_between_homes - 1

/-- Second scenario: distance traveled by Misha -/
noncomputable def misha_distance_2 : ℝ := 1

/-- Theorem: The distance between Masha's and Misha's homes is 3 km -/
theorem distance_is_three : distance_between_homes = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l849_84931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l849_84997

noncomputable def f (x : ℝ) := Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.union (Set.Icc (-3 : ℝ) (-Real.pi/2)) (Set.Ioo (0 : ℝ) (Real.pi/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l849_84997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_distance_A_short_distance_B_equidistant_C_D_l849_84945

noncomputable def short_distance (x y : ℝ) : ℝ := min (abs x) (abs y)

def equidistant (x1 y1 x2 y2 : ℝ) : Prop :=
  short_distance x1 y1 = short_distance x2 y2

theorem short_distance_A :
  short_distance (-5) (-2) = 2 := by sorry

theorem short_distance_B (m : ℝ) :
  short_distance (-2) (-2*m + 1) = 1 ↔ m = 0 ∨ m = 1 := by sorry

theorem equidistant_C_D (k : ℝ) :
  equidistant (-1) (k + 3) 4 (2*k - 3) ↔ k = 1 ∨ k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_distance_A_short_distance_B_equidistant_C_D_l849_84945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l849_84956

/-- The speed of a train given its length, time to cross a man, and the man's speed --/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 110 →
  crossing_time = 6 →
  man_speed_kmh = 5 →
  ∃ (train_speed_kmh : ℝ), abs (train_speed_kmh - 61) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l849_84956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l849_84903

/-- The time it takes for person B to complete the project alone -/
noncomputable def time_B_alone (time_AB : ℝ) (time_together : ℝ) (time_B_remaining : ℝ) : ℝ :=
  1 / ((1 - time_together / time_AB) / time_B_remaining)

/-- Theorem stating that person B takes 15 days to complete the project alone -/
theorem project_completion_time :
  let time_AB : ℝ := 6
  let time_together : ℝ := 2
  let time_B_remaining : ℝ := 10
  time_B_alone time_AB time_together time_B_remaining = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l849_84903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_value_proof_l849_84996

theorem d_value_proof (d : ℝ) : 
  (3 * (Int.floor d)^2 - 12 * (Int.floor d) + 9 = 0) ∧ 
  (4 * (d - Int.floor d)^3 - 8 * (d - Int.floor d)^2 + 3 * (d - Int.floor d) - 0.5 = 0) →
  d = 1.375 ∨ d = 3.375 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_value_proof_l849_84996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_difference_l849_84984

noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + (a * b)^2)

theorem nabla_difference : 
  nabla 3 4 - nabla 1 2 = -16/29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_difference_l849_84984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_transformation_l849_84959

/-- Represents the composition of a mixture --/
structure Mixture where
  volume : ℝ
  alcohol_percentage : ℝ
  glycerin_percentage : ℝ

/-- Calculates the new mixture after adding pure alcohol and glycerin --/
noncomputable def add_pure_components (m : Mixture) (alcohol : ℝ) (glycerin : ℝ) : Mixture :=
  { volume := m.volume + alcohol + glycerin,
    alcohol_percentage := (m.volume * m.alcohol_percentage + alcohol * 100) / (m.volume + alcohol + glycerin),
    glycerin_percentage := (m.volume * m.glycerin_percentage + glycerin * 100) / (m.volume + alcohol + glycerin) }

/-- Theorem stating that adding the calculated amounts of alcohol and glycerin results in the desired mixture --/
theorem mixture_transformation (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (alcohol glycerin : ℝ),
    let initial := Mixture.mk 12 30 10
    let final := add_pure_components initial alcohol glycerin
    (abs (final.alcohol_percentage - 45) < ε) ∧
    (abs (final.glycerin_percentage - 15) < ε) ∧
    (abs (alcohol - 4.49) < ε) ∧
    (abs (glycerin - 1.496) < ε) :=
by
  sorry

#check mixture_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_transformation_l849_84959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l849_84952

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - a * x + 1)

def q (a : ℝ) : Prop := ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → (x₁ : ℝ)^(a^2 - 2*a - 3) > (x₂ : ℝ)^(a^2 - 2*a - 3)

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  ((-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l849_84952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_manuscript_fragment_l849_84966

/-- Represents a book fragment with first and last page numbers -/
structure BookFragment where
  first_page : Nat
  last_page : Nat

/-- Check if two numbers have the same digits -/
def same_digits (a b : Nat) : Prop := sorry

/-- Calculate the number of sheets in a book fragment -/
def num_sheets (fragment : BookFragment) : Nat :=
  (fragment.last_page - fragment.first_page + 1) / 2

/-- Theorem stating the number of sheets in the specific fragment -/
theorem ancient_manuscript_fragment :
  ∀ (fragment : BookFragment),
    fragment.first_page = 435 →
    fragment.last_page > fragment.first_page →
    fragment.last_page % 2 = 0 →
    same_digits fragment.first_page fragment.last_page →
    num_sheets fragment = 50 := by
  sorry

#check ancient_manuscript_fragment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_manuscript_fragment_l849_84966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_value_l849_84973

theorem min_coefficient_value (a b Box : ℤ) : 
  (∀ x : ℝ, (a*x + b) * (b*x + a) = 15*x^2 + Box*x + 15) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  Box ≥ 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_value_l849_84973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l849_84929

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos (2 * x)

theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l849_84929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_value_l849_84910

theorem orthogonal_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) :
  a = (1, 2) →
  b = (1, 1) →
  c = a + k • b →
  (b.1 * c.1 + b.2 * c.2) = 0 →
  k = -3/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_value_l849_84910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_breakfast_shopping_cost_is_11_l849_84977

/-- The cost of Frank's breakfast shopping -/
def breakfast_shopping_cost : ℚ :=
  let bun_cost : ℚ := 1/10
  let milk_bottle_cost : ℚ := 2
  let bun_quantity : ℕ := 10
  let milk_bottle_quantity : ℕ := 2
  let egg_carton_cost : ℚ := 3 * milk_bottle_cost
  
  bun_cost * bun_quantity +
  milk_bottle_cost * milk_bottle_quantity +
  egg_carton_cost

/-- Theorem stating that Frank's breakfast shopping cost is $11 -/
theorem breakfast_shopping_cost_is_11 :
  breakfast_shopping_cost = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_breakfast_shopping_cost_is_11_l849_84977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l849_84906

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x))^2 - 1 + 2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x)

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f ((x + 2 * Real.pi / 3) / 2)

theorem function_properties (ω : ℝ) (α : ℝ) :
  (0 < ω ∧ ω < 1) →
  (∀ x, f ω x = f ω (2 * Real.pi / 3 - x)) →
  (α > 0 ∧ α < Real.pi / 2) →
  (g (f ω) (2 * α + Real.pi / 3) = 8 / 5) →
  (ω = 1 / 2 ∧ Real.sin α = (4 * Real.sqrt 3 - 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l849_84906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_3_2023_closest_to_6_l849_84920

/-- The harmonic mean of two positive real numbers -/
noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

/-- The nearest integer to a real number -/
noncomputable def nearestInteger (x : ℝ) : ℤ := ⌊x + 0.5⌋

theorem harmonic_mean_3_2023_closest_to_6 :
  nearestInteger (harmonicMean 3 2023) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_3_2023_closest_to_6_l849_84920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_squared_l849_84988

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 4 * x)

-- State the theorem
theorem inverse_g_squared : (Function.invFun g 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_squared_l849_84988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_sum_of_digits_l849_84964

theorem triangular_array_sum_of_digits (N : ℕ) : N * (N + 1) / 2 = 2145 → (N.digits 10).sum = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_sum_of_digits_l849_84964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_salary_problem_l849_84962

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary_all : ℚ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℚ) :
  total_workers = 20 →
  avg_salary_all = 750 →
  num_technicians = 5 →
  avg_salary_technicians = 900 →
  let rest_workers := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let technicians_salary := avg_salary_technicians * num_technicians
  let rest_salary := total_salary - technicians_salary
  let avg_salary_rest := rest_salary / rest_workers
  avg_salary_rest = 700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_salary_problem_l849_84962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_polar_curve_l849_84955

/-- The curve in polar coordinates --/
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 4 * Real.sin θ

/-- The focus of the curve in polar coordinates --/
noncomputable def focus : ℝ × ℝ := (1, Real.pi / 2)

/-- Theorem stating that the given point is the focus of the curve --/
theorem focus_of_polar_curve :
  ∀ ρ θ : ℝ, polar_curve ρ θ → focus = (1, Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_polar_curve_l849_84955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l849_84947

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, 
    this theorem proves the radius of the third circle. -/
theorem third_circle_radius (r₁ r₂ : ℝ) (h₁ : r₁ = 2) (h₂ : r₂ = 5) : 
  ∃ r₃ : ℝ, r₃ = (7 / (2 * (Real.sqrt 2 + Real.sqrt 10)))^2 ∧ 
  (∃ A B C : EuclideanSpace ℝ (Fin 2), 
    (‖A - B‖ = r₁ + r₂) ∧  -- Circles are externally tangent
    (∃ D : EuclideanSpace ℝ (Fin 2), 
      ‖D - A‖ = r₁ + r₃ ∧  -- Third circle is tangent to first circle
      ‖D - B‖ = r₂ + r₃ ∧  -- Third circle is tangent to second circle
      (∃ E : EuclideanSpace ℝ (Fin 2), 
        (∃ t : ℝ, E = (1 - t) • A + t • B) ∧ -- E is on the line through A and B
        ‖D - E‖ = r₃))) -- Third circle is tangent to common external tangent
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l849_84947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l849_84927

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem intersection_segment_length :
  ∃ (x1 y1 x2 y2 : ℝ),
    curve_C x1 y1 ∧ curve_C x2 y2 ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l849_84927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l849_84961

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.cos α = 3/5) 
  (h2 : Real.tan α < 0) : 
  Real.sin α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l849_84961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_acute_triangle_l849_84980

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Abstract predicate for acute triangle -/
def IsAcuteTriangle (A P Q : Point) : Prop := sorry

/-- Theorem stating the range of eccentricity for the given hyperbola conditions -/
theorem eccentricity_range_for_acute_triangle (h : Hyperbola) 
  (A P Q : Point) (F : Point) :
  -- A is the right vertex
  A.x = h.a ∧ A.y = 0 →
  -- F is the left focus
  F.x = -Real.sqrt (h.a^2 + h.b^2) ∧ F.y = 0 →
  -- P and Q are on the hyperbola
  P.x^2 / h.a^2 - P.y^2 / h.b^2 = 1 →
  Q.x^2 / h.a^2 - Q.y^2 / h.b^2 = 1 →
  -- P and Q are on a line parallel to y-axis passing through F
  P.x = F.x ∧ Q.x = F.x →
  -- APQ is an acute triangle
  IsAcuteTriangle A P Q →
  -- Then the eccentricity is between 1 and 2
  1 < eccentricity h ∧ eccentricity h < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_acute_triangle_l849_84980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_pow_n_minus_m_div_2_pow_k_l849_84999

theorem exists_n_pow_n_minus_m_div_2_pow_k (k : ℕ) (m : ℕ) (h_m_odd : Odd m) :
  ∃ n : ℕ, (2^k : ℕ) ∣ n^n - m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_pow_n_minus_m_div_2_pow_k_l849_84999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_c_value_l849_84938

theorem tan_double_angle_c_value (x c : ℝ) :
  Real.tan x = 3 / 2 →
  Real.tan (2 * x) = (5 * c - 2) / (3 * c) →
  Real.arctan (Real.tan x) = x →
  c = 10 / 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_c_value_l849_84938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sets_right_triangles_l849_84904

-- Define the four sets of triangle sides
def set1 : (ℝ × ℝ × ℝ) := (1.5, 2.5, 2)
noncomputable def set2 : (ℝ × ℝ × ℝ) := (Real.sqrt 2, Real.sqrt 2, 2)
def set3 : (ℝ × ℝ × ℝ) := (12, 16, 20)
def set4 : (ℝ × ℝ × ℝ) := (0.5, 1.2, 1.3)

-- Function to check if a set of sides forms a right triangle
def isRightTriangle (sides : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := sides
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that all four sets form right triangles
theorem all_sets_right_triangles :
  isRightTriangle set1 ∧
  isRightTriangle set2 ∧
  isRightTriangle set3 ∧
  isRightTriangle set4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sets_right_triangles_l849_84904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l849_84926

noncomputable def f (x : ℝ) : ℝ := Real.log ((x - 1)^2) / Real.log 4

noncomputable def g (x : ℝ) : ℝ := 1 - (4 : ℝ)^(x/2)

theorem inverse_function_proof (x : ℝ) (hx : x < 1) :
  Function.LeftInverse g f ∧ Function.RightInverse g f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l849_84926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_k_values_l849_84905

theorem sum_of_k_values (j k : ℕ) : 
  (j > 0) → (k > 0) →
  (1 : ℚ) / j + (1 : ℚ) / k = (1 : ℚ) / 5 → 
  ∃ (S : Finset ℕ), (∀ k' ∈ S, ∃ j' : ℕ, j' > 0 ∧ (1 : ℚ) / j' + (1 : ℚ) / k' = (1 : ℚ) / 5) ∧ 
                    (∀ k' : ℕ, k' > 0 → (∃ j' : ℕ, j' > 0 ∧ (1 : ℚ) / j' + (1 : ℚ) / k' = (1 : ℚ) / 5) → k' ∈ S) ∧
                    (S.sum id = 46) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_k_values_l849_84905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_on_exp_curve_l849_84901

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the area calculation using the Trapezoidal Rule
noncomputable def trapezoidal_area (n : ℝ) : ℝ :=
  (1 / 2) * (exp n + 2 * exp (n + 1) + 2 * exp (n + 2) + exp (n + 3))

-- Theorem statement
theorem quadrilateral_on_exp_curve (n : ℝ) :
  trapezoidal_area n = 1 / exp 1 → n = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_on_exp_curve_l849_84901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_shapes_l849_84998

theorem equal_area_shapes (carol_length carol_width jordan_length mike_base : ℝ)
  (h_carol_length : carol_length = 12)
  (h_carol_width : carol_width = 15)
  (h_jordan_length : jordan_length = 6)
  (h_mike_base : mike_base = 10) :
  let carol_area := carol_length * carol_width
  let jordan_width := carol_area / jordan_length
  let mike_height := 2 * carol_area / mike_base
  jordan_width = 30 ∧ mike_height = 36 := by
  -- Introduce the let-bindings
  let carol_area := carol_length * carol_width
  let jordan_width := carol_area / jordan_length
  let mike_height := 2 * carol_area / mike_base

  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_shapes_l849_84998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ezekiel_hike_distance_l849_84993

/-- Represents a three-day hike with given distances for each day -/
structure ThreeDayHike where
  fullDistance : ℝ
  day1Distance : ℝ
  day2Distance : ℝ
  day3Distance : ℝ

/-- Defines the properties of Ezekiel's hike -/
def ezekielHike : ThreeDayHike :=
  { fullDistance := 50
  , day1Distance := 10
  , day2Distance := 25
  , day3Distance := 15 }

/-- Theorem stating that Ezekiel's hike satisfies the given conditions -/
theorem ezekiel_hike_distance :
  ezekielHike.day1Distance + ezekielHike.day2Distance + ezekielHike.day3Distance = ezekielHike.fullDistance ∧
  ezekielHike.day2Distance = ezekielHike.fullDistance / 2 :=
by
  -- Split the goal into two parts
  constructor
  -- Prove the first part: sum of distances equals full distance
  · simp [ezekielHike]
    norm_num
  -- Prove the second part: day2Distance is half of fullDistance
  · simp [ezekielHike]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ezekiel_hike_distance_l849_84993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_identification_l849_84934

-- Define the type of real-valued functions
noncomputable def RealFunction := ℝ → ℝ

-- Define the property of direct proportionality
def IsDirectlyProportional (f : RealFunction) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define the given functions
noncomputable def f1 : RealFunction := λ x ↦ (1/2) * x
noncomputable def f2 : RealFunction := λ x ↦ 2 * x + 1
noncomputable def f3 : RealFunction := λ x ↦ 2 / x
noncomputable def f4 : RealFunction := λ x ↦ x^2

-- State the theorem
theorem direct_proportion_identification :
  IsDirectlyProportional f1 ∧
  ¬IsDirectlyProportional f2 ∧
  ¬IsDirectlyProportional f3 ∧
  ¬IsDirectlyProportional f4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_identification_l849_84934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_30_equals_sqrt3_over_3_l849_84992

open Real

-- Define the tangent of 30 degrees
noncomputable def tan_30 : ℝ := tan (30 * π / 180)

-- Define the properties of a 30-60-90 triangle
axiom triangle_30_60_90 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ a = (c / 2) ∧ b = (c * sqrt 3 / 2)

-- Theorem statement
theorem tan_30_equals_sqrt3_over_3 : tan_30 = sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_30_equals_sqrt3_over_3_l849_84992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_OAB_l849_84941

/-- The line passing through points A and B -/
def line (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- Point O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- Point A is the x-intercept of the line -/
def A : ℝ × ℝ := (4, 0)

/-- Point B is the y-intercept of the line -/
def B : ℝ × ℝ := (0, 2)

/-- The circle passing through O, A, and B -/
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

theorem circle_through_OAB : 
  line A.1 A.2 ∧ line B.1 B.2 → 
  circle_equation O.1 O.2 ∧ circle_equation A.1 A.2 ∧ circle_equation B.1 B.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_OAB_l849_84941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l849_84940

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * Real.log x + 1 / x + 2 * a * x

theorem f_properties (a : ℝ) (h : a ≤ 0) :
  (a = 0 → ∃ x : ℝ, x > 0 ∧ x = 1/2 ∧ f 0 x = 2 - 2 * Real.log 2 ∧ ∀ y > 0, f 0 y ≥ f 0 x) ∧
  (a < 0 → ∀ x > 0, x ≠ 1/2 ∧ x ≠ -1/a → (deriv (f a)) x ≠ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l849_84940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_difference_l849_84946

theorem remainder_of_difference (a b : ℕ) : 
  a % 6 = 2 → b % 6 = 3 → a > b → (a - b) % 6 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_difference_l849_84946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_l849_84902

/-- Represents a rectangle with given side lengths -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with given radius -/
structure Circle where
  radius : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ :=
  Real.pi * c.radius * c.radius

/-- Calculates the area of a quarter circle -/
noncomputable def quarterCircleArea (c : Circle) : ℝ :=
  (circleArea c) / 4

theorem area_outside_circles (r : Rectangle) (c1 c2 c3 : Circle) :
  r.width = 4 →
  r.height = 6 →
  c1.radius = 1.5 →
  c2.radius = 2.5 →
  c3.radius = 3.5 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    abs ((rectangleArea r - (quarterCircleArea c1 + quarterCircleArea c2 + quarterCircleArea c3)) - 7.7) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_l849_84902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_grid_exists_l849_84922

-- Define a 3x3 grid
def Grid := Fin 3 → Fin 3 → Fin 3

-- Define a property that checks if all elements in a list are distinct
def allDistinct (l : List ℕ) : Prop := l.Nodup

-- Define a property that checks if a grid satisfies the row, column, and diagonal constraints
def validGrid (g : Grid) : Prop :=
  (∀ i, allDistinct [g i 0, g i 1, g i 2]) ∧  -- rows
  (∀ j, allDistinct [g 0 j, g 1 j, g 2 j]) ∧  -- columns
  allDistinct [g 0 0, g 1 1, g 2 2] ∧         -- main diagonal
  allDistinct [g 0 2, g 1 1, g 2 0]           -- anti-diagonal

-- Define a property that checks if each number appears exactly twice
def eachNumberTwice (g : Grid) : Prop :=
  (List.countP (· = 1) [g 0 0, g 0 1, g 0 2, g 1 0, g 1 1, g 1 2, g 2 0, g 2 1, g 2 2] = 2) ∧
  (List.countP (· = 2) [g 0 0, g 0 1, g 0 2, g 1 0, g 1 1, g 1 2, g 2 0, g 2 1, g 2 2] = 2) ∧
  (List.countP (· = 3) [g 0 0, g 0 1, g 0 2, g 1 0, g 1 1, g 1 2, g 2 0, g 2 1, g 2 2] = 2)

-- Theorem statement
theorem no_valid_grid_exists : ¬∃ (g : Grid), validGrid g ∧ eachNumberTwice g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_grid_exists_l849_84922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_calculation_l849_84908

theorem power_calculation : (2 : ℝ)⁻¹ - (1/2 : ℝ)^0 + 2^2023 * (-1/2 : ℝ)^2023 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_calculation_l849_84908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_n_l849_84914

def f (x : ℕ) : ℕ := x^2 + x + 1

theorem characterization_of_n (n : ℕ) (hn : n > 0) :
  (∀ k : ℕ, k > 0 → k ∣ n → f k ∣ f n) ↔
  (n = 1 ∨ (∃ p : ℕ, Nat.Prime p ∧ n = p ∧ p % 3 = 1) ∨
   (∃ p : ℕ, Nat.Prime p ∧ n = p^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_n_l849_84914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_farm_coconuts_per_tree_l849_84963

/-- Represents a coconut farm with given properties -/
structure CoconutFarm where
  size : ℕ
  treesPerSquareMeter : ℕ
  harvestFrequency : ℕ
  pricePerCoconut : ℚ
  earningsAfterSixMonths : ℚ

/-- Calculates the number of coconuts per tree given a CoconutFarm -/
noncomputable def coconutsPerTree (farm : CoconutFarm) : ℚ :=
  let totalTrees := farm.size * farm.treesPerSquareMeter
  let harvestsInSixMonths := 6 / farm.harvestFrequency
  let earningsPerHarvest := farm.earningsAfterSixMonths / harvestsInSixMonths
  let coconutsPerHarvest := earningsPerHarvest / farm.pricePerCoconut
  coconutsPerHarvest / totalTrees

/-- Theorem stating that for the given farm conditions, each tree has 6 coconuts -/
theorem rohan_farm_coconuts_per_tree :
  let farm := CoconutFarm.mk 20 2 3 (1/2) 240
  coconutsPerTree farm = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_farm_coconuts_per_tree_l849_84963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l849_84994

theorem cosine_sum_identity (A B : ℝ) :
  Real.cos A + Real.cos B = 2 * Real.cos ((A + B) / 2) * Real.cos ((A - B) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l849_84994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_theorem_l849_84969

/-- Represents the number of books borrowed by each student in a class. -/
structure ClassBorrowings where
  total_students : ℕ
  zero_books : ℕ
  one_book : ℕ
  two_books : ℕ
  avg_books : ℚ

/-- Calculates the maximum number of books any single student could have borrowed. -/
def max_books_borrowed (cb : ClassBorrowings) : ℕ :=
  let total_books := (cb.total_students : ℚ) * cb.avg_books
  let accounted_books := cb.one_book + 2 * cb.two_books
  let remaining_students := cb.total_students - (cb.zero_books + cb.one_book + cb.two_books)
  let remaining_books := total_books - (accounted_books : ℚ)
  ⌊remaining_books - 3 * ((remaining_students - 1) : ℚ)⌋.toNat

/-- Theorem stating the maximum number of books borrowed by any single student. -/
theorem max_books_theorem (cb : ClassBorrowings) 
  (h1 : cb.total_students = 32)
  (h2 : cb.zero_books = 2)
  (h3 : cb.one_book = 12)
  (h4 : cb.two_books = 10)
  (h5 : cb.avg_books = 2) :
  max_books_borrowed cb = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_theorem_l849_84969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_work_days_l849_84917

/-- Given workers x and y, their work rates, and y's partial work, calculate remaining days for x --/
theorem remaining_work_days (x_days y_days y_worked : ℕ) (hx : x_days > 0) (hy : y_days > 0) 
  (hw : y_worked < y_days) : 
  (1 - (1 / y_days : ℚ) * y_worked) / (1 / x_days : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_work_days_l849_84917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_expression_l849_84987

/-- Given two parallel vectors a and b, prove that the algebraic expression equals 5 -/
theorem parallel_vectors_expression (θ : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.cos θ, Real.sin θ]
  let b : Fin 2 → ℝ := ![1, -2]
  (∃ (k : ℝ), a = k • b) →
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_expression_l849_84987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l849_84951

theorem inequality_equivalence (x : ℝ) : 
  (2 : ℝ)^(2*x^2 - 6*x + 3) + (6 : ℝ)^(x^2 - 3*x + 1) ≥ (3 : ℝ)^(2*x^2 - 6*x + 3) ↔ 
  (3 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 5) / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l849_84951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circle_numbers_l849_84928

def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_arrangement (a : Fin 9 → ℕ) : Prop :=
  ∀ i, a i ∈ digits ∧ ∀ j, i ≠ j → a i ≠ a j

def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def circle_numbers (a : Fin 9 → ℕ) : Fin 9 → ℕ :=
  λ i ↦ three_digit_number (a i) (a ((i + 1) % 9)) (a ((i + 2) % 9))

theorem sum_of_circle_numbers (a : Fin 9 → ℕ) :
  is_valid_arrangement a →
  (Finset.sum Finset.univ (circle_numbers a)) = 4995 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circle_numbers_l849_84928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_required_average_l849_84919

def total_rounds : ℕ := 5
def qualification_threshold : ℚ := 90/100
def first_round_score : ℚ := 87/100
def second_round_score : ℚ := 92/100
def third_round_score : ℚ := 85/100

theorem minimum_required_average (remaining_rounds : ℕ) : 
  remaining_rounds = total_rounds - 3 →
  (qualification_threshold * total_rounds - 
   (first_round_score + second_round_score + third_round_score)) / remaining_rounds = 93/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_required_average_l849_84919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_integers_count_l849_84950

/-- The number of different 5-digit positive integers, given that a 5-digit number cannot start with 0. -/
theorem five_digit_integers_count : ℕ := by
  -- Define the set of 5-digit positive integers that don't start with 0
  let five_digit_set := Finset.filter (λ n : ℕ => 10000 ≤ n ∧ n ≤ 99999) (Finset.range 100000)
  
  -- Define the count of elements in the set
  let count := five_digit_set.card
  
  -- Prove that the count is equal to 90000
  have h : count = 90000 := by sorry
  
  -- Return the result
  exact 90000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_integers_count_l849_84950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_calculation_l849_84965

/-- Calculates the total amount spent on umbrellas, raincoats, and a waterproof bag,
    considering discounts and a partial refund for a returned item. -/
theorem total_spent_calculation
  (umbrella_count : ℕ)
  (umbrella_price : ℚ)
  (raincoat_count : ℕ)
  (raincoat_price : ℚ)
  (bag_price : ℚ)
  (discount_rate : ℚ)
  (refund_rate : ℚ)
  (h_umbrella_count : umbrella_count = 2)
  (h_umbrella_price : umbrella_price = 8)
  (h_raincoat_count : raincoat_count = 3)
  (h_raincoat_price : raincoat_price = 15)
  (h_bag_price : bag_price = 25)
  (h_discount_rate : discount_rate = 0.1)
  (h_refund_rate : refund_rate = 0.8)
  : discounted_total - refund = 65.4 :=
  let total_before_discount := umbrella_count * umbrella_price + raincoat_count * raincoat_price + bag_price
  let discounted_total := total_before_discount * (1 - discount_rate)
  let refund := raincoat_price * refund_rate
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_calculation_l849_84965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l849_84983

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ

noncomputable def focus (p : Parabola) : Point :=
  { x := 1 / (4 * p.a), y := 0 }

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def lineThruPoint (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

def onParabola (p : Point) (parab : Parabola) : Prop :=
  p.y^2 = 4 * parab.a * p.x

theorem parabola_line_intersection (parab : Parabola) (l : Line) (A : Point) :
  parab.a = 1 →
  l.slope = Real.sqrt 3 →
  lineThruPoint l (focus parab) →
  onParabola A parab →
  lineThruPoint l A →
  A.y > 0 →
  distance { x := 0, y := 0 } A = Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l849_84983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l849_84924

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_lambda (l : ℝ) : 
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (-1, l)
  parallel a b → l = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l849_84924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_sum_equals_one_l849_84909

-- Define the fractional part function as noncomputable
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem fractional_sum_equals_one (x : ℝ) (h : x^3 + 1/x^3 = 18) : 
  frac x + frac (1/x) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_sum_equals_one_l849_84909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l849_84957

/-- Given a triangle ABC with the following properties:
  - f(x) = m · n, where m = (2 cos x, 1) and n = (cos x, √3 sin 2x)
  - f(A) = 2
  - b = 1
  - S_△ABC = √3/2
  Prove that (b + c) / (sin B + sin C) = 2 -/
theorem triangle_ratio_theorem (A B C : ℝ) (a b c : ℝ) :
  let f := λ x ↦ 2 * Real.cos x * Real.cos x + Real.sqrt 3 * Real.sin (2 * x)
  f A = 2 →
  b = 1 →
  1/2 * a * b * Real.sin C = Real.sqrt 3 / 2 →
  (b + c) / (Real.sin B + Real.sin C) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l849_84957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_proposition_l849_84925

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations between lines and planes
axiom perpendicular : Line → Plane → Prop
axiom parallel : Line → Plane → Prop
axiom contains : Plane → Line → Prop
axiom perpendicular_lines : Line → Line → Prop
axiom parallel_lines : Line → Line → Prop
axiom parallel_planes : Plane → Plane → Prop

-- Define the propositions
def proposition1 (l : Line) (α : Plane) : Prop :=
  ∀ m n : Line, (contains α m ∧ contains α n ∧ perpendicular_lines l m ∧ perpendicular_lines l n) → perpendicular l α

def proposition2 (m l : Line) (α : Plane) : Prop :=
  parallel m α ∧ perpendicular l α → perpendicular_lines m l

def proposition3 (l : Line) (α : Plane) : Prop :=
  parallel l α → ∀ m : Line, contains α m → parallel_lines l m

def proposition4 (m l : Line) (α β : Plane) : Prop :=
  contains α m ∧ contains β l ∧ parallel_planes α β → parallel_lines m l

theorem only_one_correct_proposition :
  ∃! i : Fin 4, match i with
    | 0 => ∀ l α, proposition1 l α
    | 1 => ∀ m l α, proposition2 m l α
    | 2 => ∀ l α, proposition3 l α
    | 3 => ∀ m l α β, proposition4 m l α β
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_proposition_l849_84925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l849_84986

theorem cos_theta_value (θ : Real) 
  (h1 : Real.sin θ = -4/5) 
  (h2 : Real.tan θ > 0) : 
  Real.cos θ = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l849_84986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_total_area_shaded_percentage_is_100_l849_84900

/-- Represents a rectangle with coordinates of two opposite corners -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The square PQRS -/
def squarePQRS : Rectangle :=
  { x1 := 0, y1 := 0, x2 := 6, y2 := 6 }

/-- The three shaded rectangles -/
def shadedRectangles : List Rectangle :=
  [{ x1 := 0, y1 := 0, x2 := 2, y2 := 2 },
   { x1 := 3, y1 := 0, x2 := 5, y2 := 5 },
   { x1 := 6, y1 := 0, x2 := 0, y2 := 6 }]

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  |r.x2 - r.x1| * |r.y2 - r.y1|

/-- Theorem: The shaded area equals the total area of the square -/
theorem shaded_area_equals_total_area :
  rectangleArea squarePQRS = rectangleArea (shadedRectangles[2]) :=
by
  sorry

/-- Theorem: The shaded area is 100% of the square's area -/
theorem shaded_percentage_is_100 :
  (rectangleArea (shadedRectangles[2]) / rectangleArea squarePQRS) * 100 = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_total_area_shaded_percentage_is_100_l849_84900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l849_84943

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in ℝ² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : Vector2D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

theorem triangle_sin_A (t : Triangle) (G BA BC : Vector2D) :
  t.a^2 + t.c^2 - t.b^2 = t.a * t.c →
  t.c = 2 →
  magnitude G = Real.sqrt 19 / 3 →
  G = Vector2D.mk ((BA.x + BC.x) / 3) ((BA.y + BC.y) / 3) →
  Real.sin t.A = 3 * Real.sqrt 21 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l849_84943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trace_difference_equals_half_n_n_minus_one_l849_84916

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem trace_difference_equals_half_n_n_minus_one
  (h : Matrix.rank (A * B - B * A + 1) = 1) :
  Matrix.trace (A * B * A * B) - Matrix.trace (A * A * B * B) = (n * (n - 1) : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trace_difference_equals_half_n_n_minus_one_l849_84916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deluxe_time_fraction_is_four_ninths_l849_84907

/-- Represents the production details of a stereo company -/
structure StereoProduction where
  totalStereos : ℝ
  basicFraction : ℝ
  deluxeTimeFactor : ℝ

/-- Calculates the fraction of total production time spent on deluxe stereos -/
noncomputable def deluxeTimeFraction (prod : StereoProduction) : ℝ :=
  let basicTime := prod.basicFraction * prod.totalStereos
  let deluxeTime := (1 - prod.basicFraction) * prod.totalStereos * prod.deluxeTimeFactor
  deluxeTime / (basicTime + deluxeTime)

/-- Theorem stating that the fraction of total production time spent on deluxe stereos is 4/9 -/
theorem deluxe_time_fraction_is_four_ninths :
  ∀ (prod : StereoProduction),
    prod.basicFraction = 2/3 ∧ prod.deluxeTimeFactor = 1.6 →
    deluxeTimeFraction prod = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deluxe_time_fraction_is_four_ninths_l849_84907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_worked_32_hours_l849_84971

def hours_worked (standard_hours : ℕ) (extra_hours : ℕ) : ℕ :=
  standard_hours + extra_hours

def pay (standard_rate : ℝ) (overtime_rate : ℝ) (standard_hours : ℕ) (total_hours : ℕ) : ℝ :=
  standard_rate * (standard_hours : ℝ) + overtime_rate * ((total_hours - standard_hours) : ℝ)

theorem harry_worked_32_hours 
  (x y : ℝ) 
  (harry_standard_hours james_standard_hours james_total_hours : ℕ) 
  (harry_total_hours : ℕ) :
  harry_standard_hours = 30 →
  james_standard_hours = 40 →
  james_total_hours = 41 →
  pay x y harry_standard_hours harry_total_hours = pay x y james_standard_hours james_total_hours →
  harry_total_hours = 32 := by
  sorry

#check harry_worked_32_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_worked_32_hours_l849_84971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangements_of_1213_l849_84933

def digits : List Nat := [1, 2, 1, 3]

def is_valid_arrangement (arr : List Nat) : Bool :=
  arr.length = 4 && arr.head? != some 0 && arr.toFinset = digits.toFinset

def count_valid_arrangements : Nat :=
  (List.permutations digits).filter is_valid_arrangement |>.length

theorem rearrangements_of_1213 :
  count_valid_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangements_of_1213_l849_84933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l849_84968

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at the origin and focus on the y-axis -/
structure Parabola where
  focus : Point

/-- Check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the parabola -/
def isOnParabola (para : Parabola) (p : Point) : Prop :=
  p.x^2 = 4 * para.focus.y * p.y

/-- The main theorem -/
theorem parabola_point_theorem (para : Parabola) (p : Point) :
  para.focus = Point.mk 0 2 →
  p = Point.mk (8 * Real.sqrt 6) 48 →
  isOnParabola para p ∧
  isInFirstQuadrant p ∧
  distance p para.focus = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l849_84968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l849_84939

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity 
  (h : Hyperbola a b) 
  (P A B : Point) 
  (m n : ℝ) :
  (∃ (OP OA OB : Point), 
    OP.x = m * OA.x + n * OB.x ∧
    OP.y = m * OA.y + n * OB.y ∧
    m * n = 2/9) →
  eccentricity h = 3 * Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l849_84939
