import Mathlib

namespace NUMINAMATH_CALUDE_transverse_axis_length_l55_5596

-- Define the hyperbola M
def hyperbola_M (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the parabola N
def parabola_N (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of parabola N
def focus_N : ℝ × ℝ := (2, 0)

-- Define the condition that the right focus of M is the focus of N
def right_focus_condition (a b : ℝ) : Prop := (a, 0) = focus_N

-- Define the intersection points P and Q
def intersection_points (P Q : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola_M a b P.1 P.2 ∧ parabola_N P.1 P.2 ∧
  hyperbola_M a b Q.1 Q.2 ∧ parabola_N Q.1 Q.2

-- Define the condition that PF = FQ
def PF_equals_FQ (P Q : ℝ × ℝ) : Prop :=
  (P.1 - focus_N.1)^2 + (P.2 - focus_N.2)^2 =
  (Q.1 - focus_N.1)^2 + (Q.2 - focus_N.2)^2

-- Main theorem
theorem transverse_axis_length (a b : ℝ) (P Q : ℝ × ℝ) :
  hyperbola_M a b a 0 →
  right_focus_condition a b →
  intersection_points P Q a b →
  PF_equals_FQ P Q →
  2 * a = 4 * Real.sqrt 2 - 4 :=
sorry

end NUMINAMATH_CALUDE_transverse_axis_length_l55_5596


namespace NUMINAMATH_CALUDE_r_value_when_n_is_three_l55_5502

theorem r_value_when_n_is_three :
  let n : ℕ := 3
  let s : ℕ := 2^n - 1
  let r : ℕ := 2^s + s
  r = 135 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_three_l55_5502


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l55_5577

theorem sin_two_alpha_value (α : ℝ) 
  (h : (Real.cos (π - 2*α)) / (Real.sin (α - π/4)) = -Real.sqrt 2 / 2) : 
  Real.sin (2*α) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l55_5577


namespace NUMINAMATH_CALUDE_max_value_on_circle_l55_5500

theorem max_value_on_circle (x y : ℝ) : 
  (x - 1)^2 + y^2 = 4 → 
  ∃ b : ℝ, (∀ x' y' : ℝ, (x' - 1)^2 + y'^2 = 4 → 2*x' + y'^2 ≤ b) ∧ 
           (∃ x'' y'' : ℝ, (x'' - 1)^2 + y''^2 = 4 ∧ 2*x'' + y''^2 = b) ∧
           b = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l55_5500


namespace NUMINAMATH_CALUDE_plane_distance_l55_5518

/-- Given a plane flying east at 300 km/h and west at 400 km/h for a total of 7 hours,
    the distance traveled from the airport is 1200 km. -/
theorem plane_distance (speed_east speed_west total_time : ℝ) 
    (h1 : speed_east = 300)
    (h2 : speed_west = 400)
    (h3 : total_time = 7) : 
  (total_time * speed_east * speed_west) / (speed_east + speed_west) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_plane_distance_l55_5518


namespace NUMINAMATH_CALUDE_sword_length_proof_l55_5555

/-- The length of Christopher's sword in inches -/
def christopher_sword_length : ℕ := 15

/-- The length of Jameson's sword in inches -/
def jameson_sword_length : ℕ := 2 * christopher_sword_length + 3

/-- The length of June's sword in inches -/
def june_sword_length : ℕ := jameson_sword_length + 5

theorem sword_length_proof :
  (jameson_sword_length = 2 * christopher_sword_length + 3) ∧
  (june_sword_length = jameson_sword_length + 5) ∧
  (june_sword_length = christopher_sword_length + 23) →
  christopher_sword_length = 15 := by
sorry

#eval christopher_sword_length

end NUMINAMATH_CALUDE_sword_length_proof_l55_5555


namespace NUMINAMATH_CALUDE_range_of_a_given_decreasing_function_l55_5598

-- Define a decreasing function on the real line
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- State the theorem
theorem range_of_a_given_decreasing_function (f : ℝ → ℝ) (h : DecreasingFunction f) :
  ∀ a : ℝ, a ∈ Set.univ :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_given_decreasing_function_l55_5598


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l55_5556

/-- The number of passengers who landed on time in Newberg last year -/
def on_time_passengers : ℕ := 14507

/-- The number of passengers who landed late in Newberg last year -/
def late_passengers : ℕ := 213

/-- The number of passengers who had connecting flights in Newberg last year -/
def connecting_passengers : ℕ := 320

/-- The total number of passengers who landed in Newberg last year -/
def total_passengers : ℕ := on_time_passengers + late_passengers + connecting_passengers

theorem newberg_airport_passengers :
  total_passengers = 15040 :=
sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l55_5556


namespace NUMINAMATH_CALUDE_solution_product_l55_5594

theorem solution_product (p q : ℝ) : 
  (p - 3) * (3 * p + 18) = p^2 - 15 * p + 54 →
  (q - 3) * (3 * q + 18) = q^2 - 15 * q + 54 →
  p ≠ q →
  (p + 2) * (q + 2) = -80 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l55_5594


namespace NUMINAMATH_CALUDE_inequality_proof_l55_5506

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^1999 + b^2000 ≥ a^2000 + b^2001) : 
  a^2000 + b^2000 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l55_5506


namespace NUMINAMATH_CALUDE_sum_in_base_6_l55_5535

/-- Converts a number from base 6 to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem sum_in_base_6 :
  let a := toBase10 [4, 3, 2, 1]  -- 1234₆
  let b := toBase10 [4, 3, 2]     -- 234₆
  let c := toBase10 [4, 3]        -- 34₆
  toBase6 (a + b + c) = [0, 5, 5, 2] -- 2550₆
  := by sorry

end NUMINAMATH_CALUDE_sum_in_base_6_l55_5535


namespace NUMINAMATH_CALUDE_three_digit_sum_l55_5524

theorem three_digit_sum (A B : ℕ) : 
  A < 10 → 
  B < 10 → 
  100 ≤ 14 * 10 + A → 
  14 * 10 + A < 1000 → 
  100 ≤ 100 * B + 73 → 
  100 * B + 73 < 1000 → 
  14 * 10 + A + 100 * B + 73 = 418 → 
  A = 5 := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_l55_5524


namespace NUMINAMATH_CALUDE_eighth_of_two_to_forty_l55_5515

theorem eighth_of_two_to_forty (x : ℤ) : (1 / 8 : ℚ) * (2 ^ 40 : ℚ) = (2 : ℚ) ^ x → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_two_to_forty_l55_5515


namespace NUMINAMATH_CALUDE_total_distance_walked_l55_5538

def distance_school_to_david : ℝ := 0.2
def distance_david_to_home : ℝ := 0.7

theorem total_distance_walked : 
  distance_school_to_david + distance_david_to_home = 0.9 := by sorry

end NUMINAMATH_CALUDE_total_distance_walked_l55_5538


namespace NUMINAMATH_CALUDE_vector_scalar_product_l55_5537

/-- Given two vectors in R², prove that their scalar product equals 14 -/
theorem vector_scalar_product (a b : ℝ × ℝ) : 
  a = (2, 3) → b = (-1, 2) → (a + 2 • b) • b = 14 := by
  sorry

end NUMINAMATH_CALUDE_vector_scalar_product_l55_5537


namespace NUMINAMATH_CALUDE_hyperbola_m_equation_l55_5595

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of a hyperbola in the form y²/a - x²/b = c -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a - x^2 / h.b = h.c

/-- Two hyperbolas have common asymptotes if they have the same a/b ratio -/
def common_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

theorem hyperbola_m_equation 
  (n : Hyperbola)
  (hn_eq : hyperbola_equation n = fun x y ↦ y^2 / 4 - x^2 / 2 = 1)
  (m : Hyperbola)
  (hm_asymp : common_asymptotes m n)
  (hm_point : hyperbola_equation m (-2) 4) :
  hyperbola_equation m = fun x y ↦ y^2 / 8 - x^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_equation_l55_5595


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l55_5574

theorem binomial_expansion_example : (7 + 2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l55_5574


namespace NUMINAMATH_CALUDE_rationalize_denominator_l55_5542

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l55_5542


namespace NUMINAMATH_CALUDE_students_behind_yoongi_count_l55_5522

/-- The number of students in the line. -/
def total_students : ℕ := 20

/-- Jungkook's position in the line. -/
def jungkook_position : ℕ := 3

/-- The number of students between Jungkook and Yoongi. -/
def students_between : ℕ := 5

/-- Yoongi's position in the line. -/
def yoongi_position : ℕ := jungkook_position + students_between + 1

/-- The number of students behind Yoongi. -/
def students_behind_yoongi : ℕ := total_students - yoongi_position

theorem students_behind_yoongi_count : students_behind_yoongi = 11 := by sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_count_l55_5522


namespace NUMINAMATH_CALUDE_sin_sum_identity_l55_5585

theorem sin_sum_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l55_5585


namespace NUMINAMATH_CALUDE_inequality_proof_l55_5561

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) : 
  (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l55_5561


namespace NUMINAMATH_CALUDE_machine_production_l55_5516

/-- The number of shirts produced by a machine in a given time -/
def shirts_produced (shirts_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  shirts_per_minute * minutes

/-- Theorem: A machine that produces 6 shirts per minute, operating for 12 minutes, will produce 72 shirts -/
theorem machine_production :
  shirts_produced 6 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_l55_5516


namespace NUMINAMATH_CALUDE_distribute_five_two_correct_l55_5539

def number_of_correct_locations : ℕ := 2
def total_objects : ℕ := 5

/-- The number of ways to distribute n distinct objects to n distinct locations
    such that exactly k objects are in their correct locations -/
def distribute (n k : ℕ) : ℕ := sorry

theorem distribute_five_two_correct :
  distribute total_objects number_of_correct_locations = 20 := by sorry

end NUMINAMATH_CALUDE_distribute_five_two_correct_l55_5539


namespace NUMINAMATH_CALUDE_composition_three_reflections_is_glide_reflection_l55_5525

-- Define a type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for lines in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a reflection transformation
def Reflection (l : Line2D) : Point2D → Point2D := sorry

-- Define a translation transformation
def Translation (dx dy : ℝ) : Point2D → Point2D := sorry

-- Define a glide reflection transformation
def GlideReflection (l : Line2D) (t : ℝ) : Point2D → Point2D := sorry

-- Define a predicate to check if three lines pass through the same point
def passThroughSamePoint (l1 l2 l3 : Line2D) : Prop := sorry

-- Define a predicate to check if three lines are parallel to the same line
def parallelToSameLine (l1 l2 l3 : Line2D) : Prop := sorry

-- Theorem statement
theorem composition_three_reflections_is_glide_reflection 
  (l1 l2 l3 : Line2D) 
  (h1 : ¬ passThroughSamePoint l1 l2 l3) 
  (h2 : ¬ parallelToSameLine l1 l2 l3) :
  ∃ (l : Line2D) (t : ℝ), 
    ∀ p : Point2D, 
      (Reflection l3 ∘ Reflection l2 ∘ Reflection l1) p = GlideReflection l t p :=
sorry

end NUMINAMATH_CALUDE_composition_three_reflections_is_glide_reflection_l55_5525


namespace NUMINAMATH_CALUDE_inverse_negation_correct_l55_5590

/-- Represents a triangle ABC -/
structure Triangle where
  isIsosceles : Bool
  hasEqualAngles : Bool

/-- The original proposition -/
def originalProposition (t : Triangle) : Prop :=
  ¬t.isIsosceles → ¬t.hasEqualAngles

/-- The inverse negation of the original proposition -/
def inverseNegation (t : Triangle) : Prop :=
  t.hasEqualAngles → t.isIsosceles

/-- Theorem stating that the inverse negation is correct -/
theorem inverse_negation_correct :
  ∀ t : Triangle, inverseNegation t ↔ ¬(¬originalProposition t) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_correct_l55_5590


namespace NUMINAMATH_CALUDE_systematic_sampling_first_stage_l55_5589

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a stage in the sampling process -/
inductive SamplingStage
  | First
  | Later

/-- Defines the relationship between sampling methods and stages -/
def sampling_relationship (method : SamplingMethod) (stage : SamplingStage) : Prop :=
  match method, stage with
  | SamplingMethod.Systematic, SamplingStage.First => true
  | _, _ => false

/-- Theorem stating that systematic sampling generally uses simple random sampling in the first stage -/
theorem systematic_sampling_first_stage :
  sampling_relationship SamplingMethod.Systematic SamplingStage.First = true :=
by
  sorry

#check systematic_sampling_first_stage

end NUMINAMATH_CALUDE_systematic_sampling_first_stage_l55_5589


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l55_5510

-- Define the constants for the cylinder
def cylinder_height : ℝ := 10
def cylinder_diameter : ℝ := 10

-- Define the theorem
theorem sphere_cylinder_equal_area (r : ℝ) :
  (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) →
  r = 5 := by
  sorry


end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l55_5510


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l55_5504

theorem binomial_coefficient_equality (x : ℕ) : 
  Nat.choose 20 (3 * x) = Nat.choose 20 (x + 4) → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l55_5504


namespace NUMINAMATH_CALUDE_base_conversion_problem_l55_5579

theorem base_conversion_problem (n C D : ℕ) : 
  n > 0 ∧ 
  C < 8 ∧ 
  D < 6 ∧ 
  n = 8 * C + D ∧ 
  n = 6 * D + C → 
  n = 43 := by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l55_5579


namespace NUMINAMATH_CALUDE_total_present_age_is_72_l55_5550

/-- Given three people p, q, and r, prove that their total present age is 72 years -/
theorem total_present_age_is_72 
  (p q r : ℕ) -- Present ages of p, q, and r
  (h1 : p - 12 = (q - 12) / 2) -- 12 years ago, p was half of q's age
  (h2 : r - 12 = (p - 12) + (q - 12) - 3) -- r was 3 years younger than the sum of p and q's ages 12 years ago
  (h3 : ∃ (x : ℕ), p = 3*x ∧ q = 4*x ∧ r = 5*x) -- The ratio of their present ages is 3 : 4 : 5
  : p + q + r = 72 := by
  sorry


end NUMINAMATH_CALUDE_total_present_age_is_72_l55_5550


namespace NUMINAMATH_CALUDE_second_year_interest_rate_problem_solution_l55_5558

/-- Given an initial investment, interest rates, and final value, calculate the second year's interest rate -/
theorem second_year_interest_rate 
  (initial_investment : ℝ) 
  (first_year_rate : ℝ) 
  (final_value : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate)
  let second_year_rate := (final_value / first_year_value) - 1
  second_year_rate * 100

/-- Prove that the second year's interest rate is 4% given the problem conditions -/
theorem problem_solution :
  second_year_interest_rate 15000 0.05 16380 = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_problem_solution_l55_5558


namespace NUMINAMATH_CALUDE_a_range_l55_5580

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_domain : ∀ x, f x ≠ 0 → -7 < x ∧ x < 7
axiom f_condition : ∀ a, f (1 - a) + f (2*a - 5) < 0

-- Theorem statement
theorem a_range : 
  ∃ a₁ a₂, a₁ = 4 ∧ a₂ = 6 ∧ 
  (∀ a, (f (1 - a) + f (2*a - 5) < 0) → a₁ < a ∧ a < a₂) :=
sorry

end NUMINAMATH_CALUDE_a_range_l55_5580


namespace NUMINAMATH_CALUDE_correct_weight_calculation_l55_5588

theorem correct_weight_calculation (class_size : ℕ) 
  (incorrect_avg : ℚ) (misread_weight : ℚ) (correct_avg : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  misread_weight = 56 →
  correct_avg = 58.6 →
  (class_size : ℚ) * correct_avg - (class_size : ℚ) * incorrect_avg + misread_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_correct_weight_calculation_l55_5588


namespace NUMINAMATH_CALUDE_player_A_can_destroy_six_cups_six_cups_is_maximum_l55_5560

/-- Represents the state of the game with cups and pebbles -/
structure GameState where
  cups : ℕ
  pebbles : List ℕ

/-- Represents a move in the game -/
inductive Move
  | redistribute : List ℕ → Move
  | destroy_empty : Move
  | switch : ℕ → ℕ → Move

/-- Player A's strategy function -/
def player_A_strategy (state : GameState) : List ℕ :=
  sorry

/-- Player B's action function -/
def player_B_action (state : GameState) (move : Move) : GameState :=
  sorry

/-- Simulates the game for a given number of moves -/
def play_game (initial_state : GameState) (num_moves : ℕ) : GameState :=
  sorry

/-- Theorem stating that player A can guarantee at least 6 cups are destroyed -/
theorem player_A_can_destroy_six_cups :
  ∃ (strategy : GameState → List ℕ),
    ∀ (num_moves : ℕ),
      let final_state := play_game {cups := 10, pebbles := List.replicate 10 10} num_moves
      final_state.cups ≤ 4 :=
sorry

/-- Theorem stating that 6 is the maximum number of cups that can be guaranteed to be destroyed -/
theorem six_cups_is_maximum :
  ∀ (strategy : GameState → List ℕ),
    ∃ (num_moves : ℕ),
      let final_state := play_game {cups := 10, pebbles := List.replicate 10 10} num_moves
      final_state.cups > 4 :=
sorry

end NUMINAMATH_CALUDE_player_A_can_destroy_six_cups_six_cups_is_maximum_l55_5560


namespace NUMINAMATH_CALUDE_inequality_system_solution_l55_5591

theorem inequality_system_solution (x : ℝ) : 
  ((x + 3) / 2 ≤ x + 2 ∧ 2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l55_5591


namespace NUMINAMATH_CALUDE_common_roots_product_l55_5592

-- Define the two polynomial equations
def poly1 (K : ℝ) (x : ℝ) : ℝ := x^3 + K*x + 20
def poly2 (L : ℝ) (x : ℝ) : ℝ := x^3 + L*x^2 + 100

-- Define the theorem
theorem common_roots_product (K L : ℝ) :
  (∃ (u v : ℝ), u ≠ v ∧ 
    poly1 K u = 0 ∧ poly1 K v = 0 ∧
    poly2 L u = 0 ∧ poly2 L v = 0) →
  (∃ (p : ℝ), p = 10 * Real.rpow 2 (1/3) ∧
    ∃ (u v : ℝ), u ≠ v ∧ 
      poly1 K u = 0 ∧ poly1 K v = 0 ∧
      poly2 L u = 0 ∧ poly2 L v = 0 ∧
      u * v = p) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l55_5592


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l55_5567

theorem nested_fraction_equality : (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l55_5567


namespace NUMINAMATH_CALUDE_square_ceiling_lights_l55_5549

/-- The number of lights on each side of the square ceiling -/
def lights_per_side : ℕ := 20

/-- The minimum number of lights needed for the entire square ceiling -/
def min_lights_needed : ℕ := 4 * lights_per_side - 4

theorem square_ceiling_lights : min_lights_needed = 76 := by
  sorry

end NUMINAMATH_CALUDE_square_ceiling_lights_l55_5549


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l55_5533

theorem smallest_n_for_factorization : 
  let can_be_factored (n : ℤ) := ∃ (A B : ℤ), 
    (A * B = 60) ∧ 
    (6 * B + A = n) ∧ 
    (∀ x, 6 * x^2 + n * x + 60 = (6 * x + A) * (x + B))
  ∀ n : ℤ, can_be_factored n → n ≥ 66
  ∧ can_be_factored 66 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_l55_5533


namespace NUMINAMATH_CALUDE_prob_not_square_l55_5593

def total_figures : ℕ := 10
def num_triangles : ℕ := 5
def num_squares : ℕ := 3
def num_circles : ℕ := 2

theorem prob_not_square :
  (num_triangles + num_circles : ℚ) / total_figures = 7 / 10 :=
sorry

end NUMINAMATH_CALUDE_prob_not_square_l55_5593


namespace NUMINAMATH_CALUDE_even_function_property_l55_5543

/-- A function f is even on an interval [-a, a] -/
def IsEvenOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-a) a, f x = f (-x)

theorem even_function_property
  (f : ℝ → ℝ) (h_even : IsEvenOn f 6) (h_gt : f 3 > f 1) :
  f (-1) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l55_5543


namespace NUMINAMATH_CALUDE_circle_area_ratio_l55_5530

/-- Given two circles X and Y, if an arc of 60° on circle X has the same length as an arc of 40° on circle Y, 
    then the ratio of the area of circle X to the area of circle Y is 9/4. -/
theorem circle_area_ratio (X Y : ℝ → ℝ → Prop) (R_X R_Y : ℝ) :
  (∃ L : ℝ, L = (60 / 360) * (2 * Real.pi * R_X) ∧ L = (40 / 360) * (2 * Real.pi * R_Y)) →
  (R_X > 0 ∧ R_Y > 0) →
  (X = λ x y => (x - 0)^2 + (y - 0)^2 = R_X^2) →
  (Y = λ x y => (x - 0)^2 + (y - 0)^2 = R_Y^2) →
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l55_5530


namespace NUMINAMATH_CALUDE_solve_system_l55_5562

-- Define the variables x and y
variable (x y : ℤ)

-- State the theorem
theorem solve_system : 
  (3:ℝ)^x = 27^(y+1) → (16:ℝ)^y = 2^(x-8) → 2*x + y = -29 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l55_5562


namespace NUMINAMATH_CALUDE_natasha_dimes_problem_l55_5573

theorem natasha_dimes_problem :
  ∃! n : ℕ, 100 < n ∧ n < 200 ∧
    n % 3 = 2 ∧
    n % 4 = 2 ∧
    n % 5 = 2 ∧
    n % 7 = 2 ∧
    n = 182 := by
  sorry

end NUMINAMATH_CALUDE_natasha_dimes_problem_l55_5573


namespace NUMINAMATH_CALUDE_edward_lawn_mowing_earnings_l55_5566

/-- Edward's lawn mowing earnings problem -/
theorem edward_lawn_mowing_earnings 
  (rate : ℕ) -- Rate per lawn mowed
  (total_lawns : ℕ) -- Total number of lawns to mow
  (forgotten_lawns : ℕ) -- Number of lawns forgotten
  (h1 : rate = 4) -- Edward earns 4 dollars for each lawn
  (h2 : total_lawns = 17) -- Edward had 17 lawns to mow
  (h3 : forgotten_lawns = 9) -- Edward forgot to mow 9 lawns
  : (total_lawns - forgotten_lawns) * rate = 32 := by
  sorry

end NUMINAMATH_CALUDE_edward_lawn_mowing_earnings_l55_5566


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l55_5583

theorem geometric_sequence_sum (u v : ℝ) : 
  (∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧
    u = a * r^3 ∧
    v = a * r^4 ∧
    4 = a * r^5 ∧
    1 = a * r^6) →
  u + v = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l55_5583


namespace NUMINAMATH_CALUDE_sin_equals_cos_690_l55_5569

theorem sin_equals_cos_690 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) 
  (h2 : Real.sin (n * π / 180) = Real.cos (690 * π / 180)) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_sin_equals_cos_690_l55_5569


namespace NUMINAMATH_CALUDE_count_square_functions_l55_5552

-- Define the type for our function
def SquareFunction := Set ℤ → Set ℤ

-- Define what it means for a function to be in our family
def is_in_family (f : SquareFunction) : Prop :=
  ∃ (domain : Set ℤ),
    (∀ x ∈ domain, f domain = {y | ∃ x ∈ domain, y = x^2}) ∧
    (f domain = {1, 4})

-- State the theorem
theorem count_square_functions : 
  ∃! (n : ℕ), ∃ (functions : Finset SquareFunction),
    functions.card = n ∧
    (∀ f ∈ functions, is_in_family f) ∧
    (∀ f, is_in_family f → f ∈ functions) ∧
    n = 8 := by sorry

end NUMINAMATH_CALUDE_count_square_functions_l55_5552


namespace NUMINAMATH_CALUDE_solve_equation_l55_5528

theorem solve_equation (m n : ℝ) : 
  |m - 2| + n^2 - 8*n + 16 = 0 → m = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l55_5528


namespace NUMINAMATH_CALUDE_sqrt_expression_defined_l55_5599

theorem sqrt_expression_defined (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 - 2 * (a - 1) * x + 3 * a - 3 ≥ 0) ↔ a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_expression_defined_l55_5599


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l55_5568

/-- A pentagon formed by cutting a triangular corner from a rectangular piece of paper -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {13, 19, 20, 25, 31}

/-- The area of a CornerCutPentagon -/
def area (p : CornerCutPentagon) : ℕ :=
  745

theorem corner_cut_pentagon_area (p : CornerCutPentagon) : area p = 745 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l55_5568


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l55_5578

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let descentDistances := List.range (numBounces + 1) |>.map (fun n => initialHeight * bounceRatio^n)
  let ascentDistances := List.range numBounces |>.map (fun n => initialHeight * bounceRatio^(n+1))
  (descentDistances.sum + ascentDistances.sum)

/-- The problem statement -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 20
  let bounceRatio : ℝ := 2/3
  let numBounces : ℕ := 4
  abs (totalDistance initialHeight bounceRatio numBounces - 80) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l55_5578


namespace NUMINAMATH_CALUDE_box_cube_volume_l55_5557

/-- Given a box with dimensions 10 cm x 18 cm x 4 cm, filled completely with 60 identical cubes,
    the volume of each cube is 8 cubic centimeters. -/
theorem box_cube_volume (length width height : ℕ) (num_cubes : ℕ) (cube_volume : ℕ) :
  length = 10 ∧ width = 18 ∧ height = 4 ∧ num_cubes = 60 →
  length * width * height = num_cubes * cube_volume →
  cube_volume = 8 := by
  sorry

#check box_cube_volume

end NUMINAMATH_CALUDE_box_cube_volume_l55_5557


namespace NUMINAMATH_CALUDE_toy_price_reduction_l55_5517

theorem toy_price_reduction :
  ∃! x : ℕ, 1 ≤ x ∧ x ≤ 12 ∧
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ (13 - x) * y = 781) ∧
  (∀ z : ℕ, z > x → ¬∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ (13 - z) * y = 781) :=
by sorry

end NUMINAMATH_CALUDE_toy_price_reduction_l55_5517


namespace NUMINAMATH_CALUDE_custom_operation_result_l55_5559

-- Define the custom operation *
def star (a b : ℕ) : ℕ := a + 2 * b

-- State the theorem
theorem custom_operation_result : star (star 2 4) 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l55_5559


namespace NUMINAMATH_CALUDE_min_distance_to_line_l55_5576

theorem min_distance_to_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  A = (-2, 0) →
  B = (0, 3) →
  (∀ x y, l x y ↔ x - y + 1 = 0) →
  ∃ P : ℝ × ℝ, l P.1 P.2 ∧
    (∀ Q : ℝ × ℝ, l Q.1 Q.2 → Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
                               Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤
                               Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) +
                               Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)) ∧
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = Real.sqrt 17 :=
by sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l55_5576


namespace NUMINAMATH_CALUDE_children_share_distribution_l55_5544

theorem children_share_distribution (total : ℝ) (share_ac : ℝ) 
  (h1 : total = 15800)
  (h2 : share_ac = 7022.222222222222) :
  total - share_ac = 8777.777777777778 := by
  sorry

end NUMINAMATH_CALUDE_children_share_distribution_l55_5544


namespace NUMINAMATH_CALUDE_buyOneGetOneFreeIsCheaper_finalCostIs216_l55_5501

/-- Represents the total cost of Pauline's purchase with a given discount and sales tax. -/
def totalCost (totalBeforeTax : ℝ) (selectedItemsTotal : ℝ) (discount : ℝ) (salesTaxRate : ℝ) : ℝ :=
  let discountedTotal := totalBeforeTax - selectedItemsTotal * discount
  discountedTotal * (1 + salesTaxRate)

/-- Theorem stating that the Buy One, Get One Free offer is cheaper than the 15% discount offer. -/
theorem buyOneGetOneFreeIsCheaper :
  let totalBeforeTax : ℝ := 250
  let selectedItemsTotal : ℝ := 100
  let remainingItemsTotal : ℝ := totalBeforeTax - selectedItemsTotal
  let discountRate : ℝ := 0.15
  let buyOneGetOneFreeDiscount : ℝ := 0.5
  let salesTaxRate : ℝ := 0.08
  totalCost totalBeforeTax selectedItemsTotal buyOneGetOneFreeDiscount salesTaxRate <
  totalCost totalBeforeTax selectedItemsTotal discountRate salesTaxRate :=
by sorry

/-- Calculates the final cost with the Buy One, Get One Free offer. -/
def finalCost : ℝ :=
  let totalBeforeTax : ℝ := 250
  let selectedItemsTotal : ℝ := 100
  let buyOneGetOneFreeDiscount : ℝ := 0.5
  let salesTaxRate : ℝ := 0.08
  totalCost totalBeforeTax selectedItemsTotal buyOneGetOneFreeDiscount salesTaxRate

/-- Theorem stating that the final cost is $216. -/
theorem finalCostIs216 : finalCost = 216 :=
by sorry

end NUMINAMATH_CALUDE_buyOneGetOneFreeIsCheaper_finalCostIs216_l55_5501


namespace NUMINAMATH_CALUDE_orthogonal_families_l55_5551

/-- A family of curves in the x-y plane -/
structure Curve :=
  (equation : ℝ → ℝ → ℝ → Prop)

/-- The given family of curves x^2 + y^2 = 2ax -/
def given_family : Curve :=
  ⟨λ a x y ↦ x^2 + y^2 = 2*a*x⟩

/-- The orthogonal family of curves x^2 + y^2 = Cy -/
def orthogonal_family : Curve :=
  ⟨λ C x y ↦ x^2 + y^2 = C*y⟩

/-- Two curves are orthogonal if their tangent lines are perpendicular at each intersection point -/
def orthogonal (c1 c2 : Curve) : Prop :=
  ∀ a C x y, c1.equation a x y → c2.equation C x y →
    ∃ m1 m2 : ℝ, (m1 * m2 = -1) ∧
      (∀ h, h ≠ 0 → (c1.equation a (x + h) (y + m1*h) ↔ c1.equation a x y)) ∧
      (∀ h, h ≠ 0 → (c2.equation C (x + h) (y + m2*h) ↔ c2.equation C x y))

/-- The main theorem stating that the given family and the orthogonal family are indeed orthogonal -/
theorem orthogonal_families : orthogonal given_family orthogonal_family :=
sorry

end NUMINAMATH_CALUDE_orthogonal_families_l55_5551


namespace NUMINAMATH_CALUDE_paperclip_capacity_l55_5563

/-- Given that a box of volume 18 cm³ can hold 60 paperclips, and the storage density
    decreases by 10% in larger boxes, prove that a box of volume 72 cm³ can hold 216 paperclips. -/
theorem paperclip_capacity (small_volume small_capacity large_volume : ℝ) 
    (h1 : small_volume = 18)
    (h2 : small_capacity = 60)
    (h3 : large_volume = 72)
    (h4 : large_volume > small_volume) :
    let density_ratio := large_volume / small_volume
    let unadjusted_capacity := small_capacity * density_ratio
    let adjusted_capacity := unadjusted_capacity * 0.9
    adjusted_capacity = 216 := by
  sorry


end NUMINAMATH_CALUDE_paperclip_capacity_l55_5563


namespace NUMINAMATH_CALUDE_triangle_side_length_l55_5575

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  (1 / 2) * b * c * Real.sin A = 1 →
  a = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l55_5575


namespace NUMINAMATH_CALUDE_new_device_significantly_improved_l55_5564

-- Define the sample means and variances
def x_bar : ℝ := 10
def y_bar : ℝ := 10.3
def s1_squared : ℝ := 0.036
def s2_squared : ℝ := 0.04

-- Define the significant improvement criterion
def significant_improvement (x_bar y_bar s1_squared s2_squared : ℝ) : Prop :=
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_squared + s2_squared) / 10)

-- Theorem statement
theorem new_device_significantly_improved :
  significant_improvement x_bar y_bar s1_squared s2_squared := by
  sorry


end NUMINAMATH_CALUDE_new_device_significantly_improved_l55_5564


namespace NUMINAMATH_CALUDE_ellipse_area_condition_l55_5597

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    if the area of the right triangle formed by a point on the ellipse,
    the center, and the right focus is √3, then a² = 2√3 + 4 and b² = 2√3 -/
theorem ellipse_area_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  let triangle_area (x y : ℝ) := (1/2) * (c/2) * y
  ∃ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ∧ triangle_area x y = Real.sqrt 3 →
  a^2 = 2 * Real.sqrt 3 + 4 ∧ b^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_condition_l55_5597


namespace NUMINAMATH_CALUDE_not_power_of_two_l55_5541

theorem not_power_of_two (a b : ℕ+) : ¬ ∃ k : ℕ, (36 * a + b) * (a + 36 * b) = 2^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_two_l55_5541


namespace NUMINAMATH_CALUDE_unit_circle_arc_angle_l55_5546

/-- The central angle (in radians) corresponding to an arc of length 1 in a unit circle is 1. -/
theorem unit_circle_arc_angle (θ : ℝ) : θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_arc_angle_l55_5546


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l55_5582

def U : Set Nat := {0,1,2,3,4,5,6,7,8,9}
def A : Set Nat := {0,1,3,5,8}
def B : Set Nat := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l55_5582


namespace NUMINAMATH_CALUDE_equation_solution_l55_5507

theorem equation_solution : ∃ x : ℝ, (x / 2 - 1 = 3) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l55_5507


namespace NUMINAMATH_CALUDE_equation_3x_eq_4y_is_linear_l55_5508

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The equation 3x = 4y is a linear equation in two variables. -/
theorem equation_3x_eq_4y_is_linear :
  IsLinearEquationInTwoVariables (fun x y => 3 * x - 4 * y) :=
sorry

end NUMINAMATH_CALUDE_equation_3x_eq_4y_is_linear_l55_5508


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l55_5570

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions for a valid triangle
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the quadratic expression
def quadratic_expr (t : Triangle) (x : ℝ) : ℝ :=
  t.b^2 * x^2 + (t.b^2 + t.c^2 - t.a^2) * x + t.c^2

-- Theorem statement
theorem quadratic_always_positive (t : Triangle) :
  ∀ x : ℝ, quadratic_expr t x > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l55_5570


namespace NUMINAMATH_CALUDE_charity_race_fundraising_l55_5553

/-- Proves that the amount raised by each of the ten students is $20 -/
theorem charity_race_fundraising
  (total_students : ℕ)
  (special_students : ℕ)
  (regular_amount : ℕ)
  (total_raised : ℕ)
  (h1 : total_students = 30)
  (h2 : special_students = 10)
  (h3 : regular_amount = 30)
  (h4 : total_raised = 800)
  (h5 : total_raised = special_students * X + (total_students - special_students) * regular_amount)
  : X = 20 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_fundraising_l55_5553


namespace NUMINAMATH_CALUDE_max_self_intersection_points_seven_segments_l55_5526

/-- A closed polyline is a sequence of connected line segments that form a closed loop. -/
def ClosedPolyline (n : ℕ) := Fin n → ℝ × ℝ

/-- The number of self-intersection points in a closed polyline. -/
def selfIntersectionPoints (p : ClosedPolyline 7) : ℕ := sorry

/-- The maximum number of self-intersection points in any closed polyline with 7 segments. -/
def maxSelfIntersectionPoints : ℕ := sorry

/-- Theorem: The maximum number of self-intersection points in a closed polyline with 7 segments is 14. -/
theorem max_self_intersection_points_seven_segments :
  maxSelfIntersectionPoints = 14 := by sorry

end NUMINAMATH_CALUDE_max_self_intersection_points_seven_segments_l55_5526


namespace NUMINAMATH_CALUDE_coffee_thermoses_count_l55_5512

-- Define the conversion factor from gallons to pints
def gallons_to_pints : ℚ := 8

-- Define the total amount of coffee in gallons
def total_coffee_gallons : ℚ := 9/2

-- Define the number of thermoses Genevieve drank
def thermoses_consumed : ℕ := 3

-- Define the amount of coffee Genevieve consumed in pints
def coffee_consumed_pints : ℕ := 6

-- Theorem to prove
theorem coffee_thermoses_count : 
  (total_coffee_gallons * gallons_to_pints) / (coffee_consumed_pints / thermoses_consumed) = 18 := by
  sorry

end NUMINAMATH_CALUDE_coffee_thermoses_count_l55_5512


namespace NUMINAMATH_CALUDE_die_visible_combinations_l55_5505

/-- A die is represented as a cube with 6 faces, 12 edges, and 8 vertices -/
structure Die :=
  (faces : Fin 6)
  (edges : Fin 12)
  (vertices : Fin 8)

/-- The number of visible faces from a point in space can be 1, 2, or 3 -/
inductive VisibleFaces
  | one
  | two
  | three

/-- The number of combinations for each type of view -/
def combinationsForView (v : VisibleFaces) : ℕ :=
  match v with
  | VisibleFaces.one => 6    -- One face visible: 6 possibilities
  | VisibleFaces.two => 12   -- Two faces visible: 12 possibilities
  | VisibleFaces.three => 8  -- Three faces visible: 8 possibilities

/-- The total number of different visible face combinations -/
def totalCombinations (d : Die) : ℕ :=
  (combinationsForView VisibleFaces.one) +
  (combinationsForView VisibleFaces.two) +
  (combinationsForView VisibleFaces.three)

theorem die_visible_combinations (d : Die) :
  totalCombinations d = 26 := by
  sorry

end NUMINAMATH_CALUDE_die_visible_combinations_l55_5505


namespace NUMINAMATH_CALUDE_pyramid_volume_l55_5547

/-- 
Given a pyramid with a rhombic base:
- d₁ and d₂ are the diagonals of the rhombus base
- d₁ > d₂
- The height of the pyramid passes through the vertex of the acute angle of the rhombus
- Q is the area of the diagonal section conducted through the shorter diagonal

This theorem states that the volume of such a pyramid is (d₁ / 12) * √(16Q² - d₁²d₂²)
-/
theorem pyramid_volume (d₁ d₂ Q : ℝ) (h₁ : d₁ > d₂) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : Q > 0) :
  let volume := d₁ / 12 * Real.sqrt (16 * Q^2 - d₁^2 * d₂^2)
  volume > 0 ∧ volume^3 = (d₁^3 / 1728) * (16 * Q^2 - d₁^2 * d₂^2) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l55_5547


namespace NUMINAMATH_CALUDE_siblings_age_sum_l55_5565

theorem siblings_age_sum (ages : Fin 6 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ ages i) 
  (h_mean : (Finset.univ.sum ages) / 6 = 10)
  (h_median : (ages (Fin.mk 2 (by norm_num)) + ages (Fin.mk 3 (by norm_num))) / 2 = 12) :
  ages 0 + ages 5 = 12 := by
sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l55_5565


namespace NUMINAMATH_CALUDE_integer_between_sqrt_11_and_sqrt_19_l55_5509

theorem integer_between_sqrt_11_and_sqrt_19 :
  ∃! x : ℤ, Real.sqrt 11 < x ∧ x < Real.sqrt 19 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_11_and_sqrt_19_l55_5509


namespace NUMINAMATH_CALUDE_marketing_specialization_percentage_l55_5511

theorem marketing_specialization_percentage
  (initial_finance : Real)
  (increased_finance : Real)
  (marketing_after_increase : Real)
  (h1 : initial_finance = 88)
  (h2 : increased_finance = 90)
  (h3 : marketing_after_increase = 43.333333333333336)
  (h4 : increased_finance - initial_finance = 2) :
  initial_finance + marketing_after_increase + 2 = 45.333333333333336 + 88 := by
  sorry

end NUMINAMATH_CALUDE_marketing_specialization_percentage_l55_5511


namespace NUMINAMATH_CALUDE_symmetry_of_curves_l55_5571

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the point of symmetry
def point_of_symmetry : ℝ × ℝ := (3, 5)

-- Define the symmetric curve
def symmetric_curve (x y : ℝ) : Prop := (x - 6)^2 + 4*(y - 10)^2 = 4

-- Theorem statement
theorem symmetry_of_curves :
  ∀ (x y : ℝ), original_curve x y ↔ symmetric_curve (2*point_of_symmetry.1 - x) (2*point_of_symmetry.2 - y) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_curves_l55_5571


namespace NUMINAMATH_CALUDE_linear_system_solution_l55_5584

theorem linear_system_solution (x y : ℚ) 
  (eq1 : 3 * x - y = 9) 
  (eq2 : 2 * y - x = 1) : 
  5 * x + 4 * y = 39 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l55_5584


namespace NUMINAMATH_CALUDE_range_of_a_l55_5572

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≥ 1
def q (x a : ℝ) : Prop := x ≤ a

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- Theorem statement
theorem range_of_a (x a : ℝ) :
  (∀ x, necessary_not_sufficient (p x) (q x a)) →
  a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l55_5572


namespace NUMINAMATH_CALUDE_jewelry_sales_problem_l55_5513

/-- Represents the jewelry sales problem --/
theorem jewelry_sales_problem 
  (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (bracelets_sold earrings_sold ensembles_sold : ℕ)
  (total_revenue : ℚ)
  (h1 : necklace_price = 25)
  (h2 : bracelet_price = 15)
  (h3 : earring_price = 10)
  (h4 : ensemble_price = 45)
  (h5 : bracelets_sold = 10)
  (h6 : earrings_sold = 20)
  (h7 : ensembles_sold = 2)
  (h8 : total_revenue = 565) :
  ∃ (necklaces_sold : ℕ), 
    necklace_price * necklaces_sold + 
    bracelet_price * bracelets_sold + 
    earring_price * earrings_sold + 
    ensemble_price * ensembles_sold = total_revenue ∧
    necklaces_sold = 5 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_sales_problem_l55_5513


namespace NUMINAMATH_CALUDE_dinosaur_book_cost_l55_5548

def dictionary_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def total_cost : ℕ := 37

theorem dinosaur_book_cost :
  ∃ (dinosaur_cost : ℕ), 
    dictionary_cost + dinosaur_cost + cookbook_cost = total_cost ∧
    dinosaur_cost = 19 :=
by sorry

end NUMINAMATH_CALUDE_dinosaur_book_cost_l55_5548


namespace NUMINAMATH_CALUDE_binary_subtraction_l55_5581

def binary_to_decimal (b : ℕ) : ℕ := 
  if b = 0 then 0
  else if b % 10 = 1 then 1 + 2 * (binary_to_decimal (b / 10))
  else 2 * (binary_to_decimal (b / 10))

def binary_1111111111 : ℕ := 1111111111
def binary_11111 : ℕ := 11111

theorem binary_subtraction :
  binary_to_decimal binary_1111111111 - binary_to_decimal binary_11111 = 992 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_l55_5581


namespace NUMINAMATH_CALUDE_total_weight_of_mixtures_l55_5521

/-- Represents a mixture of vegetable ghee -/
structure Mixture where
  ratio_a : ℚ
  ratio_b : ℚ
  total_volume : ℚ

/-- Calculates the weight of a mixture in kg -/
def mixture_weight (m : Mixture) (weight_a weight_b : ℚ) : ℚ :=
  let total_ratio := m.ratio_a + m.ratio_b
  let volume_a := (m.ratio_a / total_ratio) * m.total_volume
  let volume_b := (m.ratio_b / total_ratio) * m.total_volume
  (volume_a * weight_a + volume_b * weight_b) / 1000

def mixture1 : Mixture := ⟨3, 2, 6⟩
def mixture2 : Mixture := ⟨5, 3, 4⟩
def mixture3 : Mixture := ⟨9, 4, 6.5⟩

def weight_a : ℚ := 900
def weight_b : ℚ := 750

theorem total_weight_of_mixtures :
  mixture_weight mixture1 weight_a weight_b +
  mixture_weight mixture2 weight_a weight_b +
  mixture_weight mixture3 weight_a weight_b = 13.965 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_mixtures_l55_5521


namespace NUMINAMATH_CALUDE_train_length_l55_5586

/-- Given a train traveling at 90 kmph and crossing a pole in 4 seconds, its length is 100 meters. -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed_kmph = 90 → 
  crossing_time = 4 → 
  train_length = speed_kmph * (1000 / 3600) * crossing_time →
  train_length = 100 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l55_5586


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l55_5527

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 8 + a 15 = Real.pi →
  Real.cos (a 4 + a 12) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l55_5527


namespace NUMINAMATH_CALUDE_simplify_power_l55_5536

theorem simplify_power (y : ℝ) : (3 * y^4)^4 = 81 * y^16 := by sorry

end NUMINAMATH_CALUDE_simplify_power_l55_5536


namespace NUMINAMATH_CALUDE_flower_bed_area_l55_5532

theorem flower_bed_area (total_posts : ℕ) (post_spacing : ℝ) 
  (h1 : total_posts = 24)
  (h2 : post_spacing = 5)
  (h3 : ∃ (short_side long_side : ℕ), 
    short_side + 1 + long_side + 1 = total_posts ∧ 
    long_side + 1 = 3 * (short_side + 1)) :
  (short_side * post_spacing) * (long_side * post_spacing) = 600 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_l55_5532


namespace NUMINAMATH_CALUDE_dan_baseball_cards_l55_5531

theorem dan_baseball_cards (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 97) 
  (h2 : remaining_cards = 82) : 
  initial_cards - remaining_cards = 15 := by
  sorry

end NUMINAMATH_CALUDE_dan_baseball_cards_l55_5531


namespace NUMINAMATH_CALUDE_rectangle_area_is_48_l55_5514

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle PQRS with points U and V on its diagonal -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point
  U : Point
  V : Point

/-- Given conditions for the rectangle problem -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  -- PQRS is a rectangle (implied by other conditions)
  -- PQ is parallel to RS (implied by rectangle property)
  (rect.P.x - rect.Q.x = rect.R.x - rect.S.x) ∧ 
  (rect.P.y - rect.Q.y = rect.R.y - rect.S.y) ∧ 
  -- PQ = RS
  ((rect.P.x - rect.Q.x)^2 + (rect.P.y - rect.Q.y)^2 = 
   (rect.R.x - rect.S.x)^2 + (rect.R.y - rect.S.y)^2) ∧
  -- U and V lie on diagonal PS
  ((rect.U.x - rect.P.x) * (rect.S.y - rect.P.y) = 
   (rect.U.y - rect.P.y) * (rect.S.x - rect.P.x)) ∧
  ((rect.V.x - rect.P.x) * (rect.S.y - rect.P.y) = 
   (rect.V.y - rect.P.y) * (rect.S.x - rect.P.x)) ∧
  -- U is between P and V
  ((rect.U.x - rect.P.x) * (rect.V.x - rect.U.x) ≥ 0) ∧
  ((rect.U.y - rect.P.y) * (rect.V.y - rect.U.y) ≥ 0) ∧
  -- Angle PUV = 90°
  ((rect.P.x - rect.U.x) * (rect.V.x - rect.U.x) + 
   (rect.P.y - rect.U.y) * (rect.V.y - rect.U.y) = 0) ∧
  -- Angle QVR = 90°
  ((rect.Q.x - rect.V.x) * (rect.R.x - rect.V.x) + 
   (rect.Q.y - rect.V.y) * (rect.R.y - rect.V.y) = 0) ∧
  -- PU = 4
  ((rect.P.x - rect.U.x)^2 + (rect.P.y - rect.U.y)^2 = 16) ∧
  -- UV = 2
  ((rect.U.x - rect.V.x)^2 + (rect.U.y - rect.V.y)^2 = 4) ∧
  -- VS = 6
  ((rect.V.x - rect.S.x)^2 + (rect.V.y - rect.S.y)^2 = 36)

/-- The area of a rectangle -/
def rectangle_area (rect : Rectangle) : ℝ :=
  abs ((rect.P.x - rect.Q.x) * (rect.Q.y - rect.R.y) - 
       (rect.P.y - rect.Q.y) * (rect.Q.x - rect.R.x))

/-- Theorem stating that the area of the rectangle is 48 -/
theorem rectangle_area_is_48 (rect : Rectangle) : 
  rectangle_conditions rect → rectangle_area rect = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_48_l55_5514


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l55_5529

/-- Given a boat that travels 8 km along a stream and 2 km against the stream in one hour,
    prove that its speed in still water is 5 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 8 →
    boat_speed - stream_speed = 2 →
    boat_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l55_5529


namespace NUMINAMATH_CALUDE_sum_one_implies_not_both_greater_than_one_l55_5520

theorem sum_one_implies_not_both_greater_than_one (a b : ℝ) :
  a + b = 1 → ¬(a > 1 ∧ b > 1) := by
sorry

end NUMINAMATH_CALUDE_sum_one_implies_not_both_greater_than_one_l55_5520


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l55_5519

theorem sqrt_x_plus_one_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l55_5519


namespace NUMINAMATH_CALUDE_division_subtraction_problem_l55_5545

theorem division_subtraction_problem : (12 / (2/3)) - 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_problem_l55_5545


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l55_5540

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence based on the incircle tangent points -/
def nextTriangle (t : Triangle) : Triangle :=
  sorry

/-- Checks if a triangle is valid (satisfies the triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The sequence of triangles starting from the initial triangle -/
def triangleSequence : ℕ → Triangle
  | 0 => { a := 1015, b := 1016, c := 1017 }
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Finds the index of the last valid triangle in the sequence -/
def lastValidTriangleIndex : ℕ :=
  sorry

theorem last_triangle_perimeter :
  perimeter (triangleSequence lastValidTriangleIndex) = 762 / 128 :=
sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l55_5540


namespace NUMINAMATH_CALUDE_mike_dogs_count_l55_5523

/-- Represents the number of dogs Mike has -/
def number_of_dogs : ℕ := 2

/-- Weight of a cup of dog food in pounds -/
def cup_weight : ℚ := 1/4

/-- Number of cups each dog eats per feeding -/
def cups_per_feeding : ℕ := 6

/-- Number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- Number of bags of dog food Mike buys per month -/
def bags_per_month : ℕ := 9

/-- Weight of each bag of dog food in pounds -/
def bag_weight : ℕ := 20

/-- Number of days in a month -/
def days_per_month : ℕ := 30

theorem mike_dogs_count :
  number_of_dogs = 
    (bags_per_month * bag_weight) / 
    (cups_per_feeding * feedings_per_day * cup_weight * days_per_month) := by
  sorry

end NUMINAMATH_CALUDE_mike_dogs_count_l55_5523


namespace NUMINAMATH_CALUDE_divisibility_property_l55_5534

theorem divisibility_property (a b c d e n : ℤ) 
  (h_odd : Odd n)
  (h_sum_div : n ∣ (a + b + c + d + e))
  (h_sum_squares_div : n ∣ (a^2 + b^2 + c^2 + d^2 + e^2)) :
  n ∣ (a^5 + b^5 + c^5 + d^5 + e^5 - 5*a*b*c*d*e) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l55_5534


namespace NUMINAMATH_CALUDE_total_cost_of_pens_l55_5587

/-- The cost of a single pen in dollars -/
def cost_per_pen : ℚ := 2

/-- The number of pens -/
def number_of_pens : ℕ := 10

/-- The total cost of pens -/
def total_cost : ℚ := cost_per_pen * number_of_pens

/-- Theorem stating that the total cost of 10 pens is $20 -/
theorem total_cost_of_pens : total_cost = 20 := by sorry

end NUMINAMATH_CALUDE_total_cost_of_pens_l55_5587


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l55_5503

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 
    and asymptotes y = ±(√3/2)x is √7/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = Real.sqrt 3 / 2) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l55_5503


namespace NUMINAMATH_CALUDE_expression_equals_36_l55_5554

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l55_5554
