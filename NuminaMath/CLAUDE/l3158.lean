import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3158_315864

-- Define the universal set U
def U : Set ℤ := {x | 0 < x ∧ x < 5}

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {2, 3}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3158_315864


namespace NUMINAMATH_CALUDE_square_to_three_squares_l3158_315834

/-- A partition of a square is a list of polygons that cover the square without overlap -/
def Partition (a : ℝ) := List (List (ℝ × ℝ))

/-- A square is a list of four points representing its vertices -/
def Square := List (ℝ × ℝ)

/-- Predicate to check if a partition is valid (covers the whole square without overlap) -/
def is_valid_partition (a : ℝ) (p : Partition a) : Prop := sorry

/-- Predicate to check if a list of points forms a square -/
def is_square (s : Square) : Prop := sorry

/-- Predicate to check if a partition can be rearranged to form given squares -/
def can_form_squares (a : ℝ) (p : Partition a) (squares : List Square) : Prop := sorry

/-- Theorem stating that a square can be cut into 4 parts to form 3 squares -/
theorem square_to_three_squares (a : ℝ) : 
  ∃ (p : Partition a) (s₁ s₂ s₃ : Square), 
    is_valid_partition a p ∧ 
    p.length = 4 ∧
    is_square s₁ ∧ is_square s₂ ∧ is_square s₃ ∧
    can_form_squares a p [s₁, s₂, s₃] := by
  sorry

end NUMINAMATH_CALUDE_square_to_three_squares_l3158_315834


namespace NUMINAMATH_CALUDE_valid_distributions_count_l3158_315824

def number_of_valid_distributions : ℕ :=
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ 360) (Nat.divisors 360)).card

theorem valid_distributions_count : number_of_valid_distributions = 22 := by
  sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l3158_315824


namespace NUMINAMATH_CALUDE_mike_savings_rate_l3158_315875

theorem mike_savings_rate (carol_initial : ℕ) (carol_weekly : ℕ) (mike_initial : ℕ) (weeks : ℕ) :
  carol_initial = 60 →
  carol_weekly = 9 →
  mike_initial = 90 →
  weeks = 5 →
  ∃ (mike_weekly : ℕ),
    carol_initial + carol_weekly * weeks = mike_initial + mike_weekly * weeks ∧
    mike_weekly = 3 :=
by sorry

end NUMINAMATH_CALUDE_mike_savings_rate_l3158_315875


namespace NUMINAMATH_CALUDE_molecular_weight_15_C2H5Cl_12_O2_l3158_315844

/-- Calculates the molecular weight of a given number of moles of C2H5Cl and O2 -/
def molecularWeight (moles_C2H5Cl : ℝ) (moles_O2 : ℝ) : ℝ :=
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.01
  let atomic_weight_Cl := 35.45
  let atomic_weight_O := 16.00
  let mw_C2H5Cl := 2 * atomic_weight_C + 5 * atomic_weight_H + atomic_weight_Cl
  let mw_O2 := 2 * atomic_weight_O
  moles_C2H5Cl * mw_C2H5Cl + moles_O2 * mw_O2

theorem molecular_weight_15_C2H5Cl_12_O2 :
  molecularWeight 15 12 = 1351.8 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_15_C2H5Cl_12_O2_l3158_315844


namespace NUMINAMATH_CALUDE_candy_distribution_l3158_315815

/-- Represents the number of candies eaten by each person -/
structure CandyCount where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Calculates the total number of candies eaten by all three people -/
def total_candies (count : CandyCount) : ℕ :=
  count.andrey + count.boris + count.denis

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_to_boris : ℚ
  andrey_to_denis : ℚ

/-- Theorem stating the correct number of candies eaten by each person -/
theorem candy_distribution (rates : EatingRates) : 
  ∃ (count : CandyCount), 
    rates.andrey_to_boris = 4 / 3 ∧ 
    rates.andrey_to_denis = 6 / 7 ∧
    total_candies count = 70 ∧
    count.andrey = 24 ∧
    count.boris = 18 ∧
    count.denis = 28 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3158_315815


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l3158_315809

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 12*x + y^2 - 6*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 + 8*y + 34 = 0}
  (shortest_distance : ℝ) →
  shortest_distance = Real.sqrt 170 - 3 - Real.sqrt 7 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ),
    p1 ∈ circle1 → p2 ∈ circle2 →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_between_circles_l3158_315809


namespace NUMINAMATH_CALUDE_ab_ratio_for_inscribed_triangle_l3158_315810

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - p.y)^2 / e.a^2 + (p.x + p.y)^2 / e.b^2 = 1

/-- Checks if two points form a line parallel to y = x -/
def isParallelToYEqualX (p1 p2 : Point) : Prop :=
  p1.x - p1.y = p2.x - p2.y

/-- Theorem: AB/b ratio for an equilateral triangle inscribed in a specific ellipse -/
theorem ab_ratio_for_inscribed_triangle
  (e : Ellipse)
  (t : EquilateralTriangle)
  (h1 : t.A = ⟨0, e.b⟩)
  (h2 : isOnEllipse t.A e ∧ isOnEllipse t.B e ∧ isOnEllipse t.C e)
  (h3 : isParallelToYEqualX t.B t.C)
  (h4 : e.a = e.b * Real.sqrt 2)  -- Condition for focus at vertex C
  : Real.sqrt ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2) / e.b = 8/5 :=
sorry

end NUMINAMATH_CALUDE_ab_ratio_for_inscribed_triangle_l3158_315810


namespace NUMINAMATH_CALUDE_negation_of_all_men_honest_l3158_315849

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a man and being honest
variable (man : U → Prop)
variable (honest : U → Prop)

-- State the theorem
theorem negation_of_all_men_honest :
  (¬ ∀ x, man x → honest x) ↔ (∃ x, man x ∧ ¬ honest x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_men_honest_l3158_315849


namespace NUMINAMATH_CALUDE_third_islander_statement_l3158_315830

-- Define the types of islanders
inductive IslanderType
| Knight
| Liar

-- Define the islanders
def A : IslanderType := IslanderType.Liar
def B : IslanderType := IslanderType.Knight
def C : IslanderType := IslanderType.Knight

-- Define the statements made by the islanders
def statement_A : Prop := ∀ x, x ≠ A → IslanderType.Liar = x
def statement_B : Prop := ∃! x, x ≠ B ∧ IslanderType.Knight = x

-- Theorem to prove
theorem third_islander_statement :
  (A = IslanderType.Liar) →
  (B = IslanderType.Knight) →
  (C = IslanderType.Knight) →
  statement_A →
  statement_B →
  (∃! x, x ≠ C ∧ IslanderType.Knight = x) :=
by sorry

end NUMINAMATH_CALUDE_third_islander_statement_l3158_315830


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l3158_315804

/-- Given that M(5,3) is the midpoint of AB and A has coordinates (10,2), 
    prove that the sum of coordinates of point B is 4. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (5, 3) → 
  A = (10, 2) → 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l3158_315804


namespace NUMINAMATH_CALUDE_problem_solution_l3158_315876

theorem problem_solution (a : ℝ) : 
  (∀ b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  a * 15 * 11 = 1 →
  a = 6 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3158_315876


namespace NUMINAMATH_CALUDE_statement_equivalence_l3158_315874

theorem statement_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3158_315874


namespace NUMINAMATH_CALUDE_complement_M_in_U_l3158_315889

def U : Set ℕ := {x | x < 5 ∧ x > 0}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}

theorem complement_M_in_U : (U \ M) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l3158_315889


namespace NUMINAMATH_CALUDE_sacred_words_count_l3158_315894

-- Define the number of letters in the alien script
variable (n : ℕ)

-- Define the length of sacred words
variable (k : ℕ)

-- Condition that k is less than half of n
variable (h : k < n / 2)

-- Define a function to calculate the number of sacred k-words
def num_sacred_words (n k : ℕ) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * Nat.factorial k / k

-- Theorem statement
theorem sacred_words_count (n k : ℕ) (h : k < n / 2) :
  num_sacred_words n k = n * Nat.choose (n - k - 1) (k - 1) * Nat.factorial k / k :=
by sorry

-- Example for n = 10 and k = 4
example : num_sacred_words 10 4 = 600 :=
by sorry

end NUMINAMATH_CALUDE_sacred_words_count_l3158_315894


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l3158_315836

theorem number_of_elements_in_set (initial_average : ℝ) (incorrect_number : ℝ) (correct_number : ℝ) (correct_average : ℝ) :
  initial_average = 16 ∧ 
  incorrect_number = 26 ∧ 
  correct_number = 46 ∧ 
  correct_average = 18 →
  ∃ n : ℕ, n = 10 ∧ 
    n * initial_average = (n - 1) * initial_average + incorrect_number ∧
    n * correct_average = (n - 1) * initial_average + correct_number :=
by sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l3158_315836


namespace NUMINAMATH_CALUDE_perp_bisector_x_intercept_range_l3158_315851

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the perpendicular bisector intersection with x-axis
def perp_bisector_x_intercept (A B : PointOnParabola) : ℝ :=
  sorry -- Definition of x₀ in terms of A and B

-- Theorem statement
theorem perp_bisector_x_intercept_range (A B : PointOnParabola) :
  A ≠ B → perp_bisector_x_intercept A B > 1 :=
sorry

end NUMINAMATH_CALUDE_perp_bisector_x_intercept_range_l3158_315851


namespace NUMINAMATH_CALUDE_third_term_of_sequence_l3158_315870

/-- Given a sequence {a_n} with S_n as the sum of the first n terms, and S_n = n^2 + n, prove a_3 = 6 -/
theorem third_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^2 + n) : a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_sequence_l3158_315870


namespace NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_leq_two_l3158_315827

theorem quadratic_inequality_iff_abs_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ abs a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_leq_two_l3158_315827


namespace NUMINAMATH_CALUDE_first_grade_muffins_l3158_315871

/-- The number of muffins baked by Mrs. Brier's class -/
def muffins_brier : ℕ := 18

/-- The number of muffins baked by Mrs. MacAdams's class -/
def muffins_macadams : ℕ := 20

/-- The number of muffins baked by Mrs. Flannery's class -/
def muffins_flannery : ℕ := 17

/-- The total number of muffins baked by first grade -/
def total_muffins : ℕ := muffins_brier + muffins_macadams + muffins_flannery

theorem first_grade_muffins : total_muffins = 55 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_muffins_l3158_315871


namespace NUMINAMATH_CALUDE_sum_of_80th_equation_l3158_315825

/-- The sum of the nth equation in the series -/
def seriesSum (n : ℕ) : ℕ := (2 * n + 1) + (5 * n - 1)

/-- The sum of the 80th equation in the series is 560 -/
theorem sum_of_80th_equation : seriesSum 80 = 560 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_80th_equation_l3158_315825


namespace NUMINAMATH_CALUDE_bens_old_car_cost_l3158_315813

/-- The cost of Ben's old car in dollars -/
def old_car_cost : ℝ := 1900

/-- The cost of Ben's new car in dollars -/
def new_car_cost : ℝ := 3800

/-- The amount Ben received from selling his old car in dollars -/
def old_car_sale : ℝ := 1800

/-- The amount Ben still owes on his new car in dollars -/
def remaining_debt : ℝ := 2000

/-- Theorem stating that the cost of Ben's old car was $1900 -/
theorem bens_old_car_cost :
  old_car_cost = 1900 ∧
  new_car_cost = 2 * old_car_cost ∧
  new_car_cost = old_car_sale + remaining_debt :=
by sorry

end NUMINAMATH_CALUDE_bens_old_car_cost_l3158_315813


namespace NUMINAMATH_CALUDE_tan_beta_value_l3158_315872

theorem tan_beta_value (α β : Real) 
  (h1 : (Real.sin α * Real.cos α) / (Real.cos (2 * α) + 1) = 1)
  (h2 : Real.tan (α - β) = 3) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3158_315872


namespace NUMINAMATH_CALUDE_velociraptor_catch_up_time_l3158_315837

/-- The time it takes for a velociraptor to catch up to a person, given their respective speeds and initial head start. -/
theorem velociraptor_catch_up_time 
  (your_speed : ℝ)
  (velociraptor_speed : ℝ)
  (head_start_time : ℝ)
  (h1 : your_speed = 10)
  (h2 : velociraptor_speed = 15 * Real.sqrt 2)
  (h3 : head_start_time = 3) :
  (head_start_time * your_speed) / (velociraptor_speed / Real.sqrt 2 - your_speed) = 6 := by
  sorry

#check velociraptor_catch_up_time

end NUMINAMATH_CALUDE_velociraptor_catch_up_time_l3158_315837


namespace NUMINAMATH_CALUDE_jellybean_count_l3158_315832

/-- The number of blue jellybeans in a jar -/
def blue_jellybeans (total purple orange red : ℕ) : ℕ :=
  total - (purple + orange + red)

/-- Theorem: In a jar with 200 total jellybeans, 26 purple, 40 orange, and 120 red jellybeans,
    there are 14 blue jellybeans. -/
theorem jellybean_count : blue_jellybeans 200 26 40 120 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l3158_315832


namespace NUMINAMATH_CALUDE_cafeteria_apple_count_l3158_315883

/-- Calculates the final number of apples in the cafeteria after a series of operations -/
def final_apple_count (initial : ℕ) (monday_used monday_bought tuesday_used tuesday_bought wednesday_used : ℕ) : ℕ :=
  initial - monday_used + monday_bought - tuesday_used + tuesday_bought - wednesday_used

/-- Theorem stating that given the initial number of apples and daily changes, the final number of apples is 46 -/
theorem cafeteria_apple_count : 
  final_apple_count 17 2 23 4 15 3 = 46 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apple_count_l3158_315883


namespace NUMINAMATH_CALUDE_problem_solution_l3158_315895

/-- The function f(x) = x^2 - 2ax + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem problem_solution (a : ℝ) (h : a > 1) :
  /- Part 1 -/
  (∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) →
  a = 2
  ∧
  /- Part 2 -/
  ((∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) ∧
   (∀ x ∈ Set.Icc 1 2, f a x ≤ 0)) →
  a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3158_315895


namespace NUMINAMATH_CALUDE_xiao_ming_final_score_l3158_315862

/-- Calculate the final score given individual scores and weights -/
def final_score (content_score language_score demeanor_score : ℝ)
  (content_weight language_weight demeanor_weight : ℝ) : ℝ :=
  content_score * content_weight +
  language_score * language_weight +
  demeanor_score * demeanor_weight

/-- Theorem stating that Xiao Ming's final score is 86.2 -/
theorem xiao_ming_final_score :
  final_score 85 90 82 0.6 0.3 0.1 = 86.2 := by
  sorry

#eval final_score 85 90 82 0.6 0.3 0.1

end NUMINAMATH_CALUDE_xiao_ming_final_score_l3158_315862


namespace NUMINAMATH_CALUDE_student_count_correct_l3158_315890

/-- Represents the changes in student numbers for a grade --/
structure GradeChanges where
  initial : Nat
  left : Nat
  joined : Nat
  transferredIn : Nat
  transferredOut : Nat

/-- Calculates the final number of students in a grade --/
def finalStudents (changes : GradeChanges) : Nat :=
  changes.initial - changes.left + changes.joined + changes.transferredIn - changes.transferredOut

/-- Theorem: The calculated final numbers of students in each grade and their total are correct --/
theorem student_count_correct (fourth : GradeChanges) (fifth : GradeChanges) (sixth : GradeChanges) 
    (h4 : fourth = ⟨4, 3, 42, 0, 10⟩)
    (h5 : fifth = ⟨10, 5, 25, 10, 5⟩)
    (h6 : sixth = ⟨15, 7, 30, 5, 0⟩) : 
    finalStudents fourth = 33 ∧ 
    finalStudents fifth = 35 ∧ 
    finalStudents sixth = 43 ∧
    finalStudents fourth + finalStudents fifth + finalStudents sixth = 111 := by
  sorry

end NUMINAMATH_CALUDE_student_count_correct_l3158_315890


namespace NUMINAMATH_CALUDE_omelet_time_is_100_l3158_315899

/-- Time to prepare and cook omelets -/
def total_omelet_time (
  pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (mushroom_slice_time : ℕ)
  (tomato_dice_time : ℕ)
  (cheese_grate_time : ℕ)
  (vegetable_saute_time : ℕ)
  (egg_cheese_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ)
  (num_mushrooms : ℕ)
  (num_tomatoes : ℕ)
  (num_omelets : ℕ) : ℕ :=
  let prep_time := 
    pepper_chop_time * num_peppers +
    onion_chop_time * num_onions +
    mushroom_slice_time * num_mushrooms +
    tomato_dice_time * num_tomatoes +
    cheese_grate_time * num_omelets
  let cook_time := vegetable_saute_time + egg_cheese_cook_time
  let omelets_during_prep := prep_time / cook_time
  let remaining_omelets := num_omelets - omelets_during_prep
  prep_time + remaining_omelets * cook_time

/-- Theorem: The total time to prepare and cook 10 omelets is 100 minutes -/
theorem omelet_time_is_100 :
  total_omelet_time 3 4 2 3 1 4 6 8 4 6 6 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_omelet_time_is_100_l3158_315899


namespace NUMINAMATH_CALUDE_inscribed_polygon_sides_l3158_315814

theorem inscribed_polygon_sides (n : ℕ) (s : ℝ) : n ≥ 3 →
  s = 2 * Real.sin (Real.pi / n) →  -- side length formula
  1 < s →
  s < Real.sqrt 2 →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_polygon_sides_l3158_315814


namespace NUMINAMATH_CALUDE_inequality_implies_a_squared_gt_3b_l3158_315806

theorem inequality_implies_a_squared_gt_3b (a b c : ℝ) 
  (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : 
  a^2 > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_squared_gt_3b_l3158_315806


namespace NUMINAMATH_CALUDE_hotel_cost_calculation_l3158_315841

theorem hotel_cost_calculation 
  (cost_per_night_per_person : ℕ) 
  (number_of_people : ℕ) 
  (number_of_nights : ℕ) 
  (h1 : cost_per_night_per_person = 40)
  (h2 : number_of_people = 3)
  (h3 : number_of_nights = 3) :
  cost_per_night_per_person * number_of_people * number_of_nights = 360 :=
by sorry

end NUMINAMATH_CALUDE_hotel_cost_calculation_l3158_315841


namespace NUMINAMATH_CALUDE_f_properties_l3158_315800

noncomputable def f (x : ℝ) := Real.log ((1 + x) / (1 - x))

theorem f_properties :
  ∃ (k : ℝ),
    (∀ x ∈ Set.Ioo 0 1, f x > k * (x + x^3 / 3)) ∧
    (∀ k' > k, ∃ x ∈ Set.Ioo 0 1, f x ≤ k' * (x + x^3 / 3)) ∧
    k = 2 ∧
    (∀ x ∈ Set.Ioo 0 1, f x > 2 * (x + x^3 / 3)) ∧
    (∀ h ∈ Set.Ioo 0 1, (f h - f 0) / h = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3158_315800


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3158_315866

/-- Given that p and r · q are inversely proportional, prove that p = 128/15 when q = 10 and r = 3,
    given that p = 16 when q = 8 and r = 2 -/
theorem inverse_proportion_problem (p q r : ℝ) (h1 : ∃ k, p * (r * q) = k) 
  (h2 : p = 16 ∧ q = 8 ∧ r = 2) : 
  (q = 10 ∧ r = 3) → p = 128 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3158_315866


namespace NUMINAMATH_CALUDE_triangle_side_length_l3158_315863

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = 3 ∧ A = π/6 ∧ B = π/12 →
  c = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3158_315863


namespace NUMINAMATH_CALUDE_hyperbola_single_intersection_lines_l3158_315887

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : a > 0)
  (positive_b : b > 0)

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  c : ℝ

/-- Function to check if a line intersects a hyperbola at only one point -/
def intersects_at_one_point (h : Hyperbola) (l : Line) : Prop :=
  ∃! p : Point, p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.y = l.m * p.x + l.c

/-- Theorem statement -/
theorem hyperbola_single_intersection_lines 
  (h : Hyperbola) 
  (p : Point) 
  (h_eq : h.a = 1 ∧ h.b = 2) 
  (p_eq : p.x = 1 ∧ p.y = 1) :
  ∃! (lines : Finset Line), 
    lines.card = 4 ∧ 
    ∀ l ∈ lines, intersects_at_one_point h l ∧ p.y = l.m * p.x + l.c :=
sorry

end NUMINAMATH_CALUDE_hyperbola_single_intersection_lines_l3158_315887


namespace NUMINAMATH_CALUDE_probability_r25_to_r35_correct_l3158_315877

def bubble_pass (s : List ℝ) : List ℝ := sorry

def probability_r25_to_r35 (n : ℕ) : ℚ :=
  if n ≥ 50 then 1 / 1260 else 0

theorem probability_r25_to_r35_correct (s : List ℝ) (h : s.length = 50) 
  (h_distinct : s.Nodup) : 
  probability_r25_to_r35 s.length = 1 / 1260 := by sorry

end NUMINAMATH_CALUDE_probability_r25_to_r35_correct_l3158_315877


namespace NUMINAMATH_CALUDE_prep_time_score_relation_student_score_for_six_hours_l3158_315808

/-- Represents the direct variation between score and preparation time -/
def score_variation (prep_time : ℝ) (score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ score = k * prep_time

/-- Theorem stating the relationship between preparation time and test score -/
theorem prep_time_score_relation (initial_prep_time initial_score new_prep_time : ℝ) :
  initial_prep_time > 0 →
  initial_score > 0 →
  new_prep_time > 0 →
  score_variation initial_prep_time initial_score →
  score_variation new_prep_time (new_prep_time * initial_score / initial_prep_time) :=
by sorry

/-- Main theorem proving the specific case from the problem -/
theorem student_score_for_six_hours :
  let initial_prep_time : ℝ := 4
  let initial_score : ℝ := 80
  let new_prep_time : ℝ := 6
  score_variation initial_prep_time initial_score →
  score_variation new_prep_time 120 :=
by sorry

end NUMINAMATH_CALUDE_prep_time_score_relation_student_score_for_six_hours_l3158_315808


namespace NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l3158_315812

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 6 * 6

/-- The number of ways to get a sum of 6 when rolling two dice -/
def favorable_outcomes : ℕ := 5

/-- The probability of getting a sum of 6 when rolling two fair dice -/
def prob_sum_six : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_six_is_five_thirty_sixths :
  prob_sum_six = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l3158_315812


namespace NUMINAMATH_CALUDE_small_triangles_to_large_triangle_area_ratio_l3158_315818

theorem small_triangles_to_large_triangle_area_ratio :
  let small_side_length : ℝ := 2
  let small_triangle_count : ℕ := 8
  let small_triangle_perimeter : ℝ := 3 * small_side_length
  let large_triangle_perimeter : ℝ := small_triangle_count * small_triangle_perimeter
  let large_side_length : ℝ := large_triangle_perimeter / 3
  let triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2
  let small_triangle_area : ℝ := triangle_area small_side_length
  let large_triangle_area : ℝ := triangle_area large_side_length
  (small_triangle_count * small_triangle_area) / large_triangle_area = 1 / 8 := by
  sorry

#check small_triangles_to_large_triangle_area_ratio

end NUMINAMATH_CALUDE_small_triangles_to_large_triangle_area_ratio_l3158_315818


namespace NUMINAMATH_CALUDE_boys_in_row_l3158_315891

/-- Represents the number of boys in a row with given conditions -/
def number_of_boys (left_position right_position between : ℕ) : ℕ :=
  left_position + between + right_position

/-- Theorem stating that under the given conditions, the number of boys in the row is 24 -/
theorem boys_in_row :
  let rajan_position := 6
  let vinay_position := 10
  let boys_between := 8
  number_of_boys rajan_position vinay_position boys_between = 24 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_row_l3158_315891


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3158_315826

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -2) (h2 : b = 1/3) :
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3158_315826


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l3158_315823

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Predicate for a number being in Pascal's triangle -/
def inPascalTriangle (x : ℕ) : Prop := ∃ n k, pascal n k = x

/-- The set of four-digit numbers in Pascal's triangle -/
def fourDigitPascalNumbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999 ∧ inPascalTriangle x}

/-- The third smallest element in a set of natural numbers -/
noncomputable def thirdSmallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal :
  thirdSmallest fourDigitPascalNumbers = 1002 := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l3158_315823


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3158_315861

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → perpendicular_line x y → x * 2 + y * 1 = -1) ∧
  (perpendicular_line point_P.1 point_P.2) ∧
  (∀ x y, given_line x y → ∀ a b, perpendicular_line a b → 
    (y - point_P.2) * (x - point_P.1) = -(b - point_P.2) * (a - point_P.1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3158_315861


namespace NUMINAMATH_CALUDE_expression_evaluation_l3158_315867

theorem expression_evaluation (b : ℝ) (h : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3158_315867


namespace NUMINAMATH_CALUDE_determinant_zero_exists_l3158_315882

def matrix (x a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + a, x + b, x + c],
    ![x + b, x + c, x + a],
    ![x + c, x + a, x + b]]

theorem determinant_zero_exists (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) : 
  ∃ x : ℝ, Matrix.det (matrix x a b c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_exists_l3158_315882


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l3158_315873

theorem arithmetic_mean_of_specific_numbers :
  let numbers : List ℝ := [-5, 3.5, 12, 20]
  (numbers.sum / numbers.length : ℝ) = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l3158_315873


namespace NUMINAMATH_CALUDE_x_convergence_bound_l3158_315829

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem x_convergence_bound : 
  ∃ m : ℕ, 243 ≤ m ∧ m ≤ 728 ∧ 
    x m ≤ 6 + 1 / 2^18 ∧ 
    ∀ k < m, x k > 6 + 1 / 2^18 :=
sorry

end NUMINAMATH_CALUDE_x_convergence_bound_l3158_315829


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l3158_315853

theorem existence_of_special_integer :
  ∃ (n : ℕ), n ≥ 2^2018 ∧
  ∀ (x y u v : ℕ), u > 1 → v > 1 → n ≠ x^u + y^v :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l3158_315853


namespace NUMINAMATH_CALUDE_inscribed_circle_probability_l3158_315845

theorem inscribed_circle_probability (a b : ℝ) (h_right_triangle : a = 8 ∧ b = 15) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  1 - (π * r^2) / (a * b / 2) = 1 - 3 * π / 20 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_probability_l3158_315845


namespace NUMINAMATH_CALUDE_total_money_calculation_l3158_315833

theorem total_money_calculation (p q r total : ℝ) 
  (h1 : r = (2/3) * total) 
  (h2 : r = 3600) : 
  total = 5400 := by
sorry

end NUMINAMATH_CALUDE_total_money_calculation_l3158_315833


namespace NUMINAMATH_CALUDE_triangle_inequality_l3158_315855

/-- For any triangle with sides a, b, c, angle A opposite side a, and semiperimeter p,
    the inequality (bc cos A) / (b + c) + a < p < (bc + a^2) / a holds. -/
theorem triangle_inequality (a b c : ℝ) (A : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : 0 < A ∧ A < π) :
  let p := (a + b + c) / 2
  (b * c * Real.cos A) / (b + c) + a < p ∧ p < (b * c + a^2) / a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3158_315855


namespace NUMINAMATH_CALUDE_a_n_property_smallest_n_for_perfect_square_sum_l3158_315885

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

def is_sum_or_diff_of_squares (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a * a + b * b ∨ x = a * a - b * b ∨ x = b * b - a * a

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

def a_n (n : ℕ) : ℕ := 10^n - 2

def sum_of_squares_of_digits (x : ℕ) : ℕ :=
  (x.digits 10).map (λ d => d * d) |>.sum

theorem a_n_property (n : ℕ) (h : n > 2) :
  ¬(is_sum_or_diff_of_squares (a_n n)) ∧
  ∀ m : ℕ, m > a_n n → m ≤ largest_n_digit_number n → is_sum_or_diff_of_squares m :=
sorry

theorem smallest_n_for_perfect_square_sum :
  ∀ n : ℕ, n < 66 → ¬(is_perfect_square (sum_of_squares_of_digits (a_n n))) ∧
  is_perfect_square (sum_of_squares_of_digits (a_n 66)) :=
sorry

end NUMINAMATH_CALUDE_a_n_property_smallest_n_for_perfect_square_sum_l3158_315885


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3158_315856

def total_crayons : ℕ := 15
def karls_selection : ℕ := 3
def friends_selection : ℕ := 4

def selection_ways : ℕ := Nat.choose total_crayons karls_selection * 
                           Nat.choose (total_crayons - karls_selection) friends_selection

theorem crayon_selection_theorem : 
  selection_ways = 225225 := by sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3158_315856


namespace NUMINAMATH_CALUDE_journey_time_l3158_315811

/-- Represents a journey with constant speed -/
structure Journey where
  quarter_time : ℝ  -- Time to cover 1/4 of the journey
  third_time : ℝ    -- Time to cover 1/3 of the journey

/-- The total time for the journey -/
def total_time (j : Journey) : ℝ :=
  (j.third_time - j.quarter_time) * 12 - j.quarter_time

/-- Theorem stating that for the given journey, the total time is 280 minutes -/
theorem journey_time (j : Journey) 
  (h1 : j.quarter_time = 20) 
  (h2 : j.third_time = 45) : 
  total_time j = 280 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_l3158_315811


namespace NUMINAMATH_CALUDE_rental_cost_equality_l3158_315852

/-- The daily rate for Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate for Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

/-- The mileage at which the cost is the same for both companies -/
def equal_cost_mileage : ℝ := 150

theorem rental_cost_equality :
  safety_daily_rate + safety_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage := by
  sorry

#check rental_cost_equality

end NUMINAMATH_CALUDE_rental_cost_equality_l3158_315852


namespace NUMINAMATH_CALUDE_max_crates_third_trip_l3158_315840

/-- Given a trailer with a maximum weight capacity and a minimum weight per crate,
    prove the maximum number of crates for the third trip. -/
theorem max_crates_third_trip
  (max_weight : ℕ)
  (min_crate_weight : ℕ)
  (trip1_crates : ℕ)
  (trip2_crates : ℕ)
  (h_max_weight : max_weight = 750)
  (h_min_crate_weight : min_crate_weight = 150)
  (h_trip1 : trip1_crates = 3)
  (h_trip2 : trip2_crates = 4)
  (h_weight_constraint : ∀ n : ℕ, n * min_crate_weight ≤ max_weight → n ≤ trip1_crates ∨ n ≤ trip2_crates ∨ n ≤ max_weight / min_crate_weight) :
  (max_weight / min_crate_weight : ℕ) = 5 :=
sorry

end NUMINAMATH_CALUDE_max_crates_third_trip_l3158_315840


namespace NUMINAMATH_CALUDE_rental_cost_equation_l3158_315831

/-- The monthly cost of renting a car. -/
def R : ℝ := sorry

/-- The monthly cost of the new car. -/
def new_car_cost : ℝ := 30

/-- The number of months in a year. -/
def months_in_year : ℕ := 12

/-- The difference in total cost over a year. -/
def cost_difference : ℝ := 120

/-- Theorem stating the relationship between rental cost and new car cost. -/
theorem rental_cost_equation : 
  months_in_year * R - months_in_year * new_car_cost = cost_difference := by
  sorry

end NUMINAMATH_CALUDE_rental_cost_equation_l3158_315831


namespace NUMINAMATH_CALUDE_granger_age_is_42_l3158_315821

/-- Mr. Granger's current age -/
def granger_age : ℕ := sorry

/-- Mr. Granger's son's current age -/
def son_age : ℕ := sorry

/-- First condition: Mr. Granger's age is 10 years more than twice his son's age -/
axiom condition1 : granger_age = 2 * son_age + 10

/-- Second condition: Last year, Mr. Granger's age was 4 years less than 3 times his son's age -/
axiom condition2 : granger_age - 1 = 3 * (son_age - 1) - 4

/-- Theorem: Mr. Granger's age is 42 years -/
theorem granger_age_is_42 : granger_age = 42 := by sorry

end NUMINAMATH_CALUDE_granger_age_is_42_l3158_315821


namespace NUMINAMATH_CALUDE_neg_eight_celsius_meaning_l3158_315865

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℤ
  unit : String
  deriving Repr

/-- Converts a temperature to its representation relative to zero -/
def tempRelativeToZero (t : Temperature) : String :=
  if t.value > 0 then
    s!"{t.value}°C above zero"
  else if t.value < 0 then
    s!"{-t.value}°C below zero"
  else
    "0°C"

/-- The convention for representing temperatures -/
axiom temp_convention (t : Temperature) : 
  t.value > 0 → tempRelativeToZero t = s!"{t.value}°C above zero"

/-- Theorem: -8°C represents 8°C below zero -/
theorem neg_eight_celsius_meaning :
  let t : Temperature := ⟨-8, "C"⟩
  tempRelativeToZero t = "8°C below zero" := by
  sorry

end NUMINAMATH_CALUDE_neg_eight_celsius_meaning_l3158_315865


namespace NUMINAMATH_CALUDE_j20_most_suitable_for_census_l3158_315896

/-- Represents a survey option -/
inductive SurveyOption
  | HuaweiPhoneBattery
  | J20Components
  | SpringFestivalMovie
  | HomeworkTime

/-- Determines if a survey option is suitable for a comprehensive survey (census) -/
def isSuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.HuaweiPhoneBattery => False
  | SurveyOption.J20Components => True
  | SurveyOption.SpringFestivalMovie => False
  | SurveyOption.HomeworkTime => False

/-- Theorem stating that the J20Components survey is the most suitable for a comprehensive survey -/
theorem j20_most_suitable_for_census :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.J20Components →
    ¬(isSuitableForCensus option) ∧ isSuitableForCensus SurveyOption.J20Components :=
by sorry

end NUMINAMATH_CALUDE_j20_most_suitable_for_census_l3158_315896


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l3158_315828

theorem boat_stream_speed_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed > stream_speed) 
  (h2 : stream_speed > 0) 
  (h3 : distance > 0) 
  (h4 : distance / (boat_speed - stream_speed) = 2 * (distance / (boat_speed + stream_speed))) :
  boat_speed / stream_speed = 3 := by
sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l3158_315828


namespace NUMINAMATH_CALUDE_function_extrema_l3158_315838

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - 1/x + 1/x^2

theorem function_extrema (a : ℝ) :
  a ≠ 0 →
  (∃ (xmax xmin : ℝ), xmax > 0 ∧ xmin > 0 ∧
    (∀ x > 0, f a x ≤ f a xmax) ∧
    (∀ x > 0, f a x ≥ f a xmin)) ↔
  -1/8 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_l3158_315838


namespace NUMINAMATH_CALUDE_square_to_eight_acute_triangles_l3158_315835

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define an acute-angled triangle
def IsAcuteAngled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2 ∧
  t.b^2 + t.c^2 > t.a^2 ∧
  t.c^2 + t.a^2 > t.b^2

-- Theorem: A square can be divided into 8 acute-angled triangles
theorem square_to_eight_acute_triangles (s : Square) :
  ∃ (t₁ t₂ t₃ t₄ t₅ t₆ t₇ t₈ : Triangle),
    IsAcuteAngled t₁ ∧
    IsAcuteAngled t₂ ∧
    IsAcuteAngled t₃ ∧
    IsAcuteAngled t₄ ∧
    IsAcuteAngled t₅ ∧
    IsAcuteAngled t₆ ∧
    IsAcuteAngled t₇ ∧
    IsAcuteAngled t₈ :=
  sorry

end NUMINAMATH_CALUDE_square_to_eight_acute_triangles_l3158_315835


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3158_315879

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 - 5*t)
def line2 (s : ℝ) : ℝ × ℝ := (4 + 2*s, -6 - 5*s)

def direction : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance :
  let v := (3 - 4, -2 - (-6))
  let proj_v := ((v.1 * direction.1 + v.2 * direction.2) / (direction.1^2 + direction.2^2)) • direction
  let c := (4 + proj_v.1, -6 + proj_v.2)
  Real.sqrt ((3 - c.1)^2 + (-2 - c.2)^2) = 31 / 29 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3158_315879


namespace NUMINAMATH_CALUDE_dara_wait_time_l3158_315888

/-- Calculates the number of years Dara has to wait to reach the adjusted minimum age for employment. -/
def years_to_wait (current_min_age : ℕ) (jane_age : ℕ) (tom_age_diff : ℕ) (old_min_age : ℕ) : ℕ :=
  let dara_current_age := (jane_age + 6) / 2 - 6
  let years_passed := tom_age_diff + jane_age - old_min_age
  let periods_passed := years_passed / 5
  let new_min_age := current_min_age + periods_passed
  new_min_age - dara_current_age

/-- The number of years Dara has to wait is 16. -/
theorem dara_wait_time : years_to_wait 25 28 10 24 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dara_wait_time_l3158_315888


namespace NUMINAMATH_CALUDE_flour_already_added_l3158_315880

/-- Given a cake recipe and Mary's current progress, calculate how many cups of flour
    she has already put in. -/
theorem flour_already_added
  (total_required : ℕ)  -- Total cups of flour required by the recipe
  (more_needed : ℕ)     -- Cups of flour Mary still needs to add
  (h1 : total_required = 9)  -- The recipe requires 9 cups of flour
  (h2 : more_needed = 7)     -- Mary needs to add 7 more cups
  : total_required - more_needed = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_already_added_l3158_315880


namespace NUMINAMATH_CALUDE_sum_greatest_odd_divisors_l3158_315886

/-- The sum of the greatest odd divisors of natural numbers from 1 to 2^n -/
def S (n : ℕ) : ℕ :=
  (Finset.range (2^n + 1)).sum (λ m => Nat.gcd m ((2^n).div m))

/-- For any natural number n, 3 times the sum of the greatest odd divisors
    of natural numbers from 1 to 2^n equals 4^n + 2 -/
theorem sum_greatest_odd_divisors (n : ℕ) : 3 * S n = 4^n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_greatest_odd_divisors_l3158_315886


namespace NUMINAMATH_CALUDE_s_range_l3158_315859

-- Define the piecewise function
noncomputable def s (t : ℝ) : ℝ :=
  if t ≥ 1 then 3 * t else 4 * t - t^2

-- State the theorem
theorem s_range :
  Set.range s = Set.Icc (-5 : ℝ) 9 := by sorry

end NUMINAMATH_CALUDE_s_range_l3158_315859


namespace NUMINAMATH_CALUDE_simple_interest_example_l3158_315878

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the simple interest on $10000 at 9% per annum for 12 months is $900 -/
theorem simple_interest_example : simple_interest 10000 0.09 1 = 900 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_example_l3158_315878


namespace NUMINAMATH_CALUDE_parallel_vector_m_values_l3158_315839

def vector_a (m : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then 2 else m

theorem parallel_vector_m_values (m : ℝ) :
  (∃ b : Fin 2 → ℝ, b ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ vector_a m = λ i => k * b i) →
  m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_m_values_l3158_315839


namespace NUMINAMATH_CALUDE_numbers_sum_l3158_315897

/-- Given the conditions about Mickey's, Jayden's, and Coraline's numbers, 
    prove that their sum is 180. -/
theorem numbers_sum (M J C : ℕ) : 
  M = J + 20 →  -- Mickey's number is greater than Jayden's by 20
  J = C - 40 →  -- Jayden's number is 40 less than Coraline's
  C = 80 →      -- Coraline's number is 80
  M + J + C = 180 := by
sorry

end NUMINAMATH_CALUDE_numbers_sum_l3158_315897


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_l3158_315816

theorem sqrt_fifth_power : (Real.sqrt ((Real.sqrt 5)^4))^5 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_l3158_315816


namespace NUMINAMATH_CALUDE_cube_difference_l3158_315857

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3158_315857


namespace NUMINAMATH_CALUDE_train_speed_first_part_l3158_315869

/-- Represents a train journey with two parts -/
structure TrainJourney where
  x : ℝ  -- distance of the first part
  v : ℝ  -- speed of the first part
  total_distance : ℝ  -- total distance of the journey
  average_speed : ℝ  -- average speed of the entire journey

/-- The speed of the first part of the journey is 40 kmph given the conditions -/
theorem train_speed_first_part (j : TrainJourney) 
  (h1 : j.total_distance = 3 * j.x)  -- total distance is 3x
  (h2 : j.average_speed = 24)  -- average speed is 24 kmph
  (h3 : j.x > 0)  -- distance is positive
  : j.v = 40 := by
  sorry

#check train_speed_first_part

end NUMINAMATH_CALUDE_train_speed_first_part_l3158_315869


namespace NUMINAMATH_CALUDE_average_sale_calculation_l3158_315848

def sales : List ℕ := [6535, 6927, 6855, 7230, 6562]
def required_sale : ℕ := 4891
def num_months : ℕ := 6

theorem average_sale_calculation :
  (sales.sum + required_sale) / num_months = 6500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_calculation_l3158_315848


namespace NUMINAMATH_CALUDE_apples_given_theorem_l3158_315820

/-- Represents the number of apples Joan gave to Melanie -/
def apples_given_to_melanie (initial_apples current_apples : ℕ) : ℕ :=
  initial_apples - current_apples

theorem apples_given_theorem (initial_apples current_apples : ℕ) 
  (h1 : initial_apples = 43)
  (h2 : current_apples = 16) :
  apples_given_to_melanie initial_apples current_apples = 27 := by
  sorry

end NUMINAMATH_CALUDE_apples_given_theorem_l3158_315820


namespace NUMINAMATH_CALUDE_divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n_l3158_315802

theorem divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n (n : ℕ) :
  let m := ⌈(Real.sqrt 3 + 1)^(2*n)⌉
  ∃ k : ℕ, m = 2^(n+1) * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n_l3158_315802


namespace NUMINAMATH_CALUDE_natural_number_squares_l3158_315807

theorem natural_number_squares (x y : ℕ) : 
  1 + x + x^2 + x^3 + x^4 = y^2 ↔ (x = 0 ∧ y = 1) ∨ (x = 3 ∧ y = 11) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_squares_l3158_315807


namespace NUMINAMATH_CALUDE_deposit_percentage_l3158_315805

def deposit : ℝ := 120
def remaining : ℝ := 1080

theorem deposit_percentage :
  (deposit / (deposit + remaining)) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_deposit_percentage_l3158_315805


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3158_315822

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * x = 2022 * (x ^ (2021 / 2022)) - 1) ↔
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3158_315822


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3158_315842

theorem arctan_equation_solution :
  ∀ x : ℝ, 3 * Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 → x = -250/37 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3158_315842


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l3158_315847

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 11 = 36) (h3 : a 8 = 24) : a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l3158_315847


namespace NUMINAMATH_CALUDE_min_a_value_l3158_315803

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

def holds_inequality (a : ℤ) : Prop :=
  ∀ x > 0, f x ≤ ((↑a / 2) - 1) * x^2 + ↑a * x - 1

theorem min_a_value :
  ∃ a : ℤ, holds_inequality a ∧ ∀ b : ℤ, b < a → ¬(holds_inequality b) :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l3158_315803


namespace NUMINAMATH_CALUDE_smallest_block_volume_l3158_315854

theorem smallest_block_volume (a b c : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 252 → 
  a * b * c ≥ 392 ∧ 
  ∃ (a' b' c' : ℕ), (a' - 1) * (b' - 1) * (c' - 1) = 252 ∧ a' * b' * c' = 392 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l3158_315854


namespace NUMINAMATH_CALUDE_B_2_2_equals_12_l3158_315868

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2_equals_12 : B 2 2 = 12 := by sorry

end NUMINAMATH_CALUDE_B_2_2_equals_12_l3158_315868


namespace NUMINAMATH_CALUDE_equation_solution_l3158_315843

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 2 ∧ x ≠ -2 ∧ (x / (x - 2) + 2 / (x^2 - 4) = 1) ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3158_315843


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l3158_315884

theorem prime_square_sum_equation (p q : ℕ) : 
  (Prime p ∧ Prime q) → 
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

#check prime_square_sum_equation

end NUMINAMATH_CALUDE_prime_square_sum_equation_l3158_315884


namespace NUMINAMATH_CALUDE_slope_of_line_l3158_315881

/-- The slope of a line defined by the equation 4y = 5x + 20 is 5/4. -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x + 20 → (y - 5) / (x - (-5)) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3158_315881


namespace NUMINAMATH_CALUDE_overtake_time_l3158_315850

/-- The time it takes for person B to overtake person A given their speeds and start times -/
theorem overtake_time (speed_A speed_B : ℝ) (start_delay : ℝ) : 
  speed_A = 5 →
  speed_B = 5.555555555555555 →
  start_delay = 0.5 →
  speed_B > speed_A →
  (start_delay * speed_A) / (speed_B - speed_A) = 4.5 := by
  sorry

#check overtake_time

end NUMINAMATH_CALUDE_overtake_time_l3158_315850


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l3158_315858

theorem coloring_book_shelves (initial_stock : ℝ) (acquired : ℝ) (books_per_shelf : ℝ) 
  (h1 : initial_stock = 40.0)
  (h2 : acquired = 20.0)
  (h3 : books_per_shelf = 4.0) :
  (initial_stock + acquired) / books_per_shelf = 15 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l3158_315858


namespace NUMINAMATH_CALUDE_solution_fraction_l3158_315892

theorem solution_fraction (initial_amount : ℝ) (first_day_fraction : ℝ) (second_day_addition : ℝ) : 
  initial_amount = 4 →
  first_day_fraction = 1/2 →
  second_day_addition = 1 →
  (initial_amount - first_day_fraction * initial_amount + second_day_addition) / initial_amount = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_fraction_l3158_315892


namespace NUMINAMATH_CALUDE_solve_abc_l3158_315819

def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

theorem solve_abc (a b c : ℝ) :
  A a ≠ B b c ∧
  A a ∩ B b c = {-3} ∧
  A a ∪ B b c = {-3, 1, 4} →
  a = -1 ∧ b = 2 ∧ c = -3 := by
sorry

end NUMINAMATH_CALUDE_solve_abc_l3158_315819


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3158_315801

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x > y - 1) ∧
  (∃ x y : ℝ, x > y - 1 ∧ ¬(x > y)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3158_315801


namespace NUMINAMATH_CALUDE_books_to_tables_ratio_l3158_315846

theorem books_to_tables_ratio 
  (num_tables : ℕ) 
  (total_books : ℕ) 
  (h1 : num_tables = 500) 
  (h2 : total_books = 100000) : 
  (total_books / num_tables : ℚ) = 200 := by
sorry

end NUMINAMATH_CALUDE_books_to_tables_ratio_l3158_315846


namespace NUMINAMATH_CALUDE_min_decimal_digits_l3158_315860

def fraction : ℚ := 987654321 / (2^30 * 5^2)

theorem min_decimal_digits (f : ℚ) (h : f = fraction) : 
  (∃ (n : ℕ), n ≥ 30 ∧ ∃ (m : ℤ), f * 10^n = m) ∧ 
  (∀ (k : ℕ), k < 30 → ¬∃ (m : ℤ), f * 10^k = m) := by
  sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l3158_315860


namespace NUMINAMATH_CALUDE_initial_volume_calculation_l3158_315893

theorem initial_volume_calculation (initial_percentage : Real) 
  (final_percentage : Real) (pure_alcohol_added : Real) 
  (h1 : initial_percentage = 0.35)
  (h2 : final_percentage = 0.50)
  (h3 : pure_alcohol_added = 1.8) : 
  ∃ (initial_volume : Real), 
    initial_volume * initial_percentage + pure_alcohol_added = 
    (initial_volume + pure_alcohol_added) * final_percentage ∧ 
    initial_volume = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_volume_calculation_l3158_315893


namespace NUMINAMATH_CALUDE_median_of_special_list_l3158_315898

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total count of numbers in the list -/
def total_count : ℕ := triangular_number 250

/-- The position of the median in the list -/
def median_position : ℕ := total_count / 2 + 1

/-- The number that appears at the median position -/
def median_number : ℕ := 177

theorem median_of_special_list :
  median_number = 177 ∧
  triangular_number (median_number - 1) < median_position ∧
  median_position ≤ triangular_number median_number :=
sorry

end NUMINAMATH_CALUDE_median_of_special_list_l3158_315898


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3158_315817

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^205 + A*x + B = 0) → 
  A + B = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3158_315817
