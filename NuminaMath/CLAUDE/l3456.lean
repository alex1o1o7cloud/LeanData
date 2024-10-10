import Mathlib

namespace class_size_l3456_345694

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 40) :
  french + german - both + neither = 94 := by
  sorry

end class_size_l3456_345694


namespace valid_n_set_l3456_345683

def is_valid_n (n : ℕ) : Prop :=
  ∃ m : ℕ,
    n > 1 ∧
    (∀ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n → ∃ k : ℕ, k ∣ m ∧ k ≠ 1 ∧ k ≠ m ∧ k = d + 1) ∧
    (∀ k : ℕ, k ∣ m ∧ k ≠ 1 ∧ k ≠ m → ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n ∧ k = d + 1)

theorem valid_n_set : {n : ℕ | is_valid_n n} = {4, 8} := by sorry

end valid_n_set_l3456_345683


namespace solution_range_l3456_345666

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem solution_range (a b c : ℝ) :
  f a b c 1.1 < 3 ∧ 
  f a b c 1.2 < 3 ∧ 
  f a b c 1.3 < 3 ∧ 
  f a b c 1.4 > 3 →
  ∃ x, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
by
  sorry

end solution_range_l3456_345666


namespace complex_number_location_l3456_345646

theorem complex_number_location : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (Complex.I : ℂ) / (3 + Complex.I) = ⟨x, y⟩ := by
  sorry

end complex_number_location_l3456_345646


namespace pie_chart_most_suitable_for_air_composition_l3456_345626

/-- Represents different types of graphs -/
inductive GraphType
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

/-- Represents a component of air -/
structure AirComponent where
  name : String
  percentage : Float

/-- Determines if a graph type is suitable for representing percentage composition -/
def isSuitableForPercentageComposition (graphType : GraphType) : Prop :=
  match graphType with
  | GraphType.PieChart => True
  | _ => False

/-- The air composition representation problem -/
theorem pie_chart_most_suitable_for_air_composition 
  (components : List AirComponent) 
  (hComponents : components.all (λ c => c.percentage ≥ 0 ∧ c.percentage ≤ 100)) 
  (hTotalPercentage : components.foldl (λ acc c => acc + c.percentage) 0 = 100) :
  isSuitableForPercentageComposition GraphType.PieChart ∧ 
  (∀ g : GraphType, isSuitableForPercentageComposition g → g = GraphType.PieChart) :=
sorry

end pie_chart_most_suitable_for_air_composition_l3456_345626


namespace inequality_proof_l3456_345656

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a :=
by sorry

end inequality_proof_l3456_345656


namespace inequality_impossibility_l3456_345624

theorem inequality_impossibility (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end inequality_impossibility_l3456_345624


namespace sqrt_450_simplified_l3456_345634

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplified_l3456_345634


namespace largest_inscribed_circle_radius_largest_inscribed_circle_radius_proof_l3456_345661

/-- The radius of the largest inscribed circle in a square with side length 15,
    outside two congruent equilateral triangles sharing one side and each having
    one vertex on a vertex of the square. -/
theorem largest_inscribed_circle_radius : ℝ :=
  let square_side : ℝ := 15
  let triangle_side : ℝ := (square_side * Real.sqrt 6 - square_side * Real.sqrt 2) / 2
  let circle_radius : ℝ := square_side / 2 - (square_side * Real.sqrt 6 - square_side * Real.sqrt 2) / 8
  circle_radius

/-- Proof that the radius of the largest inscribed circle is correct. -/
theorem largest_inscribed_circle_radius_proof :
  largest_inscribed_circle_radius = 7.5 - (15 * Real.sqrt 6 - 15 * Real.sqrt 2) / 4 := by
  sorry

end largest_inscribed_circle_radius_largest_inscribed_circle_radius_proof_l3456_345661


namespace quadratic_rewrite_product_l3456_345610

/-- Given a quadratic equation 16x^2 - 40x - 24 that can be rewritten as (dx + e)^2 + f,
    where d, e, and f are integers, prove that de = -20 -/
theorem quadratic_rewrite_product (d e f : ℤ) : 
  (∀ x, 16 * x^2 - 40 * x - 24 = (d * x + e)^2 + f) → d * e = -20 := by
  sorry

end quadratic_rewrite_product_l3456_345610


namespace average_of_abc_is_three_l3456_345675

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 1503 * C - 3006 * A = 6012)
  (eq2 : 1503 * B + 4509 * A = 7509) :
  (A + B + C) / 3 = 3 := by
sorry

end average_of_abc_is_three_l3456_345675


namespace train_length_l3456_345681

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 300 → time = 15 → ∃ length : ℝ, 
  (abs (length - 1249.95) < 0.01) ∧ 
  (length = speed * 1000 / 3600 * time) := by
  sorry

end train_length_l3456_345681


namespace parabola_y_relationship_l3456_345662

-- Define the parabola function
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the theorem
theorem parabola_y_relationship (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : parabola (-2) c = y₁)
  (h2 : parabola 0 c = y₂)
  (h3 : parabola (5/3) c = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry


end parabola_y_relationship_l3456_345662


namespace equal_roots_quadratic_l3456_345622

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + a*y + 1 = 0 → y = x) → 
  a = -2 ∨ a = 2 := by
sorry

end equal_roots_quadratic_l3456_345622


namespace playground_children_count_l3456_345639

/-- The number of boys on the playground at recess -/
def num_boys : ℕ := 27

/-- The number of girls on the playground at recess -/
def num_girls : ℕ := 35

/-- The total number of children on the playground at recess -/
def total_children : ℕ := num_boys + num_girls

theorem playground_children_count : total_children = 62 := by
  sorry

end playground_children_count_l3456_345639


namespace smallest_integer_with_remainders_l3456_345628

theorem smallest_integer_with_remainders : 
  ∃ x : ℕ, 
    (x > 0) ∧ 
    (x % 5 = 4) ∧ 
    (x % 6 = 5) ∧ 
    (x % 7 = 6) ∧ 
    (∀ y : ℕ, y > 0 → y % 5 = 4 → y % 6 = 5 → y % 7 = 6 → x ≤ y) ∧
    x = 209 := by
  sorry

end smallest_integer_with_remainders_l3456_345628


namespace fraction_difference_equals_one_l3456_345691

theorem fraction_difference_equals_one (x y : ℝ) (h : x ≠ y) :
  x / (x - y) - y / (x - y) = 1 := by
  sorry

end fraction_difference_equals_one_l3456_345691


namespace product_of_roots_l3456_345670

theorem product_of_roots (x : ℝ) : 
  (∃ a b c : ℝ, a * b * c = -9 ∧ 
   ∀ x, 4 * x^3 - 2 * x^2 - 25 * x + 36 = 0 ↔ (x = a ∨ x = b ∨ x = c)) :=
by sorry

end product_of_roots_l3456_345670


namespace intersection_of_intervals_solution_interval_l3456_345689

theorem intersection_of_intervals : Set.Ioo (1/2 : ℝ) (3/5 : ℝ) = 
  Set.inter 
    (Set.Ioo (1/2 : ℝ) (3/4 : ℝ)) 
    (Set.Ioo (2/5 : ℝ) (3/5 : ℝ)) := by sorry

theorem solution_interval (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 
  x ∈ Set.Ioo (1/2 : ℝ) (3/5 : ℝ) := by sorry

end intersection_of_intervals_solution_interval_l3456_345689


namespace sqrt_equation_solution_l3456_345699

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end sqrt_equation_solution_l3456_345699


namespace smallest_third_altitude_l3456_345620

/-- Represents a scalene triangle with altitudes --/
structure ScaleneTriangle where
  -- The lengths of the three sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the three altitudes
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  -- Conditions for a scalene triangle
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Area equality for altitudes
  area_equality : a * h_a = b * h_b ∧ b * h_b = c * h_c

/-- The theorem stating the smallest possible integer length for the third altitude --/
theorem smallest_third_altitude (t : ScaleneTriangle) 
  (h1 : t.h_a = 6 ∨ t.h_b = 6 ∨ t.h_c = 6)
  (h2 : t.h_a = 8 ∨ t.h_b = 8 ∨ t.h_c = 8)
  (h3 : ∃ (n : ℕ), t.h_a = n ∨ t.h_b = n ∨ t.h_c = n) :
  ∃ (h : ScaleneTriangle), h.h_a = 6 ∧ h.h_b = 8 ∧ h.h_c = 2 :=
sorry

end smallest_third_altitude_l3456_345620


namespace factor_sum_18_with_2_l3456_345660

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem factor_sum_18_with_2 (x : ℕ) 
  (h1 : x > 0) 
  (h2 : sum_of_factors x = 18) 
  (h3 : 2 ∣ x) : 
  x = 10 := by
  sorry

end factor_sum_18_with_2_l3456_345660


namespace number_of_dogs_l3456_345606

/-- The number of dogs at a farm, given the number of fish, cats, and total pets. -/
theorem number_of_dogs (fish : ℕ) (cats : ℕ) (total_pets : ℕ) : 
  fish = 72 → cats = 34 → total_pets = 149 → total_pets = fish + cats + 43 :=
by sorry

end number_of_dogs_l3456_345606


namespace divisibility_property_not_true_for_p_2_l3456_345686

theorem divisibility_property (p a n : ℕ) : 
  Nat.Prime p → p ≠ 2 → a > 0 → n > 0 → (p^n ∣ a^p - 1) → (p^(n-1) ∣ a - 1) :=
by sorry

-- The statement is not true for p = 2
theorem not_true_for_p_2 : 
  ∃ (a n : ℕ), a > 0 ∧ n > 0 ∧ (2^n ∣ a^2 - 1) ∧ ¬(2^(n-1) ∣ a - 1) :=
by sorry

end divisibility_property_not_true_for_p_2_l3456_345686


namespace max_intersection_points_circle_rectangle_l3456_345676

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rectangle in a plane -/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment -/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The maximum number of intersection points between a circle and a rectangle -/
def maxIntersectionPoints (c : Circle) (r : Rectangle) : ℕ :=
  (intersectionPointsCircleLine c (r.corners 0) (r.corners 1)) +
  (intersectionPointsCircleLine c (r.corners 1) (r.corners 2)) +
  (intersectionPointsCircleLine c (r.corners 2) (r.corners 3)) +
  (intersectionPointsCircleLine c (r.corners 3) (r.corners 0))

/-- Theorem: The maximum number of intersection points between a circle and a rectangle is 8 -/
theorem max_intersection_points_circle_rectangle :
  ∀ (c : Circle) (r : Rectangle), maxIntersectionPoints c r ≤ 8 ∧
  ∃ (c : Circle) (r : Rectangle), maxIntersectionPoints c r = 8 :=
sorry

end max_intersection_points_circle_rectangle_l3456_345676


namespace family_weight_theorem_l3456_345678

/-- Represents the weights of a family with three generations -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- The total weight of the family -/
def FamilyWeights.total (w : FamilyWeights) : ℝ :=
  w.mother + w.daughter + w.grandchild

/-- The conditions given in the problem -/
def WeightConditions (w : FamilyWeights) : Prop :=
  w.daughter + w.grandchild = 60 ∧
  w.grandchild = (1/5) * w.mother ∧
  w.daughter = 50

/-- Theorem stating that given the conditions, the total weight is 110 kg -/
theorem family_weight_theorem (w : FamilyWeights) (h : WeightConditions w) :
  w.total = 110 := by
  sorry


end family_weight_theorem_l3456_345678


namespace unique_pronunciations_in_C_l3456_345673

/-- Represents a Chinese character with its pronunciation --/
structure ChineseChar :=
  (char : String)
  (pronunciation : String)

/-- Represents a group of words with underlined characters --/
structure WordGroup :=
  (name : String)
  (underlinedChars : List ChineseChar)

/-- Check if all pronunciations in a list are unique --/
def allUniquePronunciations (chars : List ChineseChar) : Prop :=
  ∀ i j, i ≠ j → (chars.get i).pronunciation ≠ (chars.get j).pronunciation

/-- The four word groups from the problem --/
def groupA : WordGroup := sorry
def groupB : WordGroup := sorry
def groupC : WordGroup := sorry
def groupD : WordGroup := sorry

/-- The main theorem to prove --/
theorem unique_pronunciations_in_C :
  allUniquePronunciations groupC.underlinedChars ∧
  ¬allUniquePronunciations groupA.underlinedChars ∧
  ¬allUniquePronunciations groupB.underlinedChars ∧
  ¬allUniquePronunciations groupD.underlinedChars :=
sorry

end unique_pronunciations_in_C_l3456_345673


namespace base_85_subtraction_divisibility_l3456_345651

theorem base_85_subtraction_divisibility (b : ℤ) : 
  (0 ≤ b ∧ b ≤ 20) → 
  (∃ k : ℤ, 346841047 * 85^8 + 4 * 85^7 + 1 * 85^5 + 4 * 85^4 + 8 * 85^3 + 6 * 85^2 + 4 * 85 + 3 - b = 17 * k) → 
  b = 3 := by
sorry

end base_85_subtraction_divisibility_l3456_345651


namespace octal_multiplication_53_26_l3456_345679

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Multiplies two octal numbers --/
def octal_multiply (a b : ℕ) : ℕ :=
  decimal_to_octal (octal_to_decimal a * octal_to_decimal b)

theorem octal_multiplication_53_26 :
  octal_multiply 53 26 = 1662 := by sorry

end octal_multiplication_53_26_l3456_345679


namespace absolute_value_sum_l3456_345604

theorem absolute_value_sum (y q : ℝ) (h1 : |y - 5| = q) (h2 : y > 5) : y + q = 2*q + 5 := by
  sorry

end absolute_value_sum_l3456_345604


namespace expression_simplification_l3456_345614

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((((x + 2)^2 * (x^2 - 2*x + 2)^2) / (x^3 + 8)^2)^2 * 
   (((x - 2)^2 * (x^2 + 2*x + 2)^2) / (x^3 - 8)^2)^2) = 1 := by
  sorry

end expression_simplification_l3456_345614


namespace opposite_numbers_sum_power_l3456_345665

/-- If a and b are opposite numbers, then (a+b)^2023 = 0 -/
theorem opposite_numbers_sum_power (a b : ℝ) : a = -b → (a + b)^2023 = 0 := by
  sorry

end opposite_numbers_sum_power_l3456_345665


namespace x_value_l3456_345663

/-- A sequence where the differences between successive terms increase by 3 each time -/
def increasing_diff_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3 * n + 3

/-- The specific sequence from the problem -/
def our_sequence (a : ℕ → ℕ) : Prop :=
  increasing_diff_sequence a ∧ a 0 = 2 ∧ a 5 = 47

theorem x_value (a : ℕ → ℕ) (h : our_sequence a) : a 4 = 32 := by
  sorry

end x_value_l3456_345663


namespace equal_area_split_line_slope_l3456_345677

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Checks if a line splits the area of circles equally -/
def splitAreaEqually (line : Line) (circles : List Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem equal_area_split_line_slope :
  let circles : List Circle := [
    { center := (10, 100), radius := 4 },
    { center := (13, 82),  radius := 4 },
    { center := (15, 90),  radius := 4 }
  ]
  let line : Line := { slope := 0.5, passesThrough := (13, 82) }
  splitAreaEqually line circles ∧ 
  ∀ (m : ℝ), splitAreaEqually { slope := m, passesThrough := (13, 82) } circles → 
    |m| = 0.5 := by
  sorry

end equal_area_split_line_slope_l3456_345677


namespace residue_mod_13_l3456_345657

theorem residue_mod_13 : (250 * 11 - 20 * 6 + 5^2) % 13 = 3 := by sorry

end residue_mod_13_l3456_345657


namespace line_segment_endpoint_l3456_345690

/-- Given a line segment with midpoint (3, 4) and one endpoint (7, 10), 
    prove that the other endpoint is (-1, -2). -/
theorem line_segment_endpoint (M A B : ℝ × ℝ) : 
  M = (3, 4) → A = (7, 10) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B = (-1, -2) := by
  sorry

end line_segment_endpoint_l3456_345690


namespace monotonicity_intervals_k_range_l3456_345642

/-- The function f(x) = xe^(kx) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x * Real.exp (k * x)

/-- Monotonicity intervals for f(x) when k > 0 -/
theorem monotonicity_intervals (k : ℝ) (h : k > 0) :
  (∀ x₁ x₂, x₁ < x₂ ∧ - 1 / k < x₁ → f k x₁ < f k x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < - 1 / k → f k x₁ > f k x₂) :=
sorry

/-- Range of k when f(x) is monotonically increasing in (-1, 1) -/
theorem k_range (k : ℝ) (h : k ≠ 0) :
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f k x₁ < f k x₂) →
  (k ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1) :=
sorry

end monotonicity_intervals_k_range_l3456_345642


namespace x_coordinate_of_R_is_one_l3456_345608

/-- The curve on which point R lies -/
def curve (x y : ℝ) : Prop := y = -2 * x^2 + 5 * x - 2

/-- Predicate to check if OMRN is a square -/
def is_square (O M R N : ℝ × ℝ) : Prop := sorry

/-- Theorem stating that the x-coordinate of R is 1 -/
theorem x_coordinate_of_R_is_one 
  (R : ℝ × ℝ) 
  (h1 : curve R.1 R.2)
  (h2 : is_square (0, 0) (R.1, 0) R (0, R.2)) : 
  R.1 = 1 := by sorry

end x_coordinate_of_R_is_one_l3456_345608


namespace article_cost_l3456_345682

theorem article_cost (decreased_price : ℝ) (decrease_percentage : ℝ) (actual_cost : ℝ) :
  decreased_price = 200 ∧
  decrease_percentage = 20 ∧
  decreased_price = actual_cost * (1 - decrease_percentage / 100) →
  actual_cost = 250 := by
  sorry

end article_cost_l3456_345682


namespace final_dislikes_is_300_l3456_345659

/-- Represents the number of likes and dislikes on a YouTube video -/
structure VideoStats where
  likes : ℕ
  dislikes : ℕ

/-- Calculates the final number of dislikes after changes -/
def finalDislikes (original : VideoStats) : ℕ :=
  3 * original.dislikes

/-- Theorem: Given the conditions, the final number of dislikes is 300 -/
theorem final_dislikes_is_300 (original : VideoStats) 
    (h1 : original.likes = 3 * original.dislikes)
    (h2 : original.likes = 100 + 2 * original.dislikes) : 
  finalDislikes original = 300 := by
  sorry

#eval finalDislikes {likes := 300, dislikes := 100}

end final_dislikes_is_300_l3456_345659


namespace units_digit_34_plus_47_base_8_l3456_345655

def base_8_addition (a b : Nat) : Nat :=
  (a + b) % 8

theorem units_digit_34_plus_47_base_8 :
  base_8_addition (34 % 8) (47 % 8) = 3 := by
  sorry

end units_digit_34_plus_47_base_8_l3456_345655


namespace initial_deposit_l3456_345613

theorem initial_deposit (P R : ℝ) : 
  P + (P * R * 3) / 100 = 9200 →
  P + (P * (R + 2.5) * 3) / 100 = 9800 →
  P = 8000 := by
sorry

end initial_deposit_l3456_345613


namespace rectangle_side_length_l3456_345641

theorem rectangle_side_length (perimeter width : ℝ) (h1 : perimeter = 40) (h2 : width = 8) :
  perimeter / 2 - width = 12 :=
by sorry

end rectangle_side_length_l3456_345641


namespace johns_nap_hours_l3456_345671

/-- Calculates the total hours of naps taken over a given number of days -/
def total_nap_hours (naps_per_week : ℕ) (hours_per_nap : ℕ) (total_days : ℕ) : ℕ :=
  (total_days / 7) * naps_per_week * hours_per_nap

/-- Theorem: John's total nap hours in 70 days -/
theorem johns_nap_hours :
  total_nap_hours 3 2 70 = 60 := by
  sorry

end johns_nap_hours_l3456_345671


namespace trig_identity_1_trig_identity_2_l3456_345697

-- Part 1
theorem trig_identity_1 (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

-- Part 2
theorem trig_identity_2 (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = Real.sqrt 5 / 5)
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  2 * β = π/4 := by
  sorry

end trig_identity_1_trig_identity_2_l3456_345697


namespace sqrt_four_ninths_l3456_345667

theorem sqrt_four_ninths :
  Real.sqrt (4/9) = 2/3 ∨ Real.sqrt (4/9) = -2/3 := by
  sorry

end sqrt_four_ninths_l3456_345667


namespace jim_initial_tree_rows_l3456_345631

/-- Proves that Jim started with 2 rows of trees given the problem conditions -/
theorem jim_initial_tree_rows : ∀ (initial_rows : ℕ), 
  (∀ (row : ℕ), row > 0 → row ≤ initial_rows + 5 → 4 * row ≤ 56) ∧
  (2 * (4 * (initial_rows + 5)) = 56) →
  initial_rows = 2 :=
by
  sorry

end jim_initial_tree_rows_l3456_345631


namespace find_number_l3456_345644

theorem find_number (n x : ℝ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 := by
  sorry

end find_number_l3456_345644


namespace meet_after_four_turns_l3456_345601

-- Define the number of points on the circular track
def num_points : ℕ := 15

-- Define Alice's clockwise movement per turn
def alice_move : ℕ := 4

-- Define Bob's counterclockwise movement per turn
def bob_move : ℕ := 11

-- Define the starting point for both Alice and Bob
def start_point : ℕ := 15

-- Function to calculate the new position after a move
def new_position (current : ℕ) (move : ℕ) : ℕ :=
  ((current + move - 1) % num_points) + 1

-- Function to calculate Alice's position after n turns
def alice_position (n : ℕ) : ℕ :=
  new_position start_point (n * alice_move)

-- Function to calculate Bob's position after n turns
def bob_position (n : ℕ) : ℕ :=
  new_position start_point (n * (num_points - bob_move))

-- Theorem stating that Alice and Bob meet after 4 turns
theorem meet_after_four_turns :
  ∃ n : ℕ, n = 4 ∧ alice_position n = bob_position n :=
sorry

end meet_after_four_turns_l3456_345601


namespace max_xyz_value_l3456_345616

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + z = (x + z) * (y + z)) :
  x * y * z ≤ 1 / 27 := by
sorry

end max_xyz_value_l3456_345616


namespace phone_profit_fraction_l3456_345649

theorem phone_profit_fraction (num_phones : ℕ) (initial_investment : ℚ) (selling_price : ℚ) :
  num_phones = 200 →
  initial_investment = 3000 →
  selling_price = 20 →
  (num_phones * selling_price - initial_investment) / initial_investment = 1/3 := by
sorry

end phone_profit_fraction_l3456_345649


namespace non_monotonic_quadratic_l3456_345685

/-- A function f is not monotonic on an interval [a, b] if there exists
    x, y in [a, b] such that x < y and f(x) > f(y) -/
def NotMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ f x > f y

/-- The quadratic function f(x) = 4x^2 - kx - 8 -/
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

theorem non_monotonic_quadratic (k : ℝ) :
  NotMonotonic (f k) 5 8 ↔ k ∈ Set.Ioo 40 64 := by
  sorry

end non_monotonic_quadratic_l3456_345685


namespace idaho_to_nevada_distance_l3456_345650

/-- Represents the road trip from Washington to Nevada via Idaho -/
structure RoadTrip where
  wash_to_idaho : ℝ     -- Distance from Washington to Idaho
  idaho_to_nevada : ℝ   -- Distance from Idaho to Nevada (to be proven)
  speed_to_idaho : ℝ    -- Speed from Washington to Idaho
  speed_to_nevada : ℝ   -- Speed from Idaho to Nevada
  total_time : ℝ        -- Total travel time

/-- The road trip satisfies the given conditions -/
def satisfies_conditions (trip : RoadTrip) : Prop :=
  trip.wash_to_idaho = 640 ∧
  trip.speed_to_idaho = 80 ∧
  trip.speed_to_nevada = 50 ∧
  trip.total_time = 19 ∧
  trip.total_time = trip.wash_to_idaho / trip.speed_to_idaho + trip.idaho_to_nevada / trip.speed_to_nevada

theorem idaho_to_nevada_distance (trip : RoadTrip) 
  (h : satisfies_conditions trip) : trip.idaho_to_nevada = 550 := by
  sorry

end idaho_to_nevada_distance_l3456_345650


namespace cube_with_holes_surface_area_l3456_345621

/-- Calculates the total surface area of a cube with holes --/
def totalSurfaceArea (cubeEdgeLength : ℝ) (holeEdgeLength : ℝ) : ℝ :=
  let originalSurfaceArea := 6 * cubeEdgeLength^2
  let holeArea := 6 * holeEdgeLength^2
  let exposedInsideArea := 6 * 4 * holeEdgeLength^2
  originalSurfaceArea - holeArea + exposedInsideArea

/-- The total surface area of a cube with edge length 4 and holes of side length 2 is 168 --/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 4 2 = 168 := by
  sorry

#eval totalSurfaceArea 4 2

end cube_with_holes_surface_area_l3456_345621


namespace light_path_length_l3456_345648

-- Define the cube side length
def cube_side : ℝ := 10

-- Define the reflection point coordinates relative to the face
def reflect_x : ℝ := 4
def reflect_y : ℝ := 6

-- Define the number of reflections needed
def num_reflections : ℕ := 10

-- Theorem statement
theorem light_path_length :
  let path_length := (num_reflections : ℝ) * Real.sqrt (cube_side^2 + reflect_x^2 + reflect_y^2)
  path_length = 10 * Real.sqrt 152 :=
by sorry

end light_path_length_l3456_345648


namespace trig_expression_simplification_l3456_345643

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π/2 + α) * Real.sin (π + α) * Real.tan (3*π + α)) /
  (Real.cos (3*π/2 + α) * Real.sin (-α)) = 1 := by
  sorry

end trig_expression_simplification_l3456_345643


namespace book_cost_l3456_345607

theorem book_cost (book_cost bookmark_cost : ℚ) 
  (total_cost : book_cost + bookmark_cost = (7/2 : ℚ))
  (price_difference : book_cost = bookmark_cost + 3) : 
  book_cost = (13/4 : ℚ) := by
sorry

end book_cost_l3456_345607


namespace area_ratio_quad_to_decagon_l3456_345693

-- Define a regular decagon
structure RegularDecagon where
  vertices : Fin 10 → ℝ × ℝ
  is_regular : sorry

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Define the quadrilateral ACEG within the decagon
def quadACEG (d : RegularDecagon) : List (ℝ × ℝ) :=
  [d.vertices 0, d.vertices 2, d.vertices 4, d.vertices 6]

-- Define the decagon as a list of points
def decagonPoints (d : RegularDecagon) : List (ℝ × ℝ) :=
  (List.range 10).map d.vertices

theorem area_ratio_quad_to_decagon (d : RegularDecagon) :
  area (quadACEG d) / area (decagonPoints d) = 1 / 4 := by
  sorry

end area_ratio_quad_to_decagon_l3456_345693


namespace shawn_pebble_groups_l3456_345638

theorem shawn_pebble_groups :
  let total_pebbles : ℕ := 40
  let red_pebbles : ℕ := 9
  let blue_pebbles : ℕ := 13
  let remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
  let blue_yellow_diff : ℕ := 7
  let yellow_pebbles : ℕ := blue_pebbles - blue_yellow_diff
  let num_colors : ℕ := 3  -- purple, yellow, and green
  ∃ (num_groups : ℕ),
    num_groups > 0 ∧
    num_groups ∣ remaining_pebbles ∧
    num_groups % num_colors = 0 ∧
    remaining_pebbles / num_groups = yellow_pebbles ∧
    num_groups = 3
  := by sorry

end shawn_pebble_groups_l3456_345638


namespace unique_score_with_four_ways_l3456_345696

/-- AMC scoring system -/
structure AMCScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total_questions : ℕ
  score : ℕ

/-- Predicate for valid AMC score -/
def is_valid_score (s : AMCScore) : Prop :=
  s.correct + s.unanswered + s.incorrect = s.total_questions ∧
  s.score = 7 * s.correct + 3 * s.unanswered

/-- Theorem: Unique score with exactly four distinct ways to achieve it -/
theorem unique_score_with_four_ways :
  ∃! S : ℕ, 
    (∃ scores : Finset AMCScore, 
      (∀ s ∈ scores, is_valid_score s ∧ s.total_questions = 30 ∧ s.score = S) ∧
      scores.card = 4 ∧
      (∀ s : AMCScore, is_valid_score s ∧ s.total_questions = 30 ∧ s.score = S → s ∈ scores)) ∧
    S = 240 := by
  sorry

end unique_score_with_four_ways_l3456_345696


namespace johnson_family_seating_l3456_345600

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_seatings (n : ℕ) : ℕ := factorial n

def boys_not_adjacent (boys girls : ℕ) : ℕ := 
  2 * factorial boys * factorial girls

theorem johnson_family_seating (boys girls : ℕ) : 
  boys = 5 → girls = 4 → 
  total_seatings (boys + girls) - boys_not_adjacent boys girls = 357120 := by
  sorry

end johnson_family_seating_l3456_345600


namespace exam_score_problem_l3456_345625

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 150 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 42 :=
by sorry

end exam_score_problem_l3456_345625


namespace find_S_l3456_345611

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}
def B : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b < 0}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | x - 3 < 4 ∧ 4 ≤ 2*x}

-- State the theorem
theorem find_S (a b : ℝ) : 
  A ∩ B = ∅ → 
  A ∪ B = AUnionB → 
  {x | x = a + b} = {23} := by sorry

end find_S_l3456_345611


namespace mark_fruit_count_l3456_345632

/-- The number of pieces of fruit Mark had at the beginning of the week -/
def total_fruit (kept_for_next_week : ℕ) (brought_to_school : ℕ) (eaten_first_four_days : ℕ) : ℕ :=
  kept_for_next_week + brought_to_school + eaten_first_four_days

/-- Theorem stating that Mark had 10 pieces of fruit at the beginning of the week -/
theorem mark_fruit_count : total_fruit 2 3 5 = 10 := by
  sorry

end mark_fruit_count_l3456_345632


namespace cookies_problem_l3456_345605

theorem cookies_problem (millie mike frank : ℕ) : 
  millie = 4 →
  mike = 3 * millie →
  frank = mike / 2 - 3 →
  frank = 3 := by
sorry

end cookies_problem_l3456_345605


namespace lost_in_mountains_second_group_size_l3456_345645

theorem lost_in_mountains (initial_people : ℕ) (initial_days : ℕ) (days_after_sharing : ℕ) : ℕ :=
  let initial_portions := initial_people * initial_days
  let remaining_portions := initial_portions - initial_people
  let total_people := initial_people + (remaining_portions / (days_after_sharing + 1) - initial_people)
  remaining_portions / (days_after_sharing + 1) - initial_people

theorem second_group_size :
  lost_in_mountains 9 5 3 = 3 := by
  sorry

end lost_in_mountains_second_group_size_l3456_345645


namespace average_goals_l3456_345695

theorem average_goals (layla_goals : ℕ) (kristin_difference : ℕ) (num_games : ℕ) :
  layla_goals = 104 →
  kristin_difference = 24 →
  num_games = 4 →
  (layla_goals + (layla_goals - kristin_difference)) / num_games = 46 :=
by
  sorry

end average_goals_l3456_345695


namespace geometric_number_difference_l3456_345619

/-- A function that checks if a 3-digit number has distinct digits forming a geometric sequence --/
def is_geometric_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b * b = a * c

/-- The largest 3-digit number with distinct digits forming a geometric sequence --/
def largest_geometric_number : ℕ := 964

/-- The smallest 3-digit number with distinct digits forming a geometric sequence --/
def smallest_geometric_number : ℕ := 124

theorem geometric_number_difference :
  is_geometric_number largest_geometric_number ∧
  is_geometric_number smallest_geometric_number ∧
  (∀ n : ℕ, is_geometric_number n → 
    smallest_geometric_number ≤ n ∧ n ≤ largest_geometric_number) ∧
  largest_geometric_number - smallest_geometric_number = 840 := by
  sorry

end geometric_number_difference_l3456_345619


namespace smallest_integer_with_remainders_l3456_345617

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  M > 0 ∧
  M % 4 = 3 ∧
  M % 5 = 4 ∧
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  ∀ n : ℕ, n > 0 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → M ≤ n :=
by
  -- Proof goes here
  sorry

end smallest_integer_with_remainders_l3456_345617


namespace distinct_arrangements_count_is_eight_l3456_345669

/-- Represents a key on the keychain -/
inductive Key
| House
| Car
| Work
| Garage
| Other

/-- Represents a pair of adjacent keys -/
structure KeyPair :=
  (first : Key)
  (second : Key)

/-- Represents an arrangement of keys on the keychain -/
structure KeyArrangement :=
  (pair1 : KeyPair)
  (pair2 : KeyPair)
  (single : Key)

/-- Checks if two KeyArrangements are considered identical (up to rotation and reflection) -/
def are_identical (a b : KeyArrangement) : Prop := sorry

/-- The set of all valid key arrangements -/
def valid_arrangements : Set KeyArrangement :=
  {arr | (arr.pair1.first = Key.House ∧ arr.pair1.second = Key.Car) ∨
         (arr.pair1.first = Key.Car ∧ arr.pair1.second = Key.House) ∧
         (arr.pair2.first = Key.Work ∧ arr.pair2.second = Key.Garage) ∨
         (arr.pair2.first = Key.Garage ∧ arr.pair2.second = Key.Work) ∧
         arr.single = Key.Other}

/-- The number of distinct arrangements -/
def distinct_arrangement_count : ℕ := sorry

theorem distinct_arrangements_count_is_eight :
  distinct_arrangement_count = 8 := by sorry

end distinct_arrangements_count_is_eight_l3456_345669


namespace simplify_expression_1_simplify_expression_2_l3456_345654

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  x^2 + (3*x - 5) - (4*x - 1) = x^2 - x - 4 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b := by sorry

end simplify_expression_1_simplify_expression_2_l3456_345654


namespace inscribed_cube_volume_l3456_345664

/-- A pyramid with a square base and right-angled isosceles triangular lateral faces -/
structure Pyramid :=
  (base_side : ℝ)

/-- A cube inscribed in a pyramid -/
structure InscribedCube :=
  (side_length : ℝ)

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.side_length ^ 3

/-- Predicate for a cube being properly inscribed in the pyramid -/
def is_properly_inscribed (p : Pyramid) (c : InscribedCube) : Prop :=
  c.side_length > 0 ∧ c.side_length < p.base_side

theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube) 
  (h1 : p.base_side = 2)
  (h2 : is_properly_inscribed p c) :
  cube_volume c = 10 + 6 * Real.sqrt 3 :=
sorry

end inscribed_cube_volume_l3456_345664


namespace triangle_area_l3456_345652

/-- Given a triangle ABC where angle A is 30°, angle B is 45°, and side a is 2,
    prove that the area of the triangle is √3 + 1. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 2 →
  (1/2) * a * b * Real.sin (π - A - B) = Real.sqrt 3 + 1 :=
by sorry

end triangle_area_l3456_345652


namespace hyperbola_asymptote_l3456_345687

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and focal distance length 8,
    prove that its asymptote equation is y = ±(√7/3)x -/
theorem hyperbola_asymptote (m : ℝ) :
  (∀ x y, x^2 / 9 - y^2 / m = 1) →  -- Hyperbola equation
  (∃ c, c = 4 ∧ c^2 = 9 + m) →      -- Focal distance condition
  (∃ k, ∀ x, y = k * x ∨ y = -k * x) ∧ k = Real.sqrt 7 / 3 :=
by sorry

end hyperbola_asymptote_l3456_345687


namespace parabola_intersection_difference_l3456_345612

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

def intersection_points : Set ℝ := {x : ℝ | parabola1 x = parabola2 x}

theorem parabola_intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧ c ≥ a ∧ c - a = 2/5 :=
sorry

end parabola_intersection_difference_l3456_345612


namespace square_difference_l3456_345627

theorem square_difference : (625 : ℤ)^2 - (375 : ℤ)^2 = 250000 := by
  sorry

end square_difference_l3456_345627


namespace point_coordinates_l3456_345653

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the second quadrant
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- Define distance to x-axis
def distToXAxis (p : Point) : ℝ :=
  |p.y|

-- Define distance to y-axis
def distToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (M : Point) :
  secondQuadrant M ∧ distToXAxis M = 1 ∧ distToYAxis M = 2 →
  M.x = -2 ∧ M.y = 1 := by sorry

end point_coordinates_l3456_345653


namespace intersection_line_equation_l3456_345688

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def line_through_intersections (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 26/3

theorem intersection_line_equation :
  let c1 : Circle := ⟨(2, -3), 10⟩
  let c2 : Circle := ⟨(-4, 7), 6⟩
  ∀ x y : ℝ,
    (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧
    (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2 →
    line_through_intersections c1 c2 x y :=
by sorry

end intersection_line_equation_l3456_345688


namespace tan_product_ninth_pi_l3456_345629

theorem tan_product_ninth_pi : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 1 := by
  sorry

end tan_product_ninth_pi_l3456_345629


namespace vending_machine_probability_l3456_345658

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 10

/-- The price of the cheapest toy in cents -/
def min_price : ℕ := 50

/-- The price increment between toys in cents -/
def price_increment : ℕ := 25

/-- The price of Sam's favorite toy in cents -/
def favorite_toy_price : ℕ := 225

/-- The number of quarters Sam has initially -/
def initial_quarters : ℕ := 12

/-- The value of Sam's bill in cents -/
def bill_value : ℕ := 2000

/-- The probability that Sam has to break his twenty-dollar bill -/
def probability_break_bill : ℚ := 8/9

theorem vending_machine_probability :
  let total_permutations := Nat.factorial num_toys
  let favorable_permutations := Nat.factorial (num_toys - 1) + Nat.factorial (num_toys - 2)
  probability_break_bill = 1 - (favorable_permutations : ℚ) / total_permutations :=
by sorry

end vending_machine_probability_l3456_345658


namespace maddies_mom_milk_consumption_l3456_345672

/-- Represents the weekly coffee consumption scenario of Maddie's mom -/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  ounces_per_bag : ℚ
  price_per_bag : ℚ
  price_per_gallon_milk : ℚ
  weekly_coffee_budget : ℚ

/-- Calculates the amount of milk in gallons used per week -/
def milk_gallons_per_week (c : CoffeeConsumption) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, 
    the amount of milk used per week is 0.5 gallons -/
theorem maddies_mom_milk_consumption :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    ounces_per_cup := 3/2,
    ounces_per_bag := 21/2,
    price_per_bag := 8,
    price_per_gallon_milk := 4,
    weekly_coffee_budget := 18
  }
  milk_gallons_per_week c = 1/2 :=
by sorry

end maddies_mom_milk_consumption_l3456_345672


namespace service_cost_equations_global_connect_more_cost_effective_l3456_345668

/-- Represents the cost of a mobile communication service based on monthly fee and per-minute rate -/
def service_cost (monthly_fee : ℝ) (per_minute_rate : ℝ) (minutes : ℝ) : ℝ :=
  monthly_fee + per_minute_rate * minutes

/-- Theorem stating the cost equations for Global Connect and Quick Connect services -/
theorem service_cost_equations 
  (x : ℝ) 
  (y1 : ℝ) 
  (y2 : ℝ) : 
  y1 = service_cost 50 0.4 x ∧ 
  y2 = service_cost 0 0.6 x :=
sorry

/-- Theorem stating that Global Connect is more cost-effective for 300 minutes of calls -/
theorem global_connect_more_cost_effective : 
  service_cost 50 0.4 300 < service_cost 0 0.6 300 :=
sorry

end service_cost_equations_global_connect_more_cost_effective_l3456_345668


namespace x_squared_minus_y_squared_l3456_345615

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 1 / 45) : x^2 - y^2 = 8 / 675 := by
  sorry

end x_squared_minus_y_squared_l3456_345615


namespace divisibility_problem_l3456_345635

theorem divisibility_problem :
  (∃ n : ℕ, n = 9 ∧ (1100 + n) % 53 = 0 ∧ ∀ k : ℕ, k < n → (1100 + k) % 53 ≠ 0) ∧
  (∃ m : ℕ, m = 0 ∧ (1100 - m) % 71 = 0 ∧ ∀ k : ℕ, k < m → (1100 - k) % 71 ≠ 0) ∧
  (∃ X : ℤ, X = 534 ∧ (1100 + X) % 19 = 0 ∧ (1100 + X) % 43 = 0) :=
by sorry

end divisibility_problem_l3456_345635


namespace dart_probability_l3456_345684

/-- The probability of a dart landing within a circular target area inscribed in a regular hexagonal dartboard -/
theorem dart_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_area := Real.pi * s^2
  circle_area / hexagon_area = 2 * Real.pi / (3 * Real.sqrt 3) := by
  sorry

end dart_probability_l3456_345684


namespace santiagos_number_l3456_345618

theorem santiagos_number (amelia santiago : ℂ) : 
  amelia * santiago = 20 + 15 * Complex.I ∧ 
  amelia = 4 - 5 * Complex.I →
  santiago = (5 : ℚ) / 41 + (160 : ℚ) / 41 * Complex.I :=
by
  sorry

end santiagos_number_l3456_345618


namespace intersection_of_A_and_B_l3456_345630

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end intersection_of_A_and_B_l3456_345630


namespace max_prob_div_by_10_min_nonzero_prob_div_by_10_l3456_345647

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ
  h : length > 0

/-- The probability of a number in the segment being divisible by 10 -/
def prob_div_by_10 (s : Segment) : ℚ :=
  (s.length.divisors.filter (· % 10 = 0)).card / s.length

/-- The maximum probability of a number in any segment being divisible by 10 is 1 -/
theorem max_prob_div_by_10 : ∃ s : Segment, prob_div_by_10 s = 1 :=
  sorry

/-- The minimum non-zero probability of a number in any segment being divisible by 10 is 1/19 -/
theorem min_nonzero_prob_div_by_10 : 
  ∀ s : Segment, prob_div_by_10 s ≠ 0 → prob_div_by_10 s ≥ 1/19 :=
  sorry

end max_prob_div_by_10_min_nonzero_prob_div_by_10_l3456_345647


namespace large_balls_can_make_l3456_345623

/-- The number of rubber bands in a small ball -/
def small_ball_bands : ℕ := 50

/-- The number of rubber bands in a large ball -/
def large_ball_bands : ℕ := 300

/-- The total number of rubber bands Michael brought to class -/
def total_bands : ℕ := 5000

/-- The number of small balls Michael has already made -/
def small_balls_made : ℕ := 22

/-- The number of large balls Michael can make with the remaining rubber bands -/
theorem large_balls_can_make : ℕ := by
  sorry

end large_balls_can_make_l3456_345623


namespace motorcyclists_speeds_l3456_345680

/-- The length of the circular track in meters -/
def track_length : ℝ := 1000

/-- The time interval between overtakes in minutes -/
def overtake_interval : ℝ := 2

/-- The initial speed of motorcyclist A in meters per minute -/
def speed_A : ℝ := 1000

/-- The initial speed of motorcyclist B in meters per minute -/
def speed_B : ℝ := 1500

/-- Theorem stating the conditions and the conclusion about the motorcyclists' speeds -/
theorem motorcyclists_speeds :
  (speed_B - speed_A) * overtake_interval = track_length ∧
  (2 * speed_A - speed_B) * overtake_interval = track_length →
  speed_A = 1000 ∧ speed_B = 1500 := by
  sorry

end motorcyclists_speeds_l3456_345680


namespace optimal_cup_purchase_l3456_345640

/-- Represents the profit optimization problem for cup sales --/
structure CupSalesProblem where
  costA : ℕ
  priceA : ℕ
  costB : ℕ
  priceB : ℕ
  totalCups : ℕ
  budget : ℕ

/-- Calculates the profit for a given number of cup A --/
def profit (p : CupSalesProblem) (x : ℕ) : ℤ :=
  (p.priceA - p.costA) * x + (p.priceB - p.costB) * (p.totalCups - x)

/-- Checks if the purchase is within budget --/
def withinBudget (p : CupSalesProblem) (x : ℕ) : Prop :=
  p.costA * x + p.costB * (p.totalCups - x) ≤ p.budget

/-- Theorem stating the optimal solution and maximum profit --/
theorem optimal_cup_purchase (p : CupSalesProblem) 
  (h1 : p.costA = 100)
  (h2 : p.priceA = 150)
  (h3 : p.costB = 85)
  (h4 : p.priceB = 120)
  (h5 : p.totalCups = 160)
  (h6 : p.budget = 15000) :
  ∃ (x : ℕ), x = 93 ∧ 
             withinBudget p x ∧ 
             profit p x = 6995 ∧ 
             ∀ (y : ℕ), withinBudget p y → profit p y ≤ profit p x :=
by sorry

end optimal_cup_purchase_l3456_345640


namespace maria_number_puzzle_l3456_345637

theorem maria_number_puzzle (x : ℝ) : 
  (((x + 3) * 2 - 4) / 3 = 10) → x = 14 := by
  sorry

end maria_number_puzzle_l3456_345637


namespace odd_function_sum_l3456_345603

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end odd_function_sum_l3456_345603


namespace roots_theorem_l3456_345698

theorem roots_theorem :
  (∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2) ∧
  (∃ x y : ℝ, x^2 = 9 ∧ y^2 = 9 ∧ x = 3 ∧ y = -3) ∧
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) :=
by sorry

end roots_theorem_l3456_345698


namespace tip_amount_is_24_l3456_345602

-- Define the cost of haircuts
def womens_haircut_cost : ℚ := 48
def childrens_haircut_cost : ℚ := 36

-- Define the number of each type of haircut
def num_womens_haircuts : ℕ := 1
def num_childrens_haircuts : ℕ := 2

-- Define the tip percentage
def tip_percentage : ℚ := 20 / 100

-- Theorem statement
theorem tip_amount_is_24 :
  let total_cost := womens_haircut_cost * num_womens_haircuts + childrens_haircut_cost * num_childrens_haircuts
  tip_percentage * total_cost = 24 := by
  sorry

end tip_amount_is_24_l3456_345602


namespace remainder_seven_divisors_of_sixtyone_l3456_345609

theorem remainder_seven_divisors_of_sixtyone : 
  (Finset.filter (fun n : ℕ => n > 7 ∧ 61 % n = 7) (Finset.range 62)).card = 4 := by
  sorry

end remainder_seven_divisors_of_sixtyone_l3456_345609


namespace factorization_of_polynomial_l3456_345674

theorem factorization_of_polynomial (b : ℝ) :
  348 * b^2 + 87 * b + 261 = 87 * (4 * b^2 + b + 3) := by
  sorry

end factorization_of_polynomial_l3456_345674


namespace water_storage_calculation_l3456_345633

/-- Calculates the total volume of water stored in jars of different sizes -/
theorem water_storage_calculation (total_jars : ℕ) (h1 : total_jars = 24) :
  let jars_per_size := total_jars / 3
  let quart_volume := jars_per_size * (1 / 4 : ℚ)
  let half_gallon_volume := jars_per_size * (1 / 2 : ℚ)
  let gallon_volume := jars_per_size * 1
  quart_volume + half_gallon_volume + gallon_volume = 14 := by
  sorry

#check water_storage_calculation

end water_storage_calculation_l3456_345633


namespace ellipse_condition_l3456_345636

/-- 
A non-degenerate ellipse is represented by the equation 
3x^2 + 9y^2 - 12x + 27y = b if and only if b > -129/4
-/
theorem ellipse_condition (b : ℝ) : 
  (∃ (x y : ℝ), 3*x^2 + 9*y^2 - 12*x + 27*y = b ∧ 
    ∀ (x' y' : ℝ), 3*x'^2 + 9*y'^2 - 12*x' + 27*y' = b → (x', y') ≠ (x, y)) ↔ 
  b > -129/4 := by
sorry

end ellipse_condition_l3456_345636


namespace intersection_area_greater_than_half_l3456_345692

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the intersection of two rectangles -/
structure Intersection (r1 r2 : Rectangle) where
  area : ℝ

/-- Theorem: Given two equal rectangles whose contours intersect at 8 points,
    the area of their intersection is greater than half the area of each rectangle -/
theorem intersection_area_greater_than_half 
  (r1 r2 : Rectangle) 
  (h_equal : r1 = r2) 
  (h_intersect : ∃ (pts : Finset (ℝ × ℝ)), pts.card = 8) 
  (i : Intersection r1 r2) : 
  i.area > (1/2) * r1.area := by
  sorry

end intersection_area_greater_than_half_l3456_345692
