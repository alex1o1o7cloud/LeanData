import Mathlib

namespace pizza_consumption_l3376_337660

theorem pizza_consumption (n : ℕ) (first_trip : ℚ) (subsequent_trips : ℚ) : 
  n = 6 → 
  first_trip = 2/3 → 
  subsequent_trips = 1/2 → 
  (1 - (1 - first_trip) * subsequent_trips^(n-1) : ℚ) = 191/192 := by
  sorry

end pizza_consumption_l3376_337660


namespace farm_entrance_fee_for_students_l3376_337662

theorem farm_entrance_fee_for_students :
  let num_students : ℕ := 35
  let num_adults : ℕ := 4
  let adult_fee : ℚ := 6
  let total_cost : ℚ := 199
  let student_fee : ℚ := (total_cost - num_adults * adult_fee) / num_students
  student_fee = 5 := by sorry

end farm_entrance_fee_for_students_l3376_337662


namespace circle_y_axis_intersection_sum_l3376_337664

theorem circle_y_axis_intersection_sum (h k r : ℝ) : 
  h = 5 → k = -3 → r = 13 →
  let y₁ := k + (r^2 - h^2).sqrt
  let y₂ := k - (r^2 - h^2).sqrt
  y₁ + y₂ = -6 :=
by sorry

end circle_y_axis_intersection_sum_l3376_337664


namespace translation_theorem_l3376_337630

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_theorem :
  let A : Point := { x := -2, y := 3 }
  let A' : Point := translate (translate A 0 (-3)) 4 0
  A'.x = 2 ∧ A'.y = 0 := by
  sorry

end translation_theorem_l3376_337630


namespace equation_solutions_l3376_337687

theorem equation_solutions :
  (∀ x : ℝ, 4 * (2 * x - 1)^2 = 36 ↔ x = 2 ∨ x = -1) ∧
  (∀ x : ℝ, (1/4) * (2 * x + 3)^3 - 54 = 0 ↔ x = 3/2) :=
by sorry

end equation_solutions_l3376_337687


namespace terminal_side_equivalence_l3376_337628

/-- Two angles have the same terminal side if their difference is an integer multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = 360 * k

/-- Prove that -330° has the same terminal side as 30° -/
theorem terminal_side_equivalence : same_terminal_side (-330) 30 := by
  sorry

end terminal_side_equivalence_l3376_337628


namespace cone_prism_volume_ratio_l3376_337678

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism. -/
theorem cone_prism_volume_ratio
  (a b h_c h_p : ℝ)
  (h_ab : b < a)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_pos_h_c : h_c > 0)
  (h_pos_h_p : h_p > 0) :
  (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) = π * b * h_c / (12 * a * h_p) :=
by sorry

end cone_prism_volume_ratio_l3376_337678


namespace max_value_xy_over_x2_plus_y2_l3376_337646

theorem max_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 2/7 ≤ y ∧ y ≤ 1/2) :
  x * y / (x^2 + y^2) ≤ 2/5 := by
  sorry

end max_value_xy_over_x2_plus_y2_l3376_337646


namespace integer_partition_impossibility_l3376_337634

theorem integer_partition_impossibility : 
  ¬ (∃ (A B C : Set ℤ), 
    (∀ (n : ℤ), (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
                (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
                (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
                (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
                (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
                (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)) ∧
    (A ∪ B ∪ C = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅)) :=
by sorry

end integer_partition_impossibility_l3376_337634


namespace fraction_invariance_l3376_337612

theorem fraction_invariance (x y : ℝ) :
  (2 * x + y) / (3 * x + y) = (2 * (10 * x) + 10 * y) / (3 * (10 * x) + 10 * y) :=
by sorry

end fraction_invariance_l3376_337612


namespace smallest_b_for_factorization_l3376_337674

theorem smallest_b_for_factorization (b : ℕ) : b = 121 ↔ 
  (b > 0 ∧ 
   ∃ (r s : ℕ), r * s = 2020 ∧ r > s ∧
   ∀ (x : ℤ), x^2 + b*x + 2020 = (x + r) * (x + s) ∧
   ∀ (b' : ℕ), b' > 0 → 
     (∃ (r' s' : ℕ), r' * s' = 2020 ∧ r' > s' ∧
     ∀ (x : ℤ), x^2 + b'*x + 2020 = (x + r') * (x + s')) →
     b ≤ b') := by
sorry

end smallest_b_for_factorization_l3376_337674


namespace rug_purchase_price_l3376_337638

/-- Proves that the purchase price per rug is $40 given the selling price, number of rugs, and total profit -/
theorem rug_purchase_price
  (selling_price : ℝ)
  (num_rugs : ℕ)
  (total_profit : ℝ)
  (h1 : selling_price = 60)
  (h2 : num_rugs = 20)
  (h3 : total_profit = 400) :
  (selling_price * num_rugs - total_profit) / num_rugs = 40 := by
  sorry

end rug_purchase_price_l3376_337638


namespace lcm_48_180_l3376_337601

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l3376_337601


namespace value_of_a_l3376_337656

-- Define the conversion rate from paise to rupees
def paiseToRupees (paise : ℚ) : ℚ := paise / 100

-- Define the problem statement
theorem value_of_a (a : ℚ) (h : (0.5 / 100) * a = paiseToRupees 70) : a = 140 := by
  sorry

end value_of_a_l3376_337656


namespace rationalize_denominator_l3376_337693

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
     (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F) ∧
    A = -13 ∧ B = -9 ∧ C = 3 ∧ D = 2 ∧ E = 165 ∧ F = 51 := by
  sorry

end rationalize_denominator_l3376_337693


namespace range_of_a_for_quadratic_function_l3376_337667

theorem range_of_a_for_quadratic_function (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 := by
sorry

end range_of_a_for_quadratic_function_l3376_337667


namespace interview_segment_ratio_l3376_337690

/-- Represents the lengths of three interview segments in a radio show. -/
structure InterviewSegments where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the ratio of the third segment to the second segment is 1:2
    given the conditions of the radio show. -/
theorem interview_segment_ratio
  (segments : InterviewSegments)
  (total_time : segments.first + segments.second + segments.third = 90)
  (first_twice_others : segments.first = 2 * (segments.second + segments.third))
  (third_length : segments.third = 10) :
  segments.third / segments.second = 1 / 2 := by
  sorry

end interview_segment_ratio_l3376_337690


namespace dormitory_to_city_distance_l3376_337670

theorem dormitory_to_city_distance : ∃ (d : ℝ), 
  (1/5 : ℝ) * d + (2/3 : ℝ) * d + 14 = d ∧ d = 105 := by
  sorry

end dormitory_to_city_distance_l3376_337670


namespace cube_root_function_l3376_337647

theorem cube_root_function (k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = k * (x ^ (1/3))) →
  y 64 = 4 →
  y 8 = 2 :=
by
  sorry

end cube_root_function_l3376_337647


namespace simplify_expression_l3376_337621

theorem simplify_expression : (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by sorry

end simplify_expression_l3376_337621


namespace add_multiply_round_problem_l3376_337671

theorem add_multiply_round_problem : 
  let a := 73.5891
  let b := 24.376
  let sum := a + b
  let product := sum * 2
  (product * 100).round / 100 = 195.93 := by sorry

end add_multiply_round_problem_l3376_337671


namespace banana_orange_equivalence_l3376_337655

/-- Given that 2/3 of 15 bananas are worth 12 oranges,
    prove that 1/4 of 20 bananas are worth 6 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (2 / 3 : ℚ) * 15 * banana_value = 12 * orange_value →
  (1 / 4 : ℚ) * 20 * banana_value = 6 * orange_value :=
by sorry

end banana_orange_equivalence_l3376_337655


namespace contrapositive_inequality_l3376_337625

theorem contrapositive_inequality (a b c : ℝ) :
  (¬(a < b) → ¬(a + c < b + c)) ↔ (a + c ≥ b + c → a ≥ b) := by sorry

end contrapositive_inequality_l3376_337625


namespace sum_remainder_l3376_337663

theorem sum_remainder (x y z : ℕ+) 
  (hx : x ≡ 30 [ZMOD 59])
  (hy : y ≡ 27 [ZMOD 59])
  (hz : z ≡ 4 [ZMOD 59]) :
  (x + y + z : ℤ) ≡ 2 [ZMOD 59] := by
  sorry

end sum_remainder_l3376_337663


namespace inequality_proof_l3376_337668

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end inequality_proof_l3376_337668


namespace circle_arrangement_divisible_by_three_l3376_337608

/-- A type representing the arrangement of numbers in a circle. -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Predicate to check if two numbers differ by 1, 2, or factor of two. -/
def ValidDifference (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

/-- Theorem stating that in any arrangement of 99 natural numbers in a circle
    where any two neighboring numbers differ either by 1, or by 2,
    or by a factor of two, at least one of these numbers is divisible by 3. -/
theorem circle_arrangement_divisible_by_three
  (arr : CircularArrangement 99)
  (h : ∀ i : Fin 99, ValidDifference (arr i) (arr (i + 1))) :
  ∃ i : Fin 99, 3 ∣ arr i :=
sorry

end circle_arrangement_divisible_by_three_l3376_337608


namespace lucas_units_digit_l3376_337627

-- Define Lucas numbers
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem lucas_units_digit :
  unitsDigit (lucas (lucas 15)) = 7 := by sorry

end lucas_units_digit_l3376_337627


namespace reciprocal_roots_sum_l3376_337698

theorem reciprocal_roots_sum (α β : ℝ) : 
  (∃ a b : ℝ, (7 * a^2 + 2 * a + 6 = 0) ∧ 
              (7 * b^2 + 2 * b + 6 = 0) ∧ 
              (α = 1 / a) ∧ 
              (β = 1 / b)) →
  α + β = -1/3 := by
sorry

end reciprocal_roots_sum_l3376_337698


namespace cone_circumscribed_sphere_surface_area_l3376_337685

/-- Given a cone with base area π and lateral area twice the base area, 
    the surface area of its circumscribed sphere is 16π/3 -/
theorem cone_circumscribed_sphere_surface_area 
  (base_area : ℝ) 
  (lateral_area : ℝ) 
  (h1 : base_area = π) 
  (h2 : lateral_area = 2 * base_area) : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    4 * π * r^2 = 16 * π / 3 := by
sorry

end cone_circumscribed_sphere_surface_area_l3376_337685


namespace three_card_selection_count_l3376_337691

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- 
  Theorem: The number of ways to select three different cards in sequence 
  from a standard deck is 132600.
-/
theorem three_card_selection_count : 
  standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2) = 132600 := by
  sorry


end three_card_selection_count_l3376_337691


namespace total_flowers_and_stems_l3376_337624

def roses : ℕ := 12
def carnations : ℕ := 15
def lilies : ℕ := 10
def tulips : ℕ := 8
def daisies : ℕ := 5
def orchids : ℕ := 3
def babys_breath : ℕ := 10

theorem total_flowers_and_stems :
  roses + carnations + lilies + tulips + daisies + orchids + babys_breath = 63 := by
  sorry

end total_flowers_and_stems_l3376_337624


namespace line_intercepts_sum_and_product_l3376_337672

/-- Given a line with equation y - 2 = -3(x + 5), prove that the sum of its
    x-intercept and y-intercept is -52/3, and their product is 169/3. -/
theorem line_intercepts_sum_and_product :
  let f : ℝ → ℝ := λ x => -3 * (x + 5) + 2
  let x_intercept := -13 / 3
  let y_intercept := f 0
  (x_intercept + y_intercept = -52 / 3) ∧ (x_intercept * y_intercept = 169 / 3) := by
sorry

end line_intercepts_sum_and_product_l3376_337672


namespace hyperbola_focal_length_l3376_337619

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the focal length of the hyperbola is 2√5 -/
theorem hyperbola_focal_length 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (h_distance : p/2 + a = 4) 
  (h_intersection : -1 = -2*b/a ∧ -2 = -p/2) : 
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5 := by
sorry

end hyperbola_focal_length_l3376_337619


namespace pepper_remaining_l3376_337606

theorem pepper_remaining (initial : Real) (used : Real) (remaining : Real) : 
  initial = 0.25 → used = 0.16 → remaining = initial - used → remaining = 0.09 := by
  sorry

end pepper_remaining_l3376_337606


namespace lg_calculation_l3376_337643

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_calculation : lg 25 - 2 * lg (1/2) = 2 := by
  sorry

end lg_calculation_l3376_337643


namespace band_sections_fraction_l3376_337603

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℝ) 
  (h1 : trumpet_fraction = 0.5)
  (h2 : trombone_fraction = 0.12) :
  trumpet_fraction + trombone_fraction = 0.62 := by
  sorry

end band_sections_fraction_l3376_337603


namespace determinant_equation_solution_l3376_337629

-- Define the determinant operation
def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem determinant_equation_solution :
  ∃ (x : ℝ), determinant (x + 1) x (2*x - 6) (2*(x - 1)) = 10 ∧ x = 2 :=
by
  sorry

end determinant_equation_solution_l3376_337629


namespace dot_product_bounds_l3376_337632

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + (P.2^2 / 3) = 1

-- Define the circle
def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 + 1)^2 + Q.2^2 = 1

-- Define a tangent line from a point to the circle
def is_tangent (P A : ℝ × ℝ) : Prop :=
  is_on_circle A ∧ ((P.1 - A.1) * (A.1 + 1) + (P.2 - A.2) * A.2 = 0)

-- Define the dot product of two vectors
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

-- The main theorem
theorem dot_product_bounds (P A B : ℝ × ℝ) :
  is_on_ellipse P → is_tangent P A → is_tangent P B →
  2 * Real.sqrt 2 - 3 ≤ dot_product P A B ∧ dot_product P A B ≤ 56 / 9 := by
  sorry

end dot_product_bounds_l3376_337632


namespace peaches_per_basket_proof_l3376_337602

/-- The number of peaches in each basket originally -/
def peaches_per_basket : ℕ := 25

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The number of peaches eaten by farmers -/
def eaten_peaches : ℕ := 5

/-- The number of peaches in each small box after packing -/
def peaches_per_box : ℕ := 15

/-- The number of small boxes after packing -/
def num_boxes : ℕ := 8

theorem peaches_per_basket_proof :
  peaches_per_basket * num_baskets = 
    num_boxes * peaches_per_box + eaten_peaches :=
by sorry

end peaches_per_basket_proof_l3376_337602


namespace maria_test_scores_l3376_337659

def test_scores : List ℤ := [94, 92, 91, 75, 68]

theorem maria_test_scores :
  let scores := test_scores
  (scores.length = 5) ∧
  (scores.take 3 = [91, 75, 68]) ∧
  (scores.sum / scores.length = 84) ∧
  (∀ s ∈ scores, s < 95) ∧
  (∀ s ∈ scores, s ≥ 65) ∧
  scores.Nodup ∧
  scores.Sorted (· ≥ ·) :=
by sorry

end maria_test_scores_l3376_337659


namespace expression_evaluation_l3376_337653

theorem expression_evaluation (a b : ℚ) (h1 : a = -3) (h2 : b = 1/3) :
  (a - 3*b) * (a + 3*b) + (a - 3*b)^2 = 24 := by sorry

end expression_evaluation_l3376_337653


namespace fraction_equality_l3376_337633

theorem fraction_equality (a b c : ℝ) 
  (h1 : a / b = 20) 
  (h2 : b / c = 10) : 
  (a + b) / (b + c) = 210 / 11 := by
  sorry

end fraction_equality_l3376_337633


namespace fraction_inequality_l3376_337631

theorem fraction_inequality (x : ℝ) (h : x ≠ 2) :
  (x + 1) / (x - 2) ≥ 0 ↔ x ≤ -1 ∨ x > 2 := by
  sorry

end fraction_inequality_l3376_337631


namespace complex_fraction_calculation_l3376_337652

theorem complex_fraction_calculation : 
  ((5 / 8 : ℚ) * (3 / 7) - (2 / 3) * (1 / 4)) * ((7 / 9 : ℚ) * (2 / 5) * (1 / 2) * 5040) = 79 := by
  sorry

end complex_fraction_calculation_l3376_337652


namespace cake_volume_and_icing_sum_l3376_337684

/-- Represents a point in 3D space -/
structure Point3D where
  x : Real
  y : Real
  z : Real

/-- Represents a triangular piece of cake -/
structure CakePiece where
  corner : Point3D
  midpoint1 : Point3D
  midpoint2 : Point3D

/-- Calculates the volume of the triangular cake piece -/
def volume (piece : CakePiece) : Real :=
  sorry

/-- Calculates the area of icing on the triangular cake piece -/
def icingArea (piece : CakePiece) : Real :=
  sorry

/-- The main theorem to prove -/
theorem cake_volume_and_icing_sum (cubeEdgeLength : Real) (piece : CakePiece) : 
  cubeEdgeLength = 3 →
  piece.corner = ⟨0, 0, 0⟩ →
  piece.midpoint1 = ⟨3, 3, 1.5⟩ →
  piece.midpoint2 = ⟨1.5, 3, 3⟩ →
  volume piece + icingArea piece = 24 :=
sorry

end cake_volume_and_icing_sum_l3376_337684


namespace impossibility_of_distinct_differences_l3376_337614

theorem impossibility_of_distinct_differences : ¬ ∃ (a : Fin 2010 → Fin 2010),
  Function.Injective a ∧ 
  (∀ (i j : Fin 2010), i ≠ j → |a i - i| ≠ |a j - j|) :=
by sorry

end impossibility_of_distinct_differences_l3376_337614


namespace existence_of_integer_roots_l3376_337666

theorem existence_of_integer_roots : ∃ (a b c d e f : ℤ),
  (∀ x : ℤ, (x + a) * (x^2 + b*x + c) * (x^3 + d*x^2 + e*x + f) = 0 ↔ 
    x = a ∨ x^2 + b*x + c = 0 ∨ x^3 + d*x^2 + e*x + f = 0) ∧
  (∃! (r₁ r₂ r₃ r₄ r₅ r₆ : ℤ), 
    {r₁, r₂, r₃, r₄, r₅, r₆} = {a} ∪ 
      {x : ℤ | x^2 + b*x + c = 0} ∪ 
      {x : ℤ | x^3 + d*x^2 + e*x + f = 0}) :=
sorry

end existence_of_integer_roots_l3376_337666


namespace polygon_count_l3376_337648

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of distinct convex polygons with 3 or more sides 
    that can be drawn using some or all of n points marked on a circle as vertices -/
def num_polygons (n : ℕ) : ℕ := 2^n - (n.choose 0 + n.choose 1 + n.choose 2)

theorem polygon_count : num_polygons n = 4017 := by
  sorry

end polygon_count_l3376_337648


namespace pressure_area_relation_l3376_337689

/-- Proves that given pressure P = F/S, force F = 50N, and P > 500Pa, the area S < 0.1m² -/
theorem pressure_area_relation (F S P : ℝ) : 
  F = 50 → P = F / S → P > 500 → S < 0.1 := by
  sorry

end pressure_area_relation_l3376_337689


namespace f_properties_l3376_337607

noncomputable def f (x : ℝ) : ℝ := Real.log (x / (x^2 + 1))

theorem f_properties :
  (∀ x : ℝ, x > 0 → f x ≠ 0) ∧
  (∀ x : ℝ, 0 < x → x < 1 → ∀ y : ℝ, x < y → y < 1 → f x < f y) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > x → f x > f y) :=
sorry

end f_properties_l3376_337607


namespace right_triangle_longest_altitudes_sum_l3376_337615

theorem right_triangle_longest_altitudes_sum (a b c : ℝ) : 
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 → 
  (max a b + min a b) = 21 := by
  sorry

end right_triangle_longest_altitudes_sum_l3376_337615


namespace triangle_angle_identity_l3376_337661

theorem triangle_angle_identity (α β γ : Real) (h : α + β + γ = π) :
  2 * Real.sin α * Real.sin β * Real.cos γ = Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin γ ^ 2 := by
  sorry

end triangle_angle_identity_l3376_337661


namespace arc_angle_proof_l3376_337600

/-- Given a circle with radius 3 cm and an arc length of π/2 cm, 
    prove that the corresponding central angle is 30°. -/
theorem arc_angle_proof (r : ℝ) (l : ℝ) (θ : ℝ) : 
  r = 3 → l = π / 2 → θ = (l * 180) / (π * r) → θ = 30 := by
  sorry

end arc_angle_proof_l3376_337600


namespace divisibility_condition_l3376_337616

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem divisibility_condition (a b : ℕ) :
  (a^2 + b^2 + 1) % (a * b) = 0 ↔
  ((a = 1 ∧ b = 1) ∨ ∃ n : ℕ, n ≥ 1 ∧ a = fibonacci (2*n + 1) ∧ b = fibonacci (2*n - 1)) :=
sorry

end divisibility_condition_l3376_337616


namespace hyperbola_equation_l3376_337622

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (0, Real.sqrt 10) →
  F₂ = (0, -Real.sqrt 10) →
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * 
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2 →
  M.2^2 / 9 - M.1^2 = 1 := by
  sorry


end hyperbola_equation_l3376_337622


namespace red_car_speed_is_10_l3376_337618

/-- The speed of the black car in miles per hour -/
def black_car_speed : ℝ := 50

/-- The initial distance between the cars in miles -/
def initial_distance : ℝ := 20

/-- The time it takes for the black car to overtake the red car in hours -/
def overtake_time : ℝ := 0.5

/-- The speed of the red car in miles per hour -/
def red_car_speed : ℝ := 10

theorem red_car_speed_is_10 :
  red_car_speed = 10 :=
by sorry

end red_car_speed_is_10_l3376_337618


namespace equality_of_fractions_l3376_337657

theorem equality_of_fractions (x y z k : ℝ) :
  (9 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 15 / (z - y)) → k = 24 := by
  sorry

end equality_of_fractions_l3376_337657


namespace tangent_half_identities_l3376_337610

theorem tangent_half_identities (α : Real) (h : Real.tan α = 1/2) :
  ((4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2/3) ∧
  (Real.sin α ^ 2 - Real.sin (2 * α) = -3/5) := by
  sorry

end tangent_half_identities_l3376_337610


namespace bat_ball_cost_difference_l3376_337688

theorem bat_ball_cost_difference (bat_cost ball_cost : ℕ) : 
  (2 * bat_cost + 3 * ball_cost = 1300) →
  (3 * bat_cost + 2 * ball_cost = 1200) →
  (ball_cost - bat_cost = 100) := by
sorry

end bat_ball_cost_difference_l3376_337688


namespace min_value_is_nine_l3376_337682

/-- Two circles C₁ and C₂ with centers and radii -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0

/-- The circles have only one common tangent -/
axiom one_common_tangent (c : TwoCircles) : 4 * c.a^2 + c.b^2 = 1

/-- The minimum value of 1/a² + 1/b² is 9 -/
theorem min_value_is_nine (c : TwoCircles) : 
  ∀ ε > 0, (1 / c.a^2 + 1 / c.b^2) > 9 - ε :=
sorry

end min_value_is_nine_l3376_337682


namespace arthur_muffins_l3376_337611

theorem arthur_muffins (total : ℕ) (more : ℕ) (initial : ℕ) 
    (h1 : total = 83)
    (h2 : more = 48)
    (h3 : total = initial + more) :
  initial = 35 := by
  sorry

end arthur_muffins_l3376_337611


namespace soccer_team_combinations_l3376_337675

def soccer_team_size : ℕ := 16
def quadruplets_count : ℕ := 4
def starting_lineup_size : ℕ := 7
def max_quadruplets_in_lineup : ℕ := 2

theorem soccer_team_combinations :
  (Nat.choose (soccer_team_size - quadruplets_count) starting_lineup_size) +
  (quadruplets_count * Nat.choose (soccer_team_size - quadruplets_count) (starting_lineup_size - 1)) +
  (Nat.choose quadruplets_count 2 * Nat.choose (soccer_team_size - quadruplets_count) (starting_lineup_size - 2)) = 9240 := by
  sorry

end soccer_team_combinations_l3376_337675


namespace stating_prize_distribution_orders_l3376_337636

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := num_players - 1

/-- 
Theorem stating that the number of possible prize distribution orders
in a tournament with 6 players following the described elimination format is 32
-/
theorem prize_distribution_orders :
  (2 : ℕ) ^ num_games = 32 := by
  sorry

end stating_prize_distribution_orders_l3376_337636


namespace min_value_of_expression_l3376_337680

theorem min_value_of_expression (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_product : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end min_value_of_expression_l3376_337680


namespace closest_multiple_of_18_to_2021_l3376_337642

def closest_multiple (n m : ℕ) : ℕ :=
  let q := n / m
  let r := n % m
  if r ≤ m / 2 then q * m else (q + 1) * m

theorem closest_multiple_of_18_to_2021 :
  closest_multiple 2021 18 = 2016 :=
sorry

end closest_multiple_of_18_to_2021_l3376_337642


namespace thousand_ring_date_l3376_337654

/-- Represents a time with hour and minute components -/
structure Time where
  hour : Nat
  minute : Nat

/-- Represents a date with year, month, and day components -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Counts the number of bell rings from a given start time and date until the nth ring -/
def countBellRings (startTime : Time) (startDate : Date) (n : Nat) : Date :=
  sorry

/-- The bell ringing pattern: once at 45 minutes past each hour and according to the hour every hour -/
axiom bell_pattern : ∀ (t : Time), (t.minute = 45 ∧ t.hour ≠ 0) ∨ (t.minute = 0 ∧ t.hour ≠ 0)

/-- The starting time is 10:30 AM on January 1, 2021 -/
def startTime : Time := { hour := 10, minute := 30 }
def startDate : Date := { year := 2021, month := 1, day := 1 }

/-- The theorem to prove -/
theorem thousand_ring_date : 
  countBellRings startTime startDate 1000 = { year := 2021, month := 1, day := 11 } :=
sorry

end thousand_ring_date_l3376_337654


namespace quadratic_equation_solution_l3376_337623

theorem quadratic_equation_solution :
  ∃ x : ℝ, x^2 + 4*x + 3 = -(x + 3)*(x + 5) ∧ x = -3 :=
by
  sorry

end quadratic_equation_solution_l3376_337623


namespace system_no_solution_l3376_337665

def has_no_solution (a b c : ℤ) : Prop :=
  (a * b = 6) ∧ (b * c = 8) ∧ (c / 4 ≠ c / (4 * b))

theorem system_no_solution :
  ∀ a b c : ℤ, has_no_solution a b c ↔ 
    ((a = -6 ∧ b = -1 ∧ c = -8) ∨
     (a = -3 ∧ b = -2 ∧ c = -4) ∨
     (a = 3 ∧ b = 2 ∧ c = 4)) :=
by sorry

end system_no_solution_l3376_337665


namespace sara_quarters_l3376_337613

/-- The number of quarters Sara initially had -/
def initial_quarters : ℕ := 783

/-- The number of quarters Sara's dad borrowed -/
def borrowed_quarters : ℕ := 271

/-- The number of quarters Sara has now -/
def remaining_quarters : ℕ := initial_quarters - borrowed_quarters

theorem sara_quarters : remaining_quarters = 512 := by
  sorry

end sara_quarters_l3376_337613


namespace exam_marks_theorem_l3376_337673

theorem exam_marks_theorem (T : ℝ) 
  (h1 : 0.40 * T + 40 = 160) 
  (h2 : 0.60 * T - 160 = 20) : True :=
by sorry

end exam_marks_theorem_l3376_337673


namespace hyperbola_condition_l3376_337699

theorem hyperbola_condition (m : ℝ) :
  (m > 0 → m * (m + 2) > 0) ∧ ¬(m * (m + 2) > 0 → m > 0) :=
by sorry

end hyperbola_condition_l3376_337699


namespace total_animals_count_l3376_337650

/-- The total number of dangerous animals pointed out by the teacher in the swamp area -/
def total_dangerous_animals : ℕ := 250

/-- The number of crocodiles observed -/
def crocodiles : ℕ := 42

/-- The number of alligators observed -/
def alligators : ℕ := 35

/-- The number of vipers observed -/
def vipers : ℕ := 10

/-- The number of water moccasins observed -/
def water_moccasins : ℕ := 28

/-- The number of cottonmouth snakes observed -/
def cottonmouth_snakes : ℕ := 15

/-- The number of piranha fish in the school -/
def piranha_fish : ℕ := 120

/-- Theorem stating that the total number of dangerous animals is the sum of all observed species -/
theorem total_animals_count :
  total_dangerous_animals = crocodiles + alligators + vipers + water_moccasins + cottonmouth_snakes + piranha_fish :=
by
  sorry

end total_animals_count_l3376_337650


namespace characterization_of_complete_sets_l3376_337637

def is_complete (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

def complete_sets : Set (Set ℕ) :=
  {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, Set.univ}

theorem characterization_of_complete_sets :
  ∀ A : Set ℕ, A.Nonempty → (is_complete A ↔ A ∈ complete_sets) := by
  sorry

end characterization_of_complete_sets_l3376_337637


namespace sqrt_pattern_l3376_337645

-- Define the square root function
noncomputable def sqrt (x : ℝ) := Real.sqrt x

-- Define the approximation relation
def approximately_equal (x y : ℝ) := ∃ (ε : ℝ), ε > 0 ∧ |x - y| < ε

-- State the theorem
theorem sqrt_pattern :
  (sqrt 0.0625 = 0.25) →
  (approximately_equal (sqrt 0.625) 0.791) →
  (sqrt 625 = 25) →
  (sqrt 6250 = 79.1) →
  (sqrt 62500 = 250) →
  (sqrt 625000 = 791) →
  (sqrt 6.25 = 2.5) ∧ (approximately_equal (sqrt 62.5) 7.91) :=
by sorry

end sqrt_pattern_l3376_337645


namespace polar_to_cartesian_line_l3376_337692

/-- The polar equation r = 1 / (sin θ + cos θ) represents a line in Cartesian coordinates -/
theorem polar_to_cartesian_line :
  ∀ (θ : ℝ) (r : ℝ), r = 1 / (Real.sin θ + Real.cos θ) →
  ∃ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x + y = 1 :=
by sorry

end polar_to_cartesian_line_l3376_337692


namespace sequence_sum_l3376_337604

theorem sequence_sum (a b c d : ℕ) 
  (h1 : 0 < a ∧ a < b ∧ b < c ∧ c < d)
  (h2 : b * a = c * a)
  (h3 : c - b = d - c)
  (h4 : d - a = 36) : 
  a + b + c + d = 1188 := by
  sorry

end sequence_sum_l3376_337604


namespace arithmetic_sequence_product_l3376_337626

-- Define an arithmetic sequence of integers
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- Theorem statement
theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  is_arithmetic_sequence a →
  is_increasing_sequence a →
  a 4 * a 5 = 13 →
  a 3 * a 6 = -275 :=
by sorry

end arithmetic_sequence_product_l3376_337626


namespace compute_expression_l3376_337676

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 := by
  sorry

end compute_expression_l3376_337676


namespace quadratic_inequality_empty_solution_set_l3376_337697

/-- Given a quadratic equation ax² + bx + c = 0 with a > 0 and no real roots,
    the solution set of ax² + bx + c < 0 is empty. -/
theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ) 
  (h_a_pos : a > 0)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ :=
sorry

end quadratic_inequality_empty_solution_set_l3376_337697


namespace imaginary_part_of_complex_fraction_l3376_337635

theorem imaginary_part_of_complex_fraction : Complex.im ((2 + Complex.I) / (1 - 2 * Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l3376_337635


namespace fish_catch_total_l3376_337609

/-- The total number of fish caught by Leo, Agrey, and Sierra -/
def total_fish (leo agrey sierra : ℕ) : ℕ := leo + agrey + sierra

/-- Theorem stating the total number of fish caught given the conditions -/
theorem fish_catch_total :
  ∀ (leo agrey sierra : ℕ),
    leo = 40 →
    agrey = leo + 20 →
    sierra = agrey + 15 →
    total_fish leo agrey sierra = 175 := by
  sorry

end fish_catch_total_l3376_337609


namespace systematic_sample_fourth_element_l3376_337649

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is in the systematic sample --/
def inSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

/-- The theorem to be proved --/
theorem systematic_sample_fourth_element :
  ∀ s : SystematicSample,
    s.population_size = 48 →
    s.sample_size = 4 →
    inSample s 5 →
    inSample s 29 →
    inSample s 41 →
    inSample s 17 :=
by
  sorry

end systematic_sample_fourth_element_l3376_337649


namespace base3_10201_equals_100_l3376_337695

def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * 3^i) 0

theorem base3_10201_equals_100 :
  base3ToDecimal [1, 0, 2, 0, 1] = 100 := by
  sorry

end base3_10201_equals_100_l3376_337695


namespace tangent_line_to_circle_l3376_337620

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), x + 2*y = r ∧ x^2 + y^2 = 2*r^2) →
  (∀ (x y : ℝ), x + 2*y = r → x^2 + y^2 ≥ 2*r^2) →
  r = Real.sqrt 10 := by
sorry

end tangent_line_to_circle_l3376_337620


namespace prize_interval_is_1000_l3376_337651

/-- Represents the prize structure of an international competition --/
structure PrizeStructure where
  totalPrize : ℕ
  firstPrize : ℕ
  numPositions : ℕ
  hasPrizeInterval : Bool

/-- Calculates the interval between prizes --/
def calculatePrizeInterval (ps : PrizeStructure) : ℕ :=
  sorry

/-- Theorem stating that the prize interval is 1000 given the specific conditions --/
theorem prize_interval_is_1000 (ps : PrizeStructure) 
  (h1 : ps.totalPrize = 15000)
  (h2 : ps.firstPrize = 5000)
  (h3 : ps.numPositions = 5)
  (h4 : ps.hasPrizeInterval = true) : 
  calculatePrizeInterval ps = 1000 := by
  sorry

end prize_interval_is_1000_l3376_337651


namespace jonathan_weekly_deficit_l3376_337639

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Jonathan's daily calorie intake -/
def calorie_intake (d : Day) : ℕ :=
  match d with
  | Day.Monday => 2500
  | Day.Tuesday => 2600
  | Day.Wednesday => 2400
  | Day.Thursday => 2700
  | Day.Friday => 2300
  | Day.Saturday => 3500
  | Day.Sunday => 2400

/-- Jonathan's daily calorie expenditure -/
def calorie_expenditure (d : Day) : ℕ :=
  match d with
  | Day.Monday => 3000
  | Day.Tuesday => 3200
  | Day.Wednesday => 2900
  | Day.Thursday => 3100
  | Day.Friday => 2800
  | Day.Saturday => 3000
  | Day.Sunday => 2700

/-- Calculate the daily caloric deficit -/
def daily_deficit (d : Day) : ℤ :=
  (calorie_expenditure d : ℤ) - (calorie_intake d : ℤ)

/-- The weekly caloric deficit -/
def weekly_deficit : ℤ :=
  (daily_deficit Day.Monday) +
  (daily_deficit Day.Tuesday) +
  (daily_deficit Day.Wednesday) +
  (daily_deficit Day.Thursday) +
  (daily_deficit Day.Friday) +
  (daily_deficit Day.Saturday) +
  (daily_deficit Day.Sunday)

/-- Theorem: Jonathan's weekly caloric deficit is 2800 calories -/
theorem jonathan_weekly_deficit : weekly_deficit = 2800 := by
  sorry

end jonathan_weekly_deficit_l3376_337639


namespace cans_of_frosting_needed_l3376_337669

/-- The number of cans of frosting Bob needs to frost the remaining cakes -/
theorem cans_of_frosting_needed (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (cans_per_cake : ℕ) : 
  cakes_per_day = 10 → days = 5 → cakes_eaten = 12 → cans_per_cake = 2 →
  (cakes_per_day * days - cakes_eaten) * cans_per_cake = 76 := by sorry

end cans_of_frosting_needed_l3376_337669


namespace two_a_plus_a_equals_three_a_l3376_337686

theorem two_a_plus_a_equals_three_a (a : ℝ) : 2 * a + a = 3 * a := by
  sorry

end two_a_plus_a_equals_three_a_l3376_337686


namespace system_solution_and_M_minimum_l3376_337605

-- Define the system of equations
def system (x y t : ℝ) : Prop :=
  x - 3*y = 4 - t ∧ x + y = 3*t

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  -3 ≤ t ∧ t ≤ 1

-- Define M
def M (x y t : ℝ) : ℝ :=
  2*x - y - t

theorem system_solution_and_M_minimum :
  (∃ t, t_range t ∧ system 1 (-1) t) ∧
  (∀ x y t, t_range t → system x y t → M x y t ≥ -3) ∧
  (∃ x y t, t_range t ∧ system x y t ∧ M x y t = -3) :=
sorry

end system_solution_and_M_minimum_l3376_337605


namespace rectangle_fourth_vertex_l3376_337617

-- Define a structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a structure for a rectangle
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the theorem
theorem rectangle_fourth_vertex 
  (ABCD : Rectangle)
  (h1 : ABCD.A = ⟨0, 1⟩)
  (h2 : ABCD.B = ⟨1, 0⟩)
  (h3 : ABCD.C = ⟨3, 2⟩)
  : ABCD.D = ⟨2, 3⟩ := by
  sorry

end rectangle_fourth_vertex_l3376_337617


namespace fraction_factorization_l3376_337696

theorem fraction_factorization (a b c : ℝ) : 
  ((a^3 - b^3)^4 + (b^3 - c^3)^4 + (c^3 - a^3)^4) / ((a - b)^4 + (b - c)^4 + (c - a)^4)
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end fraction_factorization_l3376_337696


namespace painter_rooms_problem_l3376_337644

theorem painter_rooms_problem (time_per_room : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) :
  time_per_room = 8 →
  rooms_painted = 8 →
  time_remaining = 16 →
  rooms_painted + (time_remaining / time_per_room) = 10 :=
by sorry

end painter_rooms_problem_l3376_337644


namespace product_remainder_l3376_337679

theorem product_remainder (k : ℕ) : ∃ n : ℕ, 
  n = 5 * k + 1 ∧ (14452 * 15652 * n) % 5 = 4 := by
  sorry

end product_remainder_l3376_337679


namespace student_signup_combinations_l3376_337641

theorem student_signup_combinations :
  let num_students : ℕ := 3
  let num_groups : ℕ := 4
  num_groups ^ num_students = 64 :=
by sorry

end student_signup_combinations_l3376_337641


namespace josh_marbles_after_gift_l3376_337658

/-- The number of marbles Josh has after receiving marbles from Jack -/
theorem josh_marbles_after_gift (original : ℝ) (gift : ℝ) (total : ℝ)
  (h1 : original = 22.5)
  (h2 : gift = 20.75)
  (h3 : total = original + gift) :
  total = 43.25 := by
  sorry

end josh_marbles_after_gift_l3376_337658


namespace fifth_term_of_arithmetic_sequence_l3376_337683

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 2)
  (h3 : d = 1) : 
  a 5 = 6 := by
sorry

end fifth_term_of_arithmetic_sequence_l3376_337683


namespace sum_of_extreme_prime_factors_of_2730_l3376_337694

theorem sum_of_extreme_prime_factors_of_2730 : 
  ∃ (smallest largest : ℕ), 
    smallest.Prime ∧ 
    largest.Prime ∧ 
    smallest ∣ 2730 ∧ 
    largest ∣ 2730 ∧ 
    (∀ p : ℕ, p.Prime → p ∣ 2730 → p ≥ smallest ∧ p ≤ largest) ∧ 
    smallest + largest = 15 := by
  sorry

end sum_of_extreme_prime_factors_of_2730_l3376_337694


namespace john_payment_l3376_337681

def lawyer_fee (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (prep_time_multiplier : ℕ) : ℕ :=
  upfront_fee + hourly_rate * (court_hours + prep_time_multiplier * court_hours)

theorem john_payment (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (prep_time_multiplier : ℕ) :
  upfront_fee = 1000 →
  hourly_rate = 100 →
  court_hours = 50 →
  prep_time_multiplier = 2 →
  lawyer_fee upfront_fee hourly_rate court_hours prep_time_multiplier / 2 = 8000 :=
by
  sorry

#check john_payment

end john_payment_l3376_337681


namespace cubic_quartic_system_solution_l3376_337640

theorem cubic_quartic_system_solution (x y : ℝ) 
  (h1 : x^3 + y^3 = 1) 
  (h2 : x^4 + y^4 = 1) : 
  (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end cubic_quartic_system_solution_l3376_337640


namespace arccos_one_half_l3376_337677

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by sorry

end arccos_one_half_l3376_337677
