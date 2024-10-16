import Mathlib

namespace NUMINAMATH_CALUDE_max_stickers_for_player_l1985_198522

theorem max_stickers_for_player (num_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) :
  num_players = 25 →
  avg_stickers = 4 →
  min_stickers = 1 →
  ∃ (max_stickers : ℕ), max_stickers = 76 ∧
    ∀ (player_stickers : ℕ),
      (player_stickers * num_players ≤ num_players * avg_stickers) ∧
      (∀ (i : ℕ), i < num_players → min_stickers ≤ player_stickers) →
      player_stickers ≤ max_stickers :=
by sorry

end NUMINAMATH_CALUDE_max_stickers_for_player_l1985_198522


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l1985_198593

theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ∃ (r s : Prop), (r ∨ s) ∧ ¬(r ∧ s) := by
  sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l1985_198593


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l1985_198504

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -4) →
  (translate_right (reflect_x center) 5) = (8, 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l1985_198504


namespace NUMINAMATH_CALUDE_digit_complex_count_l1985_198552

/-- The set of digits from 0 to 9 -/
def Digits : Finset ℕ := Finset.range 10

/-- The set of non-zero digits from 1 to 9 -/
def NonZeroDigits : Finset ℕ := Finset.range 9 \ {0}

/-- A complex number formed by digits and i -/
structure DigitComplex where
  real : Digits
  imag : NonZeroDigits

/-- The number of complex numbers that can be formed using digits and i -/
def numDigitComplex : ℕ := Finset.card (Finset.product Digits NonZeroDigits)

theorem digit_complex_count : numDigitComplex = 90 := by sorry

end NUMINAMATH_CALUDE_digit_complex_count_l1985_198552


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1985_198507

theorem modulus_of_complex_fraction :
  let z : ℂ := (-3 + I) / (2 + I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1985_198507


namespace NUMINAMATH_CALUDE_combined_age_when_mike_is_24_l1985_198584

/-- Calculates the combined age of Mike, Barbara, Tom, and Peter when Mike is 24 years old -/
def combinedAgeWhenMikeIs24 (mikesInitialAge barbarasInitialAge tomsInitialAge petersInitialAge : ℕ) : ℕ :=
  let ageIncrease := 24 - mikesInitialAge
  24 + (barbarasInitialAge + ageIncrease) + (tomsInitialAge + ageIncrease) + (petersInitialAge + ageIncrease)

/-- Theorem stating the combined age of Mike, Barbara, Tom, and Peter when Mike is 24 years old -/
theorem combined_age_when_mike_is_24 :
  ∀ (mikesInitialAge : ℕ),
    mikesInitialAge = 16 →
    ∀ (barbarasInitialAge : ℕ),
      barbarasInitialAge = mikesInitialAge / 2 →
      ∀ (tomsInitialAge : ℕ),
        tomsInitialAge = barbarasInitialAge + 4 →
        ∀ (petersInitialAge : ℕ),
          petersInitialAge = 2 * tomsInitialAge →
          combinedAgeWhenMikeIs24 mikesInitialAge barbarasInitialAge tomsInitialAge petersInitialAge = 92 :=
by
  sorry


end NUMINAMATH_CALUDE_combined_age_when_mike_is_24_l1985_198584


namespace NUMINAMATH_CALUDE_hexagonal_field_fencing_cost_l1985_198547

/-- Represents the cost of fencing for a single side of the hexagonal field -/
structure SideCost where
  length : ℝ
  costPerMeter : ℝ

/-- Calculates the total cost of fencing for an irregular hexagonal field -/
def totalFencingCost (sides : List SideCost) : ℝ :=
  sides.foldl (fun acc side => acc + side.length * side.costPerMeter) 0

/-- Theorem stating that the total cost of fencing for the given hexagonal field is 289 rs. -/
theorem hexagonal_field_fencing_cost :
  let sides : List SideCost := [
    ⟨10, 3⟩, ⟨20, 2⟩, ⟨15, 4⟩, ⟨18, 3.5⟩, ⟨12, 2.5⟩, ⟨22, 3⟩
  ]
  totalFencingCost sides = 289 := by
  sorry


end NUMINAMATH_CALUDE_hexagonal_field_fencing_cost_l1985_198547


namespace NUMINAMATH_CALUDE_solve_equation_l1985_198559

theorem solve_equation (x : ℝ) (h : 9 - 4/x = 7 + 8/x) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1985_198559


namespace NUMINAMATH_CALUDE_cody_final_amount_l1985_198583

/-- Given an initial amount, a gift amount, and an expense amount, 
    calculate the final amount of money. -/
def finalAmount (initial gift expense : ℕ) : ℕ :=
  initial + gift - expense

/-- Theorem stating that given the specific values from the problem,
    the final amount is 35 dollars. -/
theorem cody_final_amount : 
  finalAmount 45 9 19 = 35 := by sorry

end NUMINAMATH_CALUDE_cody_final_amount_l1985_198583


namespace NUMINAMATH_CALUDE_circles_externally_separate_l1985_198516

theorem circles_externally_separate (m n : ℝ) : 
  2 > 0 ∧ m > 0 ∧ 
  (2 : ℝ)^2 - 10*2 + n = 0 ∧ 
  m^2 - 10*m + n = 0 → 
  n > 2 + m :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_separate_l1985_198516


namespace NUMINAMATH_CALUDE_min_five_dollar_frisbees_l1985_198540

/-- Given a total of 115 frisbees sold for $450, with prices of $3, $4, and $5,
    the minimum number of $5 frisbees sold is 1. -/
theorem min_five_dollar_frisbees :
  ∀ (x y z : ℕ),
    x + y + z = 115 →
    3 * x + 4 * y + 5 * z = 450 →
    z ≥ 1 ∧
    ∀ (a b c : ℕ),
      a + b + c = 115 →
      3 * a + 4 * b + 5 * c = 450 →
      c ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_five_dollar_frisbees_l1985_198540


namespace NUMINAMATH_CALUDE_olaf_toy_cars_l1985_198558

/-- Represents the toy car collection problem -/
def toy_car_problem (initial_collection : ℕ) (grandpa_factor : ℕ) (dad_gift : ℕ) 
  (mum_dad_diff : ℕ) (auntie_gift : ℕ) (final_total : ℕ) : Prop :=
  ∃ (uncle_gift : ℕ),
    initial_collection + (grandpa_factor * uncle_gift) + uncle_gift + 
    dad_gift + (dad_gift + mum_dad_diff) + auntie_gift = final_total ∧
    auntie_gift - uncle_gift = 1

/-- The specific instance of the toy car problem -/
theorem olaf_toy_cars : 
  toy_car_problem 150 2 10 5 6 196 := by
  sorry

end NUMINAMATH_CALUDE_olaf_toy_cars_l1985_198558


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_perpendicular_diagonals_l1985_198586

/-- Represents a point on a circle --/
structure CirclePoint where
  angle : Real

/-- Represents a quadrilateral formed by midpoints of arcs on a circle --/
structure MidpointQuadrilateral where
  p1 : CirclePoint
  p2 : CirclePoint
  p3 : CirclePoint
  p4 : CirclePoint

/-- Calculates the angle between two diagonals of a quadrilateral --/
def diagonalAngle (q : MidpointQuadrilateral) : Real :=
  -- Implementation details omitted
  sorry

/-- States that the diagonals of a quadrilateral formed by midpoints of four arcs on a circle are perpendicular --/
theorem midpoint_quadrilateral_perpendicular_diagonals 
  (c : CirclePoint → CirclePoint → CirclePoint → CirclePoint → MidpointQuadrilateral) :
  ∀ (p1 p2 p3 p4 : CirclePoint), 
    diagonalAngle (c p1 p2 p3 p4) = Real.pi / 2 := by
  sorry

#check midpoint_quadrilateral_perpendicular_diagonals

end NUMINAMATH_CALUDE_midpoint_quadrilateral_perpendicular_diagonals_l1985_198586


namespace NUMINAMATH_CALUDE_pillowcase_material_proof_l1985_198538

/-- The amount of material needed for one pillowcase -/
def pillowcase_material : ℝ := 1.25

theorem pillowcase_material_proof :
  let total_material : ℝ := 5000
  let third_bale_ratio : ℝ := 0.22
  let sheet_pillowcase_diff : ℝ := 3.25
  let sheets_sewn : ℕ := 150
  let pillowcases_sewn : ℕ := 240
  ∃ (first_bale second_bale third_bale : ℝ),
    first_bale + second_bale + third_bale = total_material ∧
    3 * first_bale = second_bale ∧
    third_bale = third_bale_ratio * total_material ∧
    sheets_sewn * (pillowcase_material + sheet_pillowcase_diff) + pillowcases_sewn * pillowcase_material = first_bale :=
by sorry

end NUMINAMATH_CALUDE_pillowcase_material_proof_l1985_198538


namespace NUMINAMATH_CALUDE_mod_nine_equiv_l1985_198529

theorem mod_nine_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_nine_equiv_l1985_198529


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l1985_198595

theorem negation_of_forall_exp_gt_x :
  ¬(∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l1985_198595


namespace NUMINAMATH_CALUDE_square_perimeter_l1985_198506

theorem square_perimeter (rectangle_length rectangle_width : ℝ)
  (h1 : rectangle_length = 50)
  (h2 : rectangle_width = 10)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  let square_perimeter := 4 * square_side
  square_perimeter = 200 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1985_198506


namespace NUMINAMATH_CALUDE_heart_ratio_equals_one_l1985_198544

-- Define the ♥ operation
def heart (n m : ℕ) : ℕ := n^3 * m^3

-- Theorem statement
theorem heart_ratio_equals_one : (heart 3 2) / (heart 2 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_equals_one_l1985_198544


namespace NUMINAMATH_CALUDE_simplify_expression_l1985_198579

theorem simplify_expression (x : ℝ) : 3*x + 4*x^3 + 2 - (7 - 3*x - 4*x^3) = 8*x^3 + 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1985_198579


namespace NUMINAMATH_CALUDE_polynomial_equality_l1985_198564

/-- Given a polynomial function q(x) satisfying the equation
    q(x) + (x^6 + 2x^4 + 5x^2 + 8x) = (3x^4 + 18x^3 + 20x^2 + 5x + 2),
    prove that q(x) = -x^6 + x^4 + 18x^3 + 15x^2 - 3x + 2 -/
theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (x^6 + 2*x^4 + 5*x^2 + 8*x) = (3*x^4 + 18*x^3 + 20*x^2 + 5*x + 2)) →
  (∀ x, q x = -x^6 + x^4 + 18*x^3 + 15*x^2 - 3*x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1985_198564


namespace NUMINAMATH_CALUDE_polly_happy_tweets_l1985_198596

/-- Represents the number of tweets Polly makes per minute in different states -/
structure PollyTweets where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration Polly spends in each state -/
structure Duration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given tweet rates and durations -/
def totalTweets (tweets : PollyTweets) (duration : Duration) : ℕ :=
  tweets.happy * duration.happy +
  tweets.hungry * duration.hungry +
  tweets.mirror * duration.mirror

/-- Theorem stating that Polly tweets 18 times per minute when happy -/
theorem polly_happy_tweets (tweets : PollyTweets) (duration : Duration) :
  tweets.hungry = 4 ∧
  tweets.mirror = 45 ∧
  duration.happy = 20 ∧
  duration.hungry = 20 ∧
  duration.mirror = 20 ∧
  totalTweets tweets duration = 1340 →
  tweets.happy = 18 := by
  sorry

end NUMINAMATH_CALUDE_polly_happy_tweets_l1985_198596


namespace NUMINAMATH_CALUDE_pen_pencil_cost_ratio_l1985_198598

/-- Given a pen and pencil with a total cost of $6, where the pen costs $4,
    prove that the ratio of the cost of the pen to the cost of the pencil is 4:1. -/
theorem pen_pencil_cost_ratio :
  ∀ (pen_cost pencil_cost : ℚ),
  pen_cost + pencil_cost = 6 →
  pen_cost = 4 →
  pen_cost / pencil_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_ratio_l1985_198598


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_equals_closed_open_interval_A_disjoint_B_iff_m_leq_neg_two_l1985_198561

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (I)
theorem intersection_A_complement_B_equals_closed_open_interval :
  (A ∩ (U \ B 3)) = Set.Icc 3 4 := by sorry

-- Theorem for part (II)
theorem A_disjoint_B_iff_m_leq_neg_two (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_equals_closed_open_interval_A_disjoint_B_iff_m_leq_neg_two_l1985_198561


namespace NUMINAMATH_CALUDE_problem_solution_l1985_198556

theorem problem_solution (x : ℚ) : 5 * x - 8 = 12 * x + 18 → 4 * (x - 3) = -188 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1985_198556


namespace NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_l1985_198557

/-- Represents the total number of products in the box -/
def total_products : ℕ := 5

/-- Represents the number of first-class products in the box -/
def first_class_products : ℕ := 3

/-- Represents the number of second-class products in the box -/
def second_class_products : ℕ := 2

/-- Represents the probability of drawing a second-class item on the first draw -/
def prob_second_class_first : ℚ := second_class_products / total_products

/-- Represents the probability of drawing a first-class item on the second draw,
    given that a second-class item was drawn first -/
def prob_first_class_second_given_second_first : ℚ := first_class_products / (total_products - 1)

/-- Represents the probability of drawing a first-class item on the first draw -/
def prob_first_class_first : ℚ := first_class_products / total_products

/-- Represents the probability of drawing a first-class item on the second draw,
    given that a first-class item was drawn first -/
def prob_first_class_second_given_first_first : ℚ := (first_class_products - 1) / (total_products - 1)

theorem probability_second_class_first_given_first_class_second :
  (prob_second_class_first * prob_first_class_second_given_second_first) /
  (prob_second_class_first * prob_first_class_second_given_second_first +
   prob_first_class_first * prob_first_class_second_given_first_first) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_l1985_198557


namespace NUMINAMATH_CALUDE_salary_calculation_l1985_198548

theorem salary_calculation (S : ℚ) 
  (food_expense : S / 5 = S * (1 / 5))
  (rent_expense : S / 10 = S * (1 / 10))
  (clothes_expense : S * 3 / 5 = S * (3 / 5))
  (remaining : S - (S / 5) - (S / 10) - (S * 3 / 5) = 18000) :
  S = 180000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l1985_198548


namespace NUMINAMATH_CALUDE_square_sum_equals_ten_l1985_198536

theorem square_sum_equals_ten (a b : ℝ) 
  (h1 : a + 3 = (b - 1)^2) 
  (h2 : b + 3 = (a - 1)^2) 
  (h3 : a ≠ b) : 
  a^2 + b^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_ten_l1985_198536


namespace NUMINAMATH_CALUDE_point_coordinates_l1985_198508

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ (x y : ℝ),
    fourth_quadrant x y →
    distance_to_x_axis y = 3 →
    distance_to_y_axis x = 4 →
    (x, y) = (4, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l1985_198508


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1985_198520

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1985_198520


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1985_198572

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_cond : 4 * a 2 = a 4) :
  let S_4 := (a 1) * (1 - q^4) / (1 - q)
  (S_4) / (a 2 + a 5) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1985_198572


namespace NUMINAMATH_CALUDE_extra_sweets_per_child_l1985_198521

theorem extra_sweets_per_child (total_children : ℕ) (absent_children : ℕ) (sweets_per_present_child : ℕ) :
  total_children = 190 →
  absent_children = 70 →
  sweets_per_present_child = 38 →
  (total_children - absent_children) * sweets_per_present_child / total_children - 
    ((total_children - absent_children) * sweets_per_present_child / total_children) = 14 := by
  sorry

end NUMINAMATH_CALUDE_extra_sweets_per_child_l1985_198521


namespace NUMINAMATH_CALUDE_sphere_hemisphere_radius_equality_l1985_198551

/-- The radius of a sphere is equal to the radius of each of two hemispheres 
    that have the same total volume as the original sphere. -/
theorem sphere_hemisphere_radius_equality (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * r^3) = (2 * (2 / 3 * Real.pi * r^3)) := by
  sorry

#check sphere_hemisphere_radius_equality

end NUMINAMATH_CALUDE_sphere_hemisphere_radius_equality_l1985_198551


namespace NUMINAMATH_CALUDE_one_third_minus_zero_point_three_three_three_l1985_198517

theorem one_third_minus_zero_point_three_three_three :
  (1 : ℚ) / 3 - (333 : ℚ) / 1000 = 1 / (3 * 1000) := by sorry

end NUMINAMATH_CALUDE_one_third_minus_zero_point_three_three_three_l1985_198517


namespace NUMINAMATH_CALUDE_sum_inequality_l1985_198542

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1985_198542


namespace NUMINAMATH_CALUDE_two_intersecting_circles_common_tangents_l1985_198528

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of common external tangents for two intersecting circles -/
def commonExternalTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- The distance between two points in a 2D plane -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem two_intersecting_circles_common_tangents :
  let c1 : Circle := { center := (1, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 3 }
  distance c1.center c2.center < c1.radius + c2.radius →
  commonExternalTangents c1 c2 = 2 :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_circles_common_tangents_l1985_198528


namespace NUMINAMATH_CALUDE_y_value_l1985_198527

theorem y_value (x y : ℝ) (h1 : x^2 = y - 7) (h2 : x = 6) : y = 43 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1985_198527


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_5_times_2_pow_3_l1985_198539

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 3^5 × 2^3 is 4 -/
theorem units_digit_of_3_pow_5_times_2_pow_3 :
  unitsDigit (3^5 * 2^3) = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_5_times_2_pow_3_l1985_198539


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l1985_198569

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  marks_per_correct : ℕ
  total_score : ℕ
  correct_answers : ℕ

/-- Calculates the marks lost per wrong answer in an examination -/
def marks_lost_per_wrong (exam : Examination) : ℚ :=
  let wrong_answers := exam.total_questions - exam.correct_answers
  let total_marks_for_correct := exam.marks_per_correct * exam.correct_answers
  let total_marks_lost := total_marks_for_correct - exam.total_score
  total_marks_lost / wrong_answers

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
  (h1 : exam.total_questions = 60)
  (h2 : exam.marks_per_correct = 4)
  (h3 : exam.total_score = 150)
  (h4 : exam.correct_answers = 42) :
  marks_lost_per_wrong exam = 1 := by
  sorry

#eval marks_lost_per_wrong { 
  total_questions := 60, 
  marks_per_correct := 4, 
  total_score := 150, 
  correct_answers := 42 
}

end NUMINAMATH_CALUDE_marks_lost_is_one_l1985_198569


namespace NUMINAMATH_CALUDE_min_value_of_S_l1985_198567

theorem min_value_of_S (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_S_l1985_198567


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_four_l1985_198581

/-- The function f(x) = ax³ - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

/-- The set of values for 'a' that satisfy the condition -/
def A : Set ℝ := {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ 0}

theorem f_nonnegative_iff_a_eq_four : A = {4} := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_four_l1985_198581


namespace NUMINAMATH_CALUDE_cube_root_7200_simplification_l1985_198501

theorem cube_root_7200_simplification : 
  ∃ (c d : ℕ+), (c.val : ℝ) * (d.val : ℝ)^(1/3) = 7200^(1/3) ∧ 
  (∀ (c' d' : ℕ+), (c'.val : ℝ) * (d'.val : ℝ)^(1/3) = 7200^(1/3) → d'.val ≤ d.val) →
  c.val + d.val = 452 := by
sorry

end NUMINAMATH_CALUDE_cube_root_7200_simplification_l1985_198501


namespace NUMINAMATH_CALUDE_stella_toilet_paper_stocking_l1985_198570

/-- Proves that Stella stocks 1 roll per day in each bathroom given the conditions --/
theorem stella_toilet_paper_stocking :
  let num_bathrooms : ℕ := 6
  let days_per_week : ℕ := 7
  let rolls_per_pack : ℕ := 12
  let weeks : ℕ := 4
  let packs_bought : ℕ := 14
  
  let total_rolls : ℕ := packs_bought * rolls_per_pack
  let rolls_per_week : ℕ := total_rolls / weeks
  let rolls_per_day : ℕ := rolls_per_week / days_per_week
  let rolls_per_bathroom_per_day : ℕ := rolls_per_day / num_bathrooms

  rolls_per_bathroom_per_day = 1 :=
by sorry

end NUMINAMATH_CALUDE_stella_toilet_paper_stocking_l1985_198570


namespace NUMINAMATH_CALUDE_find_s_l1985_198575

theorem find_s (r s : ℝ) (hr : r > 1) (hs : s > 1) 
  (h1 : 1/r + 1/s = 1) (h2 : r*s = 9) : 
  s = (9 + 3*Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_s_l1985_198575


namespace NUMINAMATH_CALUDE_four_variable_inequality_l1985_198500

theorem four_variable_inequality (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) 
  (h5 : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_four_variable_inequality_l1985_198500


namespace NUMINAMATH_CALUDE_complex_cube_theorem_l1985_198546

theorem complex_cube_theorem : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / (1 - 2*i)
  z^3 = -i := by sorry

end NUMINAMATH_CALUDE_complex_cube_theorem_l1985_198546


namespace NUMINAMATH_CALUDE_exactly_two_points_l1985_198590

/-- Given two points A and B in a plane that are 12 units apart, this function
    returns the number of points C such that triangle ABC has a perimeter of 36 units,
    an area of 72 square units, and is isosceles. -/
def count_valid_points (A B : ℝ × ℝ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly two points C satisfying the conditions. -/
theorem exactly_two_points (A B : ℝ × ℝ) 
    (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12) : 
    count_valid_points A B = 2 :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_points_l1985_198590


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1985_198515

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1985_198515


namespace NUMINAMATH_CALUDE_original_price_satisfies_conditions_l1985_198576

/-- The original price of a dish that satisfies the given conditions --/
def original_price : ℝ := 42

/-- John's payment given the original price --/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's payment given the original price --/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions --/
theorem original_price_satisfies_conditions :
  john_payment original_price - jane_payment original_price = 0.63 :=
sorry

end NUMINAMATH_CALUDE_original_price_satisfies_conditions_l1985_198576


namespace NUMINAMATH_CALUDE_max_range_of_five_numbers_l1985_198524

theorem max_range_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- Distinct and ordered
  (a + b + c + d + e) / 5 = 13 →   -- Average is 13
  c = 15 →                         -- Median is 15
  e - a ≤ 33 :=                    -- Maximum range is at most 33
by sorry

end NUMINAMATH_CALUDE_max_range_of_five_numbers_l1985_198524


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l1985_198513

def eggplant_pounds : ℝ := 5
def eggplant_price : ℝ := 2
def zucchini_pounds : ℝ := 4
def zucchini_price : ℝ := 2
def tomato_pounds : ℝ := 4
def tomato_price : ℝ := 3.5
def onion_pounds : ℝ := 3
def onion_price : ℝ := 1
def basil_pounds : ℝ := 1
def basil_price : ℝ := 2.5
def basil_unit : ℝ := 0.5
def yield_quarts : ℝ := 4

theorem ratatouille_cost_per_quart :
  let total_cost := eggplant_pounds * eggplant_price +
                    zucchini_pounds * zucchini_price +
                    tomato_pounds * tomato_price +
                    onion_pounds * onion_price +
                    basil_pounds / basil_unit * basil_price
  total_cost / yield_quarts = 10 := by sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l1985_198513


namespace NUMINAMATH_CALUDE_largest_square_tile_l1985_198592

theorem largest_square_tile (a b : ℕ) (ha : a = 72) (hb : b = 90) :
  ∃ (s : ℕ), s = Nat.gcd a b ∧ 
  s * (a / s) = a ∧ 
  s * (b / s) = b ∧
  ∀ (t : ℕ), t * (a / t) = a → t * (b / t) = b → t ≤ s :=
sorry

end NUMINAMATH_CALUDE_largest_square_tile_l1985_198592


namespace NUMINAMATH_CALUDE_function_inequality_relation_l1985_198511

theorem function_inequality_relation (f : ℝ → ℝ) (a b : ℝ) :
  (f = λ x => 3 * x + 1) →
  (a > 0 ∧ b > 0) →
  (∀ x, |x - 1| < b → |f x - 4| < a) →
  a - 3 * b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_function_inequality_relation_l1985_198511


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1985_198562

/-- An isosceles triangle with side lengths satisfying a specific equation has a perimeter of either 10 or 11. -/
theorem isosceles_triangle_perimeter (x y : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧
    (a = b ∨ a + a = b ∨ b + b = a)) →  -- isosceles condition
  |x^2 - 9| + (y - 4)^2 = 0 →
  (x + y + min x y = 10) ∨ (x + y + min x y = 11) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1985_198562


namespace NUMINAMATH_CALUDE_jennas_stickers_l1985_198580

/-- Given that the ratio of Kate's stickers to Jenna's stickers is 7:4 and Kate has 21 stickers,
    prove that Jenna has 12 stickers. -/
theorem jennas_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) : 
  (kate_stickers : ℚ) / jenna_stickers = 7 / 4 → kate_stickers = 21 → jenna_stickers = 12 := by
  sorry

end NUMINAMATH_CALUDE_jennas_stickers_l1985_198580


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l1985_198573

theorem point_on_unit_circle (x : ℝ) (θ : ℝ) :
  (∃ (M : ℝ × ℝ), M = (x, 1) ∧ M.1 = x * Real.cos θ ∧ M.2 = x * Real.sin θ) →
  Real.cos θ = (Real.sqrt 2 / 2) * x →
  x = -1 ∨ x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l1985_198573


namespace NUMINAMATH_CALUDE_linear_equation_passes_through_points_l1985_198589

/-- The linear equation passing through points A(1, 2) and B(3, 4) -/
def linear_equation (x y : ℝ) : Prop := y = x + 1

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- Point B -/
def point_B : ℝ × ℝ := (3, 4)

/-- Theorem: The linear equation passes through points A and B -/
theorem linear_equation_passes_through_points :
  linear_equation point_A.1 point_A.2 ∧ linear_equation point_B.1 point_B.2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_passes_through_points_l1985_198589


namespace NUMINAMATH_CALUDE_gcf_64_144_l1985_198549

theorem gcf_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcf_64_144_l1985_198549


namespace NUMINAMATH_CALUDE_office_employees_l1985_198565

theorem office_employees (total_employees : ℕ) : 
  (total_employees : ℝ) * 0.25 * 0.6 = 120 → total_employees = 800 := by
  sorry

end NUMINAMATH_CALUDE_office_employees_l1985_198565


namespace NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l1985_198563

/-- Given two parallel vectors a and b, prove that x = 1 -/
theorem parallel_vectors_x_equals_one (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 4)
  (∃ (k : ℝ), a = k • b) →
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l1985_198563


namespace NUMINAMATH_CALUDE_lemonade_stand_problem_l1985_198599

theorem lemonade_stand_problem (bea_price dawn_price : ℚ) (bea_glasses : ℕ) (earnings_difference : ℚ) :
  bea_price = 25 / 100 →
  dawn_price = 28 / 100 →
  bea_glasses = 10 →
  earnings_difference = 26 / 100 →
  ∃ dawn_glasses : ℕ,
    dawn_glasses = 8 ∧
    bea_price * bea_glasses = dawn_price * dawn_glasses + earnings_difference :=
by
  sorry

#check lemonade_stand_problem

end NUMINAMATH_CALUDE_lemonade_stand_problem_l1985_198599


namespace NUMINAMATH_CALUDE_min_mozart_and_bach_not_beethoven_l1985_198543

def total_surveyed : ℕ := 150
def mozart_fans : ℕ := 120
def bach_fans : ℕ := 105
def beethoven_fans : ℕ := 45

theorem min_mozart_and_bach_not_beethoven :
  ∃ (mozart_set bach_set beethoven_set : Finset (Fin total_surveyed)),
    mozart_set.card = mozart_fans ∧
    bach_set.card = bach_fans ∧
    beethoven_set.card = beethoven_fans ∧
    ((mozart_set ∩ bach_set) \ beethoven_set).card ≥ 75 ∧
    ∀ (m b be : Finset (Fin total_surveyed)),
      m.card = mozart_fans →
      b.card = bach_fans →
      be.card = beethoven_fans →
      ((m ∩ b) \ be).card ≥ 75 :=
by sorry

end NUMINAMATH_CALUDE_min_mozart_and_bach_not_beethoven_l1985_198543


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l1985_198523

theorem abs_neg_two_equals_two : abs (-2 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l1985_198523


namespace NUMINAMATH_CALUDE_no_linear_factor_l1985_198555

theorem no_linear_factor (x y z : ℤ) : 
  ¬ ∃ (a b c d : ℤ), (a*x + b*y + c*z + d) ∣ (x^2 - y^2 - z^2 + 2*x*y + x + y - z) :=
sorry

end NUMINAMATH_CALUDE_no_linear_factor_l1985_198555


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l1985_198534

/-- Given a function f(x) = a*sin(x) + b*x^3 + 5 where f(1) = 3, prove that f(-1) = 7 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * x^3 + 5) 
  (h2 : f 1 = 3) : 
  f (-1) = 7 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l1985_198534


namespace NUMINAMATH_CALUDE_find_x_l1985_198587

theorem find_x (a b c d x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + x) = 4 * a / (3 * b)) 
  (h4 : c = 4 * a) 
  (h5 : d = 3 * b) :
  x = a * b / (3 * b - 4 * a) :=
by sorry

end NUMINAMATH_CALUDE_find_x_l1985_198587


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1985_198526

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1985_198526


namespace NUMINAMATH_CALUDE_alex_savings_dimes_l1985_198541

/-- Proves that given $6.35 in dimes and quarters, with 5 more dimes than quarters, the number of dimes is 22 -/
theorem alex_savings_dimes : 
  ∀ (d q : ℕ), 
    (d : ℚ) * 0.1 + (q : ℚ) * 0.25 = 6.35 → -- Total value in dollars
    d = q + 5 → -- 5 more dimes than quarters
    d = 22 := by sorry

end NUMINAMATH_CALUDE_alex_savings_dimes_l1985_198541


namespace NUMINAMATH_CALUDE_count_squares_specific_grid_l1985_198518

/-- Represents a grid with a diagonal line --/
structure DiagonalGrid :=
  (width : Nat)
  (height : Nat)
  (diagonalLength : Nat)

/-- Counts the number of squares in a diagonal grid --/
def countSquares (g : DiagonalGrid) : Nat :=
  sorry

/-- The specific 6x5 grid with a diagonal in the top-left 3x3 square --/
def specificGrid : DiagonalGrid :=
  { width := 6, height := 5, diagonalLength := 3 }

/-- Theorem stating that the number of squares in the specific grid is 64 --/
theorem count_squares_specific_grid :
  countSquares specificGrid = 64 := by sorry

end NUMINAMATH_CALUDE_count_squares_specific_grid_l1985_198518


namespace NUMINAMATH_CALUDE_get_ready_time_l1985_198554

/-- The time it takes for Jack and his three toddlers to get ready -/
def total_time (jack_socks jack_shoes jack_jacket toddler_socks toddler_shoes toddler_laces num_toddlers : ℕ) : ℕ :=
  let jack_time := jack_socks + jack_shoes + jack_jacket
  let toddler_time := toddler_socks + toddler_shoes + toddler_laces
  jack_time + num_toddlers * toddler_time

/-- Theorem stating that it takes 33 minutes for Jack and his three toddlers to get ready -/
theorem get_ready_time : total_time 2 4 3 2 5 1 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_get_ready_time_l1985_198554


namespace NUMINAMATH_CALUDE_complex_division_result_l1985_198537

theorem complex_division_result : 
  let z : ℂ := (3 + 7*I) / I
  (z.re = 7) ∧ (z.im = -3) := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l1985_198537


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1985_198502

/-- Given a geometric sequence {a_n} where the 5th term is equal to the constant term
    in the expansion of (x + 1/x)^4, prove that a_3 * a_7 = 36 -/
theorem geometric_sequence_property (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 5 = 6) →  -- 5th term is equal to the constant term in (x + 1/x)^4
  a 3 * a 7 = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1985_198502


namespace NUMINAMATH_CALUDE_angle_terminal_side_range_l1985_198578

theorem angle_terminal_side_range (θ : Real) (a : Real) :
  (∃ (x y : Real), x = a - 2 ∧ y = a + 2 ∧ x = y * Real.tan θ) →
  Real.cos θ ≤ 0 →
  Real.sin θ > 0 →
  a ∈ Set.Ioo (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_range_l1985_198578


namespace NUMINAMATH_CALUDE_remainder_three_power_800_mod_17_l1985_198503

theorem remainder_three_power_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_800_mod_17_l1985_198503


namespace NUMINAMATH_CALUDE_brandy_caffeine_excess_l1985_198525

/-- Represents the caffeine consumption and limits for an individual -/
structure CaffeineProfile where
  weight : ℝ
  additionalTolerance : ℝ
  coffeeConsumption : ℕ
  coffeinePer : ℝ
  energyDrinkConsumption : ℕ
  energyDrinkCaffeine : ℝ
  standardLimit : ℝ

/-- Calculates the total safe caffeine amount for an individual -/
def totalSafeAmount (profile : CaffeineProfile) : ℝ :=
  profile.weight * profile.standardLimit + profile.additionalTolerance

/-- Calculates the total caffeine consumed -/
def totalConsumed (profile : CaffeineProfile) : ℝ :=
  (profile.coffeeConsumption : ℝ) * profile.coffeinePer +
  (profile.energyDrinkConsumption : ℝ) * profile.energyDrinkCaffeine

/-- Theorem stating that Brandy has exceeded her safe caffeine limit by 470 mg -/
theorem brandy_caffeine_excess (brandy : CaffeineProfile)
  (h1 : brandy.weight = 60)
  (h2 : brandy.additionalTolerance = 50)
  (h3 : brandy.coffeeConsumption = 2)
  (h4 : brandy.coffeinePer = 95)
  (h5 : brandy.energyDrinkConsumption = 4)
  (h6 : brandy.energyDrinkCaffeine = 120)
  (h7 : brandy.standardLimit = 2.5) :
  totalConsumed brandy - totalSafeAmount brandy = 470 := by
  sorry

end NUMINAMATH_CALUDE_brandy_caffeine_excess_l1985_198525


namespace NUMINAMATH_CALUDE_monkey_swinging_speed_l1985_198531

/-- Represents the speed and time of a monkey's movement --/
structure MonkeyMovement where
  speed : ℝ
  time : ℝ

/-- Calculates the total distance traveled by the monkey --/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.time + swinging.speed * swinging.time

/-- Theorem: The monkey's swinging speed is 10 feet per second --/
theorem monkey_swinging_speed 
  (running_speed : ℝ) 
  (running_time : ℝ) 
  (swinging_time : ℝ) 
  (total_distance : ℝ)
  (h1 : running_speed = 15)
  (h2 : running_time = 5)
  (h3 : swinging_time = 10)
  (h4 : total_distance = 175)
  (h5 : totalDistance 
    { speed := running_speed, time := running_time } 
    { speed := (total_distance - running_speed * running_time) / swinging_time, time := swinging_time } = total_distance) :
  (total_distance - running_speed * running_time) / swinging_time = 10 := by
  sorry

#check monkey_swinging_speed

end NUMINAMATH_CALUDE_monkey_swinging_speed_l1985_198531


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l1985_198535

def distribute_candy (n : ℕ) (k : ℕ) (min_red min_blue : ℕ) : ℕ :=
  sorry

theorem candy_distribution_theorem :
  distribute_candy 8 4 2 2 = 2048 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l1985_198535


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l1985_198591

/-- Proves that adding 70 ounces of 60% salt solution to 70 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_solution_mixture : 
  let initial_volume : ℝ := 70
  let initial_concentration : ℝ := 0.2
  let added_volume : ℝ := 70
  let added_concentration : ℝ := 0.6
  let final_concentration : ℝ := 0.4
  (initial_volume * initial_concentration + added_volume * added_concentration) / (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l1985_198591


namespace NUMINAMATH_CALUDE_only_jia_can_formulate_quadratic_l1985_198519

/-- Represents a person in the problem -/
inductive Person
  | jia
  | yi
  | bing
  | ding

/-- Checks if a number is congruent to 1 modulo 3 -/
def is_cong_1_mod_3 (n : ℤ) : Prop := n % 3 = 1

/-- Checks if a number is congruent to 2 modulo 3 -/
def is_cong_2_mod_3 (n : ℤ) : Prop := n % 3 = 2

/-- Represents the conditions for each person's quadratic equation -/
def satisfies_conditions (person : Person) (p q x₁ x₂ : ℤ) : Prop :=
  match person with
  | Person.jia => is_cong_1_mod_3 p ∧ is_cong_1_mod_3 q ∧ is_cong_1_mod_3 x₁ ∧ is_cong_1_mod_3 x₂
  | Person.yi => is_cong_2_mod_3 p ∧ is_cong_2_mod_3 q ∧ is_cong_2_mod_3 x₁ ∧ is_cong_2_mod_3 x₂
  | Person.bing => is_cong_1_mod_3 p ∧ is_cong_1_mod_3 q ∧ is_cong_2_mod_3 x₁ ∧ is_cong_2_mod_3 x₂
  | Person.ding => is_cong_2_mod_3 p ∧ is_cong_2_mod_3 q ∧ is_cong_1_mod_3 x₁ ∧ is_cong_1_mod_3 x₂

/-- Represents a valid quadratic equation -/
def is_valid_quadratic (p q x₁ x₂ : ℤ) : Prop :=
  x₁ + x₂ = -p ∧ x₁ * x₂ = q

/-- The main theorem stating that only Jia can formulate a valid quadratic equation -/
theorem only_jia_can_formulate_quadratic :
  ∀ person : Person,
    (∃ p q x₁ x₂ : ℤ, satisfies_conditions person p q x₁ x₂ ∧ is_valid_quadratic p q x₁ x₂) ↔
    person = Person.jia :=
sorry


end NUMINAMATH_CALUDE_only_jia_can_formulate_quadratic_l1985_198519


namespace NUMINAMATH_CALUDE_rice_sale_proof_l1985_198588

/-- Calculates the daily amount of rice to be sold given initial amount, additional amount, and number of days -/
def daily_rice_sale (initial_tons : ℕ) (additional_kg : ℕ) (days : ℕ) : ℕ :=
  (initial_tons * 1000 + additional_kg) / days

/-- Proves that given 4 tons of rice initially, with an additional 4000 kilograms transported in,
    and needing to be sold within 4 days, the amount of rice to be sold each day is 2000 kilograms -/
theorem rice_sale_proof : daily_rice_sale 4 4000 4 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_rice_sale_proof_l1985_198588


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1985_198533

/-- Proves that given the conditions, the principal amount is 20,000 --/
theorem compound_interest_problem (P : ℝ) : 
  P * ((1 + 0.1)^4 - (1 + 0.2)^2) = 482 → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1985_198533


namespace NUMINAMATH_CALUDE_solve_complex_equation_l1985_198566

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := 2 - 3 * i * z = -4 + 5 * i * z

-- State the theorem
theorem solve_complex_equation :
  ∃ z : ℂ, equation z ∧ z = -3/4 * i :=
sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l1985_198566


namespace NUMINAMATH_CALUDE_factorization_x3_plus_5x_l1985_198571

theorem factorization_x3_plus_5x (x : ℂ) : x^3 + 5*x = x * (x - Complex.I * Real.sqrt 5) * (x + Complex.I * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x3_plus_5x_l1985_198571


namespace NUMINAMATH_CALUDE_yoojeong_line_length_l1985_198553

/-- Conversion factor from centimeters to millimeters -/
def cm_to_mm : ℕ := 10

/-- The length of the reference line in centimeters -/
def reference_length_cm : ℕ := 31

/-- The difference in millimeters between the reference line and Yoojeong's line -/
def difference_mm : ℕ := 3

/-- The length of Yoojeong's line in millimeters -/
def yoojeong_line_mm : ℕ := reference_length_cm * cm_to_mm - difference_mm

theorem yoojeong_line_length : yoojeong_line_mm = 307 :=
by sorry

end NUMINAMATH_CALUDE_yoojeong_line_length_l1985_198553


namespace NUMINAMATH_CALUDE_baseball_league_games_l1985_198585

/-- The number of games played in a baseball league -/
def total_games (n : ℕ) (games_per_matchup : ℕ) : ℕ :=
  n * (n - 1) * games_per_matchup / 2

/-- Theorem: In a 12-team league where each team plays 4 games with every other team, 
    the total number of games played is 264 -/
theorem baseball_league_games : 
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_l1985_198585


namespace NUMINAMATH_CALUDE_max_value_of_five_integers_with_mean_eleven_l1985_198514

theorem max_value_of_five_integers_with_mean_eleven (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  (a + b + c + d + e : ℚ) / 5 = 11 →
  max a (max b (max c (max d e))) ≤ 45 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_five_integers_with_mean_eleven_l1985_198514


namespace NUMINAMATH_CALUDE_playground_girls_l1985_198545

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) :
  total_children = 97 → boys = 44 → girls = total_children - boys → girls = 53 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l1985_198545


namespace NUMINAMATH_CALUDE_division_4512_by_32_l1985_198512

theorem division_4512_by_32 : ∃ (q r : ℕ), 4512 = 32 * q + r ∧ r < 32 ∧ q = 141 ∧ r = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_4512_by_32_l1985_198512


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1985_198568

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1985_198568


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1985_198510

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 8*x + 12 = 0 ↔ x = 6 ∨ x = 2) ∧
  (∀ x : ℝ, (x - 3)^2 = 2*x*(x - 3) ↔ x = 3 ∨ x = -3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1985_198510


namespace NUMINAMATH_CALUDE_angle_measure_l1985_198530

theorem angle_measure : 
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ (180 - x) = 3 * x + 10 → x = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1985_198530


namespace NUMINAMATH_CALUDE_division_problem_l1985_198597

theorem division_problem (total : ℝ) (a b c : ℝ) (h1 : total = 1080) 
  (h2 : a = (1/3) * (b + c)) (h3 : a = b + 30) (h4 : a + b + c = total) 
  (h5 : ∃ f : ℝ, b = f * (a + c)) : 
  ∃ f : ℝ, b = f * (a + c) ∧ f = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1985_198597


namespace NUMINAMATH_CALUDE_relay_team_selection_l1985_198532

-- Define the number of sprinters
def total_sprinters : ℕ := 6

-- Define the number of sprinters to be selected
def selected_sprinters : ℕ := 4

-- Define a function to calculate the number of ways to select and arrange sprinters
def relay_arrangements (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

-- Define a function to calculate the number of ways with restrictions
def restricted_arrangements (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

theorem relay_team_selection :
  restricted_arrangements total_sprinters selected_sprinters = 252 :=
sorry

end NUMINAMATH_CALUDE_relay_team_selection_l1985_198532


namespace NUMINAMATH_CALUDE_rectangle_area_15_20_l1985_198550

/-- The area of a rectangular field with given length and width -/
def rectangle_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular field with length 15 meters and width 20 meters is 300 square meters -/
theorem rectangle_area_15_20 :
  rectangle_area 15 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_15_20_l1985_198550


namespace NUMINAMATH_CALUDE_divisible_by_five_unit_digits_l1985_198582

theorem divisible_by_five_unit_digits :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 5 = 0 ↔ n % 10 ∈ S) ∧ Finset.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_five_unit_digits_l1985_198582


namespace NUMINAMATH_CALUDE_division_problem_l1985_198505

theorem division_problem : (100 : ℚ) / ((5 / 2) * 3) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1985_198505


namespace NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_400_l1985_198594

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_400 :
  (25 : ℝ) / 100 * 400 = 100 := by sorry

end NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_400_l1985_198594


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1985_198509

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x^2 - 5*x + 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x + 6 > 0 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1985_198509


namespace NUMINAMATH_CALUDE_hotel_room_charges_l1985_198560

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = 1.8 * G := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l1985_198560


namespace NUMINAMATH_CALUDE_cross_in_square_l1985_198574

theorem cross_in_square (s : ℝ) : 
  s > 0 → 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → 
  s = 36 := by
sorry

end NUMINAMATH_CALUDE_cross_in_square_l1985_198574


namespace NUMINAMATH_CALUDE_max_individual_points_is_23_l1985_198577

/-- Represents a basketball team -/
structure BasketballTeam where
  players : Nat
  totalPoints : Nat
  minPointsPerPlayer : Nat

/-- Calculates the maximum points a single player could have scored -/
def maxIndividualPoints (team : BasketballTeam) : Nat :=
  team.totalPoints - (team.players - 1) * team.minPointsPerPlayer

/-- Theorem: The maximum points an individual player could have scored is 23 -/
theorem max_individual_points_is_23 (team : BasketballTeam) 
  (h1 : team.players = 12)
  (h2 : team.totalPoints = 100)
  (h3 : team.minPointsPerPlayer = 7) :
  maxIndividualPoints team = 23 := by
  sorry

#eval maxIndividualPoints ⟨12, 100, 7⟩

end NUMINAMATH_CALUDE_max_individual_points_is_23_l1985_198577
