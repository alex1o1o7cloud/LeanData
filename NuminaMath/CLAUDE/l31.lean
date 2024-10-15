import Mathlib

namespace NUMINAMATH_CALUDE_value_of_2a_minus_3b_l31_3141

-- Define the functions f, g, and h
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_minus_3b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a - 3 * b = 22 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_minus_3b_l31_3141


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l31_3128

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a3 : a 3 = -3) : 
  ∀ n : ℕ, a n = -2 * n + 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l31_3128


namespace NUMINAMATH_CALUDE_road_repair_hours_l31_3119

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 33)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 11)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people1 * days1).lcm (people2 * days2 * hours2) / (people1 * days1) = 
        people2 * days2 * hours2 * (people1 * days1).lcm (people2 * days2 * hours2) / (people2 * days2 * hours2)) :
  (people1 * days1).lcm (people2 * days2 * hours2) / (people1 * days1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_hours_l31_3119


namespace NUMINAMATH_CALUDE_max_sum_on_ellipse_l31_3185

theorem max_sum_on_ellipse :
  ∀ x y : ℝ, (x - 2)^2 / 4 + (y - 1)^2 = 1 →
  ∀ x' y' : ℝ, (x' - 2)^2 / 4 + (y' - 1)^2 = 1 →
  x + y ≤ 3 + Real.sqrt 5 ∧
  ∃ x₀ y₀ : ℝ, (x₀ - 2)^2 / 4 + (y₀ - 1)^2 = 1 ∧ x₀ + y₀ = 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_ellipse_l31_3185


namespace NUMINAMATH_CALUDE_parabola_properties_l31_3131

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 6*x - 1

-- Theorem statement
theorem parabola_properties :
  -- Vertex coordinates
  (∃ (x y : ℝ), x = -3 ∧ y = -10 ∧ ∀ (t : ℝ), f t ≥ f x) ∧
  -- Axis of symmetry
  (∀ (x : ℝ), f (x - 3) = f (-x - 3)) ∧
  -- Y-axis intersection point
  f 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l31_3131


namespace NUMINAMATH_CALUDE_prob_at_least_two_hits_eq_81_125_l31_3101

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots. -/
def prob_at_least_two_hits : ℝ := 
  Finset.sum (Finset.range (n + 1) \ Finset.range 2) (λ k => 
    (n.choose k : ℝ) * p^k * (1 - p)^(n - k))

theorem prob_at_least_two_hits_eq_81_125 : 
  prob_at_least_two_hits = 81 / 125 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_hits_eq_81_125_l31_3101


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l31_3162

-- Define the hyperbola equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 3) + y^2 / (2 - k) = 1

-- Define the condition for foci on y-axis
def foci_on_y_axis (k : ℝ) : Prop :=
  2 - k > 0 ∧ k - 3 < 0

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, hyperbola_equation x y k) ∧ foci_on_y_axis k → k < 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l31_3162


namespace NUMINAMATH_CALUDE_steves_return_speed_l31_3187

/-- Proves that given a round trip of 60 km (30 km each way), where the return speed is twice 
    the outbound speed, and the total travel time is 6 hours, the return speed is 15 km/h. -/
theorem steves_return_speed 
  (distance : ℝ) 
  (total_time : ℝ) 
  (speed_to_work : ℝ) 
  (speed_from_work : ℝ) : 
  distance = 30 →
  total_time = 6 →
  speed_from_work = 2 * speed_to_work →
  distance / speed_to_work + distance / speed_from_work = total_time →
  speed_from_work = 15 := by
  sorry


end NUMINAMATH_CALUDE_steves_return_speed_l31_3187


namespace NUMINAMATH_CALUDE_square_area_increase_l31_3154

/-- The increase in area of a square when its side length is increased -/
theorem square_area_increase (initial_side : ℝ) (increase : ℝ) : 
  initial_side = 6 → increase = 1 → 
  (initial_side + increase)^2 - initial_side^2 = 13 := by
  sorry

#check square_area_increase

end NUMINAMATH_CALUDE_square_area_increase_l31_3154


namespace NUMINAMATH_CALUDE_total_cost_of_flowers_l31_3198

/-- The cost of a single flower in dollars -/
def flower_cost : ℕ := 3

/-- The number of roses bought -/
def roses_bought : ℕ := 2

/-- The number of daisies bought -/
def daisies_bought : ℕ := 2

/-- The total number of flowers bought -/
def total_flowers : ℕ := roses_bought + daisies_bought

/-- The theorem stating the total cost of the flowers -/
theorem total_cost_of_flowers : total_flowers * flower_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_flowers_l31_3198


namespace NUMINAMATH_CALUDE_unique_rectangle_with_given_perimeter_and_area_l31_3172

theorem unique_rectangle_with_given_perimeter_and_area : 
  ∃! (w h : ℕ+), (2 * (w + h) = 80) ∧ (w * h = 400) :=
by sorry

end NUMINAMATH_CALUDE_unique_rectangle_with_given_perimeter_and_area_l31_3172


namespace NUMINAMATH_CALUDE_negation_equivalence_l31_3143

/-- A function f: ℝ → ℝ is monotonically increasing on (0, +∞) if for all x₁, x₂ ∈ (0, +∞),
    x₁ < x₂ implies f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The negation of the existence of a real k such that y = k/x is monotonically increasing
    on (0, +∞) is equivalent to the statement that for all real k, y = k/x is not
    monotonically increasing on (0, +∞) -/
theorem negation_equivalence : 
  (¬ ∃ k : ℝ, MonotonicallyIncreasing (fun x ↦ k / x)) ↔ 
  (∀ k : ℝ, ¬ MonotonicallyIncreasing (fun x ↦ k / x)) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l31_3143


namespace NUMINAMATH_CALUDE_permutations_of_33377_l31_3146

/-- The number of permutations of a multiset with 5 elements, where 3 elements are the same and 2 elements are the same -/
def permutations_of_multiset : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

theorem permutations_of_33377 : permutations_of_multiset = 10 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_33377_l31_3146


namespace NUMINAMATH_CALUDE_basketball_players_count_l31_3121

theorem basketball_players_count (total_athletes : ℕ) 
  (football_ratio baseball_ratio soccer_ratio basketball_ratio : ℕ) : 
  total_athletes = 104 →
  football_ratio = 10 →
  baseball_ratio = 7 →
  soccer_ratio = 5 →
  basketball_ratio = 4 →
  (basketball_ratio * total_athletes) / (football_ratio + baseball_ratio + soccer_ratio + basketball_ratio) = 16 := by
  sorry

end NUMINAMATH_CALUDE_basketball_players_count_l31_3121


namespace NUMINAMATH_CALUDE_largest_touching_sphere_radius_l31_3130

/-- A regular tetrahedron inscribed in a unit sphere -/
structure InscribedTetrahedron where
  /-- The tetrahedron is regular -/
  isRegular : Bool
  /-- The tetrahedron is inscribed in a unit sphere -/
  isInscribed : Bool

/-- A sphere touching the unit sphere internally and the tetrahedron externally -/
structure TouchingSphere where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The sphere touches the unit sphere internally -/
  touchesUnitSphereInternally : Bool
  /-- The sphere touches the tetrahedron externally -/
  touchesTetrahedronExternally : Bool

/-- The theorem stating the radius of the largest touching sphere -/
theorem largest_touching_sphere_radius 
  (t : InscribedTetrahedron) 
  (s : TouchingSphere) 
  (h1 : t.isRegular = true) 
  (h2 : t.isInscribed = true)
  (h3 : s.touchesUnitSphereInternally = true)
  (h4 : s.touchesTetrahedronExternally = true) :
  s.radius = 1/3 :=
sorry

end NUMINAMATH_CALUDE_largest_touching_sphere_radius_l31_3130


namespace NUMINAMATH_CALUDE_lizzys_final_money_l31_3151

/-- Calculates the final amount of money Lizzy has after a series of transactions -/
def lizzys_money (
  mother_gave : ℕ)
  (father_gave : ℕ)
  (candy_cost : ℕ)
  (uncle_gave : ℕ)
  (toy_price : ℕ)
  (discount_percent : ℕ)
  (change_dollars : ℕ)
  (change_cents : ℕ) : ℕ :=
  let initial := mother_gave + father_gave
  let after_candy := initial - candy_cost + uncle_gave
  let discounted_price := toy_price - (toy_price * discount_percent / 100)
  let after_toy := after_candy - discounted_price
  let final := after_toy + change_dollars * 100 + change_cents
  final

theorem lizzys_final_money :
  lizzys_money 80 40 50 70 90 20 1 10 = 178 := by
  sorry

end NUMINAMATH_CALUDE_lizzys_final_money_l31_3151


namespace NUMINAMATH_CALUDE_first_degree_function_characterization_l31_3127

-- Define a first-degree function
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem first_degree_function_characterization
  (f : ℝ → ℝ) 
  (h1 : FirstDegreeFunction f)
  (h2 : ∀ x : ℝ, f (f x) = 4 * x + 6) :
  (∀ x : ℝ, f x = 2 * x + 2) ∨ (∀ x : ℝ, f x = -2 * x - 6) :=
sorry

end NUMINAMATH_CALUDE_first_degree_function_characterization_l31_3127


namespace NUMINAMATH_CALUDE_expected_difference_l31_3112

/-- The number of students in the school -/
def total_students : ℕ := 100

/-- The number of classes and teachers -/
def num_classes : ℕ := 5

/-- The distribution of students across classes -/
def class_sizes : List ℕ := [40, 40, 10, 5, 5]

/-- The expected number of students per class when choosing a teacher at random -/
def t : ℚ := (total_students : ℚ) / num_classes

/-- The expected number of students per class when choosing a student at random -/
def s : ℚ := (List.sum (List.map (fun x => x * x) class_sizes) : ℚ) / total_students

theorem expected_difference :
  t - s = -27/2 := by sorry

end NUMINAMATH_CALUDE_expected_difference_l31_3112


namespace NUMINAMATH_CALUDE_ceiling_abs_negative_l31_3199

theorem ceiling_abs_negative : ⌈|(-52.7 : ℝ)|⌉ = 53 := by sorry

end NUMINAMATH_CALUDE_ceiling_abs_negative_l31_3199


namespace NUMINAMATH_CALUDE_max_value_of_expression_l31_3144

theorem max_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5) ≤ 17 ∧ 
  ∃ (y : ℝ), (4 * y^2 + 8 * y + 21) / (4 * y^2 + 8 * y + 5) = 17 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l31_3144


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l31_3133

/-- The number of candy pieces left after combining and eating some. -/
def candy_left (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  katie_candy + sister_candy - eaten_candy

/-- Theorem stating the number of candy pieces left in the given scenario. -/
theorem halloween_candy_theorem :
  candy_left 8 23 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l31_3133


namespace NUMINAMATH_CALUDE_max_score_15_cards_l31_3105

/-- The score of a hand of cards -/
def score (R B Y : ℕ) : ℕ :=
  R + 2 * R * B + 3 * B * Y

/-- The theorem stating the maximum score achievable with 15 cards -/
theorem max_score_15_cards :
  ∃ R B Y : ℕ,
    R + B + Y = 15 ∧
    ∀ R' B' Y' : ℕ, R' + B' + Y' = 15 →
      score R' B' Y' ≤ score R B Y ∧
      score R B Y = 168 :=
sorry

end NUMINAMATH_CALUDE_max_score_15_cards_l31_3105


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l31_3157

/-- The probability of a specific pairing in a class with random pairings. -/
theorem specific_pairing_probability
  (total_students : ℕ)
  (non_participating : ℕ)
  (h1 : total_students = 32)
  (h2 : non_participating = 1)
  : (1 : ℚ) / (total_students - non_participating - 1) = 1 / 30 :=
by sorry

end NUMINAMATH_CALUDE_specific_pairing_probability_l31_3157


namespace NUMINAMATH_CALUDE_count_four_digit_integers_thousands_4_l31_3171

/-- The count of four-digit positive integers with the thousands digit 4 -/
def fourDigitIntegersWithThousands4 : ℕ :=
  (Finset.range 10).card * (Finset.range 10).card * (Finset.range 10).card

/-- Theorem stating that the count of four-digit positive integers with the thousands digit 4 is 1000 -/
theorem count_four_digit_integers_thousands_4 :
  fourDigitIntegersWithThousands4 = 1000 := by sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_thousands_4_l31_3171


namespace NUMINAMATH_CALUDE_min_comparisons_correct_l31_3191

/-- Represents a deck of cards numbered from 1 to n -/
def Deck (n : ℕ) := Fin n

/-- Checks if two numbers are consecutive -/
def are_consecutive (a b : ℕ) : Prop := (a + 1 = b) ∨ (b + 1 = a)

/-- The minimum number of comparisons needed to guarantee finding a consecutive pair -/
def min_comparisons (n : ℕ) := n - 2

theorem min_comparisons_correct (n : ℕ) (h : n ≥ 100) :
  ∀ (d : Deck n), 
    ∃ (f : Fin (min_comparisons n) → Deck n × Deck n),
      ∀ (g : Deck n × Deck n → Bool),
        (∀ (i j : Deck n), g (i, j) = true ↔ are_consecutive i.val j.val) →
        ∃ (i : Fin (min_comparisons n)), g (f i) = true :=
sorry

#check min_comparisons_correct

end NUMINAMATH_CALUDE_min_comparisons_correct_l31_3191


namespace NUMINAMATH_CALUDE_complex_imaginary_condition_l31_3138

theorem complex_imaginary_condition (a : ℝ) : 
  let z : ℂ := (1 - 3*Complex.I) * (a - Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_condition_l31_3138


namespace NUMINAMATH_CALUDE_childrens_home_total_l31_3178

theorem childrens_home_total (toddlers teenagers newborns : ℕ) : 
  teenagers = 5 * toddlers →
  toddlers = 6 →
  newborns = 4 →
  toddlers + teenagers + newborns = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_childrens_home_total_l31_3178


namespace NUMINAMATH_CALUDE_number_times_fifteen_equals_150_l31_3159

theorem number_times_fifteen_equals_150 :
  ∃ x : ℝ, 15 * x = 150 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_times_fifteen_equals_150_l31_3159


namespace NUMINAMATH_CALUDE_range_equivalence_l31_3136

/-- The function f(x) = -x³ + 3bx --/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + 3*b*x

/-- The theorem stating the equivalence between the range of f and the value of b --/
theorem range_equivalence (b : ℝ) :
  (∀ y ∈ Set.range (f b), y ∈ Set.Icc 0 1) ∧
  (∀ y ∈ Set.Icc 0 1, ∃ x ∈ Set.Icc 0 1, f b x = y) ↔
  b = (2 : ℝ)^(1/3) / 2 := by
sorry

end NUMINAMATH_CALUDE_range_equivalence_l31_3136


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_l31_3168

/-- Two terms are like terms if they have the same variables raised to the same powers. -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c, term1 x y = c * term2 x y

/-- The first term in our problem -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^(2*m) * y^m

/-- The second term in our problem -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := x^(4-n) * y^(n-1)

theorem like_terms_imply_sum (m n : ℕ) : 
  like_terms (term1 m) (term2 n) → m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_l31_3168


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l31_3165

theorem students_playing_neither_sport (total : ℕ) (hockey : ℕ) (basketball : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_basketball : basketball = 16)
  (h_both : both = 10) :
  total - (hockey + basketball - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l31_3165


namespace NUMINAMATH_CALUDE_hyperbola_chord_midpoint_l31_3180

/-- Given a hyperbola x²/a² - y²/b² = 1 where a, b > 0,
    the midpoint of any chord with slope 1 lies on the line x/a² - y/b² = 0 -/
theorem hyperbola_chord_midpoint (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  x^2 / a^2 - y^2 / b^2 = 1 →
  ∃ (m : ℝ), (x + m) / a^2 - (y + m) / b^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_chord_midpoint_l31_3180


namespace NUMINAMATH_CALUDE_unique_integer_solution_l31_3156

theorem unique_integer_solution : ∃! (n : ℤ), n + 10 > 11 ∧ -4*n > -12 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l31_3156


namespace NUMINAMATH_CALUDE_storks_and_birds_l31_3153

theorem storks_and_birds (initial_storks initial_birds new_birds : ℕ) :
  initial_storks = 6 →
  initial_birds = 2 →
  new_birds = 3 →
  initial_storks - (initial_birds + new_birds) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_and_birds_l31_3153


namespace NUMINAMATH_CALUDE_seedling_ratio_l31_3132

theorem seedling_ratio (first_day : ℕ) (total : ℕ) : 
  first_day = 200 → total = 1200 → 
  (total - first_day) / first_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_seedling_ratio_l31_3132


namespace NUMINAMATH_CALUDE_complex_modulus_proof_l31_3139

theorem complex_modulus_proof : 
  let z : ℂ := Complex.mk (3/4) (-5/6)
  ‖z‖ = Real.sqrt 127 / 12 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_proof_l31_3139


namespace NUMINAMATH_CALUDE_al2co3_3_weight_l31_3129

/-- The molecular weight of a compound given its composition and atomic weights -/
def molecular_weight (al_weight c_weight o_weight : ℝ) (num_moles : ℝ) : ℝ :=
  let co3_weight := c_weight + 3 * o_weight
  let al2co3_3_weight := 2 * al_weight + 3 * co3_weight
  num_moles * al2co3_3_weight

/-- Theorem stating the molecular weight of 6 moles of Al2(CO3)3 -/
theorem al2co3_3_weight : 
  molecular_weight 26.98 12.01 16.00 6 = 1403.94 := by
  sorry

end NUMINAMATH_CALUDE_al2co3_3_weight_l31_3129


namespace NUMINAMATH_CALUDE_min_value_of_f_l31_3169

noncomputable def f (x : ℝ) := x^2 + 2*x + 6/x + 9/x^2 + 4

theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 10 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l31_3169


namespace NUMINAMATH_CALUDE_quadratic_root_value_l31_3126

theorem quadratic_root_value (k : ℚ) : 
  ((-25 - Real.sqrt 369) / 12 : ℝ) ∈ {x : ℝ | 6 * x^2 + 25 * x + k = 0} → k = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l31_3126


namespace NUMINAMATH_CALUDE_xy_value_given_equation_l31_3181

theorem xy_value_given_equation :
  ∀ x y : ℝ, 2*x^2 + 2*x*y + y^2 - 6*x + 9 = 0 → x^y = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_given_equation_l31_3181


namespace NUMINAMATH_CALUDE_trajectory_is_line_with_equal_tangents_l31_3113

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 4
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 - 4 = (x - 3)^2 + (y - 2)^2 - 1

-- Define tangent length squared to O1
def tangent_length_sq_O1 (x y : ℝ) : ℝ := (x + 1)^2 + (y + 1)^2 - 4

-- Define tangent length squared to O2
def tangent_length_sq_O2 (x y : ℝ) : ℝ := (x - 3)^2 + (y - 2)^2 - 1

-- Theorem statement
theorem trajectory_is_line_with_equal_tangents :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, trajectory x y ↔ a * x + b * y + c = 0) ∧
    (∀ x y : ℝ, trajectory x y → tangent_length_sq_O1 x y = tangent_length_sq_O2 x y) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_with_equal_tangents_l31_3113


namespace NUMINAMATH_CALUDE_min_value_z_l31_3193

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := 4*x + y

/-- The feasible region defined by the constraints -/
def feasible_region (x y : ℝ) : Prop :=
  3*x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

/-- The theorem stating that the minimum value of z in the feasible region is 7 -/
theorem min_value_z :
  ∀ x y : ℝ, feasible_region x y → z x y ≥ 7 ∧ ∃ x₀ y₀ : ℝ, feasible_region x₀ y₀ ∧ z x₀ y₀ = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_z_l31_3193


namespace NUMINAMATH_CALUDE_dance_attendance_l31_3145

theorem dance_attendance (girls boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l31_3145


namespace NUMINAMATH_CALUDE_bell_ringing_problem_l31_3177

theorem bell_ringing_problem (S B : ℕ) : 
  S = (1/3 : ℚ) * B + 4 →
  B = 36 →
  S + B = 52 := by sorry

end NUMINAMATH_CALUDE_bell_ringing_problem_l31_3177


namespace NUMINAMATH_CALUDE_line_angle_and_triangle_conditions_l31_3116

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def l₁ : Line := { a := 2, b := -1, c := -10 }
def l₂ : Line := { a := 4, b := 3, c := -10 }
def l₃ (a : ℝ) : Line := { a := a, b := 2, c := -8 }

/-- The angle between two lines -/
def angle_between (l1 l2 : Line) : ℝ := sorry

/-- Whether three lines can form a triangle -/
def can_form_triangle (l1 l2 l3 : Line) : Prop := sorry

theorem line_angle_and_triangle_conditions :
  (angle_between l₁ l₂ = Real.arctan 2) ∧
  (∀ a : ℝ, ¬(can_form_triangle l₁ l₂ (l₃ a)) ↔ (a = -4 ∨ a = 8/3 ∨ a = 3)) := by sorry

end NUMINAMATH_CALUDE_line_angle_and_triangle_conditions_l31_3116


namespace NUMINAMATH_CALUDE_y_equals_negative_two_at_x_two_l31_3142

/-- A linear function y = kx - 1 where y decreases as x increases -/
structure DecreasingLinearFunction where
  k : ℝ
  h1 : k < 0

/-- The value of y when x = 2 for a decreasing linear function -/
def y_at_2 (f : DecreasingLinearFunction) : ℝ :=
  f.k * 2 - 1

/-- Theorem stating that y = -2 when x = 2 for a decreasing linear function -/
theorem y_equals_negative_two_at_x_two (f : DecreasingLinearFunction) :
  y_at_2 f = -2 :=
sorry

end NUMINAMATH_CALUDE_y_equals_negative_two_at_x_two_l31_3142


namespace NUMINAMATH_CALUDE_square_side_length_l31_3106

/-- Given a regular triangle and a square with specific perimeter conditions, 
    prove that the side length of the square is 8 cm. -/
theorem square_side_length 
  (triangle_perimeter : ℝ) 
  (total_perimeter : ℝ) 
  (h1 : triangle_perimeter = 46) 
  (h2 : total_perimeter = 78) : 
  (total_perimeter - triangle_perimeter) / 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l31_3106


namespace NUMINAMATH_CALUDE_det_sum_of_matrices_l31_3111

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 6; 2, 3]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 1, 0]

theorem det_sum_of_matrices : Matrix.det (A + B) = -3 := by sorry

end NUMINAMATH_CALUDE_det_sum_of_matrices_l31_3111


namespace NUMINAMATH_CALUDE_function_properties_l31_3197

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  Real.log ((3 - a) * Real.exp x + 1) - Real.log (3 * a) - 2 * x

theorem function_properties :
  (∃ (m : ℝ), m = 2 ∧ ∀ x : ℝ, x ≥ 0 → f x ≥ m) ∧
  (∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 → g a x₁ ≤ f x₂ - 2) →
    a ≥ 1 ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_function_properties_l31_3197


namespace NUMINAMATH_CALUDE_z_reciprocal_modulus_l31_3155

theorem z_reciprocal_modulus (i : ℂ) (z : ℂ) : 
  i^2 = -1 → 
  z = i + 2*i^2 + 3*i^3 + 4*i^4 + 5*i^5 + 6*i^6 + 7*i^7 + 8*i^8 → 
  Complex.abs (z⁻¹) = Real.sqrt 2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_z_reciprocal_modulus_l31_3155


namespace NUMINAMATH_CALUDE_c_value_l31_3186

theorem c_value (a b c : ℚ) : 
  8 = (2 / 100) * a → 
  2 = (8 / 100) * b → 
  c = b / a → 
  c = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_c_value_l31_3186


namespace NUMINAMATH_CALUDE_smallest_group_size_l31_3103

theorem smallest_group_size : ∃ n : ℕ, n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 6 = 5 ∧ 
  n % 8 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 
    (m % 3 = 2 ∧ m % 6 = 5 ∧ m % 8 = 7) → 
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_l31_3103


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l31_3164

-- Define the sets
def set1 : Set ℝ := {x | x * (x - 3) < 0}
def set2 : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem sufficient_but_not_necessary : set1 ⊆ set2 ∧ set1 ≠ set2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l31_3164


namespace NUMINAMATH_CALUDE_fred_initial_cards_l31_3152

/-- The number of baseball cards Keith bought from Fred -/
def cards_bought : ℕ := 22

/-- The number of baseball cards Fred has now -/
def cards_remaining : ℕ := 18

/-- The initial number of baseball cards Fred had -/
def initial_cards : ℕ := cards_bought + cards_remaining

theorem fred_initial_cards : initial_cards = 40 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_cards_l31_3152


namespace NUMINAMATH_CALUDE_power_of_product_l31_3194

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l31_3194


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l31_3108

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 4 * a + b = a * b) :
  (∃ (min : ℝ), min = 9 ∧ ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = x * y → x + y ≥ min) ∧
  (∀ x t : ℝ, t ∈ Set.Icc (-1) 3 → |x - a| + |x - b| ≥ t^2 - 2*t) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l31_3108


namespace NUMINAMATH_CALUDE_exact_three_primes_l31_3184

/-- The polynomial function f(n) = n^3 - 8n^2 + 20n - 13 -/
def f (n : ℕ) : ℤ := n^3 - 8*n^2 + 20*n - 13

/-- Predicate for primality -/
def isPrime (n : ℤ) : Prop := n > 1 ∧ (∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0))

theorem exact_three_primes : 
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n ∈ s, isPrime (f n) ∧ 
    ∀ n : ℕ, n > 0 → isPrime (f n) → n ∈ s :=
sorry

end NUMINAMATH_CALUDE_exact_three_primes_l31_3184


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l31_3147

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l31_3147


namespace NUMINAMATH_CALUDE_insufficient_comparisons_l31_3140

/-- Represents a comparison of three elements -/
structure TripleComparison (α : Type) where
  a : α
  b : α
  c : α

/-- The type of all possible orderings of n distinct elements -/
def Orderings (n : ℕ) := Fin n → Fin n

/-- The number of possible orderings for n distinct elements -/
def num_orderings (n : ℕ) : ℕ := n.factorial

/-- The maximum number of orderings that can be eliminated by a single triple comparison -/
def max_eliminated_by_comparison (n : ℕ) : ℕ := (n - 2).factorial

/-- The number of comparisons allowed -/
def num_comparisons : ℕ := 9

/-- The number of distinct elements to be ordered -/
def num_elements : ℕ := 5

/-- Theorem stating that the given number of comparisons is insufficient -/
theorem insufficient_comparisons :
  ∃ (remaining : ℕ), remaining > 1 ∧
  remaining ≤ num_orderings num_elements - num_comparisons * max_eliminated_by_comparison num_elements :=
sorry

end NUMINAMATH_CALUDE_insufficient_comparisons_l31_3140


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l31_3104

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l31_3104


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l31_3110

/-- Given a triangle with sides 6, 8, and 10 units, formed by the centers of
    three mutually externally tangent circles, the sum of the areas of these
    circles is 56π. -/
theorem sum_of_circle_areas (r s t : ℝ) : 
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  π * (r^2 + s^2 + t^2) = 56 * π :=
by sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l31_3110


namespace NUMINAMATH_CALUDE_tiffany_lives_l31_3135

/-- Calculate the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem: Given Tiffany's initial lives, lives lost, and lives gained,
    prove that her final number of lives is 56 -/
theorem tiffany_lives : final_lives 43 14 27 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_l31_3135


namespace NUMINAMATH_CALUDE_one_tetrahedron_formed_l31_3183

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the set of available triangles -/
def AvailableTriangles : Finset Triangle := sorry

/-- Checks if a set of four triangles can form a tetrahedron -/
def CanFormTetrahedron (t1 t2 t3 t4 : Triangle) : Prop := sorry

/-- Counts the number of tetrahedrons that can be formed -/
def CountTetrahedrons (triangles : Finset Triangle) : ℕ := sorry

/-- The main theorem stating that exactly one tetrahedron can be formed -/
theorem one_tetrahedron_formed :
  CountTetrahedrons AvailableTriangles = 1 := by sorry

end NUMINAMATH_CALUDE_one_tetrahedron_formed_l31_3183


namespace NUMINAMATH_CALUDE_sqrt_five_position_l31_3122

/-- Given a sequence where the square of the n-th term is 3n - 1, 
    prove that 2√5 is the 7th term of this sequence. -/
theorem sqrt_five_position (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ n, a n ^ 2 = 3 * n - 1) : 
  a 7 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_position_l31_3122


namespace NUMINAMATH_CALUDE_number_of_girls_in_group_l31_3124

theorem number_of_girls_in_group (girls_avg_weight : ℝ) (boys_avg_weight : ℝ) 
  (total_avg_weight : ℝ) (num_boys : ℕ) (total_students : ℕ) :
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  num_boys = 5 →
  total_students = 10 →
  total_avg_weight = 50 →
  ∃ (num_girls : ℕ), num_girls = 5 ∧ 
    (girls_avg_weight * num_girls + boys_avg_weight * num_boys) / total_students = total_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_number_of_girls_in_group_l31_3124


namespace NUMINAMATH_CALUDE_proportional_function_m_value_l31_3176

/-- A proportional function passing through a specific point -/
def proportional_function_through_point (k m : ℝ) : Prop :=
  4 * 2 = 3 - m

/-- Theorem: If the proportional function y = 4x passes through (2, 3-m), then m = -5 -/
theorem proportional_function_m_value (m : ℝ) :
  proportional_function_through_point 4 m → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_m_value_l31_3176


namespace NUMINAMATH_CALUDE_bird_count_theorem_l31_3148

/-- The number of birds on a fence after a series of additions and removals --/
def final_bird_count (initial : ℕ) (first_add : ℕ) (first_remove : ℕ) (second_add : ℕ) (third_add : ℚ) : ℚ :=
  let T : ℕ := initial + first_add
  let W : ℕ := T - first_remove + second_add
  (W : ℚ) / 2 + third_add

/-- Theorem stating the final number of birds on the fence --/
theorem bird_count_theorem : 
  final_bird_count 12 8 5 3 (5/2) = 23/2 := by sorry

end NUMINAMATH_CALUDE_bird_count_theorem_l31_3148


namespace NUMINAMATH_CALUDE_inequality_proof_l31_3188

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c = 1) : 
  (a + 1/a) * (b + 1/b) * (c + 1/c) ≥ 1000/27 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l31_3188


namespace NUMINAMATH_CALUDE_selling_price_calculation_l31_3120

def calculate_selling_price (initial_price maintenance_cost repair_cost transportation_cost : ℝ)
  (tax_rate currency_loss_rate depreciation_rate profit_margin : ℝ) : ℝ :=
  let total_expenses := initial_price + maintenance_cost + repair_cost + transportation_cost
  let after_tax := total_expenses * (1 + tax_rate)
  let after_currency_loss := after_tax * (1 - currency_loss_rate)
  let after_depreciation := after_currency_loss * (1 - depreciation_rate)
  after_depreciation * (1 + profit_margin)

theorem selling_price_calculation :
  calculate_selling_price 10000 2000 5000 1000 0.1 0.05 0.15 0.5 = 23982.75 :=
by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l31_3120


namespace NUMINAMATH_CALUDE_defective_tubes_count_l31_3137

/-- The probability of selecting two defective tubes without replacement -/
def prob_two_defective : ℝ := 0.05263157894736842

/-- The total number of picture tubes in the consignment -/
def total_tubes : ℕ := 20

/-- The number of defective picture tubes in the consignment -/
def num_defective : ℕ := 5

theorem defective_tubes_count :
  (num_defective : ℝ) / total_tubes * ((num_defective - 1) : ℝ) / (total_tubes - 1) = prob_two_defective := by
  sorry

end NUMINAMATH_CALUDE_defective_tubes_count_l31_3137


namespace NUMINAMATH_CALUDE_valentines_day_cards_l31_3114

theorem valentines_day_cards (boys girls : ℕ) : 
  boys * girls = boys + girls + 18 → boys * girls = 40 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_cards_l31_3114


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l31_3134

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 6 →
  picture_book_shelves = 4 →
  total_books = 54 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l31_3134


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l31_3150

-- Define a quadratic function
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

-- State the theorem
theorem quadratic_function_uniqueness (g : ℝ → ℝ) :
  (∃ a b : ℝ, g = QuadraticFunction a b) →  -- g is quadratic
  g 1 = 1 →                                 -- g(1) = 1
  g (-1) = 5 →                              -- g(-1) = 5
  g 0 = 0 →                                 -- g(0) = 0 (passes through origin)
  g = QuadraticFunction 3 (-2) :=           -- g(x) = 3x^2 - 2x
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l31_3150


namespace NUMINAMATH_CALUDE_triangle_properties_l31_3160

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c →
  Real.sin A * Real.sin B = (Real.cos (C / 2))^2 →
  ((a^2 + b^2 - c^2) / 4 + (c^2 * (Real.cos (C / 2))^2)) = 7 →
  A = Real.pi / 6 ∧ B = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l31_3160


namespace NUMINAMATH_CALUDE_operation_on_number_l31_3115

theorem operation_on_number (x : ℝ) : x^2 = 25 → 2*x = x/5 + 9 := by
  sorry

end NUMINAMATH_CALUDE_operation_on_number_l31_3115


namespace NUMINAMATH_CALUDE_sqrt_y_fourth_power_l31_3123

theorem sqrt_y_fourth_power (y : ℝ) : (Real.sqrt y)^4 = 256 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_y_fourth_power_l31_3123


namespace NUMINAMATH_CALUDE_books_in_childrens_section_l31_3102

theorem books_in_childrens_section
  (initial_books : ℕ)
  (books_left : ℕ)
  (history_books : ℕ)
  (fiction_books : ℕ)
  (misplaced_books : ℕ)
  (h1 : initial_books = 51)
  (h2 : books_left = 16)
  (h3 : history_books = 12)
  (h4 : fiction_books = 19)
  (h5 : misplaced_books = 4) :
  initial_books + misplaced_books - history_books - fiction_books - books_left = 8 :=
by sorry

end NUMINAMATH_CALUDE_books_in_childrens_section_l31_3102


namespace NUMINAMATH_CALUDE_expression_value_l31_3125

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- absolute value of m is 2
  : (a + b) / (4 * m) + m^2 - 3 * c * d = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l31_3125


namespace NUMINAMATH_CALUDE_ramanujan_hardy_game_l31_3163

theorem ramanujan_hardy_game (h r : ℂ) : 
  h * r = 48 - 12*I ∧ h = 6 + 2*I → r = 39/5 - 21/5*I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_game_l31_3163


namespace NUMINAMATH_CALUDE_triangle_sine_problem_l31_3192

theorem triangle_sine_problem (D E F : ℝ) (h_area : (1/2) * D * E * Real.sin F = 72) 
  (h_geometric_mean : Real.sqrt (D * E) = 15) : Real.sin F = 16/25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_problem_l31_3192


namespace NUMINAMATH_CALUDE_total_mileage_scientific_notation_l31_3100

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The total mileage of national expressways -/
def totalMileage : ℕ := 108000

/-- Theorem: The scientific notation of the total mileage is 1.08 × 10^5 -/
theorem total_mileage_scientific_notation :
  ∃ (sn : ScientificNotation), sn.coefficient = 1.08 ∧ sn.exponent = 5 ∧ (sn.coefficient * (10 : ℝ) ^ sn.exponent = totalMileage) :=
sorry

end NUMINAMATH_CALUDE_total_mileage_scientific_notation_l31_3100


namespace NUMINAMATH_CALUDE_largest_fraction_l31_3117

theorem largest_fraction (a b c d e : ℚ) 
  (ha : a = 3/10) (hb : b = 9/20) (hc : c = 12/25) (hd : d = 27/50) (he : e = 49/100) :
  d = max a (max b (max c (max d e))) :=
sorry

end NUMINAMATH_CALUDE_largest_fraction_l31_3117


namespace NUMINAMATH_CALUDE_contradiction_proof_l31_3109

theorem contradiction_proof (a b : ℕ) : a < 2 → b < 2 → a + b < 3 := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l31_3109


namespace NUMINAMATH_CALUDE_gcd_of_42_77_105_l31_3196

theorem gcd_of_42_77_105 : Nat.gcd 42 (Nat.gcd 77 105) = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_of_42_77_105_l31_3196


namespace NUMINAMATH_CALUDE_line_direction_vector_value_l31_3118

def point := ℝ × ℝ

def direction_vector (a : ℝ) : point := (a, -2)

def line_passes_through (p1 p2 : point) (v : point) : Prop :=
  ∃ t : ℝ, p2 = (p1.1 + t * v.1, p1.2 + t * v.2)

theorem line_direction_vector_value :
  ∀ a : ℝ,
  line_passes_through (-3, 6) (2, -1) (direction_vector a) →
  a = 10/7 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_value_l31_3118


namespace NUMINAMATH_CALUDE_symmetry_example_l31_3166

/-- A point in 3D space is represented by its x, y, and z coordinates. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are the same,
    and their y and z coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = -q.y ∧ p.z = -q.z

/-- The theorem states that the point (-2, -1, -4) is symmetric to the point (-2, 1, 4)
    with respect to the x-axis. -/
theorem symmetry_example : 
  symmetric_wrt_x_axis (Point3D.mk (-2) 1 4) (Point3D.mk (-2) (-1) (-4)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_example_l31_3166


namespace NUMINAMATH_CALUDE_combined_liquid_fraction_l31_3182

/-- Represents the capacity of a beaker -/
structure Beaker where
  capacity : ℝ
  filled : ℝ
  density : ℝ

/-- The problem setup -/
def problemSetup : Prop := ∃ (small large third : Beaker),
  -- Small beaker conditions
  small.filled = (1/2) * small.capacity ∧
  small.density = 1.025 ∧
  -- Large beaker conditions
  large.capacity = 5 * small.capacity ∧
  large.filled = (1/5) * large.capacity ∧
  large.density = 1 ∧
  -- Third beaker conditions
  third.capacity = (1/2) * large.capacity ∧
  third.filled = (3/4) * third.capacity ∧
  third.density = 0.85

/-- The theorem to prove -/
theorem combined_liquid_fraction (h : problemSetup) :
  ∃ (small large third : Beaker),
  (large.filled + small.filled + third.filled) / large.capacity = 27/40 := by
  sorry

end NUMINAMATH_CALUDE_combined_liquid_fraction_l31_3182


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l31_3174

/-- The wholesale price of a pair of pants -/
def wholesale_price : ℝ := 20

/-- The retail price of a pair of pants -/
def retail_price : ℝ := 36

/-- The markup factor applied to the wholesale price -/
def markup_factor : ℝ := 1.8

theorem wholesale_price_calculation :
  wholesale_price * markup_factor = retail_price :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l31_3174


namespace NUMINAMATH_CALUDE_basketball_lineups_l31_3189

def total_players : ℕ := 12
def players_per_lineup : ℕ := 5
def point_guards_per_lineup : ℕ := 1

def number_of_lineups : ℕ :=
  total_players * (Nat.choose (total_players - 1) (players_per_lineup - 1))

theorem basketball_lineups :
  number_of_lineups = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineups_l31_3189


namespace NUMINAMATH_CALUDE_initial_distance_between_trucks_l31_3195

/-- Theorem: Initial distance between two trucks
Given:
- Two trucks X and Y traveling in the same direction
- Truck X's speed is 47 mph
- Truck Y's speed is 53 mph
- It takes 3 hours for Truck Y to overtake and be 5 miles ahead of Truck X
Prove: The initial distance between Truck X and Truck Y is 23 miles
-/
theorem initial_distance_between_trucks
  (speed_x : ℝ)
  (speed_y : ℝ)
  (overtake_time : ℝ)
  (ahead_distance : ℝ)
  (h1 : speed_x = 47)
  (h2 : speed_y = 53)
  (h3 : overtake_time = 3)
  (h4 : ahead_distance = 5)
  : ∃ (initial_distance : ℝ),
    initial_distance = (speed_y - speed_x) * overtake_time + ahead_distance :=
by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_trucks_l31_3195


namespace NUMINAMATH_CALUDE_commodity_price_problem_l31_3170

theorem commodity_price_problem (price1 price2 : ℕ) : 
  price1 + price2 = 827 →
  price1 = price2 + 127 →
  price1 = 477 := by
sorry

end NUMINAMATH_CALUDE_commodity_price_problem_l31_3170


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_means_l31_3173

/-- A linear regression line for a set of data points -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The sample means of a dataset -/
structure SampleMeans where
  x_mean : ℝ
  y_mean : ℝ

/-- Theorem stating that two regression lines for the same dataset intersect at the sample means -/
theorem regression_lines_intersect_at_means 
  (m n : RegressionLine) (means : SampleMeans) : 
  ∃ (x y : ℝ), 
    x = means.x_mean ∧ 
    y = means.y_mean ∧ 
    y = m.slope * x + m.intercept ∧ 
    y = n.slope * x + n.intercept := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersect_at_means_l31_3173


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l31_3175

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 2) :
  1 / x + 2 / y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ = 2 ∧ 1 / x₀ + 2 / y₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l31_3175


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l31_3158

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 20)
  (h2 : selling_price = 25) :
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l31_3158


namespace NUMINAMATH_CALUDE_HNO3_calculation_l31_3190

-- Define the chemical equation
def chemical_equation : String := "CaO + 2 HNO₃ → Ca(NO₃)₂ + H₂O"

-- Define the initial amount of CaO in moles
def initial_CaO : ℝ := 7

-- Define the stoichiometric ratio of HNO₃ to CaO
def stoichiometric_ratio : ℝ := 2

-- Define atomic weights
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Theorem to prove
theorem HNO3_calculation (chemical_equation : String) (initial_CaO : ℝ) 
  (stoichiometric_ratio : ℝ) (atomic_weight_H : ℝ) (atomic_weight_N : ℝ) 
  (atomic_weight_O : ℝ) :
  let moles_HNO3 : ℝ := initial_CaO * stoichiometric_ratio
  let molecular_weight_HNO3 : ℝ := atomic_weight_H + atomic_weight_N + 3 * atomic_weight_O
  (moles_HNO3 = 14 ∧ molecular_weight_HNO3 = 63.02) :=
by
  sorry

end NUMINAMATH_CALUDE_HNO3_calculation_l31_3190


namespace NUMINAMATH_CALUDE_simplify_and_sum_coefficients_l31_3107

theorem simplify_and_sum_coefficients (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (15*d + 11 + 18*d^2) + (3*d + 2) = a*d^2 + b*d + c ∧ a + b + c = 49 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_coefficients_l31_3107


namespace NUMINAMATH_CALUDE_john_swimming_laps_l31_3161

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem john_swimming_laps :
  base7ToBase10 [3, 1, 2, 5] = 1823 := by
  sorry

end NUMINAMATH_CALUDE_john_swimming_laps_l31_3161


namespace NUMINAMATH_CALUDE_beach_population_l31_3167

theorem beach_population (initial_group : ℕ) (joined : ℕ) (left : ℕ) : 
  initial_group = 3 → joined = 100 → left = 40 → 
  initial_group + joined - left = 63 := by
sorry

end NUMINAMATH_CALUDE_beach_population_l31_3167


namespace NUMINAMATH_CALUDE_circular_garden_radius_l31_3179

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2 → r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l31_3179


namespace NUMINAMATH_CALUDE_required_third_subject_score_l31_3149

def average_score_two_subjects : ℝ := 88
def target_average_three_subjects : ℝ := 90
def number_of_subjects : ℕ := 3

theorem required_third_subject_score :
  let total_score_two_subjects := average_score_two_subjects * 2
  let total_score_three_subjects := target_average_three_subjects * number_of_subjects
  total_score_three_subjects - total_score_two_subjects = 94 := by
  sorry

end NUMINAMATH_CALUDE_required_third_subject_score_l31_3149
