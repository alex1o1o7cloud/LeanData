import Mathlib

namespace brownie_pieces_count_l3167_316792

/-- Represents the dimensions of a rectangular shape -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of smaller rectangles that can fit into a larger rectangle -/
def number_of_pieces (tray : Dimensions) (piece : Dimensions) : ℕ :=
  (area tray) / (area piece)

theorem brownie_pieces_count :
  let tray := Dimensions.mk 24 16
  let piece := Dimensions.mk 2 2
  number_of_pieces tray piece = 96 := by
  sorry

end brownie_pieces_count_l3167_316792


namespace sqrt_of_sqrt_81_l3167_316727

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 ∨ Real.sqrt (Real.sqrt 81) = -3 := by
  sorry

end sqrt_of_sqrt_81_l3167_316727


namespace last_digit_of_power_of_two_plus_one_l3167_316701

theorem last_digit_of_power_of_two_plus_one (n : ℕ) (h : n ≥ 2) :
  (2^(2^n) + 1) % 10 = 7 := by
  sorry

end last_digit_of_power_of_two_plus_one_l3167_316701


namespace temperature_difference_l3167_316782

theorem temperature_difference (M L N : ℝ) : 
  (M = L + N) →  -- Minneapolis is N degrees warmer than St. Louis at noon
  (|((L + N) - 6) - (L + 4)| = 3) →  -- Temperature difference at 5:00 PM
  (N = 13 ∨ N = 7) ∧ (13 * 7 = 91) :=
by sorry

end temperature_difference_l3167_316782


namespace zero_of_f_l3167_316716

-- Define the function f(x) = 2x + 7
def f (x : ℝ) : ℝ := 2 * x + 7

-- Theorem stating that the zero of f(x) is -7/2
theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -7/2 := by
sorry

end zero_of_f_l3167_316716


namespace raindrop_probability_l3167_316760

/-- The probability of a raindrop landing on the third slope of a triangular pyramid roof -/
theorem raindrop_probability (α β : Real) : 
  -- The roof is a triangular pyramid with all plane angles at the vertex being right angles
  -- The red slope is inclined at an angle α to the horizontal
  -- The blue slope is inclined at an angle β to the horizontal
  -- We assume 0 ≤ α ≤ π/2 and 0 ≤ β ≤ π/2 to ensure valid angles
  0 ≤ α ∧ α ≤ π/2 ∧ 0 ≤ β ∧ β ≤ π/2 →
  -- The probability of a raindrop landing on the green slope
  ∃ (p : Real), p = 1 - (Real.cos α)^2 - (Real.cos β)^2 ∧ 0 ≤ p ∧ p ≤ 1 :=
by sorry

end raindrop_probability_l3167_316760


namespace percentage_of_cat_owners_l3167_316740

def total_students : ℕ := 300
def cat_owners : ℕ := 45

theorem percentage_of_cat_owners : 
  (cat_owners : ℚ) / (total_students : ℚ) * 100 = 15 := by
  sorry

end percentage_of_cat_owners_l3167_316740


namespace circle_and_line_intersection_l3167_316784

-- Define the circle C
def circle_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + a = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y - 3 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define perpendicularity of vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem circle_and_line_intersection (a : ℝ) :
  (∃ (x y : ℝ), circle_C x y a ∧ line_l x y) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ a ∧ line_l x₁ y₁ ∧
    circle_C x₂ y₂ a ∧ line_l x₂ y₂ ∧
    perpendicular (x₁, y₁) (x₂, y₂)) →
  (∀ (x y : ℝ), line_m x y ↔ (x = -2 ∧ y = 1) ∨ (x + y + 1 = 0)) ∧
  a = -18 := by sorry

end circle_and_line_intersection_l3167_316784


namespace sufficient_not_necessary_condition_l3167_316769

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, b > a ∧ a > 0 → 1/a > 1/b) ∧
  ¬(∀ a b : ℝ, 1/a > 1/b → b > a ∧ a > 0) :=
by sorry

end sufficient_not_necessary_condition_l3167_316769


namespace side_to_perimeter_ratio_l3167_316724

/-- Represents a square garden -/
structure SquareGarden where
  side_length : ℝ

/-- Calculate the perimeter of a square garden -/
def perimeter (g : SquareGarden) : ℝ := 4 * g.side_length

/-- Theorem stating the ratio of side length to perimeter for a 15-foot square garden -/
theorem side_to_perimeter_ratio (g : SquareGarden) (h : g.side_length = 15) :
  g.side_length / perimeter g = 1 / 4 := by
  sorry

end side_to_perimeter_ratio_l3167_316724


namespace complex_modulus_problem_l3167_316749

theorem complex_modulus_problem (z : ℂ) : 
  z = ((1 - I) * (2 - I)) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l3167_316749


namespace negation_of_universal_quantifier_l3167_316766

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x ≥ Real.sqrt 2 → x^2 ≥ 2) ↔ (∃ x : ℝ, x ≥ Real.sqrt 2 ∧ x^2 < 2) :=
by sorry

end negation_of_universal_quantifier_l3167_316766


namespace linear_regression_not_guaranteed_point_l3167_316712

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  dataPoints : List Point

/-- Checks if a point is in the list of data points -/
def isDataPoint (p : Point) (lr : LinearRegression) : Prop :=
  p ∈ lr.dataPoints

/-- Theorem: The linear regression line is not guaranteed to pass through (6.5, 8) -/
theorem linear_regression_not_guaranteed_point (lr : LinearRegression) 
  (h1 : isDataPoint ⟨2, 3⟩ lr)
  (h2 : isDataPoint ⟨5, 7⟩ lr)
  (h3 : isDataPoint ⟨8, 9⟩ lr)
  (h4 : isDataPoint ⟨11, 13⟩ lr) :
  ¬ ∀ (regression_line : Point → Prop), 
    (∀ p, isDataPoint p lr → regression_line p) → 
    regression_line ⟨6.5, 8⟩ :=
by
  sorry

end linear_regression_not_guaranteed_point_l3167_316712


namespace work_completion_time_l3167_316711

theorem work_completion_time 
  (days_a : ℝ) 
  (days_b : ℝ) 
  (h1 : days_a = 12) 
  (h2 : days_b = 24) : 
  1 / (1 / days_a + 1 / days_b) = 8 := by
  sorry

end work_completion_time_l3167_316711


namespace initial_blue_balls_l3167_316765

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 15 → removed = 3 → prob = 1/3 → 
  ∃ (initial_blue : ℕ), 
    initial_blue = 7 ∧ 
    (initial_blue - removed : ℚ) / (total - removed) = prob :=
by sorry

end initial_blue_balls_l3167_316765


namespace largest_solution_and_ratio_l3167_316743

theorem largest_solution_and_ratio (x a b c d : ℤ) : 
  (7 * x / 5 + 3 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (a = -15 ∧ b = 1 ∧ c = 785 ∧ d = 14) →
  (x = (-15 + Real.sqrt 785) / 14 ∧ a * c * d / b = -164850) := by
  sorry

end largest_solution_and_ratio_l3167_316743


namespace square_sum_of_solution_l3167_316739

theorem square_sum_of_solution (x y : ℝ) : 
  x * y = 8 → 
  x^2 * y + x * y^2 + x + y = 80 → 
  x^2 + y^2 = 5104 / 81 := by
sorry

end square_sum_of_solution_l3167_316739


namespace arithmetic_sequence_fifth_term_l3167_316733

/-- Given an arithmetic sequence with the first four terms x^2 + 2y, x^2 - 2y, x+y, and x-y,
    the fifth term of the sequence is x - 5y. -/
theorem arithmetic_sequence_fifth_term 
  (x y : ℝ) 
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x^2 + 2*y)
  (h2 : seq 1 = x^2 - 2*y)
  (h3 : seq 2 = x + y)
  (h4 : seq 3 = x - y)
  (h_arithmetic : ∀ n, seq (n + 1) - seq n = seq 1 - seq 0) :
  seq 4 = x - 5*y :=
sorry

end arithmetic_sequence_fifth_term_l3167_316733


namespace inclination_angle_range_l3167_316761

open Set

-- Define the line equation
def line_equation (x y : ℝ) (α : ℝ) : Prop :=
  x * Real.sin α + y + 2 = 0

-- Define the range of the inclination angle
def inclination_range : Set ℝ :=
  Icc 0 (Real.pi / 4) ∪ Ico (3 * Real.pi / 4) Real.pi

-- Theorem statement
theorem inclination_angle_range :
  ∀ α, (∃ x y, line_equation x y α) → α ∈ inclination_range :=
sorry

end inclination_angle_range_l3167_316761


namespace coaches_in_conference_l3167_316718

theorem coaches_in_conference (rowers : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) 
  (h1 : rowers = 60)
  (h2 : votes_per_rower = 3)
  (h3 : votes_per_coach = 5) :
  (rowers * votes_per_rower) / votes_per_coach = 36 :=
by sorry

end coaches_in_conference_l3167_316718


namespace nail_trimming_sounds_l3167_316768

/-- Represents the number of customers --/
def num_customers : Nat := 3

/-- Represents the number of appendages per customer --/
def appendages_per_customer : Nat := 4

/-- Represents the number of nails per appendage --/
def nails_per_appendage : Nat := 4

/-- Calculates the total number of nail trimming sounds --/
def total_nail_sounds : Nat :=
  num_customers * appendages_per_customer * nails_per_appendage

/-- Theorem stating that the total number of nail trimming sounds is 48 --/
theorem nail_trimming_sounds :
  total_nail_sounds = 48 := by
  sorry

end nail_trimming_sounds_l3167_316768


namespace satellite_selection_probabilities_l3167_316771

/-- The number of geostationary Earth orbit (GEO) satellites -/
def num_geo : ℕ := 3

/-- The number of inclined geosynchronous orbit (IGSO) satellites -/
def num_igso : ℕ := 3

/-- The total number of satellites to select -/
def num_select : ℕ := 2

/-- The probability of selecting exactly one GEO satellite and one IGSO satellite -/
def prob_one_geo_one_igso : ℚ := 3/5

/-- The probability of selecting at least one IGSO satellite -/
def prob_at_least_one_igso : ℚ := 4/5

theorem satellite_selection_probabilities :
  (num_geo = 3 ∧ num_igso = 3 ∧ num_select = 2) →
  (prob_one_geo_one_igso = 3/5 ∧ prob_at_least_one_igso = 4/5) :=
by sorry

end satellite_selection_probabilities_l3167_316771


namespace aprons_to_sew_is_49_l3167_316735

def total_aprons : ℕ := 150
def aprons_sewn_initially : ℕ := 13

def aprons_sewn_today (initial : ℕ) : ℕ := 3 * initial

def remaining_aprons (total sewn : ℕ) : ℕ := total - sewn

def aprons_to_sew_tomorrow (remaining : ℕ) : ℕ := remaining / 2

theorem aprons_to_sew_is_49 : 
  aprons_to_sew_tomorrow (remaining_aprons total_aprons (aprons_sewn_initially + aprons_sewn_today aprons_sewn_initially)) = 49 := by
  sorry

end aprons_to_sew_is_49_l3167_316735


namespace nearest_gardeners_to_flower_l3167_316785

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Represents a gardener -/
structure Gardener where
  position : Point

/-- Represents a flower -/
structure Flower where
  position : Point

/-- Theorem: The three nearest gardeners to a flower in the top-left quarter
    of a 2x2 grid are those at the top-left, top-right, and bottom-left corners -/
theorem nearest_gardeners_to_flower 
  (gardenerA : Gardener) 
  (gardenerB : Gardener)
  (gardenerC : Gardener)
  (gardenerD : Gardener)
  (flower : Flower)
  (h1 : gardenerA.position = ⟨0, 2⟩)
  (h2 : gardenerB.position = ⟨2, 2⟩)
  (h3 : gardenerC.position = ⟨0, 0⟩)
  (h4 : gardenerD.position = ⟨2, 0⟩)
  (h5 : 0 < flower.position.x ∧ flower.position.x < 1)
  (h6 : 1 < flower.position.y ∧ flower.position.y < 2) :
  squaredDistance flower.position gardenerA.position < squaredDistance flower.position gardenerD.position ∧
  squaredDistance flower.position gardenerB.position < squaredDistance flower.position gardenerD.position ∧
  squaredDistance flower.position gardenerC.position < squaredDistance flower.position gardenerD.position :=
by sorry

end nearest_gardeners_to_flower_l3167_316785


namespace number_greater_than_one_eighth_l3167_316741

theorem number_greater_than_one_eighth : ∃ x : ℝ, x = 1/8 + 0.0020000000000000018 ∧ x = 0.1270000000000000018 := by
  sorry

end number_greater_than_one_eighth_l3167_316741


namespace range_of_m_l3167_316723

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + y = 1)
  (h_ineq : ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 4 * x^2 + y^2 + Real.sqrt (x * y) - m < 0) :
  m > 17/16 := by
sorry

end range_of_m_l3167_316723


namespace complex_division_l3167_316776

theorem complex_division (i : ℂ) (h : i * i = -1) : (3 - 4*i) / i = -4 - 3*i := by
  sorry

end complex_division_l3167_316776


namespace penny_collection_difference_l3167_316794

theorem penny_collection_difference (cassandra_pennies james_pennies : ℕ) : 
  cassandra_pennies = 5000 →
  james_pennies < cassandra_pennies →
  cassandra_pennies + james_pennies = 9724 →
  cassandra_pennies - james_pennies = 276 := by
sorry

end penny_collection_difference_l3167_316794


namespace cousins_in_rooms_l3167_316777

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 4 cousins and 4 identical rooms -/
theorem cousins_in_rooms : distribute 4 4 = 15 := by
  sorry

end cousins_in_rooms_l3167_316777


namespace caterpillars_on_tree_l3167_316746

theorem caterpillars_on_tree (initial : ℕ) (hatched : ℕ) (left : ℕ) : 
  initial = 14 → hatched = 4 → left = 8 → 
  initial + hatched - left = 10 := by sorry

end caterpillars_on_tree_l3167_316746


namespace sheridan_cats_l3167_316762

def current_cats : ℕ := sorry
def needed_cats : ℕ := 32
def total_cats : ℕ := 43

theorem sheridan_cats : current_cats = 11 := by
  sorry

end sheridan_cats_l3167_316762


namespace library_book_distribution_l3167_316786

/-- The number of ways to distribute books between the library and checked out -/
def distributeBooks (total : ℕ) (minInLibrary : ℕ) (minCheckedOut : ℕ) : ℕ :=
  (total - minInLibrary - minCheckedOut + 1)

/-- Theorem: There are 6 ways to distribute 10 books with at least 2 in the library and 3 checked out -/
theorem library_book_distribution :
  distributeBooks 10 2 3 = 6 := by
  sorry

end library_book_distribution_l3167_316786


namespace original_number_proof_l3167_316753

theorem original_number_proof (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end original_number_proof_l3167_316753


namespace range_of_m_plus_n_l3167_316710

/-- Given a function f(x) = me^x + x^2 + nx where the set of roots of f and f∘f are equal and non-empty,
    prove that the range of m + n is [0, 4). -/
theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, m * Real.exp x + x^2 + n * x = 0) →
  {x | m * Real.exp x + x^2 + n * x = 0} = {x | m * Real.exp (m * Real.exp x + x^2 + n * x) + 
    (m * Real.exp x + x^2 + n * x)^2 + n * (m * Real.exp x + x^2 + n * x) = 0} →
  m + n ∈ Set.Icc 0 4 ∧ ¬(m + n = 4) :=
by sorry

end range_of_m_plus_n_l3167_316710


namespace inequality_equivalence_l3167_316728

theorem inequality_equivalence (x : ℝ) : 
  (5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10) ↔ (3 < x ∧ x < 60 / 19) :=
by sorry

end inequality_equivalence_l3167_316728


namespace angle_bisector_m_abs_z_over_one_plus_i_l3167_316748

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m - 6) (m^2 - 2*m - 15)

-- Theorem 1: When z is on the angle bisector of the first and third quadrants, m = -3
theorem angle_bisector_m (m : ℝ) : z m = Complex.mk (z m).re (z m).re → m = -3 := by
  sorry

-- Theorem 2: When m = -1, |z/(1+i)| = √74
theorem abs_z_over_one_plus_i : Complex.abs (z (-1) / (1 + Complex.I)) = Real.sqrt 74 := by
  sorry

end angle_bisector_m_abs_z_over_one_plus_i_l3167_316748


namespace sum_removal_equals_half_l3167_316790

theorem sum_removal_equals_half :
  let original_sum := (1 : ℚ) / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  let removed_terms := 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  original_sum - removed_terms = 1 / 2 := by
sorry

end sum_removal_equals_half_l3167_316790


namespace all_inequalities_true_l3167_316729

theorem all_inequalities_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  (x + z > y + z) ∧
  (x - 2*z > y - 2*z) ∧
  (x*z^2 > y*z^2) ∧
  (x/z > y/z) ∧
  (x - z^2 > y - z^2) := by
  sorry

end all_inequalities_true_l3167_316729


namespace product_of_sums_l3167_316738

theorem product_of_sums (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := by
  sorry

end product_of_sums_l3167_316738


namespace cubic_root_sum_l3167_316734

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 20*r^2 + 18*r - 7 = 0 →
  s^3 - 20*s^2 + 18*s - 7 = 0 →
  t^3 - 20*t^2 + 18*t - 7 = 0 →
  (r / ((1/r) + s*t)) + (s / ((1/s) + t*r)) + (t / ((1/t) + r*s)) = 91/2 := by
sorry

end cubic_root_sum_l3167_316734


namespace line_properties_l3167_316797

/-- A line passing through point A(4, -1) with equal intercepts on x and y axes --/
def line_with_equal_intercepts : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 = 3) ∨ (p.1 + 4 * p.2 = 0)}

/-- The point A(4, -1) --/
def point_A : ℝ × ℝ := (4, -1)

/-- Theorem stating that the line passes through point A and has equal intercepts --/
theorem line_properties :
  point_A ∈ line_with_equal_intercepts ∧
  ∃ a : ℝ, (a, 0) ∈ line_with_equal_intercepts ∧ (0, a) ∈ line_with_equal_intercepts :=
by sorry

end line_properties_l3167_316797


namespace fraction_equality_y_value_l3167_316772

theorem fraction_equality_y_value (a b c d y : ℚ) 
  (h1 : a ≠ b) 
  (h2 : a ≠ 0) 
  (h3 : c ≠ d) 
  (h4 : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) := by
sorry

end fraction_equality_y_value_l3167_316772


namespace f_at_six_l3167_316773

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

-- Theorem stating that f(6) = 3658
theorem f_at_six : f 6 = 3658 := by sorry

end f_at_six_l3167_316773


namespace magnitude_relationship_l3167_316758

theorem magnitude_relationship (x : ℝ) (h : 0 < x ∧ x < π/4) :
  let A := Real.cos (x^(Real.sin (x^(Real.sin x))))
  let B := Real.sin (x^(Real.cos (x^(Real.sin x))))
  let C := Real.cos (x^(Real.sin (x * x^(Real.cos x))))
  B < A ∧ A < C := by
  sorry

end magnitude_relationship_l3167_316758


namespace expression_value_l3167_316755

theorem expression_value (a b c : ℤ) (ha : a = 10) (hb : b = 15) (hc : c = 3) :
  (a - (b - c)) - ((a - b) + c) = 0 := by
  sorry

end expression_value_l3167_316755


namespace minimum_value_of_expression_l3167_316731

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem minimum_value_of_expression (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k : ℕ, a k > 0) →
  a m * a n = 4 * (a 2)^2 →
  (2 : ℝ) / m + 1 / (2 * n) ≥ 3 / 4 :=
sorry

end minimum_value_of_expression_l3167_316731


namespace scientific_notation_of_41800000000_l3167_316745

theorem scientific_notation_of_41800000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 41800000000 = a * (10 : ℝ) ^ n ∧ a = 4.18 ∧ n = 10 := by
  sorry

end scientific_notation_of_41800000000_l3167_316745


namespace sin_two_phi_l3167_316779

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_two_phi_l3167_316779


namespace ladder_construction_theorem_l3167_316789

/-- Represents the ladder construction problem --/
def LadderProblem (totalWood rungeLength rungSpacing heightNeeded : ℝ) : Prop :=
  let inchesToFeet : ℝ → ℝ := (· / 12)
  let rungLengthFeet := inchesToFeet rungeLength
  let rungSpacingFeet := inchesToFeet rungSpacing
  let verticalDistanceBetweenRungs := rungLengthFeet + rungSpacingFeet
  let numRungs := heightNeeded / verticalDistanceBetweenRungs
  let woodForRungs := numRungs * rungLengthFeet
  let woodForSides := heightNeeded * 2
  let totalWoodNeeded := woodForRungs + woodForSides
  let remainingWood := totalWood - totalWoodNeeded
  remainingWood = 162.5 ∧ totalWoodNeeded ≤ totalWood

theorem ladder_construction_theorem :
  LadderProblem 300 18 6 50 :=
sorry

end ladder_construction_theorem_l3167_316789


namespace days_worked_l3167_316725

/-- Proves that given the conditions of the problem, the number of days worked is 23 -/
theorem days_worked (total_days : ℕ) (daily_wage : ℕ) (daily_forfeit : ℕ) (net_earnings : ℕ) 
  (h1 : total_days = 25)
  (h2 : daily_wage = 20)
  (h3 : daily_forfeit = 5)
  (h4 : net_earnings = 450) :
  ∃ (worked_days : ℕ), 
    worked_days * daily_wage - (total_days - worked_days) * daily_forfeit = net_earnings ∧ 
    worked_days = 23 := by
  sorry

#check days_worked

end days_worked_l3167_316725


namespace equal_area_rectangles_l3167_316757

/-- Given two rectangles of equal area, where one rectangle has dimensions 9 inches by 20 inches,
    and the other has a width of 15 inches, prove that the length of the second rectangle is 12 inches. -/
theorem equal_area_rectangles (carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_width = 15)
    (h2 : jordan_length = 9)
    (h3 : jordan_width = 20)
    (h4 : carol_width * carol_length = jordan_length * jordan_width)
    : carol_length = 12 := by
  sorry

end equal_area_rectangles_l3167_316757


namespace eight_N_plus_nine_is_perfect_square_l3167_316750

theorem eight_N_plus_nine_is_perfect_square (n : ℕ) : 
  let N := 2^(4*n + 1) - 4^n - 1
  (∃ k : ℤ, N = 9 * k) → 
  ∃ m : ℕ, 8 * N + 9 = m^2 := by
sorry

end eight_N_plus_nine_is_perfect_square_l3167_316750


namespace jose_profit_share_l3167_316791

structure Partner where
  investment : ℕ
  duration : ℕ

def totalInvestmentTime (partners : List Partner) : ℕ :=
  partners.foldl (fun acc p => acc + p.investment * p.duration) 0

def profitShare (partner : Partner) (partners : List Partner) (totalProfit : ℕ) : ℚ :=
  (partner.investment * partner.duration : ℚ) / (totalInvestmentTime partners : ℚ) * totalProfit

theorem jose_profit_share :
  let tom : Partner := { investment := 30000, duration := 12 }
  let jose : Partner := { investment := 45000, duration := 10 }
  let angela : Partner := { investment := 60000, duration := 8 }
  let rebecca : Partner := { investment := 75000, duration := 6 }
  let partners : List Partner := [tom, jose, angela, rebecca]
  let totalProfit : ℕ := 72000
  abs (profitShare jose partners totalProfit - 18620.69) < 0.01 := by
  sorry

end jose_profit_share_l3167_316791


namespace edward_candy_purchase_l3167_316747

/-- The number of candy pieces Edward can buy given his tickets and the candy cost --/
theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : 
  whack_a_mole_tickets = 3 →
  skee_ball_tickets = 5 →
  candy_cost = 4 →
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := by
  sorry

end edward_candy_purchase_l3167_316747


namespace fraction_inequality_l3167_316795

theorem fraction_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end fraction_inequality_l3167_316795


namespace range_m_f_less_than_one_solution_sets_f_geq_mx_range_m_f_nonnegative_in_interval_l3167_316732

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

/-- Theorem for the range of m when f(x) < 1 for all x in ℝ -/
theorem range_m_f_less_than_one :
  ∀ m : ℝ, (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

/-- Theorem for the solution sets of f(x) ≥ (m+1)x -/
theorem solution_sets_f_geq_mx (m : ℝ) :
  (m = -1 ∧ {x : ℝ | x ≥ 1} = {x : ℝ | f m x ≥ (m + 1) * x}) ∨
  (m > -1 ∧ {x : ℝ | x ≤ (m - 1) / (m + 1) ∨ x ≥ 1} = {x : ℝ | f m x ≥ (m + 1) * x}) ∨
  (m < -1 ∧ {x : ℝ | 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)} = {x : ℝ | f m x ≥ (m + 1) * x}) :=
sorry

/-- Theorem for the range of m when f(x) ≥ 0 for all x in [-1/2, 1/2] -/
theorem range_m_f_nonnegative_in_interval :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → f m x ≥ 0) ↔ m ≥ 1 :=
sorry

end range_m_f_less_than_one_solution_sets_f_geq_mx_range_m_f_nonnegative_in_interval_l3167_316732


namespace hyperbola_parameters_l3167_316717

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the product of the slopes of its two asymptotes is -2
    and its focal length is 6, then a² = 3 and b² = 6 -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b^2 / a^2 = 2) →  -- product of slopes of asymptotes is -2
  (6^2 = 4 * (a^2 + b^2)) →  -- focal length is 6
  a^2 = 3 ∧ b^2 = 6 := by
  sorry

end hyperbola_parameters_l3167_316717


namespace line_equivalence_l3167_316778

theorem line_equivalence :
  ∀ (x y : ℝ),
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - (-1)) = 0 ↔
  y = (3/4 : ℝ) * x - (5/2 : ℝ) :=
by sorry

end line_equivalence_l3167_316778


namespace equation_solution_l3167_316754

theorem equation_solution (x : ℝ) : 
  12 * Real.sin x - 5 * Real.cos x = 13 ↔ 
  ∃ k : ℤ, x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end equation_solution_l3167_316754


namespace necessary_not_sufficient_l3167_316704

theorem necessary_not_sufficient (a b : ℝ) : 
  (∀ a b : ℝ, b ≥ 0 → a^2 + b ≥ 0) ∧ 
  (∃ a b : ℝ, a^2 + b ≥ 0 ∧ b < 0) := by
sorry

end necessary_not_sufficient_l3167_316704


namespace worker_pay_calculation_l3167_316736

/-- Calculate the worker's pay given the following conditions:
  * The total period is 60 days
  * The pay rate for working is Rs. 20 per day
  * The deduction rate for idle days is Rs. 3 per day
  * The number of idle days is 40 days
-/
def worker_pay (total_days : ℕ) (work_rate : ℕ) (idle_rate : ℕ) (idle_days : ℕ) : ℕ :=
  let work_days := total_days - idle_days
  let earnings := work_days * work_rate
  let deductions := idle_days * idle_rate
  earnings - deductions

theorem worker_pay_calculation :
  worker_pay 60 20 3 40 = 280 := by
  sorry

end worker_pay_calculation_l3167_316736


namespace fraction_simplification_l3167_316798

theorem fraction_simplification :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = 7 / 25 := by
  sorry

end fraction_simplification_l3167_316798


namespace square_coverage_l3167_316713

/-- A square can be covered by smaller squares if the total area of the smaller squares
    is greater than or equal to the area of the larger square. -/
def can_cover (large_side small_side : ℝ) (num_small_squares : ℕ) : Prop :=
  large_side^2 ≤ (small_side^2 * num_small_squares)

/-- Theorem stating that a square with side length 7 can be covered by 8 squares
    with side length 3. -/
theorem square_coverage : can_cover 7 3 8 := by
  sorry

end square_coverage_l3167_316713


namespace largest_even_not_sum_of_odd_composites_l3167_316787

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- A number is odd if it leaves a remainder of 1 when divided by 2 -/
def IsOdd (n : ℕ) : Prop :=
  n % 2 = 1

/-- The property of being expressible as the sum of two odd composite numbers -/
def IsSumOfTwoOddComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsOdd a ∧ IsOdd b ∧ IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 38 is the largest even integer that cannot be written as the sum of two odd composite numbers -/
theorem largest_even_not_sum_of_odd_composites :
  (∀ n : ℕ, n % 2 = 0 → n > 38 → IsSumOfTwoOddComposites n) ∧
  ¬IsSumOfTwoOddComposites 38 :=
sorry

end largest_even_not_sum_of_odd_composites_l3167_316787


namespace books_from_second_shop_l3167_316719

/- Define the problem parameters -/
def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def cost_shop2 : ℕ := 800
def avg_price : ℕ := 20

/- Define the function to calculate the number of books from the second shop -/
def books_shop2 : ℕ :=
  (cost_shop1 + cost_shop2) / avg_price - books_shop1

/- Theorem statement -/
theorem books_from_second_shop :
  books_shop2 = 40 :=
sorry

end books_from_second_shop_l3167_316719


namespace sons_age_few_years_back_l3167_316788

/-- Proves that the son's age a few years back is 22, given the conditions of the problem -/
theorem sons_age_few_years_back (father_current_age : ℕ) (son_current_age : ℕ) : 
  father_current_age = 44 →
  father_current_age - son_current_age = son_current_age →
  son_current_age = 22 :=
by
  sorry

#check sons_age_few_years_back

end sons_age_few_years_back_l3167_316788


namespace first_term_of_geometric_sequence_l3167_316781

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem first_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_prod : a 2 * a 3 * a 4 = 27) 
  (h_seventh : a 7 = 27) : 
  a 1 = 1 := by
sorry

end first_term_of_geometric_sequence_l3167_316781


namespace fraction_irreducible_l3167_316751

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l3167_316751


namespace used_car_seller_problem_l3167_316759

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) :
  num_clients = 9 →
  cars_per_client = 4 →
  selections_per_car = 3 →
  num_clients * cars_per_client = selections_per_car * (num_clients * cars_per_client / selections_per_car) :=
by sorry

end used_car_seller_problem_l3167_316759


namespace quadratic_prime_roots_l3167_316707

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p + q = 50 ∧ 
    p * q = k ∧ 
    k = 141 :=
sorry

end quadratic_prime_roots_l3167_316707


namespace sqrt_inequality_l3167_316744

theorem sqrt_inequality (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) < Real.sqrt (3 * a^2) := by
  sorry

end sqrt_inequality_l3167_316744


namespace divisor_problem_l3167_316700

theorem divisor_problem : ∃ (N D : ℕ), 
  (N % D = 6) ∧ 
  (N % 19 = 7) ∧ 
  (D = 39) := by
  sorry

end divisor_problem_l3167_316700


namespace equation_holds_l3167_316764

theorem equation_holds (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c ↔ b + c = 12 :=
by sorry

end equation_holds_l3167_316764


namespace ceiling_floor_sum_l3167_316708

theorem ceiling_floor_sum : ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ = 0 := by sorry

end ceiling_floor_sum_l3167_316708


namespace product_of_three_numbers_l3167_316730

theorem product_of_three_numbers (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 30)
  (h_first : x = 3 * (y + z))
  (h_second : y = 8 * z) : 
  x * y * z = 125 := by
sorry

end product_of_three_numbers_l3167_316730


namespace fraction_sum_equals_decimal_l3167_316705

theorem fraction_sum_equals_decimal : 2/10 + 4/100 + 6/1000 = 0.246 := by
  sorry

end fraction_sum_equals_decimal_l3167_316705


namespace coin_flip_difference_l3167_316742

theorem coin_flip_difference (total_flips : ℕ) (heads : ℕ) (h1 : total_flips = 211) (h2 : heads = 65) :
  total_flips - heads - heads = 81 := by
  sorry

end coin_flip_difference_l3167_316742


namespace evaluate_expression_l3167_316780

theorem evaluate_expression : (2^3)^4 * 3^2 = 36864 := by
  sorry

end evaluate_expression_l3167_316780


namespace card_area_reduction_l3167_316774

/-- Given a 5x7 inch card, if reducing one side by 2 inches results in an area of 21 square inches,
    then reducing the other side by 2 inches instead will result in an area of 25 square inches. -/
theorem card_area_reduction (length width : ℝ) : 
  length = 5 ∧ width = 7 ∧ 
  ((length - 2) * width = 21 ∨ length * (width - 2) = 21) →
  (length * (width - 2) = 25 ∨ (length - 2) * width = 25) := by
sorry

end card_area_reduction_l3167_316774


namespace volcano_eruption_percentage_l3167_316763

theorem volcano_eruption_percentage (total_volcanoes : ℕ) 
  (intact_volcanoes : ℕ) (mid_year_percentage : ℝ) 
  (end_year_percentage : ℝ) :
  total_volcanoes = 200 →
  intact_volcanoes = 48 →
  mid_year_percentage = 0.4 →
  end_year_percentage = 0.5 →
  ∃ (x : ℝ),
    x ≥ 0 ∧ x ≤ 100 ∧
    (total_volcanoes : ℝ) * (1 - x / 100) * (1 - mid_year_percentage) * (1 - end_year_percentage) = intact_volcanoes ∧
    x = 20 := by
  sorry

end volcano_eruption_percentage_l3167_316763


namespace triangle_side_length_expression_l3167_316721

/-- Given a triangle with side lengths a, b, and c, 
    the expression |a-b+c| - |a-b-c| simplifies to 2a - 2b -/
theorem triangle_side_length_expression (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  |a - b + c| - |a - b - c| = 2*a - 2*b := by sorry

end triangle_side_length_expression_l3167_316721


namespace units_digit_17_2005_l3167_316775

theorem units_digit_17_2005 (h : 17 % 10 = 7) : (17^2005) % 10 = 7 := by
  sorry

end units_digit_17_2005_l3167_316775


namespace smallest_among_given_numbers_l3167_316720

theorem smallest_among_given_numbers :
  ∀ (a b c d : ℝ), a = -1 ∧ b = 0 ∧ c = -Real.sqrt 2 ∧ d = 2 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end smallest_among_given_numbers_l3167_316720


namespace min_value_quadratic_roots_l3167_316722

theorem min_value_quadratic_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*m*x₁ + m^2 + 3*m - 2 = 0) →
  (x₂^2 + 2*m*x₂ + m^2 + 3*m - 2 = 0) →
  (∃ (min : ℝ), ∀ (m : ℝ), x₁*(x₂ + x₁) + x₂^2 ≥ min ∧ 
  ∃ (m₀ : ℝ), x₁*(x₂ + x₁) + x₂^2 = min) →
  (∃ (min : ℝ), min = 5/4 ∧ 
  ∀ (m : ℝ), x₁*(x₂ + x₁) + x₂^2 ≥ min ∧ 
  ∃ (m₀ : ℝ), x₁*(x₂ + x₁) + x₂^2 = min) :=
sorry

end min_value_quadratic_roots_l3167_316722


namespace inequality_proof_l3167_316752

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (a b c : ℝ)
  (ha : Real.sqrt a = x * (y - z)^2)
  (hb : Real.sqrt b = y * (z - x)^2)
  (hc : Real.sqrt c = z * (x - y)^2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) := by
  sorry

end inequality_proof_l3167_316752


namespace complex_magnitude_proof_l3167_316767

theorem complex_magnitude_proof : 
  Complex.abs ((11/13 : ℂ) + (12/13 : ℂ) * Complex.I)^12 = (Real.sqrt 265 / 13)^12 := by
  sorry

end complex_magnitude_proof_l3167_316767


namespace jerry_shelves_theorem_l3167_316796

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((total_books - books_taken) + books_per_shelf - 1) / books_per_shelf

theorem jerry_shelves_theorem :
  shelves_needed 34 7 3 = 9 := by
  sorry

end jerry_shelves_theorem_l3167_316796


namespace volume_surface_area_ratio_l3167_316703

/-- A structure formed by connecting eight unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The volume of the structure in cubic units -/
  volume : ℕ
  /-- The surface area of the structure in square units -/
  surface_area : ℕ
  /-- The number of cubes is 8 -/
  cube_count : num_cubes = 8
  /-- The volume is equal to the number of cubes -/
  volume_def : volume = num_cubes
  /-- The surface area is 24 square units -/
  surface_area_def : surface_area = 24

/-- Theorem: The ratio of volume to surface area is 1/3 -/
theorem volume_surface_area_ratio (c : CubeStructure) :
  (c.volume : ℚ) / c.surface_area = 1 / 3 := by
  sorry

end volume_surface_area_ratio_l3167_316703


namespace trajectory_and_line_theorem_l3167_316714

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 36

-- Define point B
def point_B : ℝ × ℝ := (-2, 0)

-- Define the condition that P is on line segment AB
def P_on_AB (A P : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * point_B.1, t * A.2 + (1 - t) * point_B.2)

-- Define the ratio condition
def ratio_condition (A P : ℝ × ℝ) : Prop :=
  (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2 = 1/4 * ((A.1 - P.1)^2 + (A.2 - P.2)^2)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ -2

-- Define line l
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 5 = 0 ∨ x = -1

-- Define the intersection condition
def intersection_condition (M N : ℝ × ℝ) : Prop :=
  trajectory_C M.1 M.2 ∧ trajectory_C N.1 N.2 ∧
  line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12

-- Main theorem
theorem trajectory_and_line_theorem 
  (A P : ℝ × ℝ) 
  (h1 : circle_P A.1 A.2)
  (h2 : P_on_AB A P)
  (h3 : ratio_condition A P)
  (h4 : ∃ M N : ℝ × ℝ, line_l (-1) 3 ∧ intersection_condition M N) :
  trajectory_C P.1 P.2 ∧ line_l (-1) 3 :=
sorry

end trajectory_and_line_theorem_l3167_316714


namespace banana_orange_equivalence_l3167_316702

/-- Given that 3/4 of 16 bananas are worth 10 oranges, 
    prove that 3/5 of 15 bananas are worth 7.5 oranges -/
theorem banana_orange_equivalence :
  (3 / 4 : ℚ) * 16 * (1 / 10 : ℚ) = 1 →
  (3 / 5 : ℚ) * 15 * (1 / 10 : ℚ) = (15 / 2 : ℚ) * (1 / 10 : ℚ) :=
by
  sorry

end banana_orange_equivalence_l3167_316702


namespace function_properties_l3167_316706

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 + a

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a (x + π) = f a x) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f a x_min ≤ f a x) ∧
  (∃ x_min : ℝ, f a x_min = 0) →
  (a = 1) ∧
  (∀ x : ℝ, f a x ≤ 4) ∧
  (∃ k : ℤ, ∀ x : ℝ, f a x = f a (k * π / 2 + π / 6 - x)) :=
by sorry

end function_properties_l3167_316706


namespace units_digit_of_5_pow_150_plus_7_l3167_316726

theorem units_digit_of_5_pow_150_plus_7 : 
  (5^150 + 7) % 10 = 2 := by sorry

end units_digit_of_5_pow_150_plus_7_l3167_316726


namespace inequality_solution_l3167_316783

-- Define the inequality function
def f (a x : ℝ) : ℝ := x^2 - a*x + a - 1

-- Define the solution set for a > 2
def solution_set_gt2 (a : ℝ) : Set ℝ := 
  {x | x < 1 ∨ x > a - 1}

-- Define the solution set for a = 2
def solution_set_eq2 : Set ℝ := 
  {x | x < 1 ∨ x > 1}

-- Define the solution set for a < 2
def solution_set_lt2 (a : ℝ) : Set ℝ := 
  {x | x < a - 1 ∨ x > 1}

-- Theorem statement
theorem inequality_solution (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 2 ∧ x ∈ solution_set_gt2 a) ∨
    (a = 2 ∧ x ∈ solution_set_eq2) ∨
    (a < 2 ∧ x ∈ solution_set_lt2 a)) := by
  sorry

end inequality_solution_l3167_316783


namespace equation_solution_exists_l3167_316715

-- Define the possible operations
inductive Operation
  | mul
  | div

-- Define a function to apply the operation
def apply_op (op : Operation) (a b : ℕ) : ℚ :=
  match op with
  | Operation.mul => (a * b : ℚ)
  | Operation.div => (a / b : ℚ)

theorem equation_solution_exists : 
  ∃ (op1 op2 : Operation), 
    (apply_op op1 9 1307 = 100) ∧ 
    (∃ (n : ℕ), apply_op op2 14 2 = apply_op op2 n 5 ∧ n = 2) :=
by sorry

end equation_solution_exists_l3167_316715


namespace product_increase_value_l3167_316737

theorem product_increase_value (x : ℝ) (v : ℝ) : 
  x = 3 → 5 * x + v = 19 → v = 4 := by
  sorry

end product_increase_value_l3167_316737


namespace foreign_language_books_l3167_316709

theorem foreign_language_books (total : ℝ) 
  (h1 : total * (36 / 100) = total - (total * (27 / 100) + 185))
  (h2 : total * (27 / 100) = total * (36 / 100) * (75 / 100))
  (h3 : 185 = total - (total * (36 / 100) + total * (27 / 100))) :
  total = 500 := by sorry

end foreign_language_books_l3167_316709


namespace equation_equivalence_l3167_316756

theorem equation_equivalence (x : ℝ) : x^2 - 10*x - 1 = 0 ↔ (x-5)^2 = 26 := by
  sorry

end equation_equivalence_l3167_316756


namespace little_john_money_distribution_l3167_316770

theorem little_john_money_distribution 
  (initial_amount : ℚ) 
  (sweets_cost : ℚ) 
  (amount_left : ℚ) 
  (num_friends : ℕ) 
  (h1 : initial_amount = 5.1)
  (h2 : sweets_cost = 1.05)
  (h3 : amount_left = 2.05)
  (h4 : num_friends = 2) :
  let total_spent := initial_amount - amount_left
  let friends_money := total_spent - sweets_cost
  friends_money / num_friends = 1 := by
sorry

end little_john_money_distribution_l3167_316770


namespace cut_cube_theorem_l3167_316793

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes with exactly 2 painted faces -/
  two_face_cubes : ℕ
  /-- The total number of smaller cubes created -/
  total_cubes : ℕ

/-- Theorem stating that if a cube is cut such that there are 12 smaller cubes
    with 2 painted faces, then the total number of smaller cubes is 8 -/
theorem cut_cube_theorem (c : CutCube) :
  c.two_face_cubes = 12 → c.total_cubes = 8 := by
  sorry

end cut_cube_theorem_l3167_316793


namespace unique_solution_quadratic_l3167_316799

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = k + 3 * x) ↔ k = 23 / 4 := by
  sorry

end unique_solution_quadratic_l3167_316799
