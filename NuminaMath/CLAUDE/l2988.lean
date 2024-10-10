import Mathlib

namespace distinct_triangles_in_cube_l2988_298866

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  /-- The number of vertices in a cube. -/
  num_vertices : ℕ
  /-- The number of edges in a cube. -/
  num_edges : ℕ
  /-- The number of edges meeting at each vertex of a cube. -/
  edges_per_vertex : ℕ
  /-- Assertion that a cube has 8 vertices. -/
  vertices_axiom : num_vertices = 8
  /-- Assertion that a cube has 12 edges. -/
  edges_axiom : num_edges = 12
  /-- Assertion that 3 edges meet at each vertex of a cube. -/
  edges_per_vertex_axiom : edges_per_vertex = 3

/-- A function that calculates the number of distinct triangles in a cube. -/
def count_distinct_triangles (c : Cube) : ℕ :=
  c.num_vertices * (c.edges_per_vertex.choose 2) / 2

/-- Theorem stating that the number of distinct triangles formed by connecting three different edges of a cube, 
    where each set of edges shares a common vertex, is equal to 12. -/
theorem distinct_triangles_in_cube (c : Cube) : 
  count_distinct_triangles c = 12 := by
  sorry

end distinct_triangles_in_cube_l2988_298866


namespace equation_solution_inequality_solution_l2988_298893

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℚ, (1 / (x + 1) - 1 / (x + 2) = 1 / (x + 3) - 1 / (x + 4)) ↔ (x = -5/2) :=
sorry

-- Part 2: Inequality solution
theorem inequality_solution (a : ℚ) (x : ℚ) :
  x^2 - (a + 1) * x + a ≤ 0 ↔
    (a = 1 ∧ x = 1) ∨
    (a < 1 ∧ a ≤ x ∧ x ≤ 1) ∨
    (a > 1 ∧ 1 ≤ x ∧ x ≤ a) :=
sorry

end equation_solution_inequality_solution_l2988_298893


namespace reciprocal_of_negative_four_l2988_298862

theorem reciprocal_of_negative_four :
  ∃ x : ℚ, x * (-4) = 1 ∧ x = -1/4 := by sorry

end reciprocal_of_negative_four_l2988_298862


namespace median_intersection_property_l2988_298847

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Define the median intersection point
def medianIntersection (t : Triangle) : Point :=
  sorry

-- Define points M, N, P on the sides of the triangle
def dividePoint (A B : Point) (p q : ℝ) : Point :=
  sorry

-- Theorem statement
theorem median_intersection_property (ABC : Triangle) (p q : ℝ) :
  let O := medianIntersection ABC
  let M := dividePoint ABC.A ABC.B p q
  let N := dividePoint ABC.B ABC.C p q
  let P := dividePoint ABC.C ABC.A p q
  let MNP : Triangle := ⟨M, N, P⟩
  let ANBPCMTriangle : Triangle := 
    ⟨ABC.A, ABC.B, ABC.C⟩  -- This is a placeholder, as we don't have a way to define the intersection points
  (O = medianIntersection MNP) ∧ 
  (O = medianIntersection ANBPCMTriangle) :=
sorry

end median_intersection_property_l2988_298847


namespace least_positive_integer_with_specific_remainders_l2988_298834

theorem least_positive_integer_with_specific_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 4) ∧ 
  (n % 6 = 5) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 ∧ m % 11 = 10 → m ≥ n) ∧
  n = 4619 :=
by sorry

end least_positive_integer_with_specific_remainders_l2988_298834


namespace shirley_sold_54_boxes_l2988_298832

/-- The number of cases Shirley needs to deliver -/
def num_cases : ℕ := 9

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 6

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem shirley_sold_54_boxes : total_boxes = 54 := by
  sorry

end shirley_sold_54_boxes_l2988_298832


namespace square_of_87_l2988_298855

theorem square_of_87 : 87^2 = 7569 := by sorry

end square_of_87_l2988_298855


namespace nested_fraction_equality_l2988_298880

theorem nested_fraction_equality : 1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by sorry

end nested_fraction_equality_l2988_298880


namespace treasure_value_is_3049_l2988_298856

/-- Converts a list of digits in base 7 to its decimal (base 10) equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The total value of the treasures in base 10 -/
def totalTreasureValue : Nat :=
  base7ToDecimal [4, 1, 2, 3] + -- 3214₇
  base7ToDecimal [2, 5, 6, 1] + -- 1652₇
  base7ToDecimal [1, 3, 4, 2] + -- 2431₇
  base7ToDecimal [4, 5, 6]      -- 654₇

/-- Theorem stating that the total value of the treasures is 3049 -/
theorem treasure_value_is_3049 : totalTreasureValue = 3049 := by
  sorry


end treasure_value_is_3049_l2988_298856


namespace boxes_to_fill_l2988_298828

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) 
  (h1 : total_boxes = 25.75) 
  (h2 : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := by
  sorry

end boxes_to_fill_l2988_298828


namespace successive_price_reduction_l2988_298843

theorem successive_price_reduction (initial_reduction : ℝ) (subsequent_reduction : ℝ) 
  (initial_reduction_percent : initial_reduction = 0.25) 
  (subsequent_reduction_percent : subsequent_reduction = 0.40) : 
  1 - (1 - initial_reduction) * (1 - subsequent_reduction) = 0.55 := by
sorry

end successive_price_reduction_l2988_298843


namespace first_term_of_geometric_series_first_term_of_geometric_series_l2988_298892

/-- Given an infinite geometric series with sum 18 and sum of squares 72, 
    prove that the first term of the series is 72/11 -/
theorem first_term_of_geometric_series 
  (a : ℝ) -- First term of the series
  (r : ℝ) -- Common ratio of the series
  (h1 : a / (1 - r) = 18) -- Sum of the series is 18
  (h2 : a^2 / (1 - r^2) = 72) -- Sum of squares is 72
  : a = 72 / 11 := by
sorry

/-- Alternative formulation using a function for the series -/
theorem first_term_of_geometric_series' 
  (S : ℕ → ℝ) -- Geometric series as a function
  (h1 : ∃ r : ℝ, ∀ n : ℕ, S (n + 1) = r * S n) -- S is a geometric series
  (h2 : ∑' n, S n = 18) -- Sum of the series is 18
  (h3 : ∑' n, (S n)^2 = 72) -- Sum of squares is 72
  : S 0 = 72 / 11 := by
sorry

end first_term_of_geometric_series_first_term_of_geometric_series_l2988_298892


namespace roses_cut_l2988_298826

theorem roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 13)
  (h2 : initial_orchids = 84)
  (h3 : final_roses = 14)
  (h4 : final_orchids = 91) :
  final_roses - initial_roses = 1 := by
  sorry

end roses_cut_l2988_298826


namespace first_day_of_month_l2988_298865

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (day_after d n)

theorem first_day_of_month (d : DayOfWeek) :
  day_after d 29 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday :=
by
  sorry


end first_day_of_month_l2988_298865


namespace angle_half_quadrant_l2988_298870

-- Define the angle α
def α : ℝ := sorry

-- Define the integer k
def k : ℤ := sorry

-- Define the condition for α
axiom α_condition : 40 + k * 360 < α ∧ α < 140 + k * 360

-- Define the first quadrant
def first_quadrant (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 90

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop := 180 ≤ θ ∧ θ < 270

-- State the theorem
theorem angle_half_quadrant : 
  first_quadrant (α / 2) ∨ third_quadrant (α / 2) := by sorry

end angle_half_quadrant_l2988_298870


namespace sandwich_not_vegetable_percentage_l2988_298833

def sandwich_weight : ℝ := 180
def vegetable_weight : ℝ := 50

theorem sandwich_not_vegetable_percentage :
  let non_vegetable_weight := sandwich_weight - vegetable_weight
  let percentage := (non_vegetable_weight / sandwich_weight) * 100
  ∃ ε > 0, |percentage - 72.22| < ε :=
sorry

end sandwich_not_vegetable_percentage_l2988_298833


namespace length_of_AC_l2988_298829

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)  -- Points in 2D plane

-- Define the conditions
def satisfies_conditions (q : Quadrilateral) : Prop :=
  let d := (λ p1 p2 : ℝ × ℝ => ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt)
  d q.O q.A = 5 ∧
  d q.O q.C = 12 ∧
  d q.O q.B = 6 ∧
  d q.O q.D = 5 ∧
  d q.B q.D = 11

-- State the theorem
theorem length_of_AC (q : Quadrilateral) :
  satisfies_conditions q →
  let d := (λ p1 p2 : ℝ × ℝ => ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt)
  d q.A q.C = 3 * (71 : ℝ).sqrt :=
by sorry

end length_of_AC_l2988_298829


namespace trigonometric_values_l2988_298801

theorem trigonometric_values (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x - Real.cos x = -17/13) ∧ 
  (4 * Real.sin x * Real.cos x - Real.cos x^2 = -384/169) := by
  sorry

end trigonometric_values_l2988_298801


namespace quiz_winning_probability_l2988_298812

/-- The number of questions in the quiz -/
def num_questions : ℕ := 4

/-- The number of choices for each question -/
def num_choices : ℕ := 4

/-- The minimum number of correct answers needed to win -/
def min_correct : ℕ := 3

/-- The probability of answering a single question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The probability of answering a single question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the quiz -/
def prob_winning : ℚ := (num_questions.choose min_correct) * (prob_correct ^ min_correct * prob_incorrect ^ (num_questions - min_correct)) +
                        (num_questions.choose num_questions) * (prob_correct ^ num_questions)

theorem quiz_winning_probability :
  prob_winning = 13 / 256 := by
  sorry

end quiz_winning_probability_l2988_298812


namespace parents_selection_count_l2988_298823

def number_of_students : ℕ := 6
def number_of_parents : ℕ := 12
def parents_to_choose : ℕ := 4

theorem parents_selection_count : 
  (number_of_students.choose 1) * ((number_of_parents - 2).choose 1) * ((number_of_parents - 4).choose 1) = 480 :=
by sorry

end parents_selection_count_l2988_298823


namespace isosceles_triangle_perimeter_l2988_298869

-- Define an isosceles triangle with side lengths a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ a + c > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
    ((t.a = 3 ∧ t.b = 7) ∨ (t.a = 7 ∧ t.b = 3) ∨ 
     (t.b = 3 ∧ t.c = 7) ∨ (t.b = 7 ∧ t.c = 3) ∨ 
     (t.a = 3 ∧ t.c = 7) ∨ (t.a = 7 ∧ t.c = 3)) →
    perimeter t = 17 := by
  sorry

end isosceles_triangle_perimeter_l2988_298869


namespace victors_work_hours_l2988_298846

theorem victors_work_hours (hourly_rate : ℝ) (total_earnings : ℝ) (h : ℝ) : 
  hourly_rate = 6 → 
  total_earnings = 60 → 
  2 * (hourly_rate * h) = total_earnings → 
  h = 5 := by
sorry

end victors_work_hours_l2988_298846


namespace multiply_48_52_l2988_298858

theorem multiply_48_52 : 48 * 52 = 2496 := by
  sorry

end multiply_48_52_l2988_298858


namespace age_difference_proof_l2988_298822

/-- Given the ages of Katie's daughter, Lavinia's daughter, and Lavinia's son, prove that Lavinia's son is 22 years older than Lavinia's daughter. -/
theorem age_difference_proof (katie_daughter_age lavinia_daughter_age lavinia_son_age : ℕ) :
  katie_daughter_age = 12 →
  lavinia_daughter_age = katie_daughter_age - 10 →
  lavinia_son_age = 2 * katie_daughter_age →
  lavinia_son_age - lavinia_daughter_age = 22 :=
by
  sorry

end age_difference_proof_l2988_298822


namespace percentage_problem_l2988_298821

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  (0.4 * x = 160) → (p * x = 120) → p = 0.3 := by
  sorry

end percentage_problem_l2988_298821


namespace sufficient_condition_implies_range_l2988_298875

theorem sufficient_condition_implies_range (a : ℝ) : 
  (∀ x, (x - 1) * (x - 2) < 0 → x - a < 0) → a ≥ 2 := by
  sorry

end sufficient_condition_implies_range_l2988_298875


namespace class_book_count_l2988_298857

/-- Calculates the final number of books after a series of additions and subtractions. -/
def finalBookCount (initial given_away received_later traded_away received_in_trade additional : ℕ) : ℕ :=
  initial - given_away + received_later - traded_away + received_in_trade + additional

/-- Theorem stating that given the specified book counts, the final count is 93. -/
theorem class_book_count : 
  finalBookCount 54 16 23 12 9 35 = 93 := by
  sorry

end class_book_count_l2988_298857


namespace isabel_songs_total_l2988_298873

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := 2

/-- The number of songs in each album -/
def songs_per_album : ℕ := 9

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem isabel_songs_total : total_songs = 72 := by
  sorry

end isabel_songs_total_l2988_298873


namespace emmas_age_l2988_298860

/-- Given the ages and relationships between Jose, Zack, Inez, and Emma, prove Emma's age --/
theorem emmas_age (jose_age : ℕ) (zack_age : ℕ) (inez_age : ℕ) (emma_age : ℕ)
  (h1 : jose_age = 20)
  (h2 : zack_age = jose_age + 4)
  (h3 : inez_age = zack_age - 12)
  (h4 : emma_age = jose_age + 5) :
  emma_age = 25 := by
  sorry


end emmas_age_l2988_298860


namespace b_value_l2988_298851

theorem b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 := by
  sorry

end b_value_l2988_298851


namespace tempo_insurance_fraction_l2988_298877

/-- The fraction of the original value that a tempo is insured for -/
def insured_fraction (premium_rate : ℚ) (premium_amount : ℚ) (original_value : ℚ) : ℚ :=
  (premium_amount / premium_rate) / original_value

/-- Theorem stating that given the specific conditions, the insured fraction is 4/5 -/
theorem tempo_insurance_fraction :
  let premium_rate : ℚ := 13 / 1000
  let premium_amount : ℚ := 910
  let original_value : ℚ := 87500
  insured_fraction premium_rate premium_amount original_value = 4 / 5 := by
sorry


end tempo_insurance_fraction_l2988_298877


namespace rain_probability_in_tel_aviv_l2988_298816

/-- The probability of exactly k successes in n independent trials,
    where the probability of success in each trial is p. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day in Tel Aviv -/
def probabilityOfRain : ℝ := 0.5

/-- The number of randomly chosen days -/
def totalDays : ℕ := 6

/-- The number of rainy days we're interested in -/
def rainyDays : ℕ := 4

theorem rain_probability_in_tel_aviv :
  binomialProbability totalDays rainyDays probabilityOfRain = 0.234375 := by
  sorry

end rain_probability_in_tel_aviv_l2988_298816


namespace cricket_players_l2988_298885

theorem cricket_players (total : ℕ) (basketball : ℕ) (both : ℕ) 
  (h1 : total = 880) 
  (h2 : basketball = 600) 
  (h3 : both = 220) : 
  total = (total - basketball + both) + basketball - both :=
by sorry

end cricket_players_l2988_298885


namespace max_candies_theorem_l2988_298841

/-- Represents the distribution of candies among students -/
structure CandyDistribution where
  num_students : ℕ
  total_candies : ℕ
  min_candies : ℕ
  max_candies : ℕ

/-- The greatest number of candies one student could have taken -/
def max_student_candies (d : CandyDistribution) : ℕ :=
  min d.max_candies (d.total_candies - (d.num_students - 1) * d.min_candies)

/-- Theorem stating the maximum number of candies one student could have taken -/
theorem max_candies_theorem (d : CandyDistribution) 
    (h1 : d.num_students = 50)
    (h2 : d.total_candies = 50 * 7)
    (h3 : d.min_candies = 1)
    (h4 : d.max_candies = 20) :
    max_student_candies d = 20 := by
  sorry

#eval max_student_candies { num_students := 50, total_candies := 350, min_candies := 1, max_candies := 20 }

end max_candies_theorem_l2988_298841


namespace marble_ratio_proof_l2988_298881

theorem marble_ratio_proof (initial_red : ℕ) (initial_blue : ℕ) (red_taken : ℕ) (total_left : ℕ) :
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  total_left = 35 →
  ∃ (blue_taken : ℕ),
    blue_taken * red_taken = 4 * red_taken ∧
    initial_red + initial_blue = total_left + red_taken + blue_taken :=
by
  sorry

end marble_ratio_proof_l2988_298881


namespace student_lecture_selections_l2988_298871

/-- The number of different selection methods for students choosing lectures -/
def selection_methods (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: Given 4 students and 3 lectures, the number of different selection methods is 81 -/
theorem student_lecture_selections :
  selection_methods 4 3 = 81 := by
  sorry

end student_lecture_selections_l2988_298871


namespace simplified_fraction_numerator_problem_solution_l2988_298852

theorem simplified_fraction_numerator (a : ℕ) (h : a > 0) : 
  ((a + 1 : ℚ) / a - a / (a + 1)) * (a * (a + 1)) = 2 * a + 1 :=
by sorry

theorem problem_solution : 
  ((2024 : ℚ) / 2023 - 2023 / 2024) * (2023 * 2024) = 4047 :=
by sorry

end simplified_fraction_numerator_problem_solution_l2988_298852


namespace remainder_theorem_l2988_298868

theorem remainder_theorem (P D Q R D' Q' R' D'' Q'' R'' : ℕ) 
  (h1 : P = D * Q + R) 
  (h2 : Q = D' * Q' + R') 
  (h3 : Q' = D'' * Q'' + R'') : 
  P % (D * D' * D'') = D' * D * R'' + D * R' + R :=
sorry

end remainder_theorem_l2988_298868


namespace original_number_proof_l2988_298805

theorem original_number_proof (r : ℝ) : 
  r * (1 + 0.125) - r * (1 - 0.25) = 30 → r = 80 := by
  sorry

end original_number_proof_l2988_298805


namespace expression_value_l2988_298886

theorem expression_value (a b c d x : ℝ) 
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) :
  2 * x^2 - (a * b - c - d) + |a * b + 3| = 21 := by
  sorry

end expression_value_l2988_298886


namespace final_expression_l2988_298859

/-- Given a real number b, prove that doubling b, adding 4, subtracting 4b, and dividing by 2 results in -b + 2 -/
theorem final_expression (b : ℝ) : ((2 * b + 4) - 4 * b) / 2 = -b + 2 := by
  sorry

end final_expression_l2988_298859


namespace sum_of_first_5n_integers_l2988_298883

theorem sum_of_first_5n_integers (n : ℕ) : 
  (3*n*(3*n + 1))/2 = (n*(n + 1))/2 + 210 → 
  (5*n*(5*n + 1))/2 = 630 := by
sorry

end sum_of_first_5n_integers_l2988_298883


namespace max_value_of_a_l2988_298884

theorem max_value_of_a (a b c d : ℝ) 
  (h1 : b + c + d = 3 - a) 
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) : 
  ∃ (max_a : ℝ), max_a = 2 ∧ ∀ a', (∃ b' c' d', b' + c' + d' = 3 - a' ∧ 
    2 * b'^2 + 3 * c'^2 + 6 * d'^2 = 5 - a'^2) → a' ≤ max_a :=
sorry

end max_value_of_a_l2988_298884


namespace chosen_numbers_divisibility_l2988_298894

theorem chosen_numbers_divisibility 
  (S : Finset ℕ) 
  (h_card : S.card = 250) 
  (h_bound : ∀ n ∈ S, n ≤ 501) :
  ∀ t : ℤ, ∃ a₁ a₂ a₃ a₄ : ℕ, 
    a₁ ∈ S ∧ a₂ ∈ S ∧ a₃ ∈ S ∧ a₄ ∈ S ∧ 
    23 ∣ (a₁ + a₂ + a₃ + a₄ - t) :=
by sorry

end chosen_numbers_divisibility_l2988_298894


namespace ceiling_sqrt_count_l2988_298879

theorem ceiling_sqrt_count : 
  (Finset.range 226 \ Finset.range 197).card = 29 := by sorry

end ceiling_sqrt_count_l2988_298879


namespace negation_constant_geometric_sequence_l2988_298887

theorem negation_constant_geometric_sequence :
  ¬(∀ (a : ℕ → ℝ), (∀ n : ℕ, a n = a 0) → (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)) ↔
  (∃ (a : ℕ → ℝ), (∀ n : ℕ, a n = a 0) ∧ ¬(∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)) :=
by sorry

end negation_constant_geometric_sequence_l2988_298887


namespace xiao_dong_language_understanding_l2988_298848

-- Define propositions
variable (P : Prop) -- Xiao Dong understands English
variable (Q : Prop) -- Xiao Dong understands French

-- Theorem statement
theorem xiao_dong_language_understanding : 
  ¬(P ∧ Q) → (P → ¬Q) :=
by
  sorry

end xiao_dong_language_understanding_l2988_298848


namespace f_shifted_l2988_298850

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem f_shifted (x : ℝ) :
  (1 ≤ x ∧ x ≤ 3) → (2 ≤ x ∧ x ≤ 4) → f (x - 1) = 2 * x - 1 := by
  sorry

end f_shifted_l2988_298850


namespace polynomial_root_implies_coefficients_l2988_298867

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*I : ℂ) = -a/3 - (Complex.I * Real.sqrt (Complex.normSq (2 - 3*I) - a^2/9))) :
  a = -3/2 ∧ b = 65/2 := by
  sorry

end polynomial_root_implies_coefficients_l2988_298867


namespace cube_root_of_19683_l2988_298806

theorem cube_root_of_19683 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 19683) : x = 27 := by
  sorry

end cube_root_of_19683_l2988_298806


namespace inequality_solution_l2988_298827

theorem inequality_solution (x : ℝ) : (x^2 - 49) / (x + 7) < 0 ↔ x < -7 ∨ (-7 < x ∧ x < 7) := by
  sorry

end inequality_solution_l2988_298827


namespace peach_fraction_proof_l2988_298876

theorem peach_fraction_proof (martine_peaches benjy_peaches gabrielle_peaches : ℕ) : 
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  martine_peaches = 2 * benjy_peaches + 6 →
  (benjy_peaches : ℚ) / gabrielle_peaches = 1 / 3 := by
  sorry

end peach_fraction_proof_l2988_298876


namespace card_distribution_count_l2988_298820

/-- The number of ways to distribute 6 cards into 3 envelopes -/
def card_distribution : ℕ :=
  let n_cards : ℕ := 6
  let n_envelopes : ℕ := 3
  let cards_per_envelope : ℕ := 2
  let n_free_cards : ℕ := n_cards - 2  -- A and B are treated as one unit
  let ways_to_distribute_remaining : ℕ := Nat.choose n_free_cards cards_per_envelope
  let envelope_choices_for_ab : ℕ := n_envelopes
  ways_to_distribute_remaining * envelope_choices_for_ab

/-- Theorem stating that the number of card distributions is 18 -/
theorem card_distribution_count : card_distribution = 18 := by
  sorry

end card_distribution_count_l2988_298820


namespace least_multiple_24_greater_450_l2988_298831

theorem least_multiple_24_greater_450 : ∃ n : ℕ, 24 * n = 456 ∧ 456 > 450 ∧ ∀ m : ℕ, 24 * m > 450 → 24 * m ≥ 456 :=
sorry

end least_multiple_24_greater_450_l2988_298831


namespace ellipse_equation_line_equation_l2988_298836

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  foci_on_x_axis : Bool

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem
def ellipse_problem (e : Ellipse) (B : Point) (Q : Point) (F : Point) : Prop :=
  e.center = (0, 0) ∧
  e.foci_on_x_axis = true ∧
  B = ⟨0, 1⟩ ∧
  Q = ⟨0, 3/2⟩ ∧
  F.y = 0 ∧
  F.x > 0 ∧
  (F.x - 0 + 2 * Real.sqrt 2) / Real.sqrt 2 = 3

-- Theorem for the ellipse equation
theorem ellipse_equation (e : Ellipse) (B : Point) (Q : Point) (F : Point) :
  ellipse_problem e B Q F →
  ∀ x y : ℝ, (x^2 / 3 + y^2 = 1) ↔ (x^2 / e.a^2 + y^2 / e.b^2 = 1) :=
sorry

-- Theorem for the line equation
theorem line_equation (e : Ellipse) (B : Point) (Q : Point) (F : Point) (l : Line) :
  ellipse_problem e B Q F →
  (∃ M N : Point,
    M ≠ N ∧
    (M.x^2 / 3 + M.y^2 = 1) ∧
    (N.x^2 / 3 + N.y^2 = 1) ∧
    M.y = l.slope * M.x + l.intercept ∧
    N.y = l.slope * N.x + l.intercept ∧
    (M.x - B.x)^2 + (M.y - B.y)^2 = (N.x - B.x)^2 + (N.y - B.y)^2) →
  (l.slope = Real.sqrt 6 / 3 ∧ l.intercept = 3/2) ∨
  (l.slope = -Real.sqrt 6 / 3 ∧ l.intercept = 3/2) :=
sorry

end ellipse_equation_line_equation_l2988_298836


namespace geometric_sequence_problem_l2988_298845

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence a with a₁ = 1 and a₅ = 16, prove that a₃ = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a1 : a 1 = 1) 
    (h_a5 : a 5 = 16) : 
  a 3 = 4 := by
sorry


end geometric_sequence_problem_l2988_298845


namespace unripe_orange_harvest_l2988_298819

/-- The number of sacks of unripe oranges harvested per day -/
def daily_unripe_harvest : ℕ := 65

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges harvested over the harvest period -/
def total_unripe_harvest : ℕ := daily_unripe_harvest * harvest_days

theorem unripe_orange_harvest : total_unripe_harvest = 390 := by
  sorry

end unripe_orange_harvest_l2988_298819


namespace expression_simplification_l2988_298840

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -1) :
  (x + 1) / (x^2 - 2*x) / (1 + 1/x) = 1 / (x - 2) :=
by sorry

end expression_simplification_l2988_298840


namespace bobby_candy_remaining_l2988_298814

def candy_problem (initial_count : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial_count - (first_eaten + second_eaten)

theorem bobby_candy_remaining :
  candy_problem 36 17 15 = 4 := by
  sorry

end bobby_candy_remaining_l2988_298814


namespace inequality_solution_set_l2988_298803

theorem inequality_solution_set (m : ℝ) (h : m < -3) :
  {x : ℝ | (m + 3) * x^2 - (2 * m + 3) * x + m > 0} = {x : ℝ | 1 < x ∧ x < m / (m + 3)} :=
by sorry

end inequality_solution_set_l2988_298803


namespace remainder_problem_l2988_298818

theorem remainder_problem (n : ℕ) (a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : b < 102) 
  (h3 : n = 103 * c + d) 
  (h4 : d < 103) 
  (h5 : a + d = 20) : 
  b = 20 := by
sorry

end remainder_problem_l2988_298818


namespace triangle_side_length_l2988_298815

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2√3, b = 2, and the area S_ABC = √3, then c = 2 or c = 2√7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S_ABC : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 2 →
  S_ABC = Real.sqrt 3 →
  (c = 2 ∨ c = 2 * Real.sqrt 7) :=
by sorry

end triangle_side_length_l2988_298815


namespace square_difference_equals_1380_l2988_298899

theorem square_difference_equals_1380 : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end square_difference_equals_1380_l2988_298899


namespace standard_deviation_of_applicant_ages_l2988_298863

def average_age : ℕ := 10
def num_different_ages : ℕ := 17

theorem standard_deviation_of_applicant_ages :
  ∃ (s : ℕ),
    s > 0 ∧
    (average_age + s) - (average_age - s) + 1 = num_different_ages ∧
    s = 8 := by
  sorry

end standard_deviation_of_applicant_ages_l2988_298863


namespace triangle_max_perimeter_l2988_298813

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 17 →
    x + 4*x > 17 →
    x + 17 > 4*x →
    ∀ y : ℕ,
      y > 0 →
      y < 17 →
      y + 4*y > 17 →
      y + 17 > 4*y →
      x + 4*x + 17 ≥ y + 4*y + 17 →
      x + 4*x + 17 ≤ 42 :=
by
  sorry

end triangle_max_perimeter_l2988_298813


namespace fabric_per_shirt_is_two_l2988_298889

/-- Represents the daily production and fabric usage in a tailoring business -/
structure TailoringBusiness where
  shirts_per_day : ℕ
  pants_per_day : ℕ
  fabric_per_pants : ℕ
  total_fabric_3days : ℕ

/-- Calculates the amount of fabric used for each shirt -/
def fabric_per_shirt (tb : TailoringBusiness) : ℚ :=
  let total_pants_3days := tb.pants_per_day * 3
  let fabric_for_pants := total_pants_3days * tb.fabric_per_pants
  let fabric_for_shirts := tb.total_fabric_3days - fabric_for_pants
  let total_shirts_3days := tb.shirts_per_day * 3
  fabric_for_shirts / total_shirts_3days

/-- Theorem stating that the amount of fabric per shirt is 2 yards -/
theorem fabric_per_shirt_is_two (tb : TailoringBusiness) 
    (h1 : tb.shirts_per_day = 3)
    (h2 : tb.pants_per_day = 5)
    (h3 : tb.fabric_per_pants = 5)
    (h4 : tb.total_fabric_3days = 93) :
    fabric_per_shirt tb = 2 := by
  sorry

#eval fabric_per_shirt { shirts_per_day := 3, pants_per_day := 5, fabric_per_pants := 5, total_fabric_3days := 93 }

end fabric_per_shirt_is_two_l2988_298889


namespace decimal_6_to_binary_l2988_298853

def binary_representation (n : ℕ) : List Bool :=
  sorry

theorem decimal_6_to_binary :
  binary_representation 6 = [true, true, false] :=
sorry

end decimal_6_to_binary_l2988_298853


namespace complex_magnitude_problem_l2988_298825

theorem complex_magnitude_problem (w : ℂ) (h : w^2 = 45 - 21*I) : 
  Complex.abs w = (2466 : ℝ)^(1/4) := by
  sorry

end complex_magnitude_problem_l2988_298825


namespace tangent_circle_height_difference_l2988_298811

/-- A parabola with equation y = x^2 + x -/
def parabola (x : ℝ) : ℝ := x^2 + x

/-- A circle inside the parabola, tangent at two points -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ
  tangent_to_parabola1 : parabola tangentPoint1.1 = tangentPoint1.2
  tangent_to_parabola2 : parabola tangentPoint2.1 = tangentPoint2.2
  on_circle1 : (tangentPoint1.1 - center.1)^2 + (tangentPoint1.2 - center.2)^2 = radius^2
  on_circle2 : (tangentPoint2.1 - center.1)^2 + (tangentPoint2.2 - center.2)^2 = radius^2

/-- The theorem stating the height difference -/
theorem tangent_circle_height_difference (c : TangentCircle) :
  c.center.2 - c.tangentPoint1.2 = 1 :=
sorry

end tangent_circle_height_difference_l2988_298811


namespace sequence_identity_l2988_298835

def IsIncreasing (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≤ j → a i ≤ a j

def DivisorCountEqual (a : ℕ → ℕ) : Prop :=
  ∀ i j, (Nat.divisors (i + j)).card = (Nat.divisors (a i + a j)).card

theorem sequence_identity (a : ℕ → ℕ) 
    (h1 : IsIncreasing a) 
    (h2 : DivisorCountEqual a) : 
    ∀ n : ℕ, a n = n := by
  sorry

end sequence_identity_l2988_298835


namespace ten_point_circle_chords_l2988_298810

/-- The number of chords that can be drawn between n points on a circle's circumference,
    where no two adjacent points can be connected. -/
def restricted_chords (n : ℕ) : ℕ :=
  Nat.choose n 2 - n

theorem ten_point_circle_chords :
  restricted_chords 10 = 35 := by
  sorry

end ten_point_circle_chords_l2988_298810


namespace total_subjects_is_41_l2988_298802

/-- The total number of subjects taken by Millie, Monica, and Marius -/
def total_subjects (monica_subjects marius_subjects millie_subjects : ℕ) : ℕ :=
  monica_subjects + marius_subjects + millie_subjects

/-- Theorem stating the total number of subjects taken by all three students -/
theorem total_subjects_is_41 :
  ∃ (monica_subjects marius_subjects millie_subjects : ℕ),
    monica_subjects = 10 ∧
    marius_subjects = monica_subjects + 4 ∧
    millie_subjects = marius_subjects + 3 ∧
    total_subjects monica_subjects marius_subjects millie_subjects = 41 :=
by
  sorry

end total_subjects_is_41_l2988_298802


namespace boys_percentage_in_class_l2988_298800

theorem boys_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * 100) = 42857 / 100000 :=
by sorry

end boys_percentage_in_class_l2988_298800


namespace simons_raft_sticks_l2988_298891

theorem simons_raft_sticks (S : ℕ) : 
  S + (2 * S / 3) + (S + (2 * S / 3) + 9) = 129 → S = 51 := by
  sorry

end simons_raft_sticks_l2988_298891


namespace min_value_of_expression_l2988_298896

theorem min_value_of_expression (m n : ℝ) : 
  m > 0 → n > 0 → 2 * m - n * (-2) - 2 = 0 → 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 2 * m' - n' * (-2) - 2 = 0 → 
    1 / m + 2 / n ≤ 1 / m' + 2 / n') → 
  1 / m + 2 / n = 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_of_expression_l2988_298896


namespace solution_satisfies_equations_l2988_298830

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := (x^2 + 11) * Real.sqrt (21 + y^2) = 180
def equation2 (y z : ℝ) : Prop := (y^2 + 21) * Real.sqrt (z^2 - 33) = 100
def equation3 (z x : ℝ) : Prop := (z^2 - 33) * Real.sqrt (11 + x^2) = 96

-- Define the solution set
def solutionSet : Set (ℝ × ℝ × ℝ) :=
  {(5, 2, 7), (5, 2, -7), (5, -2, 7), (5, -2, -7),
   (-5, 2, 7), (-5, 2, -7), (-5, -2, 7), (-5, -2, -7)}

-- Theorem stating that all elements in the solution set satisfy the system of equations
theorem solution_satisfies_equations :
  ∀ (x y z : ℝ), (x, y, z) ∈ solutionSet →
    equation1 x y ∧ equation2 y z ∧ equation3 z x :=
by sorry

end solution_satisfies_equations_l2988_298830


namespace tree_support_uses_triangle_stability_l2988_298844

/-- A triangle formed by two supporting sticks and a tree -/
structure TreeSupport where
  stickOne : ℝ × ℝ  -- Coordinates of the first stick's base
  stickTwo : ℝ × ℝ  -- Coordinates of the second stick's base
  treeTop : ℝ × ℝ   -- Coordinates of the tree's top

/-- The property of a triangle that provides support -/
def triangleProperty : String := "stability"

/-- 
  Theorem: The property of triangles applied when using two wooden sticks 
  to support a falling tree is stability.
-/
theorem tree_support_uses_triangle_stability (support : TreeSupport) : 
  triangleProperty = "stability" := by
  sorry

end tree_support_uses_triangle_stability_l2988_298844


namespace polynomial_division_remainder_l2988_298817

/-- Given a polynomial P(z) = 4z^3 - 5z^2 - 19z + 4, when divided by 4z + 6
    with quotient z^2 - 4z + 1, prove that the remainder is 5z^2 + z - 2. -/
theorem polynomial_division_remainder
  (z : ℂ)
  (P : ℂ → ℂ)
  (D : ℂ → ℂ)
  (Q : ℂ → ℂ)
  (h1 : P z = 4 * z^3 - 5 * z^2 - 19 * z + 4)
  (h2 : D z = 4 * z + 6)
  (h3 : Q z = z^2 - 4 * z + 1)
  : ∃ R : ℂ → ℂ, P z = D z * Q z + R z ∧ R z = 5 * z^2 + z - 2 := by
  sorry

end polynomial_division_remainder_l2988_298817


namespace intersection_points_are_correct_l2988_298807

/-- The set of intersection points of the given lines -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | (∃ (x y : ℝ), p = (x, y) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3))}

/-- The theorem stating that the intersection points are (4, 0) and (-3, -10.5) -/
theorem intersection_points_are_correct :
  intersection_points = {(4, 0), (-3, -10.5)} :=
by sorry

end intersection_points_are_correct_l2988_298807


namespace inequality_theorem_l2988_298809

theorem inequality_theorem :
  (∀ (x y z : ℝ), x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧
  (∃ (k : ℝ), k > Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + 2*y^2 + 3*z^2 ≥ k * (x*y + y*z + z*x)) :=
by sorry

end inequality_theorem_l2988_298809


namespace orange_flower_count_l2988_298854

/-- Represents the number of flowers of each color in a garden -/
structure FlowerGarden where
  orange : ℕ
  red : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem orange_flower_count (g : FlowerGarden) : 
  g.orange + g.red + g.yellow + g.pink + g.purple = 105 →
  g.red = 2 * g.orange →
  g.yellow = g.red - 5 →
  g.pink = g.purple →
  g.pink + g.purple = 30 →
  g.orange = 16 := by
  sorry


end orange_flower_count_l2988_298854


namespace smallest_even_divisible_by_20_and_60_l2988_298874

theorem smallest_even_divisible_by_20_and_60 : ∃ n : ℕ, n > 0 ∧ Even n ∧ 20 ∣ n ∧ 60 ∣ n ∧ ∀ m : ℕ, m > 0 → Even m → 20 ∣ m → 60 ∣ m → n ≤ m :=
by sorry

end smallest_even_divisible_by_20_and_60_l2988_298874


namespace son_work_time_l2988_298882

theorem son_work_time (man_time son_time combined_time : ℚ) : 
  man_time = 6 →
  combined_time = 3 →
  1 / man_time + 1 / son_time = 1 / combined_time →
  son_time = 6 :=
by
  sorry

end son_work_time_l2988_298882


namespace uncertain_relationship_l2988_298808

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents the possible relationships between two lines -/
inductive LineRelationship
  | Parallel
  | Perpendicular
  | Skew

/-- Perpendicularity of two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- The relationship between two lines -/
def relationship (l1 l2 : Line3D) : LineRelationship := sorry

theorem uncertain_relationship 
  (l1 l2 l3 l4 : Line3D) 
  (h_distinct : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4)
  (h12 : perpendicular l1 l2)
  (h23 : perpendicular l2 l3)
  (h34 : perpendicular l3 l4) :
  ∃ (r : LineRelationship), relationship l1 l4 = r ∧ 
    (r = LineRelationship.Parallel ∨ 
     r = LineRelationship.Perpendicular ∨ 
     r = LineRelationship.Skew) :=
by sorry

end uncertain_relationship_l2988_298808


namespace bakery_pie_production_l2988_298888

/-- The number of pies a bakery can make in one hour, given specific pricing and profit conditions. -/
theorem bakery_pie_production (piece_price : ℚ) (pieces_per_pie : ℕ) (pie_cost : ℚ) (total_profit : ℚ) 
  (h1 : piece_price = 4)
  (h2 : pieces_per_pie = 3)
  (h3 : pie_cost = 1/2)
  (h4 : total_profit = 138) :
  (total_profit / (piece_price * ↑pieces_per_pie - pie_cost) : ℚ) = 12 := by
  sorry

end bakery_pie_production_l2988_298888


namespace subset_implies_a_geq_four_l2988_298804

theorem subset_implies_a_geq_four (a : ℝ) :
  let A : Set ℝ := {x | 1 < x ∧ x < 2}
  let B : Set ℝ := {x | x^2 - a*x + 3 ≤ 0}
  A ⊆ B → a ≥ 4 := by
  sorry

end subset_implies_a_geq_four_l2988_298804


namespace max_value_of_f_on_interval_l2988_298861

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ f c = 2 ∧ ∀ x ∈ interval, f x ≤ f c :=
sorry

end max_value_of_f_on_interval_l2988_298861


namespace derivative_of_sqrt_at_one_l2988_298849

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem derivative_of_sqrt_at_one :
  deriv f 1 = (1 : ℝ) / 2 := by sorry

end derivative_of_sqrt_at_one_l2988_298849


namespace positive_product_of_positive_factors_l2988_298898

theorem positive_product_of_positive_factors (a b : ℝ) : a > 0 → b > 0 → a * b > 0 := by
  sorry

end positive_product_of_positive_factors_l2988_298898


namespace coin_flip_probability_l2988_298824

/-- The probability of a coin landing tails up -/
def ProbTails (coin : Nat) : ℚ :=
  match coin with
  | 1 => 3/4  -- Coin A
  | 2 => 1/2  -- Coin B
  | 3 => 1/4  -- Coin C
  | _ => 0    -- Invalid coin number

/-- The probability of the desired outcome -/
def DesiredOutcome : ℚ :=
  ProbTails 1 * ProbTails 2 * (1 - ProbTails 3)

theorem coin_flip_probability :
  DesiredOutcome = 9/32 := by
  sorry

end coin_flip_probability_l2988_298824


namespace digit_sum_divisibility_l2988_298897

theorem digit_sum_divisibility (n k : ℕ) (hn : n > 0) (hk : k ≥ n) (h3 : ¬3 ∣ n) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ (∃ digits : List ℕ, m.digits 10 = digits ∧ digits.sum = k) :=
sorry

end digit_sum_divisibility_l2988_298897


namespace student_count_l2988_298837

theorem student_count (avg_student_age avg_with_teacher teacher_age : ℝ) 
  (h1 : avg_student_age = 15)
  (h2 : avg_with_teacher = 16)
  (h3 : teacher_age = 46) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * avg_student_age + teacher_age = (n + 1 : ℝ) * avg_with_teacher ∧
    n = 30 := by
  sorry

end student_count_l2988_298837


namespace problem_solution_l2988_298872

/-- Given f(x) = ax^2 + bx where a ≠ 0 and f(2) = 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem problem_solution (a b : ℝ) (ha : a ≠ 0) :
  (f a b 2 = 0) →
  /- Part I -/
  (∃! x, f a b x - x = 0) →
  (∀ x, f a b x = -1/2 * x^2 + x) ∧
  /- Part II -/
  (a = 1 →
    (∀ x ∈ Set.Icc (-1) 2, f 1 b x ≤ 3) ∧
    (∀ x ∈ Set.Icc (-1) 2, f 1 b x ≥ -1) ∧
    (∃ x ∈ Set.Icc (-1) 2, f 1 b x = 3) ∧
    (∃ x ∈ Set.Icc (-1) 2, f 1 b x = -1)) ∧
  /- Part III -/
  ((∀ x ≥ 2, f a b x ≥ 2 - a) → a ≥ 2) := by
  sorry

end problem_solution_l2988_298872


namespace initial_nails_l2988_298864

theorem initial_nails (found_nails : ℕ) (nails_to_buy : ℕ) (total_nails : ℕ) 
  (h1 : found_nails = 144)
  (h2 : nails_to_buy = 109)
  (h3 : total_nails = 500)
  : total_nails = found_nails + nails_to_buy + 247 := by
  sorry

end initial_nails_l2988_298864


namespace shape_to_square_cut_l2988_298878

/-- Represents a shape with a given area -/
structure Shape :=
  (area : ℝ)

/-- Represents a cut of a shape into three parts -/
structure Cut (s : Shape) :=
  (part1 : Shape)
  (part2 : Shape)
  (part3 : Shape)
  (sum_area : part1.area + part2.area + part3.area = s.area)

/-- Predicate to check if three shapes can form a square -/
def CanFormSquare (p1 p2 p3 : Shape) : Prop :=
  ∃ (side : ℝ), side > 0 ∧ p1.area + p2.area + p3.area = side * side

/-- Theorem stating that any shape can be cut into three parts that form a square -/
theorem shape_to_square_cut (s : Shape) : 
  ∃ (c : Cut s), CanFormSquare c.part1 c.part2 c.part3 := by
  sorry

#check shape_to_square_cut

end shape_to_square_cut_l2988_298878


namespace two_white_balls_probability_l2988_298842

def total_balls : ℕ := 9
def white_balls : ℕ := 5
def black_balls : ℕ := 4

def prob_first_white : ℚ := white_balls / total_balls
def prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)

def prob_two_white : ℚ := prob_first_white * prob_second_white

theorem two_white_balls_probability :
  prob_two_white = 5 / 18 := by sorry

end two_white_balls_probability_l2988_298842


namespace carlos_baseball_cards_l2988_298890

theorem carlos_baseball_cards :
  ∀ (jorge matias carlos : ℕ),
    jorge = matias →
    matias = carlos - 6 →
    jorge + matias + carlos = 48 →
    carlos = 20 :=
by
  sorry

end carlos_baseball_cards_l2988_298890


namespace unique_solution_inequality_l2988_298838

theorem unique_solution_inequality (x : ℝ) :
  (x > 0 ∧ x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18) ↔ x = 6 :=
by sorry

end unique_solution_inequality_l2988_298838


namespace root_between_consecutive_integers_l2988_298895

theorem root_between_consecutive_integers :
  ∃ (A B : ℤ), B = A + 1 ∧
  ∃ (x : ℝ), A < x ∧ x < B ∧ x^3 + 5*x^2 - 3*x + 1 = 0 := by
  sorry

end root_between_consecutive_integers_l2988_298895


namespace routes_3x2_grid_l2988_298839

/-- The number of routes in a grid from top-left to bottom-right -/
def numRoutes (width height : ℕ) : ℕ :=
  Nat.choose (width + height) width

theorem routes_3x2_grid : numRoutes 3 2 = 10 := by
  sorry

end routes_3x2_grid_l2988_298839
