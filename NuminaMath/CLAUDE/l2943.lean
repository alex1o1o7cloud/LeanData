import Mathlib

namespace min_cards_l2943_294365

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem min_cards : ∃ (n a b c d e : ℕ),
  n = 63 ∧
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧
  (n - a) % 5 = 0 ∧
  (n - a - b) % 3 = 0 ∧
  (n - a - b - c) % 2 = 0 ∧
  n - a - b - c - d = e ∧
  ∀ (m : ℕ), m < n →
    ¬(∃ (a' b' c' d' e' : ℕ),
      is_prime a' ∧ is_prime b' ∧ is_prime c' ∧ is_prime d' ∧ is_prime e' ∧
      a' > b' ∧ b' > c' ∧ c' > d' ∧ d' > e' ∧
      (m - a') % 5 = 0 ∧
      (m - a' - b') % 3 = 0 ∧
      (m - a' - b' - c') % 2 = 0 ∧
      m - a' - b' - c' - d' = e') :=
by sorry

end min_cards_l2943_294365


namespace spade_ace_probability_l2943_294387

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function to determine if a card is a spade -/
def is_spade : Fin 52 → Bool := sorry

/-- A function to determine if a card is an ace -/
def is_ace : Fin 52 → Bool := sorry

/-- The number of spades in a standard deck -/
def spade_count : Nat := 13

/-- The number of aces in a standard deck -/
def ace_count : Nat := 4

/-- Theorem: The probability of drawing a spade as the first card
    and an ace as the second card from a standard 52-card deck is 1/52 -/
theorem spade_ace_probability (d : Deck) :
  (Finset.filter (λ c₁ => is_spade c₁) d.cards).card * 
  (Finset.filter (λ c₂ => is_ace c₂) d.cards).card / 
  (d.cards.card * (d.cards.card - 1)) = 1 / 52 := by
  sorry

end spade_ace_probability_l2943_294387


namespace water_level_rise_l2943_294347

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel. -/
theorem water_level_rise 
  (cube_edge : ℝ) 
  (vessel_length : ℝ) 
  (vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 5) 
  (h_vessel_length : vessel_length = 10) 
  (h_vessel_width : vessel_width = 5) : 
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 2.5 := by
  sorry

end water_level_rise_l2943_294347


namespace inverse_proportion_ratio_l2943_294336

/-- Given that x is inversely proportional to y, this function represents their relationship -/
def inverse_proportion (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_ratio 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0)
  (hxy₁ : inverse_proportion x₁ y₁)
  (hxy₂ : inverse_proportion x₂ y₂)
  (hx_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end inverse_proportion_ratio_l2943_294336


namespace unique_right_angle_point_implies_radius_one_l2943_294313

-- Define the circle C
def circle_C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- Define points A and B
def point_A : ℝ × ℝ := (-4, 0)
def point_B : ℝ × ℝ := (4, 0)

-- Define the right angle condition
def right_angle (P : ℝ × ℝ) : Prop :=
  let PA := (P.1 - point_A.1, P.2 - point_A.2)
  let PB := (P.1 - point_B.1, P.2 - point_B.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0

-- Main theorem
theorem unique_right_angle_point_implies_radius_one (r : ℝ) :
  (∃! P, P ∈ circle_C r ∧ right_angle P) → r = 1 := by
  sorry

end unique_right_angle_point_implies_radius_one_l2943_294313


namespace davids_english_marks_l2943_294330

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem stating that David's marks in English are 76 --/
theorem davids_english_marks (marks : Marks) :
  marks.mathematics = 65 →
  marks.physics = 82 →
  marks.chemistry = 67 →
  marks.biology = 85 →
  average [marks.english, marks.mathematics, marks.physics, marks.chemistry, marks.biology] = 75 →
  marks.english = 76 := by
  sorry


end davids_english_marks_l2943_294330


namespace calculation_proof_l2943_294374

theorem calculation_proof (a b : ℝ) (h1 : a = 7) (h2 : b = 3) : 
  ((a^3 + b^3) / (a^2 - a*b + b^2) = 10) ∧ ((a^2 + b^2) / (a + b) = 5.8) := by
  sorry

end calculation_proof_l2943_294374


namespace greatest_a_value_l2943_294308

theorem greatest_a_value (a : ℝ) : 
  (9 * Real.sqrt ((3 * a)^2 + 1^2) - 9 * a^2 - 1) / (Real.sqrt (1 + 3 * a^2) + 2) = 3 →
  a ≤ Real.sqrt (13/3) :=
by sorry

end greatest_a_value_l2943_294308


namespace seventh_term_is_25_over_3_l2943_294376

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 7
  sixth_term : a + 5*d = 7

/-- The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_is_25_over_3 (seq : ArithmeticSequence) : 
  seq.a + 6*seq.d = 25/3 := by
  sorry

end seventh_term_is_25_over_3_l2943_294376


namespace starting_lineup_count_l2943_294392

def total_players : ℕ := 20
def point_guards : ℕ := 1
def other_players : ℕ := 7

def starting_lineup_combinations : ℕ := total_players * (Nat.choose (total_players - point_guards) other_players)

theorem starting_lineup_count :
  starting_lineup_combinations = 1007760 :=
sorry

end starting_lineup_count_l2943_294392


namespace rectangular_prism_problem_l2943_294333

/-- Represents a rectangular prism with dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total number of faces of unit cubes in the prism. -/
def totalFaces (p : RectangularPrism) : ℕ := 6 * p.a * p.b * p.c

/-- Calculates the number of red faces in the prism. -/
def redFaces (p : RectangularPrism) : ℕ := 2 * (p.a * p.b + p.b * p.c + p.a * p.c)

/-- Theorem stating the conditions and result for the rectangular prism problem. -/
theorem rectangular_prism_problem (p : RectangularPrism) :
  p.a + p.b + p.c = 12 →
  3 * redFaces p = totalFaces p →
  p.a = 3 ∧ p.b = 4 ∧ p.c = 5 := by
  sorry


end rectangular_prism_problem_l2943_294333


namespace anya_balloons_count_l2943_294331

def total_balloons : ℕ := 672
def num_colors : ℕ := 4

theorem anya_balloons_count : 
  let balloons_per_color := total_balloons / num_colors
  let anya_balloons := balloons_per_color / 2
  anya_balloons = 84 := by sorry

end anya_balloons_count_l2943_294331


namespace wine_cork_price_difference_l2943_294394

/-- 
Given:
- The price of a bottle of wine with a cork
- The price of the cork
Prove that the difference in price between a bottle of wine with a cork and without a cork
is equal to the price of the cork.
-/
theorem wine_cork_price_difference 
  (price_with_cork : ℝ) 
  (price_cork : ℝ) 
  (h1 : price_with_cork = 2.10)
  (h2 : price_cork = 0.05) :
  price_with_cork - (price_with_cork - price_cork) = price_cork :=
by sorry

end wine_cork_price_difference_l2943_294394


namespace yellow_highlighters_count_l2943_294391

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters : ℕ := 11 - (4 + 5)

/-- The total number of highlighters -/
def total_highlighters : ℕ := 11

/-- The number of pink highlighters -/
def pink_highlighters : ℕ := 4

/-- The number of blue highlighters -/
def blue_highlighters : ℕ := 5

theorem yellow_highlighters_count :
  yellow_highlighters = 2 :=
by sorry

end yellow_highlighters_count_l2943_294391


namespace base_of_second_term_l2943_294359

theorem base_of_second_term (e : ℕ) (base : ℚ) 
  (h1 : e = 35)
  (h2 : (1/5 : ℚ)^e * base^18 = 1/(2*(10^35))) :
  base = 1/4 := by sorry

end base_of_second_term_l2943_294359


namespace condition_necessary_not_sufficient_l2943_294358

/-- The equation of a potential ellipse -/
def is_potential_ellipse (m n x y : ℝ) : Prop :=
  x^2 / m + y^2 / n = 1

/-- The condition mn > 0 -/
def condition_mn_positive (m n : ℝ) : Prop :=
  m * n > 0

/-- Definition of an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem condition_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → condition_mn_positive m n) ∧
  (∃ m n : ℝ, condition_mn_positive m n ∧ ¬is_ellipse m n) :=
by sorry

end condition_necessary_not_sufficient_l2943_294358


namespace max_dots_on_surface_l2943_294306

/-- The sum of dots on a standard die -/
def standardDieSum : ℕ := 21

/-- The maximum number of dots visible on a die with 5 visible faces -/
def maxDotsOn5Faces : ℕ := 20

/-- The number of dots visible on a die with 4 visible faces -/
def dotsOn4Faces : ℕ := 14

/-- The maximum number of dots visible on a die with 2 visible faces -/
def maxDotsOn2Faces : ℕ := 11

/-- The number of dice with 5 visible faces -/
def numDice5Faces : ℕ := 6

/-- The number of dice with 4 visible faces -/
def numDice4Faces : ℕ := 5

/-- The number of dice with 2 visible faces -/
def numDice2Faces : ℕ := 2

theorem max_dots_on_surface :
  numDice5Faces * maxDotsOn5Faces +
  numDice4Faces * dotsOn4Faces +
  numDice2Faces * maxDotsOn2Faces = 212 :=
by sorry

end max_dots_on_surface_l2943_294306


namespace no_consecutive_product_for_nine_power_minus_seven_l2943_294316

theorem no_consecutive_product_for_nine_power_minus_seven :
  ∀ n : ℕ, ¬∃ k : ℕ, 9^n - 7 = k * (k + 1) :=
by sorry

end no_consecutive_product_for_nine_power_minus_seven_l2943_294316


namespace homes_cleaned_l2943_294318

theorem homes_cleaned (earning_per_home : ℝ) (total_earned : ℝ) (h1 : earning_per_home = 46.0) (h2 : total_earned = 12696) :
  total_earned / earning_per_home = 276 := by
  sorry

end homes_cleaned_l2943_294318


namespace symmetric_point_xoz_plane_l2943_294317

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in three-dimensional space -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetric_point_xoz_plane :
  let A : Point3D := ⟨-1, 2, 3⟩
  let Q : Point3D := symmetricPointXOZ A
  Q = ⟨-1, -2, 3⟩ := by sorry

end symmetric_point_xoz_plane_l2943_294317


namespace cylinder_minus_cones_volume_l2943_294380

/-- The volume of space in a cylinder not occupied by three cones -/
theorem cylinder_minus_cones_volume (h_cyl : ℝ) (r_cyl : ℝ) (h_cone : ℝ) (r_cone : ℝ) :
  h_cyl = 36 →
  r_cyl = 10 →
  h_cone = 18 →
  r_cone = 10 →
  (π * r_cyl^2 * h_cyl) - 3 * (1/3 * π * r_cone^2 * h_cone) = 1800 * π :=
by sorry

end cylinder_minus_cones_volume_l2943_294380


namespace password_count_l2943_294312

/-- The number of possible values for the last two digits of a birth year. -/
def year_choices : ℕ := 100

/-- The number of possible values for the birth month. -/
def month_choices : ℕ := 12

/-- The number of possible values for the birth date. -/
def day_choices : ℕ := 31

/-- The total number of possible six-digit passwords. -/
def total_passwords : ℕ := year_choices * month_choices * day_choices

theorem password_count : total_passwords = 37200 := by
  sorry

end password_count_l2943_294312


namespace num_regions_correct_l2943_294372

/-- A structure representing a collection of planes in 3D space -/
structure PlaneCollection where
  n : ℕ
  intersection_of_two : ∀ p q : Fin n, p ≠ q → Line
  intersection_of_three : ∀ p q r : Fin n, p ≠ q ∧ q ≠ r ∧ p ≠ r → Point
  no_four_intersect : ∀ p q r s : Fin n, p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ p ≠ r ∧ p ≠ s ∧ q ≠ s → ¬ Point

/-- The number of non-overlapping regions created by n planes -/
def num_regions (pc : PlaneCollection) : ℕ :=
  (pc.n^3 + 5*pc.n + 6) / 6

/-- Theorem stating that the number of regions is correct -/
theorem num_regions_correct (pc : PlaneCollection) :
  num_regions pc = (pc.n^3 + 5*pc.n + 6) / 6 := by
  sorry

end num_regions_correct_l2943_294372


namespace square_reciprocal_sum_l2943_294327

theorem square_reciprocal_sum (m : ℝ) (h : m + 1/m = 5) : 
  m^2 + 1/m^2 + 4 = 27 := by
sorry

end square_reciprocal_sum_l2943_294327


namespace kim_shoes_problem_l2943_294353

theorem kim_shoes_problem (num_pairs : ℕ) (prob_same_color : ℚ) : 
  num_pairs = 7 →
  prob_same_color = 7692307692307693 / 100000000000000000 →
  (1 : ℚ) / (num_pairs * 2 - 1) = prob_same_color →
  num_pairs * 2 = 14 := by
  sorry

end kim_shoes_problem_l2943_294353


namespace pizza_order_theorem_l2943_294321

theorem pizza_order_theorem : 
  (1 : ℚ) / 2 + (1 : ℚ) / 3 + (1 : ℚ) / 6 = 1 := by sorry

end pizza_order_theorem_l2943_294321


namespace boat_speed_l2943_294395

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 11) 
  (h2 : against_stream = 3) : ℝ :=
by
  -- The speed of the boat in still water is 7 km/hr
  sorry

#check boat_speed

end boat_speed_l2943_294395


namespace quadratic_coefficient_l2943_294307

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 27 = (x + n)^2 + 3) → 
  b = 4 * Real.sqrt 6 := by
sorry

end quadratic_coefficient_l2943_294307


namespace inequality_proof_l2943_294314

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (a + b + c) / 3 :=
by sorry

end inequality_proof_l2943_294314


namespace quadratic_equation_solution_l2943_294342

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => x * (2 * x + 4) - (10 + 5 * x)
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 5/2 := by
sorry

end quadratic_equation_solution_l2943_294342


namespace min_value_theorem_l2943_294338

theorem min_value_theorem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 20 * b = 2) (h2 : c + 20 * d = 2) :
  (1 / a + 1 / (b * c * d)) ≥ 441 / 2 :=
by sorry

end min_value_theorem_l2943_294338


namespace max_total_score_is_four_l2943_294384

/-- Represents an instructor's scoring for a set of problems -/
structure InstructorScoring :=
  (scores : List ℕ)
  (one_count : ℕ)
  (h_scores : ∀ s ∈ scores, s = 0 ∨ s = 1)
  (h_one_count : one_count = 3)
  (h_one_count_correct : scores.count 1 = one_count)

/-- Calculates the rounded mean of three scores -/
def roundedMean (a b c : ℕ) : ℕ :=
  (a + b + c + 1) / 3

/-- Calculates the total score based on three instructors' scorings -/
def totalScore (i1 i2 i3 : InstructorScoring) : ℕ :=
  List.sum (List.zipWith3 roundedMean i1.scores i2.scores i3.scores)

/-- The main theorem stating that the maximum possible total score is 4 -/
theorem max_total_score_is_four (i1 i2 i3 : InstructorScoring) :
  totalScore i1 i2 i3 ≤ 4 :=
sorry

#check max_total_score_is_four

end max_total_score_is_four_l2943_294384


namespace max_crates_on_trip_l2943_294381

theorem max_crates_on_trip (crate_weight : ℝ) (max_weight : ℝ) (h1 : crate_weight ≥ 1250) (h2 : max_weight = 6250) :
  ⌊max_weight / crate_weight⌋ = 5 :=
sorry

end max_crates_on_trip_l2943_294381


namespace max_automobiles_on_ferry_l2943_294325

/-- Represents the capacity of the ferry in tons -/
def ferry_capacity : ℝ := 50

/-- Represents the minimum weight of an automobile in pounds -/
def min_auto_weight : ℝ := 1600

/-- Represents the conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2000

/-- Theorem stating the maximum number of automobiles that can be loaded onto the ferry -/
theorem max_automobiles_on_ferry :
  ⌊(ferry_capacity * tons_to_pounds) / min_auto_weight⌋ = 62 := by
  sorry

end max_automobiles_on_ferry_l2943_294325


namespace program_output_correct_l2943_294388

/-- The output function of Xiao Wang's program -/
def program_output (n : ℕ+) : ℚ :=
  n / (n^2 + 1)

/-- The theorem stating the correctness of the program output -/
theorem program_output_correct (n : ℕ+) :
  program_output n = n / (n^2 + 1) := by
  sorry

end program_output_correct_l2943_294388


namespace first_rewind_time_l2943_294398

theorem first_rewind_time (total_time second_rewind_time first_segment second_segment third_segment : ℕ) 
  (h1 : total_time = 120)
  (h2 : second_rewind_time = 15)
  (h3 : first_segment = 35)
  (h4 : second_segment = 45)
  (h5 : third_segment = 20) :
  total_time - (first_segment + second_segment + third_segment) - second_rewind_time = 5 := by
sorry

end first_rewind_time_l2943_294398


namespace ten_not_diff_of_squares_five_is_diff_of_squares_seven_is_diff_of_squares_eight_is_diff_of_squares_nine_is_diff_of_squares_l2943_294354

-- Define a function to check if a number is a difference of two squares
def is_diff_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

-- Theorem stating that 10 cannot be expressed as the difference of two squares
theorem ten_not_diff_of_squares : ¬ is_diff_of_squares 10 :=
sorry

-- Theorems stating that 5, 7, 8, and 9 can be expressed as the difference of two squares
theorem five_is_diff_of_squares : is_diff_of_squares 5 :=
sorry

theorem seven_is_diff_of_squares : is_diff_of_squares 7 :=
sorry

theorem eight_is_diff_of_squares : is_diff_of_squares 8 :=
sorry

theorem nine_is_diff_of_squares : is_diff_of_squares 9 :=
sorry

end ten_not_diff_of_squares_five_is_diff_of_squares_seven_is_diff_of_squares_eight_is_diff_of_squares_nine_is_diff_of_squares_l2943_294354


namespace brown_children_divisibility_l2943_294326

theorem brown_children_divisibility :
  ∃! n : ℕ, n ∈ Finset.range 10 ∧ ¬(7773 % n = 0) :=
by
  -- The proof goes here
  sorry

end brown_children_divisibility_l2943_294326


namespace distance_to_origin_l2943_294311

/-- The distance from point P(1, 2, 2) to the origin (0, 0, 0) is 3. -/
theorem distance_to_origin : Real.sqrt (1^2 + 2^2 + 2^2) = 3 := by
  sorry


end distance_to_origin_l2943_294311


namespace minimum_apples_in_basket_l2943_294309

theorem minimum_apples_in_basket (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) → n ≥ 62 :=
by sorry

end minimum_apples_in_basket_l2943_294309


namespace sprinkle_cans_remaining_l2943_294366

theorem sprinkle_cans_remaining (initial : ℕ) (final : ℕ) 
  (h1 : initial = 12) 
  (h2 : final = initial / 2 - 3) : 
  final = 3 := by
  sorry

end sprinkle_cans_remaining_l2943_294366


namespace product_of_cubic_fractions_l2943_294341

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 84 := by
  sorry

end product_of_cubic_fractions_l2943_294341


namespace sum_and_ratio_implies_difference_l2943_294339

theorem sum_and_ratio_implies_difference (x y : ℝ) : 
  x + y = 540 → x / y = 0.75 → y - x = 77.143 := by
sorry

end sum_and_ratio_implies_difference_l2943_294339


namespace frank_cookie_fraction_l2943_294303

/-- Given the number of cookies for Millie, calculate Mike's cookies -/
def mikeCookies (millieCookies : ℕ) : ℕ := 3 * millieCookies

/-- Calculate the fraction of Frank's cookies compared to Mike's -/
def frankFraction (frankCookies millieCookies : ℕ) : ℚ :=
  frankCookies / (mikeCookies millieCookies)

/-- Theorem: Frank's fraction of cookies compared to Mike's is 1/4 -/
theorem frank_cookie_fraction :
  frankFraction 3 4 = 1 / 4 := by
  sorry

end frank_cookie_fraction_l2943_294303


namespace reciprocal_of_negative_two_thirds_l2943_294386

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let y : ℚ := -3/2
  (x * y = 1) → y = x⁻¹ := by
  sorry

end reciprocal_of_negative_two_thirds_l2943_294386


namespace lines_are_parallel_l2943_294385

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem lines_are_parallel : 
  let line1 : Line := { a := 2, b := -1, c := 7 }
  let line2 : Line := { a := 2, b := -1, c := 1 }
  parallel line1 line2 := by
  sorry

end lines_are_parallel_l2943_294385


namespace certain_number_value_l2943_294373

/-- Represents the number system in the certain country -/
structure CountryNumber where
  value : ℕ

/-- Multiplication operation in the country's number system -/
def country_mul (a b : CountryNumber) : CountryNumber :=
  ⟨a.value * b.value⟩

/-- Division operation in the country's number system -/
def country_div (a b : CountryNumber) : CountryNumber :=
  ⟨a.value / b.value⟩

/-- Equality in the country's number system -/
def country_eq (a b : CountryNumber) : Prop :=
  a.value = b.value

theorem certain_number_value :
  ∀ (eight seven five : CountryNumber),
    country_eq (country_div eight seven) five →
    ∀ (x : CountryNumber),
      country_eq (country_div x ⟨5⟩) ⟨35⟩ →
      country_eq x ⟨175⟩ :=
by sorry

end certain_number_value_l2943_294373


namespace cos_20_minus_cos_40_l2943_294348

theorem cos_20_minus_cos_40 : Real.cos (20 * Real.pi / 180) - Real.cos (40 * Real.pi / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end cos_20_minus_cos_40_l2943_294348


namespace roots_sum_angle_l2943_294396

theorem roots_sum_angle (a : ℝ) (α β : ℝ) : 
  a > 2 → 
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  (Real.tan α)^2 + 3*a*(Real.tan α) + 3*a + 1 = 0 →
  (Real.tan β)^2 + 3*a*(Real.tan β) + 3*a + 1 = 0 →
  α + β = π/4 := by
sorry

end roots_sum_angle_l2943_294396


namespace arithmetic_sequence_12th_term_l2943_294334

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 9)
  (h_6 : a 6 = 15) :
  a 12 = 27 := by
  sorry

end arithmetic_sequence_12th_term_l2943_294334


namespace no_real_roots_of_quadratic_l2943_294383

theorem no_real_roots_of_quadratic : 
  ¬∃ (x : ℝ), x^2 - 4*x + 8 = 0 := by
sorry

end no_real_roots_of_quadratic_l2943_294383


namespace largest_prime_divisor_l2943_294349

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

-- Define the number 2102012 in base 3
def number_base_3 : List Nat := [2, 1, 0, 2, 0, 1, 2]

-- Convert the base 3 number to decimal
def number_decimal : Nat := base_3_to_decimal number_base_3

-- Statement to prove
theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ number_decimal ∧ 
  ∀ (q : Nat), Nat.Prime q → q ∣ number_decimal → q ≤ p :=
by sorry

end largest_prime_divisor_l2943_294349


namespace contrapositive_isosceles_angles_l2943_294329

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := sorry

-- Define what it means for two angles of a triangle to be equal
def twoAnglesEqual (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_angles (t : Triangle) :
  (¬(isIsosceles t) → ¬(twoAnglesEqual t)) ↔ (twoAnglesEqual t → isIsosceles t) := by
  sorry

end contrapositive_isosceles_angles_l2943_294329


namespace john_soup_vegetables_l2943_294324

/-- Represents the weights of vegetables used in John's soup recipe --/
structure SoupVegetables where
  carrots : ℝ
  potatoes : ℝ
  bell_peppers : ℝ

/-- Calculates the total weight of vegetables used in the soup --/
def total_vegetable_weight (v : SoupVegetables) : ℝ :=
  v.carrots + v.potatoes + v.bell_peppers

/-- Represents John's soup recipe --/
structure SoupRecipe where
  beef_bought : ℝ
  beef_unused : ℝ
  vegetables : SoupVegetables

/-- Theorem stating the correct weights of vegetables in John's soup --/
theorem john_soup_vegetables (recipe : SoupRecipe) : 
  recipe.beef_bought = 4 ∧ 
  recipe.beef_unused = 1 ∧ 
  total_vegetable_weight recipe.vegetables = 2 * (recipe.beef_bought - recipe.beef_unused) ∧
  recipe.vegetables.carrots = recipe.vegetables.potatoes ∧
  recipe.vegetables.bell_peppers = 2 * recipe.vegetables.carrots →
  recipe.vegetables = SoupVegetables.mk 1.5 1.5 3 := by
  sorry


end john_soup_vegetables_l2943_294324


namespace quadratic_and_squared_equation_solutions_l2943_294350

theorem quadratic_and_squared_equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 - 3 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 3 * x₂ + 1 = 0) ∧ x₁ = 1/2 ∧ x₂ = 1) ∧
  (∃ y₁ y₂ : ℝ, ((y₁ - 2)^2 = (2 * y₁ + 3)^2) ∧ ((y₂ - 2)^2 = (2 * y₂ + 3)^2) ∧ y₁ = -5 ∧ y₂ = -1/3) :=
by sorry

end quadratic_and_squared_equation_solutions_l2943_294350


namespace percentage_problem_l2943_294371

theorem percentage_problem (P : ℝ) : 
  (0.20 * 30 = P / 100 * 16 + 2) → P = 25 := by
  sorry

end percentage_problem_l2943_294371


namespace triangle_min_perimeter_l2943_294375

theorem triangle_min_perimeter (a b c : ℕ) : 
  a = 24 → b = 37 → c > 0 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x : ℕ, x > 0 → x + b > a ∧ a + b > x ∧ a + x > b → a + b + x ≥ a + b + c) →
  a + b + c = 75 := by sorry

end triangle_min_perimeter_l2943_294375


namespace eighth_prime_is_19_l2943_294320

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Theorem: The 8th prime number is 19 -/
theorem eighth_prime_is_19 : nthPrime 8 = 19 := by sorry

end eighth_prime_is_19_l2943_294320


namespace dog_reachable_area_l2943_294393

/-- The area outside a regular hexagon reachable by a tethered dog -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 2 → rope_length = 5 → 
  (π * rope_length^2 : ℝ) = 25 * π := by
  sorry

#check dog_reachable_area

end dog_reachable_area_l2943_294393


namespace min_value_expression_min_value_achievable_l2943_294332

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b) + (b / c) + (c / a) + Real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2) ≥ 3 + Real.sqrt 3 :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / b) + (b / c) + (c / a) + Real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2) = 3 + Real.sqrt 3 :=
by sorry

end min_value_expression_min_value_achievable_l2943_294332


namespace bakery_problem_l2943_294397

/-- Calculates the number of cookies remaining in the last bag -/
def cookies_in_last_bag (total_cookies : ℕ) (bag_capacity : ℕ) : ℕ :=
  total_cookies % bag_capacity

theorem bakery_problem (total_cookies : ℕ) (choc_chip : ℕ) (oatmeal : ℕ) (sugar : ℕ) 
  (bag_capacity : ℕ) (h1 : total_cookies = choc_chip + oatmeal + sugar) 
  (h2 : choc_chip = 154) (h3 : oatmeal = 86) (h4 : sugar = 52) (h5 : bag_capacity = 16) :
  (cookies_in_last_bag choc_chip bag_capacity = 10) ∧ 
  (cookies_in_last_bag oatmeal bag_capacity = 6) ∧ 
  (cookies_in_last_bag sugar bag_capacity = 4) := by
  sorry

#eval cookies_in_last_bag 154 16  -- Should output 10
#eval cookies_in_last_bag 86 16   -- Should output 6
#eval cookies_in_last_bag 52 16   -- Should output 4

end bakery_problem_l2943_294397


namespace power_of_point_formula_l2943_294356

/-- The power of a point with respect to a circle -/
def power_of_point (d R : ℝ) : ℝ := d^2 - R^2

/-- Theorem: The power of a point with respect to a circle is d^2 - R^2,
    where d is the distance from the point to the center of the circle,
    and R is the radius of the circle. -/
theorem power_of_point_formula (d R : ℝ) :
  power_of_point d R = d^2 - R^2 := by sorry

end power_of_point_formula_l2943_294356


namespace jacob_calorie_consumption_l2943_294310

/-- Jacob's calorie consumption problem -/
theorem jacob_calorie_consumption (planned_max : ℕ) (breakfast lunch dinner : ℕ) 
  (h1 : planned_max < 1800)
  (h2 : breakfast = 400)
  (h3 : lunch = 900)
  (h4 : dinner = 1100) :
  breakfast + lunch + dinner - planned_max = 600 :=
by sorry

end jacob_calorie_consumption_l2943_294310


namespace smallest_value_of_roots_sum_l2943_294368

/-- 
Given a quadratic equation x^2 - t*x + q with roots α and β,
where α + β = α^2 + β^2 = α^3 + β^3 = ... = α^2010 + β^2010,
the smallest possible value of 1/α^2011 + 1/β^2011 is 2.
-/
theorem smallest_value_of_roots_sum (t q α β : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → α^n + β^n = α + β) →
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  (∀ x : ℝ, x^2 - t*x + q = 0 → x = α ∨ x = β) →
  (1 / α^2011 + 1 / β^2011) ≥ 2 :=
by sorry

end smallest_value_of_roots_sum_l2943_294368


namespace rectangular_to_polar_conversion_l2943_294335

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 4 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 4 * Real.sqrt 6) ∧ 
  (θ = π / 8) ∧ 
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * π) :=
by sorry

end rectangular_to_polar_conversion_l2943_294335


namespace smallest_k_for_fraction_equation_l2943_294363

theorem smallest_k_for_fraction_equation : 
  (∃ k : ℕ, k > 0 ∧ 
    (∃ a b : ℕ, a > 500000 ∧ 
      1 / (a : ℚ) + 1 / ((a + k) : ℚ) = 1 / (b : ℚ))) ∧ 
  (∀ k : ℕ, k > 0 → k < 1001 → 
    ¬(∃ a b : ℕ, a > 500000 ∧ 
      1 / (a : ℚ) + 1 / ((a + k) : ℚ) = 1 / (b : ℚ))) := by
  sorry

end smallest_k_for_fraction_equation_l2943_294363


namespace bank_interest_calculation_l2943_294389

theorem bank_interest_calculation 
  (initial_deposit : ℝ) 
  (interest_rate : ℝ) 
  (years : ℕ) 
  (h1 : initial_deposit = 5600) 
  (h2 : interest_rate = 0.07) 
  (h3 : years = 2) : 
  initial_deposit + years * (initial_deposit * interest_rate) = 6384 :=
by
  sorry

end bank_interest_calculation_l2943_294389


namespace point_side_line_range_l2943_294300

/-- Given that the points (3,-1) and (-4,-3) are on the same side of the line 3x-2y+a=0,
    prove that the range of values for a is (-∞,-11) ∪ (6,+∞). -/
theorem point_side_line_range (a : ℝ) : 
  (3 * 3 - 2 * (-1) + a) * (3 * (-4) - 2 * (-3) + a) > 0 ↔ 
  a ∈ Set.Iio (-11) ∪ Set.Ioi 6 :=
sorry

end point_side_line_range_l2943_294300


namespace map_distance_calculation_l2943_294379

/-- Given a map scale and a measured distance on the map, calculate the actual distance in kilometers. -/
theorem map_distance_calculation (scale : ℚ) (map_distance : ℚ) (actual_distance : ℚ) :
  scale = 1 / 1000000 →
  map_distance = 12 →
  actual_distance = map_distance / scale / 100000 →
  actual_distance = 120 := by
  sorry

end map_distance_calculation_l2943_294379


namespace perpendicular_lengths_determine_side_length_l2943_294319

/-- An equilateral triangle with a point inside and perpendiculars to the sides -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The lengths of the perpendiculars from the interior point to the sides
  perp_length_1 : ℝ
  perp_length_2 : ℝ
  perp_length_3 : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  side_length_pos : 0 < side_length
  perp_lengths_pos : 0 < perp_length_1 ∧ 0 < perp_length_2 ∧ 0 < perp_length_3
  perp_sum_bound : perp_length_1 + perp_length_2 + perp_length_3 < side_length * (3 / 2)

/-- The theorem stating the relationship between the perpendicular lengths and the side length -/
theorem perpendicular_lengths_determine_side_length 
  (t : EquilateralTriangleWithPoint) 
  (h1 : t.perp_length_1 = 2) 
  (h2 : t.perp_length_2 = 3) 
  (h3 : t.perp_length_3 = 4) : 
  t.side_length = 6 * Real.sqrt 3 := by
  sorry

end perpendicular_lengths_determine_side_length_l2943_294319


namespace smallest_n_with_three_pairs_l2943_294351

/-- The function f(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 = n -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 50 is the smallest positive integer n for which f(n) = 3 -/
theorem smallest_n_with_three_pairs : ∀ k : ℕ, 0 < k → k < 50 → f k ≠ 3 ∧ f 50 = 3 := by
  sorry

end smallest_n_with_three_pairs_l2943_294351


namespace power_division_equivalence_l2943_294337

theorem power_division_equivalence : 8^15 / 64^5 = 32768 := by
  have h1 : 8 = 2^3 := by sorry
  have h2 : 64 = 2^6 := by sorry
  sorry

end power_division_equivalence_l2943_294337


namespace magnitude_of_a_is_two_l2943_294352

def a (x : ℝ) : Fin 2 → ℝ := ![1, x]
def b (x : ℝ) : Fin 2 → ℝ := ![-1, x]

theorem magnitude_of_a_is_two (x : ℝ) :
  (∀ i : Fin 2, ((2 • a x - b x) • b x) = 0) → 
  Real.sqrt ((a x 0) ^ 2 + (a x 1) ^ 2) = 2 := by sorry

end magnitude_of_a_is_two_l2943_294352


namespace light_path_in_cube_l2943_294340

theorem light_path_in_cube (cube_side : ℝ) (reflect_point_dist1 : ℝ) (reflect_point_dist2 : ℝ) :
  cube_side = 12 ∧ reflect_point_dist1 = 7 ∧ reflect_point_dist2 = 5 →
  ∃ (m n : ℕ), 
    (m = 12 ∧ n = 218) ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ n)) ∧
    (m * Real.sqrt n = 12 * cube_side) :=
by sorry

end light_path_in_cube_l2943_294340


namespace parallel_lines_m_value_l2943_294399

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let l1 : Line := ⟨6, m, -1⟩
  let l2 : Line := ⟨2, -1, 1⟩
  parallel l1 l2 → m = -3 := by
  sorry

end parallel_lines_m_value_l2943_294399


namespace prob_at_least_75_cents_is_correct_l2943_294302

-- Define the coin types and their quantities
structure CoinBox :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (quarters : Nat)

-- Define the function to calculate the total number of coins
def totalCoins (box : CoinBox) : Nat :=
  box.pennies + box.nickels + box.dimes + box.quarters

-- Define the function to calculate the number of ways to choose 7 coins
def waysToChoose7 (box : CoinBox) : Nat :=
  Nat.choose (totalCoins box) 7

-- Define the probability of drawing coins worth at least 75 cents
def probAtLeast75Cents (box : CoinBox) : Rat :=
  2450 / waysToChoose7 box

-- State the theorem
theorem prob_at_least_75_cents_is_correct (box : CoinBox) :
  box.pennies = 4 ∧ box.nickels = 5 ∧ box.dimes = 7 ∧ box.quarters = 3 →
  probAtLeast75Cents box = 2450 / 50388 :=
by sorry

end prob_at_least_75_cents_is_correct_l2943_294302


namespace gas_consumption_reduction_l2943_294305

theorem gas_consumption_reduction (initial_price : ℝ) (initial_consumption : ℝ) 
  (h1 : initial_price > 0) (h2 : initial_consumption > 0) :
  let price_after_increases := initial_price * 1.3 * 1.2
  let new_consumption := initial_consumption * initial_price / price_after_increases
  let reduction_percentage := (initial_consumption - new_consumption) / initial_consumption * 100
  reduction_percentage = (1 - 1 / (1.3 * 1.2)) * 100 := by
  sorry

end gas_consumption_reduction_l2943_294305


namespace arithmetic_sequence_sum_l2943_294323

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 = -1 →
  a 3 = 4 →
  a 4 + a 5 = 17 := by
sorry

end arithmetic_sequence_sum_l2943_294323


namespace bennys_work_days_l2943_294315

theorem bennys_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days_worked : ℕ) : 
  hours_per_day = 3 →
  total_hours = 18 →
  days_worked * hours_per_day = total_hours →
  days_worked = 6 := by
sorry

end bennys_work_days_l2943_294315


namespace quadratic_sum_of_constants_l2943_294304

theorem quadratic_sum_of_constants (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 49 = (x + b)^2 + c) → b + c = -61 := by
  sorry

end quadratic_sum_of_constants_l2943_294304


namespace production_line_b_units_l2943_294328

theorem production_line_b_units (total_units : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_units = 5000 →
  ratio_a = 1 →
  ratio_b = 2 →
  ratio_c = 2 →
  (total_units * ratio_b) / (ratio_a + ratio_b + ratio_c) = 2000 := by
sorry

end production_line_b_units_l2943_294328


namespace sector_area_l2943_294390

theorem sector_area (r : Real) (θ : Real) (h1 : r = Real.pi) (h2 : θ = 2 * Real.pi / 3) :
  (1 / 2) * r * r * θ = Real.pi^3 / 6 := by
  sorry

end sector_area_l2943_294390


namespace ellipse_param_sum_l2943_294360

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to the foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  dist_sum : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def computeEllipseParams (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem: for the given ellipse, the sum of its parameters is 18 -/
theorem ellipse_param_sum :
  let e := Ellipse.mk (4, 2) (10, 2) 10
  let params := computeEllipseParams e
  params.h + params.k + params.a + params.b = 18 :=
sorry

end ellipse_param_sum_l2943_294360


namespace train_crossing_time_l2943_294362

/-- Proves that a train 300 meters long, traveling at 90 km/hr, will take 12 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 300 →
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 12 := by
  sorry

end train_crossing_time_l2943_294362


namespace smaller_circle_radius_l2943_294364

theorem smaller_circle_radius (R : ℝ) (r : ℝ) : 
  R = 12 →  -- Larger circle radius is 12 meters
  4 * (2 * r) = 2 * R →  -- Four smaller circles' diameters equal larger circle's diameter
  r = 3  -- Radius of smaller circle is 3 meters
:= by sorry

end smaller_circle_radius_l2943_294364


namespace car_speed_problem_l2943_294367

/-- Given a car traveling for two hours with an average speed of 95 km/h
    and a second hour speed of 70 km/h, prove that the speed in the first hour is 120 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 70 ∧ average_speed = 95 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 120 := by
  sorry

end car_speed_problem_l2943_294367


namespace g_values_l2943_294369

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom sum_one : ∀ x, g x + f x = 1
axiom g_odd : ∀ x, g (x + 1) = -g (-x + 1)
axiom f_odd : ∀ x, f (2 - x) = -f (2 + x)

-- Define the theorem
theorem g_values : g 0 = -1 ∧ g 1 = 0 ∧ g 2 = 1 := by
  sorry

end g_values_l2943_294369


namespace parallel_plane_intersection_lines_parallel_l2943_294301

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation between a plane and a line
variable (intersects : Plane → Plane → Line → Prop)

-- Theorem statement
theorem parallel_plane_intersection_lines_parallel 
  (P1 P2 P3 : Plane) (l1 l2 : Line) :
  parallel_planes P1 P2 →
  intersects P3 P1 l1 →
  intersects P3 P2 l2 →
  -- Conclusion: l1 and l2 are parallel
  parallel_planes P1 P2 := by sorry

end parallel_plane_intersection_lines_parallel_l2943_294301


namespace function_value_at_eight_l2943_294377

theorem function_value_at_eight (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) :
  f 8 = 26 := by
sorry

end function_value_at_eight_l2943_294377


namespace croissant_price_is_three_l2943_294322

def discount_rate : ℝ := 0.1
def discount_threshold : ℝ := 50
def num_quiches : ℕ := 2
def price_per_quiche : ℝ := 15
def num_croissants : ℕ := 6
def num_biscuits : ℕ := 6
def price_per_biscuit : ℝ := 2
def final_price : ℝ := 54

def total_cost (price_per_croissant : ℝ) : ℝ :=
  num_quiches * price_per_quiche + 
  num_croissants * price_per_croissant + 
  num_biscuits * price_per_biscuit

theorem croissant_price_is_three :
  ∃ (price_per_croissant : ℝ),
    price_per_croissant = 3 ∧
    total_cost price_per_croissant > discount_threshold ∧
    (1 - discount_rate) * total_cost price_per_croissant = final_price :=
sorry

end croissant_price_is_three_l2943_294322


namespace expected_digits_20_sided_die_l2943_294357

/-- The expected number of digits when rolling a fair 20-sided die with numbers 1 to 20 -/
theorem expected_digits_20_sided_die : 
  let die_faces : Finset ℕ := Finset.range 20
  let one_digit_count : ℕ := (die_faces.filter (λ n => n < 10)).card
  let two_digit_count : ℕ := (die_faces.filter (λ n => n ≥ 10)).card
  let total_faces : ℕ := die_faces.card
  let expected_value : ℚ := (one_digit_count * 1 + two_digit_count * 2) / total_faces
  expected_value = 31 / 20 := by
sorry

end expected_digits_20_sided_die_l2943_294357


namespace volume_cube_sphere_region_l2943_294361

/-- The volume of the region within a cube of side length 4 cm, outside an inscribed sphere
    tangent to the cube, and closest to one vertex of the cube. -/
theorem volume_cube_sphere_region (π : ℝ) (h : π = Real.pi) :
  let a : ℝ := 4
  let cube_volume : ℝ := a ^ 3
  let sphere_radius : ℝ := a / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  let outside_sphere_volume : ℝ := cube_volume - sphere_volume
  let region_volume : ℝ := (1 / 8) * outside_sphere_volume
  region_volume = 8 * (1 - π / 6) :=
by sorry

end volume_cube_sphere_region_l2943_294361


namespace quadrilateral_area_l2943_294370

-- Define the radius of the large circle
def R : ℝ := 6

-- Define the centers of the smaller circles
structure Center where
  x : ℝ
  y : ℝ

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : Center
  B : Center
  C : Center
  D : Center

-- Define the condition that the circles touch
def circles_touch (q : Quadrilateral) : Prop := sorry

-- Define the condition that A and C touch at the center of the large circle
def AC_touch_center (q : Quadrilateral) : Prop := sorry

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) :
  circles_touch q →
  AC_touch_center q →
  area q = 24 := by
  sorry

end quadrilateral_area_l2943_294370


namespace volume_P4_l2943_294378

/-- Recursive definition of the volume of Pᵢ --/
def volume (i : ℕ) : ℚ :=
  match i with
  | 0 => 1
  | n + 1 => volume n + (4^n * (1 / 27))

/-- Theorem stating the volume of P₄ --/
theorem volume_P4 : volume 4 = 367 / 27 := by
  sorry

#eval volume 4

end volume_P4_l2943_294378


namespace f_two_zeros_sum_greater_than_two_l2943_294345

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a/2) * x^2 + (a-1) * x

theorem f_two_zeros_sum_greater_than_two (a : ℝ) (h : a > 2) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  f a x₁ = 0 ∧ f a x₂ = 0 ∧
  (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ∧
  x₁ + x₂ > 2 := by sorry

end f_two_zeros_sum_greater_than_two_l2943_294345


namespace perimeter_invariant_under_translation_l2943_294343

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a hexagon formed by the intersection of two equilateral triangles -/
structure IntersectionHexagon where
  triangle1 : EquilateralTriangle
  triangle2 : EquilateralTriangle

/-- Calculates the perimeter of the intersection hexagon -/
def perimeter (h : IntersectionHexagon) : ℝ :=
  sorry

/-- Represents a parallel translation of a triangle -/
def parallelTranslation (t : EquilateralTriangle) (v : ℝ × ℝ) : EquilateralTriangle :=
  sorry

/-- The theorem stating that the perimeter remains constant under parallel translation -/
theorem perimeter_invariant_under_translation 
  (h : IntersectionHexagon) 
  (v : ℝ × ℝ) 
  (h' : IntersectionHexagon := ⟨h.triangle1, parallelTranslation h.triangle2 v⟩) : 
  perimeter h = perimeter h' :=
sorry

end perimeter_invariant_under_translation_l2943_294343


namespace inequality_system_solution_l2943_294344

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end inequality_system_solution_l2943_294344


namespace a_10_value_l2943_294382

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_10_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 4 →
  a 6 = 6 →
  a 10 = 9 := by
sorry

end a_10_value_l2943_294382


namespace age_difference_l2943_294355

-- Define the ages
def katie_daughter_age : ℕ := 12
def lavinia_daughter_age : ℕ := katie_daughter_age - 10
def lavinia_son_age : ℕ := 2 * katie_daughter_age

-- Theorem statement
theorem age_difference : lavinia_son_age - lavinia_daughter_age = 22 := by
  sorry

end age_difference_l2943_294355


namespace quadruple_sum_square_l2943_294346

theorem quadruple_sum_square (a b c d m n : ℕ+) : 
  a^2 + b^2 + c^2 + d^2 = 1989 →
  a + b + c + d = m^2 →
  max a (max b (max c d)) = n^2 →
  m = 9 ∧ n = 6 := by
sorry

end quadruple_sum_square_l2943_294346
