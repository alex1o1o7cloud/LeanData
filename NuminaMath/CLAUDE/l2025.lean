import Mathlib

namespace mother_age_now_is_70_l2025_202511

/-- Jessica's age now -/
def jessica_age_now : ℕ := 40

/-- Years passed since mother's death -/
def years_passed : ℕ := 10

/-- Jessica's age when her mother died -/
def jessica_age_then : ℕ := jessica_age_now - years_passed

/-- Mother's age when she died -/
def mother_age_then : ℕ := 2 * jessica_age_then

/-- Mother's age now if she were alive -/
def mother_age_now : ℕ := mother_age_then + years_passed

theorem mother_age_now_is_70 : mother_age_now = 70 := by
  sorry

end mother_age_now_is_70_l2025_202511


namespace seniors_in_sample_is_fifty_l2025_202560

/-- Represents a school population with stratified sampling -/
structure SchoolPopulation where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  (senior_le_total : senior_students ≤ total_students)
  (sample_le_total : sample_size ≤ total_students)

/-- Calculates the number of senior students in a stratified sample -/
def seniors_in_sample (school : SchoolPopulation) : ℕ :=
  (school.senior_students * school.sample_size) / school.total_students

/-- Theorem stating that for the given school population, the number of seniors in the sample is 50 -/
theorem seniors_in_sample_is_fifty (school : SchoolPopulation)
  (h1 : school.total_students = 2000)
  (h2 : school.senior_students = 500)
  (h3 : school.sample_size = 200) :
  seniors_in_sample school = 50 := by
  sorry

end seniors_in_sample_is_fifty_l2025_202560


namespace bookshelf_problem_l2025_202521

theorem bookshelf_problem (shelf_length : ℕ) (total_books : ℕ) (thin_book_thickness : ℕ) (thick_book_thickness : ℕ) :
  shelf_length = 200 →
  total_books = 46 →
  thin_book_thickness = 3 →
  thick_book_thickness = 5 →
  ∃ (thin_books thick_books : ℕ),
    thin_books + thick_books = total_books ∧
    thin_books * thin_book_thickness + thick_books * thick_book_thickness = shelf_length ∧
    thin_books = 15 :=
by sorry

end bookshelf_problem_l2025_202521


namespace initial_forks_count_l2025_202581

theorem initial_forks_count (F : ℚ) : 
  (F + 2) + (F + 11) + (2 * F + 20) + (F / 2 + 2) = 62 ↔ F = 6 := by
  sorry

end initial_forks_count_l2025_202581


namespace worker_speed_ratio_l2025_202571

/-- Given two workers a and b, where b can complete a work in 60 days,
    and a and b together can complete the work in 12 days,
    prove that the ratio of a's speed to b's speed is 4:1 -/
theorem worker_speed_ratio (a b : ℝ) 
    (h₁ : b = 1 / 60)  -- b's speed (work per day)
    (h₂ : a + b = 1 / 12) -- combined speed of a and b (work per day)
    : a / b = 4 := by
  sorry

end worker_speed_ratio_l2025_202571


namespace union_complement_eq_set_l2025_202598

universe u

def U : Finset ℕ := {1,2,3,4,5}
def M : Finset ℕ := {1,4}
def N : Finset ℕ := {2,5}

theorem union_complement_eq_set : N ∪ (U \ M) = {2,3,5} := by sorry

end union_complement_eq_set_l2025_202598


namespace find_unknown_number_l2025_202506

theorem find_unknown_number (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 507, 2, 684, 42] → 
  average = 223 → 
  ∃ x : ℕ, x = 48 ∧ (List.sum known_numbers + x) / 6 = average := by
sorry


end find_unknown_number_l2025_202506


namespace train_length_l2025_202579

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), 
  abs (length - (speed * (5/18) * time)) < 0.01 :=
by sorry

end train_length_l2025_202579


namespace largest_number_from_digits_l2025_202550

def digits : List Nat := [6, 3]

theorem largest_number_from_digits :
  (63 : Nat) = digits.foldl (fun acc d => acc * 10 + d) 0 ∧
  ∀ (n : Nat), n ≠ 63 → n < 63 ∨ ¬(∃ (perm : List Nat), perm.Perm digits ∧ n = perm.foldl (fun acc d => acc * 10 + d) 0) :=
by sorry

end largest_number_from_digits_l2025_202550


namespace parabola_properties_l2025_202569

/-- A parabola with vertex at the origin, symmetric about the x-axis, passing through (-3, -6) -/
def parabola (x y : ℝ) : Prop := y^2 = -12*x

theorem parabola_properties :
  (parabola 0 0) ∧ 
  (∀ x y : ℝ, parabola x y → parabola x (-y)) ∧
  (parabola (-3) (-6)) := by
  sorry

end parabola_properties_l2025_202569


namespace min_words_for_certification_l2025_202500

theorem min_words_for_certification (total_words : ℕ) (min_score : ℚ) : 
  total_words = 800 → 
  min_score = 9/10 → 
  ∃ (words_to_learn : ℕ), 
    (words_to_learn : ℚ) / total_words ≥ min_score ∧ 
    ∀ (w : ℕ), (w : ℚ) / total_words ≥ min_score → w ≥ words_to_learn ∧
    words_to_learn = 720 := by
  sorry

end min_words_for_certification_l2025_202500


namespace min_c_value_sinusoidal_l2025_202577

/-- Given a sinusoidal function y = a * sin(b * x + c) where a > 0 and b > 0,
    if the function reaches its minimum at x = 0,
    then the smallest possible value of c is 3π/2. -/
theorem min_c_value_sinusoidal (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, a * Real.sin (b * x + c) ≥ a * Real.sin c) →
  c ≥ 3 * Real.pi / 2 ∧ 
  ∀ c' ≥ 3 * Real.pi / 2, c' < c → ∃ x : ℝ, a * Real.sin (b * x + c') < a * Real.sin c' :=
by sorry

end min_c_value_sinusoidal_l2025_202577


namespace wheat_yield_problem_l2025_202524

theorem wheat_yield_problem (plot1_area plot2_area : ℝ) 
  (h1 : plot2_area = plot1_area + 0.5)
  (h2 : 210 / plot1_area = 210 / plot2_area + 1) :
  210 / plot1_area = 21 ∧ 210 / plot2_area = 20 := by
sorry

end wheat_yield_problem_l2025_202524


namespace dans_marbles_l2025_202507

theorem dans_marbles (dans_marbles mary_marbles : ℕ) 
  (h1 : mary_marbles = 2 * dans_marbles)
  (h2 : mary_marbles = 10) : 
  dans_marbles = 5 := by
sorry

end dans_marbles_l2025_202507


namespace least_sum_of_bases_l2025_202510

theorem least_sum_of_bases (c d : ℕ+) (h : 3 * c.val + 8 = 8 * d.val + 3) :
  ∃ (c' d' : ℕ+), 3 * c'.val + 8 = 8 * d'.val + 3 ∧ c'.val + d'.val ≤ c.val + d.val ∧ c'.val + d'.val = 13 :=
sorry

end least_sum_of_bases_l2025_202510


namespace square_plus_reciprocal_square_l2025_202514

theorem square_plus_reciprocal_square (x : ℝ) (h : x^4 + 1/x^4 = 23) : 
  x^2 + 1/x^2 = 5 := by
sorry

end square_plus_reciprocal_square_l2025_202514


namespace total_chapter_difference_is_97_l2025_202565

-- Define the structure of a book
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

-- Define the series of books
def book_series : List Book := [
  { chapter1 := 48, chapter2 := 11, chapter3 := 24 },
  { chapter1 := 35, chapter2 := 18, chapter3 := 28 },
  { chapter1 := 62, chapter2 := 19, chapter3 := 12 }
]

-- Define the function to calculate the difference between first and second chapters
def chapter_difference (book : Book) : ℕ :=
  book.chapter1 - book.chapter2

-- Theorem statement
theorem total_chapter_difference_is_97 :
  (List.map chapter_difference book_series).sum = 97 := by
  sorry

end total_chapter_difference_is_97_l2025_202565


namespace hit_at_least_once_miss_both_times_mutually_exclusive_hit_at_least_once_miss_both_times_complementary_l2025_202504

-- Define the sample space
def Ω : Type := Unit

-- Define the event of hitting the target at least once
def hit_at_least_once : Set Ω := sorry

-- Define the event of missing the target both times
def miss_both_times : Set Ω := sorry

-- Theorem: hit_at_least_once and miss_both_times are mutually exclusive
theorem hit_at_least_once_miss_both_times_mutually_exclusive :
  hit_at_least_once ∩ miss_both_times = ∅ :=
sorry

-- Theorem: hit_at_least_once and miss_both_times are complementary
theorem hit_at_least_once_miss_both_times_complementary :
  hit_at_least_once ∪ miss_both_times = Set.univ :=
sorry

end hit_at_least_once_miss_both_times_mutually_exclusive_hit_at_least_once_miss_both_times_complementary_l2025_202504


namespace stratified_sampling_is_most_appropriate_l2025_202555

/-- Represents the different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | SystematicSampling
  | StratifiedSampling

/-- Represents a product category --/
structure ProductCategory where
  name : String
  count : Nat

/-- Represents a population of products --/
structure ProductPopulation where
  categories : List ProductCategory
  total : Nat

/-- Determines the most appropriate sampling method for a given product population and sample size --/
def mostAppropriateSamplingMethod (population : ProductPopulation) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- The given product population --/
def givenPopulation : ProductPopulation :=
  { categories := [
      { name := "First-class", count := 10 },
      { name := "Second-class", count := 25 },
      { name := "Defective", count := 5 }
    ],
    total := 40
  }

/-- The given sample size --/
def givenSampleSize : Nat := 8

/-- Theorem stating that Stratified Sampling is the most appropriate method for the given scenario --/
theorem stratified_sampling_is_most_appropriate :
  mostAppropriateSamplingMethod givenPopulation givenSampleSize = SamplingMethod.StratifiedSampling :=
  sorry

end stratified_sampling_is_most_appropriate_l2025_202555


namespace geometric_sequence_property_l2025_202503

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property :
  ∀ a : ℕ → ℝ,
  (is_geometric_sequence a → ∀ n : ℕ, a n ^ 2 = a (n - 1) * a (n + 1)) ∧
  (∃ a : ℕ → ℝ, (∀ n : ℕ, a n ^ 2 = a (n - 1) * a (n + 1)) ∧ ¬is_geometric_sequence a) :=
by sorry

end geometric_sequence_property_l2025_202503


namespace range_of_m_for_always_solvable_equation_l2025_202559

theorem range_of_m_for_always_solvable_equation :
  (∀ m : ℝ, ∃ x : ℝ, 4 * Real.cos x + Real.sin x ^ 2 + m - 4 = 0) →
  (∀ m : ℝ, m ∈ Set.Icc 0 8) :=
by sorry

end range_of_m_for_always_solvable_equation_l2025_202559


namespace third_row_is_10302_l2025_202561

/-- Represents a 3x5 grid of numbers -/
def Grid := Fin 3 → Fin 5 → Nat

/-- Check if a grid satisfies the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 3, ∃! j : Fin 5, g i j = 1) ∧
  (∀ i : Fin 3, ∃! j : Fin 5, g i j = 2) ∧
  (∀ i : Fin 3, ∃! j : Fin 5, g i j = 3) ∧
  (∀ j : Fin 5, ∃! i : Fin 3, g i j = 1) ∧
  (∀ j : Fin 5, ∃! i : Fin 3, g i j = 2) ∧
  (∀ j : Fin 5, ∃! i : Fin 3, g i j = 3)

/-- The sequence of numbers along the track -/
def track_sequence : Nat → Nat
| n => (n % 3) + 1

/-- The theorem stating that the third row of a valid grid is [1,0,3,0,2] -/
theorem third_row_is_10302 (g : Grid) (h : is_valid_grid g) :
  (g 2 0 = 1) ∧ (g 2 1 = 0) ∧ (g 2 2 = 3) ∧ (g 2 3 = 0) ∧ (g 2 4 = 2) :=
sorry

end third_row_is_10302_l2025_202561


namespace horner_method_evaluation_l2025_202595

def horner_polynomial (x : ℝ) : ℝ :=
  ((((4 * x + 3) * x + 4) * x + 2) * x + 5) * x - 7 * x + 9

theorem horner_method_evaluation :
  horner_polynomial 4 = 20669 :=
by sorry

end horner_method_evaluation_l2025_202595


namespace tile_border_ratio_l2025_202536

theorem tile_border_ratio (n s d : ℝ) (h1 : n = 30) 
  (h2 : (n^2 * s^2) / ((n*s + 2*n*d)^2) = 0.81) : d/s = 1/18 := by
  sorry

end tile_border_ratio_l2025_202536


namespace max_integers_less_than_negative_five_l2025_202553

theorem max_integers_less_than_negative_five (a b c d e : ℤ) 
  (sum_condition : a + b + c + d + e = 20) : 
  (∃ (count : ℕ), count ≤ 4 ∧ 
    (count = (if a < -5 then 1 else 0) + 
             (if b < -5 then 1 else 0) + 
             (if c < -5 then 1 else 0) + 
             (if d < -5 then 1 else 0) + 
             (if e < -5 then 1 else 0)) ∧
    ∀ (other_count : ℕ), 
      (other_count = (if a < -5 then 1 else 0) + 
                     (if b < -5 then 1 else 0) + 
                     (if c < -5 then 1 else 0) + 
                     (if d < -5 then 1 else 0) + 
                     (if e < -5 then 1 else 0)) → 
      other_count ≤ count) ∧
  ¬(∃ (impossible_count : ℕ), impossible_count = 5 ∧
    impossible_count = (if a < -5 then 1 else 0) + 
                       (if b < -5 then 1 else 0) + 
                       (if c < -5 then 1 else 0) + 
                       (if d < -5 then 1 else 0) + 
                       (if e < -5 then 1 else 0)) :=
by sorry

end max_integers_less_than_negative_five_l2025_202553


namespace union_and_subset_l2025_202543

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | 3/2 < x ∧ x < 4}
def P (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

theorem union_and_subset :
  (A ∪ B = {x | 1 < x ∧ x < 4}) ∧
  (∀ a : ℝ, P a ⊆ A ∪ B ↔ 1 ≤ a ∧ a ≤ 2) :=
by sorry

end union_and_subset_l2025_202543


namespace no_prime_sum_53_l2025_202586

theorem no_prime_sum_53 : ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 53 := by
  sorry

end no_prime_sum_53_l2025_202586


namespace rectangle_area_with_three_squares_l2025_202558

/-- Given three non-overlapping squares where one has an area of 4 square inches,
    and another has double the side length of the first two,
    the total area of the rectangle containing all three squares is 24 square inches. -/
theorem rectangle_area_with_three_squares (s₁ s₂ s₃ : Real) : 
  s₁ * s₁ = 4 →  -- Area of the first square (shaded) is 4
  s₂ = s₁ →      -- Second square has the same side length as the first
  s₃ = 2 * s₁ →  -- Third square has double the side length of the first two
  s₁ * s₁ + s₂ * s₂ + s₃ * s₃ = 24 := by
  sorry


end rectangle_area_with_three_squares_l2025_202558


namespace cow_count_is_twelve_l2025_202540

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 12 under the given conditions -/
theorem cow_count_is_twelve (count : AnimalCount) 
    (h : totalLegs count = 2 * totalHeads count + 24) : 
    count.cows = 12 := by
  sorry


end cow_count_is_twelve_l2025_202540


namespace prob_two_red_cards_l2025_202515

/-- Probability of drawing two red cards in succession from a special deck -/
theorem prob_two_red_cards (total_cards : Nat) (red_cards : Nat) 
  (h1 : total_cards = 60)
  (h2 : red_cards = 36) : 
  (red_cards * (red_cards - 1)) / (total_cards * (total_cards - 1)) = 21 / 59 := by
  sorry

end prob_two_red_cards_l2025_202515


namespace right_triangle_third_side_l2025_202575

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a = 5 → b = 12 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 13 ∨ c = Real.sqrt 119 := by
sorry

end right_triangle_third_side_l2025_202575


namespace triangle_area_with_perimeter_12_l2025_202545

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area_with_perimeter_12 (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → c + a > b → 
  (a : ℝ) * b * (12 : ℝ) / 4 = 6 := by sorry

end triangle_area_with_perimeter_12_l2025_202545


namespace range_of_a_l2025_202548

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (∀ x, p x ↔ (x ≤ 1/2 ∨ x ≥ 1)) →
  (∀ x, q x ↔ (x - a) * (x - a - 1) ≤ 0) →
  (∀ x, ¬(q x) → p x) →
  (∃ x, ¬(q x) ∧ ¬(p x)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l2025_202548


namespace number_of_green_balls_l2025_202526

/-- Given a bag with blue and green balls, prove the number of green balls. -/
theorem number_of_green_balls
  (blue_balls : ℕ)
  (total_balls : ℕ)
  (h1 : blue_balls = 10)
  (h2 : (blue_balls : ℚ) / total_balls = 1 / 5)
  : total_balls - blue_balls = 40 := by
  sorry

end number_of_green_balls_l2025_202526


namespace initial_condition_recursive_relation_diamonds_in_tenth_figure_l2025_202532

/-- The number of diamonds in the n-th figure of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

/-- The sequence starts with 4 diamonds for n = 1 -/
theorem initial_condition : num_diamonds 1 = 4 := by sorry

/-- The recursive relation for n ≥ 2 -/
theorem recursive_relation (n : ℕ) (h : n ≥ 2) :
  num_diamonds n = num_diamonds (n-1) + 12 * (n-1) := by sorry

/-- The main theorem: The number of diamonds in the 10th figure is 112 -/
theorem diamonds_in_tenth_figure : num_diamonds 10 = 112 := by sorry

end initial_condition_recursive_relation_diamonds_in_tenth_figure_l2025_202532


namespace tangent_line_at_one_a_upper_bound_l2025_202588

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * x + a / x

/-- Theorem for part (1) of the problem -/
theorem tangent_line_at_one (a : ℝ) :
  a = 2 → ∃ A B C : ℝ, A = 3 ∧ B = 2 ∧ C = -6 ∧
  ∀ x y : ℝ, y = f a x → (x = 1 → A * x + B * y + C = 0) :=
sorry

/-- Theorem for part (2) of the problem -/
theorem a_upper_bound :
  (∀ x : ℝ, x > 1 → f a x < 0) → a ≤ 1/2 :=
sorry

end tangent_line_at_one_a_upper_bound_l2025_202588


namespace base9_perfect_square_last_digit_l2025_202513

/-- Represents a number in base 9 of the form ab5d -/
structure Base9Number where
  a : ℕ
  b : ℕ
  d : ℕ
  a_nonzero : a ≠ 0
  b_range : b < 9
  d_range : d < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 45 + n.d

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base9_perfect_square_last_digit
  (n : Base9Number)
  (h : isPerfectSquare (toDecimal n)) :
  n.d = 0 := by
  sorry

#check base9_perfect_square_last_digit

end base9_perfect_square_last_digit_l2025_202513


namespace meaningful_fraction_l2025_202582

/-- For a fraction 3x/(5-x) to be meaningful, x must not equal 5 -/
theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 * x / (5 - x)) ↔ x ≠ 5 :=
by sorry

end meaningful_fraction_l2025_202582


namespace star_value_of_a_l2025_202501

-- Define the star operation
def star (a b : ℝ) : ℝ := 3 * a - b^3

-- Theorem statement
theorem star_value_of_a :
  ∀ a : ℝ, star a 3 = 18 → a = 15 := by
  sorry

end star_value_of_a_l2025_202501


namespace smallest_student_count_l2025_202531

theorem smallest_student_count (sophomore freshman junior : ℕ) : 
  sophomore * 4 = freshman * 7 →
  junior * 7 = sophomore * 6 →
  sophomore > 0 →
  freshman > 0 →
  junior > 0 →
  ∀ (s f j : ℕ), 
    s * 4 = f * 7 →
    j * 7 = s * 6 →
    s > 0 → f > 0 → j > 0 →
    s + f + j ≥ sophomore + freshman + junior :=
by
  sorry

#eval 7 + 4 + 6 -- Expected output: 17

end smallest_student_count_l2025_202531


namespace cousin_reading_time_l2025_202539

/-- Given reading speeds and book lengths, calculate cousin's reading time -/
theorem cousin_reading_time
  (my_speed : ℝ)
  (my_time : ℝ)
  (my_book_length : ℝ)
  (cousin_speed_ratio : ℝ)
  (cousin_book_length_ratio : ℝ)
  (h1 : my_time = 180) -- 3 hours in minutes
  (h2 : cousin_speed_ratio = 5)
  (h3 : cousin_book_length_ratio = 1.5)
  : (cousin_book_length_ratio * my_book_length) / (cousin_speed_ratio * my_speed) = 54 := by
  sorry

end cousin_reading_time_l2025_202539


namespace ps_length_l2025_202523

/-- Represents a quadrilateral PQRS with specific properties -/
structure Quadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  sinR : ℝ
  cosQ : ℝ
  R_obtuse : Bool

/-- The theorem stating the length of PS in the given quadrilateral -/
theorem ps_length (quad : Quadrilateral)
  (h1 : quad.PQ = 6)
  (h2 : quad.QR = 7)
  (h3 : quad.RS = 25)
  (h4 : quad.sinR = 4/5)
  (h5 : quad.cosQ = -4/5)
  (h6 : quad.R_obtuse = true) :
  ∃ (PS : ℝ), PS^2 = 794 :=
sorry

end ps_length_l2025_202523


namespace sunset_time_calculation_l2025_202573

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of a time period in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given the sunrise time and daylight duration -/
def calculate_sunset (sunrise : Time) (daylight : Duration) : Time :=
  sorry

theorem sunset_time_calculation :
  let sunrise : Time := ⟨6, 22⟩
  let daylight : Duration := ⟨11, 36⟩
  let calculated_sunset : Time := calculate_sunset sunrise daylight
  calculated_sunset = ⟨18, 58⟩ := by sorry

end sunset_time_calculation_l2025_202573


namespace root_sum_fraction_l2025_202562

/-- Given a, b, and c are the roots of x^3 - 8x^2 + 11x - 3 = 0,
    prove that (a/(bc - 1)) + (b/(ac - 1)) + (c/(ab - 1)) = 13 -/
theorem root_sum_fraction (a b c : ℝ) 
  (h1 : a^3 - 8*a^2 + 11*a - 3 = 0)
  (h2 : b^3 - 8*b^2 + 11*b - 3 = 0)
  (h3 : c^3 - 8*c^2 + 11*c - 3 = 0) :
  a / (b * c - 1) + b / (a * c - 1) + c / (a * b - 1) = 13 := by
  sorry

end root_sum_fraction_l2025_202562


namespace taxi_distance_range_l2025_202535

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  baseFare : ℝ  -- Base fare in yuan
  baseDistance : ℝ  -- Distance covered by base fare in km
  additionalFarePerUnit : ℝ  -- Additional fare per unit distance in yuan
  additionalDistanceUnit : ℝ  -- Unit of additional distance in km
  fuelSurcharge : ℝ  -- Fuel surcharge in yuan
  totalFare : ℝ  -- Total fare paid in yuan

/-- Theorem stating the range of possible distances for the given fare structure and total fare -/
theorem taxi_distance_range (ride : TaxiRide)
  (h1 : ride.baseFare = 6)
  (h2 : ride.baseDistance = 2)
  (h3 : ride.additionalFarePerUnit = 1)
  (h4 : ride.additionalDistanceUnit = 0.5)
  (h5 : ride.fuelSurcharge = 1)
  (h6 : ride.totalFare = 9) :
  ∃ x : ℝ, 2.5 < x ∧ x ≤ 3 ∧
    ride.totalFare = ride.baseFare + ((x - ride.baseDistance) / ride.additionalDistanceUnit) * ride.additionalFarePerUnit + ride.fuelSurcharge :=
by sorry


end taxi_distance_range_l2025_202535


namespace steves_day_assignments_l2025_202580

/-- Given that Steve's day is divided into sleeping, school, family time, and assignments,
    prove that the fraction of the day spent on assignments is 1/12. -/
theorem steves_day_assignments (total_hours : ℝ) (sleep_fraction : ℝ) (school_fraction : ℝ) 
    (family_hours : ℝ) (assignment_fraction : ℝ) : 
    total_hours = 24 ∧ 
    sleep_fraction = 1/3 ∧ 
    school_fraction = 1/6 ∧ 
    family_hours = 10 ∧ 
    sleep_fraction + school_fraction + (family_hours / total_hours) + assignment_fraction = 1 →
    assignment_fraction = 1/12 := by
  sorry

end steves_day_assignments_l2025_202580


namespace fuel_consumption_l2025_202512

/-- Represents the fuel consumption of a heating plant -/
structure HeatingPlant where
  hours : ℝ
  fuel : ℝ
  rate : ℝ
  hRate : rate = fuel / hours

/-- Given a heating plant that consumes 7 liters of fuel in 21 hours,
    prove that it consumes 30 liters of fuel in 90 hours -/
theorem fuel_consumption (plant : HeatingPlant) 
  (h1 : plant.hours = 21) 
  (h2 : plant.fuel = 7) : 
  plant.rate * 90 = 30 := by
  sorry

end fuel_consumption_l2025_202512


namespace min_value_product_l2025_202570

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (3 * a + b) * (a + 3 * c) * (2 * b * c + 4) ≥ 384 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (3 * a₀ + b₀) * (a₀ + 3 * c₀) * (2 * b₀ * c₀ + 4) = 384 :=
by sorry

end min_value_product_l2025_202570


namespace brent_kitkat_count_l2025_202533

/-- The number of Kit-Kat bars Brent received -/
def kitKat : ℕ := sorry

/-- The number of Hershey kisses Brent received -/
def hersheyKisses : ℕ := 3 * kitKat

/-- The number of Nerds boxes Brent received -/
def nerds : ℕ := 8

/-- The initial number of lollipops Brent received -/
def initialLollipops : ℕ := 11

/-- The number of Baby Ruths Brent had -/
def babyRuths : ℕ := 10

/-- The number of Reese Peanut butter cups Brent had -/
def reeseCups : ℕ := babyRuths / 2

/-- The number of lollipops Brent gave to his sister -/
def givenLollipops : ℕ := 5

/-- The total number of candy pieces Brent had left -/
def totalCandyLeft : ℕ := 49

theorem brent_kitkat_count : kitKat = 5 := by
  sorry

end brent_kitkat_count_l2025_202533


namespace bipartite_perfect_matching_l2025_202587

/-- A bipartite graph with 20 vertices in each part and degree 2 for all vertices -/
structure BipartiteGraph :=
  (U V : Finset ℕ)
  (E : Finset (ℕ × ℕ))
  (hU : U.card = 20)
  (hV : V.card = 20)
  (hE : ∀ u ∈ U, (E.filter (λ e => e.1 = u)).card = 2)
  (hE' : ∀ v ∈ V, (E.filter (λ e => e.2 = v)).card = 2)

/-- A perfect matching in a bipartite graph -/
def PerfectMatching (G : BipartiteGraph) :=
  ∃ M : Finset (ℕ × ℕ), M ⊆ G.E ∧ 
    (∀ u ∈ G.U, (M.filter (λ e => e.1 = u)).card = 1) ∧
    (∀ v ∈ G.V, (M.filter (λ e => e.2 = v)).card = 1)

/-- Theorem: A bipartite graph with 20 vertices in each part and degree 2 for all vertices has a perfect matching -/
theorem bipartite_perfect_matching (G : BipartiteGraph) : PerfectMatching G := by
  sorry

end bipartite_perfect_matching_l2025_202587


namespace calculation_results_l2025_202574

theorem calculation_results : 
  ((-12) - (-20) + (-8) - 15 = -15) ∧
  (-3^2 + (2/3 - 1/2 + 5/8) * (-24) = -28) ∧
  (-1^2023 + 3 * (-2)^2 - (-6) / (-1/3)^2 = 65) := by
sorry

end calculation_results_l2025_202574


namespace round_trip_time_l2025_202518

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream's speed, and the distance to travel. -/
theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 6 →
  distance = 210 →
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 84 :=
by sorry

end round_trip_time_l2025_202518


namespace carter_ate_twelve_green_mms_l2025_202578

/-- Represents the number of M&Ms of each color in the jar -/
structure JarContents where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- The initial state of the jar -/
def initial_jar : JarContents := { green := 20, red := 20, yellow := 0 }

/-- The state of the jar after all actions -/
def final_jar (green_eaten : ℕ) : JarContents :=
  { green := initial_jar.green - green_eaten,
    red := initial_jar.red / 2,
    yellow := 14 }

/-- The total number of M&Ms in the jar -/
def total_mms (jar : JarContents) : ℕ := jar.green + jar.red + jar.yellow

/-- The probability of picking a green M&M from the jar -/
def green_probability (jar : JarContents) : ℚ :=
  jar.green / (total_mms jar : ℚ)

theorem carter_ate_twelve_green_mms :
  ∃ (green_eaten : ℕ),
    green_eaten ≤ initial_jar.green ∧
    green_probability (final_jar green_eaten) = 1/4 ∧
    green_eaten = 12 := by sorry

end carter_ate_twelve_green_mms_l2025_202578


namespace max_ab_value_l2025_202556

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (2 : ℝ) / (2 * a - 3) * (b / 2) = 1) : a * b ≤ 9 / 8 := by
  sorry

end max_ab_value_l2025_202556


namespace geometric_series_example_l2025_202505

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  (a - a * r^n) / (1 - r)

theorem geometric_series_example : 
  let a : ℚ := 1/5
  let r : ℚ := -1/5
  let n : ℕ := 6
  geometric_series_sum a r n = 1562/9375 := by sorry

end geometric_series_example_l2025_202505


namespace no_solution_mod_seven_l2025_202585

theorem no_solution_mod_seven (m : ℤ) : 
  (0 ≤ m ∧ m ≤ 6) →
  (m = 4 ↔ ∀ x y : ℤ, (3 * x^2 - 10 * x * y - 8 * y^2) % 7 ≠ m % 7) :=
by sorry

end no_solution_mod_seven_l2025_202585


namespace bela_has_winning_strategy_l2025_202516

/-- Represents a stick with a certain length -/
structure Stick :=
  (length : ℕ)

/-- Represents the game state -/
structure GameState :=
  (sticks : List Stick)

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- Represents a player's strategy -/
def Strategy := GameState → Stick

/-- Represents the initial game state with 99 sticks -/
def initialGameState : GameState :=
  { sticks := List.range 99 |>.map (fun n => ⟨n + 1⟩) }

/-- Represents Béla's winning strategy -/
noncomputable def belasStrategy : Strategy :=
  sorry

/-- Theorem stating that Béla has a winning strategy -/
theorem bela_has_winning_strategy :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.sticks.length = 3 →
      ¬(∃ (s1 s2 s3 : Stick), s1 ∈ game.sticks ∧ s2 ∈ game.sticks ∧ s3 ∈ game.sticks ∧ canFormTriangle s1 s2 s3) :=
by
  sorry

end bela_has_winning_strategy_l2025_202516


namespace max_profit_is_2180_l2025_202519

/-- Represents the production and profit constraints for products A and B --/
structure ProductionConstraints where
  steel_A : ℝ
  nonferrous_A : ℝ
  steel_B : ℝ
  nonferrous_B : ℝ
  profit_A : ℝ
  profit_B : ℝ
  steel_reserve : ℝ
  nonferrous_reserve : ℝ

/-- Represents the production quantities of products A and B --/
structure ProductionQuantities where
  qty_A : ℝ
  qty_B : ℝ

/-- Calculates the total profit given production quantities and constraints --/
def calculateProfit (q : ProductionQuantities) (c : ProductionConstraints) : ℝ :=
  q.qty_A * c.profit_A + q.qty_B * c.profit_B

/-- Checks if the production quantities satisfy the resource constraints --/
def isValidProduction (q : ProductionQuantities) (c : ProductionConstraints) : Prop :=
  q.qty_A * c.steel_A + q.qty_B * c.steel_B ≤ c.steel_reserve ∧
  q.qty_A * c.nonferrous_A + q.qty_B * c.nonferrous_B ≤ c.nonferrous_reserve ∧
  q.qty_A ≥ 0 ∧ q.qty_B ≥ 0

/-- Theorem stating that the maximum profit under given constraints is 2180 --/
theorem max_profit_is_2180 (c : ProductionConstraints)
    (h1 : c.steel_A = 10 ∧ c.nonferrous_A = 23)
    (h2 : c.steel_B = 70 ∧ c.nonferrous_B = 40)
    (h3 : c.profit_A = 80 ∧ c.profit_B = 100)
    (h4 : c.steel_reserve = 700 ∧ c.nonferrous_reserve = 642) :
    ∃ (q : ProductionQuantities),
      isValidProduction q c ∧
      calculateProfit q c = 2180 ∧
      ∀ (q' : ProductionQuantities),
        isValidProduction q' c → calculateProfit q' c ≤ 2180 := by
  sorry

end max_profit_is_2180_l2025_202519


namespace largest_gcd_of_sum_1005_l2025_202502

theorem largest_gcd_of_sum_1005 :
  ∃ (a b : ℕ+), a + b = 1005 ∧
  ∀ (c d : ℕ+), c + d = 1005 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 335 :=
sorry

end largest_gcd_of_sum_1005_l2025_202502


namespace revenue_calculation_l2025_202568

def calculateDailyRevenue (booksSold : ℕ) (price : ℚ) (discount : ℚ) : ℚ :=
  booksSold * (price * (1 - discount))

def totalRevenue (initialStock : ℕ) (price : ℚ) 
  (mondaySales : ℕ) (mondayDiscount : ℚ)
  (tuesdaySales : ℕ) (tuesdayDiscount : ℚ)
  (wednesdaySales : ℕ) (wednesdayDiscount : ℚ)
  (thursdaySales : ℕ) (thursdayDiscount : ℚ)
  (fridaySales : ℕ) (fridayDiscount : ℚ) : ℚ :=
  calculateDailyRevenue mondaySales price mondayDiscount +
  calculateDailyRevenue tuesdaySales price tuesdayDiscount +
  calculateDailyRevenue wednesdaySales price wednesdayDiscount +
  calculateDailyRevenue thursdaySales price thursdayDiscount +
  calculateDailyRevenue fridaySales price fridayDiscount

theorem revenue_calculation :
  totalRevenue 800 25 
    60 (10 / 100)
    10 0
    20 (5 / 100)
    44 (15 / 100)
    66 (20 / 100) = 4330 := by
  sorry

end revenue_calculation_l2025_202568


namespace train_distance_problem_l2025_202528

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 60) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 540 :=
by sorry

end train_distance_problem_l2025_202528


namespace perpendicular_vectors_m_value_l2025_202542

/-- Two-dimensional vector type -/
def Vec2 := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vec2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v w : Vec2) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_m_value (m : ℝ) :
  let a : Vec2 := (1, m)
  let b : Vec2 := (2, -m)
  perpendicular a b → m = -Real.sqrt 2 ∨ m = Real.sqrt 2 := by
  sorry

end perpendicular_vectors_m_value_l2025_202542


namespace alcohol_mixture_proof_l2025_202522

theorem alcohol_mixture_proof (x y z : Real) : 
  x = 112.5 ∧ 
  y = 112.5 ∧ 
  z = 225 ∧
  x + y + z = 450 ∧
  0.10 * x + 0.30 * y + 0.50 * z = 0.35 * 450 :=
by sorry

end alcohol_mixture_proof_l2025_202522


namespace ellipse_triangle_perimeter_l2025_202534

/-- Perimeter of triangle ABF₂ in an ellipse with given parameters -/
theorem ellipse_triangle_perimeter (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b = 4)
    (h4 : c / a = 3 / 5) (h5 : a^2 = b^2 + c^2) :
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  ∀ A B : ℝ × ℝ, A ∈ ellipse → B ∈ ellipse →
    ∃ t : ℝ, A = F₁ + t • (1, 0) ∧ B = F₁ + t • (1, 0) →
      dist A B + dist A F₂ + dist B F₂ = 20 :=
by
  sorry

end ellipse_triangle_perimeter_l2025_202534


namespace reynalds_volleyballs_l2025_202530

/-- The number of volleyballs in Reynald's purchase --/
def num_volleyballs (total : ℕ) (soccer : ℕ) : ℕ :=
  total - (soccer + (soccer + 5) + (2 * soccer) + (soccer + 10))

/-- Theorem stating the number of volleyballs Reynald bought --/
theorem reynalds_volleyballs : num_volleyballs 145 20 = 30 := by
  sorry

#eval num_volleyballs 145 20

end reynalds_volleyballs_l2025_202530


namespace f_properties_l2025_202547

def f (x : ℝ) : ℝ := |x| - 1

theorem f_properties : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
sorry

end f_properties_l2025_202547


namespace square_root_meaningful_l2025_202509

theorem square_root_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by
  sorry

end square_root_meaningful_l2025_202509


namespace absolute_value_inequality_l2025_202592

theorem absolute_value_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b := by
  sorry

end absolute_value_inequality_l2025_202592


namespace f_increasing_l2025_202529

-- Define the function f(x) = x^3 - 1
def f (x : ℝ) : ℝ := x^3 - 1

-- Theorem stating that f is increasing over its domain
theorem f_increasing : StrictMono f := by sorry

end f_increasing_l2025_202529


namespace min_value_x2_plus_y2_l2025_202590

theorem min_value_x2_plus_y2 (x y : ℝ) (h : x^2 + 2 * Real.sqrt 3 * x * y - y^2 = 1) :
  ∃ (m : ℝ), m = (1 : ℝ) / 2 ∧ ∀ (a b : ℝ), a^2 + 2 * Real.sqrt 3 * a * b - b^2 = 1 → a^2 + b^2 ≥ m :=
by sorry

end min_value_x2_plus_y2_l2025_202590


namespace analysis_method_seeks_sufficient_condition_l2025_202564

/-- The type of conditions in logical reasoning -/
inductive ConditionType
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- Represents the analysis method for proving inequalities -/
structure AnalysisMethod where
  seekCauseFromResult : ConditionType

/-- The theorem stating that "seeking the cause from the result" in the analysis method
    aims to find a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (m : AnalysisMethod), m.seekCauseFromResult = ConditionType.Sufficient := by
  sorry

end analysis_method_seeks_sufficient_condition_l2025_202564


namespace no_valid_arrangement_l2025_202599

/-- Represents a seating arrangement of delegates around a circular table. -/
def SeatingArrangement := Fin 54 → Fin 27

/-- Checks if two positions in the circular table are separated by exactly 9 other positions. -/
def isSeparatedByNine (a b : Fin 54) : Prop :=
  (b - a) % 54 = 10 ∨ (a - b) % 54 = 10

/-- Represents a valid seating arrangement where each country's delegates are separated by 9 others. -/
def isValidArrangement (arrangement : SeatingArrangement) : Prop :=
  ∀ country : Fin 27, ∃ a b : Fin 54,
    a ≠ b ∧
    arrangement a = country ∧
    arrangement b = country ∧
    isSeparatedByNine a b

/-- Theorem stating that a valid seating arrangement is impossible. -/
theorem no_valid_arrangement : ¬∃ arrangement : SeatingArrangement, isValidArrangement arrangement := by
  sorry

end no_valid_arrangement_l2025_202599


namespace raisin_nut_mixture_cost_fraction_l2025_202552

theorem raisin_nut_mixture_cost_fraction :
  ∀ (R : ℚ),
  R > 0 →
  let raisin_pounds : ℚ := 3
  let nut_pounds : ℚ := 4
  let raisin_cost_per_pound : ℚ := R
  let nut_cost_per_pound : ℚ := 4 * R
  let total_raisin_cost : ℚ := raisin_pounds * raisin_cost_per_pound
  let total_nut_cost : ℚ := nut_pounds * nut_cost_per_pound
  let total_mixture_cost : ℚ := total_raisin_cost + total_nut_cost
  (total_raisin_cost / total_mixture_cost) = 3 / 19 :=
by
  sorry

end raisin_nut_mixture_cost_fraction_l2025_202552


namespace possible_m_values_l2025_202563

def A : Set ℝ := {x | x^2 - 9*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values :
  {m : ℝ | A ∪ B m = A} = {0, 1, -(1/10)} := by sorry

end possible_m_values_l2025_202563


namespace cos_330_degrees_l2025_202544

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l2025_202544


namespace farmer_water_capacity_l2025_202538

/-- Calculates the total water capacity for a farmer's trucks -/
theorem farmer_water_capacity 
  (num_trucks : ℕ) 
  (tanks_per_truck : ℕ) 
  (tank_capacity : ℕ) 
  (h1 : num_trucks = 5)
  (h2 : tanks_per_truck = 4)
  (h3 : tank_capacity = 200) : 
  num_trucks * tanks_per_truck * tank_capacity = 4000 := by
  sorry

#check farmer_water_capacity

end farmer_water_capacity_l2025_202538


namespace count_special_numbers_in_range_l2025_202583

def count_multiples_of_three (n : ℕ) : ℕ :=
  (n / 3 : ℕ)

def count_special_numbers (n : ℕ) : ℕ :=
  ((n + 2) / 12 : ℕ)

theorem count_special_numbers_in_range : 
  count_multiples_of_three 2015 = 671 ∧ 
  count_special_numbers 2015 = 167 := by
  sorry

end count_special_numbers_in_range_l2025_202583


namespace germination_probability_approx_095_l2025_202549

/-- Represents a batch of seeds with its germination data -/
structure SeedBatch where
  seeds : ℕ
  germinations : ℕ

/-- Calculates the germination rate for a batch of seeds -/
def germinationRate (batch : SeedBatch) : ℚ :=
  batch.germinations / batch.seeds

/-- The data from the experiment -/
def seedData : List SeedBatch := [
  ⟨100, 96⟩,
  ⟨300, 284⟩,
  ⟨400, 380⟩,
  ⟨600, 571⟩,
  ⟨1000, 948⟩,
  ⟨2000, 1902⟩,
  ⟨3000, 2848⟩
]

/-- The average germination rate from the experiment -/
def averageGerminationRate : ℚ :=
  (seedData.map germinationRate).sum / seedData.length

/-- Theorem stating that the probability of germination is approximately 0.95 -/
theorem germination_probability_approx_095 :
  ∃ ε > 0, abs (averageGerminationRate - 95/100) < ε ∧ ε < 1/100 :=
sorry

end germination_probability_approx_095_l2025_202549


namespace rachel_age_l2025_202589

/-- Given that Rachel is 4 years older than Leah and the sum of their ages is 34,
    prove that Rachel is 19 years old. -/
theorem rachel_age (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4)
  (h2 : rachel_age + leah_age = 34) : 
  rachel_age = 19 := by
  sorry

end rachel_age_l2025_202589


namespace no_integer_pairs_sqrt_l2025_202537

theorem no_integer_pairs_sqrt : 
  ¬∃ (x y : ℕ), 0 < x ∧ x < y ∧ Real.sqrt 2756 = Real.sqrt x + Real.sqrt y := by
  sorry

end no_integer_pairs_sqrt_l2025_202537


namespace chloe_second_level_treasures_l2025_202541

/-- Represents the game scenario for Chloe's treasure hunt --/
structure GameScenario where
  points_per_treasure : ℕ
  treasures_first_level : ℕ
  total_score : ℕ

/-- Calculates the number of treasures found on the second level --/
def treasures_second_level (game : GameScenario) : ℕ :=
  (game.total_score - game.points_per_treasure * game.treasures_first_level) / game.points_per_treasure

/-- Theorem stating that Chloe found 3 treasures on the second level --/
theorem chloe_second_level_treasures :
  ∃ (game : GameScenario),
    game.points_per_treasure = 9 ∧
    game.treasures_first_level = 6 ∧
    game.total_score = 81 ∧
    treasures_second_level game = 3 := by
  sorry

end chloe_second_level_treasures_l2025_202541


namespace battle_station_staffing_l2025_202527

/-- The number of job openings in the battle station -/
def num_jobs : ℕ := 5

/-- The total number of applicants -/
def total_applicants : ℕ := 30

/-- The number of suitable applicants -/
def suitable_applicants : ℕ := total_applicants - (total_applicants / 3)

/-- The number of candidates qualified for the Radio Specialist role -/
def radio_qualified : ℕ := 5

/-- The number of ways to staff the battle station -/
def staffing_ways : ℕ := radio_qualified * (suitable_applicants - 1) * (suitable_applicants - 2) * (suitable_applicants - 3) * (suitable_applicants - 4)

theorem battle_station_staffing :
  staffing_ways = 292320 :=
sorry

end battle_station_staffing_l2025_202527


namespace jane_visited_six_more_l2025_202517

/-- The number of rehabilitation centers visited by each person --/
structure RehabCenters where
  lisa : ℕ
  jude : ℕ
  han : ℕ
  jane : ℕ

/-- The conditions of the problem --/
def problem_conditions (rc : RehabCenters) : Prop :=
  rc.lisa = 6 ∧
  rc.jude = rc.lisa / 2 ∧
  rc.han = 2 * rc.jude - 2 ∧
  rc.jane > 2 * rc.han ∧
  rc.lisa + rc.jude + rc.han + rc.jane = 27

/-- The theorem to be proved --/
theorem jane_visited_six_more (rc : RehabCenters) : 
  problem_conditions rc → rc.jane = 2 * rc.han + 6 := by
  sorry


end jane_visited_six_more_l2025_202517


namespace k_value_theorem_l2025_202566

theorem k_value_theorem (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 = (x + 1)^2) → k = 2 := by
sorry

end k_value_theorem_l2025_202566


namespace jane_sunflower_seeds_l2025_202597

/-- Given that Jane has 9 cans and places 6 seeds in each can,
    prove that the total number of sunflower seeds is 54. -/
theorem jane_sunflower_seeds (num_cans : ℕ) (seeds_per_can : ℕ) 
    (h1 : num_cans = 9) 
    (h2 : seeds_per_can = 6) : 
  num_cans * seeds_per_can = 54 := by
  sorry

end jane_sunflower_seeds_l2025_202597


namespace min_value_sqrt_fraction_min_value_achieved_l2025_202576

theorem min_value_sqrt_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4*a^2 + b^2)).sqrt) / (a * b) ≥ ((1 + 2^(2/3)) * (4 + 2^(2/3)) / 2^(2/3)).sqrt :=
sorry

theorem min_value_achieved (a : ℝ) (ha : a > 0) :
  let b := a * (2^(1/3))
  ((a^2 + b^2) * (4*a^2 + b^2)).sqrt / (a * b) = ((1 + 2^(2/3)) * (4 + 2^(2/3)) / 2^(2/3)).sqrt :=
sorry

end min_value_sqrt_fraction_min_value_achieved_l2025_202576


namespace f_sum_negative_l2025_202572

/-- A function f satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- The main theorem -/
theorem f_sum_negative
  (f : ℝ → ℝ)
  (h_f : f_properties f)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ < 0)
  (h_prod : x₁ * x₂ < 0) :
  f x₁ + f x₂ < 0 :=
sorry

end f_sum_negative_l2025_202572


namespace small_sphere_diameter_small_sphere_diameter_value_l2025_202584

/-- The diameter of small spheres fitting in the corners of a cube with a larger sphere inside --/
theorem small_sphere_diameter 
  (cube_side : ℝ) 
  (large_sphere_diameter : ℝ) 
  (h_cube : cube_side = 32) 
  (h_large_sphere : large_sphere_diameter = 30) : 
  ℝ :=
let large_sphere_radius := large_sphere_diameter / 2
let small_sphere_radius := (cube_side * Real.sqrt 3 / 2 - large_sphere_radius) / (Real.sqrt 3 + 1)
2 * small_sphere_radius

theorem small_sphere_diameter_value :
  small_sphere_diameter 32 30 rfl rfl = 63 - 31 * Real.sqrt 3 :=
by sorry

end small_sphere_diameter_small_sphere_diameter_value_l2025_202584


namespace class_average_proof_l2025_202591

theorem class_average_proof (x : ℝ) : 
  (0.25 * x + 0.5 * 65 + 0.25 * 90 = 75) → x = 80 := by
  sorry

end class_average_proof_l2025_202591


namespace chess_club_members_count_l2025_202554

theorem chess_club_members_count : ∃! n : ℕ, 
  300 ≤ n ∧ n ≤ 400 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 := by
  sorry

end chess_club_members_count_l2025_202554


namespace m_equals_five_l2025_202593

theorem m_equals_five (m : ℝ) (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 := by
  sorry

end m_equals_five_l2025_202593


namespace red_shirt_pairs_red_shirt_pairs_correct_l2025_202567

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) : ℕ :=
  by
  have h1 : green_students = 67 := by sorry
  have h2 : red_students = 89 := by sorry
  have h3 : total_students = 156 := by sorry
  have h4 : total_pairs = 78 := by sorry
  have h5 : green_green_pairs = 25 := by sorry
  have h6 : total_students = green_students + red_students := by sorry
  have h7 : green_students + red_students = total_pairs * 2 := by sorry

  -- The number of pairs where both students are wearing red shirts
  sorry

theorem red_shirt_pairs_correct : red_shirt_pairs 67 89 156 78 25 = 36 := by sorry

end red_shirt_pairs_red_shirt_pairs_correct_l2025_202567


namespace red_balls_count_l2025_202596

theorem red_balls_count (yellow_balls : ℕ) (total_balls : ℕ) (prob_yellow : ℚ) :
  yellow_balls = 10 →
  prob_yellow = 5/8 →
  total_balls ≤ 32 →
  total_balls = yellow_balls + (total_balls - yellow_balls) →
  (yellow_balls : ℚ) / total_balls = prob_yellow →
  total_balls - yellow_balls = 6 :=
by sorry

end red_balls_count_l2025_202596


namespace batsman_average_increase_l2025_202557

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  initialInnings : ℕ
  newScore : ℝ
  newAverage : ℝ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (b : Batsman) : ℝ :=
  b.newAverage - b.initialAverage

/-- Theorem: The batsman's average increases by 5 runs -/
theorem batsman_average_increase (b : Batsman) 
  (h1 : b.initialInnings = 10)
  (h2 : b.newScore = 95)
  (h3 : b.newAverage = 45)
  : averageIncrease b = 5 := by
  sorry

#check batsman_average_increase

end batsman_average_increase_l2025_202557


namespace cos_120_degrees_l2025_202520

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end cos_120_degrees_l2025_202520


namespace kanul_total_amount_l2025_202508

theorem kanul_total_amount 
  (raw_materials : ℝ) 
  (machinery : ℝ) 
  (cash_percentage : ℝ) 
  (total : ℝ) :
  raw_materials = 80000 →
  machinery = 30000 →
  cash_percentage = 0.20 →
  raw_materials + machinery + cash_percentage * total = total →
  total = 137500 :=
by sorry

end kanul_total_amount_l2025_202508


namespace mulch_cost_theorem_l2025_202551

/-- The cost of mulch in dollars per cubic foot -/
def mulch_cost_per_cubic_foot : ℝ := 5

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of mulch in cubic yards -/
def mulch_volume_cubic_yards : ℝ := 8

/-- Theorem: The cost of 8 cubic yards of mulch is 1080 dollars -/
theorem mulch_cost_theorem :
  mulch_volume_cubic_yards * cubic_yards_to_cubic_feet * mulch_cost_per_cubic_foot = 1080 := by
  sorry

end mulch_cost_theorem_l2025_202551


namespace periodic_function_value_l2025_202594

def f (x : ℝ) : ℝ := sorry

theorem periodic_function_value (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 4^x - 1) →
  f (-5.5) = 1 := by sorry

end periodic_function_value_l2025_202594


namespace factorization_x4_minus_y4_factorization_x3y_minus_2x2y2_plus_xy3_factorization_4x2_minus_4x_plus_1_factorization_4ab2_plus_1_plus_4ab_l2025_202546

-- (1)
theorem factorization_x4_minus_y4 (x y : ℝ) :
  x^4 - y^4 = (x^2 + y^2) * (x + y) * (x - y) := by sorry

-- (2)
theorem factorization_x3y_minus_2x2y2_plus_xy3 (x y : ℝ) :
  x^3*y - 2*x^2*y^2 + x*y^3 = x*y*(x - y)^2 := by sorry

-- (3)
theorem factorization_4x2_minus_4x_plus_1 (x : ℝ) :
  4*x^2 - 4*x + 1 = (2*x - 1)^2 := by sorry

-- (4)
theorem factorization_4ab2_plus_1_plus_4ab (a b : ℝ) :
  4*(a - b)^2 + 1 + 4*(a - b) = (2*a - 2*b + 1)^2 := by sorry

end factorization_x4_minus_y4_factorization_x3y_minus_2x2y2_plus_xy3_factorization_4x2_minus_4x_plus_1_factorization_4ab2_plus_1_plus_4ab_l2025_202546


namespace multiple_algorithms_exist_l2025_202525

/-- Represents a problem type -/
structure ProblemType where
  description : String

/-- Represents an algorithm -/
structure Algorithm where
  steps : List String

/-- Predicate to check if an algorithm solves a problem type -/
def solves (a : Algorithm) (p : ProblemType) : Prop :=
  sorry  -- Definition of what it means for an algorithm to solve a problem

/-- Theorem: There can exist multiple valid algorithms for a given problem type -/
theorem multiple_algorithms_exist (p : ProblemType) : 
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ solves a1 p ∧ solves a2 p := by
  sorry

#check multiple_algorithms_exist

end multiple_algorithms_exist_l2025_202525
