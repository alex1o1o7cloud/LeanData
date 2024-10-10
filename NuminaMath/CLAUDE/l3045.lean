import Mathlib

namespace sixth_group_number_l3045_304532

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  total_groups : ℕ
  first_group_number : ℕ
  eighth_group_number : ℕ

/-- Theorem stating the number drawn in the sixth group. -/
theorem sixth_group_number (s : SystematicSampling)
  (h1 : s.total_students = 800)
  (h2 : s.total_groups = 50)
  (h3 : s.eighth_group_number = 9 * s.first_group_number)
  : (s.first_group_number + 5 * (s.total_students / s.total_groups)) = 94 := by
  sorry


end sixth_group_number_l3045_304532


namespace brownies_shared_with_guests_l3045_304544

/-- The number of brownies shared with dinner guests -/
def brownies_shared (total : ℕ) (tina_days : ℕ) (husband_days : ℕ) (left : ℕ) : ℕ :=
  total - (2 * tina_days + husband_days) - left

/-- Theorem stating the number of brownies shared with dinner guests -/
theorem brownies_shared_with_guests :
  brownies_shared 24 5 5 5 = 4 := by
  sorry

end brownies_shared_with_guests_l3045_304544


namespace reader_current_page_l3045_304566

/-- Represents a book with a given number of pages -/
structure Book where
  pages : ℕ

/-- Represents a reader with a constant reading rate -/
structure Reader where
  rate : ℕ  -- pages per hour

/-- The current state of reading -/
structure ReadingState where
  book : Book
  reader : Reader
  previousPage : ℕ
  hoursAgo : ℕ
  hoursLeft : ℕ

/-- Calculate the current page number of the reader -/
def currentPage (state : ReadingState) : ℕ :=
  state.previousPage + state.reader.rate * state.hoursAgo

/-- Theorem: Given the conditions, prove that the reader's current page is 90 -/
theorem reader_current_page
  (state : ReadingState)
  (h1 : state.book.pages = 210)
  (h2 : state.previousPage = 60)
  (h3 : state.hoursAgo = 1)
  (h4 : state.hoursLeft = 4)
  (h5 : currentPage state + state.reader.rate * state.hoursLeft = state.book.pages) :
  currentPage state = 90 := by
  sorry


end reader_current_page_l3045_304566


namespace mike_initial_nickels_l3045_304597

/-- The number of nickels Mike's dad borrowed -/
def borrowed_nickels : ℕ := 75

/-- The number of nickels Mike has now -/
def current_nickels : ℕ := 12

/-- The number of nickels Mike had initially -/
def initial_nickels : ℕ := borrowed_nickels + current_nickels

theorem mike_initial_nickels :
  initial_nickels = 87 :=
by sorry

end mike_initial_nickels_l3045_304597


namespace system_solution_l3045_304578

theorem system_solution : 
  ∃ (x y : ℚ), 5 * x + 3 * y = 17 ∧ 3 * x + 5 * y = 16 → x = 37 / 16 := by
  sorry

end system_solution_l3045_304578


namespace product_of_three_numbers_l3045_304553

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 210 → 
  9 * x = y - 12 → 
  9 * x = z + 12 → 
  x < y → 
  x < z → 
  x * y * z = 746397 := by
sorry

end product_of_three_numbers_l3045_304553


namespace basketball_lineup_combinations_l3045_304587

theorem basketball_lineup_combinations (n : ℕ) (k₁ k₂ k₃ : ℕ) 
  (h₁ : n = 12) (h₂ : k₁ = 2) (h₃ : k₂ = 2) (h₄ : k₃ = 1) : 
  (n.choose k₁) * ((n - k₁).choose k₂) * ((n - k₁ - k₂).choose k₃) = 23760 :=
by sorry

end basketball_lineup_combinations_l3045_304587


namespace exam_average_l3045_304588

theorem exam_average (students1 : ℕ) (average1 : ℚ) (students2 : ℕ) (average2 : ℚ) : 
  students1 = 15 → 
  average1 = 75 / 100 → 
  students2 = 10 → 
  average2 = 95 / 100 → 
  (students1 * average1 + students2 * average2) / (students1 + students2) = 83 / 100 := by
sorry

end exam_average_l3045_304588


namespace polynomial_division_remainder_l3045_304517

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (3 * X ^ 3 - 2 * X ^ 2 - 23 * X + 60) = (X - 6) * q + (-378) := by
  sorry

end polynomial_division_remainder_l3045_304517


namespace ones_digit_of_large_power_l3045_304561

theorem ones_digit_of_large_power : ∃ n : ℕ, n > 0 ∧ 37^(37*(28^28)) ≡ 1 [ZMOD 10] :=
sorry

end ones_digit_of_large_power_l3045_304561


namespace parabola_y_relationship_l3045_304536

/-- Given a parabola y = -x² + 1 and three points on it, prove the relationship between their y-coordinates -/
theorem parabola_y_relationship : ∀ (y₁ y₂ y₃ : ℝ),
  ((-2 : ℝ), y₁) ∈ {(x, y) | y = -x^2 + 1} →
  ((-1 : ℝ), y₂) ∈ {(x, y) | y = -x^2 + 1} →
  ((3 : ℝ), y₃) ∈ {(x, y) | y = -x^2 + 1} →
  y₂ > y₁ ∧ y₁ > y₃ :=
by sorry

end parabola_y_relationship_l3045_304536


namespace allocation_schemes_l3045_304571

/-- The number of volunteers --/
def num_volunteers : ℕ := 5

/-- The number of tasks --/
def num_tasks : ℕ := 4

/-- The number of volunteers who cannot perform a specific task --/
def num_restricted : ℕ := 2

/-- Calculates the number of ways to allocate tasks to volunteers with restrictions --/
def num_allocations (n v t r : ℕ) : ℕ :=
  -- n: total number of volunteers
  -- v: number of volunteers to be selected
  -- t: number of tasks
  -- r: number of volunteers who cannot perform a specific task
  sorry

/-- Theorem stating the number of allocation schemes --/
theorem allocation_schemes :
  num_allocations num_volunteers num_tasks num_tasks num_restricted = 72 :=
by sorry

end allocation_schemes_l3045_304571


namespace opposite_of_neg_2020_l3045_304551

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem stating that the opposite of -2020 is 2020. -/
theorem opposite_of_neg_2020 : opposite (-2020) = 2020 := by sorry

end opposite_of_neg_2020_l3045_304551


namespace sum_of_absolute_coefficients_l3045_304542

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (3*x - 1)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| = 4^8 := by
  sorry

end sum_of_absolute_coefficients_l3045_304542


namespace initial_sum_equation_l3045_304506

/-- 
Given:
- Two interest rates: 15% and 10% per annum
- Compound interest applied annually for 3 years
- The difference in total interest between the two rates is Rs. 1500

Prove that the initial sum P satisfies the equation:
P * ((1 + 0.15)^3 - (1 + 0.10)^3) = 1500
-/
theorem initial_sum_equation (P : ℝ) : 
  P * ((1 + 0.15)^3 - (1 + 0.10)^3) = 1500 :=
by sorry

end initial_sum_equation_l3045_304506


namespace third_side_length_l3045_304572

theorem third_side_length (a b : ℝ) (θ : ℝ) (ha : a = 11) (hb : b = 15) (hθ : θ = 150 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos θ ∧ c = Real.sqrt (346 + 165 * Real.sqrt 3) :=
sorry

end third_side_length_l3045_304572


namespace problem_solution_l3045_304500

theorem problem_solution :
  (∀ a b : ℝ, 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b) ∧
  (∀ x y : ℝ, (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = -6*x*y + 5*y^2) := by
sorry

end problem_solution_l3045_304500


namespace music_library_avg_mb_per_hour_l3045_304599

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  space : ℕ

/-- Calculates the average megabytes per hour for a given music library -/
def avgMBPerHour (lib : MusicLibrary) : ℚ :=
  lib.space / (lib.days * 24)

/-- Theorem stating that for a music library with 15 days of music and 21,600 MB of space,
    the average megabytes per hour is 60 -/
theorem music_library_avg_mb_per_hour :
  let lib : MusicLibrary := { days := 15, space := 21600 }
  avgMBPerHour lib = 60 := by
  sorry

end music_library_avg_mb_per_hour_l3045_304599


namespace boys_without_calculators_l3045_304505

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ)
  (h1 : total_boys = 16)
  (h2 : students_with_calculators = 22)
  (h3 : girls_with_calculators = 13) :
  total_boys - (students_with_calculators - girls_with_calculators) = 7 := by
  sorry

end boys_without_calculators_l3045_304505


namespace arithmetic_sequence_sum_l3045_304540

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  arithmetic_sequence a d →
  a 1 = 0 →
  d ≠ 0 →
  a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) →
  m = 37 := by
  sorry

end arithmetic_sequence_sum_l3045_304540


namespace sequence_always_terminates_l3045_304515

def units_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def next_term (n : ℕ) : ℕ :=
  if units_digit n ≤ 5 then remove_last_digit n else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate next_term k a₀) = 0

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

end sequence_always_terminates_l3045_304515


namespace min_value_problem_l3045_304598

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / (x + 2)) + (1 / (y + 1)) ≥ 9/4 := by
  sorry

end min_value_problem_l3045_304598


namespace fractional_inequality_solution_set_l3045_304534

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := by sorry

end fractional_inequality_solution_set_l3045_304534


namespace greatest_valid_divisor_l3045_304583

def is_valid_divisor (n : ℕ) : Prop :=
  ∃ (r q : ℕ),
    (3510 % n = r) ∧ (18330 % n = r) ∧ (23790 % n = r) ∧
    (5680 % n = q) ∧ (14660 % n = q) ∧ (19050 % n = q)

theorem greatest_valid_divisor : 
  ∃! (n : ℕ), is_valid_divisor n ∧ ∀ (m : ℕ), is_valid_divisor m → m ≤ n :=
by sorry

end greatest_valid_divisor_l3045_304583


namespace problem_solution_l3045_304558

theorem problem_solution (c d : ℚ) 
  (eq1 : 4 + c = 5 - d) 
  (eq2 : 5 + d = 9 + c) : 
  4 - c = 11/2 := by
sorry

end problem_solution_l3045_304558


namespace cubic_root_sum_l3045_304593

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 28 * a - 2 = 0) →
  (40 * b^3 - 60 * b^2 + 28 * b - 2 = 0) →
  (40 * c^3 - 60 * c^2 + 28 * c - 2 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  a ≠ b →
  b ≠ c →
  a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.5 :=
by sorry

end cubic_root_sum_l3045_304593


namespace fourth_buoy_distance_l3045_304531

/-- Represents the distance of a buoy from the beach -/
def buoy_distance (n : ℕ) (interval : ℝ) : ℝ := n * interval

theorem fourth_buoy_distance 
  (h1 : buoy_distance 3 interval = 72) 
  (h2 : interval > 0) : 
  buoy_distance 4 interval = 96 :=
sorry

end fourth_buoy_distance_l3045_304531


namespace carter_goals_l3045_304579

theorem carter_goals (carter shelby judah : ℝ) 
  (shelby_half : shelby = carter / 2)
  (judah_calc : judah = 2 * shelby - 3)
  (total_goals : carter + shelby + judah = 7) :
  carter = 4 := by
sorry

end carter_goals_l3045_304579


namespace convex_ngon_coincidence_l3045_304523

/-- A convex n-gon in a 2D plane -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry  -- Axiom for convexity

/-- Predicate to check if one n-gon's vertices lie within another -/
def vertices_within (P Q : ConvexNGon n) : Prop := sorry

/-- Predicate to check if two n-gons are congruent -/
def congruent (P Q : ConvexNGon n) : Prop := sorry

/-- Predicate to check if vertices of two n-gons coincide -/
def vertices_coincide (P Q : ConvexNGon n) : Prop := sorry

/-- Theorem: If two congruent convex n-gons have the vertices of one within the other, 
    then their vertices coincide -/
theorem convex_ngon_coincidence (n : ℕ) (P Q : ConvexNGon n) 
  (h1 : vertices_within P Q) (h2 : congruent P Q) : 
  vertices_coincide P Q := by
  sorry

end convex_ngon_coincidence_l3045_304523


namespace quaternary_to_decimal_10231_l3045_304596

/-- Converts a quaternary (base 4) number to its decimal equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, digit) acc => acc + digit * (4 ^ i)) 0

/-- The quaternary representation of the number -/
def quaternary_num : List Nat := [1, 3, 2, 0, 1]

theorem quaternary_to_decimal_10231 :
  quaternary_to_decimal quaternary_num = 301 := by
  sorry

end quaternary_to_decimal_10231_l3045_304596


namespace geometric_sequence_product_l3045_304549

theorem geometric_sequence_product (a : ℝ) (r : ℝ) (h1 : a = 8/3) (h2 : a * r^4 = 27/2) :
  (a * r) * (a * r^2) * (a * r^3) = 216 := by
  sorry

end geometric_sequence_product_l3045_304549


namespace all_triplets_sum_to_two_l3045_304570

theorem all_triplets_sum_to_two :
  (3/4 + 1/4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3/5 + 4/5 + 3/5 = 2) ∧
  (2 + (-3) + 3 = 2) := by
  sorry

end all_triplets_sum_to_two_l3045_304570


namespace curve_c_and_max_dot_product_l3045_304582

-- Define the curve C
def C (x y : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ ((y / (x + 2)) * (y / (x - 2)) = -1/4)

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

-- Theorem statement
theorem curve_c_and_max_dot_product :
  -- Part 1: Equation of curve C
  (∀ x y : ℝ, C x y ↔ x^2/4 + y^2 = 1 ∧ x ≠ 2 ∧ x ≠ -2) ∧
  -- Part 2: Maximum value of OP · OQ
  (∀ k : ℝ, ∃ x1 y1 x2 y2 : ℝ,
    C x1 y1 ∧ C x2 y2 ∧  -- P and Q are on curve C
    y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1) ∧  -- P and Q are on line through D(1,0)
    dot_product x1 y1 x2 y2 ≤ 1/4) ∧
  (∃ x1 y1 x2 y2 : ℝ,
    C x1 y1 ∧ C x2 y2 ∧
    x1 = 1 ∧ x2 = 1 ∧  -- Line perpendicular to x-axis
    dot_product x1 y1 x2 y2 = 1/4) :=
by sorry

end curve_c_and_max_dot_product_l3045_304582


namespace median_to_AC_altitude_to_AB_l3045_304576

-- Define the triangle ABC
def A : ℝ × ℝ := (8, 5)
def B : ℝ × ℝ := (4, -2)
def C : ℝ × ℝ := (-6, 3)

-- Define the equations of the lines
def median_equation (x y : ℝ) : Prop := 2 * x + y - 6 = 0
def altitude_equation (x y : ℝ) : Prop := 4 * x + 7 * y + 3 = 0

-- Theorem for the median equation
theorem median_to_AC : 
  ∃ (m : ℝ × ℝ → ℝ × ℝ → Prop), 
    (∀ p, m ((A.1 + C.1) / 2, (A.2 + C.2) / 2) p ↔ m B p) ∧
    (∀ x y, m (x, y) (x, y) ↔ median_equation x y) :=
sorry

-- Theorem for the altitude equation
theorem altitude_to_AB :
  ∃ (l : ℝ × ℝ → ℝ × ℝ → Prop),
    (∀ p, l C p → (B.2 - A.2) * (p.1 - C.1) = (A.1 - B.1) * (p.2 - C.2)) ∧
    (∀ x y, l (x, y) (x, y) ↔ altitude_equation x y) :=
sorry

end median_to_AC_altitude_to_AB_l3045_304576


namespace spade_sum_equals_negative_sixteen_l3045_304556

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_sum_equals_negative_sixteen :
  (spade 2 3) + (spade 5 6) = -16 := by sorry

end spade_sum_equals_negative_sixteen_l3045_304556


namespace suv_length_sum_l3045_304509

/-- Represents the length of a line segment in the grid -/
inductive SegmentLength
  | Straight : SegmentLength  -- Length 1
  | Slanted : SegmentLength   -- Length √2

/-- Counts the number of each type of segment in a letter -/
structure LetterSegments :=
  (straight : ℕ)
  (slanted : ℕ)

/-- Represents the SUV acronym -/
structure SUVAcronym :=
  (S : LetterSegments)
  (U : LetterSegments)
  (V : LetterSegments)

def suv : SUVAcronym :=
  { S := { straight := 5, slanted := 4 },
    U := { straight := 6, slanted := 0 },
    V := { straight := 0, slanted := 2 } }

theorem suv_length_sum :
  let total_straight := suv.S.straight + suv.U.straight + suv.V.straight
  let total_slanted := suv.S.slanted + suv.U.slanted + suv.V.slanted
  total_straight + total_slanted * Real.sqrt 2 = 11 + 6 * Real.sqrt 2 :=
by sorry

end suv_length_sum_l3045_304509


namespace sector_area_120_deg_sqrt3_radius_l3045_304585

/-- The area of a sector with a central angle of 120° and a radius of √3 is π. -/
theorem sector_area_120_deg_sqrt3_radius (π : Real) : 
  let central_angle : Real := 120 * π / 180  -- Convert 120° to radians
  let radius : Real := Real.sqrt 3
  let sector_area : Real := (1/2) * radius^2 * central_angle
  sector_area = π := by
  sorry

end sector_area_120_deg_sqrt3_radius_l3045_304585


namespace cubic_root_product_sum_l3045_304510

theorem cubic_root_product_sum (p q r : ℝ) : 
  (6 * p^3 - 9 * p^2 + 14 * p - 10 = 0) →
  (6 * q^3 - 9 * q^2 + 14 * q - 10 = 0) →
  (6 * r^3 - 9 * r^2 + 14 * r - 10 = 0) →
  p * q + p * r + q * r = 7/3 := by
sorry

end cubic_root_product_sum_l3045_304510


namespace min_value_of_expression_l3045_304564

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 7) :
  (1 / (1 + a) + 4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 7 ∧
    1 / (1 + a₀) + 4 / (2 + b₀) = (13 + 4 * Real.sqrt 3) / 14 :=
sorry

end min_value_of_expression_l3045_304564


namespace equation_solution_l3045_304527

theorem equation_solution (a : ℚ) : (3/2 * 2 - 2*a = 0) → (2*a - 1 = 2) := by
  sorry

end equation_solution_l3045_304527


namespace lisa_candies_l3045_304512

/-- The number of candies Lisa eats on Mondays and Wednesdays combined per week -/
def candies_mon_wed : ℕ := 4

/-- The number of candies Lisa eats on other days combined per week -/
def candies_other_days : ℕ := 5

/-- The number of weeks it takes Lisa to eat all her candies -/
def weeks_to_eat_all : ℕ := 4

/-- The total number of candies Lisa has -/
def total_candies : ℕ := (candies_mon_wed + candies_other_days) * weeks_to_eat_all

theorem lisa_candies : total_candies = 36 := by
  sorry

end lisa_candies_l3045_304512


namespace candle_flower_groupings_l3045_304516

theorem candle_flower_groupings :
  let n_candles : ℕ := 4
  let k_candles : ℕ := 2
  let n_flowers : ℕ := 9
  let k_flowers : ℕ := 8
  (n_candles.choose k_candles) * (n_flowers.choose k_flowers) = 27 := by
  sorry

end candle_flower_groupings_l3045_304516


namespace binomial_30_3_l3045_304586

theorem binomial_30_3 : Nat.choose 30 3 = 12180 := by
  sorry

end binomial_30_3_l3045_304586


namespace triangle_abc_properties_l3045_304513

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  (1/2) * a * b * Real.sin C = 6 * Real.sqrt 3 →
  c * Real.sin C - a * Real.sin A = (b - a) * Real.sin B →
  C = π / 3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end triangle_abc_properties_l3045_304513


namespace tuition_room_board_difference_l3045_304522

/-- Given the total cost and tuition cost at State University, prove the difference between tuition and room and board costs. -/
theorem tuition_room_board_difference (total_cost tuition_cost : ℕ) 
  (h1 : total_cost = 2584)
  (h2 : tuition_cost = 1644)
  (h3 : tuition_cost > total_cost - tuition_cost) :
  tuition_cost - (total_cost - tuition_cost) = 704 := by
  sorry

end tuition_room_board_difference_l3045_304522


namespace special_linear_functions_f_one_l3045_304580

/-- Two linear functions satisfying specific properties -/
class SpecialLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  h_sum : ∀ x, f x + g x = 2
  h_comp : ∀ x, f (f x) = g (g x)
  h_f_zero : f 0 = 2022
  h_linear_f : ∃ a b : ℝ, ∀ x, f x = a * x + b
  h_linear_g : ∃ c d : ℝ, ∀ x, g x = c * x + d

/-- The main theorem stating f(1) = 2021 -/
theorem special_linear_functions_f_one
  (S : SpecialLinearFunctions) : S.f 1 = 2021 := by
  sorry

end special_linear_functions_f_one_l3045_304580


namespace smallest_c_for_inequality_l3045_304591

theorem smallest_c_for_inequality : 
  ∃ c : ℝ, c > 0 ∧ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c * |x - y| ≥ (x + y) / 2) ∧
  (∀ c' : ℝ, c' > 0 → 
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c' * |x - y| ≥ (x + y) / 2) → 
    c' ≥ c) ∧
  c = 1/2 := by
sorry

end smallest_c_for_inequality_l3045_304591


namespace z_purely_imaginary_z_in_second_quadrant_l3045_304520

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (2*m^2 - 3*m - 2) (m^2 - 3*m + 2)

/-- Theorem: z is purely imaginary if and only if m = -1/2 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = -1/2 := by sorry

/-- Theorem: z is in the second quadrant if and only if -1/2 < m < 1 -/
theorem z_in_second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1/2 < m ∧ m < 1 := by sorry

end z_purely_imaginary_z_in_second_quadrant_l3045_304520


namespace nine_friends_with_pears_l3045_304575

/-- The number of friends carrying pears -/
def friends_with_pears (total_friends orange_friends : ℕ) : ℕ :=
  total_friends - orange_friends

/-- Proof that 9 friends were carrying pears -/
theorem nine_friends_with_pears :
  friends_with_pears 15 6 = 9 := by
  sorry

end nine_friends_with_pears_l3045_304575


namespace cube_divisors_of_four_divisor_number_l3045_304533

/-- If an integer n has exactly 4 positive divisors (including 1 and n),
    then n^3 has 16 positive divisors. -/
theorem cube_divisors_of_four_divisor_number (n : ℕ) :
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 4 ∧ 1 ∈ d ∧ n ∈ d) →
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n^3} ∧ d.card = 16) :=
sorry

end cube_divisors_of_four_divisor_number_l3045_304533


namespace hyperbola_standard_equation_l3045_304537

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (0, Real.sqrt 10) →
  F₂ = (0, -Real.sqrt 10) →
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * 
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2 →
  M.2^2 / 9 - M.1^2 = 1 :=
by sorry

end hyperbola_standard_equation_l3045_304537


namespace company_capital_expenditure_l3045_304529

theorem company_capital_expenditure (C : ℚ) (C_pos : C > 0) :
  let raw_material_cost : ℚ := C / 4
  let remaining_after_raw : ℚ := C - raw_material_cost
  let machinery_cost : ℚ := remaining_after_raw / 10
  C - raw_material_cost - machinery_cost = (27 / 40) * C := by
  sorry

end company_capital_expenditure_l3045_304529


namespace decimal_25_to_binary_l3045_304508

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

theorem decimal_25_to_binary :
  decimal_to_binary 25 = [true, true, false, false, true] := by
  sorry

#eval decimal_to_binary 25

end decimal_25_to_binary_l3045_304508


namespace bird_count_proof_l3045_304555

theorem bird_count_proof (cardinals : ℕ) (robins : ℕ) (blue_jays : ℕ) (sparrows : ℕ) (pigeons : ℕ) :
  cardinals = 3 ∧
  robins = 4 * cardinals ∧
  blue_jays = 2 * cardinals ∧
  sparrows = 3 * cardinals + 1 ∧
  pigeons = 3 * blue_jays →
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry


end bird_count_proof_l3045_304555


namespace prob_three_green_out_of_six_l3045_304557

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 7

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of draws -/
def num_draws : ℕ := 6

/-- The number of green marbles we want to draw -/
def target_green : ℕ := 3

/-- The probability of drawing a green marble in a single draw -/
def prob_green : ℚ := green_marbles / total_marbles

/-- The probability of drawing a purple marble in a single draw -/
def prob_purple : ℚ := purple_marbles / total_marbles

/-- The probability of drawing exactly 3 green marbles out of 6 draws -/
theorem prob_three_green_out_of_six :
  (Nat.choose num_draws target_green : ℚ) * prob_green ^ target_green * prob_purple ^ (num_draws - target_green) =
  185220 / 1000000 := by
  sorry

end prob_three_green_out_of_six_l3045_304557


namespace centroid_distance_theorem_l3045_304514

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points --/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Distance from a point to a line --/
def distanceToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- Centroid of a triangle --/
def centroid (t : Triangle) : Point :=
  sorry

/-- Theorem: The distance from the centroid to a line equals the average of distances from vertices to the line --/
theorem centroid_distance_theorem (t : Triangle) (l : Line) :
  distanceToLine (centroid t) l = (distanceToLine t.A l + distanceToLine t.B l + distanceToLine t.C l) / 3 :=
by sorry

end centroid_distance_theorem_l3045_304514


namespace no_real_solutions_l3045_304581

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 9 ≠ 0 := by
  sorry

end no_real_solutions_l3045_304581


namespace fourth_sample_number_l3045_304504

def systematic_sample (population_size : ℕ) (sample_size : ℕ) (sample : Finset ℕ) : Prop :=
  sample.card = sample_size ∧
  ∃ k, ∀ x ∈ sample, ∃ i, x = k + i * (population_size / sample_size)

theorem fourth_sample_number
  (population_size : ℕ)
  (sample_size : ℕ)
  (sample : Finset ℕ)
  (h1 : population_size = 56)
  (h2 : sample_size = 4)
  (h3 : 6 ∈ sample)
  (h4 : 34 ∈ sample)
  (h5 : 48 ∈ sample)
  (h6 : systematic_sample population_size sample_size sample) :
  20 ∈ sample :=
sorry

end fourth_sample_number_l3045_304504


namespace triangle_cosine_problem_l3045_304590

theorem triangle_cosine_problem (A B C : ℝ) (a b : ℝ) :
  -- Conditions
  A + B + C = Real.pi ∧  -- Sum of angles in a triangle
  B = (A + B + C) / 3 ∧  -- Angles form arithmetic sequence
  a = 8 ∧ 
  b = 7 ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  -- Conclusion
  Real.cos C = -11/14 ∨ Real.cos C = -13/14 := by
sorry


end triangle_cosine_problem_l3045_304590


namespace unique_zero_implies_a_gt_one_a_gt_one_implies_unique_zero_characterization_of_a_l3045_304539

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

/-- The statement that f has exactly one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f a x = 0

/-- The main theorem: if f has exactly one zero in (0, 1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 := by
  sorry

/-- The converse: if a > 1, then f has exactly one zero in (0, 1) -/
theorem a_gt_one_implies_unique_zero :
  ∀ a : ℝ, a > 1 → has_unique_zero_in_interval a := by
  sorry

/-- The final theorem: f has exactly one zero in (0, 1) if and only if a > 1 -/
theorem characterization_of_a :
  ∀ a : ℝ, has_unique_zero_in_interval a ↔ a > 1 := by
  sorry

end unique_zero_implies_a_gt_one_a_gt_one_implies_unique_zero_characterization_of_a_l3045_304539


namespace wilfred_carrot_consumption_l3045_304567

/-- The number of carrots Wilfred eats on Tuesday -/
def tuesday_carrots : ℕ := 4

/-- The number of carrots Wilfred eats on Wednesday -/
def wednesday_carrots : ℕ := 6

/-- The number of carrots Wilfred plans to eat on Thursday -/
def thursday_carrots : ℕ := 5

/-- The total number of carrots Wilfred wants to eat from Tuesday to Thursday -/
def total_carrots : ℕ := tuesday_carrots + wednesday_carrots + thursday_carrots

theorem wilfred_carrot_consumption :
  total_carrots = 15 := by sorry

end wilfred_carrot_consumption_l3045_304567


namespace meaningful_square_root_range_l3045_304592

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / x) ↔ x ≥ -3 ∧ x ≠ 0 := by
sorry

end meaningful_square_root_range_l3045_304592


namespace no_triple_primes_l3045_304565

theorem no_triple_primes : ¬ ∃ p : ℕ, Prime p ∧ Prime (p + 7) ∧ Prime (p + 14) := by
  sorry

end no_triple_primes_l3045_304565


namespace square_difference_equals_one_l3045_304543

theorem square_difference_equals_one : 1.99^2 - 1.98 * 1.99 + 0.99^2 = 1 := by
  sorry

end square_difference_equals_one_l3045_304543


namespace geometric_sequence_sum_l3045_304546

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 2) →
  (a 5 + a 6 + a 7 + a 8 = 12) :=
by sorry

end geometric_sequence_sum_l3045_304546


namespace largest_number_proof_l3045_304552

def largest_number (a b : ℕ+) : Prop :=
  let hcf := Nat.gcd a b
  let lcm := Nat.lcm a b
  hcf = 154 ∧
  ∃ (k : ℕ), lcm = hcf * 19 * 23 * 37 * k ∧
  max a b = hcf * 19 * 23 * 37

theorem largest_number_proof (a b : ℕ+) (h : largest_number a b) :
  max a b = 2493726 :=
by sorry

end largest_number_proof_l3045_304552


namespace triangle_count_is_55_l3045_304560

/-- The number of distinct triangles formed from 12 points on a circle's circumference,
    where one specific point is always a vertex. -/
def num_triangles (total_points : ℕ) (fixed_points : ℕ) : ℕ :=
  Nat.choose (total_points - fixed_points) (2 : ℕ)

/-- Theorem stating that the number of triangles formed under the given conditions is 55. -/
theorem triangle_count_is_55 : num_triangles 12 1 = 55 := by
  sorry

end triangle_count_is_55_l3045_304560


namespace smallest_gcd_l3045_304528

theorem smallest_gcd (m n p : ℕ+) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ (n' p' : ℕ+), Nat.gcd m n' = 180 ∧ Nat.gcd m p' = 240 ∧ Nat.gcd n' p' = 60 ∧
  ∀ (n'' p'' : ℕ+), Nat.gcd m n'' = 180 → Nat.gcd m p'' = 240 → Nat.gcd n'' p'' ≥ 60 := by
  sorry

end smallest_gcd_l3045_304528


namespace percentage_of_b_l3045_304538

/-- Given that 8 is 4% of a, a certain percentage of b is 4, and c equals b / a, 
    prove that the percentage of b is 1 / (50c) -/
theorem percentage_of_b (a b c : ℝ) (h1 : 8 = 0.04 * a) (h2 : ∃ p, p * b = 4) (h3 : c = b / a) :
  ∃ p, p * b = 4 ∧ p = 1 / (50 * c) := by
  sorry

end percentage_of_b_l3045_304538


namespace inverse_proportion_problem_l3045_304545

/-- Given that x and y are inversely proportional, x + y = 30, and x - y = 10,
    prove that when x = 3, y = 200/3 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : x + y = 30) (h3 : x - y = 10) : 
  x = 3 → y = 200 / 3 := by
  sorry

end inverse_proportion_problem_l3045_304545


namespace expression_evaluation_l3045_304594

theorem expression_evaluation : 
  (200^2 - 13^2) / (140^2 - 23^2) * ((140 - 23) * (140 + 23)) / ((200 - 13) * (200 + 13)) + 1/10 = 11/10 := by
  sorry

end expression_evaluation_l3045_304594


namespace cubic_sum_theorem_l3045_304507

theorem cubic_sum_theorem (a b c : ℂ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 3) 
  (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = 8 := by
sorry

end cubic_sum_theorem_l3045_304507


namespace students_playing_both_sports_l3045_304563

theorem students_playing_both_sports 
  (total : ℕ) 
  (hockey : ℕ) 
  (basketball : ℕ) 
  (neither : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_basketball : basketball = 16)
  (h_neither : neither = 4) :
  hockey + basketball - (total - neither) = 10 :=
by sorry

end students_playing_both_sports_l3045_304563


namespace simplify_fraction_product_l3045_304518

theorem simplify_fraction_product : 
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 := by
  sorry

end simplify_fraction_product_l3045_304518


namespace hexagon_division_l3045_304503

/-- Given a regular hexagon with area 21.12 square centimeters divided into 6 equal pieces,
    prove that the area of each piece is 3.52 square centimeters. -/
theorem hexagon_division (hexagon_area : ℝ) (num_pieces : ℕ) (piece_area : ℝ) :
  hexagon_area = 21.12 ∧ num_pieces = 6 ∧ piece_area = hexagon_area / num_pieces →
  piece_area = 3.52 := by
  sorry

end hexagon_division_l3045_304503


namespace trinomial_expansion_terms_l3045_304526

/-- The number of different terms in the expansion of (a+b+c)^n -/
def num_terms_trinomial (n : ℕ) : ℕ :=
  Nat.choose (n + 2) 2

/-- The number of different terms in the expansion of (a+b)^n -/
def num_terms_binomial (n : ℕ) : ℕ := n + 1

theorem trinomial_expansion_terms :
  num_terms_trinomial 10 = 66 ∧ num_terms_binomial 10 = 11 := by sorry

end trinomial_expansion_terms_l3045_304526


namespace beach_pets_l3045_304559

theorem beach_pets (total : ℕ) (cat_only : ℕ) (both : ℕ) (dog_only : ℕ) (neither : ℕ) : 
  total = 522 →
  total = cat_only + both + dog_only + neither →
  (both : ℚ) / (cat_only + both) = 1/5 →
  (dog_only : ℚ) / (both + dog_only) = 7/10 →
  (neither : ℚ) / (neither + dog_only) = 1/2 →
  neither = 126 :=
by sorry

end beach_pets_l3045_304559


namespace quadratic_polynomial_root_sum_l3045_304589

theorem quadratic_polynomial_root_sum (Q : ℝ → ℝ) (a b c : ℝ) :
  (∀ x : ℝ, Q x = a * x^2 + b * x + c) →
  (∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) →
  (∃ r₁ r₂ : ℝ, ∀ x : ℝ, Q x = a * (x - r₁) * (x - r₂)) →
  r₁ + r₂ = 0 :=
by sorry

end quadratic_polynomial_root_sum_l3045_304589


namespace polynomial_expansion_l3045_304511

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x^3 + 4 * x - 5) * (4 * x^4 - 3 * x^2 + 2 * x - 7) = 
    12 * x^7 + 9 * x^5 + 10 * x^4 + 53 * x^2 - 14 * x + 25 := by
  sorry

end polynomial_expansion_l3045_304511


namespace min_distance_C₁_to_C₂_sum_distances_to_intersection_points_l3045_304577

-- Define the curves and point
def C₁ : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 1}
def C₂ : Set (ℝ × ℝ) := {(x, y) | y = x + 2}
def C₃ : Set (ℝ × ℝ) := {(x, y) | (x/2)^2 + (y/Real.sqrt 3)^2 = 1}
def P : ℝ × ℝ := (-1, 1)

-- State the theorems to be proved
theorem min_distance_C₁_to_C₂ :
  ∃ d : ℝ, d = Real.sqrt 2 - 1 ∧
  ∀ p ∈ C₁, ∀ q ∈ C₂, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
sorry

theorem sum_distances_to_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ C₂ ∧ A ∈ C₃ ∧ B ∈ C₂ ∧ B ∈ C₃ ∧
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end min_distance_C₁_to_C₂_sum_distances_to_intersection_points_l3045_304577


namespace imaginary_part_of_z2_l3045_304547

theorem imaginary_part_of_z2 (z₁ : ℂ) (h : z₁ = 1 - 2*I) :
  Complex.im ((z₁ + 1) / (z₁ - 1)) = 1 := by
  sorry

end imaginary_part_of_z2_l3045_304547


namespace dependent_variable_influences_l3045_304562

/-- Represents a linear regression model --/
structure LinearRegressionModel where
  y : ℝ  -- dependent variable
  x : ℝ  -- independent variable
  b : ℝ  -- slope
  a : ℝ  -- intercept
  e : ℝ  -- random error term

/-- The linear regression equation --/
def linear_regression_equation (model : LinearRegressionModel) : ℝ :=
  model.b * model.x + model.a + model.e

/-- Theorem stating that the dependent variable is influenced by both the independent variable and other factors --/
theorem dependent_variable_influences (model : LinearRegressionModel) :
  ∃ (other_factors : ℝ), model.y = linear_regression_equation model ∧ model.e ≠ 0 := by
  sorry

end dependent_variable_influences_l3045_304562


namespace square_ratio_theorem_l3045_304568

theorem square_ratio_theorem :
  let area_ratio : ℚ := 18 / 98
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  ∃ (a b c : ℕ), 
    side_ratio = (a : ℝ) * Real.sqrt b / c ∧
    a = 3 ∧ b = 1 ∧ c = 7 ∧
    a + b + c = 11 := by
  sorry

end square_ratio_theorem_l3045_304568


namespace max_n_value_l3045_304584

theorem max_n_value (n : ℤ) (h1 : 101 * n^2 ≤ 2525) (h2 : n ≤ 5) : n ≤ 5 ∧ ∃ m : ℤ, m = 5 ∧ 101 * m^2 ≤ 2525 := by
  sorry

end max_n_value_l3045_304584


namespace distance_focus_to_asymptotes_l3045_304554

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (1, 0)

-- Define the asymptotes of the hyperbola
def hyperbola_asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem distance_focus_to_asymptotes :
  ∃ (d : ℝ), d = Real.sqrt 3 / 2 ∧
  ∀ (x y : ℝ), hyperbola_asymptotes x y →
    d = (|Real.sqrt 3 * parabola_focus.1 + parabola_focus.2|) / Real.sqrt (3^2 + 1) :=
sorry

end distance_focus_to_asymptotes_l3045_304554


namespace remaining_wire_length_l3045_304535

def total_wire_length : ℝ := 60
def square_side_length : ℝ := 9

theorem remaining_wire_length :
  total_wire_length - 4 * square_side_length = 24 := by
  sorry

end remaining_wire_length_l3045_304535


namespace gcd_lcm_sum_12_80_l3045_304501

theorem gcd_lcm_sum_12_80 : Nat.gcd 12 80 + Nat.lcm 12 80 = 244 := by sorry

end gcd_lcm_sum_12_80_l3045_304501


namespace sine_increasing_for_acute_angles_l3045_304530

theorem sine_increasing_for_acute_angles (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  Real.sin α < Real.sin β := by
  sorry

end sine_increasing_for_acute_angles_l3045_304530


namespace sameColorPairsTheorem_l3045_304519

/-- The number of ways to choose a pair of socks of the same color -/
def sameColorPairs (total white brown green : ℕ) : ℕ :=
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose green 2

/-- Theorem: The number of ways to choose a pair of socks of the same color
    from 12 distinguishable socks (5 white, 5 brown, and 2 green) is 21 -/
theorem sameColorPairsTheorem :
  sameColorPairs 12 5 5 2 = 21 := by
  sorry

end sameColorPairsTheorem_l3045_304519


namespace inequality_proof_l3045_304521

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end inequality_proof_l3045_304521


namespace bus_trip_speed_l3045_304595

/-- Proves that for a trip of 420 miles, if increasing the average speed by 10 mph
    reduces the travel time by 2 hours, then the original average speed was 42 mph. -/
theorem bus_trip_speed (v : ℝ) (h : v > 0) :
  (420 / v) - (420 / (v + 10)) = 2 → v = 42 := by
sorry

end bus_trip_speed_l3045_304595


namespace maria_backpack_sheets_l3045_304541

/-- The number of sheets of paper in Maria's desk -/
def sheets_in_desk : ℕ := 50

/-- The total number of sheets of paper Maria has -/
def total_sheets : ℕ := 91

/-- The number of sheets of paper in Maria's backpack -/
def sheets_in_backpack : ℕ := total_sheets - sheets_in_desk

theorem maria_backpack_sheets : sheets_in_backpack = 41 := by
  sorry

end maria_backpack_sheets_l3045_304541


namespace unique_number_satisfying_conditions_l3045_304502

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def swap_hundreds_units (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * 1000 + b * 100 + d * 10 + c

def swap_thousands_tens (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  c * 1000 + b * 100 + a * 10 + d

theorem unique_number_satisfying_conditions (n : ℕ) :
  is_valid_number n ∧
  n + swap_hundreds_units n = 3332 ∧
  n + swap_thousands_tens n = 7886 ↔
  n = 1468 := by sorry

end unique_number_satisfying_conditions_l3045_304502


namespace barn_paint_area_l3045_304524

/-- Calculates the total area to be painted for a barn with given dimensions and conditions -/
def total_area_to_paint (length width height : ℝ) (door_width door_height : ℝ) : ℝ :=
  let wall_area := 2 * (width * height) * 2 + (length * height) * 2
  let roof_area := length * width
  let ceiling_area := length * width
  wall_area + roof_area + ceiling_area

/-- Theorem stating that the total area to be painted for the given barn is 860 square yards -/
theorem barn_paint_area :
  total_area_to_paint 15 10 8 2 3 = 860 := by sorry

end barn_paint_area_l3045_304524


namespace repeating_decimal_length_seven_thirteenths_l3045_304548

/-- The length of the repeating block in the decimal expansion of 7/13 is 6. -/
theorem repeating_decimal_length_seven_thirteenths : ∃ (d : ℕ+) (n : ℕ),
  (7 : ℚ) / 13 = (n : ℚ) / (10 ^ d.val - 1 : ℚ) ∧ d = 6 := by
  sorry

end repeating_decimal_length_seven_thirteenths_l3045_304548


namespace inequality_solution_set_l3045_304569

theorem inequality_solution_set (x : ℝ) :
  (3 / (x + 2) + 4 / (x + 8) ≥ 3 / 4) ↔ 
  (x ∈ Set.Icc (-10.125) (-8) ∪ Set.Ico (-2) 4.125) :=
by sorry

end inequality_solution_set_l3045_304569


namespace no_computers_infected_after_attack_l3045_304525

/-- Represents a circular network of computers -/
structure ComputerNetwork where
  size : Nat
  is_circular : size > 0

/-- Represents a virus in the network -/
structure Virus where
  target : Nat
  current : Nat

/-- The state of the network during the attack -/
structure NetworkState where
  network : ComputerNetwork
  viruses : List Virus
  infected : Finset Nat

/-- Simulates the propagation of a single virus -/
def propagate_virus (state : NetworkState) (v : Virus) : NetworkState :=
  sorry

/-- Simulates the entire virus attack -/
def simulate_attack (initial_state : NetworkState) : NetworkState :=
  sorry

/-- Theorem stating that after the attack, no computers remain infected -/
theorem no_computers_infected_after_attack (n : ComputerNetwork) 
  (h : n.size = 100) : 
  let initial_state : NetworkState := {
    network := n,
    viruses := List.range n.size |>.map (λ i => { target := i, current := i }),
    infected := ∅
  }
  let final_state := simulate_attack initial_state
  final_state.infected.card = 0 := by
  sorry

end no_computers_infected_after_attack_l3045_304525


namespace p_sufficient_not_necessary_for_q_l3045_304573

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)
def q (x : ℝ) : Prop := x^2 = 3 * x + 4

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l3045_304573


namespace inscribed_semicircle_radius_l3045_304550

/-- An isosceles triangle with a semicircle inscribed -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle lies along the base of the triangle -/
  diameter_on_base : radius * 2 ≤ base

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangleWithSemicircle)
    (h_base : t.base = 20)
    (h_height : t.height = 18) :
    t.radius = 180 / (Real.sqrt 424 + 10) := by
  sorry

end inscribed_semicircle_radius_l3045_304550


namespace offspring_ratio_l3045_304574

-- Define the traits
inductive Cotyledon
| Green
| Brown

inductive SeedShape
| Round
| Kidney

-- Define the genotypes
structure Genotype where
  cotyledon : Bool  -- True for Y, False for y
  seedShape : Bool  -- True for R, False for r

-- Define the phenotypes
def phenotype (g : Genotype) : Cotyledon × SeedShape :=
  (if g.cotyledon then Cotyledon.Green else Cotyledon.Brown,
   if g.seedShape then SeedShape.Round else SeedShape.Kidney)

-- Define the inheritance rule
def inherit (parent1 parent2 : Genotype) : Genotype :=
  { cotyledon := parent1.cotyledon || parent2.cotyledon
  , seedShape := parent1.seedShape || parent2.seedShape }

-- Define the parental combinations
def parent1 : Genotype := { cotyledon := true, seedShape := true }   -- YyRr
def parent2 : Genotype := { cotyledon := false, seedShape := false } -- yyrr
def parent3 : Genotype := { cotyledon := true, seedShape := false }  -- Yyrr
def parent4 : Genotype := { cotyledon := false, seedShape := true }  -- yyRr

-- Theorem statement
theorem offspring_ratio 
  (independent_inheritance : ∀ p1 p2, inherit p1 p2 = inherit p2 p1) :
  (∃ (f : Genotype → Genotype → Fin 4), 
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Round → f parent1 parent2 = 0) ∧
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Kidney → f parent1 parent2 = 1) ∧
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Round → f parent1 parent2 = 2) ∧
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Kidney → f parent1 parent2 = 3)) ∧
  (∃ (f : Genotype → Genotype → Fin 4), 
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Round → f parent3 parent4 = 0) ∧
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Kidney → f parent3 parent4 = 1) ∧
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Round → f parent3 parent4 = 2) ∧
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Kidney → f parent3 parent4 = 3)) :=
by sorry

end offspring_ratio_l3045_304574
