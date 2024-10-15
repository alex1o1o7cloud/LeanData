import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_equality_l665_66579

theorem square_difference_equality : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l665_66579


namespace NUMINAMATH_CALUDE_log_inequality_l665_66578

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 3 / Real.log 4 →
  b = Real.log 4 / Real.log 3 →
  c = Real.log 3 / Real.log 5 →
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l665_66578


namespace NUMINAMATH_CALUDE_half_inequality_l665_66559

theorem half_inequality (a b : ℝ) (h : a > b) : a/2 > b/2 := by
  sorry

end NUMINAMATH_CALUDE_half_inequality_l665_66559


namespace NUMINAMATH_CALUDE_quadrilateral_property_l665_66549

-- Define a quadrilateral as a tuple of four natural numbers
def Quadrilateral := (ℕ × ℕ × ℕ × ℕ)

-- Define a property that each side divides the sum of the other three
def DivisibilityProperty (q : Quadrilateral) : Prop :=
  let (a, b, c, d) := q
  (a ∣ b + c + d) ∧ (b ∣ a + c + d) ∧ (c ∣ a + b + d) ∧ (d ∣ a + b + c)

-- Define a property that at least two sides are equal
def TwoSidesEqual (q : Quadrilateral) : Prop :=
  let (a, b, c, d) := q
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d

-- The main theorem
theorem quadrilateral_property (q : Quadrilateral) :
  DivisibilityProperty q → TwoSidesEqual q :=
by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_property_l665_66549


namespace NUMINAMATH_CALUDE_distribute_six_books_three_people_l665_66573

/-- The number of ways to distribute n different books among k people, 
    with each person getting at least 1 book -/
def distribute_books (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 6 different books among 3 people, 
    with each person getting at least 1 book, can be done in 540 ways -/
theorem distribute_six_books_three_people : 
  distribute_books 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_distribute_six_books_three_people_l665_66573


namespace NUMINAMATH_CALUDE_inequality_preservation_l665_66591

theorem inequality_preservation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : 
  a - c < b - c := by
sorry

end NUMINAMATH_CALUDE_inequality_preservation_l665_66591


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l665_66546

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to (tc + b), then t = -24/17 -/
theorem parallel_vectors_t_value (a b c : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (-3, 4))
  (h2 : b = (-1, 5))
  (h3 : c = (2, 3))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 - c.1, a.2 - c.2) = k • (t * c.1 + b.1, t * c.2 + b.2)) :
  t = -24/17 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l665_66546


namespace NUMINAMATH_CALUDE_perpendicular_lines_l665_66528

/-- Two lines y = ax - 2 and y = (a + 2)x + 1 are perpendicular if and only if a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∧ y = (a + 2) * x + 1 → a * (a + 2) = -1) ↔ 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l665_66528


namespace NUMINAMATH_CALUDE_prize_probability_l665_66539

theorem prize_probability (odds_favorable : ℕ) (odds_unfavorable : ℕ) 
  (h_odds : odds_favorable = 5 ∧ odds_unfavorable = 6) :
  let total_outcomes := odds_favorable + odds_unfavorable
  let prob_not_prize := odds_unfavorable / total_outcomes
  (prob_not_prize ^ 2 : ℚ) = 36 / 121 :=
by sorry

end NUMINAMATH_CALUDE_prize_probability_l665_66539


namespace NUMINAMATH_CALUDE_max_trees_in_garden_l665_66513

def garden_width : ℝ := 27.9
def tree_interval : ℝ := 3.1

theorem max_trees_in_garden : 
  ⌊garden_width / tree_interval⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_max_trees_in_garden_l665_66513


namespace NUMINAMATH_CALUDE_coffee_cost_calculation_coffee_cost_calculation_proof_l665_66506

/-- The daily cost of making coffee given a coffee machine purchase and previous coffee consumption habits. -/
theorem coffee_cost_calculation (machine_cost : ℝ) (discount : ℝ) (previous_coffees_per_day : ℕ) 
  (previous_coffee_price : ℝ) (payback_days : ℕ) (daily_cost : ℝ) : Prop :=
  machine_cost = 200 ∧ 
  discount = 20 ∧
  previous_coffees_per_day = 2 ∧
  previous_coffee_price = 4 ∧
  payback_days = 36 →
  daily_cost = 3

/-- Proof of the coffee cost calculation theorem. -/
theorem coffee_cost_calculation_proof : 
  coffee_cost_calculation 200 20 2 4 36 3 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_calculation_coffee_cost_calculation_proof_l665_66506


namespace NUMINAMATH_CALUDE_tims_bodyguard_payment_l665_66531

/-- The amount Tim pays his bodyguards in a week -/
def weekly_payment (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Theorem stating the total amount Tim pays his bodyguards in a week -/
theorem tims_bodyguard_payment :
  weekly_payment 2 20 8 7 = 2240 := by
  sorry

#eval weekly_payment 2 20 8 7

end NUMINAMATH_CALUDE_tims_bodyguard_payment_l665_66531


namespace NUMINAMATH_CALUDE_probability_three_girls_l665_66518

/-- The probability of choosing 3 girls from a group of 15 members (8 girls and 7 boys) -/
theorem probability_three_girls (total : ℕ) (girls : ℕ) (boys : ℕ) (chosen : ℕ) : 
  total = 15 → girls = 8 → boys = 7 → chosen = 3 →
  (Nat.choose girls chosen : ℚ) / (Nat.choose total chosen : ℚ) = 8 / 65 := by
sorry

end NUMINAMATH_CALUDE_probability_three_girls_l665_66518


namespace NUMINAMATH_CALUDE_a_less_than_one_l665_66587

-- Define the function f
def f (x : ℝ) : ℝ := -x^5 - 3*x^3 - 5*x + 3

-- State the theorem
theorem a_less_than_one (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_one_l665_66587


namespace NUMINAMATH_CALUDE_exactly_one_female_probability_l665_66555

def total_students : ℕ := 50
def male_students : ℕ := 30
def female_students : ℕ := 20
def group_size : ℕ := 5

def male_in_group : ℕ := male_students * group_size / total_students
def female_in_group : ℕ := female_students * group_size / total_students

theorem exactly_one_female_probability : 
  (male_in_group * female_in_group * 2) / (group_size * (group_size - 1)) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_female_probability_l665_66555


namespace NUMINAMATH_CALUDE_polynomial_properties_l665_66545

/-- The polynomial -8x³y^(m+1) + xy² - 3/4x³ + 6y is a sixth-degree quadrinomial -/
def is_sixth_degree_quadrinomial (m : ℕ) : Prop :=
  3 + (m + 1) = 6

/-- The monomial 2/5πx^ny^(5-m) has the same degree as the polynomial -/
def monomial_same_degree (m n : ℕ) : Prop :=
  n + (5 - m) = 6

/-- The polynomial coefficients sum to -7/4 -/
def coefficients_sum : ℚ :=
  -8 + 1 + (-3/4) + 6

theorem polynomial_properties :
  ∃ (m n : ℕ),
    is_sixth_degree_quadrinomial m ∧
    monomial_same_degree m n ∧
    m = 2 ∧
    n = 3 ∧
    coefficients_sum = -7/4 := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l665_66545


namespace NUMINAMATH_CALUDE_sum_of_digits_of_M_l665_66508

/-- Represents a four-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  is_four_digit : 1000 ≤ d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ d1 * 1000 + d2 * 100 + d3 * 10 + d4 < 10000

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  n.d1 * 1000 + n.d2 * 100 + n.d3 * 10 + n.d4

/-- The product of digits of a four-digit number -/
def FourDigitNumber.digitProduct (n : FourDigitNumber) : Nat :=
  n.d1 * n.d2 * n.d3 * n.d4

/-- The sum of digits of a four-digit number -/
def FourDigitNumber.digitSum (n : FourDigitNumber) : Nat :=
  n.d1 + n.d2 + n.d3 + n.d4

/-- M is the greatest four-digit number whose digits have a product of 24 -/
def M : FourDigitNumber :=
  sorry

theorem sum_of_digits_of_M :
  M.digitProduct = 24 ∧ 
  (∀ n : FourDigitNumber, n.digitProduct = 24 → n.value ≤ M.value) →
  M.digitSum = 13 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_M_l665_66508


namespace NUMINAMATH_CALUDE_shari_walked_13_miles_l665_66519

/-- Represents Shari's walking pattern -/
structure WalkingPattern where
  rate1 : ℝ  -- Rate for the first phase in miles per hour
  time1 : ℝ  -- Time for the first phase in hours
  rate2 : ℝ  -- Rate for the second phase in miles per hour
  time2 : ℝ  -- Time for the second phase in hours

/-- Calculates the total distance walked given a WalkingPattern -/
def totalDistance (w : WalkingPattern) : ℝ :=
  w.rate1 * w.time1 + w.rate2 * w.time2

/-- Shari's actual walking pattern -/
def sharisWalk : WalkingPattern :=
  { rate1 := 4
    time1 := 2
    rate2 := 5
    time2 := 1 }

/-- Theorem stating that Shari walked 13 miles in total -/
theorem shari_walked_13_miles :
  totalDistance sharisWalk = 13 := by
  sorry


end NUMINAMATH_CALUDE_shari_walked_13_miles_l665_66519


namespace NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l665_66529

/-- Represents a house in the square yard -/
inductive House
| NorthEast
| NorthWest
| SouthEast
| SouthWest

/-- Represents a quarrel between two friends -/
structure Quarrel where
  house1 : House
  house2 : House
  day : Nat

/-- Define what it means for two houses to be neighbors -/
def are_neighbors (h1 h2 : House) : Bool :=
  match h1, h2 with
  | House.NorthEast, House.NorthWest => true
  | House.NorthEast, House.SouthEast => true
  | House.NorthWest, House.SouthWest => true
  | House.SouthEast, House.SouthWest => true
  | House.NorthWest, House.NorthEast => true
  | House.SouthEast, House.NorthEast => true
  | House.SouthWest, House.NorthWest => true
  | House.SouthWest, House.SouthEast => true
  | _, _ => false

/-- The main theorem to prove -/
theorem quarrel_between_opposite_houses 
  (total_friends : Nat) 
  (quarrels : List Quarrel) 
  (h1 : total_friends = 77)
  (h2 : quarrels.length = 365)
  (h3 : ∀ q ∈ quarrels, q.house1 ≠ q.house2)
  (h4 : ∀ h1 h2 : House, are_neighbors h1 h2 → 
    ∃ q ∈ quarrels, (q.house1 = h1 ∧ q.house2 = h2) ∨ (q.house1 = h2 ∧ q.house2 = h1)) :
  ∃ q ∈ quarrels, ¬are_neighbors q.house1 q.house2 := by
sorry

end NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l665_66529


namespace NUMINAMATH_CALUDE_hiking_problem_l665_66584

/-- Hiking problem -/
theorem hiking_problem (R_up : ℝ) (R_down : ℝ) (T_up : ℝ) (T_down : ℝ) (D_down : ℝ) :
  R_up = 7 →
  R_down = 1.5 * R_up →
  T_up = T_down →
  D_down = 21 →
  T_up = 2 :=
by sorry

end NUMINAMATH_CALUDE_hiking_problem_l665_66584


namespace NUMINAMATH_CALUDE_tangent_unique_tangent_values_l665_66520

/-- A line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ k : ℝ,
  (1 : ℝ)^3 + a * 1 + b = 3 ∧  -- The point (1, 3) is on the curve
  k * 1 + 1 = 3 ∧              -- The point (1, 3) is on the line
  3 * (1 : ℝ)^2 + a = k        -- The slope of the curve at x = 1 equals the slope of the line

/-- The values of a and b for which the line is tangent to the curve at (1, 3) are unique -/
theorem tangent_unique : ∃! (a b : ℝ), is_tangent a b :=
sorry

/-- The unique values of a and b for which the line is tangent to the curve at (1, 3) are -1 and 3 respectively -/
theorem tangent_values : ∃! (a b : ℝ), is_tangent a b ∧ a = -1 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_unique_tangent_values_l665_66520


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l665_66552

/-- Represents the circular path and the walkers' characteristics -/
structure CircularPath where
  totalBlocks : ℕ
  janeSpeedMultiplier : ℕ

/-- Represents the distance walked by each person when they meet -/
structure MeetingPoint where
  hectorDistance : ℕ
  janeDistance : ℕ

/-- Calculates the meeting point given a circular path -/
def calculateMeetingPoint (path : CircularPath) : MeetingPoint :=
  sorry

/-- Theorem stating that Hector walks 6 blocks when they meet -/
theorem meeting_point_theorem (path : CircularPath) 
  (h1 : path.totalBlocks = 24)
  (h2 : path.janeSpeedMultiplier = 3) :
  (calculateMeetingPoint path).hectorDistance = 6 :=
  sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l665_66552


namespace NUMINAMATH_CALUDE_ratio_of_recurring_decimals_l665_66535

/-- The value of the repeating decimal 0.848484... -/
def recurring_84 : ℚ := 84 / 99

/-- The value of the repeating decimal 0.212121... -/
def recurring_21 : ℚ := 21 / 99

/-- Theorem stating that the ratio of the two repeating decimals is equal to 4 -/
theorem ratio_of_recurring_decimals : recurring_84 / recurring_21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_recurring_decimals_l665_66535


namespace NUMINAMATH_CALUDE_repeating_decimal_6_is_two_thirds_l665_66571

def repeating_decimal_6 : ℚ := 0.6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666

theorem repeating_decimal_6_is_two_thirds : repeating_decimal_6 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_6_is_two_thirds_l665_66571


namespace NUMINAMATH_CALUDE_square_side_length_l665_66532

theorem square_side_length (total_width total_height : ℕ) 
  (h_width : total_width = 4040)
  (h_height : total_height = 2420)
  (h_rectangles_equal : ∃ (r : ℕ), r = r) -- R₁ and R₂ have identical dimensions
  (h_squares : ∃ (s r : ℕ), s + r = s + r) -- S₁ and S₃ side length = S₂ side length + R₁ side length
  : ∃ (s : ℕ), s = 810 ∧ 
    ∃ (r : ℕ), 2 * r + s = total_height ∧ 
                2 * r + 3 * s = total_width :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l665_66532


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l665_66592

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ n : ℕ, p n 2 → (∃ m : ℕ, n = 2 * m)) →
  (∀ n : ℕ, p 2 n → n = 2 ∨ n > 2) →
  p (3^20 + 11^14) 2 ∧ ∀ q : ℕ, p (3^20 + 11^14) q → q ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l665_66592


namespace NUMINAMATH_CALUDE_dividend_in_terms_of_a_l665_66511

theorem dividend_in_terms_of_a (a : ℝ) :
  let divisor := 25 * quotient
  let divisor' := 7 * remainder
  let quotient_minus_remainder := 15
  let remainder := 3 * a
  let dividend := divisor * quotient + remainder
  dividend = 225 * a^2 + 1128 * a + 5625 := by
  sorry

end NUMINAMATH_CALUDE_dividend_in_terms_of_a_l665_66511


namespace NUMINAMATH_CALUDE_straight_line_no_dot_l665_66502

/-- Represents the properties of an alphabet with dots and straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  dotOnly : ℕ
  allHaveEither : Bool

/-- Theorem: In the given alphabet, the number of letters with a straight line but no dot is 36 -/
theorem straight_line_no_dot (a : Alphabet) 
  (h1 : a.total = 60)
  (h2 : a.both = 20)
  (h3 : a.dotOnly = 4)
  (h4 : a.allHaveEither = true) : 
  a.total - a.both - a.dotOnly = 36 := by
  sorry

#check straight_line_no_dot

end NUMINAMATH_CALUDE_straight_line_no_dot_l665_66502


namespace NUMINAMATH_CALUDE_correct_article_usage_l665_66538

/-- Represents the possible article choices for each blank -/
inductive Article
  | A
  | The
  | None

/-- Represents the sentence structure with two article blanks -/
structure Sentence where
  first_blank : Article
  second_blank : Article

/-- Defines the correct article usage based on the given conditions -/
def correct_usage : Sentence :=
  { first_blank := Article.A,  -- Gottlieb Daimler is referred to generally
    second_blank := Article.The }  -- The car invention is referred to specifically

/-- Theorem stating that the correct usage is "a" for the first blank and "the" for the second -/
theorem correct_article_usage :
  correct_usage = { first_blank := Article.A, second_blank := Article.The } :=
by sorry

end NUMINAMATH_CALUDE_correct_article_usage_l665_66538


namespace NUMINAMATH_CALUDE_center_top_second_row_value_l665_66588

/-- Represents a 4x4 grid of real numbers -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Checks if a sequence of 4 real numbers is arithmetic -/
def IsArithmeticSequence (s : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 3, s (i + 1) - s i = d

/-- The property that each row and column of the grid is an arithmetic sequence -/
def GridProperty (g : Grid) : Prop :=
  (∀ i : Fin 4, IsArithmeticSequence (λ j ↦ g i j)) ∧
  (∀ j : Fin 4, IsArithmeticSequence (λ i ↦ g i j))

theorem center_top_second_row_value
  (g : Grid)
  (h_grid : GridProperty g)
  (h_first_row : g 0 0 = 4 ∧ g 0 3 = 16)
  (h_last_row : g 3 0 = 10 ∧ g 3 3 = 40) :
  g 1 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_center_top_second_row_value_l665_66588


namespace NUMINAMATH_CALUDE_laptop_sale_price_l665_66501

def original_price : ℝ := 1000.00
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15

theorem laptop_sale_price :
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 612.00 := by
sorry

end NUMINAMATH_CALUDE_laptop_sale_price_l665_66501


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l665_66542

theorem largest_multiple_of_9_under_100 : 
  ∃ n : ℕ, n * 9 = 99 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l665_66542


namespace NUMINAMATH_CALUDE_james_flowers_per_day_l665_66537

theorem james_flowers_per_day 
  (total_volunteers : ℕ) 
  (days_worked : ℕ) 
  (total_flowers : ℕ) 
  (h1 : total_volunteers = 5)
  (h2 : days_worked = 2)
  (h3 : total_flowers = 200)
  (h4 : total_flowers % (total_volunteers * days_worked) = 0) :
  total_flowers / (total_volunteers * days_worked) = 20 := by
sorry

end NUMINAMATH_CALUDE_james_flowers_per_day_l665_66537


namespace NUMINAMATH_CALUDE_projectile_max_height_l665_66594

/-- The height of the projectile as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- The time at which the projectile reaches its maximum height -/
def t_max : ℝ := 1

theorem projectile_max_height :
  ∃ (max_height : ℝ), max_height = h t_max ∧ 
  ∀ (t : ℝ), h t ≤ max_height ∧
  max_height = 45 := by
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l665_66594


namespace NUMINAMATH_CALUDE_custom_equation_solution_l665_66551

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- State the theorem
theorem custom_equation_solution :
  ∀ x : ℝ, star 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_custom_equation_solution_l665_66551


namespace NUMINAMATH_CALUDE_dinner_cost_calculation_l665_66512

theorem dinner_cost_calculation (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
    (h_total : total_cost = 27.5)
    (h_tax : tax_rate = 0.1)
    (h_tip : tip_rate = 0.15) : 
  ∃ (base_cost : ℝ), 
    base_cost * (1 + tax_rate + tip_rate) = total_cost ∧ 
    base_cost = 22 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_calculation_l665_66512


namespace NUMINAMATH_CALUDE_expected_red_pairs_value_l665_66564

/-- Represents a standard 104-card deck -/
structure Deck :=
  (cards : Finset (Fin 104))
  (size : cards.card = 104)

/-- Represents the color of a card -/
inductive Color
| Red
| Black

/-- Function to determine the color of a card -/
def color (card : Fin 104) : Color :=
  if card.val ≤ 51 then Color.Red else Color.Black

/-- Number of red cards in the deck -/
def num_red_cards : Nat := 52

/-- Calculates the expected number of adjacent red card pairs in a 104-card deck -/
def expected_red_pairs (d : Deck) : ℚ :=
  (num_red_cards : ℚ) * ((num_red_cards - 1) / (d.cards.card - 1))

/-- Theorem: The expected number of adjacent red card pairs is 2652/103 -/
theorem expected_red_pairs_value (d : Deck) :
  expected_red_pairs d = 2652 / 103 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_value_l665_66564


namespace NUMINAMATH_CALUDE_intersection_sum_l665_66581

theorem intersection_sum (m b : ℝ) : 
  (2 * m * 3 + 3 = 9) →  -- First line passes through (3, 9)
  (4 * 3 + b = 9) →      -- Second line passes through (3, 9)
  b + 2 * m = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l665_66581


namespace NUMINAMATH_CALUDE_samson_sandwiches_l665_66521

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

theorem samson_sandwiches (monday_lunch : ℕ) (monday_dinner : ℕ) (monday_total : ℕ) :
  monday_lunch = 3 →
  monday_dinner = 2 * monday_lunch →
  monday_total = monday_lunch + monday_dinner →
  monday_total = tuesday_breakfast + 8 →
  tuesday_breakfast = 1 := by sorry

end NUMINAMATH_CALUDE_samson_sandwiches_l665_66521


namespace NUMINAMATH_CALUDE_sin_sum_equality_l665_66523

theorem sin_sum_equality : 
  Real.sin (7 * π / 30) + Real.sin (11 * π / 30) = 
  Real.sin (π / 30) + Real.sin (13 * π / 30) + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l665_66523


namespace NUMINAMATH_CALUDE_unique_plants_count_l665_66530

/-- Represents a flower bed -/
structure FlowerBed where
  plants : ℕ

/-- Represents the overlap between two flower beds -/
structure Overlap where
  plants : ℕ

/-- Represents the overlap among three flower beds -/
structure TripleOverlap where
  plants : ℕ

/-- Calculates the total number of unique plants across three overlapping flower beds -/
def totalUniquePlants (a b c : FlowerBed) (ab ac bc : Overlap) (abc : TripleOverlap) : ℕ :=
  a.plants + b.plants + c.plants - ab.plants - ac.plants - bc.plants + abc.plants

/-- Theorem stating that the total number of unique plants across three specific overlapping flower beds is 1320 -/
theorem unique_plants_count :
  let a : FlowerBed := ⟨600⟩
  let b : FlowerBed := ⟨550⟩
  let c : FlowerBed := ⟨400⟩
  let ab : Overlap := ⟨60⟩
  let ac : Overlap := ⟨110⟩
  let bc : Overlap := ⟨90⟩
  let abc : TripleOverlap := ⟨30⟩
  totalUniquePlants a b c ab ac bc abc = 1320 := by
  sorry

end NUMINAMATH_CALUDE_unique_plants_count_l665_66530


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l665_66577

theorem rectangle_dimensions (x y : ℝ) : 
  (2*x + y) * (2*y) = 90 ∧ x*y = 10 → x = 2 ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l665_66577


namespace NUMINAMATH_CALUDE_week_net_change_l665_66524

/-- The net change in stock exchange points for a week -/
def net_change (monday tuesday wednesday thursday friday : Int) : Int :=
  monday + tuesday + wednesday + thursday + friday

/-- Theorem stating that the net change for the given week is -119 -/
theorem week_net_change :
  net_change (-150) 106 (-47) 182 (-210) = -119 := by
  sorry

end NUMINAMATH_CALUDE_week_net_change_l665_66524


namespace NUMINAMATH_CALUDE_same_color_probability_l665_66507

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (purple : Nat)
  (green : Nat)
  (orange : Nat)
  (glittery : Nat)
  (total : Nat)
  (h1 : purple + green + orange + glittery = total)
  (h2 : total = 30)

/-- The probability of rolling the same color on two identical colored dice -/
def sameProbability (d : ColoredDie) : Rat :=
  (d.purple^2 + d.green^2 + d.orange^2 + d.glittery^2) / d.total^2

/-- Two 30-sided dice with specified colored sides -/
def twoDice : ColoredDie :=
  { purple := 6
    green := 10
    orange := 12
    glittery := 2
    total := 30
    h1 := by rfl
    h2 := by rfl }

theorem same_color_probability :
  sameProbability twoDice = 71 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l665_66507


namespace NUMINAMATH_CALUDE_johnny_table_legs_l665_66544

/-- Given the number of tables, planks per surface, and total planks,
    calculate the number of planks needed for the legs of each table. -/
def planksForLegs (numTables : ℕ) (planksPerSurface : ℕ) (totalPlanks : ℕ) : ℕ :=
  (totalPlanks - numTables * planksPerSurface) / numTables

/-- Theorem stating that given the specific values in the problem,
    the number of planks needed for the legs of each table is 4. -/
theorem johnny_table_legs :
  planksForLegs 5 5 45 = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnny_table_legs_l665_66544


namespace NUMINAMATH_CALUDE_complex_equation_solution_l665_66515

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 + a*i)*i = 3 + i) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l665_66515


namespace NUMINAMATH_CALUDE_grading_problem_solution_l665_66503

/-- Represents the grading scenario of Teacher Wang --/
structure GradingScenario where
  initial_rate : ℕ            -- Initial grading rate (assignments per hour)
  new_rate : ℕ                -- New grading rate (assignments per hour)
  change_time : ℕ             -- Time at which the grading rate changed (in hours)
  time_saved : ℕ              -- Time saved due to rate change (in hours)
  total_assignments : ℕ       -- Total number of assignments in the batch

/-- Theorem stating the solution to the grading problem --/
theorem grading_problem_solution (scenario : GradingScenario) : 
  scenario.initial_rate = 6 →
  scenario.new_rate = 8 →
  scenario.change_time = 2 →
  scenario.time_saved = 3 →
  scenario.total_assignments = 84 :=
by sorry

end NUMINAMATH_CALUDE_grading_problem_solution_l665_66503


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l665_66517

theorem sqrt_a_sqrt_a_sqrt_a (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a * Real.sqrt a * Real.sqrt a) = a := by sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l665_66517


namespace NUMINAMATH_CALUDE_infinite_sum_reciprocal_squared_plus_two_l665_66589

/-- The infinite sum of 1/(n^2(n+2)) from n=1 to infinity is equal to π^2/12 -/
theorem infinite_sum_reciprocal_squared_plus_two : 
  ∑' (n : ℕ), 1 / (n^2 * (n + 2 : ℝ)) = π^2 / 12 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_reciprocal_squared_plus_two_l665_66589


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l665_66562

theorem no_rational_solutions_for_positive_k : ¬ ∃ (k : ℕ+), ∃ (x : ℚ), k.val * x^2 + 16 * x + k.val = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l665_66562


namespace NUMINAMATH_CALUDE_wax_calculation_l665_66522

theorem wax_calculation (total_wax : ℕ) (additional_wax : ℕ) (possessed_wax : ℕ) : 
  total_wax = 353 → additional_wax = 22 → possessed_wax = total_wax - additional_wax → possessed_wax = 331 := by
  sorry

end NUMINAMATH_CALUDE_wax_calculation_l665_66522


namespace NUMINAMATH_CALUDE_divisibility_theorem_l665_66533

theorem divisibility_theorem (K M N : ℤ) (hK : K ≠ 0) (hM : M ≠ 0) (hN : N ≠ 0) (hcoprime : Nat.Coprime K.natAbs M.natAbs) :
  ∃ x : ℤ, ∃ y : ℤ, M * x + N = K * y := by
sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l665_66533


namespace NUMINAMATH_CALUDE_prob_12th_roll_last_l665_66582

/-- The number of sides on the die -/
def n : ℕ := 8

/-- The number of rolls -/
def r : ℕ := 12

/-- The probability of rolling the same number on the rth roll as on the (r-1)th roll -/
def p_same : ℚ := 1 / n

/-- The probability of rolling a different number on the rth roll from the (r-1)th roll -/
def p_diff : ℚ := (n - 1) / n

/-- The probability that the rth roll is the last roll in the sequence -/
def prob_last_roll (r : ℕ) : ℚ := p_diff^(r - 2) * p_same

theorem prob_12th_roll_last :
  prob_last_roll r = (n - 1)^(r - 2) / n^r := by sorry

end NUMINAMATH_CALUDE_prob_12th_roll_last_l665_66582


namespace NUMINAMATH_CALUDE_davids_physics_marks_l665_66572

def english_marks : ℕ := 36
def math_marks : ℕ := 35
def chemistry_marks : ℕ := 57
def biology_marks : ℕ := 55
def average_marks : ℕ := 45
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  let total_marks := average_marks * num_subjects
  let known_marks := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks := total_marks - known_marks
  physics_marks = 42 := by sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l665_66572


namespace NUMINAMATH_CALUDE_integer_x_is_seven_l665_66599

theorem integer_x_is_seven (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 9 > x ∧ x > 6)
  (h4 : 8 > x ∧ x > 0)
  (h5 : x + 1 < 9) :
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_x_is_seven_l665_66599


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l665_66554

theorem smallest_root_of_quadratic (x : ℝ) :
  (10 * x^2 - 48 * x + 44 = 0) →
  (∀ y : ℝ, 10 * y^2 - 48 * y + 44 = 0 → x ≤ y) →
  x = 1.234 := by
sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l665_66554


namespace NUMINAMATH_CALUDE_derivative_exponential_sine_derivative_rational_function_derivative_logarithm_derivative_polynomial_product_derivative_cosine_l665_66543

-- Function 1: y = e^(sin x)
theorem derivative_exponential_sine (x : ℝ) :
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x :=
sorry

-- Function 2: y = (x + 3) / (x + 2)
theorem derivative_rational_function (x : ℝ) :
  deriv (fun x => (x + 3) / (x + 2)) x = -1 / (x + 2)^2 :=
sorry

-- Function 3: y = ln(2x + 3)
theorem derivative_logarithm (x : ℝ) :
  deriv (fun x => Real.log (2 * x + 3)) x = 2 / (2 * x + 3) :=
sorry

-- Function 4: y = (x^2 + 2)(2x - 1)
theorem derivative_polynomial_product (x : ℝ) :
  deriv (fun x => (x^2 + 2) * (2 * x - 1)) x = 6 * x^2 - 2 * x + 4 :=
sorry

-- Function 5: y = cos(2x + π/3)
theorem derivative_cosine (x : ℝ) :
  deriv (fun x => Real.cos (2 * x + Real.pi / 3)) x = -2 * Real.sin (2 * x + Real.pi / 3) :=
sorry

end NUMINAMATH_CALUDE_derivative_exponential_sine_derivative_rational_function_derivative_logarithm_derivative_polynomial_product_derivative_cosine_l665_66543


namespace NUMINAMATH_CALUDE_sum_set_bounds_l665_66560

theorem sum_set_bounds (A : Finset ℕ) (S : Finset ℕ) :
  A.card = 100 →
  S = Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A) →
  199 ≤ S.card ∧ S.card ≤ 5050 := by
  sorry

end NUMINAMATH_CALUDE_sum_set_bounds_l665_66560


namespace NUMINAMATH_CALUDE_sum_of_B_elements_l665_66548

/-- A finite set with two elements -/
inductive TwoElementSet
  | e1
  | e2

/-- The mapping f from A to B -/
def f (x : TwoElementSet) : ℝ :=
  match x with
  | TwoElementSet.e1 => 1^2
  | TwoElementSet.e2 => 3^2

/-- The set B as a function from TwoElementSet to ℝ -/
def B : TwoElementSet → ℝ := f

theorem sum_of_B_elements : (B TwoElementSet.e1) + (B TwoElementSet.e2) = 10 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_B_elements_l665_66548


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l665_66527

theorem solution_set_of_inequalities :
  ∀ x : ℝ, (2 * x > -1 ∧ x - 1 ≤ 0) ↔ (-1/2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l665_66527


namespace NUMINAMATH_CALUDE_f_properties_l665_66580

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties :
  ∃ (x₀ : ℝ),
    (∀ x > 0, HasDerivAt f (Real.log x + 1) x) ∧
    (HasDerivAt f 2 x₀ → x₀ = Real.exp 1) ∧
    (∀ x ≥ Real.exp (-1), StrictMono f) ∧
    (∃! p, f p = -Real.exp (-1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l665_66580


namespace NUMINAMATH_CALUDE_max_red_tiles_100x100_l665_66505

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents the number of colors used for tiling -/
def num_colors : ℕ := 4

/-- Defines the property that no two tiles of the same color touch each other -/
def no_adjacent_same_color (g : Grid) : Prop := sorry

/-- The maximum number of tiles of a single color in the grid -/
def max_single_color_tiles (g : Grid) : ℕ := (g.size ^ 2) / 4

/-- Theorem stating the maximum number of red tiles in a 100x100 grid -/
theorem max_red_tiles_100x100 (g : Grid) (h1 : g.size = 100) (h2 : no_adjacent_same_color g) : 
  max_single_color_tiles g = 2500 := by sorry

end NUMINAMATH_CALUDE_max_red_tiles_100x100_l665_66505


namespace NUMINAMATH_CALUDE_positive_integer_square_minus_five_times_zero_l665_66540

theorem positive_integer_square_minus_five_times_zero (w : ℕ+) 
  (h : w.val ^ 2 - 5 * w.val = 0) : w.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_square_minus_five_times_zero_l665_66540


namespace NUMINAMATH_CALUDE_brick_length_calculation_l665_66516

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℝ := 2500  -- 25 meters = 2500 cm
def courtyard_width : ℝ := 1600   -- 16 meters = 1600 cm

-- Define the brick properties
def brick_width : ℝ := 10         -- 10 cm
def total_bricks : ℕ := 20000

-- Define the theorem
theorem brick_length_calculation :
  ∃ (brick_length : ℝ),
    brick_length > 0 ∧
    brick_length * brick_width * total_bricks = courtyard_length * courtyard_width ∧
    brick_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l665_66516


namespace NUMINAMATH_CALUDE_complement_intersection_equals_d_l665_66556

-- Define the universe
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}

-- Define sets M and N
def M : Set Char := {'a', 'b', 'c'}
def N : Set Char := {'a', 'c', 'e'}

-- State the theorem
theorem complement_intersection_equals_d :
  (U \ M) ∩ (U \ N) = {'d'} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_d_l665_66556


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l665_66596

theorem quadratic_equation_roots (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*a*x₁ + a^2 - 4 = 0) ∧ 
  (x₂^2 - 2*a*x₂ + a^2 - 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l665_66596


namespace NUMINAMATH_CALUDE_car_rental_cost_l665_66567

theorem car_rental_cost (total_cost : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) :
  total_cost = 46.12 ∧ 
  miles_driven = 214 ∧ 
  cost_per_mile = 0.08 →
  ∃ daily_rental_cost : ℝ, 
    daily_rental_cost = 29 ∧ 
    total_cost = daily_rental_cost + miles_driven * cost_per_mile :=
by sorry

end NUMINAMATH_CALUDE_car_rental_cost_l665_66567


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l665_66595

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the set {1, 2}
def set_1_2 : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary_condition :
  (∀ m ∈ set_1_2, log10 m < 1) ∧
  (∃ m : ℝ, log10 m < 1 ∧ m ∉ set_1_2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l665_66595


namespace NUMINAMATH_CALUDE_total_books_l665_66504

theorem total_books (jason_books mary_books : ℕ) 
  (h1 : jason_books = 18) 
  (h2 : mary_books = 42) : 
  jason_books + mary_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l665_66504


namespace NUMINAMATH_CALUDE_inscribed_triangle_polygon_sides_l665_66534

/-- A triangle inscribed in a circle with specific angle relationships -/
structure InscribedTriangle where
  -- The circle in which the triangle is inscribed
  circle : Real
  -- The angles of the triangle
  angleA : Real
  angleB : Real
  angleC : Real
  -- The number of sides of the regular polygon
  n : ℕ
  -- Conditions
  angle_sum : angleA + angleB + angleC = 180
  angle_B : angleB = 3 * angleA
  angle_C : angleC = 5 * angleA
  polygon_arc : (360 : Real) / n = 140

/-- Theorem: The number of sides of the regular polygon is 5 -/
theorem inscribed_triangle_polygon_sides (t : InscribedTriangle) : t.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_polygon_sides_l665_66534


namespace NUMINAMATH_CALUDE_necklace_labeling_theorem_l665_66561

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

structure Necklace :=
  (beads : ℕ)

def valid_labeling (n : ℕ) (A B : Necklace) (labeling : ℕ → ℕ) : Prop :=
  (∀ i, i < A.beads + B.beads → n ≤ labeling i ∧ labeling i ≤ n + 32) ∧
  (∀ i j, i ≠ j → i < A.beads + B.beads → j < A.beads + B.beads → labeling i ≠ labeling j) ∧
  (∀ i, i < A.beads - 1 → is_coprime (labeling i) (labeling (i + 1))) ∧
  (is_coprime (labeling 0) (labeling (A.beads - 1))) ∧
  (∀ i, A.beads ≤ i ∧ i < A.beads + B.beads - 1 → is_coprime (labeling i) (labeling (i + 1))) ∧
  (is_coprime (labeling A.beads) (labeling (A.beads + B.beads - 1)))

theorem necklace_labeling_theorem (n : ℕ) (A B : Necklace) 
  (h_n_odd : is_odd n) (h_n_ge_1 : n ≥ 1) (h_A : A.beads = 14) (h_B : B.beads = 19) :
  ∃ labeling : ℕ → ℕ, valid_labeling n A B labeling :=
sorry

end NUMINAMATH_CALUDE_necklace_labeling_theorem_l665_66561


namespace NUMINAMATH_CALUDE_even_sum_and_sum_greater_20_count_l665_66585

def IntSet := Finset (Nat)

def range_1_to_20 : IntSet := Finset.range 20

def even_sum_pairs (s : IntSet) : Nat :=
  (s.filter (λ x => x ≤ 20)).card

def sum_greater_20_pairs (s : IntSet) : Nat :=
  (s.filter (λ x => x ≤ 20)).card

theorem even_sum_and_sum_greater_20_count :
  (even_sum_pairs range_1_to_20 = 90) ∧
  (sum_greater_20_pairs range_1_to_20 = 100) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_and_sum_greater_20_count_l665_66585


namespace NUMINAMATH_CALUDE_linear_function_k_value_l665_66547

/-- Given that the point (-1, -2) lies on the graph of y = kx - 4 and k ≠ 0, prove that k = -2 -/
theorem linear_function_k_value (k : ℝ) : k ≠ 0 ∧ -2 = k * (-1) - 4 → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l665_66547


namespace NUMINAMATH_CALUDE_order_of_magnitude_l665_66586

noncomputable def a : ℝ := Real.exp (Real.exp 1)
noncomputable def b : ℝ := Real.pi ^ Real.pi
noncomputable def c : ℝ := Real.exp Real.pi
noncomputable def d : ℝ := Real.pi ^ (Real.exp 1)

theorem order_of_magnitude : a < d ∧ d < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l665_66586


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l665_66557

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), 2^504 * k = 14^504 - 8^252 ∧ 
  ∀ (m : ℕ), 2^m * k = 14^504 - 8^252 → m ≤ 504 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l665_66557


namespace NUMINAMATH_CALUDE_license_plate_count_l665_66536

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The total number of characters in the license plate -/
def total_chars : ℕ := 8

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 6

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 2

/-- The number of positions where the two-letter word can be placed -/
def word_positions : ℕ := total_chars - num_plate_letters + 1

/-- The number of positions for the fixed digit 7 -/
def fixed_digit_positions : ℕ := total_chars - 1

theorem license_plate_count :
  (fixed_digit_positions) * (num_letters ^ num_plate_letters) * (num_digits ^ (num_plate_digits - 1)) = 47320000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l665_66536


namespace NUMINAMATH_CALUDE_product_xyz_l665_66553

theorem product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 162)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 198)
  (h4 : x + y + z = 26) :
  x * y * z = 2294.67 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l665_66553


namespace NUMINAMATH_CALUDE_five_plumbers_three_areas_l665_66590

/-- The number of ways to assign plumbers to residential areas. -/
def assignment_plans (n_plumbers : ℕ) (n_areas : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating the number of assignment plans for 5 plumbers and 3 areas. -/
theorem five_plumbers_three_areas : 
  assignment_plans 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_plumbers_three_areas_l665_66590


namespace NUMINAMATH_CALUDE_inverse_g_150_l665_66593

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^4 + 6

-- State the theorem
theorem inverse_g_150 : 
  g ((2 : ℝ) * (3 : ℝ)^(1/4)) = 150 :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_150_l665_66593


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l665_66565

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  positive : ∀ n, a n > 0
  initial : a 1 = 1
  geometric : (a 3) * (a 11) = (a 4 + 5/2)^2
  arithmetic : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

/-- The theorem to be proved -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) (m n : ℕ) 
  (h : m - n = 8) : seq.a m - seq.a n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l665_66565


namespace NUMINAMATH_CALUDE_total_hamburgers_made_l665_66570

def initial_hamburgers : ℝ := 9.0
def additional_hamburgers : ℝ := 3.0

theorem total_hamburgers_made :
  initial_hamburgers + additional_hamburgers = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburgers_made_l665_66570


namespace NUMINAMATH_CALUDE_parallelogram_values_l665_66514

/-- Represents a parallelogram EFGH with given side lengths and area formula -/
structure Parallelogram where
  x : ℝ
  y : ℝ
  ef : ℝ := 5 * x + 7
  fg : ℝ := 4 * y + 1
  gh : ℝ := 27
  he : ℝ := 19
  area : ℝ := 2 * x^2 + y^2 + 5 * x * y + 3

/-- Theorem stating the values of x, y, and area for the given parallelogram -/
theorem parallelogram_values (p : Parallelogram) :
  p.x = 4 ∧ p.y = 4.5 ∧ p.area = 145.25 := by sorry

end NUMINAMATH_CALUDE_parallelogram_values_l665_66514


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l665_66583

/-- Given a rectangle with length 4x inches and width 3x + 4 inches,
    where its area is twice its perimeter, prove that x = 1. -/
theorem rectangle_area_perimeter_relation (x : ℝ) : 
  (4 * x) * (3 * x + 4) = 2 * (2 * (4 * x) + 2 * (3 * x + 4)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l665_66583


namespace NUMINAMATH_CALUDE_square_sum_difference_l665_66509

theorem square_sum_difference (n : ℕ) : 
  (2*n+1)^2 - (2*n-1)^2 + (2*n-3)^2 - (2*n-5)^2 + (2*n-7)^2 - (2*n-9)^2 + 
  (2*n-11)^2 - (2*n-13)^2 + (2*n-15)^2 - (2*n-17)^2 + (2*n-19)^2 - 
  (2*n-21)^2 + (2*n-23)^2 - (2*n-25)^2 + (2*n-27)^2 = 389 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_difference_l665_66509


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_eight_l665_66526

/-- Represents a geometric sequence with common ratio 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => 2 * GeometricSequence a n

/-- Sum of the first n terms of the geometric sequence -/
def SumGeometric (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (GeometricSequence a) |>.sum

theorem geometric_sequence_sum_eight (a : ℝ) :
  SumGeometric a 4 = 1 → SumGeometric a 8 = 17 := by
  sorry

#check geometric_sequence_sum_eight

end NUMINAMATH_CALUDE_geometric_sequence_sum_eight_l665_66526


namespace NUMINAMATH_CALUDE_johnson_martinez_tie_l665_66568

/-- Represents the months of the baseball season --/
inductive Month
| Mar
| Apr
| May
| Jun
| Jul
| Aug
| Sep

/-- Calculates the cumulative home runs for a player --/
def cumulativeHomeRuns (monthlyData : List Nat) : List Nat :=
  List.scanl (· + ·) 0 monthlyData

/-- Checks if two lists are equal up to a certain index --/
def equalUpTo (l1 l2 : List Nat) (index : Nat) : Bool :=
  (l1.take index) = (l2.take index)

/-- Finds the first index where two lists become equal --/
def firstEqualIndex (l1 l2 : List Nat) : Option Nat :=
  (List.range l1.length).find? (fun i => l1[i]! = l2[i]!)

theorem johnson_martinez_tie (johnsonData martinezData : List Nat) 
    (h1 : johnsonData = [3, 8, 15, 12, 5, 7, 14])
    (h2 : martinezData = [0, 3, 9, 20, 7, 12, 13]) : 
    firstEqualIndex 
      (cumulativeHomeRuns johnsonData) 
      (cumulativeHomeRuns martinezData) = some 6 := by
  sorry

#check johnson_martinez_tie

end NUMINAMATH_CALUDE_johnson_martinez_tie_l665_66568


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l665_66576

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l665_66576


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l665_66597

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 2/5, then the ratio of A's area to B's area is 4:25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 5) (h2 : b / d = 2 / 5) :
  (a * b) / (c * d) = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l665_66597


namespace NUMINAMATH_CALUDE_average_of_numbers_l665_66598

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125320.5 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l665_66598


namespace NUMINAMATH_CALUDE_inequality_proof_l665_66566

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + 3*c) / (3*a + 3*b + 2*c) + 
  (a + 3*b + c) / (3*a + 2*b + 3*c) + 
  (3*a + b + c) / (2*a + 3*b + 3*c) ≥ 15/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l665_66566


namespace NUMINAMATH_CALUDE_stream_speed_l665_66563

/-- Proves that given a boat with a speed of 57 km/h in still water, 
    if the time taken to row upstream is twice the time taken to row downstream 
    for the same distance, then the speed of the stream is 19 km/h. -/
theorem stream_speed (d : ℝ) (h : d > 0) : 
  let boat_speed := 57
  let stream_speed := 19
  (d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 19 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l665_66563


namespace NUMINAMATH_CALUDE_expression_evaluation_l665_66500

theorem expression_evaluation : 
  |Real.sqrt 3 - 2| + (Real.pi - Real.sqrt 10)^0 - Real.sqrt 12 = 3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l665_66500


namespace NUMINAMATH_CALUDE_radius_of_inscribed_circle_in_curvilinear_triangle_l665_66525

/-- 
Given a rhombus with height h and acute angle α, and two inscribed circles:
1. One circle inscribed in the rhombus
2. Another circle inscribed in the curvilinear triangle formed by the rhombus and the first circle

This theorem states that the radius r of the second circle (inscribed in the curvilinear triangle)
is equal to (h/2) * tan²(45° - α/4)
-/
theorem radius_of_inscribed_circle_in_curvilinear_triangle 
  (h : ℝ) (α : ℝ) (h_pos : h > 0) (α_acute : 0 < α ∧ α < π/2) :
  ∃ r : ℝ, r = (h/2) * (Real.tan (π/4 - α/4))^2 ∧ 
  r > 0 ∧ 
  r < h/2 := by
sorry

end NUMINAMATH_CALUDE_radius_of_inscribed_circle_in_curvilinear_triangle_l665_66525


namespace NUMINAMATH_CALUDE_sons_age_l665_66541

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l665_66541


namespace NUMINAMATH_CALUDE_decimal_124_to_base_5_has_three_consecutive_digits_l665_66574

/-- Convert a decimal number to base 5 --/
def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Check if a list of digits has three consecutive identical digits --/
def has_three_consecutive_digits (digits : List ℕ) : Prop :=
  ∃ i, i + 2 < digits.length ∧
       digits[i]! = digits[i+1]! ∧
       digits[i+1]! = digits[i+2]!

/-- The main theorem --/
theorem decimal_124_to_base_5_has_three_consecutive_digits :
  has_three_consecutive_digits (to_base_5 124) :=
sorry

end NUMINAMATH_CALUDE_decimal_124_to_base_5_has_three_consecutive_digits_l665_66574


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l665_66569

/-- Proves that the speed of a boat in still water is 13 km/hr given the conditions -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 68)
  (h3 : downstream_time = 4)
  : ∃ (boat_speed : ℝ), boat_speed = 13 ∧ 
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l665_66569


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l665_66558

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -x^2 + 2*x + 3 > 0}
def B : Set ℝ := {x : ℝ | x - 2 < 0}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l665_66558


namespace NUMINAMATH_CALUDE_steve_calculation_l665_66550

theorem steve_calculation (x : ℝ) : (x / 8) - 20 = 12 → (x * 8) + 20 = 2068 := by
  sorry

end NUMINAMATH_CALUDE_steve_calculation_l665_66550


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l665_66575

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  failed_english = 50 →
  failed_both = 25 →
  passed_both = 50 →
  ∃ failed_hindi : ℝ, failed_hindi = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l665_66575


namespace NUMINAMATH_CALUDE_closest_point_on_line_l665_66510

/-- The line equation y = 2x - 4 -/
def line_equation (x : ℝ) : ℝ := 2 * x - 4

/-- The point we're finding the closest point to -/
def given_point : ℝ × ℝ := (3, 1)

/-- The claimed closest point on the line -/
def closest_point : ℝ × ℝ := (2.6, 1.2)

/-- Theorem stating that the closest_point is on the line and is the closest to given_point -/
theorem closest_point_on_line :
  (line_equation closest_point.1 = closest_point.2) ∧
  ∀ (p : ℝ × ℝ), (line_equation p.1 = p.2) →
    (closest_point.1 - given_point.1)^2 + (closest_point.2 - given_point.2)^2 ≤
    (p.1 - given_point.1)^2 + (p.2 - given_point.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l665_66510
