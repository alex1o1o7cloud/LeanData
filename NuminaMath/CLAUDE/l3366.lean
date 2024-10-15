import Mathlib

namespace NUMINAMATH_CALUDE_tribe_leadership_combinations_l3366_336623

def tribe_size : ℕ := 12
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem tribe_leadership_combinations :
  (tribe_size) *
  (tribe_size - num_chiefs) *
  (tribe_size - num_chiefs - 1) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs) num_inferior_officers_per_chief) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs - num_inferior_officers_per_chief) num_inferior_officers_per_chief) = 221760 :=
by sorry

end NUMINAMATH_CALUDE_tribe_leadership_combinations_l3366_336623


namespace NUMINAMATH_CALUDE_gcd_40304_30213_l3366_336611

theorem gcd_40304_30213 : Nat.gcd 40304 30213 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_40304_30213_l3366_336611


namespace NUMINAMATH_CALUDE_no_primes_from_200_l3366_336650

def change_one_digit (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ (i : Fin 3), ∃ (d : Fin 10), 
    m = n + d * (10 ^ i.val) - (n / (10 ^ i.val) % 10) * (10 ^ i.val)}

theorem no_primes_from_200 :
  ∀ n ∈ change_one_digit 200, ¬ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_no_primes_from_200_l3366_336650


namespace NUMINAMATH_CALUDE_total_books_on_cart_l3366_336692

/-- The number of books Nancy shelved from the cart -/
structure BookCart where
  top_history : ℕ
  top_romance : ℕ
  top_poetry : ℕ
  bottom_western : ℕ
  bottom_biography : ℕ
  bottom_scifi : ℕ
  bottom_culinary : ℕ

/-- The theorem stating the total number of books on the cart -/
theorem total_books_on_cart (cart : BookCart) : 
  cart.top_history = 12 →
  cart.top_romance = 8 →
  cart.top_poetry = 4 →
  cart.bottom_western = 5 →
  cart.bottom_biography = 6 →
  cart.bottom_scifi = 3 →
  cart.bottom_culinary = 2 →
  ∃ (total : ℕ), total = 88 ∧ 
    total = cart.top_history + cart.top_romance + cart.top_poetry + 
            (cart.bottom_western + cart.bottom_biography + cart.bottom_scifi + cart.bottom_culinary) * 4 :=
by sorry


end NUMINAMATH_CALUDE_total_books_on_cart_l3366_336692


namespace NUMINAMATH_CALUDE_proposition_relationship_l3366_336638

theorem proposition_relationship (x y : ℝ) :
  (∀ x y, x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 3) ∧ x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l3366_336638


namespace NUMINAMATH_CALUDE_consecutive_majors_probability_l3366_336645

/-- Represents the number of people around the table -/
def total_people : ℕ := 12

/-- Represents the number of math majors -/
def math_majors : ℕ := 5

/-- Represents the number of physics majors -/
def physics_majors : ℕ := 4

/-- Represents the number of biology majors -/
def biology_majors : ℕ := 3

/-- Represents the probability of the desired seating arrangement -/
def seating_probability : ℚ := 1 / 5775

theorem consecutive_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := 
    Nat.factorial (math_majors - 1) * Nat.factorial physics_majors * Nat.factorial biology_majors
  (favorable_arrangements : ℚ) / total_arrangements = seating_probability := by
  sorry

end NUMINAMATH_CALUDE_consecutive_majors_probability_l3366_336645


namespace NUMINAMATH_CALUDE_ship_cannot_escape_illumination_l3366_336695

/-- Represents a lighthouse with a rotating beam -/
structure Lighthouse where
  beam_length : ℝ
  beam_velocity : ℝ

/-- Represents a ship moving towards the lighthouse -/
structure Ship where
  speed : ℝ
  initial_distance : ℝ

/-- Theorem: A ship cannot reach the lighthouse without being illuminated -/
theorem ship_cannot_escape_illumination (L : Lighthouse) (S : Ship) 
  (h1 : S.speed ≤ L.beam_velocity / 8)
  (h2 : S.initial_distance = L.beam_length) : 
  ∃ (t : ℝ), t > 0 ∧ S.initial_distance - S.speed * t > 0 ∧ 
  2 * π * L.beam_length / L.beam_velocity ≥ t :=
sorry

end NUMINAMATH_CALUDE_ship_cannot_escape_illumination_l3366_336695


namespace NUMINAMATH_CALUDE_degree_of_minus_x_cubed_y_is_four_degree_of_minus_x_cubed_y_is_not_three_l3366_336639

/-- Represents a monomial in variables x and y -/
structure Monomial :=
  (coeff : ℤ)
  (x_power : ℕ)
  (y_power : ℕ)

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.x_power + m.y_power

/-- The monomial -x³y -/
def mono : Monomial :=
  { coeff := -1, x_power := 3, y_power := 1 }

/-- Theorem stating that the degree of -x³y is 4 -/
theorem degree_of_minus_x_cubed_y_is_four :
  degree mono = 4 :=
sorry

/-- Theorem stating that the degree of -x³y is not 3 -/
theorem degree_of_minus_x_cubed_y_is_not_three :
  degree mono ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_degree_of_minus_x_cubed_y_is_four_degree_of_minus_x_cubed_y_is_not_three_l3366_336639


namespace NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l3366_336663

/-- The number of days it takes for all pollywogs to disappear from Elmer's pond -/
def days_until_empty (initial_pollywogs : ℕ) (maturation_rate : ℕ) (melvin_catch_rate : ℕ) (melvin_catch_days : ℕ) : ℕ :=
  let first_phase := melvin_catch_days
  let pollywogs_after_first_phase := initial_pollywogs - (maturation_rate + melvin_catch_rate) * first_phase
  let second_phase := pollywogs_after_first_phase / maturation_rate
  first_phase + second_phase

/-- Theorem stating that it takes 44 days for all pollywogs to disappear from Elmer's pond -/
theorem pollywogs_disappear_in_44_days :
  days_until_empty 2400 50 10 20 = 44 := by
  sorry

end NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l3366_336663


namespace NUMINAMATH_CALUDE_lunch_cost_distribution_l3366_336667

theorem lunch_cost_distribution (total_cost : ℕ) 
  (your_cost first_friend_extra second_friend_less third_friend_multiplier : ℕ) :
  total_cost = 100 ∧ 
  first_friend_extra = 15 ∧ 
  second_friend_less = 20 ∧ 
  third_friend_multiplier = 2 →
  ∃ (your_amount : ℕ),
    your_amount = 21 ∧
    your_amount + (your_amount + first_friend_extra) + 
    (your_amount - second_friend_less) + (your_amount * third_friend_multiplier) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_lunch_cost_distribution_l3366_336667


namespace NUMINAMATH_CALUDE_exactly_three_true_l3366_336640

theorem exactly_three_true : 
  (∀ x > 0, x > Real.sin x) ∧ 
  ((∀ x, x - Real.sin x = 0 → x = 0) ↔ (∀ x, x ≠ 0 → x - Real.sin x ≠ 0)) ∧ 
  (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧ 
  ¬(¬(∀ x : ℝ, x - Real.log x > 0) ↔ (∃ x : ℝ, x - Real.log x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_true_l3366_336640


namespace NUMINAMATH_CALUDE_correct_calculation_l3366_336620

theorem correct_calculation (x : ℝ) (h : x - 21 = 52) : 40 * x = 2920 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3366_336620


namespace NUMINAMATH_CALUDE_profit_increase_after_cost_decrease_l3366_336672

theorem profit_increase_after_cost_decrease (x y : ℝ) (a : ℝ) 
  (h1 : y - x = x * (a / 100))  -- Initial profit percentage
  (h2 : y - 0.9 * x = 0.9 * x * ((a + 20) / 100))  -- New profit percentage
  : a = 80 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_after_cost_decrease_l3366_336672


namespace NUMINAMATH_CALUDE_quadratic_root_implies_q_l3366_336600

theorem quadratic_root_implies_q (p q : ℝ) : 
  (∃ (x : ℂ), 3 * x^2 + p * x + q = 0 ∧ x = 3 + 4*I) → q = 75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_q_l3366_336600


namespace NUMINAMATH_CALUDE_min_value_expression_l3366_336633

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c = 1) :
  (8 * a * c^2 + a) / (b * c) + 32 / (a + 1) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3366_336633


namespace NUMINAMATH_CALUDE_range_of_a_l3366_336642

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 3 ≤ 0) ↔ -3 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3366_336642


namespace NUMINAMATH_CALUDE_min_disks_theorem_l3366_336609

/-- The number of labels -/
def n : ℕ := 60

/-- The minimum number of disks with the same label we want to guarantee -/
def k : ℕ := 12

/-- The sum of arithmetic sequence from 1 to m -/
def sum_to (m : ℕ) : ℕ := m * (m + 1) / 2

/-- The total number of disks -/
def total_disks : ℕ := sum_to n

/-- The function to calculate the minimum number of disks to draw -/
def min_disks_to_draw : ℕ := sum_to (k - 1) + (n - (k - 1)) * (k - 1) + 1

/-- The theorem stating the minimum number of disks to draw -/
theorem min_disks_theorem : min_disks_to_draw = 606 := by sorry

end NUMINAMATH_CALUDE_min_disks_theorem_l3366_336609


namespace NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l3366_336641

theorem sin_arccos_eight_seventeenths : 
  Real.sin (Real.arccos (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_eight_seventeenths_l3366_336641


namespace NUMINAMATH_CALUDE_jorges_gifts_count_l3366_336631

/-- The number of gifts Jorge gave at Rosalina's wedding --/
def jorges_gifts (total_gifts emilios_gifts pedros_gifts : ℕ) : ℕ :=
  total_gifts - (emilios_gifts + pedros_gifts)

theorem jorges_gifts_count :
  jorges_gifts 21 11 4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_jorges_gifts_count_l3366_336631


namespace NUMINAMATH_CALUDE_arrangements_count_l3366_336653

/-- The number of arrangements of 6 people with specific conditions -/
def num_arrangements : ℕ :=
  let total_people : ℕ := 6
  let num_teachers : ℕ := 1
  let num_male_students : ℕ := 2
  let num_female_students : ℕ := 3
  let male_students_arrangements : ℕ := 2  -- A_{2}^{2}
  let female_adjacent_pair_selections : ℕ := 3  -- C_{3}^{2}
  let remaining_people_arrangements : ℕ := 12  -- A_{3}^{3}
  male_students_arrangements * female_adjacent_pair_selections * remaining_people_arrangements

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count : num_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3366_336653


namespace NUMINAMATH_CALUDE_lawn_mowing_solution_l3366_336603

/-- Represents the lawn mowing problem --/
def LawnMowingProblem (lawn_length lawn_width swath_width overlap flowerbed_diameter walking_rate : ℝ) : Prop :=
  let effective_width := (swath_width - overlap) / 12  -- Convert to feet
  let flowerbed_area := Real.pi * (flowerbed_diameter / 2) ^ 2
  let mowing_area := lawn_length * lawn_width - flowerbed_area
  let num_strips := lawn_width / effective_width
  let total_distance := num_strips * lawn_length
  let mowing_time := total_distance / walking_rate
  mowing_time = 2

/-- The main theorem stating the solution to the lawn mowing problem --/
theorem lawn_mowing_solution :
  LawnMowingProblem 100 160 30 6 20 4000 := by
  sorry

#check lawn_mowing_solution

end NUMINAMATH_CALUDE_lawn_mowing_solution_l3366_336603


namespace NUMINAMATH_CALUDE_poetry_competition_results_l3366_336607

-- Define the contingency table
def a : ℕ := 6
def b : ℕ := 9
def c : ℕ := 4
def d : ℕ := 1
def n : ℕ := 20

-- Define K^2 calculation
def K_squared : ℚ := (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d) : ℚ)

-- Define probabilities for student C
def prob_buzz : ℚ := 3/5
def prob_correct_buzz : ℚ := 4/5

-- Define the score variable X
inductive Score
| neg_one : Score
| zero : Score
| two : Score

-- Define the probability distribution of X
def prob_X : Score → ℚ
| Score.neg_one => prob_buzz * (1 - prob_correct_buzz)
| Score.zero => 1 - prob_buzz
| Score.two => prob_buzz * prob_correct_buzz

-- Define the expected value of X
def E_X : ℚ := -1 * prob_X Score.neg_one + 0 * prob_X Score.zero + 2 * prob_X Score.two

-- Define the condition for p
def p_condition (p : ℚ) : Prop := 
  |3 * p + 2.52 - (4 * p + 1.68)| ≤ 1/10 ∧ 0 < p ∧ p < 1

theorem poetry_competition_results :
  K_squared < 3841/1000 ∧
  prob_X Score.neg_one = 12/100 ∧
  prob_X Score.zero = 2/5 ∧
  prob_X Score.two = 24/50 ∧
  E_X = 21/25 ∧
  ∀ p, p_condition p ↔ 37/50 ≤ p ∧ p ≤ 47/50 :=
sorry

end NUMINAMATH_CALUDE_poetry_competition_results_l3366_336607


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l3366_336683

-- Statement 1
theorem inequality_one (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 2) - Real.sqrt (a + 6) > Real.sqrt a - Real.sqrt (a + 4) := by
  sorry

-- Statement 2
theorem inequality_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l3366_336683


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3366_336691

theorem area_between_concentric_circles :
  ∀ (r : ℝ),
  r > 0 →
  3 * r - r = 3 →
  π * (3 * r)^2 - π * r^2 = 18 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3366_336691


namespace NUMINAMATH_CALUDE_even_function_implies_f_2_eq_3_l3366_336606

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_f_2_eq_3 :
  (∀ x : ℝ, f a x = f a (-x)) → f a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_f_2_eq_3_l3366_336606


namespace NUMINAMATH_CALUDE_solution_set_f_geq_8_range_of_a_when_solution_exists_l3366_336694

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the solution set of f(x) ≥ 8
theorem solution_set_f_geq_8 :
  {x : ℝ | f x ≥ 8} = {x : ℝ | x ≤ -5 ∨ x ≥ 3} := by sorry

-- Theorem for the range of a when the solution set of f(x) < a^2 - 3a is not empty
theorem range_of_a_when_solution_exists (a : ℝ) :
  (∃ x, f x < a^2 - 3*a) → (a < -1 ∨ a > 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_8_range_of_a_when_solution_exists_l3366_336694


namespace NUMINAMATH_CALUDE_monic_quadratic_unique_l3366_336658

/-- A monic quadratic polynomial is a polynomial of the form x^2 + bx + c -/
def MonicQuadraticPolynomial (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem monic_quadratic_unique (b c : ℝ) :
  let g := MonicQuadraticPolynomial b c
  g 0 = 8 ∧ g 1 = 14 → b = 5 ∧ c = 8 := by sorry

end NUMINAMATH_CALUDE_monic_quadratic_unique_l3366_336658


namespace NUMINAMATH_CALUDE_intersection_point_l3366_336625

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 1) / 7 = (y - 2) / 1 ∧ (y - 2) / 1 = (z - 6) / (-1)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  4 * x + y - 6 * z - 5 = 0

/-- The theorem stating that (8, 3, 5) is the unique point of intersection -/
theorem intersection_point :
  ∃! (x y z : ℝ), line x y z ∧ plane x y z ∧ x = 8 ∧ y = 3 ∧ z = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3366_336625


namespace NUMINAMATH_CALUDE_orchard_theorem_l3366_336651

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji = (o.total * 3) / 4 ∧
  o.pure_gala = 42 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

/-- The theorem stating that under the given conditions, 
    the number of pure Fuji plus cross-pollinated trees is 238 -/
theorem orchard_theorem (o : Orchard) 
  (h : orchard_conditions o) : o.pure_fuji + o.cross_pollinated = 238 := by
  sorry

end NUMINAMATH_CALUDE_orchard_theorem_l3366_336651


namespace NUMINAMATH_CALUDE_softball_team_size_l3366_336697

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 6 →
  (men : ℚ) / (women : ℚ) = 6 / 10 →
  men + women = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l3366_336697


namespace NUMINAMATH_CALUDE_society_member_sum_or_double_l3366_336659

theorem society_member_sum_or_double {n : ℕ} (hn : n = 1978) :
  ∀ (f : Fin n → Fin 6),
  ∃ (i : Fin 6) (a b c : Fin n),
    f a = i ∧ f b = i ∧ f c = i ∧
    (a.val + 1 = b.val + c.val + 2 ∨ a.val + 1 = 2 * (b.val + 1)) := by
  sorry


end NUMINAMATH_CALUDE_society_member_sum_or_double_l3366_336659


namespace NUMINAMATH_CALUDE_football_players_count_l3366_336654

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 40)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 11) :
  ∃ football : ℕ, football = 26 ∧ 
    football + tennis - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_football_players_count_l3366_336654


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3366_336624

/-- Given a quadrilateral EFGH with extended sides, prove the reconstruction formula for point E -/
theorem quadrilateral_reconstruction 
  (E F G H E' F' G' H' : ℝ × ℝ) 
  (h1 : E' - F = 2 * (E - F))
  (h2 : F' - G = 2 * (F - G))
  (h3 : G' - H = 2 * (G - H))
  (h4 : H' - E = 2 * (H - E)) :
  E = (1/79 : ℝ) • E' + (26/79 : ℝ) • F' + (26/79 : ℝ) • G' + (52/79 : ℝ) • H' := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3366_336624


namespace NUMINAMATH_CALUDE_firewood_sacks_filled_l3366_336698

/-- Calculates the number of sacks filled with firewood -/
def sacks_filled (wood_per_sack : ℕ) (total_wood : ℕ) : ℕ :=
  total_wood / wood_per_sack

/-- Theorem stating that the number of sacks filled is 4 -/
theorem firewood_sacks_filled :
  let wood_per_sack : ℕ := 20
  let total_wood : ℕ := 80
  sacks_filled wood_per_sack total_wood = 4 := by
  sorry

end NUMINAMATH_CALUDE_firewood_sacks_filled_l3366_336698


namespace NUMINAMATH_CALUDE_sum_of_2_and_odd_prime_last_digit_l3366_336614

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def last_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_2_and_odd_prime_last_digit (p : ℕ) 
  (h_prime : is_prime p) 
  (h_odd : p % 2 = 1) 
  (h_greater_7 : p > 7) 
  (h_sum_not_single_digit : p + 2 ≥ 10) : 
  last_digit (p + 2) = 1 ∨ last_digit (p + 2) = 3 ∨ last_digit (p + 2) = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_2_and_odd_prime_last_digit_l3366_336614


namespace NUMINAMATH_CALUDE_triangle_area_is_eight_l3366_336622

-- Define the slopes and intersection point
def slope1 : ℝ := -1
def slope2 : ℝ := 3
def intersection : ℝ × ℝ := (1, 3)

-- Define the lines
def line1 (x : ℝ) : ℝ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℝ) : ℝ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℝ) : Prop := x - y = 2

-- Define the points of the triangle
def pointA : ℝ × ℝ := intersection
def pointB : ℝ × ℝ := (-1, -3)  -- Intersection of line2 and line3
def pointC : ℝ × ℝ := (3, 1)    -- Intersection of line1 and line3

-- Theorem statement
theorem triangle_area_is_eight :
  let area := (1/2) * abs (
    pointA.1 * (pointB.2 - pointC.2) +
    pointB.1 * (pointC.2 - pointA.2) +
    pointC.1 * (pointA.2 - pointB.2)
  )
  area = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_eight_l3366_336622


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3366_336605

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 7 * 2 / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3366_336605


namespace NUMINAMATH_CALUDE_inequality_solution_difference_l3366_336687

theorem inequality_solution_difference : ∃ (M m : ℝ),
  (∀ x, 4 * x * (x - 5) ≤ 375 → x ≤ M ∧ m ≤ x) ∧
  (4 * M * (M - 5) ≤ 375) ∧
  (4 * m * (m - 5) ≤ 375) ∧
  (M - m = 20) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_difference_l3366_336687


namespace NUMINAMATH_CALUDE_aria_cookie_expense_is_2356_l3366_336699

/-- The amount Aria spent on cookies in March -/
def aria_cookie_expense : ℕ :=
  let cookies_per_day : ℕ := 4
  let cost_per_cookie : ℕ := 19
  let days_in_march : ℕ := 31
  cookies_per_day * cost_per_cookie * days_in_march

/-- Theorem stating that Aria spent 2356 dollars on cookies in March -/
theorem aria_cookie_expense_is_2356 : aria_cookie_expense = 2356 := by
  sorry

end NUMINAMATH_CALUDE_aria_cookie_expense_is_2356_l3366_336699


namespace NUMINAMATH_CALUDE_y_derivative_l3366_336612

noncomputable def y (x : ℝ) : ℝ :=
  (2 * x^2 - x + 1/2) * Real.arctan ((x^2 - 1) / (x * Real.sqrt 3)) - 
  x^3 / (2 * Real.sqrt 3) - (Real.sqrt 3 / 2) * x

theorem y_derivative (x : ℝ) (hx : x ≠ 0) : 
  deriv y x = (4 * x - 1) * Real.arctan ((x^2 - 1) / (x * Real.sqrt 3)) + 
  (Real.sqrt 3 * (x^2 + 1) * (3 * x^2 - 2 * x - x^4)) / (2 * (x^4 + x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3366_336612


namespace NUMINAMATH_CALUDE_yoongi_has_fewest_l3366_336648

/-- Represents the number of apples each person has -/
structure AppleCount where
  yoongi : Nat
  jungkook : Nat
  yuna : Nat

/-- Defines the given apple counts -/
def given_apples : AppleCount :=
  { yoongi := 4
  , jungkook := 9
  , yuna := 5 }

/-- Theorem: Yoongi has the fewest apples -/
theorem yoongi_has_fewest (a : AppleCount := given_apples) :
  a.yoongi < a.jungkook ∧ a.yoongi < a.yuna :=
by sorry

end NUMINAMATH_CALUDE_yoongi_has_fewest_l3366_336648


namespace NUMINAMATH_CALUDE_strawberry_basket_price_is_9_l3366_336673

/-- Represents the harvest and sales information for Nathan's garden --/
structure GardenSales where
  strawberry_plants : ℕ
  tomato_plants : ℕ
  strawberries_per_plant : ℕ
  tomatoes_per_plant : ℕ
  fruits_per_basket : ℕ
  tomato_basket_price : ℕ
  total_revenue : ℕ

/-- Calculates the price of a basket of strawberries --/
def strawberry_basket_price (g : GardenSales) : ℚ :=
  let total_strawberries := g.strawberry_plants * g.strawberries_per_plant
  let total_tomatoes := g.tomato_plants * g.tomatoes_per_plant
  let strawberry_baskets := total_strawberries / g.fruits_per_basket
  let tomato_baskets := total_tomatoes / g.fruits_per_basket
  let tomato_revenue := tomato_baskets * g.tomato_basket_price
  let strawberry_revenue := g.total_revenue - tomato_revenue
  strawberry_revenue / strawberry_baskets

/-- Theorem stating that the price of a basket of strawberries is 9 --/
theorem strawberry_basket_price_is_9 (g : GardenSales) 
  (h1 : g.strawberry_plants = 5)
  (h2 : g.tomato_plants = 7)
  (h3 : g.strawberries_per_plant = 14)
  (h4 : g.tomatoes_per_plant = 16)
  (h5 : g.fruits_per_basket = 7)
  (h6 : g.tomato_basket_price = 6)
  (h7 : g.total_revenue = 186) :
  strawberry_basket_price g = 9 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_basket_price_is_9_l3366_336673


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l3366_336613

/-- Two lines with the same non-zero y-intercept, one with slope 8 and x-intercept (s, 0),
    the other with slope 4 and x-intercept (t, 0), have s/t = 1/2 -/
theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) : 
  (0 = 8 * s + b) → (0 = 4 * t + b) → s / t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l3366_336613


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l3366_336630

/-- Given a cubic polynomial with two equal integer roots, prove |ab| = 5832 -/
theorem cubic_polynomial_roots (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) ∧ 
   (r ≠ s)) → 
  |a * b| = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l3366_336630


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3366_336608

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem quadratic_function_properties :
  (∃ x, f x = 1 ∧ ∀ y, f y ≥ f x) ∧
  f 0 = 3 ∧ f 2 = 3 ∧
  (∀ a : ℝ, (0 < a ∧ a < 1/2) ↔ 
    ¬(∀ x y : ℝ, 2*a ≤ x ∧ x < y ∧ y ≤ a+1 → f x < f y ∨ f x > f y)) ∧
  (∀ m : ℝ, m < 1 ↔ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x > 2*x + 2*m + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3366_336608


namespace NUMINAMATH_CALUDE_exist_three_points_in_small_circle_l3366_336617

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a set of points within a unit square -/
def PointsInUnitSquare (points : Set Point) : Prop :=
  ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1

/-- Checks if three points can be enclosed by a circle with radius 1/7 -/
def CanBeEnclosedByCircle (p1 p2 p3 : Point) : Prop :=
  ∃ (center : Point), (center.x - p1.x)^2 + (center.y - p1.y)^2 ≤ (1/7)^2 ∧
                      (center.x - p2.x)^2 + (center.y - p2.y)^2 ≤ (1/7)^2 ∧
                      (center.x - p3.x)^2 + (center.y - p3.y)^2 ≤ (1/7)^2

/-- Main theorem: In any set of 51 points within a unit square, 
    there exist three points that can be enclosed by a circle with radius 1/7 -/
theorem exist_three_points_in_small_circle 
  (points : Set Point) 
  (h1 : PointsInUnitSquare points) 
  (h2 : Fintype points) 
  (h3 : Fintype.card points = 51) :
  ∃ p1 p2 p3 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
               p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
               CanBeEnclosedByCircle p1 p2 p3 :=
by sorry

end NUMINAMATH_CALUDE_exist_three_points_in_small_circle_l3366_336617


namespace NUMINAMATH_CALUDE_complex_multiplication_l3366_336618

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3366_336618


namespace NUMINAMATH_CALUDE_clarence_oranges_l3366_336652

/-- The total number of oranges Clarence has -/
def total_oranges (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Clarence has 8 oranges in total -/
theorem clarence_oranges :
  total_oranges 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_l3366_336652


namespace NUMINAMATH_CALUDE_no_valid_chess_sequence_l3366_336602

/-- Represents a sequence of moves on a 6x6 chessboard -/
def ChessSequence := Fin 36 → Fin 36

/-- Checks if the difference between consecutive terms alternates between 1 and 2 -/
def validMoves (seq : ChessSequence) : Prop :=
  ∀ i : Fin 35, (i.val % 2 = 0 → |seq (i + 1) - seq i| = 1) ∧
                (i.val % 2 = 1 → |seq (i + 1) - seq i| = 2)

/-- Checks if all elements in the sequence are distinct -/
def allDistinct (seq : ChessSequence) : Prop :=
  ∀ i j : Fin 36, i ≠ j → seq i ≠ seq j

/-- The main theorem: no valid chess sequence exists -/
theorem no_valid_chess_sequence :
  ¬∃ (seq : ChessSequence), validMoves seq ∧ allDistinct seq :=
sorry

end NUMINAMATH_CALUDE_no_valid_chess_sequence_l3366_336602


namespace NUMINAMATH_CALUDE_sum_of_coordinates_for_symmetric_points_l3366_336686

-- Define the points P and Q
def P (x : ℝ) : ℝ × ℝ := (x, -3)
def Q (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the property of being symmetric with respect to the origin
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem sum_of_coordinates_for_symmetric_points (x y : ℝ) :
  symmetric_about_origin (P x) (Q y) → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_for_symmetric_points_l3366_336686


namespace NUMINAMATH_CALUDE_total_marbles_l3366_336671

theorem total_marbles (blue red orange : ℕ) : 
  blue = red + orange → -- Half of the marbles are blue
  red = 6 →             -- There are 6 red marbles
  orange = 6 →          -- There are 6 orange marbles
  blue + red + orange = 24 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_l3366_336671


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_solution_satisfies_conditions_l3366_336678

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x = 1 ∧ m * y^2 + 2 * y = 1) ↔ 
  (m > -1 ∧ m ≠ 0) :=
by sorry

theorem solution_satisfies_conditions : 
  1 > -1 ∧ 1 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_solution_satisfies_conditions_l3366_336678


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l3366_336616

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral (P : Type*) [MetricSpace P] :=
  (A B C D : P)
  (cyclic : ∃ (center : P) (radius : ℝ), dist center A = radius ∧ dist center B = radius ∧ dist center C = radius ∧ dist center D = radius)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)

/-- The theorem states that in a cyclic quadrilateral ABCD where AB is the longest side,
    the sum of AB and BD is greater than the sum of AC and CD. -/
theorem cyclic_quadrilateral_inequality {P : Type*} [MetricSpace P] (Q : CyclicQuadrilateral P) :
  (∀ X Y : P, dist Q.A Q.B ≥ dist X Y) →
  dist Q.A Q.B + dist Q.B Q.D > dist Q.A Q.C + dist Q.C Q.D :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l3366_336616


namespace NUMINAMATH_CALUDE_angle_of_inclination_30_degrees_l3366_336629

theorem angle_of_inclination_30_degrees (x y : ℝ) :
  2 * x - 2 * Real.sqrt 3 * y + 1 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_30_degrees_l3366_336629


namespace NUMINAMATH_CALUDE_knockout_tournament_matches_l3366_336643

/-- The number of matches in a knockout tournament -/
def num_matches (n : ℕ) : ℕ := n - 1

/-- A knockout tournament with 64 players -/
def tournament_size : ℕ := 64

theorem knockout_tournament_matches :
  num_matches tournament_size = 63 := by
  sorry

end NUMINAMATH_CALUDE_knockout_tournament_matches_l3366_336643


namespace NUMINAMATH_CALUDE_kannon_bananas_l3366_336628

/-- Proves that Kannon had 1 banana last night given the conditions of the problem -/
theorem kannon_bananas : 
  ∀ (bananas_last_night : ℕ),
    (3 + bananas_last_night + 4) +  -- fruits last night
    ((3 + 4) + 10 * bananas_last_night + 2 * (3 + 4)) = 39 → -- fruits today
    bananas_last_night = 1 := by
  sorry

end NUMINAMATH_CALUDE_kannon_bananas_l3366_336628


namespace NUMINAMATH_CALUDE_f_of_3_equals_13_l3366_336690

theorem f_of_3_equals_13 (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = 2 * x + 5) : f 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_13_l3366_336690


namespace NUMINAMATH_CALUDE_age_when_billy_was_born_l3366_336680

/-- Proves the age when Billy was born given the current ages -/
theorem age_when_billy_was_born
  (my_current_age billy_current_age : ℕ)
  (h1 : my_current_age = 4 * billy_current_age)
  (h2 : billy_current_age = 4)
  : my_current_age - billy_current_age = my_current_age - billy_current_age :=
by sorry

end NUMINAMATH_CALUDE_age_when_billy_was_born_l3366_336680


namespace NUMINAMATH_CALUDE_subsidy_and_job_creation_l3366_336604

/-- Data for SZ province's "home appliances to the countryside" program in 2008 -/
structure ProgramData2008 where
  new_shops : ℕ
  jobs_created : ℕ
  units_sold : ℕ
  sales_amount : ℝ
  consumption_increase : ℝ
  subsidy_rate : ℝ

/-- Data for the program from 2008 to 2010 -/
structure ProgramData2008To2010 where
  total_jobs : ℕ
  increase_2010_vs_2009 : ℝ
  jobs_increase_2010_vs_2009 : ℝ

/-- Theorem about the subsidy funds needed in 2008 and job creation rate -/
theorem subsidy_and_job_creation 
  (data_2008 : ProgramData2008)
  (data_2008_to_2010 : ProgramData2008To2010)
  (h1 : data_2008.new_shops = 8000)
  (h2 : data_2008.jobs_created = 75000)
  (h3 : data_2008.units_sold = 1130000)
  (h4 : data_2008.sales_amount = 1.6 * 10^9)
  (h5 : data_2008.consumption_increase = 1.7)
  (h6 : data_2008.subsidy_rate = 0.13)
  (h7 : data_2008_to_2010.total_jobs = 247000)
  (h8 : data_2008_to_2010.increase_2010_vs_2009 = 0.5)
  (h9 : data_2008_to_2010.jobs_increase_2010_vs_2009 = 10/81) :
  ∃ (subsidy_funds : ℝ) (jobs_per_point : ℝ),
    subsidy_funds = 2.08 * 10^9 ∧ 
    jobs_per_point = 20000 := by
  sorry

end NUMINAMATH_CALUDE_subsidy_and_job_creation_l3366_336604


namespace NUMINAMATH_CALUDE_consumption_decrease_l3366_336635

theorem consumption_decrease (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  let new_price := 1.4 * original_price
  let new_budget := 1.12 * (original_price * original_quantity)
  let new_quantity := new_budget / new_price
  new_quantity / original_quantity = 0.8 := by sorry

end NUMINAMATH_CALUDE_consumption_decrease_l3366_336635


namespace NUMINAMATH_CALUDE_tangent_to_ln_curve_l3366_336655

theorem tangent_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = (Real.log x) / x) →
  (k * 0 = Real.log 0) →
  k = 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_to_ln_curve_l3366_336655


namespace NUMINAMATH_CALUDE_product_derivative_at_zero_l3366_336669

/-- Given differentiable real functions f, g, h, prove that (fgh)'(0) = 16 -/
theorem product_derivative_at_zero
  (f g h : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (hh : Differentiable ℝ h)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 2)
  (hh0 : h 0 = 3)
  (hgh : deriv (g * h) 0 = 4)
  (hhf : deriv (h * f) 0 = 5)
  (hfg : deriv (f * g) 0 = 6) :
  deriv (f * g * h) 0 = 16 := by
sorry

end NUMINAMATH_CALUDE_product_derivative_at_zero_l3366_336669


namespace NUMINAMATH_CALUDE_zero_point_of_f_l3366_336656

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem zero_point_of_f : 
  ∃ (x : ℝ), f x = 0 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_zero_point_of_f_l3366_336656


namespace NUMINAMATH_CALUDE_tom_rare_cards_l3366_336676

/-- The number of rare cards in Tom's deck -/
def rare_cards : ℕ := 19

/-- The number of uncommon cards in Tom's deck -/
def uncommon_cards : ℕ := 11

/-- The number of common cards in Tom's deck -/
def common_cards : ℕ := 30

/-- The cost of a rare card in dollars -/
def rare_cost : ℚ := 1

/-- The cost of an uncommon card in dollars -/
def uncommon_cost : ℚ := 1/2

/-- The cost of a common card in dollars -/
def common_cost : ℚ := 1/4

/-- The total cost of Tom's deck in dollars -/
def total_cost : ℚ := 32

theorem tom_rare_cards : 
  rare_cards * rare_cost + 
  uncommon_cards * uncommon_cost + 
  common_cards * common_cost = total_cost := by sorry

end NUMINAMATH_CALUDE_tom_rare_cards_l3366_336676


namespace NUMINAMATH_CALUDE_inequality_proof_l3366_336688

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3366_336688


namespace NUMINAMATH_CALUDE_total_money_is_84_l3366_336679

/-- Represents the money redistribution process among three people. -/
def redistribute (j a t : ℚ) : Prop :=
  ∃ (j₁ a₁ t₁ j₂ a₂ t₂ j₃ a₃ t₃ : ℚ),
    -- Step 1: Jan's redistribution
    j₁ + a₁ + t₁ = j + a + t ∧
    a₁ = 2 * a ∧
    t₁ = 2 * t ∧
    -- Step 2: Toy's redistribution
    j₂ + a₂ + t₂ = j₁ + a₁ + t₁ ∧
    j₂ = 2 * j₁ ∧
    a₂ = 2 * a₁ ∧
    -- Step 3: Amy's redistribution
    j₃ + a₃ + t₃ = j₂ + a₂ + t₂ ∧
    j₃ = 2 * j₂ ∧
    t₃ = 2 * t₂

/-- The main theorem stating the total amount of money. -/
theorem total_money_is_84 :
  ∀ j a t : ℚ, t = 48 → redistribute j a t → j + a + t = 84 :=
by sorry

end NUMINAMATH_CALUDE_total_money_is_84_l3366_336679


namespace NUMINAMATH_CALUDE_equation_solutions_l3366_336621

/-- Given two equations about x and k -/
theorem equation_solutions (x k : ℚ) : 
  (3 * (2 * x - 1) = k + 2 * x) →
  ((x - k) / 2 = x + 2 * k) →
  (
    /- Part 1 -/
    (x = 4 → (x - k) / 2 = x + 2 * k → x = -65) ∧
    /- Part 2 -/
    (∃ x, 3 * (2 * x - 1) = k + 2 * x ∧ (x - k) / 2 = x + 2 * k) → k = -1/7
  ) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3366_336621


namespace NUMINAMATH_CALUDE_ellipse_equation_and_slope_l3366_336662

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- Theorem about the equation of the ellipse and the slope of line l -/
theorem ellipse_equation_and_slope (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.eccentricity = Real.sqrt 3 / 2)
  (h4 : e.passes_through = (Real.sqrt 2, Real.sqrt 2 / 2)) :
  (∃ (x y : ℝ), x^2 / 4 + y^2 = 1) ∧ 
  (∃ (k : ℝ), k = 1/2 ∨ k = -1/2) := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_slope_l3366_336662


namespace NUMINAMATH_CALUDE_max_sum_n_l3366_336637

/-- An arithmetic sequence with first term 11 and common difference -2 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  11 - 2 * (n - 1)

/-- The sum of the first n terms of the arithmetic sequence -/
def sumOfTerms (n : ℕ) : ℤ :=
  n * (arithmeticSequence 1 + arithmeticSequence n) / 2

/-- The value of n that maximizes the sum of the first n terms -/
theorem max_sum_n : ∃ (n : ℕ), n = 6 ∧ 
  ∀ (m : ℕ), sumOfTerms m ≤ sumOfTerms n :=
sorry

end NUMINAMATH_CALUDE_max_sum_n_l3366_336637


namespace NUMINAMATH_CALUDE_ages_sum_l3366_336674

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3366_336674


namespace NUMINAMATH_CALUDE_three_bus_interval_l3366_336646

/-- Given a circular bus route with two buses operating at an interval of 21 minutes,
    this theorem proves that when three buses operate on the same route at the same speed,
    the new interval between consecutive buses is 14 minutes. -/
theorem three_bus_interval (interval_two_buses : ℕ) (h : interval_two_buses = 21) :
  let total_time := 2 * interval_two_buses
  let interval_three_buses := total_time / 3
  interval_three_buses = 14 := by
sorry

end NUMINAMATH_CALUDE_three_bus_interval_l3366_336646


namespace NUMINAMATH_CALUDE_total_books_read_proof_l3366_336675

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  48 * c * s

/-- Theorem stating that the total number of books read by the entire student body in one year
    is equal to 48 * c * s, where c is the number of classes and s is the number of students per class -/
theorem total_books_read_proof (c s : ℕ) :
  total_books_read c s = 48 * c * s :=
by sorry

end NUMINAMATH_CALUDE_total_books_read_proof_l3366_336675


namespace NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l3366_336664

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (h : ℝ), (x - h)^2 + (y - (3*h - 5))^2 = 1 ∧ (3 - h)^2 + (3 - (3*h - 5))^2 = 1 ∧ (2 - h)^2 + (4 - (3*h - 5))^2 = 1

-- Define the circle with diameter PQ
def circle_PQ (m x y : ℝ) : Prop := x^2 + y^2 = m^2

theorem circle_equation_and_intersection_range :
  ∃ (h : ℝ), 
    (∀ x y : ℝ, circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 1) ∧
    (∀ m : ℝ, m > 0 → 
      (∃ x y : ℝ, circle_C x y ∧ circle_PQ m x y) ↔ 
      (4 ≤ m ∧ m ≤ 6)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l3366_336664


namespace NUMINAMATH_CALUDE_total_shells_is_195_l3366_336634

/-- The total number of conch shells owned by David, Mia, Ava, and Alice -/
def total_shells (david_shells : ℕ) : ℕ :=
  let mia_shells := 4 * david_shells
  let ava_shells := mia_shells + 20
  let alice_shells := ava_shells / 2
  david_shells + mia_shells + ava_shells + alice_shells

/-- Theorem stating that the total number of shells is 195 when David has 15 shells -/
theorem total_shells_is_195 : total_shells 15 = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_195_l3366_336634


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l3366_336666

theorem infinitely_many_primes_of_form (m n : ℤ) : 
  ∃ (S : Set Nat), Set.Infinite S ∧ ∀ p ∈ S, Prime p ∧ ∃ m n : ℤ, p = m^2 + m*n + n^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l3366_336666


namespace NUMINAMATH_CALUDE_final_shape_independent_of_initial_fold_l3366_336647

/-- Represents a square sheet of paper -/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents the folded state of the paper -/
inductive FoldedState
  | Unfolded
  | FoldedOnce
  | FoldedTwice
  | FoldedThrice

/-- Represents the initial fold direction -/
inductive FoldDirection
  | MN
  | AB

/-- Represents the final shape after unfolding -/
structure FinalShape :=
  (shape : Set (ℝ × ℝ))

/-- Function to fold the paper -/
def fold (s : Square) (state : FoldedState) : FoldedState :=
  match state with
  | FoldedState.Unfolded => FoldedState.FoldedOnce
  | FoldedState.FoldedOnce => FoldedState.FoldedTwice
  | FoldedState.FoldedTwice => FoldedState.FoldedThrice
  | FoldedState.FoldedThrice => FoldedState.FoldedThrice

/-- Function to cut and unfold the paper -/
def cutAndUnfold (s : Square) (state : FoldedState) (dir : FoldDirection) : FinalShape :=
  sorry

/-- Theorem stating that the final shape is independent of initial fold direction -/
theorem final_shape_independent_of_initial_fold (s : Square) :
  ∀ (dir1 dir2 : FoldDirection),
    cutAndUnfold s (fold s (fold s (fold s FoldedState.Unfolded))) dir1 =
    cutAndUnfold s (fold s (fold s (fold s FoldedState.Unfolded))) dir2 :=
  sorry

end NUMINAMATH_CALUDE_final_shape_independent_of_initial_fold_l3366_336647


namespace NUMINAMATH_CALUDE_percent_decrease_l3366_336644

theorem percent_decrease (original_price sale_price : ℝ) (h1 : original_price = 100) (h2 : sale_price = 50) :
  (original_price - sale_price) / original_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_l3366_336644


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3366_336636

theorem perfect_square_condition (x y : ℕ) :
  (∃ (n : ℕ), (x + y)^2 + 3*x + y + 1 = n^2) ↔ x = y :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3366_336636


namespace NUMINAMATH_CALUDE_star_polygon_n_is_24_l3366_336696

/-- Represents an n-pointed star polygon -/
structure StarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  angles_congruent : True  -- Represents that A₁, A₂, ..., Aₙ are congruent and B₁, B₂, ..., Bₙ are congruent
  angle_difference : angle_B = angle_A + 15

/-- Theorem stating that in a star polygon with the given properties, n = 24 -/
theorem star_polygon_n_is_24 (star : StarPolygon) : star.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_n_is_24_l3366_336696


namespace NUMINAMATH_CALUDE_popcorn_probability_l3366_336627

theorem popcorn_probability : 
  let white_ratio : ℚ := 3/4
  let yellow_ratio : ℚ := 1/4
  let white_pop_prob : ℚ := 2/3
  let yellow_pop_prob : ℚ := 3/4
  let fizz_prob : ℚ := 1/4
  
  let white_pop_fizz : ℚ := white_ratio * white_pop_prob * fizz_prob
  let yellow_pop_fizz : ℚ := yellow_ratio * yellow_pop_prob * fizz_prob
  let total_pop_fizz : ℚ := white_pop_fizz + yellow_pop_fizz
  
  white_pop_fizz / total_pop_fizz = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_probability_l3366_336627


namespace NUMINAMATH_CALUDE_cost_decrease_l3366_336682

theorem cost_decrease (original_cost : ℝ) (decrease_percentage : ℝ) (new_cost : ℝ) : 
  original_cost = 200 →
  decrease_percentage = 50 →
  new_cost = original_cost * (1 - decrease_percentage / 100) →
  new_cost = 100 := by
sorry

end NUMINAMATH_CALUDE_cost_decrease_l3366_336682


namespace NUMINAMATH_CALUDE_two_number_difference_l3366_336601

theorem two_number_difference (a b : ℕ) : 
  a + b = 20460 → 
  b % 12 = 0 → 
  a = b / 10 → 
  b - a = 17314 := by sorry

end NUMINAMATH_CALUDE_two_number_difference_l3366_336601


namespace NUMINAMATH_CALUDE_point_on_curve_l3366_336610

noncomputable def tangent_slope (x : ℝ) : ℝ := 1 + Real.log x

theorem point_on_curve (x y : ℝ) (h : y = x * Real.log x) :
  tangent_slope x = 2 → x = Real.exp 1 ∧ y = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l3366_336610


namespace NUMINAMATH_CALUDE_lcm_six_fifteen_l3366_336632

theorem lcm_six_fifteen : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_six_fifteen_l3366_336632


namespace NUMINAMATH_CALUDE_limit_equals_third_derivative_at_one_l3366_336670

-- Define a real-valued function f that is differentiable on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- State the theorem
theorem limit_equals_third_derivative_at_one :
  (∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ) \ {1},
    |((f (1 + (x - 1)) - f 1) / (3 * (x - 1))) - (1/3 * deriv f 1)| < ε) :=
sorry

end NUMINAMATH_CALUDE_limit_equals_third_derivative_at_one_l3366_336670


namespace NUMINAMATH_CALUDE_red_ants_count_l3366_336660

theorem red_ants_count (total : ℕ) (black : ℕ) (red : ℕ) : 
  total = 900 → black = 487 → total = red + black → red = 413 := by
  sorry

end NUMINAMATH_CALUDE_red_ants_count_l3366_336660


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l3366_336649

theorem museum_entrance_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : 
  num_students = 20 → num_teachers = 3 → ticket_price = 5 → 
  (num_students + num_teachers) * ticket_price = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l3366_336649


namespace NUMINAMATH_CALUDE_unique_pie_purchase_l3366_336657

/-- Represents the number of pies bought by each classmate -/
structure PiePurchase where
  kostya : Nat
  volodya : Nat
  tolya : Nat

/-- Checks if a PiePurchase satisfies all the conditions of the problem -/
def isValidPurchase (p : PiePurchase) : Prop :=
  p.kostya + p.volodya + p.tolya = 13 ∧
  p.tolya = 2 * p.kostya ∧
  p.kostya < p.volodya ∧
  p.volodya < p.tolya

/-- The theorem stating that there is only one valid solution to the problem -/
theorem unique_pie_purchase :
  ∃! p : PiePurchase, isValidPurchase p ∧ p = ⟨3, 4, 6⟩ := by
  sorry

end NUMINAMATH_CALUDE_unique_pie_purchase_l3366_336657


namespace NUMINAMATH_CALUDE_no_obtuse_equilateral_triangle_l3366_336689

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∧ t.angle2 = t.angle3

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: An obtuse equilateral triangle cannot exist
theorem no_obtuse_equilateral_triangle :
  ¬ ∃ (t : Triangle), t.isEquilateral ∧ t.isObtuse ∧ t.angle1 + t.angle2 + t.angle3 = 180 :=
by sorry

end NUMINAMATH_CALUDE_no_obtuse_equilateral_triangle_l3366_336689


namespace NUMINAMATH_CALUDE_evenness_condition_l3366_336619

/-- Given a real number ω, prove that there exists a real number a such that 
    f(x+a) is an even function, where f(x) = (x-6)^2 * sin(ωx), 
    if and only if ω = π/4 -/
theorem evenness_condition (ω : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, (((x + a - 6)^2 * Real.sin (ω * (x + a))) = 
                      (((-x) + a - 6)^2 * Real.sin (ω * ((-x) + a)))))
  ↔ 
  ω = π / 4 := by
sorry

end NUMINAMATH_CALUDE_evenness_condition_l3366_336619


namespace NUMINAMATH_CALUDE_composite_n_pow_2016_plus_4_l3366_336693

theorem composite_n_pow_2016_plus_4 (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^2016 + 4 = a * b :=
by
  sorry

end NUMINAMATH_CALUDE_composite_n_pow_2016_plus_4_l3366_336693


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l3366_336677

theorem washing_machine_capacity (pounds_per_machine : ℕ) (num_machines : ℕ) 
  (h1 : pounds_per_machine = 28) 
  (h2 : num_machines = 8) : 
  pounds_per_machine * num_machines = 224 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l3366_336677


namespace NUMINAMATH_CALUDE_even_function_decreasing_interval_l3366_336668

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the decreasing interval
def decreasingInterval (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x ≤ y → f y ≤ f x}

-- State the theorem
theorem even_function_decreasing_interval :
  ∀ m : ℝ, isEven (f m) → decreasingInterval (f m) = Set.Ici 0 :=
sorry

end NUMINAMATH_CALUDE_even_function_decreasing_interval_l3366_336668


namespace NUMINAMATH_CALUDE_complex_number_problem_l3366_336661

/-- Given a complex number z = b - 2i where b is real, and z / (2 - i) is real,
    prove that z = 4 - 2i and for (z + ai)² to be in the fourth quadrant, -2 < a < 2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) (h1 : z = b - 2*I) 
  (h2 : ∃ (r : ℝ), z / (2 - I) = r) :
  z = 4 - 2*I ∧ 
  ∀ (a : ℝ), (z + a*I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im < 0} → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3366_336661


namespace NUMINAMATH_CALUDE_sevenDigitIntegers_eq_630_l3366_336626

/-- The number of different positive, seven-digit integers that can be formed
    using the digits 2, 2, 3, 5, 5, 9, and 9 -/
def sevenDigitIntegers : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, seven-digit integers
    that can be formed using the digits 2, 2, 3, 5, 5, 9, and 9 is 630 -/
theorem sevenDigitIntegers_eq_630 : sevenDigitIntegers = 630 := by
  sorry

end NUMINAMATH_CALUDE_sevenDigitIntegers_eq_630_l3366_336626


namespace NUMINAMATH_CALUDE_pipes_remaining_proof_l3366_336684

/-- The number of pipes in a triangular pyramid with n layers -/
def triangular_pyramid (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of pipes available -/
def total_pipes : ℕ := 200

/-- The maximum number of complete layers in the pyramid -/
def max_layers : ℕ := 19

/-- The number of pipes left over -/
def pipes_left_over : ℕ := total_pipes - triangular_pyramid max_layers

theorem pipes_remaining_proof :
  pipes_left_over = 10 :=
sorry

end NUMINAMATH_CALUDE_pipes_remaining_proof_l3366_336684


namespace NUMINAMATH_CALUDE_factorial_ratio_l3366_336685

theorem factorial_ratio : Nat.factorial 30 / Nat.factorial 28 = 870 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3366_336685


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3366_336681

/-- 
Given a parabola y = ax^2 + bx + c with vertex (h, h) and y-intercept (0, -2h), 
where h ≠ 0, the value of b is 6.
-/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 → 
  (∀ x y, y = a * x^2 + b * x + c ↔ y - h = a * (x - h)^2) → 
  c = -2 * h → 
  b = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3366_336681


namespace NUMINAMATH_CALUDE_linda_original_money_l3366_336615

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- The amount Lucy would give to Linda -/
def transfer_amount : ℕ := 5

theorem linda_original_money :
  linda_original = 10 :=
by
  have h1 : lucy_original - transfer_amount = linda_original + transfer_amount :=
    sorry
  sorry

end NUMINAMATH_CALUDE_linda_original_money_l3366_336615


namespace NUMINAMATH_CALUDE_jake_eighth_week_hours_l3366_336665

def hours_worked : List ℕ := [14, 9, 12, 15, 11, 13, 10]
def total_weeks : ℕ := 8
def required_average : ℕ := 12

theorem jake_eighth_week_hours :
  ∃ (x : ℕ), 
    (List.sum hours_worked + x) / total_weeks = required_average ∧
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_jake_eighth_week_hours_l3366_336665
