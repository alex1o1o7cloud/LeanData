import Mathlib

namespace kareem_has_largest_number_l2027_202719

def jose_final (start : ℕ) : ℕ :=
  ((start - 2) * 4) + 5

def thuy_final (start : ℕ) : ℕ :=
  ((start * 3) - 3) - 4

def kareem_final (start : ℕ) : ℕ :=
  ((start - 3) + 4) * 3

theorem kareem_has_largest_number :
  kareem_final 20 > jose_final 15 ∧ kareem_final 20 > thuy_final 15 :=
by sorry

end kareem_has_largest_number_l2027_202719


namespace cylinder_volume_l2027_202714

/-- The volume of a cylinder whose lateral surface unfolds into a square with side length 4 -/
theorem cylinder_volume (h : Real) (r : Real) : 
  h = 4 ∧ 2 * Real.pi * r = 4 → Real.pi * r^2 * h = 16 / Real.pi :=
by sorry

end cylinder_volume_l2027_202714


namespace twentieth_term_is_negative_49_l2027_202785

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDiff : ℤ

/-- The nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1 : ℤ) * seq.commonDiff

/-- The theorem stating that the 20th term of the specific arithmetic sequence is -49. -/
theorem twentieth_term_is_negative_49 :
  let seq := ArithmeticSequence.mk 8 (-3)
  nthTerm seq 20 = -49 := by
  sorry

end twentieth_term_is_negative_49_l2027_202785


namespace S_properties_l2027_202755

def S : Set ℤ := {x | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2}

theorem S_properties : 
  (∀ x ∈ S, ¬(3 ∣ x)) ∧ (∃ x ∈ S, 11 ∣ x) := by
  sorry

end S_properties_l2027_202755


namespace book_pages_problem_l2027_202763

theorem book_pages_problem :
  ∃ (n k : ℕ), 
    n > 0 ∧ 
    k > 0 ∧ 
    k < n ∧ 
    n * (n + 1) / 2 - (2 * k + 1) = 4979 :=
sorry

end book_pages_problem_l2027_202763


namespace league_games_count_l2027_202704

/-- The number of games played in a league season -/
def games_in_season (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k

theorem league_games_count :
  games_in_season 20 7 = 1330 := by
  sorry

end league_games_count_l2027_202704


namespace total_seeds_after_trading_is_2340_l2027_202771

/-- Represents the number of watermelon seeds each person has -/
structure SeedCount where
  bom : ℕ
  gwi : ℕ
  yeon : ℕ
  eun : ℕ

/-- Calculates the total number of seeds after trading -/
def totalSeedsAfterTrading (initial : SeedCount) : ℕ :=
  let bomAfter := initial.bom - 50
  let gwiAfter := initial.gwi + (initial.yeon * 20 / 100)
  let yeonAfter := initial.yeon - (initial.yeon * 20 / 100)
  let eunAfter := initial.eun + 50
  bomAfter + gwiAfter + yeonAfter + eunAfter

/-- Theorem stating that the total number of seeds after trading is 2340 -/
theorem total_seeds_after_trading_is_2340 (initial : SeedCount) 
  (h1 : initial.yeon = 3 * initial.gwi)
  (h2 : initial.gwi = initial.bom + 40)
  (h3 : initial.eun = 2 * initial.gwi)
  (h4 : initial.bom = 300) :
  totalSeedsAfterTrading initial = 2340 := by
  sorry

#eval totalSeedsAfterTrading { bom := 300, gwi := 340, yeon := 1020, eun := 680 }

end total_seeds_after_trading_is_2340_l2027_202771


namespace new_person_weight_l2027_202703

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 10 →
  avg_increase = 2.5 →
  old_weight = 65 →
  (n : ℝ) * avg_increase + old_weight = 90 :=
by sorry

end new_person_weight_l2027_202703


namespace x_value_in_sequence_l2027_202733

def fibonacci_like_sequence (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | n+2 => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n+1)

theorem x_value_in_sequence :
  ∃ (start : ℕ), 
    (fibonacci_like_sequence (-1) 2 (start + 2) = 3) ∧
    (fibonacci_like_sequence (-1) 2 (start + 3) = 5) ∧
    (fibonacci_like_sequence (-1) 2 (start + 4) = 8) ∧
    (fibonacci_like_sequence (-1) 2 (start + 5) = 13) ∧
    (fibonacci_like_sequence (-1) 2 (start + 6) = 21) ∧
    (fibonacci_like_sequence (-1) 2 (start + 7) = 34) ∧
    (fibonacci_like_sequence (-1) 2 (start + 8) = 55) := by
  sorry

end x_value_in_sequence_l2027_202733


namespace gas_cost_calculation_l2027_202706

/-- Calculates the total cost of filling up a car's gas tank multiple times with different gas prices -/
theorem gas_cost_calculation (tank_capacity : ℝ) (prices : List ℝ) :
  tank_capacity = 12 ∧ 
  prices = [3, 3.5, 4, 4.5] →
  (prices.map (· * tank_capacity)).sum = 180 := by
  sorry

#check gas_cost_calculation

end gas_cost_calculation_l2027_202706


namespace total_spent_is_correct_l2027_202795

def batman_price : ℚ := 13.60
def superman_price : ℚ := 5.06
def batman_discount : ℚ := 0.10
def superman_discount : ℚ := 0.05
def sales_tax : ℚ := 0.08
def game1_price : ℚ := 7.25
def game2_price : ℚ := 12.50

def total_spent : ℚ :=
  let batman_discounted := batman_price * (1 - batman_discount)
  let superman_discounted := superman_price * (1 - superman_discount)
  let batman_with_tax := batman_discounted * (1 + sales_tax)
  let superman_with_tax := superman_discounted * (1 + sales_tax)
  batman_with_tax + superman_with_tax + game1_price + game2_price

theorem total_spent_is_correct : total_spent = 38.16 := by
  sorry

end total_spent_is_correct_l2027_202795


namespace b_current_age_l2027_202715

/-- Given two people A and B, where:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is currently 8 years older than B.
    This theorem proves that B's current age is 38 years. -/
theorem b_current_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 8) : 
  b = 38 := by
sorry

end b_current_age_l2027_202715


namespace weight_of_A_l2027_202742

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 87 kg -/
theorem weight_of_A (A B C D E : ℝ) : 
  ((A + B + C) / 3 = 60) →
  ((A + B + C + D) / 4 = 65) →
  (E = D + 3) →
  ((B + C + D + E) / 4 = 64) →
  A = 87 := by
sorry

end weight_of_A_l2027_202742


namespace min_marked_points_l2027_202740

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of points in a plane -/
structure PointConfiguration where
  points : Finset Point
  unique_distances : ∀ p q r s : Point, p ∈ points → q ∈ points → r ∈ points → s ∈ points →
    p ≠ q → r ≠ s → (p.x - q.x)^2 + (p.y - q.y)^2 ≠ (r.x - s.x)^2 + (r.y - s.y)^2

/-- The set of points marked as closest to at least one other point -/
def marked_points (config : PointConfiguration) : Finset Point :=
  sorry

/-- The theorem stating the minimum number of marked points -/
theorem min_marked_points (config : PointConfiguration) :
  config.points.card = 2018 →
  (marked_points config).card ≥ 404 :=
sorry

end min_marked_points_l2027_202740


namespace books_read_in_common_l2027_202778

theorem books_read_in_common (tony_books dean_books breanna_books total_books : ℕ) 
  (h1 : tony_books = 23)
  (h2 : dean_books = 12)
  (h3 : breanna_books = 17)
  (h4 : total_books = 47)
  (h5 : ∃ (common : ℕ), common > 0 ∧ common ≤ min tony_books dean_books)
  (h6 : ∃ (all_common : ℕ), all_common > 0 ∧ all_common ≤ min tony_books (min dean_books breanna_books)) :
  ∃ (x : ℕ), x = 3 ∧ 
    tony_books + dean_books + breanna_books - x - 1 = total_books :=
by sorry

end books_read_in_common_l2027_202778


namespace absolute_value_of_five_minus_pi_plus_two_l2027_202732

theorem absolute_value_of_five_minus_pi_plus_two : |5 - Real.pi + 2| = 7 - Real.pi := by
  sorry

end absolute_value_of_five_minus_pi_plus_two_l2027_202732


namespace masters_proportion_in_team_l2027_202721

/-- Represents a team of juniors and masters in a shooting tournament. -/
structure ShootingTeam where
  juniors : ℕ
  masters : ℕ

/-- Calculates the proportion of masters in the team. -/
def mastersProportion (team : ShootingTeam) : ℚ :=
  team.masters / (team.juniors + team.masters)

/-- The theorem stating the proportion of masters in the team under given conditions. -/
theorem masters_proportion_in_team (team : ShootingTeam) 
  (h1 : 22 * team.juniors + 47 * team.masters = 41 * (team.juniors + team.masters)) :
  mastersProportion team = 19 / 25 := by
  sorry

#eval (19 : ℚ) / 25  -- To verify that 19/25 is indeed equal to 0.76

end masters_proportion_in_team_l2027_202721


namespace age_sum_problem_l2027_202766

theorem age_sum_problem (twin1_age twin2_age youngest_age : ℕ) :
  twin1_age = twin2_age →
  twin1_age > youngest_age →
  youngest_age < 10 →
  twin1_age * twin2_age * youngest_age = 72 →
  twin1_age + twin2_age + youngest_age = 14 := by
  sorry

end age_sum_problem_l2027_202766


namespace perimeter_area_sum_l2027_202775

/-- A parallelogram with vertices at (2,3), (2,8), (9,8), and (9,3) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, 3)
  v2 : ℝ × ℝ := (2, 8)
  v3 : ℝ × ℝ := (9, 8)
  v4 : ℝ × ℝ := (9, 3)

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  2 * (abs (p.v3.1 - p.v1.1) + abs (p.v2.2 - p.v1.2))

/-- Calculate the area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  abs (p.v3.1 - p.v1.1) * abs (p.v2.2 - p.v1.2)

/-- The sum of the perimeter and area of the parallelogram is 59 -/
theorem perimeter_area_sum (p : Parallelogram) : perimeter p + area p = 59 := by
  sorry

end perimeter_area_sum_l2027_202775


namespace fuel_change_calculation_l2027_202770

/-- Calculates the change received when fueling a vehicle --/
theorem fuel_change_calculation (tank_capacity : ℝ) (initial_fuel : ℝ) (fuel_cost : ℝ) (payment : ℝ) :
  tank_capacity = 150 →
  initial_fuel = 38 →
  fuel_cost = 3 →
  payment = 350 →
  payment - (tank_capacity - initial_fuel) * fuel_cost = 14 := by
  sorry

#check fuel_change_calculation

end fuel_change_calculation_l2027_202770


namespace distance_major_minor_endpoints_l2027_202735

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  4 * (x - 3)^2 + 16 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the semi-major and semi-minor axes
def a : ℝ := 4
def b : ℝ := 2

-- Define a point on the major axis
def point_on_major_axis (x y : ℝ) : Prop :=
  ellipse x y ∧ y = center.2

-- Define a point on the minor axis
def point_on_minor_axis (x y : ℝ) : Prop :=
  ellipse x y ∧ x = center.1

-- Theorem: The distance between an endpoint of the major axis
-- and an endpoint of the minor axis is 2√5
theorem distance_major_minor_endpoints :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    point_on_major_axis x₁ y₁ ∧
    point_on_minor_axis x₂ y₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 20 :=
sorry

end distance_major_minor_endpoints_l2027_202735


namespace max_mogs_bill_can_buy_l2027_202788

/-- The cost of one mag -/
def mag_cost : ℕ := 3

/-- The cost of one mig -/
def mig_cost : ℕ := 4

/-- The cost of one mog -/
def mog_cost : ℕ := 8

/-- The total amount Bill will spend -/
def total_spent : ℕ := 100

/-- Theorem stating the maximum number of mogs Bill can buy -/
theorem max_mogs_bill_can_buy :
  ∃ (mags migs mogs : ℕ),
    mags ≥ 1 ∧
    migs ≥ 1 ∧
    mogs ≥ 1 ∧
    mag_cost * mags + mig_cost * migs + mog_cost * mogs = total_spent ∧
    mogs = 10 ∧
    (∀ (m : ℕ), m > 10 →
      ¬∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧
        mag_cost * x + mig_cost * y + mog_cost * m = total_spent) :=
sorry

end max_mogs_bill_can_buy_l2027_202788


namespace visible_bird_legs_count_l2027_202726

theorem visible_bird_legs_count :
  let crows : ℕ := 4
  let pigeons : ℕ := 3
  let flamingos : ℕ := 5
  let sparrows : ℕ := 8
  let crow_legs : ℕ := 2
  let pigeon_legs : ℕ := 2
  let flamingo_legs : ℕ := 3
  let sparrow_legs : ℕ := 2
  crows * crow_legs + pigeons * pigeon_legs + flamingos * flamingo_legs + sparrows * sparrow_legs = 45 :=
by sorry

end visible_bird_legs_count_l2027_202726


namespace book_costs_proof_l2027_202736

theorem book_costs_proof (total_cost : ℝ) (book1 book2 book3 book4 book5 : ℝ) :
  total_cost = 24 ∧
  book1 = book2 + 2 ∧
  book3 = book1 + 4 ∧
  book4 = book3 - 3 ∧
  book5 = book2 ∧
  book1 ≠ book2 ∧ book1 ≠ book3 ∧ book1 ≠ book4 ∧ book1 ≠ book5 ∧
  book2 ≠ book3 ∧ book2 ≠ book4 ∧
  book3 ≠ book4 ∧ book3 ≠ book5 ∧
  book4 ≠ book5 →
  book1 = 4.6 ∧ book2 = 2.6 ∧ book3 = 8.6 ∧ book4 = 5.6 ∧ book5 = 2.6 ∧
  total_cost = book1 + book2 + book3 + book4 + book5 := by
sorry

end book_costs_proof_l2027_202736


namespace alice_bushes_l2027_202781

/-- The number of bushes needed to cover three sides of a yard --/
def bushes_needed (side_length : ℕ) (sides : ℕ) (bush_width : ℕ) : ℕ :=
  (side_length * sides) / bush_width

/-- Theorem: Alice needs 24 bushes for her yard --/
theorem alice_bushes :
  bushes_needed 24 3 3 = 24 := by
  sorry

end alice_bushes_l2027_202781


namespace bronze_ball_balance_l2027_202731

theorem bronze_ball_balance (a : Fin 10 → ℝ) : 
  ∃ (S : Finset (Fin 10)), 
    (S.sum (λ i => |a (i + 1) - a i|)) = 
    ((Finset.univ \ S).sum (λ i => |a (i + 1) - a i|)) := by
  sorry


end bronze_ball_balance_l2027_202731


namespace proportion_equality_l2027_202713

theorem proportion_equality (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 := by
  sorry

end proportion_equality_l2027_202713


namespace no_solution_implies_b_bounded_l2027_202777

theorem no_solution_implies_b_bounded (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) →
  abs b ≤ 1 := by
  sorry

end no_solution_implies_b_bounded_l2027_202777


namespace tourist_survival_l2027_202772

theorem tourist_survival (initial_tourists : ℕ) (eaten : ℕ) (poison_fraction : ℚ) (recovery_fraction : ℚ) : initial_tourists = 30 → eaten = 2 → poison_fraction = 1/2 → recovery_fraction = 1/7 → 
  let remaining_after_eaten := initial_tourists - eaten
  let poisoned := (remaining_after_eaten : ℚ) * poison_fraction
  let recovered := poisoned * recovery_fraction
  (remaining_after_eaten : ℚ) - poisoned + recovered = 16 := by
  sorry

end tourist_survival_l2027_202772


namespace intersection_problem_l2027_202783

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the theorem
theorem intersection_problem (a b c d : ℝ) :
  (f a b 2 = 4) →  -- The graphs intersect at x = 2
  (g c d 2 = 4) →  -- The graphs intersect at x = 2
  (b + c = 1) →    -- Given condition
  (4 * a + d = 1)  -- What we want to prove
:= by sorry

end intersection_problem_l2027_202783


namespace b_not_unique_l2027_202751

-- Define the line equation
def line_equation (y : ℝ) : ℝ := 8 * y + 5

-- Define the points on the line
def point1 (m B : ℝ) : ℝ × ℝ := (m, B)
def point2 (m B : ℝ) : ℝ × ℝ := (m + 2, B + 0.25)

-- Theorem stating that B cannot be uniquely determined
theorem b_not_unique (m B : ℝ) : 
  line_equation B = (point1 m B).1 ∧ 
  line_equation (B + 0.25) = (point2 m B).1 → 
  ∃ (B' : ℝ), B' ≠ B ∧ 
    line_equation B' = (point1 m B').1 ∧ 
    line_equation (B' + 0.25) = (point2 m B').1 :=
by
  sorry


end b_not_unique_l2027_202751


namespace divisibility_of_power_tower_plus_one_l2027_202722

theorem divisibility_of_power_tower_plus_one (a : ℕ) : 
  ∃ n : ℕ, ∀ k : ℕ, a ∣ n^(n^k) + 1 := by
  sorry

end divisibility_of_power_tower_plus_one_l2027_202722


namespace lower_bound_of_fraction_l2027_202774

theorem lower_bound_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (3 * a) + 3 / b ≥ 8 / 3 := by
sorry

end lower_bound_of_fraction_l2027_202774


namespace entire_group_is_population_l2027_202794

/-- Represents a group of students who took a test -/
structure StudentGroup where
  size : ℕ
  scores : Finset ℝ
  h_size : scores.card = size

/-- Represents a sample extracted from a larger group -/
structure Sample (group : StudentGroup) where
  size : ℕ
  scores : Finset ℝ
  h_size : scores.card = size
  h_subset : scores ⊆ group.scores

/-- Definition of a population in statistical terms -/
def isPopulation (group : StudentGroup) : Prop :=
  ∀ (sample : Sample group), sample.scores ⊆ group.scores

/-- The theorem to be proved -/
theorem entire_group_is_population 
  (entireGroup : StudentGroup) 
  (sample : Sample entireGroup) 
  (h_entire_size : entireGroup.size = 5000) 
  (h_sample_size : sample.size = 200) : 
  isPopulation entireGroup := by
  sorry

end entire_group_is_population_l2027_202794


namespace even_function_implies_a_equals_one_l2027_202754

/-- Given that f(x) = x³(a⋅2^x - 2^(-x)) is an even function, prove that a = 1 --/
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 := by
sorry

end even_function_implies_a_equals_one_l2027_202754


namespace correct_classification_l2027_202711

-- Define the set of statement numbers
def StatementNumbers : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the function that classifies numbers as precise or approximate
def classify : Nat → Bool
| 1 => true  -- Xiao Ming's books (precise)
| 2 => true  -- War cost (precise)
| 3 => true  -- DVD sales (precise)
| 4 => false -- Brain cells (approximate)
| 5 => true  -- Xiao Hong's score (precise)
| 6 => false -- Coal reserves (approximate)
| _ => false -- For completeness

-- Theorem statement
theorem correct_classification :
  {n ∈ StatementNumbers | classify n = true} = {1, 2, 3, 5} ∧
  {n ∈ StatementNumbers | classify n = false} = {4, 6} := by
  sorry


end correct_classification_l2027_202711


namespace lcm_ratio_sum_l2027_202765

theorem lcm_ratio_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  Nat.lcm a b = 54 → a * 3 = b * 2 → a + b = 45 := by
  sorry

end lcm_ratio_sum_l2027_202765


namespace cuboid_surface_area_l2027_202702

theorem cuboid_surface_area (h : ℝ) (sum_edges : ℝ) (surface_area : ℝ) : 
  sum_edges = 100 ∧ 
  20 * h = sum_edges ∧ 
  surface_area = 2 * (2*h * 2*h + 2*h * h + 2*h * h) → 
  surface_area = 400 := by
sorry

end cuboid_surface_area_l2027_202702


namespace quadratic_roots_range_l2027_202776

theorem quadratic_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) →
  a < 1 ∧ a ≠ 0 := by
sorry

end quadratic_roots_range_l2027_202776


namespace quadratic_equation_roots_l2027_202705

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + m * x = -6) ∧ (7 * 3^2 + m * 3 = -6) → 
  (7 * (2/7)^2 + m * (2/7) = -6) := by
sorry

end quadratic_equation_roots_l2027_202705


namespace notebook_pen_equation_l2027_202761

theorem notebook_pen_equation (x : ℝ) : 
  (5 * (x - 2) + 3 * x = 14) ↔ 
  (∃ (notebook_price : ℝ), 
    notebook_price = x - 2 ∧ 
    5 * notebook_price + 3 * x = 14) :=
by sorry

end notebook_pen_equation_l2027_202761


namespace consecutive_odd_squares_difference_l2027_202780

theorem consecutive_odd_squares_difference (n : ℕ) : 
  ∃ k : ℤ, (2*n + 1)^2 - (2*n - 1)^2 = 8 * k := by
  sorry

end consecutive_odd_squares_difference_l2027_202780


namespace tangent_line_at_one_a_range_when_f_negative_l2027_202745

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_line_at_one (h : ℝ → ℝ := f 2) :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + h 1 ↔ x + y + 1 = 0 :=
sorry

theorem a_range_when_f_negative (a : ℝ) :
  (∀ x > 0, f a x < 0) → a > Real.exp (-1) :=
sorry

end tangent_line_at_one_a_range_when_f_negative_l2027_202745


namespace simplify_polynomial_expression_l2027_202767

/-- Given two polynomials A and B in x and y, prove that 2A - B simplifies to a specific form. -/
theorem simplify_polynomial_expression (x y : ℝ) :
  let A := 2 * x^2 + x * y - 3
  let B := -x^2 + 2 * x * y - 1
  2 * A - B = 5 * x^2 - 5 := by
  sorry

end simplify_polynomial_expression_l2027_202767


namespace greatest_multiple_of_four_l2027_202759

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^3 < 500 → x ≤ 4 ∧ ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^3 < 500 ∧ y = 4 :=
by sorry

end greatest_multiple_of_four_l2027_202759


namespace negative_number_identification_l2027_202720

theorem negative_number_identification :
  let a := -(-2)
  let b := abs (-2)
  let c := (-2)^2
  let d := (-2)^3
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 0) := by sorry

end negative_number_identification_l2027_202720


namespace spider_group_ratio_l2027_202786

/-- Represents a group of spiders -/
structure SpiderGroup where
  /-- Number of spiders in the group -/
  count : ℕ
  /-- Number of legs per spider -/
  legsPerSpider : ℕ
  /-- The group has more spiders than half the legs of a single spider -/
  more_than_half : count > legsPerSpider / 2
  /-- Total number of legs in the group -/
  totalLegs : ℕ
  /-- The total legs is the product of count and legs per spider -/
  total_legs_eq : totalLegs = count * legsPerSpider

/-- The theorem to be proved -/
theorem spider_group_ratio (g : SpiderGroup)
  (h1 : g.legsPerSpider = 8)
  (h2 : g.totalLegs = 112) :
  (g.count : ℚ) / (g.legsPerSpider / 2 : ℚ) = 7 / 2 :=
sorry

end spider_group_ratio_l2027_202786


namespace quadratic_inequality_roots_l2027_202725

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x, -x^2 + b*x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 := by
  sorry

end quadratic_inequality_roots_l2027_202725


namespace sqrt_plus_inverse_geq_two_l2027_202737

theorem sqrt_plus_inverse_geq_two (x : ℝ) (hx : x > 0) : Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by
  sorry

end sqrt_plus_inverse_geq_two_l2027_202737


namespace indistinguishable_balls_in_boxes_l2027_202709

/-- The number of partitions of n indistinguishable objects into k or fewer non-empty parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The balls are indistinguishable -/
def balls : ℕ := 4

/-- The boxes are indistinguishable -/
def boxes : ℕ := 4

theorem indistinguishable_balls_in_boxes : partition_count balls boxes = 5 := by sorry

end indistinguishable_balls_in_boxes_l2027_202709


namespace zero_last_to_appear_l2027_202701

-- Define the Fibonacci sequence modulo 9
def fibMod9 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => (fibMod9 n + fibMod9 (n + 1)) % 9

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fibMod9 k = d

-- Define a function to check if all digits from 0 to 8 have appeared
def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d, d ≤ 8 → digitAppeared d n

-- The main theorem
theorem zero_last_to_appear :
  ∃ n, allDigitsAppeared n ∧
    ¬(∃ k < n, allDigitsAppeared k) ∧
    fibMod9 n = 0 :=
  sorry

end zero_last_to_appear_l2027_202701


namespace ellipse_problem_l2027_202744

def given_ellipse (x y : ℝ) : Prop :=
  8 * x^2 / 81 + y^2 / 36 = 1

def reference_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

def required_ellipse (x y : ℝ) : Prop :=
  x^2 / 15 + y^2 / 10 = 1

theorem ellipse_problem (x₀ : ℝ) (h1 : given_ellipse x₀ 2) (h2 : x₀ < 0) :
  x₀ = -3 ∧
  ∀ (x y : ℝ), (x = x₀ ∧ y = 2 → required_ellipse x y) ∧
  (∃ (c : ℝ), ∀ (x y : ℝ), reference_ellipse x y ↔ x^2 + y^2 = 9 + 4 - c^2 ∧ c^2 = 5) ∧
  (∃ (c : ℝ), ∀ (x y : ℝ), required_ellipse x y ↔ x^2 + y^2 = 15 + 10 - c^2 ∧ c^2 = 5) :=
by sorry

end ellipse_problem_l2027_202744


namespace tornado_distance_ratio_l2027_202717

/-- Given the distances traveled by various objects in a tornado, prove the ratio of
    the lawn chair's distance to the car's distance. -/
theorem tornado_distance_ratio :
  ∀ (car_distance lawn_chair_distance birdhouse_distance : ℝ),
  car_distance = 200 →
  birdhouse_distance = 1200 →
  birdhouse_distance = 3 * lawn_chair_distance →
  lawn_chair_distance / car_distance = 2 := by
  sorry

end tornado_distance_ratio_l2027_202717


namespace pencil_buyers_difference_l2027_202710

theorem pencil_buyers_difference (pencil_cost : ℕ) 
  (h1 : pencil_cost > 0)
  (h2 : 234 % pencil_cost = 0)
  (h3 : 312 % pencil_cost = 0) :
  312 / pencil_cost - 234 / pencil_cost = 6 :=
by sorry

end pencil_buyers_difference_l2027_202710


namespace competition_scores_l2027_202716

theorem competition_scores (x y z w : ℝ) 
  (hA : x = (y + z + w) / 3 + 2)
  (hB : y = (x + z + w) / 3 - 3)
  (hC : z = (x + y + w) / 3 + 3) :
  (x + y + z) / 3 - w = 2 := by sorry

end competition_scores_l2027_202716


namespace king_crown_payment_l2027_202757

/-- Calculates the total amount paid for a crown, including tip -/
def totalAmountPaid (crownCost : ℝ) (tipRate : ℝ) : ℝ :=
  crownCost + (crownCost * tipRate)

/-- Theorem: The king pays $22,000 for a $20,000 crown with a 10% tip -/
theorem king_crown_payment :
  totalAmountPaid 20000 0.1 = 22000 := by
  sorry

end king_crown_payment_l2027_202757


namespace cubic_polynomial_satisfies_conditions_l2027_202797

-- Define the cubic polynomial
def q (x : ℚ) : ℚ := 7/4 * x^3 - 19 * x^2 + 149/4 * x + 6

-- Theorem statement
theorem cubic_polynomial_satisfies_conditions :
  q 1 = -6 ∧ q 3 = -20 ∧ q 4 = -42 ∧ q 5 = -60 := by
  sorry


end cubic_polynomial_satisfies_conditions_l2027_202797


namespace system_solution_sum_reciprocals_l2027_202756

theorem system_solution_sum_reciprocals (x₀ y₀ : ℚ) :
  x₀ / 3 + y₀ / 5 = 1 ∧ x₀ / 5 + y₀ / 3 = 1 →
  1 / x₀ + 1 / y₀ = 16 / 15 := by
sorry

end system_solution_sum_reciprocals_l2027_202756


namespace lent_sum_theorem_l2027_202791

/-- Represents the sum of money lent in two parts -/
structure LentSum where
  first_part : ℕ
  second_part : ℕ
  total : ℕ

/-- Calculates the interest on a principal amount for a given rate and time -/
def calculate_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time

theorem lent_sum_theorem (s : LentSum) :
  s.second_part = 1672 →
  calculate_interest s.first_part 3 8 = calculate_interest s.second_part 5 3 →
  s.total = s.first_part + s.second_part →
  s.total = 2717 := by
    sorry

end lent_sum_theorem_l2027_202791


namespace arithmetic_calculations_l2027_202723

theorem arithmetic_calculations :
  (24 - |(-2)| + (-16) - 8 = -2) ∧
  ((-2) * (3/2) / (-3/4) * 4 = 16) ∧
  (-1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1/6) := by
  sorry

end arithmetic_calculations_l2027_202723


namespace residue_calculation_l2027_202764

theorem residue_calculation : 196 * 18 - 21 * 9 + 5 ≡ 14 [ZMOD 18] := by
  sorry

end residue_calculation_l2027_202764


namespace remainder_5_divisors_2002_l2027_202773

def divides_with_remainder_5 (d : ℕ) : Prop :=
  ∃ q : ℕ, 2007 = d * q + 5

def divisors_of_2002 : Set ℕ :=
  {d : ℕ | d > 0 ∧ 2002 % d = 0}

theorem remainder_5_divisors_2002 :
  {d : ℕ | divides_with_remainder_5 d} = {d ∈ divisors_of_2002 | d > 5} :=
by sorry

end remainder_5_divisors_2002_l2027_202773


namespace power_difference_l2027_202700

theorem power_difference (a m n : ℝ) (hm : a^m = 12) (hn : a^n = 3) : a^(m-n) = 4 := by
  sorry

end power_difference_l2027_202700


namespace percentage_of_part_to_whole_l2027_202750

theorem percentage_of_part_to_whole (total : ℝ) (part : ℝ) : 
  total > 0 → part ≥ 0 → part ≤ total → (part / total) * 100 = 25 → total = 400 ∧ part = 100 := by
  sorry

end percentage_of_part_to_whole_l2027_202750


namespace cos_negative_300_degrees_l2027_202741

theorem cos_negative_300_degrees : Real.cos (-300 * π / 180) = 1 / 2 := by
  sorry

end cos_negative_300_degrees_l2027_202741


namespace driver_total_stops_l2027_202718

/-- The total number of stops made by a delivery driver -/
def total_stops (initial_stops additional_stops : ℕ) : ℕ :=
  initial_stops + additional_stops

/-- Theorem: The delivery driver made 7 stops in total -/
theorem driver_total_stops :
  total_stops 3 4 = 7 := by
  sorry

end driver_total_stops_l2027_202718


namespace new_ratio_after_transaction_l2027_202707

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the transaction of selling horses and buying cows -/
def performTransaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

/-- Theorem stating the new ratio of horses to cows after the transaction -/
theorem new_ratio_after_transaction (initial : FarmAnimals)
    (h1 : initial.horses = 4 * initial.cows)
    (h2 : (performTransaction initial).horses = (performTransaction initial).cows + 60) :
    (performTransaction initial).horses / (performTransaction initial).cows = 7 / 3 := by
  sorry


end new_ratio_after_transaction_l2027_202707


namespace x_power_125_minus_reciprocal_l2027_202762

theorem x_power_125_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^125 - 1/x^125 = Real.sqrt 3 := by
  sorry

end x_power_125_minus_reciprocal_l2027_202762


namespace largest_n_for_factorization_l2027_202793

/-- 
Given a quadratic expression 3x^2 + nx + 54, this theorem states that 163 is the largest 
value of n for which the expression can be factored as the product of two linear factors 
with integer coefficients.
-/
theorem largest_n_for_factorization : 
  ∀ n : ℤ, (∃ a b c d : ℤ, 3*x^2 + n*x + 54 = (a*x + b) * (c*x + d)) → n ≤ 163 :=
by sorry

end largest_n_for_factorization_l2027_202793


namespace percent_value_in_quarters_l2027_202790

theorem percent_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let total_dimes_value : ℕ := num_dimes * dime_value
  let total_quarters_value : ℕ := num_quarters * quarter_value
  let total_value : ℕ := total_dimes_value + total_quarters_value
  (total_quarters_value : ℝ) / (total_value : ℝ) * 100 = 65.22 :=
by sorry

end percent_value_in_quarters_l2027_202790


namespace vicente_spent_25_l2027_202749

/-- The total amount Vicente spent on rice and meat -/
def total_spent (rice_kg : ℕ) (rice_price : ℚ) (meat_lb : ℕ) (meat_price : ℚ) : ℚ :=
  (rice_kg : ℚ) * rice_price + (meat_lb : ℚ) * meat_price

/-- Proof that Vicente spent $25 on his purchase -/
theorem vicente_spent_25 :
  total_spent 5 2 3 5 = 25 := by
  sorry

end vicente_spent_25_l2027_202749


namespace scooter_gain_percentage_l2027_202758

/-- Calculates the overall gain percentage for three scooters -/
def overall_gain_percentage (purchase_price_A purchase_price_B purchase_price_C : ℚ)
                            (repair_cost_A repair_cost_B repair_cost_C : ℚ)
                            (selling_price_A selling_price_B selling_price_C : ℚ) : ℚ :=
  let total_cost := purchase_price_A + purchase_price_B + purchase_price_C +
                    repair_cost_A + repair_cost_B + repair_cost_C
  let total_revenue := selling_price_A + selling_price_B + selling_price_C
  let total_gain := total_revenue - total_cost
  (total_gain / total_cost) * 100

/-- Theorem stating that the overall gain percentage for the given scooter transactions is 10% -/
theorem scooter_gain_percentage :
  overall_gain_percentage 4700 3500 5400 600 800 1000 5800 4800 7000 = 10 := by
  sorry

end scooter_gain_percentage_l2027_202758


namespace complex_number_properties_l2027_202748

def i : ℂ := Complex.I

theorem complex_number_properties (z : ℂ) (h : z * (2 - i) = i ^ 2020) :
  (Complex.im z = 1/5) ∧ (Complex.re z > 0 ∧ Complex.im z > 0) := by
  sorry

end complex_number_properties_l2027_202748


namespace final_result_proof_l2027_202739

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 1376) :
  (chosen_number / 8 : ℚ) - 160 = 12 := by
  sorry

end final_result_proof_l2027_202739


namespace sqrt_of_36_l2027_202728

theorem sqrt_of_36 : Real.sqrt 36 = 6 := by
  sorry

end sqrt_of_36_l2027_202728


namespace not_both_bidirectional_l2027_202743

-- Define the two proof methods
inductive ProofMethod
| Synthetic
| Analytic

-- Define the reasoning direction
inductive ReasoningDirection
| CauseToEffect
| EffectToCause

-- Define the properties of the proof methods
def methodDirection (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

-- Theorem statement
theorem not_both_bidirectional :
  ¬(∀ (m : ProofMethod), 
    (methodDirection m = ReasoningDirection.CauseToEffect ∧
     methodDirection m = ReasoningDirection.EffectToCause)) :=
by
  sorry

end not_both_bidirectional_l2027_202743


namespace intersection_of_P_and_M_l2027_202746

def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem intersection_of_P_and_M : P ∩ M = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end intersection_of_P_and_M_l2027_202746


namespace odd_periodic_function_property_l2027_202784

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_period : HasPeriod f 5) 
    (h1 : f 1 = 1) 
    (h2 : f 2 = 2) : 
  f 3 - f 4 = -1 := by
sorry

end odd_periodic_function_property_l2027_202784


namespace female_democrats_count_l2027_202779

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 840 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 140 :=
by sorry

end female_democrats_count_l2027_202779


namespace complex_equation_solution_l2027_202708

theorem complex_equation_solution :
  ∀ z : ℂ, z * (1 - Complex.I) = (1 + Complex.I)^3 → z = -2 := by
  sorry

end complex_equation_solution_l2027_202708


namespace triangle_nature_l2027_202724

theorem triangle_nature (a b c : ℝ) (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5)
  (h_perimeter : a + b + c = 36) : a^2 + b^2 = c^2 := by
  sorry

end triangle_nature_l2027_202724


namespace intersection_point_of_lines_l2027_202799

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Definition of the first line: y = -3x -/
def line1 (x y : ℚ) : Prop := y = -3 * x

/-- Definition of the second line: y - 3 = 9x -/
def line2 (x y : ℚ) : Prop := y - 3 = 9 * x

/-- Theorem stating that (-1/4, 3/4) is the unique intersection point of the two lines -/
theorem intersection_point_of_lines :
  ∃! p : IntersectionPoint, line1 p.x p.y ∧ line2 p.x p.y ∧ p.x = -1/4 ∧ p.y = 3/4 := by
  sorry

end intersection_point_of_lines_l2027_202799


namespace f_odd_and_periodic_l2027_202768

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Define odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define periodic function
def is_periodic (f : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x

-- Theorem statement
theorem f_odd_and_periodic : is_odd f ∧ is_periodic f := by sorry

end f_odd_and_periodic_l2027_202768


namespace larger_quadrilateral_cyclic_l2027_202798

/-- A configuration of five cyclic quadrilaterals forming a larger quadrilateral -/
structure FiveQuadrilateralsConfig where
  /-- The angles of the five smaller quadrilaterals -/
  angles : Fin 10 → ℝ
  /-- Each smaller quadrilateral is cyclic -/
  cyclic_small : ∀ i : Fin 5, angles (2*i) + angles (2*i+1) = 180
  /-- The sum of angles around each internal vertex is 360° -/
  vertex_sum : angles 1 + angles 7 + angles 8 = 360 ∧ angles 3 + angles 5 + angles 9 = 360

/-- The theorem stating that the larger quadrilateral is cyclic -/
theorem larger_quadrilateral_cyclic (config : FiveQuadrilateralsConfig) :
  config.angles 0 + config.angles 2 + config.angles 4 + config.angles 6 = 180 :=
sorry

end larger_quadrilateral_cyclic_l2027_202798


namespace number_puzzle_l2027_202760

theorem number_puzzle (x y : ℝ) (h1 : x + y = 25) (h2 : x - y = 15) : x^2 - y^3 = 275 := by
  sorry

end number_puzzle_l2027_202760


namespace five_letter_words_same_start_end_l2027_202796

theorem five_letter_words_same_start_end (alphabet_size : ℕ) (word_length : ℕ) : 
  alphabet_size = 26 → word_length = 5 → 
  (alphabet_size ^ (word_length - 2)) * alphabet_size = 456976 := by
  sorry

end five_letter_words_same_start_end_l2027_202796


namespace sum_of_x_and_y_is_two_l2027_202782

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end sum_of_x_and_y_is_two_l2027_202782


namespace complex_division_equality_l2027_202787

theorem complex_division_equality : (3 - I) / (1 + I) = 1 - 2*I := by sorry

end complex_division_equality_l2027_202787


namespace simplify_sqrt_sum_l2027_202727

theorem simplify_sqrt_sum (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2)^2) + Real.sqrt ((a - 8)^2) = 6 := by
  sorry

end simplify_sqrt_sum_l2027_202727


namespace square_of_complex_number_l2027_202752

theorem square_of_complex_number :
  let z : ℂ := 5 - 3*I
  z^2 = 16 - 30*I := by sorry

end square_of_complex_number_l2027_202752


namespace john_task_completion_l2027_202729

-- Define the start time and end time of the first three tasks
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes
def end_three_tasks : Nat := 12 * 60 + 15  -- 12:15 PM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem john_task_completion 
  (h1 : end_three_tasks - start_time = (num_tasks - 1) * ((end_three_tasks - start_time) / (num_tasks - 1)))
  (h2 : (end_three_tasks - start_time) % (num_tasks - 1) = 0) :
  end_three_tasks + ((end_three_tasks - start_time) / (num_tasks - 1)) = 13 * 60 + 20 := by
sorry


end john_task_completion_l2027_202729


namespace unique_two_digit_number_l2027_202747

/-- A 2-digit positive integer is represented by its tens and ones digits -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a 2-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem unique_two_digit_number :
  ∃! (c : TwoDigitNumber), 
    c.tens + c.ones = 10 ∧ 
    c.tens * c.ones = 25 ∧ 
    c.value = 55 := by
  sorry

end unique_two_digit_number_l2027_202747


namespace pants_price_is_6_l2027_202753

-- Define variables for pants and shirt prices
variable (pants_price : ℝ)
variable (shirt_price : ℝ)

-- Define Peter's purchase
def peter_total : ℝ := 2 * pants_price + 5 * shirt_price

-- Define Jessica's purchase
def jessica_total : ℝ := 2 * shirt_price

-- Theorem stating the price of one pair of pants
theorem pants_price_is_6 
  (h1 : peter_total = 62)
  (h2 : jessica_total = 20) :
  pants_price = 6 := by
sorry

end pants_price_is_6_l2027_202753


namespace partial_fraction_decomposition_sum_l2027_202738

theorem partial_fraction_decomposition_sum (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_decomposition_sum_l2027_202738


namespace ballon_arrangements_l2027_202730

theorem ballon_arrangements :
  let total_letters : Nat := 6
  let repeated_letters : Nat := 2
  Nat.factorial total_letters / Nat.factorial repeated_letters = 360 := by
  sorry

end ballon_arrangements_l2027_202730


namespace digit_count_problem_l2027_202769

theorem digit_count_problem (n : ℕ) 
  (h1 : (n : ℝ) * 500 = 14 * 390 + 6 * 756.67)
  (h2 : n > 0) : 
  n = 20 := by
  sorry

end digit_count_problem_l2027_202769


namespace characterization_of_expressible_numbers_l2027_202789

theorem characterization_of_expressible_numbers (n : ℕ) :
  (∃ k : ℕ, n = k + 2 * Int.floor (Real.sqrt k) + 2) ↔
  (∀ y : ℕ, n ≠ y^2 ∧ n ≠ y^2 - 1) :=
sorry

end characterization_of_expressible_numbers_l2027_202789


namespace janet_masud_sibling_ratio_l2027_202712

/-- The number of Masud's siblings -/
def masud_siblings : ℕ := 60

/-- The number of Carlos' siblings -/
def carlos_siblings : ℕ := (3 * masud_siblings) / 4

/-- The number of Janet's siblings -/
def janet_siblings : ℕ := carlos_siblings + 135

/-- The ratio of Janet's siblings to Masud's siblings -/
def sibling_ratio : ℚ := janet_siblings / masud_siblings

theorem janet_masud_sibling_ratio :
  sibling_ratio = 3 / 1 := by sorry

end janet_masud_sibling_ratio_l2027_202712


namespace tank_plastering_cost_l2027_202792

/-- Calculate the cost of plastering a tank's walls and bottom -/
def plasteringCost (length width depth : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * costPerSquareMeter

/-- Theorem: The cost of plastering a tank with given dimensions is 558 rupees -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

end tank_plastering_cost_l2027_202792


namespace unique_solution_l2027_202734

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 8

def are_distinct (a b c d e f g h : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h

def four_digit_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem unique_solution (a b c d e f g h : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  is_valid_digit e ∧ is_valid_digit f ∧ is_valid_digit g ∧ is_valid_digit h ∧
  are_distinct a b c d e f g h ∧
  four_digit_number a b c d + e * f * g * h = 2011 →
  four_digit_number a b c d = 1563 := by
sorry

end unique_solution_l2027_202734
