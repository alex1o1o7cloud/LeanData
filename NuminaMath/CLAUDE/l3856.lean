import Mathlib

namespace onion_harvest_weight_l3856_385660

/-- Calculates the total weight of onions harvested given the number of trips, 
    initial number of bags, increase in bags per trip, and weight per bag. -/
def totalOnionWeight (trips : ℕ) (initialBags : ℕ) (increase : ℕ) (weightPerBag : ℕ) : ℕ :=
  let finalBags := initialBags + (trips - 1) * increase
  let totalBags := trips * (initialBags + finalBags) / 2
  totalBags * weightPerBag

/-- Theorem stating that the total weight of onions harvested is 29,000 kilograms
    given the specific conditions of the problem. -/
theorem onion_harvest_weight :
  totalOnionWeight 20 10 2 50 = 29000 := by
  sorry

end onion_harvest_weight_l3856_385660


namespace absolute_value_inequality_solution_set_l3856_385639

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| + |x + 1| ≤ 5} = Set.Icc (-2) 3 := by
  sorry

end absolute_value_inequality_solution_set_l3856_385639


namespace angle_measure_problem_l3856_385607

theorem angle_measure_problem (angle_B angle_small_triangle : ℝ) :
  angle_B = 120 →
  angle_small_triangle = 50 →
  ∃ angle_A : ℝ,
    angle_A = 70 ∧
    angle_A + angle_small_triangle + (180 - angle_B) = 180 :=
by sorry

end angle_measure_problem_l3856_385607


namespace angle_conversion_l3856_385622

theorem angle_conversion (angle : Real) : ∃ (k : Int) (α : Real),
  angle * Real.pi / 180 = 2 * k * Real.pi + α ∧ 0 < α ∧ α < 2 * Real.pi :=
by
  -- The angle -1485° in radians is equal to -1485 * π / 180
  -- We need to prove that this is equal to -10π + 7π/4
  -- and that 7π/4 satisfies the conditions for α
  sorry

#check angle_conversion (-1485)

end angle_conversion_l3856_385622


namespace range_of_a_l3856_385613

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {-1, 0, 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : A a ∩ B = {0, 1} → a ∈ Set.Icc (-1) 0 ∧ a ≠ 0 := by
  sorry

end range_of_a_l3856_385613


namespace game_score_invariant_final_score_difference_l3856_385641

def game_score (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem game_score_invariant (n : ℕ) (h : n ≥ 2) :
  ∀ (moves : List (ℕ × ℕ × ℕ)),
    moves.all (λ (a, b, c) => a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 1 ∧ b + c = a) →
    moves.foldl (λ acc (a, b, c) => acc + b * c) 0 = game_score n :=
  sorry

theorem final_score_difference (n : ℕ) (h : n ≥ 2) :
  let M := game_score n
  let m := game_score n
  M - m = 0 :=
  sorry

end game_score_invariant_final_score_difference_l3856_385641


namespace second_mission_duration_l3856_385614

def planned_duration : ℕ := 5
def actual_duration_increase : ℚ := 60 / 100
def total_mission_time : ℕ := 11

theorem second_mission_duration :
  let actual_first_mission := planned_duration + (planned_duration * actual_duration_increase).floor
  let second_mission := total_mission_time - actual_first_mission
  second_mission = 3 := by
  sorry

end second_mission_duration_l3856_385614


namespace black_coverage_probability_theorem_l3856_385687

/-- Represents the square with black regions -/
structure ColoredSquare where
  side_length : ℝ
  triangle_leg : ℝ
  diamond_side : ℝ

/-- Represents the circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin covering part of the black region -/
def black_coverage_probability (square : ColoredSquare) (coin : Coin) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem black_coverage_probability_theorem (square : ColoredSquare) (coin : Coin) :
  square.side_length = 10 ∧
  square.triangle_leg = 3 ∧
  square.diamond_side = 3 * Real.sqrt 2 ∧
  coin.diameter = 2 →
  black_coverage_probability square coin = (48 + 12 * Real.sqrt 2 + 2 * Real.pi) / 100 :=
sorry

end black_coverage_probability_theorem_l3856_385687


namespace factor_implies_a_value_l3856_385686

theorem factor_implies_a_value (a b : ℤ) :
  (∀ x : ℝ, (x^2 - x - 1 = 0) → (a*x^19 + b*x^18 + 1 = 0)) →
  a = 1597 := by
  sorry

end factor_implies_a_value_l3856_385686


namespace xy_value_l3856_385610

theorem xy_value (x y : ℝ) (h : x^2 + y^2 + 4*x - 6*y + 13 = 0) : x^y = -8 := by
  sorry

end xy_value_l3856_385610


namespace building_height_from_shadows_l3856_385618

/-- Given a flagstaff and a building casting shadows under similar conditions,
    calculate the height of the building using the concept of similar triangles. -/
theorem building_height_from_shadows
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagstaff_height : flagstaff_height = 17.5)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  ∃ (building_height : ℝ),
    (building_height / building_shadow = flagstaff_height / flagstaff_shadow) ∧
    (abs (building_height - 12.44) < 0.01) :=
sorry

end building_height_from_shadows_l3856_385618


namespace max_value_ab_l3856_385635

theorem max_value_ab (a b : ℝ) (h : ∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) :
  a * b ≤ Real.exp 3 / 2 := by
  sorry

end max_value_ab_l3856_385635


namespace triangle_area_is_63_l3856_385662

/-- The area of a triangle formed by three lines -/
def triangleArea (m1 m2 : ℚ) : ℚ :=
  let x1 : ℚ := 1
  let y1 : ℚ := 1
  let x2 : ℚ := (14/5)
  let y2 : ℚ := (23/5)
  let x3 : ℚ := (11/2)
  let y3 : ℚ := (5/2)
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The theorem stating that the area of the triangle is 6.3 -/
theorem triangle_area_is_63 :
  triangleArea (3/2) (1/3) = 63/10 := by sorry

end triangle_area_is_63_l3856_385662


namespace simplify_fraction_l3856_385689

theorem simplify_fraction (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a + 2*b ≠ 0) :
  (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) - 2 = -a / (a + b) := by
  sorry

end simplify_fraction_l3856_385689


namespace eunji_exam_result_l3856_385632

def exam_problem (exam_a_total exam_b_total exam_a_wrong exam_b_extra_wrong : ℕ) : Prop :=
  let exam_a_right := exam_a_total - exam_a_wrong
  let exam_b_wrong := exam_a_wrong + exam_b_extra_wrong
  let exam_b_right := exam_b_total - exam_b_wrong
  exam_a_right + exam_b_right = 9

theorem eunji_exam_result :
  exam_problem 12 15 8 2 := by
  sorry

end eunji_exam_result_l3856_385632


namespace utopia_national_park_elephant_rate_l3856_385667

/-- Proves that the rate of new elephants entering Utopia National Park is 1500 per hour --/
theorem utopia_national_park_elephant_rate : 
  let initial_elephants : ℕ := 30000
  let exodus_duration : ℕ := 4
  let exodus_rate : ℕ := 2880
  let new_elephants_duration : ℕ := 7
  let final_elephants : ℕ := 28980
  
  let elephants_after_exodus := initial_elephants - exodus_duration * exodus_rate
  let new_elephants := final_elephants - elephants_after_exodus
  let new_elephants_rate := new_elephants / new_elephants_duration
  
  new_elephants_rate = 1500 := by
  sorry

end utopia_national_park_elephant_rate_l3856_385667


namespace cone_base_radius_l3856_385637

/-- Proves that a cone with a lateral surface made from a sector of a circle
    with radius 9 cm and central angle 240° has a circular base with radius 6 cm. -/
theorem cone_base_radius (r : ℝ) (θ : ℝ) (h1 : r = 9) (h2 : θ = 240 * π / 180) :
  r * θ / (2 * π) = 6 := by
  sorry

end cone_base_radius_l3856_385637


namespace rectangle_max_area_l3856_385633

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) →
  l * w = 100 :=
by sorry

end rectangle_max_area_l3856_385633


namespace books_for_sale_l3856_385634

/-- The total number of books for sale is the sum of initial books and additional books found. -/
theorem books_for_sale (initial_books additional_books : ℕ) :
  initial_books = 33 → additional_books = 26 →
  initial_books + additional_books = 59 := by
  sorry

end books_for_sale_l3856_385634


namespace calculation_proof_l3856_385694

theorem calculation_proof : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end calculation_proof_l3856_385694


namespace ellipse_foci_y_axis_l3856_385655

/-- An ellipse with foci on the y-axis represented by the equation x²/a - y²/b = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b < 0 ∧ -b > a

theorem ellipse_foci_y_axis (e : Ellipse) : Real.sqrt (-e.b) > Real.sqrt e.a := by
  sorry

end ellipse_foci_y_axis_l3856_385655


namespace quadratic_equation_roots_l3856_385631

theorem quadratic_equation_roots (m : ℤ) :
  (∃ a b : ℕ+, a ≠ b ∧ 
    (a : ℝ)^2 + m * (a : ℝ) - m + 1 = 0 ∧
    (b : ℝ)^2 + m * (b : ℝ) - m + 1 = 0) →
  m = -5 := by
sorry

end quadratic_equation_roots_l3856_385631


namespace mini_crossword_probability_l3856_385643

/-- Represents a crossword puzzle -/
structure Crossword :=
  (size : Nat)
  (num_clues : Nat)
  (prob_know_clue : ℚ)

/-- Calculates the probability of filling in all unshaded squares in a crossword -/
def probability_fill_crossword (c : Crossword) : ℚ :=
  sorry

/-- The specific crossword from the problem -/
def mini_crossword : Crossword :=
  { size := 5
  , num_clues := 10
  , prob_know_clue := 1/2
  }

/-- Theorem stating the probability of filling in all unshaded squares in the mini crossword -/
theorem mini_crossword_probability :
  probability_fill_crossword mini_crossword = 11/128 :=
sorry

end mini_crossword_probability_l3856_385643


namespace piece_sequence_properties_l3856_385688

/-- Represents the number of small squares in the nth piece -/
def pieceSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the sum of small squares in pieces 1 to n -/
def totalSquares (n : ℕ) : ℕ := n * n

/-- Represents the sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- Represents the sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem piece_sequence_properties :
  (pieceSquares 50 = 99) ∧
  (totalSquares 50 = 2500) ∧
  (sumFirstEvenNumbers 50 = 2550) ∧
  (sumIntegers 100 = 5050) := by
  sorry

end piece_sequence_properties_l3856_385688


namespace total_pets_l3856_385648

theorem total_pets (dogs : ℕ) (fish : ℕ) (cats : ℕ)
  (h1 : dogs = 43)
  (h2 : fish = 72)
  (h3 : cats = 34) :
  dogs + fish + cats = 149 := by
  sorry

end total_pets_l3856_385648


namespace sum_of_differences_l3856_385658

def S : Finset ℕ := Finset.range 11

def pairDifference (i j : ℕ) : ℕ := 
  if i < j then 2^j - 2^i else 2^i - 2^j

def N : ℕ := Finset.sum (S.product S) (fun (p : ℕ × ℕ) => pairDifference p.1 p.2)

theorem sum_of_differences : N = 16398 := by
  sorry

end sum_of_differences_l3856_385658


namespace determinant_zero_l3856_385603

theorem determinant_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1, Real.sin (a + b), Real.sin a],
    ![Real.sin (a + b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]
  Matrix.det M = 0 := by
sorry

end determinant_zero_l3856_385603


namespace problem_solution_l3856_385642

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x + f a (2 * x)

noncomputable def g (x : ℝ) : ℝ := f 2 x - f 2 (-x)

theorem problem_solution :
  (∀ a : ℝ, a > 0 → (∃ x : ℝ, F a x = 3) → (∀ y : ℝ, F a y ≥ 3) → a = 6) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → 2 * m + 3 * n = (⨆ x, g x) →
    (∀ p q : ℝ, p > 0 → q > 0 → 1 / p + 2 / (3 * q) ≥ 2) ∧
    (∃ r s : ℝ, r > 0 ∧ s > 0 ∧ 1 / r + 2 / (3 * s) = 2)) :=
by sorry

end problem_solution_l3856_385642


namespace xy_divides_x2_plus_y2_plus_1_l3856_385657

theorem xy_divides_x2_plus_y2_plus_1 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (x * y) ∣ (x^2 + y^2 + 1)) : 
  (x^2 + y^2 + 1) / (x * y) = 3 := by
sorry

end xy_divides_x2_plus_y2_plus_1_l3856_385657


namespace vector_addition_l3856_385647

variable {V : Type*} [AddCommGroup V]

theorem vector_addition (A B C : V) (a b : V) 
  (h1 : B - A = a) (h2 : C - B = b) : C - A = a + b := by
  sorry

end vector_addition_l3856_385647


namespace red_bellies_percentage_l3856_385630

/-- Represents the total number of minnows in the pond -/
def total_minnows : ℕ := 50

/-- Represents the number of minnows with red bellies -/
def red_bellies : ℕ := 20

/-- Represents the number of minnows with white bellies -/
def white_bellies : ℕ := 15

/-- Represents the percentage of minnows with green bellies -/
def green_bellies_percent : ℚ := 30 / 100

/-- Theorem stating that the percentage of minnows with red bellies is 40% -/
theorem red_bellies_percentage :
  (red_bellies : ℚ) / total_minnows * 100 = 40 := by
  sorry

/-- Lemma verifying that the total number of minnows is correct -/
lemma total_minnows_check :
  total_minnows = red_bellies + white_bellies + (green_bellies_percent * total_minnows) := by
  sorry

end red_bellies_percentage_l3856_385630


namespace square_inequality_l3856_385673

theorem square_inequality (a b : ℝ) : a > |b| → a^2 > b^2 := by sorry

end square_inequality_l3856_385673


namespace mango_cost_theorem_l3856_385626

/-- The cost of mangoes in dollars per pound -/
def cost_per_pound (total_cost : ℚ) (total_pounds : ℚ) : ℚ :=
  total_cost / total_pounds

/-- The cost of a given weight of mangoes in dollars -/
def cost_of_weight (cost_per_pound : ℚ) (weight : ℚ) : ℚ :=
  cost_per_pound * weight

theorem mango_cost_theorem (total_cost : ℚ) (total_pounds : ℚ) 
  (h : total_cost = 12 ∧ total_pounds = 10) : 
  cost_of_weight (cost_per_pound total_cost total_pounds) (1/2) = 0.6 := by
  sorry

#eval cost_of_weight (cost_per_pound 12 10) (1/2)

end mango_cost_theorem_l3856_385626


namespace five_fridays_in_august_l3856_385666

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to count occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem stating that if July has five Tuesdays, August will have five Fridays -/
theorem five_fridays_in_august 
  (july : Month) 
  (august : Month) 
  (h1 : july.days = 31) 
  (h2 : august.days = 31) 
  (h3 : countDayOccurrences july DayOfWeek.Tuesday = 5) :
  countDayOccurrences august DayOfWeek.Friday = 5 :=
sorry

end five_fridays_in_august_l3856_385666


namespace train_length_proof_l3856_385650

theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 ∧ crossing_time = 36 ∧ train_speed = 40 →
  train_speed * crossing_time - bridge_length = 1140 := by
  sorry

end train_length_proof_l3856_385650


namespace temperature_conversion_l3856_385617

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 35 → k = 95 := by
  sorry

end temperature_conversion_l3856_385617


namespace cyclist_journey_solution_l3856_385693

/-- Represents the cyclist's journey with flat, uphill, and downhill segments -/
structure CyclistJourney where
  flat : ℝ
  uphill : ℝ
  downhill : ℝ

/-- Checks if the given journey satisfies all conditions -/
def is_valid_journey (j : CyclistJourney) : Prop :=
  -- Total distance is 80 km
  j.flat + j.uphill + j.downhill = 80 ∧
  -- Forward journey time (47/12 hours)
  j.flat / 21 + j.uphill / 12 + j.downhill / 30 = 47 / 12 ∧
  -- Return journey time (14/3 hours)
  j.flat / 21 + j.uphill / 30 + j.downhill / 12 = 14 / 3

/-- The theorem stating the correct lengths of the journey segments -/
theorem cyclist_journey_solution :
  ∃ (j : CyclistJourney), is_valid_journey j ∧ j.flat = 35 ∧ j.uphill = 15 ∧ j.downhill = 30 :=
by
  sorry


end cyclist_journey_solution_l3856_385693


namespace mirror_position_l3856_385620

theorem mirror_position (wall_width mirror_width : ℝ) (h1 : wall_width = 26) (h2 : mirror_width = 4) :
  let distance := (wall_width - mirror_width) / 2
  distance = 11 := by sorry

end mirror_position_l3856_385620


namespace secret_society_friendships_l3856_385671

/-- Represents a member of the secret society -/
structure Member where
  balance : Int

/-- Represents the secret society -/
structure SecretSociety where
  members : Finset Member
  friendships : Finset (Member × Member)
  
/-- A function that represents giving one dollar to all friends -/
def giveDollarToFriends (s : SecretSociety) (m : Member) : SecretSociety :=
  sorry

/-- A predicate that checks if money can be arbitrarily redistributed -/
def canRedistributeArbitrarily (s : SecretSociety) : Prop :=
  sorry

theorem secret_society_friendships 
  (s : SecretSociety) 
  (h1 : s.members.card = 2011) 
  (h2 : canRedistributeArbitrarily s) : 
  s.friendships.card = 2010 :=
sorry

end secret_society_friendships_l3856_385671


namespace unique_sum_of_squares_l3856_385664

theorem unique_sum_of_squares (p q r : ℕ+) : 
  p + q + r = 30 →
  Nat.gcd p.val q.val + Nat.gcd q.val r.val + Nat.gcd r.val p.val = 10 →
  p ^ 2 + q ^ 2 + r ^ 2 = 584 := by
  sorry

end unique_sum_of_squares_l3856_385664


namespace daniels_age_l3856_385663

theorem daniels_age (emily_age : ℕ) (brianna_age : ℕ) (daniel_age : ℕ) : 
  emily_age = 48 →
  brianna_age = emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age * 2 = brianna_age →
  daniel_age = 13 := by
sorry

end daniels_age_l3856_385663


namespace operation_result_l3856_385606

def at_op (a b : ℝ) : ℝ := a * b - b^2 + b^3

def hash_op (a b : ℝ) : ℝ := a + b - a * b^2 + a * b^3

theorem operation_result : (at_op 7 3) / (hash_op 7 3) = 39 / 136 := by
  sorry

end operation_result_l3856_385606


namespace sine_symmetry_axis_l3856_385690

/-- The symmetry axis of the graph of y = sin(x - π/3) is x = -π/6 -/
theorem sine_symmetry_axis :
  ∀ x : ℝ, (∀ y : ℝ, y = Real.sin (x - π/3)) →
  (∃ k : ℤ, x = -π/6 + k * π) :=
sorry

end sine_symmetry_axis_l3856_385690


namespace cindy_envelopes_left_l3856_385697

/-- Calculates the number of envelopes Cindy has left after giving some to her friends -/
def envelopes_left (initial : ℕ) (friends : ℕ) (per_friend : ℕ) : ℕ :=
  initial - friends * per_friend

/-- Proves that Cindy has 22 envelopes left -/
theorem cindy_envelopes_left : 
  envelopes_left 37 5 3 = 22 := by
  sorry

end cindy_envelopes_left_l3856_385697


namespace range_of_function_l3856_385677

open Real

theorem range_of_function (f : ℝ → ℝ) (x : ℝ) :
  (f = fun x ↦ sin (x/2) * cos (x/2) + cos (x/2)^2) →
  (x ∈ Set.Ioo 0 (π/2)) →
  ∃ y, y ∈ Set.Ioc (1/2) ((Real.sqrt 2 + 1)/2) ∧ ∃ x, f x = y ∧
  ∀ z, (∃ x, f x = z) → z ∈ Set.Ioc (1/2) ((Real.sqrt 2 + 1)/2) :=
by sorry

end range_of_function_l3856_385677


namespace sentence_A_most_appropriate_l3856_385675

/-- Represents a sentence to be evaluated for appropriateness --/
inductive Sentence
| A
| B
| C
| D

/-- Criteria for evaluating the appropriateness of a sentence --/
structure EvaluationCriteria :=
  (identity : Bool)
  (status : Bool)
  (occasion : Bool)
  (audience : Bool)
  (purpose : Bool)
  (respectfulLanguage : Bool)
  (toneOfDiscourse : Bool)

/-- Evaluates a sentence based on the given criteria --/
def evaluateSentence (s : Sentence) (c : EvaluationCriteria) : Bool :=
  match s with
  | Sentence.A => c.identity ∧ c.status ∧ c.occasion ∧ c.audience ∧ c.purpose ∧ c.respectfulLanguage ∧ c.toneOfDiscourse
  | Sentence.B => false
  | Sentence.C => false
  | Sentence.D => false

/-- The criteria used for evaluation --/
def criteria : EvaluationCriteria :=
  { identity := true
  , status := true
  , occasion := true
  , audience := true
  , purpose := true
  , respectfulLanguage := true
  , toneOfDiscourse := true }

/-- Theorem stating that sentence A is the most appropriate --/
theorem sentence_A_most_appropriate :
  ∀ s : Sentence, s ≠ Sentence.A → ¬(evaluateSentence s criteria) ∧ evaluateSentence Sentence.A criteria :=
sorry

end sentence_A_most_appropriate_l3856_385675


namespace siamese_twins_case_l3856_385640

/-- Represents a person on trial --/
structure Defendant where
  guilty : Bool

/-- Represents a pair of defendants --/
structure DefendantPair where
  defendant1 : Defendant
  defendant2 : Defendant
  areConjoined : Bool

/-- Represents the judge's decision --/
def judgeDecision (pair : DefendantPair) : Bool :=
  pair.defendant1.guilty ≠ pair.defendant2.guilty → 
  (pair.defendant1.guilty ∨ pair.defendant2.guilty) → 
  pair.areConjoined

theorem siamese_twins_case (pair : DefendantPair) :
  pair.defendant1.guilty ≠ pair.defendant2.guilty →
  (pair.defendant1.guilty ∨ pair.defendant2.guilty) →
  judgeDecision pair →
  pair.areConjoined := by
  sorry


end siamese_twins_case_l3856_385640


namespace original_plan_calculation_l3856_385685

def thursday_sales : ℕ := 210
def friday_sales : ℕ := 2 * thursday_sales
def saturday_sales : ℕ := 130
def sunday_sales : ℕ := saturday_sales / 2
def excess_sales : ℕ := 325

def total_sales : ℕ := thursday_sales + friday_sales + saturday_sales + sunday_sales

theorem original_plan_calculation :
  total_sales - excess_sales = 500 := by sorry

end original_plan_calculation_l3856_385685


namespace train_speed_calculation_l3856_385695

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 265 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l3856_385695


namespace blocks_used_for_tower_l3856_385698

theorem blocks_used_for_tower (initial_blocks : ℕ) (remaining_blocks : ℕ) : 
  initial_blocks = 97 → remaining_blocks = 72 → initial_blocks - remaining_blocks = 25 := by
  sorry

end blocks_used_for_tower_l3856_385698


namespace age_difference_l3856_385661

theorem age_difference (A B : ℕ) : 
  B = 39 → 
  A + 10 = 2 * (B - 10) → 
  A - B = 9 := by
sorry

end age_difference_l3856_385661


namespace max_square_pen_area_l3856_385608

def fencing_length : ℝ := 64

def square_pen_area (side_length : ℝ) : ℝ := side_length ^ 2

def perimeter_constraint (side_length : ℝ) : Prop := 4 * side_length = fencing_length

theorem max_square_pen_area :
  ∃ (side_length : ℝ), perimeter_constraint side_length ∧
    ∀ (x : ℝ), perimeter_constraint x → square_pen_area x ≤ square_pen_area side_length ∧
    square_pen_area side_length = 256 :=
  sorry

end max_square_pen_area_l3856_385608


namespace baker_cakes_sold_l3856_385674

theorem baker_cakes_sold (bought : ℕ) (difference : ℕ) (sold : ℕ) : 
  bought = 154 → difference = 63 → bought = sold + difference → sold = 91 := by
  sorry

end baker_cakes_sold_l3856_385674


namespace park_problem_solution_l3856_385645

/-- The problem setup -/
structure ParkProblem where
  distance_to_park : ℝ
  mother_speed_ratio : ℝ
  time_difference : ℝ
  distance_to_company : ℝ
  mother_run_speed : ℝ
  available_time : ℝ

/-- The solution to be proved -/
structure ParkSolution where
  mother_speed : ℝ
  min_run_time : ℝ

/-- The main theorem to be proved -/
theorem park_problem_solution (p : ParkProblem) 
  (h1 : p.distance_to_park = 4320)
  (h2 : p.mother_speed_ratio = 1.2)
  (h3 : p.time_difference = 12)
  (h4 : p.distance_to_company = 2940)
  (h5 : p.mother_run_speed = 150)
  (h6 : p.available_time = 30) :
  ∃ (s : ParkSolution), 
    s.mother_speed = 72 ∧ 
    s.min_run_time = 10 ∧
    (p.distance_to_park / s.mother_speed - p.distance_to_park / (s.mother_speed / p.mother_speed_ratio) = p.time_difference) ∧
    ((p.distance_to_company - p.mother_run_speed * s.min_run_time) / s.mother_speed + s.min_run_time ≤ p.available_time) := by
  sorry

end park_problem_solution_l3856_385645


namespace runners_meet_time_l3856_385684

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ

/-- Represents the circular track -/
structure Track where
  length : ℝ

/-- Calculates the time when all runners meet again -/
def meeting_time (track : Track) (runners : List Runner) : ℝ :=
  sorry

theorem runners_meet_time (track : Track) (runners : List Runner) :
  track.length = 600 ∧
  runners = [
    Runner.mk 4.5,
    Runner.mk 4.9,
    Runner.mk 5.1
  ] →
  meeting_time track runners = 3000 := by
  sorry

end runners_meet_time_l3856_385684


namespace valentines_to_cinco_l3856_385625

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

def valentinesDay : Date := ⟨2, 14⟩
def cincoMayo : Date := ⟨5, 5⟩

/-- Given that February 14 is a Tuesday, calculate the day of the week for any date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Calculate the number of days between two dates, inclusive -/
def daysBetween (d1 d2 : Date) : Nat := sorry

theorem valentines_to_cinco : 
  dayOfWeek valentinesDay = DayOfWeek.Tuesday →
  (dayOfWeek cincoMayo = DayOfWeek.Friday ∧ 
   daysBetween valentinesDay cincoMayo = 81) := by
  sorry

end valentines_to_cinco_l3856_385625


namespace problem_solution_problem_solution_2_l3856_385638

def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 :=
sorry

theorem problem_solution_2 :
  ∃ S : Set ℝ, S = Set.Icc (-4) (-1) ∪ Set.Ioc 5 6 ∧
  ∀ x, x ∈ S ↔ (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) :=
sorry

end problem_solution_problem_solution_2_l3856_385638


namespace parabola_vertex_l3856_385678

/-- The vertex of the parabola y = 2x^2 - 4x - 7 is at the point (1, -9). -/
theorem parabola_vertex (x y : ℝ) : y = 2 * x^2 - 4 * x - 7 → (1, -9) = (x, y) := by
  sorry

end parabola_vertex_l3856_385678


namespace scott_cake_sales_l3856_385612

theorem scott_cake_sales (smoothie_price : ℕ) (cake_price : ℕ) (smoothies_sold : ℕ) (total_revenue : ℕ) :
  smoothie_price = 3 →
  cake_price = 2 →
  smoothies_sold = 40 →
  total_revenue = 156 →
  ∃ (cakes_sold : ℕ), smoothie_price * smoothies_sold + cake_price * cakes_sold = total_revenue ∧ cakes_sold = 18 := by
  sorry

end scott_cake_sales_l3856_385612


namespace arctan_tan_difference_l3856_385628

theorem arctan_tan_difference (θ : Real) :
  0 ≤ θ ∧ θ ≤ π →
  Real.arctan (Real.tan (5 * π / 12) - 3 * Real.tan (π / 12)) = 3 * π / 4 := by
  sorry

end arctan_tan_difference_l3856_385628


namespace triangle_is_right_angled_l3856_385656

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if b² = c² + a² - ca and sin A = 2 sin C, then the triangle is right-angled. -/
theorem triangle_is_right_angled 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 = c^2 + a^2 - c*a) 
  (h2 : Real.sin A = 2 * Real.sin C) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h5 : A + B + C = Real.pi) : 
  ∃ (X : ℝ), (X = A ∨ X = B ∨ X = C) ∧ X = Real.pi / 2 := by
  sorry


end triangle_is_right_angled_l3856_385656


namespace parabola_vertex_on_x_axis_l3856_385604

theorem parabola_vertex_on_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 - (a + 2)*x + 9 = 0 ∧ 
   ∀ y : ℝ, y^2 - (a + 2)*y + 9 ≥ x^2 - (a + 2)*x + 9) →
  a = 4 ∨ a = -8 :=
by sorry

end parabola_vertex_on_x_axis_l3856_385604


namespace faster_train_speed_l3856_385692

theorem faster_train_speed 
  (train_length : ℝ) 
  (slower_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 80) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 36 / 3600) : 
  ∃ (faster_speed : ℝ), faster_speed = 52 := by
  sorry

end faster_train_speed_l3856_385692


namespace student_weight_loss_l3856_385669

/-- The amount of weight a student needs to lose to weigh twice as much as his sister. -/
def weight_to_lose (total_weight sister_weight : ℝ) : ℝ :=
  total_weight - sister_weight - 2 * sister_weight

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 104)
  (h2 : student_weight = 71) :
  weight_to_lose total_weight (total_weight - student_weight) = 5 := by
  sorry

#eval weight_to_lose 104 33

end student_weight_loss_l3856_385669


namespace jan_skipping_speed_ratio_jan_skipping_speed_ratio_is_two_l3856_385623

/-- The ratio of Jan's skipping speed after training to her speed before training -/
theorem jan_skipping_speed_ratio : ℝ :=
  let speed_before : ℝ := 70  -- skips per minute
  let total_skips_after : ℝ := 700
  let total_minutes_after : ℝ := 5
  let speed_after : ℝ := total_skips_after / total_minutes_after
  speed_after / speed_before

/-- Proof that the ratio of Jan's skipping speed after training to her speed before training is 2 -/
theorem jan_skipping_speed_ratio_is_two :
  jan_skipping_speed_ratio = 2 := by
  sorry

end jan_skipping_speed_ratio_jan_skipping_speed_ratio_is_two_l3856_385623


namespace football_competition_kicks_l3856_385627

/-- Calculates the number of penalty kicks required for a football competition --/
def penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : ℕ :=
  goalkeepers * (total_players - 1)

/-- Theorem: Given 24 players with 4 goalkeepers, 92 penalty kicks are required --/
theorem football_competition_kicks : penalty_kicks 24 4 = 92 := by
  sorry

end football_competition_kicks_l3856_385627


namespace intersection_points_l3856_385683

-- Define the equations
def eq1 (x y : ℝ) : Prop := 4 + (x + 2) * y = x^2
def eq2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 16

-- Theorem stating the intersection points
theorem intersection_points :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1 = -2 ∧ y1 = -4) ∧
    (x2 = -2 ∧ y2 = 4) ∧
    (x3 = 2 ∧ y3 = 0) ∧
    eq1 x1 y1 ∧ eq2 x1 y1 ∧
    eq1 x2 y2 ∧ eq2 x2 y2 ∧
    eq1 x3 y3 ∧ eq2 x3 y3 := by
  sorry

end intersection_points_l3856_385683


namespace sufficient_but_not_necessary_l3856_385636

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x > 0) ∧ (∃ x, x > 0 ∧ ¬(x > 1)) := by
  sorry

end sufficient_but_not_necessary_l3856_385636


namespace slope_divides_area_in_half_l3856_385644

/-- L-shaped region in the xy-plane -/
structure LShapedRegion where
  vertices : List (ℝ × ℝ)
  is_l_shaped : vertices = [(0,0), (0,4), (4,4), (4,2), (6,2), (6,0)]

/-- Line passing through the origin -/
structure LineFromOrigin where
  slope : ℝ

/-- Function to calculate the area of the L-shaped region -/
def area (r : LShapedRegion) : ℝ :=
  20 -- The total area of the L-shaped region

/-- Function to calculate the area divided by a line -/
def area_divided_by_line (r : LShapedRegion) (l : LineFromOrigin) : ℝ × ℝ :=
  sorry -- Returns a pair of areas divided by the line

/-- Theorem stating that the slope 1/2 divides the area in half -/
theorem slope_divides_area_in_half (r : LShapedRegion) :
  let l := LineFromOrigin.mk (1/2)
  let (area1, area2) := area_divided_by_line r l
  area1 = area2 ∧ area1 + area2 = area r :=
sorry

end slope_divides_area_in_half_l3856_385644


namespace line_parallel_perpendicular_plane_l3856_385665

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_plane 
  (m n : Line) (α : Plane) :
  parallel_line_plane m α → 
  perpendicular_line_plane n α → 
  perpendicular_line_line m n :=
sorry

end line_parallel_perpendicular_plane_l3856_385665


namespace prob_sum_gt_15_eq_5_108_l3856_385649

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The minimum possible sum when rolling three dice -/
def min_sum : ℕ := 3

/-- The maximum possible sum when rolling three dice -/
def max_sum : ℕ := 18

/-- The total number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := num_faces ^ 3

/-- The number of favorable outcomes (sum > 15) -/
def favorable_outcomes : ℕ := 10

/-- The probability of rolling three dice and getting a sum greater than 15 -/
def prob_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_gt_15_eq_5_108 : prob_sum_gt_15 = 5 / 108 := by
  sorry

end prob_sum_gt_15_eq_5_108_l3856_385649


namespace sum_of_z_values_l3856_385619

theorem sum_of_z_values (f : ℝ → ℝ) (h : ∀ x, f (x / 3) = x^2 + x + 1) :
  let z₁ := (2 : ℝ) / 9
  let z₂ := -(1 : ℝ) / 3
  (f (3 * z₁) = 7 ∧ f (3 * z₂) = 7) ∧ z₁ + z₂ = -(1 : ℝ) / 9 :=
by sorry

end sum_of_z_values_l3856_385619


namespace percentage_problem_l3856_385621

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 4) : x = 20 := by
  sorry

end percentage_problem_l3856_385621


namespace difference_of_equal_distinct_prime_factors_l3856_385682

def distinctPrimeFactors (n : ℕ) : Finset ℕ :=
  sorry

theorem difference_of_equal_distinct_prime_factors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ (distinctPrimeFactors a).card = (distinctPrimeFactors b).card :=
sorry

end difference_of_equal_distinct_prime_factors_l3856_385682


namespace lukes_coin_piles_l3856_385668

/-- Given that Luke has an equal number of piles of quarters and dimes,
    each pile contains 3 coins, and the total number of coins is 30,
    prove that the number of piles of quarters is 5. -/
theorem lukes_coin_piles (num_quarter_piles num_dime_piles : ℕ)
  (h1 : num_quarter_piles = num_dime_piles)
  (h2 : ∀ pile, pile = num_quarter_piles ∨ pile = num_dime_piles → 3 * pile = num_quarter_piles * 3 + num_dime_piles * 3)
  (h3 : num_quarter_piles * 3 + num_dime_piles * 3 = 30) :
  num_quarter_piles = 5 := by
  sorry

end lukes_coin_piles_l3856_385668


namespace max_point_range_l3856_385605

/-- Given a differentiable function f : ℝ → ℝ and a real number a, 
    if f'(x) = a(x-1)(x-a) for all x and f attains a maximum at x = a, 
    then 0 < a < 1 -/
theorem max_point_range (f : ℝ → ℝ) (a : ℝ) 
    (h1 : Differentiable ℝ f) 
    (h2 : ∀ x, deriv f x = a * (x - 1) * (x - a))
    (h3 : IsLocalMax f a) : 
    0 < a ∧ a < 1 := by
  sorry


end max_point_range_l3856_385605


namespace tan_series_equality_l3856_385615

theorem tan_series_equality (x : ℝ) (h : |Real.tan x| < 1) :
  8.407 * ((1 - Real.tan x)⁻¹) / ((1 + Real.tan x)⁻¹) = 1 + Real.sin (2 * x) ↔
  ∃ k : ℤ, x = k * Real.pi := by sorry

end tan_series_equality_l3856_385615


namespace cyclic_sum_inequality_l3856_385659

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z + y * z / x + z * x / y) > 2 * (x^3 + y^3 + z^3)^(1/3) := by
  sorry

end cyclic_sum_inequality_l3856_385659


namespace exists_non_intersecting_line_l3856_385609

/-- Represents a domino on a grid -/
structure Domino where
  x1 : Nat
  y1 : Nat
  x2 : Nat
  y2 : Nat

/-- Represents a 6x6 grid covered by dominoes -/
structure DominoGrid where
  dominoes : List Domino
  domino_count : dominoes.length = 18
  covers_grid : ∀ x y, x < 6 ∧ y < 6 → ∃ d ∈ dominoes, 
    ((d.x1 = x ∧ d.y1 = y) ∨ (d.x2 = x ∧ d.y2 = y))
  valid_dominoes : ∀ d ∈ dominoes, 
    (d.x1 = d.x2 ∧ d.y2 = d.y1 + 1) ∨ (d.y1 = d.y2 ∧ d.x2 = d.x1 + 1)

/-- Main theorem: There exists a grid line not intersecting any domino -/
theorem exists_non_intersecting_line (grid : DominoGrid) :
  (∃ x : Nat, x < 5 ∧ ∀ d ∈ grid.dominoes, d.x1 ≠ x + 1 ∨ d.x2 ≠ x + 1) ∨
  (∃ y : Nat, y < 5 ∧ ∀ d ∈ grid.dominoes, d.y1 ≠ y + 1 ∨ d.y2 ≠ y + 1) :=
sorry

end exists_non_intersecting_line_l3856_385609


namespace fraction_product_theorem_l3856_385646

/-- The type of fractions in the sequence -/
def Fraction (n : ℕ) := { k : ℕ // 2 ≤ k ∧ k ≤ n }

/-- The sequence of fractions -/
def fractionSequence (n : ℕ) : List (Fraction n) :=
  List.range (n - 1) |>.map (fun i => ⟨i + 2, by sorry⟩)

/-- The product of the original sequence of fractions -/
def originalProduct (n : ℕ) : ℚ :=
  (fractionSequence n).foldl (fun acc f => acc * (f.val : ℚ) / ((f.val - 1) : ℚ)) 1

/-- A function that determines whether a fraction should be reciprocated -/
def reciprocate (n : ℕ) : Fraction n → Bool := sorry

/-- The product after reciprocating some fractions -/
def modifiedProduct (n : ℕ) : ℚ :=
  (fractionSequence n).foldl
    (fun acc f => 
      if reciprocate n f
      then acc * ((f.val - 1) : ℚ) / (f.val : ℚ)
      else acc * (f.val : ℚ) / ((f.val - 1) : ℚ))
    1

/-- The main theorem -/
theorem fraction_product_theorem (n : ℕ) (h : n > 2) :
  (∃ (reciprocate : Fraction n → Bool), modifiedProduct n = 1) ↔ ∃ (a : ℕ), n = a^2 ∧ a > 1 := by
  sorry

end fraction_product_theorem_l3856_385646


namespace shifted_sine_value_l3856_385651

theorem shifted_sine_value (g f : ℝ → ℝ) :
  (∀ x, g x = Real.sin (x - π/6)) →
  (∀ x, f x = g (x - π/6)) →
  f (π/6) = -1/2 := by
  sorry

end shifted_sine_value_l3856_385651


namespace parabola_symmetry_axis_l3856_385601

/-- The axis of symmetry of a parabola y^2 = mx has the equation x = -m/4 -/
def axis_of_symmetry (m : ℝ) : ℝ → Prop :=
  fun x ↦ x = -m/4

/-- A point (x, y) lies on the parabola y^2 = mx -/
def on_parabola (m : ℝ) : ℝ × ℝ → Prop :=
  fun p ↦ p.2^2 = m * p.1

theorem parabola_symmetry_axis (m : ℝ) :
  axis_of_symmetry m (-m^2) →
  on_parabola m (-m^2, 3) →
  m = 1/4 := by
  sorry

end parabola_symmetry_axis_l3856_385601


namespace f_inequality_solution_comparison_theorem_l3856_385611

def f (x : ℝ) : ℝ := -abs x - abs (x + 2)

theorem f_inequality_solution (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 := by sorry

theorem comparison_theorem (x a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = Real.sqrt 5) :
  a^2 + b^2/4 ≥ f x + 3 := by sorry

end f_inequality_solution_comparison_theorem_l3856_385611


namespace display_rows_for_225_cans_l3856_385696

/-- Represents a pyramidal display of cans -/
structure CanDisplay where
  rows : ℕ

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 4 * n - 3

/-- The total number of cans in the display -/
def total_cans (d : CanDisplay) : ℕ :=
  (d.rows * (cans_in_row 1 + cans_in_row d.rows)) / 2

/-- The theorem stating that a display with 225 cans has 11 rows -/
theorem display_rows_for_225_cans :
  ∃ (d : CanDisplay), total_cans d = 225 ∧ d.rows = 11 :=
by sorry

end display_rows_for_225_cans_l3856_385696


namespace equation_pattern_l3856_385681

theorem equation_pattern (n : ℕ) (hn : n ≥ 1) :
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end equation_pattern_l3856_385681


namespace necessary_not_sufficient_l3856_385653

/-- The function f(x) = x - a --/
def f (a : ℝ) (x : ℝ) : ℝ := x - a

/-- The open interval (0, 1) --/
def open_unit_interval : Set ℝ := { x | 0 < x ∧ x < 1 }

/-- f has a zero in (0, 1) --/
def has_zero_in_unit_interval (a : ℝ) : Prop :=
  ∃ x ∈ open_unit_interval, f a x = 0

theorem necessary_not_sufficient :
  (∀ a : ℝ, has_zero_in_unit_interval a → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ ¬has_zero_in_unit_interval a) := by sorry

end necessary_not_sufficient_l3856_385653


namespace moving_sidewalk_speed_l3856_385602

/-- The speed of a moving sidewalk given a child's running parameters -/
theorem moving_sidewalk_speed
  (child_speed : ℝ)
  (with_distance : ℝ)
  (with_time : ℝ)
  (against_distance : ℝ)
  (against_time : ℝ)
  (h1 : child_speed = 74)
  (h2 : with_distance = 372)
  (h3 : with_time = 4)
  (h4 : against_distance = 165)
  (h5 : against_time = 3)
  : ∃ (sidewalk_speed : ℝ),
    sidewalk_speed = 19 ∧
    with_distance = (child_speed + sidewalk_speed) * with_time ∧
    against_distance = (child_speed - sidewalk_speed) * against_time :=
by sorry

end moving_sidewalk_speed_l3856_385602


namespace min_female_participants_l3856_385616

/-- Proves the minimum number of female students participating in community work -/
theorem min_female_participants (male_students female_students : ℕ) 
  (total_participants : ℕ) (h1 : male_students = 22) (h2 : female_students = 18) 
  (h3 : total_participants = ((male_students + female_students) * 6) / 10) : 
  ∃ (female_participants : ℕ), 
    female_participants ≥ 2 ∧ 
    female_participants ≤ female_students ∧
    female_participants + male_students ≥ total_participants :=
by sorry

end min_female_participants_l3856_385616


namespace cupcakes_sold_l3856_385679

theorem cupcakes_sold (initial : ℕ) (additional : ℕ) (final : ℕ) : 
  initial = 19 → additional = 10 → final = 24 → 
  initial + additional - final = 5 := by
sorry

end cupcakes_sold_l3856_385679


namespace min_specialists_needed_l3856_385699

/-- Represents the number of specialists in energy efficiency -/
def energy_efficiency : ℕ := 95

/-- Represents the number of specialists in waste management -/
def waste_management : ℕ := 80

/-- Represents the number of specialists in water conservation -/
def water_conservation : ℕ := 110

/-- Represents the number of specialists in both energy efficiency and waste management -/
def energy_waste : ℕ := 30

/-- Represents the number of specialists in both waste management and water conservation -/
def waste_water : ℕ := 35

/-- Represents the number of specialists in both energy efficiency and water conservation -/
def energy_water : ℕ := 25

/-- Represents the number of specialists in all three areas -/
def all_three : ℕ := 15

/-- Theorem stating the minimum number of specialists needed -/
theorem min_specialists_needed : 
  energy_efficiency + waste_management + water_conservation - 
  energy_waste - waste_water - energy_water + all_three = 210 := by
  sorry

end min_specialists_needed_l3856_385699


namespace fraction_inequality_l3856_385624

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end fraction_inequality_l3856_385624


namespace davids_physics_marks_l3856_385629

/-- Given David's marks in various subjects and his average, prove his marks in Physics --/
theorem davids_physics_marks
  (english : ℕ)
  (mathematics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_chemistry : chemistry = 87)
  (h_biology : biology = 95)
  (h_average : average = 89)
  (h_subjects : total_subjects = 5) :
  ∃ (physics : ℕ), physics = 92 ∧
    average * total_subjects = english + mathematics + physics + chemistry + biology :=
by sorry

end davids_physics_marks_l3856_385629


namespace prob_five_dice_three_matching_l3856_385600

/-- The probability of rolling at least three matching dice out of five fair six-sided dice -/
def prob_at_least_three_matching (n : ℕ) (s : ℕ) : ℚ :=
  -- n is the number of dice
  -- s is the number of sides on each die
  sorry

/-- Theorem stating that the probability of rolling at least three matching dice
    out of five fair six-sided dice is equal to 23/108 -/
theorem prob_five_dice_three_matching :
  prob_at_least_three_matching 5 6 = 23 / 108 := by
  sorry

end prob_five_dice_three_matching_l3856_385600


namespace line_perpendicular_sufficient_condition_l3856_385676

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perp : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpToPlane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_sufficient_condition
  (l m : Line) (α β : Plane)
  (h1 : intersect α β = l)
  (h2 : subset m β)
  (h3 : perpToPlane m α) :
  perp l m :=
sorry

end line_perpendicular_sufficient_condition_l3856_385676


namespace inequality_proofs_l3856_385691

theorem inequality_proofs (x : ℝ) : 
  (6 + 3 * x > 30 → x > 8) ∧ 
  (1 - x < 3 - (x - 5) / 2 → x > -9) := by
  sorry

end inequality_proofs_l3856_385691


namespace hotel_flat_fee_l3856_385654

/-- Given a hotel charging a flat fee for the first night and a fixed amount for additional nights,
    prove that the flat fee is $60 if a 4-night stay costs $205 and a 7-night stay costs $350. -/
theorem hotel_flat_fee (flat_fee nightly_fee : ℚ) : 
  (flat_fee + 3 * nightly_fee = 205) →
  (flat_fee + 6 * nightly_fee = 350) →
  flat_fee = 60 := by sorry

end hotel_flat_fee_l3856_385654


namespace rectangle_area_rectangle_area_proof_l3856_385652

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 56) : ℝ :=
  let side_length := perimeter / 8
  let square_area := side_length ^ 2
  3 * square_area

theorem rectangle_area_proof :
  rectangle_area 56 rfl = 147 := by sorry

end rectangle_area_rectangle_area_proof_l3856_385652


namespace basketball_game_score_l3856_385672

theorem basketball_game_score (a r b d : ℕ) : 
  -- Raiders' scores form a geometric sequence
  0 < a ∧ 1 < r ∧ 
  -- Wildcats' scores form an arithmetic sequence
  0 < b ∧ 0 < d ∧ 
  -- Game tied at end of first quarter
  a = b ∧ 
  -- Raiders won by one point
  a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 1 ∧ 
  -- Neither team scored more than 100 points
  a * (1 + r + r^2 + r^3) ≤ 100 ∧ 
  4 * b + 6 * d ≤ 100 →
  -- Total points in first half
  a + a * r + b + (b + d) = 34 := by
sorry

end basketball_game_score_l3856_385672


namespace apple_tree_bearing_time_l3856_385680

def time_to_bear_fruit (age_planted : ℕ) (age_first_apple : ℕ) : ℕ :=
  age_first_apple - age_planted

theorem apple_tree_bearing_time :
  let age_planted : ℕ := 4
  let age_first_apple : ℕ := 11
  time_to_bear_fruit age_planted age_first_apple = 7 := by
sorry

end apple_tree_bearing_time_l3856_385680


namespace total_cost_is_1046_l3856_385670

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 349 / 100

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 87 / 100

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem total_cost_is_1046 : total_cost = 1046 / 100 := by
  sorry

end total_cost_is_1046_l3856_385670
