import Mathlib

namespace NUMINAMATH_CALUDE_michael_chicken_count_l2815_281506

/-- Calculates the number of chickens after a given number of years -/
def chickenCount (initialCount : ℕ) (annualIncrease : ℕ) (years : ℕ) : ℕ :=
  initialCount + annualIncrease * years

/-- Theorem stating that Michael will have 1900 chickens after 9 years -/
theorem michael_chicken_count :
  chickenCount 550 150 9 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_michael_chicken_count_l2815_281506


namespace NUMINAMATH_CALUDE_sum_difference_equals_210_l2815_281562

theorem sum_difference_equals_210 : 152 + 29 + 25 + 14 - 10 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_210_l2815_281562


namespace NUMINAMATH_CALUDE_joan_video_game_spending_l2815_281541

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending : total_spent = 9.43 := by sorry

end NUMINAMATH_CALUDE_joan_video_game_spending_l2815_281541


namespace NUMINAMATH_CALUDE_typing_problem_l2815_281510

/-- Represents the typing speed of a typist in pages per hour -/
structure TypingSpeed :=
  (speed : ℝ)

/-- Represents the length of a chapter in pages -/
structure ChapterLength :=
  (pages : ℝ)

/-- Represents the time taken to type a chapter in hours -/
structure TypingTime :=
  (hours : ℝ)

theorem typing_problem (x y : TypingSpeed) (c1 c2 c3 : ChapterLength) (t1 t2 : TypingTime) :
  -- First chapter is twice as short as the second
  c1.pages = c2.pages / 2 →
  -- First chapter is three times longer than the third
  c1.pages = 3 * c3.pages →
  -- Typists retyped first chapter together in 3 hours and 36 minutes
  t1.hours = 3.6 →
  c1.pages / (x.speed + y.speed) = t1.hours →
  -- Second chapter was retyped in 8 hours
  t2.hours = 8 →
  -- First typist worked alone for 2 hours on second chapter
  2 * x.speed + 6 * (x.speed + y.speed) = c2.pages →
  -- Time for second typist to retype third chapter alone
  c3.pages / y.speed = 3 := by
sorry

end NUMINAMATH_CALUDE_typing_problem_l2815_281510


namespace NUMINAMATH_CALUDE_smallest_all_blue_count_l2815_281595

/-- Represents the colors of chameleons -/
inductive Color
| Red
| C2
| C3
| C4
| Blue

/-- Represents the result of a bite interaction between two chameleons -/
def bite_result (biter bitten : Color) : Color :=
  match biter, bitten with
  | Color.Red, Color.Red => Color.C2
  | Color.Red, Color.C2 => Color.C3
  | Color.Red, Color.C3 => Color.C4
  | Color.Red, Color.C4 => Color.Blue
  | Color.C2, Color.Red => Color.C2
  | Color.C3, Color.Red => Color.C3
  | Color.C4, Color.Red => Color.C4
  | Color.Blue, Color.Red => Color.Blue
  | _, Color.Blue => Color.Blue
  | _, _ => bitten  -- For all other cases, no color change

/-- A sequence of bites that transforms all chameleons to blue -/
def all_blue_sequence (n : ℕ) : List (Fin n × Fin n) → Prop := sorry

/-- The theorem stating that 5 is the smallest number of red chameleons that can guarantee becoming all blue -/
theorem smallest_all_blue_count :
  (∃ (seq : List (Fin 5 × Fin 5)), all_blue_sequence 5 seq) ∧
  (∀ k < 5, ¬∃ (seq : List (Fin k × Fin k)), all_blue_sequence k seq) :=
sorry

end NUMINAMATH_CALUDE_smallest_all_blue_count_l2815_281595


namespace NUMINAMATH_CALUDE_journey_speed_l2815_281583

theorem journey_speed (D : ℝ) (V : ℝ) (h1 : D > 0) (h2 : V > 0) : 
  (2 * D) / (D / V + D / 30) = 40 → V = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l2815_281583


namespace NUMINAMATH_CALUDE_skateboarder_speed_l2815_281531

/-- Proves that a skateboarder traveling 660 feet in 30 seconds is moving at a speed of 15 miles per hour, given that 1 mile equals 5280 feet. -/
theorem skateboarder_speed (distance : ℝ) (time : ℝ) (feet_per_mile : ℝ) 
  (h1 : distance = 660)
  (h2 : time = 30)
  (h3 : feet_per_mile = 5280) : 
  (distance / time) * (3600 / feet_per_mile) = 15 := by
  sorry

#check skateboarder_speed

end NUMINAMATH_CALUDE_skateboarder_speed_l2815_281531


namespace NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l2815_281524

/-- Calculates a person's total income based on given distributions and remaining amount. -/
theorem calculate_total_income (children_percentage : ℝ) (wife_percentage : ℝ) 
  (orphan_donation_percentage : ℝ) (remaining_amount : ℝ) : ℝ :=
  let total_distributed_percentage := 3 * children_percentage + wife_percentage
  let remaining_percentage := 1 - total_distributed_percentage
  let final_remaining_percentage := remaining_percentage * (1 - orphan_donation_percentage)
  remaining_amount / final_remaining_percentage

/-- Proves that the person's total income is $1,000,000 given the conditions. -/
theorem person_total_income : 
  calculate_total_income 0.2 0.3 0.05 50000 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l2815_281524


namespace NUMINAMATH_CALUDE_apple_crates_delivered_l2815_281551

/-- The number of crates delivered to a factory, given the conditions of the apple delivery problem. -/
theorem apple_crates_delivered : ℕ := by
  -- Define the number of apples per crate
  let apples_per_crate : ℕ := 180

  -- Define the number of rotten apples
  let rotten_apples : ℕ := 160

  -- Define the number of boxes and apples per box for the remaining apples
  let num_boxes : ℕ := 100
  let apples_per_box : ℕ := 20

  -- Calculate the total number of good apples
  let good_apples : ℕ := num_boxes * apples_per_box

  -- Calculate the total number of apples delivered
  let total_apples : ℕ := good_apples + rotten_apples

  -- Calculate the number of crates delivered
  let crates_delivered : ℕ := total_apples / apples_per_crate

  -- Prove that the number of crates delivered is 12
  have : crates_delivered = 12 := by sorry

  -- Return the result
  exact 12


end NUMINAMATH_CALUDE_apple_crates_delivered_l2815_281551


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2815_281560

theorem circle_radius_problem (circle_A circle_B : ℝ) : 
  circle_A = 4 * circle_B →  -- Radius of A is 4 times radius of B
  2 * circle_A = 80 →        -- Diameter of A is 80 cm
  circle_B = 10 := by        -- Radius of B is 10 cm
sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2815_281560


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_eight_l2815_281504

/-- A geometric sequence with a quadratic equation and specific conditions -/
structure GeometricSequence where
  -- The quadratic equation coefficients
  a : ℝ
  b : ℝ
  c : ℝ
  -- The condition that the quadratic equation holds for the sequence
  quad_eq : a * t^2 + b * t + c = 0
  -- The conditions given in the problem
  sum_condition : a1 + a2 = -1
  diff_condition : a1 - a3 = -3
  -- The general term of the sequence
  a_n : ℕ → ℝ

/-- The theorem stating that the fourth term of the sequence is -8 -/
theorem fourth_term_is_negative_eight (seq : GeometricSequence) :
  seq.a = 1 ∧ seq.b = -36 ∧ seq.c = 288 →
  seq.a_n 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_eight_l2815_281504


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2815_281555

def inverse_proportional_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, n > 0 → a n * a (n + 1) = k

theorem fifteenth_term_of_sequence 
  (a : ℕ → ℝ)
  (h_inv_prop : inverse_proportional_sequence a)
  (h_first_term : a 1 = 3)
  (h_second_term : a 2 = 4) :
  a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2815_281555


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l2815_281561

theorem min_value_sum_of_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / b + b / c + c / a + a / c ≥ 4 ∧
  (a / b + b / c + c / a + a / c = 4 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l2815_281561


namespace NUMINAMATH_CALUDE_inequality_proof_l2815_281585

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^3 + y^3 + z^3 = 1) :
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2815_281585


namespace NUMINAMATH_CALUDE_translation_result_l2815_281518

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- The initial point P(2,3) -/
def initialPoint : Point :=
  ⟨2, 3⟩

/-- The final point after translation -/
def finalPoint : Point :=
  translateVertical (translateHorizontal initialPoint (-3)) (-4)

theorem translation_result :
  finalPoint = ⟨-1, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l2815_281518


namespace NUMINAMATH_CALUDE_xiao_wei_wears_five_l2815_281509

/-- Represents the five people in the line -/
inductive Person : Type
  | XiaoWang
  | XiaoZha
  | XiaoTian
  | XiaoYan
  | XiaoWei

/-- Represents the hat numbers -/
inductive HatNumber : Type
  | One
  | Two
  | Three
  | Four
  | Five

/-- Function that assigns a hat number to each person -/
def hatAssignment : Person → HatNumber := sorry

/-- Function that determines if a person can see another person's hat -/
def canSee : Person → Person → Prop := sorry

/-- The hat numbers are all different -/
axiom all_different : ∀ p q : Person, p ≠ q → hatAssignment p ≠ hatAssignment q

/-- Xiao Wang cannot see any hats -/
axiom xiao_wang_sees_none : ∀ p : Person, ¬(canSee Person.XiaoWang p)

/-- Xiao Zha can only see hat 4 -/
axiom xiao_zha_sees_four : ∃! p : Person, canSee Person.XiaoZha p ∧ hatAssignment p = HatNumber.Four

/-- Xiao Tian does not see hat 3, but can see hat 1 -/
axiom xiao_tian_condition : (∃ p : Person, canSee Person.XiaoTian p ∧ hatAssignment p = HatNumber.One) ∧
                            (∀ p : Person, canSee Person.XiaoTian p → hatAssignment p ≠ HatNumber.Three)

/-- Xiao Yan can see three hats, but not hat 3 -/
axiom xiao_yan_condition : (∃ p q r : Person, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
                            canSee Person.XiaoYan p ∧ canSee Person.XiaoYan q ∧ canSee Person.XiaoYan r) ∧
                           (∀ p : Person, canSee Person.XiaoYan p → hatAssignment p ≠ HatNumber.Three)

/-- Xiao Wei can see hat 3 and hat 2 -/
axiom xiao_wei_condition : (∃ p : Person, canSee Person.XiaoWei p ∧ hatAssignment p = HatNumber.Three) ∧
                           (∃ q : Person, canSee Person.XiaoWei q ∧ hatAssignment q = HatNumber.Two)

/-- Theorem: Xiao Wei is wearing hat number 5 -/
theorem xiao_wei_wears_five : hatAssignment Person.XiaoWei = HatNumber.Five := by sorry

end NUMINAMATH_CALUDE_xiao_wei_wears_five_l2815_281509


namespace NUMINAMATH_CALUDE_lion_king_star_wars_profit_ratio_l2815_281549

/-- The ratio of profits between two movies -/
def profit_ratio (cost1 revenue1 cost2 revenue2 : ℚ) : ℚ :=
  (revenue1 - cost1) / (revenue2 - cost2)

/-- Theorem: The ratio of The Lion King's profit to Star Wars' profit is 1:2 -/
theorem lion_king_star_wars_profit_ratio :
  profit_ratio 10 200 25 405 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_lion_king_star_wars_profit_ratio_l2815_281549


namespace NUMINAMATH_CALUDE_knights_archery_skill_l2815_281523

theorem knights_archery_skill (total : ℕ) (total_pos : total > 0) : 
  let gold := (3 * total) / 8
  let silver := total - gold
  let skilled := total / 4
  ∃ (gold_skilled silver_skilled : ℕ),
    gold_skilled + silver_skilled = skilled ∧
    gold_skilled * silver = 3 * silver_skilled * gold ∧
    gold_skilled * 7 = gold * 3 := by
  sorry

end NUMINAMATH_CALUDE_knights_archery_skill_l2815_281523


namespace NUMINAMATH_CALUDE_sum_of_squares_nonzero_implies_one_nonzero_l2815_281542

theorem sum_of_squares_nonzero_implies_one_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_nonzero_implies_one_nonzero_l2815_281542


namespace NUMINAMATH_CALUDE_work_completion_smaller_group_l2815_281540

/-- Given that 22 men complete a work in 55 days, and another group completes
    the same work in 121 days, prove that the number of men in the second group is 10. -/
theorem work_completion_smaller_group : 
  ∀ (work : ℕ) (group1_size group1_days group2_days : ℕ),
    group1_size = 22 →
    group1_days = 55 →
    group2_days = 121 →
    group1_size * group1_days = work →
    ∃ (group2_size : ℕ), 
      group2_size * group2_days = work ∧
      group2_size = 10 :=
by
  sorry

#check work_completion_smaller_group

end NUMINAMATH_CALUDE_work_completion_smaller_group_l2815_281540


namespace NUMINAMATH_CALUDE_students_behind_in_line_l2815_281598

/-- Given a line of students waiting for a bus, this theorem proves
    the number of students behind a specific student. -/
theorem students_behind_in_line
  (total_students : ℕ)
  (students_in_front : ℕ)
  (h1 : total_students = 30)
  (h2 : students_in_front = 20) :
  total_students - (students_in_front + 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_in_line_l2815_281598


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2815_281530

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 5 ≤ x + 1 ∧ (x - 1) / 2 > x - 4) ↔ x < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2815_281530


namespace NUMINAMATH_CALUDE_exists_probability_outside_range_l2815_281565

/-- Represents a packet of candies -/
structure Packet :=
  (total : ℕ)
  (blue : ℕ)
  (h : blue ≤ total)

/-- Represents a box containing two packets of candies -/
structure Box :=
  (packet1 : Packet)
  (packet2 : Packet)

/-- Calculates the probability of drawing a blue candy from the box -/
def blueProbability (box : Box) : ℚ :=
  (box.packet1.blue + box.packet2.blue : ℚ) / (box.packet1.total + box.packet2.total)

/-- Theorem stating that there exists a box configuration where the probability
    of drawing a blue candy is not between 3/8 and 2/5 -/
theorem exists_probability_outside_range :
  ∃ (box : Box), ¬(3/8 < blueProbability box ∧ blueProbability box < 2/5) :=
sorry

end NUMINAMATH_CALUDE_exists_probability_outside_range_l2815_281565


namespace NUMINAMATH_CALUDE_cuatro_cuinte_equation_l2815_281512

/-- Represents a mapping from letters to digits -/
def LetterToDigit := Char → Nat

/-- Check if a mapping is valid (each letter maps to a unique digit) -/
def is_valid_mapping (m : LetterToDigit) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Convert a string to a number using the given mapping -/
def string_to_number (s : String) (m : LetterToDigit) : Nat :=
  s.foldl (fun acc c => 10 * acc + m c) 0

/-- The main theorem to prove -/
theorem cuatro_cuinte_equation (m : LetterToDigit) 
  (h_valid : is_valid_mapping m)
  (h_cuatro : string_to_number "CUATRO" m = 170349)
  (h_cuaatro : string_to_number "CUAATRO" m = 1700349)
  (h_cuinte : string_to_number "CUINTE" m = 3852345) :
  170349 + 170349 + 1700349 + 1700349 + 170349 = 3852345 := by
  sorry

/-- Lemma: The mapping satisfies the equation -/
lemma mapping_satisfies_equation (m : LetterToDigit) 
  (h_valid : is_valid_mapping m)
  (h_cuatro : string_to_number "CUATRO" m = 170349)
  (h_cuaatro : string_to_number "CUAATRO" m = 1700349)
  (h_cuinte : string_to_number "CUINTE" m = 3852345) :
  string_to_number "CUATRO" m + string_to_number "CUATRO" m + 
  string_to_number "CUAATRO" m + string_to_number "CUAATRO" m + 
  string_to_number "CUATRO" m = string_to_number "CUINTE" m := by
  sorry

end NUMINAMATH_CALUDE_cuatro_cuinte_equation_l2815_281512


namespace NUMINAMATH_CALUDE_pentagon_area_bound_l2815_281514

-- Define the pentagon ABCDE
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_convex_pentagon (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def angle (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def area (A B C D E : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem pentagon_area_bound 
  (h_convex : is_convex_pentagon A B C D E)
  (h_angle_EAB : angle E A B = 2 * π / 3)
  (h_angle_ABC : angle A B C = 2 * π / 3)
  (h_angle_ADB : angle A D B = π / 6)
  (h_angle_CDE : angle C D E = π / 3)
  (h_side_AB : distance A B = 1) :
  area A B C D E < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_bound_l2815_281514


namespace NUMINAMATH_CALUDE_zebras_total_games_l2815_281579

theorem zebras_total_games : 
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →  -- 40% win rate initially
    ∀ (final_games : ℕ) (final_wins : ℕ),
      final_games = initial_games + 11 →  -- 8 won + 3 lost = 11 more games
      final_wins = initial_wins + 8 →     -- 8 more wins
      final_wins = (11 * final_games / 20) →  -- 55% win rate finally
      final_games = 24 := by
sorry

end NUMINAMATH_CALUDE_zebras_total_games_l2815_281579


namespace NUMINAMATH_CALUDE_marks_of_a_l2815_281563

theorem marks_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e = d + 3 →
  (b + c + d + e) / 4 = 48 →
  a = 43 := by
sorry

end NUMINAMATH_CALUDE_marks_of_a_l2815_281563


namespace NUMINAMATH_CALUDE_cone_volume_and_surface_area_l2815_281557

/-- Cone with given slant height and height -/
structure Cone where
  slant_height : ℝ
  height : ℝ

/-- Volume of a cone -/
def volume (c : Cone) : ℝ := sorry

/-- Surface area of a cone -/
def surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific cone -/
theorem cone_volume_and_surface_area :
  let c : Cone := { slant_height := 17, height := 15 }
  (volume c = 320 * Real.pi) ∧ (surface_area c = 200 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_and_surface_area_l2815_281557


namespace NUMINAMATH_CALUDE_quoted_price_calculation_l2815_281574

/-- Calculates the quoted price of shares given investment details -/
theorem quoted_price_calculation (investment : ℚ) (face_value : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : 
  investment = 4455 ∧ 
  face_value = 10 ∧ 
  dividend_rate = 12 / 100 ∧ 
  annual_income = 648 → 
  (investment / (annual_income / (dividend_rate * face_value))) = 33 / 4 :=
by sorry

end NUMINAMATH_CALUDE_quoted_price_calculation_l2815_281574


namespace NUMINAMATH_CALUDE_total_toys_cost_is_20_74_l2815_281507

/-- The amount spent on toy cars -/
def toy_cars_cost : ℚ := 14.88

/-- The amount spent on toy trucks -/
def toy_trucks_cost : ℚ := 5.86

/-- The total amount spent on toys -/
def total_toys_cost : ℚ := toy_cars_cost + toy_trucks_cost

/-- Theorem stating that the total amount spent on toys is $20.74 -/
theorem total_toys_cost_is_20_74 : total_toys_cost = 20.74 := by sorry

end NUMINAMATH_CALUDE_total_toys_cost_is_20_74_l2815_281507


namespace NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l2815_281576

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem volume_sphere_minus_cylinder (r_sphere : ℝ) (r_cylinder : ℝ) : 
  r_sphere = 6 → r_cylinder = 4 → 
  ∃ V : ℝ, V = (288 - 64 * Real.sqrt 5) * Real.pi ∧
    V = (4 / 3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * Real.sqrt (r_sphere^2 - r_cylinder^2)) :=
by sorry

end NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l2815_281576


namespace NUMINAMATH_CALUDE_gunther_typing_capacity_l2815_281528

/-- Given Gunther's typing speed and work day length, prove the number of words he can type in a day --/
theorem gunther_typing_capacity (words_per_set : ℕ) (minutes_per_set : ℕ) (minutes_per_day : ℕ) 
  (h1 : words_per_set = 160)
  (h2 : minutes_per_set = 3)
  (h3 : minutes_per_day = 480) :
  (minutes_per_day / minutes_per_set) * words_per_set = 25600 := by
  sorry

#eval (480 / 3) * 160

end NUMINAMATH_CALUDE_gunther_typing_capacity_l2815_281528


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_choose_n_minus_one_l2815_281522

theorem binomial_coefficient_n_choose_n_minus_one (n : ℕ+) : 
  Nat.choose n.val (n.val - 1) = n.val := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_choose_n_minus_one_l2815_281522


namespace NUMINAMATH_CALUDE_circle_properties_l2815_281538

/-- Given that this equation represents a circle for real m, prove the statements about m, r, and the circle's center -/
theorem circle_properties (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0 →
    ∃ r : ℝ, (x - (m + 3))^2 + (y - (4*m^2 - 1))^2 = r^2) →
  (-1 < m ∧ m < 1) ∧
  (∃ r : ℝ, 0 < r ∧ r ≤ Real.sqrt 2 ∧
    ∀ x y : ℝ, x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0 →
      (x - (m + 3))^2 + (y - (4*m^2 - 1))^2 = r^2) ∧
  (∃ x y : ℝ, -1 < x ∧ x < 4 ∧ y = 4*(x - 3)^2 - 1 ∧
    x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2815_281538


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2815_281568

theorem sqrt_equation_solution (y : ℝ) :
  (y > 2) →
  (Real.sqrt (7 * y) / Real.sqrt (4 * (y - 2)) = 3) →
  y = 72 / 29 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2815_281568


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_sum_l2815_281573

theorem binomial_coefficient_ratio_sum (n k : ℕ) : 
  (2 : ℚ) / 5 = (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) →
  (∃ m l : ℕ, m ≠ l ∧ (m = n ∧ l = k ∨ l = n ∧ m = k) ∧ m + l = 23) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_sum_l2815_281573


namespace NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l2815_281536

/-- The original parabola equation -/
def original_parabola (x : ℝ) : ℝ := -x^2 - 2*x + 3

/-- The shifted parabola equation -/
def shifted_parabola (x : ℝ) : ℝ := -x^2 + 2

/-- Theorem stating that the shifted parabola passes through (-1, 1) -/
theorem shifted_parabola_passes_through_point :
  shifted_parabola (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l2815_281536


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2815_281519

theorem largest_solution_of_equation (c : ℝ) : 
  (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2815_281519


namespace NUMINAMATH_CALUDE_multiply_add_distribute_compute_expression_l2815_281535

theorem multiply_add_distribute (a b c : ℕ) : a * b + c * a = a * (b + c) := by sorry

theorem compute_expression : 45 * 27 + 73 * 45 = 4500 := by sorry

end NUMINAMATH_CALUDE_multiply_add_distribute_compute_expression_l2815_281535


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l2815_281578

def fibonacci_like_sequence (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n + 1)

theorem first_term_of_sequence (a b : ℕ) :
  fibonacci_like_sequence a b 5 = 21 ∧
  fibonacci_like_sequence a b 6 = 34 ∧
  fibonacci_like_sequence a b 7 = 55 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l2815_281578


namespace NUMINAMATH_CALUDE_impossible_tiling_l2815_281572

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile that can be placed on the board -/
inductive Tile
  | Domino    : Tile  -- 1 × 2 horizontal domino
  | Rectangle : Tile  -- 1 × 3 vertical rectangle

/-- Represents a tiling of the board -/
def Tiling := List (Tile × ℕ × ℕ)  -- List of (tile type, row, column)

/-- Check if a tiling is valid for the given board -/
def is_valid_tiling (board : Board) (tiling : Tiling) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to tile the 2003 × 2003 board -/
theorem impossible_tiling :
  ∀ (tiling : Tiling), ¬(is_valid_tiling (Board.mk 2003 2003) tiling) :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l2815_281572


namespace NUMINAMATH_CALUDE_prob_five_is_one_thirteenth_l2815_281597

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image (·.1)).card = 13)
  (suit_count : (cards.image (·.2)).card = 4)
  (unique_cards : ∀ r s, (r, s) ∈ cards ↔ r ∈ Finset.range 13 ∧ s ∈ Finset.range 4)

/-- The probability of drawing a specific rank from a standard deck -/
def prob_rank (d : Deck) (rank : Nat) : ℚ :=
  (d.cards.filter (·.1 = rank)).card / d.cards.card

/-- Theorem: The probability of drawing a 5 from a standard deck is 1/13 -/
theorem prob_five_is_one_thirteenth (d : Deck) : prob_rank d 5 = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_is_one_thirteenth_l2815_281597


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l2815_281553

theorem sum_of_absolute_roots (m : ℤ) (a b c d : ℤ) : 
  (∀ x : ℤ, x^4 - x^3 - 4023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) →
  |a| + |b| + |c| + |d| = 621 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l2815_281553


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2815_281586

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_a1 : a 1 = 2)
  (h_sum : a 2 + a 3 = 13) :
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2815_281586


namespace NUMINAMATH_CALUDE_inequality_range_difference_l2815_281559

-- Define g as a strictly increasing function
variable (g : ℝ → ℝ)

-- Define the property of g being strictly increasing
def StrictlyIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

-- Define the theorem
theorem inequality_range_difference
  (h1 : StrictlyIncreasing g)
  (h2 : ∀ x, x ≥ 0 → g x ≠ 0)
  (h3 : ∃ a b, ∀ t, (g (2*t^2 + t + 5) < g (t^2 - 3*t + 2)) ↔ (b < t ∧ t < a)) :
  ∃ a b, (∀ t, (g (2*t^2 + t + 5) < g (t^2 - 3*t + 2)) ↔ (b < t ∧ t < a)) ∧ a - b = 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_difference_l2815_281559


namespace NUMINAMATH_CALUDE_floor_sum_example_l2815_281508

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2815_281508


namespace NUMINAMATH_CALUDE_olivias_albums_l2815_281505

/-- Given a total number of pictures and a number of albums, 
    calculate the number of pictures in each album. -/
def pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) : ℕ :=
  total_pictures / num_albums

/-- Prove that given 40 total pictures and 8 albums, 
    there are 5 pictures in each album. -/
theorem olivias_albums : 
  let total_pictures : ℕ := 40
  let num_albums : ℕ := 8
  pictures_per_album total_pictures num_albums = 5 := by
  sorry

end NUMINAMATH_CALUDE_olivias_albums_l2815_281505


namespace NUMINAMATH_CALUDE_coplanar_points_l2815_281525

/-- The points (0,0,0), (1,a,0), (0,1,a), and (a,0,1) are coplanar if and only if a = -1 -/
theorem coplanar_points (a : ℝ) : 
  (Matrix.det
    ![![1, 0, a],
      ![a, 1, 0],
      ![0, a, 1]] = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_points_l2815_281525


namespace NUMINAMATH_CALUDE_max_fourth_power_sum_l2815_281516

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  ∃ (m : ℝ), (∀ x y z w : ℝ, x^3 + y^3 + z^3 + w^3 = 4 → x^4 + y^4 + z^4 + w^4 ≤ m) ∧
             (a^4 + b^4 + c^4 + d^4 = m) ∧
             m = 16 :=
sorry

end NUMINAMATH_CALUDE_max_fourth_power_sum_l2815_281516


namespace NUMINAMATH_CALUDE_ball_hits_middle_pocket_l2815_281594

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  p : ℕ
  q : ℕ
  p_odd : Odd p
  q_odd : Odd q

/-- Represents the trajectory of a ball on the billiard table -/
def ball_trajectory (table : BilliardTable) : ℕ → ℕ → Prop :=
  fun x y => y = x

/-- Represents a middle pocket on the long side of the table -/
def middle_pocket (table : BilliardTable) : ℕ → ℕ → Prop :=
  fun x y => (x = table.p / 2 ∧ (y = 0 ∨ y = 2 * table.q)) ∨ 
             (y = table.q ∧ (x = 0 ∨ x = table.p))

/-- The main theorem stating that the ball will hit a middle pocket -/
theorem ball_hits_middle_pocket (table : BilliardTable) :
  ∃ (x y : ℕ), ball_trajectory table x y ∧ middle_pocket table x y :=
sorry

end NUMINAMATH_CALUDE_ball_hits_middle_pocket_l2815_281594


namespace NUMINAMATH_CALUDE_even_function_inequality_l2815_281513

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f: ℝ → ℝ is increasing on [0, +∞) if
    for all x, y ∈ [0, +∞), x < y implies f(x) < f(y) -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) 
    (heven : EvenFunction f) (hinc : IncreasingOnNonnegative f) :
    f π > f (-2) ∧ f (-2) > f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l2815_281513


namespace NUMINAMATH_CALUDE_train_seats_theorem_l2815_281548

/-- The total number of seats on the train -/
def total_seats : ℕ := 180

/-- The number of seats in Standard Class -/
def standard_seats : ℕ := 36

/-- The fraction of total seats in Comfort Class -/
def comfort_fraction : ℚ := 1/5

/-- The fraction of total seats in Premium Class -/
def premium_fraction : ℚ := 3/5

/-- Theorem stating that the total number of seats is 180 -/
theorem train_seats_theorem :
  (standard_seats : ℚ) + comfort_fraction * total_seats + premium_fraction * total_seats = total_seats := by
  sorry

end NUMINAMATH_CALUDE_train_seats_theorem_l2815_281548


namespace NUMINAMATH_CALUDE_joey_caught_one_kg_more_than_peter_l2815_281577

/-- Given three fishers Ali, Peter, and Joey, prove that Joey caught 1 kg more fish than Peter -/
theorem joey_caught_one_kg_more_than_peter 
  (total_catch : ℝ)
  (ali_catch : ℝ)
  (peter_catch : ℝ)
  (joey_catch : ℝ)
  (h1 : total_catch = 25)
  (h2 : ali_catch = 12)
  (h3 : ali_catch = 2 * peter_catch)
  (h4 : joey_catch = peter_catch + (joey_catch - peter_catch))
  (h5 : total_catch = ali_catch + peter_catch + joey_catch) :
  joey_catch - peter_catch = 1 := by
  sorry

end NUMINAMATH_CALUDE_joey_caught_one_kg_more_than_peter_l2815_281577


namespace NUMINAMATH_CALUDE_final_number_after_ten_steps_l2815_281554

/-- Performs one step of the sequence operation -/
def step (n : ℕ) (i : ℕ) : ℕ :=
  if i % 2 = 0 then n * 3 else n / 4

/-- Performs n steps of the sequence operation -/
def iterate_steps (start : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => start
  | k + 1 => step (iterate_steps start k) k

theorem final_number_after_ten_steps :
  iterate_steps 800000 10 = 1518750 := by
  sorry

end NUMINAMATH_CALUDE_final_number_after_ten_steps_l2815_281554


namespace NUMINAMATH_CALUDE_math_problem_proof_l2815_281533

theorem math_problem_proof :
  let expression1 := -3^2 + 2^2023 * (-1/2)^2022 + (-2024)^0
  let x : ℚ := -1/2
  let y : ℚ := 1
  let expression2 := ((x + 2*y)^2 - (2*x + y)*(2*x - y) - 5*(x^2 + y^2)) / (2*x)
  expression1 = -6 ∧ expression2 = 4 := by sorry

end NUMINAMATH_CALUDE_math_problem_proof_l2815_281533


namespace NUMINAMATH_CALUDE_besfamilies_children_count_l2815_281589

/-- Represents the Besfamilies family structure and age calculations -/
structure Besfamilies where
  initialAge : ℕ  -- Family age when youngest child was born
  finalAge : ℕ    -- Family age after several years
  yearsPassed : ℕ -- Number of years passed

/-- Calculates the number of children in the Besfamilies -/
def numberOfChildren (family : Besfamilies) : ℕ :=
  ((family.finalAge - family.initialAge) / family.yearsPassed) - 2

/-- Theorem stating the number of children in the Besfamilies -/
theorem besfamilies_children_count 
  (family : Besfamilies) 
  (h1 : family.initialAge = 101)
  (h2 : family.finalAge = 150)
  (h3 : family.yearsPassed > 1)
  (h4 : (family.finalAge - family.initialAge) % family.yearsPassed = 0) :
  numberOfChildren family = 5 := by
  sorry

#eval numberOfChildren { initialAge := 101, finalAge := 150, yearsPassed := 7 }

end NUMINAMATH_CALUDE_besfamilies_children_count_l2815_281589


namespace NUMINAMATH_CALUDE_count_two_digit_numbers_tens_less_than_ones_eq_36_l2815_281502

/-- The count of two-digit numbers where the tens digit is less than the ones digit -/
def count_two_digit_numbers_tens_less_than_ones : ℕ :=
  (Finset.range 9).sum (λ t => (Finset.range (10 - t)).card)

/-- Theorem stating that the count of two-digit numbers where the tens digit is less than the ones digit is 36 -/
theorem count_two_digit_numbers_tens_less_than_ones_eq_36 :
  count_two_digit_numbers_tens_less_than_ones = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_numbers_tens_less_than_ones_eq_36_l2815_281502


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l2815_281593

/-- Given a function g(x) = x(x+3)/3, prove that g(x+3) = (x^2 + 9x + 18) / 3 -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g : ℝ → ℝ := fun x => x * (x + 3) / 3
  g (x + 3) = (x^2 + 9*x + 18) / 3 := by
sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l2815_281593


namespace NUMINAMATH_CALUDE_count_digit_nine_to_thousand_l2815_281537

/-- The number of occurrences of a digit in a specific place (units, tens, or hundreds) for numbers from 1 to 1000 -/
def occurrences_in_place : ℕ := 100

/-- The number of places (units, tens, hundreds) in numbers from 1 to 1000 -/
def num_places : ℕ := 3

/-- The digit we're counting -/
def target_digit : ℕ := 9

/-- Theorem: The number of occurrences of the digit 9 in the list of integers from 1 to 1000 is equal to 300 -/
theorem count_digit_nine_to_thousand : 
  occurrences_in_place * num_places = 300 :=
sorry

end NUMINAMATH_CALUDE_count_digit_nine_to_thousand_l2815_281537


namespace NUMINAMATH_CALUDE_root_configurations_l2815_281596

-- Define the polynomial
def polynomial (a b c x : ℂ) : ℂ := x^4 - a*x^3 - b*x + c

-- Define the theorem
theorem root_configurations (a b c : ℂ) :
  (∃ d : ℂ, d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    (∀ x : ℂ, polynomial a b c x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) →
  ((a = 0 ∧ b = 0 ∧ c ≠ 0) ∨
   (a ≠ 0 ∧ b = c ∧ c ≠ 0 ∧ c^2 + c + 1 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_root_configurations_l2815_281596


namespace NUMINAMATH_CALUDE_count_positive_area_triangles_l2815_281584

/-- The number of points in each row or column of the grid -/
def gridSize : ℕ := 6

/-- The total number of points in the grid -/
def totalPoints : ℕ := gridSize * gridSize

/-- The number of ways to choose 3 points from the total points -/
def totalCombinations : ℕ := Nat.choose totalPoints 3

/-- The number of ways to choose 3 points from a single row or column -/
def lineCombo : ℕ := Nat.choose gridSize 3

/-- The number of straight lines (rows and columns) -/
def numLines : ℕ := 2 * gridSize

/-- The number of main diagonals -/
def numMainDiagonals : ℕ := 2

/-- The number of triangles with positive area on the grid -/
def positiveAreaTriangles : ℕ := 
  totalCombinations - (numLines * lineCombo) - (numMainDiagonals * lineCombo)

theorem count_positive_area_triangles : positiveAreaTriangles = 6860 := by
  sorry

end NUMINAMATH_CALUDE_count_positive_area_triangles_l2815_281584


namespace NUMINAMATH_CALUDE_cost_of_500_apples_l2815_281587

/-- The cost of a single apple in cents -/
def apple_cost : ℕ := 5

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of apples we want to calculate the cost for -/
def apple_quantity : ℕ := 500

/-- Theorem stating that the cost of 500 apples is 25.00 dollars -/
theorem cost_of_500_apples : 
  (apple_quantity * apple_cost : ℚ) / cents_per_dollar = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_apples_l2815_281587


namespace NUMINAMATH_CALUDE_theater_seats_count_l2815_281569

/-- Represents a theater with a specific seating arrangement. -/
structure Theater where
  rows : ℕ
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater. -/
def total_seats (t : Theater) : ℕ :=
  (t.rows * (2 * t.first_row_seats + (t.rows - 1) * t.seat_increment)) / 2

/-- Theorem stating that a theater with the given properties has 720 seats. -/
theorem theater_seats_count :
  ∀ (t : Theater),
    t.rows = 15 →
    t.first_row_seats = 20 →
    t.seat_increment = 4 →
    t.last_row_seats = 76 →
    total_seats t = 720 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_seats_count_l2815_281569


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2815_281566

theorem vector_equation_solution :
  ∃ (u v : ℝ), (![3, 1] : Fin 2 → ℝ) + u • ![8, -6] = ![2, -2] + v • ![-3, 4] ∧ 
  u = -13/14 ∧ v = 15/7 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2815_281566


namespace NUMINAMATH_CALUDE_absolute_value_equation_roots_l2815_281564

theorem absolute_value_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ 
  (∀ x : ℝ, x < 0 → |x| ≠ a * x - a) → 
  a > 1 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_roots_l2815_281564


namespace NUMINAMATH_CALUDE_initial_overs_calculation_l2815_281539

/-- Proves the number of initial overs in a cricket game given specific conditions --/
theorem initial_overs_calculation (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 272) (h2 : initial_rate = 3.2) (h3 : required_rate = 6) 
  (h4 : remaining_overs = 40) :
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry


end NUMINAMATH_CALUDE_initial_overs_calculation_l2815_281539


namespace NUMINAMATH_CALUDE_f_difference_960_480_l2815_281546

def sum_of_divisors (n : ℕ) : ℕ := sorry

def f (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

theorem f_difference_960_480 : f 960 - f 480 = 1 / 40 := by sorry

end NUMINAMATH_CALUDE_f_difference_960_480_l2815_281546


namespace NUMINAMATH_CALUDE_median_inequality_l2815_281511

-- Define a triangle with sides a, b, c and medians sa, sb, sc
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sa : ℝ
  sb : ℝ
  sc : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- State the theorem
theorem median_inequality (t : Triangle) :
  (t.sa^2 / (t.b * t.c)) + (t.sb^2 / (t.c * t.a)) + (t.sc^2 / (t.a * t.b)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_l2815_281511


namespace NUMINAMATH_CALUDE_lena_calculation_l2815_281567

def double (n : ℕ) : ℕ := 2 * n

def roundToNearestTen (n : ℕ) : ℕ :=
  let remainder := n % 10
  if remainder < 5 then n - remainder else n + (10 - remainder)

theorem lena_calculation : roundToNearestTen (63 + double 29) = 120 := by
  sorry

end NUMINAMATH_CALUDE_lena_calculation_l2815_281567


namespace NUMINAMATH_CALUDE_count_solutions_correct_l2815_281526

/-- The number of integer solutions to x^2 - y^2 = 45 -/
def count_solutions : ℕ := 12

/-- A pair of integers (x, y) is a solution if x^2 - y^2 = 45 -/
def is_solution (x y : ℤ) : Prop := x^2 - y^2 = 45

/-- The theorem stating that there are exactly 12 integer solutions to x^2 - y^2 = 45 -/
theorem count_solutions_correct :
  (∃ (s : Finset (ℤ × ℤ)), s.card = count_solutions ∧ 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2)) :=
sorry


end NUMINAMATH_CALUDE_count_solutions_correct_l2815_281526


namespace NUMINAMATH_CALUDE_workers_in_first_group_l2815_281588

/-- The number of workers in the first group -/
def W : ℕ := 70

/-- The time taken by the first group to complete the job (in hours) -/
def T1 : ℕ := 3

/-- The number of workers in the second group -/
def W2 : ℕ := 30

/-- The time taken by the second group to complete the job (in hours) -/
def T2 : ℕ := 7

/-- The amount of work done (assumed to be constant for both groups) -/
def work : ℕ := W * T1

theorem workers_in_first_group :
  (W * T1 = W2 * T2) ∧ (W * T2 = W2 * T1) → W = 70 := by
  sorry

end NUMINAMATH_CALUDE_workers_in_first_group_l2815_281588


namespace NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l2815_281545

def sum_of_digits (n : ℕ) : ℕ := sorry

def A : ℕ := sum_of_digits (4444^4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B_is_seven : sum_of_digits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l2815_281545


namespace NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2005_l2815_281534

/-- The sequence a_n defined as 1 + 2^n + 3^n + 4^n + 5^n -/
def a (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

/-- Theorem stating that there are no 5 consecutive terms in the sequence a_n all divisible by 2005 -/
theorem no_five_consecutive_divisible_by_2005 :
  ∀ m : ℕ, ¬(∀ k : Fin 5, 2005 ∣ a (m + k)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2005_l2815_281534


namespace NUMINAMATH_CALUDE_optimal_price_l2815_281520

/-- Revenue function -/
def revenue (p : ℝ) : ℝ := 150 * p - 6 * p^2

/-- Constraint: price is at most 30 -/
def price_constraint (p : ℝ) : Prop := p ≤ 30

/-- Constraint: at least 40 books sold per month -/
def sales_constraint (p : ℝ) : Prop := 150 - 6 * p ≥ 40

/-- The optimal price is an integer -/
def integer_price (p : ℝ) : Prop := ∃ n : ℤ, p = n ∧ n > 0

/-- Theorem: The price of 13 maximizes revenue under given constraints -/
theorem optimal_price :
  ∀ p : ℝ, 
  price_constraint p → 
  sales_constraint p → 
  integer_price p → 
  revenue p ≤ revenue 13 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_l2815_281520


namespace NUMINAMATH_CALUDE_calculation_proof_l2815_281515

theorem calculation_proof :
  (5.42 - (3.75 - 0.58) = 2.25) ∧
  ((4/5) * 7.7 + 0.8 * 3.3 - (4/5) = 8) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2815_281515


namespace NUMINAMATH_CALUDE_fraction_equality_l2815_281550

theorem fraction_equality (a b : ℚ) (h : b ≠ 0) (h1 : a / b = 2 / 3) :
  (a - b) / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2815_281550


namespace NUMINAMATH_CALUDE_exists_valid_strategy_365_l2815_281527

/-- A strategy for sorting n elements using 3-way comparisons -/
def SortingStrategy (n : ℕ) := ℕ

/-- The number of 3-way comparisons needed to sort n elements using a given strategy -/
def comparisons (n : ℕ) (s : SortingStrategy n) : ℕ := sorry

/-- A strategy is valid if it correctly sorts n elements -/
def is_valid_strategy (n : ℕ) (s : SortingStrategy n) : Prop := sorry

/-- The main theorem: there exists a valid strategy for 365 elements using at most 1691 comparisons -/
theorem exists_valid_strategy_365 :
  ∃ (s : SortingStrategy 365), is_valid_strategy 365 s ∧ comparisons 365 s ≤ 1691 := by sorry

end NUMINAMATH_CALUDE_exists_valid_strategy_365_l2815_281527


namespace NUMINAMATH_CALUDE_smallest_x_equals_f_2001_l2815_281599

def f (x : ℝ) : ℝ := sorry

axiom f_triple (x : ℝ) (h : 0 < x) : f (3 * x) = 3 * f x

axiom f_definition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - |x - 2|

theorem smallest_x_equals_f_2001 :
  ∃ (x : ℝ), x > 0 ∧ f x = f 2001 ∧ ∀ (y : ℝ), y > 0 ∧ f y = f 2001 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_equals_f_2001_l2815_281599


namespace NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l2815_281592

/-- The number of chocolates Mike ate -/
def mike_chocolates : ℕ := 20

/-- The number of chocolates John ate -/
def john_chocolates : ℕ := mike_chocolates / 2

/-- The combined number of chocolates Mike and John ate -/
def combined_chocolates : ℕ := mike_chocolates + john_chocolates

/-- The number of chocolates Jenny ate -/
def jenny_chocolates : ℕ := 2 * combined_chocolates + 5

/-- Theorem stating that Jenny ate 65 chocolates -/
theorem jenny_ate_65_chocolates : jenny_chocolates = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l2815_281592


namespace NUMINAMATH_CALUDE_circle_equation_l2815_281581

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x-2)^2 + (y-1)^2 = 1

-- Define that a point is on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define that a point is on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Theorem statement
theorem circle_equation : 
  (point_on_line 2 1) ∧ 
  (point_on_line 6 3) ∧ 
  (∃ h k : ℝ, point_on_line h k ∧ point_on_circle h k) ∧
  (point_on_circle 2 0) ∧ 
  (point_on_circle 3 1) → 
  ∀ x y : ℝ, circle_C x y ↔ (x-2)^2 + (y-1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2815_281581


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2815_281558

theorem sufficient_not_necessary : 
  (∃ a : ℝ, a^2 > 16 ∧ a ≤ 4) ∧ 
  (∀ a : ℝ, a > 4 → a^2 > 16) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2815_281558


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2815_281571

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 2 = 2 →
  a 4 + a 5 = 4 →
  a 10 + a 11 = 16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2815_281571


namespace NUMINAMATH_CALUDE_ratio_problem_l2815_281500

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 ∧ B / C = 1/6) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5/8 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2815_281500


namespace NUMINAMATH_CALUDE_can_finish_typing_l2815_281570

/-- Proves that given a passage of 300 characters and a typing speed of 52 characters per minute, 
    it is possible to finish typing the passage in 6 minutes. -/
theorem can_finish_typing (passage_length : ℕ) (typing_speed : ℕ) (time : ℕ) : 
  passage_length = 300 → 
  typing_speed = 52 → 
  time = 6 → 
  typing_speed * time ≥ passage_length := by
sorry

end NUMINAMATH_CALUDE_can_finish_typing_l2815_281570


namespace NUMINAMATH_CALUDE_greatest_multiple_of_6_and_5_less_than_1000_l2815_281529

theorem greatest_multiple_of_6_and_5_less_than_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  6 ∣ n ∧ 
  5 ∣ n ∧ 
  n < 1000 ∧ 
  ∀ m : ℕ, (6 ∣ m ∧ 5 ∣ m ∧ m < 1000) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_6_and_5_less_than_1000_l2815_281529


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2815_281544

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {-2, -1, 0}
def B : Finset Int := {0, 1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2815_281544


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2815_281556

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2815_281556


namespace NUMINAMATH_CALUDE_floor_length_is_ten_l2815_281582

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem: Given the conditions, the floor length is 10 meters -/
theorem floor_length_is_ten (floor : FloorWithRug)
  (h1 : floor.width = 8)
  (h2 : floor.strip_width = 2)
  (h3 : floor.rug_area = 24)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.length = 10 := by
  sorry

#check floor_length_is_ten

end NUMINAMATH_CALUDE_floor_length_is_ten_l2815_281582


namespace NUMINAMATH_CALUDE_f_is_odd_and_satisfies_conditions_l2815_281580

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x + x - 2
  else if x = 0 then 0
  else -2^(-x) + x + 2

-- Theorem statement
theorem f_is_odd_and_satisfies_conditions :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x > 0, f x = 2^x + x - 2) ∧
  (f 0 = 0) ∧
  (∀ x < 0, f x = -2^(-x) + x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_and_satisfies_conditions_l2815_281580


namespace NUMINAMATH_CALUDE_even_function_inequality_l2815_281532

/-- An even function satisfying the given condition -/
def EvenFunctionWithCondition (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → x₁ ≤ 0 → x₂ ≤ 0 → (x₂ - x₁) * (f x₂ - f x₁) > 0)

/-- The main theorem -/
theorem even_function_inequality (f : ℝ → ℝ) (n : ℕ) (hn : n > 0) 
  (hf : EvenFunctionWithCondition f) : 
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l2815_281532


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2815_281501

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2815_281501


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2815_281591

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^3 + (5/2) * x^2 + 3 * Real.log x + b

theorem tangent_line_problem (b : ℝ) :
  (∃ (m : ℝ), (g b 1 = m * 1 - 5) ∧ 
              (∀ (x : ℝ), x ≠ 1 → (g b x - g b 1) / (x - 1) < m)) →
  b = 5/2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2815_281591


namespace NUMINAMATH_CALUDE_divisibility_condition_l2815_281547

theorem divisibility_condition (a n : ℕ+) : 
  n ∣ ((a + 1)^n.val - a^n.val) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2815_281547


namespace NUMINAMATH_CALUDE_cot_sixty_degrees_l2815_281521

theorem cot_sixty_degrees : Real.cos (π / 3) / Real.sin (π / 3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_sixty_degrees_l2815_281521


namespace NUMINAMATH_CALUDE_normal_lemon_tree_production_l2815_281543

/-- The number of lemons produced by a normal lemon tree per year. -/
def normal_lemon_production : ℕ := 60

/-- The number of trees in Jim's grove. -/
def jims_trees : ℕ := 1500

/-- The number of lemons Jim's grove produces per year. -/
def jims_production : ℕ := 135000

/-- Jim's trees produce 50% more lemons than normal trees. -/
def jims_tree_efficiency : ℚ := 3/2

theorem normal_lemon_tree_production :
  normal_lemon_production * jims_trees * jims_tree_efficiency = jims_production :=
by sorry

end NUMINAMATH_CALUDE_normal_lemon_tree_production_l2815_281543


namespace NUMINAMATH_CALUDE_sin_squared_plus_sin_double_l2815_281503

theorem sin_squared_plus_sin_double (α : Real) (h : Real.tan α = 1/2) :
  Real.sin α ^ 2 + Real.sin (2 * α) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_squared_plus_sin_double_l2815_281503


namespace NUMINAMATH_CALUDE_max_value_of_f_l2815_281552

def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2815_281552


namespace NUMINAMATH_CALUDE_exists_eight_numbers_sum_divisible_l2815_281590

theorem exists_eight_numbers_sum_divisible : 
  ∃ (S : Finset ℕ), 
    S.card = 8 ∧ 
    (∀ n ∈ S, n ≤ 100) ∧
    (∀ n ∈ S, (S.sum id) % n = 0) :=
sorry

end NUMINAMATH_CALUDE_exists_eight_numbers_sum_divisible_l2815_281590


namespace NUMINAMATH_CALUDE_committee_formation_with_previous_member_l2815_281575

def total_members : ℕ := 18
def committee_size : ℕ := 6
def previous_members : ℕ := 5

theorem committee_formation_with_previous_member :
  (Nat.choose total_members committee_size) - 
  (Nat.choose (total_members - previous_members) committee_size) = 16848 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_with_previous_member_l2815_281575


namespace NUMINAMATH_CALUDE_number_proof_l2815_281517

theorem number_proof (N p q : ℝ) 
  (h1 : N / p = 6)
  (h2 : N / q = 18)
  (h3 : p - q = 1 / 3) : 
  N = 3 := by
sorry

end NUMINAMATH_CALUDE_number_proof_l2815_281517
