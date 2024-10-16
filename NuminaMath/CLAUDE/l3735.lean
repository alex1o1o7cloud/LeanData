import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3735_373527

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem sufficient_but_not_necessary : 
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3735_373527


namespace NUMINAMATH_CALUDE_sum_of_powers_l3735_373513

theorem sum_of_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 5)
  (h4 : a^4 + b^4 = 7) :
  a^10 + b^10 = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3735_373513


namespace NUMINAMATH_CALUDE_circle_area_sum_l3735_373569

theorem circle_area_sum : 
  let radius : ℕ → ℝ := λ n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (radius n)^2
  let series_sum : ℝ := ∑' n, area n
  series_sum = 9*π/2 := by
sorry

end NUMINAMATH_CALUDE_circle_area_sum_l3735_373569


namespace NUMINAMATH_CALUDE_min_additional_games_proof_l3735_373528

/-- The minimum number of additional games the Sharks need to win -/
def min_additional_games : ℕ := 145

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The initial number of games won by the Sharks -/
def initial_sharks_wins : ℕ := 2

/-- Predicate to check if a given number of additional games satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (initial_sharks_wins + n : ℚ) / (initial_games + n) ≥ 98 / 100

theorem min_additional_games_proof :
  satisfies_condition min_additional_games ∧
  ∀ m : ℕ, m < min_additional_games → ¬ satisfies_condition m :=
by sorry

end NUMINAMATH_CALUDE_min_additional_games_proof_l3735_373528


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3735_373556

theorem negative_fraction_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3735_373556


namespace NUMINAMATH_CALUDE_binomial_coefficient_8_4_l3735_373539

theorem binomial_coefficient_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_8_4_l3735_373539


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l3735_373572

-- Define the displacement function
def s (t : ℝ) : ℝ := t^2 + 10

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3s :
  v 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l3735_373572


namespace NUMINAMATH_CALUDE_xiaoming_win_probability_l3735_373515

/-- The probability of winning a single round for each player -/
def win_prob : ℚ := 1 / 2

/-- The number of rounds Xiaoming needs to win to ultimately win -/
def xiaoming_rounds_needed : ℕ := 2

/-- The number of rounds Xiaojie needs to win to ultimately win -/
def xiaojie_rounds_needed : ℕ := 3

/-- The probability that Xiaoming wins 2 consecutive rounds and ultimately wins -/
def xiaoming_win_prob : ℚ := 7 / 16

theorem xiaoming_win_probability : 
  xiaoming_win_prob = 
    win_prob ^ xiaoming_rounds_needed + 
    xiaoming_rounds_needed * win_prob ^ (xiaoming_rounds_needed + 1) + 
    win_prob ^ (xiaoming_rounds_needed + xiaojie_rounds_needed - 1) :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_win_probability_l3735_373515


namespace NUMINAMATH_CALUDE_calculate_expression_l3735_373584

theorem calculate_expression : (1/3)⁻¹ + (2023 - Real.pi)^0 - Real.sqrt 12 * Real.sin (π/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3735_373584


namespace NUMINAMATH_CALUDE_meal_price_before_coupon_l3735_373529

theorem meal_price_before_coupon
  (num_people : ℕ)
  (individual_contribution : ℝ)
  (coupon_value : ℝ)
  (h1 : num_people = 3)
  (h2 : individual_contribution = 21)
  (h3 : coupon_value = 4)
  : ↑num_people * individual_contribution + coupon_value = 67 :=
by sorry

end NUMINAMATH_CALUDE_meal_price_before_coupon_l3735_373529


namespace NUMINAMATH_CALUDE_all_positive_integers_are_dapper_l3735_373521

/-- A positive integer is dapper if at least one of its multiples begins with 2008. -/
def is_dapper (n : ℕ+) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), k * n.val = 2008 * 10^m + m ∧ m < 10^m

/-- Every positive integer is dapper. -/
theorem all_positive_integers_are_dapper : ∀ (n : ℕ+), is_dapper n := by
  sorry

end NUMINAMATH_CALUDE_all_positive_integers_are_dapper_l3735_373521


namespace NUMINAMATH_CALUDE_log_condition_l3735_373593

theorem log_condition (m : ℝ) (m_pos : m > 0) (m_neq_1 : m ≠ 1) :
  (∃ a b : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 → Real.log b / Real.log a > 0) ∧
  (∃ a b : ℝ, Real.log b / Real.log a > 0 ∧ ¬(0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1)) :=
by sorry

end NUMINAMATH_CALUDE_log_condition_l3735_373593


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3735_373576

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2

-- State the theorem
theorem tangent_line_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 * f (2 - x) - x^2 + 8*x - 8) : 
  ∃ m b, ∀ x, (x - 1) * (f 1) + m * (x - 1) = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3735_373576


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l3735_373558

-- Define a type for angles
def Angle : Type := ℝ

-- Define a function to create vertical angles
def verticalAngles (α β : Angle) : Prop := α = β

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (α β : Angle) (h : verticalAngles α β) : α = β := by
  sorry

-- Note: We don't define or assume anything about other angle relationships

end NUMINAMATH_CALUDE_vertical_angles_equal_l3735_373558


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l3735_373518

def word : String := "MISSISSIPPI"

def letter_counts : List (Char × Nat) := [('M', 1), ('I', 4), ('S', 4), ('P', 2)]

def total_letters : Nat := 11

def arrangements_starting_with_p : Nat := 6300

theorem mississippi_arrangements :
  (List.sum (letter_counts.map (fun p => p.2)) = total_letters) →
  (List.length letter_counts = 4) →
  (List.any letter_counts (fun p => p.1 = 'P' ∧ p.2 ≥ 1)) →
  (arrangements_starting_with_p = (Nat.factorial (total_letters - 1)) / 
    (List.prod (letter_counts.map (fun p => 
      if p.1 = 'P' then Nat.factorial (p.2 - 1) else Nat.factorial p.2)))) :=
by sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l3735_373518


namespace NUMINAMATH_CALUDE_paper_boats_problem_l3735_373583

theorem paper_boats_problem (initial_boats : ℕ) : 
  (initial_boats : ℝ) * 0.8 - 2 = 22 → initial_boats = 30 := by
  sorry

end NUMINAMATH_CALUDE_paper_boats_problem_l3735_373583


namespace NUMINAMATH_CALUDE_bag_contains_sixty_balls_l3735_373523

/-- The number of white balls in the bag -/
def white_balls : ℕ := 22

/-- The number of green balls in the bag -/
def green_balls : ℕ := 10

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 7

/-- The number of red balls in the bag -/
def red_balls : ℕ := 15

/-- The number of purple balls in the bag -/
def purple_balls : ℕ := 6

/-- The probability of choosing a ball that is neither red nor purple -/
def prob_not_red_or_purple : ℚ := 65/100

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + green_balls + yellow_balls + red_balls + purple_balls

theorem bag_contains_sixty_balls : total_balls = 60 := by
  sorry

end NUMINAMATH_CALUDE_bag_contains_sixty_balls_l3735_373523


namespace NUMINAMATH_CALUDE_secretary_work_time_l3735_373563

theorem secretary_work_time (x y z : ℕ) (h1 : x + y + z = 80) (h2 : 2 * x = 3 * y) (h3 : 2 * x = z) : z = 40 := by
  sorry

end NUMINAMATH_CALUDE_secretary_work_time_l3735_373563


namespace NUMINAMATH_CALUDE_pencil_distribution_l3735_373571

/-- The number of ways to distribute n identical objects among k people,
    where each person gets at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The total number of pencils -/
def total_pencils : ℕ := 9

/-- Each friend must have at least one pencil -/
def min_pencils_per_friend : ℕ := 1

theorem pencil_distribution :
  distribute (total_pencils - num_friends * min_pencils_per_friend + num_friends) num_friends = 28 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3735_373571


namespace NUMINAMATH_CALUDE_laborer_average_salary_l3735_373554

/-- Calculates the average monthly salary of laborers in a factory --/
theorem laborer_average_salary
  (total_workers : ℕ)
  (total_average_salary : ℚ)
  (num_supervisors : ℕ)
  (supervisor_average_salary : ℚ)
  (num_laborers : ℕ)
  (h_total_workers : total_workers = num_supervisors + num_laborers)
  (h_num_supervisors : num_supervisors = 6)
  (h_num_laborers : num_laborers = 42)
  (h_total_average_salary : total_average_salary = 1250)
  (h_supervisor_average_salary : supervisor_average_salary = 2450) :
  let laborer_total_salary := total_workers * total_average_salary - num_supervisors * supervisor_average_salary
  (laborer_total_salary / num_laborers) = 1078.57 := by
sorry

#eval (48 * 1250 - 6 * 2450) / 42

end NUMINAMATH_CALUDE_laborer_average_salary_l3735_373554


namespace NUMINAMATH_CALUDE_committee_formations_count_l3735_373570

/-- Represents a department in the division of mathematical sciences -/
inductive Department
| Mathematics
| Statistics
| ComputerScience

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents a professor with their department and gender -/
structure Professor :=
  (department : Department)
  (gender : Gender)

/-- The total number of departments -/
def num_departments : Nat := 3

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors required in the committee -/
def male_professors_in_committee : Nat := 4

/-- The number of female professors required in the committee -/
def female_professors_in_committee : Nat := 4

/-- Calculates the number of ways to form a committee satisfying all conditions -/
def count_committee_formations : Nat :=
  sorry

/-- Theorem stating that the number of possible committee formations is 59049 -/
theorem committee_formations_count :
  count_committee_formations = 59049 := by sorry

end NUMINAMATH_CALUDE_committee_formations_count_l3735_373570


namespace NUMINAMATH_CALUDE_zachary_crunches_l3735_373581

/-- Proves that Zachary did 58 crunches given the problem conditions -/
theorem zachary_crunches : 
  ∀ (zachary_pushups zachary_crunches david_pushups david_crunches : ℕ),
  zachary_pushups = 46 →
  david_pushups = zachary_pushups + 38 →
  david_crunches = zachary_crunches - 62 →
  zachary_crunches = zachary_pushups + 12 →
  zachary_crunches = 58 := by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_l3735_373581


namespace NUMINAMATH_CALUDE_sum_of_products_l3735_373504

theorem sum_of_products (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 9)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 13) :
  a * b + c * d = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3735_373504


namespace NUMINAMATH_CALUDE_volume_of_inscribed_cube_l3735_373511

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem volume_of_inscribed_cube (outer_cube_edge : ℝ) (h : outer_cube_edge = 16) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_edge : ℝ := sphere_diameter / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 12288 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_cube_l3735_373511


namespace NUMINAMATH_CALUDE_f_properties_l3735_373594

open Real

noncomputable def f (x : ℝ) := x * log x

theorem f_properties :
  ∀ (m : ℝ), m > 0 →
  (∀ (x : ℝ), x > 0 →
    (∃ (min_value : ℝ),
      (∀ (y : ℝ), y ∈ Set.Icc m (m + 2) → f y ≥ min_value) ∧
      ((0 < m ∧ m < exp (-1)) → min_value = -(exp (-1))) ∧
      (m ≥ exp (-1) → min_value = f m))) ∧
  (∀ (x : ℝ), x > 0 → f x > x / (exp x) - 2 / exp 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3735_373594


namespace NUMINAMATH_CALUDE_equivalence_conditions_l3735_373501

theorem equivalence_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x < y) ↔ (1 / x > 1 / y) ∧ (x - y < Real.cos x - Real.cos y) ∧ (Real.exp x - Real.exp y < x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_conditions_l3735_373501


namespace NUMINAMATH_CALUDE_marshas_delivery_problem_l3735_373550

/-- Marsha's delivery problem -/
theorem marshas_delivery_problem (x : ℝ) : 
  (x + 28 + 14) * 2 = 104 → x = 10 := by sorry

end NUMINAMATH_CALUDE_marshas_delivery_problem_l3735_373550


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3735_373538

-- Define the structure for an isosceles triangle
structure IsoscelesTriangle where
  side : ℕ  -- Equal sides
  base : ℕ  -- Base

-- Define the theorem
theorem min_perimeter_isosceles_triangles 
  (t1 t2 : IsoscelesTriangle) 
  (h1 : t1 ≠ t2)  -- Noncongruent triangles
  (h2 : 2 * t1.side + t1.base = 2 * t2.side + t2.base)  -- Same perimeter
  (h3 : t1.side * t1.base = t2.side * t2.base)  -- Same area (simplified)
  (h4 : 9 * t1.base = 8 * t2.base)  -- Ratio of bases
  : 2 * t1.side + t1.base ≥ 868 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3735_373538


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3735_373564

/-- The length of the hypotenuse of a right triangle with legs 140 and 336 units is 364 units. -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 140 →
  b = 336 →
  c^2 = a^2 + b^2 →
  c = 364 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3735_373564


namespace NUMINAMATH_CALUDE_coin_distribution_ways_l3735_373534

/-- The number of coin denominations available -/
def num_denominations : ℕ := 4

/-- The number of boys receiving coins -/
def num_boys : ℕ := 6

/-- Theorem stating the number of ways to distribute coins -/
theorem coin_distribution_ways : (num_denominations ^ num_boys : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_ways_l3735_373534


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3735_373517

theorem complex_fraction_equality : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3735_373517


namespace NUMINAMATH_CALUDE_max_cards_proof_l3735_373547

/-- The maximum number of trading cards Jasmine can buy --/
def max_cards : ℕ := 8

/-- Jasmine's initial budget --/
def initial_budget : ℚ := 15

/-- Cost per card --/
def card_cost : ℚ := 1.25

/-- Fixed transaction fee --/
def transaction_fee : ℚ := 2

/-- Minimum amount Jasmine wants to keep --/
def min_remaining : ℚ := 3

theorem max_cards_proof :
  (card_cost * max_cards + transaction_fee ≤ initial_budget - min_remaining) ∧
  (∀ n : ℕ, n > max_cards → card_cost * n + transaction_fee > initial_budget - min_remaining) :=
by sorry

end NUMINAMATH_CALUDE_max_cards_proof_l3735_373547


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3735_373560

/-- The probability of a license plate containing at least one palindrome -/
theorem license_plate_palindrome_probability :
  let total_arrangements : ℕ := 26^4 * 10^4
  let letter_palindromes : ℕ := 26^2
  let digit_palindromes : ℕ := 10^2
  let both_palindromes : ℕ := letter_palindromes * digit_palindromes
  let palindrome_probability : ℚ := (letter_palindromes * 10^4 + digit_palindromes * 26^4 - both_palindromes) / total_arrangements
  palindrome_probability = 775 / 67600 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3735_373560


namespace NUMINAMATH_CALUDE_train_length_l3735_373544

/-- The length of a train given its speed and time to pass a stationary observer -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 144 → time = 4 → speed * time * (1000 / 3600) = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3735_373544


namespace NUMINAMATH_CALUDE_angle_x_is_180_l3735_373566

-- Define the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC
  angle_ABC : Real
  angle_ACB : Real
  -- Straight angles
  angle_ADC_straight : Bool
  angle_AEB_straight : Bool

-- Theorem statement
theorem angle_x_is_180 (config : GeometricConfiguration) 
  (h1 : config.angle_ABC = 50)
  (h2 : config.angle_ACB = 70)
  (h3 : config.angle_ADC_straight = true)
  (h4 : config.angle_AEB_straight = true) :
  ∃ x : Real, x = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_x_is_180_l3735_373566


namespace NUMINAMATH_CALUDE_age_ratio_after_two_years_l3735_373502

/-- Given two people a and b, where their initial age ratio is 5:3 and b's age is 6,
    prove that their age ratio after 2 years is 3:2 -/
theorem age_ratio_after_two_years 
  (a b : ℕ) 
  (h1 : a = 5 * b / 3)  -- Initial ratio condition
  (h2 : b = 6)          -- b's initial age
  : (a + 2) / (b + 2) = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_after_two_years_l3735_373502


namespace NUMINAMATH_CALUDE_DE_DB_ratio_l3735_373548

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom right_angle_ABC : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
axiom AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4
axiom BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 3
axiom right_angle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
axiom AD_length : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 15
axiom C_D_opposite : ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) *
                     ((B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1)) < 0
axiom D_parallel_AC : (D.2 - A.2) * (E.1 - C.1) = (D.1 - A.1) * (E.2 - C.2)
axiom E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (B.1 - C.1), B.2 + t * (B.2 - C.2))

-- Define the theorem
theorem DE_DB_ratio :
  Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) / Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 57 / 80 :=
sorry

end NUMINAMATH_CALUDE_DE_DB_ratio_l3735_373548


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3735_373565

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / (2 * a) + 1 / b = 1) :
  ∀ x y, x > 0 → y > 0 → 1 / (2 * x) + 1 / y = 1 → a + 2 * b ≤ x + 2 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3735_373565


namespace NUMINAMATH_CALUDE_max_value_expression_l3735_373555

theorem max_value_expression (a b c x : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a - x) * (x + Real.sqrt (x^2 + b^2 + c)) ≤ a^2 + b^2 + c :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3735_373555


namespace NUMINAMATH_CALUDE_sally_has_five_balloons_l3735_373562

/-- The number of blue balloons Sally has -/
def sallys_balloons (total joan jessica : ℕ) : ℕ :=
  total - joan - jessica

/-- Theorem stating that Sally has 5 blue balloons given the conditions -/
theorem sally_has_five_balloons :
  sallys_balloons 16 9 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_five_balloons_l3735_373562


namespace NUMINAMATH_CALUDE_sector_area_special_case_l3735_373552

/-- The area of a sector with central angle 2π/3 and radius √3 is equal to π. -/
theorem sector_area_special_case :
  let central_angle : ℝ := 2 * Real.pi / 3
  let radius : ℝ := Real.sqrt 3
  let sector_area : ℝ := (1 / 2) * radius^2 * central_angle
  sector_area = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_special_case_l3735_373552


namespace NUMINAMATH_CALUDE_same_grade_percentage_l3735_373510

/-- Represents the grade distribution for two assignments --/
structure GradeDistribution :=
  (aa ab ac ad : ℕ)
  (ba bb bc bd : ℕ)
  (ca cb cc cd : ℕ)
  (da db dc dd : ℕ)

/-- The total number of students --/
def totalStudents : ℕ := 40

/-- The grade distribution for the English class --/
def englishClassDistribution : GradeDistribution :=
  { aa := 3, ab := 2, ac := 1, ad := 0,
    ba := 1, bb := 6, bc := 3, bd := 1,
    ca := 0, cb := 2, cc := 7, cd := 2,
    da := 0, db := 1, dc := 2, dd := 2 }

/-- Calculates the number of students who received the same grade on both assignments --/
def sameGradeCount (dist : GradeDistribution) : ℕ :=
  dist.aa + dist.bb + dist.cc + dist.dd

/-- Theorem: The percentage of students who received the same grade on both assignments is 45% --/
theorem same_grade_percentage :
  (sameGradeCount englishClassDistribution : ℚ) / totalStudents * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l3735_373510


namespace NUMINAMATH_CALUDE_exponent_addition_l3735_373591

theorem exponent_addition (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l3735_373591


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_zero_zero_in_range_l3735_373500

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 2) :
  (1 / (1 - x) + 1) / ((x^2 - 4*x + 4) / (x^2 - 1)) = (x + 1) / (x - 2) :=
by sorry

-- Evaluation at x = 0
theorem evaluate_at_zero :
  (1 / (1 - 0) + 1) / ((0^2 - 4*0 + 4) / (0^2 - 1)) = -1/2 :=
by sorry

-- Range constraint
def in_range (x : ℝ) : Prop := -2 < x ∧ x < 3

-- Proof that 0 is in the range
theorem zero_in_range : in_range 0 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_zero_zero_in_range_l3735_373500


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_l3735_373541

theorem reciprocal_sum_equality (a b c : ℝ) (n : ℕ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c ≠ 0) (h5 : Odd n) 
  (h6 : 1/a + 1/b + 1/c = 1/(a+b+c)) : 
  1/a^n + 1/b^n + 1/c^n = 1/(a^n + b^n + c^n) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_l3735_373541


namespace NUMINAMATH_CALUDE_x_range_l3735_373575

theorem x_range (x : ℝ) (h1 : 2 ≤ |x - 5| ∧ |x - 5| ≤ 10) (h2 : x > 0) :
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3735_373575


namespace NUMINAMATH_CALUDE_gcd_45678_12345_l3735_373516

theorem gcd_45678_12345 : Nat.gcd 45678 12345 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45678_12345_l3735_373516


namespace NUMINAMATH_CALUDE_cubic_polynomial_conditions_l3735_373553

def f (x : ℚ) : ℚ := 15 * x^3 - 37 * x^2 + 30 * x - 8

theorem cubic_polynomial_conditions :
  f 1 = 0 ∧ f (2/3) = -4 ∧ f (4/5) = -16/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_conditions_l3735_373553


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_l3735_373559

theorem reciprocal_sum_equality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (1 / x + 1 / y = 1 / z) → z = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_l3735_373559


namespace NUMINAMATH_CALUDE_linear_relation_holds_l3735_373549

def points : List (ℤ × ℤ) := [(0, 200), (1, 160), (2, 120), (3, 80), (4, 40)]

theorem linear_relation_holds (p : ℤ × ℤ) (h : p ∈ points) : 
  (p.2 : ℤ) = 200 - 40 * p.1 := by
  sorry

end NUMINAMATH_CALUDE_linear_relation_holds_l3735_373549


namespace NUMINAMATH_CALUDE_units_digit_of_2749_pow_987_l3735_373545

theorem units_digit_of_2749_pow_987 :
  (2749^987) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2749_pow_987_l3735_373545


namespace NUMINAMATH_CALUDE_pizza_fraction_l3735_373589

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice = 1/3 →
  (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l3735_373589


namespace NUMINAMATH_CALUDE_fraction_product_equality_l3735_373537

theorem fraction_product_equality : 
  (3 / 4) * (36 / 60) * (10 / 4) * (14 / 28) * (9 / 3)^2 * (45 / 15) * (12 / 18) * (20 / 40)^3 = 27 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l3735_373537


namespace NUMINAMATH_CALUDE_truck_distance_problem_l3735_373542

/-- Proves that the initial distance between two trucks is 1025 km given the problem conditions --/
theorem truck_distance_problem (speed_A speed_B : ℝ) (extra_distance : ℝ) :
  speed_A = 90 →
  speed_B = 80 →
  extra_distance = 145 →
  ∃ (time : ℝ), 
    time > 0 ∧
    speed_A * (time + 1) = speed_B * time + extra_distance ∧
    speed_A * (time + 1) + speed_B * time = 1025 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_problem_l3735_373542


namespace NUMINAMATH_CALUDE_average_first_15_even_numbers_l3735_373524

theorem average_first_15_even_numbers : 
  let first_15_even : List ℕ := List.range 15 |>.map (fun n => 2 * (n + 1))
  (first_15_even.sum : ℚ) / 15 = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_first_15_even_numbers_l3735_373524


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l3735_373514

theorem semicircle_area_ratio :
  let AB : ℝ := 10
  let AC : ℝ := 6
  let CB : ℝ := 4
  let large_semicircle_area : ℝ := (1/2) * Real.pi * (AB/2)^2
  let small_semicircle1_area : ℝ := (1/2) * Real.pi * (AC/2)^2
  let small_semicircle2_area : ℝ := (1/2) * Real.pi * (CB/2)^2
  let shaded_area : ℝ := large_semicircle_area - small_semicircle1_area - small_semicircle2_area
  let circle_area : ℝ := Real.pi * (CB/2)^2
  (shaded_area / circle_area) = (3/2) := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l3735_373514


namespace NUMINAMATH_CALUDE_reciprocal_and_square_properties_l3735_373525

theorem reciprocal_and_square_properties : 
  (∀ x : ℝ, x ≠ 0 → (x = 1/x ↔ x = 1 ∨ x = -1)) ∧ 
  (∀ x : ℝ, x = x^2 ↔ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_square_properties_l3735_373525


namespace NUMINAMATH_CALUDE_range_of_a_l3735_373567

-- Define the sets A, B, and C
def A : Set ℝ := {x | 0 < 2*x + 4 ∧ 2*x + 4 < 10}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0}

-- Define the union of A and B
def AUB : Set ℝ := {x | x ∈ A ∨ x ∈ B}

-- Define the complement of A ∪ B
def comp_AUB : Set ℝ := {x | x ∉ AUB}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ comp_AUB → x ∈ C a) → -2 < a ∧ a < -4/3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3735_373567


namespace NUMINAMATH_CALUDE_boat_current_rate_l3735_373519

/-- Proves that given a boat with a speed of 42 km/hr in still water,
    traveling 35.2 km downstream in 44 minutes, the rate of the current is 6 km/hr. -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 42)
  (h2 : distance = 35.2)
  (h3 : time = 44 / 60) : 
  ∃ (current_rate : ℝ), 
    current_rate = 6 ∧ 
    distance = (boat_speed + current_rate) * time :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l3735_373519


namespace NUMINAMATH_CALUDE_f_10_equals_107_l3735_373597

/-- The function f defined as f(n) = n^2 - n + 17 for all n -/
def f (n : ℕ) : ℕ := n^2 - n + 17

/-- Theorem stating that f(10) = 107 -/
theorem f_10_equals_107 : f 10 = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_10_equals_107_l3735_373597


namespace NUMINAMATH_CALUDE_ellipse_focus_y_axis_alpha_range_l3735_373540

/-- Represents an ellipse with equation x^2 * sin(α) - y^2 * cos(α) = 1 --/
structure Ellipse (α : Real) where
  equation : ∀ x y, x^2 * Real.sin α - y^2 * Real.cos α = 1

/-- Predicate to check if the focus of an ellipse is on the y-axis --/
def focus_on_y_axis (e : Ellipse α) : Prop :=
  1 / Real.sin α > 0 ∧ 1 / (-Real.cos α) > 0 ∧ 1 / Real.sin α < 1 / (-Real.cos α)

theorem ellipse_focus_y_axis_alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (e : Ellipse α) (h3 : focus_on_y_axis e) : 
  Real.pi / 2 < α ∧ α < 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_y_axis_alpha_range_l3735_373540


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l3735_373568

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l3735_373568


namespace NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l3735_373546

/-- The number of ways to arrange 6 distinct lessons into 6 time slots -/
def schedule_arrangements (total_lessons : ℕ) (morning_slots : ℕ) (afternoon_slots : ℕ) 
  (morning_constraint : ℕ) (afternoon_constraint : ℕ) : ℕ := 
  (morning_slots.choose morning_constraint) * 
  (afternoon_slots.choose afternoon_constraint) * 
  (Nat.factorial (total_lessons - morning_constraint - afternoon_constraint))

/-- Theorem stating that the number of schedule arrangements is 192 -/
theorem schedule_arrangements_eq_192 : 
  schedule_arrangements 6 4 2 1 1 = 192 := by
  sorry

end NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l3735_373546


namespace NUMINAMATH_CALUDE_probability_win_first_two_given_earn_3_l3735_373536

def win_probability : ℝ := 0.6

def points_for_win (sets_won : ℕ) : ℕ :=
  if sets_won = 3 then 3 else if sets_won = 2 then 2 else 0

def points_for_loss (sets_lost : ℕ) : ℕ :=
  if sets_lost = 2 then 1 else 0

def prob_win_3_0 : ℝ := win_probability ^ 3

def prob_win_3_1 : ℝ := 3 * (win_probability ^ 2) * (1 - win_probability) * win_probability

def prob_earn_3_points : ℝ := prob_win_3_0 + prob_win_3_1

def prob_win_first_two_and_earn_3 : ℝ := 
  prob_win_3_0 + (win_probability ^ 2) * (1 - win_probability) * win_probability

theorem probability_win_first_two_given_earn_3 :
  prob_win_first_two_and_earn_3 / prob_earn_3_points = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_win_first_two_given_earn_3_l3735_373536


namespace NUMINAMATH_CALUDE_sin_translation_l3735_373588

theorem sin_translation (x : ℝ) : 
  Real.sin (3 * x + π / 4) = Real.sin (3 * (x + π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_translation_l3735_373588


namespace NUMINAMATH_CALUDE_second_half_speed_l3735_373580

def total_distance : ℝ := 336
def total_time : ℝ := 15
def first_half_speed : ℝ := 21

theorem second_half_speed : ℝ := by
  have h1 : total_distance / 2 = first_half_speed * (total_time / 2) := by sorry
  have h2 : total_distance / 2 = 24 * (total_time - total_time / 2) := by sorry
  exact 24

end NUMINAMATH_CALUDE_second_half_speed_l3735_373580


namespace NUMINAMATH_CALUDE_gcd_4034_10085_base5_l3735_373595

/-- Converts a natural number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation -/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem gcd_4034_10085_base5 :
  let g := Nat.gcd 4034 10085
  isValidBase5 (toBase5 g) ∧ toBase5 g = [2, 3, 0, 1, 3] := by
  sorry

end NUMINAMATH_CALUDE_gcd_4034_10085_base5_l3735_373595


namespace NUMINAMATH_CALUDE_sin_210_degrees_l3735_373598

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l3735_373598


namespace NUMINAMATH_CALUDE_no_identical_lines_l3735_373506

theorem no_identical_lines : ¬∃ (d k : ℝ), ∀ (x y : ℝ),
  (4 * x + d * y + k = 0 ↔ k * x - 3 * y + 18 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_identical_lines_l3735_373506


namespace NUMINAMATH_CALUDE_parabola_distances_arithmetic_l3735_373582

/-- A parabola with focus F and three points A, B, C on it. -/
structure Parabola where
  p : ℝ
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  h_p_pos : 0 < p
  h_on_parabola_1 : y₁^2 = 2 * p * x₁
  h_on_parabola_2 : y₂^2 = 2 * p * x₂
  h_on_parabola_3 : y₃^2 = 2 * p * x₃
  h_arithmetic : ∃ d : ℝ, 
    (x₂ : ℝ) - x₁ = d ∧ 
    (x₃ : ℝ) - x₂ = d

/-- If the distances from A, B, C to the focus form an arithmetic sequence,
    then x₁, x₂, x₃ form an arithmetic sequence. -/
theorem parabola_distances_arithmetic (par : Parabola) :
  ∃ d : ℝ, (par.x₂ - par.x₁ = d) ∧ (par.x₃ - par.x₂ = d) := by
  sorry

end NUMINAMATH_CALUDE_parabola_distances_arithmetic_l3735_373582


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3735_373579

/-- The smallest area of a right triangle with sides 6 and 8 -/
theorem smallest_right_triangle_area :
  let sides : Finset ℝ := {6, 8}
  ∃ (a b c : ℝ), a ∈ sides ∧ b ∈ sides ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (∀ (x y z : ℝ), x ∈ sides → y ∈ sides → z > 0 → x^2 + y^2 = z^2 →
      (1/2) * a * b ≤ (1/2) * x * y) ∧
    (1/2) * a * b = 6 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3735_373579


namespace NUMINAMATH_CALUDE_sqrt_400_div_2_l3735_373543

theorem sqrt_400_div_2 : Real.sqrt 400 / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_400_div_2_l3735_373543


namespace NUMINAMATH_CALUDE_iphone_savings_l3735_373526

/-- Represents the cost of an iPhone X in dollars -/
def iphone_cost : ℝ := 600

/-- Represents the discount percentage for buying multiple smartphones -/
def discount_percentage : ℝ := 5

/-- Represents the number of iPhones being purchased -/
def num_iphones : ℕ := 3

/-- Theorem stating that the savings from buying 3 iPhones X together with a 5% discount,
    compared to buying them individually without a discount, is $90 -/
theorem iphone_savings :
  (num_iphones * iphone_cost) * (discount_percentage / 100) = 90 := by
  sorry

end NUMINAMATH_CALUDE_iphone_savings_l3735_373526


namespace NUMINAMATH_CALUDE_triangle_properties_l3735_373574

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c * Real.sin t.C = (2 * t.b + t.a) * Real.sin t.B + (2 * t.a - 3 * t.b) * Real.sin t.A)
  (h2 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h4 : t.A + t.B + t.C = π) : 
  (t.C = π / 3) ∧ 
  (t.c = 4 → 4 < t.a + t.b ∧ t.a + t.b ≤ 8) :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3735_373574


namespace NUMINAMATH_CALUDE_ice_cream_scoop_cost_l3735_373533

/-- The cost of a "Build Your Own Hot Brownie" dessert --/
structure BrownieDessert where
  brownieCost : ℝ
  syrupCost : ℝ
  nutsCost : ℝ
  iceCreamScoops : ℕ
  totalCost : ℝ

/-- The specific dessert order made by Juanita --/
def juanitaOrder : BrownieDessert where
  brownieCost := 2.50
  syrupCost := 0.50
  nutsCost := 1.50
  iceCreamScoops := 2
  totalCost := 7.00

/-- Theorem stating that each scoop of ice cream costs $1.00 --/
theorem ice_cream_scoop_cost (order : BrownieDessert) : 
  order.brownieCost = 2.50 →
  order.syrupCost = 0.50 →
  order.nutsCost = 1.50 →
  order.iceCreamScoops = 2 →
  order.totalCost = 7.00 →
  (order.totalCost - (order.brownieCost + 2 * order.syrupCost + order.nutsCost)) / order.iceCreamScoops = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_cost_l3735_373533


namespace NUMINAMATH_CALUDE_probability_theorem_l3735_373596

def is_valid (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  1 ≤ a ∧ a ≤ 12 ∧
  1 ≤ b ∧ b ≤ 12 ∧
  1 ≤ c ∧ c ≤ 12 ∧
  a = 2 * b ∧ b = 2 * c

def total_assignments : ℕ := 12 * 11 * 10

def valid_assignments : ℕ := 3

theorem probability_theorem :
  (valid_assignments : ℚ) / total_assignments = 1 / 440 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3735_373596


namespace NUMINAMATH_CALUDE_pizza_sales_l3735_373590

theorem pizza_sales (small_price large_price total_slices total_revenue : ℕ)
  (h1 : small_price = 150)
  (h2 : large_price = 250)
  (h3 : total_slices = 5000)
  (h4 : total_revenue = 1050000) :
  ∃ (small_slices large_slices : ℕ),
    small_slices + large_slices = total_slices ∧
    small_price * small_slices + large_price * large_slices = total_revenue ∧
    small_slices = 1500 :=
by sorry

end NUMINAMATH_CALUDE_pizza_sales_l3735_373590


namespace NUMINAMATH_CALUDE_fraction_equality_l3735_373551

theorem fraction_equality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) :
  (2 * a + b) / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3735_373551


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3735_373512

/-- The total surface area of a cylinder with diameter 9 and height 15 is 175.5π -/
theorem cylinder_surface_area :
  let d : ℝ := 9  -- diameter
  let h : ℝ := 15 -- height
  let r : ℝ := d / 2 -- radius
  let base_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  let total_area : ℝ := 2 * base_area + lateral_area
  total_area = 175.5 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3735_373512


namespace NUMINAMATH_CALUDE_concentric_circles_shaded_area_l3735_373507

/-- Given two concentric circles where the smaller circle's radius is half of the larger circle's radius,
    and the area of the larger circle is 144π, the sum of the areas of the upper halves of both circles
    is equal to 90π. -/
theorem concentric_circles_shaded_area (R r : ℝ) : 
  R > 0 ∧ r = R / 2 ∧ π * R^2 = 144 * π → 
  (π * R^2) / 2 + (π * r^2) / 2 = 90 * π := by
  sorry


end NUMINAMATH_CALUDE_concentric_circles_shaded_area_l3735_373507


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_three_halves_l3735_373586

-- Define the rational function
def f (x : ℚ) : ℚ := (2 * x + 3) / (6 * x - 9)

-- Theorem statement
theorem vertical_asymptote_at_three_halves :
  ∃ (ε : ℚ), ∀ (δ : ℚ), δ > 0 → ε > 0 → 
    ∀ (x : ℚ), 0 < |x - (3/2)| ∧ |x - (3/2)| < δ → |f x| > ε :=
sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_three_halves_l3735_373586


namespace NUMINAMATH_CALUDE_ball_distribution_l3735_373599

theorem ball_distribution (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  (Nat.choose (n + k - 1 - k) (k - 1)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_l3735_373599


namespace NUMINAMATH_CALUDE_weight_loss_duration_l3735_373587

/-- Represents the weight loss pattern over a 5-month cycle -/
structure WeightLossPattern :=
  (month1 : Int)
  (month2 : Int)
  (month3 : Int)
  (month4and5 : Int)

/-- Calculates the time needed to reach the target weight -/
def timeToReachTarget (initialWeight : Int) (pattern : WeightLossPattern) (targetWeight : Int) : Int :=
  sorry

/-- The theorem statement -/
theorem weight_loss_duration :
  let initialWeight := 222
  let pattern := WeightLossPattern.mk (-12) (-6) 2 (-8)
  let targetWeight := 170
  timeToReachTarget initialWeight pattern targetWeight = 6 :=
sorry

end NUMINAMATH_CALUDE_weight_loss_duration_l3735_373587


namespace NUMINAMATH_CALUDE_coronene_bond_arrangements_l3735_373503

/-- Represents a bond arrangement in coronene -/
structure CoroneneBondArrangement where
  /-- The number of double bonds in the center hexagon -/
  center_double_bonds : Fin 4
  /-- The configuration of double bonds in the outer ring -/
  outer_configuration : Nat

/-- Defines the validity of a bond arrangement in coronene -/
def is_valid_arrangement (arrangement : CoroneneBondArrangement) : Prop :=
  -- Each carbon has exactly 4 bonds
  -- Each hydrogen has exactly 1 bond
  -- The total number of bonds is consistent with the molecule's structure
  sorry

/-- Counts the number of valid bond arrangements in coronene -/
def count_valid_arrangements : Nat :=
  -- Count of arrangements that satisfy is_valid_arrangement
  sorry

/-- Theorem stating that the number of valid bond arrangements in coronene is 20 -/
theorem coronene_bond_arrangements :
  count_valid_arrangements = 20 :=
sorry

end NUMINAMATH_CALUDE_coronene_bond_arrangements_l3735_373503


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l3735_373508

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_inequality : 
  (¬ ∃ x : ℝ, x^2 > 2^x) ↔ (∀ x : ℝ, x^2 ≤ 2^x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l3735_373508


namespace NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_value_qin_jiushao_f_3_l3735_373557

-- Define the polynomial coefficients
def a₀ : ℝ := -0.8
def a₁ : ℝ := 1.7
def a₂ : ℝ := -2.6
def a₃ : ℝ := 3.5
def a₄ : ℝ := 2
def a₅ : ℝ := 4

-- Define Qin Jiushao's algorithm
def qin_jiushao (x : ℝ) : ℝ :=
  let v₀ := a₅
  let v₁ := v₀ * x + a₄
  let v₂ := v₁ * x + a₃
  let v₃ := v₂ * x + a₂
  let v₄ := v₃ * x + a₁
  v₄ * x + a₀

-- Define the polynomial function
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Theorem stating that Qin Jiushao's algorithm gives the correct result for f(3)
theorem qin_jiushao_correct : qin_jiushao 3 = f 3 := by sorry

-- Theorem stating that f(3) equals 1209.4
theorem f_3_value : f 3 = 1209.4 := by sorry

-- Main theorem combining the above results
theorem qin_jiushao_f_3 : qin_jiushao 3 = 1209.4 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_value_qin_jiushao_f_3_l3735_373557


namespace NUMINAMATH_CALUDE_solution_volume_l3735_373522

theorem solution_volume (V : ℝ) : 
  (0.20 * V + 0.60 * 4 = 0.36 * (V + 4)) → V = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l3735_373522


namespace NUMINAMATH_CALUDE_sum_greater_than_8_probability_l3735_373532

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum of two dice is 8 or less -/
def outcomes_8_or_less : ℕ := 26

/-- The probability that the sum of two dice is greater than 8 -/
def prob_sum_greater_than_8 : ℚ := 5 / 18

theorem sum_greater_than_8_probability :
  prob_sum_greater_than_8 = 1 - (outcomes_8_or_less : ℚ) / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_sum_greater_than_8_probability_l3735_373532


namespace NUMINAMATH_CALUDE_june_design_white_tiles_l3735_373531

/-- The number of white tiles in June's design -/
def white_tiles (total : ℕ) (yellow : ℕ) (purple : ℕ) : ℕ :=
  total - (yellow + (yellow + 1) + purple)

/-- Theorem stating the number of white tiles in June's design -/
theorem june_design_white_tiles :
  white_tiles 20 3 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_june_design_white_tiles_l3735_373531


namespace NUMINAMATH_CALUDE_m_range_l3735_373577

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom even_func : ∀ x : ℝ, f (-x) = f x
axiom increasing_neg : ∀ a b : ℝ, a < b → b < 0 → (f a - f b) / (a - b) > 0
axiom condition : ∀ m : ℝ, f (2*m + 1) > f (2*m)

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x : ℝ, f (-x) = f x) → 
  (∀ a b : ℝ, a < b → b < 0 → (f a - f b) / (a - b) > 0) → 
  (f (2*m + 1) > f (2*m)) → 
  m < -1/4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3735_373577


namespace NUMINAMATH_CALUDE_tan_150_degrees_l3735_373578

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l3735_373578


namespace NUMINAMATH_CALUDE_midpoint_locus_of_constant_area_segment_l3735_373535

/-- Given a line segment PQ with endpoints on the parabola y = x^2 such that the area bounded by PQ
    and the parabola is always 4/3, the locus of the midpoint M of PQ is described by the equation
    y = x^2 + 1 -/
theorem midpoint_locus_of_constant_area_segment (P Q : ℝ × ℝ) :
  (∃ α β : ℝ, α < β ∧ 
    P = (α, α^2) ∧ Q = (β, β^2) ∧ 
    (∫ (x : ℝ) in α..β, ((β - α)⁻¹ * ((β * α^2 - α * β^2) + (β - α) * x) - x^2)) = 4/3) →
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  M.2 = M.1^2 + 1 := by
sorry


end NUMINAMATH_CALUDE_midpoint_locus_of_constant_area_segment_l3735_373535


namespace NUMINAMATH_CALUDE_modified_lottery_win_probability_l3735_373585

/-- The number of balls for the MegaBall drawing -/
def megaBallCount : ℕ := 30

/-- The number of balls for the WinnerBalls drawing -/
def winnerBallCount : ℕ := 46

/-- The number of WinnerBalls picked -/
def pickedWinnerBallCount : ℕ := 5

/-- The probability of winning the modified lottery game -/
def winProbability : ℚ := 1 / 34321980

theorem modified_lottery_win_probability :
  winProbability = 1 / (megaBallCount * (Nat.choose winnerBallCount pickedWinnerBallCount)) :=
by sorry

end NUMINAMATH_CALUDE_modified_lottery_win_probability_l3735_373585


namespace NUMINAMATH_CALUDE_probability_third_defective_correct_probability_correct_under_conditions_l3735_373509

/-- Represents the probability of drawing a defective item as the third draw
    given the conditions of the problem. -/
def probability_third_defective (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ) : ℚ :=
  7 / 36

/-- Theorem stating the probability of drawing a defective item as the third draw
    under the given conditions. -/
theorem probability_third_defective_correct :
  probability_third_defective 10 3 3 = 7 / 36 := by
  sorry

/-- Checks if the conditions of the problem are met. -/
def valid_conditions (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ) : Prop :=
  total_items = 10 ∧ defective_items = 3 ∧ items_drawn = 3

/-- Theorem stating that the probability is correct under the given conditions. -/
theorem probability_correct_under_conditions
  (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ)
  (h : valid_conditions total_items defective_items items_drawn) :
  probability_third_defective total_items defective_items items_drawn = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_defective_correct_probability_correct_under_conditions_l3735_373509


namespace NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_4_l3735_373530

theorem max_gcd_14n_plus_5_9n_plus_4 :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n > 0 → Nat.gcd (14*n + 5) (9*n + 4) ≤ k ∧
  ∃ (m : ℕ), m > 0 ∧ Nat.gcd (14*m + 5) (9*m + 4) = k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_4_l3735_373530


namespace NUMINAMATH_CALUDE_select_five_from_eight_l3735_373520

/-- The number of combinations of n items taken r at a time -/
def combination (n r : ℕ) : ℕ := sorry

/-- Theorem stating that selecting 5 items from 8 items results in 56 combinations -/
theorem select_five_from_eight : combination 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l3735_373520


namespace NUMINAMATH_CALUDE_product_one_cube_sum_inequality_l3735_373592

theorem product_one_cube_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1/a + 1/b + 1/c + 1/d) := by
  sorry

end NUMINAMATH_CALUDE_product_one_cube_sum_inequality_l3735_373592


namespace NUMINAMATH_CALUDE_davids_math_marks_l3735_373505

/-- Given David's marks in various subjects and his average, prove his Mathematics marks --/
theorem davids_math_marks
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (num_subjects : ℕ)
  (h1 : english = 96)
  (h2 : physics = 82)
  (h3 : chemistry = 87)
  (h4 : biology = 92)
  (h5 : average = 90.4)
  (h6 : num_subjects = 5)
  : ∃ (math : ℕ), math = 95 ∧ 
    (english + math + physics + chemistry + biology) / num_subjects = average :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l3735_373505


namespace NUMINAMATH_CALUDE_negation_of_at_most_two_l3735_373573

theorem negation_of_at_most_two (P : ℕ → Prop) : 
  (¬ (∃ n : ℕ, P n ∧ (∀ m : ℕ, P m → m ≤ n) ∧ n ≤ 2)) ↔ 
  (∃ a b c : ℕ, P a ∧ P b ∧ P c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_negation_of_at_most_two_l3735_373573


namespace NUMINAMATH_CALUDE_tan_22_5_identity_l3735_373561

theorem tan_22_5_identity : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_identity_l3735_373561
