import Mathlib

namespace total_red_balloons_l546_54652

/-- The number of red balloons Sara has -/
def sara_red : ℕ := 31

/-- The number of green balloons Sara has -/
def sara_green : ℕ := 15

/-- The number of red balloons Sandy has -/
def sandy_red : ℕ := 24

/-- Theorem stating the total number of red balloons Sara and Sandy have -/
theorem total_red_balloons : sara_red + sandy_red = 55 := by
  sorry

end total_red_balloons_l546_54652


namespace tenth_term_is_19_l546_54614

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (a 3 = 5) ∧ (a 6 = 11) ∧ 
  ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m * d

/-- The 10th term of the arithmetic sequence is 19 -/
theorem tenth_term_is_19 (a : ℕ → ℝ) (h : arithmetic_sequence a) : 
  a 10 = 19 := by
  sorry

end tenth_term_is_19_l546_54614


namespace bicycle_profit_percentage_l546_54680

/-- Profit percentage calculation for bicycle sale --/
theorem bicycle_profit_percentage
  (cost_price_A : ℝ)
  (profit_percentage_A : ℝ)
  (final_price : ℝ)
  (h1 : cost_price_A = 144)
  (h2 : profit_percentage_A = 25)
  (h3 : final_price = 225) :
  let selling_price_A := cost_price_A * (1 + profit_percentage_A / 100)
  let profit_B := final_price - selling_price_A
  let profit_percentage_B := (profit_B / selling_price_A) * 100
  profit_percentage_B = 25 := by sorry

end bicycle_profit_percentage_l546_54680


namespace mika_stickers_problem_l546_54699

/-- The number of stickers Mika gave to her sister -/
def stickers_given_to_sister (initial bought birthday used left : ℕ) : ℕ :=
  initial + bought + birthday - used - left

theorem mika_stickers_problem (initial bought birthday used left : ℕ) 
  (h1 : initial = 20)
  (h2 : bought = 26)
  (h3 : birthday = 20)
  (h4 : used = 58)
  (h5 : left = 2) :
  stickers_given_to_sister initial bought birthday used left = 6 := by
sorry

end mika_stickers_problem_l546_54699


namespace raw_material_expenditure_l546_54653

theorem raw_material_expenditure (x : ℝ) :
  (x ≥ 0) →
  (x ≤ 1) →
  (1 - x - (1/10) * (1 - x) = 0.675) →
  (x = 1/4) :=
by sorry

end raw_material_expenditure_l546_54653


namespace x_age_is_63_l546_54646

/-- Given the ages of three people X, Y, and Z, prove that X's current age is 63 years. -/
theorem x_age_is_63 (x y z : ℕ) : 
  (x - 3 = 2 * (y - 3)) →  -- Three years ago, X's age was twice that of Y's age
  (y - 3 = 3 * (z - 3)) →  -- Three years ago, Y's age was three times that of Z's age
  ((x + 7) + (y + 7) + (z + 7) = 130) →  -- Seven years from now, the sum of their ages will be 130 years
  x = 63 := by
sorry

end x_age_is_63_l546_54646


namespace mario_poster_count_l546_54638

/-- The number of posters Mario made -/
def mario_posters : ℕ := 18

/-- The number of posters Samantha made -/
def samantha_posters : ℕ := mario_posters + 15

/-- The total number of posters made -/
def total_posters : ℕ := 51

theorem mario_poster_count : 
  mario_posters = 18 ∧ 
  samantha_posters = mario_posters + 15 ∧ 
  mario_posters + samantha_posters = total_posters :=
by sorry

end mario_poster_count_l546_54638


namespace craftsman_jars_l546_54603

theorem craftsman_jars (jars clay_pots : ℕ) (h1 : jars = 2 * clay_pots)
  (h2 : 5 * jars + 3 * 5 * clay_pots = 200) : jars = 16 := by
  sorry

end craftsman_jars_l546_54603


namespace ab_value_l546_54692

/-- The value of a letter in the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | _ => 0

/-- The number value of a word -/
def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

/-- Theorem: The number value of "ab" is 6 -/
theorem ab_value : word_value "ab" = 6 := by
  sorry

end ab_value_l546_54692


namespace percentage_b_grades_l546_54651

def scores : List Nat := [91, 82, 56, 99, 86, 95, 88, 79, 77, 68, 83, 81, 65, 84, 93, 72, 89, 78]

def is_b_grade (score : Nat) : Bool := 85 ≤ score ∧ score ≤ 93

def count_b_grades (scores : List Nat) : Nat :=
  scores.filter is_b_grade |>.length

theorem percentage_b_grades :
  let total_students := scores.length
  let b_grade_students := count_b_grades scores
  (b_grade_students : Rat) / total_students * 100 = 27.78 := by
  sorry

end percentage_b_grades_l546_54651


namespace complex_fraction_evaluation_l546_54605

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^5 + b^5) / (a + b)^5 = -2 := by
  sorry

end complex_fraction_evaluation_l546_54605


namespace cubic_equation_solution_l546_54686

theorem cubic_equation_solution (x : ℝ) : x^3 + 64 = 0 → x = -4 := by
  sorry

end cubic_equation_solution_l546_54686


namespace cube_plus_reciprocal_cube_l546_54628

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
sorry

end cube_plus_reciprocal_cube_l546_54628


namespace paths_avoiding_diagonal_l546_54662

/-- The number of paths on an 8x8 grid from corner to corner, avoiding a diagonal line --/
def num_paths : ℕ := sorry

/-- Binomial coefficient function --/
def binom (n k : ℕ) : ℕ := sorry

theorem paths_avoiding_diagonal :
  num_paths = binom 7 1 * binom 7 1 + (binom 7 3) ^ 2 := by sorry

end paths_avoiding_diagonal_l546_54662


namespace intersection_complement_equality_l546_54634

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {2,3,5,6}
def B : Set ℕ := {1,3,4,6,7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,5} := by sorry

end intersection_complement_equality_l546_54634


namespace quadratic_equation_solution_l546_54612

theorem quadratic_equation_solution (a b m : ℤ) : 
  (∀ x, a * x^2 + 24 * x + b = (m * x - 3)^2) → 
  (a = 16 ∧ b = 9 ∧ m = -4) := by
sorry

end quadratic_equation_solution_l546_54612


namespace arithmetic_square_root_of_36_l546_54687

theorem arithmetic_square_root_of_36 : 
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 36 ∧ x = 6 := by
  sorry

end arithmetic_square_root_of_36_l546_54687


namespace count_perfect_square_factors_4410_l546_54697

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors_4410 :
  prime_factorization 4410 = [(2, 1), (3, 2), (5, 1), (7, 2)] →
  count_perfect_square_factors 4410 = 4 := by sorry

end count_perfect_square_factors_4410_l546_54697


namespace simplify_fraction_with_sqrt_3_l546_54650

theorem simplify_fraction_with_sqrt_3 :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end simplify_fraction_with_sqrt_3_l546_54650


namespace expression_value_l546_54618

theorem expression_value : (fun x : ℝ => x^2 + 3*x - 4) 2 = 6 := by
  sorry

end expression_value_l546_54618


namespace remainder_theorem_l546_54658

theorem remainder_theorem : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_theorem_l546_54658


namespace medical_team_selection_l546_54689

theorem medical_team_selection (male_doctors : ℕ) (female_doctors : ℕ) : 
  male_doctors = 6 → female_doctors = 5 → 
  (male_doctors.choose 2) * (female_doctors.choose 1) = 75 := by
sorry

end medical_team_selection_l546_54689


namespace ratio_to_percentage_l546_54679

theorem ratio_to_percentage (x : ℝ) (h : x ≠ 0) :
  (x / 2) / (3 * x / 5) = 3 / 5 → (x / 2) / (3 * x / 5) * 100 = 60 := by
  sorry

end ratio_to_percentage_l546_54679


namespace remainder_7n_mod_4_l546_54670

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l546_54670


namespace intersection_of_A_and_B_l546_54608

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem intersection_of_A_and_B : A ∩ B = {3} := by sorry

end intersection_of_A_and_B_l546_54608


namespace reading_time_difference_l546_54674

/-- Given Xanthia's and Molly's reading speeds and a book length, 
    calculate the difference in reading time in minutes. -/
theorem reading_time_difference 
  (xanthia_speed molly_speed book_length : ℕ) 
  (hx : xanthia_speed = 120)
  (hm : molly_speed = 40)
  (hb : book_length = 360) : 
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 360 := by
  sorry

#check reading_time_difference

end reading_time_difference_l546_54674


namespace doudou_mother_age_l546_54665

/-- Represents the ages of Doudou's family members -/
structure FamilyAges where
  doudou : ℕ
  brother : ℕ
  mother : ℕ
  father : ℕ

/-- The conditions of the problem -/
def problemConditions (ages : FamilyAges) : Prop :=
  ages.brother = ages.doudou + 3 ∧
  ages.mother = ages.father - 2 ∧
  ages.doudou + ages.brother + ages.mother + ages.father - 20 = 59 ∧
  ages.doudou + ages.brother + ages.mother + ages.father + 20 = 97

/-- The theorem to be proved -/
theorem doudou_mother_age (ages : FamilyAges) :
  problemConditions ages → ages.mother = 33 := by
  sorry


end doudou_mother_age_l546_54665


namespace boys_to_girls_ratio_l546_54639

theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : total_students = 416) (h2 : girls = 160) :
  (total_students - girls) / girls = 8 / 5 := by
sorry

end boys_to_girls_ratio_l546_54639


namespace cat_kittens_count_l546_54672

/-- The number of kittens born to a cat, given specific weight conditions -/
def number_of_kittens (weight_two_lightest weight_four_heaviest total_weight : ℕ) : ℕ :=
  2 + 4 + (total_weight - weight_two_lightest - weight_four_heaviest) / ((weight_four_heaviest / 4 + weight_two_lightest / 2) / 2)

/-- Theorem stating that under the given conditions, the cat gave birth to 11 kittens -/
theorem cat_kittens_count :
  number_of_kittens 80 200 500 = 11 :=
by
  sorry

#eval number_of_kittens 80 200 500

end cat_kittens_count_l546_54672


namespace domain_of_g_l546_54690

-- Define the function f with domain (-1, 0)
def f : Set ℝ := { x : ℝ | -1 < x ∧ x < 0 }

-- Define the function g(x) = f(2x+1)
def g : Set ℝ := { x : ℝ | (2*x + 1) ∈ f }

-- Theorem statement
theorem domain_of_g : g = { x : ℝ | -1 < x ∧ x < -1/2 } := by sorry

end domain_of_g_l546_54690


namespace regular_square_prism_volume_l546_54654

theorem regular_square_prism_volume (h : ℝ) (sa : ℝ) (v : ℝ) : 
  h = 2 →
  sa = 12 * Real.pi →
  (∃ (r : ℝ), sa = 4 * Real.pi * r^2 ∧ 
    ∃ (a : ℝ), (2*r)^2 = 2*a^2 + h^2 ∧ 
    v = a^2 * h) →
  v = 8 := by sorry

end regular_square_prism_volume_l546_54654


namespace isosceles_triangle_base_endpoints_locus_l546_54637

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  A : Point
  B : Point
  C : Point
  centroid : Point
  orthocenter : Point
  isIsosceles : A.x = -B.x ∧ A.y = B.y
  centroidOrigin : centroid = ⟨0, 0⟩
  orthocenterOnYAxis : orthocenter = ⟨0, 1⟩
  thirdVertexOnYAxis : C.x = 0

/-- The locus of base endpoints of an isosceles triangle -/
def locusOfBaseEndpoints (p : Point) : Prop :=
  p.x ≠ 0 ∧ 3 * (p.y - 1/2)^2 - p.x^2 = 3/4

/-- Theorem stating that the base endpoints of the isosceles triangle lie on the specified locus -/
theorem isosceles_triangle_base_endpoints_locus (triangle : IsoscelesTriangle) :
  locusOfBaseEndpoints triangle.A ∧ locusOfBaseEndpoints triangle.B :=
sorry

end isosceles_triangle_base_endpoints_locus_l546_54637


namespace males_not_listening_l546_54626

/-- Represents the survey results -/
structure SurveyResults where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Theorem stating that the number of males who don't listen is 85 -/
theorem males_not_listening (survey : SurveyResults)
  (h1 : survey.total_listeners = 160)
  (h2 : survey.total_non_listeners = 180)
  (h3 : survey.female_listeners = 75)
  (h4 : survey.male_non_listeners = 85) :
  survey.male_non_listeners = 85 := by
  sorry

#check males_not_listening

end males_not_listening_l546_54626


namespace acme_profit_l546_54649

/-- Calculates the profit for Acme's horseshoe manufacturing --/
def calculate_profit (initial_outlay : ℝ) (cost_per_set : ℝ) (price_per_set : ℝ) (num_sets : ℕ) : ℝ :=
  let revenue := price_per_set * num_sets
  let total_cost := initial_outlay + cost_per_set * num_sets
  revenue - total_cost

/-- Theorem stating that Acme's profit is $15,337.50 --/
theorem acme_profit :
  calculate_profit 12450 20.75 50 950 = 15337.50 := by
  sorry

end acme_profit_l546_54649


namespace buses_needed_l546_54631

theorem buses_needed (students : ℕ) (seats_per_bus : ℕ) (h1 : students = 28) (h2 : seats_per_bus = 7) :
  (students + seats_per_bus - 1) / seats_per_bus = 4 := by
  sorry

end buses_needed_l546_54631


namespace computation_problem_points_l546_54682

theorem computation_problem_points :
  ∀ (total_problems : ℕ) 
    (computation_problems : ℕ) 
    (word_problem_points : ℕ) 
    (total_points : ℕ),
  total_problems = 30 →
  computation_problems = 20 →
  word_problem_points = 5 →
  total_points = 110 →
  ∃ (computation_problem_points : ℕ),
    computation_problem_points * computation_problems +
    word_problem_points * (total_problems - computation_problems) = total_points ∧
    computation_problem_points = 3 :=
by sorry

end computation_problem_points_l546_54682


namespace like_terms_exponent_difference_l546_54648

theorem like_terms_exponent_difference (a b : ℝ) (m n : ℤ) :
  (∃ k : ℝ, a^(m-2) * b^(n+7) = k * a^4 * b^4) →
  m - n = 9 := by sorry

end like_terms_exponent_difference_l546_54648


namespace smallest_dual_base_palindrome_fifteen_is_dual_base_palindrome_fifteen_is_smallest_dual_base_palindrome_l546_54656

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem smallest_dual_base_palindrome : 
  ∀ n : ℕ, n > 10 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n ≥ 15 :=
by sorry

theorem fifteen_is_dual_base_palindrome : 
  isPalindrome 15 2 ∧ isPalindrome 15 4 :=
by sorry

theorem fifteen_is_smallest_dual_base_palindrome : 
  ∀ n : ℕ, n > 10 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n = 15 :=
by sorry

end smallest_dual_base_palindrome_fifteen_is_dual_base_palindrome_fifteen_is_smallest_dual_base_palindrome_l546_54656


namespace fencing_requirement_l546_54615

theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) : 
  area = 680 → uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    uncovered_side + 2 * width = 88 := by
  sorry

end fencing_requirement_l546_54615


namespace smallest_fourth_number_l546_54647

theorem smallest_fourth_number (a b : ℕ) (h1 : a * 10 + b < 100) (h2 : a * 10 + b > 0) :
  (21 + 34 + 65 + a * 10 + b) = 4 * ((2 + 1 + 3 + 4 + 6 + 5 + a + b)) →
  12 ≤ a * 10 + b :=
by sorry

end smallest_fourth_number_l546_54647


namespace tank_capacity_l546_54610

/-- Represents a water tank with a certain capacity -/
structure WaterTank where
  capacity : ℝ
  emptyWeight : ℝ
  waterWeight : ℝ
  filledWeight : ℝ
  filledPercentage : ℝ

/-- Theorem stating that a tank with the given properties has a capacity of 200 gallons -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.emptyWeight = 80)
  (h2 : tank.waterWeight = 8)
  (h3 : tank.filledWeight = 1360)
  (h4 : tank.filledPercentage = 0.8) :
  tank.capacity = 200 := by
  sorry

#check tank_capacity

end tank_capacity_l546_54610


namespace quadratic_inequality_solution_set_l546_54655

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 8 < 0 ↔ -4/3 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_solution_set_l546_54655


namespace unique_solution_3x_4y_5z_l546_54695

theorem unique_solution_3x_4y_5z :
  ∀ x y z : ℕ+, 3^(x : ℕ) + 4^(y : ℕ) = 5^(z : ℕ) → x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end unique_solution_3x_4y_5z_l546_54695


namespace a_fourth_zero_implies_a_squared_zero_l546_54668

theorem a_fourth_zero_implies_a_squared_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
sorry

end a_fourth_zero_implies_a_squared_zero_l546_54668


namespace inequality_range_l546_54625

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) ↔ a ∈ Set.Iic 10 := by
  sorry

end inequality_range_l546_54625


namespace set_equality_implies_sum_l546_54604

theorem set_equality_implies_sum (a b : ℝ) : 
  ({-1, a} : Set ℝ) = ({b, 1} : Set ℝ) → a + b = 0 := by
  sorry

end set_equality_implies_sum_l546_54604


namespace quadratic_real_roots_l546_54661

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 3) * x^2 - 4 * x + 2 = 0) ↔ k ≤ 5 := by
  sorry

end quadratic_real_roots_l546_54661


namespace triangle_area_not_integer_l546_54611

theorem triangle_area_not_integer (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) 
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ¬ ∃ (S : ℕ), (S : ℝ)^2 * 16 = (a + b + c) * ((a + b + c) - 2*a) * ((a + b + c) - 2*b) * ((a + b + c) - 2*c) :=
sorry


end triangle_area_not_integer_l546_54611


namespace johnny_video_game_cost_l546_54664

/-- The amount Johnny spent on the video game -/
def video_game_cost (september_savings october_savings november_savings amount_left : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - amount_left

/-- Theorem: Johnny spent $58 on the video game -/
theorem johnny_video_game_cost :
  video_game_cost 30 49 46 67 = 58 := by
  sorry

end johnny_video_game_cost_l546_54664


namespace probability_is_half_l546_54698

/-- The probability of drawing either a red or blue marble from a bag -/
def probability_red_or_blue (red : ℕ) (blue : ℕ) (yellow : ℕ) : ℚ :=
  (red + blue : ℚ) / (red + blue + yellow)

/-- Theorem: The probability of drawing either a red or blue marble
    from a bag containing 3 red, 2 blue, and 5 yellow marbles is 1/2 -/
theorem probability_is_half :
  probability_red_or_blue 3 2 5 = 1/2 := by
  sorry

end probability_is_half_l546_54698


namespace teacher_worked_six_months_l546_54681

/-- Calculates the number of months a teacher has worked based on given conditions -/
def teacher_months_worked (periods_per_day : ℕ) (days_per_month : ℕ) (pay_per_period : ℕ) (total_earned : ℕ) : ℕ :=
  let daily_earnings := periods_per_day * pay_per_period
  let monthly_earnings := daily_earnings * days_per_month
  total_earned / monthly_earnings

/-- Theorem stating that the teacher has worked for 6 months given the specified conditions -/
theorem teacher_worked_six_months :
  teacher_months_worked 5 24 5 3600 = 6 := by
  sorry

end teacher_worked_six_months_l546_54681


namespace inequality_proof_l546_54669

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a * Real.sqrt (c^2 + 1))) + (1 / (b * Real.sqrt (a^2 + 1))) + (1 / (c * Real.sqrt (b^2 + 1))) > 2 := by
  sorry

end inequality_proof_l546_54669


namespace alissa_has_more_present_difference_l546_54684

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := 53

/-- Alissa has more presents than Ethan -/
theorem alissa_has_more : alissa_presents > ethan_presents := by sorry

/-- The difference between Alissa's and Ethan's presents is 22 -/
theorem present_difference : alissa_presents - ethan_presents = 22 := by sorry

end alissa_has_more_present_difference_l546_54684


namespace f_composition_value_l546_54602

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.sin x

theorem f_composition_value : f (f ((7 * Real.pi) / 6)) = Real.sqrt 2 / 2 := by
  sorry

end f_composition_value_l546_54602


namespace median_moons_theorem_l546_54635

/-- Represents the two categories of planets -/
inductive PlanetCategory
| Rocky
| GasGiant

/-- Represents a planet with its category and number of moons -/
structure Planet where
  name : String
  category : PlanetCategory
  moons : ℕ

/-- The list of all planets with their data -/
def planets : List Planet := [
  ⟨"Mercury", PlanetCategory.Rocky, 0⟩,
  ⟨"Venus", PlanetCategory.Rocky, 0⟩,
  ⟨"Earth", PlanetCategory.Rocky, 1⟩,
  ⟨"Mars", PlanetCategory.Rocky, 3⟩,
  ⟨"Jupiter", PlanetCategory.GasGiant, 20⟩,
  ⟨"Saturn", PlanetCategory.GasGiant, 25⟩,
  ⟨"Uranus", PlanetCategory.GasGiant, 17⟩,
  ⟨"Neptune", PlanetCategory.GasGiant, 3⟩,
  ⟨"Pluto", PlanetCategory.GasGiant, 8⟩
]

/-- Calculate the median number of moons for a given category -/
def medianMoons (category : PlanetCategory) : ℚ := sorry

/-- The theorem stating the median number of moons for each category -/
theorem median_moons_theorem :
  medianMoons PlanetCategory.Rocky = 1/2 ∧
  medianMoons PlanetCategory.GasGiant = 17 := by sorry

end median_moons_theorem_l546_54635


namespace class_size_proof_l546_54643

theorem class_size_proof (S : ℕ) : 
  S / 2 + S / 3 + 4 = S → S = 24 := by
  sorry

end class_size_proof_l546_54643


namespace possible_values_of_a_l546_54620

def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (a = -1/2 ∨ a = 0 ∨ a = -1) :=
by sorry

end possible_values_of_a_l546_54620


namespace combination_equation_solution_permutation_equation_solution_l546_54633

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Permutation (Arrangement) -/
def permutation (n k : ℕ) : ℕ := sorry

theorem combination_equation_solution :
  ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9 →
  (binomial 9 x = binomial 9 (2*x - 3)) ↔ (x = 3 ∨ x = 4) := by sorry

theorem permutation_equation_solution :
  ∀ x : ℕ, 0 < x ∧ x ≤ 8 →
  (permutation 8 x = 6 * permutation 8 (x - 2)) ↔ x = 7 := by sorry

end combination_equation_solution_permutation_equation_solution_l546_54633


namespace ratio_equation_solution_sum_l546_54600

theorem ratio_equation_solution_sum : 
  ∃! s : ℝ, ∀ x : ℝ, (3 * x + 4) / (5 * x + 4) = (5 * x + 6) / (8 * x + 6) → s = x :=
by
  sorry

end ratio_equation_solution_sum_l546_54600


namespace smallest_third_number_lcm_l546_54619

/-- The lowest common multiple of a list of natural numbers -/
def lcm_list (l : List Nat) : Nat :=
  l.foldl Nat.lcm 1

/-- The theorem states that 10 is the smallest positive integer x
    such that the LCM of 24, 30, and x is 120 -/
theorem smallest_third_number_lcm :
  (∀ x : Nat, x > 0 → x < 10 → lcm_list [24, 30, x] ≠ 120) ∧
  lcm_list [24, 30, 10] = 120 := by
  sorry

end smallest_third_number_lcm_l546_54619


namespace slope_of_line_from_equation_l546_54616

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop := 3 / x + 4 / y = 0

-- Theorem statement
theorem slope_of_line_from_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ →
    satisfies_equation x₁ y₁ →
    satisfies_equation x₂ y₂ →
    (y₂ - y₁) / (x₂ - x₁) = -4/3 :=
by sorry

end slope_of_line_from_equation_l546_54616


namespace fiftieth_rising_number_excludes_one_three_four_l546_54632

/-- A rising number is a number where each digit is strictly greater than the previous digit. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The set of digits used to construct the rising numbers. -/
def DigitSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The function that generates the nth four-digit rising number from the DigitSet. -/
def NthRisingNumber (n : ℕ) : Finset ℕ := sorry

/-- Theorem stating that the 50th rising number does not contain 1, 3, or 4. -/
theorem fiftieth_rising_number_excludes_one_three_four :
  1 ∉ NthRisingNumber 50 ∧ 3 ∉ NthRisingNumber 50 ∧ 4 ∉ NthRisingNumber 50 := by
  sorry

end fiftieth_rising_number_excludes_one_three_four_l546_54632


namespace fifteen_students_in_neither_l546_54641

/-- Represents the number of students in different categories of a robotics club. -/
structure RoboticsClub where
  total : ℕ
  cs : ℕ
  electronics : ℕ
  both : ℕ

/-- Calculates the number of students taking neither computer science nor electronics. -/
def studentsInNeither (club : RoboticsClub) : ℕ :=
  club.total - (club.cs + club.electronics - club.both)

/-- Theorem stating that 15 students take neither computer science nor electronics. -/
theorem fifteen_students_in_neither (club : RoboticsClub)
  (h1 : club.total = 80)
  (h2 : club.cs = 52)
  (h3 : club.electronics = 38)
  (h4 : club.both = 25) :
  studentsInNeither club = 15 := by
  sorry

end fifteen_students_in_neither_l546_54641


namespace expression_simplification_l546_54663

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 2 + 1) 
  (hb : b = Real.sqrt 2 - 1) : 
  (a^2 - b^2) / a / (a + (2*a*b + b^2) / a) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l546_54663


namespace det_A_eq_121_l546_54623

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 4, 5, -3; 6, 2, 7]

theorem det_A_eq_121 : A.det = 121 := by
  sorry

end det_A_eq_121_l546_54623


namespace product_mod_seven_l546_54640

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end product_mod_seven_l546_54640


namespace box_values_equality_l546_54622

theorem box_values_equality : 40506000 = 4 * 10000000 + 5 * 100000 + 6 * 1000 := by
  sorry

end box_values_equality_l546_54622


namespace average_lawn_cuts_per_month_l546_54660

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months Mr. Roper cuts his lawn 15 times per month -/
def high_frequency_months : ℕ := 6

/-- The number of months Mr. Roper cuts his lawn 3 times per month -/
def low_frequency_months : ℕ := 6

/-- The number of times Mr. Roper cuts his lawn in high frequency months -/
def high_frequency_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn in low frequency months -/
def low_frequency_cuts : ℕ := 3

/-- Theorem stating that the average number of times Mr. Roper cuts his lawn per month is 9 -/
theorem average_lawn_cuts_per_month :
  (high_frequency_months * high_frequency_cuts + low_frequency_months * low_frequency_cuts) / months_in_year = 9 := by
  sorry

end average_lawn_cuts_per_month_l546_54660


namespace cubes_after_removing_layer_l546_54675

/-- The number of smaller cubes in one dimension of the large cube -/
def cube_dimension : ℕ := 10

/-- The total number of smaller cubes in the large cube -/
def total_cubes : ℕ := cube_dimension ^ 3

/-- The number of smaller cubes in one layer -/
def layer_cubes : ℕ := cube_dimension ^ 2

/-- Theorem: Removing one layer from a cube of 10x10x10 smaller cubes leaves 900 cubes -/
theorem cubes_after_removing_layer :
  total_cubes - layer_cubes = 900 := by
  sorry


end cubes_after_removing_layer_l546_54675


namespace units_digit_17_310_l546_54667

/-- The units digit of 7^n for n ≥ 1 -/
def unitsDigit7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | 0 => 1
  | _ => 0  -- This case should never occur

/-- The units digit of 17^n follows the same pattern as 7^n -/
axiom unitsDigit17 (n : ℕ) : n ≥ 1 → unitsDigit7 n = (17^n) % 10

theorem units_digit_17_310 : (17^310) % 10 = 9 := by
  sorry

end units_digit_17_310_l546_54667


namespace sara_pears_l546_54666

theorem sara_pears (total_pears sally_pears : ℕ) 
  (h1 : total_pears = 56)
  (h2 : sally_pears = 11) :
  total_pears - sally_pears = 45 := by
  sorry

end sara_pears_l546_54666


namespace translation_of_sine_to_cosine_l546_54609

/-- Given a function f(x) = sin(2x + π/6), prove that translating it π/6 units to the left
    results in the function g(x) = cos(2x) -/
theorem translation_of_sine_to_cosine (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 6)
  let g : ℝ → ℝ := λ x => f (x + π / 6)
  g x = Real.cos (2 * x) := by
  sorry

end translation_of_sine_to_cosine_l546_54609


namespace min_distance_to_line_l546_54659

theorem min_distance_to_line (x y : ℝ) (h : 2 * x + y + 5 = 0) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), 2 * x' + y' + 5 = 0 → Real.sqrt (x'^2 + y'^2) ≥ min_dist :=
sorry

end min_distance_to_line_l546_54659


namespace complex_equation_solution_l546_54685

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = Complex.I + z) :
  z = 1/2 - (1/2) * Complex.I :=
by sorry

end complex_equation_solution_l546_54685


namespace max_log_sum_l546_54617

theorem max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2*x + y = 20) :
  ∃ (max_val : ℝ), max_val = 2 - Real.log 2 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a + b = 20 → Real.log a + Real.log b ≤ max_val :=
sorry

end max_log_sum_l546_54617


namespace unique_solution_cube_difference_square_l546_54644

/-- A predicate that checks if a number is prime -/
def IsPrime (p : ℕ) : Prop := Nat.Prime p

/-- A predicate that checks if a number is not divisible by 3 or by another number -/
def NotDivisibleBy3OrY (z y : ℕ) : Prop := ¬(z % 3 = 0) ∧ ¬(z % y = 0)

theorem unique_solution_cube_difference_square :
  ∀ x y z : ℕ,
    x > 0 → y > 0 → z > 0 →
    IsPrime y →
    NotDivisibleBy3OrY z y →
    x^3 - y^3 = z^2 →
    x = 8 ∧ y = 7 ∧ z = 13 :=
by sorry

end unique_solution_cube_difference_square_l546_54644


namespace right_triangle_cotangent_l546_54629

theorem right_triangle_cotangent (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 12) :
  a / b = 12 / 5 := by
  sorry

end right_triangle_cotangent_l546_54629


namespace buttons_given_to_mary_l546_54645

theorem buttons_given_to_mary (initial_buttons : ℕ) (buttons_left : ℕ) : initial_buttons - buttons_left = 4 :=
by
  sorry

#check buttons_given_to_mary 9 5

end buttons_given_to_mary_l546_54645


namespace james_lifting_ratio_l546_54606

theorem james_lifting_ratio :
  let initial_total : ℝ := 2200
  let initial_weight : ℝ := 245
  let total_gain_percent : ℝ := 0.15
  let weight_gain : ℝ := 8
  let final_total : ℝ := initial_total * (1 + total_gain_percent)
  let final_weight : ℝ := initial_weight + weight_gain
  final_total / final_weight = 10
  := by sorry

end james_lifting_ratio_l546_54606


namespace meadow_orders_30_boxes_l546_54607

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  packs_per_box : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ
  total_revenue : ℕ

/-- Calculates the number of boxes ordered weekly --/
def boxes_ordered (business : DiaperBusiness) : ℕ :=
  business.total_revenue / (business.price_per_diaper * business.diapers_per_pack * business.packs_per_box)

/-- Theorem: Given the conditions, Meadow orders 30 boxes weekly --/
theorem meadow_orders_30_boxes :
  let business : DiaperBusiness := {
    packs_per_box := 40,
    diapers_per_pack := 160,
    price_per_diaper := 5,
    total_revenue := 960000
  }
  boxes_ordered business = 30 := by
  sorry

end meadow_orders_30_boxes_l546_54607


namespace population_change_theorem_l546_54636

/-- Represents the population change factor for a given percentage change -/
def change_factor (percent : ℚ) : ℚ := 1 + percent / 100

/-- Calculates the net change in population over 5 years given the yearly changes -/
def net_change (year1 year2 year3 year4 year5 : ℚ) : ℚ :=
  (change_factor year1 * change_factor year2 * change_factor year3 * 
   change_factor year4 * change_factor year5 - 1) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  if q - ⌊q⌋ < 1/2 then ⌊q⌋ else ⌈q⌉

theorem population_change_theorem :
  round_to_nearest (net_change 20 10 (-30) (-20) 10) = -19 := by sorry

end population_change_theorem_l546_54636


namespace johns_zoo_l546_54673

theorem johns_zoo (snakes : ℕ) (monkeys : ℕ) (lions : ℕ) (pandas : ℕ) (dogs : ℕ) :
  snakes = 15 ∧
  monkeys = 2 * snakes ∧
  lions = monkeys - 5 ∧
  pandas = lions + 8 ∧
  dogs = pandas / 3 →
  snakes + monkeys + lions + pandas + dogs = 114 := by
sorry

end johns_zoo_l546_54673


namespace quadratic_roots_sum_squares_l546_54624

theorem quadratic_roots_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*h*x = 8 ∧ y^2 + 4*h*y = 8 ∧ x^2 + y^2 = 20) →
  |h| = 1/2 := by
sorry

end quadratic_roots_sum_squares_l546_54624


namespace fraction_equality_l546_54601

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (5 * x + 2 * y) / (x - 5 * y) = 3) : 
  (x + 5 * y) / (5 * x - y) = 7 / 87 := by
  sorry

end fraction_equality_l546_54601


namespace g_of_2_eq_3_l546_54678

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem g_of_2_eq_3 : g 2 = 3 := by sorry

end g_of_2_eq_3_l546_54678


namespace monomial_sum_implies_m_plus_n_eq_3_l546_54693

/-- Two algebraic expressions form a monomial when added together if they have the same powers for each variable -/
def forms_monomial (expr1 expr2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), expr1 x y ≠ 0 ∧ expr2 x y ≠ 0 → x = y

/-- The first algebraic expression: 3a^m * b^2 -/
def expr1 (m : ℕ) (a b : ℕ) : ℚ := 3 * (a^m) * (b^2)

/-- The second algebraic expression: -2a^2 * b^(n+1) -/
def expr2 (n : ℕ) (a b : ℕ) : ℚ := -2 * (a^2) * (b^(n+1))

theorem monomial_sum_implies_m_plus_n_eq_3 (m n : ℕ) :
  forms_monomial (expr1 m) (expr2 n) → m + n = 3 := by
  sorry

end monomial_sum_implies_m_plus_n_eq_3_l546_54693


namespace total_hockey_games_l546_54696

/-- The number of hockey games in a season -/
def hockey_season_games (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem: The total number of hockey games in the season is 182 -/
theorem total_hockey_games : hockey_season_games 13 14 = 182 := by
  sorry

end total_hockey_games_l546_54696


namespace robin_gum_count_l546_54642

/-- The number of gum pieces Robin has after his purchases -/
def total_gum_pieces : ℕ :=
  let initial_packages := 27
  let initial_pieces_per_package := 18
  let additional_packages_1 := 15
  let additional_pieces_per_package_1 := 12
  let additional_packages_2 := 8
  let additional_pieces_per_package_2 := 25
  initial_packages * initial_pieces_per_package +
  additional_packages_1 * additional_pieces_per_package_1 +
  additional_packages_2 * additional_pieces_per_package_2

theorem robin_gum_count : total_gum_pieces = 866 := by
  sorry

end robin_gum_count_l546_54642


namespace adjacent_chair_subsets_l546_54613

/-- Given 12 chairs arranged in a circle, this function calculates the number of subsets
    containing at least three adjacent chairs. -/
def subsets_with_adjacent_chairs (num_chairs : ℕ) : ℕ :=
  if num_chairs = 12 then
    2010
  else
    0

/-- Theorem stating that for 12 chairs in a circle, there are 2010 subsets
    with at least three adjacent chairs. -/
theorem adjacent_chair_subsets :
  subsets_with_adjacent_chairs 12 = 2010 := by
  sorry

#eval subsets_with_adjacent_chairs 12

end adjacent_chair_subsets_l546_54613


namespace line_graph_most_suitable_l546_54694

/-- Represents different types of statistical graphs -/
inductive StatisticalGraph
| BarGraph
| PieChart
| LineGraph
| FrequencyDistributionGraph

/-- Characteristics of a statistical graph -/
structure GraphCharacteristics where
  showsTrend : Bool
  showsTimeProgression : Bool
  comparesCategories : Bool
  showsProportions : Bool
  showsFrequency : Bool

/-- Define the characteristics of each graph type -/
def graphProperties : StatisticalGraph → GraphCharacteristics
| StatisticalGraph.BarGraph => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := true,
    showsProportions := false,
    showsFrequency := false
  }
| StatisticalGraph.PieChart => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := false,
    showsProportions := true,
    showsFrequency := false
  }
| StatisticalGraph.LineGraph => {
    showsTrend := true,
    showsTimeProgression := true,
    comparesCategories := false,
    showsProportions := false,
    showsFrequency := false
  }
| StatisticalGraph.FrequencyDistributionGraph => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := false,
    showsProportions := false,
    showsFrequency := true
  }

/-- Defines the requirements for a graph to show temperature trends over a week -/
def suitableForTemperatureTrend (g : GraphCharacteristics) : Prop :=
  g.showsTrend ∧ g.showsTimeProgression

/-- Theorem stating that a line graph is the most suitable for showing temperature trends over a week -/
theorem line_graph_most_suitable :
  ∀ (g : StatisticalGraph), 
    suitableForTemperatureTrend (graphProperties g) → g = StatisticalGraph.LineGraph := by
  sorry

end line_graph_most_suitable_l546_54694


namespace ladder_geometric_sequence_a10_l546_54671

/-- A sequence {a_n} is an m-th order ladder geometric sequence if it satisfies
    a_{n+m}^2 = a_n × a_{n+2m} for any positive integers n and m. -/
def is_ladder_geometric (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∀ n : ℕ, (a (n + m))^2 = a n * a (n + 2*m)

theorem ladder_geometric_sequence_a10 (a : ℕ → ℝ) :
  is_ladder_geometric a 3 → a 1 = 1 → a 4 = 2 → a 10 = 8 := by
  sorry

end ladder_geometric_sequence_a10_l546_54671


namespace no_real_roots_for_geometric_sequence_quadratic_l546_54683

/-- If a, b, c form a geometric sequence, then ax^2 + bx + c = 0 has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) (h : b^2 = a*c) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end no_real_roots_for_geometric_sequence_quadratic_l546_54683


namespace fifth_number_in_row_l546_54657

-- Define Pascal's triangle
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the row we're interested in
def targetRow : ℕ → ℕ
  | 0 => 1
  | 1 => 15
  | k => pascal 15 (k - 1)

-- State the theorem
theorem fifth_number_in_row : targetRow 5 = 1365 := by
  sorry

end fifth_number_in_row_l546_54657


namespace isosceles_triangles_independent_of_coloring_l546_54677

/-- The number of isosceles triangles with vertices of the same color in a regular (6n+1)-gon -/
def num_isosceles_triangles (n : ℕ) (K : ℕ) : ℕ :=
  (1/2) * ((6*n+1 - K)*(6*n - K) + K*(K-1) - K*(6*n+1-K))

/-- Theorem stating that the number of isosceles triangles with vertices of the same color
    in a regular (6n+1)-gon is independent of the coloring scheme -/
theorem isosceles_triangles_independent_of_coloring (n : ℕ) (K : ℕ) 
    (h1 : K ≤ 6*n+1) : 
  ∀ (K' : ℕ), K' ≤ 6*n+1 → num_isosceles_triangles n K = num_isosceles_triangles n K' :=
by sorry

end isosceles_triangles_independent_of_coloring_l546_54677


namespace jessie_weight_loss_l546_54676

/-- Calculates the weight loss for Jessie based on her exercise routine --/
def weight_loss (initial_weight : ℝ) (exercise_days : ℕ) (even_day_loss : ℝ) (odd_day_loss : ℝ) : ℝ :=
  let even_days := (exercise_days - 1) / 2
  let odd_days := exercise_days - even_days
  even_days * even_day_loss + odd_days * odd_day_loss

/-- Theorem stating that Jessie's weight loss is 8.1 kg --/
theorem jessie_weight_loss :
  let initial_weight : ℝ := 74
  let exercise_days : ℕ := 25
  let even_day_loss : ℝ := 0.2 + 0.15
  let odd_day_loss : ℝ := 0.3
  weight_loss initial_weight exercise_days even_day_loss odd_day_loss = 8.1 := by
  sorry

#eval weight_loss 74 25 (0.2 + 0.15) 0.3

end jessie_weight_loss_l546_54676


namespace current_speed_is_correct_l546_54627

/-- Represents the speed of a swimmer in still water -/
def swimmer_speed : ℝ := 6.5

/-- Represents the speed of the current -/
def current_speed : ℝ := 4.5

/-- Represents the distance traveled downstream -/
def downstream_distance : ℝ := 55

/-- Represents the distance traveled upstream -/
def upstream_distance : ℝ := 10

/-- Represents the time taken for both downstream and upstream journeys -/
def travel_time : ℝ := 5

/-- Theorem stating that given the conditions, the speed of the current is 4.5 km/h -/
theorem current_speed_is_correct : 
  downstream_distance / travel_time = swimmer_speed + current_speed ∧
  upstream_distance / travel_time = swimmer_speed - current_speed →
  current_speed = 4.5 := by
  sorry

#check current_speed_is_correct

end current_speed_is_correct_l546_54627


namespace largest_n_satisfying_inequality_l546_54691

theorem largest_n_satisfying_inequality : 
  (∀ n : ℕ, n^6033 < 2011^2011 → n ≤ 12) ∧ 12^6033 < 2011^2011 := by
  sorry

end largest_n_satisfying_inequality_l546_54691


namespace xiao_dong_jump_record_l546_54688

/-- Represents the recording of a long jump result -/
def record_jump (standard : ℝ) (jump : ℝ) : ℝ :=
  jump - standard

/-- The standard for the long jump -/
def long_jump_standard : ℝ := 4.00

/-- Xiao Dong's jump distance -/
def xiao_dong_jump : ℝ := 3.85

/-- Theorem stating how Xiao Dong's jump should be recorded -/
theorem xiao_dong_jump_record :
  record_jump long_jump_standard xiao_dong_jump = -0.15 := by
  sorry

end xiao_dong_jump_record_l546_54688


namespace problem_solution_l546_54621

theorem problem_solution (a b c : ℝ) 
  (h1 : |a - 4| + |b + 5| = 0) 
  (h2 : a + c = 0) : 
  3*a + 2*b - 4*c = 18 := by
sorry

end problem_solution_l546_54621


namespace cubic_polynomial_coefficient_l546_54630

theorem cubic_polynomial_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end cubic_polynomial_coefficient_l546_54630
