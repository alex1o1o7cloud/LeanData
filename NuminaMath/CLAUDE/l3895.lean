import Mathlib

namespace remainder_problem_l3895_389540

theorem remainder_problem (N : ℤ) : N % 1927 = 131 → (3 * N) % 43 = 6 := by
  sorry

end remainder_problem_l3895_389540


namespace points_earned_is_thirteen_l3895_389573

/-- VideoGame represents the state of the game --/
structure VideoGame where
  totalEnemies : Nat
  redEnemies : Nat
  blueEnemies : Nat
  defeatedEnemies : Nat
  hits : Nat
  pointsPerEnemy : Nat
  bonusPoints : Nat
  pointsLostPerHit : Nat

/-- Calculate the total points earned in the game --/
def calculatePoints (game : VideoGame) : Int :=
  let basePoints := game.defeatedEnemies * game.pointsPerEnemy
  let bonusEarned := if (game.redEnemies - 1 > 0) && (game.blueEnemies - 1 > 0) then game.bonusPoints else 0
  let totalEarned := basePoints + bonusEarned
  let pointsLost := game.hits * game.pointsLostPerHit
  totalEarned - pointsLost

/-- Theorem stating that given the game conditions, the total points earned is 13 --/
theorem points_earned_is_thirteen :
  ∀ (game : VideoGame),
    game.totalEnemies = 6 →
    game.redEnemies = 3 →
    game.blueEnemies = 3 →
    game.defeatedEnemies = 4 →
    game.hits = 2 →
    game.pointsPerEnemy = 3 →
    game.bonusPoints = 5 →
    game.pointsLostPerHit = 2 →
    calculatePoints game = 13 := by
  sorry

end points_earned_is_thirteen_l3895_389573


namespace chocolate_boxes_pieces_per_box_l3895_389543

theorem chocolate_boxes_pieces_per_box 
  (initial_boxes : ℕ) 
  (given_away_boxes : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_boxes = 14)
  (h2 : given_away_boxes = 5)
  (h3 : remaining_pieces = 54)
  (h4 : initial_boxes > given_away_boxes) :
  (remaining_pieces / (initial_boxes - given_away_boxes) = 6) :=
by sorry

end chocolate_boxes_pieces_per_box_l3895_389543


namespace inequality_equivalence_l3895_389508

theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  x^(0.5 * Real.log x / Real.log 0.5 - 3) ≥ 0.5^(3 - 2.5 * Real.log x / Real.log 0.5) ↔ 
  0.125 ≤ x ∧ x ≤ 4 :=
by sorry

end inequality_equivalence_l3895_389508


namespace B_largest_at_200_l3895_389526

/-- B_k is defined as the binomial coefficient (800 choose k) multiplied by 0.3^k -/
def B (k : ℕ) : ℝ := (Nat.choose 800 k : ℝ) * (0.3 ^ k)

/-- Theorem stating that B_k is largest when k = 200 -/
theorem B_largest_at_200 : ∀ k : ℕ, k ≤ 800 → B k ≤ B 200 :=
sorry

end B_largest_at_200_l3895_389526


namespace abs_difference_inequality_l3895_389537

theorem abs_difference_inequality (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 := by
  sorry

end abs_difference_inequality_l3895_389537


namespace total_liquid_proof_l3895_389524

/-- The amount of oil used in cups -/
def oil_amount : ℝ := 0.17

/-- The amount of water used in cups -/
def water_amount : ℝ := 1.17

/-- The total amount of liquid used in cups -/
def total_liquid : ℝ := oil_amount + water_amount

theorem total_liquid_proof : total_liquid = 1.34 := by
  sorry

end total_liquid_proof_l3895_389524


namespace polynomial_division_remainder_l3895_389562

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X : Polynomial ℚ)^5 - 3*(X^3) + X^2 + 2 = 
  (X^2 - 4*X + 6) * q + (-22*X - 28) := by sorry

end polynomial_division_remainder_l3895_389562


namespace total_daily_salary_l3895_389571

def grocery_store_salaries (manager_salary clerk_salary : ℕ) (num_managers num_clerks : ℕ) : ℕ :=
  manager_salary * num_managers + clerk_salary * num_clerks

theorem total_daily_salary :
  grocery_store_salaries 5 2 2 3 = 16 := by
  sorry

end total_daily_salary_l3895_389571


namespace polynomial_factorization_l3895_389510

theorem polynomial_factorization (a b c : ℝ) : 
  a*(b - c)^4 + b*(c - a)^4 + c*(a - b)^4 = (a - b)*(b - c)*(c - a)*(a*b^2 + a*c^2) := by
  sorry

end polynomial_factorization_l3895_389510


namespace perpendicular_vectors_cos2theta_l3895_389584

theorem perpendicular_vectors_cos2theta (θ : ℝ) : 
  let a : ℝ × ℝ := (1, Real.cos θ)
  let b : ℝ × ℝ := (-1, 2 * Real.cos θ)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.cos (2 * θ) = 0 :=
by
  sorry

end perpendicular_vectors_cos2theta_l3895_389584


namespace longest_segment_in_cylinder_l3895_389566

/-- The longest segment in a cylinder with radius 5 cm and height 6 cm is √136 cm. -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 6) :
  Real.sqrt ((2 * r)^2 + h^2) = Real.sqrt 136 := by
  sorry

end longest_segment_in_cylinder_l3895_389566


namespace square_sum_implies_product_zero_l3895_389515

theorem square_sum_implies_product_zero (n : ℝ) : 
  (n - 2022)^2 + (2023 - n)^2 = 1 → (n - 2022) * (2023 - n) = 0 := by
  sorry

end square_sum_implies_product_zero_l3895_389515


namespace max_books_borrowed_l3895_389585

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (two_books : ℕ) (three_books : ℕ) 
  (h1 : total_students = 50)
  (h2 : no_books = 10)
  (h3 : two_books = 18)
  (h4 : three_books = 8)
  (h5 : (total_students - no_books - two_books - three_books) * 7 ≤ 
        total_students * 4 - no_books * 0 - two_books * 2 - three_books * 3) :
  ∃ (max_books : ℕ), max_books = 49 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end max_books_borrowed_l3895_389585


namespace amount_C_is_correct_l3895_389590

/-- The amount C receives when $5000 is divided among A, B, C, and D in the ratio 1:3:5:7 -/
def amount_C : ℚ :=
  let total_amount : ℚ := 5000
  let ratio_A : ℚ := 1
  let ratio_B : ℚ := 3
  let ratio_C : ℚ := 5
  let ratio_D : ℚ := 7
  let total_ratio : ℚ := ratio_A + ratio_B + ratio_C + ratio_D
  (total_amount / total_ratio) * ratio_C

theorem amount_C_is_correct : amount_C = 1562.50 := by
  sorry

end amount_C_is_correct_l3895_389590


namespace modulus_of_z_l3895_389589

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l3895_389589


namespace fib_fraction_numerator_l3895_389581

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The simplified fraction of (F₂₀₀₃/F₂₀₀₂ - F₂₀₀₄/F₂₀₀₃) has numerator 1 -/
theorem fib_fraction_numerator :
  (fib 2003 : ℚ) / fib 2002 - (fib 2004 : ℚ) / fib 2003 = 1 / (fib 2002 * fib 2003) :=
by sorry

end fib_fraction_numerator_l3895_389581


namespace triangular_weight_is_60_l3895_389505

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- First balance condition: 1 round + 1 triangular = 3 round -/
axiom balance1 : round_weight + triangular_weight = 3 * round_weight

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
axiom balance2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight

theorem triangular_weight_is_60 : triangular_weight = 60 := by sorry

end triangular_weight_is_60_l3895_389505


namespace ice_cream_yogurt_cost_difference_l3895_389509

def ice_cream_cartons : ℕ := 20
def yogurt_cartons : ℕ := 2
def ice_cream_price : ℕ := 6
def yogurt_price : ℕ := 1

theorem ice_cream_yogurt_cost_difference :
  ice_cream_cartons * ice_cream_price - yogurt_cartons * yogurt_price = 118 := by
  sorry

end ice_cream_yogurt_cost_difference_l3895_389509


namespace coronavirus_survey_census_l3895_389553

/-- A survey type -/
inductive SurveyType
| HeightOfStudents
| LightBulbLifespan
| GlobalGenderRatio
| CoronavirusExposure

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  smallGroup : Bool
  specificGroup : Bool
  completeDataNecessary : Bool

/-- Define what makes a survey suitable for a census -/
def suitableForCensus (c : SurveyCharacteristics) : Prop :=
  c.smallGroup ∧ c.specificGroup ∧ c.completeDataNecessary

/-- Assign characteristics to each survey type -/
def surveyCharacteristics : SurveyType → SurveyCharacteristics
| SurveyType.HeightOfStudents => ⟨false, true, false⟩
| SurveyType.LightBulbLifespan => ⟨true, true, false⟩
| SurveyType.GlobalGenderRatio => ⟨false, false, false⟩
| SurveyType.CoronavirusExposure => ⟨true, true, true⟩

/-- Theorem: The coronavirus exposure survey is the only one suitable for a census -/
theorem coronavirus_survey_census :
  ∀ (s : SurveyType), suitableForCensus (surveyCharacteristics s) ↔ s = SurveyType.CoronavirusExposure :=
sorry

end coronavirus_survey_census_l3895_389553


namespace square_plus_reciprocal_square_l3895_389569

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 := by
  sorry

end square_plus_reciprocal_square_l3895_389569


namespace polynomial_simplification_l3895_389501

theorem polynomial_simplification (x : ℝ) :
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end polynomial_simplification_l3895_389501


namespace achieve_any_distribution_l3895_389550

-- Define the Student type
def Student : Type := ℕ

-- Define the friendship relation
def IsFriend (s1 s2 : Student) : Prop := sorry

-- Define the candy distribution
def CandyCount : Student → Fin 7 := sorry

-- Define the property of friendship for the set of students
def FriendshipProperty (students : Set Student) : Prop :=
  ∀ s1 s2 : Student, s1 ∈ students → s2 ∈ students → s1 ≠ s2 →
    ∃ s3 ∈ students, (IsFriend s3 s1 ∧ ¬IsFriend s3 s2) ∨ (IsFriend s3 s2 ∧ ¬IsFriend s3 s1)

-- Define a step in the candy distribution process
def DistributionStep (students : Set Student) (initial : Student → Fin 7) : Student → Fin 7 := sorry

-- Theorem: Any desired candy distribution can be achieved in finitely many steps
theorem achieve_any_distribution 
  (students : Set Student) 
  (h_finite : Finite students) 
  (h_friendship : FriendshipProperty students) 
  (initial : Student → Fin 7) 
  (target : Student → Fin 7) :
  ∃ n : ℕ, ∃ steps : Fin n → (Set Student), 
    (DistributionStep students)^[n] initial = target := by
  sorry

end achieve_any_distribution_l3895_389550


namespace stating_isosceles_triangle_with_special_bisectors_l3895_389539

/-- Represents an isosceles triangle with angle bisectors -/
structure IsoscelesTriangle where
  -- Base angle of the isosceles triangle
  β : Real
  -- Ratio of the lengths of two angle bisectors
  bisector_ratio : Real

/-- 
  Theorem stating the approximate angles of an isosceles triangle 
  where one angle bisector is twice the length of another
-/
theorem isosceles_triangle_with_special_bisectors 
  (triangle : IsoscelesTriangle) 
  (h1 : triangle.bisector_ratio = 2) 
  (h2 : 76.9 ≤ triangle.β ∧ triangle.β ≤ 77.1) : 
  25.9 ≤ 180 - 2 * triangle.β ∧ 180 - 2 * triangle.β ≤ 26.1 := by
  sorry

#check isosceles_triangle_with_special_bisectors

end stating_isosceles_triangle_with_special_bisectors_l3895_389539


namespace smart_mart_puzzles_l3895_389570

/-- The number of science kits sold by Smart Mart last week -/
def science_kits : ℕ := 45

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of puzzles sold by Smart Mart last week -/
def puzzles : ℕ := science_kits - difference

/-- Theorem stating that the number of puzzles sold is 36 -/
theorem smart_mart_puzzles : puzzles = 36 := by
  sorry

end smart_mart_puzzles_l3895_389570


namespace son_times_younger_l3895_389599

theorem son_times_younger (father_age son_age : ℕ) (h1 : father_age = 36) (h2 : son_age = 9) (h3 : father_age - son_age = 27) :
  father_age / son_age = 4 := by
sorry

end son_times_younger_l3895_389599


namespace average_string_length_l3895_389528

theorem average_string_length : 
  let string1 : ℝ := 2.5
  let string2 : ℝ := 3.5
  let string3 : ℝ := 4.5
  let total_length : ℝ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 3.5 := by
  sorry

end average_string_length_l3895_389528


namespace special_hexagon_perimeter_l3895_389502

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assumption that the hexagon is equilateral
  is_equilateral : True
  -- Assumption that three nonadjacent acute interior angles measure 45°
  has_45_degree_angles : True
  -- The enclosed area of the hexagon
  area : ℝ
  -- The area is 12√2
  area_eq : area = 12 * Real.sqrt 2

/-- The perimeter of a hexagon is 6 times its side length -/
def perimeter (h : SpecialHexagon) : ℝ := 6 * h.side

/-- Theorem: The perimeter of the special hexagon is 24√2 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : perimeter h = 24 * Real.sqrt 2 := by
  sorry

end special_hexagon_perimeter_l3895_389502


namespace pencils_per_row_l3895_389552

/-- Given a total of 720 pencils arranged in 30 rows, prove that there are 24 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (total_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 720) 
  (h2 : total_rows = 30) 
  (h3 : total_pencils = total_rows * pencils_per_row) : 
  pencils_per_row = 24 := by
  sorry

end pencils_per_row_l3895_389552


namespace f_strictly_increasing_l3895_389587

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log (1/3)

theorem f_strictly_increasing :
  ∀ x y, x < y ∧ y < 1 → f x < f y :=
sorry

end f_strictly_increasing_l3895_389587


namespace workshop_workers_l3895_389511

theorem workshop_workers (avg_salary : ℝ) (tech_count : ℕ) (tech_avg_salary : ℝ) (non_tech_avg_salary : ℝ)
  (h1 : avg_salary = 8000)
  (h2 : tech_count = 7)
  (h3 : tech_avg_salary = 10000)
  (h4 : non_tech_avg_salary = 6000) :
  ∃ (total_workers : ℕ), total_workers = 14 ∧
  (tech_count * tech_avg_salary + (total_workers - tech_count) * non_tech_avg_salary) / total_workers = avg_salary :=
by
  sorry

end workshop_workers_l3895_389511


namespace tiffany_cans_l3895_389518

theorem tiffany_cans (initial_bags : ℕ) (next_day_bags : ℕ) (total_bags : ℕ) :
  initial_bags = 10 →
  next_day_bags = 3 →
  total_bags = 20 →
  total_bags - (initial_bags + next_day_bags) = 7 :=
by sorry

end tiffany_cans_l3895_389518


namespace calculate_interest_rate_l3895_389557

/-- Given a car price, total amount to pay, and loan amount, calculate the interest rate -/
theorem calculate_interest_rate 
  (car_price : ℝ) 
  (total_amount : ℝ) 
  (loan_amount : ℝ) 
  (h1 : car_price = 35000)
  (h2 : total_amount = 38000)
  (h3 : loan_amount = 20000) :
  (total_amount - loan_amount) / loan_amount * 100 = 90 := by
  sorry

#check calculate_interest_rate

end calculate_interest_rate_l3895_389557


namespace intersection_with_complement_l3895_389544

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set P
def P : Set Nat := {1, 2, 3, 4}

-- Define set Q
def Q : Set Nat := {3, 4, 5}

-- Theorem statement
theorem intersection_with_complement : P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_with_complement_l3895_389544


namespace max_area_triangle_AOB_l3895_389513

/-- The maximum area of triangle AOB formed by the intersection points of
    two lines and a curve in polar coordinates. -/
theorem max_area_triangle_AOB :
  ∀ α : Real,
  0 < α →
  α < π / 2 →
  let C₁ := {θ : Real | θ = α}
  let C₂ := {θ : Real | θ = α + π / 2}
  let C₃ := {(ρ, θ) : Real × Real | ρ = 8 * Real.sin θ}
  let A := (8 * Real.sin α, α)
  let B := (8 * Real.cos α, α + π / 2)
  A.1 ≠ 0 ∨ A.2 ≠ 0 →
  B.1 ≠ 0 ∨ B.2 ≠ 0 →
  (∃ (S : Real → Real),
    (∀ α, S α = (1/2) * 8 * Real.sin α * 8 * Real.cos α) ∧
    (∀ α, S α ≤ 16)) :=
by sorry

end max_area_triangle_AOB_l3895_389513


namespace power_of_product_l3895_389577

theorem power_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by sorry

end power_of_product_l3895_389577


namespace calculate_expression_l3895_389578

theorem calculate_expression : 
  (8/27)^(2/3) + Real.log 3 / Real.log 12 + 2 * Real.log 2 / Real.log 12 = 13/9 := by
  sorry

end calculate_expression_l3895_389578


namespace books_rebecca_received_l3895_389554

theorem books_rebecca_received (books_initial : ℕ) (books_remaining : ℕ) 
  (h1 : books_initial = 220)
  (h2 : books_remaining = 60) : 
  ∃ (rebecca_books : ℕ), 
    rebecca_books = (books_initial - books_remaining) / 4 ∧ 
    rebecca_books = 40 := by
  sorry

end books_rebecca_received_l3895_389554


namespace cylinder_surface_area_l3895_389548

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π. -/
theorem cylinder_surface_area : 
  ∀ (h c : ℝ), h = 2 → c = 2 * Real.pi → 
  2 * Real.pi * (c / (2 * Real.pi)) * (c / (2 * Real.pi)) + c * h = 6 * Real.pi := by
  sorry

end cylinder_surface_area_l3895_389548


namespace fraction_calculation_l3895_389596

theorem fraction_calculation (x y : ℚ) (hx : x = 2/3) (hy : y = 5/2) :
  (1/3) * x^7 * y^6 = 125/261 := by
  sorry

end fraction_calculation_l3895_389596


namespace finance_class_competition_l3895_389580

theorem finance_class_competition (total : ℕ) (abacus : ℕ) (cash_counting : ℕ) (neither : ℕ) :
  total = 48 →
  abacus = 28 →
  cash_counting = 23 →
  neither = 5 →
  ∃ n : ℕ, n = 8 ∧ 
    total = abacus + cash_counting - n + neither :=
by sorry

end finance_class_competition_l3895_389580


namespace square_construction_with_compass_l3895_389527

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a compass operation
def compassIntersection (c1 c2 : Circle) : Set Point :=
  { p : Point | (p.x - c1.center.x)^2 + (p.y - c1.center.y)^2 = c1.radius^2 ∧
                (p.x - c2.center.x)^2 + (p.y - c2.center.y)^2 = c2.radius^2 }

-- Define a square
structure Square where
  vertices : Fin 4 → Point

-- Theorem statement
theorem square_construction_with_compass :
  ∃ (s : Square), 
    (∀ i j : Fin 4, i ≠ j → 
      (s.vertices i).x^2 + (s.vertices i).y^2 = 
      (s.vertices j).x^2 + (s.vertices j).y^2) ∧
    (∀ i : Fin 4, 
      (s.vertices i).x^2 + (s.vertices i).y^2 = 
      ((s.vertices (i + 1)).x - (s.vertices i).x)^2 + 
      ((s.vertices (i + 1)).y - (s.vertices i).y)^2) :=
by
  sorry


end square_construction_with_compass_l3895_389527


namespace length_AE_is_5_sqrt_5_div_3_l3895_389559

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (2, 0)

-- Define E as the intersection point of AB and CD
def E : ℝ × ℝ := sorry

-- Theorem statement
theorem length_AE_is_5_sqrt_5_div_3 :
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A E = (5 * Real.sqrt 5) / 3 := by sorry

end length_AE_is_5_sqrt_5_div_3_l3895_389559


namespace james_age_when_thomas_reaches_current_l3895_389588

theorem james_age_when_thomas_reaches_current (T : ℕ) : 
  let shay_age := T + 13
  let james_current_age := T + 18
  james_current_age = 42 →
  james_current_age + (james_current_age - T) = 60 :=
by
  sorry

end james_age_when_thomas_reaches_current_l3895_389588


namespace function_difference_bound_l3895_389561

theorem function_difference_bound 
  (f : Set.Icc 0 1 → ℝ)
  (h1 : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h2 : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1 : ℝ) / 2 := by
sorry

end function_difference_bound_l3895_389561


namespace problem_statement_l3895_389521

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := by
  sorry

end problem_statement_l3895_389521


namespace sequence_sum_proof_l3895_389572

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℕ := 2^(n+1) - 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^n * n

-- Theorem statement
theorem sequence_sum_proof (n : ℕ) :
  (∀ k, S k = 2^(k+1) - 2) →
  (∀ k, b k = 2^k * k) →
  (∃ T : ℕ → ℝ, T n = (n + 1) * 2^(n + 1) - 2) :=
by sorry

end sequence_sum_proof_l3895_389572


namespace binomial_expansion_alternating_sum_l3895_389523

theorem binomial_expansion_alternating_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ - a₀ + a₃ - a₂ + a₅ - a₄ = -1 := by
sorry

end binomial_expansion_alternating_sum_l3895_389523


namespace group_materials_calculation_l3895_389564

-- Define the given quantities
def teacher_materials : ℕ := 28
def total_products : ℕ := 93

-- Define the function to calculate group materials
def group_materials : ℕ := total_products - teacher_materials

-- Theorem statement
theorem group_materials_calculation :
  group_materials = 65 :=
sorry

end group_materials_calculation_l3895_389564


namespace rahims_average_book_price_l3895_389546

/-- Calculates the average price per book given two purchases -/
def average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) : ℚ :=
  (price1 + price2 : ℚ) / (books1 + books2 : ℚ)

/-- Proves that Rahim's average price per book is 20 rupees -/
theorem rahims_average_book_price :
  average_price_per_book 50 1000 40 800 = 20 := by
  sorry

end rahims_average_book_price_l3895_389546


namespace complement_intersection_theorem_l3895_389560

-- Define the universe set U
def U : Set Nat := {2, 3, 6, 8}

-- Define set A
def A : Set Nat := {2, 3}

-- Define set B
def B : Set Nat := {2, 6, 8}

-- Theorem statement
theorem complement_intersection_theorem :
  (Aᶜ ∩ B) = {6, 8} := by
  sorry

end complement_intersection_theorem_l3895_389560


namespace base12_addition_l3895_389520

/-- Converts a base 12 number to base 10 --/
def toBase10 (x : ℕ) (y : ℕ) (z : ℕ) : ℕ := x * 144 + y * 12 + z

/-- Converts a base 10 number to base 12 --/
def toBase12 (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 144
  let b := (n % 144) / 12
  let c := n % 12
  (a, b, c)

theorem base12_addition : 
  let x := toBase10 11 4 8  -- B48 in base 12
  let y := toBase10 5 7 10  -- 57A in base 12
  toBase12 (x + y) = (5, 11, 6) := by sorry

end base12_addition_l3895_389520


namespace journeymen_ratio_after_layoff_journeymen_fraction_is_two_thirds_l3895_389568

/-- The total number of employees in the anvil factory -/
def total_employees : ℕ := 20210

/-- The fraction of employees who are journeymen -/
def journeymen_fraction : ℚ := sorry

/-- The number of journeymen after laying off half of them -/
def remaining_journeymen : ℚ := journeymen_fraction * (total_employees : ℚ) / 2

/-- The total number of employees after laying off half of the journeymen -/
def remaining_employees : ℚ := (total_employees : ℚ) - remaining_journeymen

/-- The condition that after laying off half of the journeymen, they constitute 50% of the remaining workforce -/
theorem journeymen_ratio_after_layoff : remaining_journeymen / remaining_employees = 1 / 2 := sorry

/-- The main theorem: proving that the fraction of employees who are journeymen is 2/3 -/
theorem journeymen_fraction_is_two_thirds : journeymen_fraction = 2 / 3 := sorry

end journeymen_ratio_after_layoff_journeymen_fraction_is_two_thirds_l3895_389568


namespace christmas_tree_lights_l3895_389547

theorem christmas_tree_lights (total : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : total = 350) (h2 : red = 85) (h3 : yellow = 112) (h4 : green = 65) :
  total - (red + yellow + green) = 88 := by
  sorry

end christmas_tree_lights_l3895_389547


namespace ratio_e_to_f_l3895_389504

theorem ratio_e_to_f (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 3 / 4) :
  e / f = 1 / 2 := by
  sorry

end ratio_e_to_f_l3895_389504


namespace sum_of_coefficients_l3895_389592

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
  sorry

end sum_of_coefficients_l3895_389592


namespace floor_greater_than_fraction_l3895_389567

theorem floor_greater_than_fraction (a : ℝ) (n : ℤ) 
  (h1 : a ≥ 1) (h2 : 0 ≤ n) (h3 : n ≤ a) :
  Int.floor a > (n / (n + 1 : ℝ)) * a := by
  sorry

end floor_greater_than_fraction_l3895_389567


namespace conditional_probability_rain_given_wind_l3895_389558

/-- Given probabilities of events A and B, and their intersection, prove the conditional probability P(A|B) -/
theorem conditional_probability_rain_given_wind 
  (P_A : ℚ) (P_B : ℚ) (P_A_and_B : ℚ)
  (h1 : P_A = 4/15)
  (h2 : P_B = 2/15)
  (h3 : P_A_and_B = 1/10)
  : P_A_and_B / P_B = 3/4 := by
  sorry

end conditional_probability_rain_given_wind_l3895_389558


namespace square_diagonal_triangle_l3895_389598

theorem square_diagonal_triangle (s : ℝ) (h : s = 10) :
  let diagonal := s * Real.sqrt 2
  diagonal = 10 * Real.sqrt 2 ∧ s = 10 := by
  sorry

end square_diagonal_triangle_l3895_389598


namespace gcd_of_three_numbers_l3895_389529

theorem gcd_of_three_numbers : Nat.gcd 13456 (Nat.gcd 25345 15840) = 1 := by
  sorry

end gcd_of_three_numbers_l3895_389529


namespace arithmetic_progression_common_difference_l3895_389549

theorem arithmetic_progression_common_difference 
  (a₁ : ℝ) (a₂₁ : ℝ) (d : ℝ) :
  a₁ = 3 → a₂₁ = 103 → a₂₁ = a₁ + 20 * d → d = 5 := by
  sorry

end arithmetic_progression_common_difference_l3895_389549


namespace two_pairs_exist_l3895_389591

/-- A function that checks if a number consists of three identical digits -/
def has_three_identical_digits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 100 + d * 10 + d

/-- The main theorem stating the existence of two distinct pairs of numbers
    satisfying the given conditions -/
theorem two_pairs_exist : ∃ (a b c d : ℕ),
  has_three_identical_digits (a * b) ∧
  has_three_identical_digits (a + b) ∧
  has_three_identical_digits (c * d) ∧
  has_three_identical_digits (c + d) ∧
  (a ≠ c ∨ b ≠ d) :=
sorry

end two_pairs_exist_l3895_389591


namespace min_value_theorem_l3895_389531

/-- Given positive real numbers x and y satisfying x + y = 1,
    if the minimum value of 1/x + a/y is 9, then a = 4 -/
theorem min_value_theorem (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1)
    (hmin : ∀ (u v : ℝ), u > 0 → v > 0 → u + v = 1 → 1/u + a/v ≥ 9) : a = 4 := by
  sorry

end min_value_theorem_l3895_389531


namespace bakery_children_count_l3895_389506

theorem bakery_children_count (initial_count : ℕ) : 
  initial_count + 24 - 31 = 78 → initial_count = 85 := by
  sorry

end bakery_children_count_l3895_389506


namespace function_decreasing_iff_a_in_range_l3895_389519

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

theorem function_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 0 < a ∧ a ≤ 1/3 :=
sorry

end function_decreasing_iff_a_in_range_l3895_389519


namespace g_inverse_property_l3895_389538

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

theorem g_inverse_property (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) → c + d = 9/2 := by
  sorry

end g_inverse_property_l3895_389538


namespace alcohol_percentage_in_new_mixture_l3895_389545

def original_volume : ℝ := 24
def water_volume : ℝ := 16

def alcohol_A_fraction : ℝ := 0.3
def alcohol_B_fraction : ℝ := 0.4
def alcohol_C_fraction : ℝ := 0.3

def alcohol_A_purity : ℝ := 0.8
def alcohol_B_purity : ℝ := 0.9
def alcohol_C_purity : ℝ := 0.95

def new_mixture_volume : ℝ := original_volume + water_volume

def total_pure_alcohol : ℝ :=
  original_volume * (
    alcohol_A_fraction * alcohol_A_purity +
    alcohol_B_fraction * alcohol_B_purity +
    alcohol_C_fraction * alcohol_C_purity
  )

theorem alcohol_percentage_in_new_mixture :
  (total_pure_alcohol / new_mixture_volume) * 100 = 53.1 := by
  sorry

end alcohol_percentage_in_new_mixture_l3895_389545


namespace quadratic_maximum_l3895_389514

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 24

-- State the theorem
theorem quadratic_maximum :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_max : ℝ), f x_max = 30.75) ∧
  (∀ (x : ℝ), f x ≤ 30.75) ∧
  f (3/2) = 30.75 :=
by sorry

end quadratic_maximum_l3895_389514


namespace peters_remaining_money_is_304_50_l3895_389503

/-- Represents Peter's shopping trips and calculates his remaining money. -/
def petersRemainingMoney : ℝ :=
  let initialAmount : ℝ := 500
  let firstTripPurchases : List (ℝ × ℝ) := [
    (6, 2),    -- potatoes
    (9, 3),    -- tomatoes
    (5, 4),    -- cucumbers
    (3, 5),    -- bananas
    (2, 3.5),  -- apples
    (7, 4.25), -- oranges
    (4, 6),    -- grapes
    (8, 5.5)   -- strawberries
  ]
  let secondTripPurchases : List (ℝ × ℝ) := [
    (2, 1.5),  -- additional potatoes
    (5, 2.75)  -- additional tomatoes
  ]
  let totalCost := (firstTripPurchases ++ secondTripPurchases).foldl
    (fun acc (quantity, price) => acc + quantity * price) 0
  initialAmount - totalCost

/-- Theorem stating that Peter's remaining money is $304.50 -/
theorem peters_remaining_money_is_304_50 :
  petersRemainingMoney = 304.50 := by
  sorry

#eval petersRemainingMoney

end peters_remaining_money_is_304_50_l3895_389503


namespace circle_diameter_from_area_l3895_389512

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 4 * Real.pi → d = 4 := by
  sorry

end circle_diameter_from_area_l3895_389512


namespace carnation_percentage_l3895_389582

/-- Represents the number of each type of flower in the shop -/
structure FlowerShop where
  carnations : ℝ
  violets : ℝ
  tulips : ℝ
  roses : ℝ

/-- Conditions for the flower shop inventory -/
def validFlowerShop (shop : FlowerShop) : Prop :=
  shop.violets = shop.carnations / 3 ∧
  shop.tulips = shop.violets / 3 ∧
  shop.roses = shop.tulips

/-- Theorem stating the percentage of carnations in the flower shop -/
theorem carnation_percentage (shop : FlowerShop) 
  (h : validFlowerShop shop) (h_pos : shop.carnations > 0) : 
  shop.carnations / (shop.carnations + shop.violets + shop.tulips + shop.roses) = 9 / 14 := by
  sorry

end carnation_percentage_l3895_389582


namespace max_type_a_accessories_l3895_389595

/-- Represents the cost and quantity of drone accessories. -/
structure DroneAccessories where
  costA : ℕ  -- Cost of type A accessory
  costB : ℕ  -- Cost of type B accessory
  totalQuantity : ℕ  -- Total number of accessories
  maxCost : ℕ  -- Maximum total cost

/-- Calculates the maximum number of type A accessories that can be purchased. -/
def maxTypeA (d : DroneAccessories) : ℕ :=
  let m := (d.maxCost - d.costB * d.totalQuantity) / (d.costA - d.costB)
  min m d.totalQuantity

/-- Theorem stating the maximum number of type A accessories that can be purchased. -/
theorem max_type_a_accessories (d : DroneAccessories) : 
  d.costA = 230 ∧ d.costB = 100 ∧ d.totalQuantity = 30 ∧ d.maxCost = 4180 ∧
  d.costA + 3 * d.costB = 530 ∧ 3 * d.costA + 2 * d.costB = 890 →
  maxTypeA d = 9 := by
  sorry

#eval maxTypeA { costA := 230, costB := 100, totalQuantity := 30, maxCost := 4180 }

end max_type_a_accessories_l3895_389595


namespace percentage_of_m1_products_l3895_389532

theorem percentage_of_m1_products (m1_defective : Real) (m2_defective : Real)
  (m3_non_defective : Real) (m2_percentage : Real) (total_defective : Real) :
  m1_defective = 0.03 →
  m2_defective = 0.01 →
  m3_non_defective = 0.93 →
  m2_percentage = 0.3 →
  total_defective = 0.036 →
  ∃ (m1_percentage : Real),
    m1_percentage = 0.4 ∧
    m1_percentage + m2_percentage + (1 - m1_percentage - m2_percentage) = 1 ∧
    m1_percentage * m1_defective +
    m2_percentage * m2_defective +
    (1 - m1_percentage - m2_percentage) * (1 - m3_non_defective) = total_defective :=
by sorry

end percentage_of_m1_products_l3895_389532


namespace tens_digit_of_8_pow_2013_l3895_389575

theorem tens_digit_of_8_pow_2013 : ∃ n : ℕ, 8^2013 ≡ 88 + 100*n [ZMOD 100] :=
sorry

end tens_digit_of_8_pow_2013_l3895_389575


namespace number_division_problem_l3895_389555

theorem number_division_problem : ∃ x : ℝ, x / 5 = 40 + x / 6 ∧ x = 7200 / 31 := by
  sorry

end number_division_problem_l3895_389555


namespace line_equation_transformation_l3895_389525

/-- Given a line L with equation y = (2/3)x + 4, prove that a line M with twice the slope
    and half the y-intercept of L has the equation y = (4/3)x + 2 -/
theorem line_equation_transformation (x y : ℝ) :
  let L : ℝ → ℝ := λ x => (2/3) * x + 4
  let M : ℝ → ℝ := λ x => (4/3) * x + 2
  (∀ x, M x = 2 * ((2/3) * x) + (1/2) * 4) → (∀ x, M x = (4/3) * x + 2) :=
by sorry

end line_equation_transformation_l3895_389525


namespace coconut_yield_for_six_trees_l3895_389542

/-- The yield of x trees in a coconut grove --/
def coconut_grove_yield (x : ℕ) (Y : ℕ) : Prop :=
  let total_trees := 3 * x
  let total_yield := (x + 3) * 60 + x * Y + (x - 3) * 180
  (total_yield : ℚ) / total_trees = 100

theorem coconut_yield_for_six_trees :
  coconut_grove_yield 6 120 :=
sorry

end coconut_yield_for_six_trees_l3895_389542


namespace light_ray_exits_l3895_389574

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ
  length_pos : length > 0

/-- Represents an angle formed by two segments with a common vertex -/
structure Angle where
  seg1 : Segment
  seg2 : Segment

/-- Represents a light ray traveling inside an angle -/
structure LightRay where
  angle : Angle
  position : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a light ray has exited an angle -/
def has_exited (ray : LightRay) : Prop :=
  -- Implementation details omitted
  sorry

/-- Function to update the light ray's position and direction after a reflection -/
def reflect (ray : LightRay) : LightRay :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that a light ray will eventually exit the angle -/
theorem light_ray_exits (angle : Angle) :
  ∃ (n : ℕ), ∀ (ray : LightRay), ray.angle = angle →
    has_exited (n.iterate reflect ray) :=
  sorry

end light_ray_exits_l3895_389574


namespace dave_initial_apps_l3895_389586

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 8

/-- The number of apps Dave had left after deleting -/
def remaining_apps : ℕ := 8

/-- The initial number of apps Dave had -/
def initial_apps : ℕ := deleted_apps + remaining_apps

theorem dave_initial_apps : initial_apps = 16 := by
  sorry

end dave_initial_apps_l3895_389586


namespace quadratic_integer_roots_l3895_389551

/-- The polynomial x^2 + ax + 2a has two distinct integer roots if and only if a = -1 or a = 9 -/
theorem quadratic_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0) ↔ (a = -1 ∨ a = 9) :=
sorry

end quadratic_integer_roots_l3895_389551


namespace unknown_number_value_l3895_389565

theorem unknown_number_value (x n : ℤ) : 
  x = 88320 →
  x + n + 9211 - 1569 = 11901 →
  n = -84061 := by
sorry

end unknown_number_value_l3895_389565


namespace particle_probability_l3895_389534

def probability (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * probability (x-1) y + (1/3) * probability x (y-1) + (1/3) * probability (x-1) (y-1)

theorem particle_probability :
  probability 3 3 = 7/81 :=
sorry

end particle_probability_l3895_389534


namespace table_tennis_choices_l3895_389530

theorem table_tennis_choices (rackets balls nets : ℕ) 
  (h_rackets : rackets = 7)
  (h_balls : balls = 7)
  (h_nets : nets = 3) :
  rackets * balls * nets = 147 := by
  sorry

end table_tennis_choices_l3895_389530


namespace flu_infection_equation_l3895_389583

/-- 
Given:
- One person initially has the flu
- Each person infects x people on average in each round
- There are two rounds of infection
- After two rounds, 144 people have the flu

Prove that (1 + x)^2 = 144 correctly represents the total number of infected people.
-/
theorem flu_infection_equation (x : ℝ) : (1 + x)^2 = 144 :=
sorry

end flu_infection_equation_l3895_389583


namespace fraction_unchanged_l3895_389507

theorem fraction_unchanged (x y : ℝ) : (5 * x) / (x + y) = (5 * (10 * x)) / ((10 * x) + (10 * y)) := by
  sorry

end fraction_unchanged_l3895_389507


namespace cos_sin_10_deg_equality_l3895_389556

theorem cos_sin_10_deg_equality : 
  4 * Real.cos (10 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) = -Real.sqrt 3 := by
  sorry

end cos_sin_10_deg_equality_l3895_389556


namespace alphabet_dot_no_line_l3895_389536

theorem alphabet_dot_no_line (total : ℕ) (both : ℕ) (line_no_dot : ℕ) 
  (h1 : total = 50)
  (h2 : both = 16)
  (h3 : line_no_dot = 30)
  (h4 : total = both + line_no_dot + (total - (both + line_no_dot))) :
  total - (both + line_no_dot) = 4 := by
sorry

end alphabet_dot_no_line_l3895_389536


namespace f_properties_l3895_389500

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 - 4*x + 1

-- State the theorem
theorem f_properties :
  (∃ (max_value : ℝ), max_value = 5 ∧ ∀ x, f x ≤ max_value) ∧
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧
  (∀ x y, -2 < x ∧ x < y → f x > f y) := by
  sorry

end f_properties_l3895_389500


namespace scientific_notation_of_0_00000428_l3895_389516

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_bounds : 1 ≤ |mantissa| ∧ |mantissa| < 10

/-- Conversion function from a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00000428 :
  toScientificNotation 0.00000428 = ScientificNotation.mk 4.28 (-6) sorry := by
  sorry

end scientific_notation_of_0_00000428_l3895_389516


namespace degree_three_polynomial_l3895_389579

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 6*x + 4*x^2 - 5*x^3 + 7*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The coefficient of x^4 in h(x) -/
def coeff_x4 (c : ℝ) : ℝ := 7 + 11*c

/-- The coefficient of x^3 in h(x) -/
def coeff_x3 (c : ℝ) : ℝ := -5 - 7*c

theorem degree_three_polynomial :
  ∃ c : ℝ, coeff_x4 c = 0 ∧ coeff_x3 c ≠ 0 :=
sorry

end degree_three_polynomial_l3895_389579


namespace complex_fraction_value_l3895_389576

theorem complex_fraction_value : 
  let i : ℂ := Complex.I
  (3 + i) / (1 - i) = 1 + 2*i := by sorry

end complex_fraction_value_l3895_389576


namespace stewart_farm_sheep_count_l3895_389535

/-- The number of sheep on Stewart farm given the ratio of sheep to horses and food consumption. -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (food_per_horse total_food : ℕ),
  sheep * 7 = horses * 6 →  -- Ratio of sheep to horses is 6:7
  food_per_horse = 230 →    -- Each horse eats 230 ounces per day
  horses * food_per_horse = total_food →  -- Total food consumed by horses
  total_food = 12880 →      -- Total food needed is 12,880 ounces
  sheep = 48 := by
sorry

end stewart_farm_sheep_count_l3895_389535


namespace divisors_of_216n4_l3895_389563

/-- Number of positive integer divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_216n4 (n : ℕ) (h : n > 0) (h240 : num_divisors (240 * n^3) = 240) : 
  num_divisors (216 * n^4) = 156 := by
  sorry

end divisors_of_216n4_l3895_389563


namespace solve_system_l3895_389533

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 8) (eq2 : x + 3 * y = 7) : x = 38 / 11 := by
  sorry

end solve_system_l3895_389533


namespace original_number_proof_l3895_389597

theorem original_number_proof (x : ℝ) : 
  (((x + 5) - (x - 5)) / (x + 5)) * 100 = 76.92 → x = 8 := by
  sorry

end original_number_proof_l3895_389597


namespace regular_adult_ticket_price_correct_l3895_389541

/-- The regular price of an adult movie ticket given the following conditions:
  * There are 5 adults and 2 children.
  * Children's concessions cost $3 each.
  * Adults' concessions cost $5, $6, $7, $4, and $9 respectively.
  * Total cost of the trip is $139.
  * Each child's ticket costs $7.
  * Three adults have discounts of $3, $2, and $1 on their tickets.
-/
def regular_adult_ticket_price : ℚ :=
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let child_concession_cost : ℚ := 3
  let adult_concession_costs : List ℚ := [5, 6, 7, 4, 9]
  let total_trip_cost : ℚ := 139
  let child_ticket_cost : ℚ := 7
  let adult_ticket_discounts : List ℚ := [3, 2, 1]
  18.8

theorem regular_adult_ticket_price_correct :
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let child_concession_cost : ℚ := 3
  let adult_concession_costs : List ℚ := [5, 6, 7, 4, 9]
  let total_trip_cost : ℚ := 139
  let child_ticket_cost : ℚ := 7
  let adult_ticket_discounts : List ℚ := [3, 2, 1]
  regular_adult_ticket_price = 18.8 := by
  sorry

#eval regular_adult_ticket_price

end regular_adult_ticket_price_correct_l3895_389541


namespace ceiling_negative_fraction_cubed_l3895_389517

theorem ceiling_negative_fraction_cubed : ⌈(-7/4)^3⌉ = -5 := by sorry

end ceiling_negative_fraction_cubed_l3895_389517


namespace intersection_M_N_l3895_389594

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x * Real.log x > 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = Set.Ioo 1 2 := by sorry

end intersection_M_N_l3895_389594


namespace sequence_relation_l3895_389522

theorem sequence_relation (a b : ℕ+ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ+, a n + b n = 1)
  (h3 : ∀ n : ℕ+, b (n + 1) = b n / (1 - (a n)^2)) :
  ∀ n : ℕ+, b n = n / (n + 1) := by
sorry

end sequence_relation_l3895_389522


namespace alcohol_concentration_proof_l3895_389593

/-- Proves that adding 3 liters of pure alcohol to a 6-liter solution that is 25% alcohol 
    will result in a solution that is 50% alcohol. -/
theorem alcohol_concentration_proof 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_alcohol : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.25)
  (h3 : added_alcohol = 3)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry


end alcohol_concentration_proof_l3895_389593
